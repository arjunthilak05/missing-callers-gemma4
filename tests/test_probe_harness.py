"""Exp A harness v2 against a fake OpenAI-compatible loopback server; serve plan; queue runner.

The fake server implements /tokenize (chat and plain-prompt forms), /v1/chat/completions
with prompt/completion token ids and scripted native tool calls, /v1/models and /health.
No GPU, no vLLM, no model weights.
"""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

from absent_edges.data import canonical_json
from absent_edges.graphs import GRAPH_TOOL, GraphFacts
from absent_edges.models import LocalModel, ModelProfile
from absent_edges.probe import harness, serve
from absent_edges.probe import ledger as L
from absent_edges.probe import tools
from absent_edges.probe.tools import GraphQueryTool, SandboxError, ToolBox, discover_programs

PROJECT = Path(__file__).resolve().parents[1]
FILES = {"pkg/__init__.py": "", "pkg/core.py": ("def target(x):\n    return x\n\n\ndef caller_one():\n"
                                                "    return target(1)\n\n\nclass Box:\n    def method(self):\n"
                                                "        return target(2)\n"),
         "pkg/other.py": "from pkg.core import target as aliased\n\n\ndef via_alias():\n    return aliased(3)\n"}
TARGET = "pkg/core.py::target"
SURROGATE = "\ud800"
SUBMIT = {"calls": [("submit_answer", {"sites": [{"path": "pkg/core.py", "line": 6}], "complete": "yes"})]}
SYSTEM_TEXT = ("You are a careful software engineer. You have read-only access to a checkout of the Python repository "
               "fam through the tools provided. Use the tools to inspect the repository when needed, and finish by "
               "calling submit_answer. You have at most 8 responses; call submit_answer before they run out.")
TASK_TEXT = ("Task: find every place in the library source where the function `target` (defined in `pkg/core.py`, "
             "line 1) is invoked, either called directly or passed as a callable that gets invoked, from inside "
             "another function or method. Consider only files under `pkg/`. Ignore tests, docs, examples and "
             "module-level code.")
SUBMIT_TEXT = ("Submit each invocation site as a file path and line number with submit_answer, and state whether you "
               "believe your list is complete (yes, no or unknown).")
GUIDANCE = {"A": {"C2": "Be careful with this answer. Review your assumptions and weigh the evidence before relying on it "
                        "in your final list. Inspect the relevant source and search for additional references before "
                        "you submit.",
                  "C3": "This index answer is not an exhaustive list of callers. A missing relationship does not show "
                        "that no other caller exists. Inspect the relevant source and search for additional references "
                        "before you submit."}}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, PROJECT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def c1_payload(target=TARGET, source_sha256="0" * 64, **extra):
    """A C1-shaped incoming-invokes answer (render.py's real payloads have more fields; the harness needs these)."""
    path = target.split("::")[0]
    return {"query": {"symbol": target, "relation": "invokes", "direction": "incoming"},
            "matches": [{"id": "e" * 24, "source": "pkg/core.py::caller_one", "target": target, "relation": "invokes"}],
            "endpoints": [{"id": "pkg/core.py::caller_one", "path": "pkg/core.py", "start_line": 5, "end_line": 6,
                           "snippet": "def caller_one():\n    return target(1)", "snippet_end_line": 6},
                          {"id": target, "path": path, "start_line": 1, "end_line": 2,
                           "snippet": "def target(x):\n    return x", "snippet_end_line": 2}],
            "source_sha256": source_sha256, "facts_sha256": "1" * 64, "returned_count": 1, "truncated": False,
            "error": None, "state": "matches", "symbol_indexed": True, "payload_truncated": False,
            "query_result_count": 1, "coverage": {"completeness": "unknown"}, "coverage_totals": {}, **extra}


def condition_payload(condition, target=TARGET, source_sha256="0" * 64, verification=None, pair="A"):
    base = c1_payload(target, source_sha256)
    if condition in ("C2", "C3"):
        base["guidance"] = GUIDANCE["A"][condition] + ("" if pair == "A" else " (pair B)")
    elif condition == "C4a":
        base["verification"] = verification or {
            "kind": tools.LEXICAL_KIND, "query": target, "hits": [{"path": "pkg/core.py", "line": 6,
                                                                  "text": "return target(1)"}],
            "truncated": False, "error": None, "source_sha256": source_sha256}
    return base


class FakeServer:
    """Loopback vLLM stand-in. Script entries:
    {"calls": [(name, arguments)], "content": str}  native tool calls (arguments dict or raw string)
    {"content": "..."}                               text only (format error)
    {"status": 500} / {"drop": True}                 transport failures
    {"usage_mismatch": True} / {"overlong": True}    accounting failures
    {"finish_reason": "length"}                      overrides the finish reason
    """

    def __init__(self, script=(), default=None, pad=0):
        self.script, self.default, self.pad = list(script), default, pad
        self.model = ModelProfile().model
        self.requests, self.lock, self.calls = [], threading.Lock(), 0
        self.health_status = 200  # set to 500 for an endpoint that stops answering /health
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def reply(self, status, body):
                data = b"" if body is None else json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/health":
                    self.reply(outer.health_status, None)
                elif self.path == "/v1/models":
                    self.reply(200, {"object": "list", "data": [{"id": outer.model, "object": "model",
                                                                 "root": "/fake/model", "parent": None,
                                                                 "max_model_len": 32768}]})
                else:
                    self.reply(404, {"error": "not found"})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with outer.lock:
                    outer.requests.append((self.path, body))
                if self.path == "/tokenize":
                    if "prompt" in body:
                        data = body["prompt"].encode()
                        tokens = [2 + byte for byte in data[::4]]
                    else:
                        tokens = outer.prompt_ids(body)
                    self.reply(200, {"count": len(tokens), "max_model_len": 32768, "tokens": tokens})
                    return
                if self.path != "/v1/chat/completions":
                    self.reply(404, {"error": "not found"})
                    return
                with outer.lock:
                    entry = outer.script.pop(0) if outer.script else outer.default
                    outer.calls += 1
                    number = outer.calls
                if entry is None:
                    self.reply(500, {"error": "script exhausted"})
                    return
                if entry.get("status"):
                    self.reply(entry["status"], {"error": "scripted failure"})
                    return
                if entry.get("drop"):
                    self.close_connection = True
                    self.connection.shutdown(socket.SHUT_RDWR)
                    return
                message = {"role": "assistant", "content": entry.get("content")}
                if entry.get("calls"):
                    message["tool_calls"] = [
                        {"id": f"call-{number}-{index}", "type": "function",
                         "function": {"name": name, "arguments": arguments if isinstance(arguments, str)
                                      else json.dumps(arguments)}}
                        for index, (name, arguments) in enumerate(entry["calls"])]
                completion = list(range(10, 10 + entry.get("completion_tokens", 7)))
                if not entry.get("overlong"):
                    completion = completion[:body["max_tokens"]]
                prompt = outer.prompt_ids(body)
                usage = {"prompt_tokens": len(prompt) + (1 if entry.get("usage_mismatch") else 0),
                         "completion_tokens": len(completion)}
                usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
                finish = entry.get("finish_reason") or ("tool_calls" if entry.get("calls") else "stop")
                self.reply(200, {"id": f"chatcmpl-{number}", "object": "chat.completion", "model": outer.model,
                                 "choices": [{"index": 0, "message": message, "token_ids": completion,
                                              "finish_reason": finish}],
                                 "usage": usage, "prompt_token_ids": prompt})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def prompt_ids(self, body):
        basis = canonical_json({"messages": body["messages"], "tools": body.get("tools"),
                                "kwargs": body.get("chat_template_kwargs")})
        return [2 + int.from_bytes(basis[i:i + 4], "little") % 50000 for i in range(0, len(basis), 4)] + [1] * self.pad

    @property
    def endpoint(self):
        return f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def chat_requests(self):
        return [body for path, body in self.requests if path == "/v1/chat/completions"]

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class FakeBash:
    def __init__(self, output="ok\n", error=None):
        self.output, self.error, self.commands = output, error, []

    def run(self, command):
        self.commands.append(command)
        if self.error is not None:
            raise self.error
        return {"status": "executed", "visible": {"exit_code": 0, "output": self.output, "truncated": False}}


def make_repo(root):
    for relative, text in FILES.items():
        path = Path(root) / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return Path(root)


def facts_bytes(source_sha256="0" * 64):
    nodes = [{"id": TARGET, "kind": "function", "path": "pkg/core.py", "start_line": 1, "end_line": 2,
              "snippet": "def target(x):\n    return x", "snippet_end_line": 2},
             {"id": "pkg/core.py::caller_one", "kind": "function", "path": "pkg/core.py", "start_line": 5,
              "end_line": 6, "snippet": "def caller_one():\n    return target(1)", "snippet_end_line": 6}]
    edges = [{"id": "e" * 24, "source": "pkg/core.py::caller_one", "target": TARGET, "relation": "invokes"}]
    return canonical_json({"backend": "locagent", "source_sha256": source_sha256, "nodes": nodes, "edges": edges,
                           "error": None, "coverage": {"completeness": "unknown"}})


def make_episode(condition="C1", payload=None, **overrides):
    block, variant = {"C0": ("P", "none"), "T-offered": ("T", "none"), "T-recommended": ("T", "none")}.get(
        condition, ("P", "omit_node"))
    if payload is None and condition in harness.PAYLOAD_CONDITIONS:
        payload = condition_payload(condition)
    data = None if payload is None else canonical_json(payload)
    values = dict(run_id="run-h", cell_id=f"{block}|item-h|{variant}|{condition}|s11",
                  attempt_id=f"{block}|item-h|{variant}|{condition}|s11.a001", token=1, block=block, item_id="item-h",
                  family="fam", target=TARGET, target_line=1, package_dirs=("pkg",), variant=variant,
                  condition=condition, seed=11, wording_pair="A", payload=data,
                  payload_sha256=None if data is None else sha(data), dispatch_sha256="d" * 64, items_sha256="1" * 64)
    values.update(overrides)
    return harness.EpisodeInput(**values)


def events(directory):
    return [json.loads(line) for line in (Path(directory) / "events.jsonl").read_text().splitlines()]


CONTRACT_RESULT_FIELDS = ("cell_id", "attempt_id", "block", "item_id", "family", "variant", "condition", "seed",
                          "exit_status", "submitted", "calls", "tokens", "finish_reason_length_count", "tool_calls",
                          "exposure", "graph_query_calls", "endpoint", "gpu_uuid", "dispatch_sha256", "items_sha256",
                          "code")
CONTRACT_EXPOSURE_FIELDS = ("payload_present", "payload_sha256", "expected_payload_sha256", "payload_tokens",
                            "verification_block_delivered", "verification_sha256", "verification_hits",
                            "tools_offered", "recommendation_present")


class DesignTests(unittest.TestCase):
    def test_block_design_follows_d11(self):
        """D11.4: C0 joins block P at every primary seed; S-C0 no longer exists; secondary order (D10.8)."""
        self.assertEqual(harness.BLOCK_ORDER, ("P", "S-VC", "S-NAT", "S-EDGE", "T"))
        self.assertEqual(harness.BLOCK_DESIGN["P"]["cells"], (("none", "C0"), ("omit_node", "C1"), ("omit_node", "C2"),
                                                             ("omit_node", "C3"), ("omit_node", "C4a")))
        self.assertEqual(harness.BLOCK_DESIGN["P"]["seeds"], (11, 23, 47))
        expected = {"S-VC": ((("vcomplete", "C1"), ("vcomplete", "C3")), None),
                    "S-NAT": ((("natural", "C1"), ("natural", "C3"), ("natural", "C4a")), "in_s_nat"),
                    "S-EDGE": ((("omit_edge", "C1"), ("omit_edge", "C3")), "in_s_edge"),
                    "T": ((("none", "T-offered"), ("none", "T-recommended")), "in_t")}
        for block, (cells, membership) in expected.items():
            self.assertEqual((harness.BLOCK_DESIGN[block]["cells"], harness.BLOCK_DESIGN[block]["seeds"],
                              harness.BLOCK_DESIGN[block]["membership"]), (cells, (11,), membership))
        self.assertNotIn("S-C0", harness.BLOCK_DESIGN)
        self.assertEqual(harness.item_order_key("x"), sha(b"expA-order-v2|x"))
        json.dumps(harness.design_table())

    def test_operational_appendix(self):
        appendix = harness.operational_appendix()
        self.assertEqual(appendix["episode_config"], {"max_model_calls": 8, "context_budget": 16384,
                                                      "max_response_tokens": 4096, "min_response_tokens": 512,
                                                      "temperature": 0.2, "max_consecutive_format_errors": 3})
        self.assertEqual(appendix["tools"]["schemas"]["graph_query"], GRAPH_TOOL)
        self.assertIn("Several tool calls in one response are run in their order", " ".join(appendix["response_handling"]))
        self.assertIn("Every model response counts", appendix["budget_rules"][0])
        self.assertEqual(appendix["prompts"]["recommendation"], "A graph_query tool can list indexed callers of a function.")
        json.dumps(appendix)


@unittest.skipUnless(os.name == "posix", "the ToolBox search scans with POSIX no-follow reads")
class EpisodeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = make_repo(Path(self.temporary.name) / "repo")
        self.servers = []
        self.count = 0

    def tearDown(self):
        for server in self.servers:
            with contextlib.suppress(Exception):
                server.close()
        self.temporary.cleanup()

    def server(self, script=(), **kwargs):
        server = FakeServer(script, **kwargs)
        self.servers.append(server)
        return server

    def toolbox(self, bash=None, graph=False):
        return ToolBox(self.root, ["pkg"], bash or FakeBash(), graph=GraphQueryTool(facts_bytes()) if graph else None)

    def run_episode(self, server, episode=None, bash=None, toolbox=None, **kwargs):
        self.count += 1
        directory = Path(self.temporary.name) / f"attempt-{self.count}"
        directory.mkdir()
        client = LocalModel(ModelProfile(endpoint=server.endpoint), timeout=20)
        episode = episode or make_episode()
        box = toolbox or self.toolbox(bash, graph=episode.graph_query_offered)
        result, digest = harness.run_episode(episode, directory, client=client, toolbox=box, **kwargs)
        return result, digest, directory

    def test_submit_path_records_contract_fields_exposure_and_immutable_result(self):
        server = self.server([{"calls": [("source_reference_search", {"query": TARGET})]},
                              {"calls": [("bash", {"command": "grep -rn target pkg"})]}, SUBMIT])
        result, digest, directory = self.run_episode(server, gpu_uuid="GPU-00000000-0000-0000-0000-000000000007")
        self.assertEqual(result["exit_status"], "Submitted")
        for field in CONTRACT_RESULT_FIELDS:
            self.assertIn(field, result)
        for field in CONTRACT_EXPOSURE_FIELDS:
            self.assertIn(field, result["exposure"])
        self.assertEqual(result["schema"], "expA-episode-result-v2")
        self.assertEqual(result["submitted"], {"sites": [{"path": "pkg/core.py", "line": 6}], "complete": "yes"})
        self.assertEqual((result["calls"], result["tokens"]["completion"], result["finish_reason_length_count"]),
                         (3, 21, 0))
        self.assertEqual(result["tool_calls"], {"source_reference_search": 1, "bash": 1, "submit_answer": 1})
        self.assertEqual((result["endpoint"], result["gpu_uuid"]), (server.endpoint,
                                                                    "GPU-00000000-0000-0000-0000-000000000007"))
        self.assertEqual((result["dispatch_sha256"], result["items_sha256"]), ("d" * 64, "1" * 64))
        self.assertIsNone(result["graph_query_calls"])
        self.assertEqual(result["cell_id"], "P|item-h|omit_node|C1|s11")
        self.assertEqual(digest, L.sha256_path(directory / "result.json"))
        for name in ("result.json", "events.jsonl"):
            self.assertEqual(stat.S_IMODE((directory / name).stat().st_mode), 0o444)
        self.assertEqual(result["events"]["sha256"], L.sha256_path(directory / "events.jsonl"))
        kinds = [row["kind"] for row in events(directory)]
        self.assertEqual(kinds[:3], ["episode_setup", "context_accounting", "exposure_receipt"])
        self.assertLess(kinds.index("exposure_receipt"), kinds.index("model_exchange"))
        receipt = result["exposure"]
        payload = canonical_json(condition_payload("C1"))
        self.assertEqual((receipt["payload_present"], receipt["payload_sha256"], receipt["expected_payload_sha256"]),
                         (True, sha(payload), sha(payload)))
        self.assertEqual(receipt["payload_tokens"], (len(payload) + 3) // 4)
        self.assertEqual((receipt["verification_block_delivered"], receipt["verification_sha256"],
                          receipt["verification_hits"], receipt["recommendation_present"]), (False, None, None, False))
        self.assertTrue(receipt["recorded_before_first_model_call"])
        self.assertEqual(receipt["tools_offered"], ["source_reference_search", "bash", "submit_answer"])
        chats = server.chat_requests()
        self.assertEqual(len(chats), 3)
        for body in chats:
            self.assertEqual((body["temperature"], body["seed"], body["return_token_ids"], body["tool_choice"]),
                             (0.2, 11, True, "auto"))
            self.assertIs(body["chat_template_kwargs"]["enable_thinking"], False)
            self.assertEqual(body["max_tokens"], 4096)
            self.assertEqual([tool["function"]["name"] for tool in body["tools"]],
                             ["source_reference_search", "bash", "submit_answer"])
        search_result = chats[1]["messages"][-1]
        self.assertEqual((search_result["role"], search_result["tool_call_id"]), ("tool", "call-1-0"))
        self.assertEqual(json.loads(search_result["content"])["kind"], "lexical_candidates_not_proven_edges")
        self.assertEqual(json.loads(chats[2]["messages"][-1]["content"])["output"], "ok\n")
        self.assertEqual(chats[0]["messages"][0], {"role": "system", "content": SYSTEM_TEXT})
        self.assertEqual(chats[0]["messages"][1]["content"],
                         TASK_TEXT + "\n\nA code-graph index returned this answer for incoming `invokes` relationships "
                         "of `pkg/core.py::target`:\n" + payload.decode() + "\n\n" + SUBMIT_TEXT)

    def test_prompts_follow_d6_and_d10(self):
        config = harness.EpisodeConfig()
        c0 = harness.render_messages(make_episode("C0"), config)
        self.assertEqual(c0["system"], SYSTEM_TEXT)
        self.assertEqual(c0["user"], TASK_TEXT + "\n\n" + SUBMIT_TEXT)
        self.assertEqual(c0["tools_offered"], ["source_reference_search", "bash", "submit_answer"])
        for condition in ("C1", "C2", "C3", "C4a"):
            with self.subTest(condition=condition):
                episode = make_episode(condition)
                rendered = harness.render_messages(episode, config)
                self.assertEqual(rendered["system"], c0["system"])
                self.assertEqual(rendered["user"].replace(rendered["answer_block"], "", 1), c0["user"])
                self.assertEqual(rendered["answer_block"], "\n\nA code-graph index returned this answer for incoming "
                                 "`invokes` relationships of `pkg/core.py::target`:\n" + episode.payload.decode())
                self.assertEqual(rendered["tools"], c0["tools"])
        offered = harness.render_messages(make_episode("T-offered"), config)
        self.assertEqual(offered["user"], c0["user"])
        self.assertEqual(offered["tools_offered"], ["source_reference_search", "bash", "submit_answer", "graph_query"])
        self.assertEqual(offered["tools"][-1], GRAPH_TOOL)
        recommended = harness.render_messages(make_episode("T-recommended"), config)
        self.assertEqual(recommended["user"], TASK_TEXT + " A graph_query tool can list indexed callers of a function."
                         + "\n\n" + SUBMIT_TEXT)
        self.assertTrue(recommended["recommendation_present"])
        self.assertEqual(harness.format_package_dirs(["src/attr/", "src/attrs"]), "`src/attr/`, `src/attrs/`")
        self.assertEqual(harness.format_package_dirs(["six.py"]), "`six.py`")

    def test_exposure_receipts_per_condition(self):
        """F8: verification block delivery, sha256 and hits; tools offered; recommendation."""
        c4a = make_episode("C4a")
        result, _, _ = self.run_episode(self.server([SUBMIT]), episode=c4a)
        receipt = result["exposure"]
        block = json.loads(c4a.payload)["verification"]
        self.assertEqual((receipt["verification_block_delivered"], receipt["verification_sha256"],
                          receipt["verification_hits"]), (True, sha(canonical_json(block)), 1))
        result, _, _ = self.run_episode(self.server([SUBMIT]), episode=make_episode("C3"))
        self.assertEqual((result["exposure"]["guidance_present"], result["exposure"]["verification_block_delivered"]),
                         (True, False))
        result, _, _ = self.run_episode(self.server([SUBMIT]), episode=make_episode("C0"))
        self.assertEqual((result["exposure"]["payload_present"], result["exposure"]["payload_sha256"],
                          result["exposure"]["payload_tokens"], result["exposure"]["answer_block_bytes"]),
                         (False, None, 0, 0))
        result, _, _ = self.run_episode(self.server([SUBMIT]), episode=make_episode("T-recommended"))
        self.assertEqual((result["exposure"]["recommendation_present"], result["exposure"]["graph_query_offered"],
                          result["exposure"]["tools_offered"][-1], result["graph_query_calls"]),
                         (True, True, "graph_query", []))

    def test_t_arm_graph_query_is_answered_from_natural_facts_and_logged(self):
        server = self.server([{"calls": [("graph_query", {"symbol": TARGET, "direction": "incoming"}),
                                         ("graph_query", {"symbol": "pkg/core.py::nothing"})]},
                              {"calls": [("graph_query", {"symbol": 5})]}, SUBMIT])
        result, _, directory = self.run_episode(server, episode=make_episode("T-offered"))
        self.assertEqual(result["exit_status"], "Submitted")
        calls = result["graph_query_calls"]
        self.assertEqual([(c["call_index"], c["status"], c["state"]) for c in calls],
                         [(0, "executed", "matches"), (0, "executed", "symbol_not_indexed"),
                          (1, "invalid_arguments", None)])
        expected = GraphFacts(facts_bytes()).query(TARGET, "invokes", "incoming", 30).render("C1", max_output_bytes=12000)
        chats = server.chat_requests()
        delivered = chats[1]["messages"][-2]
        self.assertEqual(json.loads(delivered["content"]), expected)
        self.assertEqual(calls[0]["answer_sha256"], sha(delivered["content"].encode()))
        self.assertEqual(calls[0]["arguments"], {"symbol": TARGET, "relation": "invokes", "direction": "incoming",
                                                 "limit": 30})
        self.assertEqual(result["tool_calls"]["graph_query"], 3)
        self.assertEqual([tool["function"]["name"] for tool in chats[0]["tools"]],
                         ["source_reference_search", "bash", "submit_answer", "graph_query"])
        logged = [row["value"] for row in events(directory) if row["kind"] == "model_visible_tool_result"]
        self.assertEqual([row["sha256"] for row in logged if row["name"] == "graph_query"],
                         [call["answer_sha256"] for call in calls])

    def test_graph_query_is_not_a_tool_outside_the_t_arms(self):
        server = self.server([{"calls": [("graph_query", {"symbol": TARGET})]}, SUBMIT])
        result, _, _ = self.run_episode(server)
        self.assertEqual((result["exit_status"], result["accounting"]["format_errors"],
                          result["accounting"]["graph_query_requests_in_format_errors"]), ("Submitted", 1, 1))
        self.assertIn("a tool call named a tool that does not exist", server.chat_requests()[1]["messages"][-1]["content"])
        with self.assertRaises(ValueError):
            self.run_episode(self.server([SUBMIT]), episode=make_episode("T-offered"), toolbox=self.toolbox())

    def test_format_error_is_visible_and_counted(self):
        server = self.server([{"content": "The callers are pkg/core.py:6."}, SUBMIT])
        result, _, directory = self.run_episode(server)
        self.assertEqual((result["exit_status"], result["accounting"]["format_errors"], result["calls"]),
                         ("Submitted", 1, 2))
        history = server.chat_requests()[1]["messages"]
        self.assertEqual(history[-2], {"role": "assistant", "content": "The callers are pkg/core.py:6."})
        self.assertEqual(history[-1]["role"], "user")
        self.assertTrue(history[-1]["content"].startswith("Format error: your response did not call a tool"))
        self.assertIn("format_error", [row["kind"] for row in events(directory)])

    def test_three_consecutive_format_errors_exhaust(self):
        server = self.server([{"content": "a"}, {"calls": [("python", {"code": "1"})]},
                              {"calls": [("bash", "{bad json")]}, SUBMIT])
        result, _, _ = self.run_episode(server)
        self.assertEqual((result["exit_status"], result["accounting"]["format_errors"]), ("FormatExhausted", 3))
        self.assertIsNone(result["submitted"])
        self.assertEqual(len(server.chat_requests()), 3)
        self.assertIn("does not exist", server.chat_requests()[2]["messages"][-1]["content"])

    def test_eight_responses_end_as_no_answer(self):
        bash = {"calls": [("bash", {"command": "ls"})]}
        server = self.server([bash] * 9)
        result, _, _ = self.run_episode(server)
        self.assertEqual((result["exit_status"], result["calls"]), ("NoAnswer", 8))
        self.assertEqual(len(server.chat_requests()), 8)
        mixed = self.server([{"content": "x"}, bash, {"content": "x"}, bash, {"content": "x"}, bash,
                             {"content": "x"}, {"content": "x"}, SUBMIT])
        result, _, _ = self.run_episode(mixed)
        self.assertEqual((result["exit_status"], result["accounting"]["format_errors"], result["calls"]),
                         ("NoAnswer", 5, 8))

    def test_context_policy_d10_12(self):
        """max_tokens = min(4096, remaining); ContextLimit when fewer than 512 tokens remain before a call."""
        first, _, _ = self.run_episode(self.server([SUBMIT]))
        base = first["exposure"]["first_prompt_tokens"]
        self.assertEqual(first["accounting"]["max_tokens_requested"], [4096])
        for remaining, expected in ((1000, ("Submitted", [1000])), (512, ("Submitted", [512])),
                                    (511, ("ContextLimit", [])), (0, ("ContextLimit", [])), (-5, ("ContextLimit", []))):
            with self.subTest(remaining=remaining):
                server = self.server([SUBMIT], pad=16384 - base - remaining)
                result, _, directory = self.run_episode(server)
                self.assertEqual((result["exit_status"], result["accounting"]["max_tokens_requested"]), expected)
                self.assertEqual([body["max_tokens"] for body in server.chat_requests()], expected[1])
                if expected[0] == "ContextLimit":
                    self.assertEqual(result["calls"], 0)
                    self.assertTrue(result["exposure"]["recorded_before_first_model_call"])
                    limit = [row["value"] for row in events(directory) if row["kind"] == "context_limit"][0]
                    self.assertEqual(limit["remaining"], remaining)

    def test_context_limit_after_a_large_tool_output(self):
        server = self.server([{"calls": [("bash", {"command": "cat big"})]}, SUBMIT], pad=8000)
        result, _, _ = self.run_episode(server, bash=FakeBash(output="y" * 30000))
        self.assertEqual((result["exit_status"], result["calls"]), ("ContextLimit", 1))
        self.assertEqual(server.chat_requests()[0]["max_tokens"], 4096)

    def test_finish_reason_length_is_counted(self):
        server = self.server([{"content": "half a tool ca", "finish_reason": "length"},
                              {"calls": [("bash", {"command": "ls"})], "finish_reason": "length"}, SUBMIT])
        result, _, _ = self.run_episode(server)
        self.assertEqual((result["exit_status"], result["finish_reason_length_count"],
                          result["accounting"]["finish_reasons"]), ("Submitted", 2, ["length", "length", "tool_calls"]))

    def test_infrastructure_failures_are_infra_interrupted(self):
        cases = {"http_500": [{"calls": [("bash", {"command": "ls"})]}, {"status": 500}],
                 "dropped": [{"drop": True}], "usage": [{"calls": [("bash", {"command": "ls"})], "usage_mismatch": True}],
                 "overlong": [{"calls": [("bash", {"command": "ls"})], "completion_tokens": 5000, "overlong": True}]}
        expected_kind = {"http_500": "server", "dropped": "server", "usage": "accounting", "overlong": "accounting"}
        for name, script in cases.items():
            with self.subTest(name=name):
                result, _, directory = self.run_episode(self.server(script))
                self.assertEqual(result["exit_status"], "InfraInterrupted")
                self.assertEqual(result["infra"]["kind"], expected_kind[name])
                self.assertIsNone(result["submitted"])
                self.assertTrue((directory / "result.json").exists())
        result, _, _ = self.run_episode(self.server(cases["http_500"]))
        self.assertEqual((result["infra"]["http_status"], result["calls"],
                          result["accounting"]["attempted_model_calls"], result["accounting"]["generation_usage_unknown"]),
                         (500, 1, 2, True))

    def test_server_down_and_stop_requests(self):
        down = self.server([SUBMIT])
        down.close()
        result, _, _ = self.run_episode(down)
        self.assertEqual((result["exit_status"], result["infra"]["kind"]), ("InfraInterrupted", "server"))
        self.assertFalse(result["exposure"]["recorded_before_first_model_call"])
        stop = threading.Event()
        stop.set()
        server = self.server([SUBMIT])
        result, _, _ = self.run_episode(server, stop_event=stop)
        self.assertEqual((result["exit_status"], result["infra"]["kind"]), ("InfraInterrupted", "stop_requested"))
        self.assertEqual(server.requests, [])

    def test_rejected_submission_envelope_then_valid(self):
        server = self.server([{"calls": [("submit_answer", {"sites": "pkg/core.py:6", "complete": "yes"})]},
                              {"calls": [("submit_answer", {"call_sites": [], "complete": "yes"})]}, SUBMIT])
        result, _, _ = self.run_episode(server)
        self.assertEqual((result["exit_status"], result["accounting"]["rejected_submissions"]), ("Submitted", 2))
        self.assertIn("submit_answer rejected: sites must be a list", server.chat_requests()[1]["messages"][-1]["content"])
        self.assertIn("unknown fields call_sites", server.chat_requests()[2]["messages"][-1]["content"])

    def test_invalid_entries_are_submitted_for_the_scorer(self):
        sites = [{"path": "/repo/pkg/core.py", "line": "6"}, {"path": "pkg/core.py", "line": 0}, "pkg/other.py:5"]
        result, _, directory = self.run_episode(self.server([{"calls": [("submit_answer", {"sites": sites,
                                                                                            "complete": "no"})]}]))
        self.assertEqual(result["exit_status"], "Submitted")
        self.assertEqual(result["submitted"], {"sites": [{"path": "pkg/core.py", "line": 6},
                                                         {"path": "pkg/core.py", "line": 0}, "pkg/other.py:5"],
                                               "complete": "no"})
        self.assertEqual(result["submission_validation"]["invalid"], [{"index": 1, "reason": "invalid_line"},
                                                                       {"index": 2, "reason": "not_an_object"}])

    def test_several_tool_calls_run_in_order_and_stop_at_an_accepted_submission(self):
        bash = FakeBash()
        server = self.server([{"calls": [("bash", {"command": "ls"}), ("source_reference_search", {"query": "target"}),
                                         ("bash", {"command": "cat pkg/core.py"})]},
                              {"calls": [("bash", {"command": "pwd"}), SUBMIT["calls"][0],
                                         ("bash", {"command": "cat x"}), SUBMIT["calls"][0]]}])
        result, _, directory = self.run_episode(server, bash=bash)
        self.assertEqual(result["exit_status"], "Submitted")
        self.assertEqual(bash.commands, ["ls", "cat pkg/core.py", "pwd"])
        self.assertEqual(result["accounting"]["not_executed_after_submission"], 2)
        tool_messages = [m for m in server.chat_requests()[1]["messages"] if m["role"] == "tool"]
        self.assertEqual([m["tool_call_id"] for m in tool_messages], ["call-1-0", "call-1-1", "call-1-2"])
        self.assertEqual([row["value"]["name"] for row in events(directory) if row["kind"] == "tool_action_not_executed"],
                         ["bash", "submit_answer"])

    def test_invalid_unicode_is_model_visible_never_infrastructure(self):
        """F5: lone surrogates in a bash command, a search query, a graph symbol or a submission."""
        server = self.server([
            {"calls": [("bash", {"command": "grep " + SURROGATE + " pkg"})]},
            {"calls": [("source_reference_search", {"query": "tar" + SURROGATE})]},
            {"calls": [("submit_answer", {"sites": [{"path": "pkg/core" + SURROGATE + ".py", "line": 6}],
                                          "complete": "yes"})]},
            {"content": "text with a lone surrogate " + SURROGATE, "calls": [("bash", {"command": "ls"})]},
            SUBMIT])
        bash = FakeBash()
        result, _, directory = self.run_episode(server, bash=bash)
        self.assertEqual(result["exit_status"], "Submitted")
        self.assertIsNone(result["infra"])
        self.assertEqual(bash.commands, [])
        self.assertEqual((result["accounting"]["invalid_unicode_arguments"], result["accounting"]["format_errors"],
                          result["accounting"]["rejected_tool_calls"], result["accounting"]["rejected_submissions"]),
                         (3, 1, 2, 1))
        chats = server.chat_requests()
        for index in (1, 2, 3):
            self.assertIn("invalid Unicode", chats[index]["messages"][-1]["content"])
        self.assertIn("your response contained invalid Unicode", chats[4]["messages"][-1]["content"])
        self.assertEqual(chats[4]["messages"][-2], {"role": "assistant", "content": "text with a lone surrogate �"})
        self.assertIn("\\ud800", (directory / "events.jsonl").read_text())  # the raw argument is kept, ASCII-escaped
        t_server = self.server([{"calls": [("graph_query", {"symbol": TARGET + SURROGATE})]}, SUBMIT])
        result, _, _ = self.run_episode(t_server, episode=make_episode("T-offered"))
        self.assertEqual((result["exit_status"], result["accounting"]["invalid_unicode_arguments"]), ("Submitted", 1))

    def test_sandbox_and_controller_failures(self):
        server = self.server([{"calls": [("bash", {"command": "ls"})]}])
        result, _, _ = self.run_episode(server, bash=FakeBash(error=SandboxError("jail mount failed")))
        self.assertEqual((result["exit_status"], result["infra"]["kind"]), ("InfraInterrupted", "sandbox"))

        class Broken:
            def execute(self, name, arguments):
                raise RuntimeError("controller defect")
        result, _, _ = self.run_episode(self.server([{"calls": [("bash", {"command": "ls"})]}]), toolbox=Broken())
        self.assertEqual((result["exit_status"], result["infra"]["kind"]), ("InfraInterrupted", "controller_exception"))
        self.assertIn("controller defect", result["infra"]["traceback"])

    def test_input_contracts(self):
        data = canonical_json(condition_payload("C1"))
        with self.assertRaises(harness.PayloadIntegrityError):
            make_episode(payload_sha256="0" * 64)
        pretty = json.dumps(condition_payload("C1"), indent=2).encode()
        with self.assertRaises(harness.PayloadIntegrityError):
            harness.EpisodeInput(**{**make_episode().__dict__, "payload": pretty, "payload_sha256": sha(pretty)})
        c2 = canonical_json(condition_payload("C2"))
        outgoing = canonical_json(dict(condition_payload("C1"), query={"symbol": TARGET, "relation": "invokes",
                                                                       "direction": "outgoing"}))
        other = canonical_json(condition_payload("C1", target="pkg/core.py::caller_one"))
        bad = [dict(condition="C0", variant="none", payload=data, payload_sha256=sha(data)),
               dict(condition="C2", payload=data, payload_sha256=sha(data)),
               dict(condition="C1", payload=c2, payload_sha256=sha(c2)),
               dict(condition="C4a", payload=data, payload_sha256=sha(data)),
               dict(payload=outgoing, payload_sha256=sha(outgoing)), dict(payload=other, payload_sha256=sha(other)),
               dict(target="pkg/core.py"), dict(target_line=0), dict(variant="vcomplete"), dict(variant="none"),
               dict(block="S-C0"), dict(block="S-VC", condition="C2"), dict(block="T"), dict(wording_pair="C"),
               dict(condition="C0", variant="omit_node", payload=None, payload_sha256=None),
               dict(dispatch_sha256="x"), dict(seed="11"), dict(package_dirs=["pkg"])]
        for overrides in bad:
            with self.subTest(overrides=sorted(overrides)), self.assertRaises(ValueError):
                harness.EpisodeInput(**{**make_episode().__dict__, **overrides})
        for condition in harness.CONDITIONS:
            make_episode(condition)
        with self.assertRaises(ValueError):
            harness.run_episode(make_episode(), self.temporary.name,
                                client=LocalModel(ModelProfile(enable_thinking=True)), toolbox=None)

    def test_attempt_directories_are_immutable(self):
        server = self.server([SUBMIT, SUBMIT])
        _, _, directory = self.run_episode(server)
        with self.assertRaises(FileExistsError):
            harness.run_episode(make_episode(), directory, client=LocalModel(ModelProfile(endpoint=server.endpoint)),
                                toolbox=self.toolbox())


GPU6 = "GPU-15870ba7-ca88-14c1-d031-0ee0aa0a0a0a"


def gpu_xml(uuid, minor, used, pids=()):
    processes = "".join(f"<process_info><pid>{pid}</pid><type>C</type><used_memory>100 MiB</used_memory></process_info>"
                        for pid in pids)
    return (f"<?xml version=\"1.0\" ?><nvidia_smi_log><gpu id=\"0\"><minor_number>{minor}</minor_number>"
            f"<uuid>{uuid}</uuid><fb_memory_usage><total>97887 MiB</total><used>{used} MiB</used></fb_memory_usage>"
            f"<processes>{processes}</processes></gpu></nvidia_smi_log>").encode()


class ServeTests(unittest.TestCase):
    """Serve plan basics; F3/F4, preflight refusals and the root runtime tools are in test_probe_runtime.py."""
    IDENTITY = {"uid": 1008, "euid": 1008, "gid": 1009, "egid": 1009, "user": "thilak.arjun", "host": "ai-core-team-1"}

    def spec(self, **overrides):
        values = dict(run="h1", instance="g6a", gpu_uuid=GPU6, port=8200)
        values.update(overrides)
        return serve.ServeSpec(**values)

    def test_spec_refuses_gpu5_and_bad_identifiers(self):
        with self.assertRaises(serve.ServePreflightError):
            self.spec(gpu_uuid=serve.GPU5_UUID)
        for overrides in (dict(gpu_uuid="5"), dict(gpu_uuid="GPU-5"), dict(port=80), dict(instance="Bad_Name"),
                          dict(gpu_memory_utilization=1), dict(max_num_seqs=0)):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.spec(**overrides)

    def test_vllm_flags_follow_the_thinking_study_plus_prefix_caching(self):
        argv = serve.vllm_argv(self.spec())
        pairs = {argv[i]: argv[i + 1] for i in range(len(argv) - 1) if argv[i].startswith("--")}
        self.assertEqual(argv[:3], [str(serve.SERVE_ENV / "bin" / "python"), "-m", "vllm.entrypoints.openai.api_server"])
        self.assertEqual((pairs["--model"], pairs["--served-model-name"], pairs["--host"], pairs["--port"],
                          pairs["--max-model-len"], pairs["--tool-call-parser"], pairs["--reasoning-parser"],
                          pairs["--dtype"], pairs["--generation-config"]),
                         (str(serve.MODEL_DIR), "absent-edges-gemma4-31b-bf16", "127.0.0.1", "8200", "32768",
                          "gemma4", "gemma4", "bfloat16", "vllm"))
        for flag in ("--enable-auto-tool-choice", "--language-model-only", "--enable-prefix-caching"):
            self.assertIn(flag, argv)

    def test_environment_is_allowlisted_and_local(self):
        env = serve.serve_environment(self.spec())
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], GPU6)
        self.assertEqual((env["HF_HUB_OFFLINE"], env["TRANSFORMERS_OFFLINE"]), ("1", "1"))
        for name in ("VLLM_CACHE_ROOT", "TRITON_CACHE_DIR", "HF_HOME", "HOME", "TMPDIR", "TORCHINDUCTOR_CACHE_DIR"):
            self.assertTrue(env[name].startswith("/dev/shm/ae-expA-h1/serve/g6a/"), name)
        self.assertFalse(set(env) & {"SSH_AUTH_SOCK", "HF_TOKEN", "USER", "KAGGLE_KEY"})

    def test_systemd_run_command_is_constructed_not_executed(self):
        with mock.patch.object(subprocess, "run") as run, mock.patch.object(subprocess, "Popen") as popen:
            plan = serve.plan(self.spec(), release_root="/rel", runtime_python="/py", identity=self.IDENTITY)
            run.assert_not_called()
            popen.assert_not_called()
        argv = plan["systemd_run_argv"]
        self.assertEqual(argv[:2], ["systemd-run", "--unit=ae-expA-serve-h1-g6a"])
        self.assertIn("--property=KillMode=control-group", argv)
        self.assertIn("--property=User=thilak.arjun", argv)
        self.assertEqual(argv[argv.index("--") + 1:argv.index("--") + 6],
                         ["/py", "-I", "-B", "/rel/scripts/expA_serve.py", "supervise"])
        self.assertEqual(serve.ServeSpec(**json.loads(argv[-1])), self.spec())
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout), mock.patch.object(serve, "current_identity", return_value=self.IDENTITY):
            self.assertEqual(load_script("expA_serve").main(["plan", "--run", "h1", "--instance", "g7a",
                                                             "--gpu-uuid", GPU6, "--port", "8201"]), 0)
        self.assertIn("sudo systemd-run --unit=ae-expA-serve-h1-g7a",
                      json.loads(stdout.getvalue())["systemd_run_command"])

    def test_gpu_observation_parsing_and_idle_rule(self):
        idle = serve.parse_gpu_xml(gpu_xml(GPU6, 6, 0), GPU6)
        self.assertEqual(idle, {"uuid": GPU6, "minor_number": 6, "used_mib": 0, "total_mib": 97887, "pids": []})
        self.assertIn("gpu_index_unknown", serve.check_idle(idle, GPU6)["failures"])
        idle["index"] = 6  # observe_gpu adds the nvidia-smi index
        self.assertTrue(serve.check_idle(idle, GPU6)["passed"])
        busy = serve.check_idle(dict(serve.parse_gpu_xml(gpu_xml(GPU6, 6, 500, [4321]), GPU6), index=6), GPU6)
        self.assertEqual(busy["failures"], ["gpu_memory_in_use_500MiB", "gpu_has_processes"])
        self.assertIn("gpu_minor_5_forbidden",
                      serve.check_idle(serve.parse_gpu_xml(gpu_xml(GPU6, 5, 0), GPU6), GPU6)["failures"])
        self.assertIn("gpu_index_5_forbidden", serve.check_idle(dict(idle, index=5), GPU6)["failures"])
        self.assertIn("gpu5_uuid_forbidden", serve.check_idle(dict(idle, uuid=serve.GPU5_UUID), GPU6)["failures"])
        self.assertTrue(serve.check_idle(dict(idle, used_mib=128), GPU6)["passed"])
        with self.assertRaises(ValueError):
            serve.parse_gpu_xml(gpu_xml(GPU6, 6, 0), "GPU-00000000-0000-0000-0000-000000000000")
        self.assertEqual(serve.parse_compute_apps(f"{GPU6}, 11\nGPU-x, 12\n", GPU6), [11])

    def test_foreign_pid_detection(self):
        groups = {101: 100, 202: 200}

        def pgid_of(pid):
            if pid not in groups:
                raise FileNotFoundError(pid)
            return groups[pid]
        # F4: an unresolvable PID (303) is never assumed ours; the watchdog re-observes it once.
        self.assertEqual(serve.unproven_pids({"pids": [101, 202, 303, 404]}, 100, own_pids={404}, pgid_of=pgid_of),
                         [202, 303])

    @unittest.skipUnless(os.name == "posix", "process groups required")
    def test_watchdog_stops_only_its_own_process_group(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(serve, "CACHE_PARENT", Path(directory)):
            spec = self.spec(watchdog_seconds=1)
            outsider = subprocess.Popen(["sleep", "30"], start_new_session=True)

            def observe(uuid):  # preflight is patched, so the first observation is the first watch
                return {"uuid": uuid, "minor_number": 6, "used_mib": 0, "total_mib": 97887,
                        "pids": [supervisor.child.pid, outsider.pid]}

            def popen(argv, **kwargs):
                return subprocess.Popen(["sleep", "30"], **kwargs)
            supervisor = serve.Supervisor(spec, observe=observe, sleep=lambda seconds: None, popen=popen,
                                          http_get=lambda path: {"data": []}, own_descendants=lambda pid: set())
            try:
                with mock.patch.object(serve, "preflight", return_value={"passed": True, "failures": []}):
                    self.assertEqual(supervisor.run(), 3)
                self.assertIsNotNone(supervisor.child.poll())
                self.assertIsNone(outsider.poll())
                names = sorted(path.name.split("-", 1)[1] for path in (spec.cache_root / "records").iterdir())
                self.assertEqual(names, ["collision.json", "launched.json", "preflight.json", "stopped.json"])
                collision = json.loads(next((spec.cache_root / "records").glob("*-collision.json")).read_text())
                self.assertEqual(collision["foreign"], [outsider.pid])
            finally:
                outsider.kill()
                outsider.wait()
            same_group = subprocess.Popen(["sleep", "30"])
            try:
                supervisor.child, supervisor.child_pgid = same_group, os.getpgrp()
                with self.assertRaises(RuntimeError):
                    supervisor.stop_own_group("test")
                self.assertIsNone(same_group.poll())
            finally:
                same_group.kill()
                same_group.wait()

    def test_failed_preflight_never_launches(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(serve, "CACHE_PARENT", Path(directory)):
            launched = []
            supervisor = serve.Supervisor(self.spec(), popen=lambda *a, **k: launched.append(a),
                                          observe=lambda uuid: {"uuid": uuid, "minor_number": 6, "used_mib": 900,
                                                                "total_mib": 97887, "pids": [77]})
            with mock.patch.object(serve, "port_is_free", return_value=True):
                self.assertEqual(supervisor.run(), 2)
            self.assertEqual(launched, [])


def _sandbox_available():
    if os.name != "posix" or not os.path.exists("/usr/bin/unshare") or "awk" not in discover_programs():
        return False
    probe = subprocess.run(["/usr/bin/unshare", "--user", "--map-root-user", "--net", "--mount", "--pid", "--fork",
                            "--kill-child", "/usr/bin/true"], capture_output=True, timeout=30)
    return probe.returncode == 0


SANDBOX = _sandbox_available()
GPU_A = "GPU-00000000-0000-0000-0000-00000000000a"
GPU_B = "GPU-00000000-0000-0000-0000-00000000000b"
ITEMS = {"item-a": {"target": TARGET, "line": 1, "pair": "A", "member": True},
         "item-b": {"target": "pkg/core.py::caller_one", "line": 5, "pair": "B", "member": False}}


def write_dispatch(base, tamper=None):
    """A synthetic v2 dispatch view for family "fam" (2 items) whose payloads match the checkout REPOS/fam."""
    from absent_edges.probe.identity import file_sha256s, scan_package_files, source_tree_sha256
    repo = make_repo(base / "repos" / "fam")
    tree = source_tree_sha256(file_sha256s(repo, scan_package_files(repo, ["pkg"])["files"]))
    directory = base / "dispatch"
    (directory / "payloads").mkdir(parents=True)
    (directory / "graph_facts").mkdir()
    store, rows = {}, []
    for item_id, spec in ITEMS.items():
        search = tools.lexical_search(repo, ["pkg"], spec["target"], limit=20, token_budget=2000)
        if tamper == "c4a_hits":
            search = dict(search, hits=search["hits"][:1])
        if tamper == "c4a_bound":
            search = dict(search, budget=dict(search["budget"], token_bound="exact:tokenizer_json"))
        cells = {}
        variants = {"omit_node": ("C1", "C2", "C3", "C4a"), "vcomplete": ("C1", "C3")}
        if spec["member"]:
            variants.update(natural=("C1", "C3", "C4a"), omit_edge=("C1", "C3"))
        for variant, conditions in variants.items():
            for condition in conditions:
                payload = condition_payload(condition, spec["target"], tree, verification=search, pair=spec["pair"])
                payload["facts_sha256"] = sha(f"{item_id}|{variant}".encode())
                encoded = canonical_json(payload)
                store[sha(encoded)] = payload
                cells.setdefault(variant, {})[condition] = sha(encoded)
        row = {"item_id": item_id, "family": "fam", "package_dirs": ["pkg"], "wording_pair": spec["pair"],
               "target": {"identity": spec["target"], "path": "pkg/core.py", "line": spec["line"],
                          "qualname": spec["target"].split("::")[1]},
               "membership": {"in_s_nat": spec["member"], "in_s_edge": spec["member"], "in_t": spec["member"]},
               "cells": cells}
        if tamper == "truth" and item_id == "item-b":
            row["omitted"] = {"identity": "pkg/core.py::Box.method"}
        rows.append(row)
    store_bytes = b"".join(canonical_json({"sha256": key, "bytes": len(canonical_json(value)), "payload": value})
                           + b"\n" for key, value in sorted(store.items()))
    (directory / "payloads" / "fam.jsonl").write_bytes(store_bytes)
    facts = facts_bytes(tree)
    (directory / "graph_facts" / "fam.json").write_bytes(facts)
    dispatch = b"".join(canonical_json(row) + b"\n" for row in rows)
    (directory / "dispatch.jsonl").write_bytes(dispatch)
    manifest = {"schema": "expA-dispatch-v2", "items_sha256": "1" * 64, "manifest_sha256": "2" * 64,
                "dispatch_jsonl_sha256": sha(dispatch), "rows": len(rows),
                "payload_store": {"fam": {"path": "payloads/fam.jsonl", "sha256": sha(store_bytes)}},
                "graph_facts": {"fam": {"path": "graph_facts/fam.json", "sha256": sha(facts)}},
                "families": {"fam": {"source_sha256": tree, "package_dirs": ["pkg"], "role": "dev"}}}
    (directory / "dispatch_manifest.json").write_bytes(canonical_json(manifest) + b"\n")
    return directory, tree


@unittest.skipUnless(os.name == "posix", "the runner checks checkouts with POSIX no-follow reads")
class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.dispatch, self.tree = write_dispatch(self.base)
        self.runner = load_script("expA_run")
        self.servers = []

    def tearDown(self):
        for server in self.servers:
            with contextlib.suppress(Exception):
                server.close()
        for path in self.base.rglob("*"):
            if path.is_dir() and not path.is_symlink():
                path.chmod(0o755)
        self.temporary.cleanup()

    def server(self, **kwargs):
        server = FakeServer(**kwargs)
        self.servers.append(server)
        return server

    def invoke(self, *argv):
        stdout = io.StringIO()
        previous = (signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT))
        try:
            with contextlib.redirect_stdout(stdout):
                code = self.runner.main(list(argv))
        finally:
            signal.signal(signal.SIGTERM, previous[0])
            signal.signal(signal.SIGINT, previous[1])
        return code, stdout.getvalue()

    def manifest(self):
        path = self.base / "checkouts" / "manifest.json"
        if not path.exists():
            code, output = self.invoke("checkout-manifest", "--out", str(path), "--repos-root", str(self.base / "repos"),
                                       "--dispatch", str(self.dispatch), "--commit", "fam=" + "c" * 40)
            self.assertEqual(code, 0)
            report = json.loads(output)
            self.assertTrue(report["families"]["fam"]["stager_accepts"])
        return path

    def live_args(self, *servers):
        endpoints = []
        for server, gpu in zip(servers, (GPU_A, GPU_B)):
            endpoints += ["--endpoint", f"{server.endpoint}={gpu}"]
        return ["--dispatch", str(self.dispatch), "--run-dir", str(self.base / "run"),
                "--ledger", str(self.base / "local" / "ledger.sqlite3"), "--scratch", str(self.base / "scratch"),
                "--checkout-manifest", str(self.manifest()), "--repos-root", str(self.base / "repos"), *endpoints]

    def test_cells_follow_the_frozen_design_and_order(self):
        view = self.runner.DispatchView(self.dispatch)
        cells = self.runner.build_cells(view, self.runner.PayloadStore(view))
        first, second = sorted(ITEMS, key=harness.item_order_key)
        expected = []
        for seed in (11, 23, 47):
            for item in (first, second):
                expected += [("P", item, variant, condition, seed) for variant, condition in
                             (("none", "C0"), ("omit_node", "C1"), ("omit_node", "C2"), ("omit_node", "C3"),
                              ("omit_node", "C4a"))]
        for item in (first, second):
            expected += [("S-VC", item, "vcomplete", "C1", 11), ("S-VC", item, "vcomplete", "C3", 11)]
        expected += [("S-NAT", "item-a", "natural", condition, 11) for condition in ("C1", "C3", "C4a")]
        expected += [("S-EDGE", "item-a", "omit_edge", condition, 11) for condition in ("C1", "C3")]
        expected += [("T", "item-a", "none", condition, 11) for condition in ("T-offered", "T-recommended")]
        self.assertEqual([(c["block"], c["item_id"], c["variant"], c["condition"], c["seed"]) for c in cells], expected)
        self.assertEqual([c["cell_id"] for c in cells[:2]], [f"P|{first}|none|C0|s11", f"P|{first}|omit_node|C1|s11"])
        for c in cells:
            self.assertEqual(c["cell_id"], f"{c['block']}|{c['item_id']}|{c['variant']}|{c['condition']}|s{c['seed']}")
            self.assertEqual(c["payload_sha256"] is None, c["condition"] in ("C0", "T-offered", "T-recommended"))
            self.assertEqual(c["wording_pair"], ITEMS[c["item_id"]]["pair"])
        self.assertEqual(len(self.runner.select_cells(cells, blocks={"P"}, seeds={11})), 10)
        self.assertEqual(len(self.runner.select_cells(cells, blocks={"S-VC", "T"})), 6)

    def test_dry_run_renders_every_prompt_and_writes_nothing(self):
        before = sorted(str(p) for p in self.base.rglob("*"))
        code, output = self.invoke("--dispatch", str(self.dispatch), "--dry-run", "--show", "3")
        summary = json.loads(output)
        self.assertEqual(code, 0)
        self.assertEqual((summary["design_cells"], summary["selected_cells"], summary["items"], summary["writes"]),
                         (41, 41, 2, "none"))
        self.assertEqual(summary["by_block"], {"P": 30, "S-EDGE": 2, "S-NAT": 3, "S-VC": 4, "T": 2})
        self.assertEqual(summary["unique_prompts"], 21)  # 10 P prompts (seed independent) + 4 + 3 + 2 + 2
        self.assertEqual(summary["sizes"]["P:none:C0"]["payload_bytes"]["max"], 0)
        self.assertGreater(summary["sizes"]["P:omit_node:C4a"]["user_bytes"]["min"],
                           summary["sizes"]["P:omit_node:C1"]["user_bytes"]["max"] - 1)
        self.assertEqual(len(summary["examples"]), 3)
        self.assertEqual(sorted(str(p) for p in self.base.rglob("*")), before)
        code, output = self.invoke("--dispatch", str(self.dispatch), "--dry-run", "--blocks", "P", "--seeds", "11")
        self.assertEqual(json.loads(output)["selected_cells"], 10)

    def test_selection_and_endpoint_arguments_are_checked(self):
        bad = [["--blocks", "S-C0"], ["--blocks", "P,X"], ["--seeds", "12"], ["--seeds", "eleven"],
               ["--endpoint", "http://127.0.0.1:9=GPU-7b8d9f89-0a8f-2d07-4833-bee70a0eeed2"],
               ["--endpoint", "http://127.0.0.1:9=5"], ["--endpoint", "http://10.0.0.1:9=" + GPU_A],
               ["--endpoint", "http://127.0.0.1:9=" + GPU_A, "--endpoint", "http://127.0.0.1:8=" + GPU_A], []]
        for extra in bad:
            with self.subTest(extra=extra), self.assertRaises(SystemExit), \
                    contextlib.redirect_stderr(io.StringIO()):
                self.runner.parse_args(["--dispatch", str(self.dispatch), *extra])

    def test_dispatch_view_refuses_truth_and_tampering(self):
        manifest_path = self.dispatch / "dispatch_manifest.json"
        original_manifest = manifest_path.read_bytes()
        original_rows = (self.dispatch / "dispatch.jsonl").read_bytes()

        def rewrite(rows=None, manifest=None, repin=True):
            data = original_rows if rows is None else b"".join(canonical_json(row) + b"\n" for row in rows)
            (self.dispatch / "dispatch.jsonl").write_bytes(data)
            document = json.loads(original_manifest) if manifest is None else manifest
            if repin:
                document["dispatch_jsonl_sha256"] = sha(data)
            manifest_path.write_bytes(canonical_json(document) + b"\n")

        def load():
            view = self.runner.DispatchView(self.dispatch)
            return self.runner.build_cells(view, self.runner.PayloadStore(view))

        rows = [json.loads(line) for line in original_rows.splitlines()]
        cases = {"truth V": [dict(rows[0], V=[{"identity": "x"}])] + rows[1:],
                 "omitted": [dict(rows[0], omitted={"identity": "pkg/core.py::caller_one"})] + rows[1:],
                 "covariates": [dict(rows[0], covariates={"natural_miss": True})] + rows[1:],
                 "target extra": [dict(rows[0], target=dict(rows[0]["target"], end_line=2))] + rows[1:],
                 "membership extra": [dict(rows[0], membership=dict(rows[0]["membership"], natural_miss=True))] + rows[1:],
                 # items.py strips s_nat_excluded_reason from the dispatch view: it would reveal natural_miss.
                 "membership reason": [dict(rows[0], membership=dict(rows[0]["membership"],
                                                                     s_nat_excluded_reason="no_natural_miss"))]
                 + rows[1:],
                 "membership null reason": [dict(rows[0], membership=dict(rows[0]["membership"],
                                                                          s_nat_excluded_reason=None))] + rows[1:],
                 "membership missing flag": [dict(rows[0], membership={"in_s_nat": True, "in_s_edge": True})]
                 + rows[1:],
                 "missing cells": [{key: value for key, value in rows[0].items() if key != "cells"}] + rows[1:],
                 "bad variant": [dict(rows[0], cells=dict(rows[0]["cells"], complete={"C1": "a" * 64}))] + rows[1:],
                 "duplicate": rows + rows[:1]}
        for name, changed in cases.items():
            with self.subTest(case=name), self.assertRaises(ValueError):
                rewrite(rows=changed)
                load()
        rewrite()
        load()
        manifest = json.loads(original_manifest)
        for name, document, repin in (("unpinned", {k: v for k, v in manifest.items() if k != "dispatch_jsonl_sha256"},
                                       False),
                                      ("wrong pin", dict(manifest, dispatch_jsonl_sha256="0" * 64), False),
                                      ("schema", dict(manifest, schema="expA-dispatch-v1"), True),
                                      ("escaping path", dict(manifest, graph_facts={"fam": {
                                          "path": "../graph.json", "sha256": "0" * 64}}), True),
                                      ("no items pin", {k: v for k, v in manifest.items() if k != "items_sha256"}, True)):
            with self.subTest(case=name), self.assertRaises(ValueError):
                rewrite(manifest=document, repin=repin)
                load()
        rewrite()
        store = self.dispatch / "payloads" / "fam.jsonl"
        original_store = store.read_bytes()
        lines = original_store.splitlines()
        tampered = [json.loads(line) for line in lines]
        tampered[0]["payload"]["state"] = "tampered"
        undispatched = dict(json.loads(lines[0])["payload"], guidance="a cell no dispatch row names")
        extra_row = canonical_json({"sha256": sha(canonical_json(undispatched)),
                                    "bytes": len(canonical_json(undispatched)), "payload": undispatched})
        for name, data in (("payload bytes", b"\n".join(canonical_json(row) for row in tampered) + b"\n"),
                           ("extra row key", b"\n".join([canonical_json(dict(json.loads(lines[0]), item="x"))]
                                                        + lines[1:]) + b"\n"),
                           ("unreferenced row", original_store + extra_row + b"\n")):
            store.write_bytes(data)
            document = json.loads(original_manifest)
            document["payload_store"]["fam"]["sha256"] = sha(data)
            with self.subTest(case=name), self.assertRaises(harness.PayloadIntegrityError):
                rewrite(manifest=document)
                load()
        store.write_bytes(original_store)
        rewrite()
        self.assertEqual(len(load()), 41)

    def test_membership_is_exactly_the_three_flags_of_the_dispatch_view(self):
        # Review finding: the runner admitted membership.s_nat_excluded_reason, which items.py strips from the
        # dispatch view because it reveals natural_miss. The runner now requires items.DISPATCH_MEMBERSHIP_KEYS.
        from absent_edges.probe import items as items_module
        self.assertEqual(self.runner.MEMBERSHIP_KEYS, frozenset(items_module.DISPATCH_MEMBERSHIP_KEYS))
        row = json.loads((self.dispatch / "dispatch.jsonl").read_bytes().splitlines()[0])
        self.assertIs(self.runner.validate_dispatch_row(row), row)
        for reason in ("no_natural_miss", None):
            leaky = dict(row, membership=dict(row["membership"], s_nat_excluded_reason=reason))
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, "exactly the boolean flags"):
                self.runner.validate_dispatch_row(leaky)
        with self.assertRaisesRegex(ValueError, "exactly the boolean flags"):
            self.runner.validate_dispatch_row(dict(row, membership=dict(row["membership"], in_t="yes")))

    def test_checkout_manifest_and_read_only_dry_run_checks(self):
        path = self.manifest()
        document = json.loads(path.read_bytes())
        entry = document["families"]["fam"]
        self.assertEqual((document["schema"], document["tree_protocol"], entry["commit"], entry["package_tree_sha256"]),
                         ("expA-clean-checkouts-v1", "expA-checkout-tree-v1", "c" * 40, self.tree))
        args = ["--dispatch", str(self.dispatch), "--dry-run", "--checkout-manifest", str(path),
                "--repos-root", str(self.base / "repos")]
        code, output = self.invoke(*args)
        report = json.loads(output)["checkouts"]["families"]["fam"]
        self.assertEqual((code, report["matches_manifest"], report["package_tree_sha256"],
                          report["c4a_blocks_equal_to_live_search"]), (0, True, self.tree, 3))
        (self.base / "repos" / "fam" / "stray.txt").write_text("leftover\n")
        report = json.loads(self.invoke(*args)[1])["checkouts"]["families"]["fam"]
        self.assertEqual((report["matches_manifest"], report["comparison"]["unexpected_files"]), (False, ["stray.txt"]))
        with self.assertRaises(FileExistsError):
            self.runner.write_checkouts_manifest(["fam"], path, repos_root=self.base / "repos",
                                                 package_dirs={"fam": ("pkg",)})

    def test_c4a_blocks_must_equal_the_live_search(self):
        for tamper, message in (("c4a_hits", "differs from the live source_reference_search"),
                                ("c4a_bound", "not the live tool's")):
            with self.subTest(tamper=tamper):
                base = self.base / tamper
                dispatch, _ = write_dispatch(base, tamper=tamper)
                manifest = base / "manifest.json"
                self.invoke("checkout-manifest", "--out", str(manifest), "--repos-root", str(base / "repos"),
                            "--dispatch", str(dispatch))
                with self.assertRaisesRegex(RuntimeError, message):
                    self.invoke("--dispatch", str(dispatch), "--dry-run", "--checkout-manifest", str(manifest),
                                "--repos-root", str(base / "repos"))

    def test_appendix_subcommand(self):
        code, output = self.invoke("appendix")
        appendix = json.loads(output)
        self.assertEqual(code, 0)
        self.assertEqual(appendix["design"]["blocks"]["P"]["cells"][0], ["none", "C0"])
        self.assertIn("source", appendix["c4_trigger_predicate"])
        self.assertIn("def c4_triggered", appendix["c4_trigger_predicate"]["source"])
        self.assertEqual(appendix["runner"]["dispatch_row_keys"], sorted(self.runner.DISPATCH_ROW_KEYS))
        self.assertIn("rejection_messages", appendix["tools"]["bash"])

    @unittest.skipUnless(os.name == "posix", "cell ids name directories ('|')")
    def test_status_counts_missing_infrastructure_as_terminal_like_the_analysis(self):
        # D11.3 as analysis.seed_completion applies it: a P cell blocked after 3 infrastructure attempts is
        # terminal, so its seed is complete and primary; the runner's status must say the same.
        view = self.runner.DispatchView(self.dispatch)
        cells = self.runner.build_cells(view, self.runner.PayloadStore(view))
        run_dir = self.base / "status-run"
        store = L.DurableStore(run_dir)
        store.initialize({"run_id": "status", "infra_attempt_cap": 3}, cells)
        seed11 = [cell for cell in cells if cell["block"] == "P" and cell["seed"] == 11]
        blocked = seed11[0]["cell_id"]
        for cell in seed11:
            for ordinal in range(1, 4 if cell["cell_id"] == blocked else 2):
                attempt_id = L.attempt_identifier(cell["cell_id"], ordinal)
                directory, _ = store.start_attempt(cell["cell_id"], attempt_id, {"ordinal": ordinal, "token": ordinal})
                L.write_immutable_json(directory / "result.json", {
                    "cell_id": cell["cell_id"], "attempt_id": attempt_id,
                    "exit_status": "InfraInterrupted" if cell["cell_id"] == blocked else "Submitted",
                    "endpoint": "http://127.0.0.1:1", "gpu_uuid": GPU_A})
        status = self.runner.run_status(run_dir)
        self.assertEqual(status["p_seeds"]["11"], {"completed": 9, "missing_infrastructure": 1})
        self.assertEqual((status["p_complete_seed_prefix"], status["p_seeds_without_missing_cells"]), ([11], []))
        self.assertEqual((status["infra_attempts"], status["cell_states"]["missing_infrastructure"]), (3, 1))

    @unittest.skipUnless(os.name == "posix", "cell ids name directories ('|')")
    def test_reconcile_closes_attempts_of_a_killed_runner_before_the_final_publish(self):
        # Review finding: a SIGKILLed runner leaves start-only attempts whose events.jsonl is never closed, so the
        # publisher never publishes them; `reconcile` closes them durably (owner verified gone) before final-publish.
        view = self.runner.DispatchView(self.dispatch)
        cells = self.runner.build_cells(view, self.runner.PayloadStore(view))
        store = L.DurableStore(self.base / "run")
        store.initialize({"run_id": "killed", "infra_attempt_cap": 3}, cells)
        gone = {"host": socket.gethostname(), "boot_id": L.boot_id(), "pid": 2 ** 22 + 7, "start_ticks": 5}
        cell = cells[0]["cell_id"]
        directory, _ = store.start_attempt(cell, f"{cell}.a001", {"ordinal": 1, "token": 1, "cell_id": cell,
                                                                  "owner_identity": gone})
        (directory / "events.jsonl").write_text("{\"kind\":\"episode_setup\"}\n")
        code, output = self.invoke("reconcile", "--run-dir", str(self.base / "run"),
                                   "--ledger", str(self.base / "local" / "ledger.sqlite3"))
        report = json.loads(output)
        self.assertEqual((code, report["run_id"], report["reconcile"]["closed_gone"], report["attempts_still_open"]),
                         (0, "killed", 1, []))
        closed = json.loads((directory / "attempt_closed.json").read_text())
        self.assertEqual((closed["status"], closed["reason"]), ("abandoned", "owner_verified_gone_without_result"))
        self.assertEqual(store.scan()[cell][0]["state"], "abandoned")

    @unittest.skipUnless(SANDBOX, "the live runner self-tests the namespace jail (run on Core2)")
    def test_live_p_first_then_secondary_restart_infra_and_rebuild(self):
        server = self.server(default={"status": 500})
        common = self.live_args(server)
        code, output = self.invoke(*common, "--blocks", "P", "--concurrency", "1")
        self.assertEqual(code, 3)
        summary = json.loads(output[output.index("\n{") + 1:])
        self.assertEqual((summary["halt_reason"], summary["stats"]["exit:InfraInterrupted"]),
                         ("3_consecutive_infra_interruptions", 3))
        server.default = SUBMIT
        code, output = self.invoke(*common, "--blocks", "P", "--concurrency", "2")
        self.assertEqual(code, 0)
        store = L.DurableStore(self.base / "run")
        scanned = store.scan()
        self.assertEqual(sorted({cell_id.split("|")[0] for cell_id in scanned}), ["P"])
        self.assertEqual(len(scanned), 30)
        states = [a["state"] for attempts in scanned.values() for a in attempts]
        self.assertEqual((states.count("completed"), states.count("infra_interrupted")), (30, 3))
        retried = {cell_id: attempts for cell_id, attempts in scanned.items() if len(attempts) == 2}
        self.assertEqual(len(retried), 3)
        for cell_id, attempts in retried.items():
            self.assertTrue(cell_id.endswith("|s11"))
            self.assertEqual([a["attempt_id"][-4:] for a in attempts], ["a001", "a002"])
        import sqlite3
        with contextlib.closing(sqlite3.connect(self.base / "local" / "ledger.sqlite3")) as db:
            claimed = [json.loads(detail)["attempt_id"] for (detail,) in
                       db.execute("SELECT detail FROM events WHERE event='claimed' ORDER BY seq")]
        order = {attempt: index for index, attempt in enumerate(claimed)}  # commit order of the claims
        last_s11 = max(order[f"{cell_id}.a002"] for cell_id in retried)
        first_s23 = min(index for attempt, index in order.items() if "|s23.a" in attempt)
        self.assertLess(last_s11, first_s23)  # D10.8: retried s11 cells run before seed 23
        code, output = self.invoke(*common, "--blocks", "S-VC,S-NAT,S-EDGE,T")
        self.assertEqual(code, 0)
        scanned = store.scan()
        self.assertEqual(len(scanned), 41)
        t_result = json.loads((store.attempt_dir("T|item-a|none|T-recommended|s11",
                                                 "T|item-a|none|T-recommended|s11.a001") / "result.json").read_bytes())
        self.assertEqual((t_result["exposure"]["tools_offered"][-1], t_result["exposure"]["recommendation_present"],
                          t_result["graph_query_calls"], t_result["gpu_uuid"]), ("graph_query", True, [], GPU_A))
        c0 = json.loads((store.attempt_dir(f"P|item-a|none|C0|s47", "P|item-a|none|C0|s47.a001")
                         / "result.json").read_bytes())
        self.assertEqual((c0["exposure"]["payload_present"], c0["submitted"]["sites"]),
                         (False, [{"path": "pkg/core.py", "line": 6}]))
        chats = len(server.chat_requests())
        self.assertEqual(self.invoke(*common)[0], 0)
        self.assertEqual(len(server.chat_requests()), chats)  # completed cells never run again
        for suffix in ("", "-wal", "-shm"):
            Path(str(self.base / "local" / "ledger.sqlite3") + suffix).unlink(missing_ok=True)
        code, output = self.invoke(*common)
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.splitlines()[0])["reconcile"]["completed_from_durable"], 41)
        self.assertEqual(len(server.chat_requests()), chats)
        code, output = self.invoke("status", "--run-dir", str(self.base / "run"))
        status = json.loads(output)
        self.assertEqual((status["cell_states"], status["p_complete_seed_prefix"], status["infra_attempts"]),
                         ({"completed": 41}, [11, 23, 47], 3))
        self.assertEqual(sum(status["exits_by_cell_label"]["P:none:C0"].values()), 6)
        run = json.loads((self.base / "run" / "run.json").read_bytes())
        self.assertEqual((run["dispatch_sha256"], run["items_sha256"], run["design"]["blocks"]["P"]["cells"][0]),
                         (sha((self.dispatch / "dispatch.jsonl").read_bytes()), "1" * 64, ["none", "C0"]))
        manifest_entry = json.loads(self.manifest().read_bytes())["families"]["fam"]
        self.assertEqual(run["checkouts"]["tree_sha256"]["fam"], manifest_entry["tree_sha256"])
        records = sorted((self.base / "run" / "launches").iterdir())
        launches = [path for path in records if not path.name.endswith(".end.json")]
        self.assertEqual((len(launches), len(records)), (5, 10))     # every launch also writes its end record
        launch = json.loads(launches[0].read_text())
        first_end = json.loads((self.base / "run" / "launches" / f"{launch['launch']}.end.json").read_text())
        self.assertEqual((first_end["halt_reason"], first_end["endpoints"][0]["tripped"]["reason"]),
                         ("3_consecutive_infra_interruptions", "3_consecutive_infra_interruptions"))
        self.assertTrue(launch["checkouts"]["fam"]["selftest"]["passed"])
        self.assertEqual(launch["checkouts"]["fam"]["tree_sha256"], manifest_entry["tree_sha256"])
        self.assertTrue(launch["checkouts"]["fam"]["comparison"]["matches"])

    @unittest.skipUnless(SANDBOX, "the live runner self-tests the namespace jail (run on Core2)")
    def test_two_endpoints_pull_one_queue_and_record_their_gpu(self):
        first, second = self.server(default=SUBMIT), self.server(default=SUBMIT)
        for server in (first, second):
            original = server.prompt_ids

            def slow(body, original=original):
                import time
                time.sleep(0.05)
                return original(body)
            server.prompt_ids = slow
        code, output = self.invoke(*self.live_args(first, second), "--blocks", "P", "--seeds", "11",
                                   "--concurrency", "2")
        self.assertEqual(code, 0)
        summary = json.loads(output[output.index("\n{") + 1:])
        store = L.DurableStore(self.base / "run")
        pairs = set()
        for cell_id, attempts in store.scan().items():
            result = attempts[0]["result"]
            pairs.add((result["endpoint"], result["gpu_uuid"]))
            self.assertEqual(attempts[0]["start"]["gpu_uuid"], result["gpu_uuid"])
        self.assertEqual(pairs, {(first.endpoint, GPU_A), (second.endpoint, GPU_B)})
        self.assertEqual(sum(sum(row.values()) for row in summary["condition_by_endpoint"].values()), 10)
        launch = json.loads(next((self.base / "run" / "launches").iterdir()).read_text())
        self.assertEqual([entry["gpu_uuid"] for entry in launch["endpoints"]], [GPU_A, GPU_B])

    def slow(self, server, delay):
        original = server.prompt_ids

        def delayed(body, original=original):
            import time
            time.sleep(delay)
            return original(body)
        server.prompt_ids = delayed
        return server

    @unittest.skipUnless(SANDBOX, "the live runner self-tests the namespace jail (run on Core2)")
    def test_a_failing_endpoint_is_isolated_and_never_exhausts_a_cell(self):
        # Review finding (runs/breaker-*): the global breaker was reset by the healthy endpoint, healthy workers left
        # on an empty claim, and requeued tail cells used up their 3 attempts on the failing endpoint.
        bad = self.slow(self.server(default={"status": 500}), 0.05)
        good = self.slow(self.server(default=SUBMIT), 0.15)
        code, output = self.invoke(*self.live_args(bad, good), "--blocks", "P", "--seeds", "11", "--concurrency", "2",
                                   "--health-interval", "0")
        self.assertEqual(code, 0, output[-2000:])
        summary = json.loads(output[output.index("\n{") + 1:])
        self.assertIsNone(summary["halt_reason"])
        states = {entry["endpoint"]: entry for entry in summary["endpoints"]}
        self.assertEqual(states[bad.endpoint]["tripped"]["reason"], "3_consecutive_infra_interruptions")
        self.assertIsNone(states[good.endpoint]["tripped"])
        self.assertEqual(summary["selected_counts"], {"completed": 10})
        scanned = L.DurableStore(self.base / "run").scan()
        self.assertEqual(len(scanned), 10)
        retried = 0
        for cell_id, attempts in scanned.items():
            self.assertEqual(attempts[-1]["state"], "completed", cell_id)
            self.assertEqual(attempts[-1]["result"]["endpoint"], good.endpoint, cell_id)
            self.assertLessEqual(len(attempts), 2, cell_id)
            for earlier, later in zip(attempts, attempts[1:]):
                retried += 1
                self.assertEqual(earlier["state"], "infra_interrupted")
                self.assertEqual(earlier["start"]["endpoint"], bad.endpoint)
                self.assertEqual(later["start"]["endpoint"], good.endpoint)  # never the endpoint that failed it
        self.assertGreaterEqual(retried, 3)
        end = json.loads(next(p for p in (self.base / "run" / "launches").iterdir()
                              if p.name.endswith(".end.json")).read_text())
        self.assertEqual(end["endpoints"], summary["endpoints"])

    @unittest.skipUnless(SANDBOX, "the live runner self-tests the namespace jail (run on Core2)")
    def test_an_endpoint_that_stops_answering_health_is_drained(self):
        sick = self.slow(self.server(default=SUBMIT), 0.4)
        sick.health_status = 500
        good = self.slow(self.server(default=SUBMIT), 0.05)
        code, output = self.invoke(*self.live_args(sick, good), "--blocks", "P", "--seeds", "11", "--concurrency", "1",
                                   "--health-interval", "0.05", "--health-timeout", "2", "--health-failures", "2")
        self.assertEqual(code, 0, output[-2000:])
        summary = json.loads(output[output.index("\n{") + 1:])
        states = {entry["endpoint"]: entry for entry in summary["endpoints"]}
        self.assertTrue(states[sick.endpoint]["tripped"]["reason"].startswith("health_probe_failed"))
        self.assertIsNone(states[good.endpoint]["tripped"])
        self.assertEqual(summary["selected_counts"], {"completed": 10})
        self.assertLess(states[sick.endpoint]["completed"], 10)

    @unittest.skipUnless(SANDBOX, "the live runner self-tests the namespace jail (run on Core2)")
    def test_a_launch_cannot_overtake_earlier_open_segments(self):
        # Review finding (scenario_order): with P s11 cells pending, '--blocks P --seeds 23' and '--blocks S-VC'
        # dispatched later segments; only the operator kept the D10.8 / D11.3 order across launches.
        server = self.server(default=SUBMIT)
        common = self.live_args(server)
        self.assertEqual(self.invoke(*common, "--blocks", "P", "--seeds", "11", "--limit", "2", "--concurrency", "1")[0], 0)
        chats = len(server.chat_requests())
        for selection, earlier in ((["--blocks", "P", "--seeds", "23"], {"P|s11": 8}),
                                   (["--blocks", "S-VC"], {"P|s11": 8, "P|s23": 10, "P|s47": 10}),
                                   (["--blocks", "T", "--families", "fam"], {"P|s11": 8, "P|s23": 10, "P|s47": 10,
                                                                             "S-VC|s11": 4, "S-NAT|s11": 3,
                                                                             "S-EDGE|s11": 2})):
            with self.subTest(selection=selection):
                code, output = self.invoke(*common, *selection)
                refusal = json.loads(output)
                self.assertEqual((code, refusal["status"]), (2, "refused"))
                self.assertEqual(refusal["order"]["earlier_open_cells_outside_selection"], earlier)
        self.assertEqual(len(server.chat_requests()), chats)               # nothing was dispatched
        code, output = self.invoke(*common, "--blocks", "P", "--seeds", "23", "--limit", "1",
                                   "--allow-out-of-order", "pilot throughput probe")
        self.assertEqual(code, 0)
        launches = sorted(p for p in (self.base / "run" / "launches").iterdir() if not p.name.endswith(".end.json"))
        order = json.loads(launches[-1].read_text())["selection"]["order"]
        self.assertEqual((order["override_reason"], order["earlier_open_cells_outside_selection"]),
                         ("pilot throughput probe", {"P|s11": 8}))
        self.assertEqual(self.invoke(*common, "--blocks", "P", "--seeds", "11")[0], 0)   # finishing s11 is allowed
        self.assertEqual(self.invoke(*common, "--blocks", "P")[0], 0)                    # all of P in one launch
        self.assertEqual(self.invoke(*common, "--blocks", "S-VC")[0], 0)                 # P is finished now
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.runner.parse_args(["--dispatch", str(self.dispatch), "--allow-out-of-order", " ", *common[2:]])

    @unittest.skipUnless(SANDBOX, "the live runner self-tests the namespace jail (run on Core2)")
    def test_staged_checkout_must_match_the_manifest(self):
        server = self.server(default=SUBMIT)
        args = self.live_args(server)
        (self.base / "repos" / "fam" / "pkg" / "generated.py").write_text("VERSION = 1\n")
        with self.assertRaisesRegex(RuntimeError, "differs from its clean-checkout manifest entry"):
            self.invoke(*args)
        self.assertEqual(server.chat_requests(), [])

    @unittest.skipUnless(SANDBOX, "the live runner self-tests the namespace jail (run on Core2)")
    def test_limit_dispatches_exactly_n(self):
        server = self.server(default=SUBMIT)
        code, output = self.invoke(*self.live_args(server), "--concurrency", "3", "--limit", "4")
        self.assertEqual(code, 0)
        summary = json.loads(output[output.index("\n{") + 1:])
        self.assertEqual(summary["stats"]["dispatched"], 4)
        self.assertEqual(len(server.chat_requests()), 4)

    def test_nfs_ledger_is_refused_before_anything_runs(self):
        mounts = [("/", "ext4", "/dev/root"), (str(self.base / "nfs"), "nfs4", "server:/x")]
        with mock.patch.object(L, "read_mount_table", return_value=mounts), \
                mock.patch.object(L, "statfs_magic", return_value=None):
            with self.assertRaises(L.UnsafeLedgerLocation):
                L.CellLedger(self.base / "nfs" / "ledger.sqlite3", run_id="r")
        self.assertFalse((self.base / "nfs").exists())


if __name__ == "__main__":
    unittest.main()
