"""SYNTHETIC tests for scripts/expA_finalturn_devcheck.py (ANALYSIS RECORD).

Every record below is SYNTHETIC; none of it is model output.
(1) The rendering rule on hand-written rendered prompts.
(2) Hand-made dev runs in the harness's record formats (run.json, cells.jsonl and cells/<cell_id>/<attempt_id>/
    {attempt_start.json, result.json, events.jsonl}). Prompts are rendered by a stand-in for the frozen Gemma 4 chat
    template and "tokenized" by code point (token id = ord(character)). They cover the go rule and its 90% boundary,
    every receipt check, the rendering check, terminal attempts only (D11.3, compared with
    absent_edges/probe/score.py) and the exit codes.
(3) Dev runs written by the real runner and harness against the fake server of tests/test_probe_harness.py (scripted
    responses), read through the same stand-in: the check accepts the records the release actually writes.
(4) The script's constants, pinned to protocols/EXPA_FINALTURN.md, the harness and the scorer.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("expA_finalturn_devcheck",
                                               ROOT / "scripts" / "expA_finalturn_devcheck.py")
D = importlib.util.module_from_spec(_SPEC)
sys.modules["expA_finalturn_devcheck"] = D
_SPEC.loader.exec_module(D)

POSIX = os.name == "posix"
if POSIX:
    import test_expA_default_golden as G      # puts the project on sys.path
    import test_probe_harness as H
    from absent_edges.probe import harness, score

RUN_ID = "finalturn-dev-synthetic"
ITEMS = ("synthetic-item-a", "synthetic-item-b")
P_CONDITIONS = ("C0", "C1", "C2", "C3", "C4a")
SYSTEM = "SYNTHETIC system prompt: find the callers of the target; you have at most 8 responses."
USER = "SYNTHETIC task: find every place where `target` (pkg/core.py, line 3) is invoked."
FORMAT_ERROR = ("Format error: your response did not call a tool. Every response must call at least one of the "
                "offered tools (source_reference_search, bash, submit_answer). Call submit_answer when you are ready "
                "to finish.")
SUBMISSION = {"sites": [{"path": "pkg/core.py", "line": 6}], "complete": "yes"}
POLICIES = ("neutral", "deadline")


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def needle(policy):
    return "[harness notice] " + D.TEXTS[policy]


def cell_id(item, condition, seed=11, block="P"):
    return f"{block}|{item}|{'none' if condition == 'C0' else 'omit_node'}|{condition}|s{seed}"


P11 = [cell_id(item, condition) for item in ITEMS for condition in P_CONDITIONS]


def codepoints(ids):
    """The SYNTHETIC tokenizer's detokenizer: token id = ord(character)."""
    return "".join(map(chr, ids))


def render(messages):
    """SYNTHETIC stand-in for the served rendering with the frozen Gemma 4 chat template (sha256 36e3a42e...), as far
    as the dev check reads it: an assistant message's tool calls, then each following tool message as
    <|tool_response>response:NAME{value:<|"|>CONTENT<|"|>}<tool_response|>, then the message's text and <turn|> (only
    when it has text); a user message is <|turn>user\\nCONTENT<turn|>\\n (trimmed); consecutive assistant messages
    share one model turn; the generation prompt follows unless the history ends inside the model turn. Tool
    declarations and call arguments are not rendered as the template does (the check never reads them)."""
    parts = ["<bos><|turn>system\n", messages[0]["content"].strip(), "<turn|>\n"]
    previous, last = None, None
    for index, message in enumerate(messages[1:], 1):
        role = message["role"]
        if role == "tool":
            continue
        last = None
        if role == "user":
            parts += ["<|turn>user\n", message["content"].strip(), "<turn|>\n"]
        else:
            if previous != "assistant":
                parts.append("<|turn>model\n")
            names = {}
            for call in message.get("tool_calls") or ():
                names[call["id"]] = call["function"]["name"]
                parts.append(f"<|tool_call>call:{call['function']['name']}{{{call['function']['arguments']}}}"
                             "<tool_call|>")
                last = "tool_call"
            responded = False
            for follow in messages[index + 1:]:
                if follow["role"] != "tool":
                    break
                parts.append(f"<|tool_response>response:{names.get(follow.get('tool_call_id'), 'unknown')}"
                             f'{{value:<|"|>{follow["content"]}<|"|>}}<tool_response|>')
                responded, last = True, "tool_response"
            text = (message.get("content") or "").strip()
            parts.append(text)
            if last == "tool_call" and not responded:
                parts.append("<|tool_response>")
            elif not (responded and not text):
                parts.append("<turn|>\n")
        previous = role
    if last not in ("tool_call", "tool_response"):
        parts.append("<|turn>model\n<|channel>thought\n<channel|>")
    return "".join(parts)


# ---------------------------------------------------------------------------------------------------------------
# hand-made SYNTHETIC runs
# ---------------------------------------------------------------------------------------------------------------

def episode(policy, responses):
    """SYNTHETIC events of one attempt, with the kinds, values and order the harness writes, and its result fields.

    ``responses`` has one entry per model response: "tool" (a source_reference_search call), "submit" (an accepted
    submit_answer), "text" (a format error), "limit" (ContextLimit before that call) or "infra" (the chat request
    fails); missing entries are "tool". Under a policy the notice is appended before the 8th call."""
    suffix = D.PREFIX + D.TEXTS[policy]
    config = D.expected_episode_config(policy)
    receipt = {"policy": policy, "text_sha256": sha(D.TEXTS[policy]), "suffix_sha256": sha(suffix), "call_index": 7,
               "appended": False, "sent": False, "delivered": False, "appended_to": None, "tool_call_id": None,
               "prompt_tokens": None, "max_tokens": None}
    rows, messages = [], [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}]

    def write(kind, value):
        rows.append({"sequence": len(rows), "wall_time": 1.0 + len(rows), "kind": kind, "value": value})
    write("episode_setup", {"config": config, "system": SYSTEM, "user": USER})
    exit_status, infra, consecutive, calls = None, None, 0, 0
    for call_index in range(8):
        response = responses[call_index] if call_index < len(responses) else "tool"
        noticed = call_index == 7
        if noticed:
            last = messages[-1]
            appended_to = "tool_result" if last["role"] == "tool" else "format_error"
            content = last["content"] + suffix
            messages[-1] = dict(last, content=content)
            receipt.update(appended=True, appended_to=appended_to, tool_call_id=last.get("tool_call_id"))
            write("final_turn_notice", {"call_index": 7, "policy": policy, "appended_to": appended_to,
                                        "tool_call_id": last.get("tool_call_id"), "suffix": suffix,
                                        "suffix_sha256": sha(suffix), "text_sha256": sha(D.TEXTS[policy]),
                                        "previous_content_sha256": sha(last["content"]),
                                        "content_sha256": sha(content)})
        prompt = render(messages)
        write("context_accounting", {"call_index": call_index, "count": len(prompt), "max_model_len": 32768})
        if call_index == 0:
            write("exposure_receipt", {"recorded_before_first_model_call": True})
        if noticed:
            receipt["prompt_tokens"] = len(prompt)
        if response == "limit":
            write("context_limit", {"call_index": call_index, "prompt_tokens": 16384 - 511, "remaining": 511})
            exit_status = "ContextLimit"
            break
        max_tokens = 4096 - call_index          # SYNTHETIC: a distinct value per call
        if noticed:
            receipt.update(sent=True, max_tokens=max_tokens)
        if response == "infra":
            write("model_request_failed", {"call_index": call_index, "max_tokens": max_tokens})
            exit_status, infra = "InfraInterrupted", {"kind": "server", "message": "SYNTHETIC failure"}
            break
        call_id = f"call-{call_index}"
        if response == "text":
            message = {"role": "assistant", "content": "SYNTHETIC text without a tool call."}
        else:
            name, arguments = (("submit_answer", SUBMISSION) if response == "submit"
                               else ("source_reference_search", {"query": "target"}))
            message = {"role": "assistant", "content": None, "tool_calls": [
                {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}]}
        write("model_exchange", {"call_index": call_index, "max_tokens": max_tokens,
                                 "request": {"messages": copy.deepcopy(messages)},
                                 "response": {"choices": [{"message": message, "finish_reason": "stop"}],
                                              "usage": {"prompt_tokens": len(prompt), "completion_tokens": 5},
                                              "prompt_token_ids": [ord(char) for char in prompt]}})
        calls += 1
        if noticed:
            receipt["delivered"] = True
        if response == "text":
            consecutive += 1
            messages += [{"role": "assistant", "content": message["content"]},
                         {"role": "user", "content": FORMAT_ERROR}]
            write("format_error", {"call_index": call_index, "visible": FORMAT_ERROR, "consecutive": consecutive})
            if consecutive >= 3:
                exit_status = "FormatExhausted"
                break
            continue
        consecutive = 0
        messages.append(message)
        write("parsed_actions", {"call_index": call_index, "actions": [{"id": call_id, "name": name,
                                                                        "arguments": arguments}]})
        if response == "submit":
            write("submission_accepted", {"tool_call_id": call_id, "call_index": call_index, "submission": SUBMISSION})
            exit_status = "Submitted"
            break
        content = canonical({"hits": [{"path": "pkg/core.py", "line": 6}], "query": "target"}).decode("utf-8")
        write("model_visible_tool_result", {"tool_call_id": call_id, "name": name, "content": content,
                                            "sha256": sha(content)})
        messages.append({"role": "tool", "tool_call_id": call_id, "content": content})
    else:
        exit_status = "NoAnswer"
    write("episode_finished", {"exit_status": exit_status, "infra": infra,
                               "accounting": {"model_calls": calls, "final_turn_notice": receipt}})
    return rows, {"exit_status": exit_status, "infra": infra, "calls": calls, "config": config, "receipt": receipt}


def write_attempt(run_dir, cell, ordinal, policy, responses):
    attempt_id = f"{cell}.a{ordinal:03d}"
    directory = run_dir / "cells" / cell / attempt_id
    directory.mkdir(parents=True)
    start = canonical({"cell_id": cell, "attempt_id": attempt_id, "ordinal": ordinal, "launch": "SYNTHETIC"})
    (directory / "attempt_start.json").write_bytes(start)
    rows, outcome = episode(policy, responses)
    events = b"".join(json.dumps(row).encode("utf-8") + b"\n" for row in rows)
    (directory / "events.jsonl").write_bytes(events)
    block, item, variant, condition, seed = cell.split("|")
    receipt = outcome["receipt"]
    result = {"schema": "expA-episode-result-v2", "run_id": RUN_ID, "cell_id": cell, "attempt_id": attempt_id,
              "attempt": {"attempt_id": attempt_id, "ordinal": ordinal}, "block": block, "item_id": item,
              "variant": variant, "condition": condition, "seed": int(seed[1:]), "family": "synthetic",
              "exit_status": outcome["exit_status"], "calls": outcome["calls"],
              "submitted": SUBMISSION if outcome["exit_status"] == "Submitted" else None,
              "exposure": {"recorded_before_first_model_call": True, "final_turn_notice": dict(receipt)},
              "accounting": {"model_calls": outcome["calls"], "final_turn_notice": dict(receipt)},
              "events": {"path": "events.jsonl", "sha256": hashlib.sha256(events).hexdigest(), "records": len(rows)},
              "infra": outcome["infra"], "config": outcome["config"],
              "run_context": {"launch": "SYNTHETIC", "start_sha256": hashlib.sha256(start).hexdigest()}}
    (directory / "result.json").write_bytes(canonical(result))
    return directory


def write_run(directory, policy, plan, *, episode_config=None):
    """A SYNTHETIC dev run: run.json, cells.jsonl (the whole design: block P at seeds 11, 23 and 47 and S-VC at seed
    11, as a real run.json freezes it) and, for each cell of ``plan``, its attempts in order (response lists)."""
    directory.mkdir(parents=True)
    design = [cell_id(item, condition, seed) for seed in (11, 23, 47) for item in ITEMS for condition in P_CONDITIONS]
    design += [f"S-VC|{item}|vcomplete|{condition}|s11" for item in ITEMS for condition in ("C1", "C3")]
    cells = b"".join(canonical({"cell_id": cell}) + b"\n" for cell in design)
    (directory / "cells.jsonl").write_bytes(cells)
    run = {"schema": "expA-run-SYNTHETIC", "run_id": RUN_ID, "cells_sha256": hashlib.sha256(cells).hexdigest(),
           "cell_count": len(design), "model": {"profile": "gemma4-31b", "revision": "SYNTHETIC"},
           "episode_config": D.expected_episode_config(policy) if episode_config is None else episode_config}
    (directory / "run.json").write_bytes(canonical(run) + b"\n")
    for cell, attempts in plan.items():
        for ordinal, responses in enumerate(attempts, 1):
            write_attempt(directory, cell, ordinal, policy, responses)
    return directory


TOOLS8 = ["tool"] * 8
# SYNTHETIC: the plan of each block-P seed-11 cell (one response list per attempt).
STANDARD = {
    P11[0]: [["tool", "submit"]],                                    # submits at response 2: never noticed
    P11[1]: [["tool"] * 6 + ["text", "tool"]],                       # a format error at 7: notice on its message
    P11[2]: [["tool"] * 7 + ["limit"]],                              # ContextLimit at the 8th call: never sent
    P11[3]: [["tool"] * 7 + ["infra"], ["tool"] * 7 + ["submit"]],   # an infrastructure attempt, then terminal
    P11[4]: [["tool"] * 7 + ["submit"]],                             # submits in the noticed response
    **{cell: [TOOLS8] for cell in P11[5:]},                          # tool calls through response 8: NoAnswer
}


def attempt_dir(run_dir, cell, ordinal=1):
    return run_dir / "cells" / cell / f"{cell}.a{ordinal:03d}"


def edit_result(directory, change):
    path = directory / "result.json"
    result = json.loads(path.read_bytes())
    change(result)
    path.write_bytes(canonical(result))


def edit_events(directory, change, *, rebind=True):
    """Rewrite events.jsonl with ``change(rows)``; by default rebind it in result.json (sha256, records), so that only
    the check under test sees the edit."""
    path = directory / "events.jsonl"
    rows = [json.loads(line) for line in path.read_bytes().splitlines()]
    rows = change(rows) or rows
    for index, row in enumerate(rows):
        row["sequence"] = index
    raw = b"".join(json.dumps(row).encode("utf-8") + b"\n" for row in rows)
    path.write_bytes(raw)
    if rebind:
        edit_result(directory, lambda result: result.update(events=dict(
            result["events"], sha256=hashlib.sha256(raw).hexdigest(), records=len(rows))))


def set_receipt(directory, **values):
    """Change the receipt everywhere it is recorded (exposure, accounting, the episode_finished event)."""
    def change_result(result):
        for place in ("exposure", "accounting"):
            result[place]["final_turn_notice"].update(values)
    edit_result(directory, change_result)

    def change_events(rows):
        for row in rows:
            if row["kind"] == "episode_finished":
                row["value"]["accounting"]["final_turn_notice"].update(values)
    edit_events(directory, change_events)


def without(kind, call_index=7):
    def change(rows):
        return [row for row in rows if not (row["kind"] == kind and row["value"].get("call_index") == call_index)]
    return change


# ---------------------------------------------------------------------------------------------------------------
# (1) the rendering rule
# ---------------------------------------------------------------------------------------------------------------

class PlacementTests(unittest.TestCase):
    TOOL = ('<|tool_call>call:source_reference_search{query:<|"|>t<|"|>}<tool_call|>'
            '<|tool_response>response:source_reference_search{value:<|"|>{"hits":[]}<|"|>}<tool_response|>')

    def tool_prompt(self, policy, *, after=""):
        return ("<bos><|turn>system\nSYNTHETIC<turn|>\n<|turn>user\nSYNTHETIC task<turn|>\n<|turn>model\n"
                + self.TOOL * 6 + '<|tool_call>call:bash{command:<|"|>ls<|"|>}<tool_call|>'
                + '<|tool_response>response:bash{value:<|"|>{"output":"ok"}\n\n' + needle(policy)
                + '<|"|>}<tool_response|>' + after)

    def format_prompt(self, policy, *, user=FORMAT_ERROR, tail="<|turn>model\n<|channel>thought\n<channel|>"):
        return ("<bos><|turn>system\nSYNTHETIC<turn|>\n<|turn>user\nSYNTHETIC task<turn|>\n<|turn>model\n"
                + self.TOOL * 6 + "SYNTHETIC text.<turn|>\n<|turn>user\n" + user + "\n\n" + needle(policy)
                + "<turn|>\n" + tail)

    def test_the_two_places_the_notice_ends(self):
        for policy in POLICIES:
            with self.subTest(policy=policy):
                text = needle(policy)
                self.assertIsNone(D.notice_placement(self.tool_prompt(policy), text, "tool_result"))
                # Response 7's text follows its tool responses and closes the model turn: still the end of the last
                # tool response.
                self.assertIsNone(D.notice_placement(self.tool_prompt(policy, after="Let me check.<turn|>\n"), text,
                                                     "tool_result"))
                self.assertIsNone(D.notice_placement(self.format_prompt(policy), text, "format_error"))

    def test_every_misplacement_is_refused(self):
        text = needle("deadline")
        tool, fmt = self.tool_prompt("deadline"), self.format_prompt("deadline")
        cases = {
            "missing": (tool.replace(text, ""), "tool_result", "the notice occurs 0 time(s)"),
            "twice": (tool.replace(text, text + " " + text), "tool_result", "the notice occurs 2 time(s)"),
            "the other policy's too": (tool.replace(text, text + needle("neutral")), "tool_result",
                                       "'[harness notice]' 2 time(s)"),
            "text after it in its response": (tool.replace(text, text + " more"), "tool_result",
                                              "not the end of a tool response's content"),
            "a later tool response": (tool + self.TOOL, "tool_result", "a later tool response follows"),
            "outside a tool response": (tool.replace("<|tool_response>response:bash", "<|tool_call_x>response:bash"),
                                        "tool_result", "not inside a tool response"),
            "format error: text after it": (fmt.replace(text, text + " more"), "format_error",
                                            "not the end of a turn"),
            "format error: in a model turn": (fmt.replace("<turn|>\n<|turn>user\n" + FORMAT_ERROR, "\n" + FORMAT_ERROR),
                                              "format_error", "not inside a user turn"),
            "format error: another user turn": (self.format_prompt("deadline", user="SYNTHETIC user text"),
                                                "format_error", "is not the format-error message"),
            "format error: a later user turn": (self.format_prompt("deadline", tail="<|turn>user\nlater<turn|>\n"),
                                                "format_error", "a later user turn follows"),
            "appended_to mismatch": (fmt, "tool_result", "not the end of a tool response's content"),
            "appended_to unknown": (tool, None, "appended_to is None"),
        }
        for name, (prompt, appended_to, reason) in cases.items():
            with self.subTest(case=name):
                problem = D.notice_placement(prompt, text, appended_to)
                self.assertIsNotNone(problem)
                self.assertIn(reason, problem)


# ---------------------------------------------------------------------------------------------------------------
# (2) hand-made SYNTHETIC dev runs
# ---------------------------------------------------------------------------------------------------------------

@unittest.skipUnless(POSIX, "cell ids (with '|') are directory names, as on Core2")
class SyntheticRunTests(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.count = 0

    def tearDown(self):
        self.temporary.cleanup()

    def run_dir(self, policy="deadline", plan=None, **kwargs):
        self.count += 1
        return write_run(self.base / f"run-{self.count}", policy, STANDARD if plan is None else plan, **kwargs)

    def check(self, run_dir, policy="deadline", detokenize=codepoints):
        return D.check_run(run_dir, policy, detokenize)

    def test_a_dev_run_that_passes(self):
        for policy in POLICIES:
            with self.subTest(policy=policy):
                report = self.check(self.run_dir(policy), policy)
                self.assertEqual((report["go"], report["failed_checks"]), (True, []))
                self.assertEqual({name: entry["passed"] for name, entry in report["checks"].items()},
                                 dict.fromkeys(D.CHECKS, True))
                self.assertEqual({key: report["go_rule"][key] for key in ("noticed_final_responses", "valid_tool_calls",
                                                                          "format_errors", "share", "passed")},
                                 {"noticed_final_responses": 8, "valid_tool_calls": 8, "format_errors": 0,
                                  "share": 1.0, "passed": True})
                notice = report["notice"]
                self.assertEqual((notice["terminal_attempts"], notice["appended"], notice["sent"], notice["delivered"]),
                                 (10, 9, 8, 8))
                self.assertEqual(notice["appended_to"], {"tool_result": 8, "format_error": 1})
                self.assertEqual(notice["delivered_by_appended_to"], {"tool_result": 7, "format_error": 1})
                self.assertEqual(notice["format_error_cells"], [P11[1]])
                self.assertEqual(notice["appended_not_sent_cells"], [P11[2]])
                self.assertEqual(notice["not_reached_by_exit"], {"Submitted": 1})
                cells = report["cells"]
                self.assertEqual(cells["terminal_exits"], {"ContextLimit": 1, "NoAnswer": 6, "Submitted": 3})
                self.assertEqual(cells["states"], {"completed": 10, "missing_infrastructure": 0, "infra_pending": 0,
                                                   "not_dispatched": 0})
                self.assertEqual((cells["block_p_seed_11_in_design"], cells["dispatched"],
                                  cells["infrastructure_attempts_by_kind"]), (10, 10, {"server": 1}))
                # The checks ran on every unit: terminal attempts (10), delivered 8th requests (8).
                self.assertEqual({name: report["checks"][name]["checked"] for name in D.CHECKS},
                                 {"run_json_episode_config": 1, "run_json_cells": 1, "selection": 10,
                                  "run_complete": 10, "records": 10, "config_receipts": 10, "notice_receipts": 10,
                                  "appended_iff_event": 10, "delivered_iff_exchange": 10, "receipt_accounting": 10,
                                  "request_bytes": 8, "rendering": 8, "final_response_recorded": 8})
                self.assertEqual(sorted(sample["appended_to"] for sample in report["rendering_samples"]),
                                 ["format_error", "tool_result"])
                self.assertEqual(report["text"], D.TEXTS[policy])
                json.dumps(report, allow_nan=False)

    def test_the_go_rule_is_at_least_90_percent_of_the_noticed_final_responses(self):
        for invalid, go in ((0, True), (1, True), (2, False)):
            with self.subTest(invalid=invalid):
                plan = {cell: [["tool"] * 7 + (["text"] if index < invalid else ["tool"])]
                        for index, cell in enumerate(P11)}
                report = self.check(self.run_dir(plan=plan))
                rule = report["go_rule"]
                self.assertEqual((rule["noticed_final_responses"], rule["valid_tool_calls"], rule["format_errors"]),
                                 (10, 10 - invalid, invalid))
                self.assertEqual((rule["passed"], report["go"], report["failed_checks"]), (go, go, []))
        # No noticed final response: the rule cannot be met.
        report = self.check(self.run_dir(plan={cell: [["submit"]] for cell in P11}))
        self.assertEqual((report["go_rule"]["noticed_final_responses"], report["go_rule"]["share"],
                          report["go_rule"]["passed"], report["go"]), (0, None, False, False))

    def test_only_terminal_attempts_count(self):
        """The reviewer's case: an InfraInterrupted attempt with delivered true and parsed actions at call_index 7 is
        neither counted nor checked; the retried cell counts once, through its terminal attempt."""
        run_dir = self.run_dir()
        infra = attempt_dir(run_dir, P11[3], 1)

        def delivered_and_parsed(rows):
            exchange = {"kind": "model_exchange", "value": {"call_index": 7, "max_tokens": 4089, "request": {},
                                                            "response": {"prompt_token_ids": [1]}}}
            parsed = {"kind": "parsed_actions", "value": {"call_index": 7, "actions": []}}
            return rows[:-1] + [exchange, parsed] + rows[-1:]
        edit_events(infra, delivered_and_parsed)
        set_receipt(infra, delivered=True)
        edit_result(infra, lambda result: result["exposure"].pop("final_turn_notice"))   # broken, and never read
        report = self.check(run_dir)
        self.assertEqual((report["go"], report["failed_checks"]), (True, []))
        self.assertEqual((report["go_rule"]["noticed_final_responses"], report["go_rule"]["valid_tool_calls"]), (8, 8))
        self.assertEqual(report["checks"]["records"]["checked"], 10)
        # Terminal attempts are those of the frozen scorer (D11.3).
        attempts, _ = D.load_attempts(run_dir)
        scored = score.scan_run(run_dir)
        self.assertEqual(sorted(attempts), sorted(scored))
        for cell in attempts:
            state, terminal, infra_attempts, later = D.resolve(attempts[cell])
            expected = score.resolve_cell(scored[cell])
            self.assertEqual((state, terminal["attempt_id"] if terminal else None, len(infra_attempts)),
                             (expected["state"], expected["terminal"]["attempt_id"] if expected["terminal"] else None,
                              expected["infra_attempts"]))
            self.assertEqual(later, [])
        self.assertEqual(D.resolve(attempts[P11[3]])[1]["attempt_id"], P11[3] + ".a002")

    def test_cells_without_a_terminal_attempt(self):
        plan = dict(STANDARD)
        plan[P11[8]] = [["tool"] * 7 + ["infra"]] * 3      # missing (infrastructure): terminal in the D11.3 sense
        plan[P11[9]] = [["tool"] * 7 + ["infra"]]          # infra_pending: the run is not finished
        report = self.check(self.run_dir(plan=plan))
        self.assertEqual(report["cells"]["states"], {"completed": 8, "missing_infrastructure": 1, "infra_pending": 1,
                                                     "not_dispatched": 0})
        self.assertEqual(report["cells"]["infrastructure_attempts_by_kind"], {"server": 5})
        self.assertEqual((report["go"], report["failed_checks"]), (False, ["run_complete"]))
        self.assertEqual(report["checks"]["run_complete"]["examples"],
                         [{"where": f"cells/{P11[9]}", "problems": ["cell state infra_pending"]}])
        self.assertEqual((report["go_rule"]["noticed_final_responses"], report["notice"]["terminal_attempts"]), (6, 8))

    def test_each_receipt_check_refuses_its_defect(self):
        """One defect per case on the passing run; exactly the named checks fail and the run is no-go."""
        other = {"neutral": "deadline", "deadline": "neutral"}

        def run_json(change):
            def mutate(run_dir):
                run = json.loads((run_dir / "run.json").read_bytes())
                change(run)
                (run_dir / "run.json").write_bytes(canonical(run) + b"\n")
            return mutate

        def at(cell, function, ordinal=1):
            return lambda run_dir: function(attempt_dir(run_dir, cell, ordinal))

        def last_message(rows):
            for row in rows:
                if row["kind"] == "model_exchange" and row["value"]["call_index"] == 7:
                    message = row["value"]["request"]["messages"][-1]
                    message["content"] = message["content"][:-len(D.PREFIX + D.TEXTS["deadline"])]

        def stray_notice(rows):
            event = {"kind": "final_turn_notice", "value": {"call_index": 7, "policy": "deadline"}}
            return rows[:-1] + [event] + rows[-1:]

        def no_token_ids(rows):
            for row in rows:
                if row["kind"] == "model_exchange" and row["value"]["call_index"] == 7:
                    del row["value"]["response"]["prompt_token_ids"]

        def event_value(**values):
            def change(rows):
                for row in rows:
                    if row["kind"] == "final_turn_notice":
                        row["value"].update(values)
            return change

        def notice_before_call_6(rows):
            notice = next(row for row in rows if row["kind"] == "final_turn_notice")
            rows = [row for row in rows if row is not notice]
            index = next(i for i, row in enumerate(rows)
                         if row["kind"] == "context_accounting" and row["value"]["call_index"] == 6)
            return rows[:index] + [notice] + rows[index:]

        def notice_twice_in_request(rows):
            for row in rows:
                if row["kind"] == "model_exchange" and row["value"]["call_index"] == 7:
                    earlier = row["value"]["request"]["messages"][-3]      # response 6's tool result
                    assert earlier["role"] == "tool"
                    earlier["content"] += D.PREFIX + D.TEXTS["deadline"]

        def remove_cell(cell):
            return lambda run_dir: shutil.rmtree(run_dir / "cells" / cell)

        def reserialized(directory):
            """The same rows in other bytes (sorted keys), not rebound: result.json binds another file."""
            path = directory / "events.jsonl"
            rows = [json.loads(line) for line in path.read_bytes().splitlines()]
            path.write_bytes(b"".join(json.dumps(row, sort_keys=True).encode("utf-8") + b"\n" for row in rows))

        cases = {
            "run.json records the other policy": (
                run_json(lambda run: run.update(episode_config=D.expected_episode_config(other["deadline"]))),
                {"run_json_episode_config"}),
            "run.json without the notice record": (
                run_json(lambda run: run.update(episode_config=dict(D.FROZEN_CONFIG, final_turn_policy="deadline"))),
                {"run_json_episode_config"}),
            "cells.jsonl is not the bound file": (
                lambda run_dir: (run_dir / "cells.jsonl").write_bytes((run_dir / "cells.jsonl").read_bytes() + b"{}\n"),
                {"run_json_cells"}),
            "result.json config": (
                at(P11[5], lambda d: edit_result(d, lambda result: result.update(config=dict(D.FROZEN_CONFIG)))),
                {"config_receipts"}),
            "accounting differs from exposure": (
                at(P11[5], lambda d: edit_result(d, lambda result: result["accounting"]["final_turn_notice"].update(
                    tool_call_id="call-other"))), {"notice_receipts"}),
            "the episode_finished receipt differs": (
                at(P11[5], lambda d: edit_events(d, lambda rows: rows[-1]["value"]["accounting"]["final_turn_notice"]
                                                 .update(sent=False))), {"notice_receipts"}),
            "a wrong text sha256": (at(P11[5], lambda d: set_receipt(d, text_sha256="0" * 64)), {"notice_receipts"}),
            "a wrong suffix sha256": (at(P11[5], lambda d: set_receipt(d, suffix_sha256="0" * 64)),
                                      {"notice_receipts"}),
            "another policy in the receipt": (at(P11[5], lambda d: set_receipt(d, policy="neutral")),
                                              {"notice_receipts"}),
            "a field the receipt does not have": (at(P11[5], lambda d: set_receipt(d, extra=True)),
                                                  {"notice_receipts"}),
            "appended without its event": (at(P11[5], lambda d: edit_events(d, without("final_turn_notice"))),
                                           {"appended_iff_event", "request_bytes"}),
            "an event without appended": (at(P11[0], lambda d: edit_events(d, stray_notice)), {"appended_iff_event"}),
            "the event disagrees with the receipt": (
                at(P11[5], lambda d: edit_events(d, event_value(appended_to="format_error"))), {"appended_iff_event"}),
            "the event is not right before the call-7 preflight": (
                at(P11[5], lambda d: edit_events(d, notice_before_call_6)), {"appended_iff_event"}),
            "sent false on a delivered request": (at(P11[5], lambda d: set_receipt(d, sent=False)),
                                                  {"delivered_iff_exchange", "receipt_accounting"}),
            "delivered false with a model_exchange at 7": (at(P11[5], lambda d: set_receipt(d, delivered=False)),
                                                           {"delivered_iff_exchange"}),
            "delivered without a model_exchange at 7": (at(P11[5], lambda d: edit_events(d, without("model_exchange"))),
                                                        {"delivered_iff_exchange", "receipt_accounting"}),
            "prompt_tokens is not the preflight count": (at(P11[5], lambda d: set_receipt(d, prompt_tokens=1)),
                                                         {"receipt_accounting"}),
            "appended, not sent, without ContextLimit": (
                at(P11[2], lambda d: edit_result(d, lambda result: result.update(exit_status="NoAnswer"))),
                {"receipt_accounting"}),
            "the request lost its suffix": (at(P11[5], lambda d: edit_events(d, last_message)), {"request_bytes"}),
            "the notice twice in the request": (at(P11[5], lambda d: edit_events(d, notice_twice_in_request)),
                                                {"request_bytes"}),
            "no prompt token ids": (at(P11[5], lambda d: edit_events(d, no_token_ids)), {"rendering"}),
            "the 8th response is neither parsed nor a format error": (
                at(P11[5], lambda d: edit_events(d, without("parsed_actions"))), {"final_response_recorded"}),
            "events.jsonl is not the bound file": (at(P11[5], reserialized), {"records"}),
            "result.json names another run": (at(P11[5], lambda d: edit_result(d, lambda r: r.update(run_id="other"))),
                                              {"records"}),
            "the start record of another launch": (
                at(P11[5], lambda d: (d / "attempt_start.json").write_bytes(b'{"launch":"other"}')), {"records"}),
            "an attempt after the terminal one": (
                lambda run_dir: write_attempt(run_dir, P11[0], 2, "deadline", ["submit"]), {"records"}),
            "a cell outside block P seed 11": (
                lambda run_dir: write_attempt(run_dir, cell_id(ITEMS[0], "C1", seed=23), 1, "deadline", TOOLS8),
                {"selection"}),
            "a cell never dispatched": (remove_cell(P11[9]), {"run_complete"}),
        }
        for name, (mutate, failing) in cases.items():
            with self.subTest(case=name):
                run_dir = self.run_dir()
                mutate(run_dir)
                report = self.check(run_dir)
                self.assertEqual(set(report["failed_checks"]), failing)
                self.assertFalse(report["go"])
                for check in failing:
                    self.assertGreaterEqual(len(report["checks"][check]["examples"]), 1)

    def test_the_rendering_check_reads_the_served_prompt(self):
        """The served stack is the only place a misrendered notice shows: the request bytes are right, the
        detokenized prompt is not."""
        text = needle("deadline")

        def moved_to_the_first_tool_response(prompt):
            prompt = prompt.replace(text, "")
            return prompt.replace('<|"|>}<tool_response|>', text + '<|"|>}<tool_response|>', 1)

        cases = {  # detokenized prompt -> (failed rendering units of 8, reason)
            "twice": (lambda prompt: prompt.replace(text, text + text), 8, "the notice occurs 2 time(s)"),
            "missing": (lambda prompt: prompt.replace(text, ""), 8, "the notice occurs 0 time(s)"),
            "dropped by the template": (lambda prompt: prompt.replace("\n\n" + text, ""), 8,
                                        "the notice occurs 0 time(s)"),
            "text after it": (lambda prompt: prompt.replace(text, text + "."), 8, "not the end of"),
            "an earlier tool response": (moved_to_the_first_tool_response, 8, ""),
            "the format-error turn renamed": (
                lambda prompt: prompt.replace("<|turn>user\nFormat error: ", "<|turn>user\nNote: "), 1,
                "is not the format-error message"),
            "a user turn after it": (lambda prompt: prompt + "<|turn>user\nSYNTHETIC<turn|>\n", 1,
                                     "a later user turn follows"),
        }
        for name, (change, failed, reason) in cases.items():
            with self.subTest(case=name):
                report = self.check(self.run_dir(), detokenize=lambda ids: change(codepoints(ids)))
                rendering = report["checks"]["rendering"]
                self.assertEqual((rendering["checked"], rendering["failed"]), (8, failed))
                self.assertEqual((report["failed_checks"], report["go"]), (["rendering"], False))
                self.assertTrue(all(reason in example["problems"][0] for example in rendering["examples"]))
                self.assertEqual(report["checks"]["request_bytes"]["failed"], 0)

        def broken(ids):
            raise IndexError("SYNTHETIC: id outside the vocabulary")
        report = self.check(self.run_dir(), detokenize=broken)
        self.assertEqual((report["checks"]["rendering"]["failed"], report["go"]), (8, False))
        self.assertIn("detokenizing failed: IndexError", report["checks"]["rendering"]["examples"][0]["problems"][0])

    def test_a_run_checked_under_the_other_policy_fails(self):
        report = self.check(self.run_dir("neutral"), "deadline")
        self.assertFalse(report["go"])
        self.assertTrue({"run_json_episode_config", "config_receipts", "notice_receipts", "rendering"}
                        <= set(report["failed_checks"]))


@unittest.skipUnless(POSIX, "cell ids (with '|') are directory names, as on Core2")
class CommandLineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.run_dir = write_run(self.base / "run", "deadline", STANDARD)
        self.loaded = []

    def tearDown(self):
        self.temporary.cleanup()

    def load(self, directory):
        self.loaded.append(directory)
        return codepoints, {"directory": directory, "SYNTHETIC": True}

    def main(self, *argv, load=None):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = D.main(list(argv), load=load or self.load)
        return code, stdout.getvalue(), stderr.getvalue()

    def argv(self, out, policy="deadline", run_dir=None):
        return ["--run-dir", str(run_dir or self.run_dir), "--policy", policy,
                "--tokenizer-dir", "/SYNTHETIC/tokenizer", "--out", str(out)]

    def test_go_and_no_go(self):
        code, stdout, _ = self.main(*self.argv(self.base / "go.json"))
        report = json.loads((self.base / "go.json").read_bytes())
        self.assertEqual((code, report["go"], report["tokenizer"]), (0, True, {"directory": "/SYNTHETIC/tokenizer",
                                                                              "SYNTHETIC": True}))
        self.assertTrue(stdout.startswith(f"GO: deadline run {RUN_ID}: 8/8 noticed final responses are valid tool "
                                          "calls (100.0%; go rule >= 90%: met); notices on a format-error message: 1; "
                                          "failed checks: none"), stdout)
        self.assertEqual(report["script"]["sha256"], hashlib.sha256(
            (ROOT / "scripts" / "expA_finalturn_devcheck.py").read_bytes()).hexdigest())
        self.assertEqual(report["run"]["run_json_sha256"],
                         hashlib.sha256((self.run_dir / "run.json").read_bytes()).hexdigest())
        code, stdout, _ = self.main(*self.argv(self.base / "no-go.json", policy="neutral"))
        self.assertEqual((code, json.loads((self.base / "no-go.json").read_bytes())["go"]), (1, False))
        self.assertTrue(stdout.startswith("NO-GO: neutral run"), stdout)

    def test_nothing_is_checked_or_written_when_the_run_cannot_be_read(self):
        out = self.base / "existing.json"
        out.write_text("SYNTHETIC earlier report\n")
        code, _, stderr = self.main(*self.argv(out))
        self.assertEqual((code, out.read_text()), (2, "SYNTHETIC earlier report\n"))
        self.assertIn("refusing to overwrite", stderr)
        for name, run_dir in (("no run directory", self.base / "absent"),
                              ("no run.json", write_run(self.base / "broken", "deadline", {}))):
            with self.subTest(case=name):
                if name == "no run.json":
                    (run_dir / "run.json").unlink()
                code, _, stderr = self.main(*self.argv(self.base / "report.json", run_dir=run_dir))
                self.assertEqual(code, 2)
                self.assertIn("ERROR", stderr)
                self.assertFalse((self.base / "report.json").exists())
        bad = write_run(self.base / "bad-json", "deadline", {P11[0]: [["submit"]]})
        (attempt_dir(bad, P11[0]) / "result.json").write_bytes(b'{"cell_id": 1, "cell_id": 2}')
        code, _, stderr = self.main(*self.argv(self.base / "report.json", run_dir=bad))
        self.assertEqual(code, 2)
        self.assertIn("duplicate JSON key 'cell_id'", stderr)

    def test_the_tokenizer_is_required(self):
        def unavailable(directory):
            raise D.InputError("SYNTHETIC: no tokenizer")
        code, _, stderr = self.main(*self.argv(self.base / "report.json"), load=unavailable)
        self.assertEqual((code, (self.base / "report.json").exists()), (2, False))
        self.assertIn("SYNTHETIC: no tokenizer", stderr)
        with self.assertRaisesRegex(D.InputError, "is not a directory"):
            D.load_detokenizer(self.base / "absent")
        with mock.patch.dict(sys.modules, {"transformers": None}), \
                self.assertRaisesRegex(D.InputError, "detokenizing needs transformers"):
            D.load_detokenizer(self.base)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            D.main(self.argv(self.base / "report.json", policy="none"), load=self.load)
        self.assertEqual(raised.exception.code, 2)
        self.assertEqual(self.loaded, [])


# ---------------------------------------------------------------------------------------------------------------
# (3) runs written by the real runner and harness (fake server, scripted responses)
# ---------------------------------------------------------------------------------------------------------------

SEARCH = {"calls": [("source_reference_search", {"query": "pkg/core.py::target"})]}
SUBMIT = {"calls": [("submit_answer", SUBMISSION)]}
# Consumed in dispatch order (one endpoint, concurrency 1), then SEARCH: a submission at response 1; a format error at
# response 7 then a search at 8; a submission at 8; seven episodes of eight searches (NoAnswer).
SCRIPT = [SUBMIT] + [SEARCH] * 6 + [{"content": "SYNTHETIC: I am done."}, SEARCH] + [SEARCH] * 7 + [SUBMIT]


@unittest.skipUnless(POSIX, "the runner reads checkouts with POSIX no-follow reads")
class RunnerRecordsTests(unittest.TestCase):
    maxDiff = None

    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.base = Path(cls.temporary.name)
        runner, dispatch, manifest = G.runner_fixture(cls.base)
        cls.runs = {}
        for name, extra in (("neutral", ["--final-turn-policy", "neutral"]),
                            ("deadline", ["--final-turn-policy", "deadline"]), ("default", [])):
            server = H.FakeServer(SCRIPT, default=SEARCH)
            try:
                argv = G.live_args(cls.base, dispatch, manifest, server, run=name) + extra
                argv[argv.index("--limit") + 1] = "10"
                with mock.patch.object(G.tools, "ReadOnlyBash", G.FakeJailBash):
                    code, _ = G.invoke_runner(runner, argv)
                # The SYNTHETIC served stack: the fake server's prompt ids -> the stand-in rendering of what it got.
                prompts = {tuple(server.prompt_ids(body)): render(body["messages"]) for body in server.chat_requests()}
            finally:
                server.close()
            cls.runs[name] = (code, cls.base / name, prompts)

    @classmethod
    def tearDownClass(cls):
        for path in cls.base.rglob("*"):
            if path.is_dir() and not path.is_symlink():
                path.chmod(0o755)
        cls.temporary.cleanup()

    def check(self, name, policy):
        code, run_dir, prompts = self.runs[name]
        self.assertEqual(code, 0)
        return D.check_run(run_dir, policy, lambda ids: prompts[tuple(ids)])

    def test_the_release_records_pass(self):
        for policy in POLICIES:
            with self.subTest(policy=policy):
                report = self.check(policy, policy)
                self.assertEqual((report["go"], report["failed_checks"]), (True, []))
                self.assertEqual((report["go_rule"]["noticed_final_responses"], report["go_rule"]["valid_tool_calls"]),
                                 (9, 9))
                self.assertEqual(report["notice"]["appended_to"], {"tool_result": 8, "format_error": 1})
                self.assertEqual(report["cells"]["terminal_exits"], {"NoAnswer": 8, "Submitted": 2})
                self.assertEqual(report["cells"]["states"]["completed"], 10)
                self.assertEqual(report["checks"]["rendering"]["checked"], 9)
                # The script's receipt fields are the harness's.
                result = json.loads(next((self.runs[policy][1] / "cells").rglob("result.json")).read_bytes())
                self.assertEqual(set(result["exposure"]["final_turn_notice"]), set(D.RECEIPT_FIELDS))
                # The terminal attempts are the scorer's.
                attempts, _ = D.load_attempts(self.runs[policy][1])
                scored = score.scan_run(self.runs[policy][1])
                self.assertEqual({cell: D.resolve(attempts[cell])[1]["attempt_id"] for cell in attempts},
                                 {cell: score.resolve_cell(scored[cell])["terminal"]["attempt_id"] for cell in scored})

    def test_the_wrong_policy_and_the_default_harness_fail(self):
        report = self.check("neutral", "deadline")
        self.assertFalse(report["go"])
        self.assertTrue({"run_json_episode_config", "config_receipts", "notice_receipts"}
                        <= set(report["failed_checks"]))
        report = self.check("default", "deadline")
        self.assertFalse(report["go"])
        self.assertTrue({"run_json_episode_config", "config_receipts", "notice_receipts", "delivered_iff_exchange"}
                        <= set(report["failed_checks"]))

    def test_the_command_line_on_release_records(self):
        _, run_dir, prompts = self.runs["deadline"]
        out = Path(self.temporary.name) / "deadline-devcheck.json"
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = D.main(["--run-dir", str(run_dir), "--policy", "deadline", "--tokenizer-dir", "/SYNTHETIC",
                           "--out", str(out)], load=lambda directory: (lambda ids: prompts[tuple(ids)], None))
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out.read_bytes())["run"]["run_id"], "deadline")
        self.assertIn("9/9 noticed final responses are valid tool calls", stdout.getvalue())


# ---------------------------------------------------------------------------------------------------------------
# (4) the constants
# ---------------------------------------------------------------------------------------------------------------

class ConstantsTests(unittest.TestCase):
    def test_the_texts_are_the_frozen_protocols(self):
        if not (__import__("pathlib").Path(__file__).resolve().parents[1] / "protocols").is_dir():
            self.skipTest("the protocol texts ship in the dataset package (protocol/)")
        protocol = (ROOT / D.PROTOCOL).read_text(encoding="utf-8")
        self.assertIn('Format: `"\\n\\n[harness notice] " + TEXT`.', protocol)
        for policy in POLICIES:          # the A1 policies; deadline_info and submit_only are A4's
            self.assertIn(f"| `{policy}` | `{D.TEXTS[policy]}` |", protocol)
            self.assertEqual(D.PROTOCOLS[policy], D.PROTOCOL)
        self.assertEqual((D.A4_PROTOCOL, D.PROTOCOLS["deadline_info"], D.PROTOCOLS["submit_only"]),
                         ("protocols/EXPA_A4_DIRECTIVE.md",) * 3)
        self.assertEqual(D.TEXTS["deadline"], D.TEXTS["deadline_info"] + " " + D.TEXTS["submit_only"])
        self.assertIn("**seed 11**", protocol)
        self.assertEqual((D.PREFIX, D.MARKER, D.FINAL_CALL, D.BLOCK, D.SEED, D.GO_RULE),
                         ("\n\n[harness notice] ", "[harness notice]", 7, "P", 11, (9, 10)))
        self.assertIn("at least 90% of final responses that received a notice are valid tool calls", protocol)

    @unittest.skipUnless(POSIX, "Linux only: the harness imports serve.py (pwd)")
    def test_the_records_are_the_harness_and_the_scorer(self):
        self.assertEqual(D.TEXTS, harness.FINAL_TURN_TEXTS)
        self.assertEqual((D.PREFIX, D.PROTOCOL), (harness.FINAL_TURN_PREFIX, harness.FINAL_TURN_PROTOCOL))
        self.assertEqual(D.PROTOCOLS, harness.FINAL_TURN_PROTOCOLS)
        self.assertEqual(D.FROZEN_CONFIG, harness.EpisodeConfig().record())
        for policy in D.TEXTS:
            self.assertEqual(D.expected_episode_config(policy),
                             harness.EpisodeConfig(final_turn_policy=policy).record())
        self.assertTrue(harness.FORMAT_ERROR.startswith(D.FORMAT_ERROR_START))
        self.assertEqual((set(D.COMPLETED_EXITS), D.INFRA_RETRY_CAP, set(D.TERMINAL_STATES), D._ORDINAL.pattern),
                         (set(score.COMPLETED_EXITS), score.INFRA_RETRY_CAP, set(score.TERMINAL_STATES),
                          score._ORDINAL.pattern))


if __name__ == "__main__":
    unittest.main()
