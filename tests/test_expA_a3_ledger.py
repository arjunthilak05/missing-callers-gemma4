"""Exp A follow-up A3: the evidence ledger (protocols/EXPA_A3_LEDGER.md).

(1) The ledger rules on synthetic histories (any OS): the exact text of the protocol's example; the call-token rule
    (calls, method calls, definitions, longer identifiers); the sources (tool results only: never the user message
    and its answer block, the model's own text or format-error messages); path normalization; deduplication and
    first-appearance order; the 20-entry and 2,000-byte caps; excerpts; the empty ledger; and that nothing but the
    history and the target name enters.
(2) The config record and the A3 design (any OS): the default record and the A1 records are unchanged; off and on
    record the option and every frozen rule; the default design table is unchanged and the A3 table adds the LEDGER
    block first; LEDGER cells are valid episodes.
(3) The scorer (any OS for the layout; scoring an A3 run on POSIX, like tests/test_probe_score.py): a3-ledger is d11
    plus LEDGER, detected from run.json or cell ids.
(4) The harness against the fake server (POSIX): off sends exactly the A1 deadline requests and records the ledger;
    on adds the ledger before the deadline notice on the 8th request only; earlier submissions never build it; a
    format error at response 7; real source_reference_search hits.
(5) The runner (POSIX): registration, the LEDGER-only selection, dry run, run.json, every episode, relaunch refusal.
(6) The launch script (POSIX + bash): forwarded only when registered; refused before any unit otherwise.
The default path stays pinned, unedited, by tests/test_expA_default_golden.py, and A1 by tests/test_expA_finalturn.py.
"""
import collections
import contextlib
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from absent_edges.data import canonical_json  # noqa: E402
from absent_edges.probe import harness  # noqa: E402
from absent_edges.probe import score as S  # noqa: E402
from absent_edges.probe.tools import LEXICAL_KIND  # noqa: E402

POSIX = os.name == "posix"
if POSIX:
    import test_expA_default_golden as G      # puts the project on sys.path
    import test_probe_harness as H
    import test_probe_score as PS
    from absent_edges.models import LocalModel, ModelProfile
    from absent_edges.probe import ledger as L
    from absent_edges.probe import serve
    from absent_edges.probe.tools import ToolBox

PREFIX = "\n\n[harness notice] "
DEADLINE = PREFIX + "This is your last response. Call submit_answer now."
HEADER = "Candidate call sites seen in your earlier tool outputs (unverified; check before submitting):"
FROZEN_CONFIG = {"max_model_calls": 8, "context_budget": 16384, "max_response_tokens": 4096,
                 "min_response_tokens": 512, "temperature": 0.2, "max_consecutive_format_errors": 3}
TARGET = "pkg/core.py::target"
PROTOCOL = PROJECT / "protocols" / "EXPA_A3_LEDGER.md"
BASH = {"calls": [("bash", {"command": "ls"})]}
SEARCH = {"calls": [("source_reference_search", {"query": "pkg/core.py::target"})]}
EURO = "\u20ac" * 100
SUBMIT = {"calls": [("submit_answer", {"sites": [{"path": "pkg/core.py", "line": 6}], "complete": "yes"})]}
Run = collections.namedtuple("Run", "result directory server rows")


def sha(text):
    return hashlib.sha256(text if isinstance(text, bytes) else text.encode("utf-8")).hexdigest()


def bash_result(output, call_id="call-1-0"):
    content = canonical_json({"exit_code": 0, "output": output, "truncated": False}).decode()
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def search_result(hits, call_id="call-2-0"):
    content = canonical_json({"query": "target", "kind": LEXICAL_KIND, "truncated": False, "error": None,
                              "hits": [{"path": p, "line": n, "text": t} for p, n, t in hits]}).decode()
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def history(*tool_messages, user="Task: find every place ... `target` ..."):
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": user}]
    for message in tool_messages:
        messages.append({"role": "assistant", "content": "", "tool_calls": []})
        messages.append(message)
    return messages


def ledger(*tool_messages, target=TARGET, **kwargs):
    return harness.evidence_ledger(history(*tool_messages, **kwargs), target)


# The protocol's worked example (protocols/EXPA_A3_LEDGER.md section 2.3), byte for byte.
EXAMPLE_HISTORY = (
    bash_result("./pkg/core.py:1:def target(x):\n"
                "./pkg/core.py:6:    return target(1)\n"
                "pkg/util.py:14:        out = self.retarget(y)\n"
                "/repo/pkg/jobs.py:40:    results = [job.target(t) for t in items]\n"
                "tests/test_core.py:9:    assert target(2) == 2\n", "call-3-0"),
    search_result([("pkg/core.py", 6, "    return target(1)"),
                   ("pkg/hooks.py", 22, "        hook = target  # registered, not called here"),
                   ("pkg/hooks.py", 31, "        return registry['x'].target  (value)")], "call-4-0"))
EXAMPLE_TEXT = (HEADER + "\n"
                "- pkg/core.py:6: return target(1)\n"
                "- pkg/jobs.py:40: results = [job.target(t) for t in items]\n"
                "- tests/test_core.py:9: assert target(2) == 2\n"
                "- pkg/hooks.py:31: return registry['x'].target  (value)")


# ---------------------------------------------------------------------------------------------------------------
# (1) the ledger rules
# ---------------------------------------------------------------------------------------------------------------

class LedgerRuleTests(unittest.TestCase):
    maxDiff = None

    def test_the_frozen_rules(self):
        self.assertEqual(harness.LEDGER_MODES, ("none", "off", "on"))
        self.assertEqual(harness.LEDGER_DEFAULT, "none")
        self.assertEqual(harness.LEDGER_HEADER, HEADER)
        self.assertEqual(harness.LEDGER_EMPTY, "(none)")
        self.assertEqual(harness.LEDGER_ENTRY, "- {path}:{line}: {excerpt}")
        self.assertEqual((harness.LEDGER_MAX_ENTRIES, harness.LEDGER_MAX_BYTES, harness.LEDGER_EXCERPT_CHARS),
                         (20, 2000, 120))
        self.assertEqual(harness.LEDGER_PRINTED_LINE.pattern,
                         r"(?P<path>[A-Za-z0-9_.+@~/-]+\.py):(?P<line>[1-9][0-9]*):(?P<text>.*)")
        self.assertEqual(harness.LEDGER_CALL_TOKEN, r"(?<![A-Za-z0-9_]){name}[ \t]*\(")
        self.assertEqual(harness.LEDGER_DEFINITION.pattern, r"(?:^|[^A-Za-z0-9_])def[ \t]+$")
        self.assertEqual(harness.LEDGER_PROTOCOL, "protocols/EXPA_A3_LEDGER.md")
        # Not a prompt template: run.json's prompts (and their golden digest) stay those of the prereg harness.
        self.assertNotIn("Candidate call sites", json.dumps(harness.prompt_templates()))

    def test_the_protocol_states_the_same_rules_and_example(self):

        if not (__import__("pathlib").Path(__file__).resolve().parents[1] / "protocols").is_dir():

            self.skipTest("the protocol texts ship in the dataset package (protocol/)")
        text = PROTOCOL.read_text(encoding="utf-8")
        for needle in (HEADER, "`(none)`", "`- {path}:{line}: {excerpt}`", "20 entries", "2,000 bytes",
                       "120 characters", harness.LEDGER_PRINTED_LINE.pattern, harness.LEDGER_CALL_TOKEN,
                       harness.LEDGER_DEFINITION.pattern, "seeds 61, 73 and 89"):
            with self.subTest(needle=needle):
                self.assertIn(needle, text)
        self.assertIn("```text\n" + EXAMPLE_TEXT + "\n```", text)

    def test_the_example(self):
        built = harness.evidence_ledger(history(*EXAMPLE_HISTORY), TARGET)
        self.assertEqual(built["text"], EXAMPLE_TEXT)
        self.assertEqual(built["suffix"], PREFIX + EXAMPLE_TEXT)
        self.assertEqual((built["name"], built["candidates"], built["shown"]), ("target", 4, 4))
        self.assertEqual(built["body_bytes"], len(EXAMPLE_TEXT[len(HEADER) + 1:].encode()))
        self.assertEqual([entry["source"] for entry in built["entries"]],
                         [{"message_index": 3, "tool_call_id": "call-3-0", "kind": "bash_line"}] * 3
                         + [{"message_index": 5, "tool_call_id": "call-4-0", "kind": "search_hit"}])

    def test_the_call_token(self):
        calls = ["return target(1)", "x = obj.target(2)", "self.target (3)", "f = target\t(", "[target(i) for i in y]",
                 "def target(x): return target(x - 1)", "print(target())", "a=target(", "# target(x) in a comment"]
        others = ["def target(x):", "async def target(x):", "    def  target(self, x):", "retarget(1)",
                  "target_x(1)", "targets(1)", "_target(1)", "x = target", "from pkg import target",
                  "callback=target, x=1", "target[0](1)", "Target(1)"]
        for line in calls + others:
            with self.subTest(line=line):
                built = ledger(bash_result(f"pkg/a.py:3:{line}\n"))
                self.assertEqual(built["shown"], int(line in calls))

    def test_a_method_target_uses_its_last_name(self):
        output = "pkg/a.py:3:        self.method(x)\npkg/a.py:4:    Box.method\npkg/a.py:5:    box.method_two()\n"
        built = ledger(bash_result(output), target="pkg/core.py::Box.method")
        self.assertEqual((built["name"], [e["line"] for e in built["entries"]]), ("method", [3]))
        nested = ledger(bash_result("pkg/a.py:7:    inner(1)\n"), target="pkg/other.py::outer.<locals>.inner")
        self.assertEqual((nested["name"], nested["shown"]), ("inner", 1))

    def test_only_tool_results_are_read(self):
        answer = "pkg/core.py:6: return target(1)\npkg/other.py:5:    return target(3)"
        messages = history(bash_result("ok\n"), user="Task ... answer block:\n" + answer)
        messages.insert(2, {"role": "assistant", "content": "pkg/x.py:1:target(1)"})
        messages.append({"role": "assistant", "content": "pkg/y.py:2:target(2)"})
        messages.append({"role": "user", "content": "Format error: pkg/z.py:3:target(3)"})
        built = harness.evidence_ledger(messages, TARGET)
        self.assertEqual((built["candidates"], built["text"]), (0, HEADER + "\n(none)"))
        # Results the ledger does not read: errors, graph answers, other shapes, text that is not JSON.
        for content in ('{"error":"pkg/a.py:3:target(1)"}',
                        '{"matches":[{"path":"pkg/a.py","line":3,"text":"target(1)"}]}',
                        'pkg/a.py:3:target(1)', '["pkg/a.py:3:target(1)"]',
                        '{"hits":[{"path":"pkg/a.py","line":3,"text":"target(1)"}],"kind":"other"}',
                        '{"exit_code":0,"output":["pkg/a.py:3:target(1)"]}'):
            with self.subTest(content=content):
                built = harness.evidence_ledger(history({"role": "tool", "tool_call_id": "c", "content": content}),
                                                TARGET)
                self.assertEqual(built["candidates"], 0)

    def test_printed_lines_and_paths(self):
        cases = {"./pkg/a.py:3:target(1)": ("pkg/a.py", 3), "/repo/pkg/a.py:4:target(1)": ("pkg/a.py", 4),
                 "././pkg/a.py:5:target(1)": ("pkg/a.py", 5),
                 "pkg/sub-dir/a_b.py:6:target(1)": ("pkg/sub-dir/a_b.py", 6),
                 "a.py:7:target(1)\r": ("a.py", 7), "pkg/a.py:8:12:target(1)": ("pkg/a.py", 8)}
        skipped = ["/usr/lib/python3/a.py:3:target(1)", "../a.py:3:target(1)", "pkg/../a.py:3:target(1)",
                   "pkg/a.txt:3:target(1)", "pkg/a.pyc:3:target(1)", "pkg/a.py-3-target(1)", "  pkg/a.py:3:target(1)",
                   "pkg/a.py:03:target(1)", "pkg/a.py:0:target(1)", "pkg/a.py:3 target(1)", "3:target(1)",
                   "pkg/my file.py:3:target(1)"]
        for line, expected in cases.items():
            with self.subTest(line=line):
                entries = ledger(bash_result(line + "\n"))["entries"]
                self.assertEqual([(e["path"], e["line"]) for e in entries], [expected])
        for line in skipped:
            with self.subTest(line=line):
                self.assertEqual(ledger(bash_result(line + "\n"))["candidates"], 0)
        # Search hits are read as printed (path, line, text); the same normalization applies.
        hits = search_result([("./pkg/b.py", 2, "target(1)"), ("/etc/b.py", 3, "target(1)"), ("pkg/b.py", 4, "x")])
        self.assertEqual([(e["path"], e["line"]) for e in ledger(hits)["entries"]], [("pkg/b.py", 2)])

    def test_deduplication_and_first_appearance(self):
        first = bash_result("pkg/b.py:9:  target(9)\npkg/a.py:1:  target(1)\npkg/b.py:9:  again target(9)\n", "c1")
        second = search_result([("pkg/a.py", 1, "other text target(1)"), ("pkg/c.py", 2, "target(2)")], "c2")
        third = bash_result("./pkg/c.py:2:changed target(2)\npkg/a.py:5:target(5)\n", "c3")
        built = ledger(first, second, third)
        self.assertEqual([(e["path"], e["line"], e["excerpt"]) for e in built["entries"]],
                         [("pkg/b.py", 9, "target(9)"), ("pkg/a.py", 1, "target(1)"), ("pkg/c.py", 2, "target(2)"),
                          ("pkg/a.py", 5, "target(5)")])
        self.assertEqual(built["candidates"], 4)

    def test_the_entry_cap(self):
        output = "".join(f"pkg/m.py:{n}:target({n})\n" for n in range(1, 26))
        built = ledger(bash_result(output))
        self.assertEqual((built["candidates"], built["shown"]), (25, 20))
        self.assertEqual([e["line"] for e in built["entries"]], list(range(1, 21)))
        self.assertEqual(built["text"].count("\n- "), 20)

    def test_the_byte_cap(self):
        long = "x = target(" + "a" * 200
        output = "".join(f"pkg/long_module_name_{n:02d}.py:{n}:{long}\n" for n in range(1, 20))
        built = ledger(bash_result(output))
        lines = built["text"].split("\n")[1:]
        self.assertEqual(len(lines), built["shown"])
        self.assertLessEqual(built["body_bytes"], 2000)
        self.assertEqual(built["body_bytes"], len("\n".join(lines).encode()))
        next_line = harness.LEDGER_ENTRY.format(path=f"pkg/long_module_name_{built['shown'] + 1:02d}.py",
                                                line=built["shown"] + 1, excerpt=long[:120])
        self.assertGreater(len("\n".join(lines + [next_line]).encode()), 2000)
        self.assertEqual((built["candidates"], built["shown"]), (19, 13))
        # Bytes, not characters: three-byte characters fill the cap sooner.
        wide = ledger(bash_result("".join(f"pkg/m.py:{n}:target('{EURO}')\n" for n in range(1, 20))))
        self.assertLessEqual(wide["body_bytes"], 2000)
        self.assertLess(wide["shown"], 13)

    def test_excerpts(self):
        text = "\t  value = target(1)\x0b\x1b[0m " + "y" * 200 + "  "
        excerpt = ledger(bash_result(f"pkg/a.py:3:{text}\n"))["entries"][0]["excerpt"]
        self.assertEqual(excerpt, ("value = target(1)  [0m " + "y" * 200)[:120])
        self.assertEqual(len(excerpt), 120)
        # A lone surrogate (never produced by the tools) cannot break the ledger: U+FFFD in TEXT, skipped in a path.
        content = json.dumps({"kind": LEXICAL_KIND,
                              "hits": [{"path": "pkg/a.py", "line": 1, "text": "target(\ud800)"},
                                       {"path": "pkg/\udc80.py", "line": 2, "text": "target(1)"}]})
        built = harness.evidence_ledger(history({"role": "tool", "tool_call_id": "c", "content": content}), TARGET)
        self.assertEqual([(e["path"], e["excerpt"]) for e in built["entries"]], [("pkg/a.py", "target(\ufffd)")])
        built["text"].encode("utf-8")
        unicode_line ="r\u00e9sultat = target('\u00e9t\u00e9')"
        self.assertEqual(ledger(bash_result(f"pkg/a.py:3:{unicode_line}\n"))["entries"][0]["excerpt"], unicode_line)

    def test_nothing_but_the_history_and_the_target_enters(self):
        self.assertEqual(list(inspect.signature(harness.evidence_ledger).parameters), ["messages", "target"])
        messages = history(*EXAMPLE_HISTORY)
        frozen = json.dumps(messages)
        harness.evidence_ledger(messages, TARGET)
        self.assertEqual(json.dumps(messages), frozen)     # read only
        # The same history and target always give the same bytes.
        self.assertEqual(harness.evidence_ledger(messages, TARGET), harness.evidence_ledger(json.loads(frozen), TARGET))


# ---------------------------------------------------------------------------------------------------------------
# (2) the config record and the A3 design
# ---------------------------------------------------------------------------------------------------------------

class LedgerConfigTests(unittest.TestCase):
    maxDiff = None

    def test_default_and_a1_records_are_unchanged(self):
        default = harness.EpisodeConfig()
        self.assertEqual(default.evidence_ledger, "none")
        self.assertEqual(default.record(), FROZEN_CONFIG)
        self.assertEqual(list(default.record()), list(FROZEN_CONFIG))
        self.assertEqual(harness.EpisodeConfig(evidence_ledger="none"), default)
        self.assertEqual(harness.operational_appendix()["episode_config"], FROZEN_CONFIG)
        for policy in ("neutral", "deadline"):
            record = harness.EpisodeConfig(final_turn_policy=policy).record()
            self.assertEqual(list(record), list(FROZEN_CONFIG) + ["final_turn_policy", "final_turn_notice"])
        self.assertIsNone(harness.config_design(default))
        self.assertIsNone(harness.config_design(harness.EpisodeConfig(final_turn_policy="deadline")))

    def test_an_arm_records_the_option_and_every_rule(self):
        for mode in ("off", "on"):
            with self.subTest(mode=mode):
                config = harness.EpisodeConfig(final_turn_policy="deadline", evidence_ledger=mode)
                record = config.record()
                self.assertEqual(list(record), list(FROZEN_CONFIG) + ["final_turn_policy", "final_turn_notice",
                                                                      "evidence_ledger", "evidence_ledger_rules"])
                self.assertEqual({key: record[key] for key in list(record)[:8]},
                                 harness.EpisodeConfig(final_turn_policy="deadline").record())
                rules = record["evidence_ledger_rules"]
                self.assertEqual((record["evidence_ledger"], rules["mode"], rules["delivered_to_model"]),
                                 (mode, mode, mode == "on"))
                self.assertEqual({key: rules[key] for key in ("protocol", "channel", "prefix", "header", "empty",
                                                               "entry", "max_entries", "max_bytes", "excerpt_chars",
                                                               "printed_line", "call_token", "definition",
                                                               "call_index")},
                                 {"protocol": "protocols/EXPA_A3_LEDGER.md", "channel": "in_band", "prefix": PREFIX,
                                  "header": HEADER, "empty": "(none)", "entry": "- {path}:{line}: {excerpt}",
                                  "max_entries": 20, "max_bytes": 2000, "excerpt_chars": 120,
                                  "printed_line": harness.LEDGER_PRINTED_LINE.pattern,
                                  "call_token": harness.LEDGER_CALL_TOKEN,
                                  "definition": harness.LEDGER_DEFINITION.pattern, "call_index": 7})
                self.assertEqual(harness.config_design(config), "a3-ledger")
                json.dumps(record)

    def test_invalid_options_are_refused(self):
        for bad in ("On", "", None, 1, "deadline"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                harness.EpisodeConfig(final_turn_policy="deadline", evidence_ledger=bad)
        for mode in ("off", "on"):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "needs a final_turn_policy"):
                harness.EpisodeConfig(evidence_ledger=mode)
        with self.assertRaises(ValueError):
            harness.evidence_ledger_record("none", 8)


class LedgerDesignTests(unittest.TestCase):
    maxDiff = None

    def test_the_default_design_is_unchanged(self):
        self.assertEqual(harness.BLOCK_ORDER, ("P", "S-VC", "S-NAT", "S-EDGE", "T"))
        self.assertNotIn("LEDGER", harness.BLOCK_DESIGN)
        self.assertEqual(harness.design_table_for(None), harness.design_table())
        self.assertNotIn("name", harness.design_table())
        self.assertNotIn("LEDGER", json.dumps(harness.design_table()))
        self.assertEqual(harness.design_order(), harness.BLOCK_ORDER)
        with self.assertRaises(ValueError):
            harness.design_order("a3")

    def test_the_a3_design(self):
        self.assertEqual((harness.LEDGER_BLOCK, harness.LEDGER_SEEDS, harness.LEDGER_DESIGN),
                         ("LEDGER", (61, 73, 89), "a3-ledger"))
        self.assertEqual(harness.design_order("a3-ledger"), ("LEDGER",) + harness.BLOCK_ORDER)
        table = harness.design_table_for("a3-ledger")
        self.assertEqual((table["name"], table["block_order"]), ("a3-ledger", ["LEDGER", "P", "S-VC", "S-NAT",
                                                                               "S-EDGE", "T"]))
        self.assertEqual(table["blocks"]["LEDGER"]["cells"], [["vcomplete", "C1"], ["omit_node", "C1"]])
        self.assertEqual(table["blocks"]["LEDGER"]["seeds"], [61, 73, 89])
        default = harness.design_table()
        self.assertEqual({block: value for block, value in table["blocks"].items() if block != "LEDGER"},
                         default["blocks"])
        self.assertEqual((table["order_salt"], table["cell_id_format"]), (default["order_salt"],
                                                                          default["cell_id_format"]))
        self.assertTrue(harness.block_cell("LEDGER", "vcomplete", "C1"))
        self.assertTrue(harness.block_cell("LEDGER", "omit_node", "C1"))
        self.assertFalse(harness.block_cell("LEDGER", "omit_node", "C3"))
        self.assertFalse(harness.block_cell("P", "vcomplete", "C1"))
        self.assertTrue(harness.block_cell("P", "omit_node", "C1"))

    def test_ledger_cells_are_valid_episodes(self):
        payload = canonical_json({"query": {"symbol": TARGET, "relation": "invokes", "direction": "incoming"},
                                  "matches": []})
        base = dict(run_id="r", token=1, item_id="item-h", family="fam", target=TARGET, target_line=1,
                    package_dirs=("pkg",), condition="C1", seed=61, wording_pair="A", payload=payload,
                    payload_sha256=sha(payload))
        for variant in ("vcomplete", "omit_node"):
            cell = f"LEDGER|item-h|{variant}|C1|s61"
            harness.EpisodeInput(cell_id=cell, attempt_id=cell + ".a001", block="LEDGER", variant=variant, **base)
        with self.assertRaisesRegex(ValueError, "block LEDGER has no cell"):
            harness.EpisodeInput(cell_id="x", attempt_id="x.a001", block="LEDGER", variant="natural", **base)
        with self.assertRaisesRegex(ValueError, "block must be one of P, S-VC, S-NAT, S-EDGE, T, LEDGER"):
            harness.EpisodeInput(cell_id="x", attempt_id="x.a001", block="L3", variant="omit_node", **base)


# ---------------------------------------------------------------------------------------------------------------
# (3) the scorer
# ---------------------------------------------------------------------------------------------------------------

class LedgerScoreLayoutTests(unittest.TestCase):
    def test_a3_ledger_is_d11_plus_ledger(self):
        self.assertEqual(sorted(S.DESIGNS), ["a3-ledger", "contract-v2", "d11"])
        self.assertEqual(S.DEFAULT_DESIGN, "d11")
        a3 = S.DESIGNS["a3-ledger"]
        self.assertEqual(a3[1:], S.DESIGNS["d11"])
        self.assertEqual((a3[0].name, a3[0].arms, a3[0].seeds, a3[0].membership),
                         ("LEDGER", (("vcomplete", "C1"), ("omit_node", "C1")), (61, 73, 89), None))
        self.assertEqual(S.parse_cell_id("LEDGER|item-a|omit_node|C1|s61")["block"], "LEDGER")

    def test_detection(self):
        a3 = {"design": harness.design_table_for("a3-ledger")}
        d11 = {"design": harness.design_table()}
        self.assertEqual(S.detect_design(a3)[0], "a3-ledger")
        self.assertEqual(S.detect_design(a3, ["LEDGER|i|vcomplete|C1|s61", "P|i|none|C0|s11"])[0], "a3-ledger")
        self.assertEqual(S.detect_design(d11, ["P|i|none|C0|s11"])[0], "d11")
        self.assertEqual(S.detect_design(None, ["LEDGER|i|vcomplete|C1|s61", "P|i|none|C0|s11"])[0], "a3-ledger")
        with self.assertRaises(S.BindingError):
            S.detect_design(d11, ["LEDGER|i|vcomplete|C1|s61"])
        with self.assertRaises(S.BindingError):
            S.detect_design(a3, requested="d11")


@unittest.skipUnless(POSIX, "attempt directories are named by cell id (contains '|')")
class LedgerScoreRunTests(unittest.TestCase):
    def test_an_a3_run_is_scored_under_its_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = PS.Fixture(Path(directory), run_json_extra={"design": harness.design_table_for("a3-ledger")})
            expected = fixture.records["item-a"]["variants"]["vcomplete"]["cells"]["C1"]
            # A result carrying the A1 and A3 receipts in its exposure (as the harness writes them) is bound as usual.
            exposure = {"payload_present": True, "payload_sha256": expected, "expected_payload_sha256": expected,
                        "payload_tokens": 400, "verification_block_delivered": False, "verification_sha256": None,
                        "verification_hits": None,
                        "tools_offered": ["bash", "source_reference_search", "submit_answer"],
                        "recommendation_present": False, "final_turn_notice": {"policy": "deadline"},
                        "evidence_ledger": {"mode": "on"}}
            fixture.result("LEDGER|item-a|vcomplete|C1|s61", 1, "Submitted", [{"path": "pkg/other.py", "line": 5}],
                           "yes", exposure=exposure)
            fixture.result("LEDGER|item-a|omit_node|C1|s61", 1, "NoAnswer")
            rows, audit = fixture.score()
            self.assertEqual(audit["design"], "a3-ledger")
            by = {row["cell_id"]: row for row in rows}
            self.assertEqual(len(rows), 41 + 2 * 2 * 3)        # d11 for these 2 items, plus LEDGER 2 x 2 x 3
            self.assertEqual([row["cell_id"] for row in rows[:6]],
                             [f"LEDGER|item-a|{v}|C1|s{s}" for v in ("vcomplete", "omit_node") for s in (61, 73, 89)])
            self.assertEqual((by["LEDGER|item-a|vcomplete|C1|s61"]["site_recovered"],
                              by["LEDGER|item-a|omit_node|C1|s61"]["site_recovered"],
                              by["LEDGER|item-a|omit_node|C1|s61"]["cell_state"],
                              by["LEDGER|item-b|omit_node|C1|s89"]["cell_state"]),
                             (1, 0, "completed", "not_dispatched"))


# ---------------------------------------------------------------------------------------------------------------
# (4) the harness: synthetic transcripts against the fake server
# ---------------------------------------------------------------------------------------------------------------

GREP = ("pkg/core.py:1:def target(x):\n"
        "pkg/core.py:6:    return target(1)\n"
        "pkg/other.py:1:from pkg.core import target as aliased\n")
GREP_LEDGER = HEADER + "\n- pkg/core.py:6: return target(1)"


@unittest.skipUnless(POSIX, "the ToolBox search scans with POSIX no-follow reads")
class LedgerEpisodeTests(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = H.make_repo(Path(self.temporary.name) / "repo")
        self.servers, self.count = [], 0

    def tearDown(self):
        for server in self.servers:
            with contextlib.suppress(Exception):
                server.close()
        self.temporary.cleanup()

    def run_arm(self, script, mode, *, policy="deadline", output=GREP, variant="omit_node"):
        server = H.FakeServer(script)
        self.servers.append(server)
        self.count += 1
        directory = Path(self.temporary.name) / f"attempt-{self.count}"
        directory.mkdir()
        client = LocalModel(ModelProfile(endpoint=server.endpoint), timeout=20)
        toolbox = ToolBox(self.root, ["pkg"], H.FakeBash(output))
        cell = f"LEDGER|item-h|{variant}|C1|s61"
        episode = H.make_episode("C1", block="LEDGER", variant=variant, cell_id=cell, attempt_id=cell + ".a001",
                                 seed=61)
        config = harness.EpisodeConfig(final_turn_policy=policy, evidence_ledger=mode)
        result, digest = harness.run_episode(episode, directory, client=client, toolbox=toolbox, config=config)
        self.assertEqual(digest, L.sha256_path(directory / "result.json"))
        return Run(result, directory, server, H.events(directory))

    @staticmethod
    def values(run, kind):
        return [row["value"] for row in run.rows if row["kind"] == kind]

    @staticmethod
    def digests(run):
        return [row["value"]["request_sha256"] for row in run.rows if row["kind"] in ("model_exchange",
                                                                                     "context_accounting")]

    def receipt(self, mode, **values):
        base = {"mode": mode, "call_index": 7, "computed": False, "appended": False, "sent": False,
                "delivered": False, "appended_to": None, "tool_call_id": None, "candidates": None, "entries": None,
                "body_bytes": None, "text_sha256": None, "suffix_sha256": None}
        assert set(values) <= set(base)
        return dict(base, **values)

    def built(self, mode, *, text=GREP_LEDGER, candidates=1, entries=1, appended_to="tool_result",
              tool_call_id="call-7-0"):
        on = mode == "on"
        return self.receipt(mode, computed=True, appended=on, sent=on, delivered=on, appended_to=appended_to,
                            tool_call_id=tool_call_id, candidates=candidates, entries=entries,
                            body_bytes=len(text[len(HEADER) + 1:].encode()) if entries else 0,
                            text_sha256=sha(text), suffix_sha256=sha(PREFIX + text))

    def assertReceipt(self, run, expected):
        self.assertEqual(run.result["exposure"]["evidence_ledger"], expected)
        self.assertEqual(run.result["accounting"]["evidence_ledger"], expected)
        self.assertEqual(self.values(run, "episode_finished")[0]["accounting"]["evidence_ledger"], expected)

    def test_off_sends_the_a1_requests_and_on_adds_the_ledger_before_the_notice(self):
        for variant in ("vcomplete", "omit_node"):
            a1 = self.run_arm([BASH] * 8, "none", variant=variant)
            off = self.run_arm([BASH] * 8, "off", variant=variant)
            on = self.run_arm([BASH] * 8, "on", variant=variant)
            with self.subTest(variant=variant):
                self.assertNotIn("evidence_ledger", a1.result["exposure"])
                self.assertNotIn("evidence_ledger", [row["kind"] for row in a1.rows])
                a1_chats, off_chats, on_chats = (run.server.chat_requests() for run in (a1, off, on))
                # off: every request is the A1 deadline request, byte for byte.
                self.assertEqual(self.digests(off), self.digests(a1))
                self.assertEqual(off_chats, a1_chats)
                # on: responses 1-7 identical; the 8th request differs by the ledger suffix before the notice.
                self.assertEqual(self.digests(on)[:14], self.digests(a1)[:14])
                last_a1, last_on = a1_chats[7]["messages"][-1], on_chats[7]["messages"][-1]
                tool_output = last_a1["content"][:-len(DEADLINE)]
                self.assertEqual(last_on, dict(last_a1, content=tool_output + PREFIX + GREP_LEDGER + DEADLINE))
                self.assertEqual(on_chats[7]["messages"][:-1], a1_chats[7]["messages"][:-1])
                self.assertEqual(json.dumps(on_chats).count("Candidate call sites"), 1)
                self.assertNotIn("Candidate call sites", json.dumps(off_chats))
                self.assertTrue(last_on["content"].endswith(DEADLINE))      # the deadline notice stays last
                for run, mode in ((off, "off"), (on, "on")):
                    self.assertEqual(run.result["exit_status"], "NoAnswer")
                    self.assertReceipt(run, self.built(mode))
                    kinds = [row["kind"] for row in run.rows]
                    index = kinds.index("evidence_ledger")
                    self.assertEqual((kinds.count("evidence_ledger"), kinds[index + 1], kinds[index + 2]),
                                     (1, "final_turn_notice", "context_accounting"))
                    event = self.values(run, "evidence_ledger")[0]
                    content_on = tool_output + PREFIX + GREP_LEDGER
                    self.assertEqual(event, {
                        "call_index": 7, "mode": mode, "appended": mode == "on", "appended_to": "tool_result",
                        "tool_call_id": "call-7-0", "name": "target", "text": GREP_LEDGER,
                        "text_sha256": sha(GREP_LEDGER), "suffix": PREFIX + GREP_LEDGER,
                        "suffix_sha256": sha(PREFIX + GREP_LEDGER), "candidates": 1,
                        "entries": [{"path": "pkg/core.py", "line": 6, "excerpt": "return target(1)",
                                     "source": {"message_index": 3, "tool_call_id": "call-1-0", "kind": "bash_line"}}],
                        "body_bytes": len("- pkg/core.py:6: return target(1)"),
                        "previous_content_sha256": sha(tool_output),
                        "content_sha256": sha(content_on if mode == "on" else tool_output)})
                    notice = self.values(run, "final_turn_notice")[0]
                    self.assertEqual(notice["previous_content_sha256"],
                                     sha(content_on if mode == "on" else tool_output))
                    self.assertEqual(run.result["exposure"]["final_turn_notice"]["delivered"], True)
                    record = harness.EpisodeConfig(final_turn_policy="deadline", evidence_ledger=mode).record()
                    self.assertEqual((run.result["config"], self.values(run, "episode_setup")[0]["config"]),
                                     (record, record))
                    exposure, a1_exposure = dict(run.result["exposure"]), dict(a1.result["exposure"])
                    exposure.pop("evidence_ledger")
                    if mode == "on":      # the notice's prompt_tokens and max_tokens count the ledger too
                        exposure.pop("final_turn_notice")
                        a1_exposure.pop("final_turn_notice")
                    self.assertEqual(exposure, a1_exposure)
                # The ledger counts in the 8th preflight under on only.
                counts = [[v["count"] for v in self.values(run, "context_accounting")] for run in (a1, off, on)]
                self.assertEqual(counts[1], counts[0])
                self.assertEqual(counts[2][:7], counts[0][:7])
                self.assertGreater(counts[2][7], counts[0][7])

    def test_an_episode_that_submits_early_never_builds_it(self):
        for responses in (1, 4, 7):
            script = [BASH] * (responses - 1) + [SUBMIT]
            a1 = self.run_arm(script, "none")
            for mode in ("off", "on"):
                with self.subTest(responses=responses, mode=mode):
                    run = self.run_arm(script, mode)
                    self.assertEqual((run.result["exit_status"], run.result["calls"]), ("Submitted", responses))
                    self.assertEqual(self.digests(run), self.digests(a1))
                    self.assertNotIn("evidence_ledger", [row["kind"] for row in run.rows])
                    self.assertReceipt(run, self.receipt(mode))

    def test_a_submission_in_the_last_response_saw_the_ledger(self):
        run = self.run_arm([BASH] * 7 + [SUBMIT], "on")
        self.assertEqual((run.result["exit_status"], run.result["calls"]), ("Submitted", 8))
        self.assertReceipt(run, self.built("on"))

    def test_a_format_error_at_response_7(self):
        script = [BASH] * 6 + [{"content": "pkg/core.py:11:        return target(2)"}, BASH]
        for mode in ("off", "on"):
            with self.subTest(mode=mode):
                run = self.run_arm(script, mode)
                last = run.server.chat_requests()[7]["messages"][-1]
                self.assertEqual(last["role"], "user")
                self.assertTrue(last["content"].startswith("Format error: "))
                # The model's own text is not a tool output: only the bash line of responses 1-6 is listed.
                self.assertReceipt(run, self.built(mode, appended_to="format_error", tool_call_id=None))
                self.assertEqual(("Candidate call sites" in last["content"]), mode == "on")
                self.assertTrue(last["content"].endswith(DEADLINE))

    def test_no_candidate_gives_the_empty_ledger(self):
        empty = HEADER + "\n(none)"
        run = self.run_arm([BASH] * 8, "on", output="ok\n")
        self.assertReceipt(run, self.built("on", text=empty, candidates=0, entries=0))
        self.assertTrue(run.server.chat_requests()[7]["messages"][-1]["content"].endswith(PREFIX + empty + DEADLINE))

    def test_real_search_hits_feed_the_ledger(self):
        search = {"calls": [("source_reference_search", {"query": "target"})]}
        run = self.run_arm([search] + [BASH] * 7, "on", output="ok\n")
        event = self.values(run, "evidence_ledger")[0]
        self.assertEqual([(e["path"], e["line"], e["excerpt"], e["source"]["kind"]) for e in event["entries"]],
                         [("pkg/core.py", 6, "return target(1)", "search_hit"),
                          ("pkg/core.py", 11, "return target(2)", "search_hit")])
        self.assertIn(PREFIX + event["text"] + DEADLINE, run.server.chat_requests()[7]["messages"][-1]["content"])


# ---------------------------------------------------------------------------------------------------------------
# (5) the runner
# ---------------------------------------------------------------------------------------------------------------

@unittest.skipUnless(POSIX, "the runner reads checkouts with POSIX no-follow reads")
class LedgerRunnerTests(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.runner, self.dispatch, self.manifest = G.runner_fixture(self.base)
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
        server = H.FakeServer(**kwargs)
        self.servers.append(server)
        return server

    def parse(self, *extra):
        with contextlib.redirect_stderr(io.StringIO()):
            return self.runner.parse_args(["--dispatch", str(self.dispatch), "--dry-run", *extra])

    def refused(self, *extra):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
            self.runner.parse_args(["--dispatch", str(self.dispatch), "--dry-run", *extra])
        self.assertEqual(raised.exception.code, 2)
        return stderr.getvalue()

    def live_argv(self, server, *extra, run="run", seeds=None):
        argv = G.live_args(self.base, self.dispatch, self.manifest, server, run=run)
        index = argv.index("--blocks")
        argv[index + 1] = "LEDGER"
        index = argv.index("--seeds")
        del argv[index:index + 2]
        if seeds:
            argv += ["--seeds", seeds]
        return argv + list(extra)

    def test_registration(self):
        self.assertEqual(self.runner.REGISTERED_LEDGER_DESIGNS, {
            (8, 16384, "deadline", "off"): ("protocols/EXPA_A3_LEDGER.md", ("gemma4-31b",)),
            (8, 16384, "deadline", "on"): ("protocols/EXPA_A3_LEDGER.md", ("gemma4-31b",))})
        self.assertEqual(self.parse().episode_config, harness.EpisodeConfig())
        self.assertIsNone(self.parse().design)
        self.assertEqual(self.parse("--evidence-ledger", "none").episode_config, harness.EpisodeConfig())
        for mode in ("off", "on"):
            expected = harness.EpisodeConfig(final_turn_policy="deadline", evidence_ledger=mode)
            arm = ("--final-turn-policy", "deadline", "--evidence-ledger", mode)
            for selection in (("--blocks", "LEDGER"), ("--blocks", "LEDGER", "--seeds", "61"),
                              ("--blocks", "LEDGER", "--seeds", "61,73,89")):
                with self.subTest(mode=mode, selection=selection):
                    args = self.parse(*arm, *selection)
                    self.assertEqual((args.episode_config, args.design, args.blocks), (expected, "a3-ledger",
                                                                                     {"LEDGER"}))
            registered = (f"--evidence-ledger {mode} (protocols/EXPA_A3_LEDGER.md) is registered only with "
                          "--final-turn-policy deadline, the prereg budget (8 responses, 16384 tokens) and "
                          "--model-profile gemma4-31b")
            for bad in ((), ("--final-turn-policy", "none"), ("--final-turn-policy", "neutral")):
                with self.subTest(mode=mode, bad=bad):
                    self.assertIn(registered, self.refused("--evidence-ledger", mode, "--blocks", "LEDGER", *bad))
            # Another budget or profile is refused by A1's registration first (the deadline notice needs them too).
            # Only off at 16/32768 is registered, as A6 (protocols/EXPA_A6_BUDGET16.md, tests/test_expA_a6_budget16.py).
            budget16 = (("--max-model-calls", "16", "--context-budget", "32768"),) if mode == "on" else ()
            for bad in budget16 + (("--model-profile", "gemma4-12b"), ("--model-profile", "gemma4-31b-qat")):
                with self.subTest(mode=mode, bad=bad):
                    self.assertIn("--final-turn-policy deadline (protocols/EXPA_FINALTURN.md) is registered only with "
                                  "the prereg budget", self.refused(*arm, "--blocks", "LEDGER", *bad))
            selection = (f"--evidence-ledger {mode} (protocols/EXPA_A3_LEDGER.md) runs block LEDGER only, at seeds "
                         "61, 73, 89: pass --blocks LEDGER (and --seeds from those)")
            for bad in ((), ("--blocks", "P"), ("--blocks", "LEDGER,P"), ("--blocks", "P", "--seeds", "11"),
                        ("--blocks", "LEDGER", "--seeds", "11"), ("--blocks", "LEDGER", "--seeds", "61,23")):
                with self.subTest(mode=mode, bad=bad):
                    self.assertIn(selection, self.refused(*arm, *bad))
        for bad in ("On", "", "ledger"):
            self.assertIn("invalid choice", self.refused("--evidence-ledger", bad))
        # Without the option, LEDGER and its seeds are not selectable, and A1 keeps its rule.
        self.assertIn("unknown blocks", self.refused("--blocks", "LEDGER"))
        self.assertIn("unknown seeds", self.refused("--seeds", "61"))
        self.assertIn("runs block P at seed 11 or seed 23 only", self.refused("--final-turn-policy", "deadline"))

    def test_dry_runs(self):
        argv = ["--dispatch", str(self.dispatch), "--dry-run", "--blocks", "LEDGER"]
        for mode in ("off", "on"):
            with self.subTest(mode=mode):
                code, output = G.invoke_runner(self.runner, argv + ["--final-turn-policy", "deadline",
                                                                    "--evidence-ledger", mode])
                summary = json.loads(output)
                self.assertEqual(code, 0)
                self.assertEqual(summary["design"], harness.design_table_for("a3-ledger"))
                self.assertEqual((summary["design_cells"], summary["selected_cells"], summary["by_block"]),
                                 (41 + 12, 12, {"LEDGER": 12}))
                self.assertEqual(summary["by_condition"], {"LEDGER:omit_node:C1": 6, "LEDGER:vcomplete:C1": 6})
                self.assertEqual(summary["first_cells"][:2], ["LEDGER|item-b|vcomplete|C1|s61",
                                                              "LEDGER|item-b|omit_node|C1|s61"])
                self.assertEqual(summary["episode_config"], harness.EpisodeConfig(
                    final_turn_policy="deadline", evidence_ledger=mode).record())

    def test_run_json_and_every_episode(self):
        runs = {}
        for mode in ("off", "on"):
            server = self.server(default=SEARCH)         # never submits: every episode reaches its 8th response
            argv = self.live_argv(server, "--final-turn-policy", "deadline", "--evidence-ledger", mode, run=mode)
            with mock.patch.object(G.tools, "ReadOnlyBash", G.FakeJailBash):
                code, _ = G.invoke_runner(self.runner, argv)
            self.assertEqual(code, 0)
            results = sorted((self.base / mode / "cells").rglob("result.json"))
            self.assertEqual(len(results), 1)
            runs[mode] = (json.loads((self.base / mode / "run.json").read_bytes()),
                          (self.base / mode / "cells.jsonl").read_text().splitlines(),
                          json.loads(results[0].read_bytes()), server)
        for mode, (run_json, cells, result, server) in runs.items():
            with self.subTest(mode=mode):
                record = harness.EpisodeConfig(final_turn_policy="deadline", evidence_ledger=mode).record()
                self.assertEqual(run_json["episode_config"], record)
                self.assertEqual(run_json["design"]["name"], "a3-ledger")
                self.assertEqual(run_json["design"]["cells_by_block"],
                                 {"LEDGER": 12, "P": 30, "S-EDGE": 2, "S-NAT": 3, "S-VC": 4, "T": 2})
                self.assertEqual(len(cells), 53)
                self.assertTrue(json.loads(cells[0])["cell_id"].startswith("LEDGER|"))
                self.assertEqual((result["block"], result["seed"], result["config"]), ("LEDGER", 61, record))
                receipt = result["exposure"]["evidence_ledger"]
                self.assertEqual((receipt["mode"], receipt["computed"], receipt["delivered"]),
                                 (mode, True, mode == "on"))
                self.assertEqual("Candidate call sites" in json.dumps(server.chat_requests()), mode == "on")
                self.assertEqual(S.detect_design(run_json, [json.loads(line)["cell_id"] for line in cells])[0],
                                 "a3-ledger")
        self.assertEqual({k: v for k, v in runs["off"][0].items() if k not in ("episode_config", "run_id")},
                         {k: v for k, v in runs["on"][0].items() if k not in ("episode_config", "run_id")})

    def test_a_relaunch_with_another_arm_is_refused(self):
        server = self.server(default=H.SUBMIT)
        arm = ["--final-turn-policy", "deadline", "--evidence-ledger"]
        with mock.patch.object(G.tools, "ReadOnlyBash", G.FakeJailBash):
            self.assertEqual(G.invoke_runner(self.runner, self.live_argv(server, *arm, "on"))[0], 0)
        chats = len(server.chat_requests())
        for extra in (arm[:-1] + ["--evidence-ledger", "off"],):
            with self.subTest(extra=extra):
                with mock.patch.object(G.tools, "ReadOnlyBash", G.FakeJailBash), \
                        self.assertRaisesRegex(L.LedgerInconsistency, "run.json already exists with different content"):
                    G.invoke_runner(self.runner, self.live_argv(server, *extra))
        self.assertEqual(len(server.chat_requests()), chats)
        with mock.patch.object(G.tools, "ReadOnlyBash", G.FakeJailBash):
            self.assertEqual(G.invoke_runner(self.runner, self.live_argv(server, *arm, "on"))[0], 0)
        self.assertEqual(len(server.chat_requests()), chats + 1)


# ---------------------------------------------------------------------------------------------------------------
# (6) the launch script (stubbed host, nothing is started)
# ---------------------------------------------------------------------------------------------------------------

@unittest.skipUnless(POSIX and shutil.which("bash"), "Linux only: the launch script runs on Core1")
class LedgerLaunchScriptTests(unittest.TestCase):
    maxDiff = None

    def host(self, *prepare_extra):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        host = G.LaunchHost(directory.name)
        host.run(*G.PREPARE_ARGV, "--runtime-python", str(host.python), *prepare_extra)
        host.stage("golden1")
        return host

    def runner_units(self, host, *extra, check=True):
        result = host.run("runner", "--run", "golden1", *extra, check=check)
        return result, [host.normalize(call) for call in G.units(host.sudo_calls(), "runner")]

    def runner_args(self, unit):
        tail = unit[next(i for i, value in enumerate(unit) if value.endswith("/scripts/expA_run.py")) + 1:]
        with contextlib.redirect_stderr(io.StringIO()):
            return G.load_script("expA_run").parse_args(tail)

    def test_syntax(self):
        self.assertEqual(subprocess.run(["bash", "-n", str(G.PROJECT / "scripts" / "expA_core1_launch.sh")],
                                        timeout=60).returncode, 0)

    def test_an_arm_is_forwarded_only_when_registered(self):
        host = self.host()
        golden = G.GOLDEN["launch"]["runner_units"][0]
        self.assertEqual(golden[-4:], ["--blocks", "P", "--seeds", "11"])
        for mode in ("off", "on"):
            for seeds in ((), ("--seeds", "61"), ("--seeds", "61,73,89")):
                with self.subTest(mode=mode, seeds=seeds):
                    result, units = self.runner_units(host, "--blocks", "LEDGER", *seeds, "--final-turn-policy",
                                                      "deadline", "--evidence-ledger", mode)
                    self.assertEqual(units, [golden[:-4] + ["--blocks", "LEDGER", *seeds, "--final-turn-policy",
                                                            "deadline", "--evidence-ledger", mode]])
                    self.assertIn(f"evidence ledger {mode} (protocols/EXPA_A3_LEDGER.md)", result.stderr)
                    args = self.runner_args(units[0])
                    self.assertEqual((args.episode_config, args.design), (harness.EpisodeConfig(
                        final_turn_policy="deadline", evidence_ledger=mode), "a3-ledger"))
        # none (or no option) keeps A1 and the default exactly.
        for extra in ((), ("--evidence-ledger", "none")):
            with self.subTest(extra=extra):
                result, units = self.runner_units(host, "--blocks", "P", "--seeds", "11", *extra)
                self.assertEqual(units, [golden])
                self.assertNotIn("evidence ledger", result.stderr)
                result, units = self.runner_units(host, "--blocks", "P", "--seeds", "11", "--final-turn-policy",
                                                  "deadline", *extra)
                self.assertEqual(units, [golden + ["--final-turn-policy", "deadline"]])

    def test_unregistered_arms_are_refused_before_any_unit(self):
        host = self.host()
        registered = ("--evidence-ledger on: registered only with --final-turn-policy deadline, the default budget "
                      "(8 responses, 16384 tokens) and model profile gemma4-31b (protocols/EXPA_A3_LEDGER.md)")
        selection = ("--evidence-ledger on: block LEDGER only, pass --blocks LEDGER (and --seeds from 61,73,89) "
                     "(protocols/EXPA_A3_LEDGER.md)")
        cases = ((["--blocks", "LEDGER", "--evidence-ledger", "On"],
                  "--evidence-ledger must be none, off or on, not On"),
                 (["--blocks", "LEDGER", "--evidence-ledger", "on"], registered),
                 (["--blocks", "LEDGER", "--final-turn-policy", "neutral", "--evidence-ledger", "on"], registered),
                 (["--blocks", "LEDGER", "--final-turn-policy", "deadline", "--evidence-ledger", "on",
                   "--max-model-calls", "16", "--context-budget", "32768"], "registered only with the default budget"),
                 (["--final-turn-policy", "deadline", "--evidence-ledger", "on"], selection),
                 (["--blocks", "P", "--seeds", "11", "--final-turn-policy", "deadline", "--evidence-ledger", "on"],
                  selection),
                 (["--blocks", "LEDGER", "--seeds", "11", "--final-turn-policy", "deadline", "--evidence-ledger", "on"],
                  selection),
                 (["--blocks", "LEDGER", "--seeds", "61,62", "--final-turn-policy", "deadline", "--evidence-ledger",
                   "on"], selection),
                 (["--blocks", "LEDGER", "--final-turn-policy", "deadline"],
                  "--final-turn-policy deadline: block P at seed 11 or 23 only"))
        for extra, message in cases:
            with self.subTest(extra=extra):
                result, units = self.runner_units(host, *extra, check=False)
                self.assertEqual((result.returncode, units), (2, []))
                self.assertIn(message, result.stderr)

    def test_an_arm_needs_the_default_model_profile(self):
        for profile in ("gemma4-12b", "gemma4-31b-qat"):
            with self.subTest(profile=profile):
                host = self.host("--model-profile", profile)
                result, units = self.runner_units(host, "--blocks", "LEDGER", "--final-turn-policy", "deadline",
                                                  "--evidence-ledger", "off", check=False)
                self.assertEqual((result.returncode, units), (2, []))
                # The final-turn policy is refused first (A1's rule): nothing reaches the A3 check.
                self.assertIn("--final-turn-policy deadline: registered only with the default budget",
                              result.stderr)


if __name__ == "__main__":
    unittest.main()
