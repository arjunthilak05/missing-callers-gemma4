"""SYNTHETIC tests for scripts/expA_timeline.py (ANALYSIS RECORD).

Every event, item and row below is SYNTHETIC: hand-made to exercise the visibility rules (STRICT, LOOSE
= L1 / L2), timing, repeat detection, last-action kinds and the family-weighted aggregation. None of it
is run data.
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("expA_timeline", ROOT / "scripts" / "expA_timeline.py")
T = importlib.util.module_from_spec(_SPEC)
sys.modules["expA_timeline"] = T
_SPEC.loader.exec_module(T)

# SYNTHETIC item: C* = pkg/mod.py::Holder.caller_fn, call site lines 12-13; target pkg/core.py::target_fn
ITEM = {"item_id": "synthetic-item", "family": "synthetic", "cstar": "pkg/mod.py::Holder.caller_fn",
        "cstar_simple": "caller_fn", "cstar_sites": [["pkg/mod.py", 12, 13]],
        "search_names": ["target_fn", "alias_fn"], "c4_contains_omitted_site": True}
TASK = "SYNTHETIC task: find every place where `target_fn` (defined in `pkg/core.py`, line 3) is invoked."


def ev(kind, value):
    return {"sequence": 0, "wall_time": 0.0, "kind": kind, "value": value}


def episode(steps, *, user=TASK, exit_status=None):
    """SYNTHETIC event list. Each step is one model response with at most one tool call."""
    events = [ev("episode_setup", {"user": user, "system": "SYNTHETIC system"})]
    submitted = False
    for i, step in enumerate(steps):
        prompt = step.get("prompt", 1000 * (i + 1))
        events.append(ev("context_accounting", {"call_index": i, "count": prompt}))
        if i == 0:
            events.append(ev("exposure_receipt", {"recorded_before_first_model_call": True}))
        events.append(ev("model_exchange", {"call_index": i, "response": {
            "usage": {"prompt_tokens": prompt, "completion_tokens": step.get("completion", 10)}}}))
        if "format_error" in step:
            events.append(ev("format_error", {"call_index": i, "reason": step["format_error"]}))
            continue
        action = {"id": f"call-{i}", "name": step["name"], "arguments": step["arguments"]}
        events.append(ev("parsed_actions", {"call_index": i, "actions": [action]}))
        events.append(ev("tool_action", action))
        if step["name"] == "submit_answer" and step.get("accepted", True):
            events.append(ev("submission_accepted", {"tool_call_id": action["id"], "call_index": i,
                                                     "submission": step["arguments"]}))
            submitted = True
            break
        record = {"status": step.get("status", "executed")}
        if "argv" in step:
            record["argv"] = step["argv"]
        events.append(ev("tool_output", {"tool_call_id": action["id"], "name": step["name"], "record": record}))
        content = json.dumps(step["visible"], sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        events.append(ev("model_visible_tool_result", {"tool_call_id": action["id"], "name": step["name"],
                                                       "content": content}))
    status = exit_status or ("Submitted" if submitted else "NoAnswer")
    events.append(ev("episode_finished", {"exit_status": status}))
    return events


def bash(command, output, argv, status="executed"):
    return {"name": "bash", "arguments": {"command": command}, "argv": argv, "status": status,
            "visible": {"exit_code": 0, "output": output, "truncated": False}}


def search(query, hits):
    return {"name": "source_reference_search", "arguments": {"query": query},
            "visible": {"query": query, "kind": "lexical_candidates_not_proven_edges", "hits": hits,
                        "truncated": False, "error": None}}


SUBMIT = {"name": "submit_answer", "arguments": {"sites": [{"path": "pkg/mod.py", "line": 12}], "complete": "yes"}}
NOISE = bash("ls pkg", "core.py\nmod.py\n", [["/usr/bin/ls", "pkg"]])


def run(steps, **kw):
    return T.episode_timeline(episode(steps, **kw), ITEM)


class PrimitiveTests(unittest.TestCase):
    def test_path_matches_prefixes(self):
        self.assertTrue(T.path_matches("pkg/mod.py", "pkg/mod.py"))
        self.assertTrue(T.path_matches("./pkg/mod.py", "pkg/mod.py"))
        self.assertTrue(T.path_matches("/repo/pkg/mod.py", "pkg/mod.py"))
        self.assertTrue(T.path_matches("pkg\\mod.py", "pkg/mod.py"))
        self.assertFalse(T.path_matches("xpkg/mod.py", "pkg/mod.py"))
        self.assertFalse(T.path_matches("pkg/mod.pyc", "pkg/mod.py"))

    def test_path_occurs_boundaries(self):
        self.assertTrue(T.path_occurs("see ./pkg/mod.py.", "pkg/mod.py"))
        self.assertTrue(T.path_occurs("pkg/mod.py::Holder.caller_fn", "pkg/mod.py"))
        self.assertFalse(T.path_occurs("otherpkg/mod.py", "pkg/mod.py"))
        self.assertFalse(T.path_occurs("pkg/mod.pyx", "pkg/mod.py"))

    def test_student_t_quantiles(self):
        self.assertAlmostEqual(T.student_t_ppf(0.975, 29), 2.045229642132703, places=9)
        self.assertAlmostEqual(T.student_t_ppf(0.975, 1), 12.706204736174698, places=8)
        try:
            from absent_edges.probe import analysis as frozen
        except Exception:  # the frozen package is optional here
            frozen = None
        if frozen is not None:
            for df in (1, 2, 5, 29, 120):
                self.assertAlmostEqual(T.student_t_ppf(0.975, df), frozen.student_t_ppf(0.975, df), places=10)

    def test_percentile_linear(self):
        self.assertEqual(T.percentile([1, 2, 3, 4], 0.5), 2.5)
        self.assertAlmostEqual(T.percentile([1, 2, 3, 4], 0.9), 3.7)
        self.assertIsNone(T.percentile([], 0.5))


class StrictRuleTests(unittest.TestCase):
    def test_c4a_verification_block_in_initial_prompt_is_r0(self):
        payload = {"query": {"symbol": "pkg/core.py::target_fn"}, "matches": [],
                   "verification": {"kind": "lexical_candidates_not_proven_edges",
                                    "hits": [{"line": 12, "path": "pkg/mod.py", "text": "x = target_fn()"}]}}
        user = TASK + "\n\nA code-graph index returned this answer:\n" + json.dumps(payload, sort_keys=True)
        out = run([NOISE, SUBMIT], user=user)
        self.assertEqual(out["first_cstar_visible_response"], {"strict": 0, "loose": 0})
        self.assertEqual(out["first_cstar_visible_by_clause"]["S_struct"], 0)

    def test_verification_hit_outside_span_is_not_visible(self):
        payload = {"verification": {"hits": [{"line": 14, "path": "pkg/mod.py", "text": "target_fn"}]}}
        out = run([SUBMIT], user=TASK + "\n" + json.dumps(payload))
        self.assertIsNone(out["first_cstar_visible_response"]["strict"])

    def test_recursive_grep_path_line_text(self):
        out = run([NOISE, bash("grep -rn target_fn pkg/", "pkg/core.py:3:def target_fn():\n"
                               "./pkg/mod.py:13:        y = target_fn()\n",
                               [["/usr/bin/grep", "-rn", "target_fn", "pkg/"]]), SUBMIT])
        self.assertEqual(out["first_cstar_visible_response"]["strict"], 2)
        self.assertEqual(out["first_cstar_visible_by_clause"]["S_text"], 2)

    def test_absolute_prefix_and_wrong_prefix(self):
        out = run([bash("grep -rn target_fn /repo/pkg", "/repo/pkg/mod.py:12: x = target_fn()\n",
                        [["/usr/bin/grep", "-rn", "target_fn", "/repo/pkg"]])])
        self.assertEqual(out["first_cstar_visible_response"]["strict"], 1)
        out = run([bash("grep -rn target_fn .", "xpkg/mod.py:12: x = target_fn()\n",
                        [["/usr/bin/grep", "-rn", "target_fn", "."]])])
        self.assertIsNone(out["first_cstar_visible_response"]["strict"])

    def test_structured_search_hits(self):
        out = run([search("target_fn", [{"line": 3, "path": "pkg/core.py", "text": "def target_fn():"},
                                        {"line": 12, "path": "pkg/mod.py", "text": "x = target_fn()"}]), SUBMIT])
        self.assertEqual(out["first_cstar_visible_response"]["strict"], 1)
        self.assertEqual(out["first_cstar_visible_by_clause"]["S_struct"], 1)
        self.assertEqual(out["first_target_search_response"], 1)

    def test_single_file_grep_is_missed_by_strict_but_found_by_l1(self):
        out = run([bash('grep -n "target_fn" pkg/mod.py', "12:        x = target_fn()\n",
                        [["/usr/bin/grep", "-n", "target_fn", "pkg/mod.py"]])])
        self.assertIsNone(out["first_cstar_visible_response"]["strict"])
        self.assertEqual(out["first_cstar_visible_response"]["loose"], 1)
        self.assertEqual(out["first_cstar_visible_by_clause"]["L1"], 1)


class LooseRuleTests(unittest.TestCase):
    NUMBERED = "    11\t    def caller_fn(self):\n    12\t        x = target_fn()\n"

    def test_cat_nl_ba_pipeline_counts(self):
        out = run([bash("cat pkg/mod.py | nl -ba | sed -n '10,14p'", self.NUMBERED,
                        [["/usr/bin/cat", "pkg/mod.py"], ["/usr/bin/nl", "-ba"],
                         ["/usr/bin/sed", "--sandbox", "-n", "10,14p"]])])
        self.assertEqual(out["first_cstar_visible_by_clause"]["L1"], 1)

    def test_plain_nl_numbers_are_not_file_lines(self):
        output = "    12\t        x = target_fn()\n"
        out = run([bash("nl pkg/mod.py", output, [["/usr/bin/nl", "pkg/mod.py"]])])
        self.assertIsNone(out["first_cstar_visible_by_clause"]["L1"])
        self.assertIsNone(out["first_cstar_visible_response"]["loose"])

    def test_renumbering_after_selection_does_not_count(self):
        out = run([bash("sed -n '1,20p' pkg/mod.py | cat -n", "    12\t        x = target_fn()\n",
                        [["/usr/bin/sed", "--sandbox", "-n", "1,20p", "pkg/mod.py"], ["/usr/bin/cat", "-n"]])])
        self.assertIsNone(out["first_cstar_visible_by_clause"]["L1"])

    def test_cat_n_then_sed_counts_and_cat_sn_does_not(self):
        out = run([bash("cat -n pkg/mod.py | sed -n '10,14p'", self.NUMBERED,
                        [["/usr/bin/cat", "-n", "pkg/mod.py"], ["/usr/bin/sed", "--sandbox", "-n", "10,14p"]])])
        self.assertEqual(out["first_cstar_visible_by_clause"]["L1"], 1)
        out = run([bash("cat -sn pkg/mod.py", "    12\t        x = target_fn()\n",
                        [["/usr/bin/cat", "-sn", "pkg/mod.py"]])])
        self.assertIsNone(out["first_cstar_visible_by_clause"]["L1"])

    def test_two_py_operands_disable_l1(self):
        out = run([bash("grep -n target_fn pkg/mod.py pkg/core.py", "12:        x = target_fn()\n",
                        [["/usr/bin/grep", "-n", "target_fn", "pkg/mod.py", "pkg/core.py"]])])
        self.assertIsNone(out["first_cstar_visible_by_clause"]["L1"])

    def test_l1_number_outside_span(self):
        out = run([bash("grep -n target_fn pkg/mod.py", "14:        x = target_fn()\n",
                        [["/usr/bin/grep", "-n", "target_fn", "pkg/mod.py"]])])
        self.assertIsNone(out["first_cstar_visible_by_clause"]["L1"])

    def test_l2_name_cooccurrence_with_path_in_command(self):
        out = run([NOISE, bash("cat pkg/mod.py", "class Holder:\n    def caller_fn(self):\n        return 1\n",
                               [["/usr/bin/cat", "pkg/mod.py"]])])
        self.assertIsNone(out["first_cstar_visible_response"]["strict"])
        self.assertIsNone(out["first_cstar_visible_by_clause"]["L1"])
        self.assertEqual(out["first_cstar_visible_by_clause"]["L2"], 2)
        self.assertEqual(out["first_cstar_visible_response"]["loose"], 2)

    def test_l2_needs_whole_identifier_and_the_site_file(self):
        out = run([bash("cat pkg/mod.py", "def caller_fn_extra():\n    pass\n", [["/usr/bin/cat", "pkg/mod.py"]])])
        self.assertIsNone(out["first_cstar_visible_by_clause"]["L2"])
        out = run([bash("cat pkg/core.py", "def caller_fn():\n    pass\n", [["/usr/bin/cat", "pkg/core.py"]])])
        self.assertIsNone(out["first_cstar_visible_by_clause"]["L2"])

    def test_initial_prompt_is_judged_by_strict_only(self):
        user = TASK + "\nSYNTHETIC answer naming pkg/mod.py and a snippet calling self.caller_fn()"
        out = run([SUBMIT], user=user)
        self.assertEqual(out["first_cstar_visible_response"], {"strict": None, "loose": None})

    def test_loose_never_later_than_strict(self):
        out = run([bash("grep -n target_fn pkg/mod.py", "12:        x = target_fn()\n",
                        [["/usr/bin/grep", "-n", "target_fn", "pkg/mod.py"]]),
                   bash("grep -rn target_fn pkg", "pkg/mod.py:12:        x = target_fn()\n",
                        [["/usr/bin/grep", "-rn", "target_fn", "pkg"]])])
        self.assertEqual(out["first_cstar_visible_response"], {"strict": 2, "loose": 1})


class TimingAndActionTests(unittest.TestCase):
    def test_visibility_timing_relative_to_submission(self):
        steps = [bash("grep -rn target_fn pkg", "pkg/mod.py:12: x = target_fn()\n",
                      [["/usr/bin/grep", "-rn", "target_fn", "pkg"]]), SUBMIT]
        row = dict(run(steps), exit_status="Submitted")
        self.assertEqual(row["submitted_response"], 2)
        self.assertEqual(row["n_responses"], 2)
        self.assertTrue(T.visible_before(row, "strict", row["submitted_response"]))
        same = dict(run([SUBMIT]), exit_status="Submitted")
        self.assertFalse(T.visible_before(same, "strict", same["submitted_response"]))

    def test_visible_only_in_last_responses_results_is_not_before_the_cap(self):
        steps = [NOISE] * 7 + [bash("grep -rn target_fn pkg", "pkg/mod.py:12: x = target_fn()\n",
                                    [["/usr/bin/grep", "-rn", "target_fn", "pkg"]])]
        row = dict(run(steps), exit_status="NoAnswer")
        self.assertEqual(row["n_responses"], 8)
        self.assertEqual(row["first_cstar_visible_response"]["strict"], 8)
        self.assertFalse(T.visible_before(row, "strict", row["n_responses"]))
        steps = [NOISE] * 6 + [steps[-1], NOISE]
        row = dict(run(steps), exit_status="NoAnswer")
        self.assertTrue(T.visible_before(row, "strict", row["n_responses"]))

    def test_repeat_detection_uses_canonical_arguments(self):
        a = bash("grep -rn target_fn pkg", "", [["/usr/bin/grep", "-rn", "target_fn", "pkg"]])
        b = bash("grep -rn alias_fn pkg", "", [["/usr/bin/grep", "-rn", "alias_fn", "pkg"]])
        s1 = {"name": "source_reference_search", "arguments": {"query": "target_fn", "limit": 5},
              "visible": {"hits": []}}
        s2 = {"name": "source_reference_search", "arguments": {"limit": 5, "query": "target_fn"},
              "visible": {"hits": []}}
        out = run([a, b, a, s1, s2, a])
        self.assertEqual(out["n_tool_actions"], 6)
        self.assertEqual(out["n_identical_repeat_actions"], 3)

    def test_last_action_kinds(self):
        out = run([NOISE] * 8)
        self.assertEqual(out["last_action_kind"], "tool_call")
        rejected = bash("python x.py", "", [], status="rejected")
        rejected["visible"] = {"error": "command rejected: python is not allowed"}
        out = run([NOISE] * 7 + [rejected])
        self.assertEqual(out["last_action_kind"], "tool_rejected")
        self.assertEqual(out["n_rejected_tool_calls"], 1)
        out = run([NOISE] * 7 + [{"format_error": "your response did not call a tool"}])
        self.assertEqual(out["last_action_kind"], "no_tool_call")
        self.assertEqual(out["n_responses"], 8)
        out = run([NOISE] * 7 + [{"format_error": "Tool not allowlisted"}])
        self.assertEqual(out["last_action_kind"], "format_error")
        bad_submit = dict(SUBMIT, accepted=False, status="rejected", visible={"error": "submit_answer rejected"})
        out = run([NOISE] * 7 + [bad_submit])
        self.assertEqual(out["last_action_kind"], "submit_rejected")
        out = run([NOISE, SUBMIT])
        self.assertEqual(out["last_action_kind"], "submit_accepted")

    def test_target_search_and_tokens(self):
        steps = [dict(NOISE, prompt=900, completion=7),
                 dict(bash("grep -rn alias_fn pkg", "", [["/usr/bin/grep", "-rn", "alias_fn", "pkg"]]),
                      prompt=1500, completion=9),
                 dict(SUBMIT, prompt=1400, completion=30)]
        out = run(steps)
        self.assertEqual(out["first_target_search_response"], 2)
        self.assertEqual(out["prompt_tokens_by_response"], [900, 1500, 1400])
        self.assertEqual(out["prompt_tokens_max"], 1500)
        self.assertEqual(out["prompt_tokens_final"], 1400)
        self.assertEqual(out["completion_tokens_total"], 46)
        out = run([bash("grep -rn target_fn_helper pkg", "", [["/usr/bin/grep", "-rn", "target_fn_helper", "pkg"]])])
        self.assertIsNone(out["first_target_search_response"])


def row(family, item, seed, *, exit_status="Submitted", recovered=1, submitted=3, strict=None, loose=None, n=None):
    """SYNTHETIC per-episode row in the shape the aggregation reads."""
    return {"cell_id": f"P|{item}|v|C1|s{seed}", "block": "P", "condition": "C1", "family": family, "item_id": item,
            "seed": seed, "exit_status": exit_status, "site_recovered": recovered if exit_status == "Submitted" else 0,
            "submitted_response": submitted if exit_status == "Submitted" else None,
            "n_responses": n if n is not None else (submitted if exit_status == "Submitted" else 8),
            "first_cstar_visible_response": {"strict": strict, "loose": loose},
            "first_target_search_response": 1, "last_action_kind": "tool_call", "n_tool_actions": 3,
            "n_identical_repeat_actions": 0, "n_rejected_tool_calls": 0, "prompt_tokens_max": 5000,
            "prompt_tokens_final": 5000, "problems": []}


class AggregationTests(unittest.TestCase):
    def setUp(self):
        # SYNTHETIC: family A has two items (a1 two of three seeds submitted, a2 none); family B has one item
        # (all three submitted).
        self.rows = [row("A", "a1", 11), row("A", "a1", 23, exit_status="NoAnswer", strict=2, loose=1),
                     row("A", "a1", 47),
                     row("A", "a2", 11, exit_status="NoAnswer"), row("A", "a2", 23, exit_status="NoAnswer", loose=7),
                     row("A", "a2", 47, exit_status="NoAnswer", strict=8, loose=8),
                     row("B", "b1", 11), row("B", "b1", 23), row("B", "b1", 47)]

    def test_family_weighted_submission_rate(self):
        entry = T.rate(self.rows, lambda r: r["exit_status"] == "Submitted")
        self.assertEqual((entry["micro"]["k"], entry["micro"]["n"]), (5, 9))
        fw = entry["family_weighted"]
        # item means: a1 = 2/3, a2 = 0 -> family A = 1/3; b1 = 1 -> family B = 1; mean (1/3 + 1)/2 = 2/3
        self.assertAlmostEqual(fw["mean"], 2 / 3)
        self.assertEqual((fw["n_families"], fw["n_items"], fw["df"]), (2, 3, 1))
        sd = abs(1 - 1 / 3) / math.sqrt(2)
        half = 12.706204736174698 * sd / math.sqrt(2)
        self.assertAlmostEqual(fw["lower"], 2 / 3 - half, places=7)
        self.assertAlmostEqual(fw["upper"], 2 / 3 + half, places=7)

    def test_conditional_share_drops_items_and_families_without_eligible_episodes(self):
        no_answer = lambda r: r["exit_status"] == "NoAnswer"
        loose = T.rate(self.rows, lambda r: T.visible_before(r, "loose", r["n_responses"]), no_answer)
        # NoAnswer episodes: a1/s23 (loose 1 -> visible), a2/s11 (never), a2/s23 (loose 7 -> visible),
        # a2/s47 (loose 8 -> only after the last response). Item means a1 = 1, a2 = 1/3; only family A.
        self.assertEqual((loose["micro"]["k"], loose["micro"]["n"]), (2, 4))
        self.assertAlmostEqual(loose["family_weighted"]["mean"], (1 + 1 / 3) / 2)
        self.assertEqual(loose["family_weighted"]["n_families"], 1)
        self.assertIsNone(loose["family_weighted"]["lower"])
        strict = T.rate(self.rows, lambda r: T.visible_before(r, "strict", r["n_responses"]), no_answer)
        self.assertEqual(strict["micro"]["k"], 1)

    def test_group_summary_curves_and_means(self):
        summary = T.group_summary(self.rows)
        self.assertEqual(summary["exit_status"], {"NoAnswer": 4, "Submitted": 5})
        curve = summary["cumulative_by_k"]
        self.assertEqual(curve["2"]["submitted_by_k"]["micro"]["k"], 0)
        self.assertEqual(curve["3"]["submitted_by_k"]["micro"]["k"], 5)
        self.assertEqual(curve["8"]["cstar_visible_by_k_loose"]["micro"]["k"], 3)
        self.assertEqual(curve["0"]["cstar_visible_by_k_strict"]["micro"]["k"], 0)
        self.assertAlmostEqual(summary["responses_among_submitted"]["micro_mean"], 3.0)
        self.assertEqual(summary["noanswer_last_action_kind"]["counts"], {"tool_call": 4})


if __name__ == "__main__":
    unittest.main()
