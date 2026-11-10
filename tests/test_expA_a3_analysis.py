"""scripts/expA_a3_analysis.py (protocols/EXPA_A3_LEDGER.md section 4) on SYNTHETIC score rows only.

Hand-computed fixtures: the primary M = G_off - G_on (item mean over seeds -> paired item value -> family mean ->
equal-weight mean over families, one-sample t with df = F - 1), the seed-prefix rule, exclusion of cells missing for
infrastructure, non-submitting episodes scored 0, the secondary contrasts and the non-inferiority flag, the hard-panel
secondary (seed 61, no decision), and every refusal. Standard library only; runs on any OS.
"""
import contextlib
from fractions import Fraction
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, PROJECT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


A3 = load_script("expA_a3_analysis")
T = load_script("expA_reanalyze_independent")
ITEMS_SHA = "1" * 64


def row(item, family, variant, seed, recovered, *, state="completed", exit_status=None, precision=None,
        block="LEDGER", condition="C1", items_sha=ITEMS_SHA):
    completed = state == "completed"
    if exit_status is None:
        exit_status = ("Submitted" if recovered else "NoAnswer") if completed else "Missing"
    submitted = completed and exit_status == "Submitted"
    low, high = precision if precision is not None else ((1.0, 1.0) if submitted and recovered else (0.0, 0.5))
    return {"schema": "expA-score-v2", "kind": "episode", "design": "a3-ledger",
            "cell_id": f"{block}|{item}|{variant}|{condition}|s{seed}", "block": block, "item_id": item,
            "family": family, "variant": variant, "condition": condition, "seed": seed, "cell_state": state,
            "completed": completed, "exit_status": exit_status, "site_recovered": recovered if completed else None,
            "precision_lower": low if submitted else None, "precision_upper": high if submitted else None,
            "items_sha256": items_sha}


def arm_rows(recovery, seeds=(61, 73, 89), overrides=None):
    """recovery: {(item, family): {variant: [r61, r73, r89]}} -> rows; overrides: {(item, variant, seed): kwargs}."""
    rows = []
    for (item, family), by_variant in recovery.items():
        for variant, values in by_variant.items():
            for seed, value in zip(seeds, values):
                kwargs = dict((overrides or {}).get((item, variant, seed), {}))
                rows.append(row(item, family, variant, seed, value, **kwargs))
    # The default design's other blocks are frozen with the A3 design and never dispatched.
    rows.append({"block": "P", "cell_id": "P|i1|none|C0|s11", "cell_state": "not_dispatched"})
    return rows


def write(directory, name, rows):
    path = Path(directory) / name
    path.write_bytes(b"".join(json.dumps(r, sort_keys=True).encode() + b"\n" for r in rows))
    return str(path)


# Three families, two items each. Without the ledger: complete always recovered, incomplete never (G_off = 1 for every
# item). With the ledger: complete always recovered; incomplete recovered at f1 in every seed (G_on 0, M 1), at f2 in
# one of three seeds (G_on 2/3, M 1/3), never at f3 (M 0). Family means of M: 1, 1/3, 0 -> mean 4/9.
FAMILY_OF = {"i1": "f1", "i2": "f1", "i3": "f2", "i4": "f2", "i5": "f3", "i6": "f3"}
OFF = {(i, f): {"vcomplete": [1, 1, 1], "omit_node": [0, 0, 0]} for i, f in FAMILY_OF.items()}
ON = {(i, f): {"vcomplete": [1, 1, 1], "omit_node": {"f1": [1, 1, 1], "f2": [1, 0, 0], "f3": [0, 0, 0]}[f]}
      for i, f in FAMILY_OF.items()}


class A3AnalysisTests(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.dir = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def analyse(self, off_rows, on_rows, **kwargs):
        off = A3.load_rows(write(self.dir, "off.jsonl", off_rows), "off")
        on = A3.load_rows(write(self.dir, "on.jsonl", on_rows), "on")
        return A3.analyse(off, on, **kwargs)

    def test_the_design_constants_are_the_harness_and_scorer_ones(self):
        from absent_edges.probe import harness, score
        design = harness.EXTENSION_BLOCK_DESIGN[A3.BLOCK]
        self.assertEqual(A3.SEEDS, design["seeds"])
        self.assertEqual({(variant, A3.CONDITION) for variant in A3.VARIANTS.values()}, set(design["cells"]))
        self.assertEqual((A3.VARIANTS["complete"], A3.VARIANTS["incomplete"]), ("vcomplete", "omit_node"))
        self.assertEqual(A3.HARD_SEEDS, (61,))
        self.assertEqual(score.DESIGNS["a3-ledger"][0].seeds, A3.SEEDS)
        self.assertEqual(A3.PROTOCOL, harness.LEDGER_PROTOCOL)
        self.assertEqual((A3.ALPHA, A3.NONINFERIORITY_MARGIN), (0.05, Fraction(-3, 100)))

    def test_the_hand_computed_primary(self):
        result = self.analyse(arm_rows(OFF), arm_rows(ON))
        self.assertEqual((result["status"], result["seeds_used"], result["items_with_all_four_arms"]),
                         ("analysed", [61, 73, 89], 6))
        m = result["M_reduction_in_omission_cost"]
        self.assertEqual((m["estimate_exact"], m["n_families"], m["n_items"], m["df"]), ("4/9", 3, 6, 2))
        self.assertEqual(m["family_means"], {"f1": 1.0, "f2": 1 / 3, "f3": 0.0})
        expected = T.t_statistics([Fraction(1), Fraction(1, 3), Fraction(0)], 0.95)
        self.assertEqual((m["t"], m["p_two_sided"], m["ci_low"], m["ci_high"]),
                         (expected["t"], expected["p_two_sided"], expected["lower"], expected["upper"]))
        self.assertEqual(result["G_off_omission_cost_without_ledger"]["estimate_exact"], "1/1")
        self.assertEqual(result["G_on_omission_cost_with_ledger"]["estimate_exact"], "5/9")
        self.assertEqual(result["S1_incomplete_on_minus_off"]["estimate_exact"], "4/9")
        self.assertEqual(result["S2_complete_on_minus_off"]["estimate_exact"], "0/1")
        self.assertEqual(result["arm_levels"]["incomplete_on"]["estimate_exact"], "4/9")
        decision = result["primary_decision"]
        self.assertEqual((decision["alpha"], decision["p_two_sided"], decision["reject_null_at_alpha"]),
                         (0.05, expected["p_two_sided"], expected["p_two_sided"] <= 0.05))
        self.assertEqual(decision["direction"], "ledger reduces the omission cost")
        # Submission follows recovery in this fixture: incomplete off never submits, on submits 4/9 (family mean).
        self.assertEqual(result["S4_submission"]["incomplete_off"]["estimate_exact"], "0/1")
        self.assertEqual(result["S4_submission_on_minus_off"]["incomplete"]["estimate_exact"], "4/9")
        self.assertEqual(result["exits"]["on"]["incomplete"], {"NoAnswer": 10, "Submitted": 8})

    def test_family_weighting_is_equal(self):
        # f1 gets four more items with M = 1: the family mean of f1 stays 1, so the estimate stays 4/9.
        extra = {(f"j{n}", "f1"): {"vcomplete": [1, 1, 1], "omit_node": [0, 0, 0]} for n in range(4)}
        extra_on = {(f"j{n}", "f1"): {"vcomplete": [1, 1, 1], "omit_node": [1, 1, 1]} for n in range(4)}
        result = self.analyse(arm_rows({**OFF, **extra}), arm_rows({**ON, **extra_on}))
        self.assertEqual(result["M_reduction_in_omission_cost"]["estimate_exact"], "4/9")
        self.assertEqual(result["M_reduction_in_omission_cost"]["n_items"], 10)

    def test_non_submitting_episodes_score_zero_and_infrastructure_cells_are_excluded(self):
        # i3 (f2) incomplete-on s61 becomes missing for infrastructure: its item mean is over s73, s89 (0, 0), so its
        # G_on is 1 and its M 0 instead of 1/3; i4 keeps 1/3, f2 mean 1/6; family means 1, 1/6, 0 -> 7/18.
        overrides = {("i3", "omit_node", 61): {"state": "missing_infrastructure"}}
        result = self.analyse(arm_rows(OFF), arm_rows(ON, overrides=overrides))
        self.assertEqual(result["M_reduction_in_omission_cost"]["estimate_exact"], "7/18")
        self.assertEqual(result["M_reduction_in_omission_cost"]["n_items"], 6)
        self.assertEqual(result["missing_infrastructure_cells"], {"off": 0, "on": 1})
        # A completed episode that did not submit (ContextLimit) scores 0, it is not excluded.
        overrides = {("i1", "omit_node", 61): {"exit_status": "ContextLimit"}}
        on = arm_rows(ON, overrides=overrides)
        for r in on:
            if r.get("cell_id") == "LEDGER|i1|omit_node|C1|s61":
                r["site_recovered"] = 0
        result = self.analyse(arm_rows(OFF), on)
        # i1: incomplete-on mean 2/3, G_on 1/3, M 2/3; f1 = (2/3 + 1)/2 = 5/6; (5/6 + 1/3 + 0)/3 = 7/18.
        self.assertEqual(result["M_reduction_in_omission_cost"]["estimate_exact"], "7/18")
        self.assertEqual(result["exits"]["on"]["incomplete"]["ContextLimit"], 1)

    def test_an_item_needs_all_four_arms(self):
        overrides = {("i6", "vcomplete", seed): {"state": "missing_infrastructure"} for seed in (61, 73, 89)}
        result = self.analyse(arm_rows(OFF, overrides=overrides), arm_rows(ON))
        self.assertEqual(result["items_with_all_four_arms"], 5)
        self.assertEqual(result["M_reduction_in_omission_cost"]["n_items"], 5)
        self.assertEqual(result["M_reduction_in_omission_cost"]["estimate_exact"], "4/9")   # f3 keeps i5 (M 0)

    def test_the_seed_prefix_rule(self):
        overrides = {("i2", "omit_node", 73): {"state": "not_dispatched"}}
        result = self.analyse(arm_rows(OFF), arm_rows(ON, overrides=overrides))
        self.assertEqual(result["seeds_used"], [61])
        # Seed 61 only: incomplete-on is recovered at f1 and f2 (M 1), not at f3 (M 0) -> 2/3.
        self.assertEqual(result["M_reduction_in_omission_cost"]["estimate_exact"], "2/3")
        overrides = {("i2", "omit_node", 61): {"state": "infra_pending"}}
        result = self.analyse(arm_rows(OFF, overrides=overrides), arm_rows(ON))
        self.assertEqual((result["status"], result["seeds_used"]), ("no_complete_seed", []))
        self.assertNotIn("M_reduction_in_omission_cost", result)
        # A seed with missing (infrastructure) cells is complete; the cells are excluded.
        overrides = {("i2", "omit_node", 89): {"state": "missing_infrastructure"}}
        self.assertEqual(self.analyse(arm_rows(OFF), arm_rows(ON, overrides=overrides))["seeds_used"], [61, 73, 89])

    def test_noninferiority_flag(self):
        # Complete-on loses one recovery in one seed of one f1 item: S2 = -(1/3)/2 / 3 = -1/18 (about -5.6 points).
        on = {key: dict(value) for key, value in ON.items()}
        on[("i1", "f1")] = {"vcomplete": [0, 1, 1], "omit_node": [1, 1, 1]}
        s2 = self.analyse(arm_rows(OFF), arm_rows(on))["S2_complete_on_minus_off"]
        self.assertEqual((s2["estimate_exact"], s2["noninferiority_margin"]), ("-1/18", -0.03))
        self.assertIs(s2["lower_bound_above_margin"], s2["ci_low"] > -0.03)
        self.assertFalse(s2["lower_bound_above_margin"])

    def test_precision_bounds_over_submitted_cells(self):
        result = self.analyse(arm_rows(OFF), arm_rows(ON))
        lower = result["S5_precision"]["precision_lower"]
        self.assertEqual(lower["complete_off"]["estimate_exact"], "1/1")
        self.assertEqual(lower["incomplete_off"]["n_families"], 0)       # never submitted
        self.assertEqual(lower["incomplete_on"]["n_families"], 2)        # f1 and f2 submitted

    def test_refusals(self):
        good_off, good_on = arm_rows(OFF), arm_rows(ON)
        cases = {
            "different items files": (good_off, [dict(r, items_sha256="2" * 64) if r.get("block") == "LEDGER" else r
                                                 for r in good_on]),
            "different LEDGER cells": (good_off, [r for r in good_on if r.get("item_id") != "i6"]),
            "cells outside block LEDGER were dispatched": (good_off, good_on + [
                {"block": "P", "cell_id": "P|i1|omit_node|C1|s11", "cell_state": "completed"}]),
            "not a LEDGER cell": (good_off, good_on + [row("i1", "f1", "omit_node", 11, 0)]),
            "duplicate row": (good_off, good_on + [row("i1", "f1", "omit_node", 61, 0)]),
            "no LEDGER rows": (good_off, [{"block": "P", "cell_state": "not_dispatched"}]),
            "needs site_recovered": (good_off, [dict(r, site_recovered=None) if r.get("block") == "LEDGER" else r
                                                for r in good_on])}
        for message, (off_rows, on_rows) in cases.items():
            with self.subTest(case=message), self.assertRaisesRegex(A3.Refused, message):
                self.analyse(off_rows, on_rows)

    def cli(self, *argv):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = A3.main(list(argv))
        return code, stdout.getvalue(), stderr.getvalue()

    def run_json(self, name, ledger, policy="deadline"):
        path = self.dir / name
        path.write_text(json.dumps({"episode_config": {"final_turn_policy": policy, "evidence_ledger": ledger}}))
        return str(path)

    def test_the_command(self):
        off, on = write(self.dir, "off.jsonl", arm_rows(OFF)), write(self.dir, "on.jsonl", arm_rows(ON))
        hard_off = write(self.dir, "hard_off.jsonl", arm_rows(OFF))
        hard_on = write(self.dir, "hard_on.jsonl", arm_rows(ON))
        out = self.dir / "out"
        code, stdout, _ = self.cli("--off", off, "--on", on, "--off-run-json", self.run_json("off.json", "off"),
                                   "--on-run-json", self.run_json("on.json", "on"), "--hard-off", hard_off,
                                   "--hard-on", hard_on, "--out", str(out))
        self.assertEqual(code, 0)
        result = json.loads((out / "A3_ANALYSIS.json").read_text())
        self.assertEqual(result["main"]["M_reduction_in_omission_cost"]["estimate_exact"], "4/9")
        self.assertEqual(result["inputs"]["main_run_json"]["on"]["evidence_ledger"], "on")
        hard = result["hard_panel"]
        self.assertEqual((hard["seeds_used"], hard["seeds_considered"]), ([61], [61]))
        self.assertNotIn("primary_decision", hard)
        self.assertEqual(hard["M_reduction_in_omission_cost"]["estimate_exact"], "2/3")    # seed 61 only
        self.assertIn("no significance claim", hard["M_reduction_in_omission_cost"]["note"])
        self.assertTrue(result["label"].startswith("PRESPECIFIED"))
        self.assertIn("M_reduction_in_omission_cost | +44.4", (out / "A3_TABLES.md").read_text())
        self.assertIn("# A3 evidence ledger", stdout)
        # Never overwritten; refusals write nothing.
        self.assertEqual(self.cli("--off", off, "--on", on, "--out", str(out))[0], 2)
        for argv, message in ((["--off", off, "--on", on, "--off-run-json", self.run_json("x.json", "on"),
                                "--on-run-json", self.run_json("y.json", "on")], "is 'on', not 'off'"),
                              (["--off", off, "--on", on, "--off-run-json", self.run_json("x2.json", "off", "none"),
                                "--on-run-json", self.run_json("y2.json", "on")], "not the deadline notice"),
                              (["--off", off, "--on", on, "--off-run-json", self.run_json("x3.json", "off")],
                               "give both"),
                              (["--off", off, "--on", on, "--hard-off", hard_off], "give both")):
            with self.subTest(message=message):
                target = self.dir / ("refused-" + str(abs(hash(message))))
                code, _, stderr = self.cli(*argv, "--out", str(target))
                self.assertEqual(code, 2)
                self.assertIn(message, stderr)
                self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
