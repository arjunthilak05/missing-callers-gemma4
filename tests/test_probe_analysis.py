"""Exp A analysis v2 (D9 as amended by D11), power simulation, figures and the exposure-funnel tests.

Synthetic data with known effects and true nulls only; no research outcomes are read except the
existing authenticated SAFE reports used by the exposure-funnel integration test (skipped when
those files are absent).

Environments: the statistics, analysis and command tests are stdlib only (controller env, Python
3.11). FigureTests need matplotlib and PowerTests need numpy; neither is in the controller env, so
they skip there with an explicit reason and are run with the figures env
(/mnt/kldata/absent-edges-20261018-core2/expA/dev/integrate/envs/fig/bin/python, matplotlib 3.9.2,
numpy 2.4.6): ``<fig python> -B -m unittest tests.test_probe_analysis``.
"""
from collections import Counter
import contextlib
from fractions import Fraction
import importlib.util
import io
import itertools
import json
import math
import os
from pathlib import Path
import random
import stat
import statistics
import subprocess
import sys
import tempfile
import unittest

from absent_edges.analysis import holm_adjust
from absent_edges.data import canonical_json
from absent_edges.probe import analysis as A
from absent_edges.probe import score as S

ROOT = Path(__file__).resolve().parents[1]
FIGURES_ENV = "/mnt/kldata/absent-edges-20261018-core2/expA/dev/integrate/envs/fig/bin/python"
HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None
HAS_NUMPY = importlib.util.find_spec("numpy") is not None
NO_MATPLOTLIB = (f"matplotlib is not installed in {sys.executable}; the figure tests run in the figures env "
                 f"({FIGURES_ENV} -B -m unittest tests.test_probe_analysis)")
NO_NUMPY = (f"numpy is not installed in {sys.executable}; the power tests run in the figures env "
            f"({FIGURES_ENV} -B -m unittest tests.test_probe_analysis)")


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_test_module(name):
    spec = importlib.util.spec_from_file_location(f"_{name}", ROOT / "tests" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def brute_force(values):
    observed = sum(values)
    two = sum(1 for signs in itertools.product((1, -1), repeat=len(values))
              if abs(sum(s * v for s, v in zip(signs, values))) >= abs(observed))
    return Fraction(two, 2 ** len(values))


def fast_plan(**overrides):
    return A.synthetic_plan(**overrides)


class StatisticsTests(unittest.TestCase):
    def test_student_t_quantiles_and_tails(self):
        self.assertAlmostEqual(A.student_t_ppf(0.975, 10), 2.228138851986, places=8)
        self.assertAlmostEqual(A.student_t_ppf(0.975, 29), 2.045229642132, places=8)
        self.assertAlmostEqual(A.student_t_ppf(0.995, 3), 5.840909309734, places=7)
        self.assertAlmostEqual(A.student_t_two_sided_p(2.228138851986, 10), 0.05, places=10)
        self.assertAlmostEqual(A.student_t_two_sided_p(-2.045229642132, 29), 0.05, places=10)
        self.assertEqual(A.student_t_two_sided_p(0.0, 5), 1.0)
        for t in (0.5, 3.0, 12.0):                     # closed forms: df 1 (Cauchy) and df 2
            self.assertAlmostEqual(A.student_t_two_sided_p(t, 1), 1 - 2 / math.pi * math.atan(t), places=12)
            self.assertAlmostEqual(A.student_t_two_sided_p(t, 2), 1 - t / math.sqrt(2 + t * t), places=12)
        normal = math.erfc(3.0 / math.sqrt(2))         # large df: normal tail plus the O(1/df) correction
        correction = 2 * math.exp(-4.5) / math.sqrt(2 * math.pi) * (27 + 3) / (4 * 100000)
        self.assertAlmostEqual(A.student_t_two_sided_p(3.0, 100000), normal + correction, places=9)
        self.assertLess(A.student_t_two_sided_p(40.0, 29), 1e-25)       # no 1 - cdf cancellation

    def test_t_test_by_hand(self):
        values = [Fraction(1, 10), Fraction(3, 10), Fraction(-2, 10), Fraction(4, 10)]
        mean = 0.15
        sd = statistics.stdev([0.1, 0.3, -0.2, 0.4])
        t = mean / (sd / 2)
        out = A.t_test(values)
        self.assertEqual(out["status"], "tested")
        self.assertAlmostEqual(out["t"], t, places=12)
        self.assertEqual(out["df"], 3)
        self.assertAlmostEqual(out["p_two_sided"], 2 * (1 - A.student_t_cdf(t, 3)), places=12)
        critical = A.student_t_ppf(0.975, 3)
        self.assertAlmostEqual(out["interval"]["lower"], mean - critical * sd / 2, places=12)

    def test_zero_variance_rules(self):
        plan = fast_plan()
        study = type("S", (), {"plan": plan})()
        zero = A.t_test([0, 0, 0, 0])
        self.assertEqual((zero["status"], zero["p_two_sided"], zero["interval"]["degenerate"]), ("all_zero", 1.0, True))
        out = A.infer(study, {f"f{k}": Fraction(0) for k in range(6)})
        self.assertEqual((out["p_primary"], out["p_source"]), (1.0, "all_family_differences_zero"))
        constant = A.t_test([Fraction(1, 4)] * 6)
        self.assertEqual((constant["status"], constant["p_two_sided"]), ("constant_nonzero", None))
        out = A.infer(study, {f"f{k}": Fraction(1, 4) for k in range(6)})
        self.assertEqual(out["p_source"], "sign_flip_constant_nonzero_differences")
        self.assertAlmostEqual(out["p_primary"], 2 / 2 ** 6)             # only all-plus and all-minus vectors
        single = A.infer(study, {"f0": Fraction(1, 3)})
        self.assertEqual((single["t_test"]["status"], single["p_primary"]), ("constant_nonzero", 1.0))
        self.assertEqual(A.infer(study, {})["p_source"], "not_estimable")

    def test_tiny_exact_sign_flip_by_hand(self):
        result = A.sign_flip_test([1, 2, 3])                 # |sum| >= 6 only for +++ and ---
        self.assertEqual(result["p_exact"]["two-sided"], "1/4")
        self.assertEqual(result["method"], "exact_enumeration")
        self.assertEqual(A.sign_flip_test([0, 0, 0])["p_two_sided"], 1.0)
        self.assertEqual(A.sign_flip_test([2, 0, 0])["p_two_sided"], 1.0)   # zeros stay in

    def test_gray_code_matches_brute_force_and_dp(self):
        rng = random.Random(3)
        for _ in range(25):
            values = [rng.randint(-6, 6) for _ in range(rng.randint(1, 10))]
            result = A.sign_flip_test(values)
            self.assertEqual(Fraction(result["p_exact"]["two-sided"]), brute_force(values), values)
            ints, _ = A._integer_values(values)
            self.assertEqual(A.sign_flip_counts_dp(ints, sum(ints))[:1], (result["hits"]["two-sided"],))

    def test_float_inputs_use_their_exact_binary_values(self):
        result = A.sign_flip_test([0.1, 0.2, -0.3])
        self.assertEqual(Fraction(result["p_exact"]["two-sided"]), brute_force([Fraction(0.1), Fraction(0.2),
                                                                                  Fraction(-0.3)]))

    def test_exact_threshold_and_monte_carlo_rule(self):
        twenty = A.sign_flip_test([1] * 19 + [-1], exact_max_families=20, draws=500, seed=1)
        self.assertEqual((twenty["method"], twenty["sign_vectors"]), ("exact_enumeration", 2 ** 20))
        many = A.sign_flip_test([1] * 20 + [-1], exact_max_families=20, draws=500, seed=1)
        self.assertEqual((many["method"], many["draws"]), ("monte_carlo", 500))
        self.assertEqual(many["p_two_sided"], (1 + many["hits"]["two-sided"]) / 501)

    def test_monte_carlo_agrees_with_exact(self):
        rng = random.Random(11)
        for n in (8, 14, 22):
            values = [Fraction(rng.randint(-5, 9), 6) for _ in range(n)]
            ints, _ = A._integer_values(values)
            two, _, _, total = A.sign_flip_counts_dp(ints, sum(ints))
            exact = two / total
            mc = A.sign_flip_test(values, exact_max_families=0, draws=200000, seed=n)["p_two_sided"]
            self.assertLess(abs(exact - mc), 4 * math.sqrt(exact * (1 - exact) / 200000) + 1e-4, (n, exact, mc))
            if n <= 20:
                self.assertAlmostEqual(A.sign_flip_test(values)["p_two_sided"], exact, places=15)

    def test_holm_matches_reference_and_keeps_a_missing_hypothesis_at_p1(self):
        rng = random.Random(5)
        for _ in range(50):
            p = {k: rng.random() ** 2 for k in ("H1", "H2", "H3")}
            self.assertEqual(A.holm(p), holm_adjust(p, family=("H1", "H2", "H3")))
        # Review finding: a not-estimable hypothesis shrank the family to m = 2. D11.2 fixes H1-H3, so it
        # enters with p = 1 (statsmodels' Holm with p = 1 gives the same values) and reports a null itself.
        out = A.holm({"H1": 0.01, "H2": None, "H3": 0.04})
        self.assertEqual(out["H2"], None)
        self.assertAlmostEqual(out["H1"], 0.03)
        self.assertAlmostEqual(out["H3"], 0.08)
        flipped = A.holm({"H1": 0.020, "H2": 0.024, "H3": None})
        self.assertAlmostEqual(flipped["H1"], 0.06)
        self.assertAlmostEqual(flipped["H2"], 0.06)
        self.assertFalse(any(value <= 0.05 for value in (flipped["H1"], flipped["H2"])))  # no rejection
        reference = holm_adjust({"H1": 0.020, "H2": 0.024, "H3": 1.0}, family=("H1", "H2", "H3"))
        self.assertEqual({k: v for k, v in flipped.items() if v is not None},
                         {k: v for k, v in reference.items() if k != "H3"})
        self.assertEqual(A.holm({"H1": None, "H2": None, "H3": None}), {"H1": None, "H2": None, "H3": None})

    def test_bootstrap_is_deterministic_and_resamples_families(self):
        values = [Fraction(k, 7) for k in range(-3, 9)]
        first = A.cluster_bootstrap(values, resamples=3000, seed=9, confidence=0.95)
        self.assertEqual(first, A.cluster_bootstrap(values, resamples=3000, seed=9, confidence=0.95))
        self.assertLess(first["lower"], float(sum(values) / len(values)))
        self.assertEqual(A.cluster_bootstrap([Fraction(1, 2)], resamples=200, seed=1, confidence=0.95)["lower"], 0.5)


class NullCalibrationTests(unittest.TestCase):
    """Under true nulls the primary t test is calibrated and the sign-flip paths are not anticonservative."""

    @staticmethod
    def null_families(rng, families, items, seeds=3):
        out = []
        for _ in range(families):
            fam_rate = rng.uniform(0.1, 0.9)
            diffs = []
            for _ in range(items):
                p = min(0.95, max(0.05, fam_rate + rng.gauss(0, 0.1)))
                a = sum(rng.random() < p for _ in range(seeds))
                b = sum(rng.random() < p for _ in range(seeds))
                diffs.append(Fraction(a - b, seeds))
            out.append(sum(diffs, Fraction(0)) / items)
        return out

    def test_t_test_null_p_values_are_uniform(self):
        rng = random.Random(20261101)
        p_values = [A.t_test(self.null_families(rng, 24, 6))["p_two_sided"] for _ in range(1500)]
        rate = sum(p <= 0.05 for p in p_values) / len(p_values)
        self.assertLess(abs(rate - 0.05), 3 * math.sqrt(0.05 * 0.95 / len(p_values)) + 0.005, rate)
        deciles = Counter(min(int(p * 10), 9) for p in p_values)
        for k in range(10):
            self.assertLess(abs(deciles[k] - 150) / 150, 0.3, (k, deciles))

    def test_exact_sign_flip_null_is_not_anticonservative(self):
        rng = random.Random(7)
        p_values = [A.sign_flip_test(self.null_families(rng, 10, 4))["p_two_sided"] for _ in range(800)]
        rate = sum(p <= 0.05 for p in p_values) / len(p_values)
        self.assertLess(rate, 0.05 + 3 * math.sqrt(0.05 * 0.95 / 800), rate)

    def test_monte_carlo_sign_flip_null(self):
        rng = random.Random(8)
        p_values = [A.sign_flip_test(self.null_families(rng, 22, 3), draws=400, seed=k)["p_two_sided"]
                    for k in range(300)]
        rate = sum(p <= 0.05 for p in p_values) / len(p_values)
        self.assertLess(rate, 0.05 + 3 * math.sqrt(0.05 * 0.95 / 300), rate)

    def test_pipeline_null_estimates_cover_zero(self):
        for seed in (1, 2):
            study = A.synthetic_study(scenario="null", families=22, items_per_family=4, rng_seed=seed)
            analysis, _ = A.run_analysis(fast_plan(), study["items"], study["score_rows"], study["policy_rows"])
            for h in analysis["primary"]["hypotheses"]:
                self.assertLess(abs(h["estimate"]), 0.12, (seed, h["id"]))
                self.assertLessEqual(h["t_test"]["interval"]["lower"], 0.0 + 1e-12, (seed, h["id"]))
                self.assertGreaterEqual(h["t_test"]["interval"]["upper"], -1e-12, (seed, h["id"]))


def _custom_study(outcomes, *, design="d11", families=("fa", "fb"), items_per_family=2, seeds=(11, 23, 47)):
    """A deterministic study: outcomes(item_index, family_index, condition, seed) -> (state, recovered)."""
    rng = random.Random(0)
    records = []
    for fi, family in enumerate(families):
        for k in range(items_per_family):
            records.append(A.synthetic_item(rng, family, k))
    raw = b"".join(json.dumps(r, sort_keys=True).encode() + b"\n" for r in records)
    sha = A.sha256_bytes(raw)
    items = {r["item_id"]: S.ItemTruth(r) for r in records}
    by_id = {r["item_id"]: r for r in records}
    index = {r["item_id"]: (k % items_per_family, families.index(r["family"])) for k, r in enumerate(records)}
    rows = {}
    for cell in S.Design(design).cells(items):
        k, fi = index[cell["item_id"]]
        state, recovered = outcomes(k, fi, cell["condition"], cell["seed"]) if cell["seed"] in seeds \
            else ("not_dispatched", False)
        rows[cell["cell_id"]] = A._synthetic_row(cell, by_id[cell["item_id"]], recovered, state=state,
                                                 exit_status="Submitted", items_sha256=sha, design=design, rng=rng,
                                                 endpoint="http://127.0.0.1:8141")
    policies = A.synthetic_study(families=1, items_per_family=1, dev_families=0)["policy_rows"]
    template = next(iter(policies.values()))
    policy_rows = {}
    for item_id, item in items.items():
        for variant in (S.NO_VARIANT,) + S.VARIANTS:
            for policy in S.POLICIES:
                policy_rows[(item_id, variant, policy)] = dict(template, item_id=item_id, family=item.family,
                                                               omitted=item.omitted, variant=variant, policy=policy,
                                                               items_sha256=sha)
    return items, rows, policy_rows


class KnownEffectTests(unittest.TestCase):
    def test_recovers_known_effects(self):
        study = A.synthetic_study(scenario="known_effect", families=24, items_per_family=6, rng_seed=12)
        analysis, cells = A.run_analysis(fast_plan(), study["items"], study["score_rows"], study["policy_rows"])
        primary = analysis["primary"]
        self.assertEqual(primary["primary_seeds"], [11, 23, 47])
        self.assertEqual(analysis["population"]["n_families"], 24)            # dev family excluded
        truth = study["true_contrasts"]
        for h in primary["hypotheses"]:
            self.assertEqual(h["status"], "confirmatory")
            self.assertAlmostEqual(h["estimate"], truth[h["id"]], delta=0.08, msg=h["id"])
            interval = h["t_test"]["interval"]
            self.assertLessEqual(interval["lower"], truth[h["id"]], h["id"])
            self.assertGreaterEqual(interval["upper"], truth[h["id"]], h["id"])
            self.assertEqual(h["p_source"], "t_test")
            self.assertEqual(h["sign_flip"]["method"], "monte_carlo")             # 24 families > 20
        self.assertLess(primary["hypotheses"][1]["holm_adjusted_p"], 1e-3)      # H2 true effect ~ +0.30
        self.assertTrue(primary["hypotheses"][1]["reject_at_alpha"])
        self.assertEqual(len(cells), len(S.Design("d11").cells({i: it for i, it in study["items"].items()
                                                               if it.role == "main"})))

    def test_exact_constructed_effect_follows_the_unit_flow(self):
        # fa: item0 C3 recovered at all seeds, C2 never -> +1; item1 C3 at 1 of 3 seeds -> +1/3; family +2/3.
        # fb: item0 C3 at seed 11 only -> +1/3; item1 no difference -> 0; family +1/6. Mean (2/3 + 1/6) / 2 = 5/12.
        def outcomes(k, fi, condition, seed):
            if condition != "C3":
                return "completed", False
            if fi == 0:
                return "completed", k == 0 or seed == 11
            return "completed", k == 0 and seed == 11
        items, rows, policies = _custom_study(outcomes)
        analysis, _ = A.run_analysis(fast_plan(min_families=2), items, rows, policies)
        h1 = analysis["primary"]["hypotheses"][0]
        self.assertEqual(h1["estimate_exact"], "5/12")
        self.assertEqual({u["family"]: u["difference_exact"] for u in h1["family_units"]}, {"fa": "2/3", "fb": "1/6"})
        self.assertAlmostEqual(h1["micro_item_estimate"], 5 / 12, places=12)
        self.assertEqual(analysis["primary"]["hypotheses"][1]["estimate_exact"], "0/1")
        self.assertEqual(analysis["primary"]["hypotheses"][1]["p_source"], "all_family_differences_zero")


class SeedAndMissingTests(unittest.TestCase):
    def analyze(self, **kwargs):
        study = A.synthetic_study(families=20, items_per_family=3, **kwargs)
        return A.run_analysis(fast_plan(), study["items"], study["score_rows"], study["policy_rows"])[0]

    def test_partial_last_seed_is_reported_separately(self):
        primary = self.analyze(partial_seed=47, rng_seed=4)["primary"]
        self.assertEqual((primary["primary_seeds"], primary["sensitivity_seeds"]), ([11, 23], [47]))
        self.assertEqual(primary["seed_completion"]["47"]["status"], "partial")
        per_seed = primary["sensitivity"]["per_seed"]
        self.assertEqual([r["seed"] for r in per_seed], [47])
        self.assertGreater(per_seed[0]["contrasts"]["H1"]["n_items"], 0)
        self.assertEqual(primary["sensitivity"]["all_seeds_with_data"]["seeds"], [11, 23, 47])
        self.assertTrue(all(h["seeds_used"] == [11, 23] for h in primary["hypotheses"]))

    def test_partial_middle_seed_cuts_the_prefix(self):
        primary = self.analyze(partial_seed=23, rng_seed=5)["primary"]
        self.assertEqual((primary["primary_seeds"], primary["sensitivity_seeds"]), ([11], [23, 47]))

    def test_partial_first_seed_leaves_no_primary_analysis(self):
        analysis = self.analyze(partial_seed=11, rng_seed=6)
        primary = analysis["primary"]
        self.assertEqual((primary["status"], primary["primary_seeds"]), ("not_estimable", []))
        self.assertTrue(all(h["status"] == "not_estimable" and h["holm_adjusted_p"] is None
                            for h in primary["hypotheses"]))
        self.assertIn("no complete P seed", analysis["secondary"]["rates_seed_note"])

    def test_missing_infrastructure_cells_use_completed_cells_and_bounds(self):
        primary = self.analyze(missing_fraction=0.08, rng_seed=7)["primary"]
        self.assertEqual(primary["primary_seeds"], [11, 23, 47])                # terminal, so complete
        for h in primary["hypotheses"]:
            bounds = h["bounds_missing_cells"]
            self.assertTrue(h["missing_cells_in_primary_seeds"], h["id"])
            self.assertLessEqual(bounds["lowest"]["estimate"], h["estimate"] + 1e-12)
            self.assertGreaterEqual(bounds["highest"]["estimate"], h["estimate"] - 1e-12)
            for label in ("impute_0", "impute_1"):
                self.assertLessEqual(bounds["lowest"]["estimate"], bounds[label]["estimate"] + 1e-12)
                self.assertGreaterEqual(bounds["highest"]["estimate"], bounds[label]["estimate"] - 1e-12)

    def test_item_means_use_completed_cells_only(self):
        # Item 0 of each family: C3 missing (infrastructure) at seed 47 and recovered at 11 and 23 -> mean 1.
        def outcomes(k, fi, condition, seed):
            if condition == "C3" and k == 0 and seed == 47:
                return "missing_infrastructure", False
            return "completed", condition == "C3" and k == 0
        items, rows, policies = _custom_study(outcomes)
        h1 = A.run_analysis(fast_plan(min_families=2), items, rows, policies)[0]["primary"]["hypotheses"][0]
        self.assertEqual(h1["estimate_exact"], "1/2")                          # (1 + 0) / 2 per family
        self.assertEqual(h1["missing_cells_in_primary_seeds"], {"C3:missing_infrastructure": 2})
        self.assertAlmostEqual(h1["bounds_missing_cells"]["impute_0"]["estimate"], (2 / 3 + 0) / 2)
        self.assertAlmostEqual(h1["bounds_missing_cells"]["impute_1"]["estimate"], 0.5)

    def test_a_not_estimable_h3_keeps_the_holm_family_at_three(self):
        # Review finding (dataset D10): every C0 cell missing (infrastructure) leaves the seeds primary and H3
        # without families; Holm then adjusted H1 and H2 with m = 2. D11.2 fixes the family as H1-H3.
        def outcomes(k, fi, condition, seed):
            if condition == "C0":
                return "missing_infrastructure", False
            if condition == "C3":
                return "completed", (k + fi + seed) % 3 != 0
            if condition == "C4a":
                return "completed", k == 0 or (fi == 1 and seed == 11)
            return "completed", False
        items, rows, policies = _custom_study(outcomes, families=("fa", "fb", "fc"))
        analysis, _ = A.run_analysis(fast_plan(min_families=2), items, rows, policies)
        primary = analysis["primary"]
        h1, h2, h3 = primary["hypotheses"]
        self.assertEqual(primary["primary_seeds"], [11, 23, 47])
        self.assertEqual((h3["status"], h3["p_primary"], h3["holm_adjusted_p"], h3["reject_at_alpha"]),
                         ("not_estimable", None, None, None))
        self.assertIsNotNone(h1["p_primary"])
        self.assertIsNotNone(h2["p_primary"])
        self.assertEqual((primary["holm_family"], primary["holm_entered_as_p1"]), (["H1", "H2", "H3"], ["H3"]))
        reference = holm_adjust({"H1": h1["p_primary"], "H2": h2["p_primary"], "H3": 1.0},
                                family=("H1", "H2", "H3"))
        self.assertEqual((h1["holm_adjusted_p"], h2["holm_adjusted_p"]), (reference["H1"], reference["H2"]))
        flips = holm_adjust({"H1": h1["sign_flip"]["p_two_sided"], "H2": h2["sign_flip"]["p_two_sided"], "H3": 1.0},
                            family=("H1", "H2", "H3"))
        self.assertEqual((h1["sign_flip_holm_adjusted_p"], h2["sign_flip_holm_adjusted_p"], h3["sign_flip_holm_adjusted_p"]),
                         (flips["H1"], flips["H2"], None))
        self.assertIn("enters Holm with p = 1", primary["multiplicity"])
        tables = A.render_tables_md(analysis)
        self.assertIn("Holm family H1, H2, H3 (not estimable, entered with p = 1: H3)", tables)
        self.assertNotIn("Holm across H1-H3", tables)

    def test_pending_cells_make_a_seed_partial(self):
        def outcomes(k, fi, condition, seed):
            return ("infra_pending" if (seed == 47 and k == 1 and condition == "C0") else "completed"), False
        items, rows, policies = _custom_study(outcomes)
        primary = A.run_analysis(fast_plan(min_families=2), items, rows, policies)[0]["primary"]
        self.assertEqual(primary["primary_seeds"], [11, 23])


class RuleTests(unittest.TestCase):
    def test_fewer_than_twenty_families_is_exploratory(self):
        study = A.synthetic_study(families=6, items_per_family=4, rng_seed=3)
        primary = A.run_analysis(fast_plan(), study["items"], study["score_rows"], study["policy_rows"])[0]["primary"]
        for h in primary["hypotheses"]:
            self.assertEqual((h["status"], h["confirmatory"], h["reject_at_alpha"]),
                             ("exploratory_fewer_than_minimum_families", False, None))
            self.assertIsNotNone(h["holm_adjusted_p"])
            self.assertEqual(h["sign_flip"]["method"], "exact_enumeration")

    def test_contract_v2_layout_uses_seed_11_for_h3(self):
        study = A.synthetic_study(families=20, items_per_family=3, rng_seed=9, design="contract-v2")
        analysis, _ = A.run_analysis(fast_plan(design="contract-v2"), study["items"], study["score_rows"],
                                     study["policy_rows"])
        h3 = analysis["primary"]["hypotheses"][2]
        self.assertEqual((h3["control_block"], h3["seeds_used"]), ("S-C0", [11]))
        self.assertIn("not the D11 estimand", h3["estimand_note"])
        self.assertIsNotNone(analysis["secondary"]["s_c0"]["P_C1_minus_S_C0_seed_11"])

    def test_score_rows_are_validated(self):
        study = A.synthetic_study(families=2, items_per_family=1, dev_families=0)
        rows = list(study["score_rows"].values())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.jsonl"
            bad = dict(next(r for r in rows if r["completed"]), exit_status="NoAnswer", site_recovered=1)
            path.write_bytes(b"".join(canonical_json(r) + b"\n" for r in [bad]))
            with self.assertRaisesRegex(ValueError, "non-submitted"):
                A.load_score_rows(path)
            bad = dict(next(r for r in rows if r["completed"]), cell_state="missing_infrastructure")
            path.write_bytes(canonical_json(bad) + b"\n")
            with self.assertRaisesRegex(ValueError, "disagree"):
                A.load_score_rows(path)

    def test_plan_is_validated_and_frozen(self):
        plan = A.plan_from_dict(A.default_plan_dict())
        self.assertEqual((plan.bootstrap_resamples, plan.sign_flip_draws, plan.bootstrap_seed, plan.sign_flip_seed,
                          plan.min_families_confirmatory, plan.design), (10000, 100000, 20261101, 20261101, 20, "d11"))
        for change, pattern in (({"primary": {"block": "P"}}, "frozen"), ({"p_seeds": [23, 47]}, "prefix"),
                                ({"design": "v1"}, "design"), ({"extra": 1}, "unknown"),
                                ({"secondary_seed": 23}, "secondary")):
            with self.assertRaisesRegex(ValueError, pattern):
                A.plan_from_dict(A.default_plan_dict(**change))

    def test_secondary_reports_are_descriptive(self):
        study = A.synthetic_study(families=20, items_per_family=4, rng_seed=21)
        analysis, _ = A.run_analysis(fast_plan(), study["items"], study["score_rows"], study["policy_rows"])
        rates = {(r["block"], r["condition"]): r for r in analysis["secondary"]["rates"]}
        self.assertEqual(set(rates), {("P", c) for c in ("C1", "C2", "C3", "C4a", "C0")}
                         | {("S-VC", "C1"), ("S-VC", "C3"), ("S-NAT", "C1"), ("S-NAT", "C3"), ("S-NAT", "C4a"),
                            ("S-EDGE", "C1"), ("S-EDGE", "C3"), ("T", "T-offered"), ("T", "T-recommended")})
        self.assertIn("nm_any", rates[("S-NAT", "C1")])
        self.assertNotIn("p_value", json.dumps(analysis["secondary"]))
        self.assertEqual(set(rates[("P", "C1")]["reference_policies"]), set(S.POLICIES))
        h2ref = analysis["reference_policies"]["h2_vs_c4_hits_reference"]
        self.assertEqual(h2ref["status"], "estimated")
        self.assertIn("not ceilings", h2ref["note"])
        self.assertEqual(analysis["secondary"]["s_c0"]["status"], "not_in_design")
        self.assertTrue(analysis["endpoints"]["cells"])
        self.assertEqual(set(analysis["secondary"]["strata"]) - {"role", "seeds"},
                         {"stratum", "grep_recoverability", "c4_visibility", "wording_pair"})
        counts = {r["condition"]: r["counts"] for r in analysis["exposure_audit"]["rows"] if r["block"] == "P"}
        self.assertNotIn("verification_contradicts_condition", counts["C4a"])
        self.assertNotIn("payload_contradicts_condition", counts["C0"])


class ChainTests(unittest.TestCase):
    """scripts/expA_score.py -> scripts/expA_analyze.py on the hand-made scoring fixture (real files)."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.mod = _load_test_module("test_probe_score")
        self.fixture = self.mod.Fixture(Path(self.temporary.name))
        self.score_cli, self.analyze_cli = _load_script("expA_score"), _load_script("expA_analyze")
        f = self.fixture
        f.result("P|item-a|omit_node|C1|s11", 1, "Submitted", [{"path": "pkg/core.py", "line": 6}], "yes")
        f.result("P|item-a|omit_node|C3|s11", 1, "Submitted", [{"path": "pkg/other.py", "line": 5}], "no")
        f.result("P|item-a|none|C0|s11", 1, "NoAnswer")
        f.result("P|item-b|omit_node|C4a|s11", 1, "InfraInterrupted", exposure=None)
        f.result("P|item-b|omit_node|C4a|s11", 2, "Submitted", [{"path": "pkg/other.py", "line": 5}], "unknown")
        self.out = f.base / "scored"
        self.common = ["--items", f.items_dir / "items.jsonl", "--repos-root", f.repos, "--dispatch-dir",
                       f.dispatch_dir]
        self.call(self.score_cli, "score", "--run-dir", f.run, *self.common, "--out", self.out / "scores.jsonl")
        self.call(self.score_cli, "policies", *self.common, "--out", self.out / "policies.jsonl")
        self.plan = f.base / "plan.json"
        self.call(self.analyze_cli, "plan", "--out", self.plan, "--label", "TEST FIXTURE (not research data)")

    def tearDown(self):
        for current, _, files in os.walk(self.temporary.name):
            for name in files:
                os.chmod(Path(current) / name, stat.S_IWRITE | stat.S_IREAD)
        self.temporary.cleanup()

    @staticmethod
    def call(module, *args):
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            module.main([str(a) for a in args])
        return stdout.getvalue()

    def analyze_args(self, out, items=None):
        f = self.fixture
        return ["analyze", "--items", items or f.items_dir / "items.jsonl", "--scores", self.out / "scores.jsonl",
                "--policies", self.out / "policies.jsonl", "--prereg", self.plan, "--run-dir", f.run,
                "--dispatch-dir", f.dispatch_dir, "--out", out]

    def test_one_command_chain_is_bound_and_deterministic(self):
        f = self.fixture
        summary = json.loads(self.call(self.analyze_cli, *self.analyze_args(f.base / "a1")))
        self.call(self.analyze_cli, *self.analyze_args(f.base / "a2"))
        for name in ("ANALYSIS.json", "TABLES.md", "CELLS.jsonl"):
            self.assertEqual((f.base / "a1" / name).read_bytes(), (f.base / "a2" / name).read_bytes(), name)
        analysis = json.loads((f.base / "a1" / "ANALYSIS.json").read_bytes())
        self.assertEqual(analysis["binding"]["terminal_results_verified"], 4)
        self.assertEqual(analysis["binding"]["scores_manifest_items_sha256"], f.items_sha)
        self.assertEqual(analysis["primary"]["seed_completion"]["11"]["status"], "partial")
        self.assertEqual(summary["primary_status"], "not_estimable")
        rates = {(r["block"], r["condition"]): r for r in analysis["secondary"]["rates"]}
        self.assertEqual(rates[("P", "C3")]["site_recovered"]["estimate"], 1.0)
        self.assertEqual(rates[("P", "C1")]["certifiably_false_completeness"]["count"], 1)
        self.assertEqual(analysis["secondary"]["s_nat"]["items_without_NM"], [])
        with self.assertRaises(FileExistsError):
            self.call(self.analyze_cli, *self.analyze_args(f.base / "a1"))
        with self.assertRaisesRegex(ValueError, "does not match the pin"):
            self.call(self.analyze_cli, *self.analyze_args(f.base / "a3"), "--prereg-sha256", "0" * 64)

    def test_analysis_refuses_a_rebuilt_items_file_and_tampered_runs(self):
        f = self.fixture
        rebuilt = json.loads(json.dumps(f.records))
        rebuilt["item-a"]["omitted"]["identity"] = self.mod.CALLER
        path = f.base / "rebuilt" / "items.jsonl"
        path.parent.mkdir()
        f.write_items(path, rebuilt)
        with self.assertRaisesRegex(S.BindingError, "differs from run.json"):
            self.call(self.analyze_cli, *self.analyze_args(f.base / "r1", items=path))
        f.result("P|item-b|omit_node|C1|s11", 1, "NoAnswer")               # a result the scores do not know
        with self.assertRaisesRegex(S.BindingError, "unscored"):
            self.call(self.analyze_cli, *self.analyze_args(f.base / "r2"))

    def test_stale_non_completed_cell_states_are_refused(self):
        # Review finding (t4_rundir): scores written while a cell was infra_pending (2 InfraInterrupted attempts)
        # were accepted against the final run, where a 3rd attempt made it missing (infrastructure); the primary
        # seed set changed without an error. check_binding now resolves every design cell from the run.
        f = self.fixture
        cell = "P|item-b|omit_node|C2|s11"
        f.result(cell, 1, "InfraInterrupted", exposure=None)
        f.result(cell, 2, "InfraInterrupted", exposure=None)
        early = f.base / "early"
        self.call(self.score_cli, "score", "--run-dir", f.run, *self.common, "--out", early / "scores.jsonl")
        rows = {json.loads(line)["cell_id"]: json.loads(line)
                for line in (early / "scores.jsonl").read_text().splitlines()}
        self.assertEqual((rows[cell]["cell_state"], rows[cell]["infra_attempts"]), ("infra_pending", 2))
        f.result(cell, 3, "InfraInterrupted", exposure=None)
        stale = self.analyze_args(f.base / "stale")
        stale[stale.index("--scores") + 1] = early / "scores.jsonl"
        with self.assertRaisesRegex(S.BindingError, "disagree with the run's cell states"):
            self.call(self.analyze_cli, *stale)
        self.assertFalse((f.base / "stale").exists())
        final = f.base / "final"
        self.call(self.score_cli, "score", "--run-dir", f.run, *self.common, "--out", final / "scores.jsonl")
        fresh = self.analyze_args(f.base / "fresh")
        fresh[fresh.index("--scores") + 1] = final / "scores.jsonl"
        self.call(self.analyze_cli, *fresh)
        analysis = json.loads((f.base / "fresh" / "ANALYSIS.json").read_bytes())
        self.assertEqual(analysis["binding"]["cell_states_verified"], len(rows))
        cells = {json.loads(line)["cell_id"]: json.loads(line)
                 for line in (final / "scores.jsonl").read_text().splitlines()}
        self.assertEqual((cells[cell]["cell_state"], cells[cell]["infra_attempts"]), ("missing_infrastructure", 3))
        # A cell first dispatched after scoring is refused the same way (not_dispatched -> infra_pending).
        f.result("P|item-b|omit_node|C3|s23", 1, "InfraInterrupted", exposure=None)
        late = self.analyze_args(f.base / "late")
        late[late.index("--scores") + 1] = final / "scores.jsonl"
        with self.assertRaisesRegex(S.BindingError, "1 score rows disagree"):
            self.call(self.analyze_cli, *late)

    def test_findability_report_needs_no_outcomes(self):
        f = self.fixture
        self.call(self.analyze_cli, "findability", "--items", f.items_dir / "items.jsonl", "--policies",
                  self.out / "policies.jsonl", "--out", f.base / "find")
        report = json.loads((f.base / "find" / "FINDABILITY.json").read_bytes())
        self.assertEqual(report["findability"]["grep_recovers_omitted"]["count"], 2)
        self.assertEqual(report["reference_policies"]["omit_node"]["answer_only"]["site_recovered"], 0.0)
        self.assertEqual(report["reference_policies"]["omit_node"]["c4_hits"]["site_recovered"], 1.0)
        self.assertIn("# Exp A findability", (f.base / "find" / "FINDABILITY.md").read_text())
        self.assertIsNone(report["eligibility_flow"])                     # no items manifest next to the items

    def test_findability_reports_the_eligibility_flow_of_its_own_items(self):
        f = self.fixture
        funnel = {"functions": 9, "eligible": 1, "selected": 1, "excluded": {"dunder": 3, "verified_below_2": 5}}
        manifest = {"outputs": {"items.jsonl": {"sha256": f.items_sha}},
                    "families": {"fam-a": {"role": "main", "funnel": funnel, "short_of_m": True},
                                 "fam-b": {"role": "main", "funnel": dict(funnel, selected=1), "short_of_m": True},
                                 "fam-c": {"role": "main", "funnel": dict(funnel, eligible=0, selected=0),
                                           "short_of_m": True},
                                 "dev-x": {"role": "dev", "funnel": funnel}}}
        path = f.base / "items-manifest.json"
        path.write_text(json.dumps(manifest))
        args = ["findability", "--items", f.items_dir / "items.jsonl", "--policies", self.out / "policies.jsonl",
                "--items-manifest", path]
        self.call(self.analyze_cli, *args, "--out", f.base / "find")
        flow = json.loads((f.base / "find" / "FINDABILITY.json").read_bytes())["eligibility_flow"]
        self.assertEqual(sorted(flow["families"]), ["fam-a", "fam-b", "fam-c"])     # dev families excluded
        self.assertEqual(flow["families_without_items"], ["fam-c"])
        self.assertIn("Eligibility flow per family", (f.base / "find" / "FINDABILITY.md").read_text())
        manifest["outputs"]["items.jsonl"]["sha256"] = "4" * 64
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(S.BindingError, "items manifest describes"):
            self.call(self.analyze_cli, *args, "--out", f.base / "find2")


class SyntheticCommandTests(unittest.TestCase):
    def test_synthetic_fixture_and_unbound_analysis_are_deterministic(self):
        cli = _load_script("expA_analyze")
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                cli.main(["synthetic", "--out", str(tmp / "syn"), "--families", "21", "--items-per-family", "2",
                          "--partial-seed", "47"])
                args = ["analyze", "--items", str(tmp / "syn" / "items.jsonl"), "--scores",
                        str(tmp / "syn" / "scores.jsonl"), "--policies", str(tmp / "syn" / "policies.jsonl"),
                        "--prereg", str(tmp / "syn" / "plan.json"), "--unbound-synthetic"]
                cli.main(args + ["--out", str(tmp / "o1")])
                cli.main(args + ["--out", str(tmp / "o2")])
            for name in ("ANALYSIS.json", "TABLES.md", "CELLS.jsonl"):
                self.assertEqual((tmp / "o1" / name).read_bytes(), (tmp / "o2" / name).read_bytes(), name)
            analysis = json.loads((tmp / "o1" / "ANALYSIS.json").read_bytes())
            self.assertIn("SYNTHETIC", analysis["label"])
            self.assertEqual(analysis["primary"]["primary_seeds"], [11, 23])
            self.assertEqual(analysis["plan"]["bootstrap"]["resamples"], 10000)


@unittest.skipUnless(HAS_MATPLOTLIB, NO_MATPLOTLIB)
class FigureTests(unittest.TestCase):
    def test_figures_are_deterministic(self):
        study = A.synthetic_study(families=20, items_per_family=3, rng_seed=31, partial_seed=47)
        analysis, cells = A.run_analysis(fast_plan(), study["items"], study["score_rows"], study["policy_rows"])
        figures = _load_script("expA_figures")
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            A.write_outputs(tmp / "a", analysis, cells)
            for out in ("f1", "f2"):
                with contextlib.redirect_stdout(io.StringIO()):
                    figures.main(["--analysis", str(tmp / "a" / "ANALYSIS.json"), "--out", str(tmp / out)])
            names = sorted(p.name for p in (tmp / "f1").iterdir())
            self.assertEqual(names, ["FIGURES.json"] + [f"{stem}.{ext}" for stem in ("fig1_primary", "fig2_recovery",
                                                                                     "fig3_cost")
                                                        for ext in ("png", "svg")])
            for name in names:
                data = (tmp / "f1" / name).read_bytes()
                self.assertGreater(len(data), 500, name)
                self.assertEqual(data, (tmp / "f2" / name).read_bytes(), name)


@unittest.skipUnless(HAS_NUMPY, NO_NUMPY)
class PowerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy
        cls.np = numpy
        cls.power = _load_script("expA_power")

    def config(self, **overrides):
        params = self.power.default_params()
        params.update(families=22, items_per_family=6, simulations=overrides.pop("simulations", 1500), scenarios=[])
        params.update(overrides)
        return self.power.resolve(params)

    def test_global_null_controls_the_familywise_error(self):
        config = self.config(simulations=4000)
        sim = self.power.Simulator(self.np, config, seed=1)
        reject = sim.rejections({"H1": 0.0, "H2": 0.0, "H3": 0.0})
        fwer = float(reject.any(axis=1).mean())
        self.assertLess(fwer, 0.05 + 3 * math.sqrt(0.05 * 0.95 / 4000), fwer)
        self.assertGreater(fwer, 0.02, fwer)

    def test_p_values_and_holm_match_the_analysis_module(self):
        config = self.config(simulations=200)
        sim = self.power.Simulator(self.np, config, seed=2)
        diffs = sim.family_differences({"H1": 0.1, "H2": 0.2, "H3": -0.1})
        p = sim.p_values(diffs)
        reject = sim.rejections({"H1": 0.1, "H2": 0.2, "H3": -0.1})
        for k in range(40):
            exact = {h: A.t_test([float(v) for v in diffs[k, :, j]])["p_two_sided"]
                     for j, h in enumerate(("H1", "H2", "H3"))}
            for j, h in enumerate(("H1", "H2", "H3")):
                self.assertAlmostEqual(float(p[k, j]), exact[h], delta=2e-6)
            adjusted = A.holm(exact)
            for j, h in enumerate(("H1", "H2", "H3")):
                self.assertEqual(bool(reject[k, j]), adjusted[h] <= 0.05, (k, h, adjusted))

    def test_power_grows_with_the_effect_and_the_mde_brackets_the_target(self):
        config = self.config()
        sim = self.power.Simulator(self.np, config, seed=3)
        cache = {}
        curve = [self.power.evaluate(sim, config, "H2", effect, cache) for effect in (0.0, 0.05, 0.1, 0.2)]
        self.assertEqual(curve, sorted(curve))
        self.assertGreater(curve[-1], 0.9)
        mde = self.power.minimal_detectable_effect(sim, config, "H2", 1, cache)
        self.assertGreaterEqual(mde["power_at_mde"], 0.8)
        self.assertLess(mde["power_just_below"], 0.8)
        self.assertLessEqual(mde["bracket"][1] - mde["bracket"][0], config["mde"]["tolerance"])

    def test_unequal_base_rates_drive_the_simulation_and_are_recorded(self):
        # Review finding (t5_power_project): C0, C3 and C4a base rates were range-checked, then silently replaced
        # by the anchors plus 'effects', while POWER.json echoed the input rates.
        flat = {"C0": 0.4, "C1": 0.4, "C2": 0.4, "C3": 0.4, "C4a": 0.4}
        pilot = {"C0": 0.05, "C1": 0.4, "C2": 0.4, "C3": 0.95, "C4a": 0.95}
        a = self.config(simulations=300, base_rates=flat, effects={})
        b = self.config(simulations=300, base_rates=pilot, effects={})
        self.assertEqual(a["effects"], {"H1": 0.0, "H2": 0.0, "H3": 0.0})
        for h, value in {"H1": 0.55, "H2": 0.55, "H3": 0.35}.items():
            self.assertAlmostEqual(b["effects"][h], value)
        ra, rb = self.power.run_configuration(self.np, a, 5), self.power.run_configuration(self.np, b, 5)
        for condition, rate in pilot.items():
            self.assertAlmostEqual(rb["simulated_arm_rates"]["base_rates"][condition], rate)
        self.assertEqual(rb["simulated_arm_rates"]["global_null"], {"C0": 0.4, "C1": 0.4, "C2": 0.4, "C3": 0.4,
                                                                     "C4a": 0.4})
        self.assertEqual(ra["familywise_error_global_null"], rb["familywise_error_global_null"])  # same null arms
        self.assertLess(ra["rejection_rate_at_base_rates"]["H1"], 0.2)
        self.assertGreater(rb["rejection_rate_at_base_rates"]["H1"], 0.95)
        self.assertNotEqual(json.dumps(ra["contrasts"], sort_keys=True), json.dumps(rb["contrasts"], sort_keys=True))
        arms = rb["contrasts"]["H2"]["arm_rates_at_meaningful_effect"]
        self.assertAlmostEqual(arms["C4a"], 0.5)                       # only H2 varies ...
        self.assertAlmostEqual(arms["C3"], 0.95)                       # ... H1 and H3 keep their pilot values
        self.assertAlmostEqual(arms["C0"], 0.05)
        md = self.power.render_md({"schema_version": "x", "params": {"label": "t", "alpha": 0.05, "power_target": 0.8},
                                   "params_sha256": "0" * 64, "numpy": "n", "scenarios": [rb]})
        self.assertIn("0.050/0.400/0.400/0.950/0.950", md)

    def test_effects_that_contradict_the_base_rates_are_refused(self):
        rates = {"C0": 0.30, "C1": 0.40, "C2": 0.40, "C3": 0.40, "C4a": 0.40}
        with self.assertRaisesRegex(ValueError, "derived from base_rates"):
            self.config(base_rates=rates, effects={"H1": 0.0, "H2": 0.0, "H3": 0.0})   # the e02 illustrative file
        config = self.config(base_rates=rates, effects={"H3": 0.1})
        self.assertAlmostEqual(config["effects"]["H3"], 0.1)
        with self.assertRaisesRegex(ValueError, "derived from base_rates"):
            self.config(effects={"H2": 0.1})                                         # flat 0.5 rates imply 0

    def test_fewer_families_than_the_minimum_have_no_confirmatory_power(self):
        # Review finding (t8/power15): with F < min_families_confirmatory the analysis never rejects, but the
        # power table reported Holm rejection rates as power.
        config = self.config(simulations=300, families=15)
        result = self.power.run_configuration(self.np, config, 7)
        self.assertFalse(result["confirmatory_possible"])
        self.assertIn("exploratory", result["minimum_data_rule"])
        self.assertEqual(result["familywise_error_global_null"], 0.0)
        for contrast in ("H1", "H2", "H3"):
            entry = result["contrasts"][contrast]
            self.assertEqual(entry["power_at_meaningful_effect"], 0.0)
            self.assertGreater(entry["holm_rejection_rate_at_meaningful_effect"], 0.0)
            self.assertIsNone(entry["mde_expected_direction"]["mde"])
            self.assertIn("exploratory", entry["mde_expected_direction"]["reason"])
            self.assertIn("mde", entry["mde_expected_direction_ignoring_minimum_data"])
        md = self.power.render_md({"schema_version": "x", "params": {"label": "t", "alpha": 0.05, "power_target": 0.8},
                                   "params_sha256": "0" * 64, "numpy": "n", "scenarios": [result]})
        row = next(line for line in md.splitlines() if line.startswith("| base |"))
        self.assertIn("no: exploratory (D11.3)", row)
        self.assertIn("0 (exploratory)", row)
        self.assertNotIn("0.988", row)
        enough = self.power.run_configuration(self.np, self.config(simulations=300, families=20), 7)
        self.assertTrue(enough["confirmatory_possible"])
        self.assertGreater(enough["contrasts"]["H2"]["power_at_meaningful_effect"], 0.0)

    def test_report_is_deterministic_and_handles_saturation(self):
        params = self.power.default_params()
        params.update(families=21, items_per_family=4, simulations=300)
        params["scenarios"] = [{"name": "sat", "base_rate": 0.9}, {"name": "mid", "base_rate": 0.5,
                                                                   "effect_heterogeneity_sd": 0.3}]
        first, second = self.power.run(params), self.power.run(params)
        self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True))
        sat = first[0]["contrasts"]["H1"]
        self.assertIsNone(sat["power_at_meaningful_effect"])          # 0.9 + 0.10 leaves the rate bounds
        self.assertIsNotNone(sat["mde_opposite_direction"]["mde"])
        self.assertTrue(first[0]["confirmatory_possible"])


class ExposureFunnelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.funnel = _load_script("exposure_funnel")

    def test_keypath_and_untraceable_numbers(self):
        f = self.funnel
        self.assertEqual(f.keypath(("a", 0, "*", "b.c", "d")), 'a[0][*]["b.c"].d')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.json"
            path.write_text(json.dumps({"a": {"b": 3, "c": "text"}}))
            source = f.Source(Path(tmp), "x.json", {"E9"}, A.sha256_bytes(path.read_bytes()))
            numbers = f.Numbers()
            self.assertEqual(numbers.direct("ok", source, ("a", "b")), 3)
            self.assertIsNone(numbers.direct("gone", source, ("a", "zz")))
            self.assertIsNone(numbers.direct("text", source, ("a", "c")))
            self.assertEqual(numbers.records["ok"]["source"]["key_path"], "a.b")
            self.assertEqual(numbers.records["gone"]["status"], "untraceable")
            self.assertIn("absent", numbers.records["gone"]["reason"])
            self.assertEqual(numbers.records["text"]["status"], "untraceable")

    def test_sha256_mismatch_aborts(self):
        f = self.funnel
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "x.json").write_text("{}")
            evidence = {"data": {"sources": {"E9": [{"path": "x.json", "sha256": "0" * 64}]}}}
            with self.assertRaises(SystemExit):
                f.resolve_sources(Path(tmp), [evidence, evidence], ["E9"])
            other = {"data": {"sources": {"E9": [{"path": "x.json", "sha256": "1" * 64}]}}}
            with self.assertRaises(SystemExit):
                f.resolve_sources(Path(tmp), [evidence, other], ["E9"])

    @unittest.skipUnless(all((ROOT / p).is_file() for p in (
        "staged/private-paper-evidence-audit-v1/evidence-map.safe.json",
        "staged/private-paper-progress-v14/evidence-map.safe.json",
        "staged/cap180-analysis-actions-v7/actual-fetch-a/report.safe.json",
        "reports/D90_NATIVE_CONTEXT_DIAGNOSTIC_20261017.safe.json")), "authenticated evidence files not present")
    def test_real_funnel_is_fully_traced(self):
        report = self.funnel.build(ROOT)
        summary = report["summary"]
        self.assertEqual(summary["untraceable"], 0)
        self.assertEqual(summary["cross_check_disagreements"], 0)
        self.assertEqual(summary["hash_chain_checks_failed"], 0)
        self.assertEqual(summary["claims"]["mismatch"], 0)
        self.assertEqual(summary["claims"]["fragment_not_found"], 0)
        self.assertTrue(all(s["verified"] for s in report["sources"]))
        numbers = {r["id"]: r for r in report["numbers"]}
        for nid, value in (("D90.graph_query.requested", 0), ("D90.source_reference_search.executed", 0),
                           ("D90.bash.executed", 1822), ("D90.graph_query.offered_episodes", 72),
                           ("Cap180.all.graph_query.offered_episodes", 144), ("Cap180.all.model_calls", 4408),
                           ("D90+Cap180.model_calls", 6291), ("D90+Cap180.graph_query.consumed_completed_request", 0)):
            self.assertEqual(numbers[nid]["value"], value, nid)
        for record in report["numbers"]:
            if record["status"] == "traced":
                self.assertTrue(record["source"]["file"] and len(record["source"]["sha256"]) == 64 and record["source"]["key_path"])
            else:
                self.assertEqual(record["status"], "derived")
                self.assertTrue(record["sources"] and record["derivation"])


if __name__ == "__main__":
    unittest.main()
