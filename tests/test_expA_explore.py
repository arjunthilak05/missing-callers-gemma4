"""draft SYNTHETIC fixtures only: no research outcome is read.

Tests for scripts/expA_explore.py (EXPLORATORY, post-results analysis package): the family-weighted
unit (item mean over the available seeds -> family mean -> equal-weight mean over families), the
family-clustered Student t interval (df = F - 1), paired contrasts, conditioning on submission, the
|V| tertile rule, seed concordance and the sha256 pin. Every fixture below is SYNTHETIC.

    python -I -B tests/test_expA_explore.py
"""
from fractions import Fraction
import hashlib
import importlib.util
import math
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_SPEC = importlib.util.spec_from_file_location("expA_explore_under_test", ROOT / "scripts" / "expA_explore.py")
E = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(E)

SYNTHETIC = "SYNTHETIC FIXTURE (tests/test_expA_explore.py) - not research data"
# Student t 0.975 quantiles, reference values computed with scipy.stats.t.ppf(0.975, df) on 2026-11-01.
T975 = {1: 12.706204736174694, 2: 4.302652729749462, 3: 3.1824463052837078}


def synthetic_row(recovered, *, submitted=True, calls=5, prompt=1000, completion=100):
    """One SYNTHETIC episode row with the fields the explore script reads."""
    return {"label": SYNTHETIC, "exit_status": "Submitted" if submitted else "NoAnswer",
            "site_recovered": int(recovered) if submitted else 0,
            "cost": {"calls": calls if submitted else 8, "prompt_tokens": prompt, "completion_tokens": completion,
                     "tool_calls": {"bash": 1}}}


# SYNTHETIC: three families of unequal size; item fb/i3 has only two available seeds.
SYNTHETIC_ARM = {
    "i1": [synthetic_row(1), synthetic_row(1), synthetic_row(0)],   # fa: item mean 2/3
    "i2": [synthetic_row(0), synthetic_row(0), synthetic_row(0)],   # fa: 0
    "i3": [synthetic_row(1), synthetic_row(1)],                     # fb: 1 (two seeds)
    "i4": [synthetic_row(1), synthetic_row(1), synthetic_row(1)],   # fc: 1
    "i5": [synthetic_row(0), synthetic_row(1), synthetic_row(0)],   # fc: 1/3
    "i6": [synthetic_row(0), synthetic_row(0), synthetic_row(0)],   # fc: 0
}
SYNTHETIC_FAMILY = {"i1": "fa", "i2": "fa", "i3": "fb", "i4": "fc", "i5": "fc", "i6": "fc"}


class FamilyWeightedUnitTests(unittest.TestCase):
    def test_item_mean_then_family_mean_then_equal_weight(self):
        values, total, count = E.item_means(SYNTHETIC_ARM, E.recovered)
        self.assertEqual(values, {"i1": Fraction(2, 3), "i2": Fraction(0), "i3": Fraction(1), "i4": Fraction(1),
                                  "i5": Fraction(1, 3), "i6": Fraction(0)})
        self.assertEqual((total, count), (Fraction(8), 17))
        out = E.family_weighted(values, SYNTHETIC_FAMILY, families=True)
        # families: fa = (2/3 + 0)/2 = 1/3, fb = 1, fc = (1 + 1/3 + 0)/3 = 4/9; mean = 16/27
        self.assertEqual(out["family_values"], {"fa": 1 / 3, "fb": 1.0, "fc": 4 / 9})
        self.assertEqual(out["estimate_exact"], "16/27")
        self.assertAlmostEqual(out["estimate"], 16 / 27, places=15)
        self.assertEqual((out["n_items"], out["n_families"]), (6, 3))
        # the unit differs from both the micro rate (8/17) and the unweighted item mean (1/2)
        rated = E.rate(SYNTHETIC_ARM, E.recovered, SYNTHETIC_FAMILY)
        self.assertEqual(rated["micro"], {"k": 8, "n": 17, "rate": 8 / 17})
        self.assertNotAlmostEqual(rated["estimate"], 0.5)

    def test_family_clustered_t_interval(self):
        values, _, _ = E.item_means(SYNTHETIC_ARM, E.recovered)
        out = E.family_weighted(values, SYNTHETIC_FAMILY)
        mean = 16 / 27
        sd = math.sqrt(93) / 27        # deviations -7/27, 11/27, -4/27; sum of squares 186/729; / (F - 1)
        half = T975[2] * sd / math.sqrt(3)
        self.assertEqual(out["df"], 2)
        self.assertAlmostEqual(out["family_sd"], sd, places=12)
        self.assertAlmostEqual(out["ci95_t"][0], mean - half, places=9)
        self.assertAlmostEqual(out["ci95_t"][1], mean + half, places=9)

    def test_t_critical_values_match_reference(self):
        for df, reference in T975.items():
            values = [Fraction(k) for k in range(df + 1)]      # SYNTHETIC: F = df + 1 families
            interval = E.A.t_interval(values, 0.95)
            self.assertEqual(interval["df"], df)
            self.assertAlmostEqual(interval["t_critical"], reference, places=9)

    def test_zero_variance_and_single_family(self):
        family = {"a": "f1", "b": "f2"}
        same = E.family_weighted({"a": Fraction(1, 2), "b": Fraction(1, 2)}, family)
        self.assertEqual(same["ci95_t"], [0.5, 0.5])
        self.assertIn("zero variance", same["interval_note"])
        single = E.family_weighted({"a": Fraction(1)}, {"a": "f1"})
        self.assertIsNone(single["ci95_t"])
        self.assertEqual(single["n_families"], 1)
        self.assertEqual(E.family_weighted({}, {})["n_items"], 0)


class ContrastTests(unittest.TestCase):
    def test_paired_contrast_takes_item_difference_on_shared_items(self):
        treatment = {"i1": [synthetic_row(1)], "i2": [synthetic_row(1)], "i3": [synthetic_row(0)],
                     "i4": [synthetic_row(1)]}
        control = {"i1": [synthetic_row(0)], "i2": [synthetic_row(1)], "i3": [synthetic_row(0)]}  # SYNTHETIC: no i4
        family = {"i1": "fa", "i2": "fa", "i3": "fb", "i4": "fb"}
        out = E.contrast(treatment, control, E.recovered, family, families=True)
        # shared i1..i3; diffs 1, 0, 0; fa = 1/2, fb = 0 -> 1/4
        self.assertEqual(out["n_items"], 3)
        self.assertEqual(out["estimate_exact"], "1/4")
        self.assertEqual(out["family_values"], {"fa": 0.5, "fb": 0.0})
        self.assertAlmostEqual(out["treatment"] - out["control"], out["estimate"], places=15)
        self.assertEqual(out["treatment_micro"], {"k": 2, "n": 3, "rate": 2 / 3})
        self.assertEqual(out["items_defined_treatment"], 4)

    def test_conditioning_on_submission_drops_items_without_a_submission(self):
        treatment = {"i1": [synthetic_row(1), synthetic_row(0, submitted=False)],
                     "i2": [synthetic_row(0, submitted=False), synthetic_row(0, submitted=False)]}
        control = {"i1": [synthetic_row(0), synthetic_row(1)], "i2": [synthetic_row(1), synthetic_row(1)]}
        family = {"i1": "fa", "i2": "fb"}
        given = E.contrast(treatment, control, E.recovered_given_submission, family)
        self.assertEqual(given["n_items"], 1)                  # i2 never submitted under treatment
        self.assertEqual(given["estimate_exact"], "1/2")       # i1: 1/1 - 1/2
        overall = E.contrast(treatment, control, E.recovered, family)
        self.assertEqual(overall["n_items"], 2)
        submission = E.rate(treatment, E.is_submitted, family)
        self.assertEqual(submission["micro"], {"k": 1, "n": 4, "rate": 0.25})


class RuleTests(unittest.TestCase):
    def test_v_bucket_rule_uses_nearest_rank_tertiles(self):
        n_v = {f"x{k}": v for k, v in enumerate([2] * 5 + [3] * 2 + [4, 5, 6])}   # SYNTHETIC, N = 10
        rule, bucket = E.v_bucket_rule(n_v)
        self.assertEqual((rule["c1"], rule["c2"]), (2, 3))     # 4th and 7th smallest values
        self.assertEqual(rule["items_per_bucket"], {"n_V<=2": 5, "2<n_V<=3": 2, "n_V>3": 3})
        self.assertEqual((bucket(2), bucket(3), bucket(12)), ("n_V<=2", "2<n_V<=3", "n_V>3"))

    def test_triple_concordance(self):
        same = [synthetic_row(1), synthetic_row(1), synthetic_row(1)]
        self.assertEqual(E.triple_concordance(same), (1, 1, 1, 3))
        tokens = [synthetic_row(1), synthetic_row(1, prompt=999), synthetic_row(1)]
        self.assertEqual(E.triple_concordance(tokens), (0, 1, 1, 3))
        mixed = [synthetic_row(1), synthetic_row(0, submitted=False), synthetic_row(1)]
        self.assertEqual(E.triple_concordance(mixed), (0, 0, 0, 2))

    def test_sha256_pin_refuses_other_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "synthetic.jsonl"
            path.write_bytes(b'{"label": "SYNTHETIC"}\n')
            with self.assertRaises(SystemExit):
                E.read_verified(path, "items")
            self.assertNotEqual(hashlib.sha256(path.read_bytes()).hexdigest(), E.EXPECTED_SHA256["items"])


if __name__ == "__main__":
    unittest.main()
