"""Exp A analysis, v2 (docs/EXPA_DESIGN_V2_DELTA.md D8, D9, D10.8, D10.11, D10.14 as amended by D11; lane A).

Stdlib only (the controller environment has no numpy or scipy). Deterministic for fixed
inputs and the RNG seeds declared in the analysis plan. Figures: scripts/expA_figures.py
in the figures environment (matplotlib 3.9.2).

Inputs
======
* ``items.jsonl`` (``expA-item-v2``), validated by ``score.ItemTruth``. The analysis
  population is the items whose ``role`` is in the plan's ``analysis_roles`` (``main``);
  the dev families are excluded from every analysis.
* ``scores.jsonl``: one ``expA-score-v2`` episode row per design cell
  (``scripts/expA_score.py score``). ``site_recovered`` is scored for the item's C* in
  every block, variant and condition; outside ``omit_node`` it is a baseline rate.
* ``policies.jsonl``: reference-policy rows for every item and variant
  (``scripts/expA_score.py policies``).
* The analysis plan (``expA-analysis-plan-v2``, ``plan_from_dict``) and, for the binding
  checks, the run directory and the dispatch manifest.

Binding (D10.14, validity finding F2)
=====================================
sha256(items.jsonl) must equal ``run.json``, the dispatch manifest's ``items_sha256``
and every score and policy row's ``items_sha256``; every terminal result in the run must
have exactly the score row that names its attempt and result sha256; the score rows must
cover exactly the design cells of the items. Any mismatch raises ``score.BindingError``,
so a rebuilt items file (for example with another C*) is refused.

Primary analysis (D9, D11.2-D11.4)
==================================
* Unit flow: item mean over the primary seeds (completed cells only) -> paired
  difference within item -> family mean -> equal-weight mean over families.
* H1 = C3 - C2 (P, pooled over the wording pairs), H2 = C4a - C1 (P), H3 = C1 - C0, all
  on ``site_recovered`` for C*. Two-sided one-sample t test on the family differences
  (df = F - 1), Holm across H1-H3; 95% t intervals are marginal. The Holm family never
  shrinks: a hypothesis that is not estimable enters Holm with p = 1 (the same rule for
  the sign-flip Holm column), and TABLES.md prints the family used.
* Zero variance: every family difference 0 -> p = 1 and the degenerate interval is
  flagged (not evidence of equivalence); every difference the same nonzero value (this
  includes F = 1) -> the t statistic is undefined and the sign-flip p is used.
* Primary seeds: (11, 23, 47) when every P cell of each seed is terminal (a completed
  model exit or missing (infrastructure) after the retry cap); otherwise the longest
  terminal prefix of that order. Other seed data are sensitivity analyses, and each
  partial seed is reported separately.
* Missing cells inside primary seeds: item-condition means use the completed cells;
  the bounds analysis imputes 0 and then 1 for every missing cell (and reports the
  directional extremes).
* Minimum data: a contrast with fewer than 20 contributing families is exploratory (no
  rejection is claimed). Inference is conditional on the frozen, purposively chosen
  family set.
* Sensitivity: sign-flip test (statistic: mean family difference, zeros kept; exact
  when F <= 20, else 100,000 draws with the declared seed, p = (1 + hits) / (1 + draws);
  it assumes symmetric null family effects) and the family cluster percentile
  bootstrap (10,000 resamples, declared seed).

Secondary and exploratory (D11.8, D11.9; never promoted, no significance claims)
==============================================================================
Rates by block and condition with exits, precision bounds and submitted-site counts;
reference policies next to each condition and H2 next to the c4_hits reference; S-VC
over-caution; S-NAT on nm_recall / nm_any; S-EDGE vs P at seed 11; C4 triggered
(reconstructed per item); strata of C* and findability strata; wording pairs; the T
funnel; process descriptives from telemetry; exposure audits from receipts; cost; the
condition x endpoint table. Each estimate carries nominal 95% family-clustered intervals.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics

from . import score as S

SCHEMA_VERSION = "expA-analysis-v2"
PLAN_SCHEMA = "expA-analysis-plan-v2"
PRIMARY_BLOCK, PRIMARY_VARIANT, PRIMARY_OUTCOME = "P", "omit_node", "site_recovered"
HYPOTHESES = (("H1", "C3", "C2"), ("H2", "C4a", "C1"), ("H3", "C1", "C0"))
HYPOTHESIS_MEANING = {"H1": "non-exhaustiveness guidance vs generic caution (C3 - C2)",
                      "H2": "bounded lexical evidence block vs plain answer (C4a - C1)",
                      "H3": "providing the incomplete answer as a package vs no answer (C1 - C0)"}
PRIMARY_TEST = "two-sided one-sample t test on family mean differences (df = F - 1)"
OUTCOMES = ("site_recovered", "function_localized", "recall_V_callers")
NM_OUTCOMES = ("nm_recall", "nm_any")
SYMMETRY_ASSUMPTION = ("The sign-flip test assumes that under the null hypothesis each family's mean paired "
                       "difference is symmetric about zero and that families are independent.")
SCOPE_STATEMENT = ("Inference is conditional on the frozen, purposively chosen family set (benchmark-conditional); "
                   "it is not a random sample of repositories.")
SYNTHETIC_LABEL = "SYNTHETIC FIXTURE (absent_edges.probe.analysis.synthetic_study) - not research data"
COST_FIELDS = ("calls", "prompt_tokens", "completion_tokens", "finish_reason_length_count")
GRAPH_REJECTED = frozenset({"rejected", "invalid_arguments", "error", "None"})


# --------------------------------------------------------------------------- utilities

def require(condition, message, error=ValueError):
    if not condition:
        raise error(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def dump_json(value) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


strict_loads = S.strict_loads


def _is_int(value):
    return type(value) is int


def _is_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _frac(value) -> Fraction:
    """Exact rational of an int/bool/Fraction/float (a float maps to its exact binary value)."""
    if isinstance(value, Fraction):
        return value
    if type(value) is bool:
        return Fraction(int(value))
    require(_is_number(value), f"not a finite number: {value!r}")
    return Fraction(value)


def _mean(values):
    values = list(values)
    require(values, "mean of an empty set")
    return sum((_frac(v) for v in values), Fraction(0)) / len(values)


def _quantile(sorted_values, probability):
    """Linear interpolation between order statistics (Hyndman-Fan type 7)."""
    position = (len(sorted_values) - 1) * probability
    lo, hi = math.floor(position), math.ceil(position)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (position - lo)


def _summary(values):
    values = sorted(values)
    if not values:
        return {"n": 0, "min": None, "median": None, "mean": None, "max": None}
    return {"n": len(values), "min": values[0], "median": statistics.median(values),
            "mean": statistics.fmean(values), "max": values[-1]}


def _ratio_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


# --------------------------------------------------------------------------- Student t (stdlib)

def _beta_continued_fraction(a, b, x):
    """Continued fraction of the regularized incomplete beta (modified Lentz)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 2000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-16:
            return h
    raise ArithmeticError("incomplete beta continued fraction did not converge")


def regularized_beta(x, a, b):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def student_t_cdf(t, df):
    require(df > 0, "degrees of freedom must be positive")
    tail = 0.5 * regularized_beta(df / (df + t * t), df / 2.0, 0.5)
    return 1.0 - tail if t >= 0 else tail


def student_t_two_sided_p(t, df):
    """P(|T| >= |t|) for Student's t with df degrees of freedom, without 1 - cdf cancellation."""
    require(df > 0, "degrees of freedom must be positive")
    return min(1.0, regularized_beta(df / (df + t * t), df / 2.0, 0.5))


def student_t_ppf(p, df):
    """Quantile of Student's t by bisection on the CDF (absolute accuracy about 1e-12)."""
    require(0.0 < p < 1.0, "p must be in (0, 1)")
    if p == 0.5:
        return 0.0
    if p < 0.5:
        return -student_t_ppf(1.0 - p, df)
    lo, hi = 0.0, 1.0
    while student_t_cdf(hi, df) < p:
        lo, hi = hi, hi * 2.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if student_t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-13 * max(1.0, hi):
            break
    return (lo + hi) / 2.0


def t_interval(values, confidence) -> dict:
    """Mean of values with a Student t interval (df = n - 1); descriptive use."""
    vals = [float(_frac(v)) for v in values]
    n = len(vals)
    out = {"method": "student_t", "n": n, "confidence": confidence, "mean": statistics.fmean(vals) if vals else None}
    if n < 2:
        out.update(df=None, se=None, lower=None, upper=None, note="fewer than two families: no t interval")
        return out
    sd = statistics.stdev(vals)
    se = sd / math.sqrt(n)
    critical = student_t_ppf(1.0 - (1.0 - confidence) / 2.0, n - 1)
    out.update(df=n - 1, sd=sd, se=se, t_critical=critical, lower=out["mean"] - critical * se,
               upper=out["mean"] + critical * se)
    if sd == 0:
        out["note"] = "zero variance: degenerate interval, not evidence of equivalence"
    return out


# --------------------------------------------------------------------------- resampling tests

_INDEX_CACHE: dict = {}


def _bootstrap_indices(n, resamples, seed):
    key = (n, resamples, seed)
    if key not in _INDEX_CACHE:
        rng = random.Random(seed)
        _INDEX_CACHE[key] = [[rng.randrange(n) for _ in range(n)] for _ in range(resamples)]
    return _INDEX_CACHE[key]


def cluster_bootstrap(values, *, resamples: int, seed: int, confidence: float) -> dict:
    """Percentile CI of the mean of family values; families resampled with replacement.

    The draws depend only on (number of families, resamples, seed), so estimates over the
    same family count share their resamples.
    """
    values = [float(_frac(v)) for v in values]
    n = len(values)
    require(n >= 1, "bootstrap needs at least one family")
    get = values.__getitem__
    draws = sorted(sum(map(get, row)) / n for row in _bootstrap_indices(n, resamples, seed))
    tail = (1 - confidence) / 2
    out = {"method": "family_cluster_percentile_bootstrap", "resamples": resamples, "seed": seed,
           "confidence": confidence, "n_families": n, "lower": _quantile(draws, tail),
           "upper": _quantile(draws, 1 - tail), "bootstrap_se": statistics.pstdev(draws)}
    if n < 2:
        out["note"] = "one family: the interval is degenerate"
    return out


def _integer_values(values):
    fractions = [_frac(v) for v in values]
    scale = 1
    for value in fractions:
        scale = math.lcm(scale, value.denominator)
    return [int(value * scale) for value in fractions], scale


def _exact_sign_flip_counts(ints, observed):
    """Enumerate all 2**n sign vectors in Gray-code order (one flip per step)."""
    n, target, current = len(ints), abs(observed), observed
    negative = [False] * n
    two = greater = less = 1                  # the identity vector reproduces the observed sum
    for k in range(1, 1 << n):
        i = (k & -k).bit_length() - 1
        current += 2 * ints[i] if negative[i] else -2 * ints[i]
        negative[i] = not negative[i]
        if current >= observed:
            greater += 1
        if current <= observed:
            less += 1
        if abs(current) >= target:
            two += 1
    return two, greater, less, 1 << n


def sign_flip_counts_dp(ints, observed, max_total=2_000_000):
    """Exact sign-flip counts from the subset-sum generating function prod(1 + z^|x|).

    S = sum(e_f x_f) = 2T - A with T a subset sum of |x_f| and A = sum |x_f|. Returns
    (two, greater, less, total) like the Gray-code enumeration, or None when sum |x|
    exceeds ``max_total``. Used to check the Monte Carlo path when n > 20.
    """
    magnitudes = [abs(value) for value in ints]
    total_sum = sum(magnitudes)
    if total_sum > max_total:
        return None
    n = len(magnitudes)
    width = 8 * ((n + 2 + 7) // 8)
    poly = 1
    for magnitude in magnitudes:
        poly += poly << (width * magnitude)
    step = width // 8
    raw = poly.to_bytes(step * (total_sum + 1), "little")
    two = greater = less = 0
    target = abs(observed)
    for t in range(total_sum + 1):
        count = int.from_bytes(raw[t * step:(t + 1) * step], "little")
        if not count:
            continue
        s = 2 * t - total_sum
        if s >= observed:
            greater += count
        if s <= observed:
            less += count
        if abs(s) >= target:
            two += count
    return two, greater, less, 1 << n


def _monte_carlo_sign_flip_counts(ints, observed, draws, seed):
    rng = random.Random(seed)
    n, target = len(ints), abs(observed)
    two = greater = less = 0
    for _ in range(draws):
        bits, flipped = rng.getrandbits(n), 0
        while bits:
            low = bits & -bits
            flipped += ints[low.bit_length() - 1]
            bits ^= low
        current = observed - 2 * flipped
        if current >= observed:
            greater += 1
        if current <= observed:
            less += 1
        if abs(current) >= target:
            two += 1
    return two, greater, less


def sign_flip_test(values, *, exact_max_families=20, draws=100000, seed=20261101) -> dict:
    """Family-level sign-flip test of mean(values) == 0 (values: family mean paired differences).

    The statistic is the mean family difference; zero differences stay in. Exact over all
    2**F sign vectors when F <= exact_max_families, else Monte Carlo with p = (1 + hits) /
    (1 + draws). Arithmetic is exact (values scaled to integers).
    """
    require(len(values) >= 1, "sign-flip test needs at least one family")
    ints, scale = _integer_values(values)
    n, observed = len(ints), sum(ints)
    out = {"n_families": n, "statistic": "mean family paired difference (zeros kept)",
           "observed_mean": float(Fraction(observed, scale) / n), "symmetry_assumption": SYMMETRY_ASSUMPTION}
    if n <= exact_max_families:
        two, greater, less, total = _exact_sign_flip_counts(ints, observed)
        p = {"two-sided": Fraction(two, total), "greater": Fraction(greater, total), "less": Fraction(less, total)}
        out.update(method="exact_enumeration", sign_vectors=total,
                   hits={"two-sided": two, "greater": greater, "less": less},
                   p_exact={k: _ratio_text(v) for k, v in p.items()})
    else:
        two, greater, less = _monte_carlo_sign_flip_counts(ints, observed, draws, seed)
        p = {"two-sided": Fraction(1 + two, 1 + draws), "greater": Fraction(1 + greater, 1 + draws),
             "less": Fraction(1 + less, 1 + draws)}
        out.update(method="monte_carlo", draws=draws, seed=seed,
                   hits={"two-sided": two, "greater": greater, "less": less}, p_rule="(1 + hits) / (1 + draws)")
    out.update(p_two_sided=float(p["two-sided"]), p_greater=float(p["greater"]), p_less=float(p["less"]))
    return out


HOLM_NOT_ESTIMABLE_RULE = ("the Holm family is always every preregistered hypothesis (H1, H2, H3); a hypothesis "
                           "that is not estimable enters Holm with p = 1 and its own adjusted p is reported as null")


def holm(p_values: dict) -> dict:
    """Holm step-down adjusted p-values over the whole family given (D11.2: H1, H2 and H3).

    The family never shrinks: a hypothesis that is not estimable (p None) is counted with
    p = 1, which leaves the others' adjustment at the preregistered family size m; its own
    adjusted p is reported as None.
    """
    ranked = sorted((1.0 if p is None else p, key) for key, p in p_values.items())
    m = len(ranked)
    adjusted, running = {}, 0.0
    for i, (p, key) in enumerate(ranked):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[key] = running
    return {key: None if p_values[key] is None else adjusted[key] for key in p_values}


def t_test(values, *, confidence=0.95) -> dict:
    """D11.2 primary test on family differences, with the zero-variance rules.

    Returns status ``tested`` (t statistic, df, two-sided p, marginal t interval),
    ``all_zero`` (p = 1, degenerate interval, not equivalence), ``constant_nonzero``
    (t undefined; the caller uses the sign-flip p) or ``not_estimable`` (no family).
    """
    exact = [_frac(v) for v in values]
    n = len(exact)
    if n == 0:
        return {"status": "not_estimable", "n_families": 0, "p_two_sided": None}
    mean = float(_mean(exact))
    out = {"n_families": n, "mean": mean, "confidence": confidence}
    if all(v == exact[0] for v in exact):
        degenerate = {"lower": mean, "upper": mean, "degenerate": True,
                      "note": "zero variance: degenerate interval, not evidence of equivalence"}
        if exact[0] == 0:
            out.update(status="all_zero", t=None, df=n - 1 if n > 1 else None, p_two_sided=1.0, interval=degenerate)
        else:
            out.update(status="constant_nonzero", t=None, df=n - 1 if n > 1 else None, p_two_sided=None,
                       interval=degenerate)
        return out
    floats = [float(v) for v in exact]
    sd = statistics.stdev(floats)
    se = sd / math.sqrt(n)
    df = n - 1
    t = mean / se
    critical = student_t_ppf(1.0 - (1.0 - confidence) / 2.0, df)
    out.update(status="tested", t=t, df=df, sd=sd, se=se, t_critical=critical, p_two_sided=student_t_two_sided_p(t, df),
               interval={"lower": mean - critical * se, "upper": mean + critical * se, "degenerate": False})
    return out


# --------------------------------------------------------------------------- plan

@dataclass(frozen=True)
class Plan:
    design: str = S.DEFAULT_DESIGN
    label: str = ""
    analysis_roles: tuple = ("main",)
    p_seeds: tuple = S.P_SEEDS
    secondary_seed: int = S.SECONDARY_SEED
    alpha: float = 0.05
    confidence: float = 0.95
    min_families_confirmatory: int = 20
    practically_meaningful_effect: float = 0.10
    bootstrap_resamples: int = 10000
    bootstrap_seed: int = 20261101
    sign_flip_exact_max_families: int = 20
    sign_flip_draws: int = 100000
    sign_flip_seed: int = 20261101

    def frozen_primary(self):
        return {"block": PRIMARY_BLOCK, "outcome": PRIMARY_OUTCOME, "test": PRIMARY_TEST, "multiplicity": "holm",
                "hypotheses": [{"id": h, "treatment": t, "control": c} for h, t, c in HYPOTHESES]}

    def to_dict(self):
        return {"schema_version": PLAN_SCHEMA, "label": self.label, "design": self.design,
                "analysis_roles": list(self.analysis_roles), "p_seeds": list(self.p_seeds),
                "secondary_seed": self.secondary_seed, "alpha": self.alpha, "confidence": self.confidence,
                "min_families_confirmatory": self.min_families_confirmatory,
                "practically_meaningful_effect": self.practically_meaningful_effect,
                "bootstrap": {"resamples": self.bootstrap_resamples, "seed": self.bootstrap_seed},
                "sign_flip": {"exact_max_families": self.sign_flip_exact_max_families, "draws": self.sign_flip_draws,
                              "seed": self.sign_flip_seed},
                "primary": self.frozen_primary()}


def plan_from_dict(data: dict) -> Plan:
    """Validate the preregistered analysis section (top level or under "analysis")."""
    require(isinstance(data, dict), "analysis plan must be a JSON object")
    if "schema_version" not in data and isinstance(data.get("analysis"), dict):
        data = data["analysis"]
    allowed = {"schema_version", "label", "design", "analysis_roles", "p_seeds", "secondary_seed", "alpha",
               "confidence", "min_families_confirmatory", "practically_meaningful_effect", "bootstrap", "sign_flip",
               "primary", "notes"}
    require(not set(data) - allowed, f"unknown analysis plan fields: {sorted(set(data) - allowed)}")
    require(data.get("schema_version") == PLAN_SCHEMA, f"analysis plan schema must be {PLAN_SCHEMA}")
    design = data.get("design", S.DEFAULT_DESIGN)
    require(design in S.DESIGNS, f"design must be one of {sorted(S.DESIGNS)}")
    roles = data.get("analysis_roles", ["main"])
    require(isinstance(roles, list) and roles and all(isinstance(r, str) and r for r in roles)
            and len(set(roles)) == len(roles), "analysis_roles must be a list of distinct role names")
    seeds = data.get("p_seeds", list(S.P_SEEDS))
    require(isinstance(seeds, list) and seeds == list(S.P_SEEDS)[:len(seeds)] and seeds,
            f"p_seeds must be a prefix of the frozen order {list(S.P_SEEDS)}")
    secondary = data.get("secondary_seed", S.SECONDARY_SEED)
    require(secondary == S.SECONDARY_SEED, f"secondary blocks run only seed {S.SECONDARY_SEED}")
    numbers = {}
    for name, default, low, high in (("alpha", 0.05, 0, 1), ("confidence", 0.95, 0, 1),
                                     ("practically_meaningful_effect", 0.10, 0, 1)):
        value = data.get(name, default)
        require(_is_number(value) and low < value < high, f"{name} must be in ({low}, {high})")
        numbers[name] = float(value)
    minimum = data.get("min_families_confirmatory", 20)
    require(_is_int(minimum) and minimum >= 2, "min_families_confirmatory must be an integer >= 2")
    boot = data.get("bootstrap") or {}
    require(isinstance(boot, dict) and not set(boot) - {"resamples", "seed"}, "bootstrap takes resamples and seed")
    resamples, boot_seed = boot.get("resamples", 10000), boot.get("seed", 20261101)
    require(_is_int(resamples) and resamples >= 100, "bootstrap resamples must be an integer >= 100")
    require(_is_int(boot_seed), "bootstrap seed must be an integer")
    flip = data.get("sign_flip") or {}
    require(isinstance(flip, dict) and not set(flip) - {"exact_max_families", "draws", "seed"},
            "sign_flip takes exact_max_families, draws and seed")
    exact_max, draws, flip_seed = flip.get("exact_max_families", 20), flip.get("draws", 100000), flip.get("seed",
                                                                                                         20261101)
    require(_is_int(exact_max) and 0 <= exact_max <= 24, "exact_max_families must be an integer in [0, 24]")
    require(_is_int(draws) and draws >= 100, "sign_flip draws must be an integer >= 100")
    require(_is_int(flip_seed), "sign_flip seed must be an integer")
    primary = data.get("primary")
    if primary is not None:
        require(primary == Plan().frozen_primary(), f"the primary block, outcome, test and hypotheses are frozen: "
                                                    f"{Plan().frozen_primary()}")
    label = data.get("label", "")
    require(isinstance(label, str), "label must be a string")
    return Plan(design=design, label=label, analysis_roles=tuple(roles), p_seeds=tuple(seeds), secondary_seed=secondary,
                alpha=numbers["alpha"], confidence=numbers["confidence"], min_families_confirmatory=minimum,
                practically_meaningful_effect=numbers["practically_meaningful_effect"], bootstrap_resamples=resamples,
                bootstrap_seed=boot_seed, sign_flip_exact_max_families=exact_max, sign_flip_draws=draws,
                sign_flip_seed=flip_seed)


def load_plan(path) -> tuple:
    data = Path(path).read_bytes()
    return plan_from_dict(strict_loads(data)), {"path": str(path), "sha256": sha256_bytes(data), "bytes": len(data)}


def default_plan_dict(*, label="", design=S.DEFAULT_DESIGN, **overrides) -> dict:
    data = Plan(design=design, label=label).to_dict()
    data.update(overrides)
    return data


# --------------------------------------------------------------------------- score rows

def _validate_outcomes(row, where):
    for name in S.OUTCOME_FIELDS:
        require(name in row, f"{where}: {name} is missing")
    if not row["completed"]:
        require(row["site_recovered"] is None and row["recall_V_callers"] is None,
                f"{where}: a cell without a terminal attempt has null outcomes")
        return
    require(row["site_recovered"] in (0, 1) and row["function_localized"] in (0, 1),
            f"{where}: site_recovered and function_localized must be 0 or 1")
    require(isinstance(row["recovered_V"], list) and _is_int(row["n_V"]) and row["n_V"] >= 1
            and len(row["recovered_V"]) <= row["n_V"], f"{where}: recovered_V/n_V are inconsistent")
    require(_is_number(row["recall_V_callers"]) and abs(row["recall_V_callers"] - len(row["recovered_V"]) / row["n_V"])
            < 1e-9 and row["recall_V_sites"] == row["recall_V_callers"], f"{where}: recall disagrees with recovered_V")
    cats = row["precision_categories"]
    require(isinstance(cats, dict) and set(cats) == set(S.CATEGORIES)
            and all(_is_int(v) and v >= 0 for v in cats.values()), f"{where}: precision_categories must count "
                                                                    f"{S.CATEGORIES}")
    require(sum(cats.values()) == row["n_submitted"] and 0 <= row["n_unique_sites"] <= row["n_submitted"],
            f"{where}: site counts are inconsistent")
    for name in ("precision_lower", "precision_upper", "nm_recall"):
        require(row[name] is None or (_is_number(row[name]) and 0 <= row[name] <= 1), f"{where}: {name} out of range")
    require(row["nm_any"] in (None, 0, 1), f"{where}: nm_any must be 0, 1 or null")
    require(row["complete"] in (None,) + S.CLAIMS, f"{where}: complete must be yes/no/unknown/null")
    for name in ("certifiably_false_completeness", "uncertain_despite_full_V"):
        require(row[name] in (None, True, False), f"{where}: {name} must be a boolean or null")


def load_score_rows(path) -> tuple:
    """Episode rows by cell_id (exactly one row per cell) and file metadata."""
    data = Path(path).read_bytes()
    rows, designs = {}, set()
    for lineno, line in enumerate(data.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = strict_loads(line)
        where = f"{Path(path).name}:{lineno}"
        require(isinstance(row, dict) and row.get("schema") == S.SCORE_SCHEMA and row.get("kind") == "episode",
                f"{where}: not an {S.SCORE_SCHEMA} episode row")
        parsed = S.parse_cell_id(row.get("cell_id"))
        for name in ("block", "item_id", "variant", "condition", "seed"):
            require(row.get(name) == parsed[name], f"{where}: {name} disagrees with the cell id")
        require(row.get("cell_state") in S.CELL_STATES, f"{where}: unknown cell_state {row.get('cell_state')!r}")
        require(row.get("exit_status") in S.ROW_STATUSES, f"{where}: unknown exit_status {row.get('exit_status')!r}")
        require(row.get("completed") is (row["cell_state"] == "completed")
                and row["completed"] is (row["exit_status"] in S.COMPLETED_EXITS),
                f"{where}: completed, cell_state and exit_status disagree")
        _validate_outcomes(row, where)
        if row["completed"] and row["exit_status"] != "Submitted":
            require(row["site_recovered"] == 0 and row["n_submitted"] == 0 and not row["recovered_V"],
                    f"{where}: a non-submitted episode cannot recover or submit sites")
        require(row["cell_id"] not in rows, f"{where}: duplicate row for {row['cell_id']}")
        designs.add(row.get("design"))
        rows[row["cell_id"]] = row
    require(rows, f"{path}: no score rows")
    require(len(designs) == 1 and designs <= set(S.DESIGNS), f"{path}: rows name designs {sorted(map(str, designs))}")
    return rows, {"path": str(path), "sha256": sha256_bytes(data), "bytes": len(data), "rows": len(rows),
                  "design": designs.pop()}


def load_policy_rows(path) -> tuple:
    """Policy rows by (item_id, variant, policy)."""
    data = Path(path).read_bytes()
    rows = {}
    for lineno, line in enumerate(data.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = strict_loads(line)
        where = f"{Path(path).name}:{lineno}"
        require(isinstance(row, dict) and row.get("schema") == S.POLICY_SCHEMA and row.get("kind") == "policy",
                f"{where}: not an {S.POLICY_SCHEMA} policy row")
        require(row.get("policy") in S.POLICIES and row.get("variant") in (S.NO_VARIANT,) + S.VARIANTS,
                f"{where}: unknown policy or variant")
        require(row.get("completed") is True, f"{where}: a policy row is a completed submission")
        _validate_outcomes(row, where)
        key = (row.get("item_id"), row["variant"], row["policy"])
        require(key not in rows, f"{where}: duplicate policy row {key}")
        rows[key] = row
    require(rows, f"{path}: no policy rows")
    return rows, {"path": str(path), "sha256": sha256_bytes(data), "bytes": len(data), "rows": len(rows)}


# --------------------------------------------------------------------------- binding

def check_binding(items, items_meta, score_rows, policy_rows, *, design, run_dir=None, dispatch_manifest=None,
                  scores_manifest=None, require_run=True) -> dict:
    """F2: refuse unless items, run, dispatch, score and policy rows are bound to one items file."""
    sha = items_meta["sha256"]
    audit = {"items_sha256": sha}
    layout = S.Design(design)
    if run_dir is not None:
        run_path = Path(run_dir) / "run.json"
        require(run_path.is_file(), f"{run_path} not found", S.BindingError)
        run_raw = run_path.read_bytes()
        run_json = strict_loads(run_raw)
        recorded = S.run_items_sha256(run_json)
        require(recorded == sha, f"items file sha256 {sha} differs from run.json's {recorded}", S.BindingError)
        audit.update(run_json_sha256=sha256_bytes(run_raw), run_json_items_sha256=recorded)
        scanned = S.scan_run(run_dir)
        name, _ = S.detect_design(run_json, list(scanned), design)
        require(name == design, f"the run's design {name} is not the score rows' {design}", S.BindingError)
        terminal = {}
        for cell, attempts in scanned.items():
            resolved = S.resolve_cell(attempts)
            if resolved["terminal"] is not None:
                terminal[cell] = (resolved["terminal"]["attempt_id"], resolved["terminal"]["result_sha256"])
        scored = {row["cell_id"]: (row["attempt_id"], row["result_sha256"]) for row in score_rows.values()
                  if row["completed"]}
        require(set(terminal) == set(scored), f"terminal results and scored rows differ: "
                                              f"{len(set(terminal) - set(scored))} unscored, "
                                              f"{len(set(scored) - set(terminal))} without a result file",
                S.BindingError)
        bad = sorted(cell for cell in terminal if terminal[cell] != scored[cell])
        require(not bad, f"{len(bad)} score rows name another attempt or result sha256, e.g. {bad[:2]}",
                S.BindingError)
        audit["terminal_results_verified"] = len(terminal)
        # Every design cell, not only the completed ones: seed_completion chooses the primary seeds from the
        # non-completed states, so scores taken before the last infrastructure attempt was published (or before
        # a cell was dispatched) must not be analysed against the final run (review finding, check_binding).
        stale = []
        for cell_id, row in score_rows.items():
            resolved = S.resolve_cell(scanned.get(cell_id, []))
            recorded = (row.get("cell_state"), row.get("infra_attempts"), row.get("retry_cap_exceeded"))
            current = (resolved["state"], resolved["infra_attempts"], resolved["retry_cap_exceeded"])
            if recorded != current:
                stale.append(f"{cell_id}: scored {recorded}, run {current}")
        require(not stale, f"{len(stale)} score rows disagree with the run's cell states (cell_state, infra_attempts, "
                           f"retry_cap_exceeded); re-score the final run. E.g. {stale[:2]}", S.BindingError)
        audit["cell_states_verified"] = len(score_rows)
    else:
        require(not require_run, "the analysis needs the run directory to verify the run binding", S.BindingError)
        audit["terminal_results_verified"] = None
    if dispatch_manifest is not None:
        raw = Path(dispatch_manifest).read_bytes()
        manifest = strict_loads(raw)
        require(manifest.get("items_sha256") == sha, f"items file sha256 {sha} differs from the dispatch manifest's "
                                                     f"{manifest.get('items_sha256')!r}", S.BindingError)
        audit["dispatch_manifest_sha256"] = sha256_bytes(raw)
    else:
        require(not require_run, "the analysis needs the dispatch manifest to verify the items binding",
                S.BindingError)
    if scores_manifest is not None:
        manifest = scores_manifest
        require((manifest.get("items") or {}).get("sha256") == sha, "the scores manifest names another items file",
                S.BindingError)
        audit["scores_manifest_items_sha256"] = sha
    for label, rows in (("score", score_rows.values()), ("policy", policy_rows.values())):
        stale = sum(1 for row in rows if row.get("items_sha256") != sha)
        require(not stale, f"{stale} {label} rows were computed from another items file", S.BindingError)
    cells = {cell["cell_id"]: cell for cell in layout.cells(items)}
    outside = sorted(set(score_rows) - set(cells))
    require(not outside, f"{len(outside)} score rows are outside the {design} design of these items, e.g. "
                         f"{outside[:2]}", S.BindingError)
    uncovered = sorted(set(cells) - set(score_rows))
    require(not uncovered, f"{len(uncovered)} design cells have no score row, e.g. {uncovered[:2]}", S.BindingError)
    for row in score_rows.values():
        item = items[row["item_id"]]
        require(row.get("family") == item.family and row.get("omitted") == item.omitted
                and row.get("wording_pair") == item.wording_pair,
                f"{row['cell_id']}: family, omitted caller or wording pair differs from the item", S.BindingError)
    expected_policy = {(i, v, p) for i, item in items.items() for v in (S.NO_VARIANT,) + S.VARIANTS
                       for p in S.POLICIES if v == S.NO_VARIANT or item.variants[v]["cells"]["C1"] is not None}
    require(set(policy_rows) == expected_policy, f"policy rows do not cover exactly every item and variant "
                                                 f"({len(expected_policy - set(policy_rows))} missing, "
                                                 f"{len(set(policy_rows) - expected_policy)} extra)", S.BindingError)
    for (item_id, variant, policy), row in policy_rows.items():
        require(row.get("omitted") == items[item_id].omitted and row.get("family") == items[item_id].family,
                f"policy row {policy}/{item_id}/{variant} does not match the items", S.BindingError)
    audit.update(score_rows=len(score_rows), policy_rows=len(policy_rows), design=design, design_cells=len(cells))
    return audit


# --------------------------------------------------------------------------- study table

class Study:
    def __init__(self, plan: Plan, items: dict, score_rows: dict, policy_rows: dict):
        self.plan = plan
        self.layout = S.Design(plan.design)
        self.all_items = items
        self.items = {k: v for k, v in items.items() if v.role in plan.analysis_roles}
        require(self.items, f"no items with role in {list(plan.analysis_roles)}")
        self.families = tuple(sorted({item.family for item in self.items.values()}))
        self.rows = score_rows
        self.policies = policy_rows
        self.cells = self.layout.cells(self.items)
        self.index = {(c["block"], c["item_id"], c["condition"], c["seed"]): score_rows[c["cell_id"]]
                      for c in self.cells if c["cell_id"] in score_rows}

    def arm(self, condition, block=PRIMARY_BLOCK):
        """(block, designed seeds) of a P-family arm; C0 lives in S-C0 under contract-v2."""
        if condition == "C0":
            block = self.layout.c0_block()
        spec = self.layout.block(block)
        require(condition in spec.conditions, f"block {block} has no condition {condition}")
        return block, spec.seeds

    def block_items(self, block):
        if not self.layout.has(block):
            return []
        return sorted(i for i, item in self.items.items() if self.layout.includes(item, block))

    def row(self, block, item_id, condition, seed):
        return self.index.get((block, item_id, condition, seed))

    def family_of(self, item_id):
        return self.items[item_id].family


def outcome_value(row, outcome):
    """Fraction value of a completed row's outcome, or None when undefined."""
    if outcome in ("site_recovered", "function_localized", "nm_any"):
        value = row[outcome]
        return None if value is None else Fraction(value)
    if outcome == "recall_V_callers":
        return Fraction(len(row["recovered_V"]), row["n_V"])
    if outcome == "nm_recall":
        return None if row["nm_recall"] is None else _frac(row["nm_recall"])
    value = _row_field(row, outcome)
    return None if value is None else _frac(value)


def _row_field(row, field):
    cost = row.get("cost") or {}
    telemetry = row.get("telemetry") or {}
    if field == "total_tokens":
        prompt, completion = cost.get("prompt_tokens"), cost.get("completion_tokens")
        return None if prompt is None or completion is None else prompt + completion
    if field in COST_FIELDS:
        return cost.get(field)
    if field == "complete_yes":
        return None if row.get("complete") is None else int(row["complete"] == "yes")
    if field in ("searched_target_name", "mentioned_omitted_caller_file"):
        value = telemetry.get(field)
        return None if value is None else int(bool(value))
    if field == "tool_calls_before_submit":
        return telemetry.get(field)
    value = row.get(field)
    if type(value) is bool:
        return int(value)
    return value


def item_value(study: Study, block, item_id, condition, seeds, outcome, impute=None):
    """Mean of the outcome over the item's completed cells at these seeds (D11.3).

    ``impute`` (0 or 1) replaces every cell without a terminal model exit (bounds only).
    Returns (value or None, completed cells used, imputed cells).
    """
    values, imputed = [], 0
    for seed in seeds:
        row = study.row(block, item_id, condition, seed)
        if row is None:
            continue
        if row["completed"]:
            value = outcome_value(row, outcome)
            if value is not None:
                values.append(value)
        elif impute is not None:
            values.append(Fraction(impute))
            imputed += 1
    if not values:
        return None, 0, imputed
    return sum(values, Fraction(0)) / len(values), len(values) - imputed, imputed


def family_means(study: Study, values_by_item: dict) -> dict:
    groups = defaultdict(list)
    for item_id, value in values_by_item.items():
        groups[study.family_of(item_id)].append(_frac(value))
    return {family: sum(values, Fraction(0)) / len(values) for family, values in sorted(groups.items())}


def describe(study: Study, values_by_item: dict, *, families=False) -> dict | None:
    """Family-weighted estimate with nominal t interval and cluster bootstrap (descriptive)."""
    if not values_by_item:
        return None
    plan = study.plan
    fam = family_means(study, values_by_item)
    estimate = _mean(fam.values())
    out = {"estimate": float(estimate), "estimate_exact": _ratio_text(estimate), "n_items": len(values_by_item),
           "n_families": len(fam), "micro_item_mean": float(_mean(values_by_item.values())),
           "t_interval": t_interval(list(fam.values()), plan.confidence),
           "bootstrap": cluster_bootstrap(list(fam.values()), resamples=plan.bootstrap_resamples,
                                          seed=plan.bootstrap_seed, confidence=plan.confidence)}
    if families:
        out["family_values"] = {name: float(value) for name, value in fam.items()}
    return out


def paired(study: Study, treatment: dict, control: dict, *, families=False) -> dict | None:
    """Descriptive paired difference (treatment - control) on the items both sides have."""
    shared = sorted(set(treatment) & set(control))
    if not shared:
        return None
    plan = study.plan
    diffs = {i: _frac(treatment[i]) - _frac(control[i]) for i in shared}
    fam_d = family_means(study, diffs)
    fam_t = family_means(study, {i: treatment[i] for i in shared})
    fam_c = family_means(study, {i: control[i] for i in shared})
    estimate = _mean(fam_d.values())
    out = {"n_items": len(shared), "n_families": len(fam_d), "treatment_rate": float(_mean(fam_t.values())),
           "control_rate": float(_mean(fam_c.values())), "estimate": float(estimate),
           "estimate_exact": _ratio_text(estimate), "micro_item_estimate": float(_mean(diffs.values())),
           "t_interval": t_interval(list(fam_d.values()), plan.confidence),
           "bootstrap": cluster_bootstrap(list(fam_d.values()), resamples=plan.bootstrap_resamples,
                                          seed=plan.bootstrap_seed, confidence=plan.confidence)}
    if families:
        sizes = Counter(study.family_of(i) for i in shared)
        out["family_units"] = [{"family": f, "n_items": sizes[f], "treatment": float(fam_t[f]),
                                "control": float(fam_c[f]), "difference": float(fam_d[f]),
                                "difference_exact": _ratio_text(fam_d[f])} for f in sorted(fam_d)]
    return out


def arm_values(study: Study, condition, seeds, outcome=PRIMARY_OUTCOME, items=None, impute=None, block=PRIMARY_BLOCK):
    """{item: value} for one arm at these seeds (restricted to the arm's designed seeds)."""
    arm_block, designed = study.arm(condition, block)
    use = [s for s in seeds if s in designed]
    values = {}
    for item_id in (study.block_items(arm_block) if items is None else items):
        value, _, _ = item_value(study, arm_block, item_id, condition, use, outcome, impute)
        if value is not None:
            values[item_id] = value
    return values, use


def contrast_values(study: Study, treatment, control, seeds, *, outcome=PRIMARY_OUTCOME, items=None, impute=None,
                    impute_treatment=None, impute_control=None):
    """Item differences treatment - control at the seeds both arms are designed for."""
    t_block, t_seeds = study.arm(treatment)
    c_block, c_seeds = study.arm(control)
    use = [s for s in seeds if s in t_seeds and s in c_seeds]
    base = sorted(set(study.block_items(t_block)) & set(study.block_items(c_block))) if items is None else items
    t_imp = impute if impute_treatment is None else impute_treatment
    c_imp = impute if impute_control is None else impute_control
    treat, ctrl = {}, {}
    for item_id in base:
        t_value, _, _ = item_value(study, t_block, item_id, treatment, use, outcome, t_imp)
        c_value, _, _ = item_value(study, c_block, item_id, control, use, outcome, c_imp)
        if t_value is not None and c_value is not None:
            treat[item_id], ctrl[item_id] = t_value, c_value
    return treat, ctrl, use


# --------------------------------------------------------------------------- primary (D9, D11.2-D11.4)

def seed_completion(study: Study) -> dict:
    """Per P seed: cell states and whether every P cell is terminal (D11.3)."""
    items = study.block_items(PRIMARY_BLOCK)
    spec = study.layout.block(PRIMARY_BLOCK)
    out = {}
    for seed in study.plan.p_seeds:
        rows = [study.row(PRIMARY_BLOCK, i, c, seed) for i in items for c in spec.conditions]
        states = Counter(row["cell_state"] for row in rows)
        terminal = sum(states[s] for s in S.TERMINAL_STATES)
        status = ("complete" if terminal == len(rows) else
                  "partial" if states["completed"] or states["missing_infrastructure"] or states["infra_pending"]
                  else "not_started")
        out[seed] = {"status": status, "cells": len(rows), "terminal_cells": terminal,
                     "cell_states": {s: states.get(s, 0) for s in S.CELL_STATES}}
    return out


def primary_seeds(study: Study, completion=None):
    completion = completion or seed_completion(study)
    seeds = []
    for seed in study.plan.p_seeds:
        if completion[seed]["status"] != "complete":
            break
        seeds.append(seed)
    return tuple(seeds)


def _family_diffs(study, treatment_values, control_values):
    shared = sorted(set(treatment_values) & set(control_values))
    diffs = {i: _frac(treatment_values[i]) - _frac(control_values[i]) for i in shared}
    return diffs, family_means(study, diffs)


def infer(study: Study, family_diffs: dict) -> dict:
    """Primary test (t test with the D11.2 zero-variance rules) and the two sensitivity analyses."""
    plan = study.plan
    values = [family_diffs[f] for f in sorted(family_diffs)]
    test = t_test(values, confidence=plan.confidence)
    out = {"t_test": test, "n_families": len(values)}
    if not values:
        out.update(p_primary=None, p_source="not_estimable", sign_flip=None, bootstrap=None)
        return out
    flip = sign_flip_test(values, exact_max_families=plan.sign_flip_exact_max_families, draws=plan.sign_flip_draws,
                          seed=plan.sign_flip_seed)
    boot = cluster_bootstrap(values, resamples=plan.bootstrap_resamples, seed=plan.bootstrap_seed,
                             confidence=plan.confidence)
    if test["status"] == "tested":
        p, source = test["p_two_sided"], "t_test"
    elif test["status"] == "all_zero":
        p, source = 1.0, "all_family_differences_zero"
    else:
        p, source = flip["p_two_sided"], "sign_flip_constant_nonzero_differences"
    out.update(p_primary=p, p_source=source, sign_flip=flip, bootstrap=boot)
    return out


def _bounds(study: Study, treatment, control, seeds):
    """D11.3 missing-cell bounds: impute 0, impute 1, and the directional extremes."""
    out = {}
    for label, kwargs in (("impute_0", {"impute": 0}), ("impute_1", {"impute": 1}),
                          ("lowest", {"impute_treatment": 0, "impute_control": 1}),
                          ("highest", {"impute_treatment": 1, "impute_control": 0})):
        treat, ctrl, _ = contrast_values(study, treatment, control, seeds, **kwargs)
        _, fam = _family_diffs(study, treat, ctrl)
        if not fam:
            out[label] = None
            continue
        test = t_test(list(fam.values()), confidence=study.plan.confidence)
        out[label] = {"estimate": float(_mean(fam.values())), "n_families": len(fam), "n_items": len(treat),
                      "t_interval": test.get("interval"), "t_test_status": test["status"],
                      "p_two_sided": test.get("p_two_sided")}
    return out


def _missing_cells(study: Study, treatment, control, seeds):
    counts = Counter()
    for condition in (treatment, control):
        block, designed = study.arm(condition)
        for item_id in study.block_items(block):
            for seed in seeds:
                if seed not in designed:
                    continue
                row = study.row(block, item_id, condition, seed)
                if row is not None and not row["completed"]:
                    counts[f"{condition}:{row['cell_state']}"] += 1
    return dict(sorted(counts.items()))


def primary_analysis(study: Study) -> dict:
    plan = study.plan
    completion = seed_completion(study)
    seeds = primary_seeds(study, completion)
    other = [s for s in plan.p_seeds if s not in seeds and completion[s]["status"] != "not_started"]
    out = {"block": PRIMARY_BLOCK, "outcome": PRIMARY_OUTCOME, "test": PRIMARY_TEST, "alpha": plan.alpha,
           "design": plan.design, "multiplicity": "Holm step-down across H1, H2 and H3 (primary p-values); "
                                                  + HOLM_NOT_ESTIMABLE_RULE,
           "seed_order": list(plan.p_seeds), "primary_seeds": list(seeds), "sensitivity_seeds": other,
           "seed_completion": {str(s): c for s, c in completion.items()},
           "seed_rule": ("a seed is complete when every P cell is terminal (completed model exit or missing "
                         "(infrastructure) after the retry cap); the primary seeds are the longest complete prefix "
                         "of the seed order"),
           "minimum_families_confirmatory": plan.min_families_confirmatory, "scope": SCOPE_STATEMENT,
           "symmetry_assumption_sign_flip": SYMMETRY_ASSUMPTION,
           "practically_meaningful_effect": plan.practically_meaningful_effect, "hypotheses": []}
    if not seeds:
        out["status"] = "not_estimable"
        out["reason"] = f"seed {plan.p_seeds[0]} has P cells that are not terminal; no complete seed prefix"
    else:
        out["status"] = "estimated"
    for hypothesis, treatment, control in HYPOTHESES:
        entry = {"id": hypothesis, "meaning": HYPOTHESIS_MEANING[hypothesis], "treatment": treatment,
                 "control": control, "treatment_block": study.arm(treatment)[0],
                 "control_block": study.arm(control)[0]}
        if seeds:
            treat, ctrl, used = contrast_values(study, treatment, control, seeds)
            diffs, fam = _family_diffs(study, treat, ctrl)
            entry.update(seeds_used=used, n_items=len(diffs), n_families=len(fam),
                         items_without_complete_pair=len(set(study.block_items(study.arm(treatment)[0]))
                                                         & set(study.block_items(study.arm(control)[0]))) - len(diffs))
            if used != list(seeds):
                entry["estimand_note"] = (f"both arms are designed only at seeds {used}, so the contrast uses those "
                                          "seeds (contract-v2 layout: C0 runs at seed 11 only); this is not the D11 "
                                          "estimand, which runs C0 at every P seed")
            if fam:
                estimate = _mean(fam.values())
                entry.update(estimate=float(estimate), estimate_exact=_ratio_text(estimate),
                             treatment_rate=float(_mean(family_means(study, treat).values())),
                             control_rate=float(_mean(family_means(study, ctrl).values())),
                             micro_item_estimate=float(_mean(diffs.values())))
                entry.update(infer(study, fam))
                sizes = Counter(study.family_of(i) for i in diffs)
                ft, fc = family_means(study, treat), family_means(study, ctrl)
                entry["family_units"] = [{"family": f, "n_items": sizes[f], "treatment": float(ft[f]),
                                          "control": float(fc[f]), "difference": float(fam[f]),
                                          "difference_exact": _ratio_text(fam[f])} for f in sorted(fam)]
                entry["confirmatory"] = len(fam) >= plan.min_families_confirmatory
                entry["status"] = "confirmatory" if entry["confirmatory"] else "exploratory_fewer_than_minimum_families"
                entry["abs_estimate_at_least_meaningful"] = abs(float(estimate)) >= plan.practically_meaningful_effect
            else:
                entry.update(status="not_estimable", p_primary=None, confirmatory=False,
                             reason="no item has completed cells in both arms")
            entry["missing_cells_in_primary_seeds"] = _missing_cells(study, treatment, control, seeds)
            entry["bounds_missing_cells"] = _bounds(study, treatment, control, seeds)
        else:
            entry.update(status="not_estimable", p_primary=None, confirmatory=False)
        out["hypotheses"].append(entry)
    adjusted = holm({h["id"]: h.get("p_primary") for h in out["hypotheses"]})
    flips = holm({h["id"]: (h.get("sign_flip") or {}).get("p_two_sided") for h in out["hypotheses"]})
    for entry in out["hypotheses"]:
        entry["holm_adjusted_p"] = adjusted[entry["id"]]
        entry["sign_flip_holm_adjusted_p"] = flips[entry["id"]]
        if entry.get("confirmatory") and adjusted[entry["id"]] is not None:
            entry["reject_at_alpha"] = adjusted[entry["id"]] <= plan.alpha
        else:
            entry["reject_at_alpha"] = None
    out["holm_family"] = [h["id"] for h in out["hypotheses"]]
    out["holm_entered_as_p1"] = [h["id"] for h in out["hypotheses"] if h.get("p_primary") is None]
    out["sign_flip_holm_entered_as_p1"] = [h["id"] for h in out["hypotheses"]
                                           if (h.get("sign_flip") or {}).get("p_two_sided") is None]
    out["sensitivity"] = _seed_sensitivity(study, seeds, other, completion)
    return out


def _contrast_summary(study, treatment, control, seeds):
    treat, ctrl, used = contrast_values(study, treatment, control, seeds)
    diffs, fam = _family_diffs(study, treat, ctrl)
    if not fam:
        return {"seeds_used": used, "n_items": 0, "n_families": 0}
    test = t_test(list(fam.values()), confidence=study.plan.confidence)
    return {"seeds_used": used, "n_items": len(diffs), "n_families": len(fam),
            "estimate": float(_mean(fam.values())), "t_interval": test.get("interval"),
            "t_test_status": test["status"], "p_two_sided": test.get("p_two_sided"),
            "bootstrap": cluster_bootstrap(list(fam.values()), resamples=study.plan.bootstrap_resamples,
                                           seed=study.plan.bootstrap_seed, confidence=study.plan.confidence)}


def _seed_sensitivity(study, seeds, other, completion):
    """Every seed with P data (all completed cells) and each partial or non-primary seed alone."""
    out = {"role": "sensitivity_not_primary", "per_seed": [], "all_seeds_with_data": None}
    with_data = [s for s in study.plan.p_seeds if completion[s]["cell_states"]["completed"]]
    if other and with_data:
        out["all_seeds_with_data"] = {"seeds": with_data, "contrasts": {
            h: _contrast_summary(study, t, c, with_data) for h, t, c in HYPOTHESES}}
    for seed in other:
        out["per_seed"].append({"seed": seed, "status": completion[seed]["status"],
                                "contrasts": {h: _contrast_summary(study, t, c, [seed]) for h, t, c in HYPOTHESES}})
    return out


# --------------------------------------------------------------------------- reference policies (D10.11, D11.1)

def policy_summary(study: Study, item_ids, variant) -> dict:
    """Family-weighted policy outcomes on these items for this variant ('none' for C0 and T)."""
    out = {}
    for policy in S.POLICIES:
        rows = {i: study.policies.get((i, variant, policy)) for i in item_ids}
        missing = sorted(i for i, row in rows.items() if row is None)
        rows = {i: row for i, row in rows.items() if row is not None}
        if not rows:
            out[policy] = {"items": 0, "missing_items": len(missing)}
            continue
        entry = {"items": len(rows), "missing_items": len(missing),
                 "site_recovered_items": sum(row["site_recovered"] for row in rows.values())}
        for outcome in OUTCOMES + NM_OUTCOMES + ("precision_lower", "precision_upper", "n_submitted",
                                                 "n_unique_sites"):
            values = {i: outcome_value(row, outcome) for i, row in rows.items()}
            values = {i: v for i, v in values.items() if v is not None}
            entry[outcome] = float(_mean(family_means(study, values).values())) if values else None
        out[policy] = entry
    return out


def h2_reference(study: Study, seeds) -> dict:
    """H2 (C4a - C1) next to the c4_hits reference on the same P items and seeds (descriptive)."""
    if not seeds:
        return {"status": "not_estimable"}
    treat, ctrl, used = contrast_values(study, "C4a", "C1", seeds)
    shared = sorted(set(treat) & set(ctrl))
    if not shared:
        return {"status": "not_estimable"}
    refs = {}
    for policy in S.POLICIES:
        values = {}
        for i in shared:
            row = study.policies.get((i, PRIMARY_VARIANT, policy))
            require(row is not None, f"no {policy} policy row for item {i}")
            values[i] = Fraction(row["site_recovered"])
        refs[policy] = values
    diff = {i: _frac(treat[i]) - _frac(ctrl[i]) for i in shared}
    reference_gap = {i: refs["answer_plus_c4"][i] - refs["answer_only"][i] for i in shared}
    fam_d, fam_gap = family_means(study, diff), family_means(study, reference_gap)
    mean_d, mean_gap = _mean(fam_d.values()), _mean(fam_gap.values())
    covariate = {}
    for flag in (True, False):
        ids = [i for i in shared if bool(study.items[i].covariates.get("c4_contains_omitted_site")) is flag]
        covariate["true" if flag else "false"] = paired(study, {i: treat[i] for i in ids}, {i: ctrl[i] for i in ids}) \
            or {"n_items": 0}
    disagreements = sorted(i for i in shared if study.items[i].covariates.get("c4_contains_omitted_site") is not None
                           and bool(study.items[i].covariates["c4_contains_omitted_site"])
                           != bool(refs["c4_hits"][i]))
    return {"status": "estimated", "seeds": used, "n_items": len(shared), "n_families": len(fam_d),
            "c4a_minus_c1": float(mean_d),
            "reference_rates_omit_node": {p: float(_mean(family_means(study, v).values())) for p, v in refs.items()},
            "reference_gap_answer_plus_c4_minus_answer_only": float(mean_gap),
            "c4a_minus_c1_share_of_reference_gap": float(mean_d / mean_gap) if mean_gap else None,
            "c4a_minus_c1_by_c4_contains_omitted_site": covariate,
            "covariate_vs_c4_hits_policy_disagreements": disagreements,
            "note": "Mechanical policies are references, not ceilings (D11.1): the share compares the observed H2 "
                    "difference with the recovery a policy gains by appending every C4 hit to the delivered answer."}


def _positive(count):
    """None when an items file predates the covariate, else whether it is positive."""
    return None if not isinstance(count, int) else count > 0


def findability(study: Study) -> dict:
    """D11.7 findability diagnostics of the frozen items (no model outcome is used).

    ``omitted_docref_in_answer``: a delivered omit_node snippet names C* in a Sphinx role or backticks
    (covariate ``visibility_docref_mentions``); ``omitted_name_in_answer``: any whole-word mention of
    C*'s simple name (``visibility_name_mentions``). Pre-outcome, reported per family, never a filter.
    """
    items = study.items
    rows = []
    for item_id in sorted(items):
        item = items[item_id]
        grep = study.policies.get((item_id, PRIMARY_VARIANT, "grep_all"))
        c4 = study.policies.get((item_id, PRIMARY_VARIANT, "c4_hits"))
        rows.append({"item_id": item_id, "family": item.family, "stratum": item.covariates.get("stratum"),
                     "wording_pair": item.wording_pair, "grep_recovers_omitted": None if grep is None
                     else bool(grep["site_recovered"]), "c4_hits_recovers_omitted": None if c4 is None
                     else bool(c4["site_recovered"]),
                     "c4_contains_omitted_site": item.covariates.get("c4_contains_omitted_site"),
                     "c4_trigger_fires": item.covariates.get("c4_trigger_fires"),
                     "visibility_text_matches": item.covariates.get("visibility_text_matches"),
                     "omitted_docref_in_answer": _positive(item.covariates.get("visibility_docref_mentions")),
                     "omitted_name_in_answer": _positive(item.covariates.get("visibility_name_mentions")),
                     "natural_miss": item.covariates.get("natural_miss"), "n_V": len(item.V)})

    def share(key):
        values = {r["item_id"]: Fraction(int(bool(r[key]))) for r in rows if r[key] is not None}
        return {"items": len(values), "count": int(sum(values.values())),
                "family_weighted": float(_mean(family_means(study, values).values())) if values else None}
    by_family = defaultdict(Counter)
    for r in rows:
        by_family[r["family"]]["items"] += 1
        for key in ("grep_recovers_omitted", "c4_hits_recovers_omitted", "c4_trigger_fires", "natural_miss",
                    "omitted_docref_in_answer"):
            by_family[r["family"]][key] += int(bool(r[key]))
    return {"items": len(rows), "families": len(study.families),
            "grep_recovers_omitted": share("grep_recovers_omitted"),
            "c4_hits_recovers_omitted": share("c4_hits_recovers_omitted"),
            "c4_contains_omitted_site": share("c4_contains_omitted_site"),
            "c4_trigger_fires": share("c4_trigger_fires"), "natural_miss": share("natural_miss"),
            "omitted_docref_in_answer": share("omitted_docref_in_answer"),
            "omitted_name_in_answer": share("omitted_name_in_answer"),
            "strata": dict(sorted(Counter(r["stratum"] for r in rows).items())),
            "wording_pairs": dict(sorted(Counter(r["wording_pair"] for r in rows).items())),
            "visibility_text_matches": _summary([r["visibility_text_matches"] for r in rows
                                                 if isinstance(r["visibility_text_matches"], int)]),
            "visibility_text_matches_positive": sum(1 for r in rows if (r["visibility_text_matches"] or 0) > 0),
            "V_size": dict(sorted(Counter(r["n_V"] for r in rows).items())),
            "by_family": {f: dict(sorted(c.items())) for f, c in sorted(by_family.items())},
            "membership": {flag: sum(bool(item.membership.get(flag)) for item in items.values())
                           for flag in ("in_s_nat", "in_s_edge", "in_t")},
            "s_nat_excluded_reasons": dict(sorted(Counter(str(item.membership.get("s_nat_excluded_reason"))
                                                          for item in items.values()
                                                          if not item.membership.get("in_s_nat")).items()))}


FINDABILITY_SCHEMA = "expA-findability-v2"


def eligibility_flow(items_manifest: dict, items_sha256: str, roles) -> dict:
    """D11.7 per-family eligibility flow from the items manifest bound to this items file."""
    recorded = ((items_manifest.get("outputs") or {}).get("items.jsonl") or {}).get("sha256")
    require(recorded == items_sha256, f"the items manifest describes items {recorded}, not {items_sha256}",
            S.BindingError)
    out = {}
    for family, info in sorted((items_manifest.get("families") or {}).items()):
        if not isinstance(info, dict) or info.get("role") not in roles:
            continue
        funnel = info.get("funnel") or {}
        selected = funnel.get("selected", info.get("selected"))
        out[family] = {"role": info.get("role"), "functions": funnel.get("functions"),
                       "eligible": funnel.get("eligible"), "selected": selected, "short_of_m": info.get("short_of_m"),
                       "contributes": bool(selected), "excluded": dict(sorted((funnel.get("excluded") or {}).items()))}
    return {"items_manifest_items_sha256": recorded, "families": out,
            "families_without_items": sorted(f for f, v in out.items() if not v["contributes"]),
            "note": "A family with no eligible item is reported and contributes nothing (D11.7)."}


def findability_report(plan: Plan, items: dict, policy_rows: dict, inputs: dict | None = None,
                       items_manifest: dict | None = None) -> dict:
    """D11.7 disclosure before the first main dispatch: reference policies and findability of the frozen items."""
    study = Study(plan, items, {}, policy_rows)
    flow = None
    if items_manifest is not None:
        flow = eligibility_flow(items_manifest, ((inputs or {}).get("items") or {}).get("sha256"),
                                plan.analysis_roles)
    return {"schema_version": FINDABILITY_SCHEMA, "label": plan.label, "inputs": inputs or {},
            "eligibility_flow": flow,
            "population": {"roles": list(plan.analysis_roles), "families": list(study.families),
                           "n_families": len(study.families), "n_items": len(study.items),
                           "items_by_family": dict(sorted(Counter(i.family for i in study.items.values()).items()))},
            "reference_policies": {variant: policy_summary(study, sorted(study.items), variant)
                                   for variant in (S.NO_VARIANT,) + S.VARIANTS},
            "findability": findability(study),
            "note": "No model outcome is used. Mechanical policies are references, not ceilings (D11.1); grep_all "
                    "uses frozen alias knowledge and is oracle-assisted."}


def render_findability_md(report: dict) -> str:
    population, find = report["population"], report["findability"]
    out = [f"# Exp A findability and reference policies ({report['schema_version']})", ""]
    if report.get("label"):
        out += [f"**{report['label']}**", ""]
    out += [f"Roles {population['roles']}: {population['n_families']} families, {population['n_items']} items "
            f"{population['items_by_family']}. {report['note']}", "", "## Reference policies (family-weighted)", ""]
    rows = []
    for variant, policies in report["reference_policies"].items():
        for policy, entry in policies.items():
            rows.append([variant, policy, entry.get("items", 0), entry.get("site_recovered_items", "n/a"),
                         _f(entry.get("site_recovered")), _f(entry.get("recall_V_callers")),
                         _f(entry.get("nm_any")), _f(entry.get("precision_lower")), _f(entry.get("precision_upper")),
                         _f(entry.get("n_submitted"), 2)])
    out += [_md(["Variant", "Policy", "Items", "C* items", "site_recovered", "recall", "nm_any", "prec lo", "prec hi",
                 "sites"], rows), "", "## Findability of C* (omit_node items)", ""]
    for key in ("grep_recovers_omitted", "c4_hits_recovers_omitted", "c4_contains_omitted_site", "c4_trigger_fires",
                "natural_miss", "omitted_docref_in_answer", "omitted_name_in_answer"):
        entry = find.get(key)
        if entry is None:
            continue
        out.append(f"- {key}: {entry['count']}/{entry['items']} items, family-weighted {_f(entry['family_weighted'])}")
    out += [f"- strata: {find['strata']}", f"- wording pairs: {find['wording_pairs']}",
            f"- |V|: {find['V_size']}", f"- visibility text matches > 0: {find['visibility_text_matches_positive']}",
            f"- membership: {find['membership']}", f"- S-NAT exclusions: {find['s_nat_excluded_reasons']}", "",
            "## By family", "", _md(["Family", "Counts"], [[f, json.dumps(c, sort_keys=True)]
                                                          for f, c in find["by_family"].items()]), ""]
    flow = report.get("eligibility_flow")
    if flow:
        out += ["## Eligibility flow per family (items manifest)", "",
                _md(["Family", "Functions", "Eligible", "Selected", "Short of M", "Exclusions (nonzero)"],
                    [[f, v["functions"], v["eligible"], v["selected"], v["short_of_m"],
                      json.dumps({k: n for k, n in v["excluded"].items() if n}, sort_keys=True)]
                     for f, v in flow["families"].items()]), "",
                f"Families without items: {flow['families_without_items'] or 'none'}. {flow['note']}", ""]
    return "\n".join(out)


# --------------------------------------------------------------------------- secondary (descriptive)

def _block_seeds(study, block, seeds):
    if block == PRIMARY_BLOCK:
        return list(seeds)
    return list(study.layout.block(block).seeds)


def _completed(study, block, condition, seeds, items):
    out = []
    for i in items:
        for s in seeds:
            row = study.row(block, i, condition, s)
            if row is not None and row["completed"]:
                out.append(row)
    return out


def _item_means(rows, field):
    per_item = defaultdict(list)
    for row in rows:
        value = outcome_value(row, field) if field in OUTCOMES + NM_OUTCOMES else _row_field(row, field)
        if value is not None:
            per_item[row["item_id"]].append(_frac(value))
    return {i: _mean(v) for i, v in per_item.items()}


def rate_row(study: Study, block, condition, seeds) -> dict:
    items = study.block_items(block)
    variant = study.layout.block(block).variant_of(condition)
    cells = [study.row(block, i, condition, s) for i in items for s in seeds]
    cells = [row for row in cells if row is not None]
    completed = [row for row in cells if row["completed"]]
    submitted = [row for row in completed if row["exit_status"] == "Submitted"]
    out = {"block": block, "variant": variant, "condition": condition, "seeds": list(seeds), "items": len(items),
           "cells": len(cells), "completed_cells": len(completed),
           "cell_states": {s: sum(r["cell_state"] == s for r in cells) for s in S.CELL_STATES},
           "exit_counts": {s: sum(r["exit_status"] == s for r in cells) for s in S.ROW_STATUSES}}
    outcomes = OUTCOMES + (NM_OUTCOMES if block == "S-NAT" else ())
    for outcome in outcomes:
        out[outcome] = describe(study, _item_means(completed, outcome), families=(outcome == PRIMARY_OUTCOME))
    for name in ("precision_lower", "precision_upper", "n_submitted", "n_unique_sites", "n_invalid"):
        out[name] = describe(study, _item_means(completed, name))
    out["precision_defined_cells"] = {name: sum(r[name] is not None for r in completed)
                                      for name in ("precision_lower", "precision_upper")}
    out["sites_submitted_total"] = sum(r["n_submitted"] for r in completed)
    out["categories_total"] = {c: sum(r["precision_categories"][c] for r in completed) for c in S.CATEGORIES}
    out["completeness_claims"] = {k: sum(r["complete"] == k for r in submitted) for k in S.CLAIMS}
    for flag in ("certifiably_false_completeness", "uncertain_despite_full_V"):
        out[flag] = {"count": sum(r[flag] is True for r in submitted), "submitted": len(submitted),
                     "among_submitted": describe(study, _item_means(submitted, flag))}
    out["finish_reason_length"] = {"total": sum((r.get("cost") or {}).get("finish_reason_length_count") or 0
                                                for r in completed),
                                   "episodes_with_any": sum(1 for r in completed if ((r.get("cost") or {})
                                                            .get("finish_reason_length_count") or 0) > 0)}
    out["reference_policies"] = policy_summary(study, items, variant)
    return out


def describe_rates(study: Study, seeds) -> list:
    rows = []
    for block in study.layout.blocks:
        block_seeds = _block_seeds(study, block.name, seeds)
        if not block_seeds or not study.block_items(block.name):
            continue
        for condition in block.conditions:
            rows.append(rate_row(study, block.name, condition, block_seeds))
    return rows


def _pair_rows(study, left, right, field, items):
    """Paired item values for two (block, condition, seed) cells completed on both sides."""
    lv, rv = {}, {}
    for i in items:
        a, b = study.row(left[0], i, left[1], left[2]), study.row(right[0], i, right[1], right[2])
        if a is None or b is None or not (a["completed"] and b["completed"]):
            continue
        x = outcome_value(a, field) if field in OUTCOMES + NM_OUTCOMES else _row_field(a, field)
        y = outcome_value(b, field) if field in OUTCOMES + NM_OUTCOMES else _row_field(b, field)
        if x is not None and y is not None:
            lv[i], rv[i] = _frac(x), _frac(y)
    return lv, rv


def c4_triggered(study: Study, seeds) -> dict:
    """C4 triggered = C4a when the frozen trigger fires on the item, else C1 (reconstructed, D5)."""
    items = study.block_items(PRIMARY_BLOCK)
    c1, _ = arm_values(study, "C1", seeds)
    c4, _ = arm_values(study, "C4a", seeds)
    fires = {i: bool(study.items[i].covariates.get("c4_trigger_fires")) for i in items}
    trig = {i: (c4[i] if fires[i] else c1[i]) for i in set(c1) & set(c4)}
    prevalence = family_means(study, {i: Fraction(int(fires[i])) for i in items})
    return {"role": "secondary_descriptive", "seeds": list(seeds), "trigger_items": sum(fires.values()),
            "items": len(items), "trigger_prevalence_family_weighted": float(_mean(prevalence.values()))
            if prevalence else None, "c4_triggered_minus_c1": paired(study, trig, c1)}


def _hypothesis_table(study, seeds, item_ids):
    out = {}
    for hypothesis, treatment, control in HYPOTHESES:
        treat, ctrl, _ = contrast_values(study, treatment, control, seeds, items=item_ids)
        out[hypothesis] = paired(study, treat, ctrl) or {"n_items": 0}
    return out


def strata(study: Study, seeds) -> dict:
    """Secondary strata of C* (D11.9): stratum, grep recoverability, C4 visibility, wording pair."""
    items = study.block_items(PRIMARY_BLOCK)

    def grep(i):
        row = study.policies.get((i, PRIMARY_VARIANT, "grep_all"))
        return None if row is None else ("grep_recovers" if row["site_recovered"] else "grep_misses")

    def c4(i):
        value = study.items[i].covariates.get("c4_contains_omitted_site")
        return None if value is None else ("c4_shows_site" if value else "c4_lacks_site")
    splits = {"stratum": lambda i: study.items[i].covariates.get("stratum"), "grep_recoverability": grep,
              "c4_visibility": c4, "wording_pair": lambda i: study.items[i].wording_pair}
    out = {"role": "secondary_descriptive", "seeds": list(seeds)}
    for name, key in splits.items():
        groups = defaultdict(list)
        for i in items:
            groups[key(i)].append(i)
        entry = {}
        for value in sorted(groups, key=str):
            ids = groups[value]
            rates = {}
            for condition in study.layout.block(PRIMARY_BLOCK).conditions:
                values, _ = arm_values(study, condition, seeds, items=ids)
                rates[condition] = describe(study, values)
            if "C0" not in rates and study.layout.has("S-C0"):
                values, _ = arm_values(study, "C0", seeds, items=ids)
                rates["C0"] = describe(study, values)
            entry[str(value)] = {"items": len(ids), "rates": rates, "contrasts": _hypothesis_table(study, seeds, ids)}
        out[name] = entry
    return out


def s_vc(study: Study) -> dict:
    block = "S-VC"
    if not study.layout.has(block):
        return {"status": "not_in_design"}
    seed = study.plan.secondary_seed
    items = study.block_items(block)
    out = {"role": "secondary_descriptive", "question": "over-caution on vcomplete: C3 vs C1 (seed 11)",
           "items": len(items), "differences": {}}
    for name in OUTCOMES + ("precision_lower", "precision_upper", "n_submitted", "n_unique_sites", "complete_yes",
                            "calls", "total_tokens"):
        c3, c1 = _pair_rows(study, (block, "C3", seed), (block, "C1", seed), name, items)
        out["differences"][name] = paired(study, c3, c1)
    return out


def s_nat(study: Study) -> dict:
    block = "S-NAT"
    if not study.layout.has(block):
        return {"status": "not_in_design"}
    seed = study.plan.secondary_seed
    items = study.block_items(block)
    out = {"role": "secondary_descriptive", "items": len(items), "outcomes": list(NM_OUTCOMES),
           "note": "NM = runtime-witnessed V callers absent from the natural answer; C* is not used (D11.5)",
           "items_without_NM": sorted(i for i in items if not study.items[i].nm) if any(
               study.items[i].nm is not None for i in items) else "NM not attached (no payload store at analysis)",
           "rates": {}, "differences": {}}
    for condition in study.layout.block(block).conditions:
        rows = _completed(study, block, condition, [seed], items)
        out["rates"][condition] = {name: describe(study, _item_means(rows, name)) for name in NM_OUTCOMES}
    for treatment in ("C3", "C4a"):
        for name in NM_OUTCOMES:
            t, c = _pair_rows(study, (block, treatment, seed), (block, "C1", seed), name, items)
            out["differences"][f"{treatment}_minus_C1:{name}"] = paired(study, t, c)
    return out


def s_edge_vs_p(study: Study) -> dict:
    if not study.layout.has("S-EDGE"):
        return {"status": "not_in_design"}
    seed = study.plan.secondary_seed
    items = sorted(set(study.block_items("S-EDGE")) & set(study.block_items(PRIMARY_BLOCK)))
    out = {"role": "secondary_descriptive", "question": "visible endpoint: omit_edge (S-EDGE) vs omit_node (P), "
                                                        "seed 11 only (D11.4)", "items": len(items), "differences": {}}
    for condition in ("C1", "C3"):
        e, n = _pair_rows(study, ("S-EDGE", condition, seed), (PRIMARY_BLOCK, condition, seed), PRIMARY_OUTCOME, items)
        out["differences"][f"edge_minus_node_{condition}"] = paired(study, e, n)
    c3, c1 = _pair_rows(study, ("S-EDGE", "C3", seed), ("S-EDGE", "C1", seed), PRIMARY_OUTCOME, items)
    out["differences"]["edge_C3_minus_C1"] = paired(study, c3, c1)
    return out


def s_c0_block(study: Study) -> dict:
    """contract-v2 only: P C1 vs S-C0 C0 at seed 11 (H3 is in the primary analysis either way)."""
    if not study.layout.has("S-C0"):
        return {"status": "not_in_design", "note": "C0 is part of block P (D11.4)"}
    seed = study.plan.secondary_seed
    items = sorted(set(study.block_items("S-C0")) & set(study.block_items(PRIMARY_BLOCK)))
    p, c0 = _pair_rows(study, (PRIMARY_BLOCK, "C1", seed), ("S-C0", "C0", seed), PRIMARY_OUTCOME, items)
    return {"role": "secondary_descriptive", "P_C1_minus_S_C0_seed_11": paired(study, p, c0)}


def _graph_funnel(row):
    exposure = row.get("exposure") or {}
    cost = row.get("cost") or {}
    offered = GRAPH_QUERY_NAME in (exposure.get("tools_offered") or [])
    requested_calls = (cost.get("tool_calls") or {}).get(GRAPH_QUERY_NAME, 0) or 0
    requested = requested_calls > 0 or (cost.get("graph_query_requests_in_format_errors") or 0) > 0
    statuses = cost.get("graph_query_statuses")
    if statuses is None:
        executed = None if cost.get("graph_query_calls") is None else cost["graph_query_calls"] > 0
    else:
        executed = any(status not in GRAPH_REJECTED for status in statuses)
    nonempty = cost.get("graph_query_nonempty_answers")
    return {"offered": offered, "requested": requested, "executed": executed,
            "visible_nonempty_answer": None if nonempty is None else nonempty > 0}


GRAPH_QUERY_NAME = S.GRAPH_QUERY


def t_block(study: Study) -> dict:
    block = "T"
    if not study.layout.has(block):
        return {"status": "not_in_design"}
    seed = study.plan.secondary_seed
    items = study.block_items(block)
    arms, requested_values = {}, {}
    for condition in S.T_CONDITIONS:
        rows = [study.row(block, i, condition, seed) for i in items]
        rows = [r for r in rows if r is not None]
        completed = [r for r in rows if r["completed"]]
        funnel = Counter()
        unknown = Counter()
        requested = {}
        for r in completed:
            stages = _graph_funnel(r)
            for stage, value in stages.items():
                if value is None:
                    unknown[stage] += 1
                elif value:
                    funnel[stage] += 1
            requested[r["item_id"]] = Fraction(int(stages["requested"]))
        requested_values[condition] = requested
        arms[condition] = {"cells": len(rows), "completed": len(completed),
                           "funnel": {stage: funnel.get(stage, 0) for stage in
                                      ("offered", "requested", "executed", "visible_nonempty_answer")},
                           "funnel_unknown": dict(sorted(unknown.items())),
                           "consumed": "not derivable from result receipts (needs an events-level reading)",
                           "recommendation_present": sum(1 for r in completed
                                                         if (r.get("exposure") or {}).get("recommendation_present")),
                           "requested_rate": describe(study, requested),
                           "site_recovered": describe(study, _item_means(completed, PRIMARY_OUTCOME))}
    rec, off = _pair_rows(study, (block, "T-recommended", seed), (block, "T-offered", seed), PRIMARY_OUTCOME, items)
    return {"role": "exploratory", "items": len(items), "arms": arms,
            "recommended_minus_offered_requested": paired(study, requested_values["T-recommended"],
                                                          requested_values["T-offered"]),
            "recommended_minus_offered_site_recovered": paired(study, rec, off)}


def process_descriptives(study: Study, seeds) -> list:
    """D11.9: searches for X's names, mentions of C*'s file and tool calls before submission, by condition."""
    out = []
    for block in study.layout.blocks:
        block_seeds = _block_seeds(study, block.name, seeds)
        items = study.block_items(block.name)
        if not block_seeds or not items:
            continue
        for condition in block.conditions:
            rows = _completed(study, block.name, condition, block_seeds, items)
            with_telemetry = [r for r in rows if r.get("telemetry")]
            entry = {"block": block.name, "condition": condition, "completed_cells": len(rows),
                     "cells_with_telemetry": len(with_telemetry)}
            for field in ("searched_target_name", "mentioned_omitted_caller_file", "tool_calls_before_submit"):
                entry[field] = describe(study, _item_means(with_telemetry, field))
            out.append(entry)
    return out


def exposure_audit(study: Study, seeds) -> dict:
    rows_out = []
    for block in study.layout.blocks:
        items = study.block_items(block.name)
        if not items:
            continue
        for condition in block.conditions:
            cells = [study.row(block.name, i, condition, s) for i in items for s in block.seeds]
            cells = [r for r in cells if r is not None]
            completed = [r for r in cells if r["completed"]]
            expected_payload = condition in S.ANSWER_CONDITIONS
            counts, tokens = Counter(), []
            for r in completed:
                exposure = r.get("exposure") or {}
                present = exposure.get("payload_present")
                counts["payload_delivered_" + {True: "true", False: "false"}.get(present, "unknown")] += 1
                if present is not expected_payload:
                    counts["payload_contradicts_condition"] += 1
                if expected_payload and exposure.get("payload_sha256") != exposure.get("expected_payload_sha256"):
                    counts["payload_sha256_mismatch"] += 1
                verification = exposure.get("verification_block_delivered")
                counts["verification_delivered_" + {True: "true", False: "false"}.get(verification, "unknown")] += 1
                if verification is not (condition == "C4a"):
                    counts["verification_contradicts_condition"] += 1
                offered = GRAPH_QUERY_NAME in (exposure.get("tools_offered") or [])
                if offered is not (condition in S.T_CONDITIONS):
                    counts["graph_query_offer_contradicts_condition"] += 1
                if (exposure.get("recommendation_present") is True) is not (condition == "T-recommended"):
                    counts["recommendation_contradicts_condition"] += 1
                if isinstance(exposure.get("payload_tokens"), int):
                    tokens.append(exposure["payload_tokens"])
            rows_out.append({"block": block.name, "variant": block.variant_of(condition), "condition": condition,
                             "seeds": list(block.seeds), "cells": len(cells), "completed_cells": len(completed),
                             "counts": dict(sorted(counts.items())), "payload_tokens": _summary(tokens)})
    return {"rows": rows_out, "per_cell": "CELLS.jsonl carries every receipt field per cell",
            "note": "From each terminal attempt's exposure receipt; cells without a terminal attempt have none."}


def cost_table(study: Study, seeds) -> list:
    rows = []
    for block in study.layout.blocks:
        block_seeds = _block_seeds(study, block.name, seeds)
        items = study.block_items(block.name)
        if not items or not block_seeds:
            continue
        for condition in block.conditions:
            completed = _completed(study, block.name, condition, block_seeds, items)
            with_cost = [r for r in completed if r.get("cost")]
            entry = {"block": block.name, "condition": condition, "seeds": block_seeds,
                     "completed_cells": len(completed), "cells_with_cost": len(with_cost), "per_episode_mean": {},
                     "totals": {}}
            for name in COST_FIELDS + ("total_tokens",):
                values = [v for v in (_row_field(r, name) for r in with_cost) if isinstance(v, int)]
                entry["per_episode_mean"][name] = statistics.fmean(values) if values else None
                entry["totals"][name] = sum(values)
            tools = Counter()
            for r in with_cost:
                tools.update(r["cost"].get("tool_calls") or {})
            entry["tool_calls_by_name"] = dict(sorted(tools.items()))
            entry["family_total_tokens"] = describe(study, _item_means(with_cost, "total_tokens"))
            rows.append(entry)
    return rows


def endpoint_table(study: Study) -> dict:
    """D11.8: completed cells by condition and endpoint (one shared queue)."""
    table = defaultdict(Counter)
    gpus = defaultdict(set)
    for cell in study.cells:
        row = study.rows[cell["cell_id"]]
        if not row["completed"]:
            continue
        endpoint = str(row.get("endpoint"))
        table[f"{cell['block']}|{cell['condition']}"][endpoint] += 1
        gpus[endpoint].add(str(row.get("gpu_uuid")))
    return {"cells": {key: dict(sorted(c.items())) for key, c in sorted(table.items())},
            "endpoint_gpus": {k: sorted(v) for k, v in sorted(gpus.items())}}


# --------------------------------------------------------------------------- orchestration

def _cell_rows(study: Study, seeds) -> list:
    out = []
    for cell in study.cells:
        row = study.rows[cell["cell_id"]]
        item = study.items[cell["item_id"]]
        exposure = row.get("exposure") or {}
        cost = row.get("cost") or {}
        telemetry = row.get("telemetry") or {}
        out.append({"cell_id": cell["cell_id"], "block": cell["block"], "item_id": cell["item_id"],
                    "family": item.family, "variant": cell["variant"], "condition": cell["condition"],
                    "seed": cell["seed"], "wording_pair": item.wording_pair, "stratum": item.covariates.get("stratum"),
                    "in_primary_analysis": cell["block"] == PRIMARY_BLOCK and cell["seed"] in seeds
                    or (cell["condition"] == "C0" and cell["seed"] in seeds),
                    "cell_state": row["cell_state"], "exit_status": row["exit_status"],
                    "attempt_id": row.get("attempt_id"), "infra_attempts": row.get("infra_attempts"),
                    "result_sha256": row.get("result_sha256"), "endpoint": row.get("endpoint"),
                    "gpu_uuid": row.get("gpu_uuid"), "site_recovered": row["site_recovered"],
                    "function_localized": row["function_localized"], "recall_V_callers": row["recall_V_callers"],
                    "nm_recall": row["nm_recall"], "nm_any": row["nm_any"], "n_submitted": row["n_submitted"],
                    "n_unique_sites": row["n_unique_sites"], "precision_lower": row["precision_lower"],
                    "precision_upper": row["precision_upper"], "complete": row["complete"],
                    **{name: cost.get(name) for name in COST_FIELDS},
                    **{f"exposure_{name}": exposure.get(name) for name in S.EXPOSURE_FIELDS},
                    **{f"telemetry_{name}": telemetry.get(name) for name in
                       ("searched_target_name", "mentioned_omitted_caller_file", "tool_calls_before_submit")}})
    return out


def run_analysis(plan: Plan, items: dict, score_rows: dict, policy_rows: dict, inputs: dict | None = None,
                 binding: dict | None = None, extra: dict | None = None) -> tuple:
    study = Study(plan, items, score_rows, policy_rows)
    primary = primary_analysis(study)
    seeds = primary["primary_seeds"]
    describe_seeds = seeds or [s for s in plan.p_seeds
                               if primary["seed_completion"][str(s)]["cell_states"]["completed"]]
    states = Counter(study.rows[c["cell_id"]]["cell_state"] for c in study.cells)
    by_block = defaultdict(Counter)
    for cell in study.cells:
        by_block[cell["block"]][study.rows[cell["cell_id"]]["cell_state"]] += 1
    complete = states.get("completed", 0) == len(study.cells)
    analysis = {
        "schema_version": SCHEMA_VERSION, "label": plan.label,
        "status": "complete" if complete else "incomplete_cells_reported",
        "inputs": inputs or {}, "binding": binding or {}, "plan": plan.to_dict(),
        "design": study.layout.describe(),
        "population": {"roles": list(plan.analysis_roles), "families": list(study.families),
                       "n_families": len(study.families), "n_items": len(study.items),
                       "items_by_family": dict(sorted(Counter(i.family for i in study.items.values()).items())),
                       "items_excluded_by_role": dict(sorted(Counter(i.role for i in items.values()
                                                                      if i.role not in plan.analysis_roles).items())),
                       "scope": SCOPE_STATEMENT},
        "completion": {"design_cells": len(study.cells), "cell_states": {s: states.get(s, 0) for s in S.CELL_STATES},
                       "by_block": {block: {s: counts.get(s, 0) for s in S.CELL_STATES}
                                    for block, counts in sorted(by_block.items())},
                       "retry_cap_exceeded": sum(1 for c in study.cells
                                                 if study.rows[c["cell_id"]].get("retry_cap_exceeded"))},
        "primary": primary,
        "reference_policies": {"by_variant": {variant: policy_summary(study, sorted(study.items), variant)
                                              for variant in (S.NO_VARIANT,) + S.VARIANTS},
                               "h2_vs_c4_hits_reference": h2_reference(study, seeds),
                               "note": "Model-free submissions scored like episodes (D10.11). They are references, "
                                       "not ceilings, and change no test (D11.1)."},
        "findability": findability(study),
        "secondary": {"rates": describe_rates(study, describe_seeds),
                      "rates_seed_note": "P rows use the primary seeds" if seeds else
                      "no complete P seed: P rows use every seed with completed cells (descriptive only)",
                      "c4_triggered": c4_triggered(study, seeds) if seeds else {"status": "not_estimable"},
                      "strata": strata(study, seeds) if seeds else {"status": "not_estimable"},
                      "s_vc": s_vc(study), "s_nat": s_nat(study), "s_edge_vs_p": s_edge_vs_p(study),
                      "s_c0": s_c0_block(study), "t_block": t_block(study)},
        "process_descriptives": {"note": "Reported by condition, never conditioned on, never mediators (D11.9).",
                                 "rows": process_descriptives(study, describe_seeds)},
        "exposure_audit": exposure_audit(study, seeds),
        "cost": cost_table(study, describe_seeds),
        "endpoints": endpoint_table(study),
        "methods": {
            "unit": "item mean over the primary seeds (completed cells) -> paired difference within item -> family "
                    "mean -> equal-weight mean over families",
            "primary_test": PRIMARY_TEST + "; Holm over the fixed family H1, H2, H3 (" + HOLM_NOT_ESTIMABLE_RULE
                            + "); 95% t intervals are marginal",
            "zero_variance": "all family differences 0: p = 1, degenerate interval (not equivalence); identical "
                             "nonzero differences: t undefined, the sign-flip p is used",
            "seeds": primary["seed_rule"],
            "missing_cells": "item-condition means use completed cells; bounds impute 0 and then 1 for each "
                             "missing cell, plus the directional extremes",
            "minimum_data": f"a contrast with fewer than {plan.min_families_confirmatory} contributing families is "
                            "exploratory",
            "sensitivity": "sign-flip test (exact when F <= " + str(plan.sign_flip_exact_max_families) + ", else "
                           + str(plan.sign_flip_draws) + " draws, seed " + str(plan.sign_flip_seed)
                           + ", p = (1 + hits) / (1 + draws)); family cluster percentile bootstrap ("
                           + str(plan.bootstrap_resamples) + " resamples, seed " + str(plan.bootstrap_seed) + ")",
            "symmetry_assumption": SYMMETRY_ASSUMPTION,
            "secondary": "descriptive estimates with nominal 95% family-clustered intervals, no significance "
                         "claims, never promoted",
            "denominators": "completed model exits other than Submitted score 0; cells without a terminal attempt "
                            "are excluded from means and counted"},
    }
    if extra:
        analysis.update(extra)
    return analysis, _cell_rows(study, seeds)


def _write(path: Path, data: bytes, overwrite: bool):
    if path.exists():
        require(overwrite, f"refusing to overwrite {path} (pass overwrite=True)", FileExistsError)
        os.chmod(path, 0o644)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def write_outputs(out_dir, analysis: dict, cell_rows: list, *, overwrite=False) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cells = b"".join(canonical_json(row).encode("utf-8") + b"\n" for row in cell_rows)
    analysis = dict(analysis)
    analysis["outputs"] = {"CELLS.jsonl": {"sha256": sha256_bytes(cells), "rows": len(cell_rows), "bytes": len(cells)}}
    body = dump_json(analysis)
    tables = render_tables_md(analysis).encode("utf-8")
    written = {}
    for name, data in (("CELLS.jsonl", cells), ("ANALYSIS.json", body), ("TABLES.md", tables)):
        _write(out / name, data, overwrite)
        written[name] = {"sha256": sha256_bytes(data), "bytes": len(data)}
    return written


def _scores_manifest(scores_path):
    path = Path(str(scores_path) + ".manifest.json")
    if not path.is_file():
        return None, None
    raw = path.read_bytes()
    manifest = strict_loads(raw)
    require(((manifest.get("out") or {}).get("sha256")) == sha256_bytes(Path(scores_path).read_bytes()),
            f"{path.name} does not describe {Path(scores_path).name}", S.BindingError)
    return manifest, {"path": str(path), "sha256": sha256_bytes(raw)}


def analyze_paths(*, items_path, scores_path, policies_path, plan_path, out_dir, run_dir=None, dispatch_manifest=None,
                  require_run=True, overwrite=False, plan_sha256=None, payloads_dir=None):
    plan, plan_meta = load_plan(plan_path)
    require(plan_sha256 is None or plan_meta["sha256"] == plan_sha256, "analysis plan sha256 does not match the pin")
    items, items_meta = S.load_items(items_path)
    score_rows, score_meta = load_score_rows(scores_path)
    require(score_meta["design"] == plan.design, f"the score rows use design {score_meta['design']}, the plan "
                                                 f"{plan.design}", S.BindingError)
    policy_rows, policy_meta = load_policy_rows(policies_path)
    manifest, manifest_meta = _scores_manifest(scores_path)
    binding = check_binding(items, items_meta, score_rows, policy_rows, design=plan.design, run_dir=run_dir,
                            dispatch_manifest=dispatch_manifest, scores_manifest=manifest, require_run=require_run)
    payloads = Path(payloads_dir) if payloads_dir else Path(items_path).parent / "payloads"
    if payloads.is_dir():
        store = S.PayloadStore(payloads)
        for item in items.values():
            S.attach_natural_answer(item, store)
    inputs = {"plan": plan_meta, "items": items_meta, "scores": score_meta, "policies": policy_meta}
    if manifest_meta:
        inputs["scores_manifest"] = manifest_meta
    analysis, rows = run_analysis(plan, items, score_rows, policy_rows, inputs, binding)
    return analysis, write_outputs(out_dir, analysis, rows, overwrite=overwrite)


# --------------------------------------------------------------------------- TABLES.md

def _f(value, digits=3):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _p(value):
    if value is None:
        return "n/a"
    return f"{value:.2e}" if value < 1e-3 else f"{value:.4f}"


def _ci(interval, digits=3):
    if not interval or interval.get("lower") is None:
        return "n/a"
    return f"[{interval['lower']:.{digits}f}, {interval['upper']:.{digits}f}]"


def _md(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def _est(result, digits=3):
    if not result or result.get("estimate") is None:
        return "n/a"
    return (f"{result['estimate']:+.{digits}f} t {_ci(result.get('t_interval'), digits)} "
            f"({result['n_items']} items, {result['n_families']} fam)")


def _flip(entry):
    flip = entry.get("sign_flip")
    if not flip:
        return "n/a"
    if flip["method"] == "monte_carlo":
        return f"{_p(flip['p_two_sided'])} (MC {flip['draws']}, seed {flip['seed']})"
    return f"{_p(flip['p_two_sided'])} (exact 2^{flip['n_families']})"


def render_tables_md(analysis: dict) -> str:
    primary, population = analysis["primary"], analysis["population"]
    out = [f"# Exp A analysis tables ({analysis['schema_version']})", ""]
    if analysis.get("label"):
        out += [f"**{analysis['label']}**", ""]
    out += [f"Status `{analysis['status']}`. Design `{analysis['design']['name']}`. Population roles "
            f"{population['roles']}: {population['n_families']} families, {population['n_items']} items. "
            f"Design cells {analysis['completion']['design_cells']}. {population['scope']}", ""]
    inputs = analysis.get("inputs") or {}
    if inputs:
        out += ["Inputs: " + "; ".join(f"{k} sha256 `{v.get('sha256', '')}`" for k, v in sorted(inputs.items())
                                       if isinstance(v, dict)), ""]
    out += ["## Cell states by block", "",
            _md(["Block"] + list(S.CELL_STATES),
                [[block] + [counts[s] for s in S.CELL_STATES] for block, counts in
                 analysis["completion"]["by_block"].items()]), "",
            "## P seeds (D11.3)", "",
            _md(["Seed", "Status", "Terminal / cells", "Completed", "Missing (infra)", "Pending", "Not dispatched"],
                [[seed, c["status"], f"{c['terminal_cells']}/{c['cells']}", c["cell_states"]["completed"],
                  c["cell_states"]["missing_infrastructure"], c["cell_states"]["infra_pending"],
                  c["cell_states"]["not_dispatched"]] for seed, c in primary["seed_completion"].items()]), "",
            f"Primary seeds {primary['primary_seeds']}; sensitivity seeds {primary['sensitivity_seeds']}.", "",
            "## Primary hypotheses (block P, site_recovered for C*)", "",
            f"{primary['test']}; Holm family {', '.join(primary.get('holm_family') or [])}"
            + (f" (not estimable, entered with p = 1: {', '.join(primary['holm_entered_as_p1'])})"
               if primary.get("holm_entered_as_p1") else "")
            + f"; alpha {primary['alpha']}; a contrast with fewer than "
            f"{primary['minimum_families_confirmatory']} families is exploratory.", ""]
    rows = []
    for h in primary["hypotheses"]:
        test = h.get("t_test") or {}
        interval = test.get("interval")
        rows.append([h["id"], f"{h['treatment']} - {h['control']}", h.get("status"), _f(h.get("estimate")),
                     _ci(interval), _f(test.get("t")), _f(test.get("df")), _p(h.get("p_primary")),
                     h.get("p_source", "n/a"), _p(h.get("holm_adjusted_p")), _f(h.get("reject_at_alpha")),
                     h.get("n_items", 0), h.get("n_families", 0), _flip(h), _ci(h.get("bootstrap"))])
    out += [_md(["H", "Contrast", "Status", "Estimate", "95% t CI", "t", "df", "p", "p source", "Holm p",
                 "Reject", "Items", "Families", "Sign-flip p", "Bootstrap CI"], rows), ""]
    bound_rows = []
    for h in primary["hypotheses"]:
        bounds = h.get("bounds_missing_cells") or {}
        bound_rows.append([h["id"], *[_f((bounds.get(k) or {}).get("estimate")) for k in
                                      ("impute_0", "impute_1", "lowest", "highest")],
                           json.dumps(h.get("missing_cells_in_primary_seeds", {}), sort_keys=True)])
    out += ["Missing-cell bounds (D11.3):", "",
            _md(["H", "impute 0", "impute 1", "lowest", "highest", "missing cells"], bound_rows), ""]
    for h in primary["hypotheses"]:
        if h.get("estimand_note"):
            out += [f"- {h['id']}: {h['estimand_note']}"]
    out += ["", "## Rates by block and condition (secondary rows descriptive)", "",
            analysis["secondary"]["rates_seed_note"], ""]
    rate_rows = []
    for r in analysis["secondary"]["rates"]:
        pol = r["reference_policies"]
        rate_rows.append([r["block"], r["variant"], r["condition"], ",".join(map(str, r["seeds"])),
                          f"{r['completed_cells']}/{r['cells']}", _est(r.get("site_recovered")),
                          _f((r.get("recall_V_callers") or {}).get("estimate")),
                          _f((r.get("precision_lower") or {}).get("estimate")),
                          _f((r.get("precision_upper") or {}).get("estimate")),
                          _f((r.get("n_submitted") or {}).get("estimate"), 2),
                          " ".join(f"{k[:2]}{v}" for k, v in r["exit_counts"].items() if v),
                          " / ".join(_f(pol[p].get("site_recovered")) for p in S.POLICIES)])
    out += [_md(["Block", "Variant", "Cond", "Seeds", "Done", "site_recovered (95% t)", "recall", "prec lo",
                 "prec hi", "sites", "exits", "answer / c4 / union / grep"], rate_rows), ""]
    ref = analysis["reference_policies"]["h2_vs_c4_hits_reference"]
    if ref.get("status") == "estimated":
        out += ["H2 next to the c4_hits reference (references, not ceilings): "
                f"C4a - C1 = {_f(ref['c4a_minus_c1'])}; answer+c4 - answer = "
                f"{_f(ref['reference_gap_answer_plus_c4_minus_answer_only'])}; share "
                f"{_f(ref['c4a_minus_c1_share_of_reference_gap'])}.", ""]
    find = analysis["findability"]
    out += ["## Findability of C* (frozen items, no outcomes)", "",
            f"grep recovers C*: {find['grep_recovers_omitted']['count']}/{find['grep_recovers_omitted']['items']}; "
            f"c4_hits recovers C*: {find['c4_hits_recovers_omitted']['count']}/"
            f"{find['c4_hits_recovers_omitted']['items']}; C4 trigger fires: {find['c4_trigger_fires']['count']}; "
            f"strata {find['strata']}; wording pairs {find['wording_pairs']}.", ""]
    sec = analysis["secondary"]
    out += ["## Secondary contrasts (descriptive, nominal 95% intervals)", ""]
    lines = []
    if isinstance(sec["c4_triggered"], dict) and sec["c4_triggered"].get("c4_triggered_minus_c1") is not None:
        lines.append(["C4 triggered - C1 (P)", _est(sec["c4_triggered"]["c4_triggered_minus_c1"])])
    for key, label in (("s_vc", "S-VC C3 - C1"), ("s_nat", "S-NAT"), ("s_edge_vs_p", "S-EDGE vs P")):
        for name, result in ((sec[key].get("differences") or {}).items()):
            lines.append([f"{label}: {name}", _est(result)])
    t = sec["t_block"]
    if t.get("arms"):
        lines.append(["T recommended - offered (requested)", _est(t.get("recommended_minus_offered_requested"))])
        lines.append(["T recommended - offered (site_recovered)",
                      _est(t.get("recommended_minus_offered_site_recovered"))])
    out += [_md(["Contrast", "Estimate"], lines) if lines else "No secondary contrast is estimable.", ""]
    out += ["## Exposure audit (receipts)", "",
            _md(["Block", "Cond", "Completed", "Counts"],
                [[r["block"], r["condition"], f"{r['completed_cells']}/{r['cells']}",
                  json.dumps(r["counts"], sort_keys=True)] for r in analysis["exposure_audit"]["rows"]]), "",
            "## Cost per episode (means)", "",
            _md(["Block", "Cond", "Episodes", "Calls", "Prompt tok", "Completion tok", "length stops"],
                [[r["block"], r["condition"], r["cells_with_cost"], _f(r["per_episode_mean"]["calls"], 2),
                  _f(r["per_episode_mean"]["prompt_tokens"], 0), _f(r["per_episode_mean"]["completion_tokens"], 0),
                  r["totals"]["finish_reason_length_count"]] for r in analysis["cost"]]), "",
            "## Condition x endpoint (completed cells)", "",
            _md(["Block|Cond", "Endpoints"], [[k, json.dumps(v, sort_keys=True)] for k, v in
                                              analysis["endpoints"]["cells"].items()]) or "none", ""]
    out += ["## Methods", ""] + [f"- **{k}**: {v}" for k, v in analysis["methods"].items()] + [""]
    return "\n".join(out)


# --------------------------------------------------------------------------- synthetic validation fixtures

_SYNTHETIC_RATES = {
    "known_effect": {"C0": 0.45, "C1": 0.30, "C2": 0.30, "C3": 0.50, "C4a": 0.60},
    "null": {"C0": 0.35, "C1": 0.35, "C2": 0.35, "C3": 0.35, "C4a": 0.35},
}


def _hex(rng):
    return "%064x" % rng.getrandbits(256)


def synthetic_item(rng, family, k, role="main"):
    """A valid expA-item-v2 record with made-up spans; no repository is needed."""
    path = f"{family}_pkg/mod{k % 3}.py"
    target = f"{path}::target_{k}"
    callers = [f"{path}::caller_{k}_{j}" for j in range(3)]
    sites = {c: [{"path": path, "lineno": 100 * (j + 1) + k, "end_lineno": 100 * (j + 1) + k + j, "kind": "call",
                  "witnesses": ["trace", "ast"]}] for j, c in enumerate(callers)}
    item_id = hashlib.sha256(f"{family}|{target}".encode()).hexdigest()[:20]
    stratum = ("direct", "alias", "callback", "dynamic")[k % 4]
    cells = {v: {c: _hex(rng) for c in S.ANSWER_CONDITIONS} for v in S.VARIANTS}
    return {"schema": S.ITEM_SCHEMA, "item_id": item_id, "family": family, "role": role,
            "package_dirs": [f"{family}_pkg"],
            "target": {"identity": target, "path": path, "line": 10 + k, "end_line": 12 + k,
                       "qualname": f"target_{k}", "name": f"target_{k}"},
            "V": [{"identity": c, "sources": ["trace"], "runtime_witnessed": True, "stratum": stratum,
                   "sites": sites[c]} for c in callers],
            "U": [], "omitted": {"identity": callers[0], "stratum": stratum, "rule": "first_runtime_witnessed",
                                 "order_key": _hex(rng)},
            "wording_pair": "AB"[k % 2],
            "covariates": {"stratum": stratum, "c4_contains_omitted_site": k % 3 != 0, "c4_trigger_fires": k % 5 == 0,
                           "natural_miss": k % 2 == 0, "n_V": 3, "visibility_text_matches": k % 2},
            "membership": {"in_s_nat": k % 2 == 0, "in_s_edge": k < 2, "in_t": k < 2,
                           "s_nat_excluded_reason": None if k % 2 == 0 else "no_natural_miss"},
            "lexical_sites": [{"path": path, "line": s[0]["lineno"]} for s in sites.values()],
            "render": {"source_sha256": _hex(rng), "snippet_caps": {"target": 2048, "caller": 2048},
                       "cap_bytes": 12000},
            "variants": {v: {"facts_sha256": _hex(rng), "endpoint_ids": [], "payload_truncated": False,
                             "cells": cells[v], "cell_bytes": {c: 1000 for c in S.ANSWER_CONDITIONS}}
                         for v in S.VARIANTS},
            "eligibility": {"eligible": True}}


def _clip(p):
    return min(0.98, max(0.02, p))


def _synthetic_row(cell, item, recovered, *, state, exit_status, items_sha256, design, rng, endpoint):
    completed = state == "completed"
    base = {"schema": S.SCORE_SCHEMA, "kind": "episode", "design": design, "cell_id": cell["cell_id"],
            "block": cell["block"], "item_id": cell["item_id"], "family": item["family"], "variant": cell["variant"],
            "condition": cell["condition"], "seed": cell["seed"], "wording_pair": item["wording_pair"],
            "omitted": item["omitted"]["identity"], "items_sha256": items_sha256, "cell_state": state,
            "completed": completed, "infra_attempts": 0 if state in ("completed", "not_dispatched") else 3,
            "retry_cap_exceeded": False, "attempts": []}
    n_v = len(item["V"])
    if not completed:
        base.update(exit_status=S.MISSING if state == "not_dispatched" else S.INFRA_EXIT, attempt_id=None,
                    result_sha256=None, cost=None, exposure=None, endpoint=None, gpu_uuid=None, telemetry=None,
                    site_recovered=None, function_localized=None, recall_V_callers=None, recall_V_sites=None,
                    recovered_V=[], n_V=n_v, nm_recall=None, nm_any=None, n_NM=None, n_submitted=None,
                    n_unique_sites=None, n_invalid=None, precision_categories=None, unresolved_reasons=None,
                    precision_lower=None, precision_upper=None, complete=None, certifiably_false_completeness=None,
                    uncertain_despite_full_V=None)
        return base
    submitted = exit_status == "Submitted"
    callers = [v["identity"] for v in item["V"]]
    recovered_v = sorted(([callers[0]] if recovered and submitted else []) + ([callers[1]] if submitted else []))
    n_sub = len(recovered_v) + (1 if submitted else 0)
    cats = {"v_site": len(recovered_v), "u_site": 0, "lexical_name_match": 0, "unresolved": 1 if submitted else 0,
            "duplicate": 0}
    payload = cell["condition"] in S.ANSWER_CONDITIONS
    expected = item["variants"][cell["variant"]]["cells"][cell["condition"]] if payload else None
    nm = item["covariates"]["natural_miss"]
    base.update(exit_status=exit_status, attempt_id=f"{cell['cell_id']}.a001", result_sha256=_hex(rng),
                cost={"calls": rng.randint(2, 8), "prompt_tokens": rng.randint(2000, 9000),
                      "completion_tokens": rng.randint(100, 900), "finish_reason_length_count": 0,
                      "tool_calls": {"bash": rng.randint(0, 5), "submit_answer": int(submitted)},
                      "graph_query_calls": 0, "graph_query_statuses": {}, "graph_query_nonempty_answers": 0,
                      "graph_query_requests_in_format_errors": 0},
                exposure={"payload_present": payload, "payload_sha256": expected, "expected_payload_sha256": expected,
                          "payload_tokens": 900 if payload else 0, "payload_bytes": 3000 if payload else 0,
                          "verification_block_delivered": cell["condition"] == "C4a", "verification_sha256": None,
                          "verification_hits": 5 if cell["condition"] == "C4a" else None, "guidance_present": None,
                          "tools_offered": ["source_reference_search", "bash", "submit_answer"]
                          + (["graph_query"] if cell["condition"] in S.T_CONDITIONS else []),
                          "recommendation_present": cell["condition"] == "T-recommended",
                          "recorded_before_first_model_call": True},
                endpoint=endpoint, gpu_uuid=f"GPU-{endpoint[-1]}", telemetry=None,
                site_recovered=int(recovered and submitted), function_localized=int(recovered and submitted),
                recall_V_callers=len(recovered_v) / n_v, recall_V_sites=len(recovered_v) / n_v,
                recovered_V=recovered_v, n_V=n_v,
                nm_recall=(float(callers[0] in recovered_v) if nm else None),
                nm_any=(int(callers[0] in recovered_v) if nm else None), n_NM=1 if nm else 0,
                n_submitted=n_sub, n_unique_sites=n_sub, n_invalid=0, precision_categories=cats,
                unresolved_reasons={r: (1 if (r == "not_a_known_site" and submitted) else 0)
                                    for r in S.UNRESOLVED_REASONS},
                precision_lower=(len(recovered_v) / n_sub) if n_sub else None,
                precision_upper=(len(recovered_v) / n_sub) if n_sub else None,
                complete=rng.choice(S.CLAIMS) if submitted else None)
    claim = base["complete"]
    base["certifiably_false_completeness"] = None if claim is None else (claim == "yes" and len(recovered_v) < n_v)
    base["uncertain_despite_full_V"] = None if claim is None else (claim != "yes" and len(recovered_v) == n_v)
    return base


def synthetic_study(*, scenario="known_effect", families=24, items_per_family=5, dev_families=1, rng_seed=7,
                    design=S.DEFAULT_DESIGN, rates=None, family_sd=0.08, item_sd=0.10, effect_sd=0.0,
                    partial_seed=None, partial_fraction=0.5, missing_fraction=0.0, non_submit_fraction=0.05,
                    zero_effect_conditions=()) -> dict:
    """SYNTHETIC rows in the v2 formats with known condition rates (never research data).

    Each item's recovery probability in condition c is rates[c] + family effect + item
    effect (+ a family-level effect deviation for C3 and C4a with SD ``effect_sd``),
    clipped to [0.02, 0.98]. ``partial_seed`` leaves ``partial_fraction`` of that seed's P
    cells not dispatched; ``missing_fraction`` marks P cells missing (infrastructure).
    Returns items, score rows, policy rows and the true family-weighted contrasts.
    """
    rng = random.Random(rng_seed)
    rates = dict(_SYNTHETIC_RATES[scenario] if rates is None else rates)
    for condition in zero_effect_conditions:
        rates[condition] = rates["C1"]
    records, truth_p = [], {}
    for f in range(families + dev_families):
        role = "main" if f < families else "dev"
        family = f"synfam{f:02d}" if role == "main" else f"syndev{f:02d}"
        fam_effect = rng.gauss(0, family_sd)
        deviation = {c: rng.gauss(0, effect_sd) for c in ("C3", "C4a")}
        for k in range(items_per_family):
            record = synthetic_item(rng, family, k, role)
            records.append(record)
            item_effect = rng.gauss(0, item_sd)
            truth_p[record["item_id"]] = {c: _clip(r + fam_effect + item_effect + deviation.get(c, 0.0))
                                          for c, r in rates.items()}
    raw = b"".join(json.dumps(r, sort_keys=True).encode() + b"\n" for r in records)
    items_sha = sha256_bytes(raw)
    items = {r["item_id"]: S.ItemTruth(r) for r in records}
    by_id = {r["item_id"]: r for r in records}
    layout = S.Design(design)
    rows = {}
    endpoints = ("http://127.0.0.1:8141", "http://127.0.0.1:8142")
    for cell in layout.cells(items):
        item = by_id[cell["item_id"]]
        condition = cell["condition"]
        p = truth_p[cell["item_id"]].get(condition, truth_p[cell["item_id"]]["C1"])
        state = "completed"
        if cell["block"] == PRIMARY_BLOCK or (cell["block"] == "S-C0"):
            if partial_seed is not None and cell["seed"] == partial_seed and rng.random() < partial_fraction:
                state = "not_dispatched"
            elif missing_fraction and rng.random() < missing_fraction:
                state = "missing_infrastructure"
        exit_status = "Submitted" if rng.random() >= non_submit_fraction else rng.choice(
            ("NoAnswer", "ContextLimit", "FormatExhausted"))
        recovered = rng.random() < p
        rows[cell["cell_id"]] = _synthetic_row(cell, item, recovered, state=state, exit_status=exit_status,
                                               items_sha256=items_sha, design=design, rng=rng,
                                               endpoint=endpoints[rng.randrange(2)])
    policies = {}
    for item_id, item in items.items():
        record = by_id[item_id]
        for variant in (S.NO_VARIANT,) + S.VARIANTS:
            for policy in S.POLICIES:
                if policy == "answer_only":
                    hit = variant in ("vcomplete", "natural")
                elif policy == "grep_all":
                    hit = True
                else:
                    hit = bool(record["covariates"]["c4_contains_omitted_site"])
                n_sub = 3
                callers = [v["identity"] for v in record["V"]]
                recovered_v = sorted(callers[:2] if hit else callers[1:2])
                policies[(item_id, variant, policy)] = {
                    "schema": S.POLICY_SCHEMA, "kind": "policy", "policy": policy, "item_id": item_id,
                    "family": item.family, "role": item.role, "variant": variant, "omitted": item.omitted,
                    "wording_pair": item.wording_pair, "exit_status": "Submitted", "completed": True,
                    "items_sha256": items_sha, "site_recovered": int(hit), "function_localized": int(hit),
                    "recall_V_callers": len(recovered_v) / 3, "recall_V_sites": len(recovered_v) / 3,
                    "recovered_V": recovered_v, "n_V": 3, "nm_recall": None, "nm_any": None, "n_NM": 0,
                    "n_submitted": n_sub, "n_unique_sites": n_sub, "n_invalid": 0,
                    "precision_categories": {"v_site": len(recovered_v), "u_site": 0, "lexical_name_match": 0,
                                             "unresolved": n_sub - len(recovered_v), "duplicate": 0},
                    "unresolved_reasons": {r: 0 for r in S.UNRESOLVED_REASONS},
                    "precision_lower": len(recovered_v) / n_sub, "precision_upper": len(recovered_v) / n_sub,
                    "complete": None, "certifiably_false_completeness": None, "uncertain_despite_full_V": None}
    main = [i for i, item in items.items() if item.role == "main"]
    fam_items = defaultdict(list)
    for i in main:
        fam_items[items[i].family].append(i)
    truth = {}
    for hypothesis, t, c in HYPOTHESES:
        per_family = [statistics.fmean(truth_p[i][t] - truth_p[i][c] for i in ids) for ids in fam_items.values()]
        truth[hypothesis] = statistics.fmean(per_family)
    return {"label": SYNTHETIC_LABEL, "items": items, "item_records": records, "items_sha256": items_sha,
            "items_bytes": raw, "score_rows": rows, "policy_rows": policies, "true_contrasts": truth,
            "true_rates": rates, "design": design}


def synthetic_plan(*, label=SYNTHETIC_LABEL, design=S.DEFAULT_DESIGN, resamples=2000, draws=20000,
                   min_families=20) -> Plan:
    return plan_from_dict(default_plan_dict(label=label, design=design, bootstrap={"resamples": resamples,
                                                                                   "seed": 20261101},
                                            sign_flip={"exact_max_families": 20, "draws": draws, "seed": 20261101},
                                            min_families_confirmatory=min_families))


def write_synthetic(out_dir, **kwargs) -> dict:
    """Write a SYNTHETIC fixture (items.jsonl, scores.jsonl, policies.jsonl, plan.json, TRUTH.json)."""
    study = synthetic_study(**kwargs)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {"items.jsonl": study["items_bytes"],
             "scores.jsonl": b"".join(canonical_json(r).encode() + b"\n" for r in study["score_rows"].values()),
             "policies.jsonl": b"".join(canonical_json(r).encode() + b"\n" for r in study["policy_rows"].values()),
             "plan.json": dump_json(default_plan_dict(label=SYNTHETIC_LABEL, design=study["design"])),
             "TRUTH.json": dump_json({"label": SYNTHETIC_LABEL, "true_contrasts": study["true_contrasts"],
                                      "true_rates": study["true_rates"], "kwargs": {k: v for k, v in kwargs.items()
                                                                                    if k != "rates"}})}
    written = {}
    for name, data in files.items():
        _write(out / name, data, overwrite=False)
        written[name] = sha256_bytes(data)
    return written
