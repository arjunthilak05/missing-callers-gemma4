#!/usr/bin/env python3
"""Exp A main-20261101b: independent re-analysis of the primary confirmatory results.

ANALYSIS RECORD.

What this is
    A POST-RESULTS verifiability check, written and run on 2026-11-01 after the primary
    results were opened (about 06:05 UTC). It recomputes the preregistered primary
    quantities (H1-H3 and the block-P site_recovered rates) from CELLS.jsonl alone, with
    independently written code, and compares every recomputed value with the frozen
    ANALYSIS.json. The check itself is not preregistered and adds no new inference.

Independence
    Written without opening scripts/expA_analyze.py, scripts/expA_score.py or any
    absent_edges/ analysis code. Data input: CELLS.jsonl. ANALYSIS.json is read only to
    compare. Method specification (frozen TABLES.md, Methods section):
      unit: item mean over the primary seeds (completed cells) -> paired difference within
            item -> family mean -> equal-weight mean over families
      primary_test: two-sided one-sample t test on family mean differences (df = F - 1);
            Holm over the fixed family H1, H2, H3; 95% t intervals are marginal
      denominators: completed model exits other than Submitted score 0
      primary seeds 11, 23, 47; block P only; H1 = C3 - C2, H2 = C4a - C1, H3 = C1 - C0
            (variant omit_node, except C0 with variant none); pairing by item_id

Numerics
    Standard library only. All means are exact rationals (fractions.Fraction); square roots
    use decimal at 60 digits. The Student t CDF and quantile are implemented below with the
    regularized incomplete beta function (modified Lentz continued fraction).

Usage (from the repository root)
    python -I -B scripts/expA_reanalyze_independent.py --cells PATH/CELLS.jsonl \
        --analysis reports/expA-main-20261101b/analysis/ANALYSIS.json \
        --out-dir reports/expA-main-20261101b/independent \
        [--tcdf-crosscheck reports/expA-main-20261101b/independent/TCDF_SCIPY_CROSSCHECK.json]

Exit codes: 0 verdict PASS; 1 verdict FAIL (outputs written); 2 refused (input sha256
mismatch or unusable input; nothing written).
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import shlex
import sys
from collections import Counter, defaultdict
from decimal import Decimal, localcontext
from fractions import Fraction

LABEL = "ANALYSIS RECORD"
TIMING_NOTE = (
    "POST-RESULTS verifiability check, written and run on 2026-11-01 after the primary results "
    "were opened (about 06:05 UTC). It recomputes the preregistered primary quantities with "
    "independently written code. The check itself is not preregistered and adds no new inference."
)
INDEPENDENCE_NOTE = (
    "Written without opening scripts/expA_analyze.py, scripts/expA_score.py or any absent_edges/ "
    "analysis code. Method taken from the frozen TABLES.md Methods section (see method_spec). "
    "Data input: CELLS.jsonl only. ANALYSIS.json is read only to compare."
)
METHOD_SPEC = {
    "unit": (
        "item mean over the primary seeds (completed cells) -> paired difference within item -> "
        "family mean -> equal-weight mean over families"
    ),
    "primary_test": (
        "two-sided one-sample t test on family mean differences (df = F - 1); Holm over the fixed "
        "family H1, H2, H3; 95% t intervals are marginal"
    ),
    "denominators": (
        "completed model exits other than Submitted score 0 (completed model exits per prereg: "
        "Submitted, NoAnswer, ContextLimit, FormatExhausted)"
    ),
    "seeds_and_contrasts": (
        "primary seeds 11, 23, 47; block P only; H1 = C3 - C2, H2 = C4a - C1, H3 = C1 - C0 "
        "(variant omit_node, except C0 with variant none); pairing by item_id"
    ),
    "p_rates": (
        "family-weighted rate of site_recovered per block-P condition: item mean over the primary "
        "seeds -> family mean -> equal-weight mean over families; 95% Student t interval over the "
        "family means (df = F - 1)"
    ),
}
NOT_REVERIFIED = [
    "per-cell scoring: site_recovered is taken from CELLS.jsonl as given (episodes are not re-scored "
    "and scores.jsonl is not read)",
    "sign-flip p-values and family-cluster bootstrap intervals (RNG-dependent sensitivity analyses)",
    "secondary and exploratory blocks (S-VC, S-NAT, S-EDGE, T), costs, exposure audit, process "
    "descriptives",
]

EXPECTED_SHA256 = {
    "cells": "01a84b6973c2877fd30301a53163c7c619b416168d868bf645f8c43ffb2ff6ab",
    "analysis": "f9017bf657c2472c30867ef8ace33433e4110c5597c340d1e163c279b9839d86",
}
TOLERANCE_ABS = 1e-9
BLOCK = "P"
PRIMARY_SEEDS = (11, 23, 47)
ARM_VARIANT = {"C0": "none", "C1": "omit_node", "C2": "omit_node", "C3": "omit_node", "C4a": "omit_node"}
CONDITIONS = ("C0", "C1", "C2", "C3", "C4a")
HYPOTHESES = (  # (id, treatment, control)
    ("H1", "C3", "C2"),
    ("H2", "C4a", "C1"),
    ("H3", "C1", "C0"),
)
CONFIDENCE = 0.95
ALPHA = 0.05
MEANINGFUL_EFFECT = Fraction(1, 10)
MIN_FAMILIES_CONFIRMATORY = 20
COMPLETED_MODEL_EXITS = ("Submitted", "NoAnswer", "ContextLimit", "FormatExhausted")
EXIT_COUNT_KEYS = ("ContextLimit", "FormatExhausted", "InfraInterrupted", "Missing", "NoAnswer", "Submitted")
DECIMAL_DIGITS = 60


class Refusal(Exception):
    """Input problem: refuse to run and write nothing."""


# --------------------------------------------------------------------------------------------
# Student t distribution (regularized incomplete beta function)
# --------------------------------------------------------------------------------------------

_CF_EPS = 4.0 * sys.float_info.epsilon
_CF_TINY = 1e-300
_CF_MAX_ITER = 100000


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Continued fraction of I_x(a, b) (modified Lentz). Converges fast for x < (a+1)/(a+b+2)."""
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _CF_TINY:
        d = _CF_TINY
    d = 1.0 / d
    h = d
    for m in range(1, _CF_MAX_ITER + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _CF_TINY:
            d = _CF_TINY
        c = 1.0 + aa / c
        if abs(c) < _CF_TINY:
            c = _CF_TINY
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _CF_TINY:
            d = _CF_TINY
        c = 1.0 + aa / c
        if abs(c) < _CF_TINY:
            c = _CF_TINY
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) <= _CF_EPS:
            return h
    raise ArithmeticError(f"incomplete beta continued fraction did not converge (a={a}, b={b}, x={x})")


def betainc_regularized(a: float, b: float, x: float, y: float | None = None) -> float:
    """I_x(a, b). Pass y = 1 - x when it is known more accurately than 1.0 - x."""
    if not (a > 0.0 and b > 0.0):
        raise ValueError("a and b must be positive")
    if y is None:
        y = 1.0 - x
    if x <= 0.0:
        return 0.0
    if y <= 0.0:
        return 1.0
    log_front = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(y)
    front = math.exp(log_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, y) / b


def t_two_sided_p(t: float, df: float) -> float:
    """P(|T| >= |t|) for Student t with df degrees of freedom: I_{df/(df+t^2)}(df/2, 1/2)."""
    if math.isnan(t) or not df > 0:
        raise ValueError("bad t or df")
    if math.isinf(t):
        return 0.0
    t2 = t * t
    if t2 == 0.0:
        return 1.0
    denom = df + t2
    if math.isinf(denom):
        return 0.0
    return betainc_regularized(df / 2.0, 0.5, df / denom, t2 / denom)


def t_cdf(t: float, df: float) -> float:
    """P(T <= t)."""
    if t == 0.0:
        return 0.5
    tail = 0.5 * t_two_sided_p(t, df)
    return 1.0 - tail if t > 0.0 else tail


def t_isf_two_sided(p: float, df: float) -> float:
    """The t >= 0 with P(|T| >= t) = p, by bisection down to adjacent doubles."""
    if not (0.0 < p <= 1.0):
        raise ValueError("p must be in (0, 1]")
    if p == 1.0:
        return 0.0
    lo, hi = 0.0, 1.0
    while t_two_sided_p(hi, df) > p:
        lo, hi = hi, hi * 2.0
        if math.isinf(hi):
            return math.inf
    while True:
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:
            break
        if t_two_sided_p(mid, df) > p:
            lo = mid
        else:
            hi = mid
    if abs(t_two_sided_p(lo, df) - p) <= abs(t_two_sided_p(hi, df) - p):
        return lo
    return hi


def t_ppf(q: float, df: float) -> float:
    """Quantile: the t with P(T <= t) = q. (1 - q is exact in binary floating point for q >= 0.5.)"""
    if not (0.0 < q < 1.0):
        raise ValueError("q must be in (0, 1)")
    if q == 0.5:
        return 0.0
    if q > 0.5:
        return t_isf_two_sided(2.0 * (1.0 - q), df)
    return -t_isf_two_sided(2.0 * q, df)


def tdist_selfcheck() -> dict:
    """Compare the t functions with closed forms (df = 1, 2, 3 CDF; df = 1, 2 quantile)."""
    sqrt3 = math.sqrt(3.0)
    cdf_closed = {
        1: lambda t: 0.5 + math.atan(t) / math.pi,
        2: lambda t: 0.5 + t / (2.0 * math.sqrt(2.0 + t * t)),
        3: lambda t: 0.5 + ((t / sqrt3) / (1.0 + t * t / 3.0) + math.atan(t / sqrt3)) / math.pi,
    }
    ppf_closed = {
        1: lambda q: math.tan(math.pi * (q - 0.5)),
        2: lambda q: (2.0 * q - 1.0) / math.sqrt(2.0 * q * (1.0 - q)),
    }
    t_grid = [i / 20.0 for i in range(-400, 401)]
    q_grid = [0.001, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.45, 0.55, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99, 0.999]
    cdf_max, cdf_at = 0.0, None
    for df, fn in cdf_closed.items():
        for t in t_grid:
            d = abs(t_cdf(t, df) - fn(t))
            if cdf_at is None or d > cdf_max:
                cdf_max, cdf_at = d, {"df": df, "t": t}
    ppf_max_rel, ppf_at = 0.0, None
    for df, fn in ppf_closed.items():
        for q in q_grid:
            ref = fn(q)
            d = abs(t_ppf(q, df) - ref) / abs(ref)
            if ppf_at is None or d > ppf_max_rel:
                ppf_max_rel, ppf_at = d, {"df": df, "q": q}
    cdf_limit, ppf_limit = 1e-12, 1e-10
    return {
        "cdf": {
            "dfs": sorted(cdf_closed),
            "t_grid": "-20 to 20 step 0.05 (801 points per df)",
            "max_abs_diff": cdf_max,
            "at": cdf_at,
            "limit": cdf_limit,
        },
        "quantile": {
            "dfs": sorted(ppf_closed),
            "q_grid": q_grid,
            "max_rel_diff": ppf_max_rel,
            "at": ppf_at,
            "limit": ppf_limit,
        },
        "pass": cdf_max <= cdf_limit and ppf_max_rel <= ppf_limit,
    }


# --------------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------------


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise Refusal(f"{path}:{lineno}: not JSON ({exc})") from exc
            if not isinstance(row, dict):
                raise Refusal(f"{path}:{lineno}: not a JSON object")
            rows.append(row)
    return rows


# --------------------------------------------------------------------------------------------
# Aggregation and inference
# --------------------------------------------------------------------------------------------


def fstr(x: Fraction) -> str:
    return f"{x.numerator}/{x.denominator}"


def mean_fraction(values) -> Fraction:
    values = list(values)
    return sum(values, Fraction(0)) / len(values)


def family_means(item_values: dict, item_family: dict) -> dict:
    """{family: (mean over that family's items, n_items)}."""
    groups = defaultdict(list)
    for item, value in item_values.items():
        groups[item_family[item]].append(value)
    return {fam: (mean_fraction(vals), len(vals)) for fam, vals in groups.items()}


def t_statistics(values: list, confidence: float) -> dict:
    """One-sample t summary of exact rational values: mean, sd, se, t, df, two-sided p, t interval."""
    n = len(values)
    mean = mean_fraction(values)
    out = {"n": n, "df": n - 1, "mean_exact": mean, "mean": float(mean)}
    if n < 2:
        out.update(status="not_estimable", t=None, p_two_sided=None)
        return out
    var = sum(((v - mean) ** 2 for v in values), Fraction(0)) / (n - 1)
    t_crit = t_isf_two_sided(1.0 - confidence, n - 1)
    with localcontext() as ctx:
        ctx.prec = DECIMAL_DIGITS
        mean_d = Decimal(mean.numerator) / Decimal(mean.denominator)
        var_d = Decimal(var.numerator) / Decimal(var.denominator)
        sd_d = var_d.sqrt()
        se_d = (var_d / n).sqrt()
        crit_d = Decimal(t_crit)
        lower_d = mean_d - crit_d * se_d
        upper_d = mean_d + crit_d * se_d
        t_d = mean_d / se_d if se_d > 0 else None
    out.update(
        var_exact=var,
        sd=float(sd_d),
        se=float(se_d),
        t_critical=t_crit,
        lower=float(lower_d),
        upper=float(upper_d),
        degenerate=var == 0,
        confidence=confidence,
    )
    if var == 0:
        # Spec: all differences 0 -> p = 1 (degenerate interval); identical nonzero -> t undefined.
        out.update(t=None, p_two_sided=1.0 if mean == 0 else None, status="zero_variance")
    else:
        t = float(t_d)
        out.update(t=t, p_two_sided=t_two_sided_p(t, n - 1), status="tested")
    return out


def holm_adjust(pvals: dict) -> tuple[dict, list]:
    """Holm step-down; a non-estimable hypothesis (None) enters with p = 1 and gets None back."""
    entered = {k: (1.0 if v is None else v) for k, v in pvals.items()}
    m = len(entered)
    order = sorted(entered, key=lambda k: (entered[k], k))
    adjusted = {}
    running = 0.0
    for rank, k in enumerate(order):
        running = max(running, min(1.0, (m - rank) * entered[k]))
        adjusted[k] = running
    return {k: (None if pvals[k] is None else adjusted[k]) for k in pvals}, order


def recompute(cells: list[dict]) -> dict:
    """All primary quantities from the cell rows (CELLS.jsonl)."""
    problems = []
    cell_ids = Counter(c.get("cell_id") for c in cells)
    duplicate_ids = sorted(k for k, v in cell_ids.items() if v > 1)
    if duplicate_ids:
        problems.append(f"{len(duplicate_ids)} duplicate cell_id values")

    block_cells = [c for c in cells if c.get("block") == BLOCK]
    excluded = Counter()
    selected = []
    for c in block_cells:
        cond, variant, seed = c.get("condition"), c.get("variant"), c.get("seed")
        if cond not in ARM_VARIANT or ARM_VARIANT[cond] != variant:
            excluded["condition_variant_not_in_primary_arms"] += 1
        elif seed not in PRIMARY_SEEDS:
            excluded["seed_not_primary"] += 1
        else:
            selected.append(c)

    item_family = {}
    scores = {cond: defaultdict(dict) for cond in CONDITIONS}  # cond -> item -> seed -> score
    exit_counts = {cond: Counter() for cond in CONDITIONS}
    cell_state_counts = Counter()
    completed_by_seed = Counter()
    not_completed = Counter()
    rule_zeroed = 0
    bad_scores = 0
    duplicate_slots = 0
    for c in selected:
        item, fam, cond, seed = c.get("item_id"), c.get("family"), c["condition"], c["seed"]
        if item_family.setdefault(item, fam) != fam:
            problems.append(f"item {item} has more than one family")
        cell_state_counts[c.get("cell_state")] += 1
        exit_status = c.get("exit_status")
        if not (c.get("cell_state") == "completed" and exit_status in COMPLETED_MODEL_EXITS):
            not_completed[(cond, str(c.get("cell_state")), str(exit_status))] += 1
            if exit_status == "InfraInterrupted":
                exit_counts[cond]["InfraInterrupted"] += 1
            continue
        exit_counts[cond][exit_status] += 1
        completed_by_seed[seed] += 1
        raw = c.get("site_recovered")
        if isinstance(raw, bool):
            raw = int(raw)
        if raw not in (0, 1):
            bad_scores += 1
            continue
        score = raw if exit_status == "Submitted" else 0  # denominators rule
        if exit_status != "Submitted" and raw != 0:
            rule_zeroed += 1
        if seed in scores[cond][item]:
            duplicate_slots += 1
            continue
        scores[cond][item][seed] = score
    if bad_scores:
        problems.append(f"{bad_scores} completed cells with site_recovered not in {{0, 1}}")
    if duplicate_slots:
        problems.append(f"{duplicate_slots} duplicate (condition, item, seed) cells")

    items = sorted(item_family)
    missing = {}
    for cond in CONDITIONS:
        n_missing = sum(1 for item in items for seed in PRIMARY_SEEDS if seed not in scores[cond].get(item, {}))
        if n_missing:
            missing[cond] = n_missing
        exit_counts[cond]["Missing"] = n_missing

    selected_ids = {c.get("cell_id") for c in selected}
    flag_ids = {c.get("cell_id") for c in cells if c.get("in_primary_analysis") is True}

    item_means = {
        cond: {item: Fraction(sum(by_seed.values()), len(by_seed)) for item, by_seed in scores[cond].items() if by_seed}
        for cond in CONDITIONS
    }

    rates = {}
    for cond in CONDITIONS:
        fm = family_means(item_means[cond], item_family)
        families = sorted(fm)
        rates[cond] = {
            "variant": ARM_VARIANT[cond],
            "seeds": list(PRIMARY_SEEDS),
            "items": len(items),
            "cells": len(items) * len(PRIMARY_SEEDS),
            "completed_cells": sum(len(v) for v in scores[cond].values()),
            "n_items": len(item_means[cond]),
            "n_families": len(families),
            "estimate": None,  # filled below
            "micro_item_mean": mean_fraction(item_means[cond].values()),
            "family_values": {f: fm[f][0] for f in families},
            "t_interval": t_statistics([fm[f][0] for f in families], CONFIDENCE),
            "exit_counts": {k: exit_counts[cond].get(k, 0) for k in EXIT_COUNT_KEYS},
        }
        rates[cond]["estimate"] = rates[cond]["t_interval"]["mean_exact"]

    hypotheses = {}
    for hid, tre, ctl in HYPOTHESES:
        tm, cm = item_means[tre], item_means[ctl]
        paired = sorted(set(tm) & set(cm))
        diffs = {i: tm[i] - cm[i] for i in paired}
        fd = family_means(diffs, item_family)
        ft = family_means({i: tm[i] for i in paired}, item_family)
        fc = family_means({i: cm[i] for i in paired}, item_family)
        families = sorted(fd)
        stats = t_statistics([fd[f][0] for f in families], CONFIDENCE)
        hypotheses[hid] = {
            "contrast": f"{tre} - {ctl}",
            "treatment": tre,
            "control": ctl,
            "status": "confirmatory" if len(families) >= MIN_FAMILIES_CONFIRMATORY else "exploratory",
            "n_items": len(paired),
            "n_families": len(families),
            "items_without_complete_pair": len(set(tm) | set(cm)) - len(paired),
            "estimate": stats["mean_exact"],
            "micro_item_estimate": mean_fraction(diffs.values()),
            "treatment_rate": mean_fraction(v for v, _ in ft.values()),
            "control_rate": mean_fraction(v for v, _ in fc.values()),
            "t_test": stats,
            "family_units": {
                f: {"difference": fd[f][0], "treatment": ft[f][0], "control": fc[f][0], "n_items": fd[f][1]}
                for f in families
            },
        }

    holm, holm_order = holm_adjust({hid: hypotheses[hid]["t_test"]["p_two_sided"] for hid, _, _ in HYPOTHESES})
    for hid, _, _ in HYPOTHESES:
        h = hypotheses[hid]
        h["p_primary"] = h["t_test"]["p_two_sided"]
        h["holm_adjusted_p"] = holm[hid]
        h["reject_at_alpha"] = holm[hid] is not None and holm[hid] <= ALPHA
        h["abs_estimate_at_least_meaningful"] = abs(h["estimate"]) >= MEANINGFUL_EFFECT

    family_items = Counter(item_family[i] for i in items)
    integrity = {
        "rows": len(cells),
        "duplicate_cell_ids": len(duplicate_ids),
        "block_P_rows": len(block_cells),
        "block_P_rows_excluded": dict(excluded),
        "primary_rows_selected": len(selected),
        "selected_matches_in_primary_analysis_flag": selected_ids == flag_ids,
        "cell_states_selected": dict(cell_state_counts),
        "completed_cells_by_seed": {s: completed_by_seed.get(s, 0) for s in PRIMARY_SEEDS},
        "not_completed_selected": {"|".join(k): v for k, v in sorted(not_completed.items())},
        "missing_cells_by_condition": missing,
        "non_submitted_cells_with_site_recovered_1": rule_zeroed,
        "items": len(items),
        "families": len(family_items),
        "items_by_family": dict(sorted(family_items.items())),
        "problems": problems,
        "ok": not problems,
    }
    return {
        "integrity": integrity,
        "rates": rates,
        "hypotheses": hypotheses,
        "holm": {"family": [h for h, _, _ in HYPOTHESES], "order_by_p": holm_order, "method": "Holm step-down"},
    }


# --------------------------------------------------------------------------------------------
# Comparison with ANALYSIS.json
# --------------------------------------------------------------------------------------------


def path_str(path: list) -> str:
    out = ""
    for step in path:
        if isinstance(step, dict):
            out += "[" + ",".join(f"{k}={v}" for k, v in step.items()) + "]"
        else:
            out += ("." if out else "") + str(step)
    return out


def resolve(obj, path: list):
    cur = obj
    for step in path:
        if isinstance(step, dict):
            if not isinstance(cur, list):
                return False, None
            hits = [e for e in cur if isinstance(e, dict) and all(e.get(k) == v for k, v in step.items())]
            if len(hits) != 1:
                return False, None
            cur = hits[0]
        else:
            if not isinstance(cur, dict) or step not in cur:
                return False, None
            cur = cur[step]
    return True, cur


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def compare(analysis: dict, group: str, name: str, path: list, recomputed, kind: str) -> dict:
    found, theirs = resolve(analysis, path)
    if kind == "exact":
        shown = fstr(recomputed)
    elif kind == "float":
        shown = None if recomputed is None else float(recomputed)
    else:
        shown = recomputed
    rec = {
        "group": group,
        "name": name,
        "analysis_json_field": path_str(path),
        "kind": kind,
        "recomputed": shown,
        "analysis_json": theirs if found else None,
        "abs_diff": None,
        "rel_diff": None,
        "pass": False,
    }
    if not found:
        rec["note"] = "field not found in ANALYSIS.json"
        return rec
    if kind == "float":
        if recomputed is None or not _is_number(theirs) or not math.isfinite(float(theirs)):
            rec["note"] = "not comparable as finite numbers"
            return rec
        d = abs(float(recomputed) - float(theirs))
        rec["abs_diff"] = d
        rec["rel_diff"] = d / abs(float(theirs)) if theirs != 0 else (0.0 if d == 0 else math.inf)
        rec["pass"] = d <= TOLERANCE_ABS
    elif kind == "exact":
        try:
            theirs_fr = Fraction(str(theirs))
        except (ValueError, ZeroDivisionError):
            rec["note"] = "not a rational string"
            return rec
        rec["abs_diff"] = abs(float(recomputed - theirs_fr))
        rec["pass"] = theirs_fr == recomputed
    elif kind == "int":
        ok = isinstance(theirs, int) and not isinstance(theirs, bool)
        rec["abs_diff"] = abs(theirs - recomputed) if ok else None
        rec["pass"] = ok and theirs == recomputed
    elif kind in ("bool", "str", "list"):
        rec["pass"] = type(theirs) is type(recomputed) and theirs == recomputed
    else:
        raise ValueError(kind)
    return rec


def build_comparisons(res: dict, analysis: dict, cells_meta: dict) -> list[dict]:
    comps = []

    def add(group, name, path, value, kind):
        comps.append(compare(analysis, group, name, path, value, kind))

    integ = res["integrity"]
    no_missing = not integ["missing_cells_by_condition"]
    for hid, tre, ctl in HYPOTHESES:
        h = res["hypotheses"][hid]
        st = h["t_test"]
        base = ["primary", "hypotheses", {"id": hid}]
        g = "primary_headline"
        add(g, f"{hid} estimate", base + ["estimate"], h["estimate"], "float")
        add(g, f"{hid} estimate (exact)", base + ["estimate_exact"], h["estimate"], "exact")
        add(g, f"{hid} 95% t CI lower", base + ["t_test", "interval", "lower"], st.get("lower"), "float")
        add(g, f"{hid} 95% t CI upper", base + ["t_test", "interval", "upper"], st.get("upper"), "float")
        add(g, f"{hid} t", base + ["t_test", "t"], st.get("t"), "float")
        add(g, f"{hid} df", base + ["t_test", "df"], st["df"], "int")
        add(g, f"{hid} p (t test)", base + ["t_test", "p_two_sided"], st.get("p_two_sided"), "float")
        add(g, f"{hid} p (primary)", base + ["p_primary"], h["p_primary"], "float")
        add(g, f"{hid} Holm p", base + ["holm_adjusted_p"], h["holm_adjusted_p"], "float")
        add(g, f"{hid} reject at alpha", base + ["reject_at_alpha"], h["reject_at_alpha"], "bool")

        g = "primary_detail"
        add(g, f"{hid} treatment arm", base + ["treatment"], tre, "str")
        add(g, f"{hid} control arm", base + ["control"], ctl, "str")
        add(g, f"{hid} micro item estimate", base + ["micro_item_estimate"], h["micro_item_estimate"], "float")
        add(g, f"{hid} treatment rate", base + ["treatment_rate"], h["treatment_rate"], "float")
        add(g, f"{hid} control rate", base + ["control_rate"], h["control_rate"], "float")
        add(g, f"{hid} n families", base + ["n_families"], h["n_families"], "int")
        add(g, f"{hid} n items", base + ["n_items"], h["n_items"], "int")
        add(g, f"{hid} items without complete pair", base + ["items_without_complete_pair"],
            h["items_without_complete_pair"], "int")
        add(g, f"{hid} t-test mean", base + ["t_test", "mean"], h["estimate"], "float")
        add(g, f"{hid} t-test sd", base + ["t_test", "sd"], st.get("sd"), "float")
        add(g, f"{hid} t-test se", base + ["t_test", "se"], st.get("se"), "float")
        add(g, f"{hid} t critical", base + ["t_test", "t_critical"], st.get("t_critical"), "float")
        add(g, f"{hid} t-test n families", base + ["t_test", "n_families"], st["n"], "int")
        add(g, f"{hid} interval degenerate", base + ["t_test", "interval", "degenerate"], st.get("degenerate"), "bool")
        add(g, f"{hid} t-test status", base + ["t_test", "status"], st.get("status"), "str")
        add(g, f"{hid} p source", base + ["p_source"], "t_test", "str")
        add(g, f"{hid} status", base + ["status"], h["status"], "str")
        add(g, f"{hid} confirmatory", base + ["confirmatory"], h["status"] == "confirmatory", "bool")
        add(g, f"{hid} |estimate| >= 0.1", base + ["abs_estimate_at_least_meaningful"],
            h["abs_estimate_at_least_meaningful"], "bool")
        add(g, f"{hid} seeds used", base + ["seeds_used"], list(PRIMARY_SEEDS), "list")
        if hid == "H2":
            add(g, "H2 estimate (copy in reference_policies)",
                ["reference_policies", "h2_vs_c4_hits_reference", "c4a_minus_c1"], h["estimate"], "float")

        g = "primary_family_units"
        for fam, u in h["family_units"].items():
            fb = base + ["family_units", {"family": fam}]
            add(g, f"{hid} {fam} difference", fb + ["difference"], u["difference"], "float")
            add(g, f"{hid} {fam} difference (exact)", fb + ["difference_exact"], u["difference"], "exact")
            add(g, f"{hid} {fam} treatment", fb + ["treatment"], u["treatment"], "float")
            add(g, f"{hid} {fam} control", fb + ["control"], u["control"], "float")
            add(g, f"{hid} {fam} n items", fb + ["n_items"], u["n_items"], "int")

        if no_missing:
            # With zero missing cells every missing-cell bound is the primary analysis itself.
            g = "primary_missing_cell_bounds"
            for bound in ("impute_0", "impute_1", "lowest", "highest"):
                bb = base + ["bounds_missing_cells", bound]
                add(g, f"{hid} bound {bound} estimate", bb + ["estimate"], h["estimate"], "float")
                add(g, f"{hid} bound {bound} p", bb + ["p_two_sided"], st.get("p_two_sided"), "float")
                add(g, f"{hid} bound {bound} CI lower", bb + ["t_interval", "lower"], st.get("lower"), "float")
                add(g, f"{hid} bound {bound} CI upper", bb + ["t_interval", "upper"], st.get("upper"), "float")
                add(g, f"{hid} bound {bound} n families", bb + ["n_families"], h["n_families"], "int")
                add(g, f"{hid} bound {bound} n items", bb + ["n_items"], h["n_items"], "int")

    g = "primary_detail"
    add(g, "Holm family", ["primary", "holm_family"], [hid for hid, _, _ in HYPOTHESES], "list")
    add(g, "primary seeds", ["primary", "primary_seeds"], list(PRIMARY_SEEDS), "list")
    add(g, "primary block", ["primary", "block"], BLOCK, "str")
    add(g, "primary outcome", ["primary", "outcome"], "site_recovered", "str")

    for cond in CONDITIONS:
        r = res["rates"][cond]
        ti = r["t_interval"]
        base = ["secondary", "rates", {"block": BLOCK, "condition": cond, "variant": ARM_VARIANT[cond]}]
        sr = base + ["site_recovered"]
        g = "p_rates_headline"
        add(g, f"P {cond} rate", sr + ["estimate"], r["estimate"], "float")
        add(g, f"P {cond} rate (exact)", sr + ["estimate_exact"], r["estimate"], "exact")
        add(g, f"P {cond} 95% t CI lower", sr + ["t_interval", "lower"], ti.get("lower"), "float")
        add(g, f"P {cond} 95% t CI upper", sr + ["t_interval", "upper"], ti.get("upper"), "float")

        g = "p_rates_detail"
        add(g, f"P {cond} micro item mean", sr + ["micro_item_mean"], r["micro_item_mean"], "float")
        add(g, f"P {cond} t-interval mean", sr + ["t_interval", "mean"], r["estimate"], "float")
        add(g, f"P {cond} t-interval sd", sr + ["t_interval", "sd"], ti.get("sd"), "float")
        add(g, f"P {cond} t-interval se", sr + ["t_interval", "se"], ti.get("se"), "float")
        add(g, f"P {cond} t critical", sr + ["t_interval", "t_critical"], ti.get("t_critical"), "float")
        add(g, f"P {cond} t-interval df", sr + ["t_interval", "df"], ti["df"], "int")
        add(g, f"P {cond} t-interval n", sr + ["t_interval", "n"], ti["n"], "int")
        add(g, f"P {cond} n families", sr + ["n_families"], r["n_families"], "int")
        add(g, f"P {cond} n items", sr + ["n_items"], r["n_items"], "int")
        add(g, f"P {cond} items", base + ["items"], r["items"], "int")
        add(g, f"P {cond} cells", base + ["cells"], r["cells"], "int")
        add(g, f"P {cond} completed cells", base + ["completed_cells"], r["completed_cells"], "int")
        add(g, f"P {cond} seeds", base + ["seeds"], r["seeds"], "list")
        for key in EXIT_COUNT_KEYS:
            add(g, f"P {cond} exit count {key}", base + ["exit_counts", key], r["exit_counts"][key], "int")

        g = "p_rates_family_values"
        for fam, v in r["family_values"].items():
            add(g, f"P {cond} {fam} family rate", sr + ["family_values", fam], v, "float")

    g = "population_and_inputs"
    add(g, "population n families", ["population", "n_families"], integ["families"], "int")
    add(g, "population n items", ["population", "n_items"], integ["items"], "int")
    for fam, n in integ["items_by_family"].items():
        add(g, f"population items {fam}", ["population", "items_by_family", fam], n, "int")
    add(g, "block P completed cells", ["completion", "by_block", BLOCK, "completed"],
        integ["cell_states_selected"].get("completed", 0), "int")
    for seed in PRIMARY_SEEDS:
        add(g, f"seed {seed} completed P cells", ["primary", "seed_completion", str(seed), "cell_states", "completed"],
            integ["completed_cells_by_seed"][seed], "int")
    add(g, "CELLS.jsonl sha256 recorded in ANALYSIS.json", ["outputs", "CELLS.jsonl", "sha256"], cells_meta["sha256"], "str")
    add(g, "CELLS.jsonl rows recorded in ANALYSIS.json", ["outputs", "CELLS.jsonl", "rows"], cells_meta["rows"], "int")
    add(g, "CELLS.jsonl bytes recorded in ANALYSIS.json", ["outputs", "CELLS.jsonl", "bytes"], cells_meta["bytes"], "int")
    return comps


def summarize(comps: list[dict]) -> dict:
    groups = {}
    for c in comps:
        g = groups.setdefault(c["group"], {"compared": 0, "pass": 0, "fail": 0, "max_abs_diff": None, "max_abs_diff_field": None})
        g["compared"] += 1
        g["pass" if c["pass"] else "fail"] += 1
        if c["kind"] == "float" and c["abs_diff"] is not None:
            if g["max_abs_diff"] is None or c["abs_diff"] > g["max_abs_diff"]:
                g["max_abs_diff"], g["max_abs_diff_field"] = c["abs_diff"], c["analysis_json_field"]
    floats = [c for c in comps if c["kind"] == "float" and c["abs_diff"] is not None]
    worst = max(floats, key=lambda c: c["abs_diff"], default=None)
    worst_rel = max((c for c in floats if c["rel_diff"] is not None), key=lambda c: c["rel_diff"], default=None)
    non_float = [c for c in comps if c["kind"] != "float"]
    return {
        "tolerance_abs": TOLERANCE_ABS,
        "compared": len(comps),
        "pass": sum(c["pass"] for c in comps),
        "fail": sum(not c["pass"] for c in comps),
        "float_fields": len(floats),
        "max_abs_diff": worst["abs_diff"] if worst else None,
        "max_abs_diff_field": worst["analysis_json_field"] if worst else None,
        "max_rel_diff": worst_rel["rel_diff"] if worst_rel else None,
        "max_rel_diff_field": worst_rel["analysis_json_field"] if worst_rel else None,
        "exact_int_bool_str_list_fields": len(non_float),
        "exact_int_bool_str_list_all_equal": all(c["pass"] for c in non_float),
        "by_group": groups,
        "failures": [
            {k: c.get(k) for k in ("name", "analysis_json_field", "recomputed", "analysis_json", "abs_diff", "note")}
            for c in comps
            if not c["pass"]
        ],
    }


# --------------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------------


def jsonable(x):
    if isinstance(x, Fraction):
        return {"exact": fstr(x), "float": float(x)}
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return str(x)
    return x


def command_record() -> dict:
    argv = list(getattr(sys, "orig_argv", [sys.executable] + sys.argv))
    exe = os.path.basename(argv[0]) if argv else "python"
    if exe.lower().endswith(".exe"):
        exe = exe[:-4]
    return {
        "cwd": os.getcwd().replace("\\", "/"),
        "command_line": shlex.join([exe] + argv[1:]),
        "orig_argv": argv,
        "interpreter": sys.executable,
        "python_version": sys.version.split()[0],
    }


def load_crosscheck(path: str, script_sha: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    checked = (data.get("checked_script") or {}).get("sha256")
    return {
        "path": path.replace("\\", "/"),
        "sha256": sha256_file(path),
        "checked_script_sha256": checked,
        "matches_this_script": checked == script_sha,
        "status": "current" if checked == script_sha else "STALE (made with another version of this script)",
        "summary": data.get("summary"),
        "scipy_version": data.get("scipy_version"),
        "command_line": (data.get("command") or {}).get("command_line"),
    }


def g6(x) -> str:
    return "n/a" if x is None else f"{x:+.6f}"


def gp(x) -> str:
    if x is None:
        return "n/a"
    return f"{x:.3e}" if x < 1e-3 else f"{x:.4f}"


def ge(x) -> str:
    return "n/a" if x is None else f"{x:.1e}"


def gt(x) -> str:
    return "n/a" if x is None else f"{x:.4f}"


def render_md(report: dict) -> str:
    rec = report["recomputed"]  # already JSON-ready
    summ = report["comparison_summary"]
    cmd = report["command"]
    comps = report["comparisons"]

    def subset_stats(prefix: str):
        sel = [c for c in comps if c["name"].startswith(prefix)]
        ds = [c["abs_diff"] for c in sel if c["kind"] == "float" and c["abs_diff"] is not None]
        return len(sel), all(c["pass"] for c in sel), (max(ds) if ds else None)

    L = []
    L.append("# Exp A main-20261101b: independent re-analysis of the primary results")
    L.append("")
    L.append(f"**{LABEL}.** Written by `scripts/expA_reanalyze_independent.py` at {report['generated_utc']}.")
    L.append("")
    L.append(f"**Timing.** {TIMING_NOTE}")
    L.append("")
    L.append(f"**Independence.** {INDEPENDENCE_NOTE}")
    L.append("")
    L.append(
        f"**Verdict: {report['verdict']}.** {summ['pass']} of {summ['compared']} compared values agree with "
        f"ANALYSIS.json (floats within abs {TOLERANCE_ABS:g}; exact fractions, integers, flags and labels identical). "
        f"Max abs diff {ge(summ['max_abs_diff'])} (`{summ['max_abs_diff_field']}`); max rel diff "
        f"{ge(summ['max_rel_diff'])} (`{summ['max_rel_diff_field']}`)."
    )
    if summ["failures"]:
        L.append("")
        L.append("Mismatches:")
        for f in summ["failures"]:
            note = f" ({f['note']})" if f.get("note") else ""
            L.append(f"- `{f['analysis_json_field']}`: recomputed {f['recomputed']!r}, ANALYSIS.json {f['analysis_json']!r}{note}")
    L.append("")
    L.append("## Command")
    L.append("")
    L.append("```")
    L.append(f"cd {cmd['cwd']}")
    L.append(cmd["command_line"])
    L.append("```")
    L.append("")
    L.append(f"`python` resolved to `{cmd['interpreter']}` (Python {cmd['python_version']}).")
    L.append("")
    L.append("## Inputs")
    L.append("")
    L.append("| Input | sha256 | Check |")
    L.append("|---|---|---|")
    for key in ("cells", "analysis"):
        inp = report["inputs"][key]
        L.append(f"| `{inp['path']}` | `{inp['sha256']}` | {'matches expected' if inp['verified'] else 'MISMATCH'} |")
    xc = report["inputs"].get("tcdf_crosscheck")
    if xc:
        L.append(f"| `{xc['path']}` | `{xc['sha256']}` | {xc['status']} |")
    L.append("")
    integ = rec["integrity"]
    L.append(
        f"Data: {integ['rows']} cell rows; {integ['primary_rows_selected']} block-P primary rows "
        f"({integ['items']} items, {integ['families']} families, seeds {', '.join(map(str, PRIMARY_SEEDS))}); "
        f"missing primary cells: {sum(integ['missing_cells_by_condition'].values())}; "
        f"non-Submitted cells carrying site_recovered = 1: {integ['non_submitted_cells_with_site_recovered_1']}; "
        f"selection equals the in_primary_analysis flag: {integ['selected_matches_in_primary_analysis_flag']}."
    )
    L.append("")
    L.append(f"## Primary hypotheses (block P, site_recovered; two-sided t test over {integ['families']} family means; "
             f"Holm over H1-H3)")
    L.append("")
    L.append("| H | Contrast | Estimate (exact) | 95% t CI | t (df) | p | Holm p | Reject | Fields compared | Max abs diff |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for hid, _, _ in HYPOTHESES:
        h = rec["hypotheses"][hid]
        st = h["t_test"]
        n, ok, mx = subset_stats(hid + " ")
        L.append(
            f"| {hid} | {h['contrast']} | {g6(h['estimate']['float'])} ({h['estimate']['exact']}) | "
            f"[{g6(st.get('lower'))}, {g6(st.get('upper'))}] | {gt(st.get('t'))} ({st['df']}) | "
            f"{gp(st.get('p_two_sided'))} | {gp(h['holm_adjusted_p'])} | {h['reject_at_alpha']} | "
            f"{n} ({'all pass' if ok else 'FAIL'}) | {ge(mx)} |"
        )
    L.append("")
    L.append(f"## Family-weighted block-P rates of site_recovered (95% t interval over {integ['families']} family means)")
    L.append("")
    L.append("| Cond | Variant | Rate (exact) | 95% t CI | Items | Families | Submitted / NoAnswer | Fields compared | Max abs diff |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for cond in CONDITIONS:
        r = rec["rates"][cond]
        ti = r["t_interval"]
        n, ok, mx = subset_stats(f"P {cond} ")
        L.append(
            f"| {cond} | {r['variant']} | {g6(r['estimate']['float'])} ({r['estimate']['exact']}) | "
            f"[{g6(ti.get('lower'))}, {g6(ti.get('upper'))}] | {r['n_items']} | {r['n_families']} | "
            f"{r['exit_counts']['Submitted']} / {r['exit_counts']['NoAnswer']} | "
            f"{n} ({'all pass' if ok else 'FAIL'}) | {ge(mx)} |"
        )
    L.append("")
    L.append("## Comparison summary by group")
    L.append("")
    L.append("| Group | Compared | Pass | Fail | Max abs diff (floats) |")
    L.append("|---|---|---|---|---|")
    for gname, g in summ["by_group"].items():
        L.append(f"| {gname} | {g['compared']} | {g['pass']} | {g['fail']} | {ge(g['max_abs_diff'])} |")
    L.append(f"| **all** | {summ['compared']} | {summ['pass']} | {summ['fail']} | {ge(summ['max_abs_diff'])} |")
    L.append("")
    L.append("## Student t implementation")
    L.append("")
    sc = report["tdist_selfcheck"]
    L.append(
        f"- In-script closed-form check (no scipy): CDF for df {sc['cdf']['dfs']}, max abs diff "
        f"{ge(sc['cdf']['max_abs_diff'])}; quantile for df {sc['quantile']['dfs']}, max rel diff "
        f"{ge(sc['quantile']['max_rel_diff'])}; {'pass' if sc['pass'] else 'FAIL'}."
    )
    if xc:
        s = xc.get("summary") or {}
        L.append(
            f"- Separate scipy.stats.t cross-check (scipy {xc.get('scipy_version')}; {xc['status']}): "
            f"CDF max abs diff {ge(s.get('cdf_max_abs_diff'))} over {s.get('cdf_points')} points "
            f"(df {s.get('cdf_dfs')}), {ge(s.get('cdf_max_abs_diff_df29'))} at df 29; "
            f"two-sided p max rel diff {ge(s.get('two_sided_p_max_rel_diff'))} (where scipy p > 1e-300); "
            f"quantile max abs diff {ge(s.get('ppf_max_abs_diff'))}; "
            f"t critical (df 29, 0.975) abs diff {ge(s.get('t_critical_df29_abs_diff'))}; "
            f"H1-H3 p max rel diff {ge(s.get('hypothesis_p_max_rel_diff'))}."
        )
        ref = s.get("reference_60_digits") or {}
        tc, hp = ref.get("t_critical_df29") or {}, ref.get("hypothesis_p_df29_max_rel_err") or {}
        if tc or hp:
            L.append(
                f"- 60-digit decimal reference at the analysis points (df 29), absolute error of t critical: "
                f"this script {ge(tc.get('this_script_abs_err'))}, scipy {ge(tc.get('scipy_ppf_abs_err'))}, "
                f"ANALYSIS.json {ge(tc.get('analysis_json_abs_err'))}; max relative error of the H1-H3 p-values: "
                f"this script {ge(hp.get('this_script'))}, scipy {ge(hp.get('scipy'))}, "
                f"ANALYSIS.json {ge(hp.get('analysis_json'))}."
            )
        if xc.get("command_line"):
            L.append(f"- Cross-check command (repository root): `{xc['command_line']}`")
    else:
        L.append("- scipy.stats.t cross-check: not supplied to this run.")
    L.append("")
    L.append("## Scope")
    L.append("")
    L.append("Not re-verified here:")
    for item in NOT_REVERIFIED:
        L.append(f"- {item}")
    L.append("")
    return "\n".join(L)


def write_text(path: str, text: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(tmp, path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Independent re-analysis of the Exp A primary results (stdlib only).")
    ap.add_argument("--cells", required=True, help="CELLS.jsonl of main-20261101b")
    ap.add_argument("--analysis", required=True, help="frozen ANALYSIS.json of main-20261101b")
    ap.add_argument("--out-dir", required=True, help="directory for REANALYSIS.json and REANALYSIS.md")
    ap.add_argument("--tcdf-crosscheck", default=None, help="optional JSON from the separate scipy.stats.t cross-check")
    args = ap.parse_args(argv)

    try:
        # Verify CELLS.jsonl before anything else; refuse on mismatch.
        cells_sha = sha256_file(args.cells)
        if cells_sha != EXPECTED_SHA256["cells"]:
            raise Refusal(f"CELLS.jsonl sha256 {cells_sha} != expected {EXPECTED_SHA256['cells']}")
        analysis_sha = sha256_file(args.analysis)
        if analysis_sha != EXPECTED_SHA256["analysis"]:
            raise Refusal(f"ANALYSIS.json sha256 {analysis_sha} != expected {EXPECTED_SHA256['analysis']}")
        cells = load_jsonl(args.cells)
        with open(args.analysis, "r", encoding="utf-8") as f:
            analysis = json.load(f)
        script_sha = sha256_file(os.path.abspath(__file__))
        crosscheck = load_crosscheck(args.tcdf_crosscheck, script_sha) if args.tcdf_crosscheck else None
    except (Refusal, OSError, json.JSONDecodeError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2

    selfcheck = tdist_selfcheck()
    res = recompute(cells)
    cells_meta = {"sha256": cells_sha, "rows": len(cells), "bytes": os.path.getsize(args.cells)}
    comps = build_comparisons(res, analysis, cells_meta)
    summ = summarize(comps)

    basis = {
        "input_sha256_verified": True,
        "data_integrity_ok": res["integrity"]["ok"],
        "all_comparisons_within_tolerance": summ["fail"] == 0,
        "tdist_closed_form_selfcheck_pass": selfcheck["pass"],
    }
    verdict = "PASS" if all(basis.values()) else "FAIL"
    report = {
        "label": LABEL,
        "kind": "independent re-analysis of the Exp A primary results (verifiability check)",
        "timing": TIMING_NOTE,
        "independence": INDEPENDENCE_NOTE,
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "command": command_record(),
        "script": {"path": "scripts/expA_reanalyze_independent.py", "sha256": script_sha},
        "files_read": ["--cells", "--analysis", "--tcdf-crosscheck (optional)", "this script (own sha256)"],
        "method_spec": METHOD_SPEC,
        "not_reverified": NOT_REVERIFIED,
        "inputs": {
            "cells": {
                "path": args.cells.replace("\\", "/"),
                "sha256": cells_sha,
                "expected_sha256": EXPECTED_SHA256["cells"],
                "verified": True,
                "rows": len(cells),
                "bytes": cells_meta["bytes"],
            },
            "analysis": {
                "path": args.analysis.replace("\\", "/"),
                "sha256": analysis_sha,
                "expected_sha256": EXPECTED_SHA256["analysis"],
                "verified": True,
            },
            "tcdf_crosscheck": crosscheck,
        },
        "verdict": verdict,
        "verdict_basis": basis,
        "comparison_summary": summ,
        "tdist_selfcheck": selfcheck,
        "recomputed": jsonable({
            "integrity": res["integrity"],
            "hypotheses": res["hypotheses"],
            "holm": res["holm"],
            "rates": res["rates"],
        }),
        "comparisons": comps,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, "REANALYSIS.json")
    md_path = os.path.join(args.out_dir, "REANALYSIS.md")
    write_text(json_path, json.dumps(jsonable(report), indent=1, ensure_ascii=True) + "\n")
    write_text(md_path, render_md(report))
    print(f"verdict {verdict}: {summ['pass']}/{summ['compared']} values agree "
          f"(max abs diff {summ['max_abs_diff']!r} at {summ['max_abs_diff_field']})")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
