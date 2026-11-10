"""ANALYSIS RECORD. draft script (2026-11-01).

EXPLORATORY: designed after the 31B primary results were seen (2026-11-01 ~06:05 UTC); not preregistered;
descriptive; nominal 95% family-clustered t intervals; no significance claims.

Exp A main run ``main-20261101b``: a post-results exploratory analysis package.

    python -I -B scripts/expA_explore.py \\
        --cells    <backup>/CELLS.jsonl    --scores   <backup>/scores.jsonl \\
        --items    <backup>/items.jsonl    --policies <backup>/policies.jsonl \\
        --analysis reports/expA-main-20261101b/analysis/ANALYSIS.json \\
        --out      reports/expA-main-20261101b/explore [--overwrite]

Inputs: every input's sha256 is checked against the pinned value in ``EXPECTED_SHA256`` before it is
parsed, and the bytes that were hashed are the bytes parsed. ANALYSIS.json (frozen) is read only for
the reproduction checks: this script recomputes the frozen rates, H1-H3 estimates and intervals,
per-family differences, strata, T funnel counts and S-VC differences and reports any difference.

Unit (the primary analysis's): item mean over the available seeds -> family mean -> equal-weight
mean over families; a paired contrast takes the item difference first. Intervals are Student t over
the family values (df = F - 1), computed with the frozen ``absent_edges.probe.analysis.t_interval``.
Micro counts (k/n over episodes) are given beside each family-weighted rate. No p-value is computed.
Conditioning on submission is post-treatment conditioning; those rows are descriptive only.

Sections: 1 exit partition, 2 submission / recovery-given-submission decomposition, 3 seed
concordance, 4 per-family H1-H3 table and heterogeneity, 5 strata splits, 6 tool use in P, 7 T
funnel, 8 S-VC over-caution cost, 9 precision and submitted-site categories.

Outputs: EXPLORE.json and EXPLORE.md, byte-deterministic for the pinned inputs (no timestamps, no
paths, sorted keys, no set-order dependence). Stdlib only, plus this repository's stdlib-only
``absent_edges`` package.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from absent_edges.probe import analysis as A  # noqa: E402  (frozen stdlib Student t; same conventions)

SCHEMA = "expA-explore-v1"
MODEL_LABEL = "ANALYSIS RECORD"
EXPLORATORY = ("EXPLORATORY: designed after the 31B primary results were seen (2026-11-01 ~06:05 UTC); "
               "not preregistered; descriptive; nominal 95% family-clustered t intervals; no significance claims.")
RUN = "main-20261101b"

EXPECTED_SHA256 = {
    "cells": "01a84b6973c2877fd30301a53163c7c619b416168d868bf645f8c43ffb2ff6ab",
    "scores": "a8bc820f3cadda9750adf77c763ac44b92c669109bcf570dc2298f6f181de1fa",
    "items": "f3451a3c9fba2ffd3455df17718d3e276132cf9eb0a7158240256deed32764c8",
    "policies": "b7c5ceaa419b104d4a0998c6623e2e0dd22ae24df638b1a18f9c8ac87c91c55a",
    "analysis": "f9017bf657c2472c30867ef8ace33433e4110c5597c340d1e163c279b9839d86",
}

CONFIDENCE = 0.95
BLOCKS = (("P", ("C0", "C1", "C2", "C3", "C4a")), ("S-VC", ("C1", "C3")), ("S-NAT", ("C1", "C3", "C4a")),
          ("S-EDGE", ("C1", "C3")), ("T", ("T-offered", "T-recommended")))
P_CONDITIONS = BLOCKS[0][1]
P_SEEDS = (11, 23, 47)
SECONDARY_SEED = 11
BLOCK_MEMBERSHIP = {"P": None, "S-VC": None, "S-NAT": "in_s_nat", "S-EDGE": "in_s_edge", "T": "in_t"}
HYPOTHESES = (("H1", "C3", "C2"), ("H2", "C4a", "C1"), ("H3", "C1", "C0"))
SECONDARY_PAIRS = (("S-VC", "C3", "C1"), ("S-EDGE", "C3", "C1"), ("S-NAT", "C3", "C1"), ("S-NAT", "C4a", "C1"),
                   ("T", "T-recommended", "T-offered"))
RESPONSE_BUDGET = 8
EXITS_KNOWN = ("Submitted", "NoAnswer", "ContextLimit", "FormatExhausted", "InfraInterrupted")
CATEGORIES = ("v_site", "u_site", "lexical_name_match", "unresolved", "duplicate")
NOT_IN_V = ("u_site", "lexical_name_match", "unresolved")
CLAIMS = ("yes", "no", "unknown")
FUNNEL = ("offered", "requested", "executed", "visible_nonempty_answer")
TRAJECTORY_FIELDS = ("exit_status", "calls", "prompt_tokens", "completion_tokens", "site_recovered")
PRECISION_BINS = ((0.0, 0.25, "[0, 0.25)"), (0.25, 0.5, "[0.25, 0.5)"), (0.5, 0.75, "[0.5, 0.75)"),
                  (0.75, 1.0, "[0.75, 1)"))
TOLERANCE = 1e-9
PILOT_T_NOTE = ("Pilot context: T-offered requested graph_query in 1/10 pilot episodes and T-recommended in 10/10 "
                "(erratum E-1 in protocols/EXPA_DEVIATIONS.md corrects the frozen prereg text '0/10 and 9/10').")


class InputError(SystemExit):
    pass


# --------------------------------------------------------------------------- reading


def read_verified(path, key):
    """Read the file once, check its sha256 against the pinned value, return (bytes, sha256)."""
    data = Path(path).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != EXPECTED_SHA256[key]:
        raise InputError(f"{key}: sha256 {digest} of {path} is not the pinned {EXPECTED_SHA256[key]}; not read")
    return data, digest


def _reject_constant(name):
    raise ValueError(f"non-finite JSON constant {name}")


def parse_jsonl(data, key):
    rows = []
    for number, line in enumerate(data.decode("utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line, parse_constant=_reject_constant))
            except ValueError as error:
                raise InputError(f"{key}: line {number}: {error}") from error
    return rows


# --------------------------------------------------------------------------- arithmetic (the primary unit)


def exact(value):
    """Exact rational of a bool / int / finite float (a float maps to its exact binary value)."""
    if isinstance(value, bool):
        return Fraction(int(value))
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float) and math.isfinite(value):
        return Fraction(value)
    raise ValueError(f"not a finite number: {value!r}")


def ratio_text(value):
    return f"{value.numerator}/{value.denominator}"


def item_means(rows_by_item, fn, items=None):
    """Item mean of fn(row) over the item's episodes where fn is defined (not None).

    Returns ({item: Fraction}, micro_total, micro_n); items with no defined episode are left out.
    """
    values, total, count = {}, Fraction(0), 0
    for item in sorted(rows_by_item if items is None else items):
        defined = [exact(v) for v in (fn(row) for row in rows_by_item.get(item, ())) if v is not None]
        if defined:
            subtotal = sum(defined, Fraction(0))
            values[item] = subtotal / len(defined)
            total += subtotal
            count += len(defined)
    return values, total, count


def family_weighted(values, family_of, *, families=False):
    """Family mean of item values, then the equal-weight mean over families, with a t interval (df = F - 1)."""
    if not values:
        return {"estimate": None, "estimate_exact": None, "ci95_t": None, "df": None, "family_sd": None,
                "n_items": 0, "n_families": 0}
    groups = defaultdict(list)
    for item in sorted(values):
        groups[family_of[item]].append(exact(values[item]))
    fam = {name: sum(groups[name], Fraction(0)) / len(groups[name]) for name in sorted(groups)}
    estimate = sum(fam.values(), Fraction(0)) / len(fam)
    interval = A.t_interval([fam[name] for name in sorted(fam)], CONFIDENCE)
    out = {"estimate": float(estimate), "estimate_exact": ratio_text(estimate),
           "ci95_t": None if interval.get("lower") is None else [interval["lower"], interval["upper"]],
           "df": interval.get("df"), "family_sd": interval.get("sd"), "n_items": len(values), "n_families": len(fam)}
    if interval.get("note"):
        out["interval_note"] = interval["note"]
    if families:
        out["family_values"] = {name: float(value) for name, value in fam.items()}
        out["family_items"] = {name: len(groups[name]) for name in sorted(groups)}
    return out


def micro(total, count, kind="share"):
    if kind == "share":
        if total.denominator != 1:
            raise ValueError("a share micro total must be an integer count")
        return {"k": int(total), "n": count, "rate": float(total / count) if count else None}
    return {"total": float(total), "n": count, "mean": float(total / count) if count else None}


def rate(rows_by_item, fn, family_of, *, items=None, kind="share", families=False):
    values, total, count = item_means(rows_by_item, fn, items)
    out = family_weighted(values, family_of, families=families)
    out["micro"] = micro(total, count, kind)
    return out


def contrast(t_rows, c_rows, fn, family_of, *, items=None, kind="share", families=False):
    """Paired family-clustered contrast (treatment - control) on the items where both arms are defined."""
    t_vals, _, _ = item_means(t_rows, fn, items)
    c_vals, _, _ = item_means(c_rows, fn, items)
    shared = sorted(set(t_vals) & set(c_vals))
    diffs = {i: t_vals[i] - c_vals[i] for i in shared}
    out = family_weighted(diffs, family_of, families=families)
    _, t_total, t_n = item_means(t_rows, fn, shared)
    _, c_total, c_n = item_means(c_rows, fn, shared)
    t_fw = family_weighted({i: t_vals[i] for i in shared}, family_of, families=families)
    c_fw = family_weighted({i: c_vals[i] for i in shared}, family_of, families=families)
    out.update(treatment=t_fw["estimate"], control=c_fw["estimate"], treatment_micro=micro(t_total, t_n, kind),
               control_micro=micro(c_total, c_n, kind), items_defined_treatment=len(t_vals),
               items_defined_control=len(c_vals),
               micro_item_mean_difference=float(sum(diffs.values(), Fraction(0)) / len(diffs)) if diffs else None)
    if families:
        out["family_treatment"] = t_fw.get("family_values", {})
        out["family_control"] = c_fw.get("family_values", {})
    return out


def quantile(sorted_values, probability):
    """Linear interpolation between order statistics (Hyndman-Fan type 7), as in the frozen analysis."""
    position = (len(sorted_values) - 1) * probability
    lo, hi = math.floor(position), math.ceil(position)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (position - lo)


def distribution(values):
    values = sorted(values)
    if not values:
        return {"n": 0}
    bins = {label: sum(1 for v in values if lo <= v < hi) for lo, hi, label in PRECISION_BINS}
    bins["= 1"] = sum(1 for v in values if v == 1.0)
    return {"n": len(values), "min": values[0], "p10": quantile(values, 0.10), "p25": quantile(values, 0.25),
            "median": quantile(values, 0.5), "p75": quantile(values, 0.75), "max": values[-1],
            "micro_mean": statistics.fmean(values), "share_equal_1": sum(1 for v in values if v == 1.0) / len(values),
            "histogram": bins}


def v_bucket_rule(n_v_by_item):
    """Tertile buckets of |V|, fixed from the item distribution only (no outcome is used).

    Nearest-rank cut points over the N items: c1 = the ceil(N/3)-th smallest |V|, c2 = the
    ceil(2N/3)-th smallest. Buckets: |V| <= c1, c1 < |V| <= c2, |V| > c2 (an empty bucket is dropped).
    """
    values = sorted(n_v_by_item.values())
    count = len(values)
    c1 = values[math.ceil(count / 3) - 1]
    c2 = values[math.ceil(2 * count / 3) - 1]
    labels = [f"n_V<={c1}", f"{c1}<n_V<={c2}", f"n_V>{c2}"]

    def bucket(n_v):
        return labels[0] if n_v <= c1 else labels[1] if n_v <= c2 else labels[2]
    counts = Counter(bucket(v) for v in values)
    return {"rule": ("nearest-rank tertile cut points of |V| over the items: c1 = the ceil(N/3)-th smallest |V|, "
                     "c2 = the ceil(2N/3)-th smallest; buckets |V| <= c1, c1 < |V| <= c2, |V| > c2; fixed from the "
                     "item-level |V| distribution before any outcome by bucket was computed"),
            "n_items": count, "c1": c1, "c2": c2, "labels": [label for label in labels if counts[label]],
            "items_per_bucket": {label: counts[label] for label in labels if counts[label]},
            "V_size_distribution": {str(k): v for k, v in sorted(Counter(values).items())}}, bucket


def trajectory_key(row):
    cost = row["cost"]
    return (row["exit_status"], cost["calls"], cost["prompt_tokens"], cost["completion_tokens"], row["site_recovered"])


def triple_concordance(rows):
    """(identical trajectory, identical outcome, identical exit, number recovered) for one item's seed rows."""
    keys = [trajectory_key(r) for r in rows]
    return (int(all(k == keys[0] for k in keys)), int(len({r["site_recovered"] for r in rows}) == 1),
            int(len({r["exit_status"] for r in rows}) == 1), sum(r["site_recovered"] for r in rows))


# --------------------------------------------------------------------------- per-row functions


def is_submitted(row):
    return int(row["exit_status"] == "Submitted")


def recovered(row):
    return row["site_recovered"]


def recovered_given_submission(row):
    return row["site_recovered"] if row["exit_status"] == "Submitted" else None


def nm_any(row):
    return row["nm_any"]


def nm_any_given_submission(row):
    return row["nm_any"] if row["exit_status"] == "Submitted" else None


def cost_field(name):
    return lambda row: row["cost"][name]


def total_tokens(row):
    return row["cost"]["prompt_tokens"] + row["cost"]["completion_tokens"]


def tool(name):
    return lambda row: (row["cost"]["tool_calls"] or {}).get(name, 0)


def any_tool(name):
    return lambda row: int(((row["cost"]["tool_calls"] or {}).get(name, 0) or 0) > 0)


def all_tool_calls(row):
    return sum((row["cost"]["tool_calls"] or {}).values())


def rejected_submits(row):
    attempts = (row["cost"]["tool_calls"] or {}).get("submit_answer", 0)
    return attempts - 1 if row["exit_status"] == "Submitted" else attempts


def telemetry_flag(name):
    def get(row):
        value = (row.get("telemetry") or {}).get(name)
        return None if value is None else int(bool(value))
    return get


def telemetry_count(name):
    return lambda row: (row.get("telemetry") or {}).get(name)


def given_submission(fn):
    return lambda row: fn(row) if row["exit_status"] == "Submitted" else None


def funnel_stage(stage):
    def get(row):
        value = A._graph_funnel(row)[stage]
        return None if value is None else int(value)
    return get


# --------------------------------------------------------------------------- study


class Study:
    def __init__(self, scores, cells, items, policies, items_sha256):
        self.checks = []
        self.items = {}
        for row in items:
            item_id = row["item_id"]
            if item_id in self.items:
                raise InputError(f"items: duplicate item {item_id}")
            self.items[item_id] = row
        self.family_of = {i: r["family"] for i, r in self.items.items()}
        self.families = sorted(set(self.family_of.values()))
        self.rows = {}
        self.arms = {(block, cond): defaultdict(list) for block, conds in BLOCKS for cond in conds}
        for row in scores:
            cell_id = row["cell_id"]
            if cell_id in self.rows:
                raise InputError(f"scores: duplicate cell {cell_id}")
            key = (row["block"], row["condition"])
            if key not in self.arms:
                raise InputError(f"scores: unexpected block/condition {key}")
            if row["item_id"] not in self.items:
                raise InputError(f"scores: unknown item {row['item_id']}")
            if row["family"] != self.family_of[row["item_id"]]:
                raise InputError(f"scores: {cell_id} family disagrees with items.jsonl")
            if row.get("items_sha256") != items_sha256:
                raise InputError(f"scores: {cell_id} was scored against another items file")
            self.rows[cell_id] = row
            self.arms[key][row["item_id"]].append(row)
        self.arms = {key: {item: sorted(rows, key=lambda r: r["seed"]) for item, rows in sorted(arm.items())}
                     for key, arm in self.arms.items()}
        self._check_design()
        self._check_cells(cells)
        self.policy = {}
        for row in policies:
            if row.get("items_sha256") != items_sha256:
                raise InputError("policies: a row was computed from another items file")
            key = (row["item_id"], row["variant"], row["policy"])
            if key in self.policy:
                raise InputError(f"policies: duplicate row {key}")
            self.policy[key] = row
        self._record("policies.jsonl rows (unique item x variant x policy)", 15000, len(self.policy))
        missing = sum(1 for i in self.items if (i, "omit_node", "c4_hits") not in self.policy)
        self._record("every item has an omit_node c4_hits policy row (items missing one)", 0, missing)

    def _record(self, name, expected, observed, *, tolerance=None):
        if tolerance is None:
            ok = expected == observed
            diff = None
        else:
            ok = expected is not None and observed is not None and abs(expected - observed) <= tolerance
            diff = None if expected is None or observed is None else abs(expected - observed)
        self.checks.append({"check": name, "expected": expected, "observed": observed, "abs_diff": diff, "ok": ok})

    def _check_design(self):
        all_items = sorted(self.items)
        for block, conds in BLOCKS:
            flag = BLOCK_MEMBERSHIP[block]
            designed = [i for i in all_items if flag is None or self.items[i]["membership"].get(flag)]
            seeds = P_SEEDS if block == "P" else (SECONDARY_SEED,)
            for cond in conds:
                arm = self.arms[(block, cond)]
                self._record(f"design {block}/{cond}: items equal the {len(designed)} designed items", True,
                             sorted(arm) == designed)
                bad = [i for i in arm if tuple(r["seed"] for r in arm[i]) != seeds]
                self._record(f"design {block}/{cond}: every item has exactly seeds {list(seeds)}", 0, len(bad))
        states = Counter(r["cell_state"] for r in self.rows.values())
        self._record("every scored cell is completed (cell_state)", {"completed": len(self.rows)}, dict(states))
        self._record("scored cells", 13740, len(self.rows))
        bad_receipts = Counter()
        for row in self.rows.values():
            exposure, cond = row.get("exposure") or {}, row["condition"]
            if exposure.get("guidance_present") is not (cond in ("C2", "C3")):
                bad_receipts["guidance_present"] += 1
            if exposure.get("payload_present") is not (cond in ("C1", "C2", "C3", "C4a")):
                bad_receipts["payload_present"] += 1
            if exposure.get("verification_block_delivered") is not (cond == "C4a"):
                bad_receipts["verification_block_delivered"] += 1
        self._record("exposure receipts match the condition (guidance only in C2/C3, answer payload only in "
                     "C1-C4a, verification block only in C4a): violating cells by field", {}, dict(bad_receipts))

    def _check_cells(self, cells):
        by_id = {}
        for row in cells:
            if row["cell_id"] in by_id:
                raise InputError(f"cells: duplicate cell {row['cell_id']}")
            by_id[row["cell_id"]] = row
        self._record("CELLS.jsonl cell ids equal scores.jsonl cell ids", True, set(by_id) == set(self.rows))
        same = ("block", "item_id", "family", "variant", "condition", "seed", "cell_state", "exit_status",
                "attempt_id", "result_sha256", "endpoint", "gpu_uuid", "infra_attempts", "site_recovered",
                "function_localized", "recall_V_callers", "nm_recall", "nm_any", "n_submitted", "n_unique_sites",
                "precision_lower", "precision_upper", "complete", "wording_pair")
        mismatches = Counter()
        for cell_id, c in sorted(by_id.items()):
            s = self.rows.get(cell_id)
            if s is None:
                continue
            for name in same:
                if c.get(name) != s.get(name):
                    mismatches[name] += 1
            for name in ("calls", "prompt_tokens", "completion_tokens", "finish_reason_length_count"):
                if c.get(name) != s["cost"].get(name):
                    mismatches["cost." + name] += 1
            for name in ("searched_target_name", "mentioned_omitted_caller_file", "tool_calls_before_submit"):
                if c.get("telemetry_" + name) != (s.get("telemetry") or {}).get(name):
                    mismatches["telemetry." + name] += 1
            for name in ("payload_present", "verification_block_delivered", "tools_offered", "recommendation_present"):
                if c.get("exposure_" + name) != (s.get("exposure") or {}).get(name):
                    mismatches["exposure." + name] += 1
            item = self.items[s["item_id"]]
            if c.get("stratum") != item["covariates"].get("stratum"):
                mismatches["stratum_vs_items_covariate"] += 1
            if s.get("wording_pair") != item.get("wording_pair"):
                mismatches["wording_pair_vs_items"] += 1
            if s.get("n_V") != len(item["V"]) or item["covariates"].get("n_V") != len(item["V"]):
                mismatches["n_V_vs_items"] += 1
        self._record("CELLS.jsonl agrees with scores.jsonl (and items) field by field: mismatching fields", {},
                     dict(sorted(mismatches.items())))

    def arm(self, block, cond):
        return self.arms[(block, cond)]


# --------------------------------------------------------------------------- section 1: exits


def section_exits(st):
    rows, all_noanswer, at_budget = [], 0, 0
    for block, conds in BLOCKS:
        for cond in conds:
            eps = [r for item in sorted(st.arm(block, cond)) for r in st.arm(block, cond)[item]]
            exits = Counter(r["exit_status"] for r in eps)
            calls = {}
            for status in sorted(exits):
                hist = Counter(r["cost"]["calls"] for r in eps if r["exit_status"] == status)
                calls[status] = {str(k): v for k, v in sorted(hist.items())}
            noanswer = [r for r in eps if r["exit_status"] == "NoAnswer"]
            submitted = [r for r in eps if r["exit_status"] == "Submitted"]
            all_noanswer += len(noanswer)
            at_budget += sum(r["cost"]["calls"] == RESPONSE_BUDGET for r in noanswer)
            rows.append({"block": block, "condition": cond, "episodes": len(eps),
                         "Submitted": exits.get("Submitted", 0), "NoAnswer": exits.get("NoAnswer", 0),
                         "other_exits": {k: v for k, v in sorted(exits.items()) if k not in ("Submitted", "NoAnswer")},
                         "calls_at_exit_by_exit": calls,
                         "noanswer_at_response_8": sum(r["cost"]["calls"] == RESPONSE_BUDGET for r in noanswer),
                         "noanswer_with_submit_attempt": sum(
                             (r["cost"]["tool_calls"] or {}).get("submit_answer", 0) > 0 for r in noanswer),
                         "noanswer_recovered_C_star": sum(r["site_recovered"] for r in noanswer),
                         "submitted_at_response_8": sum(r["cost"]["calls"] == RESPONSE_BUDGET for r in submitted),
                         "max_calls": max(r["cost"]["calls"] for r in eps),
                         "infra_attempts_total": sum(r.get("infra_attempts") or 0 for r in eps)})
    by_seed = {}
    for cond in P_CONDITIONS:
        arm = st.arm("P", cond)
        by_seed[cond] = {}
        for seed in P_SEEDS:
            exits = Counter(r["exit_status"] for item in sorted(arm) for r in arm[item] if r["seed"] == seed)
            by_seed[cond][str(seed)] = dict(sorted(exits.items()))
    verdict = ("CONFIRMED: every NoAnswer exit ({0}/{1}) ended at the 8th model response (the response budget)"
               if all_noanswer and at_budget == all_noanswer else
               "REFUTED: {0}/{1} NoAnswer exits ended at the 8th model response").format(at_budget, all_noanswer)
    return {"label": EXPLORATORY, "rows": rows, "P_exits_by_seed": by_seed, "noanswer_total": all_noanswer,
            "noanswer_at_response_8": at_budget, "verdict_noanswer_at_budget": verdict,
            "notes": ["calls = model responses used (cost.calls, accounting.model_calls); the harness ends an "
                      "episode as NoAnswer when 8 responses pass without an accepted submit_answer",
                      "noanswer_with_submit_attempt counts NoAnswer episodes with >= 1 submit_answer tool call (each "
                      "such call was rejected, since no submission was accepted)",
                      "format-error counts per episode are not in scores.jsonl or CELLS.jsonl (only in result.json "
                      "accounting, not read here)"]}


# --------------------------------------------------------------------------- section 2: decomposition


def decomposition_row(st, block, cond):
    arm = st.arm(block, cond)
    out = {"block": block, "condition": cond,
           "submission_rate": rate(arm, is_submitted, st.family_of),
           "recovery_given_submission": rate(arm, recovered_given_submission, st.family_of),
           "overall_recovery": rate(arm, recovered, st.family_of)}
    if block == "S-NAT":
        out["nm_any_given_submission"] = rate(arm, nm_any_given_submission, st.family_of)
        out["nm_any_overall"] = rate(arm, nm_any, st.family_of)
    return out


def section_decomposition(st):
    rows = [decomposition_row(st, block, cond) for block, conds in BLOCKS for cond in conds]
    p_contrasts = {}
    for hyp, treat, ctrl in HYPOTHESES:
        t_arm, c_arm = st.arm("P", treat), st.arm("P", ctrl)
        p_contrasts[hyp] = {"pair": f"{treat} - {ctrl}",
                            "submission_rate": contrast(t_arm, c_arm, is_submitted, st.family_of),
                            "recovery_given_submission": contrast(t_arm, c_arm, recovered_given_submission,
                                                                  st.family_of),
                            "overall_recovery": contrast(t_arm, c_arm, recovered, st.family_of)}
    secondary = {}
    for block, treat, ctrl in SECONDARY_PAIRS:
        t_arm, c_arm = st.arm(block, treat), st.arm(block, ctrl)
        entry = {"pair": f"{treat} - {ctrl}", "block": block,
                 "submission_rate": contrast(t_arm, c_arm, is_submitted, st.family_of),
                 "recovery_given_submission": contrast(t_arm, c_arm, recovered_given_submission, st.family_of),
                 "overall_recovery": contrast(t_arm, c_arm, recovered, st.family_of)}
        if block == "S-NAT":
            entry["nm_any_given_submission"] = contrast(t_arm, c_arm, nm_any_given_submission, st.family_of)
        secondary[f"{block}:{treat}-{ctrl}"] = entry
    return {"label": EXPLORATORY, "rows": rows, "P_contrasts": p_contrasts, "secondary_block_contrasts": secondary,
            "notes": ["submission_rate: item mean of 1[exit == Submitted] over the available seeds -> family -> "
                      "equal-weight mean over families",
                      "recovery_given_submission: item mean of site_recovered over the item's Submitted seeds only "
                      "(items with no Submitted seed drop out) -> family -> mean; POST-TREATMENT CONDITIONING on "
                      "submission, descriptive only",
                      "recovery-given-submission contrasts use only items where BOTH arms submitted at least once "
                      "(n_items shows how many); post-treatment conditioning, descriptive only",
                      "overall_recovery = the outcome site_recovered (non-Submitted exits score 0); the P "
                      "overall_recovery contrasts reproduce H1-H3",
                      "micro: k/n over episodes; at the micro level overall recovery = submission x "
                      "recovery-given-submission exactly; the family-weighted values need not multiply",
                      "where every family has the same number of items and every item the same number of episodes "
                      "(P, S-VC, S-EDGE, T), the family-weighted submission and overall rates equal the micro rates "
                      "exactly; conditional rates (given submission) and the unbalanced S-NAT block differ",
                      "S-NAT: site_recovered of C* is a baseline there (the block's outcomes are nm_recall/nm_any); "
                      "nm_any is decomposed as well"]}


# --------------------------------------------------------------------------- section 3: seed concordance


def section_seeds(st):
    rows, pooled = [], Counter()
    for cond in P_CONDITIONS:
        arm = st.arm("P", cond)
        traj, outcome, exit_same, patterns = {}, {}, {}, Counter()
        for item in sorted(arm):
            same_traj, same_outcome, same_exit, n_rec = triple_concordance(arm[item])
            traj[item], outcome[item], exit_same[item] = Fraction(same_traj), Fraction(same_outcome), \
                Fraction(same_exit)
            patterns[n_rec] += 1
        n = len(arm)
        entry = {"condition": cond, "triples": n,
                 "identical_trajectory": family_weighted(traj, st.family_of),
                 "identical_outcome": family_weighted(outcome, st.family_of),
                 "identical_exit": family_weighted(exit_same, st.family_of),
                 "seeds_recovered_pattern": {f"{k}_of_3": patterns.get(k, 0) for k in range(4)}}
        for name, values in (("identical_trajectory", traj), ("identical_outcome", outcome),
                             ("identical_exit", exit_same)):
            k = int(sum(values.values()))
            entry[name]["micro"] = {"k": k, "n": n, "rate": k / n if n else None}
            pooled[name] += k
        pooled["triples"] += n
        rows.append(entry)
    return {"label": EXPLORATORY, "rows": rows,
            "pooled_micro": {name: {"k": pooled[name], "n": pooled["triples"],
                                    "rate": pooled[name] / pooled["triples"] if pooled["triples"] else None}
                             for name in ("identical_trajectory", "identical_outcome", "identical_exit")},
            "definition": ("item-condition triple = the item's three P episodes (seeds 11, 23, 47) in one condition; "
                           "identical trajectory = all three share (exit_status, calls, prompt_tokens, "
                           "completion_tokens, site_recovered); identical outcome = all three share site_recovered; "
                           "family-weighted: the item indicator -> family mean -> equal-weight mean")}


# --------------------------------------------------------------------------- section 4: per-family table


def section_families(st):
    out = {"label": EXPLORATORY, "hypotheses": {}}
    for hyp, treat, ctrl in HYPOTHESES:
        result = contrast(st.arm("P", treat), st.arm("P", ctrl), recovered, st.family_of, families=True)
        diffs = result["family_values"]
        values = [diffs[f] for f in sorted(diffs)]
        exact_diffs = {}
        t_vals, _, _ = item_means(st.arm("P", treat), recovered)
        c_vals, _, _ = item_means(st.arm("P", ctrl), recovered)
        for item in sorted(set(t_vals) & set(c_vals)):
            exact_diffs.setdefault(st.family_of[item], []).append(t_vals[item] - c_vals[item])
        fam_exact = {f: sum(v, Fraction(0)) / len(v) for f, v in sorted(exact_diffs.items())}
        table = [{"family": f, "n_items": result["family_items"][f], "treatment": result["family_treatment"][f],
                  "control": result["family_control"][f], "difference": diffs[f],
                  "difference_exact": ratio_text(fam_exact[f])} for f in sorted(diffs)]
        out["hypotheses"][hyp] = {
            "pair": f"{treat} - {ctrl}", "estimate": result["estimate"], "ci95_t": result["ci95_t"],
            "n_items": result["n_items"], "n_families": result["n_families"], "families": table,
            "heterogeneity": {"sd_family_differences": statistics.stdev(values) if len(values) > 1 else None,
                              "min": min(values), "max": max(values), "median": statistics.median(values),
                              "families_positive": sum(1 for v in fam_exact.values() if v > 0),
                              "families_negative": sum(1 for v in fam_exact.values() if v < 0),
                              "families_zero": sum(1 for v in fam_exact.values() if v == 0)}}
    return out


# --------------------------------------------------------------------------- section 5: strata


def strata_splits(st):
    """{split name: (description, {item: level}, levels with contrasts, levels reported as counts only)}."""
    items = sorted(st.items)
    stratum = {i: st.items[i]["covariates"].get("stratum") for i in items}
    wording = {i: st.items[i].get("wording_pair") for i in items}
    rule, bucket = v_bucket_rule({i: len(st.items[i]["V"]) for i in items})
    v_size = {i: bucket(len(st.items[i]["V"])) for i in items}
    c4 = {}
    for i in items:
        row = st.policy.get((i, "omit_node", "c4_hits"))
        c4[i] = None if row is None else ("c4_hits_recovers_C*" if row["site_recovered"] else "c4_hits_misses_C*")
    return {
        "stratum": ("C* stratum (items covariates.stratum, pre-outcome)", stratum, ["direct", "dynamic"],
                    sorted({v for v in stratum.values() if v not in ("direct", "dynamic")}, key=str)),
        "wording_pair": ("wording pair (items wording_pair)", wording, ["A", "B"], []),
        "V_size": ("size of V, tertile buckets: " + rule["rule"], v_size, rule["labels"], []),
        "c4_hits_policy": ("frozen c4_hits reference policy recovers C* on omit_node (policies.jsonl, policy c4_hits, "
                           "variant omit_node, field site_recovered): did the C4a lexical block contain a C* site",
                           c4, ["c4_hits_recovers_C*", "c4_hits_misses_C*"], []),
    }, rule


def section_strata(st):
    splits, rule = strata_splits(st)
    out = {"label": EXPLORATORY, "V_size_rule": rule, "splits": {}}
    for name, (description, level_of, levels, count_only) in splits.items():
        entry = {"description": description, "levels": {}, "counts_only": {}}
        for level in levels:
            ids = sorted(i for i, v in level_of.items() if v == level)
            fams = sorted({st.family_of[i] for i in ids})
            level_entry = {"n_items": len(ids), "n_families": len(fams), "contrasts": {}, "rates": {}}
            for hyp, treat, ctrl in HYPOTHESES:
                level_entry["contrasts"][hyp] = contrast(st.arm("P", treat), st.arm("P", ctrl), recovered,
                                                         st.family_of, items=ids)
            for cond in P_CONDITIONS:
                level_entry["rates"][cond] = rate(st.arm("P", cond), recovered, st.family_of, items=ids)
            entry["levels"][level] = level_entry
        for level in count_only:
            ids = sorted(i for i, v in level_of.items() if v == level)
            entry["counts_only"][str(level)] = {"n_items": len(ids),
                                                "n_families": len({st.family_of[i] for i in ids})}
        missing = sorted(i for i, v in level_of.items() if v is None)
        if missing:
            entry["items_without_value"] = missing
        out["splits"][name] = entry
    agree = Counter()
    for i in sorted(st.items):
        row = st.policy.get((i, "omit_node", "c4_hits"))
        covariate = st.items[i]["covariates"].get("c4_contains_omitted_site")
        agree[f"policy={None if row is None else bool(row['site_recovered'])}|"
              f"covariate_c4_contains_omitted_site={covariate}"] += 1
    out["c4_hits_policy_vs_items_covariate"] = dict(sorted(agree.items()))
    out["notes"] = ["each level: the H1-H3 paired contrasts (site_recovered, P seeds 11/23/47) on the level's items; "
                    "families with >= 1 item in the level; t interval df = F_level - 1",
                    "strata are pre-outcome item properties; the splits were chosen after the primary results "
                    "were seen, so they are EXPLORATORY; no interaction test is made",
                    "design fact: the wording pair changes only the C2/C3 guidance sentence (prereg section 6; the "
                    "receipts check above confirms guidance only in C2/C3 cells), so for H2 (C4a - C1) and H3 "
                    "(C1 - C0) neither arm's prompt depends on the pair: the A/B split of H2/H3 is a hash partition "
                    "of the items that changes no prompt"]
    return out


# --------------------------------------------------------------------------- section 6: tool use in P


TOOL_METRICS = (
    ("bash_calls", tool("bash"), "mean"),
    ("source_reference_search_calls", tool("source_reference_search"), "mean"),
    ("submit_answer_attempts", tool("submit_answer"), "mean"),
    ("rejected_submit_answer_calls", rejected_submits, "mean"),
    ("all_tool_calls", all_tool_calls, "mean"),
    ("any_source_reference_search", any_tool("source_reference_search"), "share"),
    ("any_bash", any_tool("bash"), "share"),
    ("model_responses_calls", cost_field("calls"), "mean"),
    ("prompt_tokens", cost_field("prompt_tokens"), "mean"),
    ("completion_tokens", cost_field("completion_tokens"), "mean"),
    ("searched_target_name", telemetry_flag("searched_target_name"), "share"),
    ("mentioned_omitted_caller_file", telemetry_flag("mentioned_omitted_caller_file"), "share"),
    ("tool_calls_before_submit", telemetry_count("tool_calls_before_submit"), "mean"),
)


def section_tools(st):
    rows = {}
    by_exit = {}
    for cond in P_CONDITIONS:
        arm = st.arm("P", cond)
        rows[cond] = {name: rate(arm, fn, st.family_of, kind=kind) for name, fn, kind in TOOL_METRICS}
        by_exit[cond] = {}
        for status in ("Submitted", "NoAnswer"):
            eps = [r for item in sorted(arm) for r in arm[item] if r["exit_status"] == status]
            by_exit[cond][status] = {"episodes": len(eps)}
            for name, fn, kind in TOOL_METRICS:
                vals = [fn(r) for r in eps]
                vals = [v for v in vals if v is not None]
                by_exit[cond][status][name] = statistics.fmean(vals) if vals else None
    mismatch = 0
    for row in st.rows.values():
        calls = {k: v for k, v in (row["cost"]["tool_calls"] or {}).items() if v}
        actions = {k: v for k, v in ((row.get("telemetry") or {}).get("tool_actions") or {}).items() if v}
        mismatch += calls != actions
    equal = {}
    for cond in P_CONDITIONS:
        eps = [r for item in sorted(st.arm("P", cond)) for r in st.arm("P", cond)[item]]
        equal[cond] = {"k": sum(all_tool_calls(r) == r["cost"]["calls"] for r in eps), "n": len(eps)}
    equal_all = {"k": sum(all_tool_calls(r) == r["cost"]["calls"] for r in st.rows.values()), "n": len(st.rows)}
    return {"label": EXPLORATORY, "by_condition": rows, "micro_by_exit": by_exit,
            "cost_tool_calls_vs_telemetry_tool_actions_mismatching_episodes_all_blocks": mismatch,
            "episodes_executed_tool_calls_equal_model_responses": {"P_by_condition": equal, "all_blocks": equal_all},
            "notes": ["from scores.jsonl cost.tool_calls (executed tool actions; a submit_answer call counts whether or "
                      "not it was accepted; calls after an accepted submission are not executed and not counted)",
                      "rejected_submit_answer_calls = submit_answer attempts - 1 for Submitted episodes, = attempts for "
                      "NoAnswer episodes",
                      "searched_target_name / mentioned_omitted_caller_file / tool_calls_before_submit: score.py "
                      "telemetry from events.jsonl (process descriptives, never mediators)",
                      "family-weighted: item mean over the 3 seeds -> family -> equal-weight mean; micro: episode mean",
                      "micro_by_exit splits episodes by exit (post-treatment; descriptive only)"]}


# --------------------------------------------------------------------------- section 7: T funnel


def section_t(st):
    arms = {}
    for cond in ("T-offered", "T-recommended"):
        arm = st.arm("T", cond)
        entry = {"episodes": sum(len(v) for v in arm.values()),
                 "funnel": {stage: rate(arm, funnel_stage(stage), st.family_of) for stage in FUNNEL},
                 "submission_rate": rate(arm, is_submitted, st.family_of),
                 "site_recovered": rate(arm, recovered, st.family_of),
                 "graph_query_calls": rate(arm, tool("graph_query"), st.family_of, kind="mean")}
        requested = rate(arm, funnel_stage("requested"), st.family_of, families=True)
        entry["requested_by_family"] = {
            f: {"requested": sum(funnel_stage("requested")(r) for i in sorted(arm) if st.family_of[i] == f
                                 for r in arm[i]),
                "episodes": sum(len(arm[i]) for i in sorted(arm) if st.family_of[i] == f),
                "rate": requested["family_values"][f]} for f in sorted(requested["family_values"])}
        cross = {}
        for flag in (1, 0):
            eps = [r for i in sorted(arm) for r in arm[i] if funnel_stage("requested")(r) == flag]
            cross["requested" if flag else "not_requested"] = {
                "episodes": len(eps), "site_recovered": sum(r["site_recovered"] for r in eps),
                "submitted": sum(is_submitted(r) for r in eps)}
        entry["micro_site_recovered_by_requested_post_treatment"] = cross
        entry["recommendation_present"] = sum(bool((r.get("exposure") or {}).get("recommendation_present"))
                                              for i in sorted(arm) for r in arm[i])
        arms[cond] = entry
    rec, off = st.arm("T", "T-recommended"), st.arm("T", "T-offered")
    return {"label": EXPLORATORY, "arms": arms,
            "recommended_minus_offered": {
                "requested": contrast(rec, off, funnel_stage("requested"), st.family_of),
                "visible_nonempty_answer": contrast(rec, off, funnel_stage("visible_nonempty_answer"), st.family_of),
                "submission_rate": contrast(rec, off, is_submitted, st.family_of),
                "site_recovered": contrast(rec, off, recovered, st.family_of)},
            "pilot_note": PILOT_T_NOTE,
            "notes": ["stages from the frozen absent_edges.probe.analysis._graph_funnel: offered = graph_query in the "
                      "tools offered; requested = >= 1 executed graph_query tool action or a graph_query request inside "
                      "a format-error response; executed = some graph_query status not in {rejected, "
                      "invalid_arguments, error, None}; visible_nonempty_answer = graph_query_nonempty_answers > 0",
                      "'consumed' is not derivable from scores.jsonl (needs events-level reading)",
                      "T runs at seed 11 only, so the item value is the single episode",
                      "micro_site_recovered_by_requested_post_treatment conditions on a post-treatment choice; "
                      "descriptive only"]}


# --------------------------------------------------------------------------- section 8: S-VC over-caution


SVC_METRICS = (("submission_rate", is_submitted, "share"),
               ("recovery_given_submission", recovered_given_submission, "share"),
               ("overall_recovery", recovered, "share"),
               ("calls", cost_field("calls"), "mean"),
               ("prompt_tokens", cost_field("prompt_tokens"), "mean"),
               ("completion_tokens", cost_field("completion_tokens"), "mean"),
               ("total_tokens", total_tokens, "mean"),
               ("n_submitted_given_submission", given_submission(lambda r: r["n_submitted"]), "mean"),
               ("complete_yes_given_submission", given_submission(lambda r: int(r["complete"] == "yes")), "share"))


def section_svc(st):
    c1, c3 = st.arm("S-VC", "C1"), st.arm("S-VC", "C3")
    arms = {cond: {name: rate(st.arm("S-VC", cond), fn, st.family_of, kind=kind) for name, fn, kind in SVC_METRICS}
            for cond in ("C1", "C3")}
    diffs = {name: contrast(c3, c1, fn, st.family_of, kind=kind) for name, fn, kind in SVC_METRICS}
    return {"label": EXPLORATORY, "arms": arms, "C3_minus_C1": diffs,
            "notes": ["S-VC: vcomplete answer (C* included), seed 11; C3 adds the non-exhaustiveness guidance",
                      "*_given_submission rows condition on submission (post-treatment); their contrasts use only "
                      "items where both arms submitted; descriptive only",
                      "complete_yes_given_submission: share of submissions that claimed complete = yes"]}


# --------------------------------------------------------------------------- section 9: precision


def section_precision(st):
    rows = []
    for block, conds in BLOCKS:
        for cond in conds:
            arm = st.arm(block, cond)
            eps = [r for i in sorted(arm) for r in arm[i]]
            sub = [r for r in eps if r["exit_status"] == "Submitted"]
            categories = {c: sum(r["precision_categories"][c] for r in sub) for c in CATEGORIES}
            reasons = Counter()
            for r in sub:
                reasons.update({k: v for k, v in (r.get("unresolved_reasons") or {}).items() if v})
            entry = {"block": block, "condition": cond, "submitted_episodes": len(sub),
                     "submitted_without_entries": sum(1 for r in sub if not r["n_submitted"]),
                     "entries_total": sum(r["n_submitted"] for r in sub),
                     "entries_by_category": categories,
                     "entries_not_in_V": sum(categories[c] for c in NOT_IN_V),
                     "episodes_with_any_entry_not_in_V": sum(
                         1 for r in sub if any(r["precision_categories"][c] for c in NOT_IN_V)),
                     "episodes_with_any_duplicate": sum(1 for r in sub if r["precision_categories"]["duplicate"]),
                     "unresolved_reasons": dict(sorted(reasons.items())),
                     "completeness_claims": {c: sum(1 for r in sub if r["complete"] == c) for c in CLAIMS},
                     "certifiably_false_completeness": {
                         "k": sum(1 for r in sub if r["certifiably_false_completeness"] is True), "n": len(sub)}}
            for name in ("precision_lower", "precision_upper"):
                entry[name] = {"distribution_micro": distribution([r[name] for r in sub if r[name] is not None]),
                               "family_weighted": rate(arm, given_submission(lambda r, n=name: r[n]), st.family_of,
                                                       kind="mean")}
            entry["any_entry_not_in_V_family_weighted"] = rate(
                arm, given_submission(lambda r: int(any(r["precision_categories"][c] for c in NOT_IN_V))),
                st.family_of)
            rows.append(entry)
    return {"label": EXPLORATORY, "rows": rows,
            "notes": ["among Submitted episodes only (precision is undefined otherwise)",
                      "precision_lower = V-sites / all submitted entries; precision_upper = (V + U + lexical) / unique "
                      "valid entries (prereg section 9)",
                      "entries not in V = u_site + lexical_name_match + unresolved (category precedence V > U > "
                      "lexical > unresolved; duplicates are repeated (path, line) entries, counted separately)",
                      "family_weighted precision: item mean over the item's Submitted seeds -> family -> mean"]}


# --------------------------------------------------------------------------- reproduction checks


def reproduction_checks(st, analysis, sections):
    checks = []

    def record(name, expected, observed, tolerance=TOLERANCE):
        if isinstance(expected, (int, float)) and not isinstance(expected, bool) and isinstance(
                observed, (int, float)) and not isinstance(observed, bool):
            diff = abs(expected - observed)
            ok = diff <= tolerance
        else:
            diff, ok = None, expected == observed
        checks.append({"check": name, "expected": expected, "observed": observed, "abs_diff": diff, "ok": ok})

    decomposition = {(r["block"], r["condition"]): r for r in sections["s2"]["rows"]}
    exits = {(r["block"], r["condition"]): r for r in sections["s1"]["rows"]}
    for frozen in analysis["secondary"]["rates"]:
        key = (frozen["block"], frozen["condition"])
        mine = decomposition[key]["overall_recovery"]
        ref = frozen["site_recovered"]
        record(f"rates {key[0]}/{key[1]}: site_recovered family-weighted estimate", ref["estimate"], mine["estimate"])
        record(f"rates {key[0]}/{key[1]}: site_recovered t interval lower", ref["t_interval"]["lower"],
               mine["ci95_t"][0])
        record(f"rates {key[0]}/{key[1]}: site_recovered t interval upper", ref["t_interval"]["upper"],
               mine["ci95_t"][1])
        record(f"rates {key[0]}/{key[1]}: Submitted count", frozen["exit_counts"]["Submitted"],
               exits[key]["Submitted"])
        record(f"rates {key[0]}/{key[1]}: NoAnswer count", frozen["exit_counts"]["NoAnswer"], exits[key]["NoAnswer"])
    for frozen in analysis["primary"]["hypotheses"]:
        hyp = frozen["id"]
        mine = sections["s2"]["P_contrasts"][hyp]["overall_recovery"]
        record(f"primary {hyp}: estimate", frozen["estimate"], mine["estimate"])
        record(f"primary {hyp}: t interval lower", frozen["t_test"]["interval"]["lower"], mine["ci95_t"][0])
        record(f"primary {hyp}: t interval upper", frozen["t_test"]["interval"]["upper"], mine["ci95_t"][1])
        record(f"primary {hyp}: n_items", frozen["n_items"], mine["n_items"])
        record(f"primary {hyp}: n_families", frozen["n_families"], mine["n_families"])
        record(f"primary {hyp}: treatment rate", frozen["treatment_rate"], mine["treatment"])
        record(f"primary {hyp}: control rate", frozen["control_rate"], mine["control"])
        mine_fam = {r["family"]: r for r in sections["s4"]["hypotheses"][hyp]["families"]}
        worst = max(abs(u["difference"] - mine_fam[u["family"]]["difference"]) for u in frozen["family_units"])
        record(f"primary {hyp}: per-family differences, max abs diff over {len(frozen['family_units'])} families",
               0.0, worst)
        record(f"primary {hyp}: per-family exact differences identical", True,
               {u["family"]: u["difference_exact"] for u in frozen["family_units"]}
               == {f: r["difference_exact"] for f, r in mine_fam.items()})
    strata = analysis["secondary"]["strata"]
    for split, mine_split, mapping in (("stratum", "stratum", {"direct": "direct", "dynamic": "dynamic"}),
                                       ("wording_pair", "wording_pair", {"A": "A", "B": "B"}),
                                       ("c4_visibility", "c4_hits_policy", {"c4_shows_site": "c4_hits_recovers_C*",
                                                                            "c4_lacks_site": "c4_hits_misses_C*"})):
        for frozen_level, mine_level in mapping.items():
            for hyp, _, _ in HYPOTHESES:
                ref = strata[split][frozen_level]["contrasts"][hyp]
                got = sections["s5"]["splits"][mine_split]["levels"][mine_level]["contrasts"][hyp]
                record(f"strata {split}={frozen_level} vs explore {mine_split}={mine_level}: {hyp} estimate",
                       ref["estimate"], got["estimate"])
                record(f"strata {split}={frozen_level} vs explore {mine_split}={mine_level}: {hyp} n_items",
                       ref["n_items"], got["n_items"])
    t_frozen = analysis["secondary"]["t_block"]
    for cond, frozen in t_frozen["arms"].items():
        for stage in FUNNEL:
            record(f"T {cond}: funnel {stage} count", frozen["funnel"][stage],
                   sections["s7"]["arms"][cond]["funnel"][stage]["micro"]["k"])
        record(f"T {cond}: requested family-weighted rate", frozen["requested_rate"]["estimate"],
               sections["s7"]["arms"][cond]["funnel"]["requested"]["estimate"])
        record(f"T {cond}: site_recovered family-weighted rate", frozen["site_recovered"]["estimate"],
               sections["s7"]["arms"][cond]["site_recovered"]["estimate"])
    record("T recommended - offered: requested", t_frozen["recommended_minus_offered_requested"]["estimate"],
           sections["s7"]["recommended_minus_offered"]["requested"]["estimate"])
    record("T recommended - offered: site_recovered", t_frozen["recommended_minus_offered_site_recovered"]["estimate"],
           sections["s7"]["recommended_minus_offered"]["site_recovered"]["estimate"])
    svc = analysis["secondary"]["s_vc"]["differences"]
    for frozen_name, mine_name in (("site_recovered", "overall_recovery"), ("calls", "calls"),
                                   ("total_tokens", "total_tokens")):
        record(f"S-VC C3 - C1: {frozen_name}", svc[frozen_name]["estimate"],
               sections["s8"]["C3_minus_C1"][mine_name]["estimate"])
    find = analysis["findability"]
    record("findability: c4_hits recovers C* (count)", find["c4_hits_recovers_omitted"]["count"],
           sections["s5"]["splits"]["c4_hits_policy"]["levels"]["c4_hits_recovers_C*"]["n_items"])
    record("findability: |V| distribution", {str(k): v for k, v in find["V_size"].items()},
           sections["s5"]["V_size_rule"]["V_size_distribution"])
    record("findability: strata counts", find["strata"],
           {**{lvl: e["n_items"] for lvl, e in sections["s5"]["splits"]["stratum"]["levels"].items()},
            **{lvl: e["n_items"] for lvl, e in sections["s5"]["splits"]["stratum"]["counts_only"].items()}})
    return checks


# --------------------------------------------------------------------------- markdown


def f3(value, signed=False):
    if value is None:
        return "n/a"
    return f"{value:+.3f}" if signed else f"{value:.3f}"


def f1(value):
    return "n/a" if value is None else f"{value:.1f}"


def ci(entry, digits=3):
    interval = entry.get("ci95_t")
    if not interval:
        return "[n/a]"
    flag = " (zero variance: degenerate)" if entry.get("interval_note") else ""
    return f"[{interval[0]:.{digits}f}, {interval[1]:.{digits}f}]{flag}"


def est(entry, signed=False, digits=3):
    if entry is None or entry.get("estimate") is None:
        return "n/a"
    value = entry["estimate"]
    text = f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"
    return f"{text} {ci(entry, digits)}"


def km(entry):
    m = entry.get("micro") or {}
    if "k" in m:
        return f"{m['k']}/{m['n']}" + (f" ({m['rate']:.3f})" if m.get("rate") is not None else "")
    if "mean" in m:
        if m.get("mean") is None:
            return "n/a"
        value = f1(m["mean"]) if abs(m["mean"]) >= 100 else f3(m["mean"])
        return f"{value} (n={m['n']})"
    return "n/a"


def size(entry):
    return f"{entry.get('n_items', 0)} it / {entry.get('n_families', 0)} fam"


def cell(value):
    return str(value).replace("|", "\\|")


def table(headers, rows):
    out = ["| " + " | ".join(cell(h) for h in headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(cell(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def section_header(title):
    return [f"## {title}", "", f"> {EXPLORATORY}", ""]


def render_md(doc, sections):
    """Markdown from ``doc`` (header, inputs, checks) and the sections in their logical order."""
    s = sections
    out = [f"# Exp A {RUN}: exploratory analysis package (EXPLORE)", "",
           f"**{MODEL_LABEL}.** Generated by `scripts/expA_explore.py` (sha256 `{doc['script']['sha256']}`); "
           "every number below is computed by that script from the pinned inputs.", "",
           f"> {EXPLORATORY}", "",
           "Unit (as in the primary analysis): item mean over the available seeds -> family mean -> equal-weight mean "
           "over families; paired contrasts take the item difference first; intervals are Student t over the family "
           "values (df = F - 1, frozen `absent_edges.probe.analysis.t_interval`). Micro k/n over episodes is shown "
           "beside each family-weighted value. No p-values are computed. `it` = items, `fam` = families.", "",
           "## Inputs (sha256 verified before reading)", "",
           table(["Input", "File", "sha256"], [[k, v["file"], f"`{v['sha256']}`"] for k, v in doc["inputs"].items()]),
           ""]
    checks = doc["reproduction_checks"]
    failed = [c for c in checks if not c["ok"]]
    out += [f"**Reproduction and consistency checks:** {len(checks) - len(failed)}/{len(checks)} pass "
            "(frozen ANALYSIS.json rates, H1-H3 estimates and intervals, per-family differences, strata, T funnel, "
            "S-VC differences, findability counts; CELLS.jsonl vs scores.jsonl field agreement; design coverage)."]
    if failed:
        out += ["", "FAILED checks:", ""] + [f"- {c['check']}: expected {c['expected']!r}, observed {c['observed']!r}"
                                            for c in failed]
    out.append("")

    # 1
    s1 = s["s1_exit_partition"]
    out += section_header("1. Exit partition by block x condition")
    out += [f"**{s1['verdict_noanswer_at_budget']}.**", "",
            table(["Block", "Cond", "Episodes", "Submitted", "NoAnswer", "Other exits", "NoAnswer at resp. 8",
                   "NoAnswer w/ submit call", "Submitted at resp. 8", "max calls"],
                  [[r["block"], r["condition"], r["episodes"], r["Submitted"], r["NoAnswer"],
                    json.dumps(r["other_exits"]) if r["other_exits"] else "none", r["noanswer_at_response_8"],
                    r["noanswer_with_submit_attempt"], r["submitted_at_response_8"], r["max_calls"]]
                   for r in s1["rows"]]), "",
            "Responses used at exit (calls histogram, episodes):", "",
            table(["Block", "Cond", "Exit"] + [str(k) for k in range(1, RESPONSE_BUDGET + 1)],
                  [[r["block"], r["condition"], status] + [hist.get(str(k), 0) for k in range(1, RESPONSE_BUDGET + 1)]
                   for r in s1["rows"] for status, hist in r["calls_at_exit_by_exit"].items()]), "",
            "P exits by seed:", "",
            table(["Cond"] + [f"seed {sd} Su/No" for sd in P_SEEDS],
                  [[c] + [f"{v.get('Submitted', 0)}/{v.get('NoAnswer', 0)}" for v in by.values()]
                   for c, by in s1["P_exits_by_seed"].items()]), ""]
    out += [f"- {n}" for n in s1["notes"]] + [""]

    # 2
    s2 = s["s2_decomposition"]
    out += section_header("2. Decomposition: submission x recovery given submission")
    out += [table(["Block", "Cond", "Submission FW [95% t]", "micro", "Rec. given sub. FW [95% t]", "it/fam",
                   "micro", "Overall recovery FW [95% t]", "micro"],
                  [[r["block"], r["condition"], est(r["submission_rate"]), km(r["submission_rate"]),
                    est(r["recovery_given_submission"]), size(r["recovery_given_submission"]),
                    km(r["recovery_given_submission"]), est(r["overall_recovery"]), km(r["overall_recovery"])]
                   for r in s2["rows"]]), ""]
    nat = [r for r in s2["rows"] if r["block"] == "S-NAT"]
    out += ["S-NAT nm_any (the block's own outcome):", "",
            table(["Cond", "nm_any given sub. FW [95% t]", "micro", "nm_any overall FW [95% t]", "micro"],
                  [[r["condition"], est(r["nm_any_given_submission"]), km(r["nm_any_given_submission"]),
                    est(r["nm_any_overall"]), km(r["nm_any_overall"])] for r in nat]), "",
            "P contrasts (paired, family-clustered). Recovery-given-submission uses ONLY items where both arms "
            "submitted at least once: POST-TREATMENT CONDITIONING, descriptive only.", "",
            table(["H", "Pair", "Submission diff [95% t]", "it/fam", "Rec. given sub. diff [95% t]", "it/fam",
                   "T / C rec. given sub.", "Overall diff [95% t]"],
                  [[h, e["pair"], est(e["submission_rate"], True), size(e["submission_rate"]),
                    est(e["recovery_given_submission"], True), size(e["recovery_given_submission"]),
                    f"{f3(e['recovery_given_submission']['treatment'])} / "
                    f"{f3(e['recovery_given_submission']['control'])}", est(e["overall_recovery"], True)]
                   for h, e in s2["P_contrasts"].items()]), "",
            "Secondary blocks, within-block contrasts (seed 11; same restriction for recovery given submission):", "",
            table(["Block", "Pair", "Submission diff [95% t]", "it/fam", "Rec. given sub. diff [95% t]", "it/fam",
                   "Overall diff [95% t]"],
                  [[e["block"], e["pair"], est(e["submission_rate"], True), size(e["submission_rate"]),
                    est(e["recovery_given_submission"], True), size(e["recovery_given_submission"]),
                    est(e["overall_recovery"], True)] for e in s2["secondary_block_contrasts"].values()]), ""]
    out += [f"- {n}" for n in s2["notes"]] + [""]

    # 3
    s3 = s["s3_seed_concordance"]
    out += section_header("3. Seed concordance in P")
    out += [table(["Cond", "Triples", "Identical trajectory FW [95% t]", "micro", "Identical outcome FW [95% t]",
                   "micro", "Identical exit micro", "C* recovered in 0/1/2/3 seeds"],
                  [[r["condition"], r["triples"], est(r["identical_trajectory"]), km(r["identical_trajectory"]),
                    est(r["identical_outcome"]), km(r["identical_outcome"]), km(r["identical_exit"]),
                    "/".join(str(r["seeds_recovered_pattern"][f"{k}_of_3"]) for k in range(4))]
                   for r in s3["rows"]]), "",
            "Pooled over the five conditions (micro): " + "; ".join(
                f"{name} {v['k']}/{v['n']} ({v['rate']:.3f})" for name, v in s3["pooled_micro"].items()), "",
            f"- {s3['definition']}", ""]

    # 4
    s4 = s["s4_per_family"]
    out += section_header("4. Per-family H1-H3 (site_recovered, P, seeds 11/23/47)")
    fams = [r["family"] for r in s4["hypotheses"]["H1"]["families"]]
    by = {h: {r["family"]: r for r in e["families"]} for h, e in s4["hypotheses"].items()}
    out += [table(["Family", "Items", "H1 C3-C2", "H2 C4a-C1", "H3 C1-C0", "C0", "C1", "C2", "C3", "C4a"],
                  [[f, by["H1"][f]["n_items"], f3(by["H1"][f]["difference"], True),
                    f3(by["H2"][f]["difference"], True), f3(by["H3"][f]["difference"], True),
                    f3(by["H3"][f]["control"]), f3(by["H3"][f]["treatment"]), f3(by["H1"][f]["control"]),
                    f3(by["H1"][f]["treatment"]), f3(by["H2"][f]["treatment"])] for f in fams]), "",
            table(["H", "Estimate [95% t]", "SD of family diffs", "min", "median", "max", "fam > 0", "fam < 0",
                   "fam = 0"],
                  [[h, est({"estimate": e["estimate"], "ci95_t": e["ci95_t"]}, True),
                    f3(e["heterogeneity"]["sd_family_differences"]), f3(e["heterogeneity"]["min"], True),
                    f3(e["heterogeneity"]["median"], True), f3(e["heterogeneity"]["max"], True),
                    e["heterogeneity"]["families_positive"], e["heterogeneity"]["families_negative"],
                    e["heterogeneity"]["families_zero"]] for h, e in s4["hypotheses"].items()]), ""]

    # 5
    s5 = s["s5_strata"]
    out += section_header("5. Strata splits for H1-H3")
    rule = s5["V_size_rule"]
    out += [f"V-size bucket rule (n_V = |V|): {rule['rule']}. Cut points c1 = {rule['c1']}, c2 = {rule['c2']}; "
            "items per bucket "
            f"{json.dumps(rule['items_per_bucket'])}; |V| distribution {json.dumps(rule['V_size_distribution'])}.",
            "", f"c4_hits policy vs items covariate `c4_contains_omitted_site`: "
            f"{json.dumps(s5['c4_hits_policy_vs_items_covariate'])}.", ""]
    for name, split in s5["splits"].items():
        out += [f"**{name}**: {split['description']}", "",
                table(["Level", "it/fam", "H", "Estimate [95% t]", "it/fam (pairs)", "T rate", "C rate"],
                      [[level, f"{e['n_items']}/{e['n_families']}", h, est(c, True), size(c), f3(c["treatment"]),
                        f3(c["control"])] for level, e in split["levels"].items() for h, c in e["contrasts"].items()]),
                ""]
        out += ["Rates by condition within level (site_recovered FW [95% t]):", "",
                table(["Level"] + list(P_CONDITIONS),
                      [[level] + [est(e["rates"][c]) for c in P_CONDITIONS] for level, e in split["levels"].items()]),
                ""]
        if split["counts_only"]:
            out += ["Counts only: " + "; ".join(f"{k}: {v['n_items']} items in {v['n_families']} families"
                                                for k, v in split["counts_only"].items()), ""]
    out += [f"- {n}" for n in s5["notes"]] + [""]

    # 6
    s6 = s["s6_tool_use_P"]
    out += section_header("6. Tool use in P by condition")
    out += [table(["Metric"] + list(P_CONDITIONS),
                  [[name] + [f"{est(s6['by_condition'][c][name])}; micro {km(s6['by_condition'][c][name])}"
                             for c in P_CONDITIONS] for name, _, _ in TOOL_METRICS]), "",
            "Micro means by exit (post-treatment split, descriptive):", "",
            table(["Metric"] + [f"{c} {x}" for c in P_CONDITIONS for x in ("Su", "No")],
                  [[name] + [f3(s6["micro_by_exit"][c][x][name]) if s6["micro_by_exit"][c][x][name] is not None
                             else "n/a" for c in P_CONDITIONS for x in ("Submitted", "NoAnswer")]
                   for name, _, _ in TOOL_METRICS] +
                  [["episodes"] + [s6["micro_by_exit"][c][x]["episodes"] for c in P_CONDITIONS
                                   for x in ("Submitted", "NoAnswer")]]), "",
            f"cost.tool_calls vs telemetry.tool_actions (nonzero entries), all blocks: "
            f"{s6['cost_tool_calls_vs_telemetry_tool_actions_mismatching_episodes_all_blocks']} episodes differ.", "",
            "Episodes whose executed tool calls equal their model responses: " + "; ".join(
                f"{c} {v['k']}/{v['n']}" for c, v in
                s6["episodes_executed_tool_calls_equal_model_responses"]["P_by_condition"].items())
            + "; all blocks {k}/{n}.".format(**s6["episodes_executed_tool_calls_equal_model_responses"]["all_blocks"]),
            ""]
    out += [f"- {n}" for n in s6["notes"]] + [""]

    # 7
    s7 = s["s7_T_funnel"]
    out += section_header("7. T block: graph_query funnel per arm")
    out += [table(["Arm", "Stage", "micro k/n", "FW [95% t]"],
                  [[cond, stage, km(e["funnel"][stage]), est(e["funnel"][stage])]
                   for cond, e in s7["arms"].items() for stage in FUNNEL] +
                  [[cond, "Submitted", km(e["submission_rate"]), est(e["submission_rate"])]
                   for cond, e in s7["arms"].items()] +
                  [[cond, "site_recovered", km(e["site_recovered"]), est(e["site_recovered"])]
                   for cond, e in s7["arms"].items()] +
                  [[cond, "graph_query calls (mean)", km(e["graph_query_calls"]), est(e["graph_query_calls"])]
                   for cond, e in s7["arms"].items()]), "",
            table(["Recommended - offered", "Estimate [95% t]", "it/fam"],
                  [[k, est(v, True), size(v)] for k, v in s7["recommended_minus_offered"].items()]), "",
            "Per-family requested rates (k/n episodes):", "",
            table(["Family", "T-offered", "T-recommended"],
                  [[f, f"{s7['arms']['T-offered']['requested_by_family'][f]['requested']}/"
                       f"{s7['arms']['T-offered']['requested_by_family'][f]['episodes']}",
                    f"{s7['arms']['T-recommended']['requested_by_family'][f]['requested']}/"
                    f"{s7['arms']['T-recommended']['requested_by_family'][f]['episodes']}"]
                   for f in sorted(s7["arms"]["T-offered"]["requested_by_family"])]), "",
            "site_recovered by request status within arm (micro; POST-TREATMENT, descriptive only):", "",
            table(["Arm", "Requested: rec/sub/episodes", "Not requested: rec/sub/episodes"],
                  [[cond, "{site_recovered}/{submitted}/{episodes}".format(
                      **e["micro_site_recovered_by_requested_post_treatment"]["requested"]),
                    "{site_recovered}/{submitted}/{episodes}".format(
                        **e["micro_site_recovered_by_requested_post_treatment"]["not_requested"])]
                   for cond, e in s7["arms"].items()]), "", f"- {s7['pilot_note']}"]
    out += [f"- {n}" for n in s7["notes"]] + [""]

    # 8
    s8 = s["s8_SVC_overcaution"]
    out += section_header("8. S-VC over-caution cost: C3 vs C1 on vcomplete (seed 11)")
    out += [table(["Metric", "C1 FW [95% t]", "C1 micro", "C3 FW [95% t]", "C3 micro", "C3 - C1 [95% t]", "it/fam"],
                  [[name, est(s8["arms"]["C1"][name]), km(s8["arms"]["C1"][name]), est(s8["arms"]["C3"][name]),
                    km(s8["arms"]["C3"][name]), est(s8["C3_minus_C1"][name], True), size(s8["C3_minus_C1"][name])]
                   for name, _, _ in SVC_METRICS]), ""]
    out += [f"- {n}" for n in s8["notes"]] + [""]

    # 9
    s9 = s["s9_precision"]
    out += section_header("9. Precision and submitted entries outside V (Submitted episodes)")
    rows_lo, rows_cat = [], []
    for r in s9["rows"]:
        for name in ("precision_lower", "precision_upper"):
            d = r[name]["distribution_micro"]
            if not d.get("n"):
                rows_lo.append([r["block"], r["condition"], name, 0] + ["n/a"] * 5)
                continue
            rows_lo.append([r["block"], r["condition"], name, d["n"], est(r[name]["family_weighted"]),
                            f3(d["micro_mean"]), f"{f3(d['min'])}/{f3(d['p10'])}/{f3(d['p25'])}/{f3(d['median'])}",
                            f3(d["share_equal_1"]), json.dumps(d["histogram"])])
        rows_cat.append([r["block"], r["condition"], r["submitted_episodes"], r["entries_total"],
                         r["entries_by_category"]["v_site"], r["entries_by_category"]["u_site"],
                         r["entries_by_category"]["lexical_name_match"], r["entries_by_category"]["unresolved"],
                         r["entries_by_category"]["duplicate"], r["entries_not_in_V"],
                         f"{r['episodes_with_any_entry_not_in_V']}/{r['submitted_episodes']}",
                         json.dumps(r["unresolved_reasons"]) if r["unresolved_reasons"] else "none",
                         json.dumps(r["completeness_claims"]),
                         f"{r['certifiably_false_completeness']['k']}/{r['certifiably_false_completeness']['n']}"])
    out += [table(["Block", "Cond", "Bound", "n", "FW mean [95% t]", "micro mean", "min/p10/p25/median", "share = 1",
                   "histogram (episodes)"], rows_lo), "",
            table(["Block", "Cond", "Submitted", "Entries", "V", "U", "lexical", "unresolved", "duplicate",
                   "not in V", "episodes w/ not-in-V", "unresolved reasons", "complete claims",
                   "cert. false completeness"], rows_cat), ""]
    out += [f"- {n}" for n in s9["notes"]] + [""]
    return "\n".join(out).rstrip() + "\n"


# --------------------------------------------------------------------------- orchestration


def sorted_deep(value):
    if isinstance(value, dict):
        return {str(k): sorted_deep(value[k]) for k in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [sorted_deep(v) for v in value]
    return value


def build(paths):
    raw, digests = {}, {}
    for key in ("cells", "scores", "items", "policies", "analysis"):
        raw[key], digests[key] = read_verified(paths[key], key)
    scores = parse_jsonl(raw.pop("scores"), "scores")
    cells = parse_jsonl(raw.pop("cells"), "cells")
    items = parse_jsonl(raw.pop("items"), "items")
    policies = parse_jsonl(raw.pop("policies"), "policies")
    analysis = json.loads(raw.pop("analysis").decode("utf-8"), parse_constant=_reject_constant)
    if analysis.get("inputs", {}).get("items", {}).get("sha256") != digests["items"]:
        raise InputError("ANALYSIS.json was computed from another items file")
    if analysis.get("inputs", {}).get("scores", {}).get("sha256") != digests["scores"]:
        raise InputError("ANALYSIS.json was computed from another scores file")
    study = Study(scores, cells, items, policies, digests["items"])
    sections = {"s1": section_exits(study), "s2": section_decomposition(study), "s3": section_seeds(study),
                "s4": section_families(study), "s5": section_strata(study), "s6": section_tools(study),
                "s7": section_t(study), "s8": section_svc(study), "s9": section_precision(study)}
    checks = study.checks + reproduction_checks(study, analysis, sections)
    script_bytes = Path(__file__).read_bytes()
    named = {"s1_exit_partition": sections["s1"], "s2_decomposition": sections["s2"],
             "s3_seed_concordance": sections["s3"], "s4_per_family": sections["s4"], "s5_strata": sections["s5"],
             "s6_tool_use_P": sections["s6"], "s7_T_funnel": sections["s7"], "s8_SVC_overcaution": sections["s8"],
             "s9_precision": sections["s9"]}
    doc = {"label": MODEL_LABEL, "exploratory": EXPLORATORY, "schema": SCHEMA, "run": RUN,
           "script": {"path": "scripts/expA_explore.py", "sha256": hashlib.sha256(script_bytes).hexdigest()},
           "inputs": {key: {"file": Path(paths[key]).name, "sha256": digests[key], "sha256_verified": True}
                      for key in ("cells", "scores", "items", "policies", "analysis")},
           "unit": ("item mean over the available seeds -> family mean -> equal-weight mean over families; paired "
                    "contrasts take the item difference first; Student t interval over the family values "
                    "(df = F - 1), nominal 95%; micro k/n over episodes beside each family-weighted value; no p-values"),
           "population": {"items": len(study.items), "families": len(study.families),
                          "episodes": len(study.rows)},
           "reproduction_checks": sorted_deep(checks),
           "reproduction_checks_passed": sum(1 for c in checks if c["ok"]),
           "reproduction_checks_total": len(checks),
           "sections": sorted_deep(named)}
    return doc, named


def dump(doc):
    return (json.dumps(doc, indent=1, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def write(out_dir, doc, sections, overwrite):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payloads = {"EXPLORE.json": dump(doc), "EXPLORE.md": render_md(doc, sections).encode("utf-8")}
    for name in payloads:
        if (out / name).exists() and not overwrite:
            raise SystemExit(f"refusing to overwrite {out / name} (pass --overwrite)")
    written = {}
    for name, data in payloads.items():
        (out / name).write_bytes(data)
        written[name] = hashlib.sha256(data).hexdigest()
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description="Exp A main-20261101b exploratory analysis (post-results; "
                                                 "not preregistered)")
    for key in ("cells", "scores", "items", "policies", "analysis"):
        parser.add_argument(f"--{key}", required=True, help=f"{key} input (sha256 pinned: {EXPECTED_SHA256[key]})")
    parser.add_argument("--out", required=True, help="output directory for EXPLORE.json and EXPLORE.md")
    parser.add_argument("--overwrite", action="store_true", help="replace existing EXPLORE.json / EXPLORE.md")
    args = parser.parse_args(argv)
    doc, sections = build({k: getattr(args, k) for k in ("cells", "scores", "items", "policies", "analysis")})
    written = write(args.out, doc, sections, args.overwrite)
    print(json.dumps({"out": str(args.out), "written": written,
                      "reproduction_checks": f"{doc['reproduction_checks_passed']}/{doc['reproduction_checks_total']}"},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
