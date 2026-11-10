#!/usr/bin/env python3
"""Prespecified analyses of the Exp A follow-ups A1 (final-turn policy) and A2 (hard-omission panel); stdlib only.

ANALYSIS RECORD. Written 2026-11-04 BEFORE any A1 or A2 outcome existed, and committed
before the first follow-up cell, so that the analysis is fixed prospectively. It was exercised only on SYNTHETIC
fixtures and on the main-run (main-20261101b) scores.

finalturn   protocols/EXPA_FINALTURN.md section 3.
    python scripts/expA_followup_analysis.py finalturn --scores-neutral N/scores.jsonl \\
        --scores-deadline D/scores.jsonl --scores-none O/scores.jsonl --out DIR \\
        [--timeline-neutral N/episodes.jsonl --timeline-deadline D/episodes.jsonl --timeline-none O/episodes.jsonl]
        [--run-dir-neutral RUN --run-dir-deadline RUN --run-dir-none RUN --items items.jsonl [--workers K]]
  * Primary (the single test, alpha 0.05): per item, the mean over {C0, C1} of
    [recovery(deadline) - recovery(neutral)], recovery = site_recovered of the seed-11 block-P cell (a completed
    episode that did not submit scores 0); then the family mean; then the equal-weight mean over families; two-sided
    one-sample t (df = F - 1, scripts/expA_reanalyze_independent.t_statistics) with the frozen degenerate-case rules.
  * S1 deadline - none over {C0, C1}; S2 [C1 - C0](deadline) - [C1 - C0](neutral), and the same against none; S4
    per policy and condition: submission rate, recovery given submission, submitted sites, precision bounds and the
    kind of the final (8th) response. Nominal 95% family-clustered t intervals, no p-values, no significance claims.
  * S3 (H1-H3 per policy) is the frozen scripts/expA_analyze.py output of each run; it is not recomputed here.
  * Mechanism subgroup (descriptive): episodes still open at response 8 (the harness prepared an 8th request) in
    which C* was displayed before response 8 under the STRICT path:line rule of
    reports/expA-main-20261101b/timeline/RULES.md (first strict reveal index <= 7). The per-episode timeline is
    the one scripts/expA_timeline.py builds: either its episodes.jsonl for each run (--timeline-*), or built here
    from the run records with that script's own functions (--run-dir-* with --items). Recovery by policy.
  * Cells pair by (item_id, condition) at seed 11. Unpaired and missing cells are counted. The three runs are never
    pooled with the main run (no main-run input is read).

hard        protocols/EXPA_HARD.md section 4 with deviations D-4 and E-3 (protocols/EXPA_DEVIATIONS.md).
    python scripts/expA_followup_analysis.py hard --scores H/scores.jsonl --main-scores M/scores.jsonl \\
        --items-hard items-hard/items.jsonl [--items-main items-main/items.jsonl] \\
        [--frozen-analysis H/ANALYSIS.json] --out DIR
  * H3 = C1 - C0 on block P: item mean over the primary seeds -> paired item difference -> family mean ->
    equal-weight mean over families; two-sided one-sample t (df = F - 1). D-4: the decision is the UNADJUSTED
    p <= 0.05; no Holm. H1 (C3 - C2) and H2 (C4a - C1): estimate and nominal 95% interval only.
  * Sensitivity weightings of H3 (nominal intervals): item-weighted (every item counts once) and caller-balanced
    (every distinct C* counts once: an item's weight is 1 / the number of contributing items with its C*). Their
    intervals are family-clustered: the weighted mean is linearised into one pseudo-value per family,
    u_f = est + F * (S_f - est * W_f) / W, and t_statistics is applied to the u_f (df = F - 1). With weights
    1 / n_f this reduces exactly to the primary t over family means.
  * S-VC positive control (C1, C3, seed 11), submission rates, and the E-3 within-target contrasts: the targets
    shared with the main run (same item_id) whose main-run C* differs from the hard C* and is a direct call
    (expected 17). Joins are keyed by (run, item_id). Descriptive.
  * Primary seeds follow the frozen rule: the longest prefix of (11, 23, 47) whose block-P cells are all terminal
    (completed, or missing_infrastructure after the retry cap). Item means use completed cells.

Outputs (in --out, refused if present unless --overwrite): RESULTS.json (every number, sorted keys, input
sha256s; no timestamps) and RESULTS.md (short tables). Anything the inputs cannot support is reported as not
available, with the reason, instead of being guessed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from expA_reanalyze_independent import t_statistics  # noqa: E402  (stdlib t, exact rationals)

SCHEMA = "expA-followup-analysis-v1"
LABEL = "ANALYSIS RECORD"
CONDITIONS = ("C0", "C1", "C2", "C3", "C4a")
P_VARIANT = {"C0": "none", "C1": "omit_node", "C2": "omit_node", "C3": "omit_node", "C4a": "omit_node"}
SVC_VARIANT, SVC_CONDITIONS, SVC_SEED = "vcomplete", ("C1", "C3"), 11
POLICIES = ("neutral", "deadline", "none")
FINALTURN_SEED = 11
HARD_SEEDS = (11, 23, 47)
TERMINAL_STATES = ("completed", "missing_infrastructure")
HYPOTHESES = (("H1", "C3", "C2"), ("H2", "C4a", "C1"), ("H3", "C1", "C0"))
CONFIDENCE, ALPHA = 0.95, 0.05
BUDGET = 8
EXPECTED_WITHIN_TARGET = 17
NOT_RECOMPUTED_S3 = ("S3 (H1-H3 per policy) is computed by the frozen scripts/expA_analyze.py on each policy run "
                     "(its ANALYSIS.json); this script does not recompute it")
REQUIRED = ("cell_id", "block", "item_id", "condition", "variant", "seed", "family", "cell_state", "exit_status",
            "site_recovered")


class Refusal(SystemExit):
    pass


# --------------------------------------------------------------------------- input


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as error:
                raise Refusal(f"{path}:{number}: not JSON ({error})") from error
            if not isinstance(row, dict):
                raise Refusal(f"{path}:{number}: not a JSON object")
            rows.append(row)
    return rows


def load_scores(path):
    """Episode rows of one run (scripts/expA_score.py scores.jsonl); one row per cell_id."""
    rows = load_jsonl(path)
    seen = set()
    for number, row in enumerate(rows, 1):
        missing = [k for k in REQUIRED if k not in row]
        if missing:
            raise Refusal(f"{path}: row {number} lacks {missing}")
        if row["cell_id"] in seen:
            raise Refusal(f"{path}: duplicate cell_id {row['cell_id']}")
        seen.add(row["cell_id"])
        if row["cell_state"] == "completed" and row["site_recovered"] not in (0, 1, True, False):
            raise Refusal(f"{path}: completed cell {row['cell_id']} has site_recovered {row['site_recovered']!r}")
    if not rows:
        raise Refusal(f"{path}: no rows")
    return rows


def recovery(row):
    """Unconditional site_recovered of a completed episode (non-submitted = 0); None for a missing cell."""
    if row is None or row["cell_state"] != "completed":
        return None
    return Fraction(int(row["site_recovered"])) if row["exit_status"] == "Submitted" else Fraction(0)


def submitted(row):
    if row is None or row["cell_state"] != "completed":
        return None
    return Fraction(int(row["exit_status"] == "Submitted"))


def merge_families(maps):
    family = {}
    for source, mapping in maps:
        for item, fam in mapping.items():
            if family.setdefault(item, fam) != fam:
                raise Refusal(f"item {item} has family {family[item]!r} elsewhere and {fam!r} in {source}")
    return family


def input_record(path):
    return {"path": str(path), "sha256": sha256(path)}


# --------------------------------------------------------------------------- statistics


def fstr(x):
    return f"{x.numerator}/{x.denominator}"


def mean(values):
    values = list(values)
    return sum(values, Fraction(0)) / len(values)


def family_means(item_values, item_family):
    groups = defaultdict(list)
    for item, value in item_values.items():
        groups[item_family[item]].append(value)
    return {fam: mean(vals) for fam, vals in sorted(groups.items())}


def summarize(values, *, test=False):
    """Mean of exact values with a nominal 95% t interval (df = n - 1); with test=True also t and two-sided p.

    The frozen degenerate-case rules (absent_edges/probe/analysis.py t_test): every value equal to 0 -> p = 1 and a
    degenerate interval; every value equal to the same nonzero value (this includes n = 1) -> t undefined and the
    exact two-sided sign-flip p, which for n identical nonzero values is 2 / 2**n (1 when n = 1).
    """
    values = list(values)
    n = len(values)
    if n == 0:
        out = {"status": "not_estimable", "n": 0, "estimate": None, "ci_low": None, "ci_high": None, "df": None}
        if test:
            out.update(t=None, p_two_sided=None, p_source=None)
        return out
    m = mean(values)
    out = {"n": n, "df": n - 1, "estimate": float(m), "estimate_exact": fstr(m), "confidence": CONFIDENCE}
    if all(v == values[0] for v in values):
        out.update(status="all_zero" if m == 0 else "constant_nonzero", ci_low=float(m), ci_high=float(m),
                   degenerate_interval=True,
                   note="zero variance: degenerate interval, not evidence of equivalence")
        if test:
            if m == 0:
                out.update(t=None, p_two_sided=1.0, p_source="all_zero: p = 1 (frozen rule)")
            else:
                out.update(t=None, p_two_sided=float(min(Fraction(1), Fraction(2, 2 ** n))),
                           p_source="constant_nonzero: exact sign-flip p = 2 / 2**F (frozen rule)")
        return out
    stats = t_statistics(values, CONFIDENCE)
    out.update(status="tested" if test else "interval", ci_low=stats["lower"], ci_high=stats["upper"],
               degenerate_interval=False, se=stats["se"])
    if test:
        out.update(t=stats["t"], p_two_sided=stats["p_two_sided"], p_source="t_statistics (df = F - 1)")
    return out


def weighted_cluster(diffs, weights, item_family):
    """Weighted mean of item values with a family-clustered nominal 95% t interval (df = F - 1).

    est = sum(w d) / sum(w). Each family f contributes the pseudo-value u_f = est + F (S_f - est W_f) / W, with
    S_f = sum over its items of w d, W_f = sum of their w and W = sum W_f; mean(u_f) = est, and t_statistics over
    the u_f is the linearised (CR1-type) cluster interval. Weights 1 / n_f give u_f = family mean exactly.
    """
    items = sorted(diffs)
    if not items:
        return summarize([])
    total_w = sum((weights[i] for i in items), Fraction(0))
    est = sum((weights[i] * diffs[i] for i in items), Fraction(0)) / total_w
    s_f, w_f = defaultdict(Fraction), defaultdict(Fraction)
    for i in items:
        s_f[item_family[i]] += weights[i] * diffs[i]
        w_f[item_family[i]] += weights[i]
    n_fam = len(s_f)
    pseudo = [est + n_fam * (s_f[f] - est * w_f[f]) / total_w for f in sorted(s_f)]
    out = summarize(pseudo)
    out["estimate"], out["estimate_exact"] = float(est), fstr(est)
    out["n_items"], out["n_families"] = len(items), n_fam
    return out


def micro(values):
    values = list(values)
    if not values:
        return {"n": 0, "mean": None}
    m = mean(values)
    return {"n": len(values), "sum": float(sum(values, Fraction(0))), "mean": float(m)}


def family_weighted(item_values, item_family):
    """item value -> family mean -> equal-weight mean over families, nominal 95% t interval (no p)."""
    fm = family_means(item_values, item_family)
    out = summarize(fm.values())
    out["n_items"], out["n_families"] = len(item_values), len(fm)
    return out


def contrast(fn, needed, cells, items, item_family, *, test=False):
    """Item-level contrast fn(get) over items whose every needed (source, condition) cell has a value."""
    values, dropped = {}, Counter()
    for item in items:
        got = {key: cells[key[0]].get((item, key[1])) for key in needed}
        absent = [key for key, value in got.items() if value is None]
        if absent:
            dropped["items_missing_a_needed_cell"] += 1
            for key in absent:
                dropped[f"missing:{key[0]}:{key[1]}"] += 1
            continue
        values[item] = fn(got)
    fm = family_means(values, item_family)
    out = summarize(fm.values(), test=test)
    out.update(n_items=len(values), n_families=len(fm), dropped=dict(sorted(dropped.items())),
               family_values={f: float(v) for f, v in fm.items()},
               micro_item_mean=float(mean(values.values())) if values else None)
    return out


# --------------------------------------------------------------------------- A1: final-turn policy


def select_p(rows, seeds, label):
    """{(item, condition, seed): row} of block-P primary-arm rows at the given seeds, plus exclusion counts."""
    cells, excluded = {}, Counter()
    for row in rows:
        if row["block"] != "P":
            excluded["block_not_P"] += 1
        elif row["condition"] not in P_VARIANT or P_VARIANT[row["condition"]] != row["variant"]:
            excluded["condition_variant_not_primary"] += 1
        elif row["seed"] not in seeds:
            excluded["seed_not_selected"] += 1
        else:
            key = (row["item_id"], row["condition"], row["seed"])
            if key in cells:
                raise Refusal(f"{label}: two rows for {key}")
            cells[key] = row
    return cells, dict(sorted(excluded.items()))


def cell_census(cells_by_policy, items):
    """Per policy and condition: completed, present but not completed, absent (relative to the union of items)."""
    out = {}
    for policy, cells in cells_by_policy.items():
        out[policy] = {}
        for cond in CONDITIONS:
            states = Counter()
            for item in items:
                row = cells.get((item, cond, FINALTURN_SEED))
                states["absent" if row is None else ("completed" if row["cell_state"] == "completed"
                                                     else "not_completed:" + str(row["cell_state"]))] += 1
            out[policy][cond] = dict(sorted(states.items()))
    return out


def pairing_census(values, a, b, items):
    """Cells (item, condition) with a value in both runs, only in a, only in b, in neither."""
    out = {}
    for cond in CONDITIONS:
        c = Counter()
        for item in items:
            ia, ib = values[a].get((item, cond)) is not None, values[b].get((item, cond)) is not None
            c["paired" if ia and ib else (f"only_{a}" if ia else (f"only_{b}" if ib else "neither"))] += 1
        out[cond] = dict(sorted(c.items()))
    return out


def s4_table(cells, item_family):
    """Per condition: submission, recovery given submission, submitted sites, precision bounds (seed 11)."""
    table = {}
    for cond in CONDITIONS:
        rows = [r for (item, c, seed), r in sorted(cells.items()) if c == cond and r["cell_state"] == "completed"]
        sub = [r for r in rows if r["exit_status"] == "Submitted"]
        entry = {"completed_episodes": len(rows), "submitted_episodes": len(sub),
                 "exits": dict(sorted(Counter(r["exit_status"] for r in rows).items()))}
        metrics = {
            "submission_rate": (rows, lambda r: Fraction(int(r["exit_status"] == "Submitted"))),
            "recovery_given_submission": (sub, lambda r: Fraction(int(r["site_recovered"]))),
        }
        for field, name in (("n_submitted", "submitted_sites"), ("n_unique_sites", "unique_submitted_sites"),
                            ("precision_lower", "precision_lower"), ("precision_upper", "precision_upper")):
            if rows and all(field not in r for r in rows):
                entry[name] = {"status": "not_available", "reason": f"field {field} is absent from the score rows"}
                continue
            metrics[name] = ([r for r in sub if r.get(field) is not None], lambda r, f=field: Fraction(r[f]))
        for name, (eligible, value) in metrics.items():
            item_values = defaultdict(list)
            for r in eligible:
                item_values[r["item_id"]].append(value(r))
            means = {i: mean(v) for i, v in item_values.items()}
            entry[name] = {"micro": micro(value(r) for r in eligible),
                           "family_weighted": family_weighted(means, item_family),
                           "eligible": "submitted episodes with a value" if name not in ("submission_rate",)
                           else "completed episodes"}
        table[cond] = entry
    return table


def run_policy_record(run_dir):
    data = json.loads((Path(run_dir) / "run.json").read_text(encoding="utf-8"))
    config = data.get("episode_config")
    if not isinstance(config, dict):
        raise Refusal(f"{run_dir}/run.json has no episode_config")
    return config.get("final_turn_policy", "none"), sha256(Path(run_dir) / "run.json")


def timeline_rows_from_run(run_dir, items_path, score_rows, workers):
    """scripts/expA_timeline.py per-episode rows built with that script's own functions (rules unchanged)."""
    import expA_timeline as T  # noqa: WPS433  (imported only when run records are read)
    items = {}
    for record in load_jsonl(items_path):
        info = T.item_info(record)
        items[info["item_id"]] = info
    tasks = [r for r in score_rows if r["cell_state"] == "completed" and r.get("result_path")]
    if workers > 1:
        from multiprocessing import Pool
        with Pool(workers, initializer=T._init_worker, initargs=(items, str(run_dir))) as pool:
            results = pool.map(T.process_cell, tasks, chunksize=16)
    else:
        T._init_worker(items, str(run_dir))
        results = [T.process_cell(task) for task in tasks]
    return [row for row, _audit in results]


def open_at_response_8(trow):
    """The harness prepared an 8th request: context_accounting for call_index 7 exists, or 8 responses were made."""
    return max(len(trow.get("prompt_tokens_by_response") or []), trow.get("n_responses") or 0) >= BUDGET


def strict_visible_before_8(trow):
    first = (trow.get("first_cstar_visible_response") or {}).get("strict")
    return first is not None and first <= BUDGET - 1


def check_timeline(trows, cells, label):
    """Bind timeline rows to the seed-11 block-P score rows by cell_id; rows that disagree are left out."""
    by_cell = {r["cell_id"]: r for (_i, _c, _s), r in cells.items() if r["cell_state"] == "completed"}
    usable, problems = {}, Counter()
    for t in trows:
        if t.get("cell_id") not in by_cell:
            problems["timeline_row_not_a_selected_completed_cell"] += 1
            continue
        s = by_cell[t["cell_id"]]
        if t.get("problems"):
            problems["timeline_row_has_problems"] += 1
            continue
        if t.get("exit_status") != s["exit_status"] or int(t.get("site_recovered") or 0) != int(s["site_recovered"]):
            problems["timeline_row_disagrees_with_score_row"] += 1
            continue
        usable[(s["item_id"], s["condition"])] = t
    missing = len(set(by_cell) - {t.get("cell_id") for t in trows})
    if missing:
        problems["completed_cells_without_timeline_row"] = missing
    return usable, dict(sorted(problems.items()))


def mechanism_subgroup(timelines, cells_by_policy, item_family):
    out = {"definition": ("episodes still open at response 8 (the harness prepared an 8th request: "
                          "context_accounting for call_index 7, or 8 responses) in which C* was displayed before "
                          "response 8 under the STRICT path:line rule (first strict reveal index <= 7); "
                          "reports/expA-main-20261101b/timeline/RULES.md"),
           "status": "DESCRIPTIVE; exposure establishes availability, not recognition", "by_policy": {}}
    members = {}
    for policy in POLICIES:
        tl = timelines.get(policy)
        if tl is None:
            continue
        sub = {key: t for key, t in tl.items() if open_at_response_8(t) and strict_visible_before_8(t)}
        members[policy] = sub
        entry = {"open_at_8": sum(1 for t in tl.values() if open_at_response_8(t)), "subgroup_episodes": len(sub),
                 "by_condition": {}}
        for cond in CONDITIONS + ("C0+C1", "all"):
            conds = CONDITIONS if cond == "all" else (("C0", "C1") if cond == "C0+C1" else (cond,))
            vals = {k: recovery(cells_by_policy[policy][(k[0], k[1], FINALTURN_SEED)])
                    for k in sub if k[1] in conds}
            e = {"micro_recovery": micro(vals.values())}
            if len(conds) == 1:
                e["family_weighted_recovery"] = family_weighted({k[0]: v for k, v in vals.items()}, item_family)
            entry["by_condition"][cond] = e
        out["by_policy"][policy] = entry
    if len(members) == len(POLICIES):
        common = set.intersection(*(set(m) for m in members.values()))
        out["in_subgroup_under_all_three_policies"] = {
            "cells": len(common),
            "micro_recovery": {p: micro(recovery(cells_by_policy[p][(k[0], k[1], FINALTURN_SEED)])
                                        for k in sorted(common)) for p in POLICIES}}
    return out


def final_response_kinds(timelines):
    out = {}
    for policy, tl in sorted(timelines.items()):
        out[policy] = {}
        for cond in CONDITIONS:
            reached = [t for (i, c), t in tl.items() if c == cond and (t.get("n_responses") or 0) >= BUDGET]
            out[policy][cond] = {"episodes_with_8_responses": len(reached),
                                 "last_action_kind": dict(sorted(Counter(str(t.get("last_action_kind"))
                                                                         for t in reached).items()))}
    return out


def finalturn(args):
    paths = {"neutral": args.scores_neutral, "deadline": args.scores_deadline, "none": args.scores_none}
    inputs = {f"scores_{p}": input_record(paths[p]) for p in POLICIES}
    digests = Counter(v["sha256"] for v in inputs.values())
    warnings = [f"two or more score inputs are byte-identical ({d[:12]}...)" for d, n in digests.items() if n > 1]
    raw = {p: load_scores(paths[p]) for p in POLICIES}
    cells_by_policy, excluded = {}, {}
    for p in POLICIES:
        cells_by_policy[p], excluded[p] = select_p(raw[p], {FINALTURN_SEED}, f"scores_{p}")
    item_family = merge_families((f"scores_{p}", {k[0]: r["family"] for k, r in cells_by_policy[p].items()})
                                 for p in POLICIES)
    items = sorted(item_family)
    values = {p: {(k[0], k[1]): recovery(r) for k, r in cells_by_policy[p].items()} for p in POLICIES}

    def both_c0_c1(treat, ctrl):
        return lambda g: ((g[(treat, "C0")] - g[(ctrl, "C0")]) + (g[(treat, "C1")] - g[(ctrl, "C1")])) / 2

    def list_effect_change(treat, ctrl):
        return lambda g: (g[(treat, "C1")] - g[(treat, "C0")]) - (g[(ctrl, "C1")] - g[(ctrl, "C0")])

    def needed(*policies):
        return tuple((p, c) for p in policies for c in ("C0", "C1"))

    primary = contrast(both_c0_c1("deadline", "neutral"), needed("deadline", "neutral"), values, items, item_family,
                       test=True)
    p = primary.get("p_two_sided")
    primary.update(
        contrast="deadline - neutral, per item averaged with equal weight over C0 and C1 (seed 11, block P)",
        outcome="site_recovered (unconditional; a completed non-submitted episode scores 0)",
        unit="item mean over {C0, C1} of the paired difference -> family mean -> equal-weight mean over families",
        alpha=ALPHA, test="two-sided one-sample t over family means (df = F - 1)",
        decision=("not_estimable" if p is None else
                  ("reject H0 (deadline = neutral) at alpha 0.05" if p <= ALPHA else
                   "do not reject H0 (deadline = neutral) at alpha 0.05")),
        reject_at_alpha=None if p is None else p <= ALPHA,
        item_rule="an item enters only when all four cells (deadline and neutral, C0 and C1) are completed")
    components = {cond: contrast(lambda g, c=cond: g[("deadline", c)] - g[("neutral", c)],
                                 (("deadline", cond), ("neutral", cond)), values, items, item_family)
                  for cond in ("C0", "C1")}

    secondary = {
        "status": "SECONDARY: nominal 95% family-clustered t intervals (df = F - 1); no p-values, no significance "
                  "claims",
        "S1_deadline_minus_none": contrast(both_c0_c1("deadline", "none"), needed("deadline", "none"), values,
                                           items, item_family),
        "S2_list_effect_change_vs_neutral": contrast(list_effect_change("deadline", "neutral"),
                                                     needed("deadline", "neutral"), values, items, item_family),
        "S2_list_effect_change_vs_none": contrast(list_effect_change("deadline", "none"), needed("deadline", "none"),
                                                  values, items, item_family),
        "S2_list_effect_by_policy": {pol: contrast(lambda g, q=pol: g[(q, "C1")] - g[(q, "C0")], needed(pol),
                                                   values, items, item_family) for pol in POLICIES},
        "S3": NOT_RECOMPUTED_S3,
        "S4_by_policy": {pol: s4_table(cells_by_policy[pol], item_family) for pol in POLICIES},
    }
    for key in ("S1_deadline_minus_none", "S2_list_effect_change_vs_neutral", "S2_list_effect_change_vs_none"):
        secondary[key]["definition"] = {
            "S1_deadline_minus_none": "per item mean over {C0, C1} of [recovery(deadline) - recovery(none)]",
            "S2_list_effect_change_vs_neutral": "per item [C1 - C0](deadline) - [C1 - C0](neutral)",
            "S2_list_effect_change_vs_none": "per item [C1 - C0](deadline) - [C1 - C0](none)"}[key]

    timelines, timeline_info = {}, {}
    for pol in POLICIES:
        tpath, rdir = getattr(args, f"timeline_{pol}"), getattr(args, f"run_dir_{pol}")
        if tpath and rdir:
            raise Refusal(f"give --timeline-{pol} or --run-dir-{pol}, not both")
        if tpath:
            trows = load_jsonl(tpath)
            inputs[f"timeline_{pol}"] = input_record(tpath)
            source = "episodes.jsonl of scripts/expA_timeline.py"
        elif rdir:
            if not args.items:
                raise Refusal("--run-dir-* needs --items (the items.jsonl the run was dispatched with)")
            found, run_json_sha = run_policy_record(rdir)
            if found != pol:
                raise Refusal(f"--run-dir-{pol}: run.json episode_config.final_turn_policy is {found!r}")
            inputs[f"run_json_{pol}"] = {"path": str(Path(rdir) / "run.json"), "sha256": run_json_sha,
                                         "final_turn_policy": found}
            inputs["items"] = input_record(args.items)
            trows = timeline_rows_from_run(rdir, args.items, list(cells_by_policy[pol].values()), args.workers)
            source = "run records, read with scripts/expA_timeline.py (episode_timeline, process_cell)"
        else:
            continue
        timelines[pol], problems = check_timeline(trows, cells_by_policy[pol], pol)
        timeline_info[pol] = {"source": source, "usable_episodes": len(timelines[pol]), "problems": problems}

    if timelines:
        mechanism = mechanism_subgroup(timelines, cells_by_policy, item_family)
        mechanism["timeline_inputs"] = timeline_info
        missing = [pol for pol in POLICIES if pol not in timelines]
        if missing:
            mechanism["not_available_for"] = missing
        secondary["S4_final_response_kind"] = {
            "status": "kind of the final response among episodes that made 8 responses (scripts/expA_timeline.py "
                      "last_action_kind); the number of tool calls per response is not in the timeline rows "
                      "(in the main run every parsed response held exactly one tool call)",
            "by_policy": final_response_kinds(timelines)}
    else:
        mechanism = {"status": "not_available",
                     "reason": "needs per-episode timelines: give --timeline-<policy> (scripts/expA_timeline.py "
                               "episodes.jsonl) or --run-dir-<policy> with --items; score rows carry no "
                               "per-response visibility"}
        secondary["S4_final_response_kind"] = {"status": "not_available",
                                               "reason": "needs per-episode timelines (see the mechanism subgroup)"}

    return {
        "schema": SCHEMA, "label": LABEL, "subcommand": "finalturn", "protocol": "protocols/EXPA_FINALTURN.md",
        "status": "PRESPECIFIED analysis code, committed before any A1 outcome existed",
        "inputs": inputs, "warnings": warnings, "seed": FINALTURN_SEED,
        "pooled_with_main_run": False,
        "population": {"items": len(items), "families": len(set(item_family.values())),
                       "items_by_family": dict(sorted(Counter(item_family.values()).items())),
                       "rows_excluded": excluded, "cells": cell_census(cells_by_policy, items),
                       "pairing_deadline_vs_neutral": pairing_census(values, "deadline", "neutral", items),
                       "pairing_deadline_vs_none": pairing_census(values, "deadline", "none", items)},
        "primary": primary, "primary_components_by_condition": components,
        "secondary": secondary, "mechanism_subgroup": mechanism,
    }


# --------------------------------------------------------------------------- A2: hard panel


def primary_seeds(rows):
    """Longest prefix of (11, 23, 47) whose block-P primary-arm cells are all terminal (the frozen seed rule)."""
    status = {}
    for seed in HARD_SEEDS:
        sel = [r for r in rows if r["block"] == "P" and r["seed"] == seed
               and P_VARIANT.get(r["condition"]) == r["variant"]]
        states = Counter(r["cell_state"] for r in sel)
        status[seed] = {"cells": len(sel), "cell_states": dict(sorted(states.items())),
                        "terminal": bool(sel) and all(r["cell_state"] in TERMINAL_STATES for r in sel)}
    chosen = []
    for seed in HARD_SEEDS:
        if not status[seed]["terminal"]:
            break
        chosen.append(seed)
    return tuple(chosen), {str(k): v for k, v in status.items()}


def item_means(cells, value=recovery):
    """{(item, condition): mean over completed seeds} from {(item, condition, seed): row}."""
    acc = defaultdict(list)
    for (item, cond, _seed), row in cells.items():
        v = value(row)
        if v is not None:
            acc[(item, cond)].append(v)
    return {k: mean(v) for k, v in acc.items()}


def hypothesis(means, treat, ctrl, item_family, *, test):
    items = sorted({i for i, _c in means})
    diffs = {i: means[(i, treat)] - means[(i, ctrl)] for i in items if (i, treat) in means and (i, ctrl) in means}
    fm = family_means(diffs, item_family)
    out = summarize(fm.values(), test=test)
    out.update(contrast=f"{treat} - {ctrl}", n_items=len(diffs), n_families=len(fm),
               items_without_complete_pair=len(items) - len(diffs),
               family_values={f: float(v) for f, v in fm.items()},
               treatment_rate=float(mean(family_means({i: means[(i, treat)] for i in diffs}, item_family).values()))
               if diffs else None,
               control_rate=float(mean(family_means({i: means[(i, ctrl)] for i in diffs}, item_family).values()))
               if diffs else None)
    return out, diffs


def rates_by_condition(means, conditions, item_family):
    out = {}
    for cond in conditions:
        vals = {i: v for (i, c), v in means.items() if c == cond}
        out[cond] = {"micro_item_mean": float(mean(vals.values())) if vals else None,
                     "family_weighted": family_weighted(vals, item_family)}
    return out


def select_svc(rows):
    cells = {}
    for r in rows:
        if r["block"] == "S-VC" and r["variant"] == SVC_VARIANT and r["condition"] in SVC_CONDITIONS \
                and r["seed"] == SVC_SEED:
            cells[(r["item_id"], r["condition"], r["seed"])] = r
    return cells


def frozen_crosscheck(analysis_path, hyps):
    analysis = json.loads(Path(analysis_path).read_text(encoding="utf-8"))
    rows = {r["id"]: r for r in analysis["primary"]["hypotheses"]}
    out = {}
    for hid, _t, _c in HYPOTHESES:
        frozen = rows[hid]
        mine = hyps[hid]
        diff = None if mine["estimate"] is None or frozen.get("estimate") is None \
            else abs(mine["estimate"] - frozen["estimate"])
        out[hid] = {"recomputed": mine["estimate"], "frozen": frozen.get("estimate"), "abs_diff": diff,
                    "match": diff is not None and diff <= 1e-12}
        if hid == "H3":
            fp, mp = frozen.get("p_primary"), mine.get("p_two_sided")
            out[hid].update(frozen_p_primary=fp, recomputed_p=mp,
                            p_match=fp is not None and mp is not None and abs(fp - mp) <= 1e-9 * max(1.0, fp))
    bad = [h for h, v in out.items() if not v["match"] or v.get("p_match") is False]
    if bad:
        raise Refusal(f"recomputed hypotheses disagree with the frozen ANALYSIS.json: {bad}: {out}")
    return out


def within_target(hard_rows, main_rows, items_hard, items_main, hard_seeds, main_seeds):
    """E-3: targets shared with the main run whose main-run C* differs and is a direct call (descriptive)."""
    main_cstar = {}
    for r in main_rows:
        if isinstance(r.get("omitted"), str):
            main_cstar.setdefault(r["item_id"], r["omitted"])
    if items_main is not None:
        main_cstar = {i: rec["omitted"]["identity"] for i, rec in items_main.items()}
    main_ids = set(items_main) if items_main is not None else {r["item_id"] for r in main_rows}
    shared = sorted(set(items_hard) & main_ids)
    same = [i for i in shared if main_cstar.get(i) == items_hard[i]["omitted"]["identity"]]
    different = [i for i in shared if i not in same]
    out = {"join": "(run, item_id): item_id is identical for shared targets (E-3); each run's rows stay keyed by run",
           "shared_targets": len(shared), "same_cstar": len(same), "different_cstar": len(different)}
    if items_main is None:
        out.update(status="not_available",
                   reason="the main-run C* stratum (direct call or not) is in the main items.jsonl "
                          "(omitted.stratum), not in the score rows; give --items-main")
        return out
    strata = Counter(items_main[i]["omitted"].get("stratum") for i in different)
    targets = [i for i in different if items_main[i]["omitted"].get("stratum") == "direct"]
    out.update(different_cstar_main_strata=dict(sorted((str(k), v) for k, v in strata.items())),
               targets=len(targets), expected_targets=EXPECTED_WITHIN_TARGET,
               targets_match_expected=len(targets) == EXPECTED_WITHIN_TARGET,
               status="DESCRIPTIVE (E-3): hard C* versus the main run's direct C* on the same target")
    hard_cells, _ = select_p(hard_rows, set(hard_seeds), "scores")
    main_cells, _ = select_p(main_rows, set(main_seeds), "main_scores")
    run_means = {"hard": item_means(hard_cells), "main": item_means(main_cells)}
    family = {i: items_hard[i]["family"] for i in targets}
    per_target, by_cond = {}, {}
    for i in targets:
        per_target[i] = {"family": family[i], "hard_cstar": items_hard[i]["omitted"]["identity"],
                         "main_cstar": items_main[i]["omitted"]["identity"],
                         "hard_cstar_stratum": items_hard[i]["omitted"].get("stratum"), "recovery": {}}
        for cond in CONDITIONS:
            h, m = run_means["hard"].get((i, cond)), run_means["main"].get((i, cond))
            per_target[i]["recovery"][cond] = {"hard": None if h is None else float(h),
                                               "main": None if m is None else float(m),
                                               "hard_minus_main": None if h is None or m is None else float(h - m)}
    for cond in CONDITIONS:
        diffs = {i: run_means["hard"][(i, cond)] - run_means["main"][(i, cond)] for i in targets
                 if (i, cond) in run_means["hard"] and (i, cond) in run_means["main"]}
        by_cond[cond] = {"hard_minus_main_micro": micro(diffs.values()),
                         "hard_minus_main_family_weighted": family_weighted(diffs, family)}
    h3 = {}
    for run in ("hard", "main"):
        d = {i: run_means[run][(i, "C1")] - run_means[run][(i, "C0")] for i in targets
             if (i, "C1") in run_means[run] and (i, "C0") in run_means[run]}
        h3[run] = {"micro": micro(d.values()), "family_weighted": family_weighted(d, family)}
    out.update(by_condition=by_cond, h3_on_these_targets=h3, per_target=per_target,
               seeds={"hard": list(hard_seeds), "main": list(main_seeds)})
    return out


def hard(args):
    inputs = {"scores": input_record(args.scores), "main_scores": input_record(args.main_scores),
              "items_hard": input_record(args.items_hard)}
    rows = load_scores(args.scores)
    main_rows = load_scores(args.main_scores)
    items_hard = {r["item_id"]: r for r in load_jsonl(args.items_hard)}
    items_main = None
    if args.items_main:
        inputs["items_main"] = input_record(args.items_main)
        items_main = {r["item_id"]: r for r in load_jsonl(args.items_main)}
    problems = Counter()
    for r in rows:
        if r["item_id"] not in items_hard:
            problems["score_row_item_not_in_items_hard"] += 1
        elif r["family"] != items_hard[r["item_id"]]["family"]:
            problems["score_row_family_differs_from_items_hard"] += 1
        elif isinstance(r.get("omitted"), str) and r["omitted"] != items_hard[r["item_id"]]["omitted"]["identity"]:
            problems["score_row_omitted_differs_from_items_hard"] += 1
    if problems:
        raise Refusal(f"--scores and --items-hard disagree: {dict(problems)}")
    item_family = {i: rec["family"] for i, rec in items_hard.items()}
    cstar = {i: rec["omitted"]["identity"] for i, rec in items_hard.items()}

    seeds, seed_status = primary_seeds(rows)
    cells, excluded = select_p(rows, set(seeds), "scores")
    means = item_means(cells)
    items_with_rows = {i for (i, _c, _s) in cells}
    hyps, diffs = {}, {}
    for hid, treat, ctrl in HYPOTHESES:
        hyps[hid], diffs[hid] = hypothesis(means, treat, ctrl, item_family, test=(hid == "H3"))
    h3 = hyps["H3"]
    p = h3.get("p_two_sided")
    h3.update(role="PRIMARY (single test, alpha 0.05; EXPA_HARD.md section 4)",
              decision_rule="D-4: unadjusted two-sided p <= 0.05 (t over family means, df = F - 1, frozen "
                            "degenerate-case rules); no Holm",
              decision=("not_estimable" if p is None else
                        ("reject H0 (C1 = C0) at alpha 0.05" if p <= ALPHA else
                         "do not reject H0 (C1 = C0) at alpha 0.05")),
              reject_at_alpha=None if p is None else p <= ALPHA)
    for hid in ("H1", "H2"):
        hyps[hid]["role"] = "SECONDARY (D-4): estimate and nominal 95% interval only; no p-value, no Holm"
    hyps["H2"]["note"] = "the lexical block contains C* in no panel item: H2 tests extra lexical evidence about other callers only"

    d3 = diffs["H3"]
    n_by_cstar = Counter(cstar[i] for i in d3)
    sensitivity = {
        "status": "SECONDARY sensitivity weightings of H3: nominal 95% family-clustered intervals, no p-values",
        "item_weighted": dict(weighted_cluster(d3, {i: Fraction(1) for i in d3}, item_family),
                              definition="every contributing item has weight 1 (estimate = the micro mean of the "
                                         "item differences)"),
        "caller_balanced": dict(weighted_cluster(d3, {i: Fraction(1, n_by_cstar[cstar[i]]) for i in d3},
                                                 item_family),
                                definition="every distinct C* (items.jsonl omitted.identity) among the contributing "
                                           "items has total weight 1, split equally over its items (weight 1 / "
                                           "n items with that C*); estimate = the mean over distinct C* of the "
                                           "mean item difference of that C*'s items",
                                distinct_cstar=len(n_by_cstar),
                                items_per_cstar=dict(sorted(Counter(n_by_cstar.values()).items()))),
        "interval_method": "weighted mean linearised into one pseudo-value per family, u_f = est + F (S_f - est W_f) "
                           "/ W; t_statistics over u_f (df = F - 1); weights 1 / n_f reproduce the primary exactly",
    }

    svc_cells = select_svc(rows)
    svc_means = item_means(svc_cells)
    svc = {"status": "SECONDARY positive control (complete answers that list C*): recovery, seed 11",
           "cells": {c: dict(sorted(Counter(r["cell_state"] for (i, cc, s), r in svc_cells.items() if cc == c)
                                    .items())) for c in SVC_CONDITIONS},
           "recovery": rates_by_condition(svc_means, SVC_CONDITIONS, item_family)}
    if not svc_cells:
        svc["status"] = "not_available: no S-VC vcomplete rows at seed 11 in --scores"
    submission = {"status": "SECONDARY descriptive: share of completed episodes that submitted (item mean over the "
                            "seeds -> family mean -> equal-weight mean)",
                  "P": rates_by_condition(item_means(cells, submitted), CONDITIONS, item_family),
                  "S-VC": rates_by_condition(item_means(svc_cells, submitted), SVC_CONDITIONS, item_family)}

    main_seeds, _ = primary_seeds(main_rows)
    within = within_target(rows, main_rows, items_hard, items_main, seeds, main_seeds)

    result = {
        "schema": SCHEMA, "label": LABEL, "subcommand": "hard",
        "protocol": "protocols/EXPA_HARD.md section 4; protocols/EXPA_DEVIATIONS.md D-4, E-3",
        "status": "PRESPECIFIED analysis code, committed before any A2 outcome existed",
        "inputs": inputs, "pooled_with_main_run": False,
        "population": {"items_hard": len(items_hard), "items_with_P_rows": len(items_with_rows),
                       "families": len(set(item_family[i] for i in items_with_rows)),
                       "distinct_cstar": len(set(cstar[i] for i in items_with_rows)),
                       "seed_status": seed_status, "primary_seeds": list(seeds),
                       "rows_excluded_from_P": excluded,
                       "cells_by_condition": {c: dict(sorted(Counter(r["cell_state"] for (i, cc, s), r in
                                                                     cells.items() if cc == c).items()))
                                              for c in CONDITIONS}},
        "unit": "item mean over the primary seeds (completed cells) -> paired item difference -> family mean -> "
                "equal-weight mean over families",
        "hypotheses": hyps, "h3_sensitivity": sensitivity, "rates_P": rates_by_condition(means, CONDITIONS,
                                                                                       item_family),
        "s_vc_positive_control": svc, "submission_rates": submission, "within_target": within,
    }
    if args.frozen_analysis:
        result["inputs"]["frozen_analysis"] = input_record(args.frozen_analysis)
        result["frozen_crosscheck"] = frozen_crosscheck(args.frozen_analysis, hyps)
    return result


# --------------------------------------------------------------------------- output


def g(x, digits=3):
    return "n/a" if x is None else f"{x:+.{digits}f}" if isinstance(x, float) else str(x)


def iv(s):
    return f"{g(s.get('estimate'))} [{g(s.get('ci_low'))}, {g(s.get('ci_high'))}]"


def render_md(res):
    lines = [f"# Exp A follow-up analysis: {res['subcommand']}", "", f"{res['label']}. {res['status']}.",
             f"Protocol: {res['protocol']}. Never pooled with the main run.", "", "Inputs (sha256):", ""]
    lines += [f"- {k}: `{v['sha256']}`" for k, v in sorted(res["inputs"].items())]
    lines.append("")
    if res["subcommand"] == "finalturn":
        p = res["primary"]
        lines += ["## Primary (single test, alpha 0.05)", "",
                  "| Contrast | Estimate [95% CI] | t | df | p | F | items | Decision |", "|---|---|---|---|---|---|---|---|",
                  f"| deadline - neutral, mean of C0, C1 | {iv(p)} | {g(p.get('t'))} | {p.get('df')} | "
                  f"{'n/a' if p.get('p_two_sided') is None else format(p['p_two_sided'], '.4g')} | {p['n_families']} | "
                  f"{p['n_items']} | {p['decision']} |", "",
                  "## Secondary (nominal 95% intervals, no significance claims)", "",
                  "| Contrast | Estimate [95% CI] | F | items |", "|---|---|---|---|"]
        s = res["secondary"]
        for key in ("S1_deadline_minus_none", "S2_list_effect_change_vs_neutral", "S2_list_effect_change_vs_none"):
            lines.append(f"| {key} | {iv(s[key])} | {s[key]['n_families']} | {s[key]['n_items']} |")
        for pol, v in s["S2_list_effect_by_policy"].items():
            lines.append(f"| C1 - C0 under {pol} | {iv(v)} | {v['n_families']} | {v['n_items']} |")
        lines += ["", f"S3: {s['S3']}.", "", "S4 (family-weighted, by policy and condition):", "",
                  "| Policy | Cond | Submission | Recovery given submission | Submitted sites | Precision lo | "
                  "Precision hi |", "|---|---|---|---|---|---|---|"]
        for pol, table in s["S4_by_policy"].items():
            for cond, e in table.items():
                cells = [iv(e[k]["family_weighted"]) if "family_weighted" in e[k] else "n/a"
                         for k in ("submission_rate", "recovery_given_submission", "submitted_sites",
                                   "precision_lower", "precision_upper")]
                lines.append(f"| {pol} | {cond} | " + " | ".join(cells) + " |")
        m = res["mechanism_subgroup"]
        lines += ["", "## Mechanism subgroup (descriptive)", ""]
        if m.get("status") == "not_available":
            lines.append(f"Not available: {m['reason']}.")
        else:
            lines += [m["definition"] + ".", "", "| Policy | Subgroup episodes | Recovery C0+C1 (micro) | "
                      "Recovery all (micro) |", "|---|---|---|---|"]
            for pol, e in m["by_policy"].items():
                lines.append(f"| {pol} | {e['subgroup_episodes']} | "
                             f"{g(e['by_condition']['C0+C1']['micro_recovery']['mean'])} | "
                             f"{g(e['by_condition']['all']['micro_recovery']['mean'])} |")
    else:
        h = res["hypotheses"]
        lines += [f"Primary seeds: {res['population']['primary_seeds']}; items with block-P rows "
                  f"{res['population']['items_with_P_rows']}; families {res['population']['families']}.", "",
                  "| Hypothesis | Estimate [95% CI] | t | df | p (unadjusted) | F | items | Decision |",
                  "|---|---|---|---|---|---|---|---|"]
        h3 = h["H3"]
        lines.append(f"| H3 = C1 - C0 (primary) | {iv(h3)} | {g(h3.get('t'))} | {h3.get('df')} | "
                     f"{'n/a' if h3.get('p_two_sided') is None else format(h3['p_two_sided'], '.4g')} | "
                     f"{h3['n_families']} | {h3['n_items']} | {h3['decision']} |")
        for hid in ("H1", "H2"):
            lines.append(f"| {hid} = {h[hid]['contrast']} | {iv(h[hid])} | | {h[hid].get('df')} | (not reported, "
                         f"D-4) | {h[hid]['n_families']} | {h[hid]['n_items']} | interval only |")
        sens = res["h3_sensitivity"]
        lines += ["", "| H3 weighting | Estimate [95% CI] |", "|---|---|",
                  f"| item-weighted | {iv(sens['item_weighted'])} |",
                  f"| caller-balanced ({sens['caller_balanced']['distinct_cstar']} distinct C*) | "
                  f"{iv(sens['caller_balanced'])} |", ""]
        svc = res["s_vc_positive_control"]
        lines += ["| S-VC control | Recovery (family-weighted) |", "|---|---|"]
        for cond, e in svc["recovery"].items():
            lines.append(f"| {cond} | {iv(e['family_weighted'])} |")
        w = res["within_target"]
        lines += ["", f"Within-target (E-3): shared targets {w['shared_targets']}, same C* {w['same_cstar']}, "
                      f"different C* {w['different_cstar']}; status: {w.get('status')}"
                      + (f"; targets {w['targets']} (expected {w['expected_targets']})" if "targets" in w else "")
                      + (f"; {w['reason']}" if "reason" in w else "") + "."]
        if "frozen_crosscheck" in res:
            lines += ["", "Cross-check against the frozen ANALYSIS.json: "
                      + ", ".join(f"{k} {'match' if v['match'] else 'MISMATCH'}"
                                  for k, v in sorted(res["frozen_crosscheck"].items())) + "."]
    if res.get("warnings"):
        lines += ["", "Warnings:", ""] + [f"- {w}" for w in res["warnings"]]
    return "\n".join(lines) + "\n"


def write_outputs(result, out_dir, overwrite):
    out = Path(out_dir)
    targets = [out / "RESULTS.json", out / "RESULTS.md"]
    if not overwrite and any(t.exists() for t in targets):
        raise Refusal(f"{out_dir} already holds RESULTS.json or RESULTS.md (use --overwrite)")
    out.mkdir(parents=True, exist_ok=True)
    result["script_sha256"] = sha256(__file__)
    targets[0].write_text(json.dumps(result, indent=1, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    targets[1].write_text(render_md(result), encoding="utf-8")
    return targets


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    ft = sub.add_parser("finalturn", help="A1, protocols/EXPA_FINALTURN.md section 3")
    for pol in POLICIES:
        ft.add_argument(f"--scores-{pol}", required=True)
        ft.add_argument(f"--timeline-{pol}", help="scripts/expA_timeline.py episodes.jsonl of that run")
        ft.add_argument(f"--run-dir-{pol}", help="the run directory (read-only), instead of --timeline-*")
    ft.add_argument("--items", help="items.jsonl the runs were dispatched with (needed with --run-dir-*)")
    ft.add_argument("--workers", type=int, default=1)
    hd = sub.add_parser("hard", help="A2, protocols/EXPA_HARD.md section 4 with D-4 and E-3")
    hd.add_argument("--scores", required=True)
    hd.add_argument("--main-scores", required=True)
    hd.add_argument("--items-hard", required=True)
    hd.add_argument("--items-main", help="main items.jsonl: the main-run C* stratum for the E-3 within-target set")
    hd.add_argument("--frozen-analysis", help="the frozen ANALYSIS.json of --scores, to cross-check H1-H3")
    for p in (ft, hd):
        p.add_argument("--out", required=True)
        p.add_argument("--overwrite", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = finalturn(args) if args.command == "finalturn" else hard(args)
    write_outputs(result, args.out, args.overwrite)
    if args.command == "finalturn":
        p = result["primary"]
        brief = {"primary": {k: p.get(k) for k in ("estimate", "ci_low", "ci_high", "t", "df", "p_two_sided",
                                                    "n_families", "n_items", "decision")}}
    else:
        brief = {h: {k: v.get(k) for k in ("estimate", "ci_low", "ci_high", "p_two_sided", "n_families", "n_items")}
                 for h, v in result["hypotheses"].items()}
        brief["H3"]["decision"] = result["hypotheses"]["H3"]["decision"]
        if "frozen_crosscheck" in result:
            brief["frozen_crosscheck"] = {h: v["match"] for h, v in result["frozen_crosscheck"].items()}
    print(json.dumps(brief, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
