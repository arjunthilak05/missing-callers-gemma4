#!/usr/bin/env python3
"""Exp A follow-up A7 (impact-analysis framing): the prespecified analysis of protocols/EXPA_A7_IMPACT.md section 3.

PRESPECIFIED: written with the protocol, before any A7 cell exists. ANALYSIS RECORD.
Standard library only; exact rational means. It was exercised only on SYNTHETIC fixtures.

    python scripts/expA_a7_analysis.py --scores-impact A7/scores.jsonl --scores-original MAIN/scores.jsonl \\
        [--run-json-impact A7/run.json] --out DIR

Inputs are the scorer's rows (scripts/expA_score.py) of two runs on the same 750 main items:
    impact    the A7 run (a7impact-20261107a, --task-framing impact), which dispatches only the three A7 arms; every
              other row must be not_dispatched, or the input is refused;
    original  the main run (the original task framing), whose rows of the same three arms are the reference; its
              other rows are ignored.
The three arms, all at seed 11:
    none        block P, condition C0, variant none        (no list)
    incomplete  block P, condition C1, variant omit_node   (incomplete list)
    complete    block S-VC, condition C1, variant vcomplete (complete list)
With --run-json-impact, that run.json must record the impact framing (episode_config.task_framing impact, the
protocol's template text and its sha256, the prereg budget, no final-turn notice, ledger or review), and every launch
record next to it (launches/*.json) must select blocks P,S-VC, seed 11 and conditions C0,C1.

Outcome: site_recovered of completed cells (a completed episode that did not submit scores 0). Cells missing for
infrastructure (or not terminal) are excluded and counted. Per framing, item and arm the cell's value; an item enters
a framing's contrasts only when all three arms are completed in that framing. Item value -> family mean -> equal-weight
mean over families, with a two-sided one-sample t over the family means (df = F - 1;
scripts/expA_reanalyze_independent.t_statistics).
  PRIMARY (two tests, impact framing; Holm across them, alpha 0.05):
      T1 = rec(incomplete) - rec(none)          (an incomplete list beats no list: the H3 analogue)
      T2 = rec(complete) - rec(incomplete)      (the omission cost)
  SECONDARY (nominal 95% t intervals over family means; no p-values, no significance claims):
      T1 and T2 in the original framing (its own items with all three arms);
      framing differences (impact - original), paired by item: T1 and T2 over the items with all three arms in both
      framings; each arm's recovery over the items with that arm completed in both framings;
      per framing and arm, over the framing's items: recovery, submission (exit Submitted), completeness claims (the
      share of submitted completed cells claiming complete "yes"), responses used (cost.calls), precision bounds
      (precision_lower, precision_upper over submitted completed cells); exits (all completed cells of the arm) and
      cell states; the number of libraries (families) whose family mean of T1, and of T2, is positive, zero or
      negative.
Reading rules (fixed by the protocol): positive T1 and T2 mean both effects transfer to the impact framing; a framing
difference near zero means the framing did not matter. The two framings are different runs on different dates: the
framing comparison is not a randomized within-run contrast.
Outputs (in --out, which must not exist): A7_ANALYSIS.json (every number, the inputs with their paths as given and
their sha256, the script sha256; sorted keys, no timestamps) and A7_TABLES.md. Exit codes: 0 written; 2 refused
(nothing written).
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
from expA_reanalyze_independent import holm_adjust, t_statistics  # noqa: E402

SCHEMA = "expA-a7-analysis-v1"
PROTOCOL = "protocols/EXPA_A7_IMPACT.md"
LABEL = ("PRESPECIFIED (protocols/EXPA_A7_IMPACT.md, written before any A7 cell); ANALYSIS RECORD"
         "")
SEED = 11
# Arm -> (block, condition, variant)
ARMS = {"none": ("P", "C0", "none"), "incomplete": ("P", "C1", "omit_node"), "complete": ("S-VC", "C1", "vcomplete")}
CELL_ARM = {cell: arm for arm, cell in ARMS.items()}
FRAMINGS = ("impact", "original")
CELL_STATES = frozenset({"completed", "missing_infrastructure", "infra_pending", "not_dispatched"})
IMPACT_TEMPLATE = (
    "Task: the signature of the function `{qualname}` (defined in `{path}`, line {line}) is about to change. Find "
    "every place in the library source that must be updated: every place where it is invoked, either called directly "
    "or passed as a callable that gets invoked, from inside another function or method. Consider only files under "
    "{package_dirs}. Ignore tests, docs, examples and module-level code.")
PREREG_BUDGET = {"max_model_calls": 8, "context_budget": 16384}
A7_SELECTION = {"blocks": ["P", "S-VC"], "seeds": [SEED], "conditions": ["C0", "C1"]}
ALPHA = 0.05
CONFIDENCE = 0.95
CONTRASTS = {"T1": ("incomplete", "none"), "T2": ("complete", "incomplete")}
DEFINITIONS = {"T1": "rec(incomplete) - rec(none): an incomplete list against no list (the H3 analogue)",
               "T2": "rec(complete) - rec(incomplete): the omission cost"}
READING = ("positive T1 and T2 mean both effects transfer to the impact framing; a framing difference near zero means "
           "the framing did not matter; different runs and dates: the framing comparison is not a randomized "
           "within-run contrast (protocol section 3)")


class Refused(ValueError):
    """The inputs cannot be analysed as the protocol states (nothing is written)."""


def _require(condition, message):
    if not condition:
        raise Refused(message)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _calls(row):
    cost = row.get("cost")
    value = cost.get("calls") if isinstance(cost, dict) else None
    return value if type(value) is int and value >= 0 else None


# --------------------------------------------------------------------------- inputs


def load_rows(path, framing):
    """{arm: {item: row}} of one run's seed-11 rows of the three arms; the impact run must dispatch nothing else."""
    raw = Path(path).read_bytes()
    cells = {arm: {} for arm in ARMS}
    excluded, dispatched_elsewhere, items_sha = Counter(), Counter(), set()
    for number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        where = f"{path}:{number}"
        _require(isinstance(row, dict), f"{where}: not an object")
        arm = CELL_ARM.get((row.get("block"), row.get("condition"), row.get("variant")))
        if arm is None or row.get("seed") != SEED:
            excluded["outside_the_a7_arms"] += 1
            if framing == "impact" and row.get("cell_state") != "not_dispatched":
                dispatched_elsewhere[f"{row.get('block')}|{row.get('variant')}|{row.get('condition')}|"
                                     f"s{row.get('seed')}"] += 1
            continue
        where = f"{where} ({row.get('cell_id')})"
        _require(isinstance(row.get("item_id"), str) and isinstance(row.get("family"), str), f"{where}: item/family")
        _require(row.get("cell_state") in CELL_STATES, f"{where}: unknown cell_state {row.get('cell_state')!r}")
        if row["cell_state"] == "completed":
            _require(row.get("site_recovered") in (0, 1), f"{where}: a completed cell needs site_recovered 0 or 1")
        _require(row["item_id"] not in cells[arm], f"{where}: duplicate row")
        cells[arm][row["item_id"]] = row
        items_sha.add(row.get("items_sha256"))
    _require(any(cells.values()), f"{path}: no seed-11 rows of the three A7 arms")
    _require(len(items_sha) == 1, f"{path}: rows name {len(items_sha)} items files")
    _require(not dispatched_elsewhere, f"{path}: cells outside the three A7 arms were dispatched "
                                       f"({dict(sorted(dispatched_elsewhere.items()))}); an A7 run dispatches only "
                                       "block P C0/C1 and S-VC C1 at seed 11")
    return {"framing": framing, "path": str(path), "sha256": sha256_bytes(raw), "cells": cells,
            "items_sha256": items_sha.pop(), "rows_excluded": dict(sorted(excluded.items()))}


def check_run_json(path):
    """run.json records the impact framing; every launch record next to it selects the A7 cells."""
    path = Path(path)
    raw = path.read_bytes()
    run = json.loads(raw)
    _require(isinstance(run, dict), f"{path}: not an object")
    config = run.get("episode_config")
    _require(isinstance(config, dict), f"{path}: no episode_config")
    _require(config.get("task_framing") == "impact", f"{path}: episode_config.task_framing is "
                                                     f"{config.get('task_framing')!r}, not 'impact'")
    rules = config.get("task_framing_rules")
    _require(isinstance(rules, dict) and rules.get("protocol") == PROTOCOL and rules.get("text") == IMPACT_TEMPLATE
             and rules.get("text_sha256") == sha256_bytes(IMPACT_TEMPLATE.encode("utf-8")),
             f"{path}: episode_config.task_framing_rules is not the protocol's text and sha256")
    _require(all(config.get(key) == value for key, value in PREREG_BUDGET.items()),
             f"{path}: episode_config is not the prereg budget (8 responses, 16384 tokens)")
    _require(not {"final_turn_policy", "evidence_ledger", "submit_review"} & set(config),
             f"{path}: an A7 run has no final-turn notice, evidence ledger or submit review")
    launches = sorted(p for p in (path.parent / "launches").glob("*.json") if not p.name.endswith(".end.json"))
    _require(launches, f"{path.parent / 'launches'}: no launch records to verify the selection (blocks P,S-VC, seed "
                       "11, conditions C0,C1)")
    records = []
    for launch in launches:
        selection = json.loads(launch.read_bytes()).get("selection") or {}
        found = {key: selection.get(key) for key in A7_SELECTION}
        _require(found == A7_SELECTION, f"{launch}: selection {found}, not {A7_SELECTION}")
        records.append({"path": str(launch), "sha256": sha256_bytes(launch.read_bytes())})
    return {"path": str(path), "sha256": sha256_bytes(raw), "run_id": run.get("run_id"), "task_framing": "impact",
            "text_sha256": rules["text_sha256"], "launches": records}


# --------------------------------------------------------------------------- statistics


def summarize(values, *, test):
    """Equal-weight mean over family values with its t summary (p and t only for a test)."""
    values = list(values)
    if not values:
        return {"n_families": 0, "estimate": None, "status": "no_families"}
    st = t_statistics(values, CONFIDENCE)
    est = st["mean_exact"]
    out = {"n_families": len(values), "estimate": float(est), "estimate_exact": f"{est.numerator}/{est.denominator}",
           "ci_low": st.get("lower"), "ci_high": st.get("upper"), "df": st["df"], "status": st["status"]}
    if test:
        out.update(t=st.get("t"), p_two_sided=st.get("p_two_sided"))
    return out


def family_means(per_item, family):
    by_family = defaultdict(list)
    for item, value in per_item.items():
        by_family[family[item]].append(value)
    return {fam: sum(vals, Fraction(0)) / len(vals) for fam, vals in sorted(by_family.items())}


def family_summary(per_item, family, *, test=False):
    means = family_means(per_item, family)
    out = summarize(means.values(), test=test)
    out["n_items"] = len(per_item)
    out["family_means"] = {fam: float(value) for fam, value in means.items()}
    return out


def sign_counts(per_item, family):
    means = family_means(per_item, family)
    return {"positive": sum(1 for v in means.values() if v > 0), "zero": sum(1 for v in means.values() if v == 0),
            "negative": sum(1 for v in means.values() if v < 0), "families": len(means),
            "positive_families": sorted(fam for fam, v in means.items() if v > 0)}


# --------------------------------------------------------------------------- analysis


def completed(run):
    """{arm: {item: row}} of the completed cells."""
    return {arm: {item: row for item, row in rows.items() if row["cell_state"] == "completed"}
            for arm, rows in run["cells"].items()}


def recovery(rows):
    return {item: Fraction(row["site_recovered"]) for item, row in rows.items()}


def contrast_values(rec, items, name):
    treat, ctrl = CONTRASTS[name]
    return {item: rec[treat][item] - rec[ctrl][item] for item in items}


def arm_metrics(rows, items, family):
    """Recovery, submission, claims, responses and precision bounds of one framing's arm over ``items``."""
    rows = {item: rows[item] for item in items}
    submitted = {item: row for item, row in rows.items() if row.get("exit_status") == "Submitted"}
    out = {"recovery": family_summary(recovery(rows), family),
           "submission": family_summary({i: Fraction(int(r.get("exit_status") == "Submitted"))
                                         for i, r in rows.items()}, family),
           "completeness_claim_yes": family_summary({i: Fraction(int(r.get("complete") == "yes"))
                                                     for i, r in submitted.items()}, family),
           "responses_used": family_summary({i: Fraction(_calls(r)) for i, r in rows.items()
                                             if _calls(r) is not None}, family),
           "completed_cells_without_calls": sum(1 for r in rows.values() if _calls(r) is None)}
    for bound in ("precision_lower", "precision_upper"):
        out[bound] = family_summary({i: Fraction(r[bound]) for i, r in submitted.items()
                                     if r.get(bound) is not None}, family)
    out["completeness_claim_eligible"] = "submitted completed cells (claims exist only with a submission)"
    return out


def analyse(impact, original):
    _require(impact["items_sha256"] == original["items_sha256"], "the two runs name different items files")
    family = {}
    for run in (impact, original):
        for rows in run["cells"].values():
            for item, row in rows.items():
                family.setdefault(item, row["family"])
                _require(family[item] == row["family"], f"{item}: families differ between rows")
    runs = {"impact": impact, "original": original}
    done = {name: completed(run) for name, run in runs.items()}
    rec = {name: {arm: recovery(rows) for arm, rows in arms.items()} for name, arms in done.items()}
    items = {name: sorted(set.intersection(*(set(rec[name][arm]) for arm in ARMS))) for name in FRAMINGS}
    population = {
        "items_by_framing": {name: len(items[name]) for name in FRAMINGS},
        "families_by_framing": {name: len({family[i] for i in items[name]}) for name in FRAMINGS},
        "cell_states": {name: {arm: dict(sorted(Counter(r["cell_state"] for r in run["cells"][arm].values()).items()))
                               for arm in ARMS} for name, run in runs.items()},
        "missing_infrastructure_cells": {name: sum(1 for rows in run["cells"].values() for r in rows.values()
                                                   if r["cell_state"] == "missing_infrastructure")
                                         for name, run in runs.items()},
        "rows_excluded": {name: run["rows_excluded"] for name, run in runs.items()},
        "item_rule": "an item enters a framing's contrasts only when all three arms are completed in that framing"}
    primary = {}
    for name in CONTRASTS:
        summary = family_summary(contrast_values(rec["impact"], items["impact"], name), family, test=True)
        summary["definition"] = DEFINITIONS[name]
        primary[name] = summary
    adjusted, order = holm_adjust({name: primary[name].get("p_two_sided") for name in CONTRASTS})
    for name in CONTRASTS:
        p = adjusted[name]
        primary[name].update(p_holm=p, reject_at_alpha=None if p is None else p <= ALPHA, decision=(
            "not_estimable" if p is None else
            (f"reject H0 ({name} = 0) at alpha {ALPHA} (Holm)" if p <= ALPHA else
             f"do not reject H0 ({name} = 0) at alpha {ALPHA} (Holm)")))
    primary_block = {"framing": "impact", "tests": primary, "alpha": ALPHA, "holm_order": order,
                     "test": "two-sided one-sample t over family means (df = F - 1); Holm step-down across T1 and T2",
                     "outcome": "site_recovered (a completed non-submitted episode scores 0)",
                     "unit": "item value -> family mean -> equal-weight mean over families"}
    both = sorted(set(items["impact"]) & set(items["original"]))
    secondary = {
        "status": "SECONDARY: nominal 95% family-clustered t intervals (df = F - 1); no p-values, no significance "
                  "claims",
        "original_framing": {name: dict(family_summary(contrast_values(rec["original"], items["original"], name),
                                                       family), definition=DEFINITIONS[name]) for name in CONTRASTS},
        "framing_difference_contrasts": {
            name: dict(family_summary({i: contrast_values(rec["impact"], [i], name)[i]
                                       - contrast_values(rec["original"], [i], name)[i] for i in both}, family),
                       definition=f"impact - original of {name}, paired by item (items with all three arms in both)")
            for name in CONTRASTS},
        "framing_difference_arms": {
            arm: dict(family_summary({i: rec["impact"][arm][i] - rec["original"][arm][i]
                                      for i in sorted(set(rec["impact"][arm]) & set(rec["original"][arm]))}, family),
                      definition=f"impact - original recovery of the {arm} arm, paired by item (items with the arm "
                                 "completed in both)")
            for arm in ARMS},
        "by_framing": {name: {arm: arm_metrics(done[name][arm], items[name], family) for arm in ARMS}
                       for name in FRAMINGS},
        "exits": {name: {arm: dict(sorted(Counter(r.get("exit_status") for r in done[name][arm].values()).items()))
                         for arm in ARMS} for name in FRAMINGS},
        "libraries_by_sign": {name: {c: sign_counts(contrast_values(rec[name], items[name], c), family)
                                     for c in CONTRASTS} for name in FRAMINGS},
        "items_in_both_framings": len(both)}
    return {"items_sha256": impact["items_sha256"], "population": population, "primary": primary_block,
            "secondary": secondary, "reading": READING}


# --------------------------------------------------------------------------- outputs


def _pp(x):
    return "n/a" if x is None else f"{100 * x:+.1f}"


def _iv(summary):
    if summary.get("estimate") is None:
        return "n/a"
    low, high = summary.get("ci_low"), summary.get("ci_high")
    interval = "" if low is None else f" [{_pp(low)}, {_pp(high)}]"
    return f"{_pp(summary['estimate'])}{interval}"


def _count(summary):
    if summary.get("estimate") is None:
        return "n/a"
    low, high = summary.get("ci_low"), summary.get("ci_high")
    interval = "" if low is None else f" [{low:.2f}, {high:.2f}]"
    return f"{summary['estimate']:.2f}{interval}"


def _num(value, spec):
    return "n/a" if value is None else format(value, spec)


def tables(result):
    a = result["analysis"]
    pop, p, s = a["population"], a["primary"]["tests"], a["secondary"]
    lines = ["# A7 impact-analysis framing: prespecified analysis", "", result["label"], "",
             f"Items with all three arms: impact {pop['items_by_framing']['impact']} "
             f"({pop['families_by_framing']['impact']} families), original {pop['items_by_framing']['original']} "
             f"({pop['families_by_framing']['original']} families); in both {s['items_in_both_framings']}. "
             f"Missing for infrastructure (excluded): {pop['missing_infrastructure_cells']}. Points are percentage "
             "points of site_recovered.", "",
             "## Primary (impact framing; two tests, Holm, alpha 0.05)", "",
             "| Contrast | Estimate [95% t] | t | df | p | p (Holm) | F | items | Decision |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name in CONTRASTS:
        t = p[name]
        lines.append(f"| {name} = {t['definition']} | {_iv(t)} | {_num(t.get('t'), '.3f')} | {t.get('df')} | "
                     f"{_num(t.get('p_two_sided'), '.4g')} | {_num(t.get('p_holm'), '.4g')} | {t['n_families']} | "
                     f"{t['n_items']} | {t['decision']} |")
    lines += ["", "## Secondary (nominal 95% t intervals, no significance claims)", "",
              "| Quantity | Estimate [95% t] | F | items |", "|---|---|---|---|"]
    for name in CONTRASTS:
        q = s["original_framing"][name]
        lines.append(f"| {name} original framing | {_iv(q)} | {q['n_families']} | {q['n_items']} |")
    for name in CONTRASTS:
        q = s["framing_difference_contrasts"][name]
        lines.append(f"| {name} impact - original | {_iv(q)} | {q['n_families']} | {q['n_items']} |")
    for arm in ARMS:
        q = s["framing_difference_arms"][arm]
        lines.append(f"| recovery {arm} impact - original | {_iv(q)} | {q['n_families']} | {q['n_items']} |")
    lines += ["", "Per framing and arm (family-weighted, over the framing's items):", "",
              "| Framing | Arm | Recovery | Submission | Claims complete yes | Precision lo | Precision hi | "
              "Responses used | Exits |", "|---|---|---|---|---|---|---|---|---|"]
    for name in FRAMINGS:
        for arm in ARMS:
            m = s["by_framing"][name][arm]
            exits = ", ".join(f"{k} {v}" for k, v in s["exits"][name][arm].items()) or "none"
            lines.append(f"| {name} | {arm} | {_iv(m['recovery'])} | {_iv(m['submission'])} | "
                         f"{_iv(m['completeness_claim_yes'])} | {_iv(m['precision_lower'])} | "
                         f"{_iv(m['precision_upper'])} | {_count(m['responses_used'])} | {exits} |")
    lines += ["", "| Libraries by sign of the family mean | positive | zero | negative | families |",
              "|---|---|---|---|---|"]
    for name in FRAMINGS:
        for c in CONTRASTS:
            q = s["libraries_by_sign"][name][c]
            lines.append(f"| {c} {name} | {q['positive']} | {q['zero']} | {q['negative']} | {q['families']} |")
    lines += ["", "Reading: " + a["reading"] + ".", ""]
    return "\n".join(lines) + "\n"


def run(args):
    impact = load_rows(args.scores_impact, "impact")
    original = load_rows(args.scores_original, "original")
    _require(impact["sha256"] != original["sha256"], "the two score inputs are byte-identical")
    inputs = {"scores_impact": {"path": impact["path"], "sha256": impact["sha256"]},
              "scores_original": {"path": original["path"], "sha256": original["sha256"]}}
    if args.run_json_impact:
        inputs["run_json_impact"] = check_run_json(args.run_json_impact)
    return {"schema": SCHEMA, "label": LABEL, "protocol": PROTOCOL,
            "status": "PRESPECIFIED analysis code, committed before any A7 outcome existed",
            "script": {"path": "scripts/expA_a7_analysis.py", "sha256": sha256_bytes(Path(__file__).read_bytes())},
            "inputs": inputs, "run_json_verified": bool(args.run_json_impact), "pooled_with_other_runs": False,
            "seed": SEED, "arms": {arm: {"block": b, "condition": c, "variant": v} for arm, (b, c, v) in ARMS.items()},
            "analysis": analyse(impact, original)}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scores-impact", required=True, help="scores.jsonl of the A7 run (impact framing)")
    parser.add_argument("--scores-original", required=True,
                        help="scores.jsonl of the main run (original framing; its seed-11 rows of the three arms)")
    parser.add_argument("--run-json-impact", help="run.json of the A7 run (launches/ next to it)")
    parser.add_argument("--out", required=True, help="output directory (must not exist)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.out)
    if out.exists():
        print(f"refused: {out} exists", file=sys.stderr)
        return 2
    try:
        result = run(args)
    except (Refused, OSError, ValueError) as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    out.mkdir(parents=True)
    (out / "A7_ANALYSIS.json").write_bytes(
        (json.dumps(result, indent=1, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))
    text = tables(result)
    (out / "A7_TABLES.md").write_bytes(text.encode("utf-8"))
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
