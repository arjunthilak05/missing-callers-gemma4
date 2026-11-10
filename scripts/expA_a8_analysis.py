#!/usr/bin/env python3
"""Exp A follow-up A8 (other model families): the prespecified analysis of protocols/EXPA_A8_MODELS.md section 3.

PRESPECIFIED: written with the protocol, before any A8 cell exists. ANALYSIS RECORD.
Standard library only; exact rational means. It reuses the A7 analysis functions unchanged (scripts/expA_a7_analysis.py:
row loading, completed cells, family summaries, the two arm contrasts T1 and T2 and the paired comparison with a
reference run), so every model is analysed exactly as A7 analysed its run.

    python scripts/expA_a8_analysis.py --scores-gemma MAIN/scores.jsonl \\
        --model NAME=NONE/scores.jsonl,DEADLINE/scores.jsonl [--model ...] --out DIR

Inputs are the scorer's rows (scripts/expA_score.py) on the same 750 main items, seed 11:
    per model, run R1 (no final-turn line): block P C0 (none), P C1 (omit_node) and S-VC C1 (vcomplete), and run R2
    (the A1 deadline line): block P C0 and C1. Each must dispatch nothing else (the A7 row check refuses otherwise).
    --scores-gemma: the main run (Gemma 4 31B BF16, main-20261101b); its seed-11 rows of the same three arms are the
    reference for the model-minus-Gemma comparisons. Its other rows are ignored.
Outcome: site_recovered of completed cells (a completed episode that did not submit scores 0); cells missing for
infrastructure are excluded and counted. Item value -> family mean -> equal-weight mean over families; two-sided
one-sample t over the family means (df = F - 1).
  PRIMARY, per model (three tests; Holm across all 3 x models tests, alpha 0.05):
      T1 = rec(incomplete) - rec(none) in R1                       (the list effect)
      T2 = rec(complete) - rec(incomplete) in R1                   (the omission cost)
      T3 = mean over none and incomplete of rec(R2) - rec(R1)      (the deadline line's effect; across runs, paired
           by item over items with both arms completed in both runs)
  SECONDARY (nominal 95% intervals; no p-values): the list effect under the deadline line (R2); per arm and run:
      recovery, submission, completeness claims, responses used, precision bounds, exits (FormatExhausted included);
      model - Gemma differences of T1 and T2 paired by item; libraries by sign.
Reading rules (fixed by the protocol): an effect recurs in a model when its estimate has the sign it has for Gemma 4
31B and its Holm-adjusted p is at most 0.05; otherwise it is not detected in that model. Model runs and the Gemma
reference are different runs on different dates and servers.
Outputs (in --out, which must not exist): A8_ANALYSIS.json and A8_TABLES.md. Exit codes: 0 written; 2 refused.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import expA_a7_analysis as a7  # noqa: E402
from expA_reanalyze_independent import holm_adjust  # noqa: E402

SCHEMA = "expA-a8-analysis-v1"
PROTOCOL = "protocols/EXPA_A8_MODELS.md"
LABEL = ("PRESPECIFIED (protocols/EXPA_A8_MODELS.md, written before any A8 cell); ANALYSIS RECORD"
         "; EXPLORATORY relative to the Gemma results, which were known when A8 was designed")
ALPHA = 0.05
GEMMA_SIGNS = {"T1": 1, "T2": 1, "T3": 1}   # the Gemma 4 31B signs (main run, A1 S1), fixed in the protocol
DEFINITIONS = {"T1": "rec(incomplete) - rec(none) in R1 (the list effect)",
               "T2": "rec(complete) - rec(incomplete) in R1 (the omission cost)",
               "T3": "mean over none and incomplete of rec(R2) - rec(R1), paired by item (the deadline line's effect)"}


def _family(*runs):
    family = {}
    for run in runs:
        for rows in run["cells"].values():
            for item, row in rows.items():
                family.setdefault(item, row["family"])
                a7._require(family[item] == row["family"], f"{item}: families differ between rows")
    return family


def analyse_model(name, none_run, deadline_run, gemma_run):
    a7._require(none_run["items_sha256"] == deadline_run["items_sha256"] == gemma_run["items_sha256"],
                f"{name}: the runs name different items files")
    a7._require(all(r["cell_state"] == "not_dispatched" for r in deadline_run["cells"]["complete"].values()),
                f"{name}: R2 must not dispatch the complete arm")
    base = a7.analyse(none_run, gemma_run)          # impact := the model's R1, original := Gemma (main run)
    family = _family(none_run, deadline_run, gemma_run)
    done_none, done_dl = a7.completed(none_run), a7.completed(deadline_run)
    rec_none = {arm: a7.recovery(rows) for arm, rows in done_none.items()}
    rec_dl = {arm: a7.recovery(rows) for arm, rows in done_dl.items()}
    arms = ("none", "incomplete")
    t3_items = sorted(set.intersection(*(set(rec_none[a]) & set(rec_dl[a]) for a in arms)))
    t3 = a7.family_summary({i: sum((rec_dl[a][i] - rec_none[a][i] for a in arms), Fraction(0)) / 2 for i in t3_items},
                           family, test=True)
    t3["definition"] = DEFINITIONS["T3"]
    dl_items = sorted(set(rec_dl["none"]) & set(rec_dl["incomplete"]))
    tests = {"T1": base["primary"]["tests"]["T1"], "T2": base["primary"]["tests"]["T2"], "T3": t3}
    for key in ("T1", "T2"):
        tests[key] = {k: v for k, v in tests[key].items() if k not in ("p_holm", "reject_at_alpha", "decision")}
        tests[key]["definition"] = DEFINITIONS[key]
    secondary = {
        "list_effect_under_deadline": dict(a7.family_summary(
            {i: rec_dl["incomplete"][i] - rec_dl["none"][i] for i in dl_items}, family),
            definition="rec(incomplete) - rec(none) in R2"),
        "model_minus_gemma": {k: v for k, v in base["secondary"]["framing_difference_contrasts"].items()},
        "model_minus_gemma_arms": base["secondary"]["framing_difference_arms"],
        "gemma_reference": base["secondary"]["original_framing"],
        "r1_arms": base["secondary"]["by_framing"]["impact"],
        "r2_arms": {arm: a7.arm_metrics(done_dl[arm], dl_items, family) for arm in arms},
        "exits": {"r1": base["secondary"]["exits"]["impact"],
                  "r2": {arm: dict(sorted(Counter(r.get("exit_status") for r in done_dl[arm].values()).items()))
                         for arm in arms}},
        "libraries_by_sign": {"T1": base["secondary"]["libraries_by_sign"]["impact"]["T1"],
                              "T2": base["secondary"]["libraries_by_sign"]["impact"]["T2"],
                              "T3": a7.sign_counts({i: sum((rec_dl[a][i] - rec_none[a][i] for a in arms), Fraction(0))
                                                    / 2 for i in t3_items}, family)},
        "items": {"r1_three_arms": base["population"]["items_by_framing"]["impact"], "t3": len(t3_items),
                  "r2_both_arms": len(dl_items)},
        "missing_infrastructure_cells": {
            "r1": base["population"]["missing_infrastructure_cells"]["impact"],
            "r2": sum(1 for rows in deadline_run["cells"].values() for r in rows.values()
                      if r["cell_state"] == "missing_infrastructure")}}
    return tests, secondary


def run(args):
    out = Path(args.out)
    a7._require(not out.exists(), f"{out} exists")
    gemma = a7.load_rows(args.scores_gemma, "original")
    models, inputs = {}, {"gemma": {"path": gemma["path"], "sha256": gemma["sha256"]}}
    for spec in args.model:
        name, _, paths = spec.partition("=")
        none_path, _, dl_path = paths.partition(",")
        a7._require(name and none_path and dl_path, f"--model {spec}: NAME=NONE_SCORES,DEADLINE_SCORES")
        a7._require(name not in models, f"--model {name} given twice")
        none_run, dl_run = a7.load_rows(none_path, "impact"), a7.load_rows(dl_path, "impact")
        models[name] = analyse_model(name, none_run, dl_run, gemma)
        inputs[name] = {"r1": {"path": none_run["path"], "sha256": none_run["sha256"]},
                        "r2": {"path": dl_run["path"], "sha256": dl_run["sha256"]}}
    pvalues = {f"{m}|{t}": tests[t].get("p_two_sided") for m, (tests, _) in models.items() for t in tests}
    adjusted, order = holm_adjust(pvalues)
    result = {"schema": SCHEMA, "label": LABEL, "protocol": PROTOCOL, "alpha": ALPHA, "holm_family": sorted(pvalues),
              "holm_order": order, "inputs": inputs,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "a7_script_sha256": hashlib.sha256(Path(a7.__file__).read_bytes()).hexdigest(), "models": {}}
    for m, (tests, secondary) in models.items():
        for t, summary in tests.items():
            p = adjusted[f"{m}|{t}"]
            est = summary.get("estimate")
            same_sign = est is not None and (est > 0) == (GEMMA_SIGNS[t] > 0) and est != 0
            summary.update(p_holm=p, recurs=bool(p is not None and p <= ALPHA and same_sign))
        result["models"][m] = {"primary": tests, "secondary": secondary}
    out.mkdir(parents=True)
    (out / "A8_ANALYSIS.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    (out / "A8_TABLES.md").write_text(tables(result), encoding="utf-8")
    return result


def tables(result):
    lines = ["# A8: other model families (prespecified analysis output)", "", result["label"], "",
             "| Model | Test | Estimate [95% CI] (points) | p | Holm p | Recurs | Libraries +/0/- |", "|---|---|---|---|---|---|---|"]
    for m, block in result["models"].items():
        for t, s in block["primary"].items():
            signs = block["secondary"]["libraries_by_sign"][t]
            lines.append(f"| {m} | {t} | {a7._iv(s)} | {a7._num(s.get('p_two_sided'), '.3g')} "
                         f"| {a7._num(s.get('p_holm'), '.3g')} | {'yes' if s['recurs'] else 'no'} "
                         f"| {signs['positive']}/{signs['zero']}/{signs['negative']} |")
    lines += ["", "| Model | Run | Arm | Recovery | Submission | Claims yes | Exits |", "|---|---|---|---|---|---|---|"]
    for m, block in result["models"].items():
        sec = block["secondary"]
        for run_key, arms in (("r1", sec["r1_arms"]), ("r2", sec["r2_arms"])):
            for arm, met in arms.items():
                lines.append(f"| {m} | {run_key} | {arm} | {a7._pp(met['recovery'].get('estimate'))} "
                             f"| {a7._pp(met['submission'].get('estimate'))} "
                             f"| {a7._pp(met['completeness_claim_yes'].get('estimate'))} "
                             f"| {json.dumps(sec['exits'][run_key][arm])} |")
    return "\n".join(lines) + "\n"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--scores-gemma", required=True)
    p.add_argument("--model", action="append", required=True, metavar="NAME=NONE_SCORES,DEADLINE_SCORES")
    p.add_argument("--out", required=True)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        run(args)
    except a7.Refused as error:
        print(f"expA_a8_analysis: REFUSED: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
