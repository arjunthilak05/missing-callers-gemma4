#!/usr/bin/env python3
"""Pre-specified cross-run comparisons for Exp A (stdlib only).

--mode model   protocols/EXPA_REPL12B.md sections 5-6: for each Hk, the 12B family-mean difference minus the 31B
               family-mean difference, paired by family (nominal 95% t interval, df = F - 1), and the frozen
               classification applied in its fixed order.
--mode budget  protocols/EXPA_BUDGET16.md section 3: for each Hk, Hk(16 responses, seed 11) - Hk(8 responses, seed 11)
               paired by family, plus per-condition exits, submission, recovery given submission, responses and tokens
               for both runs at seed 11.

Unit (prereg section 10): item mean over the selected seeds -> paired item difference -> family mean. Inputs are the
frozen CELLS.jsonl of each run (block P only). The per-run estimates are cross-checked against each run's frozen
ANALYSIS.json (model mode) before anything is written. Analysis code
"""
import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from expA_reanalyze_independent import t_statistics  # noqa: E402  (stdlib t, exact rationals)

HYPOTHESES = {"H1": ("C3", "C2"), "H2": ("C4a", "C1"), "H3": ("C1", "C0")}
CONDITIONS = ("C0", "C1", "C2", "C3", "C4a")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_p(path, seeds):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["block"] == "P" and row["cell_state"] == "completed" and (seeds is None or row["seed"] in seeds):
                rows.append(row)
    return rows


def family_effects(rows):
    """{Hk: {family: Fraction}} using the primary unit."""
    values = defaultdict(list)
    family = {}
    for row in rows:
        values[(row["item_id"], row["condition"])].append(Fraction(int(row["site_recovered"] or 0)))
        family[row["item_id"]] = row["family"]
    item_mean = {key: sum(v, Fraction(0)) / len(v) for key, v in values.items()}
    out = {}
    for h, (treat, ctrl) in HYPOTHESES.items():
        groups = defaultdict(list)
        for item in family:
            if (item, treat) in item_mean and (item, ctrl) in item_mean:
                groups[family[item]].append(item_mean[(item, treat)] - item_mean[(item, ctrl)])
        out[h] = {fam: sum(d, Fraction(0)) / len(d) for fam, d in groups.items()}
    return out


def summary(values):
    stats = t_statistics(list(values), 0.95)
    return {"estimate": stats["mean"], "ci_low": stats.get("lower"), "ci_high": stats.get("upper"),
            "t": stats.get("t"), "df": stats["df"], "p_nominal": stats.get("p_two_sided"), "n_families": stats["n"]}


def paired(new, ref):
    out = {}
    for h in HYPOTHESES:
        fams = sorted(set(new[h]) & set(ref[h]))
        if set(new[h]) != set(ref[h]):
            raise SystemExit(f"{h}: family sets differ between runs")
        out[h] = summary(new[h][f] - ref[h][f] for f in fams)
    return out


def per_run(effects):
    return {h: summary(effects[h].values()) for h in HYPOTHESES}


def frozen_hypotheses(analysis_path):
    analysis = json.loads(Path(analysis_path).read_text(encoding="utf-8"))
    rows = analysis["primary"]["hypotheses"]
    out = {}
    for row in rows:
        holm = row.get("holm_adjusted_p")
        if holm is None:
            holm = (row.get("holm") or {}).get("adjusted_p")
        out[row["id"]] = {"estimate": row.get("estimate"), "holm_p": holm}
    return out


LABELS = {"model": {"second": "second-model test", "dependent": "model-dependent"},
          "checkpoint": {"second": "second-checkpoint test", "dependent": "checkpoint-dependent"}}


def classify(ref, new, delta, alpha=0.05, kind="model"):
    """protocols/EXPA_REPL12B.md section 6 (and EXPA_QAT31B.md section 6), applied in its fixed order."""
    out = {}
    for h in HYPOTHESES:
        r, n, d = ref[h], new[h], delta[h]
        ref_sig = r["holm_p"] is not None and r["holm_p"] < alpha
        new_sig = n["holm_p"] is not None and n["holm_p"] < alpha
        same = (n["estimate"] > 0) == (r["estimate"] > 0)
        if not ref_sig:
            label, addendum = LABELS[kind]["second"], "independent test"
        elif new_sig and not same:
            label = addendum = LABELS[kind]["dependent"]
        elif new_sig and same:
            excludes = d["ci_low"] > 0 or d["ci_high"] < 0
            label = addendum = "replicated, magnitude differs" if excludes else "replicated"
        elif same:
            label = addendum = "same direction, not significant"
        else:
            label = addendum = "opposite direction, not significant"
        out[h] = {"label": label, "addendum_label": addendum, "ref_holm_significant": ref_sig,
                  "new_holm_significant": new_sig, "same_sign": same}
    return out


def condition_table(rows):
    table = {}
    for cond in CONDITIONS:
        sub = [r for r in rows if r["condition"] == cond]
        submitted = [r for r in sub if r["exit_status"] == "Submitted"]
        exits = Counter(r["exit_status"] for r in sub)
        n = len(sub)
        table[cond] = {
            "episodes": n, "exits": dict(sorted(exits.items())),
            "submitted_share": len(submitted) / n if n else None,
            "recovered_share": sum(int(r["site_recovered"] or 0) for r in sub) / n if n else None,
            "recovery_given_submission": (sum(int(r["site_recovered"] or 0) for r in submitted) / len(submitted)
                                          if submitted else None),
            "mean_calls": sum(r["calls"] for r in sub) / n if n else None,
            "mean_calls_submitted": (sum(r["calls"] for r in submitted) / len(submitted) if submitted else None),
            "mean_prompt_tokens": sum(r["prompt_tokens"] for r in sub) / n if n else None,
            "mean_completion_tokens": sum(r["completion_tokens"] for r in sub) / n if n else None,
            # A Submitted episode's last response is its submission, so calls is the response at which it submitted.
            "submitted_by_response_8_share": (sum(1 for r in submitted if r["calls"] <= 8) / n if n else None),
            "recovered_by_response_8_share": (sum(int(r["site_recovered"] or 0) for r in submitted if r["calls"] <= 8)
                                              / n if n else None),
        }
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--mode", choices=("model", "budget"), required=True)
    parser.add_argument("--ref-cells", required=True)
    parser.add_argument("--new-cells", required=True)
    parser.add_argument("--ref-analysis")
    parser.add_argument("--new-analysis")
    parser.add_argument("--seed", type=int, default=11, help="budget mode: the seed both runs share")
    parser.add_argument("--labels", choices=tuple(LABELS), default="model",
                        help="model mode: wording of the classification (model = 12B test, checkpoint = QAT test)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    seeds = None if args.mode == "model" else {args.seed}
    ref_rows, new_rows = load_p(args.ref_cells, seeds), load_p(args.new_cells, seeds)
    ref_fx, new_fx = family_effects(ref_rows), family_effects(new_rows)
    result = {
        "label": "ANALYSIS RECORD",
        "mode": args.mode,
        "inputs": {"ref_cells": {"path": args.ref_cells, "sha256": sha256(args.ref_cells)},
                   "new_cells": {"path": args.new_cells, "sha256": sha256(args.new_cells)}},
        "unit": "item mean over the selected seeds -> paired item difference -> family mean; nominal 95% t over "
                "family differences (df = F - 1)",
        "ref_estimates": per_run(ref_fx),
        "new_estimates": per_run(new_fx),
    }
    if args.mode == "model":
        result["status"] = ("SECONDARY, descriptive (protocols/EXPA_REPL12B.md section 5); the classification "
                            "follows section 6 in its fixed order")
        for key in ("ref_analysis", "new_analysis"):
            result["inputs"][key] = {"path": getattr(args, key), "sha256": sha256(getattr(args, key))}
        ref_h, new_h = frozen_hypotheses(args.ref_analysis), frozen_hypotheses(args.new_analysis)
        for h in HYPOTHESES:  # the recomputed per-run estimates must equal the frozen ones
            for mine, frozen in ((result["ref_estimates"][h], ref_h[h]), (result["new_estimates"][h], new_h[h])):
                if abs(mine["estimate"] - frozen["estimate"]) > 1e-12:
                    raise SystemExit(f"{h}: recomputed {mine['estimate']} != frozen {frozen['estimate']}")
        result["frozen"] = {"ref": ref_h, "new": new_h}
        result["delta"] = paired(new_fx, ref_fx)
        result["classification_detail"] = classify(ref_h, new_h, result["delta"], kind=args.labels)
        result["classification"] = {h: v["label"] for h, v in result["classification_detail"].items()}
        both_within = {}
        new_analysis = json.loads(Path(args.new_analysis).read_text(encoding="utf-8"))
        ref_analysis = json.loads(Path(args.ref_analysis).read_text(encoding="utf-8"))
        for h in HYPOTHESES:
            ivs = []
            for analysis in (ref_analysis, new_analysis):
                row = [r for r in analysis["primary"]["hypotheses"] if r["id"] == h][0]
                interval = (row.get("t_test") or {}).get("interval") or {}
                ivs.append((interval.get("lower"), interval.get("upper")))
            both_within[h] = all(lo is not None and hi is not None and -0.10 <= lo and hi <= 0.10 for lo, hi in ivs)
        result["both_intervals_within_0_10"] = both_within
        # protocols/EXPA_REPL12B.md section 7: the same exploratory decomposition as the reference run (all seeds).
        result["conditions_status"] = "EXPLORATORY decomposition (exit partition, submission, recovery given submission)"
        result["conditions"] = {"ref": condition_table(ref_rows), "new": condition_table(new_rows)}
    else:
        result["status"] = ("EXPLORATORY (protocols/EXPA_BUDGET16.md section 3): nominal intervals, no p-values "
                            "interpreted")
        result["seed"] = args.seed
        result["did"] = paired(new_fx, ref_fx)
        result["conditions"] = {"ref_8_responses": condition_table(ref_rows),
                                "new_16_responses": condition_table(new_rows)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({h: {k: result[k][h] for k in (("delta", "classification") if args.mode == "model" else ("did",))}
                      for h in HYPOTHESES}, indent=1))


if __name__ == "__main__":
    main()
