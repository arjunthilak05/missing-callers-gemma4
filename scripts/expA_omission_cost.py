#!/usr/bin/env python3
"""Omission cost: complete answer (S-VC, vcomplete) minus incomplete answer (block P, omit_node), same items, seed 11.

EXPLORATORY and POST HOC: designed on 2026-11-04 after every analysed run's results were known (prompted by a GPT
review). Not a preregistered test; intervals are nominal. Standard library only.

    python scripts/expA_omission_cost.py --cells RUN=path/CELLS.jsonl [--cells ...] --out DIR

For each run and condition in {C1, C3} (the conditions S-VC ran): per item, site_recovered(S-VC, vcomplete, seed 11)
minus site_recovered(P, omit_node, seed 11), using completed cells only (a completed episode that did not submit
scores 0, as in the frozen analysis); then the family mean; then the equal-weight mean over families with a two-sided
one-sample t interval (df = F - 1; scripts/expA_reanalyze_independent.t_statistics). Items lacking either cell are
counted and dropped. The two answers differ only in C*'s entry (one match and its endpoint snippet), so the contrast is
the recovery cost of withholding C* under each wording. Submission shares (exit Submitted) are reported alongside.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from expA_reanalyze_independent import t_statistics  # noqa: E402

SEED = 11
CONDITIONS = ("C1", "C3")
ARMS = {"incomplete": ("P", "omit_node"), "complete": ("S-VC", "vcomplete")}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(values):
    vals = list(values)
    est = sum(vals, Fraction(0)) / len(vals)
    st = t_statistics(vals, 0.95)
    return {"estimate": float(est), "estimate_exact": f"{est.numerator}/{est.denominator}",
            "ci_low": st["lower"], "ci_high": st["upper"], "n_families": len(vals)}


def family_weighted(per_item, family):
    by_fam = defaultdict(list)
    for item, v in per_item.items():
        by_fam[family[item]].append(Fraction(v))
    return summarize(sum(vs, Fraction(0)) / len(vs) for vs in by_fam.values())


def analyse(cells_path):
    cells = {}
    family = {}
    for line in Path(cells_path).read_bytes().splitlines():
        r = json.loads(line)
        if r.get("seed") != SEED or r.get("cell_state") != "completed":
            continue
        for arm, (block, variant) in ARMS.items():
            if r["block"] == block and r.get("variant") == variant and r["condition"] in CONDITIONS:
                cells[(arm, r["condition"], r["item_id"])] = r
                family[r["item_id"]] = r["family"]
    out = {}
    for cond in CONDITIONS:
        items = sorted({k[2] for k in cells if k[1] == cond})
        paired = [i for i in items if ("incomplete", cond, i) in cells and ("complete", cond, i) in cells]
        rec = {arm: {i: int(cells[(arm, cond, i)]["site_recovered"] or 0) for i in paired} for arm in ARMS}
        sub = {arm: {i: int(cells[(arm, cond, i)]["exit_status"] == "Submitted") for i in paired} for arm in ARMS}
        diff = {i: rec["complete"][i] - rec["incomplete"][i] for i in paired}
        out[cond] = {
            "items_paired": len(paired), "items_unpaired": len(items) - len(paired),
            "families": len({family[i] for i in paired}),
            "recovery": {arm: family_weighted(rec[arm], family) for arm in ARMS},
            "submitted": {arm: family_weighted(sub[arm], family) for arm in ARMS},
            "omission_cost_complete_minus_incomplete": family_weighted(diff, family),
        }
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cells", action="append", required=True, metavar="RUN=PATH")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out_dir = Path(a.out)
    if out_dir.exists():
        sys.exit(f"refusing: {out_dir} exists")
    result = {"label": "EXPLORATORY, POST HOC (designed 2026-11-04 after all results); ANALYSIS RECORD"
                       "", "seed": SEED, "conditions": list(CONDITIONS),
              "script_sha256": sha256(__file__), "inputs": {}, "runs": {}}
    for spec in a.cells:
        run, path = spec.split("=", 1)
        result["inputs"][run] = {"path": path, "sha256": sha256(path)}
        result["runs"][run] = analyse(path)
    out_dir.mkdir(parents=True)
    (out_dir / "OMISSION_COST.json").write_bytes((json.dumps(result, indent=1, sort_keys=True) + "\n").encode())
    lines = ["# Omission cost (complete minus incomplete answer), seed 11", "", result["label"], "",
             "| Run | Cond | Items | Recovery incomplete | Recovery complete | Cost (points) [95% t] | Submitted incomplete | "
             "Submitted complete |", "|---|---|---|---|---|---|---|---|"]
    for run, conds in result["runs"].items():
        for cond, v in conds.items():
            c = v["omission_cost_complete_minus_incomplete"]
            lines.append(f"| {run} | {cond} | {v['items_paired']} | {v['recovery']['incomplete']['estimate']:.3f} | "
                         f"{v['recovery']['complete']['estimate']:.3f} | {100 * c['estimate']:+.1f} "
                         f"[{100 * c['ci_low']:+.1f}, {100 * c['ci_high']:+.1f}] | "
                         f"{v['submitted']['incomplete']['estimate']:.3f} | {v['submitted']['complete']['estimate']:.3f} |")
    (out_dir / "OMISSION_COST.md").write_bytes(("\n".join(lines) + "\n").encode())
    print("\n".join(lines))


if __name__ == "__main__":
    main()
