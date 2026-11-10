#!/usr/bin/env python3
"""Omission cost at function level. EXPLORATORY, POST HOC (2026-11-07, after the S7 worked example showed a harmed
pair whose incomplete-list episode submitted the line above C*'s call site).

For the A3 ledger-off arm (8 responses) and A6 (16 responses): complete-list minus incomplete-list recovery, scored
two ways on the same completed LEDGER cells: the frozen strict outcome ``site_recovered`` and the prespecified
secondary outcome ``function_localized`` (some valid submitted site lies inside C*'s function; score.py). Item values
average the item's seeds; the estimate is the equal-weight mean of library means (the frozen family weighting), with
a two-sided t interval over libraries. Also counts complete-only pairs (item, seed) whose incomplete-list episode was
function-localized. Reads only frozen score files; standard library only.

    python scripts/expA_function_level.py --scores NAME=SCORES.jsonl [...] --out DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import sys

# two-sided 95% t quantiles by degrees of freedom (libraries - 1)
T975 = {20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045}
ARMS = {"complete": "vcomplete", "incomplete": "omit_node"}


def variant(row):
    return row["cell_id"].split("|")[2]


def contrast(rows, outcome):
    vals = defaultdict(lambda: defaultdict(list))        # (family, item) -> arm -> values
    for r in rows:
        vals[(r["family"], r["item_id"])][variant(r)].append(r[outcome])
    fam = defaultdict(list)
    for (family, _), arms in vals.items():
        if arms.get("vcomplete") and arms.get("omit_node"):
            fam[family].append(sum(arms["vcomplete"]) / len(arms["vcomplete"])
                               - sum(arms["omit_node"]) / len(arms["omit_node"]))
    means = [sum(v) / len(v) for v in fam.values()]
    n = len(means)
    m = sum(means) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in means) / (n - 1))
    half = T975[n - 1] * sd / math.sqrt(n)
    return {"libraries": n, "estimate": m, "lo": m - half, "hi": m + half,
            "positive": sum(x > 0 for x in means), "negative": sum(x < 0 for x in means)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scores", action="append", required=True, help="NAME=PATH")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        sys.exit(f"refusing: {out} exists")
    res = {"label": "EXPLORATORY, POST HOC function-level omission cost (2026-11-07); ANALYSIS RECORD"
                    "", "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "runs": {}}
    for spec in a.scores:
        name, path = spec.split("=", 1)
        rows = [json.loads(x) for x in Path(path).read_bytes().splitlines() if x.strip()]
        rows = [r for r in rows if r.get("kind") == "episode" and r.get("completed") and r["block"] == "LEDGER"
                and variant(r) in ARMS.values()]
        by = {(r["item_id"], r["seed"], variant(r)): r for r in rows}
        pairs = [(i, s) for (i, s, v) in by if v == "vcomplete" and (i, s, "omit_node") in by]
        harmed = [(i, s) for i, s in pairs if by[(i, s, "vcomplete")]["site_recovered"] == 1
                  and by[(i, s, "omit_node")]["site_recovered"] == 0]
        entry = {"path": path, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                 "completed_cells": len(rows), "pairs": len(pairs),
                 "strict": contrast(rows, "site_recovered"),
                 "function_level": contrast(rows, "function_localized"),
                 "complete_only_pairs": len(harmed),
                 "complete_only_incomplete_function_localized":
                     sum(by[(i, s, "omit_node")]["function_localized"] for i, s in harmed)}
        res["runs"][name] = entry
        print(name, json.dumps({k: v for k, v in entry.items() if k not in ("path", "sha256")}))
    out.mkdir(parents=True)
    (out / "FUNCTION_LEVEL.json").write_bytes((json.dumps(res, indent=1) + "\n").encode())


if __name__ == "__main__":
    main()
