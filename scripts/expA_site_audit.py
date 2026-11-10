#!/usr/bin/env python3
"""Site-level truth audit of the omitted caller C* (EXPLORATORY, POST HOC; designed 2026-11-04 after an audit).

    python scripts/expA_site_audit.py --items RUN=items.jsonl --scores RUN=scores.jsonl [...] --out DIR

A C* call site is "weak" when neither the runtime trace nor jedi witnesses it (witnesses without "trace" and "J";
only LocAgent and/or the AST name match support it). For each run:
- C* sites, weak C* sites, and items with at least one weak C* site;
- recovered episodes (site_recovered = 1, completed cells, every block) whose recovery rests ONLY on weak sites:
  every submitted line that falls inside a C* site span falls inside a weak span.
Standard library only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

STRONG = {"trace", "J"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cstar_sites(items_path):
    out = {}
    for line in Path(items_path).read_bytes().splitlines():
        it = json.loads(line)
        cstar = it["omitted"]["identity"]
        sites = [s for v in it["V"] if v["identity"] == cstar for s in v["sites"]]
        out[it["item_id"]] = [(s["path"], s["lineno"], s.get("end_lineno") or s["lineno"],
                               not (set(s.get("witnesses") or []) & STRONG)) for s in sites]
    return out


def audit(items_path, scores_path):
    sites = cstar_sites(items_path)
    n_sites = sum(len(v) for v in sites.values())
    weak_sites = sum(1 for v in sites.values() for s in v if s[3])
    weak_items = sum(1 for v in sites.values() if any(s[3] for s in v))
    recovered = weak_only = 0
    for line in Path(scores_path).read_bytes().splitlines():
        r = json.loads(line)
        if r.get("cell_state") != "completed" or r.get("site_recovered") != 1:
            continue
        recovered += 1
        hits = []
        for d in r.get("site_details") or []:
            sub = d.get("submitted") or {}
            for path, lo, hi, weak in sites[r["item_id"]]:
                if sub.get("path") == path and isinstance(sub.get("line"), int) and lo <= sub["line"] <= hi:
                    hits.append(weak)
        if hits and all(hits):
            weak_only += 1
    return {"items": len(sites), "cstar_sites": n_sites, "weak_cstar_sites": weak_sites,
            "items_with_weak_cstar_site": weak_items, "recovered_episodes": recovered,
            "recovered_only_via_weak_sites": weak_only}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--items", action="append", required=True, metavar="RUN=PATH")
    ap.add_argument("--scores", action="append", required=True, metavar="RUN=PATH")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        sys.exit(f"refusing: {out} exists")
    items = dict(s.split("=", 1) for s in a.items)
    scores = dict(s.split("=", 1) for s in a.scores)
    res = {"label": "EXPLORATORY, POST HOC (2026-11-04, after an audit); ANALYSIS RECORD",
           "weak_rule": "C* site whose witnesses include neither 'trace' nor 'J'",
           "script_sha256": sha256(__file__), "inputs": {}, "runs": {}}
    for run in items:
        res["inputs"][run] = {"items": {"path": items[run], "sha256": sha256(items[run])},
                              "scores": {"path": scores[run], "sha256": sha256(scores[run])}}
        res["runs"][run] = audit(items[run], scores[run])
    out.mkdir(parents=True)
    (out / "SITE_AUDIT.json").write_bytes((json.dumps(res, indent=1, sort_keys=True) + "\n").encode())
    print(json.dumps(res["runs"], indent=1))


if __name__ == "__main__":
    main()
