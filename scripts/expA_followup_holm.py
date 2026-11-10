#!/usr/bin/env python3
"""Holm sensitivity across the follow-ups' primary tests. EXPLORATORY, POST HOC (2026-11-06; A7 added 2026-11-07).

Each follow-up was prespecified with its own primary test family at alpha 0.05 (one test each in A1-A6, two in A7). This sensitivity analysis,
designed after reviews (R9), asks whether each decision survives a Holm step-down across all of them taken as
one family. Standard library only; reads the frozen outputs and writes FOLLOWUP_HOLM.json.

    python scripts/expA_followup_holm.py --out DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ALPHA = 0.05
SOURCES = [
    ("A1 last-response line: deadline - neutral", "reports/expA-finalturn-20261102a/analysis/RESULTS.json",
     ["primary", "p_two_sided"]),
    ("A2 callers missed by name search: H3", "reports/expA-hard-20261102a/followup/RESULTS.json",
     ["hypotheses", "H3", "p_two_sided"]),
    ("A3 evidence ledger: M", "reports/expA-a3ledger-20261105/A3_ANALYSIS.json",
     ["main", "M_reduction_in_omission_cost", "p_two_sided"]),
    ("A4 final-line components: instruction x list effect", "reports/expA-a4directive-20261106/A4_RESULTS.json",
     ["analysis", "primary", "p_two_sided"]),
    ("A5 submit-time review pilot: M", "reports/expA-a5review-20261106/A5_ANALYSIS.json",
     ["main", "M_reduction_in_omission_cost", "p_two_sided"]),
    ("A6 doubled budget: R", "reports/expA-a6budget16-20261106/A6_ANALYSIS.json", None),
    ("A7 impact-analysis task: T1", "reports/expA-a7impact-20261107/A7_ANALYSIS.json",
     ["analysis", "primary", "tests", "T1", "p_two_sided"]),
    ("A7 impact-analysis task: T2", "reports/expA-a7impact-20261107/A7_ANALYSIS.json",
     ["analysis", "primary", "tests", "T2", "p_two_sided"]),
]


def dig(value, path):
    for key in path:
        value = value[key]
    return value


def a6_p(doc):
    """The A6 primary p, wherever the analysis stored it (main.primary... or primary...)."""
    for path in (["main", "primary", "p_two_sided"], ["primary", "p_two_sided"],
                 ["main", "R_reduction_in_omission_cost", "p_two_sided"]):
        try:
            return dig(doc, path), path
        except (KeyError, TypeError):
            continue
    raise KeyError("A6 primary p not found")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        sys.exit(f"refusing: {out} exists")
    tests = []
    for name, path, keys in SOURCES:
        if not Path(path).is_file():
            sys.exit(f"missing {path}")
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        p, used = (dig(doc, keys), keys) if keys else a6_p(doc)
        tests.append({"test": name, "source": path, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                      "json_path": ".".join(used), "p": float(p)})
    order = sorted(range(len(tests)), key=lambda i: tests[i]["p"])
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(tests) - rank) * tests[i]["p"]))
        tests[i]["holm_p"] = running
        tests[i]["reject_unadjusted"] = tests[i]["p"] <= ALPHA
        tests[i]["reject_holm"] = running <= ALPHA
    res = {"label": "EXPLORATORY, POST HOC sensitivity (2026-11-06, after reviews R9); the prespecified decisions "
                    "are the unadjusted ones; ANALYSIS RECORD",
           "alpha": ALPHA, "family": "the primary tests of follow-ups A1-A7", "tests": tests,
           "decisions_changed": [t["test"] for t in tests if t["reject_unadjusted"] != t["reject_holm"]]}
    out.mkdir(parents=True)
    (out / "FOLLOWUP_HOLM.json").write_bytes((json.dumps(res, indent=1) + "\n").encode())
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
