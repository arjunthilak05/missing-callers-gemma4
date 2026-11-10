#!/usr/bin/env python3
"""Generality matrix of the headline effects. EXPLORATORY, POST HOC (2026-11-07, after official-rubric reviews R11).

For each effect: its frozen estimate, the number of libraries with a positive (and with a negative) library value,
and leave-one-library-out (LOLO) stability: the minimum and maximum of the equal-weight mean when each library in
turn is dropped. Reads only frozen result files; standard library only.

    python scripts/expA_generality.py --out DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


def h3_units(path, hid="H3"):
    a = json.loads(Path(path).read_text(encoding="utf-8"))
    hyp = next(h for h in a["primary"]["hypotheses"] if h["id"] == hid)
    return {u["family"]: u["difference"] for u in hyp["family_units"]}


def dig(path, keys):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in keys:
        value = value[key]
    return value


EFFECTS = [
    ("H3, incomplete list minus no list (main run, Gemma 4 31B)", "reports/expA-main-20261101b/analysis/ANALYSIS.json",
     lambda p: h3_units(p)),
    ("H3 on Gemma 4 12B", "reports/expA-repl12b-20261101a/analysis/ANALYSIS.json", lambda p: h3_units(p)),
    ("H3 on the QAT W4A16 checkpoint", "reports/expA-qat31b-20261102a/analysis/ANALYSIS.json", lambda p: h3_units(p)),
    ("H3 on callers missed by name search", "reports/expA-hard-20261102a/ANALYSIS.json", lambda p: h3_units(p)),
    ("Deadline line minus neutral line (A1)", "reports/expA-finalturn-20261102a/analysis/RESULTS.json",
     lambda p: dig(p, ["primary", "family_values"])),
    ("Deadline line minus neutral line at a second seed (A4)", "reports/expA-a4directive-20261106/A4_RESULTS.json",
     lambda p: dig(p, ["analysis", "secondary", "S5_deadline_minus_neutral", "family_values"])),
    ("Information main effect of the last-response line (A4)", "reports/expA-a4directive-20261106/A4_RESULTS.json",
     lambda p: dig(p, ["analysis", "secondary", "S3_information_main_effect", "family_values"])),
    ("Omission cost under the deadline line, 8 responses (A3 ledger off)",
     "reports/expA-a3ledger-20261105/A3_ANALYSIS.json",
     lambda p: dig(p, ["main", "G_off_omission_cost_without_ledger", "family_means"])),
    ("Omission cost under the deadline line, 16 responses (A6)", "reports/expA-a6budget16-20261106/A6_ANALYSIS.json",
     lambda p: dig(p, ["main", "G16_omission_cost_16_responses", "family_means"])),
    ("Reduction of the omission cost with the doubled budget (A6)",
     "reports/expA-a6budget16-20261106/A6_ANALYSIS.json",
     lambda p: dig(p, ["main", "R_reduction_in_omission_cost", "family_means"])),
]


def summarise(values):
    v = {k: float(x) for k, x in values.items()}
    n = len(v)
    mean = sum(v.values()) / n
    lolo = [(sum(v.values()) - x) / (n - 1) for x in v.values()]
    return {"libraries": n, "estimate": mean, "positive": sum(x > 0 for x in v.values()),
            "negative": sum(x < 0 for x in v.values()), "zero": sum(x == 0 for x in v.values()),
            "lolo_min": min(lolo), "lolo_max": max(lolo)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        sys.exit(f"refusing: {out} exists")
    rows = []
    for name, path, get in EFFECTS:
        row = {"effect": name, "source": path, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
        row.update(summarise(get(path)))
        rows.append(row)
    res = {"label": "EXPLORATORY, POST HOC generality summary (2026-11-07); ANALYSIS RECORDhuman "
                    "review; estimates are unweighted means of the library values, equal to the frozen family-weighted "
                    "estimates",
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "effects": rows}
    out.mkdir(parents=True)
    (out / "GENERALITY.json").write_bytes((json.dumps(res, indent=1) + "\n").encode())
    for r in rows:
        print(f"{r['effect'][:62]:62} est {r['estimate']:+.3f} pos {r['positive']:2}/{r['libraries']} "
              f"neg {r['negative']:2} LOLO [{r['lolo_min']:+.3f}, {r['lolo_max']:+.3f}]")


if __name__ == "__main__":
    main()
