#!/usr/bin/env python3
"""Audits of the follow-up mechanism claims. EXPLORATORY, POST HOC.

Designed 2026-11-06 after reviews. Standard library only.

    python scripts/expA_followup_audit.py --items items.jsonl --a3-run-dir OFF_RUN --a3-scores OFF/scores.jsonl \
        --a5-run-dir LEDGER_RUN --a5-scores LEDGER/scores.jsonl --out DIR

1. Display-detector sensitivity (A3 ledger off): for every omission-harmed pair (complete list recovered C*, the
   incomplete list did not) whose incomplete episode submitted without a STRICT display of a C* call site, check two
   looser signals in any model-visible tool result: (a) a C* site's file path occurs (path token, as
   expA_timeline.path_occurs); (b) C*'s own name (last qualname component) occurs as a word. A loose hit shows the
   strict rule may have missed a display; no loose hit makes a missed display unlikely.
2. Review-ledger opportunities (A5 ledger arm): for reviewed incomplete-list episodes whose first submission lacked
   C*, whether the review ledger listed a C* call site, and whether the resubmission then recovered C*.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from expA_timeline import path_occurs, s_struct_hit, s_text_hit, unit_from_text  # noqa: E402

LAST_CALL_INDEX = 7


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_bytes().splitlines() if line.strip()]


def items_info(path):
    out = {}
    for it in read_jsonl(path):
        cstar = it["omitted"]["identity"]
        sites = [(s["path"], s["lineno"], s.get("end_lineno") or s["lineno"])
                 for v in it["V"] if v["identity"] == cstar for s in v["sites"]]
        out[it["item_id"]] = {"sites": sites, "name": cstar.split("::", 1)[1].rsplit(".", 1)[-1]}
    return out


def recovered(row):
    return int(row.get("exit_status") == "Submitted" and bool(row.get("site_recovered")))


def events(run_dir, row):
    return read_jsonl(Path(run_dir) / "cells" / row["cell_id"] / row["attempt_id"] / "events.jsonl")


def tool_texts(evts):
    return [e["value"].get("content", "") for e in evts if e["kind"] == "model_visible_tool_result"]


def strict_hit(texts, sites):
    for text in texts:
        strings, pairs = unit_from_text(text, is_json=True)
        if s_text_hit(strings, sites) or s_struct_hit(pairs, sites):
            return True
    return False


def detector_audit(info, run_dir, scores):
    cells = {(r["item_id"], r["seed"], r["variant"]): r for r in read_jsonl(scores)
             if r.get("block") == "LEDGER" and r.get("cell_state") == "completed"}
    out = Counter()
    for (item, seed, variant), comp in cells.items():
        inc = cells.get((item, seed, "omit_node"))
        if variant != "vcomplete" or inc is None or not recovered(comp) or recovered(inc):
            continue
        if inc.get("exit_status") != "Submitted":
            continue
        texts = tool_texts(events(run_dir, inc))
        sites = info[item]["sites"]
        if strict_hit(texts, sites):
            continue
        out["no_strict_display"] += 1
        path_hit = any(path_occurs(t, p) for t in texts for p, _, _ in sites)
        name = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(info[item]["name"]) + r"(?![A-Za-z0-9_])")
        name_hit = any(name.search(t) for t in texts)
        out["path_shown"] += path_hit
        out["name_shown"] += name_hit
        out["path_or_name_shown"] += path_hit or name_hit
        out["neither_shown"] += not (path_hit or name_hit)
    return dict(out)


def review_audit(info, run_dir, scores):
    out = Counter()
    for r in read_jsonl(scores):
        if r.get("block") != "LEDGER" or r.get("variant") != "omit_node" or r.get("cell_state") != "completed":
            continue
        evts = events(run_dir, r)
        reviews = [e["value"] for e in evts if e["kind"] == "submit_review"]
        if not reviews:
            continue
        review = reviews[0]
        sites = info[r["item_id"]]["sites"]
        first = review.get("submission") or {}
        first_sites = first.get("call_sites") or first.get("sites") or []
        first_has = any(isinstance(s, dict) and any(s.get("path") == p and isinstance(s.get("line"), int)
                                                    and lo <= s["line"] <= hi for p, lo, hi in sites)
                        for s in first_sites)
        if first_has:
            continue
        out["reviewed_first_without_cstar"] += 1
        entries = (review.get("ledger") or {}).get("entries") or []
        listed = any(e.get("path") == p and lo <= e.get("line", 0) <= hi for e in entries for p, lo, hi in sites)
        out["ledger_listed_cstar"] += listed
        rec = recovered(r)
        out["final_recovered"] += rec
        out["ledger_listed_cstar_and_final_recovered"] += listed and rec
    return dict(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for a in ("--items", "--a3-run-dir", "--a3-scores", "--a5-run-dir", "--a5-scores", "--out"):
        ap.add_argument(a, required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        sys.exit(f"refusing: {out} exists")
    info = items_info(a.items)
    res = {"label": "EXPLORATORY, POST HOC (2026-11-06, after reviews R9); ANALYSIS RECORDhuman "
                    "review",
           "script_sha256": sha256(__file__),
           "inputs": {k: {"path": v, "sha256": sha256(v)} for k, v in
                      (("items", a.items), ("a3_scores_off", a.a3_scores), ("a5_scores_ledger", a.a5_scores))},
           "a3_detector_audit": detector_audit(info, a.a3_run_dir, a.a3_scores),
           "a5_review_opportunities": review_audit(info, a.a5_run_dir, a.a5_scores)}
    out.mkdir(parents=True)
    (out / "FOLLOWUP_AUDIT.json").write_bytes((json.dumps(res, indent=1, sort_keys=True) + "\n").encode())
    print(json.dumps({k: res[k] for k in ("a3_detector_audit", "a5_review_opportunities")}, indent=1))


if __name__ == "__main__":
    main()
