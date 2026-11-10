#!/usr/bin/env python3
"""A3 runs: accounting bound, ledger-content subgroup and loss breakdown.

Labels: the subgroup is PRESPECIFIED as descriptive (protocols/EXPA_A3_LEDGER.md section 3, "Mechanism subgroup"); the
bound and the loss breakdown are EXPLORATORY, POST HOC (designed 2026-11-06 after the A3 results, following a GPT model
review). Standard library only.

    python scripts/expA_a3_explore.py --items items.jsonl --run-dir-off OFF --run-dir-on ON \
        --scores-off OFF/scores.jsonl --scores-on ON/scores.jsonl --out DIR

1. Accounting bound, per arm: G* = rec(complete) - rec*(incomplete), where rec* counts every completed incomplete-list
   episode that did not submit as if it had recovered C*. A positive G* means non-submission cannot explain the whole
   omission cost. Item means over the seeds, family means, equal weight over families, 95% t interval.
2. Ledger subgroup, per arm and list: episodes that reached the last response (an evidence_ledger event) whose ledger
   lists an entry inside a C* call-site span (path equal, lineno <= line <= end_lineno); recovery within it (micro).
3. Loss breakdown, per arm, incomplete list: each completed episode that did not recover C* is NoAnswer, or Submitted
   with C* never displayed, or Submitted after C* was displayed; Submitted ones are also split by whether the accepted
   submission came before the last response (call index < 7) or at it. "Displayed" is the STRICT rule of
   reports/expA-main-20261101b/timeline/RULES.md (S-text or S-struct, with expA_timeline.py's own functions) on any
   model-visible tool result; it is a lower bound on display and shows availability, not recognition.

Revision b (2026-11-06, after a review): the first version matched raw "PATH:LINE" substrings only, so search
hits (JSON path/line fields) were missed and PATH:12 also matched PATH:123; it is superseded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from expA_reanalyze_independent import t_statistics  # noqa: E402
from expA_timeline import s_struct_hit, s_text_hit, unit_from_text  # noqa: E402

LISTS = {"vcomplete": "complete", "omit_node": "incomplete"}
LAST_CALL_INDEX = 7   # the 8th response (budget 8), where the deadline line is appended


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_bytes().splitlines() if line.strip()]


def cstar_sites(items_path):
    out = {}
    for it in read_jsonl(items_path):
        cstar = it["omitted"]["identity"]
        out[it["item_id"]] = [(s["path"], s["lineno"], s.get("end_lineno") or s["lineno"])
                              for v in it["V"] if v["identity"] == cstar for s in v["sites"]]
    return out


def terminal_attempt(row):
    attempts = row.get("attempts") or []
    return row.get("attempt_id") or (attempts[-1]["attempt_id"] if attempts else None)


def events_of(run_dir, row):
    path = Path(run_dir) / "cells" / row["cell_id"] / terminal_attempt(row) / "events.jsonl"
    return read_jsonl(path)


def displayed(content, sites):
    """The timeline's STRICT rule (S-text or S-struct) on one model-visible tool result."""
    strings, pairs = unit_from_text(content, is_json=True)
    return s_text_hit(strings, sites) or s_struct_hit(pairs, sites)


def family_ci(values_by_item, family):
    groups = defaultdict(list)
    for item, value in values_by_item.items():
        groups[family[item]].append(value)
    means = [sum(v, Fraction(0)) / len(v) for _, v in sorted(groups.items())]
    st = t_statistics(means, 0.95)
    return {"estimate": float(sum(means, Fraction(0)) / len(means)), "ci_low": st["lower"], "ci_high": st["upper"],
            "n_families": len(means), "n_items": len(values_by_item)}


def analyse(arm, run_dir, scores_path, sites):
    rows = [r for r in read_jsonl(scores_path) if r.get("block") == "LEDGER" and r.get("cell_state") == "completed"]
    family, rec, rec_star = {}, defaultdict(list), defaultdict(list)
    sub = defaultdict(lambda: {"episodes": 0, "recovered": 0})
    loss = {"not_recovered": 0, "no_answer": 0, "submitted_cstar_never_displayed": 0,
            "submitted_after_cstar_displayed": 0, "never_displayed_submitted_before_last": 0,
            "never_displayed_submitted_at_last": 0}
    reached = defaultdict(int)
    for r in rows:
        lst, item = LISTS[r["variant"]], r["item_id"]
        family[item] = r["family"]
        submitted = r.get("exit_status") == "Submitted"
        got = int(bool(r.get("site_recovered"))) if submitted else 0
        rec[(lst, item)].append(got)
        rec_star[(lst, item)].append(got if submitted else 1)
        events = None
        if lst == "incomplete" and not got:
            events = events_of(run_dir, r)
            loss["not_recovered"] += 1
            if not submitted:
                loss["no_answer"] += 1
            else:
                seen = any(e["kind"] == "model_visible_tool_result" and displayed(e["value"].get("content", ""),
                                                                                    sites[item]) for e in events)
                loss["submitted_after_cstar_displayed" if seen else "submitted_cstar_never_displayed"] += 1
                if not seen:
                    at = [e["value"]["call_index"] for e in events if e["kind"] == "submission_accepted"]
                    last = bool(at) and at[-1] >= LAST_CALL_INDEX
                    loss["never_displayed_submitted_at_last" if last else "never_displayed_submitted_before_last"] += 1
        if events is None:
            events = events_of(run_dir, r)
        ledgers = [e["value"] for e in events if e["kind"] == "evidence_ledger"]
        if ledgers:
            reached[lst] += 1
            entries = ledgers[0].get("entries") or []
            if any(e["path"] == p and lo <= e["line"] <= hi for e in entries for p, lo, hi in sites[item]):
                sub[lst]["episodes"] += 1
                sub[lst]["recovered"] += got
    mean = lambda v: Fraction(sum(v), len(v))
    items = sorted({i for (_, i) in rec})
    g = {i: mean(rec[("complete", i)]) - mean(rec[("incomplete", i)]) for i in items
         if rec[("complete", i)] and rec[("incomplete", i)]}
    g_star = {i: mean(rec[("complete", i)]) - mean(rec_star[("incomplete", i)]) for i in items
              if rec[("complete", i)] and rec_star[("incomplete", i)]}
    for lst in sub:
        n = sub[lst]["episodes"]
        sub[lst]["recovery_micro"] = sub[lst]["recovered"] / n if n else None
    return {"arm": arm, "completed_rows": len(rows), "omission_cost": family_ci(g, family),
            "omission_cost_crediting_nonsubmitted": family_ci(g_star, family),
            "reached_last_response": dict(reached),
            "ledger_contains_cstar_subgroup": {k: dict(v) for k, v in sub.items()},
            "incomplete_list_losses": loss}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for a in ("--items", "--run-dir-off", "--run-dir-on", "--scores-off", "--scores-on", "--out"):
        ap.add_argument(a, required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        sys.exit(f"refusing: {out} exists")
    sites = cstar_sites(a.items)
    res = {"label": ("Subgroup PRESPECIFIED as descriptive (protocols/EXPA_A3_LEDGER.md section 3); bound and loss "
                     "breakdown EXPLORATORY, POST HOC (2026-11-06); ANALYSIS RECORD"),
           "script_sha256": sha256(__file__),
           "inputs": {k: {"path": v, "sha256": sha256(v)} for k, v in
                      (("items", a.items), ("scores_off", a.scores_off), ("scores_on", a.scores_on))},
           "arms": {"off": analyse("off", a.run_dir_off, a.scores_off, sites),
                    "on": analyse("on", a.run_dir_on, a.scores_on, sites)}}
    out.mkdir(parents=True)
    (out / "A3_EXPLORE.json").write_bytes((json.dumps(res, indent=1, sort_keys=True) + "\n").encode())
    print(json.dumps(res["arms"], indent=1))


if __name__ == "__main__":
    main()
