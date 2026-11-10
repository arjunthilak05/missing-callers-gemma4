#!/usr/bin/env python3
"""A3 paired omission outcomes and the A4 conditional instruction effect. EXPLORATORY, POST HOC.

Designed 2026-11-06 after the A3 and A4 results, following a design review (task absent-edges-final-20261106).
Standard library only; run from a directory where scripts/ is importable (or next to the imported scripts).

    python scripts/expA_followup_paired.py --items items.jsonl --a3-run-dir OFF_RUN --a3-scores OFF/scores.jsonl \
        --a4-scores-neutral N --a4-scores-info I --a4-scores-submit S --a4-scores-deadline D --out DIR

1. A3 ledger-off run (deadline line, seeds 61, 73, 89): per (item, seed) pair of completed cells, complete vs
   incomplete list: both recovered, complete only, incomplete only, neither. Within "complete only" pairs (the
   omission-harmed pairs), the incomplete episode is classed as: no answer; submitted after a matched display of a
   C* call site; submitted without a matched display, at the last response or before it. "Matched display" is the
   STRICT rule of reports/expA-main-20261101b/timeline/RULES.md (S-text or S-struct, expA_timeline.py's functions)
   on any model-visible tool result: a lower bound on display, availability not recognition.
2. A4 (seed 23, C0 and C1): the instruction's effect conditional on the information, rec(deadline) -
   rec(deadline_info), and without it, rec(submit_only) - rec(neutral), each per item averaged over C0 and C1,
   family means, equal weight over families, 95% t interval (nominal).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from expA_reanalyze_independent import t_statistics  # noqa: E402
from expA_timeline import s_struct_hit, s_text_hit, unit_from_text  # noqa: E402

LAST_CALL_INDEX = 7


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


def recovered(row):
    return int(row.get("exit_status") == "Submitted" and bool(row.get("site_recovered")))


def classify(run_dir, row, sites, last_call_index=LAST_CALL_INDEX):
    if row.get("exit_status") != "Submitted":
        return "no_answer"
    events = read_jsonl(Path(run_dir) / "cells" / row["cell_id"] / row["attempt_id"] / "events.jsonl")
    for e in events:
        if e["kind"] == "model_visible_tool_result":
            strings, pairs = unit_from_text(e["value"].get("content", ""), is_json=True)
            if s_text_hit(strings, sites) or s_struct_hit(pairs, sites):
                return "submitted_after_matched_display"
    at = [e["value"]["call_index"] for e in events if e["kind"] == "submission_accepted"]
    last = bool(at) and at[-1] >= last_call_index
    return "no_matched_display_submitted_at_last" if last else "no_matched_display_submitted_before_last"


def a3_paired(items, run_dir, scores_path, last_call_index=LAST_CALL_INDEX):
    sites = cstar_sites(items)
    cells = {}
    for r in read_jsonl(scores_path):
        if r.get("block") == "LEDGER" and r.get("cell_state") == "completed":
            cells[(r["item_id"], r["seed"], r["variant"])] = r
    pairs, harmed = Counter(), Counter()
    for (item, seed, variant), r in sorted(cells.items()):
        if variant != "vcomplete" or (item, seed, "omit_node") not in cells:
            continue
        inc = cells[(item, seed, "omit_node")]
        a, b = recovered(r), recovered(inc)
        pairs[{(1, 1): "both", (1, 0): "complete_only", (0, 1): "incomplete_only", (0, 0): "neither"}[(a, b)]] += 1
        if a and not b:
            harmed[classify(run_dir, inc, sites[item], last_call_index)] += 1
            if inc.get("exit_status") == "Submitted":
                calls = (inc.get("cost") or {}).get("calls")
                harmed["submitted_before_last_response" if isinstance(calls, int) and calls <= last_call_index
                       else "submitted_at_last_response"] += 1
    return {"pairs": dict(pairs), "complete_only_pairs_by_incomplete_episode": dict(harmed),
            "net_pairs_complete_minus_incomplete": pairs["complete_only"] - pairs["incomplete_only"]}


def a4_conditional(paths):
    rec, family = {}, {}
    for policy, path in paths.items():
        for r in read_jsonl(path):
            if r.get("block") != "P" or r.get("seed") != 23 or r.get("condition") not in ("C0", "C1") \
                    or r.get("cell_state") != "completed":
                continue
            rec[(policy, r["item_id"], r["condition"])] = recovered(r)
            family[r["item_id"]] = r["family"]

    def contrast(a, b):
        diffs = defaultdict(list)
        for item, fam in family.items():
            keys = [(p, item, c) for p in (a, b) for c in ("C0", "C1")]
            if all(k in rec for k in keys):
                d = Fraction(sum(rec[(a, item, c)] - rec[(b, item, c)] for c in ("C0", "C1")), 2)
                diffs[fam].append(d)
        means = [sum(v, Fraction(0)) / len(v) for _, v in sorted(diffs.items())]
        st = t_statistics(means, 0.95)
        return {"estimate": float(sum(means, Fraction(0)) / len(means)), "ci_low": st["lower"],
                "ci_high": st["upper"], "p_two_sided_nominal": st["p_two_sided"], "n_families": len(means),
                "n_items": sum(len(v) for v in diffs.values())}

    return {"instruction_given_information": contrast("deadline", "deadline_info"),
            "instruction_without_information": contrast("submit_only", "neutral")}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for a in ("--items", "--a3-run-dir", "--a3-scores", "--a4-scores-neutral", "--a4-scores-info",
              "--a4-scores-submit", "--a4-scores-deadline", "--out"):
        ap.add_argument(a, required=True)
    ap.add_argument("--a6-run-dir", help="optional: also pair the A6 run (16 responses; last call index 15)")
    ap.add_argument("--a6-scores")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        sys.exit(f"refusing: {out} exists")
    a4_paths = {"neutral": a.a4_scores_neutral, "deadline_info": a.a4_scores_info,
                "submit_only": a.a4_scores_submit, "deadline": a.a4_scores_deadline}
    res = {"label": "EXPLORATORY, POST HOC (2026-11-06, after a design review); ANALYSIS RECORDhuman "
                    "review",
           "script_sha256": sha256(__file__),
           "inputs": {k: {"path": v, "sha256": sha256(v)} for k, v in
                      [("items", a.items), ("a3_scores_off", a.a3_scores)] + [(f"a4_scores_{k}", v)
                                                                               for k, v in a4_paths.items()]},
           "a3_ledger_off_paired": a3_paired(a.items, a.a3_run_dir, a.a3_scores),
           "a4_conditional_instruction": a4_conditional(a4_paths)}
    if a.a6_run_dir and a.a6_scores:
        res["inputs"]["a6_scores"] = {"path": a.a6_scores, "sha256": sha256(a.a6_scores)}
        res["a6_budget16_paired"] = a3_paired(a.items, a.a6_run_dir, a.a6_scores, last_call_index=15)
    out.mkdir(parents=True)
    (out / "FOLLOWUP_PAIRED.json").write_bytes((json.dumps(res, indent=1, sort_keys=True) + "\n").encode())
    print(json.dumps({k: res[k] for k in ("a3_ledger_off_paired", "a4_conditional_instruction", "a6_budget16_paired")
                      if k in res}, indent=1))


if __name__ == "__main__":
    main()
