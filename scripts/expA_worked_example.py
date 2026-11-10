#!/usr/bin/env python3
"""One worked example: an A3 item-seed pair where only the complete list recovered C*. DESCRIPTIVE, POST HOC.

Designed 2026-11-07 for the supplement (S7), after the A3 results; ANALYSIS RECORD.
Standard library only; run from a directory where scripts/ is importable (or next to the imported scripts:
expA_a3_explore.py, expA_followup_paired.py, expA_timeline.py, expA_reanalyze_independent.py).

    python scripts/expA_worked_example.py --items items.jsonl --run-dir OFF_RUN --scores OFF/scores.jsonl --out DIR

Selection rule (fixed in this docstring before any trajectory content was read; no other pair was inspected):
1. the A3 ledger-off run (protocols/EXPA_A3_LEDGER.md: deadline line, 8 responses, seeds 61/73/89) and its frozen
   scores; rows of block LEDGER with cell_state "completed";
2. item-seed pairs with both the complete list (variant vcomplete) and the incomplete list (omit_node) completed;
3. "complete only" pairs: the complete episode recovered C* and the incomplete one did not, where recovered means
   exit_status Submitted and site_recovered (expA_followup_paired.recovered, the strict scorer's outcome);
4. pairs whose item's omitted caller C* is in the "direct" stratum (items.jsonl omitted.stratum);
5. sorted by (seed, item_id) ascending; the FIRST pair is the worked example.

Output DIR/WORKED_EXAMPLE.json: the rule and candidate counts; library, target, C* and its call sites; the lines of
the first user message present only in the complete episode (and only in the incomplete one), and for a JSON answer
line the entries (by id) and top-level fields that differ; for each episode the
ordered tool calls (name, arguments as compact JSON cut to 120 characters), whether each model-visible tool result
displays a C* call site by the STRICT rule of expA_a3_explore.displayed (S-text or S-struct; a lower bound on display,
availability not recognition), responses used, exit status, the submitted sites with their scorer categories, the
completeness claim and the scores; the incomplete episode's class (expA_followup_paired.classify); every input path
and sha256. Each events.jsonl must have the sha256 its scores row names.

Also descriptive, per tool result: "numbered_lines_naming_target", each output line that carries a number label
(grep -n "N:" or cat -n / nl "N<tab>") and names the target as a token, as [label, text cut to 120 characters]. It
shows which line numbers the agent was shown next to a call of the target; it is not the strict rule.

Revision b (2026-11-07, after reading revision a's output for the selected pair; the rule and the selected pair are
unchanged): the answer block is one JSON line, so the line diff alone showed two whole blocks; the entry-level diff,
the numbered lines, function_localized and the deadline notice's response were added. Revision a is superseded.
Revision c (2026-11-07, same rule and pair): adds n_strict_cstar_display, the count of tool results that satisfy the
strict rule, so the supplement can state it as a number; revision b is superseded.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from expA_a3_explore import cstar_sites, displayed  # noqa: E402
from expA_followup_paired import classify, recovered  # noqa: E402

ARG_CHARS = 120
NUMBERED = re.compile(r"^\s*([1-9][0-9]*)(?::|	)(.*)$")
RULE = ("A3 ledger-off run; completed LEDGER cells; item-seed pairs where the complete list (vcomplete) recovered C* "
        "and the incomplete list (omit_node) did not (recovered = Submitted and site_recovered); C* in the direct "
        "stratum; sorted by (seed, item_id) ascending; the first pair")
IMPORTED = ("expA_a3_explore.py", "expA_followup_paired.py", "expA_timeline.py", "expA_reanalyze_independent.py")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_bytes().splitlines() if line.strip()]


def select(items, scores_path):
    cells = {}
    for r in read_jsonl(scores_path):
        if r.get("block") == "LEDGER" and r.get("cell_state") == "completed":
            cells[(r["item_id"], r["seed"], r["variant"])] = r
    complete_only = []
    for (item, seed, variant), r in cells.items():
        inc = cells.get((item, seed, "omit_node"))
        if variant == "vcomplete" and inc is not None and recovered(r) and not recovered(inc):
            complete_only.append((seed, item))
    direct = sorted(p for p in complete_only if items[p[1]]["omitted"].get("stratum") == "direct")
    counts = {"complete_only_pairs": len(complete_only), "complete_only_pairs_direct_stratum": len(direct)}
    if not direct:
        sys.exit("refusing: no complete-only pair with a direct-stratum C*")
    seed, item = direct[0]
    return seed, item, cells[(item, seed, "vcomplete")], cells[(item, seed, "omit_node")], counts


def compact(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def numbered_lines(content, name):
    """[label, text] for output lines with a number label that name the target as a token (descriptive)."""
    try:
        value = json.loads(content)
    except ValueError:
        return []
    output = value.get("output") if isinstance(value, dict) else None
    if not isinstance(output, str):
        return []
    token = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(name) + r"(?![A-Za-z0-9_])")
    out = []
    for line in output.splitlines():
        m = NUMBERED.match(line)
        if m and token.search(m.group(2)):
            out.append([int(m.group(1)), m.group(2).strip()[:ARG_CHARS]])
    return out


def entry_summary(entry):
    return {k: v for k, v in sorted(entry.items()) if not isinstance(v, (dict, list)) and k != "snippet"}


def answer_diff(only_c, only_i):
    """Entries (by id) and top-level fields that differ between one JSON answer line of each list."""
    def as_dict(lines):
        found = []
        for line in lines:
            try:
                value = json.loads(line)
            except ValueError:
                continue
            if isinstance(value, dict):
                found.append(value)
        return found
    jc, ji = as_dict(only_c), as_dict(only_i)
    if len(jc) != 1 or len(ji) != 1:
        return None
    jc, ji = jc[0], ji[0]
    res = {"entries_only_in_complete": {}, "entries_only_in_incomplete": {}, "fields_differing": {}}
    for key in sorted(set(jc) | set(ji)):
        vc, vi = jc.get(key), ji.get(key)
        if vc == vi:
            continue
        if all(isinstance(v, list) and all(isinstance(x, dict) and "id" in x for x in v) for v in (vc, vi)):
            ids_c, ids_i = {x["id"] for x in vc}, {x["id"] for x in vi}
            res["entries_only_in_complete"][key] = [entry_summary(x) for x in vc if x["id"] not in ids_i]
            res["entries_only_in_incomplete"][key] = [entry_summary(x) for x in vi if x["id"] not in ids_c]
        else:
            res["fields_differing"][key] = {"complete": vc, "incomplete": vi}
    return res


def episode(run_dir, row, sites, target_name):
    path = Path(run_dir) / "cells" / row["cell_id"] / row["attempt_id"] / "events.jsonl"
    digest = sha256(path)
    expected = (row.get("telemetry") or {}).get("events_sha256")
    if expected != digest:
        sys.exit(f"refusing: {path} has sha256 {digest}, its scores row names {expected}")
    events = read_jsonl(path)
    setup = next(e["value"] for e in events if e["kind"] == "episode_setup")
    calls, results, call_index, responses, format_errors, submitted_at, notice = [], {}, None, 0, [], None, None
    for e in events:
        kind, v = e["kind"], e["value"]
        if kind == "model_exchange":
            call_index, responses = v["call_index"], responses + 1
        elif kind == "format_error":
            format_errors.append(v["call_index"])
        elif kind == "tool_action":
            args = compact(v.get("arguments"))
            calls.append({"response": call_index + 1, "tool_call_id": v.get("id"), "name": v.get("name"),
                          "arguments": args[:ARG_CHARS], "arguments_truncated": len(args) > ARG_CHARS,
                          "arguments_chars": len(args)})
        elif kind == "model_visible_tool_result":
            results[v["tool_call_id"]] = {"result_chars": len(v.get("content", "")),
                                          "strict_cstar_display": bool(displayed(v.get("content", ""), sites)),
                                          "numbered_lines_naming_target": numbered_lines(v.get("content", ""),
                                                                                         target_name)}
        elif kind == "final_turn_notice":
            notice = v["call_index"] + 1
        elif kind == "submission_accepted":
            submitted_at = v["call_index"] + 1
    for c in calls:
        c.update(results.get(c["tool_call_id"], {"result_chars": None, "strict_cstar_display": None,
                                                 "numbered_lines_naming_target": None}))
    sites_out = [{"path": d["submitted"].get("path"), "line": d["submitted"].get("line"), "category": d["category"],
                  "cstar_site": bool(d["omitted_site"])} for d in row.get("site_details") or []]
    return {"cell_id": row["cell_id"], "attempt_id": row["attempt_id"],
            "events": {"path": str(path), "sha256": digest},
            "responses_used": responses, "cost_calls": (row.get("cost") or {}).get("calls"),
            "format_error_responses": [i + 1 for i in format_errors], "deadline_notice_at_response": notice,
            "tool_calls": calls, "n_tool_calls": len(calls),
            "any_strict_cstar_display": any(bool(c["strict_cstar_display"]) for c in calls),
            "n_strict_cstar_display": sum(bool(c["strict_cstar_display"]) for c in calls),
            "exit_status": row.get("exit_status"), "submitted": row.get("exit_status") == "Submitted",
            "submitted_at_response": submitted_at, "submitted_sites": sites_out,
            "completeness_claim": row.get("complete"),
            "score": {"recovered": recovered(row), "site_recovered": row.get("site_recovered"),
                      "function_localized": row.get("function_localized"),
                      "precision_lower": row.get("precision_lower"), "precision_upper": row.get("precision_upper"),
                      "n_submitted": row.get("n_submitted"), "n_unique_sites": row.get("n_unique_sites"),
                      "precision_categories": row.get("precision_categories"),
                      "recall_V_callers": row.get("recall_V_callers"), "n_V": row.get("n_V"),
                      "certifiably_false_completeness": row.get("certifiably_false_completeness")},
            "_user": setup["user"], "_system_sha256": setup.get("system_sha256")}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for a in ("--items", "--run-dir", "--scores", "--out"):
        ap.add_argument(a, required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        sys.exit(f"refusing: {out} exists")
    items = {it["item_id"]: it for it in read_jsonl(a.items)}
    seed, item_id, row_c, row_i, counts = select(items, a.scores)
    it = items[item_id]
    sites = cstar_sites(a.items)[item_id]
    cstar = it["omitted"]["identity"]
    name = it["target"]["qualname"].rsplit(".", 1)[-1]
    comp, inc = episode(a.run_dir, row_c, sites, name), episode(a.run_dir, row_i, sites, name)
    user_c, user_i = comp.pop("_user"), inc.pop("_user")
    sys_c, sys_i = comp.pop("_system_sha256"), inc.pop("_system_sha256")
    diff = list(difflib.ndiff(user_i.splitlines(), user_c.splitlines()))
    only_c, only_i = [d[2:] for d in diff if d.startswith("+ ")], [d[2:] for d in diff if d.startswith("- ")]
    here = Path(__file__).resolve().parent
    res = {"label": "DESCRIPTIVE, POST HOC worked example (2026-11-07); ANALYSIS RECORD; "
                    "one pair chosen by a fixed rule, not a representative sample",
           "script_sha256": sha256(__file__),
           "imported_scripts_sha256": {n: sha256(here / n) for n in IMPORTED},
           "inputs": {"items": {"path": a.items, "sha256": sha256(a.items)},
                      "scores_off": {"path": a.scores, "sha256": sha256(a.scores)},
                      "run_dir_off": a.run_dir,
                      "run_json": {"path": str(Path(a.run_dir) / "run.json"),
                                   "sha256": sha256(Path(a.run_dir) / "run.json")}},
           "selection": {"rule": RULE, **counts},
           "pair": {"item_id": item_id, "seed": seed, "library": it["family"],
                    "target": {k: it["target"].get(k) for k in ("identity", "path", "line", "qualname")},
                    "cstar": {"identity": cstar, "qualname": cstar.split("::", 1)[1], "stratum":
                              it["omitted"].get("stratum"),
                              "sites": [{"path": p, "lineno": lo, "end_lineno": hi} for p, lo, hi in sites],
                              "site": f"{sites[0][0]}:{sites[0][1]}"},
                    "n_V": len(it["V"])},
           "first_user_message": {"complete_sha256": hashlib.sha256(user_c.encode()).hexdigest(),
                                  "incomplete_sha256": hashlib.sha256(user_i.encode()).hexdigest(),
                                  "system_prompt_identical": sys_c == sys_i,
                                  "lines_only_in_complete": only_c, "lines_only_in_incomplete": only_i,
                                  "answer_line_diff": answer_diff(only_c, only_i)},
           "episodes": {"complete": comp, "incomplete": inc},
           "incomplete_episode_class": classify(a.run_dir, row_i, sites)}
    out.mkdir(parents=True)
    (out / "WORKED_EXAMPLE.json").write_bytes((json.dumps(res, indent=1, sort_keys=True) + "\n").encode())
    print(json.dumps({k: res[k] for k in ("selection", "pair", "incomplete_episode_class")}, indent=1))


if __name__ == "__main__":
    main()
