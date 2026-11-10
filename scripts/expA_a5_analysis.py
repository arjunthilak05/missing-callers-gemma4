#!/usr/bin/env python3
"""Exp A follow-up A5 (submit-time review): the prespecified analysis (protocols/EXPA_A5_REVIEW.md).

PRESPECIFIED: written before any A5 cell exists, with the structure of scripts/expA_a3_analysis.py (A3, whose helpers
it reuses). ANALYSIS RECORD. Standard library only; exact rational means.

    python scripts/expA_a5_analysis.py --neutral NEUTRAL/scores.jsonl --ledger LEDGER/scores.jsonl
        [--neutral-run-json NEUTRAL/run.json --ledger-run-json LEDGER/run.json] --out DIR

Inputs are the scorer's rows (scripts/expA_score.py, design a3-ledger) of the two A5 runs: the neutral-review run and
the ledger-review run. Only block LEDGER rows are read (condition C1; variant vcomplete = the complete list,
omit_node = the incomplete list). With --*-run-json each run.json must record its arm (episode_config.submit_review),
the deadline notice and no evidence ledger.

Recovery = site_recovered of completed cells, scored on the answer the harness used (result.json submitted: the
resubmission, or the first submission when none followed; a completed episode that never submitted scores 0). Cells
missing for infrastructure are excluded and counted. Seeds: the longest prefix of 61, 73, 89 whose LEDGER cells are
all terminal (completed or missing_infrastructure) in BOTH runs (the D11.3 rule, as in A3).

Per item and arm (variant x review): the mean over the seeds of its completed cells. An item enters when all four
arms have a mean. Then, per item:
    G_neutral = rec(complete, neutral) - rec(incomplete, neutral)   (omission cost under the neutral review)
    G_ledger  = rec(complete, ledger)  - rec(incomplete, ledger)    (omission cost under the ledger review)
    M         = G_neutral - G_ledger                                (primary: the reduction in omission cost)
then the family mean over its items, then the equal-weight mean over families, with a two-sided one-sample t test
(df = F - 1, scripts/expA_reanalyze_independent.t_statistics; alpha 0.05; the single primary test).

Secondary (nominal 95% t intervals over family means, no significance claims):
    S1 rec(incomplete, ledger) - rec(incomplete, neutral)
    S2 rec(complete, ledger) - rec(complete, neutral), with the non-inferiority margin -3 points (lower bound > -0.03)
    S3 G_neutral and G_ledger; the four arm levels
    S4 submission rates (exit Submitted) per arm and their ledger - neutral differences
    S5 precision bounds (precision_lower, precision_upper) per arm over submitted completed cells
    S6 exits per arm (counts)
Descriptive (counts over completed cells at the seeds, from the scorer's submit_review fields): the share of
episodes reviewed; answer sources; the share whose FIRST submission omitted C* (first_submission.site_recovered 0);
and, among reviewed episodes that resubmitted after such a first submission, how often the resubmission added a C*
site (the final site_recovered 1), with the reverse (dropped) count.
Exit codes: 0 written; 2 refused (inputs unbound or unusable; nothing written).
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import sys

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
_SPEC = importlib.util.spec_from_file_location("expA_a3_analysis", _HERE / "expA_a3_analysis.py")
A3 = sys.modules.get("expA_a3_analysis")
if A3 is None:
    A3 = importlib.util.module_from_spec(_SPEC)
    sys.modules["expA_a3_analysis"] = A3
    _SPEC.loader.exec_module(A3)

Refused, _require, sha256_bytes = A3.Refused, A3._require, A3.sha256_bytes
family_summary, seed_prefix = A3.family_summary, A3.seed_prefix

SCHEMA = "expA-a5-analysis-v1"
PROTOCOL = "protocols/EXPA_A5_REVIEW.md"
SEEDS = A3.SEEDS                      # (61, 73, 89)
VARIANTS = A3.VARIANTS                # {"complete": "vcomplete", "incomplete": "omit_node"}
REVIEWS = ("neutral", "ledger")
ARMS = tuple((answer, review) for answer in VARIANTS for review in REVIEWS)
ALPHA = A3.ALPHA
NONINFERIORITY_MARGIN = A3.NONINFERIORITY_MARGIN
LABEL = ("PRESPECIFIED (protocols/EXPA_A5_REVIEW.md, written before any A5 cell); ANALYSIS RECORD"
         "")


def load_rows(path, review):
    """The LEDGER rows of one run's scores (A3's loader), labelled with the review arm."""
    run = A3.load_rows(path, review)
    run["review"] = run.pop("ledger")
    return run


def check_run_json(path, review):
    """A run.json must record its arm: the deadline notice, submit_review == review and no evidence ledger."""
    raw = Path(path).read_bytes()
    config = json.loads(raw).get("episode_config") or {}
    _require(config.get("final_turn_policy") == "deadline", f"{path}: episode_config is not the deadline notice")
    _require(config.get("submit_review") == review, f"{path}: episode_config.submit_review is "
                                                    f"{config.get('submit_review')!r}, not {review!r}")
    _require(config.get("evidence_ledger", "none") == "none", f"{path}: an A5 run has no last-call evidence ledger")
    _require(isinstance(config.get("submit_review_rules"), dict)
             and config["submit_review_rules"].get("protocol") == PROTOCOL, f"{path}: no A5 submit_review_rules")
    return {"path": str(path), "sha256": sha256_bytes(raw), "submit_review": review}


def item_means(runs, seeds, field="site_recovered", only_submitted=False):
    """{arm: {item: mean over the seeds' completed cells}} and the items' families (A3's rule, arms by review)."""
    means, family = {}, {}
    for answer, variant in VARIANTS.items():
        for run in runs:
            values = defaultdict(list)
            for (item, row_variant, seed), row in run["rows"].items():
                family.setdefault(item, row["family"])
                _require(family[item] == row["family"], f"{item}: families differ between rows")
                if row_variant != variant or seed not in seeds or row["cell_state"] != "completed":
                    continue
                if only_submitted and row.get("exit_status") != "Submitted":
                    continue
                value = row.get(field)
                if field == "submitted":
                    value = int(row.get("exit_status") == "Submitted")
                if field == "site_recovered" and row.get("exit_status") != "Submitted":
                    value = 0                       # a completed episode that never submitted scores 0
                if value is None:
                    continue
                values[item].append(Fraction(value))
            means[(answer, run["review"])] = {item: sum(vals, Fraction(0)) / len(vals)
                                              for item, vals in values.items()}
    return means, family


def contrasts(means, family):
    """M, G_neutral, G_ledger, S1, S2 and the arm levels over the items that have all four arms."""
    items = sorted(set.intersection(*(set(means[arm]) for arm in ARMS)))
    rec = {arm: {item: means[arm][item] for item in items} for arm in ARMS}
    g_neutral = {i: rec[("complete", "neutral")][i] - rec[("incomplete", "neutral")][i] for i in items}
    g_ledger = {i: rec[("complete", "ledger")][i] - rec[("incomplete", "ledger")][i] for i in items}
    out = {"items_with_all_four_arms": len(items),
           "M_reduction_in_omission_cost": family_summary({i: g_neutral[i] - g_ledger[i] for i in items}, family),
           "G_neutral_omission_cost_neutral_review": family_summary(g_neutral, family),
           "G_ledger_omission_cost_ledger_review": family_summary(g_ledger, family),
           "S1_incomplete_ledger_minus_neutral": family_summary(
               {i: rec[("incomplete", "ledger")][i] - rec[("incomplete", "neutral")][i] for i in items}, family),
           "S2_complete_ledger_minus_neutral": family_summary(
               {i: rec[("complete", "ledger")][i] - rec[("complete", "neutral")][i] for i in items}, family),
           "arm_levels": {f"{answer}_{review}": family_summary(rec[(answer, review)], family)
                          for answer, review in ARMS}}
    s2 = out["S2_complete_ledger_minus_neutral"]
    s2["noninferiority_margin"] = float(NONINFERIORITY_MARGIN)
    s2["lower_bound_above_margin"] = (None if s2.get("ci_low") is None
                                      else s2["ci_low"] > float(NONINFERIORITY_MARGIN))
    return out


def per_arm(means, family):
    return {f"{answer}_{review}": family_summary(means[(answer, review)], family) for answer, review in ARMS}


def review_descriptives(runs, seeds):
    """Counts over completed cells at the seeds from the scorer's submit_review fields (descriptive only)."""
    out = {}
    for run in runs:
        for answer, variant in VARIANTS.items():
            rows = [row for (_, v, s), row in sorted(run["rows"].items())
                    if v == variant and s in seeds and row["cell_state"] == "completed"]
            key = f"{answer}_{run['review']}"
            if rows and all(not isinstance(row.get("submit_review"), dict) for row in rows):
                out[key] = {"status": "not_available",
                            "reason": "the score rows carry no submit_review fields (score with the A5 scorer)"}
                continue
            reviews = [row["submit_review"] for row in rows if isinstance(row.get("submit_review"), dict)]
            reviewed = [row for row in rows if (row.get("submit_review") or {}).get("reviewed") is True]
            first = [row for row in rows if isinstance((row.get("submit_review") or {}).get("first_submission"), dict)]
            first_omitted = [row for row in first if row["submit_review"]["first_submission"]["site_recovered"] == 0]
            resubmitted_after_omission = [row for row in first_omitted
                                          if row["submit_review"].get("answer_source") == "resubmit"]
            resubmitted_after_hit = [row for row in first
                                     if row["submit_review"]["first_submission"]["site_recovered"] == 1
                                     and row["submit_review"].get("answer_source") == "resubmit"]
            added = sum(1 for row in resubmitted_after_omission if row.get("site_recovered") == 1)
            dropped = sum(1 for row in resubmitted_after_hit if row.get("site_recovered") == 0)

            def share(part, whole):
                return {"n": part, "of": whole, "share": part / whole if whole else None}
            out[key] = {
                "completed_episodes": len(rows), "with_review_fields": len(reviews),
                "reviewed": share(len(reviewed), len(rows)),
                "review_skipped_last_call": sum(1 for r in reviews if r.get("review_skipped_last_call") is True),
                "answer_source": dict(sorted(Counter(str(r.get("answer_source")) for r in reviews).items())),
                "unresolved_exits": dict(sorted(Counter(r["unresolved_exit"] for r in reviews
                                                        if r.get("unresolved_exit") is not None).items())),
                "first_submission_omitted_cstar": share(len(first_omitted), len(first)),
                "resubmission_added_cstar": share(added, len(resubmitted_after_omission)),
                "resubmission_dropped_cstar": share(dropped, len(resubmitted_after_hit)),
                "note": "descriptive: first_submission = the first well-formed submission (reviewed, or accepted at "
                        "the last call); added = the final answer recovers C* after a first submission without it"}
    return out


def analyse(neutral, ledger, seeds_allowed=SEEDS):
    _require(neutral["items_sha256"] == ledger["items_sha256"], "the two runs name different items files")
    keys_n, keys_l = set(neutral["rows"]), set(ledger["rows"])
    _require(keys_n == keys_l, f"the runs hold different LEDGER cells ({len(keys_n ^ keys_l)} differ)")
    runs = (neutral, ledger)
    seeds = seed_prefix(runs, seeds_allowed)
    out = {"seeds_considered": list(seeds_allowed), "seeds_used": seeds, "items_sha256": neutral["items_sha256"],
           "cell_states": {run["review"]: dict(sorted(Counter(r["cell_state"] for (_, _, s), r in run["rows"].items()
                                                              if s in seeds_allowed).items())) for run in runs},
           "S6_exits": {run["review"]: {answer: dict(sorted(Counter(
               r.get("exit_status") for (_, v, s), r in run["rows"].items()
               if v == variant and s in seeds and r["cell_state"] == "completed").items()))
               for answer, variant in VARIANTS.items()} for run in runs}}
    if not seeds:
        out["status"] = "no_complete_seed"
        return out
    means, family = item_means(runs, seeds)
    out.update(contrasts(means, family))
    out["missing_infrastructure_cells"] = {
        run["review"]: sum(1 for (_, _, s), r in run["rows"].items()
                           if s in seeds and r["cell_state"] == "missing_infrastructure") for run in runs}
    sub, _ = item_means(runs, seeds, field="submitted")
    out["S4_submission"] = per_arm(sub, family)
    out["S4_submission_ledger_minus_neutral"] = {
        answer: family_summary({i: sub[(answer, "ledger")][i] - sub[(answer, "neutral")][i]
                                for i in set(sub[(answer, "ledger")]) & set(sub[(answer, "neutral")])}, family)
        for answer in VARIANTS}
    out["S5_precision"] = {bound: per_arm(item_means(runs, seeds, field=bound, only_submitted=True)[0], family)
                           for bound in ("precision_lower", "precision_upper")}
    out["review_descriptives"] = review_descriptives(runs, seeds)
    m = out["M_reduction_in_omission_cost"]
    p = m.get("p_two_sided")
    out["primary_decision"] = {"test": "two-sided one-sample t over family means of M (df = F - 1)",
                               "alpha": ALPHA, "p_two_sided": p,
                               "reject_null_at_alpha": None if p is None else p <= ALPHA,
                               "direction": None if m.get("estimate") is None else
                               ("the ledger review reduces the omission cost" if m["estimate"] > 0 else
                                "the ledger review does not reduce the omission cost")}
    out["status"] = "analysed"
    return out


def tables(result):
    def fmt(summary):
        if summary.get("estimate") is None:
            return "n/a"
        low, high = summary.get("ci_low"), summary.get("ci_high")
        interval = "" if low is None else f" [{100 * low:+.1f}, {100 * high:+.1f}]"
        return f"{100 * summary['estimate']:+.1f}{interval} (F={summary['n_families']})"
    part = result["main"]
    lines = ["# A5 submit-time review: prespecified analysis", "", result["label"], "",
             f"## main (seeds used {part['seeds_used']})", ""]
    if part.get("status") != "analysed":
        return "\n".join(lines + [f"Status: {part.get('status')}", ""]) + "\n"
    lines += ["| Quantity | Points [95% t] |", "|---|---|"]
    for key in ("M_reduction_in_omission_cost", "G_neutral_omission_cost_neutral_review",
                "G_ledger_omission_cost_ledger_review", "S1_incomplete_ledger_minus_neutral",
                "S2_complete_ledger_minus_neutral"):
        lines.append(f"| {key} | {fmt(part[key])} |")
    for arm, summary in part["arm_levels"].items():
        lines.append(f"| recovery {arm} | {fmt(summary)} |")
    for arm, summary in part["S4_submission"].items():
        lines.append(f"| submitted {arm} | {fmt(summary)} |")
    d = part["primary_decision"]
    s2 = part["S2_complete_ledger_minus_neutral"]
    lines += ["", f"Primary: p = {d['p_two_sided']}, reject at 0.05: {d['reject_null_at_alpha']} ({d['direction']}).",
              f"S2 lower bound above the -3 point margin: {s2.get('lower_bound_above_margin')}.", "",
              "| Arm | Reviewed | First submission omitted C* | Resubmission added C* | Answer sources |",
              "|---|---|---|---|---|"]
    for arm, entry in part["review_descriptives"].items():
        if entry.get("status") == "not_available":
            lines.append(f"| {arm} | n/a | n/a | n/a | n/a |")
            continue

        def ratio(value):
            return f"{value['n']}/{value['of']}"
        lines.append(f"| {arm} | {ratio(entry['reviewed'])} | {ratio(entry['first_submission_omitted_cstar'])} | "
                     f"{ratio(entry['resubmission_added_cstar'])} | {entry['answer_source']} |")
    return "\n".join(lines) + "\n"


def run(args):
    result = {"label": LABEL, "protocol": PROTOCOL, "schema": SCHEMA,
              "script_sha256": sha256_bytes(Path(__file__).read_bytes()), "inputs": {}}
    neutral, ledger = load_rows(args.neutral, "neutral"), load_rows(args.ledger, "ledger")
    _require(neutral["sha256"] != ledger["sha256"], "the two score inputs are byte-identical")
    result["inputs"]["main"] = {run["review"]: {"path": run["path"], "sha256": run["sha256"]}
                                for run in (neutral, ledger)}
    if args.neutral_run_json or args.ledger_run_json:
        _require(args.neutral_run_json and args.ledger_run_json, "give both --neutral-run-json and --ledger-run-json")
        result["inputs"]["main_run_json"] = {"neutral": check_run_json(args.neutral_run_json, "neutral"),
                                             "ledger": check_run_json(args.ledger_run_json, "ledger")}
    result["main"] = analyse(neutral, ledger)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--neutral", required=True, help="scores.jsonl of the neutral-review run")
    parser.add_argument("--ledger", required=True, help="scores.jsonl of the ledger-review run")
    parser.add_argument("--neutral-run-json")
    parser.add_argument("--ledger-run-json")
    parser.add_argument("--out", required=True, help="output directory (must not exist)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.out)
    if out.exists():
        print(f"refused: {out} exists", file=sys.stderr)
        return 2
    try:
        result = run(args)
    except (Refused, OSError, json.JSONDecodeError) as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    out.mkdir(parents=True)
    (out / "A5_ANALYSIS.json").write_bytes((json.dumps(result, indent=1, sort_keys=True) + "\n").encode())
    (out / "A5_TABLES.md").write_bytes(tables(result).encode())
    print(tables(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
