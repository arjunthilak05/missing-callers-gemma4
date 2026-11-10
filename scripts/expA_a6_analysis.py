#!/usr/bin/env python3
"""Exp A follow-up A6 (doubled budget): the prespecified analysis of protocols/EXPA_A6_BUDGET16.md section 3.

PRESPECIFIED: written with the protocol, before any A6 cell exists. Standard library only; exact rational means.

    python scripts/expA_a6_analysis.py --scores-16 A6/scores.jsonl --scores-8 A3OFF/scores.jsonl
        [--run-json-16 A6/run.json --run-json-8 A3OFF/run.json] --out DIR

Inputs are the scorer's rows (scripts/expA_score.py, design a3-ledger) of two runs of A3's ledger-off arm: the A6 run
(16 responses, 32,768 tokens; a6budget16-20261106a) and the 8-response reference (8 responses, 16,384 tokens;
a3ledger-off-20261104a). Only block LEDGER rows are read (condition C1; variant vcomplete = the complete list,
omit_node = the incomplete list); every other block must be undispatched. With --run-json-16 and --run-json-8 (both
or neither) each run.json must record the deadline notice, evidence_ledger off and its budget (max_model_calls 16 and
context_budget 32768, or 8 and 16384), with the notice and the ledger at call_index max_model_calls - 1.

Recovery = site_recovered of completed cells (a completed episode that did not submit scores 0). Cells missing for
infrastructure are excluded and counted. Primary seeds: the longest prefix of 61, 73, 89 whose LEDGER cells are all
terminal (completed or missing_infrastructure) in BOTH runs.

Per item, budget (8, 16) and list (complete, incomplete): the mean over the primary seeds of its completed cells. An
item enters when all four values exist. Then, per item:
    G8  = rec(complete, 8)  - rec(incomplete, 8)        (omission cost at 8 responses)
    G16 = rec(complete, 16) - rec(incomplete, 16)       (omission cost at 16 responses)
    R   = G8 - G16                                      (primary: the reduction of the omission cost)
then the family mean over its items, then the equal-weight mean over families, with a two-sided one-sample t test
(df = F - 1, scripts/expA_reanalyze_independent.t_statistics; alpha 0.05; the single primary test).

Secondary (nominal 95% t intervals over family means, no significance claims), each over the items that enter:
    G8 and G16; rec(incomplete, 16) - rec(incomplete, 8); rec(complete, 16) - rec(complete, 8); the four levels;
    submission (exit Submitted) per arm and its 16 - 8 difference per list;
    the share of completed episodes that reached the last response, per arm and 16 - 8 per list;
    responses used (model calls) per arm and 16 - 8 per list;
    precision bounds (precision_lower, precision_upper) per arm over submitted completed cells;
    exits per arm (counts, with ContextLimit at the last call).
"Reached the last response" is read from the scorer row (the scorer keeps no final_turn_notice receipt): a completed
episode with N = max_model_calls reached its last call, and so had the deadline notice appended, exactly when
cost.calls == N (the last response was delivered: Submitted, NoAnswer or FormatExhausted at the last call) or
cost.calls == N - 1 with exit ContextLimit (the limit fell at the last call, after the notice). An earlier exit
(a submission or FormatExhausted before the last call, or ContextLimit at an earlier call) did not. Rows without an
integer cost.calls are left out of these two quantities and counted.
Exit codes: 0 written; 2 refused (inputs unbound or unusable, or DIR exists; nothing written).
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from expA_reanalyze_independent import t_statistics  # noqa: E402

PROTOCOL = "protocols/EXPA_A6_BUDGET16.md"
REFERENCE_PROTOCOL = "protocols/EXPA_A3_LEDGER.md"
BLOCK = "LEDGER"
CONDITION = "C1"
SEEDS = (61, 73, 89)
VARIANTS = {"complete": "vcomplete", "incomplete": "omit_node"}
BUDGETS = {16: 32768, 8: 16384}        # max_model_calls -> context_budget (protocols/EXPA_BUDGET16.md; the prereg)
ARMS = tuple((answer, calls) for answer in VARIANTS for calls in BUDGETS)
TERMINAL = frozenset({"completed", "missing_infrastructure"})
CELL_STATES = frozenset({"completed", "missing_infrastructure", "infra_pending", "not_dispatched"})
ALPHA = 0.05
CONFIDENCE = 0.95
LABEL = ("PRESPECIFIED (protocols/EXPA_A6_BUDGET16.md, written before any A6 cell); ANALYSIS RECORD"
         "")
REACHED_RULE = ("completed and (cost.calls == max_model_calls, or cost.calls == max_model_calls - 1 with exit "
                "ContextLimit): the episode was open at its last call, where the deadline notice is appended")


class Refused(ValueError):
    """The inputs cannot be analysed as the protocol states (nothing is written)."""


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _require(condition, message):
    if not condition:
        raise Refused(message)


def _calls(row):
    cost = row.get("cost")
    value = cost.get("calls") if isinstance(cost, dict) else None
    return value if type(value) is int and value >= 0 else None


def load_rows(path, calls):
    """The LEDGER rows of one run's scores (a run of ``calls`` responses), keyed by (item_id, variant, seed)."""
    raw = Path(path).read_bytes()
    rows, other_blocks, items_sha = {}, Counter(), set()
    for number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        _require(isinstance(row, dict), f"{path}:{number}: not an object")
        if row.get("block") != BLOCK:
            other_blocks[row.get("cell_state")] += 1
            continue
        where = f"{path}:{number} ({row.get('cell_id')})"
        _require(row.get("condition") == CONDITION and row.get("variant") in VARIANTS.values()
                 and row.get("seed") in SEEDS, f"{where}: not a LEDGER cell of the A3 design")
        _require(isinstance(row.get("item_id"), str) and isinstance(row.get("family"), str), f"{where}: item/family")
        _require(row.get("cell_state") in CELL_STATES, f"{where}: unknown cell_state {row.get('cell_state')!r}")
        if row["cell_state"] == "completed":
            _require(row.get("site_recovered") in (0, 1), f"{where}: a completed cell needs site_recovered 0 or 1")
            used = _calls(row)
            _require(used is None or used <= calls, f"{where}: cost.calls {used} exceeds the {calls}-response budget")
        key = (row["item_id"], row["variant"], row["seed"])
        _require(key not in rows, f"{where}: duplicate row")
        rows[key] = row
        items_sha.add(row.get("items_sha256"))
    _require(rows, f"{path}: no LEDGER rows (is this an A3-design run scored under design a3-ledger?)")
    _require(len(items_sha) == 1, f"{path}: rows name {len(items_sha)} items files")
    _require(set(other_blocks) <= {"not_dispatched"}, f"{path}: cells outside block LEDGER were dispatched "
                                                      f"({dict(other_blocks)}); an A3-design run dispatches LEDGER only")
    return {"calls": calls, "path": str(path), "sha256": sha256_bytes(raw), "rows": rows,
            "items_sha256": items_sha.pop(), "other_block_rows": sum(other_blocks.values())}


def check_run_json(path, calls):
    """A run.json must record A3's off arm at its budget: the deadline notice, evidence_ledger off, max_model_calls
    ``calls`` and its context budget, and the notice and the ledger at call_index calls - 1."""
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    _require(isinstance(value, dict), f"{path}: not an object")
    config = value.get("episode_config")
    _require(isinstance(config, dict), f"{path}: no episode_config")
    _require(config.get("final_turn_policy") == "deadline", f"{path}: episode_config is not the deadline notice")
    _require(config.get("evidence_ledger") == "off", f"{path}: episode_config.evidence_ledger is "
                                                     f"{config.get('evidence_ledger')!r}, not 'off'")
    budget = (config.get("max_model_calls"), config.get("context_budget"))
    _require(budget == (calls, BUDGETS[calls]), f"{path}: episode_config records max_model_calls {budget[0]!r} and "
                                                f"context_budget {budget[1]!r}, not {calls} and {BUDGETS[calls]}")
    for name in ("final_turn_notice", "evidence_ledger_rules"):
        record = config.get(name)
        _require(isinstance(record, dict) and record.get("call_index") == calls - 1,
                 f"{path}: episode_config.{name}.call_index is not {calls - 1}")
    return {"path": str(path), "sha256": sha256_bytes(raw), "max_model_calls": calls, "context_budget": BUDGETS[calls],
            "final_turn_policy": "deadline", "evidence_ledger": "off", "run_id": value.get("run_id")}


def seed_prefix(runs, seeds):
    """The longest prefix of ``seeds`` whose LEDGER cells are all terminal in every run (D11.3)."""
    prefix = []
    for seed in seeds:
        states = [row["cell_state"] for run in runs for (_, _, s), row in run["rows"].items() if s == seed]
        if not states or not set(states) <= TERMINAL:
            break
        prefix.append(seed)
    return prefix


def summarize(values):
    """Equal-weight mean over family values with its two-sided one-sample t summary."""
    values = list(values)
    if not values:
        return {"n_families": 0, "estimate": None, "status": "no_families"}
    st = t_statistics(values, CONFIDENCE)
    est = st["mean_exact"]
    return {"n_families": len(values), "estimate": float(est), "estimate_exact": f"{est.numerator}/{est.denominator}",
            "ci_low": st.get("lower"), "ci_high": st.get("upper"), "df": st["df"], "t": st.get("t"),
            "p_two_sided": st.get("p_two_sided"), "status": st["status"]}


def family_summary(per_item, family):
    by_family = defaultdict(list)
    for item, value in per_item.items():
        by_family[family[item]].append(value)
    means = {fam: sum(vals, Fraction(0)) / len(vals) for fam, vals in sorted(by_family.items())}
    out = summarize(means.values())
    out["n_items"] = len(per_item)
    out["family_means"] = {fam: float(value) for fam, value in means.items()}
    return out


def reached_last(row, calls):
    """The episode was open at its last call (see the module docstring); None when cost.calls is unavailable."""
    used = _calls(row)
    if used is None:
        return None
    return int(used == calls or (used == calls - 1 and row.get("exit_status") == "ContextLimit"))


def cell_value(row, field, calls):
    if field == "site_recovered":
        return row.get("site_recovered")
    if field == "submitted":
        return int(row.get("exit_status") == "Submitted")
    if field == "reached_last":
        return reached_last(row, calls)
    if field == "responses":
        return _calls(row)
    if row.get("exit_status") != "Submitted":         # precision bounds: submitted completed cells only
        return None
    return row.get(field)


def item_means(runs, seeds, field="site_recovered"):
    """{(answer, calls): {item: mean over the seeds' completed cells}} and the items' families."""
    means, family = {}, {}
    for answer, variant in VARIANTS.items():
        for run in runs:
            values = defaultdict(list)
            for (item, row_variant, seed), row in run["rows"].items():
                family.setdefault(item, row["family"])
                _require(family[item] == row["family"], f"{item}: families differ between rows")
                if row_variant != variant or seed not in seeds or row["cell_state"] != "completed":
                    continue
                value = cell_value(row, field, run["calls"])
                if value is None:
                    continue
                values[item].append(Fraction(value))
            means[(answer, run["calls"])] = {item: sum(vals, Fraction(0)) / len(vals)
                                             for item, vals in values.items()}
    return means, family


def arm_name(answer, calls):
    return f"{answer}_{calls}"


def per_arm(means, family, items):
    return {arm_name(answer, calls): family_summary({i: means[(answer, calls)][i] for i in items
                                                     if i in means[(answer, calls)]}, family)
            for answer, calls in ARMS}


def budget_differences(means, family, items):
    """Per list: value(16) - value(8) over the entering items that have both."""
    return {answer: family_summary({i: means[(answer, 16)][i] - means[(answer, 8)][i] for i in items
                                    if i in means[(answer, 16)] and i in means[(answer, 8)]}, family)
            for answer in VARIANTS}


def contrasts(means, family):
    """R, G8, G16, the two budget gains and the four levels over the items that have all four values."""
    items = sorted(set.intersection(*(set(means[arm]) for arm in ARMS)))
    rec = {arm: {item: means[arm][item] for item in items} for arm in ARMS}
    g8 = {i: rec[("complete", 8)][i] - rec[("incomplete", 8)][i] for i in items}
    g16 = {i: rec[("complete", 16)][i] - rec[("incomplete", 16)][i] for i in items}
    out = {"items_with_all_four_values": len(items),
           "R_reduction_in_omission_cost": family_summary({i: g8[i] - g16[i] for i in items}, family),
           "G8_omission_cost_8_responses": family_summary(g8, family),
           "G16_omission_cost_16_responses": family_summary(g16, family),
           "incomplete_16_minus_8": family_summary(
               {i: rec[("incomplete", 16)][i] - rec[("incomplete", 8)][i] for i in items}, family),
           "complete_16_minus_8": family_summary(
               {i: rec[("complete", 16)][i] - rec[("complete", 8)][i] for i in items}, family),
           "levels": {arm_name(answer, calls): family_summary(rec[(answer, calls)], family)
                      for answer, calls in ARMS}}
    gain_incomplete = out["incomplete_16_minus_8"].get("estimate")
    gain_complete = out["complete_16_minus_8"].get("estimate")
    out["incomplete_gain_exceeds_complete_gain"] = (None if gain_incomplete is None or gain_complete is None
                                                    else gain_incomplete > gain_complete)
    return out, items


def exits(run, seeds):
    """Exit counts of completed cells per list, with ContextLimit at the last call (cost.calls == N - 1)."""
    out = {}
    for answer, variant in VARIANTS.items():
        rows = [r for (_, v, s), r in run["rows"].items() if v == variant and s in seeds
                and r["cell_state"] == "completed"]
        counts = dict(sorted(Counter(r.get("exit_status") for r in rows).items()))
        counts["ContextLimit_at_last_call"] = sum(1 for r in rows if r.get("exit_status") == "ContextLimit"
                                                  and _calls(r) == run["calls"] - 1)
        out[answer] = counts
    return out


def analyse(sixteen, eight):
    _require(sixteen["items_sha256"] == eight["items_sha256"], "the two runs name different items files")
    keys16, keys8 = set(sixteen["rows"]), set(eight["rows"])
    _require(keys16 == keys8, f"the runs hold different LEDGER cells ({len(keys16 ^ keys8)} differ)")
    runs = (sixteen, eight)
    seeds = seed_prefix(runs, SEEDS)
    out = {"seeds_considered": list(SEEDS), "seeds_used": seeds, "items_sha256": sixteen["items_sha256"],
           "cell_states": {str(run["calls"]): dict(sorted(Counter(
               r["cell_state"] for (_, _, s), r in run["rows"].items() if s in SEEDS).items())) for run in runs}}
    if not seeds:
        out["status"] = "no_complete_seed"
        return out
    out["exits"] = {str(run["calls"]): exits(run, seeds) for run in runs}
    means, family = item_means(runs, seeds)
    found, items = contrasts(means, family)
    out.update(found)
    out["missing_infrastructure_cells"] = {
        str(run["calls"]): sum(1 for (_, _, s), r in run["rows"].items()
                               if s in seeds and r["cell_state"] == "missing_infrastructure") for run in runs}
    out["completed_cells_without_calls"] = {
        str(run["calls"]): sum(1 for (_, _, s), r in run["rows"].items()
                               if s in seeds and r["cell_state"] == "completed" and _calls(r) is None) for run in runs}
    for name, field in (("submission", "submitted"), ("reached_last_response", "reached_last"),
                        ("responses_used", "responses")):
        values, _ = item_means(runs, seeds, field=field)
        out[name] = per_arm(values, family, items)
        out[name + "_16_minus_8"] = budget_differences(values, family, items)
    out["reached_last_response_rule"] = REACHED_RULE
    out["precision"] = {bound: per_arm(item_means(runs, seeds, field=bound)[0], family, items)
                        for bound in ("precision_lower", "precision_upper")}
    r = out["R_reduction_in_omission_cost"]
    p = r.get("p_two_sided")
    out["primary_decision"] = {"test": "two-sided one-sample t over family means of R = G8 - G16 (df = F - 1)",
                               "alpha": ALPHA, "p_two_sided": p,
                               "reject_null_at_alpha": None if p is None else p <= ALPHA,
                               "direction": None if r.get("estimate") is None else
                               ("the omission cost is smaller at 16 responses" if r["estimate"] > 0 else
                                "the omission cost is not smaller at 16 responses")}
    out["reading"] = ("R is read with the levels and the two budget gains, never alone (protocol section 3); the "
                      "comparison is across runs from different releases, not a randomized within-run contrast")
    out["status"] = "analysed"
    return out


def tables(result):
    def fmt(summary):
        if summary.get("estimate") is None:
            return "n/a"
        low, high = summary.get("ci_low"), summary.get("ci_high")
        interval = "" if low is None else f" [{100 * low:+.1f}, {100 * high:+.1f}]"
        return f"{100 * summary['estimate']:+.1f}{interval} (F={summary['n_families']})"

    def fmt_count(summary):
        if summary.get("estimate") is None:
            return "n/a"
        low, high = summary.get("ci_low"), summary.get("ci_high")
        interval = "" if low is None else f" [{low:.2f}, {high:.2f}]"
        return f"{summary['estimate']:.2f}{interval} (F={summary['n_families']})"
    part = result["main"]
    lines = ["# A6 doubled budget: prespecified analysis", "", result["label"], "",
             f"Seeds used: {part['seeds_used']} (considered {part['seeds_considered']}).", ""]
    if part.get("status") != "analysed":
        return "\n".join(lines + [f"Status: {part.get('status')}", ""]) + "\n"
    lines += [f"Items with all four values: {part['items_with_all_four_values']}.", "",
              "| Quantity | Points [95% t] |", "|---|---|"]
    for key in ("R_reduction_in_omission_cost", "G8_omission_cost_8_responses", "G16_omission_cost_16_responses",
                "incomplete_16_minus_8", "complete_16_minus_8"):
        lines.append(f"| {key} | {fmt(part[key])} |")
    for arm, summary in part["levels"].items():
        lines.append(f"| recovery {arm} | {fmt(summary)} |")
    for name in ("submission", "reached_last_response"):
        for arm, summary in part[name].items():
            lines.append(f"| {name} {arm} | {fmt(summary)} |")
        for answer, summary in part[name + "_16_minus_8"].items():
            lines.append(f"| {name} {answer} 16 - 8 | {fmt(summary)} |")
    for bound in ("precision_lower", "precision_upper"):
        for arm, summary in part["precision"][bound].items():
            lines.append(f"| {bound} {arm} | {fmt(summary)} |")
    lines += ["", "| Responses used | Mean [95% t] |", "|---|---|"]
    for arm, summary in part["responses_used"].items():
        lines.append(f"| {arm} | {fmt_count(summary)} |")
    for answer, summary in part["responses_used_16_minus_8"].items():
        lines.append(f"| {answer} 16 - 8 | {fmt_count(summary)} |")
    lines += ["", "| Exits (completed cells) | 16 responses | 8 responses |", "|---|---|---|"]
    for answer in VARIANTS:
        names = sorted(set(part["exits"]["16"][answer]) | set(part["exits"]["8"][answer]))
        for name in names:
            lines.append(f"| {answer} {name} | {part['exits']['16'][answer].get(name, 0)} | "
                         f"{part['exits']['8'][answer].get(name, 0)} |")
    d = part["primary_decision"]
    lines += ["", f"Primary: R = G8 - G16, p = {d['p_two_sided']}, reject at 0.05: {d['reject_null_at_alpha']} "
                  f"({d['direction']}).",
              f"Incomplete-list gain exceeds complete-list gain: {part['incomplete_gain_exceeds_complete_gain']}.",
              f"Missing for infrastructure (excluded): {part['missing_infrastructure_cells']}.",
              f"Reached the last response: {part['reached_last_response_rule']}.", "", part["reading"] + ".", ""]
    return "\n".join(lines) + "\n"


def run(args):
    result = {"label": LABEL, "protocol": PROTOCOL, "reference_protocol": REFERENCE_PROTOCOL,
              "script_sha256": sha256_bytes(Path(__file__).read_bytes()), "inputs": {}}
    sixteen, eight = load_rows(args.scores_16, 16), load_rows(args.scores_8, 8)
    result["inputs"]["scores"] = {str(run["calls"]): {"path": run["path"], "sha256": run["sha256"]}
                                  for run in (sixteen, eight)}
    if args.run_json_16 or args.run_json_8:
        _require(args.run_json_16 and args.run_json_8, "give both --run-json-16 and --run-json-8")
        result["inputs"]["run_json"] = {"16": check_run_json(args.run_json_16, 16),
                                        "8": check_run_json(args.run_json_8, 8)}
    result["main"] = analyse(sixteen, eight)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scores-16", required=True, help="scores.jsonl of the A6 run (16 responses, 32768 tokens)")
    parser.add_argument("--scores-8", required=True,
                        help="scores.jsonl of the reference A3 ledger-off run (8 responses, 16384 tokens)")
    parser.add_argument("--run-json-16", help="run.json of the A6 run")
    parser.add_argument("--run-json-8", help="run.json of the reference run")
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
    (out / "A6_ANALYSIS.json").write_bytes((json.dumps(result, indent=1, sort_keys=True) + "\n").encode())
    text = tables(result)
    (out / "A6_TABLES.md").write_bytes(text.encode())
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
