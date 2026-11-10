#!/usr/bin/env python3
"""Exp A follow-up A3 (evidence ledger): the prespecified analysis of protocols/EXPA_A3_LEDGER.md section 4.

PRESPECIFIED: written with the protocol, before any A3 cell exists. Standard library only; exact rational means.

    python scripts/expA_a3_analysis.py --off OFF/scores.jsonl --on ON/scores.jsonl
        [--off-run-json OFF/run.json --on-run-json ON/run.json]
        [--hard-off HARD_OFF/scores.jsonl --hard-on HARD_ON/scores.jsonl
         [--hard-off-run-json ... --hard-on-run-json ...]] --out DIR

Inputs are the scorer's rows (scripts/expA_score.py, design a3-ledger) of the two A3 runs: the ledger-off run and the
ledger-on run. Only block LEDGER rows are read (condition C1; variant vcomplete = the complete list, omit_node = the
incomplete list). With --*-run-json each run.json must record its arm (episode_config.evidence_ledger) and the
deadline notice.

Recovery = site_recovered of completed cells (a completed episode that did not submit scores 0). Cells missing for
infrastructure are excluded and counted. Primary seeds: the longest prefix of 61, 73, 89 whose LEDGER cells are all
terminal (completed or missing_infrastructure) in BOTH runs (the D11.3 rule of the main analysis).

Per item and arm (variant x ledger): the mean over the primary seeds of its completed cells. An item enters when all
four arms have a mean. Then, per item:
    G_off = rec(complete, off) - rec(incomplete, off)      (omission cost without the ledger)
    G_on  = rec(complete, on)  - rec(incomplete, on)       (omission cost with the ledger)
    M     = G_off - G_on                                   (primary: the reduction in omission cost)
then the family mean over its items, then the equal-weight mean over families, with a two-sided one-sample t test
(df = F - 1, scripts/expA_reanalyze_independent.t_statistics; alpha 0.05; the single primary test).

Secondary (nominal 95% t intervals over family means, no significance claims):
    S1 rec(incomplete, on) - rec(incomplete, off)
    S2 rec(complete, on) - rec(complete, off), with the non-inferiority margin -3 points (lower bound > -0.03)
    S3 G_off and G_on; the four arm levels
    S4 submission rates (exit Submitted) per arm and their on - off differences
    S5 precision bounds (precision_lower, precision_upper) per arm over submitted completed cells
    S6 exits per arm (counts)
    S7 the hard panel (protocols/EXPA_HARD.md items), seed 61: M, S1 and S2 as intervals only
Exit codes: 0 written; 2 refused (inputs unbound or unusable; nothing written).
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

PROTOCOL = "protocols/EXPA_A3_LEDGER.md"
BLOCK = "LEDGER"
CONDITION = "C1"
SEEDS = (61, 73, 89)
HARD_SEEDS = (61,)
VARIANTS = {"complete": "vcomplete", "incomplete": "omit_node"}
LEDGERS = ("off", "on")
ARMS = tuple((answer, ledger) for answer in VARIANTS for ledger in LEDGERS)
TERMINAL = frozenset({"completed", "missing_infrastructure"})
CELL_STATES = frozenset({"completed", "missing_infrastructure", "infra_pending", "not_dispatched"})
ALPHA = 0.05
CONFIDENCE = 0.95
NONINFERIORITY_MARGIN = Fraction(-3, 100)
LABEL = ("PRESPECIFIED (protocols/EXPA_A3_LEDGER.md section 4, written before any A3 cell); ANALYSIS RECORD"
         "")


class Refused(ValueError):
    """The inputs cannot be analysed as the protocol states (nothing is written)."""


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _require(condition, message):
    if not condition:
        raise Refused(message)


def load_rows(path, ledger):
    """The LEDGER rows of one run's scores, keyed by (item_id, variant, seed)."""
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
        key = (row["item_id"], row["variant"], row["seed"])
        _require(key not in rows, f"{where}: duplicate row")
        rows[key] = row
        items_sha.add(row.get("items_sha256"))
    _require(rows, f"{path}: no LEDGER rows (is this an A3 run scored under design a3-ledger?)")
    _require(len(items_sha) == 1, f"{path}: rows name {len(items_sha)} items files")
    _require(set(other_blocks) <= {"not_dispatched"}, f"{path}: cells outside block LEDGER were dispatched "
                                                      f"({dict(other_blocks)}); an A3 run dispatches LEDGER only")
    return {"ledger": ledger, "path": str(path), "sha256": sha256_bytes(raw), "rows": rows,
            "items_sha256": items_sha.pop(), "other_block_rows": sum(other_blocks.values())}


def check_run_json(path, ledger):
    """A run.json must record its arm: the deadline notice and evidence_ledger == ledger."""
    raw = Path(path).read_bytes()
    config = json.loads(raw).get("episode_config") or {}
    _require(config.get("final_turn_policy") == "deadline", f"{path}: episode_config is not the deadline notice")
    _require(config.get("evidence_ledger") == ledger, f"{path}: episode_config.evidence_ledger is "
                                                      f"{config.get('evidence_ledger')!r}, not {ledger!r}")
    return {"path": str(path), "sha256": sha256_bytes(raw), "evidence_ledger": ledger}


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


def item_means(runs, seeds, field="site_recovered", only_submitted=False):
    """{arm: {item: mean over the seeds' completed cells}} and the items' families."""
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
                if value is None:
                    continue
                values[item].append(Fraction(value))
            means[(answer, run["ledger"])] = {item: sum(vals, Fraction(0)) / len(vals)
                                              for item, vals in values.items()}
    return means, family


def contrasts(means, family):
    """M, G_off, G_on, S1, S2 and the arm levels over the items that have all four arms."""
    items = sorted(set.intersection(*(set(means[arm]) for arm in ARMS)))
    rec = {arm: {item: means[arm][item] for item in items} for arm in ARMS}
    g_off = {i: rec[("complete", "off")][i] - rec[("incomplete", "off")][i] for i in items}
    g_on = {i: rec[("complete", "on")][i] - rec[("incomplete", "on")][i] for i in items}
    out = {"items_with_all_four_arms": len(items),
           "M_reduction_in_omission_cost": family_summary({i: g_off[i] - g_on[i] for i in items}, family),
           "G_off_omission_cost_without_ledger": family_summary(g_off, family),
           "G_on_omission_cost_with_ledger": family_summary(g_on, family),
           "S1_incomplete_on_minus_off": family_summary(
               {i: rec[("incomplete", "on")][i] - rec[("incomplete", "off")][i] for i in items}, family),
           "S2_complete_on_minus_off": family_summary(
               {i: rec[("complete", "on")][i] - rec[("complete", "off")][i] for i in items}, family),
           "arm_levels": {f"{answer}_{ledger}": family_summary(rec[(answer, ledger)], family)
                          for answer, ledger in ARMS}}
    s2 = out["S2_complete_on_minus_off"]
    s2["noninferiority_margin"] = float(NONINFERIORITY_MARGIN)
    s2["lower_bound_above_margin"] = (None if s2.get("ci_low") is None
                                      else s2["ci_low"] > float(NONINFERIORITY_MARGIN))
    return out


def per_arm(means, family):
    return {f"{answer}_{ledger}": family_summary(means[(answer, ledger)], family) for answer, ledger in ARMS}


def analyse(off, on, seeds_allowed=SEEDS, *, primary=True):
    _require(off["items_sha256"] == on["items_sha256"], "the two runs name different items files")
    keys_off, keys_on = set(off["rows"]), set(on["rows"])
    _require(keys_off == keys_on, f"the runs hold different LEDGER cells ({len(keys_off ^ keys_on)} differ)")
    runs = (off, on)
    seeds = seed_prefix(runs, seeds_allowed)
    out = {"seeds_considered": list(seeds_allowed), "seeds_used": seeds, "items_sha256": off["items_sha256"],
           "cell_states": {run["ledger"]: dict(sorted(Counter(r["cell_state"] for (_, _, s), r in run["rows"].items()
                                                              if s in seeds_allowed).items())) for run in runs},
           "exits": {run["ledger"]: {answer: dict(sorted(Counter(
               r.get("exit_status") for (_, v, s), r in run["rows"].items()
               if v == variant and s in seeds and r["cell_state"] == "completed").items()))
               for answer, variant in VARIANTS.items()} for run in runs}}
    if not seeds:
        out["status"] = "no_complete_seed"
        return out
    means, family = item_means(runs, seeds)
    out.update(contrasts(means, family))
    out["missing_infrastructure_cells"] = {
        run["ledger"]: sum(1 for (_, _, s), r in run["rows"].items()
                           if s in seeds and r["cell_state"] == "missing_infrastructure") for run in runs}
    sub, _ = item_means(runs, seeds, field="submitted")
    out["S4_submission"] = per_arm(sub, family)
    out["S4_submission_on_minus_off"] = {
        answer: family_summary({i: sub[(answer, "on")][i] - sub[(answer, "off")][i]
                                for i in set(sub[(answer, "on")]) & set(sub[(answer, "off")])}, family)
        for answer in VARIANTS}
    out["S5_precision"] = {bound: per_arm(item_means(runs, seeds, field=bound, only_submitted=True)[0], family)
                           for bound in ("precision_lower", "precision_upper")}
    m = out["M_reduction_in_omission_cost"]
    if primary:
        p = m.get("p_two_sided")
        out["primary_decision"] = {"test": "two-sided one-sample t over family means of M (df = F - 1)",
                                   "alpha": ALPHA, "p_two_sided": p,
                                   "reject_null_at_alpha": None if p is None else p <= ALPHA,
                                   "direction": None if m.get("estimate") is None else
                                   ("ledger reduces the omission cost" if m["estimate"] > 0 else
                                    "ledger does not reduce the omission cost")}
    else:
        m["note"] = "secondary: interval only, no significance claim"
    out["status"] = "analysed"
    return out


def tables(result):
    def fmt(summary):
        if summary.get("estimate") is None:
            return "n/a"
        low, high = summary.get("ci_low"), summary.get("ci_high")
        interval = "" if low is None else f" [{100 * low:+.1f}, {100 * high:+.1f}]"
        return f"{100 * summary['estimate']:+.1f}{interval} (F={summary['n_families']})"
    lines = ["# A3 evidence ledger: prespecified analysis", "", result["label"], ""]
    for name in ("main", "hard_panel"):
        part = result.get(name)
        if part is None:
            continue
        lines += [f"## {name} (seeds used {part['seeds_used']})", ""]
        if part.get("status") != "analysed":
            lines += [f"Status: {part.get('status')}", ""]
            continue
        lines += ["| Quantity | Points [95% t] |", "|---|---|"]
        for key in ("M_reduction_in_omission_cost", "G_off_omission_cost_without_ledger",
                    "G_on_omission_cost_with_ledger", "S1_incomplete_on_minus_off", "S2_complete_on_minus_off"):
            lines.append(f"| {key} | {fmt(part[key])} |")
        for arm, summary in part["arm_levels"].items():
            lines.append(f"| recovery {arm} | {fmt(summary)} |")
        for arm, summary in part["S4_submission"].items():
            lines.append(f"| submitted {arm} | {fmt(summary)} |")
        if "primary_decision" in part:
            d = part["primary_decision"]
            lines += ["", f"Primary: p = {d['p_two_sided']}, reject at 0.05: {d['reject_null_at_alpha']} "
                          f"({d['direction']})."]
        s2 = part["S2_complete_on_minus_off"]
        lines += [f"S2 lower bound above the -3 point margin: {s2.get('lower_bound_above_margin')}.", ""]
    return "\n".join(lines) + "\n"


def run(args):
    result = {"label": LABEL, "protocol": PROTOCOL, "script_sha256": sha256_bytes(Path(__file__).read_bytes()),
              "inputs": {}}
    off, on = load_rows(args.off, "off"), load_rows(args.on, "on")
    result["inputs"]["main"] = {run["ledger"]: {"path": run["path"], "sha256": run["sha256"]} for run in (off, on)}
    if args.off_run_json or args.on_run_json:
        _require(args.off_run_json and args.on_run_json, "give both --off-run-json and --on-run-json")
        result["inputs"]["main_run_json"] = {"off": check_run_json(args.off_run_json, "off"),
                                             "on": check_run_json(args.on_run_json, "on")}
    result["main"] = analyse(off, on)
    if args.hard_off or args.hard_on:
        _require(args.hard_off and args.hard_on, "give both --hard-off and --hard-on")
        hard_off, hard_on = load_rows(args.hard_off, "off"), load_rows(args.hard_on, "on")
        result["inputs"]["hard_panel"] = {r["ledger"]: {"path": r["path"], "sha256": r["sha256"]}
                                          for r in (hard_off, hard_on)}
        if args.hard_off_run_json or args.hard_on_run_json:
            _require(args.hard_off_run_json and args.hard_on_run_json,
                     "give both --hard-off-run-json and --hard-on-run-json")
            result["inputs"]["hard_panel_run_json"] = {"off": check_run_json(args.hard_off_run_json, "off"),
                                                       "on": check_run_json(args.hard_on_run_json, "on")}
        result["hard_panel"] = analyse(hard_off, hard_on, HARD_SEEDS, primary=False)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--off", required=True, help="scores.jsonl of the ledger-off run")
    parser.add_argument("--on", required=True, help="scores.jsonl of the ledger-on run")
    parser.add_argument("--off-run-json")
    parser.add_argument("--on-run-json")
    parser.add_argument("--hard-off", help="scores.jsonl of the hard-panel ledger-off run (secondary, seed 61)")
    parser.add_argument("--hard-on", help="scores.jsonl of the hard-panel ledger-on run (secondary, seed 61)")
    parser.add_argument("--hard-off-run-json")
    parser.add_argument("--hard-on-run-json")
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
    (out / "A3_ANALYSIS.json").write_bytes((json.dumps(result, indent=1, sort_keys=True) + "\n").encode())
    (out / "A3_TABLES.md").write_bytes(tables(result).encode())
    print(tables(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
