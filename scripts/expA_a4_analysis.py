#!/usr/bin/env python3
"""Exp A follow-up A4 (final-message components, 2 x 2): the prespecified analysis (protocols/EXPA_A4_DIRECTIVE.md).

PRESPECIFIED: written before any A4 cell exists. ANALYSIS RECORD. Standard library only;
exact rational means. It was exercised only on SYNTHETIC fixtures.

    python scripts/expA_a4_analysis.py --scores-neutral N/scores.jsonl --scores-info I/scores.jsonl \\
        --scores-submit S/scores.jsonl --scores-deadline D/scores.jsonl \\
        [--run-json-neutral N/run.json --run-json-info I/run.json --run-json-submit S/run.json
         --run-json-deadline D/run.json] [--seed 23] --out DIR

Inputs are the scorer's rows (scripts/expA_score.py) of the four A4 runs, one per final-turn policy:
    neutral        "Your previous tool call was processed normally."            (no information, no imperative)
    deadline_info  "This is your last response."                                (information I)
    submit_only    "Call submit_answer now."                                    (imperative D)
    deadline       "This is your last response. Call submit_answer now."        (I and D)
Each run dispatches block P at --seed (default 23), conditions C0 (none) and C1 (omit_node) only; every scored row of
another cell must be not_dispatched, or the input is refused. With --run-json-<arm>, that run.json must record the
policy (episode_config.final_turn_policy and its notice, the prereg budget), and the launch records next to it
(launches/*.json) must all select block P, the seed and conditions C0,C1.

Outcome: site_recovered of the completed block-P cells at the seed (a completed episode that did not submit scores 0);
cells missing for infrastructure (or not terminal) are excluded and counted. Per item and arm (policy x condition) the
cell's value; an item enters a contrast only when every cell the contrast needs is completed. Item value -> family mean
-> equal-weight mean over families (the conventions of scripts/expA_followup_analysis.py, whose contrast() and
summarize() are reused; t from scripts/expA_reanalyze_independent.t_statistics).

Factors: I = deadline information (deadline_info, deadline vs neutral, submit_only); D = submit imperative
(submit_only, deadline vs neutral, deadline_info). List effect L(policy) = rec(C1) - rec(C0).
  PRIMARY (the single test, alpha 0.05): the imperative's effect on the list effect, averaged over I:
      per item [(L(submit_only) - L(neutral)) + (L(deadline) - L(deadline_info))] / 2,
      two-sided one-sample t over family means, df = F - 1.
  Secondary (nominal 95% t intervals, no significance claims):
      S1 information's effect on L averaged over D: [(L(deadline_info) - L(neutral)) + (L(deadline) - L(submit_only))]/2
      S2 main effect of D on recovery, averaged over C0, C1 and I:
         mean_c [(rec(submit_only, c) + rec(deadline, c)) - (rec(neutral, c) + rec(deadline_info, c))] / 2
      S3 main effect of I likewise: mean_c [(rec(deadline_info, c) + rec(deadline, c)) - (rec(neutral, c)
         + rec(submit_only, c))] / 2
      S4 the D x I interaction on recovery, averaged over C0, C1:
         mean_c [(rec(deadline, c) - rec(deadline_info, c)) - (rec(submit_only, c) - rec(neutral, c))]
      S5 deadline - neutral averaged over C0, C1 (the A1 primary contrast, replicated at this seed)
      S6 per policy: L(policy), recovery levels per condition, submission rate, precision bounds (precision_lower,
         precision_upper over submitted completed cells, when the rows carry them) and exits.
Outputs (in --out, which must not exist): A4_RESULTS.json (every number, input sha256s, the script sha256, sorted
keys, no timestamps) and A4_RESULTS.md (tables). Exit codes: 0 written; 2 refused (nothing written).
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from expA_reanalyze_independent import t_statistics  # noqa: E402,F401  (used through the A1 module's summarize)

_SPEC = importlib.util.spec_from_file_location("expA_followup_analysis", _HERE / "expA_followup_analysis.py")
A1 = sys.modules.get("expA_followup_analysis")
if A1 is None:
    A1 = importlib.util.module_from_spec(_SPEC)
    sys.modules["expA_followup_analysis"] = A1
    _SPEC.loader.exec_module(A1)

SCHEMA = "expA-a4-analysis-v1"
PROTOCOL = "protocols/EXPA_A4_DIRECTIVE.md"
LABEL = ("PRESPECIFIED (protocols/EXPA_A4_DIRECTIVE.md, written before any A4 cell); ANALYSIS RECORD"
         "")
BLOCK = "P"
DEFAULT_SEED = 23
CONDITIONS = ("C0", "C1")
P_VARIANT = {"C0": "none", "C1": "omit_node"}
# Arm name -> (final-turn policy, I = deadline information, D = submit imperative)
ARMS = {"neutral": ("neutral", 0, 0), "info": ("deadline_info", 1, 0), "submit": ("submit_only", 0, 1),
        "deadline": ("deadline", 1, 1)}
POLICIES = tuple(policy for policy, _, _ in ARMS.values())
TEXTS = {"neutral": "Your previous tool call was processed normally.",
         "deadline_info": "This is your last response.",
         "submit_only": "Call submit_answer now.",
         "deadline": "This is your last response. Call submit_answer now."}
PREREG_BUDGET = {"max_model_calls": 8, "context_budget": 16384}
ALPHA = A1.ALPHA


class Refused(ValueError):
    """The inputs cannot be analysed as the protocol states (nothing is written)."""


def _require(condition, message):
    if not condition:
        raise Refused(message)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------- inputs


def load_policy_rows(path, policy, seed):
    """{(item, condition): row} of one run's block-P C0/C1 rows at ``seed``; every other row must be not_dispatched."""
    try:
        rows = A1.load_scores(path)
    except A1.Refusal as error:          # a SystemExit subclass in the A1 module: refuse here instead
        raise Refused(str(error)) from error
    cells, dispatched_elsewhere, excluded = {}, Counter(), Counter()
    for row in rows:
        selected = (row["block"] == BLOCK and row["seed"] == seed and row["condition"] in CONDITIONS
                    and P_VARIANT[row["condition"]] == row["variant"])
        if not selected:
            excluded["outside_the_a4_selection"] += 1
            if row["cell_state"] != "not_dispatched":
                dispatched_elsewhere[f"{row['block']}|{row['variant']}|{row['condition']}|s{row['seed']}"] += 1
            continue
        key = (row["item_id"], row["condition"])
        _require(key not in cells, f"{path}: two rows for {key}")
        cells[key] = row
    _require(cells, f"{path}: no block-P C0/C1 rows at seed {seed}")
    _require(not dispatched_elsewhere, f"{path}: cells outside block P, seed {seed}, C0/C1 were dispatched "
                                       f"({dict(sorted(dispatched_elsewhere.items()))}); an A4 run dispatches only "
                                       "those")
    items_sha = {row.get("items_sha256") for row in cells.values()}
    _require(len(items_sha) == 1, f"{path}: rows name {len(items_sha)} items files")
    return {"policy": policy, "path": str(path), "sha256": A1.sha256(path), "cells": cells,
            "items_sha256": items_sha.pop(), "rows_excluded": dict(sorted(excluded.items()))}


def check_run_json(path, policy, seed):
    """run.json records the policy; every launch record next to it selects block P, the seed and C0,C1."""
    path = Path(path)
    raw = path.read_bytes()
    run = json.loads(raw)
    _require(isinstance(run, dict), f"{path}: not an object")
    config = run.get("episode_config") or {}
    _require(config.get("final_turn_policy") == policy, f"{path}: episode_config.final_turn_policy is "
                                                         f"{config.get('final_turn_policy')!r}, not {policy!r}")
    notice = config.get("final_turn_notice") or {}
    _require(notice.get("text") == TEXTS[policy], f"{path}: the recorded notice text is not the {policy} text")
    _require(all(config.get(key) == value for key, value in PREREG_BUDGET.items()),
             f"{path}: episode_config is not the prereg budget (8 responses, 16384 tokens)")
    _require(config.get("evidence_ledger", "none") == "none" and config.get("submit_review", "none") == "none",
             f"{path}: an A4 run has no evidence ledger and no submit review")
    launches = sorted(p for p in (path.parent / "launches").glob("*.json") if not p.name.endswith(".end.json"))
    _require(launches, f"{path.parent / 'launches'}: no launch records to verify the selection (block P, seed "
                       f"{seed}, conditions C0,C1)")
    records = []
    for launch in launches:
        selection = json.loads(launch.read_bytes()).get("selection") or {}
        found = (selection.get("blocks"), selection.get("seeds"), selection.get("conditions"))
        _require(found == ([BLOCK], [seed], list(CONDITIONS)),
                 f"{launch}: selection blocks/seeds/conditions {found}, not ({[BLOCK]}, {[seed]}, {list(CONDITIONS)})")
        records.append({"path": str(launch), "sha256": sha256_bytes(launch.read_bytes())})
    return {"path": str(path), "sha256": sha256_bytes(raw), "final_turn_policy": policy, "launches": records}


# --------------------------------------------------------------------------- contrasts


def L(g, policy):
    """The list effect rec(C1) - rec(C0) of a policy."""
    return g[(policy, "C1")] - g[(policy, "C0")]


def mean_over_conditions(fn):
    return lambda g: sum((fn(g, c) for c in CONDITIONS), Fraction(0)) / len(CONDITIONS)


def needed(*policies, conditions=CONDITIONS):
    return tuple((policy, condition) for policy in policies for condition in conditions)


DEFINITIONS = {
    "primary": ("the imperative's effect on the list effect, averaged over information: per item "
                "[(L(submit_only) - L(neutral)) + (L(deadline) - L(deadline_info))] / 2, L = rec(C1) - rec(C0)"),
    "S1_information_effect_on_list_effect": ("per item [(L(deadline_info) - L(neutral)) + (L(deadline) - "
                                             "L(submit_only))] / 2"),
    "S2_imperative_main_effect": ("per item, mean over C0, C1 of [(rec(submit_only) + rec(deadline)) - (rec(neutral) "
                                  "+ rec(deadline_info))] / 2"),
    "S3_information_main_effect": ("per item, mean over C0, C1 of [(rec(deadline_info) + rec(deadline)) - "
                                   "(rec(neutral) + rec(submit_only))] / 2"),
    "S4_imperative_x_information_interaction": ("per item, mean over C0, C1 of [(rec(deadline) - rec(deadline_info)) "
                                                "- (rec(submit_only) - rec(neutral))]"),
    "S5_deadline_minus_neutral": "per item, mean over C0, C1 of [rec(deadline) - rec(neutral)] (A1 primary, this seed)",
}


def contrasts(values, items, family):
    def run(fn, cells, *, test=False):
        return A1.contrast(fn, cells, values, items, family, test=test)

    primary = run(lambda g: ((L(g, "submit_only") - L(g, "neutral")) + (L(g, "deadline") - L(g, "deadline_info"))) / 2,
                  needed(*POLICIES), test=True)
    p = primary.get("p_two_sided")
    primary.update(
        definition=DEFINITIONS["primary"],
        outcome="site_recovered (unconditional; a completed non-submitted episode scores 0)",
        unit="item value -> family mean -> equal-weight mean over families", alpha=ALPHA,
        test="two-sided one-sample t over family means (df = F - 1)",
        item_rule="an item enters only when all eight cells (four policies x C0, C1) are completed",
        reject_at_alpha=None if p is None else p <= ALPHA,
        decision=("not_estimable" if p is None else
                  ("reject H0 (the imperative does not change the list effect) at alpha 0.05" if p <= ALPHA else
                   "do not reject H0 (the imperative does not change the list effect) at alpha 0.05")))
    secondary = {
        "status": "SECONDARY: nominal 95% family-clustered t intervals (df = F - 1); no p-values, no significance "
                  "claims",
        "S1_information_effect_on_list_effect": run(
            lambda g: ((L(g, "deadline_info") - L(g, "neutral")) + (L(g, "deadline") - L(g, "submit_only"))) / 2,
            needed(*POLICIES)),
        "S2_imperative_main_effect": run(mean_over_conditions(
            lambda g, c: ((g[("submit_only", c)] + g[("deadline", c)]) - (g[("neutral", c)] + g[("deadline_info", c)]))
            / 2), needed(*POLICIES)),
        "S3_information_main_effect": run(mean_over_conditions(
            lambda g, c: ((g[("deadline_info", c)] + g[("deadline", c)]) - (g[("neutral", c)] + g[("submit_only", c)]))
            / 2), needed(*POLICIES)),
        "S4_imperative_x_information_interaction": run(mean_over_conditions(
            lambda g, c: (g[("deadline", c)] - g[("deadline_info", c)]) - (g[("submit_only", c)] - g[("neutral", c)])),
            needed(*POLICIES)),
        "S5_deadline_minus_neutral": run(mean_over_conditions(
            lambda g, c: g[("deadline", c)] - g[("neutral", c)]), needed("deadline", "neutral")),
    }
    for key, summary in secondary.items():
        if key in DEFINITIONS:
            summary["definition"] = DEFINITIONS[key]
    return primary, secondary


def policy_table(policy, cells, values, items, family):
    """S6 for one policy: L, recovery levels, submission, precision bounds and exits."""
    out = {"list_effect_L": A1.contrast(lambda g: L(g, policy), needed(policy), values, items, family),
           "by_condition": {}}
    for condition in CONDITIONS:
        rows = [row for (item, c), row in sorted(cells.items()) if c == condition and row["cell_state"] == "completed"]
        submitted = [row for row in rows if row["exit_status"] == "Submitted"]
        entry = {"completed_episodes": len(rows), "submitted_episodes": len(submitted),
                 "missing_infrastructure": sum(1 for (item, c), row in cells.items()
                                               if c == condition and row["cell_state"] == "missing_infrastructure"),
                 "not_completed": dict(sorted(Counter(row["cell_state"] for (item, c), row in cells.items()
                                                      if c == condition and row["cell_state"] != "completed").items())),
                 "exits": dict(sorted(Counter(row["exit_status"] for row in rows).items()))}
        metrics = {"recovery": (rows, A1.recovery), "submission_rate": (rows, A1.submitted)}
        for field in ("precision_lower", "precision_upper"):
            if rows and all(field not in row for row in rows):
                entry[field] = {"status": "not_available", "reason": f"field {field} is absent from the score rows"}
                continue
            metrics[field] = ([row for row in submitted if row.get(field) is not None],
                              lambda row, f=field: Fraction(row[f]))
        for name, (eligible, value) in metrics.items():
            item_values = {row["item_id"]: value(row) for row in eligible}
            entry[name] = {"micro": A1.micro(item_values.values()),
                           "family_weighted": A1.family_weighted(item_values, family),
                           "eligible": ("submitted completed episodes with a value" if name.startswith("precision")
                                        else "completed episodes")}
        out["by_condition"][condition] = entry
    return out


def analyse(runs, seed):
    items_files = {run["items_sha256"] for run in runs.values()}
    _require(len(items_files) == 1, f"the four runs name {len(items_files)} items files")
    family = A1.merge_families((f"scores_{policy}", {key[0]: row["family"] for key, row in run["cells"].items()})
                               for policy, run in runs.items())
    items = sorted(family)
    values = {policy: {key: A1.recovery(row) for key, row in run["cells"].items()} for policy, run in runs.items()}
    census = {policy: {condition: dict(sorted(Counter(
        "absent" if run["cells"].get((item, condition)) is None else run["cells"][(item, condition)]["cell_state"]
        for item in items).items())) for condition in CONDITIONS} for policy, run in runs.items()}
    primary, secondary = contrasts(values, items, family)
    secondary["S6_by_policy"] = {policy: policy_table(policy, run["cells"], values, items, family)
                                 for policy, run in runs.items()}
    return {"seed": seed, "items_sha256": items_files.pop(),
            "population": {"items": len(items), "families": len(set(family.values())),
                           "items_by_family": dict(sorted(Counter(family.values()).items())),
                           "cells": census,
                           "missing_infrastructure_cells": {
                               policy: sum(1 for row in run["cells"].values()
                                           if row["cell_state"] == "missing_infrastructure")
                               for policy, run in runs.items()},
                           "rows_excluded": {policy: run["rows_excluded"] for policy, run in runs.items()}},
            "factors": {policy: {"I_deadline_information": info, "D_submit_imperative": imperative}
                        for policy, info, imperative in ARMS.values()},
            "primary": primary, "secondary": secondary}


# --------------------------------------------------------------------------- outputs


def _g(x):
    return "n/a" if x is None else f"{100 * x:+.1f}"


def _iv(summary):
    low, high = summary.get("ci_low"), summary.get("ci_high")
    interval = "" if low is None else f" [{_g(low)}, {_g(high)}]"
    return f"{_g(summary.get('estimate'))}{interval}"


def tables(result):
    a = result["analysis"]
    p = a["primary"]
    lines = ["# A4 final-message components (2 x 2): prespecified analysis", "", result["label"], "",
             f"Seed {a['seed']}; items {a['population']['items']}; families {a['population']['families']}. "
             "Points are percentage points of site_recovered.", "",
             "## Primary (single test, alpha 0.05)", "",
             "| Contrast | Estimate [95% t] | t | df | p | F | items | Decision |", "|---|---|---|---|---|---|---|---|",
             f"| imperative x list effect, averaged over information | {_iv(p)} | "
             f"{'n/a' if p.get('t') is None else format(p['t'], '.3f')} | {p.get('df')} | "
             f"{'n/a' if p.get('p_two_sided') is None else format(p['p_two_sided'], '.4g')} | {p['n_families']} | "
             f"{p['n_items']} | {p['decision']} |", "",
             "## Secondary (nominal 95% t intervals, no significance claims)", "",
             "| Contrast | Estimate [95% t] | F | items |", "|---|---|---|---|"]
    s = a["secondary"]
    for key in ("S1_information_effect_on_list_effect", "S2_imperative_main_effect", "S3_information_main_effect",
                "S4_imperative_x_information_interaction", "S5_deadline_minus_neutral"):
        lines.append(f"| {key} | {_iv(s[key])} | {s[key]['n_families']} | {s[key]['n_items']} |")
    lines += ["", "S6 by policy (family-weighted):", "",
              "| Policy | L = C1 - C0 | Cond | Recovery | Submission | Precision lo | Precision hi | Missing (infra) | "
              "Exits |", "|---|---|---|---|---|---|---|---|---|"]
    for policy, table in s["S6_by_policy"].items():
        for condition, e in table["by_condition"].items():
            cells = [_iv(e[k]["family_weighted"]) if "family_weighted" in e[k] else "n/a"
                     for k in ("recovery", "submission_rate", "precision_lower", "precision_upper")]
            exits = ", ".join(f"{k} {v}" for k, v in e["exits"].items()) or "none"
            lines.append(f"| {policy} | {_iv(table['list_effect_L'])} | {condition} | " + " | ".join(cells)
                         + f" | {e['missing_infrastructure']} | {exits} |")
    return "\n".join(lines) + "\n"


def run(args):
    paths = {"neutral": args.scores_neutral, "info": args.scores_info, "submit": args.scores_submit,
             "deadline": args.scores_deadline}
    run_jsons = {"neutral": args.run_json_neutral, "info": args.run_json_info, "submit": args.run_json_submit,
                 "deadline": args.run_json_deadline}
    runs, inputs = {}, {}
    for arm, (policy, _, _) in ARMS.items():
        runs[policy] = load_policy_rows(paths[arm], policy, args.seed)
        inputs[f"scores_{policy}"] = {"path": runs[policy]["path"], "sha256": runs[policy]["sha256"]}
    digests = Counter(entry["sha256"] for entry in inputs.values())
    _require(all(count == 1 for count in digests.values()), "two or more score inputs are byte-identical")
    given = [arm for arm, path in run_jsons.items() if path]
    _require(not given or len(given) == len(run_jsons), "give all four --run-json-* or none")
    for arm, path in run_jsons.items():
        if path:
            inputs[f"run_json_{ARMS[arm][0]}"] = check_run_json(path, ARMS[arm][0], args.seed)
    return {"schema": SCHEMA, "label": LABEL, "protocol": PROTOCOL,
            "status": "PRESPECIFIED analysis code, committed before any A4 outcome existed",
            "script": {"path": "scripts/expA_a4_analysis.py", "sha256": sha256_bytes(Path(__file__).read_bytes())},
            "inputs": inputs, "run_json_verified": bool(given), "pooled_with_other_runs": False,
            "analysis": analyse(runs, args.seed)}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scores-neutral", required=True, help="scores.jsonl of the neutral run")
    parser.add_argument("--scores-info", required=True, help="scores.jsonl of the deadline_info run")
    parser.add_argument("--scores-submit", required=True, help="scores.jsonl of the submit_only run")
    parser.add_argument("--scores-deadline", required=True, help="scores.jsonl of the deadline run")
    for arm in ARMS:
        parser.add_argument(f"--run-json-{arm}", help=f"run.json of the {ARMS[arm][0]} run (launches/ next to it)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="the block-P seed (default %(default)s)")
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
    except (Refused, OSError, ValueError) as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    out.mkdir(parents=True)
    (out / "A4_RESULTS.json").write_bytes(
        (json.dumps(result, indent=1, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))
    text = tables(result)
    (out / "A4_RESULTS.md").write_bytes(text.encode("utf-8"))
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
