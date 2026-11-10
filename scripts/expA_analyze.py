"""Exp A analysis v2: one command from the scored rows of an immutable run to ANALYSIS.json + TABLES.md + CELLS.jsonl.

    python scripts/expA_analyze.py analyze --items items.jsonl --scores scores.jsonl --policies policies.jsonl
        --prereg plan.json --run-dir RUN --dispatch-dir DISPATCH --out OUT [--payloads DIR]
        [--prereg-sha256 HEX] [--overwrite]
    python scripts/expA_analyze.py findability --items items.jsonl --policies policies.jsonl [--prereg plan.json]
        [--items-manifest manifest.json] --out OUT
    python scripts/expA_analyze.py plan --out plan.json [--label TEXT] [--design d11|contract-v2] [--roles dev]
    python scripts/expA_analyze.py synthetic --out DIR [--scenario known_effect|null] [--families N]
        [--items-per-family N] [--rng-seed N] [--partial-seed S] [--missing-fraction F] [--design D]

``analyze`` refuses (score.BindingError) an items file whose sha256 differs from run.json, the
dispatch manifest, the scores manifest or the score and policy rows; score rows that do not name
each terminal result's attempt and sha256; and rows that do not cover exactly the design cells
(D10.14). The primary analysis follows D11.2-D11.4 (t test on family differences, Holm across
H1-H3, seed-prefix rule, missing-cell bounds, 20-family minimum). Inputs are read only; existing
outputs are never overwritten unless --overwrite is given. ``--unbound-synthetic`` skips the run
and dispatch checks and is accepted only for a plan labelled SYNTHETIC.

``findability`` writes FINDABILITY.json and FINDABILITY.md from the frozen items and their
reference-policy rows only (D11.7: disclosed before the first main dispatch; no outcomes).
``synthetic`` writes a SYNTHETIC validation fixture; it is never research data. Figures:
scripts/expA_figures.py in the figures environment (matplotlib). Stdlib only.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from absent_edges.probe import analysis  # noqa: E402
from absent_edges.probe import score  # noqa: E402


def _dispatch_manifest(args):
    if args.dispatch_dir and args.dispatch_manifest:
        raise SystemExit("give --dispatch-dir or --dispatch-manifest, not both")
    if args.dispatch_dir:
        return str(Path(args.dispatch_dir) / "dispatch_manifest.json")
    return args.dispatch_manifest


def run_analyze(args):
    manifest = _dispatch_manifest(args)
    if args.unbound_synthetic:
        plan, _ = analysis.load_plan(args.prereg)
        if not plan.label.startswith("SYNTHETIC"):
            raise SystemExit("--unbound-synthetic is only for plans labelled SYNTHETIC")
    elif not (args.run_dir and manifest):
        raise SystemExit("analyze needs --run-dir and --dispatch-dir for the binding checks (D10.14)")
    result, written = analysis.analyze_paths(
        items_path=args.items, scores_path=args.scores, policies_path=args.policies, plan_path=args.prereg,
        out_dir=args.out, run_dir=args.run_dir, dispatch_manifest=manifest, require_run=not args.unbound_synthetic,
        overwrite=args.overwrite, plan_sha256=args.prereg_sha256, payloads_dir=args.payloads)
    primary = result["primary"]
    return {"out": args.out, "status": result["status"], "written": written, "primary_status": primary["status"],
            "primary_seeds": primary["primary_seeds"], "sensitivity_seeds": primary["sensitivity_seeds"],
            "hypotheses": [{k: h.get(k) for k in ("id", "status", "estimate", "p_primary", "p_source",
                                                   "holm_adjusted_p", "reject_at_alpha", "n_families")}
                           for h in primary["hypotheses"]],
            "cell_states": result["completion"]["cell_states"]}


def run_findability(args):
    plan = analysis.load_plan(args.prereg)[0] if args.prereg else analysis.plan_from_dict(analysis.default_plan_dict())
    items, items_meta = score.load_items(args.items)
    policies, policy_meta = analysis.load_policy_rows(args.policies)
    stale = sum(1 for row in policies.values() if row.get("items_sha256") != items_meta["sha256"])
    if stale:
        raise score.BindingError(f"{stale} policy rows were computed from another items file")
    manifest_path = Path(args.items_manifest) if args.items_manifest else Path(args.items).parent / "manifest.json"
    items_manifest = json.loads(manifest_path.read_bytes()) if manifest_path.is_file() else None
    inputs = {"items": items_meta, "policies": policy_meta}
    if items_manifest is not None:
        inputs["items_manifest"] = {"path": str(manifest_path),
                                    "sha256": analysis.sha256_bytes(manifest_path.read_bytes())}
    report = analysis.findability_report(plan, items, policies, inputs, items_manifest)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, data in (("FINDABILITY.json", analysis.dump_json(report)),
                       ("FINDABILITY.md", analysis.render_findability_md(report).encode("utf-8"))):
        analysis._write(out / name, data, args.overwrite)
        written[name] = analysis.sha256_bytes(data)
    return {"out": str(out), "written": written, "findability": {
        k: report["findability"][k] for k in ("items", "families", "grep_recovers_omitted",
                                              "c4_hits_recovers_omitted", "c4_trigger_fires")}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("analyze", help="analyze the scored rows of an immutable run")
    run.add_argument("--items", required=True, help="items.jsonl (expA-item-v2) the run was built from")
    run.add_argument("--scores", required=True, help="scripts/expA_score.py score output")
    run.add_argument("--policies", required=True, help="scripts/expA_score.py policies output")
    run.add_argument("--prereg", required=True, help="analysis plan JSON (expA-analysis-plan-v2)")
    run.add_argument("--run-dir", help="the run directory (run.json and attempt records) for the binding checks")
    run.add_argument("--dispatch-dir", help="the run's dispatch view directory (dispatch_manifest.json)")
    run.add_argument("--dispatch-manifest", help="dispatch_manifest.json of the run's dispatch view")
    run.add_argument("--payloads", help="payload store (default: items dir/payloads; needed for S-NAT NM checks)")
    run.add_argument("--out", required=True, help="output directory for ANALYSIS.json, TABLES.md, CELLS.jsonl")
    run.add_argument("--prereg-sha256", help="refuse to run unless the plan file has this sha256")
    run.add_argument("--overwrite", action="store_true", help="replace existing outputs in --out")
    run.add_argument("--unbound-synthetic", action="store_true",
                     help="skip run/dispatch binding (only for plans labelled SYNTHETIC)")
    find = commands.add_parser("findability", help="reference policies and findability of the frozen items")
    find.add_argument("--items", required=True)
    find.add_argument("--policies", required=True)
    find.add_argument("--prereg", help="analysis plan (for analysis_roles); default plan otherwise")
    find.add_argument("--items-manifest", help="items manifest.json (default: next to items.jsonl) for the "
                                               "per-family eligibility flow")
    find.add_argument("--out", required=True)
    find.add_argument("--overwrite", action="store_true")
    plan = commands.add_parser("plan", help="write the default expA-analysis-plan-v2 JSON")
    plan.add_argument("--out", required=True)
    plan.add_argument("--label", default="")
    plan.add_argument("--design", choices=sorted(score.DESIGNS), default=score.DEFAULT_DESIGN)
    plan.add_argument("--roles", default="main", help="comma-separated analysis roles (dev only for the pilot)")
    synth = commands.add_parser("synthetic", help="write a SYNTHETIC validation fixture")
    synth.add_argument("--out", required=True)
    synth.add_argument("--scenario", choices=sorted(analysis._SYNTHETIC_RATES), default="known_effect")
    synth.add_argument("--families", type=int, default=24)
    synth.add_argument("--items-per-family", type=int, default=5)
    synth.add_argument("--dev-families", type=int, default=1)
    synth.add_argument("--rng-seed", type=int, default=7)
    synth.add_argument("--partial-seed", type=int, default=None)
    synth.add_argument("--missing-fraction", type=float, default=0.0)
    synth.add_argument("--design", choices=sorted(score.DESIGNS), default=score.DEFAULT_DESIGN)
    args = parser.parse_args(argv)
    if args.command == "synthetic":
        result = analysis.write_synthetic(
            args.out, scenario=args.scenario, families=args.families, items_per_family=args.items_per_family,
            dev_families=args.dev_families, rng_seed=args.rng_seed, partial_seed=args.partial_seed,
            missing_fraction=args.missing_fraction, design=args.design)
    elif args.command == "plan":
        path = Path(args.out)
        if path.exists():
            raise SystemExit(f"refusing to overwrite {path}")
        data = analysis.dump_json(analysis.default_plan_dict(label=args.label, design=args.design,
                                                             analysis_roles=args.roles.split(",")))
        analysis.plan_from_dict(json.loads(data))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        result = {"out": str(path), "sha256": analysis.sha256_bytes(data)}
    elif args.command == "findability":
        result = run_findability(args)
    else:
        result = run_analyze(args)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
