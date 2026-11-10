"""Exp A deterministic scoring CLI, v2 (D7, D10.11, D10.14, D11.3-D11.5): run -> scores.jsonl; items -> policies.jsonl.

    python scripts/expA_score.py score --run-dir RUN --items ITEMS/items.jsonl --dispatch-dir DISPATCH
        [--payloads ITEMS/payloads] (--repos-root ROOT | --checkouts CHECKOUTS.json)
        [--design d11|contract-v2|a3-ledger] [--no-telemetry] --out SCORES.jsonl
    python scripts/expA_score.py policies --items ITEMS/items.jsonl [--payloads ITEMS/payloads]
        [--dispatch-dir DISPATCH] (--repos-root ROOT | --checkouts CHECKOUTS.json) --out POLICIES.jsonl

``score`` writes one ``expA-score-v2`` row per cell of the run's design for these items
(absent_edges.probe.score.score_run): the terminal attempt scored, or a row with the
cell state (missing_infrastructure, infra_pending, not_dispatched) and null outcomes.
It refuses (BindingError) an items file whose sha256 differs from run.json, the dispatch
manifest or any result; a payload store that differs from the dispatch manifest; a
result for a cell outside the design; and any completed cell whose payload sha256 or
target differs from its item. The design layout (``d11``, ``contract-v2``, or ``a3-ledger`` for
the A3 runs of protocols/EXPA_A3_LEDGER.md) is the one the run's records name; ``--design``
only adds a requirement.

``policies`` writes the model-free reference-policy rows (answer_only, c4_hits,
answer_plus_c4, grep_all) for every item and variant of the frozen items (D10.11). With
``--dispatch-dir`` the items are also checked against the dispatch view.

Submitted lines are mapped with identity.FunctionIndex over each family's checkout, so
the checkout must hold the package tree the items were built from (render.source_sha256);
another tree is refused. Outputs are never overwritten; each gets an immutable
``<out>.manifest.json`` with every input hash and the code hashes. The run directory is
read only. CPU only, stdlib only.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse  # noqa: E402
import collections  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402

from absent_edges.data import canonical_json  # noqa: E402
from absent_edges.probe import score  # noqa: E402

SCORE_CLI_PROTOCOL = "expA-score-cli-v2"
ROOT = Path(__file__).resolve().parents[1]
CODE_FILES = ("absent_edges/probe/score.py", "absent_edges/probe/identity.py", "absent_edges/data.py",
              "scripts/expA_score.py")


def code_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in CODE_FILES}


def resolve_checkouts(items, repos_root=None, mapping=None):
    """{family: {"path", "package_tree_sha256", "package_files"}}, each checked against its items."""
    expected, package_dirs = collections.defaultdict(set), {}
    for item in items.values():
        package_dirs.setdefault(item.family, item.package_dirs)
        if item.package_dirs != package_dirs[item.family]:
            raise ValueError(f"items of {item.family} disagree on package_dirs")
        if item.source_sha256:
            expected[item.family].add(item.source_sha256)
    checkouts = {}
    for family in sorted(package_dirs):
        if mapping and family in mapping:
            path = Path(mapping[family])
        elif repos_root:
            path = Path(repos_root) / family
        else:
            raise ValueError(f"no checkout for family {family}; give --repos-root or --checkouts")
        if not path.is_dir():
            raise FileNotFoundError(f"checkout of {family} not found: {path}")
        tree, count = score.package_tree_sha256(path, package_dirs[family])
        if expected[family] and expected[family] != {tree}:
            raise score.BindingError(f"checkout of {family} has package tree {tree}, but its items were built from "
                                     f"{sorted(expected[family])}; refusing to score against different source")
        checkouts[family] = {"path": str(path.resolve()), "package_tree_sha256": tree, "package_files": count}
    return checkouts


def write_rows(out, rows, manifest):
    """rows -> <out> (canonical JSONL) and <out>.manifest.json, both created exclusively and made read-only."""
    out = Path(out)
    meta_path = out.with_name(out.name + ".manifest.json")
    for path in (out, meta_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    data = b"".join(canonical_json(row) + b"\n" for row in rows)
    digest = score.write_immutable(out, data)
    manifest = dict(manifest, schema=score.SCORES_MANIFEST_SCHEMA,
                    out={"path": str(out), "sha256": digest, "bytes": len(data), "rows": len(rows)})
    score.write_immutable(meta_path, canonical_json(manifest) + b"\n")
    return manifest


def _dispatch(args):
    paths = (args.dispatch, args.dispatch_manifest)
    if args.dispatch_dir:
        if any(paths):
            raise ValueError("give --dispatch-dir or --dispatch/--dispatch-manifest, not both")
        directory = Path(args.dispatch_dir)
        paths = (directory / "dispatch.jsonl", directory / "dispatch_manifest.json")
    elif any(paths) and not all(paths):
        raise ValueError("--dispatch and --dispatch-manifest go together")
    if not all(paths):
        return None
    return score.load_dispatch(*paths)


def _common(args):
    items, items_meta = score.load_items(args.items)
    payloads = score.PayloadStore(args.payloads or Path(args.items).parent / "payloads")
    mapping = json.loads(Path(args.checkouts).read_text(encoding="utf-8")) if args.checkouts else None
    checkouts = resolve_checkouts(items, args.repos_root, mapping)
    index = score.index_factory({family: info["path"] for family, info in checkouts.items()})
    return items, items_meta, payloads, checkouts, index, _dispatch(args)


def run_score(args):
    items, items_meta, payloads, checkouts, index, dispatch = _common(args)
    if dispatch is None:
        raise ValueError("score needs the dispatch view (--dispatch-dir) to bind results to it")
    rows, audit = score.score_run(args.run_dir, items, items_meta, index, dispatch=dispatch, payloads=payloads,
                                  design=args.design, telemetry=not args.no_telemetry)
    manifest = {"protocol": SCORE_CLI_PROTOCOL, "score_protocol": score.SCORE_PROTOCOL, "schema": score.SCORE_SCHEMA,
                "kind": "episode", "design": score.Design(audit["design"]).describe(),
                "items": {"path": str(args.items), "sha256": items_meta["sha256"], "items": items_meta["items"],
                          "roles": items_meta["roles"]},
                "run": {"run_dir": str(args.run_dir), "run_json_sha256": audit["run_json_sha256"]},
                "dispatch": {"dispatch_sha256": dispatch["dispatch_sha256"],
                             "manifest_sha256": dispatch["manifest_sha256"]},
                "payload_store": dict(sorted(payloads.file_sha256.items())), "checkouts": checkouts,
                "code": code_hashes(), "binding": audit,
                "by_exit_status": dict(sorted(collections.Counter(r["exit_status"] for r in rows).items())),
                "by_cell_state": dict(sorted(collections.Counter(r["cell_state"] for r in rows).items())),
                "by_block": dict(sorted(collections.Counter(r["block"] for r in rows).items()))}
    return write_rows(args.out, rows, manifest)


def run_policies(args):
    items, items_meta, payloads, checkouts, index, dispatch = _common(args)
    if dispatch is not None:
        audit = score.check_inputs_binding(items, items_meta, dispatch=dispatch, payloads=payloads)
    else:
        audit = {"items_sha256": items_meta["sha256"], "dispatch": "not given"}
    rows = score.policy_rows(items, payloads, index, items_sha256=items_meta["sha256"])
    manifest = {"protocol": SCORE_CLI_PROTOCOL, "score_protocol": score.SCORE_PROTOCOL, "schema": score.POLICY_SCHEMA,
                "kind": "policy",
                "items": {"path": str(args.items), "sha256": items_meta["sha256"], "items": items_meta["items"],
                          "roles": items_meta["roles"]},
                "payload_store": dict(sorted(payloads.file_sha256.items())), "checkouts": checkouts,
                "code": code_hashes(), "binding": audit, "summary": score.summarize_policies(rows)}
    return write_rows(args.out, rows, manifest)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("score", "policies"):
        sub = commands.add_parser(name)
        if name == "score":
            sub.add_argument("--run-dir", required=True, help="immutable run directory (run.json + attempt records)")
            sub.add_argument("--design", choices=sorted(score.DESIGNS), default=None,
                             help="require this layout (default: the one the run's records name)")
            sub.add_argument("--no-telemetry", action="store_true", help="skip events.jsonl process descriptives")
        sub.add_argument("--items", required=True, help="items.jsonl the run was built from")
        sub.add_argument("--payloads", default=None, help="payload store directory (default: items dir/payloads)")
        sub.add_argument("--dispatch-dir", default=None, help="directory with dispatch.jsonl + dispatch_manifest.json")
        sub.add_argument("--dispatch", default=None, help="dispatch.jsonl of the dispatch view")
        sub.add_argument("--dispatch-manifest", default=None, help="dispatch_manifest.json of the dispatch view")
        sub.add_argument("--repos-root", default=None, help="checkout of family F is REPOS_ROOT/F")
        sub.add_argument("--checkouts", default=None, help="JSON {family: checkout path}")
        sub.add_argument("--out", required=True, help="new JSONL output (never overwritten)")
    args = parser.parse_args(argv)
    manifest = run_score(args) if args.command == "score" else run_policies(args)
    summary = {key: manifest[key] for key in ("kind", "out", "binding") if key in manifest}
    for key in ("by_exit_status", "by_cell_state", "by_block", "summary"):
        if key in manifest:
            summary[key] = manifest[key]
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
