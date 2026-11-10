"""Exp A item builder CLI, v2 (docs/EXPA_DESIGN_V2_DELTA.md). CPU only; never touches a GPU.

  spec      --registry REG --repos-root R --static-root S [--trace-root T [--trace-subdir SUB]]
            --out SPEC.json [--roles main dev] [--families F ...] [--allow-missing-trace]
            [--require-identity-sha256 SHA]
  build     --spec SPEC.json --out ITEMS_DIR [--m 25] [--code-git-sha SHA] [--no-c4]
            [--omit-rule first_runtime_witnessed|first_not_grep_findable]
  verify    --items-dir ITEMS_DIR [--dispatch-dir DISPATCH_DIR]
  dispatch  --items-dir ITEMS_DIR --out DISPATCH_DIR [--code-git-sha SHA]
  tokens    --model-dir MODEL_DIR --out FILE [--items-dir ITEMS_DIR] [--examples N] [--extra-pairs JSON]

SPEC.json (written by ``spec``, or by hand):
  {"main_roster": ["fam", ...],              # optional; default = families with role "main"
   "families": [{"family": "more-itertools", "role": "dev",
                 "repo_root": "/abs/checkout", "package_dirs": ["more_itertools"],
                 "commit": "<sha>", "url": "https://...",
                 "trace_dir": "/abs/traces/<family>[/<run>]",   # omit when the family has no trace
                 "static_dir": "/abs/static/<family>"}]}         # locagent.* and jedi.* outputs

``spec`` reads the registry's status-ok families with the given roles (default main and
dev), lists every main family as the roster, and cross-checks each family's registry
commit against its repository HEAD (when it has .git), both static manifests and the
trace manifest; it refuses to write a spec when anything disagrees. With role main it
also pins the D12.3 trace input: --trace-subdir must be trace-1, and every main trace
manifest must name this checkout's frozen tracer.py and identity.py (the production
command is in docs/EXPA_RUNBOOK.md step 0.2).

``build`` writes items.jsonl, manifest.json, payloads/, facts/, eligibility/ and
graph_facts/ into a new directory (outputs are immutable). C4a blocks come from the
live tool's shared search (``tools.lexical_search``, UTF-8 byte bound). ``--no-c4``
builds without it (development only: C4a cells are null and the build cannot be
dispatched). ``--omit-rule first_not_grep_findable`` builds the A2 hard-omission panel
(protocols/EXPA_HARD.md): C* is the first runtime-witnessed caller, in the same omit-key
order, among the callers grep_all misses, and targets without one are excluded as
``no_grep_missed_caller``. The default rule (first_runtime_witnessed) is the frozen one.

``dispatch`` writes the runtime dispatch view (D10.7, contract item 4) from a build that
verifies: dispatch.jsonl (item id, family, target {identity, path, line, qualname},
package dirs, wording pair, membership {in_s_nat, in_s_edge, in_t} and the payload-bearing
cells of P, S-VC, S-NAT and S-EDGE), dispatch_manifest.json (items, items-manifest and
dispatch.jsonl sha256, dispatched payload store, items payload store and graph facts sha256
per family, code hashes), the payload store restricted to the dispatched cells (every row
is named by some dispatch row) and a byte-identical copy of the natural graph facts. It
contains no truth field (V, U, sites, C*, covariates, reasons) and is verified after
writing; V and C* remain derivable on the runtime host from the S-VC and P payloads, so
the tool jail is the isolation boundary.

``tokens`` audits both v2 wording pairs (D4) with the served model directory's tokenizer
on CPU: on the texts, on the live-audit mini payload and on every built item's omit_node
C1 payload with each guidance inserted. It exits non-zero when a pair is outside
WORDING_TOLERANCE.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from absent_edges.data import canonical_json  # noqa: E402
from absent_edges.graphs import GraphFacts  # noqa: E402
from absent_edges.probe import items as items_module  # noqa: E402
from absent_edges.probe import render as render_module  # noqa: E402
from absent_edges.probe import wording  # noqa: E402


def _write_new(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise SystemExit(f"{path} exists; outputs are immutable, choose a new path")
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return hashlib.sha256(data).hexdigest()


def command_spec(args):
    try:
        spec = items_module.build_spec_from_registry(
            args.registry, repos_root=args.repos_root, static_root=args.static_root, trace_root=args.trace_root,
            trace_subdir=args.trace_subdir, roles=args.roles, families=args.families,
            allow_missing_trace=args.allow_missing_trace, require_identity_sha256=args.require_identity_sha256)
    except items_module.SpecError as error:
        raise SystemExit(f"spec refused: {error}")
    digest = _write_new(args.out, json.dumps(spec, indent=1, sort_keys=True).encode("utf-8") + b"\n")
    print(json.dumps({"out": str(args.out), "sha256": digest, "families": [entry["family"] for entry in spec["families"]],
                      "main_roster": len(spec["main_roster"]), "families_without_trace": spec["families_without_trace"],
                      "identity_py_sha256": spec["identity_py_sha256"]}, indent=1, sort_keys=True))


def command_build(args):
    families, roster = items_module.load_build_spec(args.spec)
    factory = None if args.no_c4 else (
        lambda inputs: render_module.shared_lexical_search(inputs.repo_root, inputs.package_dirs))
    manifest = items_module.build_items(families, roster, args.out, m=args.m, search_factory=factory,
                                        code_git_sha=args.code_git_sha, require_c4=not args.no_c4,
                                        omit_rule=args.omit_rule)
    print(json.dumps({"out": str(args.out), "items_sha256": manifest["outputs"]["items.jsonl"]["sha256"],
                      "counts": manifest["counts"],
                      "families": {name: {"selected": summary["selected"], "funnel": summary["funnel"],
                                          "trace_status": summary["trace_status"],
                                          "record_counts": summary["record_counts"]}
                                   for name, summary in manifest["families"].items()}}, indent=1, sort_keys=True))


def command_verify(args):
    try:
        report = items_module.verify_items(args.items_dir)
        if args.dispatch_dir:
            report["dispatch"] = items_module.verify_dispatch(args.dispatch_dir, args.items_dir)
    except (ValueError, KeyError) as error:
        raise SystemExit(f"verify failed: {type(error).__name__}: {error}")
    print(json.dumps(report, sort_keys=True))


def command_dispatch(args):
    report = items_module.build_dispatch(args.items_dir, args.out, code_git_sha=args.code_git_sha)
    manifest = json.loads((Path(args.out) / "dispatch_manifest.json").read_bytes())
    print(json.dumps({"out": str(args.out), "verify": report,
                      "dispatch_sha256": manifest["dispatch_sha256"],
                      "items_sha256": manifest["items_sha256"], "payload_cell_counts": manifest["payload_cell_counts"],
                      "membership_counts": manifest["membership_counts"],
                      "payload_store_sha256": {family: entry["sha256"] for family, entry in manifest["payload_store"].items()},
                      "graph_facts_sha256": {family: entry["sha256"] for family, entry in manifest["graph_facts"].items()}},
                     indent=1, sort_keys=True))


def _synthetic_example():
    """A small C1 render used only when no built items are supplied to the token audit."""
    lines = ["def target(value):", "    return value + 1", "", "def first(value):", "    return target(value)",
             "", "def second(values):", "    return [target(v) for v in values]"]
    nodes = [{"id": "pkg/mod.py::target", "kind": "function", "path": "pkg/mod.py", "start_line": 1,
              "end_line": 2, "snippet": "\n".join(lines[0:2]), "snippet_end_line": 2},
             {"id": "pkg/mod.py::first", "kind": "function", "path": "pkg/mod.py", "start_line": 4,
              "end_line": 5, "snippet": "\n".join(lines[3:5]), "snippet_end_line": 5},
             {"id": "pkg/mod.py::second", "kind": "function", "path": "pkg/mod.py", "start_line": 7,
              "end_line": 8, "snippet": "\n".join(lines[6:8]), "snippet_end_line": 8}]
    edges = [render_module.invokes_edge(caller, "pkg/mod.py::target")
             for caller in ("pkg/mod.py::first", "pkg/mod.py::second")]
    document = render_module.facts_document("0" * 64, nodes, edges, render_module.coverage_for(["pkg/"]))
    observation = GraphFacts(canonical_json(document)).query("pkg/mod.py::target", relation="invokes",
                                                             direction="incoming", limit=30)
    return observation.render("C1", max_output_bytes=render_module.MAX_OUTPUT_BYTES)


def _audit_examples(items_dir, limit):
    """Stored omit_node C1 payloads (the primary cells); checks C2/C3 = C1 + guidance for the item's pair."""
    examples, sources = [], []
    directory = Path(items_dir)
    for raw in (directory / "items.jsonl").read_bytes().splitlines():
        item = json.loads(raw)
        cells = item["variants"]["omit_node"]["cells"]
        c1 = items_module.load_payload(directory, item["family"], cells["C1"])
        for condition in wording.CONDITIONS:
            stored = items_module.load_payload(directory, item["family"], cells[condition])
            rebuilt = {**c1, "guidance": wording.guidance(item["wording_pair"], condition)}
            if canonical_json(stored) != canonical_json(rebuilt):
                raise SystemExit(f"{item['item_id']}: stored {condition} is not C1 plus the pair's guidance")
        examples.append(c1)
        sources.append({"item_id": item["item_id"], "family": item["family"], "omit_node_C1_sha256": cells["C1"],
                        "wording_pair": item["wording_pair"]})
        if limit and len(examples) >= limit:
            break
    return examples, sources


def command_tokens(args):
    examples, sources = ([], []) if not args.items_dir else _audit_examples(args.items_dir, args.examples)
    if not examples:
        examples, sources = [_synthetic_example()], [{"synthetic": True}]
    extra = json.loads(Path(args.extra_pairs).read_bytes()) if args.extra_pairs else None
    result = wording.offline_wording_audit(args.model_dir, examples, extra_pairs=extra)
    result["example_sources"] = sources
    digest = _write_new(args.out, json.dumps(result, indent=1, sort_keys=True).encode("utf-8") + b"\n")
    keys = ("text_tokens", "chat_tokens", "mini_payload_chat_tokens", "text_absolute_difference",
            "absolute_difference", "relative_difference", "mini_payload_absolute_difference",
            "full_payload_max_absolute_difference", "full_payload_max_relative_difference", "within_tolerance")
    print(json.dumps({"out": str(args.out), "sha256": digest, "examples": len(examples),
                      "all_pairs_within_tolerance": result["all_pairs_within_tolerance"],
                      "pairs": {pair: {key: row[key] for key in keys}
                                for pair, row in result["audit"]["pairs"].items()}}, indent=1, sort_keys=True))
    if not result["all_pairs_within_tolerance"]:
        raise SystemExit("wording audit: a v2 pair is outside WORDING_TOLERANCE (D4: revise filler words, re-audit)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    spec = commands.add_parser("spec")
    spec.add_argument("--registry", required=True)
    spec.add_argument("--repos-root", required=True, help="repository checkouts: <repos-root>/<family>")
    spec.add_argument("--static-root", required=True, help="static outputs: <static-root>/<family>/{locagent,jedi}.*")
    spec.add_argument("--trace-root", help="trace runs: <trace-root>/<family>[/<trace-subdir>]/manifest.json")
    spec.add_argument("--trace-subdir")
    spec.add_argument("--roles", nargs="+", default=list(items_module.ROLES), choices=list(items_module.ROLES))
    spec.add_argument("--families", nargs="+", help="optional subset of the registry families")
    spec.add_argument("--allow-missing-trace", action="store_true",
                      help="omit trace_dir for a family without a trace manifest (its items need runtime witnesses, "
                           "so it yields none)")
    spec.add_argument("--require-identity-sha256", help="refuse inputs produced with another identity.py")
    spec.add_argument("--out", required=True)
    spec.set_defaults(handler=command_spec)
    build = commands.add_parser("build")
    build.add_argument("--spec", required=True)
    build.add_argument("--out", required=True)
    build.add_argument("--m", type=int, default=items_module.M_PER_FAMILY)
    build.add_argument("--code-git-sha")
    build.add_argument("--no-c4", action="store_true", help="development only: no shared search, C4a cells null")
    build.add_argument("--omit-rule", default=items_module.OMIT_RULE, choices=list(items_module.OMIT_RULES),
                       help="C* rule; first_not_grep_findable builds the A2 hard-omission set")
    build.set_defaults(handler=command_build)
    verify = commands.add_parser("verify")
    verify.add_argument("--items-dir", required=True)
    verify.add_argument("--dispatch-dir")
    verify.set_defaults(handler=command_verify)
    dispatch = commands.add_parser("dispatch")
    dispatch.add_argument("--items-dir", required=True)
    dispatch.add_argument("--out", required=True)
    dispatch.add_argument("--code-git-sha")
    dispatch.set_defaults(handler=command_dispatch)
    tokens = commands.add_parser("tokens")
    tokens.add_argument("--model-dir", required=True)
    tokens.add_argument("--out", required=True)
    tokens.add_argument("--items-dir")
    tokens.add_argument("--examples", type=int, default=0, help="items to audit (0 = all)")
    tokens.add_argument("--extra-pairs", help='optional JSON {name: {"C2": text, "C3": text}} candidates to audit')
    tokens.set_defaults(handler=command_tokens)
    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
