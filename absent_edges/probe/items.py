"""Exp A items v2 (docs/EXPA_DESIGN_V2_DELTA.md D1, D3, D5, D8, D10-D12; v1 docs/EXPA_DESIGN.md 6).

Selection is deterministic and outcome-blind. Roles are ``main`` and ``dev`` only (no
A/Q split, no enrichment); dev items are built for the pilot and never analysed.

Eligibility of a package function X, in funnel order (the first failing rule is
recorded; every candidate is evaluated, independent of M):

1. v1 metadata filters: out_of_scope, lambda, duplicate_identity, dunder, property,
   decorated (other than staticmethod/classmethod), coroutine, generator;
2. ``nested_locals``: the qualname contains ``<locals>`` (D10.1);
3. ``jedi_definition_error`` (D12.4): the J manifest records a failed jedi query for
   X's own definition (stage ``jedi_definition``, or ``jedi_definition_position`` when
   the definition could not even be queried), so L-and-J cannot be computed for X;
   ``locagent_unindexed`` (D12.7): X, or a J-reported caller of X, has no LocAgent node, so
   L can never report that edge. A function has an L node when a LocAgent function node maps
   to it, or when it lies in the first direct ``def __init__`` of a class LocAgent has a node
   for: LocAgent attributes those calls to the class node and static_callers maps them to the
   enclosing function (``class_node_init``). Applied when the build has the LocAgent document;
   the function-node-only count is recorded next to it;
4. ``verified_below_2`` / ``verified_above_12``: 2 <= |V| <= 12;
5. ``no_runtime_witnessed_caller``: no caller in V has a trace site;
6. ``render_budget``: the vcomplete answer cannot render untruncated even with empty
   snippets;
7. ``render_truncated``: the vcomplete, omit_node or omit_edge render is truncated in
   any way (30-match limit, shared snippet budget, or ``render()`` halving a snippet);
8. ``incidental_visibility`` (D10.2): a snippet of the omit_node C1 payload covers a
   site span of C* (same path, overlapping lines), or the omit_node graph payload
   contains C*'s identity (the C4a lexical block is not part of the graph payload and
   may expose C* by design, D11.6). An identity followed by ``.`` counts: a delivered
   nested identity ``C*.<locals>.g`` contains C*'s identity. The count of verbatim
   matches of C*'s call-line texts inside the delivered snippets is recorded as the
   covariate ``visibility_text_matches`` (D10.13, finding F7), never used as a filter.
   So is ``visibility_name_mentions`` (whole-word occurrences of C*'s simple name in the
   delivered snippets, code or prose).
9. ``docref_visibility`` (D13.1): a delivered snippet of the omit_node graph payload names
   C* in a Sphinx cross-reference role or backtick-quoted name whose last segment is C*'s
   simple name, for example a target docstring ``Used by :func:`dft` and :func:`idft```.
   The count is kept as the covariate ``visibility_docref_mentions`` (0 for every eligible item);
10. ``mechanical_diff_failed``: a D5 invariant fails (for either wording pair).

The omitted caller C* is the first runtime-witnessed caller of X in
``sha256("expA-omit-v2|" + X + "|" + C)`` order. The opt-in A2 rule
(``build_items(..., omit_rule=HARD_OMIT_RULE)``, protocols/EXPA_HARD.md) takes C* in the same
order among the callers grep_all misses (``grep_missed_callers``) and excludes a target without
one as ``no_grep_missed_caller``, right after rule 5; nothing else changes, and the manifest
records the rule. Up to M = 25 eligible targets per
family are taken in ``sha256("expA-v2|" + family + "|" + X)`` order. Among a family's
selected items: wording pair by ``sha256("expA-wording-v2|" + X)`` rank (even A, odd B);
``in_s_edge`` for the first 5 by ``sha256("expA-edge-v2|" + X)``; ``in_t`` for the
first 5 by ``sha256("expA-tool-v2|" + X)``; ``in_s_nat`` when ``natural_miss`` (a
runtime-witnessed caller absent from LocAgent's answer) and the natural render is
untruncated, else ``s_nat_excluded_reason`` says why.

Outputs under the items directory (canonical JSON, UTF-8; written once, never replaced):
  items.jsonl                   one ``expA-item-v2`` record per selected target
  manifest.json                 counts, funnel, parameters, invariants, code and input hashes
  payloads/<family>.jsonl       the payload store: unique rendered cells {"sha256","bytes","payload"}
  facts/<family>.jsonl          GraphFacts documents per item and variant
  eligibility/<family>.jsonl    every package function with its exclusion reason
  graph_facts/<family>.json     the family's natural LocAgent facts (the T arms' graph_query)

The dispatch view (D10.7, contract item 4) is written by ``build_dispatch``: only what
the runtime host needs (item id, family, target fields, package dirs, wording pair, the
three membership flags and the payload-bearing cells of P, S-VC, S-NAT and S-EDGE, and a
payload store holding exactly those cells), with no truth field. It still lets the runtime
host derive V and C* (vcomplete C1 minus omit_node C1 endpoints), and membership.in_s_nat
encodes natural_miss: the tool jail is the isolation boundary, not the file selection.
``build_spec_from_registry`` writes the build spec from the registry (``expA_items.py spec``).
"""
from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

from ..data import canonical_json
from . import render as render_module
from . import wording
from .identity import normalize_package_dirs, split_identity
from .truth import (STRATA, TRUTH_PROTOCOL, PackageScope, TruthTable, lexical_lines, lexical_names,
                    locagent_indexed_identities, read_call_sites, simple_name)

ITEMS_PROTOCOL = "expA-items-v2"
ITEM_SCHEMA = "expA-item-v2"
DISPATCH_SCHEMA = "expA-dispatch-v2"
SPEC_SCHEMA = "expA-items-spec-v2"
ROLES = ("main", "dev")
M_PER_FAMILY = 25
S_EDGE_PER_FAMILY = 5
T_PER_FAMILY = 5
MIN_VERIFIED, MAX_VERIFIED = 2, 12
SELECTION_SALT, OMIT_SALT, EDGE_SALT, TOOL_SALT = "expA-v2", "expA-omit-v2", "expA-edge-v2", "expA-tool-v2"
OMIT_RULE = "first_runtime_witnessed"
VISIBILITY_TEXT_MIN_CHARS = 8
ALLOWED_DECORATORS = frozenset({"staticmethod", "classmethod"})
PROPERTY_DECORATORS = frozenset({"property", "cached_property", "functools.cached_property",
                                 "abstractproperty", "abc.abstractproperty"})
METADATA_EXCLUSIONS = ("out_of_scope", "lambda", "duplicate_identity", "dunder", "property", "decorated",
                       "coroutine", "generator", "nested_locals")
EXCLUSION_ORDER = METADATA_EXCLUSIONS + ("jedi_definition_error", "locagent_unindexed", "verified_below_2",
                                         "verified_above_12", "no_runtime_witnessed_caller", "render_budget",
                                         "render_truncated", "incidental_visibility", "docref_visibility",
                                         "mechanical_diff_failed")
NEW_V2_RULES = ("nested_locals", "jedi_definition_error", "locagent_unindexed", "no_runtime_witnessed_caller",
                "render_truncated", "incidental_visibility", "docref_visibility", "mechanical_diff_failed")
# A2 hard omission (opt-in, ``build --omit-rule first_not_grep_findable``): C* is the first runtime-witnessed
# caller in the same sha256("expA-omit-v2|X|C") order among the callers that the grep_all reference policy
# misses (no line of X's lexical_sites lies inside any site span of C: the score.py site rule). A target with
# no such caller is excluded as ``no_grep_missed_caller``, right after rule 5. The default rule is unchanged.
HARD_OMIT_RULE = "first_not_grep_findable"
OMIT_RULES = (OMIT_RULE, HARD_OMIT_RULE)
HARD_EXCLUSION = "no_grep_missed_caller"


def exclusion_order(omit_rule=OMIT_RULE):
    """EXCLUSION_ORDER, with ``no_grep_missed_caller`` after rule 5 under the hard rule."""
    if omit_rule == OMIT_RULE:
        return EXCLUSION_ORDER
    if omit_rule != HARD_OMIT_RULE:
        raise ValueError(f"Unknown omit rule {omit_rule!r}")
    index = EXCLUSION_ORDER.index("no_runtime_witnessed_caller") + 1
    return EXCLUSION_ORDER[:index] + (HARD_EXCLUSION,) + EXCLUSION_ORDER[index:]
# J manifest error stages that mean jedi produced no reference answer for that definition (D12.4).
JEDI_DEFINITION_ERROR_STAGES = ("jedi_definition", "jedi_definition_position")
# D12.2/D12.3: the main build's trace input is trace-1 of the re-run from release dd842ab6b (trace-2 is the
# determinism audit). The spec refuses another subdir for role main and requires every main trace manifest to
# name the frozen tracer.py and identity.py of this checkout (the superseded first attempt used another tracer.py).
MAIN_TRACE_SUBDIR = "trace-1"
TRACE_CODE_FILES = ("absent_edges/probe/tracer.py", "absent_edges/probe/identity.py")
# D8 cells that carry a payload (interface contract v2 item 2). C0 (variant none) and the T
# arms (variant none, T-offered / T-recommended) have no payload, so the dispatch view
# needs nothing for them beyond the row (and membership.in_t); which seeds and blocks
# they run in is the harness's design (D8 / D11.4), not part of the dispatch view.
BLOCK_CELLS = (("P", "omit_node", ("C1", "C2", "C3", "C4a"), None),
               ("S-VC", "vcomplete", ("C1", "C3"), None),
               ("S-NAT", "natural", ("C1", "C3", "C4a"), "in_s_nat"),
               ("S-EDGE", "omit_edge", ("C1", "C3"), "in_s_edge"))
NO_PAYLOAD_CELLS = (("P", "none", ("C0",), None), ("T", "none", ("T-offered", "T-recommended"), "in_t"))
DISPATCH_ROW_KEYS = ("item_id", "family", "target", "package_dirs", "wording_pair", "membership", "cells")
DISPATCH_TARGET_KEYS = ("identity", "path", "line", "qualname")
MEMBERSHIP_KEYS = ("in_s_nat", "in_s_edge", "in_t", "s_nat_excluded_reason")
# The runtime host needs only the three flags; s_nat_excluded_reason would reveal natural_miss.
DISPATCH_MEMBERSHIP_KEYS = ("in_s_nat", "in_s_edge", "in_t")
TRUTH_KEYS = frozenset({"V", "U", "sites", "omitted", "covariates", "lexical_sites", "eligibility", "stratum",
                        "runtime_witnessed", "witnesses", "natural", "names", "verification", "order_key",
                        "match_sources", "n_V", "natural_miss", "visibility_text_matches",
                        "visibility_docref_mentions", "visibility_name_mentions",
                        "s_nat_excluded_reason", "variants", "keys"})
_PROBE_DIR = Path(__file__).resolve().parent
_HEX64 = re.compile(r"[0-9a-f]{64}")


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def selection_key(family, identity):
    return sha256_text(f"{SELECTION_SALT}|{family}|{identity}")


def omit_key(identity, caller):
    return sha256_text(f"{OMIT_SALT}|{identity}|{caller}")


def edge_key(identity):
    return sha256_text(f"{EDGE_SALT}|{identity}")


def tool_key(identity):
    return sha256_text(f"{TOOL_SALT}|{identity}")


def item_id(family, identity):
    return sha256_text(f"{ITEM_SCHEMA}|{family}|{identity}")[:20]


def decorator_base(decorator):
    """Decorator expression without '@' and call arguments, e.g. 'functools.wraps'."""
    if isinstance(decorator, dict):
        decorator = decorator.get("text") or decorator.get("name") or decorator.get("expression") or ""
    text = str(decorator).strip()
    if text.startswith("@"):
        text = text[1:].strip()
    return text.split("(", 1)[0].strip()


def target_exclusion(scope, identity, info):
    """Metadata exclusion reason (rules 1-2), or None. Verified-count bounds are separate."""
    relpath, qualname = split_identity(identity)
    name = qualname.rsplit(".", 1)[-1]
    if not scope.contains(relpath):
        return "out_of_scope"
    if info.get("is_lambda") or name == "<lambda>":
        return "lambda"
    if identity in scope.duplicates():
        return "duplicate_identity"
    if len(name) > 4 and name.startswith("__") and name.endswith("__"):
        return "dunder"
    bases = [decorator_base(item) for item in info.get("decorators") or ()]
    if any(base in PROPERTY_DECORATORS or base.endswith((".setter", ".getter", ".deleter")) for base in bases):
        return "property"
    if any(base not in ALLOWED_DECORATORS for base in bases):
        return "decorated"
    if info.get("is_async"):
        return "coroutine"
    if info.get("is_generator"):
        return "generator"
    if "<locals>" in qualname.split("."):
        return "nested_locals"
    return None


def jedi_definition_errors(manifest):
    """{identity: stage} for J manifest errors on a definition's own jedi query (D12.4)."""
    failed = {}
    for error in (manifest or {}).get("errors") or ():
        if isinstance(error, dict) and error.get("stage") in JEDI_DEFINITION_ERROR_STAGES \
                and isinstance(error.get("identity"), str):
            failed.setdefault(error["identity"], error["stage"])
    return dict(sorted(failed.items()))


def locagent_unindexed_identities(document, scope, path_prefix="", indexed=None):
    """Package functions LocAgent can never report as a caller or a callee (D12.7), sorted.

    Indexed = mapped from a LocAgent function node (``truth.locagent_indexed_identities``), or
    enclosing a line of the first direct ``def __init__`` of a class LocAgent has a node for
    (LocAgent attributes calls made there to the class node, and static_callers maps them to the
    enclosing function, ``class_node_init``). Everything else in the package has no L node, for
    example every function of a file LocAgent could not parse.
    """
    covered = set(locagent_indexed_identities(document, scope, path_prefix) if indexed is None else indexed)
    for node in document.get("nodes", []):
        if node.get("kind") != "class" or not node.get("path") or not node.get("start_line"):
            continue
        path = path_prefix + "/" + node["path"] if path_prefix else node["path"]
        if not scope.contains(path):
            continue
        name = node["id"].split("::", 1)[1].rsplit(".", 1)[-1]
        definition = scope.index.definition_node(path, node["start_line"], name, kind="class")
        init = None if definition is None else next(
            (item for item in definition.body if isinstance(item, ast.FunctionDef) and item.name == "__init__"), None)
        if init is None:
            continue
        for line in range(init.lineno, (init.end_lineno or init.lineno) + 1):
            caller = scope.index.enclosing_function(path, line)
            if caller is not None:
                covered.add(caller)
    return sorted(set(scope.table()) - covered)


def grep_missed_callers(scope, truth, target):
    """Runtime-witnessed callers of X whose every site the grep_all policy misses, sorted.

    grep_all submits X's lexical_sites (every package line naming X's qualified or simple name or a frozen
    import alias as a whole word); a caller is grep-findable when one of those lines lies inside one of its
    site spans (score.py ``site_recovered`` applied to grep_all).
    """
    lines = {}
    for path, number in lexical_lines(scope, lexical_names(scope, target, truth.aliases(target))):
        lines.setdefault(path, []).append(number)
    return sorted(caller for caller in truth.runtime_witnessed(target)
                  if not any(site["lineno"] <= number <= site["end_lineno"]
                             for site in truth.sites(target, caller) for number in lines.get(site["path"], ())))


def candidate_rows(family, scope, truth, jedi_failed=None, unindexed=None, omit_rule=OMIT_RULE):
    """Rules 1-5 for every package function, in identity order (``unindexed``: D12.7, None = not applied)."""
    jedi_failed = jedi_failed or {}
    unindexed = frozenset(unindexed or ())
    rows = []
    for identity, info in sorted(scope.table().items()):
        reason = target_exclusion(scope, identity, info)
        count = len(truth.verified(identity))
        runtime = len(truth.runtime_witnessed(identity))
        detail = None
        unindexed_callers = [caller for caller in truth.callers_by(identity, "J") if caller in unindexed]
        if reason is None and identity in jedi_failed:
            reason, detail = "jedi_definition_error", {"stage": jedi_failed[identity]}
        elif reason is None and (identity in unindexed or unindexed_callers):
            reason, detail = "locagent_unindexed", {"target_unindexed": identity in unindexed,
                                                    "j_callers_unindexed": unindexed_callers}
        elif reason is None and count < MIN_VERIFIED:
            reason = "verified_below_2"
        elif reason is None and count > MAX_VERIFIED:
            reason = "verified_above_12"
        elif reason is None and runtime == 0:
            reason = "no_runtime_witnessed_caller"
        elif reason is None and omit_rule == HARD_OMIT_RULE and not grep_missed_callers(scope, truth, identity):
            reason = HARD_EXCLUSION
        rows.append({"identity": identity, "selection_key": selection_key(family, identity),
                     "verified_count": count, "runtime_witnessed_count": runtime, "excluded": reason,
                     "detail": detail, "eligible_rank": None, "selected": False})
    return rows


def omitted_caller(identity, runtime_witnessed):
    """C* = the first runtime-witnessed caller in sha256("expA-omit-v2|" + X + "|" + C) order."""
    callers = sorted(runtime_witnessed)
    if not callers:
        raise ValueError("A target needs a runtime-witnessed caller before one can be omitted")
    return min(callers, key=lambda caller: (omit_key(identity, caller), caller))


def choose_omitted(identity, runtime_witnessed, stratum_of_caller, rule=OMIT_RULE):
    """The omitted-caller record; ``stratum_of_caller`` maps a caller to its stratum.

    ``runtime_witnessed`` is the rule's candidate pool: every runtime-witnessed caller for the default
    rule, the grep-missed ones (``grep_missed_callers``) for the hard rule.
    """
    chosen = omitted_caller(identity, runtime_witnessed)
    stratum = stratum_of_caller[chosen] if isinstance(stratum_of_caller, dict) else stratum_of_caller(chosen)
    return {"identity": chosen, "stratum": stratum, "rule": rule, "order_key": omit_key(identity, chosen)}


def first_by_key(identities, key, count):
    return set(sorted(identities, key=lambda identity: (key(identity), identity))[:count])


# ------------------------------------------------------------------ inputs
@dataclass(frozen=True)
class FamilyInputs:
    family: str
    role: str  # "main" or "dev"
    repo_root: str
    package_dirs: tuple
    commit: str = None
    url: str = None
    trace_dir: str = None        # tracer output directory (edges*.jsonl, plugin manifests, manifest.json)
    trace_edges: str = None      # alternatively one explicit call-site JSONL file
    trace_manifest: str = None
    locagent_edges: str = None
    locagent_manifest: str = None
    locagent_document: str = None
    jedi_edges: str = None
    jedi_manifest: str = None


def _first_existing(directory, names):
    if not directory:
        return None
    for name in names:
        candidate = Path(directory) / name
        if candidate.is_file():
            return str(candidate)
    return None


def family_inputs_from_spec(entry):
    role = entry.get("role", entry.get("class", "main"))
    if role not in ROLES:
        raise ValueError(f"{entry.get('family')}: role must be main or dev, not {role!r}")
    package_dirs = entry.get("package_dirs")
    if not isinstance(package_dirs, list) or not package_dirs:
        raise ValueError(f"{entry.get('family')}: package_dirs must be a non-empty list")
    static_dir, trace_dir = entry.get("static_dir"), entry.get("trace_dir")
    # A named input directory that does not exist is an error, never a silent "no trace" / "no static".
    for name, directory in (("static_dir", static_dir), ("trace_dir", trace_dir)):
        if directory and not Path(directory).is_dir():
            raise ValueError(f"{entry.get('family')}: {name} {directory} is not a directory"
                             + (" (omit trace_dir when the family has no trace run)" if name == "trace_dir" else ""))
    return FamilyInputs(
        family=entry["family"], role=role, repo_root=entry["repo_root"],
        package_dirs=tuple(package_dirs), commit=entry.get("commit"), url=entry.get("url"),
        trace_dir=trace_dir if trace_dir and Path(trace_dir).is_dir() else None,
        trace_edges=entry.get("trace_edges"),
        trace_manifest=entry.get("trace_manifest") or _first_existing(trace_dir, ["manifest.json"]),
        locagent_edges=entry.get("locagent_edges") or _first_existing(static_dir, ["locagent.edges.jsonl", "L.edges.jsonl"]),
        locagent_manifest=entry.get("locagent_manifest") or _first_existing(static_dir, ["locagent.manifest.json", "L.manifest.json"]),
        locagent_document=entry.get("locagent_document") or _first_existing(static_dir, ["locagent.document.json"]),
        jedi_edges=entry.get("jedi_edges") or _first_existing(static_dir, ["jedi.edges.jsonl", "J.edges.jsonl"]),
        jedi_manifest=entry.get("jedi_manifest") or _first_existing(static_dir, ["jedi.manifest.json", "J.manifest.json"]))


def load_build_spec(path):
    spec = json.loads(Path(path).read_bytes())
    families = [family_inputs_from_spec(entry) for entry in spec["families"]]
    names = [item.family for item in families]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate family in build spec")
    roster = spec.get("main_roster")
    if roster is None:
        roster = [item.family for item in families if item.role == "main"]
    missing = [item.family for item in families if item.role == "main" and item.family not in roster]
    if missing:
        raise ValueError(f"Main families missing from main_roster: {missing}")
    return families, list(roster)


def _file_record(path):
    if path is None:
        return None
    data = Path(path).read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def _read_json(path):
    if path is None:
        return None
    value = json.loads(Path(path).read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: manifest must be a JSON object")
    return value


def _load_trace(inputs):
    """(records, info). Records are used only from complete or partial trace runs."""
    manifest = _read_json(inputs.trace_manifest)
    status = None if manifest is None else manifest.get("status")
    info = {"status": status, "manifest_source_tree_sha256":
            (((manifest or {}).get("repo") or {}).get("source_tree") or {}).get("sha256"),
            "commit": ((((manifest or {}).get("repo") or {}).get("git") or {}).get("head")),
            "identity_py_sha256": ((manifest or {}).get("plugin_files") or {}).get("absent_edges/probe/identity.py")}
    if inputs.trace_dir is None and inputs.trace_edges is None:
        return [], {**info, "status": "unavailable", "reason": "no trace input"}
    if status in ("unavailable", "failed"):
        return [], {**info, "records_used": False}
    if inputs.trace_dir is not None:
        from .tracer import load_trace_dir  # lane T loader: merges per-worker files and checkpoints
        run = load_trace_dir(inputs.trace_dir, strict=True)
        if any(value is False for value in run.integrity.values()):
            raise ValueError(f"{inputs.family}: a trace edges file does not match its plugin manifest")
        files = {}
        for name in sorted(run.integrity):
            data = (Path(inputs.trace_dir) / name).read_bytes()
            files[name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        info.update(status=status or ("partial" if run.partial else "unknown"), loader="tracer.load_trace_dir",
                    partial=run.partial, sources=dict(run.sources), files=files,
                    duplicates_merged=run.duplicates, records_used=True)
        return list(run.records), info
    info.update(status=status or "unknown", loader="truth.read_call_sites",
                files={Path(inputs.trace_edges).name: _file_record(inputs.trace_edges)}, records_used=True)
    return read_call_sites(inputs.trace_edges), info


def _check_static_manifest(family, backend, manifest, source_hash):
    if manifest is None:
        return {"status": None, "source_tree_sha256": None}
    if manifest.get("status") == "failed":
        raise ValueError(f"{family}: static backend {backend} failed; its edges cannot be used")
    recorded = manifest.get("source_tree_sha256")
    if recorded is not None and recorded != source_hash:
        raise ValueError(f"{family}: static backend {backend} ran on a different package source tree")
    return {"status": manifest.get("status"), "source_tree_sha256": recorded,
            "commit": (manifest.get("repo") or {}).get("commit"),
            "identity_py_sha256": (manifest.get("code_sha256") or {}).get("absent_edges/probe/identity.py")}


# ------------------------------------------------------------------ evaluation
def _hits_in_sites(block, sites):
    if not block:
        return False
    return any(hit.get("path") == site["path"] and site["lineno"] <= hit.get("line", 0) <= site["end_lineno"]
               for hit in block.get("hits") or () for site in sites)


def _needles(scope, detailed_sites):
    texts = set()
    for site in detailed_sites:
        for line in site["anchors"]:
            text = (scope.line_text(site["path"], line) or "").strip()
            if len(text) >= VISIBILITY_TEXT_MIN_CHARS:
                texts.add(text)
    return sorted(texts)


def evaluate_candidate(scope, truth, renderer, target, *, search, indexed, omit_rule=OMIT_RULE):
    """Rules 5-8 for one target that passed rules 1-4: (reason or None, detail, state)."""
    verified = truth.verified(target)
    runtime = truth.runtime_witnessed(target)
    pool = runtime if omit_rule == OMIT_RULE else grep_missed_callers(scope, truth, target)
    omitted = choose_omitted(target, pool, lambda caller: truth.stratum(target, caller), omit_rule)
    natural = truth.callers_by(target, "L")
    natural_indexed = True if indexed is None else (target in indexed or bool(natural))
    try:
        prepared = renderer.prepare(target, verified, natural, omitted["identity"], None,
                                    natural_target_indexed=natural_indexed)
    except render_module.RenderBudgetError:
        return "render_budget", None, None
    truncated = [variant for variant in render_module.STRICT_VARIANTS if prepared["info"][variant]["payload_truncated"]]
    if truncated:
        return "render_truncated", {"truncated_variants": truncated}, None
    verification = None if search is None else render_module.verification_block(search, target)
    prepared["verification"] = verification
    _, detailed, _, _ = truth.site_details(target, omitted["identity"])
    omit_c1 = prepared["c1"]["omit_node"]
    text_matches = render_module.visibility_text_matches(omit_c1, _needles(scope, detailed))
    mentions = render_module.omitted_name_mentions(omit_c1, omitted["identity"])
    overlaps = render_module.snippet_site_overlaps(omit_c1, detailed)
    cells = {pair: renderer.cells(prepared, pair) for pair in wording.PAIRS}
    # D10.2 / D11.6: the identity rule covers the graph payload (C1-C3, and C4a without its lexical block).
    exposed = any(render_module.graph_identity_exposed(omitted["identity"], cells[pair]["omit_node"])
                  for pair in wording.PAIRS)
    visibility = {"snippet_covers_site": bool(overlaps), "identity_in_payload": exposed,
                  "text_matches": text_matches, "docref_mentions": mentions["docref"],
                  "name_mentions": mentions["name"],
                  "covering_endpoints": sorted({identity for identity, _ in overlaps})}
    if overlaps or exposed:
        return "incidental_visibility", visibility, None
    # D13.1: a docstring cross-reference naming C* in the delivered graph payload discloses the omitted caller.
    if mentions["docref"] > 0:
        return "docref_visibility", visibility, None
    for pair in wording.PAIRS:
        try:
            render_module.check_invariants(target, omitted["identity"], cells[pair], pair, verification=verification)
        except AssertionError as error:
            return "mechanical_diff_failed", {"pair": pair, "error": str(error)[:500]}, None
    state = {"prepared": prepared, "cells": cells, "omitted": omitted, "natural": natural,
             "natural_indexed": natural_indexed, "verification": verification, "visibility": visibility,
             "omitted_detailed_sites": detailed}
    return None, {"text_matches": text_matches}, state


def _verification_summary(block, source_hash):
    if block is None:
        return None
    data = canonical_json(block)
    return {"query": block.get("query"), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
            "hits": len(block["hits"]), "truncated": block.get("truncated"), "error": block.get("error"),
            "token_bound": (block.get("budget") or {}).get("token_bound"),
            "source_sha256_matches": block.get("source_sha256") == source_hash}


# ------------------------------------------------------------------ building
def build_family(inputs, *, m=M_PER_FAMILY, search=None, index=None, source_sha256=None, omit_rule=OMIT_RULE):
    """Build one family's items. ``search`` is query -> verification block (None: no C4a cells)."""
    order = exclusion_order(omit_rule)
    if inputs.locagent_edges is None or inputs.jedi_edges is None:
        raise ValueError(f"{inputs.family}: both static backends (L and J) are required")
    scope = PackageScope(inputs.repo_root, inputs.package_dirs, index=index)
    source_hash = source_sha256 or scope.source_sha256()
    locagent_manifest, jedi_manifest = _read_json(inputs.locagent_manifest), _read_json(inputs.jedi_manifest)
    static_status = {"L": _check_static_manifest(inputs.family, "L", locagent_manifest, source_hash),
                     "J": _check_static_manifest(inputs.family, "J", jedi_manifest, source_hash)}
    truth = TruthTable(scope)
    trace_records, trace_info = _load_trace(inputs)
    recorded_trace_tree = trace_info.get("manifest_source_tree_sha256")
    if trace_info.get("records_used") and recorded_trace_tree is not None and recorded_trace_tree != source_hash:
        raise ValueError(f"{inputs.family}: the trace ran on a different package source tree")
    record_counts = {"trace": {"records": len(trace_records), "kept": truth.add_records("trace", trace_records)}}
    for source, path in (("L", inputs.locagent_edges), ("J", inputs.jedi_edges)):
        records = read_call_sites(path)
        record_counts[source] = {"records": len(records), "kept": truth.add_records(source, records)}
    indexed = unindexed = None
    if inputs.locagent_document is not None:
        document = json.loads(Path(inputs.locagent_document).read_bytes())
        prefix = (locagent_manifest or {}).get("import_root") or ""
        indexed = locagent_indexed_identities(document, scope, prefix)
        unindexed = locagent_unindexed_identities(document, scope, prefix, indexed)
        del document
    locagent_index = {"document": _file_record(inputs.locagent_document),
                      "indexed_package_functions": None if indexed is None else len(indexed),
                      "L_callees_not_indexed": None if indexed is None else sum(
                          1 for callee in truth.callees() if truth.callers_by(callee, "L") and callee not in indexed),
                      "unindexed_rule": "not_applied_no_locagent_document" if unindexed is None else "applied",
                      "unindexed_functions": None if unindexed is None else len(unindexed),
                      "unindexed_sample": None if unindexed is None else unindexed[:50],
                      "function_node_only_unindexed": None if indexed is None else len(set(scope.table()) - indexed)}
    coverage = render_module.coverage_for(inputs.package_dirs)
    renderer = render_module.ItemRenderer(scope, source_sha256=source_hash, coverage=coverage)
    jedi_failed = jedi_definition_errors(jedi_manifest)
    table = scope.table()
    jedi_errors = {"definitions": len(jedi_failed),
                   "by_stage": dict(sorted(Counter(jedi_failed.values()).items())),
                   "not_package_functions": sorted(identity for identity in jedi_failed if identity not in table)}
    rows = candidate_rows(inputs.family, scope, truth, jedi_failed, unindexed, omit_rule)
    ranked = sorted((row for row in rows if row["excluded"] is None),
                    key=lambda row: (row["selection_key"], row["identity"]))
    eligible, render_stage = [], Counter()
    text_rule = Counter()
    for row in ranked:
        reason, detail, state = evaluate_candidate(scope, truth, renderer, row["identity"], search=search,
                                                   indexed=indexed, omit_rule=omit_rule)
        render_stage["evaluated"] += 1
        row["excluded"], row["detail"] = reason, detail
        if reason == "render_truncated":
            render_stage.update("truncated:" + variant for variant in detail["truncated_variants"])
        elif reason == "incidental_visibility":
            render_stage["visibility:snippet_covers_site"] += detail["snippet_covers_site"]
            render_stage["visibility:identity_in_payload"] += detail["identity_in_payload"]
        elif reason == "docref_visibility":
            render_stage["visibility:docref"] += 1
        if detail is not None and "text_matches" in detail:
            text_rule["reached_visibility"] += 1
            text_rule["text_rule_would_exclude"] += detail["text_matches"] > 0
        if reason is None:
            row["eligible_rank"] = len(eligible)
            if len(eligible) < m:
                eligible.append((row, state))
            else:
                eligible.append((row, None))
    selected = [(row, state) for row, state in eligible if state is not None][:m]
    targets = [row["identity"] for row, _ in selected]
    pairs = wording.assign_pairs(targets)
    in_edge = first_by_key(targets, edge_key, S_EDGE_PER_FAMILY)
    in_tool = first_by_key(targets, tool_key, T_PER_FAMILY)
    items, payloads, facts = [], {}, []
    site_notes = Counter()
    for index_, (row, state) in enumerate(selected):
        target = row["identity"]
        row["selected"] = True
        aliases = truth.aliases(target)
        names = lexical_names(scope, target, aliases)
        lines = lexical_lines(scope, names)
        verified_before = truth.verified(target)
        truth.add_records("lexical", [{"caller": caller, "callee": target, "path": path, "line": number}
                                      for path, number in lines
                                      for caller in [scope.enclosing(path, number)] if caller is not None])
        if truth.verified(target) != verified_before:
            raise AssertionError("Lexical candidates changed V(X)")
        summary = truth.summary(target)
        site_notes.update(truth.site_notes(target, [entry["identity"] for entry in summary["V"]]))
        prepared, omitted = state["prepared"], state["omitted"]
        pair = pairs[target]
        cells = state["cells"][pair]
        render_module.check_invariants(target, omitted["identity"], cells, pair, verification=state["verification"])
        variants, item_payloads = render_module.variant_records(prepared, cells)
        payloads.update(item_payloads)
        runtime = [entry["identity"] for entry in summary["V"] if entry["runtime_witnessed"]]
        natural = state["natural"]
        natural_miss = bool(set(runtime) - set(natural))
        natural_truncated = prepared["info"]["natural"]["payload_truncated"]
        in_s_nat = natural_miss and not natural_truncated
        omitted_sites = next(entry["sites"] for entry in summary["V"] if entry["identity"] == omitted["identity"])
        info = scope.table()[target]
        identifier = item_id(inputs.family, target)
        verified_set, natural_set = {entry["identity"] for entry in summary["V"]}, set(natural)
        record = {
            "schema": ITEM_SCHEMA, "item_id": identifier, "family": inputs.family, "role": inputs.role,
            "commit": inputs.commit, "package_dirs": list(inputs.package_dirs),
            "target": {"identity": target, "path": info["path"], "line": info["lineno"],
                       "end_line": info["end_lineno"], "qualname": split_identity(target)[1],
                       "name": simple_name(target)},
            "V": summary["V"], "U": summary["U"],
            "omitted": omitted,
            "wording_pair": pair,
            "covariates": {"stratum": omitted["stratum"],
                           "c4_contains_omitted_site": (None if state["verification"] is None
                                                        else _hits_in_sites(state["verification"], omitted_sites)),
                           "c4_trigger_fires": prepared["info"]["omit_node"]["c4_trigger_fires"],
                           "natural_miss": natural_miss, "n_V": len(summary["V"]),
                           "visibility_text_matches": state["visibility"]["text_matches"],
                           "visibility_docref_mentions": state["visibility"]["docref_mentions"],
                           "visibility_name_mentions": state["visibility"]["name_mentions"]},
            "membership": {"in_s_nat": in_s_nat, "in_s_edge": target in in_edge, "in_t": target in in_tool,
                           "s_nat_excluded_reason": (None if in_s_nat else
                                                     "no_natural_miss" if not natural_miss
                                                     else "natural_render_truncated")},
            "lexical_sites": [{"path": path, "line": number} for path, number in lines],
            "render": {"protocol": render_module.RENDER_PROTOCOL, "source_sha256": source_hash,
                       "snippet_caps": prepared["caps"], "cap_bytes": render_module.MAX_OUTPUT_BYTES,
                       "c4_cap_bytes": render_module.SEARCH_MAX_BYTES,
                       "query": {"relation": render_module.RELATION, "direction": render_module.DIRECTION,
                                 "limit": render_module.QUERY_LIMIT}},
            "variants": variants,
            "eligibility": {"eligible": True, "selection_key": row["selection_key"],
                            "eligible_rank": row["eligible_rank"], "selection_index": index_,
                            "rules_passed": [rule for rule in order
                                             if rule != "locagent_unindexed" or unindexed is not None],
                            "n_runtime_witnessed": len(runtime),
                            "truncation": {variant: prepared["info"][variant]["payload_truncated"]
                                           for variant in render_module.VARIANTS},
                            "incidental_visibility": {"snippet_covers_site": False, "identity_in_payload": False},
                            "mechanical_diffs": "passed_both_pairs"},
            "keys": {"wording": wording.assignment_key(target), "edge": edge_key(target), "tool": tool_key(target)},
            "names": {"simple": simple_name(target), "qualified": split_identity(target)[1],
                      "import_aliases": aliases, "lexical": names},
            "natural": {"callers": natural, "not_verified": sorted(natural_set - verified_set),
                        "missing_verified": sorted(verified_set - natural_set),
                        "missing_runtime_witnessed": sorted(set(runtime) - natural_set),
                        "target_indexed": state["natural_indexed"] if indexed is not None else None},
            "verification": _verification_summary(state["verification"], source_hash),
        }
        if record["verification"] is not None and (record["verification"]["error"]
                                                   or not record["verification"]["source_sha256_matches"]):
            raise ValueError(f"{inputs.family}: the C4 block of {target} has an error or another source tree")
        items.append(record)
        for variant in render_module.VARIANTS:
            facts.append({"item_id": identifier, "target": target, "variant": variant,
                          "facts_sha256": variants[variant]["facts_sha256"],
                          "document": prepared["documents"][variant].data})
    l_pairs = [(caller, callee) for callee in truth.callees() for caller in truth.callers_by(callee, "L")]
    graph_facts = render_module.family_facts_document(scope, l_pairs, indexed, source_sha256=source_hash,
                                                      coverage=coverage)
    exclusions = Counter(row["excluded"] for row in rows if row["excluded"])
    funnel = {"functions": len(rows),
              "excluded": {reason: exclusions.get(reason, 0) for reason in order},
              "eligible": len(eligible), "selected": len(selected),
              "eligible_not_selected": len(eligible) - len(selected),
              "render_stage": dict(sorted(render_stage.items())), "text_rule": dict(sorted(text_rule.items()))}
    summary = {"family": inputs.family, "role": inputs.role, "commit": inputs.commit, "url": inputs.url,
               "repo_root": str(inputs.repo_root), "package_dirs": list(inputs.package_dirs),
               "source_sha256": source_hash,
               "inputs": {"trace_dir": None if inputs.trace_dir is None else str(inputs.trace_dir),
                          "trace_edges": _file_record(inputs.trace_edges),
                          "trace_manifest": _file_record(inputs.trace_manifest),
                          "locagent_edges": _file_record(inputs.locagent_edges),
                          "locagent_manifest": _file_record(inputs.locagent_manifest),
                          "jedi_edges": _file_record(inputs.jedi_edges),
                          "jedi_manifest": _file_record(inputs.jedi_manifest)},
               "trace_status": trace_info["status"], "trace": trace_info,
               "trace_source_tree_matches": (None if trace_info.get("manifest_source_tree_sha256") is None
                                             else trace_info["manifest_source_tree_sha256"] == source_hash),
               "static_status": static_status, "locagent_index": locagent_index, "jedi_definition_errors": jedi_errors,
               "index_errors": dict(sorted(scope.index.errors.items())) if hasattr(scope.index, "errors") else {},
               "record_counts": record_counts, "truth_notes": dict(sorted(truth.notes.items())),
               "site_notes_selected_V": dict(sorted(site_notes.items())),
               "package_files": len(scope.files()), "functions": len(scope.table()),
               "callees_with_evidence": len(truth.callees()), "funnel": funnel,
               "selected": len(items), "short_of_m": len(items) < m,
               "omitted_name_visibility": {
                   "items_with_docref_mentions": sum(item["covariates"]["visibility_docref_mentions"] > 0
                                                     for item in items),
                   "items_with_name_mentions": sum(item["covariates"]["visibility_name_mentions"] > 0
                                                   for item in items),
                   "items_with_text_matches": sum(item["covariates"]["visibility_text_matches"] > 0
                                                  for item in items)},
               "graph_facts": {"nodes": len(graph_facts["nodes"]), "edges": len(graph_facts["edges"])}}
    return {"items": items, "payloads": payloads, "facts": facts, "rows": rows, "summary": summary,
            "graph_facts": graph_facts}


def _module_hashes():
    hashes = {}
    for name in ("truth.py", "items.py", "render.py", "wording.py", "identity.py", "tools.py", "tracer.py"):
        if (_PROBE_DIR / name).is_file():
            hashes[f"probe/{name}"] = hashlib.sha256((_PROBE_DIR / name).read_bytes()).hexdigest()
    for name in ("graphs.py", "data.py", "interventions.py"):
        hashes[name] = hashlib.sha256((_PROBE_DIR.parent / name).read_bytes()).hexdigest()
    script = _PROBE_DIR.parents[1] / "scripts" / "expA_items.py"
    if script.is_file():
        hashes["scripts/expA_items.py"] = hashlib.sha256(script.read_bytes()).hexdigest()
    return hashes


def _write_atomic(out, relative, data):
    path = Path(out) / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"{path} exists; outputs are immutable, use a new directory")
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return {"path": relative, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def _jsonl(rows):
    return b"".join(canonical_json(row) + b"\n" for row in rows)


def _histogram(values):
    return {str(key): count for key, count in sorted(Counter(values).items(), key=lambda item: str(item[0]))}


def build_items(families, main_roster, out_dir, *, m=M_PER_FAMILY, search_factory=None,
                index_factory=None, code_git_sha=None, require_c4=True, omit_rule=OMIT_RULE):
    """Build every family, then write items/, payloads/, facts/, eligibility/, graph_facts/, manifest."""
    order = exclusion_order(omit_rule)
    out = Path(out_dir)
    for name in ("items.jsonl", "manifest.json"):
        if (out / name).exists():
            raise FileExistsError(f"{out / name} exists; item builds are immutable, use a new directory")
    if type(m) is not int or m < 1:
        raise ValueError("m must be a positive integer")
    all_items, family_summaries, outputs = [], {}, {}
    for inputs in sorted(families, key=lambda item: item.family):
        search = None if search_factory is None else search_factory(inputs)
        if search is None and require_c4:
            raise ValueError("C4a requires the shared lexical search; pass require_c4=False only for development")
        index = None if index_factory is None else index_factory(inputs)
        result = build_family(inputs, m=m, search=search, index=index, omit_rule=omit_rule)
        family = inputs.family
        payload_rows = [{"sha256": sha, "bytes": len(canonical_json(payload)), "payload": payload}
                        for sha, payload in sorted(result["payloads"].items())]
        outputs[family] = {
            "payloads": _write_atomic(out, f"payloads/{family}.jsonl", _jsonl(payload_rows)),
            "facts": _write_atomic(out, f"facts/{family}.jsonl", _jsonl(result["facts"])),
            "eligibility": _write_atomic(out, f"eligibility/{family}.jsonl", _jsonl(result["rows"])),
            "graph_facts": _write_atomic(out, f"graph_facts/{family}.json", canonical_json(result["graph_facts"]))}
        outputs[family]["payloads"]["count"] = len(payload_rows)
        family_summaries[family] = {**result["summary"], "outputs": outputs[family]}
        all_items.extend(result["items"])
    items_record = _write_atomic(out, "items.jsonl", _jsonl(all_items))
    items_record["records"] = len(all_items)
    counts = {
        "families": len(family_summaries), "items": len(all_items),
        "by_role": _histogram(item["role"] for item in all_items),
        "by_family": {family: summary["selected"] for family, summary in family_summaries.items()},
        "families_short_of_m": sorted(family for family, summary in family_summaries.items() if summary["short_of_m"]),
        "families_without_items": sorted(family for family, summary in family_summaries.items()
                                         if summary["selected"] == 0),
        "families_without_locagent_unindexed_rule": sorted(
            family for family, summary in family_summaries.items()
            if summary["locagent_index"]["unindexed_rule"] != "applied"),
        "main_roster_missing_from_build": sorted(set(main_roster) - set(family_summaries)),
        "V_callers_without_sites": sum(not entry["sites"] for item in all_items for entry in item["V"]),
        "V_sites": sum(len(entry["sites"]) for item in all_items for entry in item["V"]),
        # A site witnessed by the AST rule alone (no trace, L or J line inside it), for example a
        # local variable that shares X's simple name: kept by the D2 name rule, counted here.
        "V_sites_ast_only": sum(site["witnesses"] == ["ast"] for item in all_items for entry in item["V"]
                                for site in entry["sites"]),
        "V_callers": sum(len(item["V"]) for item in all_items),
        "V_runtime_witnessed": sum(entry["runtime_witnessed"] for item in all_items for entry in item["V"]),
        "V_strata": _histogram(entry["stratum"] for item in all_items for entry in item["V"]),
        "omitted_site_kinds": _histogram(site["kind"] for item in all_items for entry in item["V"]
                                         if entry["identity"] == item["omitted"]["identity"]
                                         for site in entry["sites"]),
        "V_count_histogram": _histogram(item["covariates"]["n_V"] for item in all_items),
        "omitted_stratum": _histogram(item["covariates"]["stratum"] for item in all_items),
        "wording_pair": _histogram(item["wording_pair"] for item in all_items),
        "natural_miss": sum(item["covariates"]["natural_miss"] for item in all_items),
        "natural_render_truncated": sum(item["variants"]["natural"]["payload_truncated"] for item in all_items),
        "s_nat_excluded_natural_truncated": sum(item["membership"]["s_nat_excluded_reason"]
                                                == "natural_render_truncated" for item in all_items),
        "membership": {key: sum(item["membership"][key] for item in all_items)
                       for key in ("in_s_nat", "in_s_edge", "in_t")},
        "c4_trigger_fires": sum(item["covariates"]["c4_trigger_fires"] for item in all_items),
        "c4_contains_omitted_site": sum(bool(item["covariates"]["c4_contains_omitted_site"]) for item in all_items),
        "visibility_text_matches_positive": sum(item["covariates"]["visibility_text_matches"] > 0 for item in all_items),
        "visibility_docref_mentions_positive": sum(item["covariates"]["visibility_docref_mentions"] > 0
                                                   for item in all_items),
        "visibility_name_mentions_positive": sum(item["covariates"]["visibility_name_mentions"] > 0
                                                 for item in all_items),
        "omitted_name_visibility_by_family": {
            family: summary["omitted_name_visibility"] | {"selected": summary["selected"]}
            for family, summary in family_summaries.items()},
        "c4_built": all(item["verification"] is not None for item in all_items),
        "c4_token_bounds": sorted({str(item["verification"]["token_bound"]) for item in all_items
                                   if item["verification"] is not None}),
        "funnel_total": {reason: sum(summary["funnel"]["excluded"][reason] for summary in family_summaries.values())
                         for reason in order},
        "eligible_total": sum(summary["funnel"]["eligible"] for summary in family_summaries.values())}
    manifest = {"protocol": ITEMS_PROTOCOL, "item_schema": ITEM_SCHEMA, "truth_protocol": TRUTH_PROTOCOL,
                "render_protocol": render_module.RENDER_PROTOCOL,
                "code": {"git_sha": code_git_sha or "unknown", "modules_sha256": _module_hashes()},
                "parameters": {"m_per_family": m, "verified_bounds": [MIN_VERIFIED, MAX_VERIFIED],
                               "salts": {"selection": SELECTION_SALT, "omit": OMIT_SALT, "edge": EDGE_SALT,
                                         "tool": TOOL_SALT, "wording": wording.WORDING_SALT,
                                         "item_id": ITEM_SCHEMA},
                               "omit_rule": omit_rule, "s_edge_per_family": S_EDGE_PER_FAMILY,
                               "t_per_family": T_PER_FAMILY, "roles": list(ROLES),
                               "exclusion_order": list(order),
                               "jedi_definition_error_stages": list(JEDI_DEFINITION_ERROR_STAGES),
                               "locagent_unindexed": "D12.7: X or a J-reported caller of X has no LocAgent node "
                                                     "(function node, or the first direct __init__ of a class node: "
                                                     "class_node_init)",
                               "allowed_decorators": sorted(ALLOWED_DECORATORS),
                               "visibility_text_min_chars": VISIBILITY_TEXT_MIN_CHARS,
                               "query": {"relation": render_module.RELATION, "direction": render_module.DIRECTION,
                                         "limit": render_module.QUERY_LIMIT},
                               "max_output_bytes": render_module.MAX_OUTPUT_BYTES,
                               "snippet_max_bytes": render_module.SNIPPET_MAX_BYTES,
                               "c4a": {"search_limit": render_module.SEARCH_LIMIT,
                                       "max_bytes": render_module.SEARCH_MAX_BYTES,
                                       "token_bound": render_module.C4_TOKEN_BOUND,
                                       "kind": render_module.VERIFICATION_KIND},
                               "variants": list(render_module.VARIANTS),
                               "cells": list(render_module.CELL_CONDITIONS),
                               "payload_serialization": "canonical_json (sorted keys, compact, UTF-8)"},
                "invariants": render_module.INVARIANTS,
                "blocks": [{"block": block, "variant": variant, "conditions": list(conditions),
                            "membership": member} for block, variant, conditions, member in BLOCK_CELLS],
                "wording": wording.wording_manifest(), "main_roster": sorted(main_roster),
                "families": family_summaries, "counts": counts,
                "outputs": {"items.jsonl": items_record}}
    _write_atomic(out, "manifest.json", canonical_json(manifest) + b"\n")
    return manifest


# ------------------------------------------------------------------ reading
def read_items(items_dir):
    """(items, manifest, items_bytes, manifest_bytes) after checking items.jsonl against the manifest."""
    directory = Path(items_dir)
    manifest_bytes = (directory / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    items_bytes = (directory / "items.jsonl").read_bytes()
    if hashlib.sha256(items_bytes).hexdigest() != manifest["outputs"]["items.jsonl"]["sha256"]:
        raise ValueError("items.jsonl does not match the manifest sha256")
    items = [json.loads(raw) for raw in items_bytes.splitlines() if raw.strip()]
    return items, manifest, items_bytes, manifest_bytes


def read_payload_store(path, expected_sha256=None):
    data = Path(path).read_bytes()
    if expected_sha256 is not None and hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError(f"{path} does not match its recorded sha256")
    store = {}
    for raw in data.splitlines():
        row = json.loads(raw)
        encoded = canonical_json(row["payload"])
        if hashlib.sha256(encoded).hexdigest() != row["sha256"] or row["bytes"] != len(encoded):
            raise ValueError(f"{path}: payload row {row['sha256']} does not match its bytes")
        store[row["sha256"]] = row["payload"]
    return store


def load_payload(items_dir, family, sha256):
    """Return the stored payload with this sha256, verifying its bytes."""
    with open(Path(items_dir) / "payloads" / f"{family}.jsonl", "rb") as stream:
        for raw in stream:
            row = json.loads(raw)
            if row["sha256"] == sha256:
                if hashlib.sha256(canonical_json(row["payload"])).hexdigest() != sha256:
                    raise ValueError("Stored payload does not match its sha256")
                return row["payload"]
    raise KeyError(sha256)


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def verify_items(items_dir):
    """Re-check the frozen build from its own files: hashes, selection rules, invariants, visibility."""
    directory = Path(items_dir)
    items, manifest, _, _ = read_items(directory)
    omit_rule = manifest["parameters"].get("omit_rule", OMIT_RULE)
    order = exclusion_order(omit_rule)
    stores, family_items = {}, {}
    for family, summary in manifest["families"].items():
        outputs = summary["outputs"]
        for key in ("facts", "eligibility", "graph_facts"):
            data = (directory / outputs[key]["path"]).read_bytes()
            _check(hashlib.sha256(data).hexdigest() == outputs[key]["sha256"], f"{family}: {key} sha256 differs")
        stores[family] = read_payload_store(directory / outputs["payloads"]["path"], outputs["payloads"]["sha256"])
        facts = json.loads((directory / outputs["graph_facts"]["path"]).read_bytes())
        _check(facts.get("backend") == "locagent" and facts.get("source_sha256") == summary["source_sha256"]
               and not facts.get("error"), f"{family}: graph facts are not this family's LocAgent facts")
        rows = [json.loads(raw) for raw in (directory / outputs["eligibility"]["path"]).read_bytes().splitlines()]
        family_items[family] = {"rows": rows, "items": []}
    seen = set()
    checked = Counter()
    for item in items:
        family = item["family"]
        _check(item.get("schema") == ITEM_SCHEMA, f"{item.get('item_id')}: schema")
        _check(item["item_id"] == item_id(family, item["target"]["identity"]) and item["item_id"] not in seen,
               f"{item['item_id']}: item id")
        seen.add(item["item_id"])
        family_items[family]["items"].append(item)
        target = item["target"]["identity"]
        _check(item["role"] in ROLES, f"{item['item_id']}: role")
        _check("<locals>" not in item["target"]["qualname"].split("."), f"{item['item_id']}: nested target")
        V = {entry["identity"]: entry for entry in item["V"]}
        _check(MIN_VERIFIED <= len(V) <= MAX_VERIFIED and item["covariates"]["n_V"] == len(V),
               f"{item['item_id']}: |V|")
        _check(target not in V, f"{item['item_id']}: target in V")
        for entry in item["V"]:
            witnessed = any("trace" in site["witnesses"] for site in entry["sites"])
            _check(entry["runtime_witnessed"] == witnessed and (not witnessed or "trace" in entry["sources"]),
                   f"{item['item_id']}: runtime_witnessed of {entry['identity']}")
            _check(set(entry["sources"]) & {"trace", "L", "J"}, f"{item['item_id']}: V without a verifying source")
            _check(entry["stratum"] in STRATA + (None,), f"{item['item_id']}: stratum")
        runtime = sorted(identity for identity, entry in V.items() if entry["runtime_witnessed"])
        pool = runtime
        if omit_rule == HARD_OMIT_RULE:  # the grep_all site rule, recomputed from the item's own lexical_sites
            lexical_lines_ = [(site["path"], site["line"]) for site in item["lexical_sites"]]
            pool = [caller for caller in runtime
                    if not any(site["path"] == path and site["lineno"] <= line <= site["end_lineno"]
                               for site in V[caller]["sites"] for path, line in lexical_lines_)]
        omitted = item["omitted"]["identity"]
        _check(pool and omitted == min(pool, key=lambda caller: (omit_key(target, caller), caller)),
               f"{item['item_id']}: omitted caller rule")
        _check(item["omitted"].get("rule") == omit_rule and V[omitted]["sites"],
               f"{item['item_id']}: the omitted caller needs the build's rule and at least one site")
        _check(item["omitted"]["order_key"] == omit_key(target, omitted)
               and item["omitted"]["stratum"] == V[omitted]["stratum"] == item["covariates"]["stratum"]
               and item["omitted"]["stratum"] in STRATA, f"{item['item_id']}: omitted record")
        natural_miss = bool(set(runtime) - set(item["natural"]["callers"]))
        _check(item["covariates"]["natural_miss"] == natural_miss, f"{item['item_id']}: natural_miss")
        membership = item["membership"]
        truncated = item["variants"]["natural"]["payload_truncated"]
        _check(membership["in_s_nat"] == (natural_miss and not truncated)
               and membership["s_nat_excluded_reason"] == (None if membership["in_s_nat"] else "no_natural_miss"
                                                           if not natural_miss else "natural_render_truncated"),
               f"{item['item_id']}: S-NAT membership")
        for variant in render_module.STRICT_VARIANTS:
            _check(not item["variants"][variant]["payload_truncated"], f"{item['item_id']}: {variant} truncated")
        store = stores[family]
        rendered = {variant: {condition: (None if sha is None else store[sha])
                              for condition, sha in entry["cells"].items()}
                    for variant, entry in item["variants"].items()}
        for variant, entry in item["variants"].items():
            _check(entry["endpoint_ids"] == [node["id"] for node in rendered[variant]["C1"]["endpoints"]],
                   f"{item['item_id']}: {variant} endpoint ids")
            for condition, sha in entry["cells"].items():
                _check(entry["cell_bytes"][condition] == (None if sha is None
                                                          else len(canonical_json(rendered[variant][condition]))),
                       f"{item['item_id']}: {variant}/{condition} bytes")
        block = rendered["omit_node"]["C4a"]["verification"] if rendered["omit_node"]["C4a"] else None
        try:
            render_module.check_invariants(target, omitted, rendered, item["wording_pair"], verification=block)
        except AssertionError as error:
            raise ValueError(f"{item['item_id']}: {error}") from None
        omit_c1 = rendered["omit_node"]["C1"]
        _check(not render_module.graph_identity_exposed(omitted, rendered["omit_node"]),
               f"{item['item_id']}: omitted identity exposed in the omit_node graph payload")
        _check(not render_module.snippet_site_overlaps(omit_c1, V[omitted]["sites"]),
               f"{item['item_id']}: a delivered snippet covers a site of the omitted caller")
        _check(item["covariates"]["c4_trigger_fires"] == render_module.c4_triggered(omit_c1),
               f"{item['item_id']}: c4_trigger_fires")
        mentions = render_module.omitted_name_mentions(omit_c1, omitted)
        _check((item["covariates"].get("visibility_docref_mentions"), item["covariates"].get("visibility_name_mentions"))
               == (mentions["docref"], mentions["name"]), f"{item['item_id']}: visibility name-mention covariates")
        _check(item["covariates"]["c4_contains_omitted_site"] == (None if block is None
                                                                  else _hits_in_sites(block, V[omitted]["sites"])),
               f"{item['item_id']}: c4_contains_omitted_site")
        lexical = [(site["path"], site["line"]) for site in item["lexical_sites"]]
        _check(lexical == sorted(set(lexical)), f"{item['item_id']}: lexical_sites order")
        checked["items"] += 1
    for family, group in family_items.items():
        selected = group["items"]
        targets = [item["target"]["identity"] for item in selected]
        _check([item["eligibility"]["selection_key"] for item in selected]
               == sorted(selection_key(family, target) for target in targets), f"{family}: selection order")
        _check(all(item["eligibility"]["eligible_rank"] == item["eligibility"]["selection_index"] == position
                   for position, item in enumerate(selected)), f"{family}: selection is not the first eligible")
        rows = {row["identity"]: row for row in group["rows"]}
        _check({identity for identity, row in rows.items() if row["selected"]} == set(targets),
               f"{family}: eligibility rows disagree with the items")
        _check(all(rows[target]["excluded"] is None for target in targets),
               f"{family}: a selected target has an exclusion reason")
        _check(all(row["excluded"] in order for row in rows.values() if row["excluded"] is not None),
               f"{family}: unknown exclusion reason")
        eligible_keys = sorted(row["selection_key"] for row in rows.values() if row["excluded"] is None)
        _check(eligible_keys[:len(targets)] == [item["eligibility"]["selection_key"] for item in selected],
               f"{family}: an eligible target with a smaller key was skipped")
        pairs = wording.assign_pairs(targets)
        edge, tool = first_by_key(targets, edge_key, S_EDGE_PER_FAMILY), first_by_key(targets, tool_key, T_PER_FAMILY)
        for item in selected:
            target = item["target"]["identity"]
            _check(item["wording_pair"] == pairs[target], f"{item['item_id']}: wording pair assignment")
            _check(item["membership"]["in_s_edge"] == (target in edge) and item["membership"]["in_t"] == (target in tool),
                   f"{item['item_id']}: S-EDGE/T membership")
        checked["families"] += 1
    return {"items_checked": checked["items"], "families": checked["families"],
            "payloads": sum(len(store) for store in stores.values()), "ok": True}


# ------------------------------------------------------------------ dispatch view (D10.7)
def dispatch_cells(item):
    """{variant: {condition: sha256}} for the item's payload-bearing D8 cells."""
    cells = {}
    for block, variant, conditions, member in BLOCK_CELLS:
        if member is not None and not item["membership"][member]:
            continue
        for condition in conditions:
            sha = item["variants"][variant]["cells"][condition]
            if sha is None:
                raise ValueError(f"{item['item_id']}: {block} needs {variant}/{condition}, which was not built "
                                 "(a build without the shared search cannot be dispatched)")
            cells.setdefault(variant, {})[condition] = sha
    return cells


def dispatch_row(item):
    """The contract item 4 projection of one item: no V, U, sites, C*, covariates or reasons."""
    return {"item_id": item["item_id"], "family": item["family"],
            "target": {key: item["target"][key] for key in DISPATCH_TARGET_KEYS},
            "package_dirs": list(item["package_dirs"]), "wording_pair": item["wording_pair"],
            "membership": {key: item["membership"][key] for key in DISPATCH_MEMBERSHIP_KEYS},
            "cells": dispatch_cells(item)}


def _truth_keys(value, found=None):
    found = set() if found is None else found
    if isinstance(value, dict):
        for key, inner in value.items():
            if key in TRUTH_KEYS:
                found.add(key)
            _truth_keys(inner, found)
    elif isinstance(value, list):
        for inner in value:
            _truth_keys(inner, found)
    return found


def _dispatch_hashes():
    hashes = {}
    for name in ("items.py", "render.py", "wording.py", "truth.py"):
        hashes[f"probe/{name}"] = hashlib.sha256((_PROBE_DIR / name).read_bytes()).hexdigest()
    script = _PROBE_DIR.parents[1] / "scripts" / "expA_items.py"
    if script.is_file():
        hashes["scripts/expA_items.py"] = hashlib.sha256(script.read_bytes()).hexdigest()
    return hashes


def dispatched_store(store_bytes, referenced):
    """The rows of an items payload store whose sha256 a dispatch row names, in store order (bytes kept)."""
    lines = [raw for raw in store_bytes.splitlines() if raw.strip()]
    kept = [raw for raw in lines if json.loads(raw)["sha256"] in referenced]
    found = {json.loads(raw)["sha256"] for raw in kept}
    _check(found == set(referenced), f"{len(set(referenced) - found)} dispatched payloads are not in the items store")
    return b"".join(raw + b"\n" for raw in kept)


def build_dispatch(items_dir, out_dir, *, code_git_sha=None):
    """Write the dispatch view (contract item 4) and verify it against the items build.

    Files: ``dispatch.jsonl`` (one row per item, items.jsonl order), the payload store
    restricted to the dispatched cells (``payloads/<family>.jsonl``: exactly the rows of the
    items store that some dispatch row names, byte for byte, D10.7 "only what dispatch
    needs"), a byte-identical copy of the natural LocAgent facts (``graph_facts/<family>.json``)
    and ``dispatch_manifest.json``, which pins both the dispatched store and the items store
    (``items_payload_store``, for scoring on Core2). The build is verified (``verify_items``)
    first; a build that does not verify is never dispatched.

    The view has no truth field, but the runtime host can still derive V and C* from the
    payloads it needs (vcomplete C1 for S-VC minus the omit_node C1 endpoints): the tool
    jail, not the file selection, keeps them from the model (review findings on D10.7).
    """
    items_dir, out = Path(items_dir), Path(out_dir)
    for name in ("dispatch.jsonl", "dispatch_manifest.json"):
        if (out / name).exists():
            raise FileExistsError(f"{out / name} exists; the dispatch view is immutable, use a new directory")
    verify_items(items_dir)
    items, manifest, items_bytes, manifest_bytes = read_items(items_dir)
    rows = [dispatch_row(item) for item in items]
    referenced = {}
    for row in rows:
        referenced.setdefault(row["family"], set()).update(
            sha for conditions in row["cells"].values() for sha in conditions.values())
    families = {}
    for family in sorted({item["family"] for item in items}):
        summary = manifest["families"][family]
        store = (items_dir / summary["outputs"]["payloads"]["path"]).read_bytes()
        facts = (items_dir / summary["outputs"]["graph_facts"]["path"]).read_bytes()
        _check(hashlib.sha256(store).hexdigest() == summary["outputs"]["payloads"]["sha256"],
               f"{family}: payload store differs from the items manifest")
        _check(hashlib.sha256(facts).hexdigest() == summary["outputs"]["graph_facts"]["sha256"],
               f"{family}: graph facts differ from the items manifest")
        subset = dispatched_store(store, referenced.get(family, set()))
        families[family] = {"source_sha256": summary["source_sha256"], "package_dirs": summary["package_dirs"],
                            "role": summary["role"],
                            "payload_store": {**_write_atomic(out, f"payloads/{family}.jsonl", subset),
                                              "rows": len(referenced.get(family, ()))},
                            "items_payload_store": {"sha256": summary["outputs"]["payloads"]["sha256"],
                                                    "rows": summary["outputs"]["payloads"]["count"]},
                            "graph_facts": {**_write_atomic(out, f"graph_facts/{family}.json", facts),
                                            **summary["graph_facts"]}}
    dispatch_record = _write_atomic(out, "dispatch.jsonl", _jsonl(rows))
    payload_cells = Counter()
    for row in rows:
        for variant, conditions in row["cells"].items():
            payload_cells.update(f"{variant}/{condition}" for condition in conditions)
    dispatch_manifest = {
        "schema": DISPATCH_SCHEMA, "items_sha256": hashlib.sha256(items_bytes).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "dispatch_sha256": dispatch_record["sha256"], "rows": len(rows),
        "by_family": dict(sorted(Counter(row["family"] for row in rows).items())),
        "payload_store": {family: entry["payload_store"] for family, entry in families.items()},
        "items_payload_store": {family: entry["items_payload_store"] for family, entry in families.items()},
        "graph_facts": {family: entry["graph_facts"] for family, entry in families.items()},
        "families": {family: {key: entry[key] for key in ("source_sha256", "package_dirs", "role")}
                     for family, entry in families.items()},
        "code": {"items_build": manifest["code"], "dispatch": {"git_sha": code_git_sha or "unknown",
                                                               "modules_sha256": _dispatch_hashes()}},
        "row_keys": list(DISPATCH_ROW_KEYS), "membership_keys": list(DISPATCH_MEMBERSHIP_KEYS),
        "payload_cells": [{"block": block, "variant": variant, "conditions": list(conditions), "membership": member}
                          for block, variant, conditions, member in BLOCK_CELLS],
        "no_payload_cells": [{"block": block, "variant": variant, "conditions": list(conditions), "membership": member}
                             for block, variant, conditions, member in NO_PAYLOAD_CELLS],
        "payload_cell_counts": dict(sorted(payload_cells.items())),
        "membership_counts": {key: sum(bool(row["membership"][key]) for row in rows)
                              for key in DISPATCH_MEMBERSHIP_KEYS},
        "c4a": {"token_bound": render_module.C4_TOKEN_BOUND, "max_bytes": render_module.SEARCH_MAX_BYTES,
                "max_hits": render_module.SEARCH_LIMIT},
        "notes": ["payloads/<family>.jsonl holds exactly the items-store rows of the dispatched cells (byte-identical "
                  "rows, store order); items_payload_store pins the full items store that scoring reads on Core2",
                  "no truth field is shipped, but V and C* are derivable on the runtime host from the S-VC and P "
                  "payloads (vcomplete C1 minus omit_node C1); the tool jail is the isolation boundary",
                  "graph_facts/<family>.json are the family's natural LocAgent invokes facts for the T arms' "
                  "graph_query (repository-relative identities, the same node format as the item answers)",
                  "C0 and T cells have no payload; T exists for membership.in_t; seeds and blocks are the "
                  "harness design",
                  "the text inserted into a prompt is canonical_json(payload)"]}
    _write_atomic(out, "dispatch_manifest.json", canonical_json(dispatch_manifest) + b"\n")
    return verify_dispatch(out, items_dir)


def verify_dispatch(dispatch_dir, items_dir=None):
    """Check the dispatch view: exact row fields, no truth field, hashes, payloads and facts.

    With ``items_dir`` it also checks that every row is the projection of its item, that
    the copies are the build's own files, and that no dispatched omit_node graph payload
    names C*.
    """
    directory = Path(dispatch_dir)
    manifest_bytes = (directory / "dispatch_manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    dispatch_bytes = (directory / "dispatch.jsonl").read_bytes()
    _check(manifest.get("schema") == DISPATCH_SCHEMA, "dispatch manifest schema")
    _check(hashlib.sha256(dispatch_bytes).hexdigest() == manifest.get("dispatch_sha256"),
           "dispatch.jsonl does not match the dispatch manifest")
    manifest_truth = _truth_keys({key: value for key, value in manifest.items() if key != "code"})
    _check(not manifest_truth, f"dispatch manifest carries truth fields {sorted(manifest_truth)}")
    rows = [json.loads(raw) for raw in dispatch_bytes.splitlines() if raw.strip()]
    _check(len(rows) == manifest["rows"], "dispatch row count")
    stores = {family: read_payload_store(directory / entry["path"], entry["sha256"])
              for family, entry in manifest["payload_store"].items()}
    for family, entry in manifest["graph_facts"].items():
        data = (directory / entry["path"]).read_bytes()
        _check(hashlib.sha256(data).hexdigest() == entry["sha256"], f"{family}: graph facts sha256")
        facts = json.loads(data)
        _check(facts.get("backend") == "locagent" and not facts.get("error")
               and facts.get("source_sha256") == manifest["families"][family]["source_sha256"],
               f"{family}: graph facts are not this family's LocAgent facts")
    allowed = {variant: set(conditions) for _, variant, conditions, _ in BLOCK_CELLS}
    seen, referenced = set(), {}
    for row in rows:
        _check(set(row) == set(DISPATCH_ROW_KEYS), f"dispatch row {row.get('item_id')}: fields {sorted(row)}")
        _check(row["item_id"] not in seen, f"{row['item_id']}: duplicate dispatch row")
        seen.add(row["item_id"])
        _check(row["family"] in stores and row["family"] in manifest["graph_facts"],
               f"{row['item_id']}: no payload store or graph facts for family {row['family']}")
        _check(set(row["target"]) == set(DISPATCH_TARGET_KEYS), f"{row['item_id']}: target fields")
        _check(set(row["membership"]) == set(DISPATCH_MEMBERSHIP_KEYS)
               and all(type(value) is bool for value in row["membership"].values()),
               f"{row['item_id']}: membership fields")
        found = _truth_keys({key: value for key, value in row.items() if key != "cells"})
        _check(not found, f"{row['item_id']}: truth fields {sorted(found)}")
        _check(row["wording_pair"] in wording.PAIRS, f"{row['item_id']}: wording pair")
        for variant, conditions in row["cells"].items():
            _check(variant in allowed and set(conditions) == allowed[variant], f"{row['item_id']}: {variant} cells")
            for sha in conditions.values():
                _check(isinstance(sha, str) and _HEX64.fullmatch(sha) and sha in stores[row["family"]],
                       f"{row['item_id']}: {variant} payload {sha} is not in the payload store")
                referenced.setdefault(row["family"], set()).add(sha)
        _check({"omit_node", "vcomplete"} <= set(row["cells"]), f"{row['item_id']}: P and S-VC cells are required")
        _check(("natural" in row["cells"]) == row["membership"]["in_s_nat"]
               and ("omit_edge" in row["cells"]) == row["membership"]["in_s_edge"],
               f"{row['item_id']}: cells disagree with membership")
    for family, store in stores.items():
        unreferenced = set(store) - referenced.get(family, set())
        _check(not unreferenced, f"{family}: the dispatch payload store holds {len(unreferenced)} payloads that no "
                                 "dispatch row names (D10.7: only the dispatched cells are staged)")
    report = {"rows": len(rows), "families": len(stores), "payload_rows": sum(len(store) for store in stores.values()),
              "unreferenced_payload_rows": 0, "truth_fields": 0, "ok": True}
    if items_dir is not None:
        items, items_manifest, items_bytes, items_manifest_bytes = read_items(items_dir)
        _check(manifest["items_sha256"] == hashlib.sha256(items_bytes).hexdigest(), "items sha256")
        _check(manifest["manifest_sha256"] == hashlib.sha256(items_manifest_bytes).hexdigest(), "items manifest sha256")
        by_id = {item["item_id"]: item for item in items}
        _check([row["item_id"] for row in rows] == [item["item_id"] for item in items], "dispatch rows vs items")
        _check(set(manifest["payload_store"]) == {item["family"] for item in items}, "dispatch families vs items")
        for family, entry in manifest["payload_store"].items():
            outputs = items_manifest["families"][family]["outputs"]
            _check((manifest.get("items_payload_store") or {}).get(family, {}).get("sha256")
                   == outputs["payloads"]["sha256"], f"{family}: items_payload_store does not pin the items payload store")
            items_store = (Path(items_dir) / outputs["payloads"]["path"]).read_bytes()
            _check(hashlib.sha256(items_store).hexdigest() == outputs["payloads"]["sha256"],
                   f"{family}: the items payload store differs from the items manifest")
            subset = dispatched_store(items_store, referenced.get(family, set()))
            _check(entry["sha256"] == hashlib.sha256(subset).hexdigest(),
                   f"{family}: the dispatch payload store is not the dispatched subset of the items payload store")
            _check(manifest["graph_facts"][family]["sha256"] == outputs["graph_facts"]["sha256"],
                   f"{family}: the dispatch graph facts are not the items' graph facts")
        for row in rows:
            item = by_id[row["item_id"]]
            _check(row == dispatch_row(item), f"{row['item_id']}: dispatch row differs from its item")
            omitted = item["omitted"]["identity"]
            _check(not render_module.omitted_identity_exposed(omitted, [row]),
                   f"{row['item_id']}: the omitted identity appears in the dispatch row")
            omit_payloads = [stores[row["family"]][sha] for sha in row["cells"]["omit_node"].values()]
            _check(not render_module.graph_identity_exposed(omitted, omit_payloads),
                   f"{row['item_id']}: the omitted identity appears in a dispatched omit_node graph payload")
        report["items_checked"] = len(rows)
    return report


# ------------------------------------------------------------------ spec from the registry
class SpecError(ValueError):
    pass


def _git_head(repo):
    if not (Path(repo) / ".git").exists():
        return None
    try:
        result = subprocess.run(["git", "-c", f"safe.directory={repo}", "--no-optional-locks", "-C", str(repo),
                                 "rev-parse", "HEAD"], capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.SubprocessError):
        return "error"
    return result.stdout.strip() if result.returncode == 0 else "error"


def build_spec_from_registry(registry_path, *, repos_root, static_root, trace_root=None, trace_subdir=None,
                             roles=ROLES, families=None, allow_missing_trace=False, require_identity_sha256=None):
    """Item build spec for the registry's ``status == "ok"`` families with the given roles.

    Cross-checks each family's registry commit against its repository HEAD (when it has
    ``.git``), both static manifests and the trace manifest, and the package source tree
    across L, J and the trace, and requires the LocAgent document (the D12.7 rule needs
    it). Raises SpecError listing every problem.

    Role ``main`` pins the D12.3 trace input: ``trace_subdir`` must be
    ``MAIN_TRACE_SUBDIR`` (trace-1; trace-2 is the determinism audit), and every main
    family's trace manifest must record this checkout's frozen tracer.py and identity.py
    (``plugin_files``), so the superseded first trace attempt or another root is refused.
    Without an explicit ``require_identity_sha256``, a main spec requires every static
    and trace manifest to name this checkout's identity.py.
    """
    registry_bytes = Path(registry_path).read_bytes()
    registry = json.loads(registry_bytes)
    roles = tuple(roles)
    if not roles or not set(roles) <= set(ROLES):
        raise SpecError(f"roles must be a subset of {ROLES}")
    main = "main" in roles
    if main and (trace_root is None or trace_subdir != MAIN_TRACE_SUBDIR):
        raise SpecError(f"role main needs --trace-root and --trace-subdir {MAIN_TRACE_SUBDIR} (D12.3: trace-1 of the "
                        f"re-run is the trace input; trace-2 is the determinism audit), not {trace_subdir!r}")
    frozen_trace_code = {name: hashlib.sha256((_PROBE_DIR.parents[1] / name).read_bytes()).hexdigest()
                         for name in TRACE_CODE_FILES}
    records = [entry for entry in registry["families"] if entry.get("status") == "ok" and entry.get("role") in roles]
    if families is not None:
        unknown = sorted(set(families) - {entry["family"] for entry in records})
        if unknown:
            raise SpecError(f"families not in the registry with status ok and roles {roles}: {unknown}")
        records = [entry for entry in records if entry["family"] in set(families)]
    roster = sorted(entry["family"] for entry in registry["families"]
                    if entry.get("status") == "ok" and entry.get("role") == "main")
    local_identity = hashlib.sha256((_PROBE_DIR / "identity.py").read_bytes()).hexdigest()
    errors, entries, checks = [], [], {}
    for record in sorted(records, key=lambda entry: entry["family"]):
        family, commit = record["family"], record["commit"]
        dirs = list(normalize_package_dirs(record["package_dirs"]))
        repo = Path(repos_root) / family
        static = Path(static_root) / family
        check = {"commit": commit, "repo_head": None, "static": {}, "trace": None, "source_tree_sha256": {}}
        if not repo.is_dir():
            errors.append(f"{family}: repository {repo} is missing")
        else:
            head = _git_head(repo)
            check["repo_head"] = head
            if head is not None and head != commit:
                errors.append(f"{family}: repository HEAD {head} is not the registry commit {commit}")
        for backend in ("locagent", "jedi"):
            path = static / f"{backend}.manifest.json"
            if not path.is_file() or not (static / f"{backend}.edges.jsonl").is_file():
                errors.append(f"{family}: static {backend} outputs are missing under {static}")
                continue
            manifest = json.loads(path.read_bytes())
            repo_info = manifest.get("repo") or {}
            entry = {"status": manifest.get("status"), "commit": repo_info.get("commit"),
                     "tracked_changes": repo_info.get("tracked_changes"),
                     "identity_py_sha256": (manifest.get("code_sha256") or {}).get("absent_edges/probe/identity.py")}
            check["static"][backend] = entry
            check["source_tree_sha256"][backend] = manifest.get("source_tree_sha256")
            if entry["commit"] != commit:
                errors.append(f"{family}: static {backend} ran at commit {entry['commit']}, registry says {commit}")
            if entry["tracked_changes"]:
                errors.append(f"{family}: static {backend} ran on a checkout with tracked changes")
            if entry["status"] == "failed":
                errors.append(f"{family}: static {backend} failed")
            if list(normalize_package_dirs(manifest.get("package_dirs") or ["?"])) != dirs:
                errors.append(f"{family}: static {backend} package_dirs differ from the registry")
        if not (static / "locagent.document.json").is_file():
            errors.append(f"{family}: the LocAgent document {static / 'locagent.document.json'} is missing "
                          "(the D12.7 locagent_unindexed rule needs it)")
        trace_dir = None
        if trace_root is not None:
            candidate = Path(trace_root) / family
            if trace_subdir:
                candidate = candidate / trace_subdir
            path = candidate / "manifest.json"
            if path.is_file():
                manifest = json.loads(path.read_bytes())
                git = (manifest.get("repo") or {}).get("git") or {}
                plugin_files = manifest.get("plugin_files") or {}
                entry = {"dir": str(candidate), "status": manifest.get("status"), "commit": git.get("head"),
                         "dirty": git.get("dirty"),
                         "identity_py_sha256": plugin_files.get("absent_edges/probe/identity.py"),
                         "tracer_py_sha256": plugin_files.get("absent_edges/probe/tracer.py")}
                check["trace"] = entry
                check["source_tree_sha256"]["trace"] = ((manifest.get("repo") or {}).get("source_tree") or {}).get("sha256")
                if entry["commit"] != commit:
                    errors.append(f"{family}: the trace ran at commit {entry['commit']}, registry says {commit}")
                if entry["dirty"]:
                    errors.append(f"{family}: the trace ran on a dirty checkout")
                if list(normalize_package_dirs((manifest.get("repo") or {}).get("package_dirs") or ["?"])) != dirs:
                    errors.append(f"{family}: the trace package_dirs differ from the registry")
                if record["role"] == "main":
                    for name, expected in frozen_trace_code.items():
                        if plugin_files.get(name) != expected:
                            errors.append(f"{family}: the trace manifest records {name} {plugin_files.get(name)}, "
                                          f"not the frozen {expected} (D12.2: main traces come from the re-run at "
                                          "dd842ab6b)")
                trace_dir = str(candidate)
            elif not allow_missing_trace:
                errors.append(f"{family}: trace manifest {path} is missing")
        trees = {value for value in check["source_tree_sha256"].values() if value is not None}
        if len(trees) > 1:
            errors.append(f"{family}: L, J and the trace ran on different package source trees")
        identities = {value for value in [entry.get("identity_py_sha256") for entry in check["static"].values()]
                      + [(check["trace"] or {}).get("identity_py_sha256")] if value is not None}
        check["identity_py_sha256"] = sorted(identities)
        required_identity = require_identity_sha256
        if required_identity is None and record["role"] == "main":
            required_identity = local_identity
        if required_identity is not None and identities != {required_identity}:
            errors.append(f"{family}: identity.py hashes {sorted(identities)} differ from {required_identity}")
        checks[family] = check
        entry = {"family": family, "role": record["role"], "repo_root": str(repo), "package_dirs": dirs,
                 "commit": commit, "url": record.get("url"), "static_dir": str(static)}
        if trace_dir is not None:
            entry["trace_dir"] = trace_dir
        entries.append(entry)
    if errors:
        raise SpecError("; ".join(errors))
    return {"schema": SPEC_SCHEMA,
            "registry": {"path": str(registry_path), "sha256": hashlib.sha256(registry_bytes).hexdigest(),
                         "schema": registry.get("schema")},
            "roots": {"repos": str(repos_root), "static": str(static_root),
                      "trace": None if trace_root is None else str(trace_root), "trace_subdir": trace_subdir},
            "main_trace_pin": ({"trace_subdir": MAIN_TRACE_SUBDIR, "code_sha256": frozen_trace_code} if main
                               else None),
            "roles": list(roles), "main_roster": roster, "families": entries,
            "families_without_trace": sorted(entry["family"] for entry in entries if "trace_dir" not in entry),
            "identity_py_sha256": {"local": local_identity,
                                   "all_inputs_equal_local": all(set(check["identity_py_sha256"]) <= {local_identity}
                                                                 for check in checks.values())},
            "checks": checks}
