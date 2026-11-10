"""Exp A deterministic scoring, v2 (docs/EXPA_DESIGN_V2_DELTA.md D7, D8, D10.8, D10.11, D10.14, D11.3-D11.5).

Inputs: the frozen items (``expA-item-v2``), their payload store
(``payloads/<family>.jsonl`` rows ``{sha256, bytes, payload}``), the dispatch view
(``expA-dispatch-v2``: ``dispatch.jsonl`` + ``dispatch_manifest.json``), the run's
``run.json`` and its immutable ``expA-episode-result-v2`` attempt records, and the
pinned checkouts (only for ``identity.FunctionIndex``). No model, no GPU, no hidden
tests, gold patches or benchmark data. Stdlib only.

Design (D8 as amended by D11.4)
===============================
Cell id = ``"{block}|{item_id}|{variant}|{condition}|s{seed}"``. Two layouts are known:

``d11`` (default; D11.4, protocols/EXPA_PREREG.md section 8)::

    P       omit_node C1 C2 C3 C4a + none C0   seeds 11 23 47   every item   primary (H1-H3)
    S-VC    vcomplete C1 C3                     seed 11          every item   secondary
    S-NAT   natural   C1 C3 C4a                 seed 11          in_s_nat     secondary
    S-EDGE  omit_edge C1 C3                     seed 11          in_s_edge    secondary
    T       none      T-offered T-recommended   seed 11          in_t         exploratory

``contract-v2`` (the pre-D11 interface contract): the same, except that C0 is not in P
but in its own block ``S-C0`` (none x C0, seed 11). A run is scored under the layout
its own records name (``run.json``, ``cells.jsonl``, cell ids); records that disagree
are refused.

``a3-ledger`` (follow-up A3, protocols/EXPA_A3_LEDGER.md): ``d11`` with one more block
first, which its runs dispatch alone::

    LEDGER  vcomplete C1 + omit_node C1         seeds 61 73 89   every item   A3 (both arms)

Its two cells share condition C1 (they differ by variant), so ``Block.variant_of`` is
never used for it. A run is an A3 run when its run.json design names ``a3-ledger`` or
lists the LEDGER block, or a cell id starts with ``LEDGER|``.

Site rules (D11.5)
==================
A submitted entry is ``{"path", "line"}``. Paths: surrounding whitespace, a leading
``/repo/`` and leading ``./`` are stripped, backslashes become ``/``, empty and ``.``
segments are dropped; comparison is case-sensitive. Lines: an integer >= 1 or a string
of ASCII digits with value >= 1. An entry whose path is not a non-empty string or
whose line is anything else is **invalid**. Every entry gets exactly one category,
in submission order:

1. invalid entry -> ``unresolved`` (reason ``invalid_path`` / ``invalid_line``);
2. a valid entry whose normalized (path, line) was submitted before -> ``duplicate``;
3. ``v_site``: the line lies in ``[lineno, end_lineno]`` of a site span of a V caller;
4. ``u_site``: ... of a U caller;
5. ``lexical_name_match``: (path, line) is one of the item's ``lexical_sites`` (every
   package-scope line naming X's simple name or a frozen alias, the grep_all input);
6. ``unresolved``: everything else, with a reason: ``absolute_path`` (outside the
   checkout), ``parent_reference`` (``..``), ``out_of_scope`` (not a package-scope
   file) or ``not_a_known_site``. Unresolved does not mean false.

Outcomes (D7, D11.5)
====================
* ``site_recovered`` (primary): some valid entry has C*'s site path and a line inside
  ``[lineno, end_lineno]`` of some site in sites(C*, X). It is scored for every block,
  variant and condition; for vcomplete, natural, C0 and T it is the baseline rate for
  the same C*.
* ``function_localized``: ``enclosing_function(path, line)`` of some valid entry is C*.
* ``recall_V_callers`` (D11.5 name; ``recall_V_sites`` is the contract-v2 name of the
  same value): fraction of V callers with at least one recovered site.
* ``precision_lower`` = v_site / all submitted entries (duplicates and invalid ones
  included); ``precision_upper`` = (v_site + u_site + lexical_name_match) / unique
  valid entries. Both are null when their denominator is 0.
* ``certifiably_false_completeness``: ``complete == "yes"`` while some
  runtime-witnessed V caller has no recovered site. ``uncertain_despite_full_V``:
  ``complete != "yes"`` while every V caller has a recovered site (descriptive).
* S-NAT (D11.5): NM = runtime-witnessed V callers absent from the natural answer (the
  natural C1 payload's match sources). ``nm_recall`` = recovered share of NM and
  ``nm_any`` = 1 when any NM caller is recovered. Scored on every row of an item with a
  non-empty NM (a baseline outside S-NAT), null otherwise. C* is not used in S-NAT.
* Exits: Submitted, NoAnswer, ContextLimit, FormatExhausted (completed model outcomes)
  and InfraInterrupted. A completed episode that did not submit scores 0 on every
  recovery outcome and keeps its exit.

Attempts and cell states (D11.3)
================================
Attempts are ordered by ordinal (``<cell_id>.aNNN``). The terminal attempt is the first
with a model exit; every earlier attempt (an InfraInterrupted result, or a start record
without a result) is an infrastructure attempt. ``cell_state``:

* ``completed``: a terminal attempt exists. ``retry_cap_exceeded`` flags one preceded
  by 3 or more infrastructure attempts: the runner allows at most 3, so that attempt is
  a runner deviation, reported rather than hidden;
* ``missing_infrastructure``: no terminal attempt after ``INFRA_RETRY_CAP`` (3)
  infrastructure attempts ("missing (infrastructure)"); a fourth infrastructure attempt
  also sets ``retry_cap_exceeded``;
* ``infra_pending``: 1-2 infrastructure attempts and no terminal attempt;
* ``not_dispatched``: no attempt record. An attempt directory with neither a start
  record nor a result was never dispatched; it is counted in the audit only.

Rows of cells that are not completed carry ``exit_status`` ``InfraInterrupted`` (any
infrastructure attempt) or ``Missing`` and null outcomes: D11.3 builds item means from
completed cells only, and the analysis imputes 0 and 1 explicitly in its bounds.

Reference policies (D10.11, D11.1)
==================================
Model-free submissions scored by the same rules for every item and every variant
(``none`` = no answer block, as in C0 and T). They are references, not ceilings.

* ``answer_only``: the delivered answer's match lines: for each match of the variant's
  C1 answer, in answer order, the first line of every frozen site span of its source
  caller (V or U). Match records carry no line, so the frozen spans place them; this is
  the best case for copying the answer;
* ``c4_hits``: every hit (path, line) of the item's C4 lexical verification block;
* ``answer_plus_c4``: the union of the two (deduplicated, answer first);
* ``grep_all``: the item's ``lexical_sites`` (oracle-assisted: frozen alias knowledge).

Binding (D10.14, validity finding F2)
=====================================
Scoring raises ``BindingError`` when sha256(items.jsonl) differs from ``run.json``, the
dispatch manifest or any result; when the dispatch view's rows differ from the items or
carry truth fields; when a payload store file differs from the dispatch manifest or
run.json; when a result names another dispatch view; when a result or attempt is for a
cell outside the design; when a completed result's delivered or expected payload sha256
differs from its item's cell (C0 and T: no payload) or its target or family differs
from its item; when a cell is dispatched again after its terminal attempt; and when a
result's ``run_context.start_sha256`` is not the sha256 of the attempt_start.json in its
own attempt directory (records of two launches mixed in one attempt, for example after a
relaunch on a wiped run root). With a dispatch view that stages only the dispatched cells
(``items_payload_store`` in its manifest), the items payload store is bound through
``items_payload_store`` and run.json's store through the manifest's ``payload_store``.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re

from ..data import canonical_json
from .identity import file_sha256s, in_package_scope, normalize_relpath, scan_package_files, source_tree_sha256

SCORE_SCHEMA = "expA-score-v2"
POLICY_SCHEMA = "expA-score-v2"
SCORES_MANIFEST_SCHEMA = "expA-scores-manifest-v2"
ITEM_SCHEMA = "expA-item-v2"
RESULT_SCHEMA = "expA-episode-result-v2"
DISPATCH_SCHEMA = "expA-dispatch-v2"
SCORE_PROTOCOL = "expA-score-v2-d11"

VARIANTS = ("vcomplete", "natural", "omit_node", "omit_edge")
NO_VARIANT = "none"
ANSWER_CONDITIONS = ("C1", "C2", "C3", "C4a")
T_CONDITIONS = ("T-offered", "T-recommended")
NO_ANSWER_CONDITIONS = ("C0",) + T_CONDITIONS
CONDITIONS = ("C0",) + ANSWER_CONDITIONS + T_CONDITIONS
P_SEEDS = (11, 23, 47)
SECONDARY_SEED = 11
EXITS = ("Submitted", "NoAnswer", "ContextLimit", "FormatExhausted", "InfraInterrupted")
COMPLETED_EXITS = frozenset(EXITS[:4])
INFRA_EXIT = "InfraInterrupted"
MISSING = "Missing"
ROW_STATUSES = EXITS + (MISSING,)
INFRA_RETRY_CAP = 3
CELL_STATES = ("completed", "missing_infrastructure", "infra_pending", "not_dispatched")
TERMINAL_STATES = frozenset({"completed", "missing_infrastructure"})
CLAIMS = ("yes", "no", "unknown")
POLICIES = ("answer_only", "c4_hits", "answer_plus_c4", "grep_all")
CATEGORIES = ("v_site", "u_site", "lexical_name_match", "unresolved", "duplicate")
UNRESOLVED_REASONS = ("invalid_path", "invalid_line", "absolute_path", "parent_reference", "out_of_scope",
                      "not_a_known_site")
SITE_KINDS = frozenset({"call", "reference", "dynamic"})
WITNESSES = frozenset({"trace", "L", "J", "ast"})
STRATA = ("direct", "alias", "callback", "dynamic")
WORDING_PAIRS = ("A", "B")
SANDBOX_PREFIX = "/repo/"
VERIFICATION_KIND = "lexical_candidates_not_proven_edges"
GRAPH_QUERY = "graph_query"
TRUTH_FIELDS = frozenset({"V", "U", "sites", "omitted", "covariates", "lexical_sites"})
OUTCOME_FIELDS = ("site_recovered", "function_localized", "recall_V_callers", "recall_V_sites", "recovered_V", "n_V",
                  "nm_recall", "nm_any", "n_NM", "n_submitted", "n_unique_sites", "n_invalid", "precision_categories",
                  "unresolved_reasons", "precision_lower", "precision_upper", "complete",
                  "certifiably_false_completeness", "uncertain_despite_full_V")
_DIGITS = re.compile(r"[0-9]+")
_ORDINAL = re.compile(r"\.a([0-9]+)$")


class ScoreInputError(ValueError):
    """An input file does not follow the v2 interface contract."""


class BindingError(RuntimeError):
    """Scoring inputs are not bound to each other (items, dispatch, run, payloads, targets)."""


def _require(condition, message, error=ScoreInputError):
    if not condition:
        raise error(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_hex64(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _is_count(value):
    return type(value) is int and value >= 0


def strict_loads(data):
    """JSON with duplicate keys and non-finite numbers rejected."""
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ScoreInputError(f"duplicate JSON key {key!r}")
            out[key] = value
        return out

    def constant(token):
        raise ScoreInputError(f"non-finite JSON number {token}")
    return json.loads(data, object_pairs_hook=pairs, parse_constant=constant)


def _schema_of(record):
    return record.get("schema", record.get("schema_version")) if isinstance(record, dict) else None


def write_immutable(path, data: bytes) -> str:
    """Create ``path`` exclusively, write, fsync, make it read-only; returns its sha256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags, 0o644)
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(path, 0o444)
    return sha256_bytes(data)


def package_tree_sha256(root, package_dirs) -> tuple:
    """(sha256, file count) of the package-scope tree, as items record it in render.source_sha256."""
    files = scan_package_files(root, list(package_dirs))["files"]
    return source_tree_sha256(file_sha256s(root, files)), len(files)


# --------------------------------------------------------------------------- design (D8, D11.4)

@dataclass(frozen=True)
class Block:
    name: str
    arms: tuple          # ((variant, condition), ...); conditions are unique within a block, except LEDGER (A3)
    seeds: tuple
    membership: str | None
    role: str            # primary | secondary | exploratory

    @property
    def conditions(self):
        return tuple(condition for _, condition in self.arms)

    def variant_of(self, condition):
        for variant, name in self.arms:
            if name == condition:
                return variant
        raise ScoreInputError(f"block {self.name} has no condition {condition!r}")


_P_ANSWER_ARMS = tuple(("omit_node", condition) for condition in ANSWER_CONDITIONS)
_SECONDARY_BLOCKS = (
    Block("S-VC", (("vcomplete", "C1"), ("vcomplete", "C3")), (SECONDARY_SEED,), None, "secondary"),
    Block("S-NAT", (("natural", "C1"), ("natural", "C3"), ("natural", "C4a")), (SECONDARY_SEED,), "in_s_nat",
          "secondary"),
    Block("S-EDGE", (("omit_edge", "C1"), ("omit_edge", "C3")), (SECONDARY_SEED,), "in_s_edge", "secondary"),
    Block("T", ((NO_VARIANT, "T-offered"), (NO_VARIANT, "T-recommended")), (SECONDARY_SEED,), "in_t", "exploratory"),
)
DESIGNS = {
    "d11": (Block("P", _P_ANSWER_ARMS + ((NO_VARIANT, "C0"),), P_SEEDS, None, "primary"),) + _SECONDARY_BLOCKS,
    "contract-v2": (Block("P", _P_ANSWER_ARMS, P_SEEDS, None, "primary"),
                    Block("S-C0", ((NO_VARIANT, "C0"),), (SECONDARY_SEED,), None, "secondary")) + _SECONDARY_BLOCKS,
}
# Follow-up A3 (protocols/EXPA_A3_LEDGER.md; harness.EXTENSION_BLOCK_DESIGN): d11 with the LEDGER block first.
LEDGER_BLOCK = "LEDGER"
LEDGER_SEEDS = (61, 73, 89)
LEDGER_DESIGN = "a3-ledger"
DESIGNS[LEDGER_DESIGN] = (Block(LEDGER_BLOCK, (("vcomplete", "C1"), ("omit_node", "C1")), LEDGER_SEEDS, None, "a3"),
                          ) + DESIGNS["d11"]
DEFAULT_DESIGN = "d11"
ALL_BLOCKS = ("P", "S-C0", "S-VC", "S-NAT", "S-EDGE", "T", LEDGER_BLOCK)


class Design:
    """One frozen layout of blocks; ``cells(items)`` is the finite design of a set of items."""

    def __init__(self, name=DEFAULT_DESIGN):
        _require(name in DESIGNS, f"unknown design {name!r}; known: {sorted(DESIGNS)}")
        self.name = name
        self.blocks = DESIGNS[name]
        self.by_name = {block.name: block for block in self.blocks}

    def block(self, name) -> Block:
        _require(name in self.by_name, f"design {self.name} has no block {name!r}")
        return self.by_name[name]

    def has(self, block, condition=None):
        return block in self.by_name and (condition is None or condition in self.by_name[block].conditions)

    def c0_block(self):
        return "P" if self.has("P", "C0") else "S-C0"

    def includes(self, item, block) -> bool:
        flag = self.block(block).membership
        return True if flag is None else bool(item.membership.get(flag))

    def cells(self, items, blocks=None) -> list:
        """Every cell of the design for these items, in block, item, condition, seed order."""
        out = []
        for block in self.blocks:
            if blocks is not None and block.name not in blocks:
                continue
            for item_id in sorted(items):
                item = items[item_id]
                if not self.includes(item, block.name):
                    continue
                for variant, condition in block.arms:
                    for seed in block.seeds:
                        out.append({"cell_id": cell_id(block.name, item_id, variant, condition, seed),
                                    "block": block.name, "item_id": item_id, "family": item.family,
                                    "variant": variant, "condition": condition, "seed": seed})
        return out

    def describe(self):
        return {"name": self.name, "blocks": [{"block": b.name, "arms": [list(arm) for arm in b.arms],
                                               "seeds": list(b.seeds), "membership": b.membership, "role": b.role}
                                              for b in self.blocks]}


def cell_id(block, item_id, variant, condition, seed) -> str:
    return f"{block}|{item_id}|{variant}|{condition}|s{seed}"


def parse_cell_id(text) -> dict:
    _require(isinstance(text, str) and text.count("|") == 4, f"not a v2 cell id: {text!r}")
    block, item_id, variant, condition, seed = text.split("|")
    _require(block in ALL_BLOCKS and item_id and variant in (NO_VARIANT,) + VARIANTS and condition in CONDITIONS
             and seed.startswith("s") and seed[1:].isdigit() and str(int(seed[1:])) == seed[1:],
             f"not a v2 cell id: {text!r}")
    return {"block": block, "item_id": item_id, "variant": variant, "condition": condition, "seed": int(seed[1:])}


def _lists_containing(value, needle):
    if isinstance(value, list) and needle in value:
        return True
    if isinstance(value, dict):
        return any(_lists_containing(inner, needle) for inner in value.values())
    if isinstance(value, list):
        return any(_lists_containing(inner, needle) for inner in value)
    return False


def detect_design(run_json=None, cell_ids=(), requested=None) -> tuple:
    """(design name, evidence): every record that names a layout must agree."""
    evidence = {}
    record = run_json.get("design") if isinstance(run_json, dict) else None
    if isinstance(record, str):
        evidence["run.json design"] = record
    elif isinstance(record, dict):
        if isinstance(record.get("name"), str):
            evidence["run.json design.name"] = record["name"]
        blocks = record.get("blocks")
        names = set(blocks) if isinstance(blocks, dict) else {b.get("block") for b in blocks or ()
                                                              if isinstance(b, dict)}
        if LEDGER_BLOCK in names:          # A3 (it also holds the d11 blocks, so it is tested first)
            evidence["run.json design.blocks"] = LEDGER_DESIGN
        elif "S-C0" in names:
            evidence["run.json design.blocks"] = "contract-v2"
        elif isinstance(blocks, dict) and _lists_containing(blocks.get("P"), "C0"):
            evidence["run.json design.blocks"] = "d11"
        elif isinstance(blocks, list) and any(isinstance(b, dict) and b.get("block") == "P"
                                              and _lists_containing(b, "C0") for b in blocks):
            evidence["run.json design.blocks"] = "d11"
    ids = list(cell_ids)
    if any(text.startswith(LEDGER_BLOCK + "|") for text in ids):
        evidence["cell ids"] = LEDGER_DESIGN
    elif any(text.startswith("S-C0|") for text in ids):
        evidence["cell ids"] = "contract-v2"
    elif any(text.startswith("P|") and "|C0|" in text for text in ids):
        evidence["cell ids"] = "d11"
    if requested is not None:
        evidence["requested"] = requested
    names = set(evidence.values())
    _require(len(names) <= 1, f"records name different designs: {evidence}", BindingError)
    name = names.pop() if names else DEFAULT_DESIGN
    _require(name in DESIGNS, f"unknown design {name!r} ({evidence})", BindingError)
    return name, evidence


# --------------------------------------------------------------------------- items

@dataclass(frozen=True)
class Span:
    path: str
    lineno: int
    end_lineno: int
    kind: str

    def covers(self, path, line) -> bool:
        return path == self.path and self.lineno <= line <= self.end_lineno


def _spans(record, where):
    sites = record.get("sites")
    _require(isinstance(sites, list), f"{where}: sites must be a list")
    spans = []
    for k, site in enumerate(sites):
        here = f"{where}.sites[{k}]"
        _require(isinstance(site, dict), f"{here} must be an object")
        path = site.get("path")
        try:
            _require(isinstance(path, str) and normalize_relpath(path) == path, f"{here}.path must be normalized")
        except ValueError as error:
            raise ScoreInputError(f"{here}.path: {error}") from error
        lineno, end = site.get("lineno"), site.get("end_lineno")
        _require(type(lineno) is int and lineno >= 1 and type(end) is int and end >= lineno,
                 f"{here}: lineno/end_lineno must be integers with 1 <= lineno <= end_lineno")
        _require(site.get("kind") in SITE_KINDS, f"{here}.kind must be one of {sorted(SITE_KINDS)}")
        witnesses = site.get("witnesses")
        _require(isinstance(witnesses, list) and set(witnesses) <= WITNESSES,
                 f"{here}.witnesses must be a subset of {sorted(WITNESSES)}")
        spans.append(Span(path, lineno, end, site["kind"]))
    return tuple(spans)


class ItemTruth:
    """One validated ``expA-item-v2`` record with the site truth the scorer needs."""

    def __init__(self, record: dict):
        _require(isinstance(record, dict), "item record must be an object")
        self.record = record
        item_id = record.get("item_id")
        _require(isinstance(item_id, str) and item_id and "|" not in item_id, f"item_id must be text without '|': "
                                                                               f"{item_id!r}")
        where = f"item {item_id}"
        _require(_schema_of(record) == ITEM_SCHEMA, f"{where}: schema must be {ITEM_SCHEMA}")
        self.item_id = item_id
        self.family = record.get("family")
        _require(isinstance(self.family, str) and self.family, f"{where}: family is required")
        self.role = record.get("role")
        _require(isinstance(self.role, str) and self.role, f"{where}: role is required")
        dirs = record.get("package_dirs")
        _require(isinstance(dirs, list) and dirs and all(isinstance(d, str) for d in dirs),
                 f"{where}: package_dirs must be a list of paths")
        self.package_dirs = tuple(dirs)
        target = record.get("target")
        _require(isinstance(target, dict) and all(k in target for k in ("identity", "path", "line", "qualname")),
                 f"{where}: target needs identity, path, line and qualname")
        self.target_record = target
        self.target = target["identity"]
        _require(isinstance(self.target, str) and self.target == f"{target['path']}::{target['qualname']}",
                 f"{where}: target identity, path and qualname disagree")
        self.V, self.V_meta = {}, {}
        for k, caller in enumerate(record.get("V") or []):
            _require(isinstance(caller, dict) and isinstance(caller.get("identity"), str),
                     f"{where}: V[{k}] needs an identity")
            identity = caller["identity"]
            _require(identity not in self.V and identity != self.target, f"{where}: V[{k}] repeats {identity} or is X")
            _require(caller.get("runtime_witnessed") in (True, False), f"{where}: V[{k}].runtime_witnessed must be "
                                                                         "a boolean")
            self.V[identity] = _spans(caller, f"{where}.V[{k}]")
            self.V_meta[identity] = {"runtime_witnessed": caller["runtime_witnessed"], "stratum": caller.get("stratum"),
                                     "sources": caller.get("sources")}
        _require(self.V, f"{where}: V is empty")
        self.runtime_witnessed = frozenset(c for c, meta in self.V_meta.items() if meta["runtime_witnessed"])
        self.U = {}
        for k, caller in enumerate(record.get("U") or []):
            _require(isinstance(caller, dict) and isinstance(caller.get("identity"), str),
                     f"{where}: U[{k}] needs an identity")
            _require(caller["identity"] not in self.V and caller["identity"] not in self.U,
                     f"{where}: U[{k}] repeats {caller['identity']} or overlaps V")
            self.U[caller["identity"]] = _spans(caller, f"{where}.U[{k}]")
        omitted = record.get("omitted")
        _require(isinstance(omitted, dict) and isinstance(omitted.get("identity"), str),
                 f"{where}: omitted.identity is required")
        self.omitted = omitted["identity"]
        _require(self.omitted in self.V, f"{where}: the omitted caller must be in V")
        _require(self.omitted in self.runtime_witnessed, f"{where}: the omitted caller must be runtime-witnessed (D3)")
        _require(self.V[self.omitted], f"{where}: the omitted caller has no site span, so it can never be recovered")
        self.omitted_path = split_path(self.omitted)
        lexical = record.get("lexical_sites")
        _require(isinstance(lexical, list), f"{where}: lexical_sites must be a list")
        self.lexical_sites = []
        for k, site in enumerate(lexical):
            _require(isinstance(site, dict) and isinstance(site.get("path"), str) and type(site.get("line")) is int
                     and site["line"] >= 1, f"{where}: lexical_sites[{k}] must be {{path, line}}")
            self.lexical_sites.append((site["path"], site["line"]))
        self.lexical = frozenset(self.lexical_sites)
        self.wording_pair = record.get("wording_pair")
        _require(self.wording_pair in WORDING_PAIRS, f"{where}: wording_pair must be A or B")
        covariates = record.get("covariates")
        _require(isinstance(covariates, dict), f"{where}: covariates are required")
        _require(covariates.get("stratum") in STRATA, f"{where}: covariates.stratum must be one of {STRATA}")
        _require(covariates.get("c4_trigger_fires") in (True, False), f"{where}: covariates.c4_trigger_fires must be "
                                                                         "a boolean")
        self.covariates = covariates
        membership = record.get("membership")
        _require(isinstance(membership, dict) and all(membership.get(k) in (True, False)
                                                      for k in ("in_s_nat", "in_s_edge", "in_t")),
                 f"{where}: membership needs boolean in_s_nat, in_s_edge and in_t")
        self.membership = membership
        variants = record.get("variants")
        _require(isinstance(variants, dict) and set(variants) == set(VARIANTS),
                 f"{where}: variants must be exactly {VARIANTS}")
        for variant, info in variants.items():
            cells = (info or {}).get("cells")
            _require(isinstance(cells, dict) and set(cells) == set(ANSWER_CONDITIONS)
                     and all(v is None or _is_hex64(v) for v in cells.values()),
                     f"{where}: variants.{variant}.cells must map C1, C2, C3, C4a to a sha256 or null")
        self.variants = variants
        render = record.get("render")
        self.source_sha256 = render.get("source_sha256") if isinstance(render, dict) else None
        _require(self.source_sha256 is None or _is_hex64(self.source_sha256),
                 f"{where}: render.source_sha256 must be sha256 hex")
        self._by_path = {}
        for group, callers in (("V", self.V), ("U", self.U)):
            for identity, spans in callers.items():
                for span in spans:
                    self._by_path.setdefault(span.path, []).append((span, group, identity))
        names = record.get("names") if isinstance(record.get("names"), dict) else {}
        extra = names.get("lexical") if isinstance(names.get("lexical"), list) else []
        simple = target.get("name") or target["qualname"].rsplit(".", 1)[-1]
        self.search_names = tuple(sorted({simple, target["qualname"]} | {n for n in extra if isinstance(n, str) and n}))
        self.nm = None          # set by attach_natural_answer (needs the payload store)
        self.natural_answer = None
        self.nm_check = None

    def expected_payload(self, variant, condition):
        """Payload sha256 the cell must deliver (None for C0 and T)."""
        if condition in NO_ANSWER_CONDITIONS:
            _require(variant == NO_VARIANT, f"{self.item_id}: {condition} cells have variant 'none'", BindingError)
            return None
        _require(variant in VARIANTS and condition in ANSWER_CONDITIONS,
                 f"{self.item_id}: no payload cell for {variant}/{condition}", BindingError)
        expected = self.variants[variant]["cells"][condition]
        _require(expected is not None, f"{self.item_id}: the item has no {variant}/{condition} payload", BindingError)
        return expected

    def hits(self, path, line):
        """(V callers, U callers) whose site spans contain (path, line)."""
        found_v, found_u = set(), set()
        for span, group, identity in self._by_path.get(path, ()):
            if span.lineno <= line <= span.end_lineno:
                (found_v if group == "V" else found_u).add(identity)
        return found_v, found_u

    def caller_spans(self, identity):
        return self.V.get(identity, self.U.get(identity))

    def omitted_site_hit(self, path, line) -> bool:
        return any(span.covers(path, line) for span in self.V[self.omitted])


def split_path(identity):
    return identity.split("::", 1)[0]


def load_items(path) -> tuple:
    """({item_id: ItemTruth}, file metadata with the sha256 of the exact bytes)."""
    data = Path(path).read_bytes()
    items = {}
    for lineno, line in enumerate(data.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = ItemTruth(strict_loads(line))
        except ScoreInputError as error:
            raise ScoreInputError(f"{path}:{lineno}: {error}") from error
        _require(item.item_id not in items, f"{path}:{lineno}: duplicate item_id {item.item_id}")
        items[item.item_id] = item
    _require(items, f"{path}: no items")
    return items, {"path": str(path), "sha256": sha256_bytes(data), "bytes": len(data), "items": len(items),
                   "families": len({item.family for item in items.values()}),
                   "roles": dict(sorted(Counter(item.role for item in items.values()).items()))}


# --------------------------------------------------------------------------- payloads

class PayloadStore:
    """``payloads/<family>.jsonl`` rows ``{sha256, bytes, payload}``; every row is verified."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self._families, self.file_sha256 = {}, {}

    def family(self, family):
        if family not in self._families:
            path = self.directory / f"{family}.jsonl"
            _require(path.is_file(), f"payload store has no {path.name}", BindingError)
            raw = path.read_bytes()
            rows = {}
            for line in raw.splitlines():
                if not line.strip():
                    continue
                row = strict_loads(line)
                _require(isinstance(row, dict) and "payload" in row, f"payload row of {family} has no payload",
                         BindingError)
                encoded = canonical_json(row["payload"])
                _require(sha256_bytes(encoded) == row.get("sha256") and row.get("bytes") in (None, len(encoded)),
                         f"payload row of {family} does not match its sha256 or byte count", BindingError)
                rows[row["sha256"]] = row["payload"]
            self._families[family] = rows
            self.file_sha256[family] = sha256_bytes(raw)
        return self._families[family]

    def get(self, family, sha256):
        payload = self.family(family).get(sha256)
        _require(payload is not None, f"payload {sha256} of {family} is not in the store", BindingError)
        return payload


def answer_sources(payload, target) -> list:
    """Match sources of a delivered graph answer whose target is X, in answer order."""
    matches = payload.get("matches") if isinstance(payload, dict) else None
    _require(isinstance(matches, list), "payload has no matches list", BindingError)
    out = []
    for match in matches:
        _require(isinstance(match, dict) and isinstance(match.get("source"), str), "a match has no source",
                 BindingError)
        if match.get("target", target) == target:
            out.append(match["source"])
    return out


def verification_hits(payload) -> list:
    block = payload.get("verification") if isinstance(payload, dict) else None
    _require(isinstance(block, dict) and block.get("kind") == VERIFICATION_KIND and isinstance(block.get("hits"), list),
             f"payload has no {VERIFICATION_KIND} verification block", BindingError)
    hits = []
    for hit in block["hits"]:
        _require(isinstance(hit, dict) and isinstance(hit.get("path"), str) and type(hit.get("line")) is int,
                 "verification hits must carry path and line", BindingError)
        hits.append({"path": hit["path"], "line": hit["line"]})
    return hits


def attach_natural_answer(item: ItemTruth, payloads: PayloadStore):
    """NM (D11.5) from the natural C1 answer: runtime-witnessed V callers absent from it."""
    sha = item.variants["natural"]["cells"]["C1"]
    if sha is None:
        item.natural_answer, item.nm = None, None
        return None
    payload = payloads.get(item.family, sha)
    _require((payload.get("query") or {}).get("symbol", item.target) == item.target,
             f"{item.item_id}: the natural answer is for another symbol", BindingError)
    item.natural_answer = tuple(answer_sources(payload, item.target))
    item.nm = frozenset(item.runtime_witnessed - set(item.natural_answer))
    # Diagnostics only: the covariate is computed from LocAgent's caller list, the payload
    # may be truncated (such items are outside S-NAT), so a difference is reported, not refused.
    natural_miss = item.covariates.get("natural_miss")
    item.nm_check = {"n_NM": len(item.nm), "covariate_natural_miss": natural_miss,
                     "agrees_with_covariate": None if natural_miss not in (True, False) else natural_miss == bool(item.nm),
                     "in_s_nat_without_NM": bool(item.membership.get("in_s_nat")) and not item.nm}
    return item.nm


# --------------------------------------------------------------------------- site scoring

def normalize_site_path(path):
    """(key path, reason): reason None for a package-candidate relative path, else why it cannot be one.

    The key path is what duplicates compare on; it is None only for an invalid path.
    """
    if not isinstance(path, str) or not path.strip() or "\x00" in path:
        return None, "invalid_path"
    text = path.strip().replace("\\", "/")
    if text.startswith(SANDBOX_PREFIX):
        text = text[len(SANDBOX_PREFIX):]
    while text.startswith("./"):
        text = text[2:]
    if text.startswith("/") or re.match(r"[A-Za-z]:", text):
        return text, "absolute_path"
    parts = [part for part in text.split("/") if part not in ("", ".")]
    if not parts:
        return None, "invalid_path"
    if ".." in parts:
        return "/".join(parts), "parent_reference"
    joined = "/".join(parts)
    try:
        return normalize_relpath(joined), None
    except ValueError:
        return None, "invalid_path"


def normalize_site_line(line):
    """An integer >= 1 (a string of ASCII digits is accepted), else None."""
    if type(line) is int:
        return line if line >= 1 else None
    if isinstance(line, str) and _DIGITS.fullmatch(line):
        value = int(line)
        return value if value >= 1 else None
    return None


def categorize_sites(sites, truth: ItemTruth, index) -> list:
    """One detail record per submitted entry: category, reason, hit callers, enclosing function."""
    details, seen = [], set()
    for site in sites:
        raw_path = site.get("path") if isinstance(site, dict) else None
        raw_line = site.get("line") if isinstance(site, dict) else None
        submitted = {"path": raw_path, "line": raw_line} if isinstance(site, dict) else {"raw": repr(site)[:200]}
        detail = {"submitted": submitted, "path": None, "line": None, "valid": False, "category": None,
                  "reason": None, "v_callers": [], "u_callers": [], "enclosing_function": None, "omitted_site": False}
        details.append(detail)
        key_path, path_reason = normalize_site_path(raw_path)
        line = normalize_site_line(raw_line)
        if key_path is None or line is None:
            detail.update(category="unresolved", reason="invalid_path" if key_path is None else "invalid_line")
            continue
        detail.update(path=key_path, line=line, valid=True)
        if (key_path, line) in seen:
            detail.update(category="duplicate", reason="repeated_site")
            continue
        seen.add((key_path, line))
        if path_reason is not None:
            detail.update(category="unresolved", reason=path_reason)
            continue
        if not in_package_scope(key_path, truth.package_dirs):
            detail.update(category="unresolved", reason="out_of_scope")
            continue
        enclosing = index.enclosing_function(key_path, line) if index is not None else None
        v_callers, u_callers = truth.hits(key_path, line)
        detail.update(enclosing_function=enclosing, v_callers=sorted(v_callers), u_callers=sorted(u_callers),
                      omitted_site=truth.omitted_site_hit(key_path, line))
        if v_callers:
            detail["category"] = "v_site"
        elif u_callers:
            detail["category"] = "u_site"
        elif (key_path, line) in truth.lexical:
            detail["category"] = "lexical_name_match"
        else:
            detail.update(category="unresolved", reason="not_a_known_site")
    return details


def _nm_outcomes(truth: ItemTruth, recovered):
    if not truth.nm:
        return {"nm_recall": None, "nm_any": None, "n_NM": 0 if truth.nm is not None else None}
    hit = len(truth.nm & set(recovered))
    return {"nm_recall": hit / len(truth.nm), "nm_any": int(hit > 0), "n_NM": len(truth.nm)}


def score_submission(sites, complete, truth: ItemTruth, index) -> dict:
    """Outcomes of one submission (an episode's submitted sites or a reference policy)."""
    details = categorize_sites(sites, truth, index)
    counts = Counter(d["category"] for d in details)
    categories = {name: counts.get(name, 0) for name in CATEGORIES}
    reasons = Counter(d["reason"] for d in details if d["category"] == "unresolved")
    recovered = sorted({caller for d in details if d["category"] == "v_site" for caller in d["v_callers"]})
    n_submitted = len(details)
    n_invalid = sum(1 for d in details if not d["valid"])
    n_unique = sum(1 for d in details if d["valid"] and d["category"] != "duplicate")
    all_recovered = len(recovered) == len(truth.V)
    witnessed_missed = sorted(truth.runtime_witnessed - set(recovered))
    claim = complete if complete in CLAIMS else None
    recall = len(recovered) / len(truth.V)
    out = {
        "site_recovered": int(any(d["omitted_site"] for d in details)),
        "function_localized": int(any(d["enclosing_function"] == truth.omitted for d in details)),
        "recall_V_callers": recall, "recall_V_sites": recall, "recovered_V": recovered, "n_V": len(truth.V),
        "n_submitted": n_submitted, "n_unique_sites": n_unique, "n_invalid": n_invalid,
        "precision_categories": categories,
        "unresolved_reasons": {name: reasons.get(name, 0) for name in UNRESOLVED_REASONS},
        "precision_lower": (categories["v_site"] / n_submitted) if n_submitted else None,
        "precision_upper": ((categories["v_site"] + categories["u_site"] + categories["lexical_name_match"]) / n_unique)
        if n_unique else None,
        "complete": claim,
        "certifiably_false_completeness": None if claim is None else (claim == "yes" and bool(witnessed_missed)),
        "uncertain_despite_full_V": None if claim is None else (claim != "yes" and all_recovered),
        "runtime_witnessed_missed": witnessed_missed,
        "site_details": details,
    }
    out.update(_nm_outcomes(truth, recovered))
    return out


def _empty_outcomes(truth: ItemTruth, *, completed: bool) -> dict:
    """A completed non-submitted episode scores 0; a cell without a terminal attempt has null outcomes."""
    zero = 0 if completed else None
    nm = _nm_outcomes(truth, [])
    if not completed and nm["nm_recall"] is not None:
        nm.update(nm_recall=None, nm_any=None)
    return {"site_recovered": zero, "function_localized": zero,
            "recall_V_callers": 0.0 if completed else None, "recall_V_sites": 0.0 if completed else None,
            "recovered_V": [], "n_V": len(truth.V), "n_submitted": zero, "n_unique_sites": zero, "n_invalid": zero,
            "precision_categories": {name: 0 for name in CATEGORIES} if completed else None,
            "unresolved_reasons": {name: 0 for name in UNRESOLVED_REASONS} if completed else None,
            "precision_lower": None, "precision_upper": None, "complete": None,
            "certifiably_false_completeness": None, "uncertain_despite_full_V": None,
            "runtime_witnessed_missed": None, "site_details": [], **nm}


# --------------------------------------------------------------------------- results

def parse_result(record, *, where="result") -> dict:
    """Validate an ``expA-episode-result-v2`` record; returns it unchanged."""
    _require(isinstance(record, dict), f"{where}: result must be an object")
    _require(_schema_of(record) == RESULT_SCHEMA, f"{where}: schema must be {RESULT_SCHEMA}, got {_schema_of(record)!r}")
    for name in ("cell_id", "attempt_id"):
        _require(isinstance(record.get(name), str) and record[name], f"{where}: {name} is required")
    status = record.get("exit_status")
    _require(status in EXITS, f"{where}: exit_status must be one of {EXITS}")
    parsed = parse_cell_id(record["cell_id"])
    for name in ("block", "item_id", "variant", "condition", "seed"):
        if name in record or status != INFRA_EXIT:
            _require(record.get(name) == parsed[name], f"{where}: {name} {record.get(name)!r} disagrees with the "
                                                       "cell id")
    submitted = record.get("submitted")
    if status == "Submitted":
        _require(isinstance(submitted, dict) and isinstance(submitted.get("sites"), list)
                 and submitted.get("complete") in CLAIMS, f"{where}: a Submitted result needs submitted{{sites, "
                                                          "complete}")
    else:
        _require(submitted is None, f"{where}: only a Submitted result carries a submission")
    if status in COMPLETED_EXITS:
        _require(isinstance(record.get("exposure"), dict), f"{where}: a completed result needs its exposure receipt")
        _require(isinstance(record.get("family"), str), f"{where}: family is required")
    return record


def _graph_calls(result):
    graph = result.get("graph_query_calls")
    if _is_count(graph):
        return {"count": graph, "statuses": None, "nonempty_answers": None}
    if isinstance(graph, list):
        statuses = Counter(str(call.get("status")) for call in graph if isinstance(call, dict))
        nonempty = sum(1 for call in graph if isinstance(call, dict) and isinstance(call.get("returned_count"), int)
                       and call["returned_count"] > 0)
        return {"count": len(graph), "statuses": dict(sorted(statuses.items())), "nonempty_answers": nonempty}
    return {"count": None, "statuses": None, "nonempty_answers": None}


def _cost(result) -> dict:
    tokens = result.get("tokens") if isinstance(result.get("tokens"), dict) else {}
    tool_calls = result.get("tool_calls") if isinstance(result.get("tool_calls"), dict) else {}
    accounting = result.get("accounting") if isinstance(result.get("accounting"), dict) else {}
    graph = _graph_calls(result)
    requests_in_errors = accounting.get("graph_query_requests_in_format_errors")
    return {"calls": result.get("calls"), "prompt_tokens": tokens.get("prompt"),
            "completion_tokens": tokens.get("completion"),
            "finish_reason_length_count": result.get("finish_reason_length_count"),
            "tool_calls": {k: v for k, v in sorted(tool_calls.items()) if _is_count(v)},
            "graph_query_calls": graph["count"], "graph_query_statuses": graph["statuses"],
            "graph_query_nonempty_answers": graph["nonempty_answers"],
            "graph_query_requests_in_format_errors": requests_in_errors if _is_count(requests_in_errors) else None}


EXPOSURE_FIELDS = ("payload_present", "payload_sha256", "expected_payload_sha256", "payload_tokens", "payload_bytes",
                   "verification_block_delivered", "verification_sha256", "verification_hits", "guidance_present",
                   "tools_offered", "recommendation_present", "recorded_before_first_model_call")


def _exposure(result) -> dict | None:
    exposure = result.get("exposure")
    if not isinstance(exposure, dict):
        return None
    return {name: exposure.get(name) for name in EXPOSURE_FIELDS}


def _target_identity(value):
    if isinstance(value, dict):
        return value.get("identity")
    return value


def check_result_binding(result, item: ItemTruth, *, items_sha256, dispatch_hashes=None, dispatch_row=None,
                         payloads: PayloadStore | None = None) -> dict:
    """F2 checks for one attempt record. Raises BindingError; returns what was verified."""
    where = f"{result['cell_id']} attempt {result['attempt_id']}"
    parsed = parse_cell_id(result["cell_id"])
    completed = result["exit_status"] in COMPLETED_EXITS
    checks = {"items_sha256": False, "dispatch_sha256": False, "payload_sha256": False, "target": None,
              "payload_query_symbol": False}
    recorded_items = result.get("items_sha256")
    if completed or recorded_items is not None:
        _require(recorded_items == items_sha256, f"{where}: result items_sha256 {recorded_items!r} is not the bound "
                                                 f"items file {items_sha256}", BindingError)
        checks["items_sha256"] = True
    recorded_dispatch = result.get("dispatch_sha256")
    if dispatch_hashes is not None and (completed or recorded_dispatch is not None):
        _require(recorded_dispatch in dispatch_hashes, f"{where}: result dispatch_sha256 {recorded_dispatch!r} is not "
                                                       "the bound dispatch view", BindingError)
        checks["dispatch_sha256"] = True
    _require(result.get("family", item.family) == item.family, f"{where}: family differs from its item", BindingError)
    expected = item.expected_payload(parsed["variant"], parsed["condition"])
    exposure = result.get("exposure")
    if isinstance(exposure, dict):
        delivered, recorded_expected = exposure.get("payload_sha256"), exposure.get("expected_payload_sha256")
        if expected is None:
            _require(delivered is None and recorded_expected is None and exposure.get("payload_present") in (False, None),
                     f"{where}: a {parsed['condition']} cell must not deliver an answer payload", BindingError)
        else:
            _require(delivered == expected and recorded_expected == expected and exposure.get("payload_present") is True,
                     f"{where}: delivered payload {delivered} / expected {recorded_expected} differ from the item's "
                     f"{parsed['variant']}/{parsed['condition']} payload {expected}", BindingError)
        checks["payload_sha256"] = True
    else:
        _require(not completed, f"{where}: a completed result needs its exposure receipt", BindingError)
    sources = []
    if "target" in result:
        sources.append(("result", _target_identity(result["target"])))
    cell = result.get("cell") if isinstance(result.get("cell"), dict) else None
    if cell is not None and "target" in cell:
        sources.append(("result.cell", _target_identity(cell["target"])))
    if dispatch_row is not None:
        sources.append(("dispatch", _target_identity(dispatch_row.get("target"))))
    for source, identity in sources:
        _require(identity == item.target, f"{where}: {source} target {identity!r} differs from item target "
                                          f"{item.target}", BindingError)
    if expected is not None and payloads is not None:
        payload = payloads.get(item.family, expected)
        symbol = (payload.get("query") or {}).get("symbol") if isinstance(payload, dict) else None
        _require(symbol == item.target, f"{where}: payload query symbol {symbol!r} is not the item target",
                 BindingError)
        checks["payload_query_symbol"] = True
        sources.append(("payload", symbol))
    if completed:
        _require(sources, f"{where}: no record binds this cell's target (result, dispatch row or payload)",
                 BindingError)
    checks["target"] = sorted({source for source, _ in sources}) or None
    return checks


# --------------------------------------------------------------------------- run directory

def _ordinal(attempt_id, start=None):
    match = _ORDINAL.search(attempt_id)
    if match:
        return int(match.group(1))
    if isinstance(start, dict) and _is_count(start.get("ordinal")):
        return start["ordinal"]
    return None


def scan_run(run_dir, empty=None) -> dict:
    """{cell_id: [attempt records in ordinal order]} from every attempt directory of the run.

    Attempt directories with neither a start record nor a result (never dispatched) are
    skipped and appended to ``empty`` when a list is given.

    Layout ``cells/<cell_id>/<attempt_id>/`` (attempt_start.json, result.json,
    attempt_closed.json, events.jsonl); other layouts are found by walking the run
    directory for result.json files (no symlinks followed).
    """
    root = Path(run_dir)
    _require(root.is_dir(), f"run directory not found: {root}")
    found = {}
    directories = []
    cells_root = root / "cells"
    if cells_root.is_dir():
        for cell_dir in sorted(p for p in cells_root.iterdir() if p.is_dir() and not p.is_symlink()):
            for attempt_dir in sorted(p for p in cell_dir.iterdir() if p.is_dir() and not p.is_symlink()):
                directories.append((attempt_dir, cell_dir.name))
    else:
        for current, dirnames, filenames in os.walk(root, followlinks=False):
            dirnames[:] = sorted(name for name in dirnames if not name.startswith("."))
            if "result.json" in filenames or "attempt_start.json" in filenames:
                directories.append((Path(current), None))
    empty = [] if empty is None else empty
    for directory, cell_name in directories:
        rel = directory.relative_to(root).as_posix()
        if not (directory / "attempt_start.json").is_file() and not (directory / "result.json").is_file():
            empty.append(rel)       # created, then crashed before the start record: never dispatched
            continue
        start, start_sha = None, None
        if (directory / "attempt_start.json").is_file():
            start_raw = (directory / "attempt_start.json").read_bytes()
            start, start_sha = strict_loads(start_raw), sha256_bytes(start_raw)
        result, result_sha = None, None
        if (directory / "result.json").is_file():
            raw = (directory / "result.json").read_bytes()
            result = parse_result(strict_loads(raw), where=f"{rel}/result.json")
            result_sha = sha256_bytes(raw)
            # The runner passes sha256(attempt_start.json) to the harness (result.run_context.start_sha256): a
            # result next to another launch's start record (a relaunch on a wiped run root reusing attempt ids)
            # is refused, so the interrupted dispatch can never disappear behind a later result (review finding).
            context = result.get("run_context")
            recorded = context.get("start_sha256") if isinstance(context, dict) else None
            if recorded is not None:
                _require(recorded == start_sha, f"{rel}/result.json was written for attempt_start.json {recorded}, "
                                                f"but the attempt directory holds {start_sha}: records of two launches "
                                                "are mixed in one attempt", BindingError)
        cell = result["cell_id"] if result else (start.get("cell_id") if isinstance(start, dict) else None) or cell_name
        _require(isinstance(cell, str), f"{rel} does not name its cell")
        attempt_id = result["attempt_id"] if result else directory.name
        if cell_name is not None:
            _require(cell == cell_name and attempt_id == directory.name,
                     f"{rel} holds cell {cell} attempt {attempt_id}", BindingError)
        if isinstance(start, dict) and start.get("cell_id") is not None:
            _require(start["cell_id"] == cell, f"{rel}/attempt_start.json names cell {start['cell_id']}", BindingError)
        parse_cell_id(cell)
        found.setdefault(cell, []).append({
            "attempt_id": attempt_id, "ordinal": _ordinal(attempt_id, start), "path": rel, "has_start": start is not None,
            "start_sha256": start_sha,
            "closed": (directory / "attempt_closed.json").is_file(), "result": result, "result_sha256": result_sha,
            "events_path": str(directory / "events.jsonl") if (directory / "events.jsonl").is_file() else None})
    for cell, attempts in found.items():
        attempts.sort(key=lambda a: (a["ordinal"] if a["ordinal"] is not None else 1 << 30, a["attempt_id"]))
        ids = [a["attempt_id"] for a in attempts]
        _require(len(ids) == len(set(ids)), f"{cell}: attempt ids repeat: {ids}", BindingError)
    return found


def resolve_cell(attempts) -> dict:
    """D11.3 terminal attempt and cell state for one cell's attempts (ordinal order)."""
    infra = []
    for position, attempt in enumerate(attempts):
        result = attempt["result"]
        if result is not None and result["exit_status"] in COMPLETED_EXITS:
            later = attempts[position + 1:]
            _require(not later, f"{result['cell_id']}: {len(later)} attempt(s) after the terminal attempt "
                                f"{attempt['attempt_id']}; a completed cell is never dispatched again", BindingError)
            return {"state": "completed", "terminal": attempt, "infra_attempts": len(infra),
                    "retry_cap_exceeded": len(infra) >= INFRA_RETRY_CAP}
        infra.append(attempt)
    if not attempts:
        return {"state": "not_dispatched", "terminal": None, "infra_attempts": 0, "retry_cap_exceeded": False}
    state = "missing_infrastructure" if len(infra) >= INFRA_RETRY_CAP else "infra_pending"
    return {"state": state, "terminal": None, "infra_attempts": len(infra),
            "retry_cap_exceeded": len(infra) > INFRA_RETRY_CAP}


def recorded_hashes(value, key) -> set:
    """Every string stored under ``key`` at any depth of a JSON document."""
    out = set()
    if isinstance(value, dict):
        for name, inner in value.items():
            if name == key and isinstance(inner, str):
                out.add(inner)
            else:
                out |= recorded_hashes(inner, key)
    elif isinstance(value, list):
        for inner in value:
            out |= recorded_hashes(inner, key)
    return out


def run_items_sha256(run_json) -> str:
    """The items sha256 a run.json records (``items_sha256`` at any depth or ``items.sha256``)."""
    values = recorded_hashes(run_json, "items_sha256")
    items = run_json.get("items") if isinstance(run_json, dict) else None
    if isinstance(items, dict) and isinstance(items.get("sha256"), str):
        values.add(items["sha256"])
    _require(values, "run.json records no items sha256", BindingError)
    _require(len(values) == 1, f"run.json records several items sha256 values: {sorted(values)}", BindingError)
    return values.pop()


def load_dispatch(dispatch_path, manifest_path) -> dict:
    """The dispatch view: rows by item_id, the manifest and the hashes a result may name."""
    rows_raw, manifest_raw = Path(dispatch_path).read_bytes(), Path(manifest_path).read_bytes()
    manifest = strict_loads(manifest_raw)
    _require(isinstance(manifest, dict), "dispatch manifest must be an object")
    rows = {}
    for lineno, line in enumerate(rows_raw.decode("utf-8").splitlines(), 1):
        if line.strip():
            row = strict_loads(line)
            _require(isinstance(row, dict) and isinstance(row.get("item_id"), str), f"dispatch row {lineno} needs "
                                                                                    "item_id")
            _require(row["item_id"] not in rows, f"dispatch row {lineno} repeats {row['item_id']}")
            forbidden = TRUTH_FIELDS & set(row)
            _require(not forbidden, f"dispatch row {lineno} carries truth fields {sorted(forbidden)}", BindingError)
            rows[row["item_id"]] = row
    for key in ("dispatch_sha256", "dispatch_jsonl_sha256"):
        if isinstance(manifest.get(key), str):
            _require(manifest[key] == sha256_bytes(rows_raw), f"dispatch manifest {key} is not dispatch.jsonl's "
                                                              "sha256", BindingError)
    return {"rows": rows, "manifest": manifest, "hashes": {sha256_bytes(rows_raw), sha256_bytes(manifest_raw)},
            "dispatch_sha256": sha256_bytes(rows_raw), "manifest_sha256": sha256_bytes(manifest_raw),
            "paths": [str(dispatch_path), str(manifest_path)]}


def _family_hashes(record):
    if not isinstance(record, dict):
        return {}
    out = {}
    for family, value in record.items():
        value = value.get("sha256") if isinstance(value, dict) else value
        if isinstance(value, str):
            out[family] = value
    return out


def check_inputs_binding(items: dict, items_meta: dict, *, run_json=None, dispatch=None, payloads=None) -> dict:
    """Refuse an items file that is not the one run.json and the dispatch view were built from."""
    sha = items_meta["sha256"]
    audit = {"items_sha256": sha}
    if run_json is not None:
        recorded = run_items_sha256(run_json)
        _require(recorded == sha, f"items file sha256 {sha} differs from run.json's {recorded}", BindingError)
        audit["run_json_items_sha256"] = recorded
        recorded_dispatch = recorded_hashes(run_json, "dispatch_sha256")
        if dispatch is not None and recorded_dispatch:
            _require(recorded_dispatch <= dispatch["hashes"], f"run.json dispatch_sha256 {sorted(recorded_dispatch)} "
                                                             "is not the given dispatch view", BindingError)
            audit["run_json_dispatch_sha256"] = sorted(recorded_dispatch)
    if dispatch is not None:
        manifest = dispatch["manifest"]
        _require(_schema_of(manifest) in (DISPATCH_SCHEMA, None), f"dispatch manifest schema is "
                                                                  f"{_schema_of(manifest)!r}")
        _require(manifest.get("items_sha256") == sha, f"items file sha256 {sha} differs from the dispatch manifest's "
                                                      f"{manifest.get('items_sha256')!r}", BindingError)
        audit["dispatch_manifest_items_sha256"] = manifest["items_sha256"]
        rows = dispatch["rows"]
        _require(set(rows) == set(items), f"dispatch view items differ from the items file: "
                                          f"{len(set(rows) ^ set(items))} ids differ", BindingError)
        for item_id, row in rows.items():
            item = items[item_id]
            target = row.get("target")
            same_target = (_target_identity(target) == item.target and (
                not isinstance(target, dict) or all(target.get(k, item.target_record[k]) == item.target_record[k]
                                                    for k in ("path", "line", "qualname"))))
            _require(same_target and row.get("family") == item.family,
                     f"dispatch row {item_id}: target or family differs from the item", BindingError)
            for key, value in (("wording_pair", item.wording_pair), ("package_dirs", list(item.package_dirs))):
                if key in row:
                    _require(row[key] == value, f"dispatch row {item_id}: {key} differs from the item", BindingError)
            if isinstance(row.get("membership"), dict):
                for key in ("in_s_nat", "in_s_edge", "in_t"):
                    _require(row["membership"].get(key) == item.membership[key],
                             f"dispatch row {item_id}: membership {key} differs from the item", BindingError)
            for variant, conditions in (row.get("cells") or {}).items():
                for condition, value in (conditions or {}).items():
                    _require(value == ((item.variants.get(variant) or {}).get("cells") or {}).get(condition),
                             f"dispatch row {item_id}: {variant}/{condition} payload differs from the item",
                             BindingError)
        if payloads is not None:
            dispatched = _family_hashes(manifest.get("payload_store") or manifest.get("payload_store_sha256"))
            full = _family_hashes(manifest.get("items_payload_store"))
            run_recorded = _family_hashes(run_json.get("payload_store")) if run_json is not None else {}
            for family in sorted({item.family for item in items.values()}):
                payloads.family(family)
                if full:
                    # The dispatch view stages only the dispatched cells (D10.7); scoring reads the full items
                    # store, which the dispatch manifest pins as items_payload_store. run.json binds the staged one.
                    _require(full.get(family) == payloads.file_sha256[family], f"payload store {family}.jsonl "
                             "sha256 differs from the dispatch manifest's items_payload_store", BindingError)
                    if run_recorded:
                        _require(run_recorded.get(family) == dispatched.get(family), f"run.json payload store of "
                                 f"{family} is not the dispatch manifest's dispatched store", BindingError)
                else:  # a dispatch view whose store is a full copy of the items store
                    for label, recorded in (("dispatch manifest", dispatched), ("run.json", run_recorded)):
                        if recorded:
                            _require(recorded.get(family) == payloads.file_sha256[family],
                                     f"payload store {family}.jsonl sha256 differs from the {label}", BindingError)
            audit["payload_store_sha256"] = dict(sorted(payloads.file_sha256.items()))
            if full:
                audit["dispatched_payload_store_sha256"] = dict(sorted(dispatched.items()))
    return audit


def check_frozen_cells(run_dir, run_json, design_cells) -> dict:
    """When the run froze its design (cells.jsonl), it must be exactly this design."""
    path = Path(run_dir) / "cells.jsonl"
    if not path.is_file():
        return {"cells_jsonl": "absent"}
    raw = path.read_bytes()
    recorded = run_json.get("cells_sha256") if isinstance(run_json, dict) else None
    if isinstance(recorded, str):
        _require(recorded == sha256_bytes(raw), "cells.jsonl does not match run.json cells_sha256", BindingError)
    frozen = set()
    for line in raw.decode("utf-8").splitlines():
        if line.strip():
            row = strict_loads(line)
            _require(isinstance(row, dict) and isinstance(row.get("cell_id"), str), "cells.jsonl row needs cell_id",
                     BindingError)
            frozen.add(row["cell_id"])
    expected = {cell["cell_id"] for cell in design_cells}
    _require(frozen == expected, f"the run froze another design: {len(frozen - expected)} cells outside this design, "
                                 f"{len(expected - frozen)} design cells not frozen", BindingError)
    return {"cells_jsonl": "matches_design", "cells_jsonl_sha256": sha256_bytes(raw), "frozen_cells": len(frozen)}


# --------------------------------------------------------------------------- telemetry (D11.9, descriptive)

def _word_pattern(names):
    names = [name for name in names if name]
    if not names:
        return None
    return re.compile(r"(?<![A-Za-z0-9_])(?:" + "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
                      + r")(?![A-Za-z0-9_])")


def episode_telemetry(events_path, item: ItemTruth, expected_sha256=None) -> dict | None:
    """Process descriptives from events.jsonl: never conditioned on, never mediators."""
    if events_path is None:
        return None
    raw = Path(events_path).read_bytes()
    if expected_sha256 is not None:
        _require(sha256_bytes(raw) == expected_sha256, f"{events_path}: events.jsonl sha256 differs from result.json",
                 BindingError)
    pattern = _word_pattern(item.search_names)
    omitted_path = item.omitted_path
    searched = read_file = False
    before_submit, submitted = 0, False
    actions = Counter()
    for line in raw.decode("utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        kind, value = event.get("kind"), event.get("value")
        if kind == "submission_accepted":
            submitted = True
        if kind != "tool_action" or not isinstance(value, dict):
            continue
        name, arguments = value.get("name"), value.get("arguments")
        actions[str(name)] += 1
        if name != "submit_answer" and not submitted:
            before_submit += 1
        text = None
        if name == "bash" and isinstance(arguments, dict):
            text = arguments.get("command")
        elif name == "source_reference_search" and isinstance(arguments, dict):
            text = arguments.get("query")
        if isinstance(text, str):
            if pattern is not None and pattern.search(text):
                searched = True
            if name == "bash" and omitted_path in text:
                read_file = True
    return {"searched_target_name": searched, "mentioned_omitted_caller_file": read_file,
            "tool_calls_before_submit": before_submit, "tool_actions": dict(sorted(actions.items())),
            "events_sha256": sha256_bytes(raw)}


# --------------------------------------------------------------------------- rows

SUBMIT_REVIEW_FIELDS = ("mode", "reviewed", "review_call_index", "review_skipped_last_call", "first_call_index",
                        "answer_source", "resubmit_call_index", "unresolved_exit", "ledger_candidates",
                        "ledger_entries", "ledger_body_bytes")
FIRST_SUBMISSION_FIELDS = ("site_recovered", "function_localized", "recall_V_sites", "n_submitted", "n_unique_sites",
                           "n_invalid", "precision_lower", "precision_upper", "complete")


def submit_review_outcomes(result, item: ItemTruth, index) -> dict | None:
    """Follow-up A5 (protocols/EXPA_A5_REVIEW.md): the review receipt of a result and the outcomes of its FIRST
    submission; None for every result without the receipt (every earlier run). The row's own outcomes are those of
    result.json's ``submitted``, the answer used (the resubmission, or the first submission when none followed)."""
    accounting = result.get("accounting") if isinstance(result.get("accounting"), dict) else {}
    receipt = accounting.get("submit_review")
    if not isinstance(receipt, dict):
        return None
    out = {name: receipt.get(name) for name in SUBMIT_REVIEW_FIELDS}
    first = receipt.get("first_submission")
    if isinstance(first, dict) and isinstance(first.get("sites"), list):
        scored = score_submission(first["sites"], first.get("complete"), item, index)
        out["first_submission"] = {name: scored[name] for name in FIRST_SUBMISSION_FIELDS}
    else:
        out["first_submission"] = None
    return out


def _row_base(cell, item: ItemTruth, items_sha256, design_name):
    return {"schema": SCORE_SCHEMA, "kind": "episode", "design": design_name, "cell_id": cell["cell_id"],
            "block": cell["block"], "item_id": cell["item_id"], "family": item.family, "variant": cell["variant"],
            "condition": cell["condition"], "seed": cell["seed"], "wording_pair": item.wording_pair,
            "omitted": item.omitted, "items_sha256": items_sha256}


def score_cell(cell, item: ItemTruth, attempts, index_for_family, *, items_sha256, design_name, dispatch_hashes=None,
               dispatch_row=None, payloads=None, telemetry=True) -> dict:
    """One row per design cell (D11.3 terminal attempt, cell state, outcomes)."""
    row = _row_base(cell, item, items_sha256, design_name)
    attempts = attempts or []
    audits = {}
    for attempt in attempts:
        if attempt["result"] is not None:
            audits[attempt["attempt_id"]] = check_result_binding(
                attempt["result"], item, items_sha256=items_sha256, dispatch_hashes=dispatch_hashes,
                dispatch_row=dispatch_row, payloads=payloads)
    resolved = resolve_cell(attempts)
    row["attempts"] = [{"attempt_id": a["attempt_id"], "ordinal": a["ordinal"],
                        "exit_status": a["result"]["exit_status"] if a["result"] else None,
                        "closed": a["closed"], "result_sha256": a["result_sha256"]} for a in attempts]
    row.update(cell_state=resolved["state"], infra_attempts=resolved["infra_attempts"],
               retry_cap_exceeded=resolved["retry_cap_exceeded"])
    terminal = resolved["terminal"]
    if terminal is None:
        row.update(attempt_id=None, completed=False, exit_status=INFRA_EXIT if attempts else MISSING,
                   result_sha256=None, result_path=None, cost=None, exposure=None, binding=None, endpoint=None,
                   gpu_uuid=None, telemetry=None, **_empty_outcomes(item, completed=False))
        return row
    result = terminal["result"]
    row.update(attempt_id=result["attempt_id"], completed=True, exit_status=result["exit_status"],
               result_sha256=terminal["result_sha256"], result_path=terminal["path"] + "/result.json",
               cost=_cost(result), exposure=_exposure(result), binding=audits[terminal["attempt_id"]],
               endpoint=result.get("endpoint"), gpu_uuid=result.get("gpu_uuid"))
    expected_events = (result.get("events") or {}).get("sha256") if isinstance(result.get("events"), dict) else None
    row["telemetry"] = episode_telemetry(terminal["events_path"], item, expected_events) if telemetry else None
    if result["exit_status"] == "Submitted":
        submitted = result["submitted"]
        row.update(score_submission(submitted["sites"], submitted["complete"], item, index_for_family(item.family)))
    else:
        row.update(_empty_outcomes(item, completed=True))
    if isinstance(result.get("accounting"), dict) and "submit_review" in result["accounting"]:
        # A5 runs only (protocols/EXPA_A5_REVIEW.md): every other row is unchanged
        row["submit_review"] = submit_review_outcomes(result, item, index_for_family(item.family))
    return row


def score_run(run_dir, items: dict, items_meta: dict, index_for_family, *, dispatch, payloads, design=None,
              telemetry=True) -> tuple:
    """Rows for every design cell plus the binding audit. Raises BindingError."""
    run_path = Path(run_dir) / "run.json"
    _require(run_path.is_file(), f"{run_path} not found; a run must bind its inputs in run.json", BindingError)
    run_raw = run_path.read_bytes()
    run_json = strict_loads(run_raw)
    audit = check_inputs_binding(items, items_meta, run_json=run_json, dispatch=dispatch, payloads=payloads)
    for item in items.values():
        attach_natural_answer(item, payloads)
    empty_attempts = []
    scanned = scan_run(run_dir, empty_attempts)
    frozen_ids = []
    if (Path(run_dir) / "cells.jsonl").is_file():
        frozen_ids = [strict_loads(line)["cell_id"] for line in (Path(run_dir) / "cells.jsonl").read_bytes()
                      .decode("utf-8").splitlines() if line.strip()]
    name, evidence = detect_design(run_json, list(scanned) + frozen_ids, design)
    layout = Design(name)
    cells = layout.cells(items)
    audit.update(check_frozen_cells(run_dir, run_json, cells))
    by_id = {cell["cell_id"]: cell for cell in cells}
    outside = sorted(set(scanned) - set(by_id))
    _require(not outside, f"{len(outside)} attempt records are for cells outside the frozen {name} design of these "
                          f"items, e.g. {outside[:3]}", BindingError)
    rows = []
    for cell in cells:
        item = items[cell["item_id"]]
        rows.append(score_cell(cell, item, scanned.get(cell["cell_id"]), index_for_family,
                               items_sha256=items_meta["sha256"], design_name=name,
                               dispatch_hashes=dispatch["hashes"] if dispatch else None,
                               dispatch_row=dispatch["rows"].get(cell["item_id"]) if dispatch else None,
                               payloads=payloads, telemetry=telemetry))
    audit.update(run_json_sha256=sha256_bytes(run_raw), design=name, design_evidence=evidence,
                 design_cells=len(cells), cells_with_records=len(scanned),
                 results=sum(1 for attempts in scanned.values() for a in attempts if a["result"] is not None),
                 start_only_attempts=sum(1 for attempts in scanned.values() for a in attempts if a["result"] is None),
                 cell_states=dict(sorted(Counter(row["cell_state"] for row in rows).items())),
                 status_counts=dict(sorted(Counter(row["exit_status"] for row in rows).items())),
                 retry_cap_exceeded=sum(row["retry_cap_exceeded"] for row in rows),
                 empty_attempt_directories=len(empty_attempts),
                 natural_answer_checks={"items_with_NM": sum(1 for i in items.values() if i.nm),
                                        "covariate_disagreements": sorted(
                                            i.item_id for i in items.values()
                                            if i.nm_check and i.nm_check["agrees_with_covariate"] is False),
                                        "in_s_nat_without_NM": sorted(
                                            i.item_id for i in items.values()
                                            if i.nm_check and i.nm_check["in_s_nat_without_NM"])})
    return rows, audit


# --------------------------------------------------------------------------- reference policies

def item_verification_hits(item: ItemTruth, payloads: PayloadStore) -> list:
    """The item's C4 lexical hits; every C4a payload of the item must carry the same block."""
    blocks, hits = {}, None
    for variant in VARIANTS:
        sha = item.variants[variant]["cells"]["C4a"]
        if sha is not None:
            payload = payloads.get(item.family, sha)
            blocks[variant] = canonical_json(payload.get("verification"))
            hits = verification_hits(payload)
    _require(blocks, f"{item.item_id}: no C4a payload carries the verification block", BindingError)
    _require(len(set(blocks.values())) == 1, f"{item.item_id}: C4a verification blocks differ across variants",
             BindingError)
    return hits


def policy_submissions(item: ItemTruth, payloads: PayloadStore) -> dict:
    """{variant: {policy: (sites, diagnostics)}} for ``none`` and every variant with a C1 answer."""
    hits = item_verification_hits(item, payloads)
    grep_all = [{"path": path, "line": line} for path, line in item.lexical_sites]
    out = {}
    for variant in (NO_VARIANT,) + VARIANTS:
        if variant == NO_VARIANT:
            sources = []
        else:
            sha = item.variants[variant]["cells"]["C1"]
            if sha is None:
                continue
            payload = payloads.get(item.family, sha)
            _require((payload.get("query") or {}).get("symbol", item.target) == item.target,
                     f"{item.item_id}: the {variant} C1 answer is for another symbol", BindingError)
            sources = answer_sources(payload, item.target)
        answer, without = [], []
        for source in sources:
            spans = item.caller_spans(source)
            if not spans:
                without.append(source)
                continue
            answer.extend({"path": span.path, "line": span.lineno} for span in spans)
        union, seen = [], set()
        for site in answer + hits:
            key = (site["path"], site["line"])
            if key not in seen:
                seen.add(key)
                union.append(site)
        diagnostics = {"answer_callers": len(sources), "answer_callers_without_sites": without,
                       "omitted_in_answer": item.omitted in sources}
        out[variant] = {"answer_only": (answer, diagnostics), "c4_hits": (hits, {"hits": len(hits)}),
                        "answer_plus_c4": (union, diagnostics), "grep_all": (grep_all, {"lines": len(grep_all)})}
    return out


def policy_rows(items: dict, payloads: PayloadStore, index_for_family, *, items_sha256) -> list:
    rows = []
    for item_id in sorted(items):
        item = items[item_id]
        if item.nm is None and item.natural_answer is None:
            attach_natural_answer(item, payloads)
        for variant, policies in policy_submissions(item, payloads).items():
            for policy in POLICIES:
                sites, diagnostics = policies[policy]
                row = {"schema": POLICY_SCHEMA, "kind": "policy", "policy": policy,
                       "policy_row_id": f"policy|{policy}|{item_id}|{variant}", "cell_id": None, "attempt_id": None,
                       "block": None, "item_id": item_id, "family": item.family, "role": item.role,
                       "variant": variant, "condition": None, "seed": None, "wording_pair": item.wording_pair,
                       "omitted": item.omitted, "exit_status": "Submitted", "completed": True,
                       "items_sha256": items_sha256, "diagnostics": diagnostics}
                row.update(score_submission(sites, None, item, index_for_family(item.family)))
                rows.append(row)
    return rows


def summarize_policies(rows) -> dict:
    """Plain item counts per (role, variant, policy); the analysis reports family-weighted rates."""
    groups = {}
    for row in rows:
        key = f"{row.get('role')}/{row['variant']}/{row['policy']}"
        group = groups.setdefault(key, {"items": 0, "site_recovered": 0, "function_localized": 0, "recall": 0.0,
                                        "n_submitted": 0, "n_unique_sites": 0, "categories": Counter(),
                                        "precision_lower": [], "precision_upper": [], "nm_items": 0, "nm_any": 0})
        group["items"] += 1
        for name in ("site_recovered", "function_localized", "n_submitted", "n_unique_sites"):
            group[name] += row[name]
        group["recall"] += row["recall_V_callers"]
        group["categories"].update(row["precision_categories"])
        for name in ("precision_lower", "precision_upper"):
            if row[name] is not None:
                group[name].append(row[name])
        if row.get("nm_any") is not None:
            group["nm_items"] += 1
            group["nm_any"] += row["nm_any"]
    out = {}
    for key, g in sorted(groups.items()):
        out[key] = {"items": g["items"], "site_recovered": g["site_recovered"],
                    "function_localized": g["function_localized"],
                    "mean_recall_V_callers": g["recall"] / g["items"],
                    "mean_submitted": g["n_submitted"] / g["items"],
                    "mean_unique_sites": g["n_unique_sites"] / g["items"],
                    "categories": {name: g["categories"].get(name, 0) for name in CATEGORIES},
                    "mean_precision_lower": (sum(g["precision_lower"]) / len(g["precision_lower"]))
                    if g["precision_lower"] else None,
                    "mean_precision_upper": (sum(g["precision_upper"]) / len(g["precision_upper"]))
                    if g["precision_upper"] else None,
                    "nm_items": g["nm_items"], "nm_any": g["nm_any"]}
    return out


def index_factory(checkouts: dict):
    """family -> FunctionIndex over that family's checkout (built lazily, cached)."""
    from .identity import FunctionIndex
    cache = {}

    def get(family):
        if family not in cache:
            _require(family in checkouts, f"no checkout for family {family}")
            cache[family] = FunctionIndex(checkouts[family])
        return cache[family]
    return get


__all__ = [
    "ALL_BLOCKS", "ANSWER_CONDITIONS", "BindingError", "Block", "CATEGORIES", "CELL_STATES", "CLAIMS",
    "COMPLETED_EXITS", "CONDITIONS", "DEFAULT_DESIGN", "DESIGNS", "Design", "EXITS", "EXPOSURE_FIELDS",
    "INFRA_RETRY_CAP", "ItemTruth", "MISSING", "NO_ANSWER_CONDITIONS", "NO_VARIANT", "OUTCOME_FIELDS", "POLICIES",
    "P_SEEDS", "PayloadStore", "ROW_STATUSES", "SCORE_SCHEMA", "SECONDARY_SEED", "STRATA", "ScoreInputError", "Span",
    "TERMINAL_STATES", "T_CONDITIONS", "UNRESOLVED_REASONS", "VARIANTS", "answer_sources", "attach_natural_answer",
    "categorize_sites", "cell_id", "check_frozen_cells", "check_inputs_binding", "check_result_binding",
    "detect_design", "episode_telemetry", "index_factory", "item_verification_hits", "load_dispatch", "load_items",
    "normalize_site_line", "normalize_site_path", "package_tree_sha256", "parse_cell_id", "parse_result",
    "policy_rows", "policy_submissions", "resolve_cell", "run_items_sha256", "scan_run", "score_cell", "score_run",
    "score_submission", "sha256_bytes", "sha256_file", "strict_loads", "summarize_policies", "write_immutable",
]
