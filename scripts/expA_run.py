"""Exp A queue runner (docs/EXPA_DESIGN.md section 7; docs/EXPA_DESIGN_V2_DELTA.md D8, D10.7, D10.8, D10.14, D11).

Inputs (D10.7, interface contract item 4): ONLY the dispatch view
(dispatch.jsonl + dispatch_manifest.json), the payload store of the dispatched
cells (payloads/<family>.jsonl; a row that no design cell names is refused) and
the per-family natural LocAgent graph facts for the T arms
(graph_facts/<family>.json), all under one dispatch directory, plus the clean
checkouts and their clean checkout manifest. No items.jsonl and no truth file is
read: a dispatch row with any key outside the dispatch-view schema is refused.
The runtime host can nevertheless derive V and C* from the payloads it needs
(the vcomplete C1 endpoints of S-VC minus the omit_node C1 endpoints of P), and
membership.in_s_nat encodes natural_miss: the tool jail, which shows the model
only the repository checkout, is the isolation boundary.

Cells (D8 as amended by D11.4; D10.8 order) come from harness.BLOCK_DESIGN:
block P seed by seed (11, 23, 47), items in sha256("expA-order-v2|"+item_id)
order with an item's five cells together (none x C0, then omit_node x C1, C2,
C3, C4a), then S-VC, S-NAT, S-EDGE and T at seed 11. There is no S-C0 block.
Cell id = "{block}|{item_id}|{variant}|{condition}|s{seed}". run.json and
cells.jsonl freeze the whole design once; ``--blocks``, ``--seeds``,
``--families`` and ``--conditions`` only select which frozen cells a launch may
dispatch, so P can run first and the secondary blocks later under the same
design. The launch record lists the selection (``conditions`` only when given).

Live mode: freezes run.json (binding dispatch_sha256, items_sha256, payload
store and graph facts hashes, the clean checkout manifest and its tree hashes,
prompts, tools, episode budgets and code hashes) + cells.jsonl, opens the
SQLite lease ledger on LOCAL disk (refuses NFS/FUSE), reconciles it with the
durable attempt records, stages each family's clean checkout into local scratch
(symlinks resolving outside it are dropped and recorded, finding F6), records
each staged checkout's whole-tree sha256 and refuses it unless it equals its
clean-checkout manifest entry exactly (nothing unexpected, nothing missing,
finding F11), refuses a package tree that differs from the payloads' source or
a C4a block that differs from the live source_reference_search, self-tests the
jail, then runs N concurrent episodes per endpoint, all pulling one shared
queue. ``--endpoint URL=GPU-UUID`` is repeatable and each attempt records its
endpoint and GPU UUID. Every attempt writes attempt_start.json durably before
dispatch. A completed cell is never dispatched again; only InfraInterrupted
attempts give the cell a new attempt id (at most 3 per cell, D11.3).

Endpoint isolation: three consecutive infrastructure failures on ONE endpoint
trip that endpoint (its workers finish their episode and stop; another
endpoint's success never resets it), and so do --health-failures failed /health
probes in a row. The endpoint that last failed a cell does not take its next
attempt while another endpoint is alive, and idle workers stay while the
selection still has open cells, so a healthy endpoint picks up requeued cells.
The runner halts when every endpoint has tripped, or at once on a controller
exception. SIGTERM/SIGINT drain; a second signal stops in-flight episodes at
their next model call.

Order across launches (D10.8, D11.3): a launch is refused when it would start a
segment (P s11 < P s23 < P s47 < S-VC < S-NAT < S-EDGE < T) while an earlier
segment still has pending, leased or running cells outside the selection, unless
--allow-out-of-order REASON is given; the reason is written into the launch
record.

Model and budgets (protocols/EXPA_REPL12B.md, protocols/EXPA_BUDGET16.md): --model-profile (default gemma4-31b,
serve.MODEL_PROFILES) sets the served alias every endpoint must serve and the revision recorded in each episode's
model_profile; --max-model-calls (default 8) and --context-budget (default 16384, at most the served 32768) build the
EpisodeConfig of every episode and of the rendered system prompt, which states the response budget. run.json records
the profile and the episode config, so a relaunch with other values is refused. The defaults are the prereg values.
A profile served with a chat-template override (gemma4-31b-qat) also records it (path and sha256) in run.json's model.
--final-turn-policy (protocols/EXPA_FINALTURN.md; default none, the prereg harness) appends the neutral or deadline
notice to the last tool result before every episode's last response (harness.py). REGISTERED_DESIGNS allows it only
with the prereg budget 8/16384 and the gemma4-31b profile, and only with --blocks P and one seed, --seeds 11 (A1) or
--seeds 23 (the protocols' cells; a relaunch without that selection would dispatch unregistered cells), optionally
restricted with --conditions to C0 and/or C1. Follow-up A4 (protocols/EXPA_A4_DIRECTIVE.md) adds the policies
deadline_info and submit_only under the same rules. run.json's episode_config records the policy and the exact notice
only when it is not none (EpisodeConfig.record), so a default run.json is unchanged and a relaunch with another policy
is refused.
--evidence-ledger (protocols/EXPA_A3_LEDGER.md; default none) runs an A3 arm: off or on. Both need
--final-turn-policy deadline, the prereg budget and the gemma4-31b profile (REGISTERED_LEDGER_DESIGNS), and both build
the A3 design (harness.DESIGN_ORDERS: block LEDGER, vcomplete x C1 and omit_node x C1 at seeds 61, 73 and 89, then the
default blocks) and dispatch only --blocks LEDGER (any of its seeds). Under on, the harness appends the evidence
ledger before the deadline notice; under off it records the ledger without sending it. run.json records the option,
its rules and the A3 design only when it is not none, so a default run.json and its cells are unchanged.
--submit-review (protocols/EXPA_A5_REVIEW.md; default none) runs an A5 arm: neutral or ledger. Both need
--final-turn-policy deadline and --evidence-ledger none (the A3 off arm's model-visible requests, without its
last-call ledger record), the prereg budget and the gemma4-31b profile (REGISTERED_REVIEW_DESIGNS); both build the A3
design and dispatch only --blocks LEDGER (any of its seeds). The harness reviews the first submission made before the
last call. run.json records the option and its rules only when it is not none.
Follow-up A6 (protocols/EXPA_A6_BUDGET16.md) is A3's off arm at the registered 16-response budget: --final-turn-policy
deadline --evidence-ledger off --max-model-calls 16 --context-budget 32768 with the gemma4-31b profile
(REGISTERED_A6_DESIGNS), on the A3 design and only --blocks LEDGER (any of its seeds). The harness derives the last call
from max_model_calls: the deadline notice comes before the 16th response and the unsent ledger is built at call_index
15. No other final-turn policy, ledger arm or review runs with the 16-response budget.
--task-framing (protocols/EXPA_A7_IMPACT.md; default default, every earlier harness) runs follow-up A7: impact replaces
the task paragraph with the impact-analysis framing (harness.IMPACT_TASK_TEMPLATE). It is registered only with the
prereg budget 8/16384, the gemma4-31b profile, no final-turn policy, evidence ledger or submit review
(REGISTERED_FRAMING_DESIGNS), on the default design, and only with the A7 selection --blocks P,S-VC --seeds 11
--conditions C0,C1: block P none x C0 (no list) and omit_node x C1 (incomplete list), and S-VC vcomplete x C1 (complete
list). run.json records the framing and its text only when it is not default. On a fresh run the D10.8 order guard
refuses this selection (P seed 11 C2-C4a and P seeds 23, 47 stay open before S-VC) unless --allow-out-of-order REASON
is given.

Subcommands:
  (default)          the runner; --dry-run builds every cell, verifies every payload, renders every prompt,
                     optionally counts first-prompt tokens offline (--tokenizer-dir) and checks the source
                     checkouts read-only (--checkout-manifest + --repos-root); a dry run writes nothing
  checkout-manifest  write the clean checkout manifest (expA-clean-checkouts-v1) for the dispatch families
  appendix           print the frozen operational details (prereg Appendix A, D11.6)
  status             summarize a run directory from its durable records only (exits per block x condition,
                     condition x endpoint, primary-seed completeness, infrastructure attempts)
  reconcile          with the runner stopped: close attempts whose owner is verified gone (attempt_closed.json)
                     and rebuild the ledger from the durable records; run before the final publish
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import collections
import dataclasses
import datetime
import hashlib
import json
import os
import re
import signal
import socket
import statistics
import threading
import time
import traceback
import urllib.request
import uuid

from absent_edges.data import canonical_json
from absent_edges.models import LocalModel, ModelProfile
from absent_edges.probe import harness
from absent_edges.probe import ledger as ledger_module
from absent_edges.probe import serve
from absent_edges.probe import tools
from absent_edges.probe.identity import file_sha256s, normalize_package_dirs, scan_package_files, source_tree_sha256
from absent_edges.probe.identity import split_identity
from absent_edges.registry import FenceError

RUNNER_PROTOCOL = "expA-runner-v2"
DISPATCH_SCHEMA = "expA-dispatch-v2"
CHECKOUTS_SCHEMA = tools.CHECKOUTS_SCHEMA
GPU5_UUID = "GPU-7b8d9f89-0a8f-2d07-4833-bee70a0eeed2"
_GPU_UUID = re.compile(r"GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_FAMILY = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
# The dispatch-view schema (contract item 4). Anything else in a row is refused: the runtime host never
# receives truth (V, U, sites, the omitted identity, covariates, variants, ...), D10.7.
DISPATCH_ROW_KEYS = frozenset({"item_id", "family", "target", "package_dirs", "wording_pair", "membership", "cells"})
DISPATCH_ROW_OPTIONAL = frozenset({"schema"})
TARGET_KEYS = frozenset({"identity", "path", "line", "qualname"})
MEMBERSHIP_FLAGS = ("in_s_nat", "in_s_edge", "in_t")
# Exactly items.DISPATCH_MEMBERSHIP_KEYS: s_nat_excluded_reason would reveal natural_miss, so it is refused.
MEMBERSHIP_KEYS = frozenset(MEMBERSHIP_FLAGS)
PAYLOAD_ROW_KEYS = frozenset({"sha256", "bytes", "payload"})


class OrderError(RuntimeError):
    """A launch would overtake earlier D10.8 segments that still have open cells."""


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def segment_key(cell):
    """The ledger's queue segment of a cell: one (block, seed), D10.8 order."""
    return f"{cell['block']}|s{cell['seed']}"


def order_violations(ledger, selected, eligible):
    """Open cells outside the selection in segments ranked before the selection's last segment.

    D10.8 / D11.3: P runs first, one seed at a time (11, 23, 47), then S-VC, S-NAT, S-EDGE and T.
    Within a launch the shared queue keeps that order; across launches a selection may not start a
    later segment while an earlier one (any family) still has pending, leased or running cells.
    """
    ranks = [ledger.segment_rank(key) for key in {segment_key(cell) for cell in selected}]
    ranks = [rank for rank in ranks if rank is not None]
    if not ranks:
        return {}
    found = collections.Counter(key for cell_id, key in ledger.open_cells_before(max(ranks)) if cell_id not in eligible)
    return dict(sorted(found.items()))


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _require_sha(value, what):
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{what} must be a sha256 (64 lowercase hex digits), not {value!r}")
    return value


def _summary(values):
    return {"n": len(values), "min": min(values), "median": statistics.median(values), "max": max(values)} \
        if values else None


# ---------------------------------------------------------------------------
# Inputs: the dispatch view (contract item 4)
# ---------------------------------------------------------------------------

def _relative_file(directory, value, default):
    """A manifest path relative to the dispatch directory; absolute or escaping paths are refused."""
    if value is None:
        return directory / default
    if not isinstance(value, str) or not value or value.startswith("/") or ".." in value.split("/") \
            or "\\" in value:
        raise ValueError(f"dispatch manifest path {value!r} must be relative to the dispatch directory")
    return directory / value


def _file_entries(section, directory, default_subdir, suffix, what):
    if not isinstance(section, dict):
        raise ValueError(f"dispatch_manifest.json {what} must map family -> sha256")
    entries = {}
    for family, value in sorted(section.items()):
        if not isinstance(family, str) or not _FAMILY.fullmatch(family):
            raise ValueError(f"dispatch_manifest.json {what}: invalid family {family!r}")
        default = f"{default_subdir}/{family}{suffix}"
        if isinstance(value, str):
            entries[family] = {"path": directory / default, "sha256": _require_sha(value, f"{what}[{family}]")}
        elif isinstance(value, dict):
            entries[family] = {"path": _relative_file(directory, value.get("path"), default),
                               "sha256": _require_sha(value.get("sha256"), f"{what}[{family}].sha256")}
        else:
            raise ValueError(f"dispatch_manifest.json {what}[{family}] must be a sha256 or an object")
    return entries


def validate_dispatch_row(row):
    """One dispatch.jsonl row: exactly the dispatch-view schema, nothing else (D10.7)."""
    if not isinstance(row, dict):
        raise ValueError("dispatch rows must be objects")
    item_id = row.get("item_id")
    extra = sorted(set(row) - DISPATCH_ROW_KEYS - DISPATCH_ROW_OPTIONAL)
    if extra:
        raise ValueError(f"dispatch row {item_id!r} carries keys outside the dispatch view {extra}; the runtime "
                         "host receives only the dispatch view (no truth)")
    missing = sorted(DISPATCH_ROW_KEYS - set(row))
    if missing:
        raise ValueError(f"dispatch row {item_id!r} lacks {missing}")
    ledger_module.cell_identifier({"block": "P", "item_id": item_id, "variant": "none", "condition": "C0", "seed": 0})
    family = row["family"]
    if not isinstance(family, str) or not _FAMILY.fullmatch(family):
        raise ValueError(f"{item_id}: invalid family {family!r}")
    target = row["target"]
    if not isinstance(target, dict) or set(target) != TARGET_KEYS:
        raise ValueError(f"{item_id}: target must have exactly {sorted(TARGET_KEYS)}")
    path, qualname = split_identity(target["identity"])
    if target["path"] != path or target["qualname"] != qualname:
        raise ValueError(f"{item_id}: target path/qualname disagree with its identity")
    if type(target["line"]) is not int or target["line"] < 1:
        raise ValueError(f"{item_id}: target line must be a positive integer")
    package_dirs = row["package_dirs"]
    if not isinstance(package_dirs, list) or not package_dirs:
        raise ValueError(f"{item_id}: package_dirs must be a nonempty list")
    normalize_package_dirs(package_dirs)
    if row["wording_pair"] not in harness.WORDING_PAIRS:
        raise ValueError(f"{item_id}: wording_pair must be A or B")
    membership = row["membership"]
    if not isinstance(membership, dict) or set(membership) != MEMBERSHIP_KEYS \
            or any(type(membership.get(key)) is not bool for key in MEMBERSHIP_FLAGS):
        raise ValueError(f"{item_id}: membership must be exactly the boolean flags {', '.join(MEMBERSHIP_FLAGS)} "
                         "(s_nat_excluded_reason and every other key would reveal truth)")
    cells = row["cells"]
    if not isinstance(cells, dict):
        raise ValueError(f"{item_id}: cells must map variant -> condition -> sha256")
    for variant, by_condition in cells.items():
        if variant not in harness.VARIANTS or not isinstance(by_condition, dict):
            raise ValueError(f"{item_id}: unknown variant {variant!r} in cells")
        for condition, value in by_condition.items():
            if condition not in harness.PAYLOAD_CONDITIONS:
                raise ValueError(f"{item_id}: unknown payload condition {condition!r} in cells[{variant}]")
            if value is not None:
                _require_sha(value, f"{item_id} cells[{variant}][{condition}]")
    return row


class DispatchView:
    """dispatch.jsonl + dispatch_manifest.json; the manifest pins dispatch.jsonl and every file it names."""

    def __init__(self, directory, *, payloads_dir=None, graph_facts_dir=None):
        self.directory = Path(directory)
        self.path = self.directory / "dispatch.jsonl"
        raw = self.path.read_bytes()
        self.sha256 = sha256_bytes(raw)
        self.manifest_path = self.directory / "dispatch_manifest.json"
        manifest_raw = self.manifest_path.read_bytes()
        self.manifest_sha256 = sha256_bytes(manifest_raw)
        manifest = json.loads(manifest_raw)
        if not isinstance(manifest, dict):
            raise ValueError("dispatch_manifest.json must be an object")
        if manifest.get("schema") != DISPATCH_SCHEMA:
            raise ValueError(f"dispatch manifest schema {manifest.get('schema')!r} is not {DISPATCH_SCHEMA}")
        pins = [manifest[key] for key in ("dispatch_jsonl_sha256", "dispatch_sha256") if manifest.get(key) is not None]
        if not pins:
            raise ValueError("dispatch_manifest.json does not pin dispatch.jsonl (dispatch_jsonl_sha256)")
        if any(pin != self.sha256 for pin in pins):
            raise ValueError("dispatch.jsonl sha256 differs from the one dispatch_manifest.json pins")
        self.manifest = manifest
        self.items_sha256 = _require_sha(manifest.get("items_sha256"), "dispatch_manifest.json items_sha256")
        self.items_manifest_sha256 = manifest.get("manifest_sha256")
        if self.items_manifest_sha256 is not None:
            _require_sha(self.items_manifest_sha256, "dispatch_manifest.json manifest_sha256")
        self.payload_files = _file_entries(manifest.get("payload_store"), self.directory, "payloads", ".jsonl",
                                           "payload_store")
        self.graph_files = _file_entries(manifest.get("graph_facts") or {}, self.directory, "graph_facts", ".json",
                                         "graph_facts")
        if payloads_dir is not None:
            for family, entry in self.payload_files.items():
                entry["path"] = Path(payloads_dir) / f"{family}.jsonl"
        if graph_facts_dir is not None:
            for family, entry in self.graph_files.items():
                entry["path"] = Path(graph_facts_dir) / f"{family}.json"
        self.code = manifest.get("code")
        rows = {}
        for number, line in enumerate(raw.splitlines(), 1):
            if not line.strip():
                continue
            row = validate_dispatch_row(json.loads(line))
            if row.get("schema", DISPATCH_SCHEMA) != DISPATCH_SCHEMA:
                raise ValueError(f"dispatch.jsonl line {number}: schema {row.get('schema')!r}")
            if row["item_id"] in rows:
                raise ValueError(f"duplicate item id {row['item_id']} in dispatch.jsonl")
            rows[row["item_id"]] = row
        if not rows:
            raise ValueError("dispatch.jsonl has no rows")
        self.rows = rows
        self.order = sorted(rows, key=harness.item_order_key)
        self.families = sorted({row["family"] for row in rows.values()})
        missing = [family for family in self.families if family not in self.payload_files]
        if missing:
            raise ValueError("dispatch_manifest.json payload_store lacks families: " + ", ".join(missing))
        self.package_dirs = {}
        for row in rows.values():
            dirs = normalize_package_dirs(row["package_dirs"])
            if self.package_dirs.setdefault(row["family"], dirs) != dirs:
                raise ValueError(f"family {row['family']} has inconsistent package_dirs across rows")
        self.family_sources = {}
        declared = manifest.get("families")
        if declared is not None:
            if not isinstance(declared, dict):
                raise ValueError("dispatch_manifest.json families must be an object")
            for family in self.families:
                entry = declared.get(family)
                if not isinstance(entry, dict):
                    raise ValueError(f"dispatch_manifest.json families lacks {family}")
                if entry.get("package_dirs") is not None and \
                        normalize_package_dirs(entry["package_dirs"]) != self.package_dirs[family]:
                    raise ValueError(f"{family}: manifest package_dirs differ from the dispatch rows")
                if entry.get("source_sha256") is not None:
                    self.family_sources[family] = _require_sha(entry["source_sha256"], f"families[{family}]")

    def describe(self):
        return {"schema": DISPATCH_SCHEMA, "sha256": self.sha256, "rows": len(self.rows),
                "manifest_sha256": self.manifest_sha256, "items_sha256": self.items_sha256,
                "items_manifest_sha256": self.items_manifest_sha256,
                "payload_store": {family: entry["sha256"] for family, entry in self.payload_files.items()},
                "graph_facts": {family: entry["sha256"] for family, entry in self.graph_files.items()},
                "family_sources": dict(sorted(self.family_sources.items())), "item_build_code": self.code}


class PayloadStore:
    """payloads/<family>.jsonl rows {sha256, bytes, payload}: the file sha256 and every row are verified."""

    def __init__(self, view: DispatchView):
        self.view = view
        self.cache = {}

    def family(self, family):
        if family not in self.cache:
            entry = self.view.payload_files.get(family)
            if entry is None:
                raise KeyError(f"no payload store for family {family}")
            raw = Path(entry["path"]).read_bytes()
            if sha256_bytes(raw) != entry["sha256"]:
                raise harness.PayloadIntegrityError(f"payload store of {family} differs from the dispatch manifest")
            rows = {}
            for line in raw.splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict) or set(row) - PAYLOAD_ROW_KEYS or "payload" not in row:
                    raise harness.PayloadIntegrityError(f"payload row of {family} is not {{sha256, bytes, payload}}")
                encoded = canonical_json(row["payload"])
                if sha256_bytes(encoded) != row.get("sha256"):
                    raise harness.PayloadIntegrityError(f"payload row of {family} does not match its sha256")
                if row.get("bytes") is not None and row["bytes"] != len(encoded):
                    raise harness.PayloadIntegrityError(f"payload row of {family} has a wrong byte count")
                rows[row["sha256"]] = encoded
            self.cache[family] = rows
        return self.cache[family]

    def get(self, family, sha256):
        return self.family(family)[sha256]


def load_graph_tool(view, family):
    entry = view.graph_files.get(family)
    if entry is None:
        raise ValueError(f"dispatch_manifest.json has no graph_facts for family {family} (needed by the T arms)")
    return tools.GraphQueryTool(Path(entry["path"]).read_bytes(), expected_sha256=entry["sha256"])


# ---------------------------------------------------------------------------
# Cells (D8 + D11.4, D10.8)
# ---------------------------------------------------------------------------

def build_cells(view: DispatchView, payloads: PayloadStore = None, design=None):
    """The whole frozen design in dispatch order (``design`` None: the default; harness.LEDGER_DESIGN: A3's). With
    ``payloads``, every payload is verified."""
    cells = []
    for block in harness.design_order(design):
        design_spec = harness.block_design(block)
        for seed in design_spec["seeds"]:
            for item_id in view.order:
                row = view.rows[item_id]
                if design_spec["membership"] and not row["membership"][design_spec["membership"]]:
                    continue
                for variant, condition in design_spec["cells"]:
                    sha = None
                    if condition in harness.PAYLOAD_CONDITIONS:
                        sha = (row["cells"].get(variant) or {}).get(condition)
                        if sha is None:
                            raise ValueError(f"{item_id}: block {block} needs a {variant} {condition} payload, but "
                                             "the dispatch view has none")
                        if payloads is not None:
                            harness.check_payload_structure(condition, json.loads(payloads.get(row["family"], sha)),
                                                            row["target"]["identity"])
                    cells.append(ledger_module.make_cell(
                        block=block, item_id=item_id, family=row["family"], target=row["target"]["identity"],
                        variant=variant, condition=condition, seed=seed, wording_pair=row["wording_pair"],
                        payload_sha256=sha))
    identifiers = [cell["cell_id"] for cell in cells]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("duplicate cells")
    if payloads is not None:  # D10.7: the staged store holds exactly the dispatched cells' payloads
        for family in view.families:
            needed = {cell["payload_sha256"] for cell in cells if cell["family"] == family and cell["payload_sha256"]}
            extra = set(payloads.family(family)) - needed
            if extra:
                raise harness.PayloadIntegrityError(f"payload store of {family} holds {len(extra)} payloads that no "
                                                    "design cell names; the runtime host receives only the "
                                                    "dispatched cells (D10.7)")
    return cells


def parse_selection(text, allowed, cast=str, what="value"):
    if text is None:
        return None
    try:
        values = {cast(value.strip()) for value in text.split(",") if value.strip()}
    except ValueError as error:
        raise ValueError(f"invalid {what}: {text!r}") from error
    unknown = values - set(allowed) if allowed is not None else set()
    if not values or unknown:
        raise ValueError(f"unknown {what}: {sorted(unknown) or text!r}")
    return values


def select_cells(cells, blocks=None, seeds=None, families=None, conditions=None):
    return [cell for cell in cells if (blocks is None or cell["block"] in blocks)
            and (seeds is None or cell["seed"] in seeds) and (families is None or cell["family"] in families)
            and (conditions is None or cell["condition"] in conditions)]


def episode_input(cell, view, payloads, *, run_id, attempt_id, token):
    row = view.rows[cell["item_id"]]
    if (row["family"], row["target"]["identity"], row["wording_pair"]) != (cell["family"], cell["target"],
                                                                         cell["wording_pair"]):
        raise ValueError(f"cell {cell['cell_id']} disagrees with its dispatch row")
    payload = None if cell["payload_sha256"] is None else payloads.get(cell["family"], cell["payload_sha256"])
    return harness.EpisodeInput(
        run_id=run_id, cell_id=cell["cell_id"], attempt_id=attempt_id, token=token, block=cell["block"],
        item_id=cell["item_id"], family=cell["family"], target=cell["target"], target_line=row["target"]["line"],
        package_dirs=tuple(row["package_dirs"]), variant=cell["variant"], condition=cell["condition"],
        seed=cell["seed"], wording_pair=cell["wording_pair"], payload=payload,
        payload_sha256=cell["payload_sha256"], dispatch_sha256=view.sha256, items_sha256=view.items_sha256)


def cell_label(cell):
    return f"{cell['block']}:{cell['variant']}:{cell['condition']}"


# ---------------------------------------------------------------------------
# Clean checkouts (F6, F11)
# ---------------------------------------------------------------------------

def package_tree(root, package_dirs):
    files = scan_package_files(root, list(package_dirs))["files"]
    return source_tree_sha256(file_sha256s(root, files)), len(files)


def load_checkouts_manifest(path, families):
    """The clean checkout manifest (tools.CHECKOUTS_SCHEMA): one self-consistent entry per dispatch family."""
    raw = Path(path).read_bytes()
    manifest = json.loads(raw)
    if (not isinstance(manifest, dict) or manifest.get("schema") != tools.CHECKOUTS_SCHEMA
            or manifest.get("tree_protocol") != tools.TREE_PROTOCOL or not isinstance(manifest.get("families"), dict)):
        raise ValueError(f"{path} is not a {tools.CHECKOUTS_SCHEMA} manifest with tree_protocol {tools.TREE_PROTOCOL}")
    entries = {}
    for family in families:
        entry = manifest["families"].get(family)
        if not isinstance(entry, dict):
            raise ValueError(f"{path} has no entry for dispatch family {family}")
        problems = tools.checkout_entry_problems(entry)
        if problems:
            raise ValueError(f"{path}: the {family} entry cannot pin a checkout: {problems}")
        if entry.get("escaping_symlinks"):
            raise ValueError(f"{path}: the {family} entry lists symlinks that leave the checkout")
        entries[family] = entry
    return manifest, sha256_bytes(raw), entries


def write_checkouts_manifest(families, out_path, *, repos_root, package_dirs, commits=None):
    """The clean checkout manifest for ``families`` (checkout of F = REPOS_ROOT/F), written once, immutable."""
    entries = [tools.clean_checkout_entry(family, Path(repos_root) / family, package_dirs[family],
                                          commit=(commits or {}).get(family)) for family in sorted(families)]
    manifest = tools.checkouts_manifest(entries)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    digest = ledger_module.write_immutable_json(out_path, manifest)
    return {"path": str(out_path), "sha256": digest, "schema": tools.CHECKOUTS_SCHEMA,
            "tree_protocol": tools.TREE_PROTOCOL,
            "families": {entry["family"]: {"tree_sha256": entry["tree_sha256"], "files": len(entry["files"]),
                                           "symlinks": len(entry["symlinks"]),
                                           "directories": len(entry["directories"]),
                                           "package_tree_sha256": entry["package_tree_sha256"],
                                           "skipped": len(entry["skipped"]),
                                           "escaping_symlinks": len(entry["escaping_symlinks"]),
                                           "stager_accepts": not entry["skipped"] and not entry["escaping_symlinks"]}
                         for entry in entries}}


def verify_family_sources(family, root, view, payloads, cells, entry=None):
    """Package tree = dispatch family source = payload source = graph facts source; C4a blocks = the live search.

    ``root`` is the checkout to check (the staged copy in live mode, the source in a dry run).
    Returns (record, graph tool or None). Raises RuntimeError on any mismatch.
    """
    package_dirs = view.package_dirs[family]
    tree_sha, tree_files = package_tree(root, package_dirs)
    record = {"package_tree_sha256": tree_sha, "package_files": tree_files}
    expected = {"dispatch_manifest": view.family_sources.get(family),
                "checkout_manifest": None if entry is None else entry.get("package_tree_sha256")}
    for source, value in expected.items():
        if value is not None and value != tree_sha:
            raise RuntimeError(f"{family}: package tree {tree_sha} differs from the {source}'s {value}")
    family_cells = [cell for cell in cells if cell["family"] == family]
    sources, live, equal = set(), {}, 0
    for sha in sorted({cell["payload_sha256"] for cell in family_cells if cell["payload_sha256"]}):
        payload = json.loads(payloads.get(family, sha))
        sources.add(payload.get("source_sha256"))
        verification = payload.get("verification")
        if verification is None:
            continue
        if verification.get("budget", {}).get("token_bound") != tools.token_bound_name(None):
            raise RuntimeError(f"{family}: a C4a block uses token bound "
                               f"{verification.get('budget', {}).get('token_bound')!r}, not the live tool's")
        query = verification.get("query")
        if query not in live:
            live[query] = tools.lexical_search(root, list(package_dirs), query, limit=tools.SEARCH_MAX_HITS,
                                               token_budget=tools.SEARCH_TOKEN_BUDGET)
        if canonical_json(live[query]) != canonical_json(verification):
            raise RuntimeError(f"{family}: the C4a block of payload {sha} differs from the live "
                               f"source_reference_search for {query!r} on the checkout")
        equal += 1
    if sources and sources != {tree_sha}:
        raise RuntimeError(f"checkout of {family} has package tree {tree_sha}, but its payloads were rendered from "
                           f"{sorted(map(str, sources))}; refusing to run against different source")
    record["payload_sources"] = sorted(map(str, sources))
    record["c4a_blocks_equal_to_live_search"] = equal
    graph = None
    if any(cell["condition"] in harness.T_CONDITIONS for cell in family_cells):
        graph = load_graph_tool(view, family)
        if graph.source_sha256 != tree_sha:
            raise RuntimeError(f"graph facts of {family} were built from {graph.source_sha256}, but the package "
                               f"tree is {tree_sha}")
        record["graph_query"] = graph.describe()
    return record, graph


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

def render_cell(cell, view, payloads, config=None):
    episode = episode_input(cell, view, payloads, run_id="dry-run", attempt_id="dry-run", token=0)
    return episode, harness.render_messages(episode, config or harness.EpisodeConfig())


class OfflineTokens:
    """First-prompt tokens with the model's tokenizer files on CPU: the chat template with tools,
    add_generation_prompt=True, enable_thinking=False, add_special_tokens=False (as LocalModel.tokenize asks
    the server). An approximation of the served /tokenize count; needs transformers (no torch, no GPU)."""

    def __init__(self, model_dir):
        from transformers import AutoTokenizer
        self.directory = Path(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.directory), local_files_only=True)
        self.files = {name: sha256_bytes((self.directory / name).read_bytes())
                      for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
                      if (self.directory / name).is_file()}

    def first_prompt(self, rendered):
        messages = [{"role": "system", "content": rendered["system"]}, {"role": "user", "content": rendered["user"]}]
        text = self.tokenizer.apply_chat_template(messages, tools=rendered["tools"], add_generation_prompt=True,
                                                  enable_thinking=False, tokenize=False)
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def plain(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))


def dry_run(view, payloads, cells, selected, *, show=0, tokens=None, checkouts=None, config=None):
    config = config or harness.EpisodeConfig()
    prompts, examples, per_prompt = {}, {}, {}
    sizes = collections.defaultdict(lambda: collections.defaultdict(list))
    by_block, by_family, by_condition = collections.Counter(), collections.Counter(), collections.Counter()
    first_tokens = {}
    for cell in selected:
        episode, rendered = render_cell(cell, view, payloads, config)
        label = cell_label(cell)
        system, user = rendered["system"].encode("utf-8"), rendered["user"].encode("utf-8")
        key = sha256_bytes(system + b"\0" + user + b"\0" + canonical_json(rendered["tools"]))
        prompts.setdefault(key, label)
        sizes[label]["user_bytes"].append(len(user))
        sizes[label]["system_plus_user_bytes"].append(len(system) + len(user))
        sizes[label]["payload_bytes"].append(0 if episode.payload is None else len(episode.payload))
        if tokens is not None:
            if key not in per_prompt:
                per_prompt[key] = {"first_prompt_tokens": tokens.first_prompt(rendered),
                                   "payload_tokens": 0 if rendered["payload_text"] is None
                                   else tokens.plain(rendered["payload_text"])}
            sizes[label]["first_prompt_tokens"].append(per_prompt[key]["first_prompt_tokens"])
            sizes[label]["payload_tokens"].append(per_prompt[key]["payload_tokens"])
            first_tokens[cell["cell_id"]] = per_prompt[key]["first_prompt_tokens"]
        by_block[cell["block"]] += 1
        by_family[cell["family"]] += 1
        by_condition[label] += 1
        if show and label not in examples and len(examples) < show:
            examples[label] = {"cell_id": cell["cell_id"], "system": rendered["system"], "user": rendered["user"],
                               "tools_offered": rendered["tools_offered"]}
    summary = {"protocol": RUNNER_PROTOCOL, "mode": "dry-run", "writes": "none", "dispatch": view.describe(),
               "design": harness.design_table_for(harness.config_design(config)), "design_cells": len(cells),
               "selected_cells": len(selected),
               "items": len(view.rows), "by_block": dict(sorted(by_block.items())),
               "by_family": dict(sorted(by_family.items())), "by_condition": dict(sorted(by_condition.items())),
               "unique_prompts": len(prompts),
               "sizes": {label: {name: _summary(values) for name, values in sorted(metrics.items())}
                         for label, metrics in sorted(sizes.items())},
               "first_cells": [cell["cell_id"] for cell in selected[:10]], "examples": examples,
               "dispatch_manifest_blocks": view.manifest.get("blocks"),
               "dispatch_manifest_design_cells": view.manifest.get("design_cells"),
               "episode_config": config.record()}
    if tokens is not None:
        largest = max(first_tokens.values()) if first_tokens else 0
        deltas = collections.defaultdict(list)
        by_item = collections.defaultdict(dict)
        for cell in selected:
            if cell["block"] == "P" and cell["seed"] == harness.SEEDS[0] and cell["condition"] in ("C2", "C3"):
                by_item[cell["item_id"]][cell["condition"]] = first_tokens[cell["cell_id"]]
        for item_id, values in sorted(by_item.items()):
            if len(values) == 2:
                deltas[view.rows[item_id]["wording_pair"]].append(values["C3"] - values["C2"])
        summary["tokens"] = {"counter": "offline chat template + tokenizer.json (CPU)",
                             "tokenizer_files_sha256": tokens.files, "unique_prompts_counted": len(per_prompt),
                             "max_first_prompt_tokens": largest,
                             "max_first_prompt_plus_max_response": largest + config.max_response_tokens,
                             "context_budget": config.context_budget,
                             "every_first_prompt_leaves_a_full_response": largest + config.max_response_tokens
                             <= config.context_budget,
                             "first_prompts_hitting_context_limit": sum(
                                 1 for value in per_prompt.values()
                                 if config.context_budget - value["first_prompt_tokens"] < config.min_response_tokens),
                             "p_c3_minus_c2_first_prompt_tokens": {pair: _summary(values) for pair, values
                                                                   in sorted(deltas.items())}}
    if checkouts is not None:
        summary["checkouts"] = checkouts
    return summary


def dry_run_checkouts(view, payloads, selected, manifest_path, repos_root):
    """Read-only checks of the source checkouts a launch would stage (no copy, no writes)."""
    manifest, digest, entries = load_checkouts_manifest(manifest_path, view.families)
    report = {"manifest": str(manifest_path), "manifest_sha256": digest, "families": {}}
    for family in sorted({cell["family"] for cell in selected}):
        source = Path(repos_root) / family
        tree = tools.checkout_tree(source)
        comparison = tools.compare_tree(entries[family], tree)
        record = {"source": str(source), "tree_sha256": tree["tree_sha256"], "counts": tree["counts"],
                  "skipped_sample": tree["skipped"][:20], "matches_manifest": comparison["matches"]}
        if not comparison["matches"]:
            record["comparison"] = comparison
        else:
            record.update(verify_family_sources(family, source, view, payloads, selected, entries[family])[0])
        report["families"][family] = record
    return report


# ---------------------------------------------------------------------------
# Live runner
# ---------------------------------------------------------------------------

def http_json(endpoint, path, timeout=10):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(endpoint.rstrip("/") + path, timeout=timeout) as response:
        body = response.read(16 * 1024**2)
    return json.loads(body) if body.strip() else None


def check_endpoint(profile):
    models = http_json(profile.endpoint, "/v1/models")
    rows = models.get("data") if isinstance(models, dict) else None
    if (not isinstance(rows, list) or len(rows) != 1 or rows[0].get("id") != profile.model
            or rows[0].get("max_model_len") != profile.max_context):
        raise RuntimeError(f"{profile.endpoint} does not serve exactly {profile.model} with max_model_len "
                           f"{profile.max_context}")
    return {"endpoint": profile.endpoint, "models": models}


def parse_endpoint(text):
    """``http://127.0.0.1:PORT=GPU-<uuid>``: the loopback endpoint and the GPU UUID that serves it."""
    url, separator, gpu = text.partition("=")
    if not separator:
        raise ValueError(f"--endpoint must be URL=GPU-UUID, not {text!r}")
    if not _GPU_UUID.fullmatch(gpu):
        raise ValueError(f"--endpoint GPU must be a full lowercase GPU UUID (never an index): {gpu!r}")
    if gpu == GPU5_UUID:
        raise ValueError("GPU5 is never used")
    ModelProfile(endpoint=url)  # loopback-only check
    return url, gpu


class Runner:
    def __init__(self, args, view, payloads, cells, selected):
        self.args = args
        self.view, self.payloads, self.cells = view, payloads, cells
        self.selected = selected
        self.eligible = {cell["cell_id"] for cell in selected}
        self.run_dir = Path(args.run_dir)
        self.run_id = args.run_id or self.run_dir.name
        self.store = ledger_module.DurableStore(self.run_dir)
        self.config = args.episode_config          # --max-model-calls, --context-budget (prereg defaults 8, 16384),
                                                   # --final-turn-policy and --evidence-ledger (default none),
                                                   # --submit-review (none) and --task-framing (default)
        self.served = serve.served_model(args.model_profile)
        self.identity = ledger_module.process_identity()
        self.owner_prefix = f"{socket.gethostname()}/{os.getpid()}/{self.identity.get('start_ticks')}"
        self.lease_seconds, self.heartbeat_seconds = args.lease_seconds, max(5.0, args.lease_seconds / 10)
        self.drain, self.abort, self.finished = threading.Event(), threading.Event(), threading.Event()
        self.lock = threading.Lock()
        self.active = {}
        self.reserved = 0
        self.stats = collections.Counter()
        self.table = collections.defaultdict(collections.Counter)  # condition label -> endpoint -> exits
        self.halt_reason = None
        # Per-endpoint isolation (review finding: one failing endpoint exhausted primary cells while a healthy one
        # sat idle): a breaker per endpoint, the endpoint that last failed a cell does not take its next attempt
        # while another endpoint is alive, and a /health probe drains an endpoint that stops answering.
        self.endpoint_health = []
        self.last_failed_on = {}
        self.order_check = None
        # Every family of the design is bound in run.json, selected or not.
        self.checkouts_manifest, self.checkouts_manifest_sha256, self.checkout_entries = load_checkouts_manifest(
            args.checkout_manifest, view.families)

    # -- setup ----------------------------------------------------------------

    def manifest(self):
        """run.json: launch-invariant bindings only (no paths, times or selections)."""
        by_block = collections.Counter(cell["block"] for cell in self.cells)
        return {"protocol": RUNNER_PROTOCOL, "run_id": self.run_id,
                "dispatch": {"schema": DISPATCH_SCHEMA, "sha256": self.view.sha256, "rows": len(self.view.rows),
                             "manifest_sha256": self.view.manifest_sha256},
                "dispatch_sha256": self.view.sha256, "items_sha256": self.view.items_sha256,
                "items_manifest_sha256": self.view.items_manifest_sha256,
                "payload_store": {family: entry["sha256"] for family, entry in self.view.payload_files.items()},
                "graph_facts": {family: entry["sha256"] for family, entry in self.view.graph_files.items()},
                "family_sources": dict(sorted(self.view.family_sources.items())),
                "item_build_code": self.view.code,
                "design": {**harness.design_table_for(harness.config_design(self.config)),
                           "cells_by_block": dict(sorted(by_block.items()))},
                "families": list(self.view.families),
                "checkouts": {"schema": tools.CHECKOUTS_SCHEMA, "tree_protocol": tools.TREE_PROTOCOL,
                              "formula": tools.TREE_FORMULA, "manifest_sha256": self.checkouts_manifest_sha256,
                              "tree_sha256": {family: self.checkout_entries[family]["tree_sha256"]
                                              for family in self.view.families},
                              "package_tree_sha256": {family: self.checkout_entries[family].get("package_tree_sha256")
                                                      for family in self.view.families}},
                "episode_config": self.config.record(),       # the final-turn policy only when not none
                "infra_attempt_cap": self.args.max_attempts,
                "prompts": harness.prompt_templates(),
                "tools": {"protocol": tools.TOOLS_PROTOCOL,
                          "p_schemas_sha256": sha256_bytes(canonical_json(tools.tool_schemas())),
                          "t_schemas_sha256": sha256_bytes(canonical_json(tools.tool_schemas(graph_query=True))),
                          "search_max_hits": tools.SEARCH_MAX_HITS, "search_token_budget": tools.SEARCH_TOKEN_BUDGET,
                          "search_token_bound": tools.token_bound_name(None),
                          "bash_timeout_seconds": tools.BASH_TIMEOUT_SECONDS,
                          "bash_output_bytes": tools.BASH_OUTPUT_BYTES,
                          "graph_query": {"render": "C1", "max_output_bytes": tools.GRAPH_QUERY_MAX_OUTPUT_BYTES,
                                          "defaults": tools.GRAPH_QUERY_DEFAULTS},
                          "submission": tools.submission_policy()},
                "model": self.model_record(),
                "code": harness.code_identity()}

    def model_record(self):
        """run.json's model: the alias and profile; a profile's chat-template override only when it has one."""
        record = {"alias": self.args.model, "max_context": 32768, "enable_thinking": False,
                  "temperature": self.config.temperature, "profile": self.served.name,
                  "hub_id": self.served.hub_id, "revision": self.served.revision}
        if self.served.chat_template is not None:
            record["chat_template_override"] = self.served.chat_template_override
        return record

    def stage_family(self, family, scratch, launch):
        """Local read-only copy of the clean checkout, whole-tree checked against its manifest entry."""
        entry = self.checkout_entries[family]
        source = Path(self.args.repos_root) / family
        if not source.is_dir():
            raise FileNotFoundError(f"checkout of {family} not found: {source}")
        staging = tools.stage_checkout(source, scratch / "repos" / launch / family)
        staged = Path(staging["destination"])
        tree = tools.checkout_tree(staged)
        comparison = tools.compare_tree(entry, tree)
        record = {"source": str(source), "used": str(staged),
                  "staging": {key: staging[key] for key in ("files", "bytes", "directories", "kept_symlinks",
                                                            "neutralized_symlinks")},
                  "skipped_entries": len(staging["skipped"]), "skipped_sample": staging["skipped"][:20],
                  "tree_sha256": tree["tree_sha256"], "tree_counts": tree["counts"],
                  "expected_tree_sha256": entry["tree_sha256"], "comparison": comparison}
        if not comparison["matches"]:
            raise RuntimeError(f"staged checkout of {family} differs from its clean-checkout manifest entry; refusing "
                               f"unexpected or missing files: {json.dumps(comparison, sort_keys=True)[:4000]}")
        return staged, record

    def setup(self):
        scratch = Path(self.args.scratch)
        ledger_module.require_local_filesystem(scratch)
        scratch.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.store.initialize(self.manifest(), self.cells)
        self.ledger = ledger_module.CellLedger(self.args.ledger, run_id=self.run_id,
                                               max_attempts=self.args.max_attempts)
        reconciled = self.ledger.reconcile(self.store)
        recovered = self.ledger.recover_expired(on_abandon=self.close_abandoned)
        overtaken = order_violations(self.ledger, self.selected, self.eligible)
        override = self.args.allow_out_of_order
        self.order_check = {"selected_segments": sorted({segment_key(cell) for cell in self.selected}),
                            "earlier_open_cells_outside_selection": overtaken, "override_reason": override}
        if overtaken and not override:
            raise OrderError("the selection would start before earlier segments finish (D10.8, D11.3: P seed 11, 23 "
                             "and 47 in turn, then S-VC, S-NAT, S-EDGE, T); open cells outside the selection: "
                             f"{overtaken}. Select those segments, wait for them, or pass --allow-out-of-order REASON "
                             "(recorded in the launch record)")
        launch = (f"{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}-"
                  f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}")
        self.toolboxes, checkouts = {}, {}
        for family in sorted({cell["family"] for cell in self.selected}):
            staged, record = self.stage_family(family, scratch, launch)
            sources, graph = verify_family_sources(family, staged, self.view, self.payloads, self.selected,
                                                   self.checkout_entries[family])
            record.update(sources)
            bash = tools.ReadOnlyBash(staged, scratch / "jail" / family)
            record.update(bash=bash.describe(), selftest=bash.selftest(), contract=bash.contract_probe())
            self.toolboxes[family] = tools.ToolBox(staged, list(self.view.package_dirs[family]), bash, graph=graph)
            checkouts[family] = record
        self.profiles, self.clients, self.counters, self.gpus, endpoints = [], [], [], [], []
        for url, gpu in self.args.endpoints:
            profile = ModelProfile(endpoint=url, model=self.args.model, revision=self.served.revision,
                                   max_context=32768, enable_thinking=False)
            endpoints.append({**check_endpoint(profile), "gpu_uuid": gpu})
            counter = tools.ServerTokenCounter(profile)
            if counter("probe") < 1:
                raise RuntimeError(f"{url} returned no tokens for a plain-text probe")
            self.profiles.append(profile)
            self.clients.append(LocalModel(profile, timeout=self.args.request_timeout))
            self.counters.append(counter)
            self.gpus.append(gpu)
        self.endpoint_health = [{"consecutive_infra": 0, "infra": 0, "completed": 0, "probe_failures": 0,
                                 "tripped": None} for _ in self.profiles]
        selection = {"blocks": sorted(self.args.blocks) if self.args.blocks else None,
                     "seeds": sorted(self.args.seeds) if self.args.seeds else None,
                     "families": sorted(self.args.families) if self.args.families else None,
                     "cells": len(self.selected),
                     "by_block": dict(sorted(collections.Counter(cell["block"] for cell in self.selected).items())),
                     "order": self.order_check}
        if getattr(self.args, "conditions", None):     # --conditions (A4) only: a launch without it is unchanged
            selection["conditions"] = sorted(self.args.conditions)
        self.launch = {"protocol": RUNNER_PROTOCOL, "run_id": self.run_id, "launch": launch,
                       "started_utc": utc_now(), "runner": self.identity, "owner_prefix": self.owner_prefix,
                       "ledger": {"path": str(self.ledger.path), "location": self.ledger.location},
                       "scratch": str(scratch), "dispatch": {"directory": str(self.view.directory),
                                                             **self.view.describe()},
                       "checkout_manifest": {"path": str(self.args.checkout_manifest),
                                             "sha256": self.checkouts_manifest_sha256},
                       "repos_root": str(self.args.repos_root),
                       "selection": selection, "endpoints": endpoints, "concurrency": self.args.concurrency,
                       "lease_seconds": self.lease_seconds, "max_attempts": self.args.max_attempts,
                       "endpoint_isolation": {"breaker_consecutive_infra": self.args.max_consecutive_infra,
                                              "health_interval_seconds": self.args.health_interval,
                                              "health_timeout_seconds": self.args.health_timeout,
                                              "health_failures": self.args.health_failures,
                                              "idle_poll_seconds": self.args.idle_poll_seconds,
                                              "rule": "a tripped endpoint's workers stop; the endpoint that last "
                                                      "failed a cell does not take its next attempt while another "
                                                      "endpoint is alive; the runner halts when every endpoint is "
                                                      "tripped"},
                       "checkouts": checkouts, "reconcile": reconciled, "recovered": recovered,
                       "git_sha": self.args.git_sha, "code": harness.code_identity(), "argv": sys.argv}
        (self.run_dir / "launches").mkdir(exist_ok=True)
        ledger_module.write_immutable_json(self.run_dir / "launches" / f"{launch}.json", self.launch)
        return self.launch

    def close_abandoned(self, cell_id, attempt_id, identity):
        self.store.close_attempt(cell_id, attempt_id, {
            "protocol": ledger_module.LEDGER_PROTOCOL, "status": "abandoned",
            "reason": "owner_verified_gone_without_result", "owner_identity": identity,
            "closed_unix": time.time()})

    # -- execution ------------------------------------------------------------

    def halt(self, reason, abort=False):
        with self.lock:
            if self.halt_reason is None:
                self.halt_reason = reason
        self.drain.set()
        if abort:
            self.abort.set()

    def trip(self, endpoint_index, reason):
        """Stop one endpoint's workers (they finish their episode); halt when no endpoint is left."""
        with self.lock:
            state = self.endpoint_health[endpoint_index]
            if state["tripped"] is not None:
                return
            state["tripped"] = {"reason": reason, "utc": utc_now()}
            alive = sum(1 for entry in self.endpoint_health if entry["tripped"] is None)
            self.stats["endpoints_tripped"] += 1
        print(json.dumps({"endpoint_tripped": self.profiles[endpoint_index].endpoint,
                          "gpu_uuid": self.gpus[endpoint_index], "reason": reason, "endpoints_alive": alive},
                         sort_keys=True), file=sys.stderr, flush=True)
        if not alive:
            self.halt(reason)

    def avoid_for(self, endpoint_index):
        """Cells whose last infrastructure failure was on this endpoint, while another endpoint is alive."""
        with self.lock:
            if not any(entry["tripped"] is None for index, entry in enumerate(self.endpoint_health)
                       if index != endpoint_index):
                return None
            return {cell_id for cell_id, index in self.last_failed_on.items() if index == endpoint_index} or None

    def health_loop(self):
        """GET /health of every live endpoint; ``health_failures`` failures in a row trip it (slow failures)."""
        if self.args.health_interval <= 0:
            return
        while not self.finished.wait(self.args.health_interval):
            for index, profile in enumerate(self.profiles):
                if self.endpoint_health[index]["tripped"] is not None:
                    continue
                try:
                    http_json(profile.endpoint, "/health", timeout=self.args.health_timeout)
                    healthy = True
                except Exception:  # timeouts, refused connections and HTTP errors all count
                    healthy = False
                with self.lock:
                    state = self.endpoint_health[index]
                    state["probe_failures"] = 0 if healthy else state["probe_failures"] + 1
                    failures = state["probe_failures"]
                if failures >= self.args.health_failures:
                    self.trip(index, f"health_probe_failed_{failures}_times")

    def heartbeat_loop(self):
        while not self.finished.wait(self.heartbeat_seconds):
            with self.lock:
                active = list(self.active.items())
            for cell_id, (owner, token) in active:
                try:
                    self.ledger.heartbeat(cell_id, owner, token, self.lease_seconds)
                except FenceError as error:
                    print(f"heartbeat lost lease {cell_id}: {error}", file=sys.stderr, flush=True)
            try:
                self.ledger.recover_expired(on_abandon=self.close_abandoned)
            except Exception as error:  # recovery problems never stop running episodes
                print(f"recover_expired failed: {error}", file=sys.stderr, flush=True)

    def worker(self, endpoint_index, worker_index):
        owner = f"{self.owner_prefix}/e{endpoint_index}w{worker_index}"
        while not self.drain.is_set():
            if self.endpoint_health[endpoint_index]["tripped"] is not None:
                return
            if self.args.limit is not None:
                with self.lock:
                    if self.reserved >= self.args.limit:
                        return
                    self.reserved += 1
            try:
                claim = self.ledger.claim(owner, self.identity, lease_seconds=self.lease_seconds,
                                          eligible=self.eligible, skip=self.avoid_for(endpoint_index))
            except Exception as error:
                print(f"claim failed: {error}", file=sys.stderr, flush=True)
                self.halt(f"claim_failed:{type(error).__name__}", abort=True)
                return
            if claim is None:
                if self.args.limit is not None:
                    with self.lock:
                        self.reserved -= 1
                # Stay while the selection is open: a cell another endpoint is running may be requeued, and
                # a healthy endpoint must take it rather than the endpoint that just failed it.
                try:
                    still_open = self.ledger.has_open_cells(self.eligible)
                except Exception as error:
                    print(f"ledger read failed: {error}", file=sys.stderr, flush=True)
                    self.halt(f"claim_failed:{type(error).__name__}", abort=True)
                    return
                if not still_open:
                    return
                self.drain.wait(self.args.idle_poll_seconds)
                continue
            with self.lock:
                self.active[claim["cell_id"]] = (owner, claim["token"])
            try:
                self.run_claim(claim, endpoint_index, owner)
            except Exception as error:
                print(f"runner failure on {claim['cell_id']}: {error}\n{traceback.format_exc()}", file=sys.stderr,
                      flush=True)
                self.halt(f"runner_exception:{type(error).__name__}", abort=True)
            finally:
                with self.lock:
                    self.active.pop(claim["cell_id"], None)

    def run_claim(self, claim, endpoint_index, owner):
        cell, cell_id, token, attempt_id = claim["spec"], claim["cell_id"], claim["token"], claim["attempt_id"]
        durable = self.store.completed_result(cell_id)
        if durable is not None:
            self.ledger.complete_from_durable(cell_id, owner, token, attempt_id=durable["attempt_id"],
                                              exit_status=durable["result"]["exit_status"],
                                              result_sha256=durable["result_sha256"])
            with self.lock:
                self.stats["completed_from_durable"] += 1
            return
        try:
            episode = episode_input(cell, self.view, self.payloads, run_id=self.run_id, attempt_id=attempt_id,
                                    token=token)
        except (KeyError, ValueError, OSError) as error:
            self.ledger.block(cell_id, f"episode_input:{type(error).__name__}", owner=owner, token=token)
            with self.lock:
                self.stats["blocked"] += 1
            return
        endpoint, gpu = self.profiles[endpoint_index].endpoint, self.gpus[endpoint_index]
        record = {"protocol": RUNNER_PROTOCOL, "run_id": self.run_id, "cell_id": cell_id, "attempt_id": attempt_id,
                  "ordinal": claim["ordinal"], "token": token, "owner": owner, "owner_identity": self.identity,
                  "launch": self.launch["launch"], "endpoint": endpoint, "gpu_uuid": gpu, "cell": cell,
                  "payload_sha256": cell["payload_sha256"], "dispatch_sha256": self.view.sha256,
                  "items_sha256": self.view.items_sha256, "started_unix": time.time(), "started_utc": utc_now(),
                  "code": harness.code_identity()}
        try:
            directory, start_sha = self.store.start_attempt(cell_id, attempt_id, record)
        except OSError:
            self.ledger.release_unstarted(cell_id, owner, token, "start_record_failed")
            self.halt("durable_start_record_failed", abort=True)
            return
        self.ledger.mark_started(cell_id, owner, token, start_sha, self.lease_seconds)
        with self.lock:
            self.stats["dispatched"] += 1
            self.stats[f"dispatched:{endpoint}"] += 1
        result, result_sha = harness.run_episode(
            episode, directory, client=self.clients[endpoint_index], toolbox=self.toolboxes[cell["family"]],
            config=self.config, payload_counter=self.counters[endpoint_index], stop_event=self.abort, gpu_uuid=gpu,
            run_context={"launch": self.launch["launch"], "endpoint": endpoint, "gpu_uuid": gpu,
                         "start_sha256": start_sha, "dispatch_sha256": self.view.sha256,
                         "items_sha256": self.view.items_sha256})
        infra = result.get("infra") or {}
        if result["exit_status"] == ledger_module.INFRA_EXIT and infra.get("kind") != "stop_requested":
            with self.lock:  # before finish() requeues the cell, so this endpoint's workers already skip it
                self.last_failed_on[cell_id] = endpoint_index
        try:
            status = self.ledger.finish(cell_id, owner, token, exit_status=result["exit_status"],
                                        result_sha256=result_sha, error_code=infra.get("kind"))
        except FenceError as error:
            print(f"finish fenced for {cell_id}: {error}", file=sys.stderr, flush=True)
            status = "fenced"
        with self.lock:
            self.stats["exit:" + result["exit_status"]] += 1
            self.table[cell_label(cell)][f"{endpoint}={gpu}:{result['exit_status']}"] += 1
            state = self.endpoint_health[endpoint_index]
            consecutive = 0
            if status == "infra_interrupted" and infra.get("kind") != "stop_requested":
                state["consecutive_infra"] += 1
                state["infra"] += 1
                consecutive = state["consecutive_infra"]
            elif status == "completed":
                state["consecutive_infra"] = 0
                state["completed"] += 1
                self.last_failed_on.pop(cell_id, None)
        if infra.get("kind") == "controller_exception":
            self.halt("controller_exception", abort=True)
        elif infra.get("kind") == "stop_requested":
            pass
        elif consecutive >= self.args.max_consecutive_infra:
            # A per-endpoint breaker: another endpoint's success never resets it, and tripping one endpoint
            # leaves the others running; the runner halts only when every endpoint has tripped.
            self.trip(endpoint_index, f"{consecutive}_consecutive_infra_interruptions")

    def run(self):
        def handle(signum, frame):
            if self.drain.is_set():
                self.halt(f"signal_{signum}_abort", abort=True)
            else:
                self.halt(f"signal_{signum}_drain")
        previous = {number: signal.signal(number, handle) for number in (signal.SIGTERM, signal.SIGINT)}
        heartbeat = threading.Thread(target=self.heartbeat_loop, name="heartbeat", daemon=True)
        heartbeat.start()
        health = threading.Thread(target=self.health_loop, name="health", daemon=True)
        health.start()
        workers = [threading.Thread(target=self.worker, args=(e, w), name=f"e{e}w{w}")
                   for e in range(len(self.clients)) for w in range(self.args.concurrency)]
        try:
            for thread in workers:
                thread.start()
            for thread in workers:
                while thread.is_alive():
                    thread.join(timeout=1.0)
        finally:
            for number, handler in previous.items():
                signal.signal(number, handler)
        self.finished.set()
        heartbeat.join(timeout=5)
        health.join(timeout=self.args.health_timeout + 5)
        snapshot = self.ledger.snapshot()
        (self.run_dir / "snapshots").mkdir(exist_ok=True)
        name = f"{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}-{os.getpid()}.json"
        snapshot_sha = ledger_module.write_immutable_json(self.run_dir / "snapshots" / name, snapshot)
        selected = collections.Counter()
        for row in snapshot["cells"]:
            if row["cell_id"] in self.eligible:
                selected[row["status"]] += 1
        endpoints = [{"endpoint": profile.endpoint, "gpu_uuid": gpu, **state}
                     for profile, gpu, state in zip(self.profiles, self.gpus, self.endpoint_health)]
        summary = {"protocol": RUNNER_PROTOCOL, "run_id": self.run_id, "launch": self.launch["launch"],
                   "halt_reason": self.halt_reason, "stats": dict(self.stats), "counts": snapshot["counts"],
                   "selected_counts": dict(sorted(selected.items())), "endpoints": endpoints,
                   "condition_by_endpoint": {label: dict(sorted(row.items())) for label, row in sorted(self.table.items())},
                   "snapshot": {"path": str(self.run_dir / "snapshots" / name), "sha256": snapshot_sha}}
        # An immutable end-of-launch record next to the launch record (published with launches/): halt reason,
        # endpoint trips and counts, so a drained endpoint is visible in the durable record, not only in logs.
        ledger_module.write_immutable_json(self.run_dir / "launches" / f"{self.launch['launch']}.end.json",
                                           dict(summary, ended_utc=utc_now()))
        return summary


# ---------------------------------------------------------------------------
# status: durable records only
# ---------------------------------------------------------------------------

def launched_conditions(run_dir):
    """The union of the --conditions selections the run's launch records name (None when no launch used it)."""
    found = set()
    for path in sorted((Path(run_dir) / "launches").glob("*.json")):
        if path.name.endswith(".end.json"):
            continue
        selection = json.loads(path.read_bytes()).get("selection") or {}
        if selection.get("conditions"):
            found.update(selection["conditions"])
    return found or None


def run_status(run_dir):
    """Exits per block x condition, condition x endpoint, primary-seed completeness and infra attempts."""
    store = ledger_module.DurableStore(run_dir)
    manifest = store.manifest()
    cells = store.load_cells()
    scanned = store.scan()
    cap = manifest.get("infra_attempt_cap", 3)
    by_label = collections.defaultdict(collections.Counter)
    by_endpoint = collections.defaultdict(collections.Counter)
    seeds = collections.defaultdict(collections.Counter)
    ledger_seeds = collections.defaultdict(collections.Counter)    # A3 runs only (protocols/EXPA_A3_LEDGER.md)
    conditions = launched_conditions(run_dir)                          # A4 runs only (--conditions)
    selected_seeds = collections.defaultdict(collections.Counter)
    states, infra_attempts, unknown = collections.Counter(), 0, sorted(set(scanned) - {c["cell_id"] for c in cells})
    for cell in cells:
        attempts = scanned.get(cell["cell_id"], [])
        infra = [a for a in attempts if a["state"] in ("infra_interrupted", "abandoned")]
        infra_attempts += len(infra)
        terminal = next((a for a in attempts if a["state"] == "completed"), None)
        if terminal is not None:
            state = "completed"
            exit_status = terminal["result"]["exit_status"]
            by_label[cell_label(cell)][exit_status] += 1
            by_endpoint[cell_label(cell)][f"{terminal['result'].get('endpoint')}={terminal['result'].get('gpu_uuid')}"] += 1
        elif len(infra) >= cap:
            state = "missing_infrastructure"
        elif any(a["state"] == "started" for a in attempts):
            state = "in_flight_or_unreconciled"
        else:
            state = "not_run"
        states[state] += 1
        if cell["block"] == "P":
            seeds[cell["seed"]][state] += 1
            if conditions is not None and cell["condition"] in conditions:
                selected_seeds[cell["seed"]][state] += 1
        elif cell["block"] == harness.LEDGER_BLOCK:
            ledger_seeds[cell["seed"]][state] += 1
        if state != "completed":
            by_label[cell_label(cell)][state] += 1
    # D11.3 as the analysis applies it (analysis.seed_completion): a seed is complete when every P cell is
    # terminal, a completed model exit or missing (infrastructure) after the attempt cap.
    complete_seeds = [seed for seed in harness.SEEDS
                      if seeds[seed] and set(seeds[seed]) <= {"completed", "missing_infrastructure"}]
    prefix = []
    for seed in harness.SEEDS:
        if seed not in complete_seeds:
            break
        prefix.append(seed)
    without_missing = [seed for seed in harness.SEEDS if seeds[seed] and set(seeds[seed]) == {"completed"}]
    status = {"protocol": RUNNER_PROTOCOL, "run_dir": str(run_dir), "run_id": manifest.get("run_id"),
              "dispatch_sha256": manifest.get("dispatch_sha256"), "items_sha256": manifest.get("items_sha256"),
              "design_cells": len(cells), "cell_states": dict(sorted(states.items())),
              "infra_attempts": infra_attempts, "infra_attempt_cap": cap, "attempt_dirs_outside_design": unknown,
              "p_seeds": {str(seed): dict(sorted(counts.items())) for seed, counts in sorted(seeds.items())},
              "p_complete_seed_prefix": prefix, "p_seeds_without_missing_cells": without_missing,
              "p_seed_rule": "a seed is complete when every P cell is completed or missing (infrastructure) after the "
                             "attempt cap (D11.3, as analysis.seed_completion); the primary seeds are the longest "
                             "complete prefix of 11, 23, 47",
              "exits_by_cell_label": {label: dict(sorted(row.items())) for label, row in sorted(by_label.items())},
              "condition_by_endpoint": {label: dict(sorted(row.items()))
                                        for label, row in sorted(by_endpoint.items())}}
    if ledger_seeds:   # an A3 run: its LEDGER cells per seed (absent from every other run's status)
        status["ledger_seeds"] = {str(seed): dict(sorted(counts.items()))
                                  for seed, counts in sorted(ledger_seeds.items())}
    if conditions is not None:   # a run launched with --conditions (A4): its P cells of those conditions per seed
        status["selected_conditions"] = sorted(conditions)
        status["p_seeds_selected_conditions"] = {str(seed): dict(sorted(counts.items()))
                                                 for seed, counts in sorted(selected_seeds.items())}
        status["p_seeds_selected_conditions_terminal"] = [
            seed for seed in harness.SEEDS
            if selected_seeds[seed] and set(selected_seeds[seed]) <= {"completed", "missing_infrastructure"}]
    return status


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# (max_model_calls, context_budget): the prereg default and protocols/EXPA_BUDGET16.md.
REGISTERED_BUDGETS = {(8, 16384), (16, 32768)}
# Registered episode designs: (max_model_calls, context_budget, final_turn_policy) -> (protocol, the model profiles it
# runs with; None = any). A final-turn policy other than none runs only with the prereg budget and the frozen 31B BF16
# (protocols/EXPA_FINALTURN.md).
REGISTERED_DESIGNS = {
    (8, 16384, "none"): ("protocols/EXPA_PREREG.md", None),
    (16, 32768, "none"): ("protocols/EXPA_BUDGET16.md", None),
    (8, 16384, "neutral"): (harness.FINAL_TURN_PROTOCOL, ("gemma4-31b",)),
    # Follow-up A8 (protocols/EXPA_A8_MODELS.md) also runs the deadline notice on block P C0/C1, seed 11, with its
    # three other model families.
    (8, 16384, "deadline"): (harness.FINAL_TURN_PROTOCOL, ("gemma4-31b",) + serve.A8_PROFILES),
    (8, 16384, "deadline_info"): (harness.A4_DIRECTIVE_PROTOCOL, ("gemma4-31b",)),
    (8, 16384, "submit_only"): (harness.A4_DIRECTIVE_PROTOCOL, ("gemma4-31b",)),
}
assert {(calls, budget) for calls, budget, policy in REGISTERED_DESIGNS if policy == "none"} == REGISTERED_BUDGETS
assert {policy for _, _, policy in REGISTERED_DESIGNS} == set(harness.FINAL_TURN_POLICIES)
# Registered A3 arms (protocols/EXPA_A3_LEDGER.md): (max_model_calls, context_budget, final_turn_policy,
# evidence_ledger) -> (protocol, model profiles). Every A3 arm gets the deadline notice; none is every earlier design.
REGISTERED_LEDGER_DESIGNS = {
    (8, 16384, "deadline", "off"): (harness.LEDGER_PROTOCOL, ("gemma4-31b",)),
    (8, 16384, "deadline", "on"): (harness.LEDGER_PROTOCOL, ("gemma4-31b",)),
}
assert {mode for *_, mode in REGISTERED_LEDGER_DESIGNS} | {harness.LEDGER_DEFAULT} == set(harness.LEDGER_MODES)
# Registered A5 arms (protocols/EXPA_A5_REVIEW.md): (max_model_calls, context_budget, final_turn_policy, evidence_ledger,
# submit_review) -> (protocol, model profiles). Both arms get the deadline notice and no last-call ledger, on block
# LEDGER of the A3 design.
REGISTERED_REVIEW_DESIGNS = {
    (8, 16384, "deadline", "none", "neutral"): (harness.SUBMIT_REVIEW_PROTOCOL, ("gemma4-31b",)),
    (8, 16384, "deadline", "none", "ledger"): (harness.SUBMIT_REVIEW_PROTOCOL, ("gemma4-31b",)),
}
assert ({mode for *_, mode in REGISTERED_REVIEW_DESIGNS} | {harness.SUBMIT_REVIEW_DEFAULT}
        == set(harness.SUBMIT_REVIEW_MODES))
# Registered A6 design (protocols/EXPA_A6_BUDGET16.md): (max_model_calls, context_budget, final_turn_policy,
# evidence_ledger) -> (protocol, model profiles). A3's off arm (deadline notice; the ledger built at the last call and
# never sent) at the registered 16-response budget (protocols/EXPA_BUDGET16.md), on block LEDGER of the A3 design.
A6_PROTOCOL = "protocols/EXPA_A6_BUDGET16.md"
REGISTERED_A6_DESIGNS = {
    (16, 32768, "deadline", "off"): (A6_PROTOCOL, ("gemma4-31b",)),
}
assert not set(REGISTERED_A6_DESIGNS) & set(REGISTERED_LEDGER_DESIGNS)
assert {(calls, budget) for calls, budget, *_ in REGISTERED_A6_DESIGNS} <= REGISTERED_BUDGETS
A6_HINT = ("; with --max-model-calls 16 --context-budget 32768 only as A6 (" + A6_PROTOCOL + "): --final-turn-policy "
           "deadline --evidence-ledger off --blocks " + harness.LEDGER_BLOCK + " and --model-profile gemma4-31b")
# Block P selections a final-turn policy may run (protocols/EXPA_FINALTURN.md: seed 11; protocols/EXPA_A4_DIRECTIVE.md:
# seed 11 or 23, optionally restricted with --conditions to C0 and C1). One seed per launch.
FINAL_TURN_SEEDS = (11, 23)
FINAL_TURN_CONDITIONS = ("C0", "C1")
# Registered A7 design (protocols/EXPA_A7_IMPACT.md): (max_model_calls, context_budget, final_turn_policy,
# evidence_ledger, submit_review, task_framing) -> (protocol, model profiles), and the one selection it runs: block P
# none x C0 and omit_node x C1, and S-VC vcomplete x C1, at seed 11 (2,250 cells on the 750 main items).
A7_PROTOCOL = harness.TASK_FRAMING_PROTOCOL
REGISTERED_FRAMING_DESIGNS = {
    (8, 16384, "none", "none", "none", "impact"): (A7_PROTOCOL, ("gemma4-31b",)),
}
assert ({framing for *_, framing in REGISTERED_FRAMING_DESIGNS} | {harness.TASK_FRAMING_DEFAULT}
        == set(harness.TASK_FRAMINGS))
A7_SELECTION = {"blocks": {"P", "S-VC"}, "seeds": {11}, "conditions": {"C0", "C1"}}

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dispatch", required=True,
                        help="dispatch directory: dispatch.jsonl, dispatch_manifest.json, payloads/, graph_facts/")
    parser.add_argument("--payloads", default=None, help="override the payload store directory")
    parser.add_argument("--graph-facts", default=None, help="override the graph facts directory")
    parser.add_argument("--blocks", default=None, help="comma-separated blocks to dispatch (default: all of "
                                                       + ",".join(harness.BLOCK_ORDER) + ")")
    parser.add_argument("--seeds", default=None, help="comma-separated seeds to dispatch (default: all)")
    parser.add_argument("--families", default=None, help="comma-separated families to dispatch (default: all)")
    parser.add_argument("--conditions", default=None,
                        help="comma-separated conditions to dispatch (default: all), e.g. C0,C1 (A4, "
                             + harness.A4_DIRECTIVE_PROTOCOL + ")")
    parser.add_argument("--dry-run", action="store_true", help="render every prompt, print a summary, write nothing")
    parser.add_argument("--show", type=int, default=0, help="dry run: include up to N example prompts")
    parser.add_argument("--tokenizer-dir", default=None,
                        help="dry run: count first-prompt tokens offline with this model directory's tokenizer files")
    parser.add_argument("--run-dir", help="durable run directory (immutable records; NFS allowed)")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--ledger", help="SQLite path on LOCAL disk, e.g. /dev/shm/ae-expA-run/ledger.sqlite3")
    parser.add_argument("--scratch", help="local scratch for jails and staged checkouts, e.g. /dev/shm/ae-expA-run")
    parser.add_argument("--checkout-manifest", default=None,
                        help="the clean checkout manifest (expA-clean-checkouts-v1), e.g. inputs/checkouts/manifest.json")
    parser.add_argument("--repos-root", default=None,
                        help="clean checkouts: family F is REPOS_ROOT/F, e.g. inputs/checkouts/repos")
    parser.add_argument("--endpoint", action="append", default=[],
                        help="loopback endpoint and its GPU UUID, URL=GPU-UUID; repeat per GPU")
    parser.add_argument("--model-profile", choices=sorted(serve.MODEL_PROFILES), default=serve.DEFAULT_MODEL_PROFILE,
                        help="served model (serve.MODEL_PROFILES): the alias every endpoint must serve and the "
                             "revision recorded per episode (default %(default)s)")
    parser.add_argument("--model", default=None,
                        help="served model alias; must be the profile's (default: the profile's)")
    parser.add_argument("--max-model-calls", type=int, default=harness.EpisodeConfig().max_model_calls,
                        help="responses per episode; the system prompt states it (default %(default)s, prereg)")
    parser.add_argument("--context-budget", type=int, default=harness.EpisodeConfig().context_budget,
                        help="episode context budget in tokens, at most the served 32768 (default %(default)s, prereg)")
    parser.add_argument("--final-turn-policy", choices=harness.FINAL_TURN_POLICIES, default=harness.FINAL_TURN_DEFAULT,
                        help="notice appended to the last tool result before each episode's last response "
                             "(protocols/EXPA_FINALTURN.md): none (default, the prereg harness), neutral or deadline; "
                             "deadline_info or submit_only (protocols/EXPA_A4_DIRECTIVE.md); registered only with the "
                             "default budget and model profile, and --blocks P with --seeds 11 or --seeds 23 "
                             "(optionally --conditions from C0,C1); deadline also runs A6 (" + A6_PROTOCOL + ") with "
                             "--evidence-ledger off, --max-model-calls 16 --context-budget 32768 and --blocks "
                             + harness.LEDGER_BLOCK)
    parser.add_argument("--evidence-ledger", choices=harness.LEDGER_MODES, default=harness.LEDGER_DEFAULT,
                        help="A3 arm (protocols/EXPA_A3_LEDGER.md): none (default, every earlier design), off or on; "
                             "off and on need --final-turn-policy deadline, the default budget and model profile, and "
                             "--blocks " + harness.LEDGER_BLOCK + "; off also runs A6 (" + A6_PROTOCOL + ") with "
                             "--max-model-calls 16 --context-budget 32768")
    parser.add_argument("--submit-review", choices=harness.SUBMIT_REVIEW_MODES, default=harness.SUBMIT_REVIEW_DEFAULT,
                        help="A5 arm (" + harness.SUBMIT_REVIEW_PROTOCOL + "): none (default, every earlier design), "
                             "neutral or ledger; neutral and ledger need --final-turn-policy deadline, --evidence-ledger "
                             "none, the default budget and model profile, and --blocks " + harness.LEDGER_BLOCK)
    parser.add_argument("--task-framing", choices=harness.TASK_FRAMINGS, default=harness.TASK_FRAMING_DEFAULT,
                        help="A7 (" + A7_PROTOCOL + "): default (every earlier design) or impact; impact needs the "
                             "default budget and model profile, no final-turn policy, evidence ledger or submit "
                             "review, and exactly --blocks P,S-VC --seeds 11 --conditions C0,C1")
    parser.add_argument("--concurrency", type=int, default=4, help="concurrent episodes per endpoint")
    parser.add_argument("--limit", type=int, default=None, help="dispatch at most N episodes, then drain")
    parser.add_argument("--max-attempts", type=int, default=3,
                        help="infra-interrupted attempts per cell before it is blocked (D11.3: 3)")
    parser.add_argument("--max-consecutive-infra", type=int, default=3,
                        help="consecutive infrastructure failures that trip ONE endpoint (its workers stop); the "
                             "runner halts when every endpoint has tripped")
    parser.add_argument("--health-interval", type=float, default=30.0,
                        help="seconds between /health probes of each endpoint (0 disables)")
    parser.add_argument("--health-timeout", type=float, default=10.0)
    parser.add_argument("--health-failures", type=int, default=3,
                        help="failed /health probes in a row that trip an endpoint")
    parser.add_argument("--idle-poll-seconds", type=float, default=2.0,
                        help="how often a worker with nothing to claim re-checks while the selection is open")
    parser.add_argument("--allow-out-of-order", default=None, metavar="REASON",
                        help="start although earlier D10.8 segments still have open cells; the reason is recorded")
    parser.add_argument("--lease-seconds", type=float, default=600.0)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--git-sha", default=None)
    args = parser.parse_args(argv)
    if args.allow_out_of_order is not None and not args.allow_out_of_order.strip():
        parser.error("--allow-out-of-order needs a reason")
    served = serve.served_model(args.model_profile)
    if args.model is None:
        args.model = served.alias
    elif args.model != served.alias:
        parser.error(f"--model {args.model} is not the alias {served.alias} of --model-profile {served.name}")
    try:
        args.episode_config = harness.EpisodeConfig(max_model_calls=args.max_model_calls,
                                                    context_budget=args.context_budget)
    except ValueError as error:
        parser.error(f"--max-model-calls/--context-budget: {error}")
    # Registered designs only: a typo would be frozen into run.json and end cells as ContextLimit (never retried).
    if (args.max_model_calls, args.context_budget) not in REGISTERED_BUDGETS:
        parser.error("--max-model-calls/--context-budget must be 8/16384 (prereg) or 16/32768 "
                     "(protocols/EXPA_BUDGET16.md)")
    budget16 = (args.max_model_calls, args.context_budget) == (16, 32768)
    a6 = (args.max_model_calls, args.context_budget, args.final_turn_policy,
          args.evidence_ledger) in REGISTERED_A6_DESIGNS
    if a6:   # A6 (protocols/EXPA_A6_BUDGET16.md): the deadline notice at the 16-response budget, ledger off only
        _, profiles = REGISTERED_A6_DESIGNS[(args.max_model_calls, args.context_budget, args.final_turn_policy,
                                             args.evidence_ledger)]
        if args.model_profile not in profiles:
            parser.error(f"--final-turn-policy deadline --evidence-ledger off with --max-model-calls 16 "
                         f"--context-budget 32768 ({A6_PROTOCOL}) is registered only with --model-profile gemma4-31b")
    else:
        _, profiles = REGISTERED_DESIGNS.get((args.max_model_calls, args.context_budget, args.final_turn_policy),
                                             (None, ()))
    if profiles is not None and args.model_profile not in profiles:
        protocol = harness.FINAL_TURN_PROTOCOLS.get(args.final_turn_policy, harness.FINAL_TURN_PROTOCOL)
        parser.error(f"--final-turn-policy {args.final_turn_policy} ({protocol}) is registered only "
                     "with the prereg budget (8 responses, 16384 tokens) and --model-profile gemma4-31b"
                     + (A6_HINT if budget16 and args.final_turn_policy == "deadline" else ""))
    if args.final_turn_policy != harness.FINAL_TURN_DEFAULT:
        args.episode_config = dataclasses.replace(args.episode_config, final_turn_policy=args.final_turn_policy)
    ledger = args.evidence_ledger != harness.LEDGER_DEFAULT
    if ledger:
        key = (args.max_model_calls, args.context_budget, args.final_turn_policy, args.evidence_ledger)
        _, profiles = REGISTERED_LEDGER_DESIGNS.get(key) or REGISTERED_A6_DESIGNS.get(key) or (None, ())
        if args.model_profile not in profiles:
            parser.error(f"--evidence-ledger {args.evidence_ledger} ({harness.LEDGER_PROTOCOL}) is registered only "
                         "with --final-turn-policy deadline, the prereg budget (8 responses, 16384 tokens) and "
                         "--model-profile gemma4-31b" + (A6_HINT if budget16 else ""))
        args.episode_config = dataclasses.replace(args.episode_config, evidence_ledger=args.evidence_ledger)
    review = args.submit_review != harness.SUBMIT_REVIEW_DEFAULT
    if review:
        _, profiles = REGISTERED_REVIEW_DESIGNS.get(
            (args.max_model_calls, args.context_budget, args.final_turn_policy, args.evidence_ledger,
             args.submit_review), (None, ()))
        if args.model_profile not in profiles:
            parser.error(f"--submit-review {args.submit_review} ({harness.SUBMIT_REVIEW_PROTOCOL}) is registered only "
                         "with --final-turn-policy deadline, --evidence-ledger none, the prereg budget (8 responses, "
                         "16384 tokens) and --model-profile gemma4-31b")
        args.episode_config = dataclasses.replace(args.episode_config, submit_review=args.submit_review)
    framing = args.task_framing != harness.TASK_FRAMING_DEFAULT
    if framing:
        _, profiles = REGISTERED_FRAMING_DESIGNS.get(
            (args.max_model_calls, args.context_budget, args.final_turn_policy, args.evidence_ledger,
             args.submit_review, args.task_framing), (None, ()))
        if args.model_profile not in profiles:
            parser.error(f"--task-framing {args.task_framing} ({A7_PROTOCOL}) is registered only with the prereg "
                         "budget (8 responses, 16384 tokens), --model-profile gemma4-31b and no --final-turn-policy, "
                         "--evidence-ledger or --submit-review")
        args.episode_config = dataclasses.replace(args.episode_config, task_framing=args.task_framing)
    a3_design = ledger or review          # block LEDGER of the A3 design
    args.design = harness.config_design(args.episode_config)
    try:
        args.blocks = parse_selection(args.blocks, harness.design_order(args.design), what="blocks")
        args.seeds = parse_selection(args.seeds, harness.SEEDS + (harness.LEDGER_SEEDS if a3_design else ()), int,
                                     "seeds")
        args.families = parse_selection(args.families, None, what="families")
        args.conditions = parse_selection(args.conditions, harness.CONDITIONS, what="conditions")
        args.endpoints = [parse_endpoint(text) for text in args.endpoint]
    except ValueError as error:
        parser.error(str(error))
    if ledger and (args.blocks != {harness.LEDGER_BLOCK}
                   or (args.seeds is not None and not args.seeds <= set(harness.LEDGER_SEEDS))):
        seeds = ", ".join(map(str, harness.LEDGER_SEEDS))
        protocol = A6_PROTOCOL if a6 else harness.LEDGER_PROTOCOL
        parser.error(f"--evidence-ledger {args.evidence_ledger} ({protocol}) runs block "
                     f"{harness.LEDGER_BLOCK} only, at seeds {seeds}: pass --blocks {harness.LEDGER_BLOCK} (and "
                     "--seeds from those)")
    if review and (args.blocks != {harness.LEDGER_BLOCK}
                   or (args.seeds is not None and not args.seeds <= set(harness.LEDGER_SEEDS))):
        seeds = ", ".join(map(str, harness.LEDGER_SEEDS))
        parser.error(f"--submit-review {args.submit_review} ({harness.SUBMIT_REVIEW_PROTOCOL}) runs block "
                     f"{harness.LEDGER_BLOCK} only, at seeds {seeds}: pass --blocks {harness.LEDGER_BLOCK} (and "
                     "--seeds from those)")
    if not a3_design and args.final_turn_policy != harness.FINAL_TURN_DEFAULT and (
            args.blocks != {"P"} or args.seeds not in [{seed} for seed in FINAL_TURN_SEEDS]
            or (args.conditions is not None and not args.conditions <= set(FINAL_TURN_CONDITIONS))):
        protocol = harness.FINAL_TURN_PROTOCOLS[args.final_turn_policy]
        parser.error(f"--final-turn-policy {args.final_turn_policy} ({protocol}) runs block P at seed 11 or seed 23 "
                     "only, one seed per launch: pass --blocks P --seeds 11 or --blocks P --seeds 23 (optionally "
                     "--conditions from C0,C1)")
    if framing and (args.blocks, args.seeds, args.conditions) != (A7_SELECTION["blocks"], A7_SELECTION["seeds"],
                                                                   A7_SELECTION["conditions"]):
        parser.error(f"--task-framing {args.task_framing} ({A7_PROTOCOL}) runs only the A7 selection: pass --blocks "
                     "P,S-VC --seeds 11 --conditions C0,C1 (block P none x C0 and omit_node x C1, S-VC vcomplete x C1)")
    if len({url for url, _ in args.endpoints}) != len(args.endpoints):
        parser.error("each --endpoint URL may appear once")
    if len({gpu for _, gpu in args.endpoints}) != len(args.endpoints):
        parser.error("each --endpoint GPU UUID may appear once")
    if args.dry_run:
        if bool(args.checkout_manifest) != bool(args.repos_root):
            parser.error("a dry run checks checkouts only with both --checkout-manifest and --repos-root")
    else:
        missing = [name for name in ("run_dir", "ledger", "scratch", "checkout_manifest", "repos_root")
                   if not getattr(args, name)]
        if missing or not args.endpoints:
            parser.error("live mode needs --run-dir, --ledger, --scratch, --checkout-manifest, --repos-root and at "
                         "least one --endpoint URL=GPU-UUID")
        if args.concurrency < 1 or args.max_attempts < 1:
            parser.error("--concurrency and --max-attempts must be positive")
        if args.max_consecutive_infra < 1 or args.health_failures < 1 or args.health_timeout <= 0 \
                or args.health_interval < 0 or args.idle_poll_seconds <= 0:
            parser.error("--max-consecutive-infra, --health-failures, --health-timeout and --idle-poll-seconds must be "
                         "positive (--health-interval 0 disables the probe)")
    return args


def checkout_manifest_main(argv):
    parser = argparse.ArgumentParser(prog="expA_run.py checkout-manifest",
                                     description="Write the clean checkout manifest (expA-clean-checkouts-v1) for "
                                                 "the dispatch families, from checkouts REPOS_ROOT/<family>.")
    parser.add_argument("--out", required=True, help="manifest file to create (never overwritten)")
    parser.add_argument("--repos-root", required=True, help="checkout of family F is REPOS_ROOT/F")
    parser.add_argument("--dispatch", required=True, help="dispatch directory (families and package dirs)")
    parser.add_argument("--commit", action="append", default=[], help="FAMILY=COMMIT, recorded in the entry")
    args = parser.parse_args(argv)
    commits = dict(text.split("=", 1) for text in args.commit)
    view = DispatchView(args.dispatch)
    print(json.dumps(write_checkouts_manifest(view.families, args.out, repos_root=args.repos_root,
                                              package_dirs=view.package_dirs, commits=commits),
                     indent=2, sort_keys=True))
    return 0


def appendix_main(argv):
    """Prereg Appendix A (D11.6): operational details generated from the frozen code."""
    parser = argparse.ArgumentParser(prog="expA_run.py appendix")
    parser.parse_args(argv)
    import inspect
    appendix = harness.operational_appendix()
    try:
        from absent_edges.probe import render
        lines, start = inspect.getsourcelines(render.c4_triggered)
        appendix["c4_trigger_predicate"] = {
            "module": "absent_edges/probe/render.py", "function": "c4_triggered", "first_line": start,
            "last_line": start + len(lines) - 1, "source": "".join(lines),
            "module_sha256": sha256_bytes(Path(render.__file__).read_bytes())}
    except Exception as error:  # the appendix still lists everything this lane owns
        appendix["c4_trigger_predicate"] = {"error": f"{type(error).__name__}: {error}"}
    appendix["runner"] = {"protocol": RUNNER_PROTOCOL, "dispatch_schema": DISPATCH_SCHEMA,
                          "dispatch_row_keys": sorted(DISPATCH_ROW_KEYS), "target_keys": sorted(TARGET_KEYS),
                          "membership_keys": sorted(MEMBERSHIP_KEYS),
                          "checkouts": {"schema": tools.CHECKOUTS_SCHEMA, "tree_protocol": tools.TREE_PROTOCOL,
                                        "formula": tools.TREE_FORMULA},
                          "queue": "one shared SQLite queue on local disk pulled by every endpoint worker; claims in "
                                   "(block-seed segment, position) order; an interrupted cell returns to the back of "
                                   "its segment",
                          "halts": "3 consecutive infrastructure interruptions on one endpoint (or 3 failed /health "
                                   "probes in a row) trip that endpoint only; the runner drains when every endpoint "
                                   "has tripped; a controller exception aborts it; SIGTERM/SIGINT drain, a second "
                                   "signal aborts",
                          "retry_placement": "the endpoint that last failed a cell does not take its next attempt "
                                             "while another endpoint is alive",
                          "order": "a launch may not start a later segment while an earlier one has open cells "
                                   "outside the selection (--allow-out-of-order REASON overrides, recorded)"}
    print(json.dumps(appendix, indent=2, sort_keys=True))
    return 0


def status_main(argv):
    parser = argparse.ArgumentParser(prog="expA_run.py status")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run_status(args.run_dir), indent=2, sort_keys=True))
    return 0


def reconcile_main(argv):
    """Close attempts whose owner is verified gone and rebuild the ledger; dispatches nothing.

    Run before the final publish (the runner must be stopped): an attempt whose runner was killed has a start
    record and an unclosed events.jsonl that the publisher never publishes; this writes its durable
    attempt_closed.json (status abandoned), after which the publisher can publish the attempt.
    """
    parser = argparse.ArgumentParser(prog="expA_run.py reconcile")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--ledger", required=True, help="the run's SQLite ledger on LOCAL disk")
    parser.add_argument("--run-id", default=None, help="default: run.json's run_id")
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args(argv)
    store = ledger_module.DurableStore(args.run_dir)
    run_id = args.run_id or store.manifest().get("run_id") or Path(args.run_dir).name
    ledger = ledger_module.CellLedger(args.ledger, run_id=run_id, max_attempts=args.max_attempts)
    report = ledger.reconcile(store)

    def close(cell_id, attempt_id, identity):
        store.close_attempt(cell_id, attempt_id, {
            "protocol": ledger_module.LEDGER_PROTOCOL, "status": "abandoned",
            "reason": "owner_verified_gone_without_result", "owner_identity": identity, "closed_unix": time.time()})
    recovered = ledger.recover_expired(on_abandon=close)
    still_open = sorted(attempt["attempt_id"] for attempts in store.scan().values() for attempt in attempts
                        if attempt["state"] == "started")
    print(json.dumps({"protocol": RUNNER_PROTOCOL, "run_id": run_id, "reconcile": report, "recovered": recovered,
                      "attempts_still_open": still_open,
                      "note": "an open attempt's owner may still be alive; the final publish lists it as waiting"},
                     indent=2, sort_keys=True))
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = {"checkout-manifest": checkout_manifest_main, "appendix": appendix_main, "status": status_main,
                "reconcile": reconcile_main}
    if argv[:1] and argv[0] in commands:
        return commands[argv[0]](argv[1:])
    args = parse_args(argv)
    view = DispatchView(args.dispatch, payloads_dir=args.payloads, graph_facts_dir=args.graph_facts)
    payloads = PayloadStore(view)
    cells = build_cells(view, payloads, design=args.design)
    selected = select_cells(cells, args.blocks, args.seeds, args.families, args.conditions)
    if not selected:
        raise SystemExit("the selection (--blocks/--seeds/--families/--conditions) contains no cells of the design")
    if args.dry_run:
        tokens = OfflineTokens(args.tokenizer_dir) if args.tokenizer_dir else None
        checkouts = (dry_run_checkouts(view, payloads, selected, args.checkout_manifest, args.repos_root)
                     if args.checkout_manifest else None)
        print(json.dumps(dry_run(view, payloads, cells, selected, show=args.show, tokens=tokens,
                                 checkouts=checkouts, config=args.episode_config), indent=2, sort_keys=True))
        return 0
    runner = Runner(args, view, payloads, cells, selected)
    try:
        launch = runner.setup()
    except OrderError as error:
        print(json.dumps({"protocol": RUNNER_PROTOCOL, "status": "refused", "reason": str(error),
                          "order": runner.order_check}, indent=2, sort_keys=True), flush=True)
        return 2
    print(json.dumps({"launch": launch["launch"], "cells": len(cells), "selected": len(selected),
                      "reconcile": launch["reconcile"], "recovered": launch["recovered"]}, sort_keys=True), flush=True)
    summary = runner.run()
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0 if summary["halt_reason"] is None else 3


if __name__ == "__main__":
    raise SystemExit(main())
