"""Actual pinned graph engines, immutable facts, and matched interface renderers.

AST parsing below inventories input coverage; it does not infer graph edges.
LocAgent/PyCG alone generate structural facts. C2 and C3 render the same query
object. All graph calls should use GraphBackend.query, which refreshes by source
content on every call. GraphFacts.query is for already frozen offline snapshots.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import uuid

from .data import canonical_json, sha256_file

ENGINE_PINS = {
    "locagent": {"repository": "https://github.com/gersteinlab/LocAgent",
                 "revision": "4935b557326c154bad8e8dcf3747cc8d32d1f387", "license": "Apache-2.0"},
    "pycg": {"repository": "https://github.com/vitsalis/PyCG",
             "revision": "8d5dc40837803beef1d8d379fbf2cdad6cd94641", "license": "Apache-2.0"},
}
RELATIONS = {"locagent": ("contains", "inherits", "invokes", "imports"), "pycg": ("invokes",)}
GRAPH_PROTOCOL = "absent-edges-graph-v1"
CONTROLLED_INTERVENTION_PROTOCOL = "absent-edges-final-observation-omission-v2"
GUIDANCE = {
    "C2": "Check your work before committing to a repair. Inspect relevant source, consider plausible alternative explanations, and verify the resulting behavior with an available check. Review assumptions carefully and gather additional evidence whenever the current evidence leaves uncertainty about the change.",
    "C3": "These are relationships found by this index, not an exhaustive list of possible dependencies. Missing results do not establish that no dependency exists. Inspect relevant endpoint source, search for additional references, and verify behavior before relying on an empty or sparse indexed result.",
}
GRAPH_TOOL = {"type": "function", "function": {"name": "graph_query",
    "description": "Query relationships indexed from current repository Python source. Returns indexed matches, endpoint source, and indexing status.",
    "parameters": {"type": "object", "properties": {
        "symbol": {"type": "string", "description": "Canonical path.py::qualified.name"},
        "relation": {"type": "string", "enum": ["invokes", "imports", "inherits", "contains"]},
        "direction": {"type": "string", "enum": ["incoming", "outgoing"]},
        "limit": {"type": "integer", "minimum": 1, "maximum": 200}},
        "required": ["symbol"], "additionalProperties": False}}}
SOURCE_REFERENCE_TOOL = {"type": "function", "function": {"name": "source_reference_search",
    "description": "Search current repository Python source for qualified names, names, and lexical aliases. Matches are candidate references, not proven call edges.",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"},
        "limit": {"type": "integer", "minimum": 1, "maximum": 20}},
        "required": ["query"], "additionalProperties": False}}}


def _regular_sources(root: Path):
    """POSIX anchored no-follow traversal; no pathname check/open race."""
    if os.name != "posix":
        raise RuntimeError("Secure graph source copying requires POSIX dirfd/O_NOFOLLOW")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    root_fd = os.open(root, directory_flags)
    budget = {"bytes": 0, "files": 0, "entries": 0}

    def visit(directory_fd, prefix):
        for name in sorted(os.listdir(directory_fd)):
            budget["entries"] += 1
            if budget["entries"] > 100000:
                raise ValueError("Source entry budget exceeded")
            info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            relative = prefix + name
            if stat.S_ISLNK(info.st_mode):
                raise ValueError("Source snapshot contains a symlink")
            if stat.S_ISDIR(info.st_mode):
                if name == ".git":
                    continue
                child = os.open(name, directory_flags, dir_fd=directory_fd)
                try:
                    yield from visit(child, relative + "/")
                finally:
                    os.close(child)
            elif stat.S_ISREG(info.st_mode):
                if not name.endswith(".py"):
                    continue
                descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                                     dir_fd=directory_fd)
                with os.fdopen(descriptor, "rb") as stream:
                    before = os.fstat(stream.fileno())
                    if not stat.S_ISREG(before.st_mode) or before.st_size > 16 * 1024**2:
                        raise ValueError("Invalid or oversized source file")
                    raw = stream.read(16 * 1024**2 + 1)
                    after = os.fstat(stream.fileno())
                    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
                        raise ValueError("Source changed while copying")
                budget["bytes"] += len(raw)
                budget["files"] += 1
                if len(raw) > 16 * 1024**2 or budget["bytes"] > 128 * 1024**2 or budget["files"] > 20000:
                    raise ValueError("Source inventory budget exceeded")
                yield relative, raw
            else:
                raise ValueError("Source snapshot contains a nonregular entry")
    try:
        yield from visit(root_fd, "")
    finally:
        os.close(root_fd)


def source_inventory(root: Path, *, parse=True, copy_to=None):
    """Inventory securely opened bytes. Production parsing occurs in the worker."""
    hashes, failures, exclusions = {}, [], []
    total = 0
    for relative, raw in _regular_sources(root):
        total += len(raw)
        hashes[relative] = hashlib.sha256(raw).hexdigest()
        if copy_to is not None:
            destination = copy_to / relative
            destination.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
            descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(raw)
        if parse:
            try:
                ast.parse(raw.decode("utf-8"), filename=relative)
            except (SyntaxError, UnicodeError, RecursionError) as error:
                failures.append({"path": relative, "reason": type(error).__name__})
        if ".git" in str(Path(relative).parent):
            exclusions.append(relative)
    manifest = {"files": hashes, "parse_failures": failures,
                "locagent_path_exclusions": exclusions, "python_bytes": total}
    manifest["source_sha256"] = hashlib.sha256(canonical_json(hashes)).hexdigest()
    return manifest


def _verify_vendor(root: Path, backend: str):
    git = ["git", "-c", f"safe.directory={root}", "-C", str(root)]
    head = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    if head != ENGINE_PINS[backend]["revision"]:
        raise ValueError("Graph engine is not at the audited revision")
    if subprocess.run([*git, "diff", "--quiet", "HEAD", "--"], check=False).returncode:
        raise ValueError("Graph engine has tracked modifications")


@dataclass(frozen=True)
class StructuralObservation:
    encoded: bytes

    @property
    def data(self):
        return json.loads(self.encoded)

    def render(self, condition: str, max_output_bytes=None):
        data = self.data
        condition = condition.split("_")[0]
        if condition not in ("C1", "C2", "C3", "C4"):
            raise ValueError("Structural condition must be C1 through C4")
        if "_intervention" in data:
            frozen_limit = data["_intervention"]["max_output_bytes"]
            if max_output_bytes is not None and max_output_bytes != frozen_limit:
                raise ValueError("Controlled observation output cap is frozen; rematerialize the pair for another cap")
            result = {key: value for key, value in data.items() if not key.startswith("_")}
            if condition in GUIDANCE:
                result["guidance"] = GUIDANCE[condition]
            if len(canonical_json(result)) > frozen_limit:
                raise ValueError("Controlled observation exceeds its frozen output cap")
            return result
        if "_render_reference" in data:
            raise ValueError("Legacy controlled observations must be rematerialized against final delivered evidence")
        max_output_bytes = 28000 if max_output_bytes is None else max_output_bytes
        if max_output_bytes < 2048:
            raise ValueError("Graph output budget must be at least 2048 bytes")
        basis = data
        # Budget evidence against the longer guidance in every condition.
        result = {key: basis[key] for key in ("query", "matches", "endpoints", "source_sha256",
                                             "facts_sha256", "returned_count", "truncated", "error",
                                             "state", "symbol_indexed", "coverage")}
        result["payload_truncated"] = False
        result["query_result_count"] = basis["returned_count"]
        result["coverage_totals"] = {key: len(value) for key, value in basis["coverage"].items()
                                     if isinstance(value, list)}
        reserve = max(len(canonical_json({"guidance": text})) for text in GUIDANCE.values()) + 256
        while len(canonical_json(result)) + reserve > max_output_bytes:
            result["payload_truncated"] = True
            snippets = [node for node in result["endpoints"] if node.get("snippet")]
            lists = [value for value in result["coverage"].values() if isinstance(value, list) and value]
            if snippets:
                node = max(snippets, key=lambda item: (len(item["snippet"]), item["id"]))
                node["snippet"] = node["snippet"][:len(node["snippet"]) // 2]
                node["snippet_truncated"] = True
            elif lists:
                largest = max(lists, key=lambda value: len(canonical_json(value)))
                del largest[len(largest) // 2:]
            elif result["matches"]:
                del result["matches"][len(result["matches"]) // 2:]
            elif result["endpoints"]:
                del result["endpoints"][len(result["endpoints"]) // 2:]
            else:
                # Huge user identifiers/metadata cannot become a false empty result.
                result.update(query={"error": "query_metadata_exceeds_output_budget"},
                              coverage={"completeness": "unknown"}, error="output_metadata_limit",
                              state="query_error")
                break
        result["returned_count"] = len(result["matches"])
        if condition in GUIDANCE:
            result["guidance"] = GUIDANCE[condition]
        if len(canonical_json(result)) > max_output_bytes:
            raise ValueError("Graph metadata exceeds bounded serialization")
        return result


@dataclass(frozen=True)
class GraphFacts:
    encoded: bytes

    @property
    def data(self):
        return json.loads(self.encoded)

    def query(self, symbol: str, relation="invokes", direction="outgoing", limit=30):
        if not isinstance(symbol, str) or len(symbol.encode("utf-8")) > 4096:
            raise ValueError("Symbol must be bounded text")
        document = self.data
        nodes = {node["id"]: node for node in document["nodes"]}
        indexed = symbol in nodes
        error = document.get("error")
        state = "query_error" if error else None
        if not isinstance(limit, int) or not 1 <= limit <= 200:
            raise ValueError("limit must be between 1 and 200")
        if direction not in ("incoming", "outgoing"):
            raise ValueError("direction must be incoming or outgoing")
        matches = []
        if relation not in RELATIONS[document["backend"]]:
            state, error = "unsupported_scope", "relation_not_supported_by_backend"
        elif not error:
            if not indexed:
                state = "symbol_not_indexed"
            else:
                endpoint = "source" if direction == "outgoing" else "target"
                matches = [edge for edge in document["edges"]
                           if edge[endpoint] == symbol and edge["relation"] == relation]
                state = "matches" if matches else "indexed_no_matches"
        truncated = len(matches) > limit
        if truncated:
            state = "truncated"
        matches = matches[:limit]
        endpoint_ids = {symbol} if indexed else set()
        endpoint_ids.update(edge[key] for edge in matches for key in ("source", "target"))
        # Selected-edge interventions retain exactly the endpoint source that the
        # same query exposed before omission. This metadata is never returned.
        endpoint_ids.update(document.get("retained_query_endpoints", {}).get(
            json.dumps([symbol, relation, direction]), []))
        endpoints, source_budget = [], 12000
        for identity in sorted(endpoint_ids):
            node = dict(nodes[identity])
            snippet = node.get("snippet")
            if snippet is not None:
                raw = snippet.encode("utf-8")
                maximum = min(source_budget, 2048)
                node["snippet_truncated"] = len(raw) > maximum
                node["snippet"] = raw[:maximum].decode("utf-8", errors="ignore")
                source_budget -= len(node["snippet"].encode("utf-8"))
            endpoints.append(node)
        observation = {"query": {"symbol": symbol, "relation": relation, "direction": direction},
                       "matches": matches, "endpoints": endpoints,
                       "source_sha256": document["source_sha256"],
                       "facts_sha256": hashlib.sha256(self.encoded).hexdigest(),
                       "returned_count": len(matches), "truncated": truncated, "error": error,
                       "state": state, "symbol_indexed": indexed, "coverage": document["coverage"]}
        return StructuralObservation(canonical_json(observation))

    def omit_edges(self, edge_ids):
        """Transform whole graph facts, not a fixed delivered observation.

        Queries may refill a result limit or expose different endpoints. Use
        query_intervention for controlled fixed-evidence comparisons instead.
        """
        document = self.data
        omitted = set(edge_ids)
        available = {edge["id"] for edge in document["edges"]}
        if not omitted <= available:
            raise ValueError("Cannot omit a nonexistent edge")
        retained = document.setdefault("retained_query_endpoints", {})
        for edge in document["edges"]:
            if edge["id"] in omitted:
                for endpoint, direction in (("source", "outgoing"), ("target", "incoming")):
                    key = json.dumps([edge[endpoint], edge["relation"], direction])
                    retained[key] = sorted(set(retained.get(key, [])) | {edge["source"], edge["target"]})
        document["edges"] = [edge for edge in document["edges"] if edge["id"] not in omitted]
        return GraphFacts(canonical_json(document))

    def query_intervention(self, symbol, omitted_edge_ids, relation="invokes", direction="outgoing", limit=30,
                           *, max_output_bytes=28000):
        """Remove only edges present AFTER final baseline serialization/truncation.

        The output cap is frozen for both observations. Rendering with another
        cap is rejected. Endpoint bytes/order and the original delivered pool
        stay fixed; removed matches never cause refill or larger source snippets.
        Natural query and GraphToolSet behavior are unchanged by this API.
        """
        if type(max_output_bytes) is not int or max_output_bytes < 2048:
            raise ValueError("Controlled output cap must be an integer of at least 2048 bytes")
        if isinstance(omitted_edge_ids, (str, bytes)):
            raise ValueError("Selected edge IDs must be a collection, not one string")
        requested = list(omitted_edge_ids)
        if not requested or any(not isinstance(identity, str) or not identity for identity in requested) or len(set(requested)) != len(requested):
            raise ValueError("Select at least one distinct edge ID for the controlled intervention")
        baseline = self.query(symbol, relation, direction, limit).render("C1", max_output_bytes)
        omitted = set(requested)
        delivered = [edge["id"] for edge in baseline["matches"]]
        if len(set(delivered)) != len(delivered) or not omitted <= set(delivered):
            raise ValueError("Every selected edge must be present in the final delivered baseline after truncation")
        changed = self.omit_edges(omitted)
        intervention = json.loads(canonical_json(baseline))
        intervention["matches"] = [edge for edge in baseline["matches"] if edge["id"] not in omitted]
        intervention["returned_count"] = len(intervention["matches"])
        intervention["query_result_count"] = baseline["query_result_count"] - len(omitted)
        intervention["facts_sha256"] = hashlib.sha256(changed.encoded).hexdigest()
        if not intervention["error"] and not intervention["truncated"] and intervention["symbol_indexed"]:
            intervention["state"] = "matches" if intervention["query_result_count"] else "indexed_no_matches"
        metadata = {"protocol": CONTROLLED_INTERVENTION_PROTOCOL, "max_output_bytes": max_output_bytes,
                    "selected_edge_ids": sorted(omitted),
                    "baseline_payload_sha256": hashlib.sha256(canonical_json(baseline)).hexdigest()}
        before = StructuralObservation(canonical_json({**baseline, "_intervention": {**metadata, "variant": "baseline"}}))
        after = StructuralObservation(canonical_json({**intervention, "_intervention": {**metadata, "variant": "omitted"}}))
        audit = audit_intervention(before, after)
        if not audit["passed"]:
            raise ValueError("Controlled intervention failed effect audit: " + ", ".join(audit["failures"]))
        return before, after


def audit_intervention(before: StructuralObservation, after: StructuralObservation) -> dict:
    """Deliverable-safe manipulation audit: hashes/counts only, no source snippets.

    Passing proves the specified delivered-observation transformation, never
    graph completeness, empirical model effects, or benchmark performance.
    """
    left, right = before.data, after.data
    left_meta, right_meta = left.get("_intervention", {}), right.get("_intervention", {})
    if not left_meta or not right_meta:
        raise ValueError("Both observations must come from the frozen controlled API")
    failures = []
    for key in ("protocol", "max_output_bytes", "selected_edge_ids", "baseline_payload_sha256"):
        if left_meta.get(key) != right_meta.get(key):
            failures.append("mismatched_"+key)
    if left_meta.get("protocol") != CONTROLLED_INTERVENTION_PROTOCOL:
        failures.append("unsupported_intervention_protocol")
    if (left_meta.get("variant"), right_meta.get("variant")) != ("baseline", "omitted"):
        failures.append("invalid_variant_order")
    requested = left_meta.get("selected_edge_ids", [])
    if not isinstance(requested, list) or any(not isinstance(identity, str) or not identity for identity in requested) or len(set(requested)) != len(requested):
        raise ValueError("Controlled audit requires a distinct edge-ID list")
    baseline, intervention = before.render("C1"), after.render("C1")
    selected = set(left_meta.get("selected_edge_ids", []))
    delivered = [edge["id"] for edge in baseline["matches"]]
    if not selected or not selected <= set(delivered):
        failures.append("selected_edge_not_final_delivered")
    if intervention["matches"] != [edge for edge in baseline["matches"] if edge["id"] not in selected]:
        failures.append("delivered_pool_refill_reorder_or_wrong_omission")
    if hashlib.sha256(canonical_json(baseline)).hexdigest() != left_meta["baseline_payload_sha256"]:
        failures.append("baseline_hash_mismatch")
    for key in ("query", "endpoints", "source_sha256", "coverage", "coverage_totals", "payload_truncated", "truncated", "error", "symbol_indexed"):
        if baseline[key] != intervention[key]:
            failures.append("changed_"+key)
    if intervention["returned_count"] != len(intervention["matches"]) or baseline["returned_count"] != len(delivered):
        failures.append("delivered_count_mismatch")
    if intervention["query_result_count"] != baseline["query_result_count"]-len(selected):
        failures.append("query_count_mismatch")
    source_hash = baseline["source_sha256"]
    if not isinstance(source_hash, str) or re.fullmatch(r"[a-f0-9]{64}", source_hash) is None:
        failures.append("invalid_source_hash")
        source_hash = None
    sizes = {condition: [len(canonical_json(before.render(condition))), len(canonical_json(after.render(condition)))]
             for condition in ("C1", "C2", "C3", "C4")}
    return {"protocol": CONTROLLED_INTERVENTION_PROTOCOL, "passed": not failures, "failures": failures,
        "selected_edge_count": len(selected), "delivered_matches_before": len(delivered),
        "delivered_matches_after": len(intervention["matches"]),
        "selected_edge_identity_sha256": sorted(hashlib.sha256(identity.encode()).hexdigest() for identity in selected),
        "baseline_payload_sha256": hashlib.sha256(canonical_json(baseline)).hexdigest(),
        "intervened_payload_sha256": hashlib.sha256(canonical_json(intervention)).hexdigest(),
        "endpoint_payload_sha256": hashlib.sha256(canonical_json(baseline["endpoints"])).hexdigest(),
        "source_sha256": source_hash, "max_output_bytes": left_meta["max_output_bytes"],
        "payload_bytes_by_condition": sizes,
        "claim_scope": "Exact final delivered edge omission with fixed endpoints, source and result pool; not a complete graph or an empirical model-effect result."}


class GraphBackend:
    def __init__(self, backend, vendor_root, python, cache_root, *, timeout=180, max_iter=-1):
        if backend not in ENGINE_PINS:
            raise ValueError("Unknown graph engine")
        self.backend = backend
        self.vendor_root = Path(vendor_root).resolve(strict=True)
        self.python = str(Path(python).absolute())  # preserve venv path
        self.cache_root = Path(cache_root).resolve()
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.timeout, self.max_iter = timeout, max_iter
        if timeout <= 0 or max_iter < -1:
            raise ValueError("Invalid graph execution budget")
        _verify_vendor(self.vendor_root, backend)

    def build(self, source_root):
        root = Path(source_root).absolute()
        run = self.cache_root / ("scan-" + uuid.uuid4().hex)
        run.mkdir(mode=0o700)
        source_copy = run / "source"
        source_copy.mkdir(mode=0o700)
        try:
            inventory = source_inventory(root, parse=False, copy_to=source_copy)
        except Exception:
            shutil.rmtree(run)
            raise
        identity = {"backend": self.backend, "engine": ENGINE_PINS[self.backend],
                    "inventory": inventory, "protocol": GRAPH_PROTOCOL, "max_iter": self.max_iter,
                    "wrapper_sha256": sha256_file(Path(__file__))}
        key = hashlib.sha256(canonical_json(identity)).hexdigest()
        final = self.cache_root / (key + ".json")
        if final.exists():
            shutil.rmtree(run)
            return GraphFacts(final.read_bytes())
        request = {**identity, "source_root": str(source_copy), "vendor_root": str(self.vendor_root)}
        (run / "request.json").write_bytes(canonical_json(request))
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONNOUSERSITE": "1",
               "PYTHONPATH": str(Path(__file__).resolve().parent.parent),
               "HOME": str(run), "MPLCONFIGDIR": str(run / "matplotlib"),
               "LANG": "C.UTF-8", "PYTHONHASHSEED": "0",
               "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
        if os.name == "nt" and "SystemRoot" in os.environ:
            env["SystemRoot"] = os.environ["SystemRoot"]
        with (run / "engine.log").open("wb") as log:
            try:
                process = subprocess.run([self.python, "-m", "absent_edges.graphs", "--worker", str(run)],
                                         stdout=log, stderr=subprocess.STDOUT, cwd=run, env=env,
                                         timeout=self.timeout, check=False)
                error = "engine_process_failed" if process.returncode else None
            except subprocess.TimeoutExpired:
                error = "engine_timeout"
        output = run / "result.json"
        if not output.exists():
            document = _empty_document(request, error or "engine_missing_output")
        else:
            document = json.loads(output.read_bytes())
        if source_inventory(root, parse=False)["source_sha256"] != inventory["source_sha256"]:
            document = _empty_document(request, "source_changed_during_indexing")
        encoded = canonical_json(document)
        # Cache successful construction only. Errors are preserved per attempt and retried.
        if not document.get("error"):
            temporary = run / "cache.json"
            temporary.write_bytes(encoded)
            os.replace(temporary, final)
        shutil.rmtree(source_copy)
        return GraphFacts(encoded)

    def query(self, source_root, symbol, relation="invokes", direction="outgoing", limit=30):
        try:
            return self.build(source_root).query(symbol, relation, direction, limit)
        except (OSError, ValueError, RuntimeError) as error:
            document = {"backend": self.backend, "source_sha256": None, "nodes": [], "edges": [],
                        "coverage": {"completeness": "unknown", "scope": "source scan unavailable"},
                        "error": "source_scan_error:" + type(error).__name__}
            bounded_symbol = symbol if isinstance(symbol, str) and len(symbol.encode()) <= 4096 else "<invalid-query>"
            return GraphFacts(canonical_json(document)).query(bounded_symbol, limit=30)

    def source_reference_search(self, source_root, query, limit=20):
        """Same bounded lexical operation is available to every condition."""
        if not isinstance(query, str) or not 1 <= len(query) <= 200 or not 1 <= limit <= 20:
            raise ValueError("Invalid bounded source-reference query")
        qualified = query.rsplit("::", 1)[-1]
        simple = qualified.rsplit(".", 1)[-1]
        if not re.fullmatch(r"[A-Za-z_]\w*", simple):
            raise ValueError("Source-reference query must end in an identifier")
        run = self.cache_root / ("search-" + uuid.uuid4().hex)
        run.mkdir(mode=0o700)
        source = run / "source"
        source.mkdir(mode=0o700)
        try:
            inventory = source_inventory(Path(source_root).absolute(), parse=False, copy_to=source)
            names = {qualified, simple}
            lines_by_file = {}
            for name in inventory["files"]:
                lines = (source / name).read_text(encoding="utf-8", errors="replace").splitlines()
                lines_by_file[name] = lines
                for line in lines:
                    if "import" in line and re.search(r"\b" + re.escape(simple) + r"\b", line):
                        names.update(re.findall(r"\bas\s+([A-Za-z_]\w*)", line))
            patterns = [re.compile(r"(?<!\w)" + re.escape(name) + r"(?!\w)") for name in sorted(names)]
            candidates = []
            for filename, lines in lines_by_file.items():
                for number, line in enumerate(lines, 1):
                    if any(pattern.search(line) for pattern in patterns):
                        score = 2 if qualified in line else 1
                        candidates.append((score, filename, number, line.strip()[:400]))
            candidates.sort(key=lambda row: (-row[0], row[1], row[2]))
            result = {"query": query, "source_sha256": inventory["source_sha256"],
                      "kind": "lexical_candidates_not_proven_edges", "hits": [],
                      "truncated": False, "error": None,
                      "budget": {"max_hits": limit, "max_utf8_bytes": 1900,
                                 "token_bound": "Conservative UTF-8 byte cap with reserved token overhead"}}
            for _, filename, number, line in candidates[:limit]:
                hit = {"path": filename, "line": number, "text": line,
                       "source_sha256": inventory["files"][filename]}
                proposed = {**result, "hits": result["hits"] + [hit]}
                if len(canonical_json(proposed)) > 1900:
                    break
                result["hits"].append(hit)
            result["truncated"] = len(result["hits"]) < len(candidates)
            if source_inventory(Path(source_root).absolute(), parse=False)["source_sha256"] != inventory["source_sha256"]:
                result.update(hits=[], error="source_changed_during_search")
            return result
        finally:
            shutil.rmtree(source)


class GraphToolSet:
    """Controller adapter. Supplier must export the current agent source each call.

    Tool schemas are common across C1-C4. Count auxiliary_tool_calls against the
    episode tool budget; C4 performs the same source-search operation automatically.
    """
    schemas = [GRAPH_TOOL, SOURCE_REFERENCE_TOOL]

    def __init__(self, backend, source_supplier, condition="C1_STANDARD", *, allow_fallback=None, max_output_bytes=32000):
        self.backend, self.source_supplier = backend, source_supplier
        self.condition = condition.split("_")[0]
        if self.condition not in ("C1", "C2", "C3", "C4"):
            raise ValueError("Graph conditions must be C1 through C4")
        self.automatic_queries = set()
        self.allow_fallback = allow_fallback
        self.max_output_bytes = max_output_bytes

    @property
    def handlers(self):
        return {"graph_query": self.graph_query, "source_reference_search": self.source_reference_search}

    def source_reference_search(self, query, limit=20):
        try:
            return self.backend.source_reference_search(self.source_supplier(), query, limit)
        except (OSError, ValueError, RuntimeError) as error:
            return {"kind": "lexical_candidates_not_proven_edges", "hits": [], "source_sha256": None,
                    "error": "source_scan_error:" + type(error).__name__}

    def graph_query(self, symbol, relation="invokes", direction="outgoing", limit=30):
        try:
            source = self.source_supplier()
        except (OSError, ValueError, RuntimeError) as error:
            failed = GraphFacts(canonical_json({"backend": self.backend.backend, "source_sha256": None,
                "nodes": [], "edges": [], "coverage": {"completeness": "unknown"},
                "error": "snapshot_export_error:" + type(error).__name__}))
            return failed.query("<source-unavailable>").render(self.condition, self.max_output_bytes - 2200)
        observation = self.backend.query(source, symbol, relation, direction, limit)
        result = observation.render(self.condition, max_output_bytes=self.max_output_bytes - 2200)
        if self.condition != "C4":
            return result
        data = observation.data
        internal = {node["id"] for node in data["endpoints"] if node.get("path")}
        callers = sum(edge["source"] in internal for edge in data["matches"])
        trigger = (data["error"] or data["truncated"] or not data["symbol_indexed"] or
                   (relation == "invokes" and direction == "incoming" and callers <= 1))
        if not trigger:
            return result
        key = (data["source_sha256"], symbol, relation, direction)
        if key in self.automatic_queries:
            result["verification"] = {"status": "identical_automatic_query_not_repeated"}
            return result
        if self.allow_fallback is None or not self.allow_fallback():
            result.update(verification={"status": "fallback_not_run_tool_budget_gate"}, auxiliary_tool_calls=0)
            return result
        self.automatic_queries.add(key)
        try:
            search = self.backend.source_reference_search(source, symbol, limit=20)
        except (OSError, ValueError, RuntimeError) as error:
            search = {"hits": [], "source_sha256": None, "error": "source_scan_error:" + type(error).__name__}
        if search["source_sha256"] != data["source_sha256"]:
            search.update(hits=[], error="source_revision_differs_from_graph")
        result.update(verification=search, auxiliary_tool_calls=1)
        return result


def _empty_document(request, error=None):
    inventory = request["inventory"]
    backend = request["backend"]
    return {"backend": backend, "engine": ENGINE_PINS[backend], "protocol": GRAPH_PROTOCOL,
            "source_sha256": inventory["source_sha256"], "source_hashes": inventory["files"],
            "wrapper_sha256": request["wrapper_sha256"], "nodes": [], "edges": [], "error": error,
            "coverage": {"completeness": "unknown", "scope": "repository Python source",
                         "parse_failures": inventory["parse_failures"],
                         "excluded_paths": inventory["locagent_path_exclusions"] if backend == "locagent" else [],
                         "limitations": (["Static approximate name resolution", "Constructor method nodes omitted by engine"]
                                         if backend == "locagent" else ["Static approximate analysis", "External dependencies not indexed as repository source"]),
                         "max_iter": request["max_iter"]}}


def _node(root, canonical, engine_id, kind, filename=None, first=None, last=None):
    result = {"id": canonical, "engine_id": engine_id, "kind": kind, "path": filename,
              "start_line": first, "end_line": last, "snippet": None, "source_sha256": None}
    if filename:
        path = root / filename
        if path.is_file() and path.resolve().is_relative_to(root):
            raw = path.read_bytes()
            result["source_sha256"] = hashlib.sha256(raw).hexdigest()
            if first and last and 1 <= first <= last:
                result["snippet"] = "\n".join(raw.decode("utf-8", errors="replace").splitlines()[first-1:last])
    return result


def _edge(source, target, relation):
    value = {"source": source, "target": target, "relation": relation}
    return {"id": hashlib.sha256(canonical_json(value)).hexdigest()[:24], **value}


def _locagent(request, document):
    root = Path(request["source_root"])
    module_path = Path(request["vendor_root"]) / "dependency_graph/build_graph.py"
    spec = importlib.util.spec_from_file_location("absent_edges_actual_locagent", module_path)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    graph = engine.build_graph(str(root), fuzzy_search=True, global_import=False)
    mapping = {}
    for engine_id, data in graph.nodes(data=True):
        normalized = engine_id.replace("\\", "/")
        if data.get("type") == "directory":
            canonical, filename = "directory::" + normalized, None
        elif ":" in normalized:
            filename, qualified = normalized.split(":", 1)
            canonical = filename + "::" + qualified
        else:
            filename, canonical = normalized, normalized + "::<module>"
        mapping[engine_id] = canonical
        document["nodes"].append(_node(root, canonical, engine_id, data.get("type"), filename,
                                        data.get("start_line"), data.get("end_line")))
    document["edges"] = [_edge(mapping[u], mapping[v], data["type"])
                         for u, v, data in graph.edges(data=True)]
    document["engine_module_sha256"] = sha256_file(module_path)


def _pycg(request, document):
    root, vendor = Path(request["source_root"]), Path(request["vendor_root"])
    sys.path.insert(0, str(vendor))
    from pycg.pycg import CallGraphGenerator
    from pycg.utils.constants import CALL_GRAPH_OP
    # Python 3.11 lazily imports this during invalidate_caches; preload before
    # PyCG installs its source loader so the engine does not intercept stdlib setup.
    import importlib.metadata
    document["compatibility_bootstrap"] = ["preload importlib.metadata before PyCG import hooks"]
    entries = [str(root / name) for name in request["inventory"]["files"]]
    generator = CallGraphGenerator(entries, str(root), request["max_iter"], CALL_GRAPH_OP)
    try:
        generator.analyze()
        mapping, known = {}, {}
        classes, functions = generator.output_classes(), set(generator.output_functions())
        for module_name, module in generator.output_internal_mods().items():
            filename = module.get("filename")
            if not filename or not (root / filename).resolve().is_relative_to(root):
                continue
            filename = Path(filename).as_posix()
            mapping[module_name] = filename + "::<module>"
            known[module_name] = _node(root, mapping[module_name], module_name, "file", filename)
            for name, method in module.get("methods", {}).items():
                qualified = name[len(module_name)+1:] if name.startswith(module_name + ".") else name
                canonical = filename + "::" + (qualified if name != module_name else "<module>")
                mapping[name] = canonical
                kind = ("file" if name == module_name else "class" if name in classes
                        else "function" if name in functions else "engine_symbol")
                known[name] = _node(root, canonical, name, kind, filename,
                                    method.get("first"), method.get("last"))
        output = generator.output()
        symbols = set(output)
        symbols.update(target for targets in output.values() for target in targets)
        for name in sorted(symbols):
            if name not in mapping:
                mapping[name] = "unmapped::" + name
                known[name] = _node(root, mapping[name], name, "unmapped")
            document["nodes"].append(known[name])
        document["edges"] = [_edge(mapping[source], mapping[target], "invokes")
                             for source, targets in output.items() for target in targets]
        document["engine_module_sha256"] = sha256_file(vendor / "pycg/pycg.py")
        document["coverage"]["identity_unmapped_count"] = sum(n["kind"] == "unmapped" for n in document["nodes"])
    finally:
        generator.tearDown()


def _worker(run):
    request = json.loads((run / "request.json").read_bytes())
    if os.name == "posix":
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
        resource.setrlimit(resource.RLIMIT_FSIZE, (256 * 1024**2, 256 * 1024**2))
    _verify_vendor(Path(request["vendor_root"]), request["backend"])
    request["inventory"] = source_inventory(Path(request["source_root"]), parse=True)
    document = _empty_document(request)
    try:
        if request["backend"] == "locagent":
            _locagent(request, document)
        else:
            _pycg(request, document)
        node_ids = [node["id"] for node in document["nodes"]]
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("Canonical symbol collision")
        document["nodes"].sort(key=lambda item: item["id"])
        document["edges"] = sorted({edge["id"]: edge for edge in document["edges"]}.values(),
                                   key=lambda item: item["id"])
    except Exception as error:
        document = _empty_document(request, "engine_error:" + type(error).__name__)
        import traceback
        traceback.print_exc()
    (run / "result.json").write_bytes(canonical_json(document))


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--worker":
        raise SystemExit("Graph worker requires --worker PATH")
    _worker(Path(sys.argv[2]))
