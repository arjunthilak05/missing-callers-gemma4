"""Exp A answers rendered through the production graph classes (docs/EXPA_DESIGN_V2_DELTA.md D5, D10.2-D10.4).

Answers come only from ``GraphFacts.query(X, relation="invokes", direction="incoming",
limit=30)`` and ``StructuralObservation.render("C1", max_output_bytes=12000)``. Cells:

* ``C1``: the plain render.
* ``C2`` / ``C3``: C1 plus a ``guidance`` value, the item's single wording pair (wording.py).
* ``C4a``: C1 plus the appended ``verification`` block: the shared lexical search
  (``tools.lexical_search``, UTF-8 byte bound, at most 2,000 bytes and 20 hits) for X.
  The 12,000-byte cap applies to the C1-C3 graph answer; C4a adds at most 2,000 bytes.
* ``C0`` (and the T arms) have no answer block, hence no payload.

C4 "triggered" is not a cell: it is reconstructed per item as C4a when the frozen
``GraphToolSet.graph_query`` trigger fires on the omit_node answer, and as C1 otherwise
(``reconstruct_c4_triggered``).

Snippet pre-sizing (D10.3). ``GraphFacts.query`` shares a 12,000-byte snippet budget
across endpoints and ``render`` halves the largest snippet when the payload is over
its cap; both depend on the whole answer. Each item therefore stores line-bounded
excerpts sized once: the target keeps at most 2,048 bytes and every caller one common
cap, the largest at which the ``vcomplete`` answer renders with no truncation of any
kind. ``omit_node`` and ``omit_edge`` are subsets, so their retained endpoints are
byte-identical. A node keeps the function's ``start_line``/``end_line`` and adds
``snippet_end_line``; an excerpt cut shows as ``snippet_end_line < end_line``. The
``natural`` variant reuses the same node bytes and may still be truncated (by the
30-match limit, the shared snippet budget or the render), which removes the item from
S-NAT only (D10.10).

Invariants (D5, checked at build and by ``verify``):
* omit_node vs vcomplete: the omit_node endpoint ids are the vcomplete ids minus C* in
  the same order; matches are vcomplete's minus C*->X in the same order; every retained
  endpoint and match is byte-identical; every other field is byte-identical except the
  fields in ``DIFFERS_BY_DESIGN``.
* omit_edge vs vcomplete: the same endpoint ids (C* retained, byte-identical) and
  vcomplete's matches minus C*->X.
* C2 vs C3: only the guidance value differs; both are C1 plus ``guidance``.
* C4a vs C1: only the appended ``verification`` block differs; the block is identical
  in every variant.
* C0 has no payload.

These invariants and the identity-visibility rule cover the graph payload only
(``graph_part``: the cell without its ``verification`` block); the C4a lexical block may
expose C* by design (D11.6).
"""
from __future__ import annotations

import copy
import hashlib
import json
import re

from ..data import canonical_json
from ..graphs import GraphFacts, _edge
from . import wording

RENDER_PROTOCOL = "expA-render-v2"
BACKEND = "locagent"  # GraphFacts.query checks the relation against RELATIONS[backend]
RELATION, DIRECTION, QUERY_LIMIT = "invokes", "incoming", 30
MAX_OUTPUT_BYTES = 12000
SNIPPET_MAX_BYTES = 2048
VARIANTS = ("vcomplete", "natural", "omit_node", "omit_edge")
OMISSION_VARIANTS = ("omit_node", "omit_edge")
STRICT_VARIANTS = ("vcomplete", "omit_node", "omit_edge")  # D3: no truncation allowed
CELL_CONDITIONS = ("C1", "C2", "C3", "C4a")
VERIFICATION_KIND = "lexical_candidates_not_proven_edges"
SEARCH_LIMIT, SEARCH_MAX_BYTES = 20, 2000
C4_TOKEN_BOUND = "utf8_bytes_upper_bound"
DIFFERS_BY_DESIGN = {
    "omit_node": ("facts_sha256", "returned_count", "query_result_count", "matches", "endpoints"),
    "omit_edge": ("facts_sha256", "returned_count", "query_result_count", "matches"),
}
INVARIANTS = {
    "omit_node_vs_vcomplete": {"absent": ["the match record C*->X", "the endpoint C*"],
                               "differs_by_design": list(DIFFERS_BY_DESIGN["omit_node"]),
                               "identical": "every other match and endpoint (bytes and order), the target endpoint, "
                                            "endpoint ids and order, and every other field"},
    "omit_edge_vs_vcomplete": {"absent": ["the match record C*->X"],
                               "differs_by_design": list(DIFFERS_BY_DESIGN["omit_edge"]),
                               "identical": "every endpoint including C* (bytes and order), every other match, "
                                            "and every other field"},
    "C2_vs_C3": "only the guidance value differs; both are C1 plus guidance (the item's wording pair)",
    "C4a_vs_C1": "only the appended verification block differs; the block is identical in every variant",
    "C0": "no payload (no answer block); the T arms likewise have none",
    "C4_triggered": "not a cell: C4a when the frozen GraphToolSet trigger fires on the omit_node answer, else C1",
    "scope": "the omission and visibility invariants cover the graph payload (C1-C3, and C4a without its "
             "verification block); the C4a lexical block may expose C* by design (D11.6)",
}


class RenderBudgetError(ValueError):
    """The vcomplete answer cannot be rendered untruncated, even with empty snippets."""


def payload_text(payload):
    """The exact serialization that payload sha256 values cover (canonical JSON, UTF-8)."""
    return canonical_json(payload).decode("utf-8")


def payload_sha256(payload):
    return None if payload is None else hashlib.sha256(canonical_json(payload)).hexdigest()


def excerpt(lines, start, end, cap):
    """Leading whole source lines of [start, end] within ``cap`` UTF-8 bytes.

    Returns (text, last_line). When not even the first line fits, the first line is cut
    at a UTF-8 boundary. An empty excerpt (cap 0) has last_line None.
    """
    if type(cap) is not int or cap < 0:
        raise ValueError("Snippet cap must be a non-negative integer")
    if not (type(start) is int and type(end) is int and 1 <= start <= end):
        raise ValueError("Invalid function line span")
    selected = lines[start - 1:end]
    taken, size = [], 0
    for line in selected:
        extra = len(line.encode("utf-8")) + (1 if taken else 0)
        if size + extra > cap:
            break
        taken.append(line)
        size += extra
    if taken:
        return "\n".join(taken), start + len(taken) - 1
    if not selected or cap == 0:
        return "", None
    return selected[0].encode("utf-8")[:cap].decode("utf-8", errors="ignore"), start


def function_node(scope, identity, cap):
    """GraphFacts node for one package function with a line-bounded snippet excerpt."""
    info = scope.table()[identity]
    relpath = info["path"]
    start, end = info["lineno"], info["end_lineno"]
    text, last = excerpt(scope.lines(relpath), start, end, cap)
    return {"id": identity, "kind": "function", "path": relpath, "start_line": start, "end_line": end,
            "snippet": text, "snippet_end_line": last}


def coverage_for(package_dirs):
    return {"completeness": "unknown",
            "scope": "Python functions and methods under " + ", ".join(package_dirs)}


def invokes_edge(caller, target):
    """Production edge record: id = sha256(canonical {source,target,relation})[:24]."""
    return _edge(caller, target, RELATION)


def facts_document(source_sha256, nodes, edges, coverage):
    return {"backend": BACKEND, "source_sha256": source_sha256,
            "nodes": sorted(nodes, key=lambda node: node["id"]),
            "edges": sorted(edges, key=lambda edge: edge["id"]),
            "error": None, "coverage": coverage}


def family_facts_document(scope, edges, indexed, *, source_sha256, coverage):
    """Natural LocAgent facts of one family for the T arms' live ``graph_query`` (D10.6).

    ``edges`` are LocAgent (caller, callee) identity pairs; nodes are every function
    LocAgent indexes plus every edge endpoint, with the same node format as the item
    answers (line-bounded excerpts of at most 2,048 bytes).
    """
    pairs = sorted(set(edges))
    identities = set(indexed or ()) | {identity for pair in pairs for identity in pair}
    table = scope.table()
    nodes = [function_node(scope, identity, SNIPPET_MAX_BYTES) for identity in sorted(identities) if identity in table]
    known = {node["id"] for node in nodes}
    records = [invokes_edge(caller, callee) for caller, callee in pairs if caller in known and callee in known]
    return facts_document(source_sha256, nodes, records, coverage)


def c4_triggered(data, relation=RELATION, direction=DIRECTION):
    """The GraphToolSet.graph_query fallback rule, applied to a query observation or C1 payload."""
    internal = {node["id"] for node in data["endpoints"] if node.get("path")}
    callers = sum(edge["source"] in internal for edge in data["matches"])
    return bool(data["error"] or data["truncated"] or not data["symbol_indexed"] or
                (relation == "invokes" and direction == "incoming" and callers <= 1))


def reconstruct_c4_triggered(cells, trigger_fires):
    """C4 triggered is never run (D5): it is C4a when the trigger fires, else C1."""
    return cells["C4a"] if trigger_fires else cells["C1"]


def shared_lexical_search(repo_root, package_dirs):
    """query -> verification block, through the live tool's shared search in probe/tools.py.

    No token counter: the block is bounded by UTF-8 bytes (at most 2,000) exactly as the
    live ``source_reference_search`` tool, so C4a and the tool agree (D10.4).
    """
    try:
        from . import tools
    except ImportError as error:
        raise RuntimeError("absent_edges.probe.tools is not available; C4a cannot be rendered") from error
    search = getattr(tools, "lexical_search", None)
    if search is None:
        raise RuntimeError("absent_edges.probe.tools.lexical_search is not available")

    def run(query):
        return search(repo_root, list(package_dirs), query, limit=SEARCH_LIMIT, token_budget=SEARCH_MAX_BYTES)
    return run


def verification_block(search, target, *, require_byte_bound=True):
    block = search(target)
    if not isinstance(block, dict) or block.get("kind") != VERIFICATION_KIND:
        raise ValueError("Verification block must be labelled " + VERIFICATION_KIND)
    hits = block.get("hits")
    if not isinstance(hits, list) or len(hits) > SEARCH_LIMIT:
        raise ValueError("Verification block must hold at most 20 hits")
    if len(canonical_json(block)) > SEARCH_MAX_BYTES:
        raise ValueError("Verification block exceeds 2,000 UTF-8 bytes")
    if require_byte_bound and (block.get("budget") or {}).get("token_bound") != C4_TOKEN_BOUND:
        raise ValueError("Verification block must use the UTF-8 byte bound of the live tool")
    return json.loads(canonical_json(block))


def _strip(payload, key):
    return {name: value for name, value in payload.items() if name != key}


def _truncation(data, c1):
    query = bool(data["truncated"])
    snippet = any(node.get("snippet_truncated") for node in data["endpoints"])
    rendered = bool(c1["payload_truncated"])
    return {"query_truncated": query, "snippet_truncated": snippet, "render_payload_truncated": rendered,
            "payload_truncated": query or snippet or rendered}


class ItemRenderer:
    """Builds the four variant documents of one target and renders every cell."""

    def __init__(self, scope, *, source_sha256, coverage):
        self.scope = scope
        self.source_sha256 = source_sha256
        self.coverage = coverage

    # ------------------------------------------------------------ documents
    def _nodes(self, target, callers, target_cap, caller_cap):
        identities = sorted({target, *callers})
        return {identity: function_node(self.scope, identity, target_cap if identity == target else caller_cap)
                for identity in identities}

    def _facts(self, target, callers, nodes, *, target_indexed=True):
        ids = sorted({target, *callers} if target_indexed else set(callers))
        edges = [invokes_edge(caller, target) for caller in callers]
        return GraphFacts(canonical_json(facts_document(self.source_sha256, [nodes[i] for i in ids],
                                                        edges, self.coverage)))

    def _vcomplete_fits(self, target, verified, target_cap, caller_cap):
        nodes = self._nodes(target, verified, target_cap, caller_cap)
        observation = self._facts(target, verified, nodes).query(
            target, relation=RELATION, direction=DIRECTION, limit=QUERY_LIMIT)
        data = observation.data
        rendered = observation.render("C1", max_output_bytes=MAX_OUTPUT_BYTES)
        if _truncation(data, rendered)["payload_truncated"]:
            return False
        return len(canonical_json(rendered)) - 1 + wording.longest_guidance_bytes() <= MAX_OUTPUT_BYTES

    def fit_caps(self, target, verified):
        """(target cap, caller cap): target at most 2,048 bytes, callers the largest common cap."""
        def largest(fits):
            if fits(SNIPPET_MAX_BYTES):
                return SNIPPET_MAX_BYTES
            low, high = 0, SNIPPET_MAX_BYTES  # fits(low) holds, fits(high) does not
            while high - low > 1:
                middle = (low + high) // 2
                if fits(middle):
                    low = middle
                else:
                    high = middle
            return low

        verified = sorted(verified)
        if self._vcomplete_fits(target, verified, SNIPPET_MAX_BYTES, 0):
            return SNIPPET_MAX_BYTES, largest(lambda cap: self._vcomplete_fits(target, verified, SNIPPET_MAX_BYTES, cap))
        if not self._vcomplete_fits(target, verified, 0, 0):
            raise RenderBudgetError("vcomplete answer exceeds the render cap even with empty snippets")
        return largest(lambda cap: self._vcomplete_fits(target, verified, cap, 0)), 0

    def documents(self, target, verified, natural_callers, omitted, caps, *, natural_target_indexed=True):
        verified = sorted(verified)
        if omitted not in verified or target in verified:
            raise ValueError("Omitted caller must be a verified caller other than the target")
        if natural_callers and not natural_target_indexed:
            raise ValueError("LocAgent edges into the target imply that LocAgent indexes it")
        target_cap, caller_cap = caps
        nodes = self._nodes(target, [*verified, *natural_callers], target_cap, caller_cap)
        vcomplete = self._facts(target, verified, nodes)
        kept = [caller for caller in verified if caller != omitted]
        return {"vcomplete": vcomplete,
                "natural": self._facts(target, sorted(set(natural_callers)), nodes,
                                       target_indexed=natural_target_indexed),
                "omit_node": self._facts(target, kept, nodes),
                "omit_edge": vcomplete.omit_edges([invokes_edge(omitted, target)["id"]])}

    # ------------------------------------------------------------- rendering
    def prepare(self, target, verified, natural_callers, omitted, verification, *, natural_target_indexed=True,
                caps=None):
        """Caps, documents, the C1 render and truncation info of every variant (no guidance yet)."""
        caps = self.fit_caps(target, verified) if caps is None else caps
        documents = self.documents(target, verified, natural_callers, omitted, caps,
                                   natural_target_indexed=natural_target_indexed)
        c1, info = {}, {}
        for variant in VARIANTS:
            observation = documents[variant].query(target, relation=RELATION, direction=DIRECTION, limit=QUERY_LIMIT)
            data = observation.data
            rendered = observation.render("C1", max_output_bytes=MAX_OUTPUT_BYTES)
            c1[variant] = rendered
            info[variant] = {"state": rendered["state"], "returned_count": rendered["returned_count"],
                             "query_result_count": rendered["query_result_count"], **_truncation(data, rendered),
                             "c4_trigger_fires": c4_triggered(data),
                             "endpoint_ids": [node["id"] for node in rendered["endpoints"]],
                             "match_sources": [edge["source"] for edge in rendered["matches"]],
                             "facts_sha256": hashlib.sha256(documents[variant].encoded).hexdigest()}
        return {"caps": {"target": caps[0], "caller": caps[1]}, "documents": documents, "c1": c1, "info": info,
                "verification": verification, "target": target, "omitted": omitted}

    @staticmethod
    def cells(prepared, wording_pair):
        """{variant: {C1, C2, C3, C4a}} for one wording pair (C4a None without a verification block)."""
        if wording_pair not in wording.PAIRS:
            raise ValueError(f"Unknown wording pair {wording_pair!r}")
        verification = prepared["verification"]
        result = {}
        for variant in VARIANTS:
            c1 = prepared["c1"][variant]
            result[variant] = {
                "C1": copy.deepcopy(c1),
                "C2": {**copy.deepcopy(c1), "guidance": wording.guidance(wording_pair, "C2")},
                "C3": {**copy.deepcopy(c1), "guidance": wording.guidance(wording_pair, "C3")},
                "C4a": None if verification is None else {**copy.deepcopy(c1),
                                                          "verification": copy.deepcopy(verification)}}
        return result


def variant_records(prepared, cells):
    """items.jsonl ``variants`` entries and the payload rows of one item's cells."""
    variants, payloads = {}, {}
    for variant in VARIANTS:
        hashes = {condition: payload_sha256(cells[variant][condition]) for condition in CELL_CONDITIONS}
        for condition in CELL_CONDITIONS:
            if cells[variant][condition] is not None:
                payloads[hashes[condition]] = cells[variant][condition]
        info = prepared["info"][variant]
        variants[variant] = {
            "facts_sha256": info["facts_sha256"], "endpoint_ids": info["endpoint_ids"],
            "payload_truncated": info["payload_truncated"], "cells": hashes,
            "cell_bytes": {condition: (None if cells[variant][condition] is None
                                       else len(canonical_json(cells[variant][condition])))
                           for condition in CELL_CONDITIONS},
            "state": info["state"], "returned_count": info["returned_count"],
            "query_result_count": info["query_result_count"], "query_truncated": info["query_truncated"],
            "snippet_truncated": info["snippet_truncated"],
            "render_payload_truncated": info["render_payload_truncated"],
            "c4_trigger_fires": info["c4_trigger_fires"], "match_sources": info["match_sources"]}
    return variants, payloads


# ------------------------------------------------------------------ visibility (D10.2, D10.13, D11.6)
def graph_part(payload):
    """The graph answer of a cell: the payload without the C4a ``verification`` block.

    The omission invariants and the identity-visibility rule cover the graph payload
    only; the C4a lexical block may expose C* by design (D11.6).
    """
    return None if payload is None else _strip(payload, "verification")


def omitted_identity_exposed(omitted, payloads):
    """True when any given payload contains the omitted identity C* (D10.2).

    An occurrence followed by ``.`` counts: a nested identity such as ``f.<locals>.g``
    literally contains C*'s identity ``f`` and shows its qualified name. An occurrence
    inside a longer name (``Cls.update`` within ``Cls.update_extend``) or a longer path
    (``a.py::f`` within ``sub/a.py::f``) does not count. ``payloads`` is a dict
    (cell -> payload or None) or an iterable of payloads; callers pass ``graph_part`` of
    each cell.
    """
    pattern = re.compile(r"(?<![\w./-])" + re.escape(omitted) + r"(?!\w)")
    values = payloads.values() if isinstance(payloads, dict) else payloads
    return any(payload is not None and pattern.search(payload_text(payload)) for payload in values)


def graph_identity_exposed(omitted, cells):
    """omitted_identity_exposed over the graph part of every cell of one variant."""
    values = cells.values() if isinstance(cells, dict) else cells
    return omitted_identity_exposed(omitted, [graph_part(payload) for payload in values])


def snippet_site_overlaps(payload, sites):
    """[(endpoint id, site)] where a delivered snippet covers a site span (same path, overlapping lines)."""
    overlaps = []
    for node in payload["endpoints"]:
        last = node.get("snippet_end_line")
        if last is None or not node.get("path"):
            continue
        for site in sites:
            if node["path"] == site["path"] and node["start_line"] <= site["end_lineno"] and site["lineno"] <= last:
                overlaps.append((node["id"], site))
    return overlaps


def visibility_text_matches(payload, needles):
    """Verbatim occurrences of the omitted caller's call-line texts inside delivered snippets."""
    total = 0
    for node in payload["endpoints"]:
        snippet = node.get("snippet") or ""
        for needle in needles:
            total += snippet.count(needle)
    return total


# A Sphinx cross-reference role (:func:`x`, :meth:`~C.x`, :py:func:`m.x`, :func:`title <m.x>`) or a
# backtick-quoted dotted name (`x`, ``x``, `x()`); the captured group is the referenced dotted name.
_DOC_REFERENCE = re.compile(r"(?::[A-Za-z]+(?::[A-Za-z]+)*:)?(`{1,2})~?\.?(?:[^`<\n]*<)?([A-Za-z_][\w.]*)(?:\(\))?>?\1")


def omitted_name_mentions(payload, omitted):
    """Pre-outcome covariate: how the delivered snippets name C* (review finding on docstring cross-references).

    ``docref``: Sphinx cross-reference roles or backtick-quoted names whose last dotted segment is C*'s
    simple name (for example ``Used by :func:`dft` and :func:`idft`.`` in the target's docstring);
    ``name``: whole-word occurrences of C*'s simple name anywhere in the snippets (code or prose). Both
    count over every delivered snippet of the payload (the graph part: C1-C3, and C4a without its
    lexical block). Neither rule (D10.2) nor covariate D10.13 sees these mentions; they are recorded,
    never used as a filter.
    """
    qualname = omitted.split("::", 1)[1] if "::" in omitted else omitted
    simple = qualname.rsplit(".", 1)[-1]
    word = re.compile(r"(?<!\w)" + re.escape(simple) + r"(?!\w)")
    docref = name = 0
    for node in payload["endpoints"]:
        snippet = node.get("snippet") or ""
        name += len(word.findall(snippet))
        docref += sum(1 for match in _DOC_REFERENCE.finditer(snippet)
                      if match.group(2).rsplit(".", 1)[-1] == simple)
    return {"docref": docref, "name": name}


# ------------------------------------------------------------------ invariants (D5)
def _endpoint_map(payload):
    return {node["id"]: canonical_json(node) for node in payload["endpoints"]}


def check_invariants(target, omitted, rendered, wording_pair, *, verification=None):
    """Raise AssertionError unless every D5 rendering invariant holds for one item.

    ``rendered`` is {variant: {"C1", "C2", "C3", "C4a"}} with payload dicts (C4a may be
    None in a build without the shared search).
    """
    if set(rendered) != set(VARIANTS):
        raise AssertionError(f"variants must be exactly {VARIANTS}")
    vcomplete = rendered["vcomplete"]["C1"]
    edge = invokes_edge(omitted, target)
    edge_id = edge["id"]
    complete_matches = [canonical_json(item) for item in vcomplete["matches"]]
    if canonical_json(edge) not in complete_matches:
        raise AssertionError("The omitted edge is not delivered in the vcomplete answer")
    complete_ids = [node["id"] for node in vcomplete["endpoints"]]
    if omitted not in complete_ids or target not in complete_ids:
        raise AssertionError("vcomplete must deliver the target and the omitted caller")
    if vcomplete["payload_truncated"] or vcomplete["truncated"] or any(
            node.get("snippet_truncated") for node in vcomplete["endpoints"]):
        raise AssertionError("vcomplete must be delivered without truncation")
    expected_matches = [canonical_json(item) for item in vcomplete["matches"] if item["id"] != edge_id]
    complete_nodes = _endpoint_map(vcomplete)
    for variant in OMISSION_VARIANTS:
        payload = rendered[variant]["C1"]
        if set(payload) != set(vcomplete):
            raise AssertionError(f"{variant}: payload fields differ from vcomplete")
        if [canonical_json(item) for item in payload["matches"]] != expected_matches:
            raise AssertionError(f"{variant}: remaining match records differ from vcomplete")
        ids = [node["id"] for node in payload["endpoints"]]
        expected_ids = [identity for identity in complete_ids if identity != omitted] \
            if variant == "omit_node" else complete_ids
        if ids != expected_ids:
            raise AssertionError(f"{variant}: endpoint ids are not the vcomplete ids"
                                 + (" minus the omitted caller" if variant == "omit_node" else "")
                                 + " in the same order")
        for node in payload["endpoints"]:
            if complete_nodes[node["id"]] != canonical_json(node):
                raise AssertionError(f"{variant}: endpoint {node['id']!r} differs from vcomplete")
        for key in sorted(set(vcomplete) - set(DIFFERS_BY_DESIGN[variant])):
            if canonical_json(payload[key]) != canonical_json(vcomplete[key]):
                raise AssertionError(f"{variant}: field {key!r} differs from vcomplete")
        if payload["returned_count"] != vcomplete["returned_count"] - 1 or \
                payload["query_result_count"] != vcomplete["query_result_count"] - 1:
            raise AssertionError(f"{variant}: returned_count and query_result_count must drop by exactly one")
    blocks = set()
    for variant in VARIANTS:
        cells = rendered[variant]
        if set(cells) != set(CELL_CONDITIONS):
            raise AssertionError(f"{variant}: cells must be exactly {CELL_CONDITIONS}")
        c1 = cells["C1"]
        if c1 is None or "guidance" in c1 or "verification" in c1:
            raise AssertionError(f"{variant}: C1 must be a plain answer")
        if len(canonical_json(c1)) > MAX_OUTPUT_BYTES:
            raise AssertionError(f"{variant}: C1 exceeds the render cap")
        for condition in ("C2", "C3"):
            cell = cells[condition]
            if cell is None or set(cell) != set(c1) | {"guidance"}:
                raise AssertionError(f"{variant}: {condition} must be C1 plus guidance")
            if canonical_json(_strip(cell, "guidance")) != canonical_json(c1):
                raise AssertionError(f"{variant}: {condition} facts differ from C1")
            if cell["guidance"] != wording.guidance(wording_pair, condition):
                raise AssertionError(f"{variant}: wrong {condition} guidance for pair {wording_pair}")
            if len(canonical_json(cell)) > MAX_OUTPUT_BYTES:
                raise AssertionError(f"{variant}: {condition} exceeds the render cap")
        if canonical_json(_strip(cells["C2"], "guidance")) != canonical_json(_strip(cells["C3"], "guidance")):
            raise AssertionError(f"{variant}: C2 and C3 differ outside the guidance value")
        c4a = cells["C4a"]
        if c4a is None:
            blocks.add(None)
            continue
        if set(c4a) != set(c1) | {"verification"} or canonical_json(_strip(c4a, "verification")) != canonical_json(c1):
            raise AssertionError(f"{variant}: C4a must be C1 plus the verification block only")
        block = canonical_json(c4a["verification"])
        if len(block) > SEARCH_MAX_BYTES or c4a["verification"].get("kind") != VERIFICATION_KIND:
            raise AssertionError(f"{variant}: the verification block breaks the C4 bound or label")
        blocks.add(block)
    if len(blocks) != 1:
        raise AssertionError("C4a must be built for every variant or for none, with one identical block")
    if verification is not None and blocks != {canonical_json(verification)}:
        raise AssertionError("C4a carries a verification block other than the item's")
