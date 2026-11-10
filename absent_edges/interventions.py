"""Fixed-evidence preparation only; no repair loop or live tool registration.

Freeze source retrieval on disk first. Every condition receives the exact same
source pool. Graph supplements use only complete endpoint text in that pool.
Private preparation/audit fields must never be appended to model messages.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import urllib.request

from .data import canonical_json
from .graphs import GUIDANCE, GraphFacts, audit_intervention
from .retrieval import FrozenSourcePool

FIXED_EVIDENCE_PROTOCOL = "absent-edges-fixed-source-pool-v1"
WORDING_PAIRS = {
    "A": dict(GUIDANCE),
    "B": {
        "C2": "Treat the current evidence carefully when deciding on a repair. Check relevant source, consider alternative explanations, and verify the proposed behavior with an available check. Avoid premature conclusions, review your assumptions, and look for further evidence when the proposed change remains uncertain.",
        "C3": "Treat these indexed relationships as partial evidence when deciding on a repair. An absent relationship does not show that no dependency exists. Check the supplied endpoint source, consider additional references, and verify behavior before drawing conclusions from an empty or sparse indexed answer.",
    },
}
WORDING_TOLERANCE = {"max_absolute_tokens": 6, "max_relative_difference": 0.10,
                     "status": "proposed_before_confirmation_freeze"}


def _sha(value):
    return hashlib.sha256(value if isinstance(value, bytes) else canonical_json(value)).hexdigest()


def _restricted_graph(pool, facts):
    source, graph = pool.data, facts.data
    if graph["source_sha256"] != source["source_sha256"]:
        raise ValueError("Graph and frozen source must describe the same source revision")
    if graph.get("error"):
        raise ValueError("Cannot prepare fixed evidence from a failed graph extraction")
    if graph["backend"] not in ("locagent", "pycg"):
        raise ValueError("Unsupported extractor")
    # Source-only identities are already frozen. No graph ranking, snippets,
    # degree/path metadata, inverse lists, descriptions or external nodes enter.
    symbols = {item["symbol_id"]: item for item in source["symbols"] if item["entity_text_complete"]}
    graph_ids = {item["id"] for item in graph["nodes"]}
    nodes = [{"id": identity, "source_candidate_id": item["candidate_id"],
              "source_text_sha256": item["source_text_sha256"]}
             for identity, item in symbols.items() if identity in graph_ids]
    allowed = {item["id"] for item in nodes}
    edges = {}
    for edge in graph["edges"]:
        if edge["relation"] == "invokes" and edge["source"] in allowed and edge["target"] in allowed:
            value = {key: edge[key] for key in ("source", "target", "relation")}
            identity = _sha(value)[:24]
            edges[identity] = {"id": identity, **value}
    return GraphFacts(canonical_json({"backend": graph["backend"], "source_sha256": source["source_sha256"],
        "nodes": nodes, "edges": [edges[key] for key in sorted(edges)], "error": None,
        "coverage": {"completeness": "unknown", "scope": "invokes relationships with both complete endpoint sources in frozen pool"}}))


@dataclass(frozen=True)
class PreparedExperiment:
    encoded: bytes

    @property
    def data(self):
        return json.loads(self.encoded)

    @property
    def sha256(self):
        return _sha(self.encoded)

    def payload(self, condition):
        if condition not in ("C0", "C1", "C2", "C3"):
            raise ValueError("Fixed-evidence preparation supports exactly C0-C3")
        return self.data["model_payloads"][condition]

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(self.encoded)
            stream.flush()
            os.fsync(stream.fileno())
        return {"path": str(path), "sha256": self.sha256}


def prepare_fixed_evidence(pool_path, pool_sha256, facts, *, anchor_symbol,
                           variant="natural", omitted_edge_ids=(), wording_pair="A",
                           query_limit=30, graph_output_bytes=12000):
    """Prepare a single static intervention point after exclusive source freeze.

    variant: natural; answer_only_omission (backing graph intact); or
    stored_graph_omission (edge absent from controller's restricted backing graph).
    Both omission variants expose one frozen forward answer. No repeated/inverse,
    degree or path queries are enabled; wiring an unrestricted tool invalidates
    this design. This function does not itself authorize any P/Q evaluation.
    """
    if variant not in ("natural", "answer_only_omission", "stored_graph_omission"):
        raise ValueError("Explicit, separately labeled intervention variant required")
    if wording_pair not in WORDING_PAIRS:
        raise ValueError("Unknown disclosed wording pair")
    if type(graph_output_bytes) is not int or not 4096 <= graph_output_bytes <= 64000:
        raise ValueError("Bounded graph supplement budget required")
    if isinstance(omitted_edge_ids, (str, bytes)):
        raise ValueError("Expected a collection of edge IDs")
    selected = tuple(omitted_edge_ids)
    if (variant == "natural") != (not selected):
        raise ValueError("Natural has no omission; an omission variant requires selected edges")
    pool = FrozenSourcePool.load(pool_path, pool_sha256)
    if pool.data["anchor_selection"]["symbol_id"] != anchor_symbol:
        raise ValueError("Anchor differs from source-only retrieval freeze; no graph-based retargeting")
    restricted = _restricted_graph(pool, facts)
    if anchor_symbol not in {item["id"] for item in restricted.data["nodes"]}:
        raise ValueError("Anchor must have complete source in the frozen pool and an indexed identity")
    # Extra reserved bytes cover both wording pairs. No per-condition source or
    # edge truncation occurs after materialization.
    inner_cap = graph_output_bytes-1024
    if selected:
        before, after = restricted.query_intervention(anchor_symbol, selected, direction="outgoing",
                                                       limit=query_limit, max_output_bytes=inner_cap)
        effect_audit = audit_intervention(before, after)
        graph_payload = after.render("C1")
        baseline_payload = before.render("C1")
    else:
        graph_payload = restricted.query(anchor_symbol, direction="outgoing", limit=query_limit).render("C1", inner_cap)
        baseline_payload, effect_audit = graph_payload, None
    backing_graph = restricted.data
    if variant == "stored_graph_omission":
        backing_graph["edges"] = [edge for edge in backing_graph["edges"] if edge["id"] not in set(selected)]
    source = pool.data
    # The complete pool is frozen before graph consultation, and is byte-identical
    # across all four arms, including endpoint source in omission variants.
    shared = {"public_issue": source["public_issue"], "source_pool": source["candidates"],
              "source_coverage": source["coverage"], "source_pool_sha256": pool.sha256}
    payloads = {"C0": json.loads(canonical_json(shared))}
    for condition in ("C1", "C2", "C3"):
        supplement = json.loads(canonical_json(graph_payload))
        if condition != "C1":
            supplement["guidance"] = WORDING_PAIRS[wording_pair][condition]
        if len(canonical_json(supplement)) > graph_output_bytes:
            raise ValueError("Wording exceeds frozen supplement cap; no unmatched truncation allowed")
        payloads[condition] = {**json.loads(canonical_json(shared)), "graph_supplement": supplement}
    for condition in ("C1", "C2", "C3"):
        if {key: payloads[condition][key] for key in shared} != shared:
            raise AssertionError("Fixed source pool changed across arms")
    c2, c3 = (dict(payloads[condition]["graph_supplement"]) for condition in ("C2", "C3"))
    c2.pop("guidance")
    c3.pop("guidance")
    if c2 != c3:
        raise AssertionError("C2/C3 differ in structural evidence")
    return PreparedExperiment(canonical_json({
        "protocol": FIXED_EVIDENCE_PROTOCOL, "scope": "fixed_evidence_preparation_not_an_executed_repair",
        "variant": variant, "wording_pair": wording_pair, "wording_sha256": _sha(WORDING_PAIRS[wording_pair]),
        "wording_token_tolerance": WORDING_TOLERANCE, "frozen_pool_sha256": pool.sha256,
        "source_sha256": source["source_sha256"], "query_sha256": source["query_sha256"],
        "anchor_selection": source["anchor_selection"],
        "retriever_sha256": source["retriever_sha256"], "preparation_source_sha256": _sha(Path(__file__).read_bytes()),
        "graph_source_sha256": _sha(Path(__file__).with_name("graphs.py").read_bytes()),
        "extractor": {key: facts.data.get(key) for key in ("backend", "engine", "engine_module_sha256", "wrapper_sha256")},
        "model_payloads": payloads, "model_payload_sha256": {key: _sha(value) for key, value in payloads.items()},
        "private_control": {"never_send_to_model": True, "backing_graph": backing_graph,
            "backing_graph_sha256": _sha(backing_graph), "baseline_delivered_payload_sha256": _sha(baseline_payload),
            "omitted_edge_identity_sha256": [_sha(identity.encode()) for identity in sorted(selected)],
            "effect_audit": effect_audit, "graph_output_bytes": graph_output_bytes,
            "query_limit": query_limit, "single_forward_answer_only": True,
            "repeat_inverse_degree_path_queries": "disabled_by_fixed_evidence_protocol",
            "endpoint_source_payload_sha256": _sha(shared["source_pool"])}}))


def _tokenize_exact(local_model, messages):
    request = {"model": local_model.profile.model, "messages": messages,
               "add_generation_prompt": True, "add_special_tokens": False,
               "chat_template_kwargs": {"enable_thinking": False}}
    encoded = canonical_json(request)
    req = urllib.request.Request(local_model.profile.endpoint.rstrip("/")+"/tokenize", encoded,
                                 {"Content-Type": "application/json"})
    with local_model.opener.open(req, timeout=local_model.timeout) as response:
        raw = response.read(2*1024**2+1)
    if len(raw) > 2*1024**2:
        raise ValueError("Tokenizer response exceeded budget")
    result = json.loads(raw)
    tokens = result.get("tokens")
    if (not isinstance(tokens, list) or any(type(token) is not int or token < 0 for token in tokens)
            or type(result.get("count")) is not int or result["count"] != len(tokens)
            or result.get("max_model_len") != local_model.profile.max_context):
        raise ValueError("Exact tokenizer token-array/count/profile evidence required")
    return {"count": len(tokens), "token_ids": tokens, "request_sha256": _sha(encoded),
            "response_sha256": _sha(raw), "max_model_len": result["max_model_len"]}


def audit_wording_tokens(local_model, prepared):
    """Read-only /tokenize calls; no completion, repair or generated trajectory."""
    result = {"scope": "exact_serving_chat_template_tokens_no_generation", "profile": {
        "model": local_model.profile.model, "revision": local_model.profile.revision,
        "max_context": local_model.profile.max_context}, "tolerance": WORDING_TOLERANCE, "pairs": {},
        "synthetic_prepared_artifact_sha256": prepared.sha256}
    empty = _tokenize_exact(local_model, [{"role": "user", "content": ""}])
    result["empty_chat_template"] = empty
    for name, pair in WORDING_PAIRS.items():
        counts = {condition: _tokenize_exact(local_model, [{"role": "user", "content": text}])
                  for condition, text in pair.items()}
        full = {}
        for condition, text in pair.items():
            payload = prepared.payload("C1")
            payload["graph_supplement"]["guidance"] = text
            full[condition] = _tokenize_exact(local_model, [{"role": "user", "content": canonical_json(payload).decode()}])
        difference = abs(counts["C2"]["count"]-counts["C3"]["count"])
        relative = difference/max(1, min(counts["C2"]["count"], counts["C3"]["count"])-empty["count"])
        full_difference = abs(full["C2"]["count"]-full["C3"]["count"])
        result["pairs"][name] = {"wording_sha256": _sha(pair), "conditions": counts,
            "absolute_difference": difference, "relative_difference": relative,
            "complete_evidence_payloads": full, "complete_evidence_absolute_difference": full_difference,
            "within_proposed_tolerance": difference <= WORDING_TOLERANCE["max_absolute_tokens"] and
                                         relative <= WORDING_TOLERANCE["max_relative_difference"] and
                                         full_difference <= WORDING_TOLERANCE["max_absolute_tokens"]}
    return result
