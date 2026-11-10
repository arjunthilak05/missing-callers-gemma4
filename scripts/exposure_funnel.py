"""Claim-1 exposure funnel from existing authenticated SAFE reports only (no new runs, no model calls).

    python scripts/exposure_funnel.py [--root REPO] [--out reports/exposure-funnel-v1]
                                      [--overwrite | --check] [--no-figure]

Resolves the evidence behind manuscript paragraphs M14 (D90) and M25 (Cap180) through the two
paper evidence maps, verifies every referenced file's sha256 against both maps and the hash
chains between the reports, and extracts the tool-exposure stage counts (offered, requested,
parsed, executed, visible, consumed) for graph_query, source_reference_search, bash and
submit_patch. Every number carries its provenance: file path, sha256 and JSON key path (or,
for a derived number, the key-path pattern and the derivation). A number that cannot be traced
to a key is recorded as untraceable, never filled in. Source files are opened read-only.

Writes OUT/FUNNEL.json, OUT/FUNNEL.md and OUT/funnel.png + OUT/funnel.svg (+ OUT/FIGURE.json).
Existing outputs are never overwritten without --overwrite; --check regenerates FUNNEL.json and
FUNNEL.md in memory and compares them byte for byte with the files on disk.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "exposure-funnel-v1"
EVIDENCE_MAPS = ("staged/private-paper-evidence-audit-v1/evidence-map.safe.json",
                 "staged/private-paper-progress-v14/evidence-map.safe.json")
MANUSCRIPTS = ("staged/private-paper-progress-v14/manuscript.txt", "staged/private-paper-progress-v10/manuscript.txt")
TAKEOVER = "TAKEOVER_20261101.md"
DIAGNOSTIC_SOURCE = "staged/cap180-completion-analysis-v7/history_diagnostic.py"
PARAGRAPHS = ("M14", "M25")
TOOLS = ("graph_query", "source_reference_search", "bash", "submit_patch")
CONDITIONS = ("C0_SOURCE", "C1_STANDARD", "C2_GENERIC", "C3_NONEXHAUSTIVE", "C4_VERIFY")
CAPS = ("B32000", "B8000")
ZERO_CHECKS = ("tool_parse_failures", "malformed_requested_tool_shapes_count", "raw_unknown_requested_tool_names_count",
               "unverified_history_transitions", "source_bound_early_return_observations")
# (stage id, diagnostic key or None when derived from turns, definition with history_diagnostic.py lines)
STAGES = (
    ("offered_episodes", "tool_availability_episodes",
     "episodes whose tool inventory offered the tool (summarize L239); every completed request must carry exactly "
     "that schema list (L106)"),
    ("offered_requests", None,
     "completed model requests whose request tool list contained the tool (turns with generated=true; L121-122)"),
    ("requested", "raw_requested_known_tool_names",
     "tool calls with this function name in the native response message.tool_calls (L125-135)"),
    ("parsed", "parsed_normalized_tool_calls", "actions returned by the frozen parser (L137-147)"),
    ("parsed_unexecuted", "parsed_but_unexecuted_tool_calls", "parsed actions without an executed tool_action (L213)"),
    ("executed", "executed_tool_actions", "tool_action events dispatched to the tool handler (L44-50)"),
    ("visible", "visible_tool_results_logged",
     "hash-verified model_visible_tool_result events, i.e. tool messages for the model (L58-65)"),
    ("consumed_next_request", "tool_results_verified_in_later_preflight_history",
     "visible results found in the next request history, including a final refused request (L176-178)"),
    ("consumed_completed_request", "tool_results_verified_in_later_completed_model_request",
     "visible results found in the next request that the model actually answered (L179)"),
)
STAGE_KEYS = {stage: key for stage, key, _ in STAGES}
ROLES = {
    "d90_descriptive": ("E1", lambda d: d.get("schema_version") == "pilot-descriptive-adapter-v2" and d.get("lane") == "D90"),
    "d90_diagnostic": ("E5", lambda d: d.get("schema_version") == "D90-native-context-diagnostic-v1"),
    "cap180_report": ("E30", lambda d: d.get("status") == "terminal_descriptive" and "native_summary" in d and "rows" in d),
    "cap180_projection": ("E30", lambda d: d.get("schema_version") == "cap180-local-saved-report-projection-v1"),
    "cap180_review": ("E31", lambda d: d.get("schema_version") == "Cap180-unused-graph-tools-source-and-safe-evidence-review-v1"),
    "cap180_safe_exposure": ("E31", lambda d: d.get("schema_version") == "Cap180-safe-tool-exposure-projection-v1"),
    "cap180_source_map": ("E31", lambda d: d.get("schema_version") == "Cap180-tool-exposure-source-map-v1"),
}
# Statements checked against traced numbers: (document, verbatim fragment, stated value, number ids).
CLAIMS = (
    ("M14", "55 ContextLimit exits", 55, ["D90.exits.ContextLimit"]),
    ("M14", "35 Submitted exits", 35, ["D90.exits.Submitted"]),
    ("M14", "1,883 model calls", 1883, ["D90.model_calls"]),
    ("M14", "1,822 bash actions", 1822, ["D90.bash.executed"]),
    ("M14", "61 submit_patch actions", 61, ["D90.submit_patch.executed"]),
    ("M14", "zero graph or source-reference requests at raw, parsed, executed, visible and consumed stages", 0,
     [f"D90.{t}.{s}" for t in ("graph_query", "source_reference_search")
      for s in ("requested", "parsed", "executed", "visible", "consumed_next_request", "consumed_completed_request")]),
    ("M14", "found zero parsing failures", 0, ["D90.tool_parse_failures"]),
    ("M14", "26 source-bound early-return observations", 26, ["D90.source_bound_early_return_observations"]),
    ("M14", "All 18 task/seed initial comparisons verified identical messages across C0–C4 and identical "
            "graph-tool schemas across C1–C4", 18,
     ["D90.initial_identity.groups", "D90.initial_identity.messages_equal_C0_C4", "D90.initial_identity.schemas_equal_C1_C4"]),
    ("M14", "C0 differs in graph-tool availability", 0, ["D90.by_condition.C0_SOURCE.graph_query.offered_episodes"]),
    ("M25", "Graph-query schemas were available in 144/180 episodes", 144, ["Cap180.all.graph_query.offered_episodes"]),
    ("M25", "144/180 episodes", 180, ["Cap180.all.episodes.planned"]),
    ("M25", "source-reference search in all 180", 180, ["Cap180.all.source_reference_search.offered_episodes"]),
    ("M25", "neither tool was requested or executed", 0,
     [f"Cap180.all.{t}.{s}" for t in ("graph_query", "source_reference_search") for s in ("requested", "executed")]),
    ("M25", "contain no malformed tool shapes, unknown names or parser failures", 0,
     ["Cap180.all.malformed_requested_tool_shapes_count", "Cap180.all.raw_unknown_requested_tool_names_count",
      "Cap180.all.tool_parse_failures"]),
    ("TAKEOVER", "270 episodes, 6,291 calls", 270, ["D90+Cap180.episodes"]),
    ("TAKEOVER", "270 episodes, 6,291 calls", 6291, ["D90+Cap180.model_calls"]),
    ("TAKEOVER", "was offered in all 270 episodes", 270, ["D90+Cap180.source_reference_search.offered_episodes"]),
    ("TAKEOVER", "graph tools in 216 (all but C0)", 216, ["D90+Cap180.graph_query.offered_episodes"]),
    ("TAKEOVER", "zero requests for either tool at the raw, parsed, executed, visible and consumed stages", 0,
     [f"D90+Cap180.{t}.{s}" for t in ("graph_query", "source_reference_search")
      for s in ("requested", "parsed", "executed", "visible", "consumed_next_request", "consumed_completed_request")]),
    ("TAKEOVER", "D90 55/90", 55, ["D90.exits.ContextLimit"]),
    ("TAKEOVER", "Cap180 60 (32 kB arm)", 60, ["Cap180.B32000.exits.ContextLimit"]),
    ("TAKEOVER", "vs 33 (8 kB arm)", 33, ["Cap180.B8000.exits.ContextLimit"]),
)
# How a stated value was read where the prose does not print the number itself.
READINGS = {
    ("M14", "zero graph or source-reference requests at raw, parsed, executed, visible and consumed stages"):
        "0 at every stage for both tools; 'consumed' read as both consumed_next_request and consumed_completed_request",
    ("M14", "All 18 task/seed initial comparisons verified identical messages across C0–C4 and identical "
            "graph-tool schemas across C1–C4"):
        "18 groups, of which 18 have equal C0-C4 initial messages and 18 have equal C1-C4 tool schemas",
    ("M14", "C0 differs in graph-tool availability"): "graph_query offered in 0 of the C0 episodes (C1-C4: 18 each)",
    ("M25", "144/180 episodes"): "denominator 180 = Cap180 planned episodes",
    ("M25", "neither tool was requested or executed"): "requested = 0 and executed = 0 for both tools",
    ("TAKEOVER", "zero requests for either tool at the raw, parsed, executed, visible and consumed stages"):
        "0 at every stage for both tools in D90 + Cap180",
}
QUALITATIVE = (
    ("M25", "Source review found matching schemas, parser names and handlers; tool choice remained optional.",
     "cap180_review", ("findings", 2)),
    ("M25", "C2/C3 warnings occur in graph responses, and C4 verification runs conditionally inside graph_query",
     "cap180_review", ("findings", 3)),
    ("M25", "This establishes offered-but-unused tools, not functioning dormant backends or why the model chose bash.",
     "cap180_review", ("findings", 5)),
)


class Untraceable(Exception):
    pass


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def keypath(path) -> str:
    out = ""
    for token in path:
        if token == "*":
            out += "[*]"
        elif isinstance(token, int):
            out += f"[{token}]"
        elif re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", token):
            out += ("." if out else "") + token
        else:
            out += f"[{json.dumps(token)}]"
    return out


def squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


class Source:
    def __init__(self, root: Path, rel: str, evidence_ids, expected_sha256):
        raw = (root / rel).read_bytes()
        self.rel, self.evidence_ids, self.expected = rel, sorted(evidence_ids), expected_sha256
        self.sha256, self.bytes = sha256(raw), len(raw)
        self.data = json.loads(raw) if rel.endswith(".json") else raw.decode("utf-8")

    def get(self, path):
        node = self.data
        for token in path:
            if isinstance(node, dict) and isinstance(token, str) and token in node:
                node = node[token]
            elif isinstance(node, list) and isinstance(token, int) and 0 <= token < len(node):
                node = node[token]
            else:
                raise Untraceable(f"{self.rel}: {keypath(path)} is absent (failed at {token!r})")
        return node

    def ref(self, path, derivation=None):
        out = {"evidence": "+".join(self.evidence_ids), "file": self.rel, "sha256": self.sha256, "key_path": keypath(path)}
        if derivation:
            out["derivation"] = derivation
        return out


class Numbers:
    """Number ledger: every value has a primary provenance and optional cross-checks."""

    def __init__(self):
        self.records = {}

    def _put(self, nid, record):
        if nid in self.records:
            raise ValueError(f"duplicate number id {nid}")
        self.records[nid] = {"id": nid, **record}

    def direct(self, nid, source, path, **meta):
        try:
            value = source.get(path)
        except Untraceable as error:
            self._put(nid, {"status": "untraceable", "value": None, "reason": str(error), "cross_checks": [], **meta})
            return None
        if not (type(value) is int or (isinstance(value, float) and value.is_integer())):
            self._put(nid, {"status": "untraceable", "value": None, "cross_checks": [], **meta,
                            "reason": f"{source.rel}: {keypath(path)} is not a count ({value!r})"})
            return None
        self._put(nid, {"status": "traced", "value": int(value), "source": source.ref(path), "cross_checks": [], **meta})
        return int(value)

    def derived(self, nid, value, refs, derivation, **meta):
        self._put(nid, {"status": "derived", "value": value, "derivation": derivation, "sources": refs,
                        "cross_checks": [], **meta})
        return value

    def check(self, nid, value, refs, label):
        record = self.records[nid]
        record["cross_checks"].append({"label": label, "value": value, "sources": refs,
                                       "agrees": record["value"] is not None and value == record["value"]})

    def value(self, nid):
        record = self.records.get(nid)
        return None if record is None else record["value"]


def load_maps(root):
    maps = []
    for rel in EVIDENCE_MAPS:
        raw = (root / rel).read_bytes()
        maps.append({"path": rel, "sha256": sha256(raw), "bytes": len(raw), "data": json.loads(raw)})
    return maps


def check_paragraphs(root, maps):
    paragraphs = {}
    texts = {rel: (root / rel).read_bytes() for rel in MANUSCRIPTS}
    for pid in PARAGRAPHS:
        entries = []
        for m in maps:
            found = [p for p in m["data"]["paragraph_map"] if p.get("id") == pid]
            if len(found) != 1:
                raise SystemExit(f"{m['path']}: paragraph {pid} must appear exactly once")
            entries.append(found[0])
        agree = all(e["references"] == entries[0]["references"] and e["text_sha256"] == entries[0]["text_sha256"]
                    and e["claim_ids"] == entries[0]["claim_ids"] for e in entries)
        if not agree:
            raise SystemExit(f"evidence maps disagree on paragraph {pid}")
        entry = entries[0]
        if entry["text_sha256_scope"] != "UTF-8 paragraph excluding label and following space":
            raise SystemExit(f"{pid}: unexpected text hash scope {entry['text_sha256_scope']!r}")
        manuscripts, text = [], None
        for rel, raw in texts.items():
            lines = [line for line in raw.decode("utf-8").split("\n") if line.startswith(f"[{pid}] ")]
            body = lines[0][len(pid) + 3:] if len(lines) == 1 else None
            ok = body is not None and sha256(body.encode("utf-8")) == entry["text_sha256"]
            manuscripts.append({"path": rel, "sha256": sha256(raw), "paragraph_found": body is not None,
                                "text_sha256_verified": ok})
            if ok and text is None:
                text = body
        if text is None:
            raise SystemExit(f"{pid}: no manuscript paragraph matches the evidence-map text_sha256")
        paragraphs[pid] = {"claim_ids": entry["claim_ids"], "evidence_type": entry["evidence_type"],
                           "references": entry["references"], "text_sha256": entry["text_sha256"],
                           "text_sha256_scope": entry["text_sha256_scope"], "maps_agree": True,
                           "manuscripts": manuscripts, "text": text}
    return paragraphs


def resolve_sources(root, maps, evidence_ids):
    listed = {}
    for eid in evidence_ids:
        per_map = [m["data"]["sources"].get(eid) for m in maps]
        if any(entry is None for entry in per_map) or any(entry != per_map[0] for entry in per_map):
            raise SystemExit(f"evidence maps disagree on (or lack) source {eid}")
        for entry in per_map[0]:
            listed.setdefault((entry["path"], entry["sha256"]), set()).add(eid)
    sources = []
    for (rel, expected), eids in sorted(listed.items()):
        source = Source(root, rel, eids, expected)
        if source.sha256 != expected:
            raise SystemExit(f"sha256 mismatch for {rel}: map {expected}, file {source.sha256}")
        sources.append(source)
    return sources


def assign_roles(sources):
    roles = {}
    for role, (eid, predicate) in ROLES.items():
        matches = [s for s in sources if eid in s.evidence_ids and isinstance(s.data, dict) and predicate(s.data)]
        if len(matches) != 1:
            raise SystemExit(f"role {role}: expected exactly one {eid} source, found {[s.rel for s in matches]}")
        roles[role] = matches[0]
    return roles


def offered_requests(diagnostics, tool):
    return sum(1 for d in diagnostics for t in d["turns"] if t["generated"] and tool in (t["request_tools_available"] or []))


def stage_from_rows(diagnostics, key, tool):
    if key == "tool_availability_episodes":
        return sum(tool in d["available_tools"] for d in diagnostics)
    return sum(d[key].get(tool, 0) for d in diagnostics)


def extract_d90(roles, N):
    diag, desc = roles["d90_diagnostic"], roles["d90_descriptive"]
    base = ("aggregates", "overall")
    rows = diag.get(("rows",))
    diagnostics = [r["diagnostic"] for r in rows]
    meta = {"block": "D90"}
    N.direct("D90.episodes.planned", diag, base + ("planned_rows",), **meta)
    N.check("D90.episodes.planned", len(rows), [diag.ref(("rows",), "len")], "rows in the diagnostic")
    N.check("D90.episodes.planned", desc.get(("planned_denominator",)), [desc.ref(("planned_denominator",))], "E1 planned denominator")
    N.direct("D90.episodes.fully_verified", diag, base + ("fully_verified_rows",), **meta)
    N.direct("D90.model_calls", diag, base + ("model_calls",), **meta)
    N.check("D90.model_calls", desc.get(("observed_totals_not_complete_costs", "model_calls")),
            [desc.ref(("observed_totals_not_complete_costs", "model_calls"))], "E1 authenticated cost totals")
    N.check("D90.model_calls", sum(d["model_calls"] for d in diagnostics),
            [diag.ref(("rows", "*", "diagnostic", "model_calls"), "sum")], "sum over rows")
    for name in sorted(diag.get(base + ("agent_exits",))):
        N.direct(f"D90.exits.{name}", diag, base + ("agent_exits", name), **meta)
        N.check(f"D90.exits.{name}", sum(d["agent_exit"] == name for d in diagnostics),
                [diag.ref(("rows", "*", "diagnostic", "agent_exit"), "count")], "count over rows")
    for name in ZERO_CHECKS:
        N.direct(f"D90.{name}", diag, base + (name,), **meta)
    for tool in TOOLS:
        for stage, key, _ in STAGES:
            nid = f"D90.{tool}.{stage}"
            if key is None:
                N.derived(nid, offered_requests(diagnostics, tool),
                          [diag.ref(("rows", "*", "diagnostic", "turns", "*", "request_tools_available"))],
                          "count of turns with generated == true whose request_tools_available contains the tool",
                          block="D90", tool=tool, stage=stage)
                continue
            N.direct(nid, diag, base + (key, tool), block="D90", tool=tool, stage=stage)
            N.check(nid, stage_from_rows(diagnostics, key, tool),
                    [diag.ref(("rows", "*", "diagnostic", key), "sum")], "sum over rows")
        executed_e1 = sum((r["metrics_evidence"]["diagnostics"]["executed_tool_actions"] or {}).get(tool, 0)
                          for r in desc.get(("rows",)))
        N.check(f"D90.{tool}.executed", executed_e1,
                [desc.ref(("rows", "*", "metrics_evidence", "diagnostics", "executed_tool_actions", tool), "sum")],
                "E1 per-row executed actions")
    for tool, field in (("graph_query", "graph_tool_model_requested_calls"),
                        ("source_reference_search", "source_reference_model_requested_calls")):
        N.check(f"D90.{tool}.requested", sum(d[field] for d in diagnostics),
                [diag.ref(("rows", "*", "diagnostic", field), "sum")], "per-row requested-call counter")
    for condition in CONDITIONS:
        cbase = ("aggregates", "by_condition", condition)
        crow = [d for r, d in zip(rows, diagnostics) if r["cell"]["condition"] == condition]
        N.direct(f"D90.by_condition.{condition}.episodes", diag, cbase + ("diagnostic_rows",), block="D90", condition=condition)
        N.direct(f"D90.by_condition.{condition}.model_calls", diag, cbase + ("model_calls",), block="D90", condition=condition)
        for tool in TOOLS:
            for stage in ("offered_episodes", "requested", "executed", "visible", "consumed_completed_request"):
                nid = f"D90.by_condition.{condition}.{tool}.{stage}"
                N.direct(nid, diag, cbase + (STAGE_KEYS[stage], tool), block="D90", condition=condition, tool=tool, stage=stage)
                N.check(nid, stage_from_rows(crow, STAGE_KEYS[stage], tool),
                        [diag.ref(("rows", "*", "diagnostic", STAGE_KEYS[stage]), f"sum where cell.condition == {condition}")],
                        "sum over the condition's rows")
            nid = f"D90.by_condition.{condition}.{tool}.offered_requests"
            N.derived(nid, offered_requests(crow, tool),
                      [diag.ref(("rows", "*", "diagnostic", "turns", "*", "request_tools_available"))],
                      f"generated turns offering the tool, rows with cell.condition == {condition}",
                      block="D90", condition=condition, tool=tool, stage="offered_requests")
    checks = diag.get(("aggregates", "initial_identity_checks"))
    ref = [diag.ref(("aggregates", "initial_identity_checks", "*"))]
    N.derived("D90.initial_identity.groups", len(checks), ref, "number of task/seed groups", block="D90")
    N.derived("D90.initial_identity.complete", sum(c["complete"] is True for c in checks), ref, "groups with complete == true", block="D90")
    N.derived("D90.initial_identity.messages_equal_C0_C4",
              sum(c["initial_messages_equal_across_five_conditions"] is True for c in checks), ref,
              "groups with initial_messages_equal_across_five_conditions == true", block="D90")
    N.derived("D90.initial_identity.schemas_equal_C1_C4", sum(c["C1_C4_tool_schemas_equal"] is True for c in checks), ref,
              "groups with C1_C4_tool_schemas_equal == true", block="D90")
    N.derived("D90.initial_identity.C0_schema_differs", sum(c["C0_tool_schema_differs_from_C1"] is True for c in checks), ref,
              "groups with C0_tool_schema_differs_from_C1 == true", block="D90")


def extract_cap180(roles, N):
    rep, proj, safe = roles["cap180_report"], roles["cap180_projection"], roles["cap180_safe_exposure"]
    rows = rep.get(("rows",))
    projection_index = {entry["condition"]: k for k, entry in enumerate(proj.get(("native", "by_condition_and_cap")))}
    for scope in ("all",) + CAPS:
        prefix = f"Cap180.{scope}"
        base = ("native_summary", "all") if scope == "all" else ("native_summary", "by_cap", scope)
        pbase = ("native", "all") if scope == "all" else ("native", "by_cap", scope)
        scoped = [r for r in rows if scope == "all" or r["cell"]["cap_policy"] == scope]
        hist = [r["native"]["history"] for r in scoped]
        meta = {"block": "Cap180", "scope": scope}
        N.direct(f"{prefix}.episodes.planned", rep, base + ("planned_rows",), **meta)
        N.check(f"{prefix}.episodes.planned", len(scoped), [rep.ref(("rows",), "len" if scope == "all" else f"count where cell.cap_policy == {scope}")], "rows")
        N.direct(f"{prefix}.episodes.model_exposure_verified", rep, base + ("model_exposure_verified_rows",), **meta)
        N.direct(f"{prefix}.model_calls", rep, base + ("history", "model_calls"), **meta)
        N.check(f"{prefix}.model_calls", rep.get(base + ("observed_totals_not_complete_costs", "model_calls")),
                [rep.ref(base + ("observed_totals_not_complete_costs", "model_calls"))], "authenticated cost totals")
        N.check(f"{prefix}.model_calls", sum(h["model_calls"] for h in hist),
                [rep.ref(("rows", "*", "native", "history", "model_calls"), "sum")], "sum over rows")
        for name in sorted(rep.get(base + ("history", "agent_exits"))):
            N.direct(f"{prefix}.exits.{name}", rep, base + ("history", "agent_exits", name), **meta)
            N.check(f"{prefix}.exits.{name}", sum(h["agent_exit"] == name for h in hist),
                    [rep.ref(("rows", "*", "native", "history", "agent_exit"), "count")], "count over rows")
        for name in ZERO_CHECKS:
            N.direct(f"{prefix}.{name}", rep, base + ("history", name), **meta)
        if scope != "all":
            caps = sorted({r["cell"]["budget"]["output_bytes"] for r in scoped})
            N.derived(f"{prefix}.bash_output_cap_bytes", caps[0] if len(caps) == 1 else None,
                      [rep.ref(("rows", "*", "cell", "budget", "output_bytes"), f"distinct where cell.cap_policy == {scope}")],
                      "the single output-byte budget of the scope's rows", **meta)
        for tool in TOOLS:
            for stage, key, _ in STAGES:
                nid = f"{prefix}.{tool}.{stage}"
                if key is None:
                    N.derived(nid, offered_requests(hist, tool),
                              [rep.ref(("rows", "*", "native", "history", "turns", "*", "request_tools_available"))],
                              "count of turns with generated == true whose request_tools_available contains the tool"
                              + ("" if scope == "all" else f", rows with cell.cap_policy == {scope}"),
                              block="Cap180", scope=scope, tool=tool, stage=stage)
                    continue
                N.direct(nid, rep, base + ("history", key, tool), block="Cap180", scope=scope, tool=tool, stage=stage)
                N.check(nid, stage_from_rows(hist, key, tool), [rep.ref(("rows", "*", "native", "history", key), "sum")],
                        "sum over rows")
                try:
                    N.check(nid, proj.get(pbase + ("history", key, tool)), [proj.ref(pbase + ("history", key, tool))],
                            "saved report projection")
                except Untraceable:
                    pass
                if scope == "all" and key in safe.get(("tool_counts",)):
                    N.check(nid, safe.get(("tool_counts", key, tool)), [safe.ref(("tool_counts", key, tool))],
                            "SAFE_EXPOSURE projection")
        if scope == "all":
            for tool, field in (("graph_query", "graph_tool_model_requested_calls"),
                                ("source_reference_search", "source_reference_model_requested_calls")):
                N.check(f"{prefix}.{tool}.requested", sum(h[field] for h in hist),
                        [rep.ref(("rows", "*", "native", "history", field), "sum")], "per-row requested-call counter")
    for condition in CONDITIONS:
        crow = [r for r in rows if r["cell"]["condition"] == condition]
        hist = [r["native"]["history"] for r in crow]
        where = f"where cell.condition == {condition}"
        meta = {"block": "Cap180", "condition": condition}
        nid = f"Cap180.by_condition.{condition}.episodes"
        N.derived(nid, len(crow), [rep.ref(("rows", "*", "cell", "condition"), "count " + where)], "rows " + where, **meta)
        N.check(nid, safe.get(("conditions", condition, "episodes")), [safe.ref(("conditions", condition, "episodes"))], "SAFE_EXPOSURE")
        nid = f"Cap180.by_condition.{condition}.model_calls"
        N.derived(nid, sum(h["model_calls"] for h in hist), [rep.ref(("rows", "*", "native", "history", "model_calls"), "sum " + where)],
                  "sum of model calls " + where, **meta)
        N.check(nid, safe.get(("conditions", condition, "completed_model_requests")),
                [safe.ref(("conditions", condition, "completed_model_requests"))], "SAFE_EXPOSURE")
        for tool in TOOLS:
            for stage in ("offered_episodes", "offered_requests", "requested", "executed", "visible", "consumed_completed_request"):
                nid = f"Cap180.by_condition.{condition}.{tool}.{stage}"
                if stage == "offered_requests":
                    N.derived(nid, offered_requests(hist, tool),
                              [rep.ref(("rows", "*", "native", "history", "turns", "*", "request_tools_available"))],
                              "generated turns offering the tool, rows " + where, tool=tool, stage=stage, **meta)
                    continue
                key = STAGE_KEYS[stage]
                N.derived(nid, stage_from_rows(hist, key, tool), [rep.ref(("rows", "*", "native", "history", key, tool), "sum " + where)],
                          f"sum of {key}.{tool} " + where, tool=tool, stage=stage, **meta)
                k = projection_index[condition]
                total, refs = 0, []
                try:
                    for cap in CAPS:
                        path = ("native", "by_condition_and_cap", k, "caps", cap, "history", key, tool)
                        total += proj.get(path)
                        refs.append(proj.ref(path))
                    N.check(nid, total, refs, "saved report projection (sum of caps)")
                except Untraceable:
                    pass
                safe_key = {("graph_query", "offered_episodes"): "graph_offered", ("graph_query", "requested"): "graph_requested",
                            ("source_reference_search", "offered_episodes"): "source_reference_offered",
                            ("source_reference_search", "requested"): "source_reference_requested"}.get((tool, stage))
                if safe_key:
                    N.check(nid, safe.get(("conditions", condition, safe_key)), [safe.ref(("conditions", condition, safe_key))],
                            "SAFE_EXPOSURE")
    for name in ("C1_C2_C3_C4_initial_messages_and_schema_equal_groups", "all_five_conditions_initial_messages_equal_groups",
                 "within_task_cap_seed_groups", "paired_caps_initial_messages_and_schema_equal", "paired_caps",
                 "completed_request_tool_inventory_mismatches", "completed_model_requests", "request_hash_bound_preflights"):
        N.direct(f"Cap180.identity.{name}", safe, (name,), block="Cap180")
    N.check("Cap180.identity.completed_model_requests", N.value("Cap180.all.model_calls"),
            [rep.ref(("native_summary", "all", "history", "model_calls"))], "report model calls")


def combine(N):
    pairs = [("episodes", "D90.episodes.planned", "Cap180.all.episodes.planned"),
             ("model_calls", "D90.model_calls", "Cap180.all.model_calls")]
    pairs += [(f"{tool}.{stage}", f"D90.{tool}.{stage}", f"Cap180.all.{tool}.{stage}") for tool in TOOLS for stage, _, _ in STAGES]
    for name, left, right in pairs:
        a, b = N.value(left), N.value(right)
        tool, _, stage = name.partition(".")
        meta = {"block": "D90+Cap180"} | ({"tool": tool, "stage": stage} if stage else {})
        N.derived(f"D90+Cap180.{name}", None if a is None or b is None else a + b, [{"number": left}, {"number": right}],
                  f"{left} + {right}", **meta)


def chain_checks(roles, root):
    diag, desc = roles["d90_diagnostic"], roles["d90_descriptive"]
    rep, proj = roles["cap180_report"], roles["cap180_projection"]
    review, safe, smap = roles["cap180_review"], roles["cap180_safe_exposure"], roles["cap180_source_map"]
    diagnostic_source = (root / DIAGNOSTIC_SOURCE).read_bytes()
    listed = [s for s in smap.get(("sources",)) if s["path"] == DIAGNOSTIC_SOURCE]
    checks = [
        ("E5 accepted_report.sha256 is the E1 file", diag.get(("accepted_report", "sha256")), desc.sha256,
         diag.ref(("accepted_report", "sha256"))),
        ("SAFE_EXPOSURE source.sha256 is the Cap180 report", safe.get(("source", "sha256")), rep.sha256, safe.ref(("source", "sha256"))),
        ("PROJECTION report.sha256 is the Cap180 report", proj.get(("report", "sha256")), rep.sha256, proj.ref(("report", "sha256"))),
        ("REVIEW evidence.sha256 is SAFE_EXPOSURE", review.get(("evidence", "sha256")), safe.sha256, review.ref(("evidence", "sha256"))),
        ("REVIEW source_map.sha256 is SOURCE_MAP", review.get(("source_map", "sha256")), smap.sha256, review.ref(("source_map", "sha256"))),
        ("E5 source_pins[context_diagnostic.py] is the local history_diagnostic.py", diag.get(("source_pins", "context_diagnostic.py", "sha256")),
         sha256(diagnostic_source), diag.ref(("source_pins", "context_diagnostic.py", "sha256"))),
        ("SOURCE_MAP lists history_diagnostic.py with the same sha256", listed[0]["sha256"] if len(listed) == 1 else None,
         sha256(diagnostic_source), smap.ref(("sources", "*", "sha256"), f"entry with path == {DIAGNOSTIC_SOURCE}")),
    ]
    return [{"check": name, "recorded": recorded, "observed": observed, "agrees": recorded == observed, "source": ref}
            for name, recorded, observed, ref in checks], {"path": DIAGNOSTIC_SOURCE, "sha256": sha256(diagnostic_source)}


def check_claims(root, paragraphs, N, roles):
    takeover_raw = (root / TAKEOVER).read_bytes()
    texts = {pid: squash(p["text"]) for pid, p in paragraphs.items()}
    texts["TAKEOVER"] = squash(takeover_raw.decode("utf-8").replace("`", "").replace("**", ""))
    results = []
    for doc, fragment, stated, ids in CLAIMS:
        found = squash(fragment) in texts[doc]
        values = {nid: N.value(nid) for nid in ids}
        statuses = {nid: (N.records[nid]["status"] if nid in N.records else "absent") for nid in ids}
        if not found:
            status = "fragment_not_found"
        elif any(s in ("absent", "untraceable") or values[n] is None for n, s in statuses.items()):
            status = "untraceable"
        else:
            status = "match" if all(v == stated for v in values.values()) else "mismatch"
        results.append({"document": doc, "fragment": fragment, "stated": stated, "numbers": values, "status": status,
                        "reading": READINGS.get((doc, fragment), "the number is printed in the fragment")})
    qualitative = []
    for doc, fragment, role, path in QUALITATIVE:
        source = roles[role]
        try:
            value = source.get(path)
            text = value["claim"] if isinstance(value, dict) else value
            kind = value.get("kind") if isinstance(value, dict) else None
            status = "traced_to_text"
        except Untraceable as error:
            text, kind, status = str(error), None, "untraceable"
        qualitative.append({"document": doc, "fragment": fragment, "fragment_found": squash(fragment) in texts[doc],
                            "status": status, "source": source.ref(path), "source_kind": kind, "source_text": text})
    return results, qualitative, {"path": TAKEOVER, "sha256": sha256(takeover_raw)}


def build(root: Path) -> dict:
    root = Path(root)
    maps = load_maps(root)
    paragraphs = check_paragraphs(root, maps)
    evidence_ids = sorted({eid for p in paragraphs.values() for eid in p["references"]})
    sources = resolve_sources(root, maps, evidence_ids)
    roles = assign_roles(sources)
    N = Numbers()
    extract_d90(roles, N)
    extract_cap180(roles, N)
    combine(N)
    chains, diagnostic_source = chain_checks(roles, root)
    claims, qualitative, takeover = check_claims(root, paragraphs, N, roles)
    records = [N.records[k] for k in sorted(N.records)]
    disagreements = [{"id": r["id"], "cross_check": c} for r in records for c in r["cross_checks"] if not c["agrees"]]
    untraceable = [r for r in records if r["status"] == "untraceable" or r["value"] is None]
    role_of = {source.rel: role for role, source in roles.items()}
    return {
        "schema_version": SCHEMA,
        "scope": "Recomputed only from existing authenticated SAFE reports referenced by the paper evidence maps for "
                 "manuscript paragraphs M14 and M25. No raw histories, prompts, task bytes, model calls or new runs.",
        "script": {"path": "scripts/exposure_funnel.py", "sha256": sha256(Path(__file__).read_bytes())},
        "evidence_maps": [{k: m[k] for k in ("path", "sha256", "bytes")} | {"schema_version": m["data"].get("schema_version")}
                          for m in maps],
        "paragraphs": paragraphs,
        "sources": [{"evidence": s.evidence_ids, "path": s.rel, "role": role_of.get(s.rel), "bytes": s.bytes,
                     "sha256_expected": s.expected, "sha256_observed": s.sha256, "verified": s.sha256 == s.expected}
                    for s in sources],
        "hash_chain_checks": chains,
        "stage_definitions": {"source": diagnostic_source,
                              "stages": [{"stage": stage, "diagnostic_key": key, "definition": text} for stage, key, text in STAGES]},
        "funnel": funnel_matrix(N),
        "claims_checked": claims,
        "qualitative_statements": qualitative,
        "takeover_source": takeover,
        "summary": {"numbers": len(records), "traced_direct": sum(r["status"] == "traced" for r in records),
                    "derived": sum(r["status"] == "derived" for r in records), "untraceable": len(untraceable),
                    "cross_checks": sum(len(r["cross_checks"]) for r in records), "cross_check_disagreements": len(disagreements),
                    "claims": {s: sum(c["status"] == s for c in claims) for s in ("match", "mismatch", "untraceable", "fragment_not_found")},
                    "hash_chain_checks_failed": sum(not c["agrees"] for c in chains)},
        "cross_check_disagreements": disagreements,
        "untraceable": [{"id": r["id"], "reason": r.get("reason", "missing component")} for r in untraceable],
        "limits": ["Counts establish offered-but-unused tool schemas in these development blocks; they do not show why "
                   "the model chose bash, that dormant graph/source handlers work, or any semantic use (E31 review verdict).",
                   "Development evidence only: D90 and Cap180 are one pvlib family each, six issues, descriptive.",
                   "offered_requests counts completed model requests whose tool list contained the tool; requested..consumed "
                   "count tool calls/results; offered_episodes counts episodes."],
        "numbers": records,
    }


def funnel_matrix(N):
    blocks = {"D90": "D90", "Cap180": "Cap180.all", "Cap180 B32000": "Cap180.B32000", "Cap180 B8000": "Cap180.B8000",
              "D90+Cap180": "D90+Cap180"}
    out = {}
    for label, prefix in blocks.items():
        out[label] = {"episodes": N.value(f"{prefix}.episodes" if prefix == "D90+Cap180" else f"{prefix}.episodes.planned"),
                      "model_calls": N.value(f"{prefix}.model_calls"),
                      "stages": {stage: {tool: N.value(f"{prefix}.{tool}.{stage}") for tool in TOOLS} for stage, _, _ in STAGES}}
    return out


# ------------------------------------------------------------------------------------------- markdown

def _n(value):
    return "untraceable" if value is None else f"{value:,}"


def _table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    return "\n".join(lines + ["| " + " | ".join(str(c) for c in row) + " |" for row in rows])


def _prov(record):
    if record is None:
        return "absent"
    if record["status"] == "traced":
        s = record["source"]
        return f"{s['evidence']} `{s['key_path']}`"
    if record["status"] == "derived":
        refs = record["sources"]
        if refs and "number" in refs[0]:
            return "sum of " + " + ".join(r["number"] for r in refs)
        return "derived: " + "; ".join(f"{r['evidence']} `{r['key_path']}`" for r in refs)
    return "UNTRACEABLE: " + record.get("reason", "")


def render_md(report: dict) -> str:
    records = {r["id"]: r for r in report["numbers"]}
    s = report["summary"]
    out = ["# Exposure funnel v1 (claim 1: offered is not exercised)", "",
           report["scope"], "",
           f"Numbers: {s['numbers']} ({s['traced_direct']} read directly at a JSON key, {s['derived']} derived from keyed "
           f"rows or sums of traced numbers, {s['untraceable']} untraceable). Cross-checks: {s['cross_checks']}, "
           f"disagreements: {s['cross_check_disagreements']}. Hash-chain checks failed: {s['hash_chain_checks_failed']}. "
           f"Claims: {s['claims']}.", "",
           "Every number in FUNNEL.json `numbers` has its file, sha256 and JSON key path (or derivation). "
           "Key paths use `[*]` for every element of a list.", ""]
    out += ["## Manuscript paragraphs", ""]
    for pid, p in report["paragraphs"].items():
        verified = ", ".join(f"{m['path']} {'verified' if m['text_sha256_verified'] else 'NOT verified'}" for m in p["manuscripts"])
        out += [f"**{pid}** (claims {', '.join(p['claim_ids'])}; references {', '.join(p['references'])}; text sha256 "
                f"`{p['text_sha256']}`: {verified})", "", "> " + p["text"], ""]
    out += ["## Sources (sha256 verified against both evidence maps)", "",
            _table(["Evidence", "Role", "Path", "Bytes", "sha256", "Verified"],
                   [["+".join(x["evidence"]), x["role"], f"`{x['path']}`", f"{x['bytes']:,}", f"`{x['sha256_observed']}`",
                     x["verified"]] for x in report["sources"]]), "",
            "Evidence maps: " + "; ".join(f"`{m['path']}` sha256 `{m['sha256']}`" for m in report["evidence_maps"]) + ".", "",
            "## Hash chain between reports", "",
            _table(["Check", "Agrees", "Recorded at"],
                   [[c["check"], c["agrees"], f"{c['source']['evidence']} `{c['source']['key_path']}`"] for c in report["hash_chain_checks"]]), ""]
    d = report["stage_definitions"]
    out += ["## Stage definitions", "",
            f"From `{d['source']['path']}` (sha256 `{d['source']['sha256']}`; the same bytes are pinned by E5 "
            "`source_pins[\"context_diagnostic.py\"]` and listed in the E31 SOURCE_MAP).", "",
            _table(["Stage", "Diagnostic key", "Definition"], [[x["stage"], f"`{x['diagnostic_key']}`" if x["diagnostic_key"] else "derived", x["definition"]] for x in d["stages"]]), ""]
    for label, prefix in (("D90", "D90"), ("Cap180 (both caps)", "Cap180.all"), ("Cap180 B32000 (32,000-byte bash output cap)", "Cap180.B32000"),
                          ("Cap180 B8000 (8,000-byte bash output cap)", "Cap180.B8000"), ("D90 + Cap180", "D90+Cap180")):
        episodes = records.get(f"{prefix}.episodes" if prefix == "D90+Cap180" else f"{prefix}.episodes.planned")
        calls = records.get(f"{prefix}.model_calls")
        out += [f"## Funnel: {label}", "",
                f"Episodes {_n(episodes and episodes['value'])}; completed model requests {_n(calls and calls['value'])}.", "",
                _table(["Stage"] + list(TOOLS) + ["Provenance (graph_query cell; other tools use the same key with their name)"],
                       [[stage] + [_n(records.get(f"{prefix}.{t}.{stage}", {}).get("value")) for t in TOOLS]
                        + [_prov(records.get(f"{prefix}.graph_query.{stage}"))] for stage, _, _ in STAGES]), ""]
        exits = sorted(k.split(".")[-1] for k in records if k.startswith(f"{prefix}.exits."))
        if exits:
            out += ["Exits: " + ", ".join(f"{e} {_n(records[f'{prefix}.exits.{e}']['value'])}" for e in exits) +
                    ". Parse-level checks: " + ", ".join(f"{z} {_n(records[f'{prefix}.{z}']['value'])}" for z in ZERO_CHECKS
                                                          if f"{prefix}.{z}" in records) + ".", ""]
    for block in ("D90", "Cap180"):
        rows = []
        for condition in CONDITIONS:
            get = lambda t, s: _n(records.get(f"{block}.by_condition.{condition}.{t}.{s}", {}).get("value"))
            rows.append([condition, _n(records[f"{block}.by_condition.{condition}.episodes"]["value"]),
                         _n(records[f"{block}.by_condition.{condition}.model_calls"]["value"]),
                         get("graph_query", "offered_episodes"), get("graph_query", "offered_requests"), get("graph_query", "requested"),
                         get("source_reference_search", "offered_episodes"), get("source_reference_search", "offered_requests"),
                         get("source_reference_search", "requested"), get("bash", "executed"), get("submit_patch", "executed")])
        out += [f"## {block} by condition", "",
                _table(["Condition", "Episodes", "Model requests", "graph offered (episodes)", "graph offered (requests)",
                        "graph requested", "source-ref offered (episodes)", "source-ref offered (requests)", "source-ref requested",
                        "bash executed", "submit_patch executed"], rows), ""]
    out += ["## Statements checked against traced numbers", "",
            _table(["Document", "Fragment (verbatim)", "Stated", "Read as", "Traced", "Status"],
                   [[c["document"], c["fragment"], c["stated"], c["reading"],
                     ", ".join(sorted({_n(v) for v in c["numbers"].values()})), c["status"]]
                    for c in report["claims_checked"]]), "",
            "Qualitative statements traced to review text:", "",
            _table(["Document", "Fragment", "Status", "Source"],
                   [[q["document"], q["fragment"], q["status"], f"{q['source']['evidence']} `{q['source']['key_path']}` ({q['source_kind'] or 'verdict'})"]
                    for q in report["qualitative_statements"]]), ""]
    out += ["## Untraceable numbers", "",
            "None." if not report["untraceable"] else _table(["Id", "Reason"], [[u["id"], u["reason"]] for u in report["untraceable"]]), "",
            "## Cross-check disagreements", "",
            "None." if not report["cross_check_disagreements"] else _table(["Id", "Check"], [[x["id"], json.dumps(x["cross_check"])] for x in report["cross_check_disagreements"]]), "",
            "## Limits", ""] + [f"- {x}" for x in report["limits"]] + [""]
    return "\n".join(out)


# ------------------------------------------------------------------------------------------- figure

def render_figure(report, out_dir: Path, overwrite: bool):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    surface, ink, ink2, muted, grid, axis = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
    colors = {"graph_query": "#2a78d6", "source_reference_search": "#eb6834", "bash": muted}
    plt.rcParams.update({"svg.hashsalt": "exposure-funnel-v1", "font.family": "DejaVu Sans", "font.size": 8.5,
                         "axes.edgecolor": axis, "axes.labelcolor": ink2, "xtick.color": ink2, "ytick.color": ink2,
                         "xtick.major.size": 0, "ytick.major.size": 0, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.spines.left": False, "figure.facecolor": surface, "axes.facecolor": surface,
                         "savefig.facecolor": surface, "axes.grid": True, "axes.grid.axis": "x", "grid.color": grid,
                         "grid.linewidth": 0.8})
    stages = [("offered_requests", "offered\n(model requests)"), ("requested", "requested\n(raw calls)"),
              ("parsed", "parsed"), ("executed", "executed"), ("visible", "visible to model"),
              ("consumed_completed_request", "consumed by next\nanswered request")]
    tools = ("graph_query", "source_reference_search", "bash")
    records = {r["id"]: r for r in report["numbers"]}
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), sharey=True)
    for ax, (title, prefix) in zip(axes, (("D90", "D90"), ("Cap180 (B32000 + B8000)", "Cap180.all"))):
        top = max(records[f"{prefix}.{t}.{s}"]["value"] for t in tools for s, _ in stages)
        height = 0.24
        for j, tool in enumerate(tools):
            ys = [k + (j - 1) * (height + 0.03) for k in range(len(stages))]
            values = [records[f"{prefix}.{tool}.{s}"]["value"] for s, _ in stages]
            ax.barh(ys, values, height=height, color=colors[tool], label=tool, zorder=2)
            for y, v in zip(ys, values):
                text = f"{v:,}" if v or tool == "bash" else f"0  {tool}"
                ax.text(v + top * 0.012, y, text, va="center", ha="left", fontsize=7.2, color=ink if v == 0 else ink2)
        episodes = records[f"{prefix}.episodes.planned"]["value"]
        calls = records[f"{prefix}.model_calls"]["value"]
        graph_eps = records[f"{prefix}.graph_query.offered_episodes"]["value"]
        source_eps = records[f"{prefix}.source_reference_search.offered_episodes"]["value"]
        ax.set_title(f"{title}: {episodes} episodes, {calls:,} completed model requests\n"
                     f"graph_query offered in {graph_eps} episodes, source_reference_search in {source_eps}",
                     loc="left", fontsize=8.6, color=ink)
        ax.set_yticks(range(len(stages)), [label for _, label in stages])
        ax.set_xlim(0, top * 1.16)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.set_xlabel("Count (offered: completed model requests; other stages: tool calls or results)", fontsize=7.8)
    axes[0].invert_yaxis()  # the y axis is shared: invert once
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=3, frameon=False, fontsize=8, labelcolor=ink2, bbox_to_anchor=(0.995, 0.985))
    fig.suptitle("Tool exposure funnel: graph and source-reference tools were offered but never requested",
                 x=0.01, ha="left", fontsize=10.5, color=ink)
    sub = (f"submit_patch executed: D90 {records['D90.submit_patch.executed']['value']}, Cap180 "
           f"{records['Cap180.all.submit_patch.executed']['value']} (see FUNNEL.md). Exact counts from authenticated SAFE "
           "reports E5 (D90) and E30 (Cap180), sha256-verified; provenance per number in FUNNEL.json.")
    fig.text(0.01, 0.012, sub, fontsize=7, color=ink2, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    written = {}
    for ext in ("png", "svg"):
        path = out_dir / f"funnel.{ext}"
        if path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite {path}")
        fig.savefig(path, format=ext, dpi=200, metadata={"Date": None} if ext == "svg" else {"Software": None})
        written[path.name] = sha256(path.read_bytes())
    plt.close(fig)
    manifest = {"schema_version": "exposure-funnel-figure-v1", "matplotlib": matplotlib.__version__, "outputs": written}
    target = out_dir / "FIGURE.json"
    if target.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {target}")
    target.write_bytes((json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode())
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--out", default="reports/exposure-funnel-v1")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--overwrite", action="store_true")
    group.add_argument("--check", action="store_true", help="regenerate and compare with the files on disk")
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.root)
    out = Path(args.out) if Path(args.out).is_absolute() else root / args.out
    report = build(root)
    files = {"FUNNEL.json": (json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
             "FUNNEL.md": render_md(report).encode("utf-8")}
    if args.check:
        mismatched = [name for name, data in files.items() if not (out / name).is_file() or (out / name).read_bytes() != data]
        print(json.dumps({"check": "fail" if mismatched else "pass", "mismatched": mismatched}))
        return 1 if mismatched else 0
    out.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        if (out / name).exists() and not args.overwrite:
            raise FileExistsError(f"refusing to overwrite {out / name} (pass --overwrite)")
    for name, data in files.items():
        (out / name).write_bytes(data)
    figure = None if args.no_figure else render_figure(report, out, args.overwrite)
    print(json.dumps({"out": str(out), "summary": report["summary"],
                      "written": {name: sha256(data) for name, data in files.items()}, "figure": figure}, sort_keys=True))
    bad = report["summary"]["untraceable"] or report["summary"]["cross_check_disagreements"] or \
        report["summary"]["hash_chain_checks_failed"] or report["summary"]["claims"]["mismatch"]
    return 2 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
