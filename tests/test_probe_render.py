"""Exp A rendering (D5, D10.2-D10.4) and wording v2 (D4) tests on synthetic fixtures (no research data)."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_probe_truth import CORE, CORE_ID, TARGET, V_TARGET, fixture_records, line_of, site, write_fixture  # noqa: E402

from absent_edges.data import canonical_json  # noqa: E402
from absent_edges.graphs import GraphFacts, GraphToolSet  # noqa: E402
from absent_edges.interventions import WORDING_TOLERANCE  # noqa: E402
from absent_edges.probe import render, truth, wording  # noqa: E402

LEXICAL_KIND = "lexical_candidates_not_proven_edges"


def stub_search_for(repo_root, package_dirs):
    """Deterministic stand-in for tools.lexical_search (not the real search).

    Whole-word hits of the query's simple name in package order, greedily bounded by the
    same 20 hits and 2,000 UTF-8 bytes as the live tool, with its source_sha256 and byte bound.
    """
    scope = truth.PackageScope(repo_root, package_dirs)
    source = scope.source_sha256()

    def search(query):
        simple = query.rsplit("::", 1)[-1].rsplit(".", 1)[-1]
        lines = truth.lexical_lines(scope, [simple])
        hits = [{"path": path, "line": number, "text": scope.lines(path)[number - 1].strip()[:400]}
                for path, number in lines[:render.SEARCH_LIMIT]]
        base = {"query": query, "source_sha256": source, "kind": LEXICAL_KIND, "error": None,
                "candidate_count": len(lines),
                "budget": {"max_hits": render.SEARCH_LIMIT, "max_tokens": render.SEARCH_MAX_BYTES,
                           "token_bound": render.C4_TOKEN_BOUND}}
        count = len(hits)
        while count and len(canonical_json({**base, "hits": hits[:count], "truncated": count < len(lines)})) \
                > render.SEARCH_MAX_BYTES:
            count -= 1
        return {**base, "hits": hits[:count], "truncated": count < len(lines)}
    return search


def stub_search_factory(inputs):
    return stub_search_for(inputs.repo_root, inputs.package_dirs)


def endpoint(payload, identity):
    (node,) = [node for node in payload["endpoints"] if node["id"] == identity]
    return node


def big_fixture(root, callers=12, false_positives=28, filler=60, unreported=0):
    """``callers`` traced callers of hub (the first ``unreported`` are missing from L) plus L false positives."""
    lines = ['"""Big fixture."""', "", "", "def hub(value):", '    """Hub."""', "    return value", ""]
    trace, locagent = [], []
    for number in range(callers):
        name = f"caller_{number:02d}"
        lines += ["", f"def {name}(value):", f'    """Caller {number} with a long body."""']
        lines += [f"    # filler line {index} " + "x" * 40 for index in range(filler)]
        lines.append("    return hub(value)")
        trace.append(site(f"pkg/big.py::{name}", "pkg/big.py::hub", "pkg/big.py", line=len(lines)))
        if number >= unreported:
            locagent.append(site(f"pkg/big.py::{name}", "pkg/big.py::hub", "pkg/big.py"))
    for number in range(false_positives):
        name = f"other_{number:02d}"
        lines += ["", f"def {name}(value):", "    return value"]
        locagent.append(site(f"pkg/big.py::{name}", "pkg/big.py::hub", "pkg/big.py"))
    write_fixture(root, {"pkg/__init__.py": "", "pkg/big.py": "\n".join(lines) + "\n"})
    return trace, locagent


class FakeBackend:
    """GraphToolSet backend returning a fixed observation and search block (no engine)."""
    backend = "locagent"

    def __init__(self, observation, block):
        self.observation, self.block = observation, block

    def query(self, source_root, symbol, relation="invokes", direction="outgoing", limit=30):
        return self.observation

    def source_reference_search(self, source_root, query, limit=20):
        return copy.deepcopy(self.block)


class FixtureRenderTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = write_fixture(Path(self.directory.name) / "repo")
        self.scope = truth.PackageScope(self.root, ["pkg/"])
        trace, locagent, jedi = fixture_records()
        self.table = truth.build_truth(self.scope, trace=trace, locagent=locagent, jedi=jedi)
        self.renderer = render.ItemRenderer(self.scope, source_sha256=self.scope.source_sha256(),
                                            coverage=render.coverage_for(["pkg/"]))
        self.search = stub_search_for(self.root, ["pkg/"])
        self.omitted = CORE_ID + "indirect_alias"
        self.prepared = self.prepare(TARGET, self.omitted)
        self.cells = self.renderer.cells(self.prepared, "A")

    def tearDown(self):
        self.directory.cleanup()

    def prepare(self, target, omitted, **options):
        return self.renderer.prepare(target, self.table.verified(target), self.table.callers_by(target, "L"),
                                     omitted, render.verification_block(self.search, target), **options)

    def test_vcomplete_answer_lists_every_verified_caller(self):
        c1 = self.cells["vcomplete"]["C1"]
        self.assertEqual(sorted(edge["source"] for edge in c1["matches"]), sorted(V_TARGET))
        self.assertEqual([node["id"] for node in c1["endpoints"]], sorted(V_TARGET | {TARGET}))
        self.assertEqual((c1["state"], c1["truncated"], c1["payload_truncated"], c1["symbol_indexed"]),
                         ("matches", False, False, True))
        self.assertEqual(c1["query"], {"symbol": TARGET, "relation": "invokes", "direction": "incoming"})
        self.assertEqual(c1["coverage"], {"completeness": "unknown", "scope": "Python functions and methods under pkg/"})
        self.assertFalse(any(node["snippet_truncated"] for node in c1["endpoints"]))

    def test_omit_node_drops_exactly_the_omitted_match_and_endpoint(self):
        complete, payload = self.cells["vcomplete"]["C1"], self.cells["omit_node"]["C1"]
        edge = render.invokes_edge(self.omitted, TARGET)
        self.assertEqual([canonical_json(item) for item in payload["matches"]],
                         [canonical_json(item) for item in complete["matches"] if item["id"] != edge["id"]])
        self.assertEqual([node["id"] for node in payload["endpoints"]],
                         [node["id"] for node in complete["endpoints"] if node["id"] != self.omitted])
        for node in payload["endpoints"]:
            self.assertEqual(canonical_json(node), canonical_json(endpoint(complete, node["id"])))
        for key in set(complete) - set(render.DIFFERS_BY_DESIGN["omit_node"]):
            self.assertEqual(canonical_json(payload[key]), canonical_json(complete[key]), key)
        self.assertEqual((payload["returned_count"], payload["query_result_count"]),
                         (complete["returned_count"] - 1, complete["query_result_count"] - 1))
        self.assertNotEqual(payload["facts_sha256"], complete["facts_sha256"])

    def test_omit_edge_keeps_every_endpoint_including_the_omitted_caller(self):
        complete, payload = self.cells["vcomplete"]["C1"], self.cells["omit_edge"]["C1"]
        self.assertEqual([node["id"] for node in payload["endpoints"]], [node["id"] for node in complete["endpoints"]])
        self.assertEqual(canonical_json(endpoint(payload, self.omitted)), canonical_json(endpoint(complete, self.omitted)))
        self.assertNotIn(self.omitted, [edge["source"] for edge in payload["matches"]])
        self.assertEqual(len(payload["matches"]), len(complete["matches"]) - 1)
        document = self.prepared["documents"]["omit_edge"].data
        self.assertIn(self.omitted, document["retained_query_endpoints"][f'["{TARGET}", "invokes", "incoming"]'])

    def test_omit_node_graph_payload_never_names_the_omitted_identity(self):
        cells = self.cells["omit_node"]
        self.assertFalse(render.graph_identity_exposed(self.omitted, cells))
        for key, payload in cells.items():
            self.assertNotIn(self.omitted, render.payload_text(render.graph_part(payload)), key)
        self.assertTrue(render.graph_identity_exposed(self.omitted, self.cells["vcomplete"]))
        self.assertNotIn("verification", render.graph_part(cells["C4a"]))
        self.assertEqual(canonical_json(render.graph_part(cells["C4a"])), canonical_json(cells["C1"]))

    def test_omitted_identity_check_is_token_bounded(self):
        omitted = "pkg/a.py::Cls.update"
        for text in ("pkg/a.py::Cls.update_extend", "sub/pkg/a.py::Cls.update", "my-pkg/a.py::Cls.update"):
            self.assertFalse(render.omitted_identity_exposed(omitted, {"C1": {"id": text}}), text)
        for value in ({"id": omitted}, {"snippet": "# see pkg/a.py::Cls.update here"}, {"x": [omitted, 1]},
                      {"id": "pkg/a.py::Cls.update.<locals>.inner"}):
            self.assertTrue(render.omitted_identity_exposed(omitted, {"C0": None, "C1": value}), value)

    def test_a_delivered_nested_identity_exposes_its_enclosing_omitted_caller(self):
        # Review finding: C*.<locals>.g literally contains C*'s identity and shows its qualified name, so D10.2
        # ("the payload contains C*'s identity") makes the item ineligible. The plain substring test agreed.
        omitted = "pkg/a.py::f"
        payload = {"endpoints": [{"id": "pkg/a.py::f.<locals>.g"}, {"id": "pkg/a.py::target"}], "matches": []}
        self.assertIn(omitted, render.payload_text(payload))
        self.assertTrue(render.omitted_identity_exposed(omitted, [payload]))
        self.assertTrue(render.graph_identity_exposed(omitted, {"C1": payload, "C4a": None}))
        control = {"endpoints": [{"id": "pkg/a.py::f"}, {"id": "pkg/a.py::target"}], "matches": []}
        self.assertTrue(render.omitted_identity_exposed(omitted, [control]))
        unrelated = {"endpoints": [{"id": "pkg/a.py::ff.<locals>.g"}, {"id": "pkg/a.py::f_other"}], "matches": []}
        self.assertFalse(render.omitted_identity_exposed(omitted, [unrelated]))

    def test_c2_and_c3_differ_only_in_the_guidance_value(self):
        for pair in wording.PAIRS:
            cells = self.renderer.cells(self.prepared, pair)
            for variant in render.VARIANTS:
                c2, c3 = dict(cells[variant]["C2"]), dict(cells[variant]["C3"])
                self.assertEqual(c2.pop("guidance"), wording.guidance(pair, "C2"))
                self.assertEqual(c3.pop("guidance"), wording.guidance(pair, "C3"))
                self.assertEqual(canonical_json(c2), canonical_json(c3))
                self.assertEqual(canonical_json(c2), canonical_json(cells[variant]["C1"]))
        with self.assertRaises(ValueError):
            self.renderer.cells(self.prepared, "C")

    def test_c4a_is_c1_plus_the_verification_block(self):
        block = render.verification_block(self.search, TARGET)
        block_bytes = len(canonical_json({"verification": block}))
        for variant in render.VARIANTS:
            c4a = dict(self.cells[variant]["C4a"])
            self.assertEqual(c4a.pop("verification"), block)
            self.assertEqual(canonical_json(c4a), canonical_json(self.cells[variant]["C1"]))
            self.assertEqual(len(canonical_json(self.cells[variant]["C4a"])),
                             len(canonical_json(self.cells[variant]["C1"])) + block_bytes - 1)
        self.assertLessEqual(len(canonical_json(block)), render.SEARCH_MAX_BYTES)

    def test_c4_triggered_is_reconstructed_byte_identically_to_the_frozen_tool(self):
        """D5: C4 triggered is never run; it is C4a when the frozen trigger fires on omit_node, else C1."""
        cases = {TARGET: (self.omitted, False), CORE_ID + "Box.method": (CORE_ID + "Box.build", True)}
        for target, (omitted, expected) in cases.items():
            prepared = self.prepare(target, omitted)
            cells = self.renderer.cells(prepared, "B")["omit_node"]
            fires = prepared["info"]["omit_node"]["c4_trigger_fires"]
            self.assertEqual(fires, expected, target)
            reconstructed = render.reconstruct_c4_triggered(cells, fires)
            self.assertEqual(canonical_json(reconstructed), canonical_json(cells["C4a" if fires else "C1"]))
            observation = prepared["documents"]["omit_node"].query(target, relation="invokes", direction="incoming",
                                                                   limit=30)
            tool = GraphToolSet(FakeBackend(observation, prepared["verification"]), lambda: "unused", "C4",
                                allow_fallback=lambda: True, max_output_bytes=render.MAX_OUTPUT_BYTES + 2200)
            live = tool.graph_query(target, relation="invokes", direction="incoming", limit=30)
            self.assertEqual(live.pop("auxiliary_tool_calls", 0), 1 if fires else 0)
            self.assertEqual(canonical_json(live), canonical_json(reconstructed), target)
            self.assertEqual(render.c4_triggered(observation.data), fires)

    def test_c0_has_no_payload(self):
        for variant in render.VARIANTS:
            self.assertEqual(tuple(self.cells[variant]), render.CELL_CONDITIONS)
        variants, payloads = render.variant_records(self.prepared, self.cells)
        for variant in render.VARIANTS:
            self.assertEqual(set(variants[variant]["cells"]), {"C1", "C2", "C3", "C4a"})
            self.assertNotIn("C0", variants[variant]["cell_bytes"])
        self.assertEqual(len(payloads), len({sha for entry in variants.values() for sha in entry["cells"].values()}))

    def test_render_size_bounds(self):
        for variant in render.VARIANTS:
            for condition in ("C1", "C2", "C3"):
                self.assertLessEqual(len(canonical_json(self.cells[variant][condition])), render.MAX_OUTPUT_BYTES)
            self.assertLessEqual(len(canonical_json(self.cells[variant]["C4a"])),
                                 render.MAX_OUTPUT_BYTES + render.SEARCH_MAX_BYTES)

    def test_natural_is_the_locagent_answer_with_its_false_positive(self):
        natural = self.cells["natural"]["C1"]
        sources = sorted(edge["source"] for edge in natural["matches"])
        self.assertEqual(sources, sorted(self.table.callers_by(TARGET, "L")))
        self.assertIn("pkg/util.py::unrelated", sources)
        self.assertNotIn(CORE_ID + "indirect_alias", sources)

    def test_small_item_keeps_full_snippets(self):
        self.assertEqual(self.prepared["caps"], {"target": 2048, "caller": 2048})
        for node in self.cells["vcomplete"]["C1"]["endpoints"]:
            self.assertEqual(node["snippet_end_line"], node["end_line"])
            self.assertFalse(node["snippet_truncated"])
        node = endpoint(self.cells["vcomplete"]["C1"], CORE_ID + "multi_line")
        start = line_of(CORE, "def multi_line(")
        self.assertEqual((node["start_line"], node["end_line"]), (start, start + 4))
        self.assertEqual(node["snippet"], "\n".join(CORE.splitlines()[start - 1:start + 4]))

    def test_rendering_is_deterministic(self):
        again = self.renderer.cells(self.prepare(TARGET, self.omitted), "A")
        self.assertEqual(canonical_json(again), canonical_json(self.cells))
        first = render.variant_records(self.prepared, self.cells)
        second = render.variant_records(self.prepared, again)
        self.assertEqual(canonical_json(first[0]), canonical_json(second[0]))

    def test_invariant_checker_accepts_the_build_and_rejects_tampering(self):
        rendered = self.cells
        block = render.verification_block(self.search, TARGET)
        render.check_invariants(TARGET, self.omitted, rendered, "A", verification=block)
        retained = next(node["id"] for node in rendered["omit_node"]["C1"]["endpoints"] if node["id"] != TARGET)

        def tamper(change):
            value = copy.deepcopy(rendered)
            change(value)
            return value

        def drop_second_endpoint(value):
            nodes = value["omit_node"]["C1"]["endpoints"]
            nodes.remove(endpoint(value["omit_node"]["C1"], retained))

        def alter_snippet_byte(variant, identity):
            def change(value):
                node = endpoint(value[variant]["C1"], identity)
                node["snippet"] = ("#" if node["snippet"][:1] != "#" else "$") + node["snippet"][1:]
            return change

        def reorder(value):
            value["omit_node"]["C1"]["endpoints"].reverse()

        cases = [
            (drop_second_endpoint, "endpoint ids are not the vcomplete ids minus the omitted caller"),
            (alter_snippet_byte("omit_node", retained), "differs from vcomplete"),
            (alter_snippet_byte("omit_node", TARGET), "differs from vcomplete"),
            (alter_snippet_byte("omit_edge", self.omitted), "differs from vcomplete"),
            (reorder, "endpoint ids"),
            (lambda value: value["omit_edge"]["C1"]["endpoints"].remove(
                endpoint(value["omit_edge"]["C1"], self.omitted)), "endpoint ids are not the vcomplete ids"),
            (lambda value: value["omit_node"]["C1"]["coverage"].update(scope="other"), "field 'coverage' differs"),
            (lambda value: value["omit_edge"]["C1"].update(returned_count=0), "drop by exactly one"),
            (lambda value: value["omit_node"]["C1"]["matches"].pop(), "remaining match records differ"),
            (lambda value: value["vcomplete"]["C3"]["matches"].pop(), "C3 facts differ from C1"),
            (lambda value: value["natural"]["C2"].update(guidance=wording.guidance("B", "C2")), "wrong C2 guidance"),
            (lambda value: value["omit_edge"]["C4a"].update(extra=1), "C4a must be C1 plus the verification block"),
            (lambda value: value["omit_node"]["C4a"]["verification"]["hits"].pop(), "one identical block"),
            (lambda value: value["omit_node"].update(C0={}), "cells must be exactly"),
            (lambda value: value["vcomplete"]["C1"].update(payload_truncated=True), "without truncation"),
        ]
        for change, message in cases:
            with self.assertRaisesRegex(AssertionError, message):
                render.check_invariants(TARGET, self.omitted, tamper(change), "A", verification=block)
        with self.assertRaisesRegex(AssertionError, "not delivered in the vcomplete answer"):
            render.check_invariants(TARGET, CORE_ID + "not_a_caller", rendered, "A", verification=block)
        with self.assertRaisesRegex(AssertionError, "carries a verification block other"):
            render.check_invariants(TARGET, self.omitted, rendered, "A", verification={**block, "hits": []})
        with self.assertRaisesRegex(AssertionError, "wrong C2 guidance for pair B"):
            render.check_invariants(TARGET, self.omitted, rendered, "B", verification=block)

    def test_verification_block_contract(self):
        good = self.search(TARGET)
        self.assertEqual(render.verification_block(lambda query: good, TARGET), good)
        bad = [{**good, "kind": "graph"}, {**good, "hits": [good["hits"][0]] * 21},
               {**good, "hits": [{"path": "pkg/core.py", "line": 1, "text": "x" * 2100}]},
               {key: value for key, value in good.items() if key != "budget"}]
        for block in bad:
            with self.assertRaises(ValueError):
                render.verification_block(lambda query, block=block: block, TARGET)

    def test_omitted_must_be_verified(self):
        with self.assertRaises(ValueError):
            self.prepare(TARGET, "pkg/util.py::unrelated")

    def test_natural_answer_for_a_target_locagent_does_not_index(self):
        prepared = self.renderer.prepare(TARGET, self.table.verified(TARGET), [], self.omitted,
                                         render.verification_block(self.search, TARGET), natural_target_indexed=False)
        natural = prepared["c1"]["natural"]
        self.assertEqual((natural["state"], natural["symbol_indexed"], natural["endpoints"], natural["matches"]),
                         ("symbol_not_indexed", False, [], []))
        self.assertTrue(prepared["info"]["natural"]["c4_trigger_fires"])
        self.assertEqual(canonical_json(prepared["c1"]["vcomplete"]), canonical_json(self.prepared["c1"]["vcomplete"]))
        with self.assertRaises(ValueError):
            self.renderer.prepare(TARGET, self.table.verified(TARGET), [CORE_ID + "direct_one"], self.omitted, None,
                                  natural_target_indexed=False)

    def test_location_visibility_helpers(self):
        payload = self.cells["omit_node"]["C1"]
        node = endpoint(payload, CORE_ID + "direct_one")
        inside = {"path": node["path"], "lineno": node["end_line"], "end_lineno": node["end_line"]}
        outside = {"path": node["path"], "lineno": node["end_line"] + 2, "end_lineno": node["end_line"] + 2}
        other_file = {"path": "pkg/util.py", "lineno": node["start_line"], "end_lineno": node["end_line"]}
        self.assertEqual(render.snippet_site_overlaps(payload, [inside]), [(node["id"], inside)])
        self.assertEqual(render.snippet_site_overlaps(payload, [outside, other_file]), [])
        cut = copy.deepcopy(payload)
        endpoint(cut, node["id"])["snippet_end_line"] = node["end_line"] - 1
        self.assertEqual(render.snippet_site_overlaps(cut, [inside]), [])
        # Verbatim text of another caller's call line counts as a covariate only.
        self.assertEqual(render.visibility_text_matches(payload, ["return target(value)"]),
                         sum(endpoint_node["snippet"].count("return target(value)")
                             for endpoint_node in payload["endpoints"]))
        self.assertEqual(render.visibility_text_matches(payload, []), 0)

    def test_family_facts_document_serves_the_natural_answer(self):
        pairs = [(caller, callee) for callee in self.table.callees() for caller in self.table.callers_by(callee, "L")]
        document = render.family_facts_document(self.scope, pairs, None, source_sha256=self.scope.source_sha256(),
                                                coverage=render.coverage_for(["pkg/"]))
        self.assertEqual((document["backend"], document["error"]), ("locagent", None))
        self.assertTrue(all(len(node["snippet"].encode("utf-8")) <= render.SNIPPET_MAX_BYTES
                            for node in document["nodes"]))
        rendered = GraphFacts(canonical_json(document)).query(TARGET, relation="invokes", direction="incoming",
                                                              limit=30).render("C1", max_output_bytes=12000)
        self.assertEqual(sorted(edge["source"] for edge in rendered["matches"]),
                         sorted(self.table.callers_by(TARGET, "L")))
        self.assertEqual(canonical_json(rendered["matches"]), canonical_json(self.cells["natural"]["C1"]["matches"]))

    def test_invariants_and_differing_fields_are_declared(self):
        self.assertEqual(render.INVARIANTS["omit_node_vs_vcomplete"]["differs_by_design"],
                         ["facts_sha256", "returned_count", "query_result_count", "matches", "endpoints"])
        self.assertEqual(render.INVARIANTS["omit_edge_vs_vcomplete"]["differs_by_design"],
                         ["facts_sha256", "returned_count", "query_result_count", "matches"])
        self.assertEqual(render.VARIANTS, ("vcomplete", "natural", "omit_node", "omit_edge"))
        self.assertEqual(render.CELL_CONDITIONS, ("C1", "C2", "C3", "C4a"))


class LargeItemRenderTests(unittest.TestCase):
    def test_caps_shrink_and_invariants_hold_for_twelve_large_callers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            trace, locagent = big_fixture(root)
            scope = truth.PackageScope(root, ["pkg/"])
            table = truth.build_truth(scope, trace=trace, locagent=locagent)
            hub = "pkg/big.py::hub"
            verified = table.verified(hub)
            self.assertEqual(len(verified), 12)
            renderer = render.ItemRenderer(scope, source_sha256=scope.source_sha256(),
                                           coverage=render.coverage_for(["pkg/"]))
            omitted = "pkg/big.py::caller_05"
            block = render.verification_block(stub_search_for(root, ["pkg/"]), hub)
            prepared = renderer.prepare(hub, verified, table.callers_by(hub, "L"), omitted, block)
            self.assertEqual(prepared["caps"]["target"], 2048)
            self.assertLess(prepared["caps"]["caller"], 2048)
            cells = renderer.cells(prepared, "A")
            render.check_invariants(hub, omitted, cells, "A", verification=block)
            complete = cells["vcomplete"]["C1"]
            self.assertFalse(complete["payload_truncated"])
            caller = endpoint(complete, "pkg/big.py::caller_00")
            self.assertLess(caller["snippet_end_line"], caller["end_line"])
            self.assertLessEqual(len(caller["snippet"].encode("utf-8")), prepared["caps"]["caller"])
            for variant in render.STRICT_VARIANTS:
                self.assertFalse(prepared["info"][variant]["payload_truncated"], variant)
            natural = prepared["info"]["natural"]
            self.assertTrue(natural["query_truncated"] and natural["payload_truncated"])
            self.assertEqual((natural["state"], natural["returned_count"]), ("truncated", 30))
            self.assertTrue(natural["c4_trigger_fires"])
            larger = renderer.fit_caps(hub, verified[:6])
            self.assertGreater(larger[1], prepared["caps"]["caller"])
            # D10.3: the caller cap is the largest at which vcomplete renders untruncated.
            caps = prepared["caps"]
            self.assertTrue(renderer._vcomplete_fits(hub, sorted(verified), caps["target"], caps["caller"]))
            self.assertFalse(renderer._vcomplete_fits(hub, sorted(verified), caps["target"], caps["caller"] + 1))


class ExcerptTests(unittest.TestCase):
    def test_whole_lines_utf8_boundary_and_empty(self):
        lines = ["def f():", "    return 1", "    # tail"]
        self.assertEqual(render.excerpt(lines, 1, 3, 1000), ("def f():\n    return 1\n    # tail", 3))
        self.assertEqual(render.excerpt(lines, 1, 3, 21), ("def f():\n    return 1", 2))
        self.assertEqual(render.excerpt(lines, 1, 3, 5), ("def f", 1))
        self.assertEqual(render.excerpt(["déf"], 1, 1, 2), ("d", 1))
        self.assertEqual(render.excerpt(lines, 1, 3, 0), ("", None))
        with self.assertRaises(ValueError):
            render.excerpt(lines, 0, 3, 10)


class WordingTests(unittest.TestCase):
    EXPECTED = {
        "A": {"C2": "Be careful with this answer. Review your assumptions and weigh the evidence before relying on "
                    "it in your final list. Inspect the relevant source and search for additional references before "
                    "you submit.",
              "C3": "This index answer is not an exhaustive list of callers. A missing relationship does not show "
                    "that no other caller exists. Inspect the relevant source and search for additional references "
                    "before you submit."},
        "B": {"C2": "Treat this evidence with caution. Question your assumptions and double-check your reasoning "
                    "before you finalize the list. Check the relevant source and look for further references before "
                    "you finish.",
              "C3": "Treat these indexed relationships as partial evidence. An absent relationship does not mean "
                    "that no further caller exists. Check the relevant source and look for further references before "
                    "you finish."},
    }

    def test_v2_texts_are_exactly_the_delta_wording(self):
        for pair, texts in self.EXPECTED.items():
            for condition, text in texts.items():
                self.assertEqual(wording.guidance(pair, condition), text, (pair, condition))
        self.assertEqual(wording.ACTION_SENTENCES,
                         {"A": "Inspect the relevant source and search for additional references before you submit.",
                          "B": "Check the relevant source and look for further references before you finish."})
        for pair in wording.PAIRS:
            action = wording.ACTION_SENTENCES[pair]
            for condition in wording.CONDITIONS:
                text = wording.guidance(pair, condition)
                self.assertEqual(text, wording.FIRST_STATEMENTS[pair][condition] + " " + action)
            first = {condition: wording.FIRST_STATEMENTS[pair][condition] for condition in wording.CONDITIONS}
            self.assertNotEqual(first["C2"], first["C3"])
        self.assertEqual(wording.REVISIONS, ())

    def test_pairs_are_assigned_by_rank_within_a_family(self):
        targets = [f"pkg/m.py::f{number}" for number in range(9)]
        assigned = wording.assign_pairs(targets)
        ranked = sorted(targets, key=wording.assignment_key)
        self.assertEqual(assigned, {target: "AB"[rank % 2] for rank, target in enumerate(ranked)})
        self.assertEqual(wording.assign_pairs(list(reversed(targets))), assigned)
        self.assertEqual(wording.assignment_key(targets[0]),
                         __import__("hashlib").sha256(("expA-wording-v2|" + targets[0]).encode()).hexdigest())
        self.assertEqual(wording.assign_pairs([]), {})
        with self.assertRaises(ValueError):
            wording.assign_pairs(["pkg/m.py::f", "pkg/m.py::f"])

    def test_manifest_and_guidance_budget(self):
        manifest = wording.wording_manifest()
        self.assertEqual(manifest["protocol"], "expA-wording-v2")
        self.assertEqual(manifest["tolerance"], WORDING_TOLERANCE)
        self.assertEqual(manifest["pairs"], self.EXPECTED)
        self.assertEqual(set(manifest["pair_sha256"]), {"A", "B"})
        self.assertEqual(wording.longest_guidance_bytes(),
                         max(len(canonical_json({"guidance": wording.guidance(pair, condition)}))
                             for pair in "AB" for condition in ("C2", "C3")))
        with self.assertRaises(ValueError):
            wording.guidance("C", "C2")
        with self.assertRaises(ValueError):
            wording.guidance("A", "C1")

    def test_audit_arithmetic_on_texts_and_full_payloads(self):
        class WordCounter:
            def count_text(self, text):
                return len(text.split())

            def count_chat(self, content):
                return 5 + len(content.split())

        payload = {"matches": [], "state": "indexed_no_matches", "endpoints": []}
        result = wording.audit_token_counts(WordCounter(), [payload])
        self.assertEqual(result["empty_chat_template"], 5)
        for pair in ("A", "B"):
            row = result["pairs"][pair]
            words = {condition: len(wording.guidance(pair, condition).split()) for condition in ("C2", "C3")}
            self.assertEqual(row["text_tokens"], words)
            self.assertEqual(row["absolute_difference"], abs(words["C2"] - words["C3"]))
            self.assertAlmostEqual(row["relative_difference"], abs(words["C2"] - words["C3"]) / min(words.values()))
            self.assertEqual(len(row["full_payloads"]), 1)
            self.assertEqual(row["full_payload_max_absolute_difference"], abs(words["C2"] - words["C3"]))
            self.assertEqual(row["within_tolerance"], row["absolute_difference"] <= 6)
        for bad in ({"guidance": "x"}, {"verification": {}}):
            with self.assertRaises(ValueError):
                wording.audit_token_counts(WordCounter(), [bad])


if __name__ == "__main__":
    unittest.main()
