"""Exp A items v2: eligibility, omission, membership, builder, verify, dispatch view and spec (synthetic fixtures)."""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_probe_truth import (CORE, CORE_ID, EXTRA_ID, TARGET, TRACE_CALLERS, UTIL_ID, V_TARGET,  # noqa: E402
                              fixture_records, line_of, site, write_fixture, write_records)
from test_probe_render import big_fixture, stub_search_factory  # noqa: E402

from absent_edges.data import canonical_json  # noqa: E402
from absent_edges.probe import items, render, truth, wording  # noqa: E402

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "expA_items.py"
METHOD, HELPER = CORE_ID + "Box.method", CORE_ID + "Box.helper"
JEDI_BROKEN, VIS_TARGET = UTIL_ID + "jedi_broken", EXTRA_ID + "vis_target"
JEDI_ERROR = {"stage": "jedi_definition", "identity": JEDI_BROKEN, "error": "timeout"}
CONTRACT_FIELDS = ("item_id", "family", "role", "package_dirs", "target", "V", "U", "omitted", "wording_pair",
                   "covariates", "membership", "lexical_sites", "render", "variants", "eligibility")


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_script():
    spec = importlib.util.spec_from_file_location("expA_items_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_cli(*argv):
    with contextlib.redirect_stdout(io.StringIO()) as stream:
        load_script().main(list(argv))
    return stream.getvalue()


def make_spec(base, *, with_trace=True, roster=("fixture", "other-a"), trace_status="complete",
              locagent_document=None, jedi_errors=(JEDI_ERROR,), records=None, files=None):
    root = write_fixture(base / "repo") if files is None else write_fixture(base / "repo", files)
    jedi = {"status": "partial" if jedi_errors else "complete", "errors": list(jedi_errors)}
    data = write_records(base / "data", records or fixture_records(), trace_status=trace_status,
                         static_manifests={"jedi": jedi})
    if locagent_document is not None:
        (data / "static" / "locagent.document.json").write_text(json.dumps(locagent_document), encoding="utf-8")
    entry = {"family": "fixture", "role": "main", "repo_root": str(root), "package_dirs": ["pkg/"],
             "commit": "0" * 40, "url": "https://example.invalid/fixture", "static_dir": str(data / "static")}
    if with_trace:
        entry["trace_dir"] = str(data / "trace")
    return {"main_roster": list(roster), "families": [entry]}


def build(base, spec, name="items", **options):
    path = base / (name + ".spec.json")
    path.write_text(json.dumps(spec), encoding="utf-8")
    families, roster = items.load_build_spec(path)
    options.setdefault("search_factory", stub_search_factory)
    manifest = items.build_items(families, roster, base / name, code_git_sha="test-sha", **options)
    records = [json.loads(line) for line in (base / name / "items.jsonl").read_text(encoding="utf-8").splitlines()]
    return manifest, records


def fixture_document(root, drop=()):
    """LocAgent-like nodes for a fixture: one function node per package function, one class node per class.

    Like LocAgent, there is no node for ``Class.__init__`` (its calls belong to the class node); ids drop
    ``.<locals>``. ``drop`` removes nodes by identity (``path::qualname``).
    """
    scope = truth.PackageScope(root, ["pkg/"])
    nodes = []
    for identity, info in sorted(scope.table().items()):
        if info["name"] != "__init__" and identity not in drop:
            nodes.append({"id": identity.replace(".<locals>", ""), "kind": "function", "path": info["path"],
                          "start_line": info["lineno"]})
    for relpath in scope.files():
        for entry in scope.index.classes(relpath):
            identity = f"{relpath}::{entry['qualname']}"
            if identity not in drop:
                nodes.append({"id": identity, "kind": "class", "path": relpath, "start_line": entry["lineno"]})
    return {"nodes": nodes, "edges": []}


INIT_FILES = {"pkg/__init__.py": "",
              "pkg/holder.py": ("def setup(value):\n    return value\n\n\nclass Holder:\n"
                                "    def __init__(self, value):\n        def inner(v):\n            return setup(v)\n"
                                "        self.value = inner(value)\n\n    def use(self):\n"
                                "        return setup(self.value)\n"),
              "pkg/unparsed.py": "def lost(value):\n    return value\n"}


def many_fixture(base, count=8):
    """``count`` targets t<n>, each with two traced callers a<n> and b<n>; no L or J edges."""
    lines = ['"""Many targets."""']
    for number in range(count):
        lines += ["", "", f"def t{number}(value):", "    return value", "", "",
                  f"def a{number}(value):", f"    return t{number}(value)", "", "",
                  f"def b{number}(value):", f"    return t{number}(value) + {number}"]
    text = "\n".join(lines) + "\n"
    trace = []
    for number in range(count):
        callee = f"pkg/many.py::t{number}"
        trace.append(site(f"pkg/many.py::a{number}", callee, "pkg/many.py", f"    return t{number}(value)", text))
        trace.append(site(f"pkg/many.py::b{number}", callee, "pkg/many.py", f"return t{number}(value) + {number}",
                          text))
    files = {"pkg/__init__.py": "", "pkg/many.py": text}
    spec = make_spec(base, records=(trace, [], []), files=files, jedi_errors=())
    return spec, [f"pkg/many.py::t{number}" for number in range(count)]


class KeyAndRuleTests(unittest.TestCase):
    def test_keys_follow_the_v2_formulas(self):
        self.assertEqual(items.selection_key("fam", TARGET), sha("expA-v2|fam|" + TARGET))
        self.assertEqual(items.omit_key(TARGET, METHOD), sha("expA-omit-v2|" + TARGET + "|" + METHOD))
        self.assertEqual(items.edge_key(TARGET), sha("expA-edge-v2|" + TARGET))
        self.assertEqual(items.tool_key(TARGET), sha("expA-tool-v2|" + TARGET))
        self.assertEqual(wording.assignment_key(TARGET), sha("expA-wording-v2|" + TARGET))
        self.assertEqual(items.item_id("fam", TARGET), sha("expA-item-v2|fam|" + TARGET)[:20])
        self.assertEqual(items.ROLES, ("main", "dev"))

    def test_omitted_caller_is_the_first_runtime_witnessed_caller_by_key(self):
        runtime = [f"pkg/x.py::c{number}" for number in range(6)]
        expected = min(runtime, key=lambda caller: sha("expA-omit-v2|" + TARGET + "|" + caller))
        self.assertEqual(items.omitted_caller(TARGET, runtime), expected)
        self.assertEqual(items.omitted_caller(TARGET, list(reversed(runtime))), expected)
        record = items.choose_omitted(TARGET, runtime, {caller: "direct" for caller in runtime})
        self.assertEqual(record, {"identity": expected, "stratum": "direct", "rule": "first_runtime_witnessed",
                                  "order_key": items.omit_key(TARGET, expected)})
        with self.assertRaises(ValueError):
            items.omitted_caller(TARGET, [])

    def test_first_by_key_takes_the_lowest_keys(self):
        targets = [f"pkg/m.py::f{number}" for number in range(9)]
        self.assertEqual(items.first_by_key(targets, items.edge_key, 5),
                         set(sorted(targets, key=items.edge_key)[:5]))
        self.assertEqual(items.first_by_key(targets[:3], items.tool_key, 5), set(targets[:3]))

    def test_decorator_normalization(self):
        self.assertEqual(items.decorator_base("@functools.wraps(fn)"), "functools.wraps")
        self.assertEqual(items.decorator_base({"text": "staticmethod"}), "staticmethod")
        self.assertEqual(items.decorator_base(" classmethod "), "classmethod")

    def test_jedi_definition_errors_come_from_the_j_manifest(self):
        manifest = {"errors": [JEDI_ERROR, {"stage": "jedi_definition_position", "identity": "pkg/a.py::g",
                                            "error": "name_position_not_found"},
                               {"stage": "parse", "path": "pkg/b.py", "error": "SyntaxError"},
                               {"stage": "jedi_definition", "identity": JEDI_BROKEN, "error": "again"}]}
        self.assertEqual(items.jedi_definition_errors(manifest),
                         {JEDI_BROKEN: "jedi_definition", "pkg/a.py::g": "jedi_definition_position"})
        self.assertEqual(items.jedi_definition_errors(None), {})
        self.assertEqual(items.jedi_definition_errors({"errors": []}), {})


class EligibilityTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = write_fixture(Path(self.directory.name) / "repo")
        self.scope = truth.PackageScope(self.root, ["pkg/"])
        trace, locagent, jedi = fixture_records()
        self.table = truth.build_truth(self.scope, trace=trace, locagent=locagent, jedi=jedi)
        self.rows = {row["identity"]: row for row in
                     items.candidate_rows("fixture", self.scope, self.table, {JEDI_BROKEN: "jedi_definition"})}

    def tearDown(self):
        self.directory.cleanup()

    def test_exclusion_reasons_rules_1_to_5(self):
        expected = {CORE_ID + "Box.prop": "property", CORE_ID + "Box.gen": "generator",
                    CORE_ID + "Box.coro": "coroutine", CORE_ID + "Box.__repr__": "dunder",
                    CORE_ID + "wrapped": "decorated", CORE_ID + "decorator_user.<locals>.local": "decorated",
                    CORE_ID + "outer.<locals>.inner": "nested_locals",
                    EXTRA_ID + "vis_outer.<locals>.vis_inner": "nested_locals",
                    JEDI_BROKEN: "jedi_definition_error", CORE_ID + "direct_one": "verified_below_2",
                    CORE_ID + "Box.build": "verified_below_2", UTIL_ID + "unrelated": "verified_below_2",
                    UTIL_ID + "static_only": "no_runtime_witnessed_caller"}
        self.assertEqual({identity: self.rows[identity]["excluded"] for identity in expected}, expected)
        eligible = sorted(identity for identity, row in self.rows.items() if row["excluded"] is None)
        self.assertEqual(eligible, sorted([TARGET, METHOD, HELPER, VIS_TARGET]))
        self.assertEqual((self.rows[TARGET]["verified_count"], self.rows[TARGET]["runtime_witnessed_count"]), (10, 8))
        self.assertEqual(self.rows[JEDI_BROKEN]["detail"], {"stage": "jedi_definition"})
        self.assertEqual(set(self.rows), set(self.scope.table()))
        self.assertEqual(len(self.rows), 40)

    def test_without_a_jedi_error_the_target_is_eligible(self):
        rows = {row["identity"]: row for row in items.candidate_rows("fixture", self.scope, self.table)}
        self.assertIsNone(rows[JEDI_BROKEN]["excluded"])
        self.assertEqual(rows[JEDI_BROKEN]["runtime_witnessed_count"], 2)

    def test_locagent_unindexed_excludes_targets_and_their_j_callers(self):  # D12.7
        rows = {row["identity"]: row for row in items.candidate_rows(
            "fixture", self.scope, self.table, {JEDI_BROKEN: "jedi_definition"},
            {CORE_ID + "callback_multiline", HELPER, JEDI_BROKEN})}
        self.assertEqual((rows[TARGET]["excluded"], rows[TARGET]["detail"]),
                         ("locagent_unindexed", {"target_unindexed": False,
                                                 "j_callers_unindexed": [CORE_ID + "callback_multiline"]}))
        self.assertEqual((rows[HELPER]["excluded"], rows[HELPER]["detail"]),
                         ("locagent_unindexed", {"target_unindexed": True, "j_callers_unindexed": []}))
        self.assertIsNone(rows[METHOD]["excluded"])
        self.assertEqual(rows[JEDI_BROKEN]["excluded"], "jedi_definition_error")  # the earlier rule wins
        self.assertEqual(rows[CORE_ID + "Box.prop"]["excluded"], "property")
        trace_only = {row["identity"]: row for row in items.candidate_rows(
            "fixture", self.scope, self.table, None, {CORE_ID + "getattr_user", CORE_ID + "outer.<locals>.inner"})}
        self.assertIsNone(trace_only[TARGET]["excluded"])  # unindexed callers witnessed only at runtime do not count
        self.assertEqual(items.EXCLUSION_ORDER.index("locagent_unindexed"),
                         items.EXCLUSION_ORDER.index("jedi_definition_error") + 1)

    def test_locagent_unindexed_identities_follow_the_class_node_init_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = write_fixture(Path(directory) / "repo", INIT_FILES)
            scope = truth.PackageScope(root, ["pkg/"])
            init, inner = "pkg/holder.py::Holder.__init__", "pkg/holder.py::Holder.__init__.<locals>.inner"
            document = fixture_document(root, drop={"pkg/unparsed.py::lost", init, inner})
            self.assertEqual(items.locagent_unindexed_identities(document, scope), ["pkg/unparsed.py::lost"])
            self.assertEqual(sorted(set(scope.table()) - truth.locagent_indexed_identities(document, scope)),
                             sorted([init, inner, "pkg/unparsed.py::lost"]))  # the function-node-only reading
            without_class = fixture_document(root, drop={"pkg/unparsed.py::lost", init, inner, "pkg/holder.py::Holder"})
            self.assertEqual(items.locagent_unindexed_identities(without_class, scope),
                             sorted([init, inner, "pkg/unparsed.py::lost"]))

    def test_verified_upper_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            trace, locagent = big_fixture(root, callers=13, false_positives=0, filler=1)
            scope = truth.PackageScope(root, ["pkg/"])
            table = truth.build_truth(scope, trace=trace, locagent=locagent)
            rows = {row["identity"]: row for row in items.candidate_rows("big", scope, table)}
            self.assertEqual(rows["pkg/big.py::hub"]["excluded"], "verified_above_12")
            self.assertEqual(rows["pkg/big.py::hub"]["verified_count"], 13)


class BuilderTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.base = Path(self.directory.name)
        self.spec = make_spec(self.base)
        self.manifest, self.records = build(self.base, self.spec)
        self.by_target = {record["target"]["identity"]: record for record in self.records}
        self.out = self.base / "items"

    def tearDown(self):
        self.directory.cleanup()

    def eligibility_rows(self, out=None):
        path = (out or self.out) / "eligibility" / "fixture.jsonl"
        return {row["identity"]: row for row in map(json.loads, path.read_text(encoding="utf-8").splitlines())}

    def test_items_are_the_first_eligible_targets_in_selection_order(self):
        targets = [record["target"]["identity"] for record in self.records]
        self.assertEqual(targets, sorted([TARGET, METHOD, HELPER], key=lambda i: items.selection_key("fixture", i)))
        self.assertEqual([record["eligibility"]["selection_index"] for record in self.records], [0, 1, 2])
        self.assertEqual(self.manifest["counts"]["items"], 3)
        self.assertEqual(self.manifest["counts"]["by_family"], {"fixture": 3})
        self.assertEqual(self.manifest["counts"]["by_role"], {"main": 3})
        self.assertTrue(self.manifest["families"]["fixture"]["short_of_m"])

    def test_funnel_counts_every_rule(self):
        funnel = self.manifest["families"]["fixture"]["funnel"]
        expected = {reason: 0 for reason in items.EXCLUSION_ORDER}
        expected.update(property=1, generator=1, coroutine=1, dunder=1, decorated=2, nested_locals=2,
                        jedi_definition_error=1, no_runtime_witnessed_caller=1, incidental_visibility=1,
                        verified_below_2=26)
        self.assertEqual(funnel["excluded"], expected)
        self.assertEqual((funnel["functions"], funnel["eligible"], funnel["selected"]), (40, 3, 3))
        self.assertEqual(funnel["text_rule"]["reached_visibility"], 4)
        self.assertEqual(self.manifest["counts"]["funnel_total"], expected)
        self.assertEqual(self.manifest["families"]["fixture"]["jedi_definition_errors"],
                         {"definitions": 1, "by_stage": {"jedi_definition": 1}, "not_package_functions": []})

    def test_visibility_rule_is_location_based(self):
        row = self.eligibility_rows()[VIS_TARGET]
        self.assertEqual(row["excluded"], "incidental_visibility")
        self.assertEqual((row["detail"]["snippet_covers_site"], row["detail"]["identity_in_payload"],
                          row["detail"]["covering_endpoints"]), (True, False, [EXTRA_ID + "vis_outer"]))
        self.assertIsInstance(row["detail"]["text_matches"], int)
        for record in self.records:
            self.assertEqual(record["eligibility"]["incidental_visibility"],
                             {"snippet_covers_site": False, "identity_in_payload": False})

    def test_record_follows_the_contract(self):
        record = self.by_target[TARGET]
        self.assertEqual(record["schema"], "expA-item-v2")
        for key in CONTRACT_FIELDS:
            self.assertIn(key, record)
        self.assertNotIn("split", record)
        self.assertEqual(record["role"], "main")
        self.assertEqual(record["item_id"], items.item_id("fixture", TARGET))
        self.assertEqual(record["target"], {"identity": TARGET, "path": "pkg/core.py",
                                            "line": line_of(CORE, "def target("),
                                            "end_line": line_of(CORE, "return value + 1"),
                                            "qualname": "target", "name": "target"})
        self.assertEqual({entry["identity"] for entry in record["V"]}, V_TARGET)
        for entry in record["V"]:
            self.assertEqual(set(entry), {"identity", "sources", "runtime_witnessed", "stratum", "sites"})
            self.assertTrue({"trace", "L", "J"} & set(entry["sources"]))
        for entry in record["U"]:
            self.assertEqual(set(entry), {"identity", "sources", "self", "sites"})
        self.assertIn(UTIL_ID + "unrelated", {entry["identity"] for entry in record["U"]})
        self.assertEqual(set(record["omitted"]), {"identity", "stratum", "rule", "order_key"})
        self.assertEqual(set(record["covariates"]), {"stratum", "c4_contains_omitted_site", "c4_trigger_fires",
                                                     "natural_miss", "n_V", "visibility_text_matches",
                                                     "visibility_docref_mentions", "visibility_name_mentions"})
        self.assertEqual(set(record["membership"]), {"in_s_nat", "in_s_edge", "in_t", "s_nat_excluded_reason"})
        self.assertLessEqual({"source_sha256", "snippet_caps", "cap_bytes"}, set(record["render"]))
        self.assertEqual((record["render"]["cap_bytes"], record["render"]["snippet_caps"]),
                         (12000, {"target": 2048, "caller": 2048}))
        self.assertEqual(set(record["variants"]), set(render.VARIANTS))
        for entry in record["variants"].values():
            self.assertLessEqual({"facts_sha256", "endpoint_ids", "payload_truncated", "cells", "cell_bytes"},
                                 set(entry))
            self.assertEqual(set(entry["cells"]), {"C1", "C2", "C3", "C4a"})
        self.assertEqual(record["variants"]["omit_node"]["endpoint_ids"],
                         [identity for identity in record["variants"]["vcomplete"]["endpoint_ids"]
                          if identity != record["omitted"]["identity"]])
        self.assertEqual(record["variants"]["omit_edge"]["endpoint_ids"],
                         record["variants"]["vcomplete"]["endpoint_ids"])

    def test_omitted_caller_and_covariates(self):
        record = self.by_target[TARGET]
        runtime = sorted(CORE_ID + name for name in TRACE_CALLERS)
        expected = min(runtime, key=lambda caller: items.omit_key(TARGET, caller))
        V = {entry["identity"]: entry for entry in record["V"]}
        self.assertEqual(record["omitted"], {"identity": expected, "stratum": V[expected]["stratum"],
                                             "rule": "first_runtime_witnessed",
                                             "order_key": items.omit_key(TARGET, expected)})
        covariates = record["covariates"]
        self.assertEqual((covariates["stratum"], covariates["n_V"], covariates["natural_miss"]),
                         (V[expected]["stratum"], 10, True))
        cells = record["variants"]["omit_node"]["cells"]
        block = items.load_payload(self.out, "fixture", cells["C4a"])["verification"]
        hits = {(hit["path"], hit["line"]) for hit in block["hits"]}
        self.assertEqual(covariates["c4_contains_omitted_site"],
                         any((row["path"], line) in hits for row in V[expected]["sites"]
                             for line in range(row["lineno"], row["end_lineno"] + 1)))
        omit_c1 = items.load_payload(self.out, "fixture", cells["C1"])
        self.assertEqual(covariates["c4_trigger_fires"], render.c4_triggered(omit_c1))
        self.assertIsInstance(covariates["visibility_text_matches"], int)
        mentions = render.omitted_name_mentions(omit_c1, expected)
        self.assertEqual((covariates["visibility_docref_mentions"], covariates["visibility_name_mentions"]),
                         (mentions["docref"], mentions["name"]))
        self.assertEqual(self.by_target[METHOD]["covariates"]["c4_trigger_fires"], True)  # one caller remains
        self.assertEqual(self.by_target[METHOD]["covariates"]["n_V"], 2)
        self.assertEqual(self.by_target[HELPER]["omitted"]["identity"], UTIL_ID + "helper_caller_a")
        self.assertEqual(record["natural"]["missing_runtime_witnessed"],
                         sorted(set(runtime) - set(record["natural"]["callers"])))

    def test_membership_and_wording_pairs(self):
        targets = [record["target"]["identity"] for record in self.records]
        pairs = wording.assign_pairs(targets)
        for record in self.records:
            target = record["target"]["identity"]
            self.assertEqual(record["wording_pair"], pairs[target])
            self.assertTrue(record["membership"]["in_s_edge"] and record["membership"]["in_t"])  # three <= 5
        self.assertEqual(self.by_target[TARGET]["membership"],
                         {"in_s_nat": True, "in_s_edge": True, "in_t": True, "s_nat_excluded_reason": None})
        self.assertTrue(self.by_target[METHOD]["membership"]["in_s_nat"])
        self.assertEqual((self.by_target[HELPER]["membership"]["in_s_nat"],
                          self.by_target[HELPER]["membership"]["s_nat_excluded_reason"]), (False, "no_natural_miss"))
        self.assertFalse(self.by_target[HELPER]["covariates"]["natural_miss"])

    def test_lexical_sites_are_the_grep_all_input(self):
        scope = truth.PackageScope(self.base / "repo", ["pkg/"])
        for record in self.records:
            target = record["target"]["identity"]
            names = truth.lexical_names(scope, target)
            self.assertEqual(record["names"]["lexical"], names)
            self.assertEqual([(row["path"], row["line"]) for row in record["lexical_sites"]],
                             truth.lexical_lines(scope, names))
        lexical = {(row["path"], row["line"]) for row in self.by_target[TARGET]["lexical_sites"]}
        self.assertIn(("pkg/__init__.py", 2), lexical)
        self.assertIn(("pkg/core.py", line_of(CORE, "RESULT = target(0)")), lexical)

    def test_stored_payloads_resolve_and_satisfy_invariants(self):
        for record in self.records:
            rendered = {variant: {cell: (None if digest is None else items.load_payload(self.out, "fixture", digest))
                                  for cell, digest in entry["cells"].items()}
                        for variant, entry in record["variants"].items()}
            block = rendered["omit_node"]["C4a"]["verification"]
            render.check_invariants(record["target"]["identity"], record["omitted"]["identity"], rendered,
                                    record["wording_pair"], verification=block)
            self.assertFalse(render.graph_identity_exposed(record["omitted"]["identity"], rendered["omit_node"]))
        facts = (self.out / "facts" / "fixture.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(facts), 3 * len(render.VARIANTS))
        self.assertEqual(sum(row["selected"] for row in self.eligibility_rows().values()), 3)
        self.assertTrue(json.loads(run_cli("verify", "--items-dir", str(self.out)))["ok"])

    def test_build_is_deterministic_and_immutable(self):
        build(self.base, self.spec, name="again")
        for relative in ("items.jsonl", "manifest.json", "payloads/fixture.jsonl", "facts/fixture.jsonl",
                         "eligibility/fixture.jsonl", "graph_facts/fixture.json"):
            self.assertEqual((self.out / relative).read_bytes(), (self.base / "again" / relative).read_bytes(), relative)
        with self.assertRaises(FileExistsError):
            build(self.base, self.spec)

    def test_manifest_records_inputs_code_invariants_and_counts(self):
        family = self.manifest["families"]["fixture"]
        self.assertEqual((family["trace_status"], family["trace"]["loader"]), ("complete", "tracer.load_trace_dir"))
        self.assertEqual(family["record_counts"]["trace"], {"records": 19, "kept": 16})
        trace_file = Path(self.spec["families"][0]["trace_dir"]) / "edges.jsonl"
        self.assertEqual(family["trace"]["files"]["edges.jsonl"]["sha256"],
                         hashlib.sha256(trace_file.read_bytes()).hexdigest())
        for key in ("locagent_edges", "jedi_edges", "jedi_manifest"):
            data = Path(family["inputs"][key]["path"]).read_bytes()
            self.assertEqual(family["inputs"][key]["sha256"], hashlib.sha256(data).hexdigest())
        self.assertEqual(self.manifest["code"]["git_sha"], "test-sha")
        for module in ("probe/render.py", "probe/items.py", "probe/truth.py", "probe/wording.py", "graphs.py"):
            self.assertIn(module, self.manifest["code"]["modules_sha256"])
        self.assertEqual(self.manifest["invariants"], render.INVARIANTS)
        self.assertEqual(self.manifest["parameters"]["salts"],
                         {"selection": "expA-v2", "omit": "expA-omit-v2", "edge": "expA-edge-v2",
                          "tool": "expA-tool-v2", "wording": "expA-wording-v2", "item_id": "expA-item-v2"})
        self.assertEqual(self.manifest["parameters"]["variants"], list(render.VARIANTS))
        self.assertNotIn("splits", self.manifest)
        counts = self.manifest["counts"]
        self.assertEqual(counts["V_count_histogram"], {"10": 1, "2": 2})
        self.assertEqual(counts["membership"], {"in_s_nat": 2, "in_s_edge": 3, "in_t": 3})
        self.assertEqual((counts["natural_miss"], counts["families_without_items"], counts["V_callers_without_sites"]),
                         (2, [], 0))
        self.assertEqual(counts["main_roster_missing_from_build"], ["other-a"])
        self.assertTrue(counts["c4_built"])
        self.assertEqual(json.loads((self.out / "manifest.json").read_bytes())["outputs"]["items.jsonl"]["sha256"],
                         hashlib.sha256((self.out / "items.jsonl").read_bytes()).hexdigest())

    def test_selection_takes_the_first_m_by_key(self):
        manifest, records = build(self.base, self.spec, name="m2", m=2)
        expected = sorted([TARGET, METHOD, HELPER], key=lambda i: items.selection_key("fixture", i))[:2]
        self.assertEqual([record["target"]["identity"] for record in records], expected)
        funnel = manifest["families"]["fixture"]["funnel"]
        self.assertEqual((funnel["eligible"], funnel["selected"], funnel["eligible_not_selected"]), (3, 2, 1))
        self.assertTrue(json.loads(run_cli("verify", "--items-dir", str(self.base / "m2")))["ok"])

    def test_c4a_required_unless_development_build(self):
        with self.assertRaises(ValueError):
            build(self.base, self.spec, name="noc4", search_factory=None)
        manifest, records = build(self.base, self.spec, name="dev-noc4", search_factory=None, require_c4=False)
        self.assertFalse(manifest["counts"]["c4_built"])
        self.assertTrue(all(record["variants"]["vcomplete"]["cells"]["C4a"] is None for record in records))
        self.assertTrue(all(record["covariates"]["c4_contains_omitted_site"] is None for record in records))
        self.assertTrue(items.verify_items(self.base / "dev-noc4")["ok"])
        with self.assertRaisesRegex(ValueError, "cannot be dispatched"):
            items.build_dispatch(self.base / "dev-noc4", self.base / "dev-noc4-dispatch")
        self.assertFalse((self.base / "dev-noc4-dispatch" / "dispatch_manifest.json").exists())

    def test_missing_or_unavailable_trace_yields_no_items(self):
        for name, spec in (("notrace", make_spec(self.base / "notrace", with_trace=False)),
                           ("unavailable", make_spec(self.base / "unavailable", trace_status="unavailable"))):
            manifest, records = build(self.base / name, spec)
            family = manifest["families"]["fixture"]
            self.assertEqual(family["trace_status"], "unavailable")
            self.assertEqual(records, [])
            self.assertEqual(manifest["counts"]["families_without_items"], ["fixture"])
            # target and static_only keep two L-and-J callers each, but no runtime witness.
            self.assertEqual(family["funnel"]["excluded"]["no_runtime_witnessed_caller"], 2)
            self.assertTrue(items.verify_items(self.base / name / "items")["ok"])
        self.assertEqual(manifest["families"]["fixture"]["record_counts"]["trace"], {"records": 0, "kept": 0})

    def test_static_manifest_status_and_source_tree_are_enforced(self):
        base = self.base / "manifests"
        spec = make_spec(base)
        static = base / "data" / "static"
        good = truth.PackageScope(base / "repo", ["pkg/"]).source_sha256()

        def write(backend, status, source):
            (static / f"{backend}.manifest.json").write_text(
                json.dumps({"status": status, "source_tree_sha256": source, "import_root": ""}), encoding="utf-8")
        write("locagent", "complete", good)
        write("jedi", "partial", good)
        manifest, _ = build(base, spec, name="ok")
        self.assertEqual(manifest["families"]["fixture"]["static_status"]["J"],
                         {"status": "partial", "source_tree_sha256": good, "commit": None, "identity_py_sha256": None})
        write("jedi", "complete", "0" * 64)
        with self.assertRaisesRegex(ValueError, "different package source tree"):
            build(base, spec, name="mismatch")
        write("jedi", "failed", good)
        with self.assertRaisesRegex(ValueError, "failed"):
            build(base, spec, name="failed")

    def test_missing_trace_dir_and_foreign_trace_tree_are_rejected(self):
        base = self.base / "guards"
        spec = make_spec(base)
        bad = json.loads(json.dumps(spec))
        bad["families"][0]["trace_dir"] = str(base / "no-such-trace-run")
        path = base / "bad.spec.json"
        path.write_text(json.dumps(bad), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "trace_dir"):
            items.load_build_spec(path)  # never a silent "trace unavailable"
        trace_manifest = Path(spec["families"][0]["trace_dir"]) / "manifest.json"
        good = truth.PackageScope(base / "repo", ["pkg/"]).source_sha256()
        trace_manifest.write_text(json.dumps({"status": "complete", "repo": {"source_tree": {"sha256": good}}}),
                                  encoding="utf-8")
        manifest, _ = build(base, spec, name="same-tree")
        self.assertTrue(manifest["families"]["fixture"]["trace_source_tree_matches"])
        trace_manifest.write_text(json.dumps({"status": "complete", "repo": {"source_tree": {"sha256": "0" * 64}}}),
                                  encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "trace ran on a different package source tree"):
            build(base, spec, name="other-tree")

    def test_locagent_document_sets_natural_index_state(self):
        base = self.base / "document"
        document = fixture_document(write_fixture(self.base / "document-source"))
        manifest, records = build(base, make_spec(base, locagent_document=document))
        index = manifest["families"]["fixture"]["locagent_index"]
        self.assertEqual((index["indexed_package_functions"], index["L_callees_not_indexed"]), (40, 0))
        self.assertEqual((index["unindexed_rule"], index["unindexed_functions"], index["function_node_only_unindexed"]),
                         ("applied", 0, 0))
        self.assertEqual([record["target"]["identity"] for record in records],
                         [record["target"]["identity"] for record in self.records])  # D12.7 excludes nothing here
        self.assertTrue(all(record["natural"]["target_indexed"] is True for record in records))
        self.assertTrue(all("locagent_unindexed" in record["eligibility"]["rules_passed"] for record in records))
        self.assertFalse(any("locagent_unindexed" in record["eligibility"]["rules_passed"] for record in self.records))
        self.assertEqual(manifest["counts"]["families_without_locagent_unindexed_rule"], [])
        self.assertEqual(self.manifest["counts"]["families_without_locagent_unindexed_rule"], ["fixture"])
        facts = json.loads((base / "items" / "graph_facts" / "fixture.json").read_bytes())
        self.assertIn(TARGET, {node["id"] for node in facts["nodes"]})

    def test_locagent_unindexed_rule_in_the_build(self):  # D12.7
        base = self.base / "sparse-document"
        document = {"nodes": [{"id": TARGET, "kind": "function", "path": "pkg/core.py",
                               "start_line": line_of(CORE, "def target(")}]}
        manifest, records = build(base, make_spec(base, locagent_document=document))
        family = manifest["families"]["fixture"]
        index = family["locagent_index"]
        self.assertEqual((index["indexed_package_functions"], index["L_callees_not_indexed"]), (1, 4))
        self.assertEqual((index["unindexed_functions"], index["function_node_only_unindexed"]), (39, 39))
        # Every function but the target lacks a node, and the target's J callers lack one too:
        # 40 functions - 8 metadata exclusions - 1 jedi error = 31.
        self.assertEqual((family["funnel"]["excluded"]["locagent_unindexed"], family["funnel"]["eligible"]), (31, 0))
        self.assertEqual(records, [])
        rows = {row["identity"]: row for row in map(json.loads, (base / "items" / "eligibility" / "fixture.jsonl")
                                                    .read_text(encoding="utf-8").splitlines())}
        self.assertEqual(rows[TARGET]["detail"]["target_unindexed"], False)
        self.assertIn(CORE_ID + "direct_two", rows[TARGET]["detail"]["j_callers_unindexed"])
        self.assertTrue(items.verify_items(base / "items")["ok"])

    def test_c4a_uses_the_shared_tools_lexical_search(self):
        manifest, records = build(self.base, self.spec, name="real-c4", search_factory=lambda inputs:
                                  render.shared_lexical_search(inputs.repo_root, inputs.package_dirs))
        self.assertTrue(manifest["counts"]["c4_built"])
        self.assertEqual(manifest["counts"]["c4_token_bounds"], ["utf8_bytes_upper_bound"])
        for record in records:
            self.assertTrue(record["verification"]["source_sha256_matches"])
            self.assertIsNone(record["verification"]["error"])
            self.assertLessEqual(record["verification"]["bytes"], render.SEARCH_MAX_BYTES)
            self.assertGreaterEqual(record["verification"]["hits"], 1)
        self.assertTrue(json.loads(run_cli("verify", "--items-dir", str(self.base / "real-c4")))["ok"])

    def test_cli_build_verify_without_c4_and_spec_validation(self):
        spec_path = self.base / "cli.spec.json"
        spec_path.write_text(json.dumps(self.spec), encoding="utf-8")
        summary = json.loads(run_cli("build", "--spec", str(spec_path), "--out", str(self.base / "cli"), "--no-c4"))
        self.assertEqual(summary["counts"]["items"], 3)
        self.assertTrue(json.loads(run_cli("verify", "--items-dir", str(self.base / "cli")))["ok"])
        self.assertEqual(len((self.base / "cli" / "items.jsonl").read_bytes().splitlines()), 3)
        bad = dict(self.spec, main_roster=["someone-else"])
        spec_path.write_text(json.dumps(bad), encoding="utf-8")
        with self.assertRaises(ValueError):
            items.load_build_spec(spec_path)
        bad = json.loads(json.dumps(self.spec))
        bad["families"][0]["role"] = "spare"
        spec_path.write_text(json.dumps(bad), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "role must be main or dev"):
            items.load_build_spec(spec_path)

    def test_verify_detects_tampering(self):
        def copy(name):
            target = self.base / name
            shutil.copytree(self.out, target)
            return target

        tampered = copy("t-bytes")
        data = bytearray((tampered / "items.jsonl").read_bytes())
        data[10] ^= 1
        (tampered / "items.jsonl").write_bytes(bytes(data))
        with self.assertRaisesRegex(ValueError, "items.jsonl does not match"):
            items.verify_items(tampered)
        tampered = copy("t-store")
        store = tampered / "payloads" / "fixture.jsonl"
        store.write_bytes(store.read_bytes().replace(b"return target(value)", b"return target(valuE)", 1))
        with self.assertRaisesRegex(ValueError, "does not match"):
            items.verify_items(tampered)
        tampered = copy("t-omitted")
        records = [dict(record) for record in self.records]
        record = next(record for record in records if record["target"]["identity"] == TARGET)
        other = next(entry for entry in record["V"]
                     if entry["runtime_witnessed"] and entry["identity"] != record["omitted"]["identity"])
        record["omitted"] = {**record["omitted"], "identity": other["identity"],
                             "order_key": items.omit_key(TARGET, other["identity"])}
        raw = b"".join(canonical_json(row) + b"\n" for row in records)
        (tampered / "items.jsonl").write_bytes(raw)
        manifest = json.loads((tampered / "manifest.json").read_bytes())
        manifest["outputs"]["items.jsonl"]["sha256"] = hashlib.sha256(raw).hexdigest()
        (tampered / "manifest.json").write_bytes(canonical_json(manifest) + b"\n")
        with self.assertRaisesRegex(ValueError, "omitted caller rule"):
            items.verify_items(tampered)


class NaturalTruncationTests(unittest.TestCase):
    def test_natural_truncation_removes_only_s_nat_membership(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "big"
            trace, locagent = big_fixture(root, unreported=1)
            files = {path.relative_to(root).as_posix(): path.read_text(encoding="utf-8")
                     for path in root.rglob("*.py")}
            manifest, records = build(base, make_spec(base, records=(trace, locagent, []), files=files,
                                                      jedi_errors=()))
            (record,) = records
            self.assertEqual(record["target"]["identity"], "pkg/big.py::hub")
            self.assertEqual(record["covariates"]["n_V"], 12)
            self.assertTrue(record["covariates"]["natural_miss"])
            self.assertTrue(record["variants"]["natural"]["payload_truncated"])
            self.assertEqual(record["membership"]["s_nat_excluded_reason"], "natural_render_truncated")
            self.assertFalse(record["membership"]["in_s_nat"])
            self.assertLess(record["render"]["snippet_caps"]["caller"], 2048)
            for variant in render.STRICT_VARIANTS:
                self.assertFalse(record["variants"][variant]["payload_truncated"])
            self.assertEqual(manifest["counts"]["s_nat_excluded_natural_truncated"], 1)
            self.assertEqual(record["natural"]["missing_runtime_witnessed"], ["pkg/big.py::caller_00"])
            self.assertTrue(items.verify_items(base / "items")["ok"])


class ManyItemsAndDispatchTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.base = Path(self.directory.name)
        self.spec, self.targets = many_fixture(self.base)
        self.manifest, self.records = build(self.base, self.spec)
        self.out = self.base / "items"

    def tearDown(self):
        self.directory.cleanup()

    def test_membership_and_pairs_are_drawn_among_the_selected_items(self):
        selected = [record["target"]["identity"] for record in self.records]
        self.assertEqual(sorted(selected), sorted(self.targets))
        edge = set(sorted(selected, key=items.edge_key)[:5])
        tool = set(sorted(selected, key=items.tool_key)[:5])
        pairs = wording.assign_pairs(selected)
        for record in self.records:
            target = record["target"]["identity"]
            self.assertEqual(record["membership"]["in_s_edge"], target in edge)
            self.assertEqual(record["membership"]["in_t"], target in tool)
            self.assertEqual(record["wording_pair"], pairs[target])
            self.assertTrue(record["membership"]["in_s_nat"])  # no L edges: every runtime caller is missing
        self.assertEqual(self.manifest["counts"]["wording_pair"], {"A": 4, "B": 4})
        manifest, records = build(self.base, self.spec, name="m6", m=6)
        chosen = [record["target"]["identity"] for record in records]
        self.assertEqual(chosen, sorted(self.targets, key=lambda target: items.selection_key("fixture", target))[:6])
        self.assertEqual({record["target"]["identity"] for record in records if record["membership"]["in_s_edge"]},
                         set(sorted(chosen, key=items.edge_key)[:5]))

    def test_dispatch_view_has_only_what_the_runtime_needs(self):
        report = items.build_dispatch(self.out, self.base / "dispatch", code_git_sha="test-sha")
        self.assertEqual((report["rows"], report["items_checked"], report["ok"]), (8, 8, True))
        directory = self.base / "dispatch"
        rows = [json.loads(line) for line in (directory / "dispatch.jsonl").read_bytes().splitlines()]
        manifest = json.loads((directory / "dispatch_manifest.json").read_bytes())
        self.assertEqual(manifest["schema"], "expA-dispatch-v2")
        self.assertEqual(manifest["items_sha256"], hashlib.sha256((self.out / "items.jsonl").read_bytes()).hexdigest())
        self.assertEqual(manifest["manifest_sha256"],
                         hashlib.sha256((self.out / "manifest.json").read_bytes()).hexdigest())
        self.assertEqual(manifest["dispatch_sha256"],
                         hashlib.sha256((directory / "dispatch.jsonl").read_bytes()).hexdigest())
        self.assertEqual((directory / "graph_facts/fixture.json").read_bytes(),
                         (self.out / "graph_facts/fixture.json").read_bytes())
        # D10.7: the staged store holds exactly the dispatched cells, as byte-identical rows of the items store.
        named = {sha for row in (json.loads(line) for line in (directory / "dispatch.jsonl").read_bytes().splitlines())
                 for conditions in row["cells"].values() for sha in conditions.values()}
        full = (self.out / "payloads/fixture.jsonl").read_bytes().splitlines()
        subset = (directory / "payloads/fixture.jsonl").read_bytes().splitlines()
        self.assertEqual(subset, [raw for raw in full if json.loads(raw)["sha256"] in named])
        self.assertEqual({json.loads(raw)["sha256"] for raw in subset}, named)
        self.assertLess(len(subset), len(full))   # vcomplete C2/C4a and the cells of non-members are not shipped
        self.assertEqual(manifest["payload_store"]["fixture"]["sha256"],
                         hashlib.sha256((directory / "payloads/fixture.jsonl").read_bytes()).hexdigest())
        self.assertEqual(manifest["payload_store"]["fixture"]["rows"], len(subset))
        self.assertEqual(manifest["items_payload_store"]["fixture"]["sha256"],
                         hashlib.sha256((self.out / "payloads/fixture.jsonl").read_bytes()).hexdigest())
        self.assertEqual(manifest["graph_facts"]["fixture"]["sha256"],
                         hashlib.sha256((self.out / "graph_facts/fixture.json").read_bytes()).hexdigest())
        self.assertFalse(items._truth_keys({key: value for key, value in manifest.items() if key != "code"}))
        by_id = {record["item_id"]: record for record in self.records}
        for row in rows:
            self.assertEqual(list(row), sorted(items.DISPATCH_ROW_KEYS))
            self.assertEqual(set(row["target"]), {"identity", "path", "line", "qualname"})
            self.assertEqual(set(row["membership"]), {"in_s_nat", "in_s_edge", "in_t"})
            self.assertFalse(items._truth_keys({key: value for key, value in row.items() if key != "cells"}))
            record = by_id[row["item_id"]]
            expected = {"omit_node": {condition: record["variants"]["omit_node"]["cells"][condition]
                                      for condition in ("C1", "C2", "C3", "C4a")},
                        "vcomplete": {condition: record["variants"]["vcomplete"]["cells"][condition]
                                      for condition in ("C1", "C3")},
                        "natural": {condition: record["variants"]["natural"]["cells"][condition]
                                    for condition in ("C1", "C3", "C4a")}}
            if record["membership"]["in_s_edge"]:
                expected["omit_edge"] = {condition: record["variants"]["omit_edge"]["cells"][condition]
                                         for condition in ("C1", "C3")}
            self.assertEqual(row["cells"], expected)
            self.assertNotIn(record["omitted"]["identity"], json.dumps(row))
        self.assertEqual(manifest["membership_counts"], {"in_s_nat": 8, "in_s_edge": 5, "in_t": 5})
        self.assertEqual(manifest["no_payload_cells"][0], {"block": "P", "variant": "none", "conditions": ["C0"],
                                                           "membership": None})  # D11.4: C0 runs in block P
        self.assertEqual(manifest["payload_cell_counts"]["omit_edge/C1"], 5)
        with self.assertRaises(FileExistsError):
            items.build_dispatch(self.out, directory)
        cli = json.loads(run_cli("verify", "--items-dir", str(self.out), "--dispatch-dir", str(directory)))
        self.assertTrue(cli["ok"] and cli["dispatch"]["ok"])

    def test_dispatch_verification_rejects_tampering(self):
        items.build_dispatch(self.out, self.base / "dispatch")

        def tampered(name, change):
            directory = self.base / name
            shutil.copytree(self.base / "dispatch", directory)
            rows = [json.loads(line) for line in (directory / "dispatch.jsonl").read_bytes().splitlines()]
            change(rows)
            raw = b"".join(canonical_json(row) + b"\n" for row in rows)
            (directory / "dispatch.jsonl").write_bytes(raw)
            manifest = json.loads((directory / "dispatch_manifest.json").read_bytes())
            manifest["dispatch_sha256"] = hashlib.sha256(raw).hexdigest()
            (directory / "dispatch_manifest.json").write_bytes(canonical_json(manifest) + b"\n")
            return directory

        cases = [
            ("t-truth", lambda rows: rows[0].update(V=[]), "fields"),
            ("t-reason", lambda rows: rows[0]["membership"].update(s_nat_excluded_reason=None), "membership fields"),
            ("t-cell", lambda rows: rows[0]["cells"]["omit_node"].update(C1="0" * 64), "not in the payload store"),
            ("t-cells-membership", lambda rows: rows[0]["cells"].pop("natural"), "cells disagree with membership"),
        ]
        for name, change, message in cases:
            with self.assertRaisesRegex(ValueError, message):
                items.verify_dispatch(tampered(name, change))
        flipped = tampered("t-pair", lambda rows: rows[0].update(wording_pair="B" if rows[0]["wording_pair"] == "A"
                                                                     else "A"))
        self.assertTrue(items.verify_dispatch(flipped)["ok"])  # alone it looks valid ...
        with self.assertRaisesRegex(ValueError, "differs from its item"):
            items.verify_dispatch(flipped, self.out)  # ... but it is not the projection of its item
        # Review finding: every row of the shipped store must be named by some dispatch row (no undispatched
        # vcomplete C2/C4a, natural or omit_edge cells of non-members on the runtime host).
        padded = self.base / "t-padded"
        shutil.copytree(self.base / "dispatch", padded)
        store = padded / "payloads" / "fixture.jsonl"
        os.chmod(store, 0o644)
        store.write_bytes((self.out / "payloads" / "fixture.jsonl").read_bytes())    # the whole items store
        manifest = json.loads((padded / "dispatch_manifest.json").read_bytes())
        manifest["payload_store"]["fixture"]["sha256"] = hashlib.sha256(store.read_bytes()).hexdigest()
        (padded / "dispatch_manifest.json").write_bytes(canonical_json(manifest) + b"\n")
        with self.assertRaisesRegex(ValueError, "payloads that no dispatch row names"):
            items.verify_dispatch(padded)


DFT_MODULE = '''"""Replica of the shape of dev item a45f9ede0fce35cd6111 (more_itertools, written for this test)."""


def _complex_sumprod(v1, v2):
    """High precision sumprod() for complex numbers.

    Used by :func:`dft` and :func:`idft`.
    """
    total = 0
    for a, b in zip(v1, v2):
        total += a * b
    return total


def dft(xarr):
    """Discrete Fourier Transform.

    See :func:`idft` for the inverse Discrete Fourier Transform.
    """
    out = []
    for k in range(len(xarr)):
        out.append(_complex_sumprod(xarr, [k]))
    return out


def idft(Xarr):
    """Inverse Discrete Fourier Transform."""
    out = []
    for k in range(len(Xarr)):
        out.append(_complex_sumprod(Xarr, [k]))
    return out
'''


class DocstringCrossReferenceTests(unittest.TestCase):
    """Review finding: a docstring cross-reference naming C* is visible to the model and no D10.2 rule sees it."""

    def test_the_dev_item_shape_is_excluded_by_the_docref_rule(self):
        # D13.1: the target's "Used by :func:`dft` and :func:`idft`." and dft's "See :func:`idft` ..." would be
        # delivered in every omit_node answer cell (never in C0), so the item is ineligible.
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = write_fixture(base / "repo", {"more_itertools/__init__.py": "",
                                                 "more_itertools/more.py": DFT_MODULE})
            path, target = "more_itertools/more.py", "more_itertools/more.py::_complex_sumprod"
            trace = [site(f"{path}::dft", target, path, "out.append(_complex_sumprod(xarr", DFT_MODULE),
                     site(f"{path}::idft", target, path, "out.append(_complex_sumprod(Xarr", DFT_MODULE)]
            data = write_records(base / "data", (trace, [], []),
                                 static_manifests={"jedi": {"status": "complete", "errors": []}})
            spec = {"main_roster": [], "families": [{
                "family": "more-itertools", "role": "dev", "repo_root": str(root), "package_dirs": ["more_itertools/"],
                "commit": "0" * 40, "static_dir": str(data / "static"), "trace_dir": str(data / "trace")}]}
            manifest, records = build(base, spec, name="dft")
            self.assertEqual(records, [])
            funnel = manifest["families"]["more-itertools"]["funnel"]
            self.assertEqual(funnel["excluded"]["docref_visibility"], 1)
            self.assertEqual(funnel["excluded"]["incidental_visibility"], 0)
            self.assertEqual(manifest["counts"]["visibility_docref_mentions_positive"], 0)
            self.assertTrue(items.verify_items(base / "dft")["ok"])

    def test_docref_mentions_follow_sphinx_roles_and_backticks(self):
        def payload(*snippets):
            return {"endpoints": [{"id": f"pkg/a.py::f{k}", "snippet": text} for k, text in enumerate(snippets)]}
        omitted = "pkg/a.py::Cls.idft"
        self.assertEqual(render.omitted_name_mentions(payload("Used by :func:`dft` and :func:`idft`."), omitted),
                         {"docref": 1, "name": 1})
        cases = {":meth:`~pkg.Cls.idft`": 1, ":py:func:`inverse <pkg.a.idft>`": 1, "``idft``": 1, "`idft()`": 1,
                 ":func:`idft_extra`": 0, "`dft`": 0, "return idft(x)": 0, "plain idft text": 0}
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(render.omitted_name_mentions(payload(text), omitted)["docref"], expected)
        self.assertEqual(render.omitted_name_mentions(payload("return idft(x)", "x.idft_extra()"), omitted)["name"], 1)
        self.assertEqual(render.omitted_name_mentions(payload(""), omitted), {"docref": 0, "name": 0})


class SpecTests(unittest.TestCase):
    TREE = "f" * 64

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.base = Path(self.directory.name)
        self.identity = hashlib.sha256((Path(items.__file__).parent / "identity.py").read_bytes()).hexdigest()
        self.tracer = hashlib.sha256((Path(items.__file__).parent / "tracer.py").read_bytes()).hexdigest()
        families = [("fam-main", "main", "ok"), ("fam-main2", "main", "ok"), ("fam-dev", "dev", "ok"),
                    ("fam-spare", "spare", "ok"), ("fam-gone", None, "excluded")]
        self.registry = {"schema": "expA-registry-v1", "families": [
            {"family": name, "role": role, "status": status, "commit": sha(name)[:40], "package_dirs": ["pkg/"],
             "url": f"https://example.invalid/{name}"} for name, role, status in families]}
        self.registry_path = self.base / "registry.json"
        self.registry_path.write_text(json.dumps(self.registry), encoding="utf-8")
        for name, role, status in families:
            if status != "ok" or role not in ("main", "dev"):
                continue
            (self.base / "repos" / name / "pkg").mkdir(parents=True)
            for backend in ("locagent", "jedi"):
                self.write_static(name, backend)
            self.write_trace(name)

    def tearDown(self):
        self.directory.cleanup()

    def write_static(self, name, backend, **changes):
        static = self.base / "static" / name
        static.mkdir(parents=True, exist_ok=True)
        (static / f"{backend}.edges.jsonl").write_text("", encoding="utf-8")
        manifest = {"status": "complete", "repo": {"commit": sha(name)[:40], "tracked_changes": False},
                    "code_sha256": {"absent_edges/probe/identity.py": self.identity}, "package_dirs": ["pkg"],
                    "source_tree_sha256": self.TREE}
        manifest.update(changes)
        (static / f"{backend}.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        if backend == "locagent":
            (static / "locagent.document.json").write_text(json.dumps({"nodes": [], "edges": []}), encoding="utf-8")

    def write_trace(self, name, subdir="trace-1", **changes):
        trace = self.base / "traces" / name / subdir
        trace.mkdir(parents=True, exist_ok=True)
        manifest = {"status": "complete", "plugin_files": {"absent_edges/probe/identity.py": self.identity,
                                                           "absent_edges/probe/tracer.py": self.tracer},
                    "repo": {"git": {"head": sha(name)[:40], "dirty": False}, "package_dirs": ["pkg"],
                             "source_tree": {"sha256": self.TREE}}}
        manifest.update(changes)
        (trace / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def spec(self, **options):
        options.setdefault("trace_root", self.base / "traces")
        options.setdefault("trace_subdir", "trace-1")
        return items.build_spec_from_registry(self.registry_path, repos_root=self.base / "repos",
                                              static_root=self.base / "static", **options)

    def test_spec_lists_main_and_dev_families_with_the_full_roster(self):
        spec = self.spec()
        self.assertEqual([entry["family"] for entry in spec["families"]], ["fam-dev", "fam-main", "fam-main2"])
        self.assertEqual(spec["main_roster"], ["fam-main", "fam-main2"])
        entry = next(entry for entry in spec["families"] if entry["family"] == "fam-main")
        self.assertEqual(entry, {"family": "fam-main", "role": "main", "repo_root": str(self.base / "repos" / "fam-main"),
                                 "package_dirs": ["pkg"], "commit": sha("fam-main")[:40],
                                 "url": "https://example.invalid/fam-main",
                                 "static_dir": str(self.base / "static" / "fam-main"),
                                 "trace_dir": str(self.base / "traces" / "fam-main" / "trace-1")})
        self.assertEqual(spec["families_without_trace"], [])
        self.assertTrue(spec["identity_py_sha256"]["all_inputs_equal_local"])
        self.assertEqual(spec["registry"]["sha256"], hashlib.sha256(self.registry_path.read_bytes()).hexdigest())
        only_main = self.spec(roles=["main"], families=["fam-main"])
        self.assertEqual(([entry["family"] for entry in only_main["families"]], only_main["main_roster"]),
                         (["fam-main"], ["fam-main", "fam-main2"]))
        with self.assertRaises(items.SpecError):
            self.spec(families=["fam-spare"])

    def test_spec_cross_checks_commits_trees_and_package_dirs(self):
        self.write_static("fam-main", "jedi", repo={"commit": "1" * 40, "tracked_changes": False})
        self.write_trace("fam-dev", repo={"git": {"head": sha("fam-dev")[:40], "dirty": True}, "package_dirs": ["pkg"],
                                          "source_tree": {"sha256": "e" * 64}})
        self.write_static("fam-main2", "locagent", package_dirs=["other"])
        with self.assertRaises(items.SpecError) as raised:
            self.spec()
        message = str(raised.exception)
        for fragment in ("fam-main: static jedi ran at commit", "fam-dev: the trace ran on a dirty checkout",
                         "fam-dev: L, J and the trace ran on different package source trees",
                         "fam-main2: static locagent package_dirs differ from the registry"):
            self.assertIn(fragment, message)

    def test_missing_trace_and_identity_pins(self):
        (self.base / "traces" / "fam-dev" / "trace-1" / "manifest.json").unlink()
        with self.assertRaisesRegex(items.SpecError, "fam-dev: trace manifest"):
            self.spec()
        spec = self.spec(allow_missing_trace=True)
        self.assertEqual(spec["families_without_trace"], ["fam-dev"])
        self.assertNotIn("trace_dir", next(entry for entry in spec["families"] if entry["family"] == "fam-dev"))
        self.assertEqual(self.spec(allow_missing_trace=True, require_identity_sha256=self.identity)["families"],
                         spec["families"])
        with self.assertRaisesRegex(items.SpecError, "identity.py hashes"):
            self.spec(allow_missing_trace=True, require_identity_sha256="0" * 64)
        (self.base / "static" / "fam-main" / "locagent.document.json").unlink()
        with self.assertRaisesRegex(items.SpecError, "fam-main: the LocAgent document .* is missing"):
            self.spec(allow_missing_trace=True)  # D12.7 cannot be applied without it

    def test_main_spec_pins_the_canonical_trace_input(self):
        # Review finding: nothing pinned the D12.3 trace input; a main spec pointed at trace-2 (the determinism
        # audit) or at the superseded first attempt built without complaint.
        for name in ("fam-main", "fam-main2", "fam-dev"):
            self.write_trace(name, subdir="trace-2")
        for options in ({"trace_subdir": "trace-2"}, {"trace_subdir": None}, {"trace_root": None}):
            with self.subTest(options=options), self.assertRaisesRegex(items.SpecError, "role main needs"):
                self.spec(**options)
        dev = self.spec(roles=["dev"], trace_subdir="trace-2")  # dev builds are not pinned (pilot only)
        self.assertEqual(dev["families"][0]["trace_dir"], str(self.base / "traces" / "fam-dev" / "trace-2"))
        self.assertIsNone(dev["main_trace_pin"])
        pinned = self.spec()
        self.assertEqual(pinned["main_trace_pin"], {"trace_subdir": "trace-1", "code_sha256": {
            "absent_edges/probe/tracer.py": self.tracer, "absent_edges/probe/identity.py": self.identity}})
        # A trace written by another tracer.py (the superseded first attempt used the 2adb0b458 tracer).
        self.write_trace("fam-main2", plugin_files={"absent_edges/probe/identity.py": self.identity,
                                                    "absent_edges/probe/tracer.py": "9" * 64})
        with self.assertRaisesRegex(items.SpecError, "fam-main2: the trace manifest records "
                                                     "absent_edges/probe/tracer.py 9{64}, not the frozen"):
            self.spec()
        self.write_trace("fam-main2", plugin_files={"absent_edges/probe/tracer.py": self.tracer,
                                                    "absent_edges/probe/identity.py": "8" * 64})
        with self.assertRaisesRegex(items.SpecError, "identity.py"):
            self.spec()
        self.write_trace("fam-main2")
        self.assertEqual([entry["family"] for entry in self.spec()["families"]], ["fam-dev", "fam-main", "fam-main2"])
        with self.assertRaises(SystemExit):
            run_cli("spec", "--registry", str(self.registry_path), "--repos-root", str(self.base / "repos"),
                    "--static-root", str(self.base / "static"), "--trace-root", str(self.base / "traces"),
                    "--trace-subdir", "trace-2", "--out", str(self.base / "refused.json"))
        self.assertFalse((self.base / "refused.json").exists())

    def test_cli_spec_writes_a_loadable_spec_once(self):
        out = self.base / "spec.json"
        arguments = ["spec", "--registry", str(self.registry_path), "--repos-root", str(self.base / "repos"),
                     "--static-root", str(self.base / "static"), "--trace-root", str(self.base / "traces"),
                     "--trace-subdir", "trace-1", "--out", str(out)]
        summary = json.loads(run_cli(*arguments))
        self.assertEqual(summary["families"], ["fam-dev", "fam-main", "fam-main2"])
        families, roster = items.load_build_spec(out)
        self.assertEqual(([family.family for family in families], roster),
                         (["fam-dev", "fam-main", "fam-main2"], ["fam-main", "fam-main2"]))
        self.assertEqual({family.role for family in families}, {"main", "dev"})
        with self.assertRaises(SystemExit):
            run_cli(*arguments)


if __name__ == "__main__":
    unittest.main()
