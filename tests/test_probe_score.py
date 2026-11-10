"""Exp A scoring v2 (D7, D10.11, D10.14, D11.3-D11.5): site categories, outcomes, policies, cell states, binding.

Everything here is a hand-made fixture: a tiny repository, hand-written expA-item-v2
records, a payload store, a dispatch view, run.json and expA-episode-result-v2 files.
Expected values are computed by hand in the comments.
"""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest

from absent_edges.data import canonical_json
from absent_edges.probe import score as S

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "pkg/core.py": ("def target(x):\n"                # 1
                    "    return x\n"                  # 2
                    "\n"                              # 3
                    "\n"                              # 4
                    "def caller_one():\n"             # 5
                    "    return target(1)\n"          # 6
                    "\n"                              # 7
                    "\n"                              # 8
                    "class Box:\n"                    # 9
                    "    def method(self):\n"         # 10
                    "        value = target(\n"       # 11
                    "            2,\n"                # 12
                    "        )\n"                     # 13
                    "        return value\n"          # 14
                    "\n"                              # 15
                    "\n"                              # 16
                    "VALUE = target(3)\n"             # 17
                    "\n"                              # 18
                    "\n"                              # 19
                    "def lexical_only():\n"           # 20
                    "    # target is named here\n"    # 21
                    "    return None\n"),             # 22
    "pkg/other.py": ("from pkg.core import target as aliased\n"   # 1
                     "\n"                                          # 2
                     "\n"                                          # 3
                     "def via_alias():\n"                          # 4
                     "    result = aliased(3)\n"                   # 5
                     "    return result\n"                         # 6
                     "\n"                                          # 7
                     "\n"                                          # 8
                     "def outer():\n"                              # 9
                     "    def inner():\n"                          # 10
                     "        return aliased(4)\n"                 # 11
                     "    return inner\n"),                        # 12
}
TARGET = "pkg/core.py::target"
CALLER, METHOD, ALIAS, INNER = ("pkg/core.py::caller_one", "pkg/core.py::Box.method", "pkg/other.py::via_alias",
                                "pkg/other.py::outer.<locals>.inner")
LEXICAL = [("pkg/core.py", 1), ("pkg/core.py", 6), ("pkg/core.py", 11), ("pkg/core.py", 17), ("pkg/core.py", 21),
           ("pkg/other.py", 1), ("pkg/other.py", 5), ("pkg/other.py", 11)]
HITS = [("pkg/core.py", 1), ("pkg/core.py", 6), ("pkg/other.py", 1), ("pkg/other.py", 5)]
ANSWERS = {"vcomplete": [METHOD, CALLER, ALIAS], "omit_node": [METHOD, CALLER], "omit_edge": [METHOD, CALLER],
           "natural": [CALLER, INNER]}
P_CONDITIONS = ("C1", "C2", "C3", "C4a", "C0")


def site(path, lineno, end=None, kind="call", witnesses=("trace", "ast")):
    return {"path": path, "lineno": lineno, "end_lineno": end or lineno, "kind": kind, "witnesses": list(witnesses)}


def make_payloads(pair, hits=HITS):
    """{variant: {condition: payload}} for one item."""
    out = {}
    for variant, sources in ANSWERS.items():
        c1 = {"query": {"symbol": TARGET, "relation": "invokes", "direction": "incoming"},
              "matches": [{"id": hashlib.sha256(s.encode()).hexdigest()[:24], "source": s, "target": TARGET,
                           "relation": "invokes"} for s in sources],
              "endpoints": [{"id": s} for s in sorted(set(sources) | {TARGET} | ({ALIAS} if variant == "omit_edge"
                                                                                  else set()))],
              "payload_truncated": False}
        out[variant] = {"C1": c1, "C2": dict(c1, guidance=f"caution {pair}"),
                        "C3": dict(c1, guidance=f"not exhaustive {pair}"),
                        "C4a": dict(c1, verification={"kind": S.VERIFICATION_KIND, "query": TARGET,
                                                      "hits": [{"path": p, "line": n, "text": "x"} for p, n in hits]})}
    return out


def payload_sha(payload):
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def make_item(family, item_id, source_sha256, *, in_nat=True, in_edge=True, in_t=True, omitted=ALIAS, pair="A",
              role="main"):
    payloads = make_payloads(pair)
    record = {
        "schema": S.ITEM_SCHEMA, "item_id": item_id, "family": family, "role": role, "package_dirs": ["pkg"],
        "target": {"identity": TARGET, "path": "pkg/core.py", "line": 1, "end_line": 2, "qualname": "target",
                   "name": "target"},
        "V": [{"identity": CALLER, "sources": ["trace", "L", "J"], "runtime_witnessed": True, "stratum": "direct",
               "sites": [site("pkg/core.py", 6)]},
              {"identity": METHOD, "sources": ["L", "J"], "runtime_witnessed": False, "stratum": "direct",
               "sites": [site("pkg/core.py", 11, 13, witnesses=("L", "J", "ast"))]},
              {"identity": ALIAS, "sources": ["trace"], "runtime_witnessed": True, "stratum": "alias",
               "sites": [site("pkg/other.py", 5)]}],
        "U": [{"identity": INNER, "sources": ["L"], "self": False, "sites": [site("pkg/other.py", 11, witnesses=["L"])]}],
        "omitted": {"identity": omitted, "stratum": "alias", "rule": "first_runtime_witnessed", "order_key": "0" * 64},
        "wording_pair": pair,
        "covariates": {"stratum": "alias", "c4_contains_omitted_site": True, "c4_trigger_fires": True,
                       "natural_miss": True, "n_V": 3, "visibility_text_matches": 0},
        "membership": {"in_s_nat": in_nat, "in_s_edge": in_edge, "in_t": in_t,
                       "s_nat_excluded_reason": None if in_nat else "natural_render_truncated"},
        "lexical_sites": [{"path": p, "line": n} for p, n in LEXICAL],
        "render": {"source_sha256": source_sha256, "snippet_caps": {"target": 2048, "caller": 2048}, "cap_bytes": 12000},
        "variants": {variant: {"facts_sha256": "f" * 64, "endpoint_ids": [], "payload_truncated": False,
                               "cells": {c: payload_sha(p) for c, p in cells.items()},
                               "cell_bytes": {c: len(canonical_json(p)) for c, p in cells.items()}}
                     for variant, cells in payloads.items()},
        "eligibility": {"eligible": True}}
    return record, payloads


class Fixture:
    """Two families with one item each, payload store, dispatch view and an empty run directory."""

    def __init__(self, base: Path, run_json_extra=None):
        self.base = base
        self.repos = base / "repos"
        for family in ("fam-a", "fam-b"):
            for relative, text in FILES.items():
                path = self.repos / family / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
        self.tree = S.package_tree_sha256(self.repos / "fam-a", ["pkg"])[0]
        self.records, self.payloads = {}, {}
        for family, item_id, extra in (("fam-a", "item-a", {}),
                                       ("fam-b", "item-b", {"in_nat": False, "in_edge": False, "in_t": False,
                                                            "pair": "B"})):
            record, payloads = make_item(family, item_id, self.tree, **extra)
            self.records[item_id], self.payloads[item_id] = record, payloads
        self.items_dir = base / "items"
        (self.items_dir / "payloads").mkdir(parents=True)
        self.write_items(self.items_dir / "items.jsonl", self.records)
        for item_id, record in self.records.items():
            rows = []
            for cells in self.payloads[item_id].values():
                for payload in cells.values():
                    data = canonical_json(payload)
                    rows.append(canonical_json({"sha256": payload_sha(payload), "bytes": len(data), "payload": payload}))
            (self.items_dir / "payloads" / f"{record['family']}.jsonl").write_bytes(b"\n".join(rows) + b"\n")
        self.items_sha = hashlib.sha256((self.items_dir / "items.jsonl").read_bytes()).hexdigest()
        self.dispatch_dir = base / "dispatch"
        self.dispatch_dir.mkdir()
        self.write_dispatch()
        self.run = base / "run"
        (self.run / "cells").mkdir(parents=True)
        self.run_json_extra = run_json_extra or {}
        self.rebind_run()
        self.checkouts = {"fam-a": str(self.repos / "fam-a"), "fam-b": str(self.repos / "fam-b")}

    def write_dispatch(self, mutate=None):
        rows = [{"item_id": r["item_id"], "family": r["family"],
                 "target": {k: r["target"][k] for k in ("identity", "path", "line", "qualname")},
                 "package_dirs": r["package_dirs"], "wording_pair": r["wording_pair"],
                 "membership": r["membership"],
                 "cells": {v: dict(info["cells"]) for v, info in r["variants"].items()}}
                for r in self.records.values()]
        if mutate:
            mutate(rows)
        path = self.dispatch_dir / "dispatch.jsonl"
        if path.exists():
            os.chmod(path, 0o644)
        path.write_bytes(b"".join(canonical_json(r) + b"\n" for r in rows))
        self.dispatch_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        (self.dispatch_dir / "dispatch_manifest.json").write_bytes(canonical_json({
            "schema": S.DISPATCH_SCHEMA, "items_sha256": self.items_sha, "dispatch_sha256": self.dispatch_sha,
            "payload_store": {f: hashlib.sha256((self.items_dir / "payloads" / f"{f}.jsonl").read_bytes()).hexdigest()
                              for f in ("fam-a", "fam-b")}, "graph_facts": {}, "code": {}}))

    def rebind_run(self):
        """run.json naming the current items and dispatch view (as the runner freezes it)."""
        run_json = {"schema": "expA-run-v2", "run_id": "test", "items_sha256": self.items_sha,
                    "dispatch_sha256": self.dispatch_sha}
        run_json.update(self.run_json_extra)
        (self.run / "run.json").write_bytes(canonical_json(run_json))

    @staticmethod
    def write_items(path, records):
        path.write_bytes(b"".join(canonical_json(records[k]) + b"\n" for k in sorted(records)))

    def result(self, cell_id, ordinal, exit_status, sites=None, complete="no", events_log=None, **override):
        parsed = S.parse_cell_id(cell_id)
        record = self.records[parsed["item_id"]]
        condition, variant = parsed["condition"], parsed["variant"]
        expected = None if condition in S.NO_ANSWER_CONDITIONS else record["variants"][variant]["cells"][condition]
        body = {"schema": S.RESULT_SCHEMA, "cell_id": cell_id, "attempt_id": f"{cell_id}.a{ordinal:03d}", **parsed,
                "family": record["family"], "target": TARGET, "exit_status": exit_status,
                "submitted": {"sites": sites or [], "complete": complete} if exit_status == "Submitted" else None,
                "calls": 3, "tokens": {"prompt": 3000, "completion": 200}, "finish_reason_length_count": 0,
                "tool_calls": {"bash": 2, "submit_answer": int(exit_status == "Submitted")},
                "exposure": {"payload_present": expected is not None, "payload_sha256": expected,
                             "expected_payload_sha256": expected, "payload_tokens": 400 if expected else 0,
                             "verification_block_delivered": condition == "C4a", "verification_sha256": None,
                             "verification_hits": 4 if condition == "C4a" else None,
                             "tools_offered": ["bash", "source_reference_search", "submit_answer"]
                             + (["graph_query"] if condition.startswith("T-") else []),
                             "recommendation_present": condition == "T-recommended"},
                "graph_query_calls": [], "endpoint": "http://127.0.0.1:8141", "gpu_uuid": "GPU-test",
                "dispatch_sha256": self.dispatch_sha, "items_sha256": self.items_sha,
                "code": {"harness.py": "0" * 64}}
        body.update(override)
        directory = self.run / "cells" / cell_id / body["attempt_id"]
        directory.mkdir(parents=True)
        (directory / "attempt_start.json").write_bytes(canonical_json({"cell_id": cell_id, "ordinal": ordinal}))
        if events_log is not None:
            data = b"".join(canonical_json(e) + b"\n" for e in events_log)
            (directory / "events.jsonl").write_bytes(data)
            body.setdefault("events", {"path": "events.jsonl", "sha256": hashlib.sha256(data).hexdigest(),
                                       "records": len(events_log)})
        (directory / "result.json").write_bytes(canonical_json(body) + b"\n")
        return body

    def start_only(self, cell_id, ordinal):
        directory = self.run / "cells" / cell_id / f"{cell_id}.a{ordinal:03d}"
        directory.mkdir(parents=True)
        (directory / "attempt_start.json").write_bytes(canonical_json({"cell_id": cell_id, "ordinal": ordinal}))

    def loaded(self, items_path=None):
        items, meta = S.load_items(items_path or self.items_dir / "items.jsonl")
        return (items, meta, S.PayloadStore(self.items_dir / "payloads"), S.index_factory(self.checkouts),
                S.load_dispatch(self.dispatch_dir / "dispatch.jsonl", self.dispatch_dir / "dispatch_manifest.json"))

    def score(self, **kwargs):
        items, meta, payloads, index, dispatch = self.loaded(kwargs.pop("items_path", None))
        return S.score_run(self.run, items, meta, index, dispatch=dispatch, payloads=payloads, **kwargs)


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SiteRuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.fixture = Fixture(Path(cls.temporary.name))
        items, _, payloads, cls.index_for, _ = cls.fixture.loaded()
        cls.item = items["item-a"]
        S.attach_natural_answer(cls.item, payloads)
        cls.index = cls.index_for("fam-a")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def score(self, sites, complete="no"):
        return S.score_submission(sites, complete, self.item, self.index)

    def test_path_and_line_normalization(self):
        cases = {"/repo/pkg/core.py": ("pkg/core.py", None), "./pkg/core.py": ("pkg/core.py", None),
                 "././pkg//./core.py": ("pkg/core.py", None), "pkg\\core.py": ("pkg/core.py", None),
                 "  pkg/core.py ": ("pkg/core.py", None), "/etc/passwd": ("/etc/passwd", "absolute_path"),
                 "C:/repo/pkg/core.py": ("C:/repo/pkg/core.py", "absolute_path"),
                 "../pkg/core.py": ("../pkg/core.py", "parent_reference"),
                 "pkg/../pkg/core.py": ("pkg/../pkg/core.py", "parent_reference"),
                 "": (None, "invalid_path"), "   ": (None, "invalid_path"), None: (None, "invalid_path"),
                 7: (None, "invalid_path"), "pkg/a::b.py": (None, "invalid_path"), "./": (None, "invalid_path")}
        for raw, expected in cases.items():
            self.assertEqual(S.normalize_site_path(raw), expected, raw)
        self.assertEqual(S.normalize_site_path("Pkg/Core.py"), ("Pkg/Core.py", None))  # case-sensitive, no folding
        for raw, expected in ((5, 5), ("5", 5), ("007", 7), (0, None), ("0", None), (-3, None), ("-3", None),
                              ("5a", None), (" 5", None), (5.0, None), (True, None), (None, None), ("", None),
                              ("\u0665", None)):
            self.assertEqual(S.normalize_site_line(raw), expected, raw)

    def test_every_category_and_both_precision_bounds_by_hand(self):
        sites = [{"path": "/repo/pkg/other.py", "line": 5},     # 1 v_site, C* (via_alias) -> site_recovered
                 {"path": "./pkg/other.py", "line": "5"},       # 2 duplicate of 1
                 {"path": "pkg\\core.py", "line": 12},          # 3 v_site, METHOD span 11-13
                 {"path": "pkg/other.py", "line": 11},          # 4 u_site INNER
                 {"path": "pkg/core.py", "line": 21},           # 5 lexical_name_match (comment in lexical_only)
                 {"path": "pkg/core.py", "line": 17},           # 6 lexical_name_match (module-level line)
                 {"path": "pkg/core.py", "line": 2},            # 7 unresolved: not_a_known_site
                 {"path": "/etc/passwd", "line": 1},            # 8 unresolved: absolute_path
                 {"path": "../pkg/core.py", "line": 6},         # 9 unresolved: parent_reference
                 {"path": "tests/test_core.py", "line": 3},     # 10 unresolved: out_of_scope
                 {"path": "pkg/core.py", "line": 0},            # 11 invalid line
                 {"path": "pkg/core.py", "line": "6a"},         # 12 invalid line
                 {"path": "", "line": 6},                       # 13 invalid path
                 {"path": "pkg/core.py", "line": True},         # 14 invalid line (bool)
                 {"path": "pkg/core.py", "line": 6.0},          # 15 invalid line (float)
                 "pkg/core.py:6"]                               # 16 invalid (not an object)
        out = self.score(sites, complete="yes")
        self.assertEqual([d["category"] for d in out["site_details"]],
                         ["v_site", "duplicate", "v_site", "u_site", "lexical_name_match", "lexical_name_match"]
                         + ["unresolved"] * 10)
        self.assertEqual(out["precision_categories"], {"v_site": 2, "u_site": 1, "lexical_name_match": 2,
                                                       "unresolved": 10, "duplicate": 1})
        self.assertEqual(out["unresolved_reasons"], {"invalid_path": 2, "invalid_line": 4, "absolute_path": 1,
                                                     "parent_reference": 1, "out_of_scope": 1, "not_a_known_site": 1})
        self.assertEqual((out["n_submitted"], out["n_invalid"], out["n_unique_sites"]), (16, 6, 9))
        self.assertAlmostEqual(out["precision_lower"], 2 / 16)          # V-sites / all entries
        self.assertAlmostEqual(out["precision_upper"], 5 / 9)           # (V + U + lexical) / unique valid
        self.assertEqual(out["site_recovered"], 1)
        self.assertEqual(out["function_localized"], 1)
        self.assertEqual(out["recovered_V"], sorted([ALIAS, METHOD]))
        self.assertAlmostEqual(out["recall_V_callers"], 2 / 3)
        self.assertEqual(out["recall_V_sites"], out["recall_V_callers"])
        self.assertTrue(out["certifiably_false_completeness"])          # CALLER (runtime-witnessed) missed
        self.assertFalse(out["uncertain_despite_full_V"])
        self.assertEqual((out["nm_recall"], out["nm_any"], out["n_NM"]), (1.0, 1, 1))  # NM = {ALIAS}

    def test_empty_and_degenerate_submissions(self):
        out = self.score([], complete="unknown")
        self.assertEqual((out["n_submitted"], out["precision_lower"], out["precision_upper"]), (0, None, None))
        self.assertEqual((out["site_recovered"], out["recall_V_callers"], out["nm_any"]), (0, 0.0, 0))
        out = self.score([{"path": "", "line": 1}] * 2)
        self.assertEqual((out["n_submitted"], out["n_unique_sites"], out["precision_lower"], out["precision_upper"]),
                         (2, 0, 0.0, None))
        self.assertEqual(out["precision_categories"]["duplicate"], 0)   # invalid entries are never duplicates

    def test_function_localized_without_site_recovered(self):
        out = self.score([{"path": "pkg/other.py", "line": 6}])         # inside via_alias, not at its site
        self.assertEqual((out["site_recovered"], out["function_localized"]), (0, 1))
        self.assertEqual(out["site_details"][0]["reason"], "not_a_known_site")
        out = self.score([{"path": "pkg/other.py", "line": 4}])         # the def line: no enclosing body
        self.assertEqual((out["site_recovered"], out["function_localized"]), (0, 0))

    def test_completeness_flags_use_runtime_witnessed_callers(self):
        caller, method, alias = ({"path": "pkg/core.py", "line": 6}, {"path": "pkg/core.py", "line": 13},
                                 {"path": "pkg/other.py", "line": 5})
        out = self.score([caller, alias], complete="yes")                # METHOD missed but not runtime-witnessed
        self.assertFalse(out["certifiably_false_completeness"])
        self.assertEqual(out["runtime_witnessed_missed"], [])
        out = self.score([method, alias], complete="yes")                # CALLER missed and runtime-witnessed
        self.assertTrue(out["certifiably_false_completeness"])
        out = self.score([caller, method, alias], complete="unknown")
        self.assertTrue(out["uncertain_despite_full_V"])
        self.assertFalse(out["certifiably_false_completeness"])
        out = self.score([caller, method, alias], complete="yes")
        self.assertEqual((out["certifiably_false_completeness"], out["uncertain_despite_full_V"]), (False, False))
        out = self.score([caller], complete="maybe")
        self.assertEqual((out["complete"], out["certifiably_false_completeness"]), (None, None))

    def test_item_validation(self):
        good = self.fixture.records["item-a"]
        for mutate, pattern in (
                (lambda r: r.update(schema="expA-item-v1"), "schema"),
                (lambda r: r["omitted"].update(identity=INNER), "omitted caller must be in V"),
                (lambda r: r["omitted"].update(identity=METHOD), "runtime-witnessed"),
                (lambda r: r["V"][0].update(runtime_witnessed=None), "runtime_witnessed"),
                (lambda r: r["V"][2]["sites"].clear(), "no site span"),
                (lambda r: r["V"][0]["sites"][0].update(kind="import"), "kind"),
                (lambda r: r["V"][0]["sites"][0].update(end_lineno=1), "lineno"),
                (lambda r: r["covariates"].update(stratum="indirect"), "stratum"),
                (lambda r: r.update(wording_pair="C"), "wording_pair"),
                (lambda r: r["variants"].pop("natural"), "variants"),
                (lambda r: r["target"].update(qualname="other"), "disagree"),
                (lambda r: r["U"].append(dict(r["V"][0])), "overlaps V")):
            record = copy.deepcopy(good)
            mutate(record)
            with self.assertRaisesRegex(S.ScoreInputError, pattern):
                S.ItemTruth(record)


class PolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.fixture = Fixture(Path(cls.temporary.name))
        items, meta, cls.payloads, cls.index, _ = cls.fixture.loaded()
        cls.items = items
        cls.rows = S.policy_rows(items, cls.payloads, cls.index, items_sha256=meta["sha256"])
        cls.by_key = {(r["item_id"], r["variant"], r["policy"]): r for r in cls.rows}

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_policy_submissions_by_hand(self):
        subs = S.policy_submissions(self.items["item-a"], self.payloads)
        sites = lambda variant, policy: [(s["path"], s["line"]) for s in subs[variant][policy][0]]  # noqa: E731
        self.assertEqual(sites("omit_node", "answer_only"), [("pkg/core.py", 11), ("pkg/core.py", 6)])
        self.assertEqual(sites("vcomplete", "answer_only"), [("pkg/core.py", 11), ("pkg/core.py", 6),
                                                             ("pkg/other.py", 5)])
        self.assertEqual(sites("natural", "answer_only"), [("pkg/core.py", 6), ("pkg/other.py", 11)])
        self.assertEqual(sites("none", "answer_only"), [])
        self.assertEqual(sites("omit_node", "c4_hits"), HITS)
        self.assertEqual(sites("omit_node", "answer_plus_c4"), [("pkg/core.py", 11), ("pkg/core.py", 6),
                                                                ("pkg/core.py", 1), ("pkg/other.py", 1),
                                                                ("pkg/other.py", 5)])
        self.assertEqual(sites("omit_edge", "grep_all"), LEXICAL)
        self.assertFalse(subs["omit_node"]["answer_only"][1]["omitted_in_answer"])
        self.assertTrue(subs["vcomplete"]["answer_only"][1]["omitted_in_answer"])

    def test_policy_scores_by_hand(self):
        expected = {  # (site_recovered, recall, precision_lower, precision_upper, categories v/u/lex)
            ("omit_node", "answer_only"): (0, 2 / 3, 1.0, 1.0, (2, 0, 0)),
            ("omit_node", "c4_hits"): (1, 2 / 3, 2 / 4, 1.0, (2, 0, 2)),
            ("omit_node", "answer_plus_c4"): (1, 1.0, 3 / 5, 1.0, (3, 0, 2)),
            ("omit_node", "grep_all"): (1, 1.0, 3 / 8, 1.0, (3, 1, 4)),
            ("vcomplete", "answer_only"): (1, 1.0, 1.0, 1.0, (3, 0, 0)),
            ("natural", "answer_only"): (0, 1 / 3, 1 / 2, 1.0, (1, 1, 0)),
            ("none", "answer_only"): (0, 0.0, None, None, (0, 0, 0)),
            ("none", "c4_hits"): (1, 2 / 3, 2 / 4, 1.0, (2, 0, 2)),
        }
        for (variant, policy), (recovered, recall, low, high, cats) in expected.items():
            row = self.by_key[("item-a", variant, policy)]
            self.assertEqual(row["site_recovered"], recovered, (variant, policy))
            self.assertAlmostEqual(row["recall_V_callers"], recall, msg=(variant, policy))
            self.assertEqual(row["precision_lower"] if low is None else round(row["precision_lower"], 9),
                             low if low is None else round(low, 9), (variant, policy))
            self.assertEqual(row["precision_upper"], high, (variant, policy))
            got = row["precision_categories"]
            self.assertEqual((got["v_site"], got["u_site"], got["lexical_name_match"]), cats, (variant, policy))
        natural = self.by_key[("item-a", "natural", "answer_only")]
        self.assertEqual((natural["nm_recall"], natural["nm_any"]), (0.0, 0))   # NM = {ALIAS}, not in the answer
        grep = self.by_key[("item-a", "natural", "grep_all")]
        self.assertEqual((grep["nm_recall"], grep["nm_any"]), (1.0, 1))

    def test_rows_cover_every_item_and_variant(self):
        self.assertEqual(len(self.rows), 2 * 5 * 4)
        self.assertEqual({r["schema"] for r in self.rows}, {S.SCORE_SCHEMA})
        summary = S.summarize_policies(self.rows)
        self.assertEqual(summary["main/omit_node/c4_hits"]["site_recovered"], 2)
        self.assertEqual(summary["main/omit_node/answer_only"]["site_recovered"], 0)

    def test_verification_blocks_must_agree_across_variants(self):
        item = copy.deepcopy(self.items["item-a"])
        store = copy.deepcopy(self.payloads)
        other = make_payloads("A", hits=HITS[:2])["omit_edge"]["C4a"]
        store._families["fam-a"][payload_sha(other)] = other
        item.variants = copy.deepcopy(item.variants)
        item.variants["omit_edge"]["cells"]["C4a"] = payload_sha(other)
        with self.assertRaisesRegex(S.BindingError, "verification blocks differ"):
            S.item_verification_hits(item, store)


class RunTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = Fixture(Path(self.temporary.name))

    def tearDown(self):
        self.temporary.cleanup()

    def populate(self):
        f = self.fixture
        f.result("P|item-a|omit_node|C1|s11", 1, "Submitted", [{"path": "pkg/other.py", "line": 5}], "yes")
        f.result("P|item-a|omit_node|C2|s11", 1, "NoAnswer")
        f.result("P|item-a|none|C0|s11", 1, "Submitted", [{"path": "pkg/core.py", "line": 6}], "unknown")
        for ordinal in (1, 2, 3):
            f.result("P|item-a|omit_node|C3|s11", ordinal, "InfraInterrupted", exposure=None)
        f.result("P|item-a|omit_node|C4a|s11", 1, "InfraInterrupted", exposure=None)
        f.result("P|item-a|omit_node|C4a|s11", 2, "Submitted", [{"path": "pkg/other.py", "line": 5}], "no")
        f.start_only("P|item-b|omit_node|C1|s11", 1)
        (f.run / "cells" / "P|item-b|omit_node|C2|s11" / "P|item-b|omit_node|C2|s11.a001").mkdir(parents=True)
        f.result("T|item-a|none|T-offered|s11", 1, "Submitted", [], "unknown",
                 graph_query_calls=[{"status": "ok", "returned_count": 2}],
                 tool_calls={"graph_query": 1, "submit_answer": 1})

    def test_rows_for_every_design_cell_with_states(self):
        self.populate()
        rows, audit = self.fixture.score()
        by = {r["cell_id"]: r for r in rows}
        # d11: P 5 conditions x 3 seeds x 2 items + S-VC 2 x 2 + S-NAT 3 + S-EDGE 2 + T 2 (item-a only)
        self.assertEqual(len(rows), 30 + 4 + 3 + 2 + 2)
        self.assertEqual(audit["design"], "d11")
        self.assertEqual(audit["empty_attempt_directories"], 1)
        c1 = by["P|item-a|omit_node|C1|s11"]
        self.assertEqual((c1["cell_state"], c1["exit_status"], c1["site_recovered"], c1["completed"]),
                         ("completed", "Submitted", 1, True))
        self.assertTrue(c1["certifiably_false_completeness"])
        c2 = by["P|item-a|omit_node|C2|s11"]
        self.assertEqual((c2["cell_state"], c2["exit_status"], c2["site_recovered"], c2["recall_V_callers"]),
                         ("completed", "NoAnswer", 0, 0.0))
        c0 = by["P|item-a|none|C0|s11"]
        self.assertEqual((c0["variant"], c0["site_recovered"], c0["recall_V_callers"]), ("none", 0, 1 / 3))
        c3 = by["P|item-a|omit_node|C3|s11"]
        self.assertEqual((c3["cell_state"], c3["exit_status"], c3["site_recovered"], c3["infra_attempts"]),
                         ("missing_infrastructure", "InfraInterrupted", None, 3))
        c4 = by["P|item-a|omit_node|C4a|s11"]
        self.assertEqual((c4["cell_state"], c4["attempt_id"], c4["infra_attempts"], c4["retry_cap_exceeded"]),
                         ("completed", "P|item-a|omit_node|C4a|s11.a002", 1, False))
        b1 = by["P|item-b|omit_node|C1|s11"]
        self.assertEqual((b1["cell_state"], b1["exit_status"], b1["infra_attempts"]), ("infra_pending",
                                                                                       "InfraInterrupted", 1))
        self.assertEqual(by["P|item-b|omit_node|C2|s11"]["cell_state"], "not_dispatched")
        self.assertEqual(by["P|item-a|omit_node|C1|s23"]["exit_status"], "Missing")
        t = by["T|item-a|none|T-offered|s11"]
        self.assertEqual((t["cost"]["graph_query_calls"], t["cost"]["graph_query_nonempty_answers"]), (1, 1))
        self.assertEqual(audit["cell_states"], {"completed": 5, "infra_pending": 1, "missing_infrastructure": 1,
                                                "not_dispatched": 34})

    def test_retry_cap_exceeded_and_redispatch_after_terminal(self):
        f = self.fixture
        for ordinal in (1, 2, 3):
            f.result("P|item-a|omit_node|C1|s11", ordinal, "InfraInterrupted", exposure=None)
        f.result("P|item-a|omit_node|C1|s11", 4, "NoAnswer")
        rows, audit = f.score()
        row = {r["cell_id"]: r for r in rows}["P|item-a|omit_node|C1|s11"]
        self.assertEqual((row["cell_state"], row["retry_cap_exceeded"], audit["retry_cap_exceeded"]),
                         ("completed", True, 1))
        f.result("P|item-a|omit_node|C2|s11", 1, "NoAnswer")
        f.result("P|item-a|omit_node|C2|s11", 2, "InfraInterrupted", exposure=None)
        with self.assertRaisesRegex(S.BindingError, "after the terminal attempt"):
            f.score()

    def binding_refused(self, pattern, **kwargs):
        with self.assertRaisesRegex(S.BindingError, pattern):
            self.fixture.score(**kwargs)

    def test_rebuilt_items_with_a_different_omitted_caller_are_refused(self):
        f = self.fixture
        f.result("P|item-a|omit_node|C1|s11", 1, "Submitted", [{"path": "pkg/other.py", "line": 5}], "no")
        rows, _ = f.score()
        self.assertEqual({r["cell_id"]: r for r in rows}["P|item-a|omit_node|C1|s11"]["site_recovered"], 1)
        rebuilt = copy.deepcopy(f.records)
        rebuilt["item-a"]["omitted"]["identity"] = CALLER    # same item ids, another C*
        path = f.base / "rebuilt" / "items.jsonl"
        path.parent.mkdir()
        f.write_items(path, rebuilt)
        self.binding_refused("differs from run.json", items_path=path)
        new_sha = hashlib.sha256(path.read_bytes()).hexdigest()   # re-pointing run.json is not enough
        (f.run / "run.json").write_bytes(canonical_json({"items_sha256": new_sha, "dispatch_sha256": f.dispatch_sha}))
        self.binding_refused("dispatch manifest", items_path=path)
        manifest = json.loads((f.dispatch_dir / "dispatch_manifest.json").read_bytes())
        manifest["items_sha256"] = new_sha                         # nor is re-pointing the dispatch manifest
        (f.dispatch_dir / "dispatch_manifest.json").write_bytes(canonical_json(manifest))
        self.binding_refused("result items_sha256", items_path=path)

    def test_result_payload_must_be_the_items_payload(self):
        f = self.fixture
        wrong = f.records["item-a"]["variants"]["omit_node"]["cells"]["C2"]
        f.result("P|item-a|omit_node|C1|s11", 1, "NoAnswer",
                 exposure={"payload_present": True, "payload_sha256": wrong, "expected_payload_sha256": wrong})
        self.binding_refused("differ from the item")

    def test_answerless_cells_must_not_carry_a_payload(self):
        f = self.fixture
        some = f.records["item-a"]["variants"]["omit_node"]["cells"]["C1"]
        f.result("P|item-a|none|C0|s11", 1, "NoAnswer",
                 exposure={"payload_present": True, "payload_sha256": some, "expected_payload_sha256": None})
        self.binding_refused("must not deliver an answer payload")

    def test_result_target_family_and_hashes_must_match(self):
        for override, pattern in (({"target": "pkg/core.py::caller_one"}, "target"),
                                  ({"family": "fam-b"}, "family"),
                                  ({"dispatch_sha256": "0" * 64}, "dispatch view"),
                                  ({"items_sha256": "1" * 64}, "result items_sha256")):
            with self.subTest(override=override):
                self.tearDown()
                self.setUp()
                self.fixture.result("P|item-a|omit_node|C1|s11", 1, "NoAnswer", **override)
                self.binding_refused(pattern)

    def test_dispatch_view_and_payload_store_must_match_the_items(self):
        f = self.fixture
        f.write_dispatch(lambda rows: rows[0].update(wording_pair="B"))
        self.binding_refused("run.json dispatch_sha256")             # the run names the original view
        f.rebind_run()
        self.binding_refused("wording_pair")
        f.write_dispatch(lambda rows: rows[0].update(V=[]))
        with self.assertRaisesRegex(S.BindingError, "truth fields"):
            f.loaded()
        f.write_dispatch(lambda rows: rows[0]["cells"]["omit_node"].update(C1="2" * 64))
        f.rebind_run()
        self.binding_refused("payload differs")
        f.write_dispatch()
        f.rebind_run()
        store = f.items_dir / "payloads" / "fam-b.jsonl"
        store.write_bytes(store.read_bytes() + b"\n")               # same rows, different file bytes
        self.binding_refused("payload store fam-b.jsonl")

    def test_a_result_must_belong_to_the_start_record_of_its_attempt(self):
        # Review finding (scenario_reboot): after a relaunch on a wiped run root, post-reboot results were published
        # next to pre-reboot attempt_start.json files and scored as completed with infra_attempts 0.
        f = self.fixture
        cell = "P|item-a|omit_node|C1|s11"
        start = canonical_json({"cell_id": cell, "ordinal": 1})
        f.result(cell, 1, "Submitted", [{"path": "pkg/other.py", "line": 5}], "no",
                 run_context={"launch": "l1", "start_sha256": hashlib.sha256(start).hexdigest()})
        rows, _ = f.score()
        self.assertEqual({r["cell_id"]: r for r in rows}[cell]["cell_state"], "completed")
        f.result("P|item-a|omit_node|C2|s11", 1, "Submitted", [], "no",
                 run_context={"launch": "l2", "start_sha256": "7" * 64})
        self.binding_refused("records of two launches are mixed in one attempt")

    def test_a_dispatched_subset_store_is_bound_through_items_payload_store(self):
        # D10.7 / review finding: the dispatch view ships only the dispatched cells; scoring reads the full items
        # store, which the dispatch manifest pins as items_payload_store, while run.json pins the staged subset.
        f = self.fixture
        manifest_path = f.dispatch_dir / "dispatch_manifest.json"
        manifest = json.loads(manifest_path.read_bytes())
        full = dict(manifest["payload_store"])
        staged = {family: str(k + 5) * 64 for k, family in enumerate(sorted(full))}
        manifest["items_payload_store"] = {family: {"sha256": value, "rows": 9} for family, value in full.items()}
        manifest["payload_store"] = {family: {"path": f"payloads/{family}.jsonl", "sha256": value}
                                     for family, value in staged.items()}
        manifest_path.write_bytes(canonical_json(manifest))
        f.run_json_extra = {"payload_store": dict(staged)}
        f.rebind_run()
        _, audit = f.score()
        self.assertEqual((audit["payload_store_sha256"], audit["dispatched_payload_store_sha256"]), (full, staged))
        f.run_json_extra = {"payload_store": dict(full)}           # a run that staged another store
        f.rebind_run()
        self.binding_refused("not the dispatch manifest's dispatched store")
        f.run_json_extra = {"payload_store": dict(staged)}
        f.rebind_run()
        manifest["items_payload_store"]["fam-b"]["sha256"] = "9" * 64
        manifest_path.write_bytes(canonical_json(manifest))
        self.binding_refused("items_payload_store")

    def test_cells_outside_the_design_and_frozen_designs(self):
        f = self.fixture
        f.result("S-C0|item-a|none|C0|s11", 1, "NoAnswer")
        f.result("P|item-a|none|C0|s11", 1, "NoAnswer")
        self.binding_refused("outside the frozen")                  # S-C0 names contract-v2; P C0 is outside it
        self.tearDown()
        self.setUp()
        f = self.fixture
        (f.run / "cells.jsonl").write_bytes(canonical_json({"cell_id": "P|item-a|omit_node|C1|s11"}) + b"\n")
        self.binding_refused("froze another design")
        self.tearDown()
        self.setUp()
        f = self.fixture
        f.result("P|item-a|none|C0|s11", 1, "NoAnswer")
        self.binding_refused("different designs", design="contract-v2")
        with self.assertRaisesRegex(S.ScoreInputError, "not a v2 cell id"):
            S.parse_cell_id("P|item-a|omit_node|C4|s11")

    def test_contract_v2_layout_is_scored_when_the_run_names_it(self):
        self.tearDown()
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = Fixture(Path(self.temporary.name), run_json_extra={"design": {"blocks": {
            "P": {"variant": "omit_node", "conditions": ["C1", "C2", "C3", "C4a"], "seeds": [11, 23, 47]},
            "S-C0": {"variant": "none", "conditions": ["C0"], "seeds": [11]}}}})
        self.fixture.result("S-C0|item-b|none|C0|s11", 1, "Submitted", [{"path": "pkg/other.py", "line": 5}], "no")
        rows, audit = self.fixture.score()
        self.assertEqual(audit["design"], "contract-v2")
        # P 4 x 3 x 2 + S-C0 2 + S-VC 4 + S-NAT 3 + S-EDGE 2 + T 2
        self.assertEqual(len(rows), 24 + 2 + 4 + 3 + 2 + 2)
        self.assertEqual({r["cell_id"]: r for r in rows}["S-C0|item-b|none|C0|s11"]["site_recovered"], 1)

    def test_telemetry_from_events_and_events_hash(self):
        f = self.fixture
        events = [{"sequence": 0, "kind": "tool_action", "value": {"id": "a", "name": "bash",
                                                                    "arguments": {"command": "grep -rn aliased pkg"}}},
                  {"sequence": 1, "kind": "tool_action", "value": {"id": "b", "name": "source_reference_search",
                                                                    "arguments": {"query": "target"}}},
                  {"sequence": 2, "kind": "tool_action", "value": {"id": "c", "name": "bash",
                                                                    "arguments": {"command": "sed -n 1,9p pkg/other.py"}}},
                  {"sequence": 3, "kind": "tool_action", "value": {"id": "d", "name": "submit_answer",
                                                                    "arguments": {}}},
                  {"sequence": 4, "kind": "submission_accepted", "value": {}}]
        f.result("P|item-a|omit_node|C1|s11", 1, "Submitted", [], "no", events_log=events)
        rows, _ = f.score()
        telemetry = {r["cell_id"]: r for r in rows}["P|item-a|omit_node|C1|s11"]["telemetry"]
        self.assertEqual((telemetry["searched_target_name"], telemetry["mentioned_omitted_caller_file"],
                          telemetry["tool_calls_before_submit"]), (True, True, 3))
        f.result("P|item-a|omit_node|C2|s11", 1, "NoAnswer", events_log=events[:1], events={"sha256": "3" * 64})
        self.binding_refused("events.jsonl sha256")


class CommandTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = Fixture(Path(self.temporary.name))
        self.cli = _load_script("expA_score")

    def tearDown(self):
        for current, _, files in os.walk(self.temporary.name):
            for name in files:
                os.chmod(Path(current) / name, stat.S_IWRITE | stat.S_IREAD)
        self.temporary.cleanup()

    def run_cli(self, *args):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.cli.main([str(a) for a in args])

    def common(self):
        f = self.fixture
        return ["--items", f.items_dir / "items.jsonl", "--repos-root", f.repos, "--dispatch-dir", f.dispatch_dir]

    def test_score_and_policies_write_immutable_bound_outputs(self):
        f = self.fixture
        f.result("P|item-a|omit_node|C1|s11", 1, "Submitted", [{"path": "pkg/other.py", "line": 5}], "no")
        out = f.base / "out" / "scores.jsonl"
        self.assertEqual(self.run_cli("score", "--run-dir", f.run, *self.common(), "--out", out), 0)
        manifest = json.loads((out.parent / "scores.jsonl.manifest.json").read_bytes())
        self.assertEqual(manifest["out"]["sha256"], hashlib.sha256(out.read_bytes()).hexdigest())
        self.assertEqual((manifest["items"]["sha256"], manifest["design"]["name"]), (f.items_sha, "d11"))
        self.assertEqual(manifest["out"]["rows"], 41)
        self.assertFalse(os.stat(out).st_mode & stat.S_IWUSR)
        with self.assertRaises(FileExistsError):
            self.run_cli("score", "--run-dir", f.run, *self.common(), "--out", out)
        policies = f.base / "out" / "policies.jsonl"
        self.assertEqual(self.run_cli("policies", *self.common(), "--out", policies), 0)
        rows = [json.loads(line) for line in policies.read_text().splitlines()]
        self.assertEqual(len(rows), 40)
        self.assertEqual({r["items_sha256"] for r in rows}, {f.items_sha})

    def test_checkout_must_match_the_items_source_tree(self):
        f = self.fixture
        (f.repos / "fam-b" / "pkg" / "core.py").write_text(FILES["pkg/core.py"] + "\n# edited\n")
        with self.assertRaisesRegex(S.BindingError, "different source"):
            self.run_cli("policies", *self.common(), "--out", f.base / "p.jsonl")

    def test_rebuilt_items_file_is_refused_by_the_command(self):
        f = self.fixture
        rebuilt = copy.deepcopy(f.records)
        rebuilt["item-b"]["omitted"]["identity"] = CALLER
        path = f.base / "rebuilt.jsonl"
        f.write_items(path, rebuilt)
        args = self.common()
        args[1] = path
        with self.assertRaisesRegex(S.BindingError, "differs from run.json"):
            self.run_cli("score", "--run-dir", f.run, *args, "--payloads", f.items_dir / "payloads",
                         "--out", f.base / "s.jsonl")
        self.assertFalse((f.base / "s.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
