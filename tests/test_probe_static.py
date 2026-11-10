"""Exp A static caller backends on synthetic fixture packages.

Unit tests need no engine. Integration tests run the pinned LocAgent (graphs.GraphBackend
via the CLI) and the pinned jedi (CLI under the static env Python) when available, and
skip otherwise. The fixture covers methods, nested functions, an import alias, a module
attribute call, a callback, decorators, module-level calls, a class constructor and
test-directory exclusion.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from absent_edges.probe.identity import FunctionIndex, source_tree_sha256
from absent_edges.probe.static_callers import (
    agreement_summary, definition_name_position, jedi_records, load_edges, locagent_records,
    package_import_root, reference_sites, run_static_backend, stage_package_source)

PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / "scripts" / "expA_static.py"
CORE2 = Path("/mnt/kldata/absent-edges-20261018-core2")
LOCAGENT = Path(os.environ.get("ABSENT_EDGES_LOCAGENT_ROOT", CORE2 / "vendor" / "locagent"))
GRAPH_PYTHON = Path(os.environ.get("ABSENT_EDGES_GRAPH_PYTHON", CORE2 / "envs" / "graph" / "bin" / "python"))
JEDI_PYTHON = Path(os.environ.get("ABSENT_EDGES_JEDI_PYTHON", CORE2 / "expA" / "envs" / "static" / "bin" / "python"))
LOCAGENT_READY = os.name == "posix" and LOCAGENT.is_dir() and GRAPH_PYTHON.exists()
JEDI_READY = os.name == "posix" and JEDI_PYTHON.exists()

CORE = '''\
from pkg import util as util_module
from pkg.util import transform as shifted


def helper(value):
    return value + 1


def direct_caller(value):
    return helper(value)


def alias_caller(value):
    return shifted(value)


def module_attr_caller(value):
    return util_module.transform(value)


def callback_caller(items):
    return list(map(helper, items))


REGISTRY = []


def register(function):
    REGISTRY.append(function)
    return function


@register
def decorated():
    return helper(2)


def decorate_inside():
    @register
    def local():
        return 0
    return local


def outer():
    def inner():
        return helper(3)
    return inner()


class Widget:
    def __init__(self):
        self.value = helper(4)

    def method(self):
        return self.other()

    def other(self):
        return helper(5)


def make_widget():
    return Widget()


def chained(value):
    return (util_module
            .transform(value))


helper(10)
MODULE_ALIAS = helper
'''
UTIL = '''\
def transform(value):
    return value * 2


def uses_core():
    from pkg.core import helper
    return helper(1)
'''
FIXTURE = {
    "pkg/__init__.py": "from .core import helper\n",
    "pkg/core.py": CORE,
    "pkg/util.py": UTIL,
    "pkg/tests/test_core.py": "from pkg.core import helper\n\n\ndef test_helper():\n    assert helper(1) == 2\n",
    "tests/test_top.py": "from pkg.core import direct_caller\n\n\ndef test_direct():\n    assert direct_caller(1) == 2\n",
    "setup.py": "from setuptools import setup\nsetup()\n",
}
C, U = "pkg/core.py::", "pkg/util.py::"


def line_of(source, marker, occurrence=1):
    found = 0
    for number, text in enumerate(source.splitlines(), 1):
        if marker in text:
            found += 1
            if found == occurrence:
                return number
    raise AssertionError(f"marker not found: {marker!r}")


def write_tree(root, files):
    for relpath, text in files.items():
        path = Path(root) / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


# Pairs the pinned LocAgent reports for the fixture: direct, aliased, attribute,
# nested, method and __init__ calls. It reports no callback and no decorator edge.
LOCAGENT_PAIRS = {
    (C + "direct_caller", C + "helper"): "return helper(value)",
    (C + "alias_caller", U + "transform"): "return shifted(value)",
    (C + "module_attr_caller", U + "transform"): "return util_module.transform(value)",
    (C + "decorated", C + "helper"): "return helper(2)",
    (C + "outer", C + "outer.<locals>.inner"): "return inner()",
    (C + "outer.<locals>.inner", C + "helper"): "return helper(3)",
    (C + "Widget.__init__", C + "helper"): "self.value = helper(4)",
    (C + "Widget.method", C + "Widget.other"): "return self.other()",
    (C + "Widget.other", C + "helper"): "return helper(5)",
    (C + "chained", U + "transform"): ".transform(value))",
    (U + "uses_core", C + "helper"): "return helper(1)",
}
# Pairs jedi must resolve; it also keeps callbacks, returns and decorator applications.
JEDI_REQUIRED = {
    (C + "direct_caller", C + "helper"): ("call", "return helper(value)"),
    (C + "callback_caller", C + "helper"): ("reference", "return list(map(helper, items))"),
    (C + "decorated", C + "helper"): ("call", "return helper(2)"),
    (C + "decorate_inside", C + "register"): ("decorator", "    @register"),
    (C + "decorate_inside", C + "decorate_inside.<locals>.local"): ("reference", "return local"),
    (C + "outer", C + "outer.<locals>.inner"): ("call", "return inner()"),
    (C + "outer.<locals>.inner", C + "helper"): ("call", "return helper(3)"),
    (C + "Widget.__init__", C + "helper"): ("call", "self.value = helper(4)"),
    (C + "Widget.method", C + "Widget.other"): ("call", "return self.other()"),
    (C + "Widget.other", C + "helper"): ("call", "return helper(5)"),
    (C + "module_attr_caller", U + "transform"): ("call", "return util_module.transform(value)"),
    (C + "chained", U + "transform"): ("call", ".transform(value))"),
    (U + "uses_core", C + "helper"): ("call", "return helper(1)"),
}


def expected_line(caller, marker):
    source = UTIL if caller.startswith("pkg/util.py") else CORE
    occurrence = 2 if marker == "    @register" else 1
    return line_of(source, marker.strip() if occurrence == 1 else "@register", occurrence)


class StagingTests(unittest.TestCase):
    def test_import_root(self):
        self.assertEqual(package_import_root(["boltons"]), "")
        self.assertEqual(package_import_root(["src/attr/"]), "src")
        self.assertEqual(package_import_root(["src/a", "src/b"]), "src")
        self.assertEqual(package_import_root(["src/a", "lib/b"]), "")
        self.assertEqual(package_import_root(["six.py"]), "")

    def test_stage_copies_only_package_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, stage = Path(directory) / "repo", Path(directory) / "stage"
            write_tree(repo, FIXTURE)
            staged = stage_package_source(repo, ["pkg"], stage)
            self.assertEqual(staged["files"], ["pkg/__init__.py", "pkg/core.py", "pkg/util.py"])
            copied = sorted(path.relative_to(stage).as_posix() for path in stage.rglob("*.py"))
            self.assertEqual(copied, staged["files"])
            self.assertEqual(staged["source_tree_sha256"], source_tree_sha256(staged["file_sha256"]))
            self.assertEqual(staged["file_sha256"]["pkg/core.py"], hashlib.sha256(CORE.encode()).hexdigest())
            with self.assertRaises(ValueError):
                stage_package_source(repo, ["tests"], Path(directory) / "empty")


class LocAgentMappingTests(unittest.TestCase):
    """locagent_records on a hand-built document that uses LocAgent's own naming."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        write_tree(self.root, FIXTURE)
        self.index = FunctionIndex(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def node(self, qualified, kind="function", path="pkg/core.py", marker=None):
        source = UTIL if path == "pkg/util.py" else CORE
        start = line_of(source, marker) if marker else None
        return {"id": f"{path}::{qualified}", "kind": kind, "path": path, "start_line": start, "end_line": start}

    def test_mapping_lines_and_drops(self):
        nodes = [
            self.node("helper", marker="def helper"), self.node("direct_caller", marker="def direct_caller"),
            self.node("alias_caller", marker="def alias_caller"), self.node("outer", marker="def outer"),
            self.node("outer.inner", marker="def inner"), self.node("Widget", "class", marker="class Widget"),
            self.node("make_widget", marker="def make_widget"),
            self.node("transform", path="pkg/util.py", marker="def transform"),
            self.node("ghost", marker="def helper"),
            {"id": "pkg/tests/test_core.py::test_helper", "kind": "function", "path": "pkg/tests/test_core.py",
             "start_line": 4, "end_line": 5},
            {"id": "pkg/core.py::<module>", "kind": "file", "path": "pkg/core.py", "start_line": None, "end_line": None},
        ]
        pairs = [("direct_caller", "helper"), ("alias_caller", "pkg/util.py::transform"), ("outer", "outer.inner"),
                 ("outer.inner", "helper"), ("Widget", "helper"), ("make_widget", "Widget"),
                 ("pkg/tests/test_core.py::test_helper", "helper"), ("<module>", "helper"),
                 ("direct_caller", "pkg/util.py::transform"), ("direct_caller", "ghost")]
        edges = []
        for number, (source, target) in enumerate(pairs):
            source = source if "::" in source else C + source
            target = target if "::" in target else C + target
            edges.append({"id": f"e{number:02d}", "source": source, "target": target, "relation": "invokes"})
        edges.append({"id": "zz", "source": "pkg/core.py::<module>", "target": C + "helper", "relation": "contains"})
        document = {"backend": "locagent", "error": None, "nodes": nodes, "edges": edges}
        records, report = locagent_records(document, self.index, ["pkg"])
        found = {(r["caller"], r["callee"]): r for r in records}
        self.assertEqual(found[(C + "direct_caller", C + "helper")]["line"], line_of(CORE, "return helper(value)"))
        self.assertEqual(found[(C + "direct_caller", C + "helper")]["line_source"], "ast_call_name")
        alias = found[(C + "alias_caller", U + "transform")]
        self.assertEqual((alias["line"], alias["line_source"]), (line_of(CORE, "return shifted(value)"), "ast_call_alias"))
        nested = found[(C + "outer", C + "outer.<locals>.inner")]
        self.assertEqual((nested["engine_callee"], nested["line"]), (C + "outer.inner", line_of(CORE, "return inner()")))
        self.assertIn((C + "outer.<locals>.inner", C + "helper"), found)
        init = found[(C + "Widget.__init__", C + "helper")]
        self.assertEqual((init["caller_mapping"], init["engine_caller"], init["line"]),
                         ("class_node_init", C + "Widget", line_of(CORE, "self.value = helper(4)")))
        missing = found[(C + "direct_caller", U + "transform")]
        self.assertEqual((missing["line"], missing["line_source"], missing["site_caller"]), (None, "not_found", None))
        self.assertEqual(report["invokes_edges"], 10)
        self.assertEqual(report["invokes_edges_exported"], 6)
        self.assertEqual(report["dropped_edges"], {"callee_class": 1, "callee_unmapped": 1,
                                                   "caller_kind_file": 1, "caller_out_of_scope": 1})
        self.assertEqual(report["function_nodes_unmapped"], [C + "ghost"])
        self.assertEqual(report["qualname_mismatches_count"], 0)
        self.assertFalse(any("tests/" in r["caller"] + r["callee"] for r in records))

    def test_engine_error_exports_nothing(self):
        records, report = locagent_records({"error": "engine_timeout", "nodes": [], "edges": []}, self.index, ["pkg"])
        self.assertEqual(records, [])
        self.assertEqual(report["engine_error"], "engine_timeout")

    def test_src_layout_prefix(self):
        write_tree(self.root, {"src/lib/mod.py": "def a():\n    return b()\n\n\ndef b():\n    return 1\n"})
        nodes = [{"id": "lib/mod.py::a", "kind": "function", "path": "lib/mod.py", "start_line": 1},
                 {"id": "lib/mod.py::b", "kind": "function", "path": "lib/mod.py", "start_line": 5}]
        edges = [{"id": "x", "source": "lib/mod.py::a", "target": "lib/mod.py::b", "relation": "invokes"}]
        records, _ = locagent_records({"error": None, "nodes": nodes, "edges": edges}, self.index,
                                      ["src/lib"], path_prefix="src")
        self.assertEqual([(r["caller"], r["callee"], r["path"], r["line"]) for r in records],
                         [("src/lib/mod.py::a", "src/lib/mod.py::b", "src/lib/mod.py", 2)])


REFERENCE_SOURCE = '''\
from pkg.util import target as alias_name


def user(items, obj, sort):
    target(1)
    result = map(target, items)
    sort(key=target)
    handler = target
    obj.target(2)
    target.cache_clear()
    if target is None:
        pass
    table = {"k": target, target: 1}
    return target


@target
def decorated():
    pass


def default(callback=target):
    pass


target = 3
s = "h\u00e9llo"; target(4)
'''


class ReferenceSiteTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        write_tree(self.root, {"pkg/refs.py": REFERENCE_SOURCE,
                               "pkg/defs.py": "def plain():\n    pass\nasync def coro():\n    pass\n"
                                              "def  spaced():\n    pass\nclass K:\n    def method(self): pass\n"
                                              "def \\\n    continued():\n    pass\n"})
        self.index = FunctionIndex(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def site(self, marker, occurrence=1, name="target"):
        line = line_of(REFERENCE_SOURCE, marker)
        text = self.index.line_text("pkg/refs.py", line)
        column = -1
        for _ in range(occurrence):
            column = text.index(name, column + 1)
        return reference_sites(self.index, "pkg/refs.py").get((line, column))

    def test_site_classification(self):
        self.assertEqual(self.site("from pkg.util import"), ("import", "import"))
        self.assertEqual(self.site("target(1)"), ("call", "call"))
        self.assertEqual(self.site("map(target, items)"), ("reference", "argument"))
        self.assertEqual(self.site("sort(key=target)"), ("reference", "keyword_argument"))
        self.assertEqual(self.site("handler = target"), ("reference", "assignment_value"))
        self.assertEqual(self.site("obj.target(2)"), ("call", "call"))
        self.assertEqual(self.site("target.cache_clear()"), ("attribute_base", "attribute_base"))
        self.assertEqual(self.site("if target is None"), ("reference", "comparison"))
        self.assertEqual(self.site('table = {"k": target'), ("reference", "dict_value"))
        self.assertEqual(self.site('table = {"k": target', 2), ("reference", "dict_key"))
        self.assertEqual(self.site("return target"), ("reference", "return_value"))
        self.assertEqual(self.site("@target"), ("decorator", "decorator"))
        self.assertEqual(self.site("def default(callback=target)"), ("reference", "default_value"))
        self.assertEqual(self.site("target = 3"), ("binding", "store"))
        self.assertEqual(self.site('s = "h'), ("call", "call"))

    def test_definition_name_positions(self):
        positions = {entry["name"]: definition_name_position(self.index, "pkg/defs.py", entry)
                     for entry in self.index.functions("pkg/defs.py")}
        self.assertEqual(positions, {"plain": (1, 4), "coro": (3, 10), "spaced": (5, 5),
                                     "method": (8, 8), "continued": (10, 4)})


class JediRecordTests(unittest.TestCase):
    """jedi_records on synthetic raw jedi results (no jedi needed)."""

    def test_classification_mapping_and_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            stage = Path(directory).resolve()
            write_tree(stage, FIXTURE)
            index = FunctionIndex(stage)
            core = str(stage / "pkg/core.py")

            def at(marker, name="helper", occurrence=1):
                line = line_of(CORE, marker)
                column = -1
                for _ in range(occurrence):
                    column = index.line_text("pkg/core.py", line).index(name, column + 1)
                return line, column

            tasks = [{"key": 0, "relpath": "pkg/core.py", "identity": C + "helper", "name": "helper"},
                     {"key": 1, "relpath": "pkg/core.py", "identity": C + "outer", "name": "outer"}]
            references = [   # module path, line, column, is_definition, name, support
                [core, *at("def helper"), True, "helper", 3],
                [core, *at("return helper(value)"), False, "helper", 3],
                [core, *at("return helper(value)"), False, "helper", 3],
                [core, *at("return list(map(helper, items))"), False, "helper", 2],
                [core, *at("helper(10)"), False, "helper", 3],
                [core, *at("MODULE_ALIAS = helper"), False, "helper", 3],
                [str(stage / "pkg/tests/test_core.py"), 5, 11, False, "helper", 3],
                ["/elsewhere/module.py", 1, 0, False, "helper", 3],
                [None, 1, 0, False, "helper", 3],
            ]
            results = {0: {"references": references, "error": None, "seconds": 0.5, "attempts": 3,
                           "distinct_results": 2},
                       1: {"references": [], "error": "timeout", "seconds": 300.0, "attempts": 0,
                           "distinct_results": 0}}
            records, report = jedi_records(results, tasks, index, stage, ["pkg"])
            callback_line = line_of(CORE, "return list(map(helper, items))")
            self.assertEqual([(r["caller"], r["kind"], r["context"], r["line"]) for r in records], [
                (C + "callback_caller", "reference", "argument", callback_line),
                (C + "direct_caller", "call", "call", line_of(CORE, "return helper(value)")),
            ])
            self.assertFalse(any("support" in r for r in records))
            self.assertEqual(report["unstable_definitions"],
                             [{"identity": C + "helper", "distinct_results": 2, "attempts": 3}])
            self.assertEqual([(r["caller"], r["line"], r["support"], r["attempts"]) for r in report["unstable_records"]],
                             [(C + "callback_caller", callback_line, 2, 3)])
            self.assertEqual(report["dropped_references"], {"definition": 1, "no_enclosing_function": 2,
                                                            "no_module_path": 1, "out_of_scope": 1,
                                                            "outside_project": 1})
            self.assertEqual(report["timeouts"], 1)
            self.assertEqual(report["definitions_failed"], [{"identity": C + "outer", "error": "timeout"}])


class _FakeName:
    def __init__(self, line):
        self.module_path, self.line, self.column, self.name = Path("/stage/pkg/x.py"), line, 0, "f"

    def is_definition(self):
        return False


@unittest.skipUnless(hasattr(__import__("signal"), "SIGALRM"), "SIGALRM required")
class JediAttemptTests(unittest.TestCase):
    """The worker unions attempts and escalates only definitions whose answers differ."""

    def run_task(self, flaky):
        from absent_edges.probe import static_callers
        calls = []

        class Script:
            def __init__(self, **kwargs):
                pass

            def get_references(self, *args, **kwargs):
                calls.append(1)
                extra = [_FakeName(2)] if flaky and len(calls) % 3 == 0 else []
                return [_FakeName(1)] + extra

        saved = dict(static_callers._JEDI_WORKER)
        static_callers._JEDI_WORKER.update(jedi=type("FakeJedi", (), {"Script": Script}), stage=Path("/stage"),
                                           timeout=60, attempts=5, max_attempts=12, project=None, environment=None)
        try:
            return static_callers._jedi_worker_task({"key": 0, "relpath": "pkg/x.py", "line": 1, "column": 4})
        finally:
            static_callers._JEDI_WORKER.clear()
            static_callers._JEDI_WORKER.update(saved)

    def test_stable_definition_stops_at_base_attempts(self):
        result = self.run_task(flaky=False)
        self.assertEqual((result["attempts"], result["distinct_results"], result["error"]), (5, 1, None))
        self.assertEqual(result["references"], [["/stage/pkg/x.py", 1, 0, False, "f", 5]])

    def test_unstable_definition_escalates_and_keeps_the_union(self):
        result = self.run_task(flaky=True)
        self.assertEqual((result["attempts"], result["distinct_results"]), (12, 2))
        self.assertEqual(result["references"], [["/stage/pkg/x.py", 1, 0, False, "f", 12],
                                                ["/stage/pkg/x.py", 2, 0, False, "f", 4]])


class AgreementAndGuardTests(unittest.TestCase):
    def test_agreement_summary(self):
        left = [{"caller": "a", "callee": "b", "path": "p", "line": 1, "kind": "call"},
                {"caller": "a", "callee": "c", "path": "p", "line": None, "kind": "call"}]
        right = [{"caller": "a", "callee": "b", "path": "p", "line": 1, "kind": "call"},
                 {"caller": "d", "callee": "b", "path": "p", "line": 9, "kind": "reference"}]
        summary = agreement_summary(left, right)
        self.assertEqual((summary["both_pairs"], summary["locagent_only_pairs"], summary["jedi_only_pairs"]), (1, 1, 1))
        self.assertEqual((summary["both_sites"], summary["jedi_call_pairs"]), (1, 1))

    def test_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, out = Path(directory) / "repo", Path(directory) / "out"
            write_tree(repo, FIXTURE)
            for family in ("../escape", "", "a/b", ".hidden"):
                with self.assertRaises(ValueError):
                    run_static_backend(family=family, repo_root=repo, package_dirs=["pkg"],
                                       backend="locagent", out_root=out)
            with self.assertRaises(ValueError):
                run_static_backend(family="fam", repo_root=repo, package_dirs=["pkg"], backend="pycg", out_root=out)
            (out / "fam").mkdir(parents=True)
            (out / "fam" / "locagent.manifest.json").write_text("{}")
            with self.assertRaises(FileExistsError):
                run_static_backend(family="fam", repo_root=repo, package_dirs=["pkg"], backend="locagent", out_root=out)


def run_cli(python, repo, out, backend, *extra):
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    command = [str(python), str(SCRIPT), "--family", "fixture", "--repo-root", str(repo), "--package-dirs", "pkg",
               "--backend", backend, "--out", str(out), "--locagent-root", str(LOCAGENT),
               "--graph-python", str(GRAPH_PYTHON), *extra]
    return subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=environment)


def finish(process):
    stdout, stderr = process.communicate(timeout=600)
    return process.returncode, stdout, stderr


@unittest.skipUnless(LOCAGENT_READY and JEDI_READY, "pinned LocAgent, graph env and jedi env required")
class EngineIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        base = Path(cls.temporary.name)
        cls.repo = base / "repo"
        write_tree(cls.repo, FIXTURE)
        # Concurrent L and J runs into one family directory (parallel safety).
        first = [run_cli(sys.executable, cls.repo, base / "out1", "locagent"),
                 run_cli(JEDI_PYTHON, cls.repo, base / "out1", "jedi", "--workers", "3")]
        cls.first = [finish(process) for process in first]
        # Sequential reruns with another worker count (determinism).
        cls.second = [finish(run_cli(sys.executable, cls.repo, base / "out2", "locagent")),
                      finish(run_cli(JEDI_PYTHON, cls.repo, base / "out2", "jedi", "--workers", "1"))]
        cls.out1, cls.out2 = base / "out1" / "fixture", base / "out2" / "fixture"

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_runs_complete(self):
        for code, stdout, stderr in self.first + self.second:
            self.assertEqual(code, 0, stderr[-3000:])
        self.assertEqual(sorted(path.name for path in self.out1.iterdir()),
                         ["jedi.edges.jsonl", "jedi.manifest.json", "locagent.document.json",
                          "locagent.edges.jsonl", "locagent.manifest.json"])

    def test_outputs_are_deterministic(self):
        for name in ("locagent.edges.jsonl", "jedi.edges.jsonl"):
            self.assertEqual((self.out1 / name).read_bytes(), (self.out2 / name).read_bytes(), name)

    def test_manifests(self):
        for backend in ("locagent", "jedi"):
            manifest = json.loads((self.out1 / (backend + ".manifest.json")).read_text())
            self.assertEqual((manifest["status"], manifest["errors"]), ("complete", []))
            edges = (self.out1 / (backend + ".edges.jsonl")).read_bytes()
            self.assertEqual(manifest["outputs"]["edges"]["sha256"], hashlib.sha256(edges).hexdigest())
            self.assertEqual(manifest["counts"]["package_files"], 3)
            self.assertEqual(sorted(manifest["package_file_sha256"]), ["pkg/__init__.py", "pkg/core.py", "pkg/util.py"])
            self.assertEqual(manifest["source_tree_sha256"], source_tree_sha256(manifest["package_file_sha256"]))
        pin = json.loads((self.out1 / "locagent.manifest.json").read_text())["backend_pin"]
        self.assertEqual(pin["revision"], "4935b557326c154bad8e8dcf3747cc8d32d1f387")
        document = (self.out1 / "locagent.document.json").read_bytes()
        manifest = json.loads((self.out1 / "locagent.manifest.json").read_text())
        self.assertEqual(manifest["outputs"]["locagent_document"]["sha256"], hashlib.sha256(document).hexdigest())
        self.assertEqual(manifest["report"]["dropped_edges"], {"callee_class": 1})
        jedi_pin = json.loads((self.out1 / "jedi.manifest.json").read_text())["backend_pin"]
        self.assertEqual(jedi_pin["scope"], "project")
        self.assertTrue(jedi_pin["jedi_version"])

    def test_locagent_edges(self):
        records = load_edges(self.out1 / "locagent.edges.jsonl")
        pairs = {(r["caller"], r["callee"]) for r in records}
        self.assertEqual(pairs, set(LOCAGENT_PAIRS))
        for record in records:
            marker = LOCAGENT_PAIRS[(record["caller"], record["callee"])]
            self.assertEqual(record["line"], expected_line(record["caller"], marker), record)
            self.assertEqual(record["path"], record["caller"].split("::")[0])
        by_pair = {(r["caller"], r["callee"]): r for r in records}
        self.assertEqual(by_pair[(C + "alias_caller", U + "transform")]["line_source"], "ast_call_alias")
        self.assertEqual(by_pair[(C + "Widget.__init__", C + "helper")]["caller_mapping"], "class_node_init")
        self.assertEqual(by_pair[(C + "outer", C + "outer.<locals>.inner")]["engine_callee"], C + "outer.inner")

    def test_jedi_edges(self):
        records = load_edges(self.out1 / "jedi.edges.jsonl")
        by_pair = {}
        for record in records:
            by_pair.setdefault((record["caller"], record["callee"]), []).append(record)
        for pair, (kind, marker) in JEDI_REQUIRED.items():
            self.assertIn(pair, by_pair)
            self.assertIn((kind, expected_line(pair[0], marker)),
                          {(r["kind"], r["line"]) for r in by_pair[pair]}, pair)
        for record in records:
            self.assertNotIn("tests/", record["path"])
            self.assertIsNotNone(record["caller"])
            self.assertIn(record["kind"], ("call", "decorator", "reference"))
        self.assertNotIn((C + "decorated", C + "register"), by_pair)
        pin = json.loads((self.out1 / "jedi.manifest.json").read_text())["backend_pin"]
        self.assertEqual((pin["attempts_per_definition"], pin["max_attempts_unstable"]), (5, 40))

    def test_no_module_level_or_test_callers_and_agreement(self):
        locagent = load_edges(self.out1 / "locagent.edges.jsonl")
        jedi = load_edges(self.out1 / "jedi.edges.jsonl")
        for record in locagent + jedi:
            self.assertNotIn("<module>", record["caller"])
            self.assertNotIn("test", record["caller"].split("::")[0])
        summary = agreement_summary(locagent, jedi)
        both = {(r["caller"], r["callee"]) for r in locagent} & {(r["caller"], r["callee"]) for r in jedi}
        self.assertEqual(summary["both_pairs"], len(both))
        self.assertEqual(both, set(LOCAGENT_PAIRS) - {(C + "alias_caller", U + "transform")})

    def test_existing_outputs_are_not_overwritten(self):
        code, _, stderr = finish(run_cli(sys.executable, self.repo, self.out1.parent, "locagent"))
        self.assertEqual(code, 2)
        self.assertIn("FileExistsError", stderr)

    def test_unparsable_package_file_makes_the_run_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, out = Path(directory) / "repo", Path(directory) / "out"
            write_tree(repo, {**FIXTURE, "pkg/broken.py": "def broken(:\n    pass\n"})
            code, _, stderr = finish(run_cli(sys.executable, repo, out, "locagent"))
            self.assertEqual(code, 3, stderr[-2000:])
            manifest = json.loads((out / "fixture" / "locagent.manifest.json").read_text())
            self.assertEqual(manifest["status"], "partial")
            self.assertEqual([(item["stage"], item["path"]) for item in manifest["errors"]], [("parse", "pkg/broken.py")])
            self.assertEqual({(r["caller"], r["callee"]) for r in load_edges(out / "fixture" / "locagent.edges.jsonl")},
                             set(LOCAGENT_PAIRS))


if __name__ == "__main__":
    unittest.main()
