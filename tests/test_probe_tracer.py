"""Exp A lane T tests: runtime tracer plugin and trace loaders. Synthetic fixtures only.

* Loader / convention tests run on any Python >= 3.11 (controller env).
* In-process tracer tests need sys.monitoring (Python 3.12+).
* Plugin and runner tests need a Python 3.12+ interpreter with pytest: this interpreter if
  it qualifies, else the one named by AE_TRACER_TEST_PYTHON. Runner tests also need
  ``unshare --user --map-root-user --net`` (Linux).
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import types
import unittest
import uuid

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from absent_edges.probe import tracer as T  # noqa: E402
from absent_edges.probe import tracer_plugin as P  # noqa: E402

HAS_MONITORING = hasattr(sys, "monitoring")
POSIX = os.name == "posix"

# ---------------------------------------------------------------------------
# Synthetic fixture repository. Every expected call line carries a marker.
# ---------------------------------------------------------------------------
CORE = '''\
import functools
import threading


def leaf(x):
    return x + 1


TABLE = leaf(100)  # E01


def direct(x):
    return leaf(x)  # E02


class Widget:
    DEFAULT = leaf(0)  # E03

    def method(self, x):
        return leaf(x)  # E04

    def run(self):
        return (self
                .method(1))  # E05

    @staticmethod
    def static(x):
        return leaf(x)  # E21

    @classmethod
    def build(cls):
        return cls.static(1)  # E22


def key_fn(v):
    return -v


def mapper(v):
    return v * 2


def callbacks(xs):
    ordered = sorted(xs, key=key_fn)  # E06
    doubled = list(map(mapper, xs))  # E07
    return ordered, doubled


def outer(x):
    def inner(y):
        return leaf(y)  # E08
    return inner(x)  # E09


def gen(n):
    for i in range(n):
        yield leaf(i)  # E10


def uses_gen(n):
    return list(gen(n))


async def acoro():
    return leaf(5)  # E11


def runs_coro():
    import asyncio
    return asyncio.run(acoro())


def deco(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)  # E13
    return wrapper


@deco  # E14
def decorated(x):
    return x


def calls_decorated():
    return decorated(2)  # E15


def via_lambda(xs):
    return list(map(lambda v: leaf(v), xs))  # E16


def threaded():
    out = []

    def work():
        out.append(leaf(10))  # E17

    worker = threading.Thread(target=work)
    worker.start()
    worker.join()
    return out


@functools.lru_cache(maxsize=None)
def cached(x):
    return leaf(x)  # E18


def calls_cached():
    return cached(1)  # E19


def fact(n):
    return 1 if n <= 1 else n * fact(n - 1)  # E20
'''

ALIAS = '''\
import functools

from . import core
from .core import leaf as renamed

alias = core.leaf


def via_alias():
    return renamed(3)  # A01


def via_attribute():
    return core.leaf(4)  # A02


def via_assignment():
    return alias(5)  # A03


def dispatch(name):
    table = {"leaf": core.leaf}
    return table[name](6)  # A04


def via_getattr():
    return getattr(core, "leaf")(7)  # A05


def via_partial():
    return functools.partial(core.leaf, 8)()  # A06
'''

COMPAT_TEST = "from .core import leaf\n\n\ndef compat_helper():\n    return leaf(2)\n"
HELPERS = "from ..core import leaf\n\n\ndef helper_in_tests():\n    return leaf(7)\n"
TOOL = "from PKGNAME.core import leaf\n\n\ndef tool_main():\n    return leaf(1)\n"

TESTS = '''\
import pytest

from PKGNAME import _compat_test, alias_user, core
from PKGNAME.tests import helpers_for_tests
from scripts import tool


def test_direct():
    assert core.direct(1) == 2


def test_widget():
    assert core.Widget().run() == 2
    assert core.Widget.build() == 2


def test_callbacks():
    assert core.callbacks([1, 3, 2]) == ([3, 2, 1], [2, 6, 4])


def test_nested():
    assert core.outer(1) == 2


def test_generator_excluded():
    assert core.uses_gen(3) == [1, 2, 3]


def test_coroutine():
    assert core.runs_coro() == 6


def test_decorated():
    assert core.calls_decorated() == 2


def test_lambda():
    assert core.via_lambda([1]) == [2]


def test_thread():
    assert core.threaded() == [11]


def test_cached():
    assert core.calls_cached() == 2


def test_recursion():
    assert core.fact(3) == 6


def test_alias_forms():
    assert alias_user.via_alias() == 4
    assert alias_user.via_attribute() == 5
    assert alias_user.via_assignment() == 6
    assert alias_user.dispatch("leaf") == 7
    assert alias_user.via_getattr() == 8
    assert alias_user.via_partial() == 9


def test_excluded_callers():
    assert helpers_for_tests.helper_in_tests() == 8
    assert _compat_test.compat_helper() == 3
    assert tool.tool_main() == 2


@pytest.mark.skip(reason="fixture skip")
def test_skipped():
    raise AssertionError("never runs")


@pytest.mark.xfail(strict=True, reason="fixture xfail")
def test_xfail():
    assert core.leaf(0) == 0
'''

FAILING = "from PKGNAME import core\n\n\ndef test_fails():\n    assert core.leaf(0) == 99\n"

SLEEP = '''\
import os
import signal
import time


def test_sleep():
    marker = os.environ.get("AE_FIXTURE_MARKER")
    if marker:
        with open(marker, "w") as stream:
            stream.write("started")
    if os.environ.get("AE_FIXTURE_IGNORE_SIGTERM") == "1":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    time.sleep(float(os.environ.get("AE_FIXTURE_SLEEP", "60")))
'''

OFFLINE = '''\
import os
import socket


def test_only_loopback_interface():
    with open("/proc/self/net/dev") as stream:
        rows = stream.read().splitlines()[2:]
    assert sorted(row.split(":", 1)[0].strip() for row in rows if ":" in row) == ["lo"]


def test_loopback_usable():
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    client = socket.create_connection(server.getsockname(), timeout=5)
    conn, _ = server.accept()
    client.sendall(b"x")
    assert conn.recv(1) == b"x"
    for sock in (client, conn, server):
        sock.close()


def test_external_network_unreachable():
    try:
        socket.create_connection(("1.1.1.1", 53), timeout=3).close()
    except OSError:
        return
    raise AssertionError("external network reachable inside the test namespace")


def test_environment_scrubbed():
    allowed = {"PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "TZ", "PYTHONDONTWRITEBYTECODE", "PYTHONNOUSERSITE",
               "PYTHONHASHSEED", "PY_COLORS", "NO_COLOR", "VIRTUAL_ENV", "PYTHONPATH", "PYTEST_VERSION",
               "PYTEST_CURRENT_TEST", "PWD", "OLDPWD", "SHLVL", "_"}
    extra = {name for name in os.environ if name not in allowed and not name.startswith("AE_")}
    assert not extra, sorted(extra)
    assert os.environ["PYTHONDONTWRITEBYTECODE"] == "1"
    assert os.environ["HOME"] != os.path.expanduser("~nobody")
'''

# marker, caller module, caller qualname, callee module, callee qualname, caller kind, caller scope
EXPECTED = (
    ("E01", "core", "<module>", "core", "leaf", "module", None),
    ("E02", "core", "direct", "core", "leaf", "function", None),
    ("E03", "core", "Widget", "core", "leaf", "class_body", None),
    ("E04", "core", "Widget.method", "core", "leaf", "function", None),
    ("E05", "core", "Widget.run", "core", "Widget.method", "function", None),
    ("E06", "core", "callbacks", "core", "key_fn", "function", None),
    ("E07", "core", "callbacks", "core", "mapper", "function", None),
    ("E08", "core", "outer.<locals>.inner", "core", "leaf", "function", None),
    ("E09", "core", "outer", "core", "outer.<locals>.inner", "function", None),
    ("E10", "core", "gen", "core", "leaf", "function", None),
    ("E11", "core", "acoro", "core", "leaf", "function", None),
    ("E13", "core", "deco.<locals>.wrapper", "core", "decorated", "function", None),
    ("E14", "core", "<module>", "core", "deco", "module", None),
    ("E15", "core", "calls_decorated", "core", "deco.<locals>.wrapper", "function", None),
    ("E16", "core", "via_lambda", "core", "leaf", "function", "via_lambda.<locals>.<lambda>"),
    ("E17", "core", "threaded.<locals>.work", "core", "leaf", "function", None),
    ("E18", "core", "cached", "core", "leaf", "function", None),
    ("E19", "core", "calls_cached", "core", "cached", "function", None),
    ("E20", "core", "fact", "core", "fact", "function", None),
    ("E21", "core", "Widget.static", "core", "leaf", "function", None),
    ("E22", "core", "Widget.build", "core", "Widget.static", "function", None),
    ("A01", "alias_user", "via_alias", "core", "leaf", "function", None),
    ("A02", "alias_user", "via_attribute", "core", "leaf", "function", None),
    ("A03", "alias_user", "via_assignment", "core", "leaf", "function", None),
    ("A04", "alias_user", "dispatch", "core", "leaf", "function", None),
    ("A05", "alias_user", "via_getattr", "core", "leaf", "function", None),
    ("A06", "alias_user", "via_partial", "core", "leaf", "function", None),
)
IMPORT_TIME = ("E01", "E03", "E14")
FIXTURE_OUTCOMES = {"passed": 13, "skipped": 1, "xfailed": 1}
_MARKER = re.compile(r"#\s*([EA]\d\d)\s*$")


def marker_lines(text: str) -> dict:
    found = {}
    for number, line in enumerate(text.splitlines(), 1):
        match = _MARKER.search(line)
        if match:
            found[match.group(1)] = number
    return found


def write_fixture(root: Path, pkg: str) -> dict:
    """Write the fixture repo; return {marker: expected record tuple}."""
    files = {
        f"{pkg}/__init__.py": "",
        f"{pkg}/core.py": CORE,
        f"{pkg}/alias_user.py": ALIAS,
        f"{pkg}/_compat_test.py": COMPAT_TEST,
        f"{pkg}/tests/__init__.py": "",
        f"{pkg}/tests/helpers_for_tests.py": HELPERS,
        "scripts/tool.py": TOOL.replace("PKGNAME", pkg),
        "conftest.py": "",
        "tests/test_fixture.py": TESTS.replace("PKGNAME", pkg),
        "tests/test_failing.py": FAILING.replace("PKGNAME", pkg),
        "tests/test_sleep.py": SLEEP,
        "tests/test_offline.py": OFFLINE,
    }
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    lines = {"core": marker_lines(CORE), "alias_user": marker_lines(ALIAS)}
    expected = {}
    for marker, caller_mod, caller_q, callee_mod, callee_q, kind, scope in EXPECTED:
        caller_path = f"{pkg}/{caller_mod}.py"
        expected[marker] = (f"{caller_path}::{caller_q}", f"{pkg}/{callee_mod}.py::{callee_q}", caller_path,
                            lines[caller_mod][marker], kind, scope)
    return expected


def as_tuple(record: dict) -> tuple:
    return (record["caller"], record["callee"], record["path"], record["line"], record["caller_kind"],
            record.get("caller_scope"))


def drive_fixture(root: Path, pkg: str) -> None:
    """In-process equivalent of tests/test_fixture.py (imports happen under the tracer)."""
    core = importlib.import_module(pkg + ".core")
    alias_user = importlib.import_module(pkg + ".alias_user")
    compat = importlib.import_module(pkg + "._compat_test")
    helpers = importlib.import_module(pkg + ".tests.helpers_for_tests")
    spec = importlib.util.spec_from_file_location(pkg + "_tool", root / "scripts" / "tool.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    assert core.direct(1) == 2
    assert core.Widget().run() == 2 and core.Widget.build() == 2
    assert core.callbacks([1, 3, 2]) == ([3, 2, 1], [2, 6, 4])
    assert core.outer(1) == 2
    assert core.uses_gen(3) == [1, 2, 3]
    assert core.runs_coro() == 6
    assert core.calls_decorated() == 2
    assert core.via_lambda([1]) == [2]
    assert core.threaded() == [11]
    assert core.calls_cached() == 2
    assert core.fact(3) == 6
    assert [alias_user.via_alias(), alias_user.via_attribute(), alias_user.via_assignment(),
            alias_user.dispatch("leaf"), alias_user.via_getattr(), alias_user.via_partial()] == [4, 5, 6, 7, 8, 9]
    assert helpers.helper_in_tests() == 8 and compat.compat_helper() == 3 and tool.tool_main() == 2


def load_runner():
    spec = importlib.util.spec_from_file_location("expA_trace_runner", PROJECT / "scripts" / "expA_trace.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_PYTEST_PYTHON = None


def pytest_python():
    """A Python 3.12+ interpreter with pytest, or None."""
    global _PYTEST_PYTHON
    if _PYTEST_PYTHON is None:
        _PYTEST_PYTHON = ""
        for candidate in (sys.executable, os.environ.get("AE_TRACER_TEST_PYTHON")):
            if not candidate or not Path(candidate).exists():
                continue
            check = subprocess.run([candidate, "-c", "import sys, pytest; sys.exit(0 if hasattr(sys, 'monitoring') else 3)"],
                                   capture_output=True, timeout=120, check=False)
            if check.returncode == 0:
                _PYTEST_PYTHON = candidate
                break
    return _PYTEST_PYTHON or None


def unshare_ok() -> bool:
    binary = shutil.which("unshare")
    if not POSIX or not binary:
        return False
    result = subprocess.run([binary, "--user", "--map-root-user", "--net", "--", "/bin/true"],
                            capture_output=True, timeout=60, check=False)
    return result.returncode == 0


def rec(caller, callee, line, **extra):
    return dict(caller=caller, callee=callee, path=caller.split("::", 1)[0], line=line, **extra)


# ---------------------------------------------------------------------------
# Conventions and loaders (Python 3.11+)
# ---------------------------------------------------------------------------
class IdentityConventionTests(unittest.TestCase):
    def test_identity_format_round_trip(self):
        identity = T.function_identity("src/attr/_make.py", "_ClassBuilder.build_class")
        self.assertEqual(identity, "src/attr/_make.py::_ClassBuilder.build_class")
        self.assertEqual(tuple(T.split_identity(identity)), ("src/attr/_make.py", "_ClassBuilder.build_class"))
        nested = T.function_identity("pkg/m.py", "outer.<locals>.inner")
        self.assertEqual(nested, "pkg/m.py::outer.<locals>.inner")
        self.assertEqual(tuple(T.split_identity(nested)), ("pkg/m.py", "outer.<locals>.inner"))

    def test_exclusions_follow_section_2(self):
        for path in ("tests/test_x.py", "test/x.py", "testing/x.py", "pkg/x_test.py", "pkg/test_x.py",
                     "conftest.py", "pkg/conftest.py", "docs/conf.py", "examples/e.py", "benchmarks/b.py",
                     "scripts/s.py", "setup.py", "noxfile.py", "pkg/tests/helpers.py"):
            self.assertTrue(T.is_test_or_excluded_path(path), path)
        for path in ("src/attr/_make.py", "pkg/core.py", "pkg/testers.py", "pkg/contest.py", "pkg/latest.py"):
            self.assertFalse(T.is_test_or_excluded_path(path), path)

    def test_package_scope_is_component_wise(self):
        self.assertTrue(T.in_package_scope("src/attr/_make.py", ["src/attr"]))
        self.assertFalse(T.in_package_scope("src/attrs/_make.py", ["src/attr"]))
        self.assertTrue(T.path_in_scope("src/attr/_make.py", ("src/attr",)))
        self.assertFalse(T.path_in_scope("src/attr/tests/test_make.py", ("src/attr",)))
        self.assertFalse(T.path_in_scope("src/attr/data.txt", ("src/attr",)))

    def test_identity_implementation_is_recorded(self):
        self.assertIn(T.IDENTITY_IMPL, ("absent_edges.probe.identity", "fallback:absent_edges.probe.tracer"))


class NormalizationTests(unittest.TestCase):
    def test_caller_qualname_normalization(self):
        cases = {
            "f": ("f", "function", None),
            "A.m": ("A.m", "function", None),
            "outer.<locals>.inner": ("outer.<locals>.inner", "function", None),
            "f.<locals>.<lambda>": ("f", "function", "f.<locals>.<lambda>"),
            "A.m.<locals>.<genexpr>": ("A.m", "function", "A.m.<locals>.<genexpr>"),
            "f.<locals>.<lambda>.<locals>.<lambda>": ("f", "function", "f.<locals>.<lambda>.<locals>.<lambda>"),
            "<lambda>": ("<module>", "module", "<lambda>"),
            "<genexpr>": ("<module>", "module", "<genexpr>"),
            "A.<lambda>": ("A", "class_body", "A.<lambda>"),
            "f.<locals>.A.<lambda>": ("f.<locals>.A", "class_body", "f.<locals>.A.<lambda>"),
            "<module>": ("<module>", "module", None),
            # CPython names a generator expression nested in another one "<genexpr>.<genexpr>".
            "A.m.<locals>.<genexpr>.<genexpr>": ("A.m", "function", "A.m.<locals>.<genexpr>.<genexpr>"),
            "f.<locals>.<genexpr>.<lambda>": ("f", "function", "f.<locals>.<genexpr>.<lambda>"),
            "A.<genexpr>.<genexpr>": ("A", "class_body", "A.<genexpr>.<genexpr>"),
            "<genexpr>.<genexpr>": ("<module>", "module", "<genexpr>.<genexpr>"),
        }
        for qualname, expected in cases.items():
            self.assertEqual(T.normalize_caller_qualname(qualname), expected, qualname)
        with self.assertRaises(ValueError):
            T.normalize_caller_qualname("")

    def test_package_dirs(self):
        self.assertEqual(T.normalize_package_dirs(["./src/attr/", "pkg", "pkg/"]), ("pkg", "src/attr"))
        self.assertEqual(T.normalize_package_dirs(os.pathsep.join(["b", "a"])), ("a", "b"))
        self.assertEqual(T.normalize_package_dirs(["."]), ("",))  # the whole repository
        self.assertEqual(T.package_dirs_env([".", "src/attr"]), os.pathsep.join([".", "src/attr"]))
        self.assertEqual(T.normalize_package_dirs(T.package_dirs_env(["src/attr", "."])), ("", "src/attr"))
        for bad in (["../x"], ["/abs"], [], ["a//b"]):
            with self.assertRaises(ValueError, msg=repr(bad)):
                T.normalize_package_dirs(bad)

    def test_record_defaults_and_optional_fields(self):
        record = T.normalize_record(rec("p/a.py::f", "p/b.py::g", 3))
        self.assertEqual(record, {"caller": "p/a.py::f", "callee": "p/b.py::g", "path": "p/a.py", "line": 3,
                                  "caller_kind": "function"})
        module = T.normalize_record(rec("p/a.py::<module>", "p/b.py::g", 1))
        self.assertEqual(module["caller_kind"], "module")
        full = T.normalize_record(rec("p/a.py::f", "p/a.py::g", 9, caller_scope="f.<locals>.<lambda>",
                                      first_test="t::x", caller_kind="function"))
        self.assertEqual(full["caller_scope"], "f.<locals>.<lambda>")
        self.assertEqual(T.call_site(full), {"caller": "p/a.py::f", "callee": "p/a.py::g", "path": "p/a.py", "line": 9})

    def test_record_rejections(self):
        bad = [
            "not a dict",
            rec("/p/a.py::f", "p/b.py::g", 3),
            rec("p\\a.py::f", "p/b.py::g", 3),
            rec("p/../a.py::f", "p/b.py::g", 3),
            {"caller": "p/a.py", "callee": "p/b.py::g", "path": "p/a.py", "line": 3},
            {"caller": "p/a.py::f", "callee": "p/b.py::g", "path": "p/b.py", "line": 3},
            rec("p/a.py::f", "p/b.py::g", 0),
            rec("p/a.py::f", "p/b.py::g", True),
            rec("p/a.py::f", "p/b.py::g", "3"),
            rec("p/a.py::f", "p/b.py::g", 3, caller_kind="lambda"),
            rec("p/a.py::f", "p/b.py::f.<locals>.<lambda>", 3),
            rec("p/a.py::f", "p/b.py::<module>", 3),
            rec("p/a.py::f.<locals>.<lambda>", "p/b.py::g", 3),
            rec("p/a.py::<module>", "p/b.py::g", 3, caller_kind="function"),
            rec("p/a.py::f", "p/b.py::g", 3, caller_kind="module"),
            rec("p/a.py::f", "p/b.py::g", 3, caller_scope=""),
        ]
        for raw in bad:
            with self.assertRaises(T.TraceRecordError, msg=repr(raw)):
                T.normalize_record(raw)

    def test_dedupe_keeps_smallest_first_test(self):
        rows = [rec("p/a.py::f", "p/b.py::g", 3, first_test="z"), rec("p/a.py::f", "p/b.py::g", 3, first_test="a")]
        rows = [T.normalize_record(row) for row in rows]
        merged = T.dedupe_records(rows)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["first_test"], "a")


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="ae-tracer-load-"))

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_atomic_writers_leave_no_temporaries(self):
        rows = [T.normalize_record(rec("p/a.py::f", "p/b.py::g", 3))]
        sha = T.write_jsonl_atomic(self.dir / "edges.jsonl", rows)
        self.assertEqual(sha, T.sha256_file(self.dir / "edges.jsonl"))
        T.write_json_atomic(self.dir / "m.json", {"b": 1, "a": [1, 2]})
        self.assertEqual(json.loads((self.dir / "m.json").read_text()), {"a": [1, 2], "b": 1})
        self.assertEqual(sorted(p.name for p in self.dir.iterdir()), ["edges.jsonl", "m.json"])

    def test_load_edges_strict_and_lenient(self):
        good = T.canonical_json(rec("p/a.py::f", "p/b.py::g", 3, first_test="z"))
        dup = T.canonical_json(rec("p/a.py::f", "p/b.py::g", 3, first_test="a"))
        path = self.dir / "edges.jsonl"
        path.write_text("\n".join([good, dup, "{not json", T.canonical_json(rec("p/a.py::f", "p/b.py::g", 0)), ""]),
                        encoding="utf-8")
        with self.assertRaises(T.TraceRecordError):
            T.load_edges(path)
        loaded = T.load_edges(path, strict=False)
        self.assertEqual(len(loaded.records), 1)
        self.assertEqual(loaded.records[0]["first_test"], "a")
        self.assertEqual(loaded.duplicates, 1)
        self.assertEqual([error["line"] for error in loaded.errors], [3, 4])

    def test_trace_dir_merges_processes_and_falls_back_to_checkpoints(self):
        main_rows = [T.normalize_record(rec("p/a.py::f", "p/b.py::g", 3))]
        worker_rows = [T.normalize_record(rec("p/a.py::h", "p/b.py::g", 5)),
                       T.normalize_record(rec("p/a.py::f", "p/b.py::g", 3))]
        sha = T.write_jsonl_atomic(self.dir / "edges.jsonl", main_rows)
        T.write_json_atomic(self.dir / "plugin_manifest.json", {"edges_file": "edges.jsonl", "edges_sha256": sha,
                                                                "partial": False})
        T.write_jsonl_atomic(self.dir / "edges.partial.jsonl", [T.normalize_record(rec("p/a.py::x", "p/b.py::g", 1))])
        T.write_jsonl_atomic(self.dir / "edges.gw1.partial.jsonl", worker_rows)
        run = T.load_trace_dir(self.dir)
        self.assertEqual(run.sources, {"edges": "final", "edges.gw1": "checkpoint"})
        self.assertTrue(run.partial)
        self.assertEqual(run.integrity, {"edges.jsonl": True, "edges.gw1.partial.jsonl": None})
        self.assertEqual({r["caller"] for r in run.records}, {"p/a.py::f", "p/a.py::h"})
        self.assertEqual(run.duplicates, 1)

    def test_trace_dir_complete_and_tampered(self):
        rows = [T.normalize_record(rec("p/a.py::f", "p/b.py::g", 3))]
        T.write_jsonl_atomic(self.dir / "edges.jsonl", rows)
        T.write_json_atomic(self.dir / "plugin_manifest.json", {"edges_file": "edges.jsonl", "edges_sha256": "0" * 64,
                                                                "partial": False})
        run = T.load_trace_dir(self.dir)
        self.assertFalse(run.partial)
        self.assertEqual(run.integrity, {"edges.jsonl": False})

    def test_observed_callers_and_call_sites(self):
        rows = [T.normalize_record(r) for r in (
            rec("p/a.py::f", "p/b.py::g", 3), rec("p/a.py::f", "p/b.py::g", 7), rec("p/c.py::h", "p/b.py::g", 2),
            rec("p/b.py::g", "p/b.py::g", 9), rec("p/a.py::<module>", "p/b.py::g", 1),
            rec("p/a.py::K", "p/b.py::g", 4, caller_kind="class_body"), rec("p/a.py::f", "p/b.py::k", 5))]
        callers = T.observed_callers(rows)
        self.assertEqual(callers["p/b.py::g"], frozenset({"p/a.py::f", "p/c.py::h", "p/b.py::g"}))
        self.assertEqual(T.observed_callers(rows, include_self=False)["p/b.py::g"],
                         frozenset({"p/a.py::f", "p/c.py::h"}))
        everything = T.observed_callers(rows, caller_kinds=T.CALLER_KINDS)["p/b.py::g"]
        self.assertIn("p/a.py::<module>", everything)
        self.assertIn("p/a.py::K", everything)
        sites = T.observed_call_sites(rows)["p/b.py::g"]
        self.assertEqual([(s["caller"], s["line"]) for s in sites],
                         [("p/a.py::f", 3), ("p/a.py::f", 7), ("p/b.py::g", 9), ("p/c.py::h", 2)])
        self.assertTrue(all(set(s) == {"caller", "callee", "path", "line"} for s in sites))
        summary = T.summarize_edges(rows)
        self.assertEqual(summary["edges"], 7)
        self.assertEqual(summary["edges_by_caller_kind"], {"class_body": 1, "function": 5, "module": 1})
        self.assertEqual(summary["callees_with_ge2_callers"], 1)
        self.assertEqual(summary["callees_with_ge2_callers_excluding_self"], 1)
        self.assertEqual(summary["self_call_records"], 1)

    def test_source_tree_digest(self):
        (self.dir / "pkg" / "sub").mkdir(parents=True)
        (self.dir / "pkg" / "__pycache__").mkdir()
        (self.dir / "pkg" / "a.py").write_text("x = 1\n")
        (self.dir / "pkg" / "sub" / "b.py").write_text("y = 2\n")
        (self.dir / "pkg" / "__pycache__" / "a.cpython-313.pyc").write_bytes(b"junk")
        (self.dir / "pkg" / "notes.txt").write_text("ignored")
        first = T.source_tree_digest(self.dir, ["pkg"])
        self.assertEqual(first["files"], 2)
        self.assertEqual(first, T.source_tree_digest(self.dir, ["pkg/"]))
        (self.dir / "pkg" / "a.py").write_text("x = 2\n")
        self.assertNotEqual(first["sha256"], T.source_tree_digest(self.dir, ["pkg"])["sha256"])

    @unittest.skipIf(T.IDENTITY_IMPL.startswith("fallback"), "needs absent_edges.probe.identity.FunctionIndex")
    def test_identity_cross_check(self):
        source = textwrap.dedent('''\
            import functools

            def deco(fn):
                @functools.wraps(fn)
                def wrapper(*a):
                    return fn(*a)
                return wrapper

            class K:
                def m(self):
                    return helper(lambda v: v)

                class Inner:
                    async def am(self):
                        return 1

            def helper(x):
                return x
            ''')
        (self.dir / "pkg").mkdir()
        (self.dir / "pkg" / "m.py").write_text(source)
        records = [T.normalize_record(r) for r in (
            rec("pkg/m.py::K.m", "pkg/m.py::helper", 11),
            rec("pkg/m.py::deco.<locals>.wrapper", "pkg/m.py::deco", 6),
            rec("pkg/m.py::K.m", "pkg/m.py::missing", 11),
            rec("pkg/m.py::K.m", "pkg/m.py::helper", 18),
            rec("pkg/m.py::<module>", "pkg/m.py::K.Inner.am", 1))]
        report = T.check_identities(records, self.dir)
        self.assertEqual(report["function_caller_records_checked"], 4)
        self.assertEqual(report["unknown_callee_identities"], 1)
        self.assertEqual(report["examples_unknown_callees"], ["pkg/m.py::missing"])
        self.assertEqual(report["unknown_caller_identities"], 0)
        self.assertEqual(report["enclosing_mismatches"], 1)
        self.assertEqual(report["examples_enclosing_mismatches"][0]["static_enclosing"], "pkg/m.py::helper")


class RunnerHelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_runner()

    def test_build_test_argv(self):
        build = self.runner.build_test_argv
        py = "/env/bin/python"
        self.assertEqual(build(["python", "-m", "pytest", "-q", "tests"], py, plugin=True),
                         [py, "-m", "pytest", "-p", P.__name__, "-p", "no:cacheprovider", "-q", "tests"])
        self.assertEqual(build(["pytest", "-x"], py, plugin=True)[:7],
                         [py, "-m", "pytest", "-p", P.__name__, "-p", "no:cacheprovider"])
        self.assertEqual(build(["python3", "-X", "dev", "-m", "pytest"], py, plugin=False, cacheprovider=True),
                         [py, "-X", "dev", "-m", "pytest"])
        for bad in (["tox"], ["python", "-m", "unittest"], []):
            with self.assertRaises(ValueError):
                build(bad, py, plugin=True)

    def test_set_env_policy(self):
        self.assertEqual(self.runner.parse_set_env(["A_B=1"]), {"A_B": "1"})
        for bad in ("GITHUB_TOKEN=x", "AWS_SECRET_ACCESS_KEY=x", "AE_TRACE_ROOT=/x", "HOME=/x", "nonsense"):
            with self.assertRaises(ValueError, msg=bad):
                self.runner.parse_set_env([bad])

    def test_summary_parser(self):
        parse = self.runner.parse_pytest_summary
        self.assertEqual(parse("x\n===== 13 passed, 1 skipped, 1 xfailed in 0.52s =====\n")["outcomes"],
                         {"passed": 13, "skipped": 1, "xfailed": 1})
        parsed = parse("1 failed, 2 passed, 3 errors, 4 warnings in 3.00s")
        self.assertEqual(parsed["outcomes"], {"failed": 1, "passed": 2, "error": 3, "warning": 4})
        self.assertEqual(parsed["ran"], 6)
        self.assertIsNone(parse("no summary here"))

    def test_status_decision(self):
        decide = self.runner.decide_status
        run = {"killed_after_grace": False, "terminated_at_cap": False}
        final = {"partial_reason": None, "tests": {"ran": 3}}
        self.assertEqual(decide("trace", run, None, final, None, None), ("complete", None, 3))
        self.assertEqual(decide("trace", dict(run, killed_after_grace=True), None, None, final, None),
                         ("partial", "checkpoint_only", 3))
        self.assertEqual(decide("trace", run, None, {"partial_reason": None, "tests": {"ran": 0}}, None, None)[0],
                         "unavailable")
        self.assertEqual(decide("trace", run, None, None, None, None)[0], "failed")
        self.assertEqual(decide("plain", dict(run, terminated_at_cap=True), None, None, None, {"ran": 2}),
                         ("partial", "sigterm_at_cap", 2))


class CalleeExclusionTests(unittest.TestCase):
    def test_co_flag_and_name_rules(self):
        source = ("def f():\n    return [x for x in ()]\n"
                  "def g():\n    yield 1\n"
                  "async def c():\n    pass\n"
                  "async def ag():\n    yield 1\n"
                  "def h():\n    return (x for x in ())\n"
                  "lam = lambda: 1\n"
                  "class K:\n    def m(self):\n        pass\n")
        module = compile(source, "x.py", "exec")
        codes = {}

        def walk(code):
            for const in code.co_consts:
                if isinstance(const, types.CodeType):
                    codes[const.co_qualname] = const
                    walk(const)

        walk(module)
        self.assertEqual(P.callee_exclusion(module), "module")
        self.assertIsNone(P.callee_exclusion(codes["f"]))
        self.assertEqual(P.callee_exclusion(codes["g"]), "generator")
        self.assertEqual(P.callee_exclusion(codes["c"]), "coroutine")
        self.assertEqual(P.callee_exclusion(codes["ag"]), "async_generator")
        self.assertEqual(P.callee_exclusion(codes["h.<locals>.<genexpr>"]), "generator")
        self.assertEqual(P.callee_exclusion(codes["<lambda>"]), "lambda")
        self.assertEqual(P.callee_exclusion(codes["K"]), "class_body")
        self.assertIsNone(P.callee_exclusion(codes["K.m"]))


# ---------------------------------------------------------------------------
# In-process tracer (Python 3.12+)
# ---------------------------------------------------------------------------
@unittest.skipUnless(HAS_MONITORING, "sys.monitoring requires Python 3.12+")
class InProcessTracerTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="ae-tracer-inproc-")).resolve()
        self.pkg = "aefx_" + uuid.uuid4().hex[:10]
        self.expected = write_fixture(self.root, self.pkg)
        sys.path.insert(0, str(self.root))

    def tearDown(self):
        sys.path.remove(str(self.root))
        for name in list(sys.modules):
            if name == self.pkg or name.startswith((self.pkg + ".", self.pkg + "_")):
                del sys.modules[name]
        shutil.rmtree(self.root, ignore_errors=True)

    def test_exact_edges_on_synthetic_fixture(self):
        tracer = P.EdgeTracer(self.root, [self.pkg])
        with tracer:
            tool_id = tracer.tool_id
            self.assertEqual(sys.monitoring.get_tool(tool_id), P.TOOL_NAME)
            drive_fixture(self.root, self.pkg)
        self.assertIsNone(sys.monitoring.get_tool(tool_id))
        rows, stats = tracer.records()
        got = {as_tuple(row) for row in rows}
        expected = set(self.expected.values())
        self.assertEqual(sorted(got - expected), [], "unexpected edges")
        self.assertEqual(sorted(expected - got), [], "missing edges")
        self.assertEqual(stats, {"dropped_line_unknown": 0, "merged_duplicate_rows": 0})
        counters = tracer.counters()
        for reason in ("generator", "coroutine", "lambda", "class_body", "module", "out_of_scope", "traced"):
            self.assertGreaterEqual(counters["callee_code_objects"].get(reason, 0), 1, reason)
        self.assertEqual(counters["frame_mismatches"], 0)
        self.assertEqual(counters["classify_errors"], 0)
        self.assertGreaterEqual(counters["caller_code_objects"]["anonymous_scope_normalized"], 1)
        # every record must pass the loader's validation
        self.assertEqual([T.normalize_record(row) for row in rows], rows)

    def test_function_caller_sets(self):
        tracer = P.EdgeTracer(self.root, [self.pkg])
        with tracer:
            drive_fixture(self.root, self.pkg)
        rows, _ = tracer.records()
        callers = T.observed_callers(rows)
        leaf = f"{self.pkg}/core.py::leaf"
        self.assertNotIn(f"{self.pkg}/core.py::<module>", callers[leaf])
        self.assertNotIn(f"{self.pkg}/core.py::Widget", callers[leaf])
        self.assertIn(f"{self.pkg}/core.py::via_lambda", callers[leaf])
        self.assertIn(f"{self.pkg}/core.py::threaded.<locals>.work", callers[leaf])
        # decorated-wrapper attribution: the target's only observed caller is the wrapper
        self.assertEqual(callers[f"{self.pkg}/core.py::decorated"],
                         frozenset({f"{self.pkg}/core.py::deco.<locals>.wrapper"}))
        self.assertNotIn(f"{self.pkg}/core.py::gen", callers)
        self.assertNotIn(f"{self.pkg}/core.py::acoro", callers)

    def test_tool_ids_are_claimed_and_released(self):
        first = P.EdgeTracer(self.root, [self.pkg]).start()
        try:
            second = P.EdgeTracer(self.root, [self.pkg]).start()
            try:
                self.assertNotEqual(first.tool_id, second.tool_id)
            finally:
                second.stop()
            with self.assertRaises(P.TracerError):
                first.start()
        finally:
            first.stop()
            first.stop()  # idempotent
        self.assertIsNone(sys.monitoring.get_tool(first.released_tool_id))


# ---------------------------------------------------------------------------
# pytest plugin (subprocess; Python 3.12+ with pytest)
# ---------------------------------------------------------------------------
@unittest.skipUnless(POSIX, "POSIX only")
class PluginTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.python = pytest_python()
        if not cls.python:
            raise unittest.SkipTest("needs a Python 3.12+ interpreter with pytest (set AE_TRACER_TEST_PYTHON)")
        cls.runner = load_runner()

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="ae-tracer-plugin-")).resolve()
        self.root = self.tmp / "repo"
        self.root.mkdir()
        self.out = self.tmp / "out"
        self.overlay = self.tmp / "overlay"
        self.runner.stage_overlay(self.overlay)
        self.expected = write_fixture(self.root, "pkgx")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def env(self, mode="trace", **extra):
        env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": str(self.tmp), "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
               "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1", "PYTHONPATH": str(self.overlay),
               T.ENV_ROOT: str(self.root), T.ENV_PACKAGE_DIRS: "pkgx", T.ENV_OUT: str(self.out), T.ENV_MODE: mode,
               T.ENV_CHECKPOINT: "0"}
        env.update({key: str(value) for key, value in extra.items()})
        return env

    def pytest_argv(self, *selection):
        return [self.python, "-m", "pytest", "-p", P.__name__, "-p", "no:cacheprovider", "-q", *selection]

    def run_pytest(self, *selection, mode="trace", **extra):
        return subprocess.run(self.pytest_argv(*selection), cwd=self.root, env=self.env(mode, **extra),
                              capture_output=True, text=True, timeout=600, check=False)

    def plugin_manifest(self, name="plugin_manifest.json"):
        return json.loads((self.out / name).read_text(encoding="utf-8"))

    def edges(self):
        return {as_tuple(record) for record in T.load_trace_dir(self.out).records}

    def expect(self, *markers):
        return {self.expected[marker] for marker in markers}


class PytestPluginTests(PluginTestBase):
    def test_trace_mode_exact_edges_counts_and_integrity(self):
        result = self.run_pytest("tests/test_fixture.py")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.edges(), set(self.expected.values()))
        manifest = self.plugin_manifest()
        self.assertEqual(manifest["tests"]["outcomes"], FIXTURE_OUTCOMES)
        self.assertEqual(manifest["tests"]["collected"], 15)
        self.assertEqual(manifest["tests"]["ran"], 15)
        self.assertFalse(manifest["partial"])
        self.assertIsNone(manifest["partial_reason"])
        self.assertEqual(manifest["exit_status"], 0)
        self.assertEqual(manifest["mode"], "trace")
        self.assertEqual(manifest["identity_impl"], T.IDENTITY_IMPL)
        self.assertEqual(manifest["counts"]["frame_mismatches"], 0)
        self.assertEqual(manifest["counts"]["classify_errors"], 0)
        run = T.load_trace_dir(self.out)
        self.assertEqual(run.integrity, {"edges.jsonl": True})
        self.assertFalse(run.partial)
        first = {as_tuple(r): r.get("first_test") for r in run.records}
        for marker in IMPORT_TIME:
            self.assertEqual(first[self.expected[marker]], "<collection>", marker)
        self.assertEqual(first[self.expected["E02"]], "tests/test_fixture.py::test_direct")
        self.assertEqual(first[self.expected["E17"]], "tests/test_fixture.py::test_thread")
        self.assertEqual(sorted(p.name for p in self.out.iterdir()), ["edges.jsonl", "plugin_manifest.json"])
        header = subprocess.run([self.python, "-m", "pytest", "-p", P.__name__, "-p", "no:cacheprovider", "--co",
                                 "tests/test_fixture.py"],
                                cwd=self.root, env=self.env(mode="count", **{T.ENV_OUT: self.tmp / "hdr"}),
                                capture_output=True, text=True, timeout=300, check=False)
        self.assertIn("absent-edges tracer: mode=count", header.stdout)

    def test_count_mode_is_the_untraced_baseline(self):
        result = self.run_pytest("tests/test_fixture.py", mode="count")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        manifest = self.plugin_manifest()
        self.assertEqual(manifest["mode"], "count")
        self.assertEqual(manifest["tests"]["outcomes"], FIXTURE_OUTCOMES)
        self.assertIsNone(manifest["edges_file"])
        self.assertIsNone(manifest["monitoring_tool_id"])
        self.assertFalse(list(self.out.glob("edges*.jsonl")))

    def test_failures_are_counted_not_fatal(self):
        result = self.run_pytest("tests/test_fixture.py::test_direct", "tests/test_failing.py")
        self.assertEqual(result.returncode, 1)
        manifest = self.plugin_manifest()
        self.assertEqual(manifest["tests"]["outcomes"], {"passed": 1, "failed": 1})
        self.assertFalse(manifest["partial"])
        self.assertEqual(self.edges(), self.expect(*IMPORT_TIME, "E02"))

    def test_soft_deadline_stops_between_tests_and_marks_partial(self):
        result = self.run_pytest("tests/test_fixture.py", **{T.ENV_DEADLINE: repr(time.time() - 1)})
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        manifest = self.plugin_manifest()
        self.assertTrue(manifest["partial"])
        self.assertEqual(manifest["partial_reason"], "wall_time_cap")
        self.assertEqual(manifest["tests"]["outcomes"], {"passed": 1})
        self.assertEqual(self.edges(), self.expect(*IMPORT_TIME, "E02"))
        self.assertTrue(T.load_trace_dir(self.out).partial)

    def test_sigterm_flushes_partial_results(self):
        marker = self.tmp / "started"
        process = subprocess.Popen(self.pytest_argv("tests/test_fixture.py::test_direct", "tests/test_sleep.py"),
                                   cwd=self.root, env=self.env(AE_FIXTURE_MARKER=marker, AE_FIXTURE_SLEEP=120),
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            deadline = time.monotonic() + 120
            while not marker.exists():
                self.assertIsNone(process.poll(), "pytest exited before the sleeping test started")
                self.assertLess(time.monotonic(), deadline, "sleeping test never started")
                time.sleep(0.05)
            process.send_signal(signal.SIGTERM)
            stdout, stderr = process.communicate(timeout=120)
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()
        self.assertEqual(process.returncode, 2, stdout + stderr)
        manifest = self.plugin_manifest()
        self.assertTrue(manifest["partial"])
        self.assertEqual(manifest["partial_reason"], "sigterm")
        self.assertTrue(manifest["sigterm_handler_installed"])
        self.assertEqual(self.edges(), self.expect(*IMPORT_TIME, "E02"))

    def test_checkpoints_are_written_then_removed(self):
        result = self.run_pytest("tests/test_fixture.py", **{T.ENV_CHECKPOINT: "0.000001"})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        manifest = self.plugin_manifest()
        self.assertGreaterEqual(manifest["checkpoints_written"], 14)
        self.assertEqual(manifest["checkpoint_errors"], 0)
        self.assertFalse([p.name for p in self.out.iterdir() if T.PARTIAL_MARK in p.name])
        self.assertEqual(self.edges(), set(self.expected.values()))

    def test_xdist_workers_write_separate_mergeable_files(self):
        if subprocess.run([self.python, "-c", "import xdist"], capture_output=True, check=False).returncode != 0:
            self.skipTest("pytest-xdist is not installed in the plugin interpreter")
        result = self.run_pytest("-n", "2", "tests/test_fixture.py")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        names = {p.name for p in self.out.iterdir()}
        self.assertTrue({"edges.jsonl", "edges.gw0.jsonl", "edges.gw1.jsonl", "plugin_manifest.json",
                         "plugin_manifest.gw0.json", "plugin_manifest.gw1.json"} <= names, sorted(names))
        run = T.load_trace_dir(self.out)
        self.assertEqual({as_tuple(r) for r in run.records}, set(self.expected.values()))
        self.assertTrue(run.integrity and all(run.integrity.values()), run.integrity)
        self.assertEqual(self.plugin_manifest()["tests"]["outcomes"], FIXTURE_OUTCOMES)

    def test_missing_configuration_is_a_usage_error(self):
        env = self.env()
        del env[T.ENV_ROOT]
        result = subprocess.run(self.pytest_argv("tests/test_fixture.py"), cwd=self.root, env=env,
                                capture_output=True, text=True, timeout=300, check=False)
        self.assertEqual(result.returncode, 4, result.stdout + result.stderr)
        self.assertIn(T.ENV_ROOT, result.stderr)


# ---------------------------------------------------------------------------
# Runner script end to end (unshare + plugin interpreter)
# ---------------------------------------------------------------------------
class RunnerEndToEndTests(PluginTestBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not unshare_ok():
            raise unittest.SkipTest("unshare --user --map-root-user --net is unavailable")

    def run_runner(self, *selection, out=None, options=(), timeout=900):
        argv = [sys.executable, str(PROJECT / "scripts" / "expA_trace.py"), "--family", "synthetic-fixture",
                "--repo", str(self.root), "--package-dir", "pkgx", "--python", self.python,
                "--out", str(out or self.out), *options, "--", "python", "-m", "pytest", "-q", *selection]
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        return subprocess.run(argv, env=env, capture_output=True, text=True, timeout=timeout, check=False)

    def run_manifest(self, out=None):
        return json.loads(((out or self.out) / T.RUN_MANIFEST_FILE).read_text(encoding="utf-8"))

    def test_complete_offline_run(self):
        result = self.run_runner("tests/test_fixture.py", "tests/test_offline.py")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        manifest = self.run_manifest()
        self.assertEqual(manifest["status"], "complete", manifest.get("error"))
        self.assertEqual(manifest["isolation"]["netns_interfaces"], ["lo"])
        self.assertTrue(manifest["isolation"]["loopback_up"])
        self.assertEqual(manifest["env"]["PYTHONDONTWRITEBYTECODE"], "1")
        self.assertEqual(manifest["tests"]["plugin"]["outcomes"], dict(FIXTURE_OUTCOMES, passed=17))
        self.assertEqual(manifest["edges"]["integrity"], {"edges.jsonl": True})
        self.assertEqual(manifest["edges"]["summary"]["edges"], len(self.expected))
        self.assertTrue(manifest["source_unchanged_after_run"])
        self.assertEqual(manifest["repo"]["source_tree"]["files"], 3)  # __init__, core, alias_user
        self.assertIn("absent_edges/probe/tracer_plugin.py", manifest["plugin_files"])
        self.assertEqual(self.edges(), set(self.expected.values()))
        self.assertFalse(list(self.out.glob(".scratch-*")))
        self.assertFalse(list(self.root.rglob("__pycache__")))

    def test_count_and_plain_modes_match_trace_outcomes(self):
        counts = {}
        for mode in ("count", "plain"):
            out = self.tmp / f"out-{mode}"
            result = self.run_runner("tests/test_fixture.py", out=out, options=("--mode", mode))
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            manifest = self.run_manifest(out)
            self.assertEqual(manifest["status"], "complete", manifest.get("error"))
            counts[mode] = (manifest["tests"]["plugin"] or {}).get("outcomes") or \
                manifest["tests"]["summary_line"]["outcomes"]
        self.assertEqual(counts["count"], FIXTURE_OUTCOMES)
        self.assertEqual(counts["plain"], FIXTURE_OUTCOMES)

    def test_wall_time_cap_sigterm_keeps_partial_edges(self):
        result = self.run_runner("tests/test_fixture.py::test_direct", "tests/test_sleep.py",
                                 options=("--wall-time-cap", "8", "--soft-margin", "0", "--grace", "30",
                                          "--set-env", "AE_FIXTURE_SLEEP=120"))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        manifest = self.run_manifest()
        self.assertEqual(manifest["status"], "partial", manifest.get("error"))
        self.assertEqual(manifest["partial_reason"], "sigterm")
        self.assertTrue(manifest["run"]["terminated_at_cap"])
        self.assertFalse(manifest["run"]["killed_after_grace"])
        self.assertEqual(self.edges(), self.expect(*IMPORT_TIME, "E02"))

    def test_kill_after_grace_falls_back_to_checkpoint(self):
        result = self.run_runner("tests/test_fixture.py::test_direct", "tests/test_sleep.py",
                                 options=("--wall-time-cap", "8", "--soft-margin", "0", "--grace", "2",
                                          "--checkpoint-seconds", "0.000001", "--set-env", "AE_FIXTURE_SLEEP=120",
                                          "--set-env", "AE_FIXTURE_IGNORE_SIGTERM=1"))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        manifest = self.run_manifest()
        self.assertEqual(manifest["status"], "partial", manifest.get("error"))
        self.assertEqual(manifest["partial_reason"], "checkpoint_only")
        self.assertTrue(manifest["run"]["killed_after_grace"])
        self.assertEqual(manifest["edges"]["sources"], {"edges": "checkpoint"})
        self.assertEqual(self.edges(), self.expect(*IMPORT_TIME, "E02"))

    def test_refuses_to_reuse_an_output_directory(self):
        self.out.mkdir()
        (self.out / T.RUN_MANIFEST_FILE).write_text("{}")
        result = self.run_runner("tests/test_fixture.py::test_direct")
        self.assertEqual(result.returncode, 2)
        self.assertIn("refusing to reuse", result.stderr)


if __name__ == "__main__":
    unittest.main()
