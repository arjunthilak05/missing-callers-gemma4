"""Exp A truth tests (v2, call-site level) on a synthetic fixture repository (no benchmark or research data).

This module also provides the shared fixture used by test_probe_items.py and
test_probe_render.py. It uses the static lane's absent_edges.probe.identity.

The fixture covers every D2 site kind: single- and multi-line calls, a call through an
import alias (relative, absolute, multi-line and package re-export), a callable reference
passed as an argument on one line and inside a multi-line receiving call, a local
alias and a getattr string (dynamic sites), a decorator application, a nested def, a
recursive self-call, module-level and test-file callers, and targets for every
eligibility rule of items.py (D3, D10.1, D10.2, D12.4).
"""
import json
from pathlib import Path
import tempfile
import unittest

from absent_edges.probe import truth

CORE = '''"""Core fixture module."""
import sys


def target(value):
    """Return value plus one."""
    return value + 1


def direct_one(value):
    return target(value)


def direct_two(value):
    result = target(value)
    return result


def multi_line(value):
    total = target(
        value,
    )
    return total


def indirect_alias(values):
    fn = target
    return [fn(v) for v in values]


def callback_user(values):
    return list(map(target, values))


def callback_multiline(values):
    return sorted(
        values,
        key=target,
    )


def getattr_user(value):
    return getattr(sys.modules[__name__], "target")(value)


def not_a_caller(value):
    return value * 2


class Box:
    def method(self, value):
        return target(value) * 2

    @staticmethod
    def helper(value):
        return Box().method(value)

    @classmethod
    def build(cls, value):
        return cls().method(value)

    @property
    def prop(self):
        return 1

    def gen(self):
        yield target(1)

    async def coro(self):
        return target(2)

    def __repr__(self):
        return "Box()"


def decorated(fn):
    return fn


@decorated
def wrapped(value):
    return target(value)


def recursive(n):
    if n <= 0:
        return 0
    return recursive(n - 1) + direct_one(n)


def outer(value):
    def inner(v):
        return target(v)
    return inner(value)


def decorator_user():
    @target
    def local():
        return 1
    return local


RESULT = target(0)
'''

UTIL = '''from .core import target as tgt
from .core import Box


def uses_alias(value):
    return tgt(value)


def unrelated(value):
    return value


def helper_caller_a(value):
    return Box.helper(value)


def helper_caller_b(value):
    return Box.helper(value) + 1


def static_only(value):
    return value


def static_user_a(value):
    return static_only(value)


def static_user_b(value):
    return static_only(value) + 1


def jedi_broken(value):
    return value


def jedi_user_a(value):
    return jedi_broken(value)


def jedi_user_b(value):
    return jedi_broken(value) - 1
'''

EXTRA = '''from pkg.core import target as abs_alias
from pkg import target as pkg_alias
from .core import (
    target as multi_alias,
)
from .util import unrelated as target_like


def uses_abs_alias(value):
    return abs_alias(value)


def uses_pkg_alias(value):
    return pkg_alias(value)


def uses_multi_alias(value):
    return multi_alias(value)


def uses_wrong_alias(value):
    return target_like(value)


def vis_target(value):
    return value


def vis_outer(value):
    def vis_inner(v):
        return vis_target(v)
    return vis_inner(value) + vis_target(value)
'''

INIT = '"""Fixture package."""\nfrom .core import target\n'

TESTS = '''from pkg.core import target


def test_target():
    assert target(1) == 2
'''

FILES = {"pkg/__init__.py": INIT, "pkg/core.py": CORE, "pkg/util.py": UTIL, "pkg/extra.py": EXTRA,
         "tests/test_core.py": TESTS}
CORE_ID, UTIL_ID, EXTRA_ID = "pkg/core.py::", "pkg/util.py::", "pkg/extra.py::"
TARGET = CORE_ID + "target"
# V(target): eight runtime-witnessed callers plus two verified by L and J.
TRACE_CALLERS = ("direct_one", "multi_line", "indirect_alias", "callback_user", "callback_multiline",
                 "getattr_user", "Box.method", "outer.<locals>.inner")
V_TARGET = {CORE_ID + name for name in TRACE_CALLERS} | {CORE_ID + "direct_two", UTIL_ID + "uses_alias"}


def line_of(text, needle, occurrence=1):
    seen = 0
    for number, line in enumerate(text.splitlines(), 1):
        if needle in line:
            seen += 1
            if seen == occurrence:
                return number
    raise AssertionError(f"needle not found: {needle!r}")


def write_fixture(root, files=FILES):
    root = Path(root)
    for relpath, text in files.items():
        path = root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
    return root


def site(caller, callee, path, needle=None, text=None, line=None):
    if needle is not None:
        line = line_of(text, needle)
    return {"caller": caller, "callee": callee, "path": path, "line": line}


def fixture_records():
    """(trace, locagent, jedi) records in the section 2 schema; trace records are tracer-valid."""
    core, util, extra = "pkg/core.py", "pkg/util.py", "pkg/extra.py"
    trace = [
        site(CORE_ID + "direct_one", TARGET, core, "    return target(value)", CORE),
        site(CORE_ID + "multi_line", TARGET, core, "total = target(", CORE),
        site(CORE_ID + "indirect_alias", TARGET, core, "return [fn(v) for v in values]", CORE),
        site(CORE_ID + "callback_user", TARGET, core, "return list(map(target, values))", CORE),
        site(CORE_ID + "callback_multiline", TARGET, core, "return sorted(", CORE),
        site(CORE_ID + "getattr_user", TARGET, core, "return getattr(sys.modules", CORE),
        site(CORE_ID + "Box.method", TARGET, core, "return target(value) * 2", CORE),
        site(CORE_ID + "outer.<locals>.inner", TARGET, core, "return target(v)", CORE),
        site(CORE_ID + "recursive", CORE_ID + "recursive", core, "return recursive(n - 1)", CORE),
        site(CORE_ID + "recursive", CORE_ID + "direct_one", core, "return recursive(n - 1)", CORE),
        site(CORE_ID + "<module>", TARGET, core, "RESULT = target(0)", CORE),
        site("tests/test_core.py::test_target", TARGET, "tests/test_core.py", "assert target(1)", TESTS),
        site(UTIL_ID + "helper_caller_a", CORE_ID + "Box.helper", util, "return Box.helper(value)", UTIL),
        site(CORE_ID + "Box.helper", CORE_ID + "Box.method", core, "return Box().method(value)", CORE),
        site(CORE_ID + "Box.build", CORE_ID + "Box.method", core, "return cls().method(value)", CORE),
        site(UTIL_ID + "jedi_user_a", UTIL_ID + "jedi_broken", util, "return jedi_broken(value)", UTIL),
        site(UTIL_ID + "jedi_user_b", UTIL_ID + "jedi_broken", util, "return jedi_broken(value) - 1", UTIL),
        site(EXTRA_ID + "vis_outer.<locals>.vis_inner", EXTRA_ID + "vis_target", extra, "return vis_target(v)",
             EXTRA),
        site(CORE_ID + "direct_one", "pkg/missing.py::nothing", core, "    return target(value)", CORE),
    ]
    locagent = [
        site(CORE_ID + "direct_one", TARGET, core, "    return target(value)", CORE),
        site(CORE_ID + "direct_two", TARGET, core),
        site(UTIL_ID + "uses_alias", TARGET, util, "return tgt(value)", UTIL),
        site(CORE_ID + "Box.method", TARGET, core),
        site(UTIL_ID + "unrelated", TARGET, util),
        site(UTIL_ID + "helper_caller_a", CORE_ID + "Box.helper", util),
        site(UTIL_ID + "helper_caller_b", CORE_ID + "Box.helper", util),
        site(CORE_ID + "Box.helper", CORE_ID + "Box.method", core),
        site(UTIL_ID + "static_user_a", UTIL_ID + "static_only", util, "return static_only(value)", UTIL),
        site(UTIL_ID + "static_user_b", UTIL_ID + "static_only", util, "return static_only(value) + 1", UTIL),
        site(EXTRA_ID + "vis_outer", EXTRA_ID + "vis_target", extra, "return vis_inner(value) + vis_target(value)",
             EXTRA),
    ]
    jedi = [
        site(CORE_ID + "direct_two", TARGET, core, "result = target(value)", CORE),
        site(CORE_ID + "direct_two", TARGET, core, "    return result", CORE),  # inside no site: unplaced
        site(UTIL_ID + "uses_alias", TARGET, util, "return tgt(value)", UTIL),
        site(CORE_ID + "callback_user", TARGET, core, "return list(map(target, values))", CORE),
        site(CORE_ID + "callback_multiline", TARGET, core, "key=target,", CORE),
        site(CORE_ID + "decorator_user", TARGET, core, "@target", CORE),
        site(UTIL_ID + "helper_caller_b", CORE_ID + "Box.helper", util, "return Box.helper(value) + 1", UTIL),
        site(CORE_ID + "Box.build", CORE_ID + "Box.method", core, "return cls().method(value)", CORE),
        site(UTIL_ID + "static_user_a", UTIL_ID + "static_only", util, "return static_only(value)", UTIL),
        site(UTIL_ID + "static_user_b", UTIL_ID + "static_only", util, "return static_only(value) + 1", UTIL),
        site(EXTRA_ID + "vis_outer", EXTRA_ID + "vis_target", extra, "return vis_inner(value) + vis_target(value)",
             EXTRA),
    ]
    return trace, locagent, jedi


def write_records(directory, records, *, trace_status="complete", static_manifests=None):
    """Lay records out like the tracer (trace/edges.jsonl + manifest.json) and static lanes."""
    directory.mkdir(parents=True, exist_ok=True)
    trace, locagent, jedi = records
    for name, rows in (("trace/edges.jsonl", trace), ("static/locagent.edges.jsonl", locagent),
                       ("static/jedi.edges.jsonl", jedi)):
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    (directory / "trace/manifest.json").write_text(json.dumps({"status": trace_status}), encoding="utf-8")
    for backend, manifest in (static_manifests or {}).items():
        (directory / "static" / f"{backend}.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return directory


def span(path, text, needle, end_needle=None, kind="call", witnesses=("ast",)):
    lineno = line_of(text, needle)
    end = lineno if end_needle is None else line_of(text, end_needle)
    return {"path": path, "lineno": lineno, "end_lineno": end, "kind": kind, "witnesses": list(witnesses)}


class TruthFixtureTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = write_fixture(Path(self.directory.name) / "repo")
        self.scope = truth.PackageScope(self.root, ["pkg/"])
        trace, locagent, jedi = fixture_records()
        self.table = truth.build_truth(self.scope, trace=trace, locagent=locagent, jedi=jedi)

    def tearDown(self):
        self.directory.cleanup()

    def sites(self, caller, callee=TARGET):
        return self.table.sites(callee, caller)

    def test_scope_excludes_tests_and_lambdas(self):
        self.assertEqual(self.scope.files(), ("pkg/__init__.py", "pkg/core.py", "pkg/extra.py", "pkg/util.py"))
        self.assertIn(TARGET, self.scope.table())
        self.assertNotIn("tests/test_core.py::test_target", self.scope.table())
        self.assertTrue(all("<lambda>" not in identity for identity in self.scope.table()))
        self.assertFalse(self.scope.contains("tests/test_core.py"))
        self.assertFalse(self.scope.contains("../escape.py"))

    def test_verified_is_trace_union_static_agreement(self):
        self.assertEqual(set(self.table.verified(TARGET)), V_TARGET)
        self.assertEqual(self.table.verified(CORE_ID + "Box.method"), [CORE_ID + "Box.build", CORE_ID + "Box.helper"])
        self.assertEqual(self.table.verified(CORE_ID + "Box.helper"),
                         [UTIL_ID + "helper_caller_a", UTIL_ID + "helper_caller_b"])
        self.assertEqual(self.table.verified(UTIL_ID + "static_only"),
                         [UTIL_ID + "static_user_a", UTIL_ID + "static_user_b"])
        self.assertEqual(self.table.verified(EXTRA_ID + "vis_target"),
                         [EXTRA_ID + "vis_outer", EXTRA_ID + "vis_outer.<locals>.vis_inner"])

    def test_unverified_holds_single_backend_candidates_and_the_self_call(self):
        self.assertEqual(set(self.table.unverified(TARGET)), {UTIL_ID + "unrelated", CORE_ID + "decorator_user"})
        self.assertFalse(set(self.table.unverified(TARGET)) & set(self.table.verified(TARGET)))
        recursive = CORE_ID + "recursive"
        self.assertEqual(self.table.verified(recursive), [])
        self.assertEqual([(row["identity"], row["self"]) for row in self.table.summary(recursive)["U"]],
                         [(recursive, True)])
        self.assertEqual(self.table.verified(CORE_ID + "direct_one"), [recursive])

    def test_runtime_witnessed_means_at_least_one_trace_site(self):
        self.assertEqual(self.table.runtime_witnessed(TARGET), sorted(CORE_ID + name for name in TRACE_CALLERS))
        self.assertEqual(self.table.runtime_witnessed(UTIL_ID + "static_only"), [])
        for entry in self.table.summary(TARGET)["V"]:
            self.assertEqual(entry["runtime_witnessed"],
                             any("trace" in site["witnesses"] for site in entry["sites"]), entry["identity"])
            self.assertEqual(entry["runtime_witnessed"], "trace" in entry["sources"], entry["identity"])

    def test_call_sites_single_and_multi_line_with_witnesses(self):
        core = "pkg/core.py"
        self.assertEqual(self.sites(CORE_ID + "direct_one"),
                         [span(core, CORE, "    return target(value)", witnesses=("trace", "L", "ast"))])
        multi = self.sites(CORE_ID + "multi_line")
        start = line_of(CORE, "total = target(")
        self.assertEqual(multi, [{"path": core, "lineno": start, "end_lineno": start + 2, "kind": "call",
                                  "witnesses": ["trace", "ast"]}])
        # L reported direct_two without a line (no witness); J's second line lies inside no site.
        self.assertEqual(self.sites(CORE_ID + "direct_two"),
                         [span(core, CORE, "result = target(value)", witnesses=("J", "ast"))])
        notes = self.table.site_notes(TARGET, [CORE_ID + "direct_two"])
        self.assertEqual((notes["L_without_line"], notes["J_unplaced"]), (1, 1))
        method = self.sites(CORE_ID + "Box.method")
        self.assertEqual(method, [span(core, CORE, "return target(value) * 2", witnesses=("trace", "ast"))])

    def test_reference_site_is_the_reference_node_line(self):
        core = "pkg/core.py"
        self.assertEqual(self.sites(CORE_ID + "callback_user"),
                         [span(core, CORE, "return list(map(target, values))", kind="reference",
                               witnesses=("trace", "J", "ast"))])
        # D11.5: the reference span is the reference node's own line, never the receiving call.
        # The trace reports the multi-line receiving call's first line, which no AST site
        # contains, so it is a separate single-line dynamic site.
        self.assertEqual(self.sites(CORE_ID + "callback_multiline"),
                         [span(core, CORE, "return sorted(", kind="dynamic", witnesses=("trace",)),
                          span(core, CORE, "key=target,", kind="reference", witnesses=("J", "ast"))])
        self.assertEqual(self.table.stratum(TARGET, CORE_ID + "callback_multiline"), "callback")

    def test_dynamic_sites_for_a_local_alias_and_a_getattr_string(self):
        core = "pkg/core.py"
        self.assertEqual(self.sites(CORE_ID + "indirect_alias"),
                         [span(core, CORE, "return [fn(v) for v in values]", kind="dynamic", witnesses=("trace",))])
        self.assertEqual(self.sites(CORE_ID + "getattr_user"),
                         [span(core, CORE, "return getattr(sys.modules", kind="dynamic", witnesses=("trace",))])
        for name in ("indirect_alias", "getattr_user"):
            self.assertEqual(self.table.stratum(TARGET, CORE_ID + name), "dynamic")

    def test_import_aliases_are_resolved_through_the_ast(self):
        self.assertEqual(truth.import_aliases(self.scope, TARGET),
                         {"pkg/extra.py": ["abs_alias", "multi_alias", "pkg_alias"], "pkg/util.py": ["tgt"]})
        self.assertEqual(truth.import_aliases(self.scope, CORE_ID + "Box.method"), {})
        self.assertEqual(truth.import_aliases(self.scope, UTIL_ID + "unrelated"), {"pkg/extra.py": ["target_like"]})
        self.assertEqual(self.sites(UTIL_ID + "uses_alias"),
                         [span("pkg/util.py", UTIL, "return tgt(value)", witnesses=("L", "J", "ast"))])
        self.assertEqual(self.table.stratum(TARGET, UTIL_ID + "uses_alias"), "alias")
        aliases = truth.import_aliases(self.scope, TARGET)
        for name in ("uses_abs_alias", "uses_pkg_alias", "uses_multi_alias"):
            sites, _, raw, _ = truth.caller_sites(self.scope, EXTRA_ID + name, TARGET, [], aliases)
            self.assertEqual([(site["kind"], site["witnesses"]) for site in sites], [("call", ["ast"])], name)
            self.assertEqual(truth.stratum_of(raw), "alias", name)
        sites, _, raw, _ = truth.caller_sites(self.scope, EXTRA_ID + "uses_wrong_alias", TARGET, [], aliases)
        self.assertEqual((sites, truth.stratum_of(raw)), ([], None))

    def test_decorator_application_is_a_call_site(self):
        self.assertEqual(self.sites(CORE_ID + "decorator_user"),
                         [span("pkg/core.py", CORE, "@target", witnesses=("J", "ast"))])

    def test_nested_def_bodies_belong_to_the_nested_caller(self):
        aliases = truth.import_aliases(self.scope, TARGET)
        sites, _, _, _ = truth.caller_sites(self.scope, CORE_ID + "outer", TARGET, [], aliases)
        self.assertEqual(sites, [])
        self.assertEqual(self.sites(CORE_ID + "outer.<locals>.inner"),
                         [span("pkg/core.py", CORE, "return target(v)", witnesses=("trace", "ast"))])
        vis = EXTRA_ID + "vis_target"
        self.assertEqual(self.sites(EXTRA_ID + "vis_outer", vis),
                         [span("pkg/extra.py", EXTRA, "return vis_inner(value) + vis_target(value)",
                               witnesses=("L", "J", "ast"))])
        self.assertEqual(self.sites(EXTRA_ID + "vis_outer.<locals>.vis_inner", vis),
                         [span("pkg/extra.py", EXTRA, "return vis_target(v)", witnesses=("trace", "ast"))])

    def test_strata_come_from_site_kinds_and_alias_resolution(self):
        expected = {"direct_one": "direct", "direct_two": "direct", "multi_line": "direct", "Box.method": "direct",
                    "outer.<locals>.inner": "direct", "callback_user": "callback", "callback_multiline": "callback",
                    "indirect_alias": "dynamic", "getattr_user": "dynamic"}
        expected = {CORE_ID + name: stratum for name, stratum in expected.items()}
        expected[UTIL_ID + "uses_alias"] = "alias"
        self.assertEqual({caller: self.table.stratum(TARGET, caller) for caller in V_TARGET}, expected)
        self.assertEqual({entry["identity"]: entry["stratum"] for entry in self.table.summary(TARGET)["V"]}, expected)

    def test_stratum_precedence(self):
        def raw(kind, via):
            return {"path": "p.py", "lineno": 1, "end_lineno": 1, "kind": kind, "via": via, "anchor": 1}
        self.assertEqual(truth.stratum_of([raw("dynamic", None), raw("reference", "name"), raw("call", "alias"),
                                           raw("call", "attribute")]), "direct")
        self.assertEqual(truth.stratum_of([raw("dynamic", None), raw("reference", "name"), raw("call", "alias")]),
                         "alias")
        self.assertEqual(truth.stratum_of([raw("dynamic", None), raw("reference", "alias")]), "callback")
        self.assertEqual(truth.stratum_of([raw("dynamic", None)]), "dynamic")
        self.assertIsNone(truth.stratum_of([]))

    def test_record_fields_follow_the_contract(self):
        summary = self.table.summary(TARGET)
        for entry in summary["V"]:
            self.assertEqual(set(entry), {"identity", "sources", "runtime_witnessed", "stratum", "sites"})
        for entry in summary["U"]:
            self.assertEqual(set(entry), {"identity", "sources", "self", "sites"})
        for entry in summary["V"] + summary["U"]:
            self.assertEqual(entry["sources"], [name for name in truth.SOURCES if name in entry["sources"]])
            for row in entry["sites"]:
                self.assertEqual(set(row), {"path", "lineno", "end_lineno", "kind", "witnesses"})
                self.assertIn(row["kind"], truth.SITE_KINDS)
                self.assertEqual(row["witnesses"], [name for name in truth.WITNESSES if name in row["witnesses"]])
                self.assertLessEqual(row["lineno"], row["end_lineno"])
            keys = [(row["path"], row["lineno"], row["end_lineno"]) for row in entry["sites"]]
            self.assertEqual(keys, sorted(set(keys)))

    def test_lexical_names_and_lines(self):
        names = truth.lexical_names(self.scope, TARGET)
        # D10.11: X's simple and qualified names plus the AST-resolved frozen aliases (multi-line included).
        self.assertEqual(names, ["abs_alias", "multi_alias", "pkg_alias", "target", "tgt"])
        self.assertEqual(names, sorted({"target"} | {alias for values in truth.import_aliases(self.scope, TARGET)
                                                     .values() for alias in values}))
        lines = truth.lexical_lines(self.scope, names)
        self.assertEqual(lines, sorted(set(lines)))
        self.assertIn(("pkg/__init__.py", 2), lines)  # module-level re-export: a grep_all line, not a caller
        self.assertIn(("pkg/extra.py", line_of(EXTRA, "    target as multi_alias,")), lines)
        self.assertIn(("pkg/extra.py", line_of(EXTRA, "return multi_alias(value)")), lines)
        self.assertIn(("pkg/core.py", line_of(CORE, 'return getattr(sys.modules')), lines)
        self.assertNotIn(("pkg/extra.py", line_of(EXTRA, "return target_like(value)")), lines)
        self.assertNotIn(("tests/test_core.py", 1), lines)
        self.assertEqual(truth.lexical_names(self.scope, CORE_ID + "Box.method"), ["Box.method", "method"])

    def test_an_as_name_aliasing_another_object_is_not_a_lexical_name(self):
        # Review finding (dev item 72121b3fb9ee6ee03af1, more_itertools/recipes.py::batched): the import line
        # `from itertools import batched as itertools_batched` names X's simple name but aliases the stdlib
        # function. grep_all and lexical_name_match use X's names and its frozen aliases only (D10.11).
        recipes = ("from itertools import batched as itertools_batched\n\n\n"
                   "def batched(iterable, n):\n    return itertools_batched(iterable, n)\n\n\n"
                   "def user(values):\n    return batched(values, 2)\n")
        other = "from .recipes import batched as chunked\n\n\ndef via_alias(values):\n    return chunked(values, 3)\n"
        with tempfile.TemporaryDirectory() as directory:
            root = write_fixture(Path(directory) / "repo", {"pkg/__init__.py": "", "pkg/recipes.py": recipes,
                                                            "pkg/other.py": other})
            scope = truth.PackageScope(root, ["pkg/"])
            target = "pkg/recipes.py::batched"
            self.assertEqual(truth.import_aliases(scope, target), {"pkg/other.py": ["chunked"]})
            names = truth.lexical_names(scope, target)
            self.assertEqual(names, ["batched", "chunked"])
            self.assertNotIn("itertools_batched", names)
            lines = truth.lexical_lines(scope, names)
            self.assertNotIn(("pkg/recipes.py", line_of(recipes, "return itertools_batched(")), lines)
            self.assertIn(("pkg/recipes.py", line_of(recipes, "from itertools import batched")), lines)
            self.assertIn(("pkg/recipes.py", line_of(recipes, "return batched(values, 2)")), lines)
            self.assertIn(("pkg/other.py", line_of(other, "return chunked(values, 3)")), lines)
            callers = {record["caller"] for record in truth.lexical_call_sites(scope, target)} - {target}
            self.assertEqual(callers, {"pkg/recipes.py::user", "pkg/other.py::via_alias"})

    def test_lexical_candidates_enter_unverified_never_verified(self):
        before = self.table.verified(TARGET)
        records = truth.lexical_call_sites(self.scope, TARGET)
        self.table.add_records("lexical", records)
        self.assertEqual(self.table.verified(TARGET), before)
        unverified = set(self.table.unverified(TARGET)) - {TARGET}
        self.assertEqual(unverified, {UTIL_ID + "unrelated", CORE_ID + "decorator_user", CORE_ID + "Box.gen",
                                      CORE_ID + "Box.coro", CORE_ID + "wrapped", EXTRA_ID + "uses_abs_alias",
                                      EXTRA_ID + "uses_pkg_alias", EXTRA_ID + "uses_multi_alias"})
        callers = {record["caller"] for record in records}
        self.assertIn(UTIL_ID + "uses_alias", callers)  # alias line `return tgt(value)`
        self.assertNotIn(EXTRA_ID + "uses_wrong_alias", callers)
        self.assertTrue(all(record["path"].startswith("pkg/") for record in records))
        record = {entry["identity"]: entry for entry in self.table.summary(TARGET)["U"]}[EXTRA_ID + "uses_pkg_alias"]
        self.assertEqual((record["sources"], [site["kind"] for site in record["sites"]]), (["lexical"], ["call"]))

    def test_anonymous_scope_caller_maps_to_enclosing_function(self):
        record = site(CORE_ID + "indirect_alias.<locals>.<listcomp>", TARGET, "pkg/core.py",
                      "return [fn(v) for v in values]", CORE)
        self.assertTrue(self.table.add("trace", record))
        self.assertEqual(self.table.notes["trace:caller_mapped_to_enclosing_function"], 1)
        self.assertEqual(len(self.sites(CORE_ID + "indirect_alias")), 1)  # the same dynamic site

    def test_module_level_test_and_unresolved_records_are_dropped(self):
        notes = self.table.notes
        self.assertEqual(notes["trace:caller_module_level"], 1)
        self.assertEqual(notes["trace:caller_out_of_scope"], 1)
        self.assertEqual(notes["trace:callee_unresolved"], 1)  # pkg/missing.py is in scope but not indexed
        self.assertEqual(notes["trace:kept"], 16)
        for identity in ("pkg/core.py::<module>", "tests/test_core.py::test_target"):
            self.assertNotIn(identity, self.table.evidence(TARGET))

    def test_evidence_lines_keep_provenance(self):
        rows = self.table.evidence_lines(TARGET, CORE_ID + "direct_two")
        self.assertIn({"path": "pkg/core.py", "line": line_of(CORE, "result = target(value)"), "source": "J"}, rows)
        self.assertIn({"path": "pkg/core.py", "line": None, "source": "L"}, rows)
        self.assertEqual(self.table.caller_record(TARGET, CORE_ID + "indirect_alias", verified=True)["sources"],
                         ["trace"])

    def test_locagent_indexed_identities_mapping(self):
        def node(identity, path, needle):
            return {"id": identity, "kind": "function", "path": path, "start_line": line_of(CORE, needle)}
        document = {"nodes": [node("pkg/core.py::target", "pkg/core.py", "def target("),
                              node("pkg/core.py::Box.method", "pkg/core.py", "def method("),
                              node("pkg/core.py::ghost", "pkg/core.py", "def target("),
                              {"id": "pkg/core.py::Box", "kind": "class", "path": "pkg/core.py", "start_line": 1},
                              node("tests/test_core.py::test_target", "tests/test_core.py", "def target(")]}
        self.assertEqual(truth.locagent_indexed_identities(document, self.scope),
                         {TARGET, CORE_ID + "Box.method"})
        prefixed = {"nodes": [node("core.py::direct_one", "core.py", "def direct_one(")]}
        self.assertEqual(truth.locagent_indexed_identities(prefixed, self.scope, "pkg"), {CORE_ID + "direct_one"})

    def test_source_decoding_and_lines(self):
        self.assertEqual(truth.source_lines("a\r\nb\rc\nd\x0ce\n"), ["a", "b", "c", "d\x0ce"])
        self.assertEqual(truth.source_lines(""), [])
        self.assertEqual(truth.decode_source(b"\xef\xbb\xbfx = 1\n"), "x = 1\n")
        latin = "# -*- coding: latin-1 -*-\ns = 'caf\xe9'\n".encode("latin-1")
        self.assertEqual(truth.source_lines(truth.decode_source(latin))[1], "s = 'caf\xe9'")

    def test_call_site_validation(self):
        good = {"caller": "pkg/a.py::f", "callee": "pkg/a.py::g", "path": "pkg/a.py", "line": 3}
        self.assertEqual(truth.validate_call_site(good), good)
        for bad in ({**good, "line": 0}, {**good, "line": "3"}, {**good, "path": "/abs.py"},
                    {**good, "caller": "no-separator"}, {**good, "callee": "../x.py::g"}, [good]):
            with self.assertRaises(ValueError):
                truth.validate_call_site(bad)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "edges.jsonl"
            path.write_text(json.dumps(good) + "\n\n{broken\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                truth.read_call_sites(path)
            path.write_text(json.dumps(good) + "\n\n", encoding="utf-8")
            self.assertEqual(truth.read_call_sites(path), [good])

    def test_unknown_source_rejected(self):
        with self.assertRaises(ValueError):
            self.table.add_records("pycg", [])
        self.assertEqual(truth.canonical_source("locagent"), "L")
        self.assertEqual(truth.canonical_source("jedi"), "J")

    def test_package_source_hash_is_stable(self):
        first = self.scope.source_sha256()
        self.assertEqual(first, truth.PackageScope(self.root, ["pkg/"]).source_sha256())
        (self.root / "tests/test_core.py").write_text("# changed outside package scope\n", encoding="utf-8")
        self.assertEqual(first, truth.PackageScope(self.root, ["pkg/"]).source_sha256())
        (self.root / "pkg/__init__.py").write_text("# changed\n", encoding="utf-8")
        self.assertNotEqual(first, truth.PackageScope(self.root, ["pkg/"]).source_sha256())


if __name__ == "__main__":
    unittest.main()
