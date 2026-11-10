"""Exp A shared identities: identity strings, package scope and the AST function index.

Qualnames, first lines and generator/async flags are checked against the code objects
CPython itself compiles from the same fixture, so the index matches ``__qualname__``,
``co_firstlineno`` and ``co_flags`` exactly under the running interpreter.
"""
import hashlib
import inspect
import os
from pathlib import Path
import tempfile
import types
import unittest

from absent_edges.data import canonical_json
from absent_edges.probe.identity import (
    FunctionIndex, file_sha256s, function_identity, in_package_scope, is_test_or_excluded_path,
    normalize_package_dirs, package_files, scan_package_files, source_tree_sha256, split_identity)

QUALNAME_FIXTURE = '''\
import functools
from typing import overload


def plain():
    return 1


@functools.lru_cache(maxsize=None)
@staticmethod
def decorated(value):
    return value


async def coroutine():
    return 1


async def async_generator():
    yield 1


def generator():
    yield 1


def not_a_generator():
    def nested_generator():
        yield 1
    return nested_generator


def default_yield_owner():
    def child(value=(yield)):
        return value
    return child


class Outer:
    attribute = len("class body")

    def method(self):
        def helper():
            return 1
        return helper()

    @property
    def value(self):
        return 1

    @value.setter
    def value(self, new):
        pass

    class Inner:
        def deep(self):
            return [lambda: 1 for _ in range(2)]

    @staticmethod
    def static():
        return (lambda: 2)()


def factory():
    class Local:
        def method(self):
            return 1
    return Local


def declares_global():
    global promoted

    def promoted():
        return 1
    return promoted


@overload
def over(value: int) -> int: ...
@overload
def over(value: str) -> str: ...
def over(value):
    return value


if functools:  # not constant-folded, so both code objects exist on every Python
    def conditional():
        return 1
else:
    def conditional():
        return 2


def multi_line(
    first,
    second=len("default"),
):
    return first


def with_comprehension(items):
    return [len(item) for item in items]


def decorates_first():
    @functools.wraps(plain)
    def wrapper():
        return plain()
    return wrapper


def holds_class():
    class Holder:
        @staticmethod
        def build():
            return 1
    return Holder


handler = lambda value: value
plain()  # module-level call
'''


def line_of(source, marker, occurrence=1):
    found = 0
    for number, text in enumerate(source.splitlines(), 1):
        if marker in text:
            found += 1
            if found == occurrence:
                return number
    raise AssertionError(f"marker not found: {marker!r}")


def code_objects(code):
    stack = [code]
    while stack:
        current = stack.pop()
        yield current
        stack.extend(const for const in current.co_consts if isinstance(const, types.CodeType))


class IdentityStringTests(unittest.TestCase):
    def test_identity_round_trip_and_normalization(self):
        identity = function_identity("src/attr/_make.py", "_ClassBuilder.build_class")
        self.assertEqual(identity, "src/attr/_make.py::_ClassBuilder.build_class")
        self.assertEqual(split_identity(identity), ("src/attr/_make.py", "_ClassBuilder.build_class"))
        self.assertEqual(function_identity("pkg\\mod.py", "outer.<locals>.inner"), "pkg/mod.py::outer.<locals>.inner")
        self.assertEqual(function_identity("./pkg/mod.py", "f"), "pkg/mod.py::f")
        self.assertEqual(split_identity("pkg/mod.py::outer.<locals>.<lambda>"), ("pkg/mod.py", "outer.<locals>.<lambda>"))

    def test_invalid_identities_are_rejected(self):
        for relpath, qualname in (("/abs.py", "f"), ("../up.py", "f"), ("a//b.py", "f"), ("C:/x.py", "f"),
                                  ("a.py", ""), ("a.py", "a..b"), ("a.py", "a b"), ("a.py", "a::b"),
                                  ("a::b.py", "f"), ("", "f")):
            with self.assertRaises(ValueError, msg=(relpath, qualname)):
                function_identity(relpath, qualname)
        for identity in ("no-separator", "a.py::b::c", "./a.py::f", "a.py::", "::f", None):
            with self.assertRaises(ValueError, msg=identity):
                split_identity(identity)


class PackageScopeTests(unittest.TestCase):
    def test_excluded_paths(self):
        excluded = ["tests/x.py", "pkg/tests/x.py", "pkg/test/x.py", "pkg/testing/x.py", "docs/conf.py",
                    "examples/a.py", "benchmarks/b.py", "scripts/s.py", "conftest.py", "pkg/conftest.py",
                    "setup.py", "noxfile.py", "pkg/test_util.py", "pkg/util_test.py"]
        kept = ["pkg/testutils.py", "pkg/contest.py", "pkg/core.py", "pkg/attest.py", "src/pkg/latest.py",
                "pkg/testing_helpers.py", "pkg/script.py"]
        for path in excluded:
            self.assertTrue(is_test_or_excluded_path(path), path)
        for path in kept:
            self.assertFalse(is_test_or_excluded_path(path), path)

    def test_in_package_scope(self):
        self.assertTrue(in_package_scope("src/attr/_make.py", ["src/attr/"]))
        self.assertTrue(in_package_scope("src/attr/sub/mod.py", "src/attr"))
        self.assertFalse(in_package_scope("src/attr/tests/t.py", ["src/attr"]))
        self.assertFalse(in_package_scope("src/attrs/x.py", ["src/attr"]))
        self.assertFalse(in_package_scope("src/attr/data.txt", ["src/attr"]))
        self.assertFalse(in_package_scope("src/attr/stub.pyi", ["src/attr"]))
        self.assertTrue(in_package_scope("six.py", ["six.py"]))
        self.assertFalse(in_package_scope("six_extra.py", ["six.py"]))
        self.assertTrue(in_package_scope("other/mod.py", ["pkg", "other"]))
        self.assertTrue(in_package_scope("x.py", ["."]))
        self.assertFalse(in_package_scope("tests/x.py", ["."]))
        self.assertEqual(normalize_package_dirs(["./pkg/", "pkg", "src\\b"]), ("pkg", "src/b"))
        self.assertEqual(normalize_package_dirs("src/attr/"), ("src/attr",))
        with self.assertRaises(ValueError):
            normalize_package_dirs([])

    def test_scan_and_tree_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relpath in ("pkg/__init__.py", "pkg/core.py", "pkg/sub/deep.py", "pkg/tests/test_core.py",
                            "pkg/test_helpers.py", "pkg/notes.txt", "tests/test_top.py", "setup.py",
                            "pkg/__pycache__/core.cpython-311.py"):
                path = root / relpath
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(("# " + relpath + "\n").encode())
            scan = scan_package_files(root, ["pkg", "missing"])
            self.assertEqual(scan["files"], ["pkg/__init__.py", "pkg/core.py", "pkg/sub/deep.py"])
            self.assertEqual(scan["missing_package_dirs"], ["missing"])
            self.assertEqual(package_files(root, "pkg"), scan["files"])
            hashes = file_sha256s(root, scan["files"])
            self.assertEqual(source_tree_sha256(hashes), hashlib.sha256(canonical_json(hashes)).hexdigest())
            self.assertEqual(hashes["pkg/core.py"], hashlib.sha256(b"# pkg/core.py\n").hexdigest())

    @unittest.skipUnless(os.name == "posix", "symlinks and secure inventory require POSIX")
    def test_symlinks_are_skipped_and_digest_matches_graph_inventory(self):
        from absent_edges.graphs import source_inventory
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            (root / "pkg").mkdir(parents=True)
            (root / "pkg" / "core.py").write_text("def f():\n    return 1\n")
            (root / "pkg" / "util.py").write_text("def g():\n    return 2\n")
            self.assertEqual(source_tree_sha256(file_sha256s(root, package_files(root, ["pkg"]))),
                             source_inventory(root, parse=False)["source_sha256"])
            outside = Path(directory) / "outside.py"
            outside.write_text("SECRET = 1\n")
            (root / "pkg" / "link.py").symlink_to(outside)
            (root / "pkg" / "linkdir").symlink_to(Path(directory))
            scan = scan_package_files(root, ["pkg"])
            self.assertEqual(scan["files"], ["pkg/core.py", "pkg/util.py"])
            self.assertEqual(scan["skipped_symlinks"], ["pkg/link.py", "pkg/linkdir/"])
            index = FunctionIndex(root)
            self.assertEqual(index.functions("pkg/link.py"), [])
            self.assertIn("pkg/link.py", index.errors)


class FunctionIndexTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "pkg").mkdir()
        (self.root / "pkg" / "mod.py").write_text(QUALNAME_FIXTURE, encoding="utf-8")
        self.index = FunctionIndex(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def test_qualnames_lines_and_flags_match_cpython_code_objects(self):
        code = compile(QUALNAME_FIXTURE, "pkg/mod.py", "exec", dont_inherit=True)
        functions = [item for item in code_objects(code) if item.co_flags & inspect.CO_OPTIMIZED]
        named = {(item.co_firstlineno, item.co_name, item.co_qualname): item
                 for item in functions if not item.co_name.startswith("<")}
        entries = {(entry["first_lineno"], entry["name"], entry["qualname"]): entry
                   for entry in self.index.functions("pkg/mod.py")}
        self.assertEqual(set(entries), set(named))
        for key, item in named.items():
            entry = entries[key]
            generator = inspect.CO_GENERATOR | inspect.CO_ASYNC_GENERATOR
            coroutine = inspect.CO_COROUTINE | inspect.CO_ASYNC_GENERATOR
            self.assertEqual(entry["is_generator"], bool(item.co_flags & generator), key)
            self.assertEqual(entry["is_async"], bool(item.co_flags & coroutine), key)
            self.assertEqual(entry["identity"], "pkg/mod.py::" + item.co_qualname)
        lambdas = {(item.co_firstlineno, item.co_qualname) for item in functions if item.co_name == "<lambda>"}
        mine = {(entry["first_lineno"], entry["qualname"])
                for entry in self.index.functions("pkg/mod.py", include_lambdas=True) if entry["is_lambda"]}
        self.assertEqual(mine, lambdas)
        self.assertEqual(len(lambdas), 3)

    def test_expected_identities_and_metadata(self):
        entries = self.index.functions("pkg/mod.py")
        by_qualname = {}
        for entry in entries:
            by_qualname.setdefault(entry["qualname"], []).append(entry)
        for qualname in ("Outer.method.<locals>.helper", "factory.<locals>.Local.method", "promoted",
                         "not_a_generator.<locals>.nested_generator", "Outer.Inner.deep"):
            self.assertIn(qualname, by_qualname)
        self.assertNotIn("declares_global.<locals>.promoted", by_qualname)
        self.assertFalse(any(entry["is_lambda"] for entry in entries))
        decorated = by_qualname["decorated"][0]
        self.assertEqual(decorated["decorators"], ["functools.lru_cache(maxsize=None)", "staticmethod"])
        self.assertEqual(decorated["first_lineno"], line_of(QUALNAME_FIXTURE, "@functools.lru_cache"))
        self.assertEqual(decorated["lineno"], line_of(QUALNAME_FIXTURE, "def decorated"))
        self.assertEqual(decorated["body_lineno"], decorated["lineno"] + 1)
        self.assertEqual([entry["decorators"] for entry in by_qualname["Outer.value"]], [["property"], ["value.setter"]])
        self.assertEqual([entry["occurrence"] for entry in by_qualname["over"]], [0, 1, 2])
        self.assertEqual([entry["occurrence"] for entry in by_qualname["conditional"]], [0, 1])
        self.assertTrue(by_qualname["Outer.method"][0]["is_method"])
        self.assertFalse(by_qualname["Outer.method.<locals>.helper"][0]["is_method"])
        self.assertEqual(by_qualname["Outer.method.<locals>.helper"][0]["parent_qualname"], "Outer.method")
        self.assertIsNone(by_qualname["plain"][0]["parent_qualname"])
        self.assertTrue(by_qualname["default_yield_owner"][0]["is_generator"])
        self.assertFalse(by_qualname["not_a_generator"][0]["is_generator"])
        self.assertTrue(by_qualname["async_generator"][0]["is_async"] and by_qualname["async_generator"][0]["is_generator"])
        self.assertTrue(by_qualname["coroutine"][0]["is_async"] and not by_qualname["coroutine"][0]["is_generator"])
        self.assertFalse(by_qualname["plain"][0]["is_dunder"])
        self.assertEqual(by_qualname["decorates_first"][0]["body_lineno"],
                         line_of(QUALNAME_FIXTURE, "@functools.wraps(plain)"))
        self.assertIn("holds_class.<locals>.Holder.build", by_qualname)
        self.assertEqual(len(self.index.functions_at("pkg/mod.py", line_of(QUALNAME_FIXTURE, "def deep"))), 1)
        node = self.index.definition_node("pkg/mod.py", line_of(QUALNAME_FIXTURE, "class Outer"), "Outer", kind="class")
        self.assertEqual(node.name, "Outer")
        self.assertIn("factory.<locals>.Local", [entry["qualname"] for entry in self.index.classes("pkg/mod.py")])

    def test_enclosing_function(self):
        source, find = QUALNAME_FIXTURE, self.index.enclosing_function
        cases = {
            ("return 1", 1): "plain",
            ("@functools.lru_cache", 1): None,
            ("def plain", 1): None,
            ('attribute = len("class body")', 1): None,
            ("return helper()", 1): "Outer.method",
            ("def helper", 1): "Outer.method",
            ("return 1", 3): "Outer.method.<locals>.helper",
            ("return [lambda: 1 for _ in range(2)]", 1): "Outer.Inner.deep",
            ("return (lambda: 2)()", 1): "Outer.static",
            ("class Local:", 1): "factory",
            ("def method(self):", 2): None,
            ("return Local", 1): "factory",
            ("def child(value=(yield)):", 1): "default_yield_owner",
            ('second=len("default"),', 1): None,
            ("return first", 1): "multi_line",
            ("return [len(item) for item in items]", 1): "with_comprehension",
            ("handler = lambda value: value", 1): None,
            ("# module-level call", 1): None,
            ("@value.setter", 1): None,
            ("return 1", 4): "Outer.value",
            # A decorated first statement starts at its decorator, inside the parent body.
            ("@functools.wraps(plain)", 1): "decorates_first",
            ("return plain()", 1): "decorates_first.<locals>.wrapper",
            ("class Holder:", 1): "holds_class",
            ("@staticmethod", 3): None,
        }
        for (marker, occurrence), expected in cases.items():
            line = line_of(source, marker, occurrence)
            wanted = None if expected is None else "pkg/mod.py::" + expected
            self.assertEqual(find("pkg/mod.py", line), wanted, (marker, occurrence, line))
        nested = line_of(source, "return 1", 5)
        self.assertEqual(find("pkg/mod.py", nested), "pkg/mod.py::factory.<locals>.Local.method")
        self.assertIsNone(find("pkg/mod.py", 0))
        self.assertIsNone(find("pkg/mod.py", 10 ** 6))
        self.assertIsNone(find("pkg/missing.py", 3))
        self.assertNotIn("pkg/missing.py", self.index.errors)
        with self.assertRaises(TypeError):
            find("pkg/mod.py", "3")
        with self.assertRaises(TypeError):
            find("pkg/mod.py", True)
        with self.assertRaises(ValueError):
            find("../pkg/mod.py", 3)

    def test_parse_failures_are_recorded_not_raised(self):
        (self.root / "pkg" / "broken.py").write_text("def broken(:\n    pass\n")
        self.assertEqual(self.index.functions("pkg/broken.py"), [])
        self.assertIsNone(self.index.enclosing_function("pkg/broken.py", 1))
        self.assertTrue(self.index.errors["pkg/broken.py"].startswith("SyntaxError"))

    def test_line_endings_form_feeds_and_unicode_columns(self):
        source = b"def a():\r\n    pass\r\n\x0c\r\ndef b():\r\n    return 1\r\n"
        (self.root / "pkg" / "crlf.py").write_bytes(source)
        self.assertEqual(self.index.enclosing_function("pkg/crlf.py", 5), "pkg/crlf.py::b")
        self.assertEqual([entry["lineno"] for entry in self.index.functions("pkg/crlf.py")], [1, 4])
        (self.root / "pkg" / "uni.py").write_text('def f():\n    s = "h\u00e9llo"; return g()\n', encoding="utf-8")
        text = self.index.line_text("pkg/uni.py", 2)
        byte_column = text.encode("utf-8").index(b"g()")
        self.assertEqual(self.index.char_column("pkg/uni.py", 2, byte_column), text.index("g()"))

    def test_package_functions_are_sorted(self):
        (self.root / "pkg" / "a.py").write_text("def first():\n    pass\n")
        (self.root / "pkg" / "tests").mkdir()
        (self.root / "pkg" / "tests" / "test_x.py").write_text("def test_x():\n    pass\n")
        identities = [entry["identity"] for entry in self.index.package_functions(["pkg"])]
        self.assertEqual(identities[0], "pkg/a.py::first")
        self.assertFalse(any("tests/" in identity for identity in identities))
        self.assertEqual(len(identities), len(self.index.functions("pkg/mod.py")) + 1)


if __name__ == "__main__":
    unittest.main()
