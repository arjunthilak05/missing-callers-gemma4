"""Shared Exp A identities (docs/EXPA_DESIGN.md section 2).

A function identity is ``"<relpath>::<qualname>"``. ``relpath`` is a POSIX path relative
to the repository root and ``qualname`` is CPython's ``__qualname__`` with ``.<locals>``
segments kept verbatim, for example ``src/attr/_make.py::_ClassBuilder.build_class`` or
``pkg/core.py::outer.<locals>.inner``.

FunctionIndex derives identities from the AST of the pinned source. Nothing is imported
or executed. Conventions shared by every lane:

* Functions are ``def`` / ``async def`` statements. Lambdas are not functions: they are
  left out of ``functions()`` unless ``include_lambdas=True`` and are never returned by
  ``enclosing_function``.
* ``enclosing_function(path, line)`` returns the innermost function whose *body* contains
  the line. Decorators, default values and annotations are evaluated in the enclosing
  scope, so their lines (above the first body line) belong to that scope. Lambdas,
  comprehensions and generator expressions are transparent. Class-body lines outside any
  method and module-level lines return ``None``; they are never function callers.
* Duplicate definitions (``typing.overload`` stubs, property setters, conditional
  definitions) share one identity, exactly as their ``__qualname__`` values do.
* Qualnames follow CPython's compiler: ``.<locals>`` after a function or lambda scope, a
  plain dot after a class, and a bare name for a function or class whose name the
  enclosing scope declares ``global``.
* Source is parsed by the running interpreter. Files that do not parse (for example newer
  syntax under Python 3.11) have no functions and are listed in ``FunctionIndex.errors``.
"""
from __future__ import annotations

import ast
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import tokenize

__all__ = [
    "EXCLUDED_DIR_NAMES", "EXCLUDED_FILE_NAMES", "IDENTITY_SEPARATOR", "MODULE_QUALNAME",
    "FunctionIndex", "file_sha256s", "function_identity", "in_package_scope",
    "is_test_or_excluded_path", "normalize_package_dirs", "normalize_relpath",
    "package_files", "scan_package_files", "source_tree_sha256", "split_identity",
]

IDENTITY_SEPARATOR = "::"
MODULE_QUALNAME = "<module>"
# Directory names excluded at any depth, and file names excluded anywhere (section 2).
EXCLUDED_DIR_NAMES = frozenset({"tests", "test", "testing", "docs", "examples", "benchmarks", "scripts"})
EXCLUDED_FILE_NAMES = frozenset({"conftest.py", "setup.py", "noxfile.py"})
# PEP 709 (3.12) inlines list/set/dict comprehensions; generator expressions keep a scope.
_SCOPED_COMPREHENSIONS = ((ast.GeneratorExp,) if sys.version_info >= (3, 12)
                          else (ast.GeneratorExp, ast.ListComp, ast.SetComp, ast.DictComp))
_COMPREHENSION_NAMES = {ast.GeneratorExp: "<genexpr>", ast.ListComp: "<listcomp>",
                        ast.SetComp: "<setcomp>", ast.DictComp: "<dictcomp>"}
_SPECIAL_SEGMENT = re.compile(r"<[a-z]+>")
_LINE_BREAK = re.compile(r"\r\n|\r|\n")


def _canonical_json(value) -> bytes:
    """Same encoding as absent_edges.data.canonical_json (kept local: stdlib-only module)."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def normalize_relpath(relpath) -> str:
    """Return a POSIX path relative to the repository root, or raise ValueError."""
    if isinstance(relpath, os.PathLike):
        relpath = os.fspath(relpath)
    if not isinstance(relpath, str) or not relpath or "\x00" in relpath:
        raise ValueError(f"relpath must be a non-empty string: {relpath!r}")
    text = relpath.replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    if text.startswith("/") or re.match(r"[A-Za-z]:", text):
        raise ValueError(f"relpath must be relative to the repository root: {relpath!r}")
    if any(part in ("", ".", "..") for part in text.split("/")):
        raise ValueError(f"relpath must be a normalized relative path: {relpath!r}")
    if IDENTITY_SEPARATOR in text:
        raise ValueError(f"relpath cannot contain {IDENTITY_SEPARATOR!r}: {relpath!r}")
    return text


def _check_qualname(qualname) -> str:
    if not isinstance(qualname, str) or not qualname:
        raise ValueError(f"qualname must be a non-empty string: {qualname!r}")
    for segment in qualname.split("."):
        if not (segment.isidentifier() or _SPECIAL_SEGMENT.fullmatch(segment)):
            raise ValueError(f"invalid qualname: {qualname!r}")
    return qualname


def function_identity(relpath, qualname) -> str:
    """``"<relpath>::<qualname>"`` with a normalized POSIX relpath."""
    return normalize_relpath(relpath) + IDENTITY_SEPARATOR + _check_qualname(qualname)


def split_identity(identity) -> tuple:
    """Inverse of function_identity: ``(relpath, qualname)``. Rejects unnormalized input."""
    if not isinstance(identity, str) or identity.count(IDENTITY_SEPARATOR) != 1:
        raise ValueError(f"not a function identity: {identity!r}")
    relpath, qualname = identity.split(IDENTITY_SEPARATOR)
    if normalize_relpath(relpath) != relpath:
        raise ValueError(f"identity relpath is not normalized: {identity!r}")
    return relpath, _check_qualname(qualname)


def is_test_or_excluded_path(relpath) -> bool:
    """True for test, docs, example, benchmark and script paths and setup/nox files."""
    parts = normalize_relpath(relpath).split("/")
    if any(part in EXCLUDED_DIR_NAMES for part in parts):
        return True
    name = parts[-1]
    if name in EXCLUDED_FILE_NAMES:
        return True
    return name.endswith(".py") and (name.startswith("test_") or name.endswith("_test.py"))


def normalize_package_dirs(package_dirs) -> tuple:
    """Sorted unique package directories (or single-module ``.py`` files); ``"."`` -> ``""``."""
    if isinstance(package_dirs, (str, os.PathLike)):
        package_dirs = [package_dirs]
    result = set()
    for value in package_dirs:
        text = os.fspath(value) if isinstance(value, os.PathLike) else value
        if not isinstance(text, str):
            raise ValueError(f"package dir must be a string: {value!r}")
        text = text.replace("\\", "/").rstrip("/")
        while text.startswith("./"):
            text = text[2:]
        result.add("" if text in ("", ".") else normalize_relpath(text))
    if not result:
        raise ValueError("at least one package dir is required")
    return tuple(sorted(result))


def in_package_scope(relpath, package_dirs) -> bool:
    """True for a ``.py`` file under a package dir that is not a test/excluded path."""
    path = normalize_relpath(relpath)
    if not path.endswith(".py") or is_test_or_excluded_path(path):
        return False
    return any(directory == "" or path == directory or path.startswith(directory + "/")
               for directory in normalize_package_dirs(package_dirs))


def scan_package_files(root, package_dirs) -> dict:
    """Walk the package dirs (no symlinks followed) and return the in-scope regular files.

    Returns ``{"package_dirs", "files", "skipped_symlinks", "missing_package_dirs"}``;
    every list is sorted, so the result is deterministic.
    """
    root = Path(root)
    directories = normalize_package_dirs(package_dirs)
    files, symlinks, missing = set(), set(), []
    for directory in directories:
        base = root / directory if directory else root
        if directory.endswith(".py"):
            if base.is_symlink():
                symlinks.add(directory)
            elif base.is_file():
                if in_package_scope(directory, directories):
                    files.add(directory)
            else:
                missing.append(directory)
            continue
        if base.is_symlink() or not base.is_dir():
            missing.append(directory)
            continue
        for current, dirnames, filenames in os.walk(base, followlinks=False):
            current_path = Path(current)
            relative_dir = current_path.relative_to(root).as_posix()
            prefix = "" if relative_dir == "." else relative_dir + "/"
            kept = []
            for name in sorted(dirnames):
                if (current_path / name).is_symlink():
                    symlinks.add(prefix + name + "/")
                elif name not in EXCLUDED_DIR_NAMES and name != "__pycache__" and not name.startswith("."):
                    kept.append(name)
            dirnames[:] = kept
            for name in sorted(filenames):
                if not name.endswith(".py"):
                    continue
                relative = prefix + name
                full = current_path / name
                if full.is_symlink():
                    symlinks.add(relative)
                elif full.is_file() and in_package_scope(relative, directories):
                    files.add(relative)
    return {"package_dirs": list(directories), "files": sorted(files),
            "skipped_symlinks": sorted(symlinks), "missing_package_dirs": sorted(missing)}


def package_files(root, package_dirs) -> list:
    """Sorted repository-relative paths of the in-scope regular ``.py`` files."""
    return scan_package_files(root, package_dirs)["files"]


def file_sha256s(root, relpaths) -> dict:
    root = Path(root)
    return {normalize_relpath(path): hashlib.sha256((root / path).read_bytes()).hexdigest()
            for path in relpaths}


def source_tree_sha256(file_hashes: dict) -> str:
    """sha256 of the canonical JSON ``{relpath: sha256(bytes)}`` map (graphs.py convention)."""
    return hashlib.sha256(_canonical_json(dict(file_hashes))).hexdigest()


class _Frame:
    __slots__ = ("kind", "qualname", "global_names", "entry")

    def __init__(self, kind, qualname, global_names=frozenset(), entry=None):
        self.kind, self.qualname = kind, qualname
        self.global_names, self.entry = global_names, entry


def _scope_globals(node) -> frozenset:
    """Names declared ``global`` directly in this function or class scope."""
    names, stack = set(), list(node.body)
    while stack:
        current = stack.pop()
        if isinstance(current, ast.Global):
            names.update(current.names)
        elif not isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            stack.extend(ast.iter_child_nodes(current))
    return frozenset(names)


def _child_qualname(frame, name, *, binds_name) -> str:
    """CPython compiler_set_qualname for a child scope named ``name`` inside ``frame``."""
    if frame.kind == "module":
        return name
    if binds_name and name in frame.global_names:
        return name
    if frame.kind in ("function", "lambda"):
        return frame.qualname + ".<locals>." + name
    return frame.qualname + "." + name


def _signature_parts(node) -> list:
    """Sub-expressions of a def evaluated in the enclosing scope."""
    args = node.args
    parts = list(node.decorator_list) + list(args.defaults)
    parts += [value for value in args.kw_defaults if value is not None]
    for argument in (*args.posonlyargs, *args.args, args.vararg, *args.kwonlyargs, args.kwarg):
        if argument is not None and argument.annotation is not None:
            parts.append(argument.annotation)
    if node.returns is not None:
        parts.append(node.returns)
    return parts


def _statement_start(statement) -> int:
    """First line of a statement; a decorated def/class starts at its first decorator."""
    decorators = getattr(statement, "decorator_list", None)
    return min([statement.lineno] + [item.lineno for item in decorators]) if decorators else statement.lineno


def _decode_lines(raw: bytes) -> list:
    try:
        encoding, _ = tokenize.detect_encoding(io.BytesIO(raw).readline)
    except SyntaxError:
        encoding = "utf-8"
    text = raw.decode(encoding, errors="replace")
    # CPython line numbering: only \r\n, \r and \n end a line (not \f or \v).
    return _LINE_BREAK.split(text)


class _FileIndex:
    __slots__ = ("functions", "classes", "owner", "lines", "tree", "sha256", "nodes_by_line")

    def __init__(self, functions=(), classes=(), owner=(), lines=(), tree=None, sha256=None,
                 nodes_by_line=None):
        self.functions, self.classes, self.owner = list(functions), list(classes), list(owner)
        self.lines, self.tree, self.sha256 = list(lines), tree, sha256
        self.nodes_by_line = nodes_by_line or {}


def _index_source(relpath: str, raw: bytes) -> _FileIndex:
    tree = ast.parse(raw, filename=relpath)
    lines = _decode_lines(raw)
    functions, classes, regions, nodes_by_line = [], [], [], {}
    module = _Frame("module", None)
    stack = [(node, module) for node in reversed(tree.body)]
    while stack:
        node, frame = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            qualname = _child_qualname(frame, node.name, binds_name=True)
            entry = {
                "identity": function_identity(relpath, qualname), "qualname": qualname,
                "name": node.name, "lineno": node.lineno, "end_lineno": node.end_lineno,
                "first_lineno": min([node.lineno] + [item.lineno for item in node.decorator_list]),
                "body_lineno": _statement_start(node.body[0]), "col_offset": node.col_offset,
                "decorators": [ast.unparse(item) for item in node.decorator_list],
                "is_async": isinstance(node, ast.AsyncFunctionDef), "is_generator": False,
                "is_lambda": False, "is_method": frame.kind == "class",
                "is_dunder": len(node.name) > 4 and node.name.startswith("__") and node.name.endswith("__"),
                "parent_qualname": frame.qualname, "occurrence": 0,
            }
            functions.append(entry)
            nodes_by_line.setdefault(node.lineno, []).append(node)
            regions.append((entry["body_lineno"], node.end_lineno, entry["identity"]))
            inner = _Frame("function", qualname, _scope_globals(node), entry)
            stack.extend((child, inner) for child in reversed(node.body))
            stack.extend((child, frame) for child in reversed(_signature_parts(node)))
        elif isinstance(node, ast.ClassDef):
            qualname = _child_qualname(frame, node.name, binds_name=True)
            body_lineno = _statement_start(node.body[0])
            classes.append({"qualname": qualname, "name": node.name, "lineno": node.lineno,
                            "end_lineno": node.end_lineno, "body_lineno": body_lineno,
                            "col_offset": node.col_offset, "parent_qualname": frame.qualname,
                            "decorators": [ast.unparse(item) for item in node.decorator_list]})
            nodes_by_line.setdefault(node.lineno, []).append(node)
            regions.append((body_lineno, node.end_lineno, None))
            inner = _Frame("class", qualname, _scope_globals(node))
            outer = list(node.decorator_list) + list(node.bases) + [item.value for item in node.keywords]
            stack.extend((child, inner) for child in reversed(node.body))
            stack.extend((child, frame) for child in reversed(outer))
        elif isinstance(node, ast.Lambda):
            qualname = _child_qualname(frame, "<lambda>", binds_name=False)
            entry = {
                "identity": function_identity(relpath, qualname), "qualname": qualname,
                "name": "<lambda>", "lineno": node.lineno, "end_lineno": node.end_lineno,
                "first_lineno": node.lineno, "body_lineno": node.body.lineno,
                "col_offset": node.col_offset, "decorators": [], "is_async": False,
                "is_generator": False, "is_lambda": True, "is_method": False, "is_dunder": False,
                "parent_qualname": frame.qualname, "occurrence": 0,
            }
            functions.append(entry)
            inner = _Frame("lambda", qualname, frozenset(), entry)
            outer = list(node.args.defaults) + [value for value in node.args.kw_defaults if value is not None]
            stack.append((node.body, inner))
            stack.extend((child, frame) for child in reversed(outer))
        elif isinstance(node, _SCOPED_COMPREHENSIONS):
            qualname = _child_qualname(frame, _COMPREHENSION_NAMES[type(node)], binds_name=False)
            inner = _Frame("comprehension", qualname)
            elements = [node.key, node.value] if isinstance(node, ast.DictComp) else [node.elt]
            for position, generator in enumerate(node.generators):
                stack.append((generator.iter, frame if position == 0 else inner))
                stack.append((generator.target, inner))
                stack.extend((condition, inner) for condition in generator.ifs)
            stack.extend((element, inner) for element in elements)
        else:
            if isinstance(node, (ast.Yield, ast.YieldFrom)) and frame.entry is not None:
                frame.entry["is_generator"] = True
            stack.extend((child, frame) for child in reversed(list(ast.iter_child_nodes(node))))
    functions.sort(key=lambda item: (item["lineno"], item["col_offset"], item["is_lambda"]))
    classes.sort(key=lambda item: (item["lineno"], item["col_offset"]))
    seen = {}
    for entry in functions:
        entry["occurrence"] = seen.get(entry["identity"], 0)
        seen[entry["identity"]] = entry["occurrence"] + 1
    # Scope regions are laminar (properly nested): paint outer regions first so the
    # innermost def/class body owns each line. Class bodies paint None.
    owner = [None] * max(len(lines), 1)
    for start, end, value in sorted(regions, key=lambda item: (item[0], -item[1])):
        for line in range(start, min(end, len(owner)) + 1):
            owner[line - 1] = value
    return _FileIndex(functions, classes, owner, lines, tree, hashlib.sha256(raw).hexdigest(),
                      nodes_by_line)


def _copy_entry(entry: dict) -> dict:
    return {**entry, "decorators": list(entry["decorators"])}


class FunctionIndex:
    """Lazy per-file AST index of one repository checkout rooted at ``root``."""

    def __init__(self, root, *, max_file_bytes=16 * 1024 * 1024):
        root = Path(root)
        if not root.is_dir():
            raise ValueError(f"FunctionIndex root is not a directory: {root}")
        self.root = root.resolve()
        self.max_file_bytes = int(max_file_bytes)
        self.errors = {}   # relpath -> "ExceptionType: message" for files that exist but fail
        self._files = {}

    def _file(self, relpath) -> _FileIndex:
        path = normalize_relpath(relpath)
        cached = self._files.get(path)
        if cached is not None:
            return cached
        record = _FileIndex()
        full = self.root / path
        try:
            resolved = full.resolve(strict=True)
        except (OSError, RuntimeError):
            resolved = None   # missing file: no functions, not an error
        if resolved is not None:
            try:
                if not resolved.is_relative_to(self.root):
                    raise ValueError("path resolves outside the index root")
                if not resolved.is_file():
                    raise ValueError("not a regular file")
                raw = resolved.read_bytes()
                if len(raw) > self.max_file_bytes:
                    raise ValueError("file exceeds the index byte budget")
                record = _index_source(path, raw)
            except (OSError, ValueError, SyntaxError, RecursionError, MemoryError) as error:
                self.errors[path] = f"{type(error).__name__}: {error}"
        self._files[path] = record
        return record

    def functions(self, relpath, *, include_lambdas=False) -> list:
        """Definitions in source order: dicts with identity, qualname, lineno, end_lineno,
        decorators, is_async, is_generator, is_lambda (plus name, first_lineno = the
        decorated co_firstlineno, body_lineno, col_offset, is_method, is_dunder,
        parent_qualname and occurrence among same-identity definitions)."""
        return [_copy_entry(entry) for entry in self._file(relpath).functions
                if include_lambdas or not entry["is_lambda"]]

    def classes(self, relpath) -> list:
        return [dict(entry, decorators=list(entry["decorators"])) for entry in self._file(relpath).classes]

    def enclosing_function(self, relpath, line):
        """Identity of the innermost function whose body contains ``line``, else None."""
        if isinstance(line, bool) or not isinstance(line, int):
            raise TypeError("line must be an int")
        owner = self._file(relpath).owner
        return owner[line - 1] if 1 <= line <= len(owner) else None

    def functions_at(self, relpath, lineno, name=None) -> list:
        """Non-lambda definitions whose ``def`` keyword is on ``lineno`` (optionally named)."""
        return [_copy_entry(entry) for entry in self._file(relpath).functions
                if not entry["is_lambda"] and entry["lineno"] == lineno
                and (name is None or entry["name"] == name)]

    def definition_node(self, relpath, lineno, name, kind="function"):
        """The AST FunctionDef/AsyncFunctionDef (``kind="function"``) or ClassDef node."""
        wanted = (ast.FunctionDef, ast.AsyncFunctionDef) if kind == "function" else (ast.ClassDef,)
        for node in self._file(relpath).nodes_by_line.get(lineno, []):
            if isinstance(node, wanted) and node.name == name:
                return node
        return None

    def tree(self, relpath):
        return self._file(relpath).tree

    def file_sha256(self, relpath):
        return self._file(relpath).sha256

    def line_text(self, relpath, line) -> str:
        lines = self._file(relpath).lines
        return lines[line - 1] if 1 <= line <= len(lines) else ""

    def char_column(self, relpath, line, byte_column) -> int:
        """Convert an AST UTF-8 byte column to a code-point column (jedi/parso columns)."""
        text = self.line_text(relpath, line)
        if text.isascii():
            return byte_column
        return len(text.encode("utf-8")[:byte_column].decode("utf-8", errors="ignore"))

    def package_functions(self, package_dirs, *, include_lambdas=False) -> list:
        """All definitions in the in-scope files under ``package_dirs``, sorted by path/line."""
        return [entry for path in package_files(self.root, package_dirs)
                for entry in self.functions(path, include_lambdas=include_lambdas)]
