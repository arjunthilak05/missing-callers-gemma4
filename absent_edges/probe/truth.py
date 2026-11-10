"""Exp A ground truth at call-site level (docs/EXPA_DESIGN_V2_DELTA.md D2, D11.5; docs/EXPA_DESIGN.md 2, 6).

Inputs are call-site records ``{"caller", "callee", "path", "line"}`` (JSONL) from the
runtime trace, backend L (LocAgent) and backend J (jedi), plus lexical candidates
computed here from the pinned source. Identities are ``"relpath::qualname"``
(``absent_edges.probe.identity``).

Callers (unchanged from v1)::

    V(X) = {C : the trace observed C->X} | {C : L and J both report C->X}
    U(X) = every other reported caller (one static backend only, or lexical only)

Only package-scope function/method callers count. Module-level and class-body code is
never a caller. A caller frame that is not an indexed function (lambda, generator
expression) is attributed to its enclosing function by the call-site line. A recursive
self-call never enters V(X); it is kept in U(X) flagged ``self``. Lexical candidates
never enter V(X).

Sites (D2, spans per D11.5). For a caller C of X, ``sites(C, X)`` are invocation spans
``{path, lineno, end_lineno, kind, witnesses}`` with ``kind`` in call | reference |
dynamic and ``witnesses`` a subset of trace | L | J | ast. Spans are tight: never an
enclosing statement.

* AST sites, from C's own code (its body; the decorators, defaults and annotations of a
  def or class nested in C; lambda and comprehension bodies; never the body of a
  nested def or class):

  - ``call``: an ``ast.Call`` whose callee expression is a Name equal to X's simple
    name or to a frozen import alias of X, or an Attribute whose attribute is X's
    simple name. Span: the Call node's ``lineno..end_lineno``. A bare decorator
    ``@name`` / ``@obj.name`` applied inside C is an invocation too (span: the
    decorator expression's lines).
  - ``reference``: such a Name/Attribute passed as an argument of a call (positional,
    keyword or starred, possibly inside a list/tuple/set/dict value, a conditional
    expression or a boolean operation). Span: the reference node's own
    ``lineno..end_lineno`` (normally one line), not the receiving call.

* Trace sites: each traced call line is mapped to the innermost AST site of C whose
  span contains it (that site gets the ``trace`` witness). When no AST site contains
  it, the site is that single line with kind ``dynamic``: ``getattr`` with a string, a
  local variable bound to X, a lazy iterator consumed on a later line, or a callback
  invoked by a multi-line receiving call whose first line is not the reference line.

* L and J call-site lines are witnesses: each one is attached to the innermost site of
  C containing it. A line inside no site is counted as unplaced evidence (it is never
  turned into a site). A V caller verified only by L and J can therefore have no site
  at all (for example ``fn = X`` followed by ``fn()``); it is never C* (C* is
  runtime-witnessed) and the build counts such callers.

``runtime_witnessed``: C has at least one trace site (equivalently, the trace observed
C->X). Stratum of a caller, from site kinds and alias resolution: ``direct`` (a call
site through X's simple name or an attribute ending in it), else ``alias`` (a call site
through an import alias), else ``callback`` (a reference site), else ``dynamic`` (only
dynamic sites); ``None`` when C has no site at all (possible only for a caller verified
by L and J through a non-argument reference, never for a runtime-witnessed caller).

Frozen import aliases of a module-level function X: ``from M import X as A`` statements
(anywhere in a package file) where M resolves to X's module, or to a package
``__init__`` that itself imports X from X's module; the alias applies in that file.
Methods and nested functions have no import aliases.

Lexical names (frozen per target, D10.11): X's qualified and simple names and the frozen
import aliases above, nothing else. The lexical scan (every package line naming one of
them as a whole word) is the ``grep_all`` input, the ``lexical_name_match`` scoring
category and the source of lexical-only U callers. The live tool's broader rule (every
``as NAME`` on an import line that names the simple name, ``tools.lexical_candidates``
and ``graphs.GraphBackend.source_reference_search``) is used only by C4a and the live
``source_reference_search``: it also picks up aliases of other objects (``from itertools
import batched as itertools_batched`` for a package function ``batched``).
"""
from __future__ import annotations

import ast
from collections import Counter
import io
import json
from pathlib import Path, PurePosixPath
import posixpath
import re
import tokenize

from .identity import (FunctionIndex, file_sha256s, function_identity, in_package_scope,
                       is_test_or_excluded_path, normalize_package_dirs, package_files,
                       source_tree_sha256, split_identity)

TRUTH_PROTOCOL = "expA-truth-v2"
SOURCES = ("trace", "L", "J", "lexical")
VERIFYING_SOURCES = ("trace", "L", "J")
WITNESSES = ("trace", "L", "J", "ast")
SITE_KINDS = ("call", "reference", "dynamic")
STRATA = ("direct", "alias", "callback", "dynamic")
_KIND_RANK = {kind: rank for rank, kind in enumerate(SITE_KINDS)}
_SOURCE_NAMES = {"trace": "trace", "traces": "trace", "tracer": "trace", "runtime": "trace",
                 "L": "L", "locagent": "L", "J": "J", "jedi": "J", "lexical": "lexical"}
_NEWLINES = re.compile(r"\r\n|\r|\n")

__all__ = ["TRUTH_PROTOCOL", "SOURCES", "VERIFYING_SOURCES", "WITNESSES", "SITE_KINDS", "STRATA",
           "PackageScope", "TruthTable", "build_truth", "canonical_source", "decode_source",
           "function_identity", "import_aliases", "lexical_call_sites", "lexical_lines", "lexical_names",
           "locagent_indexed_identities", "read_call_sites", "simple_name", "source_lines",
           "stratum_of", "validate_call_site"]


def canonical_source(name):
    try:
        return _SOURCE_NAMES[name]
    except KeyError:
        raise ValueError(f"Unknown call-site source {name!r}") from None


def _relpath_ok(path):
    return (isinstance(path, str) and bool(path) and not path.startswith("/") and "\\" not in path
            and ".." not in PurePosixPath(path).parts and ":" not in path)


def validate_call_site(record, where="record"):
    """Return the four call-site fields of a record, or raise ValueError."""
    if not isinstance(record, dict):
        raise ValueError(f"{where}: call-site record must be an object")
    for key in ("caller", "callee"):
        value = record.get(key)
        if not isinstance(value, str) or "::" not in value or len(value.encode("utf-8")) > 4096:
            raise ValueError(f"{where}: {key} must be a bounded 'relpath::qualname' identity")
        if not _relpath_ok(split_identity(value)[0]):
            raise ValueError(f"{where}: {key} has an invalid relpath")
    path, line = record.get("path"), record.get("line")
    if path is not None and not _relpath_ok(path):
        raise ValueError(f"{where}: path must be a POSIX path relative to the repository root")
    if line is not None and (type(line) is not int or line < 1):
        raise ValueError(f"{where}: line must be a positive integer or null")
    return {"caller": record["caller"], "callee": record["callee"], "path": path, "line": line}


def read_call_sites(path):
    """Read one JSONL call-site file. Blank lines are skipped; every record is validated."""
    records = []
    with open(path, "rb") as stream:
        for number, raw in enumerate(stream, 1):
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise ValueError(f"{path}:{number}: invalid JSON") from error
            records.append(validate_call_site(value, where=f"{path}:{number}"))
    return records


def source_lines(text):
    """Split on the newlines the Python tokenizer counts (keeps AST line numbers aligned)."""
    lines = _NEWLINES.split(text)
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def decode_source(raw):
    """Decode Python source as the interpreter does (BOM / PEP 263 cookie, else UTF-8)."""
    try:
        encoding, _ = tokenize.detect_encoding(io.BytesIO(raw).readline)
    except (SyntaxError, LookupError):
        encoding = "utf-8"
    return raw.decode(encoding, errors="replace")


def simple_name(identity):
    return split_identity(identity)[1].rsplit(".", 1)[-1]


def import_root(package_dirs):
    """Directory that must be on sys.path to import the package (``src`` for src layouts)."""
    roots = {posixpath.dirname(directory) if directory else ""
             for directory in normalize_package_dirs(package_dirs)}
    return roots.pop() if len(roots) == 1 else ""


class PackageScope:
    """Package-scope source of one pinned repository checkout (read-only)."""

    def __init__(self, repo_root, package_dirs, index=None):
        self.root = Path(repo_root)
        if not self.root.is_dir():
            raise ValueError(f"Repository root is not a directory: {repo_root}")
        self.package_dirs = tuple(package_dirs)
        normalize_package_dirs(self.package_dirs)  # validates; raises ValueError
        self.index = index if index is not None else FunctionIndex(self.root)
        self._files = None
        self._functions, self._lines, self._enclosing = {}, {}, {}
        self._table = self._duplicates = self._modules = None

    def contains(self, relpath):
        try:
            return (_relpath_ok(relpath) and bool(in_package_scope(relpath, self.package_dirs))
                    and not is_test_or_excluded_path(relpath))
        except ValueError:
            return False

    def files(self):
        """Sorted package-scope ``.py`` relpaths (identity.package_files: no symlinks, no tests)."""
        if self._files is None:
            self._files = tuple(package_files(self.root, self.package_dirs))
        return self._files

    def functions(self, relpath):
        if relpath not in self._functions:
            self._functions[relpath] = [dict(item) for item in self.index.functions(relpath)]
        return self._functions[relpath]

    def table(self):
        """identity -> metadata (plus ``path``) for every non-lambda package-scope function."""
        if self._table is None:
            table, counts = {}, Counter()
            for relpath in self.files():
                for info in self.functions(relpath):
                    if info.get("is_lambda"):
                        continue
                    identity = info["identity"]
                    counts[identity] += 1
                    table.setdefault(identity, {**info, "path": relpath})
            self._table = table
            self._duplicates = frozenset(identity for identity, count in counts.items() if count > 1)
        return self._table

    def duplicates(self):
        self.table()
        return self._duplicates

    def definitions(self, identity):
        """Every AST def node with this identity (duplicate definitions share one identity)."""
        relpath = split_identity(identity)[0]
        nodes = []
        for info in self.functions(relpath):
            if info["identity"] == identity and not info.get("is_lambda"):
                node = self.index.definition_node(relpath, info["lineno"], info["name"])
                if node is not None:
                    nodes.append(node)
        return nodes

    def lines(self, relpath):
        if relpath not in self._lines:
            if not _relpath_ok(relpath):
                raise ValueError("Invalid relpath")
            self._lines[relpath] = source_lines(decode_source((self.root / relpath).read_bytes()))
        return self._lines[relpath]

    def line_text(self, relpath, line):
        if line is None or not self.contains(relpath) or not (self.root / relpath).is_file():
            return None
        lines = self.lines(relpath)
        return lines[line - 1] if 1 <= line <= len(lines) else None

    def enclosing(self, relpath, line):
        key = (relpath, line)
        if key not in self._enclosing:
            self._enclosing[key] = self.index.enclosing_function(relpath, line)
        return self._enclosing[key]

    def source_sha256(self):
        """identity.source_tree_sha256 over the package-scope files (the static manifests' value)."""
        return source_tree_sha256(file_sha256s(self.root, self.files()))

    # ---------------------------------------------------------------- modules
    def modules(self):
        """{relpath: dotted module name} for every package file (``__init__`` -> its package)."""
        if self._modules is None:
            root = import_root(self.package_dirs)
            modules = {}
            for relpath in self.files():
                relative = relpath[len(root) + 1:] if root else relpath
                parts = relative[:-3].split("/")
                if parts[-1] == "__init__":
                    parts = parts[:-1]
                if parts and all(part.isidentifier() for part in parts):
                    modules[relpath] = ".".join(parts)
            self._modules = modules
        return self._modules

    def module_file(self, module):
        for relpath, name in self.modules().items():
            if name == module:
                return relpath
        return None

    def resolve_from(self, relpath, module, level):
        """Absolute module named by ``from <level dots><module> import ...`` in ``relpath``."""
        if not level:
            return module
        current = self.modules().get(relpath)
        if current is None:
            return None
        package = current if relpath.rsplit("/", 1)[-1] == "__init__.py" else current.rpartition(".")[0]
        parts = package.split(".") if package else []
        if level - 1 > len(parts):
            return None
        base = parts[:len(parts) - (level - 1)]
        if module:
            base += module.split(".")
        return ".".join(base) or None


# ---------------------------------------------------------------------- aliases
def _import_froms(scope, relpath):
    tree = scope.index.tree(relpath)
    return [] if tree is None else [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]


def _binds_from(scope, relpath, simple, target_module):
    """True when ``relpath`` binds ``simple`` by ``from <target module> import simple``."""
    for node in _import_froms(scope, relpath):
        if scope.resolve_from(relpath, node.module, node.level) != target_module:
            continue
        if any(alias.name == simple and alias.asname in (None, simple) for alias in node.names):
            return True
    return False


def import_aliases(scope, target):
    """{relpath: sorted alias names} of AST-resolved ``from M import X as A`` aliases of X."""
    relpath, qualname = split_identity(target)
    if "." in qualname:
        return {}
    target_module = scope.modules().get(relpath)
    if target_module is None:
        return {}
    result = {}
    for path in scope.files():
        for node in _import_froms(scope, path):
            for alias in node.names:
                if alias.name != qualname or not alias.asname or alias.asname == qualname:
                    continue
                module = scope.resolve_from(path, node.module, node.level)
                if module is None:
                    continue
                accepted = module == target_module
                if not accepted:
                    exporter = scope.module_file(module)
                    accepted = (exporter is not None and exporter.rsplit("/", 1)[-1] == "__init__.py"
                                and _binds_from(scope, exporter, qualname, target_module))
                if accepted:
                    result.setdefault(path, set()).add(alias.asname)
    return {path: sorted(names) for path, names in sorted(result.items())}


# ---------------------------------------------------------------------- AST sites
_CONTAINERS = (ast.List, ast.Tuple, ast.Set, ast.Starred)


class _Region:
    """AST nodes evaluated in one caller's own scope, with parents and decorator ids."""
    __slots__ = ("nodes", "parents", "decorators")

    def __init__(self, definitions):
        self.nodes, self.parents, self.decorators = [], {}, set()
        stack = [(child, definition) for definition in definitions for child in reversed(definition.body)]
        while stack:
            node, parent = stack.pop()
            self.parents[id(node)] = parent
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.decorators.update(id(item) for item in node.decorator_list)
                stack.extend((part, node) for part in reversed(_signature_parts(node)))
                continue  # the nested body is another function's scope
            if isinstance(node, ast.ClassDef):
                self.decorators.update(id(item) for item in node.decorator_list)
                outer = list(node.decorator_list) + list(node.bases) + [item.value for item in node.keywords]
                stack.extend((part, node) for part in reversed(outer))
                continue  # class bodies are never function callers
            self.nodes.append(node)
            stack.extend((child, node) for child in reversed(list(ast.iter_child_nodes(node))))


def _signature_parts(node):
    """Sub-expressions of a nested def evaluated in the enclosing (caller's) scope."""
    args = node.args
    parts = list(node.decorator_list) + list(args.defaults) + [value for value in args.kw_defaults if value is not None]
    for argument in (*args.posonlyargs, *args.args, args.vararg, *args.kwonlyargs, args.kwarg):
        if argument is not None and argument.annotation is not None:
            parts.append(argument.annotation)
    if node.returns is not None:
        parts.append(node.returns)
    return parts


def _via(node, simple, aliases):
    if isinstance(node, ast.Name):
        if node.id == simple:
            return "name"
        if node.id in aliases:
            return "alias"
    elif isinstance(node, ast.Attribute) and node.attr == simple:
        return "attribute"
    return None


def _name_line(node):
    return node.end_lineno if isinstance(node, ast.Attribute) else node.lineno


def _receiver(node, parents):
    """The Call that receives ``node`` as an argument (through containers/conditionals), else None."""
    current, parent = node, parents.get(id(node))
    while True:
        if isinstance(parent, _CONTAINERS):
            pass
        elif isinstance(parent, ast.Dict) and any(value is current for value in parent.values):
            pass
        elif isinstance(parent, ast.IfExp) and parent.test is not current:
            pass
        elif isinstance(parent, ast.BoolOp):
            pass
        else:
            break
        current, parent = parent, parents.get(id(parent))
    if isinstance(parent, ast.keyword):
        current, parent = parent, parents.get(id(parent))
    if isinstance(parent, ast.Call) and parent.func is not current and (
            any(item is current for item in parent.args) or any(item is current for item in parent.keywords)):
        return parent
    return None


def _ast_sites(region, relpath, simple, aliases):
    """Raw AST sites: dicts {path, lineno, end_lineno, kind, via, anchor}."""
    sites = []
    for node in region.nodes:
        if isinstance(node, ast.Call):
            via = _via(node.func, simple, aliases)
            if via:
                sites.append({"path": relpath, "lineno": node.lineno, "end_lineno": node.end_lineno,
                              "kind": "call", "via": via, "anchor": _name_line(node.func)})
            continue
        if not isinstance(node, (ast.Name, ast.Attribute)) or not isinstance(getattr(node, "ctx", None), ast.Load):
            continue
        via = _via(node, simple, aliases)
        if not via:
            continue
        parent = region.parents.get(id(node))
        if isinstance(parent, ast.Call) and parent.func is node:
            continue  # already the Call site above
        if id(node) in region.decorators:
            sites.append({"path": relpath, "lineno": node.lineno, "end_lineno": node.end_lineno,
                          "kind": "call", "via": via, "anchor": _name_line(node)})
            continue
        if _receiver(node, region.parents) is not None:
            sites.append({"path": relpath, "lineno": node.lineno, "end_lineno": node.end_lineno,
                          "kind": "reference", "via": via, "anchor": _name_line(node)})
    return sites


def _innermost(sites, path, line):
    containing = [site for site in sites if site["path"] == path and site["lineno"] <= line <= site["end_lineno"]]
    if not containing:
        return None
    return min(containing, key=lambda site: (site["end_lineno"] - site["lineno"], -site["lineno"],
                                             _KIND_RANK[site["kind"]]))


def stratum_of(raw_sites):
    """Caller stratum from raw site kinds and how the call names X (see the module docstring)."""
    if any(site["kind"] == "call" and site["via"] in ("name", "attribute") for site in raw_sites):
        return "direct"
    if any(site["kind"] == "call" and site["via"] == "alias" for site in raw_sites):
        return "alias"
    if any(site["kind"] == "reference" for site in raw_sites):
        return "callback"
    if any(site["kind"] == "dynamic" for site in raw_sites):
        return "dynamic"
    return None


def caller_sites(scope, caller, callee, evidence, aliases, regions=None):
    """sites(C, X) with witnesses. Returns (sites, detailed, raw_sites, notes).

    ``evidence`` is an iterable of (path, line, source) evidence lines of C->X;
    ``aliases`` is ``import_aliases(scope, callee)``. ``sites`` are the serializable
    D2 dicts sorted by (path, lineno, end_lineno, kind); ``detailed`` are the same
    dicts plus ``anchors`` (the lines of the name tokens, or the traced line of a
    dynamic site); ``raw_sites`` are the unmerged sites with ``via`` (name |
    attribute | alias, None for dynamic) for the stratum.
    """
    relpath = split_identity(caller)[0]
    notes = Counter()
    if regions is not None and caller in regions:
        region = regions[caller]
    else:
        region = _Region(scope.definitions(caller))
        if regions is not None:
            regions[caller] = region
    raw = _ast_sites(region, relpath, simple_name(callee), set(aliases.get(relpath, ())))
    merged = {}
    for site in raw:
        key = (site["path"], site["lineno"], site["end_lineno"])
        entry = merged.setdefault(key, {"path": site["path"], "lineno": site["lineno"],
                                        "end_lineno": site["end_lineno"], "kind": site["kind"],
                                        "witnesses": {"ast"}, "anchors": set()})
        if _KIND_RANK[site["kind"]] < _KIND_RANK[entry["kind"]]:
            entry["kind"] = site["kind"]
        entry["anchors"].add(site["anchor"])
    sites = list(merged.values())
    ordered = sorted(evidence, key=lambda row: (row[0], -1 if row[1] is None else row[1], row[2]))
    for path, line, source in ordered:  # trace first: it creates dynamic sites
        if source != "trace" or line is None:
            continue
        site = _innermost([site for site in sites if site["kind"] != "dynamic"], path, line)
        if site is None:
            site = next((item for item in sites if item["kind"] == "dynamic" and item["path"] == path
                         and item["lineno"] == line), None)
            if site is None:
                site = {"path": path, "lineno": line, "end_lineno": line, "kind": "dynamic",
                        "witnesses": set(), "anchors": {line}}
                sites.append(site)
                raw.append({"path": path, "lineno": line, "end_lineno": line, "kind": "dynamic",
                            "via": None, "anchor": line})
                notes["dynamic_sites"] += 1
        site["witnesses"].add("trace")
    for path, line, source in ordered:
        if source not in ("L", "J"):
            continue
        if line is None:
            notes[source + "_without_line"] += 1
            continue
        site = _innermost(sites, path, line)
        if site is None:
            notes[source + "_unplaced"] += 1
            continue
        site["witnesses"].add(source)
    sites.sort(key=lambda site: (site["path"], site["lineno"], site["end_lineno"], _KIND_RANK[site["kind"]]))
    public = [{"path": site["path"], "lineno": site["lineno"], "end_lineno": site["end_lineno"],
               "kind": site["kind"], "witnesses": [name for name in WITNESSES if name in site["witnesses"]]}
              for site in sites]
    detailed = [{**row, "anchors": sorted(site["anchors"])} for row, site in zip(public, sites)]
    return public, detailed, raw, notes


# ---------------------------------------------------------------------- truth table
class TruthTable:
    """Per-callee caller evidence with provenance, call-site lines and D2 sites."""

    def __init__(self, scope):
        self.scope = scope
        self._evidence = {}
        self._aliases, self._sites, self._regions = {}, {}, {}
        self.notes = Counter()

    def add_records(self, source, records):
        source = canonical_source(source)
        kept = 0
        for record in records:
            kept += self.add(source, record)
        return kept

    def add(self, source, record):
        source = canonical_source(source)
        record = validate_call_site(record)
        self.notes[source + ":records"] += 1
        callee = record["callee"]
        if callee not in self.scope.table():
            in_scope = self.scope.contains(split_identity(callee)[0])
            self.notes[source + (":callee_unresolved" if in_scope else ":callee_out_of_scope")] += 1
            return False
        caller, note = self._normalize_caller(record)
        if note:
            self.notes[source + ":" + note] += 1
        if caller is None:
            return False
        entry = self._evidence.setdefault(callee, {}).setdefault(caller, {"sources": set(), "lines": set()})
        entry["sources"].add(source)
        entry["lines"].add((record["path"] or split_identity(caller)[0], record["line"], source))
        self._sites.pop((callee, caller), None)
        self.notes[source + ":kept"] += 1
        return True

    def _normalize_caller(self, record):
        caller, line = record["caller"], record["line"]
        relpath, qualname = split_identity(caller)
        path = record["path"] or relpath
        if not self.scope.contains(relpath):
            return None, "caller_out_of_scope"
        if path != relpath:
            return None, "caller_path_mismatch"
        if caller in self.scope.table():
            return caller, None
        if qualname == "<module>" or line is None:
            return None, "caller_module_level" if qualname == "<module>" else "caller_unresolved"
        mapped = self.scope.enclosing(relpath, line)
        if mapped is None:
            return None, "caller_module_or_class_level"
        if mapped in self.scope.table():
            return mapped, "caller_mapped_to_enclosing_function"
        return None, "caller_unresolved"

    # ----------------------------------------------------------------- queries
    def callees(self):
        return sorted(self._evidence)

    def evidence(self, callee):
        return self._evidence.get(callee, {})

    def callers_by(self, callee, source):
        source = canonical_source(source)
        return sorted(caller for caller, entry in self.evidence(callee).items() if source in entry["sources"])

    def verified(self, callee):
        return sorted(caller for caller, entry in self.evidence(callee).items()
                      if caller != callee and ("trace" in entry["sources"] or {"L", "J"} <= entry["sources"]))

    def unverified(self, callee):
        verified = set(self.verified(callee))
        return sorted(caller for caller in self.evidence(callee) if caller not in verified)

    def runtime_witnessed(self, callee):
        """Verified callers with at least one trace site (D2); the same test as ``caller_record``."""
        return [caller for caller in self.verified(callee)
                if "trace" in self.evidence(callee)[caller]["sources"]
                and any("trace" in site["witnesses"] for site in self.sites(callee, caller))]

    def evidence_lines(self, callee, caller):
        entry = self.evidence(callee).get(caller)
        if entry is None:
            return []
        rows = sorted(entry["lines"], key=lambda row: (row[0], -1 if row[1] is None else row[1], row[2]))
        return [{"path": path, "line": line, "source": source} for path, line, source in rows]

    def aliases(self, callee):
        if callee not in self._aliases:
            self._aliases[callee] = import_aliases(self.scope, callee)
        return self._aliases[callee]

    def site_details(self, callee, caller):
        """(sites, detailed sites with anchors, raw sites, notes) for one caller of ``callee``."""
        key = (callee, caller)
        if key not in self._sites:
            entry = self.evidence(callee).get(caller)
            lines = () if entry is None else entry["lines"]
            self._sites[key] = caller_sites(self.scope, caller, callee, lines, self.aliases(callee), self._regions)
        return self._sites[key]

    def sites(self, callee, caller):
        return self.site_details(callee, caller)[0]

    def stratum(self, callee, caller):
        return stratum_of(self.site_details(callee, caller)[2])

    def caller_record(self, callee, caller, *, verified):
        entry = self.evidence(callee)[caller]
        sites = self.sites(callee, caller)
        record = {"identity": caller, "sources": [source for source in SOURCES if source in entry["sources"]]}
        if verified:
            record["runtime_witnessed"] = any("trace" in site["witnesses"] for site in sites)
            record["stratum"] = self.stratum(callee, caller)
        else:
            record["self"] = caller == callee
        record["sites"] = sites
        return record

    def summary(self, callee):
        verified = self.verified(callee)
        return {"V": [self.caller_record(callee, caller, verified=True) for caller in verified],
                "U": [self.caller_record(callee, caller, verified=False) for caller in self.unverified(callee)]}

    def site_notes(self, callee, callers):
        notes = Counter()
        for caller in callers:
            notes.update(self.site_details(callee, caller)[3])
        return notes


# ---------------------------------------------------------------------- lexical
def lexical_names(scope, target, aliases=None):
    """X's qualified and simple names plus its AST-resolved frozen import aliases (D2, D10.11).

    The live tool's lexical ``import ... as NAME`` rule is deliberately not used here: an
    ``as NAME`` on an import line that merely mentions X's simple name may alias another
    object (validity review, item 72121b3fb9ee6ee03af1).
    """
    qualified = split_identity(target)[1]
    simple = qualified.rsplit(".", 1)[-1]
    names = {qualified, simple}
    if aliases is None:
        aliases = import_aliases(scope, target)
    for values in aliases.values():
        names.update(values)
    return sorted(names)


def lexical_lines(scope, names):
    """Every package-scope (path, line) whose text names one of ``names`` as a whole word."""
    patterns = [re.compile(r"(?<!\w)" + re.escape(name) + r"(?!\w)") for name in names]
    found = []
    for relpath in scope.files():
        for number, line in enumerate(scope.lines(relpath), 1):
            if any(pattern.search(line) for pattern in patterns):
                found.append((relpath, number))
    return found


def lexical_call_sites(scope, target, names=None):
    """Lexical lines mapped to their enclosing function (U candidates); module/class lines give none."""
    names = lexical_names(scope, target) if names is None else names
    records = []
    for relpath, number in lexical_lines(scope, names):
        caller = scope.enclosing(relpath, number)
        if caller is not None:
            records.append({"caller": caller, "callee": target, "path": relpath, "line": number})
    return records


def build_truth(scope, *, trace=(), locagent=(), jedi=(), lexical=()):
    table = TruthTable(scope)
    for source, records in (("trace", trace), ("L", locagent), ("J", jedi), ("lexical", lexical)):
        table.add_records(source, records)
    return table


def locagent_indexed_identities(document, scope, path_prefix=""):
    """Package function identities that have a node in a LocAgent GraphFacts document.

    Same node-to-identity mapping as static_callers.locagent_records: a ``function`` node
    maps to the definition named like the node on the node's ``start_line``
    (``FunctionIndex.functions_at``). ``path_prefix`` is the static manifest's
    ``import_root``.
    """
    indexed = set()
    for node in document.get("nodes", []):
        if node.get("kind") != "function" or not node.get("path") or not node.get("start_line"):
            continue
        path = path_prefix + "/" + node["path"] if path_prefix else node["path"]
        if not scope.contains(path):
            continue
        name = node["id"].split("::", 1)[1].rsplit(".", 1)[-1]
        matches = scope.index.functions_at(path, node["start_line"], name)
        if matches:
            indexed.add(matches[-1]["identity"])
    return indexed
