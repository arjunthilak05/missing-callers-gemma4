"""Exp A static caller backends (docs/EXPA_DESIGN.md section 4).

Backend L (``locagent``): the pinned LocAgent graph built by ``graphs.GraphBackend``
(unchanged) over a staged copy of the package-scope source, rooted at the package import
root so absolute imports resolve. Every ``invokes`` edge whose endpoints map to
package-scope functions is exported. LocAgent edges carry no line, so call-site lines are
located by scanning the caller's AST the way LocAgent collects invocation names; each
record says how (``line_source``). LocAgent has no node for ``Class.__init__``: it
attributes calls made in ``__init__`` to the class node, and those sites are mapped to the
function that encloses them (``caller_mapping = "class_node_init"``). Edges to class nodes
(constructor calls) are not function edges and are dropped with a count.

Backend J (``jedi``): the pinned jedi ``Script.get_references(scope="project")`` on every
package-scope function definition name, over the same staged source. Calls, decorator
applications and value-passing references (arguments, assignments, returns, containers)
are kept and mapped with ``FunctionIndex.enclosing_function``. jedi's project-search caps
(30 parsed / 2,000 opened files per search, which depend on directory order) are raised
so results are complete and order independent; this is recorded in the manifest.

Neither backend infers an edge from text: L edges come from LocAgent and J edges from
jedi's name resolution. The AST scans only place lines and classify reference sites.
"""
from __future__ import annotations

import ast
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import io
import json
import multiprocessing
import os
from pathlib import Path
import posixpath
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import tokenize
import uuid

from ..data import canonical_json, sha256_file
from ..graphs import ENGINE_PINS, GRAPH_PROTOCOL, GraphBackend
from .identity import (FunctionIndex, file_sha256s, in_package_scope, normalize_package_dirs,
                       scan_package_files, source_tree_sha256)

__all__ = [
    "BACKENDS", "STATIC_SCHEMA", "VALUE_CONTEXTS", "agreement_summary", "definition_name_position",
    "jedi_records", "load_edges", "locagent_records", "package_import_root", "reference_sites",
    "run_jedi", "run_locagent", "run_static_backend", "stage_package_source",
]

STATIC_SCHEMA = "absent-edges-expA-static-v1"
BACKENDS = ("locagent", "jedi")
CORE2_ROOT = Path("/mnt/kldata/absent-edges-20261018-core2")
DEFAULT_LOCAGENT_ROOT = CORE2_ROOT / "vendor" / "locagent"
DEFAULT_GRAPH_PYTHON = CORE2_ROOT / "envs" / "graph" / "bin" / "python"
DEFAULT_TIMEOUTS = {"locagent": 900, "jedi": 300}   # engine run / per jedi definition (s)
LOCAGENT_BUILD_OPTIONS = {"fuzzy_search": True, "global_import": False}   # graphs._locagent
JEDI_FILE_LIMIT = 10 ** 9
# Union over attempts (see _jedi_worker_task): every definition gets DEFAULT_JEDI_ATTEMPTS;
# a definition whose answers differ keeps being queried up to DEFAULT_JEDI_MAX_ATTEMPTS.
DEFAULT_JEDI_ATTEMPTS = 5
DEFAULT_JEDI_MAX_ATTEMPTS = 40
KEPT_KINDS = ("call", "decorator", "reference")
# A reference counts as a callable reference when the function object is passed as a value.
VALUE_CONTEXTS = frozenset({
    "argument", "keyword_argument", "starred", "assignment_value", "return_value", "yield_value",
    "container_element", "dict_value", "lambda_body", "conditional_value", "default_value",
})
_FAMILY_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_DEF_PREFIX = re.compile(r"(?:async\s+)?def\s+")
_CODE_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------- staging

def package_import_root(package_dirs) -> str:
    """Directory that must be on sys.path to import the package (``src`` for src layouts)."""
    roots = {posixpath.dirname(directory) if directory else ""
             for directory in normalize_package_dirs(package_dirs)}
    return roots.pop() if len(roots) == 1 else ""


def stage_package_source(repo_root, package_dirs, destination) -> dict:
    """Copy the in-scope ``.py`` files, keeping repository-relative paths."""
    repo_root, destination = Path(repo_root), Path(destination)
    scan = scan_package_files(repo_root, package_dirs)
    if not scan["files"]:
        raise ValueError("no package-scope Python files under " + ", ".join(scan["package_dirs"]))
    hashes, total = {}, 0
    for relpath in scan["files"]:
        raw = (repo_root / relpath).read_bytes()
        target = destination / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        hashes[relpath] = hashlib.sha256(raw).hexdigest()
        total += len(raw)
    roots = {posixpath.dirname(directory) if directory else "" for directory in scan["package_dirs"]}
    return {**scan, "file_sha256": hashes, "source_tree_sha256": source_tree_sha256(hashes),
            "python_bytes": total, "import_root": package_import_root(scan["package_dirs"]),
            "mixed_import_roots": len(roots) > 1}


# ---------------------------------------------------------------- backend L (LocAgent)

def _locagent_function_region(def_node) -> list:
    """Nodes LocAgent's ``traverse_call`` visits for a function caller.

    LocAgent parses the function's source segment, which starts at the ``def`` keyword,
    so decorators are outside it. Nested def/class statements are skipped; lambdas and
    comprehensions are included (their calls belong to the enclosing function).
    """
    decorators = {id(item) for item in def_node.decorator_list}
    stack = [child for child in ast.iter_child_nodes(def_node) if id(child) not in decorators]
    region = []
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        region.append(node)
        stack.extend(ast.iter_child_nodes(node))
    return region


def _call_sites(nodes) -> list:
    """(line, invoked name) for Name/Attribute calls; an attribute call reports the line
    of its attribute name, the line CPython 3.11+ reports for the call."""
    sites = []
    for node in nodes:
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                sites.append((func.lineno, func.id))
            elif isinstance(func, ast.Attribute):
                sites.append((func.end_lineno, func.attr))
    return sites


def _name_sites(nodes) -> list:
    sites = []
    for node in nodes:
        if isinstance(node, ast.Name):
            sites.append((node.lineno, node.id))
        elif isinstance(node, ast.Attribute):
            sites.append((node.end_lineno, node.attr))
    return sites


def _decorator_name_sites(decorators) -> list:
    """LocAgent ``process_decorator_node``: a bare Name, else every Call-with-Name function
    and every Attribute name inside the decorator expression."""
    sites = []
    for decorator in decorators:
        if isinstance(decorator, ast.Name):
            sites.append((decorator.lineno, decorator.id))
            continue
        for node in ast.walk(decorator):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                sites.append((node.func.lineno, node.func.id))
            elif isinstance(node, ast.Attribute):
                sites.append((node.end_lineno, node.attr))
    return sites


def _import_aliases(tree) -> dict:
    """{imported name: {asname, ...}} from every ``from ... import name as asname``."""
    aliases = defaultdict(set)
    if tree is not None:
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.asname and alias.name != "*":
                        aliases[alias.name].add(alias.asname)
    return aliases


def _edge_lines(calls, names, simple, own_aliases, package_aliases) -> list:
    """Locate the call-site line(s) of one LocAgent edge, most specific evidence first."""
    lines = sorted({line for line, name in calls if name == simple})
    if lines:
        return [(line, "ast_call_name") for line in lines]
    for aliases, source in ((own_aliases.get(simple, set()), "ast_call_alias"),
                            (package_aliases.get(simple, set()), "ast_call_package_alias")):
        lines = sorted({line for line, name in calls if name in aliases})
        if lines:
            return [(line, source) for line in lines]
    wanted = {simple} | own_aliases.get(simple, set())
    lines = sorted({line for line, name in names if name in wanted})
    if lines:
        return [(line, "ast_reference") for line in lines]
    return [(None, "not_found")]


def locagent_records(document, index, package_dirs, *, path_prefix="") -> tuple:
    """Map one LocAgent GraphFacts document to call-site records and a report.

    ``index`` is a FunctionIndex over the staged (or identical) source with repository-
    relative paths; ``path_prefix`` is the engine root relative to the repository root.
    """
    nodes = {node["id"]: node for node in document.get("nodes", [])}
    edges = sorted(document.get("edges", []), key=lambda edge: edge.get("id", ""))
    report = {
        "engine_error": document.get("error"), "engine_nodes": len(nodes),
        "engine_nodes_by_kind": dict(sorted(Counter(str(node.get("kind")) for node in nodes.values()).items())),
        "engine_edges_by_relation": dict(sorted(Counter(edge["relation"] for edge in edges).items())),
        "function_nodes_in_scope": 0, "function_nodes_mapped": 0, "function_nodes_unmapped": [],
        "qualname_mismatches": [], "invokes_edges": 0, "invokes_edges_exported": 0,
        "dropped_edges": Counter(), "dropped_sites": Counter(), "line_source": Counter(),
        "caller_mapping": Counter(),
    }
    records = {}
    if document.get("error"):
        return [], _finish_report(report)

    def repo_path(node):
        path = node.get("path")
        if not path:
            return None
        return path_prefix + "/" + path if path_prefix else path

    def engine_name(node_id):
        return node_id.split("::", 1)[1].rsplit(".", 1)[-1]

    functions = {}
    for node_id, node in sorted(nodes.items()):
        if node.get("kind") != "function":
            continue
        path = repo_path(node)
        if path is None or not in_package_scope(path, package_dirs):
            continue
        report["function_nodes_in_scope"] += 1
        matches = (index.functions_at(path, node["start_line"], engine_name(node_id))
                   if node.get("start_line") else [])
        if not matches:
            report["function_nodes_unmapped"].append(node_id)
            continue
        entry = matches[-1]
        functions[node_id] = (path, entry)
        report["function_nodes_mapped"] += 1
        if entry["qualname"].replace(".<locals>", "") != node_id.split("::", 1)[1]:
            report["qualname_mismatches"].append([node_id, entry["identity"]])

    region_cache, alias_cache, package_alias_cache = {}, {}, []

    def caller_sites(node_id, node):
        if node_id in region_cache:
            return region_cache[node_id]
        path = repo_path(node)
        sites = None
        if node.get("kind") == "function":
            _, entry = functions[node_id]
            definition = index.definition_node(path, entry["lineno"], entry["name"])
            if definition is not None:
                region = _locagent_function_region(definition)
                sites = (_call_sites(region), _name_sites(region))
        else:   # class node: LocAgent analyze_init over the first direct ``def __init__``
            definition = index.definition_node(path, node.get("start_line"), engine_name(node_id), kind="class")
            if definition is not None:
                init = next((item for item in definition.body
                             if isinstance(item, ast.FunctionDef) and item.name == "__init__"), None)
                if init is None:
                    sites = ([], [])
                else:
                    region = list(ast.walk(init))
                    sites = (_call_sites(region) + _decorator_name_sites(init.decorator_list),
                             _name_sites(region))
        region_cache[node_id] = sites
        return sites

    def aliases(path):
        if path not in alias_cache:
            alias_cache[path] = _import_aliases(index.tree(path))
        return alias_cache[path]

    def package_aliases():
        if not package_alias_cache:
            merged = defaultdict(set)
            for path in sorted({path for path, _ in functions.values()}):
                for name, values in aliases(path).items():
                    merged[name] |= values
            package_alias_cache.append(merged)
        return package_alias_cache[0]

    def add(record):
        key = (record["caller"], record["callee"], record["path"], record["line"], record["line_source"])
        if key not in records:
            records[key] = record
            report["line_source"][record["line_source"]] += 1
            report["caller_mapping"][record["caller_mapping"]] += 1

    for edge in edges:
        if edge.get("relation") != "invokes":
            continue
        report["invokes_edges"] += 1
        source, target = nodes.get(edge["source"]), nodes.get(edge["target"])
        problem = _endpoint_problem(target, edge["target"], functions, repo_path, package_dirs, "callee")
        if problem is None:
            if source is None:
                problem = "caller_missing_node"
            elif source.get("kind") not in ("function", "class"):
                problem = "caller_kind_" + str(source.get("kind"))
            elif repo_path(source) is None or not in_package_scope(repo_path(source), package_dirs):
                problem = "caller_out_of_scope"
            elif source.get("kind") == "function" and edge["source"] not in functions:
                problem = "caller_unmapped"
        if problem is not None:
            report["dropped_edges"][problem] += 1
            continue
        path = repo_path(source)
        _, callee_entry = functions[edge["target"]]
        sites = caller_sites(edge["source"], source)
        if sites is None:
            report["dropped_edges"]["caller_definition_not_found"] += 1
            continue
        found = _edge_lines(sites[0], sites[1], callee_entry["name"], aliases(path), package_aliases())
        exported = False
        for line, how in found:
            kind = "call" if how.startswith("ast_call") else ("reference" if how == "ast_reference" else "unknown")
            if source.get("kind") == "function":
                caller, mapping = functions[edge["source"]][1]["identity"], "function_node"
                site_caller = index.enclosing_function(path, line) if line is not None else None
            else:
                caller = index.enclosing_function(path, line) if line is not None else None
                mapping, site_caller = "class_node_init", caller
                if caller is None:
                    report["dropped_sites"]["class_node_" + ("site_not_found" if line is None
                                                             else "site_outside_function")] += 1
                    continue
            add({"backend": "locagent", "caller": caller, "callee": callee_entry["identity"],
                 "path": path, "line": line, "kind": kind, "line_source": how,
                 "caller_mapping": mapping, "site_caller": site_caller,
                 "engine_caller": edge["source"], "engine_callee": edge["target"]})
            exported = True
        if exported:
            report["invokes_edges_exported"] += 1
        else:
            report["dropped_edges"]["class_node_no_function_site"] += 1
    ordered = sorted(records.values(), key=_record_sort_key)
    return ordered, _finish_report(report)


def _endpoint_problem(node, node_id, functions, repo_path, package_dirs, role):
    if node is None:
        return role + "_missing_node"
    if node.get("kind") == "class":
        return role + "_class"
    if node.get("kind") != "function":
        return role + "_kind_" + str(node.get("kind"))
    path = repo_path(node)
    if path is None or not in_package_scope(path, package_dirs):
        return role + "_out_of_scope"
    if node_id not in functions:
        return role + "_unmapped"
    return None


def _finish_report(report) -> dict:
    result = {}
    for key, value in report.items():
        if isinstance(value, Counter):
            result[key] = dict(sorted(value.items()))
        elif isinstance(value, list):
            result[key + "_count"] = len(value)
            result[key] = value[:50]
        else:
            result[key] = value
    return result


def _record_sort_key(record):
    line = record.get("line")
    return (record["caller"], record["callee"], record["path"], -1 if line is None else line,
            record.get("column") or 0, record.get("line_source", ""), record.get("kind", ""))


def run_locagent(repo_root, package_dirs, work_dir, *, vendor_root=DEFAULT_LOCAGENT_ROOT,
                 graph_python=DEFAULT_GRAPH_PYTHON, timeout=DEFAULT_TIMEOUTS["locagent"]) -> dict:
    """Build the pinned LocAgent graph for the package scope and export its edges."""
    work_dir = Path(work_dir)
    stage = work_dir / "stage"
    staged = stage_package_source(repo_root, package_dirs, stage)
    engine_root = stage / staged["import_root"] if staged["import_root"] else stage
    backend = GraphBackend("locagent", vendor_root, graph_python, work_dir / "graph-cache", timeout=timeout)
    started = time.monotonic()
    facts = backend.build(engine_root)
    engine_seconds = time.monotonic() - started
    document = facts.data
    index = FunctionIndex(stage)
    started = time.monotonic()
    records, report = locagent_records(document, index, staged["package_dirs"], path_prefix=staged["import_root"])
    report["engine_seconds"] = round(engine_seconds, 3)
    report["mapping_seconds"] = round(time.monotonic() - started, 3)
    report["engine_source_sha256"] = document.get("source_sha256")
    report["engine_module_sha256"] = document.get("engine_module_sha256")
    report["engine_parse_failures"] = document.get("coverage", {}).get("parse_failures", [])
    if document.get("error"):
        logs = sorted((work_dir / "graph-cache").glob("scan-*/engine.log"))
        report["engine_log_tail"] = (logs[-1].read_text(errors="replace")[-4000:] if logs else None)
    return {"records": records, "report": report, "staged": staged, "index": index,
            "document": facts.encoded, "error": document.get("error")}


# ---------------------------------------------------------------- backend J (jedi)

def definition_name_position(index, relpath, entry):
    """(line, code-point column) of a definition's name token, or None."""
    line = entry["lineno"]
    text = index.line_text(relpath, line)
    column = index.char_column(relpath, line, entry["col_offset"])
    match = _DEF_PREFIX.match(text, column)
    if match and text.startswith(entry["name"], match.end()):
        return line, match.end()
    # The name is not on the ``def`` line (explicit line continuation): tokenize ahead.
    snippet = "\n".join(index.line_text(relpath, number) for number in range(line, line + 8)) + "\n"
    seen_def = False
    try:
        for token in tokenize.generate_tokens(io.StringIO(snippet).readline):
            if token.type != tokenize.NAME or (token.start[0] == 1 and token.start[1] < column):
                continue
            if seen_def:
                return (line + token.start[0] - 1, token.start[1]) if token.string == entry["name"] else None
            seen_def = token.string == "def"
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    return None


def _value_context(node, parent) -> str:
    if isinstance(parent, ast.Call):
        return "argument"
    if isinstance(parent, ast.keyword):
        return "keyword_argument"
    if isinstance(parent, ast.Starred):
        return "starred"
    if isinstance(parent, ast.AnnAssign) and parent.annotation is node:
        return "annotation"
    if isinstance(parent, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
        return "assignment_value"
    if isinstance(parent, ast.Return):
        return "return_value"
    if isinstance(parent, (ast.Yield, ast.YieldFrom)):
        return "yield_value"
    if isinstance(parent, (ast.List, ast.Tuple, ast.Set)):
        return "container_element"
    if isinstance(parent, ast.Dict):
        return "dict_key" if any(key is node for key in parent.keys) else "dict_value"
    if isinstance(parent, ast.Lambda):
        return "lambda_body"
    if isinstance(parent, ast.IfExp):
        return "condition" if parent.test is node else "conditional_value"
    if isinstance(parent, ast.BoolOp):
        return "conditional_value"
    if isinstance(parent, ast.arguments):
        return "default_value"
    if isinstance(parent, ast.Compare):
        return "comparison"
    if isinstance(parent, ast.Subscript):
        return "subscript_value" if parent.value is node else "subscript_index"
    if isinstance(parent, ast.FormattedValue):
        return "fstring"
    if isinstance(parent, ast.Expr):
        return "expression_statement"
    if isinstance(parent, ast.arg) or (isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef))
                                       and parent.returns is node):
        return "annotation"
    if isinstance(parent, ast.ClassDef):
        return "class_header"
    return "other:" + type(parent).__name__


def _classify_site(node, parent, is_decorator) -> tuple:
    context = getattr(node, "ctx", None)
    if isinstance(context, ast.Store):
        return "binding", "store"
    if isinstance(context, ast.Del):
        return "binding", "del"
    if is_decorator:
        return "decorator", "decorator"
    if isinstance(parent, ast.Call) and parent.func is node:
        return "call", "call"
    if isinstance(parent, ast.Attribute) and parent.value is node:
        return "attribute_base", "attribute_base"
    return "reference", _value_context(node, parent)


def reference_sites(index, relpath) -> dict:
    """{(line, code-point column of the name token): (kind, context)} for one file.

    Kinds: call, decorator, reference (with a context), attribute_base, binding, import.
    """
    tree = index.tree(relpath)
    if tree is None:
        return {}
    parents, decorators = {}, set()
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            decorators.update(id(item) for item in node.decorator_list)
    sites = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            line, byte_column = node.lineno, node.col_offset
        elif isinstance(node, ast.Attribute):
            line = node.end_lineno
            byte_column = node.end_col_offset - len(node.attr.encode("utf-8"))
        elif isinstance(node, ast.alias) and getattr(node, "lineno", None):
            sites[(node.lineno, index.char_column(relpath, node.lineno, node.col_offset))] = ("import", "import")
            continue
        else:
            continue
        key = (line, index.char_column(relpath, line, byte_column))
        sites[key] = _classify_site(node, parents.get(id(node)), id(node) in decorators)
    return sites


class _DefinitionTimeout(BaseException):
    """BaseException so jedi's internal ``except Exception`` handlers cannot swallow it."""


_JEDI_WORKER = {}


def _raise_timeout(signum, frame):
    raise _DefinitionTimeout()


def _jedi_worker_init(stage_root, sys_path, cache_root, timeout, file_limit, memory_limit, attempts=1,
                      max_attempts=1):
    if memory_limit:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
    import jedi
    from jedi.inference import references
    references._OPENED_FILE_LIMIT = file_limit
    references._PARSED_FILE_LIMIT = file_limit
    cache = Path(cache_root) / ("worker-" + str(os.getpid()))
    cache.mkdir(parents=True, exist_ok=True)
    jedi.settings.cache_directory = str(cache)
    _JEDI_WORKER.update(
        jedi=jedi, stage=Path(stage_root), timeout=int(timeout), attempts=max(1, int(attempts)),
        max_attempts=max(1, int(attempts), int(max_attempts)), environment=jedi.InterpreterEnvironment(),
        project=jedi.Project(stage_root, sys_path=list(sys_path), smart_sys_path=True,
                             load_unsafe_extensions=False))
    signal.signal(signal.SIGALRM, _raise_timeout)


def _jedi_worker_task(task) -> dict:
    """Query one definition ``attempts`` times (fresh Script each time) and union the answers.

    jedi's project-wide search is not deterministic for every reference (measured on the dev
    families: 3 of 545 boltons records varied across single-attempt runs), so each reference
    carries its support, the number of attempts that returned it. A definition whose answers
    differ is queried again, up to ``max_attempts``, so rarely returned references still land
    in the union.
    """
    state = _JEDI_WORKER
    started = time.monotonic()
    support, results, error = Counter(), set(), None
    attempts_done = 0
    for _ in range(state["max_attempts"]):
        if attempts_done >= state["attempts"] and len(results) <= 1:
            break
        try:
            signal.alarm(state["timeout"])
            try:
                script = state["jedi"].Script(path=str(state["stage"] / task["relpath"]),
                                              project=state["project"], environment=state["environment"])
                names = script.get_references(task["line"], task["column"], scope="project",
                                              include_builtins=False)
                found = frozenset((str(name.module_path) if name.module_path else None, name.line,
                                   name.column, bool(name.is_definition()), name.name) for name in names)
            finally:
                signal.alarm(0)
        except _DefinitionTimeout:   # also catches an alarm that fires inside the finally
            error = "timeout"
        except Exception as exception:   # one jedi failure must not stop the family
            error = f"{type(exception).__name__}: {str(exception)[:300]}"
        if error is not None:
            break
        attempts_done += 1
        support.update(found)
        results.add(found)
    references = [[*reference, count] for reference, count in
                  sorted(support.items(), key=lambda item: tuple(str(part) for part in item[0]))]
    # references: [module_path, line, column, is_definition, name, support]
    return {"key": task["key"], "references": references, "error": error, "attempts": attempts_done,
            "distinct_results": len(results), "seconds": round(time.monotonic() - started, 3)}


def _jedi_sys_path(stage, import_root) -> list:
    """Project import roots first, then this interpreter's stdlib/site paths (minus this code)."""
    head = [str(stage / import_root)] if import_root else []
    head.append(str(stage))
    tail = []
    for entry in sys.path:
        if not entry or not os.path.isdir(entry):
            continue
        resolved = Path(entry).resolve()
        if resolved == _CODE_ROOT or _CODE_ROOT in resolved.parents:
            continue
        tail.append(str(resolved))
    return list(dict.fromkeys(head + tail))


def jedi_records(results, tasks, index, stage_root, package_dirs) -> tuple:
    """Classify jedi references and map them to caller identities."""
    stage_root = Path(stage_root).resolve()
    report = {"definitions_queried": len(tasks), "definitions_failed": [], "timeouts": 0,
              "raw_references": 0, "dropped_references": Counter(), "kinds": Counter(),
              "contexts": Counter(), "unclassified_examples": [], "jedi_seconds_total": 0.0,
              "slowest_definitions": [], "attempts_histogram": Counter(),
              "unstable_definitions": [], "unstable_records": []}
    sites_cache, records, timings = {}, {}, []
    for task in tasks:
        result = results.get(task["key"]) or {"references": [], "error": "missing_result", "seconds": None}
        attempts = result.get("attempts", 1)
        report["attempts_histogram"][str(attempts)] += 1
        if result.get("distinct_results", 1) > 1:
            report["unstable_definitions"].append(
                {"identity": task["identity"], "distinct_results": result["distinct_results"], "attempts": attempts})
        if result.get("seconds") is not None:
            report["jedi_seconds_total"] += result["seconds"]
            timings.append((result["seconds"], task["identity"]))
        if result["error"]:
            report["definitions_failed"].append({"identity": task["identity"], "error": result["error"]})
            report["timeouts"] += result["error"] == "timeout"
        for module_path, line, column, is_definition, name, support in result["references"]:
            report["raw_references"] += 1
            if module_path is None:
                report["dropped_references"]["no_module_path"] += 1
                continue
            try:
                relpath = Path(module_path).resolve().relative_to(stage_root).as_posix()
            except ValueError:
                report["dropped_references"]["outside_project"] += 1
                continue
            if not in_package_scope(relpath, package_dirs):
                report["dropped_references"]["out_of_scope"] += 1
                continue
            if is_definition:
                report["dropped_references"]["definition"] += 1
                continue
            if name != task["name"]:
                report["dropped_references"]["name_mismatch"] += 1
                continue
            if relpath not in sites_cache:
                sites_cache[relpath] = reference_sites(index, relpath)
            kind, context = sites_cache[relpath].get((line, column), ("unclassified", "unclassified"))
            if kind == "unclassified" and len(report["unclassified_examples"]) < 20:
                report["unclassified_examples"].append(
                    {"callee": task["identity"], "path": relpath, "line": line, "column": column,
                     "text": index.line_text(relpath, line).strip()[:200]})
            if kind not in KEPT_KINDS:
                report["dropped_references"][kind] += 1
                continue
            if kind == "reference" and context not in VALUE_CONTEXTS:
                report["dropped_references"]["reference:" + context] += 1
                continue
            caller = index.enclosing_function(relpath, line)
            if caller is None:
                report["dropped_references"]["no_enclosing_function"] += 1
                continue
            key = (caller, task["identity"], relpath, line, column)
            if key not in records:
                # Support counts vary run to run, so they stay out of the (byte-stable) edge file.
                records[key] = {"backend": "jedi", "caller": caller, "callee": task["identity"],
                                "path": relpath, "line": line, "column": column, "kind": kind,
                                "context": context, "line_source": "jedi_reference"}
                report["kinds"][kind] += 1
                report["contexts"][context] += 1
                if support < attempts:
                    report["unstable_records"].append(
                        {"caller": caller, "callee": task["identity"], "path": relpath, "line": line,
                         "column": column, "support": support, "attempts": attempts})
    report["jedi_seconds_total"] = round(report["jedi_seconds_total"], 3)
    report["slowest_definitions"] = [[seconds, identity] for seconds, identity in
                                     sorted(timings, key=lambda item: (-item[0], item[1]))[:10]]
    return sorted(records.values(), key=_record_sort_key), _finish_report(report)


def run_jedi(repo_root, package_dirs, work_dir, *, workers=16, timeout=DEFAULT_TIMEOUTS["jedi"],
             attempts=DEFAULT_JEDI_ATTEMPTS, max_attempts=DEFAULT_JEDI_MAX_ATTEMPTS,
             memory_limit=8 * 1024 ** 3, progress=None) -> dict:
    """Run jedi get_references(scope="project") ``attempts`` times for every package-scope
    definition and keep the union (answers below full support are listed in the report)."""
    work_dir = Path(work_dir)
    staged = stage_package_source(repo_root, package_dirs, work_dir / "stage")
    stage = (work_dir / "stage").resolve()
    index = FunctionIndex(stage)
    tasks, position_errors = [], []
    for relpath in staged["files"]:
        for entry in index.functions(relpath):
            position = definition_name_position(index, relpath, entry)
            if position is None:
                position_errors.append({"identity": entry["identity"], "error": "name_position_not_found"})
                continue
            tasks.append({"key": len(tasks), "relpath": relpath, "identity": entry["identity"],
                          "name": entry["name"], "line": position[0], "column": position[1]})
    sys_path = _jedi_sys_path(stage, staged["import_root"])
    results = {}
    started = time.monotonic()
    if tasks:
        context = multiprocessing.get_context("spawn")
        initargs = (str(stage), sys_path, str(work_dir / "jedi-cache"), timeout, JEDI_FILE_LIMIT,
                    memory_limit, attempts, max_attempts)
        with ProcessPoolExecutor(max_workers=max(1, int(workers)), mp_context=context,
                                 initializer=_jedi_worker_init, initargs=initargs) as pool:
            futures = {pool.submit(_jedi_worker_task, task): task["key"] for task in tasks}
            for done, future in enumerate(as_completed(futures), 1):
                key = futures[future]
                try:
                    results[key] = future.result()
                except (Exception, _DefinitionTimeout) as error:   # keep the failure, never invent edges
                    results[key] = {"key": key, "references": [], "seconds": None,
                                    "error": f"worker_failure: {type(error).__name__}: {error}"}
                if progress is not None and (done % 200 == 0 or done == len(tasks)):
                    progress(done, len(tasks), time.monotonic() - started)
    pool_seconds = time.monotonic() - started
    records, report = jedi_records(results, tasks, index, stage, staged["package_dirs"])
    report["definition_position_errors"] = position_errors
    report["pool_seconds"] = round(pool_seconds, 3)
    report["sys_path"] = sys_path
    report["workers"] = int(workers)
    report["attempts_per_definition"] = int(attempts)
    report["max_attempts_unstable"] = max(int(attempts), int(max_attempts))
    return {"records": records, "report": report, "staged": staged, "index": index, "error": None}


# ---------------------------------------------------------------- outputs and orchestration

def _atomic_write(path: Path, data: bytes):
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    with open(temporary, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_edges(path) -> list:
    with open(path, "rb") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _git_state(repo_root: Path) -> dict:
    git = ["git", "-c", "safe.directory=" + str(repo_root), "-C", str(repo_root)]
    try:
        head = subprocess.run([*git, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=60)
        status = subprocess.run([*git, "status", "--porcelain", "--untracked-files=no"],
                                capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "tracked_changes": None}
    if head.returncode:
        return {"commit": None, "tracked_changes": None}
    return {"commit": head.stdout.strip(), "tracked_changes": bool(status.stdout.strip()) if not status.returncode else None}


def _graph_python_versions(graph_python) -> dict:
    try:
        output = subprocess.run([str(graph_python), "-I", "-c",
                                 "import sys, networkx; print(sys.version.split()[0], networkx.__version__)"],
                                capture_output=True, text=True, timeout=120)
        python_version, networkx_version = output.stdout.split()
        return {"graph_python_version": python_version, "networkx_version": networkx_version}
    except (OSError, ValueError, subprocess.SubprocessError):
        return {"graph_python_version": None, "networkx_version": None}


def _backend_pin(backend, *, vendor_root, graph_python, timeout, workers, result, attempts=1,
                 max_attempts=1) -> dict:
    if backend == "locagent":
        return {"engine": "locagent", **ENGINE_PINS["locagent"], "vendor_root": str(vendor_root),
                "engine_module_sha256": (result or {}).get("report", {}).get("engine_module_sha256"),
                "graph_python": str(graph_python), **_graph_python_versions(graph_python),
                "build_graph_options": LOCAGENT_BUILD_OPTIONS, "graph_protocol": GRAPH_PROTOCOL,
                "graphs_py_sha256": sha256_file(_CODE_ROOT / "absent_edges" / "graphs.py"),
                "engine_timeout_s": timeout}
    import jedi
    import parso
    from jedi.inference import references
    return {"engine": "jedi", "jedi_version": jedi.__version__, "parso_version": parso.__version__,
            "python": sys.version.split()[0], "python_executable": sys.executable,
            "environment": "jedi.InterpreterEnvironment", "scope": "project", "include_builtins": False,
            "limit_overrides": {
                "jedi.inference.references._OPENED_FILE_LIMIT": [references._OPENED_FILE_LIMIT, JEDI_FILE_LIMIT],
                "jedi.inference.references._PARSED_FILE_LIMIT": [references._PARSED_FILE_LIMIT, JEDI_FILE_LIMIT]},
            "known_limits": [
                "jedi searches names of 2 or fewer characters only in modules that define them (unchanged)",
                "aliased imports (from m import f as g) are references to the import, not to later g() calls",
                "jedi reference search is not deterministic for every reference; each definition is queried "
                "attempts_per_definition times with a fresh Script (definitions whose answers differ: up to "
                "max_attempts_unstable) and the union is kept (report.unstable_definitions / "
                "report.unstable_records list answers below full support)"],
            "attempts_per_definition": attempts, "max_attempts_unstable": max(attempts, max_attempts),
            "per_attempt_timeout_s": timeout, "workers": workers}


def _counts(records, staged, index) -> dict:
    functions = [entry for path in staged["files"] for entry in index.functions(path)]
    pairs = {(record["caller"], record["callee"]) for record in records}
    return {
        "package_files": len(staged["files"]),
        "package_python_bytes": staged["python_bytes"],
        "package_functions": len(functions),
        "package_function_identities": len({entry["identity"] for entry in functions}),
        "package_functions_name_len_le2": sum(len(entry["name"]) <= 2 for entry in functions),
        "records": len(records),
        "records_with_line": sum(record.get("line") is not None for record in records),
        "distinct_pairs": len(pairs),
        "distinct_callers": len({caller for caller, _ in pairs}),
        "distinct_callees": len({callee for _, callee in pairs}),
        "self_pairs": sum(caller == callee for caller, callee in pairs),
    }


def run_static_backend(*, family, repo_root, package_dirs, backend, out_root, work_dir=None,
                       workers=16, timeout=None, vendor_root=DEFAULT_LOCAGENT_ROOT,
                       graph_python=DEFAULT_GRAPH_PYTHON, overwrite=False, keep_work=False,
                       jedi_attempts=DEFAULT_JEDI_ATTEMPTS, jedi_max_attempts=DEFAULT_JEDI_MAX_ATTEMPTS,
                       argv=None, progress=None) -> dict:
    """Run one backend for one family and write ``<out>/<family>/<backend>.*`` atomically."""
    if not isinstance(family, str) or not _FAMILY_NAME.fullmatch(family):
        raise ValueError(f"invalid family name: {family!r}")
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}")
    timeout = DEFAULT_TIMEOUTS[backend] if timeout is None else int(timeout)
    if backend == "jedi":
        import jedi  # noqa: F401  fail before any work when run without the pinned jedi env
    repo_root = Path(repo_root).resolve(strict=True)
    package_dirs = list(normalize_package_dirs(package_dirs))
    family_dir = Path(out_root) / family
    edges_path = family_dir / (backend + ".edges.jsonl")
    manifest_path = family_dir / (backend + ".manifest.json")
    if not overwrite and (edges_path.exists() or manifest_path.exists()):
        raise FileExistsError(f"outputs exist (pass overwrite): {manifest_path}")
    family_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(work_dir) if work_dir else family_dir / (".work-" + backend + "-" + uuid.uuid4().hex)
    work_dir.mkdir(parents=True, exist_ok=False)
    started_at = datetime.now(timezone.utc)
    started = time.monotonic()
    errors, result = [], None
    try:
        if backend == "locagent":
            result = run_locagent(repo_root, package_dirs, work_dir, vendor_root=vendor_root,
                                  graph_python=graph_python, timeout=timeout)
            if result["error"]:
                errors.append({"stage": "engine", "error": result["error"]})
        else:
            result = run_jedi(repo_root, package_dirs, work_dir, workers=workers, timeout=timeout,
                              attempts=jedi_attempts, max_attempts=jedi_max_attempts, progress=progress)
            for failure in result["report"]["definitions_failed"]:
                errors.append({"stage": "jedi_definition", **failure})
            for failure in result["report"]["definition_position_errors"]:
                errors.append({"stage": "jedi_definition_position", **failure})
    except Exception as error:   # record the failure; never write invented edges
        errors.append({"stage": "backend", "error": f"{type(error).__name__}: {error}"})
    wall = time.monotonic() - started
    records = result["records"] if result else []
    staged = result["staged"] if result else None
    # Counting indexes every staged file, so parse failures are known before errors are read.
    counts = _counts(records, staged, result["index"]) if staged else {"records": len(records)}
    if staged is not None:
        for path, message in sorted(result["index"].errors.items()):
            errors.append({"stage": "parse", "path": path, "error": message})
        try:
            if file_sha256s(repo_root, staged["files"]) != staged["file_sha256"]:
                errors.append({"stage": "source", "error": "source_changed_during_run"})
        except OSError as error:
            errors.append({"stage": "source", "error": f"source_unreadable_after_run: {error}"})
    fatal = result is None or (result.get("error") is not None) or any(
        item["stage"] in ("backend", "engine", "source") for item in errors)
    # Failed jedi definitions and unparsable package files are coverage gaps: "partial".
    status = "failed" if fatal else ("partial" if any(
        item["stage"].startswith("jedi") or item["stage"] == "parse" for item in errors) else "complete")
    payload = b"".join(canonical_json(record) + b"\n" for record in records)
    manifest = {
        "schema": STATIC_SCHEMA, "family": family, "backend": backend, "status": status,
        "repo": {"root": str(repo_root), **_git_state(repo_root)},
        "package_dirs": package_dirs,
        "import_root": staged["import_root"] if staged else package_import_root(package_dirs),
        "mixed_import_roots": staged["mixed_import_roots"] if staged else None,
        "source_tree_sha256": staged["source_tree_sha256"] if staged else None,
        "package_file_sha256": staged["file_sha256"] if staged else {},
        "skipped_symlinks": staged["skipped_symlinks"] if staged else [],
        "backend_pin": _backend_pin(backend, vendor_root=vendor_root, graph_python=graph_python,
                                    timeout=timeout, workers=workers, result=result, attempts=jedi_attempts,
                                    max_attempts=jedi_max_attempts),
        "counts": counts,
        "report": result["report"] if result else {},
        "errors": errors,
        "wall_time_s": round(wall, 3),
        "started_at": started_at.isoformat(), "finished_at": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(), "python": sys.version.split()[0], "argv": list(argv or []),
        "code_sha256": {relative: sha256_file(_CODE_ROOT / relative) for relative in (
            "absent_edges/probe/identity.py", "absent_edges/probe/static_callers.py",
            "absent_edges/graphs.py")},
        "outputs": {"edges": {"path": edges_path.name, "sha256": hashlib.sha256(payload).hexdigest(),
                              "records": len(records), "bytes": len(payload)}},
    }
    if backend == "locagent" and result and result.get("document") is not None:
        document_path = family_dir / "locagent.document.json"
        _atomic_write(document_path, result["document"])
        manifest["outputs"]["locagent_document"] = {
            "path": document_path.name, "sha256": hashlib.sha256(result["document"]).hexdigest(),
            "bytes": len(result["document"])}
    _atomic_write(edges_path, payload)
    _atomic_write(manifest_path, (json.dumps(manifest, indent=1, sort_keys=True, ensure_ascii=False)
                                  + "\n").encode("utf-8"))
    if not keep_work and status != "failed":
        shutil.rmtree(work_dir, ignore_errors=True)
    else:
        manifest["work_dir"] = str(work_dir)
    return manifest


def agreement_summary(locagent, jedi) -> dict:
    """Pair- and site-level overlap of two backends' records (lists of record dicts)."""
    def pairs(records, kinds=None):
        return {(record["caller"], record["callee"]) for record in records
                if kinds is None or record.get("kind") in kinds}

    def sites(records):
        return {(record["caller"], record["callee"], record["path"], record["line"])
                for record in records if record.get("line") is not None}

    left, right = pairs(locagent), pairs(jedi)
    both = left & right
    left_sites, right_sites = sites(locagent), sites(jedi)
    return {
        "locagent_pairs": len(left), "jedi_pairs": len(right), "both_pairs": len(both),
        "locagent_only_pairs": len(left - right), "jedi_only_pairs": len(right - left),
        "union_pairs": len(left | right),
        "both_pairs_distinct_callees": len({callee for _, callee in both}),
        "jedi_call_pairs": len(pairs(jedi, {"call"})),
        "both_pairs_jedi_call_kind": len(left & pairs(jedi, {"call"})),
        "locagent_sites": len(left_sites), "jedi_sites": len(right_sites),
        "both_sites": len(left_sites & right_sites),
    }
