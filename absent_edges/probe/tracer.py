"""Exp A lane T: runtime-trace conventions, loaders and observed caller sets.

See docs/EXPA_DESIGN.md sections 2 (identities, call-site schema) and 5 (runtime tracer).

Standard library only. The loaders run on the controller (Python 3.11). The pytest
plugin (``absent_edges.probe.tracer_plugin``) imports the shared conventions from
this module inside repository environments (Python 3.13), so runtime records and
post-hoc normalization use one implementation.

Observed calls are positive evidence only: an unobserved edge is not an impossible
edge, and a trace is never a complete call graph.

Identity helpers come from ``absent_edges.probe.identity`` (lane S) when that module
exists. Until it does, a built-in fallback with the same ``relpath::qualname``
formatting and the section 2 exclusion rules is used; ``IDENTITY_IMPL`` names the
implementation in effect and every manifest records it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import uuid

EDGES_SCHEMA = "absent-edges-expA-trace-edges-v1"
PLUGIN_SCHEMA = "absent-edges-expA-trace-plugin-v1"
MANIFEST_SCHEMA = "absent-edges-expA-trace-manifest-v1"

EDGES_STEM = "edges"
PLUGIN_MANIFEST_STEM = "plugin_manifest"
RUN_MANIFEST_FILE = "manifest.json"
PARTIAL_MARK = ".partial"

# Environment contract between scripts/expA_trace.py and the pytest plugin.
ENV_ROOT = "AE_TRACE_ROOT"
ENV_PACKAGE_DIRS = "AE_TRACE_PACKAGE_DIRS"
ENV_OUT = "AE_TRACE_OUT"
ENV_MODE = "AE_TRACE_MODE"
ENV_DEADLINE = "AE_TRACE_DEADLINE_EPOCH"
ENV_CHECKPOINT = "AE_TRACE_CHECKPOINT_SECONDS"
MODES = ("trace", "count")

CALLER_KINDS = ("function", "module", "class_body")
FUNCTION_CALLERS = ("function",)
MODULE_QUALNAME = "<module>"

SOURCE_TREE_METHOD = ("sha256(canonical_json({relpath: sha256(bytes)})) over the in-scope package files "
                      "(identity.package_files: regular *.py under package_dirs, excluded paths and symlinks "
                      "skipped); identity.source_tree_sha256 convention")


class TraceRecordError(ValueError):
    """A trace record that does not satisfy the section 2 call-site schema."""


# ---------------------------------------------------------------------------
# Identity conventions (section 2). Prefer lane S's module; fall back verbatim.
# ---------------------------------------------------------------------------
_EXCLUDED_DIRS = frozenset({"tests", "test", "testing", "docs", "examples", "benchmarks", "scripts"})
_EXCLUDED_FILES = frozenset({"conftest.py", "setup.py", "noxfile.py"})


def _fallback_function_identity(relpath: str, qualname: str) -> str:
    if not isinstance(relpath, str) or not relpath or "::" in relpath:
        raise ValueError(f"invalid relpath for identity: {relpath!r}")
    if not isinstance(qualname, str) or not qualname or "::" in qualname:
        raise ValueError(f"invalid qualname for identity: {qualname!r}")
    return f"{relpath}::{qualname}"


def _fallback_split_identity(identity: str) -> tuple[str, str]:
    if not isinstance(identity, str):
        raise ValueError("identity must be a string")
    relpath, sep, qualname = identity.partition("::")
    if not sep or not relpath or not qualname or "::" in qualname:
        raise ValueError(f"malformed function identity: {identity!r}")
    return relpath, qualname


def _fallback_is_test_or_excluded_path(relpath: str) -> bool:
    parts = PurePosixPath(relpath).parts
    if not parts:
        return True
    if any(part in _EXCLUDED_DIRS for part in parts[:-1]):
        return True
    name = parts[-1]
    if name in _EXCLUDED_FILES:
        return True
    return name.endswith(".py") and (name.startswith("test_") or name.endswith("_test.py"))


def _fallback_normalize_package_dirs(package_dirs) -> tuple[str, ...]:
    out = set()
    for value in package_dirs:
        if not isinstance(value, str):
            raise ValueError("package dirs must be strings")
        item = value.strip().rstrip("/")
        while item.startswith("./"):
            item = item[2:]
        if item in ("", "."):
            out.add("")  # the whole repository
            continue
        if "\\" in item or item.startswith("/") or any(p in ("", ".", "..") for p in item.split("/")):
            raise ValueError(f"package dir must be a normalized POSIX path relative to the root: {value!r}")
        out.add(item)
    if not out:
        raise ValueError("at least one package dir is required")
    return tuple(sorted(out))


def _fallback_in_package_scope(relpath: str, package_dirs) -> bool:
    if not relpath.endswith(".py") or _fallback_is_test_or_excluded_path(relpath):
        return False
    return any(d == "" or relpath == d or relpath.startswith(d + "/")
               for d in _fallback_normalize_package_dirs(package_dirs))


try:
    from absent_edges.probe import identity as _identity
except ModuleNotFoundError as _missing:  # only the module itself may be absent
    if _missing.name != "absent_edges.probe.identity":
        raise
    _identity = None

if _identity is not None:
    IDENTITY_IMPL = "absent_edges.probe.identity"
    function_identity = _identity.function_identity
    split_identity = _identity.split_identity
    is_test_or_excluded_path = _identity.is_test_or_excluded_path
    in_package_scope = _identity.in_package_scope
    _normalize_dirs = _identity.normalize_package_dirs
else:
    IDENTITY_IMPL = "fallback:absent_edges.probe.tracer"
    function_identity = _fallback_function_identity
    split_identity = _fallback_split_identity
    is_test_or_excluded_path = _fallback_is_test_or_excluded_path
    in_package_scope = _fallback_in_package_scope
    _normalize_dirs = _fallback_normalize_package_dirs


def normalize_package_dirs(package_dirs) -> tuple[str, ...]:
    """Sorted unique root-relative POSIX package dirs; ``""`` means the whole repository.

    A string is split on ``os.pathsep`` (the AE_TRACE_PACKAGE_DIRS format).
    """
    if isinstance(package_dirs, str):
        package_dirs = [part.strip() for part in package_dirs.split(os.pathsep) if part.strip()]
    return tuple(_normalize_dirs(list(package_dirs)))


def package_dirs_env(package_dirs) -> str:
    """AE_TRACE_PACKAGE_DIRS value (``"."`` stands for the whole repository)."""
    return os.pathsep.join(d or "." for d in normalize_package_dirs(package_dirs))


def path_in_scope(relpath: str, package_dirs) -> bool:
    """A root-relative POSIX ``.py`` path inside the package dirs and not a test/excluded path."""
    return (relpath.endswith(".py") and bool(in_package_scope(relpath, package_dirs))
            and not is_test_or_excluded_path(relpath))


def _is_anonymous(segment: str) -> bool:
    return segment.startswith("<") and segment != "<locals>"


def normalize_caller_qualname(qualname: str) -> tuple[str, str, str | None]:
    """Map a runtime function-code qualname to the static caller ``(qualname, kind, scope)``.

    Calls made inside lambdas, generator expressions, comprehensions (pre-3.12) and
    annotation scopes are attributed to the nearest enclosing ``def``: those anonymous
    scopes are not function identities (section 2), and a static ``enclosing_function``
    of the call line reports that ``def``. If the nearest named scope is a class body
    or the module, ``kind`` says so. ``scope`` is the original qualname when it was
    normalized, else None.

    >>> normalize_caller_qualname("f.<locals>.<lambda>")
    ('f', 'function', 'f.<locals>.<lambda>')
    >>> normalize_caller_qualname("A.<lambda>")
    ('A', 'class_body', 'A.<lambda>')
    """
    if not isinstance(qualname, str) or not qualname:
        raise ValueError("qualname must be a non-empty string")
    if qualname == MODULE_QUALNAME:
        return MODULE_QUALNAME, "module", None
    parts = qualname.split(".")
    if not _is_anonymous(parts[-1]):
        return qualname, "function", None
    kind = "function"
    while parts and _is_anonymous(parts[-1]):
        parts.pop()
        if parts and parts[-1] == "<locals>":
            parts.pop()
            kind = "function"
        elif parts and _is_anonymous(parts[-1]):
            # Nested anonymous scope without "<locals>" between, e.g. f.<locals>.<genexpr>.<genexpr>.
            continue
        else:
            kind = "class_body" if parts else "module"
            break
    if not parts:
        return MODULE_QUALNAME, "module", qualname
    return ".".join(parts), kind, qualname


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------
def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _check_relpath(relpath: str, what: str) -> None:
    if not isinstance(relpath, str) or not relpath:
        raise TraceRecordError(f"{what}: empty path")
    if "\\" in relpath or relpath.startswith("/") or (len(relpath) > 1 and relpath[1] == ":"):
        raise TraceRecordError(f"{what}: path must be POSIX and relative: {relpath!r}")
    if any(part in ("", ".", "..") for part in relpath.split("/")):
        raise TraceRecordError(f"{what}: path has empty, '.' or '..' parts: {relpath!r}")


def _split(identity, what: str) -> tuple[str, str]:
    if not isinstance(identity, str) or not identity:
        raise TraceRecordError(f"{what} must be a non-empty identity string")
    try:
        relpath, qualname = split_identity(identity)
    except (ValueError, TypeError) as error:
        raise TraceRecordError(f"{what}: {error}") from None
    _check_relpath(relpath, what)
    return relpath, qualname


def normalize_record(raw) -> dict:
    """Validate one trace record and return it in canonical form.

    Required (section 2): ``caller``, ``callee`` (``relpath::qualname``), ``path`` (the
    call-site file, which is the caller's file) and ``line`` (int >= 1). Optional:
    ``caller_kind`` (function | module | class_body; derived when absent),
    ``caller_scope`` (anonymous scope that was normalized away) and ``first_test``.
    """
    if not isinstance(raw, dict):
        raise TraceRecordError("record must be a JSON object")
    caller_rel, caller_q = _split(raw.get("caller"), "caller")
    callee_rel, callee_q = _split(raw.get("callee"), "callee")
    path = raw.get("path")
    _check_relpath(path, "path")
    if path != caller_rel:
        raise TraceRecordError(f"path {path!r} is not the caller's file {caller_rel!r}")
    line = raw.get("line")
    if type(line) is not int or line < 1:
        raise TraceRecordError(f"line must be an int >= 1, got {line!r}")
    if callee_q == MODULE_QUALNAME or _is_anonymous(callee_q.split(".")[-1]):
        raise TraceRecordError(f"callee must be a named function, got {callee_q!r}")
    kind = raw.get("caller_kind")
    if kind is None:
        kind = "module" if caller_q == MODULE_QUALNAME else "function"
    if kind not in CALLER_KINDS:
        raise TraceRecordError(f"unknown caller_kind {kind!r}")
    if (caller_q == MODULE_QUALNAME) != (kind == "module"):
        raise TraceRecordError("caller_kind 'module' must match the <module> qualname")
    if _is_anonymous(caller_q.split(".")[-1]) and caller_q != MODULE_QUALNAME:
        raise TraceRecordError(f"caller qualname must be normalized, got {caller_q!r}")
    record = {"caller": raw["caller"], "callee": raw["callee"], "path": path, "line": line, "caller_kind": kind}
    scope = raw.get("caller_scope")
    if scope is not None:
        if not isinstance(scope, str) or not scope:
            raise TraceRecordError("caller_scope must be a non-empty string")
        record["caller_scope"] = scope
    first_test = raw.get("first_test")
    if first_test is not None:
        if not isinstance(first_test, str):
            raise TraceRecordError("first_test must be a string")
        record["first_test"] = first_test
    return record


def record_key(record: dict) -> tuple:
    """Deduplication key: the call site plus how the caller was classified."""
    return (record["caller"], record["callee"], record["path"], record["line"],
            record.get("caller_kind", "function"), record.get("caller_scope") or "")


def record_sort_key(record: dict) -> tuple:
    return (record["callee"], record["caller"], record["path"], record["line"],
            record.get("caller_kind", "function"), record.get("caller_scope") or "",
            record.get("first_test") or "")


def call_site(record: dict) -> dict:
    """The section 2 call-site dict (exactly caller, callee, path, line)."""
    return {"caller": record["caller"], "callee": record["callee"], "path": record["path"], "line": record["line"]}


def dedupe_records(records) -> list[dict]:
    """Canonical sorted list; on duplicate keys the smallest ``first_test`` wins."""
    best: dict[tuple, dict] = {}
    for record in sorted(records, key=record_sort_key):
        best.setdefault(record_key(record), record)
    return sorted(best.values(), key=record_sort_key)


# ---------------------------------------------------------------------------
# Atomic writers
# ---------------------------------------------------------------------------
def _fsync_dir(directory: Path) -> None:
    if os.name != "posix":
        return
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_bytes_atomic(path, data: bytes) -> str:
    """tmp + fsync + rename (+ directory fsync). Returns the sha256 of ``data``."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with open(temporary, "xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    _fsync_dir(target.parent)
    return hashlib.sha256(data).hexdigest()


def write_json_atomic(path, value) -> str:
    return write_bytes_atomic(path, (json.dumps(value, sort_keys=True, indent=1, ensure_ascii=False,
                                                allow_nan=False) + "\n").encode("utf-8"))


def write_jsonl_atomic(path, rows) -> str:
    return write_bytes_atomic(path, "".join(canonical_json(row) + "\n" for row in rows).encode("utf-8"))


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TraceEdges:
    records: tuple
    errors: tuple = ()
    duplicates: int = 0
    files: tuple = ()
    file_sha256: dict = field(default_factory=dict)


def load_edges_files(paths, *, strict: bool = True) -> TraceEdges:
    """Load and validate JSONL call-site records from one or more files and merge them."""
    collected, errors, total, names, hashes = [], [], 0, [], {}
    for path in paths:
        path = Path(path)
        names.append(path.name)
        hashes[path.name] = sha256_file(path)
        with open(path, "r", encoding="utf-8") as stream:
            for number, text in enumerate(stream, 1):
                if not text.strip():
                    continue
                total += 1
                try:
                    collected.append(normalize_record(json.loads(text)))
                except (ValueError, TraceRecordError) as error:
                    problem = {"file": path.name, "line": number, "error": str(error)[:500]}
                    if strict:
                        raise TraceRecordError(f"{path.name}:{number}: {problem['error']}") from None
                    errors.append(problem)
    records = dedupe_records(collected)
    return TraceEdges(records=tuple(records), errors=tuple(errors), duplicates=len(collected) - len(records),
                      files=tuple(names), file_sha256=hashes)


def load_edges(path, *, strict: bool = True) -> TraceEdges:
    return load_edges_files([path], strict=strict)


@dataclass(frozen=True)
class TraceRun:
    """A trace output directory: merged records plus provenance."""
    directory: str
    records: tuple
    partial: bool
    sources: dict              # edges stem -> "final" | "checkpoint"
    plugin_manifests: dict     # file name -> manifest dict
    run_manifest: dict | None
    integrity: dict            # edges file name -> sha256 matches its plugin manifest (None: no manifest)
    errors: tuple = ()
    duplicates: int = 0


def _stem_suffix(name: str, stem: str, extension: str) -> str | None:
    """'edges.gw0.partial.jsonl' -> 'gw0.partial'; 'edges.jsonl' -> ''."""
    if name == stem + extension:
        return ""
    if name.startswith(stem + ".") and name.endswith(extension):
        return name[len(stem) + 1:-len(extension)]
    return None


def load_trace_dir(directory, *, strict: bool = True) -> TraceRun:
    """Load ``edges*.jsonl`` (one per pytest process) with checkpoint fallback per process.

    A process whose final edges file is missing contributes its latest checkpoint
    (``edges[.<worker>].partial.jsonl``) and the run is marked partial.
    """
    base = Path(directory)
    workers: dict[str, dict] = {}
    for path in sorted(base.glob(EDGES_STEM + "*.jsonl")):
        suffix = _stem_suffix(path.name, EDGES_STEM, ".jsonl")
        if suffix is None:
            continue
        if suffix == PARTIAL_MARK[1:]:
            worker, partial = "", True
        elif suffix.endswith(PARTIAL_MARK):
            worker, partial = suffix[:-len(PARTIAL_MARK)], True
        else:
            worker, partial = suffix, False
        workers.setdefault(worker, {})["checkpoint" if partial else "final"] = path
    manifests = {}
    for path in sorted(base.glob(PLUGIN_MANIFEST_STEM + "*.json")):
        suffix = _stem_suffix(path.name, PLUGIN_MANIFEST_STEM, ".json")
        if suffix is None:
            continue
        with open(path, "r", encoding="utf-8") as stream:
            manifests[path.name] = json.load(stream)
    run_manifest = None
    if (base / RUN_MANIFEST_FILE).is_file():
        with open(base / RUN_MANIFEST_FILE, "r", encoding="utf-8") as stream:
            run_manifest = json.load(stream)
    chosen, sources = [], {}
    for worker in sorted(workers):
        entry = workers[worker]
        kind = "final" if "final" in entry else "checkpoint"
        chosen.append(entry[kind])
        sources[EDGES_STEM + ("." + worker if worker else "")] = kind
    edges = load_edges_files(chosen, strict=strict) if chosen else TraceEdges(records=())
    integrity = {}
    for path in chosen:
        expected = None
        for manifest in manifests.values():
            if manifest.get("edges_file") == path.name:
                expected = manifest.get("edges_sha256")
        integrity[path.name] = None if expected is None else expected == edges.file_sha256.get(path.name)
    partial = ("checkpoint" in sources.values()
               or any(m.get("partial") for name, m in manifests.items() if PARTIAL_MARK not in name)
               or bool((run_manifest or {}).get("partial")))
    return TraceRun(directory=str(base), records=edges.records, partial=partial, sources=sources,
                    plugin_manifests=manifests, run_manifest=run_manifest, integrity=integrity,
                    errors=edges.errors, duplicates=edges.duplicates)


# ---------------------------------------------------------------------------
# Observed caller sets
# ---------------------------------------------------------------------------
def observed_call_sites(records, *, caller_kinds=FUNCTION_CALLERS) -> dict[str, tuple]:
    """callee identity -> sorted section 2 call-site dicts, for the given caller kinds."""
    grouped: dict[str, set] = {}
    for record in records:
        if record.get("caller_kind", "function") not in caller_kinds:
            continue
        grouped.setdefault(record["callee"], set()).add(
            (record["caller"], record["path"], record["line"]))
    return {callee: tuple({"caller": caller, "callee": callee, "path": path, "line": line}
                          for caller, path, line in sorted(sites))
            for callee, sites in sorted(grouped.items())}


def observed_callers(records, *, caller_kinds=FUNCTION_CALLERS, include_self: bool = True) -> dict[str, frozenset]:
    """callee identity -> frozenset of observed caller identities.

    Defaults follow section 2: only function/method callers count (module-level and
    class-body callers are excluded); a recursive self-call is an observed call site
    and is kept unless ``include_self=False``.
    """
    result: dict[str, set] = {}
    for record in records:
        if record.get("caller_kind", "function") not in caller_kinds:
            continue
        if not include_self and record["caller"] == record["callee"]:
            continue
        result.setdefault(record["callee"], set()).add(record["caller"])
    return {callee: frozenset(callers) for callee, callers in sorted(result.items())}


def summarize_edges(records) -> dict:
    records = list(records)
    by_kind: dict[str, int] = {}
    for record in records:
        kind = record.get("caller_kind", "function")
        by_kind[kind] = by_kind.get(kind, 0) + 1
    callers = observed_callers(records)
    no_self = observed_callers(records, include_self=False)
    sizes = [len(value) for value in callers.values()]
    function_records = [r for r in records if r.get("caller_kind", "function") == "function"]
    return {
        "edges": len(records),
        "edges_by_caller_kind": dict(sorted(by_kind.items())),
        "edges_function_callers": len(function_records),
        "distinct_caller_callee_pairs": len({(r["caller"], r["callee"]) for r in function_records}),
        "distinct_function_callers": len({r["caller"] for r in function_records}),
        "distinct_callees": len({r["callee"] for r in records}),
        "callees_with_function_callers": len(callers),
        "callees_with_ge2_callers": sum(1 for n in sizes if n >= 2),
        "callees_with_ge2_callers_excluding_self": sum(1 for v in no_self.values() if len(v) >= 2),
        "callees_with_2_to_12_callers": sum(1 for n in sizes if 2 <= n <= 12),
        "anonymous_scope_records": sum(1 for r in records if r.get("caller_scope")),
        "self_call_records": sum(1 for r in function_records if r["caller"] == r["callee"]),
    }


# ---------------------------------------------------------------------------
# Source provenance and identity cross-checks
# ---------------------------------------------------------------------------
def source_tree_digest(root, package_dirs) -> dict:
    """sha256 of the canonical JSON ``{relpath: sha256(bytes)}`` map of the in-scope files.

    Delegates to identity.py (``package_files`` / ``file_sha256s`` /
    ``source_tree_sha256``, the graphs.py convention) so trace and static manifests
    agree; the fallback reproduces the same definition.
    """
    if _identity is not None:
        files = _identity.package_files(root, normalize_package_dirs(package_dirs))
        digest = _identity.source_tree_sha256(_identity.file_sha256s(root, files))
        return {"sha256": digest, "files": len(files), "method": SOURCE_TREE_METHOD}
    base = Path(root)
    dirs = normalize_package_dirs(package_dirs)
    entries: dict[str, str] = {}
    for directory in dirs:
        start = base / directory if directory else base
        if directory.endswith(".py"):
            if start.is_file() and not start.is_symlink() and path_in_scope(directory, dirs):
                entries[directory] = sha256_file(start)
            continue
        for current, dirnames, filenames in os.walk(start, followlinks=False):
            current_path = Path(current)
            dirnames[:] = sorted(d for d in dirnames if d not in _EXCLUDED_DIRS and d != "__pycache__"
                                 and not d.startswith(".") and not (current_path / d).is_symlink())
            for name in sorted(filenames):
                path = current_path / name
                relative = path.relative_to(base).as_posix()
                if path.is_file() and not path.is_symlink() and path_in_scope(relative, dirs):
                    entries[relative] = sha256_file(path)
    digest = hashlib.sha256(canonical_json(entries).encode("utf-8")).hexdigest()
    return {"sha256": digest, "files": len(entries), "method": SOURCE_TREE_METHOD}


def check_identities(records, root, *, index=None, max_examples: int = 10) -> dict:
    """Cross-check runtime identities against the pinned source (needs identity.FunctionIndex).

    For every record the callee qualname must name a ``def`` in the source. For function
    callers the caller must too, and ``enclosing_function(path, line)`` of the call site
    must be the traced caller. Returns counts and a few examples of each disagreement.
    """
    if index is None:
        if _identity is None:
            raise RuntimeError("check_identities needs absent_edges.probe.identity.FunctionIndex")
        index = _identity.FunctionIndex(root)
    qualname_cache: dict[str, set] = {}

    def qualnames(relpath):
        if relpath not in qualname_cache:
            qualname_cache[relpath] = {item["qualname"] for item in index.functions(relpath)}
        return qualname_cache[relpath]

    unknown_callees, unknown_callers, mismatched = {}, {}, []
    checked = 0
    for record in records:
        callee_rel, callee_q = split_identity(record["callee"])
        if callee_q not in qualnames(callee_rel):
            unknown_callees[record["callee"]] = unknown_callees.get(record["callee"], 0) + 1
        if record.get("caller_kind", "function") != "function":
            continue
        checked += 1
        caller_rel, caller_q = split_identity(record["caller"])
        if caller_q not in qualnames(caller_rel):
            unknown_callers[record["caller"]] = unknown_callers.get(record["caller"], 0) + 1
        static = index.enclosing_function(record["path"], record["line"])
        if static != record["caller"]:
            mismatched.append({"call_site": call_site(record), "static_enclosing": static,
                               "caller_scope": record.get("caller_scope")})
    return {
        "index_impl": type(index).__module__ + "." + type(index).__name__,
        "records_checked": len(records) if hasattr(records, "__len__") else None,
        "function_caller_records_checked": checked,
        "unknown_callee_identities": len(unknown_callees),
        "unknown_caller_identities": len(unknown_callers),
        "enclosing_mismatches": len(mismatched),
        "examples_unknown_callees": sorted(unknown_callees)[:max_examples],
        "examples_unknown_callers": sorted(unknown_callers)[:max_examples],
        "examples_enclosing_mismatches": mismatched[:max_examples],
    }
