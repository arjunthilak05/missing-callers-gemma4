"""pytest plugin for Exp A (lane T): package-internal call sites via ``sys.monitoring``.

Load with ``-p absent_edges.probe.tracer_plugin``. Never install it into a repository
environment: ``scripts/expA_trace.py`` stages this package on PYTHONPATH instead.

Configuration comes only from the environment (names in ``absent_edges.probe.tracer``):

  AE_TRACE_ROOT                repository root
  AE_TRACE_PACKAGE_DIRS        root-relative package dirs, os.pathsep-separated
  AE_TRACE_OUT                 output directory
  AE_TRACE_MODE                "trace" (default) or "count" (test outcomes only, no monitoring)
  AE_TRACE_DEADLINE_EPOCH      optional soft deadline (time.time()); no new test starts after it
  AE_TRACE_CHECKPOINT_SECONDS  checkpoint period (default 120; 0 disables)

Semantics (docs/EXPA_DESIGN.md section 5):

* On PY_START the callee is the started code object. The caller is the nearest Python
  frame below it: inside the callback ``sys._getframe(1)`` is the started frame and its
  ``f_back`` is the caller. C-level trampolines (map, sorted keys, functools.partial,
  lru_cache, property, type.__call__) leave no Python frame, so the Python code that
  invoked them is the caller. Out-of-scope Python frames are never skipped.
* A call site is recorded only when both code objects live in package-scope files under
  the root. Identity is ``relpath::co_qualname``; the line is the caller frame's line
  when the callee started.
* Generator, coroutine and async-generator callees (co_flags), lambdas, other anonymous
  scopes, class bodies and module code are never callees; their PY_START is disabled for
  this tool. Callers inside lambdas/genexprs are attributed to the enclosing def
  (``caller_scope`` keeps the runtime qualname). Module-level and class-body callers are
  recorded with ``caller_kind`` module/class_body.
* Deduplicated in memory. At session finish the edges JSONL and then the plugin manifest
  are written atomically (tmp + fsync + rename). Periodic checkpoints, a soft deadline
  and a SIGTERM handler keep partial results, marked ``partial``.
"""
from __future__ import annotations

import collections
import datetime
import os
from pathlib import Path
import platform
import signal
import sys
import threading
import time

from absent_edges.probe import tracer as _T

try:  # the tracer core stays importable (and testable) without pytest
    import pytest as _pytest
except ImportError:  # pragma: no cover - exercised only in pytest-less interpreters
    _pytest = None


def _hookimpl(**options):
    if _pytest is None:
        return lambda function: function
    return _pytest.hookimpl(**options)


CO_OPTIMIZED = 0x0001
CO_GENERATOR = 0x0020
CO_COROUTINE = 0x0080
CO_ASYNC_GENERATOR = 0x0200
TOOL_NAME = "absent-edges-expA-tracer"
TOOL_ID_PREFERENCE = (3, 4, 2, 5, 0, 1)  # unreserved ids first; coverage (1) last
OUTCOME_CATEGORIES = ("passed", "failed", "error", "skipped", "xfailed", "xpassed")
_MISSING = object()

LIMITATIONS = (
    "Observed calls are positive evidence only; an unobserved edge is not an impossible edge.",
    "Only pytest processes that load the plugin are traced (all of their threads); subprocesses and "
    "forked children are not.",
    "Callees entered through a Python wrapper in package scope (for example a functools.wraps decorator) are "
    "attributed to the wrapper: the decorated target's observed caller is the wrapper, and the target's callers "
    "appear as callers of the wrapper. An out-of-scope Python wrapper hides the call entirely. C-level wrappers "
    "(functools.partial, lru_cache, staticmethod, classmethod, property) attribute the call to the real caller.",
    "Callbacks run by C code (map, filter, sorted key, re.sub) are attributed to the Python frame that invoked "
    "the C code; a lazy iterator consumed elsewhere attributes the call to the consumer, whose source may not "
    "mention the callee.",
    "Memoized callees (lru_cache) start only on cache misses; callers that only hit the cache are not observed.",
    "Calls inside lambdas, generator expressions and annotation scopes are attributed to the enclosing def "
    "(caller_scope keeps the runtime qualname). Generator, coroutine, async-generator, lambda and other "
    "anonymous callees are excluded.",
    "Module-level and class-body callers are recorded with caller_kind module/class_body; function caller sets "
    "exclude them.",
    "Code compiled from strings (doctest examples, exec, generated methods) has no file under the root and is "
    "out of scope; the package must be imported from the checkout, not from a site-packages copy.",
    "The line is the caller frame's line when the callee started: the call expression's line (CPython reports "
    "the attribute's line for a multi-line attribute call).",
)


class TracerError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def callee_exclusion(code) -> str | None:
    """Why a code object can never be a callee identity, or None if it can."""
    name, flags = code.co_name, code.co_flags
    if name == "<module>":
        return "module"
    if not flags & CO_OPTIMIZED:
        return "class_body"
    if flags & CO_ASYNC_GENERATOR:
        return "async_generator"
    if flags & CO_COROUTINE:
        return "coroutine"
    if flags & CO_GENERATOR:
        return "generator"
    if name == "<lambda>":
        return "lambda"
    if name.startswith("<"):
        return "anonymous"
    return None


class EdgeTracer:
    """Records (caller, callee, caller line) for package-scope code with a PY_START callback.

    Caches are keyed by ``id(code)`` and every classified code object is kept alive, so
    ids stay unique for the tracer's lifetime (code objects compare by value across files,
    so they are never used as dict keys).
    """

    def __init__(self, root, package_dirs):
        if not hasattr(sys, "monitoring"):
            raise TracerError("sys.monitoring is unavailable (Python 3.12+ required)")
        self.root = os.path.realpath(os.fspath(root))
        if not os.path.isdir(self.root):
            raise TracerError(f"trace root is not a directory: {root}")
        self.package_dirs = _T.normalize_package_dirs(package_dirs)
        self.cwd = os.getcwd()
        self.current_test = "<startup>"
        self.tool_id = None
        self.released_tool_id = None
        self.events = 0
        self.frame_mismatches = 0
        self.classify_errors = 0
        self.files_in_scope = 0
        self.files_out_of_scope = 0
        self.edges: dict = {}
        self.callee_code_objects = collections.Counter()
        self.caller_code_objects = collections.Counter()
        self._files: dict = {}
        self._callee: dict = {}
        self._caller: dict = {}
        self._keep: list = []
        self._callee_ids: list = []
        self._callee_index: dict = {}
        self._caller_infos: list = []
        self._caller_index: dict = {}

    # -- classification (cold path; never raises) ---------------------------
    def _relpath(self, filename):
        rel = self._files.get(filename, _MISSING)
        if rel is not _MISSING:
            return rel
        rel = None
        try:
            if filename and not (filename[0] == "<" and filename[-1] == ">"):
                path = filename if os.path.isabs(filename) else os.path.join(self.cwd, filename)
                real = os.path.realpath(path)
                prefix = self.root + os.sep
                if real.startswith(prefix):
                    candidate = real[len(prefix):].replace(os.sep, "/")
                    if _T.path_in_scope(candidate, self.package_dirs):
                        rel = candidate
        except Exception:
            self.classify_errors += 1
            rel = None
        self._files[filename] = rel
        if rel is None:
            self.files_out_of_scope += 1
        else:
            self.files_in_scope += 1
        return rel

    def _classify_callee(self, code):
        index = -1
        try:
            self._keep.append(code)
            rel = self._relpath(code.co_filename)
            reason = "out_of_scope" if rel is None else callee_exclusion(code)
            if reason is None:
                identity = _T.function_identity(rel, code.co_qualname)
                index = self._callee_index.get(identity, -1)
                if index < 0:
                    self._callee_ids.append(identity)
                    index = len(self._callee_ids) - 1
                    self._callee_index[identity] = index
                reason = "traced"
            self.callee_code_objects[reason] += 1
        except Exception:
            self.classify_errors += 1
            index = -1
        self._callee[id(code)] = index
        return index

    def _classify_caller(self, code):
        index = -1
        try:
            self._keep.append(code)
            rel = self._relpath(code.co_filename)
            if rel is None:
                self.caller_code_objects["out_of_scope"] += 1
            else:
                if code.co_name == "<module>":
                    qualname, kind, scope = _T.MODULE_QUALNAME, "module", None
                elif not code.co_flags & CO_OPTIMIZED:
                    qualname, kind, scope = code.co_qualname, "class_body", None
                else:
                    qualname, kind, scope = _T.normalize_caller_qualname(code.co_qualname)
                info = (_T.function_identity(rel, qualname), rel, kind, scope)
                index = self._caller_index.get(info, -1)
                if index < 0:
                    self._caller_infos.append(info)
                    index = len(self._caller_infos) - 1
                    self._caller_index[info] = index
                self.caller_code_objects[kind] += 1
                if scope is not None:
                    self.caller_code_objects["anonymous_scope_normalized"] += 1
        except Exception:
            self.classify_errors += 1
            index = -1
        self._caller[id(code)] = index
        return index

    def _find_frame(self, code):
        self.frame_mismatches += 1
        frame = sys._getframe(2)
        for _ in range(16):
            if frame is None:
                return None
            if frame.f_code is code:
                return frame
            frame = frame.f_back
        return None

    # -- hot path ------------------------------------------------------------
    def _make_callback(self):
        callee_get = self._callee.get
        caller_get = self._caller.get
        classify_callee = self._classify_callee
        classify_caller = self._classify_caller
        find_frame = self._find_frame
        edges = self.edges
        getframe = sys._getframe
        disable = sys.monitoring.DISABLE
        state = self

        def on_py_start(code, instruction_offset):
            callee = callee_get(id(code))
            if callee is None:
                callee = classify_callee(code)
            if callee < 0:
                return disable
            state.events += 1
            frame = getframe(1)
            if frame.f_code is not code:
                frame = find_frame(code)
                if frame is None:
                    return None
            caller_frame = frame.f_back
            if caller_frame is None:
                return None
            caller_code = caller_frame.f_code
            caller = caller_get(id(caller_code))
            if caller is None:
                caller = classify_caller(caller_code)
            if caller < 0:
                return None
            key = (caller, callee, caller_frame.f_lineno)
            if key not in edges:
                edges[key] = state.current_test
            return None

        return on_py_start

    # -- lifecycle -----------------------------------------------------------
    def start(self):
        if self.tool_id is not None:
            raise TracerError("tracer already started")
        monitoring = sys.monitoring
        for candidate in TOOL_ID_PREFERENCE:
            if monitoring.get_tool(candidate) is not None:
                continue
            try:
                monitoring.use_tool_id(candidate, TOOL_NAME)
            except ValueError:
                continue
            self.tool_id = candidate
            break
        else:
            raise TracerError("no free sys.monitoring tool id")
        try:
            monitoring.register_callback(self.tool_id, monitoring.events.PY_START, self._make_callback())
            # Events another holder of this id disabled must not hide our PY_STARTs.
            monitoring.restart_events()
            monitoring.set_events(self.tool_id, monitoring.events.PY_START)
        except BaseException:
            monitoring.free_tool_id(self.tool_id)
            self.tool_id = None
            raise
        return self

    def stop(self):
        if self.tool_id is None:
            return
        monitoring = sys.monitoring
        try:
            monitoring.set_events(self.tool_id, 0)
            monitoring.register_callback(self.tool_id, monitoring.events.PY_START, None)
        finally:
            monitoring.free_tool_id(self.tool_id)
            self.released_tool_id, self.tool_id = self.tool_id, None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc_info):
        self.stop()

    # -- output --------------------------------------------------------------
    def records(self):
        """Canonical, deduplicated call-site records plus write-time stats."""
        snapshot = list(self.edges.items())  # atomic under the GIL; lists below only grow
        caller_infos, callee_ids = list(self._caller_infos), list(self._callee_ids)
        rows, dropped = [], 0
        for (caller, callee, line), first_test in snapshot:
            if line is None or line < 1:
                dropped += 1
                continue
            identity, path, kind, scope = caller_infos[caller]
            row = {"caller": identity, "callee": callee_ids[callee], "path": path, "line": line,
                   "caller_kind": kind}
            if scope is not None:
                row["caller_scope"] = scope
            if first_test is not None:
                row["first_test"] = first_test
            rows.append(row)
        records = _T.dedupe_records(rows)
        return records, {"dropped_line_unknown": dropped, "merged_duplicate_rows": len(rows) - len(records)}

    def counters(self) -> dict:
        return {
            "py_start_events_in_scope": self.events,
            "callee_code_objects": dict(sorted(self.callee_code_objects.items())),
            "caller_code_objects": dict(sorted(self.caller_code_objects.items())),
            "files_in_scope": self.files_in_scope,
            "files_out_of_scope": self.files_out_of_scope,
            "frame_mismatches": self.frame_mismatches,
            "classify_errors": self.classify_errors,
            "edge_keys_in_memory": len(self.edges),
            "traced_callee_identities": len(self._callee_ids),
        }


# ---------------------------------------------------------------------------
# pytest integration
# ---------------------------------------------------------------------------
class PluginState:
    def __init__(self, environ=None):
        env = os.environ if environ is None else environ
        root, dirs, out = env.get(_T.ENV_ROOT), env.get(_T.ENV_PACKAGE_DIRS), env.get(_T.ENV_OUT)
        missing = [name for name, value in ((_T.ENV_ROOT, root), (_T.ENV_PACKAGE_DIRS, dirs), (_T.ENV_OUT, out))
                   if not value]
        if missing:
            raise TracerError("missing environment variable(s): " + ", ".join(missing))
        self.mode = env.get(_T.ENV_MODE) or "trace"
        if self.mode not in _T.MODES:
            raise TracerError(f"{_T.ENV_MODE} must be one of {_T.MODES}, got {self.mode!r}")
        self.root = os.path.realpath(root)
        self.package_dirs = _T.normalize_package_dirs(dirs)
        self.out = Path(out)
        self.deadline = float(env[_T.ENV_DEADLINE]) if env.get(_T.ENV_DEADLINE) else None
        self.checkpoint_seconds = float(env.get(_T.ENV_CHECKPOINT) or 120)
        self.worker = env.get("PYTEST_XDIST_WORKER") or None
        self.pid = os.getpid()
        self.started_at = utc_now()
        self.started_monotonic = time.monotonic()
        self.finished_at = None
        self.tracer = EdgeTracer(self.root, self.package_dirs) if self.mode == "trace" else None
        self.config = None
        self.tests = collections.Counter()
        self.collected = None
        self.deselected = 0
        self.collect_errors = 0
        self.partial_reason = None
        self.exit_status = None
        self.finalized = False
        self.last_checkpoint = time.monotonic()
        self.checkpoints_written = 0
        self.checkpoint_errors = 0
        self.sigterm_handler_installed = False
        self._old_sigterm = None
        self.out.mkdir(parents=True, exist_ok=True)

    def start(self):
        if self.tracer is not None:
            self.tracer.start()

    def names(self, final: bool):
        suffix = ("." + self.worker if self.worker else "") + ("" if final else _T.PARTIAL_MARK)
        return f"{_T.EDGES_STEM}{suffix}.jsonl", f"{_T.PLUGIN_MANIFEST_STEM}{suffix}.json"

    def set_test(self, label):
        if self.tracer is not None:
            self.tracer.current_test = label

    def manifest(self, final, rows, stats, edges_name, edges_sha256) -> dict:
        tracer = self.tracer
        outcomes = dict(sorted(self.tests.items()))
        tool_id = None
        if tracer is not None:
            tool_id = tracer.tool_id if tracer.tool_id is not None else tracer.released_tool_id
        counts = None
        if tracer is not None:
            counts = tracer.counters()
            counts.update(stats)
        return {
            "schema": _T.PLUGIN_SCHEMA,
            "final": final,
            "mode": self.mode,
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "python_executable": sys.executable,
            "pytest_version": getattr(_pytest, "__version__", None),
            "platform": platform.platform(),
            "pid": self.pid,
            "xdist_worker": self.worker,
            "root": self.root,
            "package_dirs": list(self.package_dirs),
            "identity_impl": _T.IDENTITY_IMPL,
            "monitoring_tool_id": tool_id,
            "started_at_utc": self.started_at,
            "finished_at_utc": self.finished_at if final else None,
            "wall_seconds": round(time.monotonic() - self.started_monotonic, 3),
            "exit_status": self.exit_status,
            "partial": (not final) or self.partial_reason is not None,
            "partial_reason": self.partial_reason if final else (self.partial_reason or "checkpoint"),
            "deadline_epoch": self.deadline,
            "checkpoint_seconds": self.checkpoint_seconds,
            "checkpoints_written": self.checkpoints_written,
            "checkpoint_errors": self.checkpoint_errors,
            "sigterm_handler_installed": self.sigterm_handler_installed,
            "tests": {"collected": self.collected, "deselected": self.deselected,
                      "collect_errors": self.collect_errors, "outcomes": outcomes,
                      "ran": sum(self.tests.get(name, 0) for name in OUTCOME_CATEGORIES)},
            "counts": counts,
            "edges_schema": _T.EDGES_SCHEMA if tracer is not None else None,
            "edges_file": edges_name,
            "edges_sha256": edges_sha256,
            "edges_count": len(rows) if tracer is not None else None,
            "limitations": list(LIMITATIONS) if tracer is not None else [],
        }

    def write(self, final: bool) -> dict:
        edges_name, manifest_name = self.names(final)
        rows, stats, sha = [], {}, None
        if self.tracer is not None:
            rows, stats = self.tracer.records()
            sha = _T.write_jsonl_atomic(self.out / edges_name, rows)
        manifest = self.manifest(final, rows, stats, edges_name if self.tracer is not None else None, sha)
        _T.write_json_atomic(self.out / manifest_name, manifest)
        return manifest

    def checkpoint(self):
        if self.finalized or os.getpid() != self.pid:
            return
        try:
            self.write(final=False)
            self.checkpoints_written += 1
        except Exception:
            self.checkpoint_errors += 1
        self.last_checkpoint = time.monotonic()

    def after_test(self, session):
        self.set_test("<between-tests>")
        if (self.deadline is not None and time.time() >= self.deadline
                and not getattr(session, "shouldstop", False)):
            session.shouldstop = "absent-edges tracer: wall-time cap reached"
            if self.partial_reason is None:
                self.partial_reason = "wall_time_cap"
        if self.checkpoint_seconds > 0 and time.monotonic() - self.last_checkpoint >= self.checkpoint_seconds:
            self.checkpoint()

    def count_report(self, report):
        category = None
        try:
            status = self.config.hook.pytest_report_teststatus(report=report, config=self.config)
            category = status[0] if status else None
        except Exception:
            if report.when == "call":
                category = report.outcome
            elif report.failed:
                category = "error"
        if category:
            self.tests[category] += 1

    def install_sigterm(self):
        if threading.current_thread() is not threading.main_thread():
            return
        try:
            current = signal.getsignal(signal.SIGTERM)
        except (AttributeError, ValueError):
            return
        if current not in (signal.SIG_DFL, None):
            return  # never replace a handler someone else installed
        self._old_sigterm = current
        signal.signal(signal.SIGTERM, _on_sigterm)
        self.sigterm_handler_installed = True

    def _set_sigterm(self, handler):
        if self.sigterm_handler_installed and threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGTERM, handler)

    def finalize(self, exit_status, *, reason=None):
        if self.finalized or os.getpid() != self.pid:
            return
        self.finalized = True
        self._set_sigterm(signal.SIG_IGN)  # a late SIGTERM must not interrupt the final write
        try:
            if self.tracer is not None:
                self.tracer.stop()
            if exit_status is not None:
                self.exit_status = int(exit_status)
            if reason and self.partial_reason is None:
                self.partial_reason = reason
            if self.partial_reason is None and self.exit_status == 2:
                self.partial_reason = "collection_errors" if self.collect_errors else "interrupted"
            if self.partial_reason is None and self.exit_status == 3:
                self.partial_reason = "internal_error"
            self.finished_at = utc_now()
            self.write(final=True)
            for name in self.names(final=False):
                try:
                    (self.out / name).unlink()
                except FileNotFoundError:
                    pass
        finally:
            self._set_sigterm(self._old_sigterm if self._old_sigterm is not None else signal.SIG_DFL)


_STATE: PluginState | None = None


def _on_sigterm(signum, frame):
    state = _STATE
    if state is not None and state.partial_reason is None:
        state.partial_reason = "sigterm"
    raise KeyboardInterrupt("absent-edges tracer: SIGTERM received")


def _usage_error(message):
    if _pytest is not None:
        return _pytest.UsageError(message)
    return TracerError(message)


def _ensure_started(config):
    global _STATE
    if _STATE is not None:
        return _STATE
    try:
        state = PluginState()
        state.config = config
        state.start()
    except (TracerError, ValueError, OSError) as error:
        raise _usage_error(f"absent-edges tracer: {error}") from None
    _STATE = state
    return state


def _owned(config):
    state = _STATE
    if state is None or state.config is not config or os.getpid() != state.pid:
        return None
    return state


@_hookimpl(tryfirst=True)
def pytest_load_initial_conftests(early_config, parser, args):
    # Start before conftest.py files import the package so import-time calls are seen.
    _ensure_started(early_config)


@_hookimpl(tryfirst=True)
def pytest_configure(config):
    state = _ensure_started(config)
    if state.config is config:
        state.install_sigterm()


def pytest_report_header(config):
    state = _owned(config)
    if state is None:
        return None
    tool = state.tracer.tool_id if state.tracer is not None else None
    return (f"absent-edges tracer: mode={state.mode} tool_id={tool} identity={_T.IDENTITY_IMPL} "
            f"package_dirs={','.join(state.package_dirs)}")


@_hookimpl(hookwrapper=True)
def pytest_collection(session):
    state = _owned(session.config)
    if state is not None:
        state.set_test("<collection>")
    yield
    if state is not None:
        state.set_test("<between-tests>")


def pytest_collectreport(report):
    state = _STATE
    if state is not None and report.failed:
        state.collect_errors += 1


def pytest_deselected(items):
    state = _STATE
    if state is not None:
        state.deselected += len(items)


def pytest_collection_finish(session):
    state = _owned(session.config)
    if state is not None:
        state.collected = len(session.items)


@_hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    state = _owned(item.config)
    if state is not None:
        state.set_test(item.nodeid)
    yield
    if state is not None:
        state.after_test(item.session)


def pytest_runtest_logreport(report):
    state = _STATE
    if state is not None and state.config is not None and os.getpid() == state.pid:
        state.count_report(report)


@_hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    state = _owned(session.config)
    if state is not None:
        state.set_test("<sessionfinish>")
        state.finalize(exitstatus)


def pytest_unconfigure(config):
    global _STATE
    state = _owned(config)
    if state is None:
        return
    if not state.finalized:
        state.finalize(None, reason="no_sessionfinish")
    _STATE = None
