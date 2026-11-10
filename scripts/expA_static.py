#!/usr/bin/env python3
"""Export Exp A static caller edges for one repository family and one backend.

Writes ``<out>/<family>/<backend>.edges.jsonl`` (one call-site record per line) and
``<out>/<family>/<backend>.manifest.json`` (backend pin, source tree sha256, counts,
errors, wall time); see docs/EXPA_DESIGN.md section 4. Backend ``locagent`` also keeps
the raw engine document as ``<out>/<family>/locagent.document.json``.

Backend ``jedi`` must run under the Python that has the pinned jedi (Core2
``expA/envs/static/bin/python``). Backend ``locagent`` runs the pinned LocAgent through
``graphs.GraphBackend`` with ``--graph-python`` (Core2 ``envs/graph/bin/python``).
Concurrent invocations are safe: each run uses its own work directory, and outputs are
written to temporary files and atomically renamed (no lock files).

Exit status: 0 complete, 3 partial (some jedi definitions failed or a package file did not
parse; recorded), 1 failed (outputs still written, with the error in the manifest), 2 usage
error (nothing written).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from absent_edges.probe.static_callers import (BACKENDS, DEFAULT_GRAPH_PYTHON,  # noqa: E402
                                               DEFAULT_JEDI_ATTEMPTS, DEFAULT_JEDI_MAX_ATTEMPTS,
                                               DEFAULT_LOCAGENT_ROOT, run_static_backend)


def _progress(done, total, seconds):
    print(f"[expA_static] {done}/{total} definitions, {seconds:.0f}s", file=sys.stderr, flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--family", required=True, help="repository family name (output subdirectory)")
    parser.add_argument("--repo-root", required=True, type=Path, help="pinned repository checkout")
    parser.add_argument("--package-dirs", required=True, nargs="+",
                        help="package source dirs (or single-module .py files) relative to the repo root")
    parser.add_argument("--backend", required=True, choices=BACKENDS)
    parser.add_argument("--out", required=True, type=Path, help="static output root (writes <out>/<family>/)")
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1),
                        help="jedi worker processes (default: min(16, cpus))")
    parser.add_argument("--timeout", type=int, default=None,
                        help="seconds: LocAgent engine run (default 900) or per jedi attempt (default 300)")
    parser.add_argument("--jedi-attempts", type=int, default=DEFAULT_JEDI_ATTEMPTS,
                        help="jedi queries per definition; the union is kept (default %(default)s)")
    parser.add_argument("--jedi-max-attempts", type=int, default=DEFAULT_JEDI_MAX_ATTEMPTS,
                        help="queries for a definition whose answers differ (default %(default)s)")
    parser.add_argument("--locagent-root", type=Path,
                        default=Path(os.environ.get("ABSENT_EDGES_LOCAGENT_ROOT", DEFAULT_LOCAGENT_ROOT)))
    parser.add_argument("--graph-python", type=Path,
                        default=Path(os.environ.get("ABSENT_EDGES_GRAPH_PYTHON", DEFAULT_GRAPH_PYTHON)))
    parser.add_argument("--work-dir", type=Path, default=None,
                        help="private scratch directory (must not exist; default under <out>/<family>/)")
    parser.add_argument("--keep-work", action="store_true", help="keep the staged source and engine files")
    parser.add_argument("--overwrite", action="store_true", help="replace existing outputs for this family/backend")
    args = parser.parse_args(argv)
    try:
        manifest = run_static_backend(
            family=args.family, repo_root=args.repo_root, package_dirs=args.package_dirs,
            backend=args.backend, out_root=args.out, work_dir=args.work_dir, workers=args.workers,
            timeout=args.timeout, vendor_root=args.locagent_root, graph_python=args.graph_python,
            overwrite=args.overwrite, keep_work=args.keep_work, jedi_attempts=args.jedi_attempts,
            jedi_max_attempts=args.jedi_max_attempts,
            argv=sys.argv if argv is None else ["expA_static.py", *argv], progress=_progress)
    except (FileExistsError, FileNotFoundError, ValueError, ImportError) as error:
        print(f"expA_static: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    summary = {key: manifest[key] for key in ("family", "backend", "status", "source_tree_sha256",
                                              "counts", "wall_time_s")}
    summary["errors"] = len(manifest["errors"])
    summary["outputs"] = manifest["outputs"]
    if "work_dir" in manifest:
        summary["work_dir"] = manifest["work_dir"]
    print(json.dumps(summary, indent=1, sort_keys=True))
    return {"complete": 0, "partial": 3}.get(manifest["status"], 1)


if __name__ == "__main__":
    sys.exit(main())
