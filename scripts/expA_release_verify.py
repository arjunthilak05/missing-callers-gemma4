#!/usr/bin/env python3
"""Verify the Absent Edges Exp A release package (v1). Standard library only; runs on a CPU in minutes.

    python -I -B reproduce/verify.py [--root DIR] [--jobs N] [--skip-shards] [--skip-analysis] [--skip-compare]
                                      [--report FILE.json]

(a) Hashes and pins.
    - MANIFEST.sha256: every listed file exists with its sha256 and no unlisted file is present.
      RELEASE_ID = sha256(MANIFEST.sha256).
    - PROVENANCE.json: every witness is re-read from its anchor (a JSON path, a sha256sum entry or a text line of a
      frozen file in this package) and compared with the file's sha256. For a sanitized file the witnesses pin the
      original bytes; the check compares them with the original sha256 recorded in SANITIZATION.md. Git blob ids and
      the deduplicated files (stored once) are checked too.
    - Shards: every member of every .tar.xz is streamed (nothing is written to disk, so the '|' in cell ids never
      becomes a file name) and compared with INDEX.jsonl. Inside each run: run.json and cells.jsonl equal the plain
      copies; every file hash in the publish ledger equals the member; every result.json names the sha256 of its
      events.jsonl and attempt_start.json; every score row names the sha256 of its terminal result.json.
(b) Analysis. For each run, the frozen analysis is re-run with the released code: code/<release>/, function
    absent_edges.probe.analysis.analyze_paths, called with the arguments that `scripts/expA_analyze.py analyze`
    passes (run binding required, plan sha256 pinned, default payload store). Every run and shard is checked in a
    fresh worker process (max_tasks_per_child=1), so a worker that imported one code release is never reused for a
    run pinned to another, whatever --jobs is. Scores are not re-computed from raw events (that needs the upstream
    source trees); the released scores are checked through the hash chains in (a). Inputs: the released scores,
    reference policies, items, analysis plan, dispatch manifest and the run's attempt records (read from the shard
    in memory; the released score.scan_run reads them through a read-only in-memory view, because cell ids contain
    '|').
    CELLS.jsonl and TABLES.md must be byte-identical. ANALYSIS.json must be byte-identical once its five recorded
    input paths (inputs.*.path, the Core2 paths at analysis time) replace the paths of this run.
(c) Independent re-analysis and cross-run comparisons (code/post-results/) are re-run and compared with the frozen
    REANALYSIS.json and COMPARE.json: identical except the fields that record the run time, command line and input
    paths. The frozen COMPARE.json files were written on Windows, so their CRLF line ends are normalised.
(d) Follow-up analyses (VERIFY_PLAN.json "followups": A1 final-turn policy, A2 hard-omission panel, A3 evidence ledger,
    and A4/A5 when their runs and outputs are in the package). Each pre-specified analysis script in
    code/post-results/ is re-run from the package's own scores (results/<run>/scores.jsonl), run.json and launch
    records, with the package root as working directory and the command line derived from the shipped output's own
    inputs record, and compared with the shipped output: byte for byte when its recorded input paths are
    package-relative (A1-A3 were computed from a package-shaped view), otherwise field by field with only the input path
    strings relabelled (the check message says which). A1's mechanism subgroup reads the episode records: they are
    streamed from the three run shards and turned into timeline rows by scripts/expA_timeline.py's own functions; with
    --skip-shards that subgroup (and S4_final_response_kind) is left out of the comparison and the message says so.
    Follow-up runs without an expA_analyze.py analysis ("scores_only" in the plan) get the shard and chain checks of
    (a) (plus their plain launch-record copies); their analysis is the follow-up check. --skip-compare skips (d).

Exit status 0 when every check passes, 1 otherwise.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath

LABEL = "Reviewed by the authors before release."
SUMS_LINE = re.compile(r"^([0-9a-f]{64}) [ *](.+)$")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_id(path) -> str:
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


class Report:
    def __init__(self):
        self.checks = []

    def add(self, name, ok, detail="", **extra):
        row = {"check": name, "ok": bool(ok), "detail": detail}
        row.update(extra)
        self.checks.append(row)
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)
        return ok

    @property
    def ok(self):
        return all(row["ok"] for row in self.checks)


# ================================================================================================ (a) hashes
def check_manifest(root: Path, report: Report):
    raw = (root / "MANIFEST.sha256").read_bytes()
    release_id = sha256_bytes(raw)
    listed, bad, missing = {}, [], []
    for line in raw.decode("utf-8").splitlines():
        m = SUMS_LINE.match(line)
        if not m:
            bad.append(line)
            continue
        listed[m.group(2)] = m.group(1)
    for rel, sha in listed.items():
        path = root / rel
        if not path.is_file():
            missing.append(rel)
        elif sha256_file(path) != sha:
            bad.append(rel)
    present = set()
    pycache = []
    for current, dirnames, filenames in os.walk(root):
        for name in filenames:
            rel = (Path(current) / name).relative_to(root).as_posix()
            if "/__pycache__/" in "/" + rel:
                pycache.append(rel)
                continue
            present.add(rel)
    extra = sorted(present - set(listed) - {"MANIFEST.sha256"})
    report.add("manifest", not bad and not missing and not extra,
               f"{len(listed)} files listed; mismatched {len(bad)}, missing {len(missing)}, unlisted {len(extra)}"
               + (f"; ignored {len(pycache)} __pycache__ files" if pycache else ""),
               mismatched=bad[:20], missing=missing[:20], unlisted=extra[:20])
    print(f"RELEASE_ID {release_id}")
    return release_id, listed


class Anchors:
    def __init__(self, root: Path):
        self.root = root
        self.json_cache, self.line_cache = {}, {}

    def json(self, rel):
        if rel not in self.json_cache:
            self.json_cache[rel] = json.loads((self.root / rel).read_text(encoding="utf-8"))
        return self.json_cache[rel]

    def lines(self, rel):
        if rel not in self.line_cache:
            self.line_cache[rel] = (self.root / rel).read_bytes().decode("utf-8", "replace").splitlines()
        return self.line_cache[rel]

    def check(self, witness, expected) -> bool:
        try:
            if witness["kind"] == "json":
                value = self.json(witness["anchor"])
                for key in witness["path"]:
                    value = value[key]
                return value == expected
            line = self.lines(witness["anchor"])[witness["line"] - 1]
            if witness["kind"] == "sums":
                m = SUMS_LINE.match(line)
                return bool(m) and m.group(1) == expected and m.group(2) == witness["entry"]
            return expected in line
        except (KeyError, IndexError, TypeError, ValueError, OSError):
            return False


def check_provenance(root: Path, listed: dict, report: Report):
    prov = json.loads((root / "PROVENANCE.json").read_text(encoding="utf-8"))
    anchors = Anchors(root)
    sanitized = {s["path"]: s for s in prov.get("sanitized", [])}
    n_files = n_wit = n_blob = 0
    failures = []
    for rec in prov["files"]:
        n_files += 1
        path = rec["path"]
        if listed.get(path) != rec["sha256"]:
            failures.append(f"{path}: sha256 differs from MANIFEST.sha256")
            continue
        if rec.get("git_blob") and git_blob_id(root / path) != rec["git_blob"]:
            failures.append(f"{path}: git blob id differs")
        else:
            n_blob += bool(rec.get("git_blob"))
        expected = rec["sha256"]
        if rec.get("original_sha256"):
            note = sanitized.get(path)
            if not note or note["original_sha256"] != rec["original_sha256"] or note["new_sha256"] != rec["sha256"]:
                failures.append(f"{path}: sanitized file without a matching SANITIZATION record")
            expected = rec["original_sha256"]
        for witness in rec.get("witnesses", []):
            n_wit += 1
            if not anchors.check(witness, expected):
                failures.append(f"{path}: witness {witness['anchor']} {witness.get('path') or witness.get('entry') or witness.get('line')} does not hold {expected[:12]}")
    for rec in prov.get("dedup", []):
        target = listed.get(rec["same_bytes_as"])
        if target != rec["sha256"]:
            failures.append(f"{rec['path']}: stored copy {rec['same_bytes_as']} has another sha256")
        for witness in rec.get("witnesses", []):
            n_wit += 1
            if not anchors.check(witness, rec["sha256"]):
                failures.append(f"{rec['path']}: witness {witness['anchor']} does not hold {rec['sha256'][:12]}")
    unlisted_in_prov = sorted(set(listed) - {r["path"] for r in prov["files"]} - {"PROVENANCE.json"})
    if unlisted_in_prov:
        failures.append(f"files in MANIFEST.sha256 without a PROVENANCE record: {unlisted_in_prov[:5]}")
    report.add("provenance", not failures,
               f"{n_files} files, {len(prov.get('dedup', []))} deduplicated files, {n_wit} witnesses re-read, "
               f"{n_blob} git blob ids, {len(sanitized)} sanitized file(s) checked against their original pins",
               failures=failures[:30])
    return prov


# ================================================================================================ shards
def _index(root: Path, rel: str):
    rows = []
    for line in (root / rel).read_bytes().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def stream_shard(root: Path, shard_rel: str, index_rel: str, keep):
    """Compare every member with the index; return (problems, kept {member: bytes}, rows by member)."""
    rows = _index(root, index_rel)
    by_member = {row["member"]: row for row in rows}
    problems, kept, seen = [], {}, 0
    with tarfile.open(root / shard_rel, mode="r|xz") as tar:
        for position, member in enumerate(tar):
            if not member.isfile():
                problems.append(f"{member.name}: not a regular file")
                continue
            if position >= len(rows) or rows[position]["member"] != member.name:
                problems.append(f"{member.name}: out of index order at {position}")
            row = by_member.get(member.name)
            handle = tar.extractfile(member)
            digest, size, data = hashlib.sha256(), 0, [] if keep(member.name) else None
            while True:
                chunk = handle.read(1 << 20)
                if not chunk:
                    break
                digest.update(chunk)
                size += len(chunk)
                if data is not None:
                    data.append(chunk)
            if row is None:
                problems.append(f"{member.name}: not in the index")
            elif row["sha256"] != digest.hexdigest() or row["bytes"] != size:
                problems.append(f"{member.name}: sha256 or size differs from the index")
            if data is not None:
                kept[member.name] = b"".join(data)
            seen += 1
    if seen != len(rows):
        problems.append(f"{len(rows) - seen} index rows have no member")
    return problems, kept, by_member


class _VFS:
    def __init__(self, files: dict, real_dir: str, run_label: str):
        self.files = files                      # rel -> bytes
        self.real_dir = real_dir
        self.label = run_label
        self.dirs = {""}
        self.children = {}
        for rel in files:
            parts = rel.split("/")
            for depth in range(1, len(parts)):
                self.dirs.add("/".join(parts[:depth]))
        for rel in list(files) + list(self.dirs):
            if rel:
                parent, _, name = rel.rpartition("/")
                self.children.setdefault(parent, set()).add(name)


class VPath:
    """Read-only view of a run's attempt records held in memory (used only by score.scan_run)."""

    def __init__(self, fs: _VFS, rel: str = ""):
        self.fs, self.rel = fs, rel

    def __truediv__(self, name):
        return VPath(self.fs, f"{self.rel}/{name}" if self.rel else str(name))

    def __fspath__(self):  # analysis.check_binding reads run.json from this directory
        return os.path.join(self.fs.real_dir, *self.rel.split("/")) if self.rel else self.fs.real_dir

    def __str__(self):
        return f"{self.fs.label}/{self.rel}" if self.rel else self.fs.label

    __repr__ = __str__

    @property
    def name(self):
        return self.rel.rpartition("/")[2]

    @property
    def parent(self):  # used by expA_timeline.process_cell (result_path.parent / events_name)
        return VPath(self.fs, self.rel.rpartition("/")[0])

    def is_dir(self):
        return self.rel in self.fs.dirs

    def is_file(self):
        return self.rel in self.fs.files

    def is_symlink(self):
        return False

    def iterdir(self):
        for name in sorted(self.fs.children.get(self.rel, ())):
            yield self / name

    def read_bytes(self):
        return self.fs.files[self.rel]

    def relative_to(self, other):
        if other.rel and not self.rel.startswith(other.rel + "/"):
            raise ValueError(f"{self} is not under {other}")
        return PurePosixPath(self.rel[len(other.rel) + 1:] if other.rel else self.rel)

    def __lt__(self, other):
        return self.rel < other.rel

    def __eq__(self, other):
        return isinstance(other, VPath) and self.rel == other.rel and self.fs is other.fs

    def __hash__(self):
        return hash(self.rel)


def _path_shim(real_path):
    class _Meta(type):
        def __call__(cls, *args, **kwargs):
            if len(args) == 1 and not kwargs and isinstance(args[0], VPath):
                return args[0]
            return real_path(*args, **kwargs)

        def __instancecheck__(cls, obj):
            return isinstance(obj, (real_path, VPath))

        def __getattr__(cls, name):
            return getattr(real_path, name)

    return _Meta("Path", (), {})


def json_diff(a, b, path="", out=None, limit=8):
    """Up to `limit` JSON paths where a and b differ (for failure messages)."""
    out = [] if out is None else out
    if len(out) >= limit:
        return out
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                out.append(f"{path}.{key} (only in {'frozen' if key in b else 'regenerated'})")
            else:
                json_diff(a[key], b[key], f"{path}.{key}", out, limit)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path} (length {len(a)} vs {len(b)})")
        for i, (x, y) in enumerate(zip(a, b)):
            json_diff(x, y, f"{path}[{i}]", out, limit)
    elif a != b or type(a) is not type(b):
        out.append(f"{path}: regenerated {str(a)[:80]!r} vs frozen {str(b)[:80]!r}")
    return out[:limit]


FLOAT_REL_TOL = 1e-12


def float_close_diff(a, b, path="", out=None):
    """(exact differences, float differences within FLOAT_REL_TOL). Python 3.12 changed float summation (sum() now
    compensates rounding), so a re-run on a newer Python can differ from the frozen file in the last digits only."""
    out = ([], []) if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                out[0].append(f"{path}.{key}")
            else:
                float_close_diff(a[key], b[key], f"{path}.{key}", out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out[0].append(f"{path} (length)")
        for i, (x, y) in enumerate(zip(a, b)):
            float_close_diff(x, y, f"{path}[{i}]", out)
    elif isinstance(a, float) and isinstance(b, float) and a != b:
        if abs(a - b) <= FLOAT_REL_TOL * max(1.0, abs(a), abs(b)):
            out[1].append(path)
        else:
            out[0].append(path)
    elif a != b or type(a) is not type(b):
        out[0].append(path)
    return out


def _run_job(root_text: str, entry: dict) -> dict:
    """Worker: stream one run shard, check its chains, re-run its analysis. Returns check rows."""
    sys.dont_write_bytecode = True
    root = Path(root_text)
    rows = []
    run = entry["run"]
    prefix = run + "/"

    def keep(name):
        rel = name[len(prefix):]
        return (rel in ("run.json", "cells.jsonl") or rel.startswith("publish/")
                or rel.endswith(("/attempt_start.json", "/result.json", "/attempt_closed.json")))

    t0 = time.time()
    problems, kept, by_member = stream_shard(root, entry["shard"], entry["index"], keep)
    rows.append(("shard " + run, not problems, f"{len(by_member)} members streamed and hashed in "
                 f"{time.time() - t0:.0f} s", problems[:20]))
    scores_only = entry.get("scores_only")
    if entry.get("analysis") is None and not scores_only:
        return {"run": run, "rows": rows}
    chain = []
    plain_copies = [("run.json", entry["run_json"]), ("cells.jsonl", entry["cells_jsonl"])]
    if scores_only:  # a follow-up run without an expA_analyze.py analysis: its launch records are copied too
        plain_copies += [(rel, f"{run_dir_of(entry)}/{rel}") for rel in scores_only.get("plain_copies", [])]
    for name, plain in plain_copies:
        if by_member.get(prefix + name, {}).get("sha256") != sha256_file(root / plain):
            chain.append(f"{plain} differs from shard member {name}")
    ledger_files = 0
    for name, data in kept.items():
        rel = name[len(prefix):]
        if rel.startswith("publish/") and rel.endswith(".json"):
            record = json.loads(data)
            for f in record.get("files") or []:
                ledger_files += 1
                row = by_member.get(prefix + f["path"])
                if row is None or row["sha256"] != f["sha256"]:
                    chain.append(f"publish ledger {rel}: {f['path']} differs")
    results = 0
    for name, data in kept.items():
        if not name.endswith("/result.json"):
            continue
        results += 1
        result = json.loads(data)
        directory = name.rsplit("/", 1)[0]
        events = result.get("events") or {}
        if events.get("sha256") and by_member.get(f"{directory}/{events.get('path', 'events.jsonl')}", {}).get(
                "sha256") != events["sha256"]:
            chain.append(f"{name}: events sha256 differs")
        context = result.get("run_context") or {}
        start = by_member.get(f"{directory}/attempt_start.json")
        if context.get("start_sha256") and (start is None or start["sha256"] != context["start_sha256"]):
            chain.append(f"{name}: attempt_start sha256 differs")
    if scores_only:
        scores_path = root / scores_only["scores"]
    else:
        inputs = entry["analysis"]["inputs"]
        scores_path = root / inputs["scores"]["stored_path"]
    scored = 0
    for line in scores_path.read_bytes().splitlines():
        row = json.loads(line)
        if row.get("result_sha256"):
            scored += 1
            member = f"{prefix}cells/{row['cell_id']}/{row['attempt_id']}/result.json"
            if by_member.get(member, {}).get("sha256") != row["result_sha256"]:
                chain.append(f"score row {row['cell_id']}: result sha256 differs")
    copies = "run.json and cells.jsonl copies" if not scores_only else \
        f"run.json, cells.jsonl and {len(plain_copies) - 2} launch-record copies"
    rows.append(("chain " + run, not chain, f"{copies}; {ledger_files} publish-ledger "
                 f"hashes; {results} result.json -> events/start hashes; {scored} score rows -> result.json"
                 + ("; no expA_analyze.py analysis (follow-up run: its pre-specified analysis is checked under "
                    "'follow-up')" if scores_only else ""),
                 chain[:20]))
    if entry.get("skip_analysis") or scores_only:
        return {"run": run, "rows": rows}
    # (b) re-run the frozen analysis with the released code
    code = root / entry["analysis"]["code"]
    sys.path.insert(0, str(code))
    from absent_edges.probe import analysis as AN  # noqa: E402
    from absent_edges.probe import score as S  # noqa: E402
    files = {}
    for name, data in kept.items():
        rel = name[len(prefix):]
        if rel.startswith("cells/"):
            files[rel] = data
    for name in by_member:  # events.jsonl presence only (scan_run reads its existence, never its bytes)
        rel = name[len(prefix):]
        if rel.startswith("cells/") and rel.endswith("/events.jsonl"):
            files.setdefault(rel, b"")
    with tempfile.TemporaryDirectory(prefix="expA-verify-") as tmp:
        real_dir = Path(tmp) / "run"
        real_dir.mkdir()
        (real_dir / "run.json").write_bytes(kept[prefix + "run.json"])
        vroot = VPath(_VFS(files, str(real_dir), run))
        out = Path(tmp) / "out"
        real_path = S.Path
        S.Path = _path_shim(real_path)
        t0 = time.time()
        try:
            AN.analyze_paths(
                items_path=str(root / inputs["items"]["stored_path"]), scores_path=str(scores_path),
                policies_path=str(root / inputs["policies"]["stored_path"]),
                plan_path=str(root / inputs["plan"]["stored_path"]), out_dir=str(out), run_dir=vroot,
                dispatch_manifest=str(root / entry["analysis"]["dispatch_manifest"]), require_run=True,
                overwrite=False, plan_sha256=inputs["plan"]["sha256"], payloads_dir=None)
        except Exception as error:  # noqa: BLE001
            rows.append(("analysis " + run, False, f"analysis refused or failed: {error!r}", []))
            return {"run": run, "rows": rows}
        finally:
            S.Path = real_path
        frozen = entry["analysis"]["frozen"]
        diffs = []
        for name in ("CELLS.jsonl", "TABLES.md"):
            if (out / name).read_bytes() != (root / frozen[name]).read_bytes():
                diffs.append(f"{name} differs")
        # The recorded input paths are replaced in the bytes (a JSON round trip would reorder the integer keys
        # that the analysis serialised with sort_keys, so it cannot be used).
        regenerated = (out / "ANALYSIS.json").read_bytes()
        new = json.loads(regenerated)
        frozen_raw = (root / frozen["ANALYSIS.json"]).read_bytes()
        frozen_doc = json.loads(frozen_raw)
        relabelled = []
        for key, meta in (new.get("inputs") or {}).items():
            original = ((frozen_doc.get("inputs") or {}).get(key) or {}).get("path")
            if original is None or meta.get("path") == original:
                continue
            old_text = json.dumps(meta["path"], ensure_ascii=False).encode("utf-8")
            new_text = json.dumps(original, ensure_ascii=False).encode("utf-8")
            if regenerated.count(b'"path": ' + old_text) != 1:
                diffs.append(f"ANALYSIS.json: inputs.{key}.path is not recorded exactly once")
                continue
            regenerated = regenerated.replace(b'"path": ' + old_text, b'"path": ' + new_text)
            meta["path"] = original
            relabelled.append(key)
        float_note = ""
        if regenerated != frozen_raw:
            exact, close = float_close_diff(new, frozen_doc)
            if not exact and close and sys.version_info >= (3, 12):
                float_note = (f" except {len(close)} float(s) that differ only in the last digits (difference "
                              f"<= {FLOAT_REL_TOL:g} x max(1, |value|); Python {sys.version_info.major}.{sys.version_info.minor} sums "
                              f"floats with compensated rounding, the frozen files were made with Python 3.11)")
            else:
                diffs.append("ANALYSIS.json differs (after relabelling the recorded input paths): "
                             + "; ".join(json_diff(new, frozen_doc) or ["same JSON values, different bytes"])
                             + (f"; beyond the float tolerance: {', '.join(exact[:8])}" if exact else ""))
        rows.append(("analysis " + run, not diffs,
                     f"re-run with {entry['analysis']['code']} in {time.time() - t0:.0f} s: CELLS.jsonl and TABLES.md "
                     f"byte-identical, ANALYSIS.json byte-identical after relabelling inputs.{{{','.join(relabelled)}}}"
                     f".path{float_note}" if not diffs else "; ".join(diffs), diffs))
    return {"run": run, "rows": rows}


def run_dir_of(entry: dict) -> str:
    return entry["run_json"].rsplit("/", 1)[0]


# ------------------------------------------------------------------------------------------------ follow-ups
def _shard_timeline_rows(root: Path, module, shards: dict):
    """A replacement for expA_followup_analysis.timeline_rows_from_run that reads the run records from the run's
    shard in-stream (cell ids contain '|', so members are never written to disk). It builds the same rows with the
    same functions of scripts/expA_timeline.py (item_info, _init_worker, process_cell), one attempt directory at a time,
    and returns them in the order of the score rows, as the original does."""
    import expA_timeline as T  # noqa: E402  (code/post-results is on sys.path)

    def timeline_rows_from_run(run_dir, items_path, score_rows, workers):
        items = {}
        for record in module.load_jsonl(items_path):
            info = T.item_info(record)
            items[info["item_id"]] = info
        tasks = [r for r in score_rows if r["cell_state"] == "completed" and r.get("result_path")]
        by_dir = {}
        for task in tasks:
            parts = task["result_path"].split("/")
            if len(parts) != 4 or parts[0] != "cells":
                raise ValueError(f"unexpected result_path {task['result_path']!r}")
            by_dir.setdefault("/".join(parts[:3]), []).append(task)
        shard = shards[str(run_dir)]
        prefix = PurePosixPath(shard).parent.name + "/"
        rows, real_path = {}, T.Path
        current, files = None, {}

        def flush():
            T._init_worker(items, VPath(_VFS(files, str(root / run_dir), str(run_dir))))
            for task in by_dir[current]:
                row, _audit = T.process_cell(task)
                rows[id(task)] = row

        T.Path = _path_shim(real_path)
        try:
            with tarfile.open(root / shard, mode="r|xz") as tar:
                for member in tar:
                    rel = member.name[len(prefix):] if member.name.startswith(prefix) else ""
                    if current is not None and not rel.startswith(current + "/"):
                        flush()
                        current, files = None, {}
                    directory = "/".join(rel.split("/")[:3])
                    if member.isfile() and directory in by_dir and rel.count("/") >= 3:
                        current = directory
                        files[rel] = tar.extractfile(member).read()
                if current is not None:
                    flush()
        finally:
            T.Path = real_path
        missing = [t["cell_id"] for t in tasks if id(t) not in rows]
        if missing:
            raise ValueError(f"{shard}: no records for {len(missing)} completed cells, e.g. {missing[:3]}")
        return [rows[id(t)] for t in tasks]

    return timeline_rows_from_run


def _input_path_pairs(new, frozen, out):
    """Set every input path string recorded in `new` to the one in `frozen`; collect (new, frozen) pairs."""
    if isinstance(new, dict) and isinstance(frozen, dict):
        for key in new:
            if key not in frozen:
                continue
            if key == "path" and isinstance(new[key], str) and isinstance(frozen[key], str):
                if new[key] != frozen[key]:
                    out.append((new[key], frozen[key]))
                    new[key] = frozen[key]
            else:
                _input_path_pairs(new[key], frozen[key], out)
    elif isinstance(new, list) and isinstance(frozen, list):
        for x, y in zip(new, frozen):
            _input_path_pairs(x, y, out)
    return out


def _drop(document, dotted):
    *parents, last = dotted.split(".")
    for key in parents:
        document = document.get(key) if isinstance(document, dict) else None
    if isinstance(document, dict):
        document.pop(last, None)


def _followup_job(root_text: str, spec: dict, use_shards: bool) -> dict:
    """Worker: re-run one follow-up's pre-specified analysis from the package's scores and code (working directory =
    the package root, so the recorded input paths are package-relative) and compare it with the shipped output."""
    import contextlib
    import importlib
    import io
    sys.dont_write_bytecode = True
    root = Path(root_text)
    name = "follow-up " + spec["name"]
    t0 = time.time()
    os.chdir(root)
    script = root / spec["script"]
    sys.path.insert(0, str(script.parent))
    module = importlib.import_module(script.stem)
    argv, ignore, how = list(spec["argv"]), [], ""
    if spec.get("timeline_shards"):
        if use_shards:
            module.timeline_rows_from_run = _shard_timeline_rows(root, module, spec["timeline_shards"])
            how = "; episode timelines read in-stream from the run shards with expA_timeline.py's own functions"
        else:
            argv, ignore = list(spec["without_shards"]["argv"]), spec["without_shards"]["ignore"]
            how = "; " + spec["without_shards"]["why"]
    with tempfile.TemporaryDirectory(prefix="expA-followup-") as tmp:
        out = Path(tmp) / "out"
        stdout, stderr = io.StringIO(), io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = module.main(argv + ["--out", str(out)])
        except SystemExit as error:
            code = error.code
        except Exception as error:  # noqa: BLE001
            return {"run": name, "rows": [(name, False, f"analysis failed: {error!r}", [])]}
        if code not in (0, None):
            return {"run": name, "rows": [(name, False, f"exit {code}: {(stderr.getvalue() or str(code)).strip()[-300:]}",
                                           [])]}
        produced = {fname: (out / fname).read_bytes() for fname in spec["outputs"]}
    diffs, notes, pairs = [], [], []
    for fname, pkg in sorted(spec["outputs"].items(), key=lambda kv: kv[0] != spec["json"]):
        new_raw, frozen_raw = produced[fname], (root / pkg).read_bytes()
        # text-mode writes on Windows: the A2 outputs were written there (CRLF), and so are this run's on Windows
        crlf = b"\r\n" in frozen_raw
        new_raw, frozen_raw = new_raw.replace(b"\r\n", b"\n"), frozen_raw.replace(b"\r\n", b"\n")
        if new_raw == frozen_raw and not ignore:
            notes.append(f"{fname} byte-identical" + (" after normalising the shipped file's CRLF line ends"
                                                      if crlf else ""))
            continue
        if fname == spec["json"]:
            new, frozen = json.loads(new_raw), json.loads(frozen_raw)
            _input_path_pairs(new.get("inputs"), frozen.get("inputs"), pairs)
            for dotted in ignore:
                _drop(new, dotted)
                _drop(frozen, dotted)
            if new == frozen:
                what = []
                if pairs:
                    what.append(f"{len(pairs)} recorded input path(s) that are not package-relative")
                if ignore:
                    what.append(", ".join(ignore))
                notes.append(f"{fname} identical field by field except " + " and ".join(what))
            else:
                diffs.append(f"{fname} differs: " + "; ".join(json_diff(new, frozen) or ["same values"]))
        elif ignore:
            notes.append(f"{fname} not compared (it renders the fields not compared)")
        else:
            text = new_raw
            for new_path, frozen_path in pairs:
                text = text.replace(new_path.encode("utf-8"), frozen_path.encode("utf-8"))
            if text == frozen_raw:
                notes.append(f"{fname} identical after the same input-path relabelling")
            else:
                diffs.append(f"{fname} differs")
    detail = (f"re-ran {spec['script']} {argv[0] if not argv[0].startswith('-') else ''}".rstrip()
              + f" in {time.time() - t0:.0f} s: " + "; ".join(diffs or notes) + how)
    return {"run": name, "rows": [(name, not diffs, detail, diffs)]}


def _truth_job(root_text: str, entry: dict) -> dict:
    root = Path(root_text)
    t0 = time.time()
    problems, _, by_member = stream_shard(root, entry["shard"], entry["index"], lambda name: False)
    return {"run": entry["shard"], "rows": [("shard " + entry["shard"], not problems,
                                             f"{len(by_member)} members streamed and hashed in "
                                             f"{time.time() - t0:.0f} s", problems[:20])]}


def check_member_witnesses(root: Path, prov: dict, report: Report):
    anchors = Anchors(root)
    failures, n = [], 0
    for shard, members in (prov.get("shard_member_witnesses") or {}).items():
        index = {row["member"]: row for row in _index(root, prov["shards"][shard]["index"])}
        for member, witnesses in members.items():
            for witness in witnesses:
                n += 1
                if not anchors.check(witness, index[member]["sha256"]):
                    failures.append(f"{shard}:{member} witness {witness['anchor']}")
    report.add("shard member witnesses", not failures, f"{n} witnesses of shard members re-read", failures=failures[:20])


# ================================================================================================ (c)
def run_reanalysis(root: Path, plan: dict, report: Report):
    spec = plan["reanalysis"]
    with tempfile.TemporaryDirectory(prefix="expA-reanalysis-") as tmp:
        cmd = [sys.executable, "-I", "-B", str(root / spec["script"]), "--cells", str(root / spec["cells"]),
               "--analysis", str(root / spec["analysis"]), "--out-dir", tmp,
               "--tcdf-crosscheck", str(root / spec["tcdf_crosscheck"])]
        proc = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True)
        if proc.returncode != 0:
            return report.add("independent re-analysis", False, f"exit {proc.returncode}: {proc.stderr.strip()[-300:]}")
        new = json.loads((Path(tmp) / "REANALYSIS.json").read_text(encoding="utf-8"))
    frozen_raw = (root / spec["frozen"]).read_bytes()
    frozen = json.loads(frozen_raw)
    for key in ("generated_utc", "command"):
        new[key] = frozen[key]
    for key in ("cells", "analysis", "tcdf_crosscheck"):
        if new["inputs"].get(key) and frozen["inputs"].get(key):
            new["inputs"][key]["path"] = frozen["inputs"][key]["path"]
    same = (json.dumps(new, indent=1, ensure_ascii=True) + "\n").encode("utf-8") == frozen_raw
    summ = new.get("comparison_summary") or {}
    report.add("independent re-analysis", same and new.get("verdict") == "PASS",
               f"verdict {new.get('verdict')}, {summ.get('pass')}/{summ.get('compared')} values agree; REANALYSIS.json "
               f"{'identical' if same else 'DIFFERS'} except generated_utc, command and inputs.*.path")


def run_comparisons(root: Path, plan: dict, report: Report):
    script = root / plan["compare_script"]
    for spec in plan["comparisons"]:
        with tempfile.TemporaryDirectory(prefix="expA-compare-") as tmp:
            out = Path(tmp) / "COMPARE.json"
            cmd = [sys.executable, "-I", "-B", str(script), "--mode", spec["mode"],
                   "--ref-cells", str(root / spec["ref_cells"]), "--new-cells", str(root / spec["new_cells"]),
                   "--out", str(out)]
            if spec["mode"] == "model":
                cmd += ["--ref-analysis", str(root / spec["ref_analysis"]),
                        "--new-analysis", str(root / spec["new_analysis"])]
                if spec.get("labels"):  # the classification wording the frozen run used (absent: the default)
                    cmd += ["--labels", spec["labels"]]
            else:
                cmd += ["--seed", str(spec["seed"])]
            proc = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True)
            if proc.returncode != 0:
                report.add("comparison " + spec["name"], False, f"exit {proc.returncode}: {proc.stderr.strip()[-300:]}")
                continue
            new = json.loads(out.read_text(encoding="utf-8"))
        frozen_raw = (root / spec["frozen"]).read_bytes()
        frozen = json.loads(frozen_raw)
        for key, meta in new["inputs"].items():
            meta["path"] = frozen["inputs"][key]["path"]
        regenerated = (json.dumps(new, indent=1, sort_keys=True) + "\n").encode("utf-8")
        same = regenerated == frozen_raw.replace(b"\r\n", b"\n")
        report.add("comparison " + spec["name"], same and new == frozen,
                   f"{spec['mode']} mode; COMPARE.json {'identical' if same else 'DIFFERS'} except inputs.*.path "
                   "(frozen file has CRLF line ends)")


# ================================================================================================ main
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--jobs", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--skip-shards", action="store_true", help="skip shard streaming and the analysis re-runs")
    parser.add_argument("--skip-analysis", action="store_true", help="stream shards but do not re-run analyses")
    parser.add_argument("--skip-compare", action="store_true", help="skip the re-analysis and comparisons")
    parser.add_argument("--report", help="write the check rows as JSON")
    args = parser.parse_args(argv)
    sys.dont_write_bytecode = True
    if sys.version_info < (3, 11):
        print(f"verify.py needs Python 3.11 or later (this is {sys.version.split()[0]})", file=sys.stderr)
        return 2
    root = Path(args.root).resolve()
    started = time.time()
    print(f"Absent Edges Exp A release verification ({LABEL})")
    print(f"root {root}; python {sys.version.split()[0]} on {sys.platform}")
    report = Report()
    release_id, listed = check_manifest(root, report)
    prov = check_provenance(root, listed, report)
    plan = json.loads((root / "reproduce/VERIFY_PLAN.json").read_text(encoding="utf-8"))
    dedup = {rec["path"]: rec["same_bytes_as"] for rec in prov.get("dedup", [])}
    jobs = []
    if not args.skip_shards:
        check_member_witnesses(root, prov, report)
        for entry in plan["runs"]:
            entry = json.loads(json.dumps(entry))
            if entry.get("analysis"):
                for meta in entry["analysis"]["inputs"].values():
                    meta["stored_path"] = dedup.get(meta["package_path"], meta["package_path"])
                entry["skip_analysis"] = args.skip_analysis
            jobs.append(("run", entry, (root / entry["shard"]).stat().st_size))
        for entry in plan["truth_shards"]:
            jobs.append(("truth", entry, (root / entry["shard"]).stat().st_size))
    if not args.skip_compare:
        for spec in plan.get("followups") or []:
            # a follow-up that streams run shards (A1's timelines) is as long as those shards together
            weight = sum((root / s).stat().st_size for s in (spec.get("timeline_shards") or {}).values()) \
                if not args.skip_shards else 0
            jobs.append(("followup", spec, weight))
    if jobs:
        jobs.sort(key=lambda job: -job[2])
        context = multiprocessing.get_context("spawn")
        workers = {"run": _run_job, "truth": _truth_job, "followup": _followup_job}
        # one task per worker process: _run_job imports absent_edges from the run's own code release, and
        # _followup_job changes its working directory to the package root
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs, mp_context=context,
                                                    max_tasks_per_child=1) as pool:
            futures = [pool.submit(workers[kind], str(root), entry, *((not args.skip_shards,)
                                                                      if kind == "followup" else ()))
                       for kind, entry, _ in jobs]
            results = [future.result() for future in futures]
        followups = [r for r in results if r["run"].startswith("follow-up ")]
        for result in sorted((r for r in results if r not in followups), key=lambda r: r["run"]):
            for name, ok, detail, problems in result["rows"]:
                report.add(name, ok, detail, problems=problems)
    else:
        followups = []
    if not args.skip_compare:
        run_reanalysis(root, plan, report)
        run_comparisons(root, plan, report)
        order = {spec["name"]: i for i, spec in enumerate(plan.get("followups") or [])}
        for result in sorted(followups, key=lambda r: order.get(r["run"][len("follow-up "):], 0)):
            for name, ok, detail, problems in result["rows"]:
                report.add(name, ok, detail, problems=problems)
    elapsed = time.time() - started
    passed = sum(row["ok"] for row in report.checks)
    print(f"SUMMARY {passed}/{len(report.checks)} checks passed in {elapsed:.0f} s; RELEASE_ID {release_id}; "
          f"{'ALL PASS' if report.ok else 'FAILURES'}")
    if args.report:
        Path(args.report).write_text(json.dumps({"release_id": release_id, "ok": report.ok, "seconds": round(elapsed),
                                                 "python": sys.version.split()[0], "platform": sys.platform,
                                                 "checks": report.checks}, indent=1) + "\n", encoding="utf-8")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
