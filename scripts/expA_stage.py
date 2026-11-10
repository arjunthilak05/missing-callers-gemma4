"""Exp A one-shot root stager for Core1 (docs/EXPA_RUNBOOK.md, docs/EXPA_CORE1_RUNTIME.md).

Run once per run with sudo on ai-core-team-1, under the root-owned /usr/bin/python3, through the sha256
loader of scripts/expA_core1_launch.sh (it executes exactly the bytes whose sha256 the coordinator pinned).
Root is needed only because the Core2 sources live in the 0700 private project directory, which Core1's
thilak.arjun (UID 1008) cannot read; the NFS export has no_root_squash. Standard library only, Python 3.8+.

It copies, from Core2 paths read through NFS, into /dev/shm/ae-expA-<run>/inputs/ (the run root must be
local tmpfs, mode 0700 and owned by the run owner):

  release/          the frozen code release: exactly the files of RELEASE_FILES.sha256, plus that file
  dispatch/         the dispatch view (contract item 4): dispatch.jsonl, dispatch_manifest.json and the
                    files that manifest pins, payloads/<family>.jsonl and graph_facts/<family>.json
  checkouts/        manifest.json, the pinned clean checkout manifest (tools.CHECKOUTS_SCHEMA, written by
                    scripts/expA_mirror_repos.py), and repos/<family>/ for every dispatch family: exactly
                    the files, symlinks and directories of the family's manifest entry
  STAGE_RECEIPT.<component>.json   what was staged, from where, with every sha256

The runner then takes --dispatch inputs/dispatch, --checkout-manifest inputs/checkouts/manifest.json and
--repos-root inputs/checkouts/repos; the supervisor and runner code come from inputs/release.

Run state (component ``run-state``, with --run-target, the Core2 run directory of this run id): every stage
call compares the records already published on Core2 with <run root>/run/ before any runner launch.

  target holds no run.json                 -> "empty" (a new run)
  target holds records, run/ is empty      -> refused, unless --restore: then every published run record
                                              (run.json, cells.jsonl, launches/, snapshots/, cells/<cell>/<attempt>/
                                              files) is copied into run/ (files 0444, directories 0700, the run
                                              owner), verified and renamed into place -> "restored"
  target holds records, run/ has run.json  -> every published record must be in run/ byte for byte
                                              -> "consistent", else refused

A reboot wipes /dev/shm: after it, the runner must never start on an empty run directory (it would dispatch
completed cells again and reuse attempt ids). After a restore the runner's reconcile completes the published
cells and closes every restored start-only attempt as abandoned (its owner's boot id differs), so the cell's
next attempt gets the next ordinal. The receipt (STAGE_RECEIPT.run-state.<time>.json) lists those attempts:
their events and results were never published and are lost.

Everything staged is owned by the run owner (thilak.arjun, UID 1008 on Core1): directories 0700, files
0444. Every file is hashed while it is copied, re-hashed after it is written, and must equal its pin: the
dispatch manifest (--expect-dispatch-manifest-sha256), RELEASE_FILES.sha256 (--expect-release-sums-sha256)
and the clean checkout manifest (--expect-checkouts-manifest-sha256). Every source is validated before
anything is written.
A component is staged in a temporary directory, verified and then renamed into place; a component that is
already staged is verified again, never rewritten.

The stager refuses items.jsonl and every truth file: a source path or staged path named items.jsonl,
truth/score/eligibility files, *.edges.jsonl, or anything under facts/, eligibility/, static/, traces/,
truth/, scores/ or items/; a dispatch directory that contains such a file; dispatch rows with any key
outside the dispatch view (V, U, sites, omitted, covariates, ...); payload-store rows other than
{sha256, bytes, payload} or payloads that carry truth keys; and graph facts that carry truth keys or are
not the natural LocAgent facts of the family's source tree.
"""
from __future__ import annotations

import argparse
import collections
import ctypes
import ctypes.util
import datetime
import hashlib
import json
import os
import pwd
import re
import socket
import stat
import sys
import uuid

STAGE_PROTOCOL = "expA-stage-v2"
RECEIPT_SCHEMA = "expA-stage-receipt-v2"
DISPATCH_SCHEMA = "expA-dispatch-v2"
CHECKOUTS_SCHEMA = "expA-clean-checkouts-v1"   # absent_edges.probe.tools.CHECKOUTS_SCHEMA
TREE_PROTOCOL = "expA-checkout-tree-v1"       # absent_edges.probe.tools.TREE_PROTOCOL (tree_digest below)
COMPONENTS = ("release", "dispatch", "checkouts")
RUN_STATE = "run-state"
RUN_TARGET_PARENTS = ("/mnt/kldata/absent-edges-20261018-core2/expA/runs",
                      "/mnt/kldata/absent-edges-20261018-core2/expA/dev/v2/runtime/publish-test")
RUN_RECORD_FILES = ("run.json", "cells.jsonl")
RUN_RECORD_DIRS = ("launches", "snapshots")
ATTEMPT_FILES = ("attempt_start.json", "events.jsonl", "result.json", "attempt_closed.json")
RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
RELEASE_SUMS = "RELEASE_FILES.sha256"
DISPATCH_FILE = "dispatch.jsonl"
DISPATCH_MANIFEST = "dispatch_manifest.json"
CHECKOUTS_MANIFEST = "manifest.json"
EXPECTED_OWNER = "thilak.arjun"
OWNER_UID_BY_HOST = {"ai-core-team-1": 1008, "ai-core-team-2": 1006}
FORBIDDEN_UIDS_BY_HOST = {"ai-core-team-1": frozenset({1006})}
RUN_PARENT = "/dev/shm"
TMPFS_MAGIC = 0x01021994
RUN_ROOT_NAME = re.compile(r"ae-expA-[a-z0-9][a-z0-9-]{0,63}")
SHA256 = re.compile(r"[0-9a-f]{64}")
FAMILY = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
MAX_RELEASE_FILE_BYTES = 64 * 1024**2
MAX_INPUT_BYTES = 1024**3
VARIANTS = frozenset({"vcomplete", "natural", "omit_node", "omit_edge"})
PAYLOAD_CONDITIONS = frozenset({"C1", "C2", "C3", "C4a"})
DISPATCH_ROW_KEYS = frozenset({"item_id", "family", "target", "package_dirs", "wording_pair", "membership", "cells"})
DISPATCH_ROW_OPTIONAL = frozenset({"schema"})
TARGET_KEYS = frozenset({"identity", "path", "line", "qualname"})
# Exactly the three flags (items.DISPATCH_MEMBERSHIP_KEYS): s_nat_excluded_reason would reveal natural_miss.
MEMBERSHIP_KEYS = frozenset({"in_s_nat", "in_s_edge", "in_t"})
PAYLOAD_ROW_KEYS = frozenset({"sha256", "bytes", "payload"})
# Keys that carry ground truth or pre-outcome covariates (contract items 3 and 4). None may reach Core1.
TRUTH_KEYS = frozenset({"V", "U", "sites", "omitted", "covariates", "lexical_sites", "eligibility", "variants",
                        "truth", "verified", "unverified", "stratum", "strata", "natural_miss", "n_V", "V_count",
                        "U_count", "c4_contains_omitted_site", "c4_trigger_fires", "visibility_text_matches",
                        "visibility_docref_mentions", "visibility_name_mentions",
                        "runtime_witnessed", "witnesses", "order_key", "omitted_caller", "natural_callers",
                        "natural_missing_verified", "natural_false_positives", "match_sources"})
TRUTH_BASENAMES = frozenset({"items.jsonl", "items.json", "truth.jsonl", "truth.json", "scores.jsonl",
                             "score.json", "eligibility.jsonl", "eligibility.json", "policies.jsonl"})
TRUTH_DIR_PARTS = frozenset({"facts", "eligibility", "static", "traces", "truth", "scores", "items"})
TRUTH_SUFFIXES = (".edges.jsonl",)
# tools.STAGE_SKIP_DIRS / STAGE_SKIP_SUFFIXES: never part of a clean checkout.
SKIP_DIRS = frozenset({".git", ".hg", ".svn", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
                       ".tox", ".nox", ".hypothesis", ".eggs", ".venv", "venv"})
SKIP_SUFFIXES = (".pyc", ".pyo")
O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
BLOCK = 1024 * 1024
MAX_DIR_FDS = 64  # open directory descriptors kept by StagedTree (LRU); the root's is never closed


class StageRefused(RuntimeError):
    """The inputs are not exactly what the pins say, or a truth file was offered."""


def canonical_json(value) -> bytes:
    """Same encoding as absent_edges.data.canonical_json."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def sha256_bytes(data) -> str:
    return hashlib.sha256(data).hexdigest()


def tree_digest(files, symlinks, directories) -> str:
    """tools.tree_digest: sha256(canonical_json({protocol, files, symlinks, directories}))."""
    return sha256_bytes(canonical_json({"protocol": TREE_PROTOCOL, "files": dict(files), "symlinks": dict(symlinks),
                                        "directories": list(directories)}))


def package_tree_sha256(files, package_files) -> str:
    """identity.source_tree_sha256 over the package files: sha256(canonical_json({relpath: sha256}))."""
    return sha256_bytes(canonical_json({path: files[path] for path in package_files}))


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def utc_stamp():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def strict_json(data, what):
    def reject(value):
        raise ValueError(f"non-finite number {value}")
    try:
        return json.loads(data, parse_constant=reject)
    except (UnicodeDecodeError, ValueError) as error:
        raise StageRefused(f"{what} is not valid JSON: {error}") from error


def normalized_relpath(value, what):
    """A safe relative POSIX path: no absolute path, no '..', no empty or '.' parts, no backslash."""
    if not isinstance(value, str) or not value or "\\" in value or "\0" in value or "\n" in value:
        raise StageRefused(f"{what}: {value!r} is not a relative path")
    if value.startswith("/") or any(part in ("", ".", "..") for part in value.split("/")):
        raise StageRefused(f"{what}: {value!r} is not a normalized relative path")
    return value


def is_truth_path(relpath):
    parts = relpath.split("/")
    return (parts[-1] in TRUTH_BASENAMES or parts[-1].endswith(TRUTH_SUFFIXES)
            or any(part in TRUTH_DIR_PARTS for part in parts[:-1]))


def refuse_truth_path(relpath, what):
    """Refuse items.jsonl and every file that holds ground truth or truth-derived data."""
    if is_truth_path(relpath):
        raise StageRefused(f"{what}: refusing to stage truth file {relpath!r}")
    return relpath


def refuse_truth_paths(component, paths):
    """refuse_truth_path for every path of a component, except the clean checkouts' repos/ trees.

    Those trees are public repository files pinned one by one by the checkout manifest (git archive at the
    pinned commit plus generated in-scope files); their directory names are not truth markers. For example,
    genshi ships examples/tutorial/geddit/static/jquery.js (main-20261101a, refused by the name rule).
    """
    for path in paths:
        if component == "checkouts" and path.startswith("repos/"):
            continue
        refuse_truth_path(path, component)


def find_truth_keys(value, path="$", depth=0):
    """Paths of truth keys anywhere in a JSON value (bounded depth)."""
    if depth > 64:
        return [path + "(too deep)"]
    found = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key in TRUTH_KEYS:
                found.append(f"{path}.{key}")
            found += find_truth_keys(child, f"{path}.{key}", depth + 1)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found += find_truth_keys(child, f"{path}[{index}]", depth + 1)
    return found


def parse_sums(text, what):
    """{relpath: sha256} from sha256sum output; normalized relative paths only."""
    entries = {}
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-f]{64}) [ *](.+)", line)
        if match is None:
            raise StageRefused(f"{what} line {number} is not sha256sum output")
        name = match.group(2)
        while name.startswith("./"):
            name = name[2:]
        name = normalized_relpath(name, f"{what} line {number}")
        if name in entries:
            raise StageRefused(f"{what} lists {name} twice")
        entries[name] = match.group(1)
    if not entries:
        raise StageRefused(f"{what} is empty")
    return entries


def symlink_stays_inside(link_relpath, target):
    """tools.symlink_escape, lexically: a relative target that resolves inside the tree."""
    if not isinstance(target, str) or not target or target.startswith("/") or "\0" in target:
        return False
    parts = link_relpath.split("/")[:-1]
    for part in target.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            if not parts:
                return False
            parts.pop()
        else:
            parts.append(part)
    return True


def implied_directories(paths, extra=()):
    directories = set(extra)
    for path in paths:
        parts = path.split("/")[:-1]
        for index in range(1, len(parts) + 1):
            directories.add("/".join(parts[:index]))
    return directories


# ---------------------------------------------------------------------------
# Filesystem helpers (no symlink is ever followed below a given root)
# ---------------------------------------------------------------------------

def read_mounts():
    """[(mount_point, fstype)] from /proc/self/mountinfo, in mount order."""
    mounts = []
    with open("/proc/self/mountinfo", encoding="utf-8", errors="surrogateescape") as stream:
        for line in stream:
            fields = line.split()
            if len(fields) < 8 or "-" not in fields[6:]:
                continue
            separator = fields.index("-", 6)
            point = re.sub(r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), fields[4])
            mounts.append((point, fields[separator + 1]))
    return mounts


def statfs_magic(path):
    if not hasattr(os, "uname") or os.uname().sysname != "Linux":
        return None
    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    buffer = ctypes.create_string_buffer(256)
    if libc.statfs(os.fsencode(path), buffer) != 0:
        return None
    return ctypes.c_long.from_buffer(buffer).value & 0xFFFFFFFF


def require_tmpfs(path):
    """The run root must be local tmpfs by both the mount table and statfs(2)."""
    real = os.path.realpath(path)
    best = None
    for point, fstype in read_mounts():
        prefix = point.rstrip("/") + "/"
        if real == point or real.startswith(prefix) or point == "/":
            if best is None or len(point) >= len(best[0]):
                best = (point, fstype)
    magic = statfs_magic(real)
    if best is None or best[1] != "tmpfs" or magic != TMPFS_MAGIC:
        raise StageRefused(f"{path} must be on local tmpfs (mount {best}, statfs magic {magic})")
    return {"mount_point": best[0], "fstype": best[1], "statfs_magic": magic}


def open_dir(path, dir_fd=None):
    return os.open(path, os.O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC, dir_fd=dir_fd)


def hash_fd(descriptor):
    digest, size = hashlib.sha256(), 0
    while True:
        chunk = os.read(descriptor, BLOCK)
        if not chunk:
            break
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def read_small(path, limit=MAX_INPUT_BYTES):
    """Read a regular file without following a final symlink; bounded."""
    descriptor = os.open(path, os.O_RDONLY | O_NOFOLLOW | O_CLOEXEC)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise StageRefused(f"{path} is not a regular file")
        if info.st_size > limit:
            raise StageRefused(f"{path} is larger than {limit} bytes")
        chunks = []
        while True:
            chunk = os.read(descriptor, BLOCK)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def scan_source_tree(root):
    """{files: {rel: size}, symlinks: {rel: target}, dirs: [rel], other: [rel]} without following symlinks."""
    files, symlinks, dirs, other = {}, {}, [], []
    root_fd = open_dir(root)
    try:
        for current, dirnames, filenames, dirfd in os.fwalk(".", dir_fd=root_fd, follow_symlinks=False):
            relative = "" if current == "." else current[2:]
            if relative:
                dirs.append(relative)
            keep = []
            for name in sorted(set(dirnames) | set(filenames)):
                info = os.lstat(name, dir_fd=dirfd)
                path = f"{relative}/{name}" if relative else name
                if stat.S_ISLNK(info.st_mode):
                    symlinks[path] = os.readlink(name, dir_fd=dirfd)
                elif stat.S_ISDIR(info.st_mode):
                    keep.append(name)
                elif stat.S_ISREG(info.st_mode):
                    files[path] = info.st_size
                else:
                    other.append(path)
            dirnames[:] = [name for name in dirnames if name in keep]
    finally:
        os.close(root_fd)
    return {"files": files, "symlinks": symlinks, "dirs": sorted(dirs), "other": sorted(other)}


class StagedTree:
    """Writes one component under a fresh directory, by directory file descriptors only.

    Directory descriptors are cached in least-recently-used order and at most MAX_DIR_FDS stay open (the
    root's never closes): the checkouts component has every directory of every dispatch family's tree, more
    than the 1024 open files a systemd service gets by default. A later ``directory`` call may close a
    returned descriptor, so callers use it at once.
    """

    def __init__(self, parent_fd, name, uid, gid):
        self.uid, self.gid = uid, gid
        os.mkdir(name, 0o700, dir_fd=parent_fd)
        self.root_fd = open_dir(name, dir_fd=parent_fd)
        os.fchown(self.root_fd, uid, gid)
        os.fchmod(self.root_fd, 0o700)
        self.dir_fds = collections.OrderedDict([("", self.root_fd)])

    def _evict(self, keep):
        while len(self.dir_fds) > MAX_DIR_FDS:
            victim = next(key for key in self.dir_fds if key not in ("", keep))
            os.close(self.dir_fds.pop(victim))

    def directory(self, relative):
        if relative in self.dir_fds:
            self.dir_fds.move_to_end(relative)
            return self.dir_fds[relative]
        parent, _, name = relative.rpartition("/")
        parent_fd = self.directory(parent)
        try:
            os.mkdir(name, 0o700, dir_fd=parent_fd)
        except FileExistsError:
            pass
        descriptor = open_dir(name, dir_fd=parent_fd)
        os.fchown(descriptor, self.uid, self.gid)
        os.fchmod(descriptor, 0o700)
        self.dir_fds[relative] = descriptor
        self._evict(keep=relative)
        return descriptor

    def copy_file(self, relpath, source, expected_sha256):
        """Copy, hashing while copying; owner and mode 0444 are set before the name is complete."""
        parent, _, name = relpath.rpartition("/")
        directory = self.directory(parent)
        source_fd = os.open(source, os.O_RDONLY | O_NOFOLLOW | O_CLOEXEC)
        try:
            before = os.fstat(source_fd)
            if not stat.S_ISREG(before.st_mode):
                raise StageRefused(f"source of {relpath} is not a regular file")
            target_fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0o600,
                                dir_fd=directory)
            digest, size = hashlib.sha256(), 0
            try:
                while True:
                    chunk = os.read(source_fd, BLOCK)
                    if not chunk:
                        break
                    digest.update(chunk)
                    size += len(chunk)
                    view = memoryview(chunk)
                    while view:
                        view = view[os.write(target_fd, view):]
                os.fsync(target_fd)
                os.fchown(target_fd, self.uid, self.gid)
                os.fchmod(target_fd, 0o444)
            finally:
                os.close(target_fd)
            after = os.fstat(source_fd)
        finally:
            os.close(source_fd)
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or size != after.st_size:
            raise StageRefused(f"source of {relpath} changed while it was copied")
        if digest.hexdigest() != expected_sha256:
            raise StageRefused(f"{relpath}: sha256 {digest.hexdigest()} differs from its pin {expected_sha256}")

    def symlink(self, relpath, target):
        parent, _, name = relpath.rpartition("/")
        directory = self.directory(parent)
        os.symlink(target, name, dir_fd=directory)
        os.chown(name, self.uid, self.gid, dir_fd=directory, follow_symlinks=False)

    def close(self):
        for descriptor in self.dir_fds.values():
            try:
                os.close(descriptor)
            except OSError:
                pass
        self.dir_fds = collections.OrderedDict()


def snapshot_staged(parent_fd, name):
    """{rel: entry} of a staged tree, read without following symlinks; files re-hashed."""
    entries = {}
    root_fd = open_dir(name, dir_fd=parent_fd)
    try:
        info = os.fstat(root_fd)
        entries[""] = {"kind": "dir", "mode": stat.S_IMODE(info.st_mode), "uid": info.st_uid, "gid": info.st_gid}
        for current, dirnames, filenames, dirfd in os.fwalk(".", dir_fd=root_fd, follow_symlinks=False):
            relative = "" if current == "." else current[2:]
            descend = []
            for entry_name in sorted(set(dirnames) | set(filenames)):
                path = f"{relative}/{entry_name}" if relative else entry_name
                info = os.lstat(entry_name, dir_fd=dirfd)
                record = {"mode": stat.S_IMODE(info.st_mode), "uid": info.st_uid, "gid": info.st_gid}
                if stat.S_ISDIR(info.st_mode):
                    record["kind"] = "dir"
                    descend.append(entry_name)
                elif stat.S_ISLNK(info.st_mode):
                    record.update(kind="symlink", target=os.readlink(entry_name, dir_fd=dirfd))
                elif stat.S_ISREG(info.st_mode):
                    descriptor = os.open(entry_name, os.O_RDONLY | O_NOFOLLOW | O_CLOEXEC, dir_fd=dirfd)
                    try:
                        record["sha256"], record["bytes"] = hash_fd(descriptor)
                    finally:
                        os.close(descriptor)
                    record["kind"] = "file"
                else:
                    record["kind"] = "other"
                entries[path] = record
            dirnames[:] = [entry for entry in dirnames if entry in descend]
    finally:
        os.close(root_fd)
    return entries


def verify_staged(parent_fd, name, files, symlinks, directories, uid, gid):
    """Exact match: files (sha256, 0444, owner), symlinks, directories (0700, owner); nothing else."""
    problems = []
    entries = snapshot_staged(parent_fd, name)
    expected_dirs = {""} | implied_directories(list(files) + list(symlinks), directories)
    for path, entry in sorted(entries.items()):
        kind = entry["kind"]
        if kind == "dir":
            if path not in expected_dirs:
                problems.append(f"unexpected directory {path!r}")
            elif (entry["mode"], entry["uid"], entry["gid"]) != (0o700, uid, gid):
                problems.append(f"directory {path!r} mode {entry['mode']:o} owner {entry['uid']}:{entry['gid']}")
        elif kind == "file":
            if path not in files:
                problems.append(f"unexpected file {path!r}")
            elif entry["sha256"] != files[path]:
                problems.append(f"file {path!r} sha256 {entry['sha256']} != {files[path]}")
            elif (entry["mode"], entry["uid"], entry["gid"]) != (0o444, uid, gid):
                problems.append(f"file {path!r} mode {entry['mode']:o} owner {entry['uid']}:{entry['gid']}")
        elif kind == "symlink":
            if symlinks.get(path) != entry["target"] or (entry["uid"], entry["gid"]) != (uid, gid):
                problems.append(f"unexpected symlink {path!r} -> {entry['target']!r}")
        else:
            problems.append(f"special file {path!r}")
    for path in files:
        if entries.get(path, {}).get("kind") != "file":
            problems.append(f"missing file {path!r}")
    for path in symlinks:
        if entries.get(path, {}).get("kind") != "symlink":
            problems.append(f"missing symlink {path!r}")
    for path in expected_dirs:
        if entries.get(path, {}).get("kind") != "dir":
            problems.append(f"missing directory {path!r}")
    return problems


# ---------------------------------------------------------------------------
# Component plans (every source is validated before anything is written)
# ---------------------------------------------------------------------------

def plan_release(release_dir, expect_sums_sha256):
    sums_raw = read_small(os.path.join(release_dir, RELEASE_SUMS), limit=16 * 1024**2)
    sums_sha = sha256_bytes(sums_raw)
    if sums_sha != expect_sums_sha256:
        raise StageRefused(f"{RELEASE_SUMS} sha256 {sums_sha} differs from the pin {expect_sums_sha256}")
    entries = parse_sums(sums_raw.decode("utf-8"), RELEASE_SUMS)
    if RELEASE_SUMS in entries:
        raise StageRefused(f"{RELEASE_SUMS} must not list itself")
    for path in entries:
        refuse_truth_path(path, "release")
    scanned = scan_source_tree(release_dir)
    bytecode = sorted(path for path in scanned["files"]
                      if path.endswith(SKIP_SUFFIXES) and "__pycache__" in path.split("/"))
    problems = []
    if scanned["symlinks"]:
        problems.append("symlinks " + ", ".join(sorted(scanned["symlinks"])[:10]))
    if scanned["other"]:
        problems.append("special files " + ", ".join(scanned["other"][:10]))
    present = set(scanned["files"]) - {RELEASE_SUMS} - set(bytecode)
    if present - set(entries):
        problems.append("unlisted files " + ", ".join(sorted(present - set(entries))[:10]))
    if set(entries) - present:
        problems.append("missing files " + ", ".join(sorted(set(entries) - present)[:10]))
    large = sorted(path for path, size in scanned["files"].items() if size > MAX_RELEASE_FILE_BYTES)
    if large:
        problems.append("oversized files " + ", ".join(large[:10]))
    if problems:
        raise StageRefused(f"release tree does not match {RELEASE_SUMS}: " + "; ".join(problems))
    files = {path: (os.path.join(release_dir, path), sha) for path, sha in entries.items()}
    files[RELEASE_SUMS] = (os.path.join(release_dir, RELEASE_SUMS), sums_sha)
    return {"files": files, "symlinks": {}, "directories": [],
            "summary": {"source": release_dir, "sums_sha256": sums_sha, "files": len(entries),
                        "ignored_bytecode": len(bytecode),
                        "stager_in_release": entries.get("scripts/expA_stage.py"),
                        "publisher_in_release": entries.get("scripts/expA_publish.py")}}


def _manifest_entry(value, family, what, default):
    """A manifest entry is a sha256 string or {"sha256": ..., "path": ...} (expA_run.py _file_entries)."""
    if isinstance(value, str):
        sha, path = value, None
    elif isinstance(value, dict):
        sha, path = value.get("sha256"), value.get("path")
    else:
        raise StageRefused(f"dispatch manifest {what} entry for {family} is malformed")
    if not isinstance(sha, str) or not SHA256.fullmatch(sha):
        raise StageRefused(f"dispatch manifest {what} entry for {family} has no sha256")
    path = normalized_relpath(path, f"dispatch manifest {what} path of {family}") if path is not None else default
    if path != default:
        raise StageRefused(f"dispatch manifest {what} of {family} is {path!r}; the runtime layout needs {default!r}")
    return sha, refuse_truth_path(path, f"dispatch manifest {what}")


def validate_dispatch_row(row, number):
    if not isinstance(row, dict):
        raise StageRefused(f"dispatch row {number} is not an object")
    truth = find_truth_keys(row)
    if truth:
        raise StageRefused(f"dispatch row {number} carries truth keys {truth[:5]}; refusing (items.jsonl?)")
    extra = set(row) - DISPATCH_ROW_KEYS - DISPATCH_ROW_OPTIONAL
    missing = DISPATCH_ROW_KEYS - set(row)
    if extra or missing:
        raise StageRefused(f"dispatch row {number} keys differ from the dispatch view: extra {sorted(extra)}, "
                           f"missing {sorted(missing)}")
    if row.get("schema", DISPATCH_SCHEMA) != DISPATCH_SCHEMA:
        raise StageRefused(f"dispatch row {number} schema is not {DISPATCH_SCHEMA}")
    if not isinstance(row["item_id"], str) or not FAMILY.fullmatch(row["item_id"]):
        raise StageRefused(f"dispatch row {number} has an invalid item_id")
    if not isinstance(row["family"], str) or not FAMILY.fullmatch(row["family"]):
        raise StageRefused(f"dispatch row {number} has an invalid family")
    if not isinstance(row["target"], dict) or set(row["target"]) - TARGET_KEYS:
        raise StageRefused(f"dispatch row {number} target keys must be within {sorted(TARGET_KEYS)}")
    if not isinstance(row["package_dirs"], list) or not all(isinstance(v, str) for v in row["package_dirs"]):
        raise StageRefused(f"dispatch row {number} package_dirs must be a list of strings")
    if row["wording_pair"] not in ("A", "B"):
        raise StageRefused(f"dispatch row {number} wording_pair must be A or B")
    if (not isinstance(row["membership"], dict) or set(row["membership"]) != MEMBERSHIP_KEYS
            or any(type(value) is not bool for value in row["membership"].values())):
        raise StageRefused(f"dispatch row {number} membership must be exactly the boolean flags "
                           f"{sorted(MEMBERSHIP_KEYS)}")
    if not isinstance(row["cells"], dict):
        raise StageRefused(f"dispatch row {number} cells must be an object")
    shas = []
    for variant, conditions in row["cells"].items():
        if variant not in VARIANTS or not isinstance(conditions, dict):
            raise StageRefused(f"dispatch row {number} has an unknown variant {variant!r}")
        for condition, sha in conditions.items():
            if condition not in PAYLOAD_CONDITIONS:
                raise StageRefused(f"dispatch row {number} has an unknown condition {condition!r}")
            if sha is None:
                continue
            if not isinstance(sha, str) or not SHA256.fullmatch(sha):
                raise StageRefused(f"dispatch row {number} cell {variant}/{condition} is not a sha256")
            shas.append(sha)
    return shas


def validate_payload_store(data, family, needed):
    rows, present = 0, set()
    for number, raw in enumerate(data.splitlines(), 1):
        if not raw.strip():
            continue
        row = strict_json(raw, f"payload store {family} row {number}")
        if not isinstance(row, dict) or set(row) != PAYLOAD_ROW_KEYS:
            raise StageRefused(f"payload store {family} row {number} keys must be exactly {sorted(PAYLOAD_ROW_KEYS)}")
        encoded = canonical_json(row["payload"])
        if sha256_bytes(encoded) != row["sha256"] or row["bytes"] != len(encoded):
            raise StageRefused(f"payload store {family} row {number} does not match its sha256 or bytes")
        truth = find_truth_keys(row["payload"])
        if truth:
            raise StageRefused(f"payload store {family} row {number} carries truth keys {truth[:5]}")
        present.add(row["sha256"])
        rows += 1
    missing = sorted(set(needed) - present)
    if missing:
        raise StageRefused(f"payload store {family} lacks {len(missing)} payloads named by dispatch cells")
    unreferenced = present - set(needed)
    if unreferenced:
        raise StageRefused(f"payload store {family} holds {len(unreferenced)} payloads that no dispatch cell names; "
                           "the runtime host receives only the dispatched cells (D10.7)")
    return {"rows": rows, "referenced": len(set(needed)), "unreferenced": 0}


def validate_graph_facts(data, family, expected_source):
    document = strict_json(data, f"graph facts of {family}")
    if not isinstance(document, dict):
        raise StageRefused(f"graph facts of {family} must be a JSON object")
    truth = sorted(key for key in document if key in TRUTH_KEYS)
    if truth:
        raise StageRefused(f"graph facts of {family} carry truth keys {truth}")
    if document.get("variant") not in (None, "natural") or document.get("backend") not in (None, "locagent"):
        raise StageRefused(f"graph facts of {family} are not the natural LocAgent facts")
    if not isinstance(document.get("nodes"), list) or not isinstance(document.get("edges"), list):
        raise StageRefused(f"graph facts of {family} are not a GraphFacts document (nodes, edges)")
    if expected_source is not None and document.get("source_sha256") != expected_source:
        raise StageRefused(f"graph facts of {family} were built from {document.get('source_sha256')}, "
                           f"not the dispatch family's source {expected_source}")
    return {"nodes": len(document["nodes"]), "edges": len(document["edges"]),
            "source_sha256": document.get("source_sha256")}


def refuse_truth_files_in(directory, what):
    """A dispatch directory that holds items.jsonl or other truth files is the wrong directory."""
    found = sorted(path for path in scan_source_tree(directory)["files"] if is_truth_path(path))
    if found:
        raise StageRefused(f"{what} {directory} contains truth files {found[:5]}; refusing to stage from it "
                           "(the runtime host receives only the dispatch view)")


def plan_dispatch(dispatch_dir, expect_manifest_sha256, expect_dispatch_sha256=None):
    refuse_truth_files_in(dispatch_dir, "dispatch directory")
    manifest_raw = read_small(os.path.join(dispatch_dir, DISPATCH_MANIFEST), limit=64 * 1024**2)
    manifest_sha = sha256_bytes(manifest_raw)
    if manifest_sha != expect_manifest_sha256:
        raise StageRefused(f"{DISPATCH_MANIFEST} sha256 {manifest_sha} differs from the pin {expect_manifest_sha256}")
    manifest = strict_json(manifest_raw, DISPATCH_MANIFEST)
    if not isinstance(manifest, dict) or manifest.get("schema") != DISPATCH_SCHEMA:
        raise StageRefused(f"{DISPATCH_MANIFEST} must be an object with schema {DISPATCH_SCHEMA}")
    truth = find_truth_keys({key: value for key, value in manifest.items() if key != "code"})
    if truth:
        raise StageRefused(f"{DISPATCH_MANIFEST} carries truth keys {truth[:5]}")
    pins = {name: manifest.get(name) for name in ("dispatch_jsonl_sha256", "dispatch_sha256")
            if manifest.get(name) is not None}
    if expect_dispatch_sha256:
        pins["operator"] = expect_dispatch_sha256
    if not pins:
        raise StageRefused("dispatch.jsonl is not pinned: the manifest has no dispatch_jsonl_sha256 and no "
                           "--expect-dispatch-sha256 was given")
    dispatch_raw = read_small(os.path.join(dispatch_dir, DISPATCH_FILE))
    dispatch_sha = sha256_bytes(dispatch_raw)
    if any(value != dispatch_sha for value in pins.values()):
        raise StageRefused(f"{DISPATCH_FILE} sha256 {dispatch_sha} differs from its pin(s) {pins}")
    needed, families, items, in_t, rows = {}, set(), set(), set(), 0
    for number, raw in enumerate(dispatch_raw.decode("utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        row = strict_json(raw, f"dispatch row {number}")
        shas = validate_dispatch_row(row, number)
        if row["item_id"] in items:
            raise StageRefused(f"dispatch item {row['item_id']} appears twice")
        items.add(row["item_id"])
        families.add(row["family"])
        needed.setdefault(row["family"], set()).update(shas)
        if row["membership"].get("in_t"):
            in_t.add(row["family"])
        rows += 1
    if not rows:
        raise StageRefused("dispatch.jsonl has no rows")
    if manifest.get("rows") is not None and manifest["rows"] != rows:
        raise StageRefused(f"{DISPATCH_MANIFEST} says {manifest['rows']} rows, dispatch.jsonl has {rows}")
    files = {DISPATCH_FILE: (os.path.join(dispatch_dir, DISPATCH_FILE), dispatch_sha),
             DISPATCH_MANIFEST: (os.path.join(dispatch_dir, DISPATCH_MANIFEST), manifest_sha)}
    payload_map, facts_map = manifest.get("payload_store"), manifest.get("graph_facts") or {}
    family_info = manifest.get("families") if isinstance(manifest.get("families"), dict) else {}
    if not isinstance(payload_map, dict) or not isinstance(facts_map, dict):
        raise StageRefused("dispatch manifest payload_store and graph_facts must map family -> file entry")
    sources = {}
    for family in sorted(families):
        info = family_info.get(family) if isinstance(family_info.get(family), dict) else {}
        source = info.get("source_sha256")
        if source is not None and not (isinstance(source, str) and SHA256.fullmatch(source)):
            raise StageRefused(f"dispatch manifest source_sha256 of {family} is not a sha256")
        sources[family] = source
    payloads, facts = {}, {}
    for family in sorted(families):
        if family not in payload_map:
            raise StageRefused(f"dispatch manifest pins no payload store for {family}")
        sha, path = _manifest_entry(payload_map[family], family, "payload_store", f"payloads/{family}.jsonl")
        data = read_small(os.path.join(dispatch_dir, path))
        if sha256_bytes(data) != sha:
            raise StageRefused(f"payload store {path} does not match its pinned sha256")
        payloads[family] = dict(validate_payload_store(data, family, needed.get(family, ())), path=path, sha256=sha)
        files[path] = (os.path.join(dispatch_dir, path), sha)
    for family in sorted(set(facts_map) | in_t):
        if family not in families:
            raise StageRefused(f"dispatch manifest pins graph facts for {family}, which has no dispatch rows")
        if family not in facts_map:
            raise StageRefused(f"family {family} has T-arm items but the manifest pins no natural graph facts")
        sha, path = _manifest_entry(facts_map[family], family, "graph_facts", f"graph_facts/{family}.json")
        data = read_small(os.path.join(dispatch_dir, path))
        if sha256_bytes(data) != sha:
            raise StageRefused(f"graph facts {path} do not match their pinned sha256")
        facts[family] = dict(validate_graph_facts(data, family, sources.get(family)), path=path, sha256=sha)
        files[path] = (os.path.join(dispatch_dir, path), sha)
    summary = {"source": dispatch_dir, "manifest_sha256": manifest_sha, "dispatch_sha256": dispatch_sha,
               "dispatch_sha256_pinned_by": sorted(pins), "schema": manifest.get("schema"), "rows": rows,
               "families": sorted(families), "t_arm_families": sorted(in_t), "payload_store": payloads,
               "graph_facts": facts, "family_sources": sources, "items_sha256": manifest.get("items_sha256"),
               "items_manifest_sha256": manifest.get("manifest_sha256"),
               "code": manifest.get("code") if isinstance(manifest.get("code"), dict) else None}
    return {"files": files, "symlinks": {}, "directories": [], "summary": summary}


def plan_checkouts(manifest_path, expect_manifest_sha256, families, family_sources, checkouts_dir=None):
    """inputs/checkouts/: the pinned clean checkout manifest and, for each dispatch family, exactly its tree."""
    raw = read_small(manifest_path, limit=MAX_INPUT_BYTES)
    manifest_sha = sha256_bytes(raw)
    if manifest_sha != expect_manifest_sha256:
        raise StageRefused(f"clean checkout manifest sha256 {manifest_sha} differs from the pin {expect_manifest_sha256}")
    manifest = strict_json(raw, "clean checkout manifest")
    if (not isinstance(manifest, dict) or manifest.get("schema") != CHECKOUTS_SCHEMA
            or manifest.get("tree_protocol") != TREE_PROTOCOL or not isinstance(manifest.get("families"), dict)):
        raise StageRefused(f"the clean checkout manifest must be {CHECKOUTS_SCHEMA} with tree_protocol {TREE_PROTOCOL}")
    files = {CHECKOUTS_MANIFEST: (manifest_path, manifest_sha)}
    symlinks, directories, summary = {}, ["repos"], {}
    for family in sorted(families):
        entry = manifest["families"].get(family)
        if not isinstance(entry, dict) or entry.get("family", family) != family:
            raise StageRefused(f"the clean checkout manifest has no entry for dispatch family {family}")
        listed_files, listed_links, listed_dirs = entry.get("files"), entry.get("symlinks"), entry.get("directories")
        if not isinstance(listed_files, dict) or not isinstance(listed_links, dict) or not isinstance(listed_dirs, list):
            raise StageRefused(f"checkout {family}: the manifest entry needs files, symlinks and directories")
        for path, sha in listed_files.items():
            normalized_relpath(path, f"checkout {family}")
            if not isinstance(sha, str) or not SHA256.fullmatch(sha):
                raise StageRefused(f"checkout {family}: {path} has no sha256")
        for path in list(listed_links) + list(listed_dirs):
            normalized_relpath(path, f"checkout {family}")
        if listed_dirs != sorted(set(listed_dirs)):
            raise StageRefused(f"checkout {family}: directories must be sorted and unique")
        for path, target in listed_links.items():
            if not symlink_stays_inside(path, target):
                raise StageRefused(f"checkout {family}: symlink {path} -> {target!r} leaves the checkout")
        if entry.get("skipped") or entry.get("escaping_symlinks"):
            raise StageRefused(f"checkout {family} is not clean: skipped entries or escaping symlinks are listed")
        for path in list(listed_files) + list(listed_links) + listed_dirs:
            if any(part in SKIP_DIRS for part in path.split("/")) or path.endswith(SKIP_SUFFIXES):
                raise StageRefused(f"checkout {family} lists {path}, which is never part of a clean checkout")
        if not implied_directories(list(listed_files) + list(listed_links)) <= set(listed_dirs):
            raise StageRefused(f"checkout {family}: directories do not cover every file and symlink parent")
        digest = tree_digest(listed_files, listed_links, listed_dirs)
        if digest != entry.get("tree_sha256"):
            raise StageRefused(f"checkout {family}: tree_sha256 does not match its file, symlink and directory lists")
        package_files = entry.get("package_files")
        if not isinstance(package_files, list) or set(package_files) - set(listed_files):
            raise StageRefused(f"checkout {family}: package files missing from the file list")
        package = package_tree_sha256(listed_files, package_files)
        if package != entry.get("package_tree_sha256"):
            raise StageRefused(f"checkout {family}: package_tree_sha256 does not match its package files")
        if family_sources.get(family) is not None and family_sources[family] != package:
            raise StageRefused(f"clean checkout of {family} has package tree {package}, but its payloads were "
                               f"rendered from {family_sources[family]}")
        source = os.path.join(checkouts_dir, family) if checkouts_dir else entry.get("path")
        if not isinstance(source, str) or not source.startswith("/"):
            raise StageRefused(f"checkout {family}: no absolute source path (give --checkouts-dir)")
        scanned = scan_source_tree(source)
        problems = []
        if scanned["other"]:
            problems.append("special files " + ", ".join(scanned["other"][:5]))
        if set(scanned["files"]) != set(listed_files):
            extra = sorted(set(scanned["files"]) - set(listed_files))
            missing = sorted(set(listed_files) - set(scanned["files"]))
            problems.append(f"file list differs (extra {extra[:5]}, missing {missing[:5]})")
        if scanned["symlinks"] != listed_links:
            problems.append("symlinks differ")
        if scanned["dirs"] != listed_dirs:
            problems.append(f"directories differ: {sorted(set(scanned['dirs']) ^ set(listed_dirs))[:5]}")
        if problems:
            raise StageRefused(f"checkout {source} is not exactly its manifest entry: " + "; ".join(problems))
        for path, sha in listed_files.items():
            files[f"repos/{family}/{path}"] = (os.path.join(source, path), sha)
        for path, target in listed_links.items():
            symlinks[f"repos/{family}/{path}"] = target
        directories += [f"repos/{family}"] + [f"repos/{family}/{path}" for path in listed_dirs]
        summary[family] = {"source": source, "tree_sha256": digest, "files": len(listed_files),
                           "symlinks": len(listed_links), "directories": len(listed_dirs),
                           "package_tree_sha256": package, "commit": entry.get("commit")}
    return {"files": files, "symlinks": symlinks, "directories": directories,
            "summary": {"manifest": manifest_path, "manifest_sha256": manifest_sha,
                        "families_in_manifest": sorted(manifest["families"]), "staged_families": summary}}


# ---------------------------------------------------------------------------
# Run state: published Core2 records vs the local run directory (reboot safety)
# ---------------------------------------------------------------------------

def run_target_path(path, allowed_parents=RUN_TARGET_PARENTS):
    normalized = os.path.normpath(path or "")
    if os.path.dirname(normalized) not in allowed_parents or not RUN_ID.fullmatch(os.path.basename(normalized)):
        raise StageRefused(f"--run-target must be <runs parent>/<run id> under one of {list(allowed_parents)}")
    return normalized


def _regular_entries(path):
    try:
        with os.scandir(path) as iterator:
            entries = sorted(iterator, key=lambda entry: entry.name)
    except (FileNotFoundError, NotADirectoryError):
        return [], []
    files = [entry for entry in entries if not entry.name.startswith(".") and entry.is_file(follow_symlinks=False)]
    dirs = [entry for entry in entries if not entry.name.startswith(".") and entry.is_dir(follow_symlinks=False)]
    return files, dirs


def published_run_records(target):
    """{relpath: size} of the run-directory records in a Core2 run dir (publish/, serve/, stage/, logs/ excluded)."""
    found = {}
    files, _ = _regular_entries(target)
    for entry in files:
        if entry.name in RUN_RECORD_FILES:
            found[entry.name] = entry.stat(follow_symlinks=False).st_size
    for directory in RUN_RECORD_DIRS:
        for entry in _regular_entries(os.path.join(target, directory))[0]:
            if entry.name.endswith(".json"):
                found[f"{directory}/{entry.name}"] = entry.stat(follow_symlinks=False).st_size
    for cell in _regular_entries(os.path.join(target, "cells"))[1]:
        for attempt in _regular_entries(cell.path)[1]:
            for entry in _regular_entries(attempt.path)[0]:
                if entry.name in ATTEMPT_FILES:
                    found[f"cells/{cell.name}/{attempt.name}/{entry.name}"] = entry.stat(follow_symlinks=False).st_size
    return found


def _file_sha256(path):
    descriptor = os.open(path, os.O_RDONLY | O_NOFOLLOW | O_CLOEXEC)
    try:
        return hash_fd(descriptor)[0]
    finally:
        os.close(descriptor)


def interrupted_attempts(records):
    """Attempts with a start record but neither a result nor a closing record (lost with the run root)."""
    attempts = {}
    for relpath in records:
        parts = relpath.split("/")
        if len(parts) == 4 and parts[0] == "cells":
            attempts.setdefault("/".join(parts[:3]), set()).add(parts[3])
    return sorted(attempt for attempt, names in attempts.items()
                  if "attempt_start.json" in names and not {"result.json", "attempt_closed.json"} & names)


def plan_run_state(run_root, run_target, restore):
    """(summary, restore plan or None); refuses an empty or diverging run directory when Core2 holds records.

    ``run_target`` was validated by ``run_target_path`` (stage() does it).
    """
    target = os.path.normpath(run_target)
    records = published_run_records(target)
    summary = {"target": target, "published_records": len(records), "run_json_published": "run.json" in records,
               "restore_requested": bool(restore)}
    if "run.json" not in records:
        if records:
            raise StageRefused(f"{target} holds {len(records)} attempt records but no run.json; inspect it by hand")
        return dict(summary, state="empty"), None
    run_dir = os.path.join(run_root, "run")
    local = scan_source_tree(run_dir) if os.path.isdir(run_dir) else {"files": {}, "symlinks": {}, "dirs": [],
                                                                        "other": []}
    if "run.json" not in local["files"]:
        if local["files"] or local["symlinks"] or local["other"] or local["dirs"]:
            raise StageRefused(f"{run_dir} has entries but no run.json; it is neither a fresh nor a restored run dir")
        if not restore:
            raise StageRefused(f"{target} already holds {len(records)} published records of this run, but {run_dir} is "
                               "empty (a reboot wiped /dev/shm?). A relaunch now would dispatch completed cells again: "
                               "restore the records first with `stage --restore`, then relaunch")
        files = {relpath: (os.path.join(target, relpath), _file_sha256(os.path.join(target, relpath)))
                 for relpath in sorted(records)}
        lost = interrupted_attempts(records)
        return (dict(summary, state="restore", restored_files=len(files),
                     records_sha256=sha256_bytes(canonical_json({path: sha for path, (_, sha) in files.items()})),
                     interrupted_attempts=lost, interrupted_attempt_count=len(lost)),
                {"files": files, "directories": sorted(implied_directories(list(files)))})
    if restore:
        raise StageRefused(f"{run_dir} already holds a run directory; --restore applies only to an empty one")
    problems = []
    for relpath, size in sorted(records.items()):
        if local["files"].get(relpath) != size:
            problems.append(f"{relpath}: {'missing' if relpath not in local['files'] else 'size differs'}")
    for relpath in RUN_RECORD_FILES:
        if relpath in records and relpath in local["files"] and not problems and \
                _file_sha256(os.path.join(target, relpath)) != _file_sha256(os.path.join(run_dir, relpath)):
            problems.append(f"{relpath}: sha256 differs")
    if problems:
        raise StageRefused(f"{run_dir} does not hold every record published to {target}: {problems[:5]} "
                           f"({len(problems)} problems)")
    return dict(summary, state="consistent", local_files=len(local["files"])), None


def apply_run_state(run_root, plan, owner):
    """Copy the published run records into the empty run/ (temp dir, verify, rename over run/)."""
    uid, gid = owner["uid"], owner["gid"]
    root_fd = open_dir(run_root)
    temporary = f".restoring-run-{uuid.uuid4().hex[:12]}"
    try:
        files = {path: sha for path, (_, sha) in plan["files"].items()}
        tree = StagedTree(root_fd, temporary, uid, gid)
        try:
            for directory in plan["directories"]:
                tree.directory(directory)
            for path in sorted(files):
                tree.copy_file(path, plan["files"][path][0], files[path])
        finally:
            tree.close()
        problems = verify_staged(root_fd, temporary, files, {}, plan["directories"], uid, gid)
        if problems:
            raise StageRefused("the restored run directory failed verification: " + "; ".join(problems[:5]))
        try:
            os.rmdir("run", dir_fd=root_fd)  # the empty run/ that prepare created
        except FileNotFoundError:
            pass
        os.rename(temporary, "run", src_dir_fd=root_fd, dst_dir_fd=root_fd)
    finally:
        os.close(root_fd)
    return {"files": len(plan["files"]), "directories": len(plan["directories"])}


# ---------------------------------------------------------------------------
# Staging
# ---------------------------------------------------------------------------

def resolve_owner(name, host=None):
    host = (host or socket.gethostname()).split(".")[0]
    try:
        entry = pwd.getpwnam(name)
    except KeyError as error:
        raise StageRefused(f"owner {name!r} does not exist on {host}") from error
    if entry.pw_uid == 0:
        raise StageRefused("staged inputs are never owned by root")
    if entry.pw_uid in FORBIDDEN_UIDS_BY_HOST.get(host, ()):
        raise StageRefused(f"UID {entry.pw_uid} belongs to another person on {host}")
    expected = OWNER_UID_BY_HOST.get(host) if name == EXPECTED_OWNER else None
    if expected is not None and entry.pw_uid != expected:
        raise StageRefused(f"{name} must be UID {expected} on {host}, found {entry.pw_uid}")
    return {"user": name, "uid": entry.pw_uid, "gid": entry.pw_gid, "host": host}


def check_run_root(run_root, uid, fs_check=require_tmpfs, run_parent=RUN_PARENT):
    if not os.path.isabs(run_root) or not RUN_ROOT_NAME.fullmatch(os.path.basename(run_root.rstrip("/"))):
        raise StageRefused("the run root must be an absolute path named ae-expA-<run>")
    if run_parent is not None and os.path.dirname(run_root.rstrip("/")) != run_parent:
        raise StageRefused(f"the run root must be directly under {run_parent}")
    info = os.lstat(run_root)
    if not stat.S_ISDIR(info.st_mode):
        raise StageRefused(f"{run_root} must be a real directory (not a symlink)")
    if info.st_uid != uid or stat.S_IMODE(info.st_mode) != 0o700:
        raise StageRefused(f"{run_root} must be mode 0700 and owned by UID {uid} (create it as the run owner)")
    return fs_check(run_root)


def write_receipt(inputs_fd, name, receipt, uid, gid):
    data = canonical_json(receipt) + b"\n"
    temporary = f".{name}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0o600,
                         dir_fd=inputs_fd)
    try:
        view = memoryview(data)
        while view:
            view = view[os.write(descriptor, view):]
        os.fsync(descriptor)
        os.fchown(descriptor, uid, gid)
        os.fchmod(descriptor, 0o444)
        os.link(temporary, name, src_dir_fd=inputs_fd, dst_dir_fd=inputs_fd)
    finally:
        os.close(descriptor)
        os.unlink(temporary, dir_fd=inputs_fd)
    return {"path": name, "sha256": sha256_bytes(data)}


def stage_component(inputs_fd, component, plan, owner, context):
    """Stage one component atomically (temp dir, verify, rename); verify instead when it exists."""
    uid, gid = owner["uid"], owner["gid"]
    files = {path: sha for path, (_, sha) in plan["files"].items()}
    refuse_truth_paths(component, list(files) + list(plan["symlinks"]))
    try:
        info = os.lstat(component, dir_fd=inputs_fd)
    except FileNotFoundError:
        info = None
    if info is not None:
        if not stat.S_ISDIR(info.st_mode):
            raise StageRefused(f"inputs/{component} exists and is not a directory")
        problems = verify_staged(inputs_fd, component, files, plan["symlinks"], plan["directories"], uid, gid)
        if problems:
            raise StageRefused(f"inputs/{component} is already staged with different content: " + "; ".join(problems[:5]))
        result = "already_staged_verified"
    else:
        temporary = f".staging-{component}-{uuid.uuid4().hex[:12]}"
        tree = StagedTree(inputs_fd, temporary, uid, gid)
        try:
            for directory in sorted(plan["directories"]):
                tree.directory(directory)
            for path in sorted(files):
                tree.copy_file(path, plan["files"][path][0], files[path])
            for path, target in sorted(plan["symlinks"].items()):
                tree.symlink(path, target)
        finally:
            tree.close()
        problems = verify_staged(inputs_fd, temporary, files, plan["symlinks"], plan["directories"], uid, gid)
        if problems:
            raise StageRefused(f"the staged copy of {component} failed verification: " + "; ".join(problems[:5]))
        os.rename(temporary, component, src_dir_fd=inputs_fd, dst_dir_fd=inputs_fd)
        result = "staged"
    receipt = {"schema": RECEIPT_SCHEMA, "protocol": STAGE_PROTOCOL, "component": component, "result": result,
               "created_utc": utc_now(), "context": context, "owner": owner, "summary": plan["summary"],
               "files": dict(sorted(files.items())), "symlinks": dict(sorted(plan["symlinks"].items())),
               "directories": sorted(plan["directories"])}
    name = (f"STAGE_RECEIPT.{component}.json" if result == "staged"
            else f"STAGE_VERIFY.{component}.{utc_stamp()}.json")
    receipt_file = write_receipt(inputs_fd, name, receipt, uid, gid)
    return {"result": result, "receipt": receipt_file, "files": len(files), "symlinks": len(plan["symlinks"])}


def open_inputs(run_root, owner):
    root_fd = open_dir(run_root)
    try:
        try:
            os.mkdir("inputs", 0o700, dir_fd=root_fd)
            created = True
        except FileExistsError:
            created = False
        inputs_fd = open_dir("inputs", dir_fd=root_fd)
    finally:
        os.close(root_fd)
    if created:
        os.fchown(inputs_fd, owner["uid"], owner["gid"])
        os.fchmod(inputs_fd, 0o700)
    info = os.fstat(inputs_fd)
    if (info.st_uid, stat.S_IMODE(info.st_mode)) != (owner["uid"], 0o700):
        os.close(inputs_fd)
        raise StageRefused("inputs/ must be mode 0700 and owned by the run owner")
    return inputs_fd


def self_identity():
    loaded = globals().get("__loader_sha256__")
    path = os.path.abspath(globals().get("__file__") or sys.argv[0])
    if loaded is None:
        try:
            with open(path, "rb") as stream:
                loaded = sha256_bytes(stream.read())
        except OSError:
            loaded = None
    return {"path": path, "sha256": loaded, "verified_by_loader": "__loader_sha256__" in globals(),
            "python": sys.version.split()[0], "executable": sys.executable, "euid": os.geteuid(),
            "host": socket.gethostname()}


def requested_components(text):
    names = [value.strip() for value in text.split(",") if value.strip()]
    if not names or set(names) - set(COMPONENTS) - {RUN_STATE}:
        raise StageRefused(f"components must be within {list(COMPONENTS) + [RUN_STATE]}")
    return [component for component in COMPONENTS + (RUN_STATE,) if component in names]


def stage(args, fs_check=require_tmpfs, host=None, run_parent=RUN_PARENT, run_target_parents=RUN_TARGET_PARENTS):
    owner = resolve_owner(args.owner, host=host)
    filesystem = check_run_root(args.run_root, owner["uid"], fs_check=fs_check, run_parent=run_parent)
    components = requested_components(args.components)
    run_state, restore_plan = None, None
    if RUN_STATE in components:  # checked before anything is staged: a refusal stages nothing
        if not args.run_target:
            raise StageRefused("the run-state component needs --run-target (the Core2 run directory of this run id)")
        run_state, restore_plan = plan_run_state(args.run_root, run_target_path(args.run_target, run_target_parents),
                                                 args.restore)
    elif args.restore:
        raise StageRefused("--restore needs the run-state component")
    context = {"run_root": args.run_root, "filesystem": filesystem, "stager": self_identity(),
               "pins": {"release_sums_sha256": args.expect_release_sums_sha256,
                        "dispatch_manifest_sha256": args.expect_dispatch_manifest_sha256,
                        "dispatch_sha256": args.expect_dispatch_sha256,
                        "checkouts_manifest_sha256": args.expect_checkouts_manifest_sha256}}
    plans, dispatch_plan = {}, None
    if {"dispatch", "checkouts"} & set(components):
        if not args.dispatch_dir or not args.expect_dispatch_manifest_sha256:
            raise StageRefused("the dispatch and checkouts components need --dispatch-dir and "
                               "--expect-dispatch-manifest-sha256")
        dispatch_plan = plan_dispatch(args.dispatch_dir, args.expect_dispatch_manifest_sha256,
                                      args.expect_dispatch_sha256)
    if "release" in components:
        if not args.release_dir or not args.expect_release_sums_sha256:
            raise StageRefused("the release component needs --release-dir and --expect-release-sums-sha256")
        plans["release"] = plan_release(args.release_dir, args.expect_release_sums_sha256)
    if "dispatch" in components:
        plans["dispatch"] = dispatch_plan
    if "checkouts" in components:
        if not args.checkouts_manifest or not args.expect_checkouts_manifest_sha256:
            raise StageRefused("the checkouts component needs --checkouts-manifest and "
                               "--expect-checkouts-manifest-sha256")
        plans["checkouts"] = plan_checkouts(args.checkouts_manifest, args.expect_checkouts_manifest_sha256,
                                            dispatch_plan["summary"]["families"],
                                            dispatch_plan["summary"]["family_sources"], args.checkouts_dir)
    inputs_fd = open_inputs(args.run_root, owner)
    results = {}
    try:
        for component in COMPONENTS:
            if component in plans:
                results[component] = stage_component(inputs_fd, component, plans[component], owner, context)
        if run_state is not None:
            if restore_plan is not None:
                run_state = dict(run_state, state="restored", restored=apply_run_state(args.run_root, restore_plan,
                                                                                        owner))
            receipt = {"schema": RECEIPT_SCHEMA, "protocol": STAGE_PROTOCOL, "component": RUN_STATE,
                       "result": run_state["state"], "created_utc": utc_now(), "context": context, "owner": owner,
                       "summary": run_state}
            results[RUN_STATE] = {"result": run_state["state"],
                                  "receipt": write_receipt(inputs_fd, f"STAGE_RECEIPT.{RUN_STATE}.{utc_stamp()}.json",
                                                           receipt, owner["uid"], owner["gid"]),
                                  "summary": run_state}
    finally:
        os.close(inputs_fd)
    return {"protocol": STAGE_PROTOCOL, "status": "staged", "run_root": args.run_root, "owner": owner,
            "components": results,
            "runner_arguments": {"--dispatch": os.path.join(args.run_root, "inputs", "dispatch"),
                                 "--checkout-manifest": os.path.join(args.run_root, "inputs", "checkouts",
                                                                     CHECKOUTS_MANIFEST),
                                 "--repos-root": os.path.join(args.run_root, "inputs", "checkouts", "repos")}}


def _sha_arg(text):
    if not SHA256.fullmatch(text or ""):
        raise argparse.ArgumentTypeError("expected 64 lowercase hex digits")
    return text


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-root", required=True, help="/dev/shm/ae-expA-<run> (0700, owned by --owner, tmpfs)")
    parser.add_argument("--owner", default=EXPECTED_OWNER)
    parser.add_argument("--components", default="release,dispatch,checkouts",
                        help="comma list of release, dispatch, checkouts, run-state")
    parser.add_argument("--run-target", default=None,
                        help="run-state: the Core2 run directory of this run id (expA/runs/<run id>)")
    parser.add_argument("--restore", action="store_true",
                        help="run-state: copy the published run records into the empty run/ (after a reboot)")
    parser.add_argument("--release-dir", help="Core2 expA/releases/<git-sha>/ containing RELEASE_FILES.sha256")
    parser.add_argument("--expect-release-sums-sha256", type=_sha_arg)
    parser.add_argument("--dispatch-dir", help="Core2 dispatch view: dispatch.jsonl, dispatch_manifest.json, ...")
    parser.add_argument("--expect-dispatch-manifest-sha256", type=_sha_arg)
    parser.add_argument("--expect-dispatch-sha256", type=_sha_arg, default=None)
    parser.add_argument("--checkouts-manifest", help="clean checkout manifest (expA-clean-checkouts-v1) written by "
                                                     "scripts/expA_mirror_repos.py")
    parser.add_argument("--expect-checkouts-manifest-sha256", type=_sha_arg)
    parser.add_argument("--checkouts-dir", default=None,
                        help="clean checkouts <dir>/<family>/ (default: each manifest entry's path)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        result = stage(args)
    except (StageRefused, OSError) as error:
        print(json.dumps({"protocol": STAGE_PROTOCOL, "status": "refused",
                          "reason": f"{type(error).__name__}: {error}"}, sort_keys=True))
        return 2
    print(json.dumps(result, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
