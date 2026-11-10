"""Exp A root publisher: Core1 /dev/shm run records -> Core2 durable run directory over NFS
(docs/EXPA_CORE1_RUNTIME.md placement 3, docs/EXPA_RUNBOOK.md).

The runner (thilak.arjun, Core1 UID 1008) writes its immutable records under /dev/shm/ae-expA-<run>/run/,
because Core1's account cannot write the 0700 private project directory on Core2. This publisher runs as
root (the NFS export has no_root_squash) inside a hardened transient unit (ProtectSystem=strict,
ReadWritePaths= only the target run directory, ReadOnlyPaths= the source run root, NoNewPrivileges, a small
capability set; see ``plan``) and copies each completed immutable record to
/mnt/kldata/absent-edges-20261018-core2/expA/runs/<run-id>/, preserving the run-directory layout:

  run.json, cells.jsonl, launches/*.json, snapshots/*.json                    (immutable, mode 0444)
  cells/<cell_id>/<attempt_id>/attempt_start.json, result.json, attempt_closed.json   (immutable)
  cells/<cell_id>/<attempt_id>/events.jsonl   once closed (0444) or once the attempt is closed by recovery;
                                              always before its result.json
  serve/<instance>/records/*.json             the serving supervisors' records (collisions included)
  stage/<sha256[:16]>/STAGE_*.json            the stager's receipts, content-addressed: a re-stage after a
                                              reboot writes new receipts under the same names
  logs/, serve/<instance>/logs/               only by ``once --final`` (the -final unit), after the runner AND
                                              every serve unit have stopped: a log that still grows would
                                              conflict with its published prefix

Every copy is hashed while it is written, then re-read from the target and its sha256 compared. Files are
created under a temporary name and published with link(2), so a name never shows partial content and an
existing name is never replaced. A target file that differs from its source is never overwritten: the
conflict is recorded instead. An attempt's events.jsonl, result.json and attempt_closed.json are published
only after that attempt's attempt_start.json was published or found identical on the target; when the target
holds a different start record (another launch reused the attempt id, for example after a reboot wiped the
run root), they are recorded as conflicts and never mixed into that attempt. Published files are chowned to
Core2 thilak.arjun (1006:1007) and made 0444; directories are 0700. The publish ledger is a directory of
immutable records in the target, publish/: started-*.json, passes/*.json (every pass that published
something, waited on an error or found a problem), conflicts/*.json and finished-*.json (with every path
still waiting at the end).

``loop`` publishes every --interval seconds and exits after a last records pass when the stop file
(<run root>/control/publish.stop by default) appears, when every cell of cells.jsonl has a completed
result, or on SIGTERM; it never publishes logs. A file that cannot be read or written (EIO, ESTALE, ...)
is recorded and retried in the next pass, never fatal. ``once --final`` exits 4 on conflicts and 5 when
any record is still waiting (for example an attempt whose runner was killed: close it first with
``expA_run.py reconcile``). ``init`` refuses a target that already holds records of this run when the
source run root has none (a reboot wiped /dev/shm: restore them with the stager first). Exit 2 means the
publisher refused (identity, source or target); 3 a transient OS error at the loop level (the loop unit
restarts on it). No timers. Standard library only, Python 3.8+ (Core1 /usr/bin/python3 is 3.10).
"""
from __future__ import annotations

import argparse
import collections
import datetime
import errno
import hashlib
import json
import os
import re
import shlex
import signal
import socket
import stat
import sys
import time
import uuid

PUBLISH_PROTOCOL = "expA-publish-v1"
LEDGER_SCHEMA = "expA-publish-ledger-v1"
TARGET_UID, TARGET_GID = 1006, 1007                 # Core2 thilak.arjun:thilak.arjun (UIDs swap between hosts)
SOURCE_OWNER_UID_BY_HOST = {"ai-core-team-1": 1008}  # Core1 thilak.arjun
PRIVATE_ROOT = "/mnt/kldata/absent-edges-20261018-core2"
ALLOWED_TARGET_PARENTS = (PRIVATE_ROOT + "/expA/runs", PRIVATE_ROOT + "/expA/dev/v2/runtime/publish-test")
RUN_PARENT = "/dev/shm"
TMPFS_MAGIC = 0x01021994
RUN_ROOT_NAME = re.compile(r"ae-expA-[a-z0-9][a-z0-9-]{0,63}")
RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
NAME = re.compile(r"[^/\0]{1,255}")
COMPLETED_EXITS = frozenset({"Submitted", "NoAnswer", "ContextLimit", "FormatExhausted"})  # ledger.COMPLETED_EXITS
ATTEMPT_FILES = ("attempt_start.json", "events.jsonl", "result.json", "attempt_closed.json")
INACCESSIBLE_PATHS = ("/data5t", "/workspace", "/mnt/eu-ai-datasets")
CAPABILITIES = ("CAP_CHOWN", "CAP_DAC_OVERRIDE", "CAP_DAC_READ_SEARCH", "CAP_FOWNER")
SYSTEM_PYTHON = "/usr/bin/python3"
# The sha256 loader: executes exactly the bytes whose sha256 the coordinator pinned (no copy, no TOCTOU).
# No '$' or '%' (systemd would expand them in ExecStart). scripts/expA_core1_launch.sh carries the same text.
LOADER = ("import hashlib,sys;p,h=sys.argv[1],sys.argv[2];b=open(p,'rb').read();d=hashlib.sha256(b).hexdigest();"
          "d==h or sys.exit('ae-expA loader: sha256 '+d+' of '+p+' is not the pinned '+h);sys.argv=[p]+sys.argv[3:];"
          "exec(compile(b,p,'exec'),{'__name__':'__main__','__file__':p,'__loader_sha256__':d,"
          "'__builtins__':__builtins__})")
O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
BLOCK = 1024 * 1024
MAX_DIR_FDS = 64  # open directory descriptors kept by Target (LRU); the root's is never closed


class PublishRefused(RuntimeError):
    """The source, the target or the identity is not what the publisher may touch."""


def canonical_json(value) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def utc_stamp():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def statfs_magic(path):
    import ctypes
    import ctypes.util
    if not hasattr(os, "uname") or os.uname().sysname != "Linux":
        return None
    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    buffer = ctypes.create_string_buffer(256)
    if libc.statfs(os.fsencode(path), buffer) != 0:
        return None
    return ctypes.c_long.from_buffer(buffer).value & 0xFFFFFFFF


def self_identity():
    loaded = globals().get("__loader_sha256__")
    path = os.path.abspath(globals().get("__file__") or sys.argv[0])
    if loaded is None:
        try:
            with open(path, "rb") as stream:
                loaded = hashlib.sha256(stream.read()).hexdigest()
        except OSError:
            loaded = None
    return {"path": path, "sha256": loaded, "verified_by_loader": "__loader_sha256__" in globals(),
            "python": sys.version.split()[0], "executable": sys.executable, "pid": os.getpid(),
            "euid": os.geteuid(), "host": socket.gethostname(), "invocation_id": os.environ.get("INVOCATION_ID")}


# ---------------------------------------------------------------------------
# Source and target validation
# ---------------------------------------------------------------------------

def check_source(run_root, *, owner_uid=None, fs_magic=statfs_magic, run_parent=RUN_PARENT):
    """The Core1 run root: /dev/shm/ae-expA-<run>, a real tmpfs directory of the run owner."""
    if not os.path.isabs(run_root) or not RUN_ROOT_NAME.fullmatch(os.path.basename(run_root.rstrip("/"))):
        raise PublishRefused("the source must be an absolute run root named ae-expA-<run>")
    if run_parent is not None and os.path.dirname(run_root.rstrip("/")) != run_parent:
        raise PublishRefused(f"the source run root must be directly under {run_parent}")
    info = os.lstat(run_root)
    if not stat.S_ISDIR(info.st_mode):
        raise PublishRefused(f"{run_root} is not a real directory")
    host = socket.gethostname().split(".")[0]
    expected = owner_uid if owner_uid is not None else SOURCE_OWNER_UID_BY_HOST.get(host)
    if expected is not None and info.st_uid != expected:
        raise PublishRefused(f"{run_root} is owned by UID {info.st_uid}, not the run owner {expected}")
    magic = fs_magic(run_root)
    if magic is not None and magic != TMPFS_MAGIC:
        raise PublishRefused(f"{run_root} is not on tmpfs (statfs magic {magic:#x})")
    return {"run_root": run_root, "owner_uid": info.st_uid, "statfs_magic": magic}


def target_path(target_parent, run_id, allowed_parents=ALLOWED_TARGET_PARENTS):
    parent = os.path.normpath(target_parent)
    if parent not in allowed_parents:
        raise PublishRefused(f"target parent {target_parent} is not one of {list(allowed_parents)}")
    if not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id):
        raise PublishRefused("run id must match [A-Za-z0-9][A-Za-z0-9._-]{0,127}")
    return os.path.join(parent, run_id)


def check_target_dir(path, uid, gid):
    info = os.lstat(path)
    if not stat.S_ISDIR(info.st_mode):
        raise PublishRefused(f"{path} is not a real directory")
    if (info.st_uid, info.st_gid, stat.S_IMODE(info.st_mode)) != (uid, gid, 0o700):
        raise PublishRefused(f"{path} is {info.st_uid}:{info.st_gid} mode {stat.S_IMODE(info.st_mode):o}, "
                             f"not {uid}:{gid} mode 700")
    return info


def _sha256_path(path):
    descriptor = os.open(path, os.O_RDONLY | O_NOFOLLOW | O_CLOEXEC)
    try:
        return _hash_fd(descriptor)[0]
    finally:
        os.close(descriptor)


def target_records(target):
    """What a Core2 run directory already holds: run.json, cells and attempt directories (no symlink followed)."""
    report = {"run_json": False, "cells": 0, "attempts": 0, "files": 0}
    try:
        info = os.lstat(os.path.join(target, "run.json"))
        report["run_json"] = stat.S_ISREG(info.st_mode)
    except FileNotFoundError:
        pass
    for cell in _entries(os.path.join(target, "cells")):
        if _is_dir(cell):
            report["cells"] += 1
            for attempt in _entries(cell.path):
                if _is_dir(attempt):
                    report["attempts"] += 1
                    report["files"] += sum(1 for entry in _entries(attempt.path) if _is_regular(entry))
    return report


def init_target(target, uid=TARGET_UID, gid=TARGET_GID, run_root=None):
    """Create the run directory (0700, uid:gid); an existing one must already be exactly that.

    With ``run_root``: refuse a target that already holds this run's records when the source run root has
    none or another run.json (a reboot wiped /dev/shm, or the run id is reused): the runner would dispatch
    completed cells again and new attempts would collide with published ones. Restore first (the stager's
    run-state component with --restore).
    """
    parent = os.path.dirname(target)
    parent_info = os.lstat(parent)
    if not stat.S_ISDIR(parent_info.st_mode):
        raise PublishRefused(f"{parent} is not a real directory")
    if parent_info.st_uid != uid:
        raise PublishRefused(f"{parent} is owned by UID {parent_info.st_uid}, not {uid}")
    try:
        os.mkdir(target, 0o700)
        created = True
    except FileExistsError:
        created = False
    if created:
        descriptor = os.open(target, os.O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
        try:
            os.fchmod(descriptor, 0o700)
            os.fchown(descriptor, uid, gid)
        finally:
            os.close(descriptor)
    check_target_dir(target, uid, gid)
    records = target_records(target)
    if run_root is not None and records["run_json"]:
        source = os.path.join(run_root, "run", "run.json")
        if not os.path.lexists(source):
            raise PublishRefused(f"{target} already holds this run's records ({records['attempts']} attempts) but "
                                 f"{run_root}/run has none: restore them first (stage --restore), never relaunch on "
                                 "an empty run directory")
        if _sha256_path(source) != _sha256_path(os.path.join(target, "run.json")):
            raise PublishRefused(f"{target}/run.json differs from {source}: another run's records")
    return {"target": target, "created": created, "uid": uid, "gid": gid, "target_records": records}


# ---------------------------------------------------------------------------
# Candidate discovery (no symlink is ever followed)
# ---------------------------------------------------------------------------

def _entries(path):
    try:
        with os.scandir(path) as iterator:
            return sorted(iterator, key=lambda entry: entry.name)
    except (FileNotFoundError, NotADirectoryError):
        return []


def _is_regular(entry):
    return entry.is_file(follow_symlinks=False) and not entry.is_symlink()


def _is_dir(entry):
    return entry.is_dir(follow_symlinks=False) and not entry.is_symlink()


def _immutable(entry):
    return stat.S_IMODE(entry.stat(follow_symlinks=False).st_mode) & 0o222 == 0


def _visible(name):
    return bool(NAME.fullmatch(name)) and not name.startswith(".")


def discover(run_root, *, final=False):
    """Ordered candidate list [(source_path, target_relpath, ready, reason)] plus unexpected entries."""
    candidates, unexpected = [], []
    run = os.path.join(run_root, "run")
    for name in ("run.json", "cells.jsonl"):
        path = os.path.join(run, name)
        try:
            info = os.lstat(path)
        except FileNotFoundError:
            continue
        ready = stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) & 0o222 == 0
        candidates.append((path, name, ready, None if ready else "not_immutable"))
    for directory in ("launches", "snapshots"):
        for entry in _entries(os.path.join(run, directory)):
            if not _visible(entry.name):
                continue
            if not _is_regular(entry) or not entry.name.endswith(".json"):
                unexpected.append(f"{directory}/{entry.name}")
                continue
            ready = _immutable(entry)
            candidates.append((entry.path, f"{directory}/{entry.name}", ready, None if ready else "not_immutable"))
    for cell in _entries(os.path.join(run, "cells")):
        if not _visible(cell.name) or not _is_dir(cell):
            unexpected.append(f"cells/{cell.name}")
            continue
        for attempt in _entries(cell.path):
            if not _visible(attempt.name) or not _is_dir(attempt):
                unexpected.append(f"cells/{cell.name}/{attempt.name}")
                continue
            files = {}
            for entry in _entries(attempt.path):
                if not _visible(entry.name):
                    continue
                if entry.name not in ATTEMPT_FILES or not _is_regular(entry):
                    unexpected.append(f"cells/{cell.name}/{attempt.name}/{entry.name}")
                    continue
                files[entry.name] = entry
            prefix = f"cells/{cell.name}/{attempt.name}/"
            for name in ATTEMPT_FILES:
                entry = files.get(name)
                if entry is None:
                    continue
                ready, reason = _immutable(entry), None
                if name == "events.jsonl" and not ready and "attempt_closed.json" in files:
                    ready = True  # the owner was verified gone and the attempt closed: nothing writes it again
                if name == "result.json" and "events.jsonl" in files and not _immutable(files["events.jsonl"]):
                    ready, reason = False, "events_not_closed"
                if not ready and reason is None:
                    reason = "not_immutable"
                candidates.append((entry.path, prefix + name, ready, reason))
    for instance in _entries(os.path.join(run_root, "serve")):
        if not _visible(instance.name) or not _is_dir(instance):
            continue
        for entry in _entries(os.path.join(instance.path, "records")):
            if _visible(entry.name) and _is_regular(entry) and entry.name.endswith(".json"):
                ready = _immutable(entry)
                candidates.append((entry.path, f"serve/{instance.name}/records/{entry.name}", ready,
                                   None if ready else "not_immutable"))
        if final:
            for entry in _entries(os.path.join(instance.path, "logs")):
                if _visible(entry.name) and _is_regular(entry):
                    candidates.append((entry.path, f"serve/{instance.name}/logs/{entry.name}", True, None))
    for entry in _entries(os.path.join(run_root, "inputs")):
        if _visible(entry.name) and _is_regular(entry) and entry.name.startswith("STAGE_") \
                and entry.name.endswith(".json"):
            ready = _immutable(entry)
            if ready:  # content-addressed: a re-stage (for example after a reboot) never conflicts
                try:
                    digest = _sha256_path(entry.path)
                except OSError:
                    candidates.append((entry.path, f"stage/{entry.name}", False, "unreadable"))
                    continue
                candidates.append((entry.path, f"stage/{digest[:16]}/{entry.name}", True, None))
            else:
                candidates.append((entry.path, f"stage/{entry.name}", False, "not_immutable"))
    if final:
        for entry in _entries(os.path.join(run_root, "logs")):
            if _visible(entry.name) and _is_regular(entry):
                candidates.append((entry.path, f"logs/{entry.name}", True, None))
    return candidates, unexpected


# ---------------------------------------------------------------------------
# Copy, verify and never overwrite
# ---------------------------------------------------------------------------

def _hash_fd(descriptor):
    digest, size = hashlib.sha256(), 0
    os.lseek(descriptor, 0, os.SEEK_SET)
    while True:
        chunk = os.read(descriptor, BLOCK)
        if not chunk:
            break
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def _fsync_dir(descriptor):
    try:
        os.fsync(descriptor)
    except OSError:
        pass


class Target:
    """The Core2 run directory, written only through directory descriptors (no symlink is followed).

    Directory descriptors are cached in least-recently-used order and at most MAX_DIR_FDS stay open (the
    root's never closes): a run has one directory per cell and per attempt, thousands in a main run, far
    more than the 1024 open files a systemd service gets by default.
    """

    def __init__(self, root, uid=TARGET_UID, gid=TARGET_GID):
        self.root, self.uid, self.gid = root, uid, gid
        check_target_dir(root, uid, gid)
        self.root_fd = os.open(root, os.O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
        self.dir_fds = collections.OrderedDict([("", self.root_fd)])

    def close(self):
        for descriptor in self.dir_fds.values():
            try:
                os.close(descriptor)
            except OSError:
                pass
        self.dir_fds = collections.OrderedDict()

    def _evict(self, keep):
        while len(self.dir_fds) > MAX_DIR_FDS:
            victim = next(key for key in self.dir_fds if key not in ("", keep))
            os.close(self.dir_fds.pop(victim))

    def directory(self, relative):
        """Open (creating 0700 uid:gid when absent) a directory below the target root.

        A later call may evict (close) the returned descriptor, so callers use it at once and never keep it.
        """
        if relative in self.dir_fds:
            self.dir_fds.move_to_end(relative)
            return self.dir_fds[relative]
        parent, _, name = relative.rpartition("/")
        parent_fd = self.directory(parent)
        try:
            os.mkdir(name, 0o700, dir_fd=parent_fd)
            created = True
        except FileExistsError:
            created = False
        descriptor = os.open(name, os.O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC, dir_fd=parent_fd)
        info = os.fstat(descriptor)
        if created or (info.st_uid, info.st_gid, stat.S_IMODE(info.st_mode)) != (self.uid, self.gid, 0o700):
            if not created and info.st_uid not in (self.uid, 0):
                os.close(descriptor)
                raise PublishRefused(f"target directory {relative} is owned by UID {info.st_uid}")
            os.fchmod(descriptor, 0o700)
            os.fchown(descriptor, self.uid, self.gid)
        if created:
            _fsync_dir(parent_fd)
        self.dir_fds[relative] = descriptor
        self._evict(keep=relative)
        return descriptor

    def publish(self, source, relpath, *, after_copy=None):
        """Copy ``source`` to ``relpath``. Returns {status, sha256, bytes, ...}; never replaces a name."""
        parent, _, name = relpath.rpartition("/")
        source_fd = os.open(source, os.O_RDONLY | O_NOFOLLOW | O_CLOEXEC)
        try:
            before = os.fstat(source_fd)
            if not stat.S_ISREG(before.st_mode):
                return {"status": "skipped", "reason": "source_not_regular"}
            directory = self.directory(parent)
            try:
                existing = os.lstat(name, dir_fd=directory)
            except FileNotFoundError:
                existing = None
            if existing is not None:
                source_sha, size = _hash_fd(source_fd)
                return self._compare_existing(directory, name, relpath, source_sha, size)
            temporary = f".{name[:200]}.publish-{uuid.uuid4().hex}.tmp"
            target_fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0o600,
                                dir_fd=directory)
            linked = False
            try:
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
                    os.fchmod(target_fd, 0o444)
                    os.fchown(target_fd, self.uid, self.gid)
                finally:
                    os.close(target_fd)
                after = os.fstat(source_fd)
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or size != after.st_size:
                    return {"status": "deferred", "reason": "source_changed_during_copy"}
                if after_copy is not None:
                    after_copy(directory, temporary)
                try:
                    os.link(temporary, name, src_dir_fd=directory, dst_dir_fd=directory)
                    linked = True
                except FileExistsError:
                    return self._compare_existing(directory, name, relpath, digest.hexdigest(), size)
            finally:
                try:
                    os.unlink(temporary, dir_fd=directory)
                except FileNotFoundError:
                    pass
            _fsync_dir(directory)
            result = self._verify(directory, name, relpath, digest.hexdigest(), size)
            result["linked"] = linked
            return result
        finally:
            os.close(source_fd)

    def _verify(self, directory, name, relpath, sha, size):
        descriptor = os.open(name, os.O_RDONLY | O_NOFOLLOW | O_CLOEXEC, dir_fd=directory)
        try:
            info = os.fstat(descriptor)
            target_sha, target_size = _hash_fd(descriptor)
        finally:
            os.close(descriptor)
        problems = []
        if (target_sha, target_size) != (sha, size):
            problems.append(f"sha256 after copy {target_sha} != source {sha}")
        if (info.st_uid, info.st_gid, stat.S_IMODE(info.st_mode)) != (self.uid, self.gid, 0o444):
            problems.append(f"owner/mode {info.st_uid}:{info.st_gid} {stat.S_IMODE(info.st_mode):o}")
        if problems:
            return {"status": "verify_failed", "path": relpath, "sha256": sha, "target_sha256": target_sha,
                    "bytes": size, "problems": problems}
        return {"status": "published", "path": relpath, "sha256": sha, "bytes": size}

    def _compare_existing(self, directory, name, relpath, source_sha, size):
        try:
            descriptor = os.open(name, os.O_RDONLY | O_NOFOLLOW | O_CLOEXEC, dir_fd=directory)
        except OSError as error:
            return {"status": "conflict", "path": relpath, "sha256": source_sha, "target_sha256": None,
                    "reason": f"target_not_a_regular_file:{type(error).__name__}"}
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode):
                return {"status": "conflict", "path": relpath, "sha256": source_sha, "target_sha256": None,
                        "reason": "target_not_a_regular_file"}
            target_sha, _ = _hash_fd(descriptor)
            if target_sha != source_sha:
                return {"status": "conflict", "path": relpath, "sha256": source_sha, "target_sha256": target_sha,
                        "reason": "target_differs_never_overwritten"}
            repaired = []
            if stat.S_IMODE(info.st_mode) != 0o444:
                os.fchmod(descriptor, 0o444)
                repaired.append("mode")
            if (info.st_uid, info.st_gid) != (self.uid, self.gid):
                os.fchown(descriptor, self.uid, self.gid)
                repaired.append("owner")
        finally:
            os.close(descriptor)
        return {"status": "present", "path": relpath, "sha256": source_sha, "bytes": size, "repaired": repaired}

    def write_record(self, relpath, value):
        """An immutable ledger record (same never-overwrite path as published files)."""
        data = canonical_json(value) + b"\n"
        parent, _, name = relpath.rpartition("/")
        directory = self.directory(parent)
        temporary = f".{name}.publish-{uuid.uuid4().hex}.tmp"
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0o600,
                             dir_fd=directory)
        try:
            try:
                view = memoryview(data)
                while view:
                    view = view[os.write(descriptor, view):]
                os.fsync(descriptor)
                os.fchmod(descriptor, 0o444)
                os.fchown(descriptor, self.uid, self.gid)
            finally:
                os.close(descriptor)
            os.link(temporary, name, src_dir_fd=directory, dst_dir_fd=directory)
        finally:
            try:
                os.unlink(temporary, dir_fd=directory)
            except FileNotFoundError:
                pass
        _fsync_dir(directory)
        return {"path": relpath, "sha256": hashlib.sha256(data).hexdigest()}


# ---------------------------------------------------------------------------
# Passes and the loop
# ---------------------------------------------------------------------------

def _read_json(path, limit=64 * 1024**2):
    descriptor = os.open(path, os.O_RDONLY | O_NOFOLLOW | O_CLOEXEC)
    try:
        if os.fstat(descriptor).st_size > limit:
            raise ValueError("too large")
        data = b""
        while True:
            chunk = os.read(descriptor, BLOCK)
            if not chunk:
                break
            data += chunk
    finally:
        os.close(descriptor)
    return json.loads(data), hashlib.sha256(data).hexdigest()


class Publisher:
    def __init__(self, run_root, target_root, *, uid=TARGET_UID, gid=TARGET_GID, out=sys.stdout):
        self.run_root, self.target_root = run_root, target_root
        self.target = Target(target_root, uid, gid)
        self.done = {}            # target relpath -> sha256 (published or verified present)
        self.conflicts = {}       # target relpath -> conflict record (each recorded once)
        self.cell_ids = None
        self.completed_cells = set()
        self.pass_number = 0
        self.last_waiting = []    # [{path, reason}] still waiting after the last pass
        self.session = f"{utc_stamp()}-{socket.gethostname().split('.')[0]}-{os.getpid()}"
        self.out = out
        self.totals = {"published": 0, "present": 0, "conflicts": 0, "verify_failed": 0, "anomalies": 0,
                       "bytes": 0}

    def close(self):
        self.target.close()

    def _note_result(self, relpath, source):
        """Track completed cells from published result.json files; check events against the result."""
        parts = relpath.split("/")
        anomalies = []
        if len(parts) == 4 and parts[0] == "cells" and parts[3] == "result.json":
            try:
                result, _ = _read_json(source)
            except (OSError, ValueError) as error:
                return [{"path": relpath, "anomaly": f"result_unreadable:{type(error).__name__}"}]
            if result.get("cell_id") not in (None, parts[1]) or result.get("attempt_id") not in (None, parts[2]):
                anomalies.append({"path": relpath, "anomaly": "result_ids_differ_from_directory"})
            events = (result.get("events") or {}).get("sha256")
            published_events = self.done.get("/".join(parts[:3] + ["events.jsonl"]))
            if events is not None and published_events is not None and events != published_events:
                anomalies.append({"path": relpath, "anomaly": "events_sha256_differs_from_result",
                                  "events_sha256": published_events, "result_events_sha256": events})
            if result.get("exit_status") in COMPLETED_EXITS:
                self.completed_cells.add(parts[1])
        return anomalies

    def _load_cells(self):
        if self.cell_ids is None and "cells.jsonl" in self.done:
            path = os.path.join(self.run_root, "run", "cells.jsonl")
            try:
                with open(path, "rb") as stream:
                    raw = stream.read()
                if hashlib.sha256(raw).hexdigest() == self.done["cells.jsonl"]:
                    self.cell_ids = {json.loads(line)["cell_id"] for line in raw.splitlines() if line.strip()}
            except (OSError, ValueError, KeyError):
                self.cell_ids = None

    def run_complete(self):
        self._load_cells()
        return bool(self.cell_ids) and self.cell_ids <= self.completed_cells

    def _record_conflict(self, relpath, record, conflicts, kind="conflict"):
        self.conflicts[relpath] = record
        self.totals["conflicts" if kind == "conflict" else "verify_failed"] += 1
        conflicts.append(record)
        self.target.write_record(f"publish/conflicts/{self.session}-p{self.pass_number:06d}-{len(conflicts):04d}.json",
                                 {"schema": LEDGER_SCHEMA, "kind": kind, **record})

    def publish_pass(self, *, final=False):
        self.pass_number += 1
        check_target_dir(self.target_root, self.target.uid, self.target.gid)
        candidates, unexpected = discover(self.run_root, final=final)
        published, conflicts, anomalies, waiting, waiting_paths, errors = [], [], [], {}, [], []

        def wait(relpath, why):
            waiting[why] = waiting.get(why, 0) + 1
            waiting_paths.append({"path": relpath, "reason": why})
        for source, relpath, ready, reason in candidates:
            if relpath in self.done or relpath in self.conflicts:
                continue
            if not ready:
                wait(relpath, reason)
                continue
            parts = relpath.split("/")
            if len(parts) == 4 and parts[0] == "cells" and parts[3] != "attempt_start.json":
                start = "/".join(parts[:3] + ["attempt_start.json"])
                if start in self.conflicts:
                    # The target holds another launch's start record for this attempt id: never mix records.
                    self._record_conflict(relpath, {"status": "conflict", "path": relpath, "source": source,
                                                    "reason": "attempt_start_differs_on_target",
                                                    "detected_utc": utc_now(), "action": "not_published"},
                                          conflicts)
                    continue
                if start not in self.done:
                    wait(relpath, "waits_for_attempt_start")
                    continue
            if relpath.endswith("/result.json"):
                events = relpath[:-len("result.json")] + "events.jsonl"
                if os.path.lexists(os.path.join(self.run_root, "run", events)) and events not in self.done:
                    wait(relpath, "result_waits_for_events")
                    continue
            try:
                outcome = self.target.publish(source, relpath)
            except FileNotFoundError:
                wait(relpath, "source_vanished")
                continue
            except OSError as error:  # EIO, ESTALE, ENOSPC, EACCES ...: record, retry next pass
                name = type(error).__name__ if error.errno is None else (errno.errorcode.get(error.errno) or "OSError")
                wait(relpath, f"error:{name}")
                errors.append({"path": relpath, "error": f"{type(error).__name__}: {error}"[:300]})
                continue
            status = outcome["status"]
            if status in ("published", "present"):
                self.done[relpath] = outcome["sha256"]
                self.totals[status] += 1
                if status == "published":
                    self.totals["bytes"] += outcome["bytes"]
                    published.append({"path": relpath, "sha256": outcome["sha256"], "bytes": outcome["bytes"]})
                elif outcome.get("repaired"):
                    anomalies.append({"path": relpath, "anomaly": "existing_target_repaired",
                                      "repaired": outcome["repaired"]})
                try:
                    anomalies += self._note_result(relpath, source)
                except OSError as error:
                    anomalies.append({"path": relpath, "anomaly": f"result_unreadable:{type(error).__name__}"})
            elif status in ("conflict", "verify_failed"):
                record = dict(outcome, source=source, detected_utc=utc_now(), action="target_kept_unchanged")
                self._record_conflict(relpath, record, conflicts, kind=status)
            else:
                wait(relpath, outcome.get("reason", status))
        self.totals["anomalies"] += len(anomalies)
        self._load_cells()
        self.last_waiting = waiting_paths
        summary = {"pass": self.pass_number, "final": final, "published": len(published), "conflicts": len(conflicts),
                   "anomalies": len(anomalies), "waiting": waiting, "errors": len(errors), "unexpected": len(unexpected),
                   "done_total": len(self.done), "completed_cells": len(self.completed_cells),
                   "design_cells": len(self.cell_ids) if self.cell_ids else None, "utc": utc_now()}
        if published or conflicts or anomalies or errors or (unexpected and self.pass_number == 1):
            self.target.write_record(f"publish/passes/{self.session}-p{self.pass_number:06d}.json",
                                     {"schema": LEDGER_SCHEMA, "kind": "pass", "session": self.session,
                                      **summary, "files": published, "conflict_paths": [c["path"] for c in conflicts],
                                      "anomalies": anomalies, "errors": errors[:200],
                                      "unexpected_sample": unexpected[:50]})
        print(json.dumps(summary, sort_keys=True), file=self.out, flush=True)
        return summary


def run_loop(publisher, *, interval, stop_file, max_passes=None, final_on_stop=False, sleep=time.sleep,
             clock=time.monotonic):
    """Publish every ``interval`` seconds; one last records pass when stopping.

    The loop never publishes logs by default: on ``run_complete`` the vLLM units are still serving and their
    logs still grow (stop-serve appends shutdown lines), so a log published here would conflict with the
    -final unit's copy. ``final_on_stop`` (tests and manual use only) publishes logs on the stop file.
    """
    stopping = {"signal": None}

    def handle(signum, frame):
        stopping["signal"] = signum
    previous = {number: signal.signal(number, handle) for number in (signal.SIGTERM, signal.SIGINT)}
    reason = None
    try:
        while True:
            publisher.publish_pass()
            if stop_file and os.path.lexists(stop_file):
                reason = "stop_file"
            elif publisher.run_complete():
                reason = "run_complete"
            elif stopping["signal"] is not None:
                reason = f"signal_{stopping['signal']}"
            elif max_passes is not None and publisher.pass_number >= max_passes:
                reason = "max_passes"
            if reason:
                break
            deadline = clock() + interval
            while clock() < deadline and stopping["signal"] is None and not (stop_file and os.path.lexists(stop_file)):
                sleep(min(1.0, max(0.0, deadline - clock())))
        final = publisher.publish_pass(final=final_on_stop and reason == "stop_file")
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)
    return reason, final


# ---------------------------------------------------------------------------
# The hardened unit (printed for the launch script; run with sudo)
# ---------------------------------------------------------------------------

def unit_properties(run_root, read_write_path, runtime_max_seconds, *, restart=False):
    """Root, but confined: everything read-only except the one target directory; no network, few capabilities.

    ProtectSystem=strict leaves the API file systems (/dev, so /dev/shm) writable; ReadOnlyPaths=/dev/shm
    closes that for every /dev/shm entry, including other users' and the run root itself. The loop unit
    restarts on a transient failure (exit 3) but not on a refusal (2) or recorded conflicts (4).
    """
    restart_policy = (["Restart=on-failure", "RestartSec=30", "RestartPreventExitStatus=2 4 5"] if restart
                      else ["Restart=no"])
    return ["Type=exec", "User=root", "KillMode=control-group", "KillSignal=SIGTERM", *restart_policy,
            "TimeoutStopSec=600", f"RuntimeMaxSec={runtime_max_seconds}", "NoNewPrivileges=yes",
            "ProtectSystem=strict", "ProtectHome=yes", "PrivateTmp=yes", "PrivateNetwork=yes",
            "ProtectKernelTunables=yes", "ProtectKernelModules=yes", "ProtectControlGroups=yes",
            "RestrictSUIDSGID=yes", "LockPersonality=yes", "UMask=0077",
            "CapabilityBoundingSet=" + " ".join(CAPABILITIES),
            "InaccessiblePaths=" + " ".join("-" + path for path in INACCESSIBLE_PATHS),
            f"ReadOnlyPaths={RUN_PARENT} {run_root}", f"ReadWritePaths={read_write_path}"]


def systemd_run_argv(*, mode, unit, run_root, target_parent, run_id, script, script_sha256, interval=30,
                     python=SYSTEM_PYTHON, runtime_max_seconds=7 * 86400, stop_file=None,
                     allowed_parents=ALLOWED_TARGET_PARENTS):
    """``init`` (creates the run dir; writable: the target parent), ``loop`` or ``final`` (writable: the run dir).

    init and final wait for the unit and pipe its output (systemd-run --wait --pipe --collect).
    """
    target = target_path(target_parent, run_id, allowed_parents)
    for name, value in (("run_root", run_root), ("script", script), ("python", python)):
        if not isinstance(value, str) or not value.startswith("/"):
            raise PublishRefused(f"{name} must be an absolute path")
    if not re.fullmatch(r"[0-9a-f]{64}", script_sha256 or ""):
        raise PublishRefused("script_sha256 must be 64 lowercase hex digits")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.@-]{0,200}", unit or ""):
        raise PublishRefused("invalid unit name")
    common = ["--run-root", run_root, "--target-parent", os.path.normpath(target_parent), "--run-id", run_id]
    if mode == "init":
        writable, options, tail = os.path.normpath(target_parent), ["--wait", "--pipe", "--collect"], ["init"] + common
    elif mode == "final":
        writable, options, tail = target, ["--wait", "--pipe", "--collect"], ["once", "--final"] + common
    elif mode == "loop":
        # --collect: a failed loop unit is unloaded, so `publisher` can start it again under the same name.
        writable, options = target, ["--collect"]
        tail = ["loop"] + common + ["--interval", str(interval)] + (["--stop-file", stop_file] if stop_file else [])
    else:
        raise PublishRefused("mode must be init, loop or final")
    command = [python, "-I", "-B", "-c", LOADER, script, script_sha256] + tail
    properties = unit_properties(run_root, writable, runtime_max_seconds, restart=mode == "loop")
    return (["systemd-run", f"--unit={unit}", f"--description=Absent Edges Exp A publisher {mode} {run_id}"]
            + options + ["--property=" + value for value in properties]
            + ["--setenv=LANG=C.UTF-8", "--setenv=PYTHONDONTWRITEBYTECODE=1", "--"] + command)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def exit_status(publisher, *, final):
    """(code, waiting paths): 4 when conflicts were recorded; for a final pass 5 when any record still waits
    (it would be lost at cleanup), else 0."""
    waiting = list(publisher.last_waiting) if final else []
    if publisher.conflicts:
        return 4, waiting
    return (5 if waiting else 0), waiting


def _require_root(args):
    if os.geteuid() != 0 and not args.allow_non_root:
        raise PublishRefused("the publisher runs as root (NFS no_root_squash); use sudo")


def _common(parser):
    parser.add_argument("--run-root", required=True, help="/dev/shm/ae-expA-<run> on Core1")
    parser.add_argument("--target-parent", default=ALLOWED_TARGET_PARENTS[0])
    parser.add_argument("--run-id", required=True, help="Core2 run directory name")
    parser.add_argument("--target-uid", type=int, default=TARGET_UID)
    parser.add_argument("--target-gid", type=int, default=TARGET_GID)
    parser.add_argument("--allow-non-root", action="store_true", help=argparse.SUPPRESS)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("init", "once", "loop", "plan"):
        command = commands.add_parser(name)
        _common(command)
        if name in ("once", "loop", "plan"):
            command.add_argument("--stop-file", default=None, help="default: <run root>/control/publish.stop")
        if name == "once":
            command.add_argument("--final", action="store_true",
                                 help="also publish logs (the runner and every serve unit have stopped); exit 5 when "
                                      "any record is still waiting")
        if name in ("loop", "plan"):
            command.add_argument("--interval", type=float, default=30.0)
        if name == "loop":
            command.add_argument("--max-passes", type=int, default=None)
            command.add_argument("--final-logs", action="store_true",
                                 help="manual use only: publish logs in the last pass on the stop file")
            command.add_argument("--no-final-logs", action="store_true", help="accepted for compatibility (default)")
        if name == "plan":
            command.add_argument("--unit", default=None)
            command.add_argument("--script", required=True, help="the staged expA_publish.py root will execute")
            command.add_argument("--script-sha256", required=True, help="pinned sha256 of that script")
            command.add_argument("--python", default=SYSTEM_PYTHON)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        target = target_path(args.target_parent, args.run_id)
        stop_file = getattr(args, "stop_file", None) or os.path.join(args.run_root, "control", "publish.stop")
        if args.command == "plan":
            unit = args.unit or f"ae-expA-publish-{os.path.basename(args.run_root.rstrip('/'))[len('ae-expA-'):]}"
            plans = {}
            for mode in ("init", "loop", "final"):
                argv = systemd_run_argv(mode=mode, unit=unit if mode == "loop" else f"{unit}-{mode}",
                                        run_root=args.run_root, target_parent=args.target_parent, run_id=args.run_id,
                                        script=args.script, script_sha256=args.script_sha256,
                                        interval=int(args.interval), python=args.python, stop_file=stop_file)
                plans[mode] = {"argv": argv, "command": "sudo " + shlex.join(argv)}
            print(json.dumps({"unit": unit, "target": target, "stop_file": stop_file, "units": plans},
                             indent=2, sort_keys=True))
            return 0
        _require_root(args)
        source = check_source(args.run_root, owner_uid=None if not args.allow_non_root else os.geteuid())
        if args.command == "init":
            print(json.dumps({"protocol": PUBLISH_PROTOCOL, "source": source,
                              **init_target(target, args.target_uid, args.target_gid, run_root=args.run_root)},
                             sort_keys=True))
            return 0
        publisher = Publisher(args.run_root, target, uid=args.target_uid, gid=args.target_gid)
        try:
            publisher.target.write_record(f"publish/started-{publisher.session}.json",
                                          {"schema": LEDGER_SCHEMA, "kind": "started", "protocol": PUBLISH_PROTOCOL,
                                           "publisher": self_identity(), "source": source, "target": target,
                                           "argv": sys.argv, "command": args.command, "utc": utc_now()})
            if args.command == "once":
                reason, final = "once", publisher.publish_pass(final=args.final)
            else:
                reason, final = run_loop(publisher, interval=args.interval, stop_file=stop_file,
                                         max_passes=args.max_passes, final_on_stop=args.final_logs)
            publisher.target.write_record(f"publish/finished-{publisher.session}.json",
                                          {"schema": LEDGER_SCHEMA, "kind": "finished", "reason": reason,
                                           "passes": publisher.pass_number, "totals": publisher.totals,
                                           "done_files": len(publisher.done), "conflicts": sorted(publisher.conflicts),
                                           "completed_cells": len(publisher.completed_cells),
                                           "design_cells": len(publisher.cell_ids) if publisher.cell_ids else None,
                                           "waiting_paths": publisher.last_waiting[:1000],
                                           "waiting_total": len(publisher.last_waiting),
                                           "last_pass": final, "utc": utc_now()})
        finally:
            publisher.close()
        code, waiting = exit_status(publisher, final=getattr(args, "final", False))
        print(json.dumps({"protocol": PUBLISH_PROTOCOL, "reason": reason, "totals": publisher.totals,
                          "conflicts": sorted(publisher.conflicts), "waiting_total": len(waiting),
                          "waiting_sample": waiting[:20]}, sort_keys=True), flush=True)
        return code
    except PublishRefused as error:
        print(json.dumps({"protocol": PUBLISH_PROTOCOL, "status": "refused",
                          "reason": f"{type(error).__name__}: {error}"}, sort_keys=True), flush=True)
        return 2
    except OSError as error:  # transient (EIO, ESTALE, ...): the loop unit restarts after RestartSec
        print(json.dumps({"protocol": PUBLISH_PROTOCOL, "status": "os_error",
                          "reason": f"{type(error).__name__}: {error}"}, sort_keys=True), flush=True)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
