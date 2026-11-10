"""Exp A cell queue (docs/EXPA_DESIGN.md section 7; v2 cells per docs/EXPA_DESIGN_V2_DELTA.md D8/D10.8).

A cell is one (block, item, variant, condition, seed). Its id is the interface
contract format ``"{block}|{item_id}|{variant}|{condition}|s{seed}"``; the frozen
cell spec also binds the family, the target identity, the item's wording pair
and the expected payload sha256 (null for C0 and the T arms).

Durable truth is a tree of immutable files under the run directory (written
from Core1 through NFS):

    <run>/run.json, <run>/cells.jsonl                      frozen at run start
    <run>/cells/<cell_id>/<attempt_id>/attempt_start.json  written BEFORE dispatch
    <run>/cells/<cell_id>/<attempt_id>/events.jsonl        harness, fsync per record
    <run>/cells/<cell_id>/<attempt_id>/result.json         harness, immutable
    <run>/cells/<cell_id>/<attempt_id>/attempt_closed.json recovery of a dead owner

The live SQLite ledger is only a lease/fencing index on LOCAL disk (for example
/dev/shm). It follows absent_edges/registry.py Ledger semantics: finite leases,
heartbeats, fencing tokens, ``recover_expired`` that never kills anything and
retains any lease whose owner is not verified gone. It refuses network or FUSE
filesystems and can be rebuilt from the durable records. A cell with a completed
result is never dispatched again; infrastructure-interrupted attempts keep their
records and the cell receives a new attempt id, up to ``max_attempts`` (3, D11.3)
interrupted or abandoned attempts, after which it is blocked ("missing
(infrastructure)").

Queue order (D10.8): cells are claimed by (segment, position). A segment is one
(block, seed) in the order the frozen cell list first names it (P s11, P s23,
P s47, S-VC, S-NAT, S-EDGE, T); position is the cell's place in that list. An
interrupted cell returns to the back of its own segment, so an outage is spread
over many cells and a retried P cell never runs after a later seed or block.
"""
from __future__ import annotations

from contextlib import contextmanager
import ctypes
import ctypes.util
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import sqlite3
import time
import uuid

from ..data import canonical_json
from ..registry import FenceError

LEDGER_PROTOCOL = "expA-ledger-v2.1"  # v2.1: (block, seed) queue segments
RUN_SCHEMA = "expA-run-v2"
COMPLETED_EXITS = frozenset({"Submitted", "NoAnswer", "ContextLimit", "FormatExhausted"})
INFRA_EXIT = "InfraInterrupted"
CELL_FIELDS = ("block", "item_id", "family", "target", "variant", "condition", "seed", "wording_pair",
               "payload_sha256")
CELL_ID_FIELDS = ("block", "item_id", "variant", "condition", "seed")
# Each id part is one safe token: the cell id doubles as a directory name under <run>/cells/.
_ID_PART = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
NETWORK_FILESYSTEMS = frozenset({
    "nfs", "nfs4", "nfsd", "cifs", "smb", "smb2", "smb3", "smbfs", "9p", "afs", "ceph", "glusterfs",
    "lustre", "gpfs", "beegfs", "ocfs2", "gfs2", "davfs", "sshfs", "orangefs", "pvfs2", "ncpfs",
    "coda", "autofs", "virtiofs"})
NETWORK_MAGIC = {0x6969: "nfs", 0x65735546: "fuse", 0x517B: "smb", 0xFF534D42: "cifs",
                 0xFE534D42: "smb2", 0x01021997: "9p", 0x00C36400: "ceph", 0x5346414F: "afs",
                 0x0BD00BD0: "lustre", 0x47504653: "gpfs", 0x7461636F: "ocfs2", 0x01161970: "gfs2",
                 0x19830326: "beegfs", 0x6E667364: "nfsd"}


class UnsafeLedgerLocation(RuntimeError):
    """SQLite, locks and sockets must stay on local disk, never NFS/FUSE."""


class LedgerInconsistency(RuntimeError):
    """Durable records and the ledger disagree in a way that needs a human."""


# ---------------------------------------------------------------------------
# Local-filesystem guard
# ---------------------------------------------------------------------------

def _unescape_mount(value):
    out, index = [], 0
    while index < len(value):
        if value[index] == "\\" and index + 3 < len(value) and value[index + 1:index + 4].isdigit():
            out.append(chr(int(value[index + 1:index + 4], 8)))
            index += 4
        else:
            out.append(value[index])
            index += 1
    return "".join(out)


def parse_mountinfo(text):
    """[(mount_point, fstype, source)] in mount order from /proc/self/mountinfo text."""
    mounts = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 8 or "-" not in fields[6:]:
            continue
        separator = fields.index("-", 6)
        if separator + 1 >= len(fields):
            continue
        fstype = fields[separator + 1]
        source = fields[separator + 2] if separator + 2 < len(fields) else ""
        mounts.append((_unescape_mount(fields[4]), fstype, _unescape_mount(source)))
    return mounts


def parse_proc_mounts(text):
    mounts = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 3:
            mounts.append((_unescape_mount(fields[1]), fields[2], _unescape_mount(fields[0])))
    return mounts


def read_mount_table():
    for path, parser in (("/proc/self/mountinfo", parse_mountinfo), ("/proc/mounts", parse_proc_mounts)):
        try:
            with open(path, encoding="utf-8", errors="surrogateescape") as stream:
                mounts = parser(stream.read())
        except OSError:
            continue
        if mounts:
            return mounts
    raise UnsafeLedgerLocation("cannot read the mount table; refusing to place SQLite on an unknown filesystem")


def _existing_ancestor(path):
    current = Path(os.path.abspath(path))
    while not current.exists():
        if current.parent == current:
            break
        current = current.parent
    return Path(os.path.realpath(current))


def resolved_path(path):
    """Symlinks resolved on the existing prefix; the not-yet-created tail is kept as given."""
    absolute = Path(os.path.abspath(path))
    ancestor = absolute
    while not ancestor.exists() and ancestor.parent != ancestor:
        ancestor = ancestor.parent
    tail = absolute.relative_to(ancestor) if ancestor != absolute else Path()
    return (Path(os.path.realpath(ancestor)) / tail).as_posix()


def mount_for(path, mounts):
    """Longest matching mount point; later mounts shadow earlier ones."""
    target = resolved_path(path)
    best, best_index = None, -1
    for index, (mount_point, fstype, source) in enumerate(mounts):
        prefix = mount_point.rstrip("/") + "/"
        if target == mount_point or target.startswith(prefix) or mount_point == "/":
            if best is None or len(mount_point) > len(best[0]) or (len(mount_point) == len(best[0]) and index > best_index):
                best, best_index = (mount_point, fstype, source), index
    return best


def statfs_magic(path):
    """f_type from statfs(2) via libc, or None when unavailable."""
    if not hasattr(os, "uname") or os.uname().sysname != "Linux":
        return None
    library = ctypes.util.find_library("c")
    libc = ctypes.CDLL(library, use_errno=True)
    buffer = ctypes.create_string_buffer(256)
    if libc.statfs(os.fsencode(str(_existing_ancestor(path))), buffer) != 0:
        return None
    return ctypes.c_long.from_buffer(buffer).value & 0xFFFFFFFF


def is_network_fstype(fstype):
    base = (fstype or "").lower()
    return (base in NETWORK_FILESYSTEMS or base.startswith("fuse") or base.startswith("nfs")
            or base.split(".", 1)[0] in NETWORK_FILESYSTEMS)


def require_local_filesystem(path, *, mounts=None, magic=None):
    """Raise UnsafeLedgerLocation unless ``path`` (or its nearest existing ancestor) is local.

    Two independent checks must both pass: the mount table fstype and the
    statfs(2) magic number. Unknown platforms fail closed.
    """
    if mounts is None:
        if os.name != "posix":
            raise UnsafeLedgerLocation("local-filesystem verification requires Linux")
        mounts = read_mount_table()
    entry = mount_for(path, mounts)
    if entry is None:
        raise UnsafeLedgerLocation(f"no mount entry covers {path}")
    mount_point, fstype, source = entry
    if is_network_fstype(fstype):
        raise UnsafeLedgerLocation(f"{path} is on {fstype} ({source} at {mount_point}); SQLite must use local disk")
    value = statfs_magic(path) if magic is None else magic(path)
    if value is not None and value in NETWORK_MAGIC:
        raise UnsafeLedgerLocation(f"{path} statfs magic {value:#x} is {NETWORK_MAGIC[value]}; SQLite must use local disk")
    return {"path": str(path), "mount_point": mount_point, "fstype": fstype, "statfs_magic": value}


# ---------------------------------------------------------------------------
# Durable immutable records
# ---------------------------------------------------------------------------

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fsync_directory(path):
    if os.name != "posix":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_immutable_bytes(path, data: bytes, mode=0o444) -> str:
    """Publish complete bytes under a new name; never overwrites (link(2) is atomic on NFS)."""
    target = Path(path)
    temporary = target.with_name("." + target.name + "." + uuid.uuid4().hex + ".tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
            if hasattr(os, "fchmod"):
                os.fchmod(stream.fileno(), mode)
        os.link(temporary, target)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    if not hasattr(os, "fchmod"):
        os.chmod(target, mode)
    fsync_directory(target.parent)
    return sha256_bytes(data)


def write_immutable_json(path, value) -> str:
    return write_immutable_bytes(path, canonical_json(value) + b"\n")


def read_json_with_hash(path):
    raw = Path(path).read_bytes()
    return json.loads(raw), sha256_bytes(raw)


def cell_identifier(spec) -> str:
    """Contract cell id ``"{block}|{item_id}|{variant}|{condition}|s{seed}"``."""
    missing = [field for field in CELL_ID_FIELDS if field not in spec]
    if missing:
        raise ValueError("cell spec lacks " + ", ".join(missing))
    for name in ("block", "item_id", "variant", "condition"):
        value = spec[name]
        if not isinstance(value, str) or not _ID_PART.fullmatch(value):
            raise ValueError(f"cell {name} must match {_ID_PART.pattern}: {value!r}")
    if type(spec["seed"]) is not int or spec["seed"] < 0:
        raise ValueError("seed must be a non-negative integer")
    return f"{spec['block']}|{spec['item_id']}|{spec['variant']}|{spec['condition']}|s{spec['seed']}"


def make_cell(**fields):
    """A frozen v2 cell spec: every CELL_FIELDS key, plus its contract cell id."""
    unknown = set(fields) - set(CELL_FIELDS)
    if unknown:
        raise ValueError("unknown cell fields: " + ", ".join(sorted(unknown)))
    missing = [field for field in CELL_FIELDS if field not in fields]
    if missing:
        raise ValueError("cell spec lacks " + ", ".join(missing))
    spec = {field: fields[field] for field in CELL_FIELDS}
    for name in ("family", "target"):
        if not isinstance(spec[name], str) or not spec[name]:
            raise ValueError(f"{name} must be nonempty text")
    if spec["wording_pair"] not in ("A", "B"):
        raise ValueError("wording_pair must be A or B")
    if spec["payload_sha256"] is not None and (not isinstance(spec["payload_sha256"], str)
                                               or not _SHA256.fullmatch(spec["payload_sha256"])):
        raise ValueError("payload_sha256 must be null or 64 lowercase hex digits")
    return {"cell_id": cell_identifier(spec), **spec}


def attempt_identifier(cell_id, ordinal) -> str:
    if type(ordinal) is not int or ordinal < 1:
        raise ValueError("attempt ordinal must be a positive integer")
    return f"{cell_id}.a{ordinal:03d}"


class DurableStore:
    """Immutable per-run and per-attempt records; the source of truth for rebuilds."""

    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)

    @property
    def run_path(self):
        return self.run_dir / "run.json"

    @property
    def cells_path(self):
        return self.run_dir / "cells.jsonl"

    def initialize(self, manifest, cells):
        """Freeze the run manifest and finite cell list once; re-runs must match byte for byte."""
        cell_ids = [cell["cell_id"] for cell in cells]
        if len(set(cell_ids)) != len(cell_ids):
            raise ValueError("duplicate cells")
        for cell in cells:
            if cell["cell_id"] != cell_identifier(cell):
                raise ValueError("cell id does not match its spec")
        cells_bytes = b"".join(canonical_json(cell) + b"\n" for cell in cells)
        frozen = dict(manifest, schema=RUN_SCHEMA, cells_sha256=sha256_bytes(cells_bytes),
                      cell_count=len(cells))
        run_bytes = canonical_json(frozen) + b"\n"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for path, data in ((self.cells_path, cells_bytes), (self.run_path, run_bytes)):
            if path.exists():
                if path.read_bytes() != data:
                    raise LedgerInconsistency(f"{path} already exists with different content")
            else:
                write_immutable_bytes(path, data)
        (self.run_dir / "cells").mkdir(exist_ok=True)
        return frozen

    def manifest(self):
        return json.loads(self.run_path.read_bytes())

    def load_cells(self):
        manifest = self.manifest()
        raw = self.cells_path.read_bytes()
        if sha256_bytes(raw) != manifest["cells_sha256"]:
            raise LedgerInconsistency("cells.jsonl does not match run.json")
        return [json.loads(line) for line in raw.splitlines() if line.strip()]

    def attempt_dir(self, cell_id, attempt_id):
        return self.run_dir / "cells" / cell_id / attempt_id

    def start_attempt(self, cell_id, attempt_id, record):
        """Create the attempt directory exclusively and its start record, durably, before dispatch."""
        directory = self.attempt_dir(cell_id, attempt_id)
        directory.parent.mkdir(parents=True, exist_ok=True)
        os.mkdir(directory, 0o755)
        fsync_directory(directory.parent)
        sha = write_immutable_json(directory / "attempt_start.json", record)
        return directory, sha

    def close_attempt(self, cell_id, attempt_id, record):
        """Durably close a start-only attempt whose owner is gone. A result always wins."""
        directory = self.attempt_dir(cell_id, attempt_id)
        if not (directory / "attempt_start.json").exists() or (directory / "result.json").exists():
            return None
        if (directory / "attempt_closed.json").exists():
            return sha256_path(directory / "attempt_closed.json")
        return write_immutable_json(directory / "attempt_closed.json", record)

    def completed_result(self, cell_id):
        """The first durable completed attempt of a cell, or None."""
        for attempt in self.scan([cell_id]).get(cell_id, []):
            if attempt.get("state") == "completed":
                return attempt
        return None

    def scan(self, cell_ids=None):
        """{cell_id: [attempt records sorted by ordinal]} from durable files only."""
        root = self.run_dir / "cells"
        found = {}
        names = sorted(cell_ids) if cell_ids is not None else (
            sorted(p.name for p in root.iterdir() if p.is_dir()) if root.is_dir() else [])
        for cell_id in names:
            cell_dir = root / cell_id
            if not cell_dir.is_dir():
                continue
            attempts = []
            for attempt_dir in sorted(p for p in cell_dir.iterdir() if p.is_dir()):
                start_path = attempt_dir / "attempt_start.json"
                if not start_path.exists():
                    # Created, then crashed before the start record: never dispatched.
                    attempts.append({"attempt_id": attempt_dir.name, "state": "missing_start",
                                     "ordinal": parse_attempt_ordinal(cell_id, attempt_dir.name), "token": 0})
                    continue
                start, start_sha = read_json_with_hash(start_path)
                entry = {"attempt_id": attempt_dir.name, "ordinal": start["ordinal"], "token": start["token"],
                         "start": start, "start_sha256": start_sha, "result": None, "result_sha256": None,
                         "closed": None}
                if (attempt_dir / "result.json").exists():
                    entry["result"], entry["result_sha256"] = read_json_with_hash(attempt_dir / "result.json")
                if (attempt_dir / "attempt_closed.json").exists():
                    entry["closed"], _ = read_json_with_hash(attempt_dir / "attempt_closed.json")
                entry["state"] = durable_attempt_state(entry)
                attempts.append(entry)
            attempts.sort(key=lambda item: item.get("ordinal", 0))
            found[cell_id] = attempts
        return found


def parse_attempt_ordinal(cell_id, attempt_id):
    prefix = cell_id + ".a"
    if not attempt_id.startswith(prefix) or not attempt_id[len(prefix):].isdigit():
        raise LedgerInconsistency("unexpected attempt directory " + attempt_id)
    return int(attempt_id[len(prefix):])


def durable_attempt_state(entry):
    result = entry.get("result")
    if result is not None:
        status = result.get("exit_status")
        if status in COMPLETED_EXITS:
            return "completed"
        if status == INFRA_EXIT:
            return "infra_interrupted"
        return "invalid_result"
    if entry.get("closed") is not None:
        return entry["closed"].get("status", "abandoned")
    return "started"


# ---------------------------------------------------------------------------
# Process identity
# ---------------------------------------------------------------------------

def boot_id():
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return None


def _start_ticks(pid):
    raw = Path(f"/proc/{pid}/stat").read_text()
    return int(raw[raw.rfind(")") + 2:].split()[19])


def process_identity(pid=None):
    pid = os.getpid() if pid is None else pid
    identity = {"host": socket.gethostname(), "boot_id": boot_id(), "pid": pid}
    try:
        identity["start_ticks"] = _start_ticks(pid)
    except (OSError, ValueError, IndexError):
        identity["start_ticks"] = None
    return identity


def inspect_identity(identity):
    """gone / alive / unknown. Only a verified-gone owner loses its lease."""
    if not identity:
        return "gone"
    if identity.get("host") != socket.gethostname():
        return "unknown"
    current_boot = boot_id()
    if current_boot is None or identity.get("start_ticks") is None:
        return "unknown"
    if identity.get("boot_id") != current_boot:
        return "gone"
    try:
        ticks = _start_ticks(identity["pid"])
    except FileNotFoundError:
        return "gone"
    except (OSError, ValueError, IndexError):
        return "unknown"
    return "alive" if ticks == identity["start_ticks"] else "gone"


# ---------------------------------------------------------------------------
# SQLite lease ledger
# ---------------------------------------------------------------------------

class CellLedger:
    """Finite cell queue with leases, heartbeats and fencing tokens on local disk."""

    def __init__(self, path, *, run_id, max_attempts=3, mounts=None, magic=None):
        self.path = Path(os.path.abspath(path))
        self.location = require_local_filesystem(self.path, mounts=mounts, magic=magic)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        require_local_filesystem(self.path.parent, mounts=mounts, magic=magic)
        if type(max_attempts) is not int or max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")
        self.run_id, self.max_attempts = run_id, max_attempts
        with self.connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS cells (
                    cell_id TEXT PRIMARY KEY, spec TEXT NOT NULL, segment INTEGER NOT NULL,
                    position INTEGER NOT NULL, status TEXT NOT NULL, token INTEGER NOT NULL DEFAULT 0,
                    attempts INTEGER NOT NULL DEFAULT 0, owner TEXT, identity TEXT,
                    lease_expiry REAL, attempt_id TEXT, result_sha256 TEXT, exit_status TEXT,
                    error_code TEXT, created REAL NOT NULL, updated REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS attempts (
                    attempt_id TEXT PRIMARY KEY, cell_id TEXT NOT NULL, ordinal INTEGER NOT NULL,
                    token INTEGER NOT NULL, owner TEXT NOT NULL, identity TEXT, claimed REAL NOT NULL,
                    started REAL, ended REAL, status TEXT NOT NULL, start_sha256 TEXT,
                    result_sha256 TEXT, exit_status TEXT, error_code TEXT,
                    UNIQUE(cell_id, ordinal));
                CREATE TABLE IF NOT EXISTS events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, timestamp REAL NOT NULL,
                    cell_id TEXT, event TEXT NOT NULL, detail TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS segments (key TEXT PRIMARY KEY, rank INTEGER NOT NULL UNIQUE);
                CREATE INDEX IF NOT EXISTS cell_queue ON cells(status, segment, position);
            """)
        with self.transaction() as db:
            row = db.execute("SELECT value FROM metadata WHERE key='run_id'").fetchone()
            if row is None:
                db.execute("INSERT INTO metadata VALUES('run_id',?)", (run_id,))
                db.execute("INSERT INTO metadata VALUES('protocol',?)", (LEDGER_PROTOCOL,))
            elif row[0] != run_id:
                raise UnsafeLedgerLocation(f"ledger belongs to run {row[0]}, not {run_id}")
            protocol = db.execute("SELECT value FROM metadata WHERE key='protocol'").fetchone()
            if protocol is None or protocol[0] != LEDGER_PROTOCOL:
                raise UnsafeLedgerLocation(f"ledger protocol {protocol[0] if protocol else None!r} is not "
                                           f"{LEDGER_PROTOCOL}; rebuild it from the durable records")

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA busy_timeout=30000")
        db.execute("PRAGMA synchronous=FULL")
        try:
            yield db
        finally:
            db.close()

    @contextmanager
    def transaction(self):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                yield db
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise

    @staticmethod
    def _event(db, cell_id, event, detail=None):
        db.execute("INSERT INTO events(timestamp,cell_id,event,detail) VALUES(?,?,?,?)",
                   (time.time(), cell_id, event, canonical_json(detail).decode()))

    @staticmethod
    def _row(row):
        if row is None:
            return None
        result = dict(row)
        for key in ("spec", "identity"):
            if result.get(key):
                result[key] = json.loads(result[key])
        return result

    # -- queue ---------------------------------------------------------------

    @staticmethod
    def _segment(db, cell):
        """Rank of the cell's (block, seed) segment; a new segment ranks after every known one."""
        key = f"{cell['block']}|s{cell['seed']}"
        row = db.execute("SELECT rank FROM segments WHERE key=?", (key,)).fetchone()
        if row is not None:
            return row[0]
        rank = db.execute("SELECT COALESCE(MAX(rank), -1) FROM segments").fetchone()[0] + 1
        db.execute("INSERT INTO segments(key,rank) VALUES(?,?)", (key, rank))
        return rank

    def enqueue(self, cells):
        """Idempotently add the finite cell list in queue order; a changed spec for a known id is corruption."""
        added = 0
        now = time.time()
        with self.transaction() as db:
            offset = db.execute("SELECT COALESCE(MAX(position), -1) FROM cells").fetchone()[0] + 1
            for index, cell in enumerate(cells):
                if cell["cell_id"] != cell_identifier(cell):
                    raise ValueError("cell id does not match its spec")
                encoded = canonical_json(cell).decode()
                old = db.execute("SELECT spec FROM cells WHERE cell_id=?", (cell["cell_id"],)).fetchone()
                if old is not None:
                    if old[0] != encoded:
                        raise LedgerInconsistency("cell spec changed for " + cell["cell_id"])
                    continue
                db.execute("INSERT INTO cells(cell_id,spec,segment,position,status,created,updated) "
                           "VALUES(?,?,?,?,?,?,?)",
                           (cell["cell_id"], encoded, self._segment(db, cell), offset + index, "pending", now, now))
                added += 1
            if added:
                self._event(db, None, "enqueued", {"cells": added})
        return added

    def claim(self, owner, identity, lease_seconds=900.0, eligible=None, skip=None):
        """Lease the next pending cell in queue order; returns token, ordinal and the new attempt id.

        ``eligible`` (a set of cell ids) restricts the claim, for example to the
        blocks and seeds selected for one launch; ``skip`` (a set of cell ids) passes
        over cells this claimant must not take now (the runner never gives a cell's next
        attempt to the endpoint that just failed it while another endpoint is alive).
        Pending rows are scanned lazily in (segment, position) order, so the first
        acceptable row ends the scan.
        """
        if not owner or not (0 < lease_seconds < 7 * 86400):
            raise ValueError("owner and a positive finite lease are required")
        query = ("SELECT * FROM cells WHERE status='pending' ORDER BY segment, position"
                 + ("" if eligible is not None or skip else " LIMIT 1"))
        with self.transaction() as db:
            now = time.time()  # inside BEGIN IMMEDIATE: claim times follow the commit order
            found, cursor = None, db.execute(query)
            try:
                for candidate in cursor:
                    if (eligible is None or candidate["cell_id"] in eligible) \
                            and not (skip and candidate["cell_id"] in skip):
                        found = candidate
                        break
            finally:
                cursor.close()
            for row in ([] if found is None else [found]):
                token, ordinal = row["token"] + 1, row["attempts"] + 1
                attempt_id = attempt_identifier(row["cell_id"], ordinal)
                encoded_identity = canonical_json(identity).decode()
                db.execute("UPDATE cells SET status='leased',token=?,attempts=?,owner=?,identity=?,lease_expiry=?,"
                           "attempt_id=?,updated=? WHERE cell_id=?",
                           (token, ordinal, owner, encoded_identity, now + lease_seconds, attempt_id, now,
                            row["cell_id"]))
                db.execute("INSERT INTO attempts(attempt_id,cell_id,ordinal,token,owner,identity,claimed,status) "
                           "VALUES(?,?,?,?,?,?,?,'claimed')",
                           (attempt_id, row["cell_id"], ordinal, token, owner, encoded_identity, now))
                self._event(db, row["cell_id"], "claimed", {"token": token, "attempt_id": attempt_id, "owner": owner})
                return {"cell_id": row["cell_id"], "spec": json.loads(row["spec"]), "token": token,
                        "ordinal": ordinal, "attempt_id": attempt_id, "owner": owner}
        return None

    @staticmethod
    def _owned(db, cell_id, owner, token):
        row = db.execute("SELECT * FROM cells WHERE cell_id=?", (cell_id,)).fetchone()
        if row is None or row["owner"] != owner or row["token"] != token or row["status"] not in ("leased", "running"):
            raise FenceError(f"stale lease for {cell_id} token {token}")
        return row

    def mark_started(self, cell_id, owner, token, start_sha256, lease_seconds=900.0):
        """Record that the durable start record exists; only then may the episode dispatch."""
        with self.transaction() as db:
            row = self._owned(db, cell_id, owner, token)
            if row["status"] != "leased":
                raise FenceError("attempt already started")
            now = time.time()
            db.execute("UPDATE cells SET status='running',lease_expiry=?,updated=? WHERE cell_id=?",
                       (now + lease_seconds, now, cell_id))
            db.execute("UPDATE attempts SET status='running',started=?,start_sha256=? WHERE attempt_id=?",
                       (now, start_sha256, row["attempt_id"]))
            self._event(db, cell_id, "started", {"attempt_id": row["attempt_id"], "start_sha256": start_sha256})

    def heartbeat(self, cell_id, owner, token, lease_seconds=900.0):
        with self.transaction() as db:
            self._owned(db, cell_id, owner, token)
            now = time.time()
            db.execute("UPDATE cells SET lease_expiry=?,updated=? WHERE cell_id=?", (now + lease_seconds, now, cell_id))

    @staticmethod
    def _interrupted(db, cell_id):
        return db.execute("SELECT count(*) FROM attempts WHERE cell_id=? AND status IN "
                          "('infra_interrupted','abandoned')", (cell_id,)).fetchone()[0]

    def _requeue(self, db, cell_id, error_code, now):
        """Return an interrupted cell to the BACK of its segment, or block it at the attempt cap.

        Requeueing at the back of the (block, seed) segment spreads an outage over
        many cells, so the runner's consecutive-failure breaker trips before any
        single cell exhausts its cap, while D10.8's seed and block order holds.
        """
        if self._interrupted(db, cell_id) >= self.max_attempts:
            db.execute("UPDATE cells SET status='blocked',owner=NULL,lease_expiry=NULL,error_code=?,updated=? "
                       "WHERE cell_id=?", ("infra_attempts_exhausted", now, cell_id))
            self._event(db, cell_id, "blocked", {"error_code": "infra_attempts_exhausted", "last": error_code})
        else:
            position = db.execute("SELECT MAX(position) FROM cells").fetchone()[0] + 1
            db.execute("UPDATE cells SET status='pending',owner=NULL,lease_expiry=NULL,error_code=?,position=?,"
                       "updated=? WHERE cell_id=?", (error_code, position, now, cell_id))

    def finish(self, cell_id, owner, token, *, exit_status, result_sha256, error_code=None):
        """Close the running attempt from its immutable result.

        A scientific exit completes the cell forever. InfraInterrupted keeps the
        attempt and returns the cell to the queue for a new attempt id.
        """
        with self.transaction() as db:
            row = self._owned(db, cell_id, owner, token)
            if row["status"] != "running":
                raise FenceError("only a started attempt can finish")
            now = time.time()
            if exit_status in COMPLETED_EXITS:
                db.execute("UPDATE cells SET status='completed',owner=NULL,lease_expiry=NULL,result_sha256=?,"
                           "exit_status=?,error_code=NULL,updated=? WHERE cell_id=?",
                           (result_sha256, exit_status, now, cell_id))
                status = "completed"
            elif exit_status == INFRA_EXIT:
                status = "infra_interrupted"
            else:
                raise ValueError("unknown exit status " + repr(exit_status))
            db.execute("UPDATE attempts SET status=?,ended=?,result_sha256=?,exit_status=?,error_code=? "
                       "WHERE attempt_id=?", (status, now, result_sha256, exit_status, error_code, row["attempt_id"]))
            if status == "infra_interrupted":
                self._requeue(db, cell_id, error_code or "infra_interrupted", now)
            self._event(db, cell_id, status, {"attempt_id": row["attempt_id"], "exit_status": exit_status,
                                              "result_sha256": result_sha256, "error_code": error_code})
            return status

    def release_unstarted(self, cell_id, owner, token, reason):
        """The durable start record was never written: nothing was dispatched."""
        with self.transaction() as db:
            row = self._owned(db, cell_id, owner, token)
            if row["status"] != "leased":
                raise FenceError("a started attempt cannot be released as unstarted")
            now = time.time()
            db.execute("UPDATE attempts SET status='not_dispatched',ended=?,error_code=? WHERE attempt_id=?",
                       (now, reason, row["attempt_id"]))
            db.execute("UPDATE cells SET status='pending',owner=NULL,lease_expiry=NULL,error_code=?,updated=? "
                       "WHERE cell_id=?", (reason, now, cell_id))
            self._event(db, cell_id, "not_dispatched", {"attempt_id": row["attempt_id"], "reason": reason})

    def block(self, cell_id, error_code, owner=None, token=None):
        """Block a cell that must not run (for example payload integrity failure)."""
        if not error_code:
            raise ValueError("blocking requires an error code")
        with self.transaction() as db:
            now = time.time()
            if owner is not None:
                row = self._owned(db, cell_id, owner, token)
                if row["status"] != "leased":
                    raise FenceError("a dispatched attempt cannot be blocked")
                db.execute("UPDATE attempts SET status='not_dispatched',ended=?,error_code=? WHERE attempt_id=?",
                           (now, error_code, row["attempt_id"]))
            else:
                row = db.execute("SELECT status FROM cells WHERE cell_id=?", (cell_id,)).fetchone()
                if row is None or row["status"] != "pending":
                    raise ValueError("only pending cells can be blocked without a lease")
            db.execute("UPDATE cells SET status='blocked',owner=NULL,lease_expiry=NULL,error_code=?,updated=? "
                       "WHERE cell_id=?", (error_code, now, cell_id))
            self._event(db, cell_id, "blocked", {"error_code": error_code})

    def unblock(self, cell_id, reason):
        if not reason or not reason.strip():
            raise ValueError("a documented reason is required")
        with self.transaction() as db:
            row = db.execute("SELECT status FROM cells WHERE cell_id=?", (cell_id,)).fetchone()
            if row is None or row["status"] != "blocked":
                raise ValueError("only blocked cells can be unblocked")
            db.execute("UPDATE cells SET status='pending',error_code=NULL,updated=? WHERE cell_id=?",
                       (time.time(), cell_id))
            self._event(db, cell_id, "unblocked", {"reason": reason})

    def complete_from_durable(self, cell_id, owner, token, *, attempt_id, exit_status, result_sha256):
        """A durable completed result already exists: close this lease without dispatching."""
        if exit_status not in COMPLETED_EXITS:
            raise ValueError("only a completed durable result can complete a cell")
        with self.transaction() as db:
            row = self._owned(db, cell_id, owner, token)
            if row["status"] != "leased":
                raise FenceError("a dispatched attempt cannot be reconciled")
            now = time.time()
            db.execute("UPDATE attempts SET status='not_dispatched',ended=?,error_code='durable_result_exists' "
                       "WHERE attempt_id=?", (now, row["attempt_id"]))
            db.execute("UPDATE cells SET status='completed',owner=NULL,lease_expiry=NULL,result_sha256=?,"
                       "exit_status=?,attempt_id=?,updated=? WHERE cell_id=?",
                       (result_sha256, exit_status, attempt_id, now, cell_id))
            self._event(db, cell_id, "completed_from_durable", {"attempt_id": attempt_id})

    # -- recovery -------------------------------------------------------------

    def recover_expired(self, inspect=inspect_identity, now=None, on_abandon=None):
        """Fence and requeue expired leases whose owner is verified gone. Kills nothing."""
        now = time.time() if now is None else now
        outcomes, abandoned = [], []
        with self.transaction() as db:
            for row in db.execute("SELECT * FROM cells WHERE status IN ('leased','running') AND lease_expiry<?",
                                  (now,)).fetchall():
                identity = json.loads(row["identity"]) if row["identity"] else None
                state = inspect(identity)
                if state != "gone":
                    outcomes.append({"cell_id": row["cell_id"], "recovery": "retained", "process_state": state})
                    continue
                attempt = db.execute("SELECT * FROM attempts WHERE attempt_id=?", (row["attempt_id"],)).fetchone()
                dispatched = row["status"] == "running"
                status = "abandoned" if dispatched else "not_dispatched"
                # Fence first: a delayed old owner now fails every owned operation.
                db.execute("UPDATE cells SET token=token+1,updated=? WHERE cell_id=?", (now, row["cell_id"]))
                if attempt is not None:
                    db.execute("UPDATE attempts SET status=?,ended=?,error_code='ABANDONED_VERIFIED_GONE' "
                               "WHERE attempt_id=?", (status, now, row["attempt_id"]))
                self._requeue(db, row["cell_id"], "ABANDONED_VERIFIED_GONE", now)
                self._event(db, row["cell_id"], "recovered_gone",
                            {"attempt_id": row["attempt_id"], "old_token": row["token"], "dispatched": dispatched})
                outcomes.append({"cell_id": row["cell_id"], "recovery": status, "process_state": state,
                                 "attempt_id": row["attempt_id"]})
                abandoned.append((row["cell_id"], row["attempt_id"], identity))
        for cell_id, attempt_id, identity in abandoned:
            if on_abandon is not None:
                on_abandon(cell_id, attempt_id, identity)
        return outcomes

    def reconcile(self, store, inspect=inspect_identity, now=None):
        """Rebuild or repair ledger state from the durable records. Safe with live runners.

        Durable evidence wins and every change is monotonic: missing cells and
        attempt rows are added; token and attempt counters rise to the durable
        maxima (start-less attempt directories included, so attempt ids are never
        reused); a durable completed result completes the cell; a start-only
        attempt whose owner is verified gone is closed durably
        (attempt_closed.json) and its cell is fenced and requeued; a start-only
        attempt whose owner may be alive keeps (or regains) its lease. A
        completed cell is never reopened.
        """
        now = time.time() if now is None else now
        cells = store.load_cells()
        self.enqueue(cells)
        scanned = store.scan([cell["cell_id"] for cell in cells])
        report = {"cells": len(cells), "attempts_seen": 0, "inserted_attempts": 0, "updated_attempts": 0,
                  "completed_from_durable": 0, "closed_gone": 0, "live_retained": 0,
                  "requeued_gone": 0, "duplicate_completed": []}
        closures = []
        with self.transaction() as db:
            for cell_id, attempts in scanned.items():
                if not attempts:
                    continue
                report["attempts_seen"] += len(attempts)
                row = db.execute("SELECT * FROM cells WHERE cell_id=?", (cell_id,)).fetchone()
                db.execute("UPDATE cells SET attempts=MAX(attempts,?),token=MAX(token,?),updated=? WHERE cell_id=?",
                           (max(a["ordinal"] for a in attempts), max(a.get("token") or 0 for a in attempts),
                            now, cell_id))
                completed = [a for a in attempts if a["state"] == "completed"]
                if len(completed) > 1:
                    report["duplicate_completed"].append(cell_id)
                live, gone_ids = [], set()
                for attempt in attempts:
                    state = attempt["state"]
                    if state == "invalid_result":
                        raise LedgerInconsistency("result with an unknown exit status in " + attempt["attempt_id"])
                    if state == "missing_start":
                        status = "not_dispatched"
                    elif state == "started":
                        owner_identity = attempt["start"].get("owner_identity")
                        if inspect(owner_identity) == "gone":
                            closures.append((cell_id, attempt["attempt_id"], owner_identity))
                            gone_ids.add(attempt["attempt_id"])
                            status = "abandoned"
                        else:
                            live.append(attempt)
                            status = "running"
                    else:
                        status = state
                    outcome = self._upsert_attempt(db, cell_id, attempt, status, now)
                    if outcome:
                        report[outcome + "_attempts"] += 1
                if completed:
                    if row["status"] != "completed":
                        first = completed[0]
                        db.execute("UPDATE cells SET status='completed',owner=NULL,lease_expiry=NULL,attempt_id=?,"
                                   "result_sha256=?,exit_status=?,error_code=NULL,updated=? WHERE cell_id=?",
                                   (first["attempt_id"], first["result_sha256"], first["result"]["exit_status"], now,
                                    cell_id))
                        self._event(db, cell_id, "completed_from_durable", {"attempt_id": first["attempt_id"]})
                        report["completed_from_durable"] += 1
                elif row["status"] in ("leased", "running") and row["attempt_id"] in gone_ids:
                    db.execute("UPDATE cells SET token=token+1,updated=? WHERE cell_id=?", (now, cell_id))
                    self._requeue(db, cell_id, "ABANDONED_VERIFIED_GONE", now)
                    self._event(db, cell_id, "recovered_gone", {"attempt_id": row["attempt_id"], "via": "reconcile"})
                    report["requeued_gone"] += 1
                elif live and row["status"] == "pending":
                    attempt = live[-1]
                    db.execute("UPDATE cells SET status='running',owner=?,identity=?,token=?,attempt_id=?,"
                               "lease_expiry=?,updated=? WHERE cell_id=?",
                               (attempt["start"].get("owner"),
                                canonical_json(attempt["start"].get("owner_identity")).decode(),
                                attempt["token"], attempt["attempt_id"], now + 60, now, cell_id))
                    self._event(db, cell_id, "live_attempt_retained", {"attempt_id": attempt["attempt_id"]})
                    report["live_retained"] += 1
                elif row["status"] == "pending" and self._interrupted(db, cell_id) >= self.max_attempts:
                    self._requeue(db, cell_id, "rebuilt_from_durable", now)
            self._event(db, None, "reconciled", report)
        for cell_id, attempt_id, identity in closures:
            if store.close_attempt(cell_id, attempt_id, {
                    "protocol": LEDGER_PROTOCOL, "status": "abandoned",
                    "reason": "owner_verified_gone_without_result", "owner_identity": identity,
                    "closed_unix": time.time()}) is not None:
                report["closed_gone"] += 1
        return report

    def _upsert_attempt(self, db, cell_id, attempt, status, now):
        start, result = attempt.get("start") or {}, attempt.get("result") or {}
        existing = db.execute("SELECT status FROM attempts WHERE attempt_id=?", (attempt["attempt_id"],)).fetchone()
        if existing is None:
            started = start.get("started_unix")
            db.execute("INSERT INTO attempts(attempt_id,cell_id,ordinal,token,owner,identity,claimed,started,ended,"
                       "status,start_sha256,result_sha256,exit_status,error_code) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                       (attempt["attempt_id"], cell_id, attempt["ordinal"], attempt.get("token") or 0,
                        start.get("owner", "durable-unknown"),
                        canonical_json(start.get("owner_identity")).decode(), started or now, started,
                        None if status == "running" else now, status, attempt.get("start_sha256"),
                        attempt.get("result_sha256"), result.get("exit_status"), "rebuilt_from_durable"))
            return "inserted"
        if (existing["status"] != status and existing["status"] != "completed"
                and status in ("completed", "infra_interrupted", "abandoned")):
            db.execute("UPDATE attempts SET status=?,ended=COALESCE(ended,?),result_sha256=?,exit_status=? "
                       "WHERE attempt_id=?", (status, now, attempt.get("result_sha256"), result.get("exit_status"),
                                              attempt["attempt_id"]))
            return "updated"
        return None

    # -- inspection -----------------------------------------------------------

    def cell(self, cell_id):
        with self.connection() as db:
            return self._row(db.execute("SELECT * FROM cells WHERE cell_id=?", (cell_id,)).fetchone())

    def attempts(self, cell_id=None):
        with self.connection() as db:
            if cell_id is None:
                rows = db.execute("SELECT * FROM attempts ORDER BY cell_id,ordinal").fetchall()
            else:
                rows = db.execute("SELECT * FROM attempts WHERE cell_id=? ORDER BY ordinal", (cell_id,)).fetchall()
            return [self._row(row) for row in rows]

    def has_open_cells(self, eligible=None):
        """True while some (eligible) cell is pending, leased or running: the selection is not finished."""
        with self.connection() as db:
            cursor = db.execute("SELECT cell_id FROM cells WHERE status IN ('pending','leased','running')")
            try:
                for (cell_id,) in cursor:
                    if eligible is None or cell_id in eligible:
                        return True
            finally:
                cursor.close()
        return False

    def open_cells_before(self, rank):
        """[(cell_id, segment key)] of pending, leased or running cells in segments ranked before ``rank``."""
        with self.connection() as db:
            keys = {row["rank"]: row["key"] for row in db.execute("SELECT key, rank FROM segments")}
            rows = db.execute("SELECT cell_id, segment FROM cells WHERE status IN ('pending','leased','running') "
                              "AND segment<? ORDER BY segment, position", (rank,)).fetchall()
        return [(row["cell_id"], keys.get(row["segment"])) for row in rows]

    def segment_rank(self, key):
        with self.connection() as db:
            row = db.execute("SELECT rank FROM segments WHERE key=?", (key,)).fetchone()
        return None if row is None else row[0]

    def segment_states(self):
        """[{"rank", "key", "statuses": {status: n}}] per (block, seed) segment, in queue order."""
        with self.connection() as db:
            keys = {row["rank"]: row["key"] for row in db.execute("SELECT key, rank FROM segments")}
            counts = {}
            for row in db.execute("SELECT segment, status, count(*) AS n FROM cells GROUP BY segment, status"):
                counts.setdefault(row["segment"], {})[row["status"]] = row["n"]
        return [{"rank": rank, "key": keys.get(rank), "statuses": dict(sorted(counts[rank].items()))}
                for rank in sorted(counts)]

    def counts(self):
        with self.connection() as db:
            cells = {row["status"]: row["n"] for row in db.execute("SELECT status,count(*) AS n FROM cells GROUP BY status")}
            attempts = {row["status"]: row["n"] for row in
                        db.execute("SELECT status,count(*) AS n FROM attempts GROUP BY status")}
        return {"cells": cells, "attempts": attempts}

    def snapshot(self):
        with self.connection() as db:
            cells = [self._row(row) for row in db.execute("SELECT * FROM cells ORDER BY segment, position")]
            attempts = [self._row(row) for row in db.execute("SELECT * FROM attempts ORDER BY cell_id,ordinal")]
        return {"protocol": LEDGER_PROTOCOL, "run_id": self.run_id, "timestamp": time.time(),
                "counts": self.counts(), "cells": cells, "attempts": attempts}

    def active_leases(self, owner_prefix):
        with self.connection() as db:
            return [self._row(row) for row in db.execute(
                "SELECT * FROM cells WHERE status IN ('leased','running') AND owner LIKE ?", (owner_prefix + "%",))]
