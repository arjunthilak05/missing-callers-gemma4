"""Transactional, local-disk job ledger. No lease is released merely by time.

The supervisor is a trusted host controller; benchmark repositories must never
have access to this database or its configuration. SQLite WAL is not supported
on network filesystems. All timestamps are Unix seconds in UTC.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import time
from typing import Any, Iterator


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: str | Path, value: Any) -> None:
    """Publish complete JSON; callers choose unique paths for immutable results."""
    import os
    import uuid
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with tmp.open("x", encoding="utf-8") as stream:
            stream.write(canonical_json(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, target)
        if os.name == "posix":
            fd = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    finally:
        tmp.unlink(missing_ok=True)


@dataclass(frozen=True)
class JobSpec:
    kind: str
    profile_hash: str
    params: dict[str, str] = field(default_factory=dict)
    dependencies: tuple[str, ...] = ()
    stage: str = "bootstrap"
    split: str = "D"
    input_hashes: dict[str, str] = field(default_factory=dict)
    model_identity: str = "none"
    gpu_uuids: tuple[str, ...] = ()
    cpu_cores: int = 1
    ram_bytes: int = 0
    disk_bytes: int = 0
    priority: int = 0
    protocol_hash: str = "none"

    def normalized(self) -> dict[str, Any]:
        obj = asdict(self)
        if not self.kind or not isinstance(self.kind, str):
            raise ValueError("kind must be a nonempty string")
        if len(self.profile_hash) != 64 or any(c not in "0123456789abcdef" for c in self.profile_hash):
            raise ValueError("profile_hash must be a SHA256")
        for name in ("cpu_cores", "ram_bytes", "disk_bytes", "priority"):
            value = obj[name]
            if type(value) is not int or (name != "priority" and value < (1 if name == "cpu_cores" else 0)):
                raise ValueError(f"invalid {name}")
        if any(not isinstance(k, str) or not isinstance(v, str) for k, v in self.params.items()):
            raise ValueError("params must contain only string keys and values")
        if len(set(self.gpu_uuids)) != len(self.gpu_uuids):
            raise ValueError("duplicate GPU UUID")
        if any(not isinstance(g, str) or not g.startswith("GPU-") for g in self.gpu_uuids):
            raise ValueError("only full GPU UUID identifiers are supported")
        obj["gpu_uuids"] = sorted(self.gpu_uuids)
        obj["dependencies"] = sorted(set(self.dependencies))
        canonical_json(obj)
        return obj

    @property
    def job_id(self) -> str:
        return "job_" + digest(self.normalized())

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> "JobSpec":
        obj = dict(obj)
        for key in ("dependencies", "gpu_uuids"):
            if key in obj:
                obj[key] = tuple(obj[key])
        result = cls(**obj)
        result.normalized()
        return result


class FenceError(RuntimeError):
    """A worker has lost ownership; it must not launch or commit anything."""


class Ledger:
    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY, spec TEXT NOT NULL, status TEXT NOT NULL,
                    priority INTEGER NOT NULL, created REAL NOT NULL, updated REAL NOT NULL,
                    attempt INTEGER NOT NULL DEFAULT 0, token INTEGER NOT NULL DEFAULT 0,
                    owner TEXT, lease_expiry REAL, process TEXT, artifact TEXT,
                    output_hash TEXT, error_code TEXT
                );
                CREATE TABLE IF NOT EXISTS attempts (
                    job_id TEXT NOT NULL, attempt INTEGER NOT NULL, token INTEGER NOT NULL,
                    owner TEXT NOT NULL, started REAL NOT NULL, ended REAL, status TEXT NOT NULL,
                    process TEXT, artifact TEXT, output_hash TEXT, error_code TEXT,
                    PRIMARY KEY(job_id, attempt)
                );
                CREATE TABLE IF NOT EXISTS gpu_leases (
                    uuid TEXT PRIMARY KEY, job_id TEXT NOT NULL, token INTEGER NOT NULL,
                    owner TEXT NOT NULL, expires REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, timestamp REAL NOT NULL,
                    job_id TEXT, event TEXT NOT NULL, detail TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS job_status ON jobs(status, priority, created);
            """)

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA busy_timeout=30000")
        db.execute("PRAGMA synchronous=FULL")
        try:
            yield db
        finally:
            db.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                yield db
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise

    @staticmethod
    def _event(db: sqlite3.Connection, job_id: str | None, event: str, detail: Any = None) -> None:
        db.execute("INSERT INTO events(timestamp,job_id,event,detail) VALUES(?,?,?,?)",
                   (time.time(), job_id, event, canonical_json(detail)))

    @staticmethod
    def _record(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        result = dict(row)
        for key in ("spec", "process"):
            if key in result and result[key]:
                result[key] = json.loads(result[key])
        return result

    def enqueue(self, spec: JobSpec) -> str:
        encoded = canonical_json(spec.normalized())
        job_id = spec.job_id
        now = time.time()
        with self.transaction() as db:
            old = db.execute("SELECT spec FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if old:
                if old["spec"] != encoded:
                    raise RuntimeError("hash collision or ledger corruption")
                return job_id
            for dep in spec.dependencies:
                if not db.execute("SELECT 1 FROM jobs WHERE job_id=?", (dep,)).fetchone():
                    raise ValueError(f"unknown dependency: {dep}; enqueue prerequisites first")
            db.execute("INSERT INTO jobs(job_id,spec,status,priority,created,updated) VALUES(?,?,?,?,?,?)",
                       (job_id, encoded, "pending", spec.priority, now, now))
            self._event(db, job_id, "enqueued")
        return job_id

    def get(self, job_id: str) -> dict[str, Any]:
        with self.connection() as db:
            result = self._record(db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone())
        if result is None:
            raise KeyError(job_id)
        return result

    def list_jobs(self) -> list[dict[str, Any]]:
        with self.connection() as db:
            return [self._record(r) for r in db.execute("SELECT * FROM jobs ORDER BY created,job_id")]

    def attempts(self, job_id: str) -> list[dict[str, Any]]:
        with self.connection() as db:
            return [self._record(r) for r in db.execute("SELECT * FROM attempts WHERE job_id=? ORDER BY attempt", (job_id,))]

    @staticmethod
    def passing_artifact(row: dict[str, Any] | sqlite3.Row) -> bool:
        """Check bytes and execution evidence, never trust a status sentence alone."""
        try:
            path = Path(row["artifact"])
            if file_hash(path) != row["output_hash"]:
                return False
            obj = json.loads(path.read_text(encoding="utf-8"))
            if not (obj["job_id"] == row["job_id"] and obj["token"] == row["token"]
                    and obj["passed"] is True and type(obj["exit_code"]) is int and obj["exit_code"] == 0):
                return False
            evidence = obj["evidence"]
            return bool(evidence) and all(file_hash(item["path"]) == item["sha256"] for item in evidence)
        except (KeyError, ValueError, TypeError, OSError):
            return False

    def _promote(self, db: sqlite3.Connection) -> None:
        for row in db.execute("SELECT * FROM jobs WHERE status IN ('pending','ready')").fetchall():
            specs = json.loads(row["spec"])
            deps = [db.execute("SELECT * FROM jobs WHERE job_id=?", (d,)).fetchone() for d in specs["dependencies"]]
            state = "ready" if all(d is not None and d["status"] == "completed" and self.passing_artifact(d) for d in deps) else "pending"
            if state != row["status"]:
                db.execute("UPDATE jobs SET status=?,updated=? WHERE job_id=?", (state, time.time(), row["job_id"]))
                self._event(db, row["job_id"], state)

    def claim(self, owner: str, available_gpus: set[str] | frozenset[str] = frozenset(),
              allowed_gpus: set[str] | frozenset[str] = frozenset(), lease_seconds: float = 86400,
              free_disk_bytes: int = 2**63 - 1, disk_reserve_bytes: int = 0,
              cpu_capacity: int | None = None, ram_capacity: int | None = None,
              eligible_job_ids: set[str] | None = None) -> dict[str, Any] | None:
        if not owner or not math.isfinite(lease_seconds) or lease_seconds <= 0:
            raise ValueError("owner and positive finite lease required")
        now = time.time()
        with self.transaction() as db:
            self._promote(db)
            active = [json.loads(r[0]) for r in db.execute("SELECT spec FROM jobs WHERE status IN ('leased','running')")]
            reserved_disk = sum(s["disk_bytes"] for s in active)
            reserved_cpu = sum(s["cpu_cores"] for s in active)
            reserved_ram = sum(s["ram_bytes"] for s in active)
            for row in db.execute("SELECT * FROM jobs WHERE status='ready' ORDER BY priority DESC,created,job_id").fetchall():
                if eligible_job_ids is not None and row["job_id"] not in eligible_job_ids:
                    continue
                spec = json.loads(row["spec"])
                uuids = set(spec["gpu_uuids"])
                if not uuids <= allowed_gpus or not uuids <= available_gpus:
                    continue
                if any(db.execute("SELECT 1 FROM gpu_leases WHERE uuid=?", (g,)).fetchone() for g in uuids):
                    continue
                if free_disk_bytes - reserved_disk - spec["disk_bytes"] < disk_reserve_bytes:
                    continue
                if cpu_capacity is not None and reserved_cpu + spec["cpu_cores"] > cpu_capacity:
                    continue
                if ram_capacity is not None and reserved_ram + spec["ram_bytes"] > ram_capacity:
                    continue
                token, attempt = row["token"] + 1, row["attempt"] + 1
                expiry = now + lease_seconds
                db.execute("UPDATE jobs SET status='leased',owner=?,token=?,attempt=?,lease_expiry=?,process=NULL,artifact=NULL,output_hash=NULL,error_code=NULL,updated=? WHERE job_id=?",
                           (owner, token, attempt, expiry, now, row["job_id"]))
                db.execute("INSERT INTO attempts(job_id,attempt,token,owner,started,status) VALUES(?,?,?,?,?,'leased')",
                           (row["job_id"], attempt, token, owner, now))
                for uuid in sorted(uuids):
                    db.execute("INSERT INTO gpu_leases VALUES(?,?,?,?,?)", (uuid, row["job_id"], token, owner, expiry))
                self._event(db, row["job_id"], "claimed", {"token": token, "owner": owner, "attempt": attempt})
                return self._record(db.execute("SELECT * FROM jobs WHERE job_id=?", (row["job_id"],)).fetchone())
        return None

    @staticmethod
    def _owned(db: sqlite3.Connection, job_id: str, owner: str, token: int) -> sqlite3.Row:
        row = db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None or row["owner"] != owner or row["token"] != token or row["status"] not in ("leased", "running"):
            raise FenceError(f"stale lease for {job_id} token {token}")
        return row

    def register_process(self, job_id: str, owner: str, token: int, identity: dict[str, Any], lease_seconds: float = 86400) -> None:
        if not {"pid", "pgid", "boot_id", "start_ticks"} <= identity.keys():
            raise ValueError("full process identity required")
        with self.transaction() as db:
            row = self._owned(db, job_id, owner, token)
            if row["process"] is not None:
                # Registration is a one-shot gate, even for a second wrapper
                # carrying the same token. Only this wrapper may launch payload.
                raise FenceError("worker already registered")
            now = time.time()
            db.execute("UPDATE jobs SET status='running',process=?,lease_expiry=?,updated=? WHERE job_id=?",
                       (canonical_json(identity), now + lease_seconds, now, job_id))
            db.execute("UPDATE attempts SET status='running',process=? WHERE job_id=? AND attempt=?",
                       (canonical_json(identity), job_id, row["attempt"]))
            db.execute("UPDATE gpu_leases SET expires=? WHERE job_id=? AND token=?", (now + lease_seconds, job_id, token))
            self._event(db, job_id, "worker_registered", identity)

    def heartbeat(self, job_id: str, owner: str, token: int, lease_seconds: float = 86400) -> None:
        with self.transaction() as db:
            self._owned(db, job_id, owner, token)
            now = time.time()
            db.execute("UPDATE jobs SET lease_expiry=?,updated=? WHERE job_id=?", (now + lease_seconds, now, job_id))
            db.execute("UPDATE gpu_leases SET expires=? WHERE job_id=? AND token=?", (now + lease_seconds, job_id, token))

    def finalize(self, job_id: str, owner: str, token: int, *, artifact: str | Path | None = None,
                 error_code: str | None = None) -> None:
        """Only the registered worker, after its payload exits, may finalize."""
        with self.transaction() as db:
            row = self._owned(db, job_id, owner, token)
            path = str(Path(artifact).resolve()) if artifact is not None else None
            output_hash = file_hash(path) if path else None
            state = "failed" if error_code else "completed"
            if not error_code:
                candidate = dict(row)
                candidate.update(artifact=path, output_hash=output_hash)
                if row["status"] != "running" or not self.passing_artifact(candidate):
                    raise ValueError("completion needs a registered worker and verified passing artifact")
            now = time.time()
            db.execute("UPDATE jobs SET status=?,artifact=?,output_hash=?,error_code=?,lease_expiry=NULL,updated=? WHERE job_id=?",
                       (state, path, output_hash, error_code, now, job_id))
            db.execute("UPDATE attempts SET status=?,ended=?,artifact=?,output_hash=?,error_code=? WHERE job_id=? AND attempt=?",
                       (state, now, path, output_hash, error_code, job_id, row["attempt"]))
            db.execute("DELETE FROM gpu_leases WHERE job_id=? AND token=?", (job_id, token))
            self._event(db, job_id, state, {"artifact": path, "sha256": output_hash, "error_code": error_code})

    def recover_expired(self, inspect_identity, now: float | None = None) -> list[dict[str, Any]]:
        """Inspect process AND process group while holding the claim transaction.

        Inspector returns gone, alive, orphan_group, or unknown. Anything except
        verified gone retains its lease. No process is killed by recovery.
        """
        now = time.time() if now is None else now
        outcomes = []
        with self.transaction() as db:
            for row in db.execute("SELECT * FROM jobs WHERE status IN ('leased','running') AND lease_expiry<?", (now,)).fetchall():
                identity = json.loads(row["process"]) if row["process"] else None
                state = inspect_identity(identity) if identity else "gone"
                if state != "gone":
                    outcomes.append({"job_id": row["job_id"], "recovery": "retained", "process_state": state})
                    continue
                # Fence first. A delayed old wrapper now fails registration.
                db.execute("UPDATE jobs SET status='failed',token=token+1,error_code='ABANDONED_VERIFIED_GONE',lease_expiry=NULL,updated=? WHERE job_id=?", (now, row["job_id"]))
                db.execute("UPDATE attempts SET status='failed',ended=?,error_code='ABANDONED_VERIFIED_GONE' WHERE job_id=? AND attempt=?", (now, row["job_id"], row["attempt"]))
                db.execute("DELETE FROM gpu_leases WHERE job_id=? AND token=?", (row["job_id"], row["token"]))
                self._event(db, row["job_id"], "recovered_gone", {"old_token": row["token"], "identity": identity})
                outcomes.append({"job_id": row["job_id"], "recovery": "failed", "process_state": state})
        return outcomes

    def retry(self, job_id: str, reason: str) -> None:
        if not reason.strip():
            raise ValueError("a documented retry reason is required")
        with self.transaction() as db:
            row = db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None or row["status"] not in ("failed", "blocked"):
                raise ValueError("only terminal unsuccessful jobs may be retried")
            db.execute("UPDATE jobs SET status='pending',owner=NULL,process=NULL,lease_expiry=NULL,artifact=NULL,output_hash=NULL,error_code=NULL,updated=? WHERE job_id=?", (time.time(), job_id))
            self._event(db, job_id, "retry_requested", {"reason": reason})

    def block(self, job_id: str, error_code: str) -> None:
        """Block only unlaunched work; never free an active allocation."""
        if not error_code:
            raise ValueError("blocking requires an explicit error code")
        with self.transaction() as db:
            row = db.execute("SELECT status FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None or row["status"] not in ("pending", "ready"):
                raise ValueError("only unlaunched jobs may be blocked")
            db.execute("UPDATE jobs SET status='blocked',error_code=?,updated=? WHERE job_id=?", (error_code, time.time(), job_id))
            self._event(db, job_id, "blocked", {"error_code": error_code})

    def snapshot(self) -> dict[str, Any]:
        with self.connection() as db:
            jobs = [self._record(r) for r in db.execute("SELECT * FROM jobs ORDER BY created,job_id")]
            leases = [dict(r) for r in db.execute("SELECT * FROM gpu_leases ORDER BY uuid")]
            counts = {r["status"]: r["n"] for r in db.execute("SELECT status,count(*) AS n FROM jobs GROUP BY status")}
            attempts = [dict(r) for r in db.execute("SELECT * FROM attempts")]
        now = time.time()
        gpu_hours = 0.0
        specs = {j["job_id"]: j["spec"] for j in jobs}
        for attempt in attempts:
            gpu_hours += max(0, (attempt["ended"] or now) - attempt["started"]) / 3600 * len(specs[attempt["job_id"]]["gpu_uuids"])
        return {"timestamp": now, "jobs": jobs, "gpu_leases": leases, "counts": counts,
                "allocated_gpu_hours": gpu_hours, "accounting_note": "lease walltime, not GPU utilization"}

    def cadence_due(self, name: str, interval: float, now: float | None = None) -> bool:
        """Persist report cadence across coordinator restarts and concurrent loops."""
        now = time.time() if now is None else now
        with self.transaction() as db:
            row = db.execute("SELECT value FROM metadata WHERE key=?", ("cadence:" + name,)).fetchone()
            if row is not None and now - float(row[0]) < interval:
                return False
            db.execute("INSERT OR REPLACE INTO metadata VALUES(?,?)", ("cadence:" + name, str(now)))
            return True
