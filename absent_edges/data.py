"""Pinned public acquisition into a private evaluator store; explicit agent allowlist.

Raw records contain solutions and hidden tests. Never log them, mount this store in
an agent container, or pass a raw record to a model. Acquisition requires a
pre-existing POSIX directory owned by the controller with mode 0700.
"""
from __future__ import annotations

import hashlib
from datetime import date, datetime
import json
import os
from pathlib import Path
import re
import stat
import urllib.request
import uuid

DATASET_ID = "SWE-bench-Live/SWE-bench-Live"
DATASET_REVISION = "b51a86422e10cfd403beb4773e5a2947953e36ec"
HARNESS_REVISION = "a145aa87c62361532dacaa243398978164b234b7"
AGENT_FIELDS = ("instance_id", "repo", "base_commit", "problem_statement")
PREVIEW_EXPOSED_IDS = frozenset({"reata__sqllineage-524", "privacyidea__privacyidea-3852",
                                "matplotlib__matplotlib-27613"})
# A rendered publisher page exposed an incomplete fourth row; exclude its repo.
PREVIEW_EXPOSED_REPOS = frozenset({"geopandas/geopandas"})
SPLIT_FILES = {name: [f"data/{name}-00000-of-00001.parquet"]
               for name in ("lite", "test", "verified")}
SPLIT_FILES["full"] = [f"data/full-{i:05d}-of-00002.parquet" for i in range(2)]


def canonical_json(value) -> bytes:
    def encode_timestamp(item):
        if isinstance(item, (datetime, date)):
            return item.isoformat()
        raise TypeError(f"Unsupported metadata type: {type(item).__name__}")
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False,
                      default=encode_timestamp).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_private_root(path) -> Path:
    root = Path(path)
    if os.name != "posix":
        raise RuntimeError("Private benchmark acquisition/grading requires POSIX permissions")
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ValueError("Private root must be an existing absolute nonsymlink directory")
    info = root.stat()
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise PermissionError("Private root must be owned by this user with mode 0700")
    # Resolve parent aliases so callers cannot accidentally use two roots.
    if root.resolve() != root:
        raise ValueError("Private root must be a canonical path")
    return root


def private_directory(root: Path, *parts: str) -> Path:
    path = root
    for part in parts:
        if not part or part in (".", "..") or "/" in part or "\\" in part:
            raise ValueError("Invalid private path component")
        path = path / part
        if path.is_symlink():
            raise ValueError("Private store cannot contain symlinks")
        path.mkdir(mode=0o700, exist_ok=True)
        if os.name == "posix" and (path.stat().st_uid != os.getuid()
                                  or stat.S_IMODE(path.stat().st_mode) != 0o700):
            raise PermissionError("Private subdirectory permissions are not 0700")
    return path


def write_private_json(path: Path, value) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError("Refusing to overwrite evaluator evidence")
    encoded = canonical_json(value) + b"\n"
    temporary = path.with_name(path.name + ".partial-" + uuid.uuid4().hex)
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # atomic no-overwrite publication
    finally:
        temporary.unlink(missing_ok=True)


def _read_private_json(root: Path, path: Path):
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError("Private input escaped private root")
    info = path.stat()
    if os.name == "posix" and (info.st_uid != os.getuid()
                              or stat.S_IMODE(info.st_mode) & 0o077):
        raise PermissionError("Private input has group/other access")
    return json.loads(path.read_text(encoding="utf-8"))


def export_agent_view(task: dict) -> dict[str, str]:
    """Only the statement is a prompt; the other fields are controller metadata."""
    result = {}
    for key in AGENT_FIELDS:
        if not isinstance(task.get(key), str) or not task[key]:
            raise ValueError(f"Missing or invalid allowed task field: {key}")
        result[key] = task[key]
    if not re.fullmatch(r"[a-fA-F0-9]{40}", result["base_commit"]):
        raise ValueError("Expected a full base commit SHA")
    if (not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", result["repo"])
            or any(part in (".", "..") for part in result["repo"].split("/"))):
        raise ValueError("Invalid repository identity")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", result["instance_id"]):
        raise ValueError("Invalid instance identity")
    return result


def select_development_tasks(tasks, count: int = 1, seed: str = "absent-edges-D-v1",
                             preferred_id: str | None = "amoffat__sh-744") -> list[str]:
    """Pre-outcome selection using safe metadata only; official smoke task first.

    Prefer distinct repository families, then fill deterministically. No patch,
    test, difficulty, or evaluator label influences this selection.
    """
    if count < 1:
        raise ValueError("count must be positive")
    views = [export_agent_view(task) for task in tasks]
    if len({v["instance_id"] for v in views}) != len(views):
        raise ValueError("Duplicate instance identities")
    views = [v for v in views if v["instance_id"] not in PREVIEW_EXPOSED_IDS
             and v["repo"] not in PREVIEW_EXPOSED_REPOS]
    views.sort(key=lambda v: hashlib.sha256(
        (seed + "\0" + v["instance_id"]).encode()).hexdigest())
    preferred = [v for v in views if v["instance_id"] == preferred_id]
    ordered = preferred + [v for v in views if v["instance_id"] != preferred_id]
    selected, deferred, repos = [], [], set()
    for view in ordered:
        if view["repo"] in repos:
            deferred.append(view["instance_id"])
        else:
            selected.append(view["instance_id"])
            repos.add(view["repo"])
    selected += deferred
    if len(selected) < count:
        raise ValueError("Insufficient eligible tasks")
    return selected[:count]


def stage_live_dataset(private_root, split: str = "lite",
                       revision: str = DATASET_REVISION) -> Path:
    """Anonymous pinned download. The returned JSON is evaluator-only."""
    root = require_private_root(private_root)
    if revision != DATASET_REVISION or split not in SPLIT_FILES:
        raise ValueError("Dataset revision/split is outside the audited pin")
    target = private_directory(root, "datasets", revision, split)
    destination = target / "records.json"
    manifest_path = target / "manifest.json"
    if destination.exists() and manifest_path.exists():
        manifest = _read_private_json(root, manifest_path)
        if (manifest["revision"] != revision or manifest["split"] != split
                or sha256_file(destination) != manifest["records_sha256"]):
            raise ValueError("Staged dataset does not match immutable manifest")
        return destination
    import pyarrow.parquet as pq
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    shards, records = [], []
    for remote_name in SPLIT_FILES[split]:
        local = target / Path(remote_name).name
        if not local.exists():
            temporary = target / (local.name + ".partial-" + uuid.uuid4().hex)
            url = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{revision}/{remote_name}"
            request = urllib.request.Request(url, headers={"User-Agent": "AbsentEdges-source-acquisition/1"})
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                with os.fdopen(fd, "wb") as stream, opener.open(request, timeout=120) as response:
                    while chunk := response.read(1024 * 1024):
                        stream.write(chunk)
                temporary.replace(local)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        elif local.is_symlink() or stat.S_IMODE(local.stat().st_mode) & 0o077:
            raise PermissionError("Unsafe existing dataset shard")
        shards.append({"file": remote_name, "sha256": sha256_file(local), "bytes": local.stat().st_size})
        records.extend(pq.read_table(local).to_pylist())
    for record in records:
        export_agent_view(record)
    if len({r["instance_id"] for r in records}) != len(records):
        raise ValueError("Duplicate dataset instance identities")
    if destination.exists():
        if sha256_file(destination) != hashlib.sha256(canonical_json(records) + b"\n").hexdigest():
            raise ValueError("Uncommitted records differ from pinned shards; preserve for investigation")
    else:
        write_private_json(destination, records)
    write_private_json(manifest_path, {"dataset": DATASET_ID, "revision": revision,
                       "split": split, "rows": len(records), "shards": shards,
                       "records_sha256": sha256_file(destination)})
    return destination


def load_private_tasks(private_root, dataset_path) -> list[dict]:
    root = require_private_root(private_root)
    path = Path(dataset_path)
    manifest = _read_private_json(root, path.parent / "manifest.json")
    if manifest.get("revision") != DATASET_REVISION or sha256_file(path) != manifest["records_sha256"]:
        raise ValueError("Dataset provenance verification failed")
    tasks = _read_private_json(root, path)
    if not isinstance(tasks, list) or len(tasks) != manifest["rows"]:
        raise ValueError("Invalid private dataset")
    return tasks
