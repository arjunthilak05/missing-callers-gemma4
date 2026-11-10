"""Exp A repository registry (lane R; docs/EXPA_DESIGN.md section 3).

Standard library only, so it runs under envs/controller (3.11), Python 3.13 and on Windows.

What this module provides:
- licence detection from the actual LICENSE/COPYING text (SPDX ids, permissive allow-list)
- the exclusion list: a scan of project files for repository families the project already used,
  plus a check of a candidate against it (by casefolded name/alias and by GitHub network root id)
- release-tag pinning rules (newest final release tag before the cutoff)
- pytest JUnit-XML parsing and the baseline status rule
- the registry record schema, validation, loading and deterministic role assignment

Nothing here touches the network or runs subprocesses; scripts/expA_repos.py does that.
"""
from __future__ import annotations

import dataclasses
import fnmatch
import hashlib
import json
import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator, Mapping, Sequence

SCHEMA = "expA-registry-v1"
EXCLUSIONS_SCHEMA = "expA-exclusions-v1"
CUTOFF = "2026-10-01T00:00:00Z"
EXPA_ROOT = "/mnt/kldata/absent-edges-20261018-core2/expA"
PYTHON_EXECUTABLE = "/usr/bin/python3.13"

STATUSES = ("ok", "tests_failed", "env_failed", "excluded")
ROLES = ("dev", "main", "spare")
# release_tag_before_cutoff: newest final release tag whose commit predates CUTOFF (the rule);
# default_branch_head_at_cutoff: no qualifying release tag; newest first-parent commit before CUTOFF;
# explicit_dev_tag: the coordinator-assigned tag of a dev family (dev families are not analysed).
PIN_RULES = ("release_tag_before_cutoff", "default_branch_head_at_cutoff", "explicit_dev_tag")
DEV_FAMILIES = ("more-itertools", "boltons")
MIN_MAIN = 22

# Baseline status rule (build parameter; restated in the prereg if it affects outcomes).
# "ok" needs a finished (not timed-out) run with at least MIN_PASSED passing tests and a pass
# fraction passed / (passed + failed + errors) of at least MIN_PASS_FRACTION.
MIN_PASSED = 20
MIN_PASS_FRACTION = 0.95
WALL_CAP_SECONDS = 1800

# Permissive licences accepted for Exp A (SPDX identifiers).
ALLOWED_LICENSES = frozenset({
    "MIT", "BSD-2-Clause", "BSD-3-Clause", "Apache-2.0", "ISC", "PSF-2.0",
})

_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_FAMILY = re.compile(r"^[a-z0-9][a-z0-9._-]{0,62}$")
_REPO = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]{0,38}/[A-Za-z0-9_.-]{1,100}$")
_URL = re.compile(r"^https://github\.com/([A-Za-z0-9][A-Za-z0-9-]{0,38}/[A-Za-z0-9_.-]{1,100})$")


class RegistryError(ValueError):
    """Raised when a registry document or record violates the schema."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(obj: Any) -> str:
    """Stable JSON text (sorted keys, 2-space indent, trailing newline)."""
    return json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def parse_utc(value: str) -> datetime:
    """Parse an ISO-8601 timestamp (``Z`` or offset) into an aware UTC datetime."""
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        raise ValueError(f"timestamp without timezone: {value!r}")
    return dt.astimezone(timezone.utc)


# ---------------------------------------------------------------------------------------------
# Licence detection
# ---------------------------------------------------------------------------------------------

_LICENSE_NAME = re.compile(r"^(licen[cs]e|copying|unlicense)([._-].*)?$", re.IGNORECASE)

# Each rule: (identifier, required phrases (all), forbidden phrases (none may appear)).
# Phrases are matched against normalised text (lowercase, punctuation-light, single spaces).
_NON_PERMISSIVE_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("AGPL", ("gnu affero general public license",)),
    ("LGPL", ("gnu lesser general public license",)),
    ("LGPL", ("gnu library general public license",)),
    ("GPL", ("gnu general public license",)),
    ("MPL", ("mozilla public license",)),
    ("EPL", ("eclipse public license",)),
    ("CDDL", ("common development and distribution license",)),
    ("EUPL", ("european union public licence",)),
    ("ZPL", ("zope public license",)),
    ("Repoze", ("repoze public license",)),
    ("Unlicense", ("this is free and unencumbered software released into the public domain",)),
    ("CC", ("creative commons",)),
)


def normalise_license_text(text: str) -> str:
    """Lowercase, drop comment markers/quotes and collapse whitespace so wrapped text matches."""
    lowered = text.lower().replace("“", '"').replace("”", '"').replace("’", "'")
    lowered = re.sub(r"^[ \t]*(#|//|\*|;|--)+", " ", lowered, flags=re.MULTILINE)
    lowered = re.sub(r"[`*_>|]", " ", lowered)
    lowered = lowered.replace('"', " ").replace("'", " ")
    return re.sub(r"\s+", " ", lowered).strip()


def _detect_permissive(norm: str) -> list[str]:
    found: list[str] = []
    if ("apache license" in norm and "version 2.0" in norm
            and ("terms and conditions for use, reproduction, and distribution" in norm
                 or "licensed under the apache license, version 2.0" in norm)):
        found.append("Apache-2.0")
    if "python software foundation license version 2" in norm or (
            "psf" in norm and "python software foundation" in norm
            and "license agreement" in norm and "python" in norm):
        found.append("PSF-2.0")
    if ("permission is hereby granted, free of charge, to any person obtaining a copy" in norm
            and "the software is provided as is" in norm):
        found.append("MIT")
    if (re.search(r"permission to use, copy, modify, (and/or|and) distribute this software for any "
                  r"purpose with or without fee is hereby granted", norm)
            and re.search(r"the software is provided as is and .{0,80}?disclaims? all warranties", norm)):
        found.append("ISC")
    if ("redistribution and use in source and binary forms, with or without modification, are "
            "permitted provided that the following conditions are met" in norm):
        if "all advertising materials mentioning features or use of this software" in norm:
            found.append("BSD-4-Clause")
        elif re.search(r"neither the names? of .{0,300}?(nor the names? of|may be used to endorse)", norm) or re.search(
                r"the names? of .{0,200}? may not be used to endorse or promote products", norm):
            found.append("BSD-3-Clause")
        else:
            found.append("BSD-2-Clause")
    return found


_DUAL_HINTS = ("either of the licenses", "either of the licences", "dual licensed", "dual-licensed",
               "at your option", "at your choice")


def detect_license_text(text: str) -> dict:
    """Classify one licence file text.

    Returns ``{"spdx": [...permissive ids...], "non_permissive": [...], "dual": bool, "unknown": bool}``.
    A file can carry several licences (for example dateutil's Apache-2.0 plus BSD-3-Clause text).
    Any GNU/MPL/EPL/... licence text is reported as non-permissive, even next to a permissive one.
    """
    norm = normalise_license_text(text)
    permissive = _detect_permissive(norm)
    non_permissive = sorted({ident for ident, phrases in _NON_PERMISSIVE_RULES
                             if all(p in norm for p in phrases)})
    return {"spdx": sorted(set(permissive)), "non_permissive": non_permissive,
            "dual": any(h in norm for h in _DUAL_HINTS),
            "unknown": not permissive and not non_permissive}


def find_license_files(repo_dir: str | os.PathLike) -> list[str]:
    """Licence files at the repository root and under LICENSES/ (sorted POSIX relpaths)."""
    root = Path(repo_dir)
    found = []
    for entry in sorted(root.iterdir(), key=lambda p: p.name):
        if entry.is_file() and _LICENSE_NAME.match(entry.name) and not entry.name.endswith((".py", ".pyc", ".pyi")):
            found.append(entry.name)
        elif entry.is_dir() and entry.name.casefold() in ("licenses", "licences", "license", "licence"):
            for sub in sorted(entry.iterdir(), key=lambda p: p.name):
                if sub.is_file():
                    found.append(f"{entry.name}/{sub.name}")
    return found


def detect_repo_license(repo_dir: str | os.PathLike) -> dict:
    """Detect the licence of a checkout from its licence files only (never from metadata).

    Result keys: ``license`` (SPDX expression or None), ``acceptable`` (bool), ``files`` (per-file
    detections with sha256) and ``reason``. Acceptable means at least one permissive licence from
    ALLOWED_LICENSES was found and no file carries a non-permissive licence.
    """
    root = Path(repo_dir)
    files = []
    permissive: list[str] = []
    non_permissive: list[str] = []
    dual = False
    for rel in find_license_files(root):
        data = (root / rel).read_bytes()
        det = detect_license_text(data.decode("utf-8", errors="replace"))
        files.append({"path": rel, "sha256": sha256_bytes(data), **det})
        permissive += det["spdx"]
        non_permissive += det["non_permissive"]
        dual = dual or det["dual"]
    permissive = sorted(set(permissive))
    non_permissive = sorted(set(non_permissive))
    if not files:
        return {"license": None, "acceptable": False, "files": [], "reason": "no licence file"}
    if non_permissive:
        return {"license": None, "acceptable": False, "files": files,
                "reason": "non-permissive licence text: " + ", ".join(non_permissive)}
    outside = sorted(set(permissive) - ALLOWED_LICENSES)
    if outside:
        return {"license": None, "acceptable": False, "files": files,
                "reason": "licence outside allow-list: " + ", ".join(outside)}
    allowed = [p for p in permissive if p in ALLOWED_LICENSES]
    if not allowed:
        return {"license": None, "acceptable": False, "files": files,
                "reason": "no recognised permissive licence text"}
    expression = allowed[0] if len(allowed) == 1 else (" OR " if dual else " AND ").join(allowed)
    return {"license": expression, "acceptable": True, "files": files, "reason": "ok"}


def license_expression_ok(expression: str | None) -> bool:
    if not expression:
        return False
    parts = re.split(r"\s+(?:AND|OR)\s+", expression.strip())
    return all(p in ALLOWED_LICENSES for p in parts)


# ---------------------------------------------------------------------------------------------
# Exclusion list: scan of project files
# ---------------------------------------------------------------------------------------------

# Default scan sources (relative to the project root); see docs/EXPA_REPOS.md.
EXCLUSION_SOURCES = (
    "manifests/**",
    "config/**",
    "configs/**",
    "tasks/BACKLOG.json",
    "reports/**/*.safe.json",
    "staged/research-catalogue-*/**",
    "staged/*catalogue*/**",
    "staged/*slot*/**",
)
# Never descend into these (our own outputs, caches).
_SKIP_DIRS = {"__pycache__", ".git", "expA"}
MAX_SCAN_BYTES = 32 << 20

_OWNER = r"[A-Za-z0-9][A-Za-z0-9-]{0,38}"
_NAME = r"[A-Za-z0-9_.-]{1,100}"
_RE_SWEBENCH_ID = re.compile(
    r"(?<![A-Za-z0-9_.-])(" + _OWNER + r")__([A-Za-z0-9][A-Za-z0-9_.-]*?)-(\d+)(?![A-Za-z0-9_.])")
_RE_SWESMITH_ID = re.compile(
    r"(?<![A-Za-z0-9_.-])(" + _OWNER + r")__([A-Za-z0-9][A-Za-z0-9_.-]*?)\.([0-9a-f]{8})(?![0-9a-z])")
_RE_REPO_FIELD = re.compile(
    r'"(upstream_repo|repo|network_root_repo|repository|repo_name)"\s*:\s*"(' + _OWNER + r")/(" + _NAME + r')"')
_RE_ROOT_REPOS = re.compile(r'"network_root_repos"\s*:\s*\[([^\]]*)\]')
_RE_QUOTED_REPO = re.compile(r'"(' + _OWNER + r")/(" + _NAME + r')"')
_RE_IMAGE_SWEBENCH = re.compile(r"sweb\.eval\.x86_64\.([a-z0-9-]+)_1776_([a-z0-9_.-]+?)-(\d+)")
_RE_IMAGE_SWESMITH = re.compile(r"swesmith\.x86_64\.([a-z0-9-]+)_1776_([a-z0-9_.-]+?)\.([0-9a-f]{8})")
_RE_GITHUB = re.compile(r"github\.com[/:](" + _OWNER + r")/([A-Za-z0-9_.-]{1,100}?)(?:\.git)?(?=[/\"'\s)#?,\]<>]|$)")

# Evidence kinds that identify a task/catalogue/diagnostic family (as opposed to a mere mention).
FAMILY_EVIDENCE = ("instance_id", "swesmith_instance", "repo_field", "network_root_repos", "docker_image")
MENTION_EVIDENCE = ("github_url",)

_ROOT_LIST_KEYS = {
    "excluded_network_roots": "excluded_network_root",
    "excluded_exposed_network_roots": "excluded_exposed_network_root",
    "current_fixed12_reserved_roots": "original_fixed_research12_whole_family_reserved",
    "observed_network_roots": "observed_known_project_history",
}


def glob_regex(pattern: str) -> "re.Pattern[str]":
    """Compile a POSIX glob where ``*``/``?`` stay inside one segment and ``**`` spans segments."""
    segs = pattern.split("/")
    out = "^"
    for i, seg in enumerate(segs):
        last = i == len(segs) - 1
        if seg == "**":
            out += ".+" if last else "(?:[^/]+/)*"
            continue
        out += "".join("[^/]*" if c == "*" else "[^/]" if c == "?" else re.escape(c) for c in seg)
        if not last:
            out += "/"
    return re.compile(out + "$")


def _dir_may_match(rel_dir: str, pattern: str) -> bool:
    """Whether files under ``rel_dir`` can match ``pattern`` (prunes the directory walk)."""
    segs = pattern.split("/")
    for i, part in enumerate(rel_dir.split("/")):
        if i >= len(segs) - 1:
            return segs[-1] == "**" or "**" in segs[:i + 1]
        if segs[i] == "**":
            return True
        if not fnmatch.fnmatchcase(part, segs[i]):
            return False
    return True


def iter_source_files(project_root: str | os.PathLike, patterns: Sequence[str] = EXCLUSION_SOURCES) -> Iterator[str]:
    """Yield sorted POSIX relpaths of files under the project root that match the patterns."""
    root = Path(project_root)
    regexes = [glob_regex(p) for p in patterns]
    out = set()
    for top in sorted({p.split("/", 1)[0] for p in patterns}):
        base = root / top
        if base.is_file():
            if any(rx.match(top) for rx in regexes):
                out.add(top)
            continue
        if not base.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            rel_dir = Path(dirpath).relative_to(root).as_posix()
            dirnames[:] = sorted(
                d for d in dirnames
                if d not in _SKIP_DIRS and any(_dir_may_match(f"{rel_dir}/{d}", p) for p in patterns))
            for name in filenames:
                rel = f"{rel_dir}/{name}"
                if any(rx.match(rel) for rx in regexes):
                    out.add(rel)
    yield from sorted(out)


def _resolve_owner(owner: str, known_owners: set[str]) -> str:
    """Strip a run/variant prefix glued to an owner (``validate-v2-pvlib`` -> ``pvlib``)."""
    if owner.casefold() in known_owners:
        return owner
    parts = owner.split("-")
    for i in range(1, len(parts)):
        cand = "-".join(parts[i:])
        if cand.casefold() in known_owners:
            return cand
    return owner


def scan_text(text: str) -> list[tuple[str, str, str]]:
    """Return raw ``(kind, owner, name)`` repository references found in one text."""
    refs: list[tuple[str, str, str]] = []
    for m in _RE_REPO_FIELD.finditer(text):
        owner, name = m.group(2), m.group(3)
        if owner.casefold() == "swesmith":
            sm = re.match(r"^(" + _OWNER + r")__(.+)\.([0-9a-f]{8})$", name)
            if sm:
                refs.append(("repo_field", sm.group(1), sm.group(2)))
            continue
        refs.append(("repo_field", owner, name))
    for m in _RE_ROOT_REPOS.finditer(text):
        for q in _RE_QUOTED_REPO.finditer(m.group(1)):
            refs.append(("network_root_repos", q.group(1), q.group(2)))
    for m in _RE_SWEBENCH_ID.finditer(text):
        refs.append(("instance_id", m.group(1), m.group(2)))
    for m in _RE_SWESMITH_ID.finditer(text):
        refs.append(("swesmith_instance", m.group(1), m.group(2)))
    for m in _RE_IMAGE_SWEBENCH.finditer(text):
        refs.append(("docker_image", m.group(1), m.group(2)))
    for m in _RE_IMAGE_SWESMITH.finditer(text):
        refs.append(("docker_image", m.group(1), m.group(2)))
    for m in _RE_GITHUB.finditer(text):
        refs.append(("github_url", m.group(1), m.group(2)))
    return refs


def _walk_json_roots(obj: Any, out: list[tuple[int, str, str | None]], mapping: dict[int, set[str]]) -> None:
    """Collect ``(root_id, reason, name_or_None)`` facts from a parsed JSON document."""
    if isinstance(obj, dict):
        root = obj.get("network_root_id")
        if type(root) is int and root > 0:
            names = obj.get("network_root_repos")
            if isinstance(names, list):
                out.append((root, "selected_family_slot", None))
                for n in names:
                    if isinstance(n, str) and _REPO.match(n):
                        mapping.setdefault(root, set()).add(n)
            reasons = obj.get("reasons")
            if isinstance(reasons, list):
                for r in reasons:
                    if isinstance(r, str):
                        out.append((root, r, None))
            reason = obj.get("reason")
            if isinstance(reason, str):
                out.append((root, reason, None))
        for key, value in obj.items():
            label = _ROOT_LIST_KEYS.get(key)
            if label and isinstance(value, list):
                for v in value:
                    if type(v) is int and v > 0:
                        out.append((v, label, None))
            if key == "family_ids" and isinstance(value, list) and obj.get("schema_version", "").startswith(
                    "research-T-mandatory-development-exclusions"):
                for v in value:
                    if isinstance(v, str) and v.isdigit():
                        out.append((int(v), "mandatory_development_exclusion", None))
            _walk_json_roots(value, out, mapping)
    elif isinstance(obj, list):
        for v in obj:
            _walk_json_roots(v, out, mapping)


def scan_project(project_root: str | os.PathLike, patterns: Sequence[str] = EXCLUSION_SOURCES) -> dict:
    """Scan project files for repository identifiers and numeric GitHub network-root reservations.

    Output (JSON-serialisable, deterministic):
    ``names``: {casefolded "owner/name": {"names": [...spellings], "kinds": {kind: count},
               "paths": [...sorted relpaths]}}
    ``roots``: {str(root_id): {"reasons": [...], "paths": [...], "names": [...]}}
    ``files_scanned``, ``bytes_scanned``, ``patterns``.
    """
    root = Path(project_root)
    raw: list[tuple[str, str, str, str]] = []  # (path, kind, owner, name)
    root_facts: dict[int, dict[str, set]] = {}
    root_names: dict[int, set[str]] = {}
    files = 0
    total = 0
    for rel in iter_source_files(root, patterns):
        p = root / rel
        try:
            size = p.stat().st_size
        except OSError:
            continue
        if size > MAX_SCAN_BYTES:
            continue
        data = p.read_bytes()
        files += 1
        total += len(data)
        text = data.decode("utf-8", errors="replace")
        for kind, owner, name in scan_text(text):
            raw.append((rel, kind, owner, name))
        if rel.endswith(".json") and ("network_root" in text or "family_ids" in text):
            try:
                doc = json.loads(text)
            except ValueError:
                doc = None
            if doc is not None:
                facts: list[tuple[int, str, str | None]] = []
                mapping: dict[int, set[str]] = {}
                _walk_json_roots(doc, facts, mapping)
                for rid, reason, _ in facts:
                    entry = root_facts.setdefault(rid, {"reasons": set(), "paths": set()})
                    entry["reasons"].add(reason)
                    entry["paths"].add(rel)
                for rid, names in mapping.items():
                    root_names.setdefault(rid, set()).update(names)
                    entry = root_facts.setdefault(rid, {"reasons": set(), "paths": set()})
                    entry["paths"].add(rel)
    # Owners seen in structured fields anchor the prefix stripping of instance ids.
    known_owners = {o.casefold() for (_, kind, o, _) in raw if kind in ("repo_field", "network_root_repos")}
    for names in root_names.values():
        for n in names:
            known_owners.add(n.split("/", 1)[0].casefold())
    names_out: dict[str, dict] = {}
    for rel, kind, owner, name in raw:
        if kind in ("instance_id", "swesmith_instance"):
            owner = _resolve_owner(owner, known_owners)
        if kind == "github_url" and name.lower().endswith(".git"):
            name = name[:-4]
        full = f"{owner}/{name}"
        key = full.casefold()
        entry = names_out.setdefault(key, {"names": set(), "kinds": {}, "paths": set()})
        entry["names"].add(full)
        entry["kinds"][kind] = entry["kinds"].get(kind, 0) + 1
        entry["paths"].add(rel)
    return {
        "patterns": list(patterns),
        "files_scanned": files,
        "bytes_scanned": total,
        "names": {k: {"names": sorted(v["names"]), "kinds": dict(sorted(v["kinds"].items())),
                      "paths": sorted(v["paths"])} for k, v in sorted(names_out.items())},
        "roots": {str(k): {"reasons": sorted(v["reasons"]), "paths": sorted(v["paths"]),
                           "names": sorted(root_names.get(k, ()))}
                  for k, v in sorted(root_facts.items())},
    }


def is_family_evidence(kinds: Mapping[str, int]) -> bool:
    return any(kinds.get(k, 0) for k in FAMILY_EVIDENCE)


@dataclasses.dataclass(frozen=True)
class ExclusionEntry:
    """One excluded repository family (a GitHub network, keyed by root id when known)."""
    family: str | None                 # canonical "owner/name" (GitHub full_name when resolved)
    network_root_id: int | None
    aliases: tuple[str, ...]           # every spelling seen (renames, mirrors, SWE-smith names)
    categories: tuple[str, ...]        # e.g. research_catalogue_family, task_family, reserved_network_root
    reasons: tuple[str, ...]           # reasons recorded by the project files
    source_paths: tuple[str, ...]      # sample of justifying relpaths (sorted)
    source_count: int                  # total justifying files

    def to_json(self) -> dict:
        return {"family": self.family, "network_root_id": self.network_root_id,
                "aliases": list(self.aliases), "categories": list(self.categories),
                "reasons": list(self.reasons), "source_paths": list(self.source_paths),
                "source_count": self.source_count}

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> "ExclusionEntry":
        return cls(family=d.get("family"), network_root_id=d.get("network_root_id"),
                   aliases=tuple(d.get("aliases", ())), categories=tuple(d.get("categories", ())),
                   reasons=tuple(d.get("reasons", ())), source_paths=tuple(d.get("source_paths", ())),
                   source_count=int(d.get("source_count", 0)))


def _name_keys(name: str) -> set[str]:
    """Casefolded keys for one spelling: ``owner/name`` and the ``owner__name`` form."""
    n = name.strip().casefold()
    if n.startswith("https://github.com/"):
        n = n[len("https://github.com/"):]
    n = n.rstrip("/")
    if n.endswith(".git"):
        n = n[:-4]
    keys = {n}
    if "/" in n:
        keys.add(n.replace("/", "__", 1))
    elif "__" in n:
        keys.add(n.replace("__", "/", 1))
    return keys


class ExclusionList:
    """Set of excluded families with lookup by name/alias and by GitHub network root id."""

    def __init__(self, entries: Iterable[ExclusionEntry]):
        self.entries = list(entries)
        self._by_name: dict[str, list[ExclusionEntry]] = {}
        self._by_root: dict[int, list[ExclusionEntry]] = {}
        for e in self.entries:
            for alias in set(e.aliases) | ({e.family} if e.family else set()):
                for key in _name_keys(alias):
                    self._by_name.setdefault(key, []).append(e)
            if e.network_root_id is not None:
                self._by_root.setdefault(int(e.network_root_id), []).append(e)

    def matches(self, repo: str | None = None, network_root_id: int | None = None,
                aliases: Iterable[str] = ()) -> list[ExclusionEntry]:
        """Entries that exclude a candidate; empty list means the candidate is not excluded."""
        hits: list[ExclusionEntry] = []
        names = [n for n in ([repo] if repo else []) + list(aliases) if n]
        for n in names:
            for key in _name_keys(n):
                hits.extend(self._by_name.get(key, ()))
        if network_root_id is not None:
            hits.extend(self._by_root.get(int(network_root_id), ()))
        seen, out = set(), []
        for h in hits:
            if id(h) not in seen:
                seen.add(id(h))
                out.append(h)
        return out

    def is_excluded(self, repo: str | None = None, network_root_id: int | None = None,
                    aliases: Iterable[str] = ()) -> bool:
        return bool(self.matches(repo, network_root_id, aliases))

    def to_json(self) -> list[dict]:
        return [e.to_json() for e in sorted(self.entries, key=_entry_sort_key)]

    @classmethod
    def from_json(cls, items: Iterable[Mapping[str, Any]]) -> "ExclusionList":
        return cls(ExclusionEntry.from_json(d) for d in items)


def _entry_sort_key(e: ExclusionEntry) -> tuple:
    return ((e.family or "~").casefold(), e.network_root_id or 0)


CATALOGUE_PATH = "staged/research-t-substantive-catalogue-v1/actual-once-a/catalogue.safe.json"
# The project refers to catalogue slots 10 and 17 as Task10 and Task17 (tasks/BACKLOG.json).
CATALOGUE_TASK_LABELS = {10: "Task10", 17: "Task17"}


def catalogue_families_from(doc: Mapping[str, Any]) -> dict[str, list[str]]:
    """``{casefolded owner/name: [labels]}`` for every family slot of the research catalogue.

    Includes the upstream names (``network_root_repos``/``upstream_repo``) and the SWE-smith mirror
    spelling taken from each candidate's ``repo``/``instance_id`` (e.g. buriy/python-readability).
    """
    out: dict[str, set[str]] = {}
    for slot in doc.get("family_slots", []):
        n = int(slot.get("slot"))
        labels = {f"research_catalogue_slot:{n:02d}", f"network_root_id:{slot.get('network_root_id')}"}
        if n in CATALOGUE_TASK_LABELS:
            labels.add(CATALOGUE_TASK_LABELS[n])
        names = set(slot.get("network_root_repos", []))
        for cand in slot.get("ordered_candidates", []):
            meta = cand.get("metadata", {})
            if isinstance(meta.get("upstream_repo"), str):
                names.add(meta["upstream_repo"])
            m = re.match(r"^(?:swesmith/)?(" + _OWNER + r")__(.+)\.[0-9a-f]{8}", str(meta.get("repo", "")))
            if m:
                names.add(f"{m.group(1)}/{m.group(2)}")
        for name in names:
            out.setdefault(name.casefold(), set()).update(labels)
    return {k: sorted(v) for k, v in sorted(out.items())}


def classify_scan_names(scan: Mapping[str, Any], non_family: Mapping[str, str] | None = None,
                        listing_patterns: Sequence[str] = ()) -> dict[str, dict]:
    """Classify every scanned name as ``family`` / ``listing_only`` / ``non_family`` / ``mention_only``.

    - ``non_family``: explicitly named tools, datasets and fixtures (with the stated reason)
    - ``mention_only``: only ``github_url`` mentions (dependencies, tools, upstream links)
    - ``listing_only``: family-type evidence occurs only in files matching ``listing_patterns``
      (public toolkit source listings such as the SWE-smith profile table); never project tasks
    - ``family``: family-type evidence in at least one project file -> excluded
    """
    non_family = {k.casefold(): v for k, v in (non_family or {}).items()}
    listing = [glob_regex(p) for p in listing_patterns]
    out: dict[str, dict] = {}
    for key, info in scan.get("names", {}).items():
        kinds = info.get("kinds", {})
        paths = info.get("paths", [])
        if key in non_family:
            cls, why = "non_family", non_family[key]
        elif not is_family_evidence(kinds):
            cls, why = "mention_only", "only github.com URL mentions"
        elif paths and all(any(rx.match(p) for rx in listing) for p in paths):
            cls, why = "listing_only", "only inside public toolkit source listings"
        else:
            cls, why = "family", "repository/instance identifiers in project files"
        out[key] = {"class": cls, "reason": why}
    return out


def build_exclusions(scan: Mapping[str, Any], resolved: Mapping[str, Any],
                     catalogue_families: Mapping[str, Sequence[str]] | None = None,
                     classes: Mapping[str, Mapping[str, str]] | None = None,
                     max_paths: int = 8) -> ExclusionList:
    """Merge the project scan with GitHub resolution into exclusion entries.

    ``resolved`` maps ``"name:<owner/name>"`` (casefolded) and ``"root:<id>"`` keys to
    ``{"full_name": str|None, "network_root_id": int|None, "network_root_full_name": str|None}``
    (produced by ``scripts/expA_repos.py exclusions``). ``catalogue_families`` maps casefolded
    names to their research-catalogue slot labels. ``classes`` comes from ``classify_scan_names``;
    only class ``family`` names (plus catalogue families) become entries. Every numeric network
    root found in project reservations or family slots becomes an entry too.
    """
    catalogue_families = catalogue_families or {}
    classes = classes or {k: {"class": "family" if is_family_evidence(v.get("kinds", {})) else "mention_only"}
                          for k, v in scan.get("names", {}).items()}
    groups: dict[str, dict] = {}

    def group_for(root: int | None, fallback: str) -> dict:
        key = f"root:{root}" if root is not None else f"name:{fallback}"
        return groups.setdefault(key, {"root": root, "family": None, "aliases": set(), "categories": set(),
                                       "reasons": set(), "paths": set()})

    for key, info in scan.get("names", {}).items():
        kinds = info.get("kinds", {})
        if key not in catalogue_families and classes.get(key, {}).get("class") != "family":
            continue
        res = resolved.get(f"name:{key}") or {}
        g = group_for(res.get("network_root_id"), key)
        g["aliases"].update(info.get("names", ()))
        if res.get("full_name"):
            g["aliases"].add(res["full_name"])
        if res.get("network_root_full_name"):
            g["family"] = res["network_root_full_name"]
            g["aliases"].add(res["network_root_full_name"])
        g["paths"].update(info.get("paths", ()))
        if key in catalogue_families:
            g["categories"].add("research_catalogue_family")
            g["reasons"].update(catalogue_families[key])
        g["categories"].add("project_task_family")
        g["reasons"].update(f"evidence:{k}" for k in sorted(kinds) if k in FAMILY_EVIDENCE)
    for key, labels in catalogue_families.items():
        if key in scan.get("names", {}):
            continue
        res = resolved.get(f"name:{key}") or {}
        g = group_for(res.get("network_root_id"), key)
        g["aliases"].add(key)
        g["categories"].add("research_catalogue_family")
        g["reasons"].update(labels)
    for rid_text, info in scan.get("roots", {}).items():
        rid = int(rid_text)
        res = resolved.get(f"root:{rid}") or {}
        g = group_for(rid, rid_text)
        reasons = set(info.get("reasons", ()))
        if reasons - {"selected_family_slot"}:
            g["categories"].add("reserved_network_root")
        if "selected_family_slot" in reasons:
            g["categories"].add("selected_family_slot")
        g["reasons"].update(reasons)
        g["paths"].update(info.get("paths", ()))
        g["aliases"].update(info.get("names", ()))
        full = res.get("network_root_full_name") or res.get("full_name")
        if full:
            g["aliases"].add(full)
            g["family"] = g["family"] or full
    entries = []
    for g in groups.values():
        family = g["family"] or (sorted(g["aliases"], key=str.casefold)[0] if g["aliases"] else None)
        paths = sorted(g["paths"])
        entries.append(ExclusionEntry(
            family=family, network_root_id=g["root"], aliases=tuple(sorted(g["aliases"], key=str.casefold)),
            categories=tuple(sorted(g["categories"])), reasons=tuple(sorted(g["reasons"])),
            source_paths=tuple(paths[:max_paths]), source_count=len(paths)))
    return ExclusionList(sorted(entries, key=_entry_sort_key))


# ---------------------------------------------------------------------------------------------
# Release-tag pinning
# ---------------------------------------------------------------------------------------------

# Release tags: optional whitelisted prefix (v, version, release, rel, r; any case; optional -/_/.
# separator and a further "v"), 1-4 numeric components separated by "." or "_", optional .postN.
# Prefixes never contain digits, so "rel_1_0_11" parses as 1.0.11 (not 11).
_TAG = re.compile(
    r"^(?:(?:version|release|rel|v|r)[-_.]?)?v?"
    r"(?P<ver>\d+(?:[._]\d+){0,3})(?:[.-]?post(?P<post>\d+))?$", re.IGNORECASE)


def release_version(tag: str) -> tuple[int, ...] | None:
    """Version key of a final-release tag (``v1.2.3``, ``1.2``, ``26.2.0``, ``rel_1_13_2``), else None.

    Pre-releases (a/b/rc/dev/alpha/beta/pre suffixes), other prefixes and non-version tags return None.
    """
    t = tag.strip()
    if t.startswith("refs/tags/"):
        t = t[len("refs/tags/"):]
    if t.endswith("^{}"):
        t = t[:-3]
    m = _TAG.match(t)
    if not m:
        return None
    parts = tuple(int(x) for x in re.split(r"[._]", m.group("ver")))
    if len(parts) == 1 and parts[0] > 1000000:  # bare dates/build numbers are not releases
        return None
    post = int(m.group("post")) if m.group("post") else 0
    return parts + (0,) * (4 - len(parts)) + (post,)


def choose_release_pin(tags: Iterable[Mapping[str, Any]], cutoff: str = CUTOFF) -> dict | None:
    """Pick the newest final release tag whose commit (and tagger, if annotated) predates cutoff.

    ``tags``: dicts with ``tag``, ``commit`` (peeled 40-hex), ``commit_date`` (ISO) and optional
    ``tagger_date`` (ISO, annotated tags). "Newest" is the highest release version; ties break on
    later commit date, then tag name. Returns the chosen dict (copied) or None.
    """
    limit = parse_utc(cutoff)
    best = None
    best_key = None
    for t in tags:
        ver = release_version(str(t.get("tag", "")))
        if ver is None or not _HEX40.match(str(t.get("commit", ""))) or not t.get("commit_date"):
            continue
        cdate = parse_utc(t["commit_date"])
        if cdate >= limit:
            continue
        if t.get("tagger_date") and parse_utc(t["tagger_date"]) >= limit:
            continue
        key = (ver, cdate, t["tag"])
        if best_key is None or key > best_key:
            best, best_key = dict(t), key
    return best


# ---------------------------------------------------------------------------------------------
# Baseline parsing and status
# ---------------------------------------------------------------------------------------------

def parse_junit(xml_bytes: bytes) -> dict:
    """Counts from a pytest JUnit XML report.

    ``collected`` = test cases in the report (including skipped); ``passed`` excludes skips,
    failures and errors. Collection errors are reported by pytest as error test cases.
    """
    root = ET.fromstring(xml_bytes)
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    cases = failed = errors = skipped = 0
    for suite in suites:
        for case in suite.iter("testcase"):
            cases += 1
            kinds = {child.tag for child in case}
            if "error" in kinds:
                errors += 1
            elif "failure" in kinds:
                failed += 1
            elif "skipped" in kinds:
                skipped += 1
    passed = cases - failed - errors - skipped
    return {"collected": cases, "passed": passed, "failed": failed, "errors": errors, "skipped": skipped}


def baseline_status(counts: Mapping[str, Any] | None, timed_out: bool = False) -> tuple[str, str]:
    """Apply the baseline rule: returns (status, reason) with status ``ok`` or ``tests_failed``."""
    if counts is None:
        return "tests_failed", "no test report"
    if timed_out:
        return "tests_failed", f"wall cap {WALL_CAP_SECONDS}s exceeded"
    passed = int(counts.get("passed", 0))
    bad = int(counts.get("failed", 0)) + int(counts.get("errors", 0))
    if passed < MIN_PASSED:
        return "tests_failed", f"only {passed} passing tests (< {MIN_PASSED})"
    frac = passed / (passed + bad)
    if frac < MIN_PASS_FRACTION:
        return "tests_failed", f"pass fraction {frac:.4f} < {MIN_PASS_FRACTION}"
    return "ok", f"{passed} passed, {bad} failed/errors (pass fraction {frac:.4f})"


# ---------------------------------------------------------------------------------------------
# Offline execution helpers (shared with the tracer lane)
# ---------------------------------------------------------------------------------------------

_SECRET_KEY = re.compile(r"(TOKEN|SECRET|PASSW|CREDENTIAL|AUTH|API_?KEY|PRIVATE|SSH_|AWS_|GH_|GITHUB_|HF_TOKEN)",
                         re.IGNORECASE)
_LO_UP = ('for ip in /usr/sbin/ip /sbin/ip /usr/bin/ip /bin/ip; do '
          'if [ -x "$ip" ]; then "$ip" link set lo up 2>/dev/null; break; fi; done; exec "$@"')


def offline_argv(argv: Sequence[str]) -> list[str]:
    """Wrap argv so it runs in a fresh user+network namespace with only loopback (brought up).

    ``unshare --user --map-root-user --net`` gives the command no route to any network; loopback is
    enabled so test suites that start local servers on 127.0.0.1 still work.
    """
    return ["unshare", "--user", "--map-root-user", "--net", "--", "/bin/sh", "-c", _LO_UP, "sh", *argv]


def scrubbed_test_env(env_dir: str | os.PathLike, tmp_dir: str | os.PathLike,
                      extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """Minimal environment for offline test runs: temp HOME, no tokens, no bytecode, fixed hash seed.

    Nothing is inherited from the caller. ``extra`` (per-family ``test_env``) may not carry
    credential-like keys.
    """
    tmp = Path(tmp_dir)
    for sub in ("home", "tmp", "cache", "config", "data"):
        (tmp / sub).mkdir(parents=True, exist_ok=True)
    env = {
        "HOME": str(tmp / "home"),
        "TMPDIR": str(tmp / "tmp"),
        "XDG_CACHE_HOME": str(tmp / "cache"),
        "XDG_CONFIG_HOME": str(tmp / "config"),
        "XDG_DATA_HOME": str(tmp / "data"),
        "PATH": f"{Path(env_dir) / 'bin'}:/usr/local/bin:/usr/bin:/bin",
        "VIRTUAL_ENV": str(env_dir),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "TZ": "UTC",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "PIP_NO_INDEX": "1",
        "UV_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
    }
    for key, value in (extra or {}).items():
        if _SECRET_KEY.search(key):
            raise RegistryError(f"credential-like key not allowed in test_env: {key}")
        env[str(key)] = str(value)
    return env


# ---------------------------------------------------------------------------------------------
# Package scope helpers
# ---------------------------------------------------------------------------------------------

try:  # shared identity API (static lane); the fallback below implements the same section 2 rule
    from .identity import is_test_or_excluded_path as _identity_excluded  # type: ignore
except Exception:  # pragma: no cover - identity.py may not exist yet during the build
    _identity_excluded = None

_EXCLUDED_DIR_PARTS = {"tests", "test", "testing", "docs", "examples", "benchmarks", "scripts"}
_EXCLUDED_FILES = {"setup.py", "noxfile.py", "conftest.py"}


def fallback_is_test_or_excluded_path(relpath: str) -> bool:
    """Section 2 exclusion rule (tests/docs/examples/benchmarks/scripts, test files, setup/nox)."""
    parts = PurePosixPath(relpath).parts
    name = parts[-1] if parts else ""
    if any(p in _EXCLUDED_DIR_PARTS for p in parts[:-1]):
        return True
    if name in _EXCLUDED_FILES:
        return True
    return name.endswith("_test.py") or (name.startswith("test_") and name.endswith(".py"))


def is_test_or_excluded_path(relpath: str) -> bool:
    if _identity_excluded is not None:
        return bool(_identity_excluded(relpath))
    return fallback_is_test_or_excluded_path(relpath)


def normalise_package_dir(path: str) -> str:
    """POSIX package dir with a trailing slash, or a single-module ``.py`` file path.

    Compatible with ``identity.normalize_package_dirs`` (which strips the trailing slash).
    """
    p = path.replace("\\", "/").strip()
    if not p or p.startswith("/") or re.match(r"^[A-Za-z]:", p):
        raise RegistryError(f"package dir must be a relative POSIX path: {path!r}")
    parts = p.rstrip("/").split("/")
    if any(x in ("..", ".", "") for x in parts):
        raise RegistryError(f"package dir must not contain empty, . or .. segments: {path!r}")
    joined = "/".join(parts)
    return joined if joined.endswith(".py") and not p.endswith("/") else joined + "/"


def package_python_files(repo_dir: str | os.PathLike, package_dirs: Sequence[str]) -> list[str]:
    """Sorted relpaths of in-scope ``.py`` files (package dirs minus section 2 exclusions)."""
    root = Path(repo_dir)
    out = set()
    for pd in package_dirs:
        base = root / normalise_package_dir(pd)
        if base.is_file() and base.suffix == ".py":
            rel = base.relative_to(root).as_posix()
            if not is_test_or_excluded_path(rel):
                out.add(rel)
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames if d not in ("__pycache__",) and not d.startswith("."))
            for name in filenames:
                if name.endswith(".py"):
                    rel = (Path(dirpath) / name).relative_to(root).as_posix()
                    if not is_test_or_excluded_path(rel):
                        out.add(rel)
    return sorted(out)


def count_loc(text: str) -> int:
    """Non-blank lines that are not pure ``#`` comments (docstrings count as code)."""
    n = 0
    for line in text.splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            n += 1
    return n


def package_stats(repo_dir: str | os.PathLike, package_dirs: Sequence[str]) -> dict:
    root = Path(repo_dir)
    files = package_python_files(root, package_dirs)
    loc = 0
    defs = 0
    for rel in files:
        text = (root / rel).read_text(encoding="utf-8", errors="replace")
        loc += count_loc(text)
        defs += len(re.findall(r"^\s*(?:async\s+)?def\s+\w+", text, flags=re.MULTILINE))
    return {"package_files": len(files), "package_loc": loc, "package_defs": defs}


# ---------------------------------------------------------------------------------------------
# Registry records
# ---------------------------------------------------------------------------------------------

@dataclasses.dataclass
class FamilyRecord:
    family: str
    url: str
    commit: str | None
    commit_date: str | None
    license: str | None
    package_dirs: list[str]
    test_command: list[str]
    python: str | None
    env_lock_sha256: str | None
    baseline: dict | None
    status: str
    role: str | None
    split: None = None
    tag: str | None = None
    pin_rule: str | None = None
    network_root_id: int | None = None
    license_files: list[dict] = dataclasses.field(default_factory=list)
    package_stats: dict | None = None
    env_path: str | None = None
    env_lock: dict | None = None
    python_executable: str = PYTHON_EXECUTABLE
    test_env: dict = dataclasses.field(default_factory=dict)
    install: dict = dataclasses.field(default_factory=dict)
    status_reason: str = ""
    exclusion_matches: list[dict] = dataclasses.field(default_factory=list)
    # In-repository files addressed to coding agents (AGENTS.md, CLAUDE.md, ...): untrusted repository
    # text that an agent reading the checkout may see; reported for the harness lane.
    agent_instruction_files: list[str] = dataclasses.field(default_factory=list)
    notes: str = ""

    @property
    def repo(self) -> str:
        m = _URL.match(self.url)
        return m.group(1) if m else ""

    def to_json(self) -> dict:
        d = dataclasses.asdict(self)
        d["repo"] = self.repo
        return d

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> "FamilyRecord":
        fields = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - fields - {"repo"}
        if unknown:
            raise RegistryError(f"unknown record fields for {d.get('family')!r}: {sorted(unknown)}")
        return cls(**{k: v for k, v in d.items() if k in fields})


_COUNT_KEYS = ("collected", "passed", "failed", "errors")


def validate_record(rec: FamilyRecord, cutoff: str = CUTOFF) -> list[str]:
    """Schema problems of one record (empty list = valid)."""
    errs: list[str] = []
    name = rec.family
    if not isinstance(name, str) or not _FAMILY.match(name):
        errs.append(f"{name!r}: family must match {_FAMILY.pattern}")
    if not isinstance(rec.url, str) or not _URL.match(rec.url):
        errs.append(f"{name}: url must be https://github.com/<owner>/<repo>")
    if rec.status not in STATUSES:
        errs.append(f"{name}: status {rec.status!r} not in {STATUSES}")
    if rec.split is not None:
        errs.append(f"{name}: split must be null until the items lane assigns it")
    if rec.role is not None and rec.role not in ROLES:
        errs.append(f"{name}: role {rec.role!r} not in {ROLES}")
    if rec.status == "excluded":
        if rec.role is not None:
            errs.append(f"{name}: excluded families carry role null")
        if not rec.exclusion_matches:
            errs.append(f"{name}: excluded without exclusion_matches")
        return errs
    if rec.exclusion_matches:
        errs.append(f"{name}: has exclusion matches but status {rec.status!r}")
    if rec.role is None:
        errs.append(f"{name}: non-excluded family needs a role")
    if not isinstance(rec.commit, str) or not _HEX40.match(rec.commit):
        errs.append(f"{name}: commit must be 40 lowercase hex")
    if rec.pin_rule not in PIN_RULES:
        errs.append(f"{name}: pin_rule must be one of {PIN_RULES}")
    if rec.pin_rule == "explicit_dev_tag" and rec.role != "dev":
        errs.append(f"{name}: explicit_dev_tag pins are only allowed for dev families")
    try:
        if rec.commit_date is None:
            errs.append(f"{name}: commit_date missing")
        elif rec.pin_rule != "explicit_dev_tag" and parse_utc(rec.commit_date) >= parse_utc(cutoff):
            errs.append(f"{name}: commit_date must be before {cutoff}")
    except ValueError as exc:
        errs.append(f"{name}: bad commit_date ({exc})")
    if rec.pin_rule in ("release_tag_before_cutoff", "explicit_dev_tag") and not rec.tag:
        errs.append(f"{name}: tag pin without tag")
    if not license_expression_ok(rec.license):
        errs.append(f"{name}: license {rec.license!r} not in the permissive allow-list")
    if not rec.package_dirs:
        errs.append(f"{name}: package_dirs empty")
    for pd in rec.package_dirs:
        try:
            if normalise_package_dir(pd) != pd:
                errs.append(f"{name}: package dir {pd!r} not normalised (POSIX, trailing slash)")
            elif is_test_or_excluded_path(pd if pd.endswith(".py") else pd + "x.py"):
                errs.append(f"{name}: package dir {pd!r} lies in an excluded area")
        except RegistryError as exc:
            errs.append(f"{name}: {exc}")
    if not rec.test_command or not all(isinstance(a, str) and a for a in rec.test_command):
        errs.append(f"{name}: test_command must be a non-empty argv list")
    if not isinstance(rec.python, str) or not rec.python.startswith("3.13"):
        if rec.status != "env_failed":
            errs.append(f"{name}: python must be a 3.13.x version string")
    if rec.status in ("ok", "tests_failed"):
        if not isinstance(rec.env_lock_sha256, str) or not _HEX64.match(rec.env_lock_sha256):
            errs.append(f"{name}: env_lock_sha256 must be 64 hex for an installed env")
        if not isinstance(rec.baseline, dict):
            errs.append(f"{name}: baseline missing")
    if isinstance(rec.baseline, dict) and rec.baseline.get("counts") is not None:
        counts = rec.baseline["counts"]
        for k in _COUNT_KEYS:
            v = counts.get(k)
            if type(v) is not int or v < 0:
                errs.append(f"{name}: baseline count {k} must be a non-negative int")
        if not errs and counts["passed"] + counts["failed"] + counts["errors"] > counts["collected"]:
            errs.append(f"{name}: passed+failed+errors exceeds collected")
    if rec.status == "ok":
        counts = (rec.baseline or {}).get("counts")
        status, _ = baseline_status(counts, bool((rec.baseline or {}).get("timed_out")))
        if status != "ok":
            errs.append(f"{name}: status ok but baseline rule gives {status}")
    return errs


def validate_registry(doc: Mapping[str, Any], cutoff: str = CUTOFF, min_main: int = MIN_MAIN,
                      dev_families: Sequence[str] = DEV_FAMILIES) -> list[str]:
    """Return every schema problem in a registry document (empty list = valid)."""
    errs: list[str] = []
    if doc.get("schema") != SCHEMA:
        errs.append(f"schema must be {SCHEMA!r}")
    if doc.get("cutoff") != cutoff:
        errs.append(f"cutoff must be {cutoff!r}")
    fams = doc.get("families")
    if not isinstance(fams, list):
        return errs + ["families must be a list"]
    records = []
    for d in fams:
        try:
            records.append(FamilyRecord.from_json(d))
        except (RegistryError, TypeError) as exc:
            errs.append(str(exc))
    names = [r.family for r in records]
    if len(set(names)) != len(names):
        errs.append("duplicate family names")
    repos = [r.repo.casefold() for r in records]
    if len(set(repos)) != len(repos):
        errs.append("duplicate repositories")
    for r in records:
        errs.extend(validate_record(r, cutoff))
    exclusions = None
    if "exclusions" in doc:
        try:
            exclusions = ExclusionList.from_json(doc["exclusions"])
        except (TypeError, ValueError) as exc:
            errs.append(f"bad exclusions: {exc}")
    if exclusions is not None:
        for r in records:
            if r.status != "excluded" and exclusions.is_excluded(r.repo, r.network_root_id):
                errs.append(f"{r.family}: matches the exclusion list but is not marked excluded")
    roles = {r.family: r.role for r in records}
    for dev in dev_families:
        if dev in roles and roles[dev] != "dev":
            errs.append(f"{dev}: dev family must have role dev")
    for r in records:
        if r.role == "dev" and r.family not in dev_families:
            errs.append(f"{r.family}: role dev reserved for {tuple(dev_families)}")
    n_main = sum(1 for r in records if r.role == "main")
    if n_main < min_main:
        errs.append(f"only {n_main} main families (< {min_main})")
    for r in records:
        if r.role == "main" and r.status == "env_failed":
            errs.append(f"{r.family}: env_failed family cannot be main")
    return errs


def load_registry(path: str | os.PathLike, strict: bool = True) -> dict:
    """Load and (by default) validate a registry document; raises RegistryError when invalid."""
    with open(path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    if strict:
        errs = validate_registry(doc)
        if errs:
            raise RegistryError("; ".join(errs[:20]))
    return doc


def families(doc: Mapping[str, Any], roles: Sequence[str] | None = None,
             statuses: Sequence[str] | None = None) -> list[FamilyRecord]:
    """Records filtered by role/status, in registry order."""
    out = []
    for d in doc.get("families", []):
        r = FamilyRecord.from_json(d)
        if roles is not None and r.role not in roles:
            continue
        if statuses is not None and r.status not in statuses:
            continue
        out.append(r)
    return out


_STATUS_RANK = {"ok": 0, "tests_failed": 1, "env_failed": 2, "excluded": 3}


CLEAN_PASS_FRACTION = 0.99


def quality_key(rec: FamilyRecord) -> tuple:
    """Deterministic quality order (lower sorts first = better).

    1. status (ok, tests_failed, env_failed, excluded)
    2. clean baseline: pass fraction >= CLEAN_PASS_FRACTION first
    3. more passing tests first (more runtime evidence for the tracer)
    4. family name (tie-break)
    """
    counts = (rec.baseline or {}).get("counts") or {}
    passed = int(counts.get("passed", 0) or 0)
    bad = int(counts.get("failed", 0) or 0) + int(counts.get("errors", 0) or 0)
    frac = passed / (passed + bad) if passed + bad else 0.0
    return (_STATUS_RANK.get(rec.status, 9), 0 if frac >= CLEAN_PASS_FRACTION else 1, -passed, rec.family)


def assign_roles(records: Sequence[FamilyRecord], n_main: int = MIN_MAIN,
                 dev_families: Sequence[str] = DEV_FAMILIES) -> list[FamilyRecord]:
    """Set ``role`` in place: dev families -> dev, excluded -> None, top ``n_main`` -> main, rest spare.

    env_failed families are never main. Returns the records sorted by (role, quality).
    """
    others = []
    for r in records:
        if r.status == "excluded":
            r.role = None
        elif r.family in dev_families:
            r.role = "dev"
        else:
            others.append(r)
    ranked = sorted(others, key=quality_key)
    main_count = 0
    for r in ranked:
        if main_count < n_main and r.status != "env_failed":
            r.role = "main"
            main_count += 1
        else:
            r.role = "spare"
    order = {"dev": 0, "main": 1, "spare": 2, None: 3}
    return sorted(records, key=lambda r: (order[r.role], quality_key(r)))
