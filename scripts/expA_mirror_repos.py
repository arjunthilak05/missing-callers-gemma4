"""Exp A clean model-visible checkouts (validity review finding F11; docs/EXPA_DESIGN_V2_DELTA.md D10.14).

Run on Core2 as thilak.arjun with the controller Python. For each family of a build spec (the
``expA_items.py`` spec format: families[] with family, repo_root (the analysis checkout), package_dirs and
commit; optional static_dir and trace_dir) it builds

  <out-root>/repos/<family>/                    git archive of the pinned commit plus the exact generated
                                                in-scope files of the analysis checkout; no .git, no
                                                test-run leftovers, no caches; directories 0755, files 0444
  <out-root>/repos-manifests/<manifest name>    the clean checkout manifest (tools.CHECKOUTS_SCHEMA): one
                                                tools.clean_checkout_entry per family (every file sha256,
                                                symlinks, directories, tree_sha256 = tools.TREE_FORMULA over
                                                the whole tree, package_tree_sha256 = the
                                                identity.source_tree_sha256 convention) plus a top-level
                                                "mirror" provenance block; deterministic, immutable (0444).
                                                The stager pins its sha256.
  <out-root>/repos-manifests/builds/*.json      one build record per invocation (times, host, code hashes)

The archive comes from a ``git clone --shared --no-checkout`` of the analysis checkout whose
$GIT_DIR/info/attributes is ``* -export-subst -export-ignore``, so archive attributes never change bytes or
drop files relative to the analysis checkout (param marks every __init__.py export-subst). The analysis
checkout itself is never written. Submodule contents are not part of git archive; they are recorded.

The generated in-scope files are the package files of the analysis checkout (identity.scan_package_files)
that the pinned commit does not track, for example a _version.py written by an editable install; each must
be listed by ``git status --porcelain --ignored`` as untracked or ignored. Everything else that is untracked
or ignored (tmp/, nohup.out, .hypothesis/, example.ics, ...) is left out and recorded. A tracked change in
the analysis checkout, or a HEAD other than the pinned commit, is refused.

The mirror's package-tree sha256 must equal the analysis checkout's (and the static and trace manifests'
when the spec names them); otherwise the family is refused and nothing of it is published. An existing
family tree or manifest is verified, never replaced. Run one instance per out root (no lock files on NFS).
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import tarfile
import uuid

from absent_edges.data import canonical_json
from absent_edges.probe import tools
from absent_edges.probe.identity import file_sha256s, normalize_package_dirs, scan_package_files, source_tree_sha256
from absent_edges.probe.ledger import write_immutable_bytes

MIRROR_PROTOCOL = "expA-mirror-repos-v1"
BUILD_SCHEMA = "expA-mirror-build-v1"
PUBLIC_ROOT = "/mnt/kldata/absent-edges-expA-public-20261101"
DEFAULT_MANIFEST_NAME = "clean-checkouts.json"
ATTRIBUTES_OVERRIDE = "* -export-subst -export-ignore\n"
DIR_MODE, FILE_MODE = 0o755, 0o444
FAMILY = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
GIT_ENV_DROP = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES",
                "GIT_CEILING_DIRECTORIES", "GIT_NAMESPACE", "GIT_ATTR_SOURCE")


class MirrorRefused(RuntimeError):
    """The analysis checkout or the built tree is not what the clean mirror may publish."""


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def git(repo, *arguments, binary=False):
    env = {key: value for key, value in os.environ.items() if key not in GIT_ENV_DROP}
    env.update(GIT_TERMINAL_PROMPT="0", GIT_OPTIONAL_LOCKS="0", LC_ALL="C")
    result = subprocess.run(["git", "-C", str(repo), *arguments], stdin=subprocess.DEVNULL, capture_output=True,
                            env=env, timeout=1800)
    if result.returncode != 0:
        raise MirrorRefused(f"git {' '.join(arguments[:3])} failed in {repo}: "
                            f"{result.stderr.decode('utf-8', 'replace').strip()[:500]}")
    return result.stdout if binary else result.stdout.decode("utf-8", "surrogateescape")


def parse_status(raw):
    """[(xy, path)] from `git status --porcelain=v1 -z`; renames/copies carry their source after."""
    entries, items, index = [], raw.split(b"\0"), 0
    while index < len(items):
        item = items[index]
        index += 1
        if not item:
            continue
        xy, path = item[:2].decode("ascii"), item[3:].decode("utf-8", "surrogateescape")
        entries.append((xy, path))
        if xy[0] in "RC":
            index += 1
    return entries


def analysis_state(repo, commit, package_dirs):
    """HEAD, tracked files, git-status listing and the package files of the analysis checkout."""
    head = git(repo, "rev-parse", "HEAD").strip()
    if head != commit:
        raise MirrorRefused(f"{repo} HEAD is {head}, not the pinned commit {commit}")
    status = parse_status(git(repo, "status", "--porcelain=v1", "-z", "--ignored=traditional",
                              "--untracked-files=all", binary=True))
    tracked_changes = sorted(path for xy, path in status if xy not in ("??", "!!"))
    if tracked_changes:
        raise MirrorRefused(f"{repo} has tracked changes {tracked_changes[:10]}; the mirror is git archive of "
                            "the pinned commit plus generated files only")
    listed = {path.rstrip("/"): xy for xy, path in status}
    tree = [line.split("\t", 1) for line in git(repo, "ls-tree", "-r", "-z", "--full-tree", commit).split("\0")
            if line]
    tracked = {path: meta.split()[0] for meta, path in tree}
    gitlinks = sorted(path for path, mode in tracked.items() if mode == "160000")
    scan = scan_package_files(repo, list(package_dirs))
    package = scan["files"]
    generated = sorted(path for path in package if path not in tracked)
    unlisted = [path for path in generated if path not in listed]
    if unlisted:
        raise MirrorRefused(f"{repo}: untracked package files {unlisted[:5]} are not listed by git status")
    excluded = sorted(path for path in listed if path not in generated)
    return {"head": head, "tracked": tracked, "gitlinks": gitlinks, "generated": generated,
            "excluded_untracked_or_ignored": excluded, "package_files": package,
            "package_tree_sha256": source_tree_sha256(file_sha256s(repo, package)),
            "skipped_symlinks": scan["skipped_symlinks"], "missing_package_dirs": scan["missing_package_dirs"]}


def export_attribute_lines(repo, commit):
    """.gitattributes lines of the pinned tree that mention export-subst or export-ignore (recorded)."""
    found = []
    for line in git(repo, "ls-tree", "-r", "-z", "--name-only", "--full-tree", commit).split("\0"):
        if line == ".gitattributes" or line.endswith("/.gitattributes"):
            text = git(repo, "show", f"{commit}:{line}")
            found += [f"{line}: {row.strip()}" for row in text.splitlines() if "export-" in row]
    return found


def _safe_member_path(name):
    path = name.rstrip("/")
    parts = path.split("/")
    if not path or path.startswith("/") or any(part in ("", ".", "..") for part in parts) or ".git" in parts:
        raise MirrorRefused(f"archive member {name!r} has an unsafe path")
    return path


def _lexically_inside(path, link):
    if link.startswith("/"):
        return "absolute_target"
    parts = path.split("/")[:-1]
    for part in link.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            if not parts:
                return "lexically_outside"
            parts.pop()
        else:
            parts.append(part)
    return None


def extract_archive(tar_path, destination):
    """Regular files, directories and symlinks only; symlinks that leave the tree are dropped and recorded."""
    destination = Path(destination)
    destination.mkdir(mode=DIR_MODE)
    symlinks, dropped = {}, []
    with tarfile.open(tar_path, "r:") as archive:
        for member in archive:
            path = _safe_member_path(member.name)
            target = destination / path
            if member.isdir():
                target.mkdir(mode=DIR_MODE, parents=True, exist_ok=True)
            elif member.isreg():
                target.parent.mkdir(mode=DIR_MODE, parents=True, exist_ok=True)
                with archive.extractfile(member) as reader, open(target, "xb") as writer:
                    shutil.copyfileobj(reader, writer, 1024 * 1024)
            elif member.issym():
                symlinks[path] = member.linkname
            else:
                raise MirrorRefused(f"archive member {member.name!r} is not a file, directory or symlink")
    for path, link in sorted(symlinks.items()):
        reason = _lexically_inside(path, link)
        if reason:
            dropped.append({"path": path, "target": link, "reason": reason, "action": "dropped"})
            continue
        target = destination / path
        target.parent.mkdir(mode=DIR_MODE, parents=True, exist_ok=True)
        os.symlink(link, target)
    return {"symlinks": len(symlinks) - len(dropped), "dropped_symlinks": dropped}


def remove_cache_entries(root):
    """Drop what tools.checkout_tree skips (VCS/cache dirs, *.pyc/*.pyo), so the tree equals its entry."""
    removed = tools.checkout_tree(root)["skipped"]
    for relative in removed:
        path = Path(root) / relative
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
    return removed


def set_modes(root):
    root = Path(root)
    for current, dirnames, filenames in os.walk(root, topdown=False, followlinks=False):
        for name in filenames:
            path = Path(current) / name
            if not path.is_symlink():
                os.chmod(path, FILE_MODE)
        for name in dirnames:
            path = Path(current) / name
            if not path.is_symlink():
                os.chmod(path, DIR_MODE)
    os.chmod(root, DIR_MODE)


def remove_tree(path):
    """Remove a build area (files are 0444, directories owned by us)."""
    path = Path(path)
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    if not path.exists():
        return
    for current, dirnames, _ in os.walk(path, followlinks=False):
        for name in dirnames:
            child = Path(current) / name
            if not child.is_symlink():
                os.chmod(child, 0o700)
    os.chmod(path, 0o700)
    shutil.rmtree(path)


def expected_package_trees(entry):
    """source_tree_sha256 recorded by the static and trace manifests named in the spec (when present)."""
    expected = {}
    static_dir = entry.get("static_dir")
    if static_dir:
        for name in ("locagent.manifest.json", "jedi.manifest.json"):
            path = Path(static_dir) / name
            if path.is_file():
                expected[f"static:{name}"] = json.loads(path.read_bytes()).get("source_tree_sha256")
    trace_dir = entry.get("trace_dir")
    if trace_dir and (Path(trace_dir) / "manifest.json").is_file():
        repo = json.loads((Path(trace_dir) / "manifest.json").read_bytes()).get("repo") or {}
        expected["trace:manifest.json"] = (repo.get("source_tree") or {}).get("sha256")
    return {key: value for key, value in expected.items() if value}


def _ensure_dir(path):
    path = Path(path)
    if not path.is_dir():
        path.mkdir(mode=DIR_MODE)
        os.chmod(path, DIR_MODE)
    return path


def build_family(spec_entry, out_root, scratch):
    """Build, verify and publish one family tree; returns (manifest entry, provenance, summary)."""
    family, repo, commit = spec_entry["family"], Path(spec_entry["repo_root"]), spec_entry["commit"]
    package_dirs = list(normalize_package_dirs(spec_entry["package_dirs"]))
    if not isinstance(family, str) or not FAMILY.fullmatch(family):
        raise MirrorRefused(f"invalid family name {family!r}")
    if not repo.is_absolute() or not repo.is_dir():
        raise MirrorRefused(f"{family}: repo_root {repo} is not an absolute directory")
    if not isinstance(commit, str) or len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise MirrorRefused(f"{family}: commit must be a full 40-hex sha")
    state = analysis_state(repo, commit, package_dirs)
    expected = expected_package_trees(spec_entry)
    disagreeing = {key: value for key, value in expected.items() if value != state["package_tree_sha256"]}
    if disagreeing:
        raise MirrorRefused(f"{family}: the analysis checkout's package tree {state['package_tree_sha256']} differs "
                            f"from {disagreeing}")
    final = _ensure_dir(_ensure_dir(out_root) / "repos") / family
    work = Path(scratch) / f"build-{family}-{uuid.uuid4().hex[:12]}"
    work.mkdir(mode=0o700, parents=True)
    try:
        shadow = work / "shadow"
        git(work, "clone", "--quiet", "--shared", "--no-checkout", str(repo), str(shadow))
        info = shadow / ".git" / "info"
        info.mkdir(parents=True, exist_ok=True)
        (info / "attributes").write_text(ATTRIBUTES_OVERRIDE)
        tar_bytes = git(shadow, "archive", "--format=tar", commit, binary=True)
        tar_path = work / "archive.tar"
        tar_path.write_bytes(tar_bytes)
        tree = work / "tree"
        extracted = extract_archive(tar_path, tree)
        for path in state["generated"]:
            source = repo / path
            if source.is_symlink() or not source.is_file():
                raise MirrorRefused(f"{family}: generated file {path} is not a regular file")
            target = tree / path
            if target.exists() or target.is_symlink():
                raise MirrorRefused(f"{family}: generated file {path} is also in the archive")
            target.parent.mkdir(mode=DIR_MODE, parents=True, exist_ok=True)
            shutil.copyfile(source, target, follow_symlinks=False)
        removed = remove_cache_entries(tree)
        set_modes(tree)
        built = tools.clean_checkout_entry(family, tree, package_dirs, commit=commit)
        if built["package_tree_sha256"] != state["package_tree_sha256"]:
            raise MirrorRefused(f"{family}: mirror package tree {built['package_tree_sha256']} != analysis "
                                f"checkout {state['package_tree_sha256']}")
        if built["skipped"] or built["escaping_symlinks"] or tools.checkout_entry_problems(built):
            raise MirrorRefused(f"{family}: the built tree is not clean: skipped {built['skipped'][:5]}, escaping "
                                f"{built['escaping_symlinks'][:5]}, {tools.checkout_entry_problems(built)}")
        if final.exists() or final.is_symlink():
            comparison = tools.compare_tree(built, tools.checkout_tree(final))
            if not comparison["matches"]:
                raise MirrorRefused(f"{final} exists and differs from the rebuilt mirror; refusing to replace it: "
                                    f"{json.dumps(comparison, sort_keys=True)[:1500]}")
            tree_status = "already_present_verified"
        else:
            os.rename(tree, final)
            tree_status = "published"
        entry = tools.clean_checkout_entry(family, final, package_dirs, commit=commit)
        if entry["tree_sha256"] != built["tree_sha256"] or entry["package_tree_sha256"] != state["package_tree_sha256"]:
            raise MirrorRefused(f"{family}: the published tree differs from the verified build")
        differences = sorted(path for path, sha in entry["files"].items()
                             if path not in state["generated"] and (repo / path).is_file()
                             and not (repo / path).is_symlink()
                             and hashlib.sha256((repo / path).read_bytes()).hexdigest() != sha)
        missing_vs_tracked = sorted(path for path, mode in state["tracked"].items()
                                    if mode not in ("160000", "120000") and path not in entry["files"])
        provenance = {
            "analysis_checkout": {"path": str(repo), "head": state["head"],
                                  "package_tree_sha256": state["package_tree_sha256"],
                                  "package_files": len(state["package_files"]),
                                  "skipped_symlinks": state["skipped_symlinks"],
                                  "missing_package_dirs": state["missing_package_dirs"]},
            "expected_package_tree_sha256": expected,
            "archive": {"method": "git archive --format=tar <commit> from a git clone --shared --no-checkout whose "
                                  "$GIT_DIR/info/attributes is '* -export-subst -export-ignore'",
                        "tar_sha256": sha256_bytes(tar_bytes), "tar_bytes": len(tar_bytes),
                        "export_attribute_lines": export_attribute_lines(repo, commit),
                        "submodules_not_included": state["gitlinks"]},
            "generated_in_scope_files": [{"path": path, "sha256": entry["files"][path]} for path in state["generated"]],
            "excluded_untracked_or_ignored": state["excluded_untracked_or_ignored"],
            "dropped_symlinks": extracted["dropped_symlinks"], "removed_cache_entries": removed,
            "tracked_files_differing_from_analysis_checkout": differences,
            "tracked_files_missing_from_tree": missing_vs_tracked,
            "modes": {"directories": "0755", "files": "0444"}}
        summary = {"family": family, "tree": str(final), "tree_status": tree_status,
                   "tree_sha256": entry["tree_sha256"], "package_tree_sha256": entry["package_tree_sha256"],
                   "files": len(entry["files"]), "symlinks": len(entry["symlinks"]),
                   "directories": len(entry["directories"]), "generated": state["generated"],
                   "excluded": len(state["excluded_untracked_or_ignored"]),
                   "dropped_symlinks": len(extracted["dropped_symlinks"]), "removed_cache_entries": len(removed),
                   "differences": differences, "submodules_not_included": state["gitlinks"]}
        return entry, provenance, summary
    finally:
        remove_tree(work)


def write_manifest(out_root, entries, provenance, name):
    """tools.checkouts_manifest(entries) plus a top-level provenance block; written once, never replaced."""
    manifest = tools.checkouts_manifest(entries)
    manifest["mirror"] = {"protocol": MIRROR_PROTOCOL, "families": dict(sorted(provenance.items()))}
    data = canonical_json(manifest) + b"\n"
    path = _ensure_dir(Path(out_root) / "repos-manifests") / name
    if path.exists():
        if path.read_bytes() != data:
            raise MirrorRefused(f"{path} exists with different content; choose another --manifest-name")
        status = "already_present_verified"
    else:
        write_immutable_bytes(path, data)
        status = "published"
    return {"path": str(path), "sha256": sha256_bytes(data), "status": status, "families": len(entries)}


def code_hashes():
    root = Path(__file__).resolve().parents[1]
    names = ["scripts/expA_mirror_repos.py", "absent_edges/probe/tools.py", "absent_edges/probe/identity.py",
             "absent_edges/probe/ledger.py", "absent_edges/data.py"]
    return {name: sha256_bytes((root / name).read_bytes()) for name in names if (root / name).is_file()}


def load_spec(path, families=None):
    spec = json.loads(Path(path).read_bytes())
    entries = spec.get("families") if isinstance(spec, dict) else None
    if not isinstance(entries, list) or not entries:
        raise MirrorRefused("the spec needs a nonempty families list")
    selected = [entry for entry in entries if families is None or entry.get("family") in families]
    if families is not None and {entry["family"] for entry in selected} != set(families):
        raise MirrorRefused(f"families not in the spec: {sorted(set(families) - {e['family'] for e in selected})}")
    for entry in selected:
        for key in ("family", "repo_root", "package_dirs", "commit"):
            if key not in entry:
                raise MirrorRefused(f"spec entry {entry.get('family')!r} lacks {key}")
    names = [entry["family"] for entry in selected]
    if len(set(names)) != len(names):
        raise MirrorRefused("duplicate families in the spec")
    return selected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--spec", required=True, help="item build spec (families[]: family, repo_root, package_dirs, "
                                                      "commit, optional static_dir/trace_dir)")
    parser.add_argument("--families", default=None, help="comma-separated subset of the spec's families")
    parser.add_argument("--out-root", default=PUBLIC_ROOT)
    parser.add_argument("--scratch", default=None, help="build area on the out root's filesystem "
                                                        "(default: <out-root>/.mirror-work)")
    parser.add_argument("--manifest-name", default=DEFAULT_MANIFEST_NAME,
                        help="manifest file name under <out-root>/repos-manifests/")
    args = parser.parse_args(argv)
    started = datetime.datetime.now(datetime.timezone.utc)
    families = None if args.families is None else [value for value in args.families.split(",") if value]
    out_root = Path(args.out_root)
    try:
        if not out_root.is_absolute():
            raise MirrorRefused("--out-root must be absolute")
        if os.path.basename(args.manifest_name) != args.manifest_name or not args.manifest_name.endswith(".json"):
            raise MirrorRefused("--manifest-name must be a plain *.json file name")
        spec_entries = load_spec(args.spec, families)
        scratch = Path(args.scratch) if args.scratch else _ensure_dir(out_root) / ".mirror-work"
        scratch.mkdir(mode=0o700, parents=True, exist_ok=True)
        entries, provenance, summaries = [], {}, []
        for spec_entry in spec_entries:
            entry, record, summary = build_family(spec_entry, out_root, scratch)
            entries.append(entry)
            provenance[entry["family"]] = record
            summaries.append(summary)
        manifest = write_manifest(out_root, entries, provenance, args.manifest_name)
    except (MirrorRefused, OSError, ValueError) as error:
        print(json.dumps({"protocol": MIRROR_PROTOCOL, "status": "refused",
                          "reason": f"{type(error).__name__}: {error}"}, sort_keys=True))
        return 2
    record = {"schema": BUILD_SCHEMA, "protocol": MIRROR_PROTOCOL, "started_utc": started.isoformat(),
              "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "host": socket.gethostname(),
              "uid": os.getuid(), "spec": {"path": str(Path(args.spec).resolve()),
                                           "sha256": sha256_bytes(Path(args.spec).read_bytes())},
              "out_root": str(out_root), "manifest": manifest, "families": summaries, "code": code_hashes(),
              "python": sys.version.split()[0], "git": git(out_root, "--version").strip()}
    builds = _ensure_dir(out_root / "repos-manifests" / "builds")
    name = f"{started.strftime('%Y%m%dT%H%M%S%fZ')}-{socket.gethostname().split('.')[0]}-{os.getpid()}.json"
    write_immutable_bytes(builds / name, canonical_json(record) + b"\n")
    print(json.dumps({"protocol": MIRROR_PROTOCOL, "status": "ok", "manifest": manifest,
                      "build_record": str(builds / name),
                      "families": {summary["family"]: {key: summary[key] for key in
                                                       ("tree_sha256", "package_tree_sha256", "files", "generated",
                                                        "excluded", "tree_status", "differences",
                                                        "submodules_not_included")}
                                   for summary in summaries}}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
