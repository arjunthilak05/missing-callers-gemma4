#!/usr/bin/env python3
"""Run one family's pytest suite offline under the Exp A runtime tracer (lane T).

docs/EXPA_DESIGN.md section 5. The test command runs inside
``unshare --user --map-root-user --net`` (only loopback is brought up), with a scrubbed
environment (fresh HOME and TMPDIR, no tokens, PYTHONDONTWRITEBYTECODE=1,
PYTHONHASHSEED=0, PYTHONNOUSERSITE=1) and a wall-time cap (default 45 min). The plugin
is never installed into the repository environment: the few modules it needs are copied
to ``<out>/plugin_src`` and only that directory is put on PYTHONPATH.

Modes:
  trace  record call sites (default)
  count  same plugin and flags, test outcomes only, no monitoring (untraced baseline)
  plain  no plugin; outcomes parsed from the pytest summary line

Partial runs keep the edges observed so far: the plugin stops starting tests at a soft
deadline (cap minus --soft-margin), the runner sends SIGTERM to the process group at the
cap (the plugin flushes and exits), then SIGKILL after --grace seconds (the latest
checkpoint is kept). All such runs are marked partial.

Outputs in --out (refused if it already holds trace outputs):
  edges.jsonl, plugin_manifest.json   written by the plugin (trace; count writes the manifest)
  manifest.json                       this runner's record: command, isolation, env, hashes, status
  stdout.log, stderr.log, plugin_src/

Example:
  python scripts/expA_trace.py --family boltons --repo /x/repos/boltons --package-dir boltons \\
      --python /x/envs/boltons/bin/python --out /x/traces/boltons -- python -m pytest -q tests
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from absent_edges.probe import tracer as T  # noqa: E402

PLUGIN_MODULE = "absent_edges.probe.tracer_plugin"
OVERLAY_REQUIRED = ("absent_edges/__init__.py", "absent_edges/probe/__init__.py",
                    "absent_edges/probe/tracer.py", "absent_edges/probe/tracer_plugin.py")
OVERLAY_OPTIONAL = ("absent_edges/probe/identity.py",)
DEFAULT_CAP_SECONDS = 45 * 60
SECRET_NAME = re.compile(r"TOKEN|SECRET|PASSW|CREDENTIAL|API_?KEY|AUTH|COOKIE|SESSION|PRIVATE", re.I)
PYTEST_EXIT = {0: "ok", 1: "tests_failed", 2: "interrupted", 3: "internal_error",
               4: "usage_error", 5: "no_tests_collected"}
OUTCOMES = ("passed", "failed", "error", "skipped", "xfailed", "xpassed")
EXIT_CODES = {"complete": 0, "partial": 0, "unavailable": 1, "failed": 3}

PROBE_CODE = r"""
import json, sys, importlib.metadata as md
info = {"version": sys.version.split()[0], "implementation": sys.implementation.name,
        "executable": sys.executable, "monitoring": hasattr(sys, "monitoring")}
try:
    import pytest
    info["pytest"] = pytest.__version__
except Exception as error:
    info["pytest"] = None
    info["pytest_error"] = type(error).__name__
info["distributions"] = sorted({(d.metadata["Name"] or "", d.version) for d in md.distributions()})
print(json.dumps(info))
"""


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def build_test_argv(command, python, *, plugin: bool, cacheprovider: bool = False) -> list[str]:
    """Normalize a pytest command to ``<python> ... -m pytest`` and inject ``-p`` options."""
    argv = [str(part) for part in command]
    if not argv:
        raise ValueError("empty test command")
    head = os.path.basename(argv[0])
    if head in ("pytest", "py.test"):
        argv = [str(python), "-m", "pytest"] + argv[1:]
    elif re.fullmatch(r"python[0-9.]*", head) or argv[0] == str(python):
        argv[0] = str(python)
    else:
        raise ValueError(f"the test command must run pytest, got {argv[0]!r}")
    index = next((i for i in range(1, len(argv) - 1) if argv[i] == "-m" and argv[i + 1] in ("pytest", "py.test")),
                 None)
    if index is None:
        raise ValueError("the test command must invoke '-m pytest'")
    extra = (["-p", PLUGIN_MODULE] if plugin else []) + ([] if cacheprovider else ["-p", "no:cacheprovider"])
    return argv[:index + 2] + extra + argv[index + 2:]


def parse_set_env(values) -> dict:
    env = {}
    for item in values or ():
        name, sep, value = item.partition("=")
        if not sep or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError(f"--set-env expects NAME=VALUE, got {item!r}")
        if SECRET_NAME.search(name):
            raise ValueError(f"refusing to pass a credential-like variable into tests: {name}")
        if name.startswith("AE_TRACE_") or name in ("PYTHONPATH", "HOME", "TMPDIR"):
            raise ValueError(f"{name} is controlled by the runner")
        env[name] = value
    return env


def scrubbed_env(python, *, home, tmp, pythonpath=None, extra=None) -> dict:
    python = Path(python)
    env = {
        "PATH": os.pathsep.join([str(python.parent), "/usr/local/bin", "/usr/bin", "/bin"]),
        "HOME": str(home), "TMPDIR": str(tmp), "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "TZ": "UTC",
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1", "PYTHONHASHSEED": "0",
        "PY_COLORS": "0", "NO_COLOR": "1",
    }
    if (python.parent.parent / "pyvenv.cfg").is_file():
        env["VIRTUAL_ENV"] = str(python.parent.parent)
    if pythonpath:
        env["PYTHONPATH"] = str(pythonpath)
    env.update(extra or {})
    return env


def probe_python(python, env) -> dict:
    result = subprocess.run([str(python), "-I", "-c", PROBE_CODE], env=env, stdin=subprocess.DEVNULL,
                            capture_output=True, text=True, timeout=180, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"interpreter probe failed ({result.returncode}): {result.stderr.strip()[-2000:]}")
    return json.loads(result.stdout)


def netns_interfaces(unshare_bin) -> list[str]:
    result = subprocess.run([unshare_bin, "--user", "--map-root-user", "--net", "--", "/bin/cat", "/proc/self/net/dev"],
                            stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"unshare preflight failed ({result.returncode}): {result.stderr.strip()[-2000:]}")
    return sorted(line.split(":", 1)[0].strip() for line in result.stdout.splitlines()[2:] if ":" in line)


def stage_overlay(destination: Path) -> dict:
    files = {}
    for relative in OVERLAY_REQUIRED + OVERLAY_OPTIONAL:
        source = PROJECT / relative
        if not source.is_file():
            if relative in OVERLAY_REQUIRED:
                raise FileNotFoundError(source)
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        files[relative] = T.sha256_file(target)
    return files


def git_info(path: Path) -> dict:
    env = {"PATH": "/usr/bin:/bin", "HOME": "/nonexistent", "GIT_CONFIG_NOSYSTEM": "1", "LC_ALL": "C"}
    info = {"head": None, "dirty": None}
    try:
        head = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], env=env, capture_output=True,
                              text=True, timeout=60, check=False)
        if head.returncode == 0:
            info["head"] = head.stdout.strip()
            status = subprocess.run(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
                                    env=env, capture_output=True, text=True, timeout=120, check=False)
            if status.returncode == 0:
                info["dirty"] = bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return info


def parse_pytest_summary(text: str) -> dict | None:
    """Outcome counts from pytest's final summary line (plain mode only)."""
    pattern = re.compile(r"(\d+) (passed|failed|errors?|skipped|xfailed|xpassed|deselected|warnings?|rerun)\b")
    for line in reversed(text.splitlines()):
        if re.search(r"\bin [0-9.]+s\b", line) and pattern.search(line):
            counts: dict[str, int] = {}
            for number, word in pattern.findall(line):
                key = {"errors": "error", "warnings": "warning"}.get(word, word)
                counts[key] = counts.get(key, 0) + int(number)
            return {"line": line.strip("= ").strip(), "outcomes": counts,
                    "ran": sum(counts.get(name, 0) for name in OUTCOMES)}
    return None


def tail_text(path: Path, limit: int = 1 << 16) -> str:
    with open(path, "rb") as stream:
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        stream.seek(max(0, size - limit))
        return stream.read().decode("utf-8", "replace")


def signal_group(pgid: int, sig) -> bool:
    try:
        os.killpg(pgid, sig)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def run_command(argv, *, cwd, env, stdout_path, stderr_path, cap, grace) -> dict:
    started = time.monotonic()
    terminated = killed = False
    with open(stdout_path, "wb") as out, open(stderr_path, "wb") as err:
        process = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                   start_new_session=True)
        try:
            try:
                returncode = process.wait(timeout=cap)
            except subprocess.TimeoutExpired:
                terminated = True
                signal_group(process.pid, signal.SIGTERM)
                try:
                    returncode = process.wait(timeout=grace)
                except subprocess.TimeoutExpired:
                    killed = True
                    signal_group(process.pid, signal.SIGKILL)
                    returncode = process.wait()
        except BaseException:
            signal_group(process.pid, signal.SIGKILL)
            process.wait()
            raise
        finally:
            leftovers = signal_group(process.pid, signal.SIGKILL)  # stray children of the test run
    return {"returncode": returncode, "terminated_at_cap": terminated, "killed_after_grace": killed,
            "leftover_processes_killed": leftovers, "wall_seconds": round(time.monotonic() - started, 3)}


def decide_status(mode, run, trace, plugin_manifest, checkpoint_manifest, summary) -> tuple[str, str | None, int]:
    """complete | partial | unavailable (no test ran) | failed (no outputs)."""
    reason = None
    if mode == "plain":
        ran = (summary or {}).get("ran", 0)
        has_outputs = summary is not None
    else:
        effective = plugin_manifest or checkpoint_manifest
        ran = ((effective or {}).get("tests") or {}).get("ran", 0)
        has_outputs = effective is not None or (trace is not None and bool(trace.sources))
        if plugin_manifest is not None:
            reason = plugin_manifest.get("partial_reason")
        elif has_outputs:
            reason = "checkpoint_only"
    if run["killed_after_grace"]:
        reason = reason or "killed_after_grace"
    elif run["terminated_at_cap"]:
        reason = reason or "sigterm_at_cap"
    partial = reason is not None or (trace is not None and trace.partial)
    if not has_outputs:
        return "failed", reason, ran
    if ran == 0:
        return "unavailable", reason, ran
    return ("partial" if partial else "complete"), reason, ran


def main(argv=None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    command: list[str] = []
    if "--" in raw:
        split = raw.index("--")
        raw, command = raw[:split], raw[split + 1:]
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--family", required=True)
    parser.add_argument("--repo", required=True, help="repository checkout root (AE_TRACE_ROOT)")
    parser.add_argument("--package-dir", action="append", required=True,
                        help="package directory relative to --repo (repeatable)")
    parser.add_argument("--python", required=True, help="repository environment interpreter")
    parser.add_argument("--out", required=True, help="output directory (fresh)")
    parser.add_argument("--mode", choices=("trace", "count", "plain"), default="trace")
    parser.add_argument("--test-command", help="pytest command as one shell-quoted string (or give it after --)")
    parser.add_argument("--cwd", default=".", help="working directory relative to --repo")
    parser.add_argument("--wall-time-cap", type=float, default=DEFAULT_CAP_SECONDS, help="seconds (default 2700)")
    parser.add_argument("--soft-margin", type=float, default=60.0,
                        help="seconds before the cap after which no new test starts (default 60)")
    parser.add_argument("--grace", type=float, default=60.0, help="seconds between SIGTERM and SIGKILL")
    parser.add_argument("--checkpoint-seconds", type=float, default=120.0)
    parser.add_argument("--env-lock", help="resolved-package lock file whose sha256 is recorded")
    parser.add_argument("--set-env", action="append", default=[], metavar="NAME=VALUE",
                        help="extra non-secret variable for the test process (recorded)")
    parser.add_argument("--isolation", choices=("unshare", "none"), default="unshare",
                        help="'none' is for local debugging only and is recorded")
    parser.add_argument("--no-loopback", action="store_true", help="leave loopback down inside the netns")
    parser.add_argument("--cacheprovider", action="store_true", help="do not add -p no:cacheprovider")
    parser.add_argument("--keep-scratch", action="store_true", help="keep the private HOME/TMPDIR")
    args = parser.parse_args(raw)
    if args.test_command:
        if command:
            parser.error("give the test command after -- or with --test-command, not both")
        command = shlex.split(args.test_command)
    if not command:
        parser.error("a pytest command is required (after --)")
    if args.wall_time_cap <= 0 or args.grace < 0 or args.soft_margin < 0:
        parser.error("--wall-time-cap must be > 0; --grace and --soft-margin must be >= 0")

    repo = Path(args.repo).resolve(strict=True)
    python = Path(args.python)
    python = python if python.is_absolute() else Path.cwd() / python
    if not repo.is_dir() or not python.exists():
        parser.error("--repo must be a directory and --python must exist")
    out = Path(args.out).absolute()
    if out.exists() and (any(out.glob("edges*.jsonl")) or any(out.glob("plugin_manifest*.json"))
                         or (out / T.RUN_MANIFEST_FILE).exists()):
        parser.error(f"refusing to reuse {out}: it already holds trace outputs")
    try:
        package_dirs = T.normalize_package_dirs(args.package_dir)
        extra_env = parse_set_env(args.set_env)
        test_argv = build_test_argv(command, python, plugin=args.mode != "plain", cacheprovider=args.cacheprovider)
    except ValueError as error:
        parser.error(str(error))
    for directory in package_dirs:
        if directory and not (repo / directory).exists():
            parser.error(f"package dir {directory!r} does not exist under {repo}")
    cwd = (repo / args.cwd).resolve()
    out.mkdir(parents=True, exist_ok=True)

    scratch = Path(tempfile.mkdtemp(prefix=".scratch-", dir=out))
    home, tmp = scratch / "home", scratch / "tmp"
    home.mkdir()
    tmp.mkdir()
    manifest: dict = {"schema": T.MANIFEST_SCHEMA, "family": args.family, "mode": args.mode,
                      "runner": {"script": "scripts/expA_trace.py", "sha256": T.sha256_file(Path(__file__)),
                                 "python": sys.version.split()[0], "project_git": git_info(PROJECT)},
                      "identity_impl_runner": T.IDENTITY_IMPL}
    try:
        overlay = out / "plugin_src"
        plugin_files = stage_overlay(overlay) if args.mode != "plain" else {}
        env = scrubbed_env(python, home=home, tmp=tmp, pythonpath=overlay if plugin_files else None,
                           extra=extra_env)
        probe = probe_python(python, scrubbed_env(python, home=home, tmp=tmp))
        if probe.get("pytest") is None:
            raise RuntimeError("pytest is not importable in the repository environment")
        if args.mode == "trace" and not probe.get("monitoring"):
            raise RuntimeError("the repository interpreter has no sys.monitoring (Python 3.12+ required)")
        isolation = {"method": "none", "loopback_up": False, "netns_interfaces": None}
        full_argv = list(test_argv)
        if args.isolation == "unshare":
            unshare_bin = shutil.which("unshare", path="/usr/bin:/bin:/usr/sbin:/sbin")
            if not unshare_bin:
                raise RuntimeError("unshare not found")
            interfaces = netns_interfaces(unshare_bin)
            if interfaces != ["lo"]:
                raise RuntimeError(f"network namespace is not isolated: interfaces {interfaces}")
            ip_bin = shutil.which("ip", path="/usr/sbin:/sbin:/usr/bin:/bin")
            inner = list(test_argv)
            if not args.no_loopback and ip_bin:
                script = (f"{shlex.quote(ip_bin)} link set lo up || "
                          f"echo 'expA_trace: could not bring up loopback' >&2; exec \"$@\"")
                inner = ["/bin/sh", "-c", script, "expA-trace"] + inner
            full_argv = [unshare_bin, "--user", "--map-root-user", "--net", "--"] + inner
            isolation = {"method": "unshare --user --map-root-user --net", "netns_interfaces": interfaces,
                         "loopback_up": bool(not args.no_loopback and ip_bin), "ip": ip_bin}
        source_before = T.source_tree_digest(repo, package_dirs)
        started_epoch = time.time()
        if args.mode != "plain":
            env.update({T.ENV_ROOT: str(repo), T.ENV_PACKAGE_DIRS: T.package_dirs_env(package_dirs),
                        T.ENV_OUT: str(out), T.ENV_MODE: args.mode,
                        T.ENV_CHECKPOINT: str(args.checkpoint_seconds)})
            soft = max(0.0, args.wall_time_cap - args.soft_margin)
            env[T.ENV_DEADLINE] = repr(started_epoch + soft)
        manifest.update({
            "repo": {"root": str(repo), "git": git_info(repo), "package_dirs": list(package_dirs),
                     "source_tree": source_before},
            "test_command": list(command), "test_argv": test_argv, "effective_argv": full_argv,
            "cwd": str(cwd), "isolation": isolation, "env": env,
            "python": {key: probe.get(key) for key in ("executable", "version", "implementation", "monitoring",
                                                        "pytest")},
            "env_lock": ({"path": str(Path(args.env_lock).resolve()), "sha256": T.sha256_file(args.env_lock)}
                         if args.env_lock else None),
            "env_distributions": {"count": len(probe["distributions"]),
                                  "sha256": hashlib.sha256(T.canonical_json(probe["distributions"])
                                                           .encode("utf-8")).hexdigest(),
                                  "items": probe["distributions"]},
            "plugin_files": plugin_files,
            "wall_time_cap_seconds": args.wall_time_cap, "soft_margin_seconds": args.soft_margin,
            "grace_seconds": args.grace, "checkpoint_seconds": args.checkpoint_seconds,
            "started_at_utc": utc_now(),
        })
        run = run_command(full_argv, cwd=cwd, env=env, stdout_path=out / "stdout.log",
                          stderr_path=out / "stderr.log", cap=args.wall_time_cap, grace=args.grace)
        manifest["finished_at_utc"] = utc_now()
        run["pytest_exit"] = PYTEST_EXIT.get(run["returncode"], "signal" if run["returncode"] < 0 else "other")
        trace = T.load_trace_dir(out, strict=False) if args.mode != "plain" else None
        plugin_manifest = trace.plugin_manifests.get(T.PLUGIN_MANIFEST_STEM + ".json") if trace else None
        checkpoint_manifest = (trace.plugin_manifests.get(T.PLUGIN_MANIFEST_STEM + T.PARTIAL_MARK + ".json")
                               if trace else None)
        summary = parse_pytest_summary(tail_text(out / "stdout.log"))
        status, reason, ran = decide_status(args.mode, run, trace, plugin_manifest, checkpoint_manifest, summary)
        source_after = T.source_tree_digest(repo, package_dirs)
        manifest.update({
            "run": run, "status": status, "partial": status == "partial", "partial_reason": reason,
            "tests": {"ran": ran, "plugin": (plugin_manifest or checkpoint_manifest or {}).get("tests"),
                      "summary_line": summary},
            "plugin_manifest": plugin_manifest or checkpoint_manifest,
            "edges": None if trace is None or args.mode != "trace" else {
                "sources": trace.sources, "integrity": trace.integrity, "load_errors": list(trace.errors)[:50],
                "load_error_count": len(trace.errors), "duplicates_merged": trace.duplicates,
                "file_sha256": {name: T.sha256_file(out / name) for name in trace.integrity},
                "summary": T.summarize_edges(trace.records)},
            "source_unchanged_after_run": source_after["sha256"] == source_before["sha256"],
            "logs": {name: T.sha256_file(out / name) for name in ("stdout.log", "stderr.log")},
        })
    except Exception as error:
        manifest.update({"status": "failed", "partial": False, "error": f"{type(error).__name__}: {error}"[:4000],
                         "finished_at_utc": utc_now()})
    finally:
        if not args.keep_scratch:
            shutil.rmtree(scratch, ignore_errors=True)
    T.write_json_atomic(out / T.RUN_MANIFEST_FILE, manifest)
    brief = {"family": args.family, "mode": args.mode, "status": manifest.get("status"),
             "partial_reason": manifest.get("partial_reason"), "error": manifest.get("error"),
             "returncode": (manifest.get("run") or {}).get("returncode"),
             "wall_seconds": (manifest.get("run") or {}).get("wall_seconds"),
             "tests": ((manifest.get("tests") or {}).get("plugin") or {}).get("outcomes")
             or ((manifest.get("tests") or {}).get("summary_line") or {}).get("outcomes"),
             "edges": ((manifest.get("edges") or {}).get("summary") or {}).get("edges"),
             "manifest": str(out / T.RUN_MANIFEST_FILE)}
    print(json.dumps(brief, sort_keys=True))
    return EXIT_CODES.get(manifest.get("status"), 3)


if __name__ == "__main__":
    raise SystemExit(main())
