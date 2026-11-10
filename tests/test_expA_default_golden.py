"""Golden values of a DEFAULT Exp A invocation, captured from the frozen release before the model-profile and
episode-budget options were added (branch expA-param-20261101, base 155a028c5, whose absent_edges/, scripts/
and tests/ equal release a53c49396).

A default invocation must keep every value below:

* serve: the vLLM argv, the hardened ``systemd-run`` unit argv (byte for byte) and the serve plan, for the plan the
  launch script requests and for the CLI defaults; the effective arguments of ``verify-model``; the model constants;
* runner: the dry-run summary; run.json's episode_config, model, prompts, tools and design; one live default
  episode's config, model profile, system prompt and first chat request;
* launch script: ``bash -n``; prepare's launch.env; the serve and runner unit argvs that ``serve`` and ``runner``
  pass to sudo (sudo, systemctl, the identity guard and the GPU check are stubbed: nothing runs as root, no GPU).

Records may gain keys (a dict here must be contained in the new value, lists and strings are exact) and launch.env
may gain lines at its end; nothing may change or disappear. tests/test_expA_param.py pins exactly what is added.
Linux only, like test_probe_harness/test_probe_runtime: serve.py imports pwd, and the runner reads checkouts with
POSIX no-follow reads. ``python -B tests/test_expA_default_golden.py --capture`` prints the current values.
"""
import contextlib
import dataclasses
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

PROJECT = Path(__file__).resolve().parents[1]
POSIX = os.name == "posix"
if POSIX:
    sys.path.insert(0, str(PROJECT))
    from absent_edges.probe import serve, tools
    import test_probe_harness as H
GPU7 = "GPU-4cb533cc-d12b-8e68-7177-39e79451bdad"
CORE1_OK = {"uid": 1008, "euid": 1008, "gid": 1009, "egid": 1009, "user": "thilak.arjun", "host": "ai-core-team-1"}
RUNTIME = "/mnt/kldata/absent-edges-expA-public-20261101/runtime/python3.11/bin/python3.11"
LAUNCH_PLAN_ARGV = ["plan", "--run", "golden1", "--instance", "g7", "--gpu-uuid", GPU7, "--port", "8141",
                    "--max-num-seqs", "8", "--gpu-memory-utilization", "0.92", "--model-check", "full",
                    "--release-root", "/dev/shm/ae-expA-golden1/inputs/release", "--runtime-python", RUNTIME]
CLI_DEFAULT_PLAN_ARGV = ["plan", "--run", "golden1", "--instance", "g7", "--gpu-uuid", GPU7, "--port", "8141"]
PREPARE_ARGV = ["prepare", "--run", "golden1", "--run-id", "golden1", "--git-sha", "155a028c5",
                "--release-dir", "/mnt/kldata/x/expA/releases/155a028c5", "--release-sums-sha256", "a" * 64,
                "--stager-sha256", "b" * 64, "--dispatch-dir", "/mnt/kldata/x/expA/dispatch",
                "--dispatch-manifest-sha256", "c" * 64, "--checkouts-manifest", "/mnt/kldata/x/clean-checkouts.json",
                "--checkouts-manifest-sha256", "d" * 64, "--instance", f"g7={GPU7}=8141", "--concurrency", "8"]


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                          .encode("utf-8")).hexdigest()


def contained(golden, actual, path="$"):
    """Problems where ``actual`` does not keep ``golden``: dict keys may be added; lists and scalars are exact."""
    if isinstance(golden, dict):
        if not isinstance(actual, dict):
            return [f"{path}: {type(actual).__name__} instead of an object"]
        problems = []
        for key, value in golden.items():
            if key not in actual:
                problems.append(f"{path}.{key}: missing")
            else:
                problems += contained(value, actual[key], f"{path}.{key}")
        return problems
    return [] if golden == actual else [f"{path}: {actual!r} != golden {golden!r}"]


def load_script(name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(f"golden_{name}", PROJECT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------------------------------------------

def cli_plan(argv):
    stdout = io.StringIO()
    with mock.patch.object(serve, "current_identity", return_value=CORE1_OK), contextlib.redirect_stdout(stdout):
        code = load_script("expA_serve").main(list(argv))
    return {"code": code, "plan": json.loads(stdout.getvalue())}


def verify_model_call(argv=("verify-model",)):
    """The verify_model arguments the verify-model CLI uses, with every default applied (nothing is read)."""
    real = serve.verify_model
    calls = []

    def record(*args, **kwargs):
        bound = inspect.signature(real).bind(*args, **kwargs)
        bound.apply_defaults()
        calls.append({name: list(value) if isinstance(value, tuple) else str(value) if isinstance(value, Path)
                      else value for name, value in bound.arguments.items()})
        return {"failures": []}
    with mock.patch.object(serve, "verify_model", side_effect=record), contextlib.redirect_stdout(io.StringIO()):
        code = load_script("expA_serve").main(list(argv))
    return {"code": code, "calls": calls}


def serve_values():
    spec = serve.ServeSpec(run="golden1", instance="g7", gpu_uuid=GPU7, port=8141)
    return {"plan_launch_script": cli_plan(LAUNCH_PLAN_ARGV), "plan_cli_defaults": cli_plan(CLI_DEFAULT_PLAN_ARGV),
            "api": {"spec": dataclasses.asdict(spec), "vllm_argv": serve.vllm_argv(spec),
                    "environment": serve.serve_environment(spec)},
            "verify_model": verify_model_call(),
            "constants": {"MODEL_DIR": str(serve.MODEL_DIR), "MODEL_MANIFEST": str(serve.MODEL_MANIFEST),
                          "MODEL_MANIFEST_SHA256": serve.MODEL_MANIFEST_SHA256, "MODEL_ALIAS": serve.MODEL_ALIAS,
                          "MODEL_REVISION": serve.MODEL_REVISION, "MAX_MODEL_LEN": serve.MAX_MODEL_LEN,
                          "MODEL_REQUIRED_FILES": list(serve.MODEL_REQUIRED_FILES),
                          "MODEL_ALLOWED_EXTRA": sorted(serve.MODEL_ALLOWED_EXTRA), "SERVE_PROTOCOL": serve.SERVE_PROTOCOL}}


# ---------------------------------------------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------------------------------------------

class FakeJailBash:
    """Stands in for tools.ReadOnlyBash, whose namespace jail cannot be proven on every test host (WSL refuses
    the /usr bind mount); the default episode below submits at once and never runs bash."""

    def __init__(self, checkout, scratch_dir, **kwargs):
        self.checkout = checkout

    def describe(self):
        return {"fake": "tests/test_expA_default_golden.py"}

    def selftest(self):
        return {"passed": True, "fake": True}

    def contract_probe(self):
        return {"fake": True}

    def run(self, command):
        raise AssertionError("the golden episode never runs bash")


def invoke_runner(runner, argv):
    stdout = io.StringIO()
    previous = (signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT))
    try:
        with contextlib.redirect_stdout(stdout):
            code = runner.main(list(argv))
    finally:
        signal.signal(signal.SIGTERM, previous[0])
        signal.signal(signal.SIGINT, previous[1])
    return code, stdout.getvalue()


def runner_fixture(base):
    """The synthetic dispatch view of test_probe_harness (family fam, 2 items) and its clean checkout manifest."""
    runner = load_script("expA_run")
    dispatch, _ = H.write_dispatch(base)
    manifest = base / "checkouts" / "manifest.json"
    code, _ = invoke_runner(runner, ["checkout-manifest", "--out", str(manifest), "--repos-root", str(base / "repos"),
                                     "--dispatch", str(dispatch), "--commit", "fam=" + "c" * 40])
    assert code == 0
    return runner, dispatch, manifest


def live_args(base, dispatch, manifest, server, run="run"):
    return ["--dispatch", str(dispatch), "--run-dir", str(base / run), "--ledger", str(base / "local" / f"{run}.sqlite3"),
            "--scratch", str(base / f"scratch-{run}"), "--checkout-manifest", str(manifest),
            "--repos-root", str(base / "repos"), "--endpoint", f"{server.endpoint}={H.GPU_A}", "--blocks", "P",
            "--seeds", "11", "--limit", "1", "--concurrency", "1", "--health-interval", "0"]


def live_episode(runner, argv, server, run_dir):
    """Run the runner for one cell; return run.json and the episode's records (endpoint port masked)."""
    with mock.patch.object(tools, "ReadOnlyBash", FakeJailBash):
        code, output = invoke_runner(runner, argv)
    results = sorted((run_dir / "cells").rglob("result.json"))
    result = json.loads(results[0].read_bytes())
    setup = json.loads((results[0].parent / "events.jsonl").read_text().splitlines()[0])["value"]
    chat = server.chat_requests()[0]

    def masked(profile):
        return dict(profile, endpoint="<fake server>")
    return {"exit_code": code, "results": len(results), "run_json": json.loads((run_dir / "run.json").read_bytes()),
            "result": {"exit_status": result["exit_status"], "config": result["config"],
                       "model_profile": masked(result["model_profile"])},
            "setup": {"config": setup["config"], "model_profile": masked(setup["model_profile"]),
                      "system": setup["system"]},
            "chat_request": {key: chat[key] for key in ("model", "max_tokens", "temperature", "seed")}}


def runner_values(base):
    runner, dispatch, manifest = runner_fixture(base)
    code, output = invoke_runner(runner, ["--dispatch", str(dispatch), "--dry-run", "--show", "2"])
    dry_run = {"code": code, "summary": json.loads(output)}
    server = H.FakeServer(default=H.SUBMIT)
    try:
        live = live_episode(runner, live_args(base, dispatch, manifest, server), server, base / "run")
    finally:
        server.close()
    run_json = live.pop("run_json")
    live["run_json"] = {"protocol": run_json["protocol"], "episode_config": run_json["episode_config"],
                        "model": run_json["model"], "infra_attempt_cap": run_json["infra_attempt_cap"],
                        "prompts_sha256": canonical_sha256(run_json["prompts"]),
                        "tools_sha256": canonical_sha256(run_json["tools"]),
                        "design_sha256": canonical_sha256(run_json["design"])}
    return {"dry_run": dry_run, "live": live}


# ---------------------------------------------------------------------------------------------------------------
# launch script (stubbed host)
# ---------------------------------------------------------------------------------------------------------------

STUBS = r'''
# ---- test stubs (tests/test_expA_default_golden.py): no Core1, no root, no GPU, nothing is started ----
EXPECTED_UID=$(id -u)
require_identity() { :; }
run_root() { printf '%s/shm/ae-expA-%s' "$AE_TEST_ROOT" "$1"; }
findmnt() { case "$*" in *FSTYPE*) echo tmpfs ;; *) echo rw,nosuid,nodev,noexec ;; esac; }
gpu_check() { check_uuid "$1"; }
verify_release() { :; }
systemctl() { return 3; }
endpoint_ready() { printf '%s %s\n' "$1" "$MODEL_ALIAS" >> "$AE_TEST_ROOT/endpoint_ready.log"; }
sudo() {
    AE_TEST_SUDO=$(( ${AE_TEST_SUDO:-0} + 1 ))
    printf '%s\0' "$@" > "$AE_TEST_ROOT/sudo/$(printf '%03d' "$AE_TEST_SUDO").argv"
}
'''

PYTHON_WRAPPER = '''#!{python}
"""The runtime Python of the stubbed launch script: the serve plan runs as if on Core1 as thilak.arjun."""
import runpy
import sys

args = sys.argv[1:]
while args and args[0] in ("-I", "-B"):
    args.pop(0)
if args[0] == "-c":
    sys.argv = ["-c"] + args[2:]
    exec(compile(args[1], "<string>", "exec"), {{"__name__": "__main__"}})
else:
    sys.path.insert(0, {project!r})
    from absent_edges.probe import serve
    serve.current_identity = lambda: {identity!r}
    sys.argv = args
    runpy.run_path(args[0], run_name="__main__")
'''


class LaunchHost:
    """scripts/expA_core1_launch.sh with its host commands stubbed, in a temporary directory."""

    def __init__(self, root):
        self.root = Path(root)
        text = (PROJECT / "scripts" / "expA_core1_launch.sh").read_text()
        tail = 'main "$@"\n'
        assert text.endswith(tail), "the launch script must end with its main call"
        self.script = self.root / "launch.sh"
        self.script.write_text(text[:-len(tail)] + STUBS + tail)
        (self.root / "sudo").mkdir()
        (self.root / "shm").mkdir()
        self.python = self.root / "bin" / "python3.11"
        self.python.parent.mkdir()
        self.python.write_text(PYTHON_WRAPPER.format(python=sys.executable, project=str(PROJECT), identity=CORE1_OK))
        self.python.chmod(0o755)

    def run(self, *argv, check=True):
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LANG": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1",
               "AE_TEST_ROOT": str(self.root), "HOME": str(self.root)}
        result = subprocess.run(["bash", str(self.script), *argv], capture_output=True, text=True, timeout=120,
                                stdin=subprocess.DEVNULL, env=env)
        if check and result.returncode != 0:
            raise AssertionError(f"launch {argv[0]} exited {result.returncode}: {result.stderr[-3000:]}")
        return result

    def run_root(self, run):
        return self.root / "shm" / f"ae-expA-{run}"

    def stage(self, run):
        """What ``stage`` leaves behind for serve and runner: the release (here the checkout) and a receipt."""
        inputs = self.run_root(run) / "inputs"
        inputs.mkdir(mode=0o700)
        (inputs / "release").symlink_to(PROJECT)
        (inputs / "STAGE_RECEIPT.run-state.golden.json").write_text("{}\n")

    def sudo_calls(self):
        calls = []
        for path in sorted((self.root / "sudo").iterdir()):
            calls.append(path.read_bytes().decode().split("\0")[:-1])
            path.unlink()
        return calls

    def normalize(self, value):
        """Temporary paths and the runner log stamp, so that values compare across hosts and runs."""
        if isinstance(value, dict):
            return {key: self.normalize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.normalize(item) for item in value]
        if isinstance(value, str):
            return re.sub(r"runner-[0-9]{8}T[0-9]{6}Z", "runner-<stamp>", value.replace(str(self.root), "<T>"))
        return value


def units(calls, kind):
    """The ``sudo -n systemd-run`` argvs of one kind (serve or runner units)."""
    return [call for call in calls if call[:2] == ["-n", "systemd-run"] and f"--unit=ae-expA-{kind}-" in call[2]]


def launch_values(base, prepare_extra=(), serve_extra=(), runner_extra=()):
    host = LaunchHost(base)
    syntax = subprocess.run(["bash", "-n", str(PROJECT / "scripts" / "expA_core1_launch.sh")], capture_output=True,
                            text=True, timeout=60)
    prepared = host.run(*PREPARE_ARGV, "--runtime-python", str(host.python), *prepare_extra)
    launch_env = (host.run_root("golden1") / "control" / "launch.env").read_text()
    host.stage("golden1")
    host.run("serve", "--run", "golden1", *serve_extra)
    serve_calls = host.sudo_calls()
    plan = json.loads((host.run_root("golden1") / "control" / "serve-g7.plan.json").read_text())
    host.run("runner", "--run", "golden1", "--blocks", "P", "--seeds", "11", *runner_extra)
    runner_calls = host.sudo_calls()
    return host.normalize({"bash_n": syntax.returncode, "prepare_stdout": prepared.stdout, "launch_env": launch_env,
                           "serve_plan": plan, "serve_units": units(serve_calls, "serve"),
                           "runner_units": units(runner_calls, "runner"),
                           "endpoint_ready": (base / "endpoint_ready.log").read_text()})


def capture():
    with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
        try:
            values = {"serve": serve_values(), "runner": runner_values(Path(first)), "launch": launch_values(Path(second))}
        finally:
            for path in Path(first).rglob("*"):
                if path.is_dir() and not path.is_symlink():
                    path.chmod(0o755)
    return values


# Captured with --capture on the frozen code (WSL Ubuntu, Python 3.12.3, 2026-11-01). The plan for the CLI
# defaults was identical to the plan the launch script requests, so it is stored once (plan_launch_script).
GOLDEN = json.loads(r"""
{
 "launch": {
  "bash_n": 0,
  "endpoint_ready": "8141 absent-edges-gemma4-31b-bf16\n",
  "launch_env": "RUN=golden1\nRUN_ID=golden1\nRELEASE_DIR=/mnt/kldata/x/expA/releases/155a028c5\nRELEASE_SUMS_SHA256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\nSTAGER_SHA256=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\nDISPATCH_DIR=/mnt/kldata/x/expA/dispatch\nDISPATCH_MANIFEST_SHA256=cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc\nCHECKOUTS_MANIFEST=/mnt/kldata/x/clean-checkouts.json\nCHECKOUTS_MANIFEST_SHA256=dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd\nCHECKOUTS_DIR=\nINSTANCES=g7=GPU-4cb533cc-d12b-8e68-7177-39e79451bdad=8141\nCONCURRENCY=8\nGIT_SHA=155a028c5\nTARGET_PARENT=/mnt/kldata/absent-edges-20261018-core2/expA/runs\nRUNTIME_PYTHON=<T>/bin/python3.11\nMAX_NUM_SEQS=8\nGPU_MEMORY_UTILIZATION=0.92\nMODEL_CHECK=full\nMIRROR_OWNER_UID=\n",
  "prepare_stdout": "RUN=golden1\nRUN_ID=golden1\nRELEASE_DIR=/mnt/kldata/x/expA/releases/155a028c5\nRELEASE_SUMS_SHA256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\nSTAGER_SHA256=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\nDISPATCH_DIR=/mnt/kldata/x/expA/dispatch\nDISPATCH_MANIFEST_SHA256=cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc\nCHECKOUTS_MANIFEST=/mnt/kldata/x/clean-checkouts.json\nCHECKOUTS_MANIFEST_SHA256=dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd\nCHECKOUTS_DIR=\nINSTANCES=g7=GPU-4cb533cc-d12b-8e68-7177-39e79451bdad=8141\nCONCURRENCY=8\nGIT_SHA=155a028c5\nTARGET_PARENT=/mnt/kldata/absent-edges-20261018-core2/expA/runs\nRUNTIME_PYTHON=<T>/bin/python3.11\nMAX_NUM_SEQS=8\nGPU_MEMORY_UTILIZATION=0.92\nMODEL_CHECK=full\nMIRROR_OWNER_UID=\n",
  "runner_units": [["-n","systemd-run","--unit=ae-expA-runner-golden1","--description=Absent Edges Exp A runner golden1","--collect","--property=Type=exec","--property=User=thilak.arjun","--property=Group=thilak.arjun","--property=WorkingDirectory=<T>/shm/ae-expA-golden1","--property=KillMode=mixed","--property=KillSignal=SIGTERM","--property=TimeoutStopSec=1800","--property=Restart=no","--property=RuntimeMaxSec=432000","--property=NoNewPrivileges=yes","--property=ProtectSystem=strict","--property=ProtectHome=yes","--property=InaccessiblePaths=-/data5t -/workspace -/mnt/eu-ai-datasets","--property=UMask=0077","--property=ReadOnlyPaths=/dev/shm","--property=ReadWritePaths=<T>/shm/ae-expA-golden1/run <T>/shm/ae-expA-golden1/scratch <T>/shm/ae-expA-golden1/ledger <T>/shm/ae-expA-golden1/logs","--property=StandardOutput=append:<T>/shm/ae-expA-golden1/logs/runner-<stamp>.out","--property=StandardError=append:<T>/shm/ae-expA-golden1/logs/runner-<stamp>.err","--setenv=PATH=/usr/bin:/bin","--setenv=LANG=C.UTF-8","--setenv=PYTHONDONTWRITEBYTECODE=1","--setenv=HOME=<T>/shm/ae-expA-golden1/scratch/home","--setenv=TMPDIR=<T>/shm/ae-expA-golden1/scratch/tmp","--","<T>/bin/python3.11","-I","-B","<T>/shm/ae-expA-golden1/inputs/release/scripts/expA_run.py","--dispatch","<T>/shm/ae-expA-golden1/inputs/dispatch","--checkout-manifest","<T>/shm/ae-expA-golden1/inputs/checkouts/manifest.json","--repos-root","<T>/shm/ae-expA-golden1/inputs/checkouts/repos","--run-dir","<T>/shm/ae-expA-golden1/run","--run-id","golden1","--ledger","<T>/shm/ae-expA-golden1/ledger/ledger.sqlite3","--scratch","<T>/shm/ae-expA-golden1/scratch","--endpoint","http://127.0.0.1:8141=GPU-4cb533cc-d12b-8e68-7177-39e79451bdad","--concurrency","8","--git-sha","155a028c5","--blocks","P","--seeds","11"]],
  "serve_plan": {"cache_root":"/dev/shm/ae-expA-golden1/serve/g7","endpoint":"http://127.0.0.1:8141","environment":{"CUDA_CACHE_PATH":"/dev/shm/ae-expA-golden1/serve/g7/cache/cuda_cache_path","CUDA_DEVICE_ORDER":"PCI_BUS_ID","CUDA_VISIBLE_DEVICES":"GPU-4cb533cc-d12b-8e68-7177-39e79451bdad","DO_NOT_TRACK":"1","FLASHINFER_WORKSPACE_BASE":"/dev/shm/ae-expA-golden1/serve/g7/cache/flashinfer_workspace_base","HF_HOME":"/dev/shm/ae-expA-golden1/serve/g7/cache/hf_home","HF_HUB_CACHE":"/dev/shm/ae-expA-golden1/serve/g7/cache/hf_hub_cache","HF_HUB_DISABLE_TELEMETRY":"1","HF_HUB_OFFLINE":"1","HOME":"/dev/shm/ae-expA-golden1/serve/g7/home","LANG":"C.UTF-8","MKL_NUM_THREADS":"1","OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","PATH":"/mnt/kldata/absent-edges-expA-public-20261101/envs/serve/bin:/usr/bin:/bin","PIP_CACHE_DIR":"/dev/shm/ae-expA-golden1/serve/g7/cache/pip_cache_dir","PYTHONDONTWRITEBYTECODE":"1","PYTHONNOUSERSITE":"1","PYTHONPYCACHEPREFIX":"/dev/shm/ae-expA-golden1/serve/g7/cache/pythonpycacheprefix","TEMP":"/dev/shm/ae-expA-golden1/serve/g7/tmp","TMP":"/dev/shm/ae-expA-golden1/serve/g7/tmp","TMPDIR":"/dev/shm/ae-expA-golden1/serve/g7/tmp","TORCHINDUCTOR_CACHE_DIR":"/dev/shm/ae-expA-golden1/serve/g7/cache/torchinductor_cache_dir","TRANSFORMERS_OFFLINE":"1","TRITON_CACHE_DIR":"/dev/shm/ae-expA-golden1/serve/g7/cache/triton_cache_dir","UV_CACHE_DIR":"/dev/shm/ae-expA-golden1/serve/g7/cache/uv_cache_dir","VLLM_CACHE_ROOT":"/dev/shm/ae-expA-golden1/serve/g7/cache/vllm_cache_root","VLLM_CONFIG_ROOT":"/dev/shm/ae-expA-golden1/serve/g7/cache/vllm_config_root","VLLM_NO_USAGE_STATS":"1","XDG_CACHE_HOME":"/dev/shm/ae-expA-golden1/serve/g7/cache/xdg_cache_home","XDG_CONFIG_HOME":"/dev/shm/ae-expA-golden1/serve/g7/cache/xdg_config_home","XDG_DATA_HOME":"/dev/shm/ae-expA-golden1/serve/g7/cache/xdg_data_home"},"flags_source":"staged/core1-thinking-study-20261031a/source/common.py","model":{"alias":"absent-edges-gemma4-31b-bf16","check":"full","manifest":"/mnt/kldata/absent-edges-expA-public-20261101/models/COPY.sha256","manifest_sha256":"7e67af6d6d633a7469c2463156f936c5a38979a7a3c61e662ce85abe0211987a","max_model_len":32768,"path":"/mnt/kldata/absent-edges-expA-public-20261101/models/gemma4-31b-bf16-518276fb","revision":"518276fb130dc81caf9a4f772e65e63ef2526493"},"notes":["Create the unit's directory as thilak.arjun first (prepare_command): ReadWritePaths must exist.","One instance per GPU and one port per instance; the supervisor refuses a busy GPU or port.","Supervisor records: <cache_root>/records/*.json; journal: journalctl -u ae-expA-serve-golden1-g7"],"planned_by":{"egid":1009,"euid":1008,"gid":1009,"host":"ai-core-team-1","uid":1008,"user":"thilak.arjun"},"preflight_rules":{"cache_filesystem":"tmpfs","expected_uid_by_host":{"ai-core-team-1":1008,"ai-core-team-2":1006},"expected_user":"thilak.arjun","forbidden_indices":[5],"forbidden_minor_numbers":[5],"forbidden_uids_by_host":{"ai-core-team-1":[1006]},"forbidden_uuids":["GPU-7b8d9f89-0a8f-2d07-4833-bee70a0eeed2"],"max_idle_used_mib":128,"processes_allowed":0},"prepare_argv":["install","-d","-m","0700","/dev/shm/ae-expA-golden1","/dev/shm/ae-expA-golden1/serve","/dev/shm/ae-expA-golden1/serve/g7"],"prepare_command":"install -d -m 0700 /dev/shm/ae-expA-golden1 /dev/shm/ae-expA-golden1/serve /dev/shm/ae-expA-golden1/serve/g7","protocol":"expA-serve-v2","run_root":"/dev/shm/ae-expA-golden1","smoke_reference":"reports/expA-serve-smoke-20261101","spec":{"expected_user":"thilak.arjun","gpu_memory_utilization":0.92,"gpu_uuid":"GPU-4cb533cc-d12b-8e68-7177-39e79451bdad","instance":"g7","max_num_seqs":8,"memory_max":"192G","mirror_owner_uid":null,"model_check":"full","model_dir":"/mnt/kldata/absent-edges-expA-public-20261101/models/gemma4-31b-bf16-518276fb","model_manifest":"/mnt/kldata/absent-edges-expA-public-20261101/models/COPY.sha256","model_manifest_sha256":"7e67af6d6d633a7469c2463156f936c5a38979a7a3c61e662ce85abe0211987a","port":8141,"reobserve_seconds":2,"run":"golden1","runtime_max_seconds":518400,"runtime_root":"/mnt/kldata/absent-edges-expA-public-20261101/runtime/python3.11","serve_env":"/mnt/kldata/absent-edges-expA-public-20261101/envs/serve","startup_timeout_seconds":1800,"watchdog_seconds":30},"systemd_run_argv":["systemd-run","--unit=ae-expA-serve-golden1-g7","--collect","--description=Absent Edges Exp A vLLM golden1/g7 (GPU-4cb533cc-d12b-8e68-7177-39e79451bdad)","--property=Type=exec","--property=User=thilak.arjun","--property=Group=thilak.arjun","--property=WorkingDirectory=/dev/shm/ae-expA-golden1/serve/g7","--property=KillMode=control-group","--property=KillSignal=SIGTERM","--property=Restart=no","--property=TimeoutStopSec=180","--property=RuntimeMaxSec=518400","--property=MemoryMax=192G","--property=MemorySwapMax=0","--property=NoNewPrivileges=yes","--property=ProtectSystem=strict","--property=ProtectHome=yes","--property=InaccessiblePaths=-/data5t -/workspace -/mnt/eu-ai-datasets","--property=UMask=0077","--property=TemporaryFileSystem=/dev/shm:rw,nosuid,nodev,mode=1777,size=64G","--property=BindReadOnlyPaths=<T>/shm/ae-expA-golden1/inputs/release","--property=BindPaths=/dev/shm/ae-expA-golden1/serve/g7","--setenv=PATH=/mnt/kldata/absent-edges-expA-public-20261101/envs/serve/bin:/usr/bin:/bin","--setenv=LANG=C.UTF-8","--setenv=PYTHONDONTWRITEBYTECODE=1","--setenv=HOME=/dev/shm/ae-expA-golden1/serve/g7/home","--setenv=TMPDIR=/dev/shm/ae-expA-golden1/serve/g7/tmp","--setenv=CUDA_DEVICE_ORDER=PCI_BUS_ID","--","<T>/bin/python3.11","-I","-B","<T>/shm/ae-expA-golden1/inputs/release/scripts/expA_serve.py","supervise","--spec","{\"expected_user\":\"thilak.arjun\",\"gpu_memory_utilization\":0.92,\"gpu_uuid\":\"GPU-4cb533cc-d12b-8e68-7177-39e79451bdad\",\"instance\":\"g7\",\"max_num_seqs\":8,\"memory_max\":\"192G\",\"mirror_owner_uid\":null,\"model_check\":\"full\",\"model_dir\":\"/mnt/kldata/absent-edges-expA-public-20261101/models/gemma4-31b-bf16-518276fb\",\"model_manifest\":\"/mnt/kldata/absent-edges-expA-public-20261101/models/COPY.sha256\",\"model_manifest_sha256\":\"7e67af6d6d633a7469c2463156f936c5a38979a7a3c61e662ce85abe0211987a\",\"port\":8141,\"reobserve_seconds\":2,\"run\":\"golden1\",\"runtime_max_seconds\":518400,\"runtime_root\":\"/mnt/kldata/absent-edges-expA-public-20261101/runtime/python3.11\",\"serve_env\":\"/mnt/kldata/absent-edges-expA-public-20261101/envs/serve\",\"startup_timeout_seconds\":1800,\"watchdog_seconds\":30}"],"systemd_run_command":"sudo systemd-run --unit=ae-expA-serve-golden1-g7 --collect '--description=Absent Edges Exp A vLLM golden1/g7 (GPU-4cb533cc-d12b-8e68-7177-39e79451bdad)' --property=Type=exec --property=User=thilak.arjun --property=Group=thilak.arjun --property=WorkingDirectory=/dev/shm/ae-expA-golden1/serve/g7 --property=KillMode=control-group --property=KillSignal=SIGTERM --property=Restart=no --property=TimeoutStopSec=180 --property=RuntimeMaxSec=518400 --property=MemoryMax=192G --property=MemorySwapMax=0 --property=NoNewPrivileges=yes --property=ProtectSystem=strict --property=ProtectHome=yes '--property=InaccessiblePaths=-/data5t -/workspace -/mnt/eu-ai-datasets' --property=UMask=0077 --property=TemporaryFileSystem=/dev/shm:rw,nosuid,nodev,mode=1777,size=64G --property=BindReadOnlyPaths=<T>/shm/ae-expA-golden1/inputs/release --property=BindPaths=/dev/shm/ae-expA-golden1/serve/g7 --setenv=PATH=/mnt/kldata/absent-edges-expA-public-20261101/envs/serve/bin:/usr/bin:/bin --setenv=LANG=C.UTF-8 --setenv=PYTHONDONTWRITEBYTECODE=1 --setenv=HOME=/dev/shm/ae-expA-golden1/serve/g7/home --setenv=TMPDIR=/dev/shm/ae-expA-golden1/serve/g7/tmp --setenv=CUDA_DEVICE_ORDER=PCI_BUS_ID -- <T>/bin/python3.11 -I -B <T>/shm/ae-expA-golden1/inputs/release/scripts/expA_serve.py supervise --spec '{\"expected_user\":\"thilak.arjun\",\"gpu_memory_utilization\":0.92,\"gpu_uuid\":\"GPU-4cb533cc-d12b-8e68-7177-39e79451bdad\",\"instance\":\"g7\",\"max_num_seqs\":8,\"memory_max\":\"192G\",\"mirror_owner_uid\":null,\"model_check\":\"full\",\"model_dir\":\"/mnt/kldata/absent-edges-expA-public-20261101/models/gemma4-31b-bf16-518276fb\",\"model_manifest\":\"/mnt/kldata/absent-edges-expA-public-20261101/models/COPY.sha256\",\"model_manifest_sha256\":\"7e67af6d6d633a7469c2463156f936c5a38979a7a3c61e662ce85abe0211987a\",\"port\":8141,\"reobserve_seconds\":2,\"run\":\"golden1\",\"runtime_max_seconds\":518400,\"runtime_root\":\"/mnt/kldata/absent-edges-expA-public-20261101/runtime/python3.11\",\"serve_env\":\"/mnt/kldata/absent-edges-expA-public-20261101/envs/serve\",\"startup_timeout_seconds\":1800,\"watchdog_seconds\":30}'","unit":"ae-expA-serve-golden1-g7","vllm_argv":["/mnt/kldata/absent-edges-expA-public-20261101/envs/serve/bin/python","-m","vllm.entrypoints.openai.api_server","--model","/mnt/kldata/absent-edges-expA-public-20261101/models/gemma4-31b-bf16-518276fb","--served-model-name","absent-edges-gemma4-31b-bf16","--dtype","bfloat16","--host","127.0.0.1","--port","8141","--max-model-len","32768","--gpu-memory-utilization","0.92","--max-num-seqs","8","--language-model-only","--enable-auto-tool-choice","--tool-call-parser","gemma4","--reasoning-parser","gemma4","--generation-config","vllm","--enable-prefix-caching"],"watchdog":{"action":"record a collision, stop only our own process group, exit (ends only this unit)","interval_seconds":30,"reobserve_seconds":2,"rule":"a GPU PID not provably ours (descendant or our process group) on two consecutive observations is foreign; hidden /proc entries are never assumed ours"}},
  "serve_units": [["-n","systemd-run","--unit=ae-expA-serve-golden1-g7","--collect","--description=Absent Edges Exp A vLLM golden1/g7 (GPU-4cb533cc-d12b-8e68-7177-39e79451bdad)","--property=Type=exec","--property=User=thilak.arjun","--property=Group=thilak.arjun","--property=WorkingDirectory=/dev/shm/ae-expA-golden1/serve/g7","--property=KillMode=control-group","--property=KillSignal=SIGTERM","--property=Restart=no","--property=TimeoutStopSec=180","--property=RuntimeMaxSec=518400","--property=MemoryMax=192G","--property=MemorySwapMax=0","--property=NoNewPrivileges=yes","--property=ProtectSystem=strict","--property=ProtectHome=yes","--property=InaccessiblePaths=-/data5t -/workspace -/mnt/eu-ai-datasets","--property=UMask=0077","--property=TemporaryFileSystem=/dev/shm:rw,nosuid,nodev,mode=1777,size=64G","--property=BindReadOnlyPaths=<T>/shm/ae-expA-golden1/inputs/release","--property=BindPaths=/dev/shm/ae-expA-golden1/serve/g7","--setenv=PATH=/mnt/kldata/absent-edges-expA-public-20261101/envs/serve/bin:/usr/bin:/bin","--setenv=LANG=C.UTF-8","--setenv=PYTHONDONTWRITEBYTECODE=1","--setenv=HOME=/dev/shm/ae-expA-golden1/serve/g7/home","--setenv=TMPDIR=/dev/shm/ae-expA-golden1/serve/g7/tmp","--setenv=CUDA_DEVICE_ORDER=PCI_BUS_ID","--","<T>/bin/python3.11","-I","-B","<T>/shm/ae-expA-golden1/inputs/release/scripts/expA_serve.py","supervise","--spec","{\"expected_user\":\"thilak.arjun\",\"gpu_memory_utilization\":0.92,\"gpu_uuid\":\"GPU-4cb533cc-d12b-8e68-7177-39e79451bdad\",\"instance\":\"g7\",\"max_num_seqs\":8,\"memory_max\":\"192G\",\"mirror_owner_uid\":null,\"model_check\":\"full\",\"model_dir\":\"/mnt/kldata/absent-edges-expA-public-20261101/models/gemma4-31b-bf16-518276fb\",\"model_manifest\":\"/mnt/kldata/absent-edges-expA-public-20261101/models/COPY.sha256\",\"model_manifest_sha256\":\"7e67af6d6d633a7469c2463156f936c5a38979a7a3c61e662ce85abe0211987a\",\"port\":8141,\"reobserve_seconds\":2,\"run\":\"golden1\",\"runtime_max_seconds\":518400,\"runtime_root\":\"/mnt/kldata/absent-edges-expA-public-20261101/runtime/python3.11\",\"serve_env\":\"/mnt/kldata/absent-edges-expA-public-20261101/envs/serve\",\"startup_timeout_seconds\":1800,\"watchdog_seconds\":30}"]]
 },
 "runner": {
  "dry_run": {"code":0,"summary":{"by_block":{"P":30,"S-EDGE":2,"S-NAT":3,"S-VC":4,"T":2},"by_condition":{"P:none:C0":6,"P:omit_node:C1":6,"P:omit_node:C2":6,"P:omit_node:C3":6,"P:omit_node:C4a":6,"S-EDGE:omit_edge:C1":1,"S-EDGE:omit_edge:C3":1,"S-NAT:natural:C1":1,"S-NAT:natural:C3":1,"S-NAT:natural:C4a":1,"S-VC:vcomplete:C1":2,"S-VC:vcomplete:C3":2,"T:none:T-offered":1,"T:none:T-recommended":1},"by_family":{"fam":41},"design":{"block_order":["P","S-VC","S-NAT","S-EDGE","T"],"blocks":{"P":{"cells":[["none","C0"],["omit_node","C1"],["omit_node","C2"],["omit_node","C3"],["omit_node","C4a"]],"membership":null,"role":"primary (H1 C3-C2, H2 C4a-C1, H3 C1-C0)","seeds":[11,23,47]},"S-EDGE":{"cells":[["omit_edge","C1"],["omit_edge","C3"]],"membership":"in_s_edge","role":"secondary","seeds":[11]},"S-NAT":{"cells":[["natural","C1"],["natural","C3"],["natural","C4a"]],"membership":"in_s_nat","role":"secondary","seeds":[11]},"S-VC":{"cells":[["vcomplete","C1"],["vcomplete","C3"]],"membership":null,"role":"secondary","seeds":[11]},"T":{"cells":[["none","T-offered"],["none","T-recommended"]],"membership":"in_t","role":"exploratory","seeds":[11]}},"cell_id_format":"{block}|{item_id}|{variant}|{condition}|s{seed}","order":"block P seed by seed (11, 23, 47), items in sha256(order_salt + item_id) order, an item's cells together in the listed order; then S-VC, S-NAT, S-EDGE and T at seed 11","order_salt":"expA-order-v2|"},"design_cells":41,"dispatch":{"family_sources":{"fam":"a259ba9ebccbb8967e53c1e99e47cd0a3995002ccf539e0fa3733aba1add61d4"},"graph_facts":{"fam":"6555e12eb5d2efe35e22f3978a4aea06ff4a633ecfb465baee06856f413f8be6"},"item_build_code":null,"items_manifest_sha256":"2222222222222222222222222222222222222222222222222222222222222222","items_sha256":"1111111111111111111111111111111111111111111111111111111111111111","manifest_sha256":"1173bb24de4fb476a61a3261b2cf433eeeb4e20732ef5f72861868c37868b5bd","payload_store":{"fam":"ec1dab71b11c3a5c9979f58a79b0ea6d83cd6929ec0fdc77f6a7aca1c77d612d"},"rows":2,"schema":"expA-dispatch-v2","sha256":"20aa9c700bd228aae2d2f3acd8fcf72b38a46c02b5f1e54eb37a4beb3b45fe48"},"dispatch_manifest_blocks":null,"dispatch_manifest_design_cells":null,"examples":{"P:none:C0":{"cell_id":"P|item-b|none|C0|s11","system":"You are a careful software engineer. You have read-only access to a checkout of the Python repository fam through the tools provided. Use the tools to inspect the repository when needed, and finish by calling submit_answer. You have at most 8 responses; call submit_answer before they run out.","tools_offered":["source_reference_search","bash","submit_answer"],"user":"Task: find every place in the library source where the function `caller_one` (defined in `pkg/core.py`, line 5) is invoked, either called directly or passed as a callable that gets invoked, from inside another function or method. Consider only files under `pkg/`. Ignore tests, docs, examples and module-level code.\n\nSubmit each invocation site as a file path and line number with submit_answer, and state whether you believe your list is complete (yes, no or unknown)."},"P:omit_node:C1":{"cell_id":"P|item-b|omit_node|C1|s11","system":"You are a careful software engineer. You have read-only access to a checkout of the Python repository fam through the tools provided. Use the tools to inspect the repository when needed, and finish by calling submit_answer. You have at most 8 responses; call submit_answer before they run out.","tools_offered":["source_reference_search","bash","submit_answer"],"user":"Task: find every place in the library source where the function `caller_one` (defined in `pkg/core.py`, line 5) is invoked, either called directly or passed as a callable that gets invoked, from inside another function or method. Consider only files under `pkg/`. Ignore tests, docs, examples and module-level code.\n\nA code-graph index returned this answer for incoming `invokes` relationships of `pkg/core.py::caller_one`:\n{\"coverage\":{\"completeness\":\"unknown\"},\"coverage_totals\":{},\"endpoints\":[{\"end_line\":6,\"id\":\"pkg/core.py::caller_one\",\"path\":\"pkg/core.py\",\"snippet\":\"def caller_one():\\n    return target(1)\",\"snippet_end_line\":6,\"start_line\":5},{\"end_line\":2,\"id\":\"pkg/core.py::caller_one\",\"path\":\"pkg/core.py\",\"snippet\":\"def target(x):\\n    return x\",\"snippet_end_line\":2,\"start_line\":1}],\"error\":null,\"facts_sha256\":\"281bd423166ede0a172715dde6eb69e01cf0b50935b21f152e0ffa496bab4794\",\"matches\":[{\"id\":\"eeeeeeeeeeeeeeeeeeeeeeee\",\"relation\":\"invokes\",\"source\":\"pkg/core.py::caller_one\",\"target\":\"pkg/core.py::caller_one\"}],\"payload_truncated\":false,\"query\":{\"direction\":\"incoming\",\"relation\":\"invokes\",\"symbol\":\"pkg/core.py::caller_one\"},\"query_result_count\":1,\"returned_count\":1,\"source_sha256\":\"a259ba9ebccbb8967e53c1e99e47cd0a3995002ccf539e0fa3733aba1add61d4\",\"state\":\"matches\",\"symbol_indexed\":true,\"truncated\":false}\n\nSubmit each invocation site as a file path and line number with submit_answer, and state whether you believe your list is complete (yes, no or unknown)."}},"first_cells":["P|item-b|none|C0|s11","P|item-b|omit_node|C1|s11","P|item-b|omit_node|C2|s11","P|item-b|omit_node|C3|s11","P|item-b|omit_node|C4a|s11","P|item-a|none|C0|s11","P|item-a|omit_node|C1|s11","P|item-a|omit_node|C2|s11","P|item-a|omit_node|C3|s11","P|item-a|omit_node|C4a|s11"],"items":2,"mode":"dry-run","protocol":"expA-runner-v2","selected_cells":41,"sizes":{"P:none:C0":{"payload_bytes":{"max":0,"median":0.0,"min":0,"n":6},"system_plus_user_bytes":{"max":762,"median":760.0,"min":758,"n":6},"user_bytes":{"max":469,"median":467.0,"min":465,"n":6}},"P:omit_node:C1":{"payload_bytes":{"max":903,"median":897.0,"min":891,"n":6},"system_plus_user_bytes":{"max":1774,"median":1764.0,"min":1754,"n":6},"user_bytes":{"max":1481,"median":1471.0,"min":1461,"n":6}},"P:omit_node:C2":{"payload_bytes":{"max":1126,"median":1115.5,"min":1105,"n":6},"system_plus_user_bytes":{"max":1997,"median":1982.5,"min":1968,"n":6},"user_bytes":{"max":1704,"median":1689.5,"min":1675,"n":6}},"P:omit_node:C3":{"payload_bytes":{"max":1131,"median":1120.5,"min":1110,"n":6},"system_plus_user_bytes":{"max":2002,"median":1987.5,"min":1973,"n":6},"user_bytes":{"max":1709,"median":1694.5,"min":1680,"n":6}},"P:omit_node:C4a":{"payload_bytes":{"max":1572,"median":1452.5,"min":1333,"n":6},"system_plus_user_bytes":{"max":2435,"median":2319.5,"min":2204,"n":6},"user_bytes":{"max":2142,"median":2026.5,"min":1911,"n":6}},"S-EDGE:omit_edge:C1":{"payload_bytes":{"max":891,"median":891,"min":891,"n":1},"system_plus_user_bytes":{"max":1754,"median":1754,"min":1754,"n":1},"user_bytes":{"max":1461,"median":1461,"min":1461,"n":1}},"S-EDGE:omit_edge:C3":{"payload_bytes":{"max":1110,"median":1110,"min":1110,"n":1},"system_plus_user_bytes":{"max":1973,"median":1973,"min":1973,"n":1},"user_bytes":{"max":1680,"median":1680,"min":1680,"n":1}},"S-NAT:natural:C1":{"payload_bytes":{"max":891,"median":891,"min":891,"n":1},"system_plus_user_bytes":{"max":1754,"median":1754,"min":1754,"n":1},"user_bytes":{"max":1461,"median":1461,"min":1461,"n":1}},"S-NAT:natural:C3":{"payload_bytes":{"max":1110,"median":1110,"min":1110,"n":1},"system_plus_user_bytes":{"max":1973,"median":1973,"min":1973,"n":1},"user_bytes":{"max":1680,"median":1680,"min":1680,"n":1}},"S-NAT:natural:C4a":{"payload_bytes":{"max":1572,"median":1572,"min":1572,"n":1},"system_plus_user_bytes":{"max":2435,"median":2435,"min":2435,"n":1},"user_bytes":{"max":2142,"median":2142,"min":2142,"n":1}},"S-VC:vcomplete:C1":{"payload_bytes":{"max":903,"median":897.0,"min":891,"n":2},"system_plus_user_bytes":{"max":1774,"median":1764.0,"min":1754,"n":2},"user_bytes":{"max":1481,"median":1471.0,"min":1461,"n":2}},"S-VC:vcomplete:C3":{"payload_bytes":{"max":1131,"median":1120.5,"min":1110,"n":2},"system_plus_user_bytes":{"max":2002,"median":1987.5,"min":1973,"n":2},"user_bytes":{"max":1709,"median":1694.5,"min":1680,"n":2}},"T:none:T-offered":{"payload_bytes":{"max":0,"median":0,"min":0,"n":1},"system_plus_user_bytes":{"max":758,"median":758,"min":758,"n":1},"user_bytes":{"max":465,"median":465,"min":465,"n":1}},"T:none:T-recommended":{"payload_bytes":{"max":0,"median":0,"min":0,"n":1},"system_plus_user_bytes":{"max":817,"median":817,"min":817,"n":1},"user_bytes":{"max":524,"median":524,"min":524,"n":1}}},"unique_prompts":21,"writes":"none"}},
  "live": {"chat_request":{"max_tokens":4096,"model":"absent-edges-gemma4-31b-bf16","seed":11,"temperature":0.2},"exit_code":0,"result":{"config":{"context_budget":16384,"max_consecutive_format_errors":3,"max_model_calls":8,"max_response_tokens":4096,"min_response_tokens":512,"temperature":0.2},"exit_status":"Submitted","model_profile":{"enable_thinking":false,"endpoint":"<fake server>","max_context":32768,"model":"absent-edges-gemma4-31b-bf16","revision":"518276fb130dc81caf9a4f772e65e63ef2526493"}},"results":1,"run_json":{"design_sha256":"b267d7d916694d2e4852b2cc6224cb603b7d07d49401fb1e2b5cba476722d040","episode_config":{"context_budget":16384,"max_consecutive_format_errors":3,"max_model_calls":8,"max_response_tokens":4096,"min_response_tokens":512,"temperature":0.2},"infra_attempt_cap":3,"model":{"alias":"absent-edges-gemma4-31b-bf16","enable_thinking":false,"max_context":32768,"temperature":0.2},"prompts_sha256":"ecf896beb4cba9e9400cbfdb797c2d38031e6cf41a35719b4a13933e8a5abeb9","protocol":"expA-runner-v2","tools_sha256":"c989baf09ef078b9ada10b5a19d49856e13429386906dad61061d1fa1f207a51"},"setup":{"config":{"context_budget":16384,"max_consecutive_format_errors":3,"max_model_calls":8,"max_response_tokens":4096,"min_response_tokens":512,"temperature":0.2},"model_profile":{"enable_thinking":false,"endpoint":"<fake server>","max_context":32768,"model":"absent-edges-gemma4-31b-bf16","revision":"518276fb130dc81caf9a4f772e65e63ef2526493"},"system":"You are a careful software engineer. You have read-only access to a checkout of the Python repository fam through the tools provided. Use the tools to inspect the repository when needed, and finish by calling submit_answer. You have at most 8 responses; call submit_answer before they run out."}}
 },
 "serve": {
  "api": {"environment":{"CUDA_CACHE_PATH":"/dev/shm/ae-expA-golden1/serve/g7/cache/cuda_cache_path","CUDA_DEVICE_ORDER":"PCI_BUS_ID","CUDA_VISIBLE_DEVICES":"GPU-4cb533cc-d12b-8e68-7177-39e79451bdad","DO_NOT_TRACK":"1","FLASHINFER_WORKSPACE_BASE":"/dev/shm/ae-expA-golden1/serve/g7/cache/flashinfer_workspace_base","HF_HOME":"/dev/shm/ae-expA-golden1/serve/g7/cache/hf_home","HF_HUB_CACHE":"/dev/shm/ae-expA-golden1/serve/g7/cache/hf_hub_cache","HF_HUB_DISABLE_TELEMETRY":"1","HF_HUB_OFFLINE":"1","HOME":"/dev/shm/ae-expA-golden1/serve/g7/home","LANG":"C.UTF-8","MKL_NUM_THREADS":"1","OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","PATH":"/mnt/kldata/absent-edges-expA-public-20261101/envs/serve/bin:/usr/bin:/bin","PIP_CACHE_DIR":"/dev/shm/ae-expA-golden1/serve/g7/cache/pip_cache_dir","PYTHONDONTWRITEBYTECODE":"1","PYTHONNOUSERSITE":"1","PYTHONPYCACHEPREFIX":"/dev/shm/ae-expA-golden1/serve/g7/cache/pythonpycacheprefix","TEMP":"/dev/shm/ae-expA-golden1/serve/g7/tmp","TMP":"/dev/shm/ae-expA-golden1/serve/g7/tmp","TMPDIR":"/dev/shm/ae-expA-golden1/serve/g7/tmp","TORCHINDUCTOR_CACHE_DIR":"/dev/shm/ae-expA-golden1/serve/g7/cache/torchinductor_cache_dir","TRANSFORMERS_OFFLINE":"1","TRITON_CACHE_DIR":"/dev/shm/ae-expA-golden1/serve/g7/cache/triton_cache_dir","UV_CACHE_DIR":"/dev/shm/ae-expA-golden1/serve/g7/cache/uv_cache_dir","VLLM_CACHE_ROOT":"/dev/shm/ae-expA-golden1/serve/g7/cache/vllm_cache_root","VLLM_CONFIG_ROOT":"/dev/shm/ae-expA-golden1/serve/g7/cache/vllm_config_root","VLLM_NO_USAGE_STATS":"1","XDG_CACHE_HOME":"/dev/shm/ae-expA-golden1/serve/g7/cache/xdg_cache_home","XDG_CONFIG_HOME":"/dev/shm/ae-expA-golden1/serve/g7/cache/xdg_config_home","XDG_DATA_HOME":"/dev/shm/ae-expA-golden1/serve/g7/cache/xdg_data_home"},"spec":{"expected_user":"thilak.arjun","gpu_memory_utilization":0.92,"gpu_uuid":"GPU-4cb533cc-d12b-8e68-7177-39e79451bdad","instance":"g7","max_num_seqs":8,"memory_max":"192G","mirror_owner_uid":null,"model_check":"full","model_dir":"/mnt/kldata/absent-edges-expA-public-20261101/models/gemma4-31b-bf16-518276fb","model_manifest":"/mnt/kldata/absent-edges-expA-public-20261101/models/COPY.sha256","model_manifest_sha256":"7e67af6d6d633a7469c2463156f936c5a38979a7a3c61e662ce85abe0211987a","port":8141,"reobserve_seconds":2,"run":"golden1","runtime_max_seconds":518400,"runtime_root":"/mnt/kldata/absent-edges-expA-public-20261101/runtime/python3.11","serve_env":"/mnt/kldata/absent-edges-expA-public-20261101/envs/serve","startup_timeout_seconds":1800,"watchdog_seconds":30},"vllm_argv":["/mnt/kldata/absent-edges-expA-public-20261101/envs/serve/bin/python","-m","vllm.entrypoints.openai.api_server","--model","/mnt/kldata/absent-edges-expA-public-20261101/models/gemma4-31b-bf16-518276fb","--served-model-name","absent-edges-gemma4-31b-bf16","--dtype","bfloat16","--host","127.0.0.1","--port","8141","--max-model-len","32768","--gpu-memory-utilization","0.92","--max-num-seqs","8","--language-model-only","--enable-auto-tool-choice","--tool-call-parser","gemma4","--reasoning-parser","gemma4","--generation-config","vllm","--enable-prefix-caching"]},
  "constants": {"MAX_MODEL_LEN":32768,"MODEL_ALIAS":"absent-edges-gemma4-31b-bf16","MODEL_ALLOWED_EXTRA":[".gitattributes"],"MODEL_DIR":"/mnt/kldata/absent-edges-expA-public-20261101/models/gemma4-31b-bf16-518276fb","MODEL_MANIFEST":"/mnt/kldata/absent-edges-expA-public-20261101/models/COPY.sha256","MODEL_MANIFEST_SHA256":"7e67af6d6d633a7469c2463156f936c5a38979a7a3c61e662ce85abe0211987a","MODEL_REQUIRED_FILES":["config.json","tokenizer.json","tokenizer_config.json","chat_template.jinja","generation_config.json","model.safetensors.index.json"],"MODEL_REVISION":"518276fb130dc81caf9a4f772e65e63ef2526493","SERVE_PROTOCOL":"expA-serve-v2"},
  "plan_launch_script": {"code":0,"plan":{"cache_root":"/dev/shm/ae-expA-golden1/serve/g7","endpoint":"http://127.0.0.1:8141","environment":{"CUDA_CACHE_PATH":"/dev/shm/ae-expA-golden1/serve/g7/cache/cuda_cache_path","CUDA_DEVICE_ORDER":"PCI_BUS_ID","CUDA_VISIBLE_DEVICES":"GPU-4cb533cc-d12b-8e68-7177-39e79451bdad","DO_NOT_TRACK":"1","FLASHINFER_WORKSPACE_BASE":"/dev/shm/ae-expA-golden1/serve/g7/cache/flashinfer_workspace_base","HF_HOME":"/dev/shm/ae-expA-golden1/serve/g7/cache/hf_home","HF_HUB_CACHE":"/dev/shm/ae-expA-golden1/serve/g7/cache/hf_hub_cache","HF_HUB_DISABLE_TELEMETRY":"1","HF_HUB_OFFLINE":"1","HOME":"/dev/shm/ae-expA-golden1/serve/g7/home","LANG":"C.UTF-8","MKL_NUM_THREADS":"1","OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","PATH":"/mnt/kldata/absent-edges-expA-public-20261101/envs/serve/bin:/usr/bin:/bin","PIP_CACHE_DIR":"/dev/shm/ae-expA-golden1/serve/g7/cache/pip_cache_dir","PYTHONDONTWRITEBYTECODE":"1","PYTHONNOUSERSITE":"1","PYTHONPYCACHEPREFIX":"/dev/shm/ae-expA-golden1/serve/g7/cache/pythonpycacheprefix","TEMP":"/dev/shm/ae-expA-golden1/serve/g7/tmp","TMP":"/dev/shm/ae-expA-golden1/serve/g7/tmp","TMPDIR":"/dev/shm/ae-expA-golden1/serve/g7/tmp","TORCHINDUCTOR_CACHE_DIR":"/dev/shm/ae-expA-golden1/serve/g7/cache/torchinductor_cache_dir","TRANSFORMERS_OFFLINE":"1","TRITON_CACHE_DIR":"/dev/shm/ae-expA-golden1/serve/g7/cache/triton_cache_dir","UV_CACHE_DIR":"/dev/shm/ae-expA-golden1/serve/g7/cache/uv_cache_dir","VLLM_CACHE_ROOT":"/dev/shm/ae-expA-golden1/serve/g7/cache/vllm_cache_root","VLLM_CONFIG_ROOT":"/dev/shm/ae-expA-golden1/serve/g7/cache/vllm_config_root","VLLM_NO_USAGE_STATS":"1","XDG_CACHE_HOME":"/dev/shm/ae-expA-golden1/serve/g7/cache/xdg_cache_home","XDG_CONFIG_HOME":"/dev/shm/ae-expA-golden1/serve/g7/cache/xdg_config_home","XDG_DATA_HOME":"/dev/shm/ae-expA-golden1/serve/g7/cache/xdg_data_home"},"flags_source":"staged/core1-thinking-study-20261031a/source/common.py","model":{"alias":"absent-edges-gemma4-31b-bf16","check":"full","manifest":"/mnt/kldata/absent-edges-expA-public-20261101/models/COPY.sha256","manifest_sha256":"7e67af6d6d633a7469c2463156f936c5a38979a7a3c61e662ce85abe0211987a","max_model_len":32768,"path":"/mnt/kldata/absent-edges-expA-public-20261101/models/gemma4-31b-bf16-518276fb","revision":"518276fb130dc81caf9a4f772e65e63ef2526493"},"notes":["Create the unit's directory as thilak.arjun first (prepare_command): ReadWritePaths must exist.","One instance per GPU and one port per instance; the supervisor refuses a busy GPU or port.","Supervisor records: <cache_root>/records/*.json; journal: journalctl -u ae-expA-serve-golden1-g7"],"planned_by":{"egid":1009,"euid":1008,"gid":1009,"host":"ai-core-team-1","uid":1008,"user":"thilak.arjun"},"preflight_rules":{"cache_filesystem":"tmpfs","expected_uid_by_host":{"ai-core-team-1":1008,"ai-core-team-2":1006},"expected_user":"thilak.arjun","forbidden_indices":[5],"forbidden_minor_numbers":[5],"forbidden_uids_by_host":{"ai-core-team-1":[1006]},"forbidden_uuids":["GPU-7b8d9f89-0a8f-2d07-4833-bee70a0eeed2"],"max_idle_used_mib":128,"processes_allowed":0},"prepare_argv":["install","-d","-m","0700","/dev/shm/ae-expA-golden1","/dev/shm/ae-expA-golden1/serve","/dev/shm/ae-expA-golden1/serve/g7"],"prepare_command":"install -d -m 0700 /dev/shm/ae-expA-golden1 /dev/shm/ae-expA-golden1/serve /dev/shm/ae-expA-golden1/serve/g7","protocol":"expA-serve-v2","run_root":"/dev/shm/ae-expA-golden1","smoke_reference":"reports/expA-serve-smoke-20261101","spec":{"expected_user":"thilak.arjun","gpu_memory_utilization":0.92,"gpu_uuid":"GPU-4cb533cc-d12b-8e68-7177-39e79451bdad","instance":"g7","max_num_seqs":8,"memory_max":"192G","mirror_owner_uid":null,"model_check":"full","model_dir":"/mnt/kldata/absent-edges-expA-public-20261101/models/gemma4-31b-bf16-518276fb","model_manifest":"/mnt/kldata/absent-edges-expA-public-20261101/models/COPY.sha256","model_manifest_sha256":"7e67af6d6d633a7469c2463156f936c5a38979a7a3c61e662ce85abe0211987a","port":8141,"reobserve_seconds":2,"run":"golden1","runtime_max_seconds":518400,"runtime_root":"/mnt/kldata/absent-edges-expA-public-20261101/runtime/python3.11","serve_env":"/mnt/kldata/absent-edges-expA-public-20261101/envs/serve","startup_timeout_seconds":1800,"watchdog_seconds":30},"systemd_run_argv":["systemd-run","--unit=ae-expA-serve-golden1-g7","--collect","--description=Absent Edges Exp A vLLM golden1/g7 (GPU-4cb533cc-d12b-8e68-7177-39e79451bdad)","--property=Type=exec","--property=User=thilak.arjun","--property=Group=thilak.arjun","--property=WorkingDirectory=/dev/shm/ae-expA-golden1/serve/g7","--property=KillMode=control-group","--property=KillSignal=SIGTERM","--property=Restart=no","--property=TimeoutStopSec=180","--property=RuntimeMaxSec=518400","--property=MemoryMax=192G","--property=MemorySwapMax=0","--property=NoNewPrivileges=yes","--property=ProtectSystem=strict","--property=ProtectHome=yes","--property=InaccessiblePaths=-/data5t -/workspace -/mnt/eu-ai-datasets","--property=UMask=0077","--property=TemporaryFileSystem=/dev/shm:rw,nosuid,nodev,mode=1777,size=64G","--property=BindReadOnlyPaths=/dev/shm/ae-expA-golden1/inputs/release","--property=BindPaths=/dev/shm/ae-expA-golden1/serve/g7","--setenv=PATH=/mnt/kldata/absent-edges-expA-public-20261101/envs/serve/bin:/usr/bin:/bin","--setenv=LANG=C.UTF-8","--setenv=PYTHONDONTWRITEBYTECODE=1","--setenv=HOME=/dev/shm/ae-expA-golden1/serve/g7/home","--setenv=TMPDIR=/dev/shm/ae-expA-golden1/serve/g7/tmp","--setenv=CUDA_DEVICE_ORDER=PCI_BUS_ID","--","/mnt/kldata/absent-edges-expA-public-20261101/runtime/python3.11/bin/python3.11","-I","-B","/dev/shm/ae-expA-golden1/inputs/release/scripts/expA_serve.py","supervise","--spec","{\"expected_user\":\"thilak.arjun\",\"gpu_memory_utilization\":0.92,\"gpu_uuid\":\"GPU-4cb533cc-d12b-8e68-7177-39e79451bdad\",\"instance\":\"g7\",\"max_num_seqs\":8,\"memory_max\":\"192G\",\"mirror_owner_uid\":null,\"model_check\":\"full\",\"model_dir\":\"/mnt/kldata/absent-edges-expA-public-20261101/models/gemma4-31b-bf16-518276fb\",\"model_manifest\":\"/mnt/kldata/absent-edges-expA-public-20261101/models/COPY.sha256\",\"model_manifest_sha256\":\"7e67af6d6d633a7469c2463156f936c5a38979a7a3c61e662ce85abe0211987a\",\"port\":8141,\"reobserve_seconds\":2,\"run\":\"golden1\",\"runtime_max_seconds\":518400,\"runtime_root\":\"/mnt/kldata/absent-edges-expA-public-20261101/runtime/python3.11\",\"serve_env\":\"/mnt/kldata/absent-edges-expA-public-20261101/envs/serve\",\"startup_timeout_seconds\":1800,\"watchdog_seconds\":30}"],"systemd_run_command":"sudo systemd-run --unit=ae-expA-serve-golden1-g7 --collect '--description=Absent Edges Exp A vLLM golden1/g7 (GPU-4cb533cc-d12b-8e68-7177-39e79451bdad)' --property=Type=exec --property=User=thilak.arjun --property=Group=thilak.arjun --property=WorkingDirectory=/dev/shm/ae-expA-golden1/serve/g7 --property=KillMode=control-group --property=KillSignal=SIGTERM --property=Restart=no --property=TimeoutStopSec=180 --property=RuntimeMaxSec=518400 --property=MemoryMax=192G --property=MemorySwapMax=0 --property=NoNewPrivileges=yes --property=ProtectSystem=strict --property=ProtectHome=yes '--property=InaccessiblePaths=-/data5t -/workspace -/mnt/eu-ai-datasets' --property=UMask=0077 --property=TemporaryFileSystem=/dev/shm:rw,nosuid,nodev,mode=1777,size=64G --property=BindReadOnlyPaths=/dev/shm/ae-expA-golden1/inputs/release --property=BindPaths=/dev/shm/ae-expA-golden1/serve/g7 --setenv=PATH=/mnt/kldata/absent-edges-expA-public-20261101/envs/serve/bin:/usr/bin:/bin --setenv=LANG=C.UTF-8 --setenv=PYTHONDONTWRITEBYTECODE=1 --setenv=HOME=/dev/shm/ae-expA-golden1/serve/g7/home --setenv=TMPDIR=/dev/shm/ae-expA-golden1/serve/g7/tmp --setenv=CUDA_DEVICE_ORDER=PCI_BUS_ID -- /mnt/kldata/absent-edges-expA-public-20261101/runtime/python3.11/bin/python3.11 -I -B /dev/shm/ae-expA-golden1/inputs/release/scripts/expA_serve.py supervise --spec '{\"expected_user\":\"thilak.arjun\",\"gpu_memory_utilization\":0.92,\"gpu_uuid\":\"GPU-4cb533cc-d12b-8e68-7177-39e79451bdad\",\"instance\":\"g7\",\"max_num_seqs\":8,\"memory_max\":\"192G\",\"mirror_owner_uid\":null,\"model_check\":\"full\",\"model_dir\":\"/mnt/kldata/absent-edges-expA-public-20261101/models/gemma4-31b-bf16-518276fb\",\"model_manifest\":\"/mnt/kldata/absent-edges-expA-public-20261101/models/COPY.sha256\",\"model_manifest_sha256\":\"7e67af6d6d633a7469c2463156f936c5a38979a7a3c61e662ce85abe0211987a\",\"port\":8141,\"reobserve_seconds\":2,\"run\":\"golden1\",\"runtime_max_seconds\":518400,\"runtime_root\":\"/mnt/kldata/absent-edges-expA-public-20261101/runtime/python3.11\",\"serve_env\":\"/mnt/kldata/absent-edges-expA-public-20261101/envs/serve\",\"startup_timeout_seconds\":1800,\"watchdog_seconds\":30}'","unit":"ae-expA-serve-golden1-g7","vllm_argv":["/mnt/kldata/absent-edges-expA-public-20261101/envs/serve/bin/python","-m","vllm.entrypoints.openai.api_server","--model","/mnt/kldata/absent-edges-expA-public-20261101/models/gemma4-31b-bf16-518276fb","--served-model-name","absent-edges-gemma4-31b-bf16","--dtype","bfloat16","--host","127.0.0.1","--port","8141","--max-model-len","32768","--gpu-memory-utilization","0.92","--max-num-seqs","8","--language-model-only","--enable-auto-tool-choice","--tool-call-parser","gemma4","--reasoning-parser","gemma4","--generation-config","vllm","--enable-prefix-caching"],"watchdog":{"action":"record a collision, stop only our own process group, exit (ends only this unit)","interval_seconds":30,"reobserve_seconds":2,"rule":"a GPU PID not provably ours (descendant or our process group) on two consecutive observations is foreign; hidden /proc entries are never assumed ours"}}},
  "verify_model": {"calls":[{"expected_manifest_sha256":"7e67af6d6d633a7469c2463156f936c5a38979a7a3c61e662ce85abe0211987a","manifest":"/mnt/kldata/absent-edges-expA-public-20261101/models/COPY.sha256","mode":"full","model_dir":"/mnt/kldata/absent-edges-expA-public-20261101/models/gemma4-31b-bf16-518276fb","quick_hash_limit":268435456,"workers":4}],"code":0}
 }
}
""")


@unittest.skipUnless(POSIX and shutil.which("bash"), "Linux only: serve.py imports pwd; the runner needs POSIX reads")
class DefaultInvocationGoldenTests(unittest.TestCase):
    maxDiff = None

    @classmethod
    def setUpClass(cls):
        cls.values = capture()

    def assertKeeps(self, golden, actual):
        self.assertEqual(contained(golden, actual), [])

    def test_serve_commands_are_byte_identical(self):
        for name in ("plan_launch_script", "plan_cli_defaults"):
            golden, actual = GOLDEN["serve"]["plan_launch_script"], self.values["serve"][name]
            with self.subTest(plan=name):
                self.assertEqual(actual["code"], golden["code"])
                for key in ("vllm_argv", "systemd_run_argv", "systemd_run_command", "prepare_argv", "environment"):
                    self.assertEqual(actual["plan"][key], golden["plan"][key], key)
        self.assertEqual(self.values["serve"]["api"]["vllm_argv"], GOLDEN["serve"]["api"]["vllm_argv"])
        self.assertEqual(self.values["serve"]["api"]["environment"], GOLDEN["serve"]["api"]["environment"])

    def test_serve_plan_and_spec_keep_every_value(self):
        for name in ("plan_launch_script", "plan_cli_defaults"):
            with self.subTest(plan=name):
                self.assertKeeps(GOLDEN["serve"]["plan_launch_script"], self.values["serve"][name])
        self.assertKeeps(GOLDEN["serve"]["api"]["spec"], self.values["serve"]["api"]["spec"])

    def test_verify_model_defaults_and_constants(self):
        golden, actual = GOLDEN["serve"]["verify_model"], self.values["serve"]["verify_model"]
        self.assertEqual((actual["code"], len(actual["calls"])), (golden["code"], 1))
        self.assertKeeps(golden["calls"][0], actual["calls"][0])
        self.assertKeeps(GOLDEN["serve"]["constants"], self.values["serve"]["constants"])

    def test_runner_dry_run_keeps_every_value(self):
        self.assertKeeps(GOLDEN["runner"]["dry_run"], self.values["runner"]["dry_run"])

    def test_runner_run_json_and_default_episode(self):
        golden, actual = GOLDEN["runner"]["live"], self.values["runner"]["live"]
        self.assertEqual((actual["exit_code"], actual["results"]), (0, 1))
        self.assertKeeps(golden["run_json"], actual["run_json"])
        self.assertEqual(actual["run_json"]["episode_config"], golden["run_json"]["episode_config"])
        for key in ("result", "setup", "chat_request"):
            self.assertEqual(actual[key], golden[key], key)

    def test_launch_script_default_path(self):
        golden, actual = GOLDEN["launch"], self.values["launch"]
        self.assertEqual(actual["bash_n"], 0)
        self.assertTrue(actual["launch_env"].startswith(golden["launch_env"]), actual["launch_env"])
        self.assertTrue(actual["prepare_stdout"].startswith(golden["prepare_stdout"]), actual["prepare_stdout"])
        self.assertEqual(actual["serve_units"], golden["serve_units"])
        self.assertEqual(actual["runner_units"], golden["runner_units"])
        self.assertEqual(actual["endpoint_ready"], golden["endpoint_ready"])
        self.assertKeeps(golden["serve_plan"], actual["serve_plan"])


if __name__ == "__main__":
    if sys.argv[1:] == ["--capture"]:
        print(json.dumps(capture(), indent=1, sort_keys=True, ensure_ascii=False))
    else:
        unittest.main()
