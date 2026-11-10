"""Exp A Core1 runtime: serving preflight and identity guard (F3), watchdog fail-closed (F4), the root
stager, the root publisher, the clean-checkout mirror (F11) and the operator script.

No GPU, no vLLM, no root: GPU observations, identities and mounts are injected; the stager and the
publisher run as the test user, chowning to that user. Linux only (os.fwalk, dir_fd, process groups).
"""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import pwd
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from absent_edges.data import canonical_json
from absent_edges.probe import serve, tools
from absent_edges.probe.identity import file_sha256s, scan_package_files, source_tree_sha256

PROJECT = Path(__file__).resolve().parents[1]
GPU7 = "GPU-4cb533cc-d12b-8e68-7177-39e79451bdad"
CORE1_OK = {"uid": 1008, "euid": 1008, "gid": 1009, "egid": 1009, "user": "thilak.arjun", "host": "ai-core-team-1"}
POSIX = os.name == "posix" and hasattr(os, "fwalk")


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"runtime_test_{name}", PROJECT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STAGE = load_script("expA_stage")
PUBLISH = load_script("expA_publish")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(path, data, mode=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data if isinstance(data, bytes) else data.encode())
    if mode is not None:
        os.chmod(path, mode)
    return path


def observation(pids=(), used=0, index=7, minor=7, uuid=GPU7):
    return {"uuid": uuid, "index": index, "minor_number": minor, "used_mib": used, "total_mib": 97887,
            "pids": sorted(pids)}


def spec(**overrides):
    values = dict(run="t1", instance="g7", gpu_uuid=GPU7, port=8201)
    values.update(overrides)
    return serve.ServeSpec(**values)


def model_fixture(root):
    """A tiny model directory with a COPY.sha256-style manifest; returns (model_dir, manifest, manifest_sha)."""
    model = Path(root) / "model"
    names = list(serve.MODEL_REQUIRED_FILES) + ["model-00001-of-00001.safetensors"]
    lines = []
    for name in sorted(names):
        data = f"content of {name}\n".encode()
        write(model / name, data)
        lines.append(f"{sha(data)}  {name}")
    write(model / ".gitattributes", "*.safetensors filter=lfs\n")
    manifest = write(Path(root) / "COPY.sha256", "\n".join(lines) + "\n")
    return model, manifest, sha(manifest.read_bytes())


# ---------------------------------------------------------------------------
# Serving: identity guard (F3)
# ---------------------------------------------------------------------------

class ServeIdentityTests(unittest.TestCase):
    def test_only_thilak_arjun_with_its_pinned_uid(self):
        self.assertEqual(serve.identity_failures(CORE1_OK), [])
        self.assertEqual(serve.identity_failures(dict(CORE1_OK, uid=1006, euid=1006, host="ai-core-team-2")), [])
        self.assertEqual(serve.identity_failures(dict(CORE1_OK, uid=4242, euid=4242, host="elsewhere")), [])

    def test_core1_uid_1006_is_refused_whatever_its_name(self):
        other = dict(CORE1_OK, uid=1006, euid=1006, user="[redacted-other-user]")
        failures = serve.identity_failures(other)
        self.assertIn("uid_1006_forbidden_on_ai-core-team-1", failures)
        self.assertIn("effective_user_[redacted-other-user]_is_not_thilak.arjun", failures)
        renamed = serve.identity_failures(dict(CORE1_OK, uid=1006, euid=1006))
        self.assertIn("uid_1006_forbidden_on_ai-core-team-1", renamed)
        self.assertIn("uid_1006_is_not_thilak.arjun_uid_1008_on_ai-core-team-1", renamed)
        fqdn = serve.identity_failures(dict(other, host="ai-core-team-1.internal"))
        self.assertIn("uid_1006_forbidden_on_ai-core-team-1", fqdn)

    def test_root_and_setuid_are_refused(self):
        self.assertIn("root_forbidden", serve.identity_failures(dict(CORE1_OK, uid=0, euid=0, user="root")))
        self.assertIn("uid_euid_mismatch", serve.identity_failures(dict(CORE1_OK, uid=1009)))

    def test_plan_and_require_identity_refuse(self):
        with self.assertRaises(serve.ServePreflightError):
            serve.require_identity(identity=dict(CORE1_OK, uid=1006, euid=1006, user="[redacted-other-user]"))
        with self.assertRaises(serve.ServePreflightError):
            serve.plan(spec(), identity=dict(CORE1_OK, uid=0, euid=0, user="root"))
        self.assertEqual(serve.plan(spec(), identity=CORE1_OK)["planned_by"], CORE1_OK)


# ---------------------------------------------------------------------------
# Serving: plan (public mirror defaults, smoke lessons, hardening)
# ---------------------------------------------------------------------------

class ServePlanTests(unittest.TestCase):
    def test_defaults_are_the_public_mirror(self):
        value = spec()
        public = "/mnt/kldata/absent-edges-expA-public-20261101/"
        self.assertTrue(value.serve_env.startswith(public) and value.model_dir.startswith(public)
                        and value.model_manifest.startswith(public))
        self.assertTrue(str(serve.RUNTIME_PYTHON).startswith(public + "runtime/python3.11/"))
        self.assertNotIn("absent-edges-20261018-core2", json.dumps(serve.plan(value, identity=CORE1_OK)))
        self.assertEqual((value.max_num_seqs, value.gpu_memory_utilization, serve.MAX_MODEL_LEN), (8, 0.92, 32768))

    def test_every_path_is_a_parameter(self):
        value = spec(serve_env="/opt/env", model_dir="/opt/model", model_manifest="/opt/COPY.sha256")
        plan = serve.plan(value, release_root="/rel", runtime_python="/py/bin/python3", identity=CORE1_OK)
        argv = plan["vllm_argv"]
        self.assertEqual(argv[0], "/opt/env/bin/python")
        self.assertEqual(argv[argv.index("--model") + 1], "/opt/model")
        self.assertTrue(plan["environment"]["PATH"].startswith("/opt/env/bin:"))
        unit = plan["systemd_run_argv"]
        tail = unit[unit.index("--") + 1:]
        self.assertEqual(tail[:5], ["/py/bin/python3", "-I", "-B", "/rel/scripts/expA_serve.py", "supervise"])
        self.assertEqual(serve.ServeSpec(**json.loads(tail[-1])), value)
        for bad in (dict(serve_env="relative/env"), dict(model_dir="/x\n/y")):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                spec(**bad)

    def test_vllm_flags_and_environment(self):
        value = spec()
        argv = serve.vllm_argv(value)
        pairs = {argv[i]: argv[i + 1] for i in range(len(argv) - 1) if argv[i].startswith("--")}
        self.assertEqual((pairs["--max-num-seqs"], pairs["--gpu-memory-utilization"], pairs["--max-model-len"],
                          pairs["--host"], pairs["--port"], pairs["--served-model-name"]),
                         ("8", "0.92", "32768", "127.0.0.1", "8201", serve.MODEL_ALIAS))
        for flag in ("--enable-prefix-caching", "--enable-auto-tool-choice", "--language-model-only"):
            self.assertIn(flag, argv)
        env = serve.serve_environment(value)
        root = "/dev/shm/ae-expA-t1/serve/g7/"
        self.assertTrue(env["PATH"].startswith(value.serve_env + "/bin:"))  # ninja (smoke run a)
        self.assertEqual((env["LANG"], env["CUDA_DEVICE_ORDER"], env["CUDA_VISIBLE_DEVICES"]),
                         ("C.UTF-8", "PCI_BUS_ID", GPU7))
        for name in ("TMPDIR", "HOME", "HF_HOME", "VLLM_CACHE_ROOT", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR",
                     "XDG_CACHE_HOME", "FLASHINFER_WORKSPACE_BASE", "CUDA_CACHE_PATH"):
            self.assertTrue(env[name].startswith(root), name)
        self.assertFalse(set(env) & {"SSH_AUTH_SOCK", "HF_TOKEN", "USER", "KAGGLE_KEY", "OPENAI_API_KEY"})

    def test_unit_is_hardened_and_runs_as_thilak_arjun(self):
        plan = serve.plan(spec(), identity=CORE1_OK)
        argv = plan["systemd_run_argv"]
        self.assertTrue(plan["systemd_run_command"].startswith("sudo systemd-run --unit=ae-expA-serve-t1-g7 "))
        properties = [value.split("=", 1)[1] for value in argv if value.startswith("--property=")]
        for expected in ("User=thilak.arjun", "Group=thilak.arjun", "NoNewPrivileges=yes", "ProtectSystem=strict",
                         "ProtectHome=yes", "KillMode=control-group", "Restart=no",
                         "InaccessiblePaths=-/data5t -/workspace -/mnt/eu-ai-datasets",
                         # vLLM needs POSIX semaphores and shared memory in /dev/shm (EROFS under ReadOnlyPaths in
                         # pilot-20261101a): a private tmpfs, with only the release (ro) and the instance dir bound in.
                         "TemporaryFileSystem=/dev/shm:rw,nosuid,nodev,mode=1777,size=64G",
                         "BindReadOnlyPaths=/dev/shm/ae-expA-t1/inputs/release",
                         "BindPaths=/dev/shm/ae-expA-t1/serve/g7",
                         # The unit lives through the D11.3 deadlines (P 11-05, secondary 11-06) from 11-01.
                         "RuntimeMaxSec=518400"):
            self.assertIn(expected, properties)
        self.assertNotIn("ReadOnlyPaths=/dev/shm", properties)
        self.assertEqual([p for p in properties if p.startswith(("ReadWritePaths=", "BindPaths="))],
                         ["BindPaths=/dev/shm/ae-expA-t1/serve/g7"])
        self.assertLess(properties.index("TemporaryFileSystem=/dev/shm:rw,nosuid,nodev,mode=1777,size=64G"),
                        properties.index("BindPaths=/dev/shm/ae-expA-t1/serve/g7"))
        self.assertIn("--collect", argv[:argv.index("--")])        # a failed unit never blocks a restart
        cli = load_script("expA_serve")
        with mock.patch.object(serve, "current_identity", return_value=CORE1_OK), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(cli.main(["plan", "--run", "t1", "--instance", "g7", "--gpu-uuid", GPU7, "--port", "8201"]),
                             0)
        self.assertEqual(json.loads(out.getvalue())["spec"]["runtime_max_seconds"], 6 * 86400)
        setenv = [value.split("=", 1)[1] for value in argv[:argv.index("--")] if value.startswith("--setenv=")]
        self.assertIn(f"PATH={serve.SERVE_ENV}/bin:/usr/bin:/bin", setenv)
        self.assertIn("TMPDIR=/dev/shm/ae-expA-t1/serve/g7/tmp", setenv)
        self.assertIn("HOME=/dev/shm/ae-expA-t1/serve/g7/home", setenv)
        self.assertIn("LANG=C.UTF-8", setenv)
        self.assertIn("CUDA_DEVICE_ORDER=PCI_BUS_ID", setenv)
        tail = argv[argv.index("--") + 1:]
        self.assertEqual(tail[3], "/dev/shm/ae-expA-t1/inputs/release/scripts/expA_serve.py")

    def test_gpu5_is_refused_at_the_spec(self):
        with self.assertRaises(serve.ServePreflightError):
            spec(gpu_uuid=serve.GPU5_UUID)
        for bad in (dict(gpu_uuid="5"), dict(gpu_uuid=serve.GPU5_UUID.upper()), dict(port=80), dict(run="Bad_Run"),
                    dict(gpu_memory_utilization=1), dict(max_num_seqs=0), dict(memory_max="lots")):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                spec(**bad)

    def test_cli_plan_refuses_the_wrong_identity(self):
        cli = load_script("expA_serve")
        out = io.StringIO()
        with mock.patch.object(serve, "current_identity", return_value=dict(CORE1_OK, uid=1006, euid=1006,
                                                                            user="[redacted-other-user]")):
            with contextlib.redirect_stdout(out):
                code = cli.main(["plan", "--run", "t1", "--instance", "g7", "--gpu-uuid", GPU7, "--port", "8201"])
        self.assertEqual(code, 2)
        self.assertIn("uid_1006_forbidden_on_ai-core-team-1", json.loads(out.getvalue())["refused"])


# ---------------------------------------------------------------------------
# Serving: preflight refusals
# ---------------------------------------------------------------------------

class ServePreflightTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.model, self.manifest, self.manifest_sha = model_fixture(self.root)
        env = self.root / "env"
        for name in ("python", "ninja"):
            write(env / "bin" / name, "#!/bin/sh\n", 0o755)
        self.spec = spec(serve_env=str(env), model_dir=str(self.model), model_manifest=str(self.manifest),
                         model_manifest_sha256=self.manifest_sha)
        self.tmpfs = [("/", "ext4", "/dev/sda1"), ("/dev/shm", "tmpfs", "tmpfs")]

    def tearDown(self):
        self.temporary.cleanup()

    def preflight(self, value=None, *, observed=None, identity=CORE1_OK, mounts=None, port_free=True, proc=None):
        value = value or self.spec
        observe = observed if callable(observed) else (lambda uuid: dict(observed or observation(), uuid=uuid))
        return serve.preflight(value, observe=observe, port_free=lambda port: port_free,
                               mounts=mounts or self.tmpfs, magic=lambda path: None, identity=identity,
                               proc_options=proc or (lambda: {"options": ["rw,nosuid"], "hidepid": None}))

    def test_passes_and_records_identity_proc_and_model(self):
        result = self.preflight()
        self.assertTrue(result["passed"], result["failures"])
        self.assertEqual(result["identity"], CORE1_OK)
        self.assertEqual(result["proc_mount"]["options"], ["rw,nosuid"])
        self.assertEqual(result["model"]["sha256_verified_files"], len(serve.MODEL_REQUIRED_FILES) + 1)
        self.assertEqual(result["model"]["unexpected_files"], [])

    def test_identity_failure_refuses_without_reading_the_model(self):
        verify = mock.Mock()
        result = serve.preflight(self.spec, observe=lambda uuid: observation(uuid=uuid), port_free=lambda p: True,
                                 mounts=self.tmpfs, magic=lambda p: None, proc_options=lambda: {},
                                 identity=dict(CORE1_OK, uid=1006, euid=1006, user="[redacted-other-user]"), verify=verify)
        self.assertFalse(result["passed"])
        self.assertIn("uid_1006_forbidden_on_ai-core-team-1", result["failures"])
        verify.assert_not_called()

    def test_gpu5_by_index_or_minor_and_busy_gpus(self):
        cases = {"gpu_index_5_forbidden": observation(index=5), "gpu_minor_5_forbidden": observation(minor=5),
                 "gpu_memory_in_use_500MiB": observation(used=500), "gpu_has_processes": observation(pids=[4321])}
        for failure, observed in cases.items():
            with self.subTest(failure=failure):
                result = self.preflight(observed=observed)
                self.assertFalse(result["passed"])
                self.assertIn(failure, result["failures"])
                self.assertEqual(result["model"], {"skipped": "earlier preflight failures"})
        self.assertIn("gpu5_uuid_forbidden", serve.check_idle(dict(observation(), uuid=serve.GPU5_UUID), GPU7)["failures"])
        self.assertTrue(serve.check_idle(observation(used=128), GPU7)["passed"])

        def broken(uuid):
            raise subprocess.TimeoutExpired("nvidia-smi", 20)
        self.assertTrue(any(f.startswith("gpu_observation_failed") for f in self.preflight(observed=broken)["failures"]))

    def test_port_cache_and_serve_env(self):
        self.assertIn("port_in_use", self.preflight(port_free=False)["failures"])
        for mounts in ([("/", "ext4", "/dev/sda1")], [("/", "ext4", "x"), ("/dev/shm", "nfs4", "h:/x")]):
            with self.subTest(mounts=mounts):
                failures = self.preflight(mounts=mounts)["failures"]
                self.assertTrue(any(f.startswith("cache_location:") for f in failures), failures)
        os.unlink(Path(self.spec.serve_env) / "bin" / "ninja")
        self.assertIn("serve_env_lacks_executable:ninja", self.preflight()["failures"])

    def test_model_files_must_match_the_pinned_manifest(self):
        self.assertIn("model_manifest_sha256_mismatch",
                      self.preflight(dataclass_replace(self.spec, model_manifest_sha256="0" * 64))["failures"])
        write(self.model / "config.json", "tampered\n")
        self.assertIn("model_file_sha256_mismatch:config.json", self.preflight()["failures"])
        os.unlink(self.model / "config.json")
        self.assertTrue(any(f.startswith("model_file_missing:config.json") for f in self.preflight()["failures"]))
        os.symlink("tokenizer.json", self.model / "config.json")
        self.assertIn("model_file_not_regular:config.json", self.preflight()["failures"])
        write(self.model / "extra.bin", "x")
        self.assertIn("model_unexpected_file:extra.bin", self.preflight()["failures"])

    def test_mirror_ownership_is_recorded_and_can_be_enforced(self):
        # Review finding: on Core1 the public mirror (and /mnt/kldata) belong to UID 1006, another person's account,
        # who could change the interpreter, vLLM or the weights. Preflight records the owner chain; with
        # mirror_owner_uid set (0 once the coordinator made the chain root-owned) it refuses anything else.
        value = dataclass_replace(self.spec, runtime_root=str(self.root / "env"))
        result = self.preflight(value)
        self.assertTrue(result["passed"], result["failures"])
        ownership = result["mirror_ownership"]
        self.assertEqual((ownership["mirror_owner_uid"], ownership["checked"], ownership["passed"]),
                         (None, "roots_and_ancestors", True))
        paths = [entry["path"] for entry in ownership["chain"]]
        self.assertIn(str(self.root / "env"), paths)
        self.assertIn("/", paths)
        self.assertEqual(len(paths), len(set(paths)))
        stranger = os.getuid() + 4242                                   # nothing below the roots is theirs
        refused = self.preflight(dataclass_replace(value, mirror_owner_uid=stranger))
        self.assertFalse(refused["passed"])
        self.assertTrue(any(f.startswith(f"mirror_not_owned_by_uid_{stranger}_or_root:") for f in refused["failures"]))
        self.assertEqual(refused["model"], {"skipped": "earlier preflight failures"})
        self.assertEqual(refused["mirror_ownership"]["checked"], "full")

        def root_owned(path):
            return {"path": str(path), "uid": 0, "gid": 0, "mode": "755", "kind": "dir",
                    "group_or_other_writable": False}
        with mock.patch.object(serve, "_owner_entry", side_effect=root_owned):
            self.assertTrue(serve.mirror_ownership(dataclass_replace(value, mirror_owner_uid=0), full=True)["passed"])

        def foreign_parent(path):
            entry = root_owned(path)
            return dict(entry, uid=1006, mode="2775", group_or_other_writable=True) if str(path) == "/mnt" else entry
        value = dataclass_replace(value, serve_env="/mnt/kldata/x/envs/serve", runtime_root="/mnt/kldata/x/rt",
                                  model_dir="/mnt/kldata/x/model", model_manifest="/mnt/kldata/x/COPY.sha256",
                                  mirror_owner_uid=0)
        with mock.patch.object(serve, "_owner_entry", side_effect=foreign_parent):
            report = serve.mirror_ownership(value)
        self.assertEqual([entry["path"] for entry in report["failures"]], ["/mnt"])   # an ancestor can rename it
        with self.assertRaises(ValueError):
            spec(mirror_owner_uid=-1)

    def test_model_manifest_must_list_the_required_files(self):
        lines = [line for line in self.manifest.read_text().splitlines() if not line.endswith("  config.json")]
        write(self.manifest, "\n".join(lines) + "\n")
        record = serve.verify_model(self.model, self.manifest, expected_manifest_sha256=None)
        self.assertIn("model_manifest_lacks:config.json", record["failures"])
        write(self.manifest, "not a sums file\n")
        self.assertTrue(serve.verify_model(self.model, self.manifest, expected_manifest_sha256=None)["failures"][0]
                        .startswith("model_manifest_invalid"))

    def test_quick_mode_hashes_only_small_files(self):
        record = serve.verify_model(self.model, self.manifest, mode="quick", expected_manifest_sha256=self.manifest_sha,
                                    quick_hash_limit=1)
        self.assertEqual(record["failures"], [])
        self.assertTrue(all(entry.get("readable_only") for entry in record["files"].values()))

    def test_proc_mount_options_are_recorded(self):
        def run(argv, **kwargs):
            self.assertEqual(argv, ["findmnt", "-no", "OPTIONS", "/proc"])
            return subprocess.CompletedProcess(argv, 0, b"rw,nosuid,nodev,noexec,relatime,hidepid=invisible\n", b"")
        record = serve.proc_mount_options(run)
        self.assertEqual((record["hidepid"], record["watchdog"]), ("invisible", "fail_closed"))

        def fails(argv, **kwargs):
            raise FileNotFoundError("findmnt")
        self.assertIsNone(serve.proc_mount_options(fails)["options"])

    def test_gpu_table_flags_gpu5_and_busy_gpus(self):
        outputs = {"--query-gpu=index,uuid,memory.used": (f"5, {serve.GPU5_UUID}, 0\n6, GPU-15870ba7-ca88-14c1-d031-"
                                                          f"0ee12500df01, 900\n7, {GPU7}, 0\n").encode(),
                   "--query-compute-apps=gpu_uuid,pid": b"GPU-15870ba7-ca88-14c1-d031-0ee12500df01, 777\n"}

        def run(argv, **kwargs):
            return subprocess.CompletedProcess(argv, 0, outputs[argv[1]], b"")
        table = serve.gpu_table(run)
        rows = {row["index"]: row for row in table["gpus"]}
        self.assertTrue(rows[5]["gpu5"] and rows[5]["idle"] and not rows[5]["eligible"])
        self.assertEqual((rows[6]["pids"], rows[6]["eligible"]), ([777], False))
        self.assertEqual(table["eligible"], [GPU7])


def dataclass_replace(value, **changes):
    import dataclasses
    return dataclasses.replace(value, **changes)


# ---------------------------------------------------------------------------
# Watchdog: fail closed (F4)
# ---------------------------------------------------------------------------

class FakeChild:
    def __init__(self, pid):
        self.pid, self.returncode = pid, None

    def poll(self):
        return self.returncode


class WatchdogTests(unittest.TestCase):
    def supervisor(self, observations, groups):
        sequence = list(observations)

        def observe(uuid):
            item = sequence.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        def pgid_of(pid):
            if isinstance(groups.get(pid), Exception):
                raise groups[pid]
            if pid not in groups:
                raise FileNotFoundError(pid)
            return groups[pid]
        value = serve.Supervisor(spec(), observe=observe, sleep=lambda seconds: None, pgid_of=pgid_of,
                                 own_descendants=lambda pid: {pid + 1})
        value.child, value.child_pgid = FakeChild(500), 500
        return value

    def test_provably_ours_needs_positive_evidence(self):
        self.assertTrue(serve.provably_ours(10, 500, own_pids={10}))
        self.assertTrue(serve.provably_ours(11, 500, pgid_of=lambda pid: 500))
        for error in (FileNotFoundError(11), PermissionError(11), ValueError("garbled")):
            def pgid_of(pid, error=error):
                raise error
            with self.subTest(error=type(error).__name__):
                self.assertFalse(serve.provably_ours(11, 500, pgid_of=pgid_of))

    def test_own_processes_are_no_collision(self):
        value = self.supervisor([observation(pids=[500, 501, 777])], {777: 500})
        status = value.watch_once()
        self.assertFalse(status["collision"])

    def test_hidden_pid_is_foreign_after_one_reobservation(self):
        hidden = PermissionError("hidepid")
        value = self.supervisor([observation(pids=[500, 9000]), observation(pids=[500, 9000])], {9000: hidden})
        status = value.watch_once()
        self.assertEqual((status["collision"], status["reason"], status["foreign"]), (True, "foreign_pid", [9000]))

    def test_vanished_pid_is_never_assumed_ours_but_a_transient_is_no_collision(self):
        value = self.supervisor([observation(pids=[500, 9001]), observation(pids=[500, 9001])], {})
        self.assertEqual(value.watch_once()["foreign"], [9001])
        transient = self.supervisor([observation(pids=[500, 9002]), observation(pids=[500])], {})
        status = transient.watch_once()
        self.assertEqual((status["collision"], status["unproven_first"], status["foreign"]), (False, [9002], []))

    def test_foreign_process_group_and_gpu5_and_lost_observation(self):
        value = self.supervisor([observation(pids=[9003]), observation(pids=[9003])], {9003: 9003})
        self.assertEqual(value.watch_once()["reason"], "foreign_pid")
        gpu5 = self.supervisor([observation(index=5, pids=[500])], {})
        self.assertEqual(gpu5.watch_once()["reason"], "gpu5")
        lost = self.supervisor([OSError("nvidia-smi gone"), OSError("still gone")], {})
        self.assertEqual(lost.watch_once()["reason"], "watchdog_observation_failed")
        recovered = self.supervisor([OSError("blip"), observation(pids=[500])], {})
        status = recovered.watch_once()
        self.assertFalse(status["collision"])
        self.assertIn("blip", status["observation_retry"])

    @unittest.skipUnless(POSIX, "process groups required")
    def test_collision_stops_only_our_group_and_records_it(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(serve, "CACHE_PARENT", Path(directory)):
            value = spec(watchdog_seconds=1)
            outsider = subprocess.Popen(["sleep", "30"], start_new_session=True)
            try:
                supervisor = serve.Supervisor(
                    value, observe=lambda uuid: observation(pids=[supervisor.child.pid, outsider.pid]),
                    sleep=lambda seconds: None, popen=lambda argv, **kw: subprocess.Popen(["sleep", "30"], **kw),
                    http_get=lambda path: {"data": []}, own_descendants=lambda pid: set(),
                    preflight_check=lambda spec, observe: {"passed": True, "failures": []})
                self.assertEqual(supervisor.run(), 3)
                self.assertIsNotNone(supervisor.child.poll())
                self.assertIsNone(outsider.poll())
                records = sorted(p.name.split("-", 1)[1] for p in (value.cache_root / "records").iterdir())
                self.assertEqual(records, ["collision.json", "launched.json", "preflight.json", "stopped.json"])
                collision = json.loads(next((value.cache_root / "records").glob("*-collision.json")).read_text())
                self.assertEqual((collision["reason"], collision["foreign"]), ("foreign_pid", [outsider.pid]))
                logs = sorted(p.name for p in (value.cache_root / "logs").iterdir())
                self.assertTrue(logs[0].startswith("vllm-") and logs[0].endswith(".stderr"))
            finally:
                outsider.kill()
                outsider.wait()


# ---------------------------------------------------------------------------
# Stager
# ---------------------------------------------------------------------------

CHECKOUT_FILES = {"pkg/__init__.py": "", "pkg/core.py": "def f(x):\n    return x\n", "README.md": "toy\n",
                  "tests/test_core.py": "def test_f():\n    pass\n"}


def build_stage_sources(base, *, mutate_rows=None, mutate_manifest=None, extra_dispatch_files=None,
                        unreferenced_payloads=0):
    """Core2-like sources: a release, a dispatch view and a clean-checkout mirror; returns paths and pins."""
    base = Path(base)
    release = base / "release"
    for relative, text in {"absent_edges/__init__.py": "", "scripts/expA_stage.py": "# stager\n",
                           "scripts/expA_publish.py": "# publisher\n", "tests/test_x.py": "pass\n"}.items():
        write(release / relative, text)
    sums = "".join(f"{sha((release / p).read_bytes())}  {p}\n" for p in sorted(
        ["absent_edges/__init__.py", "scripts/expA_stage.py", "scripts/expA_publish.py", "tests/test_x.py"]))
    write(release / "RELEASE_FILES.sha256", sums)
    repo = base / "mirror" / "repos" / "fam"
    for relative, text in CHECKOUT_FILES.items():
        write(repo / relative, text)
    package = scan_package_files(repo, ["pkg"])["files"]
    source = source_tree_sha256(file_sha256s(repo, package))
    entry = tools.clean_checkout_entry("fam", repo, ["pkg"], commit="c" * 40)
    checkouts = write(base / "mirror" / "repos-manifests" / "clean-checkouts.json",
                      canonical_json(tools.checkouts_manifest([entry])) + b"\n")
    dispatch = base / "dispatch"
    payload = {"query": {"symbol": "pkg/core.py::f"}, "matches": [], "endpoints": [],
               "coverage": {"completeness": "unknown"}}
    encoded = canonical_json(payload)
    store = canonical_json({"sha256": sha(encoded), "bytes": len(encoded), "payload": payload}) + b"\n"
    for n in range(unreferenced_payloads):   # e.g. a vcomplete C2 cell that no dispatch row names
        extra = dict(payload, guidance=f"extra {n}")
        extra_encoded = canonical_json(extra)
        store += canonical_json({"sha256": sha(extra_encoded), "bytes": len(extra_encoded), "payload": extra}) + b"\n"
    write(dispatch / "payloads" / "fam.jsonl", store)
    facts = canonical_json({"backend": "locagent", "source_sha256": source, "nodes": [], "edges": [],
                            "coverage": {"completeness": "unknown"}})
    write(dispatch / "graph_facts" / "fam.json", facts)
    rows = [{"item_id": f"item{n}", "family": "fam",
             "target": {"identity": "pkg/core.py::f", "path": "pkg/core.py", "line": 1, "qualname": "f"},
             "package_dirs": ["pkg"], "wording_pair": "AB"[n % 2],
             "membership": {"in_s_nat": False, "in_s_edge": False, "in_t": n == 0},
             "cells": {"omit_node": {c: sha(encoded) for c in ("C1", "C2", "C3", "C4a")},
                       "vcomplete": {c: sha(encoded) for c in ("C1", "C3")}}} for n in range(2)]
    if mutate_rows:
        mutate_rows(rows)
    dispatch_bytes = b"".join(canonical_json(row) + b"\n" for row in rows)
    write(dispatch / "dispatch.jsonl", dispatch_bytes)
    manifest = {"schema": "expA-dispatch-v2", "items_sha256": "1" * 64, "manifest_sha256": "2" * 64,
                "dispatch_sha256": sha(dispatch_bytes), "rows": len(rows),
                "payload_store": {"fam": {"path": "payloads/fam.jsonl", "sha256": sha(store)}},
                "graph_facts": {"fam": {"path": "graph_facts/fam.json", "sha256": sha(facts)}},
                "families": {"fam": {"source_sha256": source, "package_dirs": ["pkg"], "role": "dev"}},
                "code": {"items_build": {}}}
    if mutate_manifest:
        mutate_manifest(manifest)
    manifest_path = write(dispatch / "dispatch_manifest.json", canonical_json(manifest) + b"\n")
    for relative, text in (extra_dispatch_files or {}).items():
        write(dispatch / relative, text)
    run_root = base / "ae-expA-test"
    run_root.mkdir(mode=0o700)
    os.chmod(run_root, 0o700)
    return {"run_root": str(run_root), "release": str(release), "release_sums": sha(sums.encode()),
            "dispatch": str(dispatch), "dispatch_manifest": sha(manifest_path.read_bytes()),
            "checkouts": str(checkouts), "checkouts_sha": sha(checkouts.read_bytes()),
            "checkouts_dir": str(base / "mirror" / "repos"), "source": source}


# Run in a child process with a low soft RLIMIT_NOFILE: a component or a run has far more directories than
# a systemd service's default 1024 open files, so cached directory descriptors must stay bounded.
FD_LIMIT_STAGE = r'''
import hashlib, importlib.util, json, os, resource, sys
script, base, limit, count = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
resource.setrlimit(resource.RLIMIT_NOFILE, (limit, resource.getrlimit(resource.RLIMIT_NOFILE)[1]))
spec = importlib.util.spec_from_file_location("stage_fd_limit", script)
stage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage)
files = {}
for n in range(count):
    relative = f"repos/fam/d{n:04d}/sub/f.txt"
    path = os.path.join(base, "source", relative)
    os.makedirs(os.path.dirname(path))
    data = f"{n}\n".encode()
    with open(path, "wb") as stream:
        stream.write(data)
    files[relative] = (path, hashlib.sha256(data).hexdigest())
parent = os.open(base, os.O_RDONLY | os.O_DIRECTORY)
tree = stage.StagedTree(parent, "staged", os.getuid(), os.getgid())
peak = 0
try:
    for relative in sorted(files):
        tree.copy_file(relative, *files[relative])
        peak = max(peak, len(tree.dir_fds))
finally:
    tree.close()
problems = stage.verify_staged(parent, "staged", {k: v[1] for k, v in files.items()}, {}, [], os.getuid(), os.getgid())
print(json.dumps({"files": len(files), "peak": peak, "max": stage.MAX_DIR_FDS, "problems": problems[:3]}))
'''

FD_LIMIT_PUBLISH = r'''
import importlib.util, io, json, os, resource, sys
script, run_root, target, limit = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
resource.setrlimit(resource.RLIMIT_NOFILE, (limit, resource.getrlimit(resource.RLIMIT_NOFILE)[1]))
spec = importlib.util.spec_from_file_location("publish_fd_limit", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
uid, gid = os.getuid(), os.getgid()
module.init_target(target, uid, gid)
publisher = module.Publisher(run_root, target, uid=uid, gid=gid, out=io.StringIO())
try:
    summary = publisher.publish_pass()
    cached = len(publisher.target.dir_fds)
finally:
    publisher.close()
print(json.dumps({"published": summary["published"], "conflicts": summary["conflicts"], "cached": cached,
                  "max": module.MAX_DIR_FDS, "completed_cells": len(publisher.completed_cells)}))
'''


def stage_args(sources, components="release,dispatch,checkouts", owner=None):
    owner = owner or pwd.getpwuid(os.getuid()).pw_name
    return STAGE.parse_args(["--run-root", sources["run_root"], "--owner", owner, "--components", components,
                             "--release-dir", sources["release"], "--expect-release-sums-sha256", sources["release_sums"],
                             "--dispatch-dir", sources["dispatch"],
                             "--expect-dispatch-manifest-sha256", sources["dispatch_manifest"],
                             "--checkouts-manifest", sources["checkouts"],
                             "--expect-checkouts-manifest-sha256", sources["checkouts_sha"],
                             "--checkouts-dir", sources["checkouts_dir"]])


def run_stage(sources, **kwargs):
    return STAGE.stage(stage_args(sources, **kwargs), fs_check=lambda path: {"fstype": "test"}, host="testhost",
                       run_parent=None)


@unittest.skipUnless(POSIX, "the stager uses dir_fd and os.fwalk")
class StagerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.addCleanup(self.cleanup)

    def cleanup(self):
        for current, dirnames, _ in os.walk(self.base):
            for name in dirnames:
                try:
                    os.chmod(Path(current) / name, 0o700)
                except OSError:
                    pass
        self.temporary.cleanup()

    def fresh(self, name="s", **kwargs):
        return build_stage_sources(self.base / name, **kwargs)

    def test_stages_exactly_the_pinned_inputs_with_owner_and_modes(self):
        sources = self.fresh()
        result = run_stage(sources)
        self.assertEqual({k: v["result"] for k, v in result["components"].items()},
                         {"release": "staged", "dispatch": "staged", "checkouts": "staged"})
        inputs = Path(sources["run_root"]) / "inputs"
        self.assertEqual(sorted(p.name for p in inputs.iterdir()),
                         ["STAGE_RECEIPT.checkouts.json", "STAGE_RECEIPT.dispatch.json", "STAGE_RECEIPT.release.json",
                          "checkouts", "dispatch", "release"])
        self.assertTrue((inputs / "checkouts" / "manifest.json").is_file())
        self.assertEqual(sorted(p.relative_to(inputs / "checkouts" / "repos" / "fam").as_posix()
                                for p in (inputs / "checkouts" / "repos" / "fam").rglob("*") if p.is_file()),
                         sorted(CHECKOUT_FILES))
        for path in [inputs] + list(inputs.rglob("*")):
            info = os.lstat(path)
            self.assertEqual(info.st_uid, os.getuid(), path)
            self.assertEqual(stat.S_IMODE(info.st_mode), 0o700 if stat.S_ISDIR(info.st_mode) else 0o444, path)
        self.assertFalse(any(p.name in ("items.jsonl", "manifest.json") and "dispatch" in p.parts
                             for p in inputs.rglob("*")))
        receipt = json.loads((inputs / "STAGE_RECEIPT.dispatch.json").read_text())
        self.assertEqual(receipt["summary"]["dispatch_sha256_pinned_by"], ["dispatch_sha256"])
        self.assertEqual(result["runner_arguments"]["--checkout-manifest"],
                         str(inputs / "checkouts" / "manifest.json"))
        again = run_stage(sources)
        self.assertEqual({v["result"] for v in again["components"].values()}, {"already_staged_verified"})
        target = inputs / "dispatch" / "dispatch.jsonl"
        os.chmod(target.parent, 0o700)
        os.chmod(target, 0o644)
        target.write_text("tampered\n")
        os.chmod(target, 0o444)
        with self.assertRaisesRegex(STAGE.StageRefused, "already staged with different content"):
            run_stage(sources, components="dispatch")

    def test_many_directories_fit_a_small_open_file_limit(self):
        # 200 x 2 directories, as in a checkouts component of every dispatch family (the production
        # checkouts have about 1,450 directories); the soft limit is 96 open files.
        done = subprocess.run([sys.executable, "-B", "-c", FD_LIMIT_STAGE, str(PROJECT / "scripts" / "expA_stage.py"),
                               str(self.base), "96", "200"], capture_output=True, text=True, timeout=300)
        self.assertEqual(done.returncode, 0, done.stderr[-2000:])
        report = json.loads(done.stdout)
        self.assertEqual((report["files"], report["problems"]), (200, []))
        self.assertLessEqual(report["peak"], report["max"])

    def test_the_stager_matches_the_runner_tree_format(self):
        files, links, dirs = {"a/b.py": "0" * 64}, {"a/l": "b.py"}, ["a"]
        self.assertEqual(STAGE.tree_digest(files, links, dirs), tools.tree_digest(files, links, dirs))
        self.assertEqual((STAGE.CHECKOUTS_SCHEMA, STAGE.TREE_PROTOCOL), (tools.CHECKOUTS_SCHEMA, tools.TREE_PROTOCOL))
        self.assertEqual((STAGE.SKIP_DIRS, STAGE.SKIP_SUFFIXES), (tools.STAGE_SKIP_DIRS, tools.STAGE_SKIP_SUFFIXES))

    def test_refuses_items_jsonl_and_truth_files(self):
        cases = {
            "items.jsonl in the dispatch dir": dict(extra_dispatch_files={"items.jsonl": "{}\n"}),
            "facts/ in the dispatch dir": dict(extra_dispatch_files={"facts/fam.json": "{}\n"}),
            "truth key V in a row": dict(mutate_rows=lambda rows: rows[0].update(V=["x"])),
            "omitted in a row": dict(mutate_rows=lambda rows: rows[0]["target"].update(omitted="pkg/a.py::g")),
            "covariates in a row": dict(mutate_rows=lambda rows: rows[1].update(covariates={})),
            "s_nat_excluded_reason in a row": dict(mutate_rows=lambda rows: rows[1]["membership"].update(
                s_nat_excluded_reason="no_natural_miss")),
            "a non-boolean membership flag": dict(mutate_rows=lambda rows: rows[0]["membership"].update(in_t=1)),
            "payload path is items.jsonl": dict(mutate_manifest=lambda m: m["payload_store"]["fam"].update(
                path="items.jsonl")),
            "truth key in the manifest": dict(mutate_manifest=lambda m: m.update(V={})),
        }
        for name, kwargs in cases.items():
            with self.subTest(case=name):
                sources = self.fresh(name.replace(" ", "_"), **kwargs)
                with self.assertRaises(STAGE.StageRefused):
                    run_stage(sources)
                self.assertFalse((Path(sources["run_root"]) / "inputs").exists())
        with self.assertRaises(STAGE.StageRefused):
            STAGE.refuse_truth_path("static/fam/locagent.edges.jsonl", "test")
        self.assertTrue(STAGE.is_truth_path("items/items.jsonl") and STAGE.is_truth_path("eligibility/fam.jsonl"))

    def test_checkout_repo_trees_are_exempt_from_the_truth_name_rule(self):
        # main-20261101a: genshi ships examples/tutorial/geddit/static/jquery.js; repos/ files are pinned by the
        # checkout manifest, so their directory names are not truth markers. Everything else is still checked.
        STAGE.refuse_truth_paths("checkouts", ["manifest.json", "repos/genshi/examples/tutorial/geddit/static/jquery.js",
                                               "repos/fam/tests/items/items.json"])
        for component, path in (("checkouts", "static/fam/locagent.edges.jsonl"), ("checkouts", "items.jsonl"),
                                ("dispatch", "repos/fam/static/x.js"), ("release", "traces/fam/edges.jsonl")):
            with self.subTest(component=component, path=path), self.assertRaises(STAGE.StageRefused):
                STAGE.refuse_truth_paths(component, ["ok.json", path])

    def test_refuses_inputs_that_differ_from_their_pins(self):
        sources = self.fresh("pins")
        for field, value in (("release_sums", "0" * 64), ("dispatch_manifest", "0" * 64), ("checkouts_sha", "0" * 64)):
            with self.subTest(field=field), self.assertRaises(STAGE.StageRefused):
                run_stage(dict(sources, **{field: value}))
        payload = Path(sources["dispatch"]) / "payloads" / "fam.jsonl"
        original = payload.read_bytes()
        payload.write_bytes(original.replace(b"unknown", b"partial"))
        with self.assertRaisesRegex(STAGE.StageRefused, "payload store"):
            run_stage(sources)
        payload.write_bytes(original)
        write(Path(sources["release"]) / "scripts" / "unlisted.py", "x\n")
        with self.assertRaisesRegex(STAGE.StageRefused, "unlisted files"):
            run_stage(sources)
        os.unlink(Path(sources["release"]) / "scripts" / "unlisted.py")
        write(Path(sources["checkouts_dir"]) / "fam" / "pkg" / "extra.py", "x\n")
        with self.assertRaisesRegex(STAGE.StageRefused, "not exactly its manifest entry"):
            run_stage(sources)
        self.assertFalse((Path(sources["run_root"]) / "inputs" / "checkouts").exists())

    def test_refuses_payloads_that_no_dispatch_cell_names(self):
        # Review finding (D10.7): the shipped store was a copy of the whole items store (218 of 460 dev rows were
        # never dispatched). The stager accepts only the dispatched cells.
        sources = self.fresh("padded", unreferenced_payloads=2)
        with self.assertRaisesRegex(STAGE.StageRefused, "2 payloads that no dispatch cell names"):
            run_stage(sources)
        self.assertFalse((Path(sources["run_root"]) / "inputs" / "dispatch").exists())

    def test_refuses_a_checkout_whose_package_tree_is_not_the_payload_source(self):
        sources = self.fresh("source", mutate_manifest=lambda m: m["families"]["fam"].update(source_sha256="3" * 64))
        with self.assertRaisesRegex(STAGE.StageRefused, "graph facts of fam were built from"):
            run_stage(sources)
        sources = self.fresh("source2", mutate_manifest=lambda m: m.update(graph_facts={}))
        sources_family = json.loads(Path(sources["dispatch"], "dispatch_manifest.json").read_text())
        self.assertEqual(sources_family["graph_facts"], {})
        with self.assertRaisesRegex(STAGE.StageRefused, "T-arm items but the manifest pins no natural graph facts"):
            run_stage(sources)

    def test_owner_and_run_root_guards(self):
        sources = self.fresh("owner")
        fake = pwd.struct_passwd(("[redacted-other-user]", "x", 1006, 1006, "", "/home/t", "/bin/bash"))
        with mock.patch.object(STAGE.pwd, "getpwnam", return_value=fake):
            with self.assertRaisesRegex(STAGE.StageRefused, "belongs to another person"):
                STAGE.resolve_owner("[redacted-other-user]", host="ai-core-team-1")
        renamed = pwd.struct_passwd(("thilak.arjun", "x", 1006, 1006, "", "/home/t", "/bin/bash"))
        with mock.patch.object(STAGE.pwd, "getpwnam", return_value=renamed):
            with self.assertRaises(STAGE.StageRefused):
                STAGE.resolve_owner("thilak.arjun", host="ai-core-team-1")
        root = pwd.struct_passwd(("root", "x", 0, 0, "", "/root", "/bin/bash"))
        with mock.patch.object(STAGE.pwd, "getpwnam", return_value=root):
            with self.assertRaisesRegex(STAGE.StageRefused, "never owned by root"):
                STAGE.resolve_owner("root", host="testhost")
        os.chmod(sources["run_root"], 0o755)
        with self.assertRaisesRegex(STAGE.StageRefused, "mode 0700"):
            run_stage(sources)
        with self.assertRaisesRegex(STAGE.StageRefused, "directly under /dev/shm"):
            STAGE.check_run_root(sources["run_root"], os.getuid(), fs_check=lambda p: {})
        with self.assertRaisesRegex(STAGE.StageRefused, "named ae-expA"):
            STAGE.check_run_root(str(self.base), os.getuid(), fs_check=lambda p: {}, run_parent=None)


# ---------------------------------------------------------------------------
# Publisher
# ---------------------------------------------------------------------------

CELL_A = "P|item0|omit_node|C1|s11"
CELL_B = "P|item0|none|C0|s11"


def build_run_root(base):
    """A Core1-like run root with a finished attempt, an in-flight attempt, records, receipts and logs."""
    root = Path(base) / "ae-expA-pubtest"
    root.mkdir(mode=0o700)
    run = root / "run"
    write(run / "run.json", "{\"run_id\":\"pubtest\"}\n", 0o444)
    write(run / "cells.jsonl", "".join(json.dumps({"cell_id": c}) + "\n" for c in (CELL_A, CELL_B)), 0o444)
    write(run / ".run.json.123.tmp", "partial", 0o600)
    write(run / "launches" / "l1.json", "{}\n", 0o444)
    events = b"{\"kind\":\"episode_setup\"}\n"
    attempt = run / "cells" / CELL_A / f"{CELL_A}.a001"
    write(attempt / "attempt_start.json", "{}\n", 0o444)
    write(attempt / "events.jsonl", events, 0o444)
    write(attempt / "result.json", json.dumps({"cell_id": CELL_A, "attempt_id": f"{CELL_A}.a001",
                                               "exit_status": "Submitted", "events": {"sha256": sha(events)}}), 0o444)
    flight = run / "cells" / CELL_B / f"{CELL_B}.a001"
    write(flight / "attempt_start.json", "{}\n", 0o444)
    write(flight / "events.jsonl", "{\"kind\":\"episode_setup\"}\n", 0o644)
    write(root / "serve" / "g7" / "records" / "x-ready.json", "{}\n", 0o444)
    write(root / "serve" / "g7" / "logs" / "vllm-x.stderr", "log\n", 0o600)
    write(root / "inputs" / "STAGE_RECEIPT.release.json", "{}\n", 0o444)
    write(root / "logs" / "runner-x.out", "out\n", 0o600)
    return root


@unittest.skipUnless(POSIX, "the publisher uses dir_fd")
class PublisherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.root = build_run_root(self.base)
        self.parent = self.base / "runs"
        self.parent.mkdir(mode=0o700)
        self.uid, self.gid = os.getuid(), os.getgid()
        self.target = str(self.parent / "pubtest")
        PUBLISH.init_target(self.target, self.uid, self.gid)
        self.publisher = PUBLISH.Publisher(str(self.root), self.target, uid=self.uid, gid=self.gid, out=io.StringIO())
        self.addCleanup(self.publisher.close)
        self.addCleanup(self.temporary.cleanup)

    def tree(self):
        found = {}
        for path in sorted(Path(self.target).rglob("*")):
            info = os.lstat(path)
            found[path.relative_to(self.target).as_posix()] = (stat.S_IMODE(info.st_mode), info.st_uid, info.st_gid)
        return found

    def test_publishes_completed_immutable_records_with_owner_and_modes(self):
        summary = self.publisher.publish_pass()
        tree = self.tree()
        published = {path for path, (mode, _, _) in tree.items() if mode == 0o444 and not path.startswith("publish/")}
        self.assertEqual(published, {"run.json", "cells.jsonl", "launches/l1.json",
                                     f"cells/{CELL_A}/{CELL_A}.a001/attempt_start.json",
                                     f"cells/{CELL_A}/{CELL_A}.a001/events.jsonl",
                                     f"cells/{CELL_A}/{CELL_A}.a001/result.json",
                                     f"cells/{CELL_B}/{CELL_B}.a001/attempt_start.json",
                                     "serve/g7/records/x-ready.json",
                                     f"stage/{sha(b'{}' + bytes([10]))[:16]}/STAGE_RECEIPT.release.json"})
        self.assertEqual(summary["waiting"], {"not_immutable": 1})
        for path, (mode, uid, gid) in tree.items():
            self.assertEqual((uid, gid), (self.uid, self.gid), path)
            self.assertIn(mode, (0o444, 0o700), path)
            if os.path.isdir(os.path.join(self.target, path)):
                self.assertEqual(mode, 0o700, path)
        for path in published:
            source = (self.root / "run" / path) if not path.startswith(("serve/", "stage/")) else None
            if source is not None:
                self.assertEqual(sha(source.read_bytes()), sha(Path(self.target, path).read_bytes()))
        self.assertEqual(len(list(Path(self.target, "publish", "passes").iterdir())), 1)
        self.assertEqual(self.publisher.completed_cells, {CELL_A})
        self.assertFalse(self.publisher.run_complete())
        second = self.publisher.publish_pass()
        self.assertEqual(second["published"], 0)
        self.assertEqual(len(list(Path(self.target, "publish", "passes").iterdir())), 1)

    def test_events_wait_for_close_and_results_wait_for_events(self):
        flight = self.root / "run" / "cells" / CELL_B / f"{CELL_B}.a001"
        write(flight / "result.json", json.dumps({"exit_status": "NoAnswer", "events": {}}), 0o444)
        summary = self.publisher.publish_pass()
        self.assertEqual(summary["waiting"], {"not_immutable": 1, "events_not_closed": 1})
        os.chmod(flight / "events.jsonl", 0o444)
        self.publisher.publish_pass()
        self.assertTrue(Path(self.target, "cells", CELL_B, f"{CELL_B}.a001", "result.json").is_file())
        self.assertTrue(self.publisher.run_complete())

    def test_a_closed_attempt_publishes_its_partial_events(self):
        flight = self.root / "run" / "cells" / CELL_B / f"{CELL_B}.a001"
        write(flight / "attempt_closed.json", "{\"status\":\"abandoned\"}\n", 0o444)
        self.publisher.publish_pass()
        self.assertTrue(Path(self.target, "cells", CELL_B, f"{CELL_B}.a001", "events.jsonl").is_file())

    def test_never_overwrites_a_differing_file_and_records_the_conflict(self):
        directory = Path(self.target, "launches")
        directory.mkdir(mode=0o700)
        write(directory / "l1.json", "different\n", 0o444)
        summary = self.publisher.publish_pass()
        self.assertEqual(summary["conflicts"], 1)
        self.assertEqual((directory / "l1.json").read_text(), "different\n")
        conflicts = list(Path(self.target, "publish", "conflicts").iterdir())
        record = json.loads(conflicts[0].read_text())
        self.assertEqual((record["path"], record["reason"], record["action"]),
                         ("launches/l1.json", "target_differs_never_overwritten", "target_kept_unchanged"))
        self.assertEqual(self.publisher.publish_pass()["conflicts"], 0)  # recorded once
        identical = Path(self.target, "run.json")
        self.assertEqual(self.publisher.target.publish(str(self.root / "run" / "run.json"), "run.json")["status"],
                         "present")
        self.assertEqual(identical.read_text(), "{\"run_id\":\"pubtest\"}\n")

    def test_hundreds_of_cells_publish_within_a_small_open_file_limit(self):
        # One directory per cell and per attempt (402 here, thousands in a main run); soft limit 96.
        root = self.base / "ae-expA-many"
        root.mkdir(mode=0o700)
        run = root / "run"
        cells = [f"P|item{n:04d}|omit_node|C1|s11" for n in range(200)]
        write(run / "run.json", "{}\n", 0o444)
        write(run / "cells.jsonl", "".join(json.dumps({"cell_id": c}) + "\n" for c in cells), 0o444)
        events = b"{\"kind\":\"episode_setup\"}\n"
        for cell in cells:
            attempt = run / "cells" / cell / f"{cell}.a001"
            write(attempt / "attempt_start.json", "{}\n", 0o444)
            write(attempt / "events.jsonl", events, 0o444)
            write(attempt / "result.json", json.dumps({"cell_id": cell, "attempt_id": f"{cell}.a001",
                                                       "exit_status": "Submitted",
                                                       "events": {"sha256": sha(events)}}), 0o444)
        target = str(self.parent / "many")
        done = subprocess.run([sys.executable, "-B", "-c", FD_LIMIT_PUBLISH, str(PROJECT / "scripts" / "expA_publish.py"),
                               str(root), target, "96"], capture_output=True, text=True, timeout=300)
        self.assertEqual(done.returncode, 0, done.stderr[-2000:])
        report = json.loads(done.stdout)
        self.assertEqual((report["published"], report["conflicts"], report["completed_cells"]), (2 + 3 * 200, 0, 200))
        self.assertLessEqual(report["cached"], report["max"])
        self.assertEqual(len(list(Path(target, "cells").iterdir())), 200)

    def test_sha256_is_verified_after_the_copy(self):
        source = write(self.root / "extra.json", "{\"a\":1}\n", 0o444)

        def corrupt(directory_fd, name):  # bytes change between the copy and the link
            os.chmod(name, 0o600, dir_fd=directory_fd)
            descriptor = os.open(name, os.O_WRONLY, dir_fd=directory_fd)
            try:
                os.write(descriptor, b"X")
            finally:
                os.close(descriptor)
            os.chmod(name, 0o444, dir_fd=directory_fd)
        outcome = self.publisher.target.publish(str(source), "check/extra.json", after_copy=corrupt)
        self.assertEqual(outcome["status"], "verify_failed")
        self.assertIn("sha256 after copy", outcome["problems"][0])

    def test_the_loop_stops_on_the_stop_file_and_only_the_final_pass_publishes_logs(self):
        stop = self.root / "control" / "publish.stop"
        write(stop, "")
        reason, last = PUBLISH.run_loop(self.publisher, interval=0, stop_file=str(stop), sleep=lambda s: None)
        self.assertEqual((reason, last["final"]), ("stop_file", False))
        self.assertFalse(Path(self.target, "logs").exists())
        self.assertFalse(Path(self.target, "serve", "g7", "logs").exists())
        final = self.publisher.publish_pass(final=True)                      # the -final unit's `once --final`
        self.assertTrue(final["final"])
        self.assertTrue(Path(self.target, "logs", "runner-x.out").is_file())
        self.assertTrue(Path(self.target, "serve", "g7", "logs", "vllm-x.stderr").is_file())

    def test_the_loop_ends_when_every_cell_completed_without_publishing_growing_logs(self):
        # Review finding (pub_final_probe): on run_complete the loop published the vLLM log while the units still
        # served; stop-serve then appended shutdown lines and the -final unit recorded a conflict (exit 4).
        flight = self.root / "run" / "cells" / CELL_B / f"{CELL_B}.a001"
        os.chmod(flight / "events.jsonl", 0o444)
        write(flight / "result.json", json.dumps({"exit_status": "ContextLimit", "events": {}}), 0o444)
        reason, last = PUBLISH.run_loop(self.publisher, interval=0, stop_file=None, sleep=lambda s: None, max_passes=5)
        self.assertEqual((reason, last["final"]), ("run_complete", False))
        self.assertFalse(Path(self.target, "serve", "g7", "logs").exists())
        log = self.root / "serve" / "g7" / "logs" / "vllm-x.stderr"
        with open(log, "a") as stream:                                       # stop-serve: vLLM's shutdown lines
            stream.write("[shutdown] INFO Shutting down\n")
        final = PUBLISH.Publisher(str(self.root), self.target, uid=self.uid, gid=self.gid, out=io.StringIO())
        self.addCleanup(final.close)
        summary = final.publish_pass(final=True)
        self.assertEqual((summary["conflicts"], PUBLISH.exit_status(final, final=True)[0]), (0, 0))
        self.assertEqual(Path(self.target, "serve", "g7", "logs", "vllm-x.stderr").read_bytes(), log.read_bytes())

    def test_a_final_pass_with_waiting_records_is_not_clean(self):
        # Review finding: an attempt whose runner was SIGKILLed (events.jsonl never closed, no attempt_closed.json)
        # was never published, yet `once --final` exited 0, final-publish.ok was written and cleanup deleted it.
        summary = self.publisher.publish_pass(final=True)
        self.assertEqual(summary["waiting"], {"not_immutable": 1})
        code, waiting = PUBLISH.exit_status(self.publisher, final=True)
        self.assertEqual((code, waiting), (5, [{"path": f"cells/{CELL_B}/{CELL_B}.a001/events.jsonl",
                                                "reason": "not_immutable"}]))
        self.assertEqual(PUBLISH.exit_status(self.publisher, final=False), (0, []))   # the loop is not final
        flight = self.root / "run" / "cells" / CELL_B / f"{CELL_B}.a001"
        write(flight / "attempt_closed.json", "{\"status\":\"abandoned\"}\n", 0o444)  # `expA_run.py reconcile`
        self.publisher.publish_pass(final=True)
        self.assertEqual(PUBLISH.exit_status(self.publisher, final=True), (0, []))

    def test_records_of_another_launch_are_never_mixed_into_an_attempt(self):
        # Review finding (scenario_reboot): after a relaunch on a wiped run root, post-reboot events and results
        # were published next to the pre-reboot attempt_start.json of the same attempt id.
        attempt = Path(self.target, "cells", CELL_A, f"{CELL_A}.a001")
        attempt.mkdir(parents=True, mode=0o700)
        write(attempt / "attempt_start.json", "{\"launch\":\"before-reboot\"}\n", 0o444)
        summary = self.publisher.publish_pass()
        self.assertEqual(summary["conflicts"], 3)          # the start record, then its events and result
        reasons = {json.loads(p.read_text())["path"].rsplit("/", 1)[-1]: json.loads(p.read_text())["reason"]
                   for p in Path(self.target, "publish", "conflicts").iterdir()}
        self.assertEqual(reasons, {"attempt_start.json": "target_differs_never_overwritten",
                                   "events.jsonl": "attempt_start_differs_on_target",
                                   "result.json": "attempt_start_differs_on_target"})
        self.assertEqual(sorted(p.name for p in attempt.iterdir()), ["attempt_start.json"])
        self.assertNotIn(CELL_A, self.publisher.completed_cells)

    def test_init_reports_records_and_refuses_a_wiped_run_root(self):
        report = PUBLISH.init_target(self.target, self.uid, self.gid, run_root=str(self.root))
        self.assertEqual(report["target_records"], {"run_json": False, "cells": 0, "attempts": 0, "files": 0})
        self.publisher.publish_pass()
        report = PUBLISH.init_target(self.target, self.uid, self.gid, run_root=str(self.root))
        self.assertEqual(report["target_records"], {"run_json": True, "cells": 2, "attempts": 2, "files": 4})
        wiped = self.base / "ae-expA-wiped"                              # a reboot emptied /dev/shm
        (wiped / "run").mkdir(parents=True)
        with self.assertRaisesRegex(PUBLISH.PublishRefused, "restore them first"):
            PUBLISH.init_target(self.target, self.uid, self.gid, run_root=str(wiped))
        write(wiped / "run" / "run.json", "{\"run_id\":\"other\"}\n", 0o444)
        with self.assertRaisesRegex(PUBLISH.PublishRefused, "another run's records"):
            PUBLISH.init_target(self.target, self.uid, self.gid, run_root=str(wiped))

    def test_a_file_error_is_recorded_and_retried_not_fatal(self):
        if os.geteuid() == 0:
            self.skipTest("root reads unreadable files")
        launch = self.root / "run" / "launches" / "l1.json"
        os.chmod(launch, 0o000)
        try:
            summary = self.publisher.publish_pass()
        finally:
            os.chmod(launch, 0o444)
        self.assertEqual((summary["errors"], summary["waiting"].get("error:EACCES")), (1, 1))
        record = json.loads(sorted(Path(self.target, "publish", "passes").iterdir())[-1].read_text())
        self.assertEqual(record["errors"][0]["path"], "launches/l1.json")
        self.assertTrue(Path(self.target, "run.json").is_file())          # the rest of the pass went on
        self.assertEqual(self.publisher.publish_pass()["published"], 1)    # retried in the next pass
        self.assertTrue(Path(self.target, "launches", "l1.json").is_file())

    def test_source_and_target_guards(self):
        with self.assertRaisesRegex(PUBLISH.PublishRefused, "named ae-expA"):
            PUBLISH.check_source(str(self.base), run_parent=None)
        with self.assertRaisesRegex(PUBLISH.PublishRefused, "directly under /dev/shm"):
            PUBLISH.check_source(str(self.root))
        with self.assertRaisesRegex(PUBLISH.PublishRefused, "not on tmpfs"):
            PUBLISH.check_source(str(self.root), run_parent=None, owner_uid=self.uid, fs_magic=lambda p: 0x6969)
        with self.assertRaisesRegex(PUBLISH.PublishRefused, "not the run owner"):
            PUBLISH.check_source(str(self.root), run_parent=None, owner_uid=self.uid + 1, fs_magic=lambda p: None)
        with self.assertRaisesRegex(PUBLISH.PublishRefused, "is not one of"):
            PUBLISH.target_path("/mnt/kldata/absent-edges-20261018-core2/expA", "x")
        with self.assertRaisesRegex(PUBLISH.PublishRefused, "run id"):
            PUBLISH.target_path(PUBLISH.ALLOWED_TARGET_PARENTS[0], "../escape")
        os.chmod(self.target, 0o755)
        with self.assertRaisesRegex(PUBLISH.PublishRefused, "mode 700"):
            self.publisher.publish_pass()

    def test_the_cli_refuses_to_run_without_root(self):
        if os.geteuid() == 0:
            self.skipTest("running as root")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = PUBLISH.main(["once", "--run-root", str(self.root), "--run-id", "x"])
        self.assertEqual(code, 2)
        self.assertIn("runs as root", out.getvalue())

    def test_units_are_hardened_and_load_the_pinned_script(self):
        common = dict(unit="ae-expA-publish-t", run_root="/dev/shm/ae-expA-t", run_id="t1",
                      target_parent=PUBLISH.ALLOWED_TARGET_PARENTS[0], script="/dev/shm/ae-expA-t/p.py",
                      script_sha256="a" * 64)
        target = PUBLISH.ALLOWED_TARGET_PARENTS[0] + "/t1"
        for mode, writable, waits in (("loop", target, False), ("final", target, True),
                                      ("init", PUBLISH.ALLOWED_TARGET_PARENTS[0], True)):
            with self.subTest(mode=mode):
                argv = PUBLISH.systemd_run_argv(mode=mode, **common)
                properties = [a.split("=", 1)[1] for a in argv if a.startswith("--property=")]
                # ProtectSystem=strict leaves /dev (so /dev/shm) writable: every /dev/shm entry is read-only.
                for expected in ("NoNewPrivileges=yes", "ProtectSystem=strict", "ProtectHome=yes", "User=root",
                                 "PrivateNetwork=yes", "ReadOnlyPaths=/dev/shm /dev/shm/ae-expA-t",
                                 "ReadWritePaths=" + writable,
                                 "CapabilityBoundingSet=CAP_CHOWN CAP_DAC_OVERRIDE CAP_DAC_READ_SEARCH CAP_FOWNER"):
                    self.assertIn(expected, properties)
                self.assertEqual(len([p for p in properties if p.startswith("ReadWritePaths=")]), 1)
                self.assertEqual("--wait" in argv, waits)
                self.assertIn("--collect", argv)                # a failed unit never blocks its name
                restart = [p for p in properties if p.startswith("Restart")]
                self.assertEqual(restart, ["Restart=on-failure", "RestartSec=30", "RestartPreventExitStatus=2 4 5"]
                                 if mode == "loop" else ["Restart=no"])
                tail = argv[argv.index("--") + 1:]
                self.assertEqual(tail[:7], ["/usr/bin/python3", "-I", "-B", "-c", PUBLISH.LOADER,
                                            "/dev/shm/ae-expA-t/p.py", "a" * 64])
                self.assertNotIn("$", PUBLISH.LOADER + "".join(tail))
                self.assertNotIn("%", PUBLISH.LOADER)
        with self.assertRaises(PUBLISH.PublishRefused):
            PUBLISH.systemd_run_argv(mode="loop", **dict(common, script_sha256="bad"))

    def test_the_loader_executes_only_the_pinned_bytes(self):
        script = write(self.base / "probe.py", "import sys\nprint(sys.argv[1:], __loader_sha256__)\n")
        digest = sha(script.read_bytes())
        ok = subprocess.run([sys.executable, "-I", "-c", PUBLISH.LOADER, str(script), digest, "a", "b"],
                            capture_output=True, text=True, timeout=60)
        self.assertEqual(ok.returncode, 0, ok.stderr)
        self.assertEqual(ok.stdout.strip(), f"['a', 'b'] {digest}")
        bad = subprocess.run([sys.executable, "-I", "-c", PUBLISH.LOADER, str(script), "0" * 64, "a"],
                             capture_output=True, text=True, timeout=60)
        self.assertNotEqual(bad.returncode, 0)
        self.assertIn("is not the pinned", bad.stderr)


# ---------------------------------------------------------------------------
# Reboot: restore the published run, reconcile, publish again (review finding on Core1 reboots)
# ---------------------------------------------------------------------------

@unittest.skipUnless(POSIX, "the stager and the publisher use dir_fd")
class RebootRestoreTests(unittest.TestCase):
    def setUp(self):
        from absent_edges.probe import ledger
        self.L = ledger
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.addCleanup(self.cleanup)
        self.uid, self.gid = os.getuid(), os.getgid()
        self.owner = pwd.getpwuid(self.uid).pw_name
        self.runs = self.base / "runs"
        self.runs.mkdir(mode=0o700)
        self.target = self.runs / "rb1"
        self.root = self.base / "ae-expA-rb1"

    def cleanup(self):
        for current, dirnames, files in os.walk(self.base):
            for name in dirnames + files:
                try:
                    os.chmod(Path(current) / name, 0o700 if (Path(current) / name).is_dir() else 0o600)
                except OSError:
                    pass
        self.temporary.cleanup()

    def cells(self):
        return [self.L.make_cell(block="P", item_id=item, family="fam", target="pkg/core.py::f", variant="omit_node",
                                 condition="C1", seed=11, wording_pair="A", payload_sha256="a" * 64)
                for item in ("item0", "item1", "item2")]

    def stage(self, restore=False):
        args = STAGE.parse_args(["--run-root", str(self.root), "--owner", self.owner, "--components", "run-state",
                                 "--run-target", str(self.target)] + (["--restore"] if restore else []))
        return STAGE.stage(args, fs_check=lambda path: {"fstype": "test"}, host="testhost", run_parent=None,
                           run_target_parents=(str(self.runs),))

    def test_a_reboot_restores_published_records_and_never_redispatches(self):
        os.makedirs(self.root / "run", mode=0o700)
        os.chmod(self.root, 0o700)
        store = self.L.DurableStore(self.root / "run")
        cells = self.cells()
        store.initialize({"run_id": "rb1", "infra_attempt_cap": 3}, cells)
        before_reboot = {"host": __import__("socket").gethostname(), "boot_id": "boot-before-the-reboot", "pid": 1,
                         "start_ticks": 1}
        done, flight = cells[0]["cell_id"], cells[1]["cell_id"]
        directory, _ = store.start_attempt(done, f"{done}.a001", {"ordinal": 1, "token": 1, "cell_id": done,
                                                                  "owner_identity": before_reboot})
        events = b"{\"kind\":\"episode_setup\"}\n"
        write(directory / "events.jsonl", events, 0o444)
        self.L.write_immutable_json(directory / "result.json", {
            "schema": "expA-episode-result-v2", "cell_id": done, "attempt_id": f"{done}.a001",
            "exit_status": "Submitted", "events": {"sha256": sha(events)}})
        directory, _ = store.start_attempt(flight, f"{flight}.a001", {"ordinal": 1, "token": 1, "cell_id": flight,
                                                                      "owner_identity": before_reboot})
        write(directory / "events.jsonl", events, 0o644)                    # in flight at the reboot
        self.assertEqual(self.stage()["components"]["run-state"]["result"], "empty")
        PUBLISH.init_target(str(self.target), self.uid, self.gid, run_root=str(self.root))
        publisher = PUBLISH.Publisher(str(self.root), str(self.target), uid=self.uid, gid=self.gid, out=io.StringIO())
        publisher.publish_pass()
        publisher.close()
        published = {p.relative_to(self.target).as_posix(): sha(p.read_bytes()) for p in self.target.rglob("*")
                     if p.is_file() and not p.relative_to(self.target).as_posix().startswith(("publish/", "stage/"))}
        self.assertEqual(self.stage()["components"]["run-state"]["result"], "consistent")
        # The reboot: /dev/shm is empty; prepare recreates the run root and an empty run/.
        self.cleanup_run_root()
        os.makedirs(self.root / "run", mode=0o700)
        os.chmod(self.root, 0o700)
        with self.assertRaisesRegex(STAGE.StageRefused, "restore the records first"):
            self.stage()
        self.assertFalse((self.root / "inputs").exists())
        with self.assertRaisesRegex(PUBLISH.PublishRefused, "restore them first"):
            PUBLISH.init_target(str(self.target), self.uid, self.gid, run_root=str(self.root))
        restored = self.stage(restore=True)["components"]["run-state"]
        self.assertEqual(restored["result"], "restored")
        self.assertEqual(restored["summary"]["interrupted_attempts"], [f"cells/{flight}/{flight}.a001"])
        receipt = next((self.root / "inputs").glob("STAGE_RECEIPT.run-state.*.json"))
        self.assertEqual(json.loads(receipt.read_text())["summary"]["restored_files"], len(published))
        local = {p.relative_to(self.root / "run").as_posix(): sha(p.read_bytes())
                 for p in (self.root / "run").rglob("*") if p.is_file()}
        self.assertEqual(local, published)
        for path in (self.root / "run").rglob("*"):
            self.assertEqual(stat.S_IMODE(os.lstat(path).st_mode), 0o700 if path.is_dir() else 0o444, path)
        # The runner's reconcile on the restored directory: no completed cell is dispatched again, the attempt
        # interrupted by the reboot is closed as abandoned and the cell's next attempt is a002.
        ledger = self.L.CellLedger(self.base / "local" / "ledger.sqlite3", run_id="rb1")
        report = ledger.reconcile(self.L.DurableStore(self.root / "run"))
        self.assertEqual((report["completed_from_durable"], report["closed_gone"]), (1, 1))
        claimed = [ledger.claim("owner", {"host": "h"}, lease_seconds=60) for _ in range(3)]
        self.assertEqual(sorted(c["attempt_id"] for c in claimed if c),
                         sorted([f"{flight}.a002", f"{cells[2]['cell_id']}.a001"]))
        self.assertTrue((self.root / "run" / "cells" / flight / f"{flight}.a001" / "attempt_closed.json").is_file())
        publisher = PUBLISH.Publisher(str(self.root), str(self.target), uid=self.uid, gid=self.gid, out=io.StringIO())
        summary = publisher.publish_pass()
        publisher.close()
        self.assertEqual(summary["conflicts"], 0)       # the re-staged receipts are content-addressed, too
        self.assertTrue((self.target / "cells" / flight / f"{flight}.a001" / "attempt_closed.json").is_file())
        # The "empty" receipt (published before the reboot) and the "restored" one; the "consistent" receipt was
        # written after that pass and lost with /dev/shm, like any unpublished record.
        self.assertEqual(len(list((self.target / "stage").iterdir())), 2)
        self.assertEqual(self.stage()["components"]["run-state"]["result"], "consistent")

    def cleanup_run_root(self):
        for current, dirnames, files in os.walk(self.root):
            for name in dirnames:
                os.chmod(Path(current) / name, 0o700)
        shutil.rmtree(self.root)

    def test_the_run_state_check_refuses_a_diverging_run_directory(self):
        write(self.target / "run.json", "{\"run_id\":\"rb1\"}\n", 0o444)
        write(self.target / "cells" / "P|item0|omit_node|C1|s11" / "P|item0|omit_node|C1|s11.a001"
              / "attempt_start.json", "{}\n", 0o444)
        os.makedirs(self.root / "run", mode=0o700)
        os.chmod(self.root, 0o700)
        write(self.root / "run" / "run.json", "{\"run_id\":\"rb1\"}\n", 0o444)
        with self.assertRaisesRegex(STAGE.StageRefused, "does not hold every record"):
            self.stage()
        with self.assertRaisesRegex(STAGE.StageRefused, "applies only to an empty one"):
            self.stage(restore=True)
        with self.assertRaisesRegex(STAGE.StageRefused, "--run-target must be"):
            STAGE.run_target_path(str(self.base / "elsewhere" / "rb1"), (str(self.runs),))


# ---------------------------------------------------------------------------
# Clean-checkout mirror (F11)
# ---------------------------------------------------------------------------

def git(repo, *arguments):
    environment = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@example.invalid",
                       GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@example.invalid", GIT_CONFIG_NOSYSTEM="1")
    return subprocess.run(["git", "-C", str(repo), *arguments], capture_output=True, text=True, check=True,
                          env=environment, timeout=120).stdout.strip()


@unittest.skipUnless(POSIX and shutil.which("git"), "git and POSIX required")
class MirrorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.addCleanup(self.cleanup)
        self.mirror = load_script("expA_mirror_repos")
        repo = self.base / "analysis" / "toy"
        files = {"pkg/__init__.py": "VERSION = '$Format:%H$'\n", "pkg/core.py": "def f():\n    return 1\n",
                 "tests/test_core.py": "def test_f():\n    pass\n", "README": "toy\n",
                 ".gitignore": "pkg/_version.py\ntmp/\n", ".gitattributes": "pkg/__init__.py export-subst\nREADME export-ignore\n",
                 "docs/link_inside": None, "pkg/escape": None}
        for relative, text in files.items():
            if text is not None:
                write(repo / relative, text)
        (repo / "docs").mkdir()
        os.symlink("../README", repo / "docs" / "link_inside")
        os.symlink("../../outside", repo / "pkg" / "escape")
        git(repo.parent, "init", "-q", "toy")
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", "toy")
        self.commit = git(repo, "rev-parse", "HEAD")
        write(repo / "pkg" / "_version.py", "__version__ = '1.0'\n")      # generated, ignored, in scope
        write(repo / "tmp" / "leftover.txt", "x\n")                        # ignored leftover
        write(repo / "nohup.out", "x\n")                                   # untracked leftover
        write(repo / "pkg" / "__pycache__" / "core.cpython-311.pyc", "x")  # cache
        self.repo = repo
        self.spec = write(self.base / "spec.json", json.dumps({"families": [
            {"family": "toy", "repo_root": str(repo), "package_dirs": ["pkg"], "commit": self.commit}]}))
        self.out = self.base / "out"

    def cleanup(self):
        for current, dirnames, _ in os.walk(self.base):
            for name in dirnames:
                path = Path(current) / name
                if not path.is_symlink():
                    os.chmod(path, 0o700)
        self.temporary.cleanup()

    def run_mirror(self, *extra, spec=None):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = self.mirror.main(["--spec", str(spec or self.spec), "--out-root", str(self.out), *extra])
        return code, json.loads(out.getvalue())

    def test_clean_tree_equals_archive_plus_generated_files(self):
        code, report = self.run_mirror()
        self.assertEqual(code, 0, report)
        tree = self.out / "repos" / "toy"
        files = sorted(p.relative_to(tree).as_posix() for p in tree.rglob("*") if p.is_file() or p.is_symlink())
        self.assertEqual(files, [".gitattributes", ".gitignore", "README", "docs/link_inside", "pkg/__init__.py",
                                 "pkg/_version.py", "pkg/core.py", "tests/test_core.py"])
        self.assertEqual((tree / "pkg" / "__init__.py").read_text(), "VERSION = '$Format:%H$'\n")  # no export-subst
        self.assertTrue((tree / "docs" / "link_inside").is_symlink())
        for path in [tree] + list(tree.rglob("*")):
            if not path.is_symlink():
                self.assertEqual(stat.S_IMODE(os.lstat(path).st_mode), 0o755 if path.is_dir() else 0o444, path)
        manifest_path = self.out / "repos-manifests" / "clean-checkouts.json"
        manifest = json.loads(manifest_path.read_bytes())
        entry = manifest["families"]["toy"]
        self.assertEqual((manifest["schema"], manifest["tree_protocol"]), (tools.CHECKOUTS_SCHEMA, tools.TREE_PROTOCOL))
        self.assertEqual(tools.checkout_entry_problems(entry), [])
        self.assertEqual(entry["tree_sha256"], tools.checkout_tree(tree)["tree_sha256"])
        analysis = source_tree_sha256(file_sha256s(self.repo, scan_package_files(self.repo, ["pkg"])["files"]))
        self.assertEqual(entry["package_tree_sha256"], analysis)
        provenance = manifest["mirror"]["families"]["toy"]
        self.assertEqual([g["path"] for g in provenance["generated_in_scope_files"]], ["pkg/_version.py"])
        self.assertTrue({"nohup.out", "tmp/leftover.txt"} <= set(provenance["excluded_untracked_or_ignored"]))
        self.assertEqual([d["path"] for d in provenance["dropped_symlinks"]], ["pkg/escape"])
        self.assertIn(".gitattributes: pkg/__init__.py export-subst", provenance["archive"]["export_attribute_lines"])
        self.assertEqual(report["manifest"]["sha256"], sha(manifest_path.read_bytes()))
        staged = STAGE.plan_checkouts(str(manifest_path), report["manifest"]["sha256"], ["toy"],
                                      {"toy": analysis})  # the stager accepts the mirror's manifest
        self.assertEqual(staged["summary"]["staged_families"]["toy"]["tree_sha256"], entry["tree_sha256"])
        code, again = self.run_mirror()
        self.assertEqual((code, again["families"]["toy"]["tree_status"], again["manifest"]["status"]),
                         (0, "already_present_verified", "already_present_verified"))
        self.assertEqual(again["manifest"]["sha256"], report["manifest"]["sha256"])

    def test_refuses_tracked_changes_a_wrong_head_and_a_differing_static_tree(self):
        write(self.repo / "pkg" / "core.py", "def f():\n    return 2\n")
        code, report = self.run_mirror()
        self.assertEqual(code, 2)
        self.assertIn("tracked changes", report["reason"])
        git(self.repo, "checkout", "-q", "--", "pkg/core.py")
        wrong = write(self.base / "wrong.json", json.dumps({"families": [
            {"family": "toy", "repo_root": str(self.repo), "package_dirs": ["pkg"], "commit": "f" * 40}]}))
        code, report = self.run_mirror(spec=wrong)
        self.assertEqual(code, 2)
        self.assertIn("not the pinned commit", report["reason"])
        static = self.base / "static"
        write(static / "locagent.manifest.json", json.dumps({"source_tree_sha256": "9" * 64}))
        stale = write(self.base / "stale.json", json.dumps({"families": [
            {"family": "toy", "repo_root": str(self.repo), "package_dirs": ["pkg"], "commit": self.commit,
             "static_dir": str(static)}]}))
        code, report = self.run_mirror(spec=stale)
        self.assertEqual(code, 2)
        self.assertIn("differs from", report["reason"])
        self.assertFalse((self.out / "repos" / "toy").exists())

    def test_an_existing_tree_that_differs_is_never_replaced(self):
        self.assertEqual(self.run_mirror()[0], 0)
        target = self.out / "repos" / "toy" / "README"
        os.chmod(target, 0o644)
        target.write_text("changed\n")
        code, report = self.run_mirror()
        self.assertEqual(code, 2)
        self.assertIn("refusing to replace", report["reason"])
        self.assertEqual(target.read_text(), "changed\n")


# ---------------------------------------------------------------------------
# Operator script
# ---------------------------------------------------------------------------

class LaunchScriptTests(unittest.TestCase):
    SCRIPT = PROJECT / "scripts" / "expA_core1_launch.sh"

    def test_the_loader_matches_the_publisher(self):
        match = re.search(r'^LOADER="(.*)"$', self.SCRIPT.read_text(), re.MULTILINE)
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), PUBLISH.LOADER)

    @unittest.skipUnless(shutil.which("bash"), "bash required")
    def test_syntax_and_identity_guard(self):
        subprocess.run(["bash", "-n", str(self.SCRIPT)], check=True, timeout=60)
        result = subprocess.run(["bash", str(self.SCRIPT), "gpus"], capture_output=True, text=True, timeout=60,
                                stdin=subprocess.DEVNULL)
        if os.getuid() == 1008 and os.uname().nodename.split(".")[0] == "ai-core-team-1":
            self.skipTest("running as the Core1 operator")
        self.assertEqual(result.returncode, 2)
        self.assertRegex(result.stderr, "run as thilak.arjun|must be UID 1008|runs on ai-core-team-1")

    def test_gpu5_and_unsafe_values_are_refused_in_the_script(self):
        text = self.SCRIPT.read_text()
        self.assertIn("GPU5_UUID=" + serve.GPU5_UUID, text)
        self.assertIn('[ "$index" != "5" ]', text)
        self.assertIn("ReadWritePaths=$ROOT/run $ROOT/scratch $ROOT/ledger $ROOT/logs", text)

    def test_review_fixes_are_wired_into_the_operator_script(self):
        text = self.SCRIPT.read_text()
        # /dev/shm read-only in the stager, runner and reconcile units (ProtectSystem=strict leaves it writable).
        self.assertIn('SHM_READ_ONLY="ReadOnlyPaths=/dev/shm"', text)
        self.assertEqual(text.count('--property="$SHM_READ_ONLY"'), 3)
        # Reboot safety: every stage call checks the Core2 run directory; --restore copies it back first.
        self.assertIn("--components release,dispatch,checkouts,run-state", text)
        self.assertIn('--run-target "$TARGET_PARENT/$RUN_ID" "${restore[@]}"', text)
        self.assertIn("STAGE_RECEIPT.run-state.*.json", text)
        # Final publish: serve units stopped, abandoned attempts closed, exit 5 when records still wait.
        self.assertIn("is still active; stop-serve first", text)
        self.assertIn('reconcile --run-dir "$ROOT/run"', text)
        self.assertIn("5 = records still waiting", text)
        # Restarts: failed units never block their name.
        self.assertEqual(text.count("systemctl reset-failed"), 2)
        self.assertIn("--allow-out-of-order", text)


if __name__ == "__main__":
    unittest.main()
