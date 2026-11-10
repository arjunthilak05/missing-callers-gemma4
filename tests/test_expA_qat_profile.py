"""Exp A model profile gemma4-31b-qat: the main-track W4A16 checkpoint, served with the frozen 31B BF16 chat template.

(1) The default (gemma4-31b) and gemma4-12b records are those of the release before this profile (base 0796e6020):
    serve plans, vLLM argv, verify-model calls, run.json, one live episode and the launch script's records, compared
    by digest (BEFORE). run.json is compared without its code identity, which hashes scripts/expA_run.py itself, and
    with the temporary fixture's clean-checkout manifest sha256 masked (it differs between temporary directories).
    ``python -B tests/test_expA_qat_profile.py --capture`` prints the current digests and values.
(2) The QAT profile verifies a fake copy, manifest and template in a temporary directory, and refuses a wrong template
    sha256, a missing or symlinked template, and a wrong manifest pin; a bad template stops preflight before the
    model hash.
(3) Its vLLM argv is the frozen one plus exactly one ``--chat-template PATH``; no other profile has one; the supervisor
    launches exactly that argv, or nothing when the template is wrong.
(4) The serve plan, the preflight record and run.json record the override (path and sha256).
(5) The launch script takes ``--model-profile gemma4-31b-qat`` from prepare (launch.env) to the serve and runner units
    on a stubbed host; ``bash -n`` passes.
Linux only, like tests/test_expA_param.py: serve.py imports pwd.
"""
import contextlib
import dataclasses
import functools
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

POSIX = os.name == "posix"
if POSIX:
    import test_expA_default_golden as G      # puts the project on sys.path
    import test_probe_harness as H
    from absent_edges.probe import harness, serve

GPU7 = "GPU-4cb533cc-d12b-8e68-7177-39e79451bdad"
CORE1_OK = {"uid": 1008, "euid": 1008, "gid": 1009, "egid": 1009, "user": "thilak.arjun", "host": "ai-core-team-1"}
PUBLIC = "/mnt/kldata/absent-edges-expA-public-20261101"
QAT = "gemma4-31b-qat"
QAT_ALIAS = "absent-edges-gemma4-31b-qat-w4a16"
QAT_HUB_ID = "google/gemma-4-31B-it-qat-w4a16-ct"
QAT_REVISION = "52f3f65bc7a02d555763bc923bd1d9094898219d"
QAT_DIR = f"{PUBLIC}/models/gemma4-31b-qat-w4a16-52f3f65b"
QAT_MANIFEST = f"{PUBLIC}/models/COPY-gemma4-31b-qat-w4a16-52f3f65b.sha256"
QAT_MANIFEST_SHA256 = "067a141c41ecb627c6975e7effb8ca07abb9a2f24b28d14ed367d76d4f4f6cba"
FROZEN_TEMPLATE = f"{PUBLIC}/models/gemma4-31b-bf16-518276fb/chat_template.jinja"
FROZEN_TEMPLATE_SHA256 = "36e3a42e5cf14cd0020e72d92e1fdd9970f59b82170e421f0cbe1bb42bead3f0"
OVERRIDE = {"path": FROZEN_TEMPLATE, "sha256": FROZEN_TEMPLATE_SHA256}
# The files of the public mirror's QAT copy that `sha256sum *` lists (it skips .gitattributes, which is tolerated).
QAT_FILES = ("README.md", "chat_template.jinja", "config.json", "generation_config.json", "model.safetensors",
             "processor_config.json", "tokenizer.json", "tokenizer_config.json")
ALIASES = {"gemma4-31b": "absent-edges-gemma4-31b-bf16", "gemma4-12b": "absent-edges-gemma4-12b-bf16", QAT: QAT_ALIAS}
UNCHANGED_PROFILES = ("gemma4-31b", "gemma4-12b")
TMPFS = [("/", "ext4", "/dev/sda1"), ("/dev/shm", "tmpfs", "tmpfs")]

# (1) Digests of the records below, captured on the release before this profile (base 0796e6020, WSL Ubuntu,
# Python 3.12.3, 2026-11-02) with the functions of this file.
BEFORE = {
    "gemma4-31b/plan_launch_script": "ba43eaba5588936fa99e7ca99296014da5786aaa0dbb6c702869b16481dc8b77",
    "gemma4-31b/plan_cli_defaults": "ba43eaba5588936fa99e7ca99296014da5786aaa0dbb6c702869b16481dc8b77",
    "gemma4-31b/verify_model": "a3a594f0a60278081fbc445effadbb5a3787145af8f4cf5f4a378a0b1f265c8a",
    "gemma4-31b/vllm_argv": "46a926dff4718cfbc1b8f4feecdcb2405566cede018af4f329e9d8bef765e6cb",
    "gemma4-31b/run_json": "9e7e94719a670fb9aaea31876ec7aa2a527db9850ab8c3a0bceb183a3d363a87",
    "gemma4-31b/live_episode": "522d5d435335b969540c37cfcdda1bcf953ea938d313edba1656a509e87fb3c7",
    "gemma4-31b/launch": "c42bf66eac54b476cad5d9b06d571234b9b7b89d8efda89f1beaebb8147994e6",
    "gemma4-12b/plan_launch_script": "f7c63234bf75d2d9028ed3fca00aa3730a8d4d7a1622d9cb794b1f53158a180e",
    "gemma4-12b/plan_cli_defaults": "f7c63234bf75d2d9028ed3fca00aa3730a8d4d7a1622d9cb794b1f53158a180e",
    "gemma4-12b/verify_model": "3d4cebb6745ada210e7bb3ebfc12d9bb1da0d11ab4ab6e6115afea560d7804c0",
    "gemma4-12b/vllm_argv": "fb90657489b6529baa7a4e609f97998033ece3e56dcc6de9321ecde1c775da81",
    "gemma4-12b/run_json": "72d5eb0c927370d63722851cf9d2995cccdb3d985bbf38e6b66cd0723617d217",
    "gemma4-12b/live_episode": "e35922c3ccf369cb16f5fde3406f45cd8e8e78b8dbafa49df6dab31a689afb6f",
    "gemma4-12b/launch": "f35627cc4efd92d11c23e6d3ba8f0db1c0fd7c04c817cd60732b0289cd664591",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data if isinstance(data, bytes) else data.encode())
    return path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(value):
    return sha(canonical(value).encode("utf-8"))


def profile_argv(profile):
    return [] if profile == serve.DEFAULT_MODEL_PROFILE else ["--model-profile", profile]


def chat_template_flags(argv):
    return [item for item in argv if item == "--chat-template" or item.startswith("--chat-template=")]


def qat_copy(root):
    """A tiny copy shaped like the mirror's QAT copy (its own, newer chat_template.jinja included), its ``sha256sum *``
    manifest next to it, and a stand-in for the frozen 31B template in a sibling directory."""
    model = Path(root) / "gemma4-31b-qat-w4a16-52f3f65b"
    lines = []
    for name in QAT_FILES:
        data = f"qat copy: {name}\n".encode()
        write(model / name, data)
        lines.append(f"{sha(data)}  {name}")
    write(model / ".gitattributes", "*.safetensors filter=lfs diff=lfs merge=lfs -text\n")
    manifest = write(Path(root) / "COPY-gemma4-31b-qat-w4a16-52f3f65b.sha256", "\n".join(lines) + "\n")
    template = write(Path(root) / "gemma4-31b-bf16-518276fb" / "chat_template.jinja", "frozen 31B template stand-in\n")
    return model, manifest, template


@contextlib.contextmanager
def qat_profile(**changes):
    """serve.MODEL_PROFILES with the QAT profile changed (for example its template path and pin)."""
    profile = dataclasses.replace(serve.MODEL_PROFILES[QAT], **changes)
    with mock.patch.dict(serve.MODEL_PROFILES, {QAT: profile}):
        yield profile


def live_run(runner, base, dispatch, manifest, profile, run=None):
    """One default-budget episode of ``profile`` (cell P s11) against a fake server that serves its alias."""
    run = run or profile
    server = H.FakeServer(default=H.SUBMIT)
    server.model = ALIASES[profile]
    try:
        argv = G.live_args(base, dispatch, manifest, server, run=run) + ["--run-id", "same", *profile_argv(profile)]
        return G.live_episode(runner, argv, server, base / run)
    finally:
        server.close()


def comparable_run_json(run_json):
    """run.json without its code identity and with the temporary fixture's checkout-manifest sha256 masked."""
    value = json.loads(json.dumps(run_json))
    value.pop("code")
    value["checkouts"]["manifest_sha256"] = "<fixture>"
    return value


def unchanged_records():
    """The default and 12B records that BEFORE pins."""
    values = {}
    for profile in UNCHANGED_PROFILES:
        extra = profile_argv(profile)
        values[f"{profile}/plan_launch_script"] = G.cli_plan(G.LAUNCH_PLAN_ARGV + extra)
        values[f"{profile}/plan_cli_defaults"] = G.cli_plan(G.CLI_DEFAULT_PLAN_ARGV + extra)
        values[f"{profile}/verify_model"] = G.verify_model_call(["verify-model"] + extra)
        spec = serve.ServeSpec.for_profile(profile, run="golden1", instance="g7", gpu_uuid=GPU7, port=8141)
        values[f"{profile}/vllm_argv"] = serve.vllm_argv(spec)
    with tempfile.TemporaryDirectory() as directory:
        base = Path(directory)
        try:
            runner, dispatch, manifest = G.runner_fixture(base)
            for profile in UNCHANGED_PROFILES:
                live = live_run(runner, base, dispatch, manifest, profile)
                values[f"{profile}/run_json"] = comparable_run_json(live.pop("run_json"))
                values[f"{profile}/live_episode"] = live
        finally:
            for path in base.rglob("*"):
                if path.is_dir() and not path.is_symlink():
                    path.chmod(0o755)
    for profile in UNCHANGED_PROFILES:
        with tempfile.TemporaryDirectory() as directory:
            values[f"{profile}/launch"] = G.launch_values(Path(directory), prepare_extra=tuple(profile_argv(profile)))
    return values


# ---------------------------------------------------------------------------------------------------------------
# (1) the default and the 12B: unchanged
# ---------------------------------------------------------------------------------------------------------------

@unittest.skipUnless(POSIX and shutil.which("bash"), "Linux only: serve.py imports pwd")
class UnchangedProfilesTests(unittest.TestCase):
    maxDiff = None

    @classmethod
    def setUpClass(cls):
        cls.values = unchanged_records()

    def test_default_and_12b_records_are_those_of_the_release_before(self):
        self.assertEqual(sorted(self.values), sorted(BEFORE))
        for key, expected in sorted(BEFORE.items()):
            with self.subTest(record=key):
                self.assertEqual(digest(self.values[key]), expected, canonical(self.values[key])[:6000])

    def test_no_chat_template_anywhere_in_their_records(self):
        for profile in UNCHANGED_PROFILES:
            with self.subTest(profile=profile):
                served = serve.served_model(profile)
                self.assertEqual((served.chat_template, served.chat_template_sha256, served.chat_template_override),
                                 (None, None, None))
                self.assertNotIn("chat_template_override", served.describe())
                self.assertEqual(chat_template_flags(self.values[f"{profile}/vllm_argv"]), [])
                for name in ("plan_launch_script", "plan_cli_defaults"):
                    plan = self.values[f"{profile}/{name}"]["plan"]
                    self.assertNotIn("chat_template_override", plan["model"])
                    self.assertEqual(chat_template_flags(plan["vllm_argv"]), [])
                self.assertNotIn("chat_template_override", self.values[f"{profile}/run_json"]["model"])
                launch_plan = self.values[f"{profile}/launch"]["serve_plan"]
                self.assertNotIn("chat_template_override", launch_plan["model"])
                self.assertEqual(chat_template_flags(launch_plan["vllm_argv"]), [])
                spec = serve.ServeSpec.for_profile(profile, run="r", instance="g7", gpu_uuid=GPU7, port=8141)
                self.assertEqual(serve.mirror_roots(spec), [spec.serve_env, spec.runtime_root, spec.model_dir,
                                                            spec.model_manifest])

    def test_verify_model_prints_no_template_record_for_them(self):
        for profile in UNCHANGED_PROFILES:
            with self.subTest(profile=profile), \
                    mock.patch.object(serve, "verify_model", return_value={"failures": []}), \
                    contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(G.load_script("expA_serve").main(["verify-model", *profile_argv(profile)]), 0)
            self.assertEqual(json.loads(out.getvalue()), {"failures": []})

    def test_a_12b_preflight_record_has_the_keys_it_had(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                     "generation_config.json", "model.safetensors")
            lines = [f"{sha(write(root / 'model' / name, name).read_bytes())}  {name}" for name in names]
            manifest = write(root / "COPY.sha256", "\n".join(lines) + "\n")
            for name in ("python", "ninja"):
                write(root / "env" / "bin" / name, "#!/bin/sh\n").chmod(0o755)
            spec = serve.ServeSpec.for_profile("gemma4-12b", run="t12", instance="g7", gpu_uuid=GPU7, port=8201,
                                               serve_env=str(root / "env"), model_dir=str(root / "model"),
                                               model_manifest=str(manifest),
                                               model_manifest_sha256=sha(manifest.read_bytes()))
            result = serve.preflight(spec, observe=lambda uuid: {"uuid": uuid, "index": 7, "minor_number": 7,
                                                                 "used_mib": 0, "total_mib": 97887, "pids": []},
                                     port_free=lambda port: True, magic=lambda path: None, identity=CORE1_OK,
                                     mounts=TMPFS, proc_options=lambda: {"options": ["rw"]})
        self.assertTrue(result["passed"], result["failures"])
        self.assertEqual(sorted(result), ["cache", "checked_unix", "failures", "gpu", "identity", "mirror_ownership",
                                          "model", "passed", "proc_mount", "protocol", "spec"])


# ---------------------------------------------------------------------------------------------------------------
# The profile itself
# ---------------------------------------------------------------------------------------------------------------

@unittest.skipUnless(POSIX, "Linux only: serve.py imports pwd")
class QatProfileTests(unittest.TestCase):
    def test_the_qat_profile(self):
        profile = serve.served_model(QAT)
        self.assertEqual(dataclasses.asdict(profile), {
            "name": QAT, "hub_id": QAT_HUB_ID, "revision": QAT_REVISION, "alias": QAT_ALIAS, "model_dir": QAT_DIR,
            "manifest": QAT_MANIFEST, "manifest_sha256": QAT_MANIFEST_SHA256, "weights": "single",
            "chat_template": FROZEN_TEMPLATE, "chat_template_sha256": FROZEN_TEMPLATE_SHA256})
        self.assertEqual(profile.required_files, ("config.json", "tokenizer.json", "tokenizer_config.json",
                                                  "chat_template.jinja", "generation_config.json", "model.safetensors"))
        self.assertTrue(set(profile.required_files) <= set(QAT_FILES))
        self.assertEqual(profile.chat_template_override, OVERRIDE)
        # The override is the default copy's own template, the one models/COPY.sha256 pins.
        self.assertEqual((profile.chat_template, profile.chat_template_sha256),
                         (str(serve.MODEL_CHAT_TEMPLATE), serve.MODEL_CHAT_TEMPLATE_SHA256))
        self.assertEqual(Path(profile.chat_template).parent, serve.MODEL_DIR)
        self.assertEqual([name for name, item in sorted(serve.MODEL_PROFILES.items()) if item.chat_template], [QAT])
        self.assertEqual(serve.DEFAULT_MODEL_PROFILE, "gemma4-31b")

    def test_an_override_names_its_path_and_pins_its_sha256(self):
        qat = serve.served_model(QAT)
        for bad in (dict(chat_template=None), dict(chat_template_sha256=None),
                    dict(chat_template="models/chat_template.jinja"), dict(chat_template_sha256="0" * 63),
                    dict(chat_template_sha256=FROZEN_TEMPLATE_SHA256.upper())):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                dataclasses.replace(qat, **bad)
        plain = dataclasses.replace(qat, chat_template=None, chat_template_sha256=None)
        self.assertIsNone(plain.chat_template_override)
        for bad in (None, "0" * 63, FROZEN_TEMPLATE_SHA256.upper()):
            with self.subTest(expected=bad), self.assertRaises(ValueError):
                serve.verify_chat_template(FROZEN_TEMPLATE, bad)

    def test_a_spec_never_mixes_the_qat_profile_with_another_copy(self):
        for other in ("gemma4-31b", "gemma4-12b"):
            served = serve.served_model(other)
            for mixed in (dict(model_dir=served.model_dir), dict(model_manifest=served.manifest),
                          dict(model_manifest_sha256=served.manifest_sha256)):
                with self.subTest(other=other, mixed=mixed), \
                        self.assertRaisesRegex(ValueError, f"names the model files of profile {other}"):
                    serve.ServeSpec.for_profile(QAT, run="r", instance="g7", gpu_uuid=GPU7, port=8141, **mixed)
        with self.assertRaisesRegex(ValueError, f"names the model files of profile {QAT}"):
            serve.ServeSpec(run="r", instance="g7", gpu_uuid=GPU7, port=8141, model_dir=QAT_DIR)
        spec = serve.ServeSpec.for_profile(QAT, run="r", instance="g7", gpu_uuid=GPU7, port=8141)
        self.assertEqual((spec.model_dir, spec.model_manifest, spec.model_manifest_sha256),
                         (QAT_DIR, QAT_MANIFEST, QAT_MANIFEST_SHA256))

    def test_the_supervisor_checks_the_qat_alias(self):
        spec = serve.ServeSpec.for_profile(QAT, run="r", instance="g7", gpu_uuid=GPU7, port=8141)
        supervisor = serve.Supervisor(spec)
        self.assertTrue(supervisor._check_registry({"data": [{"id": QAT_ALIAS, "max_model_len": 32768,
                                                              "root": QAT_DIR}]}))
        for alias in (ALIASES["gemma4-31b"], ALIASES["gemma4-12b"]):
            self.assertFalse(supervisor._check_registry({"data": [{"id": alias, "max_model_len": 32768}]}))


# ---------------------------------------------------------------------------------------------------------------
# (2) verification: a fake copy, manifest and template; the refusals
# ---------------------------------------------------------------------------------------------------------------

@unittest.skipUnless(POSIX, "Linux only: serve.py imports pwd")
class QatVerifyTests(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.model, self.manifest, self.template = qat_copy(self.root)
        self.manifest_sha = sha(self.manifest.read_bytes())
        self.template_sha = sha(self.template.read_bytes())
        for name in ("python", "ninja"):
            write(self.root / "env" / "bin" / name, "#!/bin/sh\n").chmod(0o755)
        self.pinned = dict(chat_template=str(self.template), chat_template_sha256=self.template_sha)

    def spec(self, **values):
        values = dict(dict(serve_env=str(self.root / "env"), model_dir=str(self.model),
                           model_manifest=str(self.manifest), model_manifest_sha256=self.manifest_sha), **values)
        return serve.ServeSpec.for_profile(QAT, run="tq", instance="g7", gpu_uuid=GPU7, port=8202, **values)

    def preflight(self, spec=None, verify=None):
        return serve.preflight(spec or self.spec(), observe=lambda uuid: {"uuid": uuid, "index": 7, "minor_number": 7,
                                                                          "used_mib": 0, "total_mib": 97887,
                                                                          "pids": []},
                               port_free=lambda port: True, magic=lambda path: None, identity=CORE1_OK,
                               mounts=TMPFS, proc_options=lambda: {"options": ["rw"]},
                               verify=verify or serve.verify_model)

    def test_the_fake_copy_and_template_pass(self):
        with qat_profile(**self.pinned):
            result = self.preflight()
        self.assertTrue(result["passed"], result["failures"])
        self.assertEqual(result["chat_template_override"],
                         {"path": str(self.template), "expected_sha256": self.template_sha, "sha256": self.template_sha,
                          "bytes": self.template.stat().st_size, "failures": []})
        self.assertEqual(result["model"]["required_files"], list(serve.required_model_files("single")))
        self.assertEqual((result["model"]["sha256_verified_files"], result["model"]["unexpected_files"]), (8, []))
        self.assertEqual(result["spec"]["model_profile"], QAT)
        # The template and its directory are in the ownership record, like the model and its manifest.
        chain = [entry["path"] for entry in result["mirror_ownership"]["chain"]]
        self.assertIn(str(self.template), chain)
        self.assertIn(str(self.template.parent), chain)

    def test_a_wrong_missing_or_symlinked_template_is_refused_before_the_model_hash(self):
        link = self.root / "link" / "chat_template.jinja"
        link.parent.mkdir()
        link.symlink_to(self.template)
        cases = {"pinned sha256 of the real frozen template": (dict(chat_template=str(self.template)),
                                                              ["chat_template_sha256_mismatch"]),
                 "wrong pin": (dict(self.pinned, chat_template_sha256="0" * 64), ["chat_template_sha256_mismatch"]),
                 "missing": (dict(self.pinned, chat_template=str(self.root / "absent" / "chat_template.jinja")),
                             ["chat_template_missing:No such file or directory"]),
                 "symlink to the right bytes": (dict(self.pinned, chat_template=str(link)),
                                                ["chat_template_not_regular"]),
                 "a directory": (dict(self.pinned, chat_template=str(self.template.parent)),
                                 ["chat_template_not_regular"])}
        for name, (changes, failures) in cases.items():
            verify = mock.Mock()
            with self.subTest(case=name), qat_profile(**changes):
                result = self.preflight(verify=verify)
                self.assertFalse(result["passed"])
                self.assertEqual(result["failures"], failures)
                self.assertEqual(result["chat_template_override"]["failures"], failures)
                self.assertEqual(result["model"], {"skipped": "earlier preflight failures"})
                verify.assert_not_called()
        # The template changed after it was pinned.
        with qat_profile(**self.pinned):
            write(self.template, "a newer template\n")
            self.assertEqual(self.preflight()["failures"], ["chat_template_sha256_mismatch"])

    def test_a_wrong_manifest_pin_or_a_changed_file_is_refused(self):
        with qat_profile(**self.pinned):
            pinned = self.preflight(self.spec(model_manifest_sha256=None))   # the profile's own pin, 067a141c...
            self.assertEqual(pinned["spec"]["model_manifest_sha256"], QAT_MANIFEST_SHA256)
            self.assertEqual(pinned["failures"], ["model_manifest_sha256_mismatch"])
            self.assertEqual(pinned["chat_template_override"]["failures"], [])
            write(self.model / "model.safetensors", "tampered weights\n")
            self.assertEqual(self.preflight()["failures"], ["model_file_sha256_mismatch:model.safetensors"])

    def test_verify_model_checks_the_copy_and_the_template(self):
        cli = G.load_script("expA_serve")
        argv = ["verify-model", "--model-profile", QAT, "--model-dir", str(self.model),
                "--model-manifest", str(self.manifest)]

        def verify(**changes):
            with qat_profile(**changes), contextlib.redirect_stdout(io.StringIO()) as out:
                code = cli.main(argv)
            return code, json.loads(out.getvalue())
        ok = dict(self.pinned, manifest_sha256=self.manifest_sha)
        code, record = verify(**ok)
        self.assertEqual((code, record["failures"], record["sha256_verified_files"]), (0, [], 8))
        self.assertEqual(record["chat_template_override"],
                         {"path": str(self.template), "expected_sha256": self.template_sha, "sha256": self.template_sha,
                          "bytes": self.template.stat().st_size, "failures": []})
        for changes, failures in ((dict(chat_template_sha256=FROZEN_TEMPLATE_SHA256),
                                   ["chat_template_sha256_mismatch"]),
                                  (dict(chat_template=str(self.root / "absent.jinja")),
                                   ["chat_template_missing:No such file or directory"]),
                                  (dict(manifest_sha256=QAT_MANIFEST_SHA256), ["model_manifest_sha256_mismatch"])):
            with self.subTest(changes=changes):
                code, record = verify(**dict(ok, **changes))
                self.assertEqual((code, record["failures"]), (2, failures))
        # Without --model-dir and --model-manifest: the profile's copy. (A pin no manifest has: this host's mirror,
        # if it has one, fails fast instead of hashing the real weights.)
        with qat_profile(**dict(self.pinned, manifest_sha256="0" * 64)), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(cli.main(argv[:3]), 2)
        record = json.loads(out.getvalue())
        self.assertEqual((record["model_dir"], record["manifest"], record["expected_manifest_sha256"]),
                         (QAT_DIR, QAT_MANIFEST, "0" * 64))
        self.assertEqual(record["chat_template_override"]["failures"], [])


# ---------------------------------------------------------------------------------------------------------------
# (3) the vLLM argv: the frozen flags plus exactly one --chat-template; the supervisor launches exactly that
# ---------------------------------------------------------------------------------------------------------------

class FakeChild:
    """A vLLM child that has already exited: the supervisor records ``exited`` and returns 1."""
    pid = 4242424          # above the default pid_max: never a live process, never signalled
    returncode = 1

    def poll(self):
        return self.returncode


@unittest.skipUnless(POSIX, "Linux only: serve.py imports pwd")
class QatArgvTests(unittest.TestCase):
    def test_exactly_one_chat_template_and_the_frozen_flags(self):
        spec = serve.ServeSpec.for_profile(QAT, run="tq", instance="g7", gpu_uuid=GPU7, port=8141)
        argv = serve.vllm_argv(spec)
        self.assertEqual(chat_template_flags(argv), ["--chat-template"])
        self.assertEqual(argv[-2:], ["--chat-template", FROZEN_TEMPLATE])
        self.assertNotIn(QAT_DIR + "/chat_template.jinja", argv)    # the copy's own, newer template is not served
        self.assertEqual(argv[argv.index("--model") + 1], QAT_DIR)
        self.assertEqual(argv[argv.index("--served-model-name") + 1], QAT_ALIAS)
        # Everything else is the default instance's argv: the frozen flags, no --quantization (vLLM reads the
        # compressed-tensors W4A16 config from config.json).
        default = serve.ServeSpec(run="tq", instance="g7", gpu_uuid=GPU7, port=8141)
        swap = {str(serve.MODEL_DIR): QAT_DIR, ALIASES["gemma4-31b"]: QAT_ALIAS}
        self.assertEqual(argv[:-2], [swap.get(item, item) for item in serve.vllm_argv(default)])
        for flag, value in (("--dtype", "bfloat16"), ("--max-model-len", "32768"), ("--max-num-seqs", "8"),
                            ("--gpu-memory-utilization", "0.92"), ("--tool-call-parser", "gemma4"),
                            ("--reasoning-parser", "gemma4"), ("--generation-config", "vllm"),
                            ("--host", "127.0.0.1")):
            self.assertEqual(argv[argv.index(flag) + 1], value, flag)
        for flag in ("--language-model-only", "--enable-auto-tool-choice", "--enable-prefix-caching"):
            self.assertEqual(argv.count(flag), 1, flag)
        self.assertFalse([item for item in argv if item.startswith("--quantization")])
        self.assertEqual(serve.serve_environment(spec), serve.serve_environment(default))
        self.assertEqual(serve.mirror_roots(spec)[-1], FROZEN_TEMPLATE)

    def test_no_other_profile_has_a_chat_template(self):
        self.assertEqual(sorted(serve.MODEL_PROFILES), ["gemma4-12b", "gemma4-31b", QAT])
        for name in serve.MODEL_PROFILES:
            spec = serve.ServeSpec.for_profile(name, run="tq", instance="g7", gpu_uuid=GPU7, port=8141)
            with self.subTest(profile=name):
                self.assertEqual(chat_template_flags(serve.vllm_argv(spec)), ["--chat-template"] if name == QAT else [])

    def supervise(self, directory, changes, run):
        """Supervisor.run of a QAT instance whose real preflight sees a fake GPU, port, tmpfs and model check."""
        root = Path(directory)
        for name in ("python", "ninja"):
            write(root / "env" / "bin" / name, "#!/bin/sh\n").chmod(0o755)
        launched = []

        def popen(argv, **kwargs):
            launched.append(argv)
            return FakeChild()
        check = functools.partial(serve.preflight, port_free=lambda port: True, identity=CORE1_OK,
                                  mounts=[("/", "ext4", "/dev/sda1"), (str(root), "tmpfs", "tmpfs")],
                                  magic=lambda path: None, proc_options=lambda: {"options": ["rw"]},
                                  verify=lambda *args, **kwargs: {"failures": [], "fake": True})
        with qat_profile(**changes):
            spec = serve.ServeSpec.for_profile(QAT, run=run, instance="g7", gpu_uuid=GPU7, port=8203,
                                               serve_env=str(root / "env"))
            supervisor = serve.Supervisor(spec, sleep=lambda seconds: None, popen=popen, preflight_check=check,
                                          observe=lambda uuid: {"uuid": uuid, "index": 7, "minor_number": 7,
                                                                "used_mib": 0, "total_mib": 97887, "pids": []},
                                          http_get=mock.Mock(side_effect=OSError("not ready")))
            code = supervisor.run()
            expected_argv = serve.vllm_argv(spec)
        records = {path.name.split("-", 1)[1]: json.loads(path.read_text())
                   for path in sorted((spec.cache_root / "records").iterdir())}
        return code, launched, expected_argv, records

    def test_the_supervisor_launches_the_frozen_template_or_nothing(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = os.path.realpath(temporary)          # the mount table of supervise() names it resolved
            with mock.patch.object(serve, "CACHE_PARENT", Path(directory)):
                template = write(Path(directory) / "frozen" / "chat_template.jinja", "frozen 31B template stand-in\n")
                pinned = dict(chat_template=str(template), chat_template_sha256=sha(template.read_bytes()))
                code, launched, argv, records = self.supervise(directory, pinned, "tq-ok")
                self.assertEqual(code, 1)                               # the fake child exited at once
                self.assertEqual(launched, [argv])
                self.assertEqual(argv[-2:], ["--chat-template", str(template)])
                self.assertEqual(sorted(records), ["exited.json", "launched.json", "preflight.json"])
                self.assertEqual(records["launched.json"]["argv"], argv)
                self.assertEqual(records["preflight.json"]["chat_template_override"]["failures"], [])
                # The real pin against the stand-in: preflight refuses and nothing is launched.
                code, launched, _, records = self.supervise(directory, dict(chat_template=str(template)), "tq-bad")
                self.assertEqual((code, launched), (2, []))
                self.assertEqual(sorted(records), ["preflight.json"])
                self.assertEqual(records["preflight.json"]["failures"], ["chat_template_sha256_mismatch"])


# ---------------------------------------------------------------------------------------------------------------
# (4) the serve plan and run.json record the override
# ---------------------------------------------------------------------------------------------------------------

@unittest.skipUnless(POSIX, "Linux only: serve.py imports pwd")
class QatPlanTests(unittest.TestCase):
    maxDiff = None

    def test_the_plan_records_the_override(self):
        spec = serve.ServeSpec.for_profile(QAT, run="golden1", instance="g7", gpu_uuid=GPU7, port=8141)
        plan = serve.plan(spec, identity=CORE1_OK)
        self.assertEqual(plan["model"], {
            "path": QAT_DIR, "alias": QAT_ALIAS, "revision": QAT_REVISION, "max_model_len": 32768,
            "manifest": QAT_MANIFEST, "manifest_sha256": QAT_MANIFEST_SHA256, "check": "full", "profile": QAT,
            "hub_id": QAT_HUB_ID, "weights": "single", "required_files": list(serve.required_model_files("single")),
            "chat_template_override": OVERRIDE})
        self.assertEqual(plan["vllm_argv"], serve.vllm_argv(spec))
        self.assertEqual(plan["spec"]["model_profile"], QAT)
        default = serve.plan(serve.ServeSpec(run="golden1", instance="g7", gpu_uuid=GPU7, port=8141),
                             identity=CORE1_OK)
        self.assertEqual(sorted(plan), sorted(default))
        self.assertEqual(sorted(set(plan["model"]) ^ set(default["model"])), ["chat_template_override"])
        # The unit's --spec names the profile, so the supervisor rebuilds this spec and serves the override.
        unit_spec = json.loads(plan["systemd_run_argv"][-1])
        self.assertEqual(unit_spec["model_profile"], QAT)
        self.assertEqual(serve.ServeSpec(**unit_spec), spec)
        self.assertEqual(serve.vllm_argv(serve.ServeSpec(**unit_spec))[-2:], ["--chat-template", FROZEN_TEMPLATE])
        # The plan the launch script requests.
        cli = G.cli_plan(G.LAUNCH_PLAN_ARGV + ["--model-profile", QAT])
        self.assertEqual(cli["code"], 0)
        self.assertEqual(cli["plan"]["model"], plan["model"])
        self.assertEqual(cli["plan"]["vllm_argv"], plan["vllm_argv"])


@unittest.skipUnless(POSIX, "the runner reads checkouts with POSIX no-follow reads")
class QatRunnerTests(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.base = Path(temporary.name)

        def cleanup():
            for path in self.base.rglob("*"):
                if path.is_dir() and not path.is_symlink():
                    path.chmod(0o755)
            temporary.cleanup()
        self.addCleanup(cleanup)
        self.runner, self.dispatch, self.manifest = G.runner_fixture(self.base)

    def test_the_runner_options(self):
        with contextlib.redirect_stderr(io.StringIO()):
            args = self.runner.parse_args(["--dispatch", str(self.dispatch), "--dry-run", "--model-profile", QAT])
            self.assertEqual((args.model_profile, args.model, args.episode_config),
                             (QAT, QAT_ALIAS, harness.EpisodeConfig()))
            for bad in (["--model-profile", QAT, "--model", ALIASES["gemma4-31b"]], ["--model", QAT_ALIAS]):
                with self.subTest(bad=bad), self.assertRaises(SystemExit):
                    self.runner.parse_args(["--dispatch", str(self.dispatch), "--dry-run", *bad])

    def test_run_json_records_the_override_and_differs_from_the_default_only_in_model(self):
        default = live_run(self.runner, self.base, self.dispatch, self.manifest, "gemma4-31b")
        qat = live_run(self.runner, self.base, self.dispatch, self.manifest, QAT)
        self.assertEqual((qat["exit_code"], qat["results"], qat["result"]["exit_status"]), (0, 1, "Submitted"))
        self.assertEqual(qat["run_json"]["model"], {
            "alias": QAT_ALIAS, "max_context": 32768, "enable_thinking": False, "temperature": 0.2, "profile": QAT,
            "hub_id": QAT_HUB_ID, "revision": QAT_REVISION, "chat_template_override": OVERRIDE})
        self.assertEqual({key: value for key, value in qat["run_json"].items() if key != "model"},
                         {key: value for key, value in default["run_json"].items() if key != "model"})
        expected_profile = {"endpoint": "<fake server>", "model": QAT_ALIAS, "revision": QAT_REVISION,
                            "max_context": 32768, "enable_thinking": False}
        self.assertEqual((qat["result"]["model_profile"], qat["setup"]["model_profile"]),
                         (expected_profile, expected_profile))
        self.assertEqual(qat["chat_request"], dict(default["chat_request"], model=QAT_ALIAS))
        self.assertEqual((qat["result"]["config"], qat["setup"]["system"]),
                         (default["result"]["config"], default["setup"]["system"]))
        # On disk as written: the override in run.json's model.
        written = json.loads((self.base / QAT / "run.json").read_bytes())
        self.assertEqual(written["model"]["chat_template_override"], OVERRIDE)

    def test_an_endpoint_serving_another_alias_is_refused(self):
        server = H.FakeServer(default=H.SUBMIT)          # serves the 31B alias
        self.addCleanup(server.close)
        argv = G.live_args(self.base, self.dispatch, self.manifest, server, run="wrong") + ["--model-profile", QAT]
        with mock.patch.object(G.tools, "ReadOnlyBash", G.FakeJailBash), \
                self.assertRaisesRegex(RuntimeError, f"does not serve exactly {QAT_ALIAS}"):
            G.invoke_runner(self.runner, argv)
        self.assertEqual(server.chat_requests(), [])


# ---------------------------------------------------------------------------------------------------------------
# (5) the launch script: prepare -> launch.env -> serve plan -> runner unit (stubbed host, nothing is started)
# ---------------------------------------------------------------------------------------------------------------

@unittest.skipUnless(POSIX and shutil.which("bash"), "Linux only: the launch script runs on Core1")
class QatLaunchScriptTests(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.temporary = Path(temporary.name)
        self.host = G.LaunchHost(temporary.name)
        self.control = self.host.run_root("golden1") / "control"

    def prepare(self, *extra, check=True):
        return self.host.run(*G.PREPARE_ARGV, "--runtime-python", str(self.host.python), *extra, check=check)

    def endpoint_ready_log(self):
        path = self.temporary / "endpoint_ready.log"
        text = path.read_text() if path.exists() else ""
        path.unlink(missing_ok=True)
        return text

    def test_syntax(self):
        result = subprocess.run(["bash", "-n", str(G.PROJECT / "scripts" / "expA_core1_launch.sh")],
                                capture_output=True, text=True, timeout=60)
        self.assertEqual((result.returncode, result.stderr), (0, ""))

    def test_a_qat_run_flows_from_prepare_to_the_runner_unit(self):
        prepared = self.prepare("--model-profile", QAT)
        launch_env = (self.control / "launch.env").read_text()
        self.assertEqual(launch_env, G.GOLDEN["launch"]["launch_env"].replace("<T>", str(self.temporary))
                         + f"MODEL_PROFILE={QAT}\n")
        self.assertEqual(prepared.stdout, launch_env)
        self.assertEqual(self.host.run("show-config", "--run", "golden1").stdout, launch_env)
        self.host.stage("golden1")
        serve_result = self.host.run("serve", "--run", "golden1")
        serve_units = [self.host.normalize(call) for call in G.units(self.host.sudo_calls(), "serve")]
        self.assertIn(f"then starts vLLM as {QAT_ALIAS}", serve_result.stderr)
        plan = json.loads((self.control / "serve-g7.plan.json").read_text())
        self.assertEqual({key: plan["model"][key] for key in ("profile", "alias", "revision", "path", "manifest",
                                                               "manifest_sha256", "chat_template_override")},
                         {"profile": QAT, "alias": QAT_ALIAS, "revision": QAT_REVISION, "path": QAT_DIR,
                          "manifest": QAT_MANIFEST, "manifest_sha256": QAT_MANIFEST_SHA256,
                          "chat_template_override": OVERRIDE})
        self.assertEqual(plan["vllm_argv"][-2:], ["--chat-template", FROZEN_TEMPLATE])
        self.assertEqual(chat_template_flags(plan["vllm_argv"]), ["--chat-template"])
        # The serve unit is the default one except its --spec, which names the QAT profile and copy.
        golden_serve = G.GOLDEN["launch"]["serve_units"]
        self.assertEqual(len(serve_units), 1)
        self.assertEqual(serve_units[0][:-1], golden_serve[0][:-1])
        unit_spec = json.loads(serve_units[0][-1])
        changed = ("model_profile", "model_dir", "model_manifest", "model_manifest_sha256")
        self.assertEqual({key: value for key, value in unit_spec.items() if key not in changed},
                         {key: value for key, value in json.loads(golden_serve[0][-1]).items() if key not in changed})
        self.assertEqual([unit_spec[key] for key in changed], [QAT, QAT_DIR, QAT_MANIFEST, QAT_MANIFEST_SHA256])
        self.assertEqual(serve.vllm_argv(serve.ServeSpec(**unit_spec))[-2:], ["--chat-template", FROZEN_TEMPLATE])
        # The runner unit is the default one plus --model-profile; the endpoints must serve the QAT alias.
        result = self.host.run("runner", "--run", "golden1", "--blocks", "P", "--seeds", "11")
        runner_units = [self.host.normalize(call) for call in G.units(self.host.sudo_calls(), "runner")]
        self.assertEqual(self.endpoint_ready_log(), f"8141 {QAT_ALIAS}\n")
        self.assertIn(f"model {QAT} ({QAT_ALIAS}); budgets prereg defaults (8 responses, 16384 tokens)",
                      result.stderr)
        self.assertEqual(runner_units, [G.GOLDEN["launch"]["runner_units"][0] + ["--model-profile", QAT]])
        unit = runner_units[0]
        tail = unit[next(i for i, value in enumerate(unit) if value.endswith("/scripts/expA_run.py")) + 1:]
        with contextlib.redirect_stderr(io.StringIO()):
            args = G.load_script("expA_run").parse_args(tail)
        self.assertEqual((args.model_profile, args.model, args.episode_config),
                         (QAT, QAT_ALIAS, harness.EpisodeConfig()))

    def test_add_instance_keeps_the_profile(self):
        self.prepare("--model-profile", QAT)
        other = "GPU-00000000-0000-0000-0000-000000000006"      # a fake UUID: a second instance of the same run
        self.host.run("add-instance", "--run", "golden1", "--instance", f"g6={other}=8142")
        text = (self.control / "launch.env").read_text()
        self.assertIn(f"\nINSTANCES=g7={GPU7}=8141 g6={other}=8142\n", text)
        self.assertTrue(text.endswith(f"MODEL_PROFILE={QAT}\n"))

    def test_the_refusal_names_every_profile(self):
        refused = self.prepare("--model-profile", "gemma4-31b-qat-int4", check=False)
        self.assertEqual(refused.returncode, 2)
        self.assertIn("--model-profile must be gemma4-31b-qat or gemma4-31b or gemma4-12b, not gemma4-31b-qat-int4",
                      refused.stderr)
        self.assertFalse((self.control / "launch.env").exists())


if __name__ == "__main__":
    if sys.argv[1:] == ["--capture"]:
        values = unchanged_records()
        print(json.dumps({"digests": {key: digest(value) for key, value in sorted(values.items())}, "values": values},
                         indent=1, sort_keys=True, ensure_ascii=False))
    else:
        unittest.main()
