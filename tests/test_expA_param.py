"""Exp A parameterisation: model profiles (protocols/EXPA_REPL12B.md) and episode budgets (protocols/EXPA_BUDGET16.md).

The default path is pinned by tests/test_expA_default_golden.py; here are (1) exactly what a default invocation adds to
the golden records, (2) the 12B profile against a fake copy, (3) the required files of single-file and sharded copies,
(4) the runner's budget and profile options end to end, and (5) the launch script's prepare -> launch.env -> serve plan
-> runner path on a stubbed host. Linux only, like the golden test (serve.py imports pwd).
"""
import contextlib
import dataclasses
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

POSIX = os.name == "posix"
if POSIX:
    from absent_edges.probe import harness, serve
    from absent_edges.probe import ledger as L
    import test_expA_default_golden as G
    import test_probe_harness as H

GPU7 = "GPU-4cb533cc-d12b-8e68-7177-39e79451bdad"
CORE1_OK = {"uid": 1008, "euid": 1008, "gid": 1009, "egid": 1009, "user": "thilak.arjun", "host": "ai-core-team-1"}
REV_31B = "518276fb130dc81caf9a4f772e65e63ef2526493"
REV_12B = "0e2b1058541244490925fbacf8972041435691ac"
ALIAS_31B = "absent-edges-gemma4-31b-bf16"
ALIAS_12B = "absent-edges-gemma4-12b-bf16"
PUBLIC = "/mnt/kldata/absent-edges-expA-public-20261101"
SEARCH = {"calls": [("source_reference_search", {"query": "pkg/core.py::target"})]}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data if isinstance(data, bytes) else data.encode())
    return path


def model_copy(root, weights):
    """A tiny copy like the public mirror's: model files plus a ``sha256sum *`` manifest next to the directory."""
    model = Path(root) / f"model-{weights}"
    names = list(serve.MODEL_COMMON_FILES) + ["README.md", "processor_config.json"]
    names += (["model.safetensors"] if weights == "single" else
              ["model.safetensors.index.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"])
    lines = []
    for name in sorted(names):
        data = f"{weights} copy: {name}\n".encode()
        write(model / name, data)
        lines.append(f"{sha(data)}  {name}")
    manifest = write(Path(root) / f"COPY-{weights}.sha256", "\n".join(lines) + "\n")
    return model, manifest, sha(manifest.read_bytes())


def additions(golden, actual, path="$"):
    """Key paths present in ``actual`` but not in ``golden`` (dicts only; the golden test checks the rest)."""
    found = []
    if isinstance(golden, dict) and isinstance(actual, dict):
        for key, value in actual.items():
            if key not in golden:
                found.append(f"{path}.{key}")
            else:
                found += additions(golden[key], value, f"{path}.{key}")
    return found


# ---------------------------------------------------------------------------------------------------------------
# (1) a default invocation: exactly these additions to the golden records
# ---------------------------------------------------------------------------------------------------------------

@unittest.skipUnless(POSIX and shutil.which("bash"), "Linux only: serve.py imports pwd")
class DefaultAdditionsTests(unittest.TestCase):
    maxDiff = None

    @classmethod
    def setUpClass(cls):
        cls.values = G.capture()

    def test_serve_plan_adds_only_the_profile_record(self):
        golden = G.GOLDEN["serve"]["plan_launch_script"]
        for name in ("plan_launch_script", "plan_cli_defaults"):
            plan = self.values["serve"][name]["plan"]
            with self.subTest(plan=name):
                self.assertEqual(sorted(additions(golden["plan"], plan)),
                                 ["$.model.hub_id", "$.model.profile", "$.model.required_files", "$.model.weights",
                                  "$.spec.model_profile"])
                self.assertEqual((plan["spec"]["model_profile"], plan["model"]["profile"], plan["model"]["hub_id"],
                                  plan["model"]["weights"], plan["model"]["required_files"]),
                                 ("gemma4-31b", "gemma4-31b", "google/gemma-4-31B-it", "sharded",
                                  list(serve.MODEL_REQUIRED_FILES)))
                # The unit's --spec leaves the default profile out: the unit command is the frozen one.
                self.assertNotIn("model_profile", json.loads(plan["systemd_run_argv"][-1]))
        self.assertEqual(additions(G.GOLDEN["serve"]["api"]["spec"], self.values["serve"]["api"]["spec"]),
                         ["$.model_profile"])
        verify = self.values["serve"]["verify_model"]["calls"][0]
        self.assertEqual(additions(G.GOLDEN["serve"]["verify_model"]["calls"][0], verify), ["$.required_files"])
        self.assertEqual(verify["required_files"], list(serve.MODEL_REQUIRED_FILES))

    def test_runner_records_add_only_the_profile_and_the_dry_run_config(self):
        live = self.values["runner"]["live"]
        self.assertEqual(sorted(additions(G.GOLDEN["runner"]["live"]["run_json"], live["run_json"])),
                         ["$.model.hub_id", "$.model.profile", "$.model.revision"])
        self.assertEqual({key: live["run_json"]["model"][key] for key in ("profile", "hub_id", "revision")},
                         {"profile": "gemma4-31b", "hub_id": "google/gemma-4-31B-it", "revision": REV_31B})
        self.assertEqual(additions(G.GOLDEN["runner"]["dry_run"], self.values["runner"]["dry_run"]),
                         ["$.summary.episode_config"])
        self.assertEqual(self.values["runner"]["dry_run"]["summary"]["episode_config"],
                         G.GOLDEN["runner"]["live"]["run_json"]["episode_config"])

    def test_launch_records_add_only_the_profile_record(self):
        golden, actual = G.GOLDEN["launch"], self.values["launch"]
        self.assertEqual(actual["launch_env"][len(golden["launch_env"]):], "MODEL_PROFILE=gemma4-31b\n")
        self.assertEqual(actual["prepare_stdout"][len(golden["prepare_stdout"]):], "MODEL_PROFILE=gemma4-31b\n")
        self.assertEqual(sorted(additions(golden["serve_plan"], actual["serve_plan"])),
                         ["$.model.hub_id", "$.model.profile", "$.model.required_files", "$.model.weights",
                          "$.spec.model_profile"])


# ---------------------------------------------------------------------------------------------------------------
# (2) profiles; the 12B against a fake copy; (3) required files of single-file and sharded copies
# ---------------------------------------------------------------------------------------------------------------

@unittest.skipUnless(POSIX, "Linux only: serve.py imports pwd")
class ProfileTests(unittest.TestCase):
    def test_the_default_profile_is_the_frozen_31b(self):
        default = serve.MODEL_PROFILES[serve.DEFAULT_MODEL_PROFILE]
        self.assertEqual(serve.DEFAULT_MODEL_PROFILE, "gemma4-31b")
        self.assertEqual((default.alias, default.revision, default.model_dir, default.manifest, default.manifest_sha256,
                          default.required_files),
                         (serve.MODEL_ALIAS, serve.MODEL_REVISION, str(serve.MODEL_DIR), str(serve.MODEL_MANIFEST),
                          serve.MODEL_MANIFEST_SHA256, serve.MODEL_REQUIRED_FILES))
        self.assertEqual((ALIAS_31B, REV_31B, f"{PUBLIC}/models/gemma4-31b-bf16-518276fb"),
                         (default.alias, default.revision, default.model_dir))

    def test_the_12b_profile(self):
        profile = serve.served_model("gemma4-12b")
        self.assertEqual(dataclasses.asdict(profile), {
            "name": "gemma4-12b", "hub_id": "google/gemma-4-12B-it", "revision": REV_12B, "alias": ALIAS_12B,
            "model_dir": f"{PUBLIC}/models/gemma4-12b-bf16-0e2b1058",
            "manifest": f"{PUBLIC}/models/COPY-gemma4-12b-bf16-0e2b1058.sha256",
            "manifest_sha256": "06656cda567f7acc9a35a090632291e14c770ea5d989292910ebe15fb26621d4", "weights": "single",
            "chat_template": None, "chat_template_sha256": None})     # no override: the copy's own template
        self.assertEqual(profile.required_files, ("config.json", "tokenizer.json", "tokenizer_config.json",
                                                  "chat_template.jinja", "generation_config.json", "model.safetensors"))
        spec = serve.ServeSpec.for_profile("gemma4-12b", run="r12", instance="g7", gpu_uuid=GPU7, port=8141)
        self.assertEqual((spec.model_dir, spec.model_manifest, spec.model_manifest_sha256),
                         (profile.model_dir, profile.manifest, profile.manifest_sha256))
        argv = serve.vllm_argv(spec)
        self.assertEqual((argv[argv.index("--model") + 1], argv[argv.index("--served-model-name") + 1]),
                         (profile.model_dir, ALIAS_12B))
        # Only the alias and the weights differ from a 31B instance: same flags, dtype, max_model_len, environment.
        default = serve.ServeSpec(run="r12", instance="g7", gpu_uuid=GPU7, port=8141)
        self.assertEqual([item for item in serve.vllm_argv(default) if item not in (str(serve.MODEL_DIR), ALIAS_31B)],
                         [item for item in argv if item not in (profile.model_dir, ALIAS_12B)])
        self.assertEqual(serve.serve_environment(spec), serve.serve_environment(default))
        plan = serve.plan(spec, identity=CORE1_OK)
        self.assertEqual({key: plan["model"][key] for key in ("profile", "alias", "revision", "path", "manifest",
                                                               "manifest_sha256", "weights", "max_model_len")},
                         {"profile": "gemma4-12b", "alias": ALIAS_12B, "revision": REV_12B, "path": profile.model_dir,
                          "manifest": profile.manifest, "manifest_sha256": profile.manifest_sha256, "weights": "single",
                          "max_model_len": 32768})
        unit_spec = json.loads(plan["systemd_run_argv"][-1])
        self.assertEqual(unit_spec["model_profile"], "gemma4-12b")
        self.assertEqual(serve.ServeSpec(**unit_spec), spec)      # the supervisor rebuilds exactly this spec

    def test_a_spec_never_mixes_profiles(self):
        with self.assertRaisesRegex(ValueError, "names the model files of profile gemma4-31b"):
            serve.ServeSpec(run="r", instance="g7", gpu_uuid=GPU7, port=8141, model_profile="gemma4-12b")
        twelve = serve.served_model("gemma4-12b")
        for mixed in (dict(model_dir=twelve.model_dir), dict(model_manifest=twelve.manifest),
                      dict(model_manifest_sha256=twelve.manifest_sha256)):
            with self.subTest(mixed=mixed), self.assertRaisesRegex(ValueError, "names the model files of profile"):
                serve.ServeSpec(run="r", instance="g7", gpu_uuid=GPU7, port=8141, **mixed)
        with self.assertRaisesRegex(ValueError, "unknown model profile"):
            serve.ServeSpec(run="r", instance="g7", gpu_uuid=GPU7, port=8141, model_profile="gemma4-27b")
        with self.assertRaisesRegex(ValueError, "names the model files of profile gemma4-31b"):
            serve.ServeSpec.for_profile("gemma4-12b", run="r", instance="g7", gpu_uuid=GPU7, port=8141,
                                        model_dir=str(serve.MODEL_DIR))

    def test_the_supervisor_checks_the_profile_alias(self):
        spec = serve.ServeSpec.for_profile("gemma4-12b", run="r12", instance="g7", gpu_uuid=GPU7, port=8141)
        supervisor = serve.Supervisor(spec)

        def registry(alias, root=None):
            return {"data": [{"id": alias, "max_model_len": 32768, "root": root}]}
        self.assertTrue(supervisor._check_registry(registry(ALIAS_12B, spec.model_dir)))
        self.assertFalse(supervisor._check_registry(registry(ALIAS_31B)))
        self.assertFalse(serve.Supervisor(serve.ServeSpec(run="r", instance="g7", gpu_uuid=GPU7, port=8141))
                         ._check_registry(registry(ALIAS_12B)))

    def test_required_files_of_single_file_and_sharded_copies(self):
        common = ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                  "generation_config.json")
        self.assertEqual(serve.required_model_files("sharded"), common + ("model.safetensors.index.json",))
        self.assertEqual(serve.required_model_files("single"), common + ("model.safetensors",))
        self.assertEqual(serve.required_model_files("sharded"), serve.MODEL_REQUIRED_FILES)
        for bad in ("gguf", None, "Single"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                serve.required_model_files(bad)
        with tempfile.TemporaryDirectory() as directory:
            single = model_copy(directory, "single")
            sharded = model_copy(directory, "sharded")
            for (model, manifest, digest), weights, lacks in ((single, "sharded", "model.safetensors.index.json"),
                                                              (sharded, "single", "model.safetensors")):
                with self.subTest(copy=model.name, checked_as=weights):
                    record = serve.verify_model(model, manifest, expected_manifest_sha256=digest,
                                                required_files=serve.required_model_files(weights))
                    self.assertEqual(record["failures"], ["model_manifest_lacks:" + lacks])
                    self.assertEqual(record["required_files"], list(serve.required_model_files(weights)))
            for (model, manifest, digest), weights in ((single, "single"), (sharded, "sharded")):
                with self.subTest(copy=model.name, checked_as=weights):
                    record = serve.verify_model(model, manifest, expected_manifest_sha256=digest,
                                                required_files=serve.required_model_files(weights))
                    self.assertEqual(record["failures"], [])
                    self.assertEqual(record["sha256_verified_files"], len(list(model.iterdir())))
            # The default (31B, sharded) list still applies when no profile is named.
            self.assertEqual(serve.verify_model(single[0], single[1], expected_manifest_sha256=single[2])["failures"],
                             ["model_manifest_lacks:model.safetensors.index.json"])


@unittest.skipUnless(POSIX, "Linux only: serve.py imports pwd")
class TwelveBPreflightTests(unittest.TestCase):
    """The 12B profile verifies a fake single-file copy and refuses a changed file or a wrong manifest pin."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.model, self.manifest, self.digest = model_copy(self.root, "single")
        for name in ("python", "ninja"):
            write(self.root / "env" / "bin" / name, "#!/bin/sh\n").chmod(0o755)
        self.spec = serve.ServeSpec.for_profile("gemma4-12b", run="t12", instance="g7", gpu_uuid=GPU7, port=8201,
                                                serve_env=str(self.root / "env"), model_dir=str(self.model),
                                                model_manifest=str(self.manifest), model_manifest_sha256=self.digest)

    def tearDown(self):
        self.temporary.cleanup()

    def preflight(self, spec=None):
        return serve.preflight(spec or self.spec, observe=lambda uuid: {"uuid": uuid, "index": 7, "minor_number": 7,
                                                                        "used_mib": 0, "total_mib": 97887, "pids": []},
                               port_free=lambda port: True, magic=lambda path: None, identity=CORE1_OK,
                               mounts=[("/", "ext4", "/dev/sda1"), ("/dev/shm", "tmpfs", "tmpfs")],
                               proc_options=lambda: {"options": ["rw"]})

    def test_the_fake_12b_copy_passes_with_the_12b_required_files(self):
        result = self.preflight()
        self.assertTrue(result["passed"], result["failures"])
        self.assertEqual(result["model"]["required_files"], list(serve.required_model_files("single")))
        self.assertEqual(result["model"]["sha256_verified_files"], 8)
        self.assertEqual(result["spec"]["model_profile"], "gemma4-12b")

    def test_a_changed_file_or_a_wrong_manifest_pin_is_refused(self):
        refused = self.preflight(dataclasses.replace(self.spec, model_manifest_sha256="0" * 64))
        self.assertEqual(refused["failures"], ["model_manifest_sha256_mismatch"])
        write(self.model / "model.safetensors", "tampered weights\n")
        self.assertEqual(self.preflight()["failures"], ["model_file_sha256_mismatch:model.safetensors"])

    def test_the_cli_uses_the_12b_pin_and_required_files(self):
        cli = G.load_script("expA_serve")
        argv = ["verify-model", "--model-profile", "gemma4-12b", "--model-dir", str(self.model),
                "--model-manifest", str(self.manifest)]
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(cli.main(argv), 2)       # the fake manifest is not the pinned 12B manifest
        record = json.loads(out.getvalue())
        self.assertEqual((record["failures"], record["expected_manifest_sha256"]),
                         (["model_manifest_sha256_mismatch"], serve.served_model("gemma4-12b").manifest_sha256))
        pinned = dataclasses.replace(serve.served_model("gemma4-12b"), manifest_sha256=self.digest)
        with mock.patch.dict(serve.MODEL_PROFILES, {"gemma4-12b": pinned}), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(cli.main(argv), 0)
        self.assertEqual(json.loads(out.getvalue())["required_files"], list(serve.required_model_files("single")))
        with mock.patch.dict(serve.MODEL_PROFILES, {"gemma4-12b": pinned}), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(cli.main(argv[:3] + ["--model-dir", str(self.model)]), 2)
        self.assertEqual(json.loads(out.getvalue())["manifest"], pinned.manifest)   # the profile's own manifest
        with contextlib.redirect_stdout(io.StringIO()) as out, \
                mock.patch.object(serve, "current_identity", return_value=CORE1_OK):
            self.assertEqual(cli.main(["plan", "--run", "t12", "--instance", "g7", "--gpu-uuid", GPU7, "--port", "8201",
                                       "--model-profile", "gemma4-12b"]), 0)
        plan = json.loads(out.getvalue())
        self.assertEqual((plan["spec"]["model_dir"], plan["spec"]["model_manifest_sha256"], plan["model"]["alias"]),
                         (serve.served_model("gemma4-12b").model_dir,
                          serve.served_model("gemma4-12b").manifest_sha256, ALIAS_12B))


# ---------------------------------------------------------------------------------------------------------------
# (4) the runner: budget and profile options, recorded in run.json and applied to every episode
# ---------------------------------------------------------------------------------------------------------------

@unittest.skipUnless(POSIX, "the runner reads checkouts with POSIX no-follow reads")
class RunnerOptionTests(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.runner, self.dispatch, self.manifest = G.runner_fixture(self.base)
        self.servers = []

    def tearDown(self):
        for server in self.servers:
            with contextlib.suppress(Exception):
                server.close()
        for path in self.base.rglob("*"):
            if path.is_dir() and not path.is_symlink():
                path.chmod(0o755)
        self.temporary.cleanup()

    def server(self, **kwargs):
        server = H.FakeServer(**kwargs)
        self.servers.append(server)
        return server

    def parse(self, *extra):
        with contextlib.redirect_stderr(io.StringIO()):
            return self.runner.parse_args(["--dispatch", str(self.dispatch), "--dry-run", *extra])

    def test_defaults_are_the_prereg_values(self):
        args = self.parse()
        self.assertEqual((args.model_profile, args.model, args.max_model_calls, args.context_budget),
                         ("gemma4-31b", ALIAS_31B, 8, 16384))
        self.assertEqual(args.episode_config, harness.EpisodeConfig())

    def test_budget_and_profile_options_are_validated(self):
        args = self.parse("--max-model-calls", "16", "--context-budget", "32768", "--model-profile", "gemma4-12b")
        self.assertEqual((args.episode_config.max_model_calls, args.episode_config.context_budget, args.model),
                         (16, 32768, ALIAS_12B))
        self.assertEqual(dataclasses.replace(args.episode_config, max_model_calls=8, context_budget=16384),
                         harness.EpisodeConfig())   # nothing else changes
        for bad in (["--max-model-calls", "0"], ["--max-model-calls", "-3"], ["--max-model-calls", "x"],
                    ["--context-budget", "32769"], ["--context-budget", "0"], ["--context-budget", "16k"],
                    ["--model-profile", "gemma4-27b"], ["--model", ALIAS_12B],
                    ["--model-profile", "gemma4-12b", "--model", ALIAS_31B],
                    ["--context-budget", "3276"], ["--context-budget", "100"], ["--max-model-calls", "1"],
                    ["--max-model-calls", "16"], ["--max-model-calls", "16", "--context-budget", "16384"]):
            with self.subTest(bad=bad), self.assertRaises(SystemExit):
                self.parse(*bad)
        self.assertEqual(self.parse("--model", ALIAS_31B).model, ALIAS_31B)

    def test_dry_run_renders_the_budget_in_the_system_prompt(self):
        for extra, calls, budget in (((), 8, 16384), (("--max-model-calls", "16", "--context-budget", "32768"), 16,
                                                        32768)):
            with self.subTest(extra=extra):
                code, output = G.invoke_runner(self.runner, ["--dispatch", str(self.dispatch), "--dry-run", "--show",
                                                             "2", *extra])
                summary = json.loads(output)
                self.assertEqual((code, summary["episode_config"]["max_model_calls"],
                                  summary["episode_config"]["context_budget"]), (0, calls, budget))
                for example in summary["examples"].values():
                    self.assertIn(f"You have at most {calls} responses; call submit_answer before they run out.",
                                  example["system"])

    def live(self, server, *extra, run="run"):
        argv = G.live_args(self.base, self.dispatch, self.manifest, server, run=run) + list(extra)
        return G.live_episode(self.runner, argv, server, self.base / run)

    def test_sixteen_responses_and_a_32768_context_reach_every_episode(self):
        server = self.server(default=SEARCH)                    # never submits: the budget ends the episode
        live = self.live(server, "--max-model-calls", "16", "--context-budget", "32768")
        expected = dict(harness.EpisodeConfig().record(), max_model_calls=16, context_budget=32768)
        self.assertEqual((live["exit_code"], live["results"]), (0, 1))
        self.assertEqual(live["run_json"]["episode_config"], expected)
        self.assertEqual(live["result"]["config"], expected)
        self.assertEqual(live["setup"]["config"], expected)
        self.assertIn("You have at most 16 responses; call submit_answer before they run out.", live["setup"]["system"])
        self.assertEqual(live["result"]["exit_status"], "NoAnswer")
        result = json.loads(next((self.base / "run" / "cells").rglob("result.json")).read_bytes())
        self.assertEqual((result["calls"], result["accounting"]["attempted_model_calls"]), (16, 16))
        self.assertEqual(len(server.chat_requests()), 16)
        # max_tokens = min(4096, 32768 - prompt tokens): the 32,768 budget is what each request is sized against.
        attempt = next((self.base / "run" / "cells").rglob("result.json")).parent
        tokens = [event["value"]["count"] for event in H.events(attempt) if event["kind"] == "context_accounting"]
        self.assertEqual([request["max_tokens"] for request in server.chat_requests()],
                         [min(4096, 32768 - count) for count in tokens])
        # The default run (8 responses) of the same cell for comparison.
        server8 = self.server(default=SEARCH)
        default = self.live(server8, run="run8")
        self.assertEqual((default["result"]["exit_status"], default["result"]["config"]["max_model_calls"],
                          len(server8.chat_requests())), ("NoAnswer", 8, 8))
        self.assertIn("You have at most 8 responses;", default["setup"]["system"])

    def test_each_option_changes_only_its_own_run_json_field(self):
        runs = {}
        for name, extra, alias in (("default", (), ALIAS_31B),
                                   ("explicit", ("--model-profile", "gemma4-31b", "--model", ALIAS_31B,
                                                 "--max-model-calls", "8", "--context-budget", "16384"), ALIAS_31B),
                                   ("budget16", ("--max-model-calls", "16", "--context-budget", "32768"), ALIAS_31B),
                                   ("twelve", ("--model-profile", "gemma4-12b"), ALIAS_12B)):
            server = self.server(default=H.SUBMIT)
            server.model = alias
            self.assertEqual(self.live(server, "--run-id", "same", *extra, run=name)["exit_code"], 0)
            runs[name] = (self.base / name / "run.json").read_bytes()
        self.assertEqual(runs["explicit"], runs["default"])      # the defaults are exactly the prereg values
        default = json.loads(runs["default"])
        # Top-level keys of the frozen runner's run.json, unchanged; model gains profile, hub_id and revision.
        self.assertEqual(sorted(default), sorted([
            "protocol", "run_id", "dispatch", "dispatch_sha256", "items_sha256", "items_manifest_sha256",
            "payload_store", "graph_facts", "family_sources", "item_build_code", "design", "families", "checkouts",
            "episode_config", "infra_attempt_cap", "prompts", "tools", "model", "code", "schema", "cells_sha256",
            "cell_count"]))
        self.assertEqual(sorted(default["model"]), ["alias", "enable_thinking", "hub_id", "max_context", "profile",
                                                     "revision", "temperature"])
        for name, field in (("budget16", "episode_config"), ("twelve", "model")):
            with self.subTest(run=name):
                other = json.loads(runs[name])
                self.assertNotEqual(other[field], default[field])
                self.assertEqual({key: value for key, value in other.items() if key != field},
                                 {key: value for key, value in default.items() if key != field})

    def test_a_relaunch_with_another_budget_is_refused(self):
        server = self.server(default=H.SUBMIT)
        self.assertEqual(self.live(server)["exit_code"], 0)
        chats = len(server.chat_requests())
        argv = G.live_args(self.base, self.dispatch, self.manifest, server) + ["--max-model-calls", "16",
                                                                               "--context-budget", "32768"]
        with mock.patch.object(G.tools, "ReadOnlyBash", G.FakeJailBash), \
                self.assertRaisesRegex(L.LedgerInconsistency, "run.json already exists with different content"):
            G.invoke_runner(self.runner, argv)
        self.assertEqual(len(server.chat_requests()), chats)

    def test_the_12b_profile_records_its_alias_and_revision(self):
        server = self.server(default=H.SUBMIT)
        server.model = ALIAS_12B
        live = self.live(server, "--model-profile", "gemma4-12b")
        self.assertEqual((live["exit_code"], live["result"]["exit_status"]), (0, "Submitted"))
        self.assertEqual(live["run_json"]["model"], {"alias": ALIAS_12B, "max_context": 32768, "enable_thinking": False,
                                                     "temperature": 0.2, "profile": "gemma4-12b",
                                                     "hub_id": "google/gemma-4-12B-it", "revision": REV_12B})
        expected_profile = {"endpoint": "<fake server>", "model": ALIAS_12B, "revision": REV_12B,
                            "max_context": 32768, "enable_thinking": False}
        self.assertEqual(live["result"]["model_profile"], expected_profile)
        self.assertEqual(live["setup"]["model_profile"], expected_profile)
        self.assertEqual(live["chat_request"]["model"], ALIAS_12B)
        self.assertEqual(live["run_json"]["episode_config"], harness.EpisodeConfig().record())
        # An endpoint that serves the 31B alias is refused before anything is dispatched.
        wrong = self.server(default=H.SUBMIT)
        with self.assertRaisesRegex(RuntimeError, f"does not serve exactly {ALIAS_12B}"):
            self.live(wrong, "--model-profile", "gemma4-12b", run="run-wrong")
        self.assertEqual(wrong.chat_requests(), [])


# ---------------------------------------------------------------------------------------------------------------
# (5) the launch script: prepare -> launch.env -> serve plan -> runner unit (stubbed host, nothing is started)
# ---------------------------------------------------------------------------------------------------------------

@unittest.skipUnless(POSIX and shutil.which("bash"), "Linux only: the launch script runs on Core1")
class LaunchScriptTests(unittest.TestCase):
    maxDiff = None
    BUDGET16 = ["--max-model-calls", "16", "--context-budget", "32768"]

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.host = G.LaunchHost(self.temporary.name)
        self.control = self.host.run_root("golden1") / "control"

    def tearDown(self):
        self.temporary.cleanup()

    def prepare(self, *extra, check=True):
        return self.host.run(*G.PREPARE_ARGV, "--runtime-python", str(self.host.python), *extra, check=check)

    def serve_units(self):
        self.host.run("serve", "--run", "golden1")
        return [self.host.normalize(call) for call in G.units(self.host.sudo_calls(), "serve")]

    def runner_units(self, *extra, check=True):
        result = self.host.run("runner", "--run", "golden1", "--blocks", "P", "--seeds", "11", *extra, check=check)
        return result, [self.host.normalize(call) for call in G.units(self.host.sudo_calls(), "runner")]

    def endpoint_ready_log(self):
        path = Path(self.temporary.name) / "endpoint_ready.log"
        text = path.read_text() if path.exists() else ""
        path.unlink(missing_ok=True)
        return text

    def runner_args(self, unit):
        """The expA_run.py arguments of a runner unit, parsed by the runner itself."""
        tail = unit[next(i for i, value in enumerate(unit) if value.endswith("/scripts/expA_run.py")) + 1:]
        with contextlib.redirect_stderr(io.StringIO()):
            return G.load_script("expA_run").parse_args(tail)

    def test_syntax(self):
        self.assertEqual(G.subprocess.run(["bash", "-n", str(G.PROJECT / "scripts" / "expA_core1_launch.sh")],
                                          timeout=60).returncode, 0)

    def test_a_12b_run_flows_from_prepare_to_the_runner_unit(self):
        prepared = self.prepare("--model-profile", "gemma4-12b")
        launch_env = (self.control / "launch.env").read_text()
        self.assertTrue(launch_env.endswith("MIRROR_OWNER_UID=\nMODEL_PROFILE=gemma4-12b\n"), launch_env)
        self.assertEqual(prepared.stdout, launch_env)
        self.host.stage("golden1")
        serve_units = self.serve_units()
        plan = json.loads((self.control / "serve-g7.plan.json").read_text())
        twelve = serve.served_model("gemma4-12b")
        self.assertEqual((plan["model"]["profile"], plan["model"]["alias"], plan["model"]["revision"],
                          plan["model"]["path"], plan["model"]["manifest_sha256"], plan["spec"]["model_profile"]),
                         ("gemma4-12b", ALIAS_12B, REV_12B, twelve.model_dir, twelve.manifest_sha256, "gemma4-12b"))
        self.assertEqual(plan["vllm_argv"][plan["vllm_argv"].index("--served-model-name") + 1], ALIAS_12B)
        # The serve unit differs from the default one only in its --spec, which names the 12B profile and copy.
        golden_serve = G.GOLDEN["launch"]["serve_units"]
        self.assertEqual(len(serve_units), 1)
        self.assertEqual(serve_units[0][:-1], golden_serve[0][:-1])
        unit_spec = json.loads(serve_units[0][-1])
        self.assertEqual({key: value for key, value in unit_spec.items()
                          if key not in ("model_profile", "model_dir", "model_manifest", "model_manifest_sha256")},
                         {key: value for key, value in json.loads(golden_serve[0][-1]).items()
                          if key not in ("model_dir", "model_manifest", "model_manifest_sha256")})
        self.assertEqual((unit_spec["model_profile"], unit_spec["model_dir"], unit_spec["model_manifest"],
                          unit_spec["model_manifest_sha256"]),
                         ("gemma4-12b", twelve.model_dir, twelve.manifest, twelve.manifest_sha256))
        result, runner_units = self.runner_units(*self.BUDGET16)
        self.assertEqual(self.endpoint_ready_log(), f"8141 {ALIAS_12B}\n")
        self.assertIn("model gemma4-12b (absent-edges-gemma4-12b-bf16); budgets --max-model-calls 16 "
                      "--context-budget 32768", result.stderr)
        # The runner unit is the default one plus the profile and budget options.
        self.assertEqual(runner_units, [G.GOLDEN["launch"]["runner_units"][0]
                                        + ["--model-profile", "gemma4-12b"] + self.BUDGET16])
        args = self.runner_args(runner_units[0])
        self.assertEqual((args.model_profile, args.model, args.episode_config.max_model_calls,
                          args.episode_config.context_budget), ("gemma4-12b", ALIAS_12B, 16, 32768))

    def test_a_31b_run_with_16_responses_and_a_32768_context(self):
        self.prepare()
        self.assertTrue((self.control / "launch.env").read_text().endswith("MODEL_PROFILE=gemma4-31b\n"))
        self.host.stage("golden1")
        self.assertEqual(self.serve_units(), G.GOLDEN["launch"]["serve_units"])
        _, runner_units = self.runner_units(*self.BUDGET16)
        self.assertEqual(self.endpoint_ready_log(), f"8141 {ALIAS_31B}\n")
        self.assertEqual(runner_units, [G.GOLDEN["launch"]["runner_units"][0] + self.BUDGET16])
        args = self.runner_args(runner_units[0])
        self.assertEqual((args.model_profile, args.model, args.episode_config),
                         ("gemma4-31b", ALIAS_31B, harness.EpisodeConfig(max_model_calls=16, context_budget=32768)))
        _, default_units = self.runner_units()
        self.assertEqual(default_units, G.GOLDEN["launch"]["runner_units"])     # a relaunch without options

    def test_bad_profiles_and_budgets_are_refused_before_any_unit(self):
        refused = self.prepare("--model-profile", "gemma4-27b", check=False)
        self.assertEqual(refused.returncode, 2)
        self.assertIn("--model-profile must be gemma4-31b-qat or gemma4-31b or gemma4-12b, not gemma4-27b",
                      refused.stderr)
        self.assertFalse((self.control / "launch.env").exists())
        self.prepare()
        self.host.stage("golden1")
        for extra, message in ((["--max-model-calls", "0"], "--max-model-calls must be a positive integer"),
                               (["--max-model-calls", "x"], "--max-model-calls must be a positive integer"),
                               (["--context-budget", "40000"], "--context-budget must be at most 32768"),
                               (["--context-budget", "-1"], "--context-budget must be a positive integer"),
                               (["--context-budget", "16k"], "--context-budget must be a positive integer"),
                               (["--max-model-calls", "16"], "only the registered 16/32768 design"),
                               (["--context-budget", "3276"], "only the registered 16/32768 design"),
                               (["--max-model-calls", "16", "--context-budget", "16384"],
                                "only the registered 16/32768 design")):
            with self.subTest(extra=extra):
                result, units = self.runner_units(*extra, check=False)
                self.assertEqual((result.returncode, units), (2, []))
                self.assertIn(message, result.stderr)

    def test_a_launch_env_from_before_the_option_is_the_31b(self):
        self.prepare("--model-profile", "gemma4-12b")
        self.host.stage("golden1")
        path = self.control / "launch.env"
        path.chmod(0o600)
        path.write_text(path.read_text().replace("MODEL_PROFILE=gemma4-12b\n", ""))
        self.assertEqual(self.host.run("show-config", "--run", "golden1").stdout, G.GOLDEN["launch"]["launch_env"]
                         .replace("<T>", self.temporary.name))
        _, units = self.runner_units()
        self.assertEqual((units, self.endpoint_ready_log()),
                         (G.GOLDEN["launch"]["runner_units"], f"8141 {ALIAS_31B}\n"))
        path.write_text(path.read_text() + "MODEL_PROFILE=gemma4-27b\n")
        result, units = self.runner_units(check=False)
        self.assertEqual((result.returncode, units), (2, []))
        self.assertIn("unknown model profile gemma4-27b", result.stderr)

    def test_add_instance_keeps_the_profile(self):
        self.prepare("--model-profile", "gemma4-12b")
        gpu3 = "GPU-15870ba7-ca88-14c1-d031-0ee0aa0a0a0a"
        self.host.run("add-instance", "--run", "golden1", "--instance", f"g3={gpu3}=8143")
        text = (self.control / "launch.env").read_text()
        self.assertIn(f"\nINSTANCES=g7={GPU7}=8141 g3={gpu3}=8143\n", text)
        self.assertTrue(text.endswith("MODEL_PROFILE=gemma4-12b\n"))

    def test_the_script_alias_table_is_the_serve_profiles(self):
        text = (G.PROJECT / "scripts" / "expA_core1_launch.sh").read_text()
        body = text[text.index("model_alias() {"):]
        body = body[:body.index("\n}\n")]
        table = dict(G.re.findall(r"^\s+([a-z0-9-]+)\) printf '%s' ([a-z0-9-]+) ;;$", body, G.re.MULTILINE))
        self.assertEqual(table, {name: profile.alias for name, profile in serve.MODEL_PROFILES.items()})
        self.assertIn("MODEL_PROFILE_DEFAULT=" + serve.DEFAULT_MODEL_PROFILE + "\n", text)
        self.assertIn("--model-profile must be " + " or ".join(sorted(serve.MODEL_PROFILES, reverse=True)), text)

    def test_serve_refuses_a_plan_of_another_profile(self):
        # If the script's alias table and serve.py ever disagree, serve stops before starting any unit.
        text = self.host.script.read_text()
        self.host.script.write_text(text.replace("gemma4-12b) printf '%s' absent-edges-gemma4-12b-bf16",
                                                 "gemma4-12b) printf '%s' absent-edges-gemma4-31b-bf16"))
        self.prepare("--model-profile", "gemma4-12b")
        self.host.stage("golden1")
        result = self.host.run("serve", "--run", "golden1", check=False)
        self.assertEqual(result.returncode, 2)
        self.assertIn("the serve plan of g7 is not model profile gemma4-12b (absent-edges-gemma4-31b-bf16)",
                      result.stderr)
        self.assertEqual(G.units(self.host.sudo_calls(), "serve"), [])


if __name__ == "__main__":
    unittest.main()
