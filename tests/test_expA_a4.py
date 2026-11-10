"""Exp A follow-up A4: final-message components, 2 x 2 (protocols/EXPA_A4_DIRECTIVE.md). SYNTHETIC fixtures only.

(1) The texts and config records (any OS): the two new policies are appended after the A1 ones; their records name the
    A4 protocol; the default, A1 and A3 records are byte-identical to the ones before A4 (sha256 pinned from the
    previous commit).
(2) The harness against the fake server (POSIX): each new policy changes only the 8th request, by its suffix.
(3) The runner (POSIX): registration of the new policies; block P at seed 11 or 23 (one seed per launch), optionally
    --conditions from C0,C1; --conditions filters the dispatched cells, is recorded in the launch record only when
    given, and run.json is unchanged by it; status reports the selected conditions only for such runs.
(4) The A1 dev check with --seed 23 --conditions C0,C1 and the new policies (POSIX: cell ids are directory names).
(5) The prespecified analysis (scripts/expA_a4_analysis.py) on a SYNTHETIC 2-family fixture with a hand-computed
    primary, the secondary contrasts, refusals and the run.json/launch verification (any OS).
ANALYSIS RECORD.
"""
import collections
import contextlib
from fractions import Fraction
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

from absent_edges.probe import harness  # noqa: E402
from expA_reanalyze_independent import t_statistics  # noqa: E402

POSIX = os.name == "posix"
if POSIX:
    import test_expA_default_golden as G      # puts the project on sys.path
    import test_probe_harness as H
    from absent_edges.models import LocalModel, ModelProfile
    from absent_edges.probe import ledger as L
    from absent_edges.probe.tools import ToolBox

_SPEC = importlib.util.spec_from_file_location("expA_a4_analysis", PROJECT / "scripts" / "expA_a4_analysis.py")
A = importlib.util.module_from_spec(_SPEC)
sys.modules["expA_a4_analysis"] = A
_SPEC.loader.exec_module(A)

PREFIX = "\n\n[harness notice] "
A1_TEXTS = {"neutral": "Your previous tool call was processed normally.",
            "deadline": "This is your last response. Call submit_answer now."}
A4_TEXTS = {"deadline_info": "This is your last response.", "submit_only": "Call submit_answer now."}
FROZEN_CONFIG = {"max_model_calls": 8, "context_budget": 16384, "max_response_tokens": 4096,
                 "min_response_tokens": 512, "temperature": 0.2, "max_consecutive_format_errors": 3}
# sha256 of json.dumps(record) (key order kept) as the commit before A4 (ee074a2b6) produced them.
PINNED = {"none": "ff8921f6e5ef35bfbacc58037ac23081468f43a9404ea8a62ed174f1ffd9df4b",
          "neutral": "2aab8a084aedf3c705a68e9600fd61aeccc68364e9aa2a07a93fe989ca9b82a4",
          "deadline": "773cb89af25df5d771e6248f7d0d45246f35045c7a803a2a8f1a9a956bb4819b",
          "a3_off": "b1d564984b0012b8d7cff6649131da96375d38577825a6ccdd64b9a9154c00a1",
          "a3_on": "c4ae1e0f20a92e0c49b574b9826e3f541fbe735823a45ef37b2240f56cfa1ef8",
          "design_default": "6b0a49cd8eb8e6157e2d8d516f0ca2231e92b4e494c178b1c28af838ebdedb5c",
          "design_a3": "723f7aa95dd4494c2b174ac09ddc7aa353a52dcbe48297f47354ac8a76c44d0c",
          "prompts": "ff8eedeb4d7fb506bc4642c514160edd454ac8b38742e2f48196f3651dc203c3"}
BASH = {"calls": [("bash", {"command": "ls"})]}
SEARCH = {"calls": [("source_reference_search", {"query": "pkg/core.py::target"})]}


def sha(text):
    return hashlib.sha256(text if isinstance(text, bytes) else text.encode("utf-8")).hexdigest()


def digest(value):
    return sha(json.dumps(value, ensure_ascii=False))


# ---------------------------------------------------------------------------------------------------------------
# (1) texts and records
# ---------------------------------------------------------------------------------------------------------------

class A4TextTests(unittest.TestCase):
    maxDiff = None

    def test_the_policies_are_appended_with_their_exact_texts(self):
        self.assertEqual(list(harness.FINAL_TURN_TEXTS.items()), list(A1_TEXTS.items()) + list(A4_TEXTS.items()))
        self.assertEqual(harness.FINAL_TURN_POLICIES, ("none", "neutral", "deadline", "deadline_info", "submit_only"))
        self.assertEqual(harness.FINAL_TURN_TEXTS["deadline"],
                         harness.FINAL_TURN_TEXTS["deadline_info"] + " " + harness.FINAL_TURN_TEXTS["submit_only"])
        for policy, text in A4_TEXTS.items():
            self.assertEqual(harness.final_turn_suffix(policy), PREFIX + text)
        self.assertEqual(harness.FINAL_TURN_PROTOCOLS, {
            "neutral": "protocols/EXPA_FINALTURN.md", "deadline": "protocols/EXPA_FINALTURN.md",
            "deadline_info": "protocols/EXPA_A4_DIRECTIVE.md", "submit_only": "protocols/EXPA_A4_DIRECTIVE.md"})
        # Not prompt templates: run.json's prompts stay those of the prereg harness.
        self.assertEqual(digest(harness.prompt_templates()), PINNED["prompts"])

    def test_earlier_records_are_byte_identical(self):
        configs = {"none": harness.EpisodeConfig(), "neutral": harness.EpisodeConfig(final_turn_policy="neutral"),
                   "deadline": harness.EpisodeConfig(final_turn_policy="deadline"),
                   "a3_off": harness.EpisodeConfig(final_turn_policy="deadline", evidence_ledger="off"),
                   "a3_on": harness.EpisodeConfig(final_turn_policy="deadline", evidence_ledger="on")}
        for name, config in configs.items():
            with self.subTest(name=name):
                self.assertEqual(digest(config.record()), PINNED[name])
        self.assertEqual(digest(harness.design_table_for(None)), PINNED["design_default"])
        self.assertEqual(digest(harness.design_table_for("a3-ledger")), PINNED["design_a3"])
        self.assertEqual(digest(harness.EpisodeConfig(final_turn_policy="neutral").record()), PINNED["neutral"])

    def test_the_new_records_name_the_a4_protocol(self):
        for policy, text in A4_TEXTS.items():
            with self.subTest(policy=policy):
                record = harness.EpisodeConfig(final_turn_policy=policy).record()
                self.assertEqual(list(record), list(FROZEN_CONFIG) + ["final_turn_policy", "final_turn_notice"])
                notice = dict(record["final_turn_notice"])
                self.assertTrue(notice.pop("appended_to").startswith("the content of the last message before"))
                self.assertEqual(notice, {"protocol": "protocols/EXPA_A4_DIRECTIVE.md", "channel": "in_band",
                                          "prefix": PREFIX, "text": text, "text_sha256": sha(text),
                                          "suffix_sha256": sha(PREFIX + text), "call_index": 7})
                self.assertIsNone(harness.config_design(harness.EpisodeConfig(final_turn_policy=policy)))
        for bad in ("deadline-info", "Submit_only", "info", "submit"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                harness.EpisodeConfig(final_turn_policy=bad)


# ---------------------------------------------------------------------------------------------------------------
# (2) the harness
# ---------------------------------------------------------------------------------------------------------------

@unittest.skipUnless(POSIX, "the ToolBox search scans with POSIX no-follow reads")
class A4EpisodeTests(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = H.make_repo(Path(self.temporary.name) / "repo")
        self.servers, self.count = [], 0

    def tearDown(self):
        for server in self.servers:
            with contextlib.suppress(Exception):
                server.close()
        self.temporary.cleanup()

    def run_policy(self, script, policy):
        server = H.FakeServer(script)
        self.servers.append(server)
        self.count += 1
        directory = Path(self.temporary.name) / f"attempt-{self.count}"
        directory.mkdir()
        client = LocalModel(ModelProfile(endpoint=server.endpoint), timeout=20)
        toolbox = ToolBox(self.root, ["pkg"], H.FakeBash())
        result, _ = harness.run_episode(H.make_episode("C1"), directory, client=client, toolbox=toolbox,
                                        config=harness.EpisodeConfig(final_turn_policy=policy))
        return result, server, H.events(directory)

    def test_only_the_8th_request_changes_by_the_suffix(self):
        none, none_server, _ = self.run_policy([BASH] * 8, "none")
        base = none_server.chat_requests()
        for policy, text in A4_TEXTS.items():
            with self.subTest(policy=policy):
                result, server, rows = self.run_policy([BASH] * 8, policy)
                chats = server.chat_requests()
                self.assertEqual(chats[:7], base[:7])
                self.assertEqual(chats[7]["messages"][:-1], base[7]["messages"][:-1])
                self.assertEqual(chats[7]["messages"][-1]["content"], base[7]["messages"][-1]["content"] + PREFIX + text)
                self.assertEqual(json.dumps(chats).count("[harness notice]"), 1)
                notice = result["exposure"]["final_turn_notice"]
                self.assertEqual((notice["policy"], notice["text_sha256"], notice["suffix_sha256"], notice["delivered"],
                                  notice["appended_to"]), (policy, sha(text), sha(PREFIX + text), True, "tool_result"))
                self.assertEqual(result["accounting"]["final_turn_notice"], notice)
                self.assertEqual(result["config"], harness.EpisodeConfig(final_turn_policy=policy).record())
                event = [row["value"] for row in rows if row["kind"] == "final_turn_notice"]
                self.assertEqual((len(event), event[0]["suffix"], event[0]["policy"]), (1, PREFIX + text, policy))

    def test_an_early_submission_never_sees_it(self):
        for policy in A4_TEXTS:
            with self.subTest(policy=policy):
                result, server, _ = self.run_policy([BASH, H.SUBMIT], policy)
                self.assertEqual(result["exit_status"], "Submitted")
                self.assertNotIn("[harness notice]", json.dumps(server.chat_requests()))


# ---------------------------------------------------------------------------------------------------------------
# (3) the runner
# ---------------------------------------------------------------------------------------------------------------

@unittest.skipUnless(POSIX, "the runner reads checkouts with POSIX no-follow reads")
class A4RunnerTests(unittest.TestCase):
    maxDiff = None
    POLICIES = ("neutral", "deadline", "deadline_info", "submit_only")

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

    def refused(self, *extra):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
            self.runner.parse_args(["--dispatch", str(self.dispatch), "--dry-run", *extra])
        self.assertEqual(raised.exception.code, 2)
        return stderr.getvalue()

    def test_registration_and_the_seed_23_guard(self):
        self.assertEqual(self.runner.REGISTERED_DESIGNS[(8, 16384, "deadline_info")],
                         ("protocols/EXPA_A4_DIRECTIVE.md", ("gemma4-31b",)))
        self.assertEqual(self.runner.REGISTERED_DESIGNS[(8, 16384, "submit_only")],
                         ("protocols/EXPA_A4_DIRECTIVE.md", ("gemma4-31b",)))
        for policy in self.POLICIES:
            protocol = harness.FINAL_TURN_PROTOCOLS[policy]
            expected = harness.EpisodeConfig(final_turn_policy=policy)
            for selection, conditions in ((("--blocks", "P", "--seeds", "11"), None),
                                          (("--blocks", "P", "--seeds", "23"), None),
                                          (("--blocks", "P", "--seeds", "23", "--conditions", "C0,C1"), {"C0", "C1"}),
                                          (("--blocks", "P", "--seeds", "11", "--conditions", "C1,C0"), {"C0", "C1"}),
                                          (("--blocks", "P", "--seeds", "23", "--conditions", "C1"), {"C1"})):
                with self.subTest(policy=policy, selection=selection):
                    args = self.parse("--final-turn-policy", policy, *selection)
                    self.assertEqual((args.episode_config, args.blocks, args.conditions, args.design),
                                     (expected, {"P"}, conditions, None))
            message = (f"--final-turn-policy {policy} ({protocol}) runs block P at seed 11 or seed 23 only, one seed "
                       "per launch")
            for bad in ((), ("--blocks", "P"), ("--seeds", "23"), ("--blocks", "P", "--seeds", "47"),
                        ("--blocks", "P", "--seeds", "11,23"), ("--blocks", "P,S-VC", "--seeds", "23"),
                        ("--blocks", "P", "--seeds", "23", "--conditions", "C0,C2"),
                        ("--blocks", "P", "--seeds", "23", "--conditions", "C4a")):
                with self.subTest(policy=policy, bad=bad):
                    self.assertIn(message, self.refused("--final-turn-policy", policy, *bad))
            for bad in (["--max-model-calls", "16", "--context-budget", "32768"], ["--model-profile", "gemma4-12b"]):
                with self.subTest(policy=policy, bad=bad):
                    self.assertIn(f"--final-turn-policy {policy} ({protocol}) is registered only with the prereg budget",
                                  self.refused("--final-turn-policy", policy, "--blocks", "P", "--seeds", "23", *bad))
        self.assertIn("unknown conditions", self.refused("--conditions", "C9"))
        self.assertIn("unknown conditions", self.refused("--conditions", ","))
        # Without a policy --conditions is a plain filter, like --seeds and --families.
        args = self.parse("--conditions", "C3")
        self.assertEqual((args.conditions, args.episode_config), ({"C3"}, harness.EpisodeConfig()))
        self.assertIsNone(self.parse().conditions)

    def test_select_cells_filters_conditions(self):
        argv = ["--dispatch", str(self.dispatch), "--dry-run", "--blocks", "P", "--seeds", "23"]
        code, output = G.invoke_runner(self.runner, argv + ["--final-turn-policy", "submit_only"])
        everything = json.loads(output)
        self.assertEqual((code, everything["selected_cells"]), (0, 10))
        code, output = G.invoke_runner(self.runner, argv + ["--final-turn-policy", "submit_only", "--conditions",
                                                            "C0,C1"])
        summary = json.loads(output)
        self.assertEqual((code, summary["selected_cells"], summary["by_condition"]),
                         (0, 4, {"P:none:C0": 2, "P:omit_node:C1": 2}))
        self.assertTrue(all(cell.endswith("|s23") for cell in summary["first_cells"]))
        self.assertEqual(summary["design_cells"], everything["design_cells"])
        self.assertEqual(summary["episode_config"], harness.EpisodeConfig(final_turn_policy="submit_only").record())
        cells = [{"block": "P", "seed": 23, "family": "f", "condition": c} for c in ("C0", "C1", "C2")]
        self.assertEqual(self.runner.select_cells(cells, conditions={"C1"}), [cells[1]])
        self.assertEqual(self.runner.select_cells(cells), cells)

    def test_launch_records_and_status(self):
        launches = {}
        for name, extra in (("plain", ()), ("filtered", ("--conditions", "C0,C1"))):
            server = self.server(default=H.SUBMIT)
            argv = G.live_args(self.base, self.dispatch, self.manifest, server, run=name) + [
                "--final-turn-policy", "deadline_info", "--run-id", "same", *extra]
            with mock.patch.object(G.tools, "ReadOnlyBash", G.FakeJailBash):
                code, _ = G.invoke_runner(self.runner, argv)
            self.assertEqual(code, 0)
            records = [p for p in (self.base / name / "launches").iterdir() if not p.name.endswith(".end.json")]
            launches[name] = (json.loads(records[0].read_bytes())["selection"],
                              (self.base / name / "run.json").read_bytes(),
                              self.runner.run_status(self.base / name))
        plain, filtered = launches["plain"], launches["filtered"]
        self.assertNotIn("conditions", plain[0])
        self.assertEqual(sorted(plain[0]), ["blocks", "by_block", "cells", "families", "order", "seeds"])
        self.assertEqual((filtered[0]["conditions"], filtered[0]["seeds"], filtered[0]["cells"]), (["C0", "C1"], [11], 4))
        self.assertEqual(plain[0]["cells"], 10)
        self.assertEqual(filtered[1], plain[1])                     # run.json does not depend on the selection
        for key in ("selected_conditions", "p_seeds_selected_conditions", "p_seeds_selected_conditions_terminal"):
            self.assertNotIn(key, plain[2])
        self.assertEqual(filtered[2]["selected_conditions"], ["C0", "C1"])
        self.assertEqual(filtered[2]["p_seeds_selected_conditions"]["11"], {"completed": 1, "not_run": 3})
        ignored = ("run_dir", "condition_by_endpoint")         # the fake servers' ports differ
        self.assertEqual({k: v for k, v in filtered[2].items() if not k.startswith(("p_seeds_selected", "selected"))
                          and k not in ignored}, {k: v for k, v in plain[2].items() if k not in ignored})

    def test_a_seed_23_launch_needs_the_order_override_on_a_fresh_run(self):
        """The D10.8 order guard is unchanged: block P seed 23 follows seed 11 unless a reason is recorded."""
        server = self.server(default=H.SUBMIT)
        argv = G.live_args(self.base, self.dispatch, self.manifest, server, run="s23")
        argv[argv.index("--seeds") + 1] = "23"
        argv += ["--final-turn-policy", "submit_only", "--conditions", "C0,C1"]
        with mock.patch.object(G.tools, "ReadOnlyBash", G.FakeJailBash):
            code, output = G.invoke_runner(self.runner, argv)
        self.assertEqual(code, 2)
        self.assertIn("refused", output)
        with mock.patch.object(G.tools, "ReadOnlyBash", G.FakeJailBash):
            code, _ = G.invoke_runner(self.runner, argv + ["--allow-out-of-order", "A4-registers-seed-23-only"])
        self.assertEqual(code, 0)
        result = json.loads(next((self.base / "s23" / "cells").rglob("result.json")).read_bytes())
        self.assertEqual((result["seed"], result["condition"]), (23, "C0"))


@unittest.skipUnless(POSIX and shutil.which("bash"), "Linux only: the launch script runs on Core1")
class A4LaunchScriptTests(unittest.TestCase):
    """The launch script forwards the A4 policies and --conditions (stubbed host, nothing is started)."""

    def host(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        host = G.LaunchHost(directory.name)
        host.run(*G.PREPARE_ARGV, "--runtime-python", str(host.python))
        host.stage("golden1")
        return host

    def runner(self, host, *argv, check=True):
        result = host.run("runner", "--run", "golden1", *argv, check=check)
        units = [host.normalize(call) for call in G.units(host.sudo_calls(), "runner")]
        return result, units

    def args(self, unit):
        tail = unit[next(i for i, value in enumerate(unit) if value.endswith("/scripts/expA_run.py")) + 1:]
        with contextlib.redirect_stderr(io.StringIO()):
            return G.load_script("expA_run").parse_args(tail)

    def test_the_a4_policies_and_conditions_are_forwarded(self):
        host = self.host()
        for policy in ("deadline_info", "submit_only", "deadline", "neutral"):
            protocol = harness.FINAL_TURN_PROTOCOLS[policy]
            for seeds, conditions in (("23", "C0,C1"), ("11", None), ("23", None)):
                with self.subTest(policy=policy, seeds=seeds, conditions=conditions):
                    extra = ("--conditions", conditions) if conditions else ()
                    result, units = self.runner(host, "--blocks", "P", "--seeds", seeds, *extra,
                                                "--final-turn-policy", policy)
                    self.assertEqual(len(units), 1)
                    self.assertIn(f"final-turn policy {policy} ({protocol})", result.stderr)
                    args = self.args(units[0])
                    self.assertEqual((args.final_turn_policy, args.blocks, args.seeds, args.conditions),
                                     (policy, {"P"}, {int(seeds)}, {"C0", "C1"} if conditions else None))
            for bad in (("--blocks", "P", "--seeds", "47"), ("--blocks", "P", "--seeds", "11,23"),
                        ("--blocks", "P", "--seeds", "23", "--conditions", "C0,C2"), ("--seeds", "23")):
                with self.subTest(policy=policy, bad=bad):
                    result, units = self.runner(host, *bad, "--final-turn-policy", policy, check=False)
                    self.assertEqual((result.returncode, units), (2, []))
                    self.assertIn(f"--final-turn-policy {policy}: block P at seed 11 or 23 only (one seed per launch)",
                                  result.stderr)
        # --conditions alone is a selection filter, forwarded like --seeds.
        result, units = self.runner(host, "--conditions", "C3")
        self.assertEqual(self.args(units[0]).conditions, {"C3"})
        result, units = self.runner(host, "--conditions", "C3;rm", check=False)
        self.assertEqual((result.returncode, units), (2, []))


# ---------------------------------------------------------------------------------------------------------------
# (4) the A1 dev check, generalised
# ---------------------------------------------------------------------------------------------------------------

@unittest.skipUnless(POSIX, "cell ids (with '|') are directory names, as on Core2")
class A4DevCheckTests(unittest.TestCase):
    def setUp(self):
        import test_expA_finalturn_devcheck as T
        self.T, self.D = T, T.D
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def plan(self, seed, conditions):
        cells = [self.T.cell_id(item, condition, seed) for item in self.T.ITEMS for condition in conditions]
        return {cell: [["tool"] * 8] for cell in cells}

    def test_seed_23_c0_c1_runs_of_every_policy(self):
        for policy in ("deadline_info", "submit_only", "neutral", "deadline"):
            with self.subTest(policy=policy):
                run_dir = self.T.write_run(self.base / policy, policy, self.plan(23, ("C0", "C1")))
                report = self.D.check_run(run_dir, policy, self.T.codepoints, seed=23, conditions={"C0", "C1"})
                self.assertEqual((report["go"], report["failed_checks"]), (True, []))
                self.assertEqual(report["protocol"], self.D.PROTOCOLS[policy])
                self.assertEqual(report["selection"], {"block": "P", "seed": 23, "conditions": ["C0", "C1"]})
                self.assertEqual(report["cells"]["block_p_seed_23_selection_in_design"], 4)
                self.assertEqual(report["go_rule"]["noticed_final_responses"], 4)
                self.assertIn("block-P seed-23 C0/C1", report["checks"]["run_complete"]["description"])
                # Without the selection the same run is incomplete (seed 23 C2-C4a) or not the seed's.
                full = self.D.check_run(run_dir, policy, self.T.codepoints, seed=23)
                self.assertEqual(full["failed_checks"], ["run_complete"])
                seed11 = self.D.check_run(run_dir, policy, self.T.codepoints)
                self.assertIn("selection", seed11["failed_checks"])
                # The other A4 policy's record fails the config checks.
                other = {"deadline_info": "submit_only", "submit_only": "deadline_info", "neutral": "deadline",
                         "deadline": "neutral"}[policy]
                wrong = self.D.check_run(run_dir, other, self.T.codepoints, seed=23, conditions={"C0", "C1"})
                self.assertIn("run_json_episode_config", wrong["failed_checks"])

    def test_the_a1_report_is_unchanged_by_default(self):
        run_dir = self.T.write_run(self.base / "a1", "deadline", self.T.STANDARD)
        report = self.D.check_run(run_dir, "deadline", self.T.codepoints)
        self.assertNotIn("selection", report)
        self.assertIn("block_p_seed_11_in_design", report["cells"])
        self.assertEqual({name: entry["description"] for name, entry in report["checks"].items()}, self.D.CHECKS)
        self.assertEqual(report["protocol"], "protocols/EXPA_FINALTURN.md")

    def test_the_command_line(self):
        run_dir = self.T.write_run(self.base / "cli", "submit_only", self.plan(23, ("C0", "C1")))
        load = lambda directory: (self.T.codepoints, {"synthetic": True})     # noqa: E731
        with contextlib.redirect_stdout(io.StringIO()):
            code = self.D.main(["--run-dir", str(run_dir), "--policy", "submit_only", "--seed", "23", "--conditions",
                                "C0,C1", "--tokenizer-dir", "x", "--out", str(self.base / "r.json")], load=load)
        self.assertEqual(code, 0)
        with contextlib.redirect_stderr(io.StringIO()):
            code = self.D.main(["--run-dir", str(run_dir), "--policy", "submit_only", "--seed", "23", "--conditions",
                                "C0,C3", "--tokenizer-dir", "x", "--out", str(self.base / "r2.json")], load=load)
        self.assertEqual(code, 2)
        self.assertFalse((self.base / "r2.json").exists())


# ---------------------------------------------------------------------------------------------------------------
# (5) the prespecified analysis
# ---------------------------------------------------------------------------------------------------------------

# SYNTHETIC recovery (C0, C1) per item and policy at seed 23; None = NoAnswer (scores 0); "infra" = missing.
#            neutral        deadline_info   submit_only    deadline
REC = {"a1": ((0, None), (0, 1), (1, 1), (0, 1)),
       "a2": ((1, 0), (0, 0), (0, 1), (0, 1)),
       "b1": ((0, 0), (1, 1), (0, 0), (0, 1)),
       "b2": ((1, 1), (0, 1), (0, 0), (1, "infra"))}
FAMILY = {"a1": "A", "a2": "A", "b1": "B", "b2": "B"}
ORDER = ("neutral", "deadline_info", "submit_only", "deadline")
VARIANT = {"C0": "none", "C1": "omit_node", "C2": "omit_node", "C3": "omit_node", "C4a": "omit_node"}
# Hand-computed (L = C1 - C0; a NoAnswer row scores 0 even with a stray site_recovered 1):
#   a1: L = 0, 1, 0, 1 -> primary [(0 - 0) + (1 - 1)] / 2 = 0
#   a2: L = -1, 0, 1, 1 -> [(1 + 1) + (1 - 0)] / 2 = 3/2      family A = 3/4
#   b1: L = 0, 0, 0, 1 -> [(0 - 0) + (1 - 0)] / 2 = 1/2       family B = 1/2 (b2 lacks deadline C1)
#   primary = (3/4 + 1/2) / 2 = 5/8; t = 5/8 / (1/8) = 5 with df 1.
#   S5 deadline - neutral (mean of C0, C1): a1 1/2, a2 0, b1 1/2 -> A 1/4, B 1/2 -> 3/8
#   S4 interaction: a1 -1, a2 1/2, b1 -1/2 -> A -1/4, B -1/2 -> -3/8
#   S2 imperative main effect: a1 1/2, a2 1/4, b1 -1/4 -> A 3/8, B -1/4 -> 1/16
#   S3 information main effect: a1 0, a2 -1/4, b1 3/4 -> A -1/8, B 3/4 -> 5/16


def score_row(item, condition, seed, value, *, block="P", state=None):
    variant = VARIANT[condition]
    state = state or ("missing_infrastructure" if value == "infra" else "completed")
    completed = state == "completed"
    exit_status = ("Submitted" if value in (0, 1) else "NoAnswer") if completed else "InfraInterrupted"
    return {"cell_id": f"{block}|{item}|{variant}|{condition}|s{seed}", "block": block, "item_id": item,
            "condition": condition, "variant": variant, "seed": seed, "family": FAMILY[item], "cell_state": state,
            "exit_status": exit_status, "items_sha256": "1" * 64,
            "site_recovered": (1 if value is None else value) if completed else None,   # NoAnswer: stray 1
            "n_submitted": 2 if exit_status == "Submitted" else 0,
            "precision_lower": 0.5 if exit_status == "Submitted" else None,
            "precision_upper": 1.0 if exit_status == "Submitted" else None}


def policy_rows(policy, seed=23):
    index = ORDER.index(policy)
    rows = []
    for item, values in REC.items():
        for condition, value in zip(("C0", "C1"), values[index]):
            rows.append(score_row(item, condition, seed, value))
        for condition in ("C2", "C3", "C4a"):           # frozen design cells never dispatched
            rows.append(score_row(item, condition, seed, 0, state="not_dispatched"))
        rows.append(score_row(item, "C0", 11, 0, state="not_dispatched"))
    rows[0]["policy_marker"] = policy                  # the four score files differ byte for byte
    return rows


class A4AnalysisTests(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.paths = {}
        for arm, policy in zip(("neutral", "info", "submit", "deadline"), ORDER):
            self.paths[arm] = self.write(f"{arm}.jsonl", policy_rows(policy))

    def tearDown(self):
        self.temporary.cleanup()

    def write(self, name, rows):
        path = self.base / name
        path.write_bytes(b"".join(json.dumps(row, sort_keys=True).encode() + b"\n" for row in rows))
        return str(path)

    def argv(self, out="out", **paths):
        paths = dict(self.paths, **paths)
        return ["--scores-neutral", paths["neutral"], "--scores-info", paths["info"], "--scores-submit",
                paths["submit"], "--scores-deadline", paths["deadline"], "--out", str(self.base / out)]

    def main(self, argv):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
            code = A.main(argv)
        return code, err.getvalue()

    def results(self, out="out", extra=()):
        code, err = self.main(self.argv(out) + list(extra))
        self.assertEqual(code, 0, err)
        return json.loads((self.base / out / "A4_RESULTS.json").read_bytes())

    def test_the_hand_computed_primary_and_secondaries(self):
        result = self.results()
        self.assertTrue(result["label"].startswith("PRESPECIFIED") and "ANALYSIS RECORD" in result["label"])
        self.assertEqual(result["script"]["sha256"], sha((PROJECT / "scripts" / "expA_a4_analysis.py").read_bytes()))
        self.assertEqual(set(result["inputs"]), {f"scores_{policy}" for policy in ORDER})
        for arm, policy in zip(("neutral", "info", "submit", "deadline"), ORDER):
            self.assertEqual(result["inputs"][f"scores_{policy}"]["sha256"], sha(Path(self.paths[arm]).read_bytes()))
        a = result["analysis"]
        primary = a["primary"]
        self.assertEqual((primary["estimate_exact"], primary["n_items"], primary["n_families"], primary["df"]),
                         ("5/8", 3, 2, 1))
        self.assertEqual(primary["family_values"], {"A": 0.75, "B": 0.5})
        self.assertAlmostEqual(primary["t"], 5.0, places=9)
        expected = t_statistics([Fraction(3, 4), Fraction(1, 2)], 0.95)
        self.assertAlmostEqual(primary["p_two_sided"], expected["p_two_sided"], places=12)
        self.assertEqual(primary["dropped"]["items_missing_a_needed_cell"], 1)
        self.assertFalse(primary["reject_at_alpha"])          # p = 0.126 with df 1
        s = a["secondary"]
        self.assertEqual({key: s[key]["estimate_exact"] for key in (
            "S2_imperative_main_effect", "S3_information_main_effect", "S4_imperative_x_information_interaction",
            "S5_deadline_minus_neutral")}, {"S2_imperative_main_effect": "1/16", "S3_information_main_effect": "5/16",
                                            "S4_imperative_x_information_interaction": "-3/8",
                                            "S5_deadline_minus_neutral": "3/8"})
        self.assertNotIn("p_two_sided", s["S5_deadline_minus_neutral"])
        # S1: a1 [(1-0)+(1-0)]/2 = 1, a2 [(0+1)+(1-1)]/2 = 1/2, b1 [(0-0)+(1-0)]/2 = 1/2 -> A 3/4, B 1/2 -> 5/8
        self.assertEqual(s["S1_information_effect_on_list_effect"]["estimate_exact"], "5/8")
        s6 = s["S6_by_policy"]
        # L(neutral): a1 0, a2 -1, b1 0, b2 0 -> A -1/2, B 0 -> -1/4
        self.assertEqual(s6["neutral"]["list_effect_L"]["estimate_exact"], "-1/4")
        deadline_c1 = s6["deadline"]["by_condition"]["C1"]
        self.assertEqual((deadline_c1["completed_episodes"], deadline_c1["missing_infrastructure"]), (3, 1))
        neutral_c1 = s6["neutral"]["by_condition"]["C1"]
        self.assertEqual(neutral_c1["exits"], {"NoAnswer": 1, "Submitted": 3})
        self.assertEqual(neutral_c1["submission_rate"]["micro"]["mean"], 0.75)
        self.assertEqual(neutral_c1["precision_lower"]["micro"]["n"], 3)
        self.assertEqual(a["population"]["missing_infrastructure_cells"],
                         {"neutral": 0, "deadline_info": 0, "submit_only": 0, "deadline": 1})
        self.assertEqual(a["population"]["cells"]["deadline"]["C1"], {"completed": 3, "missing_infrastructure": 1})
        self.assertEqual(a["factors"]["deadline_info"], {"I_deadline_information": 1, "D_submit_imperative": 0})
        md = (self.base / "out" / "A4_RESULTS.md").read_text(encoding="utf-8")
        self.assertIn("+62.5", md)
        self.assertNotIn(b"\r\n", (self.base / "out" / "A4_RESULTS.md").read_bytes())

    def test_refusals(self):
        (self.base / "exists").mkdir()
        self.assertEqual(self.main(self.argv("exists"))[0], 2)
        dispatched = policy_rows("deadline")
        dispatched.append(score_row("a1", "C2", 47, 1))
        code, err = self.main(self.argv("o1", deadline=self.write("bad.jsonl", dispatched)))
        self.assertEqual(code, 2)
        self.assertIn("were dispatched", err)
        self.assertFalse((self.base / "o1").exists())
        code, err = self.main(self.argv("o2", info=self.paths["neutral"]))
        self.assertEqual(code, 2)
        self.assertIn("byte-identical", err)
        duplicate = policy_rows("deadline") + [score_row("a1", "C0", 23, 1)]
        duplicate[-1]["cell_id"] += "-dup"
        self.assertEqual(self.main(self.argv("o3", deadline=self.write("dup.jsonl", duplicate)))[0], 2)
        self.assertEqual(self.main(self.argv("o4") + ["--seed", "11"])[0], 2)    # no seed-11 rows

    def run_json(self, name, policy, selection):
        directory = self.base / name
        (directory / "launches").mkdir(parents=True)
        config = harness.EpisodeConfig(final_turn_policy=policy).record()
        (directory / "run.json").write_bytes(json.dumps({"run_id": name, "episode_config": config}).encode())
        (directory / "launches" / "x.json").write_bytes(json.dumps({"selection": selection}).encode())
        (directory / "launches" / "x.end.json").write_bytes(b"{}")
        return str(directory / "run.json")

    def test_run_json_and_launch_verification(self):
        good = {"blocks": ["P"], "seeds": [23], "conditions": ["C0", "C1"], "families": None}
        paths = {arm: self.run_json(arm, policy, good) for arm, policy in zip(("neutral", "info", "submit", "deadline"),
                                                                              ORDER)}
        extra = [item for arm, path in paths.items() for item in (f"--run-json-{arm}", path)]
        result = self.results("ok", extra)
        self.assertTrue(result["run_json_verified"])
        self.assertEqual(result["inputs"]["run_json_submit_only"]["final_turn_policy"], "submit_only")
        self.assertEqual(len(result["inputs"]["run_json_deadline"]["launches"]), 1)
        # A swapped policy, a seed-11 launch, a launch without --conditions, or only some run.json files: refused.
        swapped = list(extra)
        swapped[swapped.index(paths["info"])] = paths["submit"]
        self.assertEqual(self.main(self.argv("r1") + swapped)[0], 2)
        for name, selection in (("s11", dict(good, seeds=[11])), ("nocond", {"blocks": ["P"], "seeds": [23]})):
            bad = list(extra)
            bad[bad.index(paths["deadline"])] = self.run_json(name, "deadline", selection)
            code, err = self.main(self.argv("r-" + name) + bad)
            self.assertEqual(code, 2, name)
            self.assertIn("selection", err)
        self.assertEqual(self.main(self.argv("r3") + extra[:2])[0], 2)


if __name__ == "__main__":
    unittest.main()
