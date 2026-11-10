"""Exp A follow-up A6: A3's ledger-off arm at the doubled budget (protocols/EXPA_A6_BUDGET16.md). SYNTHETIC fixtures.

(1) The config records (any OS): every earlier record (default, A1, A3, the 16-response design) is byte-identical to
    the one before A6 (sha256 pinned from the commit before A6); the A6 record is the A3 off record at 16 responses and
    32,768 tokens, with the deadline notice and the ledger at call_index 15.
(2) The harness against the fake server (POSIX): in a 16-call episode the off-ledger is built and the deadline notice
    appended before the 16th response (call_index 15) only; the requests never carry the ledger; an episode that
    submits earlier never builds it; the receipts name call_index 15.
(3) The runner (POSIX): the A6 combination is registered and nothing broader; its neighbours are refused with
    messages naming protocols/EXPA_A6_BUDGET16.md; the 16/32768 design without a notice stays registered; dry run,
    run.json and an episode of a live run.
(4) The launch script (POSIX + bash): the A6 combination is forwarded at runner (the budget is a runner option, not a
    prepare option); its neighbours are refused before any unit.
(5) The dev check (scripts/expA_a3_devcheck.py, arm off) on SYNTHETIC 16-call runs (any OS): the last call is
    derived from run.json's max_model_calls; --expect-budget 16:32768 gates the budget.
(6) The analysis (scripts/expA_a6_analysis.py) on a hand-computed two-family fixture (any OS).
ANALYSIS RECORD.
"""
import collections
import contextlib
import hashlib
import importlib.util
import io
import json
import math
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
sys.path.insert(0, str(PROJECT / "tests"))

from absent_edges.probe import harness  # noqa: E402

POSIX = os.name == "posix"
if POSIX:
    import test_expA_default_golden as G      # puts the project on sys.path
    import test_probe_harness as H
    from absent_edges.models import LocalModel, ModelProfile
    from absent_edges.probe import ledger as L
    from absent_edges.probe import score as S
    from absent_edges.probe.tools import ToolBox


def load_module(name, path):
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module


A = load_module("expA_a6_analysis", PROJECT / "scripts" / "expA_a6_analysis.py")
F = load_module("test_expA_a3_devcheck", PROJECT / "tests" / "test_expA_a3_devcheck.py")   # its SYNTHETIC run builder
D = F.T                                                                                     # scripts/expA_a3_devcheck.py

PROTOCOL = "protocols/EXPA_A6_BUDGET16.md"
PREFIX = "\n\n[harness notice] "
DEADLINE = PREFIX + "This is your last response. Call submit_answer now."
HEADER = "Candidate call sites seen in your earlier tool outputs (unverified; check before submitting):"
BUDGET16 = ("--max-model-calls", "16", "--context-budget", "32768")
A6_ARM = ("--final-turn-policy", "deadline", "--evidence-ledger", "off") + BUDGET16
# sha256 of json.dumps(record) at the commit before A6 (13ec48966); the first five as pinned by tests/test_expA_a5_review.py.
PINNED = {"none": "ff8921f6e5ef35bfbacc58037ac23081468f43a9404ea8a62ed174f1ffd9df4b",
          "neutral": "2aab8a084aedf3c705a68e9600fd61aeccc68364e9aa2a07a93fe989ca9b82a4",
          "deadline": "773cb89af25df5d771e6248f7d0d45246f35045c7a803a2a8f1a9a956bb4819b",
          "a3_off": "b1d564984b0012b8d7cff6649131da96375d38577825a6ccdd64b9a9154c00a1",
          "a3_on": "c4ae1e0f20a92e0c49b574b9826e3f541fbe735823a45ef37b2240f56cfa1ef8",
          "budget16": "f52afc9af37444d4bee8917903e2e69937d6a8793fb30823f8c811bf3b6e63f7",
          "design_default": "6b0a49cd8eb8e6157e2d8d516f0ca2231e92b4e494c178b1c28af838ebdedb5c",
          "design_a3": "723f7aa95dd4494c2b174ac09ddc7aa353a52dcbe48297f47354ac8a76c44d0c"}
A6_RECORD_SHA256 = "baf1d18491ae410d6cebbcf474a69dc310b031985283aa0c6baa37af1c3c57db"   # frozen with this commit
BASH = {"calls": [("bash", {"command": "grep -rn target ."})]}
SEARCH = {"calls": [("source_reference_search", {"query": "pkg/core.py::target"})]}
SUBMIT = {"calls": [("submit_answer", {"sites": [{"path": "pkg/core.py", "line": 6}], "complete": "yes"})]}
GREP = "pkg/core.py:1:def target(x):\npkg/core.py:6:    return target(1)\n"
GREP_LEDGER = HEADER + "\n- pkg/core.py:6: return target(1)"
Run = collections.namedtuple("Run", "result server rows")


def sha(text):
    return hashlib.sha256(text if isinstance(text, bytes) else text.encode("utf-8")).hexdigest()


def digest(value):
    return sha(json.dumps(value, ensure_ascii=False))


def a6_config():
    return harness.EpisodeConfig(max_model_calls=16, context_budget=32768, final_turn_policy="deadline",
                                 evidence_ledger="off")


# ---------------------------------------------------------------------------------------------------------------
# (1) the config records
# ---------------------------------------------------------------------------------------------------------------

class A6ConfigTests(unittest.TestCase):
    maxDiff = None

    def test_earlier_records_are_byte_identical(self):
        configs = {"none": harness.EpisodeConfig(), "neutral": harness.EpisodeConfig(final_turn_policy="neutral"),
                   "deadline": harness.EpisodeConfig(final_turn_policy="deadline"),
                   "a3_off": harness.EpisodeConfig(final_turn_policy="deadline", evidence_ledger="off"),
                   "a3_on": harness.EpisodeConfig(final_turn_policy="deadline", evidence_ledger="on"),
                   "budget16": harness.EpisodeConfig(max_model_calls=16, context_budget=32768)}
        for name, config in configs.items():
            with self.subTest(name=name):
                self.assertEqual(digest(config.record()), PINNED[name])
        self.assertEqual(digest(harness.design_table_for(None)), PINNED["design_default"])
        self.assertEqual(digest(harness.design_table_for("a3-ledger")), PINNED["design_a3"])

    def test_the_a6_record(self):
        record = a6_config().record()
        self.assertEqual(digest(record), A6_RECORD_SHA256)
        a3_off = harness.EpisodeConfig(final_turn_policy="deadline", evidence_ledger="off").record()
        self.assertEqual(list(record), list(a3_off))
        # Only the two budgets and the two call indices differ from the A3 off record.
        expected = dict(a3_off, max_model_calls=16, context_budget=32768,
                        final_turn_notice=dict(a3_off["final_turn_notice"], call_index=15),
                        evidence_ledger_rules=dict(a3_off["evidence_ledger_rules"], call_index=15))
        self.assertEqual(record, expected)
        self.assertEqual((a3_off["final_turn_notice"]["call_index"], a3_off["evidence_ledger_rules"]["call_index"]),
                         (7, 7))
        self.assertEqual(harness.config_design(a6_config()), "a3-ledger")
        # The dev check rebuilds the same record from the protocol's values.
        self.assertEqual(D.expected_episode_config("off", (16, 32768)), record)
        self.assertEqual(D.expected_episode_config("off", (8, 16384)), a3_off)
        self.assertEqual(D.expected_episode_config("off"), a3_off)

    def test_the_system_prompt_states_16_responses(self):
        self.assertIn("You have at most {max_model_calls} responses", harness.SYSTEM_PROMPT_TEMPLATE)


# ---------------------------------------------------------------------------------------------------------------
# (2) the harness: the notice and the off-ledger at call_index 15
# ---------------------------------------------------------------------------------------------------------------

@unittest.skipUnless(POSIX, "the ToolBox search scans with POSIX no-follow reads")
class A6EpisodeTests(unittest.TestCase):
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

    def run_episode(self, script, config):
        server = H.FakeServer(script)
        self.servers.append(server)
        self.count += 1
        directory = Path(self.temporary.name) / f"attempt-{self.count}"
        directory.mkdir()
        client = LocalModel(ModelProfile(endpoint=server.endpoint), timeout=20)
        toolbox = ToolBox(self.root, ["pkg"], H.FakeBash(GREP))
        cell = "LEDGER|item-h|omit_node|C1|s61"
        episode = H.make_episode("C1", block="LEDGER", variant="omit_node", cell_id=cell, attempt_id=cell + ".a001",
                                 seed=61)
        result, digest_ = harness.run_episode(episode, directory, client=client, toolbox=toolbox, config=config)
        self.assertEqual(digest_, L.sha256_path(directory / "result.json"))
        return Run(result, server, H.events(directory))

    @staticmethod
    def values(run, kind):
        return [row["value"] for row in run.rows if row["kind"] == kind]

    def test_the_notice_and_the_ledger_sit_before_the_16th_response(self):
        run = self.run_episode([BASH] * 16, a6_config())
        self.assertEqual((run.result["exit_status"], run.result["calls"]), ("NoAnswer", 16))
        chats = run.server.chat_requests()
        self.assertEqual(len(chats), 16)
        for index, chat in enumerate(chats):
            text = json.dumps(chat["messages"])
            self.assertEqual("[harness notice]" in text, index == 15, index)
            self.assertNotIn("Candidate call sites", text)                  # off: never sent
        last = chats[15]["messages"][-1]
        self.assertEqual(last["role"], "tool")
        self.assertTrue(last["content"].endswith(DEADLINE))
        kinds = [row["kind"] for row in run.rows]
        index = kinds.index("evidence_ledger")
        self.assertEqual((kinds.count("evidence_ledger"), kinds.count("final_turn_notice"), kinds[index + 1],
                          kinds[index + 2]), (1, 1, "final_turn_notice", "context_accounting"))
        ledger, notice = self.values(run, "evidence_ledger")[0], self.values(run, "final_turn_notice")[0]
        self.assertEqual((ledger["call_index"], ledger["mode"], ledger["appended"], ledger["text"]),
                         (15, "off", False, GREP_LEDGER))
        self.assertEqual(self.values(run, "context_accounting")[15]["call_index"], 15)
        self.assertEqual((notice["call_index"], notice["policy"]), (15, "deadline"))
        self.assertEqual(ledger["content_sha256"], ledger["previous_content_sha256"])
        self.assertEqual(notice["previous_content_sha256"], ledger["content_sha256"])
        receipt = run.result["exposure"]["evidence_ledger"]
        self.assertEqual((receipt["call_index"], receipt["computed"], receipt["appended"], receipt["sent"],
                          receipt["delivered"]), (15, True, False, False, False))
        self.assertEqual(run.result["accounting"]["evidence_ledger"], receipt)
        final = run.result["exposure"]["final_turn_notice"]
        self.assertEqual((final["call_index"], final["appended"], final["sent"], final["delivered"]),
                         (15, True, True, True))
        self.assertEqual(run.result["config"], a6_config().record())
        self.assertIn("You have at most 16 responses", self.values(run, "episode_setup")[0]["system"])

    def test_an_episode_that_submits_before_the_16th_response_never_builds_it(self):
        for responses in (8, 9, 15):
            with self.subTest(responses=responses):
                run = self.run_episode([BASH] * (responses - 1) + [SUBMIT], a6_config())
                self.assertEqual((run.result["exit_status"], run.result["calls"]), ("Submitted", responses))
                kinds = [row["kind"] for row in run.rows]
                self.assertNotIn("evidence_ledger", kinds)
                self.assertNotIn("final_turn_notice", kinds)
                self.assertNotIn("[harness notice]", json.dumps(run.server.chat_requests()))
                self.assertEqual(run.result["exposure"]["evidence_ledger"], D.unbuilt_receipt("off", 15))

    def test_a_submission_in_the_16th_response_saw_the_notice(self):
        run = self.run_episode([BASH] * 15 + [SUBMIT], a6_config())
        self.assertEqual((run.result["exit_status"], run.result["calls"]), ("Submitted", 16))
        self.assertEqual(run.result["exposure"]["evidence_ledger"]["computed"], True)
        self.assertTrue(run.server.chat_requests()[15]["messages"][-1]["content"].endswith(DEADLINE))

    def test_the_8_response_arm_is_unchanged(self):
        run = self.run_episode([BASH] * 8, harness.EpisodeConfig(final_turn_policy="deadline", evidence_ledger="off"))
        chats = run.server.chat_requests()
        self.assertEqual([("[harness notice]" in json.dumps(chat)) for chat in chats], [False] * 7 + [True])
        self.assertEqual((self.values(run, "evidence_ledger")[0]["call_index"],
                          self.values(run, "final_turn_notice")[0]["call_index"]), (7, 7))


# ---------------------------------------------------------------------------------------------------------------
# (3) the runner
# ---------------------------------------------------------------------------------------------------------------

@unittest.skipUnless(POSIX, "the runner reads checkouts with POSIX no-follow reads")
class A6RunnerTests(unittest.TestCase):
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

    def parse(self, *extra):
        with contextlib.redirect_stderr(io.StringIO()):
            return self.runner.parse_args(["--dispatch", str(self.dispatch), "--dry-run", *extra])

    def refused(self, *extra):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
            self.runner.parse_args(["--dispatch", str(self.dispatch), "--dry-run", *extra])
        self.assertEqual(raised.exception.code, 2)
        return stderr.getvalue()

    def test_registration(self):
        self.assertEqual(self.runner.REGISTERED_A6_DESIGNS,
                         {(16, 32768, "deadline", "off"): (PROTOCOL, ("gemma4-31b",))})
        # Every earlier registry is unchanged.
        self.assertEqual(self.runner.REGISTERED_BUDGETS, {(8, 16384), (16, 32768)})
        self.assertEqual(set(self.runner.REGISTERED_DESIGNS), {
            (8, 16384, "none"), (16, 32768, "none"), (8, 16384, "neutral"), (8, 16384, "deadline"),
            (8, 16384, "deadline_info"), (8, 16384, "submit_only")})
        self.assertEqual(set(self.runner.REGISTERED_LEDGER_DESIGNS),
                         {(8, 16384, "deadline", "off"), (8, 16384, "deadline", "on")})
        for seeds in ((), ("--seeds", "61"), ("--seeds", "73"), ("--seeds", "61,73,89")):
            with self.subTest(seeds=seeds):
                args = self.parse(*A6_ARM, "--blocks", "LEDGER", *seeds)
                self.assertEqual((args.episode_config, args.design, args.blocks), (a6_config(), "a3-ledger",
                                                                                   {"LEDGER"}))
        args = self.parse(*A6_ARM, "--blocks", "LEDGER", "--model-profile", "gemma4-31b")
        self.assertEqual(args.episode_config, a6_config())
        # The registered 16-response design without a notice is unchanged (protocols/EXPA_BUDGET16.md).
        self.assertEqual(self.parse(*BUDGET16).episode_config,
                         harness.EpisodeConfig(max_model_calls=16, context_budget=32768))
        self.assertEqual(self.parse(*BUDGET16, "--final-turn-policy", "none").design, None)

    def test_neighbours_are_refused(self):
        hint = ("; with --max-model-calls 16 --context-budget 32768 only as A6 (protocols/EXPA_A6_BUDGET16.md): "
                "--final-turn-policy deadline --evidence-ledger off --blocks LEDGER and --model-profile gemma4-31b")
        a1 = ("--final-turn-policy {policy} ({protocol}) is registered only with the prereg budget (8 responses, "
              "16384 tokens) and --model-profile gemma4-31b")
        a1_deadline = a1.format(policy="deadline", protocol="protocols/EXPA_FINALTURN.md")
        cases = (
            # the ledger on at 16 responses: not registered (A3 runs on and off at 8 responses only)
            (("--final-turn-policy", "deadline", "--evidence-ledger", "on", *BUDGET16, "--blocks", "LEDGER"),
             a1_deadline + hint),
            # the deadline line at 16 responses without the ledger record, on block P or block LEDGER
            (("--final-turn-policy", "deadline", *BUDGET16, "--blocks", "P", "--seeds", "11"), a1_deadline + hint),
            (("--final-turn-policy", "deadline", *BUDGET16, "--blocks", "LEDGER"), a1_deadline + hint),
            # other notices at 16 responses (no A6 hint: A6 is the deadline line only)
            (("--final-turn-policy", "neutral", "--evidence-ledger", "off", *BUDGET16, "--blocks", "LEDGER"),
             a1.format(policy="neutral", protocol="protocols/EXPA_FINALTURN.md") + "\n"),
            (("--final-turn-policy", "deadline_info", "--evidence-ledger", "off", *BUDGET16, "--blocks", "LEDGER"),
             a1.format(policy="deadline_info", protocol="protocols/EXPA_A4_DIRECTIVE.md") + "\n"),
            # the ledger at 16 responses without the deadline line
            (("--evidence-ledger", "off", *BUDGET16, "--blocks", "LEDGER"),
             "--evidence-ledger off (protocols/EXPA_A3_LEDGER.md) is registered only with --final-turn-policy "
             "deadline, the prereg budget (8 responses, 16384 tokens) and --model-profile gemma4-31b" + hint),
            # A6 outside block LEDGER or its seeds
            (A6_ARM + ("--blocks", "P", "--seeds", "11"),
             "--evidence-ledger off (protocols/EXPA_A6_BUDGET16.md) runs block LEDGER only, at seeds 61, 73, 89"),
            (A6_ARM, "--evidence-ledger off (protocols/EXPA_A6_BUDGET16.md) runs block LEDGER only"),
            (A6_ARM + ("--blocks", "LEDGER,P"), "(protocols/EXPA_A6_BUDGET16.md) runs block LEDGER only"),
            (A6_ARM + ("--blocks", "LEDGER", "--seeds", "61,11"), "(protocols/EXPA_A6_BUDGET16.md) runs block LEDGER"),
            # another model
            (A6_ARM + ("--blocks", "LEDGER", "--model-profile", "gemma4-12b"),
             "--final-turn-policy deadline --evidence-ledger off with --max-model-calls 16 --context-budget 32768 "
             "(protocols/EXPA_A6_BUDGET16.md) is registered only with --model-profile gemma4-31b"),
            (A6_ARM + ("--blocks", "LEDGER", "--model-profile", "gemma4-31b-qat"),
             "(protocols/EXPA_A6_BUDGET16.md) is registered only with --model-profile gemma4-31b"),
            # A5 at 16 responses
            (("--final-turn-policy", "deadline", "--submit-review", "neutral", *BUDGET16, "--blocks", "LEDGER"),
             a1_deadline + hint),
            (A6_ARM + ("--submit-review", "ledger", "--blocks", "LEDGER"),
             "--submit-review ledger (protocols/EXPA_A5_REVIEW.md) is registered only"),
            # an unregistered budget
            (("--final-turn-policy", "deadline", "--evidence-ledger", "off", "--max-model-calls", "16",
              "--blocks", "LEDGER"), "must be 8/16384 (prereg) or 16/32768"),
        )
        for argv, message in cases:
            with self.subTest(argv=argv):
                self.assertIn(message, self.refused(*argv))

    def test_dry_run_run_json_and_an_episode(self):
        argv = ["--dispatch", str(self.dispatch), "--dry-run", "--blocks", "LEDGER", "--seeds", "61", *A6_ARM]
        code, output = G.invoke_runner(self.runner, argv)
        summary = json.loads(output)
        self.assertEqual((code, summary["selected_cells"], summary["by_block"], summary["design"]),
                         (0, 4, {"LEDGER": 4}, harness.design_table_for("a3-ledger")))
        self.assertEqual(summary["episode_config"], a6_config().record())
        server = H.FakeServer(default=SEARCH)            # never submits: the episode reaches its 16th response
        self.servers.append(server)
        live = G.live_args(self.base, self.dispatch, self.manifest, server, run="a6")
        live[live.index("--blocks") + 1] = "LEDGER"
        live[live.index("--seeds") + 1] = "61"
        with mock.patch.object(G.tools, "ReadOnlyBash", G.FakeJailBash):
            code, _ = G.invoke_runner(self.runner, live + list(A6_ARM))
        self.assertEqual(code, 0)
        run_json = json.loads((self.base / "a6" / "run.json").read_bytes())
        self.assertEqual((run_json["episode_config"], run_json["design"]["name"]), (a6_config().record(), "a3-ledger"))
        cells = (self.base / "a6" / "cells.jsonl").read_text().splitlines()
        self.assertEqual(S.detect_design(run_json, [json.loads(line)["cell_id"] for line in cells])[0], "a3-ledger")
        result = json.loads(next((self.base / "a6" / "cells").rglob("result.json")).read_bytes())
        self.assertEqual((result["block"], result["seed"], result["exit_status"], result["calls"]),
                         ("LEDGER", 61, "NoAnswer", 16))
        self.assertEqual((result["exposure"]["evidence_ledger"]["call_index"],
                          result["exposure"]["evidence_ledger"]["computed"],
                          result["exposure"]["final_turn_notice"]["call_index"],
                          result["exposure"]["final_turn_notice"]["delivered"]), (15, True, 15, True))
        chats = server.chat_requests()
        self.assertEqual([("[harness notice]" in json.dumps(chat)) for chat in chats], [False] * 15 + [True])
        self.assertNotIn("Candidate call sites", json.dumps(chats))
        self.assertIn("You have at most 16 responses", chats[0]["messages"][0]["content"])


# ---------------------------------------------------------------------------------------------------------------
# (4) the launch script
# ---------------------------------------------------------------------------------------------------------------

@unittest.skipUnless(POSIX and shutil.which("bash"), "Linux only: the launch script runs on Core1")
class A6LaunchScriptTests(unittest.TestCase):
    maxDiff = None

    def host(self, *prepare_extra):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        host = G.LaunchHost(directory.name)
        host.run(*G.PREPARE_ARGV, "--runtime-python", str(host.python), *prepare_extra)
        host.stage("golden1")
        return host

    def runner_units(self, host, *extra, check=True):
        result = host.run("runner", "--run", "golden1", *extra, check=check)
        return result, [host.normalize(call) for call in G.units(host.sudo_calls(), "runner")]

    def runner_args(self, unit):
        tail = unit[next(i for i, value in enumerate(unit) if value.endswith("/scripts/expA_run.py")) + 1:]
        with contextlib.redirect_stderr(io.StringIO()):
            return G.load_script("expA_run").parse_args(tail)

    def test_the_a6_combination_is_forwarded(self):
        host = self.host()
        golden = G.GOLDEN["launch"]["runner_units"][0]
        self.assertEqual(golden[-4:], ["--blocks", "P", "--seeds", "11"])
        for seeds in ((), ("--seeds", "61"), ("--seeds", "61,73,89")):
            with self.subTest(seeds=seeds):
                result, units = self.runner_units(host, "--blocks", "LEDGER", *seeds, "--final-turn-policy",
                                                  "deadline", "--evidence-ledger", "off", *BUDGET16)
                self.assertEqual(units, [golden[:-4] + ["--blocks", "LEDGER", *seeds, *BUDGET16,
                                                        "--final-turn-policy", "deadline", "--evidence-ledger", "off"]])
                self.assertIn("A6: evidence ledger off at 16 responses and 32768 tokens (protocols/EXPA_A6_BUDGET16.md;"
                              " ledger rules protocols/EXPA_A3_LEDGER.md)", result.stderr)
                self.assertIn("budgets --max-model-calls 16 --context-budget 32768", result.stderr)
                args = self.runner_args(units[0])
                self.assertEqual((args.episode_config, args.design, args.blocks), (a6_config(), "a3-ledger",
                                                                                   {"LEDGER"}))
        # The A3 off arm at 8 responses and the 16-response design without a notice are forwarded as before.
        result, units = self.runner_units(host, "--blocks", "LEDGER", "--final-turn-policy", "deadline",
                                          "--evidence-ledger", "off")
        self.assertEqual(units, [golden[:-4] + ["--blocks", "LEDGER", "--final-turn-policy", "deadline",
                                                "--evidence-ledger", "off"]])
        self.assertIn("evidence ledger off (protocols/EXPA_A3_LEDGER.md)", result.stderr)
        self.assertNotIn("A6:", result.stderr)
        _, units = self.runner_units(host, "--blocks", "P", "--seeds", "11", *BUDGET16)
        self.assertEqual(units, [golden + list(BUDGET16)])

    def test_neighbours_are_refused_before_any_unit(self):
        host = self.host()
        hint = ("; with --max-model-calls 16 --context-budget 32768 only as A6: --final-turn-policy deadline "
                "--evidence-ledger off --blocks LEDGER and model profile gemma4-31b (protocols/EXPA_A6_BUDGET16.md)")
        a1_deadline = ("--final-turn-policy deadline: registered only with the default budget (8 responses, 16384 "
                       "tokens) and model profile gemma4-31b (protocols/EXPA_FINALTURN.md)")
        cases = ((["--blocks", "LEDGER", "--final-turn-policy", "deadline", "--evidence-ledger", "on", *BUDGET16],
                  a1_deadline + hint),
                 (["--blocks", "P", "--seeds", "11", "--final-turn-policy", "deadline", *BUDGET16], a1_deadline + hint),
                 (["--blocks", "LEDGER", "--final-turn-policy", "neutral", "--evidence-ledger", "off", *BUDGET16],
                  "--final-turn-policy neutral: registered only with the default budget (8 responses, 16384 tokens) "
                  "and model profile gemma4-31b (protocols/EXPA_FINALTURN.md)\n"),
                 (["--blocks", "LEDGER", "--evidence-ledger", "off", *BUDGET16],
                  "--evidence-ledger off: registered only with --final-turn-policy deadline, the default budget "
                  "(8 responses, 16384 tokens) and model profile gemma4-31b (protocols/EXPA_A3_LEDGER.md)" + hint),
                 (["--blocks", "P", "--seeds", "11", "--final-turn-policy", "deadline", "--evidence-ledger", "off",
                   *BUDGET16],
                  "--evidence-ledger off: block LEDGER only, pass --blocks LEDGER (and --seeds from 61,73,89) "
                  "(protocols/EXPA_A6_BUDGET16.md)"),
                 (["--blocks", "LEDGER", "--seeds", "61,11", "--final-turn-policy", "deadline", "--evidence-ledger",
                   "off", *BUDGET16], "(protocols/EXPA_A6_BUDGET16.md)"),
                 (["--blocks", "LEDGER", "--final-turn-policy", "deadline", "--submit-review", "neutral", *BUDGET16],
                  a1_deadline + hint),
                 (["--blocks", "LEDGER", "--final-turn-policy", "deadline", "--evidence-ledger", "off",
                   "--max-model-calls", "16"], "only the registered 16/32768 design"))
        for extra, message in cases:
            with self.subTest(extra=extra):
                result, units = self.runner_units(host, *extra, check=False)
                self.assertEqual((result.returncode, units), (2, []))
                self.assertIn(message, result.stderr)

    def test_a6_needs_the_default_model_profile(self):
        for profile in ("gemma4-12b", "gemma4-31b-qat"):
            with self.subTest(profile=profile):
                host = self.host("--model-profile", profile)
                result, units = self.runner_units(host, "--blocks", "LEDGER", "--final-turn-policy", "deadline",
                                                  "--evidence-ledger", "off", *BUDGET16, check=False)
                self.assertEqual((result.returncode, units), (2, []))
                self.assertIn("--final-turn-policy deadline --evidence-ledger off with --max-model-calls 16 "
                              "--context-budget 32768: registered only with model profile gemma4-31b "
                              "(protocols/EXPA_A6_BUDGET16.md)", result.stderr)

    def test_the_budget_is_a_runner_option(self):
        host = self.host()
        result = host.run(*G.PREPARE_ARGV[:1], "--run", "golden2", "--max-model-calls", "16", check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(G.units(host.sudo_calls(), "runner"), [])


# ---------------------------------------------------------------------------------------------------------------
# (5) the dev check at 16 responses
# ---------------------------------------------------------------------------------------------------------------

B16 = (16, 32768)
TOOLS16 = ["tool"] * 16
# SYNTHETIC: the plan of each LEDGER seed-61 cell at 16 responses (one response list per attempt).
STANDARD16 = {
    F.DEV[0]: [["tool", "submit"]],                                  # submits at response 2: never builds it
    F.DEV[1]: [["tool"] * 9 + ["submit"]],                           # submits at response 10 (after the 8th)
    F.DEV[2]: [["tool"] * 14 + ["text", "tool"]],                    # a format error at 15: notice on its message
    F.DEV[3]: [["tool"] * 15 + ["limit"]],                           # ContextLimit at the 16th call
    F.DEV[4]: [["tool"] * 15 + ["infra"], ["tool"] * 15 + ["submit"]],   # an infrastructure attempt, then terminal
    F.DEV[5]: [["plain"] * 16],                                      # the empty ledger
    **{cell: [TOOLS16] for cell in F.DEV[6:]},                       # NoAnswer, 15 entries
}


class A6DevCheckTests(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.count = 0

    def tearDown(self):
        self.temporary.cleanup()

    def run_dir(self, plan=None, budget=B16, **kwargs):
        self.count += 1
        return F.write_run(self.base / f"run-{self.count}", "off", STANDARD16 if plan is None else plan,
                           budget=budget, **kwargs)

    def check(self, run_dir, expect_budget=None):
        return D.check_run(run_dir, "off", cell_name=F.DECODE, expect_budget=expect_budget)

    def test_an_a6_dev_run_passes_with_the_budget_expected(self):
        report = self.check(self.run_dir(), B16)
        self.assertEqual((report["go"], report["failed_checks"]), (True, []))
        self.assertEqual(list(report["checks"]), ["run_json_episode_config", "run_json_budget"] + list(D.CHECKS)[1:])
        self.assertEqual(report["budget"], {"expected": {"max_model_calls": 16, "context_budget": 32768},
                                            "recorded": {"max_model_calls": 16, "context_budget": 32768},
                                            "last_call_index": 15, "protocol": PROTOCOL, "passed": True})
        rule = report["go_rule"]
        # Noticed: every cell but DEV[0], DEV[1] (submitted early) and DEV[3] (ContextLimit before the 16th response).
        self.assertEqual((rule["noticed_final_responses"], rule["valid_tool_calls"], rule["format_errors"]), (7, 7, 0))
        self.assertIn("call_index 15", rule["rule"])
        self.assertNotIn("call_index 7", rule["rule"])
        self.assertIn("16th request", report["checks"]["request_bytes"]["description"])
        self.assertIn("call_index 15", report["checks"]["ledger_receipts"]["description"])
        last = report["last_call"]
        self.assertEqual((last["reached"], last["context_limit_at_last_call"], last["context_limit_cells"],
                          last["notice_on_format_error"], last["format_error_cells"]),
                         (8, 1, [F.DEV[3]], 1, [F.DEV[2]]))
        self.assertEqual((report["ledger"]["built"], report["ledger"]["empty"], report["ledger"]["entries_max"]),
                         (8, 1, 15))
        self.assertTrue(D.summary(report).startswith(f"GO: A3 off run {F.RUN_ID}: 7/7"))
        self.assertIn("budget 16/32768 (protocols/EXPA_A6_BUDGET16.md): recorded, last call_index 15",
                      D.summary(report))

    def test_the_last_call_is_derived_from_run_json(self):
        # Without --expect-budget the expected record is the prereg one: only the config check fails, and every other
        # check reads the run's own last call (call_index 15), not the constant 7.
        report = self.check(self.run_dir())
        self.assertEqual(report["failed_checks"], ["run_json_episode_config", "config_receipts"])
        self.assertNotIn("budget", report)
        self.assertEqual(report["go_rule"]["noticed_final_responses"], 7)
        self.assertEqual(report["last_call"]["reached"], 8)
        self.assertEqual(list(report["checks"]), list(D.CHECKS))

    def test_the_budget_check(self):
        # An 8-response run checked as A6: the budget and the record fail.
        report = self.check(self.run_dir(F.STANDARD, budget=None), B16)
        self.assertEqual(report["failed_checks"], ["run_json_episode_config", "run_json_budget", "config_receipts"])
        self.assertEqual(report["budget"]["recorded"], {"max_model_calls": 8, "context_budget": 16384})
        self.assertIn("NOT recorded", D.summary(report))
        # The same run with --expect-budget 8:16384 passes, with the extra (passed) check.
        report = self.check(self.run_dir(F.STANDARD, budget=None), (8, 16384))
        self.assertEqual((report["go"], report["budget"]["last_call_index"], report["budget"]["protocol"]),
                         (True, 7, "protocols/EXPA_A3_LEDGER.md"))
        # A run.json whose budget alone is wrong (16 calls with 16384 tokens).
        config = D.expected_episode_config("off", B16)
        config["context_budget"] = 16384
        report = self.check(self.run_dir(episode_config=config), B16)
        self.assertIn("run_json_budget", report["failed_checks"])
        self.assertIn("run_json_episode_config", report["failed_checks"])
        with self.assertRaises(D.InputError):
            D.DevCheck(self.base, "off", expect_budget=(12, 32768))

    def test_the_go_rule_at_call_index_15(self):
        for errors, go in ((1, True), (2, False)):
            plan = {cell: [["tool"] * 15 + (["text"] if index < errors else ["tool"])]
                    for index, cell in enumerate(F.DEV)}
            with self.subTest(errors=errors):
                report = self.check(self.run_dir(plan), B16)
                self.assertEqual((report["go"], report["failed_checks"], report["go_rule"]["valid_tool_calls"]),
                                 (go, [], 10 - errors))

    def test_a_ledger_at_another_call_fails(self):
        run_dir = self.run_dir()

        def move(rows):
            for row in rows:
                if row["kind"] == "evidence_ledger":
                    row["value"]["call_index"] = 7
        F.edit_events(F.attempt_dir(run_dir, F.DEV[6]), move)
        self.assertEqual(self.check(run_dir, B16)["failed_checks"], ["ledger_event"])
        run_dir = self.run_dir()
        F.set_ledger_receipt(F.attempt_dir(run_dir, F.DEV[6]), call_index=7)
        self.assertIn("ledger_receipts", self.check(run_dir, B16)["failed_checks"])

    def test_the_command_line(self):
        run_dir = self.run_dir()
        original = D.check_run
        with mock.patch.object(D, "check_run", lambda path, arm, **kwargs: original(path, arm, cell_name=F.DECODE,
                                                                                   **kwargs)):
            out = self.base / "go.json"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = D.main(["--run-dir", str(run_dir), "--arm", "off", "--expect-budget", "16:32768",
                               "--out", str(out)])
            self.assertEqual(code, 0)
            self.assertTrue(stdout.getvalue().startswith("GO: A3 off run"))
            self.assertEqual(json.loads(out.read_text(encoding="ascii"))["budget"]["last_call_index"], 15)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(D.main(["--run-dir", str(run_dir), "--arm", "off", "--out",
                                         str(self.base / "nogo.json")]), 1)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                D.main(["--run-dir", str(run_dir), "--arm", "off", "--expect-budget", "12:32768",
                        "--out", str(self.base / "bad.json")])
            self.assertFalse((self.base / "bad.json").exists())


# ---------------------------------------------------------------------------------------------------------------
# (6) the analysis
# ---------------------------------------------------------------------------------------------------------------

ITEMS_SHA = "1" * 64


def row(item, family, variant, seed, recovered, *, calls, budget, state="completed", exit_status=None,
        block="LEDGER", items_sha=ITEMS_SHA):
    """A SYNTHETIC scorer row (absent_edges/probe/score.py shape, the fields the analysis reads)."""
    completed = state == "completed"
    if exit_status is None:
        exit_status = ("Submitted" if recovered else "NoAnswer") if completed else "Missing"
    if calls is None and completed:
        calls = 3 if exit_status == "Submitted" else budget
    submitted = completed and exit_status == "Submitted"
    return {"schema": "expA-score-v2", "kind": "episode", "design": "a3-ledger",
            "cell_id": f"{block}|{item}|{variant}|C1|s{seed}", "block": block, "item_id": item, "family": family,
            "variant": variant, "condition": "C1", "seed": seed, "cell_state": state, "completed": completed,
            "exit_status": exit_status, "site_recovered": recovered if completed else None,
            "precision_lower": (1.0 if recovered else 0.0) if submitted else None,
            "precision_upper": (1.0 if recovered else 0.5) if submitted else None,
            "cost": {"calls": calls} if completed else None, "items_sha256": items_sha}


FAMILY_OF = {"i1": "f1", "i2": "f1", "i3": "f2", "i4": "f2"}
# recovery per (item, variant): seeds 61, 73, 89. Seed 89 is not terminal in the 16-response run, so it never counts;
# its values would change every estimate if it did.
REC8 = {(i, "vcomplete"): [1, 1, 1] for i in FAMILY_OF} | {(i, "omit_node"): [0, 0, 1] for i in FAMILY_OF}
REC16 = ({(i, "vcomplete"): [1, 1, 0] for i in FAMILY_OF}
         | {("i1", "omit_node"): [1, 1, 0], ("i2", "omit_node"): [1, 1, 0], ("i3", "omit_node"): [1, 0, 0],
            ("i4", "omit_node"): [0, 0, 0]})
OVERRIDES8 = {("i1", "omit_node", 73): {"exit_status": "ContextLimit", "calls": 7},      # ContextLimit at the last call
              ("i2", "omit_node", 73): {"exit_status": "ContextLimit", "calls": 3}}      # an earlier ContextLimit
OVERRIDES16 = {("i2", "vcomplete", 73): {"state": "missing_infrastructure"},             # excluded and counted
               ("i4", "omit_node", 61): {"exit_status": "Submitted", "calls": 16},       # a wrong answer, at the end
               ("i1", "omit_node", 89): {"state": "infra_pending"}}                      # seed 89 not terminal


def run_rows(recovery, budget, overrides):
    rows = []
    for (item, variant), values in recovery.items():
        for seed, value in zip((61, 73, 89), values):
            kwargs = {"calls": None, **overrides.get((item, variant, seed), {})}
            rows.append(row(item, FAMILY_OF[item], variant, seed, value, budget=budget, **kwargs))
    rows.append({"block": "P", "cell_id": "P|i1|none|C0|s11", "cell_state": "not_dispatched"})
    return rows


def write_rows(directory, name, rows):
    path = Path(directory) / name
    path.write_bytes(b"".join(json.dumps(r, sort_keys=True).encode() + b"\n" for r in rows))
    return str(path)


def run_json(calls, **changes):
    config = harness.EpisodeConfig(max_model_calls=calls, context_budget=32768 if calls == 16 else 16384,
                                   final_turn_policy="deadline", evidence_ledger="off").record()
    config.update(changes)
    return {"run_id": f"synthetic-{calls}", "episode_config": config}


class A6AnalysisTests(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.s16 = write_rows(self.base, "s16.jsonl", run_rows(REC16, 16, OVERRIDES16))
        self.s8 = write_rows(self.base, "s8.jsonl", run_rows(REC8, 8, OVERRIDES8))

    def tearDown(self):
        self.temporary.cleanup()

    def main(self, *argv):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = A.main(list(argv))
        return code, stdout.getvalue(), stderr.getvalue()

    def write_json(self, name, value):
        path = self.base / name
        path.write_bytes(json.dumps(value).encode())
        return str(path)

    def test_the_hand_computed_primary_and_secondaries(self):
        r16, r8 = self.write_json("r16.json", run_json(16)), self.write_json("r8.json", run_json(8))
        out = self.base / "out"
        code, stdout, _ = self.main("--scores-16", self.s16, "--scores-8", self.s8, "--run-json-16", r16,
                                    "--run-json-8", r8, "--out", str(out))
        self.assertEqual(code, 0)
        result = json.loads((out / "A6_ANALYSIS.json").read_bytes())
        tables = (out / "A6_TABLES.md").read_text()
        self.assertEqual(stdout, tables + "\n")
        self.assertEqual(result["label"], "PRESPECIFIED (protocols/EXPA_A6_BUDGET16.md, written before any A6 cell); "
                                          "ANALYSIS RECORD")
        self.assertIn(result["label"], tables)
        self.assertEqual(result["inputs"]["scores"], {
            "16": {"path": self.s16, "sha256": sha(Path(self.s16).read_bytes())},
            "8": {"path": self.s8, "sha256": sha(Path(self.s8).read_bytes())}})
        self.assertEqual((result["inputs"]["run_json"]["16"]["path"], result["inputs"]["run_json"]["16"]["sha256"],
                          result["inputs"]["run_json"]["8"]["max_model_calls"]),
                         (r16, sha(Path(r16).read_bytes()), 8))
        main = result["main"]
        self.assertEqual((main["status"], main["seeds_used"], main["items_with_all_four_values"]),
                         ("analysed", [61, 73], 4))
        self.assertEqual(main["missing_infrastructure_cells"], {"16": 1, "8": 0})
        # G8 = 1 for every item. G16 (complete 1 everywhere): i1 0, i2 0, i3 1/2, i4 1 -> f1 0, f2 3/4 -> 3/8.
        # R = G8 - G16: i1 1, i2 1, i3 1/2, i4 0 -> family means f1 1, f2 1/4 -> estimate 5/8, t = (5/8)/(3/8), df 1.
        r = main["R_reduction_in_omission_cost"]
        self.assertEqual((r["estimate_exact"], r["n_families"], r["n_items"], r["df"], r["family_means"]),
                         ("5/8", 2, 4, 1, {"f1": 1.0, "f2": 0.25}))
        self.assertAlmostEqual(r["t"], 5 / 3, places=9)
        self.assertAlmostEqual(r["p_two_sided"], 1 - 2 / math.pi * math.atan(5 / 3), places=6)   # df 1: Cauchy
        self.assertAlmostEqual(r["ci_low"], 0.625 - 12.706204736174705 * 0.375, places=4)
        decision = main["primary_decision"]
        self.assertEqual((decision["alpha"], decision["reject_null_at_alpha"], decision["direction"]),
                         (0.05, False, "the omission cost is smaller at 16 responses"))
        self.assertEqual(main["G8_omission_cost_8_responses"]["estimate_exact"], "1/1")
        self.assertEqual(main["G16_omission_cost_16_responses"]["estimate_exact"], "3/8")
        self.assertEqual(main["incomplete_16_minus_8"]["estimate_exact"], "5/8")
        self.assertEqual(main["complete_16_minus_8"]["estimate_exact"], "0/1")
        self.assertTrue(main["incomplete_gain_exceeds_complete_gain"])
        self.assertEqual({arm: value["estimate_exact"] for arm, value in main["levels"].items()},
                         {"complete_16": "1/1", "complete_8": "1/1", "incomplete_16": "5/8", "incomplete_8": "0/1"})
        # Submission: i4's wrong answer at seed 61 is a submission without recovery.
        self.assertEqual(main["submission"]["incomplete_16"]["estimate_exact"], "3/4")
        self.assertEqual(main["submission_16_minus_8"]["incomplete"]["estimate_exact"], "3/4")
        # Reached the last response (cost.calls == N, or N - 1 with ContextLimit):
        # 8: i1 [NoAnswer, ContextLimit at call 7] 1; i2 [NoAnswer, early ContextLimit] 1/2; i3, i4 1 -> 3/4, 1 -> 7/8.
        # 16: i1, i2 submitted early 0; i3 [early, NoAnswer] 1/2; i4 [submitted at 16, NoAnswer] 1 -> 0, 3/4 -> 3/8.
        reached = main["reached_last_response"]
        self.assertEqual({arm: value["estimate_exact"] for arm, value in reached.items()},
                         {"complete_16": "0/1", "complete_8": "0/1", "incomplete_16": "3/8", "incomplete_8": "7/8"})
        self.assertEqual(main["reached_last_response_16_minus_8"]["incomplete"]["estimate_exact"], "-1/2")
        # Responses used: 8: i1 (8 + 7)/2, i2 (8 + 3)/2 -> 13/2; i3, i4 8 -> 29/4. 16: f1 3; i3 (3 + 16)/2, i4 16 -> 51/4.
        used = main["responses_used"]
        self.assertEqual((used["incomplete_8"]["estimate_exact"], used["incomplete_16"]["estimate_exact"],
                          used["complete_16"]["estimate_exact"]), ("29/4", "63/8", "3/1"))
        self.assertEqual(main["responses_used_16_minus_8"]["incomplete"]["estimate_exact"], "5/8")
        self.assertEqual(main["exits"], {
            "16": {"complete": {"Submitted": 7, "ContextLimit_at_last_call": 0},
                   "incomplete": {"NoAnswer": 2, "Submitted": 6, "ContextLimit_at_last_call": 0}},
            "8": {"complete": {"Submitted": 8, "ContextLimit_at_last_call": 0},
                  "incomplete": {"ContextLimit": 2, "NoAnswer": 6, "ContextLimit_at_last_call": 1}}})
        # precision_lower over submitted cells: f1 1; i3 1, i4 0 (the wrong answer) -> f2 1/2 -> 3/4.
        self.assertEqual(main["precision"]["precision_lower"]["incomplete_16"]["estimate_exact"], "3/4")
        self.assertEqual(main["completed_cells_without_calls"], {"16": 0, "8": 0})
        self.assertIn("| R_reduction_in_omission_cost | +62.5 [", tables)
        self.assertIn("Primary: R = G8 - G16", tables)
        # Refused when the output directory exists.
        code, _, stderr = self.main("--scores-16", self.s16, "--scores-8", self.s8, "--out", str(out))
        self.assertEqual(code, 2)
        self.assertIn("exists", stderr)

    def test_refusals(self):
        def refused(*argv, message):
            out = self.base / f"refused-{len(list(self.base.iterdir()))}"
            code, _, stderr = self.main(*argv, "--out", str(out))
            self.assertEqual(code, 2, stderr)
            self.assertIn(message, stderr)
            self.assertFalse(out.exists())
        base = ("--scores-16", self.s16, "--scores-8", self.s8)
        r16, r8 = self.write_json("r16.json", run_json(16)), self.write_json("r8.json", run_json(8))
        refused(*base, "--run-json-16", r16, message="give both --run-json-16 and --run-json-8")
        refused("--scores-16", self.s16, "--scores-8", self.s8, "--run-json-16", r8, "--run-json-8", r16,
                message="not 16 and 32768")
        for name, changes, message in (
                ("policy", {"final_turn_policy": "neutral"}, "not the deadline notice"),
                ("ledger", {"evidence_ledger": "on"}, "not 'off'"),
                ("budget", {"context_budget": 16384}, "not 16 and 32768"),
                ("notice", {"final_turn_notice": {"call_index": 7}}, "final_turn_notice.call_index is not 15")):
            refused(*base, "--run-json-16", self.write_json(f"{name}.json", run_json(16, **changes)), "--run-json-8",
                    r8, message=message)
        # The runs hold different cells, or other items files; a block outside LEDGER dispatched; too many calls.
        fewer = write_rows(self.base, "fewer.jsonl", run_rows({k: v for k, v in REC16.items() if k[0] != "i4"}, 16,
                                                              OVERRIDES16))
        refused("--scores-16", fewer, "--scores-8", self.s8, message="the runs hold different LEDGER cells")
        rows = run_rows(REC16, 16, OVERRIDES16)
        rows[-1]["cell_state"] = "completed"
        refused("--scores-16", write_rows(self.base, "p.jsonl", rows), "--scores-8", self.s8,
                message="cells outside block LEDGER were dispatched")
        rows = run_rows(REC8, 8, {**OVERRIDES8, ("i3", "omit_node", 61): {"calls": 9}})
        refused("--scores-16", self.s16, "--scores-8", write_rows(self.base, "nine.jsonl", rows),
                message="exceeds the 8-response budget")
        rows = run_rows(REC8, 8, OVERRIDES8)
        for value in rows[:-1]:
            value["items_sha256"] = "2" * 64
        refused("--scores-16", self.s16, "--scores-8", write_rows(self.base, "items.jsonl", rows),
                message="different items files")

    def test_no_complete_seed(self):
        overrides = {**OVERRIDES16, ("i3", "vcomplete", 61): {"state": "not_dispatched"}}
        s16 = write_rows(self.base, "open.jsonl", run_rows(REC16, 16, overrides))
        out = self.base / "out"
        code, _, _ = self.main("--scores-16", s16, "--scores-8", self.s8, "--out", str(out))
        self.assertEqual(code, 0)
        result = json.loads((out / "A6_ANALYSIS.json").read_bytes())
        self.assertEqual((result["main"]["status"], result["main"]["seeds_used"]), ("no_complete_seed", []))
        self.assertIn("Status: no_complete_seed", (out / "A6_TABLES.md").read_text())

    def test_the_protocol_names_the_script_and_the_rules(self):

        if not (__import__("pathlib").Path(__file__).resolve().parents[1] / "protocols").is_dir():

            self.skipTest("the protocol texts ship in the dataset package (protocol/)")
        protocol = (PROJECT / PROTOCOL).read_text(encoding="utf-8")
        self.assertIn("scripts/expA_a6_analysis.py", protocol)
        self.assertIn("R = G8 − G16", protocol)
        self.assertIn("longest prefix of 61, 73, 89", protocol)
        self.assertEqual(A.SEEDS, (61, 73, 89))
        self.assertEqual((A.BUDGETS, A.ALPHA), ({16: 32768, 8: 16384}, 0.05))


if __name__ == "__main__":
    unittest.main()
