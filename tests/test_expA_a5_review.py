"""Exp A follow-up A5: the submit-time review (protocols/EXPA_A5_REVIEW.md). SYNTHETIC fixtures only.

(1) The constants, config records and validation (any OS): the default, A1 and A3 records are byte-identical to the
    ones before A5 (sha256 pinned from the commit before A4); neutral and ledger record the option and every frozen
    rule; the review needs the deadline notice and no evidence ledger; the review text builder.
(2) The harness against the fake server (POSIX): the review is shown once, at the first well-formed submission before
    the last call; a resubmission is accepted; without one the first answer is used (exit Submitted); a first
    submission at the last call is accepted without review; the deadline notice still follows the last tool result
    (the review, when it is that result); a rejected submission is not reviewed; the receipts.
(3) The scorer (POSIX): result.json submitted is scored; the first submission is scored into submit_review; a row
    without the receipt is unchanged.
(4) The runner and the launch script (POSIX): registration, block LEDGER only, dry run, run.json, every episode.
(5) The dev check (scripts/expA_a5_devcheck.py) on runner records and on edited copies (POSIX).
(6) The analysis (scripts/expA_a5_analysis.py) on a hand-computed fixture (any OS).
ANALYSIS RECORD.
"""
import collections
import contextlib
import copy
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

from absent_edges.data import canonical_json  # noqa: E402
from absent_edges.probe import harness  # noqa: E402
from absent_edges.probe.tools import LEXICAL_KIND  # noqa: E402

POSIX = os.name == "posix"
if POSIX:
    import test_expA_default_golden as G      # puts the project on sys.path
    import test_probe_harness as H
    import test_probe_score as PS
    from absent_edges.models import LocalModel, ModelProfile
    from absent_edges.probe import ledger as L
    from absent_edges.probe import score as S
    from absent_edges.probe.tools import ToolBox


def load_script(name):
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, PROJECT / "scripts" / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module


D = load_script("expA_a5_devcheck")
A = load_script("expA_a5_analysis")

NOTICE = "[harness notice] "
NEUTRAL = "Your answer was received. Before finishing, call submit_answer again with your final list."
INSTRUCTION = "Before finishing, review these candidate call sites, then call submit_answer again with your final list."
HEADER = "Candidate call sites seen in your earlier tool outputs (unverified; check before submitting):"
DEADLINE = "\n\n[harness notice] This is your last response. Call submit_answer now."
FROZEN_CONFIG = {"max_model_calls": 8, "context_budget": 16384, "max_response_tokens": 4096,
                 "min_response_tokens": 512, "temperature": 0.2, "max_consecutive_format_errors": 3}
PINNED = {"none": "ff8921f6e5ef35bfbacc58037ac23081468f43a9404ea8a62ed174f1ffd9df4b",
          "neutral": "2aab8a084aedf3c705a68e9600fd61aeccc68364e9aa2a07a93fe989ca9b82a4",
          "deadline": "773cb89af25df5d771e6248f7d0d45246f35045c7a803a2a8f1a9a956bb4819b",
          "a3_off": "b1d564984b0012b8d7cff6649131da96375d38577825a6ccdd64b9a9154c00a1",
          "a3_on": "c4ae1e0f20a92e0c49b574b9826e3f541fbe735823a45ef37b2240f56cfa1ef8",
          "design_default": "6b0a49cd8eb8e6157e2d8d516f0ca2231e92b4e494c178b1c28af838ebdedb5c",
          "design_a3": "723f7aa95dd4494c2b174ac09ddc7aa353a52dcbe48297f47354ac8a76c44d0c"}
BASH = {"calls": [("bash", {"command": "grep -rn target ."})]}
SEARCH = {"calls": [("source_reference_search", {"query": "pkg/core.py::target"})]}
FIRST = {"sites": [{"path": "pkg/core.py", "line": 6}], "complete": "yes"}
SECOND = {"sites": [{"path": "pkg/core.py", "line": 6}, {"path": "pkg/core.py", "line": 11}], "complete": "no"}
SUBMIT1 = {"calls": [("submit_answer", FIRST)]}
SUBMIT2 = {"calls": [("submit_answer", SECOND)]}
MALFORMED = {"calls": [("submit_answer", {"sites": "pkg/core.py:6", "complete": "yes"})]}
GREP = ("pkg/core.py:1:def target(x):\n"
        "pkg/core.py:6:    return target(1)\n")
GREP_LEDGER = HEADER + "\n- pkg/core.py:6: return target(1)"
Run = collections.namedtuple("Run", "result server rows")


def sha(text):
    return hashlib.sha256(text if isinstance(text, bytes) else text.encode("utf-8")).hexdigest()


def digest(value):
    return sha(json.dumps(value, ensure_ascii=False))


# ---------------------------------------------------------------------------------------------------------------
# (1) constants, records, validation
# ---------------------------------------------------------------------------------------------------------------

class ReviewConfigTests(unittest.TestCase):
    maxDiff = None

    def test_the_frozen_texts(self):
        self.assertEqual((harness.SUBMIT_REVIEW_DEFAULT, harness.SUBMIT_REVIEW_MODES),
                         ("none", ("none", "neutral", "ledger")))
        self.assertEqual((harness.SUBMIT_REVIEW_PREFIX, harness.SUBMIT_REVIEW_NEUTRAL, harness.SUBMIT_REVIEW_INSTRUCTION,
                          harness.SUBMIT_REVIEW_PROTOCOL),
                         (NOTICE, NEUTRAL, INSTRUCTION, "protocols/EXPA_A5_REVIEW.md"))
        self.assertFalse(any(text in json.dumps(harness.prompt_templates()) for text in (NEUTRAL, INSTRUCTION)))
        # The dev check declares them again (it does not import the harness): the same bytes.
        self.assertEqual((D.PREFIX, D.NEUTRAL, D.INSTRUCTION, D.PROTOCOL, D.HEADER),
                         (NOTICE, NEUTRAL, INSTRUCTION, harness.SUBMIT_REVIEW_PROTOCOL, harness.LEDGER_HEADER))

    def test_earlier_records_are_byte_identical(self):
        configs = {"none": harness.EpisodeConfig(), "neutral": harness.EpisodeConfig(final_turn_policy="neutral"),
                   "deadline": harness.EpisodeConfig(final_turn_policy="deadline"),
                   "a3_off": harness.EpisodeConfig(final_turn_policy="deadline", evidence_ledger="off"),
                   "a3_on": harness.EpisodeConfig(final_turn_policy="deadline", evidence_ledger="on")}
        for name, config in configs.items():
            with self.subTest(name=name):
                self.assertEqual(config.submit_review, "none")
                self.assertEqual(digest(config.record()), PINNED[name])
        self.assertEqual(harness.EpisodeConfig(submit_review="none"), harness.EpisodeConfig())
        self.assertEqual(digest(harness.design_table_for(None)), PINNED["design_default"])
        self.assertEqual(digest(harness.design_table_for("a3-ledger")), PINNED["design_a3"])
        self.assertIsNone(harness.config_design(harness.EpisodeConfig(final_turn_policy="deadline")))

    def test_an_arm_records_the_option_and_every_rule(self):
        for mode in ("neutral", "ledger"):
            with self.subTest(mode=mode):
                config = harness.EpisodeConfig(final_turn_policy="deadline", submit_review=mode)
                record = config.record()
                self.assertEqual(list(record), list(FROZEN_CONFIG) + ["final_turn_policy", "final_turn_notice",
                                                                      "submit_review", "submit_review_rules"])
                self.assertEqual({k: v for k, v in record.items() if not k.startswith("submit_review")},
                                 harness.EpisodeConfig(final_turn_policy="deadline").record())
                rules = record["submit_review_rules"]
                self.assertEqual((rules["protocol"], rules["mode"], rules["channel"], rules["prefix"],
                                  rules["before_call_index"]),
                                 ("protocols/EXPA_A5_REVIEW.md", mode, "tool_result", NOTICE, 7))
                if mode == "neutral":
                    self.assertEqual((rules["text"], rules["text_sha256"]), (NEUTRAL, sha(NEUTRAL)))
                else:
                    self.assertEqual((rules["instruction"], rules["header"], rules["max_entries"], rules["max_bytes"]),
                                     (INSTRUCTION, HEADER, 20, 2000))
                self.assertEqual(record, D.expected_episode_config(mode))     # the dev check rebuilds it exactly
                self.assertEqual(harness.config_design(config), "a3-ledger")
                json.dumps(record)

    def test_the_review_needs_the_deadline_notice_and_no_ledger(self):
        for bad in ("on", "Neutral", "", None, 1):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                harness.EpisodeConfig(final_turn_policy="deadline", submit_review=bad)
        for kwargs in ({}, {"final_turn_policy": "neutral"}, {"final_turn_policy": "submit_only"},
                       {"final_turn_policy": "deadline", "evidence_ledger": "off"},
                       {"final_turn_policy": "deadline", "evidence_ledger": "on"}):
            for mode in ("neutral", "ledger"):
                with self.subTest(kwargs=kwargs, mode=mode), \
                        self.assertRaisesRegex(ValueError, "deadline notice and no last-call evidence ledger"):
                    harness.EpisodeConfig(submit_review=mode, **kwargs)

    def test_the_review_text(self):
        bash = {"role": "tool", "tool_call_id": "call-0-0",
                "content": canonical_json({"exit_code": 0, "output": GREP, "truncated": False}).decode()}
        search = {"role": "tool", "tool_call_id": "call-1-0", "content": canonical_json(
            {"query": "target", "kind": LEXICAL_KIND, "truncated": False, "error": None,
             "hits": [{"path": "pkg/b.py", "line": 3, "text": "    x = target(2)"}]}).decode()}
        history = [{"role": "system", "content": "s"}, {"role": "user", "content": "pkg/u.py:1:  target(9)"},
                   {"role": "assistant", "content": None}, bash, search]
        self.assertEqual(harness.submit_review_text("neutral", history, "pkg/core.py::target"), (NEUTRAL, None))
        text, built = harness.submit_review_text("ledger", history, "pkg/core.py::target")
        self.assertEqual(text, GREP_LEDGER + "\n- pkg/b.py:3: x = target(2)\n" + INSTRUCTION)
        self.assertEqual(built["shown"], 2)
        text, built = harness.submit_review_text("ledger", history[:3], "pkg/core.py::target")
        self.assertEqual((text, built["shown"]), (HEADER + "\n(none)\n" + INSTRUCTION, 0))
        with self.assertRaises(ValueError):
            harness.submit_review_text("none", history, "pkg/core.py::target")


# ---------------------------------------------------------------------------------------------------------------
# (2) the harness
# ---------------------------------------------------------------------------------------------------------------

@unittest.skipUnless(POSIX, "the ToolBox search scans with POSIX no-follow reads")
class ReviewEpisodeTests(unittest.TestCase):
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

    def run_arm(self, script, mode, *, output=GREP, policy="deadline"):
        server = H.FakeServer(script, default=SEARCH)
        self.servers.append(server)
        self.count += 1
        directory = Path(self.temporary.name) / f"attempt-{self.count}"
        directory.mkdir()
        client = LocalModel(ModelProfile(endpoint=server.endpoint), timeout=20)
        toolbox = ToolBox(self.root, ["pkg"], H.FakeBash(output))
        cell = "LEDGER|item-h|omit_node|C1|s61"
        episode = H.make_episode("C1", block="LEDGER", variant="omit_node", cell_id=cell, attempt_id=cell + ".a001",
                                 seed=61)
        config = harness.EpisodeConfig(final_turn_policy=policy, submit_review=mode)
        result, digest_ = harness.run_episode(episode, directory, client=client, toolbox=toolbox, config=config)
        self.assertEqual(digest_, L.sha256_path(directory / "result.json"))
        return Run(result, server, H.events(directory))

    @staticmethod
    def values(run, kind):
        return [row["value"] for row in run.rows if row["kind"] == kind]

    def receipt(self, run):
        receipt = run.result["accounting"]["submit_review"]
        self.assertEqual(run.result["exposure"]["submit_review"], receipt)
        self.assertEqual(self.values(run, "episode_finished")[0]["accounting"]["submit_review"], receipt)
        self.assertEqual(set(receipt), set(D.RECEIPT_FIELDS))
        return receipt

    def review_text(self, mode):
        return NEUTRAL if mode == "neutral" else GREP_LEDGER + "\n" + INSTRUCTION

    def test_the_review_is_shown_once_and_a_resubmission_is_accepted(self):
        for mode in ("neutral", "ledger"):
            with self.subTest(mode=mode):
                run = self.run_arm([BASH, SUBMIT1, SUBMIT1, SUBMIT2], mode)
                text = self.review_text(mode)
                self.assertEqual((run.result["exit_status"], run.result["calls"]), ("Submitted", 3))
                self.assertEqual(run.result["submitted"], FIRST)          # the resubmission (here the same list)
                reviews = self.values(run, "submit_review")
                self.assertEqual(len(reviews), 1)
                event = reviews[0]
                self.assertEqual((event["call_index"], event["tool_call_id"], event["mode"], event["arguments"],
                                  event["submission"], event["text"], event["content"]),
                                 (1, "call-2-0", mode, FIRST, FIRST, text, NOTICE + text))
                self.assertEqual("ledger" in event, mode == "ledger")
                accepted = self.values(run, "submission_accepted")
                self.assertEqual([(a["call_index"], a["submission"]) for a in accepted], [(2, FIRST)])
                receipt = self.receipt(run)
                self.assertEqual({k: receipt[k] for k in ("reviewed", "review_call_index", "review_tool_call_id",
                                                          "first_call_index", "first_submission", "answer_source",
                                                          "resubmit_call_index", "resubmit_tool_call_id",
                                                          "unresolved_exit", "review_skipped_last_call",
                                                          "text_sha256", "content_sha256")},
                                 {"reviewed": True, "review_call_index": 1, "review_tool_call_id": "call-2-0",
                                  "first_call_index": 1, "first_submission": FIRST, "answer_source": "resubmit",
                                  "resubmit_call_index": 2, "resubmit_tool_call_id": "call-3-0",
                                  "unresolved_exit": None, "review_skipped_last_call": False,
                                  "text_sha256": sha(text), "content_sha256": sha(NOTICE + text)})
                self.assertEqual((receipt["ledger_entries"], receipt["ledger_body_bytes"]),
                                 (1, len("- pkg/core.py:6: return target(1)")) if mode == "ledger" else (None, None))
                chats = run.server.chat_requests()
                self.assertNotIn(NOTICE, json.dumps(chats[:2]))
                last = chats[2]["messages"][-1]
                self.assertEqual((last["role"], last["tool_call_id"], last["content"]),
                                 ("tool", "call-2-0", NOTICE + text))
                self.assertEqual(run.result["tool_calls"]["submit_answer"], 2)
                self.assertEqual(run.result["accounting"]["rejected_submissions"], 0)

    def test_a_different_resubmission_is_the_answer(self):
        run = self.run_arm([SUBMIT1, SUBMIT2], "neutral")
        self.assertEqual((run.result["exit_status"], run.result["submitted"]), ("Submitted", SECOND))
        self.assertEqual(self.receipt(run)["first_submission"], FIRST)

    def test_without_a_resubmission_the_first_answer_is_used(self):
        for mode in ("neutral", "ledger"):
            with self.subTest(mode=mode):
                run = self.run_arm([BASH, SUBMIT1] + [SEARCH] * 6, mode)
                self.assertEqual((run.result["exit_status"], run.result["calls"], run.result["submitted"]),
                                 ("Submitted", 8, FIRST))
                receipt = self.receipt(run)
                self.assertEqual((receipt["answer_source"], receipt["unresolved_exit"], receipt["reviewed"]),
                                 ("first_submit", "NoAnswer", True))
                self.assertEqual(self.values(run, "submission_accepted"), [])
                self.assertEqual(self.values(run, "submit_review_resolved"),
                                 [{"answer_source": "first_submit", "unresolved_exit": "NoAnswer", "submission": FIRST}])
                self.assertEqual(self.values(run, "episode_finished")[0]["exit_status"], "Submitted")
                self.assertEqual(self.values(run, "episode_finished")[0]["submission"], FIRST)
                # The deadline notice still goes on the last tool result before the 8th response.
                self.assertTrue(run.server.chat_requests()[7]["messages"][-1]["content"].endswith(DEADLINE))
                self.assertTrue(run.result["exposure"]["final_turn_notice"]["delivered"])

    def test_a_format_exhausted_episode_after_the_review_uses_the_first_answer(self):
        text = {"content": "no tool call"}
        run = self.run_arm([SUBMIT1, text, text, text], "neutral")
        self.assertEqual((run.result["exit_status"], run.result["submitted"]), ("Submitted", FIRST))
        self.assertEqual(self.receipt(run)["unresolved_exit"], "FormatExhausted")

    def test_a_review_at_response_7_carries_the_deadline_notice(self):
        for mode in ("neutral", "ledger"):
            with self.subTest(mode=mode):
                run = self.run_arm([BASH] * 6 + [SUBMIT1, SEARCH], mode)
                text = self.review_text(mode)
                last = run.server.chat_requests()[7]["messages"][-1]
                self.assertEqual((last["tool_call_id"], last["content"]), ("call-7-0", NOTICE + text + DEADLINE))
                notice = run.result["exposure"]["final_turn_notice"]
                self.assertEqual((notice["appended_to"], notice["tool_call_id"]), ("tool_result", "call-7-0"))
                receipt = self.receipt(run)
                self.assertEqual((receipt["review_call_index"], receipt["answer_source"], receipt["unresolved_exit"]),
                                 (6, "first_submit", "NoAnswer"))

    def test_a_first_submission_at_the_last_call_is_accepted_without_review(self):
        for mode in ("neutral", "ledger"):
            with self.subTest(mode=mode):
                run = self.run_arm([BASH] * 7 + [SUBMIT1], mode)
                self.assertEqual((run.result["exit_status"], run.result["calls"], run.result["submitted"]),
                                 ("Submitted", 8, FIRST))
                self.assertEqual(self.values(run, "submit_review"), [])
                receipt = self.receipt(run)
                self.assertEqual({k: receipt[k] for k in ("reviewed", "review_skipped_last_call", "first_call_index",
                                                          "first_submission", "answer_source", "unresolved_exit")},
                                 {"reviewed": False, "review_skipped_last_call": True, "first_call_index": 7,
                                  "first_submission": FIRST, "answer_source": "first_submit",
                                  "unresolved_exit": None})
                self.assertNotIn(NOTICE + NEUTRAL, json.dumps(run.server.chat_requests()))

    def test_a_second_submission_in_the_reviewed_response_is_the_resubmission(self):
        run = self.run_arm([{"calls": [("submit_answer", FIRST), ("submit_answer", SECOND)]}], "neutral")
        self.assertEqual((run.result["exit_status"], run.result["calls"], run.result["submitted"]),
                         ("Submitted", 1, SECOND))
        receipt = self.receipt(run)
        self.assertEqual((receipt["review_call_index"], receipt["resubmit_call_index"], receipt["answer_source"]),
                         (0, 0, "resubmit"))

    def test_a_rejected_submission_is_not_reviewed(self):
        run = self.run_arm([MALFORMED, SUBMIT1, SUBMIT2], "neutral")
        self.assertEqual((run.result["exit_status"], run.result["submitted"]), ("Submitted", SECOND))
        receipt = self.receipt(run)
        self.assertEqual((receipt["review_call_index"], receipt["resubmit_call_index"]), (1, 2))
        self.assertEqual(run.result["accounting"]["rejected_submissions"], 1)

    def test_an_episode_without_a_submission(self):
        run = self.run_arm([SEARCH] * 8, "ledger")
        self.assertEqual((run.result["exit_status"], run.result["submitted"]), ("NoAnswer", None))
        receipt = self.receipt(run)
        self.assertEqual({k: receipt[k] for k in ("reviewed", "answer_source", "first_submission", "unresolved_exit")},
                         {"reviewed": False, "answer_source": None, "first_submission": None,
                          "unresolved_exit": None})

    def test_an_unreviewed_episode_sends_the_deadline_requests(self):
        """Without a review the arm changes no request: the bytes of the deadline harness (A1, A3 off)."""
        for script in ([BASH] * 7 + [SUBMIT1], [SEARCH] * 8):
            base = self.run_arm(script, "none")
            for mode in ("neutral", "ledger"):
                with self.subTest(mode=mode, script=len(script)):
                    run = self.run_arm(script, mode)
                    self.assertEqual(run.server.chat_requests(), base.server.chat_requests())
                    self.assertEqual((run.result["exit_status"], run.result["submitted"]),
                                     (base.result["exit_status"], base.result["submitted"]))
                    exposure = dict(run.result["exposure"])
                    exposure.pop("submit_review")
                    self.assertEqual(exposure, base.result["exposure"])
                    self.assertNotIn("submit_review", base.result["accounting"])
                    self.assertNotIn("submit_review", [row["kind"] for row in base.rows])


# ---------------------------------------------------------------------------------------------------------------
# (3) the scorer
# ---------------------------------------------------------------------------------------------------------------

@unittest.skipUnless(POSIX, "attempt directories are named by cell id (contains '|')")
class ReviewScoreTests(unittest.TestCase):
    def test_the_answer_used_and_the_first_submission_are_scored(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = PS.Fixture(Path(directory), run_json_extra={"design": harness.design_table_for("a3-ledger")})
            hit, miss = [{"path": "pkg/other.py", "line": 5}], [{"path": "pkg/core.py", "line": 1}]
            receipt = {"mode": "neutral", "reviewed": True, "review_call_index": 2, "review_skipped_last_call": False,
                       "first_call_index": 2, "first_submission": {"sites": miss, "complete": "yes"},
                       "answer_source": "resubmit", "resubmit_call_index": 3, "unresolved_exit": None,
                       "ledger_candidates": None, "ledger_entries": None, "ledger_body_bytes": None}
            fixture.result("LEDGER|item-a|omit_node|C1|s61", 1, "Submitted", hit, "yes",
                           accounting={"model_calls": 4, "submit_review": receipt})
            fixture.result("LEDGER|item-a|vcomplete|C1|s61", 1, "Submitted", hit, "yes")
            rows, audit = fixture.score()
            by = {row["cell_id"]: row for row in rows}
            reviewed = by["LEDGER|item-a|omit_node|C1|s61"]
            self.assertEqual(reviewed["site_recovered"], 1)            # the answer used (resubmission)
            self.assertEqual(reviewed["submit_review"]["first_submission"]["site_recovered"], 0)
            self.assertEqual({k: reviewed["submit_review"][k] for k in ("answer_source", "reviewed", "mode")},
                             {"answer_source": "resubmit", "reviewed": True, "mode": "neutral"})
            self.assertNotIn("submit_review", by["LEDGER|item-a|vcomplete|C1|s61"])
            self.assertNotIn("submit_review", by["LEDGER|item-b|omit_node|C1|s61"])


# ---------------------------------------------------------------------------------------------------------------
# (4) the runner and the launch script
# ---------------------------------------------------------------------------------------------------------------

@unittest.skipUnless(POSIX, "the runner reads checkouts with POSIX no-follow reads")
class ReviewRunnerTests(unittest.TestCase):
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
        self.assertEqual(self.runner.REGISTERED_REVIEW_DESIGNS, {
            (8, 16384, "deadline", "none", "neutral"): ("protocols/EXPA_A5_REVIEW.md", ("gemma4-31b",)),
            (8, 16384, "deadline", "none", "ledger"): ("protocols/EXPA_A5_REVIEW.md", ("gemma4-31b",))})
        self.assertEqual(self.parse("--submit-review", "none").episode_config, harness.EpisodeConfig())
        for mode in ("neutral", "ledger"):
            expected = harness.EpisodeConfig(final_turn_policy="deadline", submit_review=mode)
            arm = ("--final-turn-policy", "deadline", "--submit-review", mode)
            for selection in (("--blocks", "LEDGER"), ("--blocks", "LEDGER", "--seeds", "61"),
                              ("--blocks", "LEDGER", "--seeds", "61,73")):
                with self.subTest(mode=mode, selection=selection):
                    args = self.parse(*arm, *selection)
                    self.assertEqual((args.episode_config, args.design, args.blocks),
                                     (expected, "a3-ledger", {"LEDGER"}))
            registered = (f"--submit-review {mode} (protocols/EXPA_A5_REVIEW.md) is registered only with "
                          "--final-turn-policy deadline, --evidence-ledger none")
            for bad in ((), ("--final-turn-policy", "neutral"), ("--final-turn-policy", "deadline", "--evidence-ledger",
                                                                 "off")):
                with self.subTest(mode=mode, bad=bad):
                    self.assertIn(registered, self.refused("--submit-review", mode, "--blocks", "LEDGER", *bad))
            selection = (f"--submit-review {mode} (protocols/EXPA_A5_REVIEW.md) runs block LEDGER only, at seeds "
                         "61, 73, 89")
            for bad in ((), ("--blocks", "P", "--seeds", "11"), ("--blocks", "LEDGER,P"),
                        ("--blocks", "LEDGER", "--seeds", "11")):
                with self.subTest(mode=mode, bad=bad):
                    self.assertIn(selection, self.refused(*arm, *bad))
        self.assertIn("invalid choice", self.refused("--submit-review", "on"))

    def test_dry_run_run_json_and_every_episode(self):
        argv = ["--dispatch", str(self.dispatch), "--dry-run", "--blocks", "LEDGER", "--final-turn-policy", "deadline",
                "--submit-review", "ledger"]
        code, output = G.invoke_runner(self.runner, argv)
        summary = json.loads(output)
        self.assertEqual((code, summary["selected_cells"], summary["design"]),
                         (0, 12, harness.design_table_for("a3-ledger")))
        record = harness.EpisodeConfig(final_turn_policy="deadline", submit_review="ledger").record()
        self.assertEqual(summary["episode_config"], record)
        server = H.FakeServer([SUBMIT1, SUBMIT2], default=SEARCH)
        self.servers.append(server)
        live = G.live_args(self.base, self.dispatch, self.manifest, server, run="a5")
        live[live.index("--blocks") + 1] = "LEDGER"
        live[live.index("--seeds") + 1] = "61"
        with mock.patch.object(G.tools, "ReadOnlyBash", G.FakeJailBash):
            code, _ = G.invoke_runner(self.runner, live + ["--final-turn-policy", "deadline", "--submit-review",
                                                           "ledger"])
        self.assertEqual(code, 0)
        run_json = json.loads((self.base / "a5" / "run.json").read_bytes())
        self.assertEqual((run_json["episode_config"], run_json["design"]["name"]), (record, "a3-ledger"))
        result = json.loads(next((self.base / "a5" / "cells").rglob("result.json")).read_bytes())
        self.assertEqual((result["block"], result["exit_status"], result["submitted"], result["config"]),
                         ("LEDGER", "Submitted", SECOND, record))
        self.assertEqual(result["exposure"]["submit_review"]["answer_source"], "resubmit")


@unittest.skipUnless(POSIX and shutil.which("bash"), "Linux only: the launch script runs on Core1")
class ReviewLaunchScriptTests(unittest.TestCase):
    def host(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        host = G.LaunchHost(directory.name)
        host.run(*G.PREPARE_ARGV, "--runtime-python", str(host.python))
        host.stage("golden1")
        return host

    def runner(self, host, *argv, check=True):
        result = host.run("runner", "--run", "golden1", *argv, check=check)
        return result, [host.normalize(call) for call in G.units(host.sudo_calls(), "runner")]

    def test_an_arm_is_forwarded_only_when_registered(self):
        host = self.host()
        for mode in ("neutral", "ledger"):
            with self.subTest(mode=mode):
                result, units = self.runner(host, "--blocks", "LEDGER", "--seeds", "61", "--final-turn-policy",
                                            "deadline", "--submit-review", mode)
                self.assertEqual(len(units), 1)
                self.assertEqual(units[0][-2:], ["--submit-review", mode])
                self.assertIn(f"submit review {mode} (protocols/EXPA_A5_REVIEW.md)", result.stderr)
                tail = units[0][next(i for i, v in enumerate(units[0]) if v.endswith("/scripts/expA_run.py")) + 1:]
                with contextlib.redirect_stderr(io.StringIO()):
                    args = G.load_script("expA_run").parse_args(tail)
                self.assertEqual(args.episode_config,
                                 harness.EpisodeConfig(final_turn_policy="deadline", submit_review=mode))
            for bad, message in (((), "registered only with --final-turn-policy deadline"),
                                 (("--final-turn-policy", "deadline", "--evidence-ledger", "off"),
                                  "registered only with --final-turn-policy deadline"),
                                 (("--final-turn-policy", "deadline", "--blocks", "P", "--seeds", "11"),
                                  "block LEDGER only")):
                with self.subTest(mode=mode, bad=bad):
                    extra = () if "--blocks" in bad else ("--blocks", "LEDGER")
                    result, units = self.runner(host, *extra, *bad, "--submit-review", mode, check=False)
                    self.assertEqual((result.returncode, units), (2, []))
                    self.assertIn(f"--submit-review {mode}: {message}", result.stderr)
        result, units = self.runner(host, "--submit-review", "on", check=False)
        self.assertEqual((result.returncode, units), (2, []))
        self.assertIn("--submit-review must be none, neutral or ledger, not on", result.stderr)


# ---------------------------------------------------------------------------------------------------------------
# (5) the dev check
# ---------------------------------------------------------------------------------------------------------------

# The four LEDGER seed-61 cells run in dispatch order with one worker; the fake server's script covers them in turn:
# a review then a resubmission; a review and no resubmission (NoAnswer -> the first answer); a first submission at the
# last call; no submission at all.
SCRIPT = [SEARCH, SUBMIT1, SUBMIT2] + [SUBMIT1] + [SEARCH] * 7 + [SEARCH] * 7 + [SUBMIT1] + [SEARCH] * 8


@unittest.skipUnless(POSIX, "the runner reads checkouts with POSIX no-follow reads")
class ReviewDevCheckTests(unittest.TestCase):
    maxDiff = None

    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.base = Path(cls.temporary.name)
        runner, dispatch, manifest = G.runner_fixture(cls.base)
        cls.runs = {}
        for mode in ("neutral", "ledger"):
            server = H.FakeServer(list(SCRIPT), default=SEARCH)
            try:
                argv = G.live_args(cls.base, dispatch, manifest, server, run=mode)
                argv[argv.index("--limit") + 1] = "10"
                argv[argv.index("--blocks") + 1] = "LEDGER"
                argv[argv.index("--seeds") + 1] = "61"
                with mock.patch.object(G.tools, "ReadOnlyBash", G.FakeJailBash):
                    code, _ = G.invoke_runner(runner, argv + ["--final-turn-policy", "deadline", "--submit-review",
                                                              mode])
            finally:
                server.close()
            cls.runs[mode] = (code, cls.base / mode)

    @classmethod
    def tearDownClass(cls):
        for path in cls.base.rglob("*"):
            if path.is_dir() and not path.is_symlink():
                path.chmod(0o755)
        cls.temporary.cleanup()

    def copy(self, mode):
        target = Path(tempfile.mkdtemp(dir=self.base)) / mode
        shutil.copytree(self.runs[mode][1], target)
        for path in target.rglob("*"):
            path.chmod(0o755 if path.is_dir() else 0o644)
        return target

    @staticmethod
    def attempt(run_dir, predicate):
        for result_path in sorted((run_dir / "cells").rglob("result.json")):
            result = json.loads(result_path.read_bytes())
            if predicate(result):
                return result_path.parent
        raise AssertionError("no such attempt")

    @staticmethod
    def edit(directory, change_result=None, change_events=None):
        rows = [json.loads(line) for line in (directory / "events.jsonl").read_bytes().splitlines()]
        if change_events:
            rows = change_events(rows) or rows
        for index, row in enumerate(rows):
            row["sequence"] = index
        raw = b"".join(json.dumps(row).encode("utf-8") + b"\n" for row in rows)
        (directory / "events.jsonl").write_bytes(raw)
        result = json.loads((directory / "result.json").read_bytes())
        result["events"] = dict(result["events"], sha256=hashlib.sha256(raw).hexdigest(), records=len(rows))
        if change_result:
            change_result(result)
        (directory / "result.json").write_bytes(canonical_json(result))

    def test_the_release_records_pass(self):
        for mode in ("neutral", "ledger"):
            with self.subTest(mode=mode):
                code, run_dir = self.runs[mode]
                self.assertEqual(code, 0)
                report = D.check_run(run_dir, mode)
                self.assertEqual((report["go"], report["failed_checks"]), (True, []))
                self.assertEqual((report["go_rule"]["received_review"], report["go_rule"]["valid_tool_calls"]), (2, 2))
                review = report["review"]
                self.assertEqual((review["terminal_attempts"], review["reviewed"], review["review_skipped_last_call"],
                                  review["never_submitted"]), (4, 2, 1, 1))
                self.assertEqual(review["answer_source"], {"None": 1, "first_submit": 2, "resubmit": 1})
                self.assertEqual(review["unresolved_exits"], {"NoAnswer": 1})
                self.assertEqual(report["cells"]["terminal_exits"], {"NoAnswer": 1, "Submitted": 3})
                self.assertEqual({name: entry["checked"] > 0 for name, entry in report["checks"].items()},
                                 dict.fromkeys(D.CHECKS, True))
                self.assertEqual("ledger" in report, mode == "ledger")
                # The script's receipt fields are the harness's.
                result = json.loads(next((run_dir / "cells").rglob("result.json")).read_bytes())
                self.assertEqual(set(result["exposure"]["submit_review"]), set(D.RECEIPT_FIELDS))

    def test_the_wrong_arm_and_edited_records_fail(self):
        report = D.check_run(self.runs["neutral"][1], "ledger")
        self.assertFalse(report["go"])
        self.assertTrue({"run_json_episode_config", "config_receipts", "review_receipts"}
                        <= set(report["failed_checks"]))

        def unresolved(result):
            return (result["exposure"].get("submit_review") or {}).get("unresolved_exit") == "NoAnswer"

        def resubmitted(result):
            return (result["exposure"].get("submit_review") or {}).get("answer_source") == "resubmit"
        cases = {
            "resolved event removed": (unresolved, None, lambda rows: [r for r in rows
                                                                       if r["kind"] != "submit_review_resolved"],
                                       "answer_source"),
            "exposure differs": (resubmitted, lambda result: result["exposure"]["submit_review"].update(
                answer_source="first_submit"), None, "review_receipts"),
            "review text": (resubmitted, None, lambda rows: [dict(r, value=dict(r["value"], text="SYNTHETIC"))
                                                            if r["kind"] == "submit_review" else r for r in rows],
                            "review_event"),
            "submitted is not the resubmission": (resubmitted, lambda result: result.update(submitted=FIRST), None,
                                                  "answer_source"),
            "a second review": (resubmitted, None, lambda rows: rows[:-1] + [r for r in rows
                                                                            if r["kind"] == "submit_review"]
                                + rows[-1:], "review_event"),
        }
        for name, (which, change_result, change_events, check) in cases.items():
            with self.subTest(case=name):
                run_dir = self.copy("neutral")
                self.edit(self.attempt(run_dir, which), change_result, change_events)
                report = D.check_run(run_dir, "neutral")
                self.assertFalse(report["go"])
                self.assertIn(check, report["failed_checks"])

    def test_the_command_line(self):
        out = self.base / "a5-devcheck.json"
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = D.main(["--run-dir", str(self.runs["ledger"][1]), "--arm", "ledger", "--out", str(out)])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out.read_bytes())["arm"], "ledger")
        self.assertIn("GO: A5 ledger", stdout.getvalue())
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(D.main(["--run-dir", str(self.runs["neutral"][1]), "--arm", "ledger", "--out",
                                     str(self.base / "no-go.json")]), 1)
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(D.main(["--run-dir", str(self.runs["ledger"][1]), "--arm", "ledger", "--out",
                                     str(out)]), 2)       # never overwritten


# ---------------------------------------------------------------------------------------------------------------
# (6) the analysis
# ---------------------------------------------------------------------------------------------------------------

ITEMS_SHA = "1" * 64
FAMILY_OF = {"i1": "f1", "i2": "f1", "i3": "f2"}


def score_row(item, variant, seed, recovered, *, state="completed", review=None):
    completed = state == "completed"
    exit_status = ("Submitted" if recovered is not None else "NoAnswer") if completed else "InfraInterrupted"
    row = {"schema": "expA-score-v2", "kind": "episode", "design": "a3-ledger",
           "cell_id": f"LEDGER|{item}|{variant}|C1|s{seed}", "block": "LEDGER", "item_id": item,
           "family": FAMILY_OF[item], "variant": variant, "condition": "C1", "seed": seed, "cell_state": state,
           "completed": completed, "exit_status": exit_status,
           "site_recovered": (recovered or 0) if completed else None,
           "precision_lower": 0.5 if exit_status == "Submitted" else None,
           "precision_upper": 1.0 if exit_status == "Submitted" else None, "items_sha256": ITEMS_SHA}
    if review is not None:
        row["submit_review"] = review
    return row


def review_fields(first, source, reviewed=True):
    return {"mode": "x", "reviewed": reviewed, "answer_source": source, "review_skipped_last_call": not reviewed,
            "unresolved_exit": "NoAnswer" if source == "first_submit" and reviewed else None,
            "first_submission": None if first is None else {"site_recovered": first}}


def arm(incomplete, *, reviews=None):
    """complete (vcomplete) always recovered; incomplete (omit_node) per item at seed 61; seed 73 has a cell still
    pending in every run (so the prefix is [61]); seed 89 is complete but beyond the prefix."""
    rows = []
    for item in FAMILY_OF:
        rows.append(score_row(item, "vcomplete", 61, 1, review=review_fields(1, "first_submit")))
        rows.append(score_row(item, "omit_node", 61, incomplete[item], review=(reviews or {}).get(item)))
        for variant in ("vcomplete", "omit_node"):
            rows.append(score_row(item, variant, 73, 1, state="infra_pending" if item == "i1" else "completed"))
            rows.append(score_row(item, variant, 89, 0))
    rows.append({"block": "P", "cell_id": "P|i1|none|C0|s11", "cell_state": "not_dispatched"})
    return rows


# Hand-computed: G_neutral = 1 - 0 = 1 for every item. Ledger incomplete: i1 1 (G 0, M 1), i2 0 (M 0), i3 1 (M 1).
# Family means of M: f1 1/2, f2 1 -> 3/4; t = (3/4) / (1/4) = 3 with df 1. S1 = rec(incomplete, ledger - neutral):
# i1 1, i2 0, i3 1 -> 3/4. S2 complete ledger - neutral = 0.
NEUTRAL_INCOMPLETE = {"i1": 0, "i2": None, "i3": 0}       # i2 NoAnswer -> 0
LEDGER_INCOMPLETE = {"i1": 1, "i2": 0, "i3": 1}


class ReviewAnalysisTests(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def write(self, name, rows):
        path = self.base / name
        path.write_bytes(b"".join(json.dumps(row, sort_keys=True).encode() + b"\n" for row in rows))
        return str(path)

    def main(self, *argv):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
            code = A.main(list(argv))
        return code, err.getvalue()

    def inputs(self):
        neutral = self.write("neutral.jsonl", arm(NEUTRAL_INCOMPLETE, reviews={
            "i1": review_fields(0, "resubmit"), "i3": review_fields(0, "first_submit")}))
        ledger = self.write("ledger.jsonl", arm(LEDGER_INCOMPLETE, reviews={
            "i1": review_fields(0, "resubmit"), "i2": review_fields(None, None, reviewed=False),
            "i3": review_fields(1, "resubmit")}))
        return neutral, ledger

    def test_the_hand_computed_primary_and_descriptives(self):
        neutral, ledger = self.inputs()
        code, err = self.main("--neutral", neutral, "--ledger", ledger, "--out", str(self.base / "out"))
        self.assertEqual(code, 0, err)
        result = json.loads((self.base / "out" / "A5_ANALYSIS.json").read_bytes())
        self.assertIn("ANALYSIS RECORD", result["label"])
        main = result["main"]
        self.assertEqual(main["seeds_used"], [61])
        m = main["M_reduction_in_omission_cost"]
        self.assertEqual((m["estimate_exact"], m["n_families"], m["n_items"], m["family_means"]),
                         ("3/4", 2, 3, {"f1": 0.5, "f2": 1.0}))
        self.assertAlmostEqual(m["t"], 3.0, places=9)
        self.assertEqual(main["G_neutral_omission_cost_neutral_review"]["estimate_exact"], "1/1")
        self.assertEqual(main["S1_incomplete_ledger_minus_neutral"]["estimate_exact"], "3/4")
        self.assertEqual(main["S2_complete_ledger_minus_neutral"]["estimate_exact"], "0/1")
        self.assertEqual(main["primary_decision"]["direction"], "the ledger review reduces the omission cost")
        self.assertEqual(main["S6_exits"]["neutral"]["incomplete"], {"NoAnswer": 1, "Submitted": 2})
        d = main["review_descriptives"]
        self.assertEqual(d["incomplete_ledger"]["first_submission_omitted_cstar"], {"n": 1, "of": 2, "share": 0.5})
        self.assertEqual(d["incomplete_ledger"]["resubmission_added_cstar"], {"n": 1, "of": 1, "share": 1.0})
        self.assertEqual(d["incomplete_neutral"]["resubmission_added_cstar"], {"n": 0, "of": 1, "share": 0.0})
        self.assertEqual(d["incomplete_ledger"]["reviewed"], {"n": 2, "of": 3, "share": 2 / 3})
        self.assertEqual(d["complete_neutral"]["answer_source"], {"first_submit": 3})
        self.assertIn("| incomplete_ledger | 2/3 | 1/2 | 1/1 |",
                      (self.base / "out" / "A5_TABLES.md").read_text(encoding="utf-8"))

    def test_refusals_and_run_json(self):
        neutral, ledger = self.inputs()
        (self.base / "exists").mkdir()
        self.assertEqual(self.main("--neutral", neutral, "--ledger", ledger, "--out", str(self.base / "exists"))[0], 2)
        self.assertEqual(self.main("--neutral", neutral, "--ledger", neutral, "--out", str(self.base / "o1"))[0], 2)
        configs = {mode: harness.EpisodeConfig(final_turn_policy="deadline", submit_review=mode).record()
                   for mode in ("neutral", "ledger")}
        paths = {}
        for mode, config in configs.items():
            path = self.base / f"{mode}-run.json"
            path.write_bytes(json.dumps({"episode_config": config}).encode())
            paths[mode] = str(path)
        code, err = self.main("--neutral", neutral, "--ledger", ledger, "--neutral-run-json", paths["neutral"],
                              "--ledger-run-json", paths["ledger"], "--out", str(self.base / "o2"))
        self.assertEqual(code, 0, err)
        code, err = self.main("--neutral", neutral, "--ledger", ledger, "--neutral-run-json", paths["ledger"],
                              "--ledger-run-json", paths["neutral"], "--out", str(self.base / "o3"))
        self.assertEqual(code, 2)
        self.assertIn("submit_review", err)
        dispatched = arm(NEUTRAL_INCOMPLETE)
        dispatched[-1]["cell_state"] = "completed"
        self.assertEqual(self.main("--neutral", self.write("bad.jsonl", dispatched), "--ledger", ledger, "--out",
                                   str(self.base / "o4"))[0], 2)
        self.assertFalse((self.base / "o4").exists())


if __name__ == "__main__":
    unittest.main()
