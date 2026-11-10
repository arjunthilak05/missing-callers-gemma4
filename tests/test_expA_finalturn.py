"""Exp A follow-up A1: the final-turn policy (protocols/EXPA_FINALTURN.md).

(1) The policies and their exact texts (also checked against the frozen protocol), EpisodeConfig.record() and its
    validation; the default record is the frozen six-field config, key order included.
(2) The harness against the fake server, with synthetic transcripts for every policy: the notice's text, position and
    timing (only the request for the 8th response changes, by exactly the suffix on its last message); earlier
    submissions (never see it); several tool calls on responses 7 and 8; a format error at response 7 (the notice goes
    on the format-error message, no turn is added) and FormatExhausted precedence; a context near the limit (the
    notice counts in the preflight: ContextLimit at 511 remaining, a call at 512) and a ContextLimit before the last
    call (never noticed); stop requests and infrastructure failures at the last call; the receipts (event, exposure,
    accounting, episode_finished).
(3) The runner: --final-turn-policy is registered only with the default budget and model profile, and only with
    --blocks P --seeds 11; dry runs, run.json and every episode record the policy only when it is not none; a relaunch
    with another policy is refused.
(4) The launch script: runner --final-turn-policy is forwarded only when it is not none and refused before any unit
    with a non-default budget or model profile, or without --blocks P --seeds 11; ``bash -n``.
The default path itself is pinned, unedited, by tests/test_expA_default_golden.py, tests/test_expA_param.py
(DefaultAdditionsTests) and tests/test_expA_qat_profile.py (UnchangedProfilesTests). Linux only, like those tests
(serve.py imports pwd; the runner reads checkouts with POSIX no-follow reads).
"""
import collections
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import unittest
from unittest import mock

POSIX = os.name == "posix"
if POSIX:
    import test_expA_default_golden as G      # puts the project on sys.path
    import test_probe_harness as H
    from absent_edges.models import LocalModel, ModelProfile
    from absent_edges.probe import harness, serve
    from absent_edges.probe import ledger as L
    from absent_edges.probe.tools import ToolBox

PREFIX = "\n\n[harness notice] "
TEXTS = {"neutral": "Your previous tool call was processed normally.",
         "deadline": "This is your last response. Call submit_answer now."}
POLICIES = ("none", "neutral", "deadline")
FROZEN_CONFIG = {"max_model_calls": 8, "context_budget": 16384, "max_response_tokens": 4096,
                 "min_response_tokens": 512, "temperature": 0.2, "max_consecutive_format_errors": 3}
P_CONDITIONS = ("C0", "C1", "C2", "C3", "C4a")
BASH = {"calls": [("bash", {"command": "ls"})]}
SEARCH = {"calls": [("source_reference_search", {"query": "pkg/core.py::target"})]}
SUBMIT_CALL = ("submit_answer", {"sites": [{"path": "pkg/core.py", "line": 6}], "complete": "yes"})
SUBMIT = {"calls": [SUBMIT_CALL]}
Run = collections.namedtuple("Run", "result directory server rows")


def sha(text):
    return hashlib.sha256(text if isinstance(text, bytes) else text.encode("utf-8")).hexdigest()


def suffix(policy):
    return PREFIX + TEXTS[policy]


def receipt(policy, **values):
    """The final-turn receipt of an episode; by default the one of an episode that never reached its last call."""
    base = {"policy": policy, "text_sha256": sha(TEXTS[policy]), "suffix_sha256": sha(suffix(policy)), "call_index": 7,
            "appended": False, "sent": False, "delivered": False, "appended_to": None, "tool_call_id": None,
            "prompt_tokens": None, "max_tokens": None}
    assert set(values) <= set(base)
    return dict(base, **values)


def kinds(run):
    return [row["kind"] for row in run.rows]


def values(run, kind):
    return [row["value"] for row in run.rows if row["kind"] == kind]


def counts(run):
    """The preflight token count of every call (context_accounting, by call index)."""
    return [value["count"] for value in values(run, "context_accounting")]


def request_digests(run):
    """sha256 of the exact bytes of every chat request (model_exchange) and every chat-form /tokenize preflight."""
    return ([value["request_sha256"] for value in values(run, "model_exchange")],
            [value["request_sha256"] for value in values(run, "context_accounting")])


def requests_with_notice(run):
    return [body for _, body in run.server.requests if "[harness notice]" in json.dumps(body)]


# ---------------------------------------------------------------------------------------------------------------
# (1) policies, texts and the config record
# ---------------------------------------------------------------------------------------------------------------

@unittest.skipUnless(POSIX, "Linux only: serve.py imports pwd")
class FinalTurnConfigTests(unittest.TestCase):
    maxDiff = None

    def test_the_texts_are_the_frozen_protocols(self):
        # A1's policies and texts come first and are unchanged; follow-up A4 appends its two (tests/test_expA_a4.py).
        self.assertEqual(harness.FINAL_TURN_POLICIES[:len(POLICIES)], POLICIES)
        self.assertEqual(harness.FINAL_TURN_DEFAULT, "none")
        self.assertEqual(list(harness.FINAL_TURN_TEXTS.items())[:len(TEXTS)], list(TEXTS.items()))
        self.assertEqual(harness.FINAL_TURN_PREFIX, PREFIX)
        self.assertEqual(harness.FINAL_TURN_PROTOCOL, "protocols/EXPA_FINALTURN.md")
        self.assertIsNone(harness.final_turn_suffix("none"))
        self.assertEqual(harness.final_turn_suffix("neutral"),
                         "\n\n[harness notice] Your previous tool call was processed normally.")
        self.assertEqual(harness.final_turn_suffix("deadline"),
                         "\n\n[harness notice] This is your last response. Call submit_answer now.")
        protocol = (G.PROJECT / harness.FINAL_TURN_PROTOCOL).read_text(encoding="utf-8")
        self.assertIn('Format: `"\\n\\n[harness notice] " + TEXT`.', protocol)
        for policy, text in TEXTS.items():
            self.assertIn(f"| `{policy}` | `{text}` |", protocol)
        self.assertIn("| `none` | no notice", protocol)
        # Not a prompt template: run.json's prompts (and their golden digest) stay those of the prereg harness.
        self.assertFalse(any(text in json.dumps(harness.prompt_templates()) for text in TEXTS.values()))

    def test_the_default_record_is_the_frozen_config(self):
        default = harness.EpisodeConfig()
        self.assertEqual(default.final_turn_policy, "none")
        self.assertEqual(harness.EpisodeConfig(final_turn_policy="none"), default)
        self.assertEqual(default.record(), FROZEN_CONFIG)
        self.assertEqual(list(default.record()), list(FROZEN_CONFIG))    # events.jsonl keeps key order
        self.assertEqual(default.record(), G.GOLDEN["runner"]["live"]["result"]["config"])
        self.assertEqual(harness.operational_appendix()["episode_config"], FROZEN_CONFIG)
        self.assertEqual(harness.EpisodeConfig(max_model_calls=16, context_budget=32768).record(),
                         dict(FROZEN_CONFIG, max_model_calls=16, context_budget=32768))

    def test_a_policy_record_adds_the_policy_and_its_exact_notice(self):
        for policy, text in TEXTS.items():
            with self.subTest(policy=policy):
                config = harness.EpisodeConfig(final_turn_policy=policy)
                record = config.record()
                self.assertNotEqual(config, harness.EpisodeConfig())
                self.assertEqual(list(record), list(FROZEN_CONFIG) + ["final_turn_policy", "final_turn_notice"])
                self.assertEqual({key: record[key] for key in FROZEN_CONFIG}, FROZEN_CONFIG)
                self.assertEqual(record["final_turn_policy"], policy)
                notice = dict(record["final_turn_notice"])
                self.assertTrue(notice.pop("appended_to").startswith("the content of the last message before that "
                                                                      "call: the last model-visible tool result"))
                self.assertEqual(notice, {"protocol": "protocols/EXPA_FINALTURN.md", "channel": "in_band",
                                          "prefix": PREFIX, "text": text, "text_sha256": sha(text),
                                          "suffix_sha256": sha(PREFIX + text), "call_index": 7})
                json.dumps(record)

    def test_invalid_policies_are_refused(self):
        for bad in ("reminder", "Deadline", "", None, 1, ["deadline"]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                harness.EpisodeConfig(final_turn_policy=bad)
        with self.assertRaisesRegex(ValueError, "needs max_model_calls >= 2"):
            harness.EpisodeConfig(max_model_calls=1, final_turn_policy="deadline")
        self.assertEqual(harness.EpisodeConfig(max_model_calls=1).record()["max_model_calls"], 1)


# ---------------------------------------------------------------------------------------------------------------
# (2) the harness: synthetic transcripts against the fake server
# ---------------------------------------------------------------------------------------------------------------

@unittest.skipUnless(POSIX, "the ToolBox search scans with POSIX no-follow reads")
class FinalTurnEpisodeTests(unittest.TestCase):
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

    def run_policy(self, script, policy, *, bash=None, pad=0, condition="C1", **kwargs):
        server = H.FakeServer(script, pad=pad)
        self.servers.append(server)
        self.count += 1
        directory = Path(self.temporary.name) / f"attempt-{self.count}"
        directory.mkdir()
        client = LocalModel(ModelProfile(endpoint=server.endpoint), timeout=20)
        toolbox = ToolBox(self.root, ["pkg"], bash or H.FakeBash())
        result, digest = harness.run_episode(H.make_episode(condition), directory, client=client, toolbox=toolbox,
                                             config=harness.EpisodeConfig(final_turn_policy=policy), **kwargs)
        self.assertEqual(digest, L.sha256_path(directory / "result.json"))
        return Run(result, directory, server, H.events(directory))

    def assertDelivered(self, run, policy, *, appended_to="tool_result", tool_call_id="call-7-0"):
        """The receipt of an episode whose last request carried the notice, in every place it is recorded."""
        count = counts(run)[7]
        expected = receipt(policy, appended=True, sent=True, delivered=True, appended_to=appended_to,
                           tool_call_id=tool_call_id, prompt_tokens=count, max_tokens=min(4096, 16384 - count))
        self.assertEqual(run.result["exposure"]["final_turn_notice"], expected)
        self.assertEqual(run.result["accounting"]["final_turn_notice"], expected)
        self.assertEqual(values(run, "episode_finished")[0]["accounting"]["final_turn_notice"], expected)
        self.assertEqual(run.server.chat_requests()[7]["max_tokens"], expected["max_tokens"])
        self.assertEqual(values(run, "model_exchange")[7]["response"]["usage"]["prompt_tokens"], count)
        return expected

    def test_the_notice_follows_the_7th_response_only(self):
        """bash x 8 in every P condition: requests 1-7 are the default ones; request 8 differs by exactly the suffix
        on its last message (the 7th response's tool result), and the receipts record it."""
        for condition in P_CONDITIONS:
            base = self.run_policy([BASH] * 8, "none", condition=condition)
            with self.subTest(condition=condition, policy="none"):
                self.assertEqual((base.result["exit_status"], base.result["calls"]), ("NoAnswer", 8))
                self.assertNotIn("final_turn_notice", base.result["exposure"])
                self.assertNotIn("final_turn_notice", base.result["accounting"])
                self.assertNotIn("final_turn_notice", kinds(base))
                self.assertEqual((base.result["config"], values(base, "episode_setup")[0]["config"]),
                                 (FROZEN_CONFIG, FROZEN_CONFIG))
                self.assertEqual(requests_with_notice(base), [])
            base_chats = base.server.chat_requests()
            base_last = base_chats[7]["messages"][-1]
            for policy, text in TEXTS.items():
                with self.subTest(condition=condition, policy=policy):
                    run = self.run_policy([BASH] * 8, policy, condition=condition)
                    chats = run.server.chat_requests()
                    self.assertEqual((run.result["exit_status"], run.result["calls"], len(chats)), ("NoAnswer", 8, 8))
                    # Responses 1-7: the exact request bytes of the default run (chat and /tokenize).
                    (chat_digests, tokenize_digests), (base_chat, base_tokenize) = (request_digests(run),
                                                                                    request_digests(base))
                    self.assertEqual((chat_digests[:7], tokenize_digests[:7]), (base_chat[:7], base_tokenize[:7]))
                    self.assertNotEqual((chat_digests[7], tokenize_digests[7]), (base_chat[7], base_tokenize[7]))
                    # Response 8: only the last message differs, by exactly the suffix; no message is added.
                    last = chats[7]["messages"][-1]
                    self.assertEqual(chats[7]["messages"][:-1], base_chats[7]["messages"][:-1])
                    self.assertEqual(last, dict(base_last, content=base_last["content"] + PREFIX + text))
                    self.assertEqual((last["role"], last["tool_call_id"]), ("tool", "call-7-0"))
                    self.assertEqual({key: value for key, value in chats[7].items() if key != "messages"},
                                     {key: value for key, value in base_chats[7].items() if key != "messages"})
                    # The notice is in the 8th preflight and the 8th chat request only, once each.
                    noticed = requests_with_notice(run)
                    self.assertEqual([("messages" in body, "max_tokens" in body) for body in noticed],
                                     [(True, False), (True, True)])
                    self.assertEqual([json.dumps(body).count(text) for body in noticed], [1, 1])
                    self.assertEqual(noticed[1]["messages"], chats[7]["messages"])
                    # The event: right before the 8th preflight, binding the old and the new content.
                    rows = kinds(run)
                    index = rows.index("final_turn_notice")
                    self.assertEqual((rows.count("final_turn_notice"), rows[index + 1]), (1, "context_accounting"))
                    self.assertEqual(run.rows[index + 1]["value"]["call_index"], 7)
                    self.assertEqual(values(run, "final_turn_notice"), [{
                        "call_index": 7, "policy": policy, "appended_to": "tool_result", "tool_call_id": "call-7-0",
                        "suffix": PREFIX + text, "suffix_sha256": sha(PREFIX + text), "text_sha256": sha(text),
                        "previous_content_sha256": sha(base_last["content"]), "content_sha256": sha(last["content"])}])
                    # The model_visible_tool_result events keep the tools' own output (pure JSON), as by default.
                    visible = values(run, "model_visible_tool_result")
                    self.assertEqual(visible, values(base, "model_visible_tool_result"))
                    self.assertEqual((visible[6]["tool_call_id"], visible[6]["sha256"]),
                                     ("call-7-0", sha(base_last["content"])))
                    json.loads(visible[6]["content"])
                    # The notice counts: the 8th preflight is larger, and so is the served prompt.
                    self.assertGreater(counts(run)[7], counts(base)[7])
                    self.assertEqual(counts(run)[:7], counts(base)[:7])
                    self.assertEqual(run.result["tokens"]["prompt"] - base.result["tokens"]["prompt"],
                                     counts(run)[7] - counts(base)[7])
                    self.assertDelivered(run, policy)
                    # The exposure logged before the first call is the default one; the result adds the receipt.
                    self.assertEqual(values(run, "exposure_receipt"), values(base, "exposure_receipt"))
                    exposure = dict(run.result["exposure"])
                    exposure.pop("final_turn_notice")
                    self.assertEqual(exposure, base.result["exposure"])
                    record = harness.EpisodeConfig(final_turn_policy=policy).record()
                    self.assertEqual((run.result["config"], values(run, "episode_setup")[0]["config"]), (record, record))
                    self.assertEqual((run.result["submitted"], run.result["accounting"]["format_errors"]), (None, 0))

    def test_an_episode_that_submits_before_its_last_response_never_sees_it(self):
        for responses in (1, 3, 7):
            script = [BASH] * (responses - 1) + [SUBMIT]
            base = self.run_policy(script, "none")
            for policy in TEXTS:
                with self.subTest(responses=responses, policy=policy):
                    run = self.run_policy(script, policy)
                    self.assertEqual((run.result["exit_status"], run.result["calls"]), ("Submitted", responses))
                    self.assertEqual(request_digests(run), request_digests(base))   # byte-identical requests
                    self.assertEqual(run.server.chat_requests(), base.server.chat_requests())
                    self.assertEqual(requests_with_notice(run), [])
                    self.assertNotIn("final_turn_notice", kinds(run))
                    self.assertEqual(run.result["exposure"]["final_turn_notice"], receipt(policy))
                    self.assertEqual(run.result["accounting"]["final_turn_notice"], receipt(policy))
                    self.assertEqual(run.result["submitted"], base.result["submitted"])

    def test_a_submission_in_the_last_response_is_a_reminded_submission(self):
        for policy in TEXTS:
            with self.subTest(policy=policy):
                run = self.run_policy([BASH] * 7 + [SUBMIT], policy)
                self.assertEqual((run.result["exit_status"], run.result["calls"]), ("Submitted", 8))
                self.assertEqual([value["call_index"] for value in values(run, "submission_accepted")], [7])
                self.assertDelivered(run, policy)

    def test_several_tool_calls_on_responses_7_and_8(self):
        seventh = {"calls": [("bash", {"command": "ls"}), ("source_reference_search", {"query": "target"}),
                             ("bash", {"command": "pwd"})]}
        bash_cat = ("bash", {"command": "cat pkg/core.py"})
        cases = {  # response 8 -> exit, bash commands of response 8, accepted call indexes, not executed, format errors
            "bash_then_submit": ({"calls": [bash_cat, SUBMIT_CALL]}, "Submitted", ["cat pkg/core.py"], [7], 0, 0),
            "submit_then_bash": ({"calls": [SUBMIT_CALL, ("bash", {"command": "cat x"})]}, "Submitted", [], [7], 1, 0),
            "two_submits": ({"calls": [SUBMIT_CALL, SUBMIT_CALL]}, "Submitted", [], [7], 1, 0),
            "two_tools": ({"calls": [bash_cat, ("source_reference_search", {"query": "aliased"})]}, "NoAnswer",
                          ["cat pkg/core.py"], [], 0, 0),
            "bad_envelope": ({"calls": [("submit_answer", {"sites": "pkg/core.py:6", "complete": "yes"})]},
                             "NoAnswer", [], [], 0, 0),
            "unknown_tool": ({"calls": [("python", {"code": "1"})]}, "NoAnswer", [], [], 0, 1),
            "text_only": ({"content": "The callers are pkg/core.py:6 and pkg/core.py:11."}, "NoAnswer", [], [], 0, 1)}
        for name, (eighth, exit_status, commands, accepted, not_executed, format_errors) in cases.items():
            for policy in TEXTS:
                with self.subTest(case=name, policy=policy):
                    bash = H.FakeBash()
                    run = self.run_policy([BASH] * 6 + [seventh, eighth], policy, bash=bash)
                    chats = run.server.chat_requests()
                    self.assertEqual((run.result["exit_status"], run.result["calls"], len(chats)),
                                     (exit_status, 8, 8))       # never a 9th response
                    tail = chats[7]["messages"][-3:]
                    self.assertEqual([(m["role"], m["tool_call_id"]) for m in tail],
                                     [("tool", "call-7-0"), ("tool", "call-7-1"), ("tool", "call-7-2")])
                    # Only the last tool result of response 7 carries the notice.
                    self.assertTrue(tail[-1]["content"].endswith(suffix(policy)))
                    json.loads(tail[-1]["content"][:-len(suffix(policy))])
                    for message in tail[:-1]:
                        json.loads(message["content"])
                    self.assertEqual(json.dumps(chats[7]["messages"]).count("[harness notice]"), 1)
                    self.assertDelivered(run, policy, tool_call_id="call-7-2")
                    self.assertEqual(bash.commands, ["ls"] * 6 + ["ls", "pwd"] + commands)
                    self.assertEqual([value["call_index"] for value in values(run, "submission_accepted")], accepted)
                    self.assertEqual(run.result["accounting"]["not_executed_after_submission"], not_executed)
                    self.assertEqual(run.result["accounting"]["format_errors"], format_errors)
                    if name == "bad_envelope":
                        self.assertEqual(run.result["accounting"]["rejected_submissions"], 1)
                    if exit_status == "Submitted":
                        self.assertEqual(run.result["submitted"], {"sites": [{"path": "pkg/core.py", "line": 6}],
                                                                   "complete": "yes"})

    def test_the_notice_follows_whatever_tool_result_ended_response_7(self):
        """A rejected submission at response 7 is a tool result too: the notice goes after its visible error."""
        for policy in TEXTS:
            with self.subTest(policy=policy):
                bad = {"calls": [("submit_answer", {"sites": "pkg/core.py:6", "complete": "yes"})]}
                run = self.run_policy([BASH] * 6 + [bad, SUBMIT], policy)
                last = run.server.chat_requests()[7]["messages"][-1]
                self.assertEqual(last["role"], "tool")
                self.assertTrue(last["content"].endswith(suffix(policy)))
                self.assertIn("submit_answer rejected: sites must be a list", last["content"])
                self.assertEqual((run.result["exit_status"], run.result["accounting"]["rejected_submissions"]),
                                 ("Submitted", 1))
                self.assertDelivered(run, policy)

    def test_a_format_error_at_response_7_carries_the_notice_in_its_message(self):
        script = [BASH] * 6 + [{"content": "I think I am done."}, SUBMIT]
        base = self.run_policy(script, "none")
        base_messages = base.server.chat_requests()[7]["messages"]
        self.assertTrue(base_messages[-1]["content"].startswith("Format error: your response did not call a tool"))
        for policy, text in TEXTS.items():
            with self.subTest(policy=policy):
                run = self.run_policy(script, policy)
                chats = run.server.chat_requests()
                self.assertEqual(request_digests(run)[0][:7], request_digests(base)[0][:7])
                messages = chats[7]["messages"]
                self.assertEqual(len(messages), len(base_messages))    # no message or turn is added
                self.assertEqual(messages[:-1], base_messages[:-1])
                self.assertEqual(messages[-2], {"role": "assistant", "content": "I think I am done."})
                self.assertEqual(messages[-1], {"role": "user", "content": base_messages[-1]["content"] + PREFIX + text})
                self.assertDelivered(run, policy, appended_to="format_error", tool_call_id=None)
                event = values(run, "final_turn_notice")[0]
                self.assertEqual((event["appended_to"], event["tool_call_id"], event["previous_content_sha256"]),
                                 ("format_error", None, sha(base_messages[-1]["content"])))
                # The format error itself is unchanged: its visible text and the consecutive count.
                self.assertEqual(values(run, "format_error"), values(base, "format_error"))
                self.assertEqual((run.result["exit_status"], run.result["calls"], run.result["accounting"]["format_errors"],
                                  [value["call_index"] for value in values(run, "submission_accepted")]),
                                 ("Submitted", 8, 1, [7]))

    def test_format_errors_at_the_end_keep_their_exits(self):
        for policy in TEXTS:
            with self.subTest(policy=policy, case="responses 6-8: FormatExhausted precedes NoAnswer"):
                run = self.run_policy([BASH] * 5 + [{"content": "a"}, {"content": "b"}, {"content": "c"}], policy)
                self.assertEqual((run.result["exit_status"], run.result["calls"], run.result["accounting"]["format_errors"]),
                                 ("FormatExhausted", 8, 3))
                self.assertEqual([value["consecutive"] for value in values(run, "format_error")], [1, 2, 3])
                self.assertDelivered(run, policy, appended_to="format_error", tool_call_id=None)
            with self.subTest(policy=policy, case="responses 7-8: NoAnswer"):
                run = self.run_policy([BASH] * 6 + [{"content": "a"}, {"content": "b"}], policy)
                self.assertEqual((run.result["exit_status"], run.result["calls"], run.result["accounting"]["format_errors"]),
                                 ("NoAnswer", 8, 2))
                self.assertDelivered(run, policy, appended_to="format_error", tool_call_id=None)

    def test_the_notice_counts_against_the_context_budget(self):
        """With 511 tokens left after the notice the episode ends as ContextLimit at the 8th call (appended, never
        sent); the default run of the same transcript still makes that call; with 512 left the call is made."""
        for policy in TEXTS:
            with self.subTest(policy=policy):
                with_notice = counts(self.run_policy([BASH] * 8, policy))[7]
                without = counts(self.run_policy([BASH] * 8, "none"))[7]
                self.assertGreater(with_notice, without)
                pad = 16384 - 511 - with_notice
                limited = self.run_policy([BASH] * 8, policy, pad=pad)
                self.assertEqual((limited.result["exit_status"], limited.result["calls"],
                                  len(limited.server.chat_requests())), ("ContextLimit", 7, 7))
                self.assertEqual(counts(limited)[7], 16384 - 511)
                self.assertEqual(values(limited, "context_limit"), [{"call_index": 7, "prompt_tokens": 16384 - 511,
                                                                     "context_budget": 16384, "remaining": 511,
                                                                     "min_response_tokens": 512}])
                expected = receipt(policy, appended=True, appended_to="tool_result", tool_call_id="call-7-0",
                                   prompt_tokens=16384 - 511)
                self.assertEqual(limited.result["exposure"]["final_turn_notice"], expected)
                self.assertEqual(limited.result["accounting"]["final_turn_notice"], expected)
                rows = kinds(limited)
                self.assertEqual(rows[rows.index("final_turn_notice") + 1], "context_accounting")
                # The same transcript and padding without the notice: the 8th call is made.
                default = self.run_policy([BASH] * 8, "none", pad=pad)
                self.assertEqual((default.result["exit_status"], default.result["calls"]), ("NoAnswer", 8))
                self.assertEqual(default.server.chat_requests()[7]["max_tokens"],
                                 min(4096, 16384 - (without + pad)))
                # Exactly 512 tokens left with the notice: the call is made with max_tokens 512.
                edge = self.run_policy([BASH] * 7 + [SUBMIT], policy, pad=16384 - 512 - with_notice)
                self.assertEqual((edge.result["exit_status"], edge.result["calls"]), ("Submitted", 8))
                self.assertEqual(self.assertDelivered(edge, policy)["max_tokens"], 512)

    def test_a_context_limit_before_the_last_call_never_appends_it(self):
        """The trigger counts responses only: an episode that ends as ContextLimit before its 8th call never gets the
        notice, and every request it made is the default run's."""
        pad = 16384 - 511 - counts(self.run_policy([BASH] * 8, "none"))[5]     # 511 remaining before the 6th call
        base = self.run_policy([BASH] * 8, "none", pad=pad)
        self.assertEqual((base.result["exit_status"], base.result["calls"]), ("ContextLimit", 5))
        for policy in TEXTS:
            with self.subTest(policy=policy):
                run = self.run_policy([BASH] * 8, policy, pad=pad)
                self.assertEqual((run.result["exit_status"], run.result["calls"]), ("ContextLimit", 5))
                self.assertEqual(request_digests(run), request_digests(base))
                self.assertEqual(run.server.chat_requests(), base.server.chat_requests())
                self.assertEqual(values(run, "context_limit"), values(base, "context_limit"))
                self.assertEqual(requests_with_notice(run), [])
                self.assertNotIn("final_turn_notice", kinds(run))
                self.assertEqual(run.result["exposure"]["final_turn_notice"], receipt(policy))
                self.assertEqual(run.result["accounting"]["final_turn_notice"], receipt(policy))

    def test_stop_requests_and_infrastructure_at_the_last_call(self):
        for policy in TEXTS:
            with self.subTest(policy=policy, case="stop requested while response 7's tool runs"):
                stop = threading.Event()

                class StopBash(H.FakeBash):
                    def run(self, command):
                        if len(self.commands) == 6:
                            stop.set()
                        return super().run(command)
                run = self.run_policy([BASH] * 8, policy, bash=StopBash(), stop_event=stop)
                self.assertEqual((run.result["exit_status"], run.result["infra"]["kind"], run.result["calls"]),
                                 ("InfraInterrupted", "stop_requested", 7))
                self.assertEqual(run.result["exposure"]["final_turn_notice"], receipt(policy))   # never appended
                self.assertNotIn("final_turn_notice", kinds(run))
                self.assertEqual(requests_with_notice(run), [])
            with self.subTest(policy=policy, case="HTTP 500 on the 8th request"):
                run = self.run_policy([BASH] * 7 + [{"status": 500}], policy)
                self.assertEqual((run.result["exit_status"], run.result["infra"]["kind"]), ("InfraInterrupted", "server"))
                count = counts(run)[7]
                self.assertEqual(run.result["exposure"]["final_turn_notice"],
                                 receipt(policy, appended=True, sent=True, appended_to="tool_result",
                                         tool_call_id="call-7-0", prompt_tokens=count, max_tokens=min(4096, 16384 - count)))
            with self.subTest(policy=policy, case="served prompt differs from the preflight on the 8th request"):
                run = self.run_policy([BASH] * 7 + [dict(BASH, usage_mismatch=True)], policy)
                self.assertEqual((run.result["exit_status"], run.result["infra"]["kind"]),
                                 ("InfraInterrupted", "accounting"))
                notice = run.result["accounting"]["final_turn_notice"]
                self.assertEqual((notice["appended"], notice["sent"], notice["delivered"]), (True, True, False))


# ---------------------------------------------------------------------------------------------------------------
# (3) the runner: registration, dry run, run.json, every episode, relaunch
# ---------------------------------------------------------------------------------------------------------------

@unittest.skipUnless(POSIX, "the runner reads checkouts with POSIX no-follow reads")
class FinalTurnRunnerTests(unittest.TestCase):
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
        """A dry-run parse; a notice policy gets the registered selection, block P at seed 11, unless one is given."""
        if any(policy in extra for policy in TEXTS) and "--blocks" not in extra and "--seeds" not in extra:
            extra += ("--blocks", "P", "--seeds", "11")
        with contextlib.redirect_stderr(io.StringIO()):
            return self.runner.parse_args(["--dispatch", str(self.dispatch), "--dry-run", *extra])

    def refused(self, *extra):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
            self.runner.parse_args(["--dispatch", str(self.dispatch), "--dry-run", *extra])
        self.assertEqual(raised.exception.code, 2)
        return stderr.getvalue()

    def live(self, server, *extra, run="run"):
        argv = G.live_args(self.base, self.dispatch, self.manifest, server, run=run) + list(extra)
        return G.live_episode(self.runner, argv, server, self.base / run)

    def test_the_policy_is_registered_only_with_the_default_budget_and_profile(self):
        self.assertEqual(self.runner.REGISTERED_DESIGNS, {
            (8, 16384, "none"): ("protocols/EXPA_PREREG.md", None),
            (16, 32768, "none"): ("protocols/EXPA_BUDGET16.md", None),
            (8, 16384, "neutral"): ("protocols/EXPA_FINALTURN.md", ("gemma4-31b",)),
            (8, 16384, "deadline"): ("protocols/EXPA_FINALTURN.md", ("gemma4-31b",)),
            # follow-up A4 (tests/test_expA_a4.py)
            (8, 16384, "deadline_info"): ("protocols/EXPA_A4_DIRECTIVE.md", ("gemma4-31b",)),
            (8, 16384, "submit_only"): ("protocols/EXPA_A4_DIRECTIVE.md", ("gemma4-31b",))})
        # gemma4-31b is the default profile and the frozen 31B BF16 that protocols/EXPA_FINALTURN.md inherits.
        self.assertEqual(serve.DEFAULT_MODEL_PROFILE, "gemma4-31b")
        self.assertTrue(serve.served_model("gemma4-31b").revision.startswith("518276fb"))
        self.assertEqual(self.parse().final_turn_policy, "none")
        self.assertEqual(self.parse().episode_config, harness.EpisodeConfig())
        self.assertEqual(self.parse("--final-turn-policy", "none").episode_config, harness.EpisodeConfig())
        for policy in TEXTS:
            with self.subTest(policy=policy):
                expected = harness.EpisodeConfig(final_turn_policy=policy)
                self.assertEqual(self.parse("--final-turn-policy", policy).episode_config, expected)
                self.assertEqual(self.parse("--final-turn-policy", policy, "--model-profile", "gemma4-31b",
                                            "--max-model-calls", "8", "--context-budget", "16384").episode_config,
                                 expected)
                for bad in (["--max-model-calls", "16", "--context-budget", "32768"], ["--model-profile", "gemma4-12b"],
                            ["--model-profile", "gemma4-31b-qat"],
                            ["--model-profile", "gemma4-12b", "--max-model-calls", "16", "--context-budget", "32768"]):
                    with self.subTest(bad=bad):
                        self.assertIn(f"--final-turn-policy {policy} (protocols/EXPA_FINALTURN.md) is registered only "
                                      "with the prereg budget (8 responses, 16384 tokens) and --model-profile "
                                      "gemma4-31b", self.refused("--final-turn-policy", policy, *bad))
        for bad in ("reminder", "Deadline", "", "R1"):
            with self.subTest(bad=bad):
                self.assertIn("invalid choice", self.refused("--final-turn-policy", bad))
        # Without a policy the other registered designs are unchanged.
        args = self.parse("--max-model-calls", "16", "--context-budget", "32768", "--model-profile", "gemma4-12b")
        self.assertEqual(args.episode_config, harness.EpisodeConfig(max_model_calls=16, context_budget=32768))
        self.assertEqual(self.parse("--model-profile", "gemma4-31b-qat").episode_config, harness.EpisodeConfig())

    def test_a_policy_runs_block_p_at_seed_11_only(self):
        """run.json freezes the whole design: a notice policy without exactly --blocks P --seeds 11 (e.g. a relaunch
        after an endpoint trip) would dispatch unregistered cells under the notice, so it is refused. (Follow-up A4
        also registers --blocks P --seeds 23, one seed per launch: tests/test_expA_a4.py.)"""
        for policy in TEXTS:
            args = self.parse("--final-turn-policy", policy, "--blocks", "P", "--seeds", "11")
            self.assertEqual((args.blocks, args.seeds, args.episode_config),
                             ({"P"}, {11}, harness.EpisodeConfig(final_turn_policy=policy)))
            for selection in ((), ("--blocks", "P"), ("--seeds", "11"), ("--blocks", "P,S-VC", "--seeds", "11"),
                              ("--blocks", "P", "--seeds", "11,23")):
                with self.subTest(policy=policy, selection=selection):
                    self.assertIn(f"--final-turn-policy {policy} (protocols/EXPA_FINALTURN.md) runs block P at seed 11 "
                                  "or seed 23 only, one seed per launch: pass --blocks P --seeds 11",
                                  self.refused("--final-turn-policy", policy, *selection))
        # The default and none keep every selection.
        for extra in ((), ("--final-turn-policy", "none")):
            for selection in ((), ("--blocks", "P,S-VC", "--seeds", "11,23")):
                with self.subTest(extra=extra, selection=selection):
                    self.assertEqual(self.parse(*extra, *selection).episode_config, harness.EpisodeConfig())

    def test_a_dry_run_records_the_policy_and_renders_the_same_prompts(self):
        argv = ["--dispatch", str(self.dispatch), "--dry-run", "--show", "3", "--blocks", "P", "--seeds", "11"]
        code, output = G.invoke_runner(self.runner, argv)
        default = json.loads(output)
        self.assertEqual((code, default["episode_config"]), (0, FROZEN_CONFIG))
        for policy in TEXTS:
            with self.subTest(policy=policy):
                code, output = G.invoke_runner(self.runner, argv + ["--final-turn-policy", policy])
                summary = json.loads(output)
                self.assertEqual(code, 0)
                self.assertEqual(summary.pop("episode_config"),
                                 harness.EpisodeConfig(final_turn_policy=policy).record())
                self.assertEqual(summary, {key: value for key, value in default.items() if key != "episode_config"})

    def test_run_json_and_every_episode_record_the_policy_only_when_it_is_not_none(self):
        runs = {}
        for name, extra in (("default", ()), ("none", ("--final-turn-policy", "none")),
                            ("neutral", ("--final-turn-policy", "neutral")),
                            ("deadline", ("--final-turn-policy", "deadline"))):
            server = self.server(default=SEARCH)     # never submits: every episode reaches its 8th response
            live = self.live(server, "--run-id", "same", *extra, run=name)
            self.assertEqual((live["exit_code"], live["results"], live["result"]["exit_status"]), (0, 1, "NoAnswer"))
            result = json.loads(next((self.base / name / "cells").rglob("result.json")).read_bytes())
            runs[name] = (live, (self.base / name / "run.json").read_bytes(), server, result)
        self.assertEqual(runs["none"][1], runs["default"][1])     # --final-turn-policy none: run.json byte for byte
        default = json.loads(runs["default"][1])
        self.assertEqual(default["episode_config"], FROZEN_CONFIG)
        for name in ("default", "none"):
            live, _, server, result = runs[name]
            self.assertEqual((live["result"]["config"], live["setup"]["config"]), (FROZEN_CONFIG, FROZEN_CONFIG))
            self.assertNotIn("final_turn_notice", result["exposure"])
            self.assertNotIn("[harness notice]", json.dumps(server.chat_requests()))
        self.assertEqual(runs["none"][2].chat_requests(), runs["default"][2].chat_requests())
        for policy in TEXTS:
            with self.subTest(policy=policy):
                live, raw, server, result = runs[policy]
                record = harness.EpisodeConfig(final_turn_policy=policy).record()
                run_json = json.loads(raw)
                self.assertEqual(run_json["episode_config"], record)
                self.assertEqual({key: value for key, value in run_json.items() if key != "episode_config"},
                                 {key: value for key, value in default.items() if key != "episode_config"})
                self.assertEqual((live["result"]["config"], live["setup"]["config"]), (record, record))
                self.assertEqual(live["setup"]["system"], runs["default"][0]["setup"]["system"])
                chats = server.chat_requests()
                self.assertEqual(chats[:7], runs["default"][2].chat_requests()[:7])
                self.assertEqual(chats[7]["messages"][-1]["content"],
                                 runs["default"][2].chat_requests()[7]["messages"][-1]["content"] + suffix(policy))
                notice = result["exposure"]["final_turn_notice"]
                self.assertEqual((notice["policy"], notice["appended"], notice["sent"], notice["delivered"],
                                  notice["appended_to"]), (policy, True, True, True, "tool_result"))
                self.assertEqual(result["accounting"]["final_turn_notice"], notice)

    def test_a_relaunch_with_another_policy_is_refused(self):
        server = self.server(default=H.SUBMIT)
        self.assertEqual(self.live(server, "--final-turn-policy", "deadline")["exit_code"], 0)
        chats = len(server.chat_requests())
        for extra in ((), ("--final-turn-policy", "none"), ("--final-turn-policy", "neutral")):
            with self.subTest(extra=extra):
                argv = G.live_args(self.base, self.dispatch, self.manifest, server) + list(extra)
                with mock.patch.object(G.tools, "ReadOnlyBash", G.FakeJailBash), \
                        self.assertRaisesRegex(L.LedgerInconsistency, "run.json already exists with different content"):
                    G.invoke_runner(self.runner, argv)
        self.assertEqual(len(server.chat_requests()), chats)
        # The same policy relaunches (and runs the next cell).
        argv = G.live_args(self.base, self.dispatch, self.manifest, server) + ["--final-turn-policy", "deadline"]
        with mock.patch.object(G.tools, "ReadOnlyBash", G.FakeJailBash):
            self.assertEqual(G.invoke_runner(self.runner, argv)[0], 0)
        self.assertEqual(len(server.chat_requests()), chats + 1)


# ---------------------------------------------------------------------------------------------------------------
# (4) the launch script (stubbed host, nothing is started)
# ---------------------------------------------------------------------------------------------------------------

@unittest.skipUnless(POSIX and shutil.which("bash"), "Linux only: the launch script runs on Core1")
class FinalTurnLaunchScriptTests(unittest.TestCase):
    maxDiff = None

    def host(self, *prepare_extra):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        host = G.LaunchHost(directory.name)
        host.run(*G.PREPARE_ARGV, "--runtime-python", str(host.python), *prepare_extra)
        host.stage("golden1")
        return host

    def runner_units(self, host, *extra, check=True):
        result = host.run("runner", "--run", "golden1", "--blocks", "P", "--seeds", "11", *extra, check=check)
        return result, [host.normalize(call) for call in G.units(host.sudo_calls(), "runner")]

    def runner_args(self, unit):
        tail = unit[next(i for i, value in enumerate(unit) if value.endswith("/scripts/expA_run.py")) + 1:]
        with contextlib.redirect_stderr(io.StringIO()):
            return G.load_script("expA_run").parse_args(tail)

    def test_syntax(self):
        self.assertEqual(subprocess.run(["bash", "-n", str(G.PROJECT / "scripts" / "expA_core1_launch.sh")],
                                        timeout=60).returncode, 0)

    def test_a_policy_is_forwarded_only_when_it_is_not_none(self):
        host = self.host()
        golden = G.GOLDEN["launch"]["runner_units"]
        for policy in TEXTS:
            with self.subTest(policy=policy):
                result, units = self.runner_units(host, "--final-turn-policy", policy)
                self.assertEqual(units, [golden[0] + ["--final-turn-policy", policy]])
                self.assertIn(f"final-turn policy {policy} (protocols/EXPA_FINALTURN.md)", result.stderr)
                args = self.runner_args(units[0])
                self.assertEqual((args.model_profile, args.final_turn_policy, args.episode_config),
                                 ("gemma4-31b", policy, harness.EpisodeConfig(final_turn_policy=policy)))
        for extra in ((), ("--final-turn-policy", "none")):
            with self.subTest(extra=extra):
                result, units = self.runner_units(host, *extra)
                self.assertEqual(units, golden)          # the frozen runner command
                self.assertNotIn("final-turn policy", result.stderr)
                self.assertEqual(self.runner_args(units[0]).episode_config, harness.EpisodeConfig())

    def test_unregistered_policies_are_refused_before_any_unit(self):
        host = self.host()
        budget16 = ["--max-model-calls", "16", "--context-budget", "32768"]
        for extra, message in ((["--final-turn-policy", "reminder"],
                                "--final-turn-policy must be none, neutral, deadline, deadline_info or submit_only, "
                                "not reminder"),
                               (["--final-turn-policy", "Deadline"],
                                "--final-turn-policy must be none, neutral, deadline, deadline_info or submit_only, "
                                "not Deadline"),
                               (["--final-turn-policy", "deadline"] + budget16,
                                "--final-turn-policy deadline: registered only with the default budget (8 responses, "
                                "16384 tokens) and model profile gemma4-31b (protocols/EXPA_FINALTURN.md)"),
                               (budget16 + ["--final-turn-policy", "neutral"],
                                "--final-turn-policy neutral: registered only with the default budget"),
                               (["--final-turn-policy", "deadline", "--max-model-calls", "16"],
                                "only the registered 16/32768 design")):
            with self.subTest(extra=extra):
                result, units = self.runner_units(host, *extra, check=False)
                self.assertEqual((result.returncode, units), (2, []))
                self.assertIn(message, result.stderr)

    def test_a_policy_needs_block_p_at_seed_11(self):
        host = self.host()
        for policy in TEXTS:
            for selection in ((), ("--blocks", "P"), ("--seeds", "11"), ("--blocks", "P,S-VC", "--seeds", "11"),
                              ("--blocks", "P", "--seeds", "11,23")):
                with self.subTest(policy=policy, selection=selection):
                    result = host.run("runner", "--run", "golden1", *selection, "--final-turn-policy", policy,
                                      check=False)
                    self.assertEqual((result.returncode, G.units(host.sudo_calls(), "runner")), (2, []))
                    self.assertIn(f"--final-turn-policy {policy}: block P at seed 11 or 23 only (one seed per launch), "
                                  "pass --blocks P --seeds 11 or --seeds 23, optionally --conditions from C0,C1 "
                                  "(protocols/EXPA_FINALTURN.md)", result.stderr)
        # none (or no option) is the prereg harness: any selection still starts its unit.
        for extra in ((), ("--final-turn-policy", "none")):
            with self.subTest(extra=extra):
                host.run("runner", "--run", "golden1", *extra)
                units = G.units(host.sudo_calls(), "runner")
                self.assertEqual(len(units), 1)
                self.assertFalse({"--blocks", "--seeds", "--final-turn-policy"} & set(units[0]))

    def test_a_policy_needs_the_default_model_profile(self):
        for profile in ("gemma4-12b", "gemma4-31b-qat"):
            with self.subTest(profile=profile):
                host = self.host("--model-profile", profile)
                result, units = self.runner_units(host, "--final-turn-policy", "deadline", check=False)
                self.assertEqual((result.returncode, units), (2, []))
                self.assertIn("--final-turn-policy deadline: registered only with the default budget (8 responses, "
                              "16384 tokens) and model profile gemma4-31b", result.stderr)
                _, units = self.runner_units(host, "--final-turn-policy", "none")
                self.assertEqual(units, [G.GOLDEN["launch"]["runner_units"][0] + ["--model-profile", profile]])

    def test_the_script_lists_the_harness_policies(self):
        text = (G.PROJECT / "scripts" / "expA_core1_launch.sh").read_text()
        body = text[text.index("    local policy=()"):]
        body = body[:body.index("    esac\n")]
        self.assertIn('        ""|' + harness.FINAL_TURN_DEFAULT + ") ;;", body)
        self.assertIn("        " + "|".join(harness.FINAL_TURN_TEXTS) + ")", body)
        self.assertIn("must be " + ", ".join(harness.FINAL_TURN_POLICIES[:-1]) + " or "
                      + harness.FINAL_TURN_POLICIES[-1], body)


if __name__ == "__main__":
    unittest.main()
