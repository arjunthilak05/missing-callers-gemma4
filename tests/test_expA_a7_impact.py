"""Exp A follow-up A7: the impact-analysis task framing (protocols/EXPA_A7_IMPACT.md). SYNTHETIC fixtures.

(1) The config records and prompts (any OS): every earlier record (default, A1, A4, A3, A5, the 16-response design,
    A6), prompt_templates(), both design tables and the default rendered prompts are byte-identical to those of the
    commit before A7 (sha256 pinned from 97c3e4814); the impact record names the protocol, the exact text and its
    sha256; the impact task paragraph is the protocol's text exactly, and nothing else in the prompt changes; the
    validation refuses every other combination.
(2) The harness against the fake server (POSIX): an impact episode sends the impact paragraph as the first user
    message and records the framing in its config receipts.
(3) The runner (POSIX): impact is registered only with the prereg budget, the 31B profile, no other option and the
    A7 selection (--blocks P,S-VC --seeds 11 --conditions C0,C1), which selects exactly three arms (2,250 cells on the
    750 main items, 102 on the 34 dev items); dry run, run.json, the order guard and an episode of a live run.
(4) The launch script (POSIX + bash): impact is forwarded with the A7 selection only; its neighbours are refused
    before any unit.
(5) The dev check (scripts/expA_a7_devcheck.py) on SYNTHETIC runs (any OS): its frozen values are the harness's; GO,
    the exact 95% rule and each failing check.
(6) The analysis (scripts/expA_a7_analysis.py) on a hand-computed three-family fixture (any OS), and its refusals.
ANALYSIS RECORD.
"""
import collections
import contextlib
import hashlib
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
sys.path.insert(0, str(PROJECT / "tests"))

from absent_edges.data import canonical_json  # noqa: E402
from absent_edges.probe import harness  # noqa: E402

POSIX = os.name == "posix"
if POSIX:
    import test_expA_default_golden as G      # puts the project on sys.path
    import test_probe_harness as H
    from absent_edges.models import LocalModel, ModelProfile
    from absent_edges.probe import ledger as L
    from absent_edges.probe.tools import ToolBox

PROTOCOL = "protocols/EXPA_A7_IMPACT.md"
# The protocol's task paragraph, typed again from protocols/EXPA_A7_IMPACT.md section 2.
IMPACT_TEXT = ("Task: the signature of the function `{qualname}` (defined in `{path}`, line {line}) is about to "
               "change. Find every place in the library source that must be updated: every place where it is "
               "invoked, either called directly or passed as a callable that gets invoked, from inside another "
               "function or method. Consider only files under {package_dirs}. Ignore tests, docs, examples and "
               "module-level code.")
IMPACT_TEXT_SHA256 = "b2b161fc0f39cf524e2d8dee2d4d9a2fba5e42969e54eda73e62ad6d3a0c037c"
TARGET = "pkg/core.py::target"
IMPACT_TASK = ("Task: the signature of the function `target` (defined in `pkg/core.py`, line 1) is about to change. "
               "Find every place in the library source that must be updated: every place where it is invoked, either "
               "called directly or passed as a callable that gets invoked, from inside another function or method. "
               "Consider only files under `pkg/`. Ignore tests, docs, examples and module-level code.")
SUBMIT_TEXT = ("Submit each invocation site as a file path and line number with submit_answer, and state whether you "
               "believe your list is complete (yes, no or unknown).")
# sha256 of json.dumps(value) at the commit before A7 (97c3e4814); the A6 values as pinned by
# tests/test_expA_a6_budget16.py.
PINNED = {"none": "ff8921f6e5ef35bfbacc58037ac23081468f43a9404ea8a62ed174f1ffd9df4b",
          "neutral": "2aab8a084aedf3c705a68e9600fd61aeccc68364e9aa2a07a93fe989ca9b82a4",
          "deadline": "773cb89af25df5d771e6248f7d0d45246f35045c7a803a2a8f1a9a956bb4819b",
          "deadline_info": "0da3b96b303c2918c59964a518aac7b76483fba1e56f4b010f933bbea7dd7ed4",
          "submit_only": "da0a6ef7e5007d72b346455615de554ca163bc1ad304d3f4578edcd0ebe10709",
          "a3_off": "b1d564984b0012b8d7cff6649131da96375d38577825a6ccdd64b9a9154c00a1",
          "a3_on": "c4ae1e0f20a92e0c49b574b9826e3f541fbe735823a45ef37b2240f56cfa1ef8",
          "a5_neutral": "90a42da86ce292942b2d635ac310b7f314bc1c9b2b490d6958ee1d2dfb540485",
          "a5_ledger": "3274dbee205b4028057aee083c50596291cd53d710d3dbb7db911a8081d4eda0",
          "budget16": "f52afc9af37444d4bee8917903e2e69937d6a8793fb30823f8c811bf3b6e63f7",
          "a6": "baf1d18491ae410d6cebbcf474a69dc310b031985283aa0c6baa37af1c3c57db",
          "prompts": "ff8eedeb4d7fb506bc4642c514160edd454ac8b38742e2f48196f3651dc203c3",
          "design_default": "6b0a49cd8eb8e6157e2d8d516f0ca2231e92b4e494c178b1c28af838ebdedb5c",
          "design_a3": "723f7aa95dd4494c2b174ac09ddc7aa353a52dcbe48297f47354ac8a76c44d0c",
          "render_C0": "e076690cc9305961b9d3142a2c9b4f3fd138c5b1e8ba7f2c5ba7198fffce84c8",
          "render_C1": "bea3bf5fadd8aca1b7a2dd0e46e5d0f4af51ea4f365fc582798a2872f258af38"}
IMPACT_RECORD_SHA256 = "48264664f8fc7db4e916699244d1999cdd44ca2ff4d72992d8ac2b1eb0dd9165"   # frozen with this commit
RENDER_KEYS = ("system", "user", "task_paragraph", "answer_block", "tools_offered", "tools")
A7_ARM = ("--task-framing", "impact")
A7_SELECTION = ("--blocks", "P,S-VC", "--seeds", "11", "--conditions", "C0,C1")
A7_ARMS = {("P", "none", "C0", 11), ("P", "omit_node", "C1", 11), ("S-VC", "vcomplete", "C1", 11)}
MANIFESTS = PROJECT / "manifests" / "expA" / "prod-0b8ee0b7c"


def sha(text):
    return hashlib.sha256(text if isinstance(text, bytes) else text.encode("utf-8")).hexdigest()


def digest(value):
    return sha(json.dumps(value, ensure_ascii=False))


def impact_config():
    return harness.EpisodeConfig(task_framing="impact")


def episode(condition):
    """A SYNTHETIC block-P cell of item-a (C0: no list; C1: an empty incomplete list)."""
    payload = canonical_json({"query": {"symbol": TARGET, "relation": "invokes", "direction": "incoming"},
                              "results": []})
    extra = dict(payload=payload, payload_sha256=sha(payload)) if condition == "C1" else {}
    return harness.EpisodeInput(run_id="r", cell_id="c", attempt_id="c.a001", token=1, block="P", item_id="item-a",
                                family="fam", target=TARGET, target_line=1, package_dirs=("pkg",),
                                variant="none" if condition == "C0" else "omit_node", condition=condition, seed=11,
                                wording_pair="A", **extra)


# ---------------------------------------------------------------------------------------------------------------
# (1) the config records and the prompts
# ---------------------------------------------------------------------------------------------------------------

class A7ConfigTests(unittest.TestCase):
    maxDiff = None

    def test_earlier_records_and_prompts_are_byte_identical(self):
        E = harness.EpisodeConfig
        configs = {"none": E(), "neutral": E(final_turn_policy="neutral"), "deadline": E(final_turn_policy="deadline"),
                   "deadline_info": E(final_turn_policy="deadline_info"),
                   "submit_only": E(final_turn_policy="submit_only"),
                   "a3_off": E(final_turn_policy="deadline", evidence_ledger="off"),
                   "a3_on": E(final_turn_policy="deadline", evidence_ledger="on"),
                   "a5_neutral": E(final_turn_policy="deadline", submit_review="neutral"),
                   "a5_ledger": E(final_turn_policy="deadline", submit_review="ledger"),
                   "budget16": E(max_model_calls=16, context_budget=32768),
                   "a6": E(max_model_calls=16, context_budget=32768, final_turn_policy="deadline",
                           evidence_ledger="off")}
        for name, config in configs.items():
            with self.subTest(name=name):
                self.assertEqual(digest(config.record()), PINNED[name])
                self.assertNotIn("task_framing", config.record())
        # An explicit default is the default config, and so the default record.
        self.assertEqual(E(task_framing="default"), E())
        self.assertEqual(digest(harness.prompt_templates()), PINNED["prompts"])
        self.assertEqual(harness.prompt_templates()["user_task"], harness.USER_TASK_TEMPLATE)
        self.assertEqual(digest(harness.design_table_for(None)), PINNED["design_default"])
        self.assertEqual(digest(harness.design_table_for("a3-ledger")), PINNED["design_a3"])
        for condition in ("C0", "C1"):
            with self.subTest(condition=condition):
                rendered = harness.render_messages(episode(condition), E())
                self.assertEqual(digest({key: rendered[key] for key in RENDER_KEYS}), PINNED["render_" + condition])
                rendered = harness.render_messages(episode(condition), E(task_framing="default"))
                self.assertEqual(digest({key: rendered[key] for key in RENDER_KEYS}), PINNED["render_" + condition])

    def test_the_impact_text_is_the_protocols(self):

        if not (__import__("pathlib").Path(__file__).resolve().parents[1] / "protocols").is_dir():

            self.skipTest("the protocol texts ship in the dataset package (protocol/)")
        self.assertEqual(harness.IMPACT_TASK_TEMPLATE, IMPACT_TEXT)
        self.assertEqual(sha(IMPACT_TEXT), IMPACT_TEXT_SHA256)
        self.assertEqual((harness.TASK_FRAMING_PROTOCOL, harness.TASK_FRAMINGS), (PROTOCOL, ("default", "impact")))
        protocol = (PROJECT / PROTOCOL).read_text(encoding="utf-8")
        flowing = " ".join(protocol.split())
        self.assertIn(IMPACT_TEXT, flowing)
        # The placeholders and the inline code spans are those of the default template.
        for token in ("`{qualname}`", "`{path}`", "line {line}", "{package_dirs}"):
            self.assertEqual((IMPACT_TEXT.count(token), harness.USER_TASK_TEMPLATE.count(token)), (1, 1), token)
        self.assertTrue(IMPACT_TEXT.endswith("Ignore tests, docs, examples and module-level code."))
        self.assertTrue(harness.USER_TASK_TEMPLATE.endswith("Ignore tests, docs, examples and module-level code."))

    def test_the_impact_record(self):
        record = impact_config().record()
        self.assertEqual(digest(record), IMPACT_RECORD_SHA256)
        default = harness.EpisodeConfig().record()
        self.assertEqual(list(record), list(default) + ["task_framing", "task_framing_rules"])
        self.assertEqual({key: record[key] for key in default}, default)
        self.assertEqual(record["task_framing"], "impact")
        rules = record["task_framing_rules"]
        self.assertEqual((rules["protocol"], rules["framing"], rules["text"], rules["text_sha256"]),
                         (PROTOCOL, "impact", IMPACT_TEXT, IMPACT_TEXT_SHA256))
        self.assertEqual(harness.config_design(impact_config()), None)       # the default design (blocks P, S-VC)

    def test_the_impact_prompt_changes_only_the_task_paragraph(self):
        for condition in ("C0", "C1"):
            with self.subTest(condition=condition):
                default = harness.render_messages(episode(condition), harness.EpisodeConfig())
                impact = harness.render_messages(episode(condition), impact_config())
                self.assertEqual(impact["task_paragraph"], IMPACT_TASK)
                self.assertEqual(impact["user"], IMPACT_TASK + default["answer_block"] + "\n\n" + SUBMIT_TEXT)
                self.assertEqual(impact["user"], default["user"].replace(default["task_paragraph"], IMPACT_TASK, 1))
                for key in ("system", "answer_block", "payload_text", "recommendation_present", "tools",
                            "tools_offered"):
                    self.assertEqual(impact[key], default[key], key)
                self.assertTrue(default["task_paragraph"].startswith("Task: find every place in the library source "
                                                                     "where the function `target`"))
        rendered = harness.render_messages(episode("C1"), impact_config())
        self.assertIn('A code-graph index returned this answer for incoming `invokes` relationships of '
                      '`pkg/core.py::target`:\n', rendered["user"])
        self.assertIn("You have at most 8 responses", rendered["system"])

    def test_validation(self):
        E = harness.EpisodeConfig
        message = ("the impact task framing runs with the prereg budget (8 responses, 16384 tokens), no final-turn "
                   "notice, no evidence ledger and no submit review")
        for kwargs in ({"max_model_calls": 16, "context_budget": 32768}, {"max_model_calls": 16},
                       {"context_budget": 32768}, {"max_model_calls": 7}, {"final_turn_policy": "deadline"},
                       {"final_turn_policy": "neutral"}, {"final_turn_policy": "deadline_info"},
                       {"final_turn_policy": "submit_only"},
                       {"final_turn_policy": "deadline", "evidence_ledger": "off"},
                       {"final_turn_policy": "deadline", "evidence_ledger": "on"},
                       {"final_turn_policy": "deadline", "submit_review": "neutral"},
                       {"final_turn_policy": "deadline", "submit_review": "ledger"},
                       {"max_model_calls": 16, "context_budget": 32768, "final_turn_policy": "deadline",
                        "evidence_ledger": "off"}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError) as raised:
                    E(task_framing="impact", **kwargs)
                self.assertEqual(str(raised.exception), message)
        for value in ("", "Impact", "none", "original", None, 1):
            with self.subTest(value=value), self.assertRaises(ValueError) as raised:
                E(task_framing=value)
            self.assertEqual(str(raised.exception), "task_framing must be one of default, impact")
        with self.assertRaises(ValueError):
            harness.task_framing_record("default")
        # Every other option keeps its own validation under the default framing.
        self.assertEqual(E(final_turn_policy="deadline", evidence_ledger="off").task_framing, "default")


# ---------------------------------------------------------------------------------------------------------------
# (2) the harness: the first user message of an impact episode
# ---------------------------------------------------------------------------------------------------------------

@unittest.skipUnless(POSIX, "the ToolBox search scans with POSIX no-follow reads")
class A7EpisodeTests(unittest.TestCase):
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

    def run_episode(self, condition, config):
        server = H.FakeServer([{"calls": [("bash", {"command": "grep -rn target ."})]}, H.SUBMIT])
        self.servers.append(server)
        self.count += 1
        directory = Path(self.temporary.name) / f"attempt-{self.count}"
        directory.mkdir()
        client = LocalModel(ModelProfile(endpoint=server.endpoint), timeout=20)
        toolbox = ToolBox(self.root, ["pkg"], H.FakeBash("pkg/core.py:6:    return target(1)\n"))
        result, digest_ = harness.run_episode(H.make_episode(condition), directory, client=client, toolbox=toolbox,
                                              config=config)
        self.assertEqual(digest_, L.sha256_path(directory / "result.json"))
        return result, server, H.events(directory)

    def test_an_impact_episode(self):
        for condition in ("C0", "C1"):
            with self.subTest(condition=condition):
                default, default_server, _ = self.run_episode(condition, harness.EpisodeConfig())
                result, server, rows = self.run_episode(condition, impact_config())
                self.assertEqual((result["exit_status"], result["calls"]), ("Submitted", 2))
                chats = server.chat_requests()
                first = chats[0]["messages"]
                self.assertEqual([message["role"] for message in first], ["system", "user"])
                self.assertTrue(first[1]["content"].startswith(IMPACT_TASK))
                self.assertTrue(first[1]["content"].endswith("\n\n" + SUBMIT_TEXT))
                self.assertNotIn("Task: find every place", json.dumps(chats))
                baseline = default_server.chat_requests()[0]["messages"]
                self.assertEqual(first[0], baseline[0])
                self.assertEqual(first[1]["content"],
                                 baseline[1]["content"].replace(H.TASK_TEXT, IMPACT_TASK, 1))
                self.assertEqual(chats[0]["tools"], default_server.chat_requests()[0]["tools"])
                setup = rows[0]["value"]
                self.assertEqual((setup["config"], result["config"]), (impact_config().record(),) * 2)
                self.assertEqual(setup["user"], first[1]["content"])
                self.assertEqual(result["prompt"]["user_sha256"], sha(first[1]["content"]))
                self.assertNotIn("task_framing", default["config"])
                self.assertNotIn("[harness notice]", json.dumps(chats))


# ---------------------------------------------------------------------------------------------------------------
# (3) the runner
# ---------------------------------------------------------------------------------------------------------------

class ManifestView:
    """A SYNTHETIC dispatch view with the item count per family of a frozen dispatch manifest (no truth, no payload
    bytes): what build_cells reads (order, rows: family, target, wording pair, membership, payload hashes)."""

    def __init__(self, manifest_path):
        manifest = json.loads(Path(manifest_path).read_bytes())
        self.rows = {}
        for family, count in sorted(manifest["by_family"].items()):
            for index in range(count):
                item = f"{family}-{index:03d}"
                flags = {flag: index < manifest["membership_counts"][flag] // len(manifest["by_family"])
                         for flag in ("in_s_nat", "in_s_edge", "in_t")}
                cells = collections.defaultdict(dict)
                for key in manifest["payload_cell_counts"]:
                    variant, condition = key.split("/")
                    cells[variant][condition] = sha(f"{item}|{key}")
                self.rows[item] = {"family": family, "target": {"identity": f"pkg/m.py::f{index}"},
                                   "wording_pair": "AB"[index % 2], "membership": flags, "cells": dict(cells)}
        self.order = sorted(self.rows, key=harness.item_order_key)
        self.families = sorted(manifest["by_family"])
        self.rows_count = manifest["rows"]


@unittest.skipUnless(POSIX, "the runner reads checkouts with POSIX no-follow reads")
class A7RunnerTests(unittest.TestCase):
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
        self.assertEqual(self.runner.REGISTERED_FRAMING_DESIGNS,
                         {(8, 16384, "none", "none", "none", "impact"): (PROTOCOL, ("gemma4-31b",))})
        self.assertEqual(self.runner.A7_SELECTION, {"blocks": {"P", "S-VC"}, "seeds": {11}, "conditions": {"C0", "C1"}})
        # Every earlier registry is unchanged.
        self.assertEqual(self.runner.REGISTERED_BUDGETS, {(8, 16384), (16, 32768)})
        self.assertEqual(set(self.runner.REGISTERED_DESIGNS), {
            (8, 16384, "none"), (16, 32768, "none"), (8, 16384, "neutral"), (8, 16384, "deadline"),
            (8, 16384, "deadline_info"), (8, 16384, "submit_only")})
        self.assertEqual(set(self.runner.REGISTERED_LEDGER_DESIGNS),
                         {(8, 16384, "deadline", "off"), (8, 16384, "deadline", "on")})
        self.assertEqual(set(self.runner.REGISTERED_A6_DESIGNS), {(16, 32768, "deadline", "off")})
        for extra in ((), ("--model-profile", "gemma4-31b"), ("--families", "fam"), ("--blocks", "S-VC,P"),
                      ("--conditions", "C1,C0"), ("--final-turn-policy", "none", "--evidence-ledger", "none",
                                                  "--submit-review", "none")):
            with self.subTest(extra=extra):
                args = self.parse(*A7_ARM, *A7_SELECTION, *extra)
                self.assertEqual((args.episode_config, args.design, args.blocks, args.seeds, args.conditions),
                                 (impact_config(), None, {"P", "S-VC"}, {11}, {"C0", "C1"}))
        # The default framing (explicit or not) is the default config, with any selection.
        for extra in ((), ("--task-framing", "default"), ("--task-framing", "default", "--blocks", "P,S-VC")):
            with self.subTest(extra=extra):
                self.assertEqual(self.parse(*extra).episode_config, harness.EpisodeConfig())

    def test_neighbours_are_refused(self):
        option = ("--task-framing impact (protocols/EXPA_A7_IMPACT.md) is registered only with the prereg budget (8 "
                  "responses, 16384 tokens), --model-profile gemma4-31b and no --final-turn-policy, --evidence-ledger "
                  "or --submit-review")
        selection = ("--task-framing impact (protocols/EXPA_A7_IMPACT.md) runs only the A7 selection: pass --blocks "
                     "P,S-VC --seeds 11 --conditions C0,C1")
        cases = (
            (A7_SELECTION + ("--max-model-calls", "16", "--context-budget", "32768"), option),
            (A7_SELECTION + ("--model-profile", "gemma4-12b"), option),
            (A7_SELECTION + ("--model-profile", "gemma4-31b-qat"), option),
            (A7_SELECTION + ("--final-turn-policy", "deadline"), option),
            (A7_SELECTION + ("--final-turn-policy", "neutral"), option),
            (("--blocks", "LEDGER", "--final-turn-policy", "deadline", "--evidence-ledger", "off"), option),
            (("--blocks", "LEDGER", "--final-turn-policy", "deadline", "--submit-review", "neutral"), option),
            ((), selection),
            (("--blocks", "P", "--seeds", "11", "--conditions", "C0,C1"), selection),
            (("--blocks", "P,S-VC", "--seeds", "11"), selection),
            (("--blocks", "P,S-VC", "--conditions", "C0,C1"), selection),
            (("--blocks", "P,S-VC", "--seeds", "11,23", "--conditions", "C0,C1"), selection),
            (("--blocks", "P,S-VC", "--seeds", "23", "--conditions", "C0,C1"), selection),
            (("--blocks", "P,S-VC", "--seeds", "11", "--conditions", "C0,C1,C3"), selection),
            (("--blocks", "P,S-VC", "--seeds", "11", "--conditions", "C1"), selection),
            (("--blocks", "P,S-VC,S-NAT", "--seeds", "11", "--conditions", "C0,C1"), selection),
            (("--blocks", "S-VC", "--seeds", "11", "--conditions", "C1"), selection),
        )
        for argv, message in cases:
            with self.subTest(argv=argv):
                self.assertIn(message, self.refused(*A7_ARM, *argv))
        self.assertIn("invalid choice: 'original'", self.refused("--task-framing", "original", *A7_SELECTION))

    def test_the_a7_selection_is_three_arms(self):
        for name, rows, expected in (("dispatch-main", 750, 2250), ("dispatch-dev", 34, 102)):
            with self.subTest(dispatch=name):
                view = ManifestView(MANIFESTS / name / "dispatch_manifest.json")
                self.assertEqual((len(view.rows), view.rows_count), (rows, rows))
                cells = self.runner.build_cells(view, None, design=None)
                args = self.parse(*A7_ARM, *A7_SELECTION)
                selected = self.runner.select_cells(cells, args.blocks, args.seeds, args.families, args.conditions)
                self.assertEqual(len(selected), expected)
                arms = collections.Counter((c["block"], c["variant"], c["condition"], c["seed"]) for c in selected)
                self.assertEqual(arms, {arm: rows for arm in A7_ARMS})
                self.assertEqual({c["item_id"] for c in selected}, set(view.rows))
                # Dispatch order: block P seed 11 (an item's C0 then C1), then S-VC seed 11.
                self.assertEqual([c["block"] for c in selected], ["P"] * (2 * rows) + ["S-VC"] * rows)
                self.assertEqual([c["condition"] for c in selected[:4]], ["C0", "C1", "C0", "C1"])

    def test_dry_run_run_json_order_guard_and_an_episode(self):
        argv = ["--dispatch", str(self.dispatch), "--dry-run", *A7_ARM, *A7_SELECTION, "--show", "6"]
        code, output = G.invoke_runner(self.runner, argv)
        summary = json.loads(output)
        self.assertEqual((code, summary["selected_cells"], summary["by_block"]), (0, 6, {"P": 4, "S-VC": 2}))
        self.assertEqual((summary["episode_config"], summary["design"]), (impact_config().record(),
                                                                          harness.design_table_for(None)))
        shown = json.dumps(summary)
        self.assertIn("is about to change. Find every place in the library source that must be updated", shown)
        self.assertNotIn("Task: find every place", shown)
        server = H.FakeServer(default=H.SUBMIT)
        self.servers.append(server)
        live = G.live_args(self.base, self.dispatch, self.manifest, server, run="a7")
        for flag, value in (("--blocks", "P,S-VC"), ("--seeds", "11"), ("--limit", "6")):
            live[live.index(flag) + 1] = value
        live += ["--conditions", "C0,C1", *A7_ARM]
        # A fresh run: P seed 11 (C2, C3, C4a) and P seeds 23, 47 are open before S-VC (D10.8).
        with mock.patch.object(G.tools, "ReadOnlyBash", G.FakeJailBash):
            code, output = G.invoke_runner(self.runner, live)
        self.assertEqual(code, 2)
        refusal = json.loads(output)
        self.assertEqual(refusal["status"], "refused")
        self.assertEqual(refusal["order"]["selected_segments"], ["P|s11", "S-VC|s11"])
        self.assertEqual(refusal["order"]["earlier_open_cells_outside_selection"],
                         {"P|s11": 6, "P|s23": 10, "P|s47": 10})
        reason = "A7-registers-P-C0-C1-and-S-VC-C1-at-seed-11-only"
        with mock.patch.object(G.tools, "ReadOnlyBash", G.FakeJailBash):
            code, _ = G.invoke_runner(self.runner, live + ["--allow-out-of-order", reason])
        self.assertEqual(code, 0)
        run_dir = self.base / "a7"
        run_json = json.loads((run_dir / "run.json").read_bytes())
        self.assertEqual(run_json["episode_config"], impact_config().record())
        self.assertEqual((run_json["design"]["block_order"], run_json["prompts"]),
                         (list(harness.BLOCK_ORDER), harness.prompt_templates()))
        launches = [p for p in (run_dir / "launches").iterdir() if not p.name.endswith(".end.json")]
        selection = json.loads(launches[-1].read_bytes())["selection"]
        self.assertEqual((selection["blocks"], selection["seeds"], selection["conditions"], selection["cells"],
                          selection["order"]["override_reason"]), (["P", "S-VC"], [11], ["C0", "C1"], 6, reason))
        results = [json.loads(path.read_bytes()) for path in sorted((run_dir / "cells").rglob("result.json"))]
        self.assertEqual({(r["block"], r["variant"], r["condition"], r["seed"]) for r in results}, A7_ARMS)
        self.assertEqual(len(results), 6)
        for result in results:
            self.assertEqual((result["exit_status"], result["config"]), ("Submitted", impact_config().record()))
        chats = server.chat_requests()
        self.assertEqual(len(chats), 6)
        for chat in chats:
            self.assertTrue(chat["messages"][1]["content"].startswith("Task: the signature of the function `"))
        self.assertNotIn("Task: find every place", json.dumps(chats))


# ---------------------------------------------------------------------------------------------------------------
# (4) the launch script
# ---------------------------------------------------------------------------------------------------------------

@unittest.skipUnless(POSIX and shutil.which("bash"), "Linux only: the launch script runs on Core1")
class A7LaunchScriptTests(unittest.TestCase):
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

    def test_impact_is_forwarded_with_the_a7_selection(self):
        host = self.host()
        golden = G.GOLDEN["launch"]["runner_units"][0]
        self.assertEqual(golden[-4:], ["--blocks", "P", "--seeds", "11"])
        reason = "A7-registers-P-C0-C1-and-S-VC-C1-at-seed-11-only"
        result, units = self.runner_units(host, *A7_SELECTION, "--allow-out-of-order", reason, *A7_ARM)
        self.assertEqual(units, [golden[:-4] + ["--blocks", "P,S-VC", "--seeds", "11", "--conditions", "C0,C1",
                                                "--allow-out-of-order", reason, "--task-framing", "impact"]])
        self.assertIn("task framing impact (protocols/EXPA_A7_IMPACT.md)", result.stderr)
        args = self.runner_args(units[0])
        self.assertEqual((args.episode_config, args.design, args.blocks, args.seeds, args.conditions,
                          args.allow_out_of_order), (impact_config(), None, {"P", "S-VC"}, {11}, {"C0", "C1"}, reason))
        # default (or no option) adds nothing: the frozen command.
        for extra in ((), ("--task-framing", "default")):
            with self.subTest(extra=extra):
                result, units = self.runner_units(host, "--blocks", "P", "--seeds", "11", *extra)
                self.assertEqual(units, [golden])
                self.assertNotIn("task framing", result.stderr)

    def test_neighbours_are_refused_before_any_unit(self):
        host = self.host()
        option = ("--task-framing impact: registered only with the default budget (8 responses, 16384 tokens), model "
                  "profile gemma4-31b and no --final-turn-policy, --evidence-ledger or --submit-review "
                  "(protocols/EXPA_A7_IMPACT.md)")
        selection = ("--task-framing impact: the A7 selection only, pass --blocks P,S-VC --seeds 11 --conditions C0,C1 "
                     "(protocols/EXPA_A7_IMPACT.md)")
        cases = ((A7_SELECTION + ("--max-model-calls", "16", "--context-budget", "32768"), option),
                 (("--blocks", "P", "--seeds", "11", "--final-turn-policy", "deadline", "--conditions", "C0,C1"),
                  option),
                 (("--blocks", "LEDGER", "--final-turn-policy", "deadline", "--evidence-ledger", "off"), option),
                 (("--blocks", "LEDGER", "--final-turn-policy", "deadline", "--submit-review", "neutral"), option),
                 ((), selection),
                 (("--blocks", "P", "--seeds", "11", "--conditions", "C0,C1"), selection),
                 (("--blocks", "P,S-VC", "--seeds", "11"), selection),
                 (("--blocks", "P,S-VC", "--seeds", "23", "--conditions", "C0,C1"), selection),
                 (("--blocks", "P,S-VC", "--seeds", "11", "--conditions", "C0,C1,C3"), selection))
        for extra, message in cases:
            with self.subTest(extra=extra):
                result, units = self.runner_units(host, *extra, *A7_ARM, check=False)
                self.assertEqual((result.returncode, units), (2, []))
                self.assertIn(message, result.stderr)
        result, units = self.runner_units(host, *A7_SELECTION, "--task-framing", "original", check=False)
        self.assertEqual((result.returncode, units), (2, []))
        self.assertIn("--task-framing must be default or impact, not original", result.stderr)

    def test_impact_needs_the_default_model_profile(self):
        for profile in ("gemma4-12b", "gemma4-31b-qat"):
            with self.subTest(profile=profile):
                host = self.host("--model-profile", profile)
                result, units = self.runner_units(host, *A7_SELECTION, *A7_ARM, check=False)
                self.assertEqual((result.returncode, units), (2, []))
                self.assertIn("--task-framing impact: registered only with the default budget", result.stderr)


# ---------------------------------------------------------------------------------------------------------------
# (5) the dev check (scripts/expA_a7_devcheck.py) on SYNTHETIC runs
# ---------------------------------------------------------------------------------------------------------------

def load_module(name, path):
    module = sys.modules.get(name)
    if module is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module


D = load_module("expA_a7_devcheck", PROJECT / "scripts" / "expA_a7_devcheck.py")
ENCODE = (lambda name: name) if POSIX else (lambda name: name.replace("|", "%7C"))
DECODE = None if POSIX else (lambda name: name.replace("%7C", "|"))
RUN_ID = "a7impact-dev-SYNTHETIC"
DEV_ITEMS = [f"item-{index}" for index in range(7)]
DIRS = {"item-0": ["pkg"], "item-1": ["pkg", "extra.py"], "item-2": ["."]}
A7_CELLS = [f"{block}|{item}|{variant}|{condition}|s11" for item in DEV_ITEMS
            for block, variant, condition in (("P", "none", "C0"), ("P", "omit_node", "C1"))]
A7_CELLS += [f"S-VC|{item}|vcomplete|C1|s11" for item in DEV_ITEMS]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def cell_input(cell):
    block, item, variant, condition, seed = cell.split("|")
    index = int(item.split("-")[1])
    payload = None
    if condition == "C1":
        payload = canonical_json({"query": {"symbol": f"pkg/m{index}.py::Klass.f{index}", "relation": "invokes",
                                            "direction": "incoming"}, "results": []})
    return harness.EpisodeInput(run_id=RUN_ID, cell_id=cell, attempt_id=cell + ".a001", token=1, block=block,
                                item_id=item, family="synthetic", target=f"pkg/m{index}.py::Klass.f{index}",
                                target_line=index + 3, package_dirs=tuple(DIRS.get(item, ["pkg"])), variant=variant,
                                condition=condition, seed=int(seed[1:]), wording_pair="A", payload=payload,
                                payload_sha256=None if payload is None else sha(payload))


def write_dev_attempt(run_dir, cell, ordinal, outcome, config, framing_config):
    """One SYNTHETIC attempt: ``outcome`` an exit (Submitted, NoAnswer, ContextLimit, FormatExhausted), "early_limit"
    (ContextLimit before the first request) or "infra" (InfraInterrupted)."""
    attempt_id = f"{cell}.a{ordinal:03d}"
    directory = run_dir / "cells" / ENCODE(cell) / ENCODE(attempt_id)
    directory.mkdir(parents=True)
    start = canonical({"cell_id": cell, "attempt_id": attempt_id, "ordinal": ordinal, "launch": "SYNTHETIC"})
    (directory / "attempt_start.json").write_bytes(start)
    rendered = harness.render_messages(cell_input(cell), framing_config)
    rows = [{"kind": "episode_setup", "value": {"config": config, "system": rendered["system"],
                                                "user": rendered["user"]}}]
    if outcome != "early_limit":
        rows.append({"kind": "model_exchange", "value": {"call_index": 0, "request": {"messages": [
            {"role": "system", "content": rendered["system"]}, {"role": "user", "content": rendered["user"]}]}}})
    exit_status = {"early_limit": "ContextLimit", "infra": "InfraInterrupted"}.get(outcome, outcome)
    rows.append({"kind": "episode_finished", "value": {"exit_status": exit_status}})
    for index, row in enumerate(rows):
        row["sequence"] = index
    events = b"".join(json.dumps(row).encode("utf-8") + b"\n" for row in rows)
    (directory / "events.jsonl").write_bytes(events)
    episode = cell_input(cell)
    result = {"schema": "expA-episode-result-v2", "run_id": RUN_ID, "cell_id": cell, "attempt_id": attempt_id,
              "exit_status": exit_status, "infra": {"kind": "server"} if outcome == "infra" else None,
              "target": episode.target, "cell": episode.describe(), "config": config,
              "prompt": {"user_sha256": sha(rendered["user"])},
              "events": {"path": "events.jsonl", "sha256": sha(events), "records": len(rows)},
              "run_context": {"launch": "SYNTHETIC", "start_sha256": sha(start)}}
    (directory / "result.json").write_bytes(canonical(result))
    return directory


def write_dev_run(directory, plan=None, *, episode_config=None, attempt_config=None, framing_config=None,
                  extra_dispatched=()):
    """A SYNTHETIC dev run: cells.jsonl holds the A7 cells and other default-design cells (never dispatched unless
    named in ``extra_dispatched``); ``plan`` maps a cell to its attempts' outcomes (default: Submitted)."""
    directory.mkdir(parents=True)
    design = list(A7_CELLS) + [f"P|{item}|omit_node|C2|s11" for item in DEV_ITEMS]
    design += [f"P|{item}|none|C0|s23" for item in DEV_ITEMS] + [f"S-VC|{item}|vcomplete|C3|s11" for item in DEV_ITEMS]
    cells = b"".join(canonical({"cell_id": cell}) + b"\n" for cell in design)
    (directory / "cells.jsonl").write_bytes(cells)
    record = impact_config().record()
    run = {"schema": "expA-run-SYNTHETIC", "run_id": RUN_ID, "cells_sha256": sha(cells), "cell_count": len(design),
           "infra_attempt_cap": 3, "episode_config": record if episode_config is None else episode_config}
    (directory / "run.json").write_bytes(canonical(run) + b"\n")
    plan = dict(plan or {})
    for cell in A7_CELLS:
        for ordinal, outcome in enumerate(plan.get(cell, ["Submitted"]), 1):
            config = (attempt_config or {}).get(cell, record)
            write_dev_attempt(directory, cell, ordinal, outcome, config,
                              (framing_config or {}).get(cell, impact_config()))
    for cell in extra_dispatched:
        write_dev_attempt(directory, cell, 1, "Submitted", record, impact_config())
    return directory


class A7DevCheckTests(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.count = 0

    def tearDown(self):
        self.temporary.cleanup()

    def run_dir(self, plan=None, **kwargs):
        self.count += 1
        return write_dev_run(self.base / f"run-{self.count}", plan, **kwargs)

    def check(self, run_dir):
        return D.check_run(run_dir, cell_name=DECODE)

    def test_the_frozen_values_are_the_harness(self):
        self.assertEqual(D.expected_episode_config(), impact_config().record())
        self.assertEqual(D.IMPACT_TEMPLATE, harness.IMPACT_TASK_TEMPLATE)
        self.assertEqual(D.DEFAULT_OPENING, harness.USER_TASK_TEMPLATE.split("{qualname}")[0])
        self.assertEqual(D.FROZEN_CONFIG, harness.EpisodeConfig().record())
        for cell in A7_CELLS[:2] + [A7_CELLS[2], A7_CELLS[4]]:
            with self.subTest(cell=cell):
                episode = cell_input(cell)
                self.assertEqual(D.impact_paragraph(episode.target, episode.target_line, list(episode.package_dirs)),
                                 harness.render_messages(episode, impact_config())["task_paragraph"])
        self.assertEqual({D.cell_arm(cell) for cell in A7_CELLS}, {"none", "incomplete", "complete"})
        for cell in ("P|i|omit_node|C2|s11", "P|i|none|C0|s23", "S-VC|i|vcomplete|C3|s11", "LEDGER|i|vcomplete|C1|s61",
                     "P|i|omit_node|C0|s11", "bad"):
            self.assertIsNone(D.cell_arm(cell), cell)

    def test_a_good_dev_run_is_go(self):
        plan = {A7_CELLS[0]: ["NoAnswer"], A7_CELLS[1]: ["infra", "Submitted"], A7_CELLS[2]: ["early_limit"]}
        report = self.check(self.run_dir(plan))
        self.assertEqual((report["go"], report["failed_checks"]), (True, []))
        self.assertEqual(list(report["checks"]), list(D.CHECKS))
        rule = report["go_rule"]
        # 21 completed: 19 Submitted, 1 NoAnswer, 1 ContextLimit -> 20/21.
        self.assertEqual((rule["completed"], rule["submitted_or_no_answer"], rule["passed"]), (21, 20, True))
        self.assertEqual(report["first_user_message"], {"requested": 20, "not_requested": 1})
        self.assertEqual((report["infra_attempts"], report["run"]["a7_cells"], report["run"]["design_cells"]),
                         (1, 21, 42))
        self.assertEqual(report["by_arm"]["none"], {"cell_states": {"completed": 7},
                                                    "exits": {"ContextLimit": 1, "NoAnswer": 1, "Submitted": 5}})
        self.assertEqual(report["impact_text_sha256"], IMPACT_TEXT_SHA256)
        self.assertTrue(D.summary(report).startswith(f"GO: A7 impact run {RUN_ID}: 20/21 completed episodes"))

    def test_the_go_rule_is_compared_exactly(self):
        # One cell missing after 3 infrastructure attempts (20 completed) and one ContextLimit: 19/20 = 95%.
        missing = {A7_CELLS[0]: ["infra"] * 3, A7_CELLS[1]: ["ContextLimit"]}
        report = self.check(self.run_dir(missing))
        self.assertEqual((report["go"], report["go_rule"]["completed"], report["go_rule"]["submitted_or_no_answer"]),
                         (True, 20, 19))
        self.assertEqual(report["by_arm"]["none"]["cell_states"], {"completed": 6, "missing_infrastructure": 1})
        self.assertEqual(report["infra_attempts"], 3)
        # Two non-terminal-by-model exits: 19/21 < 95%.
        report = self.check(self.run_dir({A7_CELLS[0]: ["FormatExhausted"], A7_CELLS[1]: ["ContextLimit"]}))
        self.assertEqual((report["go"], report["failed_checks"]), (False, ["go_rule"]))
        self.assertTrue(D.summary(report).startswith("NO-GO"))

    def test_config_and_first_message_failures(self):
        default = harness.EpisodeConfig().record()
        report = self.check(self.run_dir(episode_config=default))
        self.assertEqual(report["failed_checks"], ["run_json_episode_config"])
        report = self.check(self.run_dir(attempt_config={A7_CELLS[3]: default}))
        self.assertEqual((report["failed_checks"], report["checks"]["config_receipts"]["failures"]),
                         (["config_receipts"], 2))           # result.json and the episode_setup event
        wrong_text = impact_config().record()
        wrong_text["task_framing_rules"] = dict(wrong_text["task_framing_rules"],
                                                text_sha256=sha(IMPACT_TEXT + " "))
        report = self.check(self.run_dir(episode_config=wrong_text, attempt_config={A7_CELLS[5]: wrong_text}))
        self.assertEqual(report["failed_checks"], ["run_json_episode_config", "config_receipts"])
        # An attempt rendered with the default framing (its config receipts claim impact).
        report = self.check(self.run_dir(framing_config={A7_CELLS[6]: harness.EpisodeConfig()}))
        self.assertEqual(report["failed_checks"], ["first_user_message"])
        self.assertEqual(report["checks"]["first_user_message"]["failures"], 2)  # not the paragraph; the default one

    def test_selection_and_completeness(self):
        report = self.check(self.run_dir(extra_dispatched=["P|item-0|none|C0|s23"]))
        self.assertEqual(report["failed_checks"], ["selection"])
        self.assertIn("not an A7 cell", report["checks"]["selection"]["examples"][0])
        run_dir = self.run_dir()
        shutil.rmtree(run_dir / "cells" / ENCODE(A7_CELLS[-1]))
        report = self.check(run_dir)
        self.assertEqual(report["failed_checks"], ["selection", "run_complete"])
        report = self.check(self.run_dir({A7_CELLS[0]: ["infra"]}))         # still pending (1 of 3 attempts)
        self.assertEqual(report["failed_checks"], ["run_complete"])

    def test_records(self):
        run_dir = self.run_dir()
        directory = run_dir / "cells" / ENCODE(A7_CELLS[0]) / ENCODE(A7_CELLS[0] + ".a001")
        path = directory / "events.jsonl"
        path.write_bytes(path.read_bytes() + b'{"kind": "extra", "value": {}, "sequence": 9}\n')
        report = self.check(run_dir)
        self.assertEqual(report["failed_checks"], ["records"])
        run_dir = self.run_dir()
        write_dev_attempt(run_dir, A7_CELLS[1], 2, "Submitted", impact_config().record(), impact_config())
        report = self.check(run_dir)
        self.assertEqual(report["failed_checks"], ["records"])
        self.assertIn("after the terminal attempt", report["checks"]["records"]["examples"][0])

    def test_the_command_line(self):
        run_dir = self.run_dir()
        out = self.base / "go.json"
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = D.main(["--run-dir", str(run_dir), "--out", str(out)], cell_name=DECODE)
        self.assertEqual(code, 0)
        self.assertTrue(stdout.getvalue().startswith("GO: A7 impact run"))
        report = json.loads(out.read_text(encoding="ascii"))
        self.assertEqual((report["go"], report["label"]), (True, "PRESPECIFIED (protocols/EXPA_A7_IMPACT.md, written "
                                                                 "before any A7 cell); ANALYSIS RECORD"
                                                                 ""))
        stderr = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
            self.assertEqual(D.main(["--run-dir", str(run_dir), "--out", str(out)], cell_name=DECODE), 2)
        self.assertIn("refusing to overwrite", stderr.getvalue())
        nogo = self.run_dir(episode_config=harness.EpisodeConfig().record())
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(D.main(["--run-dir", str(nogo), "--out", str(self.base / "nogo.json")],
                                    cell_name=DECODE), 1)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(D.main(["--run-dir", str(self.base / "absent"), "--out", str(self.base / "x.json")]), 2)
        self.assertFalse((self.base / "x.json").exists())


# ---------------------------------------------------------------------------------------------------------------
# (6) the analysis (scripts/expA_a7_analysis.py) on a hand-computed SYNTHETIC fixture
# ---------------------------------------------------------------------------------------------------------------

A = load_module("expA_a7_analysis", PROJECT / "scripts" / "expA_a7_analysis.py")
ITEMS_SHA = "1" * 64
FAMILY_OF = {"i1": "f1", "i2": "f1", "i3": "f2", "i4": "f2", "i5": "f3", "i6": "f3", "i7": "f3"}
ARM_CELLS = {"none": ("P", "none", "C0"), "incomplete": ("P", "omit_node", "C1"),
             "complete": ("S-VC", "vcomplete", "C1")}
# site_recovered per item: (none, incomplete, complete); None = missing for infrastructure.
REC_IMPACT = {"i1": (0, 1, 1), "i2": (0, 0, 1), "i3": (0, 1, 1), "i4": (0, 1, 1), "i5": (1, 0, 1), "i6": (0, 0, 0),
              "i7": (None, 1, 0)}
REC_ORIGINAL = {"i1": (0, 0, 1), "i2": (0, 0, 0), "i3": (0, 1, 1), "i4": (1, 1, 1), "i5": (0, 0, 1), "i6": (0, 0, 1),
                "i7": (0, 1, 1)}
# Per (item, arm): changes to the default cell (Submitted with complete "yes" and 3 calls when recovered, else
# NoAnswer with 8 calls).
OVERRIDES_IMPACT = {("i1", "incomplete"): {"complete": "no"},
                    ("i2", "none"): {"exit_status": "ContextLimit", "calls": 7},
                    ("i6", "complete"): {"exit_status": "Submitted", "calls": 5, "complete": "no"}}


def score_row(item, arm, recovered, *, block=None, condition=None, variant=None, seed=11, state=None, **changes):
    """A SYNTHETIC scorer row (absent_edges/probe/score.py shape, the fields the analysis reads)."""
    default_block, default_variant, default_condition = ARM_CELLS.get(arm, (None, None, None))
    block, variant, condition = block or default_block, variant or default_variant, condition or default_condition
    state = state or ("missing_infrastructure" if recovered is None else "completed")
    done = state == "completed"
    exit_status = changes.get("exit_status") or (("Submitted" if recovered else "NoAnswer") if done else "Missing")
    submitted = done and exit_status == "Submitted"
    calls = changes.get("calls", 3 if submitted else 8) if done else None
    return {"schema": "expA-score-v2", "kind": "episode", "design": "d11",
            "cell_id": f"{block}|{item}|{variant}|{condition}|s{seed}", "block": block, "item_id": item,
            "family": FAMILY_OF[item], "variant": variant, "condition": condition, "seed": seed, "cell_state": state,
            "completed": done, "exit_status": exit_status, "site_recovered": recovered if done else None,
            "precision_lower": (1.0 if recovered else 0.0) if submitted else None,
            "precision_upper": (1.0 if recovered else 0.5) if submitted else None,
            "complete": changes.get("complete", "yes") if submitted else None,
            "cost": {"calls": calls} if done else None, "items_sha256": changes.get("items_sha", ITEMS_SHA)}


def framing_rows(recovery, overrides, *, others):
    rows = []
    for item, values in recovery.items():
        for arm, value in zip(ARM_CELLS, values):
            rows.append(score_row(item, arm, value, **overrides.get((item, arm), {})))
    for item in recovery:                       # cells of the default design outside the three arms
        rows.append(score_row(item, None, 1 if others == "completed" else 0, block="P", variant="omit_node",
                              condition="C2", state=others))
        rows.append(score_row(item, None, 0, block="P", variant="none", condition="C0", seed=23, state=others))
        rows.append(score_row(item, None, 1, block="S-VC", variant="vcomplete", condition="C3", state=others))
    return rows


def write_scores(directory, name, rows):
    path = Path(directory) / name
    path.write_bytes(b"".join(json.dumps(r, sort_keys=True).encode() + b"\n" for r in rows))
    return str(path)


def a7_run_json(directory, *, config=None, selection=None):
    """A SYNTHETIC A7 run.json and one launch record next to it."""
    directory = Path(directory)
    (directory / "launches").mkdir(parents=True)
    run = {"run_id": "a7impact-SYNTHETIC", "episode_config": impact_config().record() if config is None else config}
    (directory / "run.json").write_bytes(json.dumps(run).encode())
    launch = {"selection": selection or {"blocks": ["P", "S-VC"], "seeds": [11], "conditions": ["C0", "C1"],
                                         "families": None, "cells": 21}}
    (directory / "launches" / "20261108T000000Z-host.json").write_bytes(json.dumps(launch).encode())
    (directory / "launches" / "20261108T000000Z-host.end.json").write_bytes(b"{}")
    return str(directory / "run.json")


def p_df2(t):
    """Two-sided p of a t statistic with 2 degrees of freedom."""
    return 1 - abs(t) / math.sqrt(t * t + 2)


class A7AnalysisTests(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.impact = write_scores(self.base, "impact.jsonl",
                                   framing_rows(REC_IMPACT, OVERRIDES_IMPACT, others="not_dispatched"))
        self.original = write_scores(self.base, "original.jsonl", framing_rows(REC_ORIGINAL, {}, others="completed"))

    def tearDown(self):
        self.temporary.cleanup()

    def main(self, *argv):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = A.main(list(argv))
        return code, stdout.getvalue(), stderr.getvalue()

    def test_the_hand_computed_primary_and_secondaries(self):
        run_json = a7_run_json(self.base / "a7run")
        out = self.base / "out"
        code, stdout, stderr = self.main("--scores-impact", self.impact, "--scores-original", self.original,
                                         "--run-json-impact", run_json, "--out", str(out))
        self.assertEqual(code, 0, stderr)
        result = json.loads((out / "A7_ANALYSIS.json").read_bytes())
        tables = (out / "A7_TABLES.md").read_text(encoding="utf-8")
        self.assertEqual(stdout, tables + "\n")
        self.assertEqual(result["label"], "PRESPECIFIED (protocols/EXPA_A7_IMPACT.md, written before any A7 cell); "
                                          "ANALYSIS RECORD")
        self.assertIn(result["label"], tables)
        self.assertEqual(result["inputs"]["scores_impact"],
                         {"path": self.impact, "sha256": sha(Path(self.impact).read_bytes())})
        self.assertEqual(result["inputs"]["scores_original"],
                         {"path": self.original, "sha256": sha(Path(self.original).read_bytes())})
        self.assertEqual((result["inputs"]["run_json_impact"]["path"], result["inputs"]["run_json_impact"]["sha256"],
                          result["inputs"]["run_json_impact"]["text_sha256"], len(result["inputs"]["run_json_impact"]
                                                                                  ["launches"])),
                         (run_json, sha(Path(run_json).read_bytes()), IMPACT_TEXT_SHA256, 1))
        self.assertEqual(result["script"]["sha256"], sha((PROJECT / "scripts" / "expA_a7_analysis.py").read_bytes()))
        a = result["analysis"]
        pop = a["population"]
        # i7's impact none cell is missing for infrastructure: impact has 6 items, the original 7.
        self.assertEqual((pop["items_by_framing"], pop["families_by_framing"]),
                         ({"impact": 6, "original": 7}, {"impact": 3, "original": 3}))
        self.assertEqual(pop["missing_infrastructure_cells"], {"impact": 1, "original": 0})
        self.assertEqual(pop["rows_excluded"], {"impact": {"outside_the_a7_arms": 21},
                                                "original": {"outside_the_a7_arms": 21}})
        # Impact T1 per item: i1 1, i2 0 | i3 1, i4 1 | i5 -1, i6 0 -> families 1/2, 1, -1/2 -> 1/3;
        # var 7/12, se sqrt(7)/6, t = 2/sqrt(7), df 2.
        t1, t2 = a["primary"]["tests"]["T1"], a["primary"]["tests"]["T2"]
        self.assertEqual((t1["estimate_exact"], t1["n_families"], t1["n_items"], t1["df"], t1["family_means"]),
                         ("1/3", 3, 6, 2, {"f1": 0.5, "f2": 1.0, "f3": -0.5}))
        self.assertAlmostEqual(t1["t"], 2 / math.sqrt(7), places=9)
        self.assertAlmostEqual(t1["p_two_sided"], p_df2(2 / math.sqrt(7)), places=6)
        self.assertAlmostEqual(t1["ci_low"], 1 / 3 - 4.302652729749464 * math.sqrt(7) / 6, places=5)
        # Impact T2 per item: i1 0, i2 1 | i3 0, i4 0 | i5 1, i6 0 -> 1/2, 0, 1/2 -> 1/3; se 1/6, t = 2.
        self.assertEqual((t2["estimate_exact"], t2["family_means"]), ("1/3", {"f1": 0.5, "f2": 0.0, "f3": 0.5}))
        self.assertAlmostEqual(t2["t"], 2.0, places=9)
        self.assertAlmostEqual(t2["p_two_sided"], p_df2(2.0), places=6)
        # Holm: T2 first (2 p), then T1 (max of that and its p).
        self.assertEqual(a["primary"]["holm_order"], ["T2", "T1"])
        self.assertAlmostEqual(t2["p_holm"], 2 * p_df2(2.0), places=6)
        self.assertAlmostEqual(t1["p_holm"], max(2 * p_df2(2.0), p_df2(2 / math.sqrt(7))), places=6)
        self.assertEqual((t1["reject_at_alpha"], t2["reject_at_alpha"], a["primary"]["alpha"]), (False, False, 0.05))
        self.assertEqual(t1["decision"], "do not reject H0 (T1 = 0) at alpha 0.05 (Holm)")
        s = a["secondary"]
        self.assertNotIn("p_two_sided", s["original_framing"]["T1"])
        # Original T1: f1 0, f2 1/2, f3 [0, 0, 1] 1/3 -> 5/18; T2: f1 1/2, f2 0, f3 2/3 -> 7/18.
        self.assertEqual((s["original_framing"]["T1"]["estimate_exact"], s["original_framing"]["T1"]["n_items"]),
                         ("5/18", 7))
        self.assertEqual(s["original_framing"]["T2"]["estimate_exact"], "7/18")
        # Framing differences over i1..i6: T1 1, 0 | 0, 1 | -1, 0 -> 1/6; T2 -1, 1 | 0, 0 | 0, -1 -> -1/6.
        self.assertEqual((s["framing_difference_contrasts"]["T1"]["estimate_exact"],
                          s["framing_difference_contrasts"]["T1"]["n_items"]), ("1/6", 6))
        self.assertEqual(s["framing_difference_contrasts"]["T2"]["estimate_exact"], "-1/6")
        # Arms, paired by item: none (i1..i6) 0 | -1/2 | 1/2 -> 0; incomplete (i1..i7) 1/2 | 0 | 0 -> 1/6;
        # complete (i1..i7) 1/2 | 0 | (0 - 1 - 1)/3 -> -1/18.
        arms = s["framing_difference_arms"]
        self.assertEqual({arm: (v["estimate_exact"], v["n_items"]) for arm, v in arms.items()},
                         {"none": ("0/1", 6), "incomplete": ("1/6", 7), "complete": ("-1/18", 7)})
        impact = s["by_framing"]["impact"]
        # Recovery levels over i1..i6: none 0 | 0 | 1/2 -> 1/6.
        self.assertEqual(impact["none"]["recovery"]["estimate_exact"], "1/6")
        # Submission, incomplete: 1/2 | 1 | 0 -> 1/2; complete: i6's wrong answer is a submission -> 1.
        self.assertEqual(impact["incomplete"]["submission"]["estimate_exact"], "1/2")
        self.assertEqual(impact["complete"]["submission"]["estimate_exact"], "1/1")
        # Claims (submitted only), incomplete: i1 "no" 0 | i3, i4 "yes" 1 | f3 none submitted -> 1/2 over 2 families.
        claims = impact["incomplete"]["completeness_claim_yes"]
        self.assertEqual((claims["estimate_exact"], claims["n_families"], claims["n_items"]), ("1/2", 2, 3))
        self.assertEqual(impact["complete"]["completeness_claim_yes"]["estimate_exact"], "5/6")
        # Responses used, none: (8 + 7)/2 | 8 | (3 + 8)/2 -> 7.
        self.assertEqual(impact["none"]["responses_used"]["estimate_exact"], "7/1")
        # precision_lower, complete: 1 | 1 | (1 + 0)/2 -> 5/6; precision_upper: 1 | 1 | (1 + 1/2)/2 -> 11/12.
        self.assertEqual(impact["complete"]["precision_lower"]["estimate_exact"], "5/6")
        self.assertEqual(impact["complete"]["precision_upper"]["estimate_exact"], "11/12")
        self.assertEqual(s["by_framing"]["original"]["none"]["recovery"]["estimate_exact"], "1/6")
        # Exits of every completed cell of the arm (i7's incomplete and complete cells included).
        self.assertEqual(s["exits"]["impact"], {"none": {"ContextLimit": 1, "NoAnswer": 4, "Submitted": 1},
                                                "incomplete": {"NoAnswer": 3, "Submitted": 4},
                                                "complete": {"NoAnswer": 1, "Submitted": 6}})
        self.assertEqual(pop["cell_states"]["impact"]["none"], {"completed": 6, "missing_infrastructure": 1})
        signs = s["libraries_by_sign"]
        self.assertEqual({c: (v["positive"], v["zero"], v["negative"]) for c, v in signs["impact"].items()},
                         {"T1": (2, 0, 1), "T2": (2, 1, 0)})
        self.assertEqual(signs["impact"]["T1"]["positive_families"], ["f1", "f2"])
        self.assertEqual({c: (v["positive"], v["zero"], v["negative"]) for c, v in signs["original"].items()},
                         {"T1": (2, 1, 0), "T2": (2, 1, 0)})
        self.assertIn("| T1 = rec(incomplete) - rec(none)", tables)
        self.assertIn("| T2 impact - original | -16.7 [", tables)
        self.assertIn("Reading: positive T1 and T2 mean both effects transfer", tables)
        # Refused when the output directory exists.
        code, _, stderr = self.main("--scores-impact", self.impact, "--scores-original", self.original,
                                    "--out", str(out))
        self.assertEqual(code, 2)
        self.assertIn("exists", stderr)

    def test_without_run_json(self):
        out = self.base / "plain"
        code, _, _ = self.main("--scores-impact", self.impact, "--scores-original", self.original, "--out", str(out))
        self.assertEqual(code, 0)
        result = json.loads((out / "A7_ANALYSIS.json").read_bytes())
        self.assertEqual((sorted(result["inputs"]), result["run_json_verified"]),
                         (["scores_impact", "scores_original"], False))

    def test_refusals(self):
        def refused(*argv, message):
            out = self.base / f"refused-{len(list(self.base.iterdir()))}"
            code, _, stderr = self.main(*argv, "--out", str(out))
            self.assertEqual(code, 2, stderr)
            self.assertIn(message, stderr)
            self.assertFalse(out.exists())
        base = ("--scores-impact", self.impact, "--scores-original", self.original)
        # The impact run dispatched cells outside the three arms.
        rows = framing_rows(REC_IMPACT, OVERRIDES_IMPACT, others="completed")
        refused("--scores-impact", write_scores(self.base, "wide.jsonl", rows), "--scores-original", self.original,
                message="cells outside the three A7 arms were dispatched")
        # Different items files; duplicate rows; an unscored completed cell; byte-identical inputs.
        rows = framing_rows(REC_ORIGINAL, {}, others="completed")
        for row in rows:
            row["items_sha256"] = "2" * 64
        refused("--scores-impact", self.impact, "--scores-original", write_scores(self.base, "items.jsonl", rows),
                message="the two runs name different items files")
        rows = framing_rows(REC_IMPACT, OVERRIDES_IMPACT, others="not_dispatched")
        refused("--scores-impact", write_scores(self.base, "dup.jsonl", rows + rows[:1]), "--scores-original",
                self.original, message="duplicate row")
        rows[0]["site_recovered"] = None
        refused("--scores-impact", write_scores(self.base, "none.jsonl", rows), "--scores-original", self.original,
                message="needs site_recovered 0 or 1")
        refused("--scores-impact", self.impact, "--scores-original", self.impact,
                message="byte-identical")
        # run.json: another framing, another text, a notice, another budget, another selection, no launches.
        for name, config, message in (
                ("default", harness.EpisodeConfig().record(), "task_framing is None, not 'impact'"),
                ("text", dict(impact_config().record(), task_framing_rules=dict(
                    impact_config().record()["task_framing_rules"], text_sha256="0" * 64)),
                 "not the protocol's text and sha256"),
                ("notice", dict(impact_config().record(), final_turn_policy="deadline"), "no final-turn notice"),
                ("budget", dict(impact_config().record(), max_model_calls=16), "not the prereg budget")):
            with self.subTest(run_json=name):
                refused(*base, "--run-json-impact", a7_run_json(self.base / f"rj-{name}", config=config),
                        message=message)
        refused(*base, "--run-json-impact", a7_run_json(self.base / "rj-sel", selection={
            "blocks": ["P"], "seeds": [11], "conditions": ["C0", "C1"]}), message="selection")
        lonely = self.base / "rj-lonely"
        lonely.mkdir()
        (lonely / "run.json").write_bytes(json.dumps({"episode_config": impact_config().record()}).encode())
        refused(*base, "--run-json-impact", str(lonely / "run.json"), message="no launch records")

    def test_the_protocol_names_the_script_and_the_rules(self):

        if not (__import__("pathlib").Path(__file__).resolve().parents[1] / "protocols").is_dir():

            self.skipTest("the protocol texts ship in the dataset package (protocol/)")
        protocol = (PROJECT / PROTOCOL).read_text(encoding="utf-8")
        self.assertIn("scripts/expA_a7_analysis.py", protocol)
        self.assertIn("Holm across them, alpha 0.05", protocol)
        self.assertEqual((A.SEED, A.ALPHA, A.IMPACT_TEMPLATE), (11, 0.05, harness.IMPACT_TASK_TEMPLATE))
        self.assertEqual(A.ARMS, {"none": ("P", "C0", "none"), "incomplete": ("P", "C1", "omit_node"),
                                  "complete": ("S-VC", "C1", "vcomplete")})
        self.assertEqual(A.CONTRASTS, {"T1": ("incomplete", "none"), "T2": ("complete", "incomplete")})


if __name__ == "__main__":
    unittest.main()
