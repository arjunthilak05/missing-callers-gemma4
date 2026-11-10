"""SYNTHETIC tests for scripts/expA_a3_devcheck.py (ANALYSIS RECORD).

Every record below is SYNTHETIC; none of it is model output.
(1) Hand-made A3 dev runs in the harness's record formats (run.json, cells.jsonl and cells/<cell_id>/<attempt_id>/
    {attempt_start.json, result.json, events.jsonl}), with the events, receipts and request bytes the harness writes
    under evidence_ledger off and on (shapes as in tests/test_expA_a3_ledger.py). They cover the go rule, every
    gated check and the reported sizes. On a filesystem that cannot hold '|' (Windows) the cell directories are
    written with '%7C' and read through the check's directory-name hook.
(2) The command line and exit codes.
(3) The script's constants, pinned to the harness source (parsed, not imported) and to the protocol.
"""
from __future__ import annotations

import ast
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("expA_a3_devcheck", ROOT / "scripts" / "expA_a3_devcheck.py")
T = importlib.util.module_from_spec(_SPEC)
sys.modules["expA_a3_devcheck"] = T
_SPEC.loader.exec_module(T)

POSIX = os.name == "posix"
ENCODE = (lambda name: name) if POSIX else (lambda name: name.replace("|", "%7C"))
DECODE = None if POSIX else (lambda name: name.replace("%7C", "|"))

RUN_ID = "a3ledger-dev-synthetic"
ITEMS = ("synthetic-item-a", "synthetic-item-b", "synthetic-item-c", "synthetic-item-d", "synthetic-item-e")
SYSTEM = "SYNTHETIC system prompt: find the callers of the target; you have at most 8 responses."
USER = "SYNTHETIC task: find every place where `pkg.core.target` (pkg/core.py, line 3) is invoked."
FORMAT_ERROR = ("Format error: your response did not call a tool. Every response must call at least one of the "
                "offered tools (source_reference_search, bash, submit_answer). Call submit_answer when you are ready "
                "to finish.")
SUBMISSION = {"sites": [{"path": "pkg/core.py", "line": 6}], "complete": "yes"}
DEADLINE = "\n\n[harness notice] This is your last response. Call submit_answer now."


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def cell_id(item, variant, seed=61, block="LEDGER", condition="C1"):
    return f"{block}|{item}|{variant}|{condition}|s{seed}"


DEV = [cell_id(item, variant) for item in ITEMS for variant in ("vcomplete", "omit_node")]


def episode(arm, responses, budget=None):
    """SYNTHETIC events of one attempt in the order the harness writes them, and its result fields.

    ``responses``: one entry per response: "tool" (a bash call printing one call site of target), "plain" (a bash
    call printing nothing the ledger reads), "submit", "text" (a format error), "limit" (ContextLimit before that
    call) or "infra"; missing entries are "tool". At the last call the ledger is built (and, under on, appended)
    right before the deadline notice. ``budget`` (max_model_calls, context_budget): default the prereg (8, 16384);
    (16, 32768) gives an A6 episode (tests/test_expA_a6_budget16.py), whose last call is call_index 15."""
    calls_allowed, context_budget = budget or (8, 16384)
    final_call = calls_allowed - 1
    config = T.expected_episode_config(arm, budget)
    notice = {"policy": "deadline", "text_sha256": sha(DEADLINE[len(T.PREFIX):]), "suffix_sha256": sha(DEADLINE),
              "call_index": final_call, "appended": False, "sent": False, "delivered": False, "appended_to": None,
              "tool_call_id": None, "prompt_tokens": None, "max_tokens": None}
    ledger = T.unbuilt_receipt(arm, final_call)
    rows, messages, entries = [], [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}], []

    def write(kind, value):
        rows.append({"sequence": len(rows), "wall_time": 1.0 + len(rows), "kind": kind, "value": value})
    write("episode_setup", {"config": config, "system": SYSTEM, "user": USER})
    exit_status, infra, consecutive, calls = None, None, 0, 0
    for call_index in range(calls_allowed):
        response = responses[call_index] if call_index < len(responses) else "tool"
        noticed = call_index == final_call
        if noticed:
            last = messages[-1]
            appended_to = "tool_result" if last["role"] == "tool" else "format_error"
            lines = [T.ENTRY.format(**entry) for entry in entries]
            text = T.HEADER + "\n" + ("\n".join(lines) if lines else T.EMPTY)
            body_bytes = len("\n".join(lines).encode("utf-8"))
            content = last["content"] + (T.PREFIX + text if arm == "on" else "")
            ledger.update(computed=True, appended=arm == "on", appended_to=appended_to,
                          tool_call_id=last.get("tool_call_id"), candidates=len(entries), entries=len(entries),
                          body_bytes=body_bytes, text_sha256=sha(text), suffix_sha256=sha(T.PREFIX + text))
            write("evidence_ledger", {"call_index": final_call, "mode": arm, "appended": arm == "on",
                                      "appended_to": appended_to, "tool_call_id": last.get("tool_call_id"),
                                      "name": "target", "text": text, "text_sha256": sha(text),
                                      "suffix": T.PREFIX + text, "suffix_sha256": sha(T.PREFIX + text),
                                      "candidates": len(entries), "entries": copy.deepcopy(entries),
                                      "body_bytes": body_bytes, "previous_content_sha256": sha(last["content"]),
                                      "content_sha256": sha(content)})
            final = content + DEADLINE
            messages[-1] = dict(last, content=final)
            notice.update(appended=True, appended_to=appended_to, tool_call_id=last.get("tool_call_id"))
            write("final_turn_notice", {"call_index": final_call, "policy": "deadline", "appended_to": appended_to,
                                        "tool_call_id": last.get("tool_call_id"), "suffix": DEADLINE,
                                        "suffix_sha256": sha(DEADLINE), "text_sha256": notice["text_sha256"],
                                        "previous_content_sha256": sha(content), "content_sha256": sha(final)})
        count = len(json.dumps(messages))
        write("context_accounting", {"call_index": call_index, "count": count, "max_model_len": 32768})
        if call_index == 0:
            write("exposure_receipt", {"recorded_before_first_model_call": True})
        if noticed:
            notice["prompt_tokens"] = count
        if response == "limit":
            write("context_limit", {"call_index": call_index, "prompt_tokens": context_budget - 511, "remaining": 511})
            exit_status = "ContextLimit"
            break
        max_tokens = 4096 - call_index
        if noticed:
            notice.update(sent=True, max_tokens=max_tokens)
            ledger["sent"] = ledger["appended"]
        if response == "infra":
            write("model_request_failed", {"call_index": call_index, "max_tokens": max_tokens})
            exit_status, infra = "InfraInterrupted", {"kind": "server", "message": "SYNTHETIC failure"}
            break
        call_id = f"call-{call_index}-0"
        if response == "text":
            message = {"role": "assistant", "content": "SYNTHETIC text without a tool call."}
        else:
            name, arguments = (("submit_answer", SUBMISSION) if response == "submit"
                               else ("bash", {"command": "grep -rn target ."}))
            message = {"role": "assistant", "content": None, "tool_calls": [
                {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}]}
        write("model_exchange", {"call_index": call_index, "max_tokens": max_tokens,
                                 "request": {"messages": copy.deepcopy(messages)},
                                 "response": {"choices": [{"message": message, "finish_reason": "stop"}],
                                              "usage": {"prompt_tokens": count, "completion_tokens": 5}}})
        calls += 1
        if noticed:
            notice["delivered"] = True
            ledger["delivered"] = ledger["appended"]
        if response == "text":
            consecutive += 1
            messages += [{"role": "assistant", "content": message["content"]},
                         {"role": "user", "content": FORMAT_ERROR}]
            write("format_error", {"call_index": call_index, "visible": FORMAT_ERROR, "consecutive": consecutive})
            if consecutive >= 3:
                exit_status = "FormatExhausted"
                break
            continue
        consecutive = 0
        messages.append(message)
        write("parsed_actions", {"call_index": call_index, "actions": [{"id": call_id, "name": name,
                                                                        "arguments": arguments}]})
        if response == "submit":
            write("submission_accepted", {"tool_call_id": call_id, "call_index": call_index, "submission": SUBMISSION})
            exit_status = "Submitted"
            break
        if response == "tool":
            output = f"pkg/m{call_index}.py:{call_index + 1}:    y = target({call_index})\n"
            entries.append({"path": f"pkg/m{call_index}.py", "line": call_index + 1,
                            "excerpt": f"y = target({call_index})",
                            "source": {"message_index": len(messages), "tool_call_id": call_id, "kind": "bash_line"}})
        else:
            output = "ok\n"
        content = canonical({"exit_code": 0, "output": output, "truncated": False}).decode("utf-8")
        write("model_visible_tool_result", {"tool_call_id": call_id, "name": name, "content": content,
                                            "sha256": sha(content)})
        messages.append({"role": "tool", "tool_call_id": call_id, "content": content})
    else:
        exit_status = "NoAnswer"
    write("episode_finished", {"exit_status": exit_status, "infra": infra,
                               "accounting": {"model_calls": calls, "final_turn_notice": dict(notice),
                                              "evidence_ledger": dict(ledger)}})
    return rows, {"exit_status": exit_status, "infra": infra, "calls": calls, "config": config, "notice": notice,
                  "ledger": ledger}


def attempt_dir(run_dir, cell, ordinal=1):
    return run_dir / "cells" / ENCODE(cell) / ENCODE(f"{cell}.a{ordinal:03d}")


def write_attempt(run_dir, cell, ordinal, arm, responses, budget=None):
    attempt_id = f"{cell}.a{ordinal:03d}"
    directory = attempt_dir(run_dir, cell, ordinal)
    directory.mkdir(parents=True)
    start = canonical({"cell_id": cell, "attempt_id": attempt_id, "ordinal": ordinal, "launch": "SYNTHETIC"})
    (directory / "attempt_start.json").write_bytes(start)
    rows, outcome = episode(arm, responses, budget)
    events = b"".join(json.dumps(row).encode("utf-8") + b"\n" for row in rows)
    (directory / "events.jsonl").write_bytes(events)
    block, item, variant, condition, seed = cell.split("|")
    notice, ledger = outcome["notice"], outcome["ledger"]
    result = {"schema": "expA-episode-result-v2", "run_id": RUN_ID, "cell_id": cell, "attempt_id": attempt_id,
              "attempt": {"attempt_id": attempt_id, "ordinal": ordinal}, "block": block, "item_id": item,
              "variant": variant, "condition": condition, "seed": int(seed[1:]), "family": "synthetic",
              "exit_status": outcome["exit_status"], "calls": outcome["calls"],
              "submitted": SUBMISSION if outcome["exit_status"] == "Submitted" else None,
              "exposure": {"recorded_before_first_model_call": True, "final_turn_notice": dict(notice),
                           "evidence_ledger": dict(ledger)},
              "accounting": {"model_calls": outcome["calls"], "final_turn_notice": dict(notice),
                             "evidence_ledger": dict(ledger)},
              "events": {"path": "events.jsonl", "sha256": hashlib.sha256(events).hexdigest(), "records": len(rows)},
              "infra": outcome["infra"], "config": outcome["config"],
              "run_context": {"launch": "SYNTHETIC", "start_sha256": hashlib.sha256(start).hexdigest()}}
    (directory / "result.json").write_bytes(canonical(result))
    return directory


TOOLS8 = ["tool"] * 8
# SYNTHETIC: the plan of each LEDGER seed-61 cell (one response list per attempt).
STANDARD = {
    DEV[0]: [["tool", "submit"]],                                    # submits at response 2: never builds it
    DEV[1]: [["tool"] * 6 + ["text", "tool"]],                       # a format error at 7: notice on its message
    DEV[2]: [["tool"] * 7 + ["limit"]],                              # ContextLimit at the 8th call: never sent
    DEV[3]: [["tool"] * 7 + ["infra"], ["tool"] * 7 + ["submit"]],   # an infrastructure attempt, then terminal
    DEV[4]: [["tool"] * 7 + ["submit"]],                             # submits in the noticed response
    DEV[5]: [["plain"] * 8],                                         # the empty ledger
    **{cell: [TOOLS8] for cell in DEV[6:]},                          # NoAnswer, 7 entries
}


def write_run(directory, arm, plan, *, episode_config=None, extra_cells=(), budget=None):
    """A SYNTHETIC dev run: run.json, cells.jsonl (LEDGER at seeds 61, 73 and 89, then a default-block cell, as an
    A3 design freezes it) and, for each cell of ``plan``, its attempts in order (``budget`` as in episode())."""
    directory.mkdir(parents=True)
    design = [cell_id(item, variant, seed) for seed in (61, 73, 89) for item in ITEMS
              for variant in ("vcomplete", "omit_node")]
    design += [f"P|{item}|omit_node|C1|s11" for item in ITEMS] + list(extra_cells)
    cells = b"".join(canonical({"cell_id": cell}) + b"\n" for cell in design)
    (directory / "cells.jsonl").write_bytes(cells)
    run = {"schema": "expA-run-SYNTHETIC", "run_id": RUN_ID, "cells_sha256": hashlib.sha256(cells).hexdigest(),
           "cell_count": len(design), "model": {"profile": "gemma4-31b", "revision": "SYNTHETIC"},
           "episode_config": T.expected_episode_config(arm, budget) if episode_config is None else episode_config}
    (directory / "run.json").write_bytes(canonical(run) + b"\n")
    for cell, attempts in plan.items():
        for ordinal, responses in enumerate(attempts, 1):
            write_attempt(directory, cell, ordinal, arm, responses, budget)
    return directory


def edit_result(directory, change):
    path = directory / "result.json"
    result = json.loads(path.read_bytes())
    change(result)
    path.write_bytes(canonical(result))


def edit_events(directory, change):
    """Rewrite events.jsonl with ``change(rows)`` and rebind it in result.json, so only the check under test sees it."""
    path = directory / "events.jsonl"
    rows = [json.loads(line) for line in path.read_bytes().splitlines()]
    rows = change(rows) or rows
    for index, row in enumerate(rows):
        row["sequence"] = index
    raw = b"".join(json.dumps(row).encode("utf-8") + b"\n" for row in rows)
    path.write_bytes(raw)
    edit_result(directory, lambda result: result.update(events=dict(
        result["events"], sha256=hashlib.sha256(raw).hexdigest(), records=len(rows))))


def set_ledger_receipt(directory, **values):
    """Change the ledger receipt everywhere it is recorded (exposure, accounting, episode_finished)."""
    def change_result(result):
        for place in ("exposure", "accounting"):
            result[place]["evidence_ledger"].update(values)
    edit_result(directory, change_result)

    def change_events(rows):
        for row in rows:
            if row["kind"] == "episode_finished":
                row["value"]["accounting"]["evidence_ledger"].update(values)
    edit_events(directory, change_events)


def last_request(rows, final_call=7):
    return [row for row in rows if row["kind"] == "model_exchange" and row["value"]["call_index"] == final_call][0]


class SyntheticRunTests(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.count = 0

    def tearDown(self):
        self.temporary.cleanup()

    def run_dir(self, arm, plan=None, **kwargs):
        self.count += 1
        return write_run(self.base / f"run-{self.count}", arm, STANDARD if plan is None else plan, **kwargs)

    def check(self, run_dir, arm):
        return T.check_run(run_dir, arm, cell_name=DECODE)

    def assertFailed(self, report, *names):
        self.assertFalse(report["go"])
        self.assertEqual(report["failed_checks"], list(names))

    def test_a_dev_run_that_passes_in_each_arm(self):
        for arm in T.ARMS:
            with self.subTest(arm=arm):
                report = self.check(self.run_dir(arm), arm)
                self.assertEqual((report["go"], report["failed_checks"]), (True, []))
                self.assertEqual({name: entry["passed"] for name, entry in report["checks"].items()},
                                 dict.fromkeys(T.CHECKS, True))
                self.assertGreater(report["checks"]["request_bytes"]["checked"], 0)
                rule = report["go_rule"]
                self.assertEqual((rule["noticed_final_responses"], rule["valid_tool_calls"], rule["share"]),
                                 (8, 8, 1.0))
                ledger, last = report["ledger"], report["last_call"]
                on = arm == "on"
                # built: every cell but DEV[0]; sent/delivered: also not DEV[2] (ContextLimit at the last call)
                self.assertEqual((ledger["built"], ledger["appended"], ledger["sent"], ledger["delivered"]),
                                 (9, 9 * on, 8 * on, 8 * on))
                self.assertEqual((ledger["empty"], ledger["entries_max"], ledger["entries_by_source"]),
                                 (1, 7, {"bash_line": 6 + 7 * 7}))
                self.assertEqual((last["reached"], last["context_limit_at_last_call"], last["context_limit_cells"],
                                  last["notice_on_format_error"], last["format_error_cells"]),
                                 (9, 1, [DEV[2]], 1, [DEV[1]]))
                self.assertEqual(report["cells"]["states"]["completed"], 10)
                self.assertEqual(report["cells"]["infrastructure_attempts_by_kind"], {"server": 1})
                self.assertTrue(T.summary(report).startswith(f"GO: A3 {arm} run {RUN_ID}: 8/8"))

    def test_the_wrong_arm_fails_the_config_and_the_receipts(self):
        report = self.check(self.run_dir("off"), "on")
        for name in ("run_json_episode_config", "config_receipts", "ledger_receipts", "ledger_event",
                     "request_bytes"):
            self.assertIn(name, report["failed_checks"])

    def test_run_json_episode_config(self):
        for change in (lambda c: c.pop("evidence_ledger_rules"), lambda c: c.update(final_turn_policy="neutral"),
                       lambda c: c["evidence_ledger_rules"].update(max_entries=21)):
            config = T.expected_episode_config("on")
            change(config)
            with self.subTest(config=sorted(config)):
                self.assertFailed(self.check(self.run_dir("on", episode_config=config), "on"),
                                  "run_json_episode_config")

    def test_selection_and_completeness(self):
        stray = cell_id("synthetic-item-a", "vcomplete", seed=73)
        report = self.check(self.run_dir("on", plan=dict(STANDARD, **{stray: [TOOLS8]})), "on")
        self.assertFailed(report, "selection")
        plan = dict(STANDARD)
        plan.pop(DEV[9])
        self.assertFailed(self.check(self.run_dir("on", plan=plan), "on"), "run_complete")

    def test_the_go_rule_boundary(self):
        # 10 noticed responses: 9 valid passes (90%), 8 valid fails.
        cells = [cell_id(item, variant) for item in ITEMS for variant in ("vcomplete", "omit_node")]
        for errors, go in ((1, True), (2, False)):
            plan = {cell: [["tool"] * 7 + (["text"] if index < errors else ["tool"])]
                    for index, cell in enumerate(cells)}
            with self.subTest(errors=errors):
                report = self.check(self.run_dir("on", plan=plan), "on")
                self.assertEqual((report["go"], report["failed_checks"], report["go_rule"]["valid_tool_calls"]),
                                 (go, [], 10 - errors))

    def test_a_missing_or_misplaced_ledger_event(self):
        for arm in T.ARMS:
            run_dir = self.run_dir(arm)
            directory = attempt_dir(run_dir, DEV[7])
            edit_events(directory, lambda rows: [row for row in rows if row["kind"] != "evidence_ledger"])
            with self.subTest(arm=arm, case="missing"):     # the request then has no ledger event to match either
                self.assertFailed(self.check(run_dir, arm), "ledger_event", "request_bytes")
            run_dir = self.run_dir(arm)
            directory = attempt_dir(run_dir, DEV[7])

            def move(rows):
                index = [row["kind"] for row in rows].index("evidence_ledger")
                event = rows.pop(index)
                rows.insert(index - 1, event)
            edit_events(directory, move)
            with self.subTest(arm=arm, case="misplaced"):
                self.assertFailed(self.check(run_dir, arm), "ledger_event")

    def test_an_event_that_disagrees(self):
        cases = {"text": lambda e: e.update(text=e["text"] + " "),
                 "entries": lambda e: e["entries"].pop(),
                 "body_bytes": lambda e: e.update(body_bytes=e["body_bytes"] + 1),
                 "appended": lambda e: e.update(appended=not e["appended"]),
                 "source": lambda e: e["entries"][0]["source"].update(kind="user_message")}
        for name, change in cases.items():
            run_dir = self.run_dir("off")

            def apply(rows, change=change):
                for row in rows:
                    if row["kind"] == "evidence_ledger":
                        change(row["value"])
            edit_events(attempt_dir(run_dir, DEV[6]), apply)
            with self.subTest(case=name):
                self.assertFailed(self.check(run_dir, "off"), "ledger_event")

    def test_receipts(self):
        for arm, values in (("on", {"delivered": False}), ("on", {"entries": 3}), ("off", {"appended": True}),
                            ("off", {"computed": False}), ("on", {"text_sha256": "0" * 64})):
            run_dir = self.run_dir(arm)
            set_ledger_receipt(attempt_dir(run_dir, DEV[6]), **values)
            with self.subTest(arm=arm, values=values):
                self.assertFailed(self.check(run_dir, arm), "ledger_event")
        run_dir = self.run_dir("on")
        edit_result(attempt_dir(run_dir, DEV[6]), lambda r: r["accounting"]["evidence_ledger"].update(entries=0))
        self.assertFailed(self.check(run_dir, "on"), "ledger_receipts")
        run_dir = self.run_dir("on")        # an episode that never reached its last call but claims a ledger
        set_ledger_receipt(attempt_dir(run_dir, DEV[0]), computed=True)
        self.assertFailed(self.check(run_dir, "on"), "ledger_event")

    def test_request_bytes(self):
        def tamper(change):
            def apply(rows):
                change(last_request(rows)["value"]["request"]["messages"])
            return apply
        # on: the ledger missing from the request, or the header twice
        run_dir = self.run_dir("on")
        edit_events(attempt_dir(run_dir, DEV[6]), tamper(
            lambda m: m[-1].update(content=m[-1]["content"].replace(T.HEADER, "Candidates:"))))
        self.assertIn("request_bytes", self.check(run_dir, "on")["failed_checks"])
        run_dir = self.run_dir("on")
        edit_events(attempt_dir(run_dir, DEV[6]), tamper(
            lambda m: m[1].update(content=m[1]["content"] + T.HEADER)))
        self.assertFailed(self.check(run_dir, "on"), "request_bytes")
        # off: the header in any request
        run_dir = self.run_dir("off")

        def leak(rows):
            first = [row for row in rows if row["kind"] == "model_exchange"][2]
            first["value"]["request"]["messages"][1]["content"] += T.HEADER
        edit_events(attempt_dir(run_dir, DEV[6]), leak)
        self.assertFailed(self.check(run_dir, "off"), "request_bytes")

    def test_an_off_run_that_sent_the_ledger_fails(self):
        # The records of an on episode inside an off run (config and receipts relabelled off).
        run_dir = self.run_dir("off")
        directory = attempt_dir(run_dir, DEV[6])
        on_rows, _ = episode("on", TOOLS8)

        def swap(rows):
            for row, on_row in zip(rows, on_rows):
                if row["kind"] == "model_exchange" and row["value"]["call_index"] == 7:
                    row["value"]["request"] = on_row["value"]["request"]
        edit_events(directory, swap)
        self.assertIn("request_bytes", self.check(run_dir, "off")["failed_checks"])


class CommandLineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.run_dir = write_run(self.base / "run", "on", STANDARD)
        original = T.check_run
        self.patch = mock.patch.object(T, "check_run", lambda run_dir, arm: original(run_dir, arm, cell_name=DECODE))
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.temporary.cleanup()

    def main(self, *argv):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = T.main(list(argv))
        return code, stdout.getvalue(), stderr.getvalue()

    def test_exit_codes(self):
        out = self.base / "go.json"
        code, stdout, _ = self.main("--run-dir", str(self.run_dir), "--arm", "on", "--out", str(out))
        self.assertEqual(code, 0)
        self.assertTrue(stdout.startswith("GO: A3 on run"))
        report = json.loads(out.read_text(encoding="ascii"))
        self.assertEqual((report["go"], report["arm"], report["schema"]), (True, "on", T.SCHEMA))
        code, stdout, _ = self.main("--run-dir", str(self.run_dir), "--arm", "off", "--out", str(self.base / "x.json"))
        self.assertEqual(code, 1)
        self.assertTrue(stdout.startswith("NO-GO: A3 off run"))
        code, _, stderr = self.main("--run-dir", str(self.run_dir), "--arm", "on", "--out", str(out))
        self.assertEqual(code, 2)
        self.assertIn("refusing to overwrite", stderr)
        code, _, stderr = self.main("--run-dir", str(self.base / "nope"), "--arm", "on", "--out",
                                    str(self.base / "y.json"))
        self.assertEqual(code, 2)
        self.assertFalse((self.base / "y.json").exists())


def _harness_constants():
    """Module-level LEDGER_* string constants and the evidence_ledger_record dict literal of harness.py, parsed."""
    tree = ast.parse((ROOT / "absent_edges" / "probe" / "harness.py").read_text(encoding="utf-8"))
    constants, record = {}, {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            value = node.value
            if isinstance(value, ast.Call) and value.args:       # re.compile("...")
                value = value.args[0]
            with contextlib.suppress(ValueError):
                constants[node.targets[0].id] = ast.literal_eval(value)
        if isinstance(node, ast.FunctionDef) and node.name == "evidence_ledger_record":
            returned = [n for n in ast.walk(node) if isinstance(n, ast.Return)][0].value
            for key, value in zip(returned.keys, returned.values):
                with contextlib.suppress(ValueError):
                    record[ast.literal_eval(key)] = ast.literal_eval(value)
    return constants, record


class ConstantsTests(unittest.TestCase):
    def test_the_constants_are_the_harness(self):
        constants, record = _harness_constants()
        self.assertEqual((T.HEADER, T.EMPTY, T.ENTRY, T.MAX_ENTRIES, T.MAX_BYTES, T.EXCERPT_CHARS, T.PRINTED_LINE,
                          T.CALL_TOKEN, T.DEFINITION, T.PROTOCOL, T.PREFIX),
                         tuple(constants[name] for name in (
                             "LEDGER_HEADER", "LEDGER_EMPTY", "LEDGER_ENTRY", "LEDGER_MAX_ENTRIES", "LEDGER_MAX_BYTES",
                             "LEDGER_EXCERPT_CHARS", "LEDGER_PRINTED_LINE", "LEDGER_CALL_TOKEN", "LEDGER_DEFINITION",
                             "LEDGER_PROTOCOL", "FINAL_TURN_PREFIX")))
        self.assertEqual((T.BLOCK, T.SEED), (constants["LEDGER_BLOCK"], 61))
        self.assertIn(T.SEED, constants["LEDGER_SEEDS"])
        for key, value in T.RULE_TEXTS.items():
            self.assertEqual(record[key], value)
        rules = T.expected_rules("on")
        self.assertEqual({key for key in rules}, set(record) | {"mode", "delivered_to_model", "prefix", "header",
                                                                "empty", "entry", "max_entries", "max_bytes",
                                                                "excerpt_chars", "printed_line", "call_token",
                                                                "definition", "call_index", "protocol"})
        self.assertEqual(T.DEADLINE_SUFFIX, DEADLINE)

    def test_the_protocol(self):

        if not (__import__("pathlib").Path(__file__).resolve().parents[1] / "protocols").is_dir():

            self.skipTest("the protocol texts ship in the dataset package (protocol/)")
        protocol = (ROOT / T.PROTOCOL).read_text(encoding="utf-8")
        self.assertIn(T.HEADER, protocol)
        self.assertIn("block LEDGER, seed 61", protocol)
        self.assertIn("at least 90% of the final responses that received the notice are valid tool calls", protocol)

    @unittest.skipUnless(POSIX, "Linux only: the harness imports serve.py (pwd)")
    def test_the_records_are_the_harness(self):
        import test_expA_default_golden  # noqa: F401  (puts the project on sys.path)
        from absent_edges.probe import harness
        for arm in T.ARMS:
            self.assertEqual(T.expected_episode_config(arm),
                             harness.EpisodeConfig(final_turn_policy="deadline", evidence_ledger=arm).record())


if __name__ == "__main__":
    unittest.main()
