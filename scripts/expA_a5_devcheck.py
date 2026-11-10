#!/usr/bin/env python3
"""Exp A follow-up A5: the dev-family check of a submit-review run (protocols/EXPA_A5_REVIEW.md).

    python -B scripts/expA_a5_devcheck.py --run-dir RUN_DIR --arm neutral|ledger [--seed 61] --out REPORT.json

Run once per dev run (block LEDGER at one seed, 61 for the pilot), over the published run. Standard library only: it
reads run.json, cells.jsonl and cells/<cell_id>/<attempt_id>/ (attempt_start.json, result.json, events.jsonl) and
writes only the report. The run layout, the terminal-attempt rule (D11.3), the record bindings and the A1 receipt
checks are those of scripts/expA_finalturn_devcheck.py; the ledger format is that of scripts/expA_a3_devcheck.py.

Go, as the CHECKS below spell them out for the records:
- every terminal episode's review receipt (result.json exposure.submit_review = accounting.submit_review = the
  episode_finished event's) is consistent with its events: at most one submit_review event, only before the last
  call (call_index < 7) and only for the episode's first well-formed submission; reviewed iff that event exists;
- the delivered review is the tool result of the reviewed call: "[harness notice] " + TEXT, with TEXT the neutral
  sentence, or (ledger arm) the recorded ledger in the frozen format (header + entries, or "(none)") + "\\n" + the
  review instruction; it is in every later request, never in an earlier one, and the ledger header never appears in
  a neutral run;
- answer_source is recorded and agrees with the events: "resubmit" iff a submission_accepted event follows the review
  (and result.json submitted is that submission); "first_submit" after a review without one (a
  submit_review_resolved event, exit Submitted, submitted = the first submission), or for a first submission at the
  last call (no review, review_skipped_last_call); none when nothing was submitted;
- at least 90% of the reviewed episodes whose next response was requested (a model_exchange right after the reviewed
  call) answered it with a well-formed tool call (parsed_actions, not format_error);
- run.json and every result record the arm's exact episode_config (the deadline notice, submit_review = the arm and
  its rules, no evidence ledger), and every dispatched cell is a LEDGER|<item>|vcomplete or omit_node|C1|s<seed> cell
  (and every such cell of cells.jsonl is terminal).
Reported, not gated: the share reviewed, answer sources, unresolved exits, resubmissions in the reviewed response,
ledger sizes (ledger arm).

Exit status: 0 go; 1 no-go (the report lists each failed check with examples); 2 usage error or unreadable run
records (no report is written).
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
from pathlib import Path
import sys

_HERE = Path(__file__).resolve().parent


def _load(name):
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, _HERE / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module


A1 = _load("expA_finalturn_devcheck")
A3 = _load("expA_a3_devcheck")
InputError, sha256_text, sha256_bytes, sha256_file = A1.InputError, A1.sha256_text, A1.sha256_bytes, A1.sha256_file
_events, _at = A1._events, A1._at

SCHEMA = "expA-a5-devcheck-v1"
PROTOCOL = "protocols/EXPA_A5_REVIEW.md"
POLICY = "deadline"                                       # both arms run with --final-turn-policy deadline
ARMS = ("neutral", "ledger")
FINAL_CALL = A1.FINAL_CALL                                # 7
DEADLINE_SUFFIX = A1.PREFIX + A1.TEXTS[POLICY]
# The frozen review (absent_edges/probe/harness.py SUBMIT_REVIEW_*), declared again rather than imported so that the
# check does not share code with what it audits; tests/test_expA_a5_review.py pins these values to the harness.
PREFIX = "[harness notice] "
NEUTRAL = "Your answer was received. Before finishing, call submit_answer again with your final list."
INSTRUCTION = "Before finishing, review these candidate call sites, then call submit_answer again with your final list."
HEADER = A3.HEADER
RULE_TEXTS = {
    "when": "the first well-formed submit_answer (accepted by the submission validation) at a call_index before the "
            "last call: it is recorded, not accepted, and its tool result is the prefix + TEXT",
    "resubmission": "the next well-formed submit_answer is accepted as final; there is no second review",
    "no_resubmission": "an episode that ends without one (NoAnswer, ContextLimit, FormatExhausted) is Submitted with "
                       "the first submission; InfraInterrupted stays infrastructure",
    "last_call": "a first submit_answer at the last call is accepted at once (no review)",
}
LEDGER_SOURCES = ("the model-visible tool results already in the history when the first submission is made "
                  "(protocols/EXPA_A3_LEDGER.md sources and selection, unchanged)")
RECEIPT_FIELDS = ("mode", "before_call_index", "reviewed", "review_call_index", "review_tool_call_id", "text_sha256",
                  "content_sha256", "ledger_candidates", "ledger_entries", "ledger_body_bytes",
                  "review_skipped_last_call", "first_call_index", "first_submission", "first_submission_validation",
                  "answer_source", "resubmit_call_index", "resubmit_tool_call_id", "unresolved_exit")
UNRESOLVED_EXITS = ("NoAnswer", "ContextLimit", "FormatExhausted")
BLOCK, SEED, CONDITION, VARIANTS = "LEDGER", 61, "C1", ("vcomplete", "omit_node")
GO_RULE = A1.GO_RULE                                      # valid / received >= 9/10, compared exactly

CHECKS = {
    "run_json_episode_config": "run.json episode_config is the arm's exact record (the six prereg fields, the "
                               "deadline notice, submit_review = the arm and submit_review_rules; no evidence ledger)",
    "run_json_cells": "cells.jsonl is the file run.json binds (cells_sha256)",
    "selection": "every dispatched cell is a LEDGER|<item>|vcomplete or omit_node|C1|s<seed> cell of cells.jsonl",
    "run_complete": "every LEDGER cell at the seed in cells.jsonl is terminal in the D11.3 sense (a completed attempt, "
                    "or missing after 3 infrastructure attempts)",
    "records": "each terminal attempt: its directory names its cell and attempt, result.json binds attempt_start.json "
               "and events.jsonl (sha256 and rows), its run_id is run.json's, and no attempt follows it",
    "config_receipts": "result.json config and the episode_setup event's config are the arm's record",
    "notice_receipts": "the A1 final_turn_notice receipt is consistent (as the A1 dev check reads it)",
    "review_receipts": "result.json exposure.submit_review equals accounting.submit_review and the episode_finished "
                       "event's, has exactly the receipt fields, mode = the arm and before_call_index 7",
    "review_event": "at most one submit_review event, before the last call, for the first well-formed submission; "
                    "reviewed iff it exists; the event, the receipt and the frozen text agree (text and content "
                    "sha256, first submission, ledger entries/candidates/bytes and caps); the reviewed call's "
                    "model-visible tool result is the event's content",
    "answer_source": "answer_source agrees with the events and result.json: resubmit iff a submission_accepted event "
                     "follows the review (submitted = it); first_submit after a review with a submit_review_resolved "
                     "event (exit Submitted, submitted = the first submission) or for a first submission at the last "
                     "call (review_skipped_last_call); none without a submission",
    "request_bytes": "the review content is the reviewed call's tool result in every later request (with the deadline "
                     "suffix when it is the message the notice follows) and in no earlier request; the ledger header "
                     "never occurs in a neutral run, and in a ledger run only from the request after the review",
    "next_response_recorded": "each requested response after a review has exactly one parsed_actions or format_error "
                              "event",
}


def expected_rules(arm):
    """submit_review_record(arm, 8) of the release, rebuilt from the protocol's values."""
    value = {"protocol": PROTOCOL, "mode": arm, "channel": "tool_result", "prefix": PREFIX,
             "before_call_index": FINAL_CALL, **RULE_TEXTS}
    if arm == "neutral":
        value.update(text=NEUTRAL, text_sha256=sha256_text(NEUTRAL))
    else:
        value.update(instruction=INSTRUCTION, instruction_sha256=sha256_text(INSTRUCTION),
                     text="the evidence ledger text + '\\n' + instruction", header=A3.HEADER, empty=A3.EMPTY,
                     entry=A3.ENTRY, max_entries=A3.MAX_ENTRIES, max_bytes=A3.MAX_BYTES,
                     excerpt_chars=A3.EXCERPT_CHARS, printed_line=A3.PRINTED_LINE, call_token=A3.CALL_TOKEN,
                     definition=A3.DEFINITION, sources=LEDGER_SOURCES)
    return value


def expected_episode_config(arm):
    """EpisodeConfig(final_turn_policy="deadline", submit_review=arm).record() of the release."""
    if arm not in ARMS:
        raise InputError(f"--arm must be one of {', '.join(ARMS)}")
    return dict(A1.expected_episode_config(POLICY), submit_review=arm, submit_review_rules=expected_rules(arm))


def unreviewed_receipt(arm):
    return {"mode": arm, "before_call_index": FINAL_CALL, "reviewed": False, "review_call_index": None,
            "review_tool_call_id": None, "text_sha256": None, "content_sha256": None, "ledger_candidates": None,
            "ledger_entries": None, "ledger_body_bytes": None}


class Checks(A1.Checks):
    def __init__(self, descriptions=None):
        super().__init__(descriptions or CHECKS)


class DevCheck(A1.DevCheck):
    """The A5 check; reuses the A1 check's records() and receipts() (records, config_receipts, notice_receipts)."""

    def __init__(self, run_dir, arm, *, seed=SEED, cell_name=None):
        if arm not in ARMS:
            raise InputError(f"--arm must be one of {', '.join(ARMS)}")
        if type(seed) is not int or seed < 0:
            raise InputError("--seed must be a non-negative integer")
        super().__init__(run_dir, POLICY, None)
        self.arm, self.ledger_seed = arm, seed
        self.expected = expected_episode_config(arm)
        self.checks = Checks({name: text.replace("s<seed>", f"s{seed}").replace("at the seed", f"at seed {seed}")
                              for name, text in CHECKS.items()})
        self.cell_name = cell_name or (lambda name: name)
        self.review = collections.Counter()
        self.sources = collections.Counter()
        self.unresolved = collections.Counter()
        self.ledger_sizes = {"entries": [], "body_bytes": [], "candidates": [], "empty": 0}
        self.next_response = {"received": 0, "valid": 0, "format_errors": 0, "not_requested": 0}

    def is_registered(self, cell_id):
        parts = A3.cell_parts(cell_id)
        return bool(parts) and parts[0] == BLOCK and parts[2] in VARIANTS and parts[3] == CONDITION \
            and parts[4] == f"s{self.ledger_seed}"

    # -- the run ---------------------------------------------------------------------------------------------------

    def run(self, tokenizer=None):
        if not self.run_dir.is_dir():
            raise InputError(f"run directory not found: {self.run_dir}")
        run, run_raw = A1.read_json(self.run_dir / "run.json", "run.json")
        if not isinstance(run, dict):
            raise InputError("run.json is not an object")
        try:
            cells_raw = (self.run_dir / "cells.jsonl").read_bytes()
        except OSError as error:
            raise InputError(f"cells.jsonl: {error}") from error
        design = [A1.loads(line, f"cells.jsonl line {number}")
                  for number, line in enumerate(cells_raw.splitlines(), 1) if line.strip()]
        design_ids = {cell.get("cell_id") for cell in design if isinstance(cell, dict)}
        registered = {cell for cell in design_ids if self.is_registered(cell)}
        self.run_json = run
        config = run.get("episode_config")
        self.checks.check("run_json_episode_config", [] if config == self.expected else [
            f"episode_config {json.dumps(config, sort_keys=True)[:600]}"], "run.json")
        self.checks.check("run_json_cells", [] if sha256_bytes(cells_raw) == run.get("cells_sha256") else [
            "sha256(cells.jsonl) is not run.json cells_sha256"], "cells.jsonl")

        found, empty = A1.load_attempts(self.run_dir)
        attempts = {}
        for directory_name, cell_attempts in found.items():
            for attempt in cell_attempts:
                attempt["attempt_id"] = self.cell_name(attempt["attempt_id"])
            attempts[self.cell_name(directory_name)] = cell_attempts
        states, infra_kinds, resolved = collections.Counter(), collections.Counter(), {}
        for cell, cell_attempts in attempts.items():
            self.checks.check("selection", [] if cell in registered else [
                f"cell {cell} is not a {BLOCK}|<item>|vcomplete or omit_node|{CONDITION}|s{self.ledger_seed} cell of "
                "cells.jsonl"], f"cells/{cell}")
            resolved[cell] = A1.resolve(cell_attempts)
        for cell in sorted(registered | set(resolved)):
            state, terminal, infra, later = resolved.get(cell, ("not_dispatched", None, [], []))
            states[state] += 1
            for attempt in infra:
                result = attempt["result"]
                record = result.get("infra") if isinstance(result, dict) else None
                kind = record.get("kind") if isinstance(record, dict) else None
                infra_kinds[str(kind) if result is not None else "start_record_only"] += 1
            if cell in registered:
                self.checks.check("run_complete", [] if state in A1.TERMINAL_STATES else [f"cell state {state}"],
                                  f"cells/{cell}")
            if terminal is not None:
                self.attempt(cell, terminal, later)

        nxt = self.next_response
        share = nxt["valid"] / nxt["received"] if nxt["received"] else None
        rule_passed = nxt["received"] > 0 and GO_RULE[1] * nxt["valid"] >= GO_RULE[0] * nxt["received"]
        failed = self.checks.failed()
        terminal = sum(self.exits.values())
        sizes = self.ledger_sizes

        def mean(values):
            return sum(values) / len(values) if values else None
        report = {
            "schema": SCHEMA, "protocol": PROTOCOL, "arm": self.arm, "final_turn_policy": POLICY,
            "seed": self.ledger_seed, "go": rule_passed and not failed, "failed_checks": failed,
            "go_rule": {"rule": "at least 90% of the reviewed episodes whose next response was requested answered it "
                                "with a well-formed tool call (parsed_actions, not format_error)",
                        "received_review": nxt["received"], "valid_tool_calls": nxt["valid"],
                        "format_errors": nxt["format_errors"], "share": share, "threshold": GO_RULE[0] / GO_RULE[1],
                        "passed": rule_passed},
            "checks": self.checks.report(),
            "review": {"terminal_attempts": terminal, "reviewed": self.review["reviewed"],
                       "reviewed_share": self.review["reviewed"] / terminal if terminal else None,
                       "review_skipped_last_call": self.review["skipped_last_call"],
                       "never_submitted": self.review["never_submitted"],
                       "answer_source": dict(sorted(self.sources.items())),
                       "unresolved_exits": dict(sorted(self.unresolved.items())),
                       "resubmitted_in_the_reviewed_response": self.review["same_response"],
                       "reviewed_next_response_not_requested": nxt["not_requested"],
                       "review_call_index_histogram": dict(sorted(
                           (str(k[len("call:"):]), v) for k, v in self.review.items() if k.startswith("call:"))),
                       "not_gated": "reported, not gated"},
            "cells": {f"ledger_seed_{self.ledger_seed}_in_design": len(registered), "dispatched": len(attempts),
                      "states": {state: states[state] for state in ("completed", "missing_infrastructure",
                                                                    "infra_pending", "not_dispatched")},
                      "terminal_exits": dict(sorted(self.exits.items())),
                      "infrastructure_attempts_by_kind": dict(sorted(infra_kinds.items())),
                      "empty_attempt_directories": len(empty),
                      "terminal_only": "each cell's terminal attempt (D11.3: the first with a completed model exit); "
                                       "infrastructure attempts are not checked or counted"},
            "run": {"run_dir": str(self.run_dir), "run_id": run.get("run_id"), "run_json_sha256": sha256_bytes(run_raw),
                    "cells_jsonl_sha256": sha256_bytes(cells_raw), "model": run.get("model")},
            "script": {"path": "scripts/expA_a5_devcheck.py", "sha256": sha256_file(__file__)},
        }
        if self.arm == "ledger":
            built = len(sizes["entries"])
            report["ledger"] = {"built": built, "entries_mean": mean(sizes["entries"]),
                                "entries_max": max(sizes["entries"], default=None),
                                "empty": sizes["empty"], "empty_share": sizes["empty"] / built if built else None,
                                "candidates_mean": mean(sizes["candidates"]),
                                "body_bytes_mean": mean(sizes["body_bytes"]),
                                "body_bytes_max": max(sizes["body_bytes"], default=None),
                                "not_gated": "reported, not gated"}
        return report

    # -- one terminal attempt ----------------------------------------------------------------------------------------

    def attempt(self, cell, attempt, later):
        where, result = attempt["where"], attempt["result"]
        self.exits[result.get("exit_status")] += 1
        rows = self.records(cell, attempt, later)
        by_kind = _events(rows)
        notice = self.receipts(where, result, by_kind)          # config_receipts, notice_receipts (A1)
        receipt = self.review_receipts(where, result, by_kind)
        reviews = [value for _, value in by_kind.get("submit_review", ())]
        event = self.review_event(where, by_kind, reviews, receipt)
        self.answer_source(where, result, by_kind, receipt, event)
        self.request(where, by_kind, event, notice)
        if not event:
            return
        review_call = event.get("call_index")
        exchanges = _at(by_kind, "model_exchange", review_call + 1) if isinstance(review_call, int) else []
        if not exchanges:
            self.next_response["not_requested"] += 1
            return
        parsed = _at(by_kind, "parsed_actions", review_call + 1)
        errors = _at(by_kind, "format_error", review_call + 1)
        self.checks.check("next_response_recorded", [] if len(parsed) + len(errors) == 1 else [
            f"{len(parsed)} parsed_actions and {len(errors)} format_error event(s) at call_index {review_call + 1}"],
            where)
        self.next_response["received"] += 1
        self.next_response["valid"] += len(parsed) == 1 and not errors
        self.next_response["format_errors"] += bool(errors)

    def review_receipts(self, where, result, by_kind):
        exposure = result.get("exposure") if isinstance(result.get("exposure"), dict) else {}
        accounting = result.get("accounting") if isinstance(result.get("accounting"), dict) else {}
        receipt = exposure.get("submit_review")
        finished = [value for _, value in by_kind.get("episode_finished", ())]
        finished_accounting = finished[0].get("accounting") if len(finished) == 1 and isinstance(finished[0], dict) \
            else None
        problems = []
        if not isinstance(receipt, dict) or set(receipt) != set(RECEIPT_FIELDS):
            problems.append("exposure.submit_review is missing or does not have exactly the receipt fields")
            receipt = receipt if isinstance(receipt, dict) else {}
        if accounting.get("submit_review") != receipt:
            problems.append("accounting.submit_review differs from exposure.submit_review")
        if not isinstance(finished_accounting, dict) or finished_accounting.get("submit_review") != receipt:
            problems.append("the episode_finished event's submit_review differs from the result's")
        if receipt and (receipt.get("mode"), receipt.get("before_call_index")) != (self.arm, FINAL_CALL):
            problems.append(f"mode {receipt.get('mode')!r} before_call_index {receipt.get('before_call_index')!r}, "
                            f"expected {self.arm!r} {FINAL_CALL}")
        flags = [flag for flag in ("reviewed", "review_skipped_last_call") if type(receipt.get(flag)) is not bool]
        if receipt and flags:
            problems.append(f"{', '.join(flags)} not true or false")
        if "evidence_ledger" in exposure or "evidence_ledger" in accounting or by_kind.get("evidence_ledger"):
            problems.append("an A5 episode has no last-call evidence ledger")
        self.checks.check("review_receipts", problems, where)
        return receipt

    def review_event(self, where, by_kind, reviews, receipt):
        """The review_event check; returns the submit_review event ({} when there is not exactly one)."""
        problems = []
        if not reviews:
            expected = unreviewed_receipt(self.arm)
            problems += [f"receipt {key} {receipt.get(key)!r} without a review"[:300] for key, value in expected.items()
                         if receipt and receipt.get(key) != value]
            self.checks.check("review_event", problems, where)
            return {}
        if len(reviews) > 1:
            self.checks.check("review_event", [f"{len(reviews)} submit_review events"], where)
            return {}
        event = reviews[0]
        if not isinstance(event, dict):
            self.checks.check("review_event", ["the submit_review event is not an object"], where)
            return {}
        call_index, text = event.get("call_index"), event.get("text")
        if type(call_index) is not int or not 0 <= call_index < FINAL_CALL:
            problems.append(f"a review at call_index {call_index!r} (only before the last call, {FINAL_CALL})")
        earlier = [value for value in _accepted(by_kind) if isinstance(call_index, int)
                   and value.get("call_index", FINAL_CALL + 1) < call_index]
        if earlier:
            problems.append("a submission was accepted before the review")
        if not isinstance(text, str):
            problems.append("the event has no text")
            self.checks.check("review_event", problems, where)
            return event
        if self.arm == "neutral":
            if text != NEUTRAL:
                problems.append("the text is not the neutral review sentence")
            if "ledger" in event:
                problems.append("a neutral review carries a ledger")
        else:
            ledger = event.get("ledger") if isinstance(event.get("ledger"), dict) else {}
            entries = ledger.get("entries")
            rebuilt = A3.ledger_text(entries) if isinstance(entries, list) else None
            if rebuilt is None:
                problems.append("the ledger entries are missing or malformed")
            else:
                if rebuilt[0] != ledger.get("text") or text != rebuilt[0] + "\n" + INSTRUCTION:
                    problems.append("the text is not the frozen ledger of the recorded entries + '\\n' + instruction")
                if ledger.get("body_bytes") != rebuilt[1]:
                    problems.append(f"body_bytes {ledger.get('body_bytes')!r}, the entries give {rebuilt[1]}")
                if len(entries) > A3.MAX_ENTRIES or rebuilt[1] > A3.MAX_BYTES:
                    problems.append("the ledger breaks a cap")
                if not isinstance(ledger.get("candidates"), int) or ledger["candidates"] < len(entries):
                    problems.append(f"candidates {ledger.get('candidates')!r} with {len(entries)} entries")
                if any(not isinstance(e, dict) or not isinstance(e.get("source"), dict)
                       or e["source"].get("kind") not in A3.ENTRY_SOURCES for e in entries):
                    problems.append("an entry has no source of a recorded kind")
                if (receipt.get("ledger_entries"), receipt.get("ledger_candidates"),
                        receipt.get("ledger_body_bytes")) != (len(entries), ledger.get("candidates"), rebuilt[1]):
                    problems.append("the receipt's ledger sizes are not the event's")
                elif not problems:
                    sizes = self.ledger_sizes
                    sizes["entries"].append(len(entries))
                    sizes["candidates"].append(ledger["candidates"])
                    sizes["body_bytes"].append(rebuilt[1])
                    sizes["empty"] += not entries
        content = PREFIX + text
        expected_event = {"mode": self.arm, "text_sha256": sha256_text(text), "content": content,
                          "content_sha256": sha256_text(content)}
        problems += [f"event {key} {event.get(key)!r}, expected {value!r}"[:300]
                     for key, value in expected_event.items() if event.get(key) != value]
        expected_receipt = {"reviewed": True, "review_call_index": call_index,
                            "review_tool_call_id": event.get("tool_call_id"), "text_sha256": sha256_text(text),
                            "content_sha256": sha256_text(content), "review_skipped_last_call": False,
                            "first_call_index": call_index, "first_submission": event.get("submission"),
                            "first_submission_validation": event.get("validation")}
        problems += [f"receipt {key} {receipt.get(key)!r}, expected {value!r}"[:300]
                     for key, value in expected_receipt.items() if receipt.get(key) != value]
        visible = [value for _, value in by_kind.get("model_visible_tool_result", ())
                   if isinstance(value, dict) and value.get("tool_call_id") == event.get("tool_call_id")]
        if len(visible) != 1 or visible[0].get("content") != content:
            problems.append("the reviewed call's model-visible tool result is not the event's content")
        self.checks.check("review_event", problems, where)
        if not problems:
            self.review["reviewed"] += 1
            self.review[f"call:{call_index}"] += 1
        return event

    def answer_source(self, where, result, by_kind, receipt, event):
        accepted = _accepted(by_kind)
        resolved = [value for _, value in by_kind.get("submit_review_resolved", ())]
        source, status, submitted = receipt.get("answer_source"), result.get("exit_status"), result.get("submitted")
        problems = []
        if len(accepted) > 1:
            problems.append(f"{len(accepted)} submission_accepted events")
        if event:
            review_call = event.get("call_index")
            if accepted:
                value = accepted[0]
                if resolved:
                    problems.append("a resubmitted episode has a submit_review_resolved event")
                if source != "resubmit" or receipt.get("resubmit_call_index") != value.get("call_index") \
                        or receipt.get("resubmit_tool_call_id") != value.get("tool_call_id"):
                    problems.append(f"answer_source {source!r} / resubmit fields do not name the accepted resubmission")
                if isinstance(review_call, int) and value.get("call_index", -1) < review_call:
                    problems.append("the accepted submission precedes the review")
                if status != "Submitted" or submitted != value.get("submission"):
                    problems.append("result.json does not carry the resubmission as the Submitted answer")
                if receipt.get("unresolved_exit") is not None:
                    problems.append("unresolved_exit set although the agent resubmitted")
                if review_call == value.get("call_index"):
                    self.review["same_response"] += 1
            elif status == "InfraInterrupted":
                problems.append("an infrastructure attempt is not terminal")
            else:
                unresolved = receipt.get("unresolved_exit")
                if len(resolved) != 1 or not isinstance(resolved[0], dict) \
                        or resolved[0].get("unresolved_exit") != unresolved \
                        or resolved[0].get("answer_source") != "first_submit":
                    problems.append("a review without a resubmission needs exactly one submit_review_resolved event "
                                    "agreeing with the receipt")
                if source != "first_submit" or unresolved not in UNRESOLVED_EXITS:
                    problems.append(f"answer_source {source!r} with unresolved_exit {unresolved!r}")
                if status != "Submitted" or submitted != receipt.get("first_submission") \
                        or submitted != event.get("submission"):
                    problems.append("result.json does not carry the first submission as the Submitted answer")
                if receipt.get("resubmit_call_index") is not None or receipt.get("resubmit_tool_call_id") is not None:
                    problems.append("resubmit fields set without a resubmission")
                if unresolved in UNRESOLVED_EXITS:
                    self.unresolved[unresolved] += 1
        else:
            if resolved:
                problems.append("a submit_review_resolved event without a review")
            if accepted:
                value = accepted[0]
                if value.get("call_index") != FINAL_CALL:
                    problems.append(f"a first submission accepted at call_index {value.get('call_index')!r} without a "
                                    "review (only the last call skips it)")
                expected = {"review_skipped_last_call": True, "first_call_index": value.get("call_index"),
                            "first_submission": value.get("submission"), "answer_source": "first_submit",
                            "resubmit_call_index": None, "unresolved_exit": None}
                problems += [f"receipt {key} {receipt.get(key)!r}, expected {expected_value!r}"[:300]
                             for key, expected_value in expected.items() if receipt.get(key) != expected_value]
                if status != "Submitted" or submitted != value.get("submission"):
                    problems.append("result.json does not carry the accepted submission")
                if not problems:
                    self.review["skipped_last_call"] += 1
            else:
                expected = {"review_skipped_last_call": False, "first_call_index": None, "first_submission": None,
                            "answer_source": None, "resubmit_call_index": None, "unresolved_exit": None}
                problems += [f"receipt {key} {receipt.get(key)!r}, expected {expected_value!r}"[:300]
                             for key, expected_value in expected.items() if receipt.get(key) != expected_value]
                if status == "Submitted":
                    problems.append("Submitted without any accepted or reviewed submission")
                if not problems:
                    self.review["never_submitted"] += 1
        self.checks.check("answer_source", problems, where)
        if not problems:
            self.sources[str(source)] += 1

    def request(self, where, by_kind, event, notice):
        """The request bytes the harness recorded (model_exchange.request.messages) of every call."""
        problems, checked = [], False
        review_call = event.get("call_index") if event else None
        tool_call_id = event.get("tool_call_id") if event else None
        content = event.get("content") if event else None
        for _, exchange in by_kind.get("model_exchange", ()):
            if not isinstance(exchange, dict):
                continue
            request = exchange.get("request") if isinstance(exchange.get("request"), dict) else {}
            messages = request.get("messages")
            call = exchange.get("call_index")
            if not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages):
                problems.append(f"call {call!r}: the request has no messages")
                continue
            checked = True
            dumped = json.dumps(messages, ensure_ascii=False)
            after = isinstance(review_call, int) and isinstance(call, int) and call > review_call
            headers = dumped.count(HEADER)
            expected_headers = 1 if (after and self.arm == "ledger") else 0
            if headers != expected_headers:
                problems.append(f"call {call!r}: the ledger header occurs {headers} time(s), expected "
                                f"{expected_headers}")
            reviewed = [m for m in messages if m.get("role") == "tool" and tool_call_id is not None
                        and m.get("tool_call_id") == tool_call_id]
            if not after:
                if reviewed:
                    problems.append(f"call {call!r}: the review appears before it was made")
                continue
            if len(reviewed) != 1:
                problems.append(f"call {call!r}: {len(reviewed)} tool message(s) of the reviewed call")
                continue
            noticed = (call == FINAL_CALL and messages[-1] is reviewed[0] and notice.get("appended_to") == "tool_result"
                       and notice.get("tool_call_id") == tool_call_id)
            if reviewed[0].get("content") != (content + DEADLINE_SUFFIX if noticed else content):
                problems.append(f"call {call!r}: the reviewed call's tool message is not the review content"
                                + (" + the deadline suffix" if noticed else ""))
        if checked or problems:
            self.checks.check("request_bytes", problems, where)


def _accepted(by_kind):
    return [value for _, value in by_kind.get("submission_accepted", ()) if isinstance(value, dict)]


def check_run(run_dir, arm, *, seed=SEED, cell_name=None):
    """The report of one dev run (see the module docstring); raises InputError when the run cannot be checked."""
    return DevCheck(run_dir, arm, seed=seed, cell_name=cell_name).run()


def summary(report):
    rule, review = report["go_rule"], report["review"]
    share = "n/a" if rule["share"] is None else f"{rule['share']:.1%}"
    reviewed = "n/a" if review["reviewed_share"] is None else f"{review['reviewed_share']:.1%}"
    return (f"{'GO' if report['go'] else 'NO-GO'}: A5 {report['arm']} run {report['run']['run_id']}: "
            f"{rule['valid_tool_calls']}/{rule['received_review']} responses after a review are valid tool calls "
            f"({share}; go rule >= 90%: {'met' if rule['passed'] else 'not met'}); reviewed {review['reviewed']} of "
            f"{review['terminal_attempts']} ({reviewed}); answer sources {review['answer_source']}; failed checks: "
            f"{', '.join(report['failed_checks']) or 'none'}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", required=True, help="the published dev run (run.json, cells.jsonl, cells/)")
    parser.add_argument("--arm", required=True, choices=ARMS, help="the run's submit_review arm")
    parser.add_argument("--seed", type=int, default=SEED, help="the LEDGER seed the run dispatched (default 61)")
    parser.add_argument("--out", required=True, help="the JSON report to create (never overwritten)")
    args = parser.parse_args(argv)
    out = Path(args.out)
    try:
        if out.exists() or out.is_symlink():
            raise InputError(f"refusing to overwrite {out}")
        report = check_run(args.run_dir, args.arm, seed=args.seed)
        A1.write_report(out, report)
    except (InputError, OSError) as error:
        print(f"expA_a5_devcheck: ERROR: {error}", file=sys.stderr)
        return 2
    print(summary(report))
    print(f"report: {out}")
    return 0 if report["go"] else 1


if __name__ == "__main__":
    sys.exit(main())
