#!/usr/bin/env python3
"""Exp A follow-up A3: the dev-family check before the lock (protocols/EXPA_A3_LEDGER.md, section 2.6).

    python -B scripts/expA_a3_devcheck.py --run-dir RUN_DIR --arm off|on [--expect-budget 16:32768] --out REPORT.json

Run once per dev run (block LEDGER, seed 61, the dev families' dispatch view), over the published run. Standard
library only: it reads run.json, cells.jsonl and cells/<cell_id>/<attempt_id>/ (attempt_start.json, result.json,
events.jsonl) and writes only the report. The run layout, the terminal-attempt rule (D11.3), the record bindings and
the A1 receipt checks are those of scripts/expA_finalturn_devcheck.py (the A1 dev check), whose helpers are reused.

Go, in each arm (section 2.6), as the CHECKS below spell it out for the records:
- every episode that reached its last call (a final_turn_notice event exists) records the ledger: exactly one
  evidence_ledger event, the row right before the final_turn_notice event, agreeing with the receipt in
  result.json exposure.evidence_ledger, accounting.evidence_ledger and the episode_finished event's (computed true;
  "on": appended, and sent and delivered with the notice; "off": never appended, sent or delivered); an episode that
  did not reach it has no evidence_ledger event and the unbuilt receipt;
- under "on", every delivered 8th request's last message ends with "\\n\\n[harness notice] " + the recorded ledger
  text + the deadline suffix, with the ledger header exactly once in the request; under "off", no request contains
  the header (and the 8th request's last message is the A1 deadline message);
- at least 90% of the final responses that received the notice (terminal attempts with a model_exchange at
  call_index 7) are valid tool calls (a parsed_actions event, not a format_error event, at call_index 7);
- run.json records the arm's exact episode_config (deadline notice, evidence_ledger = the arm and its rules) and every
  dispatched cell is LEDGER|<item>|vcomplete or omit_node|C1|s61 (and every such cell of cells.jsonl is terminal).
Reported, not gated: ledger sizes (entries, empty share, body bytes, candidates), ContextLimit at the last call, and
notices appended to a format-error message.

Follow-up A6 (protocols/EXPA_A6_BUDGET16.md): the same check reads an A6 dev run (arm off, block LEDGER, seed 61,
16 responses and a 32,768-token context). The last call is not a constant: it is run.json's
episode_config.max_model_calls - 1 (call_index 7 at the prereg 8 responses, 15 at 16), and every check above reads that
call (the "8th request" is then the 16th). With --expect-budget CALLS:TOKENS (8:16384 or 16:32768) the arm's expected
episode_config is the record at that budget and one more gated check, run_json_budget, requires run.json to record
max_model_calls CALLS and context_budget TOKENS; the report then has a "budget" section. Without the option the
expected record is the prereg one and the report of an 8-response run is unchanged.

The served prompt (token ids) is not detokenized here: the A1 check covered that rendering of the in-band notice on
the same message; this check reads the request bytes the harness records (model_exchange.request.messages).

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
_SPEC = importlib.util.spec_from_file_location("expA_finalturn_devcheck", _HERE / "expA_finalturn_devcheck.py")
A1 = sys.modules.get("expA_finalturn_devcheck")
if A1 is None:
    A1 = importlib.util.module_from_spec(_SPEC)
    sys.modules["expA_finalturn_devcheck"] = A1
    _SPEC.loader.exec_module(A1)

InputError, sha256_text, sha256_bytes, sha256_file = A1.InputError, A1.sha256_text, A1.sha256_bytes, A1.sha256_file
_events, _at = A1._events, A1._at

SCHEMA = "expA-a3-devcheck-v1"
PROTOCOL = "protocols/EXPA_A3_LEDGER.md"
POLICY = "deadline"                                       # section 2.2: both arms use --final-turn-policy deadline
ARMS = ("off", "on")
PREFIX = A1.PREFIX                                        # "\n\n[harness notice] "
DEADLINE_SUFFIX = PREFIX + A1.TEXTS[POLICY]
FINAL_CALL = A1.FINAL_CALL                                # 7
# The frozen ledger (section 2.3; absent_edges/probe/harness.py LEDGER_*), declared again rather than imported so
# that the check does not share code with what it audits; the tests pin these values to the harness source.
HEADER = "Candidate call sites seen in your earlier tool outputs (unverified; check before submitting):"
EMPTY = "(none)"
ENTRY = "- {path}:{line}: {excerpt}"
MAX_ENTRIES, MAX_BYTES, EXCERPT_CHARS = 20, 2000, 120
PRINTED_LINE = r"(?P<path>[A-Za-z0-9_.+@~/-]+\.py):(?P<line>[1-9][0-9]*):(?P<text>.*)"
CALL_TOKEN = r"(?<![A-Za-z0-9_]){name}[ \t]*\("
DEFINITION = r"(?:^|[^A-Za-z0-9_])def[ \t]+$"
RULE_TEXTS = {
    "sources": "the model-visible tool results already in the history at the last call, in order: each "
               "source_reference_search hit (path, line, text) in hit order, and each bash output line that "
               "is exactly PATH:LINE:TEXT; paths normalized as submit_answer does (/repo/ and ./ stripped), "
               "absolute or '..' paths skipped",
    "selection": "TEXT contains the target name (last qualname component) as a call token that is not a "
                 "definition; deduplicated by (path, line), first appearance kept, in order of first "
                 "appearance; excerpt = TEXT with control characters as spaces, stripped, first 120 "
                 "characters; entries added while at most 20 and at most 2,000 bytes",
    "placement": "appended to the same message as the final-turn notice, before it",
}
RECEIPT_FIELDS = ("mode", "call_index", "computed", "appended", "sent", "delivered", "appended_to", "tool_call_id",
                  "candidates", "entries", "body_bytes", "text_sha256", "suffix_sha256")
ENTRY_SOURCES = ("search_hit", "bash_line")
BLOCK, SEED, CONDITION, VARIANTS = "LEDGER", 61, "C1", ("vcomplete", "omit_node")
# --expect-budget (protocols/EXPA_A6_BUDGET16.md): the registered (max_model_calls, context_budget) of an A3-design run.
PREREG_BUDGET = (A1.FROZEN_CONFIG["max_model_calls"], A1.FROZEN_CONFIG["context_budget"])     # (8, 16384)
BUDGETS = {"8:16384": PREREG_BUDGET, "16:32768": (16, 32768)}
A6_PROTOCOL = "protocols/EXPA_A6_BUDGET16.md"
BUDGET_PROTOCOLS = {PREREG_BUDGET: PROTOCOL, (16, 32768): A6_PROTOCOL}
GO_RULE = A1.GO_RULE                                      # valid / noticed >= 9/10, compared exactly
EXAMPLES = A1.EXAMPLES

CHECKS = {
    "run_json_episode_config": "run.json episode_config is the arm's exact record (the six prereg fields, the "
                               "deadline notice, evidence_ledger = the arm and evidence_ledger_rules)",
    "run_json_cells": "cells.jsonl is the file run.json binds (cells_sha256)",
    "selection": "every dispatched cell is a LEDGER|<item>|vcomplete or omit_node|C1|s61 cell of cells.jsonl",
    "run_complete": "every LEDGER seed-61 cell of cells.jsonl is terminal in the D11.3 sense (a completed attempt, or "
                    "missing after 3 infrastructure attempts)",
    "records": "each terminal attempt: its directory names its cell and attempt, result.json binds attempt_start.json "
               "and events.jsonl (sha256 and rows), its run_id is run.json's, and no attempt follows it",
    "config_receipts": "result.json config and the episode_setup event's config are the arm's record",
    "notice_receipts": "the A1 final_turn_notice receipt is consistent (as the A1 dev check reads it)",
    "ledger_receipts": "result.json exposure.evidence_ledger equals accounting.evidence_ledger and the "
                       "episode_finished event's, has exactly the receipt fields, mode = the arm and call_index 7",
    "ledger_event": "an episode that reached its last call (a final_turn_notice event) has exactly one evidence_ledger "
                    "event, the row right before the final_turn_notice event, agreeing with the receipt (text sha256, "
                    "suffix, entries, candidates, body bytes, appended, appended_to, tool_call_id, message sha256s) and "
                    "with the frozen text format; computed is true; on: appended, and sent and delivered as the "
                    "notice; off: never appended, sent or delivered. An episode that did not reach it has no event "
                    "and the unbuilt receipt",
    "request_bytes": "on: the delivered 8th request's last message ends with '\\n\\n[harness notice] ' + the ledger "
                     "text + the deadline suffix, is the content the final_turn_notice event binds, and the ledger "
                     "header occurs exactly once in the request (never in an earlier request); off: no request "
                     "contains the header and the 8th request's last message ends with the deadline suffix alone",
    "final_response_recorded": "each delivered final response has exactly one parsed_actions or format_error event "
                               "at call_index 7",
}
BUDGET_CHECK = ("run_json_budget", "run.json episode_config records max_model_calls {calls} and context_budget "
                "{tokens} (--expect-budget, {protocol})")


def ordinal(number):
    """1st, 2nd, 3rd, 8th, 16th, ..."""
    suffix = "th" if 10 <= number % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return f"{number}{suffix}"


def describe(text, final_call):
    """A CHECKS text (written for call_index 7, the 8th request) for a run whose last call is ``final_call``."""
    if final_call == FINAL_CALL:
        return text
    return text.replace(f"call_index {FINAL_CALL}", f"call_index {final_call}").replace(
        ordinal(FINAL_CALL + 1), ordinal(final_call + 1))


def expected_rules(arm, final_call=FINAL_CALL):
    """evidence_ledger_record(arm, final_call + 1) of the release, rebuilt from the protocol's values."""
    return {"protocol": PROTOCOL, "mode": arm, "delivered_to_model": arm == "on", "channel": "in_band",
            "prefix": PREFIX, "header": HEADER, "empty": EMPTY, "entry": ENTRY, "max_entries": MAX_ENTRIES,
            "max_bytes": MAX_BYTES, "excerpt_chars": EXCERPT_CHARS, "printed_line": PRINTED_LINE,
            "call_token": CALL_TOKEN, "definition": DEFINITION, "call_index": final_call, **RULE_TEXTS}


def expected_episode_config(arm, budget=None):
    """EpisodeConfig(final_turn_policy="deadline", evidence_ledger=arm).record() of the release; with ``budget``
    (max_model_calls, context_budget), the record of EpisodeConfig(max_model_calls=..., context_budget=..., same
    options): the two budget fields and the notice's and the rules' call_index (max_model_calls - 1) change."""
    if arm not in ARMS:
        raise InputError(f"--arm must be one of {', '.join(ARMS)}")
    config = dict(A1.expected_episode_config(POLICY), evidence_ledger=arm, evidence_ledger_rules=expected_rules(arm))
    if budget is None or tuple(budget) == PREREG_BUDGET:
        return config
    if tuple(budget) not in BUDGET_PROTOCOLS:
        raise InputError(f"--expect-budget must be one of {', '.join(BUDGETS)}")
    calls, tokens = budget
    config.update(max_model_calls=calls, context_budget=tokens,
                  final_turn_notice=dict(config["final_turn_notice"], call_index=calls - 1),
                  evidence_ledger_rules=expected_rules(arm, calls - 1))
    return config


def unbuilt_receipt(arm, final_call=FINAL_CALL):
    return {"mode": arm, "call_index": final_call, "computed": False, "appended": False, "sent": False,
            "delivered": False, "appended_to": None, "tool_call_id": None, "candidates": None, "entries": None,
            "body_bytes": None, "text_sha256": None, "suffix_sha256": None}


def ledger_text(entries):
    """(text, body_bytes) the frozen format gives these recorded entries, or None when an entry is malformed."""
    lines = []
    for entry in entries:
        if not (isinstance(entry, dict) and isinstance(entry.get("path"), str) and type(entry.get("line")) is int
                and isinstance(entry.get("excerpt"), str)):
            return None
        lines.append(ENTRY.format(path=entry["path"], line=entry["line"], excerpt=entry["excerpt"]))
    body = "\n".join(lines)
    return HEADER + "\n" + (body if lines else EMPTY), len(body.encode("utf-8"))


def cell_parts(cell_id):
    parts = cell_id.split("|") if isinstance(cell_id, str) else []
    return parts if len(parts) == 5 else None


def is_ledger_cell(cell_id):
    parts = cell_parts(cell_id)
    return bool(parts) and parts[0] == BLOCK and parts[2] in VARIANTS and parts[3] == CONDITION \
        and parts[4] == f"s{SEED}"


class Checks(A1.Checks):
    def __init__(self, descriptions=None):
        self.results = {name: {"description": description, "checked": 0, "failed": 0, "examples": []}
                        for name, description in (descriptions or CHECKS).items()}

    def describe(self, final_call):
        """Restate the descriptions for a run whose last call is ``final_call`` (unchanged at call_index 7)."""
        for entry in self.results.values():
            entry["description"] = describe(entry["description"], final_call)


class DevCheck(A1.DevCheck):
    """The A3 check; reuses the A1 check's records() and receipts() (records, config_receipts, notice_receipts)."""

    def __init__(self, run_dir, arm, *, cell_name=None, expect_budget=None):
        if arm not in ARMS:
            raise InputError(f"--arm must be one of {', '.join(ARMS)}")
        if expect_budget is not None and tuple(expect_budget) not in BUDGET_PROTOCOLS:
            raise InputError(f"--expect-budget must be one of {', '.join(BUDGETS)}")
        super().__init__(run_dir, POLICY, None)
        self.arm = arm
        self.expect_budget = None if expect_budget is None else tuple(expect_budget)
        self.expected = expected_episode_config(arm, self.expect_budget)
        descriptions = dict(CHECKS)
        if self.expect_budget is not None:      # gated only when asked for: an earlier report is unchanged
            name, text = BUDGET_CHECK
            calls, tokens = self.expect_budget
            items = list(CHECKS.items())
            descriptions = dict(items[:1] + [(name, text.format(calls=calls, tokens=tokens,
                                                                protocol=BUDGET_PROTOCOLS[self.expect_budget]))]
                                + items[1:])
        self.checks = Checks(descriptions)
        self.final_call = FINAL_CALL            # run(): run.json's max_model_calls - 1
        # Directory name -> cell or attempt id. Identity on the hosts (cell ids, with '|', are directory names);
        # a test hook for filesystems that cannot hold '|'.
        self.cell_name = cell_name or (lambda name: name)
        self.sizes = {"built": 0, "entries": [], "candidates": [], "body_bytes": [], "empty": 0, "capped": 0,
                      "by_source": collections.Counter()}
        self.last_call = collections.Counter()

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
        registered = {cell for cell in design_ids if is_ledger_cell(cell)}
        self.run_json = run
        config = run.get("episode_config")
        # The last call is the run's (protocols/EXPA_A6_BUDGET16.md): run.json's max_model_calls - 1; a run.json
        # without a usable max_model_calls is checked at the prereg call_index 7 (and fails the config check).
        recorded = config if isinstance(config, dict) else {}
        calls = recorded.get("max_model_calls")
        self.final_call = calls - 1 if type(calls) is int and calls >= 2 else FINAL_CALL
        self.checks.describe(self.final_call)
        self.checks.check("run_json_episode_config", [] if config == self.expected else [
            f"episode_config {json.dumps(config, sort_keys=True)[:600]}"], "run.json")
        if self.expect_budget is not None:
            found = (recorded.get("max_model_calls"), recorded.get("context_budget"))
            self.checks.check(BUDGET_CHECK[0], [] if found == self.expect_budget else [
                f"max_model_calls {found[0]!r} and context_budget {found[1]!r}, expected "
                f"{self.expect_budget[0]} and {self.expect_budget[1]}"], "run.json")
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
                f"cell {cell} is not a {BLOCK}|<item>|vcomplete or omit_node|{CONDITION}|s{SEED} cell of "
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

        go = self.go
        share = go["valid"] / go["noticed"] if go["noticed"] else None
        rule_passed = go["noticed"] > 0 and GO_RULE[1] * go["valid"] >= GO_RULE[0] * go["noticed"]
        failed = self.checks.failed()
        sizes = self.sizes
        built = sizes["built"]

        def mean(values):
            return sum(values) / len(values) if values else None
        report = {
            "schema": SCHEMA, "protocol": PROTOCOL, "arm": self.arm, "final_turn_policy": POLICY,
            "header": HEADER, "go": rule_passed and not failed, "failed_checks": failed,
            "go_rule": {"rule": describe("at least 90% of the final responses that received the notice (terminal "
                                         "attempts with a model_exchange at call_index 7) are valid tool calls "
                                         "(parsed_actions, not format_error, at call_index 7)", self.final_call),
                        "noticed_final_responses": go["noticed"], "valid_tool_calls": go["valid"],
                        "format_errors": go["format_errors"], "share": share, "threshold": GO_RULE[0] / GO_RULE[1],
                        "passed": rule_passed},
            "checks": self.checks.report(),
            "ledger": {"terminal_attempts": sum(self.exits.values()), "built": built,
                       "appended": self.last_call["ledger_appended"], "sent": self.last_call["ledger_sent"],
                       "delivered": self.last_call["ledger_delivered"],
                       "entries_mean": mean(sizes["entries"]), "entries_max": max(sizes["entries"], default=None),
                       "entries_histogram": dict(sorted(collections.Counter(sizes["entries"]).items())),
                       "empty": sizes["empty"], "empty_share": sizes["empty"] / built if built else None,
                       "candidates_mean": mean(sizes["candidates"]),
                       "capped": sizes["capped"],
                       "body_bytes_mean": mean(sizes["body_bytes"]),
                       "body_bytes_max": max(sizes["body_bytes"], default=None),
                       "entries_by_source": dict(sorted(sizes["by_source"].items())),
                       "not_gated": "sizes are reported, not gated (section 2.6)"},
            "last_call": {"reached": self.last_call["reached"],
                          "context_limit_at_last_call": self.last_call["context_limit"],
                          "context_limit_cells": sorted(self.lists["context_limit"]),
                          "notice_on_format_error": self.last_call["format_error"],
                          "format_error_cells": sorted(self.lists["format_error"]),
                          "not_reached_by_exit": dict(sorted(self.not_reached.items())),
                          "not_gated": "reported, not gated (section 2.6)"},
            "cells": {"ledger_seed_61_in_design": len(registered), "dispatched": len(attempts),
                      "states": {state: states[state] for state in ("completed", "missing_infrastructure",
                                                                    "infra_pending", "not_dispatched")},
                      "terminal_exits": dict(sorted(self.exits.items())),
                      "infrastructure_attempts_by_kind": dict(sorted(infra_kinds.items())),
                      "empty_attempt_directories": len(empty),
                      "terminal_only": "each cell's terminal attempt (D11.3: the first with a completed model exit); "
                                       "infrastructure attempts are not checked or counted"},
            "run": {"run_dir": str(self.run_dir), "run_id": run.get("run_id"), "run_json_sha256": sha256_bytes(run_raw),
                    "cells_jsonl_sha256": sha256_bytes(cells_raw), "model": run.get("model")},
            "script": {"path": "scripts/expA_a3_devcheck.py", "sha256": sha256_file(__file__)},
        }
        if self.expect_budget is not None:      # --expect-budget only: the report of an earlier check is unchanged
            report["budget"] = {
                "expected": {"max_model_calls": self.expect_budget[0], "context_budget": self.expect_budget[1]},
                "recorded": {"max_model_calls": recorded.get("max_model_calls"),
                             "context_budget": recorded.get("context_budget")},
                "last_call_index": self.final_call, "protocol": BUDGET_PROTOCOLS[self.expect_budget],
                "passed": report["checks"][BUDGET_CHECK[0]]["passed"]}
        return report

    # -- one terminal attempt ----------------------------------------------------------------------------------------

    def attempt(self, cell, attempt, later):
        where, result = attempt["where"], attempt["result"]
        self.exits[result.get("exit_status")] += 1
        rows = self.records(cell, attempt, later)
        by_kind = _events(rows)
        notice = self.receipts(where, result, by_kind)          # config_receipts, notice_receipts (A1)
        receipt = self.ledger_receipts(where, result, by_kind)
        notices = by_kind.get("final_turn_notice", [])
        ledgers = by_kind.get("evidence_ledger", [])
        reached = len(notices) >= 1
        final = self.final_call
        exchanges = _at(by_kind, "model_exchange", final)
        if reached:
            self.last_call["reached"] += 1
            if notice.get("appended_to") == "format_error":
                self.last_call["format_error"] += 1
                self.lists["format_error"].append(cell)
            if len(_at(by_kind, "context_limit", final)) == 1:
                self.last_call["context_limit"] += 1
                self.lists["context_limit"].append(cell)
        else:
            self.not_reached[result.get("exit_status")] += 1
        for flag in ("appended", "sent", "delivered"):
            self.last_call["ledger_" + flag] += receipt.get(flag) is True

        event = self.ledger_event(where, rows, by_kind, notices, ledgers, receipt, notice, reached)
        self.request(where, by_kind, event, notices)
        if len(exchanges) != 1:
            return
        parsed, errors = _at(by_kind, "parsed_actions", final), _at(by_kind, "format_error", final)
        self.checks.check("final_response_recorded", [] if len(parsed) + len(errors) == 1 else [
            f"{len(parsed)} parsed_actions and {len(errors)} format_error event(s) at call_index {final}"], where)
        self.go["noticed"] += 1
        self.go["valid"] += len(parsed) == 1 and not errors
        self.go["format_errors"] += bool(errors)

    def ledger_receipts(self, where, result, by_kind):
        exposure = result.get("exposure") if isinstance(result.get("exposure"), dict) else {}
        accounting = result.get("accounting") if isinstance(result.get("accounting"), dict) else {}
        receipt = exposure.get("evidence_ledger")
        finished = [value for _, value in by_kind.get("episode_finished", ())]
        finished_accounting = finished[0].get("accounting") if len(finished) == 1 and isinstance(finished[0], dict) \
            else None
        problems = []
        if not isinstance(receipt, dict) or set(receipt) != set(RECEIPT_FIELDS):
            problems.append("exposure.evidence_ledger is missing or does not have exactly the receipt fields")
            receipt = receipt if isinstance(receipt, dict) else {}
        if accounting.get("evidence_ledger") != receipt:
            problems.append("accounting.evidence_ledger differs from exposure.evidence_ledger")
        if not isinstance(finished_accounting, dict) or finished_accounting.get("evidence_ledger") != receipt:
            problems.append("the episode_finished event's evidence_ledger differs from the result's")
        if receipt and (receipt.get("mode"), receipt.get("call_index")) != (self.arm, self.final_call):
            problems.append(f"mode {receipt.get('mode')!r} call_index {receipt.get('call_index')!r}, expected "
                            f"{self.arm!r} {self.final_call}")
        flags = [flag for flag in ("computed", "appended", "sent", "delivered") if type(receipt.get(flag)) is not bool]
        if receipt and flags:
            problems.append(f"{', '.join(flags)} not true or false")
        self.checks.check("ledger_receipts", problems, where)
        return receipt

    def ledger_event(self, where, rows, by_kind, notices, ledgers, receipt, notice, reached):
        """The ledger_event check; returns the evidence_ledger event ({} when there is not exactly one)."""
        problems = []
        if not reached:
            if ledgers:
                problems.append(f"{len(ledgers)} evidence_ledger event(s) without a final_turn_notice event")
            if receipt and receipt != unbuilt_receipt(self.arm, self.final_call):
                problems.append("the last call was not reached but the receipt is not the unbuilt receipt")
            self.checks.check("ledger_event", problems, where)
            return {}
        if len(ledgers) != 1 or len(notices) != 1:
            problems.append(f"{len(ledgers)} evidence_ledger and {len(notices)} final_turn_notice event(s)")
            self.checks.check("ledger_event", problems, where)
            return {}
        (index, event), (notice_index, notice_event) = ledgers[0], notices[0]
        if not isinstance(event, dict) or not isinstance(notice_event, dict):
            self.checks.check("ledger_event", ["an event value is not an object"], where)
            return {}
        if index + 1 != notice_index:
            problems.append("the evidence_ledger event is not the row right before the final_turn_notice event")
        on = self.arm == "on"
        text = event.get("text")
        entries = event.get("entries")
        if not isinstance(text, str) or not isinstance(entries, list):
            problems.append("the event has no text or entries")
            self.checks.check("ledger_event", problems, where)
            return event
        rebuilt = ledger_text(entries)
        if rebuilt is None:
            problems.append("an entry is malformed")
        else:
            if rebuilt[0] != text:
                problems.append("the text is not the frozen format of the recorded entries")
            if event.get("body_bytes") != rebuilt[1]:
                problems.append(f"body_bytes {event.get('body_bytes')!r}, the entries give {rebuilt[1]}")
        if len(entries) > MAX_ENTRIES or not isinstance(event.get("body_bytes"), int) \
                or event["body_bytes"] > MAX_BYTES:
            problems.append("the ledger breaks a cap")
        if not isinstance(event.get("candidates"), int) or event["candidates"] < len(entries):
            problems.append(f"candidates {event.get('candidates')!r} with {len(entries)} entries")
        for entry in entries:
            source = entry.get("source") if isinstance(entry, dict) else None
            if not isinstance(source, dict) or source.get("kind") not in ENTRY_SOURCES:
                problems.append("an entry has no source of a recorded kind")
                break
        expected_event = {"call_index": self.final_call, "mode": self.arm, "appended": on,
                          "text_sha256": sha256_text(text), "suffix": PREFIX + text,
                          "suffix_sha256": sha256_text(PREFIX + text),
                          "appended_to": notice_event.get("appended_to"),
                          "tool_call_id": notice_event.get("tool_call_id")}
        problems += [f"event {key} {event.get(key)!r}, expected {value!r}"[:300]
                     for key, value in expected_event.items() if event.get(key) != value]
        # The message chain: under on the ledger is appended and the notice follows it; under off it is not sent.
        if event.get("content_sha256") != notice_event.get("previous_content_sha256"):
            problems.append("the event's content_sha256 is not the message the final_turn_notice event follows")
        if not on and event.get("content_sha256") != event.get("previous_content_sha256"):
            problems.append("off: the message changed when the ledger was built")
        if on and event.get("content_sha256") == event.get("previous_content_sha256"):
            problems.append("on: the message did not change when the ledger was appended")
        # The receipt.
        expected_receipt = {"mode": self.arm, "call_index": self.final_call, "computed": True, "appended": on,
                            "sent": on and notice.get("sent") is True,
                            "delivered": on and notice.get("delivered") is True,
                            "appended_to": event.get("appended_to"), "tool_call_id": event.get("tool_call_id"),
                            "candidates": event.get("candidates"), "entries": len(entries),
                            "body_bytes": event.get("body_bytes"), "text_sha256": sha256_text(text),
                            "suffix_sha256": sha256_text(PREFIX + text)}
        problems += [f"receipt {key} {receipt.get(key)!r}, expected {value!r}"[:300]
                     for key, value in expected_receipt.items() if receipt.get(key) != value]
        self.checks.check("ledger_event", problems, where)
        if not problems:
            sizes = self.sizes
            sizes["built"] += 1
            sizes["entries"].append(len(entries))
            sizes["candidates"].append(event["candidates"])
            sizes["body_bytes"].append(event["body_bytes"])
            sizes["empty"] += not entries
            sizes["capped"] += event["candidates"] > len(entries)
            sizes["by_source"].update(entry["source"]["kind"] for entry in entries)
        return event

    def request(self, where, by_kind, event, notices):
        """The request bytes the harness recorded (model_exchange.request.messages) of every call."""
        problems = []
        on = self.arm == "on"
        notice_event = notices[0][1] if len(notices) == 1 and isinstance(notices[0][1], dict) else {}
        checked = False
        last = ordinal(self.final_call + 1)          # "8th" at the prereg budget
        for _, exchange in by_kind.get("model_exchange", ()):
            if not isinstance(exchange, dict):
                continue
            request = exchange.get("request") if isinstance(exchange.get("request"), dict) else {}
            messages = request.get("messages")
            if not isinstance(messages, list) or not messages or not all(isinstance(m, dict) for m in messages):
                problems.append(f"call {exchange.get('call_index')!r}: the request has no messages")
                continue
            checked = True
            headers = json.dumps(messages, ensure_ascii=False).count(HEADER)
            final = exchange.get("call_index") == self.final_call
            expected_headers = 1 if on and final else 0
            if headers != expected_headers:
                problems.append(f"call {exchange.get('call_index')!r}: the ledger header occurs {headers} time(s), "
                                f"expected {expected_headers}")
            if not final:
                continue
            content = messages[-1].get("content")
            if not isinstance(content, str):
                problems.append(f"the {last} request's last message has no text content")
                continue
            if sha256_text(content) != notice_event.get("content_sha256"):
                problems.append(f"the {last} request's last message is not the content the final_turn_notice event "
                                "binds")
            if on:
                text = event.get("text")
                tail = PREFIX + text + DEADLINE_SUFFIX if isinstance(text, str) else None
                if tail is None or not content.endswith(tail):
                    problems.append("the last message does not end with the prefix, the ledger text and the "
                                    "deadline suffix")
                elif sha256_text(content[:-len(DEADLINE_SUFFIX)]) != event.get("content_sha256"):
                    problems.append("the content before the deadline suffix is not the one the ledger event binds")
            else:
                if not content.endswith(DEADLINE_SUFFIX):
                    problems.append("the last message does not end with the deadline suffix")
                elif sha256_text(content[:-len(DEADLINE_SUFFIX)]) != event.get("previous_content_sha256"):
                    problems.append("off: the content before the deadline suffix is not the unledgered message")
        if checked or problems:
            self.checks.check("request_bytes", problems, where)


def check_run(run_dir, arm, *, cell_name=None, expect_budget=None):
    """The report of one dev run (see the module docstring); raises InputError when the run cannot be checked."""
    return DevCheck(run_dir, arm, cell_name=cell_name, expect_budget=expect_budget).run()


def summary(report):
    rule, ledger, last = report["go_rule"], report["ledger"], report["last_call"]
    share = "n/a" if rule["share"] is None else f"{rule['share']:.1%}"
    entries = "n/a" if ledger["entries_mean"] is None else f"{ledger['entries_mean']:.2f}"
    empty = "n/a" if ledger["empty_share"] is None else f"{ledger['empty_share']:.1%}"
    return (f"{'GO' if report['go'] else 'NO-GO'}: A3 {report['arm']} run {report['run']['run_id']}: "
            f"{rule['valid_tool_calls']}/{rule['noticed_final_responses']} noticed final responses are valid tool "
            f"calls ({share}; go rule >= 90%: {'met' if rule['passed'] else 'not met'}); ledgers built "
            f"{ledger['built']} (entries mean {entries}, empty {empty}); ContextLimit at last call "
            f"{last['context_limit_at_last_call']}; notices on a format-error message {last['notice_on_format_error']}; "
            f"failed checks: {', '.join(report['failed_checks']) or 'none'}"
            + ("" if "budget" not in report else
               f"; budget {report['budget']['expected']['max_model_calls']}/"
               f"{report['budget']['expected']['context_budget']} ({report['budget']['protocol']}): "
               f"{'recorded' if report['budget']['passed'] else 'NOT recorded'}, last call_index "
               f"{report['budget']['last_call_index']}"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", required=True, help="the published dev run (run.json, cells.jsonl, cells/)")
    parser.add_argument("--arm", required=True, choices=ARMS, help="the run's evidence_ledger arm")
    parser.add_argument("--expect-budget", choices=sorted(BUDGETS), default=None,
                        help="CALLS:TOKENS the run must record (16:32768 for the A6 dev run, "
                             + A6_PROTOCOL + "); also sets the expected episode_config")
    parser.add_argument("--out", required=True, help="the JSON report to create (never overwritten)")
    args = parser.parse_args(argv)
    out = Path(args.out)
    try:
        if out.exists() or out.is_symlink():
            raise InputError(f"refusing to overwrite {out}")
        if args.expect_budget is None:          # the A3 check, exactly as before
            report = check_run(args.run_dir, args.arm)
        else:
            report = check_run(args.run_dir, args.arm, expect_budget=BUDGETS[args.expect_budget])
        A1.write_report(out, report)
    except (InputError, OSError) as error:
        print(f"expA_a3_devcheck: ERROR: {error}", file=sys.stderr)
        return 2
    print(summary(report))
    print(f"report: {out}")
    return 0 if report["go"] else 1


if __name__ == "__main__":
    sys.exit(main())
