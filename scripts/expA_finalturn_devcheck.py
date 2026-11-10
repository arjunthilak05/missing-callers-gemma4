#!/usr/bin/env python3
"""Exp A follow-up A1: the dev-family check before the lock (protocols/EXPA_FINALTURN.md, section 2).

    python -B scripts/expA_finalturn_devcheck.py --run-dir RUN_DIR \\
        --policy neutral|deadline|deadline_info|submit_only [--seed 11|23] [--conditions C0,C1] \\
        --tokenizer-dir MODEL_DIR --out REPORT.json

Run once per dev run (block P, seed 11), on Core2 (CPU only) over the published run, with a Python that has
transformers (the serve env); transformers is imported only to detokenize, everything else is the standard library.
It reads run.json, cells.jsonl and cells/<cell_id>/<attempt_id>/ (attempt_start.json, result.json, events.jsonl) and
writes only the report.

Only each cell's TERMINAL attempt is checked: D11.3, as absent_edges/probe/score.py resolves it, the first attempt in
ordinal order with a completed model exit. An InfraInterrupted attempt can carry delivered true and parsed actions at
call_index 7; it is never counted, so a retried cell counts once.

The protocol's go rule, as the A1 review (finding 3) spelled it out for the records:
- At least 90% of the final responses that received the notice (the terminal attempts with a model_exchange at
  call_index 7) are valid tool calls: a parsed_actions event, not a format_error event, at call_index 7.
- The notice is recorded in every receipt (the CHECKS below):
  - run.json episode_config is the policy's exact record;
  - in every terminal result.json, exposure.final_turn_notice == accounting.final_turn_notice, with the policy and
    the sha256 of its text and suffix;
  - appended is true iff a final_turn_notice event exists, and delivered is true iff a model_exchange exists at
    call_index 7;
  - for each delivered 8th request, the served prompt (model_exchange.response.prompt_token_ids, detokenized with the
    model's tokenizer) holds "[harness notice] " + TEXT exactly once, at the end of the last tool response or of the
    format-error turn (REMINDER_FEASIBILITY.md section 2.6). This is the only check on the served stack.
- The run is the protocol's and is finished: every dispatched cell is block P at seed 11, every block-P seed-11 cell
  is terminal (completed, or missing after 3 infrastructure attempts), and the records are bound to each other.
Also reported, not gated: the count of notices with appended_to "format_error" (the reading of REMINDER_FEASIBILITY.md
section 2.5 that the A1 review, finding 2, asks to report per policy), the terminal exits, the cells missing after 3
infrastructure attempts and the infrastructure attempts by kind.

Follow-up A4 (protocols/EXPA_A4_DIRECTIVE.md): --policy also takes deadline_info and submit_only (the two halves of
the deadline text; their run.json records name the A4 protocol), --seed 23 checks a block-P seed-23 run instead of
seed 11, and --conditions C0,C1 restricts the registered cells (selection and run_complete) to those conditions. The
go rule and every other check are unchanged; the defaults (seed 11, every condition) are the A1 check.

Exit status: 0 go (the go rule and every check pass); 1 no-go (the report lists each failed check with examples);
2 usage error, unreadable run records or no tokenizer (no report is written).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import re
import sys

SCHEMA = "expA-finalturn-devcheck-v1"
PROTOCOL = "protocols/EXPA_FINALTURN.md"
# The frozen notice (protocols/EXPA_FINALTURN.md section 2; absent_edges/probe/harness.py FINAL_TURN_*). Declared here
# again rather than imported, so that the check does not share code with what it audits; the tests pin these values
# to the protocol and to the harness.
PREFIX = "\n\n[harness notice] "
MARKER = "[harness notice]"
TEXTS = {"neutral": "Your previous tool call was processed normally.",
         "deadline": "This is your last response. Call submit_answer now.",
         "deadline_info": "This is your last response.",          # A4 (protocols/EXPA_A4_DIRECTIVE.md)
         "submit_only": "Call submit_answer now."}                # A4
A4_PROTOCOL = "protocols/EXPA_A4_DIRECTIVE.md"
PROTOCOLS = {"neutral": PROTOCOL, "deadline": PROTOCOL, "deadline_info": A4_PROTOCOL, "submit_only": A4_PROTOCOL}
# EpisodeConfig.record() of the prereg harness; a notice policy adds final_turn_policy and final_turn_notice.
FROZEN_CONFIG = {"max_model_calls": 8, "context_budget": 16384, "max_response_tokens": 4096,
                 "min_response_tokens": 512, "temperature": 0.2, "max_consecutive_format_errors": 3}
FINAL_CALL = FROZEN_CONFIG["max_model_calls"] - 1        # the call_index of the 8th (last) response
APPENDED_TO_RULE = ("the content of the last message before that call: the last model-visible tool result, "
                    "or the format-error message after a format error; never sent after a submission")
RECEIPT_FIELDS = ("policy", "text_sha256", "suffix_sha256", "call_index", "appended", "sent", "delivered",
                  "appended_to", "tool_call_id", "prompt_tokens", "max_tokens")
APPENDED_TO = ("tool_result", "format_error")
FORMAT_ERROR_START = "Format error: "                     # harness.FORMAT_ERROR
COMPLETED_EXITS = ("Submitted", "NoAnswer", "ContextLimit", "FormatExhausted")    # D11.3 model exits
INFRA_RETRY_CAP = 3                                       # D11.3
TERMINAL_STATES = ("completed", "missing_infrastructure")
GO_RULE = (9, 10)                                         # valid / noticed >= 9/10, compared exactly
BLOCK, SEED = "P", 11                                     # the protocol's cells
SEEDS = (11, 23)                                          # --seed: A1 runs seed 11; A4 also seed 23
CONDITIONS = ("C0", "C1")                                 # --conditions: A4 may restrict block P to C0, C1
# How the frozen Gemma 4 chat template (sha256 36e3a42e..., REMINDER_FEASIBILITY.md section 2.6) renders the two
# places the notice can end: a string tool result is <|tool_response>response:NAME{value:<|"|>CONTENT<|"|>}
# <tool_response|> (the model turn's text, if any, follows its tool responses), and a user message is
# <|turn>user\nCONTENT<turn|> (content trimmed).
TOOL_RESPONSE_OPEN, TOOL_RESPONSE_CLOSE = "<|tool_response>", "<tool_response|>"
STRING_VALUE_END = '<|"|>}'
USER_TURN_OPEN, TURN_CLOSE = "<|turn>user\n", "<turn|>"
EXAMPLES = 20                                             # failing attempts listed per check
_ORDINAL = re.compile(r"\.a([0-9]+)$")                    # score.py: <cell_id>.aNNN

CHECKS = {
    "run_json_episode_config": "run.json episode_config is the policy's exact record (the six prereg fields, "
                               "final_turn_policy and final_turn_notice)",
    "run_json_cells": "cells.jsonl is the file run.json binds (cells_sha256)",
    "selection": "every dispatched cell is a block-P seed-11 cell of cells.jsonl",
    "run_complete": "every block-P seed-11 cell of cells.jsonl is terminal in the D11.3 sense (a completed attempt, "
                    "or missing after 3 infrastructure attempts)",
    "records": "each terminal attempt: its directory names its cell and attempt, result.json binds attempt_start.json "
               "and events.jsonl (sha256 and rows), its run_id is run.json's, and no attempt follows it",
    "config_receipts": "result.json config and the episode_setup event's config are the policy's record",
    "notice_receipts": "result.json exposure.final_turn_notice equals accounting.final_turn_notice and the "
                       "episode_finished event's, has exactly the receipt fields, the policy, the sha256 of its text "
                       "and suffix and call_index 7, and appended_to is set exactly when appended",
    "appended_iff_event": "appended is true iff exactly one final_turn_notice event exists; the event agrees with the "
                          "receipt and comes right before the call-7 context_accounting",
    "delivered_iff_exchange": "delivered is true iff exactly one model_exchange exists at call_index 7; sent equals "
                              "delivered (a sent, undelivered request is InfraInterrupted, never terminal); delivered "
                              "implies appended",
    "receipt_accounting": "prompt_tokens is the call-7 preflight count when appended (else null), max_tokens the "
                          "call-7 max_tokens when sent (else null); a notice appended but not sent ends the episode "
                          "as ContextLimit at call_index 7",
    "request_bytes": "the delivered 8th request: its last message is the noticed one (a tool message with the "
                     "receipt's tool_call_id, or the format-error user message after an assistant message), its "
                     "content ends with the suffix and matches the event's sha256s, and '[harness notice]' occurs "
                     "once in the request",
    "rendering": "the detokenized served prompt of each delivered 8th request (response.prompt_token_ids) has "
                 "'[harness notice] ' + TEXT exactly once, at the end of the last tool response or of the "
                 "format-error user turn",
    "final_response_recorded": "each delivered final response has exactly one parsed_actions or format_error event "
                               "at call_index 7",
}


class InputError(Exception):
    """The run cannot be checked (exit 2): a usage error, an unreadable record or no tokenizer."""


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_text(text):
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def loads(data, where):
    """JSON with duplicate keys and non-finite numbers refused (as score.strict_loads)."""
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise InputError(f"{where}: duplicate JSON key {key!r}")
            out[key] = value
        return out

    def constant(name):
        raise InputError(f"{where}: non-finite number {name}")
    try:
        return json.loads(data, object_pairs_hook=pairs, parse_constant=constant)
    except ValueError as error:          # JSONDecodeError and UnicodeDecodeError
        raise InputError(f"{where}: not valid JSON ({error})") from error


def read_json(path, where):
    try:
        data = path.read_bytes()
    except OSError as error:
        raise InputError(f"{where}: {error}") from error
    return loads(data, where), data


def expected_episode_config(policy):
    """EpisodeConfig(final_turn_policy=policy).record() of the release, rebuilt from the protocol's values."""
    text = TEXTS[policy]
    return dict(FROZEN_CONFIG, final_turn_policy=policy, final_turn_notice={
        "protocol": PROTOCOLS[policy], "channel": "in_band", "prefix": PREFIX, "text": text,
        "text_sha256": sha256_text(text), "suffix_sha256": sha256_text(PREFIX + text), "call_index": FINAL_CALL,
        "appended_to": APPENDED_TO_RULE})


def cell_block_seed(cell_id):
    parts = cell_id.split("|") if isinstance(cell_id, str) else []
    if len(parts) != 5 or not re.fullmatch(r"s[0-9]+", parts[4]):
        return None, None
    return parts[0], int(parts[4][1:])


def cell_condition(cell_id):
    parts = cell_id.split("|") if isinstance(cell_id, str) else []
    return parts[3] if len(parts) == 5 else None


def check_descriptions(seed=SEED, conditions=None):
    """CHECKS as this check states them: the A1 text for seed 11 and every condition, else the selection named."""
    if seed == SEED and conditions is None:
        return dict(CHECKS)
    cells = f"block-P seed-{seed}" + ("" if conditions is None else f" {'/'.join(sorted(conditions))}")
    return {name: text.replace("block-P seed-11", cells) for name, text in CHECKS.items()}


def ordinal(attempt_id, start):
    match = _ORDINAL.search(attempt_id)
    if match:
        return int(match.group(1))
    if isinstance(start, dict) and type(start.get("ordinal")) is int and start["ordinal"] >= 0:
        return start["ordinal"]
    return None


def load_attempts(run_dir):
    """{cell directory: [attempt, ...]} in D11.3 order (score.scan_run, cells/<cell_id>/<attempt_id>/ layout), and
    the attempt directories with neither a start record nor a result (created, never dispatched)."""
    root = run_dir / "cells"
    if not root.is_dir():
        raise InputError(f"{root} is not a directory")
    found, empty = {}, []
    for cell_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.is_symlink()):
        for directory in sorted(p for p in cell_dir.iterdir() if p.is_dir() and not p.is_symlink()):
            where = f"cells/{cell_dir.name}/{directory.name}"
            start_path, result_path = directory / "attempt_start.json", directory / "result.json"
            if not start_path.is_file() and not result_path.is_file():
                empty.append(where)
                continue
            start = start_sha256 = result = None
            if start_path.is_file():
                start, raw = read_json(start_path, where + "/attempt_start.json")
                start_sha256 = sha256_bytes(raw)
            if result_path.is_file():
                result, _ = read_json(result_path, where + "/result.json")
                if not isinstance(result, dict):
                    raise InputError(f"{where}/result.json is not an object")
            found.setdefault(cell_dir.name, []).append({
                "where": where, "directory": directory, "attempt_id": directory.name, "start": start,
                "start_sha256": start_sha256, "result": result, "ordinal": ordinal(directory.name, start)})
    for attempts in found.values():
        attempts.sort(key=lambda a: (a["ordinal"] if a["ordinal"] is not None else 1 << 30, a["attempt_id"]))
    return found, empty


def resolve(attempts):
    """D11.3, as score.resolve_cell: (state, the terminal attempt or None, earlier infrastructure attempts, later
    attempts)."""
    for position, attempt in enumerate(attempts):
        result = attempt["result"]
        if result is not None and result.get("exit_status") in COMPLETED_EXITS:
            return "completed", attempt, attempts[:position], attempts[position + 1:]
    if not attempts:
        return "not_dispatched", None, [], []
    return ("missing_infrastructure" if len(attempts) >= INFRA_RETRY_CAP else "infra_pending"), None, attempts, []


def notice_placement(text, needle, appended_to):
    """None when ``needle`` ("[harness notice] " + TEXT) occurs exactly once in the detokenized prompt ``text``, at the
    end of the last tool response (appended_to tool_result) or of the last user turn, which must be the format-error
    message (format_error); otherwise the reason."""
    count, markers = text.count(needle), text.count(MARKER)
    if count != 1 or markers != 1:
        return f"the notice occurs {count} time(s), {MARKER!r} {markers} time(s)"
    start = text.index(needle)
    end = start + len(needle)
    if appended_to == "tool_result":
        if not text.startswith(STRING_VALUE_END + TOOL_RESPONSE_CLOSE, end):
            return "the notice is not the end of a tool response's content"
        if text.rfind(TOOL_RESPONSE_OPEN, 0, start) <= text.rfind(TOOL_RESPONSE_CLOSE, 0, start):
            return "the notice is not inside a tool response"
        if text.rfind(TOOL_RESPONSE_CLOSE) != end + len(STRING_VALUE_END):
            return "a later tool response follows the noticed one"
        return None
    if appended_to == "format_error":
        if not text.startswith(TURN_CLOSE, end):
            return "the notice is not the end of a turn"
        turn = text.rfind(USER_TURN_OPEN, 0, start)
        if turn < 0 or TURN_CLOSE in text[turn:start]:
            return "the notice is not inside a user turn"
        if not text.startswith(USER_TURN_OPEN + FORMAT_ERROR_START, turn):
            return "the noticed user turn is not the format-error message"
        if USER_TURN_OPEN in text[end:]:
            return "a later user turn follows the noticed one"
        return None
    return f"appended_to is {appended_to!r}"


def _around(text, needle, width=160):
    """A bounded excerpt around the notice (or the prompt's end), for a failure example."""
    index = text.find(needle)
    if index < 0:
        return text[-2 * width:]
    return text[max(0, index - width):index + len(needle) + width]


class Checks:
    def __init__(self, descriptions=None):
        self.results = {name: {"description": description, "checked": 0, "failed": 0, "examples": []}
                        for name, description in (descriptions or CHECKS).items()}

    def check(self, name, problems, where):
        """Record one checked unit; ``problems`` is a list of reasons (empty: passed)."""
        entry = self.results[name]
        entry["checked"] += 1
        if problems:
            entry["failed"] += 1
            if len(entry["examples"]) < EXAMPLES:
                entry["examples"].append({"where": where, "problems": problems})
        return not problems

    def report(self):
        return {name: dict(entry, passed=entry["failed"] == 0) for name, entry in self.results.items()}

    def failed(self):
        return [name for name, entry in self.results.items() if entry["failed"]]


def _events(rows):
    by_kind = collections.defaultdict(list)
    for index, row in enumerate(rows):
        if isinstance(row, dict):
            by_kind[row.get("kind")].append((index, row.get("value")))
    return by_kind


def _at(by_kind, kind, call_index=FINAL_CALL):
    return [value for _, value in by_kind.get(kind, ()) if isinstance(value, dict)
            and value.get("call_index") == call_index]


class DevCheck:
    # The call_index receipts() expects: the prereg 8th response. A subclass may set it per run (the A3 check of an A6
    # run, protocols/EXPA_A6_BUDGET16.md, reads it from run.json's max_model_calls); this check never changes it.
    final_call = FINAL_CALL

    def __init__(self, run_dir, policy, detokenize, *, seed=SEED, conditions=None):
        if policy not in TEXTS:
            raise InputError(f"--policy must be one of {', '.join(TEXTS)}")
        if seed not in SEEDS:
            raise InputError(f"--seed must be one of {', '.join(map(str, SEEDS))}")
        if conditions is not None and (not conditions or not set(conditions) <= set(CONDITIONS)):
            raise InputError(f"--conditions must be taken from {','.join(CONDITIONS)}")
        self.run_dir, self.policy, self.detokenize = Path(run_dir), policy, detokenize
        self.seed = seed
        self.conditions = None if conditions is None else frozenset(conditions)
        self.text = TEXTS[policy]
        self.suffix = PREFIX + self.text
        self.needle = self.suffix.lstrip("\n")
        self.expected = expected_episode_config(policy)
        self.checks = Checks(check_descriptions(seed, self.conditions))
        self.go = {"noticed": 0, "valid": 0, "format_errors": 0}
        self.notice = collections.Counter()
        self.lists = collections.defaultdict(list)
        self.exits = collections.Counter()
        self.not_reached = collections.Counter()
        self.samples = {}

    def registered_cell(self, cell_id):
        """A cell of the run's registered selection: block P at the seed (and one of the conditions, when given)."""
        return cell_block_seed(cell_id) == (BLOCK, self.seed) and (
            self.conditions is None or cell_condition(cell_id) in self.conditions)

    # -- the run ---------------------------------------------------------------------------------------------------

    def run(self, tokenizer=None):
        if not self.run_dir.is_dir():
            raise InputError(f"run directory not found: {self.run_dir}")
        run, run_raw = read_json(self.run_dir / "run.json", "run.json")
        if not isinstance(run, dict):
            raise InputError("run.json is not an object")
        try:
            cells_raw = (self.run_dir / "cells.jsonl").read_bytes()
        except OSError as error:
            raise InputError(f"cells.jsonl: {error}") from error
        design = [loads(line, f"cells.jsonl line {number}")
                  for number, line in enumerate(cells_raw.splitlines(), 1) if line.strip()]
        design_ids = {cell.get("cell_id") for cell in design if isinstance(cell, dict)}
        registered = {cell for cell in design_ids if self.registered_cell(cell)}
        self.run_json = run
        config = run.get("episode_config")
        self.checks.check("run_json_episode_config", [] if config == self.expected else [
            f"episode_config {json.dumps(config, sort_keys=True)[:600]}"], "run.json")
        self.checks.check("run_json_cells", [] if sha256_bytes(cells_raw) == run.get("cells_sha256") else [
            "sha256(cells.jsonl) is not run.json cells_sha256"], "cells.jsonl")

        attempts, empty = load_attempts(self.run_dir)
        states = collections.Counter()
        infra_kinds = collections.Counter()
        resolved = {}
        selection = f"block-{BLOCK} seed-{self.seed}" + (
            "" if self.conditions is None else " " + "/".join(sorted(self.conditions)))
        for cell, cell_attempts in attempts.items():
            self.checks.check("selection", [] if cell in registered else [
                f"cell {cell} is not a {selection} cell of cells.jsonl"], f"cells/{cell}")
            resolved[cell] = resolve(cell_attempts)
        for cell in sorted(registered | set(resolved)):
            state, terminal, infra, later = resolved.get(cell, ("not_dispatched", None, [], []))
            states[state] += 1
            for attempt in infra:
                result = attempt["result"]
                record = result.get("infra") if isinstance(result, dict) else None
                kind = record.get("kind") if isinstance(record, dict) else None
                infra_kinds[str(kind) if result is not None else "start_record_only"] += 1
            if cell in registered:
                self.checks.check("run_complete", [] if state in TERMINAL_STATES else [f"cell state {state}"],
                                  f"cells/{cell}")
            if terminal is not None:
                self.attempt(cell, terminal, later)

        go = self.go
        share = go["valid"] / go["noticed"] if go["noticed"] else None
        rule_passed = go["noticed"] > 0 and GO_RULE[1] * go["valid"] >= GO_RULE[0] * go["noticed"]
        checks = self.checks.report()
        failed = self.checks.failed()
        cells_in_design = ("block_p_seed_11_in_design" if self.seed == SEED and self.conditions is None else
                           f"block_p_seed_{self.seed}_selection_in_design")
        report = {
            "schema": SCHEMA, "protocol": PROTOCOLS[self.policy], "policy": self.policy, "text": self.text,
            "text_sha256": sha256_text(self.text), "suffix_sha256": sha256_text(self.suffix),
            "go": rule_passed and not failed, "failed_checks": failed,
            "go_rule": {"rule": "at least 90% of the final responses that received the notice (terminal attempts with "
                                "a model_exchange at call_index 7) are valid tool calls (parsed_actions, not "
                                "format_error, at call_index 7)",
                        "noticed_final_responses": go["noticed"], "valid_tool_calls": go["valid"],
                        "format_errors": go["format_errors"], "share": share, "threshold": GO_RULE[0] / GO_RULE[1],
                        "passed": rule_passed},
            "checks": checks,
            "notice": {"terminal_attempts": sum(self.exits.values()), "appended": self.notice["appended"],
                       "sent": self.notice["sent"], "delivered": self.notice["delivered"],
                       "appended_to": {kind: self.notice["appended_to:" + kind] for kind in APPENDED_TO},
                       "delivered_by_appended_to": {kind: self.notice["delivered:" + kind] for kind in APPENDED_TO},
                       "format_error_cells": sorted(self.lists["format_error"]),
                       "appended_not_sent_cells": sorted(self.lists["appended_not_sent"]),
                       "not_reached_by_exit": dict(sorted(self.not_reached.items())),
                       "format_error_note": "appended_to format_error: the notice went on the format-error message "
                                           "after a format error at response 7 (REMINDER_FEASIBILITY.md section "
                                           "2.5); reported per policy (A1 review, finding 2), not gated"},
            "cells": {cells_in_design: len(registered), "dispatched": len(attempts),
                      "states": {state: states[state] for state in ("completed", "missing_infrastructure",
                                                                    "infra_pending", "not_dispatched")},
                      "terminal_exits": dict(sorted(self.exits.items())),
                      "infrastructure_attempts_by_kind": dict(sorted(infra_kinds.items())),
                      "empty_attempt_directories": len(empty),
                      "terminal_only": "each cell's terminal attempt (D11.3: the first with a completed model exit); "
                                       "infrastructure attempts are not checked or counted"},
            "rendering_samples": [self.samples[key] for key in sorted(self.samples)],
            "run": {"run_dir": str(self.run_dir), "run_id": run.get("run_id"), "run_json_sha256": sha256_bytes(run_raw),
                    "cells_jsonl_sha256": sha256_bytes(cells_raw), "model": run.get("model")},
            "tokenizer": tokenizer,
            "script": {"path": "scripts/expA_finalturn_devcheck.py", "sha256": sha256_file(__file__)},
        }
        if self.seed != SEED or self.conditions is not None:     # an A4 selection; the A1 report is unchanged
            report["selection"] = {"block": BLOCK, "seed": self.seed,
                                   "conditions": None if self.conditions is None else sorted(self.conditions)}
        return report

    # -- one terminal attempt ----------------------------------------------------------------------------------------

    def attempt(self, cell, attempt, later):
        where, result = attempt["where"], attempt["result"]
        self.exits[result.get("exit_status")] += 1
        rows = self.records(cell, attempt, later)
        by_kind = _events(rows)
        receipt = self.receipts(where, result, by_kind)
        notices = [value for _, value in by_kind.get("final_turn_notice", ())]
        exchanges = _at(by_kind, "model_exchange")
        appended, sent, delivered = (receipt.get(flag) is True for flag in ("appended", "sent", "delivered"))
        self.notice.update({"appended": appended, "sent": sent, "delivered": delivered})
        if appended:
            self.notice["appended_to:" + str(receipt.get("appended_to"))] += 1
            if receipt.get("appended_to") == "format_error":
                self.lists["format_error"].append(cell)
            if not sent:
                self.lists["appended_not_sent"].append(cell)
        else:
            self.not_reached[result.get("exit_status")] += 1
        if delivered:
            self.notice["delivered:" + str(receipt.get("appended_to"))] += 1

        # appended iff a final_turn_notice event exists, and the event agrees with the receipt
        problems = []
        if appended != (len(notices) == 1) or len(notices) > 1:
            problems.append(f"appended {receipt.get('appended')!r} with {len(notices)} final_turn_notice event(s)")
        if len(notices) == 1 and isinstance(notices[0], dict):
            event = notices[0]
            expected = {"call_index": FINAL_CALL, "policy": self.policy, "appended_to": receipt.get("appended_to"),
                        "tool_call_id": receipt.get("tool_call_id"), "suffix": self.suffix,
                        "suffix_sha256": sha256_text(self.suffix), "text_sha256": sha256_text(self.text)}
            problems += [f"final_turn_notice event {key} {event.get(key)!r}, expected {value!r}"
                         for key, value in expected.items() if event.get(key) != value]
            index = by_kind["final_turn_notice"][0][0]
            following = rows[index + 1] if index + 1 < len(rows) else None
            if not (isinstance(following, dict) and following.get("kind") == "context_accounting"
                    and isinstance(following.get("value"), dict)
                    and following["value"].get("call_index") == FINAL_CALL):
                problems.append("the final_turn_notice event is not followed by the call-7 context_accounting")
        self.checks.check("appended_iff_event", problems, where)

        # delivered iff a model_exchange exists at call_index 7
        problems = []
        if delivered != (len(exchanges) == 1) or len(exchanges) > 1:
            problems.append(f"delivered {receipt.get('delivered')!r} with {len(exchanges)} model_exchange(s) at "
                            f"call_index {FINAL_CALL}")
        if sent != delivered:
            problems.append(f"sent {receipt.get('sent')!r} but delivered {receipt.get('delivered')!r}")
        if delivered and not appended:
            problems.append("delivered without appended")
        self.checks.check("delivered_iff_exchange", problems, where)

        # the accounting fields of the receipt
        problems = []
        preflight = _at(by_kind, "context_accounting")
        if appended:
            if len(preflight) != 1 or receipt.get("prompt_tokens") != preflight[0].get("count"):
                problems.append(f"prompt_tokens {receipt.get('prompt_tokens')!r} is not the call-7 preflight count")
        elif receipt.get("prompt_tokens") is not None:
            problems.append("prompt_tokens set without a notice")
        if sent:
            if len(exchanges) != 1 or receipt.get("max_tokens") != exchanges[0].get("max_tokens"):
                problems.append(f"max_tokens {receipt.get('max_tokens')!r} is not the call-7 max_tokens")
        elif receipt.get("max_tokens") is not None:
            problems.append("max_tokens set although nothing was sent")
        if appended and not sent and (result.get("exit_status") != "ContextLimit"
                                      or len(_at(by_kind, "context_limit")) != 1):
            problems.append(f"a notice appended but not sent, exit {result.get('exit_status')!r} without a "
                            "context_limit at call_index 7")
        self.checks.check("receipt_accounting", problems, where)

        if len(exchanges) != 1:
            return
        exchange = exchanges[0]
        event = notices[0] if len(notices) == 1 and isinstance(notices[0], dict) else {}
        self.request(where, exchange, receipt, event)
        self.rendering(where, cell, exchange, receipt)
        parsed, errors = _at(by_kind, "parsed_actions"), _at(by_kind, "format_error")
        self.checks.check("final_response_recorded", [] if len(parsed) + len(errors) == 1 else [
            f"{len(parsed)} parsed_actions and {len(errors)} format_error event(s) at call_index {FINAL_CALL}"], where)
        self.go["noticed"] += 1
        self.go["valid"] += len(parsed) == 1 and not errors
        self.go["format_errors"] += bool(errors)

    def records(self, cell, attempt, later):
        """Bindings of a terminal attempt; returns its events.jsonl rows ([] when it has none)."""
        where, result, problems = attempt["where"], attempt["result"], []
        if result.get("cell_id") != cell or result.get("attempt_id") != attempt["attempt_id"]:
            problems.append(f"result.json names cell {result.get('cell_id')!r} attempt {result.get('attempt_id')!r}")
        start = attempt["start"]
        if isinstance(start, dict) and start.get("cell_id") not in (None, cell):
            problems.append(f"attempt_start.json names cell {start.get('cell_id')!r}")
        context = result.get("run_context")
        recorded = context.get("start_sha256") if isinstance(context, dict) else None
        if recorded is not None and recorded != attempt["start_sha256"]:
            problems.append("result.json run_context.start_sha256 is not the sha256 of attempt_start.json")
        if result.get("run_id") != self.run_json.get("run_id"):
            problems.append(f"run_id {result.get('run_id')!r} is not run.json's")
        if later:
            problems.append(f"{len(later)} attempt(s) after the terminal attempt")
        rows = []
        path = attempt["directory"] / "events.jsonl"
        if not path.is_file():
            problems.append("no events.jsonl")
        else:
            raw = path.read_bytes()
            rows = [loads(line, f"{where}/events.jsonl line {number}")
                    for number, line in enumerate(raw.splitlines(), 1)]
            binding = result.get("events") if isinstance(result.get("events"), dict) else {}
            if binding.get("sha256") != sha256_bytes(raw) or binding.get("records") != len(rows):
                problems.append("events.jsonl is not the file result.json binds (sha256, records)")
            if [row.get("sequence") if isinstance(row, dict) else None for row in rows] != list(range(len(rows))):
                problems.append("events.jsonl sequence numbers are not 0, 1, 2, ...")
        self.checks.check("records", problems, where)
        return rows

    def receipts(self, where, result, by_kind):
        """config_receipts and notice_receipts; returns the receipt (an empty dict when it is unusable)."""
        setups = [value for _, value in by_kind.get("episode_setup", ())]
        problems = []
        if result.get("config") != self.expected:
            problems.append("result.json config is not the policy's record")
        if len(setups) != 1 or not isinstance(setups[0], dict) or setups[0].get("config") != self.expected:
            problems.append("the episode_setup event's config is not the policy's record")
        self.checks.check("config_receipts", problems, where)

        exposure = result.get("exposure") if isinstance(result.get("exposure"), dict) else {}
        accounting = result.get("accounting") if isinstance(result.get("accounting"), dict) else {}
        receipt = exposure.get("final_turn_notice")
        finished = [value for _, value in by_kind.get("episode_finished", ())]
        finished_accounting = finished[0].get("accounting") if len(finished) == 1 and isinstance(finished[0], dict) \
            else None
        problems = []
        if not isinstance(receipt, dict) or set(receipt) != set(RECEIPT_FIELDS):
            problems.append("exposure.final_turn_notice is missing or does not have exactly the receipt fields")
            receipt = receipt if isinstance(receipt, dict) else {}
        if accounting.get("final_turn_notice") != receipt:
            problems.append("accounting.final_turn_notice differs from exposure.final_turn_notice")
        if not isinstance(finished_accounting, dict) or finished_accounting.get("final_turn_notice") != receipt:
            problems.append("the episode_finished event's final_turn_notice differs from the result's")
        expected = {"policy": self.policy, "text_sha256": sha256_text(self.text),
                    "suffix_sha256": sha256_text(self.suffix), "call_index": self.final_call}
        problems += [f"{key} {receipt.get(key)!r}, expected {value!r}" for key, value in expected.items()
                     if receipt and receipt.get(key) != value]
        flags = [flag for flag in ("appended", "sent", "delivered") if type(receipt.get(flag)) is not bool]
        if receipt and flags:
            problems.append(f"{', '.join(flags)} not true or false")
        if receipt and (receipt.get("appended_to") in APPENDED_TO) != (receipt.get("appended") is True):
            problems.append(f"appended_to {receipt.get('appended_to')!r} with appended {receipt.get('appended')!r}")
        self.checks.check("notice_receipts", problems, where)
        return receipt

    def request(self, where, exchange, receipt, event):
        """The exact bytes of the delivered 8th request (model_exchange.request)."""
        request = exchange.get("request") if isinstance(exchange.get("request"), dict) else {}
        messages = request.get("messages")
        problems = []
        if not isinstance(messages, list) or len(messages) < 3 or not all(isinstance(m, dict) for m in messages):
            self.checks.check("request_bytes", ["the request has no messages"], where)
            return
        last, appended_to = messages[-1], receipt.get("appended_to")
        if appended_to == "tool_result":
            if last.get("role") != "tool" or last.get("tool_call_id") != receipt.get("tool_call_id"):
                problems.append(f"the last message is {last.get('role')!r} {last.get('tool_call_id')!r}, not the "
                                f"tool result {receipt.get('tool_call_id')!r}")
        elif appended_to == "format_error":
            if last.get("role") != "user" or messages[-2].get("role") != "assistant" \
                    or receipt.get("tool_call_id") is not None:
                problems.append("the last message is not a format-error user message after an assistant message")
            elif not str(last.get("content")).startswith(FORMAT_ERROR_START):
                problems.append("the noticed user message is not the format-error message")
        else:
            problems.append(f"appended_to {appended_to!r}")
        content = last.get("content")
        if not isinstance(content, str) or not content.endswith(self.suffix):
            problems.append("the last message does not end with the suffix")
        else:
            if sha256_text(content) != event.get("content_sha256"):
                problems.append("the last message is not the content the final_turn_notice event binds")
            if sha256_text(content[:-len(self.suffix)]) != event.get("previous_content_sha256"):
                problems.append("the content before the suffix is not the one the final_turn_notice event binds")
        markers = json.dumps(messages, ensure_ascii=False).count(MARKER)
        if markers != 1:
            problems.append(f"{MARKER!r} occurs {markers} time(s) in the request")
        self.checks.check("request_bytes", problems, where)

    def rendering(self, where, cell, exchange, receipt):
        response = exchange.get("response") if isinstance(exchange.get("response"), dict) else {}
        ids = response.get("prompt_token_ids")
        if not isinstance(ids, list) or not ids or any(type(token) is not int or token < 0 for token in ids):
            self.checks.check("rendering", ["model_exchange.response.prompt_token_ids is missing or invalid"], where)
            return
        try:
            text = self.detokenize(ids)
        except Exception as error:     # e.g. ids outside the tokenizer's vocabulary: this prompt cannot be read
            self.checks.check("rendering", [f"detokenizing failed: {type(error).__name__}: {error}"], where)
            return
        if not isinstance(text, str):
            raise InputError("the detokenizer did not return text")
        appended_to = receipt.get("appended_to")
        problem = notice_placement(text, self.needle, appended_to)
        excerpt = _around(text, self.needle)
        self.checks.check("rendering", [] if problem is None else [problem, excerpt], where)
        if problem is None and appended_to not in self.samples:
            self.samples[appended_to] = {"cell": cell, "appended_to": appended_to, "prompt_tokens": len(ids),
                                         "around_notice": excerpt}


def check_run(run_dir, policy, detokenize, *, tokenizer=None, seed=SEED, conditions=None):
    """The report of one dev run (see the module docstring); raises InputError when the run cannot be checked."""
    return DevCheck(run_dir, policy, detokenize, seed=seed, conditions=conditions).run(tokenizer)


def load_detokenizer(directory):
    """(detokenize, record) with the model's own tokenizer files, offline; needs transformers (no torch, no GPU)."""
    directory = Path(directory)
    if not directory.is_dir():
        raise InputError(f"--tokenizer-dir {directory} is not a directory")
    try:
        import transformers
        from transformers import AutoTokenizer
    except ImportError as error:
        raise InputError(f"detokenizing needs transformers (the Core2 serve env has it): {error}") from error
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True)
    except Exception as error:     # any loader failure means the rendering check cannot run
        raise InputError(f"cannot load the tokenizer from {directory}: {error}") from error

    def detokenize(ids):
        # Special tokens kept (the template's markers) and no clean-up: the served prompt's exact text.
        return tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    files = {name: sha256_file(directory / name)
             for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja")
             if (directory / name).is_file()}
    return detokenize, {"directory": str(directory), "files_sha256": files, "class": type(tokenizer).__name__,
                        "transformers": getattr(transformers, "__version__", None),
                        "decode": "skip_special_tokens=False, clean_up_tokenization_spaces=False"}


def write_report(path, report):
    """The report, created exclusively (an existing file is never replaced)."""
    data = (json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def summary(report):
    rule = report["go_rule"]
    share = "n/a" if rule["share"] is None else f"{rule['share']:.1%}"
    return (f"{'GO' if report['go'] else 'NO-GO'}: {report['policy']} run {report['run']['run_id']}: "
            f"{rule['valid_tool_calls']}/{rule['noticed_final_responses']} noticed final responses are valid tool "
            f"calls ({share}; go rule >= 90%: {'met' if rule['passed'] else 'not met'}); notices on a format-error "
            f"message: {report['notice']['appended_to']['format_error']}; failed checks: "
            f"{', '.join(report['failed_checks']) or 'none'}")


def main(argv=None, *, load=load_detokenizer):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", required=True, help="the published dev run (run.json, cells.jsonl, cells/)")
    parser.add_argument("--policy", required=True, choices=sorted(TEXTS), help="the run's final-turn policy")
    parser.add_argument("--seed", type=int, choices=SEEDS, default=SEED,
                        help="the run's block-P seed (default %(default)s; A4 also 23)")
    parser.add_argument("--conditions", default=None,
                        help="comma-separated conditions the run dispatched, from C0,C1 (A4; default: all)")
    parser.add_argument("--tokenizer-dir", required=True,
                        help="the served model's directory (tokenizer files), e.g. "
                             "/mnt/kldata/absent-edges-expA-public-20261101/models/gemma4-31b-bf16-518276fb")
    parser.add_argument("--out", required=True, help="the JSON report to create (never overwritten)")
    args = parser.parse_args(argv)
    out = Path(args.out)
    try:
        conditions = None
        if args.conditions is not None:
            conditions = {value.strip() for value in args.conditions.split(",") if value.strip()}
            if not conditions or not conditions <= set(CONDITIONS):
                raise InputError(f"--conditions must be taken from {','.join(CONDITIONS)}, not {args.conditions!r}")
        if out.exists() or out.is_symlink():
            raise InputError(f"refusing to overwrite {out}")
        detokenize, tokenizer = load(args.tokenizer_dir)
        report = check_run(args.run_dir, args.policy, detokenize, tokenizer=tokenizer, seed=args.seed,
                           conditions=conditions)
        write_report(out, report)
    except (InputError, OSError) as error:
        print(f"expA_finalturn_devcheck: ERROR: {error}", file=sys.stderr)
        return 2
    print(summary(report))
    print(f"report: {out}")
    return 0 if report["go"] else 1


if __name__ == "__main__":
    sys.exit(main())
