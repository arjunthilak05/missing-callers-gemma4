#!/usr/bin/env python3
"""Exp A follow-up A7: the dev-family check of an impact-framing run (protocols/EXPA_A7_IMPACT.md section 2).

    python -B scripts/expA_a7_devcheck.py --run-dir RUN_DIR --out REPORT.json

Run once on the published dev run (the dev dispatch, launched with --task-framing impact --blocks P,S-VC --seeds 11
--conditions C0,C1), before the lock and before any main-family cell. Standard library only: it reads run.json,
cells.jsonl and cells/<cell_id>/<attempt_id>/ (attempt_start.json, result.json, events.jsonl) and writes only the
report. The frozen values (the six prereg config fields and the impact template) are declared again here rather than
imported, so that the check shares no code with the harness it audits; tests/test_expA_a7_impact.py pins them to it.

Go, every check below passed (CHECKS spells each out):
- run.json and every terminal attempt (result.json config and the episode_setup event's config) record exactly the
  impact config: the six prereg fields, task_framing impact and task_framing_rules naming protocols/EXPA_A7_IMPACT.md,
  the exact template text and its sha256 (and nothing else: no final-turn notice, ledger or review);
- the delivered first user message of every terminal attempt (the user message of the call_index 0 request, or the
  episode_setup user message when no request was made) starts with the exact impact paragraph rendered for the cell
  (its qualname, path, line and package directories), equals the episode_setup message and the result's
  user_sha256, and never contains the default paragraph;
- at least 95% of the completed episodes (terminal attempts) end Submitted or NoAnswer without a harness error
  (result infra null), compared exactly;
- the dispatched cells are exactly the A7 cells of cells.jsonl at seed 11, P|<item>|none|C0|s11,
  P|<item>|omit_node|C1|s11 and S-VC|<item>|vcomplete|C1|s11: none outside them, every one dispatched and terminal in
  the D11.3 sense (a completed attempt, or missing after the infrastructure attempt cap);
- the records of each terminal attempt are bound: its directory names its cell and attempt, result.json names the
  run, cell and attempt, binds attempt_start.json and events.jsonl (sha256 and rows), and no attempt follows it.
Reported, not gated: exits and cell states per arm, infrastructure attempts, cells missing for infrastructure.

Exit status: 0 GO; 1 NO-GO (the report lists each failed check with examples); 2 usage error or unreadable run records
(no report is written). PRESPECIFIED with protocols/EXPA_A7_IMPACT.md; ANALYSIS RECORD.
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

SCHEMA = "expA-a7-devcheck-v1"
PROTOCOL = "protocols/EXPA_A7_IMPACT.md"
LABEL = ("PRESPECIFIED (protocols/EXPA_A7_IMPACT.md, written before any A7 cell); ANALYSIS RECORD"
         "")
IMPACT_TEMPLATE = (
    "Task: the signature of the function `{qualname}` (defined in `{path}`, line {line}) is about to change. Find "
    "every place in the library source that must be updated: every place where it is invoked, either called directly "
    "or passed as a callable that gets invoked, from inside another function or method. Consider only files under "
    "{package_dirs}. Ignore tests, docs, examples and module-level code.")
DEFAULT_OPENING = "Task: find every place in the library source where the function `"
REPLACES = ("the task paragraph (USER_TASK_TEMPLATE), with the same placeholders; the system prompt, answer block, "
            "recommendation and submit sentences, budgets and tools are unchanged")
FROZEN_CONFIG = {"max_model_calls": 8, "context_budget": 16384, "max_response_tokens": 4096,
                 "min_response_tokens": 512, "temperature": 0.2, "max_consecutive_format_errors": 3}
SEED = 11
ARMS = {("P", "none", "C0"): "none", ("P", "omit_node", "C1"): "incomplete", ("S-VC", "vcomplete", "C1"): "complete"}
COMPLETED_EXITS = ("Submitted", "NoAnswer", "ContextLimit", "FormatExhausted")
GO_EXITS = ("Submitted", "NoAnswer")
GO_SHARE = (95, 100)                    # ok / completed >= 95/100, compared exactly
INFRA_CAP_DEFAULT = 3
EXAMPLES = 10
_ORDINAL = re.compile(r"\.a([0-9]{3,})$")

CHECKS = {
    "run_json_episode_config": "run.json episode_config is exactly the impact record (the six prereg fields, "
                               "task_framing impact and task_framing_rules: protocol, framing, text, text_sha256, "
                               "replaces)",
    "run_json_cells": "cells.jsonl is the file run.json binds (cells_sha256)",
    "selection": "the dispatched cells are exactly the A7 cells of cells.jsonl at seed 11 (P none C0, P omit_node C1, "
                 "S-VC vcomplete C1): none outside them and every one dispatched",
    "run_complete": "every A7 cell of cells.jsonl is terminal in the D11.3 sense (a completed attempt, or missing "
                    "after the infrastructure attempt cap)",
    "records": "each terminal attempt: its directory names its cell and attempt; result.json names the run, cell and "
               "attempt, binds attempt_start.json and events.jsonl (sha256 and rows); no attempt follows it",
    "config_receipts": "each terminal attempt's result.json config and episode_setup event config are the impact "
                       "record (task_framing impact, the exact text and its sha256)",
    "first_user_message": "each terminal attempt's delivered first user message starts with the exact impact paragraph "
                          "rendered for its cell, equals the episode_setup message and the result's user_sha256, and "
                          "does not contain the default paragraph",
    "go_rule": "at least 95% of the completed episodes end Submitted or NoAnswer without a harness error",
}


class InputError(Exception):
    """The run cannot be checked (exit 2)."""


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_text(text):
    return sha256_bytes(text.encode("utf-8"))


def expected_episode_config():
    """EpisodeConfig(task_framing="impact").record() of the release, rebuilt from the protocol's values."""
    return dict(FROZEN_CONFIG, task_framing="impact", task_framing_rules={
        "protocol": PROTOCOL, "framing": "impact", "text": IMPACT_TEMPLATE, "text_sha256": sha256_text(IMPACT_TEMPLATE),
        "replaces": REPLACES})


def format_package_dirs(package_dirs):
    """The harness's rendering of package directories (sorted, unique; "" or "." as `./`, a .py file as itself)."""
    shown = []
    for directory in sorted({"" if d in ("", ".") else d.replace("\\", "/").rstrip("/") for d in package_dirs}):
        if directory == "":
            shown.append("`./`")
        elif directory.endswith(".py"):
            shown.append(f"`{directory}`")
        else:
            shown.append(f"`{directory}/`")
    return ", ".join(shown)


def impact_paragraph(target, line, package_dirs):
    """The impact task paragraph of one cell (target "path::qualname")."""
    path, qualname = target.split("::")
    return IMPACT_TEMPLATE.format(qualname=qualname, path=path, line=line,
                                  package_dirs=format_package_dirs(package_dirs))


def loads(data, where):
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
    except ValueError as error:
        raise InputError(f"{where}: not valid JSON ({error})") from error


def read_json(path, where):
    try:
        data = Path(path).read_bytes()
    except OSError as error:
        raise InputError(f"{where}: {error}") from error
    return loads(data, where), data


def cell_arm(cell_id):
    """The A7 arm name of a cell id at seed 11, or None."""
    parts = cell_id.split("|") if isinstance(cell_id, str) else []
    if len(parts) != 5 or parts[4] != f"s{SEED}":
        return None
    return ARMS.get((parts[0], parts[2], parts[3]))


def _ordinal(attempt_id, start):
    match = _ORDINAL.search(attempt_id)
    if match:
        return int(match.group(1))
    if isinstance(start, dict) and type(start.get("ordinal")) is int:
        return start["ordinal"]
    return None


def load_attempts(run_dir, cell_name):
    """{cell_id: [attempt, ...]} in ordinal order (cells/<cell_id>/<attempt_id>/), and empty attempt directories."""
    root = run_dir / "cells"
    if not root.is_dir():
        raise InputError(f"{root} is not a directory")
    found, empty = {}, []
    for cell_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.is_symlink()):
        cell = cell_name(cell_dir.name)
        for directory in sorted(p for p in cell_dir.iterdir() if p.is_dir() and not p.is_symlink()):
            attempt_id = cell_name(directory.name)
            where = f"cells/{cell}/{attempt_id}"
            start_path, result_path = directory / "attempt_start.json", directory / "result.json"
            if not start_path.is_file() and not result_path.is_file():
                empty.append(where)
                continue
            start = start_raw = result = None
            if start_path.is_file():
                start, start_raw = read_json(start_path, where + "/attempt_start.json")
            if result_path.is_file():
                result, _ = read_json(result_path, where + "/result.json")
                if not isinstance(result, dict):
                    raise InputError(f"{where}/result.json is not an object")
            found.setdefault(cell, []).append({
                "where": where, "directory": directory, "attempt_id": attempt_id, "start": start,
                "start_sha256": None if start_raw is None else sha256_bytes(start_raw), "result": result,
                "ordinal": _ordinal(attempt_id, start)})
    for attempts in found.values():
        attempts.sort(key=lambda a: (a["ordinal"] if a["ordinal"] is not None else 1 << 30, a["attempt_id"]))
    return found, empty


def resolve(attempts, cap):
    """D11.3: (state, the terminal attempt or None, the attempts after it)."""
    for position, attempt in enumerate(attempts):
        result = attempt["result"]
        if result is not None and result.get("exit_status") in COMPLETED_EXITS:
            return "completed", attempt, attempts[position + 1:]
    if not attempts:
        return "not_dispatched", None, []
    return ("missing_infrastructure" if len(attempts) >= cap else "infra_pending"), None, []


class Checks:
    def __init__(self):
        self.problems = {name: [] for name in CHECKS}
        self.counts = collections.Counter()

    def fail(self, name, where, message):
        self.counts[name] += 1
        if len(self.problems[name]) < EXAMPLES:
            self.problems[name].append(f"{where}: {message}")

    def report(self):
        return {name: {"description": CHECKS[name], "passed": not self.counts[name], "failures": self.counts[name],
                       "examples": self.problems[name]} for name in CHECKS}


def _events(directory, where):
    try:
        raw = (directory / "events.jsonl").read_bytes()
    except OSError as error:
        raise InputError(f"{where}/events.jsonl: {error}") from error
    rows = [loads(line, f"{where}/events.jsonl line {number}")
            for number, line in enumerate(raw.splitlines(), 1) if line.strip()]
    return rows, raw


def check_run(run_dir, *, cell_name=None):
    """The A7 dev check of one run directory: the report (a dict)."""
    run_dir = Path(run_dir)
    cell_name = cell_name or (lambda name: name)
    if not run_dir.is_dir():
        raise InputError(f"run directory not found: {run_dir}")
    run, run_raw = read_json(run_dir / "run.json", "run.json")
    if not isinstance(run, dict):
        raise InputError("run.json is not an object")
    try:
        cells_raw = (run_dir / "cells.jsonl").read_bytes()
    except OSError as error:
        raise InputError(f"cells.jsonl: {error}") from error
    design = [loads(line, f"cells.jsonl line {number}") for number, line in enumerate(cells_raw.splitlines(), 1)
              if line.strip()]
    design_ids = {cell.get("cell_id") for cell in design if isinstance(cell, dict)}
    registered = {cell for cell in design_ids if cell_arm(cell) is not None}
    expected = expected_episode_config()
    cap = run.get("infra_attempt_cap", INFRA_CAP_DEFAULT)
    cap = cap if type(cap) is int and cap >= 1 else INFRA_CAP_DEFAULT
    checks = Checks()
    if run.get("episode_config") != expected:
        checks.fail("run_json_episode_config", "run.json",
                    f"episode_config {json.dumps(run.get('episode_config'), sort_keys=True)[:600]}")
    if sha256_bytes(cells_raw) != run.get("cells_sha256"):
        checks.fail("run_json_cells", "cells.jsonl", "sha256(cells.jsonl) is not run.json cells_sha256")
    attempts, empty = load_attempts(run_dir, cell_name)
    for cell in sorted(set(attempts) - registered):
        checks.fail("selection", cell, "dispatched, but not an A7 cell of cells.jsonl at seed 11")
    for cell in sorted(registered - set(attempts)):
        checks.fail("selection", cell, "an A7 cell of cells.jsonl that was never dispatched")
    states = {arm: collections.Counter() for arm in ARMS.values()}
    exits = {arm: collections.Counter() for arm in ARMS.values()}
    go = {"completed": 0, "ok": 0, "harness_errors": 0}
    first_message = {"requested": 0, "not_requested": 0}
    infra_attempts = 0
    for cell in sorted(registered):
        arm = cell_arm(cell)
        cell_attempts = attempts.get(cell, [])
        state, terminal, later = resolve(cell_attempts, cap)
        states[arm][state] += 1
        infra_attempts += sum(1 for a in cell_attempts if a is not terminal and a not in later)
        if state not in ("completed", "missing_infrastructure"):
            checks.fail("run_complete", cell, f"state {state} ({len(cell_attempts)} attempt(s))")
        if terminal is None:
            continue
        result = terminal["result"]
        where = terminal["where"]
        exits[arm][result.get("exit_status")] += 1
        go["completed"] += 1
        if result.get("exit_status") in GO_EXITS and result.get("infra") is None:
            go["ok"] += 1
        elif result.get("infra") is not None:
            go["harness_errors"] += 1
        rows, events_raw = _events(terminal["directory"], where)
        check_records(checks, run, cell, terminal, later, rows, events_raw)
        by_kind = collections.defaultdict(list)
        for row in rows:
            if isinstance(row, dict):
                by_kind[row.get("kind")].append(row.get("value"))
        setup = by_kind["episode_setup"][0] if by_kind["episode_setup"] else None
        if result.get("config") != expected:
            checks.fail("config_receipts", where, "result.json config is not the impact record")
        if not isinstance(setup, dict) or setup.get("config") != expected:
            checks.fail("config_receipts", where, "the episode_setup event's config is not the impact record")
        check_first_message(checks, where, result, setup, by_kind, first_message)
    total, ok = go["completed"], go["ok"]
    passed = total > 0 and ok * GO_SHARE[1] >= GO_SHARE[0] * total
    if not passed:
        checks.fail("go_rule", "run", f"{ok}/{total} completed episodes end Submitted or NoAnswer without a harness "
                                      "error (go rule >= 95%)")
    report_checks = checks.report()
    failed = [name for name, value in report_checks.items() if not value["passed"]]
    return {"schema": SCHEMA, "label": LABEL, "protocol": PROTOCOL,
            "run": {"run_dir": str(run_dir), "run_id": run.get("run_id"), "run_json_sha256": sha256_bytes(run_raw),
                    "cells_sha256": sha256_bytes(cells_raw), "design_cells": len(design_ids),
                    "a7_cells": len(registered), "infra_attempt_cap": cap},
            "expected_episode_config": expected, "impact_text_sha256": sha256_text(IMPACT_TEMPLATE),
            "checks": report_checks, "failed_checks": failed, "go": not failed,
            "go_rule": {"rule": CHECKS["go_rule"] + " (result infra null), compared exactly", "completed": total,
                        "submitted_or_no_answer": ok, "harness_errors": go["harness_errors"],
                        "share": None if total == 0 else ok / total, "passed": passed},
            "first_user_message": first_message,
            "by_arm": {arm: {"cell_states": dict(sorted(states[arm].items())),
                             "exits": dict(sorted(exits[arm].items()))} for arm in ARMS.values()},
            "infra_attempts": infra_attempts, "empty_attempt_directories": empty[:EXAMPLES],
            "empty_attempt_directory_count": len(empty)}


def check_records(checks, run, cell, terminal, later, rows, events_raw):
    where, result, start = terminal["where"], terminal["result"], terminal["start"]
    attempt_id = terminal["attempt_id"]
    if not attempt_id.startswith(cell + ".a"):
        checks.fail("records", where, "the attempt directory does not name its cell")
    if (result.get("run_id"), result.get("cell_id"), result.get("attempt_id")) != (run.get("run_id"), cell,
                                                                                  attempt_id):
        checks.fail("records", where, "result.json does not name the run, the cell and the attempt")
    if not isinstance(start, dict) or start.get("cell_id") != cell:
        checks.fail("records", where, "attempt_start.json is missing or names another cell")
    context = result.get("run_context") if isinstance(result.get("run_context"), dict) else {}
    if terminal["start_sha256"] is None or context.get("start_sha256") != terminal["start_sha256"]:
        checks.fail("records", where, "result.json run_context.start_sha256 is not the sha256 of attempt_start.json")
    events = result.get("events") if isinstance(result.get("events"), dict) else {}
    if events.get("sha256") != sha256_bytes(events_raw) or events.get("records") != len(rows):
        checks.fail("records", where, "result.json events does not bind events.jsonl (sha256 and rows)")
    if later:
        checks.fail("records", where, f"{len(later)} attempt(s) after the terminal attempt")


def check_first_message(checks, where, result, setup, by_kind, counts):
    cell = result.get("cell") if isinstance(result.get("cell"), dict) else {}
    target, line, dirs = cell.get("target", result.get("target")), cell.get("target_line"), cell.get("package_dirs")
    if not (isinstance(target, str) and target.count("::") == 1 and type(line) is int and isinstance(dirs, list)
            and dirs and all(isinstance(d, str) for d in dirs)):
        checks.fail("first_user_message", where, "result.json cell lacks target, target_line or package_dirs")
        return
    paragraph = impact_paragraph(target, line, dirs)
    stated = setup.get("user") if isinstance(setup, dict) else None
    exchange = next((value for value in by_kind["model_exchange"]
                     if isinstance(value, dict) and value.get("call_index") == 0), None)
    if exchange is None:
        counts["not_requested"] += 1
        delivered = stated
    else:
        counts["requested"] += 1
        request = exchange.get("request") if isinstance(exchange.get("request"), dict) else {}
        messages = request.get("messages") if isinstance(request.get("messages"), list) else []
        user = messages[1] if len(messages) > 1 and isinstance(messages[1], dict) else {}
        delivered = user.get("content") if user.get("role") == "user" else None
    if not isinstance(delivered, str):
        checks.fail("first_user_message", where, "no first user message was recorded")
        return
    if not delivered.startswith(paragraph) or delivered.count(paragraph) != 1:
        checks.fail("first_user_message", where, f"does not start with the impact paragraph: {delivered[:200]!r}")
    if DEFAULT_OPENING in delivered:
        checks.fail("first_user_message", where, "contains the default task paragraph")
    if delivered != stated:
        checks.fail("first_user_message", where, "differs from the episode_setup user message")
    prompt = result.get("prompt") if isinstance(result.get("prompt"), dict) else {}
    if prompt.get("user_sha256") != sha256_text(delivered):
        checks.fail("first_user_message", where, "result.json prompt.user_sha256 is not the delivered message's")


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
    return (f"{'GO' if report['go'] else 'NO-GO'}: A7 impact run {report['run']['run_id']}: "
            f"{rule['submitted_or_no_answer']}/{rule['completed']} completed episodes end Submitted or NoAnswer "
            f"without a harness error ({share}; go rule >= 95%: {'met' if rule['passed'] else 'not met'}); "
            f"A7 cells {report['run']['a7_cells']}; failed checks: {', '.join(report['failed_checks']) or 'none'}")


def main(argv=None, *, cell_name=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", required=True, help="the published dev run (run.json, cells.jsonl, cells/)")
    parser.add_argument("--out", required=True, help="the JSON report to create (never overwritten)")
    args = parser.parse_args(argv)
    out = Path(args.out)
    try:
        if out.exists() or out.is_symlink():
            raise InputError(f"refusing to overwrite {out}")
        report = check_run(args.run_dir, cell_name=cell_name)
        write_report(out, report)
    except (InputError, OSError) as error:
        print(f"expA_a7_devcheck: ERROR: {error}", file=sys.stderr)
        return 2
    print(summary(report))
    print(f"report: {out}")
    return 0 if report["go"] else 1


if __name__ == "__main__":
    sys.exit(main())
