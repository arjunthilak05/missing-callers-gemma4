"""ANALYSIS RECORD. script (2026-11-01).

EXPLORATORY: designed after the 31B primary results were seen (2026-11-01 ~06:05 UTC); not preregistered;
descriptive; post-treatment quantities, no causal or mediation claims.

Exp A per-episode timelines: when did the omitted runtime-witnessed caller C* become visible to the agent,
when did it submit, how did the 8-response budget end. Stdlib only; reads the run records READ-ONLY.

    python -B scripts/expA_timeline.py --run-dir <expA>/runs/main-20261101b \\
        --items <expA>/prod-items-0b8ee0b7c/items-main/items.jsonl \\
        --scores <expA>/main-analysis-20261101b/scores.jsonl \\
        --out <expA>/main-analysis-20261101b-timeline --workers 16

Inputs: items.jsonl and scores.jsonl are read once and their sha256 checked against the pinned values
(``--items-sha256`` / ``--scores-sha256`` override them for another run). Each episode's result.json must
have the sha256 the scores row names, and its events.jsonl the sha256 result.json names.

Outputs in --out (refused if present): TIMELINE.json (aggregates; no paths or timestamps), episodes.jsonl
(one row per design cell, sorted by cell_id) and TIMELINE.md (report, with the command line, runtime and
output sha256).

Visibility rules (fixed 2026-11-01T06:55Z, before any outcome split; reports/expA-main-20261101b/timeline/
RULES.md). Not tuned after.
------------------------------------------------------------------------------------------------------
* Response k (1..n) is the k-th model response (``model_exchange`` call_index k-1).
* Model-visible units with a reveal index r: the initial user message (``episode_setup.user``, with any
  answer or verification block), r = 0; every ``model_visible_tool_result``, r = the response number j
  of the tool call it answers (that text is in the context of responses j+1, j+2, ...). Format-error
  feedback, the system message and the model's own text are not units.
* ``first_cstar_visible_response`` (per rule) = the smallest r of a unit satisfying the rule for any site
  span of C* (the item's V entry ``omitted.identity``); null if none. "C* visible before response k" <=>
  first_cstar_visible_response <= k - 1.
* Path match: a path string matches site path P when (backslashes -> /) it equals P or ends with "/P"
  (so ./, /repo/ and other absolute prefixes are allowed). Line match: lineno <= line <= end_lineno.
* STRICT (primary; a LOWER BOUND): (S-text) a ``PATH:LINE`` occurrence (PATH a maximal run of
  [A-Za-z0-9_.+@~/-], LINE digits) in the unit's text matches. The text of a tool result is every string
  value of its decoded JSON; the user message is its raw text plus every string value of each JSON
  object embedded in it. Or (S-struct) a JSON object in the unit has a string ``path`` and an integer
  ``line`` that match (source_reference_search hits, C4a verification hits). It misses output that never
  pairs the path with the line: single-file grep -n / cat -n / nl / sed -n reads, graph answers.
* LOOSE (secondary): STRICT, or for a tool-result unit (r >= 1) either
  (L1) numbered single-file view: bash with output; the harness-recorded argv (all pipeline stages,
  glob-expanded) names exactly one distinct operand ending in ".py" (not starting with "-"), and it
  matches P; the first line-numbering stage is the first stage or is preceded only by plain ``cat``
  stages (no options), where a numbering stage is grep/rg with -n in a short-option cluster or
  --line-number, cat with -n in a cluster or --number (and no -b / --number-nonblank), nl with body
  numbering a (-ba, -b a, --body-numbering=a; no -v/-i), or awk/gawk whose program contains NR; plain nl
  (non-blank lines only) is a numbering stage whose numbers are not file lines, so it never qualifies;
  and an output line matches ``^ *(\\d+)(?:[:\\t -]|$)`` with the number in the span; or
  (L2) name co-occurrence in one tool exchange: the model's own call arguments (bash command, search
  query, graph_query arguments) together with the visible result contain a path token matching P, and
  the visible result contains C*'s simple name (last dotted segment of its qualname) as a whole
  identifier. The user message (r = 0) is judged by STRICT only. LOOSE can overcount (L2 fires on a view
  of C*'s definition without the call line, and on common method names).

Other per-episode fields
------------------------
first_target_search_response: first response whose tool call contains a whole-identifier match of one of
the item's search names (simple name, qualname, frozen names.lexical; score.py episode_telemetry's
pattern) in the bash command, the source_reference_search query or the JSON of the graph_query
arguments. submitted_response: call_index + 1 of submission_accepted. last_action_kind (final
response; first match wins): submit_accepted, submit_rejected, tool_call (executed non-submit call),
tool_rejected (status rejected / invalid_arguments), no_tool_call (format error "your response did not
call a tool"), format_error (other). n_identical_repeat_actions: tool calls whose (name, canonical JSON
arguments) equals an earlier call of the same episode. Prompt tokens: context_accounting.count per
response (max, final). Completion tokens: model_exchange usage.completion_tokens.

Aggregation
-----------
Per block x condition, micro (k/n over episodes) and family-weighted: item mean over its (eligible) seeds
-> family mean over items with an eligible episode -> equal-weight mean over families, nominal 95%
Student t over families (df = F - 1; the frozen analysis' stdlib t). No p-values. Conditioning on exit
status or submission is post-treatment conditioning: descriptive only.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from multiprocessing import Pool
import os
from pathlib import Path
import platform
import re
import shlex
import statistics
import sys
import time

SCHEMA = "expA-timeline-v1"
MODEL_LABEL = "ANALYSIS RECORD"
EXPLORATORY = ("EXPLORATORY: designed after the 31B primary results were seen (2026-11-01 ~06:05 UTC); "
               "not preregistered; descriptive; post-treatment quantities, no causal or mediation claims")
EXPECTED_SHA256 = {
    "items": "f3451a3c9fba2ffd3455df17718d3e276132cf9eb0a7158240256deed32764c8",
    "scores": "a8bc820f3cadda9750adf77c763ac44b92c669109bcf570dc2298f6f181de1fa",
}
RULES_FIXED_AT = "2026-11-01T06:55Z"
BUDGET = 8
CONFIDENCE = 0.95
TOKEN_THRESHOLDS = (12000, 14000, 15872)
MAX_WORKERS = 16
RULES = ("strict", "loose")
CLAUSES = ("S_text", "S_struct", "L1", "L2")
LAST_KINDS = ("submit_accepted", "submit_rejected", "tool_call", "tool_rejected", "no_tool_call", "format_error")
NO_TOOL_CALL_REASON = "your response did not call a tool"
REJECTED_STATUSES = ("rejected", "invalid_arguments")
BLOCK_ORDER = ("P", "S-VC", "S-NAT", "S-EDGE", "T")
CONDITION_ORDER = ("C0", "C1", "C2", "C3", "C4a", "T-offered", "T-recommended")
KNOWN_P = {"submission_rate": {"C0": 0.588, "C1": 0.773, "C2": 0.735, "C3": 0.720, "C4a": 0.784},
           "recovery_given_submission_range": (0.94, 0.97)}

PATH_CHARS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.+@~/-")
PATH_LINE = re.compile(r"(?<![A-Za-z0-9_.+@~/\-])([A-Za-z0-9_.+@~/\-]+):(\d+)")
NUMBERED_LINE = re.compile(r"^ *(\d+)(?:[:\t \-]|$)", re.M)


class InputError(SystemExit):
    pass


# --------------------------------------------------------------------------- input


def read_verified(path, expected, key):
    data = Path(path).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != expected:
        raise InputError(f"{key}: sha256 {digest} of {path} is not the expected {expected}; not read")
    return data, digest


def _reject_constant(name):
    raise ValueError(f"non-finite JSON constant {name}")


def parse_jsonl(data, key):
    rows = []
    for number, line in enumerate(data.decode("utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line, parse_constant=_reject_constant))
            except ValueError as error:
                raise InputError(f"{key}: line {number}: {error}") from error
    return rows


def item_info(record):
    """What the timeline needs from one expA-item-v2 record (C* sites, names, covariates)."""
    omitted = record["omitted"]["identity"]
    entry = [caller for caller in record["V"] if caller.get("identity") == omitted]
    if len(entry) != 1 or not entry[0].get("sites"):
        raise InputError(f"item {record.get('item_id')}: omitted caller {omitted} has no unique V entry with sites")
    sites = []
    for site in entry[0]["sites"]:
        lineno, end = site["lineno"], site["end_lineno"]
        if not (isinstance(site["path"], str) and type(lineno) is int and type(end) is int and 1 <= lineno <= end):
            raise InputError(f"item {record['item_id']}: bad C* site {site}")
        sites.append([site["path"], lineno, end])
    qualname = omitted.split("::", 1)[1]
    target = record["target"]
    names = record.get("names") if isinstance(record.get("names"), dict) else {}
    extra = names.get("lexical") if isinstance(names.get("lexical"), list) else []
    simple = target.get("name") or target["qualname"].rsplit(".", 1)[-1]
    search_names = sorted({simple, target["qualname"]} | {n for n in extra if isinstance(n, str) and n})
    covariates = record.get("covariates") or {}
    return {"item_id": record["item_id"], "family": record["family"], "cstar": omitted,
            "cstar_simple": qualname.rsplit(".", 1)[-1], "cstar_sites": sites, "search_names": search_names,
            "c4_contains_omitted_site": covariates.get("c4_contains_omitted_site")}


# --------------------------------------------------------------------------- matching primitives


def path_matches(token, site_path):
    token = token.replace("\\", "/")
    return token == site_path or token.endswith("/" + site_path)


def path_occurs(text, site_path):
    """site_path occurs in text as a whole path token or a /-suffix of one (a trailing '.' is allowed)."""
    start, size = 0, len(site_path)
    while True:
        index = text.find(site_path, start)
        if index < 0:
            return False
        before = text[index - 1] if index > 0 else ""
        after = text[index + size] if index + size < len(text) else ""
        after2 = text[index + size + 1] if index + size + 1 < len(text) else ""
        before_ok = before == "" or before == "/" or before not in PATH_CHARS
        after_ok = after == "" or after not in PATH_CHARS or (after == "." and (after2 == "" or after2 not in
                                                                                 PATH_CHARS))
        if before_ok and after_ok:
            return True
        start = index + 1


def word_pattern(names):
    names = [name for name in names if name]
    if not names:
        return None
    return re.compile(r"(?<![A-Za-z0-9_])(?:" + "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
                      + r")(?![A-Za-z0-9_])")


def json_strings(value, out):
    """Every string value (not keys) of a decoded JSON value, depth first."""
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            json_strings(item, out)
    elif isinstance(value, list):
        for item in value:
            json_strings(item, out)
    return out


def json_path_lines(value, out):
    """(path, line) of every JSON object with a string 'path' and an integer 'line'."""
    if isinstance(value, dict):
        path, line = value.get("path"), value.get("line")
        if isinstance(path, str) and type(line) is int:
            out.append((path, line))
        for item in value.values():
            json_path_lines(item, out)
    elif isinstance(value, list):
        for item in value:
            json_path_lines(item, out)
    return out


def embedded_json_objects(text):
    """JSON objects embedded in free text (each '{' that starts a decodable object; nested ones skipped)."""
    decoder, found, index = json.JSONDecoder(), [], text.find("{")
    while index >= 0:
        try:
            value, end = decoder.raw_decode(text, index)
        except ValueError:
            index = text.find("{", index + 1)
            continue
        if isinstance(value, dict):
            found.append(value)
        index = text.find("{", end)
    return found


def unit_from_text(text, *, is_json):
    """(strings, structured (path, line) pairs) of one model-visible unit."""
    if is_json:
        try:
            value = json.loads(text)
        except ValueError:
            return [text], []
        return json_strings(value, []), json_path_lines(value, [])
    strings, pairs = [text], []
    for value in embedded_json_objects(text):
        json_strings(value, strings)
        json_path_lines(value, pairs)
    return strings, pairs


def s_text_hit(strings, sites):
    for text in strings:
        for path, lineno, end in sites:
            if path.rsplit("/", 1)[-1] not in text:
                continue
            for match in PATH_LINE.finditer(text):
                if path_matches(match.group(1), path) and lineno <= int(match.group(2)) <= end:
                    return True
    return False


def s_struct_hit(pairs, sites):
    return any(path_matches(p, path) and lineno <= line <= end for p, line in pairs for path, lineno, end in sites)


def _short_clusters(args):
    return [arg[1:] for arg in args if arg.startswith("-") and not arg.startswith("--") and len(arg) > 1]


def numbering_kind(stage):
    """'true' (file line numbers), 'other' (numbers that are not file lines) or None (no numbering)."""
    program = os.path.basename(stage[0]) if stage else ""
    args = list(stage[1:])
    clusters = _short_clusters(args)
    if program in ("grep", "egrep", "fgrep", "rg"):
        return "true" if "--line-number" in args or any("n" in c for c in clusters) else None
    if program == "cat":
        numbered = "--number" in args or any("n" in c for c in clusters)
        renumbered = ("--number-nonblank" in args or "--squeeze-blank" in args
                      or any("b" in c or "s" in c for c in clusters))
        if not numbered and "--number-nonblank" not in args and not any("b" in c for c in clusters):
            return None
        return "other" if renumbered else "true"
    if program == "nl":
        body_all = ("--body-numbering=a" in args or "-ba" in args
                    or any(a == "-b" and i + 1 < len(args) and args[i + 1] == "a" for i, a in enumerate(args)))
        shifted = any(a.startswith(("-v", "-i", "--starting-line-number", "--line-increment")) for a in args)
        return "true" if body_all and not shifted else "other"
    if program in ("awk", "gawk"):
        return "true" if any(re.search(r"(?<![A-Za-z0-9_])NR(?![A-Za-z0-9_])", a) for a in args) else None
    return None


def l1_applies(stages, site_path):
    """L1 preconditions on the harness-recorded argv: one .py operand matching P, true numbering."""
    if not isinstance(stages, list) or not stages or not all(isinstance(s, list) and s for s in stages):
        return False
    operands = set()
    for stage in stages:
        for arg in stage[1:]:
            if isinstance(arg, str) and arg.endswith(".py") and not arg.startswith("-"):
                operands.add(arg.replace("\\", "/")[2:] if arg.startswith("./") else arg.replace("\\", "/"))
    if len(operands) != 1 or not path_matches(next(iter(operands)), site_path):
        return False
    for index, stage in enumerate(stages):
        kind = numbering_kind(stage)
        if kind is None:
            continue
        if kind != "true":
            return False
        return all(os.path.basename(s[0]) == "cat" and not any(str(a).startswith("-") for a in s[1:])
                   for s in stages[:index])
    return False


def l1_hit(stages, output, sites):
    if not isinstance(output, str) or not output:
        return False
    for path, lineno, end in sites:
        if l1_applies(stages, path):
            for match in NUMBERED_LINE.finditer(output):
                if lineno <= int(match.group(1)) <= end:
                    return True
    return False


def name_pattern(simple):
    return re.compile(r"(?<![A-Za-z0-9_])" + re.escape(simple) + r"(?![A-Za-z0-9_])")


def l2_hit(call_strings, result_strings, sites, name):
    """name: the compiled whole-identifier pattern of C*'s simple name (name_pattern)."""
    if not any(name.search(text) for text in result_strings):
        return False
    texts = list(call_strings) + list(result_strings)
    return any(path_occurs(text, path) for path, _, _ in sites for text in texts)


# --------------------------------------------------------------------------- one episode


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def episode_timeline(events, item):
    """Per-episode timeline from the parsed events of one attempt (pure; see the module docstring)."""
    sites, simple = item["cstar_sites"], name_pattern(item["cstar_simple"])
    search = word_pattern(item["search_names"])
    first = {name: None for name in CLAUSES}
    prompt_tokens, completion_tokens = {}, {}
    responses, current = 0, None
    response_kinds = defaultdict(list)   # response number -> kinds of its calls / format error
    actions, statuses, argv = [], {}, {}
    seen_actions, repeats, rejected = set(), 0, 0
    first_target_search, submitted_response, exit_status = None, None, None
    names = Counter()
    truncated_outputs = 0
    audit = Counter()

    def mark(clause, r):
        if first[clause] is None or r < first[clause]:
            first[clause] = r

    for event in events:
        kind, value = event.get("kind"), event.get("value")
        audit["kind:" + str(kind)] += 1
        if kind == "episode_setup":
            strings, pairs = unit_from_text(value.get("user") or "", is_json=False)
            if s_text_hit(strings, sites):
                mark("S_text", 0)
            if s_struct_hit(pairs, sites):
                mark("S_struct", 0)
        elif kind == "context_accounting":
            prompt_tokens[value["call_index"]] = value["count"]
        elif kind == "model_exchange":
            responses += 1
            usage = ((value.get("response") or {}).get("usage") or {})
            completion_tokens[value["call_index"]] = usage.get("completion_tokens")
        elif kind == "format_error":
            reason = str(value.get("reason"))
            response_kinds[value["call_index"] + 1].append("no_tool_call" if reason == NO_TOOL_CALL_REASON
                                                           else "format_error")
        elif kind == "parsed_actions":
            current = value["call_index"] + 1
        elif kind == "tool_action":
            name, arguments = value.get("name"), value.get("arguments")
            names[str(name)] += 1
            key = (str(name), canonical(arguments))
            if key in seen_actions:
                repeats += 1
            seen_actions.add(key)
            actions.append({"response": current, "id": value.get("id"), "name": name, "arguments": arguments})
            if first_target_search is None and name != "submit_answer" and search is not None:
                if name == "bash" and isinstance(arguments, dict):
                    text = arguments.get("command")
                elif name == "source_reference_search" and isinstance(arguments, dict):
                    text = arguments.get("query")
                else:
                    text = canonical(arguments)
                if isinstance(text, str) and search.search(text):
                    first_target_search = current
        elif kind == "tool_output":
            record = value.get("record") or {}
            statuses[value.get("tool_call_id")] = record.get("status")
            argv[value.get("tool_call_id")] = record.get("argv")
            status = record.get("status")
            if status in REJECTED_STATUSES:
                rejected += 1
            if value.get("name") == "submit_answer":
                response_kinds[current].append("submit_rejected")
            elif status in REJECTED_STATUSES:
                response_kinds[current].append("tool_rejected")
            else:
                response_kinds[current].append("tool_call")
            audit[f"status:{value.get('name')}:{status}"] += 1
        elif kind == "model_visible_tool_result":
            call_id, content = value.get("tool_call_id"), value.get("content") or ""
            strings, pairs = unit_from_text(content, is_json=True)
            r = current
            if s_text_hit(strings, sites):
                mark("S_text", r)
            if s_struct_hit(pairs, sites):
                mark("S_struct", r)
            action = next((a for a in reversed(actions) if a["id"] == call_id), None)
            arguments = action["arguments"] if action else None
            call_strings = json_strings(arguments, []) if arguments is not None else []
            output = None
            if value.get("name") == "bash":
                try:
                    decoded = json.loads(content)
                except ValueError:
                    decoded = None
                if isinstance(decoded, dict):
                    output = decoded.get("output")
                    if decoded.get("truncated") is True:
                        truncated_outputs += 1
                if l1_hit(argv.get(call_id), output, sites):
                    mark("L1", r)
            if l2_hit(call_strings, strings, sites, simple):
                mark("L2", r)
        elif kind == "submission_accepted":
            submitted_response = value["call_index"] + 1
            response_kinds[value["call_index"] + 1].append("submit_accepted")
        elif kind == "episode_finished":
            exit_status = value.get("exit_status")
    strict = min((first[c] for c in ("S_text", "S_struct") if first[c] is not None), default=None)
    loose = min((first[c] for c in CLAUSES if first[c] is not None), default=None)
    last_kinds = response_kinds.get(responses, [])
    last_kind = next((k for k in LAST_KINDS if k in last_kinds), None)
    prompts = [prompt_tokens[i] for i in sorted(prompt_tokens)]
    completions = [completion_tokens[i] for i in sorted(completion_tokens)]
    return {"n_responses": responses, "exit_status_events": exit_status,
            "first_target_search_response": first_target_search,
            "first_cstar_visible_response": {"strict": strict, "loose": loose},
            "first_cstar_visible_by_clause": dict(first),
            "submitted_response": submitted_response, "last_action_kind": last_kind,
            "n_tool_actions": len(actions), "tool_actions_by_name": dict(sorted(names.items())),
            "n_rejected_tool_calls": rejected, "n_identical_repeat_actions": repeats,
            "n_truncated_bash_outputs": truncated_outputs,
            "prompt_tokens_by_response": prompts, "prompt_tokens_max": max(prompts) if prompts else None,
            "prompt_tokens_final": prompts[-1] if prompts else None,
            "completion_tokens_by_response": completions,
            "completion_tokens_total": sum(c for c in completions if isinstance(c, int)),
            "_audit": dict(audit)}


_ITEMS = None
_RUN_DIR = None


def _init_worker(items, run_dir):
    global _ITEMS, _RUN_DIR
    _ITEMS, _RUN_DIR = items, run_dir


def process_cell(task):
    """Read one cell's terminal attempt (sha256-bound to the scores row) and build its row."""
    score = task
    result_path = Path(_RUN_DIR) / score["result_path"]
    raw_result = result_path.read_bytes()
    problems = []
    if hashlib.sha256(raw_result).hexdigest() != score["result_sha256"]:
        problems.append("result_sha256")
    result = json.loads(raw_result)
    events_name = (result.get("events") or {}).get("path") or "events.jsonl"
    raw_events = (result_path.parent / events_name).read_bytes()
    if hashlib.sha256(raw_events).hexdigest() != (result.get("events") or {}).get("sha256"):
        problems.append("events_sha256")
    events = [json.loads(line) for line in raw_events.decode("utf-8").splitlines() if line.strip()]
    if len(events) != (result.get("events") or {}).get("records"):
        problems.append("events_records")
    item = _ITEMS[score["item_id"]]
    timeline = episode_timeline(events, item)
    audit = timeline.pop("_audit")
    if timeline["exit_status_events"] != result.get("exit_status") or result.get("exit_status") != score["exit_status"]:
        problems.append("exit_status")
    if timeline["n_responses"] != result.get("calls"):
        problems.append("calls")
    tokens = result.get("tokens") or {}
    if sum(timeline["prompt_tokens_by_response"][:timeline["n_responses"]]) != tokens.get("prompt"):
        problems.append("prompt_tokens_sum")
    if timeline["completion_tokens_total"] != tokens.get("completion"):
        problems.append("completion_tokens_sum")
    if score["exit_status"] == "Submitted" and timeline["submitted_response"] != timeline["n_responses"]:
        problems.append("submitted_response")
    telemetry = score.get("telemetry") or {}
    row = {"cell_id": score["cell_id"], "block": score["block"], "condition": score["condition"],
           "variant": score["variant"], "seed": score["seed"], "family": score["family"],
           "item_id": score["item_id"], "attempt_id": score["attempt_id"], "exit_status": score["exit_status"],
           "site_recovered": score.get("site_recovered"),
           "c4_contains_omitted_site": item["c4_contains_omitted_site"],
           "score_searched_target_name": telemetry.get("searched_target_name"),
           "problems": problems}
    timeline.pop("exit_status_events")
    row.update(timeline)
    return row, audit


# --------------------------------------------------------------------------- statistics


def _beta_continued_fraction(a, b, x):
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 2000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-16:
            return h
    raise ArithmeticError("incomplete beta continued fraction did not converge")


def regularized_beta(x, a, b):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def student_t_cdf(t, df):
    tail = 0.5 * regularized_beta(df / (df + t * t), df / 2.0, 0.5)
    return 1.0 - tail if t >= 0 else tail


def student_t_ppf(p, df):
    """Quantile by bisection on the CDF (as the frozen absent_edges.probe.analysis.student_t_ppf)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    if p == 0.5:
        return 0.0
    if p < 0.5:
        return -student_t_ppf(1.0 - p, df)
    lo, hi = 0.0, 1.0
    while student_t_cdf(hi, df) < p:
        lo, hi = hi, hi * 2.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if student_t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-13 * max(1.0, hi):
            break
    return (lo + hi) / 2.0


def t_interval(values, confidence=CONFIDENCE):
    vals = [float(v) for v in values]
    n = len(vals)
    out = {"n_families": n, "mean": math.fsum(vals) / n if n else None}
    if n < 2:
        out.update(df=None, se=None, lower=None, upper=None)
        return out
    sd = statistics.stdev(vals)
    se = sd / math.sqrt(n)
    critical = student_t_ppf(1.0 - (1.0 - confidence) / 2.0, n - 1)
    out.update(df=n - 1, se=se, lower=out["mean"] - critical * se, upper=out["mean"] + critical * se)
    return out


def family_weighted(rows, value, eligible=None):
    """Item mean over eligible seeds -> family mean -> equal-weight family mean with a Student t interval."""
    by_item = defaultdict(list)
    for row in rows:
        if eligible is None or eligible(row):
            by_item[(row["family"], row["item_id"])].append(float(value(row)))
    by_family = defaultdict(list)
    for (family, _), values in sorted(by_item.items()):
        by_family[family].append(math.fsum(values) / len(values))
    family_means = [math.fsum(v) / len(v) for _, v in sorted(by_family.items())]
    out = t_interval(family_means)
    out["n_items"] = len(by_item)
    return out


def micro(rows, indicator, eligible=None):
    pool = [row for row in rows if eligible is None or eligible(row)]
    k = sum(1 for row in pool if indicator(row))
    return {"k": k, "n": len(pool), "share": k / len(pool) if pool else None}


def rate(rows, indicator, eligible=None):
    return {"micro": micro(rows, indicator, eligible),
            "family_weighted": family_weighted(rows, lambda r: 1.0 if indicator(r) else 0.0, eligible)}


def percentile(sorted_values, q):
    """Linear interpolation between closest ranks, position (n - 1) * q."""
    if not sorted_values:
        return None
    position = (len(sorted_values) - 1) * q
    low = math.floor(position)
    high = min(low + 1, len(sorted_values) - 1)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (position - low)


# --------------------------------------------------------------------------- aggregation


def vis(row, rule):
    return row["first_cstar_visible_response"][rule]


def visible_before(row, rule, k):
    v = vis(row, rule)
    return v is not None and v <= k - 1


def group_summary(rows):
    submitted = lambda r: r["exit_status"] == "Submitted"
    no_answer = lambda r: r["exit_status"] == "NoAnswer"
    out = {"n_episodes": len(rows), "n_items": len({r["item_id"] for r in rows}),
           "n_families": len({r["family"] for r in rows}),
           "exit_status": dict(sorted(Counter(r["exit_status"] for r in rows).items())),
           "submission_rate": rate(rows, submitted),
           "recovery_given_submission": rate(rows, lambda r: r["site_recovered"] == 1, submitted),
           "site_recovered": rate(rows, lambda r: r["site_recovered"] == 1)}
    by_rule = {}
    for rule in RULES:
        entry = {
            "visible_in_initial_prompt": rate(rows, lambda r, rule=rule: vis(r, rule) == 0),
            "noanswer_visible_before_cap": rate(
                rows, lambda r, rule=rule: visible_before(r, rule, r["n_responses"]), no_answer),
            "noanswer_revealed_at_all": rate(rows, lambda r, rule=rule: vis(r, rule) is not None, no_answer),
            "submitted_visible_before_submission": rate(
                rows, lambda r, rule=rule: visible_before(r, rule, r["submitted_response"]), submitted),
            "submitted_missed_visible_before_submission": rate(
                rows, lambda r, rule=rule: visible_before(r, rule, r["submitted_response"]),
                lambda r: submitted(r) and r["site_recovered"] == 0),
            "submitted_recovered_visible_before_submission": rate(
                rows, lambda r, rule=rule: visible_before(r, rule, r["submitted_response"]),
                lambda r: submitted(r) and r["site_recovered"] == 1),
            "noanswer_first_visible_histogram": dict(sorted(Counter(
                "never" if vis(r, rule) is None else str(vis(r, rule)) for r in rows if no_answer(r)).items())),
        }
        by_rule[rule] = entry
    out["visibility"] = by_rule
    curves = {}
    for k in range(0, BUDGET + 1):
        point = {"submitted_by_k": rate(rows, lambda r, k=k: r["submitted_response"] is not None
                                        and r["submitted_response"] <= k),
                 "submitted_with_cstar_recovered_by_k": rate(
                     rows, lambda r, k=k: r["submitted_response"] is not None and r["submitted_response"] <= k
                     and r["site_recovered"] == 1)}
        for rule in RULES:
            point[f"cstar_visible_by_k_{rule}"] = rate(rows, lambda r, k=k, rule=rule: vis(r, rule) is not None
                                                       and vis(r, rule) <= k)
            point[f"submitted_with_cstar_visible_before_submission_by_k_{rule}"] = rate(
                rows, lambda r, k=k, rule=rule: r["submitted_response"] is not None and r["submitted_response"] <= k
                and visible_before(r, rule, r["submitted_response"]))
        curves[str(k)] = point
    out["cumulative_by_k"] = curves
    out["responses_among_submitted"] = {
        "micro_mean": (statistics.fmean([r["submitted_response"] for r in rows if submitted(r)])
                       if any(submitted(r) for r in rows) else None),
        "family_weighted": family_weighted(rows, lambda r: r["submitted_response"], submitted),
        "histogram": dict(sorted(Counter(str(r["submitted_response"]) for r in rows if submitted(r)).items()))}
    maxima = sorted(r["prompt_tokens_max"] for r in rows if r["prompt_tokens_max"] is not None)
    out["max_prompt_tokens"] = {
        "n": len(maxima), "median": percentile(maxima, 0.5), "p90": percentile(maxima, 0.9),
        "p99": percentile(maxima, 0.99), "max": maxima[-1] if maxima else None,
        "over": {str(t): rate(rows, lambda r, t=t: (r["prompt_tokens_max"] or 0) > t) for t in TOKEN_THRESHOLDS},
        "over_by_exit": {str(t): {status: micro(rows, lambda r, t=t: (r["prompt_tokens_max"] or 0) > t,
                                               lambda r, status=status: r["exit_status"] == status)
                                  for status in ("Submitted", "NoAnswer")} for t in TOKEN_THRESHOLDS}}
    finals = sorted(r["prompt_tokens_final"] for r in rows if no_answer(r) and r["prompt_tokens_final"] is not None)
    out["noanswer_final_prompt_tokens"] = {"n": len(finals), "median": percentile(finals, 0.5),
                                           "p90": percentile(finals, 0.9), "max": finals[-1] if finals else None}
    out["noanswer_last_action_kind"] = {
        "counts": dict(sorted(Counter(str(r["last_action_kind"]) for r in rows if no_answer(r)).items())),
        "rates": {kind: rate(rows, lambda r, kind=kind: r["last_action_kind"] == kind, no_answer)
                  for kind in LAST_KINDS}}
    actions = sum(r["n_tool_actions"] for r in rows)
    repeats = sum(r["n_identical_repeat_actions"] for r in rows)
    out["repeat_actions"] = {
        "episodes_with_repeat": rate(rows, lambda r: r["n_identical_repeat_actions"] > 0),
        "episodes_with_repeat_by_exit": {status: rate(rows, lambda r: r["n_identical_repeat_actions"] > 0,
                                                      lambda r, status=status: r["exit_status"] == status)
                                         for status in ("Submitted", "NoAnswer")},
        "repeats_per_tool_action_micro": repeats / actions if actions else None,
        "repeats": repeats, "tool_actions": actions,
        "mean_repeats_per_episode": {"micro": repeats / len(rows) if rows else None,
                                     "family_weighted": family_weighted(rows,
                                                                        lambda r: r["n_identical_repeat_actions"])}}
    out["target_search"] = {
        "searched": rate(rows, lambda r: r["first_target_search_response"] is not None),
        "first_response_histogram": dict(sorted(Counter(str(r["first_target_search_response"])
                                                        for r in rows).items()))}
    out["rejected_tool_calls"] = {"total": sum(r["n_rejected_tool_calls"] for r in rows),
                                  "episodes_with_any": micro(rows, lambda r: r["n_rejected_tool_calls"] > 0)}
    return out


def checks(rows, groups):
    out = {}
    problems = Counter(p for r in rows for p in r["problems"])
    out["binding_and_consistency_problems"] = dict(sorted(problems.items()))
    out["noanswer_all_at_budget"] = all(r["n_responses"] == BUDGET for r in rows if r["exit_status"] == "NoAnswer")
    out["exit_statuses"] = dict(sorted(Counter(r["exit_status"] for r in rows).items()))
    out["c4a_strict_r0_vs_item_covariate"] = {}
    for block in sorted({r["block"] for r in rows if r["condition"] == "C4a"}):
        agree = Counter((bool(r["c4_contains_omitted_site"]), vis(r, "strict") == 0) for r in rows
                        if r["condition"] == "C4a" and r["block"] == block)
        out["c4a_strict_r0_vs_item_covariate"][block] = {
            "covariate_true_visible_r0": agree[(True, True)], "covariate_true_not_visible_r0": agree[(True, False)],
            "covariate_false_visible_r0": agree[(False, True)],
            "covariate_false_not_visible_r0": agree[(False, False)]}
    out["non_c4a_strict_visible_r0"] = dict(sorted(Counter(
        f"{r['block']}|{r['condition']}" for r in rows if r["condition"] != "C4a" and vis(r, "strict") == 0).items()))
    match = Counter((r["block"] == "T", (r["first_target_search_response"] is not None)
                     == bool(r["score_searched_target_name"])) for r in rows)
    out["target_search_vs_score_telemetry"] = {"non_T_agree": match[(False, True)],
                                               "non_T_disagree": match[(False, False)],
                                               "T_agree": match[(True, True)], "T_disagree": match[(True, False)]}
    out["loose_never_before_strict"] = all(
        vis(r, "strict") is None or (vis(r, "loose") is not None and vis(r, "loose") <= vis(r, "strict"))
        for r in rows)
    known = {}
    for condition, value in KNOWN_P["submission_rate"].items():
        group = groups.get(f"P|{condition}")
        if group is None:
            continue
        sub = group["submission_rate"]
        rec = group["recovery_given_submission"]
        known[condition] = {"known_submission_rate": value,
                            "micro": round(sub["micro"]["share"], 4),
                            "family_weighted": round(sub["family_weighted"]["mean"], 4),
                            "recovery_given_submission_micro": round(rec["micro"]["share"], 4),
                            "recovery_given_submission_family_weighted": round(rec["family_weighted"]["mean"], 4)}
    out["known_aggregates_P"] = known
    return out


# --------------------------------------------------------------------------- report


def _num(value, digits=3):
    return "n/a" if value is None else f"{value:.{digits}f}"


def _fw(entry):
    fw = entry["family_weighted"]
    if fw["mean"] is None:
        return "n/a"
    if fw["lower"] is None:
        return f"{fw['mean']:.3f} (F={fw['n_families']})"
    return f"{fw['mean']:.3f} [{fw['lower']:.3f}, {fw['upper']:.3f}] (F={fw['n_families']})"


def _mi(entry):
    m = entry["micro"]
    return "n/a" if m["share"] is None else f"{m['k']}/{m['n']} = {m['share']:.3f}"


def render_markdown(summary, meta):
    lines = [f"# Exp A `{meta['run']}`: C* visibility and budget timeline", "",
             f"**{EXPLORATORY}.**", "", f"**{MODEL_LABEL}.**", "",
             f"Rules fixed {RULES_FIXED_AT}, before any outcome split (RULES.md; script docstring). STRICT is a "
             "lower bound; LOOSE can overcount. Micro = k/n over episodes; FW = family-weighted mean (item mean over "
             "seeds -> family mean -> equal-weight over families) with a nominal 95% Student t interval over "
             "families. Conditioning on exit status or submission is post-treatment: descriptive only.", "",
             "## Inputs, command, outputs", ""]
    for key, value in meta["inputs"].items():
        lines.append(f"- {key}: `{value['path']}` sha256 `{value['sha256']}`")
    lines += [f"- command: `{meta['command']}`", f"- cwd: `{meta['cwd']}`; python {meta['python']}; workers "
              f"{meta['workers']}; runtime {meta['runtime_s']:.1f} s (start {meta['start_utc']}, end {meta['end_utc']})"]
    for key, value in meta["outputs"].items():
        lines.append(f"- output {key}: sha256 `{value['sha256']}` ({value['bytes']} bytes)")
    lines += ["", "## Checks", "", "```", json.dumps(summary["checks"], indent=1, sort_keys=True), "```", ""]
    groups = summary["groups"]
    p_keys = [k for k in groups if k.startswith("P|")]
    lines += ["## Block P: visibility (STRICT / LOOSE)", "",
              "| cond | rule | NoAnswer: C* visible before cap (micro) | FW | Submitted: visible before submission "
              "(micro) | FW | Submitted-but-missed: visible (micro) | FW | Submitted+recovered: visible (micro) "
              "| in initial prompt (micro) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for key in p_keys:
        for rule in RULES:
            v = groups[key]["visibility"][rule]
            lines.append(f"| {key[2:]} | {rule} | {_mi(v['noanswer_visible_before_cap'])} | "
                         f"{_fw(v['noanswer_visible_before_cap'])} | {_mi(v['submitted_visible_before_submission'])} "
                         f"| {_fw(v['submitted_visible_before_submission'])} | "
                         f"{_mi(v['submitted_missed_visible_before_submission'])} | "
                         f"{_fw(v['submitted_missed_visible_before_submission'])} | "
                         f"{_mi(v['submitted_recovered_visible_before_submission'])} | "
                         f"{_mi(v['visible_in_initial_prompt'])} |")
    lines += ["", "## Block P: exits, responses, tokens", "",
              "| cond | n | submitted (micro) | FW | recovered given submitted (micro) | FW | mean responses among "
              "Submitted (micro) | FW | max prompt median / p90 / p99 / max | >12,000 | >14,000 | >15,872 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key in p_keys:
        g = groups[key]
        mp = g["max_prompt_tokens"]
        ras = g["responses_among_submitted"]
        lines.append(f"| {key[2:]} | {g['n_episodes']} | {_mi(g['submission_rate'])} | {_fw(g['submission_rate'])} | "
                     f"{_mi(g['recovery_given_submission'])} | {_fw(g['recovery_given_submission'])} | "
                     f"{_num(ras['micro_mean'])} | {_num(ras['family_weighted']['mean'])} | "
                     f"{mp['median']:.0f} / {mp['p90']:.0f} / {mp['p99']:.0f} / {mp['max']} | "
                     + " | ".join(str(mp["over"][str(t)]["micro"]["k"]) for t in TOKEN_THRESHOLDS) + " |")
    for rule in RULES:
        lines += ["", f"## Block P: cumulative curves by k responses ({rule.upper()}; micro share, FW mean)", "",
                  "| cond | curve | " + " | ".join(f"k={k}" for k in range(BUDGET + 1)) + " |",
                  "|---|---|" + "---|" * (BUDGET + 1)]
        for key in p_keys:
            curves = groups[key]["cumulative_by_k"]
            for label, field in ((f"C* visible by k", f"cstar_visible_by_k_{rule}"),
                                 ("submitted by k", "submitted_by_k"),
                                 ("submitted with C* recovered by k", "submitted_with_cstar_recovered_by_k"),
                                 ("submitted with C* visible before submission, by k",
                                  f"submitted_with_cstar_visible_before_submission_by_k_{rule}")):
                cells = []
                for k in range(BUDGET + 1):
                    e = curves[str(k)][field]
                    cells.append(f"{e['micro']['share']:.3f} / {e['family_weighted']['mean']:.3f}")
                lines.append(f"| {key[2:]} | {label} | " + " | ".join(cells) + " |")
    lines += ["", "## Block P: NoAnswer at the cap", "",
              "| cond | last action kind counts | first visible response (STRICT) | first visible response (LOOSE) |"
              " final prompt tokens median / p90 / max |", "|---|---|---|---|---|"]
    for key in p_keys:
        g = groups[key]
        fp = g["noanswer_final_prompt_tokens"]
        lines.append(f"| {key[2:]} | {json.dumps(g['noanswer_last_action_kind']['counts'], sort_keys=True)} | "
                     f"{json.dumps(g['visibility']['strict']['noanswer_first_visible_histogram'], sort_keys=True)} | "
                     f"{json.dumps(g['visibility']['loose']['noanswer_first_visible_histogram'], sort_keys=True)} | "
                     f"{fp['median']} / {fp['p90']} / {fp['max']} |")
    lines += ["", "## Block P: repeated identical actions, target search", "",
              "| cond | episodes with >=1 repeat (micro) | FW | in Submitted | in NoAnswer | repeats per tool action |"
              " searched target (micro) | rejected tool calls |", "|---|---|---|---|---|---|---|---|"]
    for key in p_keys:
        g = groups[key]
        ra = g["repeat_actions"]
        lines.append(f"| {key[2:]} | {_mi(ra['episodes_with_repeat'])} | {_fw(ra['episodes_with_repeat'])} | "
                     f"{_mi(ra['episodes_with_repeat_by_exit']['Submitted'])} | "
                     f"{_mi(ra['episodes_with_repeat_by_exit']['NoAnswer'])} | "
                     f"{ra['repeats_per_tool_action_micro']:.4f} | {_mi(g['target_search']['searched'])} | "
                     f"{g['rejected_tool_calls']['total']} |")
    lines += ["", "## Secondary blocks (S-VC, S-NAT, S-EDGE, T): visibility before cap / before submission", "",
              "In S-VC and S-NAT the delivered answer can name C* (a match record or an endpoint snippet without a "
              "path:line pair); neither rule counts that, so r = 0 visibility is understated there.", "",
              "| block/cond | n | submitted (micro) | NoAnswer: visible before cap STRICT / LOOSE (micro) | "
              "Submitted: visible before submission STRICT / LOOSE (micro) |", "|---|---|---|---|---|"]
    for key, g in groups.items():
        if key.startswith("P|"):
            continue
        vs, vl = g["visibility"]["strict"], g["visibility"]["loose"]
        lines.append(f"| {key} | {g['n_episodes']} | {_mi(g['submission_rate'])} | "
                     f"{_mi(vs['noanswer_visible_before_cap'])} / {_mi(vl['noanswer_visible_before_cap'])} | "
                     f"{_mi(vs['submitted_visible_before_submission'])} / "
                     f"{_mi(vl['submitted_visible_before_submission'])} |")
    lines += ["", "## Parse audit (all episodes)", "", "```", json.dumps(summary["parse_audit"], indent=1,
                                                                           sort_keys=True), "```", ""]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- main


def group_key(row):
    return f"{row['block']}|{row['condition']}"


def _group_order(key):
    block, condition = key.split("|", 1)
    return (BLOCK_ORDER.index(block) if block in BLOCK_ORDER else 99,
            CONDITION_ORDER.index(condition) if condition in CONDITION_ORDER else 99, key)


def summarize(rows, audit):
    grouped = defaultdict(list)
    for row in rows:
        grouped[group_key(row)].append(row)
    groups = {key: group_summary(grouped[key]) for key in sorted(grouped, key=_group_order)}
    return {"schema": SCHEMA, "labels": {"exploratory": EXPLORATORY, "model_analysis": MODEL_LABEL},
            "rules_fixed_at": RULES_FIXED_AT, "budget": BUDGET, "confidence": CONFIDENCE,
            "token_thresholds": list(TOKEN_THRESHOLDS), "groups": groups, "checks": checks(rows, groups),
            "parse_audit": dict(sorted(audit.items()))}


def dump_json(value):
    return (json.dumps(value, sort_keys=True, indent=1, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--items", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--items-sha256", default=EXPECTED_SHA256["items"])
    parser.add_argument("--scores-sha256", default=EXPECTED_SHA256["scores"])
    parser.add_argument("--run", default="main-20261101b")
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= MAX_WORKERS:
        raise InputError(f"--workers must be 1..{MAX_WORKERS} (Core2 is shared)")
    out = Path(args.out)
    targets = [out / name for name in ("TIMELINE.json", "TIMELINE.md", "episodes.jsonl")]
    existing = [str(p) for p in targets if p.exists()]
    if existing:
        raise InputError(f"refusing to overwrite {existing}")
    start_wall, start = time.time(), time.monotonic()
    items_bytes, items_sha = read_verified(args.items, args.items_sha256, "items")
    scores_bytes, scores_sha = read_verified(args.scores, args.scores_sha256, "scores")
    items = {}
    for record in parse_jsonl(items_bytes, "items"):
        info = item_info(record)
        items[info["item_id"]] = info
    scores = parse_jsonl(scores_bytes, "scores")
    tasks = []
    for row in scores:
        if row.get("kind") != "episode":
            raise InputError(f"scores row {row.get('cell_id')} is not an episode row")
        if not row.get("completed") or not row.get("result_path"):
            raise InputError(f"cell {row.get('cell_id')} has no completed terminal attempt")
        if row["omitted"] != items[row["item_id"]]["cstar"]:
            raise InputError(f"cell {row['cell_id']}: scores omitted {row['omitted']} differs from the item")
        tasks.append({key: row.get(key) for key in ("cell_id", "block", "condition", "variant", "seed", "family",
                                                     "item_id", "attempt_id", "exit_status", "site_recovered",
                                                     "result_path", "result_sha256", "telemetry")})
    rows, audit = [], Counter()
    with Pool(args.workers, initializer=_init_worker, initargs=(items, args.run_dir)) as pool:
        for row, row_audit in pool.imap_unordered(process_cell, tasks, chunksize=32):
            rows.append(row)
            audit.update(row_audit)
    rows.sort(key=lambda r: r["cell_id"])
    out.mkdir(parents=True, exist_ok=True)
    episodes_bytes = b"".join((json.dumps(r, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                                          allow_nan=False) + "\n").encode("utf-8") for r in rows)
    summary = summarize(rows, audit)
    summary["run"] = args.run
    summary["inputs"] = {"items": {"sha256": items_sha, "records": len(items)},
                         "scores": {"sha256": scores_sha, "rows": len(scores)}}
    summary["episodes_sha256"] = hashlib.sha256(episodes_bytes).hexdigest()
    timeline_bytes = dump_json(summary)
    (out / "episodes.jsonl").write_bytes(episodes_bytes)
    (out / "TIMELINE.json").write_bytes(timeline_bytes)
    end_wall = time.time()
    meta = {"run": args.run,
            "inputs": {"items": {"path": args.items, "sha256": items_sha},
                       "scores": {"path": args.scores, "sha256": scores_sha},
                       "run_dir": {"path": args.run_dir, "sha256": "n/a (per-episode sha256 binding)"}},
            "command": shlex.join([sys.executable] + (["-B"] if sys.flags.dont_write_bytecode else [])
                                  + list(sys.argv if argv is None else ["expA_timeline.py"] + argv)),
            "cwd": os.getcwd(), "python": platform.python_version(), "workers": args.workers,
            "runtime_s": time.monotonic() - start,
            "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start_wall)),
            "end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(end_wall)),
            "outputs": {"TIMELINE.json": {"sha256": hashlib.sha256(timeline_bytes).hexdigest(),
                                          "bytes": len(timeline_bytes)},
                        "episodes.jsonl": {"sha256": summary["episodes_sha256"], "bytes": len(episodes_bytes)}}}
    (out / "TIMELINE.md").write_text(render_markdown(summary, meta), encoding="utf-8", newline="\n")
    print(json.dumps({"rows": len(rows), "runtime_s": round(meta["runtime_s"], 1),
                      "outputs": meta["outputs"], "problems": summary["checks"]["binding_and_consistency_problems"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
