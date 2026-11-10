"""Exp A episode loop (docs/EXPA_DESIGN.md section 7, docs/EXPA_DESIGN_V2_DELTA.md D6, D8, D10, D11).

One episode is one conversation between the local Gemma endpoint and the Exp A
tools about one cell (block, item, variant, condition, seed). The model client
is absent_edges/models.py, unchanged: LocalModel.tokenize for an exact preflight
of every request (``/tokenize`` on the full serialized request: messages, tools,
chat template, generation prompt), LocalModel.chat with return_token_ids,
tool_choice="auto" and enable_thinking=False, and parsed_tool_calls for native
tool-call validation.

Design (D8 as amended by D11.4): block P holds an item's five primary cells,
``none`` x C0 and ``omit_node`` x {C1, C2, C3, C4a}, at seeds 11, 23 and 47.
S-C0 no longer exists. The secondary blocks S-VC, S-NAT, S-EDGE and T run at
seed 11. ``BLOCK_DESIGN`` is the single source of the matrix.

Prompts (D6 + D10.5 + D10.6): the fixed system message (ending with the budget
sentence) and the task template. Only C1, C2, C3 and C4a carry the answer
block; C0 and the T arms have none. T-recommended appends the recommendation
sentence to the task paragraph, before the submit sentence. The P and S blocks
offer source_reference_search, bash and submit_answer; the T arms also offer
graph_query (graphs.GRAPH_TOOL) answered from the family's natural LocAgent
facts.

Budgets (D10.12): at most 8 model calls, and every model response counts; a
16,384-token episode context; max_tokens = min(4096, remaining); ContextLimit
when fewer than 512 tokens remain before a call; finish_reason "length" is
counted per episode; temperature 0.2 and the cell's seed. Every assistant
message must contain a valid tool call; a message without one gets a
model-visible format error and still counts as a call. Several tool calls in
one response run in order; once a submit_answer is accepted the remaining calls
of that response are logged and not run. Exits: Submitted, NoAnswer (8 calls
used without a submission), ContextLimit, FormatExhausted (3 consecutive format
errors) and InfraInterrupted (transport, server, accounting, sandbox setup,
stop request or controller failure; only these give the cell a new attempt). A
model-caused condition is never infrastructure: arguments that are not UTF-8
encodable, for example, get a model-visible tool error (finding F5).

submit_answer (contract item 5, D11.5): the envelope must be
``{"sites": [...], "complete": "yes"|"no"|"unknown"}`` (a JSON boolean maps to
yes/no); otherwise the call is rejected with a visible reason and the model may
resubmit. Entries are never rejected one by one: a valid entry is normalized to
``{"path", "line"}`` ("/repo/" and "./" stripped, a digit-string line becomes an
integer) and an invalid entry is kept exactly as submitted, so the scorer
counts it as unresolved. ``submission_validation`` lists what was normalized.

The exposure receipt (finding F8) records whether the answer block and the C4a
verification block were delivered (with sha256 and hit count), the tools
offered and whether the recommendation sentence was present; it is logged
before the first model call with the payload's exact served-tokenizer count.
Every event is fsync'd; result.json is immutable and its sha256 is returned to
the ledger. The episode never sees ground truth: only the target, its location,
the package directories and the delivered payload enter the prompt.

Final-turn policy (protocols/EXPA_FINALTURN.md, follow-up A1): a run-level
option, default "none" (the prereg harness). Under "neutral" or "deadline" one
fixed notice, "\\n\\n[harness notice] " + the policy's text, is appended in-band
to the content of the last message before the request for the last response
(call_index max_model_calls - 1): the last model-visible tool result, or the
format-error message when the previous response was a format error. No message
or turn is added. It is appended after the stop check and before the /tokenize
preflight, so it counts against the context budget (ContextLimit when fewer than
512 tokens remain with it). An episode that has submitted never sees it. The
model_visible_tool_result event keeps the tool's own output; the
final_turn_notice event binds the appended content, and the receipt (appended,
sent, delivered) is in the result's exposure and accounting. EpisodeConfig.record
writes the policy only when it is not "none", so default records are those of
the frozen release. Follow-up A4 (protocols/EXPA_A4_DIRECTIVE.md) adds two
policies, "deadline_info" and "submit_only", whose texts are the two halves of
the deadline text; everything else about the notice is the same, and their
config records name the A4 protocol.

Evidence ledger (protocols/EXPA_A3_LEDGER.md, follow-up A3): a run-level
option, default "none" (every earlier harness). "off" and "on" are the two arms
of A3; both need a final-turn notice (the protocol registers deadline) and both
run the A3 design, whose LEDGER block (vcomplete x C1 and omit_node x C1 at
seeds 61, 73 and 89) is dispatched before the default blocks and is not part of
BLOCK_DESIGN, so the default design and its records are unchanged. At the last
call, before the final-turn notice, the harness builds the ledger
(evidence_ledger()) from agent-visible text only: the model-visible tool results
already in the history. Under "on" "\\n\\n[harness notice] " + the ledger text is
appended in-band to the same message as the final-turn notice, before it, so the
deadline notice stays last. Under "off" the same ledger is built and recorded
but not sent. The evidence_ledger event records the text, its entries and
sha256; the receipt (computed, appended, sent, delivered) is in the result's
exposure and accounting. EpisodeConfig.record writes the option only when it is
not "none".

Submit-time review (protocols/EXPA_A5_REVIEW.md, follow-up A5): a run-level
option, default "none" (every earlier harness); "neutral" and "ledger" run with
the deadline notice and no evidence ledger, on the A3 design (block LEDGER).
The agent's FIRST well-formed submit_answer at a call index before the last
call is not accepted: its arguments are recorded (submit_review event) and its
model-visible tool result is "[harness notice] " + TEXT, where TEXT is the
neutral review sentence, or the evidence ledger built from the agent-visible
history at that point (same header, entries, empty text and caps as A3)
followed by a line with the review instruction. The next well-formed
submit_answer is accepted as final (no second review). When the episode ends
without one (NoAnswer, ContextLimit or FormatExhausted), the first submission
is the answer and the exit is Submitted; InfraInterrupted stays infrastructure.
A first submit_answer at the last call is accepted at once (no review). The
deadline notice is unchanged. The receipt (reviewed, text and content sha256,
ledger sizes, the first submission, answer_source "first_submit" or
"resubmit") is in the result's exposure and accounting, and result.json's
"submitted" is always the answer used. EpisodeConfig.record writes the option
only when it is not "none".

Task framing (protocols/EXPA_A7_IMPACT.md, follow-up A7): a run-level option,
default "default" (every earlier harness: USER_TASK_TEMPLATE). Under "impact"
the task paragraph is IMPACT_TASK_TEMPLATE (an impact analysis of a signature
change) with the same placeholders; the system prompt, the answer block, the
recommendation and submit sentences, budgets and tools are unchanged. It runs
only with the prereg budget (8 responses, 16,384 tokens) and without a
final-turn notice, an evidence ledger or a submit review, on the default
design. prompt_templates() is unchanged; EpisodeConfig.record writes the option,
the protocol, the template text and its sha256 only when it is not "default".
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import datetime
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import time
import traceback

from ..data import canonical_json
from ..models import TOOL_ARGUMENT_BYTES, TOOL_CALLS_PER_MESSAGE, LocalModel, parsed_tool_calls, token_ids_digest
from .identity import normalize_package_dirs as _identity_package_dirs
from .identity import split_identity
from .ledger import COMPLETED_EXITS, INFRA_EXIT, fsync_directory, write_immutable_json
from .tools import (GRAPH_QUERY, INVALID_UNICODE, LEXICAL_KIND, SEARCH_MAX_HITS, SandboxError, ServerTokenCounter,
                    normalize_site_path, tool_names, tool_schemas, utf8_encodable, validate_submission)

HARNESS_PROTOCOL = "expA-harness-v2"
RESULT_SCHEMA = "expA-episode-result-v2"
EXIT_SUBMITTED, EXIT_NO_ANSWER, EXIT_CONTEXT_LIMIT, EXIT_FORMAT_EXHAUSTED = (
    "Submitted", "NoAnswer", "ContextLimit", "FormatExhausted")
EXIT_INFRA = INFRA_EXIT
EXITS = (EXIT_SUBMITTED, EXIT_NO_ANSWER, EXIT_CONTEXT_LIMIT, EXIT_FORMAT_EXHAUSTED, EXIT_INFRA)
assert set(EXITS) == set(COMPLETED_EXITS) | {INFRA_EXIT}
INFRA_KINDS = ("server", "accounting", "stop_requested", "sandbox", "controller_exception", "keyboard_interrupt")

# -- design (D8 as amended by D11.4; D10.8 order) -------------------------------
VARIANTS = ("vcomplete", "natural", "omit_node", "omit_edge")
NO_VARIANT = "none"
PAYLOAD_CONDITIONS = ("C1", "C2", "C3", "C4a")
T_CONDITIONS = ("T-offered", "T-recommended")
CONDITIONS = ("C0",) + PAYLOAD_CONDITIONS + T_CONDITIONS
WORDING_PAIRS = ("A", "B")
SEEDS = (11, 23, 47)
SECONDARY_SEED = 11
# Each block lists its cells as (variant, condition) in the order an item's cells are dispatched together.
BLOCK_DESIGN = {
    "P": {"cells": ((NO_VARIANT, "C0"), ("omit_node", "C1"), ("omit_node", "C2"), ("omit_node", "C3"),
                    ("omit_node", "C4a")),
          "seeds": SEEDS, "membership": None, "role": "primary (H1 C3-C2, H2 C4a-C1, H3 C1-C0)"},
    "S-VC": {"cells": (("vcomplete", "C1"), ("vcomplete", "C3")), "seeds": (SECONDARY_SEED,), "membership": None,
             "role": "secondary"},
    "S-NAT": {"cells": (("natural", "C1"), ("natural", "C3"), ("natural", "C4a")), "seeds": (SECONDARY_SEED,),
              "membership": "in_s_nat", "role": "secondary"},
    "S-EDGE": {"cells": (("omit_edge", "C1"), ("omit_edge", "C3")), "seeds": (SECONDARY_SEED,),
               "membership": "in_s_edge", "role": "secondary"},
    "T": {"cells": ((NO_VARIANT, "T-offered"), (NO_VARIANT, "T-recommended")), "seeds": (SECONDARY_SEED,),
          "membership": "in_t", "role": "exploratory"},
}
BLOCK_ORDER = ("P", "S-VC", "S-NAT", "S-EDGE", "T")
BLOCKS = BLOCK_ORDER
assert set(BLOCK_ORDER) == set(BLOCK_DESIGN)
ORDER_SALT = "expA-order-v2|"
CELL_ID_FORMAT = "{block}|{item_id}|{variant}|{condition}|s{seed}"


def design_table():
    """BLOCK_DESIGN as plain JSON (run.json, the appendix and dry runs)."""
    return {"block_order": list(BLOCK_ORDER), "order_salt": ORDER_SALT, "cell_id_format": CELL_ID_FORMAT,
            "blocks": {block: {"cells": [list(cell) for cell in design["cells"]], "seeds": list(design["seeds"]),
                               "membership": design["membership"], "role": design["role"]}
                       for block, design in BLOCK_DESIGN.items()},
            "order": "block P seed by seed (11, 23, 47), items in sha256(order_salt + item_id) order, an item's "
                     "cells together in the listed order; then S-VC, S-NAT, S-EDGE and T at seed 11"}


def item_order_key(item_id):
    return hashlib.sha256((ORDER_SALT + item_id).encode("utf-8")).hexdigest()


# -- follow-up A3 (protocols/EXPA_A3_LEDGER.md): its block and design -------------
# Not in BLOCK_DESIGN or BLOCK_ORDER: design_table(), the default cells and every earlier run are unchanged. The A3
# design is LEDGER first, then the default blocks (frozen in run.json and cells.jsonl like any design, so the payload
# store still holds only design cells; an A3 launch selects --blocks LEDGER only).
LEDGER_PROTOCOL = "protocols/EXPA_A3_LEDGER.md"
LEDGER_BLOCK = "LEDGER"
LEDGER_SEEDS = (61, 73, 89)
LEDGER_DESIGN = "a3-ledger"
EXTENSION_BLOCK_DESIGN = {
    LEDGER_BLOCK: {"cells": (("vcomplete", "C1"), ("omit_node", "C1")), "seeds": LEDGER_SEEDS, "membership": None,
                   "role": "A3 evidence ledger: complete (vcomplete) and incomplete (omit_node) answers under C1 "
                           "(protocols/EXPA_A3_LEDGER.md)"}}
DESIGN_ORDERS = {None: BLOCK_ORDER, LEDGER_DESIGN: (LEDGER_BLOCK,) + BLOCK_ORDER}
assert not set(EXTENSION_BLOCK_DESIGN) & set(BLOCK_DESIGN) and not set(LEDGER_SEEDS) & set(SEEDS)


def block_design(block):
    """The design entry of ``block`` (a default block or an extension block); None when unknown."""
    return BLOCK_DESIGN.get(block) or EXTENSION_BLOCK_DESIGN.get(block)


def design_order(design=None):
    """The block order of a design: None is the default (BLOCK_ORDER), LEDGER_DESIGN is A3's."""
    if design not in DESIGN_ORDERS:
        raise ValueError(f"unknown design {design!r}")
    return DESIGN_ORDERS[design]


def design_table_for(design=None):
    """design_table() for a design: the default table byte for byte, or A3's (named, LEDGER block first)."""
    if design is None:
        return design_table()
    order = design_order(design)
    return {"name": design, "block_order": list(order), "order_salt": ORDER_SALT, "cell_id_format": CELL_ID_FORMAT,
            "blocks": {block: {"cells": [list(cell) for cell in block_design(block)["cells"]],
                               "seeds": list(block_design(block)["seeds"]),
                               "membership": block_design(block)["membership"], "role": block_design(block)["role"]}
                       for block in order},
            "order": "block LEDGER seed by seed (61, 73, 89), items in sha256(order_salt + item_id) order, an item's "
                     "two cells together in the listed order; then the default design's blocks in their order (frozen "
                     "with the design, never dispatched under the evidence-ledger option)"}


def block_cell(block, variant, condition):
    """True when (variant, condition) is a cell of ``block``."""
    design = block_design(block)
    return design is not None and (variant, condition) in design["cells"]


# -- prompts (D6, D10.5, D10.6) -------------------------------------------------
SYSTEM_PROMPT_TEMPLATE = (
    "You are a careful software engineer. You have read-only access to a checkout of the Python repository {repo} "
    "through the tools provided. Use the tools to inspect the repository when needed, and finish by calling "
    "submit_answer. You have at most {max_model_calls} responses; call submit_answer before they run out.")
USER_TASK_TEMPLATE = (
    "Task: find every place in the library source where the function `{qualname}` (defined in `{path}`, line "
    "{line}) is invoked, either called directly or passed as a callable that gets invoked, from inside another "
    "function or method. Consider only files under {package_dirs}. Ignore tests, docs, examples and module-level "
    "code.")
ANSWER_BLOCK_TEMPLATE = ("\n\nA code-graph index returned this answer for incoming `invokes` relationships of "
                         "`{identity}`:\n{rendered_json}")
RECOMMENDATION_SENTENCE = "A graph_query tool can list indexed callers of a function."
SUBMIT_SENTENCE = ("Submit each invocation site as a file path and line number with submit_answer, and state whether "
                   "you believe your list is complete (yes, no or unknown).")
NO_TOOL_CALL = "your response did not call a tool"
INVALID_RESPONSE_UNICODE = "your response contained invalid Unicode"
UNKNOWN_TOOL = "a tool call named a tool that does not exist"
FORMAT_ERROR = ("Format error: {reason}. Every response must call at least one of the available tools ({tools}). "
                "Call submit_answer when you are ready to finish.")
SUBMIT_REJECTED = "submit_answer rejected: {reason}. Fix the arguments and call submit_answer again."
TOOL_REJECTED_UNICODE = "{name} rejected: " + INVALID_UNICODE + "."
# Final-turn policy (protocols/EXPA_FINALTURN.md): fixed bytes, identical in every condition. Not part of
# prompt_templates(): run.json's prompts stay those of the prereg harness; the notice is recorded in episode_config.
FINAL_TURN_DEFAULT = "none"
FINAL_TURN_TEXTS = {"neutral": "Your previous tool call was processed normally.",
                    "deadline": "This is your last response. Call submit_answer now.",
                    # Follow-up A4 (protocols/EXPA_A4_DIRECTIVE.md): the two components of the deadline text.
                    "deadline_info": "This is your last response.",
                    "submit_only": "Call submit_answer now."}
FINAL_TURN_POLICIES = (FINAL_TURN_DEFAULT,) + tuple(FINAL_TURN_TEXTS)
FINAL_TURN_PREFIX = "\n\n[harness notice] "
FINAL_TURN_PROTOCOL = "protocols/EXPA_FINALTURN.md"
A4_DIRECTIVE_PROTOCOL = "protocols/EXPA_A4_DIRECTIVE.md"
# The protocol each notice policy is registered under (its config record names it).
FINAL_TURN_PROTOCOLS = {"neutral": FINAL_TURN_PROTOCOL, "deadline": FINAL_TURN_PROTOCOL,
                        "deadline_info": A4_DIRECTIVE_PROTOCOL, "submit_only": A4_DIRECTIVE_PROTOCOL}
assert set(FINAL_TURN_PROTOCOLS) == set(FINAL_TURN_TEXTS)
assert FINAL_TURN_TEXTS["deadline"] == FINAL_TURN_TEXTS["deadline_info"] + " " + FINAL_TURN_TEXTS["submit_only"]
# Evidence ledger (protocols/EXPA_A3_LEDGER.md): every rule below is frozen with the protocol and pinned by
# tests/test_expA_a3_ledger.py. Not part of prompt_templates(); the rules are recorded in episode_config.
LEDGER_DEFAULT = "none"
LEDGER_MODES = (LEDGER_DEFAULT, "off", "on")
LEDGER_HEADER = ("Candidate call sites seen in your earlier tool outputs (unverified; check before "
                 "submitting):")
LEDGER_EMPTY = "(none)"
LEDGER_ENTRY = "- {path}:{line}: {excerpt}"
LEDGER_MAX_ENTRIES = 20
LEDGER_MAX_BYTES = 2000          # UTF-8 bytes of the entry lines joined by "\n" (the header is not counted)
LEDGER_EXCERPT_CHARS = 120
# A printed reference in bash output: one whole output line "PATH:LINE:TEXT" (grep -n / rg -n with a file name).
LEDGER_PRINTED_LINE = re.compile(r"(?P<path>[A-Za-z0-9_.+@~/-]+\.py):(?P<line>[1-9][0-9]*):(?P<text>.*)")
# The target's name X (the last component of its qualname) as a call token: "X(" or ".X(" (spaces or tabs allowed
# before the parenthesis), not part of a longer identifier, and not a definition ("def X(").
LEDGER_CALL_TOKEN = r"(?<![A-Za-z0-9_]){name}[ \t]*\("
LEDGER_DEFINITION = re.compile(r"(?:^|[^A-Za-z0-9_])def[ \t]+$")
_LEDGER_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
# Submit-time review (protocols/EXPA_A5_REVIEW.md): fixed bytes, pinned by tests/test_expA_a5_review.py. Not part of
# prompt_templates(); the rules are recorded in episode_config.
SUBMIT_REVIEW_PROTOCOL = "protocols/EXPA_A5_REVIEW.md"
SUBMIT_REVIEW_DEFAULT = "none"
SUBMIT_REVIEW_MODES = (SUBMIT_REVIEW_DEFAULT, "neutral", "ledger")
SUBMIT_REVIEW_PREFIX = "[harness notice] "
SUBMIT_REVIEW_NEUTRAL = "Your answer was received. Before finishing, call submit_answer again with your final list."
SUBMIT_REVIEW_INSTRUCTION = ("Before finishing, review these candidate call sites, then call submit_answer again with "
                             "your final list.")
ANSWER_SOURCES = ("first_submit", "resubmit")
# Task framing (protocols/EXPA_A7_IMPACT.md): fixed bytes, pinned by tests/test_expA_a7_impact.py. Not part of
# prompt_templates(): run.json's prompts stay those of the prereg harness; the framing is recorded in episode_config.
TASK_FRAMING_PROTOCOL = "protocols/EXPA_A7_IMPACT.md"
TASK_FRAMING_DEFAULT = "default"
IMPACT_TASK_TEMPLATE = (
    "Task: the signature of the function `{qualname}` (defined in `{path}`, line {line}) is about to change. Find "
    "every place in the library source that must be updated: every place where it is invoked, either called directly "
    "or passed as a callable that gets invoked, from inside another function or method. Consider only files under "
    "{package_dirs}. Ignore tests, docs, examples and module-level code.")
TASK_TEMPLATES = {TASK_FRAMING_DEFAULT: USER_TASK_TEMPLATE, "impact": IMPACT_TASK_TEMPLATE}
TASK_FRAMINGS = tuple(TASK_TEMPLATES)
TASK_FRAMING_BUDGET = (8, 16384)     # (max_model_calls, context_budget): the only budget "impact" runs with
_SHA256 = re.compile(r"[0-9a-f]{64}")
_ATTEMPT_ORDINAL = re.compile(r"\.a([0-9]{3,})$")


def prompt_templates():
    """Every fixed prompt string, for run.json binding and the prereg appendix."""
    return {"system": SYSTEM_PROMPT_TEMPLATE, "user_task": USER_TASK_TEMPLATE, "answer_block": ANSWER_BLOCK_TEMPLATE,
            "recommendation": RECOMMENDATION_SENTENCE, "submit": SUBMIT_SENTENCE, "format_error": FORMAT_ERROR,
            "no_tool_call": NO_TOOL_CALL, "unknown_tool": UNKNOWN_TOOL,
            "invalid_response_unicode": INVALID_RESPONSE_UNICODE, "submit_rejected": SUBMIT_REJECTED,
            "tool_rejected_unicode": TOOL_REJECTED_UNICODE,
            "user_layout": "task paragraph [+ ' ' + recommendation (T-recommended)] [+ answer block (C1-C4a)] "
                           "+ '\\n\\n' + submit sentence"}


def final_turn_suffix(policy):
    """The exact suffix ``policy`` appends before the last response ("\\n\\n[harness notice] " + text); None for none."""
    if not isinstance(policy, str) or policy not in FINAL_TURN_POLICIES:
        raise ValueError("final_turn_policy must be one of " + ", ".join(FINAL_TURN_POLICIES))
    return None if policy == FINAL_TURN_DEFAULT else FINAL_TURN_PREFIX + FINAL_TURN_TEXTS[policy]


def final_turn_record(policy, max_model_calls):
    """What a config records for a policy other than none: its protocol, exact bytes and position."""
    text = FINAL_TURN_TEXTS[policy]
    return {"protocol": FINAL_TURN_PROTOCOLS[policy], "channel": "in_band", "prefix": FINAL_TURN_PREFIX, "text": text,
            "text_sha256": _sha(text), "suffix_sha256": _sha(final_turn_suffix(policy)),
            "call_index": max_model_calls - 1,
            "appended_to": "the content of the last message before that call: the last model-visible tool result, "
                           "or the format-error message after a format error; never sent after a submission"}


def evidence_ledger_record(mode, max_model_calls):
    """What a config records for an evidence-ledger arm ("off" or "on"): its protocol and every frozen rule."""
    if mode not in LEDGER_MODES or mode == LEDGER_DEFAULT:
        raise ValueError("an evidence-ledger record needs mode off or on")
    return {"protocol": LEDGER_PROTOCOL, "mode": mode, "delivered_to_model": mode == "on", "channel": "in_band",
            "prefix": FINAL_TURN_PREFIX, "header": LEDGER_HEADER, "empty": LEDGER_EMPTY, "entry": LEDGER_ENTRY,
            "max_entries": LEDGER_MAX_ENTRIES, "max_bytes": LEDGER_MAX_BYTES, "excerpt_chars": LEDGER_EXCERPT_CHARS,
            "printed_line": LEDGER_PRINTED_LINE.pattern, "call_token": LEDGER_CALL_TOKEN,
            "definition": LEDGER_DEFINITION.pattern, "call_index": max_model_calls - 1,
            "sources": "the model-visible tool results already in the history at the last call, in order: each "
                       "source_reference_search hit (path, line, text) in hit order, and each bash output line that "
                       "is exactly PATH:LINE:TEXT; paths normalized as submit_answer does (/repo/ and ./ stripped), "
                       "absolute or '..' paths skipped",
            "selection": "TEXT contains the target name (last qualname component) as a call token that is not a "
                         "definition; deduplicated by (path, line), first appearance kept, in order of first "
                         "appearance; excerpt = TEXT with control characters as spaces, stripped, first 120 "
                         "characters; entries added while at most 20 and at most 2,000 bytes",
            "placement": "appended to the same message as the final-turn notice, before it"}


def submit_review_record(mode, max_model_calls):
    """What a config records for a submit-review arm ("neutral" or "ledger"): its protocol and every frozen rule."""
    if mode not in SUBMIT_REVIEW_MODES or mode == SUBMIT_REVIEW_DEFAULT:
        raise ValueError("a submit-review record needs mode neutral or ledger")
    value = {"protocol": SUBMIT_REVIEW_PROTOCOL, "mode": mode, "channel": "tool_result", "prefix": SUBMIT_REVIEW_PREFIX,
             "before_call_index": max_model_calls - 1,
             "when": "the first well-formed submit_answer (accepted by the submission validation) at a call_index "
                     "before the last call: it is recorded, not accepted, and its tool result is the prefix + TEXT",
             "resubmission": "the next well-formed submit_answer is accepted as final; there is no second review",
             "no_resubmission": "an episode that ends without one (NoAnswer, ContextLimit, FormatExhausted) is "
                                "Submitted with the first submission; InfraInterrupted stays infrastructure",
             "last_call": "a first submit_answer at the last call is accepted at once (no review)"}
    if mode == "neutral":
        value.update(text=SUBMIT_REVIEW_NEUTRAL, text_sha256=_sha(SUBMIT_REVIEW_NEUTRAL))
    else:
        value.update(instruction=SUBMIT_REVIEW_INSTRUCTION, instruction_sha256=_sha(SUBMIT_REVIEW_INSTRUCTION),
                     text="the evidence ledger text + '\\n' + instruction", header=LEDGER_HEADER, empty=LEDGER_EMPTY,
                     entry=LEDGER_ENTRY, max_entries=LEDGER_MAX_ENTRIES, max_bytes=LEDGER_MAX_BYTES,
                     excerpt_chars=LEDGER_EXCERPT_CHARS, printed_line=LEDGER_PRINTED_LINE.pattern,
                     call_token=LEDGER_CALL_TOKEN, definition=LEDGER_DEFINITION.pattern,
                     sources="the model-visible tool results already in the history when the first submission is "
                             "made (protocols/EXPA_A3_LEDGER.md sources and selection, unchanged)")
    return value


def task_framing_record(framing):
    """What a config records for a task framing other than default: its protocol, exact template and sha256."""
    if framing not in TASK_FRAMINGS or framing == TASK_FRAMING_DEFAULT:
        raise ValueError("a task-framing record needs framing impact")
    text = TASK_TEMPLATES[framing]
    return {"protocol": TASK_FRAMING_PROTOCOL, "framing": framing, "text": text, "text_sha256": _sha(text),
            "replaces": "the task paragraph (USER_TASK_TEMPLATE), with the same placeholders; the system prompt, "
                        "answer block, recommendation and submit sentences, budgets and tools are unchanged"}


def submit_review_text(mode, messages, target):
    """(TEXT, the ledger or None) of a review: the neutral sentence, or the ledger of ``messages`` + the instruction."""
    if mode == "neutral":
        return SUBMIT_REVIEW_NEUTRAL, None
    if mode != "ledger":
        raise ValueError("a review needs mode neutral or ledger")
    built = evidence_ledger(messages, target)
    return built["text"] + "\n" + SUBMIT_REVIEW_INSTRUCTION, built


def _ledger_excerpt(text):
    return _visible_text(_LEDGER_CONTROL.sub(" ", text).strip()[:LEDGER_EXCERPT_CHARS])


def _ledger_path(path):
    """A printed path normalized as submit_answer normalizes it; None when it cannot name a repository file."""
    path = normalize_site_path(path)
    if not path or path.startswith("/") or ".." in path.split("/") or not utf8_encodable(path):
        return None
    return path


def _printed_references(content):
    """(path, line, text, kind) of every printed reference in one model-visible tool result, in printed order."""
    try:
        value = json.loads(content)
    except (TypeError, ValueError):
        return
    if not isinstance(value, dict):
        return
    if value.get("kind") == LEXICAL_KIND and isinstance(value.get("hits"), list):
        for hit in value["hits"]:
            if isinstance(hit, dict) and isinstance(hit.get("path"), str) and type(hit.get("line")) is int \
                    and hit["line"] >= 1 and isinstance(hit.get("text"), str):
                yield hit["path"], hit["line"], hit["text"], "search_hit"
    elif "exit_code" in value and isinstance(value.get("output"), str):
        for raw in value["output"].split("\n"):
            match = LEDGER_PRINTED_LINE.fullmatch(raw[:-1] if raw.endswith("\r") else raw)
            if match:
                yield match["path"], int(match["line"]), match["text"], "bash_line"


def _is_call(text, call):
    return any(not LEDGER_DEFINITION.search(text[:match.start()]) for match in call.finditer(text))


def evidence_ledger(messages, target):
    """The evidence ledger at the last call (protocols/EXPA_A3_LEDGER.md), from agent-visible text only.

    ``messages`` is the history the model has seen (only its role "tool" messages are read) and ``target`` the
    episode's target identity, whose qualname the task paragraph shows. Nothing else enters: no V, no C*, no item,
    dispatch or grading data.
    """
    name = split_identity(target)[1].rsplit(".", 1)[-1]
    call = re.compile(LEDGER_CALL_TOKEN.format(name=re.escape(name)))
    seen, candidates = set(), []
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") != "tool":
            continue
        for printed, line, text, kind in _printed_references(message.get("content")):
            path = _ledger_path(printed)
            if path is None or (path, line) in seen or not _is_call(text, call):
                continue
            seen.add((path, line))
            candidates.append({"path": path, "line": line, "excerpt": _ledger_excerpt(text),
                               "source": {"message_index": index, "tool_call_id": message.get("tool_call_id"),
                                          "kind": kind}})
    entries, lines = [], []
    for candidate in candidates:
        line = LEDGER_ENTRY.format(**candidate)
        if len(entries) >= LEDGER_MAX_ENTRIES or len("\n".join(lines + [line]).encode("utf-8")) > LEDGER_MAX_BYTES:
            break
        entries.append(candidate)
        lines.append(line)
    body = "\n".join(lines)
    text = LEDGER_HEADER + "\n" + (body if lines else LEDGER_EMPTY)
    return {"name": name, "text": text, "suffix": FINAL_TURN_PREFIX + text, "entries": entries,
            "candidates": len(candidates), "shown": len(entries), "body_bytes": len(body.encode("utf-8"))}


@dataclass(frozen=True)
class EpisodeConfig:
    max_model_calls: int = 8
    context_budget: int = 16384
    max_response_tokens: int = 4096
    min_response_tokens: int = 512
    temperature: float = 0.2
    max_consecutive_format_errors: int = 3
    final_turn_policy: str = FINAL_TURN_DEFAULT   # protocols/EXPA_FINALTURN.md; recorded only when not "none"
    evidence_ledger: str = LEDGER_DEFAULT         # protocols/EXPA_A3_LEDGER.md; recorded only when not "none"
    submit_review: str = SUBMIT_REVIEW_DEFAULT    # protocols/EXPA_A5_REVIEW.md; recorded only when not "none"
    task_framing: str = TASK_FRAMING_DEFAULT      # protocols/EXPA_A7_IMPACT.md; recorded only when not "default"

    def __post_init__(self):
        for name in ("max_model_calls", "context_budget", "max_response_tokens", "min_response_tokens",
                     "max_consecutive_format_errors"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.context_budget > 32768 or self.min_response_tokens > self.max_response_tokens:
            raise ValueError("episode budgets must fit the served 32,768-token context")
        if not isinstance(self.temperature, (int, float)) or not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if final_turn_suffix(self.final_turn_policy) is not None and self.max_model_calls < 2:
            raise ValueError("a final-turn notice follows the previous response: it needs max_model_calls >= 2")
        if not isinstance(self.evidence_ledger, str) or self.evidence_ledger not in LEDGER_MODES:
            raise ValueError("evidence_ledger must be one of " + ", ".join(LEDGER_MODES))
        if self.evidence_ledger != LEDGER_DEFAULT and self.final_turn_policy == FINAL_TURN_DEFAULT:
            raise ValueError("the evidence ledger goes with the final-turn notice: it needs a final_turn_policy")
        if not isinstance(self.submit_review, str) or self.submit_review not in SUBMIT_REVIEW_MODES:
            raise ValueError("submit_review must be one of " + ", ".join(SUBMIT_REVIEW_MODES))
        if self.submit_review != SUBMIT_REVIEW_DEFAULT and (self.final_turn_policy != "deadline"
                                                            or self.evidence_ledger != LEDGER_DEFAULT):
            raise ValueError("the submit review runs with the deadline notice and no last-call evidence ledger: it "
                             "needs final_turn_policy deadline and evidence_ledger none")
        if not isinstance(self.task_framing, str) or self.task_framing not in TASK_FRAMINGS:
            raise ValueError("task_framing must be one of " + ", ".join(TASK_FRAMINGS))
        if self.task_framing != TASK_FRAMING_DEFAULT and (
                (self.max_model_calls, self.context_budget) != TASK_FRAMING_BUDGET
                or self.final_turn_policy != FINAL_TURN_DEFAULT or self.evidence_ledger != LEDGER_DEFAULT
                or self.submit_review != SUBMIT_REVIEW_DEFAULT):
            raise ValueError("the impact task framing runs with the prereg budget (8 responses, 16384 tokens), no "
                             "final-turn notice, no evidence ledger and no submit review")

    def record(self):
        """The config as events, result.json, run.json and dry runs record it: asdict() of the six prereg fields, plus
        final_turn_policy and its notice only when the policy is not "none", evidence_ledger and its rules only
        when the ledger is not "none", submit_review and its rules only when the review is not "none", and
        task_framing and its text only when the framing is not "default" (a default record is the frozen one)."""
        value = asdict(self)
        policy = value.pop("final_turn_policy")
        ledger = value.pop("evidence_ledger")
        review = value.pop("submit_review")
        framing = value.pop("task_framing")
        if policy != FINAL_TURN_DEFAULT:
            value["final_turn_policy"] = policy
            value["final_turn_notice"] = final_turn_record(policy, self.max_model_calls)
        if ledger != LEDGER_DEFAULT:
            value["evidence_ledger"] = ledger
            value["evidence_ledger_rules"] = evidence_ledger_record(ledger, self.max_model_calls)
        if review != SUBMIT_REVIEW_DEFAULT:
            value["submit_review"] = review
            value["submit_review_rules"] = submit_review_record(review, self.max_model_calls)
        if framing != TASK_FRAMING_DEFAULT:
            value["task_framing"] = framing
            value["task_framing_rules"] = task_framing_record(framing)
        return value


def config_design(config):
    """The design a config runs: the A3 design under an evidence-ledger or submit-review arm, else the default."""
    return LEDGER_DESIGN if (config.evidence_ledger != LEDGER_DEFAULT
                             or config.submit_review != SUBMIT_REVIEW_DEFAULT) else None


class PayloadIntegrityError(ValueError):
    """The delivered payload is not the canonical JSON its sha256 names, or not the cell's answer."""


def verify_payload(payload: bytes, expected_sha256: str):
    if not isinstance(expected_sha256, str) or not _SHA256.fullmatch(expected_sha256):
        raise PayloadIntegrityError("payload sha256 must be 64 lowercase hex digits")
    if not isinstance(payload, bytes) or hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise PayloadIntegrityError("payload bytes do not match the recorded sha256")
    try:
        decoded = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PayloadIntegrityError("payload is not JSON") from error
    if not isinstance(decoded, dict) or canonical_json(decoded) != payload:
        raise PayloadIntegrityError("payload must be a canonical JSON object")
    return decoded


def check_payload_structure(condition, payload, target):
    """The payload answers this cell's target (incoming invokes) and has exactly its condition's blocks.

    C1: no guidance, no verification. C2/C3: guidance, no verification. C4a: the
    C1 answer plus a verification block (lexical, at most 20 hits), no guidance.
    """
    if not isinstance(payload, dict):
        raise PayloadIntegrityError("payload must be an object")
    query = payload.get("query")
    if not isinstance(query, dict) or query.get("symbol") != target:
        raise PayloadIntegrityError("payload answers a different symbol than the cell's target")
    if query.get("relation") != "invokes" or query.get("direction") != "incoming":
        raise PayloadIntegrityError("payload is not an incoming invokes answer")
    guidance, verification = payload.get("guidance"), payload.get("verification")
    if condition == "C1":
        ok = "guidance" not in payload and "verification" not in payload
    elif condition in ("C2", "C3"):
        ok = isinstance(guidance, str) and bool(guidance) and "verification" not in payload
    elif condition == "C4a":
        ok = ("guidance" not in payload and isinstance(verification, dict)
              and verification.get("kind") == LEXICAL_KIND and isinstance(verification.get("hits"), list)
              and len(verification["hits"]) <= SEARCH_MAX_HITS)
    else:
        raise PayloadIntegrityError(f"{condition} has no payload")
    if not ok:
        raise PayloadIntegrityError(f"payload blocks do not match condition {condition}")
    return payload


@dataclass(frozen=True)
class EpisodeInput:
    run_id: str
    cell_id: str
    attempt_id: str
    token: int
    block: str
    item_id: str
    family: str
    target: str
    target_line: int
    package_dirs: tuple
    variant: str
    condition: str
    seed: int
    wording_pair: str
    payload: bytes | None = None
    payload_sha256: str | None = None
    dispatch_sha256: str | None = None
    items_sha256: str | None = None

    def __post_init__(self):
        for name in ("run_id", "cell_id", "attempt_id", "item_id", "family"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be nonempty text")
        split_identity(self.target)
        if type(self.target_line) is not int or self.target_line < 1:
            raise ValueError("target_line must be a positive integer")
        if type(self.seed) is not int or type(self.token) is not int:
            raise ValueError("seed and token must be integers")
        if not isinstance(self.package_dirs, tuple) or not self.package_dirs:
            raise ValueError("package_dirs must be a nonempty tuple")
        _identity_package_dirs(list(self.package_dirs))
        if self.block not in BLOCK_DESIGN and self.block not in EXTENSION_BLOCK_DESIGN:
            raise ValueError("block must be one of " + ", ".join(BLOCK_ORDER + tuple(EXTENSION_BLOCK_DESIGN)))
        if not block_cell(self.block, self.variant, self.condition):
            raise ValueError(f"block {self.block} has no cell ({self.variant!r}, {self.condition!r})")
        if self.wording_pair not in WORDING_PAIRS:
            raise ValueError("wording_pair must be A or B")
        for name in ("dispatch_sha256", "items_sha256"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not _SHA256.fullmatch(value)):
                raise ValueError(f"{name} must be null or a sha256")
        if self.condition in PAYLOAD_CONDITIONS:
            if not isinstance(self.payload, bytes) or not isinstance(self.payload_sha256, str):
                raise ValueError("C1-C4a need a payload and its sha256")
            check_payload_structure(self.condition, verify_payload(self.payload, self.payload_sha256), self.target)
        elif self.payload is not None or self.payload_sha256 is not None:
            raise ValueError("C0 and the T arms have no answer block")

    @property
    def graph_query_offered(self):
        return self.condition in T_CONDITIONS

    def describe(self):
        value = {key: getattr(self, key) for key in ("run_id", "cell_id", "attempt_id", "token", "block", "item_id",
                                                      "family", "target", "target_line", "variant", "condition",
                                                      "seed", "wording_pair", "payload_sha256", "dispatch_sha256",
                                                      "items_sha256")}
        value["package_dirs"] = list(self.package_dirs)
        value["payload_bytes"] = None if self.payload is None else len(self.payload)
        return value


def format_package_dirs(package_dirs):
    shown = []
    for directory in _identity_package_dirs(list(package_dirs)):
        if directory == "":
            shown.append("`./`")
        elif directory.endswith(".py"):
            shown.append(f"`{directory}`")
        else:
            shown.append(f"`{directory}/`")
    return ", ".join(shown)


def render_messages(episode: EpisodeInput, config: EpisodeConfig):
    """The fixed system message and user template; only the answer block and the T sentence differ by condition. The
    task paragraph is the config's task framing (USER_TASK_TEMPLATE by default)."""
    path, qualname = split_identity(episode.target)
    task = TASK_TEMPLATES[config.task_framing].format(qualname=qualname, path=path, line=episode.target_line,
                                                      package_dirs=format_package_dirs(episode.package_dirs))
    recommendation = episode.condition == "T-recommended"
    if recommendation:
        task += " " + RECOMMENDATION_SENTENCE
    if episode.condition in PAYLOAD_CONDITIONS:
        payload_text = episode.payload.decode("utf-8")
        block = ANSWER_BLOCK_TEMPLATE.format(identity=episode.target, rendered_json=payload_text)
    else:
        payload_text, block = None, ""
    user = task + block + "\n\n" + SUBMIT_SENTENCE
    tools = tool_schemas(graph_query=episode.graph_query_offered)
    return {"system": SYSTEM_PROMPT_TEMPLATE.format(repo=episode.family, max_model_calls=config.max_model_calls),
            "user": user, "task_paragraph": task, "answer_block": block, "payload_text": payload_text,
            "recommendation_present": recommendation, "tools": tools,
            "tools_offered": [tool["function"]["name"] for tool in tools]}


def _sha(text) -> str:
    return hashlib.sha256(text if isinstance(text, bytes) else text.encode("utf-8")).hexdigest()


_CODE_IDENTITY = None


def code_identity():
    """sha256 of the modules this process runs, hashed once per process (what was imported is what runs)."""
    global _CODE_IDENTITY
    if _CODE_IDENTITY is None:
        probe = Path(__file__).resolve().parent
        files = {"absent_edges/probe/" + name: probe / name for name in ("harness.py", "tools.py", "ledger.py",
                                                                        "identity.py")}
        files.update({"absent_edges/" + name: probe.parent / name
                      for name in ("models.py", "data.py", "graphs.py", "registry.py")})
        runner = probe.parent.parent / "scripts" / "expA_run.py"
        if runner.is_file():
            files["scripts/expA_run.py"] = runner
        _CODE_IDENTITY = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sorted(files.items())}
    return dict(_CODE_IDENTITY)


class EpisodeEventLog:
    """Like absent_edges.agent.EventLog: exclusive create, flush + fsync per record."""

    def __init__(self, path):
        self.path = Path(path)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        self.stream = os.fdopen(os.open(self.path, flags, 0o644), "w", encoding="utf-8", newline="\n")
        self.sequence, self.digest, self.closed = 0, hashlib.sha256(), False

    def write(self, kind, value):
        row = {"sequence": self.sequence, "wall_time": time.time(), "kind": kind, "value": value}
        line = json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
        try:
            encoded = line.encode("utf-8")
        except UnicodeEncodeError:  # lone surrogates from the model are kept, ASCII-escaped
            line = json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n"
            encoded = line.encode("utf-8")
        self.stream.write(line)
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.digest.update(encoded)
        self.sequence += 1

    def close(self):
        if not self.closed:
            self.stream.close()
            self.closed = True
            os.chmod(self.path, 0o444)
            fsync_directory(self.path.parent)
        return {"path": self.path.name, "sha256": self.digest.hexdigest(), "records": self.sequence}


class _Infra(Exception):
    def __init__(self, kind, message, detail=None):
        super().__init__(message)
        self.kind, self.detail = kind, detail or {}

    def record(self):
        return {"kind": self.kind, "message": str(self)[:4000], **self.detail}


_TRANSPORT_ERRORS = (OSError, http.client.HTTPException, ValueError)


def _transport_detail(error):
    detail = {"type": type(error).__name__}
    if getattr(error, "code", None) is not None:
        detail["http_status"] = error.code
    reason = getattr(error, "reason", None)
    if reason is not None:
        detail["reason"] = str(reason)[:500]
    return detail


def _verify_accounting(record, tokens, max_tokens):
    """The served prompt must be exactly the preflight prompt; usage and token ids must agree."""
    response = record.get("response") or {}
    usage = response.get("usage") or {}
    if any(type(usage.get(key)) is not int or usage[key] < 0 for key in ("prompt_tokens", "completion_tokens")):
        raise _Infra("accounting", "server did not report prompt and completion token usage")
    if usage["prompt_tokens"] != tokens["count"]:
        raise _Infra("accounting", "tokenize preflight count differs from the served prompt",
                     {"preflight": tokens["count"], "served": usage["prompt_tokens"]})
    prompt_ids = response.get("prompt_token_ids")
    completion_ids = response["choices"][0].get("token_ids")
    if (not isinstance(prompt_ids, list) or len(prompt_ids) != usage["prompt_tokens"]
            or not isinstance(completion_ids, list) or len(completion_ids) != usage["completion_tokens"]
            or any(type(token) is not int or token < 0 for token in prompt_ids + completion_ids)):
        raise _Infra("accounting", "prompt/completion token ids are missing or inconsistent with usage")
    if tokens["token_ids_sha256"] != token_ids_digest(prompt_ids):
        raise _Infra("accounting", "tokenize preflight token ids differ from the served prompt token ids")
    if usage["completion_tokens"] > max_tokens:
        raise _Infra("accounting", "server exceeded the requested max_tokens")
    return usage


def exposure_static(episode, prompt):
    """The receipt fields that need no server: what the prompt and tool list deliver (finding F8)."""
    payload = None if episode.payload is None else json.loads(episode.payload)
    verification = payload.get("verification") if isinstance(payload, dict) else None
    guidance = payload.get("guidance") if isinstance(payload, dict) else None
    if prompt["payload_text"] is not None and prompt["payload_text"] not in prompt["user"]:
        raise PayloadIntegrityError("the rendered user message does not contain the payload")
    return {"recorded_before_first_model_call": False, "block": episode.block, "condition": episode.condition,
            "variant": episode.variant, "wording_pair": episode.wording_pair,
            "payload_present": prompt["payload_text"] is not None,
            "payload_sha256": None if episode.payload is None else _sha(episode.payload),
            "expected_payload_sha256": episode.payload_sha256,
            "payload_bytes": 0 if episode.payload is None else len(episode.payload),
            "payload_tokens": None, "payload_token_counter": None,
            "verification_block_delivered": verification is not None,
            "verification_sha256": None if verification is None else _sha(canonical_json(verification)),
            "verification_hits": None if verification is None else len(verification.get("hits") or []),
            "verification_bytes": None if verification is None else len(canonical_json(verification)),
            "guidance_present": guidance is not None,
            "guidance_sha256": None if guidance is None else _sha(guidance),
            "tools_offered": list(prompt["tools_offered"]), "tools_sha256": _sha(canonical_json(prompt["tools"])),
            "graph_query_offered": GRAPH_QUERY in prompt["tools_offered"],
            "recommendation_present": prompt["recommendation_present"],
            "answer_block_sha256": _sha(prompt["answer_block"]),
            "answer_block_bytes": len(prompt["answer_block"].encode("utf-8")),
            "system_sha256": _sha(prompt["system"]), "user_sha256": _sha(prompt["user"]),
            "system_bytes": len(prompt["system"].encode("utf-8")), "user_bytes": len(prompt["user"].encode("utf-8")),
            "first_prompt_tokens": None, "first_prompt_token_ids_sha256": None}


def _exposure_receipt(static, prompt, tokens, counter):
    text = prompt["payload_text"]
    if text is None:
        payload_tokens, counter_name = 0, None
    else:
        try:
            payload_tokens = counter(text)
        except _TRANSPORT_ERRORS as error:
            raise _Infra("server", "payload token count failed: " + str(error), _transport_detail(error)) from error
        if type(payload_tokens) is not int or payload_tokens < 0:
            raise _Infra("accounting", "payload token counter returned an invalid count")
        counter_name = getattr(counter, "name", type(counter).__name__)
    return dict(static, recorded_before_first_model_call=True, payload_tokens=payload_tokens,
                payload_token_counter=counter_name, first_prompt_tokens=tokens["count"],
                first_prompt_token_ids_sha256=tokens["token_ids_sha256"])


def _requested_tool_names(message):
    """Raw tool names in a response that failed validation (bounded; for the T-arm request funnel)."""
    calls = message.get("tool_calls") if isinstance(message, dict) else None
    names = []
    for call in (calls if isinstance(calls, list) else [])[:TOOL_CALLS_PER_MESSAGE]:
        function = call.get("function") if isinstance(call, dict) else None
        name = function.get("name") if isinstance(function, dict) else None
        names.append(name if isinstance(name, str) and len(name) <= 128 and utf8_encodable(name) else None)
    return names


def _visible_text(value):
    """Assistant text for the history; each lone surrogate (never produced by vLLM) becomes U+FFFD."""
    if not isinstance(value, str):
        return ""
    if utf8_encodable(value):
        return value
    return "".join("�" if 0xD800 <= ord(char) <= 0xDFFF else char for char in value)


def _utc(timestamp):
    return datetime.datetime.fromtimestamp(timestamp, datetime.timezone.utc).isoformat()


def _ordinal(attempt_id):
    match = _ATTEMPT_ORDINAL.search(attempt_id)
    return int(match.group(1)) if match else None


def _append_final_turn_notice(messages, suffix, notice, events):
    """Append the notice in-band to the last model-visible message before the last call (protocols/EXPA_FINALTURN.md).

    That message is the last tool result of the previous response, or the format-error message when that response
    was a format error; nothing else can end an open episode's history. A new dict replaces it, so the
    model_visible_tool_result event already written keeps the tool's own output.
    """
    last = messages[-1]
    if last.get("role") == "tool":
        appended_to = "tool_result"
    elif last.get("role") == "user" and len(messages) > 2 and messages[-2].get("role") == "assistant":
        appended_to = "format_error"
    else:  # a controller defect, never a model outcome
        raise RuntimeError("the final-turn notice found no tool result or format-error message to follow")
    content = last["content"] + suffix
    messages[-1] = dict(last, content=content)
    notice.update(appended=True, appended_to=appended_to, tool_call_id=last.get("tool_call_id"))
    events.write("final_turn_notice", {"call_index": notice["call_index"], "policy": notice["policy"],
                                       "appended_to": appended_to, "tool_call_id": last.get("tool_call_id"),
                                       "suffix": suffix, "suffix_sha256": notice["suffix_sha256"],
                                       "text_sha256": notice["text_sha256"],
                                       "previous_content_sha256": _sha(last["content"]),
                                       "content_sha256": _sha(content)})


def _final_exposure(receipt, notice, ledger=None, review=None):
    """The result's exposure: the receipt logged before the first call, plus the final-turn receipt, the
    evidence-ledger receipt and the submit-review receipt when configured."""
    if notice is None:
        return receipt
    if ledger is None:
        exposure = dict(receipt, final_turn_notice=dict(notice))
    else:
        exposure = dict(receipt, final_turn_notice=dict(notice), evidence_ledger=dict(ledger))
    if review is not None:
        exposure["submit_review"] = dict(review)
    return exposure


def _submit_review_step(review, mode, messages, target, call_index, action, checked, events):
    """The first well-formed submission before the last call (protocols/EXPA_A5_REVIEW.md): recorded, not accepted.

    Returns the model-visible content of its tool result ("[harness notice] " + TEXT) and the tool record.
    """
    submission, validation = checked
    text, built = submit_review_text(mode, messages, target)
    content = SUBMIT_REVIEW_PREFIX + text
    review.update(reviewed=True, review_call_index=call_index, review_tool_call_id=action["id"],
                  text_sha256=_sha(text), content_sha256=_sha(content), first_call_index=call_index,
                  first_submission=submission, first_submission_validation=validation)
    event = {"call_index": call_index, "tool_call_id": action["id"], "mode": mode, "arguments": action["arguments"],
             "submission": submission, "validation": validation, "text": text, "text_sha256": review["text_sha256"],
             "content": content, "content_sha256": review["content_sha256"]}
    if built is not None:
        review.update(ledger_candidates=built["candidates"], ledger_entries=built["shown"],
                      ledger_body_bytes=built["body_bytes"])
        event["ledger"] = {"name": built["name"], "text": built["text"], "text_sha256": _sha(built["text"]),
                           "candidates": built["candidates"], "entries": built["entries"],
                           "body_bytes": built["body_bytes"]}
    events.write("submit_review", event)
    return content, {"status": "review", "mode": mode}


def _evidence_ledger_step(messages, target, receipt, events):
    """Build the ledger at the last call; under "on", append it in-band to the last message (EXPA_A3_LEDGER.md).

    Runs right before the final-turn notice is appended, so the notice follows the ledger on the same message. Under
    "off" the ledger is built and recorded only: the message, and so the request, is that of the notice alone.
    """
    last = messages[-1]
    if last.get("role") == "tool":
        appended_to = "tool_result"
    elif last.get("role") == "user" and len(messages) > 2 and messages[-2].get("role") == "assistant":
        appended_to = "format_error"
    else:  # a controller defect, never a model outcome
        raise RuntimeError("the evidence ledger found no tool result or format-error message to follow")
    built = evidence_ledger(messages, target)
    content = last["content"]
    if receipt["mode"] == "on":
        content = last["content"] + built["suffix"]
        messages[-1] = dict(last, content=content)
    receipt.update(computed=True, appended=receipt["mode"] == "on", appended_to=appended_to,
                   tool_call_id=last.get("tool_call_id"), candidates=built["candidates"], entries=built["shown"],
                   body_bytes=built["body_bytes"], text_sha256=_sha(built["text"]),
                   suffix_sha256=_sha(built["suffix"]))
    events.write("evidence_ledger", {"call_index": receipt["call_index"], "mode": receipt["mode"],
                                     "appended": receipt["appended"], "appended_to": appended_to,
                                     "tool_call_id": last.get("tool_call_id"), "name": built["name"],
                                     "text": built["text"], "text_sha256": receipt["text_sha256"],
                                     "suffix": built["suffix"], "suffix_sha256": receipt["suffix_sha256"],
                                     "candidates": built["candidates"], "entries": built["entries"],
                                     "body_bytes": built["body_bytes"],
                                     "previous_content_sha256": _sha(last["content"]),
                                     "content_sha256": _sha(content)})


def run_episode(episode: EpisodeInput, output_dir, *, client: LocalModel, toolbox, config: EpisodeConfig = None,
                payload_counter=None, stop_event=None, gpu_uuid=None, run_context=None):
    """Run one episode into ``output_dir`` (the attempt directory). Returns (result, result_sha256).

    ``toolbox.execute(name, arguments) -> (record, visible)`` serves
    source_reference_search, bash and (T arms) graph_query; submit_answer is
    handled here. Only InfraInterrupted results leave the cell eligible for
    another attempt.
    """
    config = config or EpisodeConfig()
    profile = client.profile
    if profile.enable_thinking is not False or profile.max_context != 32768:
        raise ValueError("Exp A requires enable_thinking=False and a served max_model_len of 32768")
    if episode.graph_query_offered and getattr(toolbox, "graph", None) is None:
        raise ValueError("T arms need a toolbox with the family's graph_query facts")
    output_dir = Path(output_dir)
    for name in ("events.jsonl", "result.json"):
        if (output_dir / name).exists():
            raise FileExistsError(f"{output_dir / name} already exists; attempts are immutable")
    prompt = render_messages(episode, config)
    tools = prompt["tools"]
    offered = tool_names(graph_query=episode.graph_query_offered)
    allowed = set(offered)
    static_receipt = exposure_static(episode, prompt)
    messages = [{"role": "system", "content": prompt["system"]}, {"role": "user", "content": prompt["user"]}]
    counter = payload_counter if payload_counter is not None else ServerTokenCounter(profile)
    accounting = {"model_calls": 0, "attempted_model_calls": 0, "format_errors": 0,
                  "consecutive_format_errors": 0, "tool_calls": {name: 0 for name in offered},
                  "executed_tool_calls": 0, "rejected_tool_calls": 0, "rejected_submissions": 0,
                  "invalid_unicode_arguments": 0, "not_executed_after_submission": 0,
                  "prompt_tokens": 0, "completion_tokens": 0, "max_prompt_tokens": 0,
                  "generation_usage_unknown": False, "finish_reasons": [], "finish_reason_length": 0,
                  "max_tokens_requested": [], "graph_query_requests_in_format_errors": 0}
    suffix, final_call, notice = final_turn_suffix(config.final_turn_policy), config.max_model_calls - 1, None
    if suffix is not None:  # the receipt (protocols/EXPA_FINALTURN.md); absent under the default policy
        notice = accounting["final_turn_notice"] = {
            "policy": config.final_turn_policy, "text_sha256": _sha(FINAL_TURN_TEXTS[config.final_turn_policy]),
            "suffix_sha256": _sha(suffix), "call_index": final_call, "appended": False, "sent": False,
            "delivered": False, "appended_to": None, "tool_call_id": None, "prompt_tokens": None, "max_tokens": None}
    ledger = None
    if config.evidence_ledger != LEDGER_DEFAULT:  # the receipt (protocols/EXPA_A3_LEDGER.md); absent by default
        ledger = accounting["evidence_ledger"] = {
            "mode": config.evidence_ledger, "call_index": final_call, "computed": False, "appended": False,
            "sent": False, "delivered": False, "appended_to": None, "tool_call_id": None, "candidates": None,
            "entries": None, "body_bytes": None, "text_sha256": None, "suffix_sha256": None}
    review = None
    if config.submit_review != SUBMIT_REVIEW_DEFAULT:   # the receipt (protocols/EXPA_A5_REVIEW.md); absent by default
        review = accounting["submit_review"] = {
            "mode": config.submit_review, "before_call_index": final_call, "reviewed": False,
            "review_call_index": None, "review_tool_call_id": None, "text_sha256": None, "content_sha256": None,
            "ledger_candidates": None, "ledger_entries": None, "ledger_body_bytes": None,
            "review_skipped_last_call": False, "first_call_index": None, "first_submission": None,
            "first_submission_validation": None, "answer_source": None, "resubmit_call_index": None,
            "resubmit_tool_call_id": None, "unresolved_exit": None}
    graph_calls = [] if episode.graph_query_offered else None
    exit_status, submission, validation, infra, receipt, reraise = None, None, None, None, None, None
    started_wall, started = time.time(), time.monotonic()
    events = EpisodeEventLog(output_dir / "events.jsonl")
    try:
        events.write("episode_setup", {
            "protocol": HARNESS_PROTOCOL, "cell": episode.describe(), "config": config.record(),
            "model_profile": asdict(profile), "tools": tools, "tools_sha256": _sha(canonical_json(tools)),
            "tools_offered": offered, "toolbox": toolbox.describe() if hasattr(toolbox, "describe") else None,
            "system": prompt["system"], "user": prompt["user"], "system_sha256": _sha(prompt["system"]),
            "user_sha256": _sha(prompt["user"]), "code": code_identity(), "gpu_uuid": gpu_uuid,
            "endpoint": profile.endpoint, "run_context": run_context})
        for call_index in range(config.max_model_calls):
            if stop_event is not None and stop_event.is_set():
                raise _Infra("stop_requested", "the runner requested a stop before the next model call")
            noticed = notice is not None and call_index == final_call
            if noticed:
                if ledger is not None:
                    _evidence_ledger_step(messages, episode.target, ledger, events)
                _append_final_turn_notice(messages, suffix, notice, events)
            try:
                tokens = client.tokenize(messages, tools)
            except _TRANSPORT_ERRORS as error:
                raise _Infra("server", "tokenize failed: " + str(error), _transport_detail(error)) from error
            events.write("context_accounting", {"call_index": call_index, **tokens})
            if noticed:
                notice["prompt_tokens"] = tokens["count"]   # the preflight counts the notice
            if receipt is None:
                receipt = _exposure_receipt(static_receipt, prompt, tokens, counter)
                events.write("exposure_receipt", receipt)
            remaining = config.context_budget - tokens["count"]
            if remaining < config.min_response_tokens:
                exit_status = EXIT_CONTEXT_LIMIT
                events.write("context_limit", {"call_index": call_index, "prompt_tokens": tokens["count"],
                                               "context_budget": config.context_budget, "remaining": remaining,
                                               "min_response_tokens": config.min_response_tokens})
                break
            max_tokens = min(config.max_response_tokens, remaining)
            accounting["attempted_model_calls"] += 1
            accounting["max_tokens_requested"].append(max_tokens)
            if noticed:
                notice.update(sent=True, max_tokens=max_tokens)
                if ledger is not None:
                    ledger["sent"] = ledger["appended"]
            try:
                record = client.chat(messages, tools, seed=episode.seed, max_tokens=max_tokens,
                                     temperature=config.temperature)
            except _TRANSPORT_ERRORS as error:
                accounting["generation_usage_unknown"] = True
                events.write("model_request_failed", {"call_index": call_index, "max_tokens": max_tokens,
                                                      "message": str(error)[:2000], **_transport_detail(error)})
                raise _Infra("server", "chat completion failed: " + str(error), _transport_detail(error)) from error
            events.write("model_exchange", {"call_index": call_index, "max_tokens": max_tokens, **record})
            usage = _verify_accounting(record, tokens, max_tokens)
            if noticed:
                notice["delivered"] = True   # the served prompt is the preflight prompt, notice included
                if ledger is not None:
                    ledger["delivered"] = ledger["appended"]
            accounting["model_calls"] += 1
            accounting["prompt_tokens"] += usage["prompt_tokens"]
            accounting["completion_tokens"] += usage["completion_tokens"]
            accounting["max_prompt_tokens"] = max(accounting["max_prompt_tokens"], usage["prompt_tokens"])
            choice = record["response"]["choices"][0]
            finish_reason = choice.get("finish_reason")
            accounting["finish_reasons"].append(finish_reason if isinstance(finish_reason, str) else None)
            if finish_reason == "length":
                accounting["finish_reason_length"] += 1
            message = choice["message"]
            try:
                actions = parsed_tool_calls(message, allowed)
                if not actions:
                    raise ValueError(NO_TOOL_CALL)
                assistant = {key: message[key] for key in ("role", "content", "tool_calls") if key in message}
                assistant["role"] = "assistant"
                if not utf8_encodable(assistant):
                    raise ValueError(INVALID_RESPONSE_UNICODE)
            except ValueError as error:
                accounting["format_errors"] += 1
                accounting["consecutive_format_errors"] += 1
                reason = str(error)
                if reason == "Tool not allowlisted":
                    reason = UNKNOWN_TOOL
                visible = FORMAT_ERROR.format(reason=reason[:500], tools=", ".join(offered))
                content = message.get("content") if isinstance(message, dict) else None
                requested = _requested_tool_names(message)
                accounting["graph_query_requests_in_format_errors"] += requested.count(GRAPH_QUERY)
                messages.append({"role": "assistant", "content": _visible_text(content)})
                messages.append({"role": "user", "content": visible})
                events.write("format_error", {"call_index": call_index, "reason": str(error)[:2000],
                                              "visible": visible, "requested_tools": requested,
                                              "finish_reason": finish_reason,
                                              "consecutive": accounting["consecutive_format_errors"]})
                if accounting["consecutive_format_errors"] >= config.max_consecutive_format_errors:
                    exit_status = EXIT_FORMAT_EXHAUSTED
                    break
                continue
            accounting["consecutive_format_errors"] = 0
            messages.append(assistant)
            events.write("parsed_actions", {"call_index": call_index, "actions": actions})
            for action in actions:
                name, arguments = action["name"], action["arguments"]
                if submission is not None:
                    accounting["not_executed_after_submission"] += 1
                    events.write("tool_action_not_executed", {"tool_call_id": action["id"], "name": name,
                                                              "reason": "submit_answer already accepted"})
                    continue
                events.write("tool_action", action)
                accounting["tool_calls"][name] += 1
                raw_content = None      # a submit review's tool result is plain text, not a JSON value
                if not utf8_encodable(arguments):  # finding F5: model visible, never a controller exception
                    accounting["invalid_unicode_arguments"] += 1
                    accounting["rejected_submissions" if name == "submit_answer" else "rejected_tool_calls"] += 1
                    tool_record = {"status": "rejected", "error": "invalid_unicode_arguments"}
                    visible = {"error": TOOL_REJECTED_UNICODE.format(name=name)}
                elif name == "submit_answer":
                    try:
                        checked = validate_submission(arguments)
                    except ValueError as error:
                        accounting["rejected_submissions"] += 1
                        tool_record = {"status": "rejected", "error": str(error)}
                        visible = {"error": SUBMIT_REJECTED.format(reason=error)}
                    else:
                        if review is not None and review["first_call_index"] is None and call_index < final_call:
                            raw_content, tool_record = _submit_review_step(review, config.submit_review, messages,
                                                                           episode.target, call_index, action,
                                                                           checked, events)
                        else:
                            submission, validation = checked
                            if review is not None and review["first_call_index"] is None:   # first, at the last call
                                review.update(review_skipped_last_call=True, first_call_index=call_index,
                                              first_submission=submission, first_submission_validation=validation,
                                              answer_source="first_submit")
                            elif review is not None:                                         # after the review
                                review.update(answer_source="resubmit", resubmit_call_index=call_index,
                                              resubmit_tool_call_id=action["id"])
                            events.write("submission_accepted", {"tool_call_id": action["id"],
                                                                 "call_index": call_index, "submission": submission,
                                                                 "validation": validation})
                            continue
                else:
                    tool_record, visible = toolbox.execute(name, arguments)
                    if tool_record.get("status") in ("rejected", "invalid_arguments"):
                        accounting["rejected_tool_calls"] += 1
                    else:
                        accounting["executed_tool_calls"] += 1
                content = canonical_json(visible).decode("utf-8") if raw_content is None else raw_content
                content_sha256 = _sha(content)
                if name == GRAPH_QUERY:
                    graph_calls.append({"call_index": call_index, "tool_call_id": action["id"],
                                        "status": tool_record.get("status"),
                                        "arguments": tool_record.get("arguments"),
                                        "state": tool_record.get("state"),
                                        "returned_count": tool_record.get("returned_count"),
                                        "symbol_indexed": tool_record.get("symbol_indexed"),
                                        "payload_truncated": tool_record.get("payload_truncated"),
                                        "answer_sha256": content_sha256,
                                        "answer_bytes": len(content.encode("utf-8"))})
                events.write("tool_output", {"tool_call_id": action["id"], "name": name, "record": tool_record})
                events.write("model_visible_tool_result", {"tool_call_id": action["id"], "name": name,
                                                           "content": content, "sha256": content_sha256})
                messages.append({"role": "tool", "tool_call_id": action["id"], "content": content})
            if submission is not None:
                exit_status = EXIT_SUBMITTED
                break
        else:
            exit_status = EXIT_NO_ANSWER
    except _Infra as error:
        exit_status, infra = EXIT_INFRA, error.record()
    except SandboxError as error:
        exit_status, infra = EXIT_INFRA, {"kind": "sandbox", "message": str(error)[:4000]}
    except KeyboardInterrupt as error:
        exit_status, infra, reraise = EXIT_INFRA, {"kind": "keyboard_interrupt", "message": "interrupted"}, error
    except Exception as error:  # a controller defect: preserved as evidence, flagged for a halt
        exit_status = EXIT_INFRA
        infra = {"kind": "controller_exception", "type": type(error).__name__, "message": str(error)[:4000],
                 "traceback": traceback.format_exc()[-8000:]}
    ended_wall = time.time()
    if review is not None and review["reviewed"] and review["answer_source"] is None \
            and exit_status in (EXIT_NO_ANSWER, EXIT_CONTEXT_LIMIT, EXIT_FORMAT_EXHAUSTED):
        # No resubmission before the episode ended: the first submission is the answer (protocols/EXPA_A5_REVIEW.md).
        review.update(answer_source="first_submit", unresolved_exit=exit_status)
        submission, validation = review["first_submission"], review["first_submission_validation"]
        exit_status = EXIT_SUBMITTED
    if exit_status != EXIT_SUBMITTED:
        submission, validation = None, None
    summary = {"exit_status": exit_status, "submission": submission, "submission_validation": validation,
               "accounting": accounting, "infra": infra, "graph_query_calls": graph_calls,
               "duration_s": time.monotonic() - started}
    try:
        if review is not None and review["unresolved_exit"] is not None:
            events.write("submit_review_resolved", {"answer_source": "first_submit",
                                                    "unresolved_exit": review["unresolved_exit"],
                                                    "submission": submission})
        events.write("episode_finished", summary)
    finally:
        events_record = events.close()
    result = {"schema": RESULT_SCHEMA, "protocol": HARNESS_PROTOCOL, "run_id": episode.run_id,
              "cell_id": episode.cell_id, "attempt_id": episode.attempt_id,
              "attempt": {"attempt_id": episode.attempt_id, "ordinal": _ordinal(episode.attempt_id)},
              "token": episode.token, "block": episode.block, "item_id": episode.item_id, "family": episode.family,
              "variant": episode.variant, "condition": episode.condition, "seed": episode.seed,
              "wording_pair": episode.wording_pair, "target": episode.target, "target_line": episode.target_line,
              "exit_status": exit_status, "submitted": submission, "submission_validation": validation,
              "calls": accounting["model_calls"],
              "tokens": {"prompt": accounting["prompt_tokens"], "completion": accounting["completion_tokens"]},
              "finish_reason_length_count": accounting["finish_reason_length"],
              "tool_calls": dict(accounting["tool_calls"]),
              "exposure": _final_exposure(receipt if receipt is not None else static_receipt, notice, ledger,
                                          review),
              "graph_query_calls": graph_calls, "endpoint": profile.endpoint, "gpu_uuid": gpu_uuid,
              "dispatch_sha256": episode.dispatch_sha256, "items_sha256": episode.items_sha256,
              "code": code_identity(), "cell": episode.describe(), "accounting": accounting,
              "prompt": {"system_sha256": _sha(prompt["system"]), "user_sha256": _sha(prompt["user"]),
                         "answer_block_sha256": _sha(prompt["answer_block"]),
                         "tools_sha256": _sha(canonical_json(tools))},
              "events": events_record, "infra": infra, "config": config.record(), "model_profile": asdict(profile),
              "run_context": run_context, "started_utc": _utc(started_wall), "ended_utc": _utc(ended_wall),
              "duration_s": summary["duration_s"]}
    result_sha256 = write_immutable_json(output_dir / "result.json", result)
    if reraise is not None:
        raise reraise
    return result, result_sha256


# ---------------------------------------------------------------------------
# Operational appendix (D11.6; prereg Appendix A is generated from this)
# ---------------------------------------------------------------------------

def operational_appendix():
    """The frozen operational details of an episode, generated from this code."""
    from . import tools as tools_module
    config = EpisodeConfig()
    return {
        "protocol": HARNESS_PROTOCOL, "result_schema": RESULT_SCHEMA, "design": design_table(),
        "prompts": prompt_templates(), "episode_config": config.record(),
        "budget_rules": [
            f"Every model response counts toward the limit of {config.max_model_calls}, including responses that "
            "are format errors.",
            f"Before every call the full serialized request (messages, tools, chat template, generation prompt; "
            f"enable_thinking=False) is counted with the server's /tokenize. remaining = {config.context_budget} "
            "- that count.",
            f"ContextLimit when remaining < {config.min_response_tokens} before a call; otherwise max_tokens = "
            f"min({config.max_response_tokens}, remaining).",
            "The served prompt must have exactly the preflight token count and token ids, and the completion may "
            "not exceed max_tokens; otherwise the attempt is InfraInterrupted (accounting).",
            f"FormatExhausted after {config.max_consecutive_format_errors} consecutive format errors; NoAnswer "
            f"when {config.max_model_calls} responses pass without an accepted submission.",
            "finish_reason == 'length' is counted per episode (finish_reason_length_count).",
            f"temperature {config.temperature}; the cell's seed; tool_choice 'auto'; return_token_ids true."],
        "response_handling": [
            "A response must contain at least one native tool call that vLLM's gemma4 parser returns. A response "
            "with no tool call, a malformed call, a call to a tool that is not offered, more than "
            f"{TOOL_CALLS_PER_MESSAGE} calls, arguments that are not a JSON object of at most {TOOL_ARGUMENT_BYTES} "
            "bytes, or invalid Unicode is a format error: its text (without the calls) is kept in the history and "
            "the format-error message is sent as a user message.",
            "Several tool calls in one response are run in their order. Each executed call gets its own tool "
            "message. Once a submit_answer is accepted, the remaining calls of that response are logged as not "
            "executed and the episode ends.",
            "Tool arguments that are not UTF-8 encodable (lone surrogates) are rejected with a model-visible tool "
            "error; they never reach a tool.",
            "A rejected submit_answer gets a model-visible reason and the episode continues."],
        "submission": tools_module.submission_policy(),
        "exits": {"completed": sorted(COMPLETED_EXITS), "infrastructure": EXIT_INFRA, "infra_kinds": list(INFRA_KINDS),
                  "retry_rule": "only InfraInterrupted attempts give a cell a new attempt (at most 3 per cell); a "
                                "model-caused outcome is never retried"},
        "tools": tools_module.operational_details(),
        "token_accounting_code_path": ["absent_edges/models.py LocalModel.tokenize (POST /tokenize with messages, "
                                       "tools, add_generation_prompt=True, add_special_tokens=False, "
                                       "chat_template_kwargs.enable_thinking=False)",
                                       "absent_edges/probe/harness.py run_episode (remaining, max_tokens, "
                                       "ContextLimit) and _verify_accounting",
                                       "absent_edges/probe/tools.py ServerTokenCounter (payload_tokens receipt: "
                                       "POST /tokenize with the payload text as prompt)"],
        "code": code_identity()}
