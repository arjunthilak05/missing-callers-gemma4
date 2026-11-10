"""Exp A guidance wording v2 (docs/EXPA_DESIGN_V2_DELTA.md D3, D4) and an offline token audit.

Both members of a pair share the identical action sentence; only the first statement
differs. C2 and C3 are ``first statement + " " + action sentence``. The v1 task-adapted
wording (substituted graphs.GUIDANCE / interventions.WORDING_PAIRS["B"]) is no longer
rendered anywhere.

Assignment (D3): within each family the selected items are ranked by
``sha256("expA-wording-v2|" + X)`` (ascending, 0-based); even ranks get pair A and odd
ranks pair B. One pair per item, for every seed and condition.

The offline audit counts tokens with the served model directory's tokenizer files and
chat template on CPU, mirroring ``interventions.audit_wording_tokens`` arithmetic. It is
evidence for the freeze; the served ``/tokenize`` audit
(reports/expA-serve-smoke-20261101/wording_live.json) is the authoritative one.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from ..data import canonical_json
from ..interventions import WORDING_TOLERANCE

WORDING_PROTOCOL = "expA-wording-v2"
WORDING_SALT = "expA-wording-v2"
CONDITIONS = ("C2", "C3")
PAIRS = ("A", "B")
ACTION_SENTENCES = {
    "A": "Inspect the relevant source and search for additional references before you submit.",
    "B": "Check the relevant source and look for further references before you finish.",
}
FIRST_STATEMENTS = {
    "A": {"C2": "Be careful with this answer. Review your assumptions and weigh the evidence before relying on it "
                "in your final list.",
          "C3": "This index answer is not an exhaustive list of callers. A missing relationship does not show that "
                "no other caller exists."},
    "B": {"C2": "Treat this evidence with caution. Question your assumptions and double-check your reasoning before "
                "you finalize the list.",
          "C3": "Treat these indexed relationships as partial evidence. An absent relationship does not mean that no "
                "further caller exists."},
}
# Filler-word revisions made after a failed audit (D4); none so far.
REVISIONS = ()


def _build_pairs():
    pairs = {pair: {condition: FIRST_STATEMENTS[pair][condition] + " " + ACTION_SENTENCES[pair]
                    for condition in CONDITIONS} for pair in PAIRS}
    for pair, texts in pairs.items():
        if not all(text.endswith(" " + ACTION_SENTENCES[pair]) for text in texts.values()):
            raise ValueError(f"Pair {pair}: C2 and C3 must end with the identical action sentence")
    return pairs


WORDING_PAIRS = _build_pairs()


def guidance(pair, condition):
    """Guidance text for wording pair A/B and condition C2/C3."""
    if pair not in WORDING_PAIRS:
        raise ValueError(f"Unknown wording pair {pair!r}")
    if condition not in CONDITIONS:
        raise ValueError("Guidance exists only for C2 and C3")
    return WORDING_PAIRS[pair][condition]


def _sha(value):
    return hashlib.sha256(value if isinstance(value, bytes) else canonical_json(value)).hexdigest()


def wording_sha256(pair):
    return _sha(WORDING_PAIRS[pair])


def assignment_key(target):
    return hashlib.sha256(f"{WORDING_SALT}|{target}".encode("utf-8")).hexdigest()


def assign_pairs(targets):
    """{target: "A"|"B"} for one family's selected targets: rank by key, even -> A, odd -> B."""
    targets = list(targets)
    if len(set(targets)) != len(targets):
        raise ValueError("Targets must be distinct")
    ranked = sorted(targets, key=lambda target: (assignment_key(target), target))
    return {target: PAIRS[rank % 2] for rank, target in enumerate(ranked)}


def longest_guidance_bytes():
    """Largest canonical {"guidance": text} size among all v2 texts."""
    return max(len(canonical_json({"guidance": text})) for texts in WORDING_PAIRS.values() for text in texts.values())


def wording_manifest():
    return {"protocol": WORDING_PROTOCOL, "action_sentences": ACTION_SENTENCES,
            "first_statements": FIRST_STATEMENTS, "pairs": WORDING_PAIRS,
            "composition": "C2/C3 = first statement + ' ' + the pair's shared action sentence",
            "pair_sha256": {pair: wording_sha256(pair) for pair in PAIRS},
            "assignment": {"salt": WORDING_SALT,
                           "rule": "within each family rank the selected items by sha256('expA-wording-v2|' + X) "
                                   "ascending (0-based); even rank -> A, odd rank -> B; one pair per item"},
            "revisions": list(REVISIONS), "tolerance": WORDING_TOLERANCE}


# ---------------------------------------------------------------- offline token audit

class OfflineGemmaTokenizer:
    """CPU-only counter over the model directory's tokenizer files and chat template.

    ``count_chat`` renders a single user message with ``add_generation_prompt=True``
    and ``enable_thinking=False`` (the serving request) and counts it with
    ``add_special_tokens=False``, as the served ``/tokenize`` request does.
    """

    def __init__(self, model_dir):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # tokenizer files only; never a device
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        self.model_dir = Path(model_dir)
        from tokenizers import Tokenizer  # Rust tokenizer, CPU only
        from transformers import AutoTokenizer
        self.raw = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
        self.hf = AutoTokenizer.from_pretrained(str(self.model_dir), local_files_only=True)
        import tokenizers
        import transformers
        self.versions = {"tokenizers": tokenizers.__version__, "transformers": transformers.__version__}
        self.files_sha256 = {name: _sha((self.model_dir / name).read_bytes())
                             for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
                             if (self.model_dir / name).is_file()}

    def count_text(self, text):
        hf_count = len(self.hf.encode(text, add_special_tokens=False))
        raw_count = len(self.raw.encode(text, add_special_tokens=False).ids)
        if hf_count != raw_count:
            raise ValueError("transformers and tokenizers disagree on a plain-text count")
        return hf_count

    def count_chat(self, content):
        rendered = self.hf.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                               add_generation_prompt=True, enable_thinking=False)
        return len(self.hf.encode(rendered, add_special_tokens=False))


def _within(difference, relative, tolerance):
    return difference <= tolerance["max_absolute_tokens"] and relative <= tolerance["max_relative_difference"]


def audit_token_counts(counter, examples=(), *, pairs=None, tolerance=WORDING_TOLERANCE):
    """C2 vs C3 token differences per pair, with any counter (``count_text``, ``count_chat``).

    Measured on (1) the texts (plain and as a chat message), (2) the live-audit mini
    payload ``{"guidance": text, "matches": []}`` and (3) full rendered payloads: each
    example is a C1 payload (no guidance) and C2/C3 are that payload plus the guidance,
    exactly as rendered. Relative differences use the interventions formula
    ``abs(d) / (min(chat C2, chat C3) - empty chat)``; the live-audit formula
    ``abs(d_text) / max(text C2, text C3)`` is also reported.
    """
    pairs = WORDING_PAIRS if pairs is None else pairs
    examples = [dict(payload) for payload in examples]
    for payload in examples:
        if "guidance" in payload or "verification" in payload:
            raise ValueError("Examples must be plain C1 payloads")
    empty = counter.count_chat("")
    result = {"scope": "offline_model_dir_tokenizer_and_chat_template_cpu_no_generation",
              "tolerance": tolerance, "empty_chat_template": empty, "pairs": {},
              "example_payload_sha256": [_sha(payload) for payload in examples]}
    for name, pair in pairs.items():
        text = {condition: counter.count_text(pair[condition]) for condition in CONDITIONS}
        chat = {condition: counter.count_chat(pair[condition]) for condition in CONDITIONS}
        mini = {condition: counter.count_chat(canonical_json({"guidance": pair[condition], "matches": []}).decode())
                for condition in CONDITIONS}
        difference = abs(chat["C2"] - chat["C3"])
        relative = difference / max(1, min(chat.values()) - empty)
        text_difference = abs(text["C2"] - text["C3"])
        rows = []
        for payload in examples:
            counts = {condition: counter.count_chat(canonical_json({**payload, "guidance": pair[condition]}).decode())
                      for condition in CONDITIONS}
            payload_difference = abs(counts["C2"] - counts["C3"])
            rows.append({"C2": counts["C2"], "C3": counts["C3"], "absolute_difference": payload_difference,
                         "relative_difference": payload_difference / max(1, min(counts.values()) - empty)})
        payload_max = max((row["absolute_difference"] for row in rows), default=None)
        payload_relative_max = max((row["relative_difference"] for row in rows), default=None)
        within = (_within(difference, relative, tolerance) and text_difference <= tolerance["max_absolute_tokens"]
                  and abs(mini["C2"] - mini["C3"]) <= tolerance["max_absolute_tokens"]
                  and (payload_max is None or _within(payload_max, payload_relative_max, tolerance)))
        result["pairs"][name] = {
            "wording_sha256": _sha(pair), "text_tokens": text, "chat_tokens": chat, "mini_payload_chat_tokens": mini,
            "text_absolute_difference": text_difference,
            "text_relative_difference_live_formula": text_difference / max(1, max(text.values())),
            "absolute_difference": difference, "relative_difference": relative,
            "mini_payload_absolute_difference": abs(mini["C2"] - mini["C3"]),
            "full_payloads": rows, "full_payload_max_absolute_difference": payload_max,
            "full_payload_max_relative_difference": payload_relative_max,
            "within_tolerance": within}
    return result


def offline_wording_audit(model_dir, examples=(), extra_pairs=None):
    """Token audit of the v2 pairs (and optional candidate pairs, reported separately)."""
    counter = OfflineGemmaTokenizer(model_dir)
    audit = audit_token_counts(counter, examples)
    result = {"protocol": WORDING_PROTOCOL, "model_dir": str(model_dir),
              "tokenizer_files_sha256": counter.files_sha256, "library_versions": counter.versions,
              "wording": wording_manifest(), "audit": audit,
              "all_pairs_within_tolerance": all(row["within_tolerance"] for row in audit["pairs"].values()),
              "note": "Offline CPU count with the served model directory's tokenizer and chat template; "
                      "the served /tokenize audit remains authoritative."}
    if extra_pairs:
        for name, pair in extra_pairs.items():
            if not set(CONDITIONS) <= set(pair) or not all(isinstance(pair[c], str) and pair[c] for c in CONDITIONS):
                raise ValueError(f"Candidate pair {name!r} needs non-empty C2 and C3 texts")
        result["extra_candidate_pairs"] = audit_token_counts(
            counter, examples, pairs={name: {c: pair[c] for c in CONDITIONS} for name, pair in extra_pairs.items()})
    return result
