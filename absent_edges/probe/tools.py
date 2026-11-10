"""Exp A agent tools (docs/EXPA_DESIGN.md section 7, docs/EXPA_DESIGN_V2_DELTA.md D6, D10.4, D10.6).

* ``lexical_search`` (exposed to the model as ``source_reference_search``)
  keeps the lexical semantics of ``graphs.GraphBackend.source_reference_search``
  (qualified and simple names, ``import ... as`` aliases, the same ranking and
  greedy prefix budgeting) but scans only package-scope library files and
  bounds the serialized result by a token budget. C4a verification blocks call
  this same function, so the live tool and the fallback share one
  implementation.
* ``ReadOnlyBash`` runs a validated pipeline of allowlisted read-only programs
  as argv lists (never through a shell) inside
  ``unshare --user --map-root-user --net --mount --pid``. The namespace root is
  a read-only tmpfs that contains only a read-only bind of ``/usr`` and the
  read-only repository checkout at ``/repo``; no project data, ledgers, home
  directories or ground truth are visible, and there is no network. Pathname
  expansion runs on the host with a no-follow walker bounded in entries and
  time, so symlinks can never lead it out of the checkout (finding F6).
* ``GraphQueryTool`` is ``graph_query`` for the T arms: ``graphs.GRAPH_TOOL``'s
  schema answered from the family's natural LocAgent facts through
  ``graphs.GraphFacts.query`` and rendered like C1 (no guidance, no
  verification) under the 12,000-byte cap.
* ``validate_submission`` is the ``submit_answer`` contract
  (``{"sites": [{"path", "line"}], "complete": "yes"|"no"|"unknown"}``; D11.5:
  invalid entries are kept for the scorer, never rejected one by one).
* ``checkout_tree`` / ``clean_checkout_entry`` / ``checkouts_manifest`` /
  ``compare_tree``: the clean checkout manifest (``expA-clean-checkouts-v1``,
  tree protocol ``expA-checkout-tree-v1``, shared with scripts/expA_stage.py)
  and the whole-tree check of every staged checkout (finding F11);
  ``stage_checkout`` copies a checkout to local scratch and drops symlinks that
  resolve outside it, recording them (finding F6).
* ``operational_details`` lists the frozen schemas, allowlist, rejection
  messages, truncation markers and timeouts (prereg Appendix A, D11.6).

Arguments that are not UTF-8 encodable (lone surrogates) are rejected with a
model-visible error and never reach a serializer (finding F5).

Nothing here reads hidden tests, gold patches, benchmark data or ground truth.
"""
from __future__ import annotations

import fnmatch
import glob as _glob
import hashlib
import json
import os
from pathlib import Path
import posixpath
import re
import selectors
import shutil
import signal
import stat
import subprocess
import sys
import time
import urllib.request

from ..data import canonical_json
from ..graphs import GRAPH_TOOL, RELATIONS, GraphFacts
from ..models import ModelProfile
from .identity import normalize_package_dirs as _identity_package_dirs
from .identity import scan_package_files, source_tree_sha256

TOOLS_PROTOCOL = "expA-tools-v2"
LEXICAL_KIND = "lexical_candidates_not_proven_edges"
SEARCH_MAX_HITS = 20
SEARCH_TOKEN_BUDGET = 2000
# Gemma's tokenizer is a byte-fallback BPE without a dummy prefix: every token
# consumes at least one UTF-8 byte of the input, so tokens <= UTF-8 bytes.
UTF8_BYTES_PER_TOKEN_LOWER_BOUND = 1
BASH_TIMEOUT_SECONDS = 30.0
BASH_OUTPUT_BYTES = 4000
SANDBOX_REPO = "/repo"
MAX_PIPELINE_STAGES = 8
MAX_COMMAND_BYTES = 8192
MAX_EXPANDED_ARGUMENTS = 4096
# The jail spec travels as one argv string; Linux limits one argument to 128 KiB (MAX_ARG_STRLEN).
MAX_SPEC_BYTES = 96 * 1024
GLOB_MAX_ENTRIES = 20000
GLOB_MAX_SECONDS = 5.0
GRAPH_QUERY_MAX_OUTPUT_BYTES = 12000
# graphs.GraphToolSet.graph_query / GraphFacts.query defaults, unchanged.
GRAPH_QUERY_DEFAULTS = {"relation": "invokes", "direction": "outgoing", "limit": 30}
MAX_SUBMITTED_SITES = 500
MAX_SITE_LINE = 999_999_999
INVALID_UNICODE = "the arguments contain invalid Unicode (lone surrogate code points); resend them as valid UTF-8 text"
OUTPUT_TRUNCATION_MARKER = "\n[output truncated: showing the first {shown} of {total} bytes]"
TIMEOUT_MESSAGE = "command timed out after {seconds:g} seconds"
REJECTION_PREFIX = "command rejected: "

SOURCE_REFERENCE_TOOL = {"type": "function", "function": {"name": "source_reference_search",
    "description": "Search the library's Python source (package scope only) for qualified names, names, and lexical aliases. Matches are candidate references, not proven call edges.",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"},
        "limit": {"type": "integer", "minimum": 1, "maximum": 20}},
        "required": ["query"], "additionalProperties": False}}}
BASH_TOOL = {"type": "function", "function": {"name": "bash",
    "description": "Run one read-only command in the repository checkout (working directory /repo). Allowed programs: grep, cat, head, tail, sed -n, ls, find, wc, sort, uniq, cut, nl, awk. Pipes between allowed programs are allowed; ;, &&, ||, redirections, subshells, backticks and $(...) are rejected. Commands time out after 30 seconds and output is capped at 4000 bytes.",
    "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
        "required": ["command"], "additionalProperties": False}}}
SUBMIT_ANSWER_TOOL = {"type": "function", "function": {"name": "submit_answer",
    "description": "Finish the task: submit every invocation site as a file path (relative to the repository root) and a line number, and state whether you believe the list is complete.",
    "parameters": {"type": "object", "properties": {
        "sites": {"type": "array", "items": {"type": "object", "properties": {
            "path": {"type": "string"}, "line": {"type": "integer", "minimum": 1}},
            "required": ["path", "line"], "additionalProperties": False}},
        "complete": {"type": "string", "enum": ["yes", "no", "unknown"]}},
        "required": ["sites", "complete"], "additionalProperties": False}}}
GRAPH_QUERY_TOOL = json.loads(json.dumps(GRAPH_TOOL))  # graphs.GRAPH_TOOL, byte for byte
TOOL_SCHEMAS = (SOURCE_REFERENCE_TOOL, BASH_TOOL, SUBMIT_ANSWER_TOOL)
TOOL_NAMES = tuple(tool["function"]["name"] for tool in TOOL_SCHEMAS)
GRAPH_QUERY = GRAPH_QUERY_TOOL["function"]["name"]
ALL_TOOL_NAMES = TOOL_NAMES + (GRAPH_QUERY,)


def tool_schemas(graph_query=False):
    """Fresh deep copies of the fixed schemas: the P/S tools, plus graph_query (last) for the T arms."""
    schemas = list(TOOL_SCHEMAS) + ([GRAPH_QUERY_TOOL] if graph_query else [])
    return json.loads(json.dumps(schemas))


def tool_names(graph_query=False):
    return ALL_TOOL_NAMES if graph_query else TOOL_NAMES


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def utf8_encodable(value) -> bool:
    """True when ``value`` serializes to UTF-8 (no lone surrogates anywhere, keys included)."""
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except UnicodeEncodeError:
        return False
    except (TypeError, ValueError):
        return False
    return True


def safe_echo(text, limit=200):
    """Model-visible echo of untrusted text: ASCII-escaped when it is not valid UTF-8 text."""
    if not isinstance(text, str):
        return None
    if not utf8_encodable(text):
        text = json.dumps(text, ensure_ascii=True)[1:-1]
    return text if len(text) <= limit else None


# ---------------------------------------------------------------------------
# Lexical source-reference search
# ---------------------------------------------------------------------------

_IDENTIFIER = re.compile(r"[A-Za-z_]\w*")


def normalize_package_dirs(package_dirs):
    """The shared identity normalization (sorted, no trailing slash, "" for the root)."""
    if isinstance(package_dirs, str):
        package_dirs = [package_dirs]
    if not isinstance(package_dirs, (list, tuple)) or not package_dirs:
        raise ValueError("package_dirs must be a nonempty list of relative directories")
    return _identity_package_dirs(list(package_dirs))


def _read_regular(root, relpath, budget):
    """Read one in-scope file without following a final symlink; bounded."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(Path(root) / relpath, flags)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > 16 * 1024**2:
            raise ValueError("invalid or oversized source file")
        raw = stream.read(16 * 1024**2 + 1)
    budget["files"] += 1
    budget["bytes"] += len(raw)
    if len(raw) > 16 * 1024**2 or budget["files"] > 20000 or budget["bytes"] > 128 * 1024**2:
        raise ValueError("source budget exceeded")
    return raw


def package_sources(repo_root, package_dirs):
    """Package-scope Python files: ({relpath: bytes}, {relpath: sha256}, tree sha256).

    The file set is exactly identity.scan_package_files (the scope every lane
    uses: package dirs, no tests/docs/examples/benchmarks/scripts, no symlinks).
    The tree hash is the graphs.source_inventory formula over those files.
    """
    normalize_package_dirs(package_dirs)
    root = Path(repo_root)
    scanned = scan_package_files(root, list([package_dirs] if isinstance(package_dirs, str) else package_dirs))
    if scanned["missing_package_dirs"]:
        raise FileNotFoundError("missing package directories: " + ", ".join(scanned["missing_package_dirs"]))
    budget = {"files": 0, "bytes": 0}
    files = {relpath: _read_regular(root, relpath, budget) for relpath in scanned["files"]}
    hashes = {path: _sha256(raw) for path, raw in files.items()}
    return files, hashes, source_tree_sha256(hashes)


def source_lines(raw: bytes):
    """Physical lines numbered like Python/grep (\\n, \\r\\n, \\r), not str.splitlines."""
    text = raw.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def split_query(query):
    if not isinstance(query, str) or not 1 <= len(query) <= 200:
        raise ValueError("query must be 1 to 200 characters")
    if not utf8_encodable(query):
        raise ValueError("query contains invalid Unicode")
    qualified = query.rsplit("::", 1)[-1]
    simple = qualified.rsplit(".", 1)[-1]
    if not _IDENTIFIER.fullmatch(simple):
        raise ValueError("query must end in a Python identifier")
    return qualified, simple


def lexical_candidates(files, query):
    """All ranked candidates, exactly GraphBackend.source_reference_search's rules."""
    qualified, simple = split_query(query)
    names = {qualified, simple}
    simple_pattern = re.compile(r"\b" + re.escape(simple) + r"\b")
    lines_by_file = {}
    for path, raw in files.items():
        lines = source_lines(raw)
        lines_by_file[path] = lines
        for line in lines:
            if "import" in line and simple_pattern.search(line):
                names.update(re.findall(r"\bas\s+([A-Za-z_]\w*)", line))
    patterns = [re.compile(r"(?<!\w)" + re.escape(name) + r"(?!\w)") for name in sorted(names)]
    candidates = []
    for path, lines in lines_by_file.items():
        for number, line in enumerate(lines, 1):
            if any(pattern.search(line) for pattern in patterns):
                candidates.append((2 if qualified in line else 1, path, number, line.strip()[:400]))
    candidates.sort(key=lambda row: (-row[0], row[1], row[2]))
    return {"qualified": qualified, "simple": simple, "names": sorted(names), "candidates": candidates}


def token_bound_name(token_counter):
    if token_counter is None:
        return "utf8_bytes_upper_bound"
    return "exact:" + str(getattr(token_counter, "name", type(token_counter).__name__))


def fits_token_budget(value, token_budget, token_counter=None):
    encoded = canonical_json(value)
    if token_counter is None:
        return len(encoded) <= token_budget * UTF8_BYTES_PER_TOKEN_LOWER_BOUND
    count = token_counter(encoded.decode("utf-8"))
    if type(count) is not int or count < 0:
        raise ValueError("token counter returned an invalid count")
    return count <= token_budget


def search_error(query, error):
    return {"query": safe_echo(query), "kind": LEXICAL_KIND, "hits": [], "truncated": False,
            "error": error, "source_sha256": None}


def lexical_search(repo_root, package_dirs, query, limit=SEARCH_MAX_HITS,
                   token_budget=SEARCH_TOKEN_BUDGET, token_counter=None):
    """Bounded package-scope lexical references for ``query``.

    Without ``token_counter`` the serialized result is at most ``token_budget``
    UTF-8 bytes, which bounds Gemma tokens from above. With a counter (for
    example ``ServerTokenCounter``) the exact count of the canonical JSON is
    bounded instead. Hits are the longest ranked prefix that fits, as in
    graphs.GraphBackend.source_reference_search.
    """
    if type(limit) is not int or not 1 <= limit <= SEARCH_MAX_HITS:
        raise ValueError("limit must be an integer between 1 and 20")
    if type(token_budget) is not int or not 256 <= token_budget <= 100000:
        raise ValueError("token_budget must be an integer between 256 and 100000")
    split_query(query)
    directories = normalize_package_dirs(package_dirs)
    files, _, source_sha256 = package_sources(repo_root, package_dirs)
    found = lexical_candidates(files, query)
    candidates = found["candidates"]
    base = {"query": query, "source_sha256": source_sha256, "kind": LEXICAL_KIND, "hits": [],
            "truncated": False, "error": None, "candidate_count": len(candidates),
            "scope": {"package_dirs": list(directories), "python_files": len(files)},
            "budget": {"max_hits": limit, "max_tokens": token_budget,
                       "token_bound": token_bound_name(token_counter)}}
    hits = [{"path": path, "line": number, "text": text}
            for _, path, number, text in candidates[:limit]]

    def proposal(count):
        return dict(base, hits=hits[:count], truncated=count < len(candidates))

    count = len(hits)
    if not fits_token_budget(proposal(count), token_budget, token_counter):
        count = 0
        while count < len(hits) and fits_token_budget(proposal(count + 1), token_budget, token_counter):
            count += 1
    result = proposal(count)
    if not fits_token_budget(result, token_budget, token_counter):
        raise ValueError("search metadata exceeds the token budget")
    if package_sources(repo_root, package_dirs)[2] != source_sha256:
        result.update(hits=[], error="source_changed_during_search")
    return result


source_reference_search = lexical_search


class ServerTokenCounter:
    """Exact served-tokenizer count of plain text via vLLM /tokenize (prompt form)."""
    name = "vllm_tokenize_prompt"

    def __init__(self, profile: ModelProfile, timeout: float = 60):
        if not isinstance(profile, ModelProfile):
            raise ValueError("ServerTokenCounter requires a validated loopback ModelProfile")
        self.profile, self.timeout = profile, timeout
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def __call__(self, text: str) -> int:
        if not isinstance(text, str):
            raise ValueError("token counter input must be text")
        body = json.dumps({"model": self.profile.model, "prompt": text, "add_special_tokens": False},
                          sort_keys=True, ensure_ascii=False).encode()
        request = urllib.request.Request(self.profile.endpoint.rstrip("/") + "/tokenize", body,
                                         {"Content-Type": "application/json"})
        with self.opener.open(request, timeout=self.timeout) as response:
            result = json.loads(response.read(64 * 1024**2))
        tokens = result.get("tokens")
        if (type(result.get("count")) is not int or not isinstance(tokens, list)
                or len(tokens) != result["count"]
                or result.get("max_model_len") != self.profile.max_context):
            raise ValueError("server returned an invalid plain-text token count")
        return result["count"]


class TokenizerFileCounter:
    """Exact local count from tokenizer.json (needs the `tokenizers` package; CPU only)."""
    name = "tokenizer_json"

    def __init__(self, tokenizer_json):
        from tokenizers import Tokenizer
        path = Path(tokenizer_json)
        self.sha256 = _sha256(path.read_bytes())
        self.tokenizer = Tokenizer.from_file(str(path))

    def __call__(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False).ids)


# ---------------------------------------------------------------------------
# submit_answer
# ---------------------------------------------------------------------------

_LINE_TEXT = re.compile(r"[0-9]{1,9}")
MAX_SITE_PATH_CHARS = 1024
SUBMISSION_KEYS = ("sites", "complete")
SITE_KEYS = ("path", "line")


def normalize_site_path(path):
    """Strip surrounding whitespace, a leading "/repo/" and leading "./" segments (contract item 5)."""
    text = path.strip()
    if text.startswith(SANDBOX_REPO + "/"):
        text = text[len(SANDBOX_REPO) + 1:]
    while text.startswith("./"):
        text = text[2:]
    return text


def _site_problem(site):
    """None when ``site`` is a valid {path, line} entry (extra keys allowed), else the reason it is not."""
    if not isinstance(site, dict):
        return "not_an_object"
    path, line = site.get("path"), site.get("line")
    if not isinstance(path, str) or len(path) > MAX_SITE_PATH_CHARS or not normalize_site_path(path):
        return "invalid_path"
    if isinstance(line, str) and _LINE_TEXT.fullmatch(line.strip()):
        line = int(line.strip())
    if type(line) is not int or not 1 <= line <= MAX_SITE_LINE:
        return "invalid_line"
    return None


def validate_submission(arguments):
    """Interpret submit_answer arguments; returns (submission, validation) or raises a visible ValueError.

    Contract item 5 and D11.5. Only the envelope can be rejected: it must be an
    object with exactly ``sites`` (a list of at most 500 entries) and
    ``complete`` ("yes", "no" or "unknown"; a JSON boolean maps to yes/no).
    Entries are never rejected one by one. A valid entry (an object with a path
    string and a line that is an integer >= 1 or a digit string) becomes
    ``{"path", "line"}``: the path loses surrounding whitespace, a leading
    "/repo/" and leading "./"; the line becomes an int; other keys are dropped.
    An invalid entry is kept exactly as submitted, so the scorer counts it as
    unresolved. Order and duplicates are kept (the scorer counts duplicates).
    """
    if not isinstance(arguments, dict):
        raise ValueError("arguments must be an object")
    if not utf8_encodable(arguments):
        raise ValueError(INVALID_UNICODE)
    unknown = set(arguments) - set(SUBMISSION_KEYS)
    if unknown:
        raise ValueError("unknown fields " + ", ".join(sorted(map(str, unknown))) + "; use sites and complete")
    if "sites" not in arguments or "complete" not in arguments:
        raise ValueError("sites and complete are both required")
    complete = arguments["complete"]
    if complete is True or complete is False:
        complete = "yes" if complete else "no"
    if complete not in ("yes", "no", "unknown"):
        raise ValueError('complete must be "yes", "no" or "unknown"')
    sites = arguments["sites"]
    if not isinstance(sites, list):
        raise ValueError("sites must be a list of {path, line} objects")
    if len(sites) > MAX_SUBMITTED_SITES:
        raise ValueError(f"sites is limited to {MAX_SUBMITTED_SITES} entries")
    normalized, invalid, changed, dropped = [], [], [], []
    for index, site in enumerate(sites):
        problem = _site_problem(site)
        if problem is not None:
            normalized.append(json.loads(json.dumps(site)))
            invalid.append({"index": index, "reason": problem})
            continue
        line = site["line"]
        value = {"path": normalize_site_path(site["path"]), "line": int(line.strip()) if isinstance(line, str) else line}
        if value != {"path": site["path"], "line": site["line"]}:
            changed.append(index)
        extra = sorted(str(key) for key in site if key not in SITE_KEYS)
        if extra:
            dropped.append({"index": index, "keys": extra})
        normalized.append(value)
    validation = {"entries": len(sites), "valid": len(sites) - len(invalid), "invalid": invalid,
                  "normalized_entries": changed, "dropped_keys": dropped,
                  "complete_from_boolean": isinstance(arguments["complete"], bool)}
    return {"sites": normalized, "complete": complete}, validation


def submission_policy():
    """The frozen submit_answer interpretation (prereg Appendix A)."""
    return {"schema": SUBMIT_ANSWER_TOOL["function"]["parameters"],
            "envelope": "an object with exactly sites (a list) and complete; anything else is rejected with a "
                        "visible reason and the model may call submit_answer again",
            "complete": "yes, no or unknown; a JSON boolean true/false is read as yes/no",
            "max_entries": MAX_SUBMITTED_SITES,
            "valid_entry": f"an object whose path is a string of at most {MAX_SITE_PATH_CHARS} characters that is "
                           "nonempty after normalization and whose line is an integer 1.."
                           f"{MAX_SITE_LINE} or a string of 1-9 digits",
            "normalization": "path: strip surrounding whitespace, one leading '/repo/' and every leading './'; "
                             "line: a digit string becomes an integer; keys other than path and line are dropped",
            "invalid_entries": "kept exactly as submitted (never rejected one by one); the scorer counts them as "
                               "unresolved",
            "order_and_duplicates": "kept"}


# ---------------------------------------------------------------------------
# graph_query (T arms)
# ---------------------------------------------------------------------------

_GRAPH_SCHEMA = GRAPH_QUERY_TOOL["function"]["parameters"]["properties"]


def graph_query_arguments(arguments):
    """Validate graph_query arguments against graphs.GRAPH_TOOL; defaults are GraphToolSet.graph_query's."""
    if not isinstance(arguments, dict):
        raise ValueError("arguments must be an object")
    if not utf8_encodable(arguments):
        raise ValueError(INVALID_UNICODE)
    unknown = set(arguments) - set(_GRAPH_SCHEMA)
    if unknown:
        raise ValueError("unknown fields " + ", ".join(sorted(map(str, unknown))))
    symbol = arguments.get("symbol")
    if not isinstance(symbol, str) or not symbol:
        raise ValueError("symbol must be a nonempty string (canonical path.py::qualified.name)")
    if len(symbol.encode("utf-8")) > 4096:
        raise ValueError("symbol is longer than 4096 bytes")
    value = {"symbol": symbol}
    for name in ("relation", "direction"):
        chosen = arguments.get(name, GRAPH_QUERY_DEFAULTS[name])
        if chosen not in _GRAPH_SCHEMA[name]["enum"]:
            raise ValueError(f"{name} must be one of " + ", ".join(_GRAPH_SCHEMA[name]["enum"]))
        value[name] = chosen
    limit = arguments.get("limit", GRAPH_QUERY_DEFAULTS["limit"])
    if type(limit) is not int or not _GRAPH_SCHEMA["limit"]["minimum"] <= limit <= _GRAPH_SCHEMA["limit"]["maximum"]:
        raise ValueError("limit must be an integer between 1 and 200")
    value["limit"] = limit
    return value


class GraphQueryTool:
    """graph_query over one family's natural LocAgent facts (D10.6), rendered like C1 under 12,000 bytes."""

    def __init__(self, facts_bytes: bytes, *, expected_sha256=None, max_output_bytes=GRAPH_QUERY_MAX_OUTPUT_BYTES):
        if not isinstance(facts_bytes, bytes):
            raise ValueError("graph facts must be bytes")
        digest = _sha256(facts_bytes)
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError(f"graph facts sha256 {digest} differs from the expected {expected_sha256}")
        document = json.loads(facts_bytes)
        if not isinstance(document, dict) or not isinstance(document.get("nodes"), list) \
                or not isinstance(document.get("edges"), list):
            raise ValueError("graph facts must be a GraphFacts document with nodes and edges")
        if document.get("backend") != "locagent" or document.get("backend") not in RELATIONS:
            raise ValueError("T-arm graph facts must be LocAgent facts")
        if document.get("error"):
            raise ValueError("graph facts carry an engine error: " + str(document["error"])[:200])
        ids = [node.get("id") for node in document["nodes"]]
        if len(set(ids)) != len(ids):
            raise ValueError("graph facts have duplicate node ids")
        known = set(ids)
        if any(edge.get("source") not in known or edge.get("target") not in known for edge in document["edges"]):
            raise ValueError("graph facts have an edge whose endpoint is not a node")  # GraphFacts.query would KeyError
        if type(max_output_bytes) is not int or max_output_bytes < 2048:
            raise ValueError("max_output_bytes must be an integer of at least 2048")
        self.facts = GraphFacts(facts_bytes)
        self.sha256, self.source_sha256 = digest, document.get("source_sha256")
        self.max_output_bytes = max_output_bytes
        self.counts = {"nodes": len(document["nodes"]), "edges": len(document["edges"])}

    def describe(self):
        return {"protocol": TOOLS_PROTOCOL, "facts_sha256": self.sha256, "source_sha256": self.source_sha256,
                "render": "C1", "max_output_bytes": self.max_output_bytes, "defaults": GRAPH_QUERY_DEFAULTS,
                **self.counts}

    def run(self, arguments):
        """Full record for the event log; ``visible`` is what the model receives."""
        try:
            query = graph_query_arguments(arguments)
        except ValueError as error:
            message = f"invalid_arguments: {error}"
            return {"status": "invalid_arguments", "error": message, "visible": {"error": message}}
        try:
            observation = self.facts.query(query["symbol"], query["relation"], query["direction"], query["limit"])
            visible = observation.render("C1", max_output_bytes=self.max_output_bytes)
        except ValueError as error:
            message = f"invalid_arguments: {error}"
            return {"status": "invalid_arguments", "arguments": query, "error": message, "visible": {"error": message}}
        return {"status": "executed", "arguments": query, "state": visible.get("state"),
                "returned_count": visible.get("returned_count"), "symbol_indexed": visible.get("symbol_indexed"),
                "payload_truncated": visible.get("payload_truncated"), "visible": visible}


# ---------------------------------------------------------------------------
# Read-only bash: validation
# ---------------------------------------------------------------------------

ALLOWED_PROGRAMS = ("grep", "rg", "cat", "head", "tail", "sed", "ls", "find", "wc", "sort",
                    "uniq", "cut", "nl", "awk")
FIND_FORBIDDEN = frozenset({"-exec", "-execdir", "-ok", "-okdir", "-delete", "-fprint",
                            "-fprint0", "-fprintf", "-fls"})
STAGE_ENVIRONMENT = {"PATH": "/usr/bin:/bin", "LC_ALL": "C", "LANG": "C"}


class CommandRejected(ValueError):
    """The command is outside the read-only contract; the reason is model visible."""


class SandboxError(RuntimeError):
    """The sandbox itself failed (infrastructure); never shown as a command result."""


def discover_programs(directories=("/usr/bin", "/bin")):
    """Map allowlisted names to real binaries under /usr (the only tree in the jail).

    awk is enabled only as GNU awk, because only gawk offers --sandbox.
    """
    programs = {}
    for name in ALLOWED_PROGRAMS:
        candidates = ["gawk", "awk"] if name == "awk" else [name]
        for candidate in candidates:
            for directory in directories:
                path = os.path.join(directory, candidate)
                if not os.path.exists(path):
                    continue
                real = os.path.realpath(path)
                if not real.startswith("/usr/") or not os.access(real, os.X_OK):
                    continue
                if name == "awk" and os.path.basename(real) != "gawk":
                    continue
                programs[name] = real
                break
            if name in programs:
                break
    return programs


_PIPE = object()


def _check_dollar(command, index, in_double):
    following = command[index + 1] if index + 1 < len(command) else ""
    if following in ("(", "{"):
        raise CommandRejected("$( ) and ${ } expansions are not allowed")
    if following.isalnum() or following in "_@*#?$!-":
        if following:
            raise CommandRejected("shell variable expansion is not supported")
    if not in_double and following in ("'", '"'):
        raise CommandRejected("$'...' and $\"...\" quoting is not supported")


def _lex(command):
    if not isinstance(command, str):
        raise CommandRejected("command must be a string")
    if "\x00" in command:
        raise CommandRejected("NUL bytes are not allowed")
    if not utf8_encodable(command):
        raise CommandRejected("the command contains invalid Unicode (lone surrogate code points)")
    if len(command.encode("utf-8")) > MAX_COMMAND_BYTES:
        raise CommandRejected(f"command is longer than {MAX_COMMAND_BYTES} bytes")
    if not command.strip():
        raise CommandRejected("empty command")
    tokens, word, index, size = [], None, 0, len(command)

    def finish():
        nonlocal word
        if word is not None:
            tokens.append(word)
            word = None

    while index < size:
        char = command[index]
        if char in " \t":
            finish()
            index += 1
            continue
        if char in "\n\r":
            raise CommandRejected("multiple command lines are not allowed; run one pipeline per call")
        if char == "|":
            if command.startswith("||", index):
                raise CommandRejected("'||' is not allowed; run one pipeline per call")
            if command.startswith("|&", index):
                raise CommandRejected("'|&' is not allowed")
            finish()
            tokens.append(_PIPE)
            index += 1
            continue
        if char == ";":
            raise CommandRejected("';' is not allowed; run one pipeline per call")
        if char == "&":
            raise CommandRejected("'&&' is not allowed; run one pipeline per call"
                                  if command.startswith("&&", index) else "'&' (background jobs) is not allowed")
        if char in "<>":
            raise CommandRejected("redirections are not allowed")
        if char in "()":
            raise CommandRejected("subshells and ( ) grouping are not allowed")
        if char == "`":
            raise CommandRejected("command substitution with backticks is not allowed")
        if char in "{}":
            raise CommandRejected("unquoted braces are not supported; quote the argument")
        if char == "#" and word is None:
            raise CommandRejected("shell comments are not supported")
        if char == "~" and word is None:
            raise CommandRejected("tilde expansion is not supported; use paths relative to the repository")
        if char == "$":
            _check_dollar(command, index, False)
        if char == "\\":
            if index + 1 >= size or command[index + 1] in "\n\r":
                raise CommandRejected("a trailing backslash is not allowed")
            word = (word or []) + [(command[index + 1], True)]
            index += 2
            continue
        if char == "'":
            end = command.find("'", index + 1)
            if end < 0:
                raise CommandRejected("unterminated single quote")
            word = (word or []) + [(c, True) for c in command[index + 1:end]]
            index = end + 1
            continue
        if char == '"':
            index += 1
            quoted = []
            while True:
                if index >= size:
                    raise CommandRejected("unterminated double quote")
                current = command[index]
                if current == '"':
                    index += 1
                    break
                if current == "\\" and index + 1 < size and command[index + 1] in '$`"\\':
                    quoted.append((command[index + 1], True))
                    index += 2
                    continue
                if current == "`":
                    raise CommandRejected("command substitution with backticks is not allowed")
                if current == "$":
                    _check_dollar(command, index, True)
                if current in "\n\r":
                    raise CommandRejected("multiple command lines are not allowed; run one pipeline per call")
                quoted.append((current, True))
                index += 1
            word = (word or []) + quoted
            continue
        word = (word or []) + [(char, False)]
        index += 1
    finish()
    return tokens


# -- pathname expansion: a bounded no-follow walker (finding F6) ---------------

class GlobBudget:
    """Directory entries examined and wall time for all pathname expansion of one command."""

    def __init__(self, max_entries=GLOB_MAX_ENTRIES, max_seconds=GLOB_MAX_SECONDS, clock=time.monotonic):
        self.entries, self.max_entries, self.clock = 0, max_entries, clock
        self.deadline = clock() + max_seconds

    def tick(self, count=1):
        self.entries += count
        if self.entries > self.max_entries:
            raise CommandRejected(f"pathname expansion examined more than {self.max_entries} directory entries; "
                                  "use a narrower pattern, find or grep -r")
        if self.clock() > self.deadline:
            raise CommandRejected("pathname expansion took too long; use a narrower pattern, find or grep -r")


def _has_magic(chars):
    return any(not quoted and c in "*?[" for c, quoted in chars)


def _component_matcher(chars):
    """bash/glob semantics for one path component: unquoted * ? [...] are magic; quoted text is literal."""
    pattern = "".join(_glob.escape(c) if quoted else c for c, quoted in chars)
    text = "".join(c for c, _ in chars)
    return re.compile(fnmatch.translate(pattern)), text.startswith(".")


def _lstat(path):
    try:
        return os.lstat(path)
    except OSError:
        return None


def _scan_names(directory, budget):
    try:
        with os.scandir(directory) as entries:
            found = []
            for entry in entries:
                budget.tick()
                try:
                    is_dir = entry.is_dir(follow_symlinks=False)
                except OSError:
                    is_dir = False
                found.append((entry.name, is_dir))
    except OSError:
        return []
    return sorted(found)


def _walk_pattern(root, components, budget):
    """Expand path components below ``root`` without following any symlink."""
    frontier = [("", str(root))]
    for index, chars in enumerate(components):
        last = index == len(components) - 1
        following = []
        if not _has_magic(chars):
            name = "".join(c for c, _ in chars)
            for relative, directory in frontier:
                budget.tick()
                if name == "":
                    following.append((relative if last else relative + "/", directory))
                    continue
                target = os.path.join(directory, name)
                info = _lstat(target)
                if info is None:
                    continue
                if last:
                    following.append((relative + name, target))
                elif stat.S_ISDIR(info.st_mode):  # lstat: a symlink is never descended
                    following.append((relative + name + "/", target))
        else:
            matcher, explicit_dot = _component_matcher(chars)
            for relative, directory in frontier:
                for name, is_dir in _scan_names(directory, budget):
                    if name.startswith(".") and not explicit_dot:
                        continue
                    if not utf8_encodable(name) or not matcher.match(name):
                        continue
                    if last:
                        following.append((relative + name, os.path.join(directory, name)))
                    elif is_dir:
                        following.append((relative + name + "/", os.path.join(directory, name)))
        frontier = following
        if len(frontier) > MAX_EXPANDED_ARGUMENTS:
            raise CommandRejected(f"a pattern expands to more than {MAX_EXPANDED_ARGUMENTS} paths")
        if not frontier:
            break
    return sorted(relative for relative, _ in frontier)


def _expand(chars, checkout, budget=None):
    """bash-like pathname expansion against the checkout (nullglob off, dotfiles hidden, no symlinks followed)."""
    text = "".join(c for c, _ in chars)
    if checkout is None or not _has_magic(chars):
        return [text]
    components, current = [], []
    for char, quoted in chars:
        if char == "/":
            components.append(current)
            current = []
        else:
            current.append((char, quoted))
    components.append(current)
    names = ["".join(c for c, _ in component) for component in components]
    prefix = ""
    if text.startswith("/"):
        if len(names) < 3 or names[1] != SANDBOX_REPO[1:]:
            return [text]  # only /repo exists in the jail; other absolute patterns stay literal
        prefix, components, names = SANDBOX_REPO + "/", components[2:], names[2:]
    if ".." in names:
        return [text]  # in the jail ".." of /repo is the jail root, never the host parent
    budget = budget or GlobBudget()
    matches = _walk_pattern(Path(os.path.realpath(checkout)), components, budget)
    if not matches:
        return [text]
    return [prefix + match for match in matches]


def _long_name(argument):
    return argument[2:].split("=", 1)[0]


def _forbidden_long(argument, forbidden):
    """getopt_long accepts unambiguous prefixes, so any prefix of a forbidden name is refused."""
    name = _long_name(argument)
    return bool(name) and any(option.startswith(name) for option in forbidden)


def _scan_options(args, *, takes_argument, forbidden_short, forbidden_long, long_with_argument=()):
    """Walk GNU-style options (permuted, until --). Returns non-option operands."""
    operands, index = [], 0
    while index < len(args):
        argument = args[index]
        if argument == "--":
            operands.extend(args[index + 1:])
            break
        if argument.startswith("--"):
            if _forbidden_long(argument, forbidden_long):
                raise CommandRejected(f"option {argument.split('=', 1)[0]} is not allowed")
            if "=" not in argument and _long_name(argument) in long_with_argument:
                index += 1
        elif argument.startswith("-") and argument != "-":
            for position, flag in enumerate(argument[1:], 1):
                if flag in forbidden_short:
                    raise CommandRejected(f"option -{flag} is not allowed")
                if flag in takes_argument:
                    if position == len(argument) - 1:
                        index += 1
                    break
        else:
            operands.append(argument)
        index += 1
    return operands


def _check_sed(args):
    quiet, index = False, 0
    while index < len(args):
        argument = args[index]
        if argument == "--":
            break
        if argument.startswith("--"):
            name = _long_name(argument)
            if name and "in-place".startswith(name):
                raise CommandRejected("sed -i/--in-place is not allowed")
            if (name and len(name) >= 1 and "quiet".startswith(name)) or (len(name) >= 2 and "silent".startswith(name)):
                quiet = True
            if "=" not in argument and name in ("expression", "file", "line-length"):
                index += 1
        elif argument.startswith("-") and argument != "-":
            for position, flag in enumerate(argument[1:], 1):
                if flag == "i":
                    raise CommandRejected("sed -i/--in-place is not allowed")
                if flag == "n":
                    quiet = True
                if flag in "efl":
                    if position == len(argument) - 1:
                        index += 1
                    break
        index += 1
    if not quiet:
        raise CommandRejected("sed is allowed only with -n (for example sed -n '10,20p' FILE)")
    return ["--sandbox", *args]


def _check_awk(args):
    _scan_options(args, takes_argument=set("FvfiEe"), forbidden_short=set("odpDlW"),
                  forbidden_long=("pretty-print", "profile", "dump-variables", "debug", "load"),
                  long_with_argument=("field-separator", "assign", "file", "include", "exec", "source"))
    return ["--sandbox", *args]


def _check_find(args):
    for argument in args:
        if argument in FIND_FORBIDDEN:
            raise CommandRejected(f"find {argument} is not allowed")
    return list(args)


def _check_sort(args):
    _scan_options(args, takes_argument=set("ktST"), forbidden_short={"o"},
                  forbidden_long=("output", "compress-program"),
                  long_with_argument=("key", "field-separator", "buffer-size", "temporary-directory",
                                      "files0-from", "random-source", "sort", "parallel", "batch-size"))
    return list(args)


def _check_uniq(args):
    operands = _scan_options(args, takes_argument=set("fsw"), forbidden_short=set(), forbidden_long=(),
                             long_with_argument=("skip-fields", "skip-chars", "check-chars"))
    if len(operands) > 1:
        raise CommandRejected("uniq accepts one input file; an output file operand is not allowed")
    return list(args)


def _check_tail(args):
    _scan_options(args, takes_argument=set("ncs"), forbidden_short=set("fF"),
                  forbidden_long=("follow", "retry", "pid"),
                  long_with_argument=("lines", "bytes", "sleep-interval", "max-unchanged-stats"))
    return list(args)


def _check_rg(args):
    for argument in args:
        if argument == "--":
            break
        if argument.startswith("--") and any(_long_name(argument).startswith(name)
                                              for name in ("pre", "search-zip", "hostname-bin")):
            raise CommandRejected(f"option {argument.split('=', 1)[0]} is not allowed")
        if argument.startswith("-") and not argument.startswith("--") and "z" in argument[1:]:
            raise CommandRejected("rg -z/--search-zip is not allowed")
    return list(args)


_PROGRAM_CHECKS = {"sed": _check_sed, "awk": _check_awk, "find": _check_find, "sort": _check_sort,
                   "uniq": _check_uniq, "tail": _check_tail, "rg": _check_rg}


def _program_name(word):
    for prefix in ("/usr/bin/", "/bin/"):
        if word.startswith(prefix):
            return word[len(prefix):]
    return word


def parse_readonly_command(command, *, programs, checkout=None, budget=None):
    """Validate one pipeline. Returns argv lists with absolute in-jail program paths."""
    tokens = _lex(command)
    stages, current = [], []
    for token in tokens:
        if token is _PIPE:
            if not current:
                raise CommandRejected("empty pipeline stage")
            stages.append(current)
            current = []
        else:
            current.append(token)
    if not current:
        raise CommandRejected("empty pipeline stage")
    stages.append(current)
    if len(stages) > MAX_PIPELINE_STAGES:
        raise CommandRejected(f"at most {MAX_PIPELINE_STAGES} pipeline stages are allowed")
    allowed = ", ".join(name for name in ALLOWED_PROGRAMS if name in programs)
    budget = budget or GlobBudget()
    result = []
    for words in stages:
        head = words[0]
        if any(not quoted and c in "*?[" for c, quoted in head):
            raise CommandRejected("a program name cannot be a pattern")
        word = "".join(c for c, _ in head)
        name = _program_name(word)
        if "=" in word and not word.startswith("/"):
            raise CommandRejected("environment assignments are not supported")
        if name not in programs:
            raise CommandRejected(f"program '{word}' is not allowed; allowed programs: {allowed}")
        arguments = []
        for chars in words[1:]:
            arguments.extend(_expand(chars, checkout, budget))
            if len(arguments) > MAX_EXPANDED_ARGUMENTS:
                raise CommandRejected(f"at most {MAX_EXPANDED_ARGUMENTS} arguments are allowed")
        checked = _PROGRAM_CHECKS.get(name, list)(arguments)
        result.append([programs[name], *checked])
    return result


# ---------------------------------------------------------------------------
# Read-only bash: execution in a namespace jail
# ---------------------------------------------------------------------------

# Runs as PID 1 of fresh user/net/mount/pid namespaces (mapped root). Everything
# it needs is imported before chroot; after chroot only /usr and /repo exist.
_JAIL_BOOTSTRAP = r'''
import ctypes, encodings.idna, json, os, resource, socket, subprocess, sys
def fail(message):
    os.write(2, ("ae-sandbox-setup-error: " + message + "\n").encode("utf-8", "replace"))
    os._exit(125)
def main():
    spec = json.loads(sys.argv[1])
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mount.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_ulong, ctypes.c_char_p)
    def enc(value):
        return None if value is None else value.encode()
    def mount(source, target, fstype, flags, data=None):
        if libc.mount(enc(source), enc(target), enc(fstype), flags, enc(data)) != 0:
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code), target)
    RDONLY, NOSUID, NODEV, NOEXEC, REMOUNT, BIND, REC, PRIVATE = 1, 2, 4, 8, 32, 4096, 16384, 1 << 18
    NOATIME, NODIRATIME, RELATIME = 1024, 2048, 1 << 21
    def bind_read_only(source, target):
        mount(source, target, None, BIND)
        info = os.statvfs(target)
        flags = BIND | REMOUNT | RDONLY
        for bit, value in ((os.ST_NOSUID, NOSUID), (os.ST_NODEV, NODEV), (os.ST_NOEXEC, NOEXEC),
                           (os.ST_NOATIME, NOATIME), (os.ST_NODIRATIME, NODIRATIME), (os.ST_RELATIME, RELATIME)):
            if info.f_flag & bit:
                flags |= value
        mount(None, target, None, flags)
    try:
        mount(None, "/", None, REC | PRIVATE)
        jail = spec["jail"]
        mount("ae-expA-jail", jail, "tmpfs", NOSUID | NODEV, "size=64k,mode=0755")
        for name in ("usr", "repo"):
            os.mkdir(os.path.join(jail, name), 0o755)
        for name, target in (("bin", "usr/bin"), ("lib", "usr/lib"), ("lib64", "usr/lib64"), ("sbin", "usr/sbin")):
            os.symlink(target, os.path.join(jail, name))
        bind_read_only("/usr", os.path.join(jail, "usr"))
        bind_read_only(spec["checkout"], os.path.join(jail, "repo"))
        mount(None, jail, None, REMOUNT | RDONLY | NOSUID | NODEV)
        os.chroot(jail)
        os.chdir("/repo")
        for limit, value in spec["limits"].items():
            resource.setrlimit(getattr(resource, limit), (value, value))
    except Exception as error:
        fail(type(error).__name__ + ": " + str(error))
    if spec.get("selftest"):
        report = {"uid": os.getuid(), "root_entries": sorted(os.listdir("/")), "cwd": os.getcwd()}
        try:
            with open("/repo/.ae-expA-write-probe", "x"):
                pass
            report["repo_write_errno"] = None
        except OSError as error:
            report["repo_write_errno"] = error.errno
        probe = socket.socket()
        probe.settimeout(2)
        try:
            probe.connect((b"10.255.255.1", 9))
            report["network_errno"] = None
        except OSError as error:
            report["network_errno"] = error.errno
        finally:
            probe.close()
        os.write(1, json.dumps(report, sort_keys=True).encode())
        os._exit(0)
    read_end, write_end = os.pipe()
    os.close(write_end)
    upstream, processes = read_end, []
    stages = spec["stages"]
    for position, argv in enumerate(stages):
        last = position == len(stages) - 1
        if last:
            downstream = 1
        else:
            next_read, downstream = os.pipe()
        try:
            processes.append(subprocess.Popen([os.path.basename(argv[0])] + argv[1:], executable=argv[0],
                                              stdin=upstream, stdout=downstream, stderr=1,
                                              env=spec["env"], cwd="/repo", close_fds=True))
        except OSError as error:
            os.write(1, (argv[0] + ": cannot execute: " + error.strerror + "\n").encode())
            for process in processes:
                process.kill()
            os._exit(126)
        os.close(upstream)
        if not last:
            os.close(downstream)
            upstream = next_read
    codes = [process.wait() for process in processes]
    final = codes[-1]
    os._exit(final if final >= 0 else 128 - final)
main()
'''

_JAIL_LIMITS = {"RLIMIT_AS": 4 * 1024**3, "RLIMIT_CPU": 60, "RLIMIT_FSIZE": 0, "RLIMIT_CORE": 0,
                "RLIMIT_NOFILE": 256}


def cap_output(retained: bytes, total_bytes: int, limit: int):
    """Decode and cap visible output at ``limit`` UTF-8 bytes including the marker."""
    text = retained.decode("utf-8", errors="replace")
    encoded = text.encode("utf-8")
    if total_bytes <= len(retained) and len(encoded) <= limit:
        return text, False
    marker = OUTPUT_TRUNCATION_MARKER
    room = max(0, limit - len(marker.format(shown=limit, total=total_bytes)))
    shown = encoded[:room].decode("utf-8", errors="ignore")
    return shown + marker.format(shown=len(shown.encode("utf-8")), total=total_bytes), True


class ReadOnlyBash:
    """Validated read-only pipelines executed in a namespace jail rooted at /usr + /repo."""

    def __init__(self, checkout, scratch_dir, *, timeout=BASH_TIMEOUT_SECONDS,
                 max_output_bytes=BASH_OUTPUT_BYTES, python=None, unshare="/usr/bin/unshare",
                 programs=None):
        if os.name != "posix":
            raise RuntimeError("ReadOnlyBash requires Linux namespaces")
        self.checkout = Path(os.path.realpath(checkout))
        if not self.checkout.is_dir():
            raise ValueError("checkout must be an existing directory")
        scratch = Path(os.path.abspath(scratch_dir))  # the jail bootstrap runs with cwd "/"
        scratch.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.mountpoint = scratch / "jail-mountpoint"
        self.mountpoint.mkdir(mode=0o700, exist_ok=True)
        if any(self.mountpoint.iterdir()):
            raise ValueError("jail mountpoint must be an empty directory")
        self.timeout, self.max_output_bytes = float(timeout), int(max_output_bytes)
        self.python = python or sys.executable
        self.unshare = unshare
        self.programs = dict(programs) if programs is not None else discover_programs()

    def describe(self):
        return {"protocol": TOOLS_PROTOCOL, "checkout": str(self.checkout), "programs": self.programs,
                "timeout_seconds": self.timeout, "max_output_bytes": self.max_output_bytes,
                "namespaces": ["user(map-root)", "net", "mount", "pid"], "limits": _JAIL_LIMITS,
                "environment": STAGE_ENVIRONMENT, "max_spec_bytes": MAX_SPEC_BYTES,
                "pathname_expansion": {"walker": "no-follow", "max_entries": GLOB_MAX_ENTRIES,
                                       "max_seconds": GLOB_MAX_SECONDS},
                "bootstrap_sha256": _sha256(_JAIL_BOOTSTRAP.encode())}

    def _argv(self, encoded_spec: str):
        return [self.unshare, "--user", "--map-root-user", "--net", "--mount", "--pid", "--fork",
                "--kill-child", "--", self.python, "-I", "-S", "-B", "-c", _JAIL_BOOTSTRAP, encoded_spec]

    def _spec(self, stages=(), selftest=False):
        return {"jail": str(self.mountpoint), "checkout": str(self.checkout), "stages": list(stages),
                "env": STAGE_ENVIRONMENT, "limits": _JAIL_LIMITS, "selftest": selftest}

    def _capture(self, argv, timeout):
        started = time.monotonic()
        retained, total, hasher = bytearray(), 0, hashlib.sha256()
        errors = bytearray()
        timed_out = False
        try:
            process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, env={}, cwd="/", close_fds=True,
                                       start_new_session=True)
        except OSError as error:
            raise SandboxError(f"cannot start the sandbox: {type(error).__name__}: {error}") from error
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ, "out")
        selector.register(process.stderr, selectors.EVENT_READ, "err")
        try:
            while selector.get_map():
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    timed_out = True
                    break
                for key, _ in selector.select(min(remaining, 0.25)):
                    chunk = os.read(key.fd, 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    if key.data == "err":
                        errors.extend(chunk[:max(0, 16384 - len(errors))])
                        continue
                    hasher.update(chunk)
                    total += len(chunk)
                    room = self.max_output_bytes - len(retained)
                    if room > 0:
                        retained.extend(chunk[:room])
            if not timed_out:
                try:
                    process.wait(timeout=max(0.05, timeout - (time.monotonic() - started)))
                except subprocess.TimeoutExpired:
                    timed_out = True
        finally:
            if timed_out or process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.wait()
            selector.close()
            process.stdout.close()
            process.stderr.close()
        return {"returncode": process.returncode, "retained": bytes(retained), "total": total,
                "sha256": hasher.hexdigest(), "stderr": bytes(errors), "timed_out": timed_out,
                "duration_s": time.monotonic() - started}

    @staticmethod
    def _infrastructure_failure(capture):
        stderr = capture["stderr"].decode("utf-8", errors="replace").strip()
        if capture["timed_out"]:
            return None
        if capture["returncode"] == 125 and "ae-sandbox-setup-error" in stderr:
            return stderr
        if stderr:
            return stderr[:2000]
        return None

    def selftest(self):
        """Prove isolation: mapped uid 0, jail root, read-only /repo, no network."""
        capture = self._capture(self._argv(canonical_json(self._spec(selftest=True)).decode()), 30)
        failure = self._infrastructure_failure(capture)
        if failure or capture["returncode"] != 0:
            raise SandboxError("sandbox selftest failed: " + (failure or str(capture["returncode"])))
        report = json.loads(capture["retained"])
        expected_root = ["bin", "lib", "lib64", "repo", "sbin", "usr"]
        report["passed"] = (report.get("uid") == 0 and report.get("root_entries") == expected_root
                            and report.get("cwd") == SANDBOX_REPO
                            and report.get("repo_write_errno") is not None
                            and report.get("network_errno") is not None)
        if not report["passed"]:
            raise SandboxError("sandbox selftest did not prove isolation: " + json.dumps(report, sort_keys=True))
        return report

    def contract_probe(self):
        """Prove the read-only contract on this host: every required program, sed/gawk sandboxes."""
        required = [name for name in ALLOWED_PROGRAMS if name != "rg"]
        missing = [name for name in required if name not in self.programs]
        if missing:
            raise SandboxError("required read-only programs are unavailable: " + ", ".join(missing))
        probes = {"sed_write": ("sed -n 'w /repo/.ae-probe' /usr/bin/sed", "sandbox"),
                  "awk_system": ("awk 'BEGIN{system(\"ls\")}'", "sandbox"),
                  "awk_redirect": ("awk 'BEGIN{print 1 > \"/repo/.ae-probe\"}'", "sandbox")}
        report = {}
        for name, (command, needle) in probes.items():
            visible = self.run(command)["visible"]
            report[name] = {"exit_code": visible.get("exit_code"), "refused": needle in visible.get("output", "")}
            if visible.get("exit_code") in (0, None) or not report[name]["refused"]:
                raise SandboxError(f"{name} was not refused by the program sandbox: {visible}")
        report["programs"] = dict(self.programs)
        return report

    @staticmethod
    def _rejected(command, reason):
        message = REJECTION_PREFIX + str(reason)
        return {"command": command if utf8_encodable(command) else safe_echo(command, limit=MAX_COMMAND_BYTES * 6),
                "status": "rejected", "error": message, "visible": {"error": message}}

    def run(self, command):
        """Full record for the event log; ``visible`` is what the model receives."""
        try:
            stages = parse_readonly_command(command, programs=self.programs, checkout=self.checkout)
            try:
                encoded = canonical_json(self._spec(stages)).decode()
            except UnicodeEncodeError as error:  # finding F5: never a controller exception
                raise CommandRejected("the command contains invalid Unicode (lone surrogate code points)") from error
            if len(encoded.encode("utf-8")) > MAX_SPEC_BYTES:
                raise CommandRejected(f"the command is longer than {MAX_SPEC_BYTES} bytes after pathname expansion; "
                                      "use narrower patterns")
        except CommandRejected as error:
            return self._rejected(command, error)
        capture = self._capture(self._argv(encoded), self.timeout)
        failure = self._infrastructure_failure(capture)
        if failure:
            raise SandboxError(failure)
        output, truncated = cap_output(capture["retained"], capture["total"], self.max_output_bytes)
        record = {"command": command, "status": "timed_out" if capture["timed_out"] else "executed",
                  "argv": stages, "exit_code": None if capture["timed_out"] else capture["returncode"],
                  "output": output, "truncated": truncated, "output_bytes": capture["total"],
                  "output_sha256": capture["sha256"], "duration_s": capture["duration_s"]}
        visible = {"exit_code": record["exit_code"], "output": output, "truncated": truncated}
        if capture["timed_out"]:
            visible["error"] = TIMEOUT_MESSAGE.format(seconds=self.timeout)
        record["visible"] = visible
        return record


# ---------------------------------------------------------------------------
# Model-visible checkouts: clean-checkout manifests and staging (findings F6, F11)
# ---------------------------------------------------------------------------

# One clean checkout manifest covers every dispatch family. scripts/expA_stage.py stages it (and exactly the
# trees it lists) to Core1 and reproduces tree_digest; the runner checks every staged checkout against it.
CHECKOUTS_SCHEMA = "expA-clean-checkouts-v1"
TREE_PROTOCOL = "expA-checkout-tree-v1"
TREE_FORMULA = ("sha256(canonical_json({'protocol': TREE_PROTOCOL, 'files': {relpath: sha256 of the bytes}, "
                "'symlinks': {relpath: link target}, 'directories': [every directory relpath, sorted]}))")
STAGE_SKIP_DIRS = frozenset({".git", ".hg", ".svn", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
                             ".tox", ".nox", ".hypothesis", ".eggs", ".venv", "venv"})
STAGE_SKIP_SUFFIXES = (".pyc", ".pyo")
TREE_MAX_ENTRIES = 500_000
TREE_MAX_BYTES = 4 * 1024**3


def _skipped(name, is_dir):
    return name in STAGE_SKIP_DIRS if is_dir else name.endswith(STAGE_SKIP_SUFFIXES)


def _tree_entries(root):
    """Deterministic no-follow walk: yields (relpath, kind, path) for dirs, files, symlinks and others.

    VCS and cache directories (``STAGE_SKIP_DIRS``) and compiled Python files are
    reported as kind ``skipped`` and never descended.
    """
    root = Path(root)
    info = os.lstat(root)
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"checkout root must be a real directory, not a symlink or file: {root}")
    count = 0
    pending = [("", root)]
    while pending:
        prefix, directory = pending.pop()
        with os.scandir(directory) as iterator:
            entries = sorted(iterator, key=lambda entry: entry.name)
        children = []
        for entry in entries:
            count += 1
            if count > TREE_MAX_ENTRIES:
                raise ValueError(f"checkout has more than {TREE_MAX_ENTRIES} entries")
            relative = prefix + entry.name
            mode = entry.stat(follow_symlinks=False).st_mode
            if stat.S_ISLNK(mode):
                yield relative, "symlink", Path(entry.path)
            elif stat.S_ISDIR(mode):
                if _skipped(entry.name, True):
                    yield relative, "skipped", Path(entry.path)
                else:
                    yield relative, "dir", Path(entry.path)
                    children.append((relative + "/", Path(entry.path)))
            elif stat.S_ISREG(mode):
                yield relative, ("skipped" if _skipped(entry.name, False) else "file"), Path(entry.path)
            else:
                yield relative, "other", Path(entry.path)
        pending.extend(reversed(children))


def _file_sha256(path):
    digest = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError(f"not a regular file: {path}")
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_digest(files, symlinks, directories):
    """``TREE_FORMULA``; scripts/expA_stage.py reproduces it byte for byte."""
    return _sha256(canonical_json({"protocol": TREE_PROTOCOL, "files": dict(files), "symlinks": dict(symlinks),
                                   "directories": list(directories)}))


def checkout_tree(root):
    """Whole-tree record of a model-visible checkout: every file, symlink and directory (finding F11).

    ``tree_sha256`` is ``tree_digest`` over all of them, not only package scope.
    VCS/cache directories and *.pyc/*.pyo files are listed under ``skipped``:
    staging never copies them and they are not part of the tree. Device files,
    sockets and FIFOs are listed under ``other``; staging refuses them.
    """
    files, symlinks, directories, skipped, other = {}, {}, [], [], []
    for relative, kind, path in _tree_entries(root):
        if kind == "file":
            files[relative] = _file_sha256(path)
        elif kind == "symlink":
            symlinks[relative] = os.readlink(path)
        elif kind == "dir":
            directories.append(relative)
        elif kind == "skipped":
            skipped.append(relative)
        else:
            other.append(relative)
    directories.sort()
    return {"protocol": TREE_PROTOCOL, "tree_sha256": tree_digest(files, symlinks, directories),
            "files": files, "symlinks": symlinks, "directories": directories,
            "skipped": sorted(skipped), "other": sorted(other),
            "counts": {"files": len(files), "symlinks": len(symlinks), "directories": len(directories),
                       "skipped": len(skipped), "other": len(other)}}


def symlink_escape(root, relative, target):
    """None when the symlink at ``relative`` resolves inside ``root``; else the reason it does not."""
    if target.startswith("/"):
        return "absolute_target"
    lexical = posixpath.normpath(posixpath.join(posixpath.dirname(relative), target))
    if lexical == ".." or lexical.startswith("../") or lexical.startswith("/"):
        return "lexically_outside"
    base = os.path.realpath(root)
    resolved = os.path.realpath(os.path.join(root, relative))
    if resolved != base and not resolved.startswith(base + os.sep):
        return "resolves_outside"
    return None


def stage_checkout(source, destination):
    """Copy a checkout to local scratch, read-only afterwards; returns the staging record (finding F6).

    No symlink is followed. VCS/cache directories and compiled files are not
    copied (``skipped``). Symlinks that resolve outside the checkout are dropped
    and recorded (``neutralized_symlinks``); symlinks that stay inside are kept.
    Device files, sockets and FIFOs are refused.
    """
    source, destination = Path(os.path.realpath(source)), Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"{destination} already exists")
    destination.mkdir(mode=0o755, parents=True)
    record = {"source": str(source), "destination": str(destination), "skipped": [], "neutralized_symlinks": [],
              "kept_symlinks": 0, "files": 0, "bytes": 0, "directories": 0}
    created_dirs = [destination]
    for relative, kind, path in _tree_entries(source):
        target = destination / relative
        if kind == "dir":
            target.mkdir(mode=0o755)
            created_dirs.append(target)
            record["directories"] += 1
        elif kind == "file":
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            with os.fdopen(descriptor, "rb") as reader, open(target, "xb") as writer:
                if not stat.S_ISREG(os.fstat(reader.fileno()).st_mode):
                    raise ValueError(f"not a regular file: {path}")
                shutil.copyfileobj(reader, writer, 1024 * 1024)
                record["bytes"] += writer.tell()
            if record["bytes"] > TREE_MAX_BYTES:
                raise ValueError(f"checkout is larger than {TREE_MAX_BYTES} bytes")
            os.chmod(target, 0o444)
            record["files"] += 1
        elif kind == "symlink":
            link = os.readlink(path)
            reason = symlink_escape(source, relative, link)
            if reason is None:
                os.symlink(link, target)
                record["kept_symlinks"] += 1
            else:
                record["neutralized_symlinks"].append({"path": relative, "target": link, "reason": reason,
                                                       "action": "dropped"})
        elif kind == "skipped":
            record["skipped"].append(relative)
        else:
            raise ValueError(f"checkout contains a device, socket or FIFO: {relative}")
    for directory in reversed(created_dirs):
        os.chmod(directory, 0o555)
    return record


def clean_checkout_entry(family, root, package_dirs, *, commit=None):
    """One family entry of the clean checkout manifest (``CHECKOUTS_SCHEMA``), from a checkout on disk.

    ``files``/``symlinks``/``directories`` are exactly what ``stage_checkout``
    copies: skipped entries and symlinks that leave the checkout are listed
    separately (``skipped``, ``escaping_symlinks``) and never staged.
    scripts/expA_stage.py refuses an entry with either list nonempty.
    """
    root = Path(os.path.realpath(root))
    tree = checkout_tree(root)
    if tree["other"]:
        raise ValueError(f"{family}: checkout contains devices, sockets or FIFOs: {tree['other'][:5]}")
    symlinks, escaping = {}, []
    for relative, target in sorted(tree["symlinks"].items()):
        reason = symlink_escape(root, relative, target)
        if reason is None:
            symlinks[relative] = target
        else:
            escaping.append({"path": relative, "target": target, "reason": reason})
    package = scan_package_files(root, list(package_dirs))
    if package["missing_package_dirs"]:
        raise ValueError(f"{family}: missing package directories {package['missing_package_dirs']}")
    files = tree["files"]
    return {"family": family, "path": str(root), "commit": commit, "files": files, "symlinks": symlinks,
            "directories": tree["directories"], "tree_sha256": tree_digest(files, symlinks, tree["directories"]),
            "package_dirs": list(normalize_package_dirs(package_dirs)), "package_files": package["files"],
            "package_tree_sha256": source_tree_sha256({path: files[path] for path in package["files"]}),
            "skipped": tree["skipped"], "escaping_symlinks": escaping}


def checkouts_manifest(entries):
    """The clean checkout manifest document for ``entries`` (clean_checkout_entry records)."""
    families = {}
    for entry in entries:
        if entry["family"] in families:
            raise ValueError(f"duplicate family {entry['family']} in the clean checkout manifest")
        families[entry["family"]] = entry
    return {"schema": CHECKOUTS_SCHEMA, "tree_protocol": TREE_PROTOCOL, "formula": TREE_FORMULA,
            "skip_dirs": sorted(STAGE_SKIP_DIRS), "skip_suffixes": list(STAGE_SKIP_SUFFIXES),
            "families": dict(sorted(families.items()))}


def checkout_entry_problems(entry):
    """Why a manifest entry cannot pin a model-visible checkout ([] when it can)."""
    files, symlinks, directories = entry.get("files"), entry.get("symlinks"), entry.get("directories")
    if not isinstance(files, dict) or not isinstance(symlinks, dict) or not isinstance(directories, list):
        return ["the entry needs files, symlinks and directories"]
    problems = []
    if any(not isinstance(sha, str) or len(sha) != 64 for sha in files.values()):
        problems.append("a file has no sha256")
    if directories != sorted(set(directories)):
        problems.append("directories must be sorted and unique")
    if tree_digest(files, symlinks, directories) != entry.get("tree_sha256"):
        problems.append("tree_sha256 does not match its file, symlink and directory lists")
    for relative, target in symlinks.items():
        lexical = posixpath.normpath(posixpath.join(posixpath.dirname(relative), str(target)))
        if not isinstance(target, str) or target.startswith("/") or lexical == ".." or lexical.startswith("../"):
            problems.append(f"symlink {relative} leaves the checkout")
    for path in list(files) + list(symlinks) + list(directories):
        if any(part in STAGE_SKIP_DIRS for part in path.split("/")) or path.endswith(STAGE_SKIP_SUFFIXES):
            problems.append(f"{path} is never staged")
            break
    package = entry.get("package_files")
    if isinstance(package, list) and entry.get("package_tree_sha256") is not None:
        if set(package) - set(files) or \
                source_tree_sha256({path: files[path] for path in package}) != entry["package_tree_sha256"]:
            problems.append("package_tree_sha256 does not match its package files")
    return problems


def compare_tree(entry, actual):
    """A staged checkout_tree() against its clean checkout manifest entry.

    ``matches`` requires the entry's tree_sha256 and the exact file, symlink and
    directory sets: nothing unexpected, nothing missing, nothing changed.
    """
    files = entry.get("files") if isinstance(entry.get("files"), dict) else {}
    symlinks = entry.get("symlinks") if isinstance(entry.get("symlinks"), dict) else {}
    directories = entry.get("directories") if isinstance(entry.get("directories"), list) else []
    result = {"expected_tree_sha256": entry.get("tree_sha256"), "actual_tree_sha256": actual["tree_sha256"],
              "entry_problems": checkout_entry_problems(entry),
              "unexpected_files": sorted(set(actual["files"]) - set(files))[:200],
              "missing_files": sorted(set(files) - set(actual["files"]))[:200],
              "changed_files": sorted(path for path in set(files) & set(actual["files"])
                                      if files[path] != actual["files"][path])[:200],
              "symlink_differences": sorted(set(symlinks.items()) ^ set(actual["symlinks"].items()))[:200],
              "unexpected_directories": sorted(set(actual["directories"]) - set(directories))[:200],
              "missing_directories": sorted(set(directories) - set(actual["directories"]))[:200],
              "other": list(actual.get("other", []))[:200]}
    result["matches"] = (result["expected_tree_sha256"] == result["actual_tree_sha256"]
                         and not any(result[key] for key in ("entry_problems", "unexpected_files", "missing_files",
                                                             "changed_files", "symlink_differences",
                                                             "unexpected_directories", "missing_directories",
                                                             "other")))
    return result


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

class ToolBox:
    """The Exp A tools bound to one read-only checkout and package scope (graph_query optional, T arms)."""

    def __init__(self, checkout, package_dirs, bash, *, token_budget=SEARCH_TOKEN_BUDGET,
                 token_counter=None, graph=None):
        self.checkout = Path(checkout)
        self.package_dirs = list([package_dirs] if isinstance(package_dirs, str) else package_dirs)
        normalize_package_dirs(self.package_dirs)
        self.bash = bash
        self.token_budget = token_budget
        self.token_counter = token_counter
        self.graph = graph

    def describe(self):
        bash = self.bash.describe() if hasattr(self.bash, "describe") else {"type": type(self.bash).__name__}
        return {"protocol": TOOLS_PROTOCOL, "package_dirs": self.package_dirs,
                "search": {"max_hits": SEARCH_MAX_HITS, "max_tokens": self.token_budget,
                           "token_bound": token_bound_name(self.token_counter)},
                "bash": bash, "graph_query": None if self.graph is None else self.graph.describe()}

    def search(self, arguments):
        if not isinstance(arguments, dict) or "query" not in arguments or set(arguments) - {"query", "limit"}:
            message = "source_reference_search takes query and optional limit"
            visible = search_error(arguments.get("query") if isinstance(arguments, dict) else None,
                                   "invalid_arguments: " + message)
            return {"status": "invalid_arguments", "visible": visible}
        query, limit = arguments["query"], arguments.get("limit", SEARCH_MAX_HITS)
        if not utf8_encodable(query):  # finding F5: echo safely, never serialize the raw surrogate
            return {"status": "invalid_arguments",
                    "visible": search_error(query, "invalid_query: the query contains invalid Unicode")}
        try:
            result = lexical_search(self.checkout, self.package_dirs, query, limit,
                                    self.token_budget, self.token_counter)
        except ValueError as error:
            return {"status": "invalid_arguments", "visible": search_error(query, "invalid_query: " + str(error))}
        except OSError as error:
            raise SandboxError("source scan failed: " + type(error).__name__) from error
        return {"status": "executed", "visible": result}

    def execute(self, name, arguments):
        """Return (record, visible) for search, bash and graph_query. submit_answer is the harness's."""
        if name == "source_reference_search":
            record = self.search(arguments)
        elif name == "bash":
            if not isinstance(arguments, dict) or set(arguments) != {"command"} or not isinstance(arguments["command"], str):
                message = "bash takes exactly one string argument named command"
                record = {"status": "invalid_arguments", "visible": {"error": message}}
            else:
                record = self.bash.run(arguments["command"])
        elif name == GRAPH_QUERY and self.graph is not None:
            record = self.graph.run(arguments)
        else:
            raise ValueError("unknown tool " + repr(name))
        return record, record["visible"]


# ---------------------------------------------------------------------------
# Operational details (D11.6; prereg Appendix A is generated from this)
# ---------------------------------------------------------------------------

PROGRAM_RULES = {
    "sed": "only with -n; -i/--in-place (and any prefix of it) refused; runs as GNU sed --sandbox (no w, W, r, R "
           "or e commands)",
    "awk": "GNU awk only; runs as gawk --sandbox (no system(), redirection, getline from files or new ARGV files); "
           "-o -d -p -D -l -W and --pretty-print/--profile/--dump-variables/--debug/--load refused",
    "find": "-exec -execdir -ok -okdir -delete -fprint -fprint0 -fprintf -fls refused",
    "sort": "-o/--output and --compress-program refused",
    "uniq": "at most one file operand (no output file)",
    "tail": "-f/-F/--follow/--retry/--pid refused",
    "rg": "--pre, --search-zip, --hostname-bin and -z refused (only when rg is installed)",
    "grep, cat, head, ls, wc, cut, nl": "no extra restrictions beyond the jail"}


def bash_rejection_messages():
    """Every CommandRejected message in this module, as source text (f-string placeholders kept)."""
    import ast
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    found = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call) and node.exc.args
                and getattr(node.exc.func, "id", None) == "CommandRejected"):
            found.add(ast.unparse(node.exc.args[0]))
    return sorted(found)


def operational_details():
    """Frozen tool facts: schemas, allowlist, rules, rejection messages, truncation markers, limits."""
    return {
        "protocol": TOOLS_PROTOCOL,
        "schemas": {"source_reference_search": SOURCE_REFERENCE_TOOL, "bash": BASH_TOOL,
                    "submit_answer": SUBMIT_ANSWER_TOOL, "graph_query": GRAPH_QUERY_TOOL},
        "tool_order": {"P_and_S_blocks": list(TOOL_NAMES), "T_arms": list(ALL_TOOL_NAMES)},
        "source_reference_search": {
            "kind": LEXICAL_KIND, "max_hits": SEARCH_MAX_HITS, "max_bytes": SEARCH_TOKEN_BUDGET,
            "token_bound": token_bound_name(None),
            "semantics": "graphs.GraphBackend.source_reference_search rules (qualified and simple names plus "
                         "'import ... as' aliases, qualified matches ranked first, then path and line) over "
                         "package-scope Python files only",
            "truncation": "the hits are the longest ranked prefix whose canonical JSON fits the byte budget; "
                          "'truncated' is true when fewer hits than candidates are returned",
            "errors": "invalid queries return hits [] with an 'error' text; the query is echoed ASCII-escaped "
                      "when it is not valid UTF-8"},
        "bash": {
            "allowed_programs": list(ALLOWED_PROGRAMS), "program_rules": PROGRAM_RULES,
            "find_forbidden": sorted(FIND_FORBIDDEN),
            "syntax": "one pipeline of allowlisted programs run as argv lists (no shell); quotes and backslash "
                      "escapes as in sh; unquoted * ? [ ] expand against the checkout; ; && || & redirections, "
                      "subshells, backticks, $( ), ${ }, variables, ~, comments, braces and newlines are refused",
            "max_pipeline_stages": MAX_PIPELINE_STAGES, "max_command_bytes": MAX_COMMAND_BYTES,
            "max_expanded_arguments": MAX_EXPANDED_ARGUMENTS, "max_spec_bytes": MAX_SPEC_BYTES,
            "pathname_expansion": {"walker": "no-follow (symlinks are never descended)",
                                   "max_entries": GLOB_MAX_ENTRIES, "max_seconds": GLOB_MAX_SECONDS,
                                   "no_match": "the pattern stays literal (nullglob off)",
                                   "dotfiles": "hidden unless the pattern starts with '.'"},
            "timeout_seconds": BASH_TIMEOUT_SECONDS, "timeout_message": TIMEOUT_MESSAGE,
            "output_bytes": BASH_OUTPUT_BYTES, "truncation_marker": OUTPUT_TRUNCATION_MARKER,
            "stderr": "merged into the visible output", "rejection_prefix": REJECTION_PREFIX,
            "rejection_messages": bash_rejection_messages(),
            "jail": {"namespaces": ["user (mapped root)", "net", "mount", "pid"], "root_entries":
                     ["bin", "lib", "lib64", "repo", "sbin", "usr"], "repo": SANDBOX_REPO + " (read-only bind)",
                     "usr": "read-only bind", "limits": dict(_JAIL_LIMITS), "environment": dict(STAGE_ENVIRONMENT)},
            "visible_result": "{exit_code, output, truncated} plus error on timeout; a rejected command returns "
                              "{error} only"},
        "graph_query": {"render": "C1 (no guidance, no verification)",
                        "max_output_bytes": GRAPH_QUERY_MAX_OUTPUT_BYTES, "defaults": GRAPH_QUERY_DEFAULTS,
                        "facts": "the family's natural LocAgent facts (graphs.GraphFacts.query); any symbol may "
                                 "be queried", "truncation": "graphs.StructuralObservation.render: "
                                                             "payload_truncated true when the answer was shortened"},
        "invalid_unicode": INVALID_UNICODE,
        "submit_answer": submission_policy()}
