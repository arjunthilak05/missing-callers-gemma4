"""Deterministic public-issue/source-only retrieval; repository code never runs.

The production entry point copies regular Python files with no-follow traversal,
then parses/ranks in a bounded subprocess. It accepts no graph or evaluator data.
"""
from __future__ import annotations

import ast
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import uuid

from .data import canonical_json
from .graphs import _regular_sources, source_inventory

RETRIEVAL_PROTOCOL = "absent-edges-source-bm25-entity-window-v1"


def _hash(value):
    return hashlib.sha256(value if isinstance(value, bytes) else canonical_json(value)).hexdigest()


@dataclass(frozen=True)
class PublicIssue:
    instance_id: str
    problem_statement: str

    def __post_init__(self):
        if not isinstance(self.instance_id, str) or not 1 <= len(self.instance_id.encode()) <= 512:
            raise ValueError("Bounded public issue identity required")
        if not isinstance(self.problem_statement, str) or not 1 <= len(self.problem_statement.encode()) <= 65536:
            raise ValueError("Bounded public issue text required")

    @classmethod
    def from_dict(cls, value):
        if set(value) != {"instance_id", "problem_statement"}:
            raise ValueError("Public issue allows only instance_id and problem_statement; no hints, solutions or tests")
        return cls(**value)


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = 12
    source_budget_bytes: int = 24000
    window_lines: int = 40
    window_stride: int = 20
    entity_max_lines: int = 120
    excerpt_max_bytes: int = 8192
    candidate_limit: int = 40000

    def __post_init__(self):
        bounds = {"top_k": (1, 100), "source_budget_bytes": (1024, 128*1024),
                  "window_lines": (1, 200), "window_stride": (1, self.window_lines),
                  "entity_max_lines": (1, 500), "excerpt_max_bytes": (128, 16384),
                  "candidate_limit": (1, 100000)}
        for key, (low, high) in bounds.items():
            if type(getattr(self, key)) is not int or not low <= getattr(self, key) <= high:
                raise ValueError("Invalid retrieval budget: " + key)


def lexical_terms(text):
    # Preserve whole identifiers and add deterministic snake/camel components.
    words = re.findall(r"[^\W\d]\w*|\d+", text, flags=re.UNICODE)
    result = []
    for word in words:
        result.append(word.casefold())
        parts = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", word).replace("_", " ").split()
        if len(parts) > 1:
            result.extend(part.casefold() for part in parts)
    return result


def _entities(text, path):
    found = []
    tree = ast.parse(text, filename=path)

    def visit(node, parents=()):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                qualified = parents + (child.name,)
                start = min([child.lineno] + [item.lineno for item in child.decorator_list])
                found.append({"symbol_id": path + "::" + ".".join(qualified),
                              "start_line": start, "end_line": child.end_lineno})
                visit(child, qualified)
            else:
                visit(child, parents)
    visit(tree)
    return found


@dataclass(frozen=True)
class FrozenSourcePool:
    encoded: bytes

    @property
    def data(self):
        return json.loads(self.encoded)

    @property
    def sha256(self):
        return _hash(self.encoded)

    def save(self, path):
        """Exclusive freeze; a path can never silently replace an earlier pool."""
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(self.encoded)
            stream.flush()
            os.fsync(stream.fileno())
        return {"path": str(path), "sha256": self.sha256}

    @classmethod
    def load(cls, path, sha256):
        pool = cls(Path(path).read_bytes())
        if pool.sha256 != sha256:
            raise ValueError("Frozen source pool identity changed")
        pool.validate()
        return pool

    def validate(self):
        data = self.data
        if data["protocol"] != RETRIEVAL_PROTOCOL or data["selection_basis"] != "public_issue_and_python_source_only":
            raise ValueError("Unsupported retrieval protocol")
        PublicIssue.from_dict(data["public_issue"])
        RetrievalConfig(**data["config"])
        if data["query_sha256"] != _hash(data["public_issue"]) or data["source_sha256"] != _hash(data["source_files"]):
            raise ValueError("Query or source identity mismatch")
        selected = data["candidates"]
        if len({item["candidate_id"] for item in selected}) != len(selected):
            raise ValueError("Duplicate frozen candidate identity")
        candidate_ids = {item["candidate_id"] for item in selected}
        for item in selected:
            if item["source_text_sha256"] != _hash(item["source_text"].encode()) or item["file_sha256"] != data["source_files"][item["path"]]:
                raise ValueError("Frozen candidate source identity mismatch")
        if len({item["symbol_id"] for item in data["symbols"]}) != len(data["symbols"]):
            raise ValueError("Ambiguous frozen symbol identity")
        if any(item["candidate_id"] not in candidate_ids for item in data["symbols"]):
            raise ValueError("Symbol source must be in the already frozen candidate pool")
        anchor = data["anchor_selection"]
        if anchor["basis"] not in ("declared_before_graph", "source_only_bm25_first_complete_entity"):
            raise ValueError("Unsupported source-only anchor selection")
        complete = {item["symbol_id"] for item in data["symbols"] if item["entity_text_complete"]}
        if anchor["symbol_id"] is not None and anchor["symbol_id"] not in complete:
            raise ValueError("Frozen anchor must have complete source text")


def _rank_sources(files, issue, config, retriever_sha256, *, declared_anchor_symbol=None):
    """Pure algorithm used by the worker and synthetic tests; no filesystem reads."""
    if not isinstance(issue, PublicIssue) or not isinstance(config, RetrievalConfig):
        raise TypeError("Typed public issue and retrieval configuration required")
    documents, entities, hashes, limitations = [], [], {}, []
    total_lines = covered_lines = 0
    for path, raw in sorted(files.items()):
        if not isinstance(raw, bytes) or not path.endswith(".py") or path.startswith("/") or ".." in Path(path).parts:
            raise ValueError("Expected relative Python source bytes")
        hashes[path] = _hash(raw)
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            limitations.append({"path": path, "reason": "unsupported_non_utf8_source"})
            continue
        # Python line numbers use physical CR/LF newlines. splitlines() also
        # treats literal vertical tabs/form feeds as lines and can corrupt spans.
        lines = [line for line in re.findall(r".*?(?:\r\n|\n|\r|$)", text, flags=re.S) if line]
        total_lines += len(lines)
        coverage = set()

        def candidate(start, end, kind, symbol=None):
            end = min(end, len(lines))
            actual_end, size = start-1, 0
            for index in range(start-1, end):
                length = len(lines[index].encode())
                if size + length > config.excerpt_max_bytes:
                    break
                size += length
                actual_end = index+1
            if actual_end < start:
                return
            source = "".join(lines[start-1:actual_end])
            identity = {"path": path, "kind": kind, "symbol_id": symbol, "start_line": start,
                        "end_line": actual_end, "file_sha256": hashes[path]}
            item = {**identity, "candidate_id": _hash(identity), "source_text": source,
                    "source_text_sha256": _hash(source.encode()), "span_truncated": actual_end < end}
            documents.append(item)
            if len(documents) > config.candidate_limit:
                raise ValueError("Candidate inventory limit exceeded; no silent partial ranking")
            if kind == "window":
                coverage.update(range(start, actual_end+1))
        for index in range(0, len(lines), config.window_stride):
            candidate(index+1, min(index+config.window_lines, len(lines)), "window")
        covered_lines += len(coverage)
        if len(coverage) != len(lines):
            limitations.append({"path": path, "reason": "window_byte_budget_gaps", "uncovered_lines": len(lines)-len(coverage)})
        try:
            file_entities = _entities(text, path)
        except (SyntaxError, RecursionError, ValueError) as error:
            limitations.append({"path": path, "reason": "entity_parse_" + type(error).__name__})
            file_entities = []
        for entity in file_entities:
            entities.append({**entity, "path": path})
            end = min(entity["end_line"], entity["start_line"]+config.entity_max_lines-1)
            candidate(entity["start_line"], end, "entity", entity["symbol_id"])
            if end < entity["end_line"] and documents and documents[-1]["symbol_id"] == entity["symbol_id"]:
                documents[-1]["span_truncated"] = True
    counts, frequency = [], Counter()
    for item in documents:
        terms = Counter(lexical_terms(item["source_text"]))
        terms.update(lexical_terms(item["path"]) * 2)
        terms.update(lexical_terms(item["symbol_id"] or "") * 2)
        counts.append(terms)
        frequency.update(terms.keys())
    n = len(documents)
    average = sum(sum(terms.values()) for terms in counts) / n if n else 1
    query = sorted(set(lexical_terms(issue.problem_statement)))
    for item, terms in zip(documents, counts):
        length, score = sum(terms.values()), 0.0
        for term in query:
            tf = terms[term]
            if tf:
                idf = math.log(1+(n-frequency[term]+0.5)/(frequency[term]+0.5))
                score += idf * tf*2.2/(tf+1.2*(0.25+0.75*length/max(1, average)))
        item["bm25_score"] = round(score, 12)
    documents.sort(key=lambda item: (-item["bm25_score"], item["path"], item["start_line"], item["kind"], item["candidate_id"]))
    selected, used, budget_skips = [], 0, 0
    for item in documents:
        if len(selected) >= config.top_k:
            break
        size = len(item["source_text"].encode())
        if used+size > config.source_budget_bytes:
            budget_skips += 1
            continue
        selected.append(item)
        used += size
    symbol_counts = Counter(item["symbol_id"] for item in entities)
    symbols = []
    for entity in entities:
        if symbol_counts[entity["symbol_id"]] != 1:
            continue
        matching = [item for item in selected if item["path"] == entity["path"] and
                    item["start_line"] <= entity["start_line"] and item["end_line"] >= entity["end_line"]]
        matching += [item for item in selected if item not in matching and item["symbol_id"] == entity["symbol_id"]]
        for item in matching:
            same_symbol = item["symbol_id"] == entity["symbol_id"]
            contained = item["path"] == entity["path"] and item["start_line"] <= entity["start_line"] and item["end_line"] >= entity["end_line"]
            if same_symbol or contained:
                symbols.append({"symbol_id": entity["symbol_id"], "candidate_id": item["candidate_id"],
                                "source_text_sha256": item["source_text_sha256"],
                                "entity_text_complete": contained})
                break
    complete = {item["symbol_id"] for item in symbols if item["entity_text_complete"]}
    if declared_anchor_symbol is not None:
        if not isinstance(declared_anchor_symbol, str) or declared_anchor_symbol not in complete:
            raise ValueError("Declared source-only anchor lacks complete text in selected pool")
        anchor = {"basis": "declared_before_graph", "symbol_id": declared_anchor_symbol}
    else:
        ranked = [item["symbol_id"] for item in selected if item["symbol_id"] in complete]
        ranked += [item["symbol_id"] for item in symbols if item["symbol_id"] in complete and item["symbol_id"] not in ranked]
        anchor = {"basis": "source_only_bm25_first_complete_entity", "symbol_id": ranked[0] if ranked else None}
    value = {"protocol": RETRIEVAL_PROTOCOL, "selection_basis": "public_issue_and_python_source_only",
             "public_issue": asdict(issue), "query_sha256": _hash(asdict(issue)), "config": asdict(config),
             "retriever_sha256": retriever_sha256, "source_files": hashes, "source_sha256": _hash(hashes),
             "parser_python_version": list(sys.version_info[:3]),
             "candidates": selected, "symbols": symbols,
             "anchor_selection": anchor,
             "coverage": {"scope": "all regular UTF-8 Python files in supplied sanitized snapshot; .git excluded",
                          "inventory_file_count": len(files), "inventory_bytes": sum(map(len, files.values())),
                          "decoded_line_count": total_lines, "window_covered_lines": covered_lines,
                          "windows_cover_all_supported_lines": total_lines == covered_lines,
                          "entity_identity_collisions": sum(count > 1 for count in symbol_counts.values()),
                          "limitations": limitations, "candidate_count": len(documents),
                          "selected_count": len(selected), "selected_source_bytes": used,
                          "selection_truncated": len(selected) < len(documents), "budget_skipped_candidates": budget_skips},
             "ranking": {"algorithm": "BM25", "k1": 1.2, "b": 0.75, "query_term_frequency": "binary",
                         "path_and_entity_term_weight": 2, "score_round_decimal_places": 12,
                         "tie_break": ["path", "start_line", "kind", "candidate_id"]}}
    pool = FrozenSourcePool(canonical_json(value))
    pool.validate()
    return pool


class SourceRetriever:
    def __init__(self, work_root, *, python=sys.executable, timeout=45):
        self.work_root = Path(work_root).resolve()
        self.work_root.mkdir(parents=True, exist_ok=True)
        self.python = str(Path(python).absolute())
        if not 0 < timeout <= 120:
            raise ValueError("Bounded retrieval timeout required")
        self.timeout = timeout

    def freeze(self, sanitized_source_root, public_issue, *, config=None, expected_source_sha256=None,
               declared_anchor_symbol=None):
        if os.name != "posix":
            raise RuntimeError("Secure retrieval copying requires POSIX dirfd/O_NOFOLLOW")
        if not isinstance(public_issue, PublicIssue):
            raise TypeError("Pass a PublicIssue; dataset rows are never accepted")
        config = config or RetrievalConfig()
        run = self.work_root / ("retrieve-"+uuid.uuid4().hex)
        run.mkdir(mode=0o700)
        copied = run / "source"
        copied.mkdir(mode=0o700)
        try:
            inventory = source_inventory(Path(sanitized_source_root).absolute(), parse=False, copy_to=copied)
            if expected_source_sha256 is not None and inventory["source_sha256"] != expected_source_sha256:
                raise ValueError("Sanitized current source differs from pinned source identity")
            request = {"issue": asdict(public_issue), "config": asdict(config), "inventory": inventory,
                       "declared_anchor_symbol": declared_anchor_symbol,
                       "retriever_sha256": _hash({"retrieval_source": _hash(Path(__file__).read_bytes()),
                                                  "secure_scanner_source": _hash(Path(__file__).with_name("graphs.py").read_bytes())})}
            (run / "request.json").write_bytes(canonical_json(request))
            env = {"PATH": os.environ.get("PATH", ""), "PYTHONNOUSERSITE": "1", "PYTHONHASHSEED": "0",
                   "PYTHONPATH": str(Path(__file__).resolve().parent.parent), "HOME": str(run), "LANG": "C.UTF-8"}
            with (run / "worker.log").open("wb") as log:
                result = subprocess.run([self.python, "-m", "absent_edges.retrieval", "--worker", str(run)],
                                        cwd=run, env=env, stdout=log, stderr=subprocess.STDOUT,
                                        timeout=self.timeout, check=False)
            if result.returncode or not (run / "pool.json").is_file():
                raise RuntimeError("Bounded source retrieval worker failed; no partial pool is valid")
            pool = FrozenSourcePool((run / "pool.json").read_bytes())
            pool.validate()
            if pool.data["source_sha256"] != inventory["source_sha256"]:
                raise ValueError("Copied source changed during retrieval")
            return pool
        finally:
            shutil.rmtree(run)


def _worker(run):
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (2*1024**3, 2*1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    resource.setrlimit(resource.RLIMIT_FSIZE, (32*1024**2, 32*1024**2))
    request = json.loads((run / "request.json").read_bytes())
    files = dict(_regular_sources(run / "source"))
    hashes = {path: _hash(raw) for path, raw in files.items()}
    if hashes != request["inventory"]["files"]:
        raise ValueError("Copied source identity mismatch")
    pool = _rank_sources(files, PublicIssue.from_dict(request["issue"]), RetrievalConfig(**request["config"]),
                         request["retriever_sha256"], declared_anchor_symbol=request["declared_anchor_symbol"])
    pool.save(run / "pool.json")


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--worker":
        raise SystemExit("Retrieval worker requires --worker PATH")
    _worker(Path(sys.argv[2]))
