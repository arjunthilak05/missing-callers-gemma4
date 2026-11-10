"""Exp A tools: lexical search parity/budgets, read-only bash contract and jail, submit_answer v2 (D11.5),
invalid Unicode (F5), no-follow pathname expansion and checkout staging (F6), clean checkout manifests (F11),
graph_query for the T arms, and the operational appendix."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from absent_edges.data import canonical_json
from absent_edges.graphs import GRAPH_TOOL, GraphBackend, GraphFacts
from absent_edges.probe import tools
from absent_edges.probe.identity import file_sha256s, scan_package_files, source_tree_sha256
from absent_edges.probe.tools import (BASH_TOOL, CommandRejected, GlobBudget, GraphQueryTool, ReadOnlyBash,
                                      SandboxError, ToolBox, cap_output, discover_programs, lexical_candidates,
                                      lexical_search, package_sources, parse_readonly_command, validate_submission)

PROJECT = Path(__file__).resolve().parents[1]
FILES = {
    "pkg/__init__.py": "from .core import target as tgt_alias\n",
    "pkg/core.py": ("def target(x):\n    return x\n\n\ndef caller_one():\n    return target(1)\n\n\n"
                    "class Box:\n    def method(self):\n        return target(2)\n"),
    "pkg/other.py": ("from pkg.core import target as aliased\nimport pkg.core\n\n\ndef via_alias():\n"
                     "    return aliased(3)\n\n\ndef via_module():\n    return pkg.core.target(4)\n"),
    "pkg/big.txt": "".join(f"line {n:04d} of a large non-python file\n" for n in range(400)),
    "tests/test_core.py": "from pkg.core import target as testalias\n\n\ndef test_target():\n    assert target(1) == 1\n",
    "docs/conf.py": "target = 1\n",
    "setup.py": "from pkg.core import target\n",
}
PROGRAMS = {"grep": "/usr/bin/grep", "cat": "/usr/bin/cat", "head": "/usr/bin/head", "tail": "/usr/bin/tail",
            "sed": "/usr/bin/sed", "ls": "/usr/bin/ls", "find": "/usr/bin/find", "wc": "/usr/bin/wc",
            "sort": "/usr/bin/sort", "uniq": "/usr/bin/uniq", "cut": "/usr/bin/cut", "nl": "/usr/bin/nl",
            "awk": "/usr/bin/gawk"}
SURROGATE = "\ud800"


def make_repo(root, files=FILES):
    for relative, text in files.items():
        path = Path(root) / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("utf-8"))
    return Path(root)


def _sandbox_available():
    if os.name != "posix" or not os.path.exists("/usr/bin/unshare") or "awk" not in discover_programs():
        return False
    probe = subprocess.run(["/usr/bin/unshare", "--user", "--map-root-user", "--net", "--mount", "--pid", "--fork",
                            "--kill-child", "/usr/bin/true"], capture_output=True, timeout=30)
    return probe.returncode == 0


SANDBOX = _sandbox_available()


def facts_document(source_sha256="f" * 64, backend="locagent"):
    nodes = [{"id": "pkg/core.py::target", "kind": "function", "path": "pkg/core.py", "start_line": 1, "end_line": 2,
              "snippet": "def target(x):\n    return x", "snippet_end_line": 2},
             {"id": "pkg/core.py::caller_one", "kind": "function", "path": "pkg/core.py", "start_line": 5,
              "end_line": 6, "snippet": "def caller_one():\n    return target(1)", "snippet_end_line": 6},
             {"id": "pkg/other.py::via_alias", "kind": "function", "path": "pkg/other.py", "start_line": 5,
              "end_line": 6, "snippet": "def via_alias():\n    return aliased(3)", "snippet_end_line": 6}]
    edges = [{"id": "e" * 24, "source": "pkg/core.py::caller_one", "target": "pkg/core.py::target",
              "relation": "invokes"},
             {"id": "d" * 24, "source": "pkg/other.py::via_alias", "target": "pkg/core.py::target",
              "relation": "invokes"}]
    return {"backend": backend, "source_sha256": source_sha256, "nodes": nodes, "edges": edges, "error": None,
            "coverage": {"completeness": "unknown", "scope": "Python functions and methods under pkg"}}


@unittest.skipUnless(os.name == "posix", "secure no-follow scanning requires POSIX")
class LexicalSearchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = make_repo(Path(self.temporary.name) / "repo")

    def tearDown(self):
        self.temporary.cleanup()

    def graphs_search(self, query, limit=20):
        package_only = Path(self.temporary.name) / "package-only"
        if not package_only.exists():
            shutil.copytree(self.root / "pkg", package_only)
        backend = GraphBackend.__new__(GraphBackend)  # source_reference_search needs only cache_root
        backend.cache_root = Path(self.temporary.name) / "graph-cache"
        backend.cache_root.mkdir(exist_ok=True)
        return backend.source_reference_search(package_only, query, limit)

    def test_parity_with_graph_backend_semantics(self):
        for query in ("pkg/core.py::target", "target", "pkg.core.target", "pkg/core.py::Box.method",
                      "aliased", "via_module"):
            with self.subTest(query=query):
                expected = self.graphs_search(query)
                self.assertFalse(expected["truncated"], "fixture must not hit the graphs byte cap")
                ours = lexical_search(self.root, ["pkg"], query)
                self.assertEqual(ours["kind"], "lexical_candidates_not_proven_edges")
                self.assertEqual([("pkg/" + hit["path"], hit["line"], hit["text"]) for hit in expected["hits"]],
                                 [(hit["path"], hit["line"], hit["text"]) for hit in ours["hits"]])
                self.assertFalse(ours["truncated"])

    def test_alias_ranking_and_package_scope(self):
        result = lexical_search(self.root, ["pkg"], "pkg/core.py::target")
        located = [(hit["path"], hit["line"]) for hit in result["hits"]]
        self.assertEqual(located, [("pkg/__init__.py", 1), ("pkg/core.py", 1), ("pkg/core.py", 6),
                                   ("pkg/core.py", 11), ("pkg/other.py", 1), ("pkg/other.py", 10),
                                   ("pkg/other.py", 6)])
        self.assertTrue(all(not path.startswith(("tests/", "docs/")) and path != "setup.py" for path, _ in located))
        names = lexical_candidates(package_sources(self.root, ["pkg"])[0], "pkg/core.py::target")["names"]
        self.assertEqual(names, ["aliased", "target", "tgt_alias"])  # test-file alias stays out of scope

    def test_source_hash_matches_identity_scope(self):
        files, hashes, tree = package_sources(self.root, ["pkg/"])
        scope = scan_package_files(self.root, ["pkg"])["files"]
        self.assertEqual(sorted(files), scope)
        self.assertEqual(tree, source_tree_sha256(file_sha256s(self.root, scope)))
        self.assertEqual(lexical_search(self.root, ["pkg"], "target")["source_sha256"], tree)

    def test_byte_bound_budget_truncates_and_never_exceeds(self):
        padding = "x" * 150
        lines = "".join(f"def user_{n}():\n    return target({n}) + len('{padding}')\n\n" for n in range(120))
        make_repo(self.root, {"pkg/many.py": "from pkg.core import target\n\n" + lines})
        result = lexical_search(self.root, ["pkg"], "target", limit=20, token_budget=2000)
        self.assertLessEqual(len(canonical_json(result)), 2000)
        self.assertTrue(result["truncated"])
        self.assertGreater(result["candidate_count"], 20)
        self.assertLess(len(result["hits"]), 20)
        self.assertEqual(result["budget"], {"max_hits": 20, "max_tokens": 2000, "token_bound": "utf8_bytes_upper_bound"})
        bigger = dict(result, hits=lexical_search(self.root, ["pkg"], "target", token_budget=20000)["hits"][:len(result["hits"]) + 1])
        self.assertGreater(len(canonical_json(bigger)), 2000)

    def test_exact_counter_hook_bounds_tokens_and_limit(self):
        lines = "".join(f"def user_{n}():\n    return target({n})\n\n" for n in range(60))
        make_repo(self.root, {"pkg/many.py": lines})
        counted = []

        def counter(text):
            counted.append(len(text))
            return len(text.encode()) // 4
        counter.name = "quarter_bytes_fixture"
        result = lexical_search(self.root, ["pkg"], "target", limit=20, token_budget=2000, token_counter=counter)
        self.assertEqual(len(result["hits"]), 20)
        self.assertLessEqual(len(canonical_json(result)) // 4, 2000)
        self.assertEqual(result["budget"]["token_bound"], "exact:quarter_bytes_fixture")
        self.assertTrue(counted)
        self.assertEqual(len(lexical_search(self.root, ["pkg"], "target", limit=3)["hits"]), 3)

    def test_invalid_queries_and_limits(self):
        for query, limit in (("target(", 5), ("", 5), ("x" * 201, 5), ("target", 0), ("target", 21),
                             ("target", "5"), ("target", True), ("tar" + SURROGATE, 5)):
            with self.subTest(query=ascii(query[:10]), limit=limit), self.assertRaises(ValueError):
                lexical_search(self.root, ["pkg"], query, limit)

    def test_line_numbers_follow_python_not_splitlines(self):
        make_repo(self.root, {"pkg/feed.py": "x = 1\x0c\nvalue = target\n"})
        hits = [hit for hit in lexical_search(self.root, ["pkg"], "target")["hits"] if hit["path"] == "pkg/feed.py"]
        self.assertEqual([hit["line"] for hit in hits], [2])

    def test_symlinks_are_not_followed(self):
        outside = Path(self.temporary.name) / "outside.py"
        outside.write_text("secret_target = target\n")
        (self.root / "pkg" / "link.py").symlink_to(outside)
        paths = {hit["path"] for hit in lexical_search(self.root, ["pkg"], "target")["hits"]}
        self.assertNotIn("pkg/link.py", paths)


class SubmissionTests(unittest.TestCase):
    def test_valid_entries_are_normalized(self):
        submission, validation = validate_submission({"sites": [
            {"path": " /repo/./pkg/core.py ", "line": "6"}, {"path": "pkg/core.py", "line": 11},
            {"path": "./pkg/other.py", "line": 10, "function": "via_module"}], "complete": True})
        self.assertEqual(submission, {"sites": [{"path": "pkg/core.py", "line": 6}, {"path": "pkg/core.py", "line": 11},
                                                {"path": "pkg/other.py", "line": 10}], "complete": "yes"})
        self.assertEqual(validation, {"entries": 3, "valid": 3, "invalid": [], "normalized_entries": [0, 2],
                                      "dropped_keys": [{"index": 2, "keys": ["function"]}],
                                      "complete_from_boolean": True})
        self.assertEqual(validate_submission({"sites": [], "complete": "unknown"})[0], {"sites": [], "complete": "unknown"})
        self.assertEqual(validate_submission({"sites": [], "complete": False})[0]["complete"], "no")

    def test_invalid_entries_are_kept_for_the_scorer(self):
        raw = [{"path": "pkg/core.py", "line": 0}, {"path": "pkg/core.py", "line": 6.0}, "pkg/core.py:6",
               {"path": "", "line": 3}, {"path": "./", "line": 3}, {"path": 7, "line": 3}, {"line": 3},
               {"path": "pkg/core.py", "line": "6-8"}, {"path": "pkg/core.py", "line": True},
               {"path": "pkg/core.py", "line": 10**10}, None, ["pkg/core.py", 6], {"path": "pkg/core.py", "line": 6},
               {"path": "pkg/core.py", "line": 6}]
        submission, validation = validate_submission({"sites": raw, "complete": "no"})
        self.assertEqual(submission["sites"][:-2], raw[:-2])  # kept exactly as submitted, in order
        self.assertEqual(submission["sites"][-2:], [{"path": "pkg/core.py", "line": 6}] * 2)  # duplicates kept
        self.assertEqual([entry["index"] for entry in validation["invalid"]], list(range(12)))
        self.assertEqual(validation["invalid"][0]["reason"], "invalid_line")
        self.assertEqual(validation["invalid"][2]["reason"], "not_an_object")
        self.assertEqual(validation["invalid"][3]["reason"], "invalid_path")
        self.assertEqual((validation["entries"], validation["valid"]), (14, 2))

    def test_envelope_errors_are_rejected_with_a_reason(self):
        bad = [None, [], {"sites": []}, {"complete": "yes"}, {"sites": [], "complete": "maybe"},
               {"sites": [], "complete": "Yes"}, {"sites": {}, "complete": "no"}, {"sites": "a.py:1", "complete": "no"},
               {"sites": [], "complete": "no", "extra": 1}, {"call_sites": [], "complete": "no"},
               {"sites": [{"path": "a.py", "line": 1}] * 501, "complete": "no"},
               {"sites": [{"path": "a" + SURROGATE, "line": 1}], "complete": "no"}]
        for index, arguments in enumerate(bad):
            with self.subTest(index=index), self.assertRaises(ValueError):
                validate_submission(arguments)
        with self.assertRaisesRegex(ValueError, "unknown fields call_sites; use sites and complete"):
            validate_submission({"call_sites": [], "sites": [], "complete": "no"})

    def test_policy_is_documented(self):
        policy = tools.submission_policy()
        self.assertEqual(policy["max_entries"], 500)
        self.assertIn("unresolved", policy["invalid_entries"])
        self.assertEqual(policy["schema"]["required"], ["sites", "complete"])


class CommandValidationTests(unittest.TestCase):
    def parse(self, command, checkout=None, budget=None):
        return parse_readonly_command(command, programs=PROGRAMS, checkout=checkout, budget=budget)

    def test_accepted_pipelines(self):
        accepted = {
            'grep -rn "target(" pkg': [["/usr/bin/grep", "-rn", "target(", "pkg"]],
            "grep -n target pkg/core.py | head -5": [["/usr/bin/grep", "-n", "target", "pkg/core.py"],
                                                     ["/usr/bin/head", "-5"]],
            "sed -n '1,5p' pkg/core.py": [["/usr/bin/sed", "--sandbox", "-n", "1,5p", "pkg/core.py"]],
            "sed -ne 1p -e 2p pkg/core.py": [["/usr/bin/sed", "--sandbox", "-ne", "1p", "-e", "2p", "pkg/core.py"]],
            "awk 'NR>=2 && NR<=3' pkg/core.py": [["/usr/bin/gawk", "--sandbox", "NR>=2 && NR<=3", "pkg/core.py"]],
            'find . -name "*.py" -type f': [["/usr/bin/find", ".", "-name", "*.py", "-type", "f"]],
            "cat pkg/core.py | wc -l": [["/usr/bin/cat", "pkg/core.py"], ["/usr/bin/wc", "-l"]],
            "sort f | uniq -c | sort -rn | head": [["/usr/bin/sort", "f"], ["/usr/bin/uniq", "-c"],
                                                  ["/usr/bin/sort", "-rn"], ["/usr/bin/head"]],
            "cut -d: -f1 f": [["/usr/bin/cut", "-d:", "-f1", "f"]],
            "nl f": [["/usr/bin/nl", "f"]],
            "tail -n 5 f": [["/usr/bin/tail", "-n", "5", "f"]],
            "grep -n 'def .*:$' f": [["/usr/bin/grep", "-n", "def .*:$", "f"]],
            'grep -n "foo$" f': [["/usr/bin/grep", "-n", "foo$", "f"]],
            'grep -E "a|b" f': [["/usr/bin/grep", "-E", "a|b", "f"]],
            "/usr/bin/grep -c x f": [["/usr/bin/grep", "-c", "x", "f"]],
            r"grep -n a\ b f": [["/usr/bin/grep", "-n", "a b", "f"]],
            "grep '' f": [["/usr/bin/grep", "", "f"]],
            "grep -n é f": [["/usr/bin/grep", "-n", "é", "f"]],
        }
        for command, expected in accepted.items():
            with self.subTest(command=command):
                self.assertEqual(self.parse(command), expected)

    def test_rejected_commands(self):
        rejected = ["cat a; rm b", "grep x a && cat b", "grep x a || true", "cat a > b", "cat < a",
                    "cat $(ls)", "cat `ls`", "(ls)", "ls &", 'python3 -c "print(1)"', "rm -rf x",
                    "find . -exec rm '{}' ';'", "find . -execdir ls ';'", "find . -delete", "find . -fprint out",
                    "find . -ok ls ';'", "sed -i 's/a/b/' f", "sed 's/a/b/' f", "sed --in-place -n p f",
                    "sed -n --in p f", "sed -ni p f", "awk -o x f", "awk --profile '1' f", "awk -W dump-variables 1 f",
                    "sort -o out f", "sort --output=out f", "sort --compress-program=gzip f", "sort -ro out f",
                    "uniq a b", "tail -f f", "tail -5f f", "tail --follow f", 'grep x "unterminated',
                    "grep x 'unterminated", "cat $HOME", "cat ${x}", "cat $'x'", "FOO=1 grep x f", "ls ~",
                    "grep x f # comment", "", "   ", "ls\nrm x", "| grep x", "grep x f |", "grep x f | | wc",
                    "bash -c ls", "sh", "env", "xargs ls", "git log", "curl x", "tee out", "/usr/bin/python3 x",
                    "l* -l", "grep x f |& cat", "echo {a,b}", "cat f " + "| cat " * 8, "grep x\x00 f", "dd if=x",
                    "grep " + SURROGATE + " f", "cat 'a" + SURROGATE + "b'"]
        for command in rejected:
            with self.subTest(command=ascii(command)), self.assertRaises(CommandRejected):
                self.parse(command)

    def test_reasons_are_informative(self):
        with self.assertRaisesRegex(CommandRejected, "one pipeline per call"):
            self.parse("ls; ls")
        with self.assertRaisesRegex(CommandRejected, "allowed programs: grep"):
            self.parse("python3 -c 1")
        with self.assertRaisesRegex(CommandRejected, "only with -n"):
            self.parse("sed p f")
        with self.assertRaisesRegex(CommandRejected, "invalid Unicode"):
            self.parse("grep " + SURROGATE + " f")

    @unittest.skipUnless(os.name == "posix", "glob expansion fixture uses POSIX paths")
    def test_glob_expansion_matches_bash_rules(self):
        with tempfile.TemporaryDirectory() as directory:
            root = make_repo(directory)
            (root / "pkg" / ".hidden.py").write_text("x = 1\n")
            self.assertEqual(self.parse("ls pkg/*.py", root),
                             [["/usr/bin/ls", "pkg/__init__.py", "pkg/core.py", "pkg/other.py"]])
            self.assertEqual(self.parse("ls /repo/pkg/c*.py", root), [["/usr/bin/ls", "/repo/pkg/core.py"]])
            self.assertEqual(self.parse("ls 'pkg/*.py'", root), [["/usr/bin/ls", "pkg/*.py"]])
            self.assertEqual(self.parse("ls nomatch*", root), [["/usr/bin/ls", "nomatch*"]])
            self.assertEqual(self.parse("ls ../*", root), [["/usr/bin/ls", "../*"]])
            self.assertEqual(self.parse("ls pkg/../*", root), [["/usr/bin/ls", "pkg/../*"]])
            self.assertEqual(self.parse("ls /etc/*", root), [["/usr/bin/ls", "/etc/*"]])
            self.assertEqual(self.parse("ls pkg/.h*", root), [["/usr/bin/ls", "pkg/.hidden.py"]])

    @unittest.skipUnless(os.name == "posix", "symlink fixture")
    def test_glob_never_follows_symlinks_out_of_the_checkout(self):
        """F6: a planted up-link symlink is listed as a name but never descended, and host names never leak."""
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = make_repo(base / "checkout")
            outside = base / "outside" / "secret-dir"
            outside.mkdir(parents=True)
            (outside / "host-only.txt").write_text("x\n")
            (root / "pkg" / "up_link").symlink_to("../..")
            (root / "pkg" / "abs_link").symlink_to(str(base / "outside"))
            expanded = self.parse("ls pkg/*/*", root)[0][1:]
            self.assertEqual(expanded, ["pkg/*/*"])  # no real directory under pkg: the pattern stays literal
            deep = self.parse("ls */*/*/*", root)[0][1:]
            self.assertEqual(deep, ["*/*/*/*"])
            self.assertIn("pkg/up_link", self.parse("ls pkg/*", root)[0])  # the name itself is listed
            for argv in (self.parse("ls pkg/*/*/*", root), self.parse("ls pkg/abs_link/*", root)):
                self.assertFalse(any("secret-dir" in part or "host-only" in part or "outside" in part
                                     for part in argv[0][1:]), argv)

    @unittest.skipUnless(os.name == "posix", "directory fixture")
    def test_glob_expansion_is_bounded_in_entries_and_time(self):
        with tempfile.TemporaryDirectory() as directory:
            root = make_repo(directory)
            wide = root / "wide"
            wide.mkdir()
            for index in range(60):
                (wide / f"d{index:02d}").mkdir()
                for inner in range(5):
                    (wide / f"d{index:02d}" / f"f{inner}.py").write_text("")
            with self.assertRaisesRegex(CommandRejected, "examined more than 100 directory entries"):
                self.parse("ls wide/*/*.py", root, budget=GlobBudget(max_entries=100))
            ticks = iter(range(0, 10**6, 10))
            with self.assertRaisesRegex(CommandRejected, "took too long"):
                self.parse("ls wide/*/*.py", root, budget=GlobBudget(max_seconds=5, clock=lambda: next(ticks)))
            self.assertEqual(len(self.parse("ls wide/*/*.py", root)[0]), 1 + 300)

    def test_schema_style_matches_existing_tools(self):
        self.assertEqual(BASH_TOOL["function"]["parameters"]["required"], ["command"])
        names = [tool["function"]["name"] for tool in tools.tool_schemas()]
        self.assertEqual(names, ["source_reference_search", "bash", "submit_answer"])
        self.assertEqual([tool["function"]["name"] for tool in tools.tool_schemas(graph_query=True)],
                         ["source_reference_search", "bash", "submit_answer", "graph_query"])
        self.assertEqual(tools.tool_schemas(graph_query=True)[-1], GRAPH_TOOL)
        for tool in tools.tool_schemas():
            self.assertEqual(tool["type"], "function")
            self.assertFalse(tool["function"]["parameters"]["additionalProperties"])
        submit = tools.SUBMIT_ANSWER_TOOL["function"]["parameters"]
        self.assertEqual((submit["required"], sorted(submit["properties"]["sites"]["items"]["properties"])),
                         (["sites", "complete"], ["line", "path"]))

    def test_cap_output_marks_truncation_within_limit(self):
        text, truncated = cap_output(b"a" * 100, 100, 4000)
        self.assertEqual((text, truncated), ("a" * 100, False))
        text, truncated = cap_output(b"b" * 4000, 9000, 4000)
        self.assertTrue(truncated)
        self.assertLessEqual(len(text.encode()), 4000)
        self.assertIn("of 9000 bytes]", text)
        text, truncated = cap_output("é".encode() * 2000, 4000, 4000)
        self.assertFalse(truncated)
        text, truncated = cap_output(b"\xff" * 3000, 3000, 4000)  # replacement chars expand
        self.assertTrue(truncated)
        self.assertLessEqual(len(text.encode()), 4000)

    def test_operational_details(self):
        details = tools.operational_details()
        self.assertEqual(details["bash"]["allowed_programs"], list(tools.ALLOWED_PROGRAMS))
        self.assertEqual(details["bash"]["timeout_seconds"], 30.0)
        self.assertEqual(details["bash"]["output_bytes"], 4000)
        self.assertIn("output truncated", details["bash"]["truncation_marker"])
        messages = details["bash"]["rejection_messages"]
        self.assertIn("'redirections are not allowed'", messages)
        self.assertGreater(len(messages), 25)
        self.assertEqual(details["schemas"]["graph_query"], GRAPH_TOOL)
        self.assertEqual(details["source_reference_search"]["max_bytes"], 2000)
        json.dumps(details)


@unittest.skipUnless(SANDBOX, "user/net/mount/pid namespaces and gawk required (run on Core2)")
class SandboxExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = make_repo(Path(self.temporary.name) / "repo")
        self.bash = ReadOnlyBash(self.root, Path(self.temporary.name) / "scratch")

    def tearDown(self):
        self.temporary.cleanup()

    def run_visible(self, command):
        return self.bash.run(command)["visible"]

    def test_selftest_proves_isolation(self):
        report = self.bash.selftest()
        self.assertTrue(report["passed"])
        self.assertEqual(report["uid"], 0)
        self.assertEqual(report["root_entries"], ["bin", "lib", "lib64", "repo", "sbin", "usr"])
        self.assertEqual(report["repo_write_errno"], 30)  # EROFS
        self.assertIsNotNone(report["network_errno"])

    def test_pipelines_execute_without_a_shell(self):
        visible = self.run_visible('grep -rn "target(" pkg | sort')
        self.assertEqual(visible["exit_code"], 0)
        self.assertEqual(visible["output"].splitlines(), [
            "pkg/core.py:11:        return target(2)", "pkg/core.py:1:def target(x):",
            "pkg/core.py:6:    return target(1)", "pkg/other.py:10:    return pkg.core.target(4)"])
        self.assertEqual(self.run_visible("sed -n '6p' pkg/core.py")["output"], "    return target(1)\n")
        self.assertEqual(self.run_visible("awk 'NR==1' pkg/core.py")["output"], "def target(x):\n")
        self.assertEqual(self.run_visible("grep -c def pkg/core.py pkg/other.py | cut -d: -f2 | sort -n | head -1")["output"], "2\n")
        self.assertEqual(self.run_visible("ls pkg/*.py | wc -l")["output"].strip(), "3")
        self.assertEqual(self.run_visible("nl pkg/__init__.py")["exit_code"], 0)
        missing = self.run_visible("cat missing.txt")
        self.assertNotEqual(missing["exit_code"], 0)
        self.assertIn("No such file", missing["output"])

    def test_output_cap_and_truncation_marker(self):
        record = self.bash.run("cat pkg/big.txt")
        visible = record["visible"]
        self.assertTrue(visible["truncated"])
        self.assertLessEqual(len(visible["output"].encode()), 4000)
        self.assertIn("[output truncated: showing the first", visible["output"])
        self.assertEqual(record["output_bytes"], len(FILES["pkg/big.txt"]))

    def test_timeout_kills_the_command(self):
        slow = ReadOnlyBash(self.root, Path(self.temporary.name) / "slow", timeout=0.3)
        visible = slow.run("grep -r zzz_never_present_zzz /usr")["visible"]
        self.assertIsNone(visible["exit_code"])
        self.assertIn("timed out", visible["error"])

    def test_jail_hides_host_paths_and_blocks_writes(self):
        self.assertEqual(self.run_visible("ls /")["output"].split(), ["bin", "lib", "lib64", "repo", "sbin", "usr"])
        for path in ("/etc/passwd", "/mnt", "/home", "/proc/self/environ", "/dev/shm", "/tmp"):
            with self.subTest(path=path):
                self.assertNotEqual(self.run_visible(f"ls {path}")["exit_code"], 0)
        sed = self.run_visible("sed -n 'w pkg/evil' pkg/core.py")
        self.assertNotEqual(sed["exit_code"], 0)
        awk = self.run_visible("awk 'BEGIN{print 1 > \"pkg/evil\"}'")
        self.assertNotEqual(awk["exit_code"], 0)
        system = self.run_visible("awk 'BEGIN{system(\"ls\")}'")
        self.assertIn("sandbox", system["output"])
        self.assertFalse((self.root / "pkg" / "evil").exists())

    def test_environment_is_scrubbed(self):
        names = set(self.run_visible("awk 'BEGIN{for (k in ENVIRON) print k}'")["output"].split())
        self.assertLessEqual(names, {"PATH", "LC_ALL", "LANG", "AWKPATH", "AWKLIBPATH"})

    def test_rejections_never_execute(self):
        record = self.bash.run("cat pkg/core.py; touch pkg/evil")
        self.assertEqual(record["status"], "rejected")
        self.assertTrue(record["visible"]["error"].startswith("command rejected:"))
        self.assertFalse((self.root / "pkg" / "evil").exists())

    def test_lone_surrogates_are_a_rejected_command_not_an_exception(self):
        """F5: ReadOnlyBash.run never raises on invalid Unicode; the record and visible result serialize."""
        for command in ("grep " + SURROGATE + " pkg/core.py", "cat 'pkg/core" + SURROGATE + ".py'",
                        "ls " + SURROGATE * 3):
            with self.subTest(command=ascii(command)):
                record = self.bash.run(command)
                self.assertEqual(record["status"], "rejected")
                self.assertIn("invalid Unicode", record["visible"]["error"])
                canonical_json(record["visible"])
                json.dumps(record, ensure_ascii=False).encode("utf-8")

    def test_contract_probe_proves_program_sandboxes(self):
        report = self.bash.contract_probe()
        self.assertTrue(all(report[name]["refused"] for name in ("sed_write", "awk_system", "awk_redirect")))
        self.assertFalse((self.root / ".ae-probe").exists())
        crippled = ReadOnlyBash(self.root, Path(self.temporary.name) / "crippled",
                                programs={k: v for k, v in discover_programs().items() if k != "awk"})
        with self.assertRaisesRegex(SandboxError, "awk"):
            crippled.contract_probe()

    def test_relative_paths_are_made_absolute(self):
        previous = os.getcwd()
        os.chdir(self.temporary.name)
        try:
            relative = ReadOnlyBash("repo", "relative-scratch")
            self.assertTrue(relative.selftest()["passed"])
            self.assertEqual(relative.run("ls pkg/core.py")["visible"]["output"], "pkg/core.py\n")
        finally:
            os.chdir(previous)

    def test_staged_checkout_never_exposes_escaping_symlinks(self):
        """F6 end to end: staging drops a symlink to the host; inside links stay; the jail sees only /repo."""
        base = Path(self.temporary.name)
        (base / "host-secret").mkdir()
        (base / "host-secret" / "token.txt").write_text("secret\n")
        (self.root / "pkg" / "escape").symlink_to(str(base / "host-secret"))
        (self.root / "pkg" / "inside.py").symlink_to("core.py")
        record = tools.stage_checkout(self.root, base / "staged")
        self.assertEqual([entry["path"] for entry in record["neutralized_symlinks"]], ["pkg/escape"])
        jail = ReadOnlyBash(base / "staged", base / "staged-scratch")
        self.assertEqual(jail.run("ls pkg/escape")["visible"]["exit_code"], 2)
        self.assertEqual(jail.run("cat pkg/inside.py")["visible"]["output"], FILES["pkg/core.py"])
        self.assertNotIn("secret", jail.run("grep -r secret .")["visible"]["output"])

    def test_sandbox_failures_are_infrastructure(self):
        broken = ReadOnlyBash(self.root, Path(self.temporary.name) / "broken", python="/nonexistent/python")
        with self.assertRaises(SandboxError):
            broken.run("ls")
        missing = ReadOnlyBash(self.root, Path(self.temporary.name) / "missing", unshare="/nonexistent/unshare")
        with self.assertRaises(SandboxError):
            missing.run("ls")


@unittest.skipUnless(os.name == "posix", "ToolBox search needs POSIX scanning")
class ToolBoxTests(unittest.TestCase):
    class FakeBash:
        def __init__(self):
            self.commands = []

        def run(self, command):
            self.commands.append(command)
            return {"status": "executed", "visible": {"exit_code": 0, "output": "ok\n", "truncated": False}}

    def test_dispatch_and_argument_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = make_repo(directory)
            bash = self.FakeBash()
            box = ToolBox(root, ["pkg/"], bash)
            record, visible = box.execute("source_reference_search", {"query": "pkg/core.py::target", "limit": 2})
            self.assertEqual(len(visible["hits"]), 2)
            self.assertEqual(record["status"], "executed")
            for arguments in ({"query": "x", "bogus": 1}, {"limit": 3}, {"query": "x("}, {"query": "x", "limit": 50},
                              {"query": 5}):
                with self.subTest(arguments=arguments):
                    record, visible = box.execute("source_reference_search", arguments)
                    self.assertEqual(record["status"], "invalid_arguments")
                    self.assertEqual(visible["hits"], [])
                    self.assertTrue(visible["error"])
            self.assertEqual(box.execute("bash", {"command": "ls"})[1]["output"], "ok\n")
            for arguments in ({"command": 5}, {}, {"command": "ls", "x": 1}):
                self.assertEqual(box.execute("bash", arguments)[0]["status"], "invalid_arguments")
            self.assertEqual(bash.commands, ["ls"])
            for name in ("submit_answer", "graph_query", "python"):
                with self.assertRaises(ValueError):
                    box.execute(name, {})

    def test_search_echoes_invalid_unicode_safely(self):
        """F5: a surrogate query gets a visible error whose echo is ASCII-escaped and serializes."""
        with tempfile.TemporaryDirectory() as directory:
            box = ToolBox(make_repo(directory), ["pkg"], self.FakeBash())
            for arguments in ({"query": "tar" + SURROGATE + "get"}, {"query": SURROGATE, "bogus": 1}):
                record, visible = box.execute("source_reference_search", arguments)
                self.assertEqual(record["status"], "invalid_arguments")
                self.assertEqual(visible["hits"], [])
                encoded = canonical_json(visible)
                self.assertNotIn(SURROGATE, encoded.decode("utf-8"))
            self.assertEqual(json.loads(canonical_json(box.execute("source_reference_search",
                                                                   {"query": "tar" + SURROGATE})[1]))["query"],
                             "tar\\ud800")


class GraphQueryToolTests(unittest.TestCase):
    def setUp(self):
        self.document = facts_document()
        self.raw = canonical_json(self.document)

    def test_answers_are_graph_facts_rendered_like_c1(self):
        tool = GraphQueryTool(self.raw, expected_sha256=tools._sha256(self.raw))
        record = tool.run({"symbol": "pkg/core.py::target", "direction": "incoming"})
        expected = GraphFacts(self.raw).query("pkg/core.py::target", "invokes", "incoming", 30).render(
            "C1", max_output_bytes=12000)
        self.assertEqual(record["visible"], expected)
        self.assertNotIn("guidance", record["visible"])
        self.assertNotIn("verification", record["visible"])
        self.assertEqual((record["status"], record["state"], record["returned_count"]), ("executed", "matches", 2))
        self.assertLessEqual(len(canonical_json(record["visible"])), 12000)
        default = tool.run({"symbol": "pkg/core.py::caller_one"})  # GraphToolSet defaults: invokes, outgoing, 30
        self.assertEqual(default["arguments"], {"symbol": "pkg/core.py::caller_one", "relation": "invokes",
                                                "direction": "outgoing", "limit": 30})
        self.assertEqual(default["returned_count"], 1)
        unknown = tool.run({"symbol": "pkg/missing.py::nothing"})
        self.assertEqual((unknown["state"], unknown["symbol_indexed"]), ("symbol_not_indexed", False))
        self.assertEqual(tool.describe()["facts_sha256"], tools._sha256(self.raw))

    def test_invalid_arguments_are_visible_errors(self):
        tool = GraphQueryTool(self.raw)
        for arguments in ({}, {"symbol": ""}, {"symbol": 5}, {"symbol": "x", "limit": 0}, {"symbol": "x", "limit": 201},
                          {"symbol": "x", "limit": 2.0}, {"symbol": "x", "limit": True}, {"symbol": "x", "relation": "calls"},
                          {"symbol": "x", "direction": "up"}, {"symbol": "x", "extra": 1},
                          {"symbol": "pkg/core.py::target" + SURROGATE}, {"symbol": "x" * 5000}):
            with self.subTest(arguments=ascii(arguments)[:60]):
                record = tool.run(arguments)
                self.assertEqual(record["status"], "invalid_arguments")
                self.assertTrue(record["visible"]["error"].startswith("invalid_arguments:"))
                canonical_json(record["visible"])

    def test_refuses_facts_that_are_not_natural_locagent_facts(self):
        with self.assertRaises(ValueError):
            GraphQueryTool(self.raw, expected_sha256="0" * 64)
        for document in (facts_document(backend="pycg"), dict(facts_document(), error="engine failed"),
                         dict(facts_document(), edges=facts_document()["edges"] + [
                             {"id": "c" * 24, "source": "pkg/x.py::ghost", "target": "pkg/core.py::target",
                              "relation": "invokes"}]),
                         dict(facts_document(), nodes=facts_document()["nodes"] * 2)):
            with self.assertRaises(ValueError):
                GraphQueryTool(canonical_json(document))

    def test_toolbox_routes_graph_query_only_when_given(self):
        if os.name != "posix":
            self.skipTest("ToolBox needs POSIX scanning")
        with tempfile.TemporaryDirectory() as directory:
            root = make_repo(directory)
            box = ToolBox(root, ["pkg"], ToolBoxTests.FakeBash(), graph=GraphQueryTool(self.raw))
            record, visible = box.execute("graph_query", {"symbol": "pkg/core.py::target", "direction": "incoming"})
            self.assertEqual(visible["returned_count"], 2)
            self.assertIsNotNone(box.describe()["graph_query"])


@unittest.skipUnless(os.name == "posix", "symlink and permission fixtures")
class CheckoutTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.root = make_repo(self.base / "fam")
        (self.root / "empty_dir").mkdir()
        (self.root / ".git").mkdir()
        (self.root / ".git" / "HEAD").write_text("ref\n")
        (self.root / "pkg" / "__pycache__").mkdir()
        (self.root / "pkg" / "__pycache__" / "core.cpython-311.pyc").write_bytes(b"\0")
        (self.root / "pkg" / "inside.py").symlink_to("core.py")
        (self.root / "pkg" / "escape").symlink_to("../../outside")

    def tearDown(self):
        for path in self.base.rglob("*"):
            if path.is_dir() and not path.is_symlink():
                path.chmod(0o755)
        self.temporary.cleanup()

    def test_tree_record_covers_every_file_directory_and_symlink(self):
        tree = tools.checkout_tree(self.root)
        self.assertEqual(sorted(tree["files"]), sorted(FILES))
        self.assertEqual(tree["directories"], ["docs", "empty_dir", "pkg", "tests"])
        self.assertEqual(tree["symlinks"], {"pkg/escape": "../../outside", "pkg/inside.py": "core.py"})
        self.assertEqual(tree["skipped"], [".git", "pkg/__pycache__"])
        self.assertEqual(tree["tree_sha256"], tools.tree_digest(tree["files"], tree["symlinks"], tree["directories"]))
        self.assertEqual(tree["tree_sha256"], tools._sha256(canonical_json({
            "protocol": "expA-checkout-tree-v1", "files": tree["files"], "symlinks": tree["symlinks"],
            "directories": tree["directories"]})))

    def test_clean_entry_and_staging_agree(self):
        entry = tools.clean_checkout_entry("fam", self.root, ["pkg"], commit="c" * 40)
        self.assertEqual(entry["escaping_symlinks"], [{"path": "pkg/escape", "target": "../../outside",
                                                       "reason": "lexically_outside"}])
        self.assertEqual(entry["symlinks"], {"pkg/inside.py": "core.py"})
        self.assertEqual(entry["skipped"], [".git", "pkg/__pycache__"])
        self.assertEqual(entry["package_files"], ["pkg/__init__.py", "pkg/core.py", "pkg/other.py"])
        self.assertEqual(tools.checkout_entry_problems(entry), [])
        record = tools.stage_checkout(self.root, self.base / "staged" / "fam")
        self.assertEqual(record["neutralized_symlinks"][0]["path"], "pkg/escape")
        self.assertEqual(sorted(record["skipped"]), [".git", "pkg/__pycache__"])
        staged = tools.checkout_tree(self.base / "staged" / "fam")
        comparison = tools.compare_tree(entry, staged)
        self.assertTrue(comparison["matches"], comparison)
        self.assertEqual(staged["tree_sha256"], entry["tree_sha256"])
        with self.assertRaises(PermissionError):
            (self.base / "staged" / "fam" / "pkg" / "new.py").write_text("x")

    def test_unexpected_missing_and_changed_files_are_detected(self):
        entry = tools.clean_checkout_entry("fam", self.root, ["pkg"])
        (self.root / "pkg" / "extra.txt").write_text("leftover\n")
        (self.root / "docs" / "conf.py").unlink()
        (self.root / "pkg" / "core.py").write_text("changed\n")
        (self.root / "new_dir").mkdir()
        comparison = tools.compare_tree(entry, tools.checkout_tree(self.root))
        self.assertFalse(comparison["matches"])
        self.assertEqual((comparison["unexpected_files"], comparison["missing_files"], comparison["changed_files"],
                          comparison["unexpected_directories"]),
                         (["pkg/extra.txt"], ["docs/conf.py"], ["pkg/core.py"], ["new_dir"]))
        tampered = dict(entry, tree_sha256="0" * 64)
        self.assertIn("tree_sha256 does not match its file, symlink and directory lists",
                      tools.checkout_entry_problems(tampered))

    def test_manifest_document_and_stager_digest_agree(self):
        manifest = tools.checkouts_manifest([tools.clean_checkout_entry("fam", self.root, ["pkg"])])
        self.assertEqual((manifest["schema"], manifest["tree_protocol"]),
                         ("expA-clean-checkouts-v1", "expA-checkout-tree-v1"))
        stage_path = PROJECT / "scripts" / "expA_stage.py"
        if not stage_path.is_file():
            self.skipTest("scripts/expA_stage.py is not present")
        spec = importlib.util.spec_from_file_location("expA_stage_for_test", stage_path)
        stage = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(stage)
        entry = manifest["families"]["fam"]
        self.assertEqual(stage.tree_digest(entry["files"], entry["symlinks"], entry["directories"]),
                         entry["tree_sha256"])
        self.assertEqual(stage.TREE_PROTOCOL, tools.TREE_PROTOCOL)
        self.assertEqual(stage.CHECKOUTS_SCHEMA, tools.CHECKOUTS_SCHEMA)
        package_digest = getattr(stage, "package_tree_sha256", None) or getattr(stage, "package_tree_digest", None)
        if package_digest is not None:
            self.assertEqual(package_digest(entry["files"], entry["package_files"]), entry["package_tree_sha256"])

    def test_staging_refuses_special_files_and_existing_destinations(self):
        os.mkfifo(self.root / "pkg" / "pipe")
        with self.assertRaisesRegex(ValueError, "device, socket or FIFO"):
            tools.stage_checkout(self.root, self.base / "staged-fifo")
        (self.root / "pkg" / "pipe").unlink()
        (self.base / "exists").mkdir()
        with self.assertRaises(FileExistsError):
            tools.stage_checkout(self.root, self.base / "exists")


if __name__ == "__main__":
    unittest.main()
