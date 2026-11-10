"""Exp A follow-up A2: the opt-in hard-omission rule (``build --omit-rule first_not_grep_findable``)."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_probe_items import build, make_spec, run_cli  # noqa: E402
from test_probe_truth import site  # noqa: E402

from absent_edges.data import canonical_json  # noqa: E402
from absent_edges.probe import items, truth  # noqa: E402

PATH = "pkg/hard.py"
TEXT = '''"""Hard-omission fixture."""


def target(value):
    return value + 1


def direct_a(value):
    return target(value)


def direct_b(value):
    return target(value) * 2


TABLE = {"plus": target}


def dispatch(kind, value):
    handler = TABLE[kind]
    return handler(value)


def apply(fn, value):
    return fn(value)


def plain(value):
    return value - 1


def plain_a(value):
    return plain(value)


def plain_b(value):
    return plain(value) + 1


def triple(value):
    return value * 3


def triple_a(value):
    return triple(value)


def triple_b(value):
    return triple(value) - 3


STEPS = [triple]


def run_steps(value):
    for step in STEPS:
        value = step(value)
    return value
'''
TARGET, PLAIN, TRIPLE = PATH + "::target", PATH + "::plain", PATH + "::triple"
DIRECT = [PATH + "::direct_a", PATH + "::direct_b"]
MISSED = [PATH + "::apply", PATH + "::dispatch"]  # their only sites name no lexical name of X
TRIPLE_DIRECT, TRIPLE_MISSED = [PATH + "::triple_a", PATH + "::triple_b"], [PATH + "::run_steps"]


def hard_trace():
    return [site(DIRECT[0], TARGET, PATH, "    return target(value)", TEXT),
            site(DIRECT[1], TARGET, PATH, "return target(value) * 2", TEXT),
            site(PATH + "::dispatch", TARGET, PATH, "return handler(value)", TEXT),
            site(PATH + "::apply", TARGET, PATH, "return fn(value)", TEXT),
            site(PATH + "::plain_a", PLAIN, PATH, "    return plain(value)", TEXT),
            site(PATH + "::plain_b", PLAIN, PATH, "return plain(value) + 1", TEXT),
            site(TRIPLE_DIRECT[0], TRIPLE, PATH, "    return triple(value)", TEXT),
            site(TRIPLE_DIRECT[1], TRIPLE, PATH, "return triple(value) - 3", TEXT),
            site(TRIPLE_MISSED[0], TRIPLE, PATH, "value = step(value)", TEXT)]


def hard_spec(base):
    return make_spec(base, records=(hard_trace(), [], []), files={"pkg/__init__.py": "", PATH: TEXT}, jedi_errors=())


def rewrite_items(directory, records):
    """Replace items.jsonl and re-pin its sha256 in the manifest (a tampered copy that passes the hash check)."""
    raw = b"".join(canonical_json(row) + b"\n" for row in records)
    (directory / "items.jsonl").write_bytes(raw)
    manifest = json.loads((directory / "manifest.json").read_bytes())
    manifest["outputs"]["items.jsonl"]["sha256"] = hashlib.sha256(raw).hexdigest()
    (directory / "manifest.json").write_bytes(canonical_json(manifest) + b"\n")


class HardOmissionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.base = Path(self.directory.name)
        self.spec = hard_spec(self.base)

    def tearDown(self):
        self.directory.cleanup()

    def test_fixture_discriminates_the_two_rules(self):
        # TRIPLE's default C* is a direct caller grep finds; TARGET's default C* is already grep-missed.
        self.assertIn(items.omitted_caller(TRIPLE, TRIPLE_DIRECT + TRIPLE_MISSED), TRIPLE_DIRECT)
        self.assertIn(items.omitted_caller(TARGET, DIRECT + MISSED), MISSED)

    def test_exclusion_order_inserts_the_hard_rule_after_rule_5(self):
        order = items.exclusion_order(items.HARD_OMIT_RULE)
        self.assertEqual(items.exclusion_order(), items.EXCLUSION_ORDER)
        self.assertEqual(items.exclusion_order(items.OMIT_RULE), items.EXCLUSION_ORDER)
        self.assertEqual(order.index(items.HARD_EXCLUSION), order.index("no_runtime_witnessed_caller") + 1)
        self.assertEqual([reason for reason in order if reason != items.HARD_EXCLUSION], list(items.EXCLUSION_ORDER))
        self.assertEqual(items.OMIT_RULES, (items.OMIT_RULE, items.HARD_OMIT_RULE))
        with self.assertRaises(ValueError):
            items.exclusion_order("no_such_rule")

    def test_grep_missed_callers_follow_the_grep_all_site_rule(self):
        scope = truth.PackageScope(self.base / "repo", ["pkg/"])
        table = truth.TruthTable(scope)
        table.add_records("trace", hard_trace())
        self.assertEqual(items.grep_missed_callers(scope, table, TARGET), sorted(MISSED))
        self.assertEqual(items.grep_missed_callers(scope, table, TRIPLE), TRIPLE_MISSED)
        self.assertEqual(items.grep_missed_callers(scope, table, PLAIN), [])

    def test_hard_rule_omits_the_first_grep_missed_caller(self):
        manifest, records = build(self.base, self.spec, name="hard", omit_rule=items.HARD_OMIT_RULE)
        by_target = {record["target"]["identity"]: record for record in records}
        self.assertEqual(sorted(by_target), sorted([TARGET, TRIPLE]))
        for target, direct, missed in ((TARGET, DIRECT, MISSED), (TRIPLE, TRIPLE_DIRECT, TRIPLE_MISSED)):
            with self.subTest(target=target):
                record = by_target[target]
                strata = {entry["identity"]: entry["stratum"] for entry in record["V"]}
                self.assertEqual(sorted(strata), sorted(direct + missed))
                self.assertEqual({strata[caller] for caller in missed}, {"dynamic"})
                self.assertEqual(record["omitted"]["identity"], items.omitted_caller(target, missed))
                self.assertEqual(record["omitted"]["rule"], items.HARD_OMIT_RULE)
                self.assertEqual(record["covariates"]["stratum"], "dynamic")
                self.assertIn(items.HARD_EXCLUSION, record["eligibility"]["rules_passed"])
                sites = next(entry["sites"] for entry in record["V"]
                             if entry["identity"] == record["omitted"]["identity"])
                lexical = {(row["path"], row["line"]) for row in record["lexical_sites"]}
                self.assertTrue(lexical)
                self.assertFalse(any(path == s["path"] and s["lineno"] <= line <= s["end_lineno"]
                                     for s in sites for path, line in lexical))
        # TRIPLE's default C* (a direct caller) is not grep-missed, so the hard rule takes the next one in key order.
        self.assertNotEqual(by_target[TRIPLE]["omitted"]["identity"],
                            items.omitted_caller(TRIPLE, TRIPLE_DIRECT + TRIPLE_MISSED))
        rows = {json.loads(line)["identity"]: json.loads(line)
                for line in (self.base / "hard" / "eligibility" / "fixture.jsonl").read_text("utf-8").splitlines()}
        self.assertEqual(rows[PLAIN]["excluded"], items.HARD_EXCLUSION)
        self.assertEqual(manifest["parameters"]["omit_rule"], items.HARD_OMIT_RULE)
        self.assertEqual(manifest["parameters"]["exclusion_order"], list(items.exclusion_order(items.HARD_OMIT_RULE)))
        self.assertEqual(manifest["counts"]["funnel_total"][items.HARD_EXCLUSION], 1)
        self.assertEqual(manifest["families"]["fixture"]["funnel"]["excluded"][items.HARD_EXCLUSION], 1)
        self.assertTrue(items.verify_items(self.base / "hard")["ok"])

    def test_default_rule_is_unchanged(self):
        manifest, records = build(self.base, self.spec, name="default")
        by_target = {record["target"]["identity"]: record for record in records}
        self.assertEqual(sorted(by_target), sorted([PLAIN, TARGET, TRIPLE]))
        self.assertEqual(by_target[TARGET]["omitted"]["identity"], items.omitted_caller(TARGET, DIRECT + MISSED))
        self.assertEqual(by_target[TRIPLE]["omitted"]["identity"],
                         items.omitted_caller(TRIPLE, TRIPLE_DIRECT + TRIPLE_MISSED))
        self.assertEqual(by_target[TRIPLE]["covariates"]["stratum"], "direct")
        self.assertEqual({record["omitted"]["rule"] for record in records}, {items.OMIT_RULE})
        self.assertNotIn(items.HARD_EXCLUSION, by_target[TARGET]["eligibility"]["rules_passed"])
        self.assertEqual(manifest["parameters"]["omit_rule"], items.OMIT_RULE)
        self.assertEqual(manifest["parameters"]["exclusion_order"], list(items.EXCLUSION_ORDER))
        self.assertNotIn(items.HARD_EXCLUSION, manifest["counts"]["funnel_total"])
        self.assertTrue(items.verify_items(self.base / "default")["ok"])
        # Passing the default rule explicitly writes the same bytes as not passing it.
        build(self.base, self.spec, name="default-explicit", omit_rule=items.OMIT_RULE)
        for relative in ("items.jsonl", "manifest.json", "payloads/fixture.jsonl", "facts/fixture.jsonl",
                         "eligibility/fixture.jsonl", "graph_facts/fixture.json"):
            self.assertEqual((self.base / "default" / relative).read_bytes(),
                             (self.base / "default-explicit" / relative).read_bytes(), relative)

    def test_verify_rejects_a_grep_findable_or_mislabelled_omitted_caller(self):
        build(self.base, self.spec, name="hard", omit_rule=items.HARD_OMIT_RULE)
        records = [json.loads(line) for line in (self.base / "hard" / "items.jsonl").read_bytes().splitlines()]
        position = next(index for index, record in enumerate(records) if record["target"]["identity"] == TRIPLE)
        easy = items.omitted_caller(TRIPLE, TRIPLE_DIRECT + TRIPLE_MISSED)  # what the default rule would omit

        tampered = self.base / "t-easy"
        shutil.copytree(self.base / "hard", tampered)
        changed = [dict(record) for record in records]
        changed[position] = {**records[position],
                             "omitted": {**records[position]["omitted"], "identity": easy, "stratum": "direct",
                                         "order_key": items.omit_key(TRIPLE, easy)},
                             "covariates": {**records[position]["covariates"], "stratum": "direct"}}
        rewrite_items(tampered, changed)
        with self.assertRaisesRegex(ValueError, "omitted caller rule"):
            items.verify_items(tampered)

        tampered = self.base / "t-rule"
        shutil.copytree(self.base / "hard", tampered)
        changed = [dict(record) for record in records]
        changed[position] = {**records[position], "omitted": {**records[position]["omitted"], "rule": items.OMIT_RULE}}
        rewrite_items(tampered, changed)
        with self.assertRaisesRegex(ValueError, "needs the build's rule"):
            items.verify_items(tampered)

    def test_cli_build_takes_the_omit_rule(self):
        spec_path = self.base / "cli.spec.json"
        spec_path.write_text(json.dumps(self.spec), encoding="utf-8")
        summary = json.loads(run_cli("build", "--spec", str(spec_path), "--out", str(self.base / "cli-hard"),
                                     "--no-c4", "--omit-rule", items.HARD_OMIT_RULE))
        self.assertEqual(summary["counts"]["items"], 2)
        self.assertEqual(summary["families"]["fixture"]["funnel"]["excluded"][items.HARD_EXCLUSION], 1)
        self.assertTrue(json.loads(run_cli("verify", "--items-dir", str(self.base / "cli-hard")))["ok"])
        summary = json.loads(run_cli("build", "--spec", str(spec_path), "--out", str(self.base / "cli-default"),
                                     "--no-c4"))
        self.assertEqual(summary["counts"]["items"], 3)
        self.assertNotIn(items.HARD_EXCLUSION, summary["families"]["fixture"]["funnel"]["excluded"])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            run_cli("build", "--spec", str(spec_path), "--out", str(self.base / "cli-bad"), "--no-c4",
                    "--omit-rule", "no_such_rule")
        self.assertFalse((self.base / "cli-bad").exists())


if __name__ == "__main__":
    unittest.main()
