"""Exp A follow-up A2: the grep-missed definition of the hard rule (A2 review, finding 1).

The fixture of tests/test_probe_items_hard.py cannot tell the protocol's grep-missed rule from several plausible
variants. This two-file SYNTHETIC fixture pins it:
- the pool is the runtime-witnessed callers none of whose site spans contains a grep_all line of X in the same file;
- a call through a local alias (``fn = helper; return fn(value)``) is in the pool, although grep localizes its
  function (the "strict" enclosing-function rule would drop it);
- a dynamic caller whose call line number equals a lexical line of X in ANOTHER file is in the pool (the path
  matters, in the builder and in verify);
- a multi-line call whose X-naming line is not its first line is not (the whole span counts);
- a call through a frozen import alias (``from .extra import target as tgt``) is not (aliases are lexical names);
- a caller verified by L and J only, with no trace site, is not (only runtime-witnessed callers);
- a function with one grep-findable runtime caller is excluded as verified_below_2: rule 4 comes before the hard rule.
Plus a hard build of the fixture and verify. Each case kills a mutant that the earlier tests let survive
(M1-M6 and M16 of the review).
"""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_probe_items import build, make_spec  # noqa: E402
from test_probe_truth import site  # noqa: E402

from absent_edges.probe import items, truth  # noqa: E402

EXTRA, OTHER = "pkg/extra.py", "pkg/other.py"
EXTRA_TEXT = '''"""Reviewer fixture."""


def target(value):
    return value + 1


def direct_a(value):
    return target(value)


def chained(holder, value):
    return (holder
            .target(value))


def lj_only(value):
    g = target
    return g(value)


def helper(value):
    return value - 1


def helper_direct(value):
    return helper(value)


def local_alias(value):
    fn = helper
    return fn(value)


def single(value):
    return value * 2


def single_caller(value):
    return single(value)
'''
OTHER_TEXT = '''"""Other module."""
from .extra import target as tgt

TABLE = {"t": tgt}


def via_table(value):
    handler = TABLE["t"]
    return handler(value)


def use_alias(value):
    return tgt(value)
'''
T, H, S = EXTRA + "::target", EXTRA + "::helper", EXTRA + "::single"


def records():
    """SYNTHETIC truth records: trace sites (runtime-witnessed) for every caller but lj_only, which only L and J
    report."""
    trace = [site(EXTRA + "::direct_a", T, EXTRA, "    return target(value)", EXTRA_TEXT),
             site(EXTRA + "::chained", T, EXTRA, ".target(value))", EXTRA_TEXT),
             site(OTHER + "::via_table", T, OTHER, "return handler(value)", OTHER_TEXT),
             site(OTHER + "::use_alias", T, OTHER, "return tgt(value)", OTHER_TEXT),
             site(EXTRA + "::helper_direct", H, EXTRA, "return helper(value)", EXTRA_TEXT),
             site(EXTRA + "::local_alias", H, EXTRA, "return fn(value)", EXTRA_TEXT),
             site(EXTRA + "::single_caller", S, EXTRA, "return single(value)", EXTRA_TEXT)]
    static = [site(EXTRA + "::lj_only", T, EXTRA, "return g(value)", EXTRA_TEXT)]
    return trace, static, static


class GrepMissedDefinitionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.base = Path(self.directory.name)
        self.spec = make_spec(self.base, records=records(), jedi_errors=(),
                              files={"pkg/__init__.py": "", EXTRA: EXTRA_TEXT, OTHER: OTHER_TEXT})
        self.scope = truth.PackageScope(self.base / "repo", ["pkg/"])
        self.table = truth.TruthTable(self.scope)
        trace, locagent, jedi = records()
        self.table.add_records("trace", trace)
        self.table.add_records("L", locagent)
        self.table.add_records("J", jedi)

    def tearDown(self):
        self.directory.cleanup()

    def test_pool(self):
        # via_table: its only site line number is a lexical line of X in ANOTHER file (path matters);
        # chained: multi-line call span whose X-naming line is not the first line (whole span counts);
        # use_alias: call through a frozen import alias (aliases are lexical names);
        # lj_only: verified by L and J only, no trace site (not runtime-witnessed, never in the pool).
        self.assertEqual(items.grep_missed_callers(self.scope, self.table, T), [OTHER + "::via_table"])
        # local_alias: grep LOCALIZES the function (fn = helper) but misses its site: still in the pool.
        self.assertEqual(items.grep_missed_callers(self.scope, self.table, H), [EXTRA + "::local_alias"])

    def test_exclusion_is_recorded_after_rule_5(self):
        # single has one runtime caller, and grep finds its site: verified_below_2 (rule 4) is recorded, not the
        # hard rule's exclusion.
        rows = {row["identity"]: row for row in items.candidate_rows("fixture", self.scope, self.table,
                                                                     omit_rule=items.HARD_OMIT_RULE)}
        self.assertEqual(rows[S]["excluded"], "verified_below_2")

    def test_build_and_verify(self):
        _, records_ = build(self.base, self.spec, name="hard", omit_rule=items.HARD_OMIT_RULE)
        by_target = {record["target"]["identity"]: record for record in records_}
        self.assertEqual(by_target[T]["omitted"]["identity"], OTHER + "::via_table")
        self.assertEqual(by_target[H]["omitted"]["identity"], EXTRA + "::local_alias")
        self.assertTrue(items.verify_items(self.base / "hard")["ok"])


if __name__ == "__main__":
    unittest.main()
