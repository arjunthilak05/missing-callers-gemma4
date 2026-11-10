#!/usr/bin/env python3
"""Exp A: which of the library's own tests executed the omitted caller's call site (C* -> X), and a
descriptive "broken tests" count for the agent's submissions.

ANALYSIS RECORD. Descriptive and exploratory; not preregistered; no tests of
significance. Standard library only (Python 3.10+). Every input is read, never written.

Inputs
------
--items        frozen ``items-main/items.jsonl`` (schema expA-item-v2)
--traces-root  ``traces-dd842ab6b``; the main truth input is ``<family>/<--trace-subdir>`` (trace-1, D12.3)
--scores       a scored run's ``scores.jsonl`` (rows of kind "episode" with ``site_details``)
--junit-root   optional: ``registry/work``; ``<family>/junit.xml`` is the registry's untraced baseline run
--registry     optional: ``registry/registry.json``; when given, each junit file must match its recorded sha256
--out-dir      receives TESTS_PER_CSTAR.json and TESTS_PER_CSTAR.md

What the trace records identify (absent_edges/probe/tracer_plugin.py)
--------------------------------------------------------------------
The plugin keeps ONE record per distinct (caller, callee, line, caller_kind, caller_scope) key. Each record
carries ``first_test``: the pytest node id that was running when the key was first observed, or a phase
label (``<startup>`` while conftest files load, ``<collection>`` while test modules are imported,
``<between-tests>``, ``<sessionfinish>``). Later tests that execute the same call are not recorded. The
number of distinct tests that execute a call site is therefore NOT identified by the frozen records. This
script reports the closest valid units:

* witnessed site: a site in sites(C, X) (frozen item truth) whose witnesses include "trace";
* witness records of a V caller C: trace records with callee == X, caller == C and caller_kind
  "function" (the truth build attaches exactly these; the script checks that every record falls in a
  witnessed site of C and every witnessed site holds at least one record);
* ``tests_lb``: the number of distinct pytest node ids among the first-execution labels of C*'s witness
  records. It is a LOWER BOUND on the number of distinct tests that execute some witnessed site of C*;
  phase labels are not tests and are reported separately (``phase_only`` when no node id exists);
* ``tests_lb_pass``: the same, restricted to node ids that passed in the untraced registry baseline.

Episode join ("broken tests", descriptive)
------------------------------------------
For an episode, a frozen site is covered when some valid submitted (path, line) lies on the site's path
within [lineno, end_lineno] (the rule ``site_recovered`` uses; the script checks that "any C* site
covered" reproduces ``site_recovered`` on every row). A non-submitted episode covers nothing. If X's
signature changed and only the covered sites were updated, every witnessed site left uncovered would
still make the old call. Per episode:

* ``exposed``: some witnessed C* site is uncovered (that call ran during the test session);
* ``broken_lb``: distinct node ids that first executed an uncovered witnessed C* site. Summed over the
  episodes of a condition this is a lower bound on the number of test executions (one per test per
  episode) that would reach an unchanged call. Whether such a test then fails depends on the change and
  on exception handling; "broken" is the modelled consequence, not an observation;
* ``broken_lb_pass``: the same, counting only node ids that passed in the untraced baseline;
* ``partial``: ``site_recovered`` is 1 but a witnessed C* site is uncovered;
* ``allv_*``: the same quantities over the witnessed sites of every runtime-witnessed V caller
  (secondary: a refactor relying on the submitted list misses retained callers too).

Example
-------
  python scripts/expA_tests_per_cstar.py --items items.jsonl --traces-root traces-dd842ab6b \\
      --scores scores.jsonl --junit-root registry/work --registry registry/registry.json --out-dir out/
"""
from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import sys
import xml.etree.ElementTree as ET

LABEL = "ANALYSIS RECORD"
SCHEMA = "absent-edges-expA-tests-per-cstar-v1"
PHASE_PREFIX = "<"
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 20261102
STRATA = ("direct", "alias", "callback", "dynamic")
BLOCK_ORDER = ("P", "S-VC", "S-NAT", "S-EDGE", "T")
CONDITION_ORDER = ("C0", "C1", "C2", "C3", "C4a", "T-offered", "T-recommended")
P_CONTRASTS = (("H3 analog", "C1", "C0"), ("H1 analog", "C3", "C2"), ("H2 analog", "C4a", "C1"))
JUNIT_STATES = ("passed", "failed", "error", "xfailed", "skipped", "not_found")


class InputError(RuntimeError):
    pass


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as stream:
        for number, text in enumerate(stream, 1):
            if text.strip():
                try:
                    yield json.loads(text)
                except json.JSONDecodeError as error:
                    raise InputError(f"{path}:{number}: {error}") from None


def is_phase(label: str) -> bool:
    return label.startswith(PHASE_PREFIX)


# --------------------------------------------------------------------------- inputs
def load_items(path):
    items = list(read_jsonl(path))
    seen = set()
    for item in items:
        for key in ("item_id", "family", "target", "omitted", "V"):
            if key not in item:
                raise InputError(f"item without {key}: {item.get('item_id')}")
        if item["item_id"] in seen:
            raise InputError(f"duplicate item_id {item['item_id']}")
        seen.add(item["item_id"])
    return items


def load_trace(directory: Path, family: str) -> dict:
    """Function-caller trace records grouped by callee, plus provenance of the trace run."""
    edges = directory / "edges.jsonl"
    if not edges.is_file():
        raise InputError(f"{family}: no {edges}")
    plugin = json.loads((directory / "plugin_manifest.json").read_text(encoding="utf-8"))
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    edges_sha = sha256_file(edges)
    by_callee = collections.defaultdict(list)
    records = 0
    for record in read_jsonl(edges):
        records += 1
        if record.get("caller_kind", "function") != "function":
            continue
        label = record.get("first_test")
        if not isinstance(label, str) or not label:
            raise InputError(f"{family}: a trace record has no first_test label")
        by_callee[record["callee"]].append({"caller": record["caller"], "path": record["path"],
                                            "line": record["line"], "label": label})
    tests = ((manifest.get("tests") or {}).get("plugin") or {})
    info = {
        "edges_sha256": edges_sha,
        "edges_sha256_matches_plugin_manifest": plugin.get("edges_sha256") == edges_sha,
        "records": records,
        "status": manifest.get("status"),
        "partial": manifest.get("partial"),
        "source_tree_sha256": ((manifest.get("repo") or {}).get("source_tree") or {}).get("sha256"),
        "collected": tests.get("collected"),
        "outcomes": tests.get("outcomes"),
        "wall_seconds": (manifest.get("run") or {}).get("wall_seconds"),
        "record_fields": "one record per (caller, callee, line, caller_kind, caller_scope) with first_test only",
    }
    return {"by_callee": by_callee, "info": info}


def trace_keys(path: Path) -> dict:
    """Every record of an edges file keyed like tracer.record_key, mapped to its first_test label."""
    keys = {}
    for record in read_jsonl(path):
        keys[(record["caller"], record["callee"], record["path"], record["line"],
              record.get("caller_kind", "function"), record.get("caller_scope") or "")] = record.get("first_test")
    return keys


def audit_family(main_dir: Path, audit_dir: Path, pairs: set) -> dict:
    """Records that differ between the main trace and the determinism-audit trace, and whether any of them is a
    witness record of an item's (V caller, X) pair."""
    main, audit = trace_keys(main_dir / "edges.jsonl"), trace_keys(audit_dir / "edges.jsonl")
    differing = (set(main) ^ set(audit)) | {key for key in set(main) & set(audit) if main[key] != audit[key]}
    touching = sorted({(key[0], key[1]) for key in differing if (key[0], key[1]) in pairs})
    return {"audit_edges_sha256": sha256_file(audit_dir / "edges.jsonl"),
            "only_in_main": len(set(main) - set(audit)), "only_in_audit": len(set(audit) - set(main)),
            "label_differs": sum(1 for key in set(main) & set(audit) if main[key] != audit[key]),
            "differing_records": len(differing), "item_pairs_touched_count": len(touching),
            "item_pairs_touched": [list(pair) for pair in touching]}


def junit_key(nodeid: str):
    """pytest junitxml's (classname, name) for a node id (``mangle_test_address``)."""
    path, bracket, params = nodeid.partition("[")
    names = path.split("::")
    names[0] = re.sub(r"\.py$", "", names[0].replace("/", "."))
    names[-1] += bracket + params
    return ".".join(names[:-1]), names[-1]


def load_junit(path: Path) -> dict:
    states = {}
    root = ET.parse(path).getroot()
    for case in root.iter("testcase"):
        tags = {child.tag for child in case}
        if "failure" in tags:
            state = "failed"
        elif "error" in tags:
            state = "error"
        elif "skipped" in tags:
            skipped = case.find("skipped")
            state = "xfailed" if skipped is not None and skipped.get("type") == "pytest.xfail" else "skipped"
        else:
            state = "passed"
        key = (case.get("classname") or "", case.get("name") or "")
        previous = states.get(key)
        if previous is None or (previous == "passed" and state != "passed"):
            states[key] = state  # a teardown error after a pass keeps the worse state
    return states


# --------------------------------------------------------------------------- per-item truth
def innermost(spans, path, line):
    best = None
    for span in spans:
        if span["path"] == path and span["lineno"] <= line <= span["end_lineno"]:
            if best is None or (span["end_lineno"] - span["lineno"]) < (best["end_lineno"] - best["lineno"]):
                best = span
    return best


def caller_witness(caller: dict, records: list, checks: collections.Counter) -> list:
    """Witnessed sites of one V caller with the first-execution labels attached to each."""
    witnessed = [dict(site, labels=set()) for site in caller["sites"] if "trace" in site.get("witnesses", ())]
    if bool(witnessed) != bool(caller.get("runtime_witnessed")):
        checks["runtime_witnessed_flag_disagrees_with_sites"] += 1
    mine = [record for record in records if record["caller"] == caller["identity"]]
    for record in mine:
        site = innermost(witnessed, record["path"], record["line"])
        if site is None:
            checks["witness_records_outside_witnessed_sites"] += 1
            continue
        site["labels"].add(record["label"])
    for site in witnessed:
        if not site["labels"]:
            checks["witnessed_sites_without_records"] += 1
    path = caller["identity"].split("::", 1)[0]
    for record in records:
        if record["caller"] != caller["identity"] and innermost(witnessed, record["path"], record["line"]) \
                and record["path"] == path:
            checks["foreign_caller_records_inside_witnessed_sites"] += 1
    return witnessed


def item_row(item: dict, records: list, junit, checks: collections.Counter) -> dict:
    target = item["target"]["identity"]
    omitted = item["omitted"]["identity"]
    callers = {caller["identity"]: caller for caller in item["V"]}
    if omitted not in callers:
        raise InputError(f"{item['item_id']}: C* is not in V")
    witnessed = {}
    for identity, caller in callers.items():
        sites = caller_witness(caller, records, checks)
        if sites:
            witnessed[identity] = sites
    cstar_sites = witnessed.get(omitted, [])
    if not cstar_sites:
        raise InputError(f"{item['item_id']}: C* has no witnessed site")
    labels = set().union(*(site["labels"] for site in cstar_sites))
    node_ids = sorted(label for label in labels if not is_phase(label))
    phases = sorted(label for label in labels if is_phase(label))
    states = collections.Counter()
    for node in node_ids:
        states[junit_state(junit, node)] += 1
    lines = {record["line"] for record in records if record["caller"] == omitted}
    stratum = item["omitted"].get("stratum") or (item.get("covariates") or {}).get("stratum")
    return {
        "item_id": item["item_id"], "family": item["family"], "target": target, "cstar": omitted,
        "stratum": stratum, "n_V": len(callers), "n_runtime_witnessed_V": len(witnessed),
        "cstar_sites_total": len(callers[omitted]["sites"]), "cstar_sites_witnessed": len(cstar_sites),
        "cstar_witness_lines": len(lines),
        "tests_lb": len(node_ids), "tests_lb_pass": states.get("passed", 0),
        "phase_only": not node_ids, "phase_labels": phases,
        "baseline_states": {name: states.get(name, 0) for name in JUNIT_STATES} if junit is not None else None,
        "witnessing_node_ids": node_ids,
        "cstar_witnessed_sites": [{"path": s["path"], "lineno": s["lineno"], "end_lineno": s["end_lineno"],
                                   "kind": s["kind"], "labels": sorted(s["labels"])} for s in cstar_sites],
        "_all_sites": {identity: [{"path": s["path"], "lineno": s["lineno"], "end_lineno": s["end_lineno"],
                                   "labels": frozenset(s["labels"])} for s in sites]
                       for identity, sites in witnessed.items()},
        "_cstar_spans": [(s["path"], s["lineno"], s["end_lineno"]) for s in callers[omitted]["sites"]],
    }


def junit_state(junit, node_id: str) -> str:
    if junit is None:
        return "not_found"
    return junit.get(junit_key(node_id), "not_found")


# --------------------------------------------------------------------------- summaries
def describe(values) -> dict:
    values = sorted(values)
    n = len(values)
    if n == 0:
        return {"n": 0}
    if n == 1:
        q1 = q3 = values[0]
    else:
        q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
    return {"n": n, "median": statistics.median(values), "q1": q1, "q3": q3, "mean": sum(values) / n,
            "min": values[0], "max": values[-1],
            "share_ge1": sum(1 for v in values if v >= 1) / n, "share_ge5": sum(1 for v in values if v >= 5) / n,
            "count_ge1": sum(1 for v in values if v >= 1), "count_ge5": sum(1 for v in values if v >= 5)}


def item_summary(rows) -> dict:
    return {"tests_lb": describe([r["tests_lb"] for r in rows]),
            "tests_lb_pass": describe([r["tests_lb_pass"] for r in rows]),
            "cstar_witness_lines": describe([r["cstar_witness_lines"] for r in rows]),
            "cstar_sites_witnessed": describe([r["cstar_sites_witnessed"] for r in rows]),
            "phase_only": sum(1 for r in rows if r["phase_only"]),
            "items_with_unwitnessed_cstar_site": sum(1 for r in rows
                                                     if r["cstar_sites_total"] > r["cstar_sites_witnessed"]),
            "items": len(rows)}


def family_means(rows, metric):
    by_item = collections.defaultdict(list)
    for row in rows:
        by_item[(row["family"], row["item_id"])].append(row[metric])
    by_family = collections.defaultdict(list)
    for (family, _), values in by_item.items():
        by_family[family].append(sum(values) / len(values))
    return {family: sum(v) / len(v) for family, v in by_family.items()}


def bootstrap_ci(family_values: dict, rng: random.Random):
    names = sorted(family_values)
    if len(names) < 2:
        return None
    values = [family_values[name] for name in names]
    draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        sample = [values[rng.randrange(len(values))] for _ in values]
        draws.append(sum(sample) / len(sample))
    draws.sort()
    lo = draws[int(math.floor(0.025 * (BOOTSTRAP_DRAWS - 1)))]
    hi = draws[int(math.ceil(0.975 * (BOOTSTRAP_DRAWS - 1)))]
    return [lo, hi]


def equal_weight(rows, metric, rng):
    means = family_means(rows, metric)
    if not means:
        return None
    return {"estimate": sum(means.values()) / len(means), "families": len(means),
            "bootstrap95": bootstrap_ci(means, rng)}


def paired_contrast(rows_a, rows_b, metric, rng):
    """Equal-weight family mean of item differences (a - b), items present in both."""
    def item_means(rows):
        acc = collections.defaultdict(list)
        for row in rows:
            acc[(row["family"], row["item_id"])].append(row[metric])
        return {key: sum(v) / len(v) for key, v in acc.items()}
    a, b = item_means(rows_a), item_means(rows_b)
    by_family = collections.defaultdict(list)
    for key in sorted(set(a) & set(b)):
        by_family[key[0]].append(a[key] - b[key])
    means = {family: sum(v) / len(v) for family, v in by_family.items()}
    if not means:
        return None
    return {"estimate": sum(means.values()) / len(means), "families": len(means),
            "items": sum(len(v) for v in by_family.values()), "bootstrap95": bootstrap_ci(means, rng)}


EPISODE_METRICS = ("site_recovered", "exposed", "partial", "broken_lb", "broken_lb_pass", "phase_only_exposed",
                   "allv_exposed", "allv_broken_lb", "allv_broken_lb_pass")


def aggregate(rows, rng, *, with_family_means=True) -> dict:
    out = {"episodes": len(rows), "items": len({r["item_id"] for r in rows}),
           "families": len({r["family"] for r in rows}), "seeds": sorted({r["seed"] for r in rows}),
           "exits": dict(sorted(collections.Counter(r["exit_status"] for r in rows).items()))}
    for metric in EPISODE_METRICS:
        out["sum_" + metric] = sum(r[metric] for r in rows)
    submitted = [r for r in rows if r["exit_status"] == "Submitted"]
    out["submitted"] = len(submitted)
    out["sum_broken_lb_submitted"] = sum(r["broken_lb"] for r in submitted)
    out["sum_broken_lb_not_submitted"] = out["sum_broken_lb"] - out["sum_broken_lb_submitted"]
    out["exposed_submitted"] = sum(r["exposed"] for r in submitted)
    out["sum_allv_broken_lb_submitted"] = sum(r["allv_broken_lb"] for r in submitted)
    out["allv_exposed_submitted"] = sum(r["allv_exposed"] for r in submitted)
    out["mean_broken_lb_per_episode"] = out["sum_broken_lb"] / len(rows) if rows else None
    if with_family_means:
        out["equal_weight"] = {metric: equal_weight(rows, metric, rng)
                               for metric in ("site_recovered", "exposed", "broken_lb", "broken_lb_pass",
                                              "allv_broken_lb")}
    return out


# --------------------------------------------------------------------------- episodes
def covered(spans, points) -> list:
    return [any(p == path and lo <= line <= hi for p, line in points) for path, lo, hi in spans]


def episode_rows(scores_path, items_by_id, checks: collections.Counter):
    rows = []
    for record in read_jsonl(scores_path):
        if record.get("kind") != "episode":
            checks["score_rows_not_episode"] += 1
            continue
        if record.get("cell_state") != "completed":
            checks["score_rows_not_completed"] += 1
            continue
        item = items_by_id.get(record["item_id"])
        if item is None:
            raise InputError(f"score row for unknown item {record['item_id']}")
        details = record.get("site_details") or []
        points = {(d["path"], d["line"]) for d in details
                  if d.get("valid") and isinstance(d.get("path"), str) and type(d.get("line")) is int}
        if record.get("exit_status") != "Submitted" and points:
            checks["non_submitted_rows_with_sites"] += 1
        any_cstar = any(covered(item["_cstar_spans"], points))
        if int(any_cstar) != int(record.get("site_recovered") or 0):
            checks["site_recovered_not_reproduced"] += 1
        cstar_sites = item["_all_sites"][item["cstar"]]
        missed = [site for site, hit in zip(cstar_sites, covered(
            [(s["path"], s["lineno"], s["end_lineno"]) for s in cstar_sites], points)) if not hit]
        labels = set().union(*(s["labels"] for s in missed)) if missed else set()
        nodes = {label for label in labels if not is_phase(label)}
        allv_labels = set()
        allv_missed = 0
        for identity, sites in item["_all_sites"].items():
            hits = covered([(s["path"], s["lineno"], s["end_lineno"]) for s in sites], points)
            for site, hit in zip(sites, hits):
                if not hit:
                    allv_missed += 1
                    allv_labels |= site["labels"]
        allv_nodes = {label for label in allv_labels if not is_phase(label)}
        junit = item["_junit"]
        rows.append({
            "item_id": record["item_id"], "family": record["family"], "block": record["block"],
            "variant": record["variant"], "condition": record["condition"], "seed": record["seed"],
            "exit_status": record.get("exit_status"), "stratum": item["stratum"],
            "site_recovered": int(record.get("site_recovered") or 0),
            "exposed": int(bool(missed)), "partial": int(bool(missed) and bool(record.get("site_recovered"))),
            "missed_witnessed_cstar_sites": len(missed),
            "broken_lb": len(nodes),
            "broken_lb_pass": sum(1 for n in nodes if junit_state(junit, n) == "passed"),
            "phase_only_exposed": int(bool(missed) and not nodes),
            "allv_exposed": int(allv_missed > 0), "allv_missed_witnessed_sites": allv_missed,
            "allv_broken_lb": len(allv_nodes),
            "allv_broken_lb_pass": sum(1 for n in allv_nodes if junit_state(junit, n) == "passed"),
        })
    return rows


# --------------------------------------------------------------------------- markdown
def fmt(value, digits=3):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def fmt_ci(entry, digits=3):
    if not entry:
        return "n/a"
    ci = entry.get("bootstrap95")
    text = f"{entry['estimate']:.{digits}f}"
    if ci:
        text += f" [{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]"
    return text


def summary_line(name, s):
    t = s["tests_lb"]
    return (f"| {name} | {s['items']} | {fmt(t['median'], 1)} | {fmt(t['q1'], 1)}-{fmt(t['q3'], 1)} | "
            f"{t['count_ge1']} ({100 * t['share_ge1']:.1f}%) | {t['count_ge5']} ({100 * t['share_ge5']:.1f}%) | "
            f"{t['max']} | {s['phase_only']} | {fmt(s['cstar_witness_lines']['median'], 1)} "
            f"(max {s['cstar_witness_lines']['max']}) |")


def render_md(result) -> str:
    out = [f"# Tests that execute the omitted caller's call site (Exp A, run `{result['run_label']}`)", "",
           f"**{LABEL}.** Descriptive and exploratory, written after the main results were open; not "
           "preregistered; no significance tests. Script `scripts/expA_tests_per_cstar.py` "
           f"(sha256 `{result['script_sha256'][:12]}`), generated {result['generated_at_utc']}.", ""]
    k = result["key_numbers"]
    out += ["## Key numbers", "",
            f"- **The frozen traces do not identify how many tests execute a call site.** The tracer stores one "
            "record per distinct (caller, callee, line) with only the first test that executed it "
            "(`first_test`). Per-site test counts below are lower bounds, not counts.",
            f"- Every C* has at least one call site of X that ran during the traced test session (that is how C* "
            f"was chosen). For {k['items_with_node_id']}/{k['items']} items "
            f"({100 * k['items_with_node_id'] / k['items']:.1f}%) a pytest node id is known to have executed a "
            f"witnessed C* site. For the other {k['phase_only']} "
            f"({', '.join(f'{v} {name}' for name, v in k['phase_only_by_label'].items())}) the first execution "
            "happened while pytest imported test modules (`<collection>`) or loaded conftest files "
            "(`<startup>`); whether a test function executed it later is not recorded. They cluster in a few "
            f"libraries: {', '.join(f'{f} {n}' for f, n in k['phase_only_by_family'].items())}.",
            f"- Lower bound on distinct tests per C*: median {fmt(k['tests_lb']['median'], 1)}, IQR "
            f"{fmt(k['tests_lb']['q1'], 1)}-{fmt(k['tests_lb']['q3'], 1)}, >=1 for {k['tests_lb']['count_ge1']} "
            f"items, >=5 for {k['tests_lb']['count_ge5']}. The bound is censored at the number of distinct "
            "witnessed lines, so these numbers say nothing about the upper tail.",
            f"- Untraced baseline (registry junit) of the {k['node_ids']} witnessing node ids: "
            + ", ".join(f"{name} {count}" for name, count in k["baseline_states"].items() if count) + ".",
            f"- Block P, all {k['p_episodes']:,} episodes: a signature change applied only at the submitted lines would "
            f"leave an unchanged call executed during the test session in {k['p_exposed']} episodes; at least "
            f"{k['p_broken_lb']} test executions (distinct node ids per episode, summed) would reach such a "
            f"call ({k['p_broken_lb_pass']} counting only tests that pass in the untraced baseline). "
            f"{k['p_broken_lb_not_submitted']} of the {k['p_broken_lb']} "
            f"({100 * k['p_broken_lb_not_submitted'] / max(1, k['p_broken_lb']):.1f}%) come from episodes that "
            f"never submitted. {k['p_partial']} exposed episode(s) had site_recovered = 1 but left another "
            "witnessed C* site unchanged, so site_recovered tracks 'every test-exercised C* site covered' almost "
            f"exactly. {k['p_phase_only_exposed']} exposed episodes involve only import-time (phase-only) "
            "sites and add 0 to the node-id count.",
            "- Per condition (block P, equal-weight mean of the lower bound per episode, library bootstrap 95%): "
            + "; ".join(f"{c} {fmt_ci(v)}" for c, v in k["p_equal_weight_broken_lb"].items()) + ".",
            ""]
    out += ["## Definitions", ""]
    for name, text in result["definitions"].items():
        out.append(f"- **{name}**: {text}")
    out += ["", "## Per item: tests known to execute a witnessed C* site (lower bound)", "",
            "`tests_lb` = distinct pytest node ids among the first-execution labels of C*'s witness records. "
            "Phase-only items have `tests_lb` = 0.", "",
            "| Group | Items | Median | IQR | >=1 test | >=5 tests | Max | Phase-only | Witnessed lines: median |",
            "|---|---|---|---|---|---|---|---|---|"]
    out.append(summary_line("all", result["item_summaries"]["overall"]))
    for stratum in STRATA:
        if stratum in result["item_summaries"]["by_stratum"]:
            out.append(summary_line(f"stratum {stratum}", result["item_summaries"]["by_stratum"][stratum]))
    out += ["", "Histogram of `tests_lb` over items: "
            + ", ".join(f"{key}: {value}" for key, value in result["item_summaries"]["histogram_tests_lb"].items()),
            "", "By family:", "",
            "| Family | Items | Median | IQR | >=1 test | >=5 tests | Max | Phase-only | Witnessed lines: median |",
            "|---|---|---|---|---|---|---|---|---|"]
    for family, summary in result["item_summaries"]["by_family"].items():
        out.append(summary_line(family, summary))
    out += ["", "## Episodes: tests that would reach an unchanged C* call (descriptive \"broken tests\")", "",
            "Per condition, over every completed episode. `Exposed` = some witnessed C* site uncovered by the "
            "submission. `Broken (lb)` = sum over episodes of distinct node ids that first executed an "
            "uncovered witnessed C* site (lower bound on test executions reaching an unchanged call). "
            "`pass` = only node ids that passed in the untraced baseline. `Partial` = recovered but another "
            "witnessed C* site uncovered. `Phase-only` = exposed episodes whose uncovered sites carry only "
            "collection/startup labels. Equal-weight = item mean over seeds, then library mean, then the mean "
            "over libraries, with a library-cluster percentile bootstrap (10,000 draws, seed 20261102).", "",
            "| Block | Variant | Cond | Episodes | Submitted | Recovered | Exposed | Partial | Phase-only | "
            "Broken (lb) | from non-submitted | Broken (lb, pass) | Equal-weight mean per episode [95%] |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for group in result["episodes"]["by_condition"]:
        a = group["stats"]
        out.append(f"| {group['block']} | {group['variant']} | {group['condition']} | {a['episodes']} | "
                   f"{a['submitted']} | {a['sum_site_recovered']} | {a['sum_exposed']} | {a['sum_partial']} | "
                   f"{a['sum_phase_only_exposed']} | {a['sum_broken_lb']} | {a['sum_broken_lb_not_submitted']} | "
                   f"{a['sum_broken_lb_pass']} | {fmt_ci(a['equal_weight']['broken_lb'])} |")
    contrasts = result["episodes"].get("p_contrasts") or []
    if contrasts:
        out += ["", "Block P paired differences in `broken_lb` per episode (descriptive, equal-weight, bootstrap "
                "95%): " + "; ".join(f"{c['name']} {c['a']} - {c['b']} = {fmt_ci(c['broken_lb'])}"
                                     for c in contrasts) + "."]
    out += ["", "Block P by stratum of C* (sums over episodes):", "",
            "| Cond | Stratum | Episodes | Recovered | Exposed | Partial | Broken (lb) | Broken (lb, pass) |",
            "|---|---|---|---|---|---|---|---|"]
    for group in result["episodes"]["p_by_stratum"]:
        a = group["stats"]
        out.append(f"| {group['condition']} | {group['stratum']} | {a['episodes']} | {a['sum_site_recovered']} | "
                   f"{a['sum_exposed']} | {a['sum_partial']} | {a['sum_broken_lb']} | {a['sum_broken_lb_pass']} |")
    out += ["", "Secondary: every runtime-witnessed V caller (C* and the retained callers), same rules:", "",
            "| Block | Variant | Cond | Episodes | Exposed (any V) | of which submitted | Broken (lb, any V) | "
            "of which submitted | Broken (lb, any V, pass) | Equal-weight per episode |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for group in result["episodes"]["by_condition"]:
        a = group["stats"]
        out.append(f"| {group['block']} | {group['variant']} | {group['condition']} | {a['episodes']} | "
                   f"{a['sum_allv_exposed']} | {a['allv_exposed_submitted']} | {a['sum_allv_broken_lb']} | "
                   f"{a['sum_allv_broken_lb_submitted']} | {a['sum_allv_broken_lb_pass']} | "
                   f"{fmt_ci(a['equal_weight']['allv_broken_lb'])} |")
    out += ["", "## Checks", ""]
    for name, value in result["checks"].items():
        out.append(f"- {name}: {value}")
    out += ["", "## Caveats", ""]
    out += [f"- {text}" for text in result["caveats"]]
    out += ["", "## Getting the real per-test count (follow-up, not run here)", ""]
    out += [f"- {text}" for text in result["follow_up"]]
    out += ["", "## Inputs", "", "| Input | Path | sha256 |", "|---|---|---|"]
    for name, entry in result["inputs"].items():
        out.append(f"| {name} | `{entry['path']}` | `{entry.get('sha256') or 'n/a'}` |")
    out += ["", "Per-family trace provenance and the per-item rows (node ids per witnessed site) are in "
            "`TESTS_PER_CSTAR.json`.", ""]
    return "\n".join(out)


def audit_sentence(args, family_info) -> str:
    if not args.audit_subdir:
        return "The determinism-audit trace was not compared in this run."
    audits = {family: info.get("audit") or {} for family, info in family_info.items()}
    differing = sum(a.get("differing_records", 0) for a in audits.values())
    touching = sum(a.get("item_pairs_touched_count", 0) for a in audits.values())
    families = ", ".join(sorted(f for f, a in audits.items() if a.get("differing_records"))) or "no family"
    text = (f"The determinism-audit trace ({args.audit_subdir}) differs from {args.trace_subdir} in {differing} "
            f"records ({families}); ")
    if touching == 0:
        return text + "none of them belongs to an item's (V caller, X) pair, so every label used here is the same " \
                      "in both runs."
    return text + f"{touching} (V caller, X) pairs of items are touched (listed per family in the JSON)."


# --------------------------------------------------------------------------- main
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--items", required=True)
    parser.add_argument("--traces-root", required=True)
    parser.add_argument("--trace-subdir", default="trace-1")
    parser.add_argument("--audit-subdir", help="optional determinism-audit trace (trace-2) compared with the main one")
    parser.add_argument("--scores", required=True)
    parser.add_argument("--junit-root")
    parser.add_argument("--registry")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--run-label", default="main-20261101b")
    parser.add_argument("--expect-items-sha256")
    parser.add_argument("--expect-scores-sha256")
    parser.add_argument("--display-root", action="append", default=[], metavar="LOCAL=SHOWN",
                        help="rewrite a local path prefix in the recorded input paths")
    args = parser.parse_args(argv)

    def shown(path):
        text = str(path).replace("\\", "/")
        for pair in args.display_root:
            local, _, display = pair.partition("=")
            local = local.replace("\\", "/")
            if text.startswith(local):
                return display + text[len(local):]
        return text

    checks = collections.Counter()
    inputs = {"items": {"path": shown(args.items), "sha256": sha256_file(args.items)},
              "scores": {"path": shown(args.scores), "sha256": sha256_file(args.scores)}}
    for name, expected in (("items", args.expect_items_sha256), ("scores", args.expect_scores_sha256)):
        if expected and inputs[name]["sha256"] != expected:
            raise InputError(f"{name} sha256 {inputs[name]['sha256']} != expected {expected}")
    items = load_items(args.items)
    families = sorted({item["family"] for item in items})
    registry = None
    if args.registry:
        inputs["registry"] = {"path": shown(args.registry), "sha256": sha256_file(args.registry)}
        with open(args.registry, "r", encoding="utf-8") as stream:
            registry = {entry["family"]: entry for entry in json.load(stream)["families"]}
    traces, junits, family_info = {}, {}, {}
    for family in families:
        directory = Path(args.traces_root) / family / args.trace_subdir
        trace = load_trace(directory, family)
        traces[family] = trace["by_callee"]
        info = dict(trace["info"], trace_dir=shown(directory))
        if not info["edges_sha256_matches_plugin_manifest"]:
            raise InputError(f"{family}: edges.jsonl does not match its plugin manifest")
        source_hashes = {(item.get("render") or {}).get("source_sha256") for item in items
                         if item["family"] == family}
        info["source_tree_matches_items"] = source_hashes == {info["source_tree_sha256"]}
        if not info["source_tree_matches_items"]:
            raise InputError(f"{family}: trace source tree differs from the items' render.source_sha256")
        if args.audit_subdir:
            pairs = {(caller["identity"], item["target"]["identity"]) for item in items
                     if item["family"] == family for caller in item["V"]}
            info["audit"] = audit_family(directory, Path(args.traces_root) / family / args.audit_subdir, pairs)
        junits[family] = None
        if args.junit_root:
            path = Path(args.junit_root) / family / "junit.xml"
            if path.is_file():
                digest = sha256_file(path)
                expected = ((registry or {}).get(family) or {}).get("baseline", {}).get("junit_sha256")
                if registry is not None and digest != expected:
                    raise InputError(f"{family}: junit sha256 {digest} != registry {expected}")
                junits[family] = load_junit(path)
                info["junit"] = {"path": shown(path), "sha256": digest, "testcases": len(junits[family]),
                                 "matches_registry": None if registry is None else digest == expected,
                                 "registry_counts": ((registry or {}).get(family) or {}).get("baseline", {})
                                 .get("counts")}
            else:
                checks["families_without_junit"] += 1
        family_info[family] = info
    inputs["traces_root"] = {"path": shown(args.traces_root), "sha256": None,
                             "note": f"<family>/{args.trace_subdir}; per-family edges sha256 in families"}
    if args.junit_root:
        inputs["junit_root"] = {"path": shown(args.junit_root), "sha256": None,
                                "note": "<family>/junit.xml; per-family sha256 in families"}

    rows = []
    for item in items:
        records = traces[item["family"]].get(item["target"]["identity"], [])
        row = item_row(item, records, junits[item["family"]], checks)
        row["_junit"] = junits[item["family"]]
        rows.append(row)
    items_by_id = {row["item_id"]: row for row in rows}

    node_states = collections.Counter()
    node_total = 0
    for row in rows:
        for node in row["witnessing_node_ids"]:
            node_total += 1
            node_states[junit_state(row["_junit"], node)] += 1

    def summaries(group_rows):
        return item_summary(group_rows)

    by_family = {family: summaries([r for r in rows if r["family"] == family]) for family in families}
    by_stratum = {stratum: summaries([r for r in rows if r["stratum"] == stratum])
                  for stratum in STRATA if any(r["stratum"] == stratum for r in rows)}
    overall = summaries(rows)
    histogram = dict(sorted(collections.Counter(r["tests_lb"] for r in rows).items()))

    episodes = episode_rows(args.scores, items_by_id, checks)
    rng = random.Random(BOOTSTRAP_SEED)
    groups = collections.defaultdict(list)
    for row in episodes:
        groups[(row["block"], row["variant"], row["condition"])].append(row)

    def order(key):
        block, variant, condition = key
        return (BLOCK_ORDER.index(block) if block in BLOCK_ORDER else 99,
                CONDITION_ORDER.index(condition) if condition in CONDITION_ORDER else 99, variant)

    by_condition = [{"block": b, "variant": v, "condition": c, "stats": aggregate(groups[(b, v, c)], rng)}
                    for (b, v, c) in sorted(groups, key=order)]
    p_rows = {c: [r for r in episodes if r["block"] == "P" and r["condition"] == c]
              for c in ("C0", "C1", "C2", "C3", "C4a")}
    p_contrasts = []
    for name, a, b in P_CONTRASTS:
        if p_rows.get(a) and p_rows.get(b):
            p_contrasts.append({"name": name, "a": a, "b": b,
                                "broken_lb": paired_contrast(p_rows[a], p_rows[b], "broken_lb", rng),
                                "exposed": paired_contrast(p_rows[a], p_rows[b], "exposed", rng)})
    p_by_stratum = []
    for condition in ("C0", "C1", "C2", "C3", "C4a"):
        for stratum in STRATA:
            subset = [r for r in p_rows.get(condition, []) if r["stratum"] == stratum]
            if subset:
                p_by_stratum.append({"condition": condition, "stratum": stratum,
                                     "stats": aggregate(subset, rng, with_family_means=False)})
    p_by_family = {}
    for condition in ("C0", "C1", "C2", "C3", "C4a"):
        for family in families:
            subset = [r for r in p_rows.get(condition, []) if r["family"] == family]
            if subset:
                stats = aggregate(subset, rng, with_family_means=False)
                p_by_family.setdefault(family, {})[condition] = {
                    key: stats[key] for key in ("episodes", "sum_site_recovered", "sum_exposed", "sum_partial",
                                                "sum_broken_lb", "sum_broken_lb_pass")}

    p_all = [r for r in episodes if r["block"] == "P"]
    script_path = Path(__file__).resolve()
    result = {
        "label": LABEL, "schema": SCHEMA, "run_label": args.run_label,
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/expA_tests_per_cstar.py", "script_sha256": sha256_file(script_path),
        "python": sys.version.split()[0],
        "inputs": inputs,
        "definitions": {
            "trace record": "absent_edges/probe/tracer_plugin.py keeps one record per distinct (caller, callee, "
                            "line, caller_kind, caller_scope) key with `first_test`, the pytest node id running when "
                            "the key was first seen, or a phase label (<startup>, <collection>, <between-tests>, "
                            "<sessionfinish>). Later executions are not recorded.",
            "witnessed site": "a site of sites(C, X) in the frozen item whose witnesses include 'trace'.",
            "witness records": "trace-1 records with callee == X, caller == C, caller_kind 'function', each "
                               "attached to the innermost witnessed site of C containing its line.",
            "tests_lb": "distinct pytest node ids among the labels of C*'s witness records: a lower bound on the "
                        "number of distinct tests executing some witnessed C* site of X. The true number is not "
                        "identified by the frozen records.",
            "phase_only": "C* has no node-id label: its witnessed sites were first executed at <collection> or "
                          "<startup>.",
            "baseline state": "outcome of the node id in the registry's untraced baseline junit.xml (pytest "
                              "junitxml classname/name mangling); xfailed = <skipped type='pytest.xfail'>.",
            "covered": "a valid submitted (path, line) lies on the site's path within [lineno, end_lineno] "
                       "(site_recovered's rule); non-submitted episodes cover nothing.",
            "exposed": "some witnessed C* site is uncovered.",
            "broken_lb": "distinct node ids that first executed an uncovered witnessed C* site; summed over "
                         "episodes it lower-bounds test executions that would reach an unchanged call if the "
                         "signature change were applied only at the submitted lines.",
            "broken_lb_pass": "broken_lb restricted to node ids that passed in the untraced baseline.",
            "partial": "site_recovered = 1 while a witnessed C* site is uncovered.",
            "allv_*": "the same over the witnessed sites of every runtime-witnessed V caller.",
        },
        "checks": {
            "items": len(rows), "families": len(families),
            "episode_rows": len(episodes),
            "witness_records_outside_witnessed_sites": checks["witness_records_outside_witnessed_sites"],
            "witnessed_sites_without_records": checks["witnessed_sites_without_records"],
            "foreign_caller_records_inside_witnessed_sites": checks["foreign_caller_records_inside_witnessed_sites"],
            "runtime_witnessed_flag_disagrees_with_sites": checks["runtime_witnessed_flag_disagrees_with_sites"],
            "site_recovered_not_reproduced": checks["site_recovered_not_reproduced"],
            "non_submitted_rows_with_sites": checks["non_submitted_rows_with_sites"],
            "score_rows_not_completed": checks["score_rows_not_completed"],
            "families_without_junit": checks["families_without_junit"],
            "trace_edges_match_plugin_manifests": all(i["edges_sha256_matches_plugin_manifest"]
                                                      for i in family_info.values()),
            "trace_source_trees_match_items": all(i["source_tree_matches_items"] for i in family_info.values()),
            "trace_runs_complete": all(i["status"] == "complete" and not i["partial"] for i in family_info.values()),
            "witnessing_node_ids_found_in_baseline_junit": node_total - node_states.get("not_found", 0),
            "witnessing_node_ids": node_total,
            "audit_trace": args.audit_subdir,
            "audit_families_differing": sorted(f for f, i in family_info.items()
                                               if (i.get("audit") or {}).get("differing_records")),
            "audit_differing_records": sum((i.get("audit") or {}).get("differing_records", 0)
                                           for i in family_info.values()) if args.audit_subdir else None,
            "audit_item_pairs_touched": sum(
                (i.get("audit") or {}).get("item_pairs_touched_count", 0)
                for i in family_info.values()) if args.audit_subdir else None,
        },
        "key_numbers": {
            "items": len(rows), "items_with_node_id": sum(1 for r in rows if not r["phase_only"]),
            "phase_only": sum(1 for r in rows if r["phase_only"]),
            "phase_only_by_label": dict(sorted(collections.Counter(
                "+".join(r["phase_labels"]) for r in rows if r["phase_only"]).items())),
            "tests_lb": overall["tests_lb"], "node_ids": node_total,
            "baseline_states": {name: node_states.get(name, 0) for name in JUNIT_STATES},
            "phase_only_by_family": dict(sorted(collections.Counter(
                r["family"] for r in rows if r["phase_only"]).items(), key=lambda kv: (-kv[1], kv[0]))),
            "p_episodes": len(p_all), "p_exposed": sum(r["exposed"] for r in p_all),
            "p_partial": sum(r["partial"] for r in p_all), "p_broken_lb": sum(r["broken_lb"] for r in p_all),
            "p_broken_lb_pass": sum(r["broken_lb_pass"] for r in p_all),
            "p_broken_lb_not_submitted": sum(r["broken_lb"] for r in p_all if r["exit_status"] != "Submitted"),
            "p_phase_only_exposed": sum(r["phase_only_exposed"] for r in p_all),
            "p_equal_weight_broken_lb": {g["condition"]: g["stats"]["equal_weight"]["broken_lb"]
                                         for g in by_condition if g["block"] == "P"},
        },
        "item_summaries": {"overall": overall, "by_stratum": by_stratum, "by_family": by_family,
                           "histogram_tests_lb": {str(k): v for k, v in histogram.items()}},
        "episodes": {"by_condition": by_condition, "p_contrasts": p_contrasts, "p_by_stratum": p_by_stratum,
                     "p_by_family": p_by_family},
        "families": family_info,
        "caveats": [
            "Per-site test counts are not identified: each trace record keeps only the first test that executed "
            "it, so tests_lb and broken_lb are lower bounds (mostly 1 per witnessed line). Shares with >=5 tests "
            "cannot be estimated from these records.",
            "First-execution labels depend on test order (-p no:randomly, file order). A site first reached "
            "while test modules were imported carries <collection> even if many tests call it afterwards; the "
            f"{sum(1 for r in rows if r['phase_only'])} phase-only items get 0 in the node-id bound although "
            "their C* site ran during the session.",
            "'Broken' is a model: a test that reaches an unchanged call fails only if the signature change is "
            "incompatible with that call and the error is not caught (broad except clauses in the library, "
            "pytest.raises, non-strict xfail). Import-time sites would instead make the importing test module "
            "fail to collect (and a <startup> site would stop the session), which this count does not size.",
            "Test outcomes under tracing differ from the untraced baseline for a few families (D12.5: websockets "
            "38 tests fail only under tracing; rdflib, param and petl collect fewer skips); runtime witnesses "
            "are a subset of what the full suite would show. "
            + audit_sentence(args, family_info),
            "Only runtime-witnessed sites count; a C* site witnessed only by the AST, L or J is never counted "
            "as test-exercised, and V callers that no test reaches never contribute.",
            "The secondary all-V count assumes the refactor relies on the submitted list alone; in C1-C4a the "
            "answer block also names the retained callers, which an engineer could update without the agent.",
            "Descriptive only: written after the main results were open; the bootstrap intervals are nominal "
            "library-cluster percentiles and support no confirmatory claim.",
        ],
        "follow_up": [
            "A per-test re-trace would identify the counts: run each main family's suite once more with a "
            "plugin variant that stores, for the 750 (C*, X) pairs and the other V pairs of the items, the set "
            "of node ids per (caller, callee, line) instead of the first one (the same sys.monitoring PY_START "
            "callback; one set insertion per event on those pairs only).",
            "Cost: the 30 main trace-1 runs took "
            f"{sum(i['wall_seconds'] or 0 for i in family_info.values()):.0f} s of wall time in total "
            f"(mpmath {family_info.get('mpmath', {}).get('wall_seconds')} s, rdflib "
            f"{family_info.get('rdflib', {}).get('wall_seconds')} s); CPU only, so it fits Core2's 8-process "
            "limit. It needs the frozen repos and envs read-only and a scratch copy of each checkout as the "
            "working directory, because tests write into their cwd.",
            "Report it as a post-hoc truth extension (new data, not the frozen trace), with the per-site counts "
            "joined to these episodes by the same coverage rule.",
        ],
        "items": [{key: value for key, value in row.items() if not key.startswith("_")} for row in rows],
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "TESTS_PER_CSTAR.json").write_text(json.dumps(result, indent=1, sort_keys=False,
                                                             ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "TESTS_PER_CSTAR.md").write_text(render_md(result), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "checks": result["checks"], "key_numbers": {
        key: value for key, value in result["key_numbers"].items() if key != "tests_lb"}}, indent=1))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except InputError as error:
        print(f"expA_tests_per_cstar: {error}", file=sys.stderr)
        raise SystemExit(2)
