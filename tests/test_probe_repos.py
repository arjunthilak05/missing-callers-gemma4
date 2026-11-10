"""Unit tests for absent_edges.probe.repos (Exp A lane R): licences, exclusions, pins, registry."""
import copy
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

from absent_edges.probe import repos as R

ROOT = Path(__file__).resolve().parents[1]

MIT = """MIT License

Copyright (c) 2020 Example Author

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
"""

_BSD_HEAD = """Copyright (c) 2010, Example Project
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
"""
_BSD_TAIL = """
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES ARE DISCLAIMED.
"""
BSD2 = _BSD_HEAD + _BSD_TAIL
BSD3 = _BSD_HEAD + """
* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.
""" + _BSD_TAIL
BSD3_AUTHOR_VARIANT = _BSD_HEAD + """
3. The name of the author may not be used to endorse or promote products
   derived from this software without specific prior written permission.
""" + _BSD_TAIL
BSD4 = _BSD_HEAD + """
3. All advertising materials mentioning features or use of this software
   must display the following acknowledgement: This product includes software
   developed by the Example Project.
4. Neither the name of the Example Project nor the names of its contributors
   may be used to endorse or promote products derived from this software.
""" + _BSD_TAIL

APACHE = """
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.
"""
APACHE_NOTICE = """Copyright 2017 Example

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
"""
ISC = """ISC License

Copyright (c) 2015, Example Author

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted, provided that the above
copyright notice and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES.
"""
ISC_ORG_VARIANT = ISC.replace("AND THE AUTHOR DISCLAIMS", "AND NOMINUM DISCLAIMS").replace("and/or ", "and ")
PSF = """PYTHON SOFTWARE FOUNDATION LICENSE VERSION 2
--------------------------------------------

1. This LICENSE AGREEMENT is between the Python Software Foundation
("PSF"), and the Individual or Organization ("Licensee") accessing and
otherwise using this software ("Python") in source or binary form and
its associated documentation.
"""
GPL3 = """                    GNU GENERAL PUBLIC LICENSE
                       Version 3, 29 June 2007

 Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
"""
LGPL3 = """                   GNU LESSER GENERAL PUBLIC LICENSE
                       Version 3, 29 June 2007

  This version of the GNU Lesser General Public License incorporates
the terms and conditions of version 3 of the GNU General Public
License, supplemented by the additional permissions listed below.
"""
MPL2 = """Mozilla Public License Version 2.0
==================================

1. Definitions
"""
DUAL_POINTER = """This software is made available under the terms of *either* of the licenses
found in LICENSE.APACHE or LICENSE.BSD. Contributions to this software is made
under the terms of *both* these licenses.
"""


def write_tree(root: Path, files: dict) -> None:
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")


class LicenseDetectionTests(unittest.TestCase):
    def spdx(self, text):
        return R.detect_license_text(text)["spdx"]

    def test_permissive_fixtures(self):
        self.assertEqual(self.spdx(MIT), ["MIT"])
        self.assertEqual(self.spdx(BSD2), ["BSD-2-Clause"])
        self.assertEqual(self.spdx(BSD3), ["BSD-3-Clause"])
        self.assertEqual(self.spdx(BSD3_AUTHOR_VARIANT), ["BSD-3-Clause"])
        self.assertEqual(self.spdx(APACHE), ["Apache-2.0"])
        self.assertEqual(self.spdx(APACHE_NOTICE), ["Apache-2.0"])
        self.assertEqual(self.spdx(ISC), ["ISC"])
        self.assertEqual(self.spdx(ISC_ORG_VARIANT), ["ISC"])
        self.assertEqual(self.spdx(PSF), ["PSF-2.0"])

    def test_bsd3_plural_names_variant(self):
        text = BSD3.replace("Neither the name of", "Neither the names of")
        self.assertEqual(self.spdx(text), ["BSD-3-Clause"])

    def test_commented_and_rewrapped_text(self):
        commented = "\n".join("# " + line for line in MIT.replace("\n", " ").split(". "))
        self.assertEqual(self.spdx(commented), ["MIT"])
        self.assertEqual(self.spdx(MIT.replace("\n", "\r\n")), ["MIT"])

    def test_bsd4_detected_but_not_allowed(self):
        self.assertEqual(self.spdx(BSD4), ["BSD-4-Clause"])
        with tempfile.TemporaryDirectory() as d:
            write_tree(Path(d), {"LICENSE": BSD4})
            res = R.detect_repo_license(d)
        self.assertFalse(res["acceptable"])
        self.assertIn("allow-list", res["reason"])

    def test_non_permissive_texts(self):
        self.assertIn("GPL", R.detect_license_text(GPL3)["non_permissive"])
        self.assertIn("LGPL", R.detect_license_text(LGPL3)["non_permissive"])
        self.assertIn("MPL", R.detect_license_text(MPL2)["non_permissive"])
        for text in (GPL3, LGPL3, MPL2):
            self.assertEqual(R.detect_license_text(text)["spdx"], [])

    def test_unknown_text(self):
        det = R.detect_license_text("All rights reserved. Do not copy.")
        self.assertTrue(det["unknown"])
        self.assertEqual(det["spdx"], [])

    def test_repo_dual_licence_uses_or(self):
        with tempfile.TemporaryDirectory() as d:
            write_tree(Path(d), {"LICENSE": DUAL_POINTER, "LICENSE.APACHE": APACHE, "LICENSE.BSD": BSD2})
            res = R.detect_repo_license(d)
        self.assertTrue(res["acceptable"])
        self.assertEqual(res["license"], "Apache-2.0 OR BSD-2-Clause")
        self.assertEqual([f["path"] for f in res["files"]], ["LICENSE", "LICENSE.APACHE", "LICENSE.BSD"])

    def test_repo_combined_licences_use_and(self):
        with tempfile.TemporaryDirectory() as d:
            write_tree(Path(d), {"LICENSE": APACHE_NOTICE + "\n\n" + BSD3})
            res = R.detect_repo_license(d)
        self.assertEqual(res["license"], "Apache-2.0 AND BSD-3-Clause")

    def test_repo_with_any_copyleft_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            write_tree(Path(d), {"LICENSE": MIT, "COPYING.LESSER": LGPL3})
            res = R.detect_repo_license(d)
        self.assertFalse(res["acceptable"])
        self.assertIn("LGPL", res["reason"])

    def test_repo_without_licence_file(self):
        with tempfile.TemporaryDirectory() as d:
            write_tree(Path(d), {"README.md": "MIT licensed"})
            res = R.detect_repo_license(d)
        self.assertFalse(res["acceptable"])
        self.assertEqual(res["reason"], "no licence file")

    def test_licence_file_discovery(self):
        with tempfile.TemporaryDirectory() as d:
            write_tree(Path(d), {"LICENSE.txt": MIT, "COPYING": MIT, "LICENSES/MIT.txt": MIT,
                                 "license_notes.py": "x = 1", "README": ""})
            found = R.find_license_files(d)
        self.assertEqual(sorted(found), ["COPYING", "LICENSE.txt", "LICENSES/MIT.txt"])

    def test_license_expression_ok(self):
        self.assertTrue(R.license_expression_ok("MIT"))
        self.assertTrue(R.license_expression_ok("Apache-2.0 OR BSD-2-Clause"))
        self.assertFalse(R.license_expression_ok("MIT AND GPL-3.0"))
        self.assertFalse(R.license_expression_ok(None))


class ExclusionTests(unittest.TestCase):
    def make_project(self, root: Path):
        write_tree(root, {
            "manifests/D-validation/roster.json": json.dumps({
                "tasks": [{"instance_id": "validate-v2-pvlib__pvlib-python-2249", "repo": "pvlib/pvlib-python"}]}),
            "manifests/T-roster/candidate-roster.json": json.dumps({
                "family_slots": [{"network_root_id": 16899445, "network_root_repos": ["marshmallow-code/webargs"],
                                  "ordered_candidates": []}],
                "excluded_network_roots": [111, 222],
                "exclusion_evidence": [{"network_root_id": 111, "reasons": ["original_benchmark_network_reserved"]}]}),
            "reports/A.safe.json": json.dumps({"instance_id": "amoffat__sh-744",
                                               "url": "https://github.com/gersteinlab/LocAgent.git"}),
            "reports/B.json": json.dumps({"instance_id": "notscanned__repo-1"}),
            "staged/research-catalogue-slot07-v1/inputs/catalogue.json": json.dumps({
                "family_slots": [{"slot": 7, "network_root_id": 333, "network_root_repos": ["gawel/pyquery"],
                                  "ordered_candidates": [{"metadata": {
                                      "repo": "swesmith/gawel__pyquery.811cd048",
                                      "instance_id": "gawel__pyquery.811cd048.func_basic__abc12345"}}]}]}),
            "staged/research-t-catalogue-method-map-v1/public-source/upstream_tree.json":
                '{"x": "pandas-dev__pandas.95280573"}',
            "staged/unrelated-v1/notes.json": json.dumps({"repo": "someone/elsewhere"}),
            "manifests/expA/registry.v1.json": json.dumps({"repo": "python-attrs/attrs"}),
            "tasks/BACKLOG.json": json.dumps({"image": "swebench/swesmith.x86_64.hips_1776_autograd.ac044f0d"}),
        })

    def test_scan_text_kinds(self):
        text = ('"instance_id": "validate-v2-pvlib__pvlib-python-2249" "upstream_repo": "HIPS/autograd" '
                '"repo": "swesmith/HIPS__autograd.ac044f0d" cookiecutter__cookiecutter.b4451231.func_pm__x1y2z3w4 '
                'swebench/swesmith.x86_64.cookiecutter_1776_cookiecutter.b4451231 '
                'sweb.eval.x86_64.amoffat_1776_sh-744 "network_root_repos": ["marshmallow-code/webargs"] '
                'see https://github.com/gersteinlab/LocAgent.git')
        refs = set(R.scan_text(text))
        self.assertIn(("instance_id", "validate-v2-pvlib", "pvlib-python"), refs)
        self.assertIn(("repo_field", "HIPS", "autograd"), refs)
        self.assertIn(("swesmith_instance", "cookiecutter", "cookiecutter"), refs)
        self.assertIn(("docker_image", "cookiecutter", "cookiecutter"), refs)
        self.assertIn(("docker_image", "amoffat", "sh"), refs)
        self.assertIn(("network_root_repos", "marshmallow-code", "webargs"), refs)
        self.assertIn(("github_url", "gersteinlab", "LocAgent"), refs)

    def test_source_patterns(self):
        rx = R.glob_regex("staged/*slot*/**")
        self.assertTrue(rx.match("staged/core2-slot2-static-v1/a/b.json"))
        self.assertFalse(rx.match("staged/core2-other/a.json"))
        rx = R.glob_regex("reports/**/*.safe.json")
        self.assertTrue(rx.match("reports/A.safe.json"))
        self.assertTrue(rx.match("reports/checkpoints/x/A.safe.json"))
        self.assertFalse(rx.match("reports/A.json"))

    def test_scan_build_and_match(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.make_project(root)
            files = list(R.iter_source_files(root))
            scan = R.scan_project(root)
        self.assertNotIn("manifests/expA/registry.v1.json", files)
        self.assertNotIn("reports/B.json", files)
        self.assertNotIn("staged/unrelated-v1/notes.json", files)
        names = scan["names"]
        self.assertIn("pvlib/pvlib-python", names)          # prefix "validate-v2-" stripped
        self.assertNotIn("validate-v2-pvlib/pvlib-python", names)
        self.assertNotIn("notscanned/repo", names)
        self.assertNotIn("python-attrs/attrs", names)
        self.assertEqual(scan["roots"]["111"]["reasons"], ["excluded_network_root", "original_benchmark_network_reserved"])
        self.assertIn("selected_family_slot", scan["roots"]["333"]["reasons"])
        classes = R.classify_scan_names(scan, {"gersteinlab/locagent": "tool"},
                                        ("staged/*/public-source/**",))
        self.assertEqual(classes["pandas-dev/pandas"]["class"], "listing_only")
        self.assertEqual(classes["gersteinlab/locagent"]["class"], "non_family")
        self.assertEqual(classes["pvlib/pvlib-python"]["class"], "family")
        self.assertEqual(classes["hips/autograd"]["class"], "family")
        resolved = {"name:pvlib/pvlib-python": {"full_name": "pvlib/pvlib-python", "network_root_id": 30895522,
                                                "network_root_full_name": "pvlib/pvlib-python"},
                    "root:111": {"full_name": "pallets/flask", "network_root_full_name": "pallets/flask"},
                    "root:333": {"full_name": "gawel/pyquery", "network_root_full_name": "gawel/pyquery"},
                    "name:gawel/pyquery": {"full_name": "gawel/pyquery", "network_root_id": 333,
                                           "network_root_full_name": "gawel/pyquery"}}
        cat = {"gawel/pyquery": ["research_catalogue_slot:07"]}
        excl = R.build_exclusions(scan, resolved, cat, classes)
        self.assertTrue(excl.is_excluded("PVLIB/pvlib-python"))
        self.assertTrue(excl.is_excluded("pvlib__pvlib-python"))
        self.assertTrue(excl.is_excluded(network_root_id=30895522))
        self.assertTrue(excl.is_excluded("amoffat/sh"))
        self.assertTrue(excl.is_excluded("marshmallow-code/webargs"))
        self.assertTrue(excl.is_excluded(network_root_id=16899445))
        self.assertTrue(excl.is_excluded("pallets/flask"))           # resolved reserved root name
        self.assertTrue(excl.is_excluded(network_root_id=222))       # unresolved reserved root id
        self.assertTrue(excl.is_excluded("https://github.com/gawel/pyquery.git"))
        self.assertFalse(excl.is_excluded("pandas-dev/pandas"))       # listing only
        self.assertFalse(excl.is_excluded("gersteinlab/LocAgent"))    # tool
        self.assertFalse(excl.is_excluded("python-attrs/attrs", 29918975))
        pyquery = [e for e in excl.entries if e.network_root_id == 333]
        self.assertEqual(len(pyquery), 1)                             # name and root merged
        self.assertIn("research_catalogue_family", pyquery[0].categories)
        self.assertIn("selected_family_slot", pyquery[0].categories)
        self.assertEqual(R.ExclusionList.from_json(excl.to_json()).to_json(), excl.to_json())

    def test_catalogue_families_from(self):
        doc = {"family_slots": [
            {"slot": 10, "network_root_id": 1, "network_root_repos": ["alecthomas/voluptuous"],
             "ordered_candidates": [{"metadata": {"repo": "swesmith/alecthomas__voluptuous.a7a55f83",
                                                  "upstream_repo": "alecthomas/voluptuous"}}]},
            {"slot": 21, "network_root_id": 2, "network_root_repos": ["timbertson/python-readability"],
             "ordered_candidates": [{"metadata": {"repo": "swesmith/buriy__python-readability.40256f40"}}]}]}
        fams = R.catalogue_families_from(doc)
        self.assertIn("Task10", fams["alecthomas/voluptuous"])
        self.assertIn("research_catalogue_slot:21", fams["buriy/python-readability"])
        self.assertIn("research_catalogue_slot:21", fams["timbertson/python-readability"])


class PinTests(unittest.TestCase):
    def test_release_version(self):
        self.assertEqual(R.release_version("v1.2.3"), (1, 2, 3, 0, 0))
        self.assertEqual(R.release_version("26.2.0"), (26, 2, 0, 0, 0))
        self.assertEqual(R.release_version("rel_1_13_2"), (1, 13, 2, 0, 0))
        self.assertEqual(R.release_version("RELEASE_3.2.9"), (3, 2, 9, 0, 0))
        self.assertEqual(R.release_version("release_v3.0"), (3, 0, 0, 0, 0))
        self.assertEqual(R.release_version("2.9.0.post0"), (2, 9, 0, 0, 0))
        self.assertEqual(R.release_version("1.0.post2"), (1, 0, 0, 0, 2))
        self.assertEqual(R.release_version("refs/tags/v2^{}"), (2, 0, 0, 0, 0))
        for bad in ("v2.0.0rc1", "1.0b2", "3.0.0.dev0", "v1.2.3-alpha", "nightly", "pkg-1.0", "20240101", "docs-2020"):
            self.assertIsNone(R.release_version(bad), bad)
        self.assertLess(R.release_version("rel_1_0_11"), R.release_version("rel_1_13_2"))

    def tag(self, name, sha_char, date, tagger=None):
        return {"tag": name, "commit": sha_char * 40, "commit_date": date, "tagger_date": tagger}

    def test_choose_release_pin(self):
        tags = [self.tag("v1.0.0", "a", "2025-01-01T00:00:00Z"),
                self.tag("v2.0.0", "b", "2026-03-01T00:00:00Z"),
                self.tag("v1.9.9", "c", "2026-08-01T00:00:00Z"),           # later-dated maintenance release
                self.tag("v2.1.0", "d", "2026-10-02T00:00:00Z"),           # after the cutoff
                self.tag("v2.0.1", "e", "2026-09-01T00:00:00Z", "2026-10-05T00:00:00Z"),  # tagged after cutoff
                self.tag("v3.0.0rc1", "f", "2026-09-15T00:00:00Z"),
                self.tag("v2.0.5", "g", "")]                               # undated: ignored
        chosen = R.choose_release_pin(tags)
        self.assertEqual(chosen["tag"], "v2.0.0")
        self.assertEqual(chosen["commit"], "b" * 40)
        self.assertIsNone(R.choose_release_pin([self.tag("nightly", "a", "2025-01-01T00:00:00Z")]))
        self.assertIsNone(R.choose_release_pin([self.tag("v1.0", "a", "2026-10-01T00:00:00Z")]))
        self.assertEqual(R.choose_release_pin([self.tag("v1.0", "a", "2026-09-30T23:59:59+00:00")])["tag"], "v1.0")


JUNIT = b"""<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest" errors="1" failures="1" skipped="1" tests="5">
<testcase classname="t" name="a"/><testcase classname="t" name="b"/>
<testcase classname="t" name="c"><failure message="x"/></testcase>
<testcase classname="t" name="d"><skipped message="s"/></testcase>
<testcase classname="" name="tests.test_x"><error message="collection failure"/></testcase>
</testsuite></testsuites>"""


class BaselineTests(unittest.TestCase):
    def test_parse_junit(self):
        self.assertEqual(R.parse_junit(JUNIT),
                         {"collected": 5, "passed": 2, "failed": 1, "errors": 1, "skipped": 1})
        single = b'<testsuite tests="1"><testcase classname="t" name="a"/></testsuite>'
        self.assertEqual(R.parse_junit(single)["passed"], 1)

    def test_baseline_status(self):
        ok = {"collected": 100, "passed": 96, "failed": 4, "errors": 0}
        self.assertEqual(R.baseline_status(ok)[0], "ok")
        self.assertEqual(R.baseline_status({**ok, "failed": 5, "errors": 1})[0], "tests_failed")
        self.assertEqual(R.baseline_status({"collected": 10, "passed": 10, "failed": 0, "errors": 0})[0],
                         "tests_failed")
        self.assertEqual(R.baseline_status(ok, timed_out=True)[0], "tests_failed")
        self.assertEqual(R.baseline_status(None)[0], "tests_failed")


def record(**over):
    base = dict(family="fam", url="https://github.com/owner/fam", commit="a" * 40,
                commit_date="2026-09-01T00:00:00Z", license="MIT", package_dirs=["src/fam/"],
                test_command=["python", "-m", "pytest", "tests"], python="3.13.14", env_lock_sha256="b" * 64,
                baseline={"counts": {"collected": 100, "passed": 100, "failed": 0, "errors": 0, "skipped": 0},
                          "timed_out": False},
                status="ok", role="main", tag="v1.0", pin_rule="release_tag_before_cutoff", network_root_id=1)
    base.update(over)
    return R.FamilyRecord(**base)


def registry_doc(n_main=22, extra=()):
    recs = [record(family=f"m{i:02d}", url=f"https://github.com/o{i}/m{i:02d}", network_root_id=100 + i)
            for i in range(n_main)]
    recs += [record(family="more-itertools", url="https://github.com/more-itertools/more-itertools", role="dev",
                    pin_rule="explicit_dev_tag", commit_date="2026-10-14T00:00:00Z", network_root_id=7),
             record(family="boltons", url="https://github.com/mahmoud/boltons", role="dev", network_root_id=8)]
    recs += list(extra)
    return {"schema": R.SCHEMA, "cutoff": R.CUTOFF, "families": [r.to_json() for r in recs],
            "exclusions": [R.ExclusionEntry(family="pvlib/pvlib-python", network_root_id=30895522,
                                            aliases=("pvlib/pvlib-python",), categories=("project_task_family",),
                                            reasons=(), source_paths=(), source_count=1).to_json()]}


class RegistryTests(unittest.TestCase):
    def test_valid_registry(self):
        self.assertEqual(R.validate_registry(registry_doc()), [])

    def errors_for(self, **over):
        return R.validate_record(record(**over))

    def test_record_field_errors(self):
        self.assertEqual(self.errors_for(), [])
        cases = {
            "commit": dict(commit="abc"),
            "commit_date": dict(commit_date="2026-10-01T00:00:00Z"),
            "license": dict(license="GPL-3.0"),
            "split": dict(split="A"),
            "status": dict(status="great"),
            "role": dict(role="captain"),
            "url": dict(url="https://gitlab.com/o/r"),
            "family": dict(family="Bad Name"),
            "pin_rule": dict(pin_rule="whatever"),
            "explicit_dev_tag": dict(pin_rule="explicit_dev_tag"),
            "package dir": dict(package_dirs=["src/fam/tests/"]),
            "not normalised": dict(package_dirs=["src/fam"]),
            "test_command": dict(test_command=[]),
            "python": dict(python="3.11.9"),
            "env_lock_sha256": dict(env_lock_sha256=None),
            "baseline rule": dict(baseline={"counts": {"collected": 100, "passed": 50, "failed": 50, "errors": 0}}),
            "exceeds collected": dict(baseline={"counts": {"collected": 10, "passed": 30, "failed": 0, "errors": 0}}),
        }
        for label, over in cases.items():
            self.assertTrue(self.errors_for(**over), label)

    def test_excluded_records(self):
        rec = record(status="excluded", role=None, commit=None, commit_date=None, python=None,
                     env_lock_sha256=None, baseline=None,
                     exclusion_matches=[{"family": "x/y", "network_root_id": 1, "categories": ["c"]}])
        self.assertEqual(R.validate_record(rec), [])
        rec.role = "spare"
        self.assertTrue(R.validate_record(rec))
        self.assertTrue(R.validate_record(record(status="excluded", role=None)))   # no matches recorded

    def test_registry_level_errors(self):
        self.assertTrue(any("main families" in e for e in R.validate_registry(registry_doc(n_main=21))))
        doc = registry_doc()
        doc["families"].append(copy.deepcopy(doc["families"][0]))
        self.assertTrue(any("duplicate" in e for e in R.validate_registry(doc)))
        doc = registry_doc()
        doc["families"][-1]["role"] = "main"                       # boltons must stay dev
        self.assertTrue(any("dev family" in e for e in R.validate_registry(doc)))
        doc = registry_doc(extra=[record(family="pvlib", url="https://github.com/pvlib/pvlib-python",
                                         network_root_id=30895522, role="spare")])
        self.assertTrue(any("exclusion list" in e for e in R.validate_registry(doc)))
        doc = registry_doc(extra=[record(family="broken", url="https://github.com/o/broken", status="env_failed",
                                         role="main", network_root_id=9, env_lock_sha256=None, baseline=None)])
        self.assertTrue(any("env_failed" in e for e in R.validate_registry(doc)))
        doc = registry_doc()
        doc["families"][0]["surprise"] = 1
        self.assertTrue(any("unknown record fields" in e for e in R.validate_registry(doc)))
        self.assertTrue(R.validate_registry({**registry_doc(), "schema": "other"}))

    def test_load_registry(self):
        with tempfile.TemporaryDirectory() as d:
            good = Path(d) / "good.json"
            good.write_text(R.canonical_json(registry_doc()), encoding="utf-8")
            self.assertEqual(R.load_registry(good)["schema"], R.SCHEMA)
            bad = Path(d) / "bad.json"
            bad.write_text(R.canonical_json(registry_doc(n_main=3)), encoding="utf-8")
            with self.assertRaises(R.RegistryError):
                R.load_registry(bad)
            self.assertEqual(len(R.families(R.load_registry(good), roles=["dev"])), 2)

    def test_assign_roles_deterministic(self):
        def rec(name, status, passed, failed=0):
            return record(family=name, url=f"https://github.com/o/{name}", status=status, role=None,
                          baseline={"counts": {"collected": passed + failed, "passed": passed, "failed": failed,
                                               "errors": 0}})
        recs = [rec("b", "ok", 500), rec("a", "ok", 500), rec("c", "ok", 900, 20), rec("f", "ok", 2000, 5),
                rec("d", "tests_failed", 50, 50), rec("e", "env_failed", 0), rec("boltons", "ok", 10),
                rec("x", "excluded", 0)]
        shuffled = list(reversed([copy.deepcopy(r) for r in recs]))
        out1 = [(r.family, r.role) for r in R.assign_roles(recs, n_main=3)]
        out2 = [(r.family, r.role) for r in R.assign_roles(shuffled, n_main=3)]
        self.assertEqual(out1, out2)
        roles = dict(out1)
        self.assertEqual(roles["boltons"], "dev")
        self.assertIsNone(roles["x"])
        # clean (>= 0.99) baselines first, then more passing tests, then name: c (0.978) drops to spare
        self.assertEqual([f for f, r in out1 if r == "main"], ["f", "a", "b"])
        self.assertEqual(roles["c"], "spare")
        self.assertEqual(roles["d"], "spare")
        self.assertEqual(roles["e"], "spare")                                   # env_failed never main
        many = R.assign_roles([rec("e", "env_failed", 0), rec("f", "ok", 30)], n_main=5)
        self.assertEqual({r.family: r.role for r in many}, {"e": "spare", "f": "main"})


class OfflineAndScopeTests(unittest.TestCase):
    def test_offline_argv(self):
        argv = R.offline_argv(["python", "-m", "pytest"])
        self.assertEqual(argv[:5], ["unshare", "--user", "--map-root-user", "--net", "--"])
        self.assertIn("link set lo up", argv[7])
        self.assertEqual(argv[-3:], ["python", "-m", "pytest"])

    def test_scrubbed_env(self):
        os.environ["GITHUB_TOKEN_FOR_TEST"] = "secret"
        try:
            with tempfile.TemporaryDirectory() as d:
                env = R.scrubbed_test_env("/envs/fam", d, {"MPLBACKEND": "Agg"})
                self.assertTrue(env["HOME"].startswith(d))
                self.assertEqual(env["PYTHONDONTWRITEBYTECODE"], "1")
                self.assertEqual(env["MPLBACKEND"], "Agg")
                self.assertTrue(env["PATH"].startswith(str(Path("/envs/fam") / "bin")))
                self.assertNotIn("GITHUB_TOKEN_FOR_TEST", env)
                self.assertFalse(any("TOKEN" in k for k in env))
                with self.assertRaises(R.RegistryError):
                    R.scrubbed_test_env("/envs/fam", d, {"HF_TOKEN": "x"})
        finally:
            del os.environ["GITHUB_TOKEN_FOR_TEST"]

    def test_normalise_package_dir(self):
        self.assertEqual(R.normalise_package_dir("src/attr"), "src/attr/")
        self.assertEqual(R.normalise_package_dir("src\\attr\\"), "src/attr/")
        self.assertEqual(R.normalise_package_dir("peewee.py"), "peewee.py")
        for bad in ("../x", "/abs/x", "C:\\x", "a/./b", ""):
            with self.assertRaises(R.RegistryError, msg=bad):
                R.normalise_package_dir(bad)

    def test_package_scope_and_stats(self):
        with tempfile.TemporaryDirectory() as d:
            write_tree(Path(d), {
                "src/pkg/__init__.py": "from .mod import f\n",
                "src/pkg/mod.py": "# comment\n\ndef f():\n    return g()\n\n\ndef g():\n    return 1\n",
                "src/pkg/tests/test_mod.py": "def test_f(): pass\n",
                "src/pkg/sub/test_helpers.py": "x = 1\n",
                "src/pkg/sub/thing_test.py": "x = 1\n",
                "src/pkg/conftest.py": "x = 1\n",
                "src/pkg/docs/example.py": "x = 1\n",
                "single.py": "def h():\n    pass\n",
                "setup.py": "x = 1\n",
            })
            files = R.package_python_files(d, ["src/pkg/", "single.py"])
            stats = R.package_stats(d, ["src/pkg/"])
        self.assertEqual(files, ["single.py", "src/pkg/__init__.py", "src/pkg/mod.py"])
        self.assertEqual(stats, {"package_files": 2, "package_loc": 5, "package_defs": 2})

    def test_fallback_rule_matches_identity(self):
        samples = ["src/pkg/a.py", "src/pkg/tests/a.py", "pkg/test/a.py", "pkg/testing/x.py", "docs/conf.py",
                   "pkg/test_a.py", "pkg/a_test.py", "conftest.py", "setup.py", "noxfile.py", "scripts/x.py",
                   "examples/e.py", "benchmarks/b.py", "pkg/testsuite.py", "pkg/latest.py"]
        try:
            from absent_edges.probe import identity
        except ImportError:  # identity.py (static lane) not present: fallback alone must hold
            identity = None
        for s in samples:
            fb = R.fallback_is_test_or_excluded_path(s)
            if identity is not None:
                self.assertEqual(fb, identity.is_test_or_excluded_path(s), s)
        self.assertFalse(R.fallback_is_test_or_excluded_path("pkg/latest.py"))
        self.assertTrue(R.fallback_is_test_or_excluded_path("pkg/testing/x.py"))


class CandidateTableTests(unittest.TestCase):
    """Consistency of the candidate table in scripts/expA_repos.py."""

    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("expA_repos_under_test", ROOT / "scripts" / "expA_repos.py")
        cls.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.mod)

    def test_table(self):
        cands = self.mod.CANDIDATES
        names = [c["family"] for c in cands]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(len({c["repo"].casefold() for c in cands}), len(cands))
        for dev in R.DEV_FAMILIES:
            c = next(c for c in cands if c["family"] == dev)
            self.assertTrue(c["pin_tag"])
        active = [c for c in cands if not c.get("excluded_candidate")]
        self.assertGreaterEqual(len(active) - len(R.DEV_FAMILIES), R.MIN_MAIN)
        orgs = {}
        for c in active:
            self.assertRegex(c["family"], r"^[a-z0-9][a-z0-9._-]*$")
            self.assertNotEqual(c["family"], "static")           # envs/static belongs to the static lane
            self.assertIsInstance(c["network_root_id"], int)
            self.assertTrue(c["test_args"])
            for pd in c["package_dirs"]:
                self.assertEqual(R.normalise_package_dir(pd), pd)
            orgs.setdefault(c["repo"].split("/")[0].casefold(), []).append(c["family"])
        self.assertLessEqual(max(len(v) for v in orgs.values()), 2)
        rejected = {r["repo"].casefold() for r in self.mod.REJECTED}
        self.assertFalse(rejected & {c["repo"].casefold() for c in cands})
        self.assertTrue(all(r["reason"] for r in self.mod.REJECTED))

    def test_test_command_shape(self):
        c = next(c for c in self.mod.CANDIDATES if c["family"] == "boltons")
        cmd = self.mod.test_command(c)
        self.assertEqual(cmd[:3], ["python", "-m", "pytest"])
        self.assertIn("no:cacheprovider", cmd)

    def test_registry_manifest_if_present(self):
        path = ROOT / "manifests" / "expA" / "registry.v1.json"
        if not path.exists():
            self.skipTest("registry not built yet")
        doc = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(R.validate_registry(doc), [])
        self.assertEqual(doc["validation"], {"errors": [], "valid": True})


if __name__ == "__main__":
    unittest.main()
