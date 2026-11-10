#!/usr/bin/env python3
"""Build the public Absent Edges Exp A release package (v1). Standard library only.

The package is built in two steps. Nothing is uploaded or published by this script.

1. On the research workstation (inside the git working tree):

       python -B scripts/expA_build_release_package.py export --out EXPORT.tar

   writes a tar of every repository file the package uses (read from the working tree), plus
   REPO_EXPORT.json: the HEAD commit, each file's git blob id and whether it equals HEAD, the full commit
   ids of the code releases, and the git blob ids of tracked report/manifest files (used as witnesses).

2. On Core2 (CPU only) with the controller Python:

       python -B expA_build_release_package.py build --export EXPORT.tar --private PRIVATE.json \
           [--copyright-holder NAME] [--code-url URL] [--writeup-url URL] [--contact TEXT] [--funding TEXT]

   assembles OUT_ROOT/absent-edges-expA-v1/ from the frozen Core2 records and the export, then
   python -I -B OUT_ROOT/absent-edges-expA-v1/reproduce/verify.py checks it. Fields that are not given are
   written as visible placeholders "⟦TO-FILL-AT-PUBLISH: ...⟧" and reported under unfilled_placeholders.

PRIVATE.json holds the account names to redact or to list (it is never copied into the package, and no
personal name appears in this file):
    {"other_user": [...], "own_account": [...], "own_identity": [...], "own_windows_account": [...],
     "name_fragments": [...]}
Tokens are matched case-insensitively. other_user tokens are replaced by "[redacted-other-user]" in plain files
(each change is recorded in SANITIZATION.md with the old and new sha256); a hit inside an episode or truth
shard stops the build. Every occurrence of the own_* classes is listed in SANITIZATION.md and kept.
Leak check: the name fragments (name_fragments, plus every alphabetic part of five or more letters of the
other_user, own_account and own_identity tokens) are searched case-insensitively in every published file and in
every shard member. The build fails on any fragment that does not lie inside an own_* occurrence listed in
SANITIZATION.md, and on any published file whose own_* occurrences SANITIZATION.md does not list exactly.

Principles (staged/paper-prep-20261101/release-plan/RELEASE_PACKAGE_PLAN.md): frozen files are copied byte for
byte so that their sha256 match the pins; only personal data is changed; numbers in README.md come from frozen
files through the CLAIMS table below, each with its file and JSON path.

Adding a run (for example the Gemma 4 31B QAT run) is a re-run of both steps, nothing else:
- runs are discovered in EXPA/runs: a run with run.json and a frozen analysis directory
  (<prefix>-analysis-<date><suffix>/analysis/ANALYSIS.json, or the directory named in RUN_INFO) is published with its
  analysis; a follow-up run marked "kind": "scores_only" in RUN_INFO (A3-A5: no expA_analyze.py analysis) is
  published with its frozen scores (<prefix>-analysis-<stamp>/scores.jsonl, its manifest and OUTPUTS.sha256) and
  plain copies of its launch records, or skipped as "not yet scored"; a launch without run.json is published as
  launch records only when RUN_INFO gives its note; every other run directory (in preparation, running, dev checks,
  or not yet analysed) is skipped and reported;
- follow-up analysis outputs (FOLLOWUPS, REPO_FILES, OPTIONAL_REPO_FILES) are published when present; verify.py re-runs
  each one whose runs are all in the build, with the command line derived from the output's own inputs record;
- code releases are discovered in EXPA/releases; each run's analysis code is the newest release whose
  RELEASE_FILES.sha256 matches every code hash in the run's run.json;
- protocols/EXPA_*.md and *.lock.json are exported by glob; a run's pre-specified comparison is
  reports/expA-<run>/COMPARE.json, whose reference run is found by the sha256 of its CELLS.jsonl;
- RUN_INFO only supplies the human-readable title and status of each run (a generic text is used otherwise).
Shards of unchanged runs are reused from OUT_ROOT/work/cache when the source file list (path, size, mtime) is
unchanged.
"""
from __future__ import annotations

import argparse
import calendar
import collections
import datetime
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath

LABEL = "Reviewed by the authors before release."
PACKAGE = "absent-edges-expA-v1"
SCHEMA = "absent-edges-expA-release-v1"
BUILD_FORMAT = 1  # bump to invalidate shard caches

# ------------------------------------------------------------------------------------------------ locations
EXPA = "/mnt/kldata/absent-edges-20261018-core2/expA"
PUBLIC = "/mnt/kldata/absent-edges-expA-public-20261101"
LOCAGENT_ROOT = "/mnt/kldata/absent-edges-20261018-core2/vendor/locagent"
JEDI_DIST = EXPA + "/envs/static/lib/python3.11/site-packages/jedi-0.20.1.dist-info"
OUT_ROOT = "/mnt/kldata/absent-edges-release-v1"
HARD = f"{EXPA}/hard-panel-20261102"  # A2 item build (protocol/EXPA_HARD.md and its lock)

APACHE_LICENSE_SHA256 = "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"  # apache.org text
LOCAGENT_COMMIT = "4935b557326c154bad8e8dcf3747cc8d32d1f387"
LOCAGENT_LICENSE_SHA256 = "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
JEDI_LICENSE_SHA256 = "78e60cd0b8f28694f30195482c33d76908d846b0d15278deb7332aa22ba8e412"

TAR_MTIME = calendar.timegm((2026, 11, 1, 0, 0, 0))
XZ_ARGS = ["xz", "-6", "-T8", "--block-size=25165824", "--check=crc64", "-c"]
# The same token as the public code repository's build, so that a file published in both has the same bytes.
REDACTION = b"[redacted-other-user]"
SCAN_FORMAT = 2  # bump when the shard scan changes; a cached shard is re-scanned when its pattern digest differs
PLACEHOLDER = "⟦TO-FILL-AT-PUBLISH: {}⟧"

# ------------------------------------------------------------------------------------------------ releases
# Roles of the known releases; any other release directory in EXPA/releases is published too.
RELEASE_ROLES = {
    "2adb0b458": "static truth: LocAgent (L) and jedi (J) backends",
    "dd842ab6b": "runtime traces (sys.monitoring tracer)",
    "0b8ee0b7c": "item build and dispatch views",
    "f8f2ebf2b": "release pinned by the frozen preregistration lock",
    "a53c49396": "31B main run and pilot code (deviation D-1: stager fix only)",
    "7ee6744ac": "12B second-model test, 12B G0 and 16-response run (model-profile and budget options; deviation D-2)",
    "9f4f893c3": "QAT checkpoint test: the gemma4-31b-qat model profile (release locked in protocol/EXPA_QAT31B.lock.json)",
    "4990b398a": "follow-ups A1 (final-turn policy) and A2 (hard-omission panel); locked in protocol/EXPA_FINALTURN.lock.json "
                 "and protocol/EXPA_HARD.lock.json",
    "99f8d0b12": "follow-up A3 (evidence-ledger option); locked in protocol/EXPA_A3_LEDGER.lock.json",
    "3364a1783": "follow-up A6 (doubled budget with the deadline line); locked in protocol/EXPA_A6_BUDGET16.lock.json",
    "116fa302d": "follow-ups A4 (final-line components) and A5 (submit-time review); locked in "
                 "protocol/EXPA_A4_DIRECTIVE.lock.json and protocol/EXPA_A5_REVIEW.lock.json",
    "70d801cf9": "follow-up A7 (impact-analysis task framing); locked in protocol/EXPA_A7_IMPACT.lock.json",
    "7a67a2ef4": "follow-up A8 (other model families): every A8 run. The runs started from releases f33acc1cc, 1c770c16d, "
                 "1c49f59aa and 7a67a2ef4, which differ only in model profiles; each run.json's code hashes match this "
                 "release. Locked in protocol/EXPA_A8_MODELS.lock.json",
}

# ------------------------------------------------------------------------------------------------ runs
# Human-readable title and status per run id; "analysis" only where the analysis directory does not follow the
# <prefix>-analysis-<date><suffix> pattern; "note" for launch records without run.json.
RUN_INFO = {
    "pilot-20261101b": {"analysis": "pilot-analysis-20261101a",
                        "title": "Pilot: Gemma 4 31B BF16, the 2 pilot families",
                        "status": "pilot (throughput and base rates for the power simulation); never pooled with "
                                  "main outcomes"},
    "main-20261101b": {"title": "Main run: Gemma 4 31B BF16, 30 families",
                       "status": "preregistered; H1-H3 confirmatory"},
    "repl12b-g0-20261101b": {"title": "12B G0 gate: Gemma 4 12B BF16, the 2 pilot families",
                             "status": "go/no-go gate of the 12B addendum (verdict in protocol/EXPA_REPL12B.lock.json)"},
    "repl12b-20261101a": {"title": "Second-model test: Gemma 4 12B BF16, 30 families",
                          "status": "addendum locked before any 12B main-family outcome; Holm within the 12B model"},
    "budget16-20261101b": {"title": "16-response run: Gemma 4 31B BF16, block P, one seed",
                           "status": "exploratory (protocol/EXPA_BUDGET16.md); never pooled with the main run"},
    "qat31b-g0-20261102a": {"title": "QAT G0 gate: Gemma 4 31B QAT W4A16, the 2 pilot families",
                            "status": "go/no-go gate of the QAT addendum (protocol/EXPA_QAT31B.md)"},
    "qat31b-20261102a": {"compare_labels": "checkpoint",  # expA_compare_runs.py --labels used for its COMPARE.json
                         "title": "Main-track checkpoint test: Gemma 4 31B QAT W4A16, 30 families",
                         "status": "additional checkpoint test (protocol/EXPA_QAT31B.md and its lock)"},
    "finalturn-dev-neutral-20261104b": {"title": "A1 dev check, neutral notice: Gemma 4 31B BF16, the 2 pilot families",
                                        "status": "go/no-go check of protocol/EXPA_FINALTURN.md (verdict in its lock)"},
    "finalturn-dev-deadline-20261104b": {"title": "A1 dev check, deadline notice: Gemma 4 31B BF16, the 2 pilot families",
                                         "status": "go/no-go check of protocol/EXPA_FINALTURN.md (verdict in its lock)"},
    "finalturn-neutral-20261102a": {"title": "A1 final-turn policy, neutral notice: Gemma 4 31B BF16, block P, seed 11",
                                    "status": "follow-up A1 (protocol/EXPA_FINALTURN.md and its lock); never pooled with "
                                              "the main run"},
    "finalturn-deadline-20261102a": {"title": "A1 final-turn policy, deadline notice: Gemma 4 31B BF16, block P, seed 11",
                                     "status": "follow-up A1 (protocol/EXPA_FINALTURN.md and its lock); never pooled with "
                                               "the main run"},
    "finalturn-none-20261102a": {"title": "A1 final-turn policy, no notice: Gemma 4 31B BF16, block P, seed 11",
                                 "status": "follow-up A1 (protocol/EXPA_FINALTURN.md and its lock); never pooled with the "
                                           "main run"},
    "hard-20261102a": {"title": "A2 hard-omission panel: Gemma 4 31B BF16, 236 items in 27 families",
                       "status": "follow-up A2 (protocol/EXPA_HARD.md and its lock); never pooled with the main run"},
    # Follow-up runs without an expA_analyze.py analysis ("kind": "scores_only"): published with their scores
    # (<prefix>-analysis-<stamp>/scores.jsonl); their pre-specified analysis is a follow-up output (FOLLOWUPS).
    # "code" is the release that must match the code hashes in run.json.
    "a3ledger-off-20261104a": {"kind": "scores_only", "code": "99f8d0b12",
                               "title": "A3 evidence ledger, ledger off: Gemma 4 31B BF16, block LEDGER, seeds 61, "
                                        "73 and 89",
                               "status": "follow-up A3 (protocol/EXPA_A3_LEDGER.md and its lock); never pooled with "
                                         "other runs"},
    "a3ledger-on-20261104a": {"kind": "scores_only", "code": "99f8d0b12",
                              "title": "A3 evidence ledger, ledger on: Gemma 4 31B BF16, block LEDGER, seeds 61, 73 "
                                       "and 89",
                              "status": "follow-up A3 (protocol/EXPA_A3_LEDGER.md and its lock); never pooled with "
                                        "other runs"},
    **{f"a4directive-{arm}-20261106a": {
        "kind": "scores_only", "code": "116fa302d",
        "title": f"A4 final-line components, {policy} notice: Gemma 4 31B BF16, block P, seed 23, C0 and C1",
        "status": "follow-up A4 (protocol/EXPA_A4_DIRECTIVE.md and its lock); never pooled with other runs"}
       for arm, policy in (("neutral", "neutral"), ("info", "deadline_info"), ("submit", "submit_only"),
                           ("deadline", "deadline"))},
    **{f"a5review-{arm}-20261106a": {
        "kind": "scores_only", "code": "116fa302d",
        "title": f"A5 submit-time review pilot, {arm} review: Gemma 4 31B BF16, block LEDGER, seed 61",
        "status": "follow-up A5 (protocol/EXPA_A5_REVIEW.md and its lock); a pilot; never pooled with other runs"}
       for arm in ("neutral", "ledger")},
    "a6budget16-20261106a": {
        "kind": "scores_only", "code": "3364a1783",
        "title": "A6 doubled budget (16 responses, 32,768 tokens) with the deadline line: Gemma 4 31B BF16, block "
                 "LEDGER, seeds 61, 73 and 89",
        "status": "follow-up A6 (protocol/EXPA_A6_BUDGET16.md and its lock); compared with a3ledger-off-20261104a; "
                  "never pooled with other runs"},
    "a7impact-20261107b": {
        "kind": "scores_only", "code": "70d801cf9",
        "title": "A7 impact-analysis task framing: Gemma 4 31B BF16, no list (P C0), incomplete list (P C1) and "
                 "complete list (S-VC C1), seed 11",
        "status": "follow-up A7 (protocol/EXPA_A7_IMPACT.md and its lock); compared with main-20261101b; never "
                  "pooled with other runs"},
    "a8r1-qwen-20261110a": {
        "kind": "scores_only", "code": "7a67a2ef4",
        "title": "A8 other model families, Qwen3-Coder-30B-A3B-Instruct (Alibaba): no final-turn line, no list (P C0), incomplete list (P C1) and complete list (S-VC C1), seed 11",
        "status": "follow-up A8 (protocol/EXPA_A8_MODELS.md and its lock); compared with main-20261101b; never pooled with other runs"},
    "a8r2-qwen-20261110a": {
        "kind": "scores_only", "code": "7a67a2ef4",
        "title": "A8 other model families, Qwen3-Coder-30B-A3B-Instruct (Alibaba): the A1 deadline line, no list (P C0) and incomplete list (P C1), seed 11",
        "status": "follow-up A8 (protocol/EXPA_A8_MODELS.md and its lock); compared with main-20261101b; never pooled with other runs"},
    "a8r1-glm-20261110a": {
        "kind": "scores_only", "code": "7a67a2ef4",
        "title": "A8 other model families, GLM-4.5-Air-FP8 (Zhipu AI): no final-turn line, no list (P C0), incomplete list (P C1) and complete list (S-VC C1), seed 11",
        "status": "follow-up A8 (protocol/EXPA_A8_MODELS.md and its lock); compared with main-20261101b; never pooled with other runs"},
    "a8r2-glm-20261110a": {
        "kind": "scores_only", "code": "7a67a2ef4",
        "title": "A8 other model families, GLM-4.5-Air-FP8 (Zhipu AI): the A1 deadline line, no list (P C0) and incomplete list (P C1), seed 11",
        "status": "follow-up A8 (protocol/EXPA_A8_MODELS.md and its lock); compared with main-20261101b; never pooled with other runs"},
    "a8r1-e4b-20261110a": {
        "kind": "scores_only", "code": "7a67a2ef4",
        "title": "A8 other model families, Gemma 4 E4B-it BF16 (Google): no final-turn line, no list (P C0), incomplete list (P C1) and complete list (S-VC C1), seed 11",
        "status": "follow-up A8 (protocol/EXPA_A8_MODELS.md and its lock); compared with main-20261101b; never pooled with other runs"},
    "a8r2-e4b-20261110a": {
        "kind": "scores_only", "code": "7a67a2ef4",
        "title": "A8 other model families, Gemma 4 E4B-it BF16 (Google): the A1 deadline line, no list (P C0) and incomplete list (P C1), seed 11",
        "status": "follow-up A8 (protocol/EXPA_A8_MODELS.md and its lock); compared with main-20261101b; never pooled with other runs"},
    "a8r1-e2b-20261110a": {
        "kind": "scores_only", "code": "7a67a2ef4",
        "title": "A8 other model families, Gemma 4 E2B-it BF16 (Google): no final-turn line, no list (P C0), incomplete list (P C1) and complete list (S-VC C1), seed 11",
        "status": "follow-up A8 (protocol/EXPA_A8_MODELS.md and its lock); compared with main-20261101b; never pooled with other runs"},
    "a8r2-e2b-20261110a": {
        "kind": "scores_only", "code": "7a67a2ef4",
        "title": "A8 other model families, Gemma 4 E2B-it BF16 (Google): the A1 deadline line, no list (P C0) and incomplete list (P C1), seed 11",
        "status": "follow-up A8 (protocol/EXPA_A8_MODELS.md and its lock); compared with main-20261101b; never pooled with other runs"},
    "a8r1-nemotron-20261110a": {
        "kind": "scores_only", "code": "7a67a2ef4",
        "title": "A8 other model families, Nemotron 3 Nano 30B-A3B (NVIDIA): no final-turn line, no list (P C0), incomplete list (P C1) and complete list (S-VC C1), seed 11",
        "status": "follow-up A8 (protocol/EXPA_A8_MODELS.md and its lock); compared with main-20261101b; never pooled with other runs"},
    "a8r2-nemotron-20261110a": {
        "kind": "scores_only", "code": "7a67a2ef4",
        "title": "A8 other model families, Nemotron 3 Nano 30B-A3B (NVIDIA): the A1 deadline line, no list (P C0) and incomplete list (P C1), seed 11",
        "status": "follow-up A8 (protocol/EXPA_A8_MODELS.md and its lock); compared with main-20261101b; never pooled with other runs"},
    "a7impact-20261107a": {"note": "first A7 main launch: refused at the server preflight (port still held by the dev "
                                   "run's server, port_in_use); no cell, no run.json; the main run is "
                                   "a7impact-20261107b (deviation D-7)"},
    "finalturn-dev-neutral-20261104a": {"note": "first A1 dev launch (GPU 5): the serve supervisor recorded a foreign "
                                                "process and stopped; no run.json, no cell (deviation D-5 amendment)"},
    "a5review-dev-ledger-20261106a": {"note": "first A5 ledger dev launch: refused at the server preflight (port still "
                                              "held by the previous run's server, port_in_use); no cell, no run.json; "
                                              "the dev check used a5review-dev-ledger-20261106b "
                                              "(protocol/EXPA_A5_REVIEW.md)"},
    "pilot-20261101a": {"note": "first pilot launch: the serve records end in 'exited' during startup (returncode 1); "
                                "no cell directory, no run.json; superseded by pilot-20261101b"},
    "main-20261101a": {"note": "first main launch: staging refused the clean checkouts and no cell was dispatched "
                               "(deviation D-1 in protocol/EXPA_DEVIATIONS.md); superseded by main-20261101b"},
}
RUN_ID = re.compile(r"^(?P<prefix>[a-z0-9][a-z0-9-]*?)-(?P<stamp>\d{8}[a-z])$")

# ------------------------------------------------------------------------------------------------ repo files
# research-repo path -> package path (None: a build input that is not published)
REPO_FILES = {
    "docs/EXPA_DESIGN.md": "protocol/design/EXPA_DESIGN.md",
    "docs/EXPA_DESIGN_V2_DELTA.md": "protocol/design/EXPA_DESIGN_V2_DELTA.md",
    "docs/EXPA_REPOS.md": "protocol/design/EXPA_REPOS.md",
    "manifests/expA/prod-0b8ee0b7c/HASHES.md": "registry/HASHES.md",
    "reports/exposure-funnel-v1/FUNNEL.json": "exposure-audit/FUNNEL.json",
    "reports/exposure-funnel-v1/FUNNEL.md": "exposure-audit/FUNNEL.md",
    "reports/exposure-funnel-v1/FIGURE.json": "exposure-audit/FIGURE.json",
    "reports/exposure-funnel-v1/funnel.svg": "exposure-audit/funnel.svg",
    "reports/exposure-funnel-v1/funnel.png": "exposure-audit/funnel.png",
    "reports/expA-main-20261101b/README.md": "results/main-20261101b/README.md",
    "reports/expA-main-20261101b/explore/EXPLORE.json": "results/main-20261101b/explore/EXPLORE.json",
    "reports/expA-main-20261101b/explore/EXPLORE.md": "results/main-20261101b/explore/EXPLORE.md",
    "reports/expA-main-20261101b/independent/REANALYSIS.json": "results/main-20261101b/independent/REANALYSIS.json",
    "reports/expA-main-20261101b/independent/REANALYSIS.md": "results/main-20261101b/independent/REANALYSIS.md",
    "reports/expA-main-20261101b/independent/TCDF_SCIPY_CROSSCHECK.json":
        "results/main-20261101b/independent/TCDF_SCIPY_CROSSCHECK.json",
    "reports/expA-main-20261101b/timeline/episodes.jsonl": "results/main-20261101b/timeline/episodes.jsonl",
    "reports/expA-main-20261101b/timeline/RULES.md": "results/main-20261101b/timeline/RULES.md",
    "reports/expA-main-20261101b/timeline/SHA256SUMS": "results/main-20261101b/timeline/SHA256SUMS",
    "reports/expA-main-20261101b/timeline/TIMELINE.json": "results/main-20261101b/timeline/TIMELINE.json",
    "reports/expA-main-20261101b/timeline/TIMELINE.md": "results/main-20261101b/timeline/TIMELINE.md",
    "scripts/expA_reanalyze_independent.py": "code/post-results/expA_reanalyze_independent.py",
    "scripts/expA_compare_runs.py": "code/post-results/expA_compare_runs.py",
    "scripts/expA_reanalyze_tcdf_scipy_check.py": "code/post-results/expA_reanalyze_tcdf_scipy_check.py",
    "scripts/expA_explore.py": "code/post-results/expA_explore.py",
    "scripts/expA_timeline.py": "code/post-results/expA_timeline.py",
    "scripts/expA_followup_analysis.py": "code/post-results/expA_followup_analysis.py",
    "scripts/expA_omission_cost.py": "code/post-results/expA_omission_cost.py",
    "scripts/expA_site_audit.py": "code/post-results/expA_site_audit.py",
    "reports/expA-site-audit-20261104/SITE_AUDIT.json": "results/post-results/SITE_AUDIT.json",
    "reports/expA-omission-cost-20261104/OMISSION_COST.json": "results/post-results/OMISSION_COST.json",
    "reports/expA-omission-cost-20261104/OMISSION_COST.md": "results/post-results/OMISSION_COST.md",
    "reports/expA-hard-20261102a/followup/RESULTS.json": "results/hard-20261102a/followup/RESULTS.json",
    "reports/expA-hard-20261102a/followup/RESULTS.md": "results/hard-20261102a/followup/RESULTS.md",
    "reports/expA-finalturn-20261102a/analysis/RESULTS.json": "results/finalturn-20261102a/RESULTS.json",
    "reports/expA-finalturn-20261102a/analysis/RESULTS.md": "results/finalturn-20261102a/RESULTS.md",
    "scripts/expA_a3_devcheck.py": "code/post-results/expA_a3_devcheck.py",
    "scripts/expA_followup_paired.py": "code/post-results/expA_followup_paired.py",
    "reports/expA-followup-paired-20261107/FOLLOWUP_PAIRED.json": "results/post-results/FOLLOWUP_PAIRED.json",
    "reports/expA-a3-devcheck-20261104b/DEVCHECK_off.json": "results/a3-devcheck-20261104b/DEVCHECK_off.json",
    "reports/expA-a3-devcheck-20261104b/DEVCHECK_on.json": "results/a3-devcheck-20261104b/DEVCHECK_on.json",
    # follow-up analyses A3-A5 (their outputs under results/followups/<name>/)
    "scripts/expA_a3_analysis.py": "code/post-results/expA_a3_analysis.py",
    "scripts/expA_a3_explore.py": "code/post-results/expA_a3_explore.py",
    "scripts/expA_a4_analysis.py": "code/post-results/expA_a4_analysis.py",
    "scripts/expA_a5_analysis.py": "code/post-results/expA_a5_analysis.py",
    "scripts/expA_a6_analysis.py": "code/post-results/expA_a6_analysis.py",
    "scripts/expA_followup_audit.py": "code/post-results/expA_followup_audit.py",
    "scripts/expA_followup_holm.py": "code/post-results/expA_followup_holm.py",
    "reports/expA-a6budget16-20261106/A6_ANALYSIS.json": "results/followups/a6budget16-20261106/A6_ANALYSIS.json",
    "reports/expA-a6budget16-20261106/A6_TABLES.md": "results/followups/a6budget16-20261106/A6_TABLES.md",
    "reports/expA-followup-audit-20261106/FOLLOWUP_AUDIT.json": "results/post-results/FOLLOWUP_AUDIT.json",
    "reports/expA-followup-holm-20261107/FOLLOWUP_HOLM.json": "results/post-results/FOLLOWUP_HOLM.json",
    "reports/expA-a3ledger-20261105/A3_ANALYSIS.json": "results/followups/a3ledger-20261105/A3_ANALYSIS.json",
    "reports/expA-a3ledger-20261105/A3_TABLES.md": "results/followups/a3ledger-20261105/A3_TABLES.md",
    "reports/expA-a3ledger-20261105/ledger-stats/LEDGER_STATS_off.json":
        "results/followups/a3ledger-20261105/ledger-stats/LEDGER_STATS_off.json",
    "reports/expA-a3ledger-20261105/ledger-stats/LEDGER_STATS_on.json":
        "results/followups/a3ledger-20261105/ledger-stats/LEDGER_STATS_on.json",
    "reports/expA-a3-explore-20261106b/A3_EXPLORE.json": "results/followups/a3-explore-20261106b/A3_EXPLORE.json",
    # post hoc summaries of 2026-11-07 (generality matrix, worked example, function-level scoring) and A7
    "scripts/expA_generality.py": "code/post-results/expA_generality.py",
    "reports/expA-generality-20261107/GENERALITY.json": "results/post-results/GENERALITY.json",
    "scripts/expA_worked_example.py": "code/post-results/expA_worked_example.py",
    "reports/expA-worked-example-20261107/WORKED_EXAMPLE.json": "results/post-results/WORKED_EXAMPLE.json",
    "scripts/expA_function_level.py": "code/post-results/expA_function_level.py",
    "reports/expA-function-level-20261107/FUNCTION_LEVEL.json": "results/post-results/FUNCTION_LEVEL.json",
    "scripts/expA_a7_analysis.py": "code/post-results/expA_a7_analysis.py",
    "scripts/expA_a7_devcheck.py": "code/post-results/expA_a7_devcheck.py",
    "scripts/expA_a8_analysis.py": "code/post-results/expA_a8_analysis.py",
    "reports/expA-a7-devcheck-20261107/A7DEV.json": "results/a7-devcheck-20261107/A7DEV.json",
    "scripts/expA_release_verify.py": "reproduce/verify.py",
    "scripts/expA_build_release_package.py": "reproduce/build_release_package.py",
    "staged/paper-prep-20261101/release-plan/license_audit.json": None,
    "staged/paper-prep-20261101/release-plan/package-v1/README.tmpl.md": None,
}
# Exported and published when present in the working tree at export time (skipped silently when absent): the A4 and
# A5 analysis outputs, added after their runs are scored.
OPTIONAL_REPO_FILES = {
    "reports/expA-a4directive-20261106/A4_RESULTS.json": "results/followups/a4directive-20261106/A4_RESULTS.json",
    "reports/expA-a4directive-20261106/A4_RESULTS.md": "results/followups/a4directive-20261106/A4_RESULTS.md",
    "reports/expA-a5review-20261106/A5_ANALYSIS.json": "results/followups/a5review-20261106/A5_ANALYSIS.json",
    "reports/expA-a5review-20261106/A5_TABLES.md": "results/followups/a5review-20261106/A5_TABLES.md",
    "reports/expA-a7impact-20261107/A7_ANALYSIS.json": "results/followups/a7impact-20261107/A7_ANALYSIS.json",
    "reports/expA-a7impact-20261107/A7_TABLES.md": "results/followups/a7impact-20261107/A7_TABLES.md",
    "reports/expA-a8models-20261110/A8_ANALYSIS.json": "results/followups/a8models-20261110/A8_ANALYSIS.json",
    "reports/expA-a8models-20261110/A8_TABLES.md": "results/followups/a8models-20261110/A8_TABLES.md",
    "reports/expA-a8models-20261110/A8_ROSTER.json": "results/followups/a8models-20261110/A8_ROSTER.json",
}

# ------------------------------------------------------------------------------------------------ follow-ups
# Pre-specified follow-up analyses that reproduce/verify.py re-runs from the package's own scores and code. "runs" maps
# the script's arm names to run ids; "outputs" maps each output file name of the script to its package path; "json"
# is the output compared field by field when the bytes differ. The command line is derived from the shipped output's
# own "inputs" record (write_verify_plan), with package-relative paths.
FOLLOWUPS = [
    {"name": "A1 final-turn policy", "id": "a1", "protocol": "protocol/EXPA_FINALTURN.md",
     "script": "code/post-results/expA_followup_analysis.py",
     "runs": {"neutral": "finalturn-neutral-20261102a", "deadline": "finalturn-deadline-20261102a",
              "none": "finalturn-none-20261102a"},
     "outputs": {"RESULTS.json": "results/finalturn-20261102a/RESULTS.json",
                 "RESULTS.md": "results/finalturn-20261102a/RESULTS.md"}, "json": "RESULTS.json"},
    {"name": "A2 hard-omission panel", "id": "a2", "protocol": "protocol/EXPA_HARD.md",
     "script": "code/post-results/expA_followup_analysis.py",
     "runs": {"hard": "hard-20261102a", "main": "main-20261101b"},
     "outputs": {"RESULTS.json": "results/hard-20261102a/followup/RESULTS.json",
                 "RESULTS.md": "results/hard-20261102a/followup/RESULTS.md"}, "json": "RESULTS.json"},
    {"name": "A3 evidence ledger", "id": "a3", "protocol": "protocol/EXPA_A3_LEDGER.md",
     "script": "code/post-results/expA_a3_analysis.py",
     "runs": {"off": "a3ledger-off-20261104a", "on": "a3ledger-on-20261104a"},
     "outputs": {"A3_ANALYSIS.json": "results/followups/a3ledger-20261105/A3_ANALYSIS.json",
                 "A3_TABLES.md": "results/followups/a3ledger-20261105/A3_TABLES.md"}, "json": "A3_ANALYSIS.json"},
    {"name": "A4 final-line components", "id": "a4", "protocol": "protocol/EXPA_A4_DIRECTIVE.md",
     "script": "code/post-results/expA_a4_analysis.py",
     "runs": {arm: f"a4directive-{arm}-20261106a" for arm in ("neutral", "info", "submit", "deadline")},
     "outputs": {"A4_RESULTS.json": "results/followups/a4directive-20261106/A4_RESULTS.json",
                 "A4_RESULTS.md": "results/followups/a4directive-20261106/A4_RESULTS.md"}, "json": "A4_RESULTS.json"},
    {"name": "A5 submit-time review", "id": "a5", "protocol": "protocol/EXPA_A5_REVIEW.md",
     "script": "code/post-results/expA_a5_analysis.py",
     "runs": {"neutral": "a5review-neutral-20261106a", "ledger": "a5review-ledger-20261106a"},
     "outputs": {"A5_ANALYSIS.json": "results/followups/a5review-20261106/A5_ANALYSIS.json",
                 "A5_TABLES.md": "results/followups/a5review-20261106/A5_TABLES.md"}, "json": "A5_ANALYSIS.json"},
    {"name": "A6 doubled budget", "id": "a6", "protocol": "protocol/EXPA_A6_BUDGET16.md",
     "script": "code/post-results/expA_a6_analysis.py",
     "runs": {"16": "a6budget16-20261106a", "8": "a3ledger-off-20261104a"},
     "outputs": {"A6_ANALYSIS.json": "results/followups/a6budget16-20261106/A6_ANALYSIS.json",
                 "A6_TABLES.md": "results/followups/a6budget16-20261106/A6_TABLES.md"}, "json": "A6_ANALYSIS.json"},
    {"name": "A7 impact-analysis framing", "id": "a7", "protocol": "protocol/EXPA_A7_IMPACT.md",
     "script": "code/post-results/expA_a7_analysis.py",
     "runs": {"impact": "a7impact-20261107b", "original": "main-20261101b"},
     "outputs": {"A7_ANALYSIS.json": "results/followups/a7impact-20261107/A7_ANALYSIS.json",
                 "A7_TABLES.md": "results/followups/a7impact-20261107/A7_TABLES.md"}, "json": "A7_ANALYSIS.json"},
    {"name": "A8 other model families", "id": "a8", "protocol": "protocol/EXPA_A8_MODELS.md",
     "script": "code/post-results/expA_a8_analysis.py",
     "runs": {"gemma": "main-20261101b",
              "qwen3-coder-30b-a3b|r1": "a8r1-qwen-20261110a", "qwen3-coder-30b-a3b|r2": "a8r2-qwen-20261110a",
              "glm-4-5-air-fp8|r1": "a8r1-glm-20261110a", "glm-4-5-air-fp8|r2": "a8r2-glm-20261110a",
              "gemma4-e4b|r1": "a8r1-e4b-20261110a", "gemma4-e4b|r2": "a8r2-e4b-20261110a",
              "gemma4-e2b|r1": "a8r1-e2b-20261110a", "gemma4-e2b|r2": "a8r2-e2b-20261110a",
              "nemotron3-nano-30b|r1": "a8r1-nemotron-20261110a", "nemotron3-nano-30b|r2": "a8r2-nemotron-20261110a",
              },
     "outputs": {"A8_ANALYSIS.json": "results/followups/a8models-20261110/A8_ANALYSIS.json",
                 "A8_TABLES.md": "results/followups/a8models-20261110/A8_TABLES.md"}, "json": "A8_ANALYSIS.json"},
]
# Follow-up outputs that are shipped but not re-run by verify.py (descriptive or post hoc; they read raw episode
# events), listed in the README's follow-up table.
FOLLOWUP_EXTRAS = [
    {"name": "A3 ledger statistics", "protocol": "protocol/EXPA_A3_LEDGER.md",
     "script": "code/post-results/expA_a3_devcheck.py",
     "outputs": ["results/followups/a3ledger-20261105/ledger-stats/LEDGER_STATS_off.json",
                 "results/followups/a3ledger-20261105/ledger-stats/LEDGER_STATS_on.json"],
     "why": "descriptive statistics of the ledger texts, read from the raw episode events"},
    {"name": "A3 exploration (accounting bound, ledger subgroup, loss breakdown)", "protocol": "protocol/EXPA_A3_LEDGER.md",
     "script": "code/post-results/expA_a3_explore.py",
     "outputs": ["results/followups/a3-explore-20261106b/A3_EXPLORE.json"],
     "why": "post hoc and descriptive (the subgroup is pre-specified as descriptive); reads the raw episode events"},
]
# Exported by glob: every Exp A protocol and lock (published under protocol/), and the frozen per-run extras of the
# research-repo report directories (published under results/<run>/ when <run> is in the build).
REPO_GLOBS = ["protocols/EXPA_*.md", "protocols/EXPA_*.lock.json", "reports/expA-*/COMPARE.json",
              "reports/expA-*/MODEL_COPY.sha256"]
# Tracked paths whose blob ids are exported as witnesses (byte-identical repo copies of Core2 files).
WITNESS_TREES = ["manifests/expA", "reports", "protocols", "docs", "paper/evidence", "scripts"]
# Frozen repo files that are pinned elsewhere and must be byte-identical to HEAD at export time.
MUST_EQUAL_HEAD_PREFIXES = ("protocols/", "docs/", "manifests/", "reports/")

FAMILY_EXTRA_LICENCE_PREFIXES = {  # LICENSE_AUDIT.md section 4.3 (licence files outside package scope)
    "yapf": ["third_party/yapf_third_party/_ylib2to3/", "third_party/yapf_third_party/yapf_diff/"],
    "rdflib": ["test/data/suites/w3c/"],
    "segno": ["examples/qrcode_text/font/"],
    "python-docx": ["docs/_themes/armstrong/"],
    "python-pptx": ["docs/.themes/armstrong/"],
    "param": ["doc/governance/project-docs/"],
    "genshi": ["examples/tutorial/geddit/static/"],
}
LICENCE_FILE_RE = re.compile(r"(^|/)(LICEN[CS]E|COPYING|NOTICE)[^/]*$", re.I)
HEX64 = re.compile(rb"(?<![0-9a-f])([0-9a-f]{64})(?![0-9a-f])")
HEX64_STR = re.compile(r"^[0-9a-f]{64}$")


# ================================================================================================ utilities
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_id(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def dump_json(value) -> bytes:
    return (json.dumps(value, indent=1, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def jsonl_line(value) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(message: str) -> None:
    print(f"[{utc_now()}] {message}", flush=True)


def fail(message: str) -> None:
    raise SystemExit(f"BUILD REFUSED: {message}")


def posix_sorted(paths):
    return sorted(paths, key=lambda text: text.encode("utf-8"))


def walk_files(root: Path):
    """Relative posix paths of regular files under root (refuses symlinks and special files)."""
    out = []
    for current, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames.sort()
        for name in filenames:
            full = Path(current) / name
            mode = full.lstat().st_mode
            if not (mode & 0o170000 == 0o100000):
                fail(f"not a regular file: {full}")
            out.append(full.relative_to(root).as_posix())
        for name in dirnames:
            if (Path(current) / name).is_symlink():
                fail(f"symlinked directory: {Path(current) / name}")
    return posix_sorted(out)


# ------------------------------------------------------------------------------------------------ JSON paths
_PATH_TOKEN = re.compile(r'\.?([A-Za-z0-9_\-]+|\*)|\[(\d+)\]|\[("(?:[^"\\]|\\.)*")\]|\[([^\]"]+="[^\]]*")\]')
_SELECT = re.compile(r'([A-Za-z0-9_\-]+)="([^"]*)"')


def parse_json_path(text: str):
    tokens, position = [], 0
    while position < len(text):
        match = _PATH_TOKEN.match(text, position)
        if not match:
            raise ValueError(f"bad JSON path {text!r} at {position}")
        name, index, quoted, select = match.groups()
        if name is not None:
            tokens.append(("*",) if name == "*" else ("key", name))
        elif index is not None:
            tokens.append(("index", int(index)))
        elif quoted is not None:
            tokens.append(("key", json.loads(quoted)))
        else:
            tokens.append(("select", _SELECT.findall(select)))
        position = match.end()
    return tokens


def resolve_all(document, text: str):
    values = [document]
    for token in parse_json_path(text):
        nxt = []
        for value in values:
            if token[0] == "key":
                if not isinstance(value, dict) or token[1] not in value:
                    raise KeyError(f"{text}: no key {token[1]!r}")
                nxt.append(value[token[1]])
            elif token[0] == "index":
                nxt.append(value[token[1]])
            elif token[0] == "*":
                nxt.extend(value.values() if isinstance(value, dict) else value)
            else:
                rows = [row for row in value if isinstance(row, dict)
                        and all(str(row.get(k)) == v for k, v in token[1])]
                if len(rows) != 1:
                    raise KeyError(f"{text}: selector {token[1]} matched {len(rows)} rows")
                nxt.append(rows[0])
        values = nxt
    return values


def resolve_one(document, text: str):
    values = resolve_all(document, text)
    if len(values) != 1:
        raise KeyError(f"{text}: {len(values)} values")
    return values[0]


# ================================================================================================ export
def git(repo: Path, *args, binary=False):
    out = subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)
    return out.stdout if binary else out.stdout.decode("utf-8")


def cmd_export(args) -> int:
    repo = Path(args.repo).resolve()
    head = git(repo, "rev-parse", "HEAD").strip()
    status = {}
    files = dict(REPO_FILES)
    files.update({src: dst for src, dst in OPTIONAL_REPO_FILES.items() if (repo / src).is_file()})
    for pattern in REPO_GLOBS:
        for path in sorted(repo.glob(pattern)):
            if path.is_file():
                files[path.relative_to(repo).as_posix()] = "glob"
    entries = {}
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for src in sorted(files):
            data = (repo / src).read_bytes()
            try:
                head_blob = git(repo, "rev-parse", f"HEAD:{src}").strip()
            except subprocess.CalledProcessError:
                head_blob = None
            blob = git_blob_id(data)
            if src.startswith(MUST_EQUAL_HEAD_PREFIXES) and head_blob != blob:
                raise SystemExit(f"refusing: {src} differs from HEAD (commit it first)")
            entries[src] = {"sha256": sha256_bytes(data), "bytes": len(data), "git_blob": blob,
                            "head_blob": head_blob, "equals_head": head_blob == blob}
            info = tarfile.TarInfo("repo/" + src)
            info.size, info.mtime, info.mode = len(data), TAR_MTIME, 0o644
            tar.addfile(info, io.BytesIO(data))
            status[src] = "HEAD" if head_blob == blob else ("modified" if head_blob else "untracked")
        listing = git(repo, "ls-tree", "-r", "--full-tree", "HEAD", "--", *WITNESS_TREES)
        blobs = {}
        for line in listing.splitlines():
            meta, path = line.split("\t", 1)
            _, kind, blob = meta.split()
            if kind == "blob":
                blobs.setdefault(blob, []).append(path)
        commits = git(repo, "rev-list", "HEAD").split()
        meta = {"schema": "absent-edges-expA-repo-export-v1", "head": head, "exported_utc": utc_now(),
                "files": entries, "commits": commits, "head_blobs": blobs}
        raw = dump_json(meta)
        info = tarfile.TarInfo("REPO_EXPORT.json")
        info.size, info.mtime, info.mode = len(raw), TAR_MTIME, 0o644
        tar.addfile(info, io.BytesIO(raw))
    Path(args.out).write_bytes(buffer.getvalue())
    print(json.dumps({"out": args.out, "head": head, "files": len(entries), "status": status,
                      "sha256": sha256_bytes(buffer.getvalue())}, indent=1))
    return 0


# ================================================================================================ scanning
class Scanner:
    """Finds personal tokens (case-insensitive); never prints a token itself."""

    OWN_CLASSES = ("own_account", "own_identity", "own_windows_account", "own_storage_path")

    def __init__(self, private: dict):
        self.other = sorted({t.lower().encode() for t in private.get("other_user", []) if t}, key=len, reverse=True)
        self.own = {cls: sorted({t.lower().encode() for t in private.get(cls, []) if t}) for cls in self.OWN_CLASSES}
        if not self.other or not self.own["own_account"]:
            fail("the private pattern file needs other_user and own_account entries")
        if not private.get("name_fragments"):
            fail("the private pattern file needs name_fragments (used by the leak check)")
        fragments = {t.lower() for t in private["name_fragments"] if t}
        for cls in ("other_user", "own_account", "own_identity"):
            for token in private.get(cls, []):
                fragments.update(part for part in re.split(r"[^a-z]+", token.lower()) if len(part) >= 5)
        self.fragments = sorted(f.encode() for f in fragments)
        # identifies the pattern set in the (private) shard scan cache; never published
        self.digest = sha256_bytes(json.dumps({
            "other": [t.decode() for t in self.other], "fragments": [f.decode() for f in self.fragments],
            "own": {cls: [t.decode() for t in tokens] for cls, tokens in self.own.items()},
            "redaction": REDACTION.decode(), "scan_format": SCAN_FORMAT}, sort_keys=True).encode())

    @staticmethod
    def _lines(data: bytes, offsets):
        out = []
        for offset in offsets:
            out.append(data.count(b"\n", 0, offset) + 1)
        return out

    @staticmethod
    def _find(lower: bytes, token: bytes):
        offsets, start = [], 0
        while True:
            found = lower.find(token, start)
            if found < 0:
                return offsets
            offsets.append(found)
            start = found + len(token)

    def other_hits(self, data: bytes):
        lower = data.lower()
        hits = []
        for token in self.other:
            for offset in self._find(lower, token):
                hits.append((offset, len(token)))
        # longest tokens first; drop overlaps
        hits.sort()
        merged = []
        for offset, length in hits:
            if merged and offset < merged[-1][0] + merged[-1][1]:
                continue
            merged.append((offset, length))
        return merged

    def redact(self, data: bytes):
        hits = self.other_hits(data)
        if not hits:
            return data, []
        lines = self._lines(data, [offset for offset, _ in hits])
        out, last = [], 0
        for offset, length in hits:
            out.append(data[last:offset])
            out.append(REDACTION)
            last = offset + length
        out.append(data[last:])
        return b"".join(out), lines

    def _own_spans(self, lower: bytes):
        """{class: [(offset, length)]} of the own_* tokens in lower-cased data."""
        found = {}
        for cls, tokens in self.own.items():
            spans = []
            for token in tokens:
                spans.extend((offset, len(token)) for offset in self._find(lower, token))
            if spans:
                found[cls] = sorted(spans)
        return found

    def own_hits(self, data: bytes):
        return {cls: self._lines(data, [offset for offset, _ in spans])
                for cls, spans in self._own_spans(data.lower()).items()}

    def scan(self, data: bytes):
        """(holds an other_user token, {own class: line numbers}, line numbers of name fragments that do not lie
        inside an own_* occurrence)."""
        lower = data.lower()
        other = any(lower.find(token) >= 0 for token in self.other)
        spans = self._own_spans(lower)
        own = {cls: self._lines(data, [offset for offset, _ in s]) for cls, s in spans.items()}
        flat = [span for s in spans.values() for span in s]
        unlisted = set()
        for fragment in self.fragments:
            for offset in self._find(lower, fragment):
                if not any(start <= offset and offset + len(fragment) <= start + length for start, length in flat):
                    unlisted.add(offset)
        return other, own, self._lines(data, sorted(unlisted))


# ================================================================================================ build state
class Build:
    def __init__(self, args):
        self.args = args
        self.out_root = Path(args.out_root)
        self.work = self.out_root / "work"
        self.cache = self.work / "cache"
        self.stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        self.stage_root = self.out_root / f".staging-{self.stamp}"
        self.stage = self.stage_root / PACKAGE
        self.files = {}          # package path -> record (stored files)
        self.dedup = {}          # logical package path -> record
        self.shards = {}         # package path of shard -> record
        self.sanitized = []      # records of changed files
        self.own = []            # (where, member or None, class, lines)
        self.leaks = []          # name fragments outside a listed own_* occurrence: (file, member or None, lines)
        self.listed_own = None   # the own_* occurrences that SANITIZATION.md lists (set when it is written)
        self.licence_mentions = collections.Counter()
        self.notes = []
        self.private = json.loads(Path(args.private).read_text(encoding="utf-8"))
        self.scanner = Scanner(self.private)
        self.export_dir = None
        self.export = None

    # -------------------------------------------------------------------------------- export input
    def load_export(self):
        tar_path = Path(self.args.export)
        raw = tar_path.read_bytes()
        target = self.work / f"export-{sha256_bytes(raw)[:16]}"
        if not target.is_dir():
            tmp = target.with_name(target.name + ".tmp")
            if tmp.exists():
                shutil.rmtree(tmp)
            tmp.mkdir(parents=True)
            with tarfile.open(fileobj=io.BytesIO(raw), mode="r") as tar:
                for member in tar.getmembers():
                    if not member.isfile() or member.name.startswith("/") or ".." in PurePosixPath(member.name).parts:
                        fail(f"unexpected export member {member.name}")
                    dest = tmp / member.name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(tar.extractfile(member).read())
            tmp.rename(target)
        self.export_dir = target
        self.export = json.loads((target / "REPO_EXPORT.json").read_text(encoding="utf-8"))
        self.export_sha256 = sha256_bytes(raw)
        for src, meta in self.export["files"].items():
            data = (target / "repo" / src).read_bytes()
            if sha256_bytes(data) != meta["sha256"]:
                fail(f"export file {src} does not match REPO_EXPORT.json")
        log(f"export {tar_path} (head {self.export['head'][:12]}, {len(self.export['files'])} files)")

    def repo_bytes(self, src: str) -> bytes:
        return (self.export_dir / "repo" / src).read_bytes()

    # -------------------------------------------------------------------------------- writing plain files
    def add_bytes(self, pkg: str, data: bytes, source: dict, *, sanitize=True, generated=False):
        if pkg in self.files or pkg in self.dedup:
            fail(f"duplicate package path {pkg}")
        original_sha = sha256_bytes(data)
        lines = []
        if sanitize and not generated:
            data, lines = self.scanner.redact(data)
        elif self.scanner.other_hits(data):
            fail(f"{pkg}: generated content holds an other_user token")
        record = {"path": pkg, "sha256": sha256_bytes(data), "bytes": len(data), "source": source,
                  "byte_identical": (not generated) and not lines}
        if generated:
            record["byte_identical"] = None
        if lines:
            record["original_sha256"] = original_sha
            self.sanitized.append({"path": pkg, "class": "OTHER_USER", "replacement": REDACTION.decode(),
                                   "lines": lines, "count": len(lines), "original_sha256": original_sha,
                                   "new_sha256": record["sha256"], "source": source})
        _, own, unlisted = self.scanner.scan(data)
        for cls, own_lines in own.items():
            self.own.append({"file": pkg, "member": None, "class": cls, "lines": own_lines})
        if unlisted:
            self.leaks.append({"file": pkg, "member": None, "lines": unlisted})
        dest = self.stage / pkg
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        record["git_blob"] = git_blob_id(data)
        self.files[pkg] = record
        return record

    def add_file(self, pkg: str, path, source_kind="core2", **extra):
        path = Path(path)
        if not path.is_file():
            fail(f"missing source {path}")
        source = {"kind": source_kind, "path": str(path)}
        source.update(extra)
        return self.add_bytes(pkg, path.read_bytes(), source)

    def add_repo(self, src: str, pkg: str):
        meta = self.export["files"][src]
        source = {"kind": "git", "path": src, "commit": self.export["head"], "git_blob": meta["git_blob"],
                  "equals_head": meta["equals_head"]}
        if not meta["equals_head"]:
            source["note"] = "working-tree file not yet committed at export time"
        return self.add_bytes(pkg, self.repo_bytes(src), source)

    def add_tree(self, pkg_prefix: str, root, skip=()):
        root = Path(root)
        for rel in walk_files(root):
            if any(rel == s or rel.startswith(s.rstrip("/") + "/") for s in skip):
                continue
            self.add_file(f"{pkg_prefix}/{rel}", root / rel)

    def add_same_as(self, pkg: str, source_path, stored_pkg: str):
        """A logical file that is byte-identical to an already stored package file (not stored twice)."""
        data = Path(source_path).read_bytes()
        sha = sha256_bytes(data)
        target = self.files[stored_pkg]
        if target["sha256"] != sha:
            fail(f"{pkg}: not byte-identical to {stored_pkg}")
        self.dedup[pkg] = {"path": pkg, "same_bytes_as": stored_pkg, "sha256": sha, "bytes": len(data),
                           "source": {"kind": "core2", "path": str(source_path)}, "byte_identical": True}

    # -------------------------------------------------------------------------------- shards
    def new_scan(self):
        return {"own": [], "other": [], "unlisted_fragments": [], "licence_mentions": {},
                "patterns_sha256": self.scanner.digest}

    def scan_member(self, member: str, data: bytes, scan: dict, mentions):
        other, own, unlisted = self.scanner.scan(data)
        if other:
            scan["other"].append(member)
        for cls, lines in own.items():
            scan["own"].append({"member": member, "class": cls, "lines": lines})
        if unlisted:
            scan["unlisted_fragments"].append({"member": member, "lines": unlisted})
        if member.endswith("events.jsonl"):
            for family, prefixes in FAMILY_EXTRA_LICENCE_PREFIXES.items():
                for prefix in prefixes:
                    if prefix.encode() in data:
                        mentions[f"{family}|{prefix}"] += 1

    def build_shard(self, name: str, pkg_dir: str, sources, kind: str, run=None):
        """sources: list of (member prefix, root dir, list of relative files)."""
        listing = []
        for prefix, root, rels in sources:
            for rel in rels:
                st = (Path(root) / rel).stat()
                listing.append([f"{prefix}/{rel}", st.st_size, st.st_mtime_ns, str(Path(root) / rel)])
        listing.sort(key=lambda row: row[0].encode("utf-8"))
        version = subprocess.run(["xz", "--version"], check=True, capture_output=True, text=True).stdout.split("\n")[0]
        key = sha256_bytes(json.dumps({"listing": [row[:3] for row in listing], "xz": XZ_ARGS, "xz_version": version,
                                       "mtime": TAR_MTIME, "format": BUILD_FORMAT}, sort_keys=True).encode())
        cdir = self.cache / name
        shard_file, index_file, scan_file = cdir / f"{name}.tar.xz", cdir / f"{name}.INDEX.jsonl", cdir / "SCAN.json"
        if (cdir / "KEY").is_file() and (cdir / "KEY").read_text().strip() == key and scan_file.is_file():
            cached = json.loads(scan_file.read_text(encoding="utf-8"))
            if cached.get("patterns_sha256") == self.scanner.digest:
                log(f"shard {name}: cache hit")
            else:
                # the shard bytes are still valid (same KEY); only the personal-data scan is redone
                log(f"shard {name}: cache hit; re-scanning every member (the pattern set or scan changed)")
                t0 = time.time()
                scan, mentions, seen = self.new_scan(), collections.Counter(), 0
                with tarfile.open(shard_file, mode="r|xz") as tar:
                    for member in tar:
                        if not member.isfile():
                            fail(f"shard {name}: member {member.name} is not a regular file")
                        self.scan_member(member.name, tar.extractfile(member).read(), scan, mentions)
                        seen += 1
                expected = json.loads((cdir / "META.json").read_text(encoding="utf-8"))["members"]
                if seen != expected:
                    fail(f"shard {name}: {seen} members re-scanned, META.json lists {expected}")
                scan["licence_mentions"] = dict(mentions)
                tmp = scan_file.with_suffix(".tmp")
                tmp.write_bytes(dump_json(scan))
                os.replace(tmp, scan_file)
                log(f"shard {name}: {seen} members re-scanned in {time.time() - t0:.0f} s")
        else:
            if cdir.exists():
                shutil.rmtree(cdir)
            cdir.mkdir(parents=True)
            log(f"shard {name}: {len(listing)} members, {sum(r[1] for r in listing):,} bytes -> xz")
            scan = self.new_scan()
            mentions = collections.Counter()
            tmp = shard_file.with_suffix(".tmp")
            with open(tmp, "wb") as sink:
                proc = subprocess.Popen(XZ_ARGS, stdin=subprocess.PIPE, stdout=sink)
                tar = tarfile.open(fileobj=proc.stdin, mode="w|", format=tarfile.PAX_FORMAT)
                rows = []
                for member, size, _, full in listing:
                    data = Path(full).read_bytes()
                    if len(data) != size:
                        fail(f"{full} changed while building")
                    info = tarfile.TarInfo(member)
                    info.size, info.mtime, info.mode, info.uid, info.gid = len(data), TAR_MTIME, 0o644, 0, 0
                    info.uname = info.gname = ""
                    tar.addfile(info, io.BytesIO(data))
                    rows.append({"member": member, "bytes": len(data), "sha256": sha256_bytes(data)})
                    self.scan_member(member, data, scan, mentions)
                tar.close()
                proc.stdin.close()
                if proc.wait() != 0:
                    fail(f"xz failed for {name}")
            os.replace(tmp, shard_file)
            index_file.write_bytes(b"".join(jsonl_line(r) for r in rows))
            scan["licence_mentions"] = dict(mentions)
            scan_file.write_bytes(dump_json(scan))
            meta = {"members": len(rows), "bytes_uncompressed": sum(r["bytes"] for r in rows), "xz_version": version}
            (cdir / "META.json").write_bytes(dump_json(meta))
            (cdir / "KEY").write_text(key + "\n")
        scan = json.loads(scan_file.read_text(encoding="utf-8"))
        meta = json.loads((cdir / "META.json").read_text(encoding="utf-8"))
        if scan["other"]:
            fail(f"shard {name}: {len(scan['other'])} member(s) hold an other_user token, e.g. {scan['other'][:3]}")
        for row in scan["own"]:
            self.own.append({"file": f"{pkg_dir}/{name}.tar.xz", "member": row["member"], "class": row["class"],
                             "lines": row["lines"]})
        for row in scan["unlisted_fragments"]:
            self.leaks.append({"file": f"{pkg_dir}/{name}.tar.xz", "member": row["member"], "lines": row["lines"]})
        for k, v in scan["licence_mentions"].items():
            self.licence_mentions[k] += v
        # link into the staging package
        for src, pkg in ((shard_file, f"{pkg_dir}/{name}.tar.xz"), (index_file, f"{pkg_dir}/{name}.INDEX.jsonl")):
            dest = self.stage / pkg
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(src, dest)
            except OSError:
                shutil.copyfile(src, dest)
        shard_sha = sha256_file(shard_file)
        index_raw = index_file.read_bytes()
        roots = sorted({str(root) for _, root, _ in sources})
        self.files[f"{pkg_dir}/{name}.tar.xz"] = {
            "path": f"{pkg_dir}/{name}.tar.xz", "sha256": shard_sha, "bytes": shard_file.stat().st_size,
            "source": {"kind": "shard", "roots": roots}, "byte_identical": None,
            "shard": {"kind": kind, "run": run, "index": f"{pkg_dir}/{name}.INDEX.jsonl", "members": meta["members"],
                      "bytes_uncompressed": meta["bytes_uncompressed"], "tar": "POSIX pax, regular files only, names "
                      "sorted by UTF-8 bytes, mtime 2026-11-01T00:00:00Z, uid/gid 0, empty uname/gname, mode 0644",
                      "xz": " ".join(XZ_ARGS), "xz_version": meta["xz_version"]}}
        self.files[f"{pkg_dir}/{name}.INDEX.jsonl"] = {
            "path": f"{pkg_dir}/{name}.INDEX.jsonl", "sha256": sha256_bytes(index_raw), "bytes": len(index_raw),
            "source": {"kind": "generated", "what": f"member list of {name}.tar.xz with the sha256 of every member"},
            "byte_identical": None}
        self.shards[f"{pkg_dir}/{name}.tar.xz"] = {"name": name, "index": f"{pkg_dir}/{name}.INDEX.jsonl",
                                                   "kind": kind, "run": run, "members": meta["members"],
                                                   "bytes_uncompressed": meta["bytes_uncompressed"],
                                                   "bytes": shard_file.stat().st_size}
        log(f"shard {name}: {meta['members']} members, {shard_file.stat().st_size:,} bytes compressed")


# ================================================================================================ discovery
def read_sums(path: Path) -> dict:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        sha, _, rel = line.partition("  ")
        if rel:
            out[rel] = sha
    return out


def scores_frozen(run_dir: Path, scored: Path):
    """True when <scored> holds the frozen scores of the run (scores.jsonl, its manifest, and OUTPUTS.sha256 naming
    the scores' sha256, bound to the run's run.json and cells.jsonl); otherwise the reason it is not (yet) usable."""
    scores, manifest, outputs = scored / "scores.jsonl", scored / "scores.jsonl.manifest.json", scored / "OUTPUTS.sha256"
    for path in (scores, manifest, outputs):
        if not path.is_file():
            return f"no {path.relative_to(scored.parent)}"
    sums = {PurePosixPath(rel).name: sha for rel, sha in read_sums(outputs).items()}
    if sums.get("scores.jsonl") != sha256_file(scores):
        return f"{outputs.name} does not name the sha256 of scores.jsonl"
    binding = json.loads(manifest.read_text(encoding="utf-8")).get("binding") or {}
    for key, name in (("run_json_sha256", "run.json"), ("cells_jsonl_sha256", "cells.jsonl")):
        if binding.get(key) != sha256_file(run_dir / name):
            fail(f"{scored.name}: scores.jsonl.manifest.json binding.{key} is not the sha256 of {run_dir.name}/{name}")
    return True


def discover(b: Build):
    """Releases, runs, analyses and comparisons present on Core2 and in the export."""
    commits = b.export.get("commits") or []
    order = {c: i for i, c in enumerate(commits)}            # 0 = HEAD (newest)
    b.releases = {}
    for path in sorted((Path(EXPA) / "releases").iterdir()):
        if path.is_dir() and (path / "RELEASE_FILES.sha256").is_file():
            short = path.name
            full = [c for c in commits if c.startswith(short)]
            b.releases[short] = {"role": RELEASE_ROLES.get(short, "code release (see the runs that use it)"),
                                 "commit": full[0] if len(full) == 1 else None,
                                 "order": order.get(full[0], len(commits)) if len(full) == 1 else len(commits) + 1,
                                 "sums": read_sums(path / "RELEASE_FILES.sha256")}
    prod = Path(EXPA) / "prod-items-0b8ee0b7c"
    items_split = {sha256_file(prod / f"items-{split}/items.jsonl"): split for split in ("main", "dev")}
    items_split[sha256_file(Path(HARD) / "items-hard/items.jsonl")] = "hard"
    b.runs, b.aborted, b.skipped = [], [], []
    cells_owner = {}
    for path in sorted((Path(EXPA) / "runs").iterdir()):
        if not path.is_dir():
            continue
        rid, info = path.name, RUN_INFO.get(path.name, {})
        if not (path / "run.json").is_file():
            # launch records are published only for launches known to have ended without a dispatched cell;
            # any other directory without run.json may be a run in preparation
            if info.get("note") and not any(path.rglob("result.json")):
                b.aborted.append({"run": rid, "note": info["note"]})
            else:
                b.skipped.append(f"{rid}: no run.json yet (in preparation or unknown); not published")
            continue
        m = RUN_ID.match(rid)
        analysis = info.get("analysis") or (f"{m.group('prefix')}-analysis-{m.group('stamp')}" if m else None)
        kind = info.get("kind", "analysis")
        if kind == "scores_only":
            # a follow-up run whose analysis is a follow-up output: published with its frozen scores
            scored = scores_frozen(path, Path(EXPA) / analysis) if analysis else "no analysis directory name"
            if scored is not True:
                b.skipped.append(f"{rid}: not yet scored ({scored}); not published")
                continue
        elif not analysis or not (Path(EXPA) / analysis / "analysis/ANALYSIS.json").is_file():
            b.skipped.append(f"{rid}: no frozen analysis directory ({analysis}); not published")
            continue
        run_json = json.loads((path / "run.json").read_text(encoding="utf-8"))
        candidates = [short for short, rel in b.releases.items()
                      if "scripts/expA_analyze.py" in rel["sums"]
                      and all(rel["sums"].get(p) == h for p, h in run_json["code"].items())]
        if not candidates:
            fail(f"{rid}: no release tree matches the code hashes in run.json")
        code = min(candidates, key=lambda s: b.releases[s]["order"])
        if info.get("code") and code != info["code"]:
            fail(f"{rid}: the newest release matching run.json is {code}, RUN_INFO expects {info['code']}")
        split = items_split.get(run_json.get("items_sha256"))
        if split is None:
            fail(f"{rid}: run.json names an items file that is not in the package")
        cells_sha = None
        if kind == "analysis":
            cells_sha = sha256_file(Path(EXPA) / analysis / "analysis/CELLS.jsonl")
            cells_owner[cells_sha] = rid
        b.runs.append({"run": rid, "kind": kind, "analysis": analysis, "code": code,
                       "code_candidates": sorted(candidates), "items": split, "cells_sha256": cells_sha,
                       "mtime": (path / "run.json").stat().st_mtime,
                       "title": info.get("title", rid), "status": info.get("status", "see the run's protocol")})
    b.runs.sort(key=lambda r: (r["mtime"], r["run"]))
    for run in b.runs:
        src = f"reports/expA-{run['run']}/COMPARE.json"
        if src not in b.export["files"]:
            continue
        doc = json.loads(b.repo_bytes(src))
        ref = cells_owner.get(doc["inputs"]["ref_cells"]["sha256"])
        if ref is None or doc["inputs"]["new_cells"]["sha256"] != run["cells_sha256"]:
            fail(f"{src}: its CELLS.jsonl inputs do not match the runs in the package")
        run["compare"] = {"mode": doc["mode"], "ref": ref, "frozen_source": src}
        if RUN_INFO.get(run["run"], {}).get("compare_labels"):
            run["compare"]["labels"] = RUN_INFO[run["run"]]["compare_labels"]
        if doc["mode"] == "budget":
            run["compare"]["seed"] = doc["seed"]
    # publish the known releases and those a published run uses (a release prepared for a run that has no frozen
    # analysis yet is published together with that run)
    used = {r["code"] for r in b.runs} | {c for r in b.runs for c in r["code_candidates"]}
    unused = sorted(s for s in b.releases if s not in RELEASE_ROLES and s not in used)
    for short in unused:
        del b.releases[short]
    if unused:
        b.notes.append("releases not published (no published run uses them yet): " + ", ".join(unused))
    log("runs: " + ", ".join(f"{r['run']} (code {r['code']}, items {r['items']}"
                             + (f", compare {r['compare']['mode']} vs {r['compare']['ref']}" if r.get("compare") else "")
                             + ")" for r in b.runs)
        + "; launch records: " + ", ".join(r["run"] for r in b.aborted)
        + ("; skipped: " + "; ".join(b.skipped) if b.skipped else ""))


# ================================================================================================ build steps
def step_protocol_registry(b: Build):
    for src, dst in sorted(REPO_FILES.items()):
        if dst is None:
            continue
        if src not in b.export["files"]:
            fail(f"export lacks {src}")
        b.add_repo(src, dst)
    for src, dst in sorted(OPTIONAL_REPO_FILES.items()):
        if src in b.export["files"]:
            b.add_repo(src, dst)
    for src in sorted(b.export["files"]):
        if src.startswith("protocols/EXPA_"):
            b.add_repo(src, "protocol/" + src.rsplit("/", 1)[1])
    b.add_file("protocol/analysis-plan/PLAN.main.json", f"{EXPA}/main-plan/PLAN.main.json")
    b.add_file("protocol/analysis-plan/PLAN.hard.json", f"{HARD}/plan/PLAN.hard.json")
    # registry
    b.add_file("registry/registry.v1.json", f"{EXPA}/registry/registry.json")
    b.add_file("registry/clean-checkouts.json", f"{PUBLIC}/repos-manifests/clean-checkouts.json")
    for rel in walk_files(Path(PUBLIC) / "repos-manifests/builds"):
        b.add_file(f"registry/mirror/builds/{rel}", f"{PUBLIC}/repos-manifests/builds/{rel}")
    prod = f"{EXPA}/prod-items-0b8ee0b7c"
    for name in ("public-mirror-envs.files.sha256", "public-mirror-envs.symlinks", "mirror.log", "envhash.log"):
        b.add_file(f"registry/mirror/{name}", f"{prod}/{name}")
    for name in ("COPY.sha256", "ORIGINAL.sha256", "COPY-gemma4-12b-bf16-0e2b1058.sha256",
                 "COPY-gemma4-31b-qat-w4a16-52f3f65b.sha256"):
        b.add_file(f"registry/models/{name}", f"{PUBLIC}/models/{name}")
    b.add_file("registry/models/gemma4-31b-bf16-518276fb/chat_template.jinja",
               f"{PUBLIC}/models/gemma4-31b-bf16-518276fb/chat_template.jinja",
               note="Google file from the model repository (Apache-2.0, see THIRD_PARTY_NOTICES.md)")
    registry = json.loads((b.stage / "registry/registry.v1.json").read_text(encoding="utf-8"))
    used = [f for f in registry["families"] if f.get("role") in ("main", "dev")]
    b.families = sorted(used, key=lambda f: (f["role"] != "dev", f["family"]))
    for fam in b.families:
        b.add_file(f"registry/locks/{fam['family']}.txt", f"{EXPA}/registry/locks/{fam['family']}.txt")
    builds = sorted((Path(PUBLIC) / "repos-manifests/builds").glob("*.json"))
    generated = set()
    for path in builds:
        for fam in json.loads(path.read_text(encoding="utf-8")).get("families", []):
            for rel in fam.get("generated") or []:
                rel = rel["path"] if isinstance(rel, dict) else rel
                generated.add((fam["family"], rel))
    for family, rel in sorted(generated):
        b.add_file(f"registry/generated-files/{family}/{rel}", f"{PUBLIC}/repos/{family}/{rel}",
                   note="generated version file added to the model-visible tree (see clean-checkouts.json)")
    b.generated_files = sorted(generated)
    b.notes.append(f"generated version files: {len(generated)}")


def step_items(b: Build):
    prod = Path(f"{EXPA}/prod-items-0b8ee0b7c")
    for split in ("main", "dev"):
        b.add_tree(f"items/{split}", prod / f"items-{split}")
        dispatch = prod / f"dispatch-{split}"
        for rel in walk_files(dispatch):
            if rel.startswith("graph_facts/"):
                b.add_same_as(f"dispatch/{split}/{rel}", dispatch / rel, f"items/{split}/{rel}")
            else:
                b.add_file(f"dispatch/{split}/{rel}", dispatch / rel)
    for name in ("SPEC.dev.json", "SPEC.main.json", "SPEC.main-dev.json", "build-dev.log", "build-main.log",
                 "build-dev.time", "build-main.time", "dispatch-dev.log", "dispatch-main.log", "verify-dev.log",
                 "verify-main.log", "verify-dispatch-dev.log", "verify-dispatch-main.log"):
        b.add_file(f"items/build-record/{name}", prod / name)
    b.add_file("items/build-record/step0_items.0b8ee0b7c.log", f"{EXPA}/releases/step0_items.0b8ee0b7c.log")
    b.add_file("items/build-record/step0_items.sh", f"{EXPA}/releases/step0_items.sh")
    # A2 hard panel: the same builder with --omit-rule first_not_grep_findable (BUILD.md records every command)
    hard = Path(HARD)
    b.add_tree("items/hard", hard / "items-hard")
    for rel in walk_files(hard / "dispatch-hard"):
        if rel.startswith("graph_facts/"):
            b.add_same_as(f"dispatch/hard/{rel}", hard / "dispatch-hard" / rel, f"items/hard/{rel}")
        else:
            b.add_file(f"dispatch/hard/{rel}", hard / "dispatch-hard" / rel)
    b.add_file("items/build-record-hard/BUILD.md", hard / "BUILD.md")
    b.add_file("items/build-record-hard/FACTS.json", hard / "facts/FACTS.json")
    for rel in walk_files(hard / "logs"):
        b.add_file(f"items/build-record-hard/logs/{rel}", hard / "logs" / rel)
    b.add_tree("items/build-record-hard/release-1007ba4f7", hard / "release-1007ba4f7")
    b.add_file("findability/hard/FINDABILITY.json", hard / "findability-hard/FINDABILITY.json")
    b.add_file("findability/hard/FINDABILITY.md", hard / "findability-hard/FINDABILITY.md")
    fm = prod / "findability-main"
    b.add_file("findability/main/policies.jsonl", fm / "policies.jsonl")
    b.add_file("findability/main/policies.jsonl.manifest.json", fm / "policies.jsonl.manifest.json")
    b.add_file("findability/main/PLAN.main.default.json", fm / "PLAN.main.default.json")
    b.add_file("findability/main/FINDABILITY.json", fm / "out/FINDABILITY.json")
    b.add_file("findability/main/FINDABILITY.md", fm / "out/FINDABILITY.md")


def step_results(b: Build):
    stored_by_sha = {}
    for pkg, rec in b.files.items():
        if pkg.endswith("policies.jsonl"):
            stored_by_sha.setdefault(rec["sha256"], pkg)
    for run in b.runs:
        root = Path(EXPA) / run["analysis"]
        for rel in walk_files(root):
            pkg = f"results/{run['run']}/{rel}"
            if rel.endswith(".jsonl") and "policies" in rel:
                sha = sha256_file(root / rel)
                if sha in stored_by_sha:
                    b.add_same_as(pkg, root / rel, stored_by_sha[sha])
                    continue
                rec = b.add_file(pkg, root / rel)
                stored_by_sha[rec["sha256"]] = pkg
                continue
            b.add_file(pkg, root / rel)
        for name in ("COMPARE.json", "MODEL_COPY.sha256"):
            src = f"reports/expA-{run['run']}/{name}"
            if src in b.export["files"]:
                b.add_repo(src, f"results/{run['run']}/{name}")
    # main-run timeline outputs: the repo copies are published; the Core2 copy of the script is identical
    tl = Path(EXPA) / "main-analysis-20261101b-timeline"
    b.add_file("results/main-20261101b/timeline/expA_timeline.py", tl / "expA_timeline.py")
    for rel in ("RULES.md", "TIMELINE.json", "TIMELINE.md", "episodes.jsonl", "SHA256SUMS"):
        if sha256_file(tl / rel) != b.files[f"results/main-20261101b/timeline/{rel}"]["sha256"]:
            fail(f"timeline {rel}: the repo copy differs from Core2")


def step_runs(b: Build):
    for run in b.runs:
        root = Path(EXPA) / "runs" / run["run"]
        rels = walk_files(root)
        b.build_shard(run["run"], f"runs/{run['run']}", [(run["run"], root, rels)], "episodes", run["run"])
        b.add_file(f"runs/{run['run']}/run.json", root / "run.json")
        b.add_file(f"runs/{run['run']}/cells.jsonl", root / "cells.jsonl")
        if run["kind"] == "scores_only":
            # plain copies of the launch records: the follow-up analyses read the selection recorded in them
            # (scripts/expA_a4_analysis.py --run-json-*: launches/*.json next to run.json)
            launches = [rel for rel in rels if rel.startswith("launches/")]
            if not launches:
                fail(f"{run['run']}: no launch records")
            for rel in launches:
                b.add_file(f"runs/{run['run']}/{rel}", root / rel)
    for run in b.aborted:
        root = Path(EXPA) / "runs" / run["run"]
        rels = walk_files(root)
        b.build_shard(run["run"], f"runs/aborted/{run['run']}", [(run["run"], root, rels)], "launch-records",
                      run["run"])


def step_truth(b: Build):
    b.add_file("truth/STATIC_TRACE_SUMMARY.json", f"{EXPA}/prod-dd842ab6b/STATIC_TRACE_SUMMARY.json")
    b.add_file("truth/JEDI_DEFINITION_ERRORS.json", f"{EXPA}/prod/JEDI_DEFINITION_ERRORS.json")
    static = Path(EXPA) / "static"
    families = [f["family"] for f in b.families]
    rels = [r for r in walk_files(static) if r.split("/", 1)[0] in families]
    b.build_shard("static", "truth", [("static", static, rels)], "truth")
    traces = Path(EXPA) / "traces-dd842ab6b"
    rels = [r for r in walk_files(traces) if r.split("/", 1)[0] in families]
    b.build_shard("traces", "truth", [("traces", traces, rels)], "truth")
    work = Path(EXPA) / "registry/work"
    rels = [r for r in walk_files(work) if r.split("/", 1)[0] in families]
    b.build_shard("baseline", "truth", [("baseline", work, rels)], "truth")


def step_code(b: Build):
    for short in b.releases:
        b.add_tree(f"code/{short}", Path(EXPA) / "releases" / short)


def copyright_lines(data: bytes):
    out = []
    for line in data.decode("utf-8", "replace").splitlines():
        text = line.strip()
        if re.match(r"(?i)^(copyright\b|\(c\)|©)", text) or re.match(r"(?i)^.{0,20}copyright \(c\)", text):
            lowered = text.lower()
            if re.match(r"(?i)^\(c\)\s+you must", text):  # Apache-2.0 clause 4(c), not a copyright line
                continue
            if any(s in lowered for s in ("copyright notice", "copyright holder", "[yyyy]", "copyright owner",
                                          "copyright license")):
                continue
            out.append(text)
    return out


def step_licences(b: Build):
    clean = json.loads((b.stage / "registry/clean-checkouts.json").read_text(encoding="utf-8"))["families"]
    b.licence_files = {}
    for fam in b.families:
        name = fam["family"]
        recs = []
        for lic in fam["license_files"]:
            rec = b.add_file(f"licenses/{name}/{lic['path']}", f"{PUBLIC}/repos/{name}/{lic['path']}")
            if rec["sha256"] != lic["sha256"] or clean[name]["files"].get(lic["path"]) != lic["sha256"]:
                fail(f"licence file {name}/{lic['path']} does not match the registry or clean-checkouts.json")
            recs.append((lic, rec, copyright_lines((b.stage / rec["path"]).read_bytes())))
        b.licence_files[name] = recs
    # G2-lite: licence files of in-tree third-party paths that episodes mention
    b.extra_licences = collections.defaultdict(list)
    for key, count in sorted(b.licence_mentions.items()):
        family, prefix = key.split("|", 1)
        tree = clean[family]["files"]
        for rel in sorted(tree):
            if rel.startswith(prefix) and LICENCE_FILE_RE.search(rel):
                rec = b.add_file(f"licenses/{family}/extra/{rel}", f"{PUBLIC}/repos/{family}/{rel}")
                if rec["sha256"] != tree[rel]:
                    fail(f"extra licence {family}/{rel} does not match clean-checkouts.json")
                b.extra_licences[family].append((prefix, rel, count))
        if not any(p == prefix for p, _, _ in b.extra_licences[family]):
            b.extra_licences[family].append((prefix, None, count))
    rec = b.add_file("licenses/tools/locagent-4935b557/LICENSE", f"{LOCAGENT_ROOT}/LICENSE",
                     note=f"LocAgent at commit {LOCAGENT_COMMIT} (vendor checkout used for the static backend L)")
    if rec["sha256"] != LOCAGENT_LICENSE_SHA256:
        fail("LocAgent LICENSE sha256 differs from the audited value")
    rec = b.add_file("licenses/tools/jedi-0.20.1/LICENSE.txt", f"{JEDI_DIST}/LICENSE.txt",
                     note="jedi 0.20.1 wheel metadata in the static-backend environment")
    if rec["sha256"] != JEDI_LICENSE_SHA256:
        fail("jedi LICENSE.txt sha256 differs from the audited value")
    b.add_file("licenses/tools/jedi-0.20.1/AUTHORS.txt", f"{JEDI_DIST}/AUTHORS.txt")
    yapf_licence = Path(f"{PUBLIC}/repos/yapf/LICENSE").read_bytes()
    if sha256_bytes(yapf_licence) != APACHE_LICENSE_SHA256:
        fail("the Apache-2.0 text source does not have the audited sha256")
    b.add_bytes("LICENSE", yapf_licence, {"kind": "core2", "path": f"{PUBLIC}/repos/yapf/LICENSE",
                                          "note": "Apache License 2.0 full text; byte-identical to "
                                                  "https://www.apache.org/licenses/LICENSE-2.0.txt (sha256 "
                                                  f"{APACHE_LICENSE_SHA256}, LICENSE_AUDIT.md)"})


# ================================================================================================ documents
def models_used(b: Build):
    lock = json.loads((b.stage / "protocol/EXPA_PREREG.lock.json").read_text(encoding="utf-8"))
    out = {"google/gemma-4-31B-it": lock["model"]["revision"]}
    for run in b.runs:
        model = json.loads((b.stage / f"runs/{run['run']}/run.json").read_text(encoding="utf-8"))["model"]
        if model.get("hub_id") and model.get("revision"):
            out[model["hub_id"]] = model["revision"]
    return out


def write_notice(b: Build):
    holder = b.args.copyright_holder or PLACEHOLDER.format("copyright holder")
    models = models_used(b)
    model_text = " and ".join(f"{hub} (revision {rev})" for hub, rev in sorted(models.items()))
    text = (f"Absent Edges Exp A resource, version 1\n"
            f"Copyright 2026 {holder}\n\n"
            "This product is licensed under the Apache License, Version 2.0 (see LICENSE), except for third-party\n"
            "material, which keeps its own licence: see THIRD_PARTY_NOTICES.md and licenses/.\n\n"
            "This distribution contains excerpts of third-party open-source software, each under its own licence.\n"
            "Excerpts were extracted, and in some cases truncated, by this project.\n"
            f"Episode records contain text generated by {model_text},\n"
            "models licensed under the Apache License 2.0.\n")
    b.add_bytes("NOTICE", text.encode("utf-8"), {"kind": "generated", "what": "project NOTICE"}, generated=True)


def write_third_party(b: Build):
    audit = json.loads(b.repo_bytes("staged/paper-prep-20261101/release-plan/license_audit.json"))
    by_family = {f["family"]: f for f in audit["families"]}
    models = models_used(b)
    generated = sorted({family for family, _ in getattr(b, "generated_files", [])})
    L = ["# Third-party notices", "", f"> {LABEL}", "",
         "This distribution is licensed under the Apache License 2.0 (`LICENSE`), except for third-party material, "
         "which remains under its own licence:", "",
         "- excerpts of the upstream projects listed below. They appear in `items/*/payloads/`, `items/*/facts/`, "
         "`items/*/graph_facts/`, `dispatch/*/payloads/` (and the deduplicated `dispatch/*/graph_facts/`), "
         "`truth/static.tar.xz` (`locagent.document.json` embeds node source), and in the prompts and tool outputs "
         "inside `runs/*/*.tar.xz`;"]
    if generated:
        L.append(f"- the {len(b.generated_files)} version files in `registry/generated-files/<family>/` "
                 f"({', '.join(generated)}). The projects' build backend (vcs-versioning) generated them when the "
                 "model-visible trees were built, and they are part of those trees (`registry/clean-checkouts.json`); "
                 "each falls under its project's licence;")
    L += ["- identifiers, paths, line numbers and test node ids of those projects in `items/`, `truth/` and `results/`, "
          "and in `registry/`: the file lists and sha256 of every model-visible tree (`clean-checkouts.json`), the "
          "mirror environment hash lists (`registry/mirror/`) and the environment lock files (`registry/locks/`, "
          "package names and versions). These are facts about the code; the licence files are shipped regardless;",
         "- text generated by " + ", ".join(f"`{hub}` (revision `{rev}`)" for hub, rev in sorted(models.items()))
         + " inside `runs/`; the models are licensed under Apache-2.0. Where such text reproduces upstream code, the "
           "upstream licence applies to that code;",
         "- the Gemma chat template `registry/models/gemma4-31b-bf16-518276fb/chat_template.jinja`, a file of the "
         "`google/gemma-4-31B-it` model repository (Apache-2.0).", "",
         "Excerpts were extracted from the files named in each record, and some were truncated by this project (a cut "
         "shows as `snippet_end_line < end_line`). This is the notice of changes required by Apache-2.0 section 4(b) "
         "for the Apache-licensed projects (sqlite-utils, yapf, and packaging's Apache option). Line numbers, paths "
         "and identifiers refer to the pinned commits below. No upstream project endorses this distribution; for the "
         "BSD-3-Clause projects, their names are used only for attribution.", "",
         "Each licence file is copied byte for byte from the model-visible checkout at the pinned commit into "
         "`licenses/<family>/`; its sha256 equals the value in `registry/registry.v1.json` and in "
         "`registry/clean-checkouts.json`. No tree contains a NOTICE file, so Apache-2.0 section 4(d) does not apply.",
         "", "## Upstream projects", ""]
    for fam in b.families:
        name = fam["family"]
        meta = by_family.get(name, {})
        L.append(f"### {name} ({fam['role']})")
        L.append(f"- Repository: {fam['url']} at tag `{fam['tag']}`, commit `{fam['commit']}` "
                 f"({fam['commit_date']})")
        L.append(f"- Licence: {fam['license']}")
        for lic, rec, lines in b.licence_files[name]:
            L.append(f"  - `licenses/{name}/{lic['path']}` (sha256 `{rec['sha256']}`)")
            for line in lines or ["(no copyright line in this file)"]:
                L.append(f"    - {line}")
        if name == "packaging":
            L.append("  - Election: this distribution uses the BSD-2-Clause option; both licence texts are included.")
        if name == "markdown-it-py":
            L.append("  - Both notices apply: the Python port (`LICENSE`) and the JavaScript original "
                     "(`LICENSE.markdown-it`).")
        L.append(f"- Package scope: {', '.join('`' + d + '`' for d in fam['package_dirs'])}; model-visible tree "
                 f"sha256 `{meta.get('tree_sha256', '')}`")
        for prefix, rel, count in b.extra_licences.get(name, []):
            if rel:
                L.append(f"- Episode tool outputs mention `{prefix}` ({count} events files); its licence file is "
                         f"included as `licenses/{name}/extra/{rel}`.")
            else:
                L.append(f"- Episode tool outputs mention `{prefix}` ({count} events files); that path has no "
                         "separate licence file in the tree (any licence header is inside the quoted files).")
        L.append("")
    L += ["## Tools whose outputs are included (tools not redistributed)",
          f"- LocAgent, https://github.com/gersteinlab/LocAgent at `{LOCAGENT_COMMIT}`, Apache-2.0: "
          f"`licenses/tools/locagent-4935b557/LICENSE` (sha256 `{LOCAGENT_LICENSE_SHA256}`; no NOTICE file at that "
          "commit). Its graph outputs are in `truth/static.tar.xz` (`locagent.*`) and in every rendered answer.",
          f"- jedi 0.20.1, MIT: `licenses/tools/jedi-0.20.1/LICENSE.txt` (sha256 `{JEDI_LICENSE_SHA256}`) and "
          "`AUTHORS.txt`, from the wheel metadata. Its reference outputs are in `truth/static.tar.xz` (`jedi.*`).", "",
          "## Models (weights not redistributed)"]
    for hub, rev in sorted(models.items()):
        L.append(f"- `{hub}`, revision `{rev}`, licence Apache-2.0 (model card). File hashes: `registry/models/`.")
    L += ["", "## Not included", "- Model weights, upstream source trees (see `registry/clean-checkouts.json` to "
          "rebuild them), Kaggle competition data (none was used).", ""]
    b.add_bytes("THIRD_PARTY_NOTICES.md", "\n".join(L).encode("utf-8"),
                {"kind": "generated", "what": "from registry/registry.v1.json, the shipped licence files and "
                 "license_audit.json (release-plan, 2026-11-01)"}, generated=True)


def write_sanitization(b: Build):
    b.listed_own = [dict(o) for o in b.own]  # exactly what this file lists; the final leak check compares with it
    L = ["# Sanitization record", "", f"> {LABEL}", "",
         "Policy (`RELEASE_PACKAGE_PLAN.md` section 5, option B): frozen files are published byte for byte. Only "
         "personal data is changed. Host names, GPU UUIDs, loopback endpoints and scratch paths are infrastructure "
         "identifiers and are kept. Account names are matched case-insensitively from a private pattern list that is "
         "not part of this package, so no personal name is printed here.", "",
         "## Changed files", "",
         f"A colleague's account name (class `OTHER_USER`) is replaced by `{REDACTION.decode()}`, the same token as in "
         "the companion code repository, so a file published in both has the same bytes there and here. Each changed "
         "file no longer matches the pins of its original bytes; the original sha256 is given so that the pins can "
         "still be checked against it (`reproduce/verify.py` does this).", "",
         "| File | Lines | Replacements | Original sha256 | New sha256 | Pins of the original bytes |",
         "|---|---|---|---|---|---|"]
    for rec in sorted(b.sanitized, key=lambda r: r["path"]):
        pins = rec.get("pins") or []
        L.append(f"| `{rec['path']}` | {', '.join(map(str, rec['lines']))} | {rec['count']} | `{rec['original_sha256']}` "
                 f"| `{rec['new_sha256']}` | {'; '.join(pins) if pins else '-'} |")
    if not b.sanitized:
        L.append("| (none) | | | | | |")
    L += ["", "## Kept: the author's own account names and name", "",
          "Classes: `OWN_ACCOUNT` (the author's host account on the research machines), `OWN_WINDOWS_ACCOUNT` (the "
          "author's local workstation account, inside recorded interpreter paths), `OWN_IDENTITY` (the author's "
          "name or e-mail). They are kept where they occur: inside frozen records, because the pins need the "
          "original bytes, and in the author attribution (`NOTICE`, `README.md`), by the author's choice. Every "
          "occurrence is listed below (file, or shard and member, and line numbers).", ""]
    counts = collections.Counter(o["class"] for o in b.own)
    total = collections.Counter()
    for o in b.own:
        total[o["class"]] += len(o["lines"])
    L.append("| Class | Files or members | Occurrences |")
    L.append("|---|---|---|")
    for cls in ("own_account", "own_windows_account", "own_identity"):
        L.append(f"| `{cls.upper()}` | {counts.get(cls, 0)} | {total.get(cls, 0)} |")
    L.append("")
    for o in sorted(b.own, key=lambda o: (o["class"], o["file"], o["member"] or "")):
        where = f"`{o['file']}`" + (f" member `{o['member']}`" if o["member"] else "")
        L.append(f"- `{o['class'].upper()}` {where}: line(s) {', '.join(map(str, o['lines']))}")
    L += ["", "## Leak check", "",
          "The build searches every published file and every shard member, case-insensitively, for fragments of the "
          "names in the private pattern list. It fails if a fragment lies outside an occurrence listed above, if a "
          "colleague's account name remains anywhere, or if the occurrences in a published file differ from this "
          "list.", "",
          "## Not changed", "",
          "- Upstream source text (snippets, licence files, the agent's tool outputs) keeps every author name and "
          "e-mail address it contains; that is public upstream content.",
          "- Credential-like parameter names and IPv4 literals inside upstream code are upstream text, not project "
          "secrets.", ""]
    b.add_bytes("SANITIZATION.md", "\n".join(L).encode("utf-8"),
                {"kind": "generated", "what": "sanitization and identifier record of this build"}, generated=True)


# ------------------------------------------------------------------------------------------------ claims
def _load(b: Build, rel: str):
    cache = b.__dict__.setdefault("_json_cache", {})
    if rel not in cache:
        cache[rel] = json.loads((b.stage / rel).read_text(encoding="utf-8"))
    return cache[rel]


def _fmt(value, fmt):
    if fmt is None:
        return str(value)
    if fmt == "pp":
        return f"{value * 100:.1f}"
    if fmt == "+pp":
        return f"{value * 100:+.1f}"
    if fmt == "p":
        return f"{value:.1e}" if value < 0.001 else f"{value:.3f}"
    if fmt == "list":
        items = [str(v) for v in value]
        return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]
    return format(value, fmt)


class Claims:
    def __init__(self, b: Build):
        self.b = b
        self.rows = {}

    def add(self, key, rel, path, fmt=None, op=None, about=""):
        doc = _load(self.b, rel)
        if op is None:
            value = resolve_one(doc, path)
        elif op == "sum":
            value = sum(resolve_all(doc, path))
        elif op == "count":
            value = len(resolve_all(doc, path))
        else:
            raise ValueError(op)
        self.rows[key] = {"value": value, "text": _fmt(value, fmt), "file": rel,
                          "path": path + (f" ({op})" if op else ""), "about": about}
        return value

    def derived(self, key, value, rel, how, fmt=None):
        self.rows[key] = {"value": value, "text": _fmt(value, fmt), "file": rel, "path": how, "about": ""}
        return value

    def text(self, key):
        if key not in self.rows:
            fail(f"README template uses unknown claim {key}")
        return self.rows[key]["text"]


def build_claims(b: Build) -> Claims:
    c = Claims(b)
    IM, ID = "items/main/manifest.json", "items/dev/manifest.json"
    F, DM, R = "findability/main/FINDABILITY.json", "dispatch/main/dispatch_manifest.json", "registry/registry.v1.json"
    for split, m in (("main", IM), ("dev", ID)):
        c.add(f"items.{split}", m, "counts.items", ",d")
        c.add(f"families.{split}", m, "counts.families")
        c.add(f"functions.{split}", m, "families.*.functions", ",d", "sum")
        c.add(f"L.{split}", m, "families.*.record_counts.L.kept", ",d", "sum")
        c.add(f"J.{split}", m, "families.*.record_counts.J.kept", ",d", "sum")
        c.add(f"trace.kept.{split}", m, "families.*.record_counts.trace.kept", ",d", "sum")
        c.add(f"trace.records.{split}", m, "families.*.record_counts.trace.records", ",d", "sum")
        c.add(f"payload_rows.{split}", m, "families.*.outputs.payloads.count", ",d", "sum")
        facts = sum(len((b.stage / f"items/{split}/facts/{p.name}").read_bytes().splitlines())
                    for p in (b.stage / f"items/{split}/facts").iterdir())
        c.derived(f"facts_rows.{split}", facts, f"items/{split}/facts/*.jsonl", "number of lines", ",d")
    c.add("params.m", IM, "parameters.m_per_family")
    c.add("params.c4a.max_bytes", IM, "parameters.c4a.max_bytes", ",d")
    c.add("params.c4a.limit", IM, "parameters.c4a.search_limit")
    c.add("params.snippet", IM, "parameters.snippet_max_bytes", ",d")
    c.add("params.max_output", IM, "parameters.max_output_bytes", ",d")
    c.add("params.v.lo", IM, "parameters.verified_bounds[0]")
    c.add("params.v.hi", IM, "parameters.verified_bounds[1]")
    c.add("reg.cap", R, "rules.baseline.wall_cap_seconds", ",d")
    c.add("reg.min_passed", R, "rules.baseline.min_passed")
    c.add("reg.min_fraction", R, "rules.baseline.min_pass_fraction")
    c.add("reg.cutoff", R, "cutoff")
    c.add("design.seeds.P", "runs/main-20261101b/run.json", "design.blocks.P.seeds", "list")
    c.add("main.temperature", "runs/main-20261101b/run.json", "episode_config.temperature")
    c.add("eligible.main", IM, "counts.eligible_total", ",d")
    c.add("V.callers", IM, "counts.V_callers", ",d")
    c.add("V.sites", IM, "counts.V_sites", ",d")
    c.add("V.rw", IM, "counts.V_runtime_witnessed", ",d")
    for rule, n in _load(b, IM)["counts"]["funnel_total"].items():
        c.add(f"excl.{rule}", IM, f"counts.funnel_total.{rule}", ",d")
    for k in _load(b, IM)["counts"]["V_count_histogram"]:
        c.add(f"vhist.{k}", IM, f'counts.V_count_histogram["{k}"]', ",d")
    for k in ("in_s_nat", "in_s_edge", "in_t"):
        c.add(f"membership.{k}", IM, f"counts.membership.{k}")
    for k in ("grep_recovers_omitted", "c4_hits_recovers_omitted", "c4_trigger_fires", "natural_miss",
              "omitted_name_in_answer"):
        c.add(f"find.{k}.count", F, f"findability.{k}.count")
        c.add(f"find.{k}.fw", F, f"findability.{k}.family_weighted", ".3f")
    for k in ("direct", "dynamic", "alias", "callback"):
        c.add(f"strata.{k}", F, f"findability.strata.{k}")
    for k in ("A", "B"):
        c.add(f"wording.{k}", F, f"findability.wording_pairs.{k}")
    kinds = collections.Counter()
    for line in (b.stage / "items/main/items.jsonl").read_bytes().splitlines():
        item = json.loads(line)
        ident = item["omitted"]["identity"]
        for caller in item["V"]:
            if caller["identity"] == ident:
                kinds.update(site["kind"] for site in caller["sites"])
    for k in ("call", "dynamic", "reference"):
        c.derived(f"cstar_kind.{k}", kinds.get(k, 0), "items/main/items.jsonl",
                  f"sites of the omitted caller (omitted.identity in V) with kind {k!r}", ",d")
    c.add("gf.nodes.main", DM, "graph_facts.*.nodes", ",d", "sum")
    c.add("gf.edges.main", DM, "graph_facts.*.edges", ",d", "sum")
    c.add("policies.rows.main", "findability/main/policies.jsonl.manifest.json", "out.rows", ",d")
    reg = _load(b, R)
    for role in ("main", "dev"):
        c.derived(f"reg.n.{role}", sum(1 for f in reg["families"] if f.get("role") == role), R,
                  f'families[role="{role}"] (count)')
    c.derived("reg.ok.built", sum(1 for f in reg["families"] if f.get("status") == "ok"), R,
              'families[status="ok"] (count)')
    c.derived("reg.passed.main", sum(f["baseline"]["counts"]["passed"] for f in reg["families"] if f.get("role") == "main"),
              R, 'families[role="main"].baseline.counts.passed (sum)', ",d")
    c.derived("reg.collected.main", sum(f["baseline"]["counts"]["collected"] for f in reg["families"]
                                        if f.get("role") == "main"), R,
              'families[role="main"].baseline.counts.collected (sum)', ",d")
    lic = collections.Counter(f["license"] for f in reg["families"] if f.get("role") == "main")
    for name, n in lic.items():
        c.derived(f"lic.main.{name}", n, R, f'families[role="main",license="{name}"] (count)')
    c.add("jedi.errors", "truth/JEDI_DEFINITION_ERRORS.json", "totals.failed_definitions", ",d")
    c.add("jedi.families", "truth/JEDI_DEFINITION_ERRORS.json", "totals.families_with_failed_j_queries")
    # per run
    b.run_rows = []
    for run in b.runs:
        rid = run["run"]
        A = f"results/{rid}/analysis/ANALYSIS.json"
        RJ = f"runs/{rid}/run.json"
        if run["kind"] == "scores_only":
            SM = f"results/{rid}/scores.jsonl.manifest.json"
            c.add(f"{rid}.design_cells", SM, "binding.design_cells", ",d")
            c.add(f"{rid}.completed", SM, "binding.cell_states.completed", ",d")
            exits = _load(b, SM)["binding"]["status_counts"]
            for k in exits:
                c.add(f"{rid}.exit.{k}", SM, f"binding.status_counts.{k}", ",d")
            b.run_rows.append((run, exits))
            continue
        c.add(f"{rid}.design_cells", A, "completion.design_cells", ",d")
        c.add(f"{rid}.completed", A, "completion.cell_states.completed", ",d")
        c.add(f"{rid}.not_dispatched", A, "completion.cell_states.not_dispatched", ",d")
        c.add(f"{rid}.missing_infra", A, "completion.cell_states.missing_infrastructure", ",d")
        c.add(f"{rid}.max_calls", RJ, "episode_config.max_model_calls")
        c.add(f"{rid}.context", RJ, "episode_config.context_budget", ",d")
        c.add(f"{rid}.families", A, 'primary.hypotheses[id="H3"].n_families')
        for h in ("H1", "H2", "H3"):
            base = f'primary.hypotheses[id="{h}"]'
            c.add(f"{rid}.{h}.est", A, f"{base}.estimate", "+.3f")
            c.add(f"{rid}.{h}.lo", A, f"{base}.t_test.interval.lower", "+.3f")
            c.add(f"{rid}.{h}.hi", A, f"{base}.t_test.interval.upper", "+.3f")
            c.add(f"{rid}.{h}.holm", A, f"{base}.holm_adjusted_p", "p")
            c.add(f"{rid}.{h}.status", A, f"{base}.status")
        for cond in ("C0", "C1", "C2", "C3", "C4a"):
            base = f'secondary.rates[block="P",condition="{cond}"]'
            c.add(f"{rid}.rate.{cond}", A, f"{base}.site_recovered.estimate", ".3f")
            c.add(f"{rid}.sub.{cond}", A, f"{base}.exit_counts.Submitted", ",d")
            c.add(f"{rid}.cells.{cond}", A, f"{base}.completed_cells", ",d")
        sm = _load(b, f"results/{rid}/scores.jsonl.manifest.json")
        exits = sm["binding"]["status_counts"]
        for k in exits:
            c.add(f"{rid}.exit.{k}", f"results/{rid}/scores.jsonl.manifest.json", f"binding.status_counts.{k}", ",d")
        b.run_rows.append((run, exits))
        cmp_ = run.get("compare")
        if cmp_:
            CF = f"results/{rid}/COMPARE.json"
            field = "delta" if cmp_["mode"] == "model" else "did"
            if cmp_["mode"] == "budget":
                c.add(f"{rid}.seed", CF, "seed")
            for h in ("H1", "H2", "H3"):
                c.add(f"{rid}.{field}.{h}.est", CF, f"{field}.{h}.estimate", "+.3f")
                c.add(f"{rid}.{field}.{h}.lo", CF, f"{field}.{h}.ci_low", "+.3f")
                c.add(f"{rid}.{field}.{h}.hi", CF, f"{field}.{h}.ci_high", "+.3f")
                if cmp_["mode"] == "model":
                    c.add(f"{rid}.class.{h}", CF, f"classification.{h}")
    # follow-up A3 (pre-specified primary M = G_off - G_on), from the shipped analysis output
    A3 = "results/followups/a3ledger-20261105/A3_ANALYSIS.json"
    if A3 in b.files:
        base = "main.M_reduction_in_omission_cost"
        c.add("a3.M.est", A3, f"{base}.estimate", "+pp")
        c.add("a3.M.lo", A3, f"{base}.ci_low", "+pp")
        c.add("a3.M.hi", A3, f"{base}.ci_high", "+pp")
        c.add("a3.M.p", A3, f"{base}.p_two_sided", "p")
        c.add("a3.M.families", A3, f"{base}.n_families")
        c.add("a3.G_off.est", A3, "main.G_off_omission_cost_without_ledger.estimate", "+pp")
    c.add("g0.verdict", "protocol/EXPA_REPL12B.lock.json", "g0.gate.verdict")
    if any(r["run"].startswith("qat31b-g0-") for r in b.runs):
        c.add("qat.g0.verdict", "protocol/EXPA_QAT31B.lock.json", "g0.gate.verdict")
    lock = "protocol/EXPA_PREREG.lock.json"
    c.add("lock.revision", lock, "model.revision")
    c.add("lock.prereg", lock, "prereg.sha256")
    c.add("lock.items", lock, "items_main.items_jsonl_sha256")
    c.add("lock.findability", lock, 'findability_main["FINDABILITY.json"]')
    c.add("lock.plan", lock, "analysis_plan_main_sha256")
    c.add("lock.release_files", lock, "code.release_files_sha256")
    c.add("repl12b.revision", "protocol/EXPA_REPL12B.lock.json", "model.revision")
    return c


def sizes_table(b: Build):
    groups = collections.OrderedDict()
    order = [("A", "protocol/"), ("A", "registry/"), ("A", "items/"), ("A", "dispatch/"), ("A", "findability/"),
             ("A", "results/"), ("A", "exposure-audit/"), ("A", "code/"), ("A", "licenses/"), ("A", "reproduce/"),
             ("B", "runs/"), ("C", "truth/")]
    for tier, prefix in order:
        groups[(tier, prefix)] = [0, 0]
    groups[("A", "top-level files")] = [0, 0]
    for path, rec in b.files.items():
        for tier, prefix in order:
            if path.startswith(prefix):
                groups[(tier, prefix)][0] += 1
                groups[(tier, prefix)][1] += rec["bytes"]
                break
        else:
            groups[("A", "top-level files")][0] += 1
            groups[("A", "top-level files")][1] += rec["bytes"]
    tiers = collections.Counter()
    for (tier, _), (_, size) in groups.items():
        tiers[tier] += size
    return groups, tiers


def render_readme(b: Build, claims: Claims):
    template = b.repo_bytes("staged/paper-prep-20261101/release-plan/package-v1/README.tmpl.md").decode("utf-8")
    holder = b.args.copyright_holder or PLACEHOLDER.format("copyright holder")
    variables = {
        "copyright_holder": holder,
        "code_url": b.args.code_url or PLACEHOLDER.format("code repository URL"),
        "writeup_url": b.args.writeup_url or PLACEHOLDER.format("Writeup URL"),
        "contact": b.args.contact or PLACEHOLDER.format("contact"),
        "funding": b.args.funding or PLACEHOLDER.format("funding and compute statement"),
        "build_date": b.stamp[:4] + "-" + b.stamp[4:6] + "-" + b.stamp[6:8],
        "export_head": b.export["head"],
        "n_runs": str(len(b.runs)),
        "n_releases": str(len(b.releases)),
    }
    models = models_used(b)
    variables["model_names"] = ", ".join(f"`{hub}`" for hub in sorted(models))
    variables["models_row"] = "; ".join(f"`{hub}` revision `{rev}`" for hub, rev in sorted(models.items()))
    has_qat = any(r["run"].startswith("qat31b-") and "-g0-" not in r["run"] for r in b.runs)
    qat_g0 = [r["run"] for r in b.runs if r["run"].startswith("qat31b-g0-")]
    note = ""
    if qat_g0:
        note = (f"The QAT addendum's go/no-go gate on the pilot libraries (`{qat_g0[0]}`) recorded the verdict "
                f"{claims.text('qat.g0.verdict')} in `protocol/EXPA_QAT31B.lock.json` before the first QAT "
                "main-family cell.")
    if not has_qat:
        note += (" " if note else "") + (
            "The run on the main-track checkpoint (`gemma-4-31b-it-qat-w4a16-ct`, 30 families) is designed in "
            "`protocol/EXPA_QAT31B.md`; its records are not part of this build.")
    variables["qat_note"] = note
    if b.skipped:
        variables["qat_note"] += " Runs present on the research machine but without a frozen analysis (or, for the " \
                                 "follow-up runs, frozen scores) at build time, and the dev-check runs, are not " \
                                 "included: " + ", ".join(f"`{s.split(':')[0]}`" for s in b.skipped) + "."
    groups, tiers = sizes_table(b)
    tables = {}
    L = ["| Tier | Directory | Files | Bytes |", "|---|---|---|---|"]
    for (tier, prefix), (n, size) in groups.items():
        if n:
            L.append(f"| {tier} | `{prefix}` | {n:,} | {size:,} |")
    L.append(f"| | **Tier A / B / C** | | {tiers['A']:,} / {tiers['B']:,} / {tiers['C']:,} |")
    L.append("")
    L.append("Sizes measured at build time, before `README.md`, `PROVENANCE.json` and `MANIFEST.sha256` were written "
             "(those three are a few MB). Shards are counted compressed; their uncompressed sizes are in "
             "`PROVENANCE.json` (`shard.bytes_uncompressed`).")
    tables["sizes"] = "\n".join(L)
    L = ["| Run | What | Status | Design cells | Completed | Exits | Episode records |", "|---|---|---|---|---|---|---|"]
    for run, exits in b.run_rows:
        rid = run["run"]
        ex = ", ".join(f"{k} {claims.text(f'{rid}.exit.{k}')}" for k in sorted(exits))
        L.append(f"| `{rid}` | {run['title']} | {run['status']} | {claims.text(f'{rid}.design_cells')} | "
                 f"{claims.text(f'{rid}.completed')} | {ex} | `runs/{rid}/{rid}.tar.xz` |")
    for run in b.aborted:
        L.append(f"| `{run['run']}` | launch records only | {run['note']} | - | 0 | - | "
                 f"`runs/aborted/{run['run']}/{run['run']}.tar.xz` |")
    tables["runs"] = "\n".join(L)
    L = ["| Run | H1 = C3 - C2 | H2 = C4a - C1 | H3 = C1 - C0 |", "|---|---|---|---|"]
    for run in b.runs:
        if run["kind"] != "analysis":
            continue
        rid = run["run"]
        cells = []
        for h in ("H1", "H2", "H3"):
            cells.append(f"{claims.text(f'{rid}.{h}.est')} [{claims.text(f'{rid}.{h}.lo')}, "
                         f"{claims.text(f'{rid}.{h}.hi')}], Holm p {claims.text(f'{rid}.{h}.holm')} "
                         f"({claims.text(f'{rid}.{h}.status')})")
        L.append(f"| `{rid}` ({claims.text(f'{rid}.families')} families) | " + " | ".join(cells) + " |")
    tables["hypotheses"] = "\n".join(L)
    L = ["| Run | C0 | C1 | C2 | C3 | C4a |", "|---|---|---|---|---|---|"]
    for run in b.runs:
        if run["kind"] != "analysis":
            continue
        rid = run["run"]
        L.append(f"| `{rid}` | " + " | ".join(
            f"{claims.text(f'{rid}.rate.{c}')} ({claims.text(f'{rid}.sub.{c}')}/{claims.text(f'{rid}.cells.{c}')})"
            for c in ("C0", "C1", "C2", "C3", "C4a")) + " |")
    tables["rates"] = "\n".join(L)
    L = ["| Comparison | H1 | H2 | H3 |", "|---|---|---|---|"]
    for run in b.runs:
        cmp_ = run.get("compare")
        if not cmp_:
            continue
        rid = run["run"]
        field = "delta" if cmp_["mode"] == "model" else "did"
        label = (f"`{rid}` minus `{cmp_['ref']}` (paired by family)" if cmp_["mode"] == "model" else
                 f"`{rid}` minus `{cmp_['ref']}` at seed {claims.text(f'{rid}.seed')} (difference in differences)")
        cells = []
        for h in ("H1", "H2", "H3"):
            cell = (f"{claims.text(f'{rid}.{field}.{h}.est')} [{claims.text(f'{rid}.{field}.{h}.lo')}, "
                    f"{claims.text(f'{rid}.{field}.{h}.hi')}]")
            if cmp_["mode"] == "model":
                cell += f", {claims.text(f'{rid}.class.{h}')}"
            cells.append(cell)
        L.append(f"| {label}; `results/{rid}/COMPARE.json` | " + " | ".join(cells) + " |")
    tables["compare"] = "\n".join(L)
    L = ["| Follow-up | Protocol | Runs | Outputs | Script | Re-run by `verify.py` |", "|---|---|---|---|---|---|"]
    for spec in b.followups:
        runs = ", ".join(f"`{rid}`" for rid in spec["runs"].values())
        outputs = ", ".join(f"`{path}`" for path in spec["outputs"].values()) if spec["shipped"] else "not yet"
        if spec.get("verify"):
            how = ("yes, byte for byte" if spec["verify"]["inputs_package_relative"] else
                   "yes, field by field except the recorded input paths")
        else:
            how = f"no: {spec['not_verified']}"
        L.append(f"| {spec['name']} | `{spec['protocol']}` | {runs} | {outputs} | `{spec['script']}` | {how} |")
    for spec in FOLLOWUP_EXTRAS:
        if all(path in b.files for path in spec["outputs"]):
            L.append(f"| {spec['name']} | `{spec['protocol']}` | - | "
                     + ", ".join(f"`{path}`" for path in spec["outputs"]) + f" | `{spec['script']}` | no: {spec['why']} |")
    tables["followups"] = "\n".join(L)
    rules = sorted(_load(b, "items/main/manifest.json")["counts"]["funnel_total"].items(), key=lambda kv: (-kv[1], kv[0]))
    L = ["| Rule (first failing) | Excluded |", "|---|---|"]
    zero = [k for k, v in rules if v == 0]
    for k, v in rules:
        if v:
            L.append(f"| `{k}` | {claims.text(f'excl.{k}')} |")
    if zero:
        L.append(f"| {', '.join('`' + z + '`' for z in zero)} | 0 each |")
    tables["exclusions"] = "\n".join(L)
    hist = _load(b, "items/main/manifest.json")["counts"]["V_count_histogram"]
    tables["vhist"] = "; ".join(f"{k}: {claims.text(f'vhist.{k}')}" for k in sorted(hist, key=int))
    L = ["| Family | Role | Licence | Repository @ tag |", "|---|---|---|---|"]
    for fam in b.families:
        L.append(f"| {fam['family']} | {fam['role']} | {fam['license']} | {fam['repo']} @ {fam['tag']} |")
    tables["families"] = "\n".join(L)

    def substitute(match):
        token = match.group(1).strip()
        if token.startswith("table:"):
            return tables[token[6:]]
        if token.startswith("var:"):
            return variables[token[4:]]
        return claims.text(token)

    text = re.sub(r"\{\{([^{}]+)\}\}", substitute, template)
    L = ["| Key | Value | File | JSON path or derivation |", "|---|---|---|---|"]
    for key in sorted(claims.rows):
        row = claims.rows[key]
        L.append(f"| `{key}` | {row['text']} | `{row['file']}` | `{row['path']}` |")
    text = text.replace("<!--NUMBER-SOURCES-->", "\n".join(L))
    unfilled = sorted(set(re.findall("⟦[^⟧]*⟧", text)))
    b.unfilled = unfilled
    b.add_bytes("README.md", text.encode("utf-8"), {"kind": "generated", "what": "dataset card from "
                "staged/paper-prep-20261101/release-plan/package-v1/README.tmpl.md; numbers from frozen files (see "
                "its 'Number sources' table)"}, generated=True)


# ------------------------------------------------------------------------------------------------ verify plan
def _input_records(value, out):
    """(path, sha256) of every input record in a follow-up output's "inputs" tree."""
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            out.append((value["path"], value["sha256"]))
        for key in sorted(value):
            _input_records(value[key], out)
    elif isinstance(value, list):
        for item in value:
            _input_records(item, out)
    return out


def followup_argv(spec, frozen, b: Build):
    """The command line (package-relative paths, without --out) that reproduces the shipped output, derived from the
    output's own inputs record; or (None, reason)."""
    inputs, runs = frozen.get("inputs") or {}, spec["runs"]
    by_id = {r["run"]: r for r in b.runs}

    def scores(arm):
        return f"results/{runs[arm]}/scores.jsonl"

    extra = {}
    if spec["id"] == "a1":
        if any(k.startswith("timeline_") for k in inputs):
            return None, "the shipped output was computed from timeline files that are not in the package", extra
        argv = ["finalturn"]
        for arm in ("neutral", "deadline", "none"):
            argv += [f"--scores-{arm}", scores(arm)]
        if any(k.startswith("run_json_") for k in inputs):
            base = list(argv)
            for arm in ("neutral", "deadline", "none"):
                argv += [f"--run-dir-{arm}", f"runs/{runs[arm]}"]
            argv += ["--items", f"items/{by_id[runs['neutral']]['items']}/items.jsonl"]
            # the mechanism subgroup reads the episode records of the three runs (streamed from their shards)
            extra["timeline_shards"] = {f"runs/{rid}": f"runs/{rid}/{rid}.tar.xz" for rid in runs.values()}
            extra["without_shards"] = {
                "argv": base,
                "ignore": ["mechanism_subgroup", "secondary.S4_final_response_kind", "inputs.items"]
                          + [f"inputs.run_json_{arm}" for arm in ("neutral", "deadline", "none")],
                "why": "--skip-shards: the mechanism subgroup and S4_final_response_kind need the episode records, "
                       "so they and their inputs (run.json, items) are not compared"}
        return argv, None, extra
    if spec["id"] == "a2":
        argv = ["hard", "--scores", scores("hard"), "--main-scores", scores("main"),
                "--items-hard", f"items/{by_id[runs['hard']]['items']}/items.jsonl"]
        if "items_main" in inputs:
            argv += ["--items-main", f"items/{by_id[runs['main']]['items']}/items.jsonl"]
        if "frozen_analysis" in inputs:
            argv += ["--frozen-analysis", f"results/{runs['hard']}/analysis/ANALYSIS.json"]
        return argv, None, extra
    if spec["id"] in ("a3", "a5"):
        if "hard_panel" in inputs:
            return None, "the shipped output includes hard-panel runs that are not in the package", extra
        arms = ("off", "on") if spec["id"] == "a3" else ("neutral", "ledger")
        argv = []
        for arm in arms:
            argv += [f"--{arm}", scores(arm)]
        if "main_run_json" in inputs:
            for arm in arms:
                argv += [f"--{arm}-run-json", f"runs/{runs[arm]}/run.json"]
        return argv, None, extra
    if spec["id"] == "a4":
        argv = []
        for arm in ("neutral", "info", "submit", "deadline"):
            argv += [f"--scores-{arm}", scores(arm)]
        if any(k.startswith("run_json_") for k in inputs):
            for arm in ("neutral", "info", "submit", "deadline"):
                argv += [f"--run-json-{arm}", f"runs/{runs[arm]}/run.json"]
        argv += ["--seed", str(frozen["analysis"]["seed"])]
        return argv, None, extra
    if spec["id"] == "a6":
        argv = ["--scores-16", scores("16"), "--scores-8", scores("8")]
        if "run_json" in inputs:
            argv += ["--run-json-16", f"runs/{runs['16']}/run.json", "--run-json-8", f"runs/{runs['8']}/run.json"]
        return argv, None, extra
    if spec["id"] == "a7":
        argv = ["--scores-impact", scores("impact"), "--scores-original", scores("original")]
        if any(k.startswith("run_json") for k in inputs):
            argv += ["--run-json-impact", f"runs/{runs['impact']}/run.json"]
        return argv, None, extra
    if spec["id"] == "a8":
        argv = ["--scores-gemma", scores("gemma")]
        # the order of the original analysis (the order of the runs map), which fixes the row order of A8_TABLES.md
        order = [key[:-3] for key in runs if key.endswith("|r1")]
        if sorted(order) != sorted(frozen.get("models") or {}):
            return None, "the A8 runs map and the shipped analysis name different models", extra
        for model in order:
            argv += ["--model", f"{model}={scores(model + '|r1')},{scores(model + '|r2')}"]
        return argv, None, extra
    raise ValueError(spec["id"])


def plan_followups(b: Build):
    """b.followups (README table) and the verify-plan entries of the follow-up analyses."""
    b.followups, entries = [], []
    present = {r["run"] for r in b.runs}
    for spec in FOLLOWUPS:
        row = dict(spec, shipped=all(p in b.files for p in spec["outputs"].values()),
                   runs_present=all(r in present for r in spec["runs"].values()))
        if not (row["shipped"] or row["runs_present"]):
            b.notes.append(f"follow-up {spec['name']}: neither its runs nor its outputs are in this build")
            continue
        if not row["shipped"]:
            row["not_verified"] = "its analysis output is not in this build yet"
        elif not row["runs_present"]:
            row["not_verified"] = "not all of its runs are in this build yet"
        else:
            frozen = json.loads((b.stage / spec["outputs"][spec["json"]]).read_text(encoding="utf-8"))
            argv, reason, extra = followup_argv(spec, frozen, b)
            if argv is None:
                row["not_verified"] = reason
            else:
                records = _input_records(frozen.get("inputs") or {}, [])
                mine = {rec["sha256"] for pkg, rec in b.files.items()
                        if any(pkg.startswith(f"{top}/{rid}/") for rid in spec["runs"].values()
                               for top in ("results", "runs")) or re.match(r"^items/[a-z]+/items\.jsonl$", pkg)}
                foreign = [path for path, sha in records if sha not in mine]
                if foreign:
                    fail(f"follow-up {spec['name']}: the shipped {spec['json']} names inputs whose sha256 is not that "
                         f"of a file of its runs in the package: {foreign[:5]}")
                stored = set(b.files) | set(b.dedup)
                relative = all(path in stored for path, _ in records)
                row["verify"] = {"inputs_package_relative": relative}
                entry = {"name": spec["name"], "id": spec["id"], "script": spec["script"], "argv": argv,
                         "outputs": spec["outputs"], "json": spec["json"], "inputs_package_relative": relative}
                entry.update(extra)
                entries.append(entry)
        if row.get("not_verified"):
            b.notes.append(f"follow-up {spec['name']}: not re-run by verify.py ({row['not_verified']})")
        b.followups.append(row)
    return entries


def write_verify_plan(b: Build):
    runs = []
    for run in b.runs:
        rid = run["run"]
        if run["kind"] == "scores_only":
            runs.append({"run": rid, "shard": f"runs/{rid}/{rid}.tar.xz", "index": f"runs/{rid}/{rid}.INDEX.jsonl",
                         "run_json": f"runs/{rid}/run.json", "cells_jsonl": f"runs/{rid}/cells.jsonl",
                         "analysis": None,
                         "scores_only": {"scores": f"results/{rid}/scores.jsonl",
                                         "plain_copies": sorted(p[len(f"runs/{rid}/"):] for p in b.files
                                                                if p.startswith(f"runs/{rid}/launches/"))}})
            continue
        A = json.loads((b.stage / f"results/{rid}/analysis/ANALYSIS.json").read_text(encoding="utf-8"))
        by_source = {}
        for pkg, rec in list(b.files.items()) + list(b.dedup.items()):
            by_source.setdefault(rec["source"].get("path"), pkg)
        inputs = {}
        for key, meta in A["inputs"].items():
            pkg = by_source.get(meta["path"])
            if pkg is None:
                fail(f"{rid}: analysis input {meta['path']} is not in the package")
            inputs[key] = {"package_path": pkg, "original_path": meta["path"], "sha256": meta["sha256"]}
        dispatch = f"dispatch/{run['items']}/dispatch_manifest.json"
        entry = {"run": rid, "shard": f"runs/{rid}/{rid}.tar.xz", "index": f"runs/{rid}/{rid}.INDEX.jsonl",
                 "run_json": f"runs/{rid}/run.json", "cells_jsonl": f"runs/{rid}/cells.jsonl",
                 "analysis": {"code": f"code/{run['code']}", "inputs": inputs, "dispatch_manifest": dispatch,
                              "payloads": f"items/{run['items']}/payloads",
                              "frozen": {name: f"results/{rid}/analysis/{name}"
                                         for name in ("ANALYSIS.json", "TABLES.md", "CELLS.jsonl")}}}
        runs.append(entry)
    for run in b.aborted:
        rid = run["run"]
        runs.append({"run": rid, "shard": f"runs/aborted/{rid}/{rid}.tar.xz",
                     "index": f"runs/aborted/{rid}/{rid}.INDEX.jsonl", "analysis": None})
    comparisons = []
    for run in b.runs:
        cmp_ = run.get("compare")
        if not cmp_:
            continue
        rid, ref = run["run"], cmp_["ref"]
        row = {"name": f"{rid} vs {ref}", "mode": cmp_["mode"],
               "ref_cells": f"results/{ref}/analysis/CELLS.jsonl", "new_cells": f"results/{rid}/analysis/CELLS.jsonl",
               "frozen": f"results/{rid}/COMPARE.json"}
        if cmp_["mode"] == "model":
            row["ref_analysis"] = f"results/{ref}/analysis/ANALYSIS.json"
            row["new_analysis"] = f"results/{rid}/analysis/ANALYSIS.json"
            if cmp_.get("labels"):
                row["labels"] = cmp_["labels"]
        else:
            row["seed"] = cmp_["seed"]
        comparisons.append(row)
    plan = {
        "schema": "absent-edges-expA-verify-plan-v1", "label": LABEL,
        "runs": runs,
        "truth_shards": [{"shard": p, "index": s["index"]} for p, s in sorted(b.shards.items()) if s["kind"] == "truth"],
        "reanalysis": {"script": "code/post-results/expA_reanalyze_independent.py",
                       "cells": "results/main-20261101b/analysis/CELLS.jsonl",
                       "analysis": "results/main-20261101b/analysis/ANALYSIS.json",
                       "tcdf_crosscheck": "results/main-20261101b/independent/TCDF_SCIPY_CROSSCHECK.json",
                       "frozen": "results/main-20261101b/independent/REANALYSIS.json"},
        "comparisons": comparisons,
        "compare_script": "code/post-results/expA_compare_runs.py",
        "followups": plan_followups(b),
    }
    b.add_bytes("reproduce/VERIFY_PLAN.json", dump_json(plan),
                {"kind": "generated", "what": "inputs and expected outputs for reproduce/verify.py"}, generated=True)


# ------------------------------------------------------------------------------------------------ provenance
ANCHOR_SUFFIXES = (".json", ".md", ".sha256", ".log", ".txt", ".time")


def anchor_index(b: Build):
    """{sha256: [witness]} over every byte-identical frozen file that records full sha256 values."""
    index = collections.defaultdict(list)
    for pkg, rec in sorted(b.files.items()):
        if rec.get("byte_identical") is not True:
            continue
        name = pkg.rsplit("/", 1)[-1]
        if not (name.endswith(ANCHOR_SUFFIXES) or name == "SHA256SUMS") or rec["bytes"] > 8_000_000:
            continue
        if pkg.startswith("licenses/") or pkg.startswith("code/") and not name.endswith(".sha256"):
            continue
        data = (b.stage / pkg).read_bytes()
        if name.endswith(".json"):
            try:
                doc = json.loads(data)
            except ValueError:
                doc = None
            if doc is not None:
                def walk(value, path):
                    if isinstance(value, dict):
                        for k in sorted(value):
                            walk(value[k], path + [k])
                    elif isinstance(value, list):
                        for i, v in enumerate(value):
                            walk(v, path + [i])
                    elif isinstance(value, str) and HEX64_STR.match(value):
                        index[value].append({"anchor": pkg, "kind": "json", "path": path})
                walk(doc, [])
                continue
        if name.endswith(".sha256") or name == "SHA256SUMS":
            ok = True
            for lineno, line in enumerate(data.decode("utf-8", "replace").splitlines(), 1):
                m = re.match(r"^([0-9a-f]{64}) [ *](.+)$", line)
                if m:
                    index[m.group(1)].append({"anchor": pkg, "kind": "sums", "entry": m.group(2), "line": lineno})
                elif line.strip():
                    ok = False
            if ok:
                continue
        for lineno, line in enumerate(data.splitlines(), 1):
            for m in HEX64.finditer(line):
                index[m.group(1).decode()].append({"anchor": pkg, "kind": "text", "line": lineno})
    return index


def write_provenance(b: Build):
    index = anchor_index(b)
    entries = []

    def witnesses(sha, own_path):
        out = [w for w in index.get(sha, []) if w["anchor"] != own_path]
        return out[:64], len(out)

    for pkg in posix_sorted(b.files):
        rec = dict(b.files[pkg])
        target = rec.get("original_sha256") or rec["sha256"]
        wit, total = witnesses(target, pkg)
        rec["witnesses"] = wit
        rec["witnesses_total"] = total
        if rec.get("original_sha256"):
            rec["witnesses_pin"] = "original bytes (before sanitization; see SANITIZATION.md)"
            for s in b.sanitized:
                if s["path"] == pkg:
                    s["pins"] = sorted({f"{w['anchor']}" + (f" ({w.get('entry')})" if w.get("entry") else "")
                                        for w in wit})
        entries.append(rec)
    dedup = []
    for pkg in posix_sorted(b.dedup):
        rec = dict(b.dedup[pkg])
        rec["witnesses"], rec["witnesses_total"] = witnesses(rec["sha256"], pkg)
        dedup.append(rec)
    # witnesses of shard members (truth and run shards) found in frozen package files
    member_witnesses = {}
    for shard, meta in sorted(b.shards.items()):
        rows = [json.loads(line) for line in (b.stage / meta["index"]).read_bytes().splitlines()]
        found = {}
        for row in rows:
            wit = index.get(row["sha256"])
            if wit:
                found[row["member"]] = wit[:16]
        member_witnesses[shard] = found
    b.member_witnesses = member_witnesses
    return entries, dedup


def finish(b: Build):
    entries, dedup = write_provenance(b)
    # SANITIZATION.md needs the pins found by write_provenance; write it now and append its record.
    write_sanitization(b)
    sanit = b.files["SANITIZATION.md"]
    entries.append(dict(sanit, witnesses=[], witnesses_total=0))
    entries.sort(key=lambda r: r["path"].encode("utf-8"))
    provenance = {
        "schema": "absent-edges-expA-provenance-v1", "label": LABEL, "package": PACKAGE,
        "built_utc": utc_now(), "builder": "reproduce/build_release_package.py",
        "export": {"head": b.export["head"], "sha256": b.export_sha256},
        "releases": {short: {"commit": rel["commit"], "role": rel["role"],
                             "analysis_code_for_runs": [r["run"] for r in b.runs if r["code"] == short]}
                     for short, rel in b.releases.items()},
        "runs": [{k: r.get(k) for k in ("run", "kind", "analysis", "code", "code_candidates", "items", "title",
                                        "status", "compare")} for r in b.runs],
        "followups": [{k: f.get(k) for k in ("name", "protocol", "script", "runs", "outputs", "shipped",
                                             "runs_present", "verify", "not_verified")} for f in b.followups],
        "launch_records": b.aborted, "skipped_runs": b.skipped,
        "notes": [
            "Every stored file: path, sha256, bytes, source (Core2 path, or git path at the export commit with its "
            "blob id), byte_identical (null for generated files), original_sha256 for sanitized files, and the "
            "witnesses: frozen files in this package that record the file's sha256 (json path, sums entry or "
            "text line). verify.py re-reads every witness.",
            "dedup lists logical files that are byte-identical to a stored file and are therefore stored once.",
            "shard_member_witnesses lists shard members whose sha256 is recorded in a frozen package file; every "
            "member's sha256 is in the shard's INDEX.jsonl.",
        ],
        "files": entries, "dedup": dedup, "shards": b.shards, "shard_member_witnesses": b.member_witnesses,
        "sanitized": b.sanitized,
    }
    raw = dump_json(provenance)
    (b.stage / "PROVENANCE.json").write_bytes(raw)
    b.files["PROVENANCE.json"] = {"path": "PROVENANCE.json", "sha256": sha256_bytes(raw), "bytes": len(raw)}
    manifest = []
    for pkg in posix_sorted(b.files):
        manifest.append(f"{b.files[pkg]['sha256']}  {pkg}\n")
    raw = "".join(manifest).encode("utf-8")
    (b.stage / "MANIFEST.sha256").write_bytes(raw)
    return sha256_bytes(raw), len(manifest)


def check_disk_tree(b: Build):
    on_disk = set(walk_files(b.stage))
    expected = set(b.files) | {"MANIFEST.sha256"}
    if on_disk != expected:
        fail(f"staging tree differs from the records: extra {sorted(on_disk - expected)[:5]}, "
             f"missing {sorted(expected - on_disk)[:5]}")


def final_leak_check(b: Build) -> dict:
    """Re-read every published plain file (PROVENANCE.json and MANIFEST.sha256 included): no other_user token, its
    own_* occurrences exactly as SANITIZATION.md lists them, and no name fragment outside a listed occurrence. Shard
    members were checked when their shard was scanned (b.leaks). Prints file names and line numbers only."""
    listed = collections.defaultdict(dict)
    for o in b.listed_own or []:
        if o["member"] is None:
            listed[o["file"]][o["class"]] = o["lines"]
    problems = [f"{x['file']}" + (f" member {x['member']}" if x["member"] else "") +
                f": name fragment outside a listed occurrence, line(s) {x['lines'][:10]}" for x in b.leaks]
    files = 0
    for rel in walk_files(b.stage):
        if rel.endswith(".tar.xz"):
            continue
        files += 1
        other, own, unlisted = b.scanner.scan((b.stage / rel).read_bytes())
        if other:
            problems.append(f"{rel}: holds an other_user token")
        if rel == "PROVENANCE.json":
            # PROVENANCE.json quotes the sums entries of A8 score files (OUTPUTS.sha256), whose own_storage_path
            # occurrences are listed for those files; the quotes are not listed a second time.
            own = {c: v for c, v in own.items() if c != "own_storage_path"}
        if own != listed.get(rel, {}):
            problems.append(f"{rel}: own_* occurrences {sorted((c, len(v)) for c, v in own.items())} differ from "
                            f"SANITIZATION.md {sorted((c, len(v)) for c, v in listed.get(rel, {}).items())}")
        if unlisted:
            problems.append(f"{rel}: name fragment outside a listed occurrence, line(s) {unlisted[:10]}")
    if problems:
        fail("leak check: " + str(len(problems)) + " problem(s):\n  " + "\n  ".join(problems[:60]))
    members = sum(s["members"] for s in b.shards.values())
    log(f"leak check passed: {files} plain files re-read, {members} shard members scanned; "
        f"{len(b.scanner.fragments)} name fragments")
    return {"plain_files": files, "shard_members": members, "fragments": len(b.scanner.fragments)}


def cmd_build(args) -> int:
    b = Build(args)
    b.out_root.mkdir(parents=True, exist_ok=True)
    os.chmod(b.out_root, 0o700)
    b.work.mkdir(exist_ok=True)
    b.cache.mkdir(exist_ok=True)
    for stale in b.out_root.glob(".staging-*"):  # an interrupted earlier build
        for current, _, _ in os.walk(stale):
            os.chmod(current, 0o755)
        shutil.rmtree(stale)
    b.stage.mkdir(parents=True)
    log(f"staging {b.stage}")
    b.load_export()
    discover(b)
    step_protocol_registry(b)
    log("protocol, registry done")
    step_items(b)
    log("items, dispatch, findability done")
    step_results(b)
    log("results done")
    step_code(b)
    log("code trees done")
    step_runs(b)
    step_truth(b)
    step_licences(b)
    log("licences done")
    write_notice(b)
    write_third_party(b)
    write_verify_plan(b)
    claims = build_claims(b)
    render_readme(b, claims)
    release_id, count = finish(b)
    check_disk_tree(b)
    leak_summary = final_leak_check(b)
    for path in walk_files(b.stage):
        os.chmod(b.stage / path, 0o444)
    final = b.out_root / PACKAGE
    previous = None
    if final.exists():
        previous = b.out_root / f".previous-{b.stamp}"
        final.rename(previous)
    b.stage.rename(final)
    b.stage_root.rmdir()
    if previous is not None:
        for current, dirnames, filenames in os.walk(previous):
            os.chmod(current, 0o755)
        shutil.rmtree(previous)
    groups, tiers = sizes_table(b)
    summary = {
        "package": str(final), "files": count + 1, "release_id": release_id,
        "tier_bytes": dict(tiers), "total_bytes": sum(r["bytes"] for r in b.files.values()) +
        (final / "MANIFEST.sha256").stat().st_size,
        "groups": {f"{t} {p}": {"files": n, "bytes": s} for (t, p), (n, s) in groups.items() if n},
        "sanitized": [{k: v for k, v in s.items() if k != "source"} for s in b.sanitized],
        "own_occurrences": collections.Counter(o["class"] for o in b.own),
        "own_occurrence_lines": {cls: sum(len(o["lines"]) for o in b.listed_own if o["class"] == cls)
                                 for cls in Scanner.OWN_CLASSES},
        "leak_check": leak_summary,
        "licence_mentions": dict(b.licence_mentions),
        "unfilled_placeholders": b.unfilled, "notes": b.notes,
        "export_head": b.export["head"],
        "runs": {r["run"]: {"kind": r["kind"], "code": r["code"], "items": r["items"], "compare": r.get("compare")}
                 for r in b.runs},
        "followups": {f["name"]: f.get("verify") or f.get("not_verified") for f in b.followups},
        "launch_records": [r["run"] for r in b.aborted], "skipped_runs": b.skipped,
        "releases": {short: rel["commit"] for short, rel in b.releases.items()},
    }
    (b.out_root / f"BUILD-{b.stamp}.json").write_bytes(dump_json(summary))
    print(json.dumps(summary, indent=1, default=str))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    exp = sub.add_parser("export", help="on the workstation: tar the repository files the package needs")
    exp.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    exp.add_argument("--out", required=True)
    bld = sub.add_parser("build", help="on Core2: assemble the package")
    bld.add_argument("--export", required=True)
    bld.add_argument("--private", required=True, help="JSON with the account-name classes (never published)")
    bld.add_argument("--out-root", default=OUT_ROOT)
    bld.add_argument("--copyright-holder")
    bld.add_argument("--code-url")
    bld.add_argument("--writeup-url")
    bld.add_argument("--contact")
    bld.add_argument("--funding", help="funding and compute statement for the dataset card")
    args = parser.parse_args(argv)
    sys.dont_write_bytecode = True
    return cmd_export(args) if args.command == "export" else cmd_build(args)


if __name__ == "__main__":
    raise SystemExit(main())
