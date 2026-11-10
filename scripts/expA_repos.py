"""Exp A repository registry builder (lane R; docs/EXPA_DESIGN.md section 3, docs/EXPA_REPOS.md).

Subcommands
  exclusions  scan project files, resolve GitHub network roots, write the exclusion list (local, uses gh)
  vet         GitHub metadata + network root + exclusion check for the candidate table (local, uses gh)
  list        print family names (for xargs)
  clone       clone, pin (release tag before cutoff), check licence and package size
  env         uv venv --python /usr/bin/python3.13 and install package + test dependencies (network on)
  lock        uv pip freeze -> registry/locks/<family>.txt with sha256
  baseline    run the test command OFFLINE (unshare --user --map-root-user --net, scrubbed env, 30 min cap)
  run         clone + env + lock + baseline (+ lock re-check) per family; --jobs N runs families in parallel
  assemble    merge candidates, per-family results and exclusions into registry.json; assign roles; validate
  check       validate an existing registry file

Remote layout (Core2, local XFS): $EXPA_ROOT/{repos,envs,registry}/ and $EXPA_ROOT/dev/repos/.
No GPU work. Tests never see the network, tokens or the real HOME.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from absent_edges.probe import repos as R  # noqa: E402

EXPA_ROOT = Path(os.environ.get("EXPA_ROOT", R.EXPA_ROOT))
UV = os.environ.get("UV_BIN", str(Path.home() / ".local/bin/uv"))

# Tools, datasets and fixtures that appear with repository-shaped identifiers in project files but
# were never task/training/diagnostic families (see docs/EXPA_REPOS.md, "Seen but not excluded").
NON_FAMILY_NAMES = {
    "gersteinlab/locagent": "graph backend tool pinned by the protocols (vendor/locagent)",
    "swe-bench/swe-smith": "benchmark toolkit/dataset name in provenance metadata",
    "microsoft/swe-bench-live": "benchmark dataset name in public-source cache metadata",
    "swe-bench-live/swe-bench-live": "benchmark dataset name in public-source cache metadata",
    "sola-st/dypybench": "public artifact named in public-source cache metadata",
    "james-chahwan/repo-graph": "public artifact named in public-source cache metadata",
    "synthetic/repo": "placeholder name in unit-test fixtures",
}
# Public toolkit sources copied into the project (SWE-smith profile table and upstream tree).
LISTING_PATTERNS = (
    "staged/*/public-source/**",
    "staged/*/inputs/profiles-*.py",
)

# ------------------------------------------------------------------------------------------------
# Candidate table. Fields: family, repo, network_root_id (from `vet`), package_dirs, import_names,
# extras / groups / requirements / packages (test dependencies), test_args, test_env, pin_tag
# (explicit tag, dev families only), notes.
# ------------------------------------------------------------------------------------------------
def _c(family, repo, root, dirs, imports, test_args, **kw) -> dict:
    return dict(family=family, repo=repo, network_root_id=root, package_dirs=dirs, import_names=imports,
                test_args=test_args, extras=kw.pop("extras", []), groups=kw.pop("groups", []),
                requirements=kw.pop("requirements", []), packages=kw.pop("packages", ["pytest"]),
                test_env=kw.pop("test_env", {}), pin_tag=kw.pop("pin_tag", None), notes=kw.pop("notes", ""), **kw)


CANDIDATES: list[dict] = [
    # dev families (coordinator-assigned tags; harness debugging only, excluded from analysis)
    _c("more-itertools", "more-itertools/more-itertools", 4144530, ["more_itertools/"], ["more_itertools"],
       ["tests"], pin_tag="v11.2.1", notes="dev family; tag v11.2.1 commit is dated after the cutoff"),
    _c("boltons", "mahmoud/boltons", 8307391, ["boltons/"], ["boltons"], ["tests"], pin_tag="26.2.0",
       notes="dev family"),
    # main/spare candidates (network roots from `vet`; none matches the exclusion list)
    _c("rich", "Textualize/rich", 220809393, ["rich/"], ["rich"], ["tests"], packages=["pytest", "attrs"]),
    _c("graphql-core", "graphql-python/graphql-core", 143207933, ["src/graphql/"], ["graphql"], ["tests"],
       groups=["test"], packages=[]),
    _c("elementpath", "sissaschool/elementpath", 126886685, ["elementpath/"], ["elementpath"], ["tests"],
       packages=["pytest", "lxml", "xmlschema"]),
    _c("xmlschema", "sissaschool/xmlschema", 70905710, ["xmlschema/"], ["xmlschema"], ["tests"],
       packages=["pytest", "lxml", "jinja2"]),
    _c("xlsxwriter", "jmcnamara/XlsxWriter", 7433211, ["xlsxwriter/"], ["xlsxwriter"], ["xlsxwriter/test"],
       notes="the editable (setup.py) install leaves an untracked xlsxwriter.egg-info/ in the repository root"),
    _c("petl", "petl-developers/petl", 2233194, ["petl/"], ["petl"], ["petl/test"],
       requirements=["requirements-tests.txt"]),
    _c("click", "pallets/click", 19103692, ["src/click/"], ["click"], ["tests"], groups=["tests"], packages=[]),
    _c("jinja", "pallets/jinja", 994551, ["src/jinja2/"], ["jinja2"], ["tests"],
       requirements=["requirements/tests.txt"], packages=[]),
    _c("websockets", "python-websockets/websockets", 9113587, ["src/websockets/"], ["websockets"], ["tests"],
       packages=["pytest", "python-socks[asyncio]", "trio", "werkzeug"]),
    _c("icalendar", "collective/icalendar", 2222138, ["src/icalendar/"], ["icalendar"], ["src/icalendar/tests"],
       groups=["test"], packages=[]),
    _c("markdown", "Python-Markdown/markdown", 692146, ["markdown/"], ["markdown"],
       ["tests/test_apis.py", "tests/test_extensions.py", "tests/test_meta.py", "tests/test_syntax",
        "tests/test_legacy.py"], extras=["testing"],
       notes="suite is written for unittest; tests/test_legacy.py sets warnings.simplefilter('error') at import, "
             "so it is collected last (otherwise pytest's collection warnings become errors)"),
    _c("markdown-it-py", "executablebooks/markdown-it-py", 249956697, ["markdown_it/"], ["markdown_it"], ["tests"],
       extras=["testing", "linkify"], packages=[]),
    _c("mpmath", "mpmath/mpmath", 2934512, ["mpmath/"], ["mpmath"], ["mpmath"], extras=["tests"], packages=[]),
    _c("rdflib", "RDFLib/rdflib", 3342046, ["rdflib/"], ["rdflib"], ["test", "rdflib"],
       packages=["pytest>=7.1.3,<9", "lxml", "html5rdf", "orjson"]),
    _c("prompt-toolkit", "prompt-toolkit/python-prompt-toolkit", 16241224, ["src/prompt_toolkit/"],
       ["prompt_toolkit"], ["tests"]),
    _c("param", "holoviz/param", 4191817, ["numbergen/", "param/"], ["param", "numbergen"], ["tests"],
       extras=["tests"], packages=[]),
    _c("liquid", "jg-rp/liquid", 304351404, ["liquid/"], ["liquid"], ["tests"],
       packages=["pytest", "pytest-mock", "pytest-timeout", "hypothesis", "pyyaml", "mock"]),
    _c("cmd2", "python-cmd2/cmd2", 51442999, ["cmd2/"], ["cmd2"], ["-o", "addopts=", "tests"], groups=["test"],
       packages=[], notes="ini addopts (coverage reports written into the checkout) cleared with -o addopts="),
    _c("python-pptx", "scanny/python-pptx", 6794278, ["src/pptx/"], ["pptx"], ["tests"],
       requirements=["requirements-test.txt"], packages=["pyparsing<3.3"],
       notes="pyparsing<3.3: test helpers use delimitedList, deprecated in pyparsing 3.3 (warnings are errors)"),
    _c("python-docx", "python-openxml/python-docx", 13592924, ["src/docx/"], ["docx"], ["tests"],
       requirements=["requirements-test.txt"], packages=["pyparsing<3.3"],
       notes="pyparsing<3.3: test helpers use delimitedList, deprecated in pyparsing 3.3 (warnings are errors)"),
    _c("pyparsing", "pyparsing/pyparsing", 91262163, ["pyparsing/"], ["pyparsing"], ["tests"], extras=["diagrams"],
       notes="the test run writes railroad-diagram *.html files into the repository root"),
    _c("packaging", "pypa/packaging", 19837758, ["src/packaging/"], ["packaging"], ["tests"], groups=["test"],
       packages=[]),
    _c("sqlite-utils", "simonw/sqlite-utils", 140912432, ["sqlite_utils/"], ["sqlite_utils"], ["tests"],
       packages=["pytest", "hypothesis", "cogapp"]),
    _c("plumbum", "tomerfiliba/plumbum", 4156651, ["plumbum/"], ["plumbum"], ["tests"], groups=["test"],
       packages=[]),
    _c("alembic", "sqlalchemy/alembic", 159267771, ["alembic/"], ["alembic"], ["tests"]),
    _c("zeep", "mvantellingen/python-zeep", 51690353, ["src/zeep/"], ["zeep"], ["tests"], groups=["dev"],
       packages=[]),
    _c("deepdiff", "seperman/deepdiff", 24484622, ["deepdiff/"], ["deepdiff"], ["tests"],
       extras=["test", "dev", "cli", "static"], packages=[]),
    _c("httpx", "encode/httpx", 179492861, ["httpx/"], ["httpx"], ["tests"], requirements=["requirements.txt"],
       packages=[], notes="the test run writes a file named 'test' into the repository root"),
    _c("lark", "lark-parser/lark", 80946973, ["lark/"], ["lark"], ["tests"],
       packages=["pytest", "interegular>=0.3.1,<0.4.0", "regex"]),
    _c("traitlets", "ipython/traitlets", 14993021, ["traitlets/"], ["traitlets"], ["tests"], extras=["test"],
       packages=[]),
    _c("yapf", "google/yapf", 32476524, ["yapf/"], ["yapf"], ["yapftests"],
       notes="vendored third_party/yapf_third_party is outside package scope"),
    _c("dateutil", "dateutil/dateutil", 26876372, ["src/dateutil/"], ["dateutil"], ["tests"],
       packages=["pytest<9", "pytest-cov", "hypothesis", "freezegun", "six"],
       notes="setup.cfg turns warnings into errors: pytest<9 (the 2024 release passes generators to parametrize, "
             "deprecated in pytest 9); pytest-cov only registers the no_cover mark, coverage is not enabled"),
    _c("genshi", "edgewall/genshi", 1822630, ["genshi/"], ["genshi"], ["genshi"], packages=["pytest", "babel"]),
    _c("segno", "heuer/segno", 64920252, ["segno/"], ["segno"], ["tests"],
       packages=["pytest", "pypng", "qrcode-artistic", "Pillow"]),
    # Considered, but on the exclusion list (reserved original-benchmark network roots). Never cloned;
    # kept as status=excluded records so the registry shows the check. attrs is the design-doc example.
    _c("attrs", "python-attrs/attrs", 29918975, ["src/attr/", "src/attrs/"], ["attr", "attrs"], ["tests"],
       excluded_candidate=True),
    _c("networkx", "networkx/networkx", 890377, ["networkx/"], ["networkx"], ["networkx"], excluded_candidate=True),
    _c("structlog", "hynek/structlog", 12085960, ["src/structlog/"], ["structlog"], ["tests"],
       excluded_candidate=True),
    _c("transitions", "pytransitions/transitions", 25131903, ["transitions/"], ["transitions"], ["tests"],
       excluded_candidate=True),
]

# Pool repositories inspected on Core2 (pinned, licence-checked, sized) but not proposed.
# package_loc = non-blank, non-comment lines in package scope at the pinned commit.
# Main-family count: the smallest n with at least 20 families left in split A after the design's
# 30% Q hold-out (n - round(0.3 n) >= 20 gives n = 29), plus one buffer -> 30 (A = 21, Q = 9).
# The coordinator's floor is MIN_MAIN = 22; `assemble --n-main 22` reproduces the minimum.
N_MAIN = 30
N_MAIN_REASON = ("30 = smallest n with >= 20 families in split A after the 30% Q hold-out (29), plus one buffer; "
                 "floor 22")

REJECTED: list[dict] = [
    dict(repo="getsentry/responses", stage="pre-screen", reason="package scope 1,674 LOC (< ~3k)"),
    dict(repo="pyeve/cerberus", stage="pre-screen", reason="package scope 2,564 LOC (< ~3k)"),
    dict(repo="lmfit/uncertainties", stage="pre-screen", reason="package scope 2,546 LOC (< ~3k)"),
    dict(repo="grantjenks/python-sortedcontainers", stage="pre-screen",
         reason="package scope 2,942 LOC (borderline) and newest release is from 2021"),
    dict(repo="pytoolz/toolz", stage="pre-screen", reason="package scope 3,148 LOC (borderline); larger scopes preferred"),
    dict(repo="pallets/werkzeug", stage="pre-screen", reason="third pallets repository; org cap of two (click, jinja)"),
    dict(repo="sqlalchemy/mako", stage="pre-screen",
         reason="second sqlalchemy repository; alembic preferred (larger scope); count kept near 32"),
    dict(repo="python-attrs/cattrs", stage="pre-screen",
         reason="same org as reserved python-attrs/attrs; hypothesis-heavy slow suite"),
    dict(repo="wireservice/agate", stage="pre-screen", reason="same org as reserved wireservice/csvkit"),
    dict(repo="omry/omegaconf", stage="pre-screen",
         reason="source build generates the ANTLR parser (needs Java); generated parser would sit in package scope"),
    dict(repo="uqfoundation/dill", stage="pre-screen",
         reason="pickles frames/code objects by introspection; risky under the runtime tracer"),
    dict(repo="coleifer/peewee", stage="pre-screen",
         reason="single-module layout (peewee.py) plus optional C extensions; directory scopes preferred"),
    dict(repo="construct/construct", stage="pre-screen",
         reason="logic concentrated in one ~6k-line module; newest release 2023"),
    dict(repo="pexpect/pexpect", stage="pre-screen", reason="newest release 2023; pty/timing-dependent tests"),
    dict(repo="jaraco/inflect", stage="pre-screen", reason="3,201 LOC mostly in one module"),
    dict(repo="asottile/pyupgrade", stage="pre-screen",
         reason="plugin registry dispatch (few direct internal calls); count kept near 32"),
    dict(repo="google/python-fire", stage="pre-screen", reason="small test suite; count kept near 32"),
    dict(repo="eliben/pycparser", stage="pre-screen", reason="4,186 LOC; count kept near 32"),
    dict(repo="python-poetry/tomlkit", stage="pre-screen", reason="4,631 LOC; count kept near 32"),
    dict(repo="netaddr/netaddr", stage="pre-screen", reason="5,060 LOC; count kept near 32"),
    dict(repo="pydoit/doit", stage="pre-screen", reason="5,587 LOC; count kept near 32"),
    dict(repo="miyuchina/mistletoe", stage="pre-screen", reason="3,703 LOC; count kept near 32"),
    dict(repo="kurtmckee/feedparser", stage="pre-screen", reason="3,943 LOC; count kept near 32"),
    dict(repo="jsonpickle/jsonpickle", stage="pre-screen", reason="3,302 LOC; count kept near 32"),
    dict(repo="c0fec0de/anytree", stage="pre-screen", reason="3,365 LOC; count kept near 32"),
]


def candidate_map() -> dict[str, dict]:
    out = {}
    for c in CANDIDATES:
        if c["family"] in out:
            raise SystemExit(f"duplicate family {c['family']}")
        out[c["family"]] = c
    return out


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(R.canonical_json(obj), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def run(cmd, cwd=None, env=None, timeout=None, log=None, check=False) -> subprocess.CompletedProcess:
    """Run a command (argv list); optionally append combined output to a log file."""
    proc = subprocess.run(cmd, cwd=cwd, env=env, timeout=timeout, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, errors="replace")
    if log is not None:
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(f"$ {' '.join(map(str, cmd))}\n{proc.stdout}\n[exit {proc.returncode}]\n")
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(map(str, cmd))}\n{proc.stdout[-4000:]}")
    return proc


def work_dir(family: str) -> Path:
    return EXPA_ROOT / "registry" / "work" / family


def repo_dir(family: str) -> Path:
    return EXPA_ROOT / "repos" / family


def env_dir(family: str) -> Path:
    return EXPA_ROOT / "envs" / family


def lock_path(family: str) -> Path:
    return EXPA_ROOT / "registry" / "locks" / f"{family}.txt"


# ------------------------------------------------------------------------------------------------
# GitHub metadata (local; `gh api` when available, else unauthenticated REST)
# ------------------------------------------------------------------------------------------------

def github_get(path: str):
    """GET a GitHub REST path; returns parsed JSON or None for 404/410."""
    if shutil.which("gh"):
        p = subprocess.run(["gh", "api", path], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           text=True, errors="replace", timeout=90)
        if p.returncode == 0:
            return json.loads(p.stdout)
        if "Not Found" in p.stderr or "HTTP 404" in p.stderr or "HTTP 410" in p.stderr \
                or "Repository access blocked" in p.stderr:
            return None
        raise RuntimeError(f"gh api {path} failed: {p.stderr.strip()[:300]}")
    req = urllib.request.Request(f"https://api.github.com/{path}",
                                 headers={"Accept": "application/vnd.github+json", "User-Agent": "expA-repos"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code in (404, 410, 451):
            return None
        raise


def summarize_repo(meta) -> dict:
    if not meta:
        return {"full_name": None, "id": None, "network_root_id": None, "network_root_full_name": None}
    src = meta.get("source") if meta.get("fork") else None
    return {
        "full_name": meta.get("full_name"), "id": meta.get("id"), "fork": bool(meta.get("fork")),
        "network_root_id": (src or meta).get("id"), "network_root_full_name": (src or meta).get("full_name"),
        "archived": meta.get("archived"), "default_branch": meta.get("default_branch"),
        "github_license_guess": (meta.get("license") or {}).get("spdx_id"), "language": meta.get("language"),
        "size_kb": meta.get("size"), "pushed_at": meta.get("pushed_at"), "stars": meta.get("stargazers_count"),
    }


def resolve_keys(keys: list[str], cache: dict, jobs: int = 8) -> dict:
    """Resolve ``name:<owner/name>`` and ``root:<id>`` keys through the GitHub API (cached)."""
    todo = [k for k in keys if k not in cache]

    def one(key: str):
        kind, value = key.split(":", 1)
        meta = github_get(f"repos/{value}" if kind == "name" else f"repositories/{value}")
        return key, {**summarize_repo(meta), "resolved_at": now_utc()}

    with cf.ThreadPoolExecutor(max_workers=jobs) as pool:
        for key, res in pool.map(one, todo):
            cache[key] = res
    return {k: cache[k] for k in keys}


# ------------------------------------------------------------------------------------------------
# exclusions / vet
# ------------------------------------------------------------------------------------------------

def git_head(path: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL, text=True).stdout.strip() or None
    except OSError:
        return None


def cmd_exclusions(args) -> int:
    project = Path(args.project_root).resolve()
    t0 = time.time()
    scan = R.scan_project(project)
    scan_seconds = round(time.time() - t0, 1)
    cat_path = project / R.CATALOGUE_PATH
    cat_bytes = cat_path.read_bytes()
    catalogue = json.loads(cat_bytes)
    cat_fams = R.catalogue_families_from(catalogue)
    classes = R.classify_scan_names(scan, NON_FAMILY_NAMES, LISTING_PATTERNS)
    cache_path = Path(args.cache) if args.cache else None
    cache = read_json(cache_path) if cache_path and cache_path.exists() else {}
    keys = sorted({f"name:{k}" for k, c in classes.items() if c["class"] == "family"}
                  | {f"name:{k}" for k in cat_fams}
                  | {f"root:{r}" for r in scan["roots"]})
    if args.resolve == "github":
        resolved = resolve_keys(keys, cache, jobs=args.jobs)
        if cache_path:
            write_json(cache_path, cache)
    else:
        resolved = {}
    excl = R.build_exclusions(scan, resolved, cat_fams, classes)
    not_excluded = []
    for key, c in sorted(classes.items()):
        if c["class"] != "family" and key not in cat_fams:
            info = scan["names"][key]
            not_excluded.append({"name": key, "class": c["class"], "reason": c["reason"],
                                 "kinds": info["kinds"], "source_paths": info["paths"][:5],
                                 "source_count": len(info["paths"])})
    unresolved = sorted(k for k, v in resolved.items() if not v.get("full_name"))
    doc = {
        "schema": R.EXCLUSIONS_SCHEMA,
        "generated_at": now_utc(),
        "project_git_head": git_head(project),
        "patterns": scan["patterns"],
        "listing_patterns": list(LISTING_PATTERNS),
        "non_family_names": NON_FAMILY_NAMES,
        "files_scanned": scan["files_scanned"],
        "bytes_scanned": scan["bytes_scanned"],
        "scan_seconds": scan_seconds,
        "catalogue": {"path": R.CATALOGUE_PATH, "sha256": R.sha256_bytes(cat_bytes),
                      "family_slots": len(catalogue.get("family_slots", [])),
                      "candidate_count": catalogue.get("candidate_count"),
                      "excluded_network_roots": len(catalogue.get("excluded_network_roots", []))},
        "resolution": {"method": "GitHub REST (gh api)" if args.resolve == "github" else "none",
                       "keys": len(keys), "unresolved": unresolved},
        "entries": excl.to_json(),
        "not_excluded": not_excluded,
    }
    write_json(Path(args.out), doc)
    if args.scan_out:
        write_json(Path(args.scan_out), scan)
    cats = {}
    for e in doc["entries"]:
        for c in e["categories"]:
            cats[c] = cats.get(c, 0) + 1
    print(json.dumps({"entries": len(doc["entries"]), "categories": cats, "not_excluded": len(not_excluded),
                      "unresolved": len(unresolved), "files_scanned": scan["files_scanned"],
                      "scan_seconds": scan_seconds}, indent=1))
    return 0


def load_exclusions(path: str | os.PathLike) -> R.ExclusionList:
    doc = read_json(Path(path))
    if doc.get("schema") != R.EXCLUSIONS_SCHEMA:
        raise SystemExit(f"{path}: not an {R.EXCLUSIONS_SCHEMA} document")
    return R.ExclusionList.from_json(doc["entries"])


def cmd_vet(args) -> int:
    excl = load_exclusions(args.exclusions)
    names = args.repos or [c["repo"] for c in CANDIDATES]
    cache_path = Path(args.cache) if args.cache else None
    cache = read_json(cache_path) if cache_path and cache_path.exists() else {}
    resolved = resolve_keys([f"name:{n.casefold()}" for n in names], cache, jobs=args.jobs)
    if cache_path:
        write_json(cache_path, cache)
    rows = []
    for n in names:
        res = resolved[f"name:{n.casefold()}"]
        hits = excl.matches(n, res.get("network_root_id"),
                            [x for x in (res.get("full_name"), res.get("network_root_full_name")) if x])
        rows.append({"repo": n, **res, "excluded": bool(hits),
                     "exclusion_matches": [{"family": h.family, "network_root_id": h.network_root_id,
                                            "categories": list(h.categories)} for h in hits]})
    if args.out:
        write_json(Path(args.out), rows)
    for r in rows:
        print(f"{r['repo']:45} root={r.get('network_root_id')} lic={r.get('github_license_guess')} "
              f"lang={r.get('language')} kb={r.get('size_kb')} arch={r.get('archived')} "
              f"excluded={r['excluded']} {[m['family'] for m in r['exclusion_matches']]}")
    return 0


# ------------------------------------------------------------------------------------------------
# clone / pin
# ------------------------------------------------------------------------------------------------

# In-repository files that address coding agents (untrusted repository text; reported to the harness lane).
AGENT_INSTRUCTION_GLOBS = ("AGENTS.md", "**/AGENTS.md", "CLAUDE.md", "**/CLAUDE.md", "GEMINI.md", ".cursorrules",
                           ".cursor/rules/*", ".github/copilot-instructions.md", ".windsurfrules", "llms.txt")

_TAG_FORMAT = ("%(refname:strip=2)%00%(objecttype)%00%(objectname)%00%(*objectname)%00%(*objecttype)%00"
               "%(taggerdate:iso-strict)%00%(committerdate:iso-strict)%00%(*committerdate:iso-strict)")


def list_tags(repo: Path) -> list[dict]:
    """All tags that resolve to a commit: name, peeled commit, commit date, tagger date (annotated)."""
    out = run(["git", "-C", str(repo), "for-each-ref", "refs/tags", f"--format={_TAG_FORMAT}"], check=True).stdout
    tags = []
    for line in out.splitlines():
        parts = line.split("\x00")
        if len(parts) != 8:
            continue
        name, otype, obj, peeled, ptype, tagger, cdate, pdate = parts
        if otype == "tag" and ptype == "commit" and peeled:
            tags.append({"tag": name, "commit": peeled, "commit_date": pdate, "tagger_date": tagger or None})
        elif otype == "tag" and ptype == "tag":  # nested annotated tag: peel fully
            p = run(["git", "-C", str(repo), "rev-parse", "--verify", "--quiet", f"refs/tags/{name}^{{commit}}"])
            sha = p.stdout.strip()
            if p.returncode == 0 and len(sha) == 40:
                tags.append({"tag": name, "commit": sha, "commit_date": commit_date(repo, sha),
                             "tagger_date": tagger or None})
        elif otype == "commit":
            tags.append({"tag": name, "commit": obj, "commit_date": cdate, "tagger_date": None})
    for t in tags:
        t["commit_date"] = utc_iso(t["commit_date"])
        if t["tagger_date"]:
            t["tagger_date"] = utc_iso(t["tagger_date"])
    return tags


def utc_iso(value: str) -> str:
    return R.parse_utc(value).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def commit_date(repo: Path, sha: str) -> str:
    return utc_iso(run(["git", "-C", str(repo), "show", "-s", "--format=%cI", sha], check=True).stdout.strip())


def cmd_clone_one(c: dict) -> dict:
    fam = c["family"]
    wd = work_dir(fam)
    wd.mkdir(parents=True, exist_ok=True)
    log = wd / "clone.log"
    dest = repo_dir(fam)
    url = f"https://github.com/{c['repo']}"
    if not (dest / ".git").exists():
        partial = dest.with_name(dest.name + ".partial")
        if partial.exists():
            shutil.rmtree(partial)
        p = run(["git", "clone", "--quiet", "--no-checkout", url + ".git", str(partial)], log=log, timeout=1800)
        if p.returncode != 0:
            return {"family": fam, "ok": False, "stage": "clone", "error": p.stdout[-2000:]}
        os.replace(partial, dest)
    else:
        run(["git", "-C", str(dest), "fetch", "--quiet", "--tags", "--force", "origin"], log=log, timeout=1800)
    default = run(["git", "-C", str(dest), "symbolic-ref", "--short", "refs/remotes/origin/HEAD"]).stdout.strip()
    tags = list_tags(dest)
    if c.get("pin_tag"):
        chosen = next((t for t in tags if t["tag"] == c["pin_tag"]), None)
        if chosen is None:
            return {"family": fam, "ok": False, "stage": "pin", "error": f"tag {c['pin_tag']} not found"}
        pin = {"tag": chosen["tag"], "commit": chosen["commit"], "commit_date": chosen["commit_date"],
               "tagger_date": chosen.get("tagger_date"), "pin_rule": "explicit_dev_tag"}
    else:
        chosen = R.choose_release_pin(tags, R.CUTOFF)
        if chosen is not None:
            pin = {"tag": chosen["tag"], "commit": chosen["commit"], "commit_date": chosen["commit_date"],
                   "tagger_date": chosen.get("tagger_date"), "pin_rule": "release_tag_before_cutoff"}
        else:
            sha = run(["git", "-C", str(dest), "rev-list", "-1", "--first-parent", f"--before={R.CUTOFF}",
                       default or "origin/HEAD"], check=True).stdout.strip()
            pin = {"tag": None, "commit": sha, "commit_date": commit_date(dest, sha), "tagger_date": None,
                   "pin_rule": "default_branch_head_at_cutoff"}
    run(["git", "-C", str(dest), "-c", "advice.detachedHead=false", "checkout", "--quiet", "--force", "--detach",
         pin["commit"]], log=log, check=True)
    head = run(["git", "-C", str(dest), "rev-parse", "HEAD"], check=True).stdout.strip()
    if head != pin["commit"]:
        return {"family": fam, "ok": False, "stage": "pin", "error": f"HEAD {head} != {pin['commit']}"}
    submodules = []
    if (dest / ".gitmodules").exists() and (dest / ".gitmodules").stat().st_size:
        # SSH submodule URLs (git@github.com:) are fetched over anonymous HTTPS instead.
        run(["git", "-C", str(dest), "-c", "url.https://github.com/.insteadOf=git@github.com:",
             "submodule", "update", "--init", "--recursive", "--quiet"], log=log, timeout=1800, check=True)
        submodules = run(["git", "-C", str(dest), "submodule", "status", "--recursive"], check=True).stdout.splitlines()
    agent_files = sorted({p.relative_to(dest).as_posix() for pat in AGENT_INSTRUCTION_GLOBS for p in dest.glob(pat)
                          if p.is_file() and not p.relative_to(dest).as_posix().startswith(".git/")})
    lic = R.detect_repo_license(dest)
    missing = [d for d in c["package_dirs"] if not (dest / d).exists()]
    stats = R.package_stats(dest, c["package_dirs"]) if not missing else None
    release_tags_total = sum(1 for t in tags if R.release_version(t["tag"]) is not None)
    result = {"family": fam, "ok": not missing and lic["acceptable"], "stage": "clone", "url": url,
              "default_branch": default, **pin, "head_verified": True, "submodules": submodules,
              "agent_instruction_files": agent_files,
              "tags_total": len(tags), "release_tags_total": release_tags_total,
              "license": lic["license"], "license_acceptable": lic["acceptable"], "license_reason": lic["reason"],
              "license_files": lic["files"], "package_dirs": c["package_dirs"], "missing_package_dirs": missing,
              "package_stats": stats, "cloned_at": now_utc()}
    write_json(wd / "pin.json", result)
    return result


# ------------------------------------------------------------------------------------------------
# env / lock
# ------------------------------------------------------------------------------------------------

def uv_env() -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("UV_", "PIP_", "VIRTUAL_ENV", "CONDA"))}
    env["UV_CACHE_DIR"] = str(EXPA_ROOT / "dev" / "repos" / "uv-cache")
    env["UV_PYTHON_DOWNLOADS"] = "never"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def cmd_env_one(c: dict) -> dict:
    fam = c["family"]
    wd = work_dir(fam)
    pin = wd / "pin.json"
    if not pin.exists() or not read_json(pin).get("ok"):
        return {"family": fam, "ok": False, "stage": "env", "error": "no successful pin"}
    log = wd / "install.log"
    if log.exists():
        log.unlink()
    env_path = env_dir(fam)
    src = repo_dir(fam)
    if env_path.exists():
        shutil.rmtree(env_path)
    t0 = time.time()
    uenv = uv_env()
    p = run([UV, "venv", "--quiet", "--python", R.PYTHON_EXECUTABLE, str(env_path)], env=uenv, log=log, timeout=600)
    if p.returncode != 0:
        return _env_fail(fam, "uv venv failed", p.stdout)
    py = str(env_path / "bin" / "python")
    target = f"{src}[{','.join(c['extras'])}]" if c.get("extras") else str(src)
    cmd = [UV, "pip", "install", "--python", py, "--exclude-newer", R.CUTOFF, "-e", target]
    for g in c.get("groups", []):
        cmd += ["--group", f"{src / 'pyproject.toml'}:{g}"]
    for req in c.get("requirements", []):
        cmd += ["-r", str(src / req)]
    cmd += list(c.get("packages", []))
    p = run(cmd, cwd=str(src), env=uenv, log=log, timeout=3600)
    if p.returncode != 0:
        return _env_fail(fam, "uv pip install failed", p.stdout)
    version = run([py, "-c", "import sys; print('%d.%d.%d' % sys.version_info[:3])"], env=uenv).stdout.strip()
    imports = {}
    for name in c.get("import_names", []):
        q = run([py, "-c", f"import {name}, sys; sys.stdout.write({name}.__file__ or '')"], cwd=str(EXPA_ROOT),
                env={**uenv, "PYTHONDONTWRITEBYTECODE": "1"}, timeout=300)
        imports[name] = {"ok": q.returncode == 0, "file": q.stdout.strip()[-300:]}
    bad = [n for n, v in imports.items() if not v["ok"] or not v["file"].startswith(str(src))]
    result = {"family": fam, "ok": not bad, "stage": "env", "env_path": str(env_path), "python": version,
              "python_executable": R.PYTHON_EXECUTABLE, "install_command": [str(x) for x in cmd],
              "exclude_newer": R.CUTOFF, "imports": imports, "seconds": round(time.time() - t0, 1),
              "error": (f"import outside checkout or failed: {bad}" if bad else ""), "built_at": now_utc()}
    write_json(wd / "env.json", result)
    return result


def _env_fail(fam: str, msg: str, output: str) -> dict:
    result = {"family": fam, "ok": False, "stage": "env", "error": msg, "output_tail": output[-3000:],
              "built_at": now_utc()}
    write_json(work_dir(fam) / "env.json", result)
    return result


def freeze(fam: str) -> str:
    """``uv pip freeze`` stdout only (uv's informational banner goes to stderr and is dropped)."""
    py = str(env_dir(fam) / "bin" / "python")
    p = subprocess.run([UV, "pip", "freeze", "--python", py], env=uv_env(), stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, text=True, errors="replace", timeout=300)
    if p.returncode != 0:
        raise RuntimeError(f"uv pip freeze failed for {fam}: {p.stderr[-1000:]}")
    return p.stdout


def cmd_lock_one(c: dict) -> dict:
    fam = c["family"]
    envj = work_dir(fam) / "env.json"
    if not envj.exists() or not read_json(envj).get("ok"):
        return {"family": fam, "ok": False, "stage": "lock", "error": "no successful env"}
    text = freeze(fam)
    path = lock_path(fam)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    result = {"family": fam, "ok": True, "stage": "lock", "path": str(path), "sha256": R.sha256_bytes(text.encode()),
              "packages": lines, "locked_at": now_utc()}
    write_json(work_dir(fam) / "lock.json", result)
    return result


# ------------------------------------------------------------------------------------------------
# baseline (offline)
# ------------------------------------------------------------------------------------------------

BASELINE_REPORT_ARGS = ["-q", "-rfE", "--tb=short", "--color=no"]


def test_command(c: dict) -> list[str]:
    """Recorded test command (argv; ``python`` = the family env interpreter, cwd = repo root)."""
    return ["python", "-m", "pytest", "-p", "no:cacheprovider", "-p", "no:randomly", *c.get("test_args", [])]


def cmd_baseline_one(c: dict) -> dict:
    fam = c["family"]
    wd = work_dir(fam)
    envj = wd / "env.json"
    if not envj.exists() or not read_json(envj).get("ok"):
        return {"family": fam, "ok": False, "stage": "baseline", "error": "no successful env"}
    src = repo_dir(fam)
    env_path = env_dir(fam)
    tmp_root = EXPA_ROOT / "dev" / "repos" / "tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f"{fam}-", dir=tmp_root))
    junit = wd / "junit.xml"
    log = wd / "baseline.log"
    for p in (junit, log):
        if p.exists():
            p.unlink()
    argv = test_command(c)
    argv = [str(env_path / "bin" / "python")] + argv[1:] + BASELINE_REPORT_ARGS + [f"--junitxml={junit}"]
    env = R.scrubbed_test_env(env_path, tmp, c.get("test_env", {}))
    full = R.offline_argv(argv)
    status_before = run(["git", "-C", str(src), "status", "--porcelain=v1"]).stdout.splitlines()
    t0 = time.time()
    timed_out = False
    with open(log, "w", encoding="utf-8") as fh:
        fh.write("$ " + " ".join(full) + "\n")
        fh.flush()
        proc = subprocess.Popen(full, cwd=str(src), env=env, stdout=fh, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            code = proc.wait(timeout=R.WALL_CAP_SECONDS)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                code = proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                code = proc.wait()
    wall = round(time.time() - t0, 1)
    status_after = run(["git", "-C", str(src), "status", "--porcelain=v1"]).stdout.splitlines()
    counts = None
    if junit.exists():
        try:
            counts = R.parse_junit(junit.read_bytes())
        except Exception as exc:  # malformed report: keep the failure visible
            counts = None
            (wd / "junit.error").write_text(str(exc), encoding="utf-8")
    status, reason = R.baseline_status(counts, timed_out)
    tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-25:]
    shutil.rmtree(tmp, ignore_errors=True)
    result = {"family": fam, "ok": True, "stage": "baseline", "status": status, "status_reason": reason,
              "counts": counts, "exit_code": code, "timed_out": timed_out, "wall_seconds": wall,
              "wall_cap_seconds": R.WALL_CAP_SECONDS, "offline": True, "argv": full,
              "env_keys": sorted(env), "junit_sha256": R.sha256_file(junit) if junit.exists() else None,
              "log_path": str(log), "log_tail": tail,
              "worktree_changes": sorted(set(status_after) - set(status_before)), "ran_at": now_utc()}
    write_json(wd / "baseline.json", result)
    return result


def cmd_run_one(c: dict) -> dict:
    steps = {}
    for name, fn in (("clone", cmd_clone_one), ("env", cmd_env_one), ("lock", cmd_lock_one),
                     ("baseline", cmd_baseline_one)):
        try:
            res = fn(c)
        except Exception as exc:  # record and stop this family; other families continue
            res = {"family": c["family"], "ok": False, "stage": name, "error": f"{type(exc).__name__}: {exc}"}
            write_json(work_dir(c["family"]) / f"{name}.exception.json", res)
        steps[name] = {k: res.get(k) for k in ("ok", "status", "error") if k in res}
        if not res.get("ok"):
            break
    if steps.get("baseline", {}).get("ok"):
        try:
            after = R.sha256_bytes(freeze(c["family"]).encode())
            lockj = read_json(work_dir(c["family"]) / "lock.json")
            lockj["unchanged_after_tests"] = after == lockj["sha256"]
            write_json(work_dir(c["family"]) / "lock.json", lockj)
            steps["lock_recheck"] = {"ok": lockj["unchanged_after_tests"]}
        except Exception as exc:
            steps["lock_recheck"] = {"ok": False, "error": str(exc)}
    return {"family": c["family"], "steps": steps}


def selected(args) -> list[dict]:
    """Candidates to process; flagged excluded candidates are never cloned or built."""
    cmap = candidate_map()
    names = args.family or [n for n, c in cmap.items() if not c.get("excluded_candidate")]
    unknown = [n for n in names if n not in cmap]
    if unknown:
        raise SystemExit(f"unknown families: {unknown}")
    flagged = [n for n in names if cmap[n].get("excluded_candidate")]
    if flagged:
        raise SystemExit(f"refusing to build excluded candidates: {flagged}")
    return [cmap[n] for n in names]


def parallel(fn, cands, jobs):
    results = []
    with cf.ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        futs = {pool.submit(fn, c): c["family"] for c in cands}
        for fut in cf.as_completed(futs):
            try:
                res = fut.result()
            except Exception as exc:
                res = {"family": futs[fut], "ok": False, "error": f"{type(exc).__name__}: {exc}"}
            print(json.dumps(res, sort_keys=True)[:600], flush=True)
            results.append(res)
    return results


# ------------------------------------------------------------------------------------------------
# assemble / check
# ------------------------------------------------------------------------------------------------

def build_record(c: dict, excl: R.ExclusionList) -> R.FamilyRecord:
    fam = c["family"]
    wd = work_dir(fam)
    url = f"https://github.com/{c['repo']}"
    hits = excl.matches(c["repo"], c.get("network_root_id"))
    pin = read_json(wd / "pin.json") if (wd / "pin.json").exists() else {}
    envj = read_json(wd / "env.json") if (wd / "env.json").exists() else {}
    lockj = read_json(wd / "lock.json") if (wd / "lock.json").exists() else {}
    base = read_json(wd / "baseline.json") if (wd / "baseline.json").exists() else {}
    rec = R.FamilyRecord(
        family=fam, url=url, commit=pin.get("commit"), commit_date=pin.get("commit_date"),
        license=pin.get("license"), package_dirs=list(c["package_dirs"]), test_command=test_command(c),
        python=envj.get("python"), env_lock_sha256=lockj.get("sha256"), baseline=None, status="env_failed",
        role=None, tag=pin.get("tag"), pin_rule=pin.get("pin_rule"), network_root_id=c.get("network_root_id"),
        license_files=[{k: f[k] for k in ("path", "sha256", "spdx", "non_permissive")} for f in pin.get("license_files", [])],
        package_stats=pin.get("package_stats"), env_path=envj.get("env_path"),
        env_lock=({"path": lockj.get("path"), "sha256": lockj.get("sha256"), "packages": lockj.get("packages"),
                   "unchanged_after_tests": lockj.get("unchanged_after_tests"),
                   "exclude_newer": envj.get("exclude_newer")} if lockj else None),
        test_env=dict(c.get("test_env", {})),
        install={k: c.get(k, []) for k in ("extras", "groups", "requirements", "packages")},
        agent_instruction_files=list(pin.get("agent_instruction_files", [])),
        notes=c.get("notes", ""))
    if hits:
        rec.status = "excluded"
        rec.status_reason = "matches the exclusion list"
        rec.exclusion_matches = [{"family": h.family, "network_root_id": h.network_root_id,
                                  "categories": list(h.categories)} for h in hits]
        return rec
    if not pin.get("ok"):
        rec.status_reason = "clone/pin/licence/package check failed: " + str(
            pin.get("error") or pin.get("license_reason") or pin.get("missing_package_dirs") or "not run")
        return rec
    if not envj.get("ok"):
        rec.status_reason = "environment failed: " + str(envj.get("error") or "not run")
        return rec
    if not base:
        rec.status = "tests_failed"
        rec.status_reason = "baseline not run"
        return rec
    rec.baseline = {k: base.get(k) for k in ("counts", "exit_code", "timed_out", "wall_seconds", "wall_cap_seconds",
                                             "offline", "junit_sha256", "worktree_changes", "ran_at")}
    rec.status, rec.status_reason = base.get("status", "tests_failed"), base.get("status_reason", "")
    return rec


def cmd_assemble(args) -> int:
    excl_doc = read_json(Path(args.exclusions))
    excl = R.ExclusionList.from_json(excl_doc["entries"])
    records = [build_record(c, excl) for c in CANDIDATES]
    for c, r in zip(CANDIDATES, records):
        if bool(c.get("excluded_candidate")) != (r.status == "excluded"):
            raise SystemExit(f"{c['family']}: excluded_candidate flag disagrees with the exclusion list")
    ordered = R.assign_roles(records, n_main=args.n_main)
    doc = {
        "schema": R.SCHEMA,
        "cutoff": R.CUTOFF,
        "generated_at": now_utc(),
        "code": {"repos_py_sha256": R.sha256_file(Path(R.__file__)),
                 "script_sha256": R.sha256_file(Path(__file__)), "git_head": git_head(ROOT)},
        "rules": {"allowed_licenses": sorted(R.ALLOWED_LICENSES), "pin_rules": list(R.PIN_RULES),
                  "baseline": {"min_passed": R.MIN_PASSED, "min_pass_fraction": R.MIN_PASS_FRACTION,
                               "wall_cap_seconds": R.WALL_CAP_SECONDS,
                               "offline": "unshare --user --map-root-user --net (loopback up), scrubbed env"},
                  "roles": {"dev": list(R.DEV_FAMILIES), "n_main": args.n_main, "min_main": args.min_main,
                            "order": f"status; pass fraction >= {R.CLEAN_PASS_FRACTION} first; more passing "
                                     "tests first; family name",
                            "n_main_reason": N_MAIN_REASON},
                  "python_executable": R.PYTHON_EXECUTABLE},
        "exclusions_source": {k: excl_doc.get(k) for k in (
            "schema", "generated_at", "project_git_head", "patterns", "listing_patterns", "non_family_names",
            "files_scanned", "bytes_scanned", "catalogue", "resolution")}
        | {"entries": len(excl_doc["entries"]), "sha256": R.sha256_file(Path(args.exclusions))},
        "exclusions": excl_doc["entries"],
        "exclusions_not_excluded": excl_doc.get("not_excluded", []),
        "rejected_candidates": REJECTED,
        "families": [r.to_json() for r in ordered],
    }
    errs = R.validate_registry(doc, min_main=args.min_main)
    doc["validation"] = {"errors": errs, "valid": not errs}
    write_json(Path(args.out), doc)
    counts = {}
    for r in ordered:
        counts[(r.status, r.role)] = counts.get((r.status, r.role), 0) + 1
    print(json.dumps({"families": len(ordered), "by_status_role": {f"{k[0]}/{k[1]}": v for k, v in counts.items()},
                      "errors": errs[:20]}, indent=1))
    return 1 if errs else 0


def cmd_check(args) -> int:
    doc = read_json(Path(args.registry))
    errs = R.validate_registry(doc, min_main=args.min_main)
    print(json.dumps({"valid": not errs, "errors": errs}, indent=1))
    return 1 if errs else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("exclusions")
    p.add_argument("--project-root", default=str(ROOT))
    p.add_argument("--out", required=True)
    p.add_argument("--scan-out")
    p.add_argument("--cache")
    p.add_argument("--resolve", choices=("github", "none"), default="github")
    p.add_argument("--jobs", type=int, default=8)
    p = sub.add_parser("vet")
    p.add_argument("--exclusions", required=True)
    p.add_argument("--out")
    p.add_argument("--cache")
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("repos", nargs="*")
    sub.add_parser("list")
    for name in ("clone", "env", "lock", "baseline", "run"):
        p = sub.add_parser(name)
        p.add_argument("--family", action="append")
        p.add_argument("--jobs", type=int, default=1)
    p = sub.add_parser("assemble")
    p.add_argument("--exclusions", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-main", type=int, default=N_MAIN)
    p.add_argument("--min-main", type=int, default=R.MIN_MAIN)
    p = sub.add_parser("check")
    p.add_argument("registry")
    p.add_argument("--min-main", type=int, default=R.MIN_MAIN)
    args = ap.parse_args(argv)
    if args.cmd == "exclusions":
        return cmd_exclusions(args)
    if args.cmd == "vet":
        return cmd_vet(args)
    if args.cmd == "list":
        print("\n".join(c["family"] for c in CANDIDATES if not c.get("excluded_candidate")))
        return 0
    if args.cmd in ("clone", "env", "lock", "baseline", "run"):
        fn = {"clone": cmd_clone_one, "env": cmd_env_one, "lock": cmd_lock_one, "baseline": cmd_baseline_one,
              "run": cmd_run_one}[args.cmd]
        results = parallel(fn, selected(args), args.jobs)
        failed = [r for r in results if r.get("ok") is False]
        return 1 if failed else 0
    if args.cmd == "assemble":
        return cmd_assemble(args)
    if args.cmd == "check":
        return cmd_check(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
