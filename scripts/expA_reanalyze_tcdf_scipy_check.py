#!/usr/bin/env python3
"""Cross-check of the Student t functions in scripts/expA_reanalyze_independent.py against scipy.stats.t.

ANALYSIS RECORD. POST-RESULTS verification aid written on 2026-11-01;
not preregistered.

The re-analysis script is stdlib only. This separate check uses scipy and numpy on purpose. It
imports the re-analysis script by path and records that script's sha256, so the re-analysis can
tell whether an embedded cross-check (--tcdf-crosscheck) belongs to its current version.

Checks
  1. CDF on a grid (19 df values x 8011 t values): max abs difference to scipy.stats.t.cdf.
  2. Two-sided p on the same grid: max relative difference to 2 * scipy.stats.t.sf(|t|)
     (points where the scipy value is above 1e-300).
  3. Quantile on a q grid in [0.001, 0.999]: max abs and relative difference to scipy.stats.t.ppf.
  4. The analysis points (df 29): t critical of the 95% intervals and the H1-H3 two-sided p-values,
     against scipy and against a 60-digit decimal reference (DLMF 8.17.8 hypergeometric series of
     the regularized incomplete beta function; pi from the Python decimal-documentation recipe).

Usage (from the repository root)
  python -I -B scripts/expA_reanalyze_tcdf_scipy_check.py --cells PATH/CELLS.jsonl \
      --analysis reports/expA-main-20261101b/analysis/ANALYSIS.json \
      --out reports/expA-main-20261101b/independent/TCDF_SCIPY_CROSSCHECK.json
"""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import math
import os
import shlex
import sys
from decimal import Decimal, localcontext

import numpy as np
import scipy
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(HERE, "expA_reanalyze_independent.py")
LABEL = "ANALYSIS RECORD"
DFS = [1, 2, 3, 4, 5, 7, 10, 15, 20, 25, 28, 29, 30, 40, 60, 100, 200, 500, 1000]
EXTREME_T = [50.0, 100.0, 1e3, 1e4, 1e6]
Q_GRID = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5,
          0.55, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99, 0.995, 0.999]
REF_DIGITS = 60
ANALYSIS_DF = 29


def load_target():
    spec = importlib.util.spec_from_file_location("expA_reanalyze_independent", TARGET)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- 60-digit decimal reference -------------------------------------------------------------


def dec_pi() -> Decimal:
    """pi to the current precision (recipe from the Python decimal documentation)."""
    with localcontext() as ctx:
        ctx.prec += 2
        three = Decimal(3)
        lasts, t, s, n, na, d, da = 0, three, 3, 1, 0, 0, 24
        while s != lasts:
            lasts = s
            n, na = n + na, na + 8
            d, da = d + da, da + 32
            t = (t * n) / d
            s += t
    return +s


def ref_two_sided_p(t: Decimal, df: int) -> Decimal:
    """P(|T| >= |t|) = I_x(df/2, 1/2), x = df/(df+t^2), with
    I_x(a,b) = x^a (1-x)^b / (a B(a,b)) * 2F1(a+b, 1; a+1; x)   (DLMF 8.17.8)."""
    half = Decimal(1) / 2
    a = Decimal(df) / 2
    t2 = t * t
    x = Decimal(df) / (df + t2)
    y = t2 / (df + t2)
    if df % 2 == 0:  # a = m integer: B(m, 1/2) = (m-1)! / prod_{k<m} (k + 1/2)
        m = df // 2
        prod = Decimal(1)
        for k in range(m):
            prod *= Decimal(k) + half
        beta = Decimal(math.factorial(m - 1)) / prod
    else:  # a = m + 1/2: B(m + 1/2, 1/2) = pi * prod_{k<m} (k + 1/2) / m!
        m = (df - 1) // 2
        prod = Decimal(1)
        for k in range(m):
            prod *= Decimal(k) + half
        beta = dec_pi() * prod / Decimal(math.factorial(m))
    front = (a * x.ln()).exp() * y.sqrt() / (a * beta)
    total, term, n = Decimal(0), Decimal(1), 0
    tiny = Decimal(10) ** -(REF_DIGITS + 5)
    while True:
        total += term
        term = term * (a + half + n) / (a + 1 + n) * x
        n += 1
        if term < tiny * total:
            break
    return front * total


def ref_isf_two_sided(p: Decimal, df: int) -> Decimal:
    lo, hi = Decimal(0), Decimal(1)
    while ref_two_sided_p(hi, df) > p:
        lo, hi = hi, hi * 2
    stop = Decimal(10) ** -(REF_DIGITS - 10)
    while hi - lo > stop:
        mid = (lo + hi) / 2
        if ref_two_sided_p(mid, df) > p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ---- main ------------------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="scipy.stats.t cross-check of the re-analysis t functions")
    ap.add_argument("--cells", required=True)
    ap.add_argument("--analysis", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    mod = load_target()
    cells_sha = mod.sha256_file(args.cells)
    if cells_sha != mod.EXPECTED_SHA256["cells"]:
        print(f"REFUSED: CELLS.jsonl sha256 {cells_sha} != expected", file=sys.stderr)
        return 2
    analysis_sha = mod.sha256_file(args.analysis)
    if analysis_sha != mod.EXPECTED_SHA256["analysis"]:
        print(f"REFUSED: ANALYSIS.json sha256 {analysis_sha} != expected", file=sys.stderr)
        return 2

    t_grid = [float(v) for v in np.linspace(-40.0, 40.0, 8001)]
    t_grid += EXTREME_T + [-v for v in EXTREME_T]

    # 1 and 2: CDF and two-sided p on the grid
    cdf_max, cdf_at, per_df_cdf = 0.0, None, {}
    p_rel_max, p_rel_at, p_abs_max = 0.0, None, 0.0
    points = 0
    ts = np.array(t_grid)
    for df in DFS:
        cdf_ref = stats.t.cdf(ts, df)
        p_ref = 2.0 * stats.t.sf(np.abs(ts), df)
        df_max = 0.0
        for t, cr, pr in zip(t_grid, cdf_ref, p_ref):
            points += 1
            d = abs(mod.t_cdf(t, df) - float(cr))
            df_max = max(df_max, d)
            if cdf_at is None or d > cdf_max:
                cdf_max, cdf_at = d, {"df": df, "t": t}
            pm = mod.t_two_sided_p(t, df)
            pr = float(pr)
            p_abs_max = max(p_abs_max, abs(pm - pr))
            if pr > 1e-300:
                r = abs(pm - pr) / pr
                if p_rel_at is None or r > p_rel_max:
                    p_rel_max, p_rel_at = r, {"df": df, "t": t, "this_script": pm, "scipy": pr}
        per_df_cdf[str(df)] = df_max

    # 3: quantile
    ppf_abs_max, ppf_abs_at, ppf_rel_max = 0.0, None, 0.0
    for df in DFS:
        for q in Q_GRID:
            mine = mod.t_ppf(q, df)
            ref = float(stats.t.ppf(q, df))
            d = abs(mine - ref)
            if ppf_abs_at is None or d > ppf_abs_max:
                ppf_abs_max, ppf_abs_at = d, {"df": df, "q": q, "this_script": mine, "scipy": ref}
            if ref != 0.0:
                ppf_rel_max = max(ppf_rel_max, d / abs(ref))

    # 4: analysis points (df 29), recomputed by the re-analysis code itself
    res = mod.recompute(mod.load_jsonl(args.cells))
    with open(args.analysis, "r", encoding="utf-8") as f:
        analysis = json.load(f)
    theirs = {h["id"]: h["t_test"] for h in analysis["primary"]["hypotheses"]}
    with localcontext() as ctx:
        ctx.prec = REF_DIGITS
        tail_double = Decimal(1) - Decimal(0.95)  # two-sided tail of the double 0.95 (= 2 * (1 - double 0.975))
        tcrit_ref = ref_isf_two_sided(tail_double, ANALYSIS_DF)
        tcrit_ref_exact = ref_isf_two_sided(Decimal("0.05"), ANALYSIS_DF)
        mine_tc = res["hypotheses"]["H1"]["t_test"]["t_critical"]
        scipy_ppf = float(stats.t.ppf(0.975, ANALYSIS_DF))
        scipy_isf = float(stats.t.isf(0.025, ANALYSIS_DF))
        theirs_tc = theirs["H1"]["t_critical"]
        t_critical = {
            "two_sided_tail": str(tail_double),
            "reference": str(tcrit_ref),
            "reference_for_exact_tail_0.05": str(tcrit_ref_exact),
            "this_script": mine_tc,
            "scipy_ppf_0.975": scipy_ppf,
            "scipy_isf_0.025": scipy_isf,
            "analysis_json": theirs_tc,
            "this_script_abs_err": float(abs(Decimal(mine_tc) - tcrit_ref)),
            "scipy_ppf_abs_err": float(abs(Decimal(scipy_ppf) - tcrit_ref)),
            "scipy_isf_abs_err": float(abs(Decimal(scipy_isf) - tcrit_ref)),
            "analysis_json_abs_err": float(abs(Decimal(theirs_tc) - tcrit_ref)),
            "this_script_minus_scipy_ppf": mine_tc - scipy_ppf,
        }
        hyp = {}
        for hid, h in res["hypotheses"].items():
            st = h["t_test"]
            t_mine, p_mine = st["t"], st["p_two_sided"]
            p_scipy = float(2.0 * stats.t.sf(abs(t_mine), ANALYSIS_DF))
            ref_mine_t = ref_two_sided_p(Decimal(t_mine), ANALYSIS_DF)
            t_theirs, p_theirs = theirs[hid]["t"], theirs[hid]["p_two_sided"]
            ref_theirs_t = ref_two_sided_p(Decimal(t_theirs), ANALYSIS_DF)
            hyp[hid] = {
                "t_this_script": t_mine,
                "p_reference_at_this_script_t": str(ref_mine_t),
                "p_this_script": p_mine,
                "p_scipy_at_this_script_t": p_scipy,
                "this_script_rel_err": float(abs(Decimal(p_mine) - ref_mine_t) / ref_mine_t),
                "scipy_rel_err": float(abs(Decimal(p_scipy) - ref_mine_t) / ref_mine_t),
                "this_script_vs_scipy_rel_diff": abs(p_mine - p_scipy) / p_scipy,
                "t_analysis_json": t_theirs,
                "p_analysis_json": p_theirs,
                "p_reference_at_analysis_json_t": str(ref_theirs_t),
                "analysis_json_rel_err": float(abs(Decimal(p_theirs) - ref_theirs_t) / ref_theirs_t),
            }

    summary = {
        "cdf_max_abs_diff": cdf_max,
        "cdf_max_abs_diff_at": cdf_at,
        "cdf_points": points,
        "cdf_dfs": DFS,
        "cdf_max_abs_diff_df29": per_df_cdf[str(ANALYSIS_DF)],
        "two_sided_p_max_abs_diff": p_abs_max,
        "two_sided_p_max_rel_diff": p_rel_max,
        "two_sided_p_max_rel_diff_at": p_rel_at,
        "ppf_max_abs_diff": ppf_abs_max,
        "ppf_max_abs_diff_at": ppf_abs_at,
        "ppf_max_rel_diff": ppf_rel_max,
        "t_critical_df29_abs_diff": abs(t_critical["this_script_minus_scipy_ppf"]),
        "hypothesis_p_max_rel_diff": max(v["this_script_vs_scipy_rel_diff"] for v in hyp.values()),
        "reference_60_digits": {
            "t_critical_df29": {k: t_critical[k] for k in (
                "this_script_abs_err", "scipy_ppf_abs_err", "scipy_isf_abs_err", "analysis_json_abs_err")},
            "hypothesis_p_df29_max_rel_err": {
                "this_script": max(v["this_script_rel_err"] for v in hyp.values()),
                "scipy": max(v["scipy_rel_err"] for v in hyp.values()),
                "analysis_json": max(v["analysis_json_rel_err"] for v in hyp.values()),
            },
        },
    }
    argv = list(getattr(sys, "orig_argv", [sys.executable] + sys.argv))
    exe = os.path.basename(argv[0])
    if exe.lower().endswith(".exe"):
        exe = exe[:-4]
    out = {
        "label": LABEL,
        "purpose": "Separate scipy.stats.t cross-check of the stdlib Student t functions in "
                   "scripts/expA_reanalyze_independent.py (POST-RESULTS verification aid, not preregistered).",
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "command": {"cwd": os.getcwd().replace("\\", "/"), "command_line": shlex.join([exe] + argv[1:]),
                    "interpreter": sys.executable},
        "python_version": sys.version.split()[0],
        "scipy_version": scipy.__version__,
        "numpy_version": np.__version__,
        "checked_script": {"path": "scripts/expA_reanalyze_independent.py", "sha256": mod.sha256_file(TARGET)},
        "inputs": {"cells_sha256": cells_sha, "analysis_sha256": analysis_sha},
        "grids": {
            "dfs": DFS,
            "t": "numpy.linspace(-40, 40, 8001) plus +-" + ", +-".join(f"{v:g}" for v in EXTREME_T),
            "q": Q_GRID,
        },
        "summary": summary,
        "cdf_max_abs_diff_by_df": per_df_cdf,
        "analysis_points_df29": {"t_critical": t_critical, "hypotheses": hyp},
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(out, indent=1) + "\n")
    os.replace(tmp, args.out)
    print(json.dumps(summary, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
