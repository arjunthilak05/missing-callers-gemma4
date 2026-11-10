"""Paper figures for Exp A (Absent Edges): presentation of frozen numbers only, no new analysis.

    <figures env python> -B scripts/paper_figures.py --analysis ANALYSIS.json --explore EXPLORE.json \
        --timeline TIMELINE.json [--funnel FUNNEL.json] [--extra NAME=PATH ...] --out DIR [--overwrite] [--no-pin]

Generated draft: a script that draws figures for the paper. It is not a human
review. Every plotted number is read from an input file; FIGURES.json lists each one with its file and key path
("values", the table view of the figures). The only arithmetic is presentation arithmetic (shares from frozen
k/n counts, differences of frozen counts, swarm offsets for dots) plus consistency checks between the inputs;
any failed check stops the script before a figure is written.

Environment: matplotlib is not in the controller env. Use the figures env
(/mnt/kldata/absent-edges-20261018-core2/expA/dev/integrate/envs/fig/bin/python, matplotlib 3.9.2 + numpy).
Without matplotlib the script exits with status 2. Output bytes are deterministic for one matplotlib version
(fixed svg.hashsalt, no PNG Software or SVG Date metadata, no randomness, sorted JSON).

Inputs (sha256 pinned below; --no-pin records "enforced": false instead of refusing a different file):
  --analysis  frozen confirmatory analysis of main-20261101b (schema expA-analysis-v2)
  --explore   exploratory package, post-results (expA-explore-v1; ANALYSIS RECORD)
  --timeline  exploratory visibility timeline, post-results (expA-timeline-v1; same label)
  --funnel    historical exposure audit (exposure-funnel-v1): an earlier, different repair study (D90 + Cap180,
              pvlib). Optional; without it figure 1 has only the block-T panel.
  --extra     NAME=PATH, repeatable: another expA-analysis-v2 ANALYSIS.json (for example the 12B second-model test
              or the 16-response run) drawn as a lighter row under each hypothesis in figure 3. Its family set must
              equal the main run's. Extra rows show the estimate and 95% t interval only; no p-value is printed
              for them (the 16-response run is exploratory, and model differences come from paired intervals,
              not from significance in one run and not the other).

Figures (no in-figure titles; captions live in the paper; panel letters only where a figure has panels):
  fig1_exposure   (a) the historical funnel: graph_query, source_reference_search and bash counts at the
                  offered (model requests whose tool list held the tool), requested, executed and visible stages,
                  D90 + Cap180 combined. (b) Exp A block T, graph_query offered vs recommended (150 episodes per
                  arm, seed 11): shares of episodes that requested graph_query, saw a non-empty graph answer,
                  submitted, and recovered C*, with 95% t intervals over families (marginal).
  fig2_outcomes   block P, every condition: shares of the 2,250 episodes that submitted and listed C*, submitted
                  and did not list C*, or exhausted the 8-response budget (NoAnswer); the C* recovery estimate
                  (family-weighted) with its 95% t interval; model-free reference lines grep_all and c4_hits.
  fig3_contrasts  forest plot of H1-H3: estimate, 95% t interval (marginal), the 30 family differences, dashed
                  lines at +-practically_meaningful_effect, Holm-adjusted p (H1-H3 only), optional extra rows.
  fig4_timeline   block P, one panel per condition: share of episodes submitted by response k and share with the
                  C* site visible by response k under the LOOSE rule (k = 0..8), family-weighted means with 95% t
                  bands; the 8-response cap marked.

Palette (dataviz reference instance, light mode, validated with validate_palette.py): blue #2a78d6 + orange
#eb6834 pass all-pairs (CVD dE 24.7, normal-vision dE 33.6, both >= 3:1); the two-step blue ramp #86b6ef ->
#2a78d6 passes the ordinal checks (light end 2.06:1). Grays are chrome or de-emphasis (#898781, #c3c2b7), never
the only carrier of identity: legends, direct labels and line styles carry it too.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

SCHEMA = "paper-figures-v1"
FIGURES_ENV = "/mnt/kldata/absent-edges-20261018-core2/expA/dev/integrate/envs/fig/bin/python"
PINS = {
    "analysis": "f9017bf657c2472c30867ef8ace33433e4110c5597c340d1e163c279b9839d86",
    "explore": "bf798d8776b1de686eeeda7a39ed316ac74a10d4d0fb865a4b3706f7343c350a",
    "timeline": "3d75b17a0055e1171e93b4118e3dcdfb6606a103a3110e2cf54fb9a1e2ae9ec7",
    "funnel": "9562f9a2141d6987acc47ee74b8c5e7b45f7a2713986ea2f26f3e1cdfbbc0f6e",
}
SCHEMAS = {"analysis": ("schema_version", "expA-analysis-v2"), "explore": ("schema", "expA-explore-v1"),
           "timeline": ("schema", "expA-timeline-v1"), "funnel": ("schema_version", "exposure-funnel-v1")}
REPO_PATHS = {
    "analysis": "reports/expA-main-20261101b/analysis/ANALYSIS.json",
    "explore": "reports/expA-main-20261101b/explore/EXPLORE.json",
    "timeline": "reports/expA-main-20261101b/timeline/TIMELINE.json",
    "funnel": "reports/exposure-funnel-v1/FUNNEL.json",
}

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#ffffff", "#000000", "#333333", "#777777", "#c8c8c8", "#000000"
BLUE, ORANGE, LIGHT_BLUE = "#0072b2", "#d55e00", "#8cc4e6"
EXHAUSTED_FILL = "#d0d0d0"
DOT = "#808080"
EXTRA_MARKERS = ("o", "D", "s", "^")
STEMS = ("fig1_exposure", "fig2_outcomes", "fig3_contrasts", "fig4_timeline")
DPI = 300
WIDTH = 6.27  # printed text width (A4, 1-inch margins); figures are drawn at the size they print
STYLE_DIR = Path(__file__).resolve().parents[1] / "paper" / "figures" / "style"
FONT = "Libertinus Serif"

P_ORDER = ("C0", "C1", "C2", "C3", "C4a")
LABELS = {"C0": "No list", "C1": "Incomplete list", "C2": "List + caution", "C3": "List + not-exhaustive note",
          "C4a": "List + lexical block"}
WRAPPED = {"C0": "No list", "C1": "Incomplete\nlist", "C2": "List +\ncaution", "C3": "List + not-\nexhaustive\nnote",
           "C4a": "List +\nlexical\nblock"}
H_ORDER = ("H1", "H2", "H3")
H_NAMES = {"H1": "Note effect", "H2": "Evidence effect", "H3": "List effect"}
HIST_BLOCK = "D90+Cap180"
HIST_STAGES = (("offered_requests", "offered"), ("requested", "requested"), ("executed", "executed"),
               ("visible", "visible to\nmodel"))
HIST_TOOLS = (("graph_query", BLUE), ("source_reference_search", ORANGE), ("bash", MUTED))
T_ARMS = (("T-offered", LIGHT_BLUE, "graph_query offered"), ("T-recommended", BLUE, "graph_query recommended"))
T_STAGES = (("requested", "requested\ngraph_query"), ("visible", "non-empty graph\nanswer visible"),
            ("submitted", "submitted"), ("recovered", "C* recovered"))


class Book:
    """Collects consistency checks and the table view (every plotted number with its source)."""

    def __init__(self):
        self.checks = []
        self.values = []

    def check(self, ok, what):
        self.checks.append({"check": what, "ok": bool(ok)})
        if not ok:
            raise SystemExit("consistency check failed: " + what)

    def value(self, figure, panel, series, x, value, source, lower=None, upper=None, note=None):
        row = {"figure": figure, "panel": panel, "series": series, "x": x, "value": value, "source": source}
        if lower is not None or upper is not None:
            row["lower"], row["upper"] = lower, upper
        if note:
            row["note"] = note
        self.values.append(row)


def close(a, b, tol=1e-12):
    return a is not None and b is not None and abs(float(a) - float(b)) <= tol


def load(kind, path, enforce):
    data = Path(path).read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    pin = PINS.get(kind)
    if enforce and pin and sha != pin:
        raise SystemExit(f"{kind}: {path} has sha256 {sha}, pinned {pin} (pass --no-pin to draw another file)")
    doc = json.loads(data)
    if kind in SCHEMAS:
        key, want = SCHEMAS[kind]
        if doc.get(key) != want:
            raise SystemExit(f"{kind}: {path} is not a {want} document")
    record = {"path": str(path), "sha256": sha, "bytes": len(data)}
    if pin:
        record.update({"pinned_sha256": pin, "matches_pin": sha == pin, "repo_path": REPO_PATHS[kind]})
    return doc, record


def fmt(value, digits=3, signed=False):
    text = f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"
    return text.replace("-", "−")


def fmt_p(p):
    if p is None:
        return "n/a"
    if p >= 0.001:
        return f"{p:.3f}"
    mantissa, exponent = f"{p:.1e}".split("e")
    return f"${mantissa}\\times10^{{{int(exponent)}}}$"


# ----------------------------------------------------------------------------------------------- data access
def rates_row(analysis, block, condition):
    for row in analysis["secondary"]["rates"]:
        if row["block"] == block and row["condition"] == condition:
            return row
    raise SystemExit(f"ANALYSIS.json has no secondary.rates row for {block}|{condition}")


def explore_row(explore, section, block, condition):
    for row in explore["sections"][section]["rows"]:
        if row["block"] == block and row["condition"] == condition:
            return row
    raise SystemExit(f"EXPLORE.json {section} has no row for {block}|{condition}")


def hypotheses(analysis):
    found = {h["id"]: h for h in analysis["primary"]["hypotheses"]}
    missing = [h for h in H_ORDER if h not in found]
    if missing:
        raise SystemExit(f"ANALYSIS.json lacks hypotheses {missing}")
    return found


def interval(t_interval):
    t_interval = t_interval or {}
    return t_interval.get("lower"), t_interval.get("upper")


def outcomes_data(analysis, explore, book):
    """Block P exit partition and recovery per condition (fig2)."""
    rows = []
    refs = {}
    for cond in P_ORDER:
        r = rates_row(analysis, "P", cond)
        n = r["completed_cells"]
        exits = r["exit_counts"]
        submitted, exhausted = exits["Submitted"], exits["NoAnswer"]
        others = {k: v for k, v in exits.items() if k not in ("Submitted", "NoAnswer") and v}
        book.check(not others, f"P|{cond}: no exit other than Submitted/NoAnswer (ANALYSIS exit_counts)")
        book.check(submitted + exhausted == n == r["cells"], f"P|{cond}: Submitted + NoAnswer = completed = cells")
        e = explore_row(explore, "s2_decomposition", "P", cond)
        recovered = e["overall_recovery"]["micro"]["k"]
        book.check(e["overall_recovery"]["micro"]["n"] == n and e["submission_rate"]["micro"]["k"] == submitted,
                   f"P|{cond}: EXPLORE micro counts match ANALYSIS exit counts")
        sr = r["site_recovered"]
        lo, hi = interval(sr.get("t_interval"))
        book.check(close(recovered / n, sr["estimate"]),
                   f"P|{cond}: micro recovered share equals the family-weighted estimate (balanced block)")
        book.check(close(e["overall_recovery"]["estimate"], sr["estimate"]),
                   f"P|{cond}: EXPLORE overall recovery equals ANALYSIS site_recovered")
        for policy in ("grep_all", "c4_hits"):
            value = r["reference_policies"][policy]["site_recovered"]
            refs.setdefault(policy, value)
            book.check(close(refs[policy], value), f"P|{cond}: reference {policy} equal across conditions")
        rows.append({"condition": cond, "n": n, "submitted": submitted, "exhausted": exhausted,
                     "recovered": recovered, "estimate": sr["estimate"], "lower": lo, "upper": hi})
    find = analysis["findability"]
    book.check(close(refs["grep_all"], find["grep_recovers_omitted"]["family_weighted"]),
               "grep_all reference equals findability.grep_recovers_omitted")
    book.check(close(refs["c4_hits"], find["c4_hits_recovers_omitted"]["family_weighted"]),
               "c4_hits reference equals findability.c4_hits_recovers_omitted")
    refs_counts = {"grep_all": (find["grep_recovers_omitted"]["count"], find["grep_recovers_omitted"]["items"]),
                   "c4_hits": (find["c4_hits_recovers_omitted"]["count"], find["c4_hits_recovers_omitted"]["items"])}
    return rows, refs, refs_counts


def t_block_data(analysis, explore, book):
    """Block T stage shares per arm (fig1b): ANALYSIS for requested and recovered, EXPLORE for the rest."""
    tb = analysis["secondary"]["t_block"]
    s7 = explore["sections"]["s7_T_funnel"]["arms"]
    data = {}
    for arm, _, _ in T_ARMS:
        a, x = tb["arms"][arm], s7[arm]
        n = a["completed"]
        book.check(n == a["cells"] == x["episodes"], f"T|{arm}: completed = cells = EXPLORE episodes")
        req = a["requested_rate"]
        rec = a["site_recovered"]
        vis = x["funnel"]["visible_nonempty_answer"]
        sub = x["submission_rate"]
        book.check(close(req["estimate"], x["funnel"]["requested"]["estimate"])
                   and close(interval(req["t_interval"])[0], x["funnel"]["requested"]["ci95_t"][0])
                   and close(interval(req["t_interval"])[1], x["funnel"]["requested"]["ci95_t"][1]),
                   f"T|{arm}: requested estimate and interval agree (ANALYSIS vs EXPLORE)")
        book.check(close(rec["estimate"], x["site_recovered"]["estimate"])
                   and close(interval(rec["t_interval"])[0], x["site_recovered"]["ci95_t"][0])
                   and close(interval(rec["t_interval"])[1], x["site_recovered"]["ci95_t"][1]),
                   f"T|{arm}: site_recovered estimate and interval agree (ANALYSIS vs EXPLORE)")
        book.check(vis["micro"]["k"] == a["funnel"]["visible_nonempty_answer"],
                   f"T|{arm}: visible non-empty answer count agrees (ANALYSIS funnel vs EXPLORE)")
        exits = rates_row(analysis, "T", arm)["exit_counts"]
        book.check(sub["micro"]["k"] == exits["Submitted"], f"T|{arm}: submitted count agrees with exit_counts")
        req_k = a["funnel"]["requested"]
        rec_k = x["site_recovered"]["micro"]["k"]
        stages = {
            "requested": (req["estimate"], *interval(req["t_interval"]), req_k,
                          f"ANALYSIS.json:secondary.t_block.arms.{arm}.requested_rate"),
            "visible": (vis["estimate"], vis["ci95_t"][0], vis["ci95_t"][1], vis["micro"]["k"],
                        f"EXPLORE.json:sections.s7_T_funnel.arms.{arm}.funnel.visible_nonempty_answer"),
            "submitted": (sub["estimate"], sub["ci95_t"][0], sub["ci95_t"][1], sub["micro"]["k"],
                          f"EXPLORE.json:sections.s7_T_funnel.arms.{arm}.submission_rate"),
            "recovered": (rec["estimate"], *interval(rec["t_interval"]), rec_k,
                          f"ANALYSIS.json:secondary.t_block.arms.{arm}.site_recovered"),
        }
        for stage, (est, lo, hi, k, _) in stages.items():
            book.check(close(est, k / n), f"T|{arm}|{stage}: family-weighted share equals k/n (balanced block)")
        data[arm] = {"n": n, "stages": stages}
    return data


def hist_data(funnel, book):
    block = funnel["funnel"][HIST_BLOCK]
    stages = block["stages"]
    book.check(block["model_calls"] == stages["offered_requests"]["bash"],
               "historical: bash offered in every completed model request")
    book.check(all(stages[s][t] == 0 for s in ("requested", "executed", "visible")
                   for t in ("graph_query", "source_reference_search")),
               "historical: graph_query and source_reference_search are 0 at requested, executed and visible")
    return block


def timeline_data(timeline, analysis, explore, book):
    book.check(timeline["budget"] == 8, "TIMELINE budget is 8 responses")
    out = {}
    for cond in P_ORDER:
        group = timeline["groups"][f"P|{cond}"]
        cum = group["cumulative_by_k"]
        ks = sorted(int(k) for k in cum)
        book.check(ks == list(range(9)), f"P|{cond}: cumulative_by_k has k = 0..8")
        series = {}
        for key in ("submitted_by_k", "cstar_visible_by_k_loose"):
            weighted = [cum[str(k)][key]["family_weighted"] for k in ks]
            series[key] = [(w["mean"], w["lower"], w["upper"]) for w in weighted]
        sub8 = series["submitted_by_k"][-1][0]
        sub_explore = explore_row(explore, "s2_decomposition", "P", cond)["submission_rate"]["estimate"]
        book.check(close(sub8, sub_explore), f"P|{cond}: TIMELINE submitted by k=8 equals EXPLORE submission rate")
        rec8 = cum["8"]["submitted_with_cstar_recovered_by_k"]["family_weighted"]["mean"]
        book.check(close(rec8, rates_row(analysis, "P", cond)["site_recovered"]["estimate"]),
                   f"P|{cond}: TIMELINE recovered by k=8 equals ANALYSIS site_recovered")
        out[cond] = series
    return out


# ----------------------------------------------------------------------------------------------- drawing
def _pyplot():
    try:
        import matplotlib
    except ModuleNotFoundError:
        sys.stderr.write("paper_figures.py needs matplotlib; run it with the figures env: " + FIGURES_ENV + "\n")
        raise SystemExit(2)
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import font_manager
    for font in sorted((STYLE_DIR / "fonts").glob("*.otf")):
        font_manager.fontManager.addfont(str(font))
    plt.style.use(str(STYLE_DIR / "paper.mplstyle"))
    plt.rcParams.update({"svg.hashsalt": SCHEMA})
    return matplotlib, plt, np


def _blend(ax):
    from matplotlib.transforms import blended_transform_factory
    return blended_transform_factory(ax.transAxes, ax.transData)


def _ylabels_left(fig, ax):
    """Figure x of the leftmost rendered y tick label of ax (the axes' left edge when it has none)."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    labels = [t for t in ax.get_yticklabels() if t.get_text()]
    x0 = min(t.get_window_extent(renderer).x0 for t in labels) if labels else ax.get_window_extent(renderer).x0
    return fig.transFigure.inverted().transform((x0, 0))[0]


def _xlabel_bottom(fig, ax):
    """Figure y of the bottom of ax's rendered x-axis label."""
    fig.canvas.draw()
    box = ax.xaxis.label.get_window_extent(fig.canvas.get_renderer())
    return fig.transFigure.inverted().transform((0, box.y0))[1]


def _panel_label(fig, ax, text, top_pad=0.012):
    """Panel label left-aligned with the panel's y tick labels, just above the axes."""
    fig.text(_ylabels_left(fig, ax), ax.get_position().y1 + top_pad, text, ha="left", va="bottom", color=INK,
             fontsize=9.5, linespacing=1.3)


def fig_exposure(t_data, hist, plt, np, book):
    fig = plt.figure(figsize=(WIDTH, 3.75))
    if hist is not None:
        ax_h = fig.add_axes((0.125, 0.30, 0.31, 0.53))
        ax_t = fig.add_axes((0.655, 0.30, 0.325, 0.53))
    else:
        ax_h, ax_t = None, fig.add_axes((0.30, 0.30, 0.66, 0.53))
    if ax_h is not None:
        stages = hist["stages"]
        top = max(stages[s][t] for s, _ in HIST_STAGES for t, _ in HIST_TOOLS)
        span = top * 1.36
        height = 0.25
        for i, (stage, _) in enumerate(HIST_STAGES):
            for j, (tool, color) in enumerate(HIST_TOOLS):
                y = i + (j - 1) * height
                v = stages[stage][tool]
                book.value("fig1_exposure", "a", tool, stage, v,
                           f"FUNNEL.json:funnel.{HIST_BLOCK}.stages.{stage}.{tool}",
                           note="offered = completed model requests whose tool list held the tool; other stages = "
                                "tool calls or results" if stage == "offered_requests" else None)
                if v > 0:
                    ax_h.barh(y, v, height=height * 0.84, color=color, linewidth=0, zorder=2)
                    ax_h.text(v + span * 0.015, y, f"{v:,}", va="center", ha="left", fontsize=8.5, color=INK2)
                else:
                    ax_h.plot([0, span * 0.012], [y, y], color=color, linewidth=3.2, solid_capstyle="butt", zorder=2)
                    ax_h.text(span * 0.03, y, "0", va="center", ha="left", fontsize=8.5, color=INK)
        ax_h.tick_params(axis="y", length=0)
        ax_h.set_yticks(range(len(HIST_STAGES)))
        ax_h.set_yticklabels([label for _, label in HIST_STAGES])
        ax_h.set_ylim(len(HIST_STAGES) - 0.5, -0.55)
        ax_h.set_xlim(0, span)
        ticks = [v for v in range(0, int(top) + 1, 2000)]
        ax_h.set_xticks(ticks)
        ax_h.set_xticklabels([f"{v:,}" for v in ticks])
        ax_h.xaxis.grid(True)
        ax_h.set_axisbelow(True)
        ax_h.set_xlabel("count (offered: model requests;\nother stages: tool calls or results)", fontsize=9)
        book.value("fig1_exposure", "a", "episodes", HIST_BLOCK, hist["episodes"],
                   f"FUNNEL.json:funnel.{HIST_BLOCK}.episodes")
        book.value("fig1_exposure", "a", "model requests", HIST_BLOCK, hist["model_calls"],
                   f"FUNNEL.json:funnel.{HIST_BLOCK}.model_calls")
    height = 0.36
    for i, (stage, _) in enumerate(T_STAGES):
        for j, (arm, color, name) in enumerate(T_ARMS):
            y = i + (j - 0.5) * height
            est, lo, hi, k, source = t_data[arm]["stages"][stage]
            n = t_data[arm]["n"]
            degenerate = lo is None or hi is None or hi <= lo
            book.value("fig1_exposure", "b", arm, stage, est, source, lo, hi,
                       note=f"{k}/{n} episodes" + ("; zero-variance interval (degenerate), not drawn"
                                                   if degenerate else ""))
            ax_t.barh(y, est, height=height * 0.84, color=color, linewidth=0, zorder=2)
            end = est
            if not degenerate:
                ax_t.plot([lo, hi], [y, y], color=INK, linewidth=1.3, solid_capstyle="butt", zorder=3)
                end = max(est, hi)
            ax_t.text(end + 0.025, y, f"{k}/{n}", va="center", ha="left", fontsize=8.5, color=INK2)
    ax_t.tick_params(axis="y", length=0)
    ax_t.set_yticks(range(len(T_STAGES)))
    ax_t.set_yticklabels([label for _, label in T_STAGES])
    ax_t.set_ylim(len(T_STAGES) - 0.5, -0.55)
    ax_t.set_xlim(0, 1.34)
    ax_t.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_t.set_xticklabels(["0", "0.25", "0.50", "0.75", "1"])
    ax_t.xaxis.grid(True)
    ax_t.set_axisbelow(True)
    ax_t.set_xlabel("share of episodes (library-weighted)", fontsize=9)
    legends = []
    if ax_h is not None:
        legends.append((ax_h, [plt.Rectangle((0, 0), 1, 1, color=c, linewidth=0) for _, c in HIST_TOOLS],
                        [t for t, _ in HIST_TOOLS]))
    legends.append((ax_t, [plt.Rectangle((0, 0), 1, 1, color=c, linewidth=0) for _, c, _ in T_ARMS]
                    + [plt.Line2D([], [], color=INK, linewidth=1.3)],
                    [name for _, _, name in T_ARMS] + ["95% t interval"]))
    legend_top = min(_xlabel_bottom(fig, ax) for ax, _, _ in legends) - 0.035
    for ax, handles, labels in legends:
        ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(_ylabels_left(fig, ax), legend_top),
                  bbox_transform=fig.transFigure, ncol=2, fontsize=8.5, handlelength=1.2, columnspacing=1.0,
                  borderaxespad=0)
    if ax_h is not None:
        _panel_label(fig, ax_h, "(a) Earlier repair study (pvlib)\n"
                                f"{hist['episodes']} episodes")
    _panel_label(fig, ax_t, ("(b) " if ax_h is not None else "") + "Tool-choice arm (exploratory)\n"
                            f"seed 11, {t_data['T-offered']['n']} episodes per arm")
    return fig


def fig_outcomes(rows, refs, refs_counts, plt, np, book):
    """Horizontal stacked exits per condition (bars), C* recovery with its interval (dots) and the model-free
    reference recoveries (vertical lines); condition names read on one line."""
    fig, ax = plt.subplots(figsize=(WIDTH, 3.7))
    fig.subplots_adjust(left=0.29, right=0.92, top=0.84, bottom=0.13)
    y = np.arange(len(rows))[::-1]
    height = 0.62
    n = np.array([r["n"] for r in rows], dtype=float)
    listed = np.array([r["recovered"] for r in rows]) / n
    missed = np.array([r["submitted"] - r["recovered"] for r in rows]) / n
    exhausted = np.array([r["exhausted"] for r in rows]) / n
    style = {"height": height, "edgecolor": INK, "linewidth": 0.5, "zorder": 2}
    ax.barh(y, listed, color=BLUE, **style)
    ax.barh(y, missed, left=listed, color=ORANGE, **style)
    ax.barh(y, exhausted, left=listed + missed, color=EXHAUSTED_FILL, **style)
    for i, r in enumerate(rows):
        cond = r["condition"]
        base = f"ANALYSIS.json:secondary.rates[P|{cond}]"
        book.value("fig2_outcomes", "", "submitted, C* listed", cond, float(listed[i]),
                   f"EXPLORE.json:sections.s2_decomposition.rows[P|{cond}].overall_recovery.micro",
                   note=f"{r['recovered']}/{r['n']} episodes")
        book.value("fig2_outcomes", "", "submitted, C* not listed", cond, float(missed[i]),
                   f"{base}.exit_counts.Submitted minus the recovered count",
                   note=f"{r['submitted'] - r['recovered']}/{r['n']} episodes")
        book.value("fig2_outcomes", "", "budget exhausted (NoAnswer)", cond, float(exhausted[i]),
                   f"{base}.exit_counts.NoAnswer", note=f"{r['exhausted']}/{r['n']} episodes")
        book.value("fig2_outcomes", "", "C* recovery estimate", cond, r["estimate"],
                   f"{base}.site_recovered (estimate, t_interval)", r["lower"], r["upper"])
        ax.plot([r["lower"], r["upper"]], [y[i], y[i]], color=INK, linewidth=1.5, solid_capstyle="butt", zorder=4)
        ax.plot(r["estimate"], y[i], "o", markersize=6, markerfacecolor=INK, markeredgecolor=SURFACE,
                markeredgewidth=1.4, zorder=5)
        ax.text(0.015, y[i], f"{r['estimate']:.1%}", ha="left", va="center", color="#ffffff", fontsize=8.5,
                fontweight="bold", zorder=5)
        ax.text(1.015, y[i], f"{exhausted[i]:.0%}", ha="left", va="center", color=INK2, fontsize=7.8,
                zorder=5, clip_on=False)
    styles = {"grep_all": ((0, (5, 2.5)), "grep"), "c4_hits": ((0, (1.2, 1.6)), "lexical block")}
    for policy, (dash, short) in styles.items():
        value = refs[policy]
        count, items = refs_counts[policy]
        ax.axvline(value, color=INK, linewidth=1.1, linestyle=dash, zorder=3)
        book.value("fig2_outcomes", "", f"model-free reference {policy}", "P", value,
                   f"ANALYSIS.json:secondary.rates[P|*].reference_policies.{policy}.site_recovered",
                   note=f"{count}/{items} items (findability)")
    ax.tick_params(axis="y", length=0)
    ax.set_yticks(y)
    ax.set_yticklabels([LABELS[r["condition"]] for r in rows])
    ax.set_ylim(-0.5, len(rows) - 0.5)
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_xlabel(f"share of episodes ({int(n[0]):,} per condition)")
    ax.text(1.015, len(rows) - 0.5, "ran out", ha="left", va="bottom", color=MUTED, fontsize=7.2,
            clip_on=False)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=BLUE, edgecolor=INK, linewidth=0.5),
               plt.Rectangle((0, 0), 1, 1, facecolor=ORANGE, edgecolor=INK, linewidth=0.5),
               plt.Rectangle((0, 0), 1, 1, facecolor=EXHAUSTED_FILL, edgecolor=INK, linewidth=0.5),
               plt.Line2D([], [], color=INK, linewidth=1.5, marker="o", markersize=5.5, markerfacecolor=INK,
                          markeredgecolor=SURFACE, markeredgewidth=1.2),
               plt.Line2D([], [], color=INK, linewidth=1.1, linestyle=(0, (5, 2.5))),
               plt.Line2D([], [], color=INK, linewidth=1.1, linestyle=(0, (1.2, 1.6)))]
    labels = ["submitted, C* listed", "submitted, C* missed", "ran out of responses", "recovery, 95% t interval",
              f"grep finds C* ({refs['grep_all']:.0%})", f"lexical block holds C* ({refs['c4_hits']:.0%})"]
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.02, 0.995), ncol=3, fontsize=7.8,
               handlelength=1.5, columnspacing=1.2, borderaxespad=0)
    return fig


def _swarm(values, min_dx, step):
    """Deterministic beeswarm offsets: each dot takes the smallest free offset 0, +s, -s, +2s, ..."""
    order = sorted(range(len(values)), key=lambda i: (values[i], i))
    placed, out = [], [0.0] * len(values)
    candidates = [0.0] + [sign * m * step for m in range(1, 40) for sign in (1, -1)]
    for i in order:
        for off in candidates:
            if all(abs(values[i] - px) >= min_dx or abs(off - po) >= step * 0.999 for px, po in placed):
                break
        placed.append((values[i], off))
        out[i] = off
    return out


def fig_contrasts(analysis, extras, plt, np, book):
    hyps = hypotheses(analysis)
    threshold = analysis["primary"].get("practically_meaningful_effect", 0.10)
    n_extra = len(extras)
    gap = 1.0 + 0.36 * n_extra
    rows_h = 0.98 + 0.38 * n_extra
    legend_rows = -(-(3 + n_extra) // 2)
    bottom_in, top_in = 0.57 + 0.2 * legend_rows, 0.12
    fig = plt.figure(figsize=(WIDTH, bottom_in + 3 * rows_h + 0.43 + top_in))
    height_in = fig.get_figheight()
    ax = fig.add_axes((0.315, bottom_in / height_in, 0.405, 1 - (bottom_in + top_in) / height_in))
    ys = [-(i * gap) for i in range(len(H_ORDER))]
    lows, highs, diffs_all = [], [], []
    for hid in H_ORDER:
        h = hyps[hid]
        lo, hi = interval((h.get("t_test") or {}).get("interval"))
        lows.append(lo)
        highs.append(hi)
        diffs_all += [u["difference"] for u in h.get("family_units", [])]
    for ex in extras:
        for hid in H_ORDER:
            lo, hi = interval((ex["hyps"][hid].get("t_test") or {}).get("interval"))
            if lo is not None:
                lows.append(lo)
                highs.append(hi)
    xmin = min([-threshold] + [v for v in lows if v is not None] + diffs_all) - 0.035
    xmax = max([threshold] + [v for v in highs if v is not None] + diffs_all) + 0.035
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ys[-1] - 0.42 - 0.36 * n_extra, ys[0] + 0.72)
    ax.axvline(0.0, color=AXIS, linewidth=1.0, zorder=1)
    for value in (-threshold, threshold):
        ax.axvline(value, color=MUTED, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
    axes_w_in = ax.get_position().width * WIDTH
    dot_d_in = 4.2 / 72.0
    min_dx = dot_d_in / axes_w_in * (xmax - xmin)
    trans = _blend(ax)
    for y, hid in zip(ys, H_ORDER):
        h = hyps[hid]
        n_fam = h.get("n_families")
        diffs = [u["difference"] for u in h.get("family_units", [])]
        book.check(len(diffs) == n_fam, f"{hid}: family_units count equals n_families")
        if diffs:
            book.check(close(sum(diffs) / len(diffs), h["estimate"], 1e-12),
                       f"{hid}: mean of family differences equals the estimate")
        offsets = _swarm(diffs, min_dx, 0.07)
        ax.scatter(diffs, [y + 0.36 + o for o in offsets], s=17, color=DOT, edgecolors=SURFACE, linewidths=0.6,
                   zorder=2)
        lo, hi = interval((h.get("t_test") or {}).get("interval"))
        ax.plot([lo, hi], [y, y], color=BLUE, linewidth=3.0, solid_capstyle="round", zorder=3)
        ax.plot(h["estimate"], y, "o", markersize=8.5, markerfacecolor=BLUE, markeredgewidth=0, zorder=4)
        t, c = h["treatment"], h["control"]
        ax.text(-0.035, y + 0.36, f"{H_NAMES[hid]} ({hid})", transform=trans, ha="right", va="center", color=INK,
                fontsize=9.5, fontweight="bold")
        ax.text(-0.035, y - 0.06, f"{LABELS[t]}\n− {LABELS[c]}", transform=trans, ha="right", va="center",
                color=INK2, fontsize=8.5, linespacing=1.25)
        ax.text(1.035, y + 0.36, f"Holm p = {fmt_p(h.get('holm_adjusted_p'))}", transform=trans, ha="left",
                va="center", color=INK2, fontsize=9)
        ax.text(1.035, y, f"{fmt(100 * h['estimate'], 1, signed=True)} [{fmt(100 * lo, 1)}, {fmt(100 * hi, 1)}]",
                transform=trans,
                ha="left", va="center", color=INK, fontsize=9)
        source = f"ANALYSIS.json:primary.hypotheses[{hid}]"
        book.value("fig3_contrasts", "", f"{hid} estimate", f"{t}-{c}", h["estimate"],
                   f"{source} (estimate, t_test.interval)", lo, hi)
        book.value("fig3_contrasts", "", f"{hid} Holm-adjusted p", f"{t}-{c}", h.get("holm_adjusted_p"),
                   f"{source}.holm_adjusted_p")
        book.value("fig3_contrasts", "", f"{hid} family differences", f"{t}-{c}",
                   [{"family": u["family"], "difference": u["difference"]} for u in h.get("family_units", [])],
                   f"{source}.family_units[*].difference")
        for j, ex in enumerate(extras):
            eh = ex["hyps"][hid]
            yy = y - 0.36 * (j + 1)
            elo, ehi = interval((eh.get("t_test") or {}).get("interval"))
            if eh.get("estimate") is None or elo is None:
                ax.text(0.0, yy, f"{ex['name']}: not estimable", ha="center", va="center", color=INK2, fontsize=8.5)
                book.value("fig3_contrasts", ex["name"], f"{hid} estimate", f"{t}-{c}", None,
                           f"{ex['file']}:primary.hypotheses[{hid}]", note="not estimable")
                continue
            ax.plot([elo, ehi], [yy, yy], color=LIGHT_BLUE, linewidth=2.6, solid_capstyle="round", zorder=3)
            ax.plot(eh["estimate"], yy, EXTRA_MARKERS[j % len(EXTRA_MARKERS)], markersize=7.5,
                    markerfacecolor=LIGHT_BLUE, markeredgewidth=0, zorder=4)
            name = ex["name"] if len(ex["name"]) <= 24 else ex["name"][:23] + "…"
            ax.text(1.035, yy, f"{name}\n{fmt(100 * eh['estimate'], 1, signed=True)} [{fmt(100 * elo, 1)}, "
                    f"{fmt(100 * ehi, 1)}]",
                    transform=trans, ha="left", va="center", color=INK2, fontsize=8.5, linespacing=1.15)
            book.value("fig3_contrasts", ex["name"], f"{hid} estimate", f"{t}-{c}", eh["estimate"],
                       f"{ex['file']}:primary.hypotheses[{hid}] (estimate, t_test.interval)", elo, ehi)
    ax.set_yticks([])
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    step = 0.1
    ticks = [round(step * k, 10) for k in range(int(np.floor(xmin / step)), int(np.ceil(xmax / step)) + 1)
             if xmin <= step * k <= xmax]
    ax.set_xticks(ticks)
    ax.set_xticklabels([fmt(100 * v, 0, signed=v != 0) if v != 0 else "0" for v in ticks])
    ax.set_xlabel("difference in C* recovery, points (library-weighted)")
    handles = [plt.Line2D([], [], color=BLUE, linewidth=3.0, marker="o", markersize=7, markerfacecolor=BLUE,
                          markeredgewidth=0),
               plt.Line2D([], [], color=DOT, marker="o", linestyle="", markersize=5, markeredgecolor=SURFACE),
               plt.Line2D([], [], color=MUTED, linewidth=0.9, linestyle=(0, (4, 3)))]
    labels = ["estimate, 95% t interval (marginal)",
              f"library difference ({hyps['H1'].get('n_families')} libraries)",
              f"±{100 * threshold:.0f} points: prespecified meaningful effect"]
    for j, ex in enumerate(extras):
        marker = EXTRA_MARKERS[j % len(EXTRA_MARKERS)]
        handles.append(plt.Line2D([], [], color=LIGHT_BLUE, linewidth=2.6, marker=marker, markersize=6,
                                  markerfacecolor=LIGHT_BLUE, markeredgewidth=0))
        labels.append(f"{ex['name']} (lighter rows)")
    pos = ax.get_position()
    axes_h_in, axes_w_in = pos.height * fig.get_figheight(), pos.width * WIDTH
    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=((0.12 - pos.x0 * WIDTH) / axes_w_in,
                                                                  -0.52 / axes_h_in),
              ncol=2, fontsize=8.5, handlelength=1.8, columnspacing=1.4, borderaxespad=0)
    return fig


def fig_timeline(series, plt, np, book):
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 4.9), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.105, right=0.975, top=0.895, bottom=0.105, wspace=0.14, hspace=0.62)
    ks = np.arange(9)
    for idx, cond in enumerate(P_ORDER):
        ax = axes.flat[idx]
        s = series[cond]
        for key, color, style, width, alpha, z in (
                ("cstar_visible_by_k_loose", INK2, (0, (4, 2)), 1.7, 0.10, 3),
                ("submitted_by_k", BLUE, "-", 2.0, 0.16, 4)):
            mean = np.array([v[0] for v in s[key]])
            lo = np.clip(np.array([v[1] for v in s[key]]), 0, 1)
            hi = np.clip(np.array([v[2] for v in s[key]]), 0, 1)
            ax.fill_between(ks, lo, hi, color=color, alpha=alpha, linewidth=0, zorder=z - 2)
            ax.plot(ks, mean, color=color, linewidth=width, linestyle=style, zorder=z,
                    marker="o" if key == "submitted_by_k" else None, markersize=3.2)
            ax.text(8.35, mean[-1], f"{mean[-1]:.2f}", ha="left", va="center", fontsize=8.5, color=INK)
            for k in ks:
                book.value("fig4_timeline", cond, key, int(k), s[key][k][0],
                           f"TIMELINE.json:groups.P|{cond}.cumulative_by_k.{k}.{key}.family_weighted",
                           s[key][k][1], s[key][k][2])
        ax.axvline(8, color=MUTED, linewidth=1.0, zorder=1)
        ax.set_title(LABELS[cond], loc="left", fontsize=9.5, pad=5, linespacing=1.2)
        ax.set_xlim(-0.35, 9.75)
        ax.set_ylim(0, 1.0)
        ax.set_xticks([0, 2, 4, 6, 8])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0", "0.25", "0.50", "0.75", "1"])
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
    axes[0][2].tick_params(labelbottom=True)
    for ax in (axes[1][0], axes[1][1], axes[0][2]):
        ax.set_xlabel("model responses used (k)", fontsize=9)
    for ax in (axes[0][0], axes[1][0]):
        ax.set_ylabel("share of episodes")
    key_ax = axes[1][2]
    key_ax.axis("off")
    handles = [plt.Line2D([], [], color=BLUE, linewidth=2.0, marker="o", markersize=3.2),
               plt.Line2D([], [], color=INK2, linewidth=1.7, linestyle=(0, (4, 2))),
               plt.Rectangle((0, 0), 1, 1, color=INK2, alpha=0.14, linewidth=0),
               plt.Line2D([], [], color=MUTED, linewidth=1.0)]
    labels = ["submitted by response k", "C* site visible by\nresponse k (LOOSE rule)",
              "95% t interval over\nfamilies (marginal)", "8-response cap"]
    key_ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(-0.02, 1.12), fontsize=8.5, handlelength=1.8,
                  labelspacing=0.75, borderaxespad=0)
    key_ax.text(0.0, -0.02, "Exploratory (post-results);\nLOOSE can overcount.",
                transform=key_ax.transAxes, ha="left", va="bottom", fontsize=8.5, color=INK2, linespacing=1.3)
    return fig


def _save(fig, out, stem, overwrite, plt):
    written = {}
    for ext, metadata in (("png", {"Software": None}), ("svg", {"Date": None}), ("pdf", {"CreationDate": None})):
        path = out / f"{stem}.{ext}"
        if path.exists() and not overwrite:
            raise SystemExit(f"refusing to overwrite {path} (pass --overwrite)")
        fig.savefig(path, format=ext, dpi=DPI, metadata=metadata)
        written[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    plt.close(fig)
    return written


def _extra(spec, main, enforce_unused):
    if "=" not in spec:
        raise SystemExit(f"--extra wants NAME=PATH, got {spec!r}")
    name, path = spec.split("=", 1)
    name = name.strip()
    if not name:
        raise SystemExit(f"--extra {spec!r} has an empty NAME")
    doc, record = load("extra", path, enforce_unused)
    if doc.get("schema_version") != "expA-analysis-v2":
        raise SystemExit(f"--extra {name}: {path} is not an expA-analysis-v2 document")
    if sorted(doc["population"]["families"]) != sorted(main["population"]["families"]):
        raise SystemExit(f"--extra {name}: family set differs from the main analysis")
    hyps = hypotheses(doc)
    main_h = hypotheses(main)
    for hid in H_ORDER:
        if (hyps[hid]["treatment"], hyps[hid]["control"]) != (main_h[hid]["treatment"], main_h[hid]["control"]):
            raise SystemExit(f"--extra {name}: {hid} is a different contrast")
    record.update({"name": name, "label": doc.get("label"), "status": doc.get("status"),
                   "primary_seeds": doc["primary"].get("primary_seeds"),
                   "drawn": "estimate and 95% t interval per hypothesis; no p-value printed"})
    return {"name": name, "file": Path(path).name, "hyps": hyps, "record": record}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--explore", required=True)
    parser.add_argument("--timeline", required=True)
    parser.add_argument("--funnel", help="historical exposure audit FUNNEL.json (optional)")
    parser.add_argument("--extra", action="append", default=[], metavar="NAME=PATH",
                        help="another expA-analysis-v2 ANALYSIS.json drawn lighter in figure 3 (repeatable)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-pin", action="store_true", help="draw inputs whose sha256 differs from the pins")
    args = parser.parse_args(argv)
    enforce = not args.no_pin
    matplotlib, plt, np = _pyplot()
    book = Book()
    analysis, rec_a = load("analysis", args.analysis, enforce)
    explore, rec_e = load("explore", args.explore, enforce)
    timeline, rec_t = load("timeline", args.timeline, enforce)
    inputs = {"analysis": rec_a, "explore": rec_e, "timeline": rec_t}
    book.check(explore["inputs"]["analysis"]["sha256"] == rec_a["sha256"],
               "EXPLORE.json was computed from this ANALYSIS.json (inputs.analysis.sha256)")
    book.check(timeline["inputs"]["scores"]["sha256"] == analysis["inputs"]["scores"]["sha256"]
               and timeline["inputs"]["items"]["sha256"] == analysis["inputs"]["items"]["sha256"],
               "TIMELINE.json and ANALYSIS.json read the same scores and items (sha256)")
    hist = None
    if args.funnel:
        funnel, rec_f = load("funnel", args.funnel, enforce)
        inputs["funnel"] = rec_f
        hist = hist_data(funnel, book)
    extras = [_extra(spec, analysis, False) for spec in args.extra]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "FIGURES.json"
    names = ["FIGURES.json"] + [f"{stem}.{ext}" for stem in STEMS for ext in ("png", "svg", "pdf")]
    existing = [name for name in names if (out / name).exists()]
    if existing and not args.overwrite:
        raise SystemExit(f"refusing to overwrite {', '.join(existing)} in {out} (pass --overwrite)")

    rows, refs, refs_counts = outcomes_data(analysis, explore, book)
    t_data = t_block_data(analysis, explore, book)
    series = timeline_data(timeline, analysis, explore, book)

    written = {}
    written.update(_save(fig_exposure(t_data, hist, plt, np, book), out, "fig1_exposure", args.overwrite, plt))
    written.update(_save(fig_outcomes(rows, refs, refs_counts, plt, np, book), out, "fig2_outcomes",
                         args.overwrite, plt))
    written.update(_save(fig_contrasts(analysis, extras, plt, np, book), out, "fig3_contrasts", args.overwrite,
                         plt))
    written.update(_save(fig_timeline(series, plt, np, book), out, "fig4_timeline", args.overwrite, plt))

    script = Path(__file__).resolve()
    record = {
        "schema_version": SCHEMA,
        "label": "draft",
        "note": "Figures drawn by a script from frozen inputs: presentation only, no new "
                "analysis EXPLORE.json and TIMELINE.json are ANALYSIS RECORD"
                ", and EXPLORATORY (designed after the 31B results were seen, ~06:05 UTC 2026-11-01).",
        "script": {"name": script.name, "repo_path": "scripts/paper_figures.py",
                   "sha256": hashlib.sha256(script.read_bytes()).hexdigest()},
        "environment": {"python": platform.python_version(), "matplotlib": matplotlib.__version__,
                        "numpy": np.__version__, "dpi": DPI, "font": FONT},
        "inputs": inputs,
        "pins": {"enforced": enforce, "pinned": PINS},
        "extras": [ex["record"] for ex in extras],
        "checks": book.checks,
        "checks_passed": sum(c["ok"] for c in book.checks),
        "outputs": written,
        "figures": {
            "fig1_exposure": {
                "files": ["fig1_exposure.png", "fig1_exposure.svg"],
                "panel_a": None if hist is None else
                f"historical exposure audit {HIST_BLOCK} (an earlier, different repair study on pvlib): offered = "
                "completed model requests whose tool list held the tool; requested, executed and visible count "
                "tool calls or results. Motivation only, not corroboration of Exp A.",
                "panel_b": "Exp A block T (seed 11; role exploratory in ANALYSIS.json): requested and C* recovered "
                           "from ANALYSIS.json secondary.t_block; non-empty answer visible and submitted from "
                           "EXPLORE.json section 7 (exploratory). Bars are family-weighted shares (equal to k/n "
                           "here); whiskers are 95% t intervals over 30 families (marginal); a zero-variance "
                           "interval is not drawn. The stages are not nested: submission does not require a "
                           "graph answer.",
            },
            "fig2_outcomes": {
                "files": ["fig2_outcomes.png", "fig2_outcomes.svg"],
                "description": "Block P, seeds 11/23/47, 2,250 episodes per condition. Segments: submitted and C* "
                               "listed, submitted and C* not listed, budget exhausted (NoAnswer). No ContextLimit, "
                               "FormatExhausted or other exit occurred. Point and whisker: family-weighted C* "
                               "recovery with its 95% t interval (marginal). Lines: model-free reference policies "
                               "on the frozen items (references, not ceilings). The listed/not-listed split is "
                               "bookkeeping, not mediation.",
            },
            "fig3_contrasts": {
                "files": ["fig3_contrasts.png", "fig3_contrasts.svg"],
                "description": "Confirmatory H1-H3 on block P: estimate and 95% t interval (marginal) over 30 "
                               "families, the 30 family mean differences as dots, dashed lines at the prespecified "
                               "practically meaningful effect, Holm-adjusted p over H1-H3 only. Inference is "
                               "benchmark-conditional on the frozen family set.",
                "extras": [ex["name"] for ex in extras],
            },
            "fig4_timeline": {
                "files": ["fig4_timeline.png", "fig4_timeline.svg"],
                "description": "Exploratory (post-results) TIMELINE.json, block P: family-weighted share of "
                               "episodes submitted by response k and share with the C* site visible by response k "
                               "under the LOOSE rule (can overcount; STRICT is the lower bound), 95% t bands "
                               "(marginal); vertical line at the 8-response cap. Descriptive; no causal or "
                               "mediation reading.",
            },
        },
        "palette": {"blue": BLUE, "orange": ORANGE, "light_blue": LIGHT_BLUE, "exhausted_fill": EXHAUSTED_FILL,
                    "ink": INK, "secondary_ink": INK2, "muted": MUTED, "surface": SURFACE,
                    "validation": "dataviz validate_palette.py, light mode: #2a78d6,#eb6834 all-pairs PASS "
                                  "(CVD dE 24.7, normal dE 33.6); #86b6ef,#2a78d6 ordinal PASS (light end 2.06:1)"},
        "condition_labels": LABELS,
        "values": book.values,
    }
    manifest.write_bytes((json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8"))
    print(json.dumps({"outputs": written, "checks_passed": record["checks_passed"],
                      "checks_total": len(book.checks)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
