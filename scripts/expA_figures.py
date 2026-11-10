"""Exp A figures from a v2 ANALYSIS.json (matplotlib; PNG + SVG; deterministic bytes per matplotlib version).

    <figures env python> scripts/expA_figures.py --analysis OUT/ANALYSIS.json --out OUT/figures [--overwrite]

Environment: matplotlib is NOT in the controller env. Use the figures env
(/mnt/kldata/absent-edges-20261018-core2/expA/dev/integrate/envs/fig, matplotlib 3.9.2 + numpy
2.4.6; lock file envs/fig.lock.txt). Without matplotlib this script exits with status 2 and a
message naming that env; it never skips silently.

Figure 1 (fig1_primary.*): H1-H3 on block P. Point = family-weighted estimate, solid whisker =
95% t interval (the primary interval), thin offset whisker = 95% family cluster bootstrap
interval (sensitivity), small gray dots = family differences. A solid line marks 0; dashed lines
mark the practically meaningful effect (+-0.10). The Holm-adjusted p and the test used are printed
beside each row; exploratory contrasts (< 20 families) are labelled so.
Figure 2 (fig2_recovery.*): site_recovered of C* by condition, one panel per block (S-NAT:
nm_any); point + 95% t interval for the episodes, and the four model-free reference policies
(answer_only, c4_hits, answer_plus_c4, grep_all) as hollow markers that differ by shape. The
reference policies are references, not ceilings.
Figure 3 (fig3_cost.*): mean prompt + completion tokens per episode by condition with the 95% t
interval over families, one panel per block.

Colours (dataviz reference palette, validated light mode, both slots pass): blue #2a78d6 carries
model episodes, orange #eb6834 the reference policies; gray #898781 is de-emphasised context.
Identity is always also carried by text (axis labels, legends, marker shape). FIGURES.json records
the input sha256 and each output's sha256; TABLES.md holds the table view.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
FAMILY_DOT = "#c3c2b7"
FIGURES_ENV = "/mnt/kldata/absent-edges-20261018-core2/expA/dev/integrate/envs/fig/bin/python"
POLICY_MARKERS = {"answer_only": "v", "c4_hits": "s", "answer_plus_c4": "D", "grep_all": "^"}


def _pyplot():
    try:
        import matplotlib
    except ModuleNotFoundError:
        sys.stderr.write("expA_figures.py needs matplotlib; run it with the figures env: " + FIGURES_ENV + "\n")
        raise SystemExit(2)
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "svg.hashsalt": "expA-figures-v2", "font.family": "DejaVu Sans", "font.size": 8.5,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 9.5, "axes.titleweight": "regular", "xtick.color": INK2, "ytick.color": INK2,
        "xtick.major.size": 0, "ytick.major.size": 0, "axes.grid": True, "axes.grid.axis": "y",
        "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-", "axes.spines.top": False,
        "axes.spines.right": False, "axes.spines.left": False, "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "legend.frameon": False})
    return matplotlib, plt


MAX_WIDTH_IN = 11.0


def _subtitle(fig, analysis, y):
    """The run label (e.g. SYNTHETIC) as a muted second title line, shortened to fit."""
    label = analysis.get("label") or ""
    if label:
        fig.text(0.01, y, label if len(label) <= 110 else label[:107] + "...", ha="left", va="bottom", color=MUTED,
                 fontsize=7.5)


def _width(columns):
    return min(MAX_WIDTH_IN, 0.62 * columns + 1.6)


TICKS = {"T-offered": "offered", "T-recommended": "recommended"}


def _tick(condition):
    return TICKS.get(condition, condition)


def _ratios(blocks, rows):
    """Panel widths by condition count; the T panel needs room for its longer tick labels."""
    return [max(n, 2.6) if block == "T" else n for block, n in
            ((b, sum(r["block"] == b for r in rows)) for b in blocks)]


def _panel_title(block):
    return {"T": "T (graph_query)", "S-NAT": "S-NAT nm_any"}.get(block, block)


def _p(value):
    if value is None:
        return "n/a"
    return f"{value:.1e}" if value < 1e-3 else f"{value:.3f}"


def figure_primary(analysis, plt):
    primary = analysis["primary"]
    rows = [h for h in primary["hypotheses"]][::-1]
    fig, ax = plt.subplots(figsize=(7.2, 0.75 * len(rows) + 1.6))
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    threshold = primary.get("practically_meaningful_effect", 0.10)
    ax.axvline(0.0, color=AXIS, linewidth=1.0, zorder=1)
    for value in (-threshold, threshold):
        ax.axvline(value, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)), zorder=1)
    for y, h in enumerate(rows):
        ax.text(-0.02, y, f"{h['id']}  {h['treatment']} - {h['control']}", transform=ax.get_yaxis_transform(),
                ha="right", va="center", color=INK, fontsize=8.5)
        if h.get("estimate") is None:
            ax.text(0.0, y, "not estimable", ha="center", va="center", color=INK2, fontsize=8)
            continue
        families = [u["difference"] for u in h.get("family_units", [])]
        for k, value in enumerate(families):
            offset = ((k * 7919) % 11 - 5) * 0.022
            ax.plot(value, y + 0.18 + offset, "o", color=FAMILY_DOT, markersize=3.2, zorder=2)
        interval = (h.get("t_test") or {}).get("interval") or {}
        if interval.get("lower") is not None:
            ax.plot([interval["lower"], interval["upper"]], [y, y], color=BLUE, linewidth=2.0,
                    solid_capstyle="round", zorder=3)
        boot = h.get("bootstrap") or {}
        if boot.get("lower") is not None:
            ax.plot([boot["lower"], boot["upper"]], [y - 0.16, y - 0.16], color=INK2, linewidth=0.9, zorder=3)
        ax.plot(h["estimate"], y, "o", color=BLUE, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2,
                zorder=4)
        status = "" if h.get("confirmatory") else "exploratory, "
        ax.text(1.02, y, f"{status}Holm p {_p(h.get('holm_adjusted_p'))}\n({h.get('p_source', 'n/a')})",
                transform=ax.get_yaxis_transform(), ha="left", va="center", color=INK2, fontsize=7.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([""] * len(rows))
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel("difference in C* recovery rate (family-weighted)")
    seeds = ",".join(str(s) for s in primary.get("primary_seeds", []))
    fig.suptitle(f"Primary contrasts on block P (seeds {seeds or 'none'})", x=0.01, y=0.975, ha="left", color=INK,
                 fontsize=9.5)
    _subtitle(fig, analysis, 0.895)
    handles = [plt.Line2D([], [], color=BLUE, marker="o", linewidth=2.0, markersize=6,
                          label="estimate, 95% t interval"),
               plt.Line2D([], [], color=INK2, linewidth=0.9, label="95% bootstrap interval (sensitivity)"),
               plt.Line2D([], [], color=FAMILY_DOT, marker="o", linestyle="", markersize=4,
                          label="family difference"),
               plt.Line2D([], [], color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)),
                          label=f"meaningful effect +-{threshold:.2f}")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, fontsize=7.5)
    fig.subplots_adjust(left=0.2, right=0.78, bottom=0.36, top=0.84)
    return fig


def _rate_rows(analysis):
    blocks = []
    for row in analysis["secondary"]["rates"]:
        if row["block"] not in blocks:
            blocks.append(row["block"])
    return blocks, analysis["secondary"]["rates"]


def figure_recovery(analysis, plt):
    blocks, rates = _rate_rows(analysis)
    fig, axes = plt.subplots(1, len(blocks), figsize=(_width(len(rates)), 3.8), sharey=True,
                             squeeze=False, gridspec_kw={"width_ratios": _ratios(blocks, rates)})
    for ax, block in zip(axes[0], blocks):
        rows = [r for r in rates if r["block"] == block]
        outcome = "nm_any" if block == "S-NAT" else "site_recovered"
        for x, row in enumerate(rows):
            result = row.get(outcome) or {}
            interval = result.get("t_interval") or {}
            if interval.get("lower") is not None:
                ax.plot([x, x], [max(0.0, interval["lower"]), min(1.0, interval["upper"])], color=BLUE,
                        linewidth=2.0, solid_capstyle="round", zorder=3)
            if result.get("estimate") is not None:
                ax.plot(x, result["estimate"], "o", color=BLUE, markersize=8, markeredgecolor=SURFACE,
                        markeredgewidth=2, zorder=4)
            for k, (policy, marker) in enumerate(POLICY_MARKERS.items()):
                value = (row["reference_policies"].get(policy) or {}).get(outcome)
                if value is not None:
                    ax.plot(x + 0.16 + 0.07 * k, value, marker, color=ORANGE, markerfacecolor=SURFACE, markersize=5,
                            markeredgewidth=1.2, zorder=2)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels([_tick(r["condition"]) for r in rows])
        ax.set_xlim(-0.6, len(rows) - 0.2)
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(_panel_title(block), loc="left", fontsize=8.5)
    axes[0][0].set_ylabel("C* recovered (family-weighted rate)")
    handles = [plt.Line2D([], [], color=BLUE, marker="o", linewidth=2.0, markersize=6,
                          label="episodes, 95% t interval")]
    handles += [plt.Line2D([], [], color=ORANGE, marker=m, linestyle="", markerfacecolor=SURFACE, markersize=5,
                           label=f"reference: {p}") for p, m in POLICY_MARKERS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7.5)
    fig.suptitle("Recovery of the omitted caller by block and condition", x=0.01, y=0.975, ha="left", color=INK,
                 fontsize=9.5)
    _subtitle(fig, analysis, 0.895)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.25, top=0.8, wspace=0.1)
    return fig


def figure_cost(analysis, plt):
    rows = analysis["cost"]
    blocks = []
    for row in rows:
        if row["block"] not in blocks:
            blocks.append(row["block"])
    fig, axes = plt.subplots(1, len(blocks), figsize=(_width(len(rows)), 3.4), sharey=True, squeeze=False,
                             gridspec_kw={"width_ratios": _ratios(blocks, rows)})
    for ax, block in zip(axes[0], blocks):
        mine = [r for r in rows if r["block"] == block]
        for x, row in enumerate(mine):
            result = row.get("family_total_tokens") or {}
            interval = result.get("t_interval") or {}
            if interval.get("lower") is not None:
                ax.plot([x, x], [max(0.0, interval["lower"]), interval["upper"]], color=BLUE, linewidth=2.0,
                        solid_capstyle="round", zorder=3)
            if result.get("estimate") is not None:
                ax.plot(x, result["estimate"], "o", color=BLUE, markersize=8, markeredgecolor=SURFACE,
                        markeredgewidth=2, zorder=4)
        ax.set_xticks(range(len(mine)))
        ax.set_xticklabels([_tick(r["condition"]) for r in mine])
        ax.set_xlim(-0.6, len(mine) - 0.4)
        ax.set_title("T (graph_query)" if block == "T" else block, loc="left", fontsize=8.5)
    axes[0][0].set_ylabel("prompt + completion tokens per episode")
    axes[0][0].set_ylim(bottom=0)
    fig.suptitle("Episode cost by block and condition (family-weighted, 95% t interval)", x=0.01, y=0.975, ha="left",
                 color=INK, fontsize=9.5)
    _subtitle(fig, analysis, 0.885)
    fig.subplots_adjust(left=0.1, right=0.99, bottom=0.12, top=0.8, wspace=0.1)
    return fig


def _save(fig, out, stem, overwrite, plt):
    written = {}
    for ext, metadata in (("png", {"Software": None}), ("svg", {"Date": None})):
        path = out / f"{stem}.{ext}"
        if path.exists() and not overwrite:
            raise SystemExit(f"refusing to overwrite {path} (pass --overwrite)")
        fig.savefig(path, format=ext, dpi=200, metadata=metadata)
        written[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    plt.close(fig)
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--analysis", required=True, help="ANALYSIS.json written by scripts/expA_analyze.py")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    matplotlib, plt = _pyplot()
    data = Path(args.analysis).read_bytes()
    analysis = json.loads(data)
    if analysis.get("schema_version") != "expA-analysis-v2":
        raise SystemExit(f"{args.analysis} is not an expA-analysis-v2 document")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    written = {}
    written.update(_save(figure_primary(analysis, plt), out, "fig1_primary", args.overwrite, plt))
    written.update(_save(figure_recovery(analysis, plt), out, "fig2_recovery", args.overwrite, plt))
    written.update(_save(figure_cost(analysis, plt), out, "fig3_cost", args.overwrite, plt))
    record = {"schema_version": "expA-figures-v2", "matplotlib": matplotlib.__version__,
              "analysis": {"path": str(args.analysis), "sha256": hashlib.sha256(data).hexdigest()},
              "label": analysis.get("label"), "outputs": written}
    manifest = out / "FIGURES.json"
    if manifest.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite {manifest}")
    manifest.write_bytes((json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
