"""Figures for runs (dark theme, sized for a 16:9 screen recording).

Palette: dataviz reference palette, dark mode (surface #1a1a19); only the first
three categorical slots are used so every pair stays colour-blind distinguishable.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE = "#1a1a19"
TEXT = "#f0efec"
TEXT2 = "#c3c2b7"
MUTED = "#6b6a64"
GRID = "#383835"
S1, S2, S3 = "#3987e5", "#d95926", "#199e70"  # blue (stimulated), orange (outputs), aqua (silenced)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": TEXT, "axes.labelcolor": TEXT2, "xtick.color": TEXT2, "ytick.color": TEXT2,
    "axes.edgecolor": GRID, "font.size": 9.5, "font.family": "DejaVu Sans",
})


def raster_figure(path, title, rows, spikes_by_row, duration_s, group_rates, threshold_lines=None):
    """rows: list of (label, role) role in {'stim','output','other','silenced'};
    spikes_by_row: list of arrays of spike times (s) for one trial;
    group_rates: list of (group_name, rate_hz) for the bar panel."""
    fig = plt.figure(figsize=(9.6, 5.4), dpi=150)
    gs = fig.add_gridspec(1, 2, width_ratios=[2.0, 1], wspace=0.42, left=0.15, right=0.96, top=0.80, bottom=0.11)
    ax = fig.add_subplot(gs[0])
    color = {"stim": S1, "output": S2, "silenced": S3, "other": TEXT2}
    for y, ((lab, role), ts) in enumerate(zip(rows, spikes_by_row)):
        if len(ts):
            ax.scatter(ts, np.full(len(ts), y), s=16, marker="|", linewidths=1.3, color=color[role])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    for t, (lab, role) in zip(ax.get_yticklabels(), rows):
        t.set_color(color[role] if role != "other" else TEXT2)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlim(0, duration_s)
    ax.set_xlabel("time (s) — one trial")
    ax.grid(axis="x", color=GRID, lw=0.6)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    # legend (role is also given by label colour; legend keeps identity non-colour-only)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=color[r], marker="|", ls="", ms=10, mew=2, label=l)
               for r, l in [("stim", "stimulated"), ("output", "output (motor/descending)"), ("other", "other responders"),
                            ("silenced", "silenced")] if any(rr[1] == r for rr in rows)]
    ax.legend(handles=handles, loc="lower left", fontsize=9, frameon=False, labelcolor=TEXT2,
              bbox_to_anchor=(0.0, 1.0), ncol=len(handles), handletextpad=0.3, columnspacing=1.2,
              title="Spike raster (one trial)", title_fontsize=10, alignment="left")

    bx = fig.add_subplot(gs[1])
    names = [g for g, _ in group_rates]
    vals = np.array([r for _, r in group_rates])
    yy = np.arange(len(names))
    bx.barh(yy, vals, height=0.62, color=S2, edgecolor=SURFACE, linewidth=2)
    vmax = max(100.0, float(vals.max()) if len(vals) else 100.0)  # fixed scale: conditions comparable
    for y, v in zip(yy, vals):
        bx.text(v + vmax * 0.02, y, f"{v:.0f} Hz", va="center", fontsize=9, color=TEXT)
    if threshold_lines:
        for name, thr in threshold_lines:
            if name in names:
                y = names.index(name)
                bx.plot([thr, thr], [y - 0.45, y + 0.45], color=TEXT, lw=1.4, ls=(0, (2, 2)))
                bx.text(thr, y + 0.62, f"feeding threshold {thr:.0f} Hz", fontsize=7.5, color=TEXT2, ha="center")
    bx.set_yticks(yy)
    bx.set_yticklabels(names, fontsize=9.5)
    bx.set_ylim(len(names) - 0.5, -0.5)
    bx.set_xlim(0, vmax * 1.2)
    bx.set_axisbelow(True)
    bx.set_xlabel("mean firing rate (Hz per neuron)")
    bx.grid(axis="x", color=GRID, lw=0.6)
    for s in ["top", "right"]:
        bx.spines[s].set_visible(False)
    bx.set_title("Output groups", fontsize=11, color=TEXT2, loc="left")
    fig.suptitle(title, x=0.02, y=0.975, ha="left", fontsize=13, color=TEXT, fontweight="bold")
    fig.text(0.02, 0.905, "FlyWire v783 connectome · 138,639 neurons · whole-brain LIF (Shiu et al. 2024)",
             fontsize=9.5, color=MUTED)
    fig.savefig(path)
    plt.close(fig)
    return str(path)
