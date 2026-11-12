"""Draw docs/architecture.png (one-slide architecture diagram, 16:9, dark)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

SURF, TEXT, TEXT2, MUTED, BORDER = "#1a1a19", "#f0efec", "#c3c2b7", "#8a897f", "#383835"
BLUE, ORANGE, AQUA, YELLOW = "#3987e5", "#d95926", "#199e70", "#c98500"

fig = plt.figure(figsize=(16, 9), dpi=120)
fig.patch.set_facecolor(SURF)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 160)
ax.set_ylim(0, 90)
ax.axis("off")


def box(x, y, w, h, title, lines, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5", fc="#222220", ec=color, lw=2.2))
    ax.text(x + 1.8, y + h - 3.2, title, color=TEXT, fontsize=15, fontweight="bold", va="center")
    for i, ln in enumerate(lines):
        ax.text(x + 1.8, y + h - 7.6 - i * 3.6, ln, color=TEXT2, fontsize=11.5, va="center")


def arrow(x1, y1, x2, y2, label="", color=MUTED):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=18, color=color, lw=2))
    if label:
        if x1 == x2:  # vertical arrow: label to the right
            ax.text(x1 + 1.5, (y1 + y2) / 2, label, color=MUTED, fontsize=10.5, ha="left", va="center")
        else:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 1.6, label, color=MUTED, fontsize=10.5, ha="center")


ax.text(4, 85, "FlyLab", color=TEXT, fontsize=30, fontweight="bold", va="center")
ax.text(29, 84.6, "Claude as an autonomous neuroscientist on a whole fly brain", color=TEXT2, fontsize=16, va="center")

box(4, 50, 34, 28, "Claude Code", ["CLAUDE.md: framing + house rules", "Skill: fly-neuroscientist", "   (8-step experimental method)",
                                     "Subagents: literature-checker,", "   experiment-designer"], ORANGE)
box(4, 10, 34, 30, "Hook (Pre/PostToolUse)", ["log_tool_event.py", "-> runs/events.jsonl", "", "Dashboard :8000",
                                                "live tool feed · fly video ·", "brain raster · lab report"], YELLOW)
box(48, 50, 30, 28, "MCP server: flylab", ["11 tools over stdio:", "search · trace_pathway ·", "run_simulation · lesion_screen ·",
                                             "compare · run_behavior ·", "literature · save_report"], BLUE)
box(88, 58, 68, 20, "Whole-brain model (real data)", ["FlyWire v783: 138,639 neurons · 15.1 M connected pairs",
                                                        "LIF model of Shiu et al. 2024 (fixed weights, no learning)",
                                                        "event-driven numba kernel, spike-for-spike = Eon PyTorch"], AQUA)
box(88, 30, 68, 20, "Brain -> body interface (hand-designed)", ["P9 -> forward · MDN -> backward",
                                                                  "DNa01/02 left - right -> turn",
                                                                  "MN9 > 30 Hz -> feeding (walking suppressed)"], MUTED)
box(88, 4, 68, 20, "Body: NeuroMechFly v2 (flygym, MuJoCo)", ["flygym walking controller, physics at 0.1 ms",
                                                                "sugar contact -> sugar GRN input (closed loop)",
                                                                "renders video.mp4"], ORANGE)
arrow(38, 66, 48, 66, "tool calls")
arrow(21, 50, 21, 40.5, "every tool call")
arrow(78, 68, 88, 68, "simulate")
arrow(122, 58, 122, 50.5, "output rates")
arrow(122, 30, 122, 24.5, "motor commands")
arrow(88, 12, 38, 22, "video, raster, results (runs/)")
ax.text(4, 2.5, "Sources: Dorkenwald et al. 2024 & Schlegel et al. 2024 (FlyWire) · Shiu et al. 2024 (model) · "
        "Wang-Chen et al. 2024 (NeuroMechFly v2) · Eon Systems fly-brain", color=MUTED, fontsize=10)
out = Path(__file__).resolve().parents[1] / "docs" / "architecture.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, facecolor=SURF)
print(out)
