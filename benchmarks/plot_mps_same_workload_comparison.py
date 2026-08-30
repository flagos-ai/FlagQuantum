#!/usr/bin/env python3
"""Plot the measured same-workload FlagQuantum/cuQuantum MPS outcome."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/flagquantum-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    REPO_ROOT
    / "benchmarks"
    / "results"
    / "legacy"
    / "mps_capacity_16xa800_20260806"
)
README_ASSET = REPO_ROOT / "assets" / "readme" / "mps-same-workload-comparison.png"

BLUE = "#285F9E"
GREEN = "#16866B"
GREEN_BG = "#EAF6F2"
RED = "#B44343"
RED_BG = "#FAEEEE"
INK = "#20252B"
GREY = "#66717D"
LIGHT_GREY = "#AAB2BA"
GRID = "#D9DEE3"
PANEL = "#F7F9FA"


def box(fig, xywh, *, facecolor, edgecolor=GRID, radius=0.012, linewidth=0.9):
    x, y, w, h = xywh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        transform=fig.transFigure,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        clip_on=False,
    )
    fig.add_artist(patch)
    return patch


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "text.color": INK,
            "axes.labelcolor": INK,
            "svg.fonttype": "none",
        }
    )
    fig = plt.figure(figsize=(12.4, 5.25), facecolor="white")

    fig.text(
        0.055,
        0.935,
        "Same MPS workload: completion outcome",
        fontsize=17,
        fontweight="bold",
        ha="left",
        va="top",
        color=INK,
    )
    fig.text(
        0.055,
        0.875,
        "FlagQuantum completed the measured 131,072-qubit training step; "
        "cuQuantum 26.6 did not produce an output state.",
        fontsize=9.2,
        ha="left",
        va="top",
        color=GREY,
    )

    box(fig, (0.055, 0.755, 0.89, 0.075), facecolor=PANEL)
    fig.text(
        0.5,
        0.792,
        "131,072 qubits   ·   χ=768   ·   complex64   ·   "
        "16 RY + 15 RXX   ·   center-Z observable   ·   one gradient step",
        ha="center",
        va="center",
        fontsize=9.2,
        color=INK,
        fontweight="bold",
    )

    ax = fig.add_axes((0.16, 0.285, 0.515, 0.38))
    stages = ["Initial MPS", "31 gates", "Output MPS", "Center Z", "Gradient"]
    x = np.arange(len(stages))
    fq_y, cq_y = 1.0, 0.0
    ax.hlines(fq_y, x[0], x[-1], color=GREEN, linewidth=3.0, zorder=1)
    ax.scatter(x, np.full_like(x, fq_y, dtype=float), s=150, color=GREEN,
               edgecolor="white", linewidth=1.4, zorder=3)
    for xpos in x:
        ax.text(xpos, fq_y, "✓", color="white", ha="center", va="center",
                fontsize=9, fontweight="bold", zorder=4)

    ax.hlines(cq_y, x[0], x[2], color=RED, linewidth=3.0, zorder=1)
    ax.hlines(cq_y, x[2], x[-1], color=LIGHT_GREY, linewidth=2.0,
              linestyle=(0, (3, 3)), zorder=1)
    ax.scatter(x[:2], np.full(2, cq_y), s=150, color=RED,
               edgecolor="white", linewidth=1.4, zorder=3)
    for xpos in x[:2]:
        ax.text(xpos, cq_y, "✓", color="white", ha="center", va="center",
                fontsize=9, fontweight="bold", zorder=4)
    ax.scatter([x[2]], [cq_y], s=180, marker="X", color=RED,
               edgecolor="white", linewidth=1.2, zorder=4)
    ax.scatter(x[3:], np.full(2, cq_y), s=100, facecolor="white",
               edgecolor=LIGHT_GREY, linewidth=1.4, zorder=2)
    for xpos in x[3:]:
        ax.text(xpos, cq_y, "—", color=LIGHT_GREY, ha="center", va="center",
                fontsize=9, fontweight="bold", zorder=4)

    fig.text(0.145, 0.583, "FlagQuantum", ha="right", va="center",
             fontsize=9.5, color=GREEN, fontweight="bold")
    fig.text(0.145, 0.401, "cuQuantum 26.6", ha="right", va="center",
             fontsize=9.5, color=RED, fontweight="bold")
    ax.text(4.0, 1.24, "COMPLETED  ·  367.37 s", color=GREEN, ha="right",
            va="center", fontsize=9, fontweight="bold")
    ax.text(2.0, -0.31, "FAILED  ·  483.96 s to failure", color=RED,
            ha="center", va="center", fontsize=8.6, fontweight="bold")
    ax.text(2.0, -0.50, "int64 extent overflow → negative output dimension",
            color=GREY, ha="center", va="center", fontsize=7.5)
    ax.set_xticks(x, stages)
    ax.tick_params(axis="x", length=0, pad=10, labelsize=8.2, colors=GREY)
    ax.set_xlim(-0.35, 4.35)
    ax.set_ylim(-0.64, 1.45)
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("a   Measured execution path", loc="left", pad=13,
                 fontsize=10.5, fontweight="bold", color=INK)

    box(fig, (0.72, 0.285, 0.225, 0.38), facecolor=PANEL)
    fig.text(0.735, 0.635, "b   Execution context", fontsize=10.5,
             fontweight="bold", color=INK, ha="left")
    fig.text(0.75, 0.572, "FlagQuantum", fontsize=9.5, fontweight="bold",
             color=GREEN, ha="left")
    fig.text(0.75, 0.530, "2 nodes · 16×A800", fontsize=9, color=INK, ha="left")
    fig.text(0.75, 0.493, "Distributed MPS", fontsize=8.2, color=GREY, ha="left")
    fig.add_artist(plt.Line2D([0.75, 0.915], [0.456, 0.456],
                             transform=fig.transFigure, color=GRID, linewidth=0.8))
    fig.text(0.75, 0.412, "cuQuantum NetworkState", fontsize=9.5,
             fontweight="bold", color=RED, ha="left")
    fig.text(0.75, 0.370, "1×A800 · device_id=0", fontsize=9, color=INK, ha="left")
    fig.text(0.75, 0.333, "Public MPS path is single-device", fontsize=8.2,
             color=GREY, ha="left")

    box(fig, (0.055, 0.145, 0.89, 0.075), facecolor=GREEN_BG, edgecolor="#B9DDD2")
    fig.text(
        0.5,
        0.182,
        "Same 131,072-qubit workload: FlagQuantum completed; "
        "cuQuantum 26.6 did not complete.",
        ha="center",
        va="center",
        fontsize=10.2,
        color=GREEN,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.072,
        "Completion/capability comparison—not a speedup claim. cuQuantum time is "
        "time-to-failure. Measured 2026-08-06 on NVIDIA A800 80 GB GPUs.",
        ha="center",
        va="center",
        fontsize=7.6,
        color=GREY,
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT / "mps_same_workload_comparison"
    fig.savefig(stem.with_suffix(".png"), dpi=240, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    fig.savefig(README_ASSET, dpi=180, facecolor="white")
    plt.close(fig)
    print(stem.with_suffix(".png"))


if __name__ == "__main__":
    main()
