#!/usr/bin/env python3
"""Plot sealed two-node 16×A800 MPS capacity evidence."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/flagquantum-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, LogLocator

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "benchmarks" / "results" / "local"
OUTPUT = (
    REPO_ROOT
    / "benchmarks"
    / "results"
    / "legacy"
    / "mps_capacity_16xa800_20260806"
)
CAPACITY_FILES = (
    "mps_capacity_24576q_chi768_16xa800_complete_20260805.json",
    "mps_capacity_65536q_chi768_16xa800_complete_20260805.json",
    "mps_capacity_98304q_chi768_16xa800_complete_20260805.json",
    "mps_capacity_114688q_chi768_16xa800_complete_20260805.json",
    "mps_capacity_131072q_chi768_16xa800_repeat_complete_20260806.json",
)
INITIAL_131072_CANDIDATE_SECONDS = 373.79778012260795
FP32_EPSILON = np.finfo(np.float32).eps

BLUE = "#285F9E"
ORANGE = "#D97706"
GREEN = "#16866B"
RED = "#B44343"
INK = "#20252B"
GREY = "#66717D"
GRID = "#D9DEE3"


def load_capacity(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload["rank_records"]
    return {
        "sites": int(payload["n_sites"]),
        "logical_gib": float(payload["logical_mps_bytes"]) / 2**30,
        "seconds": max(float(record["elapsed_seconds"]) for record in records),
        "peak_gib": max(
            float(record["peak_memory_bytes"]) for record in records
        )
        / 2**30,
        "discarded_weight": float(payload["discarded_weight"]),
        "error_budget": float(payload["truncation_error_budget"]),
    }


def thousands(value: float, _position: float) -> str:
    return f"{int(value):,}"


def panel_label(ax: Any, label: str) -> None:
    ax.text(
        -0.13,
        1.07,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        color=INK,
        va="top",
    )


def main() -> None:
    rows = [load_capacity(RESULTS / name) for name in CAPACITY_FILES]
    sites = np.array([row["sites"] for row in rows])
    logical_gib = np.array([row["logical_gib"] for row in rows])
    seconds = np.array([row["seconds"] for row in rows])
    peak_gib = np.array([row["peak_gib"] for row in rows])
    discarded = rows[-1]["discarded_weight"]
    error_budget = rows[-1]["error_budget"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.edgecolor": GREY,
            "axes.labelcolor": INK,
            "axes.linewidth": 0.8,
            "axes.titleweight": "bold",
            "axes.titlesize": 10,
            "xtick.color": GREY,
            "ytick.color": GREY,
            "legend.frameon": False,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12.4, 4.55),
        gridspec_kw={"width_ratios": (1.22, 1.22, 1.0)},
    )
    fig.subplots_adjust(
        left=0.065, right=0.985, bottom=0.24, top=0.76, wspace=0.42
    )

    runtime = axes[0]
    runtime.plot(sites, seconds, color=BLUE, linewidth=1.8, zorder=2)
    runtime.scatter(
        sites[:-1], seconds[:-1], s=34, color=BLUE, edgecolor="white", zorder=3
    )
    runtime.scatter(
        sites[-1],
        seconds[-1],
        s=62,
        marker="*",
        color=GREEN,
        edgecolor="white",
        linewidth=0.8,
        label="sealed repeat",
        zorder=4,
    )
    runtime.scatter(
        sites[-1],
        INITIAL_131072_CANDIDATE_SECONDS,
        s=44,
        facecolor="none",
        edgecolor=ORANGE,
        linewidth=1.3,
        label="initial candidate",
        zorder=4,
    )
    runtime.text(
        0.96,
        0.08,
        "131,072 sites\nsealed repeat: 367.37 s\ninitial candidate: 373.80 s",
        transform=runtime.transAxes,
        ha="right",
        va="bottom",
        color=GREEN,
        fontsize=7.4,
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": GRID,
            "linewidth": 0.8,
            "alpha": 0.94,
        },
    )
    runtime.set(
        title="End-to-end runtime",
        xlabel="MPS sites",
        ylabel="Wall time (s), slowest rank",
        xlim=(18_000, 137_000),
        ylim=(40, 405),
    )
    runtime.xaxis.set_major_formatter(FuncFormatter(thousands))
    runtime.grid(axis="y", color=GRID, linewidth=0.6)
    runtime.legend(loc="upper left", fontsize=7.2)
    panel_label(runtime, "a")

    memory = axes[1]
    memory.plot(sites, peak_gib, color=ORANGE, linewidth=1.8, zorder=2)
    memory.fill_between(sites, 0, peak_gib, color=ORANGE, alpha=0.10)
    memory.scatter(
        sites[:-1], peak_gib[:-1], s=34, color=ORANGE, edgecolor="white", zorder=3
    )
    memory.scatter(
        sites[-1],
        peak_gib[-1],
        s=62,
        marker="*",
        color=GREEN,
        edgecolor="white",
        linewidth=0.8,
        zorder=4,
    )
    memory.axhline(
        79.25,
        color=RED,
        linewidth=1.1,
        linestyle=(0, (4, 3)),
    )
    memory.text(
        0.05,
        0.87,
        "A800 capacity: 79.25 GiB",
        transform=memory.transAxes,
        color=RED,
        fontsize=7.2,
        bbox={
            "boxstyle": "round,pad=0.2",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.9,
        },
    )
    memory.text(
        0.96,
        0.08,
        f"131,072 sites\n{peak_gib[-1]:.2f} GiB/rank\n"
        f"{logical_gib[-1]:,.2f} GiB logical",
        transform=memory.transAxes,
        ha="right",
        va="bottom",
        color=GREEN,
        fontsize=7.4,
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": GRID,
            "linewidth": 0.8,
            "alpha": 0.94,
        },
    )
    memory.set(
        title="Rank-local memory",
        xlabel="MPS sites",
        ylabel="Peak allocated memory (GiB/rank)",
        xlim=(18_000, 137_000),
        ylim=(0, 84),
    )
    memory.xaxis.set_major_formatter(FuncFormatter(thousands))
    memory.grid(axis="y", color=GRID, linewidth=0.6)
    panel_label(memory, "b")

    precision = axes[2]
    levels = np.array([FP32_EPSILON, discarded, error_budget])
    labels = ("FP32 epsilon", "discarded weight", "error budget")
    colors = (BLUE, GREEN, RED)
    y = np.arange(3)
    precision.hlines(y, 1e-8, levels, color=colors, linewidth=2.2)
    precision.scatter(levels, y, s=45, color=colors, edgecolor="white", zorder=3)
    for value, ypos, label, color in zip(levels, y, labels, colors):
        precision.text(
            1.35e-8,
            ypos + 0.30,
            label,
            va="bottom",
            fontsize=7.2,
            color=GREY,
        )
        label_on_left = value >= 1e-2
        precision.annotate(
            f"{value:.2e}",
            (value, ypos),
            xytext=((-6 if label_on_left else 5), 6),
            textcoords="offset points",
            ha=("right" if label_on_left else "left"),
            va="bottom",
            fontsize=7.3,
            color=color,
            fontweight="bold",
        )
    precision.set_xscale("log")
    precision.set(
        title="Numerical scale",
        xlabel="Magnitude (log scale)",
        xlim=(1e-8, 1),
        yticks=(),
        ylim=(-0.55, 2.55),
    )
    precision.xaxis.set_major_locator(LogLocator(base=10, numticks=9))
    precision.grid(axis="x", color=GRID, linewidth=0.6)
    precision.text(
        0.04,
        0.73,
        f"discarded / FP32 eps = {discarded / FP32_EPSILON:.1f}×",
        transform=precision.transAxes,
        va="top",
        fontsize=7.2,
        color=GREEN,
        fontweight="bold",
    )
    panel_label(precision, "c")

    fig.suptitle(
        "FlagQuantum distributed MPS: 131,072 qubits on 16×A800",
        x=0.07,
        y=0.955,
        ha="left",
        fontsize=14,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.07,
        0.865,
        "Two nodes · complex64 · χ=768 · measured batch-one training · "
        "16/16 useful ranks · "
        "15/15 forward/reverse boundaries · expandable CUDA segments",
        fontsize=8.2,
        color=GREY,
    )
    fig.text(
        0.5,
        0.105,
        "131,072 MPS sites = 131,072 qubits  |  χ=768  |  "
        "max cut entropy log₂(χ) = 9.58 ebits",
        ha="center",
        fontsize=8.0,
        color=GREEN,
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "#EDF7F4",
            "edgecolor": "#B9DDD2",
            "linewidth": 0.8,
        },
    )
    fig.text(
        0.5,
        0.04,
        "Entanglement-limited MPS simulation; not arbitrary 131,072-qubit "
        "statevector capacity. Development evidence.",
        ha="center",
        fontsize=7.2,
        color=GREY,
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT / "mps_capacity_16xa800_scientific"
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix(f".{suffix}"), dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(stem)


if __name__ == "__main__":
    main()
