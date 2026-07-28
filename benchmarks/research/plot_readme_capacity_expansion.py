#!/usr/bin/env python3
"""Create the matched statevector capacity-expansion figure used by README."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/flagquantum-matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

GREEN = "#009E73"
BLUE = "#0072B2"
ORANGE = "#E69F00"
RED = "#C94A4A"
INK = "#17202A"
GREY = "#68737D"
MUTED = "#98A2AB"
LIGHT = "#E7ECEF"
PALE_BLUE = "#EAF3F8"
PALE_GREEN = "#E8F6F1"
PALE_RED = "#FBEFEF"
PANEL = "#F7F9FA"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def pill(
    ax: Any,
    *,
    x: float,
    y: float,
    width: float,
    text: str,
    color: str,
    facecolor: str,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            0.085,
            boxstyle="round,pad=0.006,rounding_size=0.025",
            facecolor=facecolor,
            edgecolor="none",
        )
    )
    ax.text(
        x + width / 2,
        y + 0.043,
        text,
        fontsize=7.3,
        fontweight="bold",
        color=color,
        ha="center",
        va="center",
    )


def metric(
    ax: Any,
    *,
    y: float,
    value: str,
    label: str,
    detail: str,
    color: str,
) -> None:
    ax.text(
        0.08,
        y,
        value,
        fontsize=16.5,
        fontweight="bold",
        color=color,
        va="top",
    )
    ax.text(
        0.62,
        y - 0.005,
        label,
        fontsize=8.3,
        fontweight="bold",
        color=INK,
        va="top",
    )
    ax.text(
        0.62,
        y - 0.085,
        detail,
        fontsize=6.7,
        color=GREY,
        va="top",
    )


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    source = repo / "benchmarks" / "results" / "comparison"
    oom = load(source / "statevector_training_science_35q_1xa800_oom_v12.json")
    distributed = load(
        source / "statevector_training_science_35q_2node16xa800_v12.json"
    )
    report = load(source / "statevector_training_science_35q_capacity_report_v12.json")
    output = repo / "assets" / "readme" / "capacity-expansion"
    output.parent.mkdir(parents=True, exist_ok=True)

    assert report["capacity_expansion_demonstrated"] is True
    assert report["distribution_semantics"] == "sharded_across_ranks"
    assert report["forward_distribution_semantics"] == "sharded_across_ranks"
    assert report["backward_distribution_semantics"] == "sharded_across_ranks"
    assert report["full_state_materialization"] is False

    world = int(distributed["world_size"])
    device_gib = oom["device"]["total_memory_bytes"] / 2**30
    full_state_gib = 2**35 * 8 / 2**30
    base_shard_gib = full_state_gib / world
    peak_gib = report["distributed_peak_memory_bytes_per_rank"] / 2**30
    runtime = report["distributed_end_to_end_median_seconds"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.linewidth": 0.8,
            "svg.fonttype": "none",
        }
    )
    fig = plt.figure(figsize=(10.8, 4.35), facecolor="white")
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(1.62, 0.88),
        height_ratios=(0.34, 1),
        hspace=0.04,
        wspace=0.10,
        left=0.045,
        right=0.965,
        top=0.94,
        bottom=0.10,
    )
    header = fig.add_subplot(grid[0, :])
    capacity = fig.add_subplot(grid[1, 0])
    metrics = fig.add_subplot(grid[1, 1])

    for ax in (header, capacity, metrics):
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.axis("off")

    header.text(
        0,
        0.98,
        "FLAGQUANTUM  /  QUANTUM AI FOR FLAGOS",
        fontsize=7.6,
        fontweight="bold",
        color=GREEN,
        va="top",
    )
    header.text(
        1,
        0.98,
        "MEASURED DEVELOPMENT EVIDENCE",
        fontsize=7.4,
        fontweight="bold",
        color=ORANGE,
        ha="right",
        va="top",
    )
    header.text(
        0,
        0.64,
        "35 qubits: beyond the memory wall",
        fontsize=16,
        fontweight="bold",
        color=INK,
        va="top",
    )
    header.text(
        0,
        0.22,
        "One matched differentiable statevector · complex64 · value + full gradient",
        fontsize=8.8,
        color=GREY,
        va="top",
    )

    capacity.add_patch(
        FancyBboxPatch(
            (0.0, 0.03),
            0.99,
            0.93,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            facecolor=PANEL,
            edgecolor=LIGHT,
            linewidth=1,
        )
    )

    # One-GPU memory wall.
    capacity.text(
        0.045,
        0.86,
        "ONE GPU",
        fontsize=7.4,
        fontweight="bold",
        color=MUTED,
        va="top",
    )
    capacity.text(
        0.045,
        0.74,
        "The full state does not fit",
        fontsize=11.4,
        fontweight="bold",
        color=INK,
        va="top",
    )
    pill(
        capacity,
        x=0.80,
        y=0.705,
        width=0.13,
        text="CUDA OOM",
        color=RED,
        facecolor=PALE_RED,
    )

    bar_x, bar_y, bar_w, bar_h = 0.045, 0.57, 0.89, 0.075
    capacity.add_patch(
        FancyBboxPatch(
            (bar_x, bar_y),
            bar_w,
            bar_h,
            boxstyle="round,pad=0.002,rounding_size=0.012",
            facecolor="#F1D1D1",
            edgecolor="none",
        )
    )
    available_w = bar_w * device_gib / full_state_gib
    capacity.add_patch(
        FancyBboxPatch(
            (bar_x, bar_y),
            available_w,
            bar_h,
            boxstyle="round,pad=0.002,rounding_size=0.012",
            facecolor=RED,
            edgecolor="none",
        )
    )
    capacity.text(
        bar_x,
        bar_y - 0.055,
        f"{device_gib:.1f} GiB available",
        fontsize=7.6,
        color=RED,
        va="top",
    )
    capacity.text(
        bar_x + bar_w,
        bar_y - 0.055,
        f"{full_state_gib:.0f} GiB required",
        fontsize=7.6,
        fontweight="bold",
        color=INK,
        ha="right",
        va="top",
    )

    # Sixteen sharded ranks as one continuous logical state.
    capacity.text(
        0.045,
        0.40,
        "SIXTEEN GPUS  /  TWO NODES",
        fontsize=7.4,
        fontweight="bold",
        color=GREEN,
        va="top",
    )
    capacity.text(
        0.045,
        0.28,
        "One state, sixteen owned amplitude shards",
        fontsize=11.4,
        fontweight="bold",
        color=INK,
        va="top",
    )
    pill(
        capacity,
        x=0.76,
        y=0.245,
        width=0.17,
        text="COMPLETED",
        color=GREEN,
        facecolor=PALE_GREEN,
    )

    shard_y, shard_h = 0.095, 0.075
    gap = 0.004
    shard_w = (bar_w - gap * (world - 1)) / world
    for rank in range(world):
        x = bar_x + rank * (shard_w + gap)
        capacity.add_patch(
            Rectangle(
                (x, shard_y),
                shard_w,
                shard_h,
                facecolor=BLUE if rank < world // 2 else GREEN,
                edgecolor="none",
            )
        )
    capacity.plot(
        [bar_x + bar_w / 2, bar_x + bar_w / 2],
        [shard_y - 0.018, shard_y + shard_h + 0.018],
        color="white",
        linewidth=2.5,
        zorder=3,
    )
    capacity.text(
        bar_x,
        shard_y - 0.038,
        "NODE 1  ·  RANKS 0–7",
        fontsize=6.5,
        fontweight="bold",
        color=BLUE,
        va="top",
    )
    capacity.text(
        bar_x + bar_w,
        shard_y - 0.038,
        "NODE 2  ·  RANKS 8–15",
        fontsize=6.5,
        fontweight="bold",
        color=GREEN,
        ha="right",
        va="top",
    )

    metrics.add_patch(
        FancyBboxPatch(
            (0.0, 0.03),
            0.99,
            0.93,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            facecolor="white",
            edgecolor=LIGHT,
            linewidth=1,
        )
    )
    metrics.text(
        0.08,
        0.87,
        "WHAT CHANGED",
        fontsize=7.4,
        fontweight="bold",
        color=MUTED,
        va="top",
    )
    metric(
        metrics,
        y=0.73,
        value=f"{base_shard_gib:.0f} GiB",
        label="base state / rank",
        detail=r"$2^{35}$ amplitudes ÷ 16 ranks",
        color=BLUE,
    )
    metrics.plot([0.08, 0.92], [0.535, 0.535], color=LIGHT, linewidth=1)
    metric(
        metrics,
        y=0.46,
        value=f"{peak_gib:.1f} GiB",
        label="peak / rank",
        detail=f"of {device_gib:.1f} GiB available",
        color=BLUE,
    )
    metrics.plot([0.08, 0.92], [0.265, 0.265], color=LIGHT, linewidth=1)
    metric(
        metrics,
        y=0.19,
        value=f"{runtime:.2f} s",
        label="complete step",
        detail="forward + backward sharded",
        color=GREEN,
    )

    fig.savefig(f"{output}.png", dpi=240, facecolor="white")
    fig.savefig(f"{output}.svg", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
