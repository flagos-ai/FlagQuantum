#!/usr/bin/env python3
"""Create the compact distributed-statevector evidence figure used by README."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/flagquantum-matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

GREEN = "#009E73"
BLUE = "#0072B2"
ORANGE = "#E69F00"
INK = "#17202A"
GREY = "#68737D"
LIGHT = "#E8EDF1"
PALE_GREEN = "#E8F6F1"
PALE_BLUE = "#EAF3F8"
PALE_ORANGE = "#FFF4DF"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def measurement_range(payload: dict[str, Any]) -> tuple[float, float]:
    median = float(payload["value_and_grad"]["median_seconds"])
    samples = np.asarray(payload["value_and_grad"]["samples_seconds"], dtype=float)
    return median - float(samples.min()), float(samples.max()) - median


def metric_card(
    ax: Any,
    *,
    y: float,
    value: str,
    title: str,
    detail: str,
    color: str,
    background: str,
) -> None:
    ax.text(
        0.03,
        y,
        value,
        transform=ax.transAxes,
        fontsize=22,
        fontweight="bold",
        color=color,
        va="top",
        bbox={
            "boxstyle": "round,pad=0.45,rounding_size=0.12",
            "facecolor": background,
            "edgecolor": "none",
        },
    )
    ax.text(
        0.38,
        y - 0.005,
        title,
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=INK,
        va="top",
    )
    ax.text(
        0.38,
        y - 0.085,
        detail,
        transform=ax.transAxes,
        fontsize=8.2,
        color=GREY,
        va="top",
        linespacing=1.35,
    )


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    source = repo / "benchmarks" / "results" / "comparison"
    current = (
        repo
        / "benchmarks"
        / "results"
        / "legacy"
        / "statevector_mlsys_current"
        / "generality"
    )
    output = repo / "assets" / "readme" / "statevector-scaling"
    output.parent.mkdir(parents=True, exist_ok=True)

    gpu_counts = np.asarray([1, 2, 4, 8, 16])
    payloads = []
    for count in gpu_counts:
        rerun = current / f"flagquantum_31q_d8_linear_{count}gpu_final_rerun.json"
        payloads.append(
            load(rerun)
            if rerun.exists()
            else load(
                source / f"flagquantum_31q_d8_{count}xa800_fused_final_warm2_rep5.json"
            )
        )
    runtimes = np.asarray(
        [payload["value_and_grad"]["median_seconds"] for payload in payloads],
        dtype=float,
    )
    ranges = np.asarray([measurement_range(payload) for payload in payloads]).T
    speedup_16 = runtimes[0] / runtimes[4]
    improvement_8_to_16 = 100 * (runtimes[3] - runtimes[4]) / runtimes[3]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.32,
            "svg.fonttype": "none",
        }
    )

    fig = plt.figure(figsize=(10.8, 4.8), facecolor="white")
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(1.3, 1),
        height_ratios=(0.54, 1),
        hspace=0.16,
        wspace=0.18,
    )
    header = fig.add_subplot(grid[0, :])
    chart = fig.add_subplot(grid[1, 0])
    metrics = fig.add_subplot(grid[1, 1])

    header.set(xlim=(0, 1), ylim=(0, 1))
    header.axis("off")
    header.text(
        0,
        0.99,
        "FLAGQUANTUM  /  QUANTUM AI FOR FLAGOS",
        fontsize=7.8,
        fontweight="bold",
        color=GREEN,
        va="top",
    )
    header.text(
        0,
        0.75,
        "Differentiable statevector strong scaling",
        fontsize=16,
        fontweight="bold",
        color=INK,
        va="top",
    )
    header.text(
        0,
        0.40,
        "Fixed workload · 31 qubits · 8 layers · 248 trainable parameters",
        fontsize=9.2,
        color=GREY,
        va="top",
    )
    header.text(
        0,
        0.15,
        r"Complete value + full gradient · $\langle Z_{15}\rangle$ · complex64",
        fontsize=8.5,
        color=GREY,
        va="top",
    )
    header.text(
        0.995,
        0.99,
        "MEASURED DEVELOPMENT EVIDENCE",
        fontsize=7.6,
        fontweight="bold",
        color=ORANGE,
        ha="right",
        va="top",
    )

    wire_x0, wire_x1 = 0.70, 0.98
    wire_ys = (0.64, 0.40, 0.16)
    wire_labels = (r"$q_0$", r"$q_1$", r"$q_{30}$")
    header.add_patch(
        Rectangle(
            (wire_x0 + 0.012, 0.065),
            0.225,
            0.65,
            facecolor="#F7F9FA",
            edgecolor=GREY,
            linewidth=1.0,
            linestyle=(0, (4, 3)),
            zorder=0,
        )
    )
    for wire_y, wire_label in zip(wire_ys, wire_labels):
        header.text(
            wire_x0 - 0.018,
            wire_y,
            wire_label,
            ha="right",
            va="center",
            fontsize=7.2,
            color=INK,
        )
        header.plot(
            [wire_x0, wire_x1],
            [wire_y, wire_y],
            color=INK,
            linewidth=1.0,
            zorder=1,
        )
        header.add_patch(
            Rectangle(
                (wire_x0 + 0.025, wire_y - 0.075),
                0.050,
                0.15,
                facecolor="#B7DDF0",
                edgecolor=BLUE,
                linewidth=1.0,
                zorder=3,
            )
        )
        header.text(
            wire_x0 + 0.050,
            wire_y,
            r"$R_Y$",
            ha="center",
            va="center",
            fontsize=6.5,
            color=INK,
            zorder=4,
        )
    header.text(
        wire_x0 - 0.018,
        0.28,
        r"$\vdots$",
        ha="right",
        va="center",
        fontsize=8.5,
        color=INK,
    )
    cnot_x = wire_x0 + 0.15
    header.plot(
        [cnot_x, cnot_x],
        [wire_ys[1], wire_ys[0]],
        color=INK,
        linewidth=1.0,
        zorder=2,
    )
    header.scatter([cnot_x], [wire_ys[0]], s=25, color=INK, zorder=4)
    header.scatter(
        [cnot_x],
        [wire_ys[1]],
        s=80,
        facecolor="white",
        edgecolor=INK,
        linewidth=1.0,
        zorder=4,
    )
    header.text(
        cnot_x,
        wire_ys[1],
        "+",
        ha="center",
        va="center",
        fontsize=9,
        color=INK,
        zorder=5,
    )
    header.plot(
        [cnot_x + 0.065, cnot_x + 0.065],
        [wire_ys[2], wire_ys[1]],
        color=INK,
        linewidth=1.0,
        linestyle=":",
        zorder=2,
    )
    header.scatter(
        [cnot_x + 0.065],
        [wire_ys[1]],
        s=25,
        color=INK,
        zorder=4,
    )
    header.scatter(
        [cnot_x + 0.065],
        [wire_ys[2]],
        s=80,
        facecolor="white",
        edgecolor=INK,
        linewidth=1.0,
        zorder=4,
    )
    header.text(
        cnot_x + 0.065,
        wire_ys[2],
        "+",
        ha="center",
        va="center",
        fontsize=9,
        color=INK,
        zorder=5,
    )
    header.text(
        0.952,
        0.52,
        "× 8",
        ha="left",
        va="center",
        fontsize=10,
        fontweight="bold",
        color=INK,
    )

    chart.errorbar(
        gpu_counts,
        runtimes,
        yerr=ranges,
        fmt="o-",
        color=GREEN,
        markerfacecolor=GREEN,
        markeredgecolor="white",
        markeredgewidth=1.2,
        markersize=8,
        linewidth=2.8,
        capsize=3,
        zorder=4,
    )
    chart.fill_between(
        gpu_counts[:4],
        runtimes[:4],
        runtimes[0],
        color=PALE_GREEN,
        alpha=0.75,
        zorder=1,
    )
    chart.axvspan(11.3, 18, color=PALE_BLUE, alpha=0.7, zorder=0)
    chart.axvline(11.3, color=BLUE, linewidth=1.1, linestyle="--", zorder=2)
    chart.text(
        7.0,
        31.0,
        "1 node",
        color=GREY,
        fontsize=8,
        ha="center",
        va="bottom",
    )
    chart.text(
        15.2,
        31.0,
        "2 nodes",
        color=BLUE,
        fontsize=8,
        fontweight="bold",
        ha="center",
        va="bottom",
    )
    label_offsets = {
        2: (10, 14),
        4: (10, 14),
    }
    for count, runtime in zip(gpu_counts, runtimes):
        offset = label_offsets.get(int(count), (0, 10))
        chart.annotate(
            f"{runtime:.2f}s",
            (count, runtime),
            xytext=offset,
            textcoords="offset points",
            ha="left" if int(count) in label_offsets else "center",
            fontsize=8,
            fontweight="bold",
            color=INK,
        )
    chart.set(
        xlim=(0.2, 17.3),
        ylim=(0, 34),
        xticks=gpu_counts,
        xlabel="NVIDIA A800 GPUs",
        ylabel="Value + full gradient (seconds)",
    )
    chart.set_title("End-to-end runtime", loc="left", pad=9, color=INK)
    chart.grid(True, axis="y", zorder=0)
    chart.spines[["top", "right"]].set_visible(False)

    metrics.axis("off")
    metrics.set(xlim=(0, 1), ylim=(0, 1))
    metric_card(
        metrics,
        y=0.92,
        value=f"{speedup_16:.2f}×",
        title="strong scaling on 16 GPUs",
        detail="28.84s → 4.31s for the same\nvalue-and-full-gradient workload",
        color=GREEN,
        background=PALE_GREEN,
    )
    metric_card(
        metrics,
        y=0.60,
        value="4.31s",
        title="two-node end-to-end runtime",
        detail=f"{improvement_8_to_16:.1f}% below the 8-GPU runtime\nwhile preserving sharded execution",
        color=BLUE,
        background=PALE_BLUE,
    )
    metric_card(
        metrics,
        y=0.28,
        value="248",
        title="trainable parameter gradients",
        detail="amplitude-sharded forward and backward,\nnot replicated per-rank throughput",
        color=ORANGE,
        background=PALE_ORANGE,
    )

    fig.text(
        0.5,
        0.012,
        "A800-SXM4-80GB · median of 5 measured runs after 2 warmups · error bars show min–max",
        ha="center",
        va="bottom",
        fontsize=7.4,
        color=GREY,
    )
    fig.subplots_adjust(left=0.07, right=0.985, top=0.96, bottom=0.16)
    fig.savefig(output.with_suffix(".png"), dpi=220, facecolor="white")
    fig.savefig(output.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
