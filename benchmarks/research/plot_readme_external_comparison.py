#!/usr/bin/env python3
"""Create the compact matched external-comparison figure used by README."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/flagquantum-matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

GREEN = "#009E73"
PURPLE = "#CC79A7"
GREY = "#68737D"
ORANGE = "#E69F00"
INK = "#17202A"
LIGHT = "#E8EDF1"
PALE_GREEN = "#E8F6F1"
PALE_GREY = "#F0F2F4"
PALE_PURPLE = "#F8EDF4"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def measurement(
    payload: dict[str, Any],
) -> tuple[float, tuple[float, float]]:
    median = float(payload["value_and_grad"]["median_seconds"])
    samples = np.asarray(payload["value_and_grad"]["samples_seconds"], dtype=float)
    return median, (median - float(samples.min()), float(samples.max()) - median)


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
    ax.add_patch(
        FancyBboxPatch(
            (0.02, y - 0.26),
            0.96,
            0.27,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            transform=ax.transAxes,
            facecolor=background,
            edgecolor="none",
            zorder=0,
        )
    )
    ax.text(
        0.05,
        y - 0.01,
        value,
        transform=ax.transAxes,
        fontsize=18,
        fontweight="bold",
        color=color,
        va="top",
        zorder=1,
    )
    ax.text(
        0.05,
        y - 0.125,
        title,
        transform=ax.transAxes,
        fontsize=9.2,
        fontweight="bold",
        color=INK,
        va="top",
        zorder=1,
    )
    ax.text(
        0.05,
        y - 0.205,
        detail,
        transform=ax.transAxes,
        fontsize=7.4,
        color=GREY,
        va="top",
        linespacing=1.35,
        zorder=1,
    )


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    comparison = repo / "benchmarks" / "results" / "comparison"
    current = (
        repo
        / "benchmarks"
        / "results"
        / "legacy"
        / "statevector_mlsys_current"
        / "generality"
    )
    tqd_root = (
        repo
        / "benchmarks"
        / "results"
        / "legacy"
        / "statevector_mlsys_current"
        / "tqd_current"
    )
    output = repo / "assets" / "readme" / "external-comparison"
    output.parent.mkdir(parents=True, exist_ok=True)

    gpu_counts = np.asarray([2, 4, 8, 16])
    flagquantum = []
    pennylane = []
    tqd = []
    for count in gpu_counts:
        rerun = current / f"flagquantum_31q_d8_linear_{count}gpu_final_rerun.json"
        flagquantum.append(
            load(rerun)
            if rerun.exists()
            else load(
                comparison
                / f"flagquantum_31q_d8_{count}xa800_fused_final_warm2_rep5.json"
            )
        )
        pennylane.append(
            load(
                comparison
                / (
                    "pennylane_lightning_gpu_31q_d8_"
                    f"{count}xa800_mpi_final_warm2_rep5.json"
                )
            )
        )
        tqd.append(load(tqd_root / f"tqd_31q_{count}gpu_2plus5.json"))

    fq_measurements = [measurement(payload) for payload in flagquantum]
    pl_measurements = [measurement(payload) for payload in pennylane]
    tqd_measurements = [measurement(payload) for payload in tqd]
    fq_time = np.asarray([item[0] for item in fq_measurements])
    pl_time = np.asarray([item[0] for item in pl_measurements])
    tqd_time = np.asarray([item[0] for item in tqd_measurements])
    fq_error = np.asarray([item[1] for item in fq_measurements]).T
    pl_error = np.asarray([item[1] for item in pl_measurements]).T
    tqd_error = np.asarray([item[1] for item in tqd_measurements]).T

    pl_ratio = pl_time / fq_time
    tqd_ratio = tqd_time / fq_time

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.32,
            "svg.fonttype": "none",
        }
    )

    fig = plt.figure(figsize=(10.8, 4.15), facecolor="white")
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(1.34, 1),
        height_ratios=(0.48, 1),
        hspace=0.16,
        wspace=0.20,
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
        0.72,
        "Matched external performance",
        fontsize=16,
        fontweight="bold",
        color=INK,
        va="top",
    )
    header.text(
        0,
        0.30,
        "Same 31-qubit workload · complete value + full gradient · lower is better",
        fontsize=8.8,
        color=GREY,
        va="top",
    )
    header.text(
        0.995,
        0.96,
        "MEASURED DEVELOPMENT EVIDENCE",
        fontsize=7.6,
        fontweight="bold",
        color=ORANGE,
        ha="right",
        va="top",
    )

    wire_x0, wire_x1 = 0.70, 0.98
    wire_ys = (0.62, 0.39, 0.16)
    wire_labels = (r"$q_0$", r"$q_1$", r"$q_{30}$")
    header.add_patch(
        Rectangle(
            (wire_x0 + 0.012, 0.06),
            0.225,
            0.63,
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
            fontsize=7.0,
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
                (wire_x0 + 0.025, wire_y - 0.07),
                0.050,
                0.14,
                facecolor="#B7DDF0",
                edgecolor="#0072B2",
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
            fontsize=6.2,
            color=INK,
            zorder=4,
        )
    header.text(
        wire_x0 - 0.018,
        0.275,
        r"$\vdots$",
        ha="right",
        va="center",
        fontsize=8.2,
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
    header.scatter([cnot_x], [wire_ys[0]], s=22, color=INK, zorder=4)
    header.scatter(
        [cnot_x],
        [wire_ys[1]],
        s=72,
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
        fontsize=8.5,
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
        s=22,
        color=INK,
        zorder=4,
    )
    header.scatter(
        [cnot_x + 0.065],
        [wire_ys[2]],
        s=72,
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
        fontsize=8.5,
        color=INK,
        zorder=5,
    )
    header.text(
        0.952,
        0.50,
        "× 8",
        ha="left",
        va="center",
        fontsize=10,
        fontweight="bold",
        color=INK,
    )

    series = (
        ("FlagQuantum", fq_time, fq_error, GREEN, "o", "-"),
        ("PennyLane Lightning-GPU", pl_time, pl_error, GREY, "s", "--"),
        ("TorchQuantum-Dist", tqd_time, tqd_error, PURPLE, "D", "-"),
    )
    for label, values, errors, color, marker, linestyle in series:
        chart.errorbar(
            gpu_counts,
            values,
            yerr=errors,
            label=label,
            color=color,
            marker=marker,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=1.0,
            markersize=7,
            linewidth=2.2,
            linestyle=linestyle,
            capsize=3,
            zorder=4,
        )
    chart.axvspan(11.3, 18, color="#EAF3F8", alpha=0.75, zorder=0)
    chart.axvline(11.3, color="#0072B2", linewidth=1.0, linestyle="--", zorder=1)
    chart.text(
        15.2,
        520,
        "2 nodes",
        color="#0072B2",
        fontsize=8,
        fontweight="bold",
        ha="center",
    )
    chart.set(
        xlim=(1.3, 17.2),
        ylim=(3, 650),
        yscale="log",
        xticks=gpu_counts,
        xlabel="NVIDIA A800 GPUs",
        ylabel="Value + full gradient (seconds, log scale)",
    )
    chart.grid(True, axis="y", which="major", zorder=0)
    chart.spines[["top", "right"]].set_visible(False)
    chart.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        columnspacing=1.25,
        handletextpad=0.45,
        fontsize=7.5,
    )
    chart.annotate(
        "TQD 16 GPU: CV 73.7%",
        (16, tqd_time[-1]),
        xytext=(-10, 18),
        textcoords="offset points",
        ha="right",
        color=PURPLE,
        fontsize=7.2,
    )

    metrics.axis("off")
    metrics.set(xlim=(0, 1), ylim=(0, 1))
    metric_card(
        metrics,
        y=0.99,
        value=f"{pl_ratio[:3].min():.2f}–{pl_ratio[:3].max():.2f}×",
        title="lower runtime vs. PennyLane",
        detail="Matched 2–8 GPU single-node runs",
        color=GREEN,
        background=PALE_GREEN,
    )
    metric_card(
        metrics,
        y=0.63,
        value=f"{pl_ratio[-1]:.2f}×",
        title="lower runtime at 16 GPUs",
        detail="FlagQuantum 4.31s · PennyLane 87.22s",
        color=GREY,
        background=PALE_GREY,
    )
    metric_card(
        metrics,
        y=0.27,
        value=f"{tqd_ratio.min():.1f}–{tqd_ratio.max():.1f}×",
        title="lower runtime vs. TQD",
        detail="2–16 GPUs · 16-GPU TQD result is high-variance",
        color=PURPLE,
        background=PALE_PURPLE,
    )

    fig.text(
        0.5,
        0.012,
        "A800-SXM4-80GB · 2 warmups + 5 measured runs · error bars show min–max",
        ha="center",
        va="bottom",
        fontsize=7.4,
        color=GREY,
    )
    fig.subplots_adjust(left=0.07, right=0.985, top=0.96, bottom=0.18)
    fig.savefig(output.with_suffix(".png"), dpi=220, facecolor="white")
    fig.savefig(output.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
