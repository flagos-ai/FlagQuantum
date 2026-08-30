#!/usr/bin/env python3
"""Plot the current auditable distributed-SV evidence in MLSys paper style."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/flagquantum-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

BLUE = "#0072B2"
SKY = "#56B4E9"
ORANGE = "#E69F00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
RED = "#D55E00"
GREY = "#6C757D"
LIGHT = "#E9ECEF"


def load(root: Path, name: str) -> dict[str, Any]:
    return json.loads((root / name).read_text(encoding="utf-8"))


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.titleweight": "bold",
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "grid.linewidth": 0.5,
            "grid.alpha": 0.28,
            "lines.linewidth": 1.8,
            "lines.markersize": 5.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def panel(ax: Any, label: str, title: str, *, grid: str = "y") -> None:
    ax.set_title(rf"$\bf{{{label}}}$  {title}", loc="left", pad=7)
    ax.grid(True, axis=grid, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)


def save(fig: Any, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".svg"):
        fig.savefig(output.with_suffix(suffix), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=320, bbox_inches="tight")
    plt.close(fig)


def timing(payload: dict[str, Any]) -> tuple[float, float, float]:
    return (
        payload["forward"]["median_seconds"],
        payload["backward"]["median_seconds"],
        payload["value_and_grad"]["median_seconds"],
    )


def fig0_final_scaling(root: Path) -> Any:
    gpu_counts = np.asarray([1, 2, 4, 8, 16])
    comparison_root = (
        root.parent / "legacy" / "statevector_mlsys_current" / "generality"
    )
    fq = []
    for count in gpu_counts:
        rerun = comparison_root / f"flagquantum_31q_d8_linear_{count}gpu_final_rerun.json"
        fq.append(
            json.loads(rerun.read_text(encoding="utf-8"))
            if rerun.exists()
            else load(root, f"flagquantum_31q_d8_{count}xa800_fused_final_warm2_rep5.json")
        )
    lightning = {
        count: load(
            root,
            f"pennylane_lightning_gpu_31q_d8_{count}xa800_mpi_final_warm2_rep5.json",
        )
        for count in (2, 4, 8, 16)
    }
    fq_time = np.asarray([item["value_and_grad"]["median_seconds"] for item in fq])
    pl_time = np.asarray(
        [np.nan, *[lightning[count]["value_and_grad"]["median_seconds"] for count in (2, 4, 8, 16)]]
    )

    def ranges(payloads: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
        medians = np.asarray(
            [item["value_and_grad"]["median_seconds"] for item in payloads]
        )
        samples = [item["value_and_grad"]["samples_seconds"] for item in payloads]
        return (
            medians - np.asarray([min(item) for item in samples]),
            np.asarray([max(item) for item in samples]) - medians,
        )

    fq_error = ranges(fq)
    pl_payloads = [lightning[count] for count in (2, 4, 8, 16)]
    pl_error = ranges(pl_payloads)
    fq_forward = np.asarray([item["forward"]["median_seconds"] for item in fq])
    pl_forward = np.asarray(
        [lightning[count]["forward"]["median_seconds"] for count in (2, 4, 8, 16)]
    )

    fig = plt.figure(figsize=(7.12, 5.05))
    grid = fig.add_gridspec(3, 2, height_ratios=(0.46, 1, 1))
    header = fig.add_subplot(grid[0, :])
    axes = np.asarray(
        [
            [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])],
            [fig.add_subplot(grid[2, 0]), fig.add_subplot(grid[2, 1])],
        ]
    )
    header.set(xlim=(0, 1), ylim=(0, 1))
    header.axis("off")
    header.text(
        0.01,
        0.88,
        "Matched differentiable full-width linear HEA",
        fontsize=9,
        weight="bold",
        va="top",
    )
    header.text(
        0.01,
        0.55,
        "31 qubits · 8 layers · 248 trainable RY parameters · 488 gates",
        fontsize=7.2,
        va="top",
    )
    header.text(
        0.01,
        0.27,
        r"Directed nearest-neighbor CNOT chain · $\langle Z_{15}\rangle$ · complex64",
        fontsize=7.0,
        va="top",
        color=GREY,
    )

    wire_x0, wire_x1 = 0.59, 0.94
    wire_ys = (0.75, 0.57, 0.34, 0.16)
    wire_labels = (r"$q_0$", r"$q_1$", r"$\vdots$", r"$q_{30}$")
    for wire_y, wire_label in zip(wire_ys, wire_labels):
        header.text(wire_x0 - 0.018, wire_y, wire_label, ha="right", va="center", fontsize=6.2)
        if wire_label == r"$\vdots$":
            continue
        header.plot(
            [wire_x0, wire_x1],
            [wire_y, wire_y],
            color="black",
            linewidth=0.65,
            zorder=0,
        )
        header.add_patch(
            Rectangle(
                (0.625, wire_y - 0.065),
                0.045,
                0.13,
                facecolor=SKY,
                edgecolor=BLUE,
                linewidth=0.6,
                zorder=3,
            )
        )
        header.text(
            0.6475,
            wire_y,
            r"$R_Y$",
            ha="center",
            va="center",
            fontsize=5.2,
            zorder=4,
        )
    header.plot(
        [0.735, 0.735],
        [wire_ys[1], wire_ys[0]],
        color="black",
        linewidth=0.7,
        zorder=2,
    )
    header.scatter([0.735], [wire_ys[0]], s=10, color="black", zorder=4)
    header.scatter(
        [0.735],
        [wire_ys[1]],
        s=28,
        facecolor="white",
        edgecolor="black",
        linewidth=0.65,
        zorder=4,
    )
    header.text(0.735, wire_ys[1], "+", ha="center", va="center", fontsize=6.0)
    header.plot(
        [0.80, 0.80],
        [wire_ys[-1], wire_ys[1]],
        color="black",
        linewidth=0.65,
        linestyle=":",
        zorder=2,
    )
    header.scatter([0.80], [wire_ys[1]], s=10, color="black", zorder=4)
    header.scatter(
        [0.80],
        [wire_ys[-1]],
        s=28,
        facecolor="white",
        edgecolor="black",
        linewidth=0.65,
        zorder=4,
    )
    header.text(0.80, wire_ys[-1], "+", ha="center", va="center", fontsize=6.0)
    header.add_patch(
        Rectangle(
            (0.602, 0.06),
            0.245,
            0.79,
            fill=False,
            edgecolor=GREY,
            linewidth=0.6,
            linestyle="--",
            zorder=1,
        )
    )
    header.text(0.858, 0.46, r"$\times\,8$", fontsize=8, weight="bold", va="center")
    header.text(0.94, 0.91, "schematic", ha="right", fontsize=5.7, color=GREY)
    axes[0, 0].errorbar(
        gpu_counts,
        fq_time,
        yerr=np.vstack(fq_error),
        fmt="o-",
        capsize=2,
        color=GREEN,
        label="FlagQuantum",
        zorder=4,
    )
    axes[0, 0].errorbar(
        gpu_counts[1:],
        pl_time[1:],
        yerr=np.vstack(pl_error),
        fmt="s--",
        capsize=2,
        color=GREY,
        label="PennyLane Lightning-GPU",
        zorder=3,
    )
    axes[0, 0].annotate(
        "PennyLane 1-GPU: failed",
        (1, 43),
        xytext=(8, 0),
        textcoords="offset points",
        color=GREY,
        fontsize=5.9,
        va="center",
    )
    axes[0, 0].set(
        xscale="log",
        yscale="log",
        xticks=gpu_counts,
        xticklabels=[str(item) for item in gpu_counts],
        ylabel="Value + full gradient (s)",
    )
    axes[0, 0].legend(
        frameon=False,
        ncol=1,
        loc="upper left",
        fontsize=6.1,
        labelspacing=0.30,
        handletextpad=0.55,
    )
    panel(axes[0, 0], "a", "31q end-to-end runtime", grid="both")

    fq_speedup = fq_time[0] / fq_time
    pl_speedup = pl_time[1] / pl_time
    axes[0, 1].plot(
        gpu_counts,
        fq_speedup,
        "o-",
        color=GREEN,
        label="FlagQuantum vs. 1 GPU",
    )
    axes[0, 1].plot(
        gpu_counts[1:],
        pl_speedup[1:],
        "s--",
        color=GREY,
        label="PennyLane vs. 2 GPUs",
    )
    axes[0, 1].plot(gpu_counts, gpu_counts, ":", color=BLUE, label="Ideal from 1 GPU")
    axes[0, 1].axvline(8 * np.sqrt(2), color=RED, linewidth=0.8, alpha=0.7)
    axes[0, 1].text(
        0.98,
        1.025,
        "1 node  |  2 nodes",
        transform=axes[0, 1].transAxes,
        ha="right",
        va="bottom",
        color=RED,
        fontsize=6.2,
        clip_on=False,
    )
    axes[0, 1].set(
        xscale="log",
        yscale="log",
        xticks=gpu_counts,
        xticklabels=[str(item) for item in gpu_counts],
        xlabel="GPUs",
        ylabel="Strong-scaling speedup",
    )
    axes[0, 1].legend(
        frameon=False,
        fontsize=5.7,
        loc="upper left",
        labelspacing=0.28,
        handletextpad=0.45,
    )
    panel(axes[0, 1], "b", "Strong scaling", grid="both")

    phase_x = np.asarray([0.0, 0.82, 2.0, 2.82])
    fq_indices = (3, 4)
    pl_indices = (2, 3)
    for x_pos, index in zip(phase_x[[0, 2]], fq_indices):
        forward_pct = 100 * fq_forward[index] / fq_time[index]
        backward_pct = 100 - forward_pct
        axes[1, 0].bar(x_pos, forward_pct, width=0.62, color=GREEN, zorder=3)
        axes[1, 0].bar(
            x_pos,
            backward_pct,
            bottom=forward_pct,
            width=0.62,
            color=BLUE,
            zorder=3,
        )
        axes[1, 0].text(
            x_pos,
            forward_pct / 2,
            f"Forward\n{forward_pct:.0f}%",
            ha="center",
            va="center",
            color="white",
            fontsize=5.2,
        )
        axes[1, 0].text(
            x_pos,
            forward_pct + backward_pct / 2,
            f"Backward\n{backward_pct:.0f}%",
            ha="center",
            va="center",
            color="white",
            fontsize=5.2,
        )
    for x_pos, index, total_index in zip(
        phase_x[[1, 3]], pl_indices, (3, 4)
    ):
        qnode_pct = 100 * pl_forward[index] / pl_time[total_index]
        torch_pct = 100 - qnode_pct
        axes[1, 0].bar(x_pos, qnode_pct, width=0.62, color=GREY, zorder=3)
        axes[1, 0].bar(
            x_pos,
            torch_pct,
            bottom=qnode_pct,
            width=0.62,
            color=RED,
            zorder=4,
        )
        axes[1, 0].text(
            x_pos,
            50,
            f"QNode incl.\nadjoint\n{qnode_pct:.2f}%",
            ha="center",
            va="center",
            color="white",
            fontsize=5.0,
        )
    axes[1, 0].set(
        xticks=phase_x,
        xticklabels=[
            "FlagQuantum\n8 GPUs",
            "PennyLane\n8 GPUs",
            "FlagQuantum\n16 GPUs",
            "PennyLane\n16 GPUs",
        ],
        ylabel="Framework-observed time share (%)",
        ylim=(0, 108),
    )
    axes[1, 0].axvline(1.41, color=LIGHT, linewidth=0.8)
    axes[1, 0].text(0.205, 0.94, "1 node", transform=axes[1, 0].transAxes, ha="center", fontsize=5.7)
    axes[1, 0].text(0.755, 0.94, "2 nodes", transform=axes[1, 0].transAxes, ha="center", fontsize=5.7)
    panel(axes[1, 0], "c", "Where framework time is recorded")

    compared_counts = np.asarray([2, 4, 8, 16])
    relative = pl_time[1:] / fq_time[1:]
    bar_colors = [BLUE, BLUE, BLUE, RED]
    bars = axes[1, 1].bar(compared_counts.astype(str), relative, color=bar_colors, zorder=3)
    for bar, value in zip(bars, relative):
        axes[1, 1].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.28,
            f"{value:.2f}×",
            ha="center",
            fontsize=6.6,
            weight="bold",
        )
    axes[1, 1].axhline(1, color="black", linewidth=0.8)
    axes[1, 1].set(
        xlabel="GPUs",
        ylabel="PennyLane time / FlagQuantum time",
        ylim=(0, 16.3),
    )
    axes[1, 1].text(
        3,
        12.2,
        "cross-node\ncliff",
        ha="center",
        va="center",
        color="white",
        fontsize=6.0,
        weight="bold",
    )
    panel(axes[1, 1], "d", "Relative performance")
    fig.tight_layout(h_pad=1.25, w_pad=1.4)
    return fig


def fig1_current_performance(root: Path) -> Any:
    fq8 = load(root, "flagquantum_31q_d8_8xa800_fused_final_warm2_rep5.json")
    fq16 = load(root, "flagquantum_31q_d8_16xa800_fused_final_warm2_rep5.json")
    lightning8 = load(
        root, "pennylane_lightning_gpu_31q_d8_8xa800_mpi_final_warm2_rep5.json"
    )
    lightning16 = load(
        root, "pennylane_lightning_gpu_31q_d8_16xa800_mpi_final_warm2_rep5.json"
    )
    variants = [
        (
            "No cache",
            load(root, "flagquantum_31q_d8_16xa800_2node_optimized_warm5.json"),
            GREY,
        ),
        (
            "Cached",
            load(root, "flagquantum_31q_d8_16xa800_2node_cached_nccl_ch8_warm3.json"),
            BLUE,
        ),
        (
            "Inter-node ckpt.",
            load(
                root,
                "flagquantum_31q_d8_16xa800_2node_cached_nccl_ch8_ket_checkpoint_warm3.json",
            ),
            ORANGE,
        ),
        (
            "All ckpts.",
            load(
                root,
                "flagquantum_31q_d8_16xa800_2node_cached_nccl_ch8_all_checkpoint_warm3.json",
            ),
            PURPLE,
        ),
        ("+ transpose + 1q", fq16, GREEN),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.12, 2.35))

    labels = ["8 GPUs\n1 node", "16 GPUs\n2 nodes"]
    forward = [timing(fq8)[0], timing(fq16)[0]]
    backward = [timing(fq8)[1], timing(fq16)[1]]
    x = np.arange(2)
    axes[0].bar(x, forward, color=BLUE, zorder=3)
    axes[0].bar(x, backward, bottom=forward, color=ORANGE, zorder=3)
    for index, total in enumerate(np.add(forward, backward)):
        axes[0].text(index, total + 0.10, f"{total:.2f}s", ha="center", weight="bold")
    axes[0].text(0, forward[0] / 2, "Forward", color="white", ha="center", va="center", fontsize=6.5)
    axes[0].text(
        0,
        forward[0] + backward[0] / 2,
        "Backward",
        color="white",
        ha="center",
        va="center",
        fontsize=6.5,
    )
    axes[0].set(xticks=x, xticklabels=labels, ylabel="Value + full gradient (s)")
    axes[0].set_ylim(0, 7.0)
    panel(axes[0], "a", "31q differentiable SV")

    group_x = np.arange(2)
    offset = 0.14
    lightning_time = np.asarray(
        [
            lightning8["value_and_grad"]["median_seconds"],
            lightning16["value_and_grad"]["median_seconds"],
        ]
    )
    fq_time = np.asarray(
        [
            fq8["value_and_grad"]["median_seconds"],
            fq16["value_and_grad"]["median_seconds"],
        ]
    )
    for index in group_x:
        axes[1].plot(
            [index - offset, index + offset],
            [lightning_time[index], fq_time[index]],
            color=LIGHT,
            linewidth=1.0,
            zorder=2,
        )
    axes[1].scatter(
        group_x - offset,
        lightning_time,
        marker="s",
        s=38,
        color=GREY,
        label="PennyLane Lightning-GPU",
        zorder=4,
    )
    axes[1].scatter(
        group_x + offset,
        fq_time,
        marker="o",
        s=38,
        color=GREEN,
        label="FlagQuantum",
        zorder=4,
    )
    for index, value in enumerate(lightning_time):
        axes[1].annotate(
            f"{value:.2f}s",
            (index - offset, value),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=6.0,
            weight="bold",
            color=GREY,
        )
    for index, value in enumerate(fq_time):
        axes[1].annotate(
            f"{value:.2f}s",
            (index + offset, value),
            xytext=(7, -1),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=6.0,
            weight="bold",
            color=GREEN,
        )
    axes[1].set(
        yscale="log",
        xticks=group_x,
        xticklabels=["8 GPUs\n1 node", "16 GPUs\n2 nodes"],
        ylabel="Median time (s)",
        xlim=(-0.42, 1.42),
        ylim=(3.8, 115),
    )
    handles, legend_labels = axes[1].get_legend_handles_labels()
    axes[1].legend(
        [handles[1], handles[0]],
        [legend_labels[1], legend_labels[0]],
        frameon=False,
        fontsize=5.3,
        loc="upper left",
    )
    panel(axes[1], "b", "Differentiable baseline")

    annotation_style = {
        "No cache": ((5, 5), "left"),
        "Cached": ((5, 5), "left"),
        "Inter-node ckpt.": ((5, 5), "left"),
        "All ckpts.": ((3, 9), "left"),
        "+ transpose + 1q": ((-5, -2), "right"),
    }
    for label, payload, color in variants:
        total = payload["value_and_grad"]["median_seconds"]
        memory = payload["memory"]["peak_bytes_per_rank_max"] / 2**30
        axes[2].scatter(memory, total, s=42, color=color, edgecolor="white", zorder=4)
        offset, alignment = annotation_style[label]
        axes[2].annotate(
            label,
            (memory, total),
            xytext=offset,
            textcoords="offset points",
            fontsize=6.3,
            ha=alignment,
            color=color,
        )
    axes[2].set(xlabel="Peak memory / GPU (GiB)", ylabel="Median time (s)")
    axes[2].set(xlim=(0, 96), ylim=(4.7, 20.8))
    panel(axes[2], "c", "16-GPU time–memory Pareto", grid="both")
    fig.tight_layout(w_pad=1.4)
    return fig


def fig2_optimization_journey(root: Path) -> Any:
    milestones = [
        ("Original", "flagquantum_31q_d8_8xa800_value_grad_smoke.json"),
        ("Persistent\nlayout", "flagquantum_31q_d8_8xa800_persistent_layout_smoke.json"),
        ("In-place\nbuffers", "flagquantum_31q_d8_8xa800_persistent_full_inplace_smoke.json"),
        ("Block-4\nfusion", "flagquantum_31q_d8_8xa800_local_block4_smoke.json"),
        ("Triton\nCNOT", "flagquantum_31q_d8_8xa800_block4_triton_cx_smoke.json"),
        ("Single-pass\nVJP", "flagquantum_31q_d8_8xa800_reversible_ry_fused_warm5.json"),
        ("Planner\ncache", "flagquantum_31q_d8_8xa800_final_warm5.json"),
        ("Transpose\n+ 1q", "flagquantum_31q_d8_8xa800_fused_transpose_warm3.json"),
    ]
    payloads = [load(root, filename) for _, filename in milestones]
    values = [payload["value_and_grad"]["median_seconds"] for payload in payloads]
    forward = [payload["forward"]["median_seconds"] for payload in payloads]
    backward = [payload["backward"]["median_seconds"] for payload in payloads]
    speedups = [values[0] / value for value in values]
    x = np.arange(len(milestones))
    fig, axes = plt.subplots(1, 2, figsize=(7.12, 2.45), gridspec_kw={"width_ratios": [1.55, 1]})

    axes[0].bar(x, forward, color=BLUE, label="Forward", zorder=3)
    axes[0].bar(x, backward, bottom=forward, color=ORANGE, label="Backward", zorder=3)
    for index, value in enumerate(values):
        axes[0].text(index, value + 1.2, f"{value:.1f}", ha="center", fontsize=6.2)
    axes[0].set(
        xticks=x,
        xticklabels=[label for label, _ in milestones],
        ylabel="Value + full gradient (s)",
        ylim=(0, 65),
    )
    axes[0].tick_params(axis="x", labelsize=5.7)
    axes[0].legend(frameon=False, ncol=2, loc="upper right")
    panel(axes[0], "a", "End-to-end optimization stack")

    axes[1].plot(x, speedups, "o-", color=GREEN, zorder=3)
    axes[1].fill_between(x, 1, speedups, color=GREEN, alpha=0.12)
    for index, value in enumerate(speedups):
        axes[1].annotate(
            f"{value:.1f}×",
            (index, value),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=6.4,
            weight="bold" if index == len(speedups) - 1 else "normal",
        )
    axes[1].set(
        xticks=x,
        xticklabels=[str(index + 1) for index in x],
        xlabel="Optimization stage",
        ylabel="Speedup vs. original",
        ylim=(0, max(speedups) * 1.16),
    )
    axes[1].text(
        0.97,
        0.05,
        "31q, depth 8, 8×A800\nvalue + 248 gradients",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=6.5,
        color=GREY,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
    )
    panel(axes[1], "b", "Cumulative speedup")
    fig.tight_layout(w_pad=1.5)
    return fig


def fig3_multinode_diagnosis(root: Path) -> Any:
    profile_default = load(root, "flagquantum_31q_16xa800_layout_swap_profile.json")
    profile8 = load(root, "flagquantum_31q_16xa800_layout_swap_profile_ch8.json")
    profile16 = load(root, "flagquantum_31q_16xa800_layout_swap_profile_ch16.json")
    evolution = [
        ("Initial", "flagquantum_31q_d8_16xa800_2node_optimized_warm5.json"),
        ("P2P\nch.", "flagquantum_31q_d8_16xa800_2node_nccl_ch8_warm3.json"),
        ("Planner\ncache", "flagquantum_31q_d8_16xa800_2node_cached_nccl_ch8_warm3.json"),
        (
            "Inter-node\nckpt.",
            "flagquantum_31q_d8_16xa800_2node_cached_nccl_ch8_ket_checkpoint_warm3.json",
        ),
        (
            "All\nckpts.",
            "flagquantum_31q_d8_16xa800_2node_cached_nccl_ch16_all_checkpoint_warm3.json",
        ),
        ("Transpose\n+ 1q", "flagquantum_31q_d8_16xa800_fused_transpose_warm3.json"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.12, 2.35))

    configs = [("Default", profile_default), ("8 ch.", profile8), ("16 ch.", profile16)]
    x = np.arange(3)
    intra = [payload["results"]["intra_node"]["p2p"]["median_seconds"] * 1e3 for _, payload in configs]
    inter = [payload["results"]["inter_node"]["p2p"]["median_seconds"] * 1e3 for _, payload in configs]
    width = 0.35
    axes[0].bar(x - width / 2, intra, width, color=BLUE, label="Intra-node", zorder=3)
    axes[0].bar(x + width / 2, inter, width, color=RED, label="Inter-node", zorder=3)
    axes[0].set(
        xticks=x,
        xticklabels=[label for label, _ in configs],
        ylabel="0.5-GiB P2P (ms)",
        ylim=(0, 130),
    )
    axes[0].legend(
        frameon=False,
        loc="upper center",
        ncol=2,
        fontsize=6.0,
        columnspacing=0.8,
        handletextpad=0.4,
    )
    panel(axes[0], "a", "Transport latency")

    bytes_per_exchange = profile_default["half_shard_bytes"]
    intra_bw = [bytes_per_exchange / 1e9 / (value / 1e3) for value in intra]
    inter_bw = [bytes_per_exchange / 1e9 / (value / 1e3) for value in inter]
    axes[1].plot(x, intra_bw, "o-", color=BLUE, label="Intra-node", zorder=3)
    axes[1].plot(x, inter_bw, "s-", color=RED, label="Inter-node", zorder=3)
    axes[1].set(
        xticks=x,
        xticklabels=[label for label, _ in configs],
        ylabel="Effective bandwidth (GB/s)",
    )
    axes[1].legend(frameon=False, loc="upper left")
    panel(axes[1], "b", "Effective P2P bandwidth")

    payloads = [load(root, filename) for _, filename in evolution]
    totals = [payload["value_and_grad"]["median_seconds"] for payload in payloads]
    colors = [GREY, SKY, BLUE, ORANGE, PURPLE, GREEN]
    stage_labels = [
        "Initial",
        "P2P channels",
        "Planner cache",
        "Inter-node ckpt.",
        "All ckpts.",
        "Transpose + 1q",
    ]
    stage_y = np.arange(len(evolution))
    bars = axes[2].barh(stage_y, totals, color=colors, zorder=3)
    for bar, value in zip(bars, totals):
        axes[2].text(
            value + 0.35,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}",
            ha="left",
            va="center",
            fontsize=6.3,
        )
    axes[2].set(
        yticks=stage_y,
        yticklabels=stage_labels,
        xlabel="31q value + gradient (s)",
        xlim=(0, 22),
    )
    axes[2].invert_yaxis()
    axes[2].tick_params(axis="y", labelsize=5.5, pad=2)
    panel(axes[2], "c", "16-GPU optimization sequence", grid="x")
    fig.tight_layout(w_pad=1.3)
    return fig


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    root = repo / "benchmarks" / "results" / "comparison"
    output = (
        repo / "benchmarks" / "results" / "legacy" / "statevector_mlsys_current"
    )
    style()
    figures = {
        "fig0_final_scaling": fig0_final_scaling(root),
        "fig1_current_performance": fig1_current_performance(root),
        "fig2_optimization_journey": fig2_optimization_journey(root),
        "fig3_multinode_diagnosis": fig3_multinode_diagnosis(root),
    }
    for name, figure in figures.items():
        save(figure, output / name)
    manifest = {
        "status": "measured_development_evidence",
        "figures": list(figures),
        "note": (
            "The matched final 1/2/4/8/16-GPU FlagQuantum matrix and "
            "2/4/8/16-GPU Lightning matrix use two warmups and five measured "
            "samples. Profiler and capacity/OOM evidence remain pending."
        ),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
