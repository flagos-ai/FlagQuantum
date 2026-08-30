#!/usr/bin/env python3
"""Plot generality, root-cause, and scientific-capacity SV evidence."""

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

BLUE = "#0072B2"
SKY = "#56B4E9"
ORANGE = "#E69F00"
GREEN = "#009E73"
RED = "#D55E00"
GREY = "#6C757D"
LIGHT = "#E9ECEF"


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
            "grid.alpha": 0.25,
            "lines.linewidth": 1.8,
            "lines.markersize": 5.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def panel(ax: Any, label: str, title: str, grid: str = "y") -> None:
    ax.set_title(rf"$\bf{{{label}}}$  {title}", loc="left", pad=7)
    ax.grid(True, axis=grid, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)


def save(fig: Any, stem: Path) -> None:
    for suffix in (".pdf", ".svg"):
        fig.savefig(stem.with_suffix(suffix), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=320, bbox_inches="tight")
    plt.close(fig)


def samples_error(payloads: list[dict[str, Any]]) -> np.ndarray:
    med = np.asarray([p["value_and_grad"]["median_seconds"] for p in payloads])
    samples = [p["value_and_grad"]["samples_seconds"] for p in payloads]
    return np.vstack(
        (
            med - np.asarray([min(x) for x in samples]),
            np.asarray([max(x) for x in samples]) - med,
        )
    )


def plot_generality(repo: Path, output: Path) -> None:
    comparison = repo / "benchmarks/results/comparison"
    root = output / "generality"
    counts = np.asarray([1, 2, 4, 8, 16])
    def preferred(topology: str, n: int) -> Path:
        base = root / f"flagquantum_31q_d8_{topology}_{n}gpu_warm2_rep5.json"
        final = root / f"flagquantum_31q_d8_{topology}_{n}gpu_final_rerun.json"
        optimized = root / f"flagquantum_31q_d8_{topology}_{n}gpu_dependency_schedule_warm2_rep5.json"
        dag_cx = root / f"flagquantum_31q_d8_{topology}_{n}gpu_dag_cxsegment_warm2_rep5.json"
        return final if final.exists() else dag_cx if dag_cx.exists() else optimized if optimized.exists() else base
    circuits: dict[str, list[dict[str, Any]]] = {
        "Linear": [
            read(
                root / f"flagquantum_31q_d8_linear_{n}gpu_final_rerun.json"
                if (root / f"flagquantum_31q_d8_linear_{n}gpu_final_rerun.json").exists()
                else comparison / f"flagquantum_31q_d8_{n}xa800_fused_final_warm2_rep5.json"
            )
            for n in counts
        ],
        "Ring": [
            read(preferred("ring", n))
            for n in counts
        ],
        "Brickwork": [
            read(preferred("brickwork", n))
            for n in counts
        ],
    }
    colors = {"Linear": BLUE, "Ring": GREEN, "Brickwork": ORANGE}
    markers = {"Linear": "o", "Ring": "s", "Brickwork": "^"}
    fig, axes = plt.subplots(1, 3, figsize=(7.12, 2.35))

    ax = axes[0]
    panel(ax, "a", "End-to-end value + gradient", "both")
    for name, payloads in circuits.items():
        med = np.asarray([p["value_and_grad"]["median_seconds"] for p in payloads])
        ax.errorbar(
            counts,
            med,
            yerr=samples_error(payloads),
            color=colors[name],
            marker=markers[name],
            capsize=2,
            label=name,
            zorder=3,
        )
    ax.axvline(12, color=GREY, linestyle=":", linewidth=1)
    ax.text(12.5, 31, "2 nodes", color=GREY, fontsize=6.5, rotation=90, va="top")
    ax.set(xscale="log", yscale="log", xlabel="A800 GPUs", ylabel="Median time (s)")
    ax.set_xticks(counts, labels=[str(x) for x in counts])
    handles, labels = ax.get_legend_handles_labels()

    ax = axes[1]
    panel(ax, "b", "Strong-scaling speedup", "both")
    for name, payloads in circuits.items():
        med = np.asarray([p["value_and_grad"]["median_seconds"] for p in payloads])
        ax.plot(counts, med[0] / med, color=colors[name], marker=markers[name], label=name)
    ax.plot(counts, counts, color=GREY, linestyle="--", linewidth=1.1, label="Ideal")
    ax.axvspan(8.8, 18, color=LIGHT, alpha=0.65, zorder=-1)
    ax.set(xscale="log", xlabel="A800 GPUs", ylabel="Speedup vs. 1 GPU")
    ax.set_xticks(counts, labels=[str(x) for x in counts])
    # Optimized Brickwork reaches >10x at 16 GPUs; keep the ideal line and
    # marker fully visible instead of clipping the upper half of panel b.
    max_speedup = max(
        float(series[0]["value_and_grad"]["median_seconds"] / p["value_and_grad"]["median_seconds"])
        for series in circuits.values()
        for p in series
    )
    ax.set_ylim(0, max(8.4, max_speedup * 1.12))

    ax = axes[2]
    panel(ax, "c", "16-GPU stability boundary")
    names = list(circuits)
    payloads = [circuits[name][-1] for name in names]
    med = [p["value_and_grad"]["median_seconds"] for p in payloads]
    bars = ax.bar(names, med, color=[colors[x] for x in names], width=0.66, zorder=2)
    for bar, payload in zip(bars, payloads):
        values = payload["value_and_grad"]["samples_seconds"]
        ax.scatter(
            np.full(len(values), bar.get_x() + bar.get_width() / 2),
            values,
            color="black",
            s=9,
            zorder=3,
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            max(values) + 1.1,
            f"CV {100 * payload['value_and_grad']['coefficient_of_variation']:.1f}%",
            ha="center",
            fontsize=6.4,
        )
    ax.set_ylabel("Time (s), all five samples")
    ax.set_ylim(0, 32)
    ax.tick_params(axis="x", rotation=18)
    fig.suptitle(
        "Circuit generality and the cross-node strong-scaling boundary",
        x=0.02,
        ha="left",
        fontsize=10,
        weight="bold",
        y=0.995,
    )
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.38, 0.82),
        ncol=3,
        frameon=False,
    )
    # Compact circuit key so readers can interpret the three topologies.
    for xpos, note in (
        (0.17, "Linear: adjacent chain q0—q1—…—q30"),
        (0.50, "Ring: add wrap-around q30—q0"),
        (0.83, "Brickwork: even/odd links alternate"),
    ):
        fig.text(xpos, 0.925, note, ha="center", va="top", fontsize=6.0,
                 color=GREY)
    fig.text(
        0.02,
        0.01,
        "31 qubits · 8 layers · 248 trainable RY parameters · complex64 · reversible adjoint · warmup 2 + measured 5",
        fontsize=6.7,
        color=GREY,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.78), w_pad=1.0)
    save(fig, output / "fig4_circuit_generality")


def plot_profiler(output: Path) -> None:
    root = output / "profiler"
    names = ["Linear", "Ring", "Brickwork"]
    payloads = [
        read(root / f"flagquantum_31q_d8_{name.lower()}_8gpu_profile.json")
        for name in names
    ]
    colors = [BLUE, GREEN, ORANGE]
    profiles = [p["profiler"]["rank_metrics"] for p in payloads]

    def median(key: str, index: int) -> float:
        return float(np.median([rank[key] for rank in profiles[index]]))

    fig, axes = plt.subplots(1, 3, figsize=(7.12, 2.25))
    x = np.arange(3)

    ax = axes[0]
    panel(ax, "a", "Measured runtime")
    times = [p["value_and_grad"]["median_seconds"] for p in payloads]
    bars = ax.bar(x, times, color=colors, width=0.65, zorder=2)
    ax.bar_label(bars, fmt="%.2f s", fontsize=6.5, padding=2)
    ax.set(xticks=x, xticklabels=names, ylabel="Median time (s)", ylim=(0, 7.5))

    ax = axes[1]
    panel(ax, "b", "Communication work")
    volume = np.asarray([median("logical_communication_bytes", i) for i in x]) / 1e9
    kernels = np.asarray([median("communication_kernel_count", i) for i in x])
    width = 0.36
    ax.bar(x - width / 2, volume, width, color=SKY, label="Logical volume (GB)", zorder=2)
    ax.set(xticks=x, xticklabels=names, ylabel="Logical volume (GB)")
    twin = ax.twinx()
    twin.bar(x + width / 2, kernels, width, color=RED, label="Communication kernels", zorder=2)
    twin.set_ylabel("Communication kernels")
    handles = [ax.patches[0], twin.patches[0]]
    ax.legend(
        handles,
        ["Bytes (GB)", "Comm kernels"],
        frameon=False,
        loc="upper left",
        fontsize=6.1,
    )
    twin.spines["top"].set_visible(False)

    ax = axes[2]
    panel(ax, "c", "Bandwidth is unchanged")
    bandwidth = np.asarray(
        [median("effective_logical_bandwidth_bytes_per_second", i) for i in x]
    ) / 1e9
    activity = 100 * np.asarray([median("gpu_activity_fraction", i) for i in x])
    ax.plot(x, bandwidth, color=BLUE, marker="o", label="Effective bandwidth")
    ax.set(xticks=x, xticklabels=names, ylabel="Effective bandwidth (GB/s)", ylim=(40, 52))
    twin = ax.twinx()
    twin.plot(x, activity, color=GREEN, marker="s", label="GPU activity")
    twin.set(ylabel="GPU activity (%)", ylim=(65, 82))
    twin.spines["top"].set_visible(False)
    ax.text(
        0.03,
        0.05,
        "Brickwork:\n+50% bytes\n+47% comm kernels",
        transform=ax.transAxes,
        fontsize=6.6,
        color=RED,
    )
    fig.suptitle(
        "Profiler root cause: extra communication work, not lower link bandwidth",
        x=0.02,
        ha="left",
        fontsize=10,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.90), w_pad=1.2)
    save(fig, output / "fig5_profiler_root_cause")


def plot_science(repo: Path, output: Path) -> None:
    root = repo / "benchmarks/results/comparison"
    one = read(root / "statevector_training_science_30q_1xa800_v12.json")
    sixteen = read(root / "statevector_training_science_30q_2node16xa800_v12.json")
    capacity = read(root / "statevector_training_science_35q_capacity_report_v12.json")
    oom = read(root / "statevector_training_science_35q_1xa800_oom_v12.json")

    fig, axes = plt.subplots(1, 2, figsize=(7.12, 2.35))
    ax = axes[0]
    panel(ax, "a", "Trainable workload acceleration")
    labels = ["Forward", "Backward", "End-to-end"]
    one_times = [one["forward"]["median_seconds"], one["backward"]["median_seconds"], one["end_to_end"]["median_seconds"]]
    multi_times = [
        sixteen["forward"]["median_seconds"],
        sixteen["backward"]["median_seconds"],
        sixteen["end_to_end"]["median_seconds"],
    ]
    x = np.arange(3)
    width = 0.36
    ax.bar(x - width / 2, one_times, width, color=GREY, label="1 GPU", zorder=2)
    bars = ax.bar(x + width / 2, multi_times, width, color=BLUE, label="16 GPUs / 2 nodes", zorder=2)
    for index, bar in enumerate(bars):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.7,
            f"{one_times[index] / multi_times[index]:.2f}×",
            ha="center",
            fontsize=6.7,
            color=BLUE,
            weight="bold",
        )
    ax.set(xticks=x, xticklabels=labels, ylabel="Median training-step time (s)", ylim=(0, 27))
    ax.legend(frameon=False, loc="upper left")

    ax = axes[1]
    panel(ax, "b", "Capacity expansion: 35 qubits")
    ax.set(xlim=(0, 10), ylim=(0, 10))
    ax.axis("off")
    ax.add_patch(
        plt.Rectangle((0.3, 6.0), 4.2, 2.7, facecolor="#FDEDEC", edgecolor=RED, linewidth=1.2)
    )
    ax.text(2.4, 8.1, "1 × A800 80 GB", ha="center", weight="bold", fontsize=7.5)
    ax.text(2.4, 7.15, "OOM", ha="center", fontsize=14, color=RED, weight="bold")
    ax.text(2.4, 6.35, f"measured in {oom['elapsed_seconds']:.2f} s", ha="center", fontsize=6.2)
    ax.annotate("", xy=(5.5, 7.15), xytext=(4.6, 7.15), arrowprops={"arrowstyle": "->", "lw": 1.3})
    ax.add_patch(
        plt.Rectangle((5.6, 6.0), 4.1, 2.7, facecolor="#E8F6F3", edgecolor=GREEN, linewidth=1.2)
    )
    ax.text(7.65, 8.1, "16 × A800 / 2 nodes", ha="center", weight="bold", fontsize=7.5)
    ax.text(7.65, 7.15, "Completes", ha="center", fontsize=12.5, color=GREEN, weight="bold")
    ax.text(
        7.65,
        6.35,
        f"{capacity['distributed_end_to_end_median_seconds']:.3f} s · "
        f"{capacity['distributed_peak_memory_bytes_per_rank'] / 1e9:.1f} GB/rank",
        ha="center",
        fontsize=6.5,
    )
    ax.text(
        5.0,
        4.05,
        "One logical differentiable statevector\n"
        "35 trainable parameters · Adam update · complex64",
        ha="center",
        fontsize=7.2,
    )
    ax.text(
        5.0,
        1.85,
        "Single GPU cannot allocate the workload;\n"
        "two nodes complete forward, backward, and optimizer step.",
        ha="center",
        fontsize=6.8,
        color=GREY,
    )
    fig.suptitle(
        "End-to-end scientific training evidence: speed and capacity",
        x=0.02,
        ha="left",
        fontsize=10,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.90), w_pad=1.4)
    save(fig, output / "fig6_scientific_training")


def plot_brickwork_optimization(output: Path) -> None:
    root = output / "generality"
    profiler = output / "profiler"
    counts = np.asarray([1, 2, 4, 8, 16])
    before = [
        read(root / f"flagquantum_31q_d8_brickwork_{n}gpu_warm2_rep5.json")
        for n in counts
    ]
    after = [
        before[0],
        *[
            read(
                root
                / f"flagquantum_31q_d8_brickwork_{n}gpu_dependency_schedule_warm2_rep5.json"
            )
            for n in (2, 4)
        ],
        read(
            profiler
            / "flagquantum_31q_d8_brickwork_8gpu_dependency_schedule_profile.json"
        ),
        read(
            root
            / "flagquantum_31q_d8_brickwork_16gpu_dependency_schedule_warm2_rep5.json"
        ),
    ]
    before_time = np.asarray(
        [p["value_and_grad"]["median_seconds"] for p in before]
    )
    after_time = np.asarray([p["value_and_grad"]["median_seconds"] for p in after])

    fig, axes = plt.subplots(1, 3, figsize=(7.12, 2.35))
    ax = axes[0]
    panel(ax, "a", "Strong scaling", "both")
    ax.errorbar(
        counts,
        before_time,
        yerr=samples_error(before),
        color=GREY,
        marker="o",
        linestyle="--",
        capsize=2,
        label="Before",
    )
    ax.errorbar(
        counts,
        after_time,
        yerr=samples_error(after),
        color=ORANGE,
        marker="^",
        capsize=2,
        label="Dependency-DAG schedule",
    )
    ax.set(xscale="log", yscale="log", xlabel="A800 GPUs", ylabel="Median time (s)")
    ax.set_xticks(counts, labels=[str(x) for x in counts])
    ax.legend(frameon=False, loc="lower left", fontsize=6.2)

    ax = axes[1]
    panel(ax, "b", "Communication eliminated")
    old8 = read(profiler / "flagquantum_31q_d8_brickwork_8gpu_profile.json")
    new8 = after[3]
    metrics = ["Layout\nswaps", "Logical\nbytes", "Comm\nkernels"]
    old_profile = old8["profiler"]["rank_metrics"]
    new_profile = new8["profiler"]["rank_metrics"]
    old_values = np.asarray(
        [
            old8["runtime_summary"]["persistent_layout_swap_count"],
            np.median([x["logical_communication_bytes"] for x in old_profile]),
            np.median([x["communication_kernel_count"] for x in old_profile]),
        ],
        dtype=float,
    )
    new_values = np.asarray(
        [
            new8["runtime_summary"]["persistent_layout_swap_count"],
            np.median([x["logical_communication_bytes"] for x in new_profile]),
            np.median([x["communication_kernel_count"] for x in new_profile]),
        ],
        dtype=float,
    )
    normalized = 100 * new_values / old_values
    bars = ax.bar(metrics, normalized, color=ORANGE, width=0.62, zorder=2)
    reductions = 100 - normalized
    for bar, reduction in zip(bars, reductions):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"−{reduction:.1f}%",
            ha="center",
            fontsize=7,
            weight="bold",
            color=GREEN,
        )
    ax.axhline(100, color=GREY, linestyle="--", linewidth=1)
    ax.set(ylabel="Remaining work (% of before)", ylim=(0, 112))

    ax = axes[2]
    panel(ax, "c", "16-GPU stability restored")
    samples_before = before[-1]["value_and_grad"]["samples_seconds"]
    samples_after = after[-1]["value_and_grad"]["samples_seconds"]
    sample_x = np.arange(1, 6)
    ax.plot(sample_x, samples_before, color=GREY, marker="o", linestyle="--", label="Before")
    ax.plot(sample_x, samples_after, color=ORANGE, marker="^", label="After")
    ax.set(xlabel="Measured sample", ylabel="End-to-end time (s)", xticks=sample_x)
    ax.text(
        0.97,
        0.93,
        "25.60 → 2.57 s\n62.3 → 8.0 GB/rank\nCV 31.2% → 0.27%",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.7,
        color=GREEN,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
    )
    ax.legend(frameon=False, loc="center right", fontsize=6.2)

    fig.suptitle(
        "Dependency-DAG scheduling removes Brickwork's distributed communication cliff",
        x=0.02,
        ha="left",
        fontsize=10,
        weight="bold",
    )
    fig.text(
        0.02,
        0.01,
        "31 qubits · 8 layers · 248 trainable parameters · identical circuit DAG and gradients · warmup 2 + measured 5",
        fontsize=6.7,
        color=GREY,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.91), w_pad=1.1)
    save(fig, output / "fig7_brickwork_dependency_schedule")


def plot_ring_optimization(output: Path) -> None:
    root = output / "generality"
    profiler = output / "profiler"
    counts = np.asarray([1, 2, 4, 8])
    before = [
        read(root / f"flagquantum_31q_d8_ring_{n}gpu_warm2_rep5.json")
        for n in counts
    ]
    after = [
        before[0],
        read(root / "flagquantum_31q_d8_ring_2gpu_dag_cxsegment_warm2_rep5.json"),
        read(root / "flagquantum_31q_d8_ring_4gpu_dag_cxsegment_warm2_rep5.json"),
        read(profiler / "flagquantum_31q_d8_ring_8gpu_dag_cxsegment_profile.json"),
    ]
    stages = [
        read(profiler / "flagquantum_31q_d8_ring_8gpu_profile.json"),
        read(profiler / "flagquantum_31q_d8_ring_8gpu_dependency_schedule_profile.json"),
        after[-1],
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.12, 2.35))

    ax = axes[0]
    panel(ax, "a", "Ring strong scaling", "both")
    for payloads, label, color, marker, linestyle in (
        (before, "Before", GREY, "o", "--"),
        (after, "DAG + compiled CX", GREEN, "s", "-"),
    ):
        med = np.asarray([p["value_and_grad"]["median_seconds"] for p in payloads])
        ax.errorbar(
            counts,
            med,
            yerr=samples_error(payloads),
            color=color,
            marker=marker,
            linestyle=linestyle,
            capsize=2,
            label=label,
            zorder=3,
        )
    ax.set(xscale="log", yscale="log", xlabel="A800 GPUs", ylabel="Median time (s)")
    ax.set_xticks(counts, labels=[str(x) for x in counts])
    ax.minorticks_off()
    ax.legend(frameon=False, loc="lower left", fontsize=6.5)

    ax = axes[1]
    panel(ax, "b", "DAG alone: no speedup")
    stage_labels = ["Baseline", "DAG only", "DAG +\ncompiled CX"]
    forward = np.asarray([p["forward"]["median_seconds"] for p in stages])
    backward = np.asarray([p["backward"]["median_seconds"] for p in stages])
    x = np.arange(3)
    ax.bar(x, forward, color=SKY, width=0.68, label="Forward", zorder=2)
    ax.bar(x, backward, bottom=forward, color=BLUE, width=0.68, label="Backward", zorder=2)
    totals = forward + backward
    for xpos, total in zip(x, totals):
        ax.text(xpos, total + 0.12, f"{total:.2f}s", ha="center", fontsize=6.5)
    ax.set(xticks=x, xticklabels=stage_labels, ylabel="8-GPU time (s)", ylim=(0, 6.15))
    ax.text(x[0], forward[0] / 2, "Forward", ha="center", va="center", fontsize=6.1)
    ax.text(
        x[0],
        forward[0] + backward[0] / 2,
        "Backward",
        ha="center",
        va="center",
        fontsize=6.1,
        color="white",
    )

    ax = axes[2]
    panel(ax, "c", "Half the communication")
    rank_metrics = [p["profiler"]["rank_metrics"] for p in (stages[0], stages[-1])]
    volume = np.asarray(
        [np.median([r["logical_communication_bytes"] for r in ranks]) for ranks in rank_metrics]
    )
    kernels = np.asarray(
        [np.median([r["communication_kernel_count"] for r in ranks]) for ranks in rank_metrics]
    )
    work = np.vstack((100 * volume / volume[0], 100 * kernels / kernels[0]))
    labels = ["Logical bytes", "Comm kernels"]
    width = 0.33
    x = np.arange(2)
    ax.bar(x - width / 2, work[:, 0], width, color=GREY, label="Baseline", zorder=2)
    bars = ax.bar(x + width / 2, work[:, 1], width, color=GREEN, label="Optimized", zorder=2)
    for bar, value in zip(bars, work[:, 1]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 3, f"{value:.0f}%", ha="center", fontsize=6.5)
    ax.set(xticks=x, xticklabels=labels, ylabel="Remaining work (% of baseline)", ylim=(0, 112))
    ax.legend(frameon=False, loc="upper right", fontsize=6.2)

    fig.suptitle(
        "Ring needs communication-aware scheduling and compiled execution together",
        x=0.02,
        ha="left",
        fontsize=10,
        weight="bold",
    )
    fig.text(
        0.02,
        0.01,
        "31 qubits · 8 layers · 248 trainable parameters · full value + gradient · warmup 2 + measured 5 · 16-GPU rerun pending",
        fontsize=6.7,
        color=GREY,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.91), w_pad=1.1)
    save(fig, output / "fig8_ring_dag_cxsegment")


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    output = repo / "benchmarks/results/legacy/statevector_mlsys_current"
    style()
    plot_generality(repo, output)
    plot_profiler(output)
    plot_science(repo, output)
    plot_brickwork_optimization(output)
    plot_ring_optimization(output)


if __name__ == "__main__":
    main()
