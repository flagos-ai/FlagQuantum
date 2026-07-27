#!/usr/bin/env python3
"""Plot high-load forward/backward/end-to-end statevector weak scaling."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/flagquantum-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if (
        report.get("schema_version")
        != "flagquantum.statevector.training_scaling_report.v1"
    ):
        raise ValueError("invalid training scaling report")
    points = report["points"]
    baseline_wires = int(report["baseline_n_wires"])
    local_amplitudes_power = int(math.log2(int(report["local_amplitudes"])))
    workload = report.get("workload", {})
    workload_name = workload.get("name", "statevector workload")
    distributed_points = [point for point in points if point["world_size"] >= 2]
    if not distributed_points or distributed_points[0]["world_size"] != 2:
        raise ValueError("a 2-GPU distributed baseline is required")
    worlds = [p["world_size"] for p in points]
    x = [int(math.log2(world)) for world in worlds]
    labels = [
        rf"$2^{{{power}}}$ GPU" + f"\n{point['n_wires']}q"
        for power, point in zip(x, points)
    ]
    colors = {
        "forward": "#1769aa",
        "differentiable_forward": "#6f58a8",
        "backward": "#e07a1f",
        "end_to_end": "#16856b",
    }
    names = {
        "forward": "Forward (no grad)",
        "differentiable_forward": "Forward",
        "backward": "Backward",
        "end_to_end": "Adam end-to-end",
    }
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.facecolor": "#fbfcfe",
            "axes.edgecolor": "#405166",
            "grid.color": "#c9d2dc",
            "grid.alpha": 0.55,
            "lines.linewidth": 2.2,
            "lines.markersize": 7,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 8.7))

    ax = axes[0, 0]
    for phase in ("differentiable_forward", "backward", "end_to_end"):
        values = [p["phases"][phase]["median_seconds"] for p in points]
        ax.plot(x, values, "o-", color=colors[phase], label=names[phase])
    ax.set_title("a  High-load weak scaling · measured time", loc="left")
    ax.set_ylabel("Median time (s), 3 runs")
    measured_max = max(
        p["phases"][phase]["median_seconds"]
        for p in points
        for phase in ("differentiable_forward", "backward", "end_to_end")
    )
    ax.set_ylim(0, measured_max * 1.18)
    ax.legend(frameon=False, loc="upper left", ncols=2)

    ax = axes[0, 1]
    efficiency_values = []
    for phase in ("differentiable_forward", "backward", "end_to_end"):
        values = [
            p["phases"][phase]["weak_scaling_efficiency"]["speedup"] for p in points
        ]
        efficiency_values.extend(values)
        ax.plot(x, values, "o-", color=colors[phase], label=names[phase])
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    efficiency_floor = max(0.0, math.floor((min(efficiency_values) - 0.04) * 10) / 10)
    ax.set_ylim(efficiency_floor, 1.06)
    ax.set_title("b  Weak-scaling efficiency", loc="left")
    ax.set_ylabel(f"Efficiency vs {baseline_wires}q / 1 GPU")
    ax.legend(
        frameon=True,
        facecolor="white",
        framealpha=0.88,
        edgecolor="none",
        loc="upper right",
        fontsize=9,
        labelspacing=0.35,
        handlelength=2.0,
    )

    ax = axes[1, 0]
    e2e_mem = [p["end_to_end_peak_memory_bytes"] / 2**30 for p in points]
    ax.plot(x, e2e_mem, "s-", color=colors["end_to_end"], label="End-to-end")
    memory_ceiling = max(44.0, math.ceil(max(e2e_mem) / 10.0) * 10.0)
    ax.axhspan(0, memory_ceiling, color="#16856b", alpha=0.035)
    ax.set_ylim(0, memory_ceiling)
    ax.set_title("c  Constant-memory capacity envelope", loc="left")
    ax.set_ylabel("Peak allocated (GiB/rank)")
    ax.annotate(
        f"{points[-1]['n_wires']}q · {e2e_mem[-1]:.2f} GiB/rank",
        (x[-1], e2e_mem[-1]),
        xytext=(-12, -18),
        textcoords="offset points",
        ha="right",
        va="top",
        color=colors["end_to_end"],
        weight="bold",
    )

    ax = axes[1, 1]
    differentiable = [
        p["phases"]["differentiable_forward"]["median_seconds"] for p in points
    ]
    backward = [p["phases"]["backward"]["median_seconds"] for p in points]
    optimizer = [
        max(
            0.0,
            p["phases"]["end_to_end"]["median_seconds"]
            - p["phases"]["differentiable_forward"]["median_seconds"]
            - p["phases"]["backward"]["median_seconds"],
        )
        for p in points
    ]
    totals = [a + b + c for a, b, c in zip(differentiable, backward, optimizer)]
    differentiable_share = [
        value / total for value, total in zip(differentiable, totals)
    ]
    backward_share = [value / total for value, total in zip(backward, totals)]
    optimizer_share = [value / total for value, total in zip(optimizer, totals)]
    ax.bar(
        x,
        differentiable_share,
        width=0.68,
        color=colors["differentiable_forward"],
        label="Forward",
    )
    ax.bar(
        x,
        backward_share,
        width=0.68,
        bottom=differentiable_share,
        color=colors["backward"],
        label="Backward",
    )
    bottoms = [a + b for a, b in zip(differentiable_share, backward_share)]
    ax.bar(
        x,
        optimizer_share,
        width=0.68,
        bottom=bottoms,
        color="#9aa7b4",
        label="Adam + broadcast",
    )
    ax.set_title("d  Normalized end-to-end composition", loc="left", pad=30)
    ax.set_ylabel("Share of end-to-end time")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0, 1.0)
    ax.legend(
        frameon=False,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.005),
        ncols=3,
        borderaxespad=0,
        columnspacing=1.3,
        handletextpad=0.55,
    )

    for ax in axes.flat:
        ax.set_xticks(x, labels)
        ax.set_xlim(-0.35, x[-1] + 0.35)
    for ax in axes.flat:
        ax.set_xlabel("A800 GPUs / exact-state qubits")
        ax.grid(True, axis="y")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "FlagQuantum differentiable exact-statevector scaling · "
        f"{baseline_wires}q to {points[-1]['n_wires']}q",
        fontsize=16,
        weight="bold",
        x=0.06,
        ha="left",
    )
    fig.text(
        0.06,
        0.014,
        f"complex64 · {workload_name.replace('_', ' ')} · "
        f"constant 2^{local_amplitudes_power} amplitudes/rank · "
        "all wires active · canonical layout · "
        "owner-sharded Adam · 1/2/4/8 GPU on one node; 16 GPU across two nodes",
        fontsize=9,
        color="#4c5c6d",
    )
    fig.tight_layout(rect=(0.035, 0.055, 0.99, 0.935), h_pad=2.5, w_pad=2.4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
