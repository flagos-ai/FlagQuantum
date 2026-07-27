#!/usr/bin/env python3
"""Aggregate and plot fixed-30q differentiable statevector strong scaling."""

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

PHASES = ("forward", "differentiable_forward", "backward", "end_to_end")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--json-output", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    artifacts = [
        json.loads(path.read_text(encoding="utf-8")) for path in args.artifacts
    ]
    by_world = {int(item["world_size"]): item for item in artifacts}
    worlds = sorted(by_world)
    if worlds != [1, 2, 4, 8, 16]:
        raise ValueError("complete 1/2/4/8/16 GPU measurements required")
    wires = {int(item["workload"]["n_wires"]) for item in artifacts}
    if len(wires) != 1 or any(not item["correctness"]["passed"] for item in artifacts):
        raise ValueError("fixed workload and passing correctness required")

    baseline = by_world[1]
    points = []
    for world in worlds:
        item = by_world[world]
        phases = {}
        for phase in PHASES:
            median = float(item[phase]["median_seconds"])
            speedup = float(baseline[phase]["median_seconds"]) / median
            phases[phase] = {
                "median_seconds": median,
                "speedup_vs_1_gpu": speedup,
                "parallel_efficiency": speedup / world,
                "coefficient_of_variation": item[phase]["coefficient_of_variation"],
            }
        points.append(
            {
                "world_size": world,
                "node_count": item["node_count"],
                "phases": phases,
                "peak_memory_gib_per_rank": item["end_to_end"]["peak_memory_bytes_max"]
                / 2**30,
                "gradient_absolute_error_max": item["correctness"][
                    "gradient_absolute_error_max"
                ],
            }
        )
    report = {
        "schema_version": "flagquantum.statevector.training_strong_scaling.v1",
        "benchmark": "differentiable_statevector_strong_scaling",
        "artifact_class": "derived_development_report",
        "fixed_n_wires": next(iter(wires)),
        "baseline_world_size": 1,
        "points": points,
        "validation_blockers": [
            "development_artifacts_not_release_scalability_evidence",
            "communication_dominated_three_gate_diagnostic_workload",
        ],
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    x = [int(math.log2(world)) for world in worlds]
    labels = [rf"$2^{{{power}}}$ GPU" for power in x]
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
        ax.plot(
            x,
            [point["phases"][phase]["median_seconds"] for point in points],
            "o-",
            color=colors[phase],
            label=names[phase],
        )
    ax.set_title("a  Fixed-30q measured time", loc="left")
    ax.set_ylabel("Median time (s), 3 runs")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, ncols=2, loc="upper right")

    ax = axes[0, 1]
    for phase in ("differentiable_forward", "backward", "end_to_end"):
        ax.plot(
            x,
            [point["phases"][phase]["speedup_vs_1_gpu"] for point in points],
            "o-",
            color=colors[phase],
            label=names[phase],
        )
    ax.plot(x, worlds, "--", color="#66788a", linewidth=1.5, label="Ideal")
    ax.set_title("b  Strong-scaling speedup", loc="left")
    ax.set_ylabel("Speedup vs 1 GPU")
    ax.set_ylim(0, 17)
    ax.legend(frameon=False, loc="upper right")

    ax = axes[1, 0]
    for phase in ("differentiable_forward", "backward", "end_to_end"):
        ax.plot(
            x,
            [point["phases"][phase]["parallel_efficiency"] for point in points],
            "o-",
            color=colors[phase],
            label=names[phase],
        )
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0, 1.05)
    ax.set_title("c  Parallel efficiency", loc="left")
    ax.set_ylabel("Speedup / GPU count")
    ax.legend(frameon=False, loc="upper right")

    ax = axes[1, 1]
    memory = [point["peak_memory_gib_per_rank"] for point in points]
    ideal_memory = [memory[0] / world for world in worlds]
    ax.plot(x, memory, "s-", color=colors["end_to_end"], label="Measured")
    ax.plot(x, ideal_memory, "--", color="#66788a", label="Ideal 1/N")
    for position, value in zip(x, memory):
        ax.annotate(
            f"{value:.1f}",
            (position, value),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            color=colors["end_to_end"],
            fontsize=9,
        )
    ax.set_ylim(0, 36)
    ax.set_title("d  Per-rank memory strong scaling", loc="left")
    ax.set_ylabel("Peak allocated (GiB/rank)")
    ax.legend(frameon=False, loc="upper right")

    for ax in axes.flat:
        ax.set_xticks(x, labels)
        ax.set_xlim(-0.35, x[-1] + 0.35)
        ax.set_xlabel("A800 GPUs · fixed 30q state")
        ax.grid(True, axis="y")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "FlagQuantum differentiable exact-statevector strong scaling · fixed 30q",
        fontsize=16,
        weight="bold",
        x=0.06,
        ha="left",
    )
    fig.text(
        0.06,
        0.014,
        "complex64 · fixed 2³⁰ amplitudes globally · analytic gradient validation · "
        "1/2/4/8 GPU on one node; 16 GPU across two nodes",
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
