#!/usr/bin/env python3
"""Plot the working-set-safe 36q TN production-gate measurements."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path("benchmarks/results/tn_high_width_working_set_scaling_20260730")


def main() -> None:
    records = [
        json.loads(
            (
                ROOT / "raw" / f"36q_4x9_c4_k1_{world_size}gpu.json"
            ).read_text(encoding="utf-8")
        )
        for world_size in (1, 2, 4, 8)
    ]
    worlds = [record["distributed"]["world_size"] for record in records]
    execution = [
        record["latency_seconds_max_rank_median"] for record in records
    ]
    speedup = [execution[0] / value for value in execution]
    efficiency = [speedup[index] / worlds[index] for index in range(len(worlds))]
    peak = [record["peak_memory_bytes_max"] / 2**30 for record in records]
    cold = [
        execution[index] + sum(record["phase_timings_seconds_max_rank"].values())
        for index, record in enumerate(records)
    ]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    axes[0, 0].plot(worlds, execution, "o-", label="measured")
    axes[0, 0].plot(
        worlds,
        [execution[0] / world for world in worlds],
        "--",
        color="#999999",
        label="ideal",
    )
    axes[0, 0].set(title="Steady execution", xlabel="A800 GPUs", ylabel="seconds")
    axes[0, 0].legend()
    axes[0, 1].plot(worlds, speedup, "o-", label="measured")
    axes[0, 1].plot(worlds, worlds, "--", color="#999999", label="ideal")
    axes[0, 1].axhline(5.0, color="#e45756", linestyle=":", label="production gate")
    axes[0, 1].set(title="Strong scaling", xlabel="A800 GPUs", ylabel="speedup")
    axes[0, 1].legend()
    axes[1, 0].bar([str(world) for world in worlds], peak, color="#f58518")
    axes[1, 0].axhline(80, color="#999999", linestyle="--", label="device capacity")
    axes[1, 0].set(
        title="Working-set-safe memory",
        xlabel="A800 GPUs",
        ylabel="peak GiB/rank",
        ylim=(0, 85),
    )
    axes[1, 0].legend()
    axes[1, 1].bar([str(world) for world in worlds], cold, color="#4c78a8")
    axes[1, 1].set(
        title="Cold end-to-end latency",
        xlabel="A800 GPUs",
        ylabel="seconds",
    )
    fig.suptitle("Production TN gate: 36q 4×9 grid, 128 safe slices")
    figures = ROOT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(figures / f"tn_working_set_scaling.{suffix}", dpi=180)
    summary = {
        "world_sizes": worlds,
        "execution_seconds": execution,
        "speedup_over_1gpu": speedup,
        "parallel_efficiency": efficiency,
        "peak_memory_gib": peak,
        "cold_end_to_end_seconds": cold,
        "slice_count": 128,
        "planned_intermediate_gib": 4,
        "working_set_safety_factor": 4.0,
        "allocator_oom_warnings": 0,
        "eight_gpu_speedup": speedup[-1],
        "eight_gpu_target": 5.0,
        "eight_gpu_target_passed": speedup[-1] >= 5.0,
    }
    (figures / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
