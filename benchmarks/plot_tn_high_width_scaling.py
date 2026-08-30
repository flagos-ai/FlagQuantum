#!/usr/bin/env python3
"""Plot the memory-forced 36q TN strong-scaling evidence."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path("benchmarks/results/legacy/tn_high_width_scaling_20260730")


def main() -> None:
    records = []
    for world_size in (1, 2, 4, 8):
        path = ROOT / "raw" / f"36q_4x9_c4_k1_{world_size}gpu.json"
        records.append(json.loads(path.read_text(encoding="utf-8")))
    worlds = [record["distributed"]["world_size"] for record in records]
    execution = [
        record["latency_seconds_max_rank_median"] for record in records
    ]
    speedup = [execution[0] / value for value in execution]
    efficiency = [value / world for value, world in zip(speedup, worlds, strict=True)]
    cold = [
        value + sum(record["phase_timings_seconds_max_rank"].values())
        for value, record in zip(execution, records, strict=True)
    ]
    peak_gib = [record["peak_memory_bytes_max"] / 2**30 for record in records]

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
    axes[0, 1].set(title="Strong scaling", xlabel="A800 GPUs", ylabel="speedup")
    axes[0, 1].legend()

    axes[1, 0].bar([str(world) for world in worlds], cold, color="#4c78a8")
    axes[1, 0].set(
        title="Cold end-to-end latency",
        xlabel="A800 GPUs",
        ylabel="seconds",
    )

    axes[1, 1].bar([str(world) for world in worlds], peak_gib, color="#f58518")
    axes[1, 1].axhline(80, color="#999999", linestyle="--", label="device capacity")
    axes[1, 1].set(
        title="Measured rank-local peak",
        xlabel="A800 GPUs",
        ylabel="GiB",
        ylim=(0, 85),
    )
    axes[1, 1].legend()

    fig.suptitle("Memory-forced TN: 36q 4×9 grid, 4 cycles, one amplitude")
    figures = ROOT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(figures / f"tn_high_width_scaling.{suffix}", dpi=180)

    summary = {
        "world_sizes": worlds,
        "execution_seconds": execution,
        "speedup_over_1gpu": speedup,
        "parallel_efficiency": efficiency,
        "cold_end_to_end_seconds": cold,
        "peak_memory_gib": peak_gib,
        "eight_gpu_speedup": speedup[-1],
        "eight_gpu_target_speedup": 5.0,
        "eight_gpu_target_passed": speedup[-1] >= 5.0,
        "statevector_bytes": 8 * 2**36,
        "estimated_mps_bond": 262144,
        "estimated_mps_bytes": 39582418599936,
        "slice_count": records[0]["distributed"]["expected_slice_tasks"],
    }
    (figures / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
