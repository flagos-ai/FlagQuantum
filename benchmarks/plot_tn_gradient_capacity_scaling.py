#!/usr/bin/env python3
"""Plot 36q checkpointed TN reverse strong scaling."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path("benchmarks/results/legacy/tn_gradient_capacity_scaling_20260730")


def main() -> None:
    records = [
        json.loads(
            (ROOT / "raw" / f"36q_4x9_c4_{world}gpu.json").read_text()
        )
        for world in (1, 2, 4, 8)
    ]
    worlds = [record["world_size"] for record in records]
    seconds = [record["max_execution_seconds"] for record in records]
    speedup = [seconds[0] / value for value in seconds]
    efficiency = [speedup[i] / worlds[i] for i in range(4)]
    peak = [record["max_cuda_peak_allocated_bytes"] / 2**30 for record in records]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(worlds, seconds, "o-", label="measured")
    axes[0].plot(worlds, [seconds[0] / w for w in worlds], "--", label="ideal")
    axes[0].set(xlabel="A800 GPUs", ylabel="seconds", title="Checkpointed reverse")
    axes[0].legend()
    axes[1].plot(worlds, speedup, "o-", label="measured")
    axes[1].plot(worlds, worlds, "--", label="ideal")
    axes[1].set(xlabel="A800 GPUs", ylabel="speedup", title="Strong scaling")
    axes[1].legend()
    fig.suptitle("36q 4×9 complex128 TN gradient, 128 slices")
    figures = ROOT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(figures / f"tn_gradient_capacity_scaling.{suffix}", dpi=180)
    (figures / "summary.json").write_text(
        json.dumps(
            {
                "world_sizes": worlds,
                "execution_seconds": seconds,
                "speedup": speedup,
                "parallel_efficiency": efficiency,
                "peak_memory_gib": peak,
                "eight_gpu_speedup": speedup[-1],
                "eight_gpu_efficiency": efficiency[-1],
                "gradients_finite": all(r["gradients_finite"] for r in records),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
