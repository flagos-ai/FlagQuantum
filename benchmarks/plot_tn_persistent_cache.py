#!/usr/bin/env python3
"""Plot cold-miss versus cold-hit TN persistent-plan evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path("benchmarks/results/tn_persistent_cache_20260730")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    arguments = parser.parse_args()
    root = arguments.root
    records = {
        name: json.loads((root / "raw" / f"{name}_8gpu.json").read_text())
        for name in ("cold_miss", "cold_hit")
    }
    labels = ["cold miss", "cold hit"]
    planning = [
        records[name]["phase_timings_seconds_max_rank"][
            "projection_and_slice_selection"
        ]
        for name in records
    ]
    build = [
        records[name]["phase_timings_seconds_max_rank"]["tensor_network_build"]
        for name in records
    ]
    execution = [
        records[name]["latency_seconds_max_rank_median"] for name in records
    ]
    total = [
        execution[index]
        + sum(records[name]["phase_timings_seconds_max_rank"].values())
        for index, name in enumerate(records)
    ]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].bar(labels, planning, color=("#e45756", "#54a24b"))
    axes[0].set(title="Path/slice planning phase", ylabel="seconds")
    axes[1].bar(labels, planning, label="planning")
    axes[1].bar(labels, build, bottom=planning, label="TN build")
    bottoms = [planning[index] + build[index] for index in range(2)]
    axes[1].bar(labels, execution, bottom=bottoms, label="execution")
    axes[1].set(title="Cold end-to-end latency", ylabel="seconds")
    axes[1].legend()
    fig.suptitle("Persistent TN plan cache: 36q 4×9 grid on 8×A800")

    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(figures / f"tn_persistent_cache.{suffix}", dpi=180)
    summary = {
        "planning_seconds": dict(zip(labels, planning, strict=True)),
        "execution_seconds": dict(zip(labels, execution, strict=True)),
        "cold_end_to_end_seconds": dict(zip(labels, total, strict=True)),
        "planning_speedup": planning[0] / planning[1],
        "cold_end_to_end_speedup": total[0] / total[1],
        "cache_bytes": (root / "cache" / "36q_4x9_c4_k1.json").stat().st_size,
    }
    (figures / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
