#!/usr/bin/env python3
"""Plot the measured effect of TN slicing and shared-DAG execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _load(directory: Path) -> dict[int, dict]:
    records = {}
    for path in directory.glob("40q_d4_k4_*gpu.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        records[int(payload["distributed"]["world_size"])] = payload
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("benchmarks/results/tn_cost_aware_scaling_20260730/raw"),
    )
    parser.add_argument(
        "--optimized",
        type=Path,
        default=Path("benchmarks/results/tn_shared_dag_scaling_20260730/raw"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/results/tn_shared_dag_scaling_20260730/figures"),
    )
    args = parser.parse_args()

    baseline = _load(args.baseline)
    optimized = _load(args.optimized)
    worlds = sorted(set(baseline) & set(optimized))
    if not worlds:
        raise SystemExit("no matching measurements")

    old = [baseline[n]["latency_seconds_max_rank_median"] for n in worlds]
    new = [optimized[n]["latency_seconds_max_rank_median"] for n in worlds]
    cold = [
        value
        + sum(optimized[n]["phase_timings_seconds_max_rank"].values())
        for n, value in zip(worlds, new, strict=True)
    ]
    costs = [optimized[n]["distributed"]["estimated_per_slice_cost"] for n in worlds]
    speedup = [new[0] / value for value in new]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(worlds, old, "o--", label="before shared DAG")
    ax.plot(worlds, new, "o-", label="shared DAG")
    ax.set_yscale("log")
    ax.set_title("Steady sparse-output execution")
    ax.set_xlabel("A800 GPUs")
    ax.set_ylabel("seconds (log scale)")
    ax.legend()

    ax = axes[0, 1]
    ax.bar([str(n) for n in worlds], cold, color="#4c78a8")
    ax.set_title("Cold end-to-end latency")
    ax.set_xlabel("A800 GPUs")
    ax.set_ylabel("seconds")
    ax.text(
        0.02,
        0.96,
        "planning dominates",
        transform=ax.transAxes,
        va="top",
        color="#555555",
    )

    ax = axes[1, 0]
    ax.plot(worlds, costs, "o-", color="#f58518")
    ax.set_title("Estimated rank-local contraction work")
    ax.set_xlabel("A800 GPUs")
    ax.set_ylabel("cost units per slice")

    ax = axes[1, 1]
    ax.plot(worlds, speedup, "o-", label="measured")
    ax.plot(worlds, worlds, "--", color="#999999", label="ideal")
    ax.axhline(1.0, color="#333333", linewidth=0.8)
    ax.set_title("Strong-scaling efficiency check")
    ax.set_xlabel("A800 GPUs")
    ax.set_ylabel("speedup vs 1 GPU")
    ax.legend()

    fig.suptitle("FlagQuantum TN: 40q, depth 4, four amplitudes", fontsize=14)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(args.output_dir / f"tn_shared_dag_scaling.{suffix}", dpi=180)

    summary = {
        "world_sizes": worlds,
        "before_shared_dag_seconds": old,
        "shared_dag_seconds": new,
        "cold_end_to_end_seconds": cold,
        "estimated_per_slice_cost": costs,
        "speedup_vs_1gpu": speedup,
        "best_steady_world_size": worlds[new.index(min(new))],
        "executor_improvement_1gpu": old[0] / new[0],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
