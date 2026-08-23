#!/usr/bin/env python3
"""Aggregate and plot README-grade distributed-TN scaling evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strong-dir", type=Path, required=True)
    parser.add_argument("--weak-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_run(payload: dict[str, Any], *, slices: int = 128) -> None:
    if int(payload["slice_count"]) != slices:
        raise ValueError("scaling input does not use the frozen 128-slice plan")
    if payload["gradient_aggregation_semantics"] != "reduce_to_parameter_owner":
        raise ValueError("scaling input does not use owner-sharded gradients")
    if not payload["output_finite"] or not payload["gradients_finite"]:
        raise ValueError("scaling input contains nonfinite output or gradients")
    if not payload["parameter_consistency_passed"]:
        raise ValueError("scaling input failed parameter consistency")


def aggregate(
    *, strong_dir: Path, weak_root: Path
) -> tuple[dict[str, Any], tuple[Path, ...]]:
    worlds = (1, 2, 4, 8, 16)
    strong_paths = tuple(strong_dir / f"{world}gpu.json" for world in worlds)
    strong_payloads = tuple(_load(path) for path in strong_paths)
    for world, payload in zip(worlds, strong_payloads):
        _validate_run(payload)
        if int(payload["world_size"]) != world:
            raise ValueError("strong-scaling world size does not match filename")
    strong_seconds = tuple(
        float(payload["max_execution_seconds"]) for payload in strong_payloads
    )
    strong_speedup = tuple(strong_seconds[0] / value for value in strong_seconds)
    strong_efficiency = tuple(
        speedup / world for speedup, world in zip(strong_speedup, worlds)
    )

    weak_directories = {
        2: weak_root / "2gpu",
        4: weak_root / "4gpu",
        8: weak_root / "8gpu-split",
        16: weak_root / "16gpu",
    }
    weak_paths: list[Path] = [strong_paths[0]]
    weak_points = [
        {
            "gpu_count": 1,
            "node_count": 1,
            "replica_count": 1,
            "replica_seconds": (strong_seconds[0],),
            "maximum_replica_seconds": strong_seconds[0],
            "mean_replica_seconds": strong_seconds[0],
        }
    ]
    for world in worlds[1:]:
        paths = tuple(sorted(weak_directories[world].glob("replica*.json")))
        if len(paths) != world:
            raise ValueError(
                f"weak-scaling {world}-GPU point requires {world} replicas"
            )
        payloads = tuple(_load(path) for path in paths)
        for payload in payloads:
            _validate_run(payload)
            if int(payload["world_size"]) != 1:
                raise ValueError("weak-scaling replicas must be independent ranks")
        seconds = tuple(float(item["max_execution_seconds"]) for item in payloads)
        hosts = {
            str(item["rank_results"][0]["hostname"]) for item in payloads
        }
        weak_points.append(
            {
                "gpu_count": world,
                "node_count": len(hosts),
                "replica_count": len(payloads),
                "replica_seconds": seconds,
                "maximum_replica_seconds": max(seconds),
                "mean_replica_seconds": float(np.mean(seconds)),
            }
        )
        weak_paths.extend(paths)
    baseline = float(weak_points[0]["maximum_replica_seconds"])
    for point in weak_points:
        world = int(point["gpu_count"])
        maximum = float(point["maximum_replica_seconds"])
        point["weak_scaling_efficiency"] = baseline / maximum
        point["aggregate_throughput_workloads_per_second"] = world / maximum
        point["throughput_speedup"] = world * baseline / maximum

    summary = {
        "schema": "flagquantum.tn_readme_scaling.v1",
        "benchmark": "distributed_tn_training_scaling",
        "workload": {
            "qubits": 36,
            "grid": "4x9",
            "entangling_cycles": 4,
            "dtype": "complex128",
            "slice_count": 128,
            "checkpoint_budget_gib_per_rank": 16,
            "statevector_bytes": 2**36 * 16,
            "statevector_device_infeasible": True,
            "gradient_aggregation_semantics": "reduce_to_parameter_owner",
        },
        "strong_scaling": {
            "semantics": "one_sharded_workload_fixed_problem",
            "gpu_counts": worlds,
            "seconds": strong_seconds,
            "speedup": strong_speedup,
            "parallel_efficiency": strong_efficiency,
            "sixteen_gpu_speedup": strong_speedup[-1],
            "sixteen_gpu_efficiency": strong_efficiency[-1],
        },
        "weak_scaling": {
            "semantics": "data_parallel_independent_tn_training_workloads",
            "constant_work_per_gpu": "one_complete_36q_TN_training_step",
            "points": tuple(weak_points),
            "sixteen_gpu_efficiency": weak_points[-1][
                "weak_scaling_efficiency"
            ],
            "sixteen_gpu_throughput_speedup": weak_points[-1][
                "throughput_speedup"
            ],
        },
        "source_artifacts": tuple(
            {"path": str(path), "sha256": _sha256(path)}
            for path in (*strong_paths, *weak_paths[1:])
        ),
        "scalability_claim_boundary": (
            "Strong scaling is one sharded workload. Weak scaling is aggregate "
            "throughput of independent complete 36q TN training workloads."
        ),
    }
    return summary, (*strong_paths, *weak_paths[1:])


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "axes.edgecolor": "#A8B1C1",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": "#D9DEE8",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.65,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "#FAFBFD",
            "savefig.facecolor": "white",
        }
    )


def plot(summary: dict[str, Any], output_dir: Path) -> None:
    _style()
    navy = "#132A4F"
    blue = "#3167E3"
    teal = "#00A6A6"
    orange = "#F28E2B"
    gray = "#8792A5"
    worlds = np.asarray(summary["strong_scaling"]["gpu_counts"], dtype=float)
    seconds = np.asarray(summary["strong_scaling"]["seconds"], dtype=float)
    speedup = np.asarray(summary["strong_scaling"]["speedup"], dtype=float)
    efficiency = np.asarray(
        summary["strong_scaling"]["parallel_efficiency"], dtype=float
    )
    weak = summary["weak_scaling"]["points"]
    weak_time = np.asarray(
        [point["maximum_replica_seconds"] for point in weak], dtype=float
    )
    weak_eff = np.asarray(
        [point["weak_scaling_efficiency"] for point in weak], dtype=float
    )
    throughput_speedup = np.asarray(
        [point["throughput_speedup"] for point in weak], dtype=float
    )

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.1), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(worlds, seconds, "o-", color=blue, lw=2.6, ms=7, label="Measured")
    ax.plot(
        worlds,
        seconds[0] / worlds,
        "--",
        color=gray,
        lw=1.7,
        label="Ideal 1/N",
    )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(worlds, [str(int(value)) for value in worlds])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.set_title("a  Fixed-problem strong scaling", loc="left", color=navy)
    ax.set_xlabel("A800 GPUs")
    ax.set_ylabel("Training-step time (s) ↓")
    ax.legend(loc="upper right")
    ax.annotate(
        f"{seconds[-1]:.2f} s",
        (worlds[-1], seconds[-1]),
        xytext=(-48, 14),
        textcoords="offset points",
        color=blue,
        weight="bold",
    )

    ax = axes[0, 1]
    ax.plot(worlds, speedup, "o-", color=teal, lw=2.6, ms=7, label="Measured")
    ax.plot(worlds, worlds, "--", color=gray, lw=1.7, label="Ideal")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(worlds, [str(int(value)) for value in worlds])
    ax.set_yticks(worlds, [str(int(value)) for value in worlds])
    ax.set_title("b  Speedup and parallel efficiency", loc="left", color=navy)
    ax.set_xlabel("A800 GPUs")
    ax.set_ylabel("Speedup × ↑")
    ax.legend(loc="upper left")
    ax.text(
        0.97,
        0.08,
        f"16 GPUs\n{speedup[-1]:.2f}× speedup\n{efficiency[-1] * 100:.1f}% efficiency",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=navy,
        weight="bold",
        bbox={"boxstyle": "round,pad=0.5", "fc": "white", "ec": "#CCD3DF"},
    )

    ax = axes[1, 0]
    ax.fill_between(worlds, weak_time.min(), weak_time, color=orange, alpha=0.13)
    ax.plot(worlds, weak_time, "o-", color=orange, lw=2.6, ms=7)
    ax.axhline(weak_time[0], color=gray, ls="--", lw=1.7, label="Ideal constant time")
    ax.set_xscale("log", base=2)
    ax.set_xticks(worlds, [str(int(value)) for value in worlds])
    ax.set_ylim(weak_time.min() - 3.0, weak_time.max() + 3.0)
    ax.set_title("c  Throughput weak scaling", loc="left", color=navy)
    ax.set_xlabel("A800 GPUs / independent 36q workloads")
    ax.set_ylabel("Slowest workload time (s) ↓")
    ax.legend(loc="upper right")
    ax.text(
        0.03,
        0.08,
        "Constant work/GPU\n1 complete 36q TN step",
        transform=ax.transAxes,
        color=navy,
        weight="bold",
    )

    ax = axes[1, 1]
    ax.plot(
        worlds,
        throughput_speedup,
        "o-",
        color=blue,
        lw=2.6,
        ms=7,
        label="Aggregate throughput",
    )
    ax.plot(worlds, worlds, "--", color=gray, lw=1.7, label="Ideal")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(worlds, [str(int(value)) for value in worlds])
    ax.set_yticks(worlds, [str(int(value)) for value in worlds])
    ax.set_title("d  Weak-scaling throughput", loc="left", color=navy)
    ax.set_xlabel("A800 GPUs")
    ax.set_ylabel("Throughput speedup × ↑")
    ax.legend(loc="upper left")
    ax.text(
        0.97,
        0.08,
        f"16 GPUs\n{throughput_speedup[-1]:.2f}× throughput\n{weak_eff[-1] * 100:.1f}% efficiency",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=navy,
        weight="bold",
        bbox={"boxstyle": "round,pad=0.5", "fc": "white", "ec": "#CCD3DF"},
    )

    fig.suptitle(
        "FlagQuantum · distributed tensor-network training at 36 qubits",
        fontsize=17,
        weight="bold",
        color=navy,
    )
    fig.text(
        0.5,
        -0.012,
        "4×9 grid · 4 entangling cycles · complex128 · 128 slices · checkpointed reverse · owner-sharded gradients",
        ha="center",
        color="#536078",
        fontsize=10,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "flagquantum_tn_scaling_36q_a800"
    fig.savefig(stem.with_suffix(".png"), dpi=260, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    arguments = _arguments()
    summary, _ = aggregate(
        strong_dir=arguments.strong_dir,
        weak_root=arguments.weak_root,
    )
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    (arguments.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    plot(summary, arguments.output_dir)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
