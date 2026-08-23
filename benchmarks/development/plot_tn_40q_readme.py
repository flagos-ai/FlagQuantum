#!/usr/bin/env python3
"""Build a README-grade figure for the verified 40q TN scaling run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

WORLDS = (1, 2, 4, 8, 16)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def _validated_series(input_dir: Path) -> dict[str, Any]:
    paths = tuple(
        input_dir / f"fq_40q_4c_shared64_e2e128_{world}gpu.json"
        for world in WORLDS
    )
    payloads = tuple(_load(path) for path in paths)
    baseline_value = float(payloads[0]["step_records"][0]["value_real"])
    baseline_gradient = float(payloads[0]["step_records"][0]["gradient_l2"])
    for world, payload in zip(WORLDS, payloads):
        if int(payload["world_size"]) != world:
            raise ValueError("world size does not match the input filename")
        if int(payload["slice_count"]) != 64:
            raise ValueError("all runs must use the frozen 64-slice path")
        if int(payload["total_estimated_flops"]) != 22519206255104:
            raise ValueError("all runs must use the same contraction cost")
        if not payload["output_finite"] or not payload["gradients_finite"]:
            raise ValueError("non-finite training output")
        if not payload["parameter_consistency_passed"]:
            raise ValueError("parameters diverged across ranks")
        step = payload["step_records"][0]
        if abs(float(step["value_real"]) - baseline_value) > 1e-14:
            raise ValueError("value changed across GPU counts")
        if abs(float(step["gradient_l2"]) - baseline_gradient) > 1e-14:
            raise ValueError("gradient changed across GPU counts")
        expected_tasks = 64 // world
        if any(
            int(rank["local_task_count"]) != expected_tasks
            for rank in payload["rank_results"]
        ):
            raise ValueError("slice ownership is not balanced")
    seconds = np.asarray(
        [float(payload["max_execution_seconds"]) for payload in payloads]
    )
    worlds = np.asarray(WORLDS, dtype=float)
    speedup = seconds[0] / seconds
    efficiency = speedup / worlds
    return {
        "paths": paths,
        "payloads": payloads,
        "seconds": seconds,
        "speedup": speedup,
        "efficiency": efficiency,
        "value": baseline_value,
        "gradient_l2": baseline_gradient,
    }


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "axes.edgecolor": "#AAB4C5",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": "#DCE2EC",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.72,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "#FAFBFD",
            "savefig.facecolor": "white",
        }
    )


def _card(ax, x: float, y: float, title: str, value: str, detail: str, color: str):
    ax.text(
        x,
        y,
        f"{value}\n{title}\n{detail}",
        transform=ax.transAxes,
        ha="center",
        va="center",
        linespacing=1.35,
        color="#172B4D",
        bbox={
            "boxstyle": "round,pad=0.72,rounding_size=0.15",
            "facecolor": "white",
            "edgecolor": color,
            "linewidth": 1.6,
        },
    )


def plot(series: dict[str, Any], output_dir: Path) -> None:
    _style()
    navy = "#142B4F"
    blue = "#2F6FED"
    cyan = "#00A6A6"
    orange = "#F28E2B"
    green = "#2CA56C"
    gray = "#8995A8"
    worlds = np.asarray(WORLDS, dtype=float)
    seconds = series["seconds"]
    speedup = series["speedup"]
    efficiency = series["efficiency"]
    slices = 64 / worlds

    fig = plt.figure(figsize=(13.4, 7.8), constrained_layout=True)
    grid = fig.add_gridspec(2, 6, height_ratios=(1.08, 0.92))
    ax_time = fig.add_subplot(grid[0, :3])
    ax_speed = fig.add_subplot(grid[0, 3:])
    ax_slice = fig.add_subplot(grid[1, :3])
    ax_proof = fig.add_subplot(grid[1, 3:])

    ax_time.plot(worlds, seconds, "o-", color=blue, lw=2.8, ms=7.5, label="Measured")
    ax_time.plot(worlds, seconds[0] / worlds, "--", color=gray, lw=1.7, label="Ideal 1/N")
    ax_time.fill_between(worlds, seconds[0] / worlds, seconds, color=blue, alpha=0.07)
    ax_time.set_xscale("log", base=2)
    ax_time.set_yscale("log", base=2)
    ax_time.set_xticks(worlds, [str(int(value)) for value in worlds])
    ax_time.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax_time.set_title("a  End-to-end strong scaling", loc="left", color=navy)
    ax_time.set_xlabel("NVIDIA A800 GPUs")
    ax_time.set_ylabel("Forward + backward + SGD (s) ↓")
    ax_time.legend(loc="upper right")
    for x, y in zip(worlds, seconds):
        ax_time.annotate(
            f"{y:.2f}s",
            (x, y),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            color=blue,
            fontsize=9,
            weight="bold",
        )

    ax_speed.plot(worlds, worlds, "--", color=gray, lw=1.7, label="Ideal")
    ax_speed.plot(worlds, speedup, "o-", color=cyan, lw=2.8, ms=7.5, label="FlagQuantum")
    ax_speed.set_xscale("log", base=2)
    ax_speed.set_yscale("log", base=2)
    ax_speed.set_xticks(worlds, [str(int(value)) for value in worlds])
    ax_speed.set_yticks(worlds, [str(int(value)) for value in worlds])
    ax_speed.set_title("b  Near-linear speedup through two nodes", loc="left", color=navy)
    ax_speed.set_xlabel("NVIDIA A800 GPUs")
    ax_speed.set_ylabel("Speedup × ↑")
    ax_speed.legend(loc="upper left")
    ax_speed.text(
        0.97,
        0.07,
        f"16 GPUs  ·  2 nodes\n{speedup[-1]:.2f}× speedup\n{efficiency[-1] * 100:.1f}% efficiency",
        transform=ax_speed.transAxes,
        ha="right",
        va="bottom",
        color=navy,
        weight="bold",
        bbox={"boxstyle": "round,pad=0.5", "fc": "white", "ec": "#C9D2E2"},
    )

    bars = ax_slice.bar(worlds, slices, width=worlds * 0.42, color=orange, alpha=0.88)
    ax_slice.set_xscale("log", base=2)
    ax_slice.set_xticks(worlds, [str(int(value)) for value in worlds])
    ax_slice.set_title("c  Frozen path, balanced slice ownership", loc="left", color=navy)
    ax_slice.set_xlabel("NVIDIA A800 GPUs")
    ax_slice.set_ylabel("Slices per rank")
    ax_slice.set_ylim(0, 82)
    ax_slice.axvline(11.3, color="#C5CBD6", lw=1.2)
    ax_slice.text(
        11.8,
        42,
        "node boundary",
        color=gray,
        fontsize=9,
        rotation=90,
        ha="left",
        va="center",
    )
    for bar, value in zip(bars, slices):
        ax_slice.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            str(int(value)),
            ha="center",
            color=navy,
            weight="bold",
        )
    ax_slice.text(
        0.02,
        0.96,
        "64 slices  ·  22.52T FLOPs  ·  1 GiB peak intermediate",
        transform=ax_slice.transAxes,
        ha="left",
        va="top",
        color=navy,
        fontsize=9.2,
        bbox={"boxstyle": "round,pad=0.45", "fc": "#FFF8EF", "ec": "#F2C58F"},
    )

    ax_proof.set_axis_off()
    ax_proof.set_title("d  Verified differentiable TN system", loc="left", color=navy, pad=8)
    _card(ax_proof, 0.25, 0.70, "qubits · 5×8 grid", "40q", "statevector-intractable", blue)
    _card(ax_proof, 0.75, 0.70, "training semantics", "F+B+SGD", "owner-sharded gradients", cyan)
    _card(ax_proof, 0.25, 0.27, "cross-node transport", "RDMA", "NCCL NET/IB · RoCE", green)
    _card(ax_proof, 0.75, 0.27, "numerical stability", "1–16 GPU", "identical value & ∥∇∥₂", orange)

    fig.suptitle(
        "FlagQuantum · differentiable tensor-network training at 40 qubits",
        fontsize=18,
        weight="bold",
        color=navy,
    )
    fig.text(
        0.5,
        -0.028,
        "Fixed 5×8 circuit · 4 entangling cycles · end-to-end complex128 · 64-slice cotengra path · explicit reverse · A800",
        ha="center",
        color="#536078",
        fontsize=10,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "flagquantum_tn_40q_strong_scaling"
    fig.savefig(stem.with_suffix(".png"), dpi=280, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    arguments = _arguments()
    series = _validated_series(arguments.input_dir)
    plot(series, arguments.output_dir)
    print(
        json.dumps(
            {
                "gpu_counts": WORLDS,
                "seconds": series["seconds"].tolist(),
                "speedup": series["speedup"].tolist(),
                "parallel_efficiency": series["efficiency"].tolist(),
                "value_real": series["value"],
                "gradient_l2": series["gradient_l2"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
