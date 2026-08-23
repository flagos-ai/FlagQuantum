#!/usr/bin/env python3
"""Plot an accuracy-aware FlagQuantum versus TensorCircuit-NG TN comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.2,
            "axes.titleweight": "bold",
            "axes.titlesize": 11.8,
            "axes.edgecolor": "#AAB4C5",
            "axes.grid": True,
            "grid.color": "#DCE2EC",
            "grid.alpha": 0.7,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "#FAFBFD",
            "savefig.facecolor": "white",
        }
    )


def main() -> None:
    args = _arguments()
    worlds = np.asarray((1, 2, 4, 8, 16), dtype=float)
    fq = tuple(
        _load(
            args.input_dir
            / f"fq_40q_4c_shared64_e2e128_{int(world)}gpu.json"
        )
        for world in worlds
    )
    tcng_worlds = np.asarray((1, 2, 4, 8, 16), dtype=float)
    tcng = (
        _load(args.input_dir / "tcng_40q_4c_1gpu.json"),
        *(
            _load(
                args.input_dir
                / (
                    "tcng_40q_4c_e2e128_16gpu_2node.json"
                    if int(world) == 16
                    else f"tcng_40q_4c_e2e128_{int(world)}gpu.json"
                )
            )
            for world in tcng_worlds[1:]
        ),
    )
    fq_seconds = np.asarray([item["max_execution_seconds"] for item in fq])
    tcng_seconds = np.asarray(
        [float(item["steady_step_seconds"]) for item in tcng]
    )
    fq_speedup = fq_seconds[0] / fq_seconds
    tcng_speedup = tcng_seconds[0] / tcng_seconds
    if any(int(item["slice_count"]) != 64 for item in fq):
        raise ValueError("FlagQuantum inputs do not share the 64-slice path")
    for world, item in zip(tcng_worlds, tcng):
        if int(item["devices"]) != int(world):
            raise ValueError("TensorCircuit-NG device count does not match")
        if int(item["slices"]) != 64:
            raise ValueError("TensorCircuit-NG inputs do not use 64 slices")
        if int(item["tree_flops"]) != int(fq[0]["total_estimated_flops"]):
            raise ValueError("the compared contraction FLOP counts differ")

    _style()
    navy = "#142B4F"
    blue = "#2F6FED"
    orange = "#F28E2B"
    green = "#2CA56C"
    gray = "#8995A8"
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.4), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(worlds, worlds, "--", color=gray, lw=1.6, label="Ideal")
    ax.plot(worlds, fq_speedup, "o-", color=blue, lw=2.8, ms=7.5, label="FlagQuantum")
    ax.plot(tcng_worlds, tcng_speedup, "D-", color=orange, lw=2.5, ms=6.8, label="TensorCircuit-NG")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(worlds, [str(int(value)) for value in worlds])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.set_yticks(worlds, [str(int(value)) for value in worlds])
    ax.set_title("a  Fixed-path strong-scaling speedup", loc="left", color=navy)
    ax.set_xlabel("NVIDIA A800 GPUs")
    ax.set_ylabel("Speedup × ↑")
    ax.legend(loc="upper left")
    ax.annotate(f"{tcng_speedup[-1]:.2f}×", (16, tcng_speedup[-1]), xytext=(-8, 9), textcoords="offset points", ha="right", color=orange, weight="bold")
    ax.annotate(f"{fq_speedup[-1]:.2f}×", (16, fq_speedup[-1]), xytext=(-8, -18), textcoords="offset points", ha="right", color=blue, weight="bold")
    ax.text(
        0.97,
        0.08,
        f"16 GPUs · 2 nodes\nFQ {fq_seconds[-1]:.2f}s · TC-NG {tcng_seconds[-1]:.2f}s",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=navy,
        bbox={"boxstyle": "round,pad=0.4", "fc": "white", "ec": "#C9D2E2"},
    )

    ax = axes[0, 1]
    x = np.arange(4)
    width = 0.36
    fq_bars = ax.bar(x - width / 2, fq_seconds[:4], width, color=blue, label="FlagQuantum")
    tcng_bars = ax.bar(x + width / 2, tcng_seconds[:4], width, color=orange, label="TensorCircuit-NG")
    ax.set_xticks(x, ["1", "2", "4", "8"])
    ax.set_xlabel("NVIDIA A800 GPUs")
    ax.set_ylabel("Forward + backward + SGD (s) ↓")
    ax.set_title("b  Single-node execution", loc="left", color=navy)
    ax.set_ylim(0, 47)
    ax.legend(loc="upper right")
    for bar, value in zip((*fq_bars, *tcng_bars), (*fq_seconds[:4], *tcng_seconds[:4])):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.1, f"{value:.2f}s", ha="center", weight="bold", color=navy)
    ax.text(
        0.97,
        0.71,
        f"TC-NG is {fq_seconds[0] / tcng_seconds[0]:.2f}–{fq_seconds[3] / tcng_seconds[3]:.2f}× faster\non 1–8 GPUs",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=navy,
        bbox={"boxstyle": "round,pad=0.45", "fc": "white", "ec": "#C9D2E2"},
    )

    ax = axes[1, 0]
    labels = ["value", "gradient L2"]
    fq_reference_error = [4.163336342344337e-17, 2.5926572592812409e-16]
    tcng_reference_error = [3.0e-17, 1.2e-16]
    x = np.arange(2)
    width = 0.34
    ax.bar(x - width / 2, fq_reference_error, width, color=blue, label="FlagQuantum TN")
    ax.bar(x + width / 2, tcng_reference_error, width, color=orange, label="TC-NG TN")
    ax.set_yscale("log")
    ax.set_ylim(1e-18, 1e-6)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Absolute error vs exact statevector ↓")
    ax.set_title("c  Accuracy gate · reduced 12q twin", loc="left", color=navy)
    ax.legend(loc="upper right")
    ax.text(
        0.03,
        0.88,
        "Both pass exact value + gradient checks",
        transform=ax.transAxes,
        color=green,
        weight="bold",
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#B9DEC9"},
    )

    ax = axes[1, 1]
    ax.set_axis_off()
    ax.set_title("d  Distributed training capability", loc="left", color=navy)
    rows = [
        ("Fixed-path exact reverse", "✓", "✓"),
        ("Owner-sharded gradients", "✓", "replicated"),
        ("Sharded optimizer update", "✓", "replicated"),
        ("1 / 2 / 4 / 8 GPU evidence", "✓", "✓"),
        ("Two-node · 16 GPU", "✓", "✓"),
        ("RDMA / NCCL NET/IB", "✓", "not tested"),
    ]
    table = ax.table(
        cellText=[[name, left, right] for name, left, right in rows],
        colLabels=["Capability", "FlagQuantum", "TC-NG"],
        colWidths=[0.52, 0.23, 0.25],
        cellLoc="center",
        bbox=[0.0, 0.02, 1.0, 0.88],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#D1D8E4")
        if row == 0:
            cell.set_facecolor(navy)
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        elif col == 1:
            cell.set_facecolor("#EDF5FF")
            cell.get_text().set_color(blue)
            cell.get_text().set_weight("bold")
        elif col == 2:
            cell.set_facecolor("#FFF5EA")
        else:
            cell.set_facecolor("white")

    fig.suptitle(
        "FlagQuantum vs TensorCircuit-NG · accuracy-aware TN comparison",
        fontsize=17.5,
        weight="bold",
        color=navy,
    )
    fig.text(
        0.5,
        -0.028,
        "40q · 5×8 grid · 4 cycles · end-to-end complex128 · 64 slices · 22.52T contraction FLOPs · A800",
        ha="center",
        color="#536078",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "flagquantum_vs_tcng_tn_40q"
    fig.savefig(stem.with_suffix(".png"), dpi=280, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
