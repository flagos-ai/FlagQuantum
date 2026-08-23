"""Render the 48-qubit exact differentiable-TN README hero figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = arguments()
    root = args.results
    fq = "#1769E0"
    fq_light = "#8BB9F8"
    tc = "#F28E2B"
    ink = "#183153"
    green = "#159A80"
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.labelcolor": ink,
            "axes.edgecolor": "#B9C5D3",
            "xtick.color": ink,
            "ytick.color": ink,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8), constrained_layout=True)

    full = load(root / "adapt1_adam1_1gpu.json")
    sv_gib = 4 * 1024 * 1024
    fq_gib = full["peak_cuda_allocated_bytes"] / 2**30
    axes[0].bar([0, 1], [sv_gib, fq_gib], color=["#D7DEE8", fq], width=0.58)
    axes[0].set_yscale("log")
    axes[0].set_xticks([0, 1], ["Statevector\n(theoretical)", "FlagQuantum TN\n(measured)"])
    axes[0].set_ylabel("Memory footprint (GiB, log scale)")
    axes[0].set_title("a   48 qubits beyond statevectors", loc="left")
    axes[0].grid(axis="y", which="both", alpha=0.16)
    axes[0].text(0, sv_gib * 1.15, "4 PiB", ha="center", fontweight="bold", color=ink)
    axes[0].text(1, fq_gib * 1.3, f"{fq_gib:.2f} GiB", ha="center", fontweight="bold", color=fq)
    axes[0].text(
        0.5,
        0.56,
        f"{sv_gib / fq_gib:,.0f}×\nsmaller",
        transform=axes[0].transAxes,
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color=fq,
        bbox={"boxstyle": "round,pad=0.42", "fc": "#EEF5FF", "ec": "none"},
    )

    gpus = np.array([1, 2, 4, 8])
    runs = [load(root / f"distributed_{gpu}gpu.json") for gpu in gpus]
    cold = np.array([run["execution_seconds"] for run in runs])
    core = np.array(
        [
            max(item["screening_seconds"] for item in run["rank_metrics"])
            + max(item["reverse_seconds"] for item in run["rank_metrics"])
            for run in runs
        ]
    )
    axes[1].plot(gpus, cold[0] / cold, "o-", color=fq_light, linewidth=2.2, label="Cold end-to-end")
    axes[1].plot(gpus, core[0] / core, "o-", color=fq, linewidth=2.6, label="Iterative TN core")
    axes[1].plot(gpus, gpus, "--", color="#8B97A6", linewidth=1.2, label="Ideal")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(gpus, [str(x) for x in gpus])
    axes[1].set_xlabel("A800 GPUs")
    axes[1].set_ylabel("Speedup vs 1 GPU")
    axes[1].set_title("b   Real 16-slice strong scaling", loc="left")
    axes[1].grid(alpha=0.16)
    axes[1].legend(frameon=False, loc="upper left")
    axes[1].annotate(
        f"{core[0] / core[-1]:.2f}× core",
        xy=(8, core[0] / core[-1]),
        xytext=(4.6, 7.4),
        arrowprops={"arrowstyle": "->", "color": fq},
        color=fq,
        fontweight="bold",
    )
    axes[1].text(8, cold[0] / cold[-1] + 0.28, f"{cold[0] / cold[-1]:.2f}× cold", ha="center", color="#5277A5")

    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(-0.5, 2.7)
    axes[2].axis("off")
    axes[2].set_title("c   Same 48q complex128 challenge", loc="left")
    cards = [
        (2.05, fq, "FlagQuantum", "16 candidates", "✓ ADAPT + reverse + Adam", "21.12 s"),
        (1.05, tc, "TC-NG · direct MPO", "1 candidate", "✕ path cannot execute", "2.68×10²⁹ slices"),
        (0.05, tc, "TC-NG · TT-SVD MPO", "1 candidate", "◷ XLA compile timeout", "> 180 s"),
    ]
    for y, color, name, scope, status, metric in cards:
        axes[2].add_patch(
            plt.Rectangle((0.02, y - 0.28), 0.96, 0.72, transform=axes[2].transData, color="#F7F9FC", ec="#DCE3EC", lw=1)
        )
        axes[2].add_patch(plt.Rectangle((0.02, y - 0.28), 0.018, 0.72, transform=axes[2].transData, color=color))
        axes[2].text(0.07, y + 0.24, name, color=color, fontweight="bold", fontsize=11)
        axes[2].text(0.07, y + 0.02, f"{scope}  ·  {status}", color=ink, fontsize=9.2)
        axes[2].text(0.93, y - 0.17, metric, ha="right", color=color, fontsize=12, fontweight="bold")
    axes[2].text(
        0.02,
        -0.35,
        "Exact workload · no statevector materialization · comparator scopes shown explicitly",
        color="#66788A",
        fontsize=8.5,
    )

    fig.suptitle(
        "FlagQuantum exact differentiable tensor networks — train where statevectors cannot fit",
        fontsize=16,
        fontweight="bold",
        color=ink,
    )
    fig.text(
        0.5,
        -0.015,
        "48q grid ADAPT-VQE  •  complex128  •  exact Pauli-MPO  •  forward + reverse + optimizer",
        ha="center",
        color=green,
        fontweight="bold",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
