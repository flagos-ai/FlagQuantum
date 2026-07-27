#!/usr/bin/env python3
"""Plot the measured Lightning-GPU single- and multi-node scaling boundary."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/flagquantum-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE = "#0072B2"
ORANGE = "#E69F00"
RED = "#D55E00"
GREY = "#6C757D"

ROOT = Path(__file__).resolve().parents[1] / "results" / "comparison"
OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "statevector_submission_figures"
    / "fig6_lightning_gpu_multinode_cliff"
)


def load(world: int) -> dict:
    path = ROOT / f"pennylane_lightning_gpu_28q_d8_{world}xa800_mpi.json"
    return json.loads(path.read_text(encoding="utf-8"))


def bootstrap_median_ci(samples: list[float], seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(samples)
    medians = np.median(
        rng.choice(values, size=(20_000, values.size), replace=True), axis=1
    )
    return tuple(np.quantile(medians, [0.025, 0.975]))


def main() -> None:
    worlds = [1, 8, 16]
    payloads = [load(world) for world in worlds]
    samples = [payload["value_and_grad"]["samples_seconds"] for payload in payloads]
    medians = np.asarray([np.median(values) for values in samples])
    cis = [bootstrap_median_ci(values, 440044 + world) for world, values in zip(worlds, samples)]
    errors = np.asarray(
        [[median - low for median, (low, _) in zip(medians, cis)],
         [high - median for median, (_, high) in zip(medians, cis)]]
    )
    speedups = medians[0] / medians

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.titleweight": "bold",
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.7,
            "grid.linewidth": 0.5,
            "grid.alpha": 0.3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.12, 2.35))
    colors = [BLUE, BLUE, RED]

    axes[0].bar(range(3), medians, color=colors, width=0.62)
    axes[0].errorbar(
        range(3), medians, yerr=errors, fmt="none", ecolor="black", capsize=3, lw=0.8
    )
    axes[0].set(
        xticks=range(3),
        xticklabels=["1 GPU\n1 node", "8 GPUs\n1 node", "16 GPUs\n2 nodes"],
        ylabel="Value + full gradient (s)",
        title=r"$\bf{a}$  Absolute latency",
    )
    for index, value in enumerate(medians):
        axes[0].text(index, value + max(medians) * 0.025, f"{value:.3f}", ha="center")

    axes[1].plot(range(3), speedups, "o-", color=ORANGE, lw=1.8)
    axes[1].axhline(1, color=GREY, ls="--", lw=1)
    axes[1].axvspan(1.5, 2.5, color=RED, alpha=0.08)
    axes[1].set(
        xticks=range(3),
        xticklabels=["1 GPU\n1 node", "8 GPUs\n1 node", "16 GPUs\n2 nodes"],
        ylabel="Speedup vs. 1 GPU",
        title=r"$\bf{b}$  Scale-out boundary",
    )
    axes[1].annotate(
        f"{speedups[1]:.2f}×",
        (1, speedups[1]),
        xytext=(0, -19),
        textcoords="offset points",
        ha="center",
        va="top",
        weight="bold",
    )
    axes[1].annotate(
        f"{speedups[2]:.2f}×\n(10.36× slower than 8 GPU)",
        (2, speedups[2]),
        xytext=(0, 21),
        textcoords="offset points",
        ha="center",
        va="bottom",
        color=RED,
        weight="bold",
    )
    axes[1].text(
        1.97,
        axes[1].get_ylim()[1] * 0.84,
        "inter-node",
        ha="right",
        va="top",
        color=RED,
    )

    for ax in axes:
        ax.grid(True, axis="y")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "PennyLane Lightning-GPU, 28 qubits, depth 8, adjoint full gradient",
        fontsize=9,
        weight="bold",
        y=1.02,
    )
    fig.tight_layout(w_pad=2.0)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".svg"):
        fig.savefig(OUTPUT.with_suffix(suffix), bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
