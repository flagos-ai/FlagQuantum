"""Render the audited ADAPT-VQE TN comparison figure."""

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
    fq_color = "#0B6EFD"
    tc_color = "#F28E2B"
    accent = "#16A085"
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.6), constrained_layout=True)

    audits = [load(root / f"audit_{qubits}q_append.json") for qubits in (4, 6)]
    x = np.arange(2)
    width = 0.34
    fq_seconds = [item["flagquantum_seconds"] for item in audits]
    tc_seconds = [item["tensorcircuit_ng_seconds"] for item in audits]
    axes[0].bar(x - width / 2, fq_seconds, width, color=fq_color, label="FlagQuantum")
    axes[0].bar(x + width / 2, tc_seconds, width, color=tc_color, label="TensorCircuit-NG")
    for index, item in enumerate(audits):
        axes[0].text(
            index,
            max(fq_seconds[index], tc_seconds[index]) * 1.04,
            f'{item["flagquantum_speedup"]:.1f}× faster',
            ha="center",
            color=fq_color,
            fontweight="bold",
        )
    axes[0].set_xticks(x, ("4 qubits", "6 qubits"))
    axes[0].set_ylabel("End-to-end time (s, lower is better)")
    axes[0].set_title("a  Dynamic ansatz growth")
    axes[0].legend(frameon=False, loc="upper left")
    axes[0].grid(axis="y", alpha=0.18)

    stages = [
        ("Append + autograd", load(root / "fq_12q_probe.json")),
        ("Commutator", load(root / "fq_12q_commutator_probe.json")),
        ("Block MPO", load(root / "fq_12q_block_mpo_probe.json")),
    ]
    totals = [item[1]["execution_seconds"] for item in stages]
    screening = [sum(stage["screening_seconds"] for stage in item[1]["iterations"]) for item in stages]
    colors = ["#A9C9FF", "#58A6FF", accent]
    axes[1].bar(range(3), totals, color=colors)
    for index, (total, screen) in enumerate(zip(totals, screening)):
        axes[1].text(index, total + 1.0, f"{total:.1f}s", ha="center", fontweight="bold")
        axes[1].text(index, total * 0.48, f"screen\n{screen:.1f}s", ha="center", color="#17324D")
    axes[1].set_xticks(range(3), [item[0] for item in stages], rotation=12, ha="right")
    axes[1].set_ylabel("FlagQuantum end-to-end time (s)")
    axes[1].set_title("b  Exact screening, same 12q result")
    axes[1].grid(axis="y", alpha=0.18)

    errors = [
        max(
            item["final_energy_absolute_error"],
            item["max_pool_gradient_absolute_error"],
            item["max_optimization_history_absolute_error"],
        )
        for item in audits
    ]
    axes[2].semilogy((4, 6), errors, "o-", color=accent, linewidth=2.2, markersize=7)
    axes[2].axhline(1e-12, color="#777777", linestyle="--", linewidth=1, label="audit tolerance")
    axes[2].set_xticks((4, 6))
    axes[2].set_ylim(1e-17, 1e-11)
    axes[2].set_xlabel("Qubits")
    axes[2].set_ylabel("Maximum absolute discrepancy")
    axes[2].set_title("c  complex128 trajectory agreement")
    axes[2].grid(alpha=0.18, which="both")
    axes[2].legend(frameon=False, loc="upper left")
    axes[2].text(
        0.04,
        0.08,
        "12q TC-NG comparator: not completed\nFQ follow-up: 20q depth-6 + 36q shallow completed",
        transform=axes[2].transAxes,
        fontsize=9,
        color="#555555",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#F5F5F5", "edgecolor": "none"},
    )

    fig.suptitle(
        "FlagQuantum TN for ADAPT-VQE — fast dynamic growth, audited numerics, explicit limits",
        fontsize=15,
        fontweight="bold",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")


if __name__ == "__main__":
    main()
