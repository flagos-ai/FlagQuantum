"""Visualize noisy selector calibration measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    records = payload["records"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, (left, right) = plt.subplots(1, 2, figsize=(9.2, 3.8))
    colors = {8: "#2457A7", 10: "#4B8BBE", 12: "#D65F35", 16: "#8B5FBF"}
    for n_wires in (8, 10, 12, 16):
        rows = sorted(
            (
                item
                for item in records
                if item["mode"] == "noisy_statevector"
                and item["n_wires"] == n_wires
            ),
            key=lambda item: item["channel_count"],
        )
        left.plot(
            [item["channel_count"] for item in rows],
            [item["median_seconds"] for item in rows],
            "o-",
            label=f"{n_wires} qubits",
            color=colors[n_wires],
            linewidth=1.8,
        )
    left.set(
        xlabel="Lowered channel events",
        ylabel="Median execution time (s)",
        title="(a) Batched statevector cost",
    )
    left.grid(alpha=0.22)
    left.legend(frameon=False, ncols=2)

    comparison = [
        item
        for item in records
        if item["n_wires"] == 8
        and item["depth"] == 8
        and item["noise_kind"] == "full"
    ]
    order = ("noisy_statevector", "density_matrix", "noisy_mps")
    by_mode = {item["mode"]: item for item in comparison}
    values = [by_mode[mode]["median_seconds"] for mode in order]
    bars = right.bar(
        ("SV trajectory\n128 samples", "Exact density", "MPS trajectory\n8 samples"),
        values,
        color=("#2457A7", "#6C757D", "#D65F35"),
        width=0.65,
    )
    for bar, value in zip(bars, values, strict=True):
        right.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.25,
            f"{value:.2f} s",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
    right.set(
        ylabel="Median execution time (s)",
        title="(b) 8-qubit, depth-8, 120-channel workload",
        ylim=(0, max(values) * 1.14),
    )
    right.grid(axis="y", alpha=0.22)

    figure.suptitle(
        "FlagQuantum noisy-backend selector calibration on NVIDIA A800",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    figure.text(
        0.5,
        -0.02,
        "complex64 · SV batch 64 / 128 trajectories · MPS max bond 64 / 8 trajectories · median of 2 runs",
        ha="center",
        fontsize=8.2,
        color="#444444",
    )
    figure.tight_layout(pad=1.1, w_pad=2.2)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(
            args.output_prefix.with_suffix(f".{suffix}"),
            dpi=240,
            bbox_inches="tight",
        )
    plt.close(figure)


if __name__ == "__main__":
    main()
