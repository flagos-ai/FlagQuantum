"""Plot A800 strong-scaling evidence for noisy statevector trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    records = sorted(
        (json.loads(path.read_text(encoding="utf-8")) for path in args.inputs),
        key=lambda item: item["world_size"],
    )
    worlds = np.array([item["world_size"] for item in records])
    throughput = np.array([item["trajectories_per_second"] for item in records])
    speedup = throughput / throughput[0]
    efficiency = 100 * speedup / worlds

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
        }
    )
    figure, (left, right) = plt.subplots(1, 2, figsize=(9.2, 3.8))
    color = "#2457A7"
    accent = "#D65F35"

    left.plot(worlds, throughput, "o-", color=color, linewidth=2, label="Measured")
    left.plot(
        worlds,
        throughput[0] * worlds,
        "--",
        color="#8A8A8A",
        linewidth=1.4,
        label="Ideal linear scaling",
    )
    for x_value, y_value in zip(worlds, throughput, strict=True):
        left.annotate(
            f"{y_value:.1f}",
            (x_value, y_value),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            color=color,
            fontsize=9,
        )
    left.set(
        xlabel="A800 GPUs",
        ylabel="Throughput (trajectories/s)",
        title="(a) Fixed-workload throughput",
        xticks=worlds,
    )
    left.grid(axis="y", alpha=0.22)
    left.legend(frameon=False, loc="upper left")

    bars = right.bar(worlds, efficiency, width=0.72, color=accent, alpha=0.88)
    right.axhline(100, color="#8A8A8A", linestyle="--", linewidth=1.2)
    for bar, value, gain in zip(bars, efficiency, speedup, strict=True):
        right.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() - 3.0,
            f"{value:.0f}%\n{gain:.2f}×",
            ha="center",
            va="top",
            fontsize=7.5,
            color="white",
            fontweight="bold",
        )
    right.set(
        xlabel="A800 GPUs",
        ylabel="Strong-scaling efficiency",
        title="(b) Speedup and parallel efficiency",
        xticks=worlds,
        ylim=(0, 116),
    )
    right.set_yticks((0, 25, 50, 75, 100), labels=("0%", "25%", "50%", "75%", "100%"))
    right.grid(axis="y", alpha=0.22)

    first = records[0]
    figure.suptitle(
        "FlagQuantum batched noisy-statevector trajectory scaling",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    figure.text(
        0.5,
        -0.015,
        f"{first['device_name']} · {first['n_wires']} qubits · depth {first['depth']} · "
        f"{first.get('requested_trajectories', first.get('trajectories'))} fixed global trajectories · batch "
        f"{first['trajectory_batch_size']} · median of {len(first['repeat_seconds'])} runs",
        ha="center",
        fontsize=8.5,
        color="#444444",
    )
    figure.tight_layout(pad=1.1, w_pad=2.2)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(
            args.output_prefix.with_suffix(f".{suffix}"),
            bbox_inches="tight",
            dpi=240,
        )
    plt.close(figure)


if __name__ == "__main__":
    main()
