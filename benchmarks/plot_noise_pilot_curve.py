"""Plot the A800 noisy-selector pilot-size curve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    records = payload["records"]
    sizes = [item["pilot_trajectories"] for item in records]
    sample = [item["maximum_sample_variance"] for item in records]
    upper = [item["variance_upper_confidence_bound"] for item in records]
    target = [item["estimated_trajectories_to_target"] for item in records]

    plt.rcParams.update({"font.size": 9, "axes.titleweight": "bold"})
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7), constrained_layout=True)
    axes[0].plot(sizes, sample, "o-", label="Pilot sample variance", color="#2471A3")
    axes[0].plot(
        sizes, upper, "s-", label="95% simultaneous upper bound", color="#C0392B"
    )
    axes[0].set(
        xscale="log", yscale="log", xlabel="Pilot trajectories", ylabel="Variance"
    )
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend(frameon=False)
    axes[0].set_title("(a) Statistical evidence tightens slowly")

    axes[1].plot(sizes, target, "o-", color="#117864")
    axes[1].axhline(
        payload["trajectory_cap"], color="0.4", ls="--", label="Trajectory cap"
    )
    axes[1].set(
        xscale="log",
        yscale="log",
        xlabel="Pilot trajectories",
        ylabel="Estimated trajectories to target",
    )
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].legend(frameon=False)
    axes[1].set_title("(b) Conservative time-to-target planning")
    fig.suptitle("FlagQuantum noisy simulation · A800 pilot-size study", fontsize=12)

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(args.output_prefix.with_suffix(f".{suffix}"), dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    main()
