"""Plot aligned optimizer-step energy and relative-error comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.comparison.read_text(encoding="utf-8"))
    rows = payload["rows"]
    steps = [row["optimizer_step"] for row in rows]
    switch_steps = [
        row["optimizer_step"] for row in rows if row["hybrid_stage"] == "lbfgs"
    ]
    switch = min(switch_steps) if switch_steps else None
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(8.4, 5.0))
    axis.plot(steps, [row["adam_energy"] for row in rows], label="Adam", linewidth=2)
    axis.plot(
        steps,
        [row["hybrid_energy"] for row in rows],
        label="Adam → sharded L-BFGS",
        linewidth=2,
    )
    axis.axhline(
        payload["exact_ground_energy"], color="black", linestyle="--", label="Exact"
    )
    if switch is not None:
        axis.axvline(switch, color="gray", linestyle=":", label=f"Switch @ {switch}")
    axis.set_xlabel("Optimizer step")
    axis.set_ylabel("Energy")
    axis.set_title("N=8 distributed-MPS Heisenberg VQE")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "optimizer_step_energy.svg")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.4, 5.0))
    axis.semilogy(
        steps,
        [row["adam_relative_error"] for row in rows],
        label="Adam",
        linewidth=2,
    )
    axis.semilogy(
        steps,
        [row["hybrid_relative_error"] for row in rows],
        label="Adam → sharded L-BFGS",
        linewidth=2,
    )
    axis.axhline(
        payload["convergence_tolerance"],
        color="black",
        linestyle="--",
        label="Convergence gate",
    )
    if switch is not None:
        axis.axvline(switch, color="gray", linestyle=":", label=f"Switch @ {switch}")
    axis.set_xlabel("Optimizer step")
    axis.set_ylabel("Relative energy error")
    axis.set_title("N=8 distributed-MPS convergence")
    axis.grid(alpha=0.25, which="both")
    axis.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "optimizer_step_relative_error.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
