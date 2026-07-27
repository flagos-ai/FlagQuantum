"""Plot cost-aware comparisons for the hybrid classical/quantum VQE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


LABELS = {
    "all_adam": "Classical Adam + quantum Adam",
    "adam_block_qng": "Classical Adam + quantum block-QNG",
}


def _plot(payload: dict, *, x_key: str, y_key: str, xlabel: str, ylabel: str,
          output: Path, log_y: bool = False) -> None:
    fig, axis = plt.subplots(figsize=(8.4, 5.0))
    for experiment in payload["experiments"]:
        trace = experiment["trace"]
        axis.plot(
            [point[x_key] for point in trace],
            [point[y_key] for point in trace],
            label=LABELS[experiment["method"]],
            linewidth=2,
        )
    if y_key == "energy":
        axis.axhline(
            payload["exact_ground_energy"], color="black", linestyle="--", label="Exact"
        )
    if log_y:
        axis.set_yscale("log")
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(
        f'N={payload["n_wires"]} classically gated Heisenberg VQE'
    )
    axis.grid(alpha=0.25, which="both")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output.with_suffix(".svg"))
    fig.savefig(output.with_suffix(".png"), dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.comparison.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plots = (
        ("optimizer_step", "energy", "Optimizer step", "Energy", "optimizer_step_energy", False),
        ("circuit_evaluations", "energy", "Circuit evaluations", "Energy", "circuit_evaluation_energy", False),
        ("wall_time_seconds", "energy", "Wall time (s)", "Energy", "wall_time_energy", False),
        ("circuit_evaluations", "relative_error", "Circuit evaluations", "Relative energy error", "circuit_evaluation_relative_error", True),
    )
    for x_key, y_key, xlabel, ylabel, name, log_y in plots:
        _plot(
            payload,
            x_key=x_key,
            y_key=y_key,
            xlabel=xlabel,
            ylabel=ylabel,
            output=args.output_dir / name,
            log_y=log_y,
        )


if __name__ == "__main__":
    main()
