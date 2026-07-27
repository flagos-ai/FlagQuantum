"""Plot convergence, cost, and target-crossing views for the MLP/VQE suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


LABELS = {"adam": "Adam", "block_qng": "Block-QNG", "lbfgs": "L-BFGS", "spsa": "SPSA"}


def line_plot(payload, output, x_key, y_key, xlabel, ylabel, log_y=False):
    fig, axis = plt.subplots(figsize=(8.6, 5.2))
    for experiment in payload["experiments"]:
        trace = experiment["trace"]
        axis.plot([p[x_key] for p in trace], [p[y_key] for p in trace], linewidth=2,
                  label=f'MLP Adam + quantum {LABELS[experiment["method"]]}')
    if y_key == "energy":
        axis.axhline(payload["exact_ground_energy"], color="black", linestyle="--", label="Exact")
    if log_y:
        axis.set_yscale("log")
        axis.axhline(1e-5, color="black", linestyle="--", label="1e-5 target")
    axis.set(xlabel=xlabel, ylabel=ylabel,
             title=f'N={payload["n_wires"]}, depth={payload["depth"]} MLP-conditioned Heisenberg VQE')
    axis.grid(alpha=0.25, which="both")
    axis.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".svg"))
    fig.savefig(output.with_suffix(".png"), dpi=180)
    plt.close(fig)


def target_plot(payload, output):
    targets = [1e-2, 1e-3, 1e-4, 1e-5]
    methods = [e["method"] for e in payload["experiments"]]
    width = 0.8 / len(methods)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))
    for method_index, experiment in enumerate(payload["experiments"]):
        crossings = experiment["target_crossings"]
        offsets = [i - 0.4 + width / 2 + method_index * width for i in range(len(targets))]
        evaluations = [crossings[str(t)]["circuit_evaluations"] if crossings[str(t)] else float("nan") for t in targets]
        seconds = [crossings[str(t)]["wall_time_seconds"] if crossings[str(t)] else float("nan") for t in targets]
        label = LABELS[experiment["method"]]
        axes[0].bar(offsets, evaluations, width=width, label=label)
        axes[1].bar(offsets, seconds, width=width, label=label)
    labels = [f"1e-{int(-__import__('math').log10(t))}" for t in targets]
    for axis, ylabel in zip(axes, ("Circuit evaluations", "Wall time (s)")):
        axis.set_xticks(range(len(targets)), labels)
        axis.set_xlabel("Relative-error target")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2, axis="y")
    axes[0].legend(fontsize=8)
    fig.suptitle("Cost to reach accuracy target (missing bar = not reached)")
    fig.tight_layout()
    fig.savefig(output.with_suffix(".svg"))
    fig.savefig(output.with_suffix(".png"), dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.comparison.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    specifications = (
        ("optimizer_step", "energy", "Optimizer step", "Energy", "optimizer_step_energy", False),
        ("optimizer_step", "relative_error", "Optimizer step", "Relative energy error", "optimizer_step_relative_error", True),
        ("circuit_evaluations", "energy", "Circuit evaluations", "Energy", "circuit_evaluation_energy", False),
        ("wall_time_seconds", "energy", "Wall time (s)", "Energy", "wall_time_energy", False),
        ("circuit_evaluations", "relative_error", "Circuit evaluations", "Relative energy error", "circuit_evaluation_relative_error", True),
    )
    for x, y, xlabel, ylabel, name, log_y in specifications:
        line_plot(payload, args.output_dir / name, x, y, xlabel, ylabel, log_y)
    target_plot(payload, args.output_dir / "accuracy_target_cost")


if __name__ == "__main__":
    main()
