"""Plot the measured phase composition of exact block-QNG steps."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.profile.read_text(encoding="utf-8"))
    rows = payload["records"]
    phases = (
        ("objective_gradient_seconds", "Objective + gradient"),
        ("real_jacobian_seconds", "Real Jacobian"),
        ("imag_jacobian_seconds", "Imag Jacobian"),
        ("metric_reduction_seconds", "Metric reduction"),
        ("linear_solve_seconds", "Linear solve"),
    )
    steps = [row["optimizer_step"] for row in rows]
    bottom = [0.0] * len(rows)
    fig, axis = plt.subplots(figsize=(8.4, 5.0))
    for key, label in phases:
        values = [row[key] for row in rows]
        axis.bar(steps, values, bottom=bottom, label=label)
        bottom = [left + right for left, right in zip(bottom, values)]
    other = [max(0.0, row["wall_time_seconds"] - value) for row, value in zip(rows, bottom)]
    axis.bar(steps, other, bottom=bottom, label="Other/update")
    axis.set_xlabel("QNG optimizer step")
    axis.set_ylabel("Wall time (s)")
    axis.set_title(
        f'Exact block-QNG phase breakdown: N={payload["n_wires"]}, depth={payload["depth"]}'
    )
    axis.grid(alpha=0.2, axis="y")
    axis.legend(fontsize=8)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".svg"))
    fig.savefig(args.output.with_suffix(".png"), dpi=180)


if __name__ == "__main__":
    main()
