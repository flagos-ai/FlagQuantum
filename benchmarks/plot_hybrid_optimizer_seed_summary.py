"""Plot median and seed range against circuit-evaluation budget."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.summary.read_text(encoding="utf-8"))
    fig, axis = plt.subplots(figsize=(8.4, 5.0))
    for method, record in payload["methods"].items():
        curve = record["budget_curve"]
        x = [p["circuit_evaluations"] for p in curve]
        median = [p["median_best_error"] for p in curve]
        low = [p["min_best_error"] for p in curve]
        high = [p["max_best_error"] for p in curve]
        axis.plot(x, median, marker="o", linewidth=2, label=method.replace("_", " ").title())
        axis.fill_between(x, low, high, alpha=0.18)
    axis.axhline(payload["convergence_tolerance"], color="black", linestyle="--", label="1e-5 target")
    axis.set_yscale("log")
    axis.set_xlabel("Circuit-evaluation budget")
    axis.set_ylabel("Best relative energy error")
    axis.set_title("N=4 MLP-conditioned VQE: median and 3-seed range")
    axis.grid(alpha=0.25, which="both")
    axis.legend()
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".svg"))
    fig.savefig(args.output.with_suffix(".png"), dpi=180)


if __name__ == "__main__":
    main()
