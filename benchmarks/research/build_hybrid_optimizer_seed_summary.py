"""Aggregate repeated hybrid-optimizer runs without hiding seed variability."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def best_at(trace: list[dict], budget: int) -> float | None:
    values = [p["relative_error"] for p in trace if p["circuit_evaluations"] <= budget]
    return min(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    keys = ("n_wires", "depth", "steps", "precision")
    if any(tuple(p[k] for k in keys) != tuple(payloads[0][k] for k in keys) for p in payloads[1:]):
        raise ValueError("seed runs do not describe the same workload")
    methods = sorted(set.intersection(*[{e["method"] for e in p["experiments"]} for p in payloads]))
    budgets = (25, 50, 71, 100, 121, 150, 200, 250, 301)
    summary = {}
    for method in methods:
        experiments = [next(e for e in p["experiments"] if e["method"] == method) for p in payloads]
        finals = [e["best_relative_error"] for e in experiments]
        crossings = [e["target_crossings"]["1e-05"] for e in experiments]
        summary[method] = {
            "best_error": {"median": statistics.median(finals), "min": min(finals), "max": max(finals)},
            "converged_seeds": sum(crossing is not None for crossing in crossings),
            "total_seeds": len(experiments),
            "target_1e-5_evaluations": [None if c is None else c["circuit_evaluations"] for c in crossings],
            "budget_curve": [
                {
                    "circuit_evaluations": budget,
                    "median_best_error": statistics.median(values),
                    "min_best_error": min(values),
                    "max_best_error": max(values),
                }
                for budget in budgets
                if (values := [best_at(e["trace"], budget) for e in experiments]) and all(v is not None for v in values)
            ],
        }
    output = {
        "schema": "flagquantum.hybrid_optimizer_seed_summary.v1",
        "workload": {key: payloads[0][key] for key in keys},
        "seeds": [p["seed"] for p in payloads],
        "convergence_tolerance": 1e-5,
        "methods": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
