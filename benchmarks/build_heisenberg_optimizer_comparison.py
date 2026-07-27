"""Build lightweight aligned JSON/CSV traces for Heisenberg optimizer plots."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("distribution_semantics") != "sharded_across_ranks":
        raise ValueError(f"{path} is not a sharded distributed artifact")
    if not payload.get("optimization_trace"):
        raise ValueError(f"{path} has no optimization_trace")
    return payload


def build(adam: dict, hybrid: dict) -> dict:
    matched = (
        "n_sites",
        "ansatz_depth",
        "parameter_count",
        "parameterization",
        "precision",
        "seed",
        "steps",
        "world_size",
        "initial_state",
    )
    mismatches = [key for key in matched if adam.get(key) != hybrid.get(key)]
    if mismatches:
        raise ValueError(f"optimizer artifacts are not matched: {mismatches}")
    adam_trace = adam["optimization_trace"]
    hybrid_trace = hybrid["optimization_trace"]
    if len(adam_trace) != len(hybrid_trace):
        raise ValueError("optimizer traces have different step counts")
    exact = float(adam["exact_ground_energy"])
    rows = []
    for adam_step, hybrid_step in zip(adam_trace, hybrid_trace):
        if adam_step["optimizer_step"] != hybrid_step["optimizer_step"]:
            raise ValueError("optimizer traces are not step-aligned")
        rows.append(
            {
                "optimizer_step": int(adam_step["optimizer_step"]),
                "exact_energy": exact,
                "adam_energy": float(adam_step["energy"]),
                "hybrid_energy": float(hybrid_step["energy"]),
                "adam_relative_error": float(adam_step["relative_error"]),
                "hybrid_relative_error": float(hybrid_step["relative_error"]),
                "hybrid_stage": str(hybrid_step["optimizer_stage"]),
            }
        )
    return {
        "schema": "flagquantum.heisenberg_optimizer_comparison.v1",
        "workload": {key: adam[key] for key in matched},
        "exact_ground_energy": exact,
        "convergence_tolerance": 1e-5,
        "adam_final_relative_error": rows[-1]["adam_relative_error"],
        "hybrid_final_relative_error": rows[-1]["hybrid_relative_error"],
        "adam_converged": rows[-1]["adam_relative_error"] <= 1e-5,
        "hybrid_converged": rows[-1]["hybrid_relative_error"] <= 1e-5,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adam", type=Path, required=True)
    parser.add_argument("--hybrid", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    args = parser.parse_args()
    comparison = build(_load(args.adam), _load(args.hybrid))
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with args.csv_output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(comparison["rows"][0]))
        writer.writeheader()
        writer.writerows(comparison["rows"])
    print(json.dumps({"json": str(args.json_output), "csv": str(args.csv_output)}))


if __name__ == "__main__":
    main()
