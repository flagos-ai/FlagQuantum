#!/usr/bin/env python3
"""Combine matched-workload artifacts into planner crossover thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flagquantum.runtime.observability.performance import calibrate_crossover


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payloads = [json.loads(path.read_text()) for path in args.artifacts]
    measurements = [
        {
            "world_size": item["world_size"],
            "global_elements": item["workload_dimensions"]["global_elements"],
            "median_seconds": item["median_seconds"],
            "relative_uncertainty": item["robust_coefficient_of_variation"],
        }
        for item in payloads
    ]
    world_sizes = sorted({int(row["world_size"]) for row in measurements})
    if not world_sizes or world_sizes[0] != 1:
        raise ValueError("crossover calibration requires a one-GPU baseline")
    selections = {
        str(world): calibrate_crossover(
            measurements, distributed_world_size=world
        ).__dict__
        for world in world_sizes[1:]
    }
    output = {
        "schema": "flagquantum_crossover_v1",
        "workload": "matched_global_pointwise_with_scalar_progress_reduction",
        "dimensions": ["global_elements", "world_size"],
        "measurements": measurements,
        "planner_selection_thresholds": selections,
        "source_artifacts": [str(path) for path in args.artifacts],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
