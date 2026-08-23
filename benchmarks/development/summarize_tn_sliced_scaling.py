"""Validate and summarize a TN sliced-gradient strong-scaling series."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    records = []
    for world_size in (1, 2, 4, 8):
        path = arguments.input / f"12q_8l_64s_{world_size}gpu.json"
        records.append(json.loads(path.read_text(encoding="utf-8")))
    baseline = records[0]
    identity_fields = (
        "workload",
        "dtype",
        "parameter_count",
        "sliced_labels",
        "slice_count",
        "total_estimated_flops",
    )
    for record, world_size in zip(records, (1, 2, 4, 8)):
        if record["world_size"] != world_size:
            raise ValueError("TN scaling record world size is inconsistent")
        if any(record[field] != baseline[field] for field in identity_fields):
            raise ValueError("TN scaling records do not describe one workload")
        if not record["correctness"]["passed"]:
            raise ValueError("TN scaling record failed correctness validation")
    baseline_seconds = float(baseline["max_execution_seconds"])
    scaling = []
    for record in records:
        world_size = int(record["world_size"])
        seconds = float(record["max_execution_seconds"])
        speedup = baseline_seconds / seconds
        scaling.append(
            {
                "world_size": world_size,
                "execution_seconds": seconds,
                "speedup_over_1gpu": speedup,
                "parallel_efficiency": speedup / world_size,
                "max_cuda_peak_allocated_bytes": record[
                    "max_cuda_peak_allocated_bytes"
                ],
                "correctness": record["correctness"],
            }
        )
    payload = {
        "schema_version": 1,
        "stage": "stage_c_sliced",
        "workload": baseline["workload"],
        "dtype": baseline["dtype"],
        "slice_count": baseline["slice_count"],
        "parameter_count": baseline["parameter_count"],
        "scaling": scaling,
        "eight_gpu_target_speedup": 5.0,
        "eight_gpu_measured_speedup": scaling[-1]["speedup_over_1gpu"],
        "eight_gpu_target_passed": scaling[-1]["speedup_over_1gpu"] >= 5.0,
        "distribution_semantics": "sharded_across_ranks",
        "scalability_claim_allowed": False,
        "claim_boundary": (
            "Single-node A800 development strong-scaling evidence for a "
            "small fitting workload; no single-device capacity failure."
        ),
    }
    arguments.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
