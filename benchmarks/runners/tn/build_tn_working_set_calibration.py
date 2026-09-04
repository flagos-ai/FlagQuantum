"""Build a versioned TN working-set calibration from measured CUDA records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flagquantum.runtime.planner.tn_calibration import (
    build_tn_working_set_calibration,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--accelerator-name", required=True)
    parser.add_argument("--complex-bytes", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--topology-class", required=True)
    parser.add_argument("--safety-margin", type=float, default=1.1)
    arguments = parser.parse_args()

    measurements = []
    sources = []
    for path in arguments.input:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not payload.get("completed") and payload.get("status") != "measured_success":
            raise ValueError(f"incomplete TN calibration input: {path}")
        execution_preflight = payload.get("execution_summary", {}).get(
            "working_set_preflight", {}
        )
        predicted_tensor_bytes = payload.get(
            "planned_peak_bytes",
            execution_preflight.get("tensor_peak_bytes"),
        )
        allocated_bytes = payload.get(
            "cuda_peak_allocated_bytes",
            payload.get("peak_memory_bytes_max"),
        )
        reserved_bytes = payload.get(
            "cuda_peak_reserved_bytes",
            payload.get("peak_reserved_memory_bytes_max"),
        )
        if (
            predicted_tensor_bytes is None
            or allocated_bytes is None
            or reserved_bytes is None
        ):
            raise ValueError(f"incomplete TN memory fields: {path}")
        rank_records = payload.get("rank_records")
        if rank_records:
            measurements.extend(
                {
                    "predicted_working_set_bytes": int(predicted_tensor_bytes),
                    "cuda_peak_allocated_bytes": int(record["peak_memory_bytes"]),
                    "cuda_peak_reserved_bytes": int(
                        record["peak_reserved_memory_bytes"]
                    ),
                }
                for record in rank_records
            )
        else:
            measurements.append(
                {
                    "predicted_working_set_bytes": int(predicted_tensor_bytes),
                    "cuda_peak_allocated_bytes": int(allocated_bytes),
                    "cuda_peak_reserved_bytes": int(reserved_bytes),
                }
            )
        sources.append(str(path))
    calibration = build_tn_working_set_calibration(
        measurements,
        accelerator_name=arguments.accelerator_name,
        complex_bytes=arguments.complex_bytes,
        world_size=arguments.world_size,
        topology_class=arguments.topology_class,
        safety_margin=arguments.safety_margin,
    )
    output = {
        **calibration.summary(),
        "sources": sources,
        "claim_evidence_type": "development_hardware_calibration",
        "scalability_claim_allowed": False,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
