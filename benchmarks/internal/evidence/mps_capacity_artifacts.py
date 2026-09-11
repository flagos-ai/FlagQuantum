"""Aggregate measured ISSUE-052 timing records without promoting failed gates."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = {
        world: json.loads(
            (args.input_dir / f"issue052_mps_training_{world}gpu.json").read_text()
        )
        for world in (1, 2, 4, 8)
    }
    baseline = records[1]
    identity_fields = ("family", "scaling_mode")
    identity_complete = all(
        all(record.get(field) is not None for field in identity_fields)
        and record.get("workload", {}).get("seed") is not None
        for record in records.values()
    )
    rows = []
    for world, record in records.items():
        speedup = baseline["mean_seconds"] / record["mean_seconds"]
        local_low, local_high = baseline["confidence_interval_95_seconds"]
        remote_low, remote_high = record["confidence_interval_95_seconds"]
        speedup_low = local_low / remote_high
        speedup_high = local_high / remote_low
        rows.append(
            {
                "world_size": world,
                "mean_seconds": record["mean_seconds"],
                "confidence_interval_95_seconds": record[
                    "confidence_interval_95_seconds"
                ],
                "speedup_mean": speedup,
                "speedup_ci_low": speedup_low,
                "speedup_ci_high": speedup_high,
                "scaling_efficiency": speedup / world,
                "peak_memory_bytes": record["peak_memory_bytes"],
                "memory_growth_bytes": record["memory_growth_bytes"],
                "samples": record["repetitions"],
            }
        )
    best = max(
        (row for row in rows if row["world_size"] > 1),
        key=lambda row: row["speedup_mean"],
    )
    matched = identity_complete and all(
        record["workload_sha256"] == baseline["workload_sha256"]
        for record in records.values()
    )
    passed = matched and best["speedup_ci_low"] > 1.0
    sources = []
    for world in (1, 2, 4, 8):
        path = args.input_dir / f"issue052_mps_training_{world}gpu.json"
        sources.append(
            {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        )
    payload = {
        "schema": "flagquantum.issue052.mps_crossover.v1",
        "gate": "performance",
        "evidence_source": "measured_runtime",
        "artifact_classification": "measured_accelerator_benchmark",
        "device": "NVIDIA A100-SXM4-40GB",
        "workload_sha256": baseline["workload_sha256"],
        "matched_workload": matched,
        "workload_identity_complete": identity_complete,
        "warmup": baseline["warmup"],
        "repetitions": baseline["repetitions"],
        "measurements": rows,
        "best_distributed_world_size": best["world_size"],
        "best_distributed_speedup": best["speedup_mean"],
        "best_distributed_speedup_ci": [
            best["speedup_ci_low"],
            best["speedup_ci_high"],
        ],
        "performance_gate": {
            "passed": passed,
            "threshold": "speedup_ci_low > 1.0",
            "reason": (
                "confidence_interval_excludes_no_improvement"
                if passed
                else "workload_identity_incomplete"
                if not identity_complete
                else "no_distributed_measurement_outperformed_single_gpu"
            ),
        },
        "source_artifacts": sources,
        "scalability_claim_allowed": False,
        "release_gate_allowed": False,
        "blockers": (
            ("mps_capacity_training_gate_pending",)
            if passed
            else ("mps_distributed_speedup_not_demonstrated",)
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["performance_gate"]))


if __name__ == "__main__":
    main()
