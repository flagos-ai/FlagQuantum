"""Build compact non-release evidence from the MPS gradient-bucket runner."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    raw = json.loads(args.input.read_text(encoding="utf-8"))
    records = raw["ranks"]
    world = int(raw["world_size"])
    bucket = records[0]["bucket_summary"]
    payload = {
        "schema": "flagquantum.mps_gradient_bucket_report.v1",
        "benchmark": "mps_gradient_bucket_8gpu",
        "artifact_class": "derived_development_report",
        "benchmark_evidence_class": "non_release_smoke",
        "claim_evidence_type": "development_smoke",
        "non_release_evidence": True,
        "distribution_semantics": "sharded_across_ranks",
        "world_size": world,
        "local_world_size": int(raw["local_world_size"]),
        "node_count": int(raw["node_count"]),
        "rank_placement": [
            {
                "rank": int(record["rank"]),
                "local_rank": int(record["rank"]),
                "hostname": record["hostname"],
                "device": record["device"],
            }
            for record in records
        ],
        "local_memory_bytes_by_rank": [
            int(record["bucket_summary"]["saved_tensor_bytes"])
            for record in records
        ],
        "communication_bytes": int(bucket["gradient_collective_bytes"]),
        "source_commit": args.source_commit,
        "source_tree_dirty": False,
        "raw_artifact_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "parameter_count": int(raw["parameter_count"]),
        "owner_bucket_count": int(raw["owner_bucket_count"]),
        "baseline_collective_count": int(raw["baseline_collective_count"]),
        "bucket_collective_count": int(raw["bucket_collective_count"]),
        "collective_count_reduction": float(raw["collective_count_reduction"]),
        "baseline_mean_rank_seconds": float(raw["baseline_mean_rank_seconds"]),
        "bucket_mean_rank_seconds": float(raw["bucket_mean_rank_seconds"]),
        "observed_backward_speedup": (
            float(raw["baseline_mean_rank_seconds"])
            / float(raw["bucket_mean_rank_seconds"])
        ),
        "loss_error": float(raw["loss_error"]),
        "max_gradient_error": float(raw["max_gradient_error"]),
        "max_sgd_update_error": float(raw["max_sgd_update_error"]),
        "max_adam_update_error": float(raw["max_adam_update_error"]),
        "nonowner_materialized_gradients": int(
            raw["nonowner_materialized_gradients"]
        ),
        "optimizer_resume_state_entries": int(
            raw["optimizer_resume_state_entries"]
        ),
        "passed": (
            raw["collective_count_reduction"] >= 0.9
            and max(
                raw["loss_error"],
                raw["max_gradient_error"],
                raw["max_sgd_update_error"],
                raw["max_adam_update_error"],
            )
            <= 2e-6
            and raw["nonowner_materialized_gradients"] == 0
        ),
        "scalability_claim_allowed": False,
        "release_gate_allowed": False,
        "scalability_blockers": [
            "specialized_gradient_microbenchmark",
            "single_run_unsigned_development_evidence",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
