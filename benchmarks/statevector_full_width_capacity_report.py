#!/usr/bin/env python3
"""Build a fail-closed report from matched full-width capacity evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

MATCHED_FIELDS = (
    "n_wires",
    "layers",
    "seed",
    "dtype",
    "name",
    "gate_count",
    "active_wire_count",
    "parameter_count",
    "entanglement",
)


def is_sharded_capacity_completion(distributed: dict[str, object]) -> bool:
    """Return whether one distributed run completed with sharded state lifecycle."""

    protocol = distributed.get("benchmark_protocol", {})
    correctness = distributed.get("correctness", {})
    return bool(
        distributed.get("node_count", 0) >= 2
        and distributed.get("world_size", 0) > 1
        and distributed.get("distribution_semantics") == "sharded_across_ranks"
        and distributed.get("forward_distribution_semantics")
        == "sharded_across_ranks"
        and distributed.get("backward_distribution_semantics")
        == "sharded_across_ranks"
        and isinstance(protocol, dict)
        and protocol.get("full_state_materialization") is False
        and isinstance(correctness, dict)
        and correctness.get("passed")
        and not distributed.get("validation_blockers")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-gpu-oom", type=Path, required=True)
    parser.add_argument("--distributed-completion", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()

    oom = json.loads(args.single_gpu_oom.read_text(encoding="utf-8"))
    distributed = json.loads(
        args.distributed_completion.read_text(encoding="utf-8")
    )
    oom_workload = oom["workload"]
    distributed_workload = distributed["workload"]
    mismatches = {
        field: (oom_workload.get(field), distributed_workload.get(field))
        for field in MATCHED_FIELDS
        if oom_workload.get(field) != distributed_workload.get(field)
    }
    if mismatches:
        raise ValueError(f"capacity workload mismatch: {mismatches}")
    matched_workload = {field: oom_workload[field] for field in MATCHED_FIELDS}
    matched_hash = hashlib.sha256(
        json.dumps(matched_workload, sort_keys=True).encode()
    ).hexdigest()
    single_oom = bool(
        oom.get("expectation_met") and oom.get("single_device_oom_observed")
    )
    multi_completed = is_sharded_capacity_completion(distributed)
    payload = {
        "schema_version": "flagquantum.statevector.full_width_capacity_report.v1",
        "benchmark": "statevector_full_width_capacity_report",
        "artifact_class": "measured_development_capacity_report",
        "benchmark_evidence_class": "comparison_non_release",
        "claim_evidence_type": "development_smoke",
        "non_release_evidence": True,
        "scalability_claim_allowed": False,
        "scalability_blockers": [
            "comparison benchmark is not release-certified scalability evidence"
        ],
        "matched_workload": matched_workload,
        "matched_workload_sha256": matched_hash,
        "single_gpu_oom_observed": single_oom,
        "distributed_capacity_completion_observed": multi_completed,
        "capacity_expansion_demonstrated": single_oom and multi_completed,
        "distribution_semantics": distributed.get("distribution_semantics"),
        "forward_distribution_semantics": distributed.get(
            "forward_distribution_semantics"
        ),
        "backward_distribution_semantics": distributed.get(
            "backward_distribution_semantics"
        ),
        "world_size": distributed.get("world_size"),
        "local_world_size": distributed.get("local_world_size"),
        "node_count": distributed.get("node_count"),
        "rank_placement": [
            {
                key: item[key]
                for key in ("rank", "hostname", "local_rank")
                if key in item
            }
            for item in distributed.get("rank_placement", [])
            if isinstance(item, dict)
        ],
        "local_memory_bytes_by_rank": distributed.get(
            "local_memory_bytes_by_rank"
        ),
        "communication_bytes": distributed.get("communication_bytes"),
        "communication_bytes_scope": distributed.get("communication_bytes_scope"),
        "intra_node_communication_bytes": distributed.get(
            "intra_node_communication_bytes"
        ),
        "inter_node_communication_bytes": distributed.get(
            "inter_node_communication_bytes"
        ),
        "full_state_materialization": distributed.get(
            "benchmark_protocol", {}
        ).get("full_state_materialization"),
        "single_gpu_source": str(args.single_gpu_oom),
        "distributed_source": str(args.distributed_completion),
        "distributed_world_size": distributed.get("world_size"),
        "distributed_node_count": distributed.get("node_count"),
        "distributed_peak_memory_bytes_per_rank": distributed.get("end_to_end", {}).get(
            "peak_memory_bytes_max"
        ),
        "distributed_end_to_end_median_seconds": distributed.get(
            "end_to_end", {}
        ).get("median_seconds"),
        "distributed_backward_median_seconds": distributed.get("backward", {}).get(
            "median_seconds"
        ),
        "release_gate_allowed": False,
        "blockers": (
            []
            if single_oom and multi_completed
            else ["matched_capacity_expansion_not_demonstrated"]
        ),
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, sort_keys=True), flush=True)
    if not payload["capacity_expansion_demonstrated"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
