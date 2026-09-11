"""Build auditable non-release evidence from the MPS boundary microbenchmark."""

from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    raw = json.loads(args.input.read_text(encoding="utf-8"))
    world = int(raw["world_size"])
    rank_records = raw["rank_records"]
    local_memory = [
        int(item["packed"]["transport_stats"]["buffer_pool_bytes"])
        for item in rank_records
    ]
    payload = {
        "schema": "flagquantum.mps_boundary_transport_report.v1",
        "benchmark": "mps_boundary_transport",
        "artifact_class": "derived_development_report",
        "benchmark_evidence_class": "non_release_smoke",
        "claim_evidence_type": "development_smoke",
        "non_release_evidence": True,
        "distribution_semantics": "sharded_across_ranks",
        "world_size": world,
        "local_world_size": world,
        "node_count": 1,
        "backend": raw["backend"],
        "device": raw["device"],
        "rank_placement": [
            {
                "rank": rank,
                "local_rank": rank,
                "hostname": socket.gethostname(),
                "device": f"cuda:{rank}",
            }
            for rank in range(world)
        ],
        "local_memory_bytes_by_rank": local_memory,
        "communication_bytes": int(raw["packed_logical_payload_bytes"]),
        "source_commit": args.source_commit,
        "source_tree_dirty": False,
        "legacy_physical_messages": raw["legacy_physical_messages"],
        "packed_physical_messages": raw["packed_physical_messages"],
        "message_count_reduction": raw["message_count_reduction"],
        "logical_byte_growth": raw["logical_byte_growth"],
        "legacy_total_rank_host_wait_seconds": raw[
            "legacy_total_rank_host_wait_seconds"
        ],
        "packed_total_rank_host_wait_seconds": raw[
            "packed_total_rank_host_wait_seconds"
        ],
        "p2p_wait_reduction": raw["p2p_wait_reduction"],
        "packed_rank_mean_seconds": [
            item["packed"]["mean_seconds"] for item in rank_records
        ],
        "packed_buffer_pool_hits_by_rank": [
            item["packed"]["transport_stats"]["buffer_pool_hits"]
            for item in rank_records
        ],
        "shape_mismatch_fault": raw["shape_mismatch_fault"],
        "passed": raw["passed"],
        "scalability_claim_allowed": False,
        "release_gate_allowed": False,
        "scalability_blockers": [
            "microbenchmark_not_end_to_end_training",
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
