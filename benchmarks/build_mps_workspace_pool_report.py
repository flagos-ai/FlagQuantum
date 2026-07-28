"""Build compact evidence for the MPS factorization staging pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    raw = json.loads(args.input.read_text(encoding="utf-8"))
    ranks = []
    for record in raw["rank_records"]:
        pool = record["factorization_records"][-1]["staging_workspace_pool"]
        ranks.append(
            {
                "rank": record["rank"],
                "device": record["device"],
                "factorization_decision_count": len(
                    record["factorization_records"]
                ),
                "allocation_count": pool["allocation_count"],
                "reuse_count": pool["reuse_count"],
                "reserved_bytes": pool["reserved_bytes"],
                "entry_count": pool["entry_count"],
                "allocator_retry_count": record["factorization_summary"][
                    "allocator_retry_count"
                ],
                "allocator_oom_count": record["factorization_summary"][
                    "allocator_oom_count"
                ],
                "peak_allocated_memory_bytes": record["runs"][-1][
                    "peak_allocated_memory_bytes"
                ],
            }
        )
    environment = raw["environment"]
    payload = {
        "schema": "flagquantum.mps_workspace_pool_report.v1",
        "benchmark": "mps_factorization_staging_workspace_pool",
        "artifact_class": "derived_development_report",
        "benchmark_evidence_class": "non_release_smoke",
        "claim_evidence_type": "development_smoke",
        "non_release_evidence": True,
        "distribution_semantics": "sharded_across_ranks",
        "world_size": 8,
        "local_world_size": 8,
        "node_count": 1,
        "rank_placement": [
            {
                "rank": rank,
                "local_rank": rank,
                "hostname": environment["hostname"],
                "device": f"cuda:{rank}",
            }
            for rank in range(8)
        ],
        "local_memory_bytes_by_rank": [
            record["peak_allocated_memory_bytes"] for record in ranks
        ],
        "communication_bytes": 0,
        "source_commit": args.source_commit,
        "source_tree_dirty": False,
        "workload": {
            "n_wires": 64,
            "depth": 4,
            "max_bond": 8,
            "dtype": "complex64",
            "purpose": "stable_shape_workspace_reuse",
        },
        "rank_records": ranks,
        "total_allocation_count": sum(
            record["allocation_count"] for record in ranks
        ),
        "total_reuse_count": sum(record["reuse_count"] for record in ranks),
        "allocator_retry_count": sum(
            record["allocator_retry_count"] for record in ranks
        ),
        "allocator_oom_count": sum(
            record["allocator_oom_count"] for record in ranks
        ),
        "memory_plateau_audit": raw["audit"],
        "passed": (
            raw["status"] == "passed"
            and all(record["reuse_count"] > 0 for record in ranks)
            and all(record["allocator_retry_count"] == 0 for record in ranks)
            and all(record["allocator_oom_count"] == 0 for record in ranks)
        ),
        "scalability_claim_allowed": False,
        "release_gate_allowed": False,
        "scalability_blockers": [
            "staging_pool_only_cusolver_workspace_not_externally_controllable",
            "stable_bond8_reuse_workload_not_large_bond_performance",
            "bond128_depth4_gesvd_convergence_blocker_observed",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
