"""Build a fail-closed development report from matched capacity probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SOURCE_SCHEMA = "flagquantum.single_node_training_acceptance.v2"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_report(baseline: dict[str, Any], distributed: dict[str, Any]) -> dict[str, Any]:
    if baseline.get("workload_sha256") != distributed.get("workload_sha256"):
        raise ValueError("capacity probes have different workload hashes")
    baseline_ranks = baseline.get("ranks", [])
    distributed_ranks = distributed.get("ranks", [])
    if len(baseline_ranks) != 1:
        raise ValueError("single-device baseline must contain exactly one rank")
    single = baseline_ranks[0]
    if single.get("schema") != SOURCE_SCHEMA:
        raise ValueError("single-device baseline has unsupported schema")
    if not (
        single.get("world_size") == 1
        and single.get("status") == "expected_oom"
        and single.get("expectation_met") is True
        and single.get("single_device_oom_observed") is True
    ):
        raise ValueError("single-device baseline is not a measured expected OOM")
    world_size = len(distributed_ranks)
    if world_size < 2:
        raise ValueError("distributed completion must contain at least two ranks")
    expected_ranks = set(range(world_size))
    actual_ranks = {rank.get("rank") for rank in distributed_ranks}
    if actual_ranks != expected_ranks:
        raise ValueError("distributed completion has missing or duplicate ranks")
    hostnames = {rank.get("hostname") for rank in distributed_ranks}
    if None in hostnames or "" in hostnames:
        raise ValueError("distributed completion is missing rank hostnames")
    node_count = len(hostnames)
    for rank in distributed_ranks:
        training = rank.get("training") or {}
        if not (
            rank.get("schema") == SOURCE_SCHEMA
            and rank.get("world_size") == world_size
            and rank.get("status") == "passed"
            and rank.get("expectation_met") is True
            and rank.get("single_device_oom_observed") is False
            and training.get("distribution_semantics") == "sharded_across_ranks"
            and training.get("completed_steps") == rank["workload"]["steps"]
            and training.get("node_count") == node_count
        ):
            raise ValueError("distributed rank does not prove sharded completion")
    losses = [rank["training"]["losses"] for rank in distributed_ranks]
    parameters = [rank["parameters"] for rank in distributed_ranks]
    if any(value != losses[0] for value in losses[1:]):
        raise ValueError("distributed ranks disagree on loss history")
    if any(value != parameters[0] for value in parameters[1:]):
        raise ValueError("distributed ranks disagree on updated parameters")
    return {
        "schema": "flagquantum.statevector.capacity_report.v1",
        "artifact_classification": "development_capacity_calibration",
        "release_evidence": False,
        "capacity_claim_allowed": False,
        "development_capacity_expansion_observed": True,
        "multi_node_capacity_completion_observed": node_count > 1,
        "workload_sha256": baseline["workload_sha256"],
        "workload": single["workload"],
        "single_device": {
            "status": single["status"],
            "oom_type": single.get("oom_type"),
            "peak_allocated_bytes": single["device"]["peak_allocated_bytes"],
            "peak_reserved_bytes": single["device"]["peak_reserved_bytes"],
            "total_memory_bytes": single["device"]["total_memory_bytes"],
        },
        "distributed": {
            "world_size": world_size,
            "node_count": node_count,
            "rank_placement": [
                {
                    "rank": rank["rank"],
                    "hostname": rank["hostname"],
                    "local_rank": rank["local_rank"],
                }
                for rank in distributed_ranks
            ],
            "completed_steps": distributed_ranks[0]["training"]["completed_steps"],
            "losses": losses[0],
            "updated_parameters": parameters[0],
            "peak_allocated_bytes_by_rank": [
                rank["device"]["peak_allocated_bytes"] for rank in distributed_ranks
            ],
            "peak_reserved_bytes_by_rank": [
                rank["device"]["peak_reserved_bytes"] for rank in distributed_ranks
            ],
            "communication_bytes_by_rank": [
                rank["training"]["communication_bytes"] for rank in distributed_ranks
            ],
        },
        "blockers": [
            "development_threshold_was_not_preregistered_before_measurement",
            "release_correctness_and_recovery_evidence_not_attached",
        ]
        + ([] if node_count > 1 else ["multi_node_capacity_completion_not_measured"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--distributed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(_load(args.baseline), _load(args.distributed))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
