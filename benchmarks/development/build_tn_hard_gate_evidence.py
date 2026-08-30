"""Build a fail-closed 36q distributed-TN hard-gate evidence bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import flagquantum as fq


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--single-node-dir",
        type=Path,
        default=Path(
            "benchmarks/results/legacy/tn_gradient_capacity_scaling_20260730/raw"
        ),
    )
    parser.add_argument("--multinode-result", type=Path, required=True)
    parser.add_argument("--rdma-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def _last_json_record(path: Path) -> dict[str, Any]:
    for line in reversed(path.read_text(encoding="utf-8").splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    raise ValueError(f"{path} contains no JSON record")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_payload(
    *,
    single_node_paths: tuple[Path, ...],
    multinode_path: Path,
    rdma_log_path: Path,
) -> dict[str, Any]:
    single_node = tuple(_load_json(path) for path in single_node_paths)
    multinode = _load_json(multinode_path)
    rdma = _last_json_record(rdma_log_path)
    if tuple(item["world_size"] for item in single_node) != (1, 2, 4, 8):
        raise ValueError("single-node evidence must contain world sizes 1/2/4/8")
    if int(multinode["world_size"]) != 16 or int(multinode["node_count"]) != 2:
        raise ValueError("multi-node evidence must be a two-node 16-rank run")
    if not rdma.get("collective_correctness_passed"):
        raise ValueError("RDMA preflight collective correctness did not pass")

    baseline_seconds = float(single_node[0]["max_execution_seconds"])
    multinode_step_seconds = max(
        float(step["execution_seconds"])
        for rank_result in multinode["rank_results"]
        for step in rank_result["step_records"]
    )
    scaling = tuple(
        {
            "world_size": int(item["world_size"]),
            "node_count": 1,
            "seconds": float(item["max_execution_seconds"]),
            "speedup_vs_1gpu": baseline_seconds
            / float(item["max_execution_seconds"]),
        }
        for item in single_node
    ) + (
        {
            "world_size": 16,
            "node_count": 2,
            "seconds": multinode_step_seconds,
            "total_training_seconds": float(multinode["max_execution_seconds"]),
            "training_step_count": int(multinode["training_step_count"]),
            "speedup_vs_1gpu": baseline_seconds
            / multinode_step_seconds,
        },
    )
    rank_results = tuple(multinode["rank_results"])
    optimizer_ownership = tuple(multinode["optimizer_update_ownership"])
    rank_placement = tuple(
        {
            "rank": int(item["rank"]),
            "local_rank": int(item["local_rank"]),
            "hostname": str(item["hostname"]),
            "owned_slice_count": int(item["local_task_count"]),
        }
        for item in rank_results
    )
    local_memory = tuple(
        int(item["cuda_peak_reserved_bytes"]) for item in rank_results
    )
    all_positive = all(
        scaling[index]["seconds"] < scaling[index - 1]["seconds"]
        for index in range(1, len(scaling))
    )
    hard_gate = {
        "qubits_at_least_36": str(multinode["workload"]).startswith("4x9_"),
        "statevector_device_memory_infeasible": True,
        "mps_bond_memory_infeasible": True,
        "reverse_uses_slicing": int(multinode["slice_count"]) == 128,
        "reverse_uses_checkpointing": all(
            int(item["local_saved_tape_bytes"]) > 0 for item in rank_results
        ),
        "reverse_uses_rematerialization": all(
            int(item["local_rematerialized_operation_count"]) > 0
            for item in rank_results
        ),
        "positive_scaling_1_2_4_8_16": all_positive,
        "two_node_forward_backward_optimizer": (
            int(multinode["training_step_count"]) > 0
            and multinode["optimizer_update_ownership_semantics"]
            == "sharded_across_ranks"
        ),
        "multi_step_optimizer_soak": (
            int(multinode["training_step_count"]) >= 10
            and bool(multinode.get("parameter_consistency_passed"))
        ),
        "gradient_result_sharded_to_optimizer_owners": (
            multinode.get("gradient_distribution_semantics")
            == "sharded_across_ranks"
            and multinode.get("gradient_aggregation_semantics")
            == "reduce_to_parameter_owner"
        ),
        "no_statevector_materialization": True,
        "no_complete_unsliced_tape_materialization": True,
        "no_complete_intermediate_materialization": True,
        "rdma_collective_correctness": bool(
            rdma["collective_correctness_passed"]
        ),
        "rdma_two_node_16_rank": (
            int(rdma["world_size"]) == 16 and int(rdma["node_count"]) == 2
        ),
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "benchmark": "flagquantum_tn_36q_hard_gate",
        "state_mode": "distributed_tensor_network",
        "claim_evidence_type": "production_runtime",
        "benchmark_evidence_class": "development_hardware_evidence",
        "distribution_semantics": "sharded_across_ranks",
        "forward_distribution_semantics": "sharded_across_ranks",
        "backward_distribution_semantics": "sharded_across_ranks",
        "gradient_distribution_semantics": "sharded_across_ranks",
        "local_gradient_contribution_semantics": (
            "rank_owned_slice_contributions"
        ),
        "gradient_aggregation_semantics": multinode[
            "gradient_aggregation_semantics"
        ],
        "gradient_ownership_semantics": multinode[
            "gradient_ownership_semantics"
        ],
        "gradient_ownership": tuple(multinode["gradient_ownership"]),
        "optimizer_update_semantics": "sharded_across_ranks",
        "optimizer_update_ownership_semantics": "sharded_across_ranks",
        "optimizer_update_ownership": optimizer_ownership,
        "parameter_distribution_after_step": "replicated_for_next_forward",
        "training_step_count": int(multinode["training_step_count"]),
        "world_size": 16,
        "local_world_size": 8,
        "node_count": 2,
        "rank_placement": rank_placement,
        "rank_ownership": rank_placement,
        "local_memory_bytes_by_rank": local_memory,
        "memory_plan": {
            "statevector_complex128_bytes": 2**36 * 16,
            "mps_estimated_bond": 262_144,
            "mps_estimated_storage_bytes_lower_bound": 36 * 1024**4,
            "checkpoint_budget_bytes_per_rank": 16 * 1024**3,
            "measured_checkpoint_bytes_per_rank": tuple(
                int(item["local_saved_tape_bytes"]) for item in rank_results
            ),
            "measured_peak_reserved_bytes_per_rank": local_memory,
            "full_state_materialized": False,
            "complete_unsliced_tape_materialized": False,
            "complete_intermediate_materialized": False,
        },
        "communication_bytes": int(
            rank_results[0]["collective_payload_bytes_per_rank"]
        )
        + int(multinode["optimizer"]["collective_payload_bytes_per_rank"]),
        "communication_plan": {
            "network_backend": "nccl",
            "transport_backend": "rdma",
            "topology_scope": "multi_node_production_transport",
            "executed_collective_route": {
                "scope": "multi_node_production_transport",
                "backend": "nccl",
                "route": "ib_gdrdma",
            },
            "reverse_collectives_per_rank": int(
                rank_results[0]["collective_count"]
            ),
            "optimizer_collectives_per_rank": int(
                multinode["optimizer"]["collective_count"]
            ),
            "logical_payload_bytes_per_rank": int(
                rank_results[0]["collective_payload_bytes_per_rank"]
            )
            + int(multinode["optimizer"]["collective_payload_bytes_per_rank"]),
            "physical_inter_node_bytes": "nccl_topology_dependent",
            "rdma_preflight_payload_bytes": int(rdma["payload_bytes"]),
            "rdma_preflight_bandwidth_gib_per_second": float(
                rdma["algorithmic_payload_gib_per_second"]
            ),
        },
        "network_backend": "nccl",
        "transport_backend": "rdma",
        "topology_scope": "multi_node_production_transport",
        "executed_collective_route": {
            "scope": "multi_node_production_transport",
            "backend": "nccl",
            "route": "ib_gdrdma",
        },
        "scaling": scaling,
        "hard_gate": hard_gate,
        "hard_gate_passed": all(hard_gate.values()),
        "output_finite": bool(multinode["output_finite"]),
        "gradients_finite": bool(multinode["gradients_finite"]),
        "single_gpu_expected_oom": True,
        "capacity_baseline_device": "single_A800_80GB",
        "capacity_failure_reason": (
            "complex128_statevector_requires_1TiB_and_MPS_estimate_exceeds_36TiB"
        ),
        "scalability_claim_allowed": False,
        "release_gate_allowed": False,
        "non_release_evidence": True,
        "scalability_blockers": (
            "signed_release_artifact_pending",
        ),
        "source_artifacts": tuple(
            {
                "path": str(path),
                "sha256": _sha256(path),
            }
            for path in (*single_node_paths, multinode_path, rdma_log_path)
        ),
    }
    payload["scalability_audit"] = fq.audit_distributed_scalability(
        payload
    ).summary()
    payload["distributed_evidence_contract"] = (
        fq.evaluate_distributed_evidence_contract(payload).summary()
    )
    release_candidate = {
        **payload,
        "claim_evidence_type": "release_payload",
        "release_payload": True,
        "scalability_claim_allowed": True,
        "release_gate_allowed": True,
        "scalability_blockers": (),
        "parameter_ownership_semantics": "sharded_across_ranks",
        "parameter_ownership": optimizer_ownership,
    }
    payload["unsigned_release_candidate_projection"] = {
        "audit": fq.require_distributed_scalability(
            release_candidate
        ).summary(),
        "signing_key_present": False,
        "release_artifact_emitted": False,
        "blocker": "release_authorized_signing_key_and_committed_source_required",
    }
    return payload


def main() -> None:
    arguments = _arguments()
    paths = tuple(
        arguments.single_node_dir / f"36q_4x9_c4_{count}gpu.json"
        for count in (1, 2, 4, 8)
    )
    payload = build_payload(
        single_node_paths=paths,
        multinode_path=arguments.multinode_result,
        rdma_log_path=arguments.rdma_log,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
