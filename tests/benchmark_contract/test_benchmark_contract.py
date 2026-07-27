import json
from copy import deepcopy
from pathlib import Path

import pytest

import flagquantum as fq
from benchmarks.audit_results import _json_files, audit_paths

pytestmark = [pytest.mark.benchmark_contract, pytest.mark.release_gate]

RESULTS_ROOT = Path("benchmarks/results")
SCALABILITY_ROOT = RESULTS_ROOT / "scalability"
NON_RELEASE_DIRS = (
    RESULTS_ROOT / "local",
    RESULTS_ROOT / "comparison",
    RESULTS_ROOT / "smoke",
)
CAPACITY_MODEL_FIXTURE = (
    Path("benchmarks/fixtures/capacity_models")
    / "statevector_capacity_model_8rank.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_benchmark_result_payloads_expose_contract_fields():
    payload_paths = _json_files(RESULTS_ROOT)

    assert payload_paths
    for path in payload_paths:
        payload = _load(path)
        assert "benchmark" in payload, path
        assert "distribution_semantics" in payload, path
        assert "scalability_claim_allowed" in payload, path
        assert "release_gate_allowed" in payload, path
        assert "scalability_blockers" in payload, path
        if SCALABILITY_ROOT not in path.parents:
            assert payload["non_release_evidence"] is True, path
            assert payload["scalability_claim_allowed"] is False, path
            assert payload["release_gate_allowed"] is False, path
            assert payload["benchmark_evidence_class"] in {
                "local_non_release",
                "comparison_non_release",
                "non_release_smoke",
            }, path


def _statevector_release_candidate() -> dict:
    payload = _load(CAPACITY_MODEL_FIXTURE)
    payload.update(
        {
            "claim_evidence_type": "release_payload",
            "release_payload": True,
            "payload_kind": "release_gate_capacity_payload",
            "scalability_claim_allowed": True,
            "release_gate_allowed": True,
            "scalability_blockers": [],
            "claimable_production_training": True,
        }
    )
    return payload


def test_cpu_labelled_capacity_payload_is_quarantined_from_release_scan():
    payload = _load(CAPACITY_MODEL_FIXTURE)
    summary = audit_paths([CAPACITY_MODEL_FIXTURE], require_scalability=True)

    assert payload["artifact_class"] == "plan"
    assert payload["benchmark_evidence_class"] == "capacity_model_fixture"
    assert payload["claim_evidence_type"] == "plan_preflight"
    assert payload["release_gate_allowed"] is False
    assert summary["file_count"] == 1
    assert summary["claimable_count"] == 0
    assert summary["invalid_count"] == 1
    assert summary["claimable_paths"] == ()
    assert summary["records"][0]["status"] == "provenance_rejected"


def test_scalability_release_payload_exposes_complete_release_evidence_bundle():
    payload = _load(CAPACITY_MODEL_FIXTURE)
    required_fields = {
        "claim_evidence_type",
        "distribution_semantics",
        "forward_distribution_semantics",
        "backward_distribution_semantics",
        "gradient_distribution_semantics",
        "scalability_claim_allowed",
        "release_gate_allowed",
        "world_size",
        "local_world_size",
        "node_count",
        "rank_placement",
        "rank_shards",
        "local_memory_bytes_by_rank",
        "memory_plan",
        "communication_bytes",
        "intra_node_communication_bytes",
        "inter_node_communication_bytes",
        "communication_plan",
        "single_gpu_expected_oom",
        "capacity_baseline_device",
        "capacity_failure_reason",
        "parameter_ownership",
        "gradient_ownership",
        "optimizer_update_ownership",
        "optimizer_update_semantics",
        "training_step_count",
    }

    missing = required_fields.difference(payload)
    assert not missing
    assert payload["distribution_semantics"] == "sharded_across_ranks"
    assert payload["forward_distribution_semantics"] == "sharded_across_ranks"
    assert payload["backward_distribution_semantics"] == "sharded_across_ranks"
    assert payload["gradient_distribution_semantics"] == "sharded_across_ranks"
    assert payload["world_size"] > 1
    assert payload["local_world_size"] == payload["world_size"]
    assert payload["node_count"] >= 1
    assert len(payload["rank_placement"]) == payload["world_size"]
    assert len(payload["rank_shards"]) == payload["world_size"]
    assert len(payload["local_memory_bytes_by_rank"]) == payload["world_size"]
    assert (
        sum(shard["owned_amplitudes"] for shard in payload["rank_shards"])
        == payload["state_amplitudes"]
    )
    assert payload["memory_plan"]["state_bytes_total"] == payload["total_state_bytes"]
    assert (
        payload["memory_plan"]["state_bytes_per_rank"]
        == payload["per_rank_state_bytes"]
    )
    assert payload["memory_plan"]["communication_buffer_bytes_per_rank"] > 0
    assert payload["communication_bytes"] > 0
    assert (
        payload["communication_plan"]["estimated_transfer_bytes"]
        == payload["communication_bytes"]
    )
    assert {"pair_exchange", "all_to_all"}.issubset(
        payload["communication_plan"]["patterns"]
    )
    assert len(payload["parameter_ownership"]) == payload["world_size"]
    assert len(payload["gradient_ownership"]) == payload["world_size"]
    assert len(payload["optimizer_update_ownership"]) == payload["world_size"]
    assert (
        sum(item["owned_parameter_count"] for item in payload["parameter_ownership"])
        == payload["parameter_count"]
    )
    assert (
        sum(item["owned_parameter_count"] for item in payload["gradient_ownership"])
        == payload["parameter_count"]
    )
    assert (
        sum(
            item["owned_parameter_count"]
            for item in payload["optimizer_update_ownership"]
        )
        == payload["parameter_count"]
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("single_gpu_expected_oom", False, "single_gpu_expected_oom=True"),
        ("capacity_baseline_device", "", "capacity_baseline_device"),
        ("capacity_failure_reason", "", "capacity_failure_reason"),
        (
            "optimizer_update_semantics",
            "replicated_per_rank",
            "optimizer_update_semantics",
        ),
        ("training_step_count", 0, "training_step_count > 0"),
        ("release_payload", False, "release_payload evidence must be explicit"),
    ],
)
def test_scalability_release_payload_fails_closed_when_required_evidence_is_removed(
    field, value, message
):
    payload = deepcopy(_statevector_release_candidate())
    payload[field] = value

    with pytest.raises(fq.DistributedScalabilityError) as excinfo:
        fq.require_distributed_scalability(payload)

    assert message in str(excinfo.value)


def test_non_release_payloads_do_not_pass_scalability_release_gate():
    non_release_paths = []
    for directory in NON_RELEASE_DIRS:
        non_release_paths.extend(_json_files(directory))

    assert non_release_paths
    for path in non_release_paths:
        payload = _load(path)
        audit = fq.audit_distributed_scalability(payload)
        assert audit.valid, path
        assert not audit.release_gate_allowed, path
        assert payload["scalability_claim_allowed"] is False, path
        with pytest.raises(fq.DistributedScalabilityError):
            fq.require_distributed_scalability(payload)


def test_require_scalability_scanner_rejects_non_release_payloads_when_required():
    local_payload = (
        RESULTS_ROOT / "local" / "flagquantum_1000q_structured_mps_cpu_jax.json"
    )
    smoke_payload = (
        RESULTS_ROOT / "smoke" / "statevector_pmap_all_to_all_cpu4_smoke.json"
    )
    summary = audit_paths([local_payload, smoke_payload], require_scalability=True)

    assert summary["file_count"] == 2
    assert summary["claimable_count"] == 0
    assert summary["invalid_count"] == 2
    assert summary["claimable_paths"] == ()
    assert set(summary["invalid_paths"]) == {str(local_payload), str(smoke_payload)}
    for record in summary["records"]:
        assert record["scalability_audit"]["release_gate_allowed"] is False
        assert record["scalability_audit"]["valid"] is False


def test_benchmark_audit_scanner_contract_excludes_generated_summaries():
    root_paths = _json_files(RESULTS_ROOT)
    root_summary = audit_paths(root_paths)
    release_paths = _json_files(SCALABILITY_ROOT)
    release_summary = audit_paths(release_paths, require_scalability=True)

    assert all(not path.name.endswith("audit_summary.json") for path in root_paths)
    assert root_summary["file_count"] == len(root_paths)
    assert root_summary["invalid_count"] == 0
    assert root_summary["claimable_count"] == len(root_summary["claimable_paths"])
    assert release_summary["file_count"] == len(release_paths)
    assert release_summary["claimable_count"] == 0
    assert release_summary["invalid_count"] == len(release_paths)


def _prepared_mps_release_payload() -> dict:
    parameter_ownership = (
        {
            "rank": 0,
            "owner_rank": 0,
            "parameter_id": "theta_0",
            "parameter_indices": (0,),
            "writeback_route": "rank_local_parameter_owner_writeback",
        },
        {
            "rank": 1,
            "owner_rank": 1,
            "parameter_id": "theta_1",
            "parameter_indices": (1,),
            "writeback_route": "rank_local_parameter_owner_writeback",
        },
    )
    gradient_ownership = tuple(
        {
            **item,
            "gradient_owner_rank": item["rank"],
        }
        for item in parameter_ownership
    )
    optimizer_ownership = tuple(
        {
            **item,
            "update_owner_rank": item["rank"],
        }
        for item in parameter_ownership
    )
    rank_memory = (
        {
            "rank": 0,
            "forward_tensor_bytes": 128,
            "backward_adjoint_bytes": 128,
            "boundary_gradient_buffer_bytes": 32,
            "canonicalization_temporary_bytes": 64,
            "truncation_temporary_bytes": 0,
            "estimated_peak_backward_bytes": 352,
        },
        {
            "rank": 1,
            "forward_tensor_bytes": 128,
            "backward_adjoint_bytes": 128,
            "boundary_gradient_buffer_bytes": 32,
            "canonicalization_temporary_bytes": 64,
            "truncation_temporary_bytes": 0,
            "estimated_peak_backward_bytes": 352,
        },
    )
    boundary_edges = (
        {
            "boundary_edge_id": "edge_0:1-2",
            "left_rank": 0,
            "right_rank": 1,
            "communication_bytes": 64,
            "communication_primitive": "xla_collective_permute",
            "topology_tier": "intra_node",
            "topology_route": "single_node_xla",
            "execution_status": "executed",
        },
    )
    mps_memory_plan = {
        "status": "production_measured",
        "rank_memory": rank_memory,
        "forward_tensor_bytes_by_rank": (128, 128),
        "backward_adjoint_bytes_by_rank": (128, 128),
        "boundary_gradient_buffer_bytes_by_rank": (32, 32),
        "canonicalization_temporary_bytes_by_rank": (64, 64),
        "truncation_temporary_bytes_by_rank": (0, 0),
        "per_rank_peak_bytes": (352, 352),
        "local_memory_bytes_by_rank": (352, 352),
        "communication_buffer_bytes": 64,
    }
    mps_communication_plan = {
        "status": "production_executed",
        "boundary_edge_count": 1,
        "boundary_edges": boundary_edges,
        "communication_bytes": 64,
        "estimated_transfer_bytes": 64,
        "intra_node_communication_bytes": 64,
        "inter_node_communication_bytes": 0,
        "network_backend": "xla",
        "topology_scope": "single_node_executed_collective",
    }
    return {
        "benchmark": "phase5_mps_release_contract_synthetic_test_only",
        "state_mode": "jax_sharded_mps",
        "claim_evidence_type": "release_payload",
        "release_payload": True,
        "distribution_semantics": "sharded_across_ranks",
        "mps_forward_distribution_semantics": "sharded_across_ranks",
        "mps_backward_distribution_semantics": "sharded_across_ranks",
        "gradient_distribution_semantics": "sharded_across_ranks",
        "scalability_claim_allowed": True,
        "world_size": 2,
        "local_world_size": 2,
        "node_count": 1,
        "rank_placement": (
            {"rank": 0, "node_id": 0, "device_id": "accelerator:0"},
            {"rank": 1, "node_id": 0, "device_id": "accelerator:1"},
        ),
        "rank_ownership": (
            {"rank": 0, "wires": (0, 1)},
            {"rank": 1, "wires": (2, 3)},
        ),
        "site_shard_ownership": (
            {"rank": 0, "wires": (0, 1)},
            {"rank": 1, "wires": (2, 3)},
        ),
        "bond_shard_ownership": (
            {
                "left_rank": 0,
                "right_rank": 1,
                "left_wire": 1,
                "right_wire": 2,
            },
        ),
        "backward_execution": "jax_pmap_mps_boundary_adjoint_backward",
        "boundary_adjoint_exchange": {
            "status": "production_executed",
            "communication_backend": "xla_collective_permute",
        },
        "boundary_gradient_ownership": (
            {
                "left_rank": 0,
                "right_rank": 1,
                "ownership_semantics": "adjacent_rank_boundary_adjoint",
            },
        ),
        "boundary_gradient_routes": boundary_edges,
        "parameter_gradient_ownership": gradient_ownership,
        "canonicalization_backward_strategy": {
            "status": "production_executed",
            "strategy": "cross_shard_canonicalization_pullback",
        },
        "truncation_gradient_metadata": {"status": "not_required"},
        "mps_backward_memory_plan": mps_memory_plan,
        "mps_backward_communication_plan": mps_communication_plan,
        "memory_plan": {
            "local_memory_bytes_by_rank": (352, 352),
            "communication_buffer_bytes": 64,
        },
        "local_memory_bytes_by_rank": (352, 352),
        "communication_plan": mps_communication_plan,
        "communication_bytes": 64,
        "parameter_ownership_semantics": "sharded_across_ranks",
        "gradient_ownership_semantics": "sharded_across_ranks",
        "optimizer_update_semantics": "sharded_across_ranks",
        "optimizer_update_ownership_semantics": "sharded_across_ranks",
        "parameter_ownership": parameter_ownership,
        "gradient_ownership": gradient_ownership,
        "optimizer_update_ownership": optimizer_ownership,
        "training_step_count": 1,
        "single_gpu_expected_oom": True,
        "capacity_baseline_device": "single_accelerator_24gb",
        "capacity_failure_reason": "single_accelerator_memory_budget_exceeded",
        "fallback_semantics": "none",
        "scalability_blockers": (),
        "gradient_blockers": (),
        "blockers": (),
    }


def test_mps_release_payload_contract_accepts_complete_synthetic_schema():
    payload = _prepared_mps_release_payload()

    audit = fq.require_distributed_scalability(payload)

    assert audit.valid
    assert audit.release_gate_allowed
    assert audit.claim_evidence_type == "release_payload"


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("distribution", "distribution_semantics='sharded_across_ranks'"),
        ("forward", "MPS release gate requires sharded forward evidence"),
        ("backward", "MPS release gate requires sharded backward evidence"),
        (
            "backward_execution",
            "MPS release gate requires executed production backward evidence",
        ),
        ("site_ownership", "MPS release gate requires site shard ownership"),
        ("bond_ownership", "MPS release gate requires bond shard ownership"),
        (
            "boundary_ownership",
            "MPS release gate requires boundary-gradient ownership",
        ),
        (
            "boundary_exchange",
            "MPS release gate requires executed boundary-adjoint exchange evidence",
        ),
        (
            "parameter_gradient",
            "MPS release gate requires parameter-gradient ownership",
        ),
        (
            "optimizer_ownership",
            "optimizer_update_ownership_semantics='sharded_across_ranks'",
        ),
        (
            "memory",
            "MPS release gate requires production-measured per-rank backward memory evidence",
        ),
        (
            "communication",
            "MPS release gate requires production-executed backward communication evidence",
        ),
        ("capacity_oom", "single_gpu_expected_oom=True"),
        ("capacity_device", "capacity_baseline_device"),
        ("capacity_reason", "capacity_failure_reason"),
        ("training_step", "training_step_count > 0"),
        ("fallback", "MPS release gate rejects local replay"),
        ("blockers", "MPS release gate requires blockers to be empty"),
        (
            "claim_evidence_type",
            "claim_evidence_type='production_training_benchmark' or 'release_payload'",
        ),
        ("release_payload_flag", "release_payload evidence must be explicit"),
        ("scalability_claim_allowed", "scalability_claim_allowed=True"),
    ),
)
def test_mps_release_payload_contract_fails_closed(case, message):
    payload = deepcopy(_prepared_mps_release_payload())
    if case == "distribution":
        payload["distribution_semantics"] = "replicated_per_rank"
    elif case == "forward":
        payload["mps_forward_distribution_semantics"] = "incomplete"
    elif case == "backward":
        payload["mps_backward_distribution_semantics"] = "incomplete"
    elif case == "backward_execution":
        payload["backward_execution"] = "planned_not_executed"
    elif case == "site_ownership":
        payload["site_shard_ownership"] = ()
    elif case == "bond_ownership":
        payload["bond_shard_ownership"] = ()
    elif case == "boundary_ownership":
        payload["boundary_gradient_ownership"] = ()
    elif case == "boundary_exchange":
        payload["boundary_adjoint_exchange"]["status"] = "planned"
    elif case == "parameter_gradient":
        payload["parameter_gradient_ownership"] = ()
    elif case == "optimizer_ownership":
        payload["optimizer_update_ownership"] = ()
    elif case == "memory":
        payload["mps_backward_memory_plan"]["status"] = "complete_estimate"
    elif case == "communication":
        payload["mps_backward_communication_plan"]["status"] = "planned_not_executed"
    elif case == "capacity_oom":
        payload["single_gpu_expected_oom"] = False
    elif case == "capacity_device":
        payload["capacity_baseline_device"] = ""
    elif case == "capacity_reason":
        payload["capacity_failure_reason"] = ""
    elif case == "training_step":
        payload["training_step_count"] = 0
    elif case == "fallback":
        payload["fallback_semantics"] = "full_local_mps_replay"
    elif case == "blockers":
        payload["blockers"] = ("production_backward_pending",)
    elif case == "claim_evidence_type":
        payload["claim_evidence_type"] = "production_runtime"
    elif case == "release_payload_flag":
        payload["release_payload"] = False
    elif case == "scalability_claim_allowed":
        payload["scalability_claim_allowed"] = False
    else:
        raise AssertionError(case)

    with pytest.raises(fq.DistributedScalabilityError) as excinfo:
        fq.require_distributed_scalability(payload)

    assert message in str(excinfo.value)


def test_mps_release_contract_preparation_does_not_promote_payload():
    promoted_payloads = (_load(path) for path in _json_files(SCALABILITY_ROOT))

    assert all(
        str(payload.get("state_mode", payload.get("mode", ""))).lower()
        not in {"mps", "distributed_mps", "jax_sharded_mps"}
        for payload in promoted_payloads
    )
