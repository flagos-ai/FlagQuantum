import pytest

from flagquantum.runtime.audit import DistributedScalabilityError
from flagquantum.runtime.audit.release_policy import require_distributed_scalability
from flagquantum.runtime.executors.jax.mps.backward import (
    _execute_minimal_mps_sharded_backward,
)
from flagquantum.runtime.executors.jax.mps.canonicalization import (
    _execute_minimal_mps_sharded_optimizer_step,
)
from flagquantum.runtime.executors.jax.mps.evidence import (
    _collect_jax_mps_accelerator_backward_evidence,
)

pytestmark = [
    pytest.mark.distributed,
    pytest.mark.distributed_accel,
    pytest.mark.gpu,
]


def _require_mps_accelerators():
    jax = pytest.importorskip(
        "jax",
        reason="requires JAX accelerator runtime",
    )
    devices = tuple(
        device
        for device in jax.local_devices()
        if str(getattr(device, "platform", "unknown")).lower() != "cpu"
    )
    if len(devices) < 2:
        pytest.skip(
            "requires at least two local non-CPU JAX accelerator devices "
            "for MPS backward evidence"
        )
    if int(jax.process_count()) != 1:
        pytest.skip(
            "MPS accelerator evidence is single-node only; "
            "multi-node evidence is separate"
        )
    return devices


def test_mps_accelerator_backward_evidence_uses_shared_fail_closed_contract():
    devices = _require_mps_accelerators()

    summary = _collect_jax_mps_accelerator_backward_evidence(world_size=len(devices))
    evidence = summary["accelerator_backward_evidence"]
    contract = summary["distributed_evidence_contract"]
    gate = summary["mps_backward_readiness_gate"]

    assert evidence["status"] == "measured_accelerator_probe_observed"
    assert evidence["artifact_classification"] == "measured_accelerator_probe"
    assert evidence["accelerator_backed"] is True
    assert evidence["world_size"] == len(devices)
    assert evidence["local_world_size"] == len(devices)
    assert evidence["node_count"] == 1
    assert len(evidence["rank_to_device"]) == len(devices)
    assert all(item["platform"] != "cpu" for item in evidence["device_placement"])
    assert evidence["boundary_communication_backend"] == "xla_pmap_collective_permute"
    assert evidence["backward_execution_time_seconds"] > 0
    assert evidence["communication_time_seconds"] > 0
    assert evidence["gradient_ownership_semantics"] == "sharded_across_ranks"
    memory = evidence["accelerator_memory_evidence"]
    assert memory["status"] == "probe_array_footprint_estimate"
    assert memory["device_allocator_peak_measured"] is False
    assert contract["contract_version"] == "distributed_evidence_contract_v1"
    assert contract["claim_evidence_type"] == "development_smoke"
    assert contract["claimable_production_training"] is False
    assert gate["production_training_claimable"] is False
    assert gate["fail_closed"] is True
    assert (
        "mps_accelerator_probe_not_full_sharded_backward_executor" in gate["blockers"]
    )
    assert summary["scalability_claim_allowed"] is False
    with pytest.raises(DistributedScalabilityError):
        require_distributed_scalability(summary)


def test_minimal_mps_sharded_backward_skeleton_runs_on_accelerators():
    _require_mps_accelerators()

    summary = _execute_minimal_mps_sharded_backward(
        (0.2, -0.4),
        execution_backend="accelerator",
    )

    assert summary["mps_backward_execution"] == "executed"
    assert summary["claim_evidence_type"] == "production_runtime"
    assert (
        summary["execution_classification"]
        == "accelerator_backed_production_runtime_evidence"
    )
    assert summary["boundary_communication_backend"] == ("xla_pmap_collective_permute")
    assert summary["full_mps_reconstruction_count"] == 0
    assert summary["replicated_mps_autograd"] is False
    assert summary["statevector_fallback"] is False
    assert summary["scalability_claim_allowed"] is False
    assert "mps_minimal_backward_skeleton_not_full_executor" in summary["blockers"]


def test_minimal_mps_sharded_optimizer_step_runs_on_owner_accelerators():
    _require_mps_accelerators()

    summary = _execute_minimal_mps_sharded_optimizer_step(
        (0.2, -0.4),
        learning_rate=0.05,
        execution_backend="accelerator",
    )

    assert summary["optimizer_backend"] == "jax_pmap_owner_local_sgd"
    assert summary["optimizer_execution_status"] == "production_executed"
    assert summary["optimizer_execution_scope"] == "single_node_accelerator"
    assert summary["training_step_count"] == 1
    assert summary["optimizer_update_semantics"] == "sharded_across_ranks"
    assert summary["optimizer_state_ownership_semantics"] == "sharded_across_ranks"
    assert all(
        record["rank"]
        == record["parameter_owner_rank"]
        == record["gradient_owner_rank"]
        == record["optimizer_state_owner_rank"]
        == record["update_owner_rank"]
        for record in summary["optimizer_update_ownership"]
    )
    assert summary["non_owner_update_writes"] == ()
    gate_checks = summary["mps_backward_readiness_gate"]["checks"]
    assert gate_checks["optimizer_ownership_aligned"] is True
    assert gate_checks["optimizer_training_steps_executed"] is True
    measured = summary["mps_measured_runtime_evidence"]
    assert measured["valid"] is True
    assert measured["execution_scope"] == "single_node_accelerator"
    assert len(measured["rank_memory"]) == 2
    assert len(measured["communication_events"]) == 2
    assert all(
        event["elapsed_time_seconds"] > 0 for event in measured["communication_events"]
    )
    assert all(
        event["timing_scope"] == "shared_collective_wall_clock"
        for event in measured["communication_events"]
    )
    assert measured["total_communication_time_seconds"] == pytest.approx(
        measured["communication_events"][0]["elapsed_time_seconds"]
    )
    assert (
        "accelerator_collective_wall_clock_includes_rank_step_compute"
        in measured["measurement_limitations"]
    )
    assert summary["scalability_claim_allowed"] is False
