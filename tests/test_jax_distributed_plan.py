import pytest
import torch

import flagquantum as fq
import flagquantum.runtime.planner as fqxp
from flagquantum.runtime.audit import (
    DistributedScalabilityError,
    audit_distributed_scalability,
)
from flagquantum.runtime.audit.release_policy import require_distributed_scalability
from flagquantum.runtime.backends.jax import (
    JAXDistributedQuantumPlan,
    compile_quantum_kernel,
    mps_backward,
    mps_boundary_exchange,
    mps_canonicalization,
    mps_evidence,
    mps_gradient_ownership,
    mps_pullbacks,
    plan_jax_distributed_quantum_backend,
    plan_jax_sharded_mps_training,
    plan_jax_sharded_statevector_training,
    run_jax_sharded_mps,
    run_jax_sharded_statevector,
    run_jax_sharded_tensor_network,
    runtime_environment,
)
from flagquantum.runtime.backends.jax.mps import shards as mps_shards
from flagquantum.runtime.backends.jax.mps.gradient_result import (
    JAXShardedMPSParameterGradientResult,
)
from flagquantum.runtime.backends.jax.mps.gradients import (
    jax_sharded_mps_parameter_value_and_grad,
)
from flagquantum.runtime.backends.jax.mps.planning import (
    plan_jax_sharded_mps_parameter_flow,
)
from flagquantum.runtime.backends.jax.mps.result import JAXShardedMPSResult
from flagquantum.runtime.backends.jax.mps.training_records import (
    JAXShardedMPSParameterFlowPlan,
    JAXShardedMPSTrainingPlan,
)
from flagquantum.runtime.backends.jax.statevector import (
    execution as statevector_execution,
)
from flagquantum.runtime.backends.jax.statevector.execution import (
    jax_sharded_statevector_parameter_value_and_grad,
)
from flagquantum.runtime.backends.jax.statevector.gradient_records import (
    JAXShardedStatevectorParameterGradientResult,
)
from flagquantum.runtime.backends.jax.statevector.records import (
    JAXShardedStatevectorResult,
    JAXShardedStatevectorTrainingPlan,
)
from flagquantum.runtime.backends.jax.tensor_network import (
    contraction as tensor_network_contraction,
)
from flagquantum.runtime.backends.jax.tensor_network.gradients import (
    jax_sliced_tensor_network_parameter_value_and_grad,
    jax_sliced_tensor_network_value_and_grad,
)
from flagquantum.runtime.backends.jax.tensor_network.records import (
    JAXShardedTensorNetworkResult,
    JAXSlicedTensorNetworkGradientResult,
    JAXSlicedTensorNetworkParameterGradientResult,
)
from flagquantum.runtime.execution import run_advanced
from flagquantum.simulation.tensor_network.entrypoints import (
    build_tensor_network,
    build_tensor_network_expectation,
)

pytestmark = [pytest.mark.distributed, pytest.mark.distributed_cpu]


def _first_internal_tn_label(circuit):
    plan = build_tensor_network(circuit)
    counts = {}
    for node in plan.nodes:
        for label in node.labels:
            counts[label] = counts.get(label, 0) + 1
    return next(
        label
        for label, count in counts.items()
        if count >= 2 and label not in plan.output_labels
    )


def _first_internal_tn_expectation_label(circuit):
    plan = build_tensor_network_expectation(build_tensor_network(circuit), z=(0,))
    counts = {}
    for node in plan.nodes:
        for label in node.labels:
            counts[label] = counts.get(label, 0) + 1
    return next(
        label
        for label, count in counts.items()
        if count >= 2 and label not in plan.output_labels
    )


def test_jax_distributed_statevector_plan_reports_sharding_intent_without_claim(
    monkeypatch,
):
    monkeypatch.setenv("FQ_DISTRIBUTED_PROFILE", "development")
    monkeypatch.setenv("FQ_LOCAL_WORLD_SIZE", "4")
    circuit = fq.Circuit(5)
    circuit.h(0).x(4).cx(0, 4)

    plan = plan_jax_distributed_quantum_backend(circuit, mode="statevector", bsz=2)
    summary = plan.summary()
    audit = audit_distributed_scalability(summary)

    assert isinstance(plan, JAXDistributedQuantumPlan)
    assert summary["backend"] == "jax"
    assert summary["jax_backend"] == "pmap_local_cpu"
    assert summary["mode"] == "statevector"
    assert summary["world_size"] == 4
    assert summary["local_world_size"] == 4
    assert summary["distribution_semantics"] == "requires_runtime_summary"
    assert summary["intended_distribution_semantics"] == "sharded_across_ranks"
    assert summary["scalability_claim_allowed"] is False
    assert "jax_pmap_statevector_executor_pending" in summary["scalability_blockers"]
    assert (
        "rank_local_jax_kernel_is_not_capacity_scaling"
        in summary["scalability_blockers"]
    )
    assert len(summary["rank_ownership"]) == 4
    assert len(summary["local_memory_bytes_by_rank"]) == 4
    assert (
        summary["communication_tiers"]["model"]
        == "jax_pmap_collectives_planned_from_statevector_topology"
    )
    assert audit.valid
    assert not audit.scalability_claim_allowed


@pytest.mark.parametrize(
    ("mode", "expected_family"),
    (
        ("statevector", "statevector"),
        ("mps", "mps"),
        ("tensor_network", "tensor_network"),
    ),
)
def test_jax_distributed_plans_share_distributed_evidence_contract(
    monkeypatch, mode, expected_family
):
    monkeypatch.setenv("FQ_DISTRIBUTED_PROFILE", "development")
    monkeypatch.setenv("FQ_LOCAL_WORLD_SIZE", "4")
    circuit = fq.Circuit(5)
    circuit.h(0).cx(0, 1).cx(2, 3)
    kwargs = {"max_bond": 4} if mode == "mps" else {}
    if mode == "tensor_network":
        kwargs = {"max_intermediate_size": 64}

    summary = plan_jax_distributed_quantum_backend(
        circuit, mode=mode, bsz=2, **kwargs
    ).summary()
    contract = summary["distributed_evidence_contract"]

    assert contract["contract_version"] == "distributed_evidence_contract_v1"
    assert contract["backend_family"] == expected_family
    assert contract["claim_evidence_type"] == "plan_preflight"
    assert contract["status"] == "preflight_only"
    assert contract["fail_closed"] is True
    assert contract["checks"]["sharding_semantics_available"] is True
    assert contract["checks"]["rank_ownership_reported"] is True
    assert contract["checks"]["memory_plan_reported"] is True
    assert contract["checks"]["communication_plan_reported"] is True


def test_jax_sharded_mps_and_tn_runtime_summaries_share_evidence_contract(monkeypatch):
    monkeypatch.setenv("FQ_DISTRIBUTED_PROFILE", "development")
    monkeypatch.setenv("FQ_LOCAL_WORLD_SIZE", "2")
    circuit = fq.Circuit(4)
    circuit.ry(0, theta=0.2).cx(0, 1).cx(2, 3)

    mps = run_jax_sharded_mps(circuit, world_size=2, max_bond=4).summary()
    sliced_label = _first_internal_tn_label(circuit)
    tn = run_jax_sharded_tensor_network(
        circuit, world_size=2, sliced_labels=(sliced_label,)
    ).summary()

    assert mps["distributed_evidence_contract"]["backend_family"] == "mps"
    assert tn["distributed_evidence_contract"]["backend_family"] == "tensor_network"
    assert mps["distributed_evidence_contract"]["status"] == "local_simulation"
    assert tn["distributed_evidence_contract"]["status"] == "local_simulation"
    assert (
        mps["distributed_evidence_contract"]["checks"]["rank_ownership_reported"]
        is True
    )
    assert (
        tn["distributed_evidence_contract"]["checks"]["rank_ownership_reported"] is True
    )
    assert mps["claimable_production_training"] is False
    assert tn["claimable_production_training"] is False


def test_jax_sharded_statevector_training_plan_requires_device_preflight():
    circuit = fq.Circuit(4)
    circuit.ry(0, theta=0.2).cx(2, 3)

    plan = plan_jax_sharded_statevector_training(
        circuit,
        world_size=4,
        local_world_size=2,
        distributed_profile="production",
        jax_backend="pmap",
        backward_backend="pmap",
        inspect_devices=False,
    )
    summary = plan.summary()
    audit = audit_distributed_scalability(summary)

    assert isinstance(plan, JAXShardedStatevectorTrainingPlan)
    assert summary["distribution_semantics"] == "sharded_across_ranks"
    assert summary["backward_execution"] == "jax_pmap_backward"
    assert summary["gradient_ready"] is False
    assert summary["scalability_claim_allowed"] is False
    assert "production_device_preflight_required" in summary["device_blockers"]
    assert len(summary["local_memory_bytes_by_rank"]) == 4
    assert summary["statevector_training_claimability_status"] == "preflight_only"
    assert summary["claimable_production_training"] is False
    assert (
        summary["statevector_training_claimability_gate"]["checks"][
            "memory_plan_has_per_rank_shard_and_comm_buffer"
        ]
        is True
    )
    assert (
        summary["statevector_training_claimability_gate"]["checks"][
            "communication_plan_has_statevector_route"
        ]
        is True
    )
    assert audit.valid
    assert not audit.scalability_claim_allowed


def test_jax_sharded_statevector_training_plan_can_be_claimable_after_explicit_device_preflight_assumption():
    circuit = fq.Circuit(4)
    circuit.ry(0, theta=0.2).rxx(2, 3, theta=0.1).cx(2, 3)

    plan = plan_jax_sharded_statevector_training(
        circuit,
        world_size=4,
        local_world_size=2,
        distributed_profile="production",
        jax_backend="pmap",
        backward_backend="pmap",
        assume_devices_ready=True,
    )
    summary = plan.summary()

    assert summary["distribution_semantics"] == "sharded_across_ranks"
    assert summary["scalability_claim_allowed"] is False
    assert summary["sharding_plan_available"] is True
    assert summary["production_training_preflight_ready"] is True
    assert summary["gradient_ready"] is True
    assert summary["blockers"] == ()
    assert summary["device_summary"]["assumed_ready"] is True
    assert summary["estimated_transfer_bytes"] > 0
    assert summary["statevector_training_claimability_status"] == "preflight_only"
    assert summary["claimable_production_training"] is False
    assert (
        summary["statevector_training_claimability_gate"]["checks"][
            "backward_semantics_sharded"
        ]
        is True
    )
    assert (
        summary["statevector_training_claimability_gate"]["checks"][
            "parameter_gradient_available"
        ]
        is True
    )
    assert (
        summary["statevector_training_claimability_gate"]["checks"][
            "optimizer_step_preserves_sharded_ownership"
        ]
        is False
    )
    with pytest.raises(DistributedScalabilityError):
        require_distributed_scalability(summary)


def test_jax_sharded_statevector_multi_node_plan_reports_transport_boundary():
    circuit = fq.Circuit(4)
    circuit.ry(0, theta=0.2).rxx(2, 3, theta=0.1).cx(2, 3)

    plan = plan_jax_sharded_statevector_training(
        circuit,
        world_size=4,
        local_world_size=2,
        distributed_profile="production",
        jax_backend="pmap",
        backward_backend="pmap",
        assume_devices_ready=True,
    )
    summary = plan.summary()
    transport = summary["transport_evidence"]

    assert summary["node_count"] == 2
    assert summary["transport_evidence_status"] == "topology_dependent_planning"
    assert transport["status"] == "topology_dependent_planning"
    assert transport["topology_dependent"] is True
    assert "multi_node_production_transport_evidence_missing" in transport["blockers"]
    assert (
        summary["distributed_evidence_contract"]["transport_evidence"]["status"]
        == "topology_dependent_planning"
    )
    assert summary["statevector_training_claimability_status"] == "preflight_only"
    assert summary["claimable_production_training"] is False


def test_jax_sharded_statevector_training_plan_blocks_shard_map_all_to_all():
    circuit = fq.Circuit(4)
    circuit.ry(0, theta=0.2).rxx(2, 3, theta=0.1).cx(2, 3)

    plan = plan_jax_sharded_statevector_training(
        circuit,
        world_size=4,
        local_world_size=4,
        distributed_profile="production",
        jax_backend="shard_map",
        backward_backend="shard_map",
        assume_devices_ready=True,
    )
    summary = plan.summary()

    assert summary["backward_execution"] == "jax_shard_map_backward"
    assert summary["gradient_ready"] is False
    assert summary["scalability_claim_allowed"] is False
    assert any(
        "shard_map_statevector_multi_sharded_wire_transport_pending" in blocker
        for blocker in summary["static_blockers"]
    )


def test_jax_distributed_mps_plan_tracks_site_shards_and_boundary_tiers():
    circuit = fq.Circuit(6)
    circuit.h(0).cx(1, 2).cx(3, 4).rz(5, theta=0.2)

    plan = plan_jax_distributed_quantum_backend(
        circuit,
        mode="mps",
        world_size=3,
        local_world_size=2,
        max_bond=8,
    )
    summary = plan.summary()

    assert summary["mode"] == "mps"
    assert summary["world_size"] == 3
    assert summary["node_count"] == 2
    assert summary["rank_ownership"][0]["wires"] == (0, 1)
    assert summary["rank_ownership"][1]["wires"] == (2, 3)
    assert summary["rank_ownership"][2]["wires"] == (4, 5)
    assert len(summary["local_memory_bytes_by_rank"]) == 3
    assert summary["communication_tiers"]["boundary_edge_count"] == 2
    assert summary["communication_tiers"]["inter_node_communication_bytes"] > 0
    assert (
        "jax_pmap_mps_site_sharded_executor_pending" in summary["scalability_blockers"]
    )
    assert "mps_boundary_adjoint_exchange_pending" in summary["gradient_blockers"]


def test_jax_sharded_mps_parameter_flow_tracks_rank_local_parameter_gates():
    circuit = fq.Circuit(6)
    circuit.ry(0, theta=0.1).rx(3, theta=-0.2).rz(5, theta=0.3).cx(1, 2)

    plan = plan_jax_sharded_mps_parameter_flow(
        circuit,
        world_size=3,
        local_world_size=2,
        distributed_profile="production",
    )
    summary = plan.summary()

    assert isinstance(plan, JAXShardedMPSParameterFlowPlan)
    assert summary["planner"] == "jax_sharded_mps_parameter_flow"
    assert summary["parameter_gate_count"] == 3
    assert summary["rank_local_parameter_gate_count"] == 3
    assert summary["boundary_parameter_gate_count"] == 0
    assert summary["unsupported_parameter_gate_count"] == 0
    assert summary["rank_local_parameter_vjp_ready"] is True
    assert summary["rank_parameter_counts"] == (1, 1, 1)
    assert summary["boundary_parameter_edges"] == ()
    assert summary["blockers"] == ()
    assert summary["assignments"][0]["owner_rank"] == 0
    assert summary["assignments"][1]["owner_rank"] == 1
    assert summary["assignments"][2]["owner_rank"] == 2
    assert summary["gradient_reduction_plan"]["zero_fill_non_owned_gradients"] is True
    assert summary["mps_backward_readiness_status"] == "control_plane_ready"
    assert (
        summary["mps_backward_readiness_gate"]["contract_version"]
        == "distributed_evidence_contract_v1"
    )
    assert summary["mps_backward_readiness_gate"]["fail_closed"] is True
    runtime = summary["mps_runtime_summary"]
    assert runtime["status"] == "control_plane_ready"
    assert runtime["evidence_status"] == {
        "boundary_adjoint_exchange": "pending",
        "parameter_gradient_ownership": "planned",
        "backward_memory": "pending",
        "backward_communication": "pending",
        "optimizer_update_ownership": "pending",
    }
    assert "mps_boundary_adjoint_exchange_pending" in runtime["runtime_blockers"]
    assert "mps_backward_memory_evidence_pending" in runtime["runtime_blockers"]
    assert "mps_backward_communication_evidence_pending" in runtime["runtime_blockers"]
    assert "mps_optimizer_update_ownership_pending" in runtime["runtime_blockers"]
    assert "mps_parameter_gradient_ownership_pending" not in runtime["runtime_blockers"]


def test_jax_sharded_mps_parameter_flow_marks_boundary_parameter_pullback():
    circuit = fq.Circuit(6)
    circuit.rzz(1, 2, theta=0.4).crz(3, 4, theta=-0.3).rxx(0, 5, theta=0.2)

    plan = plan_jax_sharded_mps_parameter_flow(
        circuit,
        world_size=3,
        local_world_size=2,
        distributed_profile="production",
    )
    summary = plan.summary()

    assert summary["parameter_gate_count"] == 3
    assert summary["rank_local_parameter_gate_count"] == 0
    assert summary["boundary_parameter_gate_count"] == 2
    assert summary["unsupported_parameter_gate_count"] == 1
    assert summary["rank_local_parameter_vjp_ready"] is False
    assert "mps_boundary_parameter_pullback_pending" in summary["blockers"]
    assert (
        "mps_boundary_parameter_gradient_requires_executed_adjoint_route"
        in summary["blockers"]
    )
    assert (
        "mps_boundary_parameter_gradient_requires_executed_adjoint_route"
        in summary["assignments"][0]["blockers"]
    )
    assert "mps_parameter_flow_unsupported_nonlocal_gate" in summary["blockers"]
    assert len(summary["boundary_parameter_edges"]) == 2
    assert summary["boundary_parameter_edges"][0]["left_rank"] == 0
    assert summary["boundary_parameter_edges"][0]["right_rank"] == 1
    assert summary["assignments"][2]["locality"] == "unsupported"
    assert (
        summary["assignments"][2]["gradient_route"]
        == "unsupported_without_full_mps_replay"
    )


def test_jax_sharded_mps_training_plan_fails_closed_with_rank_ownership():
    circuit = fq.Circuit(6)
    circuit.h(0).ry(1, theta=0.2).cx(1, 2).cx(3, 4)

    plan = plan_jax_sharded_mps_training(
        circuit,
        world_size=3,
        local_world_size=2,
        max_bond=8,
        distributed_profile="production",
        jax_backend="pmap",
        backward_backend="pmap",
    )
    summary = plan.summary()
    audit = audit_distributed_scalability(summary)

    assert isinstance(plan, JAXShardedMPSTrainingPlan)
    assert summary["planner"] == "jax_sharded_mps_training"
    assert summary["distribution_semantics"] == "requires_runtime_summary"
    assert summary["intended_distribution_semantics"] == "sharded_across_ranks"
    assert summary["backward_execution"] == "jax_pmap_backward"
    assert summary["gradient_ready"] is False
    assert summary["scalability_claim_allowed"] is False
    assert summary["boundary_sync_count"] == 2
    assert len(summary["rank_ownership"]) == 3
    assert len(summary["local_memory_bytes_by_rank"]) == 3
    assert summary["communication_tiers"]["boundary_edge_count"] == 2
    assert "mps_boundary_adjoint_exchange_pending" in summary["static_blockers"]
    assert "pmap_mps_rank_environment_scan_pending" in summary["static_blockers"]
    assert "pmap_mps_boundary_tensor_transport_pending" in summary["static_blockers"]
    assert "production_device_preflight_required" in summary["device_blockers"]
    assert summary["parameter_flow"]["parameter_gate_count"] == 1
    assert summary["parameter_flow"]["rank_local_parameter_gate_count"] == 1
    assert summary["parameter_flow"]["boundary_parameter_gate_count"] == 0
    assert summary["parameter_flow"]["rank_local_parameter_vjp_ready"] is True
    assert summary["full_mps_reconstruction_allowed_for_claimable_backward"] is False
    assert summary["silent_statevector_fallback_allowed"] is False
    runtime = summary["mps_runtime_summary"]
    assert runtime["status"] == "blocked"
    assert runtime["evidence_status"]["parameter_gradient_ownership"] == "planned"
    assert runtime["evidence_status"]["backward_memory"] == "planned"
    assert runtime["evidence_status"]["backward_communication"] == "planned"
    assert runtime["evidence_status"]["optimizer_update_ownership"] == "pending"
    assert "mps_boundary_adjoint_exchange_pending" in summary["blockers"]
    assert "mps_parameter_gradient_ownership_pending" in summary["blockers"]
    assert "mps_optimizer_update_ownership_pending" in summary["blockers"]
    assert "mps_production_boundary_adjoint_exchange_pending" not in summary["blockers"]
    assert "mps_parameter_gradient_runtime_pending" not in summary["blockers"]
    assert summary["mps_backward_readiness_status"] == "blocked"
    assert (
        "mps_boundary_gradient_exchange_pending"
        in summary["mps_backward_readiness_blockers"]
    )
    assert (
        "mps_optimizer_update_semantics_not_measured"
        in summary["mps_backward_readiness_blockers"]
    )
    assert audit.valid
    assert not audit.scalability_claim_allowed


def test_jax_sharded_mps_training_plan_marks_local_development_as_non_scaling():
    circuit = fq.Circuit(4)
    circuit.ry(0, theta=0.1).cx(1, 2)

    plan = plan_jax_sharded_mps_training(
        circuit,
        world_size=2,
        local_world_size=2,
        max_bond=4,
        distributed_profile="development",
    )
    summary = plan.summary()

    assert summary["backend"] == "local_simulated"
    assert summary["backward_execution"] == "local_simulated_backward"
    assert summary["local_development_gradient_ready"] is True
    assert summary["gradient_ready"] is False
    assert summary["scalability_claim_allowed"] is False
    assert "local_simulated_backward_not_capacity_scaling" in summary["device_blockers"]
    assert "mps_optimizer_update_ownership_pending" in summary["blockers"]
    assert summary["parameter_flow"]["rank_local_parameter_vjp_ready"] is True
    assert summary["mps_backward_readiness_status"] == "blocked"
    assert (
        summary["mps_backward_readiness_gate"]["fallback_semantics"]
        == "local_simulation"
    )


def test_jax_distributed_tensor_network_plan_tracks_slice_tasks():
    circuit = fq.Circuit(4)
    circuit.h(0).cx(0, 3).ry(1, theta=torch.tensor(0.2)).rzz(2, 3, theta=-0.4)

    plan = plan_jax_distributed_quantum_backend(
        circuit,
        mode="tensor_network",
        world_size=2,
        max_intermediate_size=16,
    )
    summary = plan.summary()

    assert summary["mode"] == "tensor_network"
    assert summary["world_size"] == 2
    assert set(summary["task_summary"]["tasks_by_rank"]) == {0, 1}
    assert len(summary["local_memory_bytes_by_rank"]) == 2
    assert (
        summary["communication_tiers"]["model"]
        == "jax_pmap_tensor_network_slice_reduce_planned"
    )
    assert (
        "jax_pmap_tensor_network_slice_executor_pending"
        in summary["scalability_blockers"]
    )
    assert (
        "jax_sharded_tensor_network_reverse_contraction_pending"
        in summary["gradient_blockers"]
    )


def test_rank_local_jax_kernel_summary_points_to_distributed_plan():
    params = torch.zeros((1, 1, 3), dtype=torch.float32)

    def build(theta):
        circuit = fq.Circuit(1)
        circuit.rx(0, theta=theta[0, 0, 0])
        return circuit

    kernel = compile_quantum_kernel(build, params, n_wires=1, mode="statevector")
    summary = kernel.summary()

    assert summary["distribution_semantics"] == "rank_local_replicated_kernel"
    assert summary["distributed_integration_role"] == "rank_local_jax_accelerator"
    assert summary["requires_jax_distributed_plan_for_capacity_scaling"] is True
    assert "distributed_statevector" in summary["compatible_distributed_modes"]
    audit = audit_distributed_scalability(summary)
    assert audit.valid
    assert not audit.scalability_claim_allowed
    with pytest.raises(DistributedScalabilityError):
        require_distributed_scalability(summary)


def test_runtime_selection_blocks_preflight_only_jax_statevector_training_recommendation():
    circuit = fq.Circuit(4)
    circuit.ry(0, theta=0.2).rxx(2, 3, theta=0.1).cx(2, 3)

    selection = fqxp.plan_runtime_selection(
        circuit,
        world_size=4,
        local_world_size=2,
        prefer_jax=True,
        require_gradients=True,
        distributed_profile="production",
        jax_backward_backend="pmap",
        assume_jax_devices_ready=True,
    ).summary()
    candidates = {candidate["mode"]: candidate for candidate in selection["candidates"]}
    jax_statevector = candidates["jax_sharded_statevector"]

    assert selection["recommended_mode"] != "jax_sharded_statevector"
    assert jax_statevector["distribution_semantics"] == "sharded_across_ranks"
    assert (
        jax_statevector["statevector_training_claimability_status"] == "preflight_only"
    )
    assert jax_statevector["gradient_plan"]["fail_closed"] is True
    assert jax_statevector["available"] is False
    assert "statevector_training_not_claimable" in jax_statevector["blockers"]
    assert "optimizer_update_semantics_not_measured" in jax_statevector["blockers"]


def test_distributed_runtime_summaries_attach_jax_distributed_plan(monkeypatch):
    monkeypatch.setenv("FQ_DISTRIBUTED_PROFILE", "development")
    monkeypatch.setenv("FQ_LOCAL_WORLD_SIZE", "2")
    circuit = fq.Circuit(4)
    circuit.h(0).cx(1, 2).rz(3, theta=0.2)

    statevector = run_advanced(circuit, mode="distributed_statevector", device="cpu")
    mps = run_advanced(circuit, mode="distributed_mps", max_bond=4)
    tn = run_advanced(
        circuit, mode="distributed_tensor_network", max_intermediate_size=16
    )

    statevector_jax = statevector.summary()["jax_distributed_plan"]
    mps_jax = mps.summary()["jax_distributed_plan"]
    tn_jax = tn.summary()["jax_distributed_plan"]

    assert statevector_jax["mode"] == "statevector"
    assert mps_jax["mode"] == "mps"
    assert tn_jax["mode"] == "tensor_network"
    assert statevector_jax["world_size"] == 2
    assert mps_jax["world_size"] == 2
    assert tn_jax["world_size"] == 2
    assert statevector_jax["scalability_claim_allowed"] is False
    assert mps_jax["scalability_claim_allowed"] is False
    assert tn_jax["scalability_claim_allowed"] is False
    assert (
        "rank_local_jax_kernel_is_not_capacity_scaling"
        in statevector_jax["scalability_blockers"]
    )
    assert (
        "rank_local_jax_kernel_is_not_capacity_scaling"
        in mps_jax["scalability_blockers"]
    )
    assert (
        "rank_local_jax_kernel_is_not_capacity_scaling"
        in tn_jax["scalability_blockers"]
    )


def test_jax_sharded_statevector_executor_matches_native_statevector(monkeypatch):
    monkeypatch.setenv("KMP_DUPLICATE_LIB_OK", "TRUE")
    circuit = fq.Circuit(3)
    circuit.h(0).cx(0, 2).rz(1, theta=torch.tensor(0.3)).ry(2, theta=-0.2)

    result = run_jax_sharded_statevector(circuit, world_size=2)
    summary = result.summary()
    audit = audit_distributed_scalability(summary)

    assert isinstance(result, JAXShardedStatevectorResult)
    assert summary["executor"] == "jax_sharded_statevector_executor"
    assert summary["distribution_semantics"] == "sharded_across_ranks"
    assert summary["communication_execution"] == "jax_rank_local_amplitude_exchange"
    assert summary["distributed_gate_count"] >= 1
    assert summary["full_state_reconstruction_count"] == 0
    assert len(summary["rank_shards"]) == 2
    assert summary["rank_shards"][0]["global_index_count"] == 4
    assert summary["rank_shards"][1]["global_index_count"] == 4
    assert torch.allclose(result.state(), circuit.state(), atol=1e-6)
    assert result.summary()["full_state_reconstruction_count"] == 1
    assert audit.valid
    assert audit.scalability_claim_allowed is False
    assert "production_jax_pmap_executor_pending" in summary["scalability_blockers"]
    assert "jax_sharded_statevector_backward_pending" in summary["gradient_blockers"]
    assert summary["statevector_training_claimability_status"] == "local_simulation"
    assert summary["claimable_production_training"] is False
    assert (
        summary["statevector_training_claimability_gate"]["checks"][
            "parameter_gradient_available"
        ]
        is False
    )


def test_jax_sharded_statevector_executor_uses_development_env_world_size(monkeypatch):
    monkeypatch.setenv("KMP_DUPLICATE_LIB_OK", "TRUE")
    monkeypatch.setenv("FQ_DISTRIBUTED_PROFILE", "development")
    monkeypatch.setenv("FQ_LOCAL_WORLD_SIZE", "4")
    circuit = fq.Circuit(4)
    circuit.h(0).cx(0, 3).x(1)

    result = run_jax_sharded_statevector(circuit)
    summary = result.summary()

    assert summary["world_size"] == 4
    assert summary["local_world_size"] == 4
    assert summary["node_count"] == 1
    assert summary["distribution_semantics"] == "sharded_across_ranks"
    assert (
        summary["jax_distributed_plan"]["intended_distribution_semantics"]
        == "sharded_across_ranks"
    )
    assert torch.allclose(result.state(), circuit.state(), atol=1e-6)


def test_jax_sharded_statevector_executor_single_rank_is_not_distributed(monkeypatch):
    monkeypatch.setenv("KMP_DUPLICATE_LIB_OK", "TRUE")
    circuit = fq.Circuit(2)
    circuit.h(0).cx(0, 1)

    result = run_jax_sharded_statevector(circuit, world_size=1)
    summary = result.summary()
    audit = audit_distributed_scalability(summary)

    assert summary["distribution_semantics"] == "replicated_single_rank"
    assert summary["scalability_claim_allowed"] is False
    assert "world_size_is_one" in summary["scalability_blockers"]
    assert audit.valid
    assert torch.allclose(result.state(), circuit.state(), atol=1e-6)


def test_jax_sharded_statevector_parameter_gradient_matches_jax_kernel(monkeypatch):
    monkeypatch.setenv("KMP_DUPLICATE_LIB_OK", "TRUE")
    params = torch.tensor([0.2, -0.4, 0.3], dtype=torch.float32)
    reference_params = params.detach().clone().requires_grad_(True)

    def build(theta):
        circuit = fq.Circuit(3)
        circuit.ry(0, theta=theta[0])
        circuit.rx(1, theta=theta[1])
        circuit.cx(0, 2)
        circuit.rzz(1, 2, theta=theta[2])
        return circuit

    sharded = jax_sharded_statevector_parameter_value_and_grad(
        build,
        params,
        n_wires=3,
        world_size=2,
        observable="z_sum",
        jit=False,
    )
    kernel = compile_quantum_kernel(
        build,
        reference_params.detach(),
        n_wires=3,
        mode="statevector",
        observable="z_sum",
        jit=False,
    )
    reference_value = kernel(reference_params)
    reference_value.backward()
    summary = sharded.summary()

    assert isinstance(sharded, JAXShardedStatevectorParameterGradientResult)
    assert (
        summary["gradient_execution"]
        == "jax_reverse_mode_amplitude_sharded_parameter_backprop"
    )
    assert summary["gradient_target"] == "parameterized_gate_tensors"
    assert summary["distribution_semantics"] == "sharded_across_ranks"
    assert summary["backward_backend"] == "local_simulated"
    assert summary["backward_execution"] == "local_simulated_backward"
    assert summary["full_state_reconstruction_count"] == 0
    assert (
        "jax_sharded_statevector_backward_pending" not in summary["gradient_blockers"]
    )
    assert summary["statevector_training_claimability_status"] == "local_simulation"
    assert summary["claimable_production_training"] is False
    assert torch.allclose(
        sharded.torch_value(like=reference_params), reference_value.detach(), atol=1e-5
    )
    assert torch.allclose(
        sharded.torch_gradient(like=reference_params),
        reference_params.grad.detach(),
        atol=1e-5,
    )


def test_jax_sharded_statevector_parameter_gradient_pmap_indexed_all_to_all_fails_closed(
    monkeypatch,
):
    monkeypatch.setenv("KMP_DUPLICATE_LIB_OK", "TRUE")
    params = torch.tensor([0.2], dtype=torch.float32)

    def build(theta):
        circuit = fq.Circuit(3)
        circuit.ry(0, theta=theta[0])
        circuit.cx(0, 2)
        return circuit

    with pytest.raises(
        RuntimeError, match="pmap statevector backward is not production-ready"
    ):
        jax_sharded_statevector_parameter_value_and_grad(
            build,
            params,
            n_wires=3,
            world_size=3,
            observable="z_sum",
            backward_backend="pmap",
        )


def test_jax_sharded_statevector_parameter_gradient_pmap_no_transport_requires_devices(
    monkeypatch,
):
    monkeypatch.setenv("KMP_DUPLICATE_LIB_OK", "TRUE")
    monkeypatch.setattr(
        statevector_execution,
        "_jax_device_count_summary",
        lambda: {
            "local_device_count": 1,
            "global_device_count": 1,
            "process_count": 1,
            "process_index": 0,
        },
    )
    params = torch.tensor([0.2], dtype=torch.float32)

    def build(theta):
        circuit = fq.Circuit(3)
        circuit.ry(0, theta=theta[0])
        circuit.rz(2, theta=0.1)
        return circuit

    with pytest.raises(RuntimeError, match="requires at least 2 global JAX devices"):
        jax_sharded_statevector_parameter_value_and_grad(
            build,
            params,
            n_wires=3,
            world_size=2,
            observable="z_sum",
            backward_backend="pmap",
        )


def test_jax_sharded_statevector_parameter_gradient_pmap_pair_exchange_requires_devices(
    monkeypatch,
):
    monkeypatch.setenv("KMP_DUPLICATE_LIB_OK", "TRUE")
    monkeypatch.setattr(
        statevector_execution,
        "_jax_device_count_summary",
        lambda: {
            "local_device_count": 1,
            "global_device_count": 1,
            "process_count": 1,
            "process_index": 0,
        },
    )
    params = torch.tensor([0.2], dtype=torch.float32)

    def build(theta):
        circuit = fq.Circuit(3)
        circuit.ry(2, theta=theta[0])
        return circuit

    with pytest.raises(RuntimeError, match="requires at least 2 global JAX devices"):
        jax_sharded_statevector_parameter_value_and_grad(
            build,
            params,
            n_wires=3,
            world_size=2,
            observable="z_sum",
            backward_backend="pmap",
        )


def test_jax_sharded_statevector_parameter_gradient_pmap_all_to_all_requires_devices(
    monkeypatch,
):
    monkeypatch.setenv("KMP_DUPLICATE_LIB_OK", "TRUE")
    monkeypatch.setattr(
        statevector_execution,
        "_jax_device_count_summary",
        lambda: {
            "local_device_count": 1,
            "global_device_count": 1,
            "process_count": 1,
            "process_index": 0,
        },
    )
    params = torch.tensor([0.2], dtype=torch.float32)

    def build(theta):
        circuit = fq.Circuit(3)
        circuit.ry(0, theta=theta[0])
        circuit.cx(0, 2)
        return circuit

    with pytest.raises(RuntimeError, match="requires at least 2 global JAX devices"):
        jax_sharded_statevector_parameter_value_and_grad(
            build,
            params,
            n_wires=3,
            world_size=2,
            observable="z_sum",
            backward_backend="pmap",
        )


def test_jax_sharded_statevector_parameter_gradient_pmap_preflight_accepts_multiprocess_global_devices(
    monkeypatch,
):
    monkeypatch.setattr(
        runtime_environment,
        "_jax_device_count_summary",
        lambda: {
            "local_device_count": 1,
            "global_device_count": 4,
            "process_count": 4,
            "process_index": 2,
        },
    )

    ready = runtime_environment._require_jax_production_backward_ready(
        mode="statevector",
        backend="pmap",
        world_size=4,
        blockers=(),
    )

    assert ready["backend"] == "pmap"
    assert ready["execution"] == "jax_pmap_backward"
    assert ready["device_summary"]["local_device_count"] == 1
    assert ready["device_summary"]["global_device_count"] == 4


def test_jax_sharded_statevector_parameter_gradient_pmap_no_transport_matches_kernel_when_devices_available(
    monkeypatch,
):
    monkeypatch.setenv("KMP_DUPLICATE_LIB_OK", "TRUE")
    jax = pytest.importorskip("jax")
    if len(jax.local_devices()) < 2:
        pytest.skip("requires at least two local JAX devices for pmap statevector VJP")
    params = torch.tensor([0.2, -0.4], dtype=torch.float32)
    reference_params = params.detach().clone().requires_grad_(True)

    def build(theta):
        circuit = fq.Circuit(3)
        circuit.ry(0, theta=theta[0])
        circuit.rx(1, theta=theta[1])
        circuit.rz(2, theta=0.1)
        return circuit

    sharded = jax_sharded_statevector_parameter_value_and_grad(
        build,
        params,
        n_wires=3,
        world_size=2,
        observable="z_sum",
        backward_backend="pmap",
        jit=False,
    )
    kernel = compile_quantum_kernel(
        build,
        reference_params.detach(),
        n_wires=3,
        mode="statevector",
        observable="z_sum",
        jit=False,
    )
    reference_value = kernel(reference_params)
    reference_value.backward()
    summary = sharded.summary()

    assert summary["backward_backend"] == "pmap"
    assert summary["backward_execution"] == "jax_pmap_backward"
    assert summary["production_blockers"] == ()
    assert torch.allclose(
        sharded.torch_value(like=reference_params), reference_value.detach(), atol=1e-5
    )
    assert torch.allclose(
        sharded.torch_gradient(like=reference_params),
        reference_params.grad.detach(),
        atol=1e-5,
    )


def test_jax_sharded_statevector_parameter_gradient_pmap_pair_exchange_matches_kernel_when_devices_available(
    monkeypatch,
):
    monkeypatch.setenv("KMP_DUPLICATE_LIB_OK", "TRUE")
    jax = pytest.importorskip("jax")
    if len(jax.local_devices()) < 2:
        pytest.skip(
            "requires at least two local JAX devices for pmap pair-exchange statevector VJP"
        )
    params = torch.tensor([0.2, -0.4], dtype=torch.float32)
    reference_params = params.detach().clone().requires_grad_(True)

    def build(theta):
        circuit = fq.Circuit(3)
        circuit.ry(2, theta=theta[0])
        circuit.rz(2, theta=0.1)
        circuit.rx(0, theta=theta[1])
        return circuit

    sharded = jax_sharded_statevector_parameter_value_and_grad(
        build,
        params,
        n_wires=3,
        world_size=2,
        observable="z_sum",
        backward_backend="pmap",
        jit=False,
    )
    kernel = compile_quantum_kernel(
        build,
        reference_params.detach(),
        n_wires=3,
        mode="statevector",
        observable="z_sum",
        jit=False,
    )
    reference_value = kernel(reference_params)
    reference_value.backward()
    summary = sharded.summary()

    assert summary["backward_backend"] == "pmap"
    assert summary["backward_execution"] == "jax_pmap_backward"
    assert summary["production_blockers"] == ()
    assert torch.allclose(
        sharded.torch_value(like=reference_params), reference_value.detach(), atol=1e-5
    )
    assert torch.allclose(
        sharded.torch_gradient(like=reference_params),
        reference_params.grad.detach(),
        atol=1e-5,
    )


def test_jax_sharded_statevector_parameter_gradient_pmap_all_to_all_matches_kernel_when_devices_available(
    monkeypatch,
):
    monkeypatch.setenv("KMP_DUPLICATE_LIB_OK", "TRUE")
    jax = pytest.importorskip("jax")
    if len(jax.local_devices()) < 2:
        pytest.skip(
            "requires at least two local JAX devices for pmap all-to-all statevector VJP"
        )
    params = torch.tensor([0.2, -0.4], dtype=torch.float32)
    reference_params = params.detach().clone().requires_grad_(True)

    def build(theta):
        circuit = fq.Circuit(3)
        circuit.ry(0, theta=theta[0])
        circuit.rx(1, theta=theta[1])
        circuit.cx(0, 2)
        circuit.rz(2, theta=0.1)
        return circuit

    sharded = jax_sharded_statevector_parameter_value_and_grad(
        build,
        params,
        n_wires=3,
        world_size=2,
        observable="z_sum",
        backward_backend="pmap",
        jit=False,
    )
    kernel = compile_quantum_kernel(
        build,
        reference_params.detach(),
        n_wires=3,
        mode="statevector",
        observable="z_sum",
        jit=False,
    )
    reference_value = kernel(reference_params)
    reference_value.backward()
    summary = sharded.summary()

    assert summary["backward_backend"] == "pmap"
    assert summary["backward_execution"] == "jax_pmap_backward"
    assert summary["production_blockers"] == ()
    assert torch.allclose(
        sharded.torch_value(like=reference_params), reference_value.detach(), atol=1e-5
    )
    assert torch.allclose(
        sharded.torch_gradient(like=reference_params),
        reference_params.grad.detach(),
        atol=1e-5,
    )


def test_jax_sharded_statevector_parameter_gradient_pmap_4rank_multi_sharded_all_to_all_matches_kernel(
    monkeypatch,
):
    monkeypatch.setenv("KMP_DUPLICATE_LIB_OK", "TRUE")
    jax = pytest.importorskip("jax")
    if len(jax.local_devices()) < 4:
        pytest.skip(
            "requires at least four local JAX devices for 4-rank pmap all-to-all statevector VJP"
        )
    params = torch.tensor([0.2, -0.31, 0.17, 0.07], dtype=torch.float32)
    reference_params = params.detach().clone().requires_grad_(True)

    def build(theta):
        circuit = fq.Circuit(4)
        circuit.ry(0, theta=theta[0])
        circuit.rx(1, theta=theta[1])
        circuit.rxx(2, 3, theta=theta[2])
        circuit.cx(2, 3)
        circuit.ry(3, theta=theta[3])
        return circuit

    sharded = jax_sharded_statevector_parameter_value_and_grad(
        build,
        params,
        n_wires=4,
        world_size=4,
        observable="z_sum",
        backward_backend="pmap",
        distributed_profile="production",
        jit=False,
    )
    kernel = compile_quantum_kernel(
        build,
        reference_params.detach(),
        n_wires=4,
        mode="statevector",
        observable="z_sum",
        jit=False,
    )
    reference_value = kernel(reference_params)
    reference_value.backward()
    summary = sharded.summary()

    assert summary["backward_backend"] == "pmap"
    assert summary["backward_execution"] == "jax_pmap_backward"
    assert summary["production_blockers"] == ()
    assert summary["scalability_claim_allowed"] is False
    assert summary["claim_evidence_type"] == "production_runtime"
    assert summary["release_gate_allowed"] is False
    assert summary["production_value_gradient_ready"] is True
    assert summary["single_gpu_expected_oom"] is False
    assert summary["optimizer_update_semantics"] == "not_measured"
    assert summary["training_step_count"] == 0
    assert summary["distributed_gate_count"] >= 2
    assert (
        summary["simulated_communication_bytes"] == summary["estimated_transfer_bytes"]
    )
    with pytest.raises(DistributedScalabilityError):
        require_distributed_scalability(summary)
    assert torch.allclose(
        sharded.torch_value(like=reference_params), reference_value.detach(), atol=1e-5
    )
    assert torch.allclose(
        sharded.torch_gradient(like=reference_params),
        reference_params.grad.detach(),
        atol=1e-5,
    )


def test_jax_sharded_statevector_parameter_gradient_pmap_8rank_multi_sharded_all_to_all_matches_kernel(
    monkeypatch,
):
    monkeypatch.setenv("KMP_DUPLICATE_LIB_OK", "TRUE")
    jax = pytest.importorskip("jax")
    if len(jax.local_devices()) < 8:
        pytest.skip(
            "requires at least eight local JAX devices for 8-rank pmap all-to-all statevector VJP"
        )
    params = torch.tensor([0.2, -0.31, 0.17, 0.07, -0.11, 0.13], dtype=torch.float32)
    reference_params = params.detach().clone().requires_grad_(True)

    def build(theta):
        circuit = fq.Circuit(5)
        circuit.ry(0, theta=theta[0])
        circuit.rx(1, theta=theta[1])
        circuit.rxx(3, 4, theta=theta[2])
        circuit.cx(3, 4)
        circuit.cx(2, 4)
        circuit.ry(4, theta=theta[3])
        circuit.ryy(2, 4, theta=theta[4])
        circuit.rxx(2, 3, theta=theta[5])
        return circuit

    sharded = jax_sharded_statevector_parameter_value_and_grad(
        build,
        params,
        n_wires=5,
        world_size=8,
        observable="z_sum",
        backward_backend="pmap",
        distributed_profile="production",
        jit=False,
    )
    kernel = compile_quantum_kernel(
        build,
        reference_params.detach(),
        n_wires=5,
        mode="statevector",
        observable="z_sum",
        jit=False,
    )
    reference_value = kernel(reference_params)
    reference_value.backward()
    summary = sharded.summary()

    assert summary["backward_backend"] == "pmap"
    assert summary["backward_execution"] == "jax_pmap_backward"
    assert summary["production_blockers"] == ()
    assert summary["scalability_claim_allowed"] is False
    assert summary["claim_evidence_type"] == "production_runtime"
    assert summary["release_gate_allowed"] is False
    assert summary["production_value_gradient_ready"] is True
    assert summary["optimizer_update_semantics"] == "not_measured"
    assert summary["distributed_gate_count"] >= 5
    assert (
        summary["simulated_communication_bytes"] == summary["estimated_transfer_bytes"]
    )
    with pytest.raises(DistributedScalabilityError):
        require_distributed_scalability(summary)
    assert torch.allclose(
        sharded.torch_value(like=reference_params), reference_value.detach(), atol=1e-5
    )
    assert torch.allclose(
        sharded.torch_gradient(like=reference_params),
        reference_params.grad.detach(),
        atol=1e-5,
    )


def test_jax_sharded_statevector_parameter_gradient_shard_map_multiprocess_fails_closed(
    monkeypatch,
):
    monkeypatch.setenv("KMP_DUPLICATE_LIB_OK", "TRUE")
    monkeypatch.setattr(
        statevector_execution,
        "_jax_device_count_summary",
        lambda: {
            "local_device_count": 4,
            "global_device_count": 8,
            "process_count": 2,
            "process_index": 0,
        },
    )
    params = torch.tensor([0.2], dtype=torch.float32)

    def build(theta):
        circuit = fq.Circuit(4)
        circuit.ry(0, theta=theta[0])
        return circuit

    with pytest.raises(
        RuntimeError, match="multiple JAX processes requires global-array input"
    ):
        jax_sharded_statevector_parameter_value_and_grad(
            build,
            params,
            n_wires=4,
            world_size=4,
            observable="z_sum",
            backward_backend="shard_map",
            distributed_profile="production",
        )


def test_jax_sharded_statevector_parameter_gradient_shard_map_all_to_all_fails_closed(
    monkeypatch,
):
    monkeypatch.setenv("KMP_DUPLICATE_LIB_OK", "TRUE")
    monkeypatch.setattr(
        statevector_execution,
        "_jax_device_count_summary",
        lambda: {
            "local_device_count": 4,
            "global_device_count": 4,
            "process_count": 1,
            "process_index": 0,
        },
    )
    params = torch.tensor([0.2, -0.31], dtype=torch.float32)

    def build(theta):
        circuit = fq.Circuit(4)
        circuit.ry(0, theta=theta[0])
        circuit.rxx(2, 3, theta=theta[1])
        circuit.cx(2, 3)
        return circuit

    with pytest.raises(
        RuntimeError, match="shard_map_statevector_multi_sharded_wire_transport_pending"
    ):
        jax_sharded_statevector_parameter_value_and_grad(
            build,
            params,
            n_wires=4,
            world_size=4,
            observable="z_sum",
            backward_backend="shard_map",
            distributed_profile="production",
        )


def test_jax_sharded_statevector_parameter_gradient_shard_map_pair_exchange_matches_kernel(
    monkeypatch,
):
    monkeypatch.setenv("KMP_DUPLICATE_LIB_OK", "TRUE")
    jax = pytest.importorskip("jax")
    if len(jax.local_devices()) < 4:
        pytest.skip(
            "requires at least four local JAX devices for shard_map statevector VJP"
        )
    params = torch.tensor([0.2, -0.31, 0.17], dtype=torch.float32)
    reference_params = params.detach().clone().requires_grad_(True)

    def build(theta):
        circuit = fq.Circuit(4)
        circuit.ry(0, theta=theta[0])
        circuit.ry(3, theta=theta[1])
        circuit.rx(1, theta=theta[2])
        return circuit

    sharded = jax_sharded_statevector_parameter_value_and_grad(
        build,
        params,
        n_wires=4,
        world_size=4,
        observable="z_sum",
        backward_backend="shard_map",
        distributed_profile="production",
        jit=False,
    )
    kernel = compile_quantum_kernel(
        build,
        reference_params.detach(),
        n_wires=4,
        mode="statevector",
        observable="z_sum",
        jit=False,
    )
    reference_value = kernel(reference_params)
    reference_value.backward()
    summary = sharded.summary()

    assert summary["backward_backend"] == "shard_map"
    assert summary["backward_execution"] == "jax_shard_map_backward"
    assert summary["production_blockers"] == ()
    assert summary["scalability_claim_allowed"] is False
    assert summary["claim_evidence_type"] == "production_runtime"
    assert summary["release_gate_allowed"] is False
    assert summary["production_value_gradient_ready"] is True
    assert summary["optimizer_update_semantics"] == "not_measured"
    assert summary["statevector_training_claimability_status"] == "blocked"
    assert summary["claimable_production_training"] is False
    assert (
        summary["statevector_training_claimability_gate"]["checks"][
            "parameter_gradient_available"
        ]
        is True
    )
    assert (
        summary["statevector_training_claimability_gate"]["checks"][
            "optimizer_step_preserves_sharded_ownership"
        ]
        is False
    )
    assert summary["distributed_gate_count"] >= 1
    with pytest.raises(DistributedScalabilityError):
        require_distributed_scalability(summary)
    assert torch.allclose(
        sharded.torch_value(like=reference_params), reference_value.detach(), atol=1e-5
    )
    assert torch.allclose(
        sharded.torch_gradient(like=reference_params),
        reference_params.grad.detach(),
        atol=1e-5,
    )


def test_jax_sharded_statevector_parameter_gradient_shard_map_requires_local_mesh_devices(
    monkeypatch,
):
    monkeypatch.setenv("KMP_DUPLICATE_LIB_OK", "TRUE")
    monkeypatch.setattr(
        statevector_execution,
        "_jax_device_count_summary",
        lambda: {
            "local_device_count": 1,
            "global_device_count": 4,
            "process_count": 1,
            "process_index": 0,
        },
    )
    params = torch.tensor([0.2], dtype=torch.float32)

    def build(theta):
        circuit = fq.Circuit(4)
        circuit.ry(0, theta=theta[0])
        return circuit

    with pytest.raises(RuntimeError, match="requires at least 4 local JAX devices"):
        jax_sharded_statevector_parameter_value_and_grad(
            build,
            params,
            n_wires=4,
            world_size=4,
            observable="z_sum",
            backward_backend="shard_map",
            distributed_profile="production",
        )


def test_jax_sharded_statevector_parameter_gradient_production_auto_fails_closed(
    monkeypatch,
):
    monkeypatch.setenv("KMP_DUPLICATE_LIB_OK", "TRUE")
    monkeypatch.setenv("FQ_DISTRIBUTED_PROFILE", "production")
    monkeypatch.setenv("FQ_JAX_DISTRIBUTED_BACKEND", "pmap")
    monkeypatch.setattr(
        statevector_execution,
        "_jax_device_count_summary",
        lambda: {
            "local_device_count": 1,
            "global_device_count": 1,
            "process_count": 1,
            "process_index": 0,
        },
    )
    params = torch.tensor([0.2], dtype=torch.float32)

    def build(theta):
        circuit = fq.Circuit(3)
        circuit.ry(0, theta=theta[0])
        return circuit

    with pytest.raises(RuntimeError, match="requires at least 2 global JAX devices"):
        jax_sharded_statevector_parameter_value_and_grad(
            build,
            params,
            n_wires=3,
            world_size=2,
            observable="z_sum",
        )


def test_jax_sharded_mps_executor_matches_native_mps_without_statevector_fallback(
    monkeypatch,
):
    monkeypatch.setenv("KMP_DUPLICATE_LIB_OK", "TRUE")
    circuit = fq.Circuit(4)
    circuit.h(0).cx(0, 1).ry(2, theta=torch.tensor(0.2)).cx(1, 2).rz(3, theta=-0.3)

    result = run_jax_sharded_mps(circuit, world_size=2, max_bond=8)
    summary = result.summary()
    audit = audit_distributed_scalability(summary)

    assert isinstance(result, JAXShardedMPSResult)
    assert summary["executor"] == "jax_sharded_mps_executor"
    assert summary["distribution_semantics"] == "sharded_across_ranks"
    assert summary["boundary_sync_count"] == 1
    assert summary["silent_statevector_fallback"] is False
    assert summary["full_state_fallback_count"] == 0
    assert summary["full_mps_reconstruction_count"] == 0
    assert summary["statevector_facade_count"] == 0
    assert summary["rank_shards"][0]["local_tensor_wires"] == (0, 1)
    assert summary["rank_shards"][1]["local_tensor_wires"] == (2, 3)
    assert (
        summary["communication_tiers"]["boundary_protocols"][0]["tier"] == "intra_node"
    )
    assert torch.allclose(result.to_statevector(), circuit.state(), atol=1e-5)
    after = result.summary()
    assert after["full_mps_reconstruction_count"] == 1
    assert after["statevector_facade_count"] == 1
    assert audit.valid
    assert audit.scalability_claim_allowed is False
    assert (
        "production_jax_pmap_mps_site_executor_pending"
        in summary["scalability_blockers"]
    )
    assert "mps_boundary_adjoint_exchange_pending" in summary["gradient_blockers"]
    assert summary["mps_backward_readiness_status"] == "blocked"
    assert summary["mps_backward_readiness_gate"]["fail_closed"] is True


def test_jax_sharded_mps_run_mode_uses_development_env_world_size(monkeypatch):
    monkeypatch.setenv("KMP_DUPLICATE_LIB_OK", "TRUE")
    monkeypatch.setenv("FQ_DISTRIBUTED_PROFILE", "development")
    monkeypatch.setenv("FQ_LOCAL_WORLD_SIZE", "2")
    circuit = fq.Circuit(4)
    circuit.h(0).cx(0, 1).cx(1, 2)

    result = run_advanced(circuit, mode="jax_sharded_mps", max_bond=8)
    summary = result.summary()

    assert summary["world_size"] == 2
    assert summary["distribution_semantics"] == "sharded_across_ranks"
    assert summary["jax_distributed_plan"]["mode"] == "mps"
    assert torch.allclose(result.to_statevector(), circuit.state(), atol=1e-5)


def test_jax_sharded_mps_rejects_remote_gate_without_fallback(monkeypatch):
    monkeypatch.setenv("KMP_DUPLICATE_LIB_OK", "TRUE")
    circuit = fq.Circuit(4)
    circuit.h(0).cx(0, 3)

    with pytest.raises(RuntimeError, match="without a full-MPS/statevector fallback"):
        run_jax_sharded_mps(circuit, world_size=2, max_bond=8)


def test_jax_sharded_mps_parameter_gradient_matches_jax_mps_kernel_without_statevector(
    monkeypatch,
):
    monkeypatch.setenv("KMP_DUPLICATE_LIB_OK", "TRUE")
    params = torch.tensor([0.2, -0.4, 0.3], dtype=torch.float32)
    reference_params = params.detach().clone().requires_grad_(True)

    def build(theta):
        circuit = fq.Circuit(4)
        circuit.ry(0, theta=theta[0])
        circuit.rx(1, theta=theta[1])
        circuit.cx(0, 1)
        circuit.rzz(1, 2, theta=theta[2])
        circuit.cx(2, 3)
        return circuit

    sharded = jax_sharded_mps_parameter_value_and_grad(
        build,
        params,
        n_wires=4,
        world_size=2,
        max_bond=8,
        observable="z_sum",
        jit=False,
    )
    kernel = compile_quantum_kernel(
        build,
        reference_params.detach(),
        n_wires=4,
        mode="mps",
        observable="z_sum",
        max_bond=8,
        jit=False,
    )
    reference_value = kernel(reference_params)
    reference_value.backward()
    summary = sharded.summary()

    assert isinstance(sharded, JAXShardedMPSParameterGradientResult)
    assert (
        summary["gradient_execution"]
        == "jax_reverse_mode_site_sharded_mps_parameter_backprop"
    )
    assert summary["gradient_target"] == "parameterized_gate_tensors"
    assert summary["distribution_semantics"] == "sharded_across_ranks"
    assert summary["backward_backend"] == "local_simulated"
    assert summary["backward_execution"] == "local_simulated_backward"
    assert summary["full_mps_reconstruction_count"] == 0
    assert summary["statevector_facade_count"] == 0
    assert summary["silent_statevector_fallback"] is False
    assert summary["mps_backward_readiness_status"] == "blocked"
    assert (
        summary["mps_backward_readiness_gate"]["fallback_semantics"]
        == "local_simulation"
    )
    evidence = summary["boundary_adjoint_exchange_evidence"]
    assert evidence["status"] == "local_cpu_executed"
    assert evidence["valid"] is True
    assert evidence["execution_scope"] == "development_cpu"
    assert evidence["evidence_scope"] == "boundary_exchange_probe_only"
    assert evidence["directed_exchange_count"] == 2
    assert evidence["communication_bytes"] > 0
    assert evidence["scalability_claim_allowed"] is False
    assert "mps_boundary_adjoint_exchange_local_cpu_only" in evidence["blockers"]
    assert "mps_full_sharded_backward_executor_pending" in evidence["blockers"]
    first_exchange = evidence["records"][0]
    assert first_exchange["boundary_edge_id"] == "edge_0:1-2"
    assert first_exchange["source_rank"] == 0
    assert first_exchange["target_rank"] == 1
    assert first_exchange["owned_site_ranges"] == {
        "source": (0, 1),
        "target": (2, 3),
    }
    assert first_exchange["bond_id"] == 1
    assert first_exchange["boundary_tensor_shape"]
    assert first_exchange["boundary_tensor_dtype"] == "complex64"
    assert first_exchange["communication_primitive"] == "local_cpu_point_to_point_copy"
    assert first_exchange["boundary_adjoint_exchange_status"] == "executed"
    assert first_exchange["blockers"] == ()
    assert (
        summary["distributed_evidence_contract"]["claim_evidence_type"]
        == "development_smoke"
    )
    contract_exchange = summary["distributed_evidence_contract"]["communication_plan"][
        "communication_tiers"
    ]["boundary_adjoint_exchange_evidence"]
    assert contract_exchange["status"] == "local_cpu_executed"
    assert contract_exchange["scalability_claim_allowed"] is False
    assert summary["scalability_claim_allowed"] is False
    assert torch.allclose(
        sharded.torch_value(like=reference_params), reference_value.detach(), atol=1e-5
    )
    assert torch.allclose(
        sharded.torch_gradient(like=reference_params),
        reference_params.grad.detach(),
        atol=1e-5,
    )


@pytest.mark.release_gate
@pytest.mark.benchmark_contract
def test_jax_sharded_mps_parameter_gradient_emits_rank_owned_attribution(monkeypatch):
    monkeypatch.setenv("KMP_DUPLICATE_LIB_OK", "TRUE")
    params = torch.tensor([0.2, -0.4, 0.3, 0.1], dtype=torch.float32)

    def build(theta):
        circuit = fq.Circuit(4)
        circuit.ry(0, theta=theta[2])
        circuit.rx(2, theta=theta[0])
        circuit.rzz(1, 2, theta=theta[3])
        circuit.rz(3, theta=theta[1])
        return circuit

    result = jax_sharded_mps_parameter_value_and_grad(
        build,
        params,
        n_wires=4,
        world_size=2,
        max_bond=8,
        observable="z_sum",
        jit=False,
    )
    summary = result.summary()
    evidence = summary["parameter_gradient_ownership_evidence"]

    assert evidence["status"] == "local_cpu_attributed"
    assert evidence["valid"] is True
    assert evidence["execution_scope"] == "development_cpu"
    assert evidence["evidence_scope"] == "parameter_gradient_ownership_attribution"
    assert evidence["gradient_source"] == "local_simulated_global_parameter_gradient"
    assert (
        evidence["gradient_distribution_semantics"]
        == "local_simulation_with_rank_owned_attribution"
    )
    assert evidence["parameter_count"] == 4
    assert evidence["owned_parameter_count"] == 4
    assert evidence["scalability_claim_allowed"] is False
    assert evidence["release_gate_allowed"] is False
    assert evidence["replicated_rank_autograd"] is False
    assert evidence["full_local_mps_replay"] is False
    assert evidence["statevector_fallback"] is False
    ownership = evidence["parameter_gradient_ownership"]
    assert [item["owner_rank"] for item in ownership] == [1, 1, 0, 0]
    assert [item["site_range"] for item in ownership] == [
        (2, 3),
        (2, 3),
        (0, 1),
        (0, 1),
    ]
    assert [item["crosses_shard_boundary"] for item in ownership] == [
        False,
        False,
        False,
        True,
    ]
    assert tuple(item["gradient_value"] for item in ownership) == pytest.approx(
        tuple(float(item) for item in result.torch_gradient())
    )
    boundary_record = next(
        record for record in evidence["records"] if record["crosses_shard_boundary"]
    )
    assert boundary_record["gate_id"] == "instruction_2"
    assert boundary_record["gradient_route"] == "boundary_parameter_vjp"
    assert boundary_record["touched_ranks"] == (0, 1)
    assert (
        "mps_boundary_parameter_gradient_requires_executed_adjoint_route:instruction_2"
        in boundary_record["blockers"]
    )
    assert (
        summary["mps_backward_readiness_gate"]["checks"][
            "parameter_gradient_ownership_reported"
        ]
        is True
    )
    assert summary["mps_backward_readiness_status"] == "blocked"
    assert summary["scalability_claim_allowed"] is False
    with pytest.raises(DistributedScalabilityError):
        require_distributed_scalability(summary)

    stepped = result.with_sharded_optimizer_step_evidence(learning_rate=0.05)
    stepped_summary = stepped.summary()
    evidence = stepped_summary["optimizer_step_evidence"]
    assert evidence["status"] == "local_cpu_executed"
    assert evidence["execution_scope"] == "development_cpu"
    assert evidence["training_step_count"] == 1
    assert evidence["parameter_count"] == params.numel()
    assert evidence["scalability_claim_allowed"] is False
    assert stepped_summary["parameter_ownership_semantics"] == "sharded_across_ranks"
    assert stepped_summary["gradient_ownership_semantics"] == "sharded_across_ranks"
    assert stepped_summary["optimizer_update_semantics"] == "sharded_across_ranks"
    assert (
        stepped_summary["optimizer_update_ownership_semantics"]
        == "sharded_across_ranks"
    )
    assert stepped_summary["training_step_count"] == 1
    assert all(
        record["parameter_owner_rank"]
        == record["gradient_owner_rank"]
        == record["update_owner_rank"]
        for record in stepped_summary["optimizer_update_ownership"]
    )
    assert all(
        record["writeback_route"] == "rank_local_parameter_owner_writeback"
        for record in stepped_summary["optimizer_update_ownership"]
    )
    assert torch.allclose(
        torch.as_tensor(stepped.updated_parameters),
        params - 0.05 * result.torch_gradient(like=params),
        atol=1e-6,
    )
    optimizer_checks = stepped_summary["mps_backward_readiness_gate"]["checks"]
    assert optimizer_checks["optimizer_ownership_aligned"] is True
    assert optimizer_checks["optimizer_training_steps_executed"] is True
    assert (
        "mps_optimizer_update_semantics_not_measured"
        not in stepped_summary["mps_backward_readiness_blockers"]
    )
    assert (
        "mps_production_backward_execution_not_measured"
        in stepped_summary["mps_backward_readiness_blockers"]
    )
    assert stepped_summary["mps_backward_readiness_status"] == "blocked"
    assert (
        stepped_summary["distributed_evidence_contract"]["claim_evidence_type"]
        == "development_smoke"
    )
    assert (
        stepped_summary["distributed_evidence_contract"][
            "claimable_production_training"
        ]
        is False
    )
    assert stepped_summary["distributed_evidence_contract"]["fail_closed"] is True
    assert stepped_summary["scalability_claim_allowed"] is False
    with pytest.raises(DistributedScalabilityError):
        require_distributed_scalability(stepped_summary)


@pytest.mark.parametrize("bad_learning_rate", (0.0, -0.1, float("inf"), "not-a-rate"))
def test_jax_sharded_mps_optimizer_evidence_rejects_invalid_learning_rate(
    monkeypatch,
    bad_learning_rate,
):
    monkeypatch.setenv("KMP_DUPLICATE_LIB_OK", "TRUE")
    params = torch.tensor([0.2, -0.1], dtype=torch.float32)

    def build(theta):
        circuit = fq.Circuit(4)
        circuit.ry(0, theta=theta[0])
        circuit.rx(2, theta=theta[1])
        return circuit

    result = jax_sharded_mps_parameter_value_and_grad(
        build,
        params,
        n_wires=4,
        world_size=2,
        max_bond=8,
        observable="z_sum",
        jit=False,
    )

    with pytest.raises(
        ValueError, match="learning_rate must be a positive finite number"
    ):
        result.with_sharded_optimizer_step_evidence(learning_rate=bad_learning_rate)


def test_jax_sharded_mps_optimizer_evidence_rejects_incomplete_owner_records(
    monkeypatch,
):
    monkeypatch.setenv("KMP_DUPLICATE_LIB_OK", "TRUE")
    params = torch.tensor([0.2, -0.1], dtype=torch.float32)

    def build(theta):
        circuit = fq.Circuit(4)
        circuit.ry(0, theta=theta[0])
        circuit.rx(2, theta=theta[1])
        return circuit

    result = jax_sharded_mps_parameter_value_and_grad(
        build,
        params,
        n_wires=4,
        world_size=2,
        max_bond=8,
        observable="z_sum",
        jit=False,
    )
    evidence = dict(result.parameter_gradient_ownership_evidence)
    evidence["parameter_gradient_ownership"] = evidence["parameter_gradient_ownership"][
        :-1
    ]
    result.parameter_gradient_ownership_evidence = evidence

    with pytest.raises(
        ValueError,
        match="parameter-gradient ownership must cover each parameter exactly once",
    ):
        result.with_sharded_optimizer_step_evidence()


def test_jax_sharded_mps_parameter_gradient_shared_cross_rank_owner_fails_closed(
    monkeypatch,
):
    monkeypatch.setenv("KMP_DUPLICATE_LIB_OK", "TRUE")
    params = torch.tensor([0.2], dtype=torch.float32)

    def build(theta):
        circuit = fq.Circuit(4)
        circuit.ry(0, theta=theta[0])
        circuit.rx(2, theta=theta[0])
        return circuit

    result = jax_sharded_mps_parameter_value_and_grad(
        build,
        params,
        n_wires=4,
        world_size=2,
        max_bond=8,
        observable="z_sum",
        jit=False,
    )
    summary = result.summary()
    evidence = summary["parameter_gradient_ownership_evidence"]

    assert evidence["status"] == "blocked"
    assert evidence["valid"] is False
    assert evidence["owned_parameter_count"] == 0
    assert (
        "mps_shared_parameter_cross_rank_reduction_pending:parameter_0"
        in evidence["execution_blockers"]
    )
    assert summary["parameter_gradient_ownership"] == "unknown"
    assert summary["mps_backward_readiness_status"] == "blocked"
    assert summary["scalability_claim_allowed"] is False
    with pytest.raises(
        ValueError,
        match="complete parameter-gradient ownership",
    ):
        result.with_sharded_optimizer_step_evidence()


@pytest.mark.parametrize(
    ("build_kind", "params", "expected_blocker"),
    (
        (
            "ambiguous",
            torch.tensor([0.2, 0.3], dtype=torch.float32),
            "mps_parameter_gradient_dependency_ambiguous:instruction_0:theta",
        ),
        (
            "missing",
            torch.tensor([0.2], dtype=torch.float32),
            "mps_parameter_gradient_dependency_missing:instruction_0:theta",
        ),
    ),
)
def test_jax_sharded_mps_parameter_gradient_dependency_mapping_fails_closed(
    monkeypatch, build_kind, params, expected_blocker
):
    monkeypatch.setenv("KMP_DUPLICATE_LIB_OK", "TRUE")

    def build(theta):
        circuit = fq.Circuit(4)
        if build_kind == "ambiguous":
            circuit.ry(0, theta=theta[0] + theta[1])
        else:
            circuit.ry(0, theta=theta[0] * 0.0 + 0.25)
        return circuit

    result = jax_sharded_mps_parameter_value_and_grad(
        build,
        params,
        n_wires=4,
        world_size=2,
        max_bond=8,
        observable="z_sum",
        jit=False,
    )
    summary = result.summary()
    evidence = summary["parameter_gradient_ownership_evidence"]

    assert evidence["status"] == "blocked"
    assert evidence["valid"] is False
    assert evidence["owned_parameter_count"] == 0
    assert expected_blocker in evidence["execution_blockers"]
    assert summary["parameter_gradient_ownership"] == "unknown"
    assert summary["mps_backward_readiness_status"] == "blocked"
    assert summary["scalability_claim_allowed"] is False


def test_jax_sharded_mps_parameter_gradient_ownership_helpers_remain_internal_api():
    assert hasattr(
        mps_gradient_ownership,
        "_execute_local_mps_parameter_gradient_ownership",
    )
    assert not hasattr(fq, "_execute_local_mps_parameter_gradient_ownership")


def test_jax_sharded_mps_backward_resource_evidence_accounts_every_rank(
    monkeypatch,
):
    monkeypatch.setenv("KMP_DUPLICATE_LIB_OK", "TRUE")
    params = torch.tensor([0.2, -0.4, 0.3], dtype=torch.float32)

    def build(theta):
        circuit = fq.Circuit(4)
        circuit.ry(0, theta=theta[0])
        circuit.rx(2, theta=theta[1])
        circuit.rzz(1, 2, theta=theta[2])
        return circuit

    result = jax_sharded_mps_parameter_value_and_grad(
        build,
        params,
        n_wires=4,
        world_size=2,
        max_bond=8,
        observable="z_sum",
        jit=False,
    )
    summary = result.summary()
    resource = summary["mps_backward_resource_evidence"]
    memory = summary["mps_backward_memory_plan"]
    communication = summary["mps_backward_communication_plan"]

    assert resource["status"] == "complete"
    assert resource["valid"] is True
    assert resource["claim_evidence_type"] == "development_smoke"
    assert resource["scalability_claim_allowed"] is False
    assert len(memory["rank_memory"]) == 2
    assert {item["rank"] for item in memory["rank_memory"]} == {0, 1}
    assert all(item["forward_tensor_bytes"] > 0 for item in memory["rank_memory"])
    assert all(item["backward_adjoint_bytes"] > 0 for item in memory["rank_memory"])
    assert all(
        item["canonicalization_temporary_bytes"] > 0 for item in memory["rank_memory"]
    )
    assert all(
        item["estimated_peak_backward_bytes"] >= item["forward_tensor_bytes"]
        for item in memory["rank_memory"]
    )
    assert memory["boundary_gradient_buffer_bytes_by_rank"] == tuple(
        summary["boundary_adjoint_exchange_evidence"]["local_memory_bytes_by_rank"]
    )
    assert communication["status"] == "local_cpu_accounted"
    assert communication["boundary_edge_count"] == 1
    edge = communication["boundary_edges"][0]
    assert edge["communication_bytes"] == communication["communication_bytes"]
    assert edge["communication_primitive"] == "local_cpu_point_to_point_copy"
    assert edge["topology_tier"] == "intra_node"
    assert edge["topology_route"] == "single_node_local_cpu"
    assert edge["topology_dependent"] is False
    assert edge["execution_status"] == "executed"
    assert summary["mps_backward_readiness_status"] == "blocked"
    with pytest.raises(DistributedScalabilityError):
        require_distributed_scalability(summary)


def test_jax_sharded_mps_training_resource_plan_marks_inter_node_route_topology_dependent():
    circuit = fq.Circuit(8)
    circuit.ry(0, theta=0.1)
    circuit.rzz(1, 2, theta=0.2)
    circuit.rzz(3, 4, theta=0.3)

    summary = plan_jax_sharded_mps_training(
        circuit,
        world_size=4,
        local_world_size=2,
        max_bond=8,
        distributed_profile="production",
        jax_backend="pmap",
        backward_backend="pmap",
    ).summary()
    resource = summary["mps_backward_resource_evidence"]
    memory = summary["mps_backward_memory_plan"]
    communication = summary["mps_backward_communication_plan"]

    assert resource["status"] == "complete"
    assert resource["claim_evidence_type"] == "plan_preflight"
    assert len(memory["rank_memory"]) == 4
    assert len(memory["forward_tensor_bytes_by_rank"]) == 4
    assert len(memory["backward_adjoint_bytes_by_rank"]) == 4
    assert len(memory["canonicalization_temporary_bytes_by_rank"]) == 4
    assert communication["boundary_edge_count"] == 3
    inter_node_edges = [
        edge
        for edge in communication["boundary_edges"]
        if edge["topology_tier"] == "inter_node"
    ]
    assert inter_node_edges
    assert all(edge["topology_dependent"] is True for edge in inter_node_edges)
    assert all(
        edge["topology_route"] == "topology_dependent" for edge in inter_node_edges
    )
    assert all(
        edge["communication_primitive"] == "planned_adjacent_rank_point_to_point"
        for edge in communication["boundary_edges"]
    )
    assert communication["inter_node_communication_bytes"] > 0
    assert summary["scalability_claim_allowed"] is False
    assert summary["mps_backward_readiness_status"] == "blocked"


def _valid_boundary_adjoint_exchange_records():
    common = {
        "boundary_edge_id": "edge_0:1-2",
        "owned_site_ranges": {"source": (0, 1), "target": (2, 3)},
        "bond_id": 1,
        "boundary_id": "bond_1_2",
        "boundary_tensor_shape": (1, 2, 2, 2),
        "boundary_tensor_dtype": "complex64",
        "communication_primitive": "local_cpu_point_to_point_copy",
        "local_memory_bytes": 64,
        "communication_bytes": 64,
        "boundary_adjoint_exchange_status": "executed",
        "adjoint_source": "synthetic_unit_boundary_probe",
        "blockers": (),
    }
    return (
        {
            **common,
            "direction": "left_to_right",
            "source_rank": 0,
            "target_rank": 1,
            "source_wire": 1,
            "target_wire": 2,
            "boundary_gradient_ownership": {
                "source_rank": 0,
                "target_rank": 1,
                "owner_rank": 1,
                "ownership_semantics": "received_boundary_adjoint",
            },
        },
        {
            **common,
            "direction": "right_to_left",
            "source_rank": 1,
            "target_rank": 0,
            "source_wire": 2,
            "target_wire": 1,
            "owned_site_ranges": {"source": (2, 3), "target": (0, 1)},
            "boundary_gradient_ownership": {
                "source_rank": 1,
                "target_rank": 0,
                "owner_rank": 0,
                "ownership_semantics": "received_boundary_adjoint",
            },
        },
    )


@pytest.mark.parametrize(
    ("field", "expected_blocker"),
    (
        ("boundary_edge_id", "mps_boundary_adjoint_exchange_route_missing"),
        (
            "owned_site_ranges",
            "mps_boundary_adjoint_exchange_rank_ownership_missing",
        ),
        (
            "boundary_tensor_shape",
            "mps_boundary_adjoint_exchange_tensor_shape_missing",
        ),
        ("blockers", "injected_boundary_exchange_blocker"),
    ),
)
def test_mps_boundary_adjoint_exchange_evidence_fails_closed(field, expected_blocker):
    records = [dict(record) for record in _valid_boundary_adjoint_exchange_records()]
    if field == "blockers":
        records[0][field] = ("injected_boundary_exchange_blocker",)
    else:
        records[0].pop(field)

    evidence = mps_boundary_exchange._summarize_local_mps_boundary_adjoint_exchange(
        records,
        world_size=2,
        expected_boundary_edges=1,
    )

    assert evidence["status"] == "blocked"
    assert evidence["valid"] is False
    assert evidence["scalability_claim_allowed"] is False
    assert any(expected_blocker in blocker for blocker in evidence["blockers"])


@pytest.mark.parametrize(
    ("mutate", "expected_blocker"),
    (
        (
            lambda record: record.update({"communication_primitive": "all_reduce"}),
            "mps_boundary_adjoint_exchange_primitive_missing",
        ),
        (
            lambda record: record.update({"communication_bytes": 0}),
            "mps_boundary_adjoint_exchange_communication_bytes_missing",
        ),
        (
            lambda record: record.update({"local_memory_bytes": 0}),
            "mps_boundary_adjoint_exchange_local_memory_missing",
        ),
        (
            lambda record: record.update(
                {
                    "boundary_gradient_ownership": {
                        "owner_rank": record["source_rank"],
                        "ownership_semantics": "received_boundary_adjoint",
                    }
                }
            ),
            "mps_boundary_gradient_ownership_missing",
        ),
        (
            lambda record: record.update(
                {"boundary_adjoint_exchange_status": "planned"}
            ),
            "mps_boundary_adjoint_exchange_not_executed",
        ),
    ),
)
def test_mps_boundary_adjoint_exchange_evidence_rejects_malformed_execution_metadata(
    mutate, expected_blocker
):
    records = [dict(record) for record in _valid_boundary_adjoint_exchange_records()]
    mutate(records[0])

    evidence = mps_boundary_exchange._summarize_local_mps_boundary_adjoint_exchange(
        records,
        world_size=2,
        expected_boundary_edges=1,
    )

    assert evidence["status"] == "blocked"
    assert evidence["valid"] is False
    assert evidence["scalability_claim_allowed"] is False
    assert evidence["release_gate_allowed"] is False
    assert any(expected_blocker in blocker for blocker in evidence["blockers"])


def test_mps_boundary_adjoint_exchange_helpers_remain_internal_metadata_api():
    assert not hasattr(fq, "_execute_local_mps_boundary_adjoint_exchange")
    assert not hasattr(fq, "_summarize_local_mps_boundary_adjoint_exchange")


def test_jax_sharded_mps_parameter_gradient_pmap_fails_closed(monkeypatch):
    monkeypatch.setenv("KMP_DUPLICATE_LIB_OK", "TRUE")
    params = torch.tensor([0.2], dtype=torch.float32)

    def build(theta):
        circuit = fq.Circuit(4)
        circuit.ry(0, theta=theta[0])
        circuit.cx(1, 2)
        return circuit

    with pytest.raises(RuntimeError, match="pmap mps backward is not production-ready"):
        jax_sharded_mps_parameter_value_and_grad(
            build,
            params,
            n_wires=4,
            world_size=2,
            max_bond=8,
            observable="z_sum",
            backward_backend="pmap",
        )


def test_jax_sharded_mps_parameter_gradient_shard_map_fails_closed(monkeypatch):
    monkeypatch.setenv("KMP_DUPLICATE_LIB_OK", "TRUE")
    params = torch.tensor([0.2], dtype=torch.float32)

    def build(theta):
        circuit = fq.Circuit(4)
        circuit.ry(0, theta=theta[0])
        return circuit

    with pytest.raises(
        RuntimeError, match="JAX shard_map mps backward is not production-ready"
    ):
        jax_sharded_mps_parameter_value_and_grad(
            build,
            params,
            n_wires=4,
            world_size=2,
            max_bond=8,
            observable="z_sum",
            backward_backend="shard_map",
        )


def test_jax_sharded_mps_parameter_gradient_production_auto_fails_closed(monkeypatch):
    monkeypatch.setenv("KMP_DUPLICATE_LIB_OK", "TRUE")
    monkeypatch.setenv("FQ_DISTRIBUTED_PROFILE", "production")
    monkeypatch.setenv("FQ_JAX_DISTRIBUTED_BACKEND", "pmap")
    params = torch.tensor([0.2], dtype=torch.float32)

    def build(theta):
        circuit = fq.Circuit(4)
        circuit.ry(0, theta=theta[0])
        return circuit

    with pytest.raises(RuntimeError, match="pmap mps backward is not production-ready"):
        jax_sharded_mps_parameter_value_and_grad(
            build,
            params,
            n_wires=4,
            world_size=2,
            max_bond=8,
            observable="z_sum",
        )


def test_jax_sharded_tensor_network_executor_matches_native_without_full_fallback(
    monkeypatch,
):
    monkeypatch.setenv("KMP_DUPLICATE_LIB_OK", "TRUE")
    circuit = fq.Circuit(3)
    circuit.h(0).cx(0, 2).ry(1, theta=torch.tensor(0.2))
    sliced_label = _first_internal_tn_label(circuit)

    result = run_jax_sharded_tensor_network(
        circuit, world_size=2, sliced_labels=(sliced_label,)
    )
    summary = result.summary()
    audit = audit_distributed_scalability(summary)

    assert isinstance(result, JAXShardedTensorNetworkResult)
    assert summary["executor"] == "jax_sharded_tensor_network_executor"
    assert summary["distribution_semantics"] == "sharded_across_ranks"
    assert summary["slice_task_count"] == 2
    assert summary["active_ranks"] == (0, 1)
    assert summary["rank_shards"][0]["slice_task_count"] == 1
    assert summary["rank_shards"][1]["slice_task_count"] == 1
    assert summary["silent_statevector_fallback"] is False
    assert summary["full_tensor_network_fallback_count"] == 0
    assert summary["final_state_facade_count"] == 0
    assert summary["communication_tiers"]["collective"] == "local_simulated_psum"
    assert torch.allclose(result.state(), circuit.state(), atol=1e-6)
    assert result.summary()["final_state_facade_count"] == 1
    assert audit.valid
    assert audit.scalability_claim_allowed is False
    assert (
        "production_jax_pmap_tensor_network_compute_pending"
        in summary["scalability_blockers"]
    )
    assert (
        "jax_sharded_tensor_network_reverse_contraction_pending"
        in summary["gradient_blockers"]
    )


def test_jax_sharded_tensor_network_run_mode_uses_development_env_world_size(
    monkeypatch,
):
    monkeypatch.setenv("KMP_DUPLICATE_LIB_OK", "TRUE")
    monkeypatch.setenv("FQ_DISTRIBUTED_PROFILE", "development")
    monkeypatch.setenv("FQ_LOCAL_WORLD_SIZE", "2")
    circuit = fq.Circuit(3)
    circuit.h(0).cx(0, 2)
    sliced_label = _first_internal_tn_label(circuit)

    result = run_advanced(
        circuit, mode="jax_sharded_tensor_network", sliced_labels=(sliced_label,)
    )
    summary = result.summary()

    assert summary["world_size"] == 2
    assert summary["distribution_semantics"] == "sharded_across_ranks"
    assert summary["jax_distributed_plan"]["mode"] == "tensor_network"
    assert torch.allclose(result.to_statevector(), circuit.state(), atol=1e-6)


def test_jax_sharded_tensor_network_rejects_unsliced_multi_rank(monkeypatch):
    monkeypatch.setenv("KMP_DUPLICATE_LIB_OK", "TRUE")
    circuit = fq.Circuit(3)
    circuit.h(0).cx(0, 2)

    with pytest.raises(RuntimeError, match="requires at least two slice tasks"):
        run_jax_sharded_tensor_network(circuit, world_size=2)


def test_jax_sharded_tensor_network_pmap_collective_fails_closed_without_devices(
    monkeypatch,
):
    monkeypatch.setenv("KMP_DUPLICATE_LIB_OK", "TRUE")
    monkeypatch.setattr(
        tensor_network_contraction,
        "_jax_available_local_devices",
        lambda: (),
    )
    circuit = fq.Circuit(3)
    circuit.h(0).cx(0, 2)
    sliced_label = _first_internal_tn_label(circuit)

    with pytest.raises(RuntimeError, match="requires at least 2 local JAX devices"):
        run_jax_sharded_tensor_network(
            circuit,
            world_size=2,
            sliced_labels=(sliced_label,),
            collective_backend="pmap",
        )


def test_jax_sharded_tensor_network_pmap_compute_fails_closed_without_devices(
    monkeypatch,
):
    monkeypatch.setenv("KMP_DUPLICATE_LIB_OK", "TRUE")
    monkeypatch.setattr(
        tensor_network_contraction,
        "_jax_available_local_devices",
        lambda: (),
    )
    circuit = fq.Circuit(3)
    circuit.h(0).cx(0, 2)
    sliced_label = _first_internal_tn_label(circuit)

    with pytest.raises(
        RuntimeError, match="slice compute requires at least 2 local JAX devices"
    ):
        run_jax_sharded_tensor_network(
            circuit,
            world_size=2,
            sliced_labels=(sliced_label,),
            compute_backend="pmap",
            collective_backend="pmap",
        )


def test_jax_sliced_tensor_network_reverse_mode_matches_single_rank_sliced(monkeypatch):
    monkeypatch.setenv("KMP_DUPLICATE_LIB_OK", "TRUE")
    circuit = fq.Circuit(3)
    circuit.h(0).cx(0, 2).ry(1, theta=torch.tensor(0.2))
    sliced_label = _first_internal_tn_label(circuit)

    sharded = jax_sliced_tensor_network_value_and_grad(
        circuit,
        world_size=2,
        sliced_labels=(sliced_label,),
        observable="z_sum",
    )
    single_rank = jax_sliced_tensor_network_value_and_grad(
        circuit,
        world_size=1,
        sliced_labels=(sliced_label,),
        observable="z_sum",
    )
    summary = sharded.summary()

    assert isinstance(sharded, JAXSlicedTensorNetworkGradientResult)
    assert summary["gradient_execution"] == "jax_reverse_mode_sliced_contraction"
    assert summary["distribution_semantics"] == "sharded_across_ranks"
    assert summary["gradient_target"] == "tensor_network_node_tensors"
    assert "parameter_gate_pullback_pending" in summary["gradient_blockers"]
    assert torch.allclose(sharded.torch_value(), single_rank.torch_value(), atol=1e-6)
    sharded_grads = sharded.torch_node_gradients()
    single_grads = single_rank.torch_node_gradients()
    assert len(sharded_grads) == len(single_grads)
    for left, right in zip(sharded_grads, single_grads):
        assert torch.allclose(left, right, atol=1e-5)


def test_jax_sliced_tensor_network_parameter_gradient_matches_jax_tn_kernel(
    monkeypatch,
):
    monkeypatch.setenv("KMP_DUPLICATE_LIB_OK", "TRUE")
    params = torch.tensor([0.2, -0.4, 0.3], dtype=torch.float32, requires_grad=True)

    def build(theta):
        circuit = fq.Circuit(3)
        circuit.ry(0, theta=theta[0])
        circuit.rx(1, theta=theta[1])
        circuit.cx(0, 2)
        circuit.rzz(1, 2, theta=theta[2])
        return circuit

    sliced_label = _first_internal_tn_expectation_label(build(params.detach()))
    sharded = jax_sliced_tensor_network_parameter_value_and_grad(
        build,
        params,
        n_wires=3,
        world_size=2,
        sliced_labels=(sliced_label,),
        observable="z_sum",
        jit=False,
    )
    kernel = compile_quantum_kernel(
        build,
        params.detach(),
        n_wires=3,
        mode="tensor_network",
        observable="z_sum",
        jit=False,
    )
    reference_value = kernel(params)
    reference_value.backward()
    summary = sharded.summary()

    assert isinstance(sharded, JAXSlicedTensorNetworkParameterGradientResult)
    assert summary["gradient_target"] == "parameterized_gate_tensors"
    assert (
        summary["gradient_execution"] == "jax_reverse_mode_sliced_parameter_contraction"
    )
    assert "parameter_gate_pullback_pending" not in summary["gradient_blockers"]
    assert summary["distribution_semantics"] == "sharded_across_ranks"
    assert torch.allclose(
        sharded.torch_value(like=params), reference_value.detach(), atol=1e-5
    )
    assert torch.allclose(
        sharded.torch_gradient(like=params), params.grad.detach(), atol=1e-5
    )


def test_jax_sliced_tensor_network_parameter_gradient_pmap_compute_fails_closed_without_devices(
    monkeypatch,
):
    monkeypatch.setenv("KMP_DUPLICATE_LIB_OK", "TRUE")
    monkeypatch.setattr(
        tensor_network_contraction,
        "_jax_available_local_devices",
        lambda: (),
    )
    params = torch.tensor([0.2, -0.4], dtype=torch.float32)

    def build(theta):
        circuit = fq.Circuit(3)
        circuit.ry(0, theta=theta[0])
        circuit.rx(1, theta=theta[1])
        circuit.cx(0, 2)
        return circuit

    sliced_label = _first_internal_tn_expectation_label(build(params))
    with pytest.raises(
        RuntimeError, match="slice compute requires at least 2 local JAX devices"
    ):
        jax_sliced_tensor_network_parameter_value_and_grad(
            build,
            params,
            n_wires=3,
            world_size=2,
            sliced_labels=(sliced_label,),
            compute_backend="pmap",
            collective_backend="pmap",
        )


def test_local_mps_fast_path_does_not_inherit_distributed_mps_readiness_metadata():
    circuit = fq.Circuit(3)
    circuit.h(0).cx(0, 1)

    summary = run_advanced(circuit, mode="mps", max_bond=4).summary()

    assert summary["state_mode"] == "mps"
    assert "mps_backward_readiness_gate" not in summary
    assert "mps_backward_readiness_status" not in summary
    assert "mps_runtime_summary" not in summary
    assert "mps_runtime_blockers" not in summary
    assert "mps_backward_resource_evidence" not in summary
    assert "mps_backward_memory_plan" not in summary
    assert "mps_backward_communication_plan" not in summary


def test_mps_backward_resource_evidence_helper_remains_internal_api():
    assert hasattr(mps_evidence, "_build_mps_backward_resource_evidence")
    assert not hasattr(mps_shards, "_build_mps_backward_resource_evidence")
    assert not hasattr(fq, "_build_mps_backward_resource_evidence")


def test_mps_accelerator_backward_evidence_rejects_multi_process_probe(monkeypatch):
    class AcceleratorDevice:
        platform = "gpu"
        device_kind = "test-accelerator"

    class MultiProcessJAX:
        @staticmethod
        def local_devices():
            return (AcceleratorDevice(), AcceleratorDevice())

        @staticmethod
        def process_count():
            return 2

    monkeypatch.setattr(
        mps_evidence,
        "_require_jax",
        lambda: (MultiProcessJAX(), object()),
    )

    with pytest.raises(
        RuntimeError,
        match="supports single-node local JAX devices only",
    ):
        mps_evidence._collect_jax_mps_accelerator_backward_evidence()


def test_mps_accelerator_backward_evidence_collector_remains_internal_api():
    assert hasattr(mps_evidence, "_collect_jax_mps_accelerator_backward_evidence")
    assert not hasattr(fq, "_collect_jax_mps_accelerator_backward_evidence")


def test_mps_accelerator_probe_classification_is_development_only():
    classification = mps_evidence._mps_accelerator_probe_classification()

    assert classification == {
        "claim_evidence_type": "development_smoke",
        "artifact_classification": "measured_accelerator_probe",
        "observation_status": "measured_accelerator_probe_observed",
        "memory_status": "probe_array_footprint_estimate",
    }


def test_mps_accelerator_backward_evidence_rejects_cpu_only_devices(monkeypatch):
    class CPUDevice:
        platform = "cpu"

    class CPUOnlyJAX:
        @staticmethod
        def local_devices():
            return (CPUDevice(),)

    monkeypatch.setattr(
        mps_evidence,
        "_require_jax",
        lambda: (CPUOnlyJAX(), object()),
    )

    with pytest.raises(
        RuntimeError,
        match="requires at least two local non-CPU JAX devices",
    ):
        mps_evidence._collect_jax_mps_accelerator_backward_evidence()


def test_minimal_mps_sharded_backward_skeleton_executes_without_full_replay():
    parameters = torch.tensor([0.2, -0.4], dtype=torch.float64)

    summary = mps_backward._execute_minimal_mps_sharded_backward(
        parameters,
        execution_backend="cpu",
    )

    expected_value = torch.cos(parameters[0]) * torch.cos(parameters[1])
    expected_gradient = torch.stack(
        (
            -torch.sin(parameters[0]) * torch.cos(parameters[1]),
            -torch.cos(parameters[0]) * torch.sin(parameters[1]),
        )
    )
    assert summary["mps_backward_execution"] == "executed"
    assert (
        summary["execution_classification"]
        == "cpu_local_distributed_development_execution"
    )
    assert summary["claim_evidence_type"] == "development_smoke"
    assert summary["distribution_semantics"] == "sharded_across_ranks"
    assert summary["mps_backward_distribution_semantics"] == ("sharded_across_ranks")
    assert summary["supported_gate_set"] == ("ry",)
    assert summary["fixed_bond_dimension"] == 1
    assert summary["value"] == pytest.approx(float(expected_value))
    assert summary["gradient"] == pytest.approx(
        tuple(float(item) for item in expected_gradient)
    )
    assert len(summary["site_shard_ownership"]) == 2
    assert len(summary["bond_shard_ownership"]) == 1
    assert len(summary["boundary_adjoint_exchange"]["records"]) == 2
    assert len(summary["parameter_gradient_ownership"]) == 2
    assert summary["mps_backward_memory_plan"]["status"] == ("development_measured")
    assert summary["mps_backward_communication_plan"]["status"] == (
        "development_executed"
    )
    assert summary["full_mps_reconstruction_count"] == 0
    assert summary["replicated_mps_autograd"] is False
    assert summary["statevector_fallback"] is False
    assert summary["fallback_semantics"] == "none"
    assert summary["distributed_evidence_contract"]["contract_version"] == (
        "distributed_evidence_contract_v1"
    )
    runtime = summary["mps_runtime_summary"]
    assert runtime["status"] == "blocked"
    assert runtime["evidence_status"]["boundary_adjoint_exchange"] == "executed"
    assert runtime["evidence_status"]["parameter_gradient_ownership"] == "available"
    assert runtime["evidence_status"]["backward_memory"] == "development_measured"
    assert runtime["evidence_status"]["backward_communication"] == "executed"
    assert runtime["evidence_status"]["optimizer_update_ownership"] == "pending"
    assert (
        "mps_minimal_backward_skeleton_not_full_executor"
        in summary["mps_runtime_blockers"]
    )
    assert summary["mps_backward_readiness_gate"]["fail_closed"] is True
    assert summary["scalability_claim_allowed"] is False
    with pytest.raises(DistributedScalabilityError):
        require_distributed_scalability(summary)


@pytest.mark.parametrize(
    ("parameters", "execution_backend", "message"),
    (
        ([0.2], "cpu", "exactly two rank-owned RY parameters"),
        ([0.2, -0.4, 0.1], "cpu", "exactly two rank-owned RY parameters"),
        ([0.2, -0.4], "unknown", "execution_backend must be"),
    ),
)
def test_minimal_mps_sharded_backward_skeleton_rejects_unsupported_shape_or_backend(
    parameters,
    execution_backend,
    message,
):
    with pytest.raises((ValueError, RuntimeError), match=message):
        mps_backward._execute_minimal_mps_sharded_backward(
            parameters,
            execution_backend=execution_backend,
        )


def test_mps_owner_rank_parameter_vjp_executes_one_and_same_shard_two_site_gates():
    parameters = torch.tensor([0.23, -0.41], dtype=torch.float64)

    summary = mps_pullbacks._execute_minimal_mps_owner_rank_parameter_vjp(
        parameters,
        gate_kinds=("one_site_ry", "same_shard_two_site_rxx"),
    )

    expected_value = torch.cos(parameters[0]) * torch.cos(parameters[1])
    expected_gradient = torch.stack(
        (
            -torch.sin(parameters[0]) * torch.cos(parameters[1]),
            -torch.cos(parameters[0]) * torch.sin(parameters[1]),
        )
    )
    assert summary["owner_rank_vjp_execution"] == "executed"
    assert summary["value"] == pytest.approx(float(expected_value), abs=1e-12)
    assert summary["gradient"] == pytest.approx(
        tuple(float(item) for item in expected_gradient),
        abs=1e-6,
    )
    records = summary["parameter_gradient_ownership"]
    assert tuple(record["gate_kind"] for record in records) == (
        "one_site_ry",
        "same_shard_two_site_rxx",
    )
    assert tuple(record["parameter_id"] for record in records) == (
        "theta_0",
        "theta_1",
    )
    assert all(record["vjp_execution_status"] == "executed" for record in records)
    assert all(
        record["parameter_owner_rank"] == record["gradient_owner_rank"]
        for record in records
    )
    assert records[0]["local_tensor_shape"] == (1, 2, 1)
    assert records[1]["local_tensor_shape"] == (1, 2, 2, 1)
    assert all(record["local_tensor_dtype"] == "complex128" for record in records)
    assert summary["non_owner_gradient_writes"] == ()
    assert summary["replicated_mps_autograd"] is False
    assert summary["full_mps_reconstruction_count"] == 0
    assert summary["statevector_fallback"] is False
    assert summary["scalability_claim_allowed"] is False
    assert (
        summary["distributed_evidence_contract"]["distribution_semantics"]
        == "sharded_across_ranks"
    )
    assert summary["mps_backward_readiness_status"] == "blocked"
    with pytest.raises(DistributedScalabilityError):
        require_distributed_scalability(summary)


def test_mps_owner_rank_parameter_vjp_rejects_boundary_crossing_gate():
    with pytest.raises(ValueError, match="boundary-crossing gates are unsupported"):
        mps_pullbacks._execute_minimal_mps_owner_rank_parameter_vjp(
            [0.2, -0.4],
            gate_kinds=("one_site_ry", "boundary_two_site_rxx"),
        )


def test_mps_owner_rank_parameter_vjp_helper_remains_internal_api():
    assert hasattr(mps_pullbacks, "_execute_minimal_mps_owner_rank_parameter_vjp")
    assert not hasattr(fq, "_execute_minimal_mps_owner_rank_parameter_vjp")


def test_mps_boundary_gate_adjoint_pullback_executes_without_full_replay():
    theta = 0.37
    summary = mps_pullbacks._execute_minimal_mps_boundary_gate_adjoint_pullback(theta)

    assert summary["boundary_gate_pullback_execution"] == "executed"
    theta_tensor = torch.tensor(theta, dtype=torch.float64)
    assert summary["value"] == pytest.approx(float(torch.cos(theta_tensor)), abs=1e-10)
    assert summary["gradient"] == pytest.approx(
        (-float(torch.sin(theta_tensor)),), abs=1e-10
    )
    assert len(summary["boundary_pullback_records"]) == 2
    assert summary["boundary_adjoint_exchange_evidence"]["valid"] is True
    assert (
        summary["boundary_adjoint_exchange_evidence"]["status"] == "local_cpu_executed"
    )
    assert {record["direction"] for record in summary["boundary_pullback_records"]} == {
        "left_to_right",
        "right_to_left",
    }
    assert all(
        record["communication_bytes"] > 0
        for record in summary["boundary_pullback_records"]
    )
    assert all(
        record["pullback_execution_status"] == "executed"
        for record in summary["boundary_pullback_records"]
    )
    ownership = summary["parameter_gradient_ownership"][0]
    assert ownership["parameter_owner_rank"] == ownership["gradient_owner_rank"] == 0
    assert ownership["touched_ranks"] == (0, 1)
    assert summary["non_owner_gradient_writes"] == ()
    assert summary["full_mps_reconstruction_count"] == 0
    assert summary["replicated_mps_autograd"] is False
    assert summary["statevector_fallback"] is False
    assert summary["scalability_claim_allowed"] is False


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"communication_route": ""}, "communication route"),
        ({"left_owner_rank": 1, "right_owner_rank": 0}, "adjacent owner ranks"),
    ),
)
def test_mps_boundary_gate_adjoint_pullback_fails_closed(kwargs, message):
    with pytest.raises(ValueError, match=message):
        mps_pullbacks._execute_minimal_mps_boundary_gate_adjoint_pullback(0.2, **kwargs)


def test_mps_boundary_gate_adjoint_pullback_remains_internal_api():
    assert hasattr(mps_pullbacks, "_execute_minimal_mps_boundary_gate_adjoint_pullback")
    assert not hasattr(fq, "_execute_minimal_mps_boundary_gate_adjoint_pullback")


def test_mps_canonicalization_and_exact_truncation_pullback_evidence():
    theta = 0.37
    summary = (
        mps_canonicalization._execute_minimal_mps_canonicalization_truncation_pullback(
            theta,
            truncation_mode="exact",
        )
    )

    canonical = summary["canonicalization_backward_strategy"]
    truncation = summary["truncation_gradient_metadata"]
    assert canonical["status"] == "executed"
    assert canonical["classification"] == "exact"
    assert canonical["pullback_execution_status"] == "executed"
    assert canonical["left_owner_rank"] == 0
    assert canonical["right_owner_rank"] == 1
    expected_gradient = -torch.sin(
        torch.tensor(theta, dtype=torch.float64)
    ) + 0.175 * torch.cos(torch.tensor(theta, dtype=torch.float64))
    assert summary["canonicalization_gradient"] == pytest.approx(
        float(expected_gradient), abs=1e-10
    )
    assert truncation["status"] == "exact"
    assert truncation["discarded_weight"] == 0.0
    assert truncation["retained_bond_dimension"] == 2
    assert truncation["exact_gradient_claim_allowed"] is True
    assert summary["full_mps_reconstruction_count"] == 0
    assert summary["replicated_mps_autograd"] is False
    assert summary["statevector_fallback"] is False
    assert summary["scalability_claim_allowed"] is False


def test_mps_approximate_truncation_reports_discarded_weight_and_fails_closed():
    summary = (
        mps_canonicalization._execute_minimal_mps_canonicalization_truncation_pullback(
            0.71,
            truncation_mode="approximate",
            cutoff=1e-6,
        )
    )

    truncation = summary["truncation_gradient_metadata"]
    assert truncation["status"] == "approximate"
    assert truncation["discarded_weight"] > 0.0
    assert truncation["retained_bond_dimension"] == 1
    assert truncation["exact_gradient_claim_allowed"] is False
    assert truncation["gradient_value"] is not None
    assert truncation["gradient_value"] != pytest.approx(
        summary["canonicalization_gradient"], abs=1e-10
    )
    assert "mps_truncation_pullback_approximate_not_exact" in summary["blockers"]
    assert summary["mps_backward_readiness_status"] == "blocked"


@pytest.mark.parametrize(
    ("mode", "blocker"),
    (
        ("blocked", "mps_truncation_pullback_unsupported"),
        ("non_differentiable", "mps_truncation_selection_non_differentiable"),
    ),
)
def test_mps_unsupported_truncation_pullback_is_explicitly_blocked(mode, blocker):
    summary = (
        mps_canonicalization._execute_minimal_mps_canonicalization_truncation_pullback(
            0.4,
            truncation_mode=mode,
        )
    )

    metadata = summary["truncation_gradient_metadata"]
    assert metadata["status"] == mode
    assert metadata["pullback_execution_status"] == "blocked"
    assert metadata["exact_gradient_claim_allowed"] is False
    assert blocker in summary["blockers"]
    assert summary["scalability_claim_allowed"] is False


def test_mps_canonicalization_truncation_helper_remains_internal_api():
    assert hasattr(
        mps_canonicalization,
        "_execute_minimal_mps_canonicalization_truncation_pullback",
    )
    assert not hasattr(
        fq,
        "_execute_minimal_mps_canonicalization_truncation_pullback",
    )


def test_minimal_mps_sharded_optimizer_step_preserves_owner_rank_semantics():
    parameters = torch.tensor([0.23, -0.41], dtype=torch.float64)
    learning_rate = 0.07
    summary = mps_canonicalization._execute_minimal_mps_sharded_optimizer_step(
        parameters,
        learning_rate=learning_rate,
        execution_backend="cpu",
    )

    expected_gradient = torch.stack(
        (
            -torch.sin(parameters[0]) * torch.cos(parameters[1]),
            -torch.cos(parameters[0]) * torch.sin(parameters[1]),
        )
    )
    expected_parameters = parameters - learning_rate * expected_gradient
    assert summary["updated_parameters"] == pytest.approx(
        tuple(float(item) for item in expected_parameters), abs=1e-12
    )
    assert summary["training_step_count"] == 1
    assert summary["optimizer_update_semantics"] == "sharded_across_ranks"
    assert summary["optimizer_update_ownership_semantics"] == "sharded_across_ranks"
    records = summary["optimizer_update_ownership"]
    assert len(records) == 2
    assert all(
        record["parameter_owner_rank"]
        == record["gradient_owner_rank"]
        == record["optimizer_state_owner_rank"]
        == record["update_owner_rank"]
        == record["rank"]
        for record in records
    )
    assert all(
        record["writeback_route"] == "rank_local_parameter_owner_writeback"
        for record in records
    )
    assert summary["non_owner_update_writes"] == ()
    gate_checks = summary["mps_backward_readiness_gate"]["checks"]
    assert gate_checks["optimizer_ownership_aligned"] is True
    assert gate_checks["optimizer_training_steps_executed"] is True
    assert not any(
        "optimizer_update_ownership_pending" in blocker
        for blocker in summary["mps_backward_readiness_blockers"]
    )
    assert summary["full_mps_reconstruction_count"] == 0
    assert summary["replicated_mps_autograd"] is False
    assert summary["statevector_fallback"] is False
    assert summary["scalability_claim_allowed"] is False


@pytest.mark.parametrize("learning_rate", (0.0, -0.1, float("inf")))
def test_minimal_mps_sharded_optimizer_step_rejects_invalid_learning_rate(
    learning_rate,
):
    with pytest.raises(ValueError, match="positive finite"):
        mps_canonicalization._execute_minimal_mps_sharded_optimizer_step(
            [0.2, -0.4],
            learning_rate=learning_rate,
            execution_backend="cpu",
        )


def test_minimal_mps_sharded_optimizer_step_remains_internal_api():
    assert hasattr(mps_canonicalization, "_execute_minimal_mps_sharded_optimizer_step")
    assert not hasattr(fq, "_execute_minimal_mps_sharded_optimizer_step")


def test_minimal_mps_backward_reports_measured_runtime_resources():
    summary = mps_backward._execute_minimal_mps_sharded_backward(
        [0.2, -0.4], execution_backend="cpu"
    )
    evidence = summary["mps_measured_runtime_evidence"]

    assert evidence["status"] == "measured"
    assert evidence["valid"] is True
    assert evidence["execution_scope"] == "development_cpu"
    assert evidence["allocator_peak_measured"] is False
    assert len(evidence["rank_memory"]) == 2
    assert len(evidence["communication_events"]) == 2
    assert all(record["forward_tensor_bytes"] > 0 for record in evidence["rank_memory"])
    assert all(
        record["backward_adjoint_bytes"] > 0 for record in evidence["rank_memory"]
    )
    assert all(
        event["elapsed_time_seconds"] > 0 for event in evidence["communication_events"]
    )
    assert all(
        event["timing_scope"] == "independent_event_wall_clock"
        for event in evidence["communication_events"]
    )
    assert evidence["total_communication_time_seconds"] == pytest.approx(
        sum(event["elapsed_time_seconds"] for event in evidence["communication_events"])
    )
    assert evidence["total_communication_bytes"] == summary["communication_bytes"]


def test_minimal_mps_optimizer_measures_optimizer_state_and_peak_memory():
    summary = mps_canonicalization._execute_minimal_mps_sharded_optimizer_step(
        [0.2, -0.4], execution_backend="cpu"
    )
    evidence = summary["mps_measured_runtime_evidence"]

    assert evidence["valid"] is True
    assert all(
        record["optimizer_state_bytes"] > 0 for record in evidence["rank_memory"]
    )
    assert all(
        record["optimizer_update_buffer_bytes"] > 0
        for record in evidence["rank_memory"]
    )


@pytest.mark.parametrize(
    ("section", "field", "blocker_fragment"),
    (
        ("memory", "forward_tensor_bytes", "memory_missing"),
        ("communication", "elapsed_time_seconds", "communication_missing"),
    ),
)
def test_minimal_mps_measured_runtime_evidence_fails_closed(
    section, field, blocker_fragment
):
    summary = mps_backward._execute_minimal_mps_sharded_backward(
        [0.2, -0.4], execution_backend="cpu"
    )
    evidence = summary["mps_measured_runtime_evidence"]
    memory = [dict(record) for record in evidence["rank_memory"]]
    events = [dict(record) for record in evidence["communication_events"]]
    target = memory[0] if section == "memory" else events[0]
    target.pop(field)

    malformed = mps_evidence._summarize_minimal_mps_measured_runtime_evidence(
        memory,
        events,
        execution_scope="development_cpu",
    )

    assert malformed["status"] == "blocked"
    assert malformed["valid"] is False
    assert any(blocker_fragment in blocker for blocker in malformed["blockers"])
