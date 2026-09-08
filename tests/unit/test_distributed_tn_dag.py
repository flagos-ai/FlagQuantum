"""Contracts for the planning-only distributed tensor-network DAG."""

from __future__ import annotations

from dataclasses import asdict, replace

import pytest
import torch

import flagquantum as fq
import flagquantum.runtime.executors.tensor_network.distributed_dag as distributed_dag
import flagquantum.runtime.executors.tensor_network.distributed_execution as distributed_execution
from flagquantum.runtime.executors.tensor_network.adjoint_layout import (
    plan_tn_adjoint_layouts,
    validate_tn_adjoint_tensors,
)
from flagquantum.runtime.executors.tensor_network.checkpointing import (
    execute_checkpointed_tn_reverse_dag,
    execute_tn_forward_with_checkpoint_tape,
    plan_tn_checkpoints,
)
from flagquantum.runtime.executors.tensor_network.compiled_execution import (
    compile_tn_forward_schedule,
    compile_tn_reverse_schedule,
    execute_compiled_tn_forward_with_tape,
    execute_compiled_tn_reverse_dag,
)
from flagquantum.runtime.executors.tensor_network.distributed_dag import (
    DistributedTNContractionDAG,
    DistributedTNContractionRecord,
    DistributedTNShard,
    DistributedTNValueLayout,
    plan_distributed_tn_contraction_dag,
    shard_distributed_tn_value_layout,
    with_sharded_tn_input,
    with_sharded_tn_intermediate,
)
from flagquantum.runtime.executors.tensor_network.distributed_execution import (
    execute_distributed_tn_contraction_dag,
    execute_sharded_tn_dag_operation,
    prepare_sharded_tn_dag_operation,
    required_local_tn_inputs,
)
from flagquantum.runtime.executors.tensor_network.dynamic_reverse import (
    plan_dynamic_tn_reverse_segment,
)
from flagquantum.runtime.executors.tensor_network.joint_planning import (
    DistributedTNWorkingSetPolicy,
    plan_joint_distributed_tn_execution,
)
from flagquantum.runtime.executors.tensor_network.memory_evidence import (
    build_distributed_tn_memory_evidence,
    require_distributed_tn_memory_evidence,
)
from flagquantum.runtime.executors.tensor_network.multi_axis_sharding import (
    partition_tn_tensor_for_multi_axis_shard,
    plan_multi_axis_tn_layout,
    plan_multi_axis_tn_peak_sharding,
)
from flagquantum.runtime.executors.tensor_network.partial_mesh import (
    partition_tn_tensor_for_partial_mesh,
    plan_partial_mesh_tn_layout,
    plan_partial_mesh_tn_redistribution,
)
from flagquantum.runtime.executors.tensor_network.redistribution import (
    assemble_tn_redistribution_shard,
    execute_distributed_tn_redistribution,
    pack_tn_redistribution_block,
    plan_distributed_tn_redistribution,
    plan_multi_axis_tn_redistribution,
)
from flagquantum.runtime.executors.tensor_network.reverse_dag import (
    execute_explicit_tn_reverse_dag,
    execute_tn_forward_with_tape,
    plan_explicit_tn_reverse_dag,
)
from flagquantum.runtime.executors.tensor_network.sharded_kernels import (
    combine_contracted_shards,
    combine_output_shards,
    contract_pair_for_contracted_shard,
    contract_pair_for_output_shard,
    execute_pre_sharded_pair_contraction,
    partition_tn_tensor_for_shard,
    select_sharded_pair_mode,
)
from flagquantum.runtime.executors.tensor_network.sliced_reverse import (
    execute_sliced_tn_explicit_reverse,
)
from flagquantum.simulation.tensor_network.entrypoints import (
    build_tensor_network,
    build_tensor_network_expectation,
)
from flagquantum.simulation.tensor_network.models import TensorNetworkContractionPlan


def _plan() -> TensorNetworkContractionPlan:
    circuit = fq.Circuit(4)
    circuit.h(0).cx(0, 3).ry(1, theta=0.2).rzz(2, 3, theta=-0.4)
    return build_tensor_network(circuit)


def test_distributed_tn_dag_is_deterministic_and_planning_only():
    plan = _plan()

    first = plan_distributed_tn_contraction_dag(
        plan, world_size=4, small_tensor_replication_bytes=0
    )
    second = plan_distributed_tn_contraction_dag(
        plan, world_size=4, small_tensor_replication_bytes=0
    )

    assert first.identity == second.identity
    assert first.summary() == second.summary()
    assert first.summary()["planning_only"] is True
    assert first.summary()["scalability_claim_allowed"] is False
    assert first.summary()["distribution_semantics"] == "planned_tensor_ownership"
    assert len(first.operations) == len(plan.nodes) - 1
    assert first.output_value_id == first.operations[-1].output_value_id


def test_distributed_tn_dag_accepts_native_quality_path():
    circuit = fq.Circuit(4)
    circuit.h(0).cx(0, 1).ry(2, theta=0.3).cx(2, 3)
    expectation = build_tensor_network_expectation(circuit, z=[0, 3])
    dag = plan_distributed_tn_contraction_dag(
        expectation,
        world_size=1,
        objective="quality",
    )

    dag.validate()
    assert dag.objective == "quality"
    assert len(dag.operations) == len(expectation.nodes) - 1
    assert tuple(operation.estimated_cost for operation in dag.operations) == tuple(
        step.estimated_cost for step in expectation.quality_greedy_path()
    )
    local_inputs = {
        f"input:{index}": node.tensor for index, node in enumerate(expectation.nodes)
    }
    result = execute_distributed_tn_contraction_dag(dag, local_inputs)
    assert torch.allclose(
        result.value,
        expectation.contract(strategy="quality_greedy"),
        atol=1e-6,
    )
    multistart = plan_distributed_tn_contraction_dag(
        expectation,
        world_size=1,
        objective="quality_multistart",
    )
    multistart.validate()
    assert multistart.objective == "quality_multistart"
    multistart_result = execute_distributed_tn_contraction_dag(
        multistart,
        local_inputs,
    )
    assert torch.allclose(multistart_result.value, result.value, atol=1e-6)
    reconfigured = plan_distributed_tn_contraction_dag(
        expectation,
        world_size=1,
        objective="quality_reconfigured",
    )
    reconfigured.validate()
    reconfigured_result = execute_distributed_tn_contraction_dag(
        reconfigured,
        local_inputs,
    )
    assert torch.allclose(reconfigured_result.value, result.value, atol=1e-6)


def test_distributed_tn_dag_records_cross_owner_dependencies():
    dag = plan_distributed_tn_contraction_dag(
        _plan(), world_size=3, small_tensor_replication_bytes=0
    )
    values = {value.value_id: value for value in dag.values}
    operations = {operation.operation_id: operation for operation in dag.operations}

    assert dag.communication_edges
    for edge in dag.communication_edges:
        assert edge.source_rank != edge.destination_rank
        assert edge.estimated_bytes == values[edge.value_id].nbytes
        assert edge.consumer_operation_id in operations
        assert (
            edge.destination_rank in operations[edge.consumer_operation_id].owner_ranks
        )


def test_distributed_tn_dag_replicates_only_values_below_threshold():
    threshold = 10_000
    dag = plan_distributed_tn_contraction_dag(
        _plan(),
        world_size=2,
        small_tensor_replication_bytes=threshold,
    )

    inputs = tuple(value for value in dag.values if value.producer_id is None)
    assert inputs
    assert all(value.nbytes <= threshold for value in inputs)
    assert all(value.semantics == "replicated_small" for value in inputs)
    assert all(value.owner_ranks == (0, 1) for value in inputs)
    assert all(
        value.semantics == "unique_owner"
        for value in dag.values
        if value.producer_id is not None
    )


def test_distributed_tn_dag_rejects_invalid_planning_inputs():
    with pytest.raises(ValueError, match="world_size"):
        plan_distributed_tn_contraction_dag(_plan(), world_size=0)
    with pytest.raises(ValueError, match="threshold"):
        plan_distributed_tn_contraction_dag(
            _plan(), world_size=1, small_tensor_replication_bytes=-1
        )
    with pytest.raises(ValueError, match="objective"):
        plan_distributed_tn_contraction_dag(
            _plan(), world_size=1, objective="unsupported"
        )


def test_distributed_tn_dag_executes_direct_scalar_observable():
    circuit = fq.Circuit(3)
    circuit.h(0).cx(0, 2).ry(1, theta=0.2)
    expectation = build_tensor_network_expectation(circuit, x=[0, 2], z=[1])
    dag = plan_distributed_tn_contraction_dag(expectation, world_size=1)
    local_inputs = {
        f"input:{index}": node.tensor for index, node in enumerate(expectation.nodes)
    }

    result = execute_distributed_tn_contraction_dag(dag, local_inputs)

    assert torch.allclose(
        result.value,
        expectation.contract(strategy="memory_greedy"),
        atol=1e-6,
    )
    assert required_local_tn_inputs(dag, rank=0) == tuple(local_inputs)
    assert result.summary()["full_state_materialization"] is False
    assert result.summary()["silent_statevector_fallback"] is False
    assert result.summary()["distribution_semantics"] == "single_device_fast_path"
    assert result.released_value_count > 0


def test_compiled_forward_schedule_matches_exact_tape_and_reduces_launches():
    circuit = fq.Circuit(6)
    for qubit in range(6):
        circuit.h(qubit)
    circuit.cx(0, 1).cx(2, 3).cx(4, 5)
    expectation = build_tensor_network_expectation(circuit, z=[0, 2, 4])
    dag = plan_distributed_tn_contraction_dag(
        expectation,
        world_size=1,
        objective="quality_multistart",
        small_tensor_replication_bytes=1 << 60,
    )
    inputs = {
        f"input:{index}": node.tensor for index, node in enumerate(expectation.nodes)
    }

    eager = execute_tn_forward_with_tape(dag, inputs)
    schedule = compile_tn_forward_schedule(dag)
    compiled = execute_compiled_tn_forward_with_tape(dag, inputs, schedule)

    assert set(compiled) == set(eager)
    for value_id in eager:
        torch.testing.assert_close(compiled[value_id], eager[value_id])
    assert schedule.operation_count == len(dag.operations)
    assert schedule.kernel_launch_count <= schedule.operation_count

    reverse = plan_explicit_tn_reverse_dag(dag)
    eager_reverse = execute_explicit_tn_reverse_dag(dag, reverse, eager)
    reverse_schedule = compile_tn_reverse_schedule(dag, reverse)
    compiled_reverse = execute_compiled_tn_reverse_dag(
        dag, reverse, compiled, schedule=reverse_schedule
    )
    assert set(compiled_reverse.input_cotangents) == set(eager_reverse.input_cotangents)
    for value_id in eager_reverse.input_cotangents:
        torch.testing.assert_close(
            compiled_reverse.input_cotangents[value_id],
            eager_reverse.input_cotangents[value_id],
        )
    assert reverse_schedule.operation_count == len(reverse.records)
    assert reverse_schedule.kernel_launch_count <= 2 * len(reverse.records)


def test_explicit_reverse_dag_matches_complex_autograd_input_cotangents():
    circuit = fq.Circuit(3)
    circuit.ry(0, theta=0.23).cx(0, 1).rzz(1, 2, theta=-0.37)
    expectation = build_tensor_network_expectation(circuit, z=[0, 2])
    dag = plan_distributed_tn_contraction_dag(
        expectation,
        world_size=1,
        small_tensor_replication_bytes=1 << 60,
    )
    inputs = {
        f"input:{index}": node.tensor.detach().clone().requires_grad_(True)
        for index, node in enumerate(expectation.nodes)
    }
    tape = execute_tn_forward_with_tape(dag, inputs)
    reverse = plan_explicit_tn_reverse_dag(dag)
    repeated = plan_explicit_tn_reverse_dag(dag)
    explicit = execute_explicit_tn_reverse_dag(dag, reverse, tape)
    reference = torch.autograd.grad(
        tape[dag.output_value_id],
        tuple(inputs.values()),
        grad_outputs=torch.ones_like(tape[dag.output_value_id]),
        allow_unused=True,
    )

    reverse.validate(dag)
    assert reverse.identity == repeated.identity
    assert reverse.forward_dag_identity == dag.identity
    assert reverse.summary()["uses_generic_autograd_for_contractions"] is False
    assert explicit.nonfinite_cotangent_count == 0
    for value_id, expected in zip(inputs, reference):
        if expected is None:
            assert value_id not in explicit.input_cotangents
        else:
            torch.testing.assert_close(
                explicit.input_cotangents[value_id],
                expected,
                atol=1e-5,
                rtol=1e-5,
            )


def test_explicit_reverse_tensor_cotangents_chain_to_gate_parameters():
    theta = torch.tensor(0.23, dtype=torch.float64, requires_grad=True)
    phi = torch.tensor(-0.37, dtype=torch.float64, requires_grad=True)
    circuit = fq.Circuit(3)
    circuit.ry(0, theta=theta).cx(0, 1).rzz(1, 2, theta=phi)
    expectation = build_tensor_network_expectation(circuit, z=[0, 2])
    dag = plan_distributed_tn_contraction_dag(
        expectation,
        world_size=1,
        small_tensor_replication_bytes=1 << 60,
    )
    detached_inputs = {
        f"input:{index}": node.tensor.detach().clone()
        for index, node in enumerate(expectation.nodes)
    }
    tape = execute_tn_forward_with_tape(dag, detached_inputs)
    explicit = execute_explicit_tn_reverse_dag(
        dag,
        plan_explicit_tn_reverse_dag(dag),
        tape,
    )
    differentiable_nodes = []
    node_cotangents = []
    for index, node in enumerate(expectation.nodes):
        value_id = f"input:{index}"
        if node.tensor.requires_grad and value_id in explicit.input_cotangents:
            differentiable_nodes.append(node.tensor)
            node_cotangents.append(explicit.input_cotangents[value_id])
    explicit_parameters = torch.autograd.grad(
        tuple(differentiable_nodes),
        (theta, phi),
        grad_outputs=tuple(node_cotangents),
        allow_unused=True,
    )

    theta_reference = theta.detach().clone().requires_grad_(True)
    phi_reference = phi.detach().clone().requires_grad_(True)
    reference_circuit = fq.Circuit(3)
    reference_circuit.ry(0, theta=theta_reference).cx(0, 1).rzz(
        1, 2, theta=phi_reference
    )
    reference_value = build_tensor_network_expectation(
        reference_circuit, z=[0, 2]
    ).contract(strategy="memory_greedy")
    reference_parameters = torch.autograd.grad(
        reference_value,
        (theta_reference, phi_reference),
        grad_outputs=torch.ones_like(reference_value),
    )

    for actual, expected in zip(explicit_parameters, reference_parameters):
        assert actual is not None
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("deferred_parameter_pullback", [False, True])
def test_sliced_explicit_reverse_matches_unsliced_all_parameter_gradients(
    deferred_parameter_pullback: bool,
):
    theta = torch.tensor(0.23, dtype=torch.float64, requires_grad=True)
    phi = torch.tensor(-0.37, dtype=torch.float64, requires_grad=True)
    circuit = fq.Circuit(4)
    circuit.ry(0, theta=theta).cx(0, 3).rzz(1, 2, theta=phi).cx(2, 3)
    expectation = build_tensor_network_expectation(circuit, z=[0, 2])
    baseline = expectation.contraction_profile("quality_multistart")
    target_peak = max(baseline.output_size, baseline.peak_size // 2)
    slicing = expectation.slicing_plan(max_intermediate_size=target_peak)

    result = execute_sliced_tn_explicit_reverse(
        expectation,
        slicing,
        (theta, phi),
        deferred_parameter_pullback=deferred_parameter_pullback,
    )
    reference_value = expectation.contract(strategy="greedy")
    reference_gradients = torch.autograd.grad(reference_value.real, (theta, phi))

    torch.testing.assert_close(result.value, reference_value, atol=1e-10, rtol=1e-10)
    for actual, expected in zip(result.parameter_gradients, reference_gradients):
        assert actual is not None
        torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-9)
    assert result.slice_count == slicing.n_slices
    assert result.forward_operation_count > 0
    assert result.reverse_operation_count > 0
    assert result.nonfinite_cotangent_count == 0
    assert result.nonfinite_parameter_gradient_count == 0
    assert result.summary()["execution_semantics"] == "single_device_sliced_reverse"


def test_sliced_reverse_consumes_embedded_external_pair_path():
    theta = torch.tensor(0.23, dtype=torch.float64, requires_grad=True)
    circuit = fq.Circuit(3)
    circuit.ry(0, theta=theta).cx(0, 1).cx(1, 2)
    expectation = build_tensor_network_expectation(circuit, z=[0, 2])
    slicing = expectation.slicing_plan(sliced_labels=())
    external = expectation.quality_multistart_path()
    slicing = replace(
        slicing,
        contraction_path=external,
        contraction_path_source="test_external",
    )

    result = execute_sliced_tn_explicit_reverse(
        expectation,
        slicing,
        (theta,),
    )
    reference = expectation.contract(strategy="quality_sliced")
    (reference_gradient,) = torch.autograd.grad(reference.real, (theta,))

    torch.testing.assert_close(result.value, reference)
    torch.testing.assert_close(result.parameter_gradients[0], reference_gradient)
    assert slicing.summary()["contraction_path_source"] == "test_external"
    assert slicing.summary()["contraction_path_steps"] == len(external)
    assert result.summary()["scalability_claim_allowed"] is False


def test_external_path_adaptive_reslicing_preserves_value_and_gradient():
    theta = torch.tensor(0.19, dtype=torch.float64, requires_grad=True)
    circuit = fq.Circuit(4)
    circuit.ry(0, theta=theta).cx(0, 1).cx(1, 2).cx(2, 3)
    expectation = build_tensor_network_expectation(circuit, z=[0, 3])
    base = expectation.slicing_plan(sliced_labels=())
    base = replace(
        base,
        contraction_path=expectation.quality_multistart_path(),
        contraction_path_source="test_external",
    )
    resliced = expectation.reslice_external_plan(base, target_slices=2)

    result = execute_sliced_tn_explicit_reverse(
        expectation,
        resliced,
        (theta,),
    )
    reference = expectation.contract(strategy="quality_sliced")
    (reference_gradient,) = torch.autograd.grad(reference.real, (theta,))

    assert resliced.n_slices >= 2
    assert resliced.contraction_path_source.endswith("+adaptive_reslice")
    torch.testing.assert_close(result.value, reference)
    torch.testing.assert_close(result.parameter_gradients[0], reference_gradient)


def test_sliced_checkpointed_reverse_matches_full_tape_gradients():
    theta = torch.tensor(0.23, dtype=torch.float64, requires_grad=True)
    phi = torch.tensor(-0.37, dtype=torch.float64, requires_grad=True)
    circuit = fq.Circuit(4)
    circuit.ry(0, theta=theta).cx(0, 3).rzz(1, 2, theta=phi).cx(2, 3)
    expectation = build_tensor_network_expectation(circuit, z=[0, 2])
    baseline = expectation.contraction_profile("quality_multistart")
    slicing = expectation.slicing_plan(
        max_intermediate_size=max(
            baseline.output_size,
            baseline.peak_size // 2,
        )
    )

    full = execute_sliced_tn_explicit_reverse(
        expectation,
        slicing,
        (theta, phi),
    )
    checkpointed = execute_sliced_tn_explicit_reverse(
        expectation,
        slicing,
        (theta, phi),
        checkpoint_budget_bytes=0,
    )

    torch.testing.assert_close(checkpointed.value, full.value)
    for actual, expected in zip(
        checkpointed.parameter_gradients,
        full.parameter_gradients,
    ):
        assert actual is not None and expected is not None
        torch.testing.assert_close(actual, expected)
    assert checkpointed.saved_tape_bytes > 0
    assert checkpointed.rematerialized_operation_count > 0


def test_sliced_reverse_uses_full_tape_when_it_fits_checkpoint_budget():
    theta = torch.tensor(0.23, dtype=torch.float64, requires_grad=True)
    circuit = fq.Circuit(3)
    circuit.ry(0, theta=theta).cx(0, 1).cx(1, 2)
    expectation = build_tensor_network_expectation(circuit, z=[0, 2])
    slicing = expectation.slicing_plan(sliced_labels=())

    result = execute_sliced_tn_explicit_reverse(
        expectation,
        slicing,
        (theta,),
        checkpoint_budget_bytes=1 << 30,
    )

    assert result.rematerialized_operation_count == 0


def test_checkpointed_reverse_rematerializes_and_matches_full_tape():
    circuit = fq.Circuit(4)
    circuit.ry(0, theta=0.2).cx(0, 3).rzz(1, 2, theta=-0.4)
    expectation = build_tensor_network_expectation(circuit, z=[0, 2])
    dag = plan_distributed_tn_contraction_dag(expectation, world_size=1)
    inputs = {
        f"input:{index}": node.tensor for index, node in enumerate(expectation.nodes)
    }
    reverse = plan_explicit_tn_reverse_dag(dag)
    full = execute_explicit_tn_reverse_dag(
        dag,
        reverse,
        execute_tn_forward_with_tape(dag, inputs),
    )
    plan = plan_tn_checkpoints(dag, budget_bytes=0)
    tape = execute_tn_forward_with_checkpoint_tape(dag, inputs, plan)
    checkpointed = execute_checkpointed_tn_reverse_dag(dag, reverse, plan, tape)

    assert plan.checkpoint_value_ids == ()
    assert checkpointed.rematerialized_operation_count > 0
    assert set(tape) == set(inputs) | {dag.output_value_id}
    for value_id, cotangent in full.input_cotangents.items():
        torch.testing.assert_close(checkpointed.input_cotangents[value_id], cotangent)


def test_checkpoint_planner_is_deterministic_and_budget_bounded():
    dag = plan_distributed_tn_contraction_dag(
        _plan(), world_size=1, small_tensor_replication_bytes=1 << 20
    )
    intermediate_bytes = [
        value.nbytes
        for value in dag.values
        if value.producer_id is not None and value.value_id != dag.output_value_id
    ]
    budget = max(intermediate_bytes)
    first = plan_tn_checkpoints(dag, budget_bytes=budget)
    second = plan_tn_checkpoints(dag, budget_bytes=budget)

    assert first.identity == second.identity
    assert first.summary() == second.summary()
    assert 0 < first.saved_intermediate_bytes <= budget
    assert first.checkpoint_value_ids
    with pytest.raises(ValueError, match="budget"):
        plan_tn_checkpoints(dag, budget_bytes=-1)


def test_checkpoint_budget_trades_saved_bytes_for_rematerialization():
    plan = _plan()
    dag = plan_distributed_tn_contraction_dag(plan, world_size=1)
    reverse = plan_explicit_tn_reverse_dag(dag)
    inputs = {f"input:{index}": node.tensor for index, node in enumerate(plan.nodes)}
    total = sum(
        value.nbytes
        for value in dag.values
        if value.producer_id is not None and value.value_id != dag.output_value_id
    )
    results = []
    for budget in (0, total):
        checkpoints = plan_tn_checkpoints(dag, budget_bytes=budget)
        tape = execute_tn_forward_with_checkpoint_tape(dag, inputs, checkpoints)
        results.append(
            execute_checkpointed_tn_reverse_dag(dag, reverse, checkpoints, tape)
        )

    assert results[0].saved_tape_bytes < results[1].saved_tape_bytes
    assert results[0].rematerialized_operation_count > 0
    assert results[1].rematerialized_operation_count == 0


def test_adjoint_layout_inherits_forward_mesh_and_validates_local_shapes():
    circuit = fq.Circuit(8)
    for qubit in range(8):
        circuit.ry(qubit, theta=0.1 * (qubit + 1))
    for layer in range(3):
        for qubit in range(layer % 2, 7, 2):
            circuit.cx(qubit, qubit + 1)
    expectation = build_tensor_network_expectation(circuit, z=list(range(8)))
    dag = plan_distributed_tn_contraction_dag(
        expectation,
        world_size=4,
        small_tensor_replication_bytes=1 << 60,
    )
    reverse = plan_explicit_tn_reverse_dag(dag)
    peak = plan_multi_axis_tn_peak_sharding(dag)
    first = plan_tn_adjoint_layouts(dag, reverse, mesh_labels=peak.shard_labels)
    second = plan_tn_adjoint_layouts(dag, reverse, mesh_labels=peak.shard_labels)
    layout = next(value for value in first.values if value.shard_labels)
    valid = torch.empty(layout.local_shape)

    assert first.identity == second.identity
    assert first.summary()["partitioned_value_count"] > 0
    assert first.summary()["fully_replicated_value_count"] > 0
    validate_tn_adjoint_tensors(first, {layout.value_id: valid})
    with pytest.raises(ValueError, match="rank-local shape"):
        validate_tn_adjoint_tensors(
            first, {layout.value_id: torch.empty(layout.logical_shape)}
        )


def test_multi_axis_redistribution_partitions_mesh_intersections():
    base = DistributedTNValueLayout(
        value_id="adjoint:dynamic",
        producer_id="reverse:dynamic",
        labels=(10, 11, 12, 13, 14, 15),
        shape=(2, 2, 2, 2, 2, 2),
        dtype="torch.complex64",
        nbytes=64 * 8,
        semantics="unique_owner",
        owner_ranks=(0,),
    )
    source = plan_multi_axis_tn_layout(base, shard_labels=(10, 11, 12), world_size=8)
    destination = plan_multi_axis_tn_layout(
        base, shard_labels=(13, 14, 15), world_size=8
    )
    first = plan_multi_axis_tn_redistribution(source, destination)
    second = plan_multi_axis_tn_redistribution(source, destination)

    assert first.identity == second.identity
    assert first.total_bytes == base.nbytes
    assert len(first.blocks) == 64
    assert all(block.element_count == 1 for block in first.blocks)
    assert first.summary()["network_transfer_bytes"] == 56 * 8
    assert first.summary()["self_transfer_bytes"] == 8 * 8


def test_joint_planner_binds_mesh_checkpoints_communication_and_remat():
    circuit = fq.Circuit(8)
    for qubit in range(8):
        circuit.ry(qubit, theta=0.1 * (qubit + 1))
    for layer in range(3):
        for qubit in range(layer % 2, 7, 2):
            circuit.cx(qubit, qubit + 1)
    expectation = build_tensor_network_expectation(circuit, z=list(range(8)))
    dag = plan_distributed_tn_contraction_dag(
        expectation,
        world_size=8,
        small_tensor_replication_bytes=1 << 60,
    )
    zero = plan_joint_distributed_tn_execution(dag, checkpoint_budget_local_bytes=0)
    saved = plan_joint_distributed_tn_execution(dag, checkpoint_budget_local_bytes=4096)
    repeated = plan_joint_distributed_tn_execution(
        dag, checkpoint_budget_local_bytes=4096
    )

    assert saved.identity == repeated.identity
    assert saved.mesh_shape == (2, 2, 2)
    assert len(saved.shard_labels) == 3
    assert saved.saved_checkpoint_local_bytes <= 4096
    assert saved.predicted_working_set_local_bytes == max(
        saved.predicted_forward_peak_local_bytes + saved.saved_checkpoint_local_bytes,
        saved.predicted_reverse_peak_local_bytes,
    )
    assert saved.predicted_reverse_peak_local_bytes == (
        saved.predicted_raw_input_local_bytes
        + saved.saved_checkpoint_local_bytes
        + saved.predicted_reverse_cotangent_peak_local_bytes
        + saved.predicted_rematerialization_peak_local_bytes
    )
    assert saved.memory_budget_satisfied is True
    assert saved.checkpoint_value_ids
    assert (
        saved.estimated_rematerialization_cost < zero.estimated_rematerialization_cost
    )
    assert saved.estimated_collective_network_bytes > 0
    with pytest.raises(ValueError, match="budget"):
        plan_joint_distributed_tn_execution(dag, checkpoint_budget_local_bytes=-1)


def test_joint_planner_enforces_total_rank_memory_budget():
    circuit = fq.Circuit(8)
    for qubit in range(8):
        circuit.ry(qubit, theta=0.1 * (qubit + 1))
    for layer in range(3):
        for qubit in range(layer % 2, 7, 2):
            circuit.cx(qubit, qubit + 1)
    expectation = build_tensor_network_expectation(circuit, z=list(range(8)))
    dag = plan_distributed_tn_contraction_dag(
        expectation,
        world_size=8,
        small_tensor_replication_bytes=1 << 60,
    )
    baseline = plan_joint_distributed_tn_execution(dag, checkpoint_budget_local_bytes=0)
    exact = plan_joint_distributed_tn_execution(
        dag,
        checkpoint_budget_local_bytes=1 << 30,
        memory_budget_local_bytes=baseline.predicted_working_set_local_bytes,
    )
    repeated = plan_joint_distributed_tn_execution(
        dag,
        checkpoint_budget_local_bytes=1 << 30,
        memory_budget_local_bytes=baseline.predicted_working_set_local_bytes,
    )

    assert exact.identity == repeated.identity
    assert exact.predicted_working_set_local_bytes <= (exact.memory_budget_local_bytes)
    assert exact.memory_budget_satisfied is True
    with pytest.raises(ValueError, match="memory budget"):
        plan_joint_distributed_tn_execution(
            dag,
            checkpoint_budget_local_bytes=0,
            memory_budget_local_bytes=1,
        )
    with pytest.raises(ValueError, match="memory budget"):
        plan_joint_distributed_tn_execution(
            dag,
            checkpoint_budget_local_bytes=0,
            memory_budget_local_bytes=0,
        )


def test_joint_planner_accounts_for_workspace_buffers_and_headroom():
    circuit = fq.Circuit(4)
    for qubit in range(4):
        circuit.ry(qubit, theta=0.1 * (qubit + 1))
    circuit.cx(0, 1)
    circuit.cx(2, 3)
    expectation = build_tensor_network_expectation(circuit, z=range(4))
    dag = plan_distributed_tn_contraction_dag(
        expectation,
        world_size=2,
        small_tensor_replication_bytes=1 << 60,
    )
    legacy = plan_joint_distributed_tn_execution(dag, checkpoint_budget_local_bytes=0)
    policy = DistributedTNWorkingSetPolicy(
        kernel_workspace_output_multiplier=3.0,
        communication_buffer_output_multiplier=2.0,
        allocator_headroom_fraction=0.25,
        minimum_allocator_headroom_bytes=4096,
    )
    planned = plan_joint_distributed_tn_execution(
        dag,
        checkpoint_budget_local_bytes=0,
        working_set_policy=policy,
    )
    repeated = plan_joint_distributed_tn_execution(
        dag,
        checkpoint_budget_local_bytes=0,
        working_set_policy=policy,
    )

    assert planned.identity == repeated.identity
    assert planned.identity != legacy.identity
    assert planned.predicted_tensor_working_set_local_bytes == (
        legacy.predicted_working_set_local_bytes
    )
    assert planned.predicted_kernel_workspace_local_bytes > 0
    assert planned.predicted_communication_buffer_local_bytes > 0
    assert planned.predicted_allocator_headroom_local_bytes >= 4096
    assert planned.predicted_working_set_local_bytes == sum(
        (
            planned.predicted_tensor_working_set_local_bytes,
            planned.predicted_kernel_workspace_local_bytes,
            planned.predicted_communication_buffer_local_bytes,
            planned.predicted_allocator_headroom_local_bytes,
        )
    )
    assert planned.summary()["working_set_policy"] == asdict(policy)
    with pytest.raises(ValueError, match="headroom fraction"):
        plan_joint_distributed_tn_execution(
            dag,
            checkpoint_budget_local_bytes=0,
            working_set_policy=DistributedTNWorkingSetPolicy(
                allocator_headroom_fraction=1.0
            ),
        )
    with pytest.raises(ValueError, match="memory budget"):
        plan_joint_distributed_tn_execution(
            dag,
            checkpoint_budget_local_bytes=0,
            memory_budget_local_bytes=(planned.predicted_working_set_local_bytes - 1),
            working_set_policy=policy,
        )


def test_joint_memory_evidence_is_deterministic_and_fails_closed():
    circuit = fq.Circuit(4)
    for qubit in range(4):
        circuit.ry(qubit, theta=0.1 * (qubit + 1))
    circuit.cx(0, 1)
    circuit.cx(2, 3)
    expectation = build_tensor_network_expectation(circuit, z=range(4))
    dag = plan_distributed_tn_contraction_dag(
        expectation,
        world_size=2,
        small_tensor_replication_bytes=1 << 60,
    )
    unconstrained = plan_joint_distributed_tn_execution(
        dag, checkpoint_budget_local_bytes=0
    )
    plan = plan_joint_distributed_tn_execution(
        dag,
        checkpoint_budget_local_bytes=0,
        memory_budget_local_bytes=(unconstrained.predicted_working_set_local_bytes * 2),
    )
    predicted = plan.predicted_working_set_local_bytes
    measurements = tuple(
        {
            "rank": rank,
            "cuda_peak_allocated_bytes": predicted,
            "cuda_peak_reserved_bytes": predicted + 64,
            "peak_cached_forward_bytes": (plan.predicted_raw_input_local_bytes),
            "rematerialization_peak_transient_bytes": (
                plan.predicted_rematerialization_peak_local_bytes
            ),
        }
        for rank in range(2)
    )
    first = build_distributed_tn_memory_evidence(plan, measurements)
    repeated = build_distributed_tn_memory_evidence(plan, measurements)

    assert first.identity == repeated.identity
    assert first.passed is True
    assert first.prediction_calibrated is True
    assert first.memory_budget_satisfied is True
    require_distributed_tn_memory_evidence(first)

    underpredicted = tuple(
        dict(
            item,
            cuda_peak_allocated_bytes=predicted * 2,
            cuda_peak_reserved_bytes=predicted * 2 + 64,
        )
        for item in measurements
    )
    failed = build_distributed_tn_memory_evidence(plan, underpredicted)
    assert failed.passed is False
    assert "measured_allocator_reservation_exceeds_prediction_tolerance" in (
        failed.blockers
    )
    with pytest.raises(RuntimeError, match="memory evidence failed"):
        require_distributed_tn_memory_evidence(failed)
    with pytest.raises(ValueError, match="every rank"):
        build_distributed_tn_memory_evidence(plan, measurements[:1])


def test_partial_mesh_layout_partitions_present_and_replicates_absent_labels():
    value = DistributedTNValueLayout(
        value_id="partial:test",
        producer_id="contract:test",
        labels=(10, 11, 12),
        shape=(2, 3, 2),
        dtype="torch.complex64",
        nbytes=2 * 3 * 2 * 8,
        semantics="unique_owner",
        owner_ranks=(0,),
    )
    layout = plan_partial_mesh_tn_layout(
        value,
        mesh_labels=(10, 12, 13),
        mesh_shape=(2, 2, 2),
    )
    logical = torch.arange(12).reshape(2, 3, 2)
    rank_zero = partition_tn_tensor_for_partial_mesh(logical, layout, rank=0)
    replicated_coordinate = partition_tn_tensor_for_partial_mesh(
        logical, layout, rank=1
    )

    assert layout.partitioned_mesh_labels == (10, 12)
    assert layout.replicated_mesh_labels == (13,)
    assert layout.replication_factor == 2
    assert layout.local_shape == (1, 3, 1)
    torch.testing.assert_close(rank_zero, replicated_coordinate)
    destination = plan_partial_mesh_tn_layout(
        value,
        mesh_labels=(12, 13, 14),
        mesh_shape=(2, 2, 2),
    )
    redistribution = plan_partial_mesh_tn_redistribution(layout, destination)
    assert redistribution.canonical_source_rank_count == 4
    assert redistribution.delivered_bytes == (
        value.nbytes * destination.replication_factor
    )


def test_dynamic_reverse_segment_is_deterministic_and_frontier_validated():
    circuit = fq.Circuit(8)
    for qubit in range(8):
        circuit.ry(qubit, theta=0.1 * (qubit + 1))
    for layer in range(3):
        for qubit in range(layer % 2, 7, 2):
            circuit.cx(qubit, qubit + 1)
    expectation = build_tensor_network_expectation(circuit, z=list(range(8)))
    dag = plan_distributed_tn_contraction_dag(
        expectation,
        world_size=8,
        small_tensor_replication_bytes=1 << 60,
    )
    reverse = plan_explicit_tn_reverse_dag(dag)
    first = plan_dynamic_tn_reverse_segment(
        dag,
        reverse,
        start_output_value_id="intermediate:60",
        source_mesh_labels=(20, 17, 26),
        destination_mesh_labels=(25, 13, 9),
        mesh_shape=(2, 2, 2),
        max_records=2,
    )
    second = plan_dynamic_tn_reverse_segment(
        dag,
        reverse,
        start_output_value_id="intermediate:60",
        source_mesh_labels=(20, 17, 26),
        destination_mesh_labels=(25, 13, 9),
        mesh_shape=(2, 2, 2),
        max_records=2,
    )

    assert first.identity == second.identity
    assert tuple(record.reverse_id for record in first.records) == (
        "reverse:60",
        "reverse:59",
    )
    first.validate(dag, reverse)
    assert first.summary()["record_count"] == 2
    assert first.selection_mode == "largest_ready_frontier"


def test_dynamic_reverse_waits_for_every_shared_value_contribution():
    values = (
        DistributedTNValueLayout(
            "input:0",
            None,
            (0, 1),
            (4, 4),
            "torch.float64",
            128,
            "replicated_small",
            (0, 1),
        ),
        DistributedTNValueLayout(
            "input:1",
            None,
            (1, 2),
            (4, 4),
            "torch.float64",
            128,
            "replicated_small",
            (0, 1),
        ),
        DistributedTNValueLayout(
            "input:2",
            None,
            (2, 3),
            (4, 2),
            "torch.float64",
            64,
            "replicated_small",
            (0, 1),
        ),
        DistributedTNValueLayout(
            "input:3",
            None,
            (2, 4),
            (4, 2),
            "torch.float64",
            64,
            "replicated_small",
            (0, 1),
        ),
        DistributedTNValueLayout(
            "intermediate:0",
            "operation:0",
            (0, 2),
            (4, 4),
            "torch.float64",
            128,
            "unique_owner",
            (0,),
        ),
        DistributedTNValueLayout(
            "intermediate:1",
            "operation:1",
            (0, 3),
            (4, 2),
            "torch.float64",
            64,
            "unique_owner",
            (0,),
        ),
        DistributedTNValueLayout(
            "intermediate:2",
            "operation:2",
            (0, 4),
            (4, 2),
            "torch.float64",
            64,
            "unique_owner",
            (0,),
        ),
        DistributedTNValueLayout(
            "intermediate:3",
            "operation:3",
            (3, 4),
            (2, 2),
            "torch.float64",
            32,
            "unique_owner",
            (0,),
        ),
    )
    operations = (
        DistributedTNContractionRecord(
            "operation:0",
            0,
            ("input:0", "input:1"),
            "intermediate:0",
            (0,),
            (0, 2),
            (4, 4),
            1,
            16,
        ),
        DistributedTNContractionRecord(
            "operation:1",
            1,
            ("intermediate:0", "input:2"),
            "intermediate:1",
            (0,),
            (0, 3),
            (4, 2),
            1,
            8,
        ),
        DistributedTNContractionRecord(
            "operation:2",
            2,
            ("intermediate:0", "input:3"),
            "intermediate:2",
            (0,),
            (0, 4),
            (4, 2),
            1,
            8,
        ),
        DistributedTNContractionRecord(
            "operation:3",
            3,
            ("intermediate:1", "intermediate:2"),
            "intermediate:3",
            (0,),
            (3, 4),
            (2, 2),
            1,
            4,
        ),
    )
    dag = DistributedTNContractionDAG(
        version=distributed_dag.TN_DAG_VERSION,
        identity="pending",
        world_size=2,
        objective="memory",
        small_tensor_replication_bytes=4096,
        values=values,
        operations=operations,
        communication_edges=(),
        output_value_id="intermediate:3",
    )
    object.__setattr__(
        dag, "identity", distributed_dag._dag_identity(dag._identity_payload())
    )
    reverse = plan_explicit_tn_reverse_dag(dag)
    segment = plan_dynamic_tn_reverse_segment(
        dag,
        reverse,
        start_output_value_id=dag.output_value_id,
        source_mesh_labels=(5,),
        destination_mesh_labels=(6,),
        mesh_shape=(2,),
        max_records=4,
    )

    assert tuple(record.reverse_id for record in segment.records) == (
        "reverse:3",
        "reverse:2",
        "reverse:1",
        "reverse:0",
    )


def test_distributed_tn_dag_rejects_full_state_output():
    plan = _plan()
    dag = plan_distributed_tn_contraction_dag(plan, world_size=1)
    local_inputs = {
        f"input:{index}": node.tensor for index, node in enumerate(plan.nodes)
    }

    with pytest.raises(RuntimeError, match="full-state"):
        execute_distributed_tn_contraction_dag(
            dag,
            local_inputs,
            max_output_bytes=1,
        )


def test_distributed_tn_dag_rejects_non_owned_or_missing_inputs():
    plan = _plan()
    dag = plan_distributed_tn_contraction_dag(plan, world_size=1)
    local_inputs = {
        f"input:{index}": node.tensor for index, node in enumerate(plan.nodes)
    }
    missing = dict(local_inputs)
    missing.pop(next(iter(missing)))

    with pytest.raises(ValueError, match="missing inputs"):
        execute_distributed_tn_contraction_dag(dag, missing)
    with pytest.raises(ValueError, match="non-owned inputs"):
        execute_distributed_tn_contraction_dag(
            dag,
            {**local_inputs, "input:not-owned": next(iter(local_inputs.values()))},
        )


def test_distributed_tn_value_shards_cover_one_logical_intermediate():
    dag = plan_distributed_tn_contraction_dag(
        _plan(), world_size=4, small_tensor_replication_bytes=0
    )
    candidate = max(
        (value for value in dag.values if value.producer_id is not None),
        key=lambda value: value.nbytes,
    )

    sharded = shard_distributed_tn_value_layout(candidate, world_size=4)

    assert sharded.semantics == "sharded"
    assert len(sharded.owner_ranks) >= 2
    assert sharded.shard_label in sharded.labels
    assert sharded.shard_axis is not None
    assert sharded.shards[0].start == 0
    assert sharded.shards[-1].stop == sharded.shape[sharded.shard_axis]
    assert all(
        left.stop == right.start
        for left, right in zip(sharded.shards, sharded.shards[1:])
    )
    assert sum(shard.nbytes for shard in sharded.shards) == sharded.nbytes
    assert max(shard.nbytes for shard in sharded.shards) < sharded.nbytes


def test_distributed_tn_dag_identity_fails_closed_after_layout_mutation():
    dag = plan_distributed_tn_contraction_dag(
        _plan(), world_size=1, small_tensor_replication_bytes=0
    )
    candidate_index = max(
        (
            index
            for index, value in enumerate(dag.values)
            if value.producer_id is not None
        ),
        key=lambda index: dag.values[index].nbytes,
    )
    sharded = shard_distributed_tn_value_layout(
        dag.values[candidate_index], world_size=2
    )
    mutated = type(dag)(
        version=dag.version,
        identity=dag.identity,
        world_size=dag.world_size,
        objective=dag.objective,
        small_tensor_replication_bytes=dag.small_tensor_replication_bytes,
        values=dag.values[:candidate_index]
        + (sharded,)
        + dag.values[candidate_index + 1 :],
        operations=dag.operations,
        communication_edges=dag.communication_edges,
        output_value_id=dag.output_value_id,
    )

    with pytest.raises(ValueError, match="identity"):
        mutated.validate()


def test_output_label_shards_match_full_pair_contraction():
    left = torch.randn(8, 3, dtype=torch.complex64)
    right = torch.randn(3, 4, dtype=torch.complex64)
    shards = (
        DistributedTNShard(0, 0, 3, (3, 4), 3 * 4 * 8),
        DistributedTNShard(1, 3, 6, (3, 4), 3 * 4 * 8),
        DistributedTNShard(2, 6, 8, (2, 4), 2 * 4 * 8),
    )

    partials = tuple(
        contract_pair_for_output_shard(
            left,
            (0, 1),
            right,
            (1, 2),
            (0, 2),
            shard_label=0,
            shard=shard,
        )
        for shard in shards
    )
    combined = combine_output_shards(partials, (0, 2), shard_label=0)
    reference = torch.einsum("ab,bc->ac", left, right)

    assert torch.allclose(combined, reference, atol=1e-6)
    assert max(partial.numel() for partial in partials) < reference.numel()


def test_contracted_label_shards_match_full_pair_contraction():
    left = torch.randn(5, 7, dtype=torch.complex64)
    right = torch.randn(7, 4, dtype=torch.complex64)
    shards = (
        DistributedTNShard(0, 0, 2, (5, 2), 5 * 2 * 8),
        DistributedTNShard(1, 2, 5, (5, 3), 5 * 3 * 8),
        DistributedTNShard(2, 5, 7, (5, 2), 5 * 2 * 8),
    )

    partials = tuple(
        contract_pair_for_contracted_shard(
            left,
            (0, 1),
            right,
            (1, 2),
            (0, 2),
            contracted_label=1,
            shard=shard,
        )
        for shard in shards
    )
    combined = combine_contracted_shards(partials)
    reference = torch.einsum("ab,bc->ac", left, right)

    assert torch.allclose(combined, reference, atol=1e-6)


def test_sharded_pair_kernels_reject_incompatible_labels():
    left = torch.randn(4, 3, dtype=torch.complex64)
    right = torch.randn(3, 2, dtype=torch.complex64)
    shard = DistributedTNShard(0, 0, 2, (2, 2), 2 * 2 * 8)

    with pytest.raises(ValueError, match="retained"):
        contract_pair_for_output_shard(
            left,
            (0, 1),
            right,
            (1, 2),
            (0, 2),
            shard_label=1,
            shard=shard,
        )
    with pytest.raises(ValueError, match="cannot remain"):
        contract_pair_for_contracted_shard(
            left,
            (0, 1),
            right,
            (1, 2),
            (0, 2),
            contracted_label=0,
            shard=shard,
        )


def test_pre_sharded_pair_executes_without_full_retained_output():
    left = torch.randn(8, 3, dtype=torch.complex64)
    right = torch.randn(3, 4, dtype=torch.complex64)
    shard = DistributedTNShard(0, 2, 5, (3, 4), 3 * 4 * 8)

    result = execute_pre_sharded_pair_contraction(
        left[2:5],
        (0, 1),
        right,
        (1, 2),
        (0, 2),
        mode="retained_output_shard",
        shard_label=0,
        shard=shard,
    )

    assert torch.allclose(
        result.value, torch.einsum("ab,bc->ac", left[2:5], right), atol=1e-6
    )
    assert result.summary()["full_input_materialized"] is False
    assert result.summary()["full_output_materialized"] is False
    assert result.summary()["distribution_semantics"] == (
        "rank_group_sharded_intermediate"
    )


def test_pre_sharded_pair_contracts_full_local_range_in_single_rank():
    left = torch.randn(5, 7, dtype=torch.complex64)
    right = torch.randn(7, 4, dtype=torch.complex64)
    shard = DistributedTNShard(0, 0, 7, (5, 7), 5 * 7 * 8)

    result = execute_pre_sharded_pair_contraction(
        left,
        (0, 1),
        right,
        (1, 2),
        (0, 2),
        mode="contracted_partial_reduce",
        shard_label=1,
        shard=shard,
    )

    assert torch.allclose(
        result.value, torch.einsum("ab,bc->ac", left, right), atol=1e-6
    )
    assert result.collective_count == 0


def test_pre_sharded_pair_rejects_uninitialized_multi_rank_execution():
    left = torch.randn(2, 3, dtype=torch.complex64)
    right = torch.randn(3, 2, dtype=torch.complex64)
    shard = DistributedTNShard(0, 0, 2, (2, 2), 2 * 2 * 8)

    with pytest.raises(RuntimeError, match="torch.distributed"):
        execute_pre_sharded_pair_contraction(
            left,
            (0, 1),
            right,
            (1, 2),
            (0, 2),
            mode="retained_output_shard",
            shard_label=0,
            shard=shard,
            world_size=2,
        )


def test_dag_sharded_intermediate_transform_is_identity_valid_and_deterministic():
    dag = plan_distributed_tn_contraction_dag(
        _plan(), world_size=4, small_tensor_replication_bytes=0
    )
    candidate = max(
        (
            value
            for value in dag.values
            if value.producer_id is not None and value.shape and max(value.shape) >= 2
        ),
        key=lambda value: value.nbytes,
    )

    first = with_sharded_tn_intermediate(dag, candidate.value_id)
    second = with_sharded_tn_intermediate(dag, candidate.value_id)
    transformed = next(
        value for value in first.values if value.value_id == candidate.value_id
    )
    producer = next(
        operation
        for operation in first.operations
        if operation.output_value_id == candidate.value_id
    )

    first.validate()
    assert first.identity == second.identity
    assert first.identity != dag.identity
    assert transformed.semantics == "sharded"
    assert producer.owner_ranks == transformed.owner_ranks
    assert select_sharded_pair_mode(first, producer) == (
        "retained_output_shard",
        transformed.shard_label,
    )


def test_sharded_mode_selection_rejects_unsharded_operation():
    dag = plan_distributed_tn_contraction_dag(_plan(), world_size=2)

    with pytest.raises(ValueError, match="no sharded value"):
        select_sharded_pair_mode(dag, dag.operations[0])


def test_dag_transform_rejects_raw_input_sharding():
    dag = plan_distributed_tn_contraction_dag(_plan(), world_size=2)
    input_value = next(value for value in dag.values if value.producer_id is None)

    with pytest.raises(ValueError, match="input partitioner"):
        with_sharded_tn_intermediate(dag, input_value.value_id)


def test_dag_input_sharding_and_local_partition_reconstruct_original_tensor():
    dag = plan_distributed_tn_contraction_dag(
        _plan(), world_size=4, small_tensor_replication_bytes=0
    )
    input_index, input_layout = max(
        (
            (index, value)
            for index, value in enumerate(dag.values)
            if value.producer_id is None and value.shape and max(value.shape) >= 2
        ),
        key=lambda item: item[1].nbytes,
    )

    transformed = with_sharded_tn_input(dag, input_layout.value_id)
    sharded = transformed.values[input_index]
    original = _plan().nodes[input_index].tensor
    local_parts = tuple(
        partition_tn_tensor_for_shard(
            original,
            input_layout.labels,
            shard_label=sharded.shard_label,
            shard=shard,
        )
        for shard in sharded.shards
    )

    transformed.validate()
    assert transformed.identity != dag.identity
    assert sharded.semantics == "sharded"
    assert torch.equal(
        torch.cat(local_parts, dim=sharded.shard_axis),
        original,
    )
    assert max(part.numel() for part in local_parts) < original.numel()


def test_input_partitioner_rejects_missing_label_and_out_of_range():
    tensor = torch.randn(4, 3)
    valid = DistributedTNShard(0, 0, 2, (2, 3), 2 * 3 * 4)
    invalid = DistributedTNShard(0, 3, 5, (2, 3), 2 * 3 * 4)

    with pytest.raises(ValueError, match="absent"):
        partition_tn_tensor_for_shard(tensor, (0, 1), shard_label=2, shard=valid)
    with pytest.raises(ValueError, match="exceeds"):
        partition_tn_tensor_for_shard(tensor, (0, 1), shard_label=0, shard=invalid)


def test_dag_operation_dispatches_to_retained_output_shard():
    dag = plan_distributed_tn_contraction_dag(
        _plan(), world_size=2, small_tensor_replication_bytes=0
    )
    candidate = next(
        value
        for value in dag.values
        if value.producer_id is not None and value.shape and max(value.shape) >= 2
    )
    sharded_dag = with_sharded_tn_intermediate(dag, candidate.value_id)
    operation = next(
        item
        for item in sharded_dag.operations
        if item.output_value_id == candidate.value_id
    )

    mode, label, shard = prepare_sharded_tn_dag_operation(
        sharded_dag, operation, rank=0
    )

    assert mode == "retained_output_shard"
    assert (
        label
        == next(
            value
            for value in sharded_dag.values
            if value.value_id == candidate.value_id
        ).shard_label
    )
    assert shard.rank == 0


def test_dag_sharded_operation_requires_real_distributed_runtime():
    dag = plan_distributed_tn_contraction_dag(
        _plan(), world_size=2, small_tensor_replication_bytes=0
    )
    candidate = next(
        value
        for value in dag.values
        if value.producer_id is not None and value.shape and max(value.shape) >= 2
    )
    sharded_dag = with_sharded_tn_intermediate(dag, candidate.value_id)
    operation = next(
        item
        for item in sharded_dag.operations
        if item.output_value_id == candidate.value_id
    )
    _, label, shard = prepare_sharded_tn_dag_operation(sharded_dag, operation, rank=0)
    values = {value.value_id: value for value in sharded_dag.values}
    local_inputs = {}
    for value_id in operation.input_value_ids:
        layout = values[value_id]
        shape = list(layout.shape)
        if label in layout.labels:
            shape[layout.labels.index(label)] = shard.stop - shard.start
        local_inputs[value_id] = torch.empty(tuple(shape), dtype=torch.complex64)

    with pytest.raises(RuntimeError, match="torch.distributed"):
        execute_sharded_tn_dag_operation(
            sharded_dag,
            operation,
            local_inputs,
            rank=0,
        )


def test_tn_redistribution_cross_axis_blocks_conserve_bytes():
    base = DistributedTNValueLayout(
        value_id="intermediate:test",
        producer_id="contract:test",
        labels=(10, 11),
        shape=(6, 4),
        dtype="torch.complex64",
        nbytes=6 * 4 * 8,
        semantics="unique_owner",
        owner_ranks=(0,),
    )
    source = shard_distributed_tn_value_layout(base, world_size=3, shard_label=10)
    destination = shard_distributed_tn_value_layout(base, world_size=2, shard_label=11)

    plan = plan_distributed_tn_redistribution(source, destination)
    repeated = plan_distributed_tn_redistribution(source, destination)

    assert plan.identity == repeated.identity
    assert plan.total_bytes == base.nbytes
    assert sum(block.element_count for block in plan.blocks) == 6 * 4
    assert len(plan.blocks) == len(source.shards) * len(destination.shards)
    assert plan.summary()["network_transfer_bytes"] > 0
    assert all(len(block.global_slices) == 2 for block in plan.blocks)


def test_multi_axis_binary_labels_cover_eight_rank_cartesian_mesh():
    base = DistributedTNValueLayout(
        value_id="intermediate:multi-axis",
        producer_id="contract:multi-axis",
        labels=(10, 11, 12, 13),
        shape=(2, 2, 2, 3),
        dtype="torch.complex64",
        nbytes=2 * 2 * 2 * 3 * 8,
        semantics="unique_owner",
        owner_ranks=(0,),
    )

    layout = plan_multi_axis_tn_layout(
        base,
        shard_labels=(10, 11, 12),
        world_size=8,
    )

    assert layout.world_size == 8
    assert layout.mesh_shape == (2, 2, 2)
    assert len({shard.coordinates for shard in layout.shards}) == 8
    assert sum(shard.nbytes for shard in layout.shards) == base.nbytes
    assert all(shard.local_shape == (1, 1, 1, 3) for shard in layout.shards)
    assert (
        layout.identity
        == plan_multi_axis_tn_layout(
            base,
            shard_labels=(10, 11, 12),
            world_size=8,
        ).identity
    )


def test_multi_axis_local_partials_match_full_contraction():
    left = torch.randn(2, 2, 2, 3, dtype=torch.complex64)
    right = torch.randn(2, 2, 2, 5, dtype=torch.complex64)
    left_layout = DistributedTNValueLayout(
        "left",
        None,
        (0, 1, 2, 3),
        tuple(left.shape),
        str(left.dtype),
        left.numel() * left.element_size(),
        "unique_owner",
        (0,),
    )
    right_layout = DistributedTNValueLayout(
        "right",
        None,
        (0, 1, 2, 4),
        tuple(right.shape),
        str(right.dtype),
        right.numel() * right.element_size(),
        "unique_owner",
        (0,),
    )
    left_multi = plan_multi_axis_tn_layout(
        left_layout, shard_labels=(0, 1, 2), world_size=8
    )
    right_multi = plan_multi_axis_tn_layout(
        right_layout, shard_labels=(0, 1, 2), world_size=8
    )

    partials = tuple(
        torch.einsum(
            "abcd,abce->de",
            partition_tn_tensor_for_multi_axis_shard(left, left_multi, rank=rank),
            partition_tn_tensor_for_multi_axis_shard(right, right_multi, rank=rank),
        )
        for rank in range(8)
    )

    assert torch.allclose(sum(partials), torch.einsum("abcd,abce->de", left, right))
    assert max(partial.numel() for partial in partials) < left.numel()


def test_tn_redistribution_same_axis_uses_only_overlaps():
    base = DistributedTNValueLayout(
        value_id="intermediate:test",
        producer_id="contract:test",
        labels=(10, 11),
        shape=(7, 4),
        dtype="torch.complex64",
        nbytes=7 * 4 * 8,
        semantics="unique_owner",
        owner_ranks=(0,),
    )
    source = shard_distributed_tn_value_layout(base, world_size=3, shard_label=10)
    destination = shard_distributed_tn_value_layout(base, world_size=2, shard_label=10)

    plan = plan_distributed_tn_redistribution(source, destination)

    assert plan.total_bytes == base.nbytes
    assert sum(block.element_count for block in plan.blocks) == 7 * 4
    assert all(
        block.global_slices[0][0] < block.global_slices[0][1] for block in plan.blocks
    )


def test_tn_redistribution_rejects_incompatible_layouts():
    base = DistributedTNValueLayout(
        value_id="intermediate:test",
        producer_id="contract:test",
        labels=(10, 11),
        shape=(4, 4),
        dtype="torch.complex64",
        nbytes=4 * 4 * 8,
        semantics="unique_owner",
        owner_ranks=(0,),
    )
    source = shard_distributed_tn_value_layout(base, world_size=2, shard_label=10)

    with pytest.raises(ValueError, match="two sharded"):
        plan_distributed_tn_redistribution(source, base)


@pytest.mark.parametrize("destination_label", [10, 11])
def test_tn_redistribution_pack_and_assemble_reconstructs_logical_tensor(
    destination_label,
):
    base = DistributedTNValueLayout(
        value_id="intermediate:test",
        producer_id="contract:test",
        labels=(10, 11),
        shape=(7, 6),
        dtype="torch.complex64",
        nbytes=7 * 6 * 8,
        semantics="unique_owner",
        owner_ranks=(0,),
    )
    source = shard_distributed_tn_value_layout(base, world_size=3, shard_label=10)
    destination = shard_distributed_tn_value_layout(
        base, world_size=2, shard_label=destination_label
    )
    logical = torch.arange(7 * 6, dtype=torch.float32).reshape(7, 6).to(torch.complex64)
    source_locals = {
        shard.rank: partition_tn_tensor_for_shard(
            logical,
            base.labels,
            shard_label=source.shard_label,
            shard=shard,
        )
        for shard in source.shards
    }
    plan = plan_distributed_tn_redistribution(source, destination)
    packed = tuple(
        (
            block,
            pack_tn_redistribution_block(
                source_locals[block.source_rank], source, block
            ),
        )
        for block in plan.blocks
    )
    destination_locals = tuple(
        assemble_tn_redistribution_shard(
            destination,
            destination_rank=shard.rank,
            received_blocks=(
                item for item in packed if item[0].destination_rank == shard.rank
            ),
        )
        for shard in destination.shards
    )

    reconstructed = torch.cat(destination_locals, dim=destination.shard_axis)
    assert torch.equal(reconstructed, logical)
    assert sum(tensor.numel() for _, tensor in packed) == logical.numel()


def test_tn_redistribution_assembly_rejects_missing_block():
    base = DistributedTNValueLayout(
        value_id="intermediate:test",
        producer_id="contract:test",
        labels=(10, 11),
        shape=(6, 4),
        dtype="torch.complex64",
        nbytes=6 * 4 * 8,
        semantics="unique_owner",
        owner_ranks=(0,),
    )
    source = shard_distributed_tn_value_layout(base, world_size=3, shard_label=10)
    destination = shard_distributed_tn_value_layout(base, world_size=2, shard_label=11)
    logical = torch.zeros(base.shape, dtype=torch.complex64)
    source_locals = {
        shard.rank: partition_tn_tensor_for_shard(
            logical,
            base.labels,
            shard_label=source.shard_label,
            shard=shard,
        )
        for shard in source.shards
    }
    plan = plan_distributed_tn_redistribution(source, destination)
    target_rank = destination.shards[0].rank
    blocks = tuple(
        (
            block,
            pack_tn_redistribution_block(
                source_locals[block.source_rank], source, block
            ),
        )
        for block in plan.blocks
        if block.destination_rank == target_rank
    )

    with pytest.raises(ValueError, match="uncovered"):
        assemble_tn_redistribution_shard(
            destination,
            destination_rank=target_rank,
            received_blocks=blocks[:-1],
        )


def test_tn_redistribution_execution_requires_initialized_process_group():
    base = DistributedTNValueLayout(
        value_id="intermediate:test",
        producer_id="contract:test",
        labels=(10, 11),
        shape=(4, 4),
        dtype="torch.complex64",
        nbytes=4 * 4 * 8,
        semantics="unique_owner",
        owner_ranks=(0,),
    )
    source = shard_distributed_tn_value_layout(base, world_size=2, shard_label=10)
    destination = shard_distributed_tn_value_layout(base, world_size=2, shard_label=11)
    plan = plan_distributed_tn_redistribution(source, destination)
    local = torch.zeros(source.shards[0].local_shape, dtype=torch.complex64)

    with pytest.raises(RuntimeError, match="initialized"):
        execute_distributed_tn_redistribution(
            local,
            source,
            destination,
            plan,
            rank=0,
        )


def test_tn_communication_device_resolves_nccl_through_platform_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested: list[str] = []

    monkeypatch.setattr(distributed_execution.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(distributed_execution.dist, "get_backend", lambda: "nccl")
    monkeypatch.setattr(
        distributed_execution,
        "resolve_platform_device",
        lambda device: requested.append(device) or torch.device("cuda"),
    )

    device = distributed_execution._communication_device({})

    assert device == torch.device("cuda")
    assert requested == ["cuda"]
