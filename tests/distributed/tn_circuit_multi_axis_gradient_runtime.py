"""Multi-rank explicit multi-axis TN parameter-gradient validation."""

from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist

import flagquantum as fq
from flagquantum.runtime.backends.tensor_network.distributed_dag import (
    plan_distributed_tn_contraction_dag,
)
from flagquantum.runtime.backends.tensor_network.dynamic_reverse import (
    execute_dynamic_tn_parameter_pullback,
    execute_dynamic_tn_reverse_segment,
    plan_dynamic_tn_reverse_segment,
)
from flagquantum.runtime.backends.tensor_network.multi_axis_sharding import (
    execute_multi_axis_tn_target_cone,
    plan_multi_axis_tn_layout,
    plan_multi_axis_tn_peak_sharding,
)
from flagquantum.runtime.backends.tensor_network.partial_mesh import (
    DistributedTNMeshGroupCache,
    execute_partial_mesh_reverse_pair,
    execute_partial_mesh_tn_redistribution,
    partition_tn_tensor_for_partial_mesh,
    plan_partial_mesh_tn_layout,
    plan_partial_mesh_tn_redistribution,
)
from flagquantum.runtime.backends.tensor_network.redistribution import (
    execute_multi_axis_tn_redistribution,
    plan_multi_axis_tn_redistribution,
)
from flagquantum.runtime.backends.tensor_network.reverse_dag import (
    execute_explicit_tn_reverse_dag,
    execute_tn_forward_with_tape,
    plan_explicit_tn_reverse_dag,
    plan_tn_adjoint_layouts,
    validate_tn_adjoint_tensors,
)
from flagquantum.simulation.tensor_network.entrypoints import (
    build_tensor_network_expectation,
)


def _expectation(parameters: torch.Tensor):
    circuit = fq.Circuit(8)
    for qubit in range(8):
        circuit.ry(qubit, theta=parameters[qubit])
    for layer in range(3):
        for qubit in range(layer % 2, 7, 2):
            circuit.cx(qubit, qubit + 1)
    return build_tensor_network_expectation(circuit, z=list(range(8)))


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size < 2 or world_size & (world_size - 1):
        raise RuntimeError(
            "multi-axis gradient runtime requires a power-of-two world size >= 2"
        )

    parameters = torch.linspace(
        0.1,
        0.8,
        8,
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    expectation = _expectation(parameters)
    dag_payload = [None]
    if rank == 0:
        dag_payload[0] = plan_distributed_tn_contraction_dag(
            expectation,
            world_size=world_size,
            small_tensor_replication_bytes=1 << 60,
        )
    dist.broadcast_object_list(dag_payload, src=0)
    dag = dag_payload[0]
    if dag is None:
        raise RuntimeError("gradient DAG broadcast failed")
    peak_payload = [None]
    if rank == 0:
        peak_payload[0] = plan_multi_axis_tn_peak_sharding(dag).summary()
    dist.broadcast_object_list(peak_payload, src=0)
    peak_plan = peak_payload[0]
    if not isinstance(peak_plan, dict):
        raise RuntimeError("gradient peak plan broadcast failed")
    shard_labels = tuple(int(label) for label in peak_plan["shard_labels"])
    # Keep the CPU-to-CUDA copy in the autograd graph: NCCL requires CUDA
    # tensors, while the circuit builder currently materializes TN nodes on CPU.
    local_inputs = {
        f"input:{index}": node.tensor.to(device)
        for index, node in enumerate(expectation.nodes)
    }
    forward = execute_multi_axis_tn_target_cone(
        dag,
        local_inputs,
        target_operation_id=dag.operations[-1].operation_id,
        shard_labels=shard_labels,
        retain_tape=True,
    )
    if forward.forward_tape is None:
        raise RuntimeError("gradient execution did not retain a forward tape")
    reverse = plan_explicit_tn_reverse_dag(dag)
    adjoint_layout = plan_tn_adjoint_layouts(
        dag,
        reverse,
        mesh_labels=shard_labels,
    )
    validate_tn_adjoint_tensors(
        adjoint_layout,
        forward.forward_tape,
        require_all=True,
    )
    cotangents = execute_explicit_tn_reverse_dag(
        dag,
        reverse,
        forward.forward_tape,
        retain_cotangent_tape=True,
    )
    validate_tn_adjoint_tensors(
        adjoint_layout,
        cotangents.input_cotangents,
    )
    if cotangents.cotangent_tape is None:
        raise RuntimeError("gradient execution did not retain cotangents")
    values = {value.value_id: value for value in dag.values}
    mesh_rank = len(shard_labels)
    remesh_candidates = []
    for value_id, tensor in cotangents.cotangent_tape.items():
        layout = values[value_id]
        if not all(label in layout.labels for label in shard_labels):
            continue
        alternatives = tuple(
            label
            for label, extent in zip(layout.labels, layout.shape)
            if label not in shard_labels and extent == 2
        )
        if len(alternatives) >= mesh_rank:
            remesh_candidates.append(
                (layout.nbytes, value_id, alternatives[:mesh_rank], tensor)
            )
    if not remesh_candidates:
        raise RuntimeError("gradient DAG has no dynamic adjoint remesh candidate")
    _, remesh_value_id, destination_labels, remesh_cotangent = max(
        remesh_candidates,
        key=lambda candidate: (candidate[0], candidate[1]),
    )
    remesh_layout = values[remesh_value_id]
    source_layout = plan_multi_axis_tn_layout(
        remesh_layout,
        shard_labels=shard_labels,
        world_size=world_size,
    )
    destination_layout = plan_multi_axis_tn_layout(
        remesh_layout,
        shard_labels=destination_labels,
        world_size=world_size,
    )
    outbound_plan = plan_multi_axis_tn_redistribution(source_layout, destination_layout)
    outbound = execute_multi_axis_tn_redistribution(
        remesh_cotangent,
        source_layout,
        destination_layout,
        outbound_plan,
    )
    return_plan = plan_multi_axis_tn_redistribution(destination_layout, source_layout)
    returned = execute_multi_axis_tn_redistribution(
        outbound.local_tensor,
        destination_layout,
        source_layout,
        return_plan,
    )
    torch.testing.assert_close(returned.local_tensor, remesh_cotangent)

    continuation_candidates = []
    for record in reverse.records:
        if (
            record.output_value_id not in cotangents.cotangent_tape
            or any(
                value_id not in cotangents.cotangent_tape
                for value_id in record.input_value_ids
            )
            or max(
                len(record.output_labels),
                len(record.left_labels),
                len(record.right_labels),
            )
            > 8
        ):
            continue
        record_layouts = (
            values[record.output_value_id],
            values[record.input_value_ids[0]],
            values[record.input_value_ids[1]],
        )
        alternative_labels = []
        for layout in record_layouts:
            for label, extent in zip(layout.labels, layout.shape):
                if (
                    label not in shard_labels
                    and extent == 2
                    and label not in alternative_labels
                ):
                    alternative_labels.append(label)
        if len(alternative_labels) >= mesh_rank:
            continuation_candidates.append(
                (
                    sum(layout.nbytes for layout in record_layouts),
                    record.reverse_id,
                    record,
                    tuple(alternative_labels[:mesh_rank]),
                )
            )
    if not continuation_candidates:
        raise RuntimeError("gradient DAG has no partial-mesh continuation record")
    _, _, continuation_record, continuation_labels = max(
        continuation_candidates,
        key=lambda candidate: (candidate[0], candidate[1]),
    )

    def _remesh_value(value_id, tensor, destination_mesh_labels):
        logical_layout = values[value_id]
        source = plan_partial_mesh_tn_layout(
            logical_layout,
            mesh_labels=shard_labels,
            mesh_shape=(2,) * mesh_rank,
        )
        destination = plan_partial_mesh_tn_layout(
            logical_layout,
            mesh_labels=destination_mesh_labels,
            mesh_shape=(2,) * mesh_rank,
        )
        transition = plan_partial_mesh_tn_redistribution(source, destination)
        remeshed_value = execute_partial_mesh_tn_redistribution(
            tensor, source, destination, transition
        )
        return source, destination, transition, remeshed_value

    output_source, output_destination, output_transition, output_remeshed = (
        _remesh_value(
            continuation_record.output_value_id,
            cotangents.cotangent_tape[continuation_record.output_value_id],
            continuation_labels,
        )
    )
    left_id, right_id = continuation_record.input_value_ids
    left_source, left_destination, _, left_remeshed = _remesh_value(
        left_id,
        forward.forward_tape[left_id],
        continuation_labels,
    )
    right_source, right_destination, _, right_remeshed = _remesh_value(
        right_id,
        forward.forward_tape[right_id],
        continuation_labels,
    )
    records_by_output = {record.output_value_id: record for record in reverse.records}
    second_candidates = []
    for output_id, output_layout, output_side in (
        (left_id, left_destination, "left"),
        (right_id, right_destination, "right"),
    ):
        record = records_by_output.get(output_id)
        if record is None:
            continue
        if (
            max(
                len(record.output_labels),
                len(record.left_labels),
                len(record.right_labels),
            )
            <= 8
        ):
            second_candidates.append(
                (
                    values[output_id].nbytes,
                    output_id,
                    output_layout,
                    output_side,
                    record,
                )
            )
    if not second_candidates:
        raise RuntimeError("gradient DAG has no second partial-mesh reverse record")
    (
        _,
        second_output_id,
        second_output_destination,
        second_output_side,
        second_record,
    ) = max(second_candidates, key=lambda candidate: (candidate[0], candidate[1]))
    second_left_id, second_right_id = second_record.input_value_ids
    (
        second_left_source,
        second_left_destination,
        _,
        second_left_remeshed,
    ) = _remesh_value(
        second_left_id,
        forward.forward_tape[second_left_id],
        continuation_labels,
    )
    (
        second_right_source,
        second_right_destination,
        _,
        second_right_remeshed,
    ) = _remesh_value(
        second_right_id,
        forward.forward_tape[second_right_id],
        continuation_labels,
    )
    with DistributedTNMeshGroupCache(
        mesh_labels=continuation_labels,
        mesh_shape=(2,) * mesh_rank,
    ) as group_cache:
        continued = execute_partial_mesh_reverse_pair(
            output_remeshed.local_tensor,
            output_destination,
            left_remeshed.local_tensor,
            left_destination,
            right_remeshed.local_tensor,
            right_destination,
            group_cache=group_cache,
        )
        second_output_cotangent = (
            continued.left_cotangent
            if second_output_side == "left"
            else continued.right_cotangent
        )
        second_continued = execute_partial_mesh_reverse_pair(
            second_output_cotangent,
            second_output_destination,
            second_left_remeshed.local_tensor,
            second_left_destination,
            second_right_remeshed.local_tensor,
            second_right_destination,
            group_cache=group_cache,
        )
        continuation_group_count = group_cache.group_count

    def _return_cotangent(tensor, destination, source):
        transition = plan_partial_mesh_tn_redistribution(destination, source)
        return execute_partial_mesh_tn_redistribution(
            tensor, destination, source, transition
        ).local_tensor

    continued_left_source = _return_cotangent(
        continued.left_cotangent, left_destination, left_source
    )
    continued_right_source = _return_cotangent(
        continued.right_cotangent, right_destination, right_source
    )
    logical_tape = execute_tn_forward_with_tape(
        dag,
        {
            f"input:{index}": node.tensor.detach()
            for index, node in enumerate(expectation.nodes)
        },
    )
    logical_reverse = execute_explicit_tn_reverse_dag(
        dag,
        reverse,
        logical_tape,
        retain_cotangent_tape=True,
    )
    if logical_reverse.cotangent_tape is None:
        raise RuntimeError("logical reverse did not retain cotangents")
    expected_left_destination = partition_tn_tensor_for_partial_mesh(
        logical_reverse.cotangent_tape[left_id].to(device),
        left_destination,
        rank=rank,
    )
    expected_right_destination = partition_tn_tensor_for_partial_mesh(
        logical_reverse.cotangent_tape[right_id].to(device),
        right_destination,
        rank=rank,
    )
    torch.testing.assert_close(
        continued.left_cotangent,
        expected_left_destination,
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        continued.right_cotangent,
        expected_right_destination,
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        continued_left_source,
        partition_tn_tensor_for_partial_mesh(
            logical_reverse.cotangent_tape[left_id].to(device),
            left_source,
            rank=rank,
        ),
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        continued_right_source,
        partition_tn_tensor_for_partial_mesh(
            logical_reverse.cotangent_tape[right_id].to(device),
            right_source,
            rank=rank,
        ),
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        second_continued.left_cotangent,
        partition_tn_tensor_for_partial_mesh(
            logical_reverse.cotangent_tape[second_left_id].to(device),
            second_left_destination,
            rank=rank,
        ),
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        second_continued.right_cotangent,
        partition_tn_tensor_for_partial_mesh(
            logical_reverse.cotangent_tape[second_right_id].to(device),
            second_right_destination,
            rank=rank,
        ),
        atol=1e-5,
        rtol=1e-5,
    )
    product_segment = plan_dynamic_tn_reverse_segment(
        dag,
        reverse,
        start_output_value_id=dag.output_value_id,
        source_mesh_labels=shard_labels,
        destination_mesh_labels=continuation_labels,
        mesh_shape=(2,) * mesh_rank,
        max_records=200,
    )
    with DistributedTNMeshGroupCache(
        mesh_labels=continuation_labels,
        mesh_shape=(2,) * mesh_rank,
    ) as product_group_cache:
        product_segment_result = execute_dynamic_tn_reverse_segment(
            dag,
            reverse,
            product_segment,
            forward.forward_tape,
            torch.ones_like(forward.forward_tape[dag.output_value_id]),
            group_cache=product_group_cache,
        )
    for value_id, actual in product_segment_result.cotangents.items():
        expected_layout = plan_partial_mesh_tn_layout(
            values[value_id],
            mesh_labels=continuation_labels,
            mesh_shape=(2,) * mesh_rank,
        )
        torch.testing.assert_close(
            actual,
            partition_tn_tensor_for_partial_mesh(
                logical_reverse.cotangent_tape[value_id].to(device),
                expected_layout,
                rank=rank,
            ),
            atol=1e-5,
            rtol=1e-5,
        )
    product_pullback = execute_dynamic_tn_parameter_pullback(
        dag,
        product_segment,
        product_segment_result,
        forward.forward_tape,
        (parameters,),
        retain_graph=True,
    )
    product_parameter_gradient = product_pullback.gradients[0]
    local_nodes = []
    node_cotangents = []
    for index in range(len(expectation.nodes)):
        value_id = f"input:{index}"
        local_node = forward.forward_tape[value_id]
        cotangent = cotangents.input_cotangents.get(value_id)
        if local_node.requires_grad and cotangent is not None:
            local_nodes.append(local_node)
            node_cotangents.append(cotangent)
    local_parameter_gradient = torch.autograd.grad(
        tuple(local_nodes),
        parameters,
        grad_outputs=tuple(node_cotangents),
    )[0]
    dist.all_reduce(local_parameter_gradient, op=dist.ReduceOp.SUM)

    reference_parameters = torch.linspace(
        0.1,
        0.8,
        8,
        dtype=torch.float64,
        requires_grad=True,
    )
    reference_value = _expectation(reference_parameters).contract(
        strategy="memory_greedy"
    )
    reference_gradient = torch.autograd.grad(
        reference_value,
        reference_parameters,
        grad_outputs=torch.ones_like(reference_value),
    )[0]
    torch.testing.assert_close(
        local_parameter_gradient.detach().cpu(),
        reference_gradient,
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        product_parameter_gradient.detach().cpu(),
        reference_gradient,
        atol=1e-6,
        rtol=1e-6,
    )
    print(
        json.dumps(
            {
                "rank": rank,
                "multi_axis_gradient_passed": True,
                "world_size": world_size,
                "forward_dag_identity": dag.identity,
                "reverse_dag_identity": reverse.identity,
                "adjoint_layout_identity": adjoint_layout.identity,
                "peak_plan_identity": peak_plan["identity"],
                "shard_labels": shard_labels,
                "parameter_count": parameters.numel(),
                "partitioned_adjoint_value_count": (
                    adjoint_layout.summary()["partitioned_value_count"]
                ),
                "fully_replicated_adjoint_value_count": (
                    adjoint_layout.summary()["fully_replicated_value_count"]
                ),
                "dynamic_remesh_value_id": remesh_value_id,
                "dynamic_remesh_source_labels": shard_labels,
                "dynamic_remesh_destination_labels": destination_labels,
                "dynamic_remesh_outbound_plan_identity": (outbound_plan.identity),
                "dynamic_remesh_roundtrip_passed": True,
                "dynamic_remesh_sent_bytes": outbound.sent_bytes,
                "dynamic_remesh_received_bytes": outbound.received_bytes,
                "partial_mesh_continuation_passed": True,
                "partial_mesh_continuation_reverse_id": (
                    continuation_record.reverse_id
                ),
                "partial_mesh_continuation_output_value_id": (
                    continuation_record.output_value_id
                ),
                "partial_mesh_continuation_destination_labels": (continuation_labels),
                "partial_mesh_continuation_output_plan_identity": (
                    output_transition.identity
                ),
                "partial_mesh_continuation_subgroup_collectives": (
                    continued.subgroup_collective_count
                    + second_continued.subgroup_collective_count
                ),
                "partial_mesh_continuation_record_count": 2,
                "partial_mesh_second_reverse_id": second_record.reverse_id,
                "partial_mesh_second_output_value_id": second_output_id,
                "partial_mesh_cached_group_count": (continuation_group_count),
                "product_segment_identity": product_segment.identity,
                "product_segment_passed": True,
                "product_segment_rank_consensus_validated": (
                    product_segment_result.rank_consensus_validated
                ),
                "product_segment_tensor_preflight_validated": (
                    product_segment_result.rank_tensor_preflight_validated
                ),
                "product_segment_completed": (product_segment_result.completed),
                "product_segment_record_count": len(product_segment.records),
                "product_segment_redistribution_count": (
                    product_segment_result.redistribution_count
                ),
                "product_segment_released_forward_values": (
                    product_segment_result.released_forward_value_count
                ),
                "product_segment_peak_cached_forward_bytes": (
                    product_segment_result.peak_cached_forward_bytes
                ),
                "product_parameter_pullback_passed": True,
                "product_parameter_input_cotangent_count": (
                    product_pullback.input_cotangent_count
                ),
                "product_parameter_cotangent_redistribution_count": (
                    product_pullback.cotangent_redistribution_count
                ),
                "product_parameter_replica_normalized_input_count": (
                    product_pullback.replica_normalized_input_count
                ),
                "product_parameter_allreduce_count": (
                    product_pullback.parameter_allreduce_count
                ),
                "product_parameter_nonfinite_gradient_count": (
                    product_pullback.nonfinite_gradient_count
                ),
                "product_parameter_gradient_max_abs_error": float(
                    (product_parameter_gradient.detach().cpu() - reference_gradient)
                    .abs()
                    .max()
                ),
                "collective_count_forward": forward.collective_count,
                "collective_count_parameter_gradient": 1,
                "nonfinite_cotangent_count": (cotangents.nonfinite_cotangent_count),
                "gradient_max_abs_error": float(
                    (local_parameter_gradient.detach().cpu() - reference_gradient)
                    .abs()
                    .max()
                ),
                "full_state_materialization": False,
                "silent_statevector_fallback": False,
            }
        ),
        flush=True,
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
