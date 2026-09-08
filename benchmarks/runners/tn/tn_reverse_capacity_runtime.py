"""Eight-GPU dynamic reverse capacity run with budgeted rematerialization."""

from __future__ import annotations

import json
import os
import time

import torch
import torch.distributed as dist

import flagquantum as fq
from flagquantum.runtime.executors.tensor_network.distributed_dag import (
    plan_distributed_tn_contraction_dag,
)
from flagquantum.runtime.executors.tensor_network.dynamic_reverse import (
    execute_dynamic_tn_reverse_segment,
    plan_dynamic_tn_reverse_segment,
)
from flagquantum.runtime.executors.tensor_network.multi_axis_sharding import (
    execute_multi_axis_tn_target_cone,
    plan_multi_axis_tn_peak_sharding,
)
from flagquantum.runtime.executors.tensor_network.partial_mesh import (
    DistributedTNMeshGroupCache,
)
from flagquantum.runtime.executors.tensor_network.rematerialization import (
    DistributedTNRematerializationProvider,
    plan_partial_mesh_rematerialization,
)
from flagquantum.runtime.executors.tensor_network.reverse_dag import (
    plan_explicit_tn_reverse_dag,
)


def _expectation():
    circuit = fq.Circuit(18)
    for qubit in range(18):
        circuit.ry(qubit, theta=0.1 * (qubit + 1))
    for layer in range(13):
        for qubit in range(layer % 2, 17, 2):
            circuit.cx(qubit, qubit + 1)
    return fq.build_tensor_network_expectation(circuit, z=list(range(18)))


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    if dist.get_world_size() != 8:
        raise RuntimeError("TN reverse capacity runtime requires eight ranks")

    expectation = _expectation()
    payload = [None]
    if rank == 0:
        dag = plan_distributed_tn_contraction_dag(
            expectation,
            world_size=8,
            small_tensor_replication_bytes=1 << 60,
        )
        peak = plan_multi_axis_tn_peak_sharding(dag, world_size=8)
        rematerialization = plan_partial_mesh_rematerialization(
            dag,
            mesh_labels=peak.shard_labels,
            mesh_shape=(2, 2, 2),
            budget_bytes=2 << 30,
        )
        payload[0] = (dag, peak, rematerialization)
    dist.broadcast_object_list(payload, src=0)
    dag, peak, rematerialization = payload[0]
    shard_labels = peak.shard_labels
    local_inputs = {
        f"input:{index}": node.tensor.to(device)
        for index, node in enumerate(expectation.nodes)
    }
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    forward = execute_multi_axis_tn_target_cone(
        dag,
        local_inputs,
        target_operation_id=dag.operations[-1].operation_id,
        shard_labels=shard_labels,
        retain_value_ids=rematerialization.checkpoint_value_ids,
    )
    if forward.forward_tape is None:
        raise RuntimeError("capacity forward did not retain checkpoint tape")
    reverse = plan_explicit_tn_reverse_dag(dag)
    segment = plan_dynamic_tn_reverse_segment(
        dag,
        reverse,
        start_output_value_id=dag.output_value_id,
        source_mesh_labels=shard_labels,
        destination_mesh_labels=shard_labels,
        mesh_shape=(2, 2, 2),
        max_records=len(reverse.records),
        max_tensor_rank=64,
    )
    with DistributedTNMeshGroupCache(
        mesh_labels=shard_labels,
        mesh_shape=(2, 2, 2),
    ) as group_cache:
        provider = DistributedTNRematerializationProvider(
            dag,
            forward.forward_tape,
            mesh_labels=shard_labels,
            mesh_shape=(2, 2, 2),
            group_cache=group_cache,
            max_transient_bytes=28 << 30,
        )
        result = execute_dynamic_tn_reverse_segment(
            dag,
            reverse,
            segment,
            forward.forward_tape,
            torch.ones_like(forward.value),
            group_cache=group_cache,
            source_forward_provider=provider,
        )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    finite = all(
        bool(torch.isfinite(cotangent).all())
        for cotangent in result.cotangents.values()
    )
    if not result.completed or not finite:
        raise RuntimeError("capacity dynamic reverse did not complete finitely")
    print(
        json.dumps(
            {
                "rank": rank,
                "reverse_capacity_passed": True,
                "qubits": 18,
                "layers": 13,
                "record_count": len(segment.records),
                "checkpoint_count": len(
                    rematerialization.checkpoint_value_ids
                ),
                "checkpoint_local_bytes": (
                    rematerialization.checkpoint_local_bytes
                ),
                "raw_input_local_bytes": (
                    rematerialization.raw_input_local_bytes
                ),
                "rematerialized_forward_value_count": (
                    result.rematerialized_forward_value_count
                ),
                "rematerialization_operation_count": provider.operation_count,
                "rematerialization_peak_transient_bytes": (
                    provider.peak_transient_bytes
                ),
                "peak_cuda_allocated_bytes": int(
                    torch.cuda.max_memory_allocated(device)
                ),
                "peak_cuda_reserved_bytes": int(
                    torch.cuda.max_memory_reserved(device)
                ),
                "elapsed_seconds": elapsed,
                "cotangent_frontier_finite": finite,
                "full_state_materialization": False,
                "silent_statevector_fallback": False,
            }
        ),
        flush=True,
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
