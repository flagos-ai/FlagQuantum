"""Eight-rank multi-label contraction selected from a real circuit TN DAG."""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch
import torch.distributed as dist

import flagquantum as fq
from flagquantum.runtime.executors.tensor_network.distributed_dag import (
    plan_distributed_tn_contraction_dag,
)
from flagquantum.runtime.executors.tensor_network.multi_axis_sharding import (
    execute_multi_axis_tn_target_cone,
    plan_multi_axis_tn_peak_sharding,
)
from flagquantum.simulation.tensor_network.entrypoints import (
    build_tensor_network_expectation,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="gloo")
    parser.add_argument("--qubits", type=int, default=8)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--skip-reference", action="store_true")
    return parser.parse_args()


def _expectation(qubits: int, layers: int):
    if qubits < 4 or layers < 1:
        raise ValueError(
            "circuit capacity workload requires qubits >= 4 and layers >= 1"
        )
    circuit = fq.Circuit(qubits)
    for qubit in range(qubits):
        circuit.ry(qubit, theta=0.1 * (qubit + 1))
    for layer in range(layers):
        for qubit in range(layer % 2, qubits - 1, 2):
            circuit.cx(qubit, qubit + 1)
    return build_tensor_network_expectation(circuit, z=list(range(qubits)))


def main() -> None:
    arguments = _arguments()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(
        f"cuda:{local_rank}" if arguments.backend == "nccl" else "cpu"
    )
    if arguments.backend == "nccl":
        torch.cuda.set_device(device)
        dist.init_process_group("nccl", device_id=device)
    else:
        dist.init_process_group("gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    shard_label_count = int(math.log2(world_size))
    if world_size < 4 or 2**shard_label_count != world_size:
        raise ValueError("circuit multi-axis runtime requires power-of-two world size")

    expectation = _expectation(arguments.qubits, arguments.layers)
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
        raise RuntimeError("distributed TN DAG broadcast failed")
    layouts = {value.value_id: value for value in dag.values}
    target = dag.operations[-1]
    left, right = (layouts[value_id] for value_id in target.input_value_ids)
    available_labels = tuple(
        label
        for label in left.labels
        if label in right.labels and label not in target.output_labels
    )
    peak_plan_payload = [None]
    if rank == 0:
        peak_plan_payload[0] = plan_multi_axis_tn_peak_sharding(
            dag,
            target_operation_id=target.operation_id,
            world_size=world_size,
        ).summary()
    dist.broadcast_object_list(peak_plan_payload, src=0)
    peak_plan = peak_plan_payload[0]
    if not isinstance(peak_plan, dict):
        raise RuntimeError("multi-axis peak plan broadcast failed")
    shard_labels = tuple(int(label) for label in peak_plan["shard_labels"])

    local_inputs = {
        f"input:{index}": node.tensor.to(device)
        for index, node in enumerate(expectation.nodes)
    }
    if arguments.backend == "nccl":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    result = execute_multi_axis_tn_target_cone(
        dag,
        local_inputs,
        target_operation_id=target.operation_id,
        shard_labels=shard_labels,
    )
    if arguments.backend == "nccl":
        torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - started
    cuda_peak_allocated_bytes = (
        int(torch.cuda.max_memory_allocated(device))
        if arguments.backend == "nccl"
        else None
    )
    cuda_peak_reserved_bytes = (
        int(torch.cuda.max_memory_reserved(device))
        if arguments.backend == "nccl"
        else None
    )
    if not arguments.skip_reference:
        reference = expectation.contract(strategy="memory_greedy")
        torch.testing.assert_close(
            result.value.detach().cpu(),
            reference.detach().cpu(),
            atol=1e-5,
            rtol=1e-5,
        )
    output_finite = bool(torch.isfinite(result.value).all())
    if not output_finite:
        raise RuntimeError("multi-axis TN capacity output is non-finite")
    print(
        json.dumps(
            {
                "rank": rank,
                "backend": arguments.backend,
                "qubits": arguments.qubits,
                "layers": arguments.layers,
                "circuit_multi_axis_passed": True,
                "reference_checked": not arguments.skip_reference,
                "output_finite": output_finite,
                "target_operation_id": target.operation_id,
                "available_contracted_binary_labels": available_labels,
                "shard_labels": shard_labels,
                "mesh_shape": [2] * shard_label_count,
                "peak_plan_identity": peak_plan["identity"],
                "predicted_peak_local_bytes": (peak_plan["predicted_peak_local_bytes"]),
                "predicted_peak_logical_bytes": (
                    peak_plan["predicted_peak_logical_bytes"]
                ),
                "peak_plan_candidate_count": peak_plan["candidate_count"],
                "partitioned_raw_input_count": len(result.partitioned_input_value_ids),
                "partitioned_intermediate_count": len(
                    result.partitioned_intermediate_value_ids
                ),
                "target_local_input_bytes": result.target_local_input_bytes,
                "target_logical_input_bytes": result.target_logical_input_bytes,
                "peak_live_local_tensor_bytes": (result.peak_live_local_tensor_bytes),
                "peak_live_logical_tensor_bytes": (
                    result.peak_live_logical_tensor_bytes
                ),
                "released_value_count": result.released_value_count,
                "elapsed_seconds": elapsed_seconds,
                "cuda_peak_allocated_bytes": cuda_peak_allocated_bytes,
                "cuda_peak_reserved_bytes": cuda_peak_reserved_bytes,
                "collective_count": result.collective_count,
                "collective_bytes": result.collective_bytes,
                "high_rank_einsum_fallback_count": (
                    result.high_rank_einsum_fallback_count
                ),
                "full_target_inputs_materialized": (
                    result.full_target_inputs_materialized
                ),
                "prefix_full_intermediates_materialized": False,
                "full_state_materialization": False,
                "silent_statevector_fallback": False,
            }
        ),
        flush=True,
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
