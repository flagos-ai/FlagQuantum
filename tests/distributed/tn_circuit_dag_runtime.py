"""Circuit-derived distributed TN DAG correctness check."""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.distributed as dist

import flagquantum as fq
from flagquantum.runtime.backends.tensor_network import (
    execute_sharded_tn_contraction_dag,
    plan_distributed_tn_contraction_dag,
    with_sharded_tn_intermediate,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="gloo")
    return parser.parse_args()


def _expectation():
    circuit = fq.Circuit(6)
    circuit.h(0).ry(1, theta=0.17).rx(4, theta=-0.31)
    circuit.cx(0, 1).cx(1, 2).rzz(2, 3, theta=0.23)
    circuit.cx(3, 4).ry(5, theta=0.41).cx(4, 5)
    return fq.build_tensor_network_expectation(
        circuit,
        x=[0, 3],
        z=[1, 4, 5],
    )


def _with_real_shard_chain(dag):
    values = {value.value_id: value for value in dag.values}
    consumers = {}
    for operation in dag.operations:
        for value_id in operation.input_value_ids:
            consumers[value_id] = operation

    candidates = []
    for start in dag.operations:
        initial = values[start.output_value_id]
        for label, extent in zip(initial.labels, initial.shape):
            if extent < dag.world_size:
                continue
            chain = [initial.value_id]
            current = initial
            while True:
                consumer = consumers.get(current.value_id)
                if consumer is None:
                    chain = []
                    break
                output = values[consumer.output_value_id]
                if label not in output.labels:
                    other_id = next(
                        value_id
                        for value_id in consumer.input_value_ids
                        if value_id != current.value_id
                    )
                    if label not in values[other_id].labels:
                        chain = []
                    break
                output_extent = output.shape[output.labels.index(label)]
                if output_extent < dag.world_size:
                    chain = []
                    break
                chain.append(output.value_id)
                current = output
            if chain:
                candidates.append((len(chain), int(label), tuple(chain)))
    if not candidates:
        raise RuntimeError("circuit TN DAG has no executable shard chain")
    _, label, chain = max(candidates)
    transformed = dag
    for value_id in chain:
        transformed = with_sharded_tn_intermediate(
            transformed,
            value_id,
            shard_label=label,
        )
    return transformed, label, chain


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

    expectation = _expectation()
    base_dag = plan_distributed_tn_contraction_dag(
        expectation,
        world_size=world_size,
        small_tensor_replication_bytes=1 << 60,
    )
    dag, shard_label, shard_chain = _with_real_shard_chain(base_dag)
    local_inputs = {
        f"input:{index}": node.tensor.to(device)
        for index, node in enumerate(expectation.nodes)
    }
    result = execute_sharded_tn_contraction_dag(dag, local_inputs)
    reference = expectation.contract(strategy="memory_greedy")
    torch.testing.assert_close(
        result.value.detach().cpu(),
        reference.detach().cpu(),
        atol=1e-5,
        rtol=1e-5,
    )
    assert result.sharded_value_ids == shard_chain
    assert result.collective_count == 1
    print(
        json.dumps(
            {
                "rank": rank,
                "backend": arguments.backend,
                "circuit_derived_dag_passed": True,
                "dag_operation_count": len(dag.operations),
                "replicated_operation_count": len(result.replicated_operation_ids),
                "shard_label": shard_label,
                "shard_chain": shard_chain,
                **result.summary(),
            }
        ),
        flush=True,
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
