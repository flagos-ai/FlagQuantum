"""Execute a native high-quality contraction path through the production DAG."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import flagquantum as fq
from flagquantum.runtime.executors.tensor_network.distributed_dag import (
    plan_distributed_tn_contraction_dag,
)
from flagquantum.runtime.executors.tensor_network.distributed_execution import (
    execute_distributed_tn_contraction_dag,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qubits", type=int, default=18)
    parser.add_argument("--layers", type=int, default=13)
    parser.add_argument(
        "--objective",
        choices=("quality", "quality_multistart"),
        default="quality_multistart",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    device = torch.device("cuda:0")
    circuit = fq.Circuit(arguments.qubits)
    for qubit in range(arguments.qubits):
        circuit.ry(qubit, theta=0.1 * (qubit + 1))
    for layer in range(arguments.layers):
        for qubit in range(layer % 2, arguments.qubits - 1, 2):
            circuit.cx(qubit, qubit + 1)
    expectation = fq.build_tensor_network_expectation(
        circuit,
        z=list(range(arguments.qubits)),
    )
    planning_started = time.perf_counter()
    dag = plan_distributed_tn_contraction_dag(
        expectation,
        world_size=1,
        objective=arguments.objective,
        small_tensor_replication_bytes=1 << 60,
    )
    planning_seconds = time.perf_counter() - planning_started
    local_inputs = {
        f"input:{index}": node.tensor.to(device)
        for index, node in enumerate(expectation.nodes)
    }
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    execution_started = time.perf_counter()
    result = execute_distributed_tn_contraction_dag(dag, local_inputs)
    torch.cuda.synchronize(device)
    payload = {
        "schema_version": 1,
        "workload": f"{arguments.qubits}q_{arguments.layers}l_expectation",
        "objective": arguments.objective,
        "dag_identity": dag.identity,
        "operation_count": len(dag.operations),
        "estimated_flops": sum(
            operation.estimated_cost for operation in dag.operations
        ),
        "largest_intermediate_elements": max(
            operation.intermediate_elements for operation in dag.operations
        ),
        "planning_seconds": planning_seconds,
        "execution_seconds": time.perf_counter() - execution_started,
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "output_finite": bool(torch.isfinite(result.value).all()),
        "completed": True,
        "distribution_semantics": "single_device_fast_path",
        "scalability_claim_allowed": False,
    }
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
