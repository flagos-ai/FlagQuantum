"""Single-A800 full reverse and parameter pullback for a high-quality TN path."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import torch

import flagquantum as fq
from flagquantum.runtime.executors.tensor_network.distributed_dag import (
    plan_distributed_tn_contraction_dag,
)
from flagquantum.runtime.executors.tensor_network.reverse_dag import (
    execute_explicit_tn_reverse_dag,
    execute_tn_forward_with_tape,
    plan_explicit_tn_reverse_dag,
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
    parameters = tuple(
        torch.tensor(
            0.1 * (qubit + 1),
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )
        for qubit in range(arguments.qubits)
    )
    circuit = fq.Circuit(arguments.qubits)
    for qubit, parameter in enumerate(parameters):
        circuit.ry(qubit, theta=parameter)
    for layer in range(arguments.layers):
        for qubit in range(layer % 2, arguments.qubits - 1, 2):
            circuit.cx(qubit, qubit + 1)
    expectation = fq.build_tensor_network_expectation(
        circuit,
        z=list(range(arguments.qubits)),
    )
    expectation = replace(
        expectation,
        nodes=tuple(
            replace(node, tensor=node.tensor.to(device))
            for node in expectation.nodes
        ),
    )
    if any(not node.tensor.is_cuda for node in expectation.nodes):
        raise RuntimeError("A800 TN gradient benchmark requires all inputs on CUDA")

    torch.cuda.reset_peak_memory_stats(device)
    planning_started = time.perf_counter()
    dag = plan_distributed_tn_contraction_dag(
        expectation,
        world_size=1,
        objective=arguments.objective,
        small_tensor_replication_bytes=1 << 60,
    )
    reverse = plan_explicit_tn_reverse_dag(dag)
    planning_seconds = time.perf_counter() - planning_started
    detached_inputs = {
        f"input:{index}": node.tensor.detach()
        for index, node in enumerate(expectation.nodes)
    }

    torch.cuda.synchronize(device)
    forward_started = time.perf_counter()
    tape = execute_tn_forward_with_tape(dag, detached_inputs)
    torch.cuda.synchronize(device)
    forward_seconds = time.perf_counter() - forward_started

    reverse_started = time.perf_counter()
    cotangents = execute_explicit_tn_reverse_dag(dag, reverse, tape)
    torch.cuda.synchronize(device)
    reverse_seconds = time.perf_counter() - reverse_started

    differentiable_nodes = []
    node_cotangents = []
    for index, node in enumerate(expectation.nodes):
        value_id = f"input:{index}"
        cotangent = cotangents.input_cotangents.get(value_id)
        if node.tensor.requires_grad and cotangent is not None:
            differentiable_nodes.append(node.tensor)
            node_cotangents.append(cotangent)
    pullback_started = time.perf_counter()
    gradients = torch.autograd.grad(
        tuple(differentiable_nodes),
        parameters,
        grad_outputs=tuple(node_cotangents),
        allow_unused=True,
    )
    gradients = tuple(
        torch.zeros_like(parameter) if gradient is None else gradient
        for parameter, gradient in zip(parameters, gradients, strict=True)
    )
    torch.cuda.synchronize(device)
    pullback_seconds = time.perf_counter() - pullback_started

    payload = {
        "schema_version": 1,
        "workload": f"{arguments.qubits}q_{arguments.layers}l_expectation_gradient",
        "objective": arguments.objective,
        "dag_identity": dag.identity,
        "reverse_identity": reverse.identity,
        "operation_count": len(dag.operations),
        "reverse_record_count": len(reverse.records),
        "estimated_flops": sum(
            operation.estimated_cost for operation in dag.operations
        ),
        "largest_intermediate_elements": max(
            operation.intermediate_elements for operation in dag.operations
        ),
        "planning_seconds": planning_seconds,
        "forward_seconds": forward_seconds,
        "reverse_seconds": reverse_seconds,
        "parameter_pullback_seconds": pullback_seconds,
        "forward_reverse_pullback_seconds": (
            forward_seconds + reverse_seconds + pullback_seconds
        ),
        "saved_tape_bytes": sum(
            int(value.numel()) * int(value.element_size())
            for value in tape.values()
        ),
        "input_cotangent_count": len(cotangents.input_cotangents),
        "nonfinite_cotangent_count": cotangents.nonfinite_cotangent_count,
        "parameter_count": len(parameters),
        "cuda_input_count": sum(
            int(node.tensor.is_cuda) for node in expectation.nodes
        ),
        "nonfinite_parameter_gradient_count": sum(
            int(not bool(torch.isfinite(gradient).all())) for gradient in gradients
        ),
        "gradient_max_abs": max(
            float(gradient.detach().abs().max()) for gradient in gradients
        ),
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "full_state_materialization": False,
        "silent_statevector_fallback": False,
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
