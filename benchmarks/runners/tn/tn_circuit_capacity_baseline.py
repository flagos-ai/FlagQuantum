"""Single-GPU native TN baseline for distributed multi-axis capacity evidence."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import torch

import flagquantum as fq


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qubits", type=int, default=18)
    parser.add_argument("--layers", type=int, default=9)
    parser.add_argument(
        "--strategy",
        choices=("memory_greedy", "quality_greedy"),
        default="memory_greedy",
    )
    parser.add_argument(
        "--dtype",
        choices=("complex64", "complex128"),
        default="complex64",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _emit(payload: dict, output: Path | None) -> None:
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(payload))


def main() -> None:
    arguments = _arguments()
    device = torch.device("cuda:0")
    circuit = fq.Circuit(arguments.qubits)
    for qubit in range(arguments.qubits):
        circuit.ry(qubit, theta=0.1 * (qubit + 1))
    for layer in range(arguments.layers):
        for qubit in range(layer % 2, arguments.qubits - 1, 2):
            circuit.cx(qubit, qubit + 1)
    dtype = {
        "complex64": torch.complex64,
        "complex128": torch.complex128,
    }[arguments.dtype]
    plan = fq.build_tensor_network(circuit, dtype=dtype)
    expectation = fq.build_tensor_network_expectation(
        plan,
        z=list(range(arguments.qubits)),
    )
    gpu_expectation = replace(
        expectation,
        nodes=tuple(
            replace(node, tensor=node.tensor.to(device)) for node in expectation.nodes
        ),
    )
    path = {
        "memory_greedy": gpu_expectation.memory_greedy_path,
        "quality_greedy": gpu_expectation.quality_greedy_path,
    }[arguments.strategy]()
    planned_peak_elements = max(step.intermediate_size for step in path)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    try:
        value = gpu_expectation.contract(strategy=arguments.strategy)
        torch.cuda.synchronize(device)
    except torch.OutOfMemoryError as error:
        elapsed_seconds = time.perf_counter() - started
        allocated = int(torch.cuda.memory_allocated(device))
        reserved = int(torch.cuda.memory_reserved(device))
        torch.cuda.empty_cache()
        _emit(
            {
                "baseline": f"single_gpu_native_tn_{arguments.strategy}",
                "qubits": arguments.qubits,
                "layers": arguments.layers,
                "planned_peak_elements": planned_peak_elements,
                "planned_peak_bytes": (
                    planned_peak_elements
                    * gpu_expectation.nodes[0].tensor.element_size()
                ),
                "dtype": arguments.dtype,
                "elapsed_seconds": elapsed_seconds,
                "status": "failed_oom",
                "error_type": type(error).__name__,
                "allocated_bytes_at_failure": allocated,
                "reserved_bytes_at_failure": reserved,
                "completed": False,
            },
            arguments.output,
        )
        return
    elapsed_seconds = time.perf_counter() - started
    _emit(
        {
            "baseline": f"single_gpu_native_tn_{arguments.strategy}",
            "qubits": arguments.qubits,
            "layers": arguments.layers,
            "output_shape": tuple(value.shape),
            "output_finite": bool(torch.isfinite(value).all()),
            "planned_peak_elements": planned_peak_elements,
            "planned_peak_bytes": (
                planned_peak_elements * gpu_expectation.nodes[0].tensor.element_size()
            ),
            "dtype": arguments.dtype,
            "elapsed_seconds": elapsed_seconds,
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "completed": True,
        },
        arguments.output,
    )


if __name__ == "__main__":
    main()
