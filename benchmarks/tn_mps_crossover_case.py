"""Single-A800 MPS versus general-TN sparse-output crossover case."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

import flagquantum as fq
import flagquantum.backends as fqb
import flagquantum.backends.tensor_network as fqbtn


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", choices=("chain", "binary_tree"), required=True)
    parser.add_argument("--backend", choices=("mps", "tensor_network"), required=True)
    parser.add_argument("--qubits", type=int, required=True)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--targets", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _circuit(args: argparse.Namespace) -> fq.Circuit:
    circuit = fq.Circuit(args.qubits, device="cuda", dtype=torch.complex64)
    for layer in range(args.depth):
        for wire in range(args.qubits):
            circuit.ry(wire, theta=0.011 * (1 + layer + wire % 5))
        if args.topology == "chain":
            for wire in range(layer % 2, args.qubits - 1, 2):
                circuit.cx(wire, wire + 1)
        else:
            for child in range(1, args.qubits):
                circuit.cx((child - 1) // 2, child)
    return circuit


def _targets(qubits: int, count: int) -> tuple[str, ...]:
    values = [0]
    for index in range(1, count):
        values.append((index * 0x9E3779B97F4A7C15) % (1 << qubits))
    return tuple(f"{value:0{qubits}b}" for value in values)


def main() -> None:
    args = _parser().parse_args()
    torch.cuda.set_device(0)
    circuit = _circuit(args)
    targets = _targets(args.qubits, args.targets)
    torch.cuda.reset_peak_memory_stats()
    times = []
    values = None
    observed_bond = None
    for _ in range(args.iterations):
        start = time.perf_counter()
        if args.backend == "mps":
            state = fqb.run_mps(circuit)
            values = state.amplitudes(targets)
            observed_bond = state.max_bond
        else:
            values = fqbtn.tensor_network_amplitudes(circuit, targets)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    assert values is not None
    payload = {
        "schema_version": 1,
        "benchmark": "tn_mps_crossover",
        "status": "measured_success",
        "topology": args.topology,
        "backend": args.backend,
        "qubits": args.qubits,
        "depth": args.depth,
        "target_count": args.targets,
        "iterations": args.iterations,
        "cold_start_seconds": times[0],
        "steady_state_seconds": (
            statistics.median(times[1:]) if len(times) > 1 else None
        ),
        "times_seconds": times,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "observed_max_bond": observed_bond,
        "full_state_materialized": False,
        "values_real": values.detach().cpu().real.tolist(),
        "values_imag": values.detach().cpu().imag.tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
