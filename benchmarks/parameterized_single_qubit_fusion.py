"""A/B benchmark for differentiable native single-qubit region fusion."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch

import flagquantum as fq


def _circuit(n_wires: int, layers: int, parameters: torch.Tensor) -> fq.Circuit:
    circuit = fq.Circuit(n_wires, device="cuda", dtype=torch.complex64)
    slot = 0
    for _ in range(layers):
        for wire in range(n_wires):
            circuit.rx(wire, parameters[slot]).ry(wire, parameters[slot + 1]).rz(
                wire, parameters[slot + 2]
            )
            slot += 3
        for wire in range(n_wires - 1):
            circuit.cx(wire, wire + 1)
    return circuit


def _measure(circuit, parameters, warmup, iterations):
    samples = []
    for index in range(warmup + iterations):
        parameters.grad = None
        started = time.perf_counter()
        state = circuit.state(refresh=True)
        observed = state.real[:, :1024]
        weights = torch.linspace(0.5, 1.5, observed.shape[1], device=state.device)
        loss = (observed.square() * weights).sum()
        loss.backward()
        torch.cuda.synchronize()
        if index >= warmup:
            samples.append(time.perf_counter() - started)
    return {
        "median_seconds": statistics.median(samples),
        "mean_seconds": statistics.mean(samples),
        "coefficient_of_variation": statistics.stdev(samples)
        / statistics.mean(samples),
        "samples_seconds": samples,
    }


def _profile(circuit, parameters):
    parameters.grad = None
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profiler:
        state = circuit.state(refresh=True)
        observed = state.real[:, :1024]
        weights = torch.linspace(0.5, 1.5, observed.shape[1], device=state.device)
        (observed.square() * weights).sum().backward()
        torch.cuda.synchronize()
    rows = sorted(
        profiler.key_averages(),
        key=lambda row: row.self_device_time_total,
        reverse=True,
    )
    return [
        {
            "operator": row.key,
            "self_device_time_us": row.self_device_time_total,
            "device_time_us": row.device_time_total,
            "count": row.count,
        }
        for row in rows[:20]
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-wires", type=int, default=20)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--execution-order",
        choices=("fallback-first", "fused-first"),
        default="fallback-first",
    )
    args = parser.parse_args()
    count = args.n_wires * args.layers * 3
    initial = torch.linspace(-0.4, 0.4, count, device="cuda")
    results = {}
    gradients = {}
    order = (
        (("fallback", "0"), ("fused", "1"))
        if args.execution_order == "fallback-first"
        else (("fused", "1"), ("fallback", "0"))
    )
    for label, enabled in order:
        os.environ["FQ_TRITON_SINGLE_QUBIT_MATRIX"] = enabled
        os.environ["FQ_TRITON_PARAMETERIZED_SINGLE_QUBIT_MATRIX"] = enabled
        parameters = initial.detach().clone().requires_grad_(True)
        circuit = _circuit(args.n_wires, args.layers, parameters)
        results[label] = _measure(circuit, parameters, args.warmup, args.iterations)
        if args.profile:
            results[label]["profile_top_device_operators"] = _profile(
                circuit, parameters
            )
        gradients[label] = parameters.grad.detach().clone()
    error = float((gradients["fallback"] - gradients["fused"]).abs().max().cpu())
    payload = {
        "schema": "flagquantum.parameterized_single_qubit_fusion.v1",
        "benchmark": "parameterized_single_qubit_fusion_training_step",
        "workload": {
            "n_wires": args.n_wires,
            "layers": args.layers,
            "parameter_count": count,
            "warmup": args.warmup,
            "iterations": args.iterations,
        },
        "execution_order": args.execution_order,
        "fallback": results["fallback"],
        "fused": results["fused"],
        "speedup": results["fallback"]["median_seconds"]
        / results["fused"]["median_seconds"],
        "maximum_parameter_gradient_error": error,
        "stability": {
            "maximum_coefficient_of_variation": 0.10,
            "fallback_passed": results["fallback"]["coefficient_of_variation"] <= 0.10,
            "fused_passed": results["fused"]["coefficient_of_variation"] <= 0.10,
            "comparison_claim_allowed": results["fallback"]["coefficient_of_variation"]
            <= 0.10
            and results["fused"]["coefficient_of_variation"] <= 0.10,
        },
        "release_gate_allowed": False,
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    print(encoded)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
