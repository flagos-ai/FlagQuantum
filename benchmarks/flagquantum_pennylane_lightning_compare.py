#!/usr/bin/env python3
"""Matched single-GPU differentiable statevector system comparison."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.pennylane_lightning_statevector_training import (  # noqa: E402
    build_qnode,
)
from benchmarks.pennylane_lightning_statevector_training import (  # noqa: E402
    parameters as lightning_parameters,
)
from benchmarks.statevector_training_scaling import (  # noqa: E402
    build_full_width_workload,
)
from flagquantum.runtime.executors.statevector.reverse import (  # noqa: E402
    StatevectorCheckpointPolicy,
    execute_torch_distributed_statevector_reverse,
)


def _summary(samples: list[float]) -> dict[str, Any]:
    mean = statistics.fmean(samples)
    return {
        "samples_seconds": samples,
        "sample_count": len(samples),
        "median_seconds": statistics.median(samples),
        "mean_seconds": mean,
        "coefficient_of_variation": statistics.pstdev(samples) / mean,
    }


def _measure(function: Callable[[], tuple[float, tuple[float, ...]]]) -> tuple[
    float, tuple[float, ...], float
]:
    torch.cuda.synchronize()
    started = time.perf_counter()
    value, gradients = function()
    torch.cuda.synchronize()
    return value, gradients, time.perf_counter() - started


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.warmup < 0 or args.repetitions < 3:
        raise ValueError("warmup >= 0 and repetitions >= 3 required")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    os.environ["FQ_STATEVECTOR_TRITON_VJP_ADJOINT"] = "1"
    os.environ["FQ_STATEVECTOR_FUSED_VJP_PIPELINE"] = "0"
    os.environ["FQ_STATEVECTOR_COMM_AWARE_LAYOUT"] = "1"
    os.environ["FQ_STATEVECTOR_REVERSE_EXCHANGE_WORKSPACE"] = "1"
    fq_circuit, fq_parameters, metadata = build_full_width_workload(
        args.n_wires,
        device,
        layers=args.layers,
        seed=args.seed,
        entanglement="linear",
    )
    qnode = build_qnode(args.n_wires, args.layers, mpi=False, mpi_buf_size=64)
    pl_parameters = lightning_parameters(args.n_wires, args.layers, args.seed)

    def flagquantum_value_and_grad() -> tuple[float, tuple[float, ...]]:
        for parameter in fq_parameters:
            parameter.grad = None
        result = execute_torch_distributed_statevector_reverse(
            fq_circuit,
            observable_wire=args.n_wires // 2,
            checkpoint_policy=StatevectorCheckpointPolicy(
                strategy="reversible_adjoint"
            ),
            device=device,
        )
        result.backward()
        return (
            float(result.value.detach()),
            tuple(float(parameter.grad.detach()) for parameter in fq_parameters),
        )

    def lightning_value_and_grad() -> tuple[float, tuple[float, ...]]:
        pl_parameters.grad = None
        result = qnode(pl_parameters)
        result.backward()
        assert pl_parameters.grad is not None
        return (
            float(result.detach()),
            tuple(float(item) for item in pl_parameters.grad.detach()),
        )

    for _ in range(args.warmup):
        _measure(flagquantum_value_and_grad)
        _measure(lightning_value_and_grad)
    samples = {"flagquantum": [], "pennylane_lightning_gpu": []}
    outputs: dict[str, tuple[float, tuple[float, ...]]] = {}
    functions = {
        "flagquantum": flagquantum_value_and_grad,
        "pennylane_lightning_gpu": lightning_value_and_grad,
    }
    for index in range(args.repetitions):
        order = (
            ("flagquantum", "pennylane_lightning_gpu")
            if index % 2 == 0
            else ("pennylane_lightning_gpu", "flagquantum")
        )
        for name in order:
            value, gradients, seconds = _measure(functions[name])
            outputs[name] = (value, gradients)
            samples[name].append(seconds)
    fq_value, fq_gradients = outputs["flagquantum"]
    pl_value, pl_gradients = outputs["pennylane_lightning_gpu"]
    value_error = abs(fq_value - pl_value)
    gradient_error = max(
        abs(actual - expected)
        for actual, expected in zip(fq_gradients, pl_gradients, strict=True)
    )
    fq_timing = _summary(samples["flagquantum"])
    pl_timing = _summary(samples["pennylane_lightning_gpu"])
    tolerance = 3e-5
    return {
        "schema": "flagquantum.pennylane_lightning_gpu_compare.v1",
        "benchmark": "matched_differentiable_statevector_system_comparison",
        "artifact_class": "measured_external_development_comparison",
        "release_gate_allowed": False,
        "scalability_claim_allowed": False,
        "distribution_semantics": "single_device_fast_path",
        "workload": {
            "name": metadata["name"],
            "n_wires": args.n_wires,
            "layers": args.layers,
            "gate_count": metadata["gate_count"],
            "parameter_count": metadata["parameter_count"],
            "gate_set": ["RY", "CNOT"],
            "observable": f"Z({args.n_wires // 2})",
            "dtype": "complex64",
            "seed": args.seed,
        },
        "protocol": {
            "scope": "expectation_value_plus_full_adjoint_parameter_gradient",
            "fixed_parameters_across_samples": True,
            "warmup": args.warmup,
            "repetitions": args.repetitions,
            "synchronization": "cuda_device_synchronized_wall_clock",
            "execution_order": "alternating_per_sample",
            "setup_excluded": ["circuit_construction", "device_creation"],
            "flagquantum_configuration": {
                "triton_vjp_adjoint": True,
                "fused_vjp_pipeline": False,
                "communication_aware_layout": True,
                "reverse_exchange_workspace": True,
            },
            "pennylane_configuration": {
                "device": "lightning.gpu",
                "gradient_method": "adjoint",
                "mpi": False,
            },
        },
        "correctness": {
            "passed": value_error <= tolerance and gradient_error <= tolerance,
            "absolute_tolerance": tolerance,
            "value_absolute_error": value_error,
            "gradient_absolute_error_max": gradient_error,
        },
        "flagquantum": fq_timing,
        "pennylane_lightning_gpu": {
            **pl_timing,
            "backend": "cuQuantum-backed Lightning-GPU",
            "gradient_method": "adjoint",
            "runtime_dependency_of_flagquantum": False,
        },
        "speedup_flagquantum_over_pennylane": (
            pl_timing["median_seconds"] / fq_timing["median_seconds"]
        ),
        "speedup_pennylane_over_flagquantum": (
            fq_timing["median_seconds"] / pl_timing["median_seconds"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-wires", type=int, required=True)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=440044)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()
    payload = run(args)
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    print(encoded)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(encoded + "\n", encoding="utf-8")
    if not payload["correctness"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
