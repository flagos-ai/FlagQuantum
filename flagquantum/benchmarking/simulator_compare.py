#!/usr/bin/env python3
"""Reproducible statevector comparison across interoperable simulators."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import time
from collections.abc import Callable, Sequence
from importlib import import_module, metadata
from pathlib import Path
from typing import Any, cast

import torch

import flagquantum as fq
from flagquantum.ecosystem.qiskit import (
    qiskit_statevector_to_flagquantum,
    to_qiskit,
)

from .contract import runtime_metadata, write_json_atomic

SCHEMA = "flagquantum.simulator_comparison.v1"
RUNNER = "simulator_compare"
SEED = 7319
ABSOLUTE_TOLERANCE = 1e-10


def build_workload(*, n_wires: int, layers: int) -> fq.Circuit:
    """Build one deterministic dense statevector workload in FlagQuantum IR."""

    if n_wires < 4 or layers < 1:
        raise ValueError("n_wires must be at least 4 and layers must be positive")
    circuit = fq.Circuit(n_wires, dtype=torch.complex128)
    parameter_count = layers * n_wires * 3
    values = torch.linspace(-0.43, 0.37, steps=parameter_count).reshape(
        layers, n_wires, 3
    )
    for layer in range(layers):
        for wire in range(n_wires):
            circuit.rx(wire, float(values[layer, wire, 0]))
            circuit.ry(wire, float(values[layer, wire, 1]))
            circuit.rz(wire, float(values[layer, wire, 2]))
        for wire in range(n_wires - 1):
            circuit.cx(wire, wire + 1)
        circuit.rzz(0, n_wires - 1, float(values[layer, 0, 0] * 0.25))
        circuit.swap(1, n_wires - 2)
    return circuit


def _measure_calls(
    function: Callable[[], Any], count: int
) -> tuple[tuple[float, ...], Any]:
    samples = []
    output = None
    for _ in range(count):
        started = time.perf_counter()
        output = function()
        samples.append(time.perf_counter() - started)
    return tuple(samples), output


def _timing(samples: Sequence[float]) -> dict[str, Any]:
    values = tuple(float(value) for value in samples)
    mean = statistics.fmean(values)
    median = statistics.median(values)
    median_absolute_deviation = statistics.median(
        abs(value - median) for value in values
    )
    return {
        "samples_seconds": values,
        "sample_count": len(values),
        "median_seconds": median,
        "mean_seconds": mean,
        "min_seconds": min(values),
        "max_seconds": max(values),
        "coefficient_of_variation": (
            statistics.pstdev(values) / mean if len(values) > 1 and mean else 0.0
        ),
        "relative_median_absolute_deviation": (
            median_absolute_deviation / median if median else 0.0
        ),
    }


def _qiskit_runtime(
    circuit: fq.Circuit,
    *,
    threads: int,
    setup_iterations: int,
) -> tuple[dict[str, Any], Callable[[], Any], Any]:
    try:
        qiskit = import_module("qiskit")
        aer = import_module("qiskit_aer")
    except ImportError as exc:
        raise RuntimeError(
            "simulator_compare requires the qiskit optional dependency; install "
            "it with `pip install 'flagquantum[qiskit]'`"
        ) from exc

    conversion_samples, qiskit_circuit = _measure_calls(
        lambda: to_qiskit(circuit), setup_iterations
    )
    qiskit_circuit.save_statevector()
    backend = aer.AerSimulator(
        method="statevector",
        device="CPU",
        max_parallel_threads=threads,
        max_parallel_experiments=1,
        max_parallel_shots=1,
    )
    compilation_samples, compiled = _measure_calls(
        lambda: qiskit.transpile(
            qiskit_circuit,
            backend,
            optimization_level=0,
            seed_transpiler=SEED,
        ),
        setup_iterations,
    )

    def execute() -> Any:
        result = backend.run(compiled, seed_simulator=SEED).result()
        return result.get_statevector(compiled)

    setup = {
        "conversion": _timing(conversion_samples),
        "compilation": _timing(compilation_samples),
    }
    return setup, execute, compiled


def _interleaved_samples(
    native: Callable[[], Any],
    qiskit_aer: Callable[[], Any],
    *,
    warmup: int,
    iterations: int,
    calls_per_sample: int,
) -> tuple[tuple[float, ...], tuple[float, ...], Any, Any]:
    native_output = qiskit_output = None
    for _ in range(warmup):
        native_output = native()
        qiskit_output = qiskit_aer()
    native_samples: list[float] = []
    qiskit_samples: list[float] = []
    for iteration in range(iterations):
        ordered = (
            ((native, native_samples), (qiskit_aer, qiskit_samples))
            if iteration % 2 == 0
            else ((qiskit_aer, qiskit_samples), (native, native_samples))
        )
        for function, samples in ordered:
            started = time.perf_counter()
            for _ in range(calls_per_sample):
                output = function()
            samples.append((time.perf_counter() - started) / calls_per_sample)
            if function is native:
                native_output = output
            else:
                qiskit_output = output
    return (
        tuple(native_samples),
        tuple(qiskit_samples),
        native_output,
        qiskit_output,
    )


def run_case(
    *,
    n_wires: int,
    layers: int,
    threads: int,
    warmup: int,
    iterations: int,
    setup_iterations: int,
    calls_per_sample: int,
) -> dict[str, Any]:
    """Measure one equivalent FlagQuantum and Qiskit Aer workload."""

    if (
        threads < 1
        or warmup < 0
        or iterations < 3
        or setup_iterations < 1
        or calls_per_sample < 1
    ):
        raise ValueError(
            "threads, setup_iterations, and calls_per_sample must be positive; "
            "warmup must be non-negative and iterations at least 3"
        )
    torch.set_num_threads(threads)
    circuit = build_workload(n_wires=n_wires, layers=layers)
    setup, qiskit_execute, _compiled = _qiskit_runtime(
        circuit, threads=threads, setup_iterations=setup_iterations
    )

    def native_execute() -> torch.Tensor:
        return cast(torch.Tensor, circuit.state(refresh=True))

    gc.collect()
    started = time.perf_counter()
    native_cold_output = native_execute()
    native_cold = time.perf_counter() - started
    gc.collect()
    started = time.perf_counter()
    qiskit_cold_output = qiskit_execute()
    qiskit_cold = time.perf_counter() - started

    native_samples, qiskit_samples, native_output, qiskit_output = _interleaved_samples(
        native_execute,
        qiskit_execute,
        warmup=warmup,
        iterations=iterations,
        calls_per_sample=calls_per_sample,
    )
    assert native_output is not None and qiskit_output is not None
    native_state = native_output[0].detach().cpu().to(torch.complex128)
    qiskit_state = qiskit_statevector_to_flagquantum(qiskit_output.data, n_wires).to(
        torch.complex128
    )
    cold_qiskit_state = qiskit_statevector_to_flagquantum(
        qiskit_cold_output.data, n_wires
    ).to(torch.complex128)
    max_error = max(
        float(torch.max(torch.abs(native_state - qiskit_state)).item()),
        float(
            torch.max(
                torch.abs(native_cold_output[0].detach().cpu() - cold_qiskit_state)
            ).item()
        ),
    )
    native_timing = _timing(native_samples)
    qiskit_timing = _timing(qiskit_samples)
    stability_threshold = 0.10
    return {
        "workload": {
            "name": "hardware_efficient_statevector",
            "n_wires": n_wires,
            "layers": layers,
            "gate_count": len(circuit),
            "batch_size": 1,
            "dtype": "complex128",
            "seed": SEED,
        },
        "correctness": {
            "passed": max_error <= ABSOLUTE_TOLERANCE,
            "max_abs_error": max_error,
            "absolute_tolerance": ABSOLUTE_TOLERANCE,
            "reference_semantics": "same_flagquantum_ir_exact_statevector",
        },
        "engines": {
            "flagquantum_native": {
                "version": metadata.version("flagquantum"),
                "conversion": {"applicable": False},
                "compilation": {
                    "applicable": False,
                    "note": "internal preparation is included in cold execution",
                },
                "cold_execution_seconds": native_cold,
                "steady_state": native_timing,
                "first_result_seconds": native_cold,
            },
            "qiskit_aer": {
                "version": metadata.version("qiskit-aer"),
                "qiskit_version": metadata.version("qiskit"),
                **setup,
                "cold_execution_seconds": qiskit_cold,
                "steady_state": qiskit_timing,
                "first_result_seconds": (
                    setup["conversion"]["median_seconds"]
                    + setup["compilation"]["median_seconds"]
                    + qiskit_cold
                ),
            },
        },
        "comparison": {
            "steady_state_qiskit_over_flagquantum": (
                qiskit_timing["median_seconds"] / native_timing["median_seconds"]
            ),
            "first_result_qiskit_over_flagquantum": (
                setup["conversion"]["median_seconds"]
                + setup["compilation"]["median_seconds"]
                + qiskit_cold
            )
            / native_cold,
            "ratio_semantics": (
                "values above one mean FlagQuantum is faster for this workload"
            ),
        },
        "stability": {
            "maximum_relative_median_absolute_deviation": stability_threshold,
            "passed": (
                native_timing["relative_median_absolute_deviation"]
                <= stability_threshold
                and qiskit_timing["relative_median_absolute_deviation"]
                <= stability_threshold
            ),
        },
    }


def run_benchmark(
    *,
    n_wires: Sequence[int],
    layers: int,
    threads: int,
    warmup: int,
    iterations: int,
    setup_iterations: int,
    calls_per_sample: int,
) -> dict[str, Any]:
    """Run all requested sizes and return the comparison evidence payload."""

    if not n_wires:
        raise ValueError("n_wires must contain at least one workload size")
    cases = tuple(
        run_case(
            n_wires=width,
            layers=layers,
            threads=threads,
            warmup=warmup,
            iterations=iterations,
            setup_iterations=setup_iterations,
            calls_per_sample=calls_per_sample,
        )
        for width in n_wires
    )
    correctness_passed = all(bool(case["correctness"]["passed"]) for case in cases)
    all_measurements_stable = all(bool(case["stability"]["passed"]) for case in cases)
    return runtime_metadata(
        runner=RUNNER,
        schema=SCHEMA,
        benchmark="interoperable_simulator_comparison",
        artifact_class="measured_comparison_run",
        benchmark_evidence_class="comparison_non_release",
        claim_evidence_type="unknown",
        distribution_semantics="single_process_single_device",
        scalability_claim_allowed=False,
        release_gate_allowed=False,
        non_release_evidence=True,
        scalability_blockers=("comparison_result_not_release_scalability_evidence",),
        passed=correctness_passed,
        correctness_passed=correctness_passed,
        all_measurements_stable=all_measurements_stable,
        environment={
            "machine": platform.machine(),
            "processor": platform.processor() or "unknown",
            "torch": torch.__version__,
            "torch_threads": threads,
            "device": "cpu",
        },
        methodology={
            "warmup": warmup,
            "iterations": iterations,
            "setup_iterations": setup_iterations,
            "calls_per_sample": calls_per_sample,
            "steady_state_order": "alternating_within_one_process",
            "conversion_included_in_steady_state": False,
            "compilation_included_in_steady_state": False,
            "result_retrieval_included_in_execution": True,
            "output_basis_normalization_included_in_execution": False,
            "qiskit_transpile_optimization_level": 0,
            "hidden_fallback_allowed": False,
        },
        cases=cases,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-wires", type=int, nargs="+", default=(10, 14, 18, 22, 24))
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=9)
    parser.add_argument("--setup-iterations", type=int, default=3)
    parser.add_argument("--calls-per-sample", type=int, default=5)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    payload = run_benchmark(
        n_wires=tuple(args.n_wires),
        layers=args.layers,
        threads=args.threads,
        warmup=args.warmup,
        iterations=args.iterations,
        setup_iterations=args.setup_iterations,
        calls_per_sample=args.calls_per_sample,
    )
    if args.json_output is not None:
        write_json_atomic(args.json_output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
