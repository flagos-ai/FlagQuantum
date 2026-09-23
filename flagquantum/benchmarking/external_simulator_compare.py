#!/usr/bin/env python3
"""Measure one optional external simulator against FlagQuantum IR semantics."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import time
from collections.abc import Callable, Sequence
from importlib import import_module, metadata
from pathlib import Path
from typing import Any, Literal

import torch

from flagquantum.ecosystem.cirq import export_cirq
from flagquantum.ecosystem.pennylane import export_pennylane

from .contract import runtime_metadata, write_json_atomic
from .simulator_compare import (
    ABSOLUTE_TOLERANCE,
    SCHEMA,
    SEED,
    _measure_calls,
    _timing,
    build_workload,
)

EngineName = Literal["cirq_simulator", "pennylane_lightning_qubit"]
_ENGINE_NAMES: tuple[EngineName, ...] = (
    "cirq_simulator",
    "pennylane_lightning_qubit",
)
_STABILITY_THRESHOLD = 0.10
_RUNNER_NAMES: dict[EngineName, str] = {
    "cirq_simulator": "simulator_compare_cirq",
    "pennylane_lightning_qubit": "simulator_compare_pennylane",
}


def _configure_threads(threads: int) -> dict[str, str]:
    """Apply thread limits before importing an optional simulator backend."""

    value = str(threads)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = value
    torch.set_num_threads(threads)
    return {
        name: os.environ[name]
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    }


def _prepare_cirq(
    circuit: Any,
    *,
    setup_iterations: int,
) -> tuple[dict[str, Any], Callable[[], Any]]:
    try:
        cirq = import_module("cirq")
        numpy = import_module("numpy")
    except ImportError as exc:
        raise RuntimeError(
            "simulator_compare_cirq requires the cirq optional dependency; "
            "install it with `pip install 'flagquantum[cirq]'`"
        ) from exc
    conversion_samples, converted = _measure_calls(
        lambda: export_cirq(circuit.to_ir()), setup_iterations
    )
    simulator = cirq.Simulator(dtype=numpy.complex128, seed=SEED)
    qubit_order = cirq.LineQubit.range(circuit.n_wires)

    def execute() -> Any:
        result = simulator.simulate(converted.circuit, qubit_order=qubit_order)
        return result.final_state_vector.copy()

    return (
        {
            "version": metadata.version("cirq-core"),
            "backend": "cirq.Simulator",
            "conversion": _timing(conversion_samples),
            "conversion_report": converted.report.to_dict(),
            "compilation": {
                "applicable": False,
                "note": "Cirq simulator preparation is included in execution",
            },
        },
        execute,
    )


def _prepare_pennylane(
    circuit: Any,
    *,
    setup_iterations: int,
) -> tuple[dict[str, Any], Callable[[], Any]]:
    try:
        qml = import_module("pennylane")
        numpy = import_module("numpy")
    except ImportError as exc:
        raise RuntimeError(
            "simulator_compare_pennylane requires the PennyLane optional "
            "dependency; install it with `pip install 'flagquantum[pennylane]'`"
        ) from exc
    conversion_samples, converted = _measure_calls(
        lambda: export_pennylane(circuit.to_ir()), setup_iterations
    )
    executable = qml.tape.QuantumScript(
        converted.quantum_script.operations,
        measurements=(qml.state(),),
        shots=None,
    )
    device = qml.device(
        "lightning.qubit",
        wires=range(circuit.n_wires),
        shots=None,
    )

    def execute() -> Any:
        return numpy.asarray(device.execute(executable)).copy()

    return (
        {
            "version": metadata.version("pennylane"),
            "backend": "lightning.qubit",
            "backend_version": metadata.version("pennylane-lightning"),
            "conversion": _timing(conversion_samples),
            "conversion_report": converted.report.to_dict(),
            "compilation": {
                "applicable": False,
                "note": "PennyLane device preprocessing is included in execution",
            },
        },
        execute,
    )


def _prepare_engine(
    engine: EngineName,
    circuit: Any,
    *,
    setup_iterations: int,
) -> tuple[dict[str, Any], Callable[[], Any]]:
    if engine == "cirq_simulator":
        return _prepare_cirq(circuit, setup_iterations=setup_iterations)
    if engine == "pennylane_lightning_qubit":
        return _prepare_pennylane(circuit, setup_iterations=setup_iterations)
    raise ValueError(f"unsupported external simulator engine: {engine}")


def run_case(
    *,
    engine: EngineName,
    n_wires: int,
    layers: int,
    threads: int,
    warmup: int,
    iterations: int,
    setup_iterations: int,
    calls_per_sample: int,
) -> dict[str, Any]:
    """Measure one external simulator without timing FlagQuantum execution."""

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
    _configure_threads(threads)
    circuit = build_workload(n_wires=n_wires, layers=layers)
    engine_setup, execute = _prepare_engine(
        engine,
        circuit,
        setup_iterations=setup_iterations,
    )

    reference = circuit.state(refresh=True)[0].detach().cpu().to(torch.complex128)
    gc.collect()
    started = time.perf_counter()
    output = execute()
    cold_execution = time.perf_counter() - started
    for _ in range(warmup):
        output = execute()
    samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        for _ in range(calls_per_sample):
            output = execute()
        samples.append((time.perf_counter() - started) / calls_per_sample)
    state = torch.as_tensor(output).reshape(-1).to(torch.complex128)
    max_error = float(torch.max(torch.abs(state - reference)).item())
    steady_state = _timing(samples)
    conversion_median = float(engine_setup["conversion"]["median_seconds"])
    engine_payload = {
        **engine_setup,
        "cold_execution_seconds": cold_execution,
        "first_result_seconds": conversion_median + cold_execution,
        "steady_state": steady_state,
    }
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
            "flagquantum_reference_timed": False,
        },
        "engines": {engine: engine_payload},
        "stability": {
            "maximum_relative_median_absolute_deviation": _STABILITY_THRESHOLD,
            "passed": (
                steady_state["relative_median_absolute_deviation"]
                <= _STABILITY_THRESHOLD
            ),
        },
    }


def run_benchmark(
    *,
    engine: EngineName,
    n_wires: Sequence[int],
    layers: int,
    threads: int,
    warmup: int,
    iterations: int,
    setup_iterations: int,
    calls_per_sample: int,
) -> dict[str, Any]:
    """Measure one optional simulator over the standard workload matrix."""

    if engine not in _ENGINE_NAMES:
        raise ValueError(f"unsupported external simulator engine: {engine}")
    if not n_wires:
        raise ValueError("n_wires must contain at least one workload size")
    thread_environment = _configure_threads(threads)
    cases = tuple(
        run_case(
            engine=engine,
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
        runner=_RUNNER_NAMES[engine],
        schema=SCHEMA,
        benchmark="interoperable_simulator_comparison",
        artifact_class="measured_external_engine_run",
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
        measured_engine=engine,
        environment={
            "machine": platform.machine(),
            "processor": platform.processor() or "unknown",
            "torch": torch.__version__,
            "torch_threads": threads,
            "thread_environment": thread_environment,
            "device": "cpu",
        },
        methodology={
            "warmup": warmup,
            "iterations": iterations,
            "setup_iterations": setup_iterations,
            "calls_per_sample": calls_per_sample,
            "measurement_scope": "external_engine_only",
            "flagquantum_performance_measured": False,
            "flagquantum_correctness_reference_executions_per_case": 1,
            "conversion_included_in_steady_state": False,
            "compilation_included_in_steady_state": False,
            "result_retrieval_included_in_execution": True,
            "output_basis_normalization_included_in_execution": False,
            "hidden_fallback_allowed": False,
        },
        cases=cases,
    )


def _main(engine: EngineName) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-wires", type=int, nargs="+", default=(10, 14, 18, 22, 24))
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=9)
    parser.add_argument("--setup-iterations", type=int, default=3)
    parser.add_argument("--calls-per-sample", type=int, default=10)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    payload = run_benchmark(
        engine=engine,
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


def cirq_main() -> int:
    return _main("cirq_simulator")


def pennylane_main() -> int:
    return _main("pennylane_lightning_qubit")


__all__ = (
    "EngineName",
    "cirq_main",
    "pennylane_main",
    "run_benchmark",
    "run_case",
)
