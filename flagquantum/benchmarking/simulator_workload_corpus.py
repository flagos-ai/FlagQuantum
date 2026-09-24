#!/usr/bin/env python3
"""Build and measure a feature-labelled cross-framework workload corpus."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import random
import statistics
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from importlib import import_module, metadata
from pathlib import Path
from typing import Any, Literal, cast

import torch

import flagquantum as fq
from flagquantum.core import CircuitIR

from .contract import runtime_metadata, write_json_atomic
from .simulator_compare import ABSOLUTE_TOLERANCE, SEED
from .simulator_compare import build_workload as build_hwe

SCHEMA = "flagquantum.simulator_workload_corpus.v1"
FEATURE_SCHEMA = "flagquantum.simulator_workload_features.v1"
RUNNER = "simulator_workload_corpus"
_STABILITY_THRESHOLD = 0.20

WorkloadName = Literal[
    "hardware_efficient_statevector",
    "truncated_qft_statevector",
    "random_clifford_statevector",
    "local_brickwork_statevector",
    "dense_nonlocal_statevector",
    "swap_routing_statevector",
]
EngineName = Literal[
    "flagquantum_native",
    "qiskit_aer",
    "cirq_simulator",
    "pennylane_lightning_qubit",
]

WORKLOAD_NAMES: tuple[WorkloadName, ...] = (
    "hardware_efficient_statevector",
    "truncated_qft_statevector",
    "random_clifford_statevector",
    "local_brickwork_statevector",
    "dense_nonlocal_statevector",
    "swap_routing_statevector",
)
ENGINE_NAMES: tuple[EngineName, ...] = (
    "flagquantum_native",
    "qiskit_aer",
    "cirq_simulator",
    "pennylane_lightning_qubit",
)


def _truncated_qft(n_wires: int) -> fq.Circuit:
    circuit = fq.Circuit(n_wires, dtype=torch.complex128)
    interaction_range = min(3, n_wires - 1)
    for target in range(n_wires):
        circuit.h(target)
        for distance in range(1, min(interaction_range, n_wires - target - 1) + 1):
            control = target + distance
            theta = math.pi / (2**distance)
            # A controlled phase expressed through the common portable gate subset.
            circuit.rz(control, theta / 2)
            circuit.rz(target, theta / 2)
            circuit.cx(control, target)
            circuit.rz(target, -theta / 2)
            circuit.cx(control, target)
    for left in range(n_wires // 2):
        circuit.swap(left, n_wires - left - 1)
    return circuit


def _random_clifford(n_wires: int, seed: int) -> fq.Circuit:
    circuit = fq.Circuit(n_wires, dtype=torch.complex128)
    generator = random.Random(seed + 10_007 * n_wires)
    single_qubit_gates = ("h", "s", "x")
    for _ in range(4):
        for wire in range(n_wires):
            getattr(circuit, generator.choice(single_qubit_gates))(wire)
        wires = list(range(n_wires))
        generator.shuffle(wires)
        for index in range(0, n_wires - 1, 2):
            getattr(circuit, generator.choice(("cx", "cz")))(
                wires[index], wires[index + 1]
            )
    return circuit


def _local_brickwork(n_wires: int) -> fq.Circuit:
    circuit = fq.Circuit(n_wires, dtype=torch.complex128)
    for layer in range(4):
        for wire in range(n_wires):
            angle = 0.07 * (layer + 1) * (wire + 1)
            circuit.ry(wire, angle)
            circuit.rz(wire, -0.6 * angle)
        for left in range(layer % 2, n_wires - 1, 2):
            circuit.cx(left, left + 1)
    return circuit


def _dense_nonlocal(n_wires: int) -> fq.Circuit:
    circuit = fq.Circuit(n_wires, dtype=torch.complex128)
    for wire in range(n_wires):
        circuit.h(wire)
    for left in range(n_wires):
        for right in range(left + 1, n_wires):
            circuit.cz(left, right)
    for wire in range(n_wires):
        circuit.ry(wire, 0.031 * (wire + 1))
    return circuit


def _swap_routing(n_wires: int) -> fq.Circuit:
    circuit = fq.Circuit(n_wires, dtype=torch.complex128)
    for wire in range(0, n_wires - 1, 2):
        circuit.h(wire).cx(wire, wire + 1)
    for layer in range(4):
        for wire in range(1, n_wires - 1, 2):
            circuit.swap(wire, wire + 1)
        for wire in range(n_wires):
            circuit.ry(wire, 0.01 * (layer + 1) * (wire + 1))
    return circuit


def build_workload(
    name: WorkloadName,
    *,
    n_wires: int,
    seed: int = SEED,
) -> fq.Circuit:
    """Build one deterministic exact-statevector workload from the corpus."""

    if n_wires < 4:
        raise ValueError("n_wires must be at least 4")
    if name == "hardware_efficient_statevector":
        return build_hwe(n_wires=n_wires, layers=2)
    if name == "truncated_qft_statevector":
        return _truncated_qft(n_wires)
    if name == "random_clifford_statevector":
        return _random_clifford(n_wires, seed)
    if name == "local_brickwork_statevector":
        return _local_brickwork(n_wires)
    if name == "dense_nonlocal_statevector":
        return _dense_nonlocal(n_wires)
    if name == "swap_routing_statevector":
        return _swap_routing(n_wires)
    raise ValueError(f"unsupported workload: {name}")


def _stable_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def extract_features(ir: CircuitIR) -> dict[str, Any]:
    """Extract deterministic, backend-neutral structural features from IR."""

    wire_depth = [0] * ir.n_wires
    histogram: Counter[str] = Counter()
    arity_histogram: Counter[int] = Counter()
    parameterized = 0
    two_qubit_spans: list[int] = []
    for instruction in ir.instructions:
        histogram[instruction.name] += 1
        arity_histogram[len(instruction.wires)] += 1
        if instruction.params or instruction.matrix is not None:
            parameterized += 1
        depth = max(wire_depth[wire] for wire in instruction.wires) + 1
        for wire in instruction.wires:
            wire_depth[wire] = depth
        if len(instruction.wires) == 2:
            two_qubit_spans.append(abs(instruction.wires[0] - instruction.wires[1]))

    two_qubit_count = len(two_qubit_spans)
    features: dict[str, Any] = {
        "schema": FEATURE_SCHEMA,
        "n_wires": ir.n_wires,
        "dtype": ir.dtype,
        "batch_size": int(ir.metadata.get("batch_size", 1)),
        "gate_count": len(ir.instructions),
        "logical_depth": max(wire_depth, default=0),
        "single_qubit_gate_count": arity_histogram[1],
        "two_qubit_gate_count": arity_histogram[2],
        "multi_qubit_gate_count": sum(
            count for arity, count in arity_histogram.items() if arity > 2
        ),
        "parameterized_gate_count": parameterized,
        "gate_density_per_wire": len(ir.instructions) / ir.n_wires,
        "opcode_histogram": dict(sorted(histogram.items())),
        "two_qubit_mean_wire_span": (
            statistics.fmean(two_qubit_spans) if two_qubit_spans else 0.0
        ),
        "two_qubit_max_wire_span": max(two_qubit_spans, default=0),
        "nearest_neighbor_two_qubit_fraction": (
            sum(span == 1 for span in two_qubit_spans) / two_qubit_count
            if two_qubit_count
            else 0.0
        ),
    }
    features["feature_fingerprint"] = _stable_hash(features)
    return features


def _timing(samples: Sequence[float]) -> dict[str, Any]:
    values = tuple(float(value) for value in samples)
    median = statistics.median(values)
    mean = statistics.fmean(values)
    mad = statistics.median(abs(value - median) for value in values)
    return {
        "samples_seconds": values,
        "sample_count": len(values),
        "median_seconds": median,
        "mean_seconds": mean,
        "min_seconds": min(values),
        "max_seconds": max(values),
        "relative_median_absolute_deviation": mad / median if median else 0.0,
    }


def _configure_threads(threads: int) -> dict[str, str]:
    value = str(threads)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = value
    torch.set_num_threads(threads)
    return {
        name: os.environ[name]
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    }


def _engine_callable(
    engine: EngineName,
    circuit: fq.Circuit,
    *,
    seed: int,
    threads: int,
) -> Callable[[], torch.Tensor]:
    if engine == "flagquantum_native":

        def execute_native() -> torch.Tensor:
            return cast(torch.Tensor, circuit.state(refresh=True))

        return execute_native

    if engine == "qiskit_aer":
        qiskit = import_module("flagquantum.ecosystem.qiskit")
        extensions = import_module("flagquantum.ecosystem.extensions")
        backend = qiskit.QiskitAerBackend()
        backend.start(extensions.ExtensionConfig({"seed": seed, "threads": threads}))

        def execute_qiskit() -> torch.Tensor:
            result = backend.execute(circuit)
            if result.state is None:
                raise RuntimeError("qiskit_aer returned no statevector")
            return cast(torch.Tensor, result.state)

        return execute_qiskit

    module_name = {
        "cirq_simulator": "flagquantum.ecosystem.cirq",
        "pennylane_lightning_qubit": "flagquantum.ecosystem.pennylane",
    }[engine]
    run = import_module(module_name).run
    options = fq.ExecutionOptions(seed=seed)

    def execute_external() -> torch.Tensor:
        result = run(circuit, options=options)
        if result.state is None:
            raise RuntimeError(f"{engine} returned no statevector")
        return cast(torch.Tensor, result.state)

    return execute_external


def _engine_versions(engine: EngineName) -> dict[str, str]:
    packages = {
        "flagquantum_native": ("flagquantum",),
        "qiskit_aer": ("qiskit", "qiskit-aer"),
        "cirq_simulator": ("cirq-core",),
        "pennylane_lightning_qubit": ("pennylane", "pennylane-lightning"),
    }[engine]
    return {package: metadata.version(package) for package in packages}


def run_case(
    *,
    workload: WorkloadName,
    n_wires: int,
    engines: Sequence[EngineName],
    warmup: int,
    iterations: int,
    calls_per_sample: int,
    threads: int = 1,
    seed: int = SEED,
) -> dict[str, Any]:
    """Measure one workload and width through each requested user-facing path."""

    if warmup < 0 or iterations < 3 or calls_per_sample < 1:
        raise ValueError(
            "warmup must be non-negative, iterations at least 3, and "
            "calls_per_sample positive"
        )
    if not engines or len(set(engines)) != len(engines):
        raise ValueError("engines must contain unique supported engine names")
    unknown = sorted(set(engines) - set(ENGINE_NAMES))
    if unknown:
        raise ValueError("unsupported engine(s): " + ", ".join(unknown))

    circuit = build_workload(workload, n_wires=n_wires, seed=seed)
    ir = circuit.to_ir()
    functions = {
        engine: _engine_callable(engine, circuit, seed=seed, threads=threads)
        for engine in engines
    }
    outputs: dict[EngineName, torch.Tensor] = {}
    for _ in range(warmup):
        for engine in engines:
            outputs[engine] = functions[engine]()

    samples: dict[EngineName, list[float]] = {engine: [] for engine in engines}
    for iteration in range(iterations):
        offset = iteration % len(engines)
        order = tuple(engines[offset:]) + tuple(engines[:offset])
        for engine in order:
            gc.collect()
            started = time.perf_counter()
            for _ in range(calls_per_sample):
                outputs[engine] = functions[engine]()
            samples[engine].append((time.perf_counter() - started) / calls_per_sample)

    reference_engine = (
        "flagquantum_native" if "flagquantum_native" in engines else engines[0]
    )
    reference = (
        outputs[reference_engine].detach().cpu().to(torch.complex128).reshape(-1)
    )
    engine_payload: dict[str, Any] = {}
    correctness: dict[str, Any] = {}
    for engine in engines:
        state = outputs[engine].detach().cpu().to(torch.complex128).reshape(-1)
        max_error = float(torch.max(torch.abs(state - reference)).item())
        correctness[engine] = {
            "passed": max_error <= ABSOLUTE_TOLERANCE,
            "max_abs_error": max_error,
        }
        engine_payload[engine] = {
            "versions": _engine_versions(engine),
            "end_to_end": _timing(samples[engine]),
        }

    native_seconds = (
        engine_payload["flagquantum_native"]["end_to_end"]["median_seconds"]
        if "flagquantum_native" in engine_payload
        else None
    )
    ratios = (
        {
            engine: payload["end_to_end"]["median_seconds"] / native_seconds
            for engine, payload in engine_payload.items()
            if engine != "flagquantum_native"
        }
        if native_seconds is not None
        else {}
    )
    stability_by_engine = {
        engine: (
            payload["end_to_end"]["relative_median_absolute_deviation"]
            <= _STABILITY_THRESHOLD
        )
        for engine, payload in engine_payload.items()
    }
    return {
        "workload": {
            "name": workload,
            "n_wires": n_wires,
            "seed": seed,
            "batch_size": 1,
            "dtype": "complex128",
            "gate_count": len(ir.instructions),
            "ir_content_hash": ir.content_hash,
        },
        "features": extract_features(ir),
        "correctness": {
            "passed": all(item["passed"] for item in correctness.values()),
            "reference_engine": reference_engine,
            "absolute_tolerance": ABSOLUTE_TOLERANCE,
            "engines": correctness,
        },
        "engines": engine_payload,
        "comparison": {
            "engine_over_flagquantum_median": ratios,
            "ratio_semantics": (
                "values above one mean FlagQuantum is faster for this case"
            ),
        },
        "stability": {
            "maximum_relative_median_absolute_deviation": _STABILITY_THRESHOLD,
            "passed": all(stability_by_engine.values()),
            "engines": stability_by_engine,
        },
    }


def run_benchmark(
    *,
    workloads: Sequence[WorkloadName],
    n_wires: Sequence[int],
    engines: Sequence[EngineName],
    threads: int,
    warmup: int,
    iterations: int,
    calls_per_sample: int,
    seed: int = SEED,
) -> dict[str, Any]:
    """Measure the requested cross-product and return one corpus artifact."""

    if threads < 1:
        raise ValueError("threads must be positive")
    if not workloads or not n_wires:
        raise ValueError("workloads and n_wires must each contain at least one item")
    if len(set(workloads)) != len(workloads):
        raise ValueError("workloads must be unique")
    unknown = sorted(set(workloads) - set(WORKLOAD_NAMES))
    if unknown:
        raise ValueError("unsupported workload(s): " + ", ".join(unknown))
    thread_environment = _configure_threads(threads)
    cases = tuple(
        run_case(
            workload=workload,
            n_wires=width,
            engines=engines,
            warmup=warmup,
            iterations=iterations,
            calls_per_sample=calls_per_sample,
            threads=threads,
            seed=seed,
        )
        for workload in workloads
        for width in n_wires
    )
    passed = all(bool(case["correctness"]["passed"]) for case in cases)
    all_measurements_stable = all(bool(case["stability"]["passed"]) for case in cases)
    return runtime_metadata(
        runner=RUNNER,
        schema=SCHEMA,
        benchmark="cross_framework_simulator_workload_corpus",
        artifact_class="measured_comparison_run",
        benchmark_evidence_class="comparison_non_release",
        claim_evidence_type="unknown",
        distribution_semantics="single_process_single_device",
        scalability_claim_allowed=False,
        release_gate_allowed=False,
        non_release_evidence=True,
        scalability_blockers=("comparison_result_not_release_scalability_evidence",),
        passed=passed,
        correctness_passed=passed,
        all_measurements_stable=all_measurements_stable,
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
            "calls_per_sample": calls_per_sample,
            "measurement_scope": "end_to_end_user_facing_run_call",
            "engine_order": "rotated_per_iteration_within_one_process",
            "conversion_included": True,
            "compilation_or_device_preparation_included": True,
            "result_retrieval_included": True,
            "exact_statevector": True,
            "hidden_fallback_allowed": False,
            "qiskit_max_parallel_threads": threads,
            "feature_schema": FEATURE_SCHEMA,
        },
        workloads=tuple(workloads),
        n_wires=tuple(n_wires),
        engines=tuple(engines),
        cases=cases,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workloads", nargs="+", choices=WORKLOAD_NAMES, default=WORKLOAD_NAMES
    )
    parser.add_argument("--n-wires", type=int, nargs="+", default=(10, 14, 18, 22))
    parser.add_argument(
        "--engines", nargs="+", choices=ENGINE_NAMES, default=ENGINE_NAMES
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--calls-per-sample", type=int, default=1)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    payload = run_benchmark(
        workloads=tuple(args.workloads),
        n_wires=tuple(args.n_wires),
        engines=tuple(args.engines),
        threads=args.threads,
        warmup=args.warmup,
        iterations=args.iterations,
        calls_per_sample=args.calls_per_sample,
        seed=args.seed,
    )
    if args.json_output is not None:
        write_json_atomic(args.json_output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = (
    "ENGINE_NAMES",
    "FEATURE_SCHEMA",
    "RUNNER",
    "SCHEMA",
    "WORKLOAD_NAMES",
    "build_workload",
    "extract_features",
    "main",
    "run_benchmark",
    "run_case",
)
