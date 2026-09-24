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
from copy import deepcopy
from importlib import import_module, metadata
from pathlib import Path
from textwrap import wrap
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

_ENGINE_LABELS: dict[EngineName, str] = {
    "flagquantum_native": "FlagQuantum",
    "qiskit_aer": "Qiskit Aer",
    "cirq_simulator": "Cirq",
    "pennylane_lightning_qubit": "PennyLane",
}
_WORKLOAD_LABELS: dict[WorkloadName, str] = {
    "hardware_efficient_statevector": "Hardware-efficient",
    "truncated_qft_statevector": "Truncated QFT",
    "random_clifford_statevector": "Random Clifford",
    "local_brickwork_statevector": "Local brickwork",
    "dense_nonlocal_statevector": "Dense nonlocal",
    "swap_routing_statevector": "SWAP routing",
}


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


def _case_key(case: Mapping[str, Any]) -> tuple[str, int]:
    workload = case.get("workload")
    if not isinstance(workload, Mapping):
        raise ValueError("corpus case requires workload metadata")
    name = workload.get("name")
    n_wires = workload.get("n_wires")
    if not isinstance(name, str) or not isinstance(n_wires, int):
        raise ValueError("corpus case requires workload name and integer n_wires")
    return name, n_wires


def _recompute_case_summary(case: dict[str, Any]) -> None:
    engines = case["engines"]
    native_seconds = engines["flagquantum_native"]["end_to_end"]["median_seconds"]
    if not isinstance(native_seconds, (int, float)) or native_seconds <= 0:
        raise ValueError("refreshed native median must be positive")
    case["comparison"]["engine_over_flagquantum_median"] = {
        engine: payload["end_to_end"]["median_seconds"] / native_seconds
        for engine, payload in engines.items()
        if engine != "flagquantum_native"
    }
    correctness = case["correctness"]["engines"]
    case["correctness"]["passed"] = all(
        bool(payload["passed"]) for payload in correctness.values()
    )
    stability = case["stability"]["engines"]
    case["stability"]["passed"] = all(bool(value) for value in stability.values())


def _refresh_benchmark(
    baseline: Mapping[str, Any], measured: Mapping[str, Any]
) -> dict[str, Any]:
    """Merge native-only measurements into a compatible corpus artifact.

    The baseline remains authoritative for every unmeasured engine. Workload
    identity, methodology, platform, and runtime environment must match before
    any timing is replaced.
    """

    for label, payload in (("baseline", baseline), ("measured", measured)):
        if payload.get("runner") != RUNNER or payload.get("schema") != SCHEMA:
            raise ValueError(f"{label} is not a compatible workload corpus")
    if measured.get("engines") != ["flagquantum_native"] and measured.get(
        "engines"
    ) != ("flagquantum_native",):
        raise ValueError("refresh measurements must contain only flagquantum_native")
    baseline_engines = baseline.get("engines")
    if (
        not isinstance(baseline_engines, Sequence)
        or isinstance(baseline_engines, (str, bytes))
        or "flagquantum_native" not in baseline_engines
        or len(baseline_engines) < 2
    ):
        raise ValueError("baseline must contain native and external engines")
    for field in ("methodology", "environment", "platform", "python"):
        if baseline.get(field) != measured.get(field):
            raise ValueError(f"refresh {field} does not match the baseline")

    baseline_cases = baseline.get("cases")
    measured_cases = measured.get("cases")
    if not isinstance(baseline_cases, Sequence) or not isinstance(
        measured_cases, Sequence
    ):
        raise ValueError("corpus artifacts require case sequences")
    result = deepcopy(dict(baseline))
    result_cases = result["cases"]
    indexed = {_case_key(case): case for case in result_cases}
    if len(indexed) != len(result_cases):
        raise ValueError("baseline contains duplicate workload cases")

    refreshed_keys: list[tuple[str, int]] = []
    for measured_case in measured_cases:
        key = _case_key(measured_case)
        if key in refreshed_keys:
            raise ValueError("refresh measurements contain duplicate workload cases")
        baseline_case = indexed.get(key)
        if baseline_case is None:
            raise ValueError(f"refresh case is absent from baseline: {key[0]} {key[1]}")
        for field in ("workload", "features"):
            if baseline_case[field] != measured_case[field]:
                raise ValueError(f"refresh case {field} does not match baseline: {key}")
        if set(baseline_case["engines"]) != set(baseline_engines):
            raise ValueError(f"baseline case engine set is inconsistent: {key}")
        measured_engines = measured_case["engines"]
        if set(measured_engines) != {"flagquantum_native"}:
            raise ValueError("each refresh case must contain only flagquantum_native")
        baseline_case["engines"]["flagquantum_native"] = deepcopy(
            measured_engines["flagquantum_native"]
        )
        baseline_case["correctness"]["engines"]["flagquantum_native"] = deepcopy(
            measured_case["correctness"]["engines"]["flagquantum_native"]
        )
        baseline_case["stability"]["engines"]["flagquantum_native"] = bool(
            measured_case["stability"]["engines"]["flagquantum_native"]
        )
        _recompute_case_summary(baseline_case)
        refreshed_keys.append(key)

    if not refreshed_keys:
        raise ValueError("refresh measurements contain no cases")
    result["correctness_passed"] = all(
        bool(case["correctness"]["passed"]) for case in result_cases
    )
    result["passed"] = result["correctness_passed"]
    result["all_measurements_stable"] = all(
        bool(case["stability"]["passed"]) for case in result_cases
    )
    history = list(result.get("refresh_history", ()))
    history.append(
        {
            "engines": ["flagquantum_native"],
            "preserved_engines": [
                engine for engine in baseline_engines if engine != "flagquantum_native"
            ],
            "cases": [
                {"workload": workload, "n_wires": n_wires}
                for workload, n_wires in refreshed_keys
            ],
            "workloads": list(measured["workloads"]),
            "n_wires": list(measured["n_wires"]),
            "platform": measured["platform"],
            "python": measured["python"],
            "environment": deepcopy(measured["environment"]),
            "methodology": deepcopy(measured["methodology"]),
        }
    )
    result["refresh_history"] = history
    return result


def _render_markdown(payload: Mapping[str, Any], *, artifact_name: str) -> str:
    """Render a self-contained comparison table from a corpus artifact."""

    cases = payload["cases"]
    engines = tuple(payload["engines"])
    methodology = payload["methodology"]
    stable_groups = sum(
        bool(stable)
        for case in cases
        for stable in case["stability"]["engines"].values()
    )
    total_groups = sum(len(case["engines"]) for case in cases)
    warmup_label = "warmup run" if methodology["warmup"] == 1 else "warmup runs"
    call_label = "call" if methodology["calls_per_sample"] == 1 else "calls"
    methodology_summary = (
        f"Each median uses {methodology['iterations']} retained samples after "
        f"{methodology['warmup']} {warmup_label}, with "
        f"{methodology['calls_per_sample']} measured {call_label} per sample."
    )
    stability_summary = (
        f"{stable_groups} of {total_groups} timing groups meet the declared 20% "
        "RMAD threshold; unstable measurements remain visible in the raw artifact."
    )
    version_summary = ", ".join(
        f"{package} {version}"
        for engine in engines
        for package, version in cases[0]["engines"][engine]["versions"].items()
    )
    lines = [
        "# Cross-framework simulator workload corpus (Apple arm64 CPU)",
        "",
        f"This report is generated from [`{artifact_name}`]({artifact_name}).",
        "All workloads use exact complex128 statevectors on one CPU thread. Timings",
        "include conversion, backend preparation, execution, and result retrieval.",
        *wrap(methodology_summary, width=88),
        *wrap(stability_summary, width=88),
        *wrap(
            f"Environment: {payload['platform']}; Python {payload['python']}; "
            f"{version_summary}.",
            width=88,
        ),
        "",
    ]
    if payload.get("refresh_history"):
        refreshed = payload["refresh_history"][-1]
        preserved_labels = [
            _ENGINE_LABELS[cast(EngineName, engine)]
            for engine in refreshed["preserved_engines"]
        ]
        preserved = (
            preserved_labels[0]
            if len(preserved_labels) == 1
            else ", ".join(preserved_labels[:-1]) + ", and " + preserved_labels[-1]
        )
        refresh_summary = (
            "The latest refresh remeasured only FlagQuantum native. Existing "
            f"{preserved} measurements were preserved unchanged; comparison ratios "
            "were recomputed from the refreshed native medians."
        )
        lines.extend(
            (
                *wrap(refresh_summary, width=88),
                "",
            )
        )

    external = tuple(engine for engine in engines if engine != "flagquantum_native")
    header = ["Workload", "Qubits", "Gates", "Depth"]
    header.extend(
        f"{_ENGINE_LABELS[cast(EngineName, engine)]} (s)" for engine in engines
    )
    header.extend(
        f"{_ENGINE_LABELS[cast(EngineName, engine)]} / FQ" for engine in external
    )
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(("---", *("---:" for _ in header[1:]))) + " |")
    for case in cases:
        workload = cast(WorkloadName, case["workload"]["name"])
        row = [
            _WORKLOAD_LABELS[workload],
            str(case["workload"]["n_wires"]),
            str(case["features"]["gate_count"]),
            str(case["features"]["logical_depth"]),
        ]
        row.extend(
            f"{case['engines'][engine]['end_to_end']['median_seconds']:.6f}"
            for engine in engines
        )
        ratios = case["comparison"]["engine_over_flagquantum_median"]
        row.extend(f"{ratios[engine]:.2f}x" for engine in external)
        lines.append("| " + " | ".join(row) + " |")
    latest_refresh = payload.get("refresh_history", [{}])[-1]
    reproduced_workloads = latest_refresh.get("workloads", payload["workloads"])
    reproduced_widths = latest_refresh.get("n_wires", payload["n_wires"])
    reproduction = [
        "flagquantum-benchmark run simulator_workload_corpus \\",
        f"  --workloads {' '.join(reproduced_workloads)} \\",
        f"  --n-wires {' '.join(str(value) for value in reproduced_widths)} \\",
    ]
    if payload.get("refresh_history"):
        reproduction_intro = (
            "Refresh only FlagQuantum while preserving the checked-in external data:"
        )
        reproduction.extend(
            (
                "  --engines flagquantum_native --threads 1 \\",
                f"  --warmup {methodology['warmup']} \\",
                f"  --iterations {methodology['iterations']} \\",
                f"  --calls-per-sample {methodology['calls_per_sample']} \\",
                f"  --refresh-from {artifact_name} \\",
                f"  --json-output {artifact_name} --markdown-output REPORT.md",
            )
        )
    else:
        reproduction_intro = "Reproduce the complete cross-framework corpus:"
        reproduction.extend(
            (
                f"  --engines {' '.join(engines)} --threads 1 \\",
                f"  --warmup {methodology['warmup']} \\",
                f"  --iterations {methodology['iterations']} \\",
                f"  --calls-per-sample {methodology['calls_per_sample']} \\",
                f"  --json-output {artifact_name} --markdown-output REPORT.md",
            )
        )
    lines.extend(
        (
            "",
            "Ratios above one mean FlagQuantum was faster for that measured case.",
            "These results are local comparison evidence, not a universal framework",
            "ranking or release/scalability evidence.",
            "",
            "## Reproduction",
            "",
            reproduction_intro,
            "",
            "```bash",
            *reproduction,
            "```",
            "",
        )
    )
    return "\n".join(lines)


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
    parser.add_argument("--refresh-from", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()
    if args.refresh_from is not None and tuple(args.engines) != ("flagquantum_native",):
        parser.error("--refresh-from requires --engines flagquantum_native")
    measured = run_benchmark(
        workloads=tuple(args.workloads),
        n_wires=tuple(args.n_wires),
        engines=tuple(args.engines),
        threads=args.threads,
        warmup=args.warmup,
        iterations=args.iterations,
        calls_per_sample=args.calls_per_sample,
        seed=args.seed,
    )
    payload = measured
    if args.refresh_from is not None:
        baseline = json.loads(args.refresh_from.read_text(encoding="utf-8"))
        payload = _refresh_benchmark(baseline, measured)
    if args.json_output is not None:
        write_json_atomic(args.json_output, payload)
    if args.markdown_output is not None:
        artifact_name = (
            args.json_output.name
            if args.json_output is not None
            else (
                args.refresh_from.name
                if args.refresh_from is not None
                else "simulator-workload-corpus.json"
            )
        )
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(
            _render_markdown(payload, artifact_name=artifact_name), encoding="utf-8"
        )
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
