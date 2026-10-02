#!/usr/bin/env python3
"""Measure parameter-batched exact statevector throughput across CPU bridges."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import statistics
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from importlib import import_module, metadata
from pathlib import Path
from typing import Any, Literal, cast

import torch

import flagquantum as fq

from .contract import runtime_metadata, write_json_atomic
from .simulator_compare import ABSOLUTE_TOLERANCE, SEED
from .simulator_workload_corpus import (
    WORKLOAD_NAMES,
    WorkloadName,
    _configure_threads,
    build_workload,
)

SCHEMA = "flagquantum.batched_statevector_corpus.v1"
RUNNER = "batched_statevector_corpus"
STABILITY_THRESHOLD = 0.20

EngineName = Literal[
    "flagquantum_native_batch",
    "flagquantum_native_fixed_layer_rollback",
    "flagquantum_native_static_clifford_layer_rollback",
    "flagquantum_native_parameterized_layer_rollback",
    "flagquantum_native_fused_rotation_layer_rollback",
    "flagquantum_native_clifford_matching_rollback",
    "flagquantum_native_clifford_phase_map_rollback",
    "flagquantum_native_dense_width_rollback",
    "flagquantum_native_layout_retention",
    "flagquantum_native_assembly_rollback",
    "flagquantum_native_functional_windows",
    "flagquantum_native_monolithic_batch",
    "flagquantum_native_serial",
    "qiskit_aer_bridge",
    "cirq_simulator_bridge",
    "pennylane_lightning_bridge",
    "pennylane_lightning_native_batch",
]

ENGINE_NAMES: tuple[EngineName, ...] = (
    "flagquantum_native_batch",
    "flagquantum_native_fixed_layer_rollback",
    "flagquantum_native_static_clifford_layer_rollback",
    "flagquantum_native_parameterized_layer_rollback",
    "flagquantum_native_fused_rotation_layer_rollback",
    "flagquantum_native_clifford_matching_rollback",
    "flagquantum_native_clifford_phase_map_rollback",
    "flagquantum_native_dense_width_rollback",
    "flagquantum_native_layout_retention",
    "flagquantum_native_assembly_rollback",
    "flagquantum_native_functional_windows",
    "flagquantum_native_monolithic_batch",
    "flagquantum_native_serial",
    "qiskit_aer_bridge",
    "cirq_simulator_bridge",
    "pennylane_lightning_bridge",
    "pennylane_lightning_native_batch",
)

_ENGINE_LABELS: dict[EngineName, str] = {
    "flagquantum_native_batch": "FlagQuantum native batch (budgeted)",
    "flagquantum_native_fixed_layer_rollback": (
        "FlagQuantum native batch (fixed-layer rollback)"
    ),
    "flagquantum_native_static_clifford_layer_rollback": (
        "FlagQuantum native batch (static-Clifford-layer rollback)"
    ),
    "flagquantum_native_parameterized_layer_rollback": (
        "FlagQuantum native batch (parameterized-layer rollback)"
    ),
    "flagquantum_native_fused_rotation_layer_rollback": (
        "FlagQuantum native batch (fused-rotation-layer rollback)"
    ),
    "flagquantum_native_clifford_matching_rollback": (
        "FlagQuantum native batch (Clifford-matching rollback)"
    ),
    "flagquantum_native_clifford_phase_map_rollback": (
        "FlagQuantum native batch (Clifford phase-map rollback)"
    ),
    "flagquantum_native_dense_width_rollback": (
        "FlagQuantum native batch (4-wire dense rollback)"
    ),
    "flagquantum_native_layout_retention": (
        "FlagQuantum native batch (legacy layout retention)"
    ),
    "flagquantum_native_assembly_rollback": (
        "FlagQuantum native batch (functional assembly rollback)"
    ),
    "flagquantum_native_functional_windows": (
        "FlagQuantum native batch (legacy functional windows)"
    ),
    "flagquantum_native_monolithic_batch": "FlagQuantum native monolithic batch",
    "flagquantum_native_serial": "FlagQuantum serial",
    "qiskit_aer_bridge": "Qiskit Aer bridge",
    "cirq_simulator_bridge": "Cirq bridge",
    "pennylane_lightning_bridge": "PennyLane Lightning bridge",
    "pennylane_lightning_native_batch": "PennyLane Lightning native batch",
}


@contextmanager
def _temporary_environment(**values: str) -> Iterator[None]:
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _timing(samples: Sequence[float]) -> dict[str, Any]:
    values = tuple(float(value) for value in samples)
    median = statistics.median(values)
    mad = statistics.median(abs(value - median) for value in values)
    return {
        "samples_seconds": values,
        "sample_count": len(values),
        "median_seconds": median,
        "mean_seconds": statistics.fmean(values),
        "min_seconds": min(values),
        "max_seconds": max(values),
        "relative_median_absolute_deviation": mad / median if median else 0.0,
    }


def _parameter_matrix(batch_size: int, n_wires: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed + 97 * n_wires)
    return (
        torch.rand((batch_size, n_wires), generator=generator, dtype=torch.float64)
        * 0.8
        - 0.4
    )


def build_parameter_batch(
    workload: WorkloadName,
    *,
    n_wires: int,
    batch_size: int,
    seed: int = SEED,
) -> tuple[fq.Circuit, tuple[fq.Circuit, ...]]:
    """Build one native batch and its independently parameterized scalar rows."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    base = build_workload(workload, n_wires=n_wires, seed=seed)
    parameters = _parameter_matrix(batch_size, n_wires, seed)
    batched = fq.Circuit.from_ir(base.to_ir(), bsz=batch_size)
    for wire in range(n_wires):
        batched.ry(wire, parameters[:, wire])
    scalar: list[fq.Circuit] = []
    for row in range(batch_size):
        circuit = fq.Circuit.from_ir(base.to_ir(), bsz=1)
        for wire in range(n_wires):
            circuit.ry(wire, parameters[row, wire])
        scalar.append(circuit)
    return batched, tuple(scalar)


def _stack_results(results: Sequence[Any], engine: str) -> torch.Tensor:
    states: list[torch.Tensor] = []
    for result in results:
        state = result if isinstance(result, torch.Tensor) else result.state
        if state is None:
            raise RuntimeError(f"{engine} returned no statevector")
        states.append(torch.as_tensor(state).reshape(1, -1))
    return torch.cat(states, dim=0)


def _pennylane_native_batch_callable(
    batched: fq.Circuit,
    scalar: Sequence[fq.Circuit],
) -> Callable[[], torch.Tensor]:
    """Prepare Lightning's public broadcast expansion outside warm timing."""

    qml = import_module("pennylane")
    numpy = import_module("numpy")
    conversion = import_module("flagquantum.ecosystem.pennylane.conversion")
    n_wires = batched.n_wires
    scalar_operations = tuple(
        conversion.export_pennylane(scalar[0].to_ir()).quantum_script.operations
    )
    if len(scalar_operations) < n_wires:
        raise RuntimeError("batched corpus lacks the parameterized terminal layer")
    operations = list(scalar_operations[:-n_wires])
    terminal = batched.to_ir().instructions[-n_wires:]
    for wire, instruction in enumerate(terminal):
        if instruction.name != "ry" or instruction.wires != (wire,):
            raise RuntimeError("batched corpus terminal layer is not canonical RY")
        angle = torch.as_tensor(instruction.params["theta"]).detach().cpu().numpy()
        operations.append(qml.RY(angle, wires=wire))
    script = qml.tape.QuantumScript(
        operations,
        measurements=(qml.state(),),
        shots=None,
    )
    device = qml.device(
        "lightning.qubit",
        wires=range(n_wires),
        shots=None,
        c_dtype=numpy.complex128,
    )
    program, execution_config = device.preprocess()
    circuits, postprocess = program((script,))

    def execute() -> torch.Tensor:
        native = postprocess(device.execute(circuits, execution_config))[0]
        return torch.from_numpy(numpy.asarray(native).copy()).to(torch.complex128)

    return execute


def _engine_callable(
    engine: EngineName,
    batched: fq.Circuit,
    scalar: Sequence[fq.Circuit],
    *,
    seed: int,
    threads: int,
) -> Callable[[], torch.Tensor]:
    if engine == "flagquantum_native_batch":

        def native_batch() -> torch.Tensor:
            with _temporary_environment(
                FQ_CPU_STATEVECTOR_BATCH_CHUNKING="1",
                FQ_CPU_STATEVECTOR_BATCH_BOUNDED_INITIAL_STATE="1",
                FQ_CPU_SINGLE_QUBIT_PREALLOCATE_OUTPUT="1",
                FQ_CPU_RELEASE_MATRIX_LAYOUT_INPUT="1",
            ):
                return cast(torch.Tensor, batched.state(refresh=True))

        return native_batch
    if engine == "flagquantum_native_fixed_layer_rollback":

        def native_fixed_layer_rollback() -> torch.Tensor:
            with _temporary_environment(
                FQ_CPU_STATEVECTOR_BATCH_CHUNKING="1",
                FQ_CPU_STATEVECTOR_BATCH_BOUNDED_INITIAL_STATE="1",
                FQ_CPU_SINGLE_QUBIT_PREALLOCATE_OUTPUT="1",
                FQ_CPU_RELEASE_MATRIX_LAYOUT_INPUT="1",
                FQ_CPU_NATIVE_FIXED_ONE_QUBIT_LAYER="0",
            ):
                return cast(torch.Tensor, batched.state(refresh=True))

        return native_fixed_layer_rollback
    if engine == "flagquantum_native_static_clifford_layer_rollback":

        def native_static_clifford_layer_rollback() -> torch.Tensor:
            with _temporary_environment(
                FQ_CPU_STATEVECTOR_BATCH_CHUNKING="1",
                FQ_CPU_STATEVECTOR_BATCH_BOUNDED_INITIAL_STATE="1",
                FQ_CPU_SINGLE_QUBIT_PREALLOCATE_OUTPUT="1",
                FQ_CPU_RELEASE_MATRIX_LAYOUT_INPUT="1",
                FQ_CPU_NATIVE_STATIC_CLIFFORD_LAYER="0",
            ):
                return cast(torch.Tensor, batched.state(refresh=True))

        return native_static_clifford_layer_rollback
    if engine == "flagquantum_native_parameterized_layer_rollback":

        def native_parameterized_layer_rollback() -> torch.Tensor:
            with _temporary_environment(
                FQ_CPU_STATEVECTOR_BATCH_CHUNKING="1",
                FQ_CPU_STATEVECTOR_BATCH_BOUNDED_INITIAL_STATE="1",
                FQ_CPU_SINGLE_QUBIT_PREALLOCATE_OUTPUT="1",
                FQ_CPU_RELEASE_MATRIX_LAYOUT_INPUT="1",
                FQ_CPU_NATIVE_PARAMETERIZED_ONE_QUBIT_LAYER="0",
            ):
                return cast(torch.Tensor, batched.state(refresh=True))

        return native_parameterized_layer_rollback
    if engine == "flagquantum_native_fused_rotation_layer_rollback":

        def native_fused_rotation_layer_rollback() -> torch.Tensor:
            with _temporary_environment(
                FQ_CPU_STATEVECTOR_BATCH_CHUNKING="1",
                FQ_CPU_STATEVECTOR_BATCH_BOUNDED_INITIAL_STATE="1",
                FQ_CPU_SINGLE_QUBIT_PREALLOCATE_OUTPUT="1",
                FQ_CPU_RELEASE_MATRIX_LAYOUT_INPUT="1",
                FQ_CPU_NATIVE_FUSED_ROTATION_LAYER="0",
            ):
                return cast(torch.Tensor, batched.state(refresh=True))

        return native_fused_rotation_layer_rollback
    if engine == "flagquantum_native_clifford_matching_rollback":

        def native_clifford_matching_rollback() -> torch.Tensor:
            with _temporary_environment(
                FQ_CPU_STATEVECTOR_BATCH_CHUNKING="1",
                FQ_CPU_STATEVECTOR_BATCH_BOUNDED_INITIAL_STATE="1",
                FQ_CPU_SINGLE_QUBIT_PREALLOCATE_OUTPUT="1",
                FQ_CPU_RELEASE_MATRIX_LAYOUT_INPUT="1",
                FQ_CPU_NATIVE_CLIFFORD_MATCHING="0",
            ):
                return cast(torch.Tensor, batched.state(refresh=True))

        return native_clifford_matching_rollback
    if engine == "flagquantum_native_clifford_phase_map_rollback":

        def native_clifford_phase_map_rollback() -> torch.Tensor:
            with _temporary_environment(
                FQ_CPU_STATEVECTOR_BATCH_CHUNKING="1",
                FQ_CPU_STATEVECTOR_BATCH_BOUNDED_INITIAL_STATE="1",
                FQ_CPU_SINGLE_QUBIT_PREALLOCATE_OUTPUT="1",
                FQ_CPU_RELEASE_MATRIX_LAYOUT_INPUT="1",
                FQ_CPU_NATIVE_CLIFFORD_PHASE_MAP="0",
            ):
                return cast(torch.Tensor, batched.state(refresh=True))

        return native_clifford_phase_map_rollback
    if engine == "flagquantum_native_dense_width_rollback":

        def native_dense_width_rollback() -> torch.Tensor:
            with _temporary_environment(
                FQ_CPU_STATEVECTOR_BATCH_CHUNKING="1",
                FQ_CPU_STATEVECTOR_BATCH_BOUNDED_INITIAL_STATE="1",
                FQ_CPU_SINGLE_QUBIT_PREALLOCATE_OUTPUT="1",
                FQ_CPU_RELEASE_MATRIX_LAYOUT_INPUT="1",
                FQ_CPU_ADAPTIVE_DENSE_FUSION_WIDTH="0",
            ):
                return cast(torch.Tensor, batched.state(refresh=True))

        return native_dense_width_rollback
    if engine == "flagquantum_native_layout_retention":

        def native_layout_retention() -> torch.Tensor:
            with _temporary_environment(
                FQ_CPU_STATEVECTOR_BATCH_CHUNKING="1",
                FQ_CPU_STATEVECTOR_BATCH_BOUNDED_INITIAL_STATE="1",
                FQ_CPU_SINGLE_QUBIT_PREALLOCATE_OUTPUT="1",
                FQ_CPU_RELEASE_MATRIX_LAYOUT_INPUT="0",
            ):
                return cast(torch.Tensor, batched.state(refresh=True))

        return native_layout_retention
    if engine == "flagquantum_native_assembly_rollback":

        def native_assembly_rollback() -> torch.Tensor:
            with _temporary_environment(
                FQ_CPU_STATEVECTOR_BATCH_CHUNKING="1",
                FQ_CPU_STATEVECTOR_BATCH_BOUNDED_INITIAL_STATE="1",
                FQ_CPU_STATEVECTOR_BATCH_PREALLOCATED_ASSEMBLY="0",
                FQ_CPU_SINGLE_QUBIT_PREALLOCATE_OUTPUT="1",
                FQ_CPU_RELEASE_MATRIX_LAYOUT_INPUT="1",
            ):
                return cast(torch.Tensor, batched.state(refresh=True))

        return native_assembly_rollback
    if engine == "flagquantum_native_functional_windows":

        def native_functional_windows() -> torch.Tensor:
            with _temporary_environment(
                FQ_CPU_STATEVECTOR_BATCH_CHUNKING="1",
                FQ_CPU_STATEVECTOR_BATCH_BOUNDED_INITIAL_STATE="0",
                FQ_CPU_SINGLE_QUBIT_PREALLOCATE_OUTPUT="0",
            ):
                return cast(torch.Tensor, batched.state(refresh=True))

        return native_functional_windows
    if engine == "flagquantum_native_monolithic_batch":

        def native_monolithic_batch() -> torch.Tensor:
            with _temporary_environment(FQ_CPU_STATEVECTOR_BATCH_CHUNKING="0"):
                return cast(torch.Tensor, batched.state(refresh=True))

        return native_monolithic_batch
    if engine == "flagquantum_native_serial":

        def native_serial() -> torch.Tensor:
            return _stack_results(
                tuple(circuit.state(refresh=True) for circuit in scalar), engine
            )

        return native_serial
    if engine == "pennylane_lightning_native_batch":
        return _pennylane_native_batch_callable(batched, scalar)
    if engine == "qiskit_aer_bridge":
        qiskit = import_module("flagquantum.ecosystem.qiskit")
        extensions = import_module("flagquantum.ecosystem.extensions")
        backend = qiskit.QiskitAerBackend()
        backend.start(extensions.ExtensionConfig({"seed": seed, "threads": threads}))

        def qiskit_bridge() -> torch.Tensor:
            return _stack_results(
                tuple(backend.execute(circuit) for circuit in scalar), engine
            )

        return qiskit_bridge

    module_name = {
        "cirq_simulator_bridge": "flagquantum.ecosystem.cirq",
        "pennylane_lightning_bridge": "flagquantum.ecosystem.pennylane",
    }[engine]
    run = import_module(module_name).run
    options = fq.ExecutionOptions(seed=seed)

    def repeated_bridge() -> torch.Tensor:
        return _stack_results(
            tuple(run(circuit, options=options) for circuit in scalar), engine
        )

    return repeated_bridge


def _engine_versions(engine: EngineName) -> dict[str, str]:
    packages = {
        "flagquantum_native_batch": ("flagquantum",),
        "flagquantum_native_fixed_layer_rollback": ("flagquantum",),
        "flagquantum_native_static_clifford_layer_rollback": ("flagquantum",),
        "flagquantum_native_parameterized_layer_rollback": ("flagquantum",),
        "flagquantum_native_fused_rotation_layer_rollback": ("flagquantum",),
        "flagquantum_native_clifford_matching_rollback": ("flagquantum",),
        "flagquantum_native_clifford_phase_map_rollback": ("flagquantum",),
        "flagquantum_native_dense_width_rollback": ("flagquantum",),
        "flagquantum_native_layout_retention": ("flagquantum",),
        "flagquantum_native_assembly_rollback": ("flagquantum",),
        "flagquantum_native_functional_windows": ("flagquantum",),
        "flagquantum_native_monolithic_batch": ("flagquantum",),
        "flagquantum_native_serial": ("flagquantum",),
        "qiskit_aer_bridge": ("qiskit", "qiskit-aer"),
        "cirq_simulator_bridge": ("cirq-core",),
        "pennylane_lightning_bridge": ("pennylane", "pennylane-lightning"),
        "pennylane_lightning_native_batch": (
            "pennylane",
            "pennylane-lightning",
        ),
    }[engine]
    return {package: metadata.version(package) for package in packages}


def _execution_strategy(engine: EngineName) -> str:
    if engine == "flagquantum_native_fixed_layer_rollback":
        return "native_parameter_batch_fixed_layer_rollback"
    if engine == "flagquantum_native_static_clifford_layer_rollback":
        return "native_parameter_batch_static_clifford_layer_rollback"
    if engine == "flagquantum_native_parameterized_layer_rollback":
        return "native_parameter_batch_parameterized_layer_rollback"
    if engine == "flagquantum_native_fused_rotation_layer_rollback":
        return "native_parameter_batch_fused_rotation_layer_rollback"
    if engine == "flagquantum_native_clifford_matching_rollback":
        return "native_parameter_batch_clifford_matching_rollback"
    if engine == "flagquantum_native_clifford_phase_map_rollback":
        return "native_parameter_batch_clifford_phase_map_rollback"
    if engine == "flagquantum_native_dense_width_rollback":
        return "native_parameter_batch_four_wire_dense_rollback"
    if engine == "pennylane_lightning_native_batch":
        return "framework_native_broadcast_batch"
    return (
        "native_parameter_batch"
        if engine == "flagquantum_native_batch"
        else (
            "native_parameter_batch_legacy_layout_retention"
            if engine == "flagquantum_native_layout_retention"
            else (
                "native_parameter_batch_functional_assembly_rollback"
                if engine == "flagquantum_native_assembly_rollback"
                else (
                    "native_parameter_batch_functional_windows"
                    if engine == "flagquantum_native_functional_windows"
                    else (
                        "native_monolithic_parameter_batch"
                        if engine == "flagquantum_native_monolithic_batch"
                        else (
                            "repeated_single_item_native"
                            if engine == "flagquantum_native_serial"
                            else "repeated_single_item_bridge"
                        )
                    )
                )
            )
        )
    )


def run_case(
    *,
    workload: WorkloadName,
    n_wires: int,
    batch_size: int,
    engines: Sequence[EngineName],
    threads: int,
    warmup: int,
    iterations: int,
    seed: int = SEED,
) -> dict[str, Any]:
    """Measure one independent-parameter batch through the requested engines."""
    if warmup < 0 or iterations < 3:
        raise ValueError("warmup must be non-negative and iterations at least 3")
    if not engines or len(set(engines)) != len(engines):
        raise ValueError("engines must contain unique supported engine names")
    unknown = sorted(set(engines) - set(ENGINE_NAMES))
    if unknown:
        raise ValueError("unsupported engine(s): " + ", ".join(unknown))
    batched, scalar = build_parameter_batch(
        workload, n_wires=n_wires, batch_size=batch_size, seed=seed
    )
    functions = {
        engine: _engine_callable(engine, batched, scalar, seed=seed, threads=threads)
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
            outputs[engine] = functions[engine]()
            samples[engine].append(time.perf_counter() - started)

    reference_name = (
        "flagquantum_native_batch"
        if "flagquantum_native_batch" in engines
        else engines[0]
    )
    reference = outputs[reference_name].detach().cpu().to(torch.complex128)
    engine_payload: dict[str, Any] = {}
    correctness: dict[str, Any] = {}
    stability: dict[str, bool] = {}
    for engine in engines:
        output = outputs[engine].detach().cpu().to(torch.complex128)
        max_error = float(torch.max(torch.abs(output - reference)).item())
        timing = _timing(samples[engine])
        median = float(timing["median_seconds"])
        correctness[engine] = {
            "passed": max_error <= ABSOLUTE_TOLERANCE,
            "max_abs_error": max_error,
        }
        stability[engine] = (
            timing["relative_median_absolute_deviation"] <= STABILITY_THRESHOLD
        )
        engine_payload[engine] = {
            "versions": _engine_versions(engine),
            "execution_strategy": _execution_strategy(engine),
            "batch_total": timing,
            "median_seconds_per_statevector": median / batch_size,
            "median_statevectors_per_second": batch_size / median,
        }

    native_seconds = (
        engine_payload["flagquantum_native_batch"]["batch_total"]["median_seconds"]
        if "flagquantum_native_batch" in engine_payload
        else None
    )
    ratios = (
        {
            engine: payload["batch_total"]["median_seconds"] / native_seconds
            for engine, payload in engine_payload.items()
            if engine != "flagquantum_native_batch"
        }
        if native_seconds is not None
        else {}
    )
    template_ir = scalar[0].to_ir()
    logical_bytes = (
        batch_size
        * (2**n_wires)
        * torch.empty((), dtype=torch.complex128).element_size()
    )
    return {
        "workload": {
            "name": workload,
            "n_wires": n_wires,
            "batch_size": batch_size,
            "dtype": "complex128",
            "independent_parameter_count": batch_size * n_wires,
            "gate_count_per_item": len(template_ir.instructions),
            "scalar_ir_content_hash": template_ir.content_hash,
            "logical_statevector_bytes": logical_bytes,
        },
        "correctness": {
            "passed": all(item["passed"] for item in correctness.values()),
            "reference_engine": reference_name,
            "absolute_tolerance": ABSOLUTE_TOLERANCE,
            "engines": correctness,
        },
        "engines": engine_payload,
        "comparison": {
            "engine_over_flagquantum_batch_median": ratios,
            "ratio_semantics": "values above one mean FlagQuantum batch is faster",
        },
        "stability": {
            "maximum_relative_median_absolute_deviation": STABILITY_THRESHOLD,
            "passed": all(stability.values()),
            "engines": stability,
        },
    }


def run_benchmark(
    *,
    workloads: Sequence[WorkloadName],
    n_wires: Sequence[int],
    batch_sizes: Sequence[int],
    engines: Sequence[EngineName],
    threads: int,
    warmup: int,
    iterations: int,
    seed: int = SEED,
) -> dict[str, Any]:
    """Measure the requested workload, width, and batch-size cross-product."""
    if threads < 1:
        raise ValueError("threads must be positive")
    if not workloads or not n_wires or not batch_sizes:
        raise ValueError("workloads, n_wires, and batch_sizes must not be empty")
    if any(batch_size < 1 for batch_size in batch_sizes):
        raise ValueError("batch_sizes must be positive")
    if len(set(workloads)) != len(workloads):
        raise ValueError("workloads must be unique")
    thread_environment = _configure_threads(threads)
    cases = tuple(
        run_case(
            workload=workload,
            n_wires=width,
            batch_size=batch_size,
            engines=engines,
            threads=threads,
            warmup=warmup,
            iterations=iterations,
            seed=seed,
        )
        for workload in workloads
        for width in n_wires
        for batch_size in batch_sizes
    )
    passed = all(case["correctness"]["passed"] for case in cases)
    stable = all(case["stability"]["passed"] for case in cases)
    return runtime_metadata(
        runner=RUNNER,
        schema=SCHEMA,
        benchmark="cross_framework_batched_statevector_throughput",
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
        all_measurements_stable=stable,
        platform=platform.platform(),
        environment={
            "device": "cpu",
            "machine": platform.machine(),
            "processor": platform.processor() or "unknown",
            "torch": torch.__version__,
            "torch_threads": threads,
            "thread_environment": thread_environment,
        },
        methodology={
            "warmup": warmup,
            "iterations": iterations,
            "measurement_scope": "complete_N_statevector_user_task",
            "circuit_construction_included": False,
            "conversion_and_backend_preparation_included": True,
            "result_retrieval_included": True,
            "exact_statevector": True,
            "independent_parameter_bindings": True,
            "external_bridge_batching": "repeated_single_item_bridge",
            "pennylane_native_batching": "broadcast_expand_preprocessed_once",
            "engine_order": "rotated_per_iteration_within_one_process",
        },
        workloads=tuple(workloads),
        n_wires=tuple(n_wires),
        batch_sizes=tuple(batch_sizes),
        engines=tuple(engines),
        cases=cases,
    )


def render_markdown(payload: dict[str, Any], *, artifact_name: str) -> str:
    lines = [
        "# Cross-framework CPU batched statevector throughput",
        "",
        f"Generated from [`{artifact_name}`]({artifact_name}). Each task returns N",
        "exact statevectors for N independent parameter bindings. FlagQuantum uses",
    ]
    if "pennylane_lightning_native_batch" in payload["engines"]:
        lines.extend(
            (
                "its native parameter batch. Bridge engines accept one item and are",
                "invoked repeatedly; the explicitly named PennyLane native-batch engine",
                "uses Lightning's public broadcast expansion after one-time preprocessing.",
            )
        )
    else:
        lines.extend(
            (
                "its native parameter batch; current external bridges accept one item and",
                "are therefore invoked repeatedly. This measures the FlagQuantum user-facing",
                "interop surface, not each external framework's best raw batching API.",
            )
        )
    lines.extend(
        (
            "",
            "| Workload | Qubits | Batch | Engine | Total (ms) | Per state (ms) | States/s | vs FQ batch |",
            "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
        )
    )
    for case in payload["cases"]:
        ratios = case["comparison"]["engine_over_flagquantum_batch_median"]
        for engine in payload["engines"]:
            result = case["engines"][engine]
            total = result["batch_total"]["median_seconds"]
            ratio = 1.0 if engine == "flagquantum_native_batch" else ratios[engine]
            lines.append(
                "| {workload} | {n_wires} | {batch_size} | {engine} | "
                "{total:.3f} | {per_state:.3f} | {throughput:.2f} | {ratio:.2f}x |".format(
                    workload=case["workload"]["name"],
                    n_wires=case["workload"]["n_wires"],
                    batch_size=case["workload"]["batch_size"],
                    engine=_ENGINE_LABELS[cast(EngineName, engine)],
                    total=total * 1000,
                    per_state=result["median_seconds_per_statevector"] * 1000,
                    throughput=result["median_statevectors_per_second"],
                    ratio=ratio,
                )
            )
    lines.extend(
        (
            "",
            "Ratios above one mean FlagQuantum native batch completed the identical",
            "N-statevector task faster. Results are local comparison evidence, not a",
            "universal framework ranking or release/scalability evidence.",
            "",
        )
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workloads", nargs="+", choices=WORKLOAD_NAMES, default=WORKLOAD_NAMES[:5]
    )
    parser.add_argument("--n-wires", type=int, nargs="+", default=(10, 14, 18))
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=(1, 8, 32))
    parser.add_argument(
        "--engines", nargs="+", choices=ENGINE_NAMES, default=ENGINE_NAMES
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()
    payload = run_benchmark(
        workloads=tuple(args.workloads),
        n_wires=tuple(args.n_wires),
        batch_sizes=tuple(args.batch_sizes),
        engines=tuple(args.engines),
        threads=args.threads,
        warmup=args.warmup,
        iterations=args.iterations,
        seed=args.seed,
    )
    if args.json_output is not None:
        write_json_atomic(args.json_output, payload)
    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(
            render_markdown(
                payload,
                artifact_name=(
                    args.json_output.name
                    if args.json_output is not None
                    else "batched-statevector-corpus.json"
                ),
            ),
            encoding="utf-8",
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = (
    "ENGINE_NAMES",
    "RUNNER",
    "SCHEMA",
    "build_parameter_batch",
    "main",
    "render_markdown",
    "run_benchmark",
    "run_case",
)
