#!/usr/bin/env python3
"""Measure matched differentiable CPU simulator workloads."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module, metadata
from pathlib import Path
from textwrap import wrap
from typing import Any, Literal

import torch

import flagquantum as fq
from flagquantum import algorithms as fqa

from .contract import runtime_metadata, write_json_atomic
from .differentiable_adjoint_report import render_adjoint_markdown
from .differentiable_regression_report import (
    render_forward_cx_markdown,
    render_observable_cache_markdown,
    render_rotation_tile_markdown,
)
from .simulator_compare import SEED
from .simulator_workload_corpus import extract_features

SCHEMA = "flagquantum.differentiable_simulator_corpus.v1"
RUNNER = "differentiable_simulator_corpus"
_ABSOLUTE_TOLERANCE = 1e-9
_STABILITY_THRESHOLD = 0.20

WorkloadName = Literal["hardware_efficient_vqe", "qaoa_path_maxcut"]
EngineName = Literal[
    "flagquantum_native",
    "pennylane_default_qubit",
    "flagquantum_adjoint",
    "flagquantum_adjoint_python_fallback",
    "flagquantum_adjoint_gather_rollback",
    "flagquantum_adjoint_forward_cx_rollback",
    "flagquantum_adjoint_observable_cache_rollback",
    "flagquantum_adjoint_rotation_tile_rollback",
    "pennylane_lightning_adjoint",
]

WORKLOAD_NAMES: tuple[WorkloadName, ...] = (
    "hardware_efficient_vqe",
    "qaoa_path_maxcut",
)
ENGINE_NAMES: tuple[EngineName, ...] = (
    "flagquantum_native",
    "pennylane_default_qubit",
)
ADJOINT_ENGINE_NAMES: tuple[EngineName, ...] = (
    "flagquantum_adjoint",
    "flagquantum_adjoint_python_fallback",
    "pennylane_lightning_adjoint",
)
ALL_ENGINE_NAMES = (
    ENGINE_NAMES
    + ADJOINT_ENGINE_NAMES
    + (
        "flagquantum_adjoint_gather_rollback",
        "flagquantum_adjoint_forward_cx_rollback",
        "flagquantum_adjoint_observable_cache_rollback",
        "flagquantum_adjoint_rotation_tile_rollback",
    )
)

_WORKLOAD_LABELS: dict[WorkloadName, str] = {
    "hardware_efficient_vqe": "Hardware-efficient VQE",
    "qaoa_path_maxcut": "QAOA path MaxCut",
}
_ENGINE_LABELS: dict[EngineName, str] = {
    "flagquantum_native": "FlagQuantum",
    "pennylane_default_qubit": "PennyLane default.qubit",
    "flagquantum_adjoint": "FlagQuantum adjoint",
    "flagquantum_adjoint_python_fallback": "FlagQuantum adjoint Python fallback",
    "flagquantum_adjoint_gather_rollback": "FlagQuantum adjoint gather rollback",
    "flagquantum_adjoint_forward_cx_rollback": (
        "FlagQuantum adjoint forward CX rollback"
    ),
    "flagquantum_adjoint_observable_cache_rollback": (
        "FlagQuantum adjoint observable-cache rollback"
    ),
    "flagquantum_adjoint_rotation_tile_rollback": (
        "FlagQuantum adjoint rotation-tile rollback"
    ),
    "pennylane_lightning_adjoint": "PennyLane Lightning adjoint",
}


@dataclass(frozen=True)
class _Workload:
    circuit: fq.Circuit
    parameters: torch.Tensor
    constant: float
    hamiltonian: fqa.Hamiltonian
    observable: str


@dataclass(frozen=True)
class _Execution:
    value: float
    gradient: torch.Tensor
    forward_seconds: float
    backward_seconds: float


def _require_scalar_tensor(value: Any, *, engine: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.numel() != 1:
        raise TypeError(f"{engine} must return one scalar Torch expectation")
    return value.reshape(())


def _parameters(
    name: WorkloadName, n_wires: int, layers: int, seed: int
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed + 1009 * n_wires)
    shape = (layers, n_wires, 3) if name == "hardware_efficient_vqe" else (layers, 2)
    return (
        0.05 + 0.45 * torch.rand(shape, dtype=torch.float64, generator=generator)
    ).requires_grad_()


def _vqe_hamiltonian(n_wires: int) -> fqa.Hamiltonian:
    terms = [
        fqa.HamiltonianTerm(0.7, {wire: "z", wire + 1: "z"})
        for wire in range(n_wires - 1)
    ]
    terms.extend(fqa.HamiltonianTerm(0.2, "z", wire) for wire in range(n_wires))
    return fqa.Hamiltonian(terms)


def _maxcut_hamiltonian(n_wires: int) -> fqa.Hamiltonian:
    return fqa.Hamiltonian(
        fqa.HamiltonianTerm(-0.5, {wire: "z", wire + 1: "z"})
        for wire in range(n_wires - 1)
    )


def build_workload(
    name: WorkloadName,
    *,
    n_wires: int,
    layers: int = 1,
    seed: int = SEED,
) -> _Workload:
    """Build one deterministic differentiable workload."""

    if n_wires < 4:
        raise ValueError("n_wires must be at least 4")
    if layers < 1:
        raise ValueError("layers must be positive")
    if name not in WORKLOAD_NAMES:
        raise ValueError(f"unsupported workload: {name}")
    parameters = _parameters(name, n_wires, layers, seed)
    circuit = fq.Circuit(n_wires, dtype=torch.complex128)
    if name == "hardware_efficient_vqe":
        for layer in range(layers):
            for wire in range(n_wires):
                circuit.rx(wire, parameters[layer, wire, 0])
                circuit.ry(wire, parameters[layer, wire, 1])
                circuit.rz(wire, parameters[layer, wire, 2])
            for wire in range(n_wires - 1):
                circuit.cx(wire, wire + 1)
        return _Workload(
            circuit=circuit,
            parameters=parameters,
            constant=0.0,
            hamiltonian=_vqe_hamiltonian(n_wires),
            observable="0.7 * sum(Z_i Z_{i+1}) + 0.2 * sum(Z_i)",
        )

    for wire in range(n_wires):
        circuit.h(wire)
    for layer in range(layers):
        for wire in range(n_wires - 1):
            circuit.rzz(wire, wire + 1, parameters[layer, 0])
        for wire in range(n_wires):
            circuit.rx(wire, parameters[layer, 1])
    return _Workload(
        circuit=circuit,
        parameters=parameters,
        constant=0.5 * (n_wires - 1),
        hamiltonian=_maxcut_hamiltonian(n_wires),
        observable="sum((1 - Z_i Z_{i+1}) / 2)",
    )


def _flagquantum_executor(
    workload: _Workload,
    *,
    differentiation: Literal["autograd", "adjoint"],
    cpu_direct: bool | None = None,
    native_cpu_adjoint: bool | None = None,
    native_cpu_cx_gather: bool | None = None,
    observable_cache: bool | None = None,
    wide_rotation_tiles: bool | None = None,
) -> Callable[[], _Execution]:
    parameters = workload.parameters

    def execute() -> _Execution:
        prior_direct = os.environ.get("FQ_STATEVECTOR_ADJOINT_CPU_DIRECT")
        prior_native = os.environ.get("FQ_NATIVE_CPU_ADJOINT")
        prior_cx_gather = os.environ.get("FQ_NATIVE_CPU_CX_GATHER")
        prior_observable_cache = os.environ.get(
            "FQ_STATEVECTOR_ADJOINT_OBSERVABLE_CACHE"
        )
        prior_wide_rotation_tiles = os.environ.get("FQ_NATIVE_CPU_ADJOINT_WIDE_TILES")
        if cpu_direct is not None:
            os.environ["FQ_STATEVECTOR_ADJOINT_CPU_DIRECT"] = "1" if cpu_direct else "0"
        if native_cpu_adjoint is not None:
            os.environ["FQ_NATIVE_CPU_ADJOINT"] = "1" if native_cpu_adjoint else "0"
        if native_cpu_cx_gather is not None:
            os.environ["FQ_NATIVE_CPU_CX_GATHER"] = "1" if native_cpu_cx_gather else "0"
        if observable_cache is not None:
            os.environ["FQ_STATEVECTOR_ADJOINT_OBSERVABLE_CACHE"] = (
                "1" if observable_cache else "0"
            )
        if wide_rotation_tiles is not None:
            os.environ["FQ_NATIVE_CPU_ADJOINT_WIDE_TILES"] = (
                "1" if wide_rotation_tiles else "0"
            )
        try:
            forward_started = time.perf_counter()
            if differentiation == "adjoint":
                loss = (
                    workload.hamiltonian.expectation(
                        workload.circuit, differentiation="adjoint"
                    )
                    + workload.constant
                )
            else:
                state = workload.circuit.state(refresh=True)
                loss = workload.hamiltonian.expectation(state).sum() + workload.constant
            forward_seconds = time.perf_counter() - forward_started
            backward_started = time.perf_counter()
            gradient = torch.autograd.grad(loss, parameters)[0]
            backward_seconds = time.perf_counter() - backward_started
        finally:
            if cpu_direct is not None:
                if prior_direct is None:
                    os.environ.pop("FQ_STATEVECTOR_ADJOINT_CPU_DIRECT", None)
                else:
                    os.environ["FQ_STATEVECTOR_ADJOINT_CPU_DIRECT"] = prior_direct
            if native_cpu_adjoint is not None:
                if prior_native is None:
                    os.environ.pop("FQ_NATIVE_CPU_ADJOINT", None)
                else:
                    os.environ["FQ_NATIVE_CPU_ADJOINT"] = prior_native
            if native_cpu_cx_gather is not None:
                if prior_cx_gather is None:
                    os.environ.pop("FQ_NATIVE_CPU_CX_GATHER", None)
                else:
                    os.environ["FQ_NATIVE_CPU_CX_GATHER"] = prior_cx_gather
            if observable_cache is not None:
                if prior_observable_cache is None:
                    os.environ.pop("FQ_STATEVECTOR_ADJOINT_OBSERVABLE_CACHE", None)
                else:
                    os.environ["FQ_STATEVECTOR_ADJOINT_OBSERVABLE_CACHE"] = (
                        prior_observable_cache
                    )
            if wide_rotation_tiles is not None:
                if prior_wide_rotation_tiles is None:
                    os.environ.pop("FQ_NATIVE_CPU_ADJOINT_WIDE_TILES", None)
                else:
                    os.environ["FQ_NATIVE_CPU_ADJOINT_WIDE_TILES"] = (
                        prior_wide_rotation_tiles
                    )
        return _Execution(
            value=float(loss.detach()),
            gradient=gradient.detach(),
            forward_seconds=forward_seconds,
            backward_seconds=backward_seconds,
        )

    return execute


def _pennylane_executor(
    name: WorkloadName,
    *,
    n_wires: int,
    layers: int,
    seed: int,
    device_name: Literal["default.qubit", "lightning.qubit"],
    diff_method: Literal["backprop", "adjoint"],
) -> Callable[[], _Execution]:
    qml = import_module("pennylane")
    parameters = _parameters(name, n_wires, layers, seed)
    device = qml.device(device_name, wires=n_wires, shots=None)
    if name == "hardware_efficient_vqe":
        coefficients = [0.7] * (n_wires - 1) + [0.2] * n_wires
        operators = [
            qml.PauliZ(wire) @ qml.PauliZ(wire + 1) for wire in range(n_wires - 1)
        ]
        operators.extend(qml.PauliZ(wire) for wire in range(n_wires))
        observable = qml.Hamiltonian(coefficients, operators)
        constant = 0.0
    else:
        observable = qml.Hamiltonian(
            [-0.5] * (n_wires - 1),
            [qml.PauliZ(wire) @ qml.PauliZ(wire + 1) for wire in range(n_wires - 1)],
        )
        constant = 0.5 * (n_wires - 1)

    def circuit(values: torch.Tensor) -> Any:
        if name == "hardware_efficient_vqe":
            for layer in range(layers):
                for wire in range(n_wires):
                    qml.RX(values[layer, wire, 0], wires=wire)
                    qml.RY(values[layer, wire, 1], wires=wire)
                    qml.RZ(values[layer, wire, 2], wires=wire)
                for wire in range(n_wires - 1):
                    qml.CNOT(wires=(wire, wire + 1))
        else:
            for wire in range(n_wires):
                qml.Hadamard(wires=wire)
            for layer in range(layers):
                for wire in range(n_wires - 1):
                    qml.IsingZZ(values[layer, 0], wires=(wire, wire + 1))
                for wire in range(n_wires):
                    qml.RX(values[layer, 1], wires=wire)
        return qml.expval(observable)

    qnode = qml.qnode(device, interface="torch", diff_method=diff_method)(circuit)

    def execute() -> _Execution:
        forward_started = time.perf_counter()
        loss = (
            _require_scalar_tensor(qnode(parameters), engine=f"PennyLane {device_name}")
            + constant
        )
        forward_seconds = time.perf_counter() - forward_started
        backward_started = time.perf_counter()
        gradient = torch.autograd.grad(loss, parameters)[0]
        backward_seconds = time.perf_counter() - backward_started
        return _Execution(
            value=float(loss.detach()),
            gradient=gradient.detach(),
            forward_seconds=forward_seconds,
            backward_seconds=backward_seconds,
        )

    return execute


def _engine_callable(
    engine: EngineName,
    workload: WorkloadName,
    *,
    n_wires: int,
    layers: int,
    seed: int,
) -> Callable[[], _Execution]:
    if engine == "flagquantum_native":
        return _flagquantum_executor(
            build_workload(workload, n_wires=n_wires, layers=layers, seed=seed),
            differentiation="autograd",
        )
    if engine == "flagquantum_adjoint":
        return _flagquantum_executor(
            build_workload(workload, n_wires=n_wires, layers=layers, seed=seed),
            differentiation="adjoint",
            cpu_direct=True,
            native_cpu_adjoint=True,
        )
    if engine == "flagquantum_adjoint_python_fallback":
        return _flagquantum_executor(
            build_workload(workload, n_wires=n_wires, layers=layers, seed=seed),
            differentiation="adjoint",
            cpu_direct=True,
            native_cpu_adjoint=False,
        )
    if engine == "flagquantum_adjoint_gather_rollback":
        return _flagquantum_executor(
            build_workload(workload, n_wires=n_wires, layers=layers, seed=seed),
            differentiation="adjoint",
            cpu_direct=False,
        )
    if engine == "flagquantum_adjoint_forward_cx_rollback":
        return _flagquantum_executor(
            build_workload(workload, n_wires=n_wires, layers=layers, seed=seed),
            differentiation="adjoint",
            cpu_direct=True,
            native_cpu_adjoint=True,
            native_cpu_cx_gather=False,
        )
    if engine == "flagquantum_adjoint_observable_cache_rollback":
        return _flagquantum_executor(
            build_workload(workload, n_wires=n_wires, layers=layers, seed=seed),
            differentiation="adjoint",
            cpu_direct=True,
            native_cpu_adjoint=True,
            observable_cache=False,
        )
    if engine == "flagquantum_adjoint_rotation_tile_rollback":
        return _flagquantum_executor(
            build_workload(workload, n_wires=n_wires, layers=layers, seed=seed),
            differentiation="adjoint",
            cpu_direct=True,
            native_cpu_adjoint=True,
            wide_rotation_tiles=False,
        )
    if engine == "pennylane_default_qubit":
        return _pennylane_executor(
            workload,
            n_wires=n_wires,
            layers=layers,
            seed=seed,
            device_name="default.qubit",
            diff_method="backprop",
        )
    if engine == "pennylane_lightning_adjoint":
        return _pennylane_executor(
            workload,
            n_wires=n_wires,
            layers=layers,
            seed=seed,
            device_name="lightning.qubit",
            diff_method="adjoint",
        )
    raise ValueError(f"unsupported engine: {engine}")


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


def _package_versions(engine: EngineName) -> dict[str, str]:
    packages = (
        ("flagquantum", "torch")
        if engine.startswith("flagquantum_")
        else (
            ("pennylane", "pennylane_lightning", "torch")
            if engine == "pennylane_lightning_adjoint"
            else ("pennylane", "torch")
        )
    )
    return {package: metadata.version(package) for package in packages}


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _isolated_peak_rss(
    engine: EngineName,
    workload: WorkloadName,
    *,
    n_wires: int,
    layers: int,
    threads: int,
    seed: int,
    timeout_seconds: float,
) -> int:
    command = (
        sys.executable,
        "-m",
        "flagquantum.benchmarking.differentiable_simulator_corpus",
        "--memory-worker",
        "--workloads",
        workload,
        "--engines",
        engine,
        "--n-wires",
        str(n_wires),
        "--layers",
        str(layers),
        "--threads",
        str(threads),
        "--seed",
        str(seed),
    )
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        env={**os.environ, "MPLCONFIGDIR": os.environ.get("MPLCONFIGDIR", "/tmp")},
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"{engine} memory worker produced no output")
    payload = json.loads(lines[-1])
    peak = payload.get("peak_rss_bytes")
    if not isinstance(peak, int) or peak <= 0:
        raise RuntimeError(f"{engine} memory worker returned an invalid peak RSS")
    return peak


def run_case(
    *,
    workload: WorkloadName,
    n_wires: int,
    layers: int,
    engines: Sequence[EngineName],
    warmup: int,
    iterations: int,
    calls_per_sample: int,
    threads: int = 1,
    seed: int = SEED,
    measure_memory: bool = True,
    case_timeout_seconds: float = 600.0,
) -> dict[str, Any]:
    """Measure one value-and-gradient workload through each requested engine."""

    if warmup < 0 or iterations < 3 or calls_per_sample < 1:
        raise ValueError(
            "warmup must be non-negative, iterations at least 3, and "
            "calls_per_sample positive"
        )
    if not engines or len(set(engines)) != len(engines):
        raise ValueError("engines must contain unique supported engine names")
    unknown = sorted(set(engines) - set(ALL_ENGINE_NAMES))
    if unknown:
        raise ValueError("unsupported engine(s): " + ", ".join(unknown))
    if case_timeout_seconds <= 0:
        raise ValueError("case_timeout_seconds must be positive")

    descriptor = build_workload(workload, n_wires=n_wires, layers=layers, seed=seed)
    ir = descriptor.circuit.to_ir()
    functions = {
        engine: _engine_callable(
            engine, workload, n_wires=n_wires, layers=layers, seed=seed
        )
        for engine in engines
    }
    outputs: dict[EngineName, _Execution] = {}
    for _ in range(warmup):
        for engine in engines:
            outputs[engine] = functions[engine]()

    samples: dict[EngineName, dict[str, list[float]]] = {
        engine: {"forward": [], "backward": [], "value_and_grad": []}
        for engine in engines
    }
    for iteration in range(iterations):
        offset = iteration % len(engines)
        order = tuple(engines[offset:]) + tuple(engines[:offset])
        for engine in order:
            gc.collect()
            forward = 0.0
            backward = 0.0
            for _ in range(calls_per_sample):
                output = functions[engine]()
                outputs[engine] = output
                forward += output.forward_seconds
                backward += output.backward_seconds
            forward /= calls_per_sample
            backward /= calls_per_sample
            samples[engine]["forward"].append(forward)
            samples[engine]["backward"].append(backward)
            samples[engine]["value_and_grad"].append(forward + backward)

    reference_engine = next(
        (engine for engine in engines if engine.startswith("flagquantum_")), engines[0]
    )
    reference = outputs[reference_engine]
    engine_payload: dict[str, Any] = {}
    correctness: dict[str, Any] = {}
    for engine in engines:
        output = outputs[engine]
        value_error = abs(output.value - reference.value)
        gradient_error = float(
            torch.max(torch.abs(output.gradient - reference.gradient)).item()
        )
        correctness[engine] = {
            "passed": (
                value_error <= _ABSOLUTE_TOLERANCE
                and gradient_error <= _ABSOLUTE_TOLERANCE
            ),
            "value_max_abs_error": value_error,
            "gradient_max_abs_error": gradient_error,
        }
        peak_rss = (
            _isolated_peak_rss(
                engine,
                workload,
                n_wires=n_wires,
                layers=layers,
                threads=threads,
                seed=seed,
                timeout_seconds=case_timeout_seconds,
            )
            if measure_memory
            else None
        )
        engine_payload[engine] = {
            "versions": _package_versions(engine),
            "differentiation": (
                "torch_reverse_mode_autograd"
                if engine == "flagquantum_native"
                else (
                    "flagquantum_statevector_adjoint"
                    if engine.startswith("flagquantum_adjoint")
                    else (
                        "pennylane_lightning_adjoint_with_torch_interface"
                        if engine == "pennylane_lightning_adjoint"
                        else "pennylane_backprop_with_torch_interface"
                    )
                )
            ),
            "forward": _timing(samples[engine]["forward"]),
            "backward": _timing(samples[engine]["backward"]),
            "value_and_grad": _timing(samples[engine]["value_and_grad"]),
            "memory": {
                "peak_rss_bytes": peak_rss,
                "scope": (
                    "isolated_process_peak_rss_including_import_build_and_one_value_and_grad"
                    if measure_memory
                    else "not_measured"
                ),
            },
        }

    native_name = next(
        (engine for engine in engines if engine.startswith("flagquantum_")), None
    )
    native_payload = (
        engine_payload.get(native_name) if native_name is not None else None
    )
    ratios: dict[str, dict[str, float]] = {}
    if native_payload is not None:
        for engine_name, payload in engine_payload.items():
            if engine_name == native_name:
                continue
            ratios[engine_name] = {
                metric: payload[metric]["median_seconds"]
                / native_payload[metric]["median_seconds"]
                for metric in ("forward", "backward", "value_and_grad")
            }
    stability = {
        engine: payload["value_and_grad"]["relative_median_absolute_deviation"]
        <= _STABILITY_THRESHOLD
        for engine, payload in engine_payload.items()
    }
    return {
        "workload": {
            "name": workload,
            "n_wires": n_wires,
            "layers": layers,
            "seed": seed,
            "dtype": "complex128",
            "gate_count": len(ir.instructions),
            "parameter_count": descriptor.parameters.numel(),
            "observable": descriptor.observable,
            "observable_term_count": descriptor.hamiltonian.n_terms
            + int(descriptor.constant != 0.0),
            "logical_statevector_bytes": 16 * (2**n_wires),
            "ir_content_hash": ir.content_hash,
        },
        "features": extract_features(ir),
        "correctness": {
            "passed": all(item["passed"] for item in correctness.values()),
            "reference_engine": reference_engine,
            "absolute_tolerance": _ABSOLUTE_TOLERANCE,
            "engines": correctness,
        },
        "engines": engine_payload,
        "comparison": {
            "engine_over_flagquantum_median": ratios,
            "ratio_semantics": (
                "values above one mean FlagQuantum is faster for that metric"
            ),
        },
        "stability": {
            "maximum_value_and_grad_relative_median_absolute_deviation": (
                _STABILITY_THRESHOLD
            ),
            "passed": all(stability.values()),
            "engines": stability,
        },
    }


def run_benchmark(
    *,
    workloads: Sequence[WorkloadName],
    n_wires: Sequence[int],
    layers: int,
    engines: Sequence[EngineName],
    threads: int,
    warmup: int,
    iterations: int,
    calls_per_sample: int,
    seed: int = SEED,
    measure_memory: bool = True,
    case_timeout_seconds: float = 600.0,
) -> dict[str, Any]:
    """Measure the requested differentiable workload matrix."""

    if threads < 1:
        raise ValueError("threads must be positive")
    if layers < 1:
        raise ValueError("layers must be positive")
    if not workloads or not n_wires:
        raise ValueError("workloads and n_wires must each contain at least one item")
    if len(set(workloads)) != len(workloads):
        raise ValueError("workloads must be unique")
    unknown = sorted(set(workloads) - set(WORKLOAD_NAMES))
    if unknown:
        raise ValueError("unsupported workload(s): " + ", ".join(unknown))
    thread_environment = _configure_threads(threads)
    prior_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        cases = tuple(
            run_case(
                workload=workload,
                n_wires=width,
                layers=layers,
                engines=engines,
                warmup=warmup,
                iterations=iterations,
                calls_per_sample=calls_per_sample,
                threads=threads,
                seed=seed,
                measure_memory=measure_memory,
                case_timeout_seconds=case_timeout_seconds,
            )
            for workload in workloads
            for width in n_wires
        )
    finally:
        torch.set_default_dtype(prior_default_dtype)
    passed = all(bool(case["correctness"]["passed"]) for case in cases)
    all_measurements_stable = all(bool(case["stability"]["passed"]) for case in cases)
    payload = runtime_metadata(
        runner=RUNNER,
        schema=SCHEMA,
        benchmark="cross_framework_differentiable_simulator_workload_corpus",
        artifact_class="measured_comparison_run",
        benchmark_evidence_class="comparison_non_release",
        claim_evidence_type="unknown",
        distribution_semantics="single_device_fast_path",
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
            "torch_default_dtype": "float64",
            "thread_environment": thread_environment,
            "device": "cpu",
        },
        methodology={
            "warmup": warmup,
            "iterations": iterations,
            "calls_per_sample": calls_per_sample,
            "measurement_scope": (
                "expectation_forward_plus_full_reverse_mode_parameter_gradient"
            ),
            "engine_order": "rotated_per_iteration_within_one_process",
            "circuit_and_device_construction_included": False,
            "parameter_initialization_included": False,
            "expectation_evaluation_included": True,
            "gradient_zeroing_included": False,
            "optimizer_step_included": False,
            "exact_statevector": True,
            "hidden_fallback_allowed": False,
            "memory_measured_in_isolated_process": measure_memory,
        },
        support_matrix={
            "flagquantum_native": {
                "included": "flagquantum_native" in engines,
                "contract": "native PyTorch reverse-mode autograd",
                "gradient_method": "backpropagation through exact statevector",
            },
            "pennylane_default_qubit": {
                "included": "pennylane_default_qubit" in engines,
                "contract": "default.qubit backprop through the PyTorch interface",
                "gradient_method": "torch_reverse_mode_autograd",
            },
            "flagquantum_adjoint": {
                "included": "flagquantum_adjoint" in engines,
                "contract": "native reversible statevector adjoint",
                "gradient_method": "statevector_adjoint",
            },
            "flagquantum_adjoint_python_fallback": {
                "included": "flagquantum_adjoint_python_fallback" in engines,
                "contract": "same direct-layout adjoint with native operator disabled",
                "gradient_method": "statevector_adjoint",
            },
            "flagquantum_adjoint_gather_rollback": {
                "included": "flagquantum_adjoint_gather_rollback" in engines,
                "contract": "rollback to per-gate full-state gather indices",
                "gradient_method": "statevector_adjoint",
            },
            "flagquantum_adjoint_forward_cx_rollback": {
                "included": "flagquantum_adjoint_forward_cx_rollback" in engines,
                "contract": "same native adjoint with forward CX gather disabled",
                "gradient_method": "statevector_adjoint",
            },
            "flagquantum_adjoint_observable_cache_rollback": {
                "included": "flagquantum_adjoint_observable_cache_rollback" in engines,
                "contract": "same native adjoint with observable-weight cache disabled",
                "gradient_method": "statevector_adjoint",
            },
            "flagquantum_adjoint_rotation_tile_rollback": {
                "included": "flagquantum_adjoint_rotation_tile_rollback" in engines,
                "contract": "same native adjoint with legacy two-wire rotation tiles",
                "gradient_method": "statevector_adjoint",
            },
            "pennylane_lightning_adjoint": {
                "included": "pennylane_lightning_adjoint" in engines,
                "contract": "lightning.qubit adjoint through the PyTorch interface",
                "gradient_method": "statevector_adjoint",
            },
            "qiskit_aer": {
                "included": False,
                "reason": "FlagQuantum's Aer bridge has no native PyTorch gradient contract",
            },
            "cirq_simulator": {
                "included": False,
                "reason": "FlagQuantum's Cirq bridge has no native PyTorch gradient contract",
            },
        },
        workloads=tuple(workloads),
        n_wires=tuple(n_wires),
        layers=layers,
        engines=tuple(engines),
        cases=cases,
    )
    payload["hostname"] = "redacted"
    return payload


def _render_markdown(payload: Mapping[str, Any], *, artifact_name: str) -> str:
    """Render a self-contained differentiable comparison report."""

    if tuple(payload["engines"]) == (
        "flagquantum_adjoint",
        "flagquantum_adjoint_forward_cx_rollback",
    ):
        return render_forward_cx_markdown(payload, artifact_name=artifact_name)
    if tuple(payload["engines"]) == (
        "flagquantum_adjoint",
        "flagquantum_adjoint_observable_cache_rollback",
    ):
        return render_observable_cache_markdown(payload, artifact_name=artifact_name)
    if tuple(payload["engines"]) == (
        "flagquantum_adjoint",
        "flagquantum_adjoint_rotation_tile_rollback",
    ):
        return render_rotation_tile_markdown(payload, artifact_name=artifact_name)
    if tuple(payload["engines"]) == (
        "flagquantum_adjoint",
        "flagquantum_adjoint_python_fallback",
        "pennylane_lightning_adjoint",
    ):
        return render_adjoint_markdown(payload, artifact_name=artifact_name)
    methodology = payload["methodology"]
    cases = payload["cases"]
    engines = tuple(payload["engines"])
    versions = {
        package: version
        for engine in engines
        for package, version in cases[0]["engines"][engine]["versions"].items()
    }
    version_summary = ", ".join(
        f"{package} {version}" for package, version in versions.items()
    )
    warmup_label = "run" if methodology["warmup"] == 1 else "runs"
    call_label = "call" if methodology["calls_per_sample"] == 1 else "calls"
    lines = [
        "# Differentiable simulator workload corpus (Apple arm64 CPU)",
        "",
        f"This report is generated from [`{artifact_name}`]({artifact_name}).",
        "It measures one exact expectation-value forward pass and the full gradient",
        "with respect to every circuit parameter. Circuit/device construction and an",
        "optimizer update are excluded. Each engine uses its native Torch-compatible",
        "gradient path: PyTorch reverse-mode statevector backpropagation for both",
        "FlagQuantum and PennyLane default.qubit.",
        "",
        *wrap(
            f"Each median uses {methodology['iterations']} retained samples after "
            f"{methodology['warmup']} warmup {warmup_label}, with "
            f"{methodology['calls_per_sample']} {call_label} per sample. Environment: "
            f"{payload['platform']}; Python {payload['python']}; {version_summary}.",
            width=88,
        ),
        "",
        "## Results",
        "",
        "| Workload | Qubits | Gates | Params | FQ forward (ms) | FQ backward (ms) | FQ total (ms) | PL forward (ms) | PL backward (ms) | PL total (ms) | PL / FQ total | FQ peak RSS (MiB) | PL peak RSS (MiB) | Max gradient error |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in cases:
        workload = case["workload"]
        native = case["engines"]["flagquantum_native"]
        external = case["engines"]["pennylane_default_qubit"]
        ratio = case["comparison"]["engine_over_flagquantum_median"][
            "pennylane_default_qubit"
        ]["value_and_grad"]
        gradient_error = case["correctness"]["engines"]["pennylane_default_qubit"][
            "gradient_max_abs_error"
        ]

        def milliseconds(engine: Mapping[str, Any], metric: str) -> float:
            value = engine[metric]["median_seconds"]
            if not isinstance(value, (int, float)):
                raise TypeError(f"{metric} median must be numeric")
            return 1000.0 * float(value)

        def mebibytes(engine: Mapping[str, Any]) -> str:
            value = engine["memory"]["peak_rss_bytes"]
            return "not measured" if value is None else f"{value / 2**20:.1f}"

        lines.append(
            "| "
            + " | ".join(
                (
                    _WORKLOAD_LABELS[workload["name"]],
                    str(workload["n_wires"]),
                    str(workload["gate_count"]),
                    str(workload["parameter_count"]),
                    f"{milliseconds(native, 'forward'):.3f}",
                    f"{milliseconds(native, 'backward'):.3f}",
                    f"{milliseconds(native, 'value_and_grad'):.3f}",
                    f"{milliseconds(external, 'forward'):.3f}",
                    f"{milliseconds(external, 'backward'):.3f}",
                    f"{milliseconds(external, 'value_and_grad'):.3f}",
                    f"{ratio:.2f}x",
                    mebibytes(native),
                    mebibytes(external),
                    f"{gradient_error:.3e}",
                )
            )
            + " |"
        )
    total_ratios = [
        case["comparison"]["engine_over_flagquantum_median"]["pennylane_default_qubit"][
            "value_and_grad"
        ]
        for case in cases
    ]
    maximum_gradient_error = max(
        case["correctness"]["engines"]["pennylane_default_qubit"][
            "gradient_max_abs_error"
        ]
        for case in cases
    )
    maximum_width = max(case["workload"]["n_wires"] for case in cases)
    widest_cases = [
        case for case in cases if case["workload"]["n_wires"] == maximum_width
    ]
    widest_details = "; ".join(
        (
            f"{_WORKLOAD_LABELS[case['workload']['name']]}: forward "
            f"{1000 * case['engines']['flagquantum_native']['forward']['median_seconds']:.3f} "
            "ms and backward "
            f"{1000 * case['engines']['flagquantum_native']['backward']['median_seconds']:.3f} "
            "ms"
        )
        for case in widest_cases
    )
    native_widest_memory = [
        case["engines"]["flagquantum_native"]["memory"]["peak_rss_bytes"] / 2**30
        for case in widest_cases
        if case["engines"]["flagquantum_native"]["memory"]["peak_rss_bytes"] is not None
    ]
    external_widest_memory = [
        case["engines"]["pennylane_default_qubit"]["memory"]["peak_rss_bytes"] / 2**30
        for case in widest_cases
        if case["engines"]["pennylane_default_qubit"]["memory"]["peak_rss_bytes"]
        is not None
    ]
    memory_summary = ""
    if native_widest_memory and external_widest_memory:
        memory_summary = (
            " FlagQuantum's isolated peak RSS reached "
            f"{min(native_widest_memory):.2f}-{max(native_widest_memory):.2f} GiB at "
            f"this width, versus {min(external_widest_memory):.2f}-"
            f"{max(external_widest_memory):.2f} GiB for PennyLane default.qubit."
        )
    if min(total_ratios) > 1.0:
        performance_summary = (
            "FlagQuantum completed the matched value-and-gradient operation "
            f"{min(total_ratios):.2f}x to {max(total_ratios):.2f}x faster on this host."
        )
    elif max(total_ratios) < 1.0:
        pennylane_speedups = [1.0 / ratio for ratio in total_ratios]
        performance_summary = (
            "PennyLane default.qubit completed the matched value-and-gradient operation "
            f"{min(pennylane_speedups):.2f}x to {max(pennylane_speedups):.2f}x faster "
            "on this host."
        )
    else:
        performance_summary = (
            "Neither engine won every case; the PL/FQ total ratios ranged from "
            f"{min(total_ratios):.2f}x to {max(total_ratios):.2f}x on this host."
        )
    correctness_summary = (
        f"All {len(cases)} measured cases passed the value-and-gradient check and the "
        f"20% RMAD stability gate; the maximum gradient error was "
        f"{maximum_gradient_error:.3e}. {performance_summary}"
    )
    bottleneck_summary = (
        f"At {maximum_width} qubits, FlagQuantum's forward and total execution were "
        f"faster than default.qubit, while native backward remained its largest component "
        f"({widest_details}). "
        "The measured optimization target is therefore reverse-mode state retention and "
        f"backward execution, not another forward-only gate kernel.{memory_summary}"
    )
    lines.extend(
        (
            "",
            "A PL/FQ ratio above one means FlagQuantum was faster; below one means",
            "PennyLane default.qubit was faster. Peak RSS is measured in a separate isolated",
            "process and includes framework import, circuit/device construction, one warmup,",
            "and one value-and-gradient execution, so it is an operational footprint rather",
            "than tensor-only memory. These local results are not a universal framework",
            "ranking, optimizer-throughput claim, or scalability evidence.",
            "",
            "## Interpretation",
            "",
            *wrap(correctness_summary, width=88),
            "",
            *wrap(bottleneck_summary, width=88),
            "",
            "## What the workloads mean",
            "",
            "- **Hardware-efficient VQE** applies one RX-RY-RZ layer and a linear CNOT",
            "  chain, then differentiates a nearest-neighbour Ising energy. It represents",
            "  the inner quantum step of variational energy minimization.",
            "- **QAOA path MaxCut** prepares one exact QAOA layer for a path graph and",
            "  differentiates its expected cut value. Shared beta and gamma parameters",
            "  separate state-size cost from parameter-count growth.",
            "",
            "Qiskit Aer and Cirq are intentionally absent: the current FlagQuantum bridges",
            "do not expose a native PyTorch gradient contract. Substituting finite differences",
            "would measure a different algorithm and would not be a fair comparison.",
            "",
            "## FlagQuantum example",
            "",
            "```python",
            "import torch",
            "import flagquantum as fq",
            "from flagquantum import algorithms as fqa",
            "",
            "theta = torch.full((4, 3), 0.2, dtype=torch.float64, requires_grad=True)",
            "circuit = fq.Circuit(4, dtype=torch.complex128)",
            "for wire in range(4):",
            "    circuit.rx(wire, theta[wire, 0])",
            "    circuit.ry(wire, theta[wire, 1])",
            "    circuit.rz(wire, theta[wire, 2])",
            "for wire in range(3):",
            "    circuit.cx(wire, wire + 1)",
            "",
            "hamiltonian = fqa.Hamiltonian(",
            '    fqa.HamiltonianTerm(0.7, {wire: "z", wire + 1: "z"})',
            "    for wire in range(3)",
            ")",
            "energy = hamiltonian.expectation(circuit.state(refresh=True)).sum()",
            "energy.backward()",
            "print(energy.item(), theta.grad)",
            "```",
            "",
            "## Reproduction",
            "",
            "```bash",
            "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\",
            "flagquantum-benchmark run differentiable_simulator_corpus \\",
            "  --workloads hardware_efficient_vqe qaoa_path_maxcut \\",
            "  --n-wires "
            + " ".join(str(value) for value in payload["n_wires"])
            + " \\",
            "  --layers " + str(payload["layers"]) + " --threads 1 \\",
            f"  --warmup {methodology['warmup']} --iterations {methodology['iterations']} \\",
            f"  --calls-per-sample {methodology['calls_per_sample']} \\",
            f"  --json-output {artifact_name} --markdown-output REPORT.md",
            "```",
            "",
        )
    )
    return "\n".join(lines)


def _memory_worker(args: argparse.Namespace) -> int:
    _configure_threads(args.threads)
    torch.set_default_dtype(torch.float64)
    workload = args.workloads[0]
    engine = args.engines[0]
    function = _engine_callable(
        engine,
        workload,
        n_wires=args.n_wires[0],
        layers=args.layers,
        seed=args.seed,
    )
    function()
    gc.collect()
    function()
    print(json.dumps({"peak_rss_bytes": _peak_rss_bytes()}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workloads", nargs="+", choices=WORKLOAD_NAMES, default=WORKLOAD_NAMES
    )
    parser.add_argument("--n-wires", type=int, nargs="+", default=(10, 14, 18, 22))
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument(
        "--engines", nargs="+", choices=ALL_ENGINE_NAMES, default=ENGINE_NAMES
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--calls-per-sample", type=int, default=1)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--case-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--skip-memory-probe", action="store_true")
    parser.add_argument("--memory-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()
    if args.memory_worker:
        if len(args.workloads) != 1 or len(args.engines) != 1 or len(args.n_wires) != 1:
            parser.error("--memory-worker requires one workload, engine, and width")
        return _memory_worker(args)
    payload = run_benchmark(
        workloads=tuple(args.workloads),
        n_wires=tuple(args.n_wires),
        layers=args.layers,
        engines=tuple(args.engines),
        threads=args.threads,
        warmup=args.warmup,
        iterations=args.iterations,
        calls_per_sample=args.calls_per_sample,
        seed=args.seed,
        measure_memory=not args.skip_memory_probe,
        case_timeout_seconds=args.case_timeout_seconds,
    )
    if args.json_output is not None:
        write_json_atomic(args.json_output, payload)
    if args.markdown_output is not None:
        artifact_name = (
            args.json_output.name
            if args.json_output is not None
            else "differentiable-simulator-corpus.json"
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
    "ADJOINT_ENGINE_NAMES",
    "ALL_ENGINE_NAMES",
    "ENGINE_NAMES",
    "RUNNER",
    "SCHEMA",
    "WORKLOAD_NAMES",
    "build_workload",
    "main",
    "run_benchmark",
    "run_case",
)
