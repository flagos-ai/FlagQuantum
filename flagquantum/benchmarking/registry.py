"""Lazy registry for reproducible benchmark runners."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module

RunnerMain = Callable[[], int]


@dataclass(frozen=True)
class RunnerSpec:
    """Discoverable benchmark metadata without importing the implementation."""

    name: str
    module: str
    attribute: str
    category: str
    summary: str
    hardware: str
    example: str


_RUNNERS: dict[str, RunnerSpec] = {
    "environment_probe": RunnerSpec(
        name="environment_probe",
        module="flagquantum.benchmarking.environment_probe",
        attribute="main",
        category="environment",
        summary="Record the runtime, package, CPU, and accelerator environment.",
        hardware="CPU; GPU optional",
        example=(
            "flagquantum-benchmark run environment_probe "
            "--json-output results/environment.json"
        ),
    ),
    "statevector_local": RunnerSpec(
        name="statevector_local",
        module="flagquantum.benchmarking.statevector_local",
        attribute="main",
        category="statevector",
        summary="Measure local statevector execution on CPU or one GPU.",
        hardware="CPU or one GPU",
        example=(
            "flagquantum-benchmark run statevector_local --device cpu "
            "--n-wires 12 --batch-size 4 --layers 2 --warmup 3 "
            "--iterations 10 --json-output results/statevector.json"
        ),
    ),
    "statevector_cpu_paths": RunnerSpec(
        name="statevector_cpu_paths",
        module="flagquantum.benchmarking.statevector_cpu_paths",
        attribute="main",
        category="statevector",
        summary="Measure each CPU statevector fast path separately on one host.",
        hardware="CPU only",
        example=(
            "flagquantum-benchmark run statevector_cpu_paths "
            "--n-wires 20 --layers 8 --warmup 2 --iterations 5 "
            "--json-output results/cpu-paths.json"
        ),
    ),
    "simulator_compare": RunnerSpec(
        name="simulator_compare",
        module="flagquantum.benchmarking.simulator_compare",
        attribute="main",
        category="interop",
        summary="Compare exact statevector execution across simulator engines.",
        hardware="CPU; Qiskit optional dependency required",
        example=(
            "flagquantum-benchmark run simulator_compare --n-wires 10 14 18 22 24 "
            "--layers 2 --threads 1 --warmup 3 --iterations 9 "
            "--calls-per-sample 5 "
            "--json-output benchmarks/results/comparison/simulators.json"
        ),
    ),
    "simulator_compare_cirq": RunnerSpec(
        name="simulator_compare_cirq",
        module="flagquantum.benchmarking.external_simulator_compare",
        attribute="cirq_main",
        category="interop",
        summary="Measure Cirq Simulator on the shared exact-statevector workload.",
        hardware="CPU; Cirq optional dependency required",
        example=(
            "flagquantum-benchmark run simulator_compare_cirq "
            "--n-wires 10 14 18 22 24 --layers 2 --threads 1 "
            "--json-output benchmarks/results/comparison/cirq.json"
        ),
    ),
    "simulator_compare_pennylane": RunnerSpec(
        name="simulator_compare_pennylane",
        module="flagquantum.benchmarking.external_simulator_compare",
        attribute="pennylane_main",
        category="interop",
        summary="Measure PennyLane Lightning on the shared statevector workload.",
        hardware="CPU; PennyLane optional dependency required",
        example=(
            "flagquantum-benchmark run simulator_compare_pennylane "
            "--n-wires 10 14 18 22 24 --layers 2 --threads 1 "
            "--json-output benchmarks/results/comparison/pennylane.json"
        ),
    ),
    "simulator_comparison_report": RunnerSpec(
        name="simulator_comparison_report",
        module="flagquantum.benchmarking.simulator_comparison_report",
        attribute="main",
        category="interop",
        summary="Generate validated JSON and Markdown simulator comparisons.",
        hardware="No simulator execution required",
        example=(
            "flagquantum-benchmark run simulator_comparison_report "
            "benchmarks/results/comparison/*.json "
            "--json-output comparison.json --markdown-output comparison.md"
        ),
    ),
    "simulator_workload_corpus": RunnerSpec(
        name="simulator_workload_corpus",
        module="flagquantum.benchmarking.simulator_workload_corpus",
        attribute="main",
        category="interop",
        summary="Measure a feature-labelled workload corpus across simulators.",
        hardware="CPU; Qiskit, Cirq, and PennyLane optional dependencies required",
        example=(
            "flagquantum-benchmark run simulator_workload_corpus "
            "--n-wires 10 14 18 22 --threads 1 --warmup 1 --iterations 5 "
            "--json-output benchmarks/results/comparison/workload-corpus.json"
        ),
    ),
    "statevector_weak_scaling": RunnerSpec(
        name="statevector_weak_scaling",
        module="flagquantum.benchmarking.statevector_weak_scaling",
        attribute="main",
        category="statevector",
        summary="Aggregate audited weak-scaling result payloads.",
        hardware="No accelerator required for report generation",
        example=(
            "flagquantum-benchmark run statevector_weak_scaling INPUT... "
            "--json-output results/weak.json"
        ),
    ),
    "statevector_strong_scaling": RunnerSpec(
        name="statevector_strong_scaling",
        module="flagquantum.benchmarking.statevector_strong_scaling",
        attribute="main",
        category="statevector",
        summary="Aggregate audited strong-scaling result payloads.",
        hardware="No accelerator required for report generation",
        example=(
            "flagquantum-benchmark run statevector_strong_scaling INPUT... "
            "--json-output results/strong.json"
        ),
    ),
    "statevector_training_scaling": RunnerSpec(
        name="statevector_training_scaling",
        module="flagquantum.benchmarking.statevector_training_scaling",
        attribute="main",
        category="statevector",
        summary="Aggregate distributed statevector training results.",
        hardware="No accelerator required for report generation",
        example=(
            "flagquantum-benchmark run statevector_training_scaling INPUT... "
            "--json-output results/training.json"
        ),
    ),
}


def register(
    name: str,
    module: str,
    attribute: str = "main",
    *,
    category: str = "extension",
    summary: str = "Third-party benchmark runner.",
    hardware: str = "Runner-defined",
    example: str = "",
) -> None:
    """Register a runner exactly once; duplicate names are rejected."""
    if not name or not module or not attribute:
        raise ValueError("runner registration requires name, module, and attribute")
    if re.fullmatch(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*", name) is None:
        raise ValueError("runner name must use lowercase snake_case")
    if name in _RUNNERS:
        raise ValueError(f"benchmark runner already registered: {name}")
    _RUNNERS[name] = RunnerSpec(
        name=name,
        module=module,
        attribute=attribute,
        category=category,
        summary=summary,
        hardware=hardware,
        example=example,
    )


def names() -> tuple[str, ...]:
    return tuple(sorted(_RUNNERS))


def specs() -> tuple[RunnerSpec, ...]:
    return tuple(_RUNNERS[name] for name in names())


def describe(name: str) -> RunnerSpec:
    try:
        return _RUNNERS[name]
    except KeyError as exc:
        raise KeyError(f"unknown benchmark runner: {name}") from exc


def resolve(name: str) -> RunnerMain:
    runner = describe(name)
    entrypoint = getattr(import_module(runner.module), runner.attribute)
    if not callable(entrypoint):
        raise TypeError(f"benchmark runner {name!r} entrypoint must be callable")

    def run() -> int:
        exit_code = entrypoint()
        if not isinstance(exit_code, int):
            raise TypeError(
                f"benchmark runner {name!r} must return an integer exit code"
            )
        return exit_code

    return run


__all__ = ["RunnerSpec", "describe", "names", "register", "resolve", "specs"]
