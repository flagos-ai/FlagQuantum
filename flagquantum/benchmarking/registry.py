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
