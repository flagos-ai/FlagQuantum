"""Budgeted, explicit live calibration for simulator advice."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import platform
import statistics
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace

import torch

from flagquantum.core import CircuitIR


@dataclass(frozen=True)
class CalibrationMeasurement:
    """One engine's live calibration result."""

    engine: str
    samples_seconds: tuple[float, ...]
    median_seconds: float | None
    relative_median_absolute_deviation: float | None
    correctness_passed: bool
    error: str | None


@dataclass(frozen=True)
class CalibrationOutcome:
    """Live measurements and their cache/provenance state."""

    measurements: tuple[CalibrationMeasurement, ...]
    elapsed_seconds: float
    cache_hit: bool
    evidence_sha256: str


_CACHE: dict[str, CalibrationOutcome] = {}
_CACHE_LOCK = threading.Lock()


def _runner(engine: str) -> Callable[[CircuitIR], torch.Tensor]:
    if engine == "flagquantum_native":
        run = importlib.import_module("flagquantum").run
    else:
        modules = {
            "qiskit_aer": "flagquantum.ecosystem.qiskit",
            "cirq_simulator": "flagquantum.ecosystem.cirq",
            "pennylane_lightning_qubit": "flagquantum.ecosystem.pennylane",
        }
        module_name = modules.get(engine)
        if module_name is None:
            raise ValueError(f"unsupported calibration engine {engine!r}")
        run = importlib.import_module(module_name).run

    def execute(ir: CircuitIR) -> torch.Tensor:
        result = run(ir)
        state = result.to_statevector()
        if not isinstance(state, torch.Tensor):
            raise TypeError("simulator calibration requires a torch statevector")
        return state.detach().cpu()

    return execute


def _cache_key(
    ir: CircuitIR,
    *,
    engines: Sequence[str],
    versions: Mapping[str, str | None],
    warmup: int,
    repeats: int,
    budget_seconds: float,
) -> str:
    payload = {
        "ir_content_hash": ir.content_hash,
        "engines": list(engines),
        "versions": {engine: versions.get(engine) for engine in engines},
        "warmup": warmup,
        "repeats": repeats,
        "budget_seconds": budget_seconds,
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _digest(
    ir: CircuitIR,
    measurements: Sequence[CalibrationMeasurement],
) -> str:
    payload = {
        "ir_content_hash": ir.content_hash,
        "measurements": [
            {
                "engine": item.engine,
                "samples_seconds": item.samples_seconds,
                "median_seconds": item.median_seconds,
                "relative_median_absolute_deviation": (
                    item.relative_median_absolute_deviation
                ),
                "correctness_passed": item.correctness_passed,
                "error": item.error,
            }
            for item in measurements
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _timing(samples: Sequence[float]) -> tuple[float, float]:
    median = statistics.median(samples)
    deviation = statistics.median(abs(value - median) for value in samples)
    return median, deviation / median if median else math.inf


def calibrate(
    ir: CircuitIR,
    *,
    engines: Sequence[str],
    versions: Mapping[str, str | None],
    warmup: int,
    repeats: int,
    budget_seconds: float,
    use_cache: bool,
) -> CalibrationOutcome:
    """Run exact statevector probes under a soft wall-clock budget.

    The budget is checked between backend calls. A single backend call cannot be
    interrupted safely and can therefore overrun the requested budget.
    """

    ordered = tuple(dict.fromkeys(engines))
    if "flagquantum_native" not in ordered:
        ordered = ("flagquantum_native", *ordered)
    key = _cache_key(
        ir,
        engines=ordered,
        versions=versions,
        warmup=warmup,
        repeats=repeats,
        budget_seconds=budget_seconds,
    )
    if use_cache:
        with _CACHE_LOCK:
            cached = _CACHE.get(key)
        if cached is not None:
            return replace(cached, cache_hit=True)

    started = time.perf_counter()
    per_engine_budget = budget_seconds / len(ordered)
    reference: torch.Tensor | None = None
    measurements: list[CalibrationMeasurement] = []
    for engine in ordered:
        if (
            engine != "flagquantum_native"
            and time.perf_counter() - started >= budget_seconds
        ):
            measurements.append(
                CalibrationMeasurement(
                    engine=engine,
                    samples_seconds=(),
                    median_seconds=None,
                    relative_median_absolute_deviation=None,
                    correctness_passed=False,
                    error="calibration_budget_exhausted",
                )
            )
            continue
        engine_started = time.perf_counter()
        try:
            execute = _runner(engine)
            state: torch.Tensor | None = None
            for _ in range(warmup):
                state = execute(ir)
            samples: list[float] = []
            while len(samples) < repeats:
                if (
                    len(samples) >= 2
                    and time.perf_counter() - engine_started >= per_engine_budget
                ):
                    break
                sample_started = time.perf_counter()
                state = execute(ir)
                samples.append(time.perf_counter() - sample_started)
            assert state is not None
            if engine == "flagquantum_native":
                reference = state
                correct = True
            elif reference is None:
                correct = False
            else:
                tolerance = 1e-5 if ir.dtype == "complex64" else 1e-10
                correct = bool(
                    torch.allclose(
                        state.to(torch.complex128),
                        reference.to(torch.complex128),
                        rtol=tolerance,
                        atol=tolerance,
                    )
                )
            median, relative_mad = _timing(samples)
            measurements.append(
                CalibrationMeasurement(
                    engine=engine,
                    samples_seconds=tuple(samples),
                    median_seconds=median,
                    relative_median_absolute_deviation=relative_mad,
                    correctness_passed=correct,
                    error=None if correct else "statevector_mismatch",
                )
            )
        except Exception as exc:  # backend failures are structured evidence
            measurements.append(
                CalibrationMeasurement(
                    engine=engine,
                    samples_seconds=(),
                    median_seconds=None,
                    relative_median_absolute_deviation=None,
                    correctness_passed=False,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
    elapsed = time.perf_counter() - started
    outcome = CalibrationOutcome(
        measurements=tuple(measurements),
        elapsed_seconds=elapsed,
        cache_hit=False,
        evidence_sha256=_digest(ir, measurements),
    )
    if use_cache:
        with _CACHE_LOCK:
            _CACHE[key] = outcome
    return outcome


__all__ = ("CalibrationMeasurement", "CalibrationOutcome", "calibrate")
