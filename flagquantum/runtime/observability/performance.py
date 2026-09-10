"""Typed continuous performance, memory and progress contracts."""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Literal, Sequence

PERFORMANCE_SCHEMA_VERSION = "flagquantum_performance_v1"
BenchmarkLayer = Literal[
    "micro",
    "kernel",
    "circuit",
    "training_step",
    "communication",
    "compile_time",
    "startup",
    "memory",
]


@dataclass(frozen=True)
class PerformanceThresholds:
    max_latency_regression_fraction: float = 0.10
    max_memory_regression_fraction: float = 0.10
    max_coefficient_of_variation: float = 0.10
    min_useful_work_fraction: float = 0.10
    max_memory_slope_bytes_per_step: float = 1024.0
    max_heartbeat_gap_seconds: float = 30.0


@dataclass(frozen=True)
class PerformanceTelemetrySample:
    timestamp_seconds: float
    gpu_utilization_percent: float | None
    kernel_active: bool
    memory_allocated_bytes: int
    memory_reserved_bytes: int
    communication_progress_bytes: int
    cpu_wait_seconds: float
    compilation_state: str
    completed_work_units: int


class PerformanceMonitor:
    """Profiler hook for time-series activity, memory and progress samples."""

    def __init__(self) -> None:
        self._started = time.perf_counter()
        self._samples: list[PerformanceTelemetrySample] = []

    def sample(
        self,
        *,
        gpu_utilization_percent: float | None,
        kernel_active: bool,
        memory_allocated_bytes: int,
        memory_reserved_bytes: int,
        communication_progress_bytes: int,
        cpu_wait_seconds: float,
        compilation_state: str,
        completed_work_units: int,
    ) -> PerformanceTelemetrySample:
        item = PerformanceTelemetrySample(
            timestamp_seconds=time.perf_counter() - self._started,
            gpu_utilization_percent=gpu_utilization_percent,
            kernel_active=kernel_active,
            memory_allocated_bytes=memory_allocated_bytes,
            memory_reserved_bytes=memory_reserved_bytes,
            communication_progress_bytes=communication_progress_bytes,
            cpu_wait_seconds=cpu_wait_seconds,
            compilation_state=compilation_state,
            completed_work_units=completed_work_units,
        )
        self._samples.append(item)
        return item

    @property
    def samples(self) -> tuple[PerformanceTelemetrySample, ...]:
        return tuple(self._samples)


@dataclass(frozen=True)
class PerformanceRecord:
    benchmark: str
    layer: BenchmarkLayer
    backend: str
    device: str
    distribution_semantics: str
    world_size: int
    warmup: int
    repetitions: int
    samples_seconds: tuple[float, ...]
    initialization_seconds: float
    teardown_seconds: float
    peak_memory_allocated_bytes: int
    peak_memory_reserved_bytes: int
    allocated_memory_by_step: tuple[int, ...]
    reserved_memory_by_step: tuple[int, ...]
    useful_work_seconds: float
    communication_seconds: float
    idle_seconds: float
    completed_work_units: int
    heartbeat_gaps_seconds: tuple[float, ...]
    correctness_passed: bool
    explanation: str = ""
    schema: str = PERFORMANCE_SCHEMA_VERSION

    @property
    def mean_seconds(self) -> float:
        return statistics.fmean(self.samples_seconds)

    @property
    def median_seconds(self) -> float:
        return statistics.median(self.samples_seconds)

    @property
    def standard_deviation_seconds(self) -> float:
        return (
            statistics.stdev(self.samples_seconds)
            if len(self.samples_seconds) > 1
            else 0.0
        )

    @property
    def coefficient_of_variation(self) -> float:
        return self.standard_deviation_seconds / self.mean_seconds

    @property
    def robust_coefficient_of_variation(self) -> float:
        median = self.median_seconds
        deviations = tuple(abs(value - median) for value in self.samples_seconds)
        return statistics.median(deviations) / max(median, 1e-12)

    @property
    def communication_fraction(self) -> float:
        total = (
            self.useful_work_seconds + self.communication_seconds + self.idle_seconds
        )
        return self.communication_seconds / max(total, 1e-12)

    @property
    def idle_fraction(self) -> float:
        total = (
            self.useful_work_seconds + self.communication_seconds + self.idle_seconds
        )
        return self.idle_seconds / max(total, 1e-12)

    @property
    def useful_work_fraction(self) -> float:
        total = (
            self.useful_work_seconds + self.communication_seconds + self.idle_seconds
        )
        return self.useful_work_seconds / max(total, 1e-12)

    def summary(self) -> dict[str, object]:
        return {
            **asdict(self),
            "mean_seconds": self.mean_seconds,
            "median_seconds": self.median_seconds,
            "standard_deviation_seconds": self.standard_deviation_seconds,
            "coefficient_of_variation": self.coefficient_of_variation,
            "robust_coefficient_of_variation": self.robust_coefficient_of_variation,
            "communication_fraction": self.communication_fraction,
            "idle_fraction": self.idle_fraction,
            "useful_work_fraction": self.useful_work_fraction,
            "allocated_memory_slope_bytes_per_step": memory_slope(
                self.allocated_memory_by_step
            ),
            "reserved_memory_slope_bytes_per_step": memory_slope(
                self.reserved_memory_by_step
            ),
        }


@dataclass(frozen=True)
class PerformanceGateResult:
    passed: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class CostModelCalibration:
    seconds_per_work_unit: float
    relative_uncertainty: float
    sample_count: int
    source_benchmark: str


@dataclass(frozen=True)
class CrossoverSelection:
    minimum_global_elements: int | None
    uncertainty_margin_fraction: float
    measured_world_sizes: tuple[int, ...]
    measurements: tuple[dict[str, float | int], ...]


def calibrate_crossover(
    measurements: Sequence[dict[str, float | int]],
    *,
    distributed_world_size: int,
) -> CrossoverSelection:
    """Find the first measured size where distributed wins beyond uncertainty."""
    local = {
        int(row["global_elements"]): row
        for row in measurements
        if int(row["world_size"]) == 1
    }
    distributed = {
        int(row["global_elements"]): row
        for row in measurements
        if int(row["world_size"]) == distributed_world_size
    }
    threshold = None
    margin = 0.0
    for elements in sorted(local.keys() & distributed.keys()):
        one, many = local[elements], distributed[elements]
        uncertainty = max(
            float(one["relative_uncertainty"]),
            float(many["relative_uncertainty"]),
        )
        margin = max(margin, uncertainty)
        if float(many["median_seconds"]) * (1 + uncertainty) < float(
            one["median_seconds"]
        ) * (1 - uncertainty):
            threshold = elements
            break
    return CrossoverSelection(
        minimum_global_elements=threshold,
        uncertainty_margin_fraction=margin,
        measured_world_sizes=tuple(
            sorted({int(row["world_size"]) for row in measurements})
        ),
        measurements=tuple(dict(row) for row in measurements),
    )


def memory_slope(samples: Sequence[int]) -> float:
    """Least-squares byte slope across repeated steady-state steps."""

    if len(samples) < 2:
        return 0.0
    x_mean = (len(samples) - 1) / 2
    y_mean = statistics.fmean(samples)
    numerator = sum(
        (index - x_mean) * (value - y_mean) for index, value in enumerate(samples)
    )
    denominator = sum((index - x_mean) ** 2 for index in range(len(samples)))
    return numerator / denominator


def classify_no_progress(record: PerformanceRecord, *, phase: str) -> str:
    max_gap = max(record.heartbeat_gaps_seconds, default=0.0)
    if record.completed_work_units > 0:
        return "progressing"
    if phase in {"compile", "jit", "data_loading", "synchronization", "checkpoint"}:
        return f"expected_{phase}_interval"
    return "stalled" if max_gap > 0 else "no_work_started"


def calibrate_cost_model(record: PerformanceRecord) -> CostModelCalibration:
    total_seconds = sum(record.samples_seconds)
    return CostModelCalibration(
        seconds_per_work_unit=total_seconds / max(record.completed_work_units, 1),
        relative_uncertainty=record.robust_coefficient_of_variation,
        sample_count=len(record.samples_seconds),
        source_benchmark=record.benchmark,
    )


def _current_record_issues(
    record: PerformanceRecord,
    thresholds: PerformanceThresholds,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if not record.correctness_passed:
        errors.append("correctness_check_failed")
    if record.robust_coefficient_of_variation > thresholds.max_coefficient_of_variation:
        errors.append("latency_variance_exceeds_threshold")
    if record.useful_work_fraction < thresholds.min_useful_work_fraction:
        if record.explanation:
            warnings.append("persistently_low_useful_work_fraction_explained")
        else:
            errors.append("persistently_low_useful_work_fraction")
    if (
        max(
            memory_slope(record.allocated_memory_by_step),
            memory_slope(record.reserved_memory_by_step),
        )
        > thresholds.max_memory_slope_bytes_per_step
    ):
        errors.append("unbounded_memory_growth_suspected")
    if (
        max(record.heartbeat_gaps_seconds, default=0.0)
        > thresholds.max_heartbeat_gap_seconds
    ):
        errors.append("no_progress_heartbeat_timeout")
    if record.communication_fraction > 0.5:
        warnings.append("communication_dominates_useful_work")
    if not math.isfinite(record.mean_seconds):
        errors.append("non_finite_latency")
    return errors, warnings


def _baseline_regressions(
    current: PerformanceRecord,
    baseline: PerformanceRecord,
    thresholds: PerformanceThresholds,
) -> list[str]:
    errors: list[str] = []
    latency_limit = baseline.median_seconds * (
        1 + thresholds.max_latency_regression_fraction
    )
    if current.median_seconds > latency_limit:
        errors.append("latency_regression")
    memory_limit = baseline.peak_memory_reserved_bytes * (
        1 + thresholds.max_memory_regression_fraction
    )
    if current.peak_memory_reserved_bytes > memory_limit:
        errors.append("memory_regression")
    return errors


def evaluate_performance(
    current: PerformanceRecord,
    *,
    baseline: PerformanceRecord | None = None,
    thresholds: PerformanceThresholds = PerformanceThresholds(),
) -> PerformanceGateResult:
    errors, warnings = _current_record_issues(current, thresholds)
    if baseline is not None:
        errors.extend(_baseline_regressions(current, baseline, thresholds))
    return PerformanceGateResult(not errors, tuple(errors), tuple(warnings))


__all__ = [
    "PERFORMANCE_SCHEMA_VERSION",
    "BenchmarkLayer",
    "CostModelCalibration",
    "CrossoverSelection",
    "PerformanceGateResult",
    "PerformanceMonitor",
    "PerformanceRecord",
    "PerformanceTelemetrySample",
    "PerformanceThresholds",
    "calibrate_cost_model",
    "calibrate_crossover",
    "classify_no_progress",
    "evaluate_performance",
    "memory_slope",
]
