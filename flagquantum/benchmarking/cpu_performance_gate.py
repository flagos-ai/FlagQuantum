"""Profile-aware regression gate for the maintained CPU benchmark corpora.

Wall-clock results are comparable only when the workload, measurement method,
and relevant runtime profile agree.  This module deliberately returns an
``incomparable`` verdict instead of treating a different host configuration as
a regression or a pass.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from .contract import write_json_atomic

SCHEMA = "flagquantum.cpu_performance_regression_gate.v1"
SUPPORTED_SCHEMAS = {
    "flagquantum.batched_statevector_memory.v1",
    "flagquantum.simulator_workload_corpus.v1",
    "flagquantum.differentiable_simulator_corpus.v1",
}
DEFAULT_MAX_SLOWDOWN = 1.20
DEFAULT_MAX_MEMORY_GROWTH = 1.10
DEFAULT_MINIMUM_SAMPLES = 5
DEFAULT_MINIMUM_MEMORY_PROBES = 3


def _report_metadata(*, source_schema: object) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "runner": "cpu_performance_gate",
        "benchmark": "cpu_performance_regression_gate",
        "artifact_class": "generated_comparison_report",
        "benchmark_evidence_class": "comparison_non_release",
        "claim_evidence_type": "unknown",
        "distribution_semantics": (
            "timing_single_process_memory_median_of_fresh_processes"
            if source_schema == "flagquantum.batched_statevector_memory.v1"
            else "single_process_single_device"
        ),
        "scalability_claim_allowed": False,
        "release_gate_allowed": False,
        "non_release_evidence": True,
        "scalability_blockers": ["comparison_result_not_release_scalability_evidence"],
    }


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _sequence(value: object, name: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    return value


def _required(value: object, name: str) -> object:
    if value is None or value == "":
        raise ValueError(f"benchmark payload requires {name}")
    return value


def _version_family(value: object, name: str) -> str:
    value = _required(value, name)
    parts = str(value).split(".")
    return ".".join(parts[:2])


def comparison_profile(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the runtime and measurement fields that make timings comparable."""
    environment = _mapping(payload.get("environment"), "environment")
    methodology = _mapping(payload.get("methodology"), "methodology")
    thread_environment = _mapping(
        environment.get("thread_environment"), "environment.thread_environment"
    )
    profile = {
        "schema": payload.get("schema"),
        "platform": _required(payload.get("platform"), "platform"),
        "python_family": _version_family(payload.get("python"), "python"),
        "machine": _required(environment.get("machine"), "environment.machine"),
        "device": _required(environment.get("device"), "environment.device"),
        "torch_family": _version_family(environment.get("torch"), "environment.torch"),
        "torch_threads": _required(
            environment.get("torch_threads"), "environment.torch_threads"
        ),
        "thread_environment": {
            name: _required(
                thread_environment.get(name), f"environment.thread_environment.{name}"
            )
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        "exact_statevector": _required(
            methodology.get("exact_statevector"), "methodology.exact_statevector"
        ),
    }
    if payload.get("schema") == "flagquantum.batched_statevector_memory.v1":
        profile.update(
            {
                "timing_scope": _required(
                    methodology.get("timing_scope"), "methodology.timing_scope"
                ),
                "memory_scope": _required(
                    methodology.get("memory_scope"), "methodology.memory_scope"
                ),
                "memory_api": _required(
                    methodology.get("memory_api"), "methodology.memory_api"
                ),
            }
        )
    else:
        profile.update(
            {
                "measurement_scope": _required(
                    methodology.get("measurement_scope"),
                    "methodology.measurement_scope",
                ),
                "calls_per_sample": _required(
                    methodology.get("calls_per_sample"),
                    "methodology.calls_per_sample",
                ),
            }
        )
    return profile


def _case_key(case: Mapping[str, Any], *, schema: object) -> str:
    workload = _mapping(case.get("workload"), "case.workload")
    required = (
        ("name", "n_wires", "batch_size", "dtype", "scalar_ir_content_hash")
        if schema == "flagquantum.batched_statevector_memory.v1"
        else ("name", "n_wires", "dtype", "ir_content_hash")
    )
    missing = [name for name in required if workload.get(name) is None]
    if missing:
        raise ValueError(f"case workload is missing identity fields: {missing}")
    return "/".join(str(workload[name]) for name in required)


def _measurements(
    payload: Mapping[str, Any], case: Mapping[str, Any]
) -> dict[str, Any]:
    engines = _mapping(case.get("engines"), "case.engines")
    if payload["schema"] == "flagquantum.batched_statevector_memory.v1":
        engine = _mapping(
            engines.get("flagquantum_native_batch"), "flagquantum_native_batch"
        )
        return {"batch_total": _mapping(engine.get("batch_total"), "batch_total")}
    if payload["schema"] == "flagquantum.simulator_workload_corpus.v1":
        engine = _mapping(engines.get("flagquantum_native"), "flagquantum_native")
        return {"end_to_end": _mapping(engine.get("end_to_end"), "end_to_end")}
    engine = _mapping(engines.get("flagquantum_adjoint"), "flagquantum_adjoint")
    return {
        name: _mapping(engine.get(name), name)
        for name in ("forward", "backward", "value_and_grad")
    }


def _native_engine(payload: Mapping[str, Any]) -> str:
    if payload["schema"] == "flagquantum.batched_statevector_memory.v1":
        return "flagquantum_native_batch"
    if payload["schema"] == "flagquantum.simulator_workload_corpus.v1":
        return "flagquantum_native"
    return "flagquantum_adjoint"


def _case_index(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    cases = _sequence(payload.get("cases"), "cases")
    result: dict[str, Mapping[str, Any]] = {}
    for raw_case in cases:
        case = _mapping(raw_case, "case")
        key = _case_key(case, schema=payload.get("schema"))
        if key in result:
            raise ValueError(f"duplicate benchmark case: {key}")
        result[key] = case
    return result


def evaluate(
    baseline: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    max_slowdown: float = DEFAULT_MAX_SLOWDOWN,
    max_memory_growth: float = DEFAULT_MAX_MEMORY_GROWTH,
    minimum_samples: int = DEFAULT_MINIMUM_SAMPLES,
    minimum_memory_probes: int = DEFAULT_MINIMUM_MEMORY_PROBES,
) -> dict[str, Any]:
    """Compare two corpus payloads and return a machine-readable verdict."""
    if max_slowdown < 1.0:
        raise ValueError("max_slowdown must be at least 1.0")
    if max_memory_growth < 1.0:
        raise ValueError("max_memory_growth must be at least 1.0")
    if minimum_samples < 1:
        raise ValueError("minimum_samples must be positive")
    if minimum_memory_probes < 1:
        raise ValueError("minimum_memory_probes must be positive")
    schemas = {baseline.get("schema"), current.get("schema")}
    if len(schemas) != 1 or current.get("schema") not in SUPPORTED_SCHEMAS:
        raise ValueError("baseline and current must use the same supported schema")

    baseline_profile = comparison_profile(baseline)
    current_profile = comparison_profile(current)
    profile_differences = {
        name: {"baseline": baseline_profile[name], "current": current_profile[name]}
        for name in baseline_profile
        if baseline_profile[name] != current_profile[name]
    }
    if profile_differences:
        return {
            **_report_metadata(source_schema=current.get("schema")),
            "verdict": "incomparable",
            "passed": False,
            "profile_differences": profile_differences,
            "cases": [],
        }

    baseline_cases = _case_index(baseline)
    current_cases = _case_index(current)
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    missing = sorted(set(baseline_cases) - set(current_cases))
    unexpected = sorted(set(current_cases) - set(baseline_cases))
    for key in missing:
        failures.append(f"missing case: {key}")
    for key in sorted(set(baseline_cases) & set(current_cases)):
        current_case = current_cases[key]
        correctness = _mapping(current_case.get("correctness"), "case.correctness")
        correctness_engines = (
            _mapping(correctness.get("engines"), "case.correctness.engines")
            if current["schema"] == "flagquantum.batched_statevector_memory.v1"
            else {}
        )
        stability = _mapping(current_case.get("stability"), "case.stability")
        stability_engines = _mapping(stability.get("engines"), "case.stability.engines")
        native_stable = stability_engines.get(_native_engine(current)) is True
        case_failures: list[str] = []
        if correctness.get("passed") is not True:
            case_failures.append("correctness failed")
        if current["schema"] == "flagquantum.batched_statevector_memory.v1":
            if len(correctness_engines) < 2:
                case_failures.append("independent correctness comparison missing")
            if any(
                _mapping(result, "case.correctness.engine").get("passed") is not True
                for result in correctness_engines.values()
            ):
                case_failures.append("correctness engine failed")
            baseline_correctness = _mapping(
                baseline_cases[key].get("correctness"), "baseline.correctness"
            )
            baseline_tolerance = float(
                cast(
                    Any,
                    _required(
                        baseline_correctness.get("absolute_tolerance"),
                        "baseline.correctness.absolute_tolerance",
                    ),
                )
            )
            current_tolerance = float(
                cast(
                    Any,
                    _required(
                        correctness.get("absolute_tolerance"),
                        "case.correctness.absolute_tolerance",
                    ),
                )
            )
            if current_tolerance != baseline_tolerance:
                case_failures.append(
                    "absolute tolerance changed from "
                    f"{baseline_tolerance:g} to {current_tolerance:g}"
                )
        if not native_stable:
            case_failures.append("native measurement stability failed")
        metric_rows: dict[str, Any] = {}
        baseline_metrics = _measurements(baseline, baseline_cases[key])
        current_metrics = _measurements(current, current_case)
        for name, baseline_metric in baseline_metrics.items():
            current_metric = current_metrics[name]
            baseline_seconds = float(baseline_metric["median_seconds"])
            current_seconds = float(current_metric["median_seconds"])
            sample_count = int(current_metric["sample_count"])
            if not math.isfinite(baseline_seconds) or baseline_seconds <= 0:
                raise ValueError(f"{key}: {name} baseline median must be positive")
            if not math.isfinite(current_seconds) or current_seconds <= 0:
                raise ValueError(f"{key}: {name} current median must be positive")
            ratio = current_seconds / baseline_seconds
            metric_passed = sample_count >= minimum_samples and ratio <= max_slowdown
            if sample_count < minimum_samples:
                case_failures.append(
                    f"{name} has {sample_count} samples; requires {minimum_samples}"
                )
            if ratio > max_slowdown:
                case_failures.append(
                    f"{name} slowdown {ratio:.3f} exceeds {max_slowdown:.3f}"
                )
            metric_rows[name] = {
                "baseline_median_seconds": baseline_seconds,
                "current_median_seconds": current_seconds,
                "current_sample_count": sample_count,
                "current_over_baseline": ratio,
                "passed": metric_passed,
            }
        if current["schema"] == "flagquantum.batched_statevector_memory.v1":
            baseline_engine = _mapping(
                _mapping(baseline_cases[key].get("engines"), "case.engines").get(
                    "flagquantum_native_batch"
                ),
                "flagquantum_native_batch",
            )
            current_engine = _mapping(
                _mapping(current_case.get("engines"), "case.engines").get(
                    "flagquantum_native_batch"
                ),
                "flagquantum_native_batch",
            )
            baseline_memory = _mapping(
                baseline_engine.get("isolated_memory"), "isolated_memory"
            )
            current_memory = _mapping(
                current_engine.get("isolated_memory"), "isolated_memory"
            )
            baseline_peak = int(baseline_memory["peak_rss_bytes"])
            current_peak = int(current_memory["peak_rss_bytes"])
            probe_count = int(current_memory["sample_count"])
            if baseline_peak <= 0 or current_peak <= 0:
                raise ValueError(f"{key}: peak RSS must be positive")
            memory_ratio = current_peak / baseline_peak
            memory_passed = (
                probe_count >= minimum_memory_probes
                and memory_ratio <= max_memory_growth
            )
            if probe_count < minimum_memory_probes:
                case_failures.append(
                    "peak_rss has "
                    f"{probe_count} probes; requires {minimum_memory_probes}"
                )
            if memory_ratio > max_memory_growth:
                case_failures.append(
                    f"peak_rss growth {memory_ratio:.3f} exceeds "
                    f"{max_memory_growth:.3f}"
                )
            metric_rows["peak_rss"] = {
                "baseline_bytes": baseline_peak,
                "current_bytes": current_peak,
                "current_probe_count": probe_count,
                "current_over_baseline": memory_ratio,
                "passed": memory_passed,
            }
        if case_failures:
            failures.extend(f"{key}: {message}" for message in case_failures)
        rows.append(
            {
                "case": key,
                "passed": not case_failures,
                "correctness_passed": correctness.get("passed") is True,
                "stability_passed": native_stable,
                "metrics": metric_rows,
                "failures": case_failures,
            }
        )

    return {
        **_report_metadata(source_schema=current.get("schema")),
        "verdict": "pass" if not failures else "fail",
        "passed": not failures,
        "profile": current_profile,
        "policy": {
            "max_slowdown": max_slowdown,
            "minimum_samples": minimum_samples,
            "correctness_required": True,
            "stability_required": True,
            "case_matrix_must_not_shrink": True,
            **(
                {
                    "max_memory_growth": max_memory_growth,
                    "minimum_memory_probes": minimum_memory_probes,
                }
                if current["schema"] == "flagquantum.batched_statevector_memory.v1"
                else {}
            ),
        },
        "missing_cases": missing,
        "unexpected_cases": unexpected,
        "failures": failures,
        "cases": rows,
    }


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("current", type=Path)
    parser.add_argument("--max-slowdown", type=float, default=DEFAULT_MAX_SLOWDOWN)
    parser.add_argument(
        "--max-memory-growth", type=float, default=DEFAULT_MAX_MEMORY_GROWTH
    )
    parser.add_argument("--minimum-samples", type=int, default=DEFAULT_MINIMUM_SAMPLES)
    parser.add_argument(
        "--minimum-memory-probes",
        type=int,
        default=DEFAULT_MINIMUM_MEMORY_PROBES,
    )
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    report = evaluate(
        _load(args.baseline),
        _load(args.current),
        max_slowdown=args.max_slowdown,
        max_memory_growth=args.max_memory_growth,
        minimum_samples=args.minimum_samples,
        minimum_memory_probes=args.minimum_memory_probes,
    )
    if args.json_output is not None:
        write_json_atomic(args.json_output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["verdict"] == "incomparable":
        return 2
    return 0 if report["passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["SCHEMA", "comparison_profile", "evaluate", "main"]
