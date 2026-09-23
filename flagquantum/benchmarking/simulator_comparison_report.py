#!/usr/bin/env python3
"""Validate simulator measurements and render deterministic comparison reports."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .contract import write_json_atomic

INPUT_SCHEMA = "flagquantum.simulator_comparison.v1"
SCHEMA = "flagquantum.simulator_comparison_report.v1"
RUNNER = "simulator_comparison_report"
REFERENCE_ENGINE = "flagquantum_native"

_ENGINE_LABELS = {
    "flagquantum_native": "FlagQuantum",
    "qiskit_aer": "Qiskit Aer",
    "cirq_simulator": "Cirq Simulator",
    "pennylane_lightning_qubit": "PennyLane Lightning",
}
_ENGINE_ORDER = tuple(_ENGINE_LABELS)
_ENVIRONMENT_KEYS = ("device", "machine", "processor", "torch", "torch_threads")
_METHODOLOGY_KEYS = (
    "warmup",
    "iterations",
    "setup_iterations",
    "calls_per_sample",
    "conversion_included_in_steady_state",
    "compilation_included_in_steady_state",
    "result_retrieval_included_in_execution",
    "output_basis_normalization_included_in_execution",
    "hidden_fallback_allowed",
)


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{context} must be a JSON object with string keys")
    return value


def _sequence(value: Any, context: str) -> list[Any] | tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{context} must be a JSON array")
    return value


def _finite_non_negative(value: Any, context: str, *, positive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric")
    result = float(value)
    minimum_ok = result > 0.0 if positive else result >= 0.0
    if not minimum_ok or result == float("inf") or result != result:
        qualifier = "positive" if positive else "finite and non-negative"
        raise ValueError(f"{context} must be {qualifier}")
    return result


def _canonical(value: Mapping[str, Any]) -> str:
    return json.dumps(dict(value), sort_keys=True, separators=(",", ":"))


def _fingerprint(workload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(workload).encode("utf-8")).hexdigest()


def _load_payload(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError(
            f"cannot read simulator comparison input {path}: {exc}"
        ) from exc
    try:
        payload = _mapping(json.loads(raw), str(path))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in simulator comparison input {path}") from exc
    return payload, {
        "path": path.as_posix(),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _case_index(
    payload: Mapping[str, Any], source: str
) -> tuple[dict[str, dict[str, Any]], tuple[str, ...]]:
    cases = _sequence(payload.get("cases"), f"{source}.cases")
    if not cases:
        raise ValueError(f"{source}.cases must contain at least one workload")
    indexed: dict[str, dict[str, Any]] = {}
    engine_names: tuple[str, ...] | None = None
    for position, raw_case in enumerate(cases):
        case = _mapping(raw_case, f"{source}.cases[{position}]")
        workload = _mapping(
            case.get("workload"), f"{source}.cases[{position}].workload"
        )
        fingerprint = _fingerprint(workload)
        if fingerprint in indexed:
            raise ValueError(f"{source} contains a duplicate workload: {workload}")
        engines = _mapping(case.get("engines"), f"{source}.cases[{position}].engines")
        current_names = tuple(sorted(engines))
        if not current_names:
            raise ValueError(
                f"{source}.cases[{position}] contains no engine measurements"
            )
        if engine_names is None:
            engine_names = current_names
        elif current_names != engine_names:
            raise ValueError(
                f"{source} must measure the same engines for every workload"
            )
        indexed[fingerprint] = case
    assert engine_names is not None
    return indexed, engine_names


def _comparison_identity(payload: Mapping[str, Any], source: str) -> dict[str, Any]:
    environment = _mapping(payload.get("environment"), f"{source}.environment")
    methodology = _mapping(payload.get("methodology"), f"{source}.methodology")
    identity: dict[str, Any] = {
        "hostname": payload.get("hostname"),
        "platform": payload.get("platform"),
        "python": payload.get("python"),
        "world_size": payload.get("world_size"),
    }
    for key in _ENVIRONMENT_KEYS:
        if key not in environment:
            raise ValueError(f"{source}.environment is missing fairness field {key!r}")
        identity[key] = environment[key]
    for key in _METHODOLOGY_KEYS:
        if key not in methodology:
            raise ValueError(f"{source}.methodology is missing fairness field {key!r}")
        identity[key] = methodology[key]
    return identity


def _validate_input(payload: Mapping[str, Any], source: str) -> None:
    expected = {
        "schema": INPUT_SCHEMA,
        "benchmark": "interoperable_simulator_comparison",
        "distribution_semantics": "single_process_single_device",
        "scalability_claim_allowed": False,
        "release_gate_allowed": False,
        "non_release_evidence": True,
    }
    for key, expected_value in expected.items():
        if payload.get(key) != expected_value:
            raise ValueError(
                f"{source}.{key} must be {expected_value!r}, got {payload.get(key)!r}"
            )


def _engine_order(names: Sequence[str]) -> tuple[str, ...]:
    priority = {name: index for index, name in enumerate(_ENGINE_ORDER)}
    return tuple(
        sorted(names, key=lambda name: (priority.get(name, len(priority)), name))
    )


def _engine_measurement(
    *,
    engine: str,
    case: Mapping[str, Any],
    source: str,
) -> dict[str, Any]:
    engines = _mapping(case.get("engines"), f"{source}.engines")
    measured = _mapping(engines.get(engine), f"{source}.engines.{engine}")
    timing = _mapping(
        measured.get("steady_state"), f"{source}.engines.{engine}.steady_state"
    )
    median = _finite_non_negative(
        timing.get("median_seconds"),
        f"{source}.engines.{engine}.steady_state.median_seconds",
        positive=True,
    )
    relative_mad = _finite_non_negative(
        timing.get("relative_median_absolute_deviation"),
        f"{source}.engines.{engine}.steady_state.relative_median_absolute_deviation",
        positive=False,
    )
    sample_count = timing.get("sample_count")
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count < 1
    ):
        raise ValueError(
            f"{source}.engines.{engine}.steady_state.sample_count must be positive"
        )
    correctness = _mapping(case.get("correctness"), f"{source}.correctness")
    tolerance = _finite_non_negative(
        correctness.get("absolute_tolerance"),
        f"{source}.correctness.absolute_tolerance",
        positive=True,
    )
    max_error = _finite_non_negative(
        correctness.get("max_abs_error"),
        f"{source}.correctness.max_abs_error",
        positive=False,
    )
    stability = _mapping(case.get("stability"), f"{source}.stability")
    threshold = _finite_non_negative(
        stability.get("maximum_relative_median_absolute_deviation"),
        f"{source}.stability.maximum_relative_median_absolute_deviation",
        positive=False,
    )
    version = measured.get("version")
    if not isinstance(version, str) or not version:
        raise ValueError(
            f"{source}.engines.{engine}.version must be a non-empty string"
        )
    backend = measured.get("backend")
    if backend is not None and not isinstance(backend, str):
        raise ValueError(f"{source}.engines.{engine}.backend must be a string or null")
    return {
        "label": _ENGINE_LABELS.get(engine, engine),
        "version": version,
        "backend": backend,
        "median_seconds": median,
        "median_milliseconds": median * 1000.0,
        "sample_count": sample_count,
        "relative_median_absolute_deviation": relative_mad,
        "stable": relative_mad <= threshold,
        "correctness_passed": correctness.get("passed") is True,
        "max_abs_error": max_error,
        "absolute_tolerance": tolerance,
        "source": source,
    }


def _attach_history(
    report: dict[str, Any],
    baseline: Mapping[str, Any] | None,
    *,
    baseline_source: str | None,
    threshold_percent: float,
) -> None:
    if baseline is None:
        report["historical_comparison"] = {
            "baseline_supplied": False,
            "regression_gate_enabled": False,
        }
        return
    if baseline.get("schema") != SCHEMA:
        raise ValueError(f"historical baseline must use schema {SCHEMA}")
    if baseline.get("comparison_identity") != report["comparison_identity"]:
        raise ValueError("historical baseline comparison identity is incompatible")
    baseline_rows = {
        row["workload_fingerprint"]: row
        for row in _sequence(baseline.get("rows"), "historical baseline rows")
        if isinstance(row, dict) and isinstance(row.get("workload_fingerprint"), str)
    }
    current_fingerprints = {row["workload_fingerprint"] for row in report["rows"]}
    if set(baseline_rows) != current_fingerprints:
        raise ValueError("historical baseline workload matrix is incompatible")
    changes: list[dict[str, Any]] = []
    regressions = 0
    threshold_fraction = threshold_percent / 100.0
    for row in report["rows"]:
        fingerprint = row["workload_fingerprint"]
        baseline_row = baseline_rows[fingerprint]
        old_engines = _mapping(
            baseline_row.get("engines"), "historical baseline row engines"
        )
        current_engines = _mapping(row.get("engines"), "current report row engines")
        all_engines = _engine_order(tuple(set(old_engines) | set(current_engines)))
        for engine in all_engines:
            current = current_engines.get(engine)
            old = old_engines.get(engine)
            if current is None:
                verdict = "removed"
                current_seconds = None
                baseline_seconds = _mapping(old, "baseline engine").get(
                    "median_seconds"
                )
                change_percent = None
            elif old is None:
                verdict = "new"
                current_seconds = _mapping(current, "current engine").get(
                    "median_seconds"
                )
                baseline_seconds = None
                change_percent = None
            else:
                current_seconds = _finite_non_negative(
                    _mapping(current, "current engine").get("median_seconds"),
                    "current median_seconds",
                    positive=True,
                )
                baseline_seconds = _finite_non_negative(
                    _mapping(old, "baseline engine").get("median_seconds"),
                    "baseline median_seconds",
                    positive=True,
                )
                change = current_seconds / baseline_seconds - 1.0
                change_percent = change * 100.0
                if change > threshold_fraction:
                    verdict = "regression"
                    regressions += 1
                elif change < -threshold_fraction:
                    verdict = "improvement"
                else:
                    verdict = "within_threshold"
            changes.append(
                {
                    "workload_fingerprint": fingerprint,
                    "n_wires": row["workload"]["n_wires"],
                    "engine": engine,
                    "baseline_seconds": baseline_seconds,
                    "current_seconds": current_seconds,
                    "change_percent": change_percent,
                    "verdict": verdict,
                }
            )
    report["historical_comparison"] = {
        "baseline_supplied": True,
        "baseline_source": baseline_source,
        "regression_threshold_percent": threshold_percent,
        "regression_gate_enabled": False,
        "regression_count": regressions,
        "changes": changes,
    }


def build_report(
    inputs: Sequence[str | Path],
    *,
    baseline: str | Path | None = None,
    regression_threshold_percent: float = 10.0,
) -> dict[str, Any]:
    """Build one comparison report from compatible measured artifacts.

    Args:
        inputs: Raw simulator comparison JSON artifacts.
        baseline: Optional earlier report generated by this function.
        regression_threshold_percent: Informational historical change threshold.

    Returns:
        A deterministic, non-release comparison report.

    Raises:
        ValueError: If inputs are missing, malformed, duplicated, or incompatible.
    """

    if len(inputs) < 2:
        raise ValueError("comparison reporting requires at least two input artifacts")
    threshold = _finite_non_negative(
        regression_threshold_percent,
        "regression_threshold_percent",
        positive=False,
    )
    loaded = []
    for raw_path in inputs:
        path = Path(raw_path)
        payload, source_record = _load_payload(path)
        _validate_input(payload, source_record["path"])
        cases, engines = _case_index(payload, source_record["path"])
        identity = _comparison_identity(payload, source_record["path"])
        loaded.append((payload, source_record, cases, engines, identity))

    reference_entries = [entry for entry in loaded if REFERENCE_ENGINE in entry[3]]
    if len(reference_entries) != 1:
        raise ValueError("exactly one input artifact must measure flagquantum_native")
    reference = reference_entries[0]
    reference_fingerprints = set(reference[2])
    reference_identity = reference[4]
    owners: dict[str, str] = {}
    for _payload, source, cases, engines, identity in loaded:
        if identity != reference_identity:
            raise ValueError(
                f"{source['path']} has an incompatible environment or methodology"
            )
        if set(cases) != reference_fingerprints:
            raise ValueError(f"{source['path']} has an incompatible workload matrix")
        for fingerprint in reference_fingerprints:
            reference_workload = _mapping(
                reference[2][fingerprint].get("workload"), "reference workload"
            )
            candidate_workload = _mapping(
                cases[fingerprint].get("workload"), "candidate workload"
            )
            if candidate_workload != reference_workload:
                raise ValueError(f"{source['path']} has an incompatible workload")
            reference_correctness = _mapping(
                reference[2][fingerprint].get("correctness"),
                "reference correctness",
            )
            candidate_correctness = _mapping(
                cases[fingerprint].get("correctness"),
                "candidate correctness",
            )
            for key in ("absolute_tolerance", "reference_semantics"):
                if candidate_correctness.get(key) != reference_correctness.get(key):
                    raise ValueError(
                        f"{source['path']} has incompatible correctness field {key!r}"
                    )
        for engine in engines:
            if engine in owners:
                raise ValueError(
                    f"engine {engine!r} is duplicated by {owners[engine]} and "
                    f"{source['path']}"
                )
            owners[engine] = source["path"]

    ordered_engines = _engine_order(tuple(owners))
    rows: list[dict[str, Any]] = []
    reference_cases = _sequence(reference[0].get("cases"), "reference cases")
    for raw_reference_case in reference_cases:
        reference_case = _mapping(raw_reference_case, "reference case")
        workload = _mapping(reference_case.get("workload"), "reference workload")
        fingerprint = _fingerprint(workload)
        measurements: dict[str, Any] = {}
        for _payload, source, cases, engines, _identity in loaded:
            for engine in engines:
                measurements[engine] = _engine_measurement(
                    engine=engine,
                    case=cases[fingerprint],
                    source=source["path"],
                )
        reference_seconds = measurements[REFERENCE_ENGINE]["median_seconds"]
        ratios = {
            engine: measurements[engine]["median_seconds"] / reference_seconds
            for engine in ordered_engines
        }
        rows.append(
            {
                "workload": dict(workload),
                "workload_fingerprint": fingerprint,
                "engines": {engine: measurements[engine] for engine in ordered_engines},
                "ratios_to_flagquantum": ratios,
                "ratio_semantics": (
                    "external divided by FlagQuantum; above one means FlagQuantum "
                    "was faster in this measured case"
                ),
            }
        )

    correctness_passed = all(
        measurement["correctness_passed"]
        for row in rows
        for measurement in row["engines"].values()
    )
    all_stable = all(
        measurement["stable"] for row in rows for measurement in row["engines"].values()
    )
    report: dict[str, Any] = {
        "runner": RUNNER,
        "schema": SCHEMA,
        "benchmark": "interoperable_simulator_comparison_report",
        "artifact_class": "generated_comparison_report",
        "benchmark_evidence_class": "comparison_non_release",
        "claim_evidence_type": "unknown",
        "distribution_semantics": "single_process_single_device",
        "scalability_claim_allowed": False,
        "release_gate_allowed": False,
        "non_release_evidence": True,
        "scalability_blockers": ["comparison_result_not_release_scalability_evidence"],
        "passed": correctness_passed,
        "correctness_passed": correctness_passed,
        "all_measurements_stable": all_stable,
        "reference_engine": REFERENCE_ENGINE,
        "engine_order": list(ordered_engines),
        "comparison_identity": reference_identity,
        "sources": [entry[1] for entry in loaded],
        "rows": rows,
    }
    baseline_payload = None
    baseline_source = None
    if baseline is not None:
        baseline_path = Path(baseline)
        baseline_payload, baseline_record = _load_payload(baseline_path)
        baseline_source = baseline_record["path"]
    _attach_history(
        report,
        baseline_payload,
        baseline_source=baseline_source,
        threshold_percent=threshold,
    )
    return report


def _format_seconds(seconds: float) -> str:
    return f"{seconds * 1000.0:.3f} ms"


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render a deterministic human-readable view of a comparison report."""

    if report.get("schema") != SCHEMA:
        raise ValueError(f"report must use schema {SCHEMA}")
    rows = _sequence(report.get("rows"), "report.rows")
    engines = tuple(
        str(engine) for engine in _sequence(report.get("engine_order"), "engine_order")
    )
    if REFERENCE_ENGINE not in engines:
        raise ValueError("report is missing the FlagQuantum reference engine")
    identity = _mapping(report.get("comparison_identity"), "comparison_identity")
    lines = [
        "# Cross-framework CPU simulator comparison",
        "",
        (
            "This file is generated deterministically from measured JSON artifacts. "
            "It is local comparison evidence, not a universal ranking, scalability "
            "claim, or release gate."
        ),
        "",
        "## Scope",
        "",
        f"- Device: `{identity['device']}` on `{identity['machine']}`",
        "- Precision and workload: exact statevector rows listed below",
        f"- Worker threads: `{identity['torch_threads']}`",
        (
            "- Warmups / samples / calls per sample: "
            f"`{identity['warmup']}` / `{identity['iterations']}` / "
            f"`{identity['calls_per_sample']}`"
        ),
        (
            "- Ratios are external-engine median divided by FlagQuantum median; "
            "values above `1.00x` mean FlagQuantum was faster in that row."
        ),
        "",
        "## Results",
        "",
    ]
    headers = ["Qubits", "Gates"]
    for engine in engines:
        label = _ENGINE_LABELS.get(engine, engine)
        headers.append(f"{label} median")
        if engine != REFERENCE_ENGINE:
            headers.append(f"{label} / FQ")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---:" for _ in headers) + " |")
    for raw_row in rows:
        row = _mapping(raw_row, "report row")
        workload = _mapping(row.get("workload"), "report row workload")
        measurements = _mapping(row.get("engines"), "report row engines")
        ratios = _mapping(row.get("ratios_to_flagquantum"), "report row ratios")
        values = [str(workload["n_wires"]), str(workload["gate_count"])]
        for engine in engines:
            measurement = _mapping(measurements[engine], f"measurement {engine}")
            values.append(_format_seconds(float(measurement["median_seconds"])))
            if engine != REFERENCE_ENGINE:
                values.append(f"{float(ratios[engine]):.2f}x")
        lines.append("| " + " | ".join(values) + " |")

    lines.extend(["", "## Correctness and stability", ""])
    lines.append("| Qubits | Engine | Maximum error | R-MAD | Verdict |")
    lines.append("| ---: | --- | ---: | ---: | --- |")
    for raw_row in rows:
        row = _mapping(raw_row, "report row")
        workload = _mapping(row.get("workload"), "report row workload")
        measurements = _mapping(row.get("engines"), "report row engines")
        for engine in engines:
            measurement = _mapping(measurements[engine], f"measurement {engine}")
            correctness = (
                "correct" if measurement["correctness_passed"] else "incorrect"
            )
            stability = "stable" if measurement["stable"] else "noisy"
            lines.append(
                f"| {workload['n_wires']} | {_ENGINE_LABELS.get(engine, engine)} | "
                f"{float(measurement['max_abs_error']):.3e} | "
                f"{float(measurement['relative_median_absolute_deviation']):.3f} | "
                f"{correctness}, {stability} |"
            )

    history = _mapping(report.get("historical_comparison"), "historical_comparison")
    lines.extend(["", "## Historical comparison", ""])
    if not history["baseline_supplied"]:
        lines.append(
            "No historical baseline was supplied. This report makes no performance "
            "regression verdict."
        )
    else:
        lines.append(
            f"Baseline: `{history['baseline_source']}`. The informational threshold "
            f"is `{float(history['regression_threshold_percent']):.1f}%`; it does not "
            "gate CI."
        )
        lines.extend(
            [
                "",
                "| Qubits | Engine | Change | Verdict |",
                "| ---: | --- | ---: | --- |",
            ]
        )
        for raw_change in _sequence(history.get("changes"), "historical changes"):
            change = _mapping(raw_change, "historical change")
            percent = change["change_percent"]
            rendered = "n/a" if percent is None else f"{float(percent):+.2f}%"
            lines.append(
                f"| {change['n_wires']} | "
                f"{_ENGINE_LABELS.get(change['engine'], change['engine'])} | "
                f"{rendered} | {change['verdict']} |"
            )

    lines.extend(["", "## Sources and reproduction", ""])
    for raw_source in _sequence(report.get("sources"), "report.sources"):
        source = _mapping(raw_source, "report source")
        lines.append(f"- `{source['path']}` (`sha256:{source['sha256']}`)")
    lines.extend(
        [
            "",
            (
                "Regenerate the report from the repository root without rerunning "
                "any simulator:"
            ),
            "",
            "```bash",
            "flagquantum-benchmark run simulator_comparison_report \\",
        ]
    )
    for raw_source in _sequence(report.get("sources"), "report.sources"):
        source = _mapping(raw_source, "report source")
        lines.append(f"  {source['path']} \\")
    lines.extend(
        [
            "  --json-output comparison.json \\",
            "  --markdown-output comparison.md",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _check_exact(path: Path, expected: str) -> None:
    try:
        actual = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot check generated report {path}: {exc}") from exc
    if actual != expected:
        raise ValueError(f"generated report is stale: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--regression-threshold-percent", type=float, default=10.0)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if the requested output files differ from regenerated content.",
    )
    args = parser.parse_args()
    payload = build_report(
        args.inputs,
        baseline=args.baseline,
        regression_threshold_percent=args.regression_threshold_percent,
    )
    json_text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    markdown = render_markdown(payload)
    if args.check:
        if args.json_output is None or args.markdown_output is None:
            parser.error("--check requires --json-output and --markdown-output")
        _check_exact(args.json_output, json_text)
        _check_exact(args.markdown_output, markdown)
    else:
        if args.json_output is not None:
            write_json_atomic(args.json_output, payload)
        if args.markdown_output is not None:
            args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
            args.markdown_output.write_text(markdown, encoding="utf-8")
    print(json_text, end="")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["RUNNER", "SCHEMA", "build_report", "main", "render_markdown"]
