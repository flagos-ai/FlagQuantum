"""Evidence-scoped recommendations for explicit local simulator selection."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import operator
import platform
from collections.abc import Collection, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from importlib import import_module, metadata
from pathlib import Path
from typing import Any, Literal

import torch

from flagquantum.core import ensure_circuit_ir

from ._calibration import CalibrationOutcome, calibrate
from ._evidence import (
    SOURCE_PATH,
    SOURCE_SHA256,
    WORKLOAD_MANIFEST_PATH,
    WORKLOAD_MANIFEST_SHA256,
    bundled_report,
)

RecommendationStatus = Literal["recommended", "insufficient_evidence"]
EvidenceLevel = Literal["exact", "profile", "live_calibration", "none"]

_SCHEMA = "flagquantum.simulator_comparison_report.v1"
_REFERENCE_ENGINE = "flagquantum_native"
_ENVIRONMENT_KEYS = (
    "device",
    "machine",
    "processor",
    "platform",
    "python",
    "torch",
    "torch_threads",
)


class SimulatorAdvisorEvidenceError(ValueError):
    """Raised when simulator evidence is malformed or unsafe to recommend from."""


@dataclass(frozen=True)
class SimulatorCandidate:
    """One measured simulator candidate and its eligibility evidence."""

    engine: str
    label: str
    version: str | None
    installed_version: str | None
    available: bool
    eligible: bool
    median_seconds: float | None
    relative_to_flagquantum: float | None
    relative_to_fastest: float | None
    relative_median_absolute_deviation: float | None
    exclusion_reasons: tuple[str, ...]
    calibration_error: str | None = None


@dataclass(frozen=True)
class SimulatorRecommendation:
    """A recommendation bounded to one measured workload and environment."""

    status: RecommendationStatus
    recommended_engine: str | None
    reason: str
    candidates: tuple[SimulatorCandidate, ...]
    workload_fingerprint: str | None
    evidence_source: str
    evidence_sha256: str
    environment_matches: bool
    workload_matches: bool
    native_tie_margin: float
    limitations: tuple[str, ...]
    circuit_ir_hash: str | None = None
    circuit_matches: bool | None = None
    workload_manifest_source: str | None = None
    workload_manifest_sha256: str | None = None
    evidence_level: EvidenceLevel = "none"
    confidence: float | None = None
    calibration_elapsed_seconds: float | None = None
    calibration_cache_hit: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        return asdict(self)


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise SimulatorAdvisorEvidenceError(f"{context} must be a mapping")
    return value


def _sequence(value: Any, context: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SimulatorAdvisorEvidenceError(f"{context} must be a sequence")
    return value


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        result = int(operator.index(value))
    except TypeError:
        raise TypeError(f"{name} must be an integer") from None
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _non_negative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        result = int(operator.index(value))
    except TypeError:
        raise TypeError(f"{name} must be an integer") from None
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _finite_non_negative(value: Any, context: str, *, positive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SimulatorAdvisorEvidenceError(f"{context} must be numeric")
    result = float(value)
    valid = result > 0.0 if positive else result >= 0.0
    if not valid or not math.isfinite(result):
        qualifier = "positive" if positive else "finite and non-negative"
        raise SimulatorAdvisorEvidenceError(f"{context} must be {qualifier}")
    return result


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(value), sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _load_evidence(
    evidence: Mapping[str, Any] | str | Path | None,
) -> tuple[Mapping[str, Any], str, str]:
    if evidence is None:
        return bundled_report(), SOURCE_PATH, SOURCE_SHA256
    if isinstance(evidence, Mapping):
        return evidence, "provided_mapping", _canonical_sha256(evidence)
    path = Path(evidence)
    try:
        raw = path.read_bytes()
        loaded = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except (OSError, json.JSONDecodeError) as exc:
        raise SimulatorAdvisorEvidenceError(
            f"cannot read simulator evidence {path}: {exc}"
        ) from exc
    return _mapping(loaded, str(path)), path.as_posix(), hashlib.sha256(raw).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SimulatorAdvisorEvidenceError(
                f"simulator evidence contains duplicate field {key!r}"
            )
        result[key] = value
    return result


def _validate_report(report: Mapping[str, Any]) -> None:
    required = {
        "schema": _SCHEMA,
        "passed": True,
        "correctness_passed": True,
        "non_release_evidence": True,
        "release_gate_allowed": False,
        "reference_engine": _REFERENCE_ENGINE,
    }
    for key, expected in required.items():
        if report.get(key) != expected:
            raise SimulatorAdvisorEvidenceError(
                f"evidence {key} must be {expected!r}, got {report.get(key)!r}"
            )
    identity = _mapping(report.get("comparison_identity"), "comparison_identity")
    if identity.get("hidden_fallback_allowed") is not False:
        raise SimulatorAdvisorEvidenceError(
            "evidence must prohibit hidden backend fallback"
        )
    engines = tuple(
        str(item) for item in _sequence(report.get("engine_order"), "engine_order")
    )
    if _REFERENCE_ENGINE not in engines or len(set(engines)) != len(engines):
        raise SimulatorAdvisorEvidenceError(
            "engine_order must contain one unique FlagQuantum reference"
        )
    rows = _sequence(report.get("rows"), "rows")
    if not rows:
        raise SimulatorAdvisorEvidenceError("evidence must contain measured rows")


def _runtime_environment(*, torch_threads: int) -> dict[str, Any]:
    return {
        "device": "cpu",
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_threads": torch_threads,
    }


def _distribution_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _installed_versions() -> dict[str, str]:
    result = {}
    native_version = _distribution_version("flagquantum")
    if native_version is None:
        source_version = getattr(import_module("flagquantum"), "__version__", None)
        native_version = source_version if isinstance(source_version, str) else None
    if native_version is not None:
        result[_REFERENCE_ENGINE] = native_version
    if (
        importlib.util.find_spec("flagquantum.ecosystem.qiskit.aer")
        and importlib.util.find_spec("qiskit")
        and importlib.util.find_spec("qiskit_aer")
    ):
        version = _distribution_version("qiskit-aer")
        if version is not None:
            result["qiskit_aer"] = version
    if importlib.util.find_spec(
        "flagquantum.ecosystem.cirq.simulator"
    ) and importlib.util.find_spec("cirq"):
        version = _distribution_version("cirq-core")
        if version is not None:
            result["cirq_simulator"] = version
    if (
        importlib.util.find_spec("flagquantum.ecosystem.pennylane.lightning")
        and importlib.util.find_spec("pennylane")
        and importlib.util.find_spec("pennylane_lightning")
    ):
        version = _distribution_version("pennylane")
        lightning_version = _distribution_version("pennylane-lightning")
        if version is not None and lightning_version is not None:
            result["pennylane_lightning_qubit"] = version
    return result


def _engine_versions(
    available_engines: Mapping[str, str] | Collection[str] | None,
) -> dict[str, str | None]:
    if available_engines is None:
        return dict(_installed_versions())
    if isinstance(available_engines, Mapping):
        if not all(
            isinstance(engine, str) and isinstance(version, str) and version
            for engine, version in available_engines.items()
        ):
            raise TypeError(
                "available_engines mappings require non-empty string versions"
            )
        return dict(available_engines)
    if isinstance(available_engines, (str, bytes)) or not all(
        isinstance(engine, str) for engine in available_engines
    ):
        raise TypeError("available_engines must contain only engine-name strings")
    return dict.fromkeys(available_engines)


def _expected_workload(
    *,
    n_wires: int,
    workload: str,
    layers: int,
    dtype: str,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    gate_count = layers * (4 * n_wires + 1)
    return {
        "name": workload,
        "n_wires": n_wires,
        "layers": layers,
        "gate_count": gate_count,
        "batch_size": batch_size,
        "dtype": dtype,
        "seed": seed,
    }


def _empty_candidates(
    engines: Sequence[str],
    *,
    installed_versions: Mapping[str, str | None],
    reason: str,
) -> tuple[SimulatorCandidate, ...]:
    return tuple(
        SimulatorCandidate(
            engine=engine,
            label=engine,
            version=None,
            installed_version=installed_versions.get(engine),
            available=engine in installed_versions,
            eligible=False,
            median_seconds=None,
            relative_to_flagquantum=None,
            relative_to_fastest=None,
            relative_median_absolute_deviation=None,
            exclusion_reasons=(reason,),
        )
        for engine in engines
    )


def _recommend_profile(
    *,
    n_wires: int,
    workload: str = "hardware_efficient_statevector",
    layers: int = 2,
    dtype: str = "complex128",
    batch_size: int = 1,
    seed: int = 7319,
    torch_threads: int = 1,
    environment: Mapping[str, Any] | None = None,
    available_engines: Mapping[str, str] | Collection[str] | None = None,
    native_tie_margin: float = 0.10,
    evidence: Mapping[str, Any] | str | Path | None = None,
) -> SimulatorRecommendation:
    """Recommend an explicit simulator from matching profile evidence.

    The bundled evidence covers one exact CPU statevector workload on a recorded
    ARM64 environment. Environment or workload mismatches return
    ``insufficient_evidence`` instead of extrapolating. This function never
    executes a circuit and never changes ``fq.run`` routing.

    Args:
        n_wires: Exact measured workload width to query.
        workload: Benchmark workload identity.
        layers: Exact workload layer count.
        dtype: Exact statevector dtype.
        batch_size: Exact workload batch size.
        seed: Exact deterministic workload seed.
        torch_threads: Planned Torch worker-thread count.
        environment: Optional explicit environment for reproducible queries.
        available_engines: Optional installed-engine names or version mapping.
        native_tie_margin: Prefer native execution when an external result is
            faster by no more than this fraction.
        evidence: Optional report mapping or JSON path. The packaged measured
            report projection is used by default.

    Returns:
        A structured, non-executing recommendation with evidence and limits.

    Raises:
        SimulatorAdvisorEvidenceError: If evidence is malformed or unsafe.

    """

    width = _positive_integer(n_wires, "n_wires")
    layer_count = _positive_integer(layers, "layers")
    batches = _positive_integer(batch_size, "batch_size")
    threads = _positive_integer(torch_threads, "torch_threads")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    tie_margin = _finite_non_negative(
        native_tie_margin, "native_tie_margin", positive=False
    )
    if tie_margin >= 1.0:
        raise ValueError("native_tie_margin must be less than one")

    report, source, digest = _load_evidence(evidence)
    _validate_report(report)
    engine_order = tuple(
        str(item) for item in _sequence(report.get("engine_order"), "engine_order")
    )
    installed_versions = _engine_versions(available_engines)
    unknown = set(installed_versions).difference(engine_order)
    if unknown:
        raise ValueError(
            f"available_engines contains unknown engines: {sorted(unknown)}"
        )

    requested_environment = dict(
        environment
        if environment is not None
        else _runtime_environment(torch_threads=threads)
    )
    measured_environment = _mapping(
        report.get("comparison_identity"), "comparison_identity"
    )
    environment_matches = all(
        requested_environment.get(key) == measured_environment.get(key)
        for key in _ENVIRONMENT_KEYS
    )
    expected_workload = _expected_workload(
        n_wires=width,
        workload=workload,
        layers=layer_count,
        dtype=dtype,
        batch_size=batches,
        seed=seed,
    )
    matching_row: Mapping[str, Any] | None = None
    for raw_row in _sequence(report.get("rows"), "rows"):
        row = _mapping(raw_row, "row")
        if _mapping(row.get("workload"), "row.workload") == expected_workload:
            matching_row = row
            break
    if matching_row is None:
        return SimulatorRecommendation(
            status="insufficient_evidence",
            recommended_engine=None,
            reason="workload_not_measured",
            candidates=_empty_candidates(
                engine_order,
                installed_versions=installed_versions,
                reason="workload_not_measured",
            ),
            workload_fingerprint=None,
            evidence_source=source,
            evidence_sha256=digest,
            environment_matches=environment_matches,
            workload_matches=False,
            native_tie_margin=tie_margin,
            limitations=(
                "The advisor does not interpolate or extrapolate between measured workloads.",
                "Comparison evidence is non-release and not a scalability claim.",
            ),
        )
    if not environment_matches:
        return SimulatorRecommendation(
            status="insufficient_evidence",
            recommended_engine=None,
            reason="environment_not_measured",
            candidates=_empty_candidates(
                engine_order,
                installed_versions=installed_versions,
                reason="environment_not_measured",
            ),
            workload_fingerprint=str(matching_row.get("workload_fingerprint")),
            evidence_source=source,
            evidence_sha256=digest,
            environment_matches=False,
            workload_matches=True,
            native_tie_margin=tie_margin,
            limitations=(
                "The advisor requires an exact recorded environment match.",
                "Comparison evidence is non-release and not a scalability claim.",
            ),
        )

    measurements = _mapping(matching_row.get("engines"), "row.engines")
    ratios = _mapping(
        matching_row.get("ratios_to_flagquantum"), "row.ratios_to_flagquantum"
    )
    native_measurement = _mapping(
        measurements.get(_REFERENCE_ENGINE),
        f"row.engines.{_REFERENCE_ENGINE}",
    )
    native_seconds = _finite_non_negative(
        native_measurement.get("median_seconds"),
        f"row.engines.{_REFERENCE_ENGINE}.median_seconds",
        positive=True,
    )
    prepared = []
    eligible_seconds = []
    for engine in engine_order:
        measured = _mapping(measurements.get(engine), f"row.engines.{engine}")
        measured_version = measured.get("version")
        if not isinstance(measured_version, str) or not measured_version:
            raise SimulatorAdvisorEvidenceError(
                f"row.engines.{engine}.version must be a non-empty string"
            )
        median = _finite_non_negative(
            measured.get("median_seconds"),
            f"row.engines.{engine}.median_seconds",
            positive=True,
        )
        relative_mad = _finite_non_negative(
            measured.get("relative_median_absolute_deviation"),
            f"row.engines.{engine}.relative_median_absolute_deviation",
            positive=False,
        )
        ratio = _finite_non_negative(
            ratios.get(engine),
            f"row.ratios_to_flagquantum.{engine}",
            positive=True,
        )
        expected_ratio = median / native_seconds
        if not math.isclose(ratio, expected_ratio, rel_tol=1e-12, abs_tol=1e-15):
            raise SimulatorAdvisorEvidenceError(
                f"row ratio for {engine!r} is inconsistent with measured medians"
            )
        reasons = []
        installed_version = installed_versions.get(engine)
        if engine not in installed_versions:
            reasons.append("engine_not_available")
        elif installed_version is not None and installed_version != measured_version:
            reasons.append("version_mismatch")
        if measured.get("correctness_passed") is not True:
            reasons.append("correctness_not_passed")
        if measured.get("stable") is not True:
            reasons.append("measurement_not_stable")
        eligible = not reasons
        if eligible:
            eligible_seconds.append(median)
        prepared.append(
            (
                engine,
                measured,
                measured_version,
                installed_version,
                median,
                ratio,
                relative_mad,
                eligible,
                tuple(reasons),
            )
        )
    if not eligible_seconds:
        candidates = tuple(
            SimulatorCandidate(
                engine=engine,
                label=str(measured.get("label", engine)),
                version=measured_version,
                installed_version=installed_version,
                available=engine in installed_versions,
                eligible=eligible,
                median_seconds=median,
                relative_to_flagquantum=ratio,
                relative_to_fastest=None,
                relative_median_absolute_deviation=relative_mad,
                exclusion_reasons=reasons,
            )
            for (
                engine,
                measured,
                measured_version,
                installed_version,
                median,
                ratio,
                relative_mad,
                eligible,
                reasons,
            ) in prepared
        )
        return SimulatorRecommendation(
            status="insufficient_evidence",
            recommended_engine=None,
            reason="no_eligible_engine",
            candidates=candidates,
            workload_fingerprint=str(matching_row.get("workload_fingerprint")),
            evidence_source=source,
            evidence_sha256=digest,
            environment_matches=True,
            workload_matches=True,
            native_tie_margin=tie_margin,
            limitations=("No available engine has stable, correct measured evidence.",),
        )

    fastest_seconds = min(eligible_seconds)
    candidates = tuple(
        sorted(
            (
                SimulatorCandidate(
                    engine=engine,
                    label=str(measured.get("label", engine)),
                    version=measured_version,
                    installed_version=installed_version,
                    available=engine in installed_versions,
                    eligible=eligible,
                    median_seconds=median,
                    relative_to_flagquantum=ratio,
                    relative_to_fastest=median / fastest_seconds,
                    relative_median_absolute_deviation=relative_mad,
                    exclusion_reasons=reasons,
                )
                for (
                    engine,
                    measured,
                    measured_version,
                    installed_version,
                    median,
                    ratio,
                    relative_mad,
                    eligible,
                    reasons,
                ) in prepared
            ),
            key=lambda item: (
                not item.eligible,
                item.median_seconds if item.median_seconds is not None else math.inf,
                item.engine,
            ),
        )
    )
    eligible_candidates = tuple(item for item in candidates if item.eligible)
    fastest = eligible_candidates[0]
    native = next(
        (item for item in eligible_candidates if item.engine == _REFERENCE_ENGINE),
        None,
    )
    recommended = fastest
    reason = "fastest_stable_measured_engine"
    if native is not None and fastest.engine != _REFERENCE_ENGINE:
        assert native.median_seconds is not None and fastest.median_seconds is not None
        external_speedup = native.median_seconds / fastest.median_seconds
        if external_speedup <= 1.0 + tie_margin:
            recommended = native
            reason = "native_within_tie_margin"

    return SimulatorRecommendation(
        status="recommended",
        recommended_engine=recommended.engine,
        reason=reason,
        candidates=candidates,
        workload_fingerprint=str(matching_row.get("workload_fingerprint")),
        evidence_source=source,
        evidence_sha256=digest,
        environment_matches=True,
        workload_matches=True,
        native_tie_margin=tie_margin,
        limitations=(
            "The recommendation applies only to the exact measured workload and environment.",
            "It does not execute, route, or fall back to another simulator.",
            "Comparison evidence is non-release and not a scalability claim.",
        ),
    )


def _rows_by_ir_hash(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows_by_hash: dict[str, Mapping[str, Any]] = {}
    for raw_row in _sequence(report.get("rows"), "rows"):
        row = _mapping(raw_row, "row")
        value = row.get("ir_content_hash")
        if value is None:
            continue
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise SimulatorAdvisorEvidenceError(
                "row.ir_content_hash must be a lowercase SHA-256 digest"
            )
        if value in rows_by_hash:
            raise SimulatorAdvisorEvidenceError(
                f"duplicate circuit evidence for IR hash {value!r}"
            )
        rows_by_hash[value] = row
    return rows_by_hash


def _recommend_from_calibration(
    *,
    circuit_hash: str,
    engine_order: Sequence[str],
    versions: Mapping[str, str | None],
    outcome: CalibrationOutcome,
    repeats: int,
    tie_margin: float,
) -> SimulatorRecommendation:
    labels = {
        "flagquantum_native": "FlagQuantum",
        "qiskit_aer": "Qiskit Aer",
        "cirq_simulator": "Cirq Simulator",
        "pennylane_lightning_qubit": "PennyLane Lightning",
    }
    measured = {item.engine: item for item in outcome.measurements}
    native = measured.get(_REFERENCE_ENGINE)
    native_seconds = native.median_seconds if native is not None else None
    prepared: list[SimulatorCandidate] = []
    for engine in engine_order:
        measurement = measured.get(engine)
        reasons: list[str] = []
        if engine not in versions:
            reasons.append("engine_not_available")
        if measurement is None:
            reasons.append("not_calibrated")
        else:
            if measurement.error == "calibration_budget_exhausted":
                reasons.append("calibration_budget_exhausted")
            else:
                if measurement.error is not None:
                    reasons.append("calibration_error")
                if not measurement.correctness_passed:
                    reasons.append("correctness_not_passed")
                minimum_samples = max(2, math.ceil(repeats * 2 / 3))
                if len(measurement.samples_seconds) < minimum_samples:
                    reasons.append("insufficient_calibration_samples")
                if (
                    measurement.relative_median_absolute_deviation is not None
                    and measurement.relative_median_absolute_deviation > 0.25
                ):
                    reasons.append("measurement_not_stable")
        median = measurement.median_seconds if measurement is not None else None
        relative_mad = (
            measurement.relative_median_absolute_deviation
            if measurement is not None
            else None
        )
        prepared.append(
            SimulatorCandidate(
                engine=engine,
                label=labels.get(engine, engine),
                version=versions.get(engine),
                installed_version=versions.get(engine),
                available=engine in versions,
                eligible=not reasons,
                median_seconds=median,
                relative_to_flagquantum=(
                    median / native_seconds
                    if median is not None and native_seconds is not None
                    else None
                ),
                relative_to_fastest=None,
                relative_median_absolute_deviation=relative_mad,
                exclusion_reasons=tuple(reasons),
                calibration_error=(
                    measurement.error if measurement is not None else None
                ),
            )
        )
    eligible = tuple(
        item for item in prepared if item.eligible and item.median_seconds is not None
    )
    if not eligible:
        return SimulatorRecommendation(
            status="insufficient_evidence",
            recommended_engine=None,
            reason="live_calibration_no_eligible_engine",
            candidates=tuple(prepared),
            workload_fingerprint=circuit_hash,
            evidence_source="live_calibration",
            evidence_sha256=outcome.evidence_sha256,
            environment_matches=True,
            workload_matches=True,
            native_tie_margin=tie_margin,
            limitations=(
                "Live calibration found no correct, stable engine with at least two samples.",
                "The wall-clock budget is soft because an in-flight backend call is not interrupted.",
            ),
            circuit_ir_hash=circuit_hash,
            circuit_matches=True,
            evidence_level="live_calibration",
            confidence=0.0,
            calibration_elapsed_seconds=outcome.elapsed_seconds,
            calibration_cache_hit=outcome.cache_hit,
        )

    fastest_seconds = min(
        item.median_seconds for item in eligible if item.median_seconds
    )
    candidates = tuple(
        sorted(
            (
                replace(
                    item,
                    relative_to_fastest=(
                        item.median_seconds / fastest_seconds
                        if item.eligible and item.median_seconds is not None
                        else None
                    ),
                )
                for item in prepared
            ),
            key=lambda item: (
                not item.eligible,
                item.median_seconds if item.median_seconds is not None else math.inf,
                item.engine,
            ),
        )
    )
    eligible_candidates = tuple(item for item in candidates if item.eligible)
    fastest = eligible_candidates[0]
    native_candidate = next(
        (item for item in eligible_candidates if item.engine == _REFERENCE_ENGINE),
        None,
    )
    recommended = fastest
    reason = "fastest_live_calibrated_engine"
    if native_candidate is not None and fastest.engine != _REFERENCE_ENGINE:
        assert native_candidate.median_seconds is not None
        assert fastest.median_seconds is not None
        if native_candidate.median_seconds / fastest.median_seconds <= 1.0 + tie_margin:
            recommended = native_candidate
            reason = "native_within_live_calibration_tie_margin"
    sample_fraction = min(
        len(measured[item.engine].samples_seconds) / repeats
        for item in eligible_candidates
    )
    return SimulatorRecommendation(
        status="recommended",
        recommended_engine=recommended.engine,
        reason=reason,
        candidates=candidates,
        workload_fingerprint=circuit_hash,
        evidence_source="live_calibration",
        evidence_sha256=outcome.evidence_sha256,
        environment_matches=True,
        workload_matches=True,
        native_tie_margin=tie_margin,
        limitations=(
            "This decision uses end-to-end live timings for the exact supplied circuit.",
            "The wall-clock budget is soft because an in-flight backend call is not interrupted.",
            "The calibration cache is process-local and includes circuit, versions, and settings.",
        ),
        circuit_ir_hash=circuit_hash,
        circuit_matches=True,
        evidence_level="live_calibration",
        confidence=min(0.90, 0.90 * sample_fraction),
        calibration_elapsed_seconds=outcome.elapsed_seconds,
        calibration_cache_hit=outcome.cache_hit,
    )


def recommend(
    program: Any | None = None,
    *,
    n_wires: int | None = None,
    workload: str | None = None,
    layers: int | None = None,
    dtype: str | None = None,
    batch_size: int | None = None,
    seed: int | None = None,
    torch_threads: int = 1,
    environment: Mapping[str, Any] | None = None,
    available_engines: Mapping[str, str] | Collection[str] | None = None,
    native_tie_margin: float = 0.10,
    evidence: Mapping[str, Any] | str | Path | None = None,
    calibration_budget_seconds: float | None = None,
    calibration_warmup: int = 1,
    calibration_repeats: int = 5,
    calibration_use_cache: bool = True,
) -> SimulatorRecommendation:
    """Recommend an explicit simulator from exact measured evidence.

    Passing a FlagQuantum circuit or :class:`~flagquantum.core.CircuitIR` binds
    the decision to its canonical IR content hash. A recommendation is made
    only if that exact hash occurs in the workload manifest associated with a
    measured comparison row. Profile-only queries remain available for
    evidence exploration, but their result is explicitly not circuit-bound.

    The default evidence-only path never executes the circuit. A positive
    ``calibration_budget_seconds`` explicitly enables live local probes for an
    otherwise unmeasured circuit. Neither path changes ``fq.run`` routing or
    silently falls back to another simulator.

    Args:
        program: Optional FlagQuantum circuit or canonical CircuitIR. Do not
            combine this with profile selector arguments.
        n_wires: Measured width for a profile-only evidence query.
        workload: Optional benchmark workload identity for a profile query.
        layers: Optional benchmark layer count for a profile query.
        dtype: Optional statevector dtype for a profile query.
        batch_size: Optional workload batch size for a profile query.
        seed: Optional deterministic workload seed for a profile query.
        torch_threads: Planned Torch worker-thread count.
        environment: Optional explicit environment for reproducible queries.
        available_engines: Optional installed-engine names or version mapping.
        native_tie_margin: Prefer native execution when an external result is
            faster by no more than this fraction.
        evidence: Optional report mapping or JSON path. Circuit-bound custom
            evidence must include an ``ir_content_hash`` in its matching row.
        calibration_budget_seconds: For an unmeasured circuit, explicitly run
            available simulators under this soft wall-clock budget. Omit to
            keep recommendation strictly non-executing.
        calibration_warmup: Warmup calls per live-calibrated engine.
        calibration_repeats: Maximum retained timing calls per engine.
        calibration_use_cache: Reuse an exact process-local calibration keyed
            by circuit hash, engine versions, and calibration settings.

    Returns:
        A structured recommendation with its evidence level and limits.

    Raises:
        SimulatorAdvisorEvidenceError: If evidence is malformed or unsafe.

    Examples:
        >>> from flagquantum.ecosystem.simulators import recommend
        >>> decision = recommend(n_wires=22)
        >>> decision.status in {"recommended", "insufficient_evidence"}
        True
        >>> decision.circuit_matches is None
        True
    """

    selectors = {
        "workload": workload,
        "layers": layers,
        "dtype": dtype,
        "batch_size": batch_size,
        "seed": seed,
    }
    if not isinstance(calibration_use_cache, bool):
        raise TypeError("calibration_use_cache must be a bool")
    warmup_count = _non_negative_integer(calibration_warmup, "calibration_warmup")
    repeat_count = _positive_integer(calibration_repeats, "calibration_repeats")
    calibration_budget = (
        None
        if calibration_budget_seconds is None
        else _finite_non_negative(
            calibration_budget_seconds,
            "calibration_budget_seconds",
            positive=True,
        )
    )
    if program is None:
        if n_wires is None:
            raise TypeError("recommend requires program or n_wires")
        if calibration_budget is not None:
            raise ValueError("live calibration requires a program, not n_wires")
        decision = _recommend_profile(
            n_wires=n_wires,
            workload=(
                "hardware_efficient_statevector" if workload is None else workload
            ),
            layers=2 if layers is None else layers,
            dtype="complex128" if dtype is None else dtype,
            batch_size=1 if batch_size is None else batch_size,
            seed=7319 if seed is None else seed,
            torch_threads=torch_threads,
            environment=environment,
            available_engines=available_engines,
            native_tie_margin=native_tie_margin,
            evidence=evidence,
        )
        return replace(decision, evidence_level="profile")
    if n_wires is not None or any(value is not None for value in selectors.values()):
        raise ValueError(
            "program cannot be combined with n_wires or workload profile selectors"
        )

    circuit_ir = ensure_circuit_ir(program)
    circuit_hash = circuit_ir.content_hash
    report, _, _ = _load_evidence(evidence)
    _validate_report(report)
    matching_row = _rows_by_ir_hash(report).get(circuit_hash)
    manifest_source = WORKLOAD_MANIFEST_PATH if evidence is None else None
    manifest_digest = WORKLOAD_MANIFEST_SHA256 if evidence is None else None
    if matching_row is None:
        if calibration_budget is not None:
            engine_order = tuple(
                str(item)
                for item in _sequence(report.get("engine_order"), "engine_order")
            )
            versions = _engine_versions(available_engines)
            unknown = set(versions).difference(engine_order)
            if unknown:
                raise ValueError(
                    f"available_engines contains unknown engines: {sorted(unknown)}"
                )
            tie_margin = _finite_non_negative(
                native_tie_margin, "native_tie_margin", positive=False
            )
            if tie_margin >= 1.0:
                raise ValueError("native_tie_margin must be less than one")
            outcome = calibrate(
                circuit_ir,
                engines=tuple(engine for engine in engine_order if engine in versions),
                versions=versions,
                warmup=warmup_count,
                repeats=repeat_count,
                budget_seconds=calibration_budget,
                use_cache=calibration_use_cache,
            )
            return _recommend_from_calibration(
                circuit_hash=circuit_hash,
                engine_order=engine_order,
                versions=versions,
                outcome=outcome,
                repeats=repeat_count,
                tie_margin=tie_margin,
            )
        profile_decision = _recommend_profile(
            n_wires=circuit_ir.n_wires,
            workload="hardware_efficient_statevector",
            layers=2,
            dtype="complex128",
            batch_size=1,
            seed=7319,
            torch_threads=torch_threads,
            environment=environment,
            available_engines=available_engines,
            native_tie_margin=native_tie_margin,
            evidence=evidence,
        )
        candidates = tuple(
            replace(
                candidate,
                eligible=False,
                exclusion_reasons=tuple(
                    dict.fromkeys(
                        (*candidate.exclusion_reasons, "circuit_not_measured")
                    )
                ),
            )
            for candidate in profile_decision.candidates
        )
        return replace(
            profile_decision,
            status="insufficient_evidence",
            recommended_engine=None,
            reason="circuit_not_measured",
            candidates=candidates,
            workload_fingerprint=None,
            workload_matches=False,
            limitations=(
                "The exact canonical CircuitIR hash is absent from the measured workload manifest.",
                "Matching width or gate count alone is not sufficient evidence.",
                "The advisor does not execute, route, or fall back to another simulator.",
            ),
            circuit_ir_hash=circuit_hash,
            circuit_matches=False,
            workload_manifest_source=manifest_source,
            workload_manifest_sha256=manifest_digest,
            evidence_level="none",
            confidence=0.0,
        )

    measured_workload = _mapping(matching_row.get("workload"), "row.workload")
    measured_name = measured_workload.get("name")
    measured_dtype = measured_workload.get("dtype")
    measured_seed = measured_workload.get("seed")
    if not isinstance(measured_name, str) or not measured_name:
        raise SimulatorAdvisorEvidenceError(
            "row.workload.name must be a non-empty string"
        )
    if not isinstance(measured_dtype, str) or not measured_dtype:
        raise SimulatorAdvisorEvidenceError(
            "row.workload.dtype must be a non-empty string"
        )
    if not isinstance(measured_seed, int) or isinstance(measured_seed, bool):
        raise SimulatorAdvisorEvidenceError("row.workload.seed must be an integer")
    measured_width = _positive_integer(measured_workload.get("n_wires"), "row.n_wires")
    measured_layers = _positive_integer(measured_workload.get("layers"), "row.layers")
    measured_batch_size = _positive_integer(
        measured_workload.get("batch_size"), "row.batch_size"
    )
    measured_gate_count = _positive_integer(
        measured_workload.get("gate_count"), "row.gate_count"
    )
    circuit_batch_size = circuit_ir.shape[0] if len(circuit_ir.shape) > 1 else 1
    if (
        measured_width != circuit_ir.n_wires
        or measured_dtype != circuit_ir.dtype
        or measured_gate_count != len(circuit_ir.instructions)
        or measured_batch_size != circuit_batch_size
    ):
        raise SimulatorAdvisorEvidenceError(
            "row workload metadata is inconsistent with the matched circuit IR"
        )
    decision = _recommend_profile(
        n_wires=measured_width,
        workload=measured_name,
        layers=measured_layers,
        dtype=measured_dtype,
        batch_size=measured_batch_size,
        seed=measured_seed,
        torch_threads=torch_threads,
        environment=environment,
        available_engines=available_engines,
        native_tie_margin=native_tie_margin,
        evidence=evidence,
    )
    return replace(
        decision,
        circuit_ir_hash=circuit_hash,
        circuit_matches=True,
        workload_manifest_source=manifest_source,
        workload_manifest_sha256=manifest_digest,
        evidence_level="exact",
        confidence=1.0,
        limitations=(
            *decision.limitations,
            "The recommendation is bound to the supplied circuit's canonical IR hash.",
        ),
    )


__all__ = (
    "EvidenceLevel",
    "RecommendationStatus",
    "SimulatorAdvisorEvidenceError",
    "SimulatorCandidate",
    "SimulatorRecommendation",
    "recommend",
)
