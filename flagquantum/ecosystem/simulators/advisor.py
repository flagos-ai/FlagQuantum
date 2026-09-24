"""Evidence-scoped recommendations for explicit local simulator selection."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import operator
import platform
from collections.abc import Collection, Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Literal

import torch

from ._evidence import SOURCE_PATH, SOURCE_SHA256, bundled_report

RecommendationStatus = Literal["recommended", "insufficient_evidence"]

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


def recommend(
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
    """Recommend an explicit simulator from matching measured evidence.

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

    Examples:
        >>> from flagquantum.ecosystem.simulators import recommend
        >>> decision = recommend(n_wires=22)
        >>> decision.status in {"recommended", "insufficient_evidence"}
        True
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


__all__ = (
    "RecommendationStatus",
    "SimulatorAdvisorEvidenceError",
    "SimulatorCandidate",
    "SimulatorRecommendation",
    "recommend",
)
