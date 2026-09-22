"""Longitudinal validation history for fixed regional Twin suites."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import Any

from ._atomic import write_once
from .region_model import TwinRegionModel
from .region_suite import TwinRegionSuiteEvaluation
from .series import TwinValidationSeries

_HISTORY_SCHEMA = "flagquantum.twin_region_validation_history.v1"
_HISTORY_FIELDS = {
    "schema",
    "provider",
    "backend_name",
    "physical_qubits",
    "circuit_identities",
    "directed_couplers",
    "snapshot_identities",
    "region_identities",
    "captured_at",
    "observation_count",
    "mean_twin_qpu_agreements",
    "mean_ideal_qpu_agreements",
    "mean_qpu_repeatabilities",
    "simultaneous_finite_shot_tv_radii",
    "simultaneous_tv_error_bounds",
    "confidence_levels",
    "task_counts",
    "total_shots",
    "evaluations",
}
_EVALUATION_FIELDS = {
    "schema",
    "suite_identity",
    "region_identity",
    "snapshot_identity",
    "physical_qubits",
    "validation_series",
    "directed_couplers",
    "maximum_circuit_depth",
    "confidence_level",
    "circuit_count",
    "task_count",
    "total_shots",
    "mean_twin_qpu_agreement",
    "mean_ideal_qpu_agreement",
    "mean_qpu_repeatability",
    "simultaneous_finite_shot_tv_radius",
    "simultaneous_tv_error_bound",
}


def _canonical(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _evaluation_from_dict(payload: Mapping[str, Any]) -> TwinRegionSuiteEvaluation:
    if not isinstance(payload, Mapping):
        raise TypeError("Twin region-suite evaluation must be a mapping")
    actual = set(payload)
    if actual != _EVALUATION_FIELDS:
        raise ValueError(
            "Twin region-suite evaluation fields do not match the v1 schema: "
            f"missing={sorted(_EVALUATION_FIELDS - actual)}, "
            f"unexpected={sorted(actual - _EVALUATION_FIELDS)}"
        )
    series_payloads = payload["validation_series"]
    if not isinstance(series_payloads, list) or any(
        not isinstance(item, Mapping) for item in series_payloads
    ):
        raise ValueError("Twin region-suite series must be JSON objects")
    evaluation = TwinRegionSuiteEvaluation(
        schema=str(payload["schema"]),
        suite_identity=str(payload["suite_identity"]),
        region_identity=str(payload["region_identity"]),
        snapshot_identity=str(payload["snapshot_identity"]),
        physical_qubits=tuple(payload["physical_qubits"]),
        validation_series=tuple(
            TwinValidationSeries.from_dict(item) for item in series_payloads
        ),
        directed_couplers=tuple(tuple(item) for item in payload["directed_couplers"]),
        maximum_circuit_depth=int(payload["maximum_circuit_depth"]),
        confidence_level=float(payload["confidence_level"]),
    )
    if _canonical(evaluation.to_dict()) != _canonical(payload):
        raise ValueError("Twin region-suite evaluation is not in canonical v1 form")
    return evaluation


@dataclass(frozen=True)
class TwinRegionValidationHistory:
    """Fixed regional circuit-suite evidence across calibration states.

    The history reports observations only. It does not infer a trust window,
    schedule hardware work, update a Twin, or promote a model.

    Examples:
        history = fq.twin.build_region_validation_history(
            [
                (reference_region_twin, reference_evaluation),
                (current_region_twin, current_evaluation),
            ]
        )
        print(history.mean_twin_qpu_agreements)
    """

    provider: str
    backend_name: str
    physical_qubits: tuple[int, ...]
    circuit_identities: tuple[str, ...]
    directed_couplers: tuple[tuple[int, int], ...]
    snapshot_identities: tuple[str, ...]
    region_identities: tuple[str, ...]
    captured_at: tuple[str, ...]
    evaluations: tuple[TwinRegionSuiteEvaluation, ...]
    schema: str = _HISTORY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _HISTORY_SCHEMA:
            raise ValueError("unsupported Twin region-validation history schema")
        if not self.provider.strip() or not self.backend_name.strip():
            raise ValueError(
                "region-validation history requires a provider and backend"
            )
        qubits = tuple(self.physical_qubits)
        if (
            not qubits
            or len(qubits) != len(set(qubits))
            or any(type(qubit) is not int or qubit < 0 for qubit in qubits)
        ):
            raise ValueError(
                "region-validation physical qubits must be unique "
                "non-negative integers"
            )
        circuits = tuple(self.circuit_identities)
        if len(circuits) < 2 or len(circuits) != len(set(circuits)):
            raise ValueError(
                "region-validation history requires distinct fixed circuits"
            )
        for identity in circuits:
            _require_digest(identity, "circuit_identity")
        couplers = tuple(self.directed_couplers)
        physical = set(qubits)
        if len(couplers) != len(set(couplers)) or any(
            source == target or source not in physical or target not in physical
            for source, target in couplers
        ):
            raise ValueError(
                "region-validation topology must contain unique directed "
                "couplers on the fixed mapping"
            )

        snapshots = tuple(self.snapshot_identities)
        regions = tuple(self.region_identities)
        timestamps = tuple(self.captured_at)
        evaluations = tuple(self.evaluations)
        if len(evaluations) < 2:
            raise ValueError(
                "region-validation history requires at least two observations"
            )
        if not (len(snapshots) == len(regions) == len(timestamps) == len(evaluations)):
            raise ValueError("region-validation observations must have aligned fields")
        if len(snapshots) != len(set(snapshots)):
            raise ValueError("region-validation snapshots must be unique")
        if len(regions) != len(set(regions)):
            raise ValueError("region-validation region models must be unique")
        for identity in (*snapshots, *regions):
            _require_digest(identity, "observation identity")
        if any(not isinstance(item, TwinRegionSuiteEvaluation) for item in evaluations):
            raise TypeError(
                "region-validation history requires TwinRegionSuiteEvaluation records"
            )
        parsed = tuple(datetime.fromisoformat(value) for value in timestamps)
        if any(right <= left for left, right in zip(parsed, parsed[1:], strict=False)):
            raise ValueError(
                "region-validation snapshots must have strictly increasing "
                "captured_at values"
            )

        exercised_couplers = evaluations[0].directed_couplers
        maximum_depth = evaluations[0].maximum_circuit_depth
        reports: list[str] = []
        for index, evaluation in enumerate(evaluations):
            if evaluation.snapshot_identity != snapshots[index]:
                raise ValueError(
                    "region-suite evaluation does not match its Twin snapshot"
                )
            if evaluation.region_identity != regions[index]:
                raise ValueError(
                    "region-suite evaluation does not match its regional Twin"
                )
            if evaluation.physical_qubits != qubits:
                raise ValueError(
                    "region-validation history requires one physical mapping"
                )
            if (
                tuple(item.circuit_identity for item in evaluation.validation_series)
                != circuits
            ):
                raise ValueError(
                    "region-validation history requires one ordered circuit suite"
                )
            if evaluation.directed_couplers != exercised_couplers or (
                evaluation.maximum_circuit_depth != maximum_depth
            ):
                raise ValueError(
                    "region-validation history requires one suite structure"
                )
            if any(item not in couplers for item in evaluation.directed_couplers):
                raise ValueError(
                    "region-suite interactions must remain inside the fixed topology"
                )
            reports.extend(
                report
                for series in evaluation.validation_series
                for report in series.report_identities
            )
        if len(reports) != len(set(reports)):
            raise ValueError(
                "region-validation history requires distinct hardware reports"
            )

        object.__setattr__(self, "physical_qubits", qubits)
        object.__setattr__(self, "circuit_identities", circuits)
        object.__setattr__(self, "directed_couplers", couplers)
        object.__setattr__(self, "snapshot_identities", snapshots)
        object.__setattr__(self, "region_identities", regions)
        object.__setattr__(self, "captured_at", timestamps)
        object.__setattr__(self, "evaluations", evaluations)

    @property
    def observation_count(self) -> int:
        return len(self.evaluations)

    @property
    def mean_twin_qpu_agreements(self) -> tuple[float, ...]:
        return tuple(item.mean_twin_qpu_agreement for item in self.evaluations)

    @property
    def mean_ideal_qpu_agreements(self) -> tuple[float, ...]:
        return tuple(item.mean_ideal_qpu_agreement for item in self.evaluations)

    @property
    def mean_qpu_repeatabilities(self) -> tuple[float, ...]:
        return tuple(item.mean_qpu_repeatability for item in self.evaluations)

    @property
    def simultaneous_finite_shot_tv_radii(self) -> tuple[float, ...]:
        return tuple(
            item.simultaneous_finite_shot_tv_radius for item in self.evaluations
        )

    @property
    def simultaneous_tv_error_bounds(self) -> tuple[float, ...]:
        return tuple(item.simultaneous_tv_error_bound for item in self.evaluations)

    @property
    def confidence_levels(self) -> tuple[float, ...]:
        return tuple(item.confidence_level for item in self.evaluations)

    @property
    def task_counts(self) -> tuple[int, ...]:
        return tuple(item.task_count for item in self.evaluations)

    @property
    def total_shots(self) -> tuple[int, ...]:
        return tuple(item.total_shots for item in self.evaluations)

    def append(
        self,
        region_twin: TwinRegionModel,
        evaluation: TwinRegionSuiteEvaluation,
    ) -> TwinRegionValidationHistory:
        """Return a new history with one later comparable observation."""

        _validate_observation(region_twin, evaluation)
        snapshot = region_twin.twin.snapshot
        if (
            snapshot.provider,
            snapshot.backend_name,
            snapshot.physical_qubits,
            region_twin.region.directed_couplers,
        ) != (
            self.provider,
            self.backend_name,
            self.physical_qubits,
            self.directed_couplers,
        ):
            raise ValueError(
                "appended region observation must use the same target, mapping, "
                "and topology"
            )
        return type(self)(
            provider=self.provider,
            backend_name=self.backend_name,
            physical_qubits=self.physical_qubits,
            circuit_identities=self.circuit_identities,
            directed_couplers=self.directed_couplers,
            snapshot_identities=(*self.snapshot_identities, snapshot.identity),
            region_identities=(*self.region_identities, region_twin.identity),
            captured_at=(*self.captured_at, snapshot.captured_at),
            evaluations=(*self.evaluations, evaluation),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "provider": self.provider,
            "backend_name": self.backend_name,
            "physical_qubits": list(self.physical_qubits),
            "circuit_identities": list(self.circuit_identities),
            "directed_couplers": [list(item) for item in self.directed_couplers],
            "snapshot_identities": list(self.snapshot_identities),
            "region_identities": list(self.region_identities),
            "captured_at": list(self.captured_at),
            "observation_count": self.observation_count,
            "mean_twin_qpu_agreements": list(self.mean_twin_qpu_agreements),
            "mean_ideal_qpu_agreements": list(self.mean_ideal_qpu_agreements),
            "mean_qpu_repeatabilities": list(self.mean_qpu_repeatabilities),
            "simultaneous_finite_shot_tv_radii": list(
                self.simultaneous_finite_shot_tv_radii
            ),
            "simultaneous_tv_error_bounds": list(self.simultaneous_tv_error_bounds),
            "confidence_levels": list(self.confidence_levels),
            "task_counts": list(self.task_counts),
            "total_shots": list(self.total_shots),
            "evaluations": [item.to_dict() for item in self.evaluations],
        }


def _require_digest(value: str, name: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _validate_observation(
    region_twin: TwinRegionModel,
    evaluation: TwinRegionSuiteEvaluation,
) -> None:
    if not isinstance(region_twin, TwinRegionModel):
        raise TypeError("region_twin must be a TwinRegionModel")
    if not isinstance(evaluation, TwinRegionSuiteEvaluation):
        raise TypeError("evaluation must be a TwinRegionSuiteEvaluation")
    snapshot = region_twin.twin.snapshot
    if evaluation.snapshot_identity != snapshot.identity:
        raise ValueError("region-suite evaluation does not match its Twin snapshot")
    if evaluation.region_identity != region_twin.identity:
        raise ValueError("region-suite evaluation does not match its regional Twin")
    if evaluation.physical_qubits != snapshot.physical_qubits:
        raise ValueError("region-suite evaluation does not match its physical mapping")
    topology = set(region_twin.region.directed_couplers)
    if any(item not in topology for item in evaluation.directed_couplers):
        raise ValueError("region-suite interactions fall outside the regional topology")


def build_region_validation_history(
    observations: Sequence[tuple[TwinRegionModel, TwinRegionSuiteEvaluation]],
) -> TwinRegionValidationHistory:
    """Build an offline history from chronological regional observations."""

    if isinstance(observations, (str, bytes)):
        raise TypeError("observations must pair regional Twins with evaluations")
    pairs = tuple(observations)
    if len(pairs) < 2:
        raise ValueError("region-validation history requires at least two observations")
    if any(not isinstance(pair, tuple) or len(pair) != 2 for pair in pairs):
        raise TypeError("observations must contain (regional Twin, evaluation) pairs")
    region_twins = tuple(pair[0] for pair in pairs)
    evaluations = tuple(pair[1] for pair in pairs)
    for region_twin, evaluation in zip(region_twins, evaluations, strict=True):
        _validate_observation(region_twin, evaluation)

    first = region_twins[0]
    snapshot = first.twin.snapshot
    target = (
        snapshot.provider,
        snapshot.backend_name,
        snapshot.physical_qubits,
        first.region.directed_couplers,
    )
    for region_twin in region_twins[1:]:
        current = region_twin.twin.snapshot
        if (
            current.provider,
            current.backend_name,
            current.physical_qubits,
            region_twin.region.directed_couplers,
        ) != target:
            raise ValueError(
                "region-validation history requires one target, mapping, and topology"
            )

    return TwinRegionValidationHistory(
        provider=snapshot.provider,
        backend_name=snapshot.backend_name,
        physical_qubits=snapshot.physical_qubits,
        circuit_identities=tuple(
            item.circuit_identity for item in evaluations[0].validation_series
        ),
        directed_couplers=first.region.directed_couplers,
        snapshot_identities=tuple(item.twin.snapshot.identity for item in region_twins),
        region_identities=tuple(item.identity for item in region_twins),
        captured_at=tuple(item.twin.snapshot.captured_at for item in region_twins),
        evaluations=evaluations,
    )


def load_region_validation_history(
    path: str | PathLike[str],
) -> TwinRegionValidationHistory:
    """Load a regional validation history without provider access."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Cannot load Twin region-validation history from {source}"
        ) from error
    if not isinstance(payload, Mapping):
        raise ValueError(
            "Twin region-validation history file must contain a JSON object"
        )
    actual = set(payload)
    if actual != _HISTORY_FIELDS:
        raise ValueError(
            "Twin region-validation history fields do not match the v1 schema: "
            f"missing={sorted(_HISTORY_FIELDS - actual)}, "
            f"unexpected={sorted(actual - _HISTORY_FIELDS)}"
        )
    evaluation_payloads = payload["evaluations"]
    if not isinstance(evaluation_payloads, list) or any(
        not isinstance(item, Mapping) for item in evaluation_payloads
    ):
        raise ValueError(
            "Twin region-validation history evaluations must be JSON objects"
        )
    try:
        history = TwinRegionValidationHistory(
            schema=str(payload["schema"]),
            provider=str(payload["provider"]),
            backend_name=str(payload["backend_name"]),
            physical_qubits=tuple(payload["physical_qubits"]),
            circuit_identities=tuple(payload["circuit_identities"]),
            directed_couplers=tuple(
                tuple(item) for item in payload["directed_couplers"]
            ),
            snapshot_identities=tuple(payload["snapshot_identities"]),
            region_identities=tuple(payload["region_identities"]),
            captured_at=tuple(payload["captured_at"]),
            evaluations=tuple(
                _evaluation_from_dict(item) for item in evaluation_payloads
            ),
        )
        if _canonical(history.to_dict()) != _canonical(payload):
            raise ValueError(
                "Twin region-validation history is not in canonical v1 form"
            )
        return history
    except (TypeError, ValueError, KeyError) as error:
        raise ValueError("Invalid Twin region-validation history") from error


def dump_region_validation_history(
    history: TwinRegionValidationHistory,
    path: str | PathLike[str],
) -> None:
    """Write one private history without replacing different content."""

    if not isinstance(history, TwinRegionValidationHistory):
        raise TypeError("history must be a TwinRegionValidationHistory")
    encoded = _canonical(history.to_dict()) + "\n"
    write_once(
        Path(path),
        encoded,
        label="Twin region-validation history",
        matches=lambda destination: (
            _canonical(load_region_validation_history(destination).to_dict())
            == _canonical(history.to_dict())
        ),
    )


__all__ = (
    "build_region_validation_history",
    "dump_region_validation_history",
    "load_region_validation_history",
    "TwinRegionValidationHistory",
)
