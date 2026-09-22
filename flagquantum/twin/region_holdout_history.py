"""Longitudinal history for fixed regional Twin holdout studies."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import Any

from ._atomic import write_once
from .region_holdout import (
    TwinRegionHoldoutEvaluation,
    _holdout_evaluation_from_dict,
    _report_identities,
)
from .region_model import TwinRegionModel

_HISTORY_SCHEMA = "flagquantum.twin_region_holdout_history.v1"
_HISTORY_FIELDS = {
    "schema",
    "provider",
    "backend_name",
    "physical_qubits",
    "directed_couplers",
    "reference_circuit_identities",
    "holdout_circuit_identities",
    "snapshot_identities",
    "region_identities",
    "study_identities",
    "captured_at",
    "observation_count",
    "reference_twin_qpu_agreements",
    "holdout_twin_qpu_agreements",
    "holdout_tv_error_increases",
    "reference_ideal_qpu_agreements",
    "holdout_ideal_qpu_agreements",
    "reference_qpu_repeatabilities",
    "holdout_qpu_repeatabilities",
    "simultaneous_finite_shot_tv_radii",
    "holdout_simultaneous_tv_error_bounds",
    "confidence_levels",
    "task_counts",
    "total_shots",
    "evaluations",
}


def _canonical(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _require_digest(value: str, name: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _circuit_identities(
    evaluation: TwinRegionHoldoutEvaluation,
    group: str,
) -> tuple[str, ...]:
    suite = getattr(evaluation, f"{group}_evaluation")
    return tuple(item.circuit_identity for item in suite.validation_series)


def _suite_shape(
    evaluation: TwinRegionHoldoutEvaluation,
    group: str,
) -> tuple[
    tuple[tuple[int, int], ...],
    int,
    tuple[int, ...],
    tuple[int, ...],
]:
    suite = getattr(evaluation, f"{group}_evaluation")
    return (
        suite.directed_couplers,
        suite.maximum_circuit_depth,
        tuple(item.repetitions for item in suite.validation_series),
        tuple(item.total_shots for item in suite.validation_series),
    )


@dataclass(frozen=True)
class TwinRegionHoldoutHistory:
    """Fixed reference and holdout evidence across calibration states.

    The history is observational. It does not update a Twin, infer a trust
    window, schedule hardware work, or authorize workload routing.

    Examples:
        history = fq.twin.build_region_holdout_history(
            [
                (reference_region_twin, reference_holdout_evaluation),
                (current_region_twin, current_holdout_evaluation),
            ]
        )
        print(history.holdout_twin_qpu_agreements)
    """

    provider: str
    backend_name: str
    physical_qubits: tuple[int, ...]
    directed_couplers: tuple[tuple[int, int], ...]
    reference_circuit_identities: tuple[str, ...]
    holdout_circuit_identities: tuple[str, ...]
    snapshot_identities: tuple[str, ...]
    region_identities: tuple[str, ...]
    study_identities: tuple[str, ...]
    captured_at: tuple[str, ...]
    evaluations: tuple[TwinRegionHoldoutEvaluation, ...]
    schema: str = _HISTORY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _HISTORY_SCHEMA:
            raise ValueError("unsupported Twin region holdout-history schema")
        if not self.provider.strip() or not self.backend_name.strip():
            raise ValueError("region holdout history requires a provider and backend")
        qubits = tuple(self.physical_qubits)
        if (
            not qubits
            or len(qubits) != len(set(qubits))
            or any(type(qubit) is not int or qubit < 0 for qubit in qubits)
        ):
            raise ValueError(
                "region holdout physical qubits must be unique non-negative integers"
            )
        couplers = tuple(self.directed_couplers)
        physical = set(qubits)
        if len(couplers) != len(set(couplers)) or any(
            source == target or source not in physical or target not in physical
            for source, target in couplers
        ):
            raise ValueError(
                "region holdout topology must contain unique directed couplers "
                "on the fixed mapping"
            )
        reference_circuits = tuple(self.reference_circuit_identities)
        holdout_circuits = tuple(self.holdout_circuit_identities)
        if (
            not reference_circuits
            or not holdout_circuits
            or len(reference_circuits) != len(set(reference_circuits))
            or len(holdout_circuits) != len(set(holdout_circuits))
            or set(reference_circuits) & set(holdout_circuits)
        ):
            raise ValueError(
                "region holdout history requires distinct disjoint circuit groups"
            )
        for identity in (*reference_circuits, *holdout_circuits):
            _require_digest(identity, "circuit_identity")

        snapshots = tuple(self.snapshot_identities)
        regions = tuple(self.region_identities)
        studies = tuple(self.study_identities)
        timestamps = tuple(self.captured_at)
        evaluations = tuple(self.evaluations)
        if len(evaluations) < 2:
            raise ValueError(
                "region holdout history requires at least two observations"
            )
        if not (
            len(snapshots)
            == len(regions)
            == len(studies)
            == len(timestamps)
            == len(evaluations)
        ):
            raise ValueError("region holdout observations must have aligned fields")
        for values, label in (
            (snapshots, "snapshots"),
            (regions, "region models"),
            (studies, "holdout studies"),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"region holdout {label} must be unique")
            for identity in values:
                _require_digest(identity, "observation identity")
        if any(
            not isinstance(item, TwinRegionHoldoutEvaluation) for item in evaluations
        ):
            raise TypeError(
                "region holdout history requires TwinRegionHoldoutEvaluation records"
            )
        parsed = tuple(datetime.fromisoformat(value) for value in timestamps)
        if any(right <= left for left, right in zip(parsed, parsed[1:], strict=False)):
            raise ValueError(
                "region holdout snapshots must have strictly increasing captured_at values"
            )

        reference_shape = _suite_shape(evaluations[0], "reference")
        holdout_shape = _suite_shape(evaluations[0], "holdout")
        confidence = evaluations[0].confidence_level
        task_count = evaluations[0].task_count
        total_shots = evaluations[0].total_shots
        reports: list[str] = []
        for index, evaluation in enumerate(evaluations):
            if evaluation.study_identity != studies[index]:
                raise ValueError("holdout evaluation does not match its study identity")
            for suite in (
                evaluation.reference_evaluation,
                evaluation.holdout_evaluation,
            ):
                if suite.snapshot_identity != snapshots[index]:
                    raise ValueError(
                        "holdout evaluation does not match its Twin snapshot"
                    )
                if suite.region_identity != regions[index]:
                    raise ValueError(
                        "holdout evaluation does not match its regional Twin"
                    )
                if suite.physical_qubits != qubits:
                    raise ValueError(
                        "region holdout history requires one physical mapping"
                    )
                if any(item not in couplers for item in suite.directed_couplers):
                    raise ValueError(
                        "holdout interactions must remain inside the fixed topology"
                    )
                reports.extend(_report_identities(suite))
            if _circuit_identities(evaluation, "reference") != reference_circuits:
                raise ValueError(
                    "region holdout history requires one ordered reference circuit suite"
                )
            if _circuit_identities(evaluation, "holdout") != holdout_circuits:
                raise ValueError(
                    "region holdout history requires one ordered holdout circuit suite"
                )
            if (
                _suite_shape(evaluation, "reference") != reference_shape
                or _suite_shape(evaluation, "holdout") != holdout_shape
            ):
                raise ValueError(
                    "region holdout history requires one experiment design"
                )
            if (
                not math.isclose(evaluation.confidence_level, confidence, abs_tol=1e-12)
                or evaluation.task_count != task_count
                or evaluation.total_shots != total_shots
            ):
                raise ValueError(
                    "region holdout history requires one experiment design"
                )
        if len(reports) != len(set(reports)):
            raise ValueError(
                "region holdout history requires distinct hardware reports"
            )

        object.__setattr__(self, "physical_qubits", qubits)
        object.__setattr__(self, "directed_couplers", couplers)
        object.__setattr__(self, "reference_circuit_identities", reference_circuits)
        object.__setattr__(self, "holdout_circuit_identities", holdout_circuits)
        object.__setattr__(self, "snapshot_identities", snapshots)
        object.__setattr__(self, "region_identities", regions)
        object.__setattr__(self, "study_identities", studies)
        object.__setattr__(self, "captured_at", timestamps)
        object.__setattr__(self, "evaluations", evaluations)

    @property
    def observation_count(self) -> int:
        return len(self.evaluations)

    @property
    def reference_twin_qpu_agreements(self) -> tuple[float, ...]:
        return tuple(item.reference_twin_qpu_agreement for item in self.evaluations)

    @property
    def holdout_twin_qpu_agreements(self) -> tuple[float, ...]:
        return tuple(item.holdout_twin_qpu_agreement for item in self.evaluations)

    @property
    def holdout_tv_error_increases(self) -> tuple[float, ...]:
        return tuple(item.holdout_twin_qpu_tv_increase for item in self.evaluations)

    @property
    def reference_ideal_qpu_agreements(self) -> tuple[float, ...]:
        return tuple(item.reference_ideal_qpu_agreement for item in self.evaluations)

    @property
    def holdout_ideal_qpu_agreements(self) -> tuple[float, ...]:
        return tuple(item.holdout_ideal_qpu_agreement for item in self.evaluations)

    @property
    def reference_qpu_repeatabilities(self) -> tuple[float, ...]:
        return tuple(item.reference_qpu_repeatability for item in self.evaluations)

    @property
    def holdout_qpu_repeatabilities(self) -> tuple[float, ...]:
        return tuple(item.holdout_qpu_repeatability for item in self.evaluations)

    @property
    def simultaneous_finite_shot_tv_radii(self) -> tuple[float, ...]:
        return tuple(
            item.simultaneous_finite_shot_tv_radius for item in self.evaluations
        )

    @property
    def holdout_simultaneous_tv_error_bounds(self) -> tuple[float, ...]:
        return tuple(
            item.holdout_simultaneous_tv_error_bound for item in self.evaluations
        )

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
        evaluation: TwinRegionHoldoutEvaluation,
    ) -> TwinRegionHoldoutHistory:
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
                "appended holdout observation must use the same target, mapping, "
                "and topology"
            )
        return type(self)(
            provider=self.provider,
            backend_name=self.backend_name,
            physical_qubits=self.physical_qubits,
            directed_couplers=self.directed_couplers,
            reference_circuit_identities=self.reference_circuit_identities,
            holdout_circuit_identities=self.holdout_circuit_identities,
            snapshot_identities=(*self.snapshot_identities, snapshot.identity),
            region_identities=(*self.region_identities, region_twin.identity),
            study_identities=(*self.study_identities, evaluation.study_identity),
            captured_at=(*self.captured_at, snapshot.captured_at),
            evaluations=(*self.evaluations, evaluation),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "provider": self.provider,
            "backend_name": self.backend_name,
            "physical_qubits": list(self.physical_qubits),
            "directed_couplers": [list(item) for item in self.directed_couplers],
            "reference_circuit_identities": list(self.reference_circuit_identities),
            "holdout_circuit_identities": list(self.holdout_circuit_identities),
            "snapshot_identities": list(self.snapshot_identities),
            "region_identities": list(self.region_identities),
            "study_identities": list(self.study_identities),
            "captured_at": list(self.captured_at),
            "observation_count": self.observation_count,
            "reference_twin_qpu_agreements": list(self.reference_twin_qpu_agreements),
            "holdout_twin_qpu_agreements": list(self.holdout_twin_qpu_agreements),
            "holdout_tv_error_increases": list(self.holdout_tv_error_increases),
            "reference_ideal_qpu_agreements": list(self.reference_ideal_qpu_agreements),
            "holdout_ideal_qpu_agreements": list(self.holdout_ideal_qpu_agreements),
            "reference_qpu_repeatabilities": list(self.reference_qpu_repeatabilities),
            "holdout_qpu_repeatabilities": list(self.holdout_qpu_repeatabilities),
            "simultaneous_finite_shot_tv_radii": list(
                self.simultaneous_finite_shot_tv_radii
            ),
            "holdout_simultaneous_tv_error_bounds": list(
                self.holdout_simultaneous_tv_error_bounds
            ),
            "confidence_levels": list(self.confidence_levels),
            "task_counts": list(self.task_counts),
            "total_shots": list(self.total_shots),
            "evaluations": [item.to_dict() for item in self.evaluations],
        }


def _validate_observation(
    region_twin: TwinRegionModel,
    evaluation: TwinRegionHoldoutEvaluation,
) -> None:
    if not isinstance(region_twin, TwinRegionModel):
        raise TypeError("region_twin must be a TwinRegionModel")
    if not isinstance(evaluation, TwinRegionHoldoutEvaluation):
        raise TypeError("evaluation must be a TwinRegionHoldoutEvaluation")
    snapshot = region_twin.twin.snapshot
    for suite in (evaluation.reference_evaluation, evaluation.holdout_evaluation):
        if suite.snapshot_identity != snapshot.identity:
            raise ValueError("holdout evaluation does not match its Twin snapshot")
        if suite.region_identity != region_twin.identity:
            raise ValueError("holdout evaluation does not match its regional Twin")
        if suite.physical_qubits != snapshot.physical_qubits:
            raise ValueError("holdout evaluation does not match its physical mapping")
        topology = set(region_twin.region.directed_couplers)
        if any(item not in topology for item in suite.directed_couplers):
            raise ValueError("holdout interactions fall outside the regional topology")


def build_region_holdout_history(
    observations: Sequence[tuple[TwinRegionModel, TwinRegionHoldoutEvaluation]],
) -> TwinRegionHoldoutHistory:
    """Build an offline history from chronological regional holdout results."""

    if isinstance(observations, (str, bytes)):
        raise TypeError("observations must pair regional Twins with evaluations")
    pairs = tuple(observations)
    if len(pairs) < 2:
        raise ValueError("region holdout history requires at least two observations")
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
                "region holdout history requires one target, mapping, and topology"
            )

    return TwinRegionHoldoutHistory(
        provider=snapshot.provider,
        backend_name=snapshot.backend_name,
        physical_qubits=snapshot.physical_qubits,
        directed_couplers=first.region.directed_couplers,
        reference_circuit_identities=_circuit_identities(evaluations[0], "reference"),
        holdout_circuit_identities=_circuit_identities(evaluations[0], "holdout"),
        snapshot_identities=tuple(item.twin.snapshot.identity for item in region_twins),
        region_identities=tuple(item.identity for item in region_twins),
        study_identities=tuple(item.study_identity for item in evaluations),
        captured_at=tuple(item.twin.snapshot.captured_at for item in region_twins),
        evaluations=evaluations,
    )


def load_region_holdout_history(
    path: str | PathLike[str],
) -> TwinRegionHoldoutHistory:
    """Load a regional holdout history without provider access."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Cannot load Twin region holdout history from {source}"
        ) from error
    if not isinstance(payload, Mapping):
        raise ValueError("Twin region holdout history file must contain a JSON object")
    actual = set(payload)
    if actual != _HISTORY_FIELDS:
        raise ValueError(
            "Twin region holdout history fields do not match the v1 schema: "
            f"missing={sorted(_HISTORY_FIELDS - actual)}, "
            f"unexpected={sorted(actual - _HISTORY_FIELDS)}"
        )
    evaluation_payloads = payload["evaluations"]
    if not isinstance(evaluation_payloads, list) or any(
        not isinstance(item, Mapping) for item in evaluation_payloads
    ):
        raise ValueError("Twin region holdout history evaluations must be JSON objects")
    try:
        history = TwinRegionHoldoutHistory(
            schema=str(payload["schema"]),
            provider=str(payload["provider"]),
            backend_name=str(payload["backend_name"]),
            physical_qubits=tuple(payload["physical_qubits"]),
            directed_couplers=tuple(
                tuple(item) for item in payload["directed_couplers"]
            ),
            reference_circuit_identities=tuple(payload["reference_circuit_identities"]),
            holdout_circuit_identities=tuple(payload["holdout_circuit_identities"]),
            snapshot_identities=tuple(payload["snapshot_identities"]),
            region_identities=tuple(payload["region_identities"]),
            study_identities=tuple(payload["study_identities"]),
            captured_at=tuple(payload["captured_at"]),
            evaluations=tuple(
                _holdout_evaluation_from_dict(item) for item in evaluation_payloads
            ),
        )
        if _canonical(history.to_dict()) != _canonical(payload):
            raise ValueError("Twin region holdout history is not in canonical v1 form")
        return history
    except (TypeError, ValueError, KeyError) as error:
        raise ValueError("Invalid Twin region holdout history") from error


def dump_region_holdout_history(
    history: TwinRegionHoldoutHistory,
    path: str | PathLike[str],
) -> None:
    """Write one private history without replacing different content."""

    if not isinstance(history, TwinRegionHoldoutHistory):
        raise TypeError("history must be a TwinRegionHoldoutHistory")
    encoded = _canonical(history.to_dict()) + "\n"
    write_once(
        Path(path),
        encoded,
        label="Twin region holdout history",
        matches=lambda destination: (
            _canonical(load_region_holdout_history(destination).to_dict())
            == _canonical(history.to_dict())
        ),
    )


__all__ = (
    "build_region_holdout_history",
    "dump_region_holdout_history",
    "load_region_holdout_history",
    "TwinRegionHoldoutHistory",
)
