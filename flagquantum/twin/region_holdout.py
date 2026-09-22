"""Prospective holdout studies for connected regional Twin models."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

from ..remote.qpu import DeploymentResult
from ._atomic import write_once
from .region_model import TwinRegionModel
from .region_suite import (
    TwinRegionSuiteEvaluation,
    TwinRegionValidationSuite,
    _evaluation_from_dict,
    prepare_region_validation_suite,
)
from .submission import TwinSubmission

_STUDY_SCHEMA = "flagquantum.twin_region_holdout_study.v1"
_EVALUATION_SCHEMA = "flagquantum.twin_region_holdout_evaluation.v1"
_EVALUATION_FIELDS = {
    "schema",
    "study_identity",
    "reference_evaluation",
    "holdout_evaluation",
    "confidence_level",
    "reference_circuit_count",
    "holdout_circuit_count",
    "task_count",
    "total_shots",
    "reference_twin_qpu_agreement",
    "holdout_twin_qpu_agreement",
    "holdout_twin_qpu_tv_increase",
    "reference_ideal_qpu_agreement",
    "holdout_ideal_qpu_agreement",
    "reference_qpu_repeatability",
    "holdout_qpu_repeatability",
    "simultaneous_finite_shot_tv_radius",
    "holdout_simultaneous_tv_error_bound",
}


def _identity(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _digest(value: str, name: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _circuit_identities(
    suite: TwinRegionValidationSuite,
) -> tuple[str, ...]:
    return tuple(item.prediction.circuit_identity for item in suite.experiments)


def _report_identities(
    evaluation: TwinRegionSuiteEvaluation,
) -> tuple[str, ...]:
    return tuple(
        identity
        for series in evaluation.validation_series
        for identity in series.report_identities
    )


@dataclass(frozen=True)
class TwinRegionHoldoutEvaluation:
    """Simultaneous reference and holdout evidence for one regional Twin."""

    study_identity: str
    reference_evaluation: TwinRegionSuiteEvaluation
    holdout_evaluation: TwinRegionSuiteEvaluation
    confidence_level: float
    schema: str = _EVALUATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _EVALUATION_SCHEMA:
            raise ValueError("unsupported Twin region holdout-evaluation schema")
        _digest(self.study_identity, "study_identity")
        reference = self.reference_evaluation
        holdout = self.holdout_evaluation
        if not isinstance(reference, TwinRegionSuiteEvaluation) or not isinstance(
            holdout, TwinRegionSuiteEvaluation
        ):
            raise TypeError(
                "holdout evaluation requires two TwinRegionSuiteEvaluation records"
            )
        if (
            reference.region_identity,
            reference.snapshot_identity,
            reference.physical_qubits,
        ) != (
            holdout.region_identity,
            holdout.snapshot_identity,
            holdout.physical_qubits,
        ):
            raise ValueError(
                "reference and holdout evaluations must use one regional Twin"
            )
        reference_circuits = {
            item.circuit_identity for item in reference.validation_series
        }
        holdout_circuits = {item.circuit_identity for item in holdout.validation_series}
        if reference_circuits & holdout_circuits:
            raise ValueError("reference and holdout circuits must be disjoint")
        reports = (*_report_identities(reference), *_report_identities(holdout))
        if len(reports) != len(set(reports)):
            raise ValueError(
                "reference and holdout evaluations require distinct QPU tasks"
            )
        confidence = float(self.confidence_level)
        if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
            raise ValueError("confidence_level must be finite and in (0, 1)")
        per_group_confidence = 1.0 - (1.0 - confidence) / 2.0
        if not all(
            math.isclose(item.confidence_level, per_group_confidence, abs_tol=1e-12)
            for item in (reference, holdout)
        ):
            raise ValueError(
                "reference and holdout evaluations require simultaneous confidence"
            )
        object.__setattr__(self, "confidence_level", confidence)

    @property
    def identity(self) -> str:
        return _identity(self.to_dict())

    @property
    def reference_circuit_count(self) -> int:
        return self.reference_evaluation.circuit_count

    @property
    def holdout_circuit_count(self) -> int:
        return self.holdout_evaluation.circuit_count

    @property
    def task_count(self) -> int:
        return self.reference_evaluation.task_count + self.holdout_evaluation.task_count

    @property
    def total_shots(self) -> int:
        return (
            self.reference_evaluation.total_shots + self.holdout_evaluation.total_shots
        )

    @property
    def reference_twin_qpu_agreement(self) -> float:
        return self.reference_evaluation.mean_twin_qpu_agreement

    @property
    def holdout_twin_qpu_agreement(self) -> float:
        return self.holdout_evaluation.mean_twin_qpu_agreement

    @property
    def holdout_twin_qpu_tv_increase(self) -> float:
        """Return positive values when holdout mean TV error is larger."""

        return self.reference_twin_qpu_agreement - self.holdout_twin_qpu_agreement

    @property
    def reference_ideal_qpu_agreement(self) -> float:
        return self.reference_evaluation.mean_ideal_qpu_agreement

    @property
    def holdout_ideal_qpu_agreement(self) -> float:
        return self.holdout_evaluation.mean_ideal_qpu_agreement

    @property
    def reference_qpu_repeatability(self) -> float:
        return self.reference_evaluation.mean_qpu_repeatability

    @property
    def holdout_qpu_repeatability(self) -> float:
        return self.holdout_evaluation.mean_qpu_repeatability

    @property
    def simultaneous_finite_shot_tv_radius(self) -> float:
        return max(
            self.reference_evaluation.simultaneous_finite_shot_tv_radius,
            self.holdout_evaluation.simultaneous_finite_shot_tv_radius,
        )

    @property
    def holdout_simultaneous_tv_error_bound(self) -> float:
        return self.holdout_evaluation.simultaneous_tv_error_bound

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "study_identity": self.study_identity,
            "reference_evaluation": self.reference_evaluation.to_dict(),
            "holdout_evaluation": self.holdout_evaluation.to_dict(),
            "confidence_level": self.confidence_level,
            "reference_circuit_count": self.reference_circuit_count,
            "holdout_circuit_count": self.holdout_circuit_count,
            "task_count": self.task_count,
            "total_shots": self.total_shots,
            "reference_twin_qpu_agreement": self.reference_twin_qpu_agreement,
            "holdout_twin_qpu_agreement": self.holdout_twin_qpu_agreement,
            "holdout_twin_qpu_tv_increase": self.holdout_twin_qpu_tv_increase,
            "reference_ideal_qpu_agreement": self.reference_ideal_qpu_agreement,
            "holdout_ideal_qpu_agreement": self.holdout_ideal_qpu_agreement,
            "reference_qpu_repeatability": self.reference_qpu_repeatability,
            "holdout_qpu_repeatability": self.holdout_qpu_repeatability,
            "simultaneous_finite_shot_tv_radius": (
                self.simultaneous_finite_shot_tv_radius
            ),
            "holdout_simultaneous_tv_error_bound": (
                self.holdout_simultaneous_tv_error_bound
            ),
        }


@dataclass(frozen=True)
class TwinRegionHoldoutStudy:
    """Predeclared disjoint reference and holdout regional circuit suites."""

    reference_suite: TwinRegionValidationSuite
    holdout_suite: TwinRegionValidationSuite
    schema: str = _STUDY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _STUDY_SCHEMA:
            raise ValueError("unsupported Twin region holdout-study schema")
        reference = self.reference_suite
        holdout = self.holdout_suite
        if not isinstance(reference, TwinRegionValidationSuite) or not isinstance(
            holdout, TwinRegionValidationSuite
        ):
            raise TypeError(
                "holdout study requires two TwinRegionValidationSuite objects"
            )
        if (
            reference.region_identity,
            reference.snapshot_identity,
            reference.physical_qubits,
            reference.repetitions,
        ) != (
            holdout.region_identity,
            holdout.snapshot_identity,
            holdout.physical_qubits,
            holdout.repetitions,
        ):
            raise ValueError(
                "reference and holdout suites must use one regional Twin, mapping, "
                "and repetition count"
            )
        reference_circuits = set(_circuit_identities(reference))
        holdout_circuits = set(_circuit_identities(holdout))
        if reference_circuits & holdout_circuits:
            raise ValueError("reference and holdout circuits must be disjoint")
        names = tuple(
            item.name for item in (*reference.experiments, *holdout.experiments)
        )
        if len(names) != len(set(names)):
            raise ValueError("reference and holdout experiment names must be distinct")
        shots = {item.shots for item in (*reference.experiments, *holdout.experiments)}
        if len(shots) != 1:
            raise ValueError(
                "reference and holdout circuits must use one fixed shot count"
            )

    @property
    def identity(self) -> str:
        return _identity(self.to_dict())

    @property
    def reference_circuit_count(self) -> int:
        return self.reference_suite.circuit_count

    @property
    def holdout_circuit_count(self) -> int:
        return self.holdout_suite.circuit_count

    @property
    def planned_task_count(self) -> int:
        return (
            self.reference_suite.planned_task_count
            + self.holdout_suite.planned_task_count
        )

    @property
    def planned_shots(self) -> int:
        return self.reference_suite.planned_shots + self.holdout_suite.planned_shots

    def validate_results(
        self,
        reference_submissions: Sequence[TwinSubmission],
        reference_results: Sequence[DeploymentResult],
        holdout_submissions: Sequence[TwinSubmission],
        holdout_results: Sequence[DeploymentResult],
        *,
        reference_circuits: Sequence[Any],
        holdout_circuits: Sequence[Any],
        confidence_level: float = 0.95,
    ) -> TwinRegionHoldoutEvaluation:
        """Validate all predeclared tasks without provider operations."""

        all_submissions = (*tuple(reference_submissions), *tuple(holdout_submissions))
        if any(not isinstance(item, TwinSubmission) for item in all_submissions):
            raise TypeError("submissions must contain only TwinSubmission objects")
        task_ids = tuple(item.receipt.task_id for item in all_submissions)
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("reference and holdout suites require distinct QPU tasks")
        confidence = float(confidence_level)
        if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
            raise ValueError("confidence_level must be finite and in (0, 1)")
        per_group_confidence = 1.0 - (1.0 - confidence) / 2.0
        reference_evaluation = self.reference_suite.validate_results(
            reference_submissions,
            reference_results,
            circuits=reference_circuits,
            confidence_level=per_group_confidence,
        )
        holdout_evaluation = self.holdout_suite.validate_results(
            holdout_submissions,
            holdout_results,
            circuits=holdout_circuits,
            confidence_level=per_group_confidence,
        )
        return TwinRegionHoldoutEvaluation(
            study_identity=self.identity,
            reference_evaluation=reference_evaluation,
            holdout_evaluation=holdout_evaluation,
            confidence_level=confidence,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "reference_suite": self.reference_suite.to_dict(),
            "holdout_suite": self.holdout_suite.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TwinRegionHoldoutStudy:
        """Restore a study from its strict version-1 serialized form."""

        if not isinstance(payload, Mapping):
            raise TypeError("Twin region holdout study must be a mapping")
        expected = {"schema", "reference_suite", "holdout_suite"}
        actual = set(payload)
        if actual != expected:
            raise ValueError(
                "Twin region holdout-study fields do not match the v1 schema: "
                f"missing={sorted(expected - actual)}, "
                f"unexpected={sorted(actual - expected)}"
            )
        reference = payload["reference_suite"]
        holdout = payload["holdout_suite"]
        if not isinstance(reference, Mapping) or not isinstance(holdout, Mapping):
            raise ValueError("Twin region holdout suites must be JSON objects")
        try:
            return cls(
                schema=str(payload["schema"]),
                reference_suite=TwinRegionValidationSuite.from_dict(reference),
                holdout_suite=TwinRegionValidationSuite.from_dict(holdout),
            )
        except (TypeError, ValueError) as error:
            raise ValueError("Invalid Twin region holdout study") from error


def prepare_region_holdout_study(
    region_twin: TwinRegionModel,
    reference_circuits: Sequence[Any],
    holdout_circuits: Sequence[Any],
    *,
    physical_qubits: Sequence[int],
    name: str,
    shots: int,
    repetitions: int = 2,
) -> TwinRegionHoldoutStudy:
    """Freeze disjoint reference and holdout predictions before QPU work."""

    base_name = str(name).strip()
    if not base_name:
        raise ValueError("name must be non-empty")
    reference = prepare_region_validation_suite(
        region_twin,
        reference_circuits,
        physical_qubits=physical_qubits,
        name=f"{base_name}-reference",
        shots=shots,
        repetitions=repetitions,
    )
    holdout = prepare_region_validation_suite(
        region_twin,
        holdout_circuits,
        physical_qubits=physical_qubits,
        name=f"{base_name}-holdout",
        shots=shots,
        repetitions=repetitions,
    )
    return TwinRegionHoldoutStudy(
        reference_suite=reference,
        holdout_suite=holdout,
    )


def load_region_holdout_study(path: str | PathLike[str]) -> TwinRegionHoldoutStudy:
    """Load a frozen holdout study without contacting a provider."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Cannot load Twin region holdout study from {source}"
        ) from error
    if not isinstance(payload, Mapping):
        raise ValueError("Twin region holdout-study file must contain a JSON object")
    try:
        study = TwinRegionHoldoutStudy.from_dict(payload)
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid Twin region holdout study") from error
    if study.to_dict() != payload:
        raise ValueError("Twin region holdout study is not in canonical v1 form")
    return study


def load_region_holdout_evaluation(
    path: str | PathLike[str],
) -> TwinRegionHoldoutEvaluation:
    """Load a holdout evaluation and recompute every derived metric."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Cannot load Twin region holdout evaluation from {source}"
        ) from error
    if not isinstance(payload, Mapping):
        raise ValueError(
            "Twin region holdout-evaluation file must contain a JSON object"
        )
    actual = set(payload)
    if actual != _EVALUATION_FIELDS:
        raise ValueError(
            "Twin region holdout-evaluation fields do not match the v1 schema: "
            f"missing={sorted(_EVALUATION_FIELDS - actual)}, "
            f"unexpected={sorted(actual - _EVALUATION_FIELDS)}"
        )
    reference = payload["reference_evaluation"]
    holdout = payload["holdout_evaluation"]
    if not isinstance(reference, Mapping) or not isinstance(holdout, Mapping):
        raise ValueError("Twin region holdout evaluations must be JSON objects")
    try:
        evaluation = TwinRegionHoldoutEvaluation(
            schema=str(payload["schema"]),
            study_identity=str(payload["study_identity"]),
            reference_evaluation=_evaluation_from_dict(reference),
            holdout_evaluation=_evaluation_from_dict(holdout),
            confidence_level=float(payload["confidence_level"]),
        )
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid Twin region holdout evaluation") from error
    if evaluation.to_dict() != payload:
        raise ValueError("Twin region holdout evaluation is not in canonical v1 form")
    return evaluation


def dump_region_holdout_study(
    study: TwinRegionHoldoutStudy,
    path: str | PathLike[str],
) -> None:
    """Write one private study without replacing different content."""

    if not isinstance(study, TwinRegionHoldoutStudy):
        raise TypeError("study must be a TwinRegionHoldoutStudy")
    encoded = (
        json.dumps(
            study.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        + "\n"
    )
    write_once(
        Path(path),
        encoded,
        label="Twin region holdout study",
        matches=lambda destination: load_region_holdout_study(destination) == study,
    )


def dump_region_holdout_evaluation(
    evaluation: TwinRegionHoldoutEvaluation,
    path: str | PathLike[str],
) -> None:
    """Write one private evaluation without replacing different content."""

    if not isinstance(evaluation, TwinRegionHoldoutEvaluation):
        raise TypeError("evaluation must be a TwinRegionHoldoutEvaluation")
    encoded = (
        json.dumps(
            evaluation.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )
    write_once(
        Path(path),
        encoded,
        label="Twin region holdout evaluation",
        matches=lambda destination: (
            load_region_holdout_evaluation(destination) == evaluation
        ),
    )


__all__ = (
    "dump_region_holdout_evaluation",
    "dump_region_holdout_study",
    "load_region_holdout_evaluation",
    "load_region_holdout_study",
    "prepare_region_holdout_study",
    "TwinRegionHoldoutEvaluation",
    "TwinRegionHoldoutStudy",
)
