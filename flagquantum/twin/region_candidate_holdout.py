"""Prospective holdout gates for connected regional Twin candidates."""

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
from .candidate import TwinCandidateDecision
from .candidate_submission import TwinCandidateSubmission
from .candidate_suite import (
    TwinCandidateSuite,
    TwinCandidateSuiteEvaluation,
    prepare_candidate_suite,
)
from .region_model import TwinRegionModel

_STUDY_SCHEMA = "flagquantum.twin_region_candidate_holdout_study.v1"
_EVALUATION_SCHEMA = "flagquantum.twin_region_candidate_holdout_evaluation.v1"


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


def _circuit_identities(suite: TwinCandidateSuite) -> tuple[str, ...]:
    return tuple(trial.incumbent_prediction.circuit_identity for trial in suite.trials)


def _region_scope(model: TwinRegionModel) -> tuple[Any, ...]:
    region = model.region
    return (
        region.provider,
        region.backend_name,
        region.physical_qubits,
        region.directed_couplers,
        region.supported_operations,
        region.maximum_instruction_count,
        region.maximum_circuit_depth,
    )


@dataclass(frozen=True)
class TwinRegionCandidateHoldoutEvaluation:
    """Reference and holdout evidence for one regional Twin candidate."""

    study_identity: str
    reference_evaluation: TwinCandidateSuiteEvaluation
    holdout_evaluation: TwinCandidateSuiteEvaluation
    confidence_level: float
    decision: TwinCandidateDecision
    schema: str = _EVALUATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _EVALUATION_SCHEMA:
            raise ValueError(
                "unsupported Twin region candidate holdout-evaluation schema"
            )
        _digest(self.study_identity, "study_identity")
        reference = self.reference_evaluation
        holdout = self.holdout_evaluation
        if not isinstance(reference, TwinCandidateSuiteEvaluation) or not isinstance(
            holdout, TwinCandidateSuiteEvaluation
        ):
            raise TypeError(
                "regional candidate holdout evaluation requires two candidate "
                "suite evaluations"
            )
        reference_pairs = {
            (
                item.incumbent_snapshot_identity,
                item.candidate_snapshot_identity,
            )
            for item in reference.evaluations
        }
        holdout_pairs = {
            (
                item.incumbent_snapshot_identity,
                item.candidate_snapshot_identity,
            )
            for item in holdout.evaluations
        }
        if len(reference_pairs) != 1 or reference_pairs != holdout_pairs:
            raise ValueError(
                "reference and holdout evaluations must compare the same Twins"
            )
        reference_circuits = {
            item.hardware_report.validation.circuit_identity
            for item in reference.evaluations
        }
        holdout_circuits = {
            item.hardware_report.validation.circuit_identity
            for item in holdout.evaluations
        }
        if reference_circuits & holdout_circuits:
            raise ValueError("reference and holdout circuits must be disjoint")
        task_ids = tuple(
            item.hardware_report.task_id
            for item in (*reference.evaluations, *holdout.evaluations)
        )
        if len(task_ids) != len(set(task_ids)):
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
        if self.decision != self._expected_decision():
            raise ValueError(
                "regional candidate holdout decision does not match its groups"
            )
        object.__setattr__(self, "confidence_level", confidence)

    @property
    def identity(self) -> str:
        return _identity(self.to_dict())

    @property
    def circuit_count(self) -> int:
        return (
            self.reference_evaluation.circuit_count
            + self.holdout_evaluation.circuit_count
        )

    @property
    def task_count(self) -> int:
        return self.circuit_count

    @property
    def total_shots(self) -> int:
        return (
            self.reference_evaluation.total_shots + self.holdout_evaluation.total_shots
        )

    def _expected_decision(self) -> TwinCandidateDecision:
        if (
            self.reference_evaluation.decision == "degraded"
            or self.holdout_evaluation.decision == "degraded"
        ):
            return "degraded"
        if self.holdout_evaluation.decision == "improved":
            return "improved"
        return "inconclusive"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "study_identity": self.study_identity,
            "reference_evaluation": self.reference_evaluation.to_dict(),
            "holdout_evaluation": self.holdout_evaluation.to_dict(),
            "confidence_level": self.confidence_level,
            "decision": self.decision,
            "circuit_count": self.circuit_count,
            "task_count": self.task_count,
            "total_shots": self.total_shots,
        }


@dataclass(frozen=True)
class TwinRegionCandidateHoldoutStudy:
    """Predeclared reference and holdout suites for a regional candidate."""

    incumbent_region_identity: str
    candidate_region_identity: str
    physical_qubits: tuple[int, ...]
    reference_suite: TwinCandidateSuite
    holdout_suite: TwinCandidateSuite
    schema: str = _STUDY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _STUDY_SCHEMA:
            raise ValueError("unsupported Twin region candidate holdout-study schema")
        _digest(self.incumbent_region_identity, "incumbent_region_identity")
        _digest(self.candidate_region_identity, "candidate_region_identity")
        if self.incumbent_region_identity == self.candidate_region_identity:
            raise ValueError("regional candidate study requires distinct regions")
        if not self.physical_qubits:
            raise ValueError("regional candidate study requires physical qubits")
        if len(self.physical_qubits) != len(set(self.physical_qubits)):
            raise ValueError("regional candidate physical qubits must be unique")
        if not isinstance(self.reference_suite, TwinCandidateSuite) or not isinstance(
            self.holdout_suite, TwinCandidateSuite
        ):
            raise TypeError(
                "regional candidate study requires two TwinCandidateSuite objects"
            )
        reference = self.reference_suite
        holdout = self.holdout_suite
        reference_trial = reference.trials[0]
        holdout_trial = holdout.trials[0]
        if (
            reference_trial.incumbent_snapshot.identity,
            reference_trial.candidate_snapshot.identity,
        ) != (
            holdout_trial.incumbent_snapshot.identity,
            holdout_trial.candidate_snapshot.identity,
        ):
            raise ValueError(
                "reference and holdout suites must compare the same two Twins"
            )
        if reference_trial.experiment.target_qubits != self.physical_qubits:
            raise ValueError("candidate suites do not match the regional mapping")
        if set(_circuit_identities(reference)) & set(_circuit_identities(holdout)):
            raise ValueError("reference and holdout circuits must be disjoint")
        names = tuple(
            trial.experiment.name for trial in (*reference.trials, *holdout.trials)
        )
        if len(names) != len(set(names)):
            raise ValueError("reference and holdout experiment names must be distinct")
        shots = {
            trial.experiment.shots for trial in (*reference.trials, *holdout.trials)
        }
        if len(shots) != 1:
            raise ValueError(
                "reference and holdout circuits must use one fixed shot count"
            )
        object.__setattr__(self, "physical_qubits", tuple(self.physical_qubits))

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
        return self.reference_circuit_count + self.holdout_circuit_count

    @property
    def planned_shots(self) -> int:
        return self.reference_suite.total_shots + self.holdout_suite.total_shots

    def validate_results(
        self,
        reference_submissions: Sequence[TwinCandidateSubmission],
        reference_results: Sequence[DeploymentResult],
        holdout_submissions: Sequence[TwinCandidateSubmission],
        holdout_results: Sequence[DeploymentResult],
        *,
        reference_circuits: Sequence[Any],
        holdout_circuits: Sequence[Any],
        confidence_level: float = 0.95,
    ) -> TwinRegionCandidateHoldoutEvaluation:
        """Validate all predeclared tasks without provider operations."""

        all_submissions = (*tuple(reference_submissions), *tuple(holdout_submissions))
        if any(
            not isinstance(item, TwinCandidateSubmission) for item in all_submissions
        ):
            raise TypeError(
                "submissions must contain only TwinCandidateSubmission objects"
            )
        task_ids = tuple(item.receipt.task_id for item in all_submissions)
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("reference and holdout suites require distinct QPU tasks")
        confidence = float(confidence_level)
        if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
            raise ValueError("confidence_level must be finite and in (0, 1)")
        per_group_confidence = 1.0 - (1.0 - confidence) / 2.0
        reference = self.reference_suite.validate_results(
            reference_submissions,
            reference_results,
            circuits=reference_circuits,
            confidence_level=per_group_confidence,
        )
        holdout = self.holdout_suite.validate_results(
            holdout_submissions,
            holdout_results,
            circuits=holdout_circuits,
            confidence_level=per_group_confidence,
        )
        decision: TwinCandidateDecision = "inconclusive"
        if reference.decision == "degraded" or holdout.decision == "degraded":
            decision = "degraded"
        elif holdout.decision == "improved":
            decision = "improved"
        return TwinRegionCandidateHoldoutEvaluation(
            study_identity=self.identity,
            reference_evaluation=reference,
            holdout_evaluation=holdout,
            confidence_level=confidence,
            decision=decision,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "incumbent_region_identity": self.incumbent_region_identity,
            "candidate_region_identity": self.candidate_region_identity,
            "physical_qubits": list(self.physical_qubits),
            "reference_suite": self.reference_suite.to_dict(),
            "holdout_suite": self.holdout_suite.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TwinRegionCandidateHoldoutStudy:
        """Restore a study from its strict version-1 serialized form."""

        if not isinstance(payload, Mapping):
            raise TypeError("Twin region candidate holdout study must be a mapping")
        expected = {
            "schema",
            "incumbent_region_identity",
            "candidate_region_identity",
            "physical_qubits",
            "reference_suite",
            "holdout_suite",
        }
        actual = set(payload)
        if actual != expected:
            raise ValueError(
                "Twin region candidate holdout-study fields do not match the v1 "
                f"schema: missing={sorted(expected - actual)}, "
                f"unexpected={sorted(actual - expected)}"
            )
        reference = payload["reference_suite"]
        holdout = payload["holdout_suite"]
        if not isinstance(reference, Mapping) or not isinstance(holdout, Mapping):
            raise ValueError("regional candidate suites must be JSON objects")
        try:
            return cls(
                schema=str(payload["schema"]),
                incumbent_region_identity=str(payload["incumbent_region_identity"]),
                candidate_region_identity=str(payload["candidate_region_identity"]),
                physical_qubits=tuple(payload["physical_qubits"]),
                reference_suite=TwinCandidateSuite.from_dict(reference),
                holdout_suite=TwinCandidateSuite.from_dict(holdout),
            )
        except (TypeError, ValueError) as error:
            raise ValueError("Invalid Twin region candidate holdout study") from error


def prepare_region_candidate_holdout(
    incumbent: TwinRegionModel,
    candidate: TwinRegionModel,
    reference_circuits: Sequence[Any],
    holdout_circuits: Sequence[Any],
    *,
    physical_qubits: Sequence[int],
    name: str,
    shots: int,
) -> TwinRegionCandidateHoldoutStudy:
    """Freeze a regional candidate gate before any QPU submission.

    Examples:
        study = fq.twin.prepare_region_candidate_holdout(
            incumbent_region,
            candidate_region,
            reference_circuits,
            holdout_circuits,
            physical_qubits=(20, 27, 34),
            name="regional-candidate",
            shots=1024,
        )
    """

    if not isinstance(incumbent, TwinRegionModel) or not isinstance(
        candidate, TwinRegionModel
    ):
        raise TypeError("incumbent and candidate must be TwinRegionModel objects")
    if _region_scope(incumbent) != _region_scope(candidate):
        raise ValueError(
            "regional candidate models must share target, mapping, topology, "
            "operations, and structural limits"
        )
    mapping = tuple(int(qubit) for qubit in physical_qubits)
    if mapping != incumbent.physical_qubits:
        raise ValueError(
            "physical_qubits must exactly match the regional model wire order"
        )
    frozen_reference = tuple(reference_circuits)
    frozen_holdout = tuple(holdout_circuits)
    for circuit in (*frozen_reference, *frozen_holdout):
        incumbent.predict(circuit, physical_qubits=mapping)
        candidate.predict(circuit, physical_qubits=mapping)
    base_name = str(name).strip()
    if not base_name:
        raise ValueError("name must be non-empty")
    reference = prepare_candidate_suite(
        incumbent.twin,
        candidate.twin,
        frozen_reference,
        name=f"{base_name}-reference",
        shots=shots,
    )
    holdout = prepare_candidate_suite(
        incumbent.twin,
        candidate.twin,
        frozen_holdout,
        name=f"{base_name}-holdout",
        shots=shots,
    )
    return TwinRegionCandidateHoldoutStudy(
        incumbent_region_identity=incumbent.identity,
        candidate_region_identity=candidate.identity,
        physical_qubits=mapping,
        reference_suite=reference,
        holdout_suite=holdout,
    )


def load_region_candidate_holdout_study(
    path: str | PathLike[str],
) -> TwinRegionCandidateHoldoutStudy:
    """Load a frozen regional candidate study without provider contact."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Cannot load Twin region candidate holdout study from {source}"
        ) from error
    if not isinstance(payload, Mapping):
        raise ValueError(
            "Twin region candidate holdout-study file must contain a JSON object"
        )
    study = TwinRegionCandidateHoldoutStudy.from_dict(payload)
    if study.to_dict() != payload:
        raise ValueError(
            "Twin region candidate holdout study is not in canonical v1 form"
        )
    return study


def dump_region_candidate_holdout_study(
    study: TwinRegionCandidateHoldoutStudy,
    path: str | PathLike[str],
) -> None:
    """Write one private study without replacing different content."""

    if not isinstance(study, TwinRegionCandidateHoldoutStudy):
        raise TypeError("study must be a TwinRegionCandidateHoldoutStudy")
    encoded = (
        json.dumps(
            study.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        + "\n"
    )
    write_once(
        Path(path),
        encoded,
        label="Twin region candidate holdout study",
        matches=lambda destination: (
            load_region_candidate_holdout_study(destination) == study
        ),
    )


__all__ = (
    "dump_region_candidate_holdout_study",
    "load_region_candidate_holdout_study",
    "prepare_region_candidate_holdout",
    "TwinRegionCandidateHoldoutEvaluation",
    "TwinRegionCandidateHoldoutStudy",
)
