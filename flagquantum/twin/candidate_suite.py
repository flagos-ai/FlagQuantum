"""Prospective candidate comparisons across a fixed circuit suite."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

from ..remote.qpu import DeploymentResult
from .candidate import (
    TwinCandidateDecision,
    TwinCandidateEvaluation,
    TwinCandidateTrial,
    prepare_candidate_trial,
)
from .candidate_submission import TwinCandidateSubmission
from .model import QPUDigitalTwin
from .submission import _require_fields

_SUITE_SCHEMA = "flagquantum.twin_candidate_suite.v1"
_EVALUATION_SCHEMA = "flagquantum.twin_candidate_suite_evaluation.v1"


def _identity(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _digest(value: str, name: str) -> str:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True)
class TwinCandidateSuiteEvaluation:
    """Simultaneous comparison of two Twins across fixed circuits."""

    suite_identity: str
    evaluations: tuple[TwinCandidateEvaluation, ...]
    confidence_level: float
    decision: TwinCandidateDecision
    schema: str = _EVALUATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _EVALUATION_SCHEMA:
            raise ValueError("unsupported Twin candidate-suite evaluation schema")
        _digest(self.suite_identity, "suite_identity")
        evaluations = tuple(self.evaluations)
        if len(evaluations) < 2 or any(
            not isinstance(item, TwinCandidateEvaluation) for item in evaluations
        ):
            raise ValueError(
                "candidate-suite evaluation requires at least two evaluations"
            )
        trial_identities = tuple(item.trial_identity for item in evaluations)
        task_ids = tuple(item.hardware_report.task_id for item in evaluations)
        if len(trial_identities) != len(set(trial_identities)):
            raise ValueError("candidate-suite evaluations must cover unique trials")
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("candidate-suite evaluations require distinct QPU tasks")
        incumbent_identities = {
            item.incumbent_snapshot_identity for item in evaluations
        }
        candidate_identities = {
            item.candidate_snapshot_identity for item in evaluations
        }
        if len(incumbent_identities) != 1 or len(candidate_identities) != 1:
            raise ValueError("candidate-suite evaluations must compare the same Twins")
        confidence = float(self.confidence_level)
        if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
            raise ValueError("confidence_level must be finite and in (0, 1)")
        per_trial_confidence = 1.0 - (1.0 - confidence) / len(evaluations)
        if any(
            not math.isclose(item.confidence_level, per_trial_confidence, abs_tol=1e-12)
            for item in evaluations
        ):
            raise ValueError(
                "candidate-suite evaluations require simultaneous confidence"
            )
        if self.decision != self._expected_decision():
            raise ValueError("candidate-suite decision does not match its interval")
        object.__setattr__(self, "evaluations", evaluations)
        object.__setattr__(self, "confidence_level", confidence)

    @property
    def circuit_count(self) -> int:
        return len(self.evaluations)

    @property
    def total_shots(self) -> int:
        return sum(item.hardware_report.validation.shots for item in self.evaluations)

    @property
    def mean_incumbent_hardware_total_variation(self) -> float:
        return fmean(
            item.incumbent_hardware_total_variation for item in self.evaluations
        )

    @property
    def mean_candidate_hardware_total_variation(self) -> float:
        return fmean(
            item.candidate_hardware_total_variation for item in self.evaluations
        )

    @property
    def mean_ideal_hardware_total_variation(self) -> float:
        return fmean(item.ideal_hardware_total_variation for item in self.evaluations)

    @property
    def mean_incumbent_qpu_agreement(self) -> float:
        return 1.0 - self.mean_incumbent_hardware_total_variation

    @property
    def mean_candidate_qpu_agreement(self) -> float:
        return 1.0 - self.mean_candidate_hardware_total_variation

    @property
    def mean_ideal_qpu_agreement(self) -> float:
        return 1.0 - self.mean_ideal_hardware_total_variation

    @property
    def mean_candidate_improvement(self) -> float:
        return fmean(item.candidate_improvement for item in self.evaluations)

    @property
    def candidate_improvement_error_radius(self) -> float:
        return min(
            1.0,
            fmean(item.candidate_improvement_error_radius for item in self.evaluations),
        )

    @property
    def candidate_improvement_lower_bound(self) -> float:
        return max(
            -1.0,
            self.mean_candidate_improvement - self.candidate_improvement_error_radius,
        )

    @property
    def candidate_improvement_upper_bound(self) -> float:
        return min(
            1.0,
            self.mean_candidate_improvement + self.candidate_improvement_error_radius,
        )

    def _expected_decision(self) -> TwinCandidateDecision:
        if self.candidate_improvement_lower_bound > 0.0:
            return "improved"
        if self.candidate_improvement_upper_bound < 0.0:
            return "degraded"
        return "inconclusive"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "suite_identity": self.suite_identity,
            "evaluations": [item.to_dict() for item in self.evaluations],
            "confidence_level": self.confidence_level,
            "decision": self.decision,
            "circuit_count": self.circuit_count,
            "total_shots": self.total_shots,
            "mean_incumbent_hardware_total_variation": (
                self.mean_incumbent_hardware_total_variation
            ),
            "mean_candidate_hardware_total_variation": (
                self.mean_candidate_hardware_total_variation
            ),
            "mean_ideal_hardware_total_variation": (
                self.mean_ideal_hardware_total_variation
            ),
            "mean_candidate_improvement": self.mean_candidate_improvement,
            "candidate_improvement_error_radius": (
                self.candidate_improvement_error_radius
            ),
            "candidate_improvement_lower_bound": (
                self.candidate_improvement_lower_bound
            ),
            "candidate_improvement_upper_bound": (
                self.candidate_improvement_upper_bound
            ),
        }


@dataclass(frozen=True)
class TwinCandidateSuite:
    """Two frozen Twins compared over distinct circuits on one mapping."""

    trials: tuple[TwinCandidateTrial, ...]
    schema: str = _SUITE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _SUITE_SCHEMA:
            raise ValueError("unsupported Twin candidate-suite schema")
        trials = tuple(self.trials)
        if len(trials) < 2 or any(
            not isinstance(trial, TwinCandidateTrial) for trial in trials
        ):
            raise ValueError("candidate suite requires at least two trials")
        incumbent_identities = {trial.incumbent_snapshot.identity for trial in trials}
        candidate_identities = {trial.candidate_snapshot.identity for trial in trials}
        circuit_identities = {
            trial.incumbent_prediction.circuit_identity for trial in trials
        }
        names = {trial.experiment.name for trial in trials}
        if len(incumbent_identities) != 1 or len(candidate_identities) != 1:
            raise ValueError("candidate suite must compare the same two Twins")
        if len(circuit_identities) != len(trials):
            raise ValueError("candidate suite requires distinct circuits")
        if len(names) != len(trials):
            raise ValueError("candidate suite requires distinct experiment names")
        object.__setattr__(self, "trials", trials)

    @property
    def identity(self) -> str:
        return _identity(self.to_dict())

    @property
    def circuit_count(self) -> int:
        return len(self.trials)

    @property
    def total_shots(self) -> int:
        return sum(trial.experiment.shots for trial in self.trials)

    def validate_results(
        self,
        submissions: Sequence[TwinCandidateSubmission],
        results: Sequence[DeploymentResult],
        *,
        circuits: Sequence[Any],
        confidence_level: float = 0.95,
    ) -> TwinCandidateSuiteEvaluation:
        """Validate distinct QPU results and compare the two Twins simultaneously.

        This method never submits, polls, retries, or promotes a model.
        """

        frozen_submissions = tuple(submissions)
        frozen_results = tuple(results)
        frozen_circuits = tuple(circuits)
        expected_count = len(self.trials)
        if not (
            len(frozen_submissions)
            == len(frozen_results)
            == len(frozen_circuits)
            == expected_count
        ):
            raise ValueError(
                "submissions, results, and circuits must match every suite trial"
            )
        confidence = float(confidence_level)
        if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
            raise ValueError("confidence_level must be finite and in (0, 1)")
        per_trial_confidence = 1.0 - (1.0 - confidence) / expected_count
        evaluations: list[TwinCandidateEvaluation] = []
        for trial, submission, result, circuit in zip(
            self.trials,
            frozen_submissions,
            frozen_results,
            frozen_circuits,
            strict=True,
        ):
            if not isinstance(submission, TwinCandidateSubmission):
                raise TypeError(
                    "submissions must contain TwinCandidateSubmission objects"
                )
            if submission.trial != trial:
                raise ValueError("candidate submission does not match its suite trial")
            evaluations.append(
                submission.validate_result(
                    result,
                    circuit=circuit,
                    confidence_level=per_trial_confidence,
                )
            )
        mean_improvement = fmean(item.candidate_improvement for item in evaluations)
        error_radius = min(
            1.0,
            fmean(item.candidate_improvement_error_radius for item in evaluations),
        )
        decision: TwinCandidateDecision = "inconclusive"
        if mean_improvement - error_radius > 0.0:
            decision = "improved"
        elif mean_improvement + error_radius < 0.0:
            decision = "degraded"
        return TwinCandidateSuiteEvaluation(
            suite_identity=self.identity,
            evaluations=tuple(evaluations),
            confidence_level=confidence,
            decision=decision,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "trials": [trial.to_dict() for trial in self.trials],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TwinCandidateSuite:
        """Restore a suite from its strict version-1 serialized form."""

        if not isinstance(payload, Mapping):
            raise TypeError("Twin candidate suite must be a mapping")
        _require_fields(payload, {"schema", "trials"}, name="Twin candidate suite")
        trials = payload["trials"]
        if not isinstance(trials, list) or any(
            not isinstance(trial, Mapping) for trial in trials
        ):
            raise ValueError("Twin candidate suite trials must be JSON objects")
        try:
            return cls(
                schema=str(payload["schema"]),
                trials=tuple(TwinCandidateTrial.from_dict(trial) for trial in trials),
            )
        except (TypeError, ValueError) as error:
            raise ValueError("Invalid Twin candidate suite") from error


def prepare_candidate_suite(
    incumbent: QPUDigitalTwin,
    candidate: QPUDigitalTwin,
    circuits: Sequence[Any],
    *,
    name: str,
    shots: int,
) -> TwinCandidateSuite:
    """Freeze candidate comparisons for distinct circuits on one mapping.

    Preparing a suite is offline. Every hardware submission remains an explicit
    ``trial.experiment.submit(provider)`` call made by the application.
    """

    frozen_circuits = tuple(circuits)
    if len(frozen_circuits) < 2:
        raise ValueError("candidate suite requires at least two circuits")
    base_name = str(name).strip()
    if not base_name:
        raise ValueError("name must be non-empty")
    return TwinCandidateSuite(
        trials=tuple(
            prepare_candidate_trial(
                incumbent,
                candidate,
                circuit,
                name=f"{base_name}-{index:02d}",
                shots=shots,
            )
            for index, circuit in enumerate(frozen_circuits, start=1)
        )
    )


def load_candidate_suite(path: str | PathLike[str]) -> TwinCandidateSuite:
    """Load a candidate suite offline without provider contact."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot load Twin candidate suite from {source}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("Twin candidate suite file must contain a JSON object")
    return TwinCandidateSuite.from_dict(payload)


def dump_candidate_suite(
    suite: TwinCandidateSuite,
    path: str | PathLike[str],
) -> None:
    """Write a private, create-once candidate-suite file."""

    if not isinstance(suite, TwinCandidateSuite):
        raise TypeError("suite must be a TwinCandidateSuite")
    destination = Path(path)
    encoded = (
        json.dumps(
            suite.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )
    try:
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(encoded)
    except FileExistsError:
        try:
            existing = load_candidate_suite(destination)
        except ValueError as error:
            raise ValueError(
                f"Refusing to replace invalid Twin candidate suite at {destination}"
            ) from error
        if existing.identity != suite.identity:
            raise ValueError(
                f"Refusing to replace different Twin candidate suite at {destination}"
            )
    except OSError as error:
        raise ValueError(
            f"Cannot write Twin candidate suite to {destination}"
        ) from error


__all__ = (
    "dump_candidate_suite",
    "load_candidate_suite",
    "prepare_candidate_suite",
    "TwinCandidateSuite",
    "TwinCandidateSuiteEvaluation",
)
