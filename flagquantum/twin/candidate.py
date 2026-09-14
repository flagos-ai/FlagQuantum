"""Prospective same-result comparisons for incumbent and candidate QPU Twins."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, Mapping

from ..core.ir import ensure_circuit_ir
from ..remote.qpu import DeploymentResult, ProviderTaskHandle
from ._statistics import finite_shot_tv_radius
from .experiment import TwinExperiment, TwinHardwareReport
from .model import QPUDigitalTwin, TwinSnapshot
from .prediction import TwinPrediction
from .submission import (
    _experiment_from_dict,
    _prediction_from_dict,
    _require_fields,
)
from .validation import TwinValidationReport

_TRIAL_SCHEMA = "flagquantum.twin_candidate_trial.v1"
_EVALUATION_SCHEMA = "flagquantum.twin_candidate_evaluation.v1"

TwinCandidateDecision = Literal["improved", "degraded", "inconclusive"]


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


def _snapshot_from_dict(payload: Mapping[str, Any]) -> TwinSnapshot:
    _require_fields(
        payload,
        {
            "schema",
            "provider",
            "backend_name",
            "captured_at",
            "physical_qubits",
            "calibration_identity",
            "noise_model_identity",
        },
        name="Twin snapshot",
    )
    try:
        return TwinSnapshot(
            schema=str(payload["schema"]),
            provider=str(payload["provider"]),
            backend_name=str(payload["backend_name"]),
            captured_at=str(payload["captured_at"]),
            physical_qubits=tuple(payload["physical_qubits"]),
            calibration_identity=_digest(
                str(payload["calibration_identity"]), "calibration_identity"
            ),
            noise_model_identity=_digest(
                str(payload["noise_model_identity"]), "noise_model_identity"
            ),
        )
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid Twin snapshot") from error


@dataclass(frozen=True)
class TwinCandidateEvaluation:
    """Evidence-qualified comparison against one shared hardware result."""

    trial_identity: str
    hardware_report: TwinHardwareReport
    incumbent_validation: TwinValidationReport
    finite_shot_tv_radius: float
    candidate_improvement_error_radius: float
    confidence_level: float
    decision: TwinCandidateDecision
    schema: str = _EVALUATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _EVALUATION_SCHEMA:
            raise ValueError("unsupported Twin candidate-evaluation schema")
        _digest(self.trial_identity, "trial_identity")
        if not isinstance(self.hardware_report, TwinHardwareReport):
            raise TypeError("hardware_report must be a TwinHardwareReport")
        if not isinstance(self.incumbent_validation, TwinValidationReport):
            raise TypeError("incumbent_validation must be a TwinValidationReport")
        candidate = self.hardware_report.validation
        incumbent = self.incumbent_validation
        if (
            incumbent.circuit_identity != candidate.circuit_identity
            or incumbent.hardware_probabilities != candidate.hardware_probabilities
            or incumbent.shots != candidate.shots
        ):
            raise ValueError("candidate evaluations require one shared hardware result")
        if incumbent.snapshot_identity == candidate.snapshot_identity:
            raise ValueError("candidate evaluation requires distinct Twin snapshots")
        if not math.isclose(
            incumbent.ideal_hardware_total_variation,
            candidate.ideal_hardware_total_variation,
            abs_tol=1e-12,
        ):
            raise ValueError("candidate evaluations require one ideal baseline")
        confidence = float(self.confidence_level)
        if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
            raise ValueError("confidence_level must be finite and in (0, 1)")
        expected_radius = finite_shot_tv_radius(
            outcome_count=len(candidate.hardware_probabilities),
            shots=candidate.shots,
            confidence_level=confidence,
        )
        radius = float(self.finite_shot_tv_radius)
        if not math.isclose(radius, expected_radius, abs_tol=1e-12):
            raise ValueError("finite_shot_tv_radius does not match the hardware result")
        expected_error_radius = min(1.0, 2.0 * radius)
        error_radius = float(self.candidate_improvement_error_radius)
        if not math.isclose(error_radius, expected_error_radius, abs_tol=1e-12):
            raise ValueError("candidate improvement radius must equal two shot radii")
        if self.decision != self._expected_decision():
            raise ValueError(
                "candidate decision does not match its confidence interval"
            )
        object.__setattr__(self, "confidence_level", confidence)
        object.__setattr__(self, "finite_shot_tv_radius", radius)
        object.__setattr__(self, "candidate_improvement_error_radius", error_radius)

    @property
    def incumbent_snapshot_identity(self) -> str:
        return str(self.incumbent_validation.snapshot_identity)

    @property
    def candidate_snapshot_identity(self) -> str:
        return str(self.hardware_report.validation.snapshot_identity)

    @property
    def incumbent_hardware_total_variation(self) -> float:
        return float(self.incumbent_validation.twin_hardware_total_variation)

    @property
    def candidate_hardware_total_variation(self) -> float:
        return float(self.hardware_report.validation.twin_hardware_total_variation)

    @property
    def ideal_hardware_total_variation(self) -> float:
        return float(self.hardware_report.validation.ideal_hardware_total_variation)

    @property
    def candidate_improvement(self) -> float:
        return (
            self.incumbent_hardware_total_variation
            - self.candidate_hardware_total_variation
        )

    @property
    def candidate_improvement_lower_bound(self) -> float:
        return max(
            -1.0,
            self.candidate_improvement - self.candidate_improvement_error_radius,
        )

    @property
    def candidate_improvement_upper_bound(self) -> float:
        return min(
            1.0,
            self.candidate_improvement + self.candidate_improvement_error_radius,
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
            "trial_identity": self.trial_identity,
            "hardware_report": self.hardware_report.to_dict(),
            "incumbent_validation": self.incumbent_validation.to_dict(),
            "finite_shot_tv_radius": self.finite_shot_tv_radius,
            "candidate_improvement_error_radius": (
                self.candidate_improvement_error_radius
            ),
            "confidence_level": self.confidence_level,
            "decision": self.decision,
            "incumbent_snapshot_identity": self.incumbent_snapshot_identity,
            "candidate_snapshot_identity": self.candidate_snapshot_identity,
            "incumbent_hardware_total_variation": (
                self.incumbent_hardware_total_variation
            ),
            "candidate_hardware_total_variation": (
                self.candidate_hardware_total_variation
            ),
            "ideal_hardware_total_variation": self.ideal_hardware_total_variation,
            "candidate_improvement": self.candidate_improvement,
            "candidate_improvement_lower_bound": (
                self.candidate_improvement_lower_bound
            ),
            "candidate_improvement_upper_bound": (
                self.candidate_improvement_upper_bound
            ),
        }


@dataclass(frozen=True)
class TwinCandidateTrial:
    """Two predictions frozen before one candidate-bound hardware experiment."""

    incumbent_snapshot: TwinSnapshot
    candidate_snapshot: TwinSnapshot
    incumbent_prediction: TwinPrediction
    experiment: TwinExperiment
    schema: str = _TRIAL_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _TRIAL_SCHEMA:
            raise ValueError("unsupported Twin candidate-trial schema")
        if not isinstance(self.incumbent_snapshot, TwinSnapshot) or not isinstance(
            self.candidate_snapshot, TwinSnapshot
        ):
            raise TypeError("candidate trial snapshots must be TwinSnapshot records")
        if not isinstance(self.incumbent_prediction, TwinPrediction):
            raise TypeError("incumbent_prediction must be a TwinPrediction")
        if not isinstance(self.experiment, TwinExperiment):
            raise TypeError("experiment must be a TwinExperiment")
        incumbent = self.incumbent_snapshot
        candidate = self.candidate_snapshot
        if (
            incumbent.provider,
            incumbent.backend_name,
            incumbent.physical_qubits,
        ) != (
            candidate.provider,
            candidate.backend_name,
            candidate.physical_qubits,
        ):
            raise ValueError("candidate trial requires one target and mapping")
        if incumbent.provider != "quafu":
            raise ValueError("candidate hardware trials currently require Quafu")
        if incumbent.identity == candidate.identity:
            raise ValueError("candidate trial requires distinct Twin snapshots")
        if datetime.fromisoformat(candidate.captured_at) <= datetime.fromisoformat(
            incumbent.captured_at
        ):
            raise ValueError("candidate Twin snapshot must be later than incumbent")
        if self.incumbent_prediction.snapshot_identity != incumbent.identity:
            raise ValueError("incumbent prediction does not match its Twin snapshot")
        candidate_prediction = self.experiment.prediction
        if (
            self.experiment.snapshot_identity != candidate.identity
            or candidate_prediction.snapshot_identity != candidate.identity
        ):
            raise ValueError("candidate experiment does not match its Twin snapshot")
        if (
            self.experiment.backend_name != candidate.backend_name
            or self.experiment.target_qubits != candidate.physical_qubits
        ):
            raise ValueError("candidate experiment changes the target or mapping")
        if (
            self.incumbent_prediction.circuit_identity
            != candidate_prediction.circuit_identity
            or self.incumbent_prediction.n_wires != candidate_prediction.n_wires
        ):
            raise ValueError("candidate trial requires one fixed circuit")
        if (
            self.incumbent_prediction.ideal_probabilities
            != candidate_prediction.ideal_probabilities
        ):
            raise ValueError("candidate trial requires one ideal baseline")

    @property
    def identity(self) -> str:
        return _identity(self.to_dict())

    def validate_result(
        self,
        result: DeploymentResult,
        *,
        receipt: ProviderTaskHandle,
        circuit: Any,
        confidence_level: float = 0.95,
    ) -> TwinCandidateEvaluation:
        """Validate one result and compare both predictions against its counts.

        Args:
            result: Terminal provider result for the candidate experiment.
            receipt: Exact receipt returned by the explicit submission.
            circuit: Original FlagQuantum circuit frozen by the trial.
            confidence_level: Confidence used for the finite-shot interval.

        Returns:
            An evidence-qualified comparison using the shared hardware counts.

        Raises:
            RuntimeError: If any circuit, task, program, target, mapping, or
                result identity is not bound to the frozen trial.
        """

        ir = ensure_circuit_ir(circuit)
        if ir.content_hash != self.incumbent_prediction.circuit_identity:
            raise RuntimeError("circuit does not match the frozen candidate trial")
        hardware_report = self.experiment.validate_result(result, receipt=receipt)
        self.experiment.evidence_from_report(
            hardware_report,
            circuit=circuit,
            confidence_level=confidence_level,
        )
        incumbent_validation = self.incumbent_prediction.compare_counts(
            hardware_report.counts
        )
        radius = finite_shot_tv_radius(
            outcome_count=2**self.incumbent_prediction.n_wires,
            shots=incumbent_validation.shots,
            confidence_level=confidence_level,
        )
        improvement = (
            incumbent_validation.twin_hardware_total_variation
            - hardware_report.validation.twin_hardware_total_variation
        )
        improvement_radius = min(1.0, 2.0 * radius)
        lower = max(-1.0, improvement - improvement_radius)
        upper = min(1.0, improvement + improvement_radius)
        decision: TwinCandidateDecision = "inconclusive"
        if lower > 0.0:
            decision = "improved"
        elif upper < 0.0:
            decision = "degraded"
        return TwinCandidateEvaluation(
            trial_identity=self.identity,
            hardware_report=hardware_report,
            incumbent_validation=incumbent_validation,
            finite_shot_tv_radius=radius,
            candidate_improvement_error_radius=improvement_radius,
            confidence_level=confidence_level,
            decision=decision,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "incumbent_snapshot": self.incumbent_snapshot.to_dict(),
            "candidate_snapshot": self.candidate_snapshot.to_dict(),
            "incumbent_prediction": self.incumbent_prediction.to_dict(),
            "experiment": self.experiment.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TwinCandidateTrial:
        """Restore a trial from its strict version-1 serialized form."""

        if not isinstance(payload, Mapping):
            raise TypeError("Twin candidate trial must be a mapping")
        _require_fields(
            payload,
            {
                "schema",
                "incumbent_snapshot",
                "candidate_snapshot",
                "incumbent_prediction",
                "experiment",
            },
            name="Twin candidate trial",
        )
        incumbent_snapshot = payload["incumbent_snapshot"]
        candidate_snapshot = payload["candidate_snapshot"]
        incumbent_prediction = payload["incumbent_prediction"]
        experiment = payload["experiment"]
        if not all(
            isinstance(member, Mapping)
            for member in (
                incumbent_snapshot,
                candidate_snapshot,
                incumbent_prediction,
                experiment,
            )
        ):
            raise ValueError("Twin candidate trial members must be JSON objects")
        try:
            return cls(
                schema=str(payload["schema"]),
                incumbent_snapshot=_snapshot_from_dict(incumbent_snapshot),
                candidate_snapshot=_snapshot_from_dict(candidate_snapshot),
                incumbent_prediction=_prediction_from_dict(incumbent_prediction),
                experiment=_experiment_from_dict(experiment),
            )
        except (TypeError, ValueError) as error:
            raise ValueError("Invalid Twin candidate trial") from error


def prepare_candidate_trial(
    incumbent: QPUDigitalTwin,
    candidate: QPUDigitalTwin,
    circuit: Any,
    *,
    name: str,
    shots: int,
) -> TwinCandidateTrial:
    """Freeze two predictions and one canonical candidate experiment.

    Preparing a trial is offline. Hardware submission remains the explicit
    ``trial.experiment.submit(provider)`` operation.

    Args:
        incumbent: Currently retained frozen QPU Twin.
        candidate: Later frozen Twin for the same target and mapping.
        circuit: One FlagQuantum circuit to predict before hardware execution.
        name: Provider-visible experiment name.
        shots: Positive Quafu shot count divisible by 1,024.

    Returns:
        A frozen comparison trial containing both predictions and one canonical
        hardware experiment.

    Examples:
        trial = fq.twin.prepare_candidate_trial(
            incumbent,
            candidate,
            circuit,
            name="candidate-bell",
            shots=1024,
        )
        # Only this later, explicit operation submits hardware work:
        receipt = trial.experiment.submit(provider)
    """

    if not isinstance(incumbent, QPUDigitalTwin) or not isinstance(
        candidate, QPUDigitalTwin
    ):
        raise TypeError("incumbent and candidate must be QPUDigitalTwin objects")
    return TwinCandidateTrial(
        incumbent_snapshot=incumbent.snapshot,
        candidate_snapshot=candidate.snapshot,
        incumbent_prediction=incumbent.predict(circuit),
        experiment=TwinExperiment.prepare(
            candidate,
            circuit,
            name=name,
            shots=shots,
        ),
    )


__all__ = (
    "prepare_candidate_trial",
    "TwinCandidateDecision",
    "TwinCandidateEvaluation",
    "TwinCandidateTrial",
)
