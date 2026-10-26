"""Immutable, human-gated records for one QPU Twin evolution cycle."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from os import PathLike
from pathlib import Path
from typing import Any, Literal, cast

from ._atomic import write_once
from .candidate import TwinCandidateDecision
from .candidate_fit import TwinFittedCandidate
from .candidate_suite import TwinCandidateSuiteEvaluation
from .freshness import TwinFreshnessAssessment, TwinFreshnessStatus
from .model import QPUDigitalTwin
from .parameter_estimation import TwinParameterEstimate
from .probe_refresh import TwinProbeRefreshPlan

TwinEvolutionCycleStage = Literal[
    "prepared",
    "parameter_estimated",
    "candidate_fitted",
    "holdout_evaluated",
]
TwinEvolutionRecommendation = Literal[
    "retain_incumbent",
    "collect_more_evidence",
    "release_candidate",
]

_SCHEMA = "flagquantum.twin_evolution_cycle.v1"
_STAGES = {
    "prepared",
    "parameter_estimated",
    "candidate_fitted",
    "holdout_evaluated",
}
_RECOMMENDATIONS = {
    "retain_incumbent",
    "collect_more_evidence",
    "release_candidate",
}


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


def _freshness_identity(assessment: TwinFreshnessAssessment) -> str:
    drift = assessment.calibration_drift
    series = assessment.validation_series
    return _identity(
        {
            "status": assessment.status,
            "snapshot_identity": assessment.snapshot_identity,
            "evaluated_at": assessment.evaluated_at,
            "calibration_age_seconds": assessment.calibration_age_seconds,
            "validation_snapshot_age_seconds": (
                assessment.validation_snapshot_age_seconds
            ),
            "next_refresh_at": assessment.next_refresh_at,
            "calibration_drift": None if drift is None else drift.to_dict(),
            "validation_series": None if series is None else series.to_dict(),
            "reasons": list(assessment.reasons),
        }
    )


def _evaluation_identity(evaluation: TwinCandidateSuiteEvaluation) -> str:
    return _identity(evaluation.to_dict())


def _recommendation(
    stage: TwinEvolutionCycleStage,
    freshness_status: TwinFreshnessStatus,
    holdout_decision: TwinCandidateDecision | None,
) -> TwinEvolutionRecommendation:
    if stage == "holdout_evaluated":
        return cast(
            TwinEvolutionRecommendation,
            {
                "improved": "release_candidate",
                "degraded": "retain_incumbent",
                "inconclusive": "collect_more_evidence",
            }[cast(str, holdout_decision)],
        )
    if stage == "prepared" and freshness_status == "current_under_policy":
        return "retain_incumbent"
    return "collect_more_evidence"


@dataclass(frozen=True)
class TwinEvolutionCycle:
    """Append-only references for one prospective Twin refresh workflow.

    The record contains no credentials, provider operations, source evidence,
    routing authority, or release authority. ``release_candidate`` is only a
    recommendation; an explicit release API remains the sole release boundary.
    """

    incumbent_snapshot_identity: str
    freshness_identity: str
    freshness_status: TwinFreshnessStatus
    probe_plan_identity: str
    target: str
    physical_qubits: tuple[int, ...]
    stage: TwinEvolutionCycleStage = "prepared"
    parameter_estimate_identity: str | None = None
    probe_bank_identity: str | None = None
    candidate_snapshot_identity: str | None = None
    candidate_fit_identity: str | None = None
    holdout_evaluation_identity: str | None = None
    holdout_suite_identity: str | None = None
    holdout_decision: TwinCandidateDecision | None = None
    recommended_action: TwinEvolutionRecommendation = "collect_more_evidence"
    schema: str = _SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _SCHEMA:
            raise ValueError("unsupported Twin evolution-cycle schema")
        for name in (
            "incumbent_snapshot_identity",
            "freshness_identity",
            "probe_plan_identity",
        ):
            _digest(str(getattr(self, name)), name)
        if self.freshness_status not in {
            "current_under_policy",
            "refresh_due",
            "insufficient_evidence",
        }:
            raise ValueError("evolution cycle requires compatible freshness")
        if not self.target.strip() or ":" not in self.target:
            raise ValueError("target must use provider:backend form")
        physical_qubits = tuple(self.physical_qubits)
        if (
            not physical_qubits
            or len(physical_qubits) != len(set(physical_qubits))
            or any(type(qubit) is not int or qubit < 0 for qubit in physical_qubits)
        ):
            raise ValueError("physical_qubits must be unique non-negative integers")
        if self.stage not in _STAGES:
            raise ValueError("unsupported Twin evolution-cycle stage")
        if self.recommended_action not in _RECOMMENDATIONS:
            raise ValueError("unsupported Twin evolution recommendation")
        optional_digests = (
            "parameter_estimate_identity",
            "probe_bank_identity",
            "candidate_snapshot_identity",
            "candidate_fit_identity",
            "holdout_evaluation_identity",
            "holdout_suite_identity",
        )
        for name in optional_digests:
            value = getattr(self, name)
            if value is not None:
                _digest(value, name)

        stage_index = (
            "prepared",
            "parameter_estimated",
            "candidate_fitted",
            "holdout_evaluated",
        ).index(self.stage)
        groups = (
            (),
            ("parameter_estimate_identity", "probe_bank_identity"),
            ("candidate_snapshot_identity", "candidate_fit_identity"),
            (
                "holdout_evaluation_identity",
                "holdout_suite_identity",
                "holdout_decision",
            ),
        )
        for index, names in enumerate(groups[1:], start=1):
            present = tuple(getattr(self, name) is not None for name in names)
            if (index <= stage_index and not all(present)) or (
                index > stage_index and any(present)
            ):
                raise ValueError("evolution-cycle references do not match its stage")
        if (
            self.candidate_snapshot_identity is not None
            and self.candidate_snapshot_identity == self.incumbent_snapshot_identity
        ):
            raise ValueError("candidate snapshot must differ from the incumbent")
        expected = _recommendation(
            self.stage, self.freshness_status, self.holdout_decision
        )
        if self.recommended_action != expected:
            raise ValueError("recommendation does not match evolution-cycle evidence")
        object.__setattr__(self, "physical_qubits", physical_qubits)

    @property
    def identity(self) -> str:
        """Return the deterministic identity of this cycle state."""

        return _identity(self.to_dict())

    def with_parameter_estimate(
        self, estimate: TwinParameterEstimate
    ) -> TwinEvolutionCycle:
        """Append a bound parameter estimate and return a new cycle state."""

        if self.stage != "prepared":
            raise ValueError("parameter estimate can only follow the prepared stage")
        if not isinstance(estimate, TwinParameterEstimate):
            raise TypeError("estimate must be a TwinParameterEstimate")
        if (
            estimate.snapshot_identity != self.incumbent_snapshot_identity
            or estimate.target != self.target
            or estimate.physical_qubits != self.physical_qubits
        ):
            raise ValueError("parameter estimate does not match the evolution cycle")
        return replace(
            self,
            stage="parameter_estimated",
            parameter_estimate_identity=estimate.identity,
            probe_bank_identity=estimate.probe_bank_identity,
            recommended_action="collect_more_evidence",
        )

    def with_candidate(self, candidate: TwinFittedCandidate) -> TwinEvolutionCycle:
        """Append a fitted candidate reference and return a new cycle state."""

        if self.stage != "parameter_estimated":
            raise ValueError("candidate can only follow parameter estimation")
        if not isinstance(candidate, TwinFittedCandidate):
            raise TypeError("candidate must be a TwinFittedCandidate")
        fit = candidate.fit
        snapshot = candidate.snapshot
        if (
            fit.incumbent_snapshot_identity != self.incumbent_snapshot_identity
            or fit.parameter_estimate_identity != self.parameter_estimate_identity
            or fit.probe_bank_identity != self.probe_bank_identity
            or f"{snapshot.provider}:{snapshot.backend_name}" != self.target
            or snapshot.physical_qubits != self.physical_qubits
        ):
            raise ValueError("fitted candidate does not match the evolution cycle")
        return replace(
            self,
            stage="candidate_fitted",
            candidate_snapshot_identity=snapshot.identity,
            candidate_fit_identity=fit.identity,
            recommended_action="collect_more_evidence",
        )

    def with_holdout_evaluation(
        self, evaluation: TwinCandidateSuiteEvaluation
    ) -> TwinEvolutionCycle:
        """Append fixed holdout results and return a recommendation-only state."""

        if self.stage != "candidate_fitted":
            raise ValueError("holdout evaluation can only follow candidate fitting")
        if not isinstance(evaluation, TwinCandidateSuiteEvaluation):
            raise TypeError("evaluation must be a TwinCandidateSuiteEvaluation")
        if any(
            item.incumbent_snapshot_identity != self.incumbent_snapshot_identity
            or item.candidate_snapshot_identity != self.candidate_snapshot_identity
            for item in evaluation.evaluations
        ):
            raise ValueError("holdout evaluation does not match the evolution cycle")
        return replace(
            self,
            stage="holdout_evaluated",
            holdout_evaluation_identity=_evaluation_identity(evaluation),
            holdout_suite_identity=evaluation.suite_identity,
            holdout_decision=evaluation.decision,
            recommended_action=_recommendation(
                "holdout_evaluated", self.freshness_status, evaluation.decision
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical, reference-only JSON representation."""

        return {
            "schema": self.schema,
            "incumbent_snapshot_identity": self.incumbent_snapshot_identity,
            "freshness_identity": self.freshness_identity,
            "freshness_status": self.freshness_status,
            "probe_plan_identity": self.probe_plan_identity,
            "target": self.target,
            "physical_qubits": list(self.physical_qubits),
            "stage": self.stage,
            "parameter_estimate_identity": self.parameter_estimate_identity,
            "probe_bank_identity": self.probe_bank_identity,
            "candidate_snapshot_identity": self.candidate_snapshot_identity,
            "candidate_fit_identity": self.candidate_fit_identity,
            "holdout_evaluation_identity": self.holdout_evaluation_identity,
            "holdout_suite_identity": self.holdout_suite_identity,
            "holdout_decision": self.holdout_decision,
            "recommended_action": self.recommended_action,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TwinEvolutionCycle:
        """Restore a strict v1 evolution-cycle record."""

        expected = {
            "schema",
            "incumbent_snapshot_identity",
            "freshness_identity",
            "freshness_status",
            "probe_plan_identity",
            "target",
            "physical_qubits",
            "stage",
            "parameter_estimate_identity",
            "probe_bank_identity",
            "candidate_snapshot_identity",
            "candidate_fit_identity",
            "holdout_evaluation_identity",
            "holdout_suite_identity",
            "holdout_decision",
            "recommended_action",
        }
        if set(payload) != expected:
            raise ValueError("Twin evolution-cycle fields do not match the v1 schema")
        try:
            cycle = cls(
                schema=str(payload["schema"]),
                incumbent_snapshot_identity=str(payload["incumbent_snapshot_identity"]),
                freshness_identity=str(payload["freshness_identity"]),
                freshness_status=cast(TwinFreshnessStatus, payload["freshness_status"]),
                probe_plan_identity=str(payload["probe_plan_identity"]),
                target=str(payload["target"]),
                physical_qubits=tuple(int(item) for item in payload["physical_qubits"]),
                stage=cast(TwinEvolutionCycleStage, payload["stage"]),
                parameter_estimate_identity=_optional_text(
                    payload["parameter_estimate_identity"]
                ),
                probe_bank_identity=_optional_text(payload["probe_bank_identity"]),
                candidate_snapshot_identity=_optional_text(
                    payload["candidate_snapshot_identity"]
                ),
                candidate_fit_identity=_optional_text(
                    payload["candidate_fit_identity"]
                ),
                holdout_evaluation_identity=_optional_text(
                    payload["holdout_evaluation_identity"]
                ),
                holdout_suite_identity=_optional_text(
                    payload["holdout_suite_identity"]
                ),
                holdout_decision=cast(
                    TwinCandidateDecision | None, payload["holdout_decision"]
                ),
                recommended_action=cast(
                    TwinEvolutionRecommendation, payload["recommended_action"]
                ),
            )
        except (TypeError, ValueError, KeyError) as error:
            raise ValueError("Invalid Twin evolution cycle") from error
        if cycle.to_dict() != dict(payload):
            raise ValueError("Twin evolution cycle is not in canonical v1 form")
        return cycle


def _optional_text(value: Any) -> str | None:
    return None if value is None else str(value)


def prepare_evolution_cycle(
    *,
    incumbent: QPUDigitalTwin,
    freshness: TwinFreshnessAssessment,
    probe_plan: TwinProbeRefreshPlan,
) -> TwinEvolutionCycle:
    """Prepare an offline cycle bound to one incumbent and probe plan."""

    if not isinstance(incumbent, QPUDigitalTwin):
        raise TypeError("incumbent must be a QPUDigitalTwin")
    if not isinstance(freshness, TwinFreshnessAssessment):
        raise TypeError("freshness must be a TwinFreshnessAssessment")
    if not isinstance(probe_plan, TwinProbeRefreshPlan):
        raise TypeError("probe_plan must be a TwinProbeRefreshPlan")
    snapshot = incumbent.snapshot
    target = f"{snapshot.provider}:{snapshot.backend_name}"
    if freshness.status == "incompatible_history":
        raise ValueError("incompatible freshness cannot start an evolution cycle")
    if freshness.snapshot_identity != snapshot.identity:
        raise ValueError("freshness does not describe the incumbent Twin")
    if (
        probe_plan.candidate_snapshot_identity != snapshot.identity
        or probe_plan.target != target
        or probe_plan.physical_qubits != snapshot.physical_qubits
    ):
        raise ValueError("probe plan does not match the incumbent Twin")
    action: TwinEvolutionRecommendation = (
        "retain_incumbent"
        if freshness.status == "current_under_policy"
        else "collect_more_evidence"
    )
    return TwinEvolutionCycle(
        incumbent_snapshot_identity=snapshot.identity,
        freshness_identity=_freshness_identity(freshness),
        freshness_status=freshness.status,
        probe_plan_identity=probe_plan.identity,
        target=target,
        physical_qubits=snapshot.physical_qubits,
        recommended_action=action,
    )


def dump_evolution_cycle(cycle: TwinEvolutionCycle, path: str | PathLike[str]) -> None:
    """Save a cycle state once without replacing another artifact."""

    if not isinstance(cycle, TwinEvolutionCycle):
        raise TypeError("cycle must be a TwinEvolutionCycle")
    destination = Path(path)
    encoded = json.dumps(cycle.to_dict(), sort_keys=True, indent=2) + "\n"
    write_once(
        destination,
        encoded,
        label="Twin evolution cycle",
        matches=lambda existing: (
            load_evolution_cycle(existing).identity == cycle.identity
        ),
    )


def load_evolution_cycle(path: str | PathLike[str]) -> TwinEvolutionCycle:
    """Load and strictly validate a saved evolution-cycle state."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot load Twin evolution cycle from {source}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("Twin evolution cycle must be a JSON object")
    return TwinEvolutionCycle.from_dict(payload)


__all__ = (
    "dump_evolution_cycle",
    "load_evolution_cycle",
    "prepare_evolution_cycle",
    "TwinEvolutionCycle",
    "TwinEvolutionCycleStage",
    "TwinEvolutionRecommendation",
)
