"""Private offline validation of anonymous provider-sandbox observation facts."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import Enum

from .deployment_canary_readiness import (
    CanaryReadinessReport,
    CanaryReadinessStatus,
)
from .deployment_rehearsal import OfflineRehearsalReport, RehearsalOutcome

_SHA256 = re.compile(r"[0-9a-f]{64}")


def _digest(value: dict[str, object]) -> str:
    encoded = json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_digest(value: str, label: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"sandbox observation {label} must be lowercase SHA-256")


def _require_nonnegative(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"sandbox observation {label} must be nonnegative")


class SandboxObservationState(str, Enum):
    NOT_ATTEMPTED = "not_attempted"
    INTENT_RECORDED = "intent_recorded"
    ATTEMPT_RECORDED = "attempt_recorded"
    ACCEPTANCE_CONFIRMED = "acceptance_confirmed"
    ACCEPTANCE_REJECTED = "acceptance_rejected"
    TERMINAL_RESULT_RECORDED = "terminal_result_recorded"
    OUTCOME_UNKNOWN = "outcome_unknown"


class SandboxObservationDecision(str, Enum):
    OBSERVATION_ACCEPTED = "observation_accepted"
    OBSERVATION_REJECTED = "observation_rejected"
    RECONCILIATION_REQUIRED = "reconciliation_required"
    OBSERVATION_INCOMPLETE = "observation_incomplete"


class SandboxLimit(str, Enum):
    QUEUE_DEPTH = "observed_queue_depth"
    QUOTA_UNITS = "observed_quota_units"
    COST_MICROUNITS = "observed_cost_microunits"
    RETENTION_DURATION = "observed_retention_duration_ns"


_INCOMPLETE_STATES = (SandboxObservationState.NOT_ATTEMPTED,)
_REJECTED_STATES = (
    SandboxObservationState.INTENT_RECORDED,
    SandboxObservationState.ATTEMPT_RECORDED,
    SandboxObservationState.ACCEPTANCE_REJECTED,
)
_ACCEPTED_STATES = (
    SandboxObservationState.INTENT_RECORDED,
    SandboxObservationState.ATTEMPT_RECORDED,
    SandboxObservationState.ACCEPTANCE_CONFIRMED,
    SandboxObservationState.TERMINAL_RESULT_RECORDED,
)
_UNKNOWN_STATES = (
    SandboxObservationState.INTENT_RECORDED,
    SandboxObservationState.ATTEMPT_RECORDED,
    SandboxObservationState.OUTCOME_UNKNOWN,
)


@dataclass(frozen=True)
class SandboxLimitResult:
    limit: SandboxLimit
    observed: int
    maximum: int
    passed: bool

    def __post_init__(self) -> None:
        if not isinstance(self.limit, SandboxLimit):
            raise ValueError("sandbox observation limit must use its closed enum")
        _require_nonnegative(self.observed, "observed limit value")
        _require_nonnegative(self.maximum, "maximum limit value")
        if not isinstance(self.passed, bool):
            raise ValueError("sandbox observation limit result must be boolean")
        if self.passed is not (self.observed <= self.maximum):
            raise ValueError("sandbox observation limit result does not match values")

    def to_dict(self) -> dict[str, object]:
        return {
            "limit": self.limit.value,
            "observed": self.observed,
            "maximum": self.maximum,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class SandboxObservationSnapshot:
    states: tuple[SandboxObservationState, ...]
    sandbox_target_identity: str
    request_identity: str
    idempotency_identity: str
    readiness_evidence_identity: str
    rehearsal_evidence_identity: str
    kill_switch_available_and_current: bool
    rollback_evidence_present: bool
    operator_acknowledgement_present: bool
    conformance_current: bool
    target_identity_matches: bool
    non_billable_synthetic_workload: bool
    observed_queue_depth: int
    observed_quota_units: int
    observed_cost_microunits: int
    observed_retention_duration_ns: int

    def __post_init__(self) -> None:
        states = tuple(self.states)
        if not states or any(
            not isinstance(item, SandboxObservationState) for item in states
        ):
            raise ValueError("sandbox observation states must use the closed enum")
        for name in (
            "sandbox_target_identity",
            "request_identity",
            "idempotency_identity",
            "readiness_evidence_identity",
            "rehearsal_evidence_identity",
        ):
            _require_digest(getattr(self, name), name.replace("_", " "))
        for name in (
            "kill_switch_available_and_current",
            "rollback_evidence_present",
            "operator_acknowledgement_present",
            "conformance_current",
            "target_identity_matches",
            "non_billable_synthetic_workload",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"sandbox observation {name} must be boolean")
        for name in (
            "observed_queue_depth",
            "observed_quota_units",
            "observed_cost_microunits",
            "observed_retention_duration_ns",
        ):
            _require_nonnegative(getattr(self, name), name.replace("_", " "))
        object.__setattr__(self, "states", states)

    def identity_dict(self) -> dict[str, object]:
        return {
            "states": [item.value for item in self.states],
            "sandbox_target_identity": self.sandbox_target_identity,
            "request_identity": self.request_identity,
            "idempotency_identity": self.idempotency_identity,
            "readiness_evidence_identity": self.readiness_evidence_identity,
            "rehearsal_evidence_identity": self.rehearsal_evidence_identity,
            "kill_switch_available_and_current": self.kill_switch_available_and_current,
            "rollback_evidence_present": self.rollback_evidence_present,
            "operator_acknowledgement_present": self.operator_acknowledgement_present,
            "conformance_current": self.conformance_current,
            "target_identity_matches": self.target_identity_matches,
            "non_billable_synthetic_workload": self.non_billable_synthetic_workload,
            "observed_queue_depth": self.observed_queue_depth,
            "observed_quota_units": self.observed_quota_units,
            "observed_cost_microunits": self.observed_cost_microunits,
            "observed_retention_duration_ns": self.observed_retention_duration_ns,
        }

    @property
    def observation_identity(self) -> str:
        return _digest(self.identity_dict())

    @property
    def encoded_size(self) -> int:
        return len(
            json.dumps(
                self.identity_dict(),
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        )


@dataclass(frozen=True)
class SandboxObservationPolicy:
    maximum_state_count: int
    maximum_input_bytes: int
    maximum_evidence_bytes: int
    maximum_queue_depth: int
    maximum_quota_units: int
    maximum_cost_microunits: int
    maximum_retention_duration_ns: int

    def __post_init__(self) -> None:
        for name in (
            "maximum_state_count",
            "maximum_input_bytes",
            "maximum_evidence_bytes",
            "maximum_queue_depth",
            "maximum_quota_units",
            "maximum_cost_microunits",
            "maximum_retention_duration_ns",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"sandbox observation policy {name} must be positive")
        if self.maximum_input_bytes < 1024 or self.maximum_evidence_bytes < 1024:
            raise ValueError(
                "sandbox observation input and evidence limits must be >= 1024 bytes"
            )

    def identity_dict(self) -> dict[str, object]:
        return {
            "maximum_state_count": self.maximum_state_count,
            "maximum_input_bytes": self.maximum_input_bytes,
            "maximum_evidence_bytes": self.maximum_evidence_bytes,
            "maximum_queue_depth": self.maximum_queue_depth,
            "maximum_quota_units": self.maximum_quota_units,
            "maximum_cost_microunits": self.maximum_cost_microunits,
            "maximum_retention_duration_ns": self.maximum_retention_duration_ns,
        }

    @property
    def policy_identity(self) -> str:
        return _digest(self.identity_dict())


@dataclass(frozen=True)
class SandboxObservationReport:
    decision: SandboxObservationDecision
    states: tuple[SandboxObservationState, ...]
    sandbox_target_identity: str
    request_identity: str
    idempotency_identity: str
    readiness_evidence_identity: str
    rehearsal_evidence_identity: str
    observation_identity: str
    policy_identity: str
    limit_results: tuple[SandboxLimitResult, ...]
    evidence_identity: str

    def __post_init__(self) -> None:
        if not isinstance(self.decision, SandboxObservationDecision):
            raise ValueError("sandbox observation decision must use its closed enum")
        if any(not isinstance(item, SandboxObservationState) for item in self.states):
            raise ValueError(
                "sandbox observation report states must use the closed enum"
            )
        limits = tuple(sorted(self.limit_results, key=lambda item: item.limit.value))
        if len(limits) != len(SandboxLimit) or {item.limit for item in limits} != set(
            SandboxLimit
        ):
            raise ValueError("sandbox observation report requires every limit once")
        object.__setattr__(self, "limit_results", limits)
        for name in (
            "sandbox_target_identity",
            "request_identity",
            "idempotency_identity",
            "readiness_evidence_identity",
            "rehearsal_evidence_identity",
            "observation_identity",
            "policy_identity",
            "evidence_identity",
        ):
            _require_digest(getattr(self, name), name.replace("_", " "))
        if self.evidence_identity != _digest(self.identity_dict()):
            raise ValueError(
                "sandbox observation evidence identity does not match content"
            )

    def identity_dict(self) -> dict[str, object]:
        return {
            "decision": self.decision.value,
            "states": [item.value for item in self.states],
            "sandbox_target_identity": self.sandbox_target_identity,
            "request_identity": self.request_identity,
            "idempotency_identity": self.idempotency_identity,
            "readiness_evidence_identity": self.readiness_evidence_identity,
            "rehearsal_evidence_identity": self.rehearsal_evidence_identity,
            "observation_identity": self.observation_identity,
            "policy_identity": self.policy_identity,
            "limit_results": [item.to_dict() for item in self.limit_results],
        }

    @property
    def encoded_size(self) -> int:
        return len(
            json.dumps(
                {**self.identity_dict(), "evidence_identity": self.evidence_identity},
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        )


def _limit_results(
    observation: SandboxObservationSnapshot, policy: SandboxObservationPolicy
) -> tuple[SandboxLimitResult, ...]:
    pairs = (
        (
            SandboxLimit.QUEUE_DEPTH,
            observation.observed_queue_depth,
            policy.maximum_queue_depth,
        ),
        (
            SandboxLimit.QUOTA_UNITS,
            observation.observed_quota_units,
            policy.maximum_quota_units,
        ),
        (
            SandboxLimit.COST_MICROUNITS,
            observation.observed_cost_microunits,
            policy.maximum_cost_microunits,
        ),
        (
            SandboxLimit.RETENTION_DURATION,
            observation.observed_retention_duration_ns,
            policy.maximum_retention_duration_ns,
        ),
    )
    return tuple(
        SandboxLimitResult(limit, observed, maximum, observed <= maximum)
        for limit, observed, maximum in pairs
    )


def evaluate_provider_sandbox_observation(
    readiness_report: CanaryReadinessReport,
    rehearsal_report: OfflineRehearsalReport,
    observation: SandboxObservationSnapshot,
    policy: SandboxObservationPolicy,
) -> SandboxObservationReport:
    """Validate anonymous caller-supplied observation facts without side effects."""

    if not isinstance(readiness_report, CanaryReadinessReport):
        raise ValueError("sandbox observation requires a CanaryReadinessReport")
    if not isinstance(rehearsal_report, OfflineRehearsalReport):
        raise ValueError("sandbox observation requires an OfflineRehearsalReport")
    if not isinstance(observation, SandboxObservationSnapshot):
        raise ValueError("sandbox observation requires an observation snapshot")
    if not isinstance(policy, SandboxObservationPolicy):
        raise ValueError("sandbox observation requires an explicit policy")

    limits = _limit_results(observation, policy)
    parents_ready = (
        readiness_report.status
        is CanaryReadinessStatus.READY_FOR_SEPARATE_ACTIVATION_REVIEW
        and rehearsal_report.outcome is RehearsalOutcome.OFFLINE_REHEARSAL_PASSED
        and rehearsal_report.readiness_evidence_identity
        == readiness_report.evidence_identity
        and observation.readiness_evidence_identity
        == readiness_report.evidence_identity
        and observation.rehearsal_evidence_identity
        == rehearsal_report.evidence_identity
    )
    safety_ready = (
        observation.kill_switch_available_and_current
        and observation.rollback_evidence_present
        and observation.operator_acknowledgement_present
        and observation.conformance_current
        and observation.target_identity_matches
        and observation.non_billable_synthetic_workload
    )
    within_limits = (
        len(observation.states) <= policy.maximum_state_count
        and observation.encoded_size <= policy.maximum_input_bytes
        and all(item.passed for item in limits)
    )

    if not parents_ready or not safety_ready or not within_limits:
        decision = SandboxObservationDecision.OBSERVATION_REJECTED
    elif observation.states == _INCOMPLETE_STATES:
        decision = SandboxObservationDecision.OBSERVATION_INCOMPLETE
    elif observation.states == _UNKNOWN_STATES:
        decision = SandboxObservationDecision.RECONCILIATION_REQUIRED
    elif observation.states == _ACCEPTED_STATES:
        decision = SandboxObservationDecision.OBSERVATION_ACCEPTED
    else:
        decision = SandboxObservationDecision.OBSERVATION_REJECTED

    values = {
        "decision": decision.value,
        "states": [item.value for item in observation.states],
        "sandbox_target_identity": observation.sandbox_target_identity,
        "request_identity": observation.request_identity,
        "idempotency_identity": observation.idempotency_identity,
        "readiness_evidence_identity": readiness_report.evidence_identity,
        "rehearsal_evidence_identity": rehearsal_report.evidence_identity,
        "observation_identity": observation.observation_identity,
        "policy_identity": policy.policy_identity,
        "limit_results": [
            item.to_dict() for item in sorted(limits, key=lambda item: item.limit.value)
        ],
    }
    report = SandboxObservationReport(
        decision,
        observation.states,
        observation.sandbox_target_identity,
        observation.request_identity,
        observation.idempotency_identity,
        readiness_report.evidence_identity,
        rehearsal_report.evidence_identity,
        observation.observation_identity,
        policy.policy_identity,
        limits,
        _digest(values),
    )
    if report.encoded_size > policy.maximum_evidence_bytes:
        raise ValueError("sandbox observation evidence exceeds its explicit limit")
    return report


def accepted_observation_states() -> tuple[SandboxObservationState, ...]:
    return _ACCEPTED_STATES


def rejected_observation_states() -> tuple[SandboxObservationState, ...]:
    return _REJECTED_STATES


def unknown_observation_states() -> tuple[SandboxObservationState, ...]:
    return _UNKNOWN_STATES


def incomplete_observation_states() -> tuple[SandboxObservationState, ...]:
    return _INCOMPLETE_STATES
