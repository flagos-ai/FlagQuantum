"""Private deterministic failure rehearsal with no executable side effects."""

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

_SHA256 = re.compile(r"[0-9a-f]{64}")


def _digest(value: dict[str, object]) -> str:
    encoded = json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_digest(value: str, label: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"deployment rehearsal {label} must be lowercase SHA-256")


class RehearsalScenario(str, Enum):
    KILL_SWITCH_UNAVAILABLE = "kill_switch_unavailable"
    KILL_SWITCH_STATE_STALE = "kill_switch_state_stale"
    CONFORMANCE_REVOKED = "conformance_revoked"
    TARGET_IDENTITY_MISMATCH = "target_identity_mismatch"
    READINESS_EVIDENCE_UNAVAILABLE = "readiness_evidence_unavailable"
    QUEUE_BUDGET_BREACH = "queue_budget_breach"
    PROVIDER_QUOTA_BUDGET_BREACH = "provider_quota_budget_breach"
    MONETARY_COST_BUDGET_BREACH = "monetary_cost_budget_breach"
    PRIVACY_BOUNDARY_BREACH = "privacy_boundary_breach"
    UNKNOWN_SUBMISSION_OUTCOME = "unknown_submission_outcome"
    OPERATOR_ACKNOWLEDGEMENT_TIMEOUT = "operator_acknowledgement_timeout"
    ROLLBACK_RECOVERY_TIMEOUT = "rollback_recovery_timeout"
    RETENTION_OR_DELETION_BREACH = "retention_or_deletion_breach"


class RehearsalState(str, Enum):
    NOT_STARTED = "not_started"
    INJECTED = "injected"
    DETECTED = "detected"
    ADMISSION_BLOCKED = "admission_blocked"
    OPERATOR_ACKNOWLEDGED = "operator_acknowledged"
    ROLLBACK_VERIFIED = "rollback_verified"
    EVIDENCE_SEALED = "evidence_sealed"
    COMPLETED = "completed"
    REHEARSAL_FAILED = "rehearsal_failed"


class SimulatedRehearsalAction(str, Enum):
    RECORD_DETECTION = "record_detection"
    BLOCK_NEW_CANDIDATE_ADMISSION = "block_new_candidate_admission"
    PRESERVE_LEGACY_AUTHORITY = "preserve_legacy_authority"
    REQUEST_KILL_SWITCH_REVIEW = "request_kill_switch_review"
    FREEZE_CANDIDATE_RETRY = "freeze_candidate_retry"
    REQUIRE_SUBMISSION_RECONCILIATION = "require_submission_reconciliation"
    MARK_CONFORMANCE_UNUSABLE = "mark_conformance_unusable"
    ESCALATE_ANONYMOUS_OWNER_ROLE = "escalate_anonymous_owner_role"
    VERIFY_SYNTHETIC_ROLLBACK = "verify_synthetic_rollback"
    REQUEST_EVIDENCE_RESTRICTION_OR_DELETION = (
        "request_evidence_restriction_or_deletion"
    )
    SEAL_ANONYMOUS_REHEARSAL_EVIDENCE = "seal_anonymous_rehearsal_evidence"


class RehearsalOutcome(str, Enum):
    OFFLINE_REHEARSAL_PASSED = "offline_rehearsal_passed"
    OFFLINE_REHEARSAL_FAILED = "offline_rehearsal_failed"
    OFFLINE_REHEARSAL_INCOMPLETE = "offline_rehearsal_incomplete"


class RehearsalRole(str, Enum):
    ACTIVATION_APPROVER = "activation_approver"
    KILL_SWITCH_OWNER = "kill_switch_owner"
    PROVIDER_ADAPTER_OWNER = "provider_adapter_owner"
    PRIVACY_RETENTION_OWNER = "privacy_retention_owner"
    INCIDENT_COMMANDER = "incident_commander"
    COST_QUOTA_OWNER = "cost_quota_owner"


class RehearsalObjective(str, Enum):
    DETECTION = "detection_duration_ns"
    OPERATOR_ACKNOWLEDGEMENT = "operator_acknowledgement_duration_ns"
    ROLLBACK_RECOVERY = "rollback_recovery_duration_ns"
    EVIDENCE_RESTRICTION_OR_DELETION = "evidence_restriction_or_deletion_duration_ns"


_NORMAL_STATES = (
    RehearsalState.NOT_STARTED,
    RehearsalState.INJECTED,
    RehearsalState.DETECTED,
    RehearsalState.ADMISSION_BLOCKED,
    RehearsalState.OPERATOR_ACKNOWLEDGED,
    RehearsalState.ROLLBACK_VERIFIED,
    RehearsalState.EVIDENCE_SEALED,
    RehearsalState.COMPLETED,
)

_CORE_ACTIONS = frozenset(
    {
        SimulatedRehearsalAction.RECORD_DETECTION,
        SimulatedRehearsalAction.BLOCK_NEW_CANDIDATE_ADMISSION,
        SimulatedRehearsalAction.PRESERVE_LEGACY_AUTHORITY,
        SimulatedRehearsalAction.SEAL_ANONYMOUS_REHEARSAL_EVIDENCE,
    }
)

_SCENARIO_REQUIREMENTS = {
    RehearsalScenario.KILL_SWITCH_UNAVAILABLE: (
        frozenset(
            {
                RehearsalRole.KILL_SWITCH_OWNER,
                RehearsalRole.INCIDENT_COMMANDER,
            }
        ),
        _CORE_ACTIONS
        | {
            SimulatedRehearsalAction.FREEZE_CANDIDATE_RETRY,
            SimulatedRehearsalAction.ESCALATE_ANONYMOUS_OWNER_ROLE,
        },
    ),
    RehearsalScenario.KILL_SWITCH_STATE_STALE: (
        frozenset(
            {
                RehearsalRole.KILL_SWITCH_OWNER,
                RehearsalRole.INCIDENT_COMMANDER,
            }
        ),
        _CORE_ACTIONS
        | {
            SimulatedRehearsalAction.REQUEST_KILL_SWITCH_REVIEW,
            SimulatedRehearsalAction.FREEZE_CANDIDATE_RETRY,
            SimulatedRehearsalAction.ESCALATE_ANONYMOUS_OWNER_ROLE,
        },
    ),
    RehearsalScenario.CONFORMANCE_REVOKED: (
        frozenset(
            {
                RehearsalRole.PROVIDER_ADAPTER_OWNER,
                RehearsalRole.INCIDENT_COMMANDER,
            }
        ),
        _CORE_ACTIONS
        | {
            SimulatedRehearsalAction.MARK_CONFORMANCE_UNUSABLE,
            SimulatedRehearsalAction.VERIFY_SYNTHETIC_ROLLBACK,
        },
    ),
    RehearsalScenario.TARGET_IDENTITY_MISMATCH: (
        frozenset(
            {
                RehearsalRole.PROVIDER_ADAPTER_OWNER,
                RehearsalRole.INCIDENT_COMMANDER,
            }
        ),
        _CORE_ACTIONS
        | {
            SimulatedRehearsalAction.MARK_CONFORMANCE_UNUSABLE,
            SimulatedRehearsalAction.FREEZE_CANDIDATE_RETRY,
        },
    ),
    RehearsalScenario.READINESS_EVIDENCE_UNAVAILABLE: (
        frozenset(
            {
                RehearsalRole.ACTIVATION_APPROVER,
                RehearsalRole.INCIDENT_COMMANDER,
            }
        ),
        _CORE_ACTIONS
        | {
            SimulatedRehearsalAction.FREEZE_CANDIDATE_RETRY,
            SimulatedRehearsalAction.ESCALATE_ANONYMOUS_OWNER_ROLE,
        },
    ),
    RehearsalScenario.QUEUE_BUDGET_BREACH: (
        frozenset(
            {
                RehearsalRole.INCIDENT_COMMANDER,
                RehearsalRole.COST_QUOTA_OWNER,
            }
        ),
        _CORE_ACTIONS
        | {
            SimulatedRehearsalAction.REQUEST_KILL_SWITCH_REVIEW,
            SimulatedRehearsalAction.VERIFY_SYNTHETIC_ROLLBACK,
        },
    ),
    RehearsalScenario.PROVIDER_QUOTA_BUDGET_BREACH: (
        frozenset(
            {
                RehearsalRole.PROVIDER_ADAPTER_OWNER,
                RehearsalRole.COST_QUOTA_OWNER,
            }
        ),
        _CORE_ACTIONS
        | {
            SimulatedRehearsalAction.FREEZE_CANDIDATE_RETRY,
            SimulatedRehearsalAction.VERIFY_SYNTHETIC_ROLLBACK,
        },
    ),
    RehearsalScenario.MONETARY_COST_BUDGET_BREACH: (
        frozenset(
            {
                RehearsalRole.INCIDENT_COMMANDER,
                RehearsalRole.COST_QUOTA_OWNER,
            }
        ),
        _CORE_ACTIONS
        | {
            SimulatedRehearsalAction.REQUEST_KILL_SWITCH_REVIEW,
            SimulatedRehearsalAction.FREEZE_CANDIDATE_RETRY,
            SimulatedRehearsalAction.VERIFY_SYNTHETIC_ROLLBACK,
        },
    ),
    RehearsalScenario.PRIVACY_BOUNDARY_BREACH: (
        frozenset(
            {
                RehearsalRole.PRIVACY_RETENTION_OWNER,
                RehearsalRole.INCIDENT_COMMANDER,
            }
        ),
        _CORE_ACTIONS
        | {
            SimulatedRehearsalAction.REQUEST_KILL_SWITCH_REVIEW,
            SimulatedRehearsalAction.FREEZE_CANDIDATE_RETRY,
            SimulatedRehearsalAction.REQUEST_EVIDENCE_RESTRICTION_OR_DELETION,
        },
    ),
    RehearsalScenario.UNKNOWN_SUBMISSION_OUTCOME: (
        frozenset(
            {
                RehearsalRole.PROVIDER_ADAPTER_OWNER,
                RehearsalRole.INCIDENT_COMMANDER,
                RehearsalRole.COST_QUOTA_OWNER,
            }
        ),
        _CORE_ACTIONS
        | {
            SimulatedRehearsalAction.REQUEST_KILL_SWITCH_REVIEW,
            SimulatedRehearsalAction.FREEZE_CANDIDATE_RETRY,
            SimulatedRehearsalAction.REQUIRE_SUBMISSION_RECONCILIATION,
            SimulatedRehearsalAction.ESCALATE_ANONYMOUS_OWNER_ROLE,
        },
    ),
    RehearsalScenario.OPERATOR_ACKNOWLEDGEMENT_TIMEOUT: (
        frozenset(
            {
                RehearsalRole.KILL_SWITCH_OWNER,
                RehearsalRole.INCIDENT_COMMANDER,
            }
        ),
        _CORE_ACTIONS
        | {
            SimulatedRehearsalAction.FREEZE_CANDIDATE_RETRY,
            SimulatedRehearsalAction.ESCALATE_ANONYMOUS_OWNER_ROLE,
        },
    ),
    RehearsalScenario.ROLLBACK_RECOVERY_TIMEOUT: (
        frozenset(
            {
                RehearsalRole.ACTIVATION_APPROVER,
                RehearsalRole.KILL_SWITCH_OWNER,
                RehearsalRole.INCIDENT_COMMANDER,
            }
        ),
        _CORE_ACTIONS
        | {
            SimulatedRehearsalAction.REQUEST_KILL_SWITCH_REVIEW,
            SimulatedRehearsalAction.FREEZE_CANDIDATE_RETRY,
            SimulatedRehearsalAction.ESCALATE_ANONYMOUS_OWNER_ROLE,
        },
    ),
    RehearsalScenario.RETENTION_OR_DELETION_BREACH: (
        frozenset(
            {
                RehearsalRole.PRIVACY_RETENTION_OWNER,
                RehearsalRole.INCIDENT_COMMANDER,
            }
        ),
        _CORE_ACTIONS
        | {
            SimulatedRehearsalAction.REQUEST_EVIDENCE_RESTRICTION_OR_DELETION,
            SimulatedRehearsalAction.ESCALATE_ANONYMOUS_OWNER_ROLE,
        },
    ),
}


@dataclass(frozen=True)
class SyntheticObjectiveResult:
    objective: RehearsalObjective
    observed_ns: int
    maximum_ns: int
    passed: bool

    def __post_init__(self) -> None:
        if not isinstance(self.objective, RehearsalObjective):
            raise ValueError("rehearsal objective must use its closed enum")
        for name in ("observed_ns", "maximum_ns"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"rehearsal objective {name} must be nonnegative")
        if not isinstance(self.passed, bool):
            raise ValueError("rehearsal objective passed flag must be boolean")
        if self.passed is not (self.observed_ns <= self.maximum_ns):
            raise ValueError("rehearsal objective result does not match its limit")

    def to_dict(self) -> dict[str, object]:
        return {
            "objective": self.objective.value,
            "observed_ns": self.observed_ns,
            "maximum_ns": self.maximum_ns,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class RehearsalObservationSnapshot:
    states: tuple[RehearsalState, ...]
    simulated_actions: tuple[SimulatedRehearsalAction, ...]
    responsible_roles: tuple[RehearsalRole, ...]
    detection_duration_ns: int
    operator_acknowledgement_duration_ns: int
    rollback_recovery_duration_ns: int
    evidence_restriction_or_deletion_duration_ns: int

    def __post_init__(self) -> None:
        states = tuple(self.states)
        actions = tuple(sorted(self.simulated_actions, key=lambda item: item.value))
        roles = tuple(sorted(self.responsible_roles, key=lambda item: item.value))
        if any(not isinstance(item, RehearsalState) for item in states):
            raise ValueError("rehearsal states must use the closed enum")
        if any(not isinstance(item, SimulatedRehearsalAction) for item in actions):
            raise ValueError("rehearsal actions must use the closed enum")
        if any(not isinstance(item, RehearsalRole) for item in roles):
            raise ValueError("rehearsal roles must use the closed enum")
        if len(actions) != len(set(actions)) or len(roles) != len(set(roles)):
            raise ValueError("rehearsal actions and roles must be unique")
        for name in (
            "detection_duration_ns",
            "operator_acknowledgement_duration_ns",
            "rollback_recovery_duration_ns",
            "evidence_restriction_or_deletion_duration_ns",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"rehearsal {name} must be nonnegative")
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "simulated_actions", actions)
        object.__setattr__(self, "responsible_roles", roles)

    def identity_dict(self) -> dict[str, object]:
        return {
            "states": [item.value for item in self.states],
            "simulated_actions": [item.value for item in self.simulated_actions],
            "responsible_roles": [item.value for item in self.responsible_roles],
            "detection_duration_ns": self.detection_duration_ns,
            "operator_acknowledgement_duration_ns": self.operator_acknowledgement_duration_ns,
            "rollback_recovery_duration_ns": self.rollback_recovery_duration_ns,
            "evidence_restriction_or_deletion_duration_ns": self.evidence_restriction_or_deletion_duration_ns,
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
class RehearsalPolicy:
    maximum_scenario_count: int
    maximum_state_transitions: int
    maximum_simulated_actions: int
    maximum_input_bytes: int
    maximum_evidence_bytes: int
    maximum_detection_duration_ns: int
    maximum_operator_acknowledgement_duration_ns: int
    maximum_rollback_recovery_duration_ns: int
    maximum_evidence_restriction_or_deletion_duration_ns: int

    def __post_init__(self) -> None:
        if self.maximum_scenario_count != 1:
            raise ValueError("rehearsal policy requires exactly one scenario")
        for name in (
            "maximum_state_transitions",
            "maximum_simulated_actions",
            "maximum_input_bytes",
            "maximum_evidence_bytes",
            "maximum_detection_duration_ns",
            "maximum_operator_acknowledgement_duration_ns",
            "maximum_rollback_recovery_duration_ns",
            "maximum_evidence_restriction_or_deletion_duration_ns",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"rehearsal policy {name} must be positive")
        if self.maximum_input_bytes < 1024 or self.maximum_evidence_bytes < 1024:
            raise ValueError(
                "rehearsal input and evidence limits must be >= 1024 bytes"
            )

    def identity_dict(self) -> dict[str, object]:
        return {
            "maximum_scenario_count": self.maximum_scenario_count,
            "maximum_state_transitions": self.maximum_state_transitions,
            "maximum_simulated_actions": self.maximum_simulated_actions,
            "maximum_input_bytes": self.maximum_input_bytes,
            "maximum_evidence_bytes": self.maximum_evidence_bytes,
            "maximum_detection_duration_ns": self.maximum_detection_duration_ns,
            "maximum_operator_acknowledgement_duration_ns": self.maximum_operator_acknowledgement_duration_ns,
            "maximum_rollback_recovery_duration_ns": self.maximum_rollback_recovery_duration_ns,
            "maximum_evidence_restriction_or_deletion_duration_ns": self.maximum_evidence_restriction_or_deletion_duration_ns,
        }

    @property
    def policy_identity(self) -> str:
        return _digest(self.identity_dict())


@dataclass(frozen=True)
class OfflineRehearsalReport:
    scenario: RehearsalScenario
    outcome: RehearsalOutcome
    states: tuple[RehearsalState, ...]
    simulated_actions: tuple[SimulatedRehearsalAction, ...]
    responsible_roles: tuple[RehearsalRole, ...]
    readiness_evidence_identity: str
    observation_identity: str
    policy_identity: str
    objective_results: tuple[SyntheticObjectiveResult, ...]
    evidence_identity: str

    def __post_init__(self) -> None:
        if not isinstance(self.scenario, RehearsalScenario):
            raise ValueError("rehearsal scenario must use its closed enum")
        if not isinstance(self.outcome, RehearsalOutcome):
            raise ValueError("rehearsal outcome must use its closed enum")
        if any(not isinstance(item, RehearsalState) for item in self.states):
            raise ValueError("rehearsal report states must use the closed enum")
        if any(
            not isinstance(item, SimulatedRehearsalAction)
            for item in self.simulated_actions
        ):
            raise ValueError("rehearsal report actions must use the closed enum")
        if any(not isinstance(item, RehearsalRole) for item in self.responsible_roles):
            raise ValueError("rehearsal report roles must use the closed enum")
        objectives = tuple(
            sorted(self.objective_results, key=lambda item: item.objective.value)
        )
        if len(objectives) != len(RehearsalObjective) or {
            item.objective for item in objectives
        } != set(RehearsalObjective):
            raise ValueError("rehearsal report requires every objective exactly once")
        object.__setattr__(self, "objective_results", objectives)
        for name in (
            "readiness_evidence_identity",
            "observation_identity",
            "policy_identity",
            "evidence_identity",
        ):
            _require_digest(getattr(self, name), name.replace("_", " "))
        if self.evidence_identity != _digest(self.identity_dict()):
            raise ValueError("rehearsal evidence identity does not match content")

    def identity_dict(self) -> dict[str, object]:
        return {
            "scenario": self.scenario.value,
            "outcome": self.outcome.value,
            "states": [item.value for item in self.states],
            "simulated_actions": [item.value for item in self.simulated_actions],
            "responsible_roles": [item.value for item in self.responsible_roles],
            "readiness_evidence_identity": self.readiness_evidence_identity,
            "observation_identity": self.observation_identity,
            "policy_identity": self.policy_identity,
            "objective_results": [item.to_dict() for item in self.objective_results],
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


def _objective_results(
    observations: RehearsalObservationSnapshot, policy: RehearsalPolicy
) -> tuple[SyntheticObjectiveResult, ...]:
    pairs = (
        (
            RehearsalObjective.DETECTION,
            observations.detection_duration_ns,
            policy.maximum_detection_duration_ns,
        ),
        (
            RehearsalObjective.OPERATOR_ACKNOWLEDGEMENT,
            observations.operator_acknowledgement_duration_ns,
            policy.maximum_operator_acknowledgement_duration_ns,
        ),
        (
            RehearsalObjective.ROLLBACK_RECOVERY,
            observations.rollback_recovery_duration_ns,
            policy.maximum_rollback_recovery_duration_ns,
        ),
        (
            RehearsalObjective.EVIDENCE_RESTRICTION_OR_DELETION,
            observations.evidence_restriction_or_deletion_duration_ns,
            policy.maximum_evidence_restriction_or_deletion_duration_ns,
        ),
    )
    return tuple(
        SyntheticObjectiveResult(objective, observed, maximum, observed <= maximum)
        for objective, observed, maximum in pairs
    )


def run_offline_deployment_rehearsal(
    readiness_report: CanaryReadinessReport,
    scenario: RehearsalScenario,
    observations: RehearsalObservationSnapshot,
    policy: RehearsalPolicy,
) -> OfflineRehearsalReport:
    """Validate one synthetic rehearsal and return evidence without side effects."""

    if not isinstance(readiness_report, CanaryReadinessReport):
        raise ValueError("deployment rehearsal requires a CanaryReadinessReport")
    if not isinstance(scenario, RehearsalScenario):
        raise ValueError("deployment rehearsal requires a closed scenario")
    if not isinstance(observations, RehearsalObservationSnapshot):
        raise ValueError("deployment rehearsal requires an observation snapshot")
    if not isinstance(policy, RehearsalPolicy):
        raise ValueError("deployment rehearsal requires an explicit policy")

    required_roles, required_actions = _SCENARIO_REQUIREMENTS[scenario]
    objectives = _objective_results(observations, policy)
    within_resource_limits = (
        len(observations.states) <= policy.maximum_state_transitions
        and len(observations.simulated_actions) <= policy.maximum_simulated_actions
        and observations.encoded_size <= policy.maximum_input_bytes
    )
    exact_rehearsal = (
        observations.states == _NORMAL_STATES
        and frozenset(observations.simulated_actions) == required_actions
        and frozenset(observations.responsible_roles) == required_roles
        and all(item.passed for item in objectives)
        and within_resource_limits
    )
    if (
        readiness_report.status
        is not CanaryReadinessStatus.READY_FOR_SEPARATE_ACTIVATION_REVIEW
    ):
        outcome = RehearsalOutcome.OFFLINE_REHEARSAL_INCOMPLETE
    elif exact_rehearsal:
        outcome = RehearsalOutcome.OFFLINE_REHEARSAL_PASSED
    else:
        outcome = RehearsalOutcome.OFFLINE_REHEARSAL_FAILED

    values = {
        "scenario": scenario.value,
        "outcome": outcome.value,
        "states": [item.value for item in observations.states],
        "simulated_actions": [item.value for item in observations.simulated_actions],
        "responsible_roles": [item.value for item in observations.responsible_roles],
        "readiness_evidence_identity": readiness_report.evidence_identity,
        "observation_identity": observations.observation_identity,
        "policy_identity": policy.policy_identity,
        "objective_results": [
            item.to_dict()
            for item in sorted(objectives, key=lambda item: item.objective.value)
        ],
    }
    report = OfflineRehearsalReport(
        scenario,
        outcome,
        observations.states,
        observations.simulated_actions,
        observations.responsible_roles,
        readiness_report.evidence_identity,
        observations.observation_identity,
        policy.policy_identity,
        objectives,
        _digest(values),
    )
    if report.encoded_size > policy.maximum_evidence_bytes:
        raise ValueError("deployment rehearsal evidence exceeds its explicit limit")
    return report


def rehearsal_requirements(
    scenario: RehearsalScenario,
) -> tuple[frozenset[RehearsalRole], frozenset[SimulatedRehearsalAction]]:
    """Return immutable closed requirements for anonymous fixture construction."""

    if not isinstance(scenario, RehearsalScenario):
        raise ValueError("deployment rehearsal requires a closed scenario")
    return _SCENARIO_REQUIREMENTS[scenario]


def normal_rehearsal_states() -> tuple[RehearsalState, ...]:
    """Return the immutable normal state sequence for offline fixtures."""

    return _NORMAL_STATES
