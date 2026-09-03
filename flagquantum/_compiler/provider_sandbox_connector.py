"""Private deterministic connector contracts over an offline scripted transport."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import Enum

_SHA256 = re.compile(r"[0-9a-f]{64}")


def _digest(value: dict[str, object]) -> str:
    encoded = json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_digest(value: str, label: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"sandbox connector {label} must be lowercase SHA-256")


def _require_nonnegative(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"sandbox connector {label} must be nonnegative")


class SandboxConnectorOperation(str, Enum):
    PREFLIGHT = "preflight"
    SUBMIT_ONCE = "submit_once"
    READ_STATUS = "read_status"
    REQUEST_CANCEL = "request_cancel"
    READ_RESULT = "read_result"


class SandboxLifecycleState(str, Enum):
    PREPARED = "prepared"
    ADMITTED = "admitted"
    SUBMISSION_STARTED = "submission_started"
    ACCEPTED = "accepted"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    OUTCOME_UNKNOWN = "outcome_unknown"


class SandboxTerminalState(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    OUTCOME_UNKNOWN = "outcome_unknown"


class SandboxConnectorError(str, Enum):
    NONE = "none"
    POLICY_REJECTED = "policy_rejected"
    TARGET_ATTESTATION_INVALID = "target_attestation_invalid"
    CREDENTIAL_REFERENCE_INVALID = "credential_reference_invalid"
    CAPABILITY_MISMATCH = "capability_mismatch"
    QUOTA_OR_COST_LIMIT = "quota_or_cost_limit"
    TRANSPORT_TRANSCRIPT_EXHAUSTED = "transport_transcript_exhausted"
    PROVIDER_REJECTED = "provider_rejected"
    PROVIDER_FAILED = "provider_failed"
    CANCELLATION_FAILED = "cancellation_failed"
    SUBMISSION_OUTCOME_UNKNOWN = "submission_outcome_unknown"
    RESULT_INVALID = "result_invalid"


@dataclass(frozen=True)
class SandboxTargetAttestation:
    provider_namespace_identity: str
    sandbox_target_identity: str
    capability_identity: str
    attestation_identity: str
    non_billable: bool
    sandbox_only: bool
    expires_after_sequence: int

    def __post_init__(self) -> None:
        for name in (
            "provider_namespace_identity",
            "sandbox_target_identity",
            "capability_identity",
            "attestation_identity",
        ):
            _require_digest(getattr(self, name), name.replace("_", " "))
        if not isinstance(self.non_billable, bool) or not isinstance(
            self.sandbox_only, bool
        ):
            raise ValueError("sandbox connector target flags must be boolean")
        _require_nonnegative(self.expires_after_sequence, "attestation expiry")

    def identity_dict(self) -> dict[str, object]:
        return {
            "provider_namespace_identity": self.provider_namespace_identity,
            "sandbox_target_identity": self.sandbox_target_identity,
            "capability_identity": self.capability_identity,
            "attestation_identity": self.attestation_identity,
            "non_billable": self.non_billable,
            "sandbox_only": self.sandbox_only,
            "expires_after_sequence": self.expires_after_sequence,
        }


@dataclass(frozen=True)
class CredentialReference:
    reference_identity: str
    provider_namespace_identity: str
    scope_identity: str
    expires_after_sequence: int

    def __post_init__(self) -> None:
        for name in (
            "reference_identity",
            "provider_namespace_identity",
            "scope_identity",
        ):
            _require_digest(getattr(self, name), name.replace("_", " "))
        _require_nonnegative(self.expires_after_sequence, "credential reference expiry")

    def identity_dict(self) -> dict[str, object]:
        return {
            "reference_identity": self.reference_identity,
            "provider_namespace_identity": self.provider_namespace_identity,
            "scope_identity": self.scope_identity,
            "expires_after_sequence": self.expires_after_sequence,
        }


@dataclass(frozen=True)
class SandboxConnectorRequest:
    operations: tuple[SandboxConnectorOperation, ...]
    request_identity: str
    program_identity: str
    artifact_identity: str
    idempotency_identity: str
    requested_target_identity: str
    capability_identity: str
    sequence: int
    kill_switch_available_and_current: bool
    admission_approved: bool
    operator_acknowledgement_present: bool
    rollback_evidence_present: bool
    conformance_current: bool

    def __post_init__(self) -> None:
        operations = tuple(self.operations)
        if not operations or any(
            not isinstance(item, SandboxConnectorOperation) for item in operations
        ):
            raise ValueError("sandbox connector operations must use the closed enum")
        if len(operations) != len(set(operations)):
            raise ValueError("sandbox connector operations must be unique")
        for name in (
            "request_identity",
            "program_identity",
            "artifact_identity",
            "idempotency_identity",
            "requested_target_identity",
            "capability_identity",
        ):
            _require_digest(getattr(self, name), name.replace("_", " "))
        _require_nonnegative(self.sequence, "request sequence")
        for name in (
            "kill_switch_available_and_current",
            "admission_approved",
            "operator_acknowledgement_present",
            "rollback_evidence_present",
            "conformance_current",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"sandbox connector {name} must be boolean")
        object.__setattr__(self, "operations", operations)

    def identity_dict(self) -> dict[str, object]:
        return {
            "operations": [item.value for item in self.operations],
            "request_identity": self.request_identity,
            "program_identity": self.program_identity,
            "artifact_identity": self.artifact_identity,
            "idempotency_identity": self.idempotency_identity,
            "requested_target_identity": self.requested_target_identity,
            "capability_identity": self.capability_identity,
            "sequence": self.sequence,
            "kill_switch_available_and_current": self.kill_switch_available_and_current,
            "admission_approved": self.admission_approved,
            "operator_acknowledgement_present": self.operator_acknowledgement_present,
            "rollback_evidence_present": self.rollback_evidence_present,
            "conformance_current": self.conformance_current,
        }


@dataclass(frozen=True)
class ScriptedSandboxResponse:
    operation: SandboxConnectorOperation
    lifecycle_state: SandboxLifecycleState
    actual_target_identity: str
    quota_units: int
    cost_microunits: int
    error_class: SandboxConnectorError

    def __post_init__(self) -> None:
        if not isinstance(self.operation, SandboxConnectorOperation):
            raise ValueError("scripted response operation must use the closed enum")
        if not isinstance(self.lifecycle_state, SandboxLifecycleState):
            raise ValueError("scripted response state must use the closed enum")
        if not isinstance(self.error_class, SandboxConnectorError):
            raise ValueError("scripted response error must use the closed enum")
        _require_digest(self.actual_target_identity, "actual target identity")
        _require_nonnegative(self.quota_units, "quota units")
        _require_nonnegative(self.cost_microunits, "cost microunits")

    def identity_dict(self) -> dict[str, object]:
        return {
            "operation": self.operation.value,
            "lifecycle_state": self.lifecycle_state.value,
            "actual_target_identity": self.actual_target_identity,
            "quota_units": self.quota_units,
            "cost_microunits": self.cost_microunits,
            "error_class": self.error_class.value,
        }


@dataclass(frozen=True)
class ScriptedSandboxTransport:
    responses: tuple[ScriptedSandboxResponse, ...]

    def __post_init__(self) -> None:
        responses = tuple(self.responses)
        if not responses or any(
            not isinstance(item, ScriptedSandboxResponse) for item in responses
        ):
            raise ValueError("scripted transport requires closed response values")
        object.__setattr__(self, "responses", responses)

    def identity_dict(self) -> dict[str, object]:
        return {"responses": [item.identity_dict() for item in self.responses]}

    @property
    def transcript_identity(self) -> str:
        return _digest(self.identity_dict())


@dataclass(frozen=True)
class SandboxConnectorPolicy:
    maximum_operation_count: int
    maximum_transcript_count: int
    maximum_input_bytes: int
    maximum_evidence_bytes: int
    maximum_quota_units: int
    maximum_cost_microunits: int
    maximum_sequence_value: int

    def __post_init__(self) -> None:
        for name in (
            "maximum_operation_count",
            "maximum_transcript_count",
            "maximum_input_bytes",
            "maximum_evidence_bytes",
            "maximum_quota_units",
            "maximum_cost_microunits",
            "maximum_sequence_value",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"sandbox connector policy {name} must be positive")
        if self.maximum_input_bytes < 1024 or self.maximum_evidence_bytes < 1024:
            raise ValueError(
                "sandbox connector input and evidence limits must be >= 1024 bytes"
            )

    def identity_dict(self) -> dict[str, object]:
        return {
            "maximum_operation_count": self.maximum_operation_count,
            "maximum_transcript_count": self.maximum_transcript_count,
            "maximum_input_bytes": self.maximum_input_bytes,
            "maximum_evidence_bytes": self.maximum_evidence_bytes,
            "maximum_quota_units": self.maximum_quota_units,
            "maximum_cost_microunits": self.maximum_cost_microunits,
            "maximum_sequence_value": self.maximum_sequence_value,
        }

    @property
    def policy_identity(self) -> str:
        return _digest(self.identity_dict())


@dataclass(frozen=True)
class SandboxConnectorReport:
    operations: tuple[SandboxConnectorOperation, ...]
    lifecycle_states: tuple[SandboxLifecycleState, ...]
    terminal_state: SandboxTerminalState
    error_class: SandboxConnectorError
    request_identity: str
    program_identity: str
    artifact_identity: str
    idempotency_identity: str
    provider_namespace_identity: str
    requested_target_identity: str
    actual_target_identity: str
    credential_reference_identity: str
    quota_units: int
    cost_microunits: int
    transcript_identity: str
    policy_identity: str
    evidence_label: str
    evidence_identity: str

    def __post_init__(self) -> None:
        if any(
            not isinstance(item, SandboxConnectorOperation) for item in self.operations
        ):
            raise ValueError("connector report operations must use the closed enum")
        if any(
            not isinstance(item, SandboxLifecycleState)
            for item in self.lifecycle_states
        ):
            raise ValueError("connector report states must use the closed enum")
        if not isinstance(self.terminal_state, SandboxTerminalState):
            raise ValueError("connector report terminal state must use the closed enum")
        if not isinstance(self.error_class, SandboxConnectorError):
            raise ValueError("connector report error must use the closed enum")
        for name in (
            "request_identity",
            "program_identity",
            "artifact_identity",
            "idempotency_identity",
            "provider_namespace_identity",
            "requested_target_identity",
            "actual_target_identity",
            "credential_reference_identity",
            "transcript_identity",
            "policy_identity",
            "evidence_identity",
        ):
            _require_digest(getattr(self, name), name.replace("_", " "))
        _require_nonnegative(self.quota_units, "report quota units")
        _require_nonnegative(self.cost_microunits, "report cost microunits")
        if self.evidence_label != "offline_scripted_evidence":
            raise ValueError("connector report must retain its offline evidence label")
        if self.evidence_identity != _digest(self.identity_dict()):
            raise ValueError("connector evidence identity does not match content")

    def identity_dict(self) -> dict[str, object]:
        return {
            "operations": [item.value for item in self.operations],
            "lifecycle_states": [item.value for item in self.lifecycle_states],
            "terminal_state": self.terminal_state.value,
            "error_class": self.error_class.value,
            "request_identity": self.request_identity,
            "program_identity": self.program_identity,
            "artifact_identity": self.artifact_identity,
            "idempotency_identity": self.idempotency_identity,
            "provider_namespace_identity": self.provider_namespace_identity,
            "requested_target_identity": self.requested_target_identity,
            "actual_target_identity": self.actual_target_identity,
            "credential_reference_identity": self.credential_reference_identity,
            "quota_units": self.quota_units,
            "cost_microunits": self.cost_microunits,
            "transcript_identity": self.transcript_identity,
            "policy_identity": self.policy_identity,
            "evidence_label": self.evidence_label,
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


_VALID_TRANSITIONS = {
    SandboxLifecycleState.PREPARED: {SandboxLifecycleState.ADMITTED},
    SandboxLifecycleState.ADMITTED: {
        SandboxLifecycleState.SUBMISSION_STARTED,
    },
    SandboxLifecycleState.SUBMISSION_STARTED: {
        SandboxLifecycleState.ACCEPTED,
        SandboxLifecycleState.FAILED,
        SandboxLifecycleState.OUTCOME_UNKNOWN,
    },
    SandboxLifecycleState.ACCEPTED: {
        SandboxLifecycleState.RUNNING,
        SandboxLifecycleState.CANCELLED,
        SandboxLifecycleState.FAILED,
    },
    SandboxLifecycleState.RUNNING: {
        SandboxLifecycleState.SUCCEEDED,
        SandboxLifecycleState.FAILED,
        SandboxLifecycleState.CANCELLED,
    },
}


def _input_size(*values: dict[str, object]) -> int:
    return len(
        json.dumps(
            values, ensure_ascii=True, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
    )


def _terminal(state: SandboxLifecycleState) -> SandboxTerminalState:
    return SandboxTerminalState(state.value)


def _lifecycle_states(
    responses: tuple[ScriptedSandboxResponse, ...],
) -> tuple[SandboxLifecycleState, ...]:
    states = [SandboxLifecycleState.PREPARED]
    for response in responses:
        if response.operation is SandboxConnectorOperation.SUBMIT_ONCE:
            states.append(SandboxLifecycleState.SUBMISSION_STARTED)
        states.append(response.lifecycle_state)
    return tuple(states)


def run_scripted_sandbox_connector(
    request: SandboxConnectorRequest,
    attestation: SandboxTargetAttestation,
    credential_reference: CredentialReference,
    transport: ScriptedSandboxTransport,
    policy: SandboxConnectorPolicy,
) -> SandboxConnectorReport:
    """Evaluate one finite scripted connector transcript without external access."""

    if not isinstance(request, SandboxConnectorRequest):
        raise ValueError("sandbox connector requires a connector request")
    if not isinstance(attestation, SandboxTargetAttestation):
        raise ValueError("sandbox connector requires a target attestation")
    if not isinstance(credential_reference, CredentialReference):
        raise ValueError("sandbox connector requires a credential reference")
    if not isinstance(transport, ScriptedSandboxTransport):
        raise ValueError("sandbox connector requires a scripted transport")
    if not isinstance(policy, SandboxConnectorPolicy):
        raise ValueError("sandbox connector requires an explicit policy")

    responses = transport.responses
    lifecycle_states = _lifecycle_states(responses)
    last = responses[-1]
    input_size = _input_size(
        request.identity_dict(),
        attestation.identity_dict(),
        credential_reference.identity_dict(),
        transport.identity_dict(),
        policy.identity_dict(),
    )
    structure_valid = (
        len(request.operations) == len(responses)
        and all(
            operation is response.operation
            for operation, response in zip(request.operations, responses, strict=True)
        )
        and all(
            right in _VALID_TRANSITIONS.get(left, set())
            for left, right in zip(lifecycle_states, lifecycle_states[1:])
        )
        and last.lifecycle_state.value in {item.value for item in SandboxTerminalState}
    )
    identity_valid = (
        request.requested_target_identity == attestation.sandbox_target_identity
        and request.capability_identity == attestation.capability_identity
        and credential_reference.provider_namespace_identity
        == attestation.provider_namespace_identity
        and all(
            item.actual_target_identity == request.requested_target_identity
            for item in responses
        )
    )
    safety_valid = (
        attestation.non_billable
        and attestation.sandbox_only
        and request.kill_switch_available_and_current
        and request.admission_approved
        and request.operator_acknowledgement_present
        and request.rollback_evidence_present
        and request.conformance_current
        and request.sequence <= attestation.expires_after_sequence
        and request.sequence <= credential_reference.expires_after_sequence
    )
    quota_units = max(item.quota_units for item in responses)
    cost_microunits = max(item.cost_microunits for item in responses)
    resource_valid = (
        len(request.operations) <= policy.maximum_operation_count
        and len(responses) <= policy.maximum_transcript_count
        and input_size <= policy.maximum_input_bytes
        and quota_units <= policy.maximum_quota_units
        and cost_microunits <= policy.maximum_cost_microunits
        and request.sequence <= policy.maximum_sequence_value
    )

    if not structure_valid:
        terminal_state = SandboxTerminalState.FAILED
        error_class = SandboxConnectorError.TRANSPORT_TRANSCRIPT_EXHAUSTED
    elif not identity_valid:
        terminal_state = SandboxTerminalState.FAILED
        error_class = SandboxConnectorError.TARGET_ATTESTATION_INVALID
    elif not safety_valid:
        terminal_state = SandboxTerminalState.FAILED
        error_class = SandboxConnectorError.POLICY_REJECTED
    elif not resource_valid:
        terminal_state = SandboxTerminalState.FAILED
        error_class = SandboxConnectorError.QUOTA_OR_COST_LIMIT
    else:
        terminal_state = _terminal(last.lifecycle_state)
        error_class = last.error_class
        if (
            terminal_state is SandboxTerminalState.SUCCEEDED
            and error_class is not SandboxConnectorError.NONE
        ):
            terminal_state = SandboxTerminalState.FAILED
            error_class = SandboxConnectorError.RESULT_INVALID
        elif terminal_state is SandboxTerminalState.OUTCOME_UNKNOWN:
            error_class = SandboxConnectorError.SUBMISSION_OUTCOME_UNKNOWN
        elif (
            terminal_state is SandboxTerminalState.FAILED
            and error_class is SandboxConnectorError.NONE
        ):
            error_class = SandboxConnectorError.PROVIDER_FAILED

    values = {
        "operations": [item.value for item in request.operations],
        "lifecycle_states": [item.value for item in lifecycle_states],
        "terminal_state": terminal_state.value,
        "error_class": error_class.value,
        "request_identity": request.request_identity,
        "program_identity": request.program_identity,
        "artifact_identity": request.artifact_identity,
        "idempotency_identity": request.idempotency_identity,
        "provider_namespace_identity": attestation.provider_namespace_identity,
        "requested_target_identity": request.requested_target_identity,
        "actual_target_identity": last.actual_target_identity,
        "credential_reference_identity": credential_reference.reference_identity,
        "quota_units": quota_units,
        "cost_microunits": cost_microunits,
        "transcript_identity": transport.transcript_identity,
        "policy_identity": policy.policy_identity,
        "evidence_label": "offline_scripted_evidence",
    }
    report = SandboxConnectorReport(
        request.operations,
        lifecycle_states,
        terminal_state,
        error_class,
        request.request_identity,
        request.program_identity,
        request.artifact_identity,
        request.idempotency_identity,
        attestation.provider_namespace_identity,
        request.requested_target_identity,
        last.actual_target_identity,
        credential_reference.reference_identity,
        quota_units,
        cost_microunits,
        transport.transcript_identity,
        policy.policy_identity,
        "offline_scripted_evidence",
        _digest(values),
    )
    if report.encoded_size > policy.maximum_evidence_bytes:
        raise ValueError("sandbox connector evidence exceeds its explicit limit")
    return report
