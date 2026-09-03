from __future__ import annotations

import hashlib
import inspect
import json
import os
import subprocess
import sys
from copy import deepcopy
from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import pytest

import flagquantum as fq
from flagquantum._compiler.provider_sandbox_connector import (
    CredentialReference,
    SandboxConnectorError,
    SandboxConnectorOperation,
    SandboxConnectorPolicy,
    SandboxConnectorReport,
    SandboxConnectorRequest,
    SandboxLifecycleState,
    SandboxTargetAttestation,
    SandboxTerminalState,
    ScriptedSandboxResponse,
    ScriptedSandboxTransport,
    run_scripted_sandbox_connector,
)
from tools import public_api_snapshot

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/internal_ir/deployment_bridge_stage6.json"
PROPOSAL = ROOT / "contracts/deployment-bridge-stage6-entry-proposal.json"


def _identity(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _attestation(**changes: object) -> SandboxTargetAttestation:
    values = {
        "provider_namespace_identity": _identity("anonymous-provider-namespace"),
        "sandbox_target_identity": _identity("anonymous-sandbox-target"),
        "capability_identity": _identity("anonymous-capability"),
        "attestation_identity": _identity("anonymous-attestation"),
        "non_billable": True,
        "sandbox_only": True,
        "expires_after_sequence": 100,
    }
    values.update(changes)
    return SandboxTargetAttestation(**values)


def _credential(**changes: object) -> CredentialReference:
    values = {
        "reference_identity": _identity("anonymous-credential-reference"),
        "provider_namespace_identity": _identity("anonymous-provider-namespace"),
        "scope_identity": _identity("anonymous-sandbox-scope"),
        "expires_after_sequence": 100,
    }
    values.update(changes)
    return CredentialReference(**values)


def _policy(**changes: int) -> SandboxConnectorPolicy:
    values = {
        "maximum_operation_count": 5,
        "maximum_transcript_count": 5,
        "maximum_input_bytes": 65536,
        "maximum_evidence_bytes": 65536,
        "maximum_quota_units": 100,
        "maximum_cost_microunits": 100,
        "maximum_sequence_value": 100,
    }
    values.update(changes)
    return SandboxConnectorPolicy(**values)


def _scripts():
    operation = SandboxConnectorOperation
    state = SandboxLifecycleState
    error = SandboxConnectorError
    target = _identity("anonymous-sandbox-target")

    def response(op, lifecycle, failure=error.NONE, quota=10, cost=10):
        return ScriptedSandboxResponse(op, lifecycle, target, quota, cost, failure)

    return {
        "succeeded": (
            response(operation.PREFLIGHT, state.ADMITTED),
            response(operation.SUBMIT_ONCE, state.ACCEPTED),
            response(operation.READ_STATUS, state.RUNNING),
            response(operation.READ_RESULT, state.SUCCEEDED),
        ),
        "rejected": (
            response(operation.PREFLIGHT, state.ADMITTED),
            response(
                operation.SUBMIT_ONCE,
                state.FAILED,
                error.PROVIDER_REJECTED,
            ),
        ),
        "cancelled": (
            response(operation.PREFLIGHT, state.ADMITTED),
            response(operation.SUBMIT_ONCE, state.ACCEPTED),
            response(operation.READ_STATUS, state.RUNNING),
            response(operation.REQUEST_CANCEL, state.CANCELLED),
        ),
        "unknown": (
            response(operation.PREFLIGHT, state.ADMITTED),
            response(
                operation.SUBMIT_ONCE,
                state.OUTCOME_UNKNOWN,
                error.SUBMISSION_OUTCOME_UNKNOWN,
            ),
        ),
    }


def _request(responses, **changes: object) -> SandboxConnectorRequest:
    values = {
        "operations": tuple(item.operation for item in responses),
        "request_identity": _identity("anonymous-request"),
        "program_identity": _identity("anonymous-program"),
        "artifact_identity": _identity("anonymous-artifact"),
        "idempotency_identity": _identity("anonymous-idempotency"),
        "requested_target_identity": _identity("anonymous-sandbox-target"),
        "capability_identity": _identity("anonymous-capability"),
        "sequence": 10,
        "kill_switch_available_and_current": True,
        "admission_approved": True,
        "operator_acknowledgement_present": True,
        "rollback_evidence_present": True,
        "conformance_current": True,
    }
    values.update(changes)
    return SandboxConnectorRequest(**values)


def _run(name: str, **request_changes: object) -> SandboxConnectorReport:
    responses = _scripts()[name]
    return run_scripted_sandbox_connector(
        _request(responses, **request_changes),
        _attestation(),
        _credential(),
        ScriptedSandboxTransport(responses),
        _policy(),
    )


def test_runtime_taxonomies_and_report_match_the_approved_proposal() -> None:
    proposal = json.loads(PROPOSAL.read_text(encoding="utf-8"))

    assert {item.value for item in SandboxConnectorOperation} == set(
        proposal["operation_taxonomy"]
    )
    assert {item.value for item in SandboxLifecycleState} == set(
        proposal["lifecycle_state_taxonomy"]
    )
    assert {item.value for item in SandboxTerminalState} == set(
        proposal["terminal_state_taxonomy"]
    )
    assert {item.value for item in SandboxConnectorError} == set(
        proposal["error_taxonomy"]
    )
    assert set(proposal["normalized_report_contract"]["required_facts"]) <= {
        item.name for item in fields(SandboxConnectorReport)
    }


def test_four_scripted_outcomes_have_deterministic_golden_evidence() -> None:
    golden = json.loads(FIXTURE.read_text(encoding="utf-8"))["outcomes"]
    terminals = {
        "succeeded": SandboxTerminalState.SUCCEEDED,
        "rejected": SandboxTerminalState.FAILED,
        "cancelled": SandboxTerminalState.CANCELLED,
        "unknown": SandboxTerminalState.OUTCOME_UNKNOWN,
    }

    for name, terminal in terminals.items():
        report = _run(name)
        assert report.terminal_state is terminal
        assert report.evidence_identity == golden[name]
        assert report.evidence_label == "offline_scripted_evidence"
        assert report.encoded_size <= 65536


def test_lifecycle_includes_prepared_and_submission_started() -> None:
    report = _run("succeeded")

    assert report.lifecycle_states == (
        SandboxLifecycleState.PREPARED,
        SandboxLifecycleState.ADMITTED,
        SandboxLifecycleState.SUBMISSION_STARTED,
        SandboxLifecycleState.ACCEPTED,
        SandboxLifecycleState.RUNNING,
        SandboxLifecycleState.SUCCEEDED,
    )


def test_missing_extra_reordered_or_mismatched_transcript_fails_closed() -> None:
    responses = _scripts()["succeeded"]
    cases = (
        responses[:-1],
        (*responses, responses[-1]),
        (responses[1], responses[0], *responses[2:]),
    )
    request = _request(responses)

    for transcript in cases:
        report = run_scripted_sandbox_connector(
            request,
            _attestation(),
            _credential(),
            ScriptedSandboxTransport(transcript),
            _policy(maximum_transcript_count=8),
        )
        assert report.terminal_state is SandboxTerminalState.FAILED
        assert (
            report.error_class is SandboxConnectorError.TRANSPORT_TRANSCRIPT_EXHAUSTED
        )


def test_target_capability_and_credential_namespace_mismatch_fail_closed() -> None:
    responses = _scripts()["succeeded"]
    cases = (
        (
            _request(responses, requested_target_identity=_identity("wrong-target")),
            _attestation(),
            _credential(),
        ),
        (
            _request(responses, capability_identity=_identity("wrong-capability")),
            _attestation(),
            _credential(),
        ),
        (
            _request(responses),
            _attestation(),
            _credential(provider_namespace_identity=_identity("wrong-provider")),
        ),
    )

    for request, attestation, credential in cases:
        report = run_scripted_sandbox_connector(
            request,
            attestation,
            credential,
            ScriptedSandboxTransport(responses),
            _policy(),
        )
        assert report.terminal_state is SandboxTerminalState.FAILED
        assert report.error_class is SandboxConnectorError.TARGET_ATTESTATION_INVALID


def test_each_safety_gate_and_expiry_fails_closed() -> None:
    responses = _scripts()["succeeded"]
    for field_name in (
        "kill_switch_available_and_current",
        "admission_approved",
        "operator_acknowledgement_present",
        "rollback_evidence_present",
        "conformance_current",
    ):
        report = run_scripted_sandbox_connector(
            _request(responses, **{field_name: False}),
            _attestation(),
            _credential(),
            ScriptedSandboxTransport(responses),
            _policy(),
        )
        assert report.error_class is SandboxConnectorError.POLICY_REJECTED

    for attestation, credential in (
        (_attestation(non_billable=False), _credential()),
        (_attestation(sandbox_only=False), _credential()),
        (_attestation(expires_after_sequence=9), _credential()),
        (_attestation(), _credential(expires_after_sequence=9)),
    ):
        report = run_scripted_sandbox_connector(
            _request(responses),
            attestation,
            credential,
            ScriptedSandboxTransport(responses),
            _policy(),
        )
        assert report.error_class is SandboxConnectorError.POLICY_REJECTED


def test_quota_and_cost_boundaries_pass_and_breaches_fail_closed() -> None:
    responses = _scripts()["succeeded"]
    boundary = tuple(
        ScriptedSandboxResponse(
            item.operation,
            item.lifecycle_state,
            item.actual_target_identity,
            100,
            100,
            item.error_class,
        )
        for item in responses
    )
    passed = run_scripted_sandbox_connector(
        _request(boundary),
        _attestation(),
        _credential(),
        ScriptedSandboxTransport(boundary),
        _policy(),
    )
    assert passed.terminal_state is SandboxTerminalState.SUCCEEDED

    for quota, cost in ((101, 100), (100, 101)):
        breached = tuple(
            ScriptedSandboxResponse(
                item.operation,
                item.lifecycle_state,
                item.actual_target_identity,
                quota,
                cost,
                item.error_class,
            )
            for item in responses
        )
        report = run_scripted_sandbox_connector(
            _request(breached),
            _attestation(),
            _credential(),
            ScriptedSandboxTransport(breached),
            _policy(),
        )
        assert report.error_class is SandboxConnectorError.QUOTA_OR_COST_LIMIT


def test_unknown_outcome_preserves_idempotency_and_has_no_retry_or_fallback() -> None:
    report = _run("unknown")

    assert report.terminal_state is SandboxTerminalState.OUTCOME_UNKNOWN
    assert report.error_class is SandboxConnectorError.SUBMISSION_OUTCOME_UNKNOWN
    assert report.idempotency_identity == _identity("anonymous-idempotency")
    assert (
        sum(item is SandboxConnectorOperation.SUBMIT_ONCE for item in report.operations)
        == 1
    )
    assert all(
        "retry" not in item.value and "fallback" not in item.value
        for item in SandboxConnectorOperation
    )


def test_inputs_and_report_are_immutable_and_contain_no_secret_fields() -> None:
    responses = _scripts()["succeeded"]
    request = _request(responses)
    attestation = _attestation()
    credential = _credential()
    transport = ScriptedSandboxTransport(responses)
    policy = _policy()
    before = deepcopy((request, attestation, credential, transport, policy))
    report = run_scripted_sandbox_connector(
        request, attestation, credential, transport, policy
    )

    assert (request, attestation, credential, transport, policy) == before
    assert {item.name for item in fields(SandboxConnectorReport)}.isdisjoint(
        {"endpoint", "password", "secret", "token", "provider_job_id", "raw_result"}
    )
    with pytest.raises(FrozenInstanceError):
        report.error_class = SandboxConnectorError.PROVIDER_FAILED


def test_private_connector_has_no_runtime_provider_network_or_public_path() -> None:
    source = inspect.getsource(
        sys.modules[run_scripted_sandbox_connector.__module__]
    ).lower()

    for forbidden in (
        "from .runtime_adapters import",
        "from .provider_conformance import",
        "requests",
        "urllib",
        "socket",
        "subprocess",
        ".submit(",
        ".execute(",
        "os.environ",
    ):
        assert forbidden not in source
    assert public_api_snapshot.validate() == ()
    assert not hasattr(fq, "run_scripted_sandbox_connector")


def test_evidence_identities_are_stable_across_python_hash_seeds() -> None:
    script = f"""
import json
import runpy
namespace = runpy.run_path({str(Path(__file__))!r})
print(json.dumps({{
    name: namespace['_run'](name).evidence_identity
    for name in ('succeeded', 'rejected', 'cancelled', 'unknown')
}}, sort_keys=True))
"""

    def identities(seed: str) -> dict[str, str]:
        environment = dict(os.environ)
        environment["PYTHONHASHSEED"] = seed
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(completed.stdout)

    golden = json.loads(FIXTURE.read_text(encoding="utf-8"))["outcomes"]
    assert identities("1") == identities("8675309") == golden


def test_contract_types_reject_ambiguous_values() -> None:
    responses = _scripts()["succeeded"]
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        _attestation(sandbox_target_identity="target")
    with pytest.raises(ValueError, match="must be boolean"):
        _request(responses, admission_approved=1)
    with pytest.raises(ValueError, match="must be unique"):
        _request(responses, operations=(SandboxConnectorOperation.PREFLIGHT,) * 2)
    with pytest.raises(ValueError, match=">= 1024"):
        _policy(maximum_input_bytes=100)
    with pytest.raises(ValueError, match="connector request"):
        run_scripted_sandbox_connector(
            object(),
            _attestation(),
            _credential(),
            ScriptedSandboxTransport(responses),
            _policy(),
        )
