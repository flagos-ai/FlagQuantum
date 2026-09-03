from __future__ import annotations

import hashlib
import inspect
import json
import os
import subprocess
import sys
from copy import deepcopy
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

import flagquantum as fq
from flagquantum._compiler.deployment_rehearsal import (
    RehearsalPolicy,
    RehearsalScenario,
    rehearsal_requirements,
    run_offline_deployment_rehearsal,
)
from flagquantum._compiler.provider_sandbox_observation import (
    SandboxLimit,
    SandboxObservationDecision,
    SandboxObservationPolicy,
    SandboxObservationReport,
    SandboxObservationSnapshot,
    SandboxObservationState,
    accepted_observation_states,
    evaluate_provider_sandbox_observation,
    incomplete_observation_states,
    rejected_observation_states,
    unknown_observation_states,
)
from tests.internal_ir.test_deployment_bridge_stage3 import _evaluate as _readiness
from tests.internal_ir.test_deployment_bridge_stage4 import (
    _observations as _rehearsal_observations,
)
from tools import public_api_snapshot

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/internal_ir/deployment_bridge_stage5.json"
PROPOSAL = ROOT / "contracts/deployment-bridge-stage5-entry-proposal.json"


def _identity(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _parents():
    readiness = _readiness()
    scenario = RehearsalScenario.UNKNOWN_SUBMISSION_OUTCOME
    _, actions = rehearsal_requirements(scenario)
    rehearsal = run_offline_deployment_rehearsal(
        readiness,
        scenario,
        _rehearsal_observations(scenario, simulated_actions=tuple(actions)),
        RehearsalPolicy(1, 8, 11, 65536, 65536, 100, 100, 100, 100),
    )
    return readiness, rehearsal


def _policy(**changes: int) -> SandboxObservationPolicy:
    values = {
        "maximum_state_count": 4,
        "maximum_input_bytes": 65536,
        "maximum_evidence_bytes": 65536,
        "maximum_queue_depth": 100,
        "maximum_quota_units": 100,
        "maximum_cost_microunits": 100,
        "maximum_retention_duration_ns": 100,
    }
    values.update(changes)
    return SandboxObservationPolicy(**values)


def _snapshot(
    states: tuple[SandboxObservationState, ...], **changes: object
) -> SandboxObservationSnapshot:
    readiness, rehearsal = _parents()
    values = {
        "states": states,
        "sandbox_target_identity": _identity("anonymous-sandbox-target"),
        "request_identity": _identity("anonymous-request"),
        "idempotency_identity": _identity("anonymous-idempotency-key"),
        "readiness_evidence_identity": readiness.evidence_identity,
        "rehearsal_evidence_identity": rehearsal.evidence_identity,
        "kill_switch_available_and_current": True,
        "rollback_evidence_present": True,
        "operator_acknowledgement_present": True,
        "conformance_current": True,
        "target_identity_matches": True,
        "non_billable_synthetic_workload": True,
        "observed_queue_depth": 10,
        "observed_quota_units": 20,
        "observed_cost_microunits": 30,
        "observed_retention_duration_ns": 40,
    }
    values.update(changes)
    return SandboxObservationSnapshot(**values)


def _run(states: tuple[SandboxObservationState, ...]) -> SandboxObservationReport:
    readiness, rehearsal = _parents()
    return evaluate_provider_sandbox_observation(
        readiness, rehearsal, _snapshot(states), _policy()
    )


def test_runtime_taxonomies_exactly_match_the_approved_proposal() -> None:
    proposal = json.loads(PROPOSAL.read_text(encoding="utf-8"))

    assert {item.value for item in SandboxObservationState} == set(
        proposal["observation_state_taxonomy"]
    )
    assert {item.value for item in SandboxObservationDecision} == set(
        proposal["decision_taxonomy"]
    )
    assert {item.value for item in SandboxLimit} == set(
        proposal["quota_cost_and_retention_contract"][
            "caller_supplied_nonnegative_observations"
        ]
    )


def test_four_lifecycle_classes_have_deterministic_golden_evidence() -> None:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))["lifecycles"]
    cases = {
        "accepted": (
            accepted_observation_states(),
            SandboxObservationDecision.OBSERVATION_ACCEPTED,
        ),
        "rejected": (
            rejected_observation_states(),
            SandboxObservationDecision.OBSERVATION_REJECTED,
        ),
        "unknown": (
            unknown_observation_states(),
            SandboxObservationDecision.RECONCILIATION_REQUIRED,
        ),
        "incomplete": (
            incomplete_observation_states(),
            SandboxObservationDecision.OBSERVATION_INCOMPLETE,
        ),
    }
    for name, (states, decision) in cases.items():
        report = _run(states)
        assert report.decision is decision
        assert report.evidence_identity == fixture[name]
        assert report.encoded_size <= 65536


def test_lifecycle_helpers_exactly_match_the_approved_orders() -> None:
    proposal = json.loads(PROPOSAL.read_text(encoding="utf-8"))
    expected = {
        tuple(item.value for item in incomplete_observation_states()),
        tuple(item.value for item in rejected_observation_states()),
        tuple(item.value for item in accepted_observation_states()),
        tuple(item.value for item in unknown_observation_states()),
    }

    assert expected == {
        tuple(item) for item in proposal["lifecycle_contract"]["normal_orders"]
    }


def test_omitted_reordered_duplicated_and_crossed_states_fail_closed() -> None:
    accepted = accepted_observation_states()
    cases = (
        accepted[:-1],
        (accepted[1], accepted[0], *accepted[2:]),
        (accepted[0], accepted[0], *accepted[1:]),
        (*accepted[:-1], SandboxObservationState.OUTCOME_UNKNOWN),
    )

    for states in cases:
        assert _run(states).decision is SandboxObservationDecision.OBSERVATION_REJECTED


def test_parent_and_identity_mismatch_fail_closed() -> None:
    readiness, rehearsal = _parents()
    cases = (
        _snapshot(
            accepted_observation_states(),
            readiness_evidence_identity=_identity("wrong-readiness"),
        ),
        _snapshot(
            accepted_observation_states(),
            rehearsal_evidence_identity=_identity("wrong-rehearsal"),
        ),
    )
    for observation in cases:
        report = evaluate_provider_sandbox_observation(
            readiness, rehearsal, observation, _policy()
        )
        assert report.decision is SandboxObservationDecision.OBSERVATION_REJECTED


def test_each_safety_gate_fails_closed() -> None:
    readiness, rehearsal = _parents()
    for name in (
        "kill_switch_available_and_current",
        "rollback_evidence_present",
        "operator_acknowledgement_present",
        "conformance_current",
        "target_identity_matches",
        "non_billable_synthetic_workload",
    ):
        report = evaluate_provider_sandbox_observation(
            readiness,
            rehearsal,
            _snapshot(accepted_observation_states(), **{name: False}),
            _policy(),
        )
        assert report.decision is SandboxObservationDecision.OBSERVATION_REJECTED


def test_limit_boundaries_pass_and_each_breach_fails_closed() -> None:
    readiness, rehearsal = _parents()
    boundary = _snapshot(
        accepted_observation_states(),
        observed_queue_depth=100,
        observed_quota_units=100,
        observed_cost_microunits=100,
        observed_retention_duration_ns=100,
    )
    passed = evaluate_provider_sandbox_observation(
        readiness, rehearsal, boundary, _policy()
    )
    assert passed.decision is SandboxObservationDecision.OBSERVATION_ACCEPTED
    assert all(item.passed for item in passed.limit_results)

    for name in (
        "observed_queue_depth",
        "observed_quota_units",
        "observed_cost_microunits",
        "observed_retention_duration_ns",
    ):
        failed = evaluate_provider_sandbox_observation(
            readiness,
            rehearsal,
            _snapshot(accepted_observation_states(), **{name: 101}),
            _policy(),
        )
        assert failed.decision is SandboxObservationDecision.OBSERVATION_REJECTED
        assert any(not item.passed for item in failed.limit_results)


def test_unknown_outcome_preserves_identity_and_requires_reconciliation() -> None:
    observation = _snapshot(unknown_observation_states())
    report = _run(unknown_observation_states())

    assert report.decision is SandboxObservationDecision.RECONCILIATION_REQUIRED
    assert report.idempotency_identity == observation.idempotency_identity
    assert report.request_identity == observation.request_identity
    assert not any(
        "retry" in item.value or "fallback" in item.value
        for item in SandboxObservationDecision
    )


def test_inputs_and_outputs_are_immutable_and_privacy_safe() -> None:
    readiness, rehearsal = _parents()
    observation = _snapshot(accepted_observation_states())
    policy = _policy()
    before = deepcopy((readiness, rehearsal, observation, policy))
    report = evaluate_provider_sandbox_observation(
        readiness, rehearsal, observation, policy
    )

    assert (readiness, rehearsal, observation, policy) == before
    text = repr(report).lower()
    for forbidden in (
        "credential",
        "endpoint",
        "provider_job",
        "receipt",
        "raw_program",
        "raw_result",
    ):
        assert forbidden not in text
    with pytest.raises(FrozenInstanceError):
        report.decision = SandboxObservationDecision.OBSERVATION_REJECTED


def test_private_evaluator_has_no_runtime_provider_network_or_public_path() -> None:
    source = inspect.getsource(
        sys.modules[evaluate_provider_sandbox_observation.__module__]
    )

    for forbidden in (
        "from .runtime_adapters import",
        "from .provider_conformance import",
        "requests",
        "urllib",
        ".submit(",
        ".execute(",
        "socket",
        "credential",
    ):
        assert forbidden not in source.lower()
    assert public_api_snapshot.validate() == ()
    assert not hasattr(fq, "evaluate_provider_sandbox_observation")


def test_evidence_identities_are_stable_across_python_hash_seeds() -> None:
    script = f"""
import json
import runpy
namespace = runpy.run_path({str(Path(__file__))!r})
cases = {{
    'accepted': namespace['accepted_observation_states'](),
    'rejected': namespace['rejected_observation_states'](),
    'unknown': namespace['unknown_observation_states'](),
    'incomplete': namespace['incomplete_observation_states'](),
}}
print(json.dumps({{
    name: namespace['_run'](states).evidence_identity
    for name, states in cases.items()
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

    golden = json.loads(FIXTURE.read_text(encoding="utf-8"))["lifecycles"]
    assert identities("1") == identities("8675309") == golden


def test_policy_snapshot_and_argument_types_reject_ambiguous_values() -> None:
    with pytest.raises(ValueError, match=">= 1024"):
        _policy(maximum_input_bytes=100)
    with pytest.raises(ValueError, match="must be nonnegative"):
        _snapshot(accepted_observation_states(), observed_queue_depth=-1)
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        _snapshot(accepted_observation_states(), request_identity="request")
    with pytest.raises(ValueError, match="must be boolean"):
        _snapshot(accepted_observation_states(), conformance_current=1)
    with pytest.raises(ValueError, match="closed enum"):
        _snapshot(("intent_recorded",))
    with pytest.raises(ValueError, match="CanaryReadinessReport"):
        evaluate_provider_sandbox_observation(
            object(), _parents()[1], _snapshot(accepted_observation_states()), _policy()
        )
