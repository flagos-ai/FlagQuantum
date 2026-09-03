from __future__ import annotations

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
from flagquantum._compiler.deployment_rehearsal import (
    OfflineRehearsalReport,
    RehearsalObjective,
    RehearsalObservationSnapshot,
    RehearsalOutcome,
    RehearsalPolicy,
    RehearsalRole,
    RehearsalScenario,
    RehearsalState,
    SimulatedRehearsalAction,
    normal_rehearsal_states,
    rehearsal_requirements,
    run_offline_deployment_rehearsal,
)
from tests.internal_ir.test_deployment_bridge_stage3 import _evaluate as _readiness
from tools import public_api_snapshot

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/internal_ir/deployment_bridge_stage4.json"
PROPOSAL = ROOT / "contracts/deployment-bridge-stage4-entry-proposal.json"


def _policy(**changes: int) -> RehearsalPolicy:
    values = {
        "maximum_scenario_count": 1,
        "maximum_state_transitions": 8,
        "maximum_simulated_actions": 11,
        "maximum_input_bytes": 65536,
        "maximum_evidence_bytes": 65536,
        "maximum_detection_duration_ns": 100,
        "maximum_operator_acknowledgement_duration_ns": 100,
        "maximum_rollback_recovery_duration_ns": 100,
        "maximum_evidence_restriction_or_deletion_duration_ns": 100,
    }
    values.update(changes)
    return RehearsalPolicy(**values)


def _observations(
    scenario: RehearsalScenario, **changes: object
) -> RehearsalObservationSnapshot:
    roles, actions = rehearsal_requirements(scenario)
    values = {
        "states": normal_rehearsal_states(),
        "simulated_actions": tuple(actions),
        "responsible_roles": tuple(roles),
        "detection_duration_ns": 10,
        "operator_acknowledgement_duration_ns": 20,
        "rollback_recovery_duration_ns": 30,
        "evidence_restriction_or_deletion_duration_ns": 40,
    }
    values.update(changes)
    return RehearsalObservationSnapshot(**values)


def _run(scenario: RehearsalScenario) -> OfflineRehearsalReport:
    return run_offline_deployment_rehearsal(
        _readiness(), scenario, _observations(scenario), _policy()
    )


def test_runtime_taxonomies_and_report_exactly_match_the_approved_proposal() -> None:
    proposal = json.loads(PROPOSAL.read_text(encoding="utf-8"))

    assert {item.value for item in RehearsalScenario} == set(
        proposal["scenario_taxonomy"]
    )
    assert {item.value for item in RehearsalState} == set(proposal["state_taxonomy"])
    assert {item.value for item in SimulatedRehearsalAction} == set(
        proposal["simulated_action_taxonomy"]
    )
    assert {item.value for item in RehearsalOutcome} == set(
        proposal["outcome_taxonomy"]
    )
    assert {item.value for item in RehearsalRole} == set(proposal["role_taxonomy"])
    assert [item.name for item in fields(OfflineRehearsalReport)] == proposal[
        "privacy_identity_and_report_contract"
    ]["report_fields"]


def test_all_thirteen_anonymous_scenarios_pass_with_golden_evidence() -> None:
    golden = json.loads(FIXTURE.read_text(encoding="utf-8"))["scenarios"]

    assert set(golden) == {item.value for item in RehearsalScenario}
    for scenario in RehearsalScenario:
        report = _run(scenario)
        assert report.outcome is RehearsalOutcome.OFFLINE_REHEARSAL_PASSED
        assert report.evidence_identity == golden[scenario.value]
        assert report.encoded_size <= 65536


def test_runtime_requirements_exactly_match_the_approved_scenario_matrix() -> None:
    proposal = json.loads(PROPOSAL.read_text(encoding="utf-8"))
    matrix = {item["scenario"]: item for item in proposal["scenario_response_contract"]}

    for scenario in RehearsalScenario:
        roles, actions = rehearsal_requirements(scenario)
        assert {item.value for item in roles} == set(
            matrix[scenario.value]["required_roles"]
        )
        assert {item.value for item in actions} == set(
            matrix[scenario.value]["required_actions"]
        )


def test_omitted_reordered_duplicated_and_failed_states_fail_closed() -> None:
    scenario = RehearsalScenario.KILL_SWITCH_UNAVAILABLE
    normal = normal_rehearsal_states()
    states = (
        normal[:-1],
        (normal[1], normal[0], *normal[2:]),
        (normal[0], normal[0], *normal[1:]),
        (*normal[:-1], RehearsalState.REHEARSAL_FAILED),
    )

    for sequence in states:
        result = run_offline_deployment_rehearsal(
            _readiness(),
            scenario,
            _observations(scenario, states=sequence),
            _policy(maximum_state_transitions=16),
        )
        assert result.outcome is RehearsalOutcome.OFFLINE_REHEARSAL_FAILED


def test_missing_or_extra_action_and_role_fail_closed() -> None:
    scenario = RehearsalScenario.UNKNOWN_SUBMISSION_OUTCOME
    roles, actions = rehearsal_requirements(scenario)
    cases = (
        _observations(scenario, simulated_actions=tuple(actions)[1:]),
        _observations(
            scenario,
            simulated_actions=(
                *actions,
                SimulatedRehearsalAction.MARK_CONFORMANCE_UNUSABLE,
            ),
        ),
        _observations(scenario, responsible_roles=tuple(roles)[1:]),
        _observations(
            scenario,
            responsible_roles=(*roles, RehearsalRole.PRIVACY_RETENTION_OWNER),
        ),
    )

    for observations in cases:
        result = run_offline_deployment_rehearsal(
            _readiness(), scenario, observations, _policy()
        )
        assert result.outcome is RehearsalOutcome.OFFLINE_REHEARSAL_FAILED


def test_every_synthetic_objective_boundary_passes_and_breach_fails() -> None:
    scenario = RehearsalScenario.QUEUE_BUDGET_BREACH
    boundary = _observations(
        scenario,
        detection_duration_ns=100,
        operator_acknowledgement_duration_ns=100,
        rollback_recovery_duration_ns=100,
        evidence_restriction_or_deletion_duration_ns=100,
    )
    passed = run_offline_deployment_rehearsal(
        _readiness(), scenario, boundary, _policy()
    )
    assert passed.outcome is RehearsalOutcome.OFFLINE_REHEARSAL_PASSED
    assert all(item.passed for item in passed.objective_results)
    assert {item.objective for item in passed.objective_results} == set(
        RehearsalObjective
    )

    for field_name in (
        "detection_duration_ns",
        "operator_acknowledgement_duration_ns",
        "rollback_recovery_duration_ns",
        "evidence_restriction_or_deletion_duration_ns",
    ):
        failed = run_offline_deployment_rehearsal(
            _readiness(),
            scenario,
            _observations(scenario, **{field_name: 101}),
            _policy(),
        )
        assert failed.outcome is RehearsalOutcome.OFFLINE_REHEARSAL_FAILED
        assert any(not item.passed for item in failed.objective_results)


def test_nonready_parent_is_incomplete_and_never_promoted() -> None:
    scenario = RehearsalScenario.READINESS_EVIDENCE_UNAVAILABLE
    report = run_offline_deployment_rehearsal(
        _readiness(activation=False), scenario, _observations(scenario), _policy()
    )

    assert report.outcome is RehearsalOutcome.OFFLINE_REHEARSAL_INCOMPLETE
    assert "production" not in repr(report).lower()
    assert "canary_active" not in repr(report).lower()


def test_unknown_submission_freezes_retry_and_never_models_fallback() -> None:
    report = _run(RehearsalScenario.UNKNOWN_SUBMISSION_OUTCOME)

    assert SimulatedRehearsalAction.FREEZE_CANDIDATE_RETRY in report.simulated_actions
    assert (
        SimulatedRehearsalAction.REQUIRE_SUBMISSION_RECONCILIATION
        in report.simulated_actions
    )
    assert all("fallback" not in item.value for item in SimulatedRehearsalAction)
    assert all(
        "retry" not in item.value or "freeze" in item.value
        for item in report.simulated_actions
    )


def test_inputs_and_report_are_immutable_and_privacy_safe() -> None:
    readiness = _readiness()
    scenario = RehearsalScenario.PRIVACY_BOUNDARY_BREACH
    observations = _observations(scenario)
    policy = _policy()
    before = deepcopy((readiness, observations, policy))
    report = run_offline_deployment_rehearsal(readiness, scenario, observations, policy)

    assert (readiness, observations, policy) == before
    assert {item.name for item in fields(OfflineRehearsalReport)}.isdisjoint(
        {
            "adapter",
            "artifact",
            "backend",
            "binding",
            "credential",
            "endpoint",
            "job_id",
            "metadata",
            "payload",
            "person",
            "program",
            "provider",
            "receipt",
            "result",
        }
    )
    with pytest.raises(FrozenInstanceError):
        report.outcome = RehearsalOutcome.OFFLINE_REHEARSAL_FAILED


def test_private_engine_has_no_runtime_provider_network_or_public_path() -> None:
    source = inspect.getsource(sys.modules[run_offline_deployment_rehearsal.__module__])

    for forbidden in (
        "from .runtime_adapters import",
        "from .provider_conformance import",
        "from .shadow_harness import",
        "requests",
        "urllib",
        ".submit(",
        ".execute(",
        "sleep(",
    ):
        assert forbidden not in source
    assert public_api_snapshot.validate() == ()
    assert not hasattr(fq, "run_offline_deployment_rehearsal")


def test_evidence_identities_are_stable_across_python_hash_seeds() -> None:
    script = f"""
import json
import runpy
namespace = runpy.run_path({str(Path(__file__))!r})
Scenario = namespace['RehearsalScenario']
print(json.dumps({{
    item.value: namespace['_run'](item).evidence_identity
    for item in Scenario
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

    golden = json.loads(FIXTURE.read_text(encoding="utf-8"))["scenarios"]
    assert identities("1") == identities("8675309") == golden


def test_policy_and_observation_types_reject_ambiguous_values() -> None:
    with pytest.raises(ValueError, match="exactly one scenario"):
        _policy(maximum_scenario_count=2)
    with pytest.raises(ValueError, match=">= 1024"):
        _policy(maximum_input_bytes=100)
    with pytest.raises(ValueError, match="must be nonnegative"):
        _observations(
            RehearsalScenario.KILL_SWITCH_UNAVAILABLE, detection_duration_ns=-1
        )
    with pytest.raises(ValueError, match="must be unique"):
        _observations(
            RehearsalScenario.KILL_SWITCH_UNAVAILABLE,
            responsible_roles=(RehearsalRole.INCIDENT_COMMANDER,) * 2,
        )
