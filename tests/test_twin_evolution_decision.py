"""Scenarios for explicit, non-executing Twin evolution decisions."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace

import pytest

import flagquantum as fq
from tests.test_twin_evolution_cycle import _complete_cycle
from tests.test_twin_region_release_assessment import _release

pytestmark = pytest.mark.integration


def _released_cycle():
    completed = _complete_cycle()[3]
    _, _, release = _release()
    cycle = replace(
        completed,
        incumbent_snapshot_identity=release.incumbent_snapshot_identity,
        candidate_snapshot_identity=release.candidate_snapshot_identity,
        target=release.target,
        physical_qubits=release.physical_qubits,
    )
    return cycle, release


def test_release_decision_binds_cycle_and_explicit_release() -> None:
    cycle, release = _released_cycle()

    decision = fq.twin.record_evolution_decision(
        cycle,
        decision="release_candidate",
        decided_at="2026-08-15T11:00:00+08:00",
        decision_authority="calibration-operator",
        release=release,
    )

    assert decision.cycle_identity == cycle.identity
    assert decision.recommended_action == "release_candidate"
    assert decision.decision == "release_candidate"
    assert decision.followed_recommendation is True
    assert decision.release_identity == release.identity
    assert decision.released_region_identity == release.candidate_region_identity
    assert decision.candidate_snapshot_identity == release.candidate_snapshot_identity
    assert decision.holdout_evaluation_identity == cycle.holdout_evaluation_identity
    with pytest.raises(FrozenInstanceError):
        decision.decision = "retain_incumbent"  # type: ignore[misc]


def test_operator_may_retain_candidate_without_claiming_a_release() -> None:
    cycle = _complete_cycle()[3]

    decision = fq.twin.record_evolution_decision(
        cycle,
        decision="retain_incumbent",
        decided_at="2026-08-15T11:00:00+08:00",
        decision_authority="calibration-operator",
    )

    assert decision.recommended_action == "release_candidate"
    assert decision.decision == "retain_incumbent"
    assert decision.followed_recommendation is False
    assert decision.release_identity is None
    assert decision.released_region_identity is None


def test_release_fails_closed_without_a_completed_matching_gate() -> None:
    prepared = _complete_cycle()[0]
    cycle, release = _released_cycle()

    with pytest.raises(ValueError, match="completed holdout"):
        fq.twin.record_evolution_decision(
            prepared,
            decision="release_candidate",
            decided_at="2026-08-15T11:00:00+08:00",
            decision_authority="calibration-operator",
            release=release,
        )
    with pytest.raises(ValueError, match="explicit release"):
        fq.twin.record_evolution_decision(
            cycle,
            decision="release_candidate",
            decided_at="2026-08-15T11:00:00+08:00",
            decision_authority="calibration-operator",
        )
    with pytest.raises(ValueError, match="does not match"):
        fq.twin.record_evolution_decision(
            replace(cycle, target="quafu:Baihua"),
            decision="release_candidate",
            decided_at="2026-08-15T11:00:00+08:00",
            decision_authority="calibration-operator",
            release=release,
        )
    with pytest.raises(ValueError, match="non-release"):
        fq.twin.record_evolution_decision(
            cycle,
            decision="collect_more_evidence",
            decided_at="2026-08-15T11:00:00+08:00",
            decision_authority="calibration-operator",
            release=release,
        )


def test_decision_persistence_is_private_create_once_and_tamper_rejecting(
    tmp_path,
) -> None:
    cycle = _complete_cycle()[3]
    decision = fq.twin.record_evolution_decision(
        cycle,
        decision="collect_more_evidence",
        decided_at="2026-08-15T11:00:00+08:00",
        decision_authority="calibration-review-board",
    )
    destination = tmp_path / "evolution-decision.json"

    fq.twin.dump_evolution_decision(decision, destination)
    fq.twin.dump_evolution_decision(decision, destination)
    restored = fq.twin.load_evolution_decision(destination)

    assert restored == decision
    assert restored.identity == decision.identity
    assert destination.stat().st_mode & 0o777 == 0o600
    payload = json.loads(destination.read_text())
    payload["decision"] = "release_candidate"
    destination.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Invalid Twin evolution decision"):
        fq.twin.load_evolution_decision(destination)


def test_decision_requires_attributed_timezone_aware_input() -> None:
    cycle = _complete_cycle()[3]

    with pytest.raises(ValueError, match="timezone"):
        fq.twin.record_evolution_decision(
            cycle,
            decision="retain_incumbent",
            decided_at="2026-08-15T11:00:00",
            decision_authority="calibration-operator",
        )
    with pytest.raises(ValueError, match="authority"):
        fq.twin.record_evolution_decision(
            cycle,
            decision="retain_incumbent",
            decided_at="2026-08-15T11:00:00+08:00",
            decision_authority=" ",
        )


def test_decision_has_no_release_or_execution_authority() -> None:
    decision = fq.twin.record_evolution_decision(
        _complete_cycle()[3],
        decision="retain_incumbent",
        decided_at="2026-08-15T11:00:00+08:00",
        decision_authority="calibration-operator",
    )

    for name in (
        "activate",
        "apply",
        "promote",
        "release",
        "route",
        "submit",
        "poll",
        "retry",
        "notify",
    ):
        assert not hasattr(decision, name)
