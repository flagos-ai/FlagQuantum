"""Conformance for the pure dynamic post-routing rematch."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

import flagquantum as fq
import flagquantum.runtime.dynamic as dynamic_runtime
from flagquantum.core.target_capabilities import TargetIdentity
from flagquantum.deployment import CloudBackendProfile
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.conformance import DynamicFeatureSet
from flagquantum.runtime.dynamic.post_routing import (
    _evaluate_post_routing_policy,
    evaluate_dynamic_post_routing_rematch,
)
from flagquantum.runtime.dynamic.remote_capabilities import (
    dynamic_remote_requirements,
    dynamic_remote_snapshot,
)

pytestmark = pytest.mark.unit

_CAPTURED_AT = datetime(2026, 10, 24, 8, 0, tzinfo=timezone.utc)
_EVALUATED_AT = _CAPTURED_AT + timedelta(minutes=1)


def _program(*, conditional: bool = True) -> DynamicCircuit:
    circuit = DynamicCircuit(2).measure(0, classical_bit=0)
    if conditional:
        circuit.conditional("x", 1, classical_bit=0)
    return circuit


def _snapshot(*, ttl: timedelta = timedelta(hours=1)):
    backend = CloudBackendProfile(
        provider="provider",
        name="backend",
        n_qubits=8,
        supports_dynamic_circuits=True,
        dynamic_dialect="openqasm3",
    )
    features = DynamicFeatureSet(
        name="test-target",
        supported_conditional_gates=("x",),
        max_condition_bits=1,
        condition_values=(1,),
    )
    return dynamic_remote_snapshot(
        backend,
        features=features,
        captured_at=_CAPTURED_AT,
        ttl=ttl,
        provider_version="test-sdk-1",
        target_revision="calibration-1",
        environment_id="test",
        evidence_id="declaration",
        evidence_sha256="a" * 64,
    )


def _evaluate(
    *,
    routed: DynamicCircuit | None = None,
    snapshot=None,
    preflight_snapshot_id: str | None = None,
    provider_legality: str = "accepted",
    route_requirements: tuple[str, ...] = (),
    route_facts: dict[str, str] | None = None,
):
    source = dynamic_remote_requirements(_program())
    candidate = snapshot or _snapshot()
    return evaluate_dynamic_post_routing_rematch(
        source,
        dynamic_remote_requirements(routed or _program()),
        presented_source_requirement_set_id=source.requirement_set_id,
        requested_target_identity=candidate.target_identity,
        preflight_snapshot_id=preflight_snapshot_id or candidate.snapshot_id,
        postrouting_snapshot=candidate,
        evaluated_at=_EVALUATED_AT,
        route_requirements=route_requirements,
        route_facts=route_facts,
        provider_legality=provider_legality,
    )


def test_unchanged_route_matches_one_exact_snapshot() -> None:
    decision = _evaluate(
        route_requirements=("operation:cx",),
        route_facts={"operation:cx": "supported"},
    )

    assert decision.compatible
    assert decision.blocker_codes == ()


def test_source_semantic_loss_cannot_be_hidden_by_routed_requirements() -> None:
    decision = _evaluate(routed=_program(conditional=False))

    assert not decision.compatible
    assert decision.blocker_codes == (
        "dynamic_source_semantic_lost:classical_feed_forward",
        "dynamic_source_semantic_lost:condition_values",
        "dynamic_source_semantic_lost:max_condition_bits",
        "dynamic_source_semantic_lost:supported_conditional_gates",
    )


def test_stale_snapshot_and_missing_route_fact_are_distinct() -> None:
    snapshot = _snapshot(ttl=timedelta(seconds=1))
    decision = _evaluate(
        snapshot=snapshot,
        route_requirements=("directed_edge:0->1",),
    )

    assert decision.blocker_codes == (
        "dynamic_postrouting_snapshot_stale",
        "dynamic_route_capability_fact_missing:directed_edge:0->1",
    )


def test_exact_target_identity_and_snapshot_identity_are_immutable() -> None:
    snapshot = _snapshot()
    other_identity = TargetIdentity(
        target_id="provider:other",
        target_class="remote_qpu",
        provider="provider",
        provider_version="test-sdk-1",
        target_revision="calibration-1",
        environment_id="test",
    )
    source = dynamic_remote_requirements(_program())
    decision = evaluate_dynamic_post_routing_rematch(
        source,
        dynamic_remote_requirements(_program()),
        presented_source_requirement_set_id=source.requirement_set_id,
        requested_target_identity=other_identity,
        preflight_snapshot_id="different-snapshot",
        postrouting_snapshot=snapshot,
        evaluated_at=_EVALUATED_AT,
        provider_legality="accepted",
    )

    assert decision.blocker_codes[:2] == (
        "dynamic_postrouting_target_identity_mismatch",
        "dynamic_postrouting_snapshot_identity_mismatch",
    )


def test_same_target_id_cannot_hide_a_revision_change() -> None:
    snapshot = _snapshot()
    source = dynamic_remote_requirements(_program())
    requested = replace(
        snapshot.target_identity,
        target_revision="different-calibration",
    )

    decision = evaluate_dynamic_post_routing_rematch(
        source,
        dynamic_remote_requirements(_program()),
        presented_source_requirement_set_id=source.requirement_set_id,
        requested_target_identity=requested,
        preflight_snapshot_id=snapshot.snapshot_id,
        postrouting_snapshot=snapshot,
        evaluated_at=_EVALUATED_AT,
        provider_legality="accepted",
    )

    assert decision.blocker_codes[0] == ("dynamic_postrouting_target_identity_mismatch")


def test_internal_gate_is_not_a_public_export() -> None:
    assert "evaluate_dynamic_post_routing_rematch" not in fq.__all__
    assert not hasattr(dynamic_runtime, "evaluate_dynamic_post_routing_rematch")


@pytest.mark.parametrize(
    "state", ("missing", "unsupported", "unknown", "unmeasured", "not_exposed")
)
def test_route_fact_states_have_precise_blockers(state: str) -> None:
    facts = {} if state == "missing" else {"operation:cx": state}

    blockers = _evaluate_post_routing_policy(
        source_requirement_set_id="source",
        presented_source_requirement_set_id="source",
        source_requirements=(),
        preserved_source_requirements=(),
        requested_target_id="provider:backend",
        postrouting_target_id="provider:backend",
        preflight_snapshot_id="snapshot",
        postrouting_snapshot_id="snapshot",
        snapshot_fresh=True,
        route_requirements=("operation:cx",),
        route_facts=facts,
        provider_legality="accepted",
    )

    assert blockers == (f"dynamic_route_capability_fact_{state}:operation:cx",)
