"""Offline integration contract for routed dynamic capability rematching."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

import flagquantum as fq
import flagquantum.deployment as deployment
from flagquantum.compiler import CouplingMap
from flagquantum.core.target_capabilities import (
    CapabilityFact,
    FactExposure,
    FactSource,
    SupportStatus,
)
from flagquantum.deployment import CloudBackendProfile
from flagquantum.deployment.dynamic_rematch import (
    evaluate_dynamic_routing_rematch,
)
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.conformance import DynamicFeatureSet
from flagquantum.runtime.dynamic.remote_capabilities import (
    dynamic_remote_requirements,
    dynamic_remote_snapshot,
)
from flagquantum.runtime.dynamic.routing import route_dynamic_circuit

pytestmark = pytest.mark.unit

_CAPTURED_AT = datetime(2026, 10, 24, 8, 0, tzinfo=timezone.utc)
_EVALUATED_AT = _CAPTURED_AT + timedelta(minutes=1)


def _source() -> DynamicCircuit:
    circuit = DynamicCircuit(3).measure(0, classical_bit=0)
    return circuit.conditional("cx", (0, 2), classical_bit=0)


def _snapshot(*, gates: tuple[str, ...] = ("cx", "swap"), capacity: int = 3):
    backend = CloudBackendProfile(
        provider="provider",
        name="backend",
        n_qubits=3,
        supports_dynamic_circuits=True,
        dynamic_dialect="openqasm3",
    )
    snapshot = dynamic_remote_snapshot(
        backend,
        features=DynamicFeatureSet(
            name="test-target",
            supported_conditional_gates=("cx",),
            max_condition_bits=1,
            condition_values=(1,),
        ),
        captured_at=_CAPTURED_AT,
        ttl=timedelta(hours=1),
        provider_version="test-sdk-1",
        target_revision="calibration-1",
        environment_id="test",
        evidence_id="declaration",
        evidence_sha256="a" * 64,
    )
    source = FactSource(kind="provider_declared", ref="declaration")
    return replace(
        snapshot,
        snapshot_id="",
        facts=(
            *snapshot.facts,
            CapabilityFact(
                name="gates.native",
                value=gates,
                support_status=SupportStatus.VERIFIED,
                fact_exposure=FactExposure.DECLARED,
                source=source,
            ),
            CapabilityFact(
                name="qubits.physical_capacity",
                value=capacity,
                support_status=SupportStatus.VERIFIED,
                fact_exposure=FactExposure.DECLARED,
                source=source,
            ),
        ),
    )


def _connectivity() -> dict[str, str]:
    return {
        "coupling_edge:0--1": "supported",
        "coupling_edge:1--2": "supported",
    }


def _evaluate(*, snapshot=None, connectivity=None):
    source = _source()
    requirements = dynamic_remote_requirements(source)
    candidate = snapshot or _snapshot()
    routed = route_dynamic_circuit(source, CouplingMap.line(3))
    return evaluate_dynamic_routing_rematch(
        requirements,
        routed,
        presented_source_requirement_set_id=requirements.requirement_set_id,
        requested_target_identity=candidate.target_identity,
        preflight_snapshot_id=candidate.snapshot_id,
        postrouting_snapshot=candidate,
        evaluated_at=_EVALUATED_AT,
        coupling_map=CouplingMap.line(3),
        connectivity_facts=_connectivity() if connectivity is None else connectivity,
        provider_legality="accepted",
    )


def test_legal_offline_route_rematches_without_external_activity() -> None:
    decision = _evaluate()

    assert decision.compatible
    assert decision.rematch.blocker_codes == ()
    assert decision.route_requirements == (
        "coupling_edge:0--1",
        "coupling_edge:1--2",
        "operation:cx",
        "operation:swap",
        "physical_qubit:0",
        "physical_qubit:1",
        "physical_qubit:2",
    )
    assert len(decision.routed_artifact_sha256) == 64
    assert len(decision.routing_evidence_sha256) == 64


def test_route_induced_gate_and_capacity_requirements_fail_closed() -> None:
    decision = _evaluate(snapshot=_snapshot(gates=("cx",), capacity=2))

    assert decision.rematch.blocker_codes == (
        "dynamic_route_capability_fact_unsupported:operation:swap",
        "dynamic_route_capability_fact_unsupported:physical_qubit:2",
    )


def test_connectivity_cannot_be_inferred_from_coarse_capacity() -> None:
    decision = _evaluate(connectivity={})

    assert decision.rematch.blocker_codes == (
        "dynamic_route_capability_fact_missing:coupling_edge:0--1",
        "dynamic_route_capability_fact_missing:coupling_edge:1--2",
    )


def test_route_fact_cannot_override_core_snapshot_evidence() -> None:
    source = _source()
    requirements = dynamic_remote_requirements(source)
    snapshot = _snapshot()

    with pytest.raises(ValueError, match="cannot be overridden"):
        evaluate_dynamic_routing_rematch(
            requirements,
            route_dynamic_circuit(source, CouplingMap.line(3)),
            presented_source_requirement_set_id=requirements.requirement_set_id,
            requested_target_identity=snapshot.target_identity,
            preflight_snapshot_id=snapshot.snapshot_id,
            postrouting_snapshot=snapshot,
            evaluated_at=_EVALUATED_AT,
            coupling_map=CouplingMap.line(3),
            connectivity_facts={"operation:swap": "supported"},
            provider_legality="accepted",
        )


def test_routing_rematch_bridge_is_not_public() -> None:
    assert "evaluate_dynamic_routing_rematch" not in fq.__all__
    assert not hasattr(deployment, "evaluate_dynamic_routing_rematch")
