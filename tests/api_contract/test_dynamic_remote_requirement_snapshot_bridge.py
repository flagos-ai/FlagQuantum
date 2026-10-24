"""Conformance for the internal dynamic requirement/snapshot bridge."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

import flagquantum as fq
import flagquantum.runtime.dynamic as dynamic_runtime
from flagquantum.deployment import CloudBackendProfile
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.conformance import DynamicFeatureSet
from flagquantum.runtime.dynamic.remote_capabilities import (
    DYNAMIC_REMOTE_EXTENSION,
    dynamic_remote_requirements,
    dynamic_remote_snapshot,
    match_dynamic_remote_capabilities,
)

pytestmark = pytest.mark.unit

_CAPTURED_AT = datetime(2026, 10, 23, 8, 0, tzinfo=timezone.utc)
_EVALUATED_AT = _CAPTURED_AT + timedelta(minutes=1)


def _backend(**changes: object) -> CloudBackendProfile:
    values: dict[str, object] = {
        "provider": "amazon-braket",
        "name": "iqm-test-device",
        "n_qubits": 8,
        "supports_dynamic_circuits": True,
        "dynamic_dialect": "braket_iqm",
    }
    values.update(changes)
    return CloudBackendProfile(**values)  # type: ignore[arg-type]


def _features(**changes: object) -> DynamicFeatureSet:
    values: dict[str, object] = {
        "name": "test-dynamic-target",
        "supported_conditional_gates": ("x", "h"),
        "max_condition_bits": 2,
        "condition_values": (0, 1),
        "supports_disjunctive_conditions": True,
    }
    values.update(changes)
    return DynamicFeatureSet(**values)  # type: ignore[arg-type]


def _feedback_program(*, gate: str = "x", equals: int = 1) -> DynamicCircuit:
    circuit = DynamicCircuit(2).measure(0, classical_bit=0)
    return circuit.conditional(gate, 1, classical_bit=0, equals=equals)


def _snapshot(
    *,
    backend: CloudBackendProfile | None = None,
    features: DynamicFeatureSet | None = None,
    fact_states: dict[str, str] | None = None,
    evidence_state: str = "declared",
    ttl: timedelta = timedelta(hours=1),
):
    return dynamic_remote_snapshot(
        backend or _backend(),
        features=_features() if features is None else features,
        captured_at=_CAPTURED_AT,
        ttl=ttl,
        provider_version="test-sdk-1",
        target_revision="calibration-7",
        environment_id="us-east-1",
        evidence_id="device-declaration",
        evidence_sha256="a" * 64,
        evidence_state=evidence_state,
        fact_states=fact_states,
    )


def _codes(result: object) -> tuple[str, ...]:
    return tuple(blocker.code for blocker in result.blockers)  # type: ignore[attr-defined]


def test_requirements_are_derived_from_ordered_circuit_ir() -> None:
    circuit = DynamicCircuit(3)
    circuit.measure(0, classical_bit=0)
    circuit.measure(1, classical_bit=1)
    circuit.measure(0, classical_bit=0)
    circuit.reset(2)
    circuit.conditional("x", 2, any_of=({0: 1}, {1: 0}))

    requirements = dynamic_remote_requirements(
        circuit,
        require_mid_circuit_measurements=True,
    )
    payload = requirements.to_dict()["extensions"][DYNAMIC_REMOTE_EXTENSION]

    assert payload["source"] == "ordered CircuitIR instructions"
    assert payload["required"] == {
        "mid_circuit_measurement": True,
        "reset": True,
        "repeated_measurement": True,
        "classical_feed_forward": True,
        "supported_conditional_gates": ["x"],
        "max_condition_bits": 1,
        "condition_values": [0, 1],
        "supports_disjunctive_conditions": True,
        "returns_mid_circuit_measurements": True,
    }
    assert {item.name for item in requirements.requirements} == {
        "target.class",
        "qubits.logical_capacity",
    }
    assert requirements == type(requirements).from_json(requirements.to_json())


def test_declared_device_scoped_features_match_without_network_access() -> None:
    requirements = dynamic_remote_requirements(_feedback_program())
    snapshot = _snapshot()

    result = match_dynamic_remote_capabilities(
        requirements,
        snapshot,
        evaluated_at=_EVALUATED_AT,
        expected_target_identity=snapshot.target_identity,
    )

    assert result.executable
    assert result.blockers == ()
    assert snapshot.target_identity.target_id == "amazon-braket:iqm-test-device"
    assert snapshot == type(snapshot).from_json(snapshot.to_json())
    extension = snapshot.to_dict()["extensions"][DYNAMIC_REMOTE_EXTENSION]
    assert extension["facts"]["mid_circuit_measurement"] == {
        "state": "supported",
        "value": True,
        "evidence_state": "declared",
        "source_ref": "device-declaration",
    }


@pytest.mark.parametrize(
    ("evidence_state", "level", "exposure"),
    (
        ("declared", "basic", "declared"),
        ("verified", "observable", "observed"),
        ("certified", "certification", "observed"),
    ),
)
def test_evidence_states_reuse_core_levels_and_exposures(
    evidence_state: str,
    level: str,
    exposure: str,
) -> None:
    snapshot = _snapshot(evidence_state=evidence_state)

    assert snapshot.evidence_refs[0].level.value == level
    assert {fact.fact_exposure.value for fact in snapshot.facts} == {exposure}
    dynamic_facts = snapshot.to_dict()["extensions"][DYNAMIC_REMOTE_EXTENSION]["facts"]
    assert {fact["evidence_state"] for fact in dynamic_facts.values()} == {
        evidence_state
    }


def test_coarse_backend_boolean_cannot_admit_execution() -> None:
    requirements = dynamic_remote_requirements(_feedback_program())
    coarse_only = dynamic_remote_snapshot(
        _backend(supports_dynamic_circuits=True),
        features=None,
        captured_at=_CAPTURED_AT,
        ttl=timedelta(hours=1),
        provider_version="test-sdk-1",
        target_revision="calibration-7",
        environment_id="us-east-1",
        evidence_id="device-declaration",
        evidence_sha256="b" * 64,
    )

    result = match_dynamic_remote_capabilities(
        requirements,
        coarse_only,
        evaluated_at=_EVALUATED_AT,
    )

    assert not result.executable
    assert set(_codes(result)) == {
        "dynamic_capability_fact_missing:mid_circuit_measurement",
        "dynamic_capability_fact_missing:classical_feed_forward",
        "dynamic_capability_fact_missing:supported_conditional_gates",
        "dynamic_capability_fact_missing:max_condition_bits",
        "dynamic_capability_fact_missing:condition_values",
    }


@pytest.mark.parametrize(
    ("state", "code"),
    (
        ("missing", "dynamic_capability_fact_missing:mid_circuit_measurement"),
        ("unknown", "dynamic_capability_fact_unknown:mid_circuit_measurement"),
        ("unmeasured", "dynamic_capability_fact_unmeasured:mid_circuit_measurement"),
        (
            "unsupported",
            "dynamic_capability_fact_unsupported:mid_circuit_measurement",
        ),
        (
            "not_exposed",
            "dynamic_capability_fact_not_exposed:mid_circuit_measurement",
        ),
    ),
)
def test_non_accepted_fact_states_fail_with_distinct_blockers(
    state: str,
    code: str,
) -> None:
    requirements = dynamic_remote_requirements(_feedback_program())
    snapshot = _snapshot(fact_states={"mid_circuit_measurement": state})

    result = match_dynamic_remote_capabilities(
        requirements,
        snapshot,
        evaluated_at=_EVALUATED_AT,
    )

    assert not result.executable
    assert code in _codes(result)


def test_stale_snapshot_and_wrong_target_are_named_separately() -> None:
    requirements = dynamic_remote_requirements(_feedback_program())
    snapshot = _snapshot(ttl=timedelta(minutes=1))
    other_target = replace(snapshot.target_identity, target_id="amazon-braket:other")

    result = match_dynamic_remote_capabilities(
        requirements,
        snapshot,
        evaluated_at=_CAPTURED_AT + timedelta(minutes=2),
        expected_target_identity=other_target,
    )

    assert not result.executable
    assert "dynamic_capability_snapshot_stale" in _codes(result)
    assert "dynamic_target_identity_mismatch" in _codes(result)


def test_capacity_and_dynamic_value_mismatches_remain_distinct() -> None:
    requirements = dynamic_remote_requirements(
        DynamicCircuit(3)
        .measure(0, classical_bit=0)
        .conditional("h", 2, classical_bit=0, equals=0)
    )
    snapshot = _snapshot(
        backend=_backend(n_qubits=2),
        features=_features(
            supported_conditional_gates=("x",),
            condition_values=(1,),
        ),
    )

    result = match_dynamic_remote_capabilities(
        requirements,
        snapshot,
        evaluated_at=_EVALUATED_AT,
    )

    assert not result.executable
    assert "value_mismatch" in _codes(result)
    assert "dynamic_capability_value_mismatch:supported_conditional_gates" in _codes(
        result
    )
    assert "dynamic_capability_value_mismatch:condition_values" in _codes(result)


def test_requested_mid_circuit_data_is_not_invented() -> None:
    requirements = dynamic_remote_requirements(
        _feedback_program(),
        require_mid_circuit_measurements=True,
    )
    snapshot = _snapshot(features=_features(returns_mid_circuit_measurements=False))

    result = match_dynamic_remote_capabilities(
        requirements,
        snapshot,
        evaluated_at=_EVALUATED_AT,
    )

    assert not result.executable
    assert _codes(result) == (
        "dynamic_capability_fact_unsupported:returns_mid_circuit_measurements",
    )


def test_bridge_stays_internal_and_rejects_contradictory_provider_facts() -> None:
    assert "dynamic_remote_requirements" not in fq.__all__
    assert "dynamic_remote_requirements" not in dynamic_runtime.__all__

    with pytest.raises(ValueError, match="contradict"):
        dynamic_remote_snapshot(
            _backend(supports_dynamic_circuits=False),
            features=_features(),
            captured_at=_CAPTURED_AT,
            ttl=timedelta(hours=1),
            provider_version="test-sdk-1",
            target_revision="calibration-7",
            environment_id="us-east-1",
            evidence_id="device-declaration",
            evidence_sha256="c" * 64,
        )
