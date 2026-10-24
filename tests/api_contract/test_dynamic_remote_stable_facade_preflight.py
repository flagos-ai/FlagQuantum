"""Stable-facade admission for remote dynamic circuits."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
import torch

import flagquantum as fq
import flagquantum._api as fq_api
import flagquantum.errors as fqe
from flagquantum.core.target_capabilities import TargetCapabilitySnapshot
from flagquantum.deployment import CloudBackendProfile
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.conformance import DynamicFeatureSet
from flagquantum.runtime.dynamic.remote_capabilities import dynamic_remote_snapshot
from flagquantum.runtime.dynamic.remote_preflight import (
    evaluate_dynamic_remote_preflight,
)

pytestmark = pytest.mark.unit

_AT = datetime(2026, 10, 24, 1, 0, tzinfo=timezone.utc)


def _feedback_program() -> DynamicCircuit:
    circuit = DynamicCircuit(2).h(0).measure(0, classical_bit=0)
    return circuit.conditional("x", 1, classical_bit=0)


def _declared_snapshot(
    *,
    name: str = "iqm-test-device",
) -> TargetCapabilitySnapshot:
    backend = CloudBackendProfile(
        provider="amazon-braket",
        name=name,
        n_qubits=8,
        supports_dynamic_circuits=True,
        dynamic_dialect="braket_iqm",
    )
    features = DynamicFeatureSet(
        name="declared-test-features",
        supported_conditional_gates=("x",),
        max_condition_bits=1,
        condition_values=(0, 1),
    )
    return dynamic_remote_snapshot(
        backend,
        features=features,
        captured_at=_AT,
        ttl=timedelta(hours=1),
        provider_version="test-sdk-1",
        target_revision="calibration-1",
        environment_id="test-region",
        evidence_id="device-declaration",
        evidence_sha256="a" * 64,
    )


@pytest.mark.parametrize(
    "target",
    (
        "quafu:Baihua",
        "azure:quantinuum.qpu.h2-1",
        "amazon-braket:non-iqm-device",
        "amazon-braket:iqm-device-without-snapshot",
        "unknown-provider:unknown-device",
    ),
)
def test_stable_remote_dynamic_targets_fail_before_external_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    imported: list[str] = []
    original_import_module = fq_api.import_module

    def tracked_import(name: str, package: str | None = None) -> object:
        imported.append(name)
        return original_import_module(name, package)

    def reject_compile(*args: object, **kwargs: object) -> None:
        raise AssertionError("dynamic refusal must occur before compilation")

    monkeypatch.setattr(fq_api, "import_module", tracked_import)
    monkeypatch.setattr(fq_api, "compile", reject_compile)
    before = torch.random.get_rng_state()

    with pytest.raises(fqe.CapabilityError) as raised:
        fq.run(
            _feedback_program(),
            compiler="qsteed",
            target=target,
            shots=128,
        )

    message = str(raised.value)
    assert f"target={target}" in message
    assert "dynamic_capability_fact_missing:mid_circuit_measurement" in message
    assert "missing_fact:qubits.logical_capacity" in message
    assert not any(name.startswith(".remote") for name in imported)
    assert torch.equal(torch.random.get_rng_state(), before)


def test_remote_dynamic_execution_requires_one_explicit_target() -> None:
    with pytest.raises(
        fqe.CapabilityError,
        match="dynamic_remote_target_required",
    ):
        fq.run(_feedback_program(), compiler="qsteed", shots=8)

    with pytest.raises(
        fqe.CapabilityError,
        match="dynamic_remote_target_invalid",
    ):
        fq.run(_feedback_program(), target="quafu", shots=8)

    with pytest.raises(
        fqe.CapabilityError,
        match="dynamic_remote_target_invalid",
    ):
        fq.run(_feedback_program(), target=" quafu:Baihua", shots=8)


def test_remote_refusal_preserves_the_explicit_target_spelling() -> None:
    with pytest.raises(fqe.CapabilityError) as raised:
        fq.run(_feedback_program(), target="Quafu:Baihua", shots=8)

    assert "target=Quafu:Baihua" in str(raised.value)


def test_preflight_reuses_requirement_snapshot_matcher_for_inspection() -> None:
    snapshot = _declared_snapshot()

    decision = evaluate_dynamic_remote_preflight(
        _feedback_program(),
        target="amazon-braket:iqm-test-device",
        snapshot=snapshot,
        evaluated_at=_AT + timedelta(minutes=1),
    )

    assert decision.compatible
    assert decision.blocker_codes == ()
    assert decision.target == "amazon-braket:iqm-test-device"
    assert decision.requirement_set_id
    assert decision.snapshot_id == snapshot.snapshot_id


def test_preflight_reports_wrong_snapshot_identity_without_fallback() -> None:
    snapshot = _declared_snapshot(name="other-device")

    decision = evaluate_dynamic_remote_preflight(
        _feedback_program(),
        target="amazon-braket:requested-device",
        snapshot=snapshot,
        evaluated_at=_AT + timedelta(minutes=1),
    )

    assert not decision.compatible
    assert "dynamic_target_identity_mismatch" in decision.blocker_codes


def test_experimental_assessment_and_stable_refusal_share_one_matcher() -> None:
    circuit = _feedback_program()
    backend = CloudBackendProfile(
        provider="quafu",
        name="Baihua",
        n_qubits=156,
        basis_gates=("x",),
        supports_openqasm=True,
        supports_dynamic_circuits=True,
        max_classical_bits=1,
    )

    report = fq.experimental.dynamic.assess_dynamic_backend(circuit, backend)
    decision = evaluate_dynamic_remote_preflight(circuit, target="quafu:Baihua")

    assert not report.compatible
    assert report.blockers == decision.blocker_codes
    with pytest.raises(fqe.CapabilityError) as raised:
        fq.run(circuit, target="quafu:Baihua", shots=8)
    assert str(raised.value).endswith(", ".join(report.blockers))


def test_local_dynamic_execution_and_static_remote_dispatch_are_not_reclassified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local = fq.run(_feedback_program(), shots=4)
    assert local.samples.shape == (4, 2)

    marker = object()

    def execute_static(*args: object, **kwargs: object) -> object:
        return marker

    monkeypatch.setattr(
        "flagquantum.remote.qpu.execution.execute_quafu",
        execute_static,
    )
    result = fq.run(fq.Circuit(1).x(0), target="quafu:Baihua", shots=1024)

    assert result is marker
