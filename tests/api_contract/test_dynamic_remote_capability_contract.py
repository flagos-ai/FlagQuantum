"""Candidate contract for remote dynamic-circuit capability negotiation.

This stage freezes admission semantics and proves the stable route remains
closed. It performs no discovery, credential lookup, provider construction, or
transport call.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import flagquantum as fq
import flagquantum.errors as fqe
import flagquantum.runtime.dynamic.conformance as fqdc
from flagquantum.deployment import CloudBackendProfile
from flagquantum.dynamic import DynamicCircuit

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "contracts" / "dynamic-remote-capability-v1-candidate.json"


def _contract() -> dict[str, object]:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def _feedback_program(*, gate: str = "x", equals: int = 1) -> DynamicCircuit:
    circuit = DynamicCircuit(2).measure(0, classical_bit=0)
    return circuit.conditional(gate, 1, classical_bit=0, equals=equals)


def test_contract_records_exact_approval_and_inactive_delivery_stage() -> None:
    contract = _contract()

    assert contract["status"] == "approved_for_staged_implementation"
    assert contract["approval"] == {
        "approved_by": "API owner via explicit user directive",
        "approved_at": "2026-10-23",
        "directive": "approve API-CHANGE-086-DYNAMIC-REMOTE-CAPABILITY-CONTRACT",
        "approved_scope": (
            "remote dynamic capability candidate contract and failing-first "
            "conformance baseline"
        ),
    }
    assert contract["delivery_stage"] == {
        "name": "contract_and_conformance_baseline",
        "runtime_activation": False,
        "provider_activation": False,
        "network_access": False,
        "task_submission": False,
        "public_behavior_change": False,
        "successor": "requirement_and_snapshot_bridge",
    }


def test_contract_reuses_authorities_without_public_schema_changes() -> None:
    contract = _contract()

    assert contract["existing_authorities"] == {
        "program": "flagquantum.dynamic.DynamicCircuit",
        "semantic_ir": "flagquantum.CircuitIR@1.0",
        "dynamic_feature_vocabulary": (
            "flagquantum.runtime.dynamic.conformance.DynamicFeatureSet"
        ),
        "target_capability_authority": "flagquantum.core.target_capabilities",
        "provider_projection": "flagquantum.deployment.CloudBackendProfile",
        "provider_boundary": "flagquantum.remote.qpu",
        "execution": "flagquantum.run",
        "result": "flagquantum.ExecutionResult",
    }
    compatibility = contract["compatibility"]
    assert all(
        compatibility[name] is False
        for name in (
            "public_signature_change",
            "root_export_change",
            "circuit_ir_schema_change",
            "execution_plan_schema_change",
            "deployment_package_schema_change",
            "execution_result_schema_change",
            "cloud_backend_profile_schema_change",
        )
    )
    assert "DynamicFeatureSet" not in fq.__all__


def test_contract_freezes_ir_derived_feature_vector() -> None:
    requirements = _contract()["requirement_vector"]

    assert requirements["source"] == "ordered CircuitIR instructions"
    assert requirements["fields"] == [
        "mid_circuit_measurement",
        "reset",
        "repeated_measurement",
        "classical_feed_forward",
        "supported_conditional_gates",
        "max_condition_bits",
        "condition_values",
        "supports_disjunctive_conditions",
        "returns_mid_circuit_measurements",
    ]
    assert requirements["caller_declaration_accepted"] is False
    assert requirements["coarse_dynamic_boolean_sufficient"] is False
    assert requirements["requirements_may_weaken_after_routing"] is False


def test_current_feature_oracle_distinguishes_boolean_hidden_requirements() -> None:
    supports_dynamic = CloudBackendProfile(
        provider="example",
        name="coarse-only",
        n_qubits=2,
        supports_dynamic_circuits=True,
    )
    circuit = _feedback_program(gate="h", equals=0)

    assert supports_dynamic.supports_dynamic_circuits is True
    report = fqdc.assess_dynamic_features(circuit, fqdc.BRAKET_IQM_DYNAMIC_FEATURES)

    assert not report.compatible
    assert report.blockers == (
        "condition_value_unsupported",
        "conditional_gate_unsupported:h",
    )


def test_contract_distinguishes_fact_and_evidence_states() -> None:
    contract = _contract()

    assert contract["fact_states"] == [
        "supported",
        "unsupported",
        "unknown",
        "unmeasured",
        "not_exposed",
        "stale",
        "missing",
    ]
    assert contract["evidence_states"] == {
        "declared": "fresh device-scoped provider declaration",
        "verified": "retained controlled task evidence for the named target",
        "certified": ("independently reviewable evidence satisfying release policy"),
        "stable_provider_promotion_requires": "certified",
        "simulator_success_is_hardware_evidence": False,
        "sdk_syntax_is_hardware_evidence": False,
        "provider_family_name_is_device_evidence": False,
    }
    assert set(contract["blocker_categories"]) == {
        "missing",
        "stale",
        "unknown",
        "unmeasured",
        "unsupported",
        "not_exposed",
        "value_mismatch",
        "target_identity_mismatch",
        "routing_weakened_requirement",
        "result_form_unavailable",
    }


def test_candidate_target_stays_closed_before_provider_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported: list[str] = []
    original_import = __import__

    def record_import(
        name: str,
        globals: object = None,
        locals: object = None,
        fromlist: object = (),
        level: int = 0,
    ) -> object:
        if name.startswith("flagquantum.remote.qpu"):
            imported.append(name)
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", record_import)

    with pytest.raises(fqe.CapabilityError, match="dynamic_remote_preflight"):
        fq.run(
            _feedback_program(),
            target="amazon-braket:<device-arn>",
            shots=128,
        )

    assert imported == []


def test_contract_guarantees_no_side_effects_on_refusal() -> None:
    fail_closed = _contract()["fail_closed"]

    assert fail_closed["mandatory_fact_not_accepted"] == (
        "CapabilityError before compilation, packaging, or submission"
    )
    assert fail_closed["unknown_target"] == ("CapabilityError without local fallback")
    assert fail_closed["rng_consumed_on_refusal"] is False
    assert fail_closed["task_submitted_on_refusal"] is False
    assert fail_closed["charge_possible_on_refusal"] is False
    assert fail_closed["partial_result_on_refusal"] is False


def test_result_contract_never_invents_provider_data() -> None:
    result = _contract()["result_projection"]

    assert result["type"] == "flagquantum.ExecutionResult"
    assert result["mid_circuit_measurements"] == (
        "returned only when the provider supplies them"
    )
    assert result["invent_missing_measurements"] is False
    assert result["provider_native_task_exposed"] is False
    assert result["raw_provider_payload_exposed"] is False
