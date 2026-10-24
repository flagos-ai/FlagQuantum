"""Candidate contract for fail-closed dynamic post-routing rematch."""

from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest
import torch

import flagquantum as fq
import flagquantum._api as fq_api
import flagquantum.errors as fqe
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.post_routing import _evaluate_post_routing_policy

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "contracts" / "dynamic-post-routing-rematch-v1-candidate.json"


def _contract() -> dict[str, Any]:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError("candidate contract must be a JSON object")
    return cast(dict[str, Any], payload)


def _feedback_program() -> DynamicCircuit:
    circuit = DynamicCircuit(2)
    circuit.gate("h", 0)
    circuit.measure(0, classical_bit=0)
    return circuit.conditional("x", 1, classical_bit=0)


def test_contract_records_exact_approval_and_inactive_stage() -> None:
    contract = _contract()

    assert contract["status"] == "approved_for_staged_implementation"
    assert contract["approval"] == {
        "approved_by": "API owner via explicit user directive",
        "approved_at": "2026-10-24",
        "directive": ("approve API-CHANGE-089-DYNAMIC-POST-ROUTING-CAPABILITY-REMATCH"),
        "approved_scope": (
            "credential-free post-routing rematch contract and staged implementation"
        ),
    }
    assert contract["delivery_stage"] == {
        "name": "failing_first_contract",
        "runtime_activation": False,
        "provider_activation": False,
        "network_access": False,
        "task_submission": False,
        "public_behavior_change": False,
        "successor": "pure_runtime_rematch",
    }


def test_contract_reuses_existing_authorities_without_public_change() -> None:
    contract = _contract()

    assert contract["existing_authorities"] == {
        "program": "flagquantum.dynamic.DynamicCircuit",
        "semantic_ir": "flagquantum.CircuitIR@1.0",
        "source_requirements": ("flagquantum.core.target_capabilities.RequirementSet"),
        "target_snapshot": (
            "flagquantum.core.target_capabilities.TargetCapabilitySnapshot"
        ),
        "dynamic_requirement_codec": "org.flagquantum.dynamic-remote-v1",
        "routing_evidence": "flagquantum.deployment.routing-evidence.v1",
        "execution": "flagquantum.run",
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
            "positive_remote_target_enabled",
        )
    )
    assert compatibility["dynamic_circuit_and_circuit_remain_distinct"] is True
    assert "post_routing_rematch" not in fq.__all__


def test_contract_freezes_monotonic_identity_and_decision_order() -> None:
    contract = _contract()

    assert contract["identity_inputs"] == [
        "source_requirement_set_id",
        "requested_target_identity",
        "preflight_snapshot_id",
        "postrouting_snapshot_id",
        "routed_artifact_sha256",
        "routing_evidence_sha256",
    ]
    assert contract["monotonicity"] == {
        "source_requirement_set_is_immutable": True,
        "effective_requirements": (
            "preserved source requirements plus route-induced requirements"
        ),
        "source_requirement_may_disappear": False,
        "route_requirement_may_bypass_snapshot_match": False,
        "coarse_dynamic_boolean_sufficient": False,
        "silent_snapshot_refresh_allowed": False,
        "silent_target_substitution_allowed": False,
        "silent_local_fallback_allowed": False,
    }
    assert contract["decision_order"] == [
        "source_requirement_identity",
        "source_semantic_preservation",
        "target_identity",
        "snapshot_identity",
        "snapshot_freshness",
        "route_induced_capability_facts",
        "provider_specific_legality",
    ]


def test_contract_freezes_precise_blocker_vocabulary() -> None:
    assert _contract()["blocker_codes"] == {
        "source_requirement_identity_missing": (
            "dynamic_source_requirement_identity_missing"
        ),
        "source_requirement_identity_changed": (
            "dynamic_source_requirement_identity_changed"
        ),
        "source_semantic_lost": "dynamic_source_semantic_lost:<requirement>",
        "target_identity_mismatch": ("dynamic_postrouting_target_identity_mismatch"),
        "snapshot_identity_mismatch": (
            "dynamic_postrouting_snapshot_identity_mismatch"
        ),
        "snapshot_stale": "dynamic_postrouting_snapshot_stale",
        "route_fact_missing": ("dynamic_route_capability_fact_missing:<requirement>"),
        "route_fact_unknown": ("dynamic_route_capability_fact_unknown:<requirement>"),
        "route_fact_unmeasured": (
            "dynamic_route_capability_fact_unmeasured:<requirement>"
        ),
        "route_fact_unsupported": (
            "dynamic_route_capability_fact_unsupported:<requirement>"
        ),
        "route_fact_not_exposed": (
            "dynamic_route_capability_fact_not_exposed:<requirement>"
        ),
        "provider_legality_unavailable": ("dynamic_provider_legality_unavailable"),
        "provider_legality_rejected": "dynamic_provider_legality_rejected",
    }


def test_candidate_vectors_freeze_reference_rematch_semantics() -> None:
    for vector in _contract()["conformance_vectors"]:
        inputs = {key: value for key, value in vector.items() if key != "name"}
        expected = tuple(inputs.pop("expected_blockers"))

        assert _evaluate_post_routing_policy(**inputs) == expected, vector["name"]


def test_contract_guarantees_no_external_boundary_or_fallback() -> None:
    boundaries = _contract()["external_boundaries"]

    assert boundaries
    assert all(value is False for value in boundaries.values())


def test_current_user_workflow_still_refuses_before_external_activity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported: list[str] = []

    def tracked_import(name: str, package: str | None = None) -> ModuleType:
        imported.append(name)
        return import_module(name, package)

    def reject_compile(*args: object, **kwargs: object) -> None:
        raise AssertionError("refusal must happen before compilation")

    monkeypatch.setattr(fq_api, "import_module", tracked_import)
    monkeypatch.setattr(fq_api, "compile", reject_compile)
    before = torch.random.get_rng_state()

    with pytest.raises(fqe.CapabilityError, match="dynamic_remote_preflight_failed"):
        fq.run(_feedback_program(), target="quafu:Baihua", shots=8)

    assert not any(name.startswith(".remote") for name in imported)
    assert torch.equal(torch.random.get_rng_state(), before)
