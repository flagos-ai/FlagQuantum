"""Proposal contract for the Quafu dynamic provider vertical slice."""

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

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "contracts" / "quafu-dynamic-vertical-slice-v1-candidate.json"


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


def test_proposal_requires_approval_and_makes_no_support_claim() -> None:
    contract = _contract()

    assert contract["status"] == "proposed"
    assert contract["approval"] == {
        "required_directive": ("approve API-CHANGE-090-QUAFU-DYNAMIC-VERTICAL-SLICE"),
        "approved": False,
        "authorized_stages": [],
    }
    assert contract["provider_slice"] == {
        "provider": "quafu",
        "candidate_backend": "Baihua",
        "positive_dynamic_support_claim": False,
        "stable_user_entry": "flagquantum.run",
        "public_provider_factory_added": False,
    }


def test_proposal_reuses_authorities_and_freezes_gate_order() -> None:
    contract = _contract()

    assert contract["existing_authorities"] == {
        "program": "flagquantum.dynamic.DynamicCircuit",
        "source_requirements": ("flagquantum.core.target_capabilities.RequirementSet"),
        "target_snapshot": (
            "flagquantum.core.target_capabilities.TargetCapabilitySnapshot"
        ),
        "post_routing_rematch": "flagquantum.runtime.dynamic.post_routing",
        "deployment_package": "flagquantum.deployment.cloud.DeploymentPackage",
        "provider": "flagquantum.remote.qpu.QuafuProvider",
        "execution": "flagquantum.run",
    }
    assert contract["decision_order"] == [
        "source_requirement_identity",
        "target_snapshot_completeness",
        "pre_routing_capability_match",
        "routing_semantic_preservation",
        "post_routing_capability_rematch",
        "quafu_provider_legality",
        "deployment_package_sealing",
        "mocked_exactly_once_task_creation",
        "normalized_result_validation",
    ]


def test_proposal_rejects_coarse_or_incidental_evidence() -> None:
    contract = _contract()

    assert (
        "coarse_supports_dynamic_circuits_boolean"
        in contract["non_authoritative_signals"]
    )
    assert "successful_static_job" in contract["non_authoritative_signals"]
    required = set(contract["authoritative_dynamic_facts"])
    assert {
        "mid_circuit_measurement",
        "classical_feed_forward",
        "conditional_operations",
        "directed_connectivity",
        "program_dialect",
    } <= required


def test_proposal_freezes_exactly_once_mock_boundary() -> None:
    policy = _contract()["exactly_once_policy"]

    assert policy == {
        "initial_transport": "quafu_task_api_v1_mock",
        "maximum_post_requests_per_execution_attempt": 1,
        "idempotency_identity_bound_to_package": True,
        "retry_after_ambiguous_post": False,
        "legacy_fallback_after_post": False,
        "legacy_dynamic_submission_enabled": False,
    }


def test_proposal_authorizes_no_external_or_public_boundary() -> None:
    boundaries = _contract()["proposal_boundaries"]

    assert boundaries
    assert all(value is False for value in boundaries.values())
    assert "QuafuProvider" not in fq.__all__


def test_current_user_workflow_remains_fail_closed(
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
        fq.run(_feedback_program(), target="quafu:Baihua", shots=1024)

    assert not any(name.startswith(".remote") for name in imported)
    assert torch.equal(torch.random.get_rng_state(), before)
