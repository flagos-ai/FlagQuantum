from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]
_CONTRACT = _ROOT / "contracts" / "research-capability-services-v1-candidate.json"


def _candidate() -> dict[str, object]:
    payload = json.loads(_CONTRACT.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_candidate_names_the_transport_neutral_boundary() -> None:
    candidate = _candidate()

    assert candidate["status"] == "approved_for_external_adapter"
    assert candidate["stable_core_change"] is False
    assert candidate["transport_neutral"] is True
    assert candidate["namespace"] == "flagquantum.services"
    assert candidate["authorities"] == {
        "research_capability_contract": "flagquantum",
        "target_requirement_matching": "flagquantum.core.target_capabilities",
        "product_maturity": "capability-maturity.toml",
        "protocol_adaptation": "external",
        "natural_language_reasoning": "external",
        "tenant_and_durable_task_control": "external",
    }
    assert candidate["existing_contracts"] == {
        "target_requirements_and_facts": "ARCH-003",
        "execution_request_and_policy": "ARCH-004",
        "execution_result_and_evidence": "ARCH-005",
    }


def test_candidate_has_exactly_the_p0_capability_identities() -> None:
    capabilities = _candidate()["capabilities"]
    assert isinstance(capabilities, list)

    by_id = {item["capability_id"]: item for item in capabilities}
    assert tuple(by_id) == (
        "ground_state.solve.vqe",
        "ground_state.solve.exact",
        "validate.energy_variance",
    )
    assert by_id["ground_state.solve.vqe"]["mandatory_validators"] == [
        "ground_state.solve.exact",
        "validate.energy_variance",
    ]
    assert {item["implementation_status"] for item in capabilities} == {
        "available_experimental_service"
    }
    assert {item["implementation"] for item in capabilities} == {
        "flagquantum.services.execute_research_capability"
    }
    assert by_id["ground_state.solve.vqe"]["fallback_allowed"] is True
    assert by_id["ground_state.solve.vqe"]["fallback_policy"] == (
        "explicit_provider_unavailable_only"
    )
    assert all(
        item["fallback_allowed"] is False
        for item in capabilities
        if item["capability_id"] != "ground_state.solve.vqe"
    )


def test_candidate_uses_domain_specific_service_names() -> None:
    assert _candidate()["operations"] == [
        "search_research_capabilities",
        "describe_research_capability",
        "estimate_research_capability",
        "validate_research_capability_input",
        "execute_research_capability",
        "validate_research_evidence",
    ]


def test_candidate_pins_the_exact_review_signatures() -> None:
    from flagquantum import services

    candidate = _candidate()
    expected = candidate["operation_signatures"]

    assert set(expected) == set(candidate["operations"])
    assert {
        name: str(inspect.signature(getattr(services, name))) for name in expected
    } == expected


def test_candidate_pins_schema_resources_to_installed_flagquantum() -> None:
    from flagquantum import services

    candidate = _candidate()
    expected = candidate["schema_resource_operations"]

    assert {
        name: str(inspect.signature(getattr(services, name))) for name in expected
    } == expected
    assert candidate["schema_resource_policy"] == {
        "authority": "installed_flagquantum",
        "adapter_may_copy_or_redefine": False,
        "unknown_schema_ids_fail_closed": True,
        "returned_documents_are_isolated_copies": True,
        "json_schema_dialect": "https://json-schema.org/draft/2020-12/schema",
    }


def test_candidate_pins_every_p0_schema_identity() -> None:
    schemas = _candidate()["schema_identities"]

    assert schemas == {
        "pauli_sum": "flagquantum.pauli_sum.v1",
        "manifest": "flagquantum.research_capability_manifest.v1",
        "vqe_request": "flagquantum.ground_state_vqe_request.v1",
        "vqe_result": "flagquantum.ground_state_vqe_result.v1",
        "exact_request": "flagquantum.exact_ground_state_request.v1",
        "exact_result": "flagquantum.exact_sector_reference.v1",
        "variance_request": "flagquantum.energy_variance_request.v1",
        "variance_result": "flagquantum.energy_variance_reference.v1",
        "validation_report": "flagquantum.research_validation_report.v1",
        "structured_error": "flagquantum.research_capability_error.v1",
        "evidence_bundle": "flagquantum.research_capability_evidence_bundle.v1",
    }


def test_evidence_bundle_aggregates_without_replacing_existing_authorities() -> None:
    evidence = _candidate()["evidence_bundle"]
    assert evidence["composition"] == "application_service_aggregation"
    assert evidence["replaces_execution_result"] is False
    assert evidence["replaces_arch_005_execution_record"] is False
    assert evidence["required_fields"] == [
        "schema",
        "run_id",
        "capability",
        "status",
        "result",
        "metrics",
        "artifacts",
        "execution",
        "validation",
        "attempts",
        "warnings",
        "provenance",
        "error",
    ]
    assert evidence["schema_validation"] == {
        "unknown_top_level_fields_fail": True,
        "unknown_capability_fields_fail": True,
        "unknown_provenance_fields_fail": True,
        "unsupported_schema_versions_fail": True,
        "installed_capability_identity_required": True,
        "provenance_is_content_hashed": True,
    }
    assert evidence["provenance_required_fields"] == [
        "package",
        "package_version",
        "python_version",
        "torch_version",
        "input_sha256",
        "deterministic",
        "random_seed",
        "fallback_used",
        "content_sha256",
    ]
    assert evidence["execution_required_observability"] == [
        "target",
        "execution_mode",
        "started_at",
        "completed_at",
        "duration_seconds",
        "provider_order",
        "fallback_used",
        "stage_durations_seconds",
        "call_counts",
    ]


def test_status_and_failure_stages_are_closed_for_p0() -> None:
    candidate = _candidate()

    assert candidate["statuses"] == [
        "succeeded",
        "unsupported",
        "validation_failed",
        "failed",
    ]
    assert candidate["error_stages"] == [
        "catalog_match",
        "input_validation",
        "resource_estimation",
        "environment_preflight",
        "compilation",
        "execution",
        "result_parsing",
        "scientific_validation",
        "artifact_persistence",
    ]


def test_p0_allows_only_explicit_provider_fallback() -> None:
    p0_scope = _candidate()["p0_scope"]
    assert p0_scope["optimizer_convergence_diagnostic"] == {
        "window": 5,
        "energy_span_tolerance_hartree": 0.000005,
        "budget_exhaustion_requires_unestablished_convergence": True,
        "substitutes_for_scientific_validation": False,
    }
    assert p0_scope["fallback_authorizations"] == {
        "provider": "explicit_only",
        "backend": False,
        "device": False,
        "cpu": False,
        "precision": False,
        "algorithm": False,
        "approximation": False,
    }


def test_approved_external_adapter_cannot_reimplement_domain_logic() -> None:
    candidate = _candidate()

    assert "mcp_sdk_dependency" in candidate["prohibited_in_main_repository"]
    assert candidate["external_adapter_gate"] == {
        "description_allowed_before_maturity_binding": True,
        "execution_tool_allowed_before_api_approval": False,
        "domain_logic_reimplementation_allowed": False,
    }
    assert candidate["approval"] == {
        "approval_token": (
            "approve API_CHANGE_PROPOSAL_070_RESEARCH_CAPABILITY_SERVICES"
        ),
        "approval_record_target": (
            "docs/api-changes/FQ-RESEARCH-CAPABILITY-SERVICES-20261013.md"
        ),
        "api_owner": "approved",
        "services_domain_owner": "approved",
        "approved_at": "2026-10-13",
        "approved_by": (
            "API owner and Services domain owner via explicit user directive"
        ),
        "external_execution_adapter_authorized": True,
        "maturity_owner": "approved",
        "approved_maturity": "production_supported",
        "released_version": None,
        "external_compatible_range": None,
        "release_artifact_gate": (
            "blocked: PyPI 0.3.0rc3 and 0.3.0 wheels omit the approved research service API"
        ),
        "requested_decision": (
            "approve the exact namespace operations, signatures, value shapes, "
            "schemas, fail-closed behavior, and experimental maturity binding"
        ),
    }
    assert candidate["provider_conformance"] == {
        "native_and_fake_share_consumer": True,
        "semantic_projection_equivalent": True,
        "provider_exception_maps_to_structured_failure": True,
        "fake_registration_public": False,
        "multi_provider_routing_deferred_to_m3": False,
        "test": "tests/team/services/test_research_provider_conformance.py",
    }
    assert candidate["provider_routing"] == {
        "status": "m3_route_explanation_implemented",
        "capability_scope": ["ground_state.solve.vqe"],
        "default_provider_order": ["flagquantum-native"],
        "provider_order_is_explicit": True,
        "multiple_providers_require_authorization": True,
        "fallback_trigger": "provider_unavailable_only",
        "unknown_provider_is_audited": True,
        "ordinary_execution_failure_triggers_fallback": False,
        "scientific_validation_failure_triggers_fallback": False,
        "attempt_history_is_content_hashed": True,
        "provider_registration_public": False,
        "installed_external_providers": ["qiskit-aer", "pennylane-lightning"],
        "external_provider_dependency_is_optional": True,
        "external_provider_gradient_method": "exact_parameter_shift",
        "backend_executions_reported_separately": True,
        "candidate_scoring": "caller_order_rank_without_reordering",
        "route_explanation_is_content_hashed": True,
        "provider_backend_execution_estimate": True,
        "tests": [
            "tests/team/services/test_research_provider_routing.py",
            "tests/team/ecosystem/test_qiskit_research_provider.py",
            "tests/optional/test_qiskit_aer_backend.py",
            "tests/team/ecosystem/test_pennylane_research_provider.py",
            "tests/test_pennylane_interop.py",
            "tests/optional/test_research_external_provider_parity.py",
        ],
    }


def test_approval_names_an_existing_record_and_exact_token() -> None:
    approval = _candidate()["approval"]
    record = _ROOT / approval["approval_record_target"]

    assert record.is_file()
    source = record.read_text(encoding="utf-8")
    assert approval["approval_token"] in source
    assert "Decision: approved" in source
