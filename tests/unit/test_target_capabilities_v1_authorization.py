from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUTHORIZATION = (
    ROOT / "contracts/target-capabilities-v1-implementation-authorization.json"
)
ADR = (
    ROOT
    / "docs/architecture/decisions/ARCH_003_CAPABILITY_REQUIREMENT_DISCOVERY_EVIDENCE.md"
)
DECISION = ROOT / "docs/development/VNEXT_PHASE2_CAPABILITIES_DECISION.md"


def _authorization() -> dict[str, object]:
    return json.loads(AUTHORIZATION.read_text(encoding="utf-8"))


def test_capability_v1_has_one_three_type_core_lineage() -> None:
    payload = _authorization()

    assert payload["status"] == "authorized_internal_minimal_implementation"
    assert set(payload["canonical_types"]) == {
        "capability_requirement",
        "requirement_set",
        "target_capability_snapshot",
    }
    assert payload["canonical_types"]["capability_requirement"]["role"] == (
        "single_predicate"
    )
    assert payload["no_fourth_capability_type"] is True
    assert payload["platform_candidate_disposition"] == "adapter_input_only"


def test_fact_snapshot_and_requirement_fields_are_exact_and_closed() -> None:
    payload = _authorization()

    assert payload["canonical_types"]["capability_requirement"]["fields"] == [
        "name",
        "operator",
        "value",
        "strength",
        "source",
        "minimum_evidence_level",
        "accepted_exposures",
    ]
    assert payload["fact_fields"] == [
        "name",
        "value",
        "support_status",
        "fact_exposure",
        "source",
        "blockers",
    ]
    assert payload["target_identity_fields"] == [
        "target_id",
        "target_class",
        "provider",
        "provider_version",
        "target_revision",
        "environment_id",
    ]
    assert payload["scope_fields"] == [
        "device_ids",
        "dtype",
        "kernel",
        "workload_id",
        "world_size",
        "node_count",
    ]
    assert payload["evidence_reference_fields"] == [
        "evidence_id",
        "sha256",
        "level",
        "scope",
    ]


def test_support_exposure_evidence_and_matching_are_fail_closed() -> None:
    payload = _authorization()

    assert payload["enums"]["support_status"] == [
        "unknown",
        "unmeasured",
        "unsupported",
        "verified",
    ]
    assert payload["enums"]["fact_exposure"] == [
        "observed",
        "declared",
        "not_exposed",
        "unknown",
        "not_applicable",
    ]
    assert payload["enums"]["evidence_level"] == [
        "basic",
        "observable",
        "certification",
    ]
    assert payload["enums"]["requirement_strength"] == [
        "mandatory",
        "preference",
    ]
    assert payload["enums"]["comparison_operator"] == [
        "equals",
        "at_least",
        "at_most",
        "contains_all",
        "covers",
    ]
    assert all(payload["matching_rules"].values())
    assert payload["evidence_aggregation"] == {
        "required_threshold": (
            "strongest_applicable_mandatory_requirement_or_claim_gate"
        ),
        "available_ceiling": "weakest_indispensable_evidence_reference",
        "passes_when": (
            "available_ceiling_at_least_required_threshold_and_every_fact_check_passes"
        ),
        "verified_declared_allowed_only_by_named_validation_profile": True,
        "bare_declaration_defaults_to": "unmeasured_declared",
        "runtime_fact_exposure_required_for": [
            "capacity",
            "latency",
            "physical_route",
            "absence_of_fallback",
        ],
        "runtime_fact_required_exposure": "observed",
    }


def test_precision_and_fallback_axes_cannot_be_collapsed() -> None:
    payload = _authorization()

    assert payload["precision_axes"] == [
        "native_dtype",
        "effective_dtype",
        "storage_dtype",
        "parameter_dtype",
        "accumulator_dtype",
        "software_mechanism",
    ]
    assert payload["precision_rules"] == {
        "software_mechanism_is_separate_axis": True,
        "software_extension_may_satisfy_effective_precision": True,
        "software_extension_may_satisfy_native_precision": False,
        "double_single_is_native_fp64_or_complex128": False,
    }
    profiles = payload["capability_validation_profiles"]
    declared = set(profiles["authoritative_static_declaration_allowed"])
    observed = set(profiles["observed_required"])
    assert declared.isdisjoint(observed)
    assert declared | observed == set(payload["v1_capability_names"])
    assert {
        "memory.available_bytes",
        "precision.native_dtype",
        "precision.effective_dtype",
    }.issubset(observed)
    assert payload["fallback_authorization_axes"] == [
        "backend",
        "device",
        "cpu",
        "precision",
        "algorithm",
        "approximation",
    ]
    fallback = payload["fallback_rules"]
    assert fallback["default"] == "forbidden"
    assert fallback["axes_do_not_imply_each_other"] is True
    assert fallback["cpu_is_independent_candidate"] is True
    assert fallback["replacement_candidate_requires_own_snapshot"] is True
    assert fallback["replacement_candidate_requires_full_rematch"] is True


def test_ownership_deferred_scope_and_non_authorization_are_explicit() -> None:
    payload = _authorization()

    assert payload["ownership"] == {
        "compiler_target_legality": "compiler",
        "platform_discovery": "platform_provider_or_execution_provider",
        "runtime_matching_and_policy": "runtime",
        "simulation_candidates_and_cost_hints": "simulation",
        "execution_evidence": "runtime_and_executing_provider",
        "claim_eligibility": "audit_and_release",
    }
    assert {
        "dynamic_control_and_mid_circuit_operations",
        "checkpoint_and_restart",
        "realtime_sessions_and_latency",
        "full_topology_placement_and_links",
    }.issubset(payload["deferred_from_v1"])
    assert (
        payload["extensions"][
            "unknown_snapshot_extensions_may_satisfy_core_requirement"
        ]
        is False
    )
    assert (
        "product_implementation_changes_in_this_decision_commit"
        in payload["not_authorized"]
    )
    assert "domestic_hardware_verification_claim" in payload["not_authorized"]
    assert payload["legacy_contract_policy"]["modify_historical_contracts"] is False


def test_decision_documents_link_the_machine_authorization() -> None:
    adr = ADR.read_text(encoding="utf-8")
    decision = DECISION.read_text(encoding="utf-8")

    for name in (
        "CapabilityRequirement",
        "RequirementSet",
        "TargetCapabilitySnapshot",
        "target-capabilities-v1-implementation-authorization.json",
    ):
        assert name in adr
        assert name in decision
    assert "PlatformCapabilitySnapshotCandidate" in adr
    assert "第四" in adr
    assert "国产硬件" in adr
