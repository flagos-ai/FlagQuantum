from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.check_domestic_single_card_contract import contract_errors
from tools.validate_domestic_single_card import attestation_errors, evidence_errors

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def _attestation() -> dict[str, object]:
    return {
        "schema": "flagquantum_domestic_single_card_attestation_v1",
        "status": "verified",
        "accelerator_class": "domestic_accelerator",
        "provider": "torch_fl",
        "logical_device": "flagos:0",
        "evidence_source": "provisioner_owned_runtime_probe",
        "collected_at": "2026-08-25T00:00:00Z",
        "provisioner": "torch-fl-ci-owner",
        "physical_device": {
            "vendor": "example-vendor",
            "model": "example-model",
            "architecture": "example-architecture",
            "device_identifier": "example-device-identifier",
            "logical_device_name": "Example FlagOS accelerator",
        },
        "software": {
            "driver_version": "1.2.3",
            "vendor_runtime_version": "4.5.6",
            "torch_fl_revision": "a" * 40,
            "container_image_digest": "sha256:" + "b" * 64,
        },
        "route_audit": {
            "provider_internal_route_audited": True,
            "logical_to_physical_device_verified": True,
            "physical_device_probe_sha256": "c" * 64,
            "cpu_fallback_allowed": False,
            "cpu_fallback_observed": False,
            "host_execution_observed": False,
        },
    }


def _phase(**extra: object) -> dict[str, object]:
    return {
        "status": "passed",
        "device": "flagos:0",
        "provider": "torch_fl",
        "logical_device_residency": True,
        **extra,
    }


def _evidence() -> dict[str, object]:
    return {
        "schema": "flagquantum_domestic_single_card_evidence_v1",
        "status": "passed",
        "artifact_class": "hardware_execution_candidate",
        "distribution_semantics": "single_device_fast_path",
        "source": {"revision": "e" * 40, "tree_dirty": False},
        "attestation": {"sha256": "d" * 64, "payload": _attestation()},
        "phases": {
            "p0_forward": _phase(flagquantum_host_fallback=False),
            "p1_expectation_gradient": _phase(flagquantum_host_fallback=False),
            "p2_selective_double_single": _phase(flagquantum_host_fallback=False),
            "p3_full_double_single": _phase(state_host_fallback=False),
            "p4_device_double_single": _phase(
                parameter_host_fallback=False, state_host_fallback=False
            ),
            "p5_double_single_sgd": _phase(
                accelerator_float64_tensor_materialized=False,
                tensor_grad_used=False,
                parameter_word_count=2,
            ),
        },
        "claims": {
            "domestic_accelerator_hardware_exercised": True,
            "single_device_correctness_evidence": True,
            "provider_route_no_cpu_fallback_attested": True,
            "domestic_accelerator_certification_candidate": True,
            "hardware_certification": False,
            "convergence_certification": False,
            "flagcx_collectives_validated": False,
            "scalability_claim_allowed": False,
            "performance_claim_allowed": False,
            "production_claim_allowed": False,
        },
    }


def test_domestic_single_card_contract_and_template_fail_closed() -> None:
    contract = tomllib.loads(
        (
            ROOT / "contracts" / "domestic-single-card-certification-contract.toml"
        ).read_text(encoding="utf-8")
    )
    template = json.loads(
        (ROOT / "ci/domestic_single_card_attestation.template.json").read_text(
            encoding="utf-8"
        )
    )
    assert contract_errors(contract, template) == ()
    assert attestation_errors(template)


def test_verified_provisioner_attestation_is_accepted() -> None:
    assert attestation_errors(_attestation()) == ()


def test_attestation_rejects_provider_cpu_fallback() -> None:
    payload = _attestation()
    route = payload["route_audit"]
    assert isinstance(route, dict)
    route["cpu_fallback_observed"] = True
    assert "no-host-fallback" in " ".join(attestation_errors(payload))


def test_complete_candidate_evidence_is_accepted() -> None:
    assert evidence_errors(_evidence()) == ()


def test_evidence_rejects_missing_phase_residency() -> None:
    payload = _evidence()
    phases = payload["phases"]
    assert isinstance(phases, dict)
    phases["p4_device_double_single"]["logical_device_residency"] = False
    assert "residency" in " ".join(evidence_errors(payload))


def test_evidence_rejects_hardware_promotion() -> None:
    payload = _evidence()
    claims = payload["claims"]
    assert isinstance(claims, dict)
    claims["hardware_certification"] = True
    assert "forbidden promotion" in " ".join(evidence_errors(payload))
