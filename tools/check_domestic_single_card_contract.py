#!/usr/bin/env python3
"""Validate the fail-closed domestic single-card certification contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "contracts" / "domestic-single-card-certification-contract.toml"


def contract_errors(
    contract: dict[str, Any], template: dict[str, Any]
) -> tuple[str, ...]:
    errors: list[str] = []
    expected = {
        "schema": "flagquantum_domestic_single_card_certification_contract_v1",
        "maturity": "experimental",
        "execution_semantics": "single_device_fast_path",
        "provider": "torch_fl",
        "logical_device": "flagos:0",
        "artifact_class": "hardware_execution_candidate",
    }
    if any(contract.get(name) != value for name, value in expected.items()):
        errors.append("domestic single-card contract identity drifted")

    matrix = contract.get("acceptance_matrix", {})
    if matrix.get("phases") != [
        "p0_forward",
        "p1_expectation_gradient",
        "p2_selective_double_single",
        "p3_full_double_single",
        "p4_device_double_single",
        "p5_double_single_sgd",
    ]:
        errors.append("domestic single-card P0-P5 matrix drifted")
    if matrix.get("depths") != [8, 32, 128] or matrix.get("seeds") != [0, 7]:
        errors.append("domestic single-card numerical matrix drifted")
    if matrix.get("optimizer_steps") != [1, 16, 64]:
        errors.append("domestic single-card optimizer matrix drifted")

    attestation = contract.get("attestation", {})
    required_attestation = {
        "schema": "flagquantum_domestic_single_card_attestation_v1",
        "accelerator_class": "domestic_accelerator",
        "evidence_source": "provisioner_owned_runtime_probe",
        "provider_internal_route_audited": True,
        "logical_to_physical_device_verified": True,
        "cpu_fallback_allowed": False,
        "cpu_fallback_observed": False,
        "host_execution_observed": False,
    }
    if any(
        attestation.get(name) != value for name, value in required_attestation.items()
    ):
        errors.append("domestic single-card attestation policy drifted")

    claims = contract.get("claim_policy", {})
    required_false = {
        "hardware_certification",
        "convergence_certification",
        "flagcx_collectives_validated",
        "scalability_claim_allowed",
        "performance_claim_allowed",
        "production_claim_allowed",
    }
    if any(claims.get(name) is not False for name in required_false):
        errors.append("domestic single-card contract contains a forbidden promotion")

    paths = contract.get("verification_paths", {})
    for path in paths.values():
        if not isinstance(path, str) or not (ROOT / path).is_file():
            errors.append(
                f"domestic single-card verification path is missing: {path!r}"
            )

    if template.get("status") != "unverified_template":
        errors.append("domestic attestation template must remain unverified")
    if template.get("accelerator_class") == "domestic_accelerator":
        errors.append("domestic attestation template must not certify hardware")
    route = template.get("route_audit", {})
    if route.get("provider_internal_route_audited") is not False:
        errors.append("domestic attestation template must fail closed")
    return tuple(errors)


def main() -> int:
    contract = tomllib.loads(CONTRACT.read_text(encoding="utf-8"))
    template_path = ROOT / contract["verification_paths"]["attestation_template"]
    template = json.loads(template_path.read_text(encoding="utf-8"))
    errors = contract_errors(contract, template)
    if errors:
        print("\n".join(errors))
        return 1
    print("Domestic single-card certification contract passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
