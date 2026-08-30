#!/usr/bin/env python3
"""Validate checked-in full Double-Single P3 A800 evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

from flagquantum.runtime.capabilities import load_operator_profile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = ROOT / "artifacts/split_real_imag_double_single_a800_20260824.json"
CONTRACT = (
    ROOT / "contracts" / "split-real-imag-statevector-p3-double-single-contract.toml"
)


def _load_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def evidence_errors(payload: dict[str, Any]) -> tuple[str, ...]:
    errors: list[str] = []
    if (
        payload.get("schema")
        != "flagquantum_split_real_imag_double_single_a800_evidence_v1"
    ):
        errors.append("unexpected split P3 A800 evidence schema")
    if payload.get("status") != "passed":
        errors.append("split P3 A800 evidence must have passed status")

    source = payload.get("source", {})
    if (
        len(str(source.get("revision", ""))) != 40
        or source.get("tree_dirty") is not False
    ):
        errors.append("split P3 A800 evidence requires a clean full source revision")
    if len(str(source.get("archive_sha256", ""))) != 64:
        errors.append("split P3 A800 source archive hash is missing")

    profile = load_operator_profile("split_real_imag_statevector_p3_double_single")
    profile_payload = payload.get("operator_profile", {})
    if profile_payload != {"name": profile.name, "sha256": profile.profile_hash}:
        errors.append("split P3 A800 operator profile identity drifted")

    contract = tomllib.loads(CONTRACT.read_text(encoding="utf-8"))
    if payload.get("matrix") != {
        "depths": contract["thresholds"]["depths"],
        "seeds": contract["thresholds"]["seeds"],
    }:
        errors.append("split P3 A800 conformance matrix drifted")
    contract_thresholds = {
        name: value
        for name, value in contract["thresholds"].items()
        if name not in {"depths", "seeds"}
    }
    if payload.get("thresholds") != contract_thresholds:
        errors.append("split P3 A800 numerical thresholds drifted")

    expected_cases = {
        (depth, seed)
        for depth in contract["thresholds"]["depths"]
        for seed in contract["thresholds"]["seeds"]
    }
    cases = payload.get("cases", ())
    actual_cases = {(case.get("depth"), case.get("seed")) for case in cases}
    if actual_cases != expected_cases or len(cases) != len(expected_cases):
        errors.append("split P3 A800 evidence case coverage is incomplete")
    upper_bounds = {
        "max_state_abs_error": "max_state_absolute_error",
        "state_infidelity": "max_state_infidelity",
        "norm_drift": "max_norm_drift",
        "expectation_abs_error": "max_expectation_absolute_error",
        "gradient_relative_error": "max_gradient_relative_error",
    }
    lower_bounds = {
        "expectation_improvement_factor": "min_expectation_improvement_factor",
        "gradient_improvement_factor": "min_gradient_improvement_factor",
    }
    for case in cases:
        if not case.get("passed", False):
            errors.append("split P3 A800 evidence contains a failed case")
        for result_name, threshold_name in upper_bounds.items():
            if float(case.get(result_name, float("inf"))) > float(
                contract["thresholds"][threshold_name]
            ):
                errors.append(f"split P3 A800 {result_name} exceeds contract")
        for result_name, threshold_name in lower_bounds.items():
            if float(case.get(result_name, 0.0)) < float(
                contract["thresholds"][threshold_name]
            ):
                errors.append(f"split P3 A800 {result_name} is below contract")

    routes = {route.get("name"): route for route in payload.get("routes", ())}
    native = routes.get("native_cuda", {})
    flagos = routes.get("torch_fl_flagos", {})
    if (
        native.get("status") != "passed"
        or native.get("device") != "cuda:0"
        or native.get("provider") != "pytorch_cuda"
        or native.get("logical_device_residency") is not True
        or native.get("host_gate_encoding") is not True
        or native.get("state_host_fallback") is not False
    ):
        errors.append("split P3 A800 native CUDA route is incomplete")
    if (
        flagos.get("status") != "passed"
        or flagos.get("device") != "flagos:0"
        or flagos.get("provider") != "torch_fl"
        or flagos.get("logical_device_residency") is not True
        or flagos.get("host_gate_encoding") is not True
        or flagos.get("state_host_fallback") is not False
        or flagos.get("provider_internal_route_audited") is not False
        or flagos.get("opt_in_integration_test") != "passed"
    ):
        errors.append("split P3 A800 Torch-FL flagos route is incomplete")

    claims = payload.get("claims", {})
    if claims.get("maturity") != "experimental":
        errors.append("split P3 A800 evidence must remain experimental")
    required_claims = {
        "full_state_double_single": True,
        "host_gate_encoding": True,
        "state_host_fallback": False,
        "device_only_double_single_trigonometry": False,
        "convergence_certification": False,
        "hardware_certification": False,
        "domestic_accelerator_certification": False,
        "provider_internal_route_audited": False,
        "scalability_claim_allowed": False,
        "performance_claim_allowed": False,
        "production_claim_allowed": False,
    }
    if any(claims.get(name) is not value for name, value in required_claims.items()):
        errors.append("split P3 A800 evidence contains a forbidden claim promotion")
    return tuple(errors)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", nargs="?", type=Path, default=DEFAULT_ARTIFACT)
    args = parser.parse_args()
    errors = evidence_errors(_load_json(args.artifact))
    if errors:
        print("\n".join(errors))
        return 1
    print("Split real/imag P3 A800 evidence passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
