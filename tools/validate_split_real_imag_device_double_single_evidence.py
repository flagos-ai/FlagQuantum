#!/usr/bin/env python3
"""Validate checked-in P4 device Double-Single A800 evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = (
    ROOT / "artifacts/split_real_imag_device_double_single_a800_20260825.json"
)
CONTRACT = (
    ROOT
    / "contracts"
    / "split-real-imag-statevector-p4-device-double-single-contract.toml"
)
HISTORICAL_OPERATOR_PROFILE = {
    "name": "split_real_imag_statevector_p4_device_double_single",
    "sha256": "beb0e58248a93d5f9ab17fe61b869f0a4c2925039c29eb9426101ef0c569ac73",
}


def _load_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def evidence_errors(payload: dict[str, Any]) -> tuple[str, ...]:
    errors: list[str] = []
    if (
        payload.get("schema")
        != "flagquantum_split_real_imag_device_double_single_a800_evidence_v1"
    ):
        errors.append("unexpected split P4 A800 evidence schema")
    if payload.get("status") != "passed":
        errors.append("split P4 A800 evidence must have passed status")

    source = payload.get("source", {})
    if (
        len(str(source.get("revision", ""))) != 40
        or source.get("tree_dirty") is not False
    ):
        errors.append("split P4 A800 evidence requires a clean full source revision")
    if len(str(source.get("archive_sha256", ""))) != 64:
        errors.append("split P4 A800 source archive hash is missing")

    if payload.get("operator_profile") != HISTORICAL_OPERATOR_PROFILE:
        errors.append("split P4 A800 historical operator profile identity drifted")

    contract = tomllib.loads(CONTRACT.read_text(encoding="utf-8"))
    thresholds = contract["thresholds"]
    expected_matrix = {
        name: thresholds[name]
        for name in ("depths", "seeds", "stability_depths", "stability_seeds")
    }
    if payload.get("matrix") != expected_matrix:
        errors.append("split P4 A800 conformance matrix drifted")
    expected_thresholds = {
        name: value for name, value in thresholds.items() if name not in expected_matrix
    }
    if payload.get("thresholds") != expected_thresholds:
        errors.append("split P4 A800 numerical thresholds drifted")

    expected_cases = {
        (depth, seed) for depth in thresholds["depths"] for seed in thresholds["seeds"]
    }
    cases = payload.get("cases", ())
    if {
        (case.get("depth"), case.get("seed")) for case in cases
    } != expected_cases or len(cases) != len(expected_cases):
        errors.append("split P4 A800 evidence case coverage is incomplete")
    upper_bounds = {
        "max_state_abs_error": "max_state_absolute_error",
        "state_infidelity": "max_state_infidelity",
        "norm_drift": "max_norm_drift",
        "expectation_abs_error": "max_expectation_absolute_error",
        "gradient_relative_error": "max_gradient_relative_error",
    }
    for case in cases:
        if not case.get("passed", False):
            errors.append("split P4 A800 evidence contains a failed case")
        for result_name, threshold_name in upper_bounds.items():
            if float(case.get(result_name, float("inf"))) > float(
                thresholds[threshold_name]
            ):
                errors.append(f"split P4 A800 {result_name} exceeds contract")

    expected_stability = {
        (depth, seed)
        for depth in thresholds["stability_depths"]
        for seed in thresholds["stability_seeds"]
    }
    stability = payload.get("stability_cases", ())
    if {
        (case.get("depth"), case.get("seed")) for case in stability
    } != expected_stability or len(stability) != len(expected_stability):
        errors.append("split P4 A800 stability coverage is incomplete")
    for case in stability:
        if not case.get("passed", False):
            errors.append("split P4 A800 evidence contains failed stability")
        for result_name, threshold_name in {
            "max_state_abs_error": "max_state_absolute_error",
            "state_infidelity": "max_state_infidelity",
            "norm_drift": "max_norm_drift",
        }.items():
            if float(case.get(result_name, float("inf"))) > float(
                thresholds[threshold_name]
            ):
                errors.append(f"split P4 A800 stability {result_name} exceeds contract")

    trig = payload.get("trigonometry", {})
    if (
        trig.get("passed") is not True
        or trig.get("max_absolute_angle")
        != contract["trigonometry"]["max_absolute_angle"]
        or float(trig.get("max_absolute_error", float("inf")))
        > float(thresholds["max_trigonometry_absolute_error"])
    ):
        errors.append("split P4 A800 trigonometry evidence is incomplete")
    gates = payload.get("gate_generation", {})
    if gates.get("passed") is not True or float(
        gates.get("max_entry_absolute_error", float("inf"))
    ) > float(thresholds["max_gate_entry_absolute_error"]):
        errors.append("split P4 A800 gate-generation evidence is incomplete")

    environment = payload.get("environment", {})
    expected_locks = {
        "native_cuda_image_id": "sha256:a6a6d2c4d50c9ac503eeae562e841add24a1b71a7e70410709c6ca6a87722d7b",
        "flagos_image_id": "sha256:8244e20acbad1a3ede024404b13d129fa6d83933e1320f93b75e404a85efbb4b",
        "torch_fl_revision": "2e00b393cf80088706b460a187aef185d3a283f4",
        "torch_fl_source_archive_sha256": "cde803ab5d5f4d8cdd9a730215ffd8b4d0f98ef8de7c808296ca2b61e5e47444",
    }
    if any(environment.get(name) != value for name, value in expected_locks.items()):
        errors.append("split P4 A800 environment lock identity drifted")
    if len(str(environment.get("torch_fl_wheel_sha256", ""))) != 64:
        errors.append("split P4 A800 Torch-FL wheel hash is missing")

    routes = {route.get("name"): route for route in payload.get("routes", ())}
    required_route_fields = {
        "status": "passed",
        "logical_device_residency": True,
        "device_only_double_single_trigonometry": True,
        "host_gate_encoding": False,
        "parameter_host_fallback": False,
        "state_host_fallback": False,
        "complex_accelerator_tensor_materialized": False,
    }
    native = routes.get("native_cuda", {})
    if (
        native.get("device") != "cuda:0"
        or native.get("provider") != "pytorch_cuda"
        or any(
            native.get(name) != value for name, value in required_route_fields.items()
        )
        or len(str(native.get("raw_result_sha256", ""))) != 64
    ):
        errors.append("split P4 A800 native CUDA route is incomplete")
    flagos = routes.get("torch_fl_flagos", {})
    if (
        flagos.get("device") != "flagos:0"
        or flagos.get("provider") != "torch_fl"
        or any(
            flagos.get(name) != value for name, value in required_route_fields.items()
        )
        or flagos.get("provider_internal_route_audited") is not False
        or flagos.get("flagcx_collectives_validated") is not False
        or flagos.get("opt_in_integration_test") != "passed"
        or len(str(flagos.get("raw_result_sha256", ""))) != 64
    ):
        errors.append("split P4 A800 Torch-FL flagos route is incomplete")

    claims = payload.get("claims", {})
    if claims.get("maturity") != "experimental":
        errors.append("split P4 A800 evidence must remain experimental")
    required_claims = {
        "full_state_double_single": True,
        "device_only_double_single_trigonometry": True,
        "host_gate_encoding": False,
        "parameter_host_fallback": False,
        "state_host_fallback": False,
        "complex_accelerator_tensor_materialized": False,
        "host_sync_safety_checks": True,
        "native_autograd_certification": False,
        "convergence_certification": False,
        "hardware_certification": False,
        "domestic_accelerator_certification": False,
        "provider_internal_route_audited": False,
        "flagcx_collectives_validated": False,
        "scalability_claim_allowed": False,
        "performance_claim_allowed": False,
        "production_claim_allowed": False,
    }
    if any(claims.get(name) is not value for name, value in required_claims.items()):
        errors.append("split P4 A800 evidence contains a forbidden claim promotion")
    return tuple(errors)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", nargs="?", type=Path, default=DEFAULT_ARTIFACT)
    args = parser.parse_args()
    errors = evidence_errors(_load_json(args.artifact))
    if errors:
        print("\n".join(errors))
        return 1
    print("Split real/imag P4 A800 evidence passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
