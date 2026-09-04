#!/usr/bin/env python3
"""Validate the device-generated Double-Single P4 contract."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, cast

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "contracts"
    / "split-real-imag-statevector-p4-device-double-single-contract.toml"
)
IMPLEMENTATION = (
    ROOT
    / "flagquantum/runtime/backends/statevector/split_real_imag_device_double_single.py"
)
GATES = ROOT / "flagquantum/simulation/double_single_device_gates.py"
PROFILE = (
    ROOT
    / "flagquantum/runtime/profiles/split_real_imag_statevector_p4_device_double_single.json"
)


def load_toml(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], tomllib.loads(path.read_text(encoding="utf-8")))


def contract_errors(contract: dict[str, Any]) -> tuple[str, ...]:
    errors: list[str] = []
    expected = {
        "schema": "flagquantum_split_real_imag_statevector_p4_device_double_single_contract_v1",
        "maturity": "experimental",
        "operator_profile": "split_real_imag_statevector_p4_device_double_single",
        "base_executor": "split_real_imag_statevector_p3_double_single",
        "representation": "double_single_fp32_complex",
        "gate_generation": "double_single_fp32_device_trigonometry",
        "distribution_semantics": "single_device_fast_path",
        "runtime_default": False,
        "host_scalar_parameter_ingestion_allowed": True,
        "host_gate_encoding_allowed": False,
        "parameter_host_fallback_allowed": False,
        "state_host_fallback_allowed": False,
        "complex_accelerator_tensor_allowed": False,
        "host_sync_safety_checks": True,
        "native_autograd_claim_allowed": False,
        "convergence_claim_allowed": False,
        "hardware_certification": False,
    }
    for name, value in expected.items():
        if contract.get(name) != value:
            errors.append(f"split real/imag P4 contract {name} must be {value!r}")

    scope = contract.get("scope", {})
    if set(scope.get("parameterized_gates", ())) != {
        "rx",
        "ry",
        "rz",
        "rxx",
        "ryy",
        "rzz",
    }:
        errors.append("split real/imag P4 parameterized gate scope drifted")
    required_unsupported = {
        "parameter_expressions",
        "float64_parameter_tensors",
        "custom_matrices",
        "angles_outside_certified_range",
        "native_autograd",
        "optimizer_step_api",
        "compiled_execution",
        "distributed_execution",
        "automatic_runtime_selection",
        "convergence_certification",
        "provider_internal_route_audit",
        "vendor_certification",
        "performance_claim",
    }
    if not required_unsupported <= set(scope.get("unsupported", ())):
        errors.append("split real/imag P4 fail-closed scope drifted")

    gate_source = GATES.read_text(encoding="utf-8")
    if "torch.sin" in gate_source or "torch.cos" in gate_source:
        errors.append("P4 must use explicit Double-Single trigonometry")
    if "complex128" in gate_source or "float64" in gate_source:
        errors.append("P4 device gate encoding must remain FP32-only")
    if ".cpu(" in gate_source:
        errors.append("P4 device gate encoding must not move parameters to CPU")
    tree = ast.parse(gate_source)
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    if not any(name.endswith("numerics.double_single") for name in imports):
        errors.append("P4 gates must compose the Double-Single numerical layer")

    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    requirements = profile.get("requirements", ())
    if profile.get("name") != contract.get("operator_profile"):
        errors.append("split real/imag P4 operator profile identity drifted")
    if not requirements or any(
        item.get("dtypes") != ["float32"] for item in requirements
    ):
        errors.append("split real/imag P4 profile must be float32-only")
    required_operators = {"aten::round", "aten::remainder", "aten::where"}
    if not required_operators <= {item.get("operator") for item in requirements}:
        errors.append("split real/imag P4 range-reduction profile is incomplete")

    thresholds = contract.get("thresholds", {})
    expected_matrix = {
        "depths": [8, 32, 128],
        "seeds": [0, 7],
        "stability_depths": [512],
        "stability_seeds": [0],
    }
    if any(thresholds.get(name) != value for name, value in expected_matrix.items()):
        errors.append("split real/imag P4 conformance matrix drifted")
    for name, value in thresholds.items():
        if name not in expected_matrix and float(value) <= 0.0:
            errors.append(f"split real/imag P4 threshold {name} must be positive")
    if contract.get("trigonometry", {}).get("max_absolute_angle") != 1024.0:
        errors.append("split real/imag P4 certified angle range drifted")

    for name, raw_path in contract.get("verification", {}).items():
        if not isinstance(raw_path, str) or not (ROOT / raw_path).is_file():
            errors.append(f"split real/imag P4 verification path {name!r} is missing")
    return tuple(errors)


def main() -> int:
    errors = contract_errors(load_toml(CONTRACT))
    if errors:
        print("\n".join(errors))
        return 1
    print("Split real/imag statevector P4 device Double-Single contract passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
