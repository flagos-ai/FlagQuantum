#!/usr/bin/env python3
"""Validate the selective Double-Single split statevector P2 contract."""

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
CONTRACT = ROOT / "split-real-imag-statevector-p2-precision-contract.toml"
IMPLEMENTATION = (
    ROOT / "flagquantum/runtime/backends/statevector/split_real_imag_precision.py"
)
PROFILE = (
    ROOT / "flagquantum/runtime/profiles/split_real_imag_statevector_p2_precision.json"
)


def load_toml(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], tomllib.loads(path.read_text(encoding="utf-8")))


def contract_errors(contract: dict[str, Any]) -> tuple[str, ...]:
    errors: list[str] = []
    expected = {
        "schema": "flagquantum_split_real_imag_statevector_p2_precision_contract_v1",
        "maturity": "experimental",
        "implementation": "flagquantum.runtime.backends.statevector.split_real_imag_precision",
        "operator_profile": "split_real_imag_statevector_p2_precision",
        "base_executor": "split_real_imag_statevector_p1",
        "representation": "split_real_imag",
        "state_storage_dtype": "float32",
        "kernel_compute_dtype": "float32",
        "reduction_dtype": "double_single_fp32",
        "reference_dtype": "complex128_cpu_only",
        "gradient_method": "parameter_shift",
        "distribution_semantics": "single_device_fast_path",
        "runtime_default": False,
        "host_fallback_allowed": False,
        "complex_accelerator_tensor_allowed": False,
        "full_state_double_single_claim_allowed": False,
        "native_autograd_claim_allowed": False,
        "convergence_claim_allowed": False,
        "hardware_certification": False,
    }
    for name, value in expected.items():
        if contract.get(name) != value:
            errors.append(f"split real/imag P2 contract {name} must be {value!r}")

    required_ds = {
        "pauli_inner_product",
        "hamiltonian_term_sum",
        "parameter_shift_gradient_accumulation",
    }
    scope = contract.get("scope", {})
    if set(scope.get("double_single_operations", ())) != required_ds:
        errors.append("split real/imag P2 Double-Single operation scope drifted")
    required_unsupported = {
        "full_state_double_single",
        "double_single_gate_generation",
        "decomposition",
        "native_autograd",
        "optimizer_step_api",
        "distributed_execution",
        "automatic_runtime_selection",
        "convergence_certification",
        "vendor_certification",
    }
    if not required_unsupported <= set(scope.get("unsupported", ())):
        errors.append("split real/imag P2 fail-closed scope drifted")

    tree = ast.parse(IMPLEMENTATION.read_text(encoding="utf-8"))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    if not any(name.endswith("numerics.double_single") for name in imports):
        errors.append("split real/imag P2 must compose Double-Single primitives")
    source = IMPLEMENTATION.read_text(encoding="utf-8")
    if "torch.complex(" in source or "view_as_complex" in source:
        errors.append("split real/imag P2 materializes accelerator complex tensors")

    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    requirements = profile.get("requirements", ())
    if profile.get("name") != contract.get("operator_profile"):
        errors.append("split real/imag P2 operator profile identity drifted")
    if not requirements or any(
        item.get("dtypes") != ["float32"] for item in requirements
    ):
        errors.append("split real/imag P2 profile must be float32-only")
    if any(item.get("backward", False) for item in requirements):
        errors.append("split real/imag P2 profile must remain forward-only")

    thresholds = contract.get("thresholds", {})
    if thresholds.get("depths") != [8, 32, 128] or thresholds.get("seeds") != [0, 7]:
        errors.append("split real/imag P2 conformance matrix drifted")
    for name in (
        "max_expectation_absolute_error",
        "max_gradient_relative_error",
        "min_expectation_improvement_factor",
        "min_gradient_improvement_factor",
    ):
        if float(thresholds.get(name, 0.0)) <= 0.0:
            errors.append(f"split real/imag P2 threshold {name} must be positive")

    for name, raw_path in contract.get("verification", {}).items():
        if not isinstance(raw_path, str) or not (ROOT / raw_path).is_file():
            errors.append(f"split real/imag P2 verification path {name!r} is missing")
    return tuple(errors)


def main() -> int:
    errors = contract_errors(load_toml(CONTRACT))
    if errors:
        print("\n".join(errors))
        return 1
    print("Split real/imag statevector P2 precision contract passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
