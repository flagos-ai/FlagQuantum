#!/usr/bin/env python3
"""Validate the full Double-Single split-statevector P3 contract."""

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
    ROOT / "contracts" / "split-real-imag-statevector-p3-double-single-contract.toml"
)
IMPLEMENTATION = (
    ROOT / "flagquantum/runtime/executors/statevector/split_real_imag_double_single.py"
)
GATES = ROOT / "flagquantum/simulation/statevector/double_single_host_gates.py"
PROFILE = (
    ROOT
    / "flagquantum/runtime/profiles/split_real_imag_statevector_p3_double_single.json"
)


def load_toml(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], tomllib.loads(path.read_text(encoding="utf-8")))


def contract_errors(contract: dict[str, Any]) -> tuple[str, ...]:
    errors: list[str] = []
    expected = {
        "schema": "flagquantum_split_real_imag_statevector_p3_double_single_contract_v1",
        "maturity": "experimental",
        "implementation": "flagquantum.runtime.executors.statevector.split_real_imag_double_single",
        "gate_implementation": "flagquantum.simulation.statevector.double_single_host_gates",
        "operator_profile": "split_real_imag_statevector_p3_double_single",
        "base_executor": "split_real_imag_statevector_p2_precision",
        "representation": "double_single_fp32_complex",
        "state_storage_dtype": "four_float32_words_per_complex_amplitude",
        "kernel_compute_dtype": "float32",
        "reduction_dtype": "double_single_fp32",
        "normalization_dtype": "double_single_fp32",
        "distribution_semantics": "single_device_fast_path",
        "runtime_default": False,
        "host_gate_encoding_required": True,
        "state_host_fallback_allowed": False,
        "complex_accelerator_tensor_allowed": False,
        "native_autograd_claim_allowed": False,
        "convergence_claim_allowed": False,
        "hardware_certification": False,
    }
    for name, value in expected.items():
        if contract.get(name) != value:
            errors.append(f"split real/imag P3 contract {name} must be {value!r}")

    scope = contract.get("scope", {})
    required_ds = {
        "state_storage",
        "gate_application",
        "periodic_normalization",
        "pauli_inner_product",
        "hamiltonian_term_sum",
        "parameter_shift_gradient_accumulation",
    }
    if set(scope.get("double_single_operations", ())) != required_ds:
        errors.append("split real/imag P3 Double-Single operation scope drifted")
    required_unsupported = {
        "device_only_double_single_trigonometry",
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
        errors.append("split real/imag P3 fail-closed scope drifted")

    tree = ast.parse(IMPLEMENTATION.read_text(encoding="utf-8"))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    if not any(name.endswith("numerics.double_single") for name in imports):
        errors.append("split real/imag P3 must compose Double-Single primitives")
    source = IMPLEMENTATION.read_text(encoding="utf-8")
    gate_source = GATES.read_text(encoding="utf-8")
    if "view_as_complex" in source:
        errors.append("split real/imag P3 materializes accelerator complex tensors")
    if 'device="cpu"' not in gate_source or "from_complex128" not in gate_source:
        errors.append("split real/imag P3 host gate encoding must remain explicit")

    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    requirements = profile.get("requirements", ())
    if profile.get("name") != contract.get("operator_profile"):
        errors.append("split real/imag P3 operator profile identity drifted")
    if not requirements or any(
        item.get("dtypes") != ["float32"] for item in requirements
    ):
        errors.append("split real/imag P3 profile must be float32-only")
    if any(item.get("backward", False) for item in requirements):
        errors.append("split real/imag P3 profile must remain forward-only")

    thresholds = contract.get("thresholds", {})
    if thresholds.get("depths") != [8, 32, 128] or thresholds.get("seeds") != [0, 7]:
        errors.append("split real/imag P3 conformance matrix drifted")
    for name in (
        "max_state_absolute_error",
        "max_state_infidelity",
        "max_norm_drift",
        "max_expectation_absolute_error",
        "max_gradient_relative_error",
        "min_expectation_improvement_factor",
        "min_gradient_improvement_factor",
    ):
        if float(thresholds.get(name, 0.0)) <= 0.0:
            errors.append(f"split real/imag P3 threshold {name} must be positive")

    for name, raw_path in contract.get("verification", {}).items():
        if not isinstance(raw_path, str) or not (ROOT / raw_path).is_file():
            errors.append(f"split real/imag P3 verification path {name!r} is missing")
    return tuple(errors)


def main() -> int:
    errors = contract_errors(load_toml(CONTRACT))
    if errors:
        print("\n".join(errors))
        return 1
    print("Split real/imag statevector P3 Double-Single contract passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
