#!/usr/bin/env python3
"""Validate the fail-closed Double-Single FP32 capability contract."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "contracts" / "double-single-contract.toml"
IMPLEMENTATION = ROOT / "flagquantum/numerics/double_single.py"
EXPECTED_SUPPORTED = {
    "real_add",
    "real_subtract",
    "real_multiply",
    "real_reciprocal",
    "real_reciprocal_sqrt",
    "real_sqrt",
    "real_sin_cos_bounded",
    "real_sum",
    "real_dot",
    "complex_add",
    "complex_multiply",
    "complex_abs_squared",
    "split_statevector_pauli_inner_product",
    "split_statevector_hamiltonian_sum",
    "split_statevector_parameter_shift_accumulation",
    "split_statevector_full_storage",
    "split_statevector_gate_application",
    "split_statevector_periodic_normalization",
    "split_statevector_device_gate_generation",
}
EXPECTED_UNSUPPORTED = {
    "unbounded_device_trigonometry",
    "optimized_gate_kernel_dispatch",
    "distributed_collectives",
    "optimizer_master_state",
    "compiled_execution",
    "torch_fl_provider_ownership",
    "vendor_certification",
}


def load_toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def contract_errors(contract: dict[str, Any]) -> tuple[str, ...]:
    errors: list[str] = []
    expected_scalars = {
        "schema": "flagquantum_double_single_contract_v1",
        "maturity": "experimental",
        "implementation": "flagquantum.numerics.double_single",
        "storage_dtype": "float32",
        "compute_dtype": "float32",
        "reference_dtype": "float64_and_complex128_cpu_only",
        "runtime_integration": "selective_p2_full_state_p3_and_device_gates_p4",
        "default_selection_allowed": False,
        "torch_fl_dependency_allowed": False,
        "external_dependency_allowed": False,
    }
    for name, expected in expected_scalars.items():
        if contract.get(name) != expected:
            errors.append(f"Double-Single contract {name} must be {expected!r}")
    arithmetic = contract.get("arithmetic", {})
    expected_arithmetic = {
        "representation": "unevaluated_normalized_high_plus_low",
        "splitter": 4097.0,
        "sum_algorithm": "knuth_two_sum",
        "renormalization_algorithm": "knuth_two_sum_general",
        "product_algorithm": "dekker_split_product",
        "reduction_order": "deterministic_input_order",
        "autograd": "pytorch_composed_operations",
    }
    if arithmetic != expected_arithmetic:
        errors.append("Double-Single arithmetic contract drifted")
    scope = contract.get("scope", {})
    if set(scope.get("supported", ())) != EXPECTED_SUPPORTED:
        errors.append("Double-Single supported primitive scope drifted")
    if set(scope.get("unsupported", ())) != EXPECTED_UNSUPPORTED:
        errors.append("Double-Single fail-closed scope drifted")
    thresholds = contract.get("thresholds", {})
    if set(thresholds) != {
        "cancellation_sum",
        "cancellation_dot",
        "complex_phase_chain",
        "device_trigonometry",
    }:
        errors.append("Double-Single conformance cases drifted")
    for name, policy in thresholds.items():
        if not isinstance(policy, dict):
            errors.append(f"Double-Single threshold {name!r} must be a table")
            continue
        if float(policy.get("max_absolute_error", -1.0)) < 0.0:
            errors.append(f"Double-Single threshold {name!r} has invalid error bound")
        if float(policy.get("min_improvement_factor", 0.0)) <= 1.0:
            errors.append(f"Double-Single threshold {name!r} must improve on FP32")
    for name, raw_path in contract.get("verification", {}).items():
        if not isinstance(raw_path, str) or not (ROOT / raw_path).is_file():
            errors.append(f"Double-Single verification path {name!r} does not exist")

    tree = ast.parse(IMPLEMENTATION.read_text(encoding="utf-8"))
    forbidden_imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    if any(
        name == "torch_fl"
        or name.startswith("torch_fl.")
        or name.startswith("flagquantum.runtime")
        for name in forbidden_imports
    ):
        errors.append("Double-Single primitives must not import runtime or Torch-FL")
    source = IMPLEMENTATION.read_text(encoding="utf-8")
    if "torch.cuda" in source:
        errors.append("Double-Single primitives must remain device-generic")
    return tuple(errors)


def main() -> int:
    errors = contract_errors(load_toml(CONTRACT))
    if errors:
        print("\n".join(errors))
        return 1
    print("Double-Single FP32 contract passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
