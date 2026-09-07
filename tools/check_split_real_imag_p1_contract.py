#!/usr/bin/env python3
"""Validate the fail-closed split real/imag statevector P1 contract."""

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
CONTRACT = ROOT / "contracts" / "split-real-imag-statevector-p1-contract.toml"
IMPLEMENTATION = ROOT / "flagquantum/runtime/backends/statevector/split_real_imag.py"
SIMULATION = ROOT / "flagquantum/simulation/statevector/split_real_imag.py"
PROFILE = ROOT / "flagquantum/runtime/profiles/split_real_imag_statevector_p1.json"


def _load_toml(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], tomllib.loads(path.read_text(encoding="utf-8")))


def _attribute_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _attribute_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def contract_errors(contract: dict[str, Any]) -> tuple[str, ...]:
    errors: list[str] = []
    expected = {
        "schema": "flagquantum_split_real_imag_statevector_p1_contract_v1",
        "maturity": "experimental",
        "implementation": "flagquantum.runtime.backends.statevector.split_real_imag",
        "operator_profile": "split_real_imag_statevector_p1",
        "representation": "split_real_imag",
        "storage_dtype": "float32",
        "compute_dtype": "float32",
        "reference_dtype": "complex128_cpu_only",
        "gradient_method": "parameter_shift",
        "distribution_semantics": "single_device_fast_path",
        "runtime_default": False,
        "host_fallback_allowed": False,
        "complex_accelerator_tensor_allowed": False,
        "native_autograd_claim_allowed": False,
        "hardware_certification": False,
    }
    for name, value in expected.items():
        if contract.get(name) != value:
            errors.append(f"split real/imag P1 contract {name} must be {value!r}")

    scope = contract.get("scope", {})
    required_observables = {
        "pauli_term",
        "pauli_hamiltonian",
        "flagquantum_ir_observable",
    }
    if set(scope.get("supported_observables", ())) != required_observables:
        errors.append("split real/imag P1 observable scope drifted")
    required_unsupported = {
        "anonymous_trainable_parameters",
        "parameter_expressions",
        "trainable_hamiltonian_coefficients",
        "native_autograd",
        "optimizer_step_api",
        "distributed_execution",
        "automatic_runtime_selection",
        "vendor_certification",
    }
    if not required_unsupported <= set(scope.get("unsupported", ())):
        errors.append("split real/imag P1 fail-closed scope drifted")

    tree = ast.parse(IMPLEMENTATION.read_text(encoding="utf-8"))
    constants = {
        target.id: node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    gate_node = constants.get("SPLIT_REAL_IMAG_PARAMETER_SHIFT_GATES")
    if not isinstance(gate_node, ast.Call):
        errors.append("split real/imag P1 parameter-shift gate declaration is missing")
    elif set(ast.literal_eval(gate_node.args[0])) != set(
        scope.get("parameter_shift_gates", ())
    ):
        errors.append("split real/imag P1 parameter-shift gates drifted")

    runtime_functions = {
        "_expectation_from_bound_p1_ir",
        "execute_split_real_imag_expectation",
        "parameter_shift_split_real_imag_gradient",
    }
    found = set()
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name not in runtime_functions:
            continue
        found.add(node.name)
        calls = {
            _attribute_name(item.func)
            for item in ast.walk(node)
            if isinstance(item, ast.Call)
        }
        forbidden = {"torch.complex", "torch.view_as_complex"} & calls
        if forbidden:
            errors.append(f"{node.name} materializes complex tensors: {forbidden}")
    if found != runtime_functions:
        errors.append("split real/imag P1 runtime execution functions are missing")

    simulation_tree = ast.parse(SIMULATION.read_text(encoding="utf-8"))
    numerical_kernel = next(
        (
            node
            for node in simulation_tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "pauli_term_expectation"
        ),
        None,
    )
    if numerical_kernel is None:
        errors.append("split real/imag P1 simulation expectation kernel is missing")
    else:
        calls = {
            _attribute_name(item.func)
            for item in ast.walk(numerical_kernel)
            if isinstance(item, ast.Call)
        }
        forbidden = {"torch.complex", "torch.view_as_complex"} & calls
        if forbidden:
            errors.append(
                "pauli_term_expectation materializes complex tensors: " f"{forbidden}"
            )

    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    if profile.get("name") != contract.get("operator_profile"):
        errors.append("split real/imag P1 operator profile identity drifted")
    requirements = profile.get("requirements", ())
    if not requirements or any(
        item.get("dtypes") != ["float32"] for item in requirements
    ):
        errors.append("split real/imag P1 operator profile must be float32-only")
    if not any(item.get("backward", False) for item in requirements):
        errors.append("split real/imag P1 profile must probe backward support")

    thresholds = contract.get("thresholds", {})
    if thresholds.get("depths") != [8, 32, 128] or thresholds.get("seeds") != [0, 7]:
        errors.append("split real/imag P1 conformance matrix drifted")
    for name in (
        "max_expectation_absolute_error",
        "max_expectation_relative_error",
        "max_gradient_relative_error",
        "min_gradient_cosine_similarity",
        "max_norm_drift",
    ):
        if float(thresholds.get(name, 0.0)) <= 0.0:
            errors.append(f"split real/imag P1 threshold {name} must be positive")

    for name, raw_path in contract.get("verification", {}).items():
        if not isinstance(raw_path, str) or not (ROOT / raw_path).is_file():
            errors.append(
                f"split real/imag P1 verification path {name!r} does not exist"
            )
    return tuple(errors)


def main() -> int:
    errors = contract_errors(_load_toml(CONTRACT))
    if errors:
        print("\n".join(errors))
        return 1
    print("Split real/imag statevector P1 contract passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
