#!/usr/bin/env python3
"""Validate the fail-closed split real/imag statevector P0 contract."""

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
CONTRACT = ROOT / "contracts" / "split-real-imag-statevector-contract.toml"
IMPLEMENTATION = ROOT / "flagquantum/runtime/executors/statevector/split_real_imag.py"
KERNEL = ROOT / "flagquantum/simulation/statevector/split_real_imag.py"
PROFILE = ROOT / "flagquantum/runtime/profiles/split_real_imag_statevector_p0.json"


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
        "schema": "flagquantum_split_real_imag_statevector_contract_v1",
        "maturity": "experimental",
        "implementation": "flagquantum.runtime.executors.statevector.split_real_imag",
        "gate_implementation": "flagquantum.simulation.statevector.split_real_imag",
        "operator_profile": "split_real_imag_statevector_p0",
        "representation": "split_real_imag",
        "storage_dtype": "float32",
        "compute_dtype": "float32",
        "gate_generation_dtype": "float32",
        "reference_dtype": "complex128_cpu_only",
        "distribution_semantics": "single_device_fast_path",
        "runtime_default": False,
        "host_fallback_allowed": False,
        "complex_accelerator_tensor_allowed": False,
        "torch_fl_core_dependency_allowed": False,
        "external_dependency_allowed": False,
    }
    for name, value in expected.items():
        if contract.get(name) != value:
            errors.append(f"split real/imag contract {name} must be {value!r}")

    scope = contract.get("scope", {})
    if scope.get("supported_state") != "batch_one_zero_state":
        errors.append("split real/imag P0 state scope drifted")
    unsupported = set(scope.get("unsupported", ()))
    required_unsupported = {
        "custom_initial_state",
        "batched_state",
        "custom_matrices",
        "trainable_parameters",
        "gradients",
        "optimizer_step",
        "distributed_execution",
        "automatic_runtime_selection",
        "vendor_certification",
    }
    if not required_unsupported <= unsupported:
        errors.append("split real/imag fail-closed scope drifted")

    implementation_source = IMPLEMENTATION.read_text(encoding="utf-8")
    kernel_source = KERNEL.read_text(encoding="utf-8")
    tree = ast.parse(implementation_source)
    kernel_tree = ast.parse(kernel_source)
    gates_node = next(
        (
            node
            for node in kernel_tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "SPLIT_REAL_IMAG_SUPPORTED_GATES"
                for target in node.targets
            )
        ),
        None,
    )
    if gates_node is None or not isinstance(gates_node.value, ast.Call):
        errors.append("split real/imag supported gate declaration is missing")
    else:
        declared = set(ast.literal_eval(gates_node.value.args[0]))
        if declared != set(scope.get("supported_gates", ())):
            errors.append("split real/imag supported gates drifted from contract")

    executor_names = {
        "run_split_real_imag_statevector",
        "instruction_matrix_pair",
        "apply_gate_pair",
        "execute_split_real_imag_statevector",
    }
    for node in (*tree.body, *kernel_tree.body):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name not in executor_names:
            continue
        calls = {
            _attribute_name(item.func)
            for item in ast.walk(node)
            if isinstance(item, ast.Call)
        }
        forbidden = {"torch.complex", "torch.view_as_complex"} & calls
        if forbidden:
            errors.append(
                f"{node.name} materializes complex tensors: {sorted(forbidden)}"
            )
    source = implementation_source + kernel_source
    if "torch.cuda" in source or "torch_fl" in source:
        errors.append("split executor must remain platform-neutral")

    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    if profile.get("name") != contract.get("operator_profile"):
        errors.append("split real/imag operator profile identity drifted")
    requirements = profile.get("requirements", ())
    if not requirements or any(
        item.get("dtypes") != ["float32"] for item in requirements
    ):
        errors.append("split real/imag operator profile must be float32-only")
    if any(item.get("backward", False) for item in requirements):
        errors.append("split real/imag P0 profile must remain forward-only")

    thresholds = contract.get("thresholds", {})
    if thresholds.get("depths") != [8, 32, 128]:
        errors.append("split real/imag conformance depths drifted")
    for name in (
        "max_state_absolute_error",
        "max_norm_drift",
        "max_state_infidelity",
    ):
        if float(thresholds.get(name, 0.0)) <= 0.0:
            errors.append(f"split real/imag threshold {name} must be positive")

    for name, raw_path in contract.get("verification", {}).items():
        if not isinstance(raw_path, str) or not (ROOT / raw_path).is_file():
            errors.append(f"split real/imag verification path {name!r} does not exist")
    return tuple(errors)


def main() -> int:
    errors = contract_errors(_load_toml(CONTRACT))
    if errors:
        print("\n".join(errors))
        return 1
    print("Split real/imag statevector P0 contract passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
