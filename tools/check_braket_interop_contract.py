#!/usr/bin/env python3
"""Validate the machine-readable Amazon Braket circuit contract."""

from __future__ import annotations

import argparse
import ast
import importlib
from importlib import metadata
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

from flagquantum.core.ir import IR_VERSION
from flagquantum.core.operator_schema import OPERATOR_SCHEMAS
from flagquantum.ecosystem.braket.conversion import (
    _BRAKET_GATE_TO_FLAGQUANTUM,
    _FLAGQUANTUM_TO_BRAKET_METHOD,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "contracts" / "braket-circuit-interop-contract.toml"
POLICY = ROOT / "dependency-policy.toml"
EXPECTED_REQUIREMENT = "amazon-braket-sdk>=1.117,<2; python_version >= '3.11'"
EXPECTED_VERSIONS = ["1.117.0", "1.127.1"]
CONVERSION = ROOT / "flagquantum/ecosystem/braket/conversion.py"
EXPECTED_SEMANTICS = {
    "artifact": "braket.circuits.Circuit",
    "direction": "bidirectional_static_circuit",
    "qubit_mapping": "braket_integer_qubit_index_equals_flagquantum_wire",
    "wire_extent": "explicit_identity_preserves_unreferenced_flagquantum_wires",
    "operation_order": "braket_instruction_order",
    "statevector_order": ("ascending_braket_qubits_with_wire_zero_most_significant"),
    "parameters": "bound_finite_real_scalars_only",
    "loss_policy": "fail_closed_unless_allow_lossy_true",
}
EXPECTED_PATHS = {
    "integration": "tests/test_braket_interop.py",
    "conformance": "tests/test_braket_interop_conformance.py",
    "core_isolation": "tests/unit/test_braket_interop_boundary.py",
}


def load_toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def conversion_issue_codes(path: Path = CONVERSION) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        str(node.args[1].value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_issue"
        and len(node.args) >= 2
        and isinstance(node.args[1], ast.Constant)
        and isinstance(node.args[1].value, str)
    }


def contract_errors(
    contract: dict[str, Any], policy: dict[str, Any]
) -> tuple[str, ...]:
    errors: list[str] = []
    if contract.get("schema") != "flagquantum_braket_circuit_interop_contract_v1":
        errors.append("Braket circuit interop schema drifted")
    if contract.get("ir_version") != IR_VERSION:
        errors.append("Braket circuit contract must target current IR")
    if contract.get("implementation_status") != "implemented":
        errors.append("Braket circuit v1 must declare its implemented adapter status")
    if contract.get("public_api_available") is not True:
        errors.append("Braket circuit public API must remain available")
    if contract.get("dependency_extra") != "braket":
        errors.append("Braket circuit interop must use only the braket extra")
    if policy.get("extras", {}).get("braket") != [EXPECTED_REQUIREMENT]:
        errors.append("Braket optional dependency range drifted")
    if "braket" not in policy.get("classes", {}).get("interop", ()):
        errors.append("Braket must remain an interop extra")
    if "braket" not in policy.get("aggregates", {}).get("interop-all", ()):
        errors.append("Braket must remain in the portable interop aggregate")
    if "braket" not in policy.get("import_policy", {}).get(
        "core_forbidden_imports", ()
    ):
        errors.append("Braket must remain forbidden from core imports")
    versions = contract.get("amazon_braket_sdk_versions")
    if versions != EXPECTED_VERSIONS or versions != policy.get("tested", {}).get(
        "braket"
    ):
        errors.append("Braket certification lanes must be 1.117.0 and 1.127.1")
    if contract.get("python_minimum") != "3.11":
        errors.append("Braket SDK requires Python 3.11 or newer")
    for field in (
        "core_import_allowed",
        "runtime_execution_allowed",
        "cloud_submission_allowed",
        "autograd_bridge_allowed",
    ):
        if contract.get(field) is not False:
            errors.append(f"Braket contract field {field!r} must remain false")

    semantics = contract.get("semantics", {})
    for name, expected in EXPECTED_SEMANTICS.items():
        if semantics.get(name) != expected:
            errors.append(f"Braket semantic {name!r} drifted")

    unsupported = contract.get("unsupported", {})
    declared_issues = set(unsupported.get("issue_codes", ()))
    emitted_issues = conversion_issue_codes()
    if declared_issues != emitted_issues:
        errors.append(
            "Braket issue-code coverage drifted: "
            f"declared={sorted(declared_issues)}, emitted={sorted(emitted_issues)}"
        )
    excluded = set(unsupported.get("flagquantum_opcodes", ()))
    operations = contract.get("operations", ())
    mapped = [
        operation["flagquantum"]
        for operation in operations
        if isinstance(operation, dict) and isinstance(operation.get("flagquantum"), str)
    ]
    if len(mapped) != len(set(mapped)):
        errors.append("Braket operation mappings must be unique")
    expected_opcodes = set(OPERATOR_SCHEMAS) - excluded
    if set(mapped) != expected_opcodes:
        errors.append(
            "Braket opcode coverage drifted: "
            f"missing={sorted(expected_opcodes-set(mapped))}, "
            f"unexpected={sorted(set(mapped)-expected_opcodes)}"
        )
    gate_types: list[str] = []
    for operation in operations:
        if not isinstance(operation, dict):
            errors.append("Braket operations must be tables")
            continue
        for field in ("braket_gate_type", "braket_form", "flagquantum"):
            if not isinstance(operation.get(field), str) or not operation[field]:
                errors.append(f"Braket operation is missing {field}")
        if isinstance(operation.get("braket_gate_type"), str):
            gate_types.append(operation["braket_gate_type"])
    if len(gate_types) != len(set(gate_types)):
        errors.append("Braket gate mappings must be unambiguous")
    pairs = {
        operation.get("braket_gate_type"): operation.get("flagquantum")
        for operation in operations
        if isinstance(operation, dict)
    }
    if pairs != _BRAKET_GATE_TO_FLAGQUANTUM:
        errors.append("Braket operation contract and import mapping drifted")
    if set(_FLAGQUANTUM_TO_BRAKET_METHOD) != set(mapped):
        errors.append("Braket operation contract and export mapping drifted")

    verification = contract.get("verification", {})
    for name, expected_path in {
        "policy": "tests/unit/test_braket_interop_contract.py",
        **EXPECTED_PATHS,
    }.items():
        if verification.get(name) != expected_path:
            errors.append(f"Braket {name} path drifted")
        elif not (ROOT / expected_path).is_file():
            errors.append(f"Braket verification path does not exist: {expected_path}")
    return tuple(errors)


def sdk_errors(contract: dict[str, Any]) -> tuple[str, ...]:
    try:
        braket = importlib.import_module("braket.circuits")
    except ModuleNotFoundError:
        return ("Braket SDK verification requires the braket optional extra",)
    errors: list[str] = []
    installed = metadata.version("amazon-braket-sdk")
    if installed not in contract.get("amazon_braket_sdk_versions", ()):
        errors.append(f"Braket SDK version {installed!r} is not a certified lane")
    for operation in contract.get("operations", ()):
        gate_type = operation["braket_gate_type"]
        method = _FLAGQUANTUM_TO_BRAKET_METHOD[operation["flagquantum"]]
        if not hasattr(braket.gates, gate_type):
            errors.append(f"Braket gate type {gate_type!r} is unavailable")
        if not hasattr(braket.Circuit, method):
            errors.append(f"Braket Circuit method {method!r} is unavailable")
    return tuple(errors)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-sdk", action="store_true")
    args = parser.parse_args(argv)
    contract = load_toml(CONTRACT)
    errors = list(contract_errors(contract, load_toml(POLICY)))
    if args.verify_sdk:
        errors.extend(sdk_errors(contract))
    if errors:
        print("\n".join(errors))
        return 1
    print("Amazon Braket circuit interoperability contract passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
