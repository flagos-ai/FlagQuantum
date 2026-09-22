#!/usr/bin/env python3
"""Validate the machine-readable Cirq interoperability contract."""

from __future__ import annotations

import argparse
import ast
import importlib
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

from flagquantum.core.ir import IR_VERSION
from flagquantum.core.operator_schema import OPERATOR_SCHEMAS
from flagquantum.ecosystem.cirq.conversion import _CIRQ_SYMBOL_TO_FLAGQUANTUM

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "contracts" / "cirq-interop-contract.toml"
POLICY = ROOT / "dependency-policy.toml"
CONVERSION = ROOT / "flagquantum/ecosystem/cirq/conversion.py"
EXPECTED_VERSIONS = ["1.6.1", "1.7.0"]
EXPECTED_SEMANTICS = {
    "artifact": "cirq.Circuit",
    "qubit_mapping": "contiguous_line_qubit_index_equals_flagquantum_wire",
    "operation_order": "stable_moment_then_operation_order",
    "moment_packing": "flattened_and_not_preserved",
    "statevector_order": ("explicit_line_qubit_order_with_wire_zero_most_significant"),
    "parameters": "bound_real_scalars_or_v1_symbolic_arithmetic",
    "loss_policy": "fail_closed_unless_allow_lossy_true",
}
EXPECTED_SYMBOLIC_PARAMETER_CONTRACT = {
    "implementation_status": "implemented",
    "public_api_change": False,
    "source_protocol": "cirq_parameter_protocol_with_sympy_expressions",
    "symbol_discovery": "cirq.parameter_symbols",
    "binding_reference": "cirq.resolve_parameters",
    "target_representation": "flagquantum_parameter_or_parameter_expression",
    "parameter_identity": "unique_name",
    "supported_nodes": ["symbol", "real_constant", "add", "mul", "neg"],
    "unsupported_nodes": [
        "complex_constant",
        "function",
        "power",
        "division_by_unbound_parameter",
        "unknown_symbol",
    ],
    "canonicalization": "deterministic_left_fold_after_sympy_canonicalization",
    "import_failure": "reject_before_instruction_enters_ir",
    "export_failure": "reject_before_cirq_operation_creation",
    "issue_code": "unsupported_parameter_expression",
    "binding_equivalence": (
        "cirq_resolve_parameters_matches_flagquantum_bind_parameters"
    ),
    "external_object_retention": False,
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
    if contract.get("schema") != "flagquantum_cirq_interop_contract_v1":
        errors.append("Cirq interop schema drifted")
    if contract.get("ir_version") != IR_VERSION:
        errors.append("Cirq contract must target current IR")
    if contract.get("implementation_status") != "implemented":
        errors.append("Cirq v1 must declare its implemented adapter status")
    if contract.get("public_api_available") is not True:
        errors.append("Cirq public API must remain available after implementation")
    if contract.get("dependency_extra") != "cirq":
        errors.append("Cirq must use only its optional extra")
    if contract.get("core_import_allowed") is not False:
        errors.append("Cirq core import must remain forbidden")
    if contract.get("runtime_execution_allowed") is not False:
        errors.append("Cirq runtime execution must remain outside v1")
    versions = contract.get("cirq_core_versions")
    if versions != EXPECTED_VERSIONS or versions != policy.get("tested", {}).get(
        "cirq"
    ):
        errors.append("Cirq certification lanes must be 1.6.1 and 1.7.0")
    semantics = contract.get("semantics", {})
    for name, expected in EXPECTED_SEMANTICS.items():
        if semantics.get(name) != expected:
            errors.append(f"Cirq semantic {name!r} drifted")

    symbolic_parameters = contract.get("symbolic_parameter_contract", {})
    if symbolic_parameters != EXPECTED_SYMBOLIC_PARAMETER_CONTRACT:
        errors.append("Cirq symbolic-parameter extension contract drifted")

    unsupported = contract.get("unsupported", {})
    if "symbolic_parameters" in unsupported.get("cirq_features", ()):
        errors.append("Cirq symbolic parameters must not remain globally unsupported")
    if "unsupported_parameter_expression" not in unsupported.get("issue_codes", ()):
        errors.append("Cirq must declare its symbolic-expression rejection code")
    declared_issues = set(unsupported.get("issue_codes", ()))
    emitted_issues = conversion_issue_codes()
    if declared_issues != emitted_issues:
        errors.append(
            "Cirq issue-code coverage drifted: "
            f"declared={sorted(declared_issues)}, emitted={sorted(emitted_issues)}"
        )
    excluded = set(unsupported.get("flagquantum_opcodes", ()))
    operations = contract.get("operations", ())
    mapped = [
        operation.get("flagquantum")
        for operation in operations
        if isinstance(operation, dict)
    ]
    if len(mapped) != len(set(mapped)):
        errors.append("Cirq operation mappings must be unique")
    pairs = {
        operation.get("cirq_symbol"): operation.get("flagquantum")
        for operation in operations
        if isinstance(operation, dict)
    }
    if pairs != _CIRQ_SYMBOL_TO_FLAGQUANTUM:
        errors.append("Cirq operation contract and adapter mapping drifted")
    expected = set(OPERATOR_SCHEMAS) - excluded
    if set(mapped) != expected:
        errors.append(
            "Cirq opcode coverage drifted: "
            f"missing={sorted(expected-set(mapped))}, "
            f"unexpected={sorted(set(mapped)-expected)}"
        )
    for operation in operations:
        if not isinstance(operation, dict):
            errors.append("Cirq operations must be tables")
            continue
        for field in ("cirq_symbol", "cirq_gate_type", "cirq_form"):
            if not isinstance(operation.get(field), str) or not operation[field]:
                errors.append(f"Cirq operation is missing {field}")
    for raw_path in contract.get("verification", {}).values():
        if not (ROOT / str(raw_path)).is_file():
            errors.append(f"Cirq verification path does not exist: {raw_path}")
    return tuple(errors)


def sdk_errors(contract: dict[str, Any]) -> tuple[str, ...]:
    try:
        cirq = importlib.import_module("cirq")
    except ModuleNotFoundError:
        return ("Cirq SDK verification requires the cirq optional extra",)
    errors: list[str] = []
    installed = str(cirq.__version__)
    if installed not in contract.get("cirq_core_versions", ()):
        errors.append(f"Cirq SDK version {installed!r} is not a certified lane")
    constructors = {"rx": cirq.rx(0.5), "ry": cirq.ry(0.5), "rz": cirq.rz(0.5)}
    for operation in contract.get("operations", ()):
        symbol = operation["cirq_symbol"]
        gate = constructors.get(symbol, getattr(cirq, symbol, None))
        if gate is None:
            errors.append(f"Cirq symbol {symbol!r} is unavailable")
            continue
        gate_type = type(gate).__name__
        declared_type = operation["cirq_gate_type"]
        compatible_types = {
            "X": {"XPowGate", "_PauliX"},
            "Y": {"YPowGate", "_PauliY"},
            "Z": {"ZPowGate", "_PauliZ"},
        }.get(symbol, {declared_type})
        if gate_type not in compatible_types:
            errors.append(
                f"Cirq symbol {symbol!r} type drifted: "
                f"expected {sorted(compatible_types)}, got {gate_type!r}"
            )
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
    print("Cirq interoperability contract passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
