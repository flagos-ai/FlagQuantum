#!/usr/bin/env python3
"""Validate the machine-readable Qiskit interoperability contract."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

from flagquantum.core.ir import IR_VERSION
from flagquantum.core.operator_schema import OPERATOR_SCHEMAS

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "qiskit-interop-contract.toml"
DEPENDENCY_POLICY = ROOT / "dependency-policy.toml"
CONVERSION = ROOT / "flagquantum" / "interop" / "qiskit" / "conversion.py"
EXPECTED_SCHEMA = "flagquantum_qiskit_interop_contract_v1"
EXPECTED_SEMANTICS = {
    "quantum_wire_mapping": "flat_qiskit_bit_index_equals_flagquantum_wire",
    "classical_bit_mapping": ("flat_qiskit_bit_index_equals_flagquantum_classical_bit"),
    "flagquantum_statevector_order": "wire_zero_most_significant",
    "qiskit_statevector_order": "qubit_zero_least_significant",
    "statevector_comparison": "reverse_qiskit_tensor_axes_before_comparison",
    "parameter_identity": "unique_name",
    "global_phase": "preserved",
    "loss_policy": "fail_closed_unless_allow_lossy_true",
}


def conversion_issue_codes(path: Path = CONVERSION) -> set[str]:
    """Extract report codes emitted by the adapter without importing Qiskit."""

    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        str(node.args[0].value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "QiskitConversionIssue"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    }


def load_toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def contract_errors(
    contract: dict[str, Any], dependency_policy: dict[str, Any]
) -> tuple[str, ...]:
    errors: list[str] = []
    if contract.get("schema") != EXPECTED_SCHEMA:
        errors.append(f"Qiskit interop schema must be {EXPECTED_SCHEMA}")
    if contract.get("ir_version") != IR_VERSION:
        errors.append("Qiskit interop contract must target the current IR version")
    if contract.get("maturity") != "experimental":
        errors.append("Qiskit interop v1 must remain experimental")
    if contract.get("dependency_extra") != "qiskit":
        errors.append("Qiskit interop must use only the qiskit extra")
    if contract.get("core_import_allowed") is not False:
        errors.append("Qiskit core import must remain forbidden")

    tested = dependency_policy.get("tested", {})
    versions = contract.get("qiskit_versions")
    if not isinstance(tested, dict) or versions != tested.get("qiskit"):
        errors.append("Qiskit version lanes must match dependency-policy.toml")
    if versions != ["2.0", "2.5"]:
        errors.append("Qiskit certification lanes must be exactly 2.0 and 2.5")
    if contract.get("aer_versions") != ["0.17"]:
        errors.append("Qiskit Aer certification lane must be 0.17")

    semantics = contract.get("semantics")
    if semantics != EXPECTED_SEMANTICS:
        errors.append("Qiskit wire, statevector, parameter, or loss semantics drifted")

    unsupported = contract.get("unsupported")
    if not isinstance(unsupported, dict):
        errors.append("Qiskit contract is missing [unsupported]")
        unsupported = {}
    excluded = unsupported.get("flagquantum_opcodes")
    if not isinstance(excluded, list) or not all(
        isinstance(name, str) for name in excluded
    ):
        errors.append("unsupported FlagQuantum opcodes must be a string list")
        excluded_names: set[str] = set()
    else:
        excluded_names = set(excluded)

    declared_issue_codes = unsupported.get("issue_codes")
    emitted_issue_codes = conversion_issue_codes()
    if (
        not isinstance(declared_issue_codes, list)
        or set(declared_issue_codes) != emitted_issue_codes
    ):
        errors.append(
            "Qiskit conversion issue-code coverage drifted: "
            f"declared={sorted(declared_issue_codes or [])}, "
            f"emitted={sorted(emitted_issue_codes)}"
        )

    operations = contract.get("operations")
    if not isinstance(operations, list):
        return tuple(errors + ["Qiskit contract operations must be an array"])
    seen_qiskit: set[str] = set()
    seen_flagquantum: set[str] = set()
    for index, operation in enumerate(operations):
        if not isinstance(operation, dict):
            errors.append(f"Qiskit operation {index} must be a table")
            continue
        qiskit_name = operation.get("qiskit")
        flagquantum_name = operation.get("flagquantum")
        if not isinstance(qiskit_name, str) or not isinstance(flagquantum_name, str):
            errors.append(f"Qiskit operation {index} has invalid names")
            continue
        if qiskit_name in seen_qiskit:
            errors.append(f"duplicate Qiskit operation {qiskit_name!r}")
        if flagquantum_name in seen_flagquantum:
            errors.append(f"duplicate FlagQuantum operation {flagquantum_name!r}")
        seen_qiskit.add(qiskit_name)
        seen_flagquantum.add(flagquantum_name)
        schema = OPERATOR_SCHEMAS.get(flagquantum_name)
        if schema is None:
            errors.append(f"unknown FlagQuantum operation {flagquantum_name!r}")
            continue
        parameters = operation.get("flagquantum_parameters")
        if parameters != list(schema.parameters):
            errors.append(
                f"FlagQuantum parameter order drifted for {flagquantum_name!r}"
            )
        qiskit_parameters = operation.get("qiskit_parameters")
        if not isinstance(qiskit_parameters, list) or len(qiskit_parameters) != len(
            schema.parameters
        ):
            errors.append(f"Qiskit parameter arity drifted for {qiskit_name!r}")

    expected_supported = set(OPERATOR_SCHEMAS) - excluded_names
    if seen_flagquantum != expected_supported:
        errors.append(
            "Qiskit bidirectional opcode coverage drifted: "
            f"missing={sorted(expected_supported - seen_flagquantum)}, "
            f"unexpected={sorted(seen_flagquantum - expected_supported)}"
        )

    verification = contract.get("verification")
    if not isinstance(verification, dict):
        errors.append("Qiskit contract is missing [verification]")
    else:
        for name, raw_path in verification.items():
            if not isinstance(raw_path, str) or not (ROOT / raw_path).is_file():
                errors.append(f"Qiskit verification path {name!r} does not exist")
    return tuple(errors)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--dependency-policy", type=Path, default=DEPENDENCY_POLICY)
    args = parser.parse_args(argv)
    errors = contract_errors(
        load_toml(args.contract), load_toml(args.dependency_policy)
    )
    if errors:
        print("\n".join(errors))
        return 1
    print("Qiskit interoperability contract passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
