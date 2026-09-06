#!/usr/bin/env python3
"""Validate the machine-readable PennyLane interoperability contract."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

from flagquantum.core.ir import IR_VERSION
from flagquantum.core.operator_schema import OPERATOR_SCHEMAS
from flagquantum.ecosystem.pennylane.conversion import _PENNYLANE_TO_FLAGQUANTUM

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "contracts" / "pennylane-interop-contract.toml"
POLICY = ROOT / "dependency-policy.toml"
CONVERSION = ROOT / "flagquantum/ecosystem/pennylane/conversion.py"


def load_toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def conversion_issue_codes(path: Path = CONVERSION) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        str(node.args[0].value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "PennyLaneConversionIssue"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    }


def contract_errors(
    contract: dict[str, Any], policy: dict[str, Any]
) -> tuple[str, ...]:
    errors: list[str] = []
    if contract.get("schema") != "flagquantum_pennylane_interop_contract_v1":
        errors.append("PennyLane interop schema drifted")
    if contract.get("ir_version") != IR_VERSION:
        errors.append("PennyLane contract must target current IR")
    if contract.get("dependency_extra") != "pennylane":
        errors.append("PennyLane must use only its optional extra")
    if contract.get("core_import_allowed") is not False:
        errors.append("PennyLane core import must remain forbidden")
    if contract.get("runtime_execution_allowed") is not False:
        errors.append("PennyLane runtime execution must remain outside v1")
    versions = contract.get("pennylane_versions")
    if versions != ["0.44.1", "0.45.1"] or versions != policy.get("tested", {}).get(
        "pennylane"
    ):
        errors.append("PennyLane certification lanes must be 0.44.1 and 0.45.1")
    unsupported = contract.get("unsupported", {})
    declared = set(unsupported.get("issue_codes", ()))
    emitted = conversion_issue_codes()
    if declared != emitted:
        errors.append(
            f"PennyLane issue-code coverage drifted: declared={sorted(declared)}, emitted={sorted(emitted)}"
        )
    excluded = set(unsupported.get("flagquantum_opcodes", ()))
    operations = contract.get("operations", ())
    pairs = {
        item.get("pennylane"): item.get("flagquantum")
        for item in operations
        if isinstance(item, dict)
    }
    mapped = list(pairs.values())
    if len(mapped) != len(set(mapped)):
        errors.append("PennyLane operation mappings must be unique")
    expected = set(OPERATOR_SCHEMAS) - excluded
    if set(mapped) != expected:
        errors.append(
            f"PennyLane opcode coverage drifted: missing={sorted(expected-set(mapped))}, unexpected={sorted(set(mapped)-expected)}"
        )
    if pairs != _PENNYLANE_TO_FLAGQUANTUM:
        errors.append("PennyLane operation contract and adapter mapping drifted")
    parameter_contract = contract.get("parameter_contract", {})
    expected_parameters = {
        pennylane: list(OPERATOR_SCHEMAS[flagquantum].parameters)
        for pennylane, flagquantum in pairs.items()
        if flagquantum in OPERATOR_SCHEMAS
    }
    if parameter_contract != expected_parameters:
        errors.append("PennyLane parameter names or order drifted")
    for raw_path in contract.get("verification", {}).values():
        if not (ROOT / str(raw_path)).is_file():
            errors.append(f"PennyLane verification path does not exist: {raw_path}")
    return tuple(errors)


def main() -> int:
    errors = contract_errors(load_toml(CONTRACT), load_toml(POLICY))
    if errors:
        print("\n".join(errors))
        return 1
    print("PennyLane interoperability contract passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
