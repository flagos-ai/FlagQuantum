#!/usr/bin/env python3
"""Validate the machine-readable Amazon Braket circuit contract."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

from flagquantum.core.ir import IR_VERSION
from flagquantum.core.operator_schema import OPERATOR_SCHEMAS

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "contracts" / "braket-circuit-interop-contract.toml"
POLICY = ROOT / "dependency-policy.toml"
EXPECTED_REQUIREMENT = "amazon-braket-sdk>=1.117,<2; python_version >= '3.11'"
EXPECTED_VERSIONS = ["1.117.0", "1.127.1"]
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
EXPECTED_PLANNED_PATHS = {
    "planned_integration": "tests/test_braket_interop.py",
    "planned_conformance": "tests/test_braket_interop_conformance.py",
    "planned_core_isolation": "tests/unit/test_braket_interop_boundary.py",
}


def load_toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def contract_errors(
    contract: dict[str, Any], policy: dict[str, Any]
) -> tuple[str, ...]:
    errors: list[str] = []
    if contract.get("schema") != "flagquantum_braket_circuit_interop_contract_v1":
        errors.append("Braket circuit interop schema drifted")
    if contract.get("ir_version") != IR_VERSION:
        errors.append("Braket circuit contract must target current IR")
    if contract.get("implementation_status") != "contract_only":
        errors.append("Braket circuit v1 must remain contract-only until implemented")
    if contract.get("public_api_available") is not False:
        errors.append(
            "Braket circuit API must remain unavailable before implementation"
        )
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
    if contract.get("amazon_braket_sdk_versions") != EXPECTED_VERSIONS:
        errors.append("Braket candidate lanes must be 1.117.0 and 1.127.1")
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

    verification = contract.get("verification", {})
    policy_path = verification.get("policy")
    if not isinstance(policy_path, str) or not (ROOT / policy_path).is_file():
        errors.append("Braket policy verification path does not exist")
    for name, expected_path in EXPECTED_PLANNED_PATHS.items():
        if verification.get(name) != expected_path:
            errors.append(f"Braket {name} path drifted")
    if verification.get("sdk_lane_deferred_until_implementation") is not True:
        errors.append("Braket SDK lane must remain explicitly deferred")
    if (ROOT / "flagquantum" / "ecosystem" / "braket").exists():
        errors.append("Braket adapter exists while contract declares contract-only")
    return tuple(errors)


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser().parse_args(argv)
    errors = contract_errors(load_toml(CONTRACT), load_toml(POLICY))
    if errors:
        print("\n".join(errors))
        return 1
    print("Amazon Braket circuit interoperability contract passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
