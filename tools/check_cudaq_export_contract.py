#!/usr/bin/env python3
"""Validate the machine-readable CUDA-Q export contract."""

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
CONTRACT = ROOT / "contracts" / "cudaq-export-contract.toml"
POLICY = ROOT / "dependency-policy.toml"
EXPECTED_VERSIONS = ["0.15.1", "0.16.0.post1"]
EXPECTED_PLATFORMS = ["linux_x86_64", "linux_aarch64", "macos_arm64_cpu_only"]
EXPECTED_SEMANTICS = {
    "artifact": "cudaq.Kernel",
    "direction": "flagquantum_ir_to_cudaq_kernel_only",
    "construction": "cudaq.make_kernel_dynamic_builder",
    "qubit_mapping": "qalloc_index_equals_flagquantum_wire",
    "operation_order": "flagquantum_ir_operation_order",
    "statevector_order": (
        "cudaq_wire_zero_is_lsb_and_requires_explicit_conformance_reordering"
    ),
    "parameters": "bound_finite_real_scalars_only",
    "loss_policy": "fail_closed_without_lossy_export",
}
EXPECTED_REQUIREMENT = "cudaq>=0.15.1,<0.17; python_version >= '3.11'"


def load_toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def contract_errors(
    contract: dict[str, Any], policy: dict[str, Any]
) -> tuple[str, ...]:
    errors: list[str] = []
    if contract.get("schema") != "flagquantum_cudaq_export_contract_v1":
        errors.append("CUDA-Q export schema drifted")
    if contract.get("ir_version") != IR_VERSION:
        errors.append("CUDA-Q contract must target current IR")
    if contract.get("implementation_status") != "contract_only":
        errors.append("CUDA-Q v1 must remain contract-only until its exporter lands")
    if contract.get("public_api_available") is not False:
        errors.append("CUDA-Q public API must remain unavailable before implementation")
    if contract.get("dependency_extra") != "cudaq":
        errors.append("CUDA-Q must use only its optional extra")
    if policy.get("extras", {}).get("cudaq") != [EXPECTED_REQUIREMENT]:
        errors.append("CUDA-Q optional dependency range drifted")
    if policy.get("classes", {}).get("heterogeneous_toolchain") != ["cudaq"]:
        errors.append("CUDA-Q must remain an isolated heterogeneous toolchain")
    if "cudaq" in policy.get("aggregates", {}).get("interop-all", ()):
        errors.append("CUDA-Q must remain outside the portable interop aggregate")
    if "cudaq" not in policy.get("import_policy", {}).get("core_forbidden_imports", ()):
        errors.append("CUDA-Q must remain forbidden from core imports")
    if contract.get("cudaq_versions") != EXPECTED_VERSIONS:
        errors.append("CUDA-Q candidate lanes must be 0.15.1 and 0.16.0.post1")
    if contract.get("supported_platforms") != EXPECTED_PLATFORMS:
        errors.append("CUDA-Q supported-platform boundary drifted")
    if contract.get("core_import_allowed") is not False:
        errors.append("CUDA-Q core import must remain forbidden")
    if contract.get("runtime_execution_allowed") is not False:
        errors.append("CUDA-Q execution must remain outside export v1")
    if contract.get("autograd_bridge_allowed") is not False:
        errors.append("CUDA-Q autograd must remain outside export v1")

    semantics = contract.get("semantics", {})
    for name, expected in EXPECTED_SEMANTICS.items():
        if semantics.get(name) != expected:
            errors.append(f"CUDA-Q semantic {name!r} drifted")

    unsupported = contract.get("unsupported", {})
    excluded = set(unsupported.get("flagquantum_opcodes", ()))
    operations = contract.get("operations", ())
    mapped = [
        operation.get("flagquantum")
        for operation in operations
        if isinstance(operation, dict)
    ]
    if len(mapped) != len(set(mapped)):
        errors.append("CUDA-Q operation mappings must be unique")
    expected = set(OPERATOR_SCHEMAS) - excluded
    if set(mapped) != expected:
        errors.append(
            "CUDA-Q opcode coverage drifted: "
            f"missing={sorted(expected-set(mapped))}, "
            f"unexpected={sorted(set(mapped)-expected)}"
        )
    for operation in operations:
        if not isinstance(operation, dict):
            errors.append("CUDA-Q operations must be tables")
            continue
        for field in ("cudaq_builder_method", "cudaq_form"):
            if not isinstance(operation.get(field), str) or not operation[field]:
                errors.append(f"CUDA-Q operation is missing {field}")

    verification = contract.get("verification", {})
    policy_path = verification.get("policy")
    if not isinstance(policy_path, str) or not (ROOT / policy_path).is_file():
        errors.append("CUDA-Q policy verification path does not exist")
    if verification.get("sdk_lane_deferred_until_implementation") is not True:
        errors.append("CUDA-Q SDK lane must remain explicitly deferred")
    if (ROOT / "flagquantum" / "ecosystem" / "cudaq").exists():
        errors.append("CUDA-Q adapter exists while contract declares contract-only")
    return tuple(errors)


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser().parse_args(argv)
    errors = contract_errors(load_toml(CONTRACT), load_toml(POLICY))
    if errors:
        print("\n".join(errors))
        return 1
    print("CUDA-Q export contract passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
