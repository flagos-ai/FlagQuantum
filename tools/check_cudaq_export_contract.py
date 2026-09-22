#!/usr/bin/env python3
"""Validate the machine-readable CUDA-Q export contract."""

from __future__ import annotations

import argparse
from importlib import import_module
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
EXPECTED_PLATFORMS = ["linux_x86_64", "linux_aarch64"]
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
    if contract.get("implementation_status") != "implemented":
        errors.append("CUDA-Q v1 implementation status must remain implemented")
    if contract.get("public_api_available") is not True:
        errors.append("CUDA-Q public API must remain available after implementation")
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
    for name, expected_semantic in EXPECTED_SEMANTICS.items():
        if semantics.get(name) != expected_semantic:
            errors.append(f"CUDA-Q semantic {name!r} drifted")

    unsupported = contract.get("unsupported", {})
    excluded = set(unsupported.get("flagquantum_opcodes", ()))
    operations = contract.get("operations", ())
    mapped = [
        operation["flagquantum"]
        for operation in operations
        if isinstance(operation, dict) and isinstance(operation.get("flagquantum"), str)
    ]
    if len(mapped) != len(set(mapped)):
        errors.append("CUDA-Q operation mappings must be unique")
    expected_opcodes = set(OPERATOR_SCHEMAS) - excluded
    if set(mapped) != expected_opcodes:
        errors.append(
            "CUDA-Q opcode coverage drifted: "
            f"missing={sorted(expected_opcodes-set(mapped))}, "
            f"unexpected={sorted(set(mapped)-expected_opcodes)}"
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
    for name in ("policy", "integration", "conformance", "core_isolation"):
        path = verification.get(name)
        if not isinstance(path, str) or not (ROOT / path).is_file():
            errors.append(f"CUDA-Q {name} verification path does not exist")
    if verification.get("sdk_lane") != ".github/workflows/ci.yml#cudaq-optional":
        errors.append("CUDA-Q SDK lane reference drifted")
    if not (ROOT / "flagquantum" / "ecosystem" / "cudaq").is_dir():
        errors.append("CUDA-Q adapter implementation is missing")
    return tuple(errors)


def sdk_errors(contract: dict[str, Any]) -> tuple[str, ...]:
    try:
        cudaq = import_module("cudaq")
    except ImportError:
        return ("CUDA-Q SDK is not installed",)
    version = str(getattr(cudaq, "__version__", "unknown"))
    if version not in contract.get("cudaq_versions", ()):
        return (f"uncertified CUDA-Q SDK version {version}",)
    kernel = cudaq.make_kernel()
    kernel.qalloc(2)
    errors = []
    for operation in contract.get("operations", ()):
        method = operation.get("cudaq_builder_method")
        if not callable(getattr(kernel, str(method), None)):
            errors.append(f"CUDA-Q kernel builder method {method!r} is unavailable")
    return tuple(errors)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-sdk", action="store_true")
    args = parser.parse_args(argv)
    contract = load_toml(CONTRACT)
    errors = contract_errors(contract, load_toml(POLICY))
    if args.verify_sdk:
        errors += sdk_errors(contract)
    if errors:
        print("\n".join(errors))
        return 1
    print("CUDA-Q export contract passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
