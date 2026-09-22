#!/usr/bin/env python3
"""Validate and render the cross-framework interoperability gap matrix."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

from flagquantum.core.operator_schema import OPERATOR_SCHEMAS

ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = ROOT / "contracts" / "interop-capability-gap-matrix.toml"
EXPECTED_FRAMEWORKS = ("qiskit", "cirq", "pennylane", "cudaq", "braket")
EXPECTED_CAPABILITIES = (
    "circuit_conversion",
    "operation_coverage",
    "bound_parameters",
    "symbolic_parameters",
    "measurements",
    "noise_channels",
    "dynamic_control",
    "custom_unitary",
    "autograd_bridge",
    "runtime_execution",
)
VALID_STATUSES = ("supported", "partial", "unsupported", "out_of_scope")
VALID_DIRECTIONS = ("bidirectional", "export_only")


def load_toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _resolve_path(payload: Mapping[str, Any], dotted_path: str) -> Any:
    current: Any = payload
    for part in dotted_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise KeyError(dotted_path)
        current = current[part]
    return current


def _frameworks(matrix: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    raw = matrix.get("frameworks", ())
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return ()
    return tuple(item for item in raw if isinstance(item, Mapping))


def _capabilities(framework: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    raw = framework.get("capabilities", ())
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return ()
    return tuple(item for item in raw if isinstance(item, Mapping))


def matrix_errors(matrix: Mapping[str, Any], *, root: Path = ROOT) -> tuple[str, ...]:
    """Return fail-closed validation errors for a parsed matrix."""

    errors: list[str] = []
    if matrix.get("schema") != "flagquantum_interop_capability_gap_matrix_v1":
        errors.append("interoperability capability-gap schema drifted")
    if matrix.get("maturity") != "experimental":
        errors.append("interoperability capability-gap matrix must remain experimental")
    if matrix.get("source_of_truth") != "adapter_contracts":
        errors.append("adapter contracts must remain the capability matrix source")
    if tuple(matrix.get("status_values", ())) != VALID_STATUSES:
        errors.append("capability status vocabulary drifted")
    if tuple(matrix.get("capability_order", ())) != EXPECTED_CAPABILITIES:
        errors.append("capability inventory or ordering drifted")

    frameworks = _frameworks(matrix)
    names = [str(item.get("name", "")) for item in frameworks]
    if tuple(names) != EXPECTED_FRAMEWORKS:
        errors.append(
            "framework inventory or ordering drifted: "
            f"expected={list(EXPECTED_FRAMEWORKS)}, actual={names}"
        )

    seen_contracts: set[str] = set()
    canonical_opcodes = set(OPERATOR_SCHEMAS)
    for framework in frameworks:
        name = str(framework.get("name", ""))
        contract_name = framework.get("contract")
        if not isinstance(contract_name, str) or not contract_name.startswith(
            "contracts/"
        ):
            errors.append(f"{name}: contract must be a repository contract path")
            continue
        if contract_name in seen_contracts:
            errors.append(f"{name}: adapter contract is reused: {contract_name}")
        seen_contracts.add(contract_name)
        contract_path = root / contract_name
        if not contract_path.is_file():
            errors.append(f"{name}: adapter contract does not exist: {contract_name}")
            continue
        contract = load_toml(contract_path)
        if contract.get("maturity") != "experimental":
            errors.append(f"{name}: adapter contract maturity drifted")
        if framework.get("direction") not in VALID_DIRECTIONS:
            errors.append(f"{name}: conversion direction is invalid")
        next_gap = framework.get("next_gap")
        if not isinstance(next_gap, str) or not next_gap.strip():
            errors.append(f"{name}: next_gap must explain the next bounded slice")

        capabilities = _capabilities(framework)
        capability_names = [str(item.get("name", "")) for item in capabilities]
        if tuple(capability_names) != EXPECTED_CAPABILITIES:
            errors.append(
                f"{name}: capability inventory or ordering drifted: "
                f"expected={list(EXPECTED_CAPABILITIES)}, actual={capability_names}"
            )
        by_name = {str(item.get("name", "")): item for item in capabilities}
        for capability_name, capability in by_name.items():
            if capability.get("status") not in VALID_STATUSES:
                errors.append(f"{name}.{capability_name}: invalid status")
            reason = capability.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                errors.append(f"{name}.{capability_name}: reason is required")
            evidence = capability.get("evidence")
            if not isinstance(evidence, list) or not evidence:
                errors.append(f"{name}.{capability_name}: evidence is required")
                continue
            for path in evidence:
                if not isinstance(path, str) or not path:
                    errors.append(f"{name}.{capability_name}: invalid evidence path")
                    continue
                try:
                    _resolve_path(contract, path)
                except KeyError:
                    errors.append(
                        f"{name}.{capability_name}: evidence path drifted: {path}"
                    )

        operations = contract.get("operations", ())
        mapped = {
            str(item["flagquantum"])
            for item in operations
            if isinstance(item, Mapping) and isinstance(item.get("flagquantum"), str)
        }
        unsupported = contract.get("unsupported", {})
        excluded = (
            set(unsupported.get("flagquantum_opcodes", ()))
            if isinstance(unsupported, Mapping)
            else set()
        )
        if mapped & excluded:
            errors.append(f"{name}: mapped and unsupported opcodes overlap")
        if mapped | excluded != canonical_opcodes:
            errors.append(
                f"{name}: operation partition drifted: "
                f"missing={sorted(canonical_opcodes - mapped - excluded)}, "
                f"unknown={sorted((mapped | excluded) - canonical_opcodes)}"
            )
        coverage = by_name.get("operation_coverage", {})
        expected_coverage_status = "supported" if not excluded else "partial"
        if coverage.get("status") != expected_coverage_status:
            errors.append(
                f"{name}.operation_coverage: status must be "
                f"{expected_coverage_status!r} for its opcode partition"
            )
        conversion = by_name.get("circuit_conversion", {})
        expected_conversion_status = (
            "supported" if framework.get("direction") == "bidirectional" else "partial"
        )
        if conversion.get("status") != expected_conversion_status:
            errors.append(
                f"{name}.circuit_conversion: status must be "
                f"{expected_conversion_status!r} for {framework.get('direction')!r}"
            )

    return tuple(errors)


def build_report(matrix: Mapping[str, Any], *, root: Path = ROOT) -> dict[str, Any]:
    """Build the derived machine-readable report after validation."""

    errors = matrix_errors(matrix, root=root)
    if errors:
        raise ValueError("\n".join(errors))
    canonical_opcodes = set(OPERATOR_SCHEMAS)
    reports: list[dict[str, Any]] = []
    for framework in _frameworks(matrix):
        contract_name = str(framework["contract"])
        contract = load_toml(root / contract_name)
        mapped = sorted(
            {
                str(item["flagquantum"])
                for item in contract["operations"]
                if isinstance(item, Mapping)
            }
        )
        missing = sorted(canonical_opcodes - set(mapped))
        reports.append(
            {
                "framework": framework["name"],
                "contract": contract_name,
                "direction": framework["direction"],
                "capabilities": {
                    item["name"]: {
                        "status": item["status"],
                        "reason": item["reason"],
                        "evidence": item["evidence"],
                    }
                    for item in _capabilities(framework)
                },
                "operations": {
                    "supported": mapped,
                    "missing": missing,
                    "supported_count": len(mapped),
                    "total_count": len(canonical_opcodes),
                },
                "next_gap": framework["next_gap"],
            }
        )
    return {
        "schema": "flagquantum_interop_capability_gap_report_v1",
        "generated_from": MATRIX_PATH.relative_to(ROOT).as_posix(),
        "frameworks": reports,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json", action="store_true", help="write the validated derived report"
    )
    args = parser.parse_args(argv)
    matrix = load_toml(MATRIX_PATH)
    errors = matrix_errors(matrix)
    if errors:
        print("\n".join(errors))
        return 1
    if args.json:
        print(json.dumps(build_report(matrix), indent=2, sort_keys=True))
    else:
        print("interoperability capability-gap matrix passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
