#!/usr/bin/env python3
"""Validate the machine-readable OpenQASM import contract against the importer.

``contracts/openqasm-import-v1-candidate.json`` states three kinds of claim:
the refusal vocabulary, the supported version lanes, and eight behavioral
rules. A claim nobody reads is decoration, so this gate is the reader for the
ones no test reads, and the ledger of readers for the ones a test does read.

It reconciles the contract with the shipped importer rather than with a second
copy of the same facts:

* ``refusal_issue_codes`` is compared with ``openqasm_import._ISSUE_CODES``,
  and every declared code must appear as a literal argument of ``_refuse(...)``
  in that module, so the vocabulary cannot drift from what the importer raises;
* ``supported_versions`` is compared with ``openqasm_import.SUPPORTED_VERSIONS``;
* ``signatures.from_openqasm`` is compared with the signature the root
  namespace really exposes;
* ``root_additions`` are required to be exported and ``root_removals`` to be
  absent, so the interchange cannot quietly widen or narrow the Stable Core;
* every key of ``rules`` must name at least one reader below, either a check in
  this file or a test function that exists in the witness test module, so a new
  rule cannot be added without the code that enforces it.
"""

from __future__ import annotations

import argparse
import ast
import inspect
import json
from pathlib import Path
from typing import Any

import flagquantum as fq
from flagquantum.compiler import openqasm_import
from flagquantum.core.ir import IR_VERSION

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "contracts" / "openqasm-import-v1-candidate.json"
IMPORTER = ROOT / "flagquantum" / "compiler" / "openqasm_import.py"
GATE_TABLES = ROOT / "flagquantum" / "compiler" / "openqasm_gates.py"
PACKAGE = ROOT / "flagquantum"
WITNESS_TESTS = ROOT / "tests" / "test_openqasm_import.py"

EXPECTED_SCHEMA_VERSION = "1.0"
EXPECTED_STATUS = "implementation_authorized"
EXPECTED_ROOT_ADDITIONS = ("from_openqasm",)
EXPECTED_ROOT_REMOVALS: tuple[str, ...] = ()
REQUIRED_VERSION_LANES = (2.0, 3.0)
EXPECTED_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "approval",
        "implementation_authorized",
        "root_additions",
        "root_removals",
        "signatures",
        "refusal_issue_codes",
        "supported_versions",
        "rules",
        "qubit_naming_migration",
    }
)
GATE_TABLE_NAMES = ("EMITTED_GATES", "FIXED_GATES", "PARAMETERIZED_GATES")
SHARED_TABLE_MODULE = GATE_TABLES.relative_to(ROOT).as_posix()

# The importer is a construction-time reader. It may depend on the core IR and
# on the shared gate spelling tables, and on nothing that plans or executes.
IMPORTER_ALLOWED_PREFIXES = (
    "flagquantum.compiler",
    "flagquantum.core",
    "flagquantum.errors",
)
IMPORTER_ALLOWED_STDLIB = frozenset(
    {"__future__", "dataclasses", "math", "re", "typing"}
)

# `rules` is the part of the contract that has no natural home in a test file.
# Each entry names either a check in this module (`tool:<name>`) or a test
# function that must exist in WITNESS_TESTS. A rule absent from this mapping
# fails the gate, which is what stops a new rule from being written as prose.
RULE_READERS: dict[str, tuple[str, ...]] = {
    "accepts_only_canonical_subset": (
        "test_openqasm3_three_angle_u_gate_is_the_canonical_u3_spelling",
        "test_refusals_are_fail_closed_and_name_an_issue_code",
    ),
    "refusals_name_an_issue_code": (
        "test_refusals_are_fail_closed_and_name_an_issue_code",
        "test_declared_refusal_codes_are_closed_and_every_one_is_reachable",
    ),
    "import_is_construction_time_only": ("tool:construction_time_only",),
    "ir_version_unchanged": ("tool:ir_version_unchanged",),
    "bound_parameters_only": ("test_refusals_are_fail_closed_and_name_an_issue_code",),
    "terminal_measurement_required": (
        "test_terminal_measurement_records_which_qubit_each_classical_bit_reads",
        "test_unsupported_statements_after_the_measurement_block_are_refused",
    ),
    "single_quantum_register_required": (
        "test_a_second_quantum_register_declaration_is_refused",
    ),
    "single_classical_register_required": (
        "test_a_second_classical_register_declaration_is_refused",
    ),
    "duplicate_emitter_implementation": ("tool:gate_tables_defined_once",),
}

TOOL_CHECKS = (
    "construction_time_only",
    "gate_tables_defined_once",
    "ir_version_unchanged",
)


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _module_level_assignment_names(path: Path) -> set[str]:
    """Names assigned at module level, including annotated assignments."""

    names: set[str] = set()
    for node in _tree(path).body:
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        for target in targets:
            if isinstance(target, ast.Name):
                names.add(target.id)
    return names


def _raise_literal_codes(path: Path) -> tuple[set[str], list[str]]:
    """Issue codes passed as a literal to ``_refuse``, plus non-literal calls."""

    codes: set[str] = set()
    dynamic: list[str] = []
    for node in ast.walk(_tree(path)):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Name) and node.func.id == "_refuse"):
            continue
        if (
            node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            codes.add(node.args[0].value)
        else:
            dynamic.append(f"line {node.lineno}")
    return codes, dynamic


def _imported_roots(path: Path) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module)
    return roots


def _witness_test_names(path: Path = WITNESS_TESTS) -> set[str]:
    return {
        node.name
        for node in _tree(path).body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    }


def construction_time_only(importer: Path = IMPORTER) -> tuple[str, ...]:
    """The importer may construct a program; it may not plan or run one."""

    errors: list[str] = []
    for root in sorted(_imported_roots(importer)):
        if root in IMPORTER_ALLOWED_STDLIB:
            continue
        if not root.startswith(IMPORTER_ALLOWED_PREFIXES):
            errors.append(
                f"the OpenQASM importer must not depend on {root!r}; it is a "
                "construction-time reader, not a planner or an executor"
            )
    for name in ("plan", "run", "ExecutionOptions", "ExecutionPlan", "Circuit"):
        if name in _module_level_assignment_names(importer):
            errors.append(f"the OpenQASM importer must not define {name!r}")
    return tuple(errors)


def gate_tables_defined_once(package: Path = PACKAGE) -> tuple[str, ...]:
    """Emission and import must share one copy of each gate spelling table."""

    errors: list[str] = []
    for name in GATE_TABLE_NAMES:
        owners = sorted(
            path.relative_to(ROOT).as_posix()
            for path in package.rglob("*.py")
            if name in _module_level_assignment_names(path)
        )
        if owners != [SHARED_TABLE_MODULE]:
            errors.append(
                f"{name} must be defined exactly once, in {SHARED_TABLE_MODULE}; "
                f"found {owners}"
            )
    return tuple(errors)


def ir_version_unchanged(
    contract: dict[str, Any], ir_version: str = IR_VERSION
) -> tuple[str, ...]:
    """Interchange added no IR field and therefore bumped no IR version."""

    errors: list[str] = []
    if ir_version != "1.0":
        errors.append(
            f"the interchange contract assumes IR version 1.0, found {ir_version}"
        )
    if "ir_version" in contract:
        errors.append(
            "the OpenQASM import contract must not pin an IR version; the "
            "interchange added no IR field and IR_VERSION is unchanged"
        )
    return tuple(errors)


def contract_errors(
    contract: dict[str, Any],
    *,
    exported: tuple[str, ...] = tuple(fq.__all__),
    signature: str | None = None,
    issue_codes: tuple[str, ...] = tuple(openqasm_import._ISSUE_CODES),
    versions: tuple[float, ...] = tuple(openqasm_import.SUPPORTED_VERSIONS),
    witness_tests: set[str] | None = None,
) -> tuple[str, ...]:
    """Reconcile one loaded contract with the shipped importer."""

    if signature is None:
        signature = str(inspect.signature(fq.from_openqasm))
    errors: list[str] = []
    if contract.get("schema_version") != EXPECTED_SCHEMA_VERSION:
        errors.append(
            f"OpenQASM import schema_version must be {EXPECTED_SCHEMA_VERSION}"
        )
    if contract.get("status") != EXPECTED_STATUS:
        errors.append(f"OpenQASM import status must be {EXPECTED_STATUS}")
    if contract.get("implementation_authorized") is not True:
        errors.append("OpenQASM import must declare implementation_authorized")
    unexpected = sorted(set(contract) - EXPECTED_KEYS)
    if unexpected:
        errors.append(f"OpenQASM import contract has unread keys {unexpected}")
    missing = sorted(EXPECTED_KEYS - set(contract))
    if missing:
        errors.append(f"OpenQASM import contract is missing keys {missing}")

    approval = contract.get("approval")
    if not isinstance(approval, dict):
        errors.append("OpenQASM import contract needs an [approval] record")
    else:
        record = approval.get("approval_record")
        if not isinstance(record, str) or not (ROOT / record).is_file():
            errors.append(f"approval record {record!r} does not exist")
        if not approval.get("approved_at") or not approval.get("approved_by"):
            errors.append("the approval record needs approved_at and approved_by")

    additions = contract.get("root_additions")
    removals = contract.get("root_removals")
    if tuple(additions or ()) != EXPECTED_ROOT_ADDITIONS:
        errors.append(
            f"root_additions must be {list(EXPECTED_ROOT_ADDITIONS)}, found {additions}"
        )
    if tuple(removals or ()) != EXPECTED_ROOT_REMOVALS:
        errors.append(f"root_removals must be empty, found {removals}")
    for name in additions or ():
        if name not in exported:
            errors.append(f"declared root addition {name!r} is not exported")
    for name in removals or ():
        if name in exported:
            errors.append(f"declared root removal {name!r} is still exported")

    declared_signatures = contract.get("signatures")
    if not isinstance(declared_signatures, dict) or set(declared_signatures) != set(
        additions or ()
    ):
        errors.append("signatures must declare every root addition exactly once")
    elif declared_signatures.get("from_openqasm") != signature:
        errors.append(
            "the declared from_openqasm signature does not match the exposed one: "
            f"declared {declared_signatures.get('from_openqasm')!r}, exposed {signature!r}"
        )

    declared_codes = contract.get("refusal_issue_codes")
    if not isinstance(declared_codes, list) or not all(
        isinstance(code, str) for code in declared_codes
    ):
        errors.append("refusal_issue_codes must be a list of strings")
    else:
        if declared_codes != sorted(declared_codes):
            errors.append("refusal_issue_codes must be listed in sorted order")
        if len(declared_codes) != len(set(declared_codes)):
            errors.append("refusal_issue_codes must not repeat a code")
        if sorted(declared_codes) != sorted(issue_codes):
            errors.append(
                "the declared refusal vocabulary differs from the importer's: "
                f"contract {sorted(set(declared_codes) - set(issue_codes))} extra, "
                f"importer {sorted(set(issue_codes) - set(declared_codes))} unlisted"
            )

    declared_versions = contract.get("supported_versions")
    if not isinstance(declared_versions, list):
        errors.append("supported_versions must be a list")
    else:
        if declared_versions != sorted(declared_versions):
            errors.append("supported_versions must be listed in ascending order")
        if [float(version) for version in declared_versions] != list(versions):
            errors.append(
                f"declared version lanes {declared_versions} do not match the "
                f"importer's {list(versions)}"
            )
        for lane in REQUIRED_VERSION_LANES:
            if lane not in [float(version) for version in declared_versions]:
                errors.append(f"the OpenQASM {lane} lane must remain supported")

    migration = contract.get("qubit_naming_migration")
    if not isinstance(migration, str) or not (ROOT / migration).is_file():
        errors.append(f"qubit_naming_migration {migration!r} does not exist")

    rules = contract.get("rules")
    if not isinstance(rules, dict) or not rules:
        errors.append("the OpenQASM import contract must declare its rules")
    else:
        unread = sorted(set(rules) - set(RULE_READERS))
        if unread:
            errors.append(f"rules with no reader in this gate: {unread}")
        for name, readers in RULE_READERS.items():
            if name not in rules:
                errors.append(f"reader {name!r} names a rule the contract dropped")
                continue
            for reader in readers:
                if reader.startswith("tool:"):
                    if reader.removeprefix("tool:") not in TOOL_CHECKS:
                        errors.append(f"rule {name!r} names unknown check {reader!r}")
                elif reader not in (
                    witness_tests
                    if witness_tests is not None
                    else _witness_test_names()
                ):
                    errors.append(
                        f"rule {name!r} names test {reader!r}, which does not exist in "
                        f"{WITNESS_TESTS.relative_to(ROOT).as_posix()}"
                    )
    return tuple(errors)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    args = parser.parse_args(argv)

    try:
        contract = json.loads(args.contract.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"OpenQASM import contract is not valid JSON: {exc}")
        return 1
    if not isinstance(contract, dict):
        print("OpenQASM import contract must be a JSON object")
        return 1

    codes, dynamic = _raise_literal_codes(IMPORTER)
    errors = list(
        contract_errors(contract, signature=str(inspect.signature(fq.from_openqasm)))
    )
    if dynamic:
        errors.append(
            "the OpenQASM importer raises a computed issue code at "
            f"{', '.join(dynamic)}; every refusal must name a declared literal code"
        )
    for code in sorted(codes):
        if code not in openqasm_import._ISSUE_CODES:
            errors.append(f"the importer refuses with the undeclared code {code!r}")
    for code in openqasm_import._ISSUE_CODES:
        if code not in codes:
            errors.append(f"the declared code {code!r} is never raised by the importer")
    errors.extend(construction_time_only())
    errors.extend(gate_tables_defined_once())
    errors.extend(ir_version_unchanged(contract))

    if errors:
        print("\n".join(sorted(set(errors))))
        return 1
    print(
        "OpenQASM import contract passed: "
        f"{len(contract['refusal_issue_codes'])} refusal codes, "
        f"{len(contract['supported_versions'])} version lanes, "
        f"{len(contract['rules'])} rules each with a reader"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
