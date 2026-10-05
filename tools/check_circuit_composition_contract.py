#!/usr/bin/env python3
"""Validate the construction-time composition contract against the implementation.

`contracts/circuit-composition-contract.toml` is the repository's own statement of what
`Circuit.compose` and `Circuit.adjoint` guarantee and of every way they refuse. This
gate reads that statement back against the code: a contracted refusal whose phrase no
longer occurs in its source, a refusal the contract does not list, a declared adjoint
rule the operator schema does not define, an operation the contract calls absent but
that someone added, an opcode census that no longer supports the unreachable row, or a
composition operation without an expand-and-compare test all fail here.

The instruction-by-instruction requirement is the reason this gate exists next to the
conformance test rather than inside it: the contract names one expansion test per
provided operation, and this gate refuses a contract that names a test which is not
there.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 and older
    import tomli as tomllib

import flagquantum as fq
from flagquantum.core import ADJOINT_RULES, OPERATOR_SCHEMAS
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.errors import CapabilityError, ValidationError

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "contracts" / "circuit-composition-contract.toml"

#: The exception class each contracted `exception` name denotes. A name absent here is a
#: contracted class this gate cannot observe, which is a failure rather than a pass.
EXCEPTIONS: dict[str, type[BaseException]] = {
    "TypeError": TypeError,
    "ValidationError": ValidationError,
    "CapabilityError": CapabilityError,
}

#: The instruction fields a second IR would have to add. `ir_version_effect = "none"`
#: claims none of them appeared, so they are read off the IR rather than trusted.
FORBIDDEN_IR_FIELDS = frozenset({"adjoint", "compose", "control", "power"})


def _load_toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _test_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def contract_errors(contract: dict[str, Any]) -> tuple[str, ...]:
    errors: list[str] = []

    expected = {
        "schema": "flagquantum_circuit_composition_contract_v1",
        "maturity": "development_evidence",
        "surface": "construction_time_program_composition",
        "ir_version_effect": "none",
        "root_export_effect": "none",
    }
    for name, value in expected.items():
        if contract.get(name) != value:
            errors.append(f"circuit composition contract {name} must be {value!r}")
    for name in (
        "implementation",
        "placement_implementation",
        "adjoint_rule_source",
    ):
        raw = contract.get(name)
        if not isinstance(raw, str) or not (ROOT / raw).is_file():
            errors.append(f"circuit composition {name} path {raw!r} does not exist")
    for raw in contract.get("authorization", ()):
        if not isinstance(raw, str) or not (ROOT / raw).is_file():
            errors.append(f"circuit composition authorization {raw!r} does not exist")

    scope = contract.get("scope", {})
    provided = list(scope.get("provided", ()))
    not_provided = list(scope.get("not_provided", ()))
    if provided != ["Circuit.compose", "Circuit.adjoint"]:
        errors.append("circuit composition provided surface drifted")
    if not_provided != ["Circuit.control", "Circuit.power"]:
        errors.append("circuit composition absent surface drifted")
    for dotted in provided:
        owner, _, attribute = dotted.partition(".")
        if attribute == "" or not hasattr(getattr(fq, owner, None), attribute):
            errors.append(f"contracted operation {dotted!r} does not exist")
    for dotted in not_provided:
        owner, _, attribute = dotted.partition(".")
        if attribute and hasattr(getattr(fq, owner, None), attribute):
            errors.append(f"contracted-absent operation {dotted!r} exists")
    for entry in ("control", "power"):
        if entry in fq.__all__:
            errors.append(f"{entry!r} is a root export but is contracted as absent")

    placement = contract.get("placement", {})
    compose_signature = inspect.signature(fq.Circuit.compose)
    arguments = [
        name
        for name, parameter in compose_signature.parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
    ]
    if arguments != list(placement.get("arguments", ())):
        errors.append(
            f"Circuit.compose keywords drifted: contract={placement.get('arguments')}, "
            f"implementation={arguments}"
        )
    for name in ("args", "kwargs"):
        if name in compose_signature.parameters:
            errors.append(f"Circuit.compose accepts a variadic {name} parameter")
    # These four flags are claims the contract makes about itself. A flag this gate does
    # not read is a flag nobody checks, so each one is named here explicitly.
    for reason in (
        "arguments_are_mutually_exclusive",
        "occupied_target_qubits_are_legal",
        "existing_instructions_preserved",
    ):
        if placement.get(reason) is not True:
            errors.append(f"circuit composition {reason} is not contracted as true")
    if placement.get("default_placement") != "identity":
        errors.append("circuit composition default placement must be the identity map")

    # The contract claims composition adds no IR field and no root export. Both are
    # claims about absence, so both are read off the structures that would have grown.
    ir_fields = {field.name for field in dataclasses.fields(Instruction)}
    ir_fields |= {field.name for field in dataclasses.fields(CircuitIR)}
    added = sorted(FORBIDDEN_IR_FIELDS & ir_fields)
    if added:
        errors.append(f"FlagQuantum IR gained composition fields: {added}")
    exported = sorted(
        name for name in fq.__all__ if "compose" in name or "adjoint" in name
    )
    if exported:
        errors.append(f"composition reached the root exports: {exported}")

    adjoint = contract.get("adjoint", {})
    if list(adjoint.get("declared_opcode_rules", ())) != list(ADJOINT_RULES):
        errors.append(
            "circuit composition declared adjoint rules drifted from ADJOINT_RULES: "
            f"contract={adjoint.get('declared_opcode_rules')}, "
            f"implementation={list(ADJOINT_RULES)}"
        )
    if adjoint.get("matrix_route_precedes_opcode_rule") is not True:
        errors.append("circuit composition must contract the matrix route as preceding")
    if adjoint.get("matrix_route") != "conjugate_transpose":
        errors.append("circuit composition matrix route must be conjugate_transpose")
    adjoint_arguments = list(inspect.signature(fq.Circuit.adjoint).parameters)
    if adjoint_arguments != ["self"]:
        errors.append(f"Circuit.adjoint grew parameters: {adjoint_arguments}")

    refused = _refusal_errors(contract)
    errors.extend(refused)
    errors.extend(_vocabulary_errors(contract))
    errors.extend(_relabelling_errors(contract))
    errors.extend(_census_errors(contract))
    errors.extend(_verification_errors(contract))
    return tuple(errors)


def _refusal_errors(contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    refusals = contract.get("refusals", ())
    codes = [row.get("code") for row in refusals]
    duplicates = sorted({code for code in codes if codes.count(code) > 1})
    if duplicates:
        errors.append(f"circuit composition refusal codes repeat: {duplicates}")
    entry_points = {row.get("entry_point") for row in refusals}
    if not entry_points <= {"compose", "adjoint"}:
        errors.append(
            f"circuit composition refusals name unknown entry points: {entry_points}"
        )
    if len(refusals) != 17:
        errors.append(
            f"circuit composition must contract 17 refusals, found {len(refusals)}"
        )
    for row in refusals:
        code = row.get("code")
        if not isinstance(code, str) or not code:
            errors.append(f"circuit composition refusal {row!r} has no code")
            continue
        if row.get("entry_point") not in {"compose", "adjoint"}:
            errors.append(f"circuit composition refusal {code!r} has no entry point")
        name = row.get("exception")
        if name not in EXCEPTIONS:
            errors.append(
                f"circuit composition refusal {code!r} names unknown {name!r}"
            )
        if not isinstance(row.get("reason", row.get("trigger")), str) or not (
            row.get("reason") or row.get("trigger")
        ):
            errors.append(f"circuit composition refusal {code!r} states no cause")
        reachable = row.get("reachable")
        if not isinstance(reachable, bool):
            errors.append(f"circuit composition refusal {code!r} has no reachable flag")
        elif reachable:
            if not isinstance(row.get("trigger"), str) or not row.get("trigger"):
                errors.append(
                    f"circuit composition refusal {code!r} is reachable with no trigger"
                )
        elif not isinstance(row.get("reachability_note"), str) or not row.get(
            "reachability_note"
        ):
            errors.append(
                f"circuit composition refusal {code!r} is unreachable with no reachability note"
            )
    return errors


def _module_of(path: Path, node: ast.ImportFrom, *, root: Path) -> str:
    """Resolve one `from ... import ...` to the dotted module it names.

    A relative import is resolved against the importing file's own package, so
    `from ..core.qubit_mapping import remap_qubits` in `flagquantum/twin/region_model.py`
    and `from flagquantum.core.qubit_mapping import remap_qubits` elsewhere both name the
    same module. Comparing resolved names is what makes the reader set a measurement
    instead of a spelling contest.
    """

    if not node.level:
        return node.module or ""
    # For a module the containing directory is its package; for a package's own
    # `__init__.py` the containing directory is that package too. One dot is the package
    # itself, so each further dot steps one directory up.
    package = path.relative_to(root).parts[:-1]
    base = package[: len(package) - (node.level - 1)]
    return ".".join((*base, node.module or ""))


def _package_readers(module: str, owner: str, *, root: Path) -> list[str]:
    """Every module under `flagquantum/` that imports `owner` from `module`."""

    readers: list[str] = []
    for path in sorted((root / "flagquantum").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if _module_of(path, node, root=root) != module:
                continue
            if any(alias.name == owner for alias in node.names):
                readers.append(str(path.relative_to(root).as_posix()))
                break
    return readers


def _relabelling_errors(contract: dict[str, Any], *, root: Path = ROOT) -> list[str]:
    """The internal-only row records who reads the relabelling rule it names.

    Two implementations of one relabelling rule are more dangerous than two different
    rules, because the copies drift apart silently. The row already names the rule's
    module and owner, so its readers are read back off the import graph: a package
    module that reads the rule without being recorded, a recorded reader that stopped
    reading it, and the retired private wording reappearing anywhere in the package all
    fail here. The scan is the shipped package -- a test may call the rule directly
    without becoming an implementation of it. `root` is the tree to measure, which is
    what lets the conformance test measure the rule against a tree it built.
    """

    errors: list[str] = []
    rows = contract.get("issue_codes", {}).get("internal_only", ())
    if len(rows) != 1:
        return errors
    row = rows[0]
    source = row.get("source")
    owner = row.get("owner")
    declared = row.get("consumers")
    if (
        not isinstance(source, str)
        or not isinstance(owner, str)
        or not isinstance(declared, list)
        or not declared
        or any(not isinstance(name, str) or not name for name in declared)
    ):
        errors.append("circuit composition relabelling readers are not recorded")
        return errors

    recorded = sorted(dict.fromkeys(declared))
    for name in recorded:
        path = root / name
        if not path.is_file():
            errors.append(f"circuit composition relabelling reader {name!r} is missing")
            continue
        if owner not in path.read_text(encoding="utf-8"):
            errors.append(
                f"circuit composition relabelling reader {name!r} never calls {owner}"
            )

    module = source[: -len(".py")].replace("/", ".") if source.endswith(".py") else ""
    if not module:
        errors.append(
            f"circuit composition relabelling rule source {source!r} is not a module"
        )
        return errors
    measured = _package_readers(module, owner, root=root)
    if recorded != measured:
        errors.append(
            "circuit composition relabelling readers drifted from the import graph: "
            f"contract={recorded}, implementation={measured}"
        )

    retired = row.get("retired_wording")
    if isinstance(retired, str) and retired:
        for path in sorted((root / "flagquantum").rglob("*.py")):
            if retired in path.read_text(encoding="utf-8"):
                errors.append(
                    "circuit composition relabelling reader "
                    f"{path.relative_to(root).as_posix()!r} kept the retired private "
                    f"wording {retired!r}"
                )
    return errors


def _vocabulary_errors(contract: dict[str, Any]) -> list[str]:
    """Every contracted phrase must still occur in the source it names.

    The vocabulary is delivered as an exception class plus a frozen phrase, so a phrase
    that no longer occurs is a refusal whose documented handle has moved. This holds for
    the unreachable row too: nothing else observes it.
    """

    errors: list[str] = []
    issue_codes = contract.get("issue_codes", {})
    if issue_codes.get("delivery") != "exception_class_and_message_phrase":
        errors.append("circuit composition refusal delivery form drifted")
    if issue_codes.get("vocabulary_frozen") is not True:
        errors.append("circuit composition refusal vocabulary must be frozen")
    rows: list[dict[str, Any]] = list(contract.get("refusals", ()))
    rows.extend(issue_codes.get("internal_only", ()))
    for row in rows:
        phrase = row.get("message_phrase")
        source = row.get("message_source", row.get("source"))
        if not isinstance(phrase, str) or not phrase:
            errors.append(
                f"circuit composition row {row.get('code', row.get('owner'))!r} "
                "has no message phrase"
            )
            continue
        if not isinstance(source, str) or not (ROOT / source).is_file():
            errors.append(
                f"circuit composition phrase source {source!r} does not exist"
            )
            continue
        if phrase not in (ROOT / source).read_text(encoding="utf-8"):
            errors.append(
                f"circuit composition phrase {phrase!r} no longer occurs in {source}"
            )
    for row in issue_codes.get("internal_only", ()):
        if row.get("exception") != "ValueError":
            errors.append(
                "circuit composition internal-only refusal must stay a ValueError"
            )
        if not isinstance(row.get("owner"), str) or not row.get("owner"):
            errors.append("circuit composition internal-only refusal has no owner")
    return errors


def _census_errors(contract: dict[str, Any]) -> list[str]:
    """The opcode census is the measurement behind the unreachable adjoint row."""

    registered = {
        opcode for opcode, schema in OPERATOR_SCHEMAS.items() if schema.opcode == opcode
    }
    refusals = sorted(
        opcode
        for opcode in registered
        if OPERATOR_SCHEMAS[opcode].adjoint == "not_applicable"
    )
    invertible = sorted(registered - set(refusals))
    census = contract.get("verification", {}).get("opcode_census", "")
    errors: list[str] = []
    if len(invertible) + len(refusals) != len(OPERATOR_SCHEMAS):
        errors.append("operator schema census does not cover every registered opcode")
    if (
        f"{len(invertible)} of {len(OPERATOR_SCHEMAS)} registered opcodes invert"
        not in census
    ):
        errors.append(
            "circuit composition opcode census drifted: "
            f"{len(invertible)} of {len(OPERATOR_SCHEMAS)} registered opcodes invert"
        )
    contracted_unreachable = [
        row for row in contract.get("refusals", ()) if row.get("reachable") is False
    ]
    if len(contracted_unreachable) != 1:
        errors.append(
            "circuit composition must contract exactly one unreachable refusal"
        )
    elif not refusals:
        errors.append(
            "circuit composition unreachable row needs a refusing opcode family"
        )
    return errors


def _verification_errors(contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    verification = contract.get("verification", {})
    for name, value in verification.items():
        if name == "expansion_tests":
            continue
        if not isinstance(value, str) or not value:
            errors.append(
                f"circuit composition verification {name!r} is not a statement"
            )
        # A value that names a file is checked as a file; the census is a measurement
        # and is checked by its own content below.
        elif value.endswith(".py") and not (ROOT / value).is_file():
            errors.append(
                f"circuit composition verification path {value!r} does not exist"
            )

    # One expand-and-compare test per provided operation: the contract names it, and this
    # gate refuses a name that is not a test in the conformance file.
    raw_contract = verification.get("contract")
    if not isinstance(raw_contract, str) or not (ROOT / raw_contract).is_file():
        return errors
    names = _test_names(ROOT / raw_contract)
    expansions = verification.get("expansion_tests", {})
    provided = list(contract.get("scope", {}).get("provided", ()))
    missing = [operation for operation in provided if operation not in expansions]
    if missing:
        errors.append(
            f"circuit composition operations without an expansion test: {missing}"
        )
    for operation, test in expansions.items():
        if operation not in provided:
            errors.append(f"expansion test names absent operation {operation!r}")
        if test not in names:
            errors.append(f"expansion test {test!r} for {operation!r} does not exist")
    return errors


def main() -> int:
    errors = contract_errors(_load_toml(CONTRACT))
    if errors:
        print("\n".join(errors))
        return 1
    print("Circuit composition contract passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
