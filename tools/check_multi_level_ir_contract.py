#!/usr/bin/env python3
"""Validate the internal multi-level IR contract against the boundary module.

``contracts/multi-level-ir-internal-v1-candidate.json`` declares the boundary
between the public IR and the internal program, quantum, and target levels. A
claim nobody reads is decoration, so this gate is the reader for the claims no
test reads, and the ledger of readers for the ones a test does read.

It reconciles the contract with the shipped boundary rather than with a second
copy of the same facts:

* the declared levels, support states, value kinds, linear kinds, terminators,
  and realized exits are compared with ``flagquantum.core.ir.levels``;
* every declared diagnostic code must be raised by a literal call somewhere in
  the internal boundary, and every literal call must be declared, so the
  vocabulary cannot drift from what the boundary refuses with;
* ``forbidden_states`` must be unreachable: absent from the declared states and
  absent from every string literal the boundary contains;
* the boundary must stay internal -- no root export, no entry in the Stable Core
  snapshot, and no pass manager, pass registry, analysis spec, executable
  artifact, or runtime adapter of its own;
* every key of ``rules`` must name at least one reader below, either a check in
  this file or a test function that exists in the witness test module.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import json
from pathlib import Path
from typing import Any

import flagquantum as fq
from flagquantum.core.ir import IR_VERSION, CircuitIR

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "contracts" / "multi-level-ir-internal-v1-candidate.json"
BOUNDARY = ROOT / "flagquantum" / "core" / "ir" / "levels.py"
BOUNDARY_MODULE = "flagquantum.core.ir.levels"
IR_PACKAGE = ROOT / "flagquantum" / "core" / "ir"
PUBLIC_IR_MODULE = IR_PACKAGE / "__init__.py"
PUBLIC_SNAPSHOT = ROOT / "docs" / "public_api_v1.json"
WITNESS_TESTS = ROOT / "tests" / "unit" / "test_multi_level_ir_contract.py"

#: Directories whose JSON is not part of the recorded public IR.
SKIPPED_DIRECTORIES = frozenset(
    {".git", ".venv", ".scratch", "__pycache__", "node_modules"}
)
CANONICAL_KIND = "flagquantum.circuit_ir"

EXPECTED_SCHEMA = "flagquantum.multi_level_ir_internal_contract"
EXPECTED_VERSION = "0.1"
EXPECTED_STATUS = "implementation_authorized"
EXPECTED_OWNER = "core"
EXPECTED_PUBLIC_LEVEL = "circuit"
EXPECTED_INTERNAL_LEVELS = ("program", "quantum", "target")
EXPECTED_REALIZED_EXITS = ("program",)
EXPECTED_SUPPORT_STATES = (
    "supported_exact",
    "unsupported_with_diagnostics",
    "invalid_input",
)
EXPECTED_FORBIDDEN_STATES = ("best_effort",)
EXPECTED_VALUE_KINDS = ("bool", "index", "scalar", "tensor", "qubit")
EXPECTED_LINEAR_VALUE_KINDS = ("qubit",)
EXPECTED_TERMINATORS = ("return", "branch")
EXPECTED_KEYS = frozenset(
    {
        "schema",
        "version",
        "status",
        "owner",
        "approval_basis",
        "decision",
        "taxonomy_source",
        "existing_authorities",
        "public_level",
        "internal_levels",
        "realized_exits",
        "declared_but_unimplemented_exits",
        "support_states",
        "forbidden_states",
        "value_kinds",
        "linear_value_kinds",
        "terminators",
        "diagnostic_codes",
        "rules",
        "exclusions",
    }
)

# Names a second boundary must not grow. Engineering decision principle 11 and
# proposal 062 section 4: extend the pass infrastructure that exists, do not
# introduce a second one.
FORBIDDEN_SECOND_BOUNDARY_NAMES = (
    "PassManager",
    "PassRegistry",
    "PassSpec",
    "AnalysisSpec",
    "AnalysisManager",
    "ExecutableArtifact",
    "RuntimeAdapter",
)

# Names the boundary must keep out of the public surface.
INTERNAL_ONLY_NAMES = (
    "ProgramRecord",
    "LevelConversion",
    "LevelDiagnostic",
    "ValueRef",
    "DIAGNOSTIC_LEVELS",
    "lower_to_level",
    "round_trip",
    "verify_program_record",
)

# Each entry names either a check in this module (`tool:<name>`) or a test
# function that must exist in WITNESS_TESTS. A rule absent from this mapping
# fails the gate, which is what stops a new rule from being written as prose.
RULE_READERS: dict[str, tuple[str, ...]] = {
    "level_entry_and_exit": (
        "tool:realized_exits_are_declared",
        "test_lower_to_level_realizes_only_declared_exits",
    ),
    "value_identity": ("test_value_identity_is_scope_and_index",),
    "linearity": (
        "test_linear_value_has_exactly_one_consumer",
        "test_linear_value_must_be_forwarded_by_a_branch",
        "test_a_returning_block_consumes_its_linear_argument",
    ),
    "joins": (
        "test_join_requires_matching_argument_kinds",
        "test_an_illegal_control_flow_graph_is_invalid_input",
    ),
    "failure_categories": (
        "test_a_supported_result_carries_no_diagnostic_and_a_refusal_carries_one",
        "test_every_refusal_names_a_declared_code",
    ),
    "round_trip": (
        "test_static_circuit_round_trips_exactly",
        "test_a_restored_circuit_is_equal_to_the_one_that_entered",
        "test_a_payload_that_is_not_canonical_is_refused",
        "tool:committed_payloads_round_trip",
    ),
    "rejection": (
        "test_dynamic_instruction_is_refused",
        "test_conditional_instruction_is_refused",
        "test_tampered_payload_is_refused",
    ),
    "no_best_effort": (
        "tool:forbidden_states_are_unreachable",
        "test_no_best_effort_state_exists",
    ),
    "no_second_pass_manager": ("tool:no_second_boundary",),
    "no_root_export": (
        "tool:no_root_export",
        "test_the_boundary_is_not_a_root_export",
    ),
}

TOOL_CHECKS = (
    "realized_exits_are_declared",
    "forbidden_states_are_unreachable",
    "no_second_boundary",
    "no_root_export",
    "committed_payloads_round_trip",
)


def refusal_sources() -> tuple[Path, ...]:
    """Return every module that belongs to the internal level boundary.

    The boundary module states the vocabulary and delegates the lowering to the
    level packages beside it, so the codes it refuses with and the states it must
    not spell live in more than one file. Reading only `levels.py` would accept a
    code that a delegated module raises without declaring, and would miss one it
    raises without declaring it. The public IR root module is the only file under
    the package that is not part of the internal boundary.
    """

    return tuple(
        sorted(path for path in IR_PACKAGE.rglob("*.py") if path != PUBLIC_IR_MODULE)
    )


def _tree(path: Path = BOUNDARY) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _module_level_assignment_names(path: Path = BOUNDARY) -> set[str]:
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


def _raised_codes(path: Path = BOUNDARY) -> tuple[set[str], list[str]]:
    """Codes passed as a literal to ``_diagnostic``, plus non-literal calls."""

    codes: set[str] = set()
    dynamic: list[str] = []
    for node in ast.walk(_tree(path)):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Name) and node.func.id == "_diagnostic"):
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


def _string_literals(path: Path = BOUNDARY) -> set[str]:
    return {
        node.value
        for node in ast.walk(_tree(path))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


def _witness_test_names(path: Path = WITNESS_TESTS) -> set[str]:
    return {
        node.name
        for node in _tree(path).body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    }


def _resolve_authority(name: str) -> str | None:
    """Return an error when one declared authority does not exist here."""

    if "/" in name or name.endswith((".json", ".md", ".toml", ".py")):
        return None if (ROOT / name).exists() else f"{name!r} does not exist"
    parts = name.split(".")
    for cut in range(len(parts), 0, -1):
        try:
            module = importlib.import_module(".".join(parts[:cut]))
        except ImportError:
            continue
        for attribute in parts[cut:]:
            if not hasattr(module, attribute):
                return f"{name!r} has no attribute {attribute!r}"
            module = getattr(module, attribute)
        return None
    return f"{name!r} resolves to no module in this tree"


def realized_exits_are_declared(
    contract: dict[str, Any], *, boundary: Any = None
) -> tuple[str, ...]:
    """Only declared levels may be exits, and each one must be reachable."""

    if boundary is None:
        boundary = importlib.import_module(BOUNDARY_MODULE)
    errors: list[str] = []
    internal = tuple(contract.get("internal_levels") or ())
    realized = tuple(contract.get("realized_exits") or ())
    unimplemented = tuple(contract.get("declared_but_unimplemented_exits") or ())
    for level in realized:
        if level not in internal:
            errors.append(f"realized exit {level!r} is not a declared internal level")
    if sorted(realized + unimplemented) != sorted(internal):
        errors.append(
            "every internal level must appear exactly once as realized or "
            f"unimplemented: realized {list(realized)}, unimplemented "
            f"{list(unimplemented)}, declared {list(internal)}"
        )
    if set(realized) & set(unimplemented):
        errors.append("a level cannot be both realized and unimplemented")
    if "circuit" in realized or "circuit" in unimplemented:
        errors.append("the public level is the entry point, not an internal exit")
    for level in unimplemented:
        try:
            conversion = boundary.lower_to_level(
                boundary.CircuitIR(n_wires=1, instructions=()), level
            )
        except Exception as exc:
            errors.append(
                f"the unimplemented exit {level!r} raised {type(exc).__name__} "
                f"instead of failing closed with diagnostics: {exc}"
            )
            continue
        if conversion.state != "unsupported_with_diagnostics":
            errors.append(
                f"the unimplemented exit {level!r} must fail closed with "
                f"diagnostics, found {conversion.state!r}"
            )
        elif {item.code for item in conversion.diagnostics} != {
            "level.unsupported_exit"
        }:
            errors.append(
                f"the unimplemented exit {level!r} must refuse with "
                "level.unsupported_exit"
            )
    return tuple(errors)


def forbidden_states_are_unreachable(
    contract: dict[str, Any], *, boundary: Any = None
) -> tuple[str, ...]:
    """No forbidden state may be declared or appear as a literal."""

    if boundary is None:
        boundary = importlib.import_module(BOUNDARY_MODULE)
    errors: list[str] = []
    declared = set(contract.get("support_states") or ())
    forbidden = tuple(contract.get("forbidden_states") or ())
    if not forbidden:
        errors.append("the contract must name the states it forbids")
    for state in forbidden:
        if state in declared:
            errors.append(
                f"the forbidden state {state!r} is also declared as supported"
            )
        if state in tuple(getattr(boundary, "SUPPORT_STATES", ())):
            errors.append(f"the boundary declares the forbidden state {state!r}")
        for source in refusal_sources():
            if state in _string_literals(source):
                errors.append(
                    f"{source.relative_to(ROOT)} contains the literal {state!r}; a "
                    "state that is not a support state must not be spelled anywhere "
                    "in the internal boundary"
                )
    return tuple(errors)


def no_second_boundary(
    contract: dict[str, Any], *, boundary: Any = None
) -> tuple[str, ...]:
    """The boundary extends the existing pass infrastructure; it replaces none."""

    if boundary is None:
        boundary = importlib.import_module(BOUNDARY_MODULE)
    errors: list[str] = []
    assigned = _module_level_assignment_names()
    for name in FORBIDDEN_SECOND_BOUNDARY_NAMES:
        if name in assigned or hasattr(boundary, name):
            errors.append(
                f"the internal level boundary must not define {name!r}; proposal "
                "062 section 4 keeps one pass infrastructure"
            )
    if "flagquantum.compiler.pass_manager.PassManager" not in json.dumps(
        contract.get("existing_authorities") or {}
    ):
        errors.append(
            "the contract must name the existing pass infrastructure it extends"
        )
    return tuple(errors)


def no_root_export(
    contract: dict[str, Any], *, snapshot: str | None = None
) -> tuple[str, ...]:
    """The boundary is internal: no root export and no snapshot entry."""

    errors: list[str] = []
    if snapshot is None:
        snapshot = PUBLIC_SNAPSHOT.read_text(encoding="utf-8")
    for name in INTERNAL_ONLY_NAMES:
        if name in fq.__all__:
            errors.append(f"the internal boundary adds the root export {name!r}")
        if f'"{name}"' in snapshot:
            errors.append(
                f"the Stable Core snapshot records the internal name {name!r}"
            )
    declared = json.dumps(contract.get("existing_authorities") or {})
    if "docs/public_api_v1.json" not in declared:
        errors.append("the contract must name the Stable Core snapshot it stays out of")
    module = importlib.import_module(BOUNDARY_MODULE)
    if hasattr(module, "__all__"):
        errors.append(
            "the boundary module must not declare __all__; it is not a public surface"
        )
    return tuple(errors)


def _committed_payloads() -> list[tuple[str, dict[str, Any]]]:
    """Every canonical `CircuitIR` payload recorded under this tree."""

    found: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(ROOT.rglob("*.json")):
        if SKIPPED_DIRECTORIES & set(path.relative_to(ROOT).parts):
            continue
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        stack: list[tuple[str, Any]] = [("", document)]
        while stack:
            where, node = stack.pop()
            if isinstance(node, dict):
                if node.get("kind") == CANONICAL_KIND:
                    found.append((f"{path.relative_to(ROOT)}{where}", node))
                for key, value in node.items():
                    stack.append((f"{where}.{key}", value))
            elif isinstance(node, list):
                for index, value in enumerate(node):
                    stack.append((f"{where}[{index}]", value))
    return found


def committed_payloads_round_trip(*, boundary: Any = None) -> tuple[str, ...]:
    """Every canonical payload recorded here must still cross and come back.

    The public IR did not change, so this is the migration evidence: the circuits
    this tree already records are read back through the level boundary and must
    return the payload they entered with. A payload that is refused on the way in,
    or that comes back different, is a broken read of the public IR rather than a
    conversion the boundary may skip.
    """

    if boundary is None:
        boundary = importlib.import_module(BOUNDARY_MODULE)
    payloads = _committed_payloads()
    if not payloads:
        return (
            "no canonical circuit payload is recorded in this tree, so nothing here "
            "demonstrates that reading the public IR still works",
        )
    errors: list[str] = []
    for where, payload in payloads:
        try:
            circuit = CircuitIR.from_dict(payload)
        except Exception as exc:
            errors.append(f"{where} no longer reads as CircuitIR: {exc}")
            continue
        conversion = boundary.circuit_to_program(circuit)
        if conversion.state != "supported_exact" or conversion.record is None:
            errors.append(
                f"{where} is refused on the way in with "
                f"{sorted({item.code for item in conversion.diagnostics})}"
            )
            continue
        restored = boundary.program_to_circuit(conversion.record)
        if restored.state != "supported_exact":
            errors.append(
                f"{where} does not come back: "
                f"{sorted({item.code for item in restored.diagnostics})}"
            )
        elif restored.canonical_payload != circuit.to_dict():
            errors.append(f"{where} comes back as a different payload")
    return tuple(errors)


def contract_errors(
    contract: dict[str, Any],
    *,
    rules: dict[str, tuple[str, ...]] = RULE_READERS,
    witness_tests: set[str] | None = None,
) -> tuple[str, ...]:
    """Reconcile one loaded contract with the shipped boundary."""

    boundary = importlib.import_module(BOUNDARY_MODULE)
    errors: list[str] = []
    if contract.get("schema") != EXPECTED_SCHEMA:
        errors.append(f"the contract schema must be {EXPECTED_SCHEMA}")
    if contract.get("version") != EXPECTED_VERSION:
        errors.append(f"the contract version must be {EXPECTED_VERSION}")
    if contract.get("status") != EXPECTED_STATUS:
        errors.append(f"the contract status must be {EXPECTED_STATUS}")
    if contract.get("owner") != EXPECTED_OWNER:
        errors.append(f"the contract owner must be {EXPECTED_OWNER}")

    unexpected = sorted(set(contract) - EXPECTED_KEYS)
    if unexpected:
        errors.append(f"the internal level contract has unread keys {unexpected}")
    missing = sorted(EXPECTED_KEYS - set(contract))
    if missing:
        errors.append(f"the internal level contract is missing keys {missing}")

    for key in ("approval_basis", "decision", "taxonomy_source"):
        value = contract.get(key)
        if not isinstance(value, str) or not (ROOT / value).is_file():
            errors.append(f"{key} {value!r} does not exist")

    authorities = contract.get("existing_authorities")
    if not isinstance(authorities, dict) or not authorities:
        errors.append("the contract must name the authorities it extends")
    else:
        for key, value in sorted(authorities.items()):
            if not isinstance(value, str):
                errors.append(f"existing_authorities.{key} must be a string")
                continue
            problem = _resolve_authority(value)
            if problem:
                errors.append(f"existing_authorities.{key}: {problem}")

    declared_levels = {
        "public_level": (contract.get("public_level"), EXPECTED_PUBLIC_LEVEL),
        "internal_levels": (
            tuple(contract.get("internal_levels") or ()),
            EXPECTED_INTERNAL_LEVELS,
        ),
        "realized_exits": (
            tuple(contract.get("realized_exits") or ()),
            EXPECTED_REALIZED_EXITS,
        ),
        "support_states": (
            tuple(contract.get("support_states") or ()),
            EXPECTED_SUPPORT_STATES,
        ),
        "forbidden_states": (
            tuple(contract.get("forbidden_states") or ()),
            EXPECTED_FORBIDDEN_STATES,
        ),
        "value_kinds": (tuple(contract.get("value_kinds") or ()), EXPECTED_VALUE_KINDS),
        "linear_value_kinds": (
            tuple(contract.get("linear_value_kinds") or ()),
            EXPECTED_LINEAR_VALUE_KINDS,
        ),
        "terminators": (
            tuple(contract.get("terminators") or ()),
            EXPECTED_TERMINATORS,
        ),
    }
    for key, (found, expected) in declared_levels.items():
        if found != expected:
            errors.append(f"{key} must be {list(expected)}, found {found}")

    for attribute, key in (
        ("PUBLIC_LEVEL", "public_level"),
        ("INTERNAL_LEVELS", "internal_levels"),
        ("REALIZED_EXITS", "realized_exits"),
        ("SUPPORT_STATES", "support_states"),
        ("VALUE_KINDS", "value_kinds"),
        ("LINEAR_VALUE_KINDS", "linear_value_kinds"),
        ("TERMINATORS", "terminators"),
    ):
        shipped = getattr(boundary, attribute, ())
        shipped = (shipped,) if isinstance(shipped, str) else tuple(shipped)
        declared = contract.get(key)
        declared_tuple = (
            (declared,) if isinstance(declared, str) else tuple(declared or ())
        )
        if shipped != declared_tuple:
            errors.append(
                f"the declared {key} differs from the boundary's {attribute}: "
                f"contract {declared_tuple}, boundary {shipped}"
            )

    declared_codes = contract.get("diagnostic_codes")
    if not isinstance(declared_codes, list) or not declared_codes:
        errors.append("diagnostic_codes must be a non-empty list")
    else:
        codes: list[str] = []
        for entry in declared_codes:
            if not isinstance(entry, dict) or set(entry) != {
                "code",
                "level",
                "meaning",
            }:
                errors.append(
                    f"each diagnostic entry needs code, level, and meaning: {entry!r}"
                )
                continue
            codes.append(str(entry.get("code")))
            if not str(entry.get("meaning") or "").strip():
                errors.append(f"diagnostic {entry.get('code')!r} states no meaning")
        if codes != sorted(codes):
            errors.append("diagnostic_codes must be listed in sorted order")
        if len(codes) != len(set(codes)):
            errors.append("diagnostic_codes must not repeat a code")
        registry = dict(getattr(boundary, "DIAGNOSTIC_LEVELS", {}))
        if sorted(codes) != sorted(registry):
            errors.append(
                "the declared diagnostic vocabulary differs from the boundary's: "
                f"contract {sorted(set(codes) - set(registry))} extra, "
                f"boundary {sorted(set(registry) - set(codes))} unlisted"
            )
        for entry in declared_codes:
            code = entry.get("code")
            if code in registry and entry.get("level") != registry[code]:
                errors.append(
                    f"diagnostic {code!r} declares level {entry.get('level')!r}, "
                    f"but the boundary raises it at {registry[code]!r}"
                )
            if (
                entry.get("level")
                not in (EXPECTED_PUBLIC_LEVEL,) + EXPECTED_INTERNAL_LEVELS
            ):
                errors.append(
                    f"diagnostic {code!r} names the undeclared level "
                    f"{entry.get('level')!r}"
                )

    exclusions = contract.get("exclusions")
    if not isinstance(exclusions, list) or not exclusions:
        errors.append("exclusions must be a non-empty list")
    elif not all(isinstance(item, str) and item.strip() for item in exclusions):
        errors.append("every exclusion must be a non-empty string")

    declared_rules = contract.get("rules")
    if not isinstance(declared_rules, dict) or not declared_rules:
        errors.append("the internal level contract must declare its rules")
    else:
        unread = sorted(set(declared_rules) - set(rules))
        if unread:
            errors.append(f"rules with no reader in this gate: {unread}")
        for name, readers in rules.items():
            if name not in declared_rules:
                errors.append(f"reader {name!r} names a rule the contract dropped")
                continue
            if not str(declared_rules[name]).strip():
                errors.append(f"rule {name!r} states nothing")
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
        print(f"the internal level contract is not valid JSON: {exc}")
        return 1
    if not isinstance(contract, dict):
        print("the internal level contract must be a JSON object")
        return 1

    sources = refusal_sources()
    codes: set[str] = set()
    dynamic: list[str] = []
    for source in sources:
        found, computed = _raised_codes(source)
        codes |= found
        dynamic.extend(f"{source.relative_to(ROOT)}:{entry}" for entry in computed)
    errors = list(contract_errors(contract))
    if BOUNDARY not in sources:
        errors.append(
            f"the refusal scan does not cover {BOUNDARY.relative_to(ROOT)}, so the "
            "gate would pass on a boundary it never read"
        )
    if dynamic:
        errors.append(
            "the internal boundary raises a computed diagnostic code at "
            f"{', '.join(dynamic)}; every refusal must name a declared literal code"
        )
    registry = dict(
        getattr(importlib.import_module(BOUNDARY_MODULE), "DIAGNOSTIC_LEVELS", {})
    )
    for code in sorted(codes):
        if code not in registry:
            errors.append(f"the boundary refuses with the undeclared code {code!r}")
    for code in sorted(registry):
        if code not in codes:
            errors.append(f"the declared code {code!r} is never raised by the boundary")
    errors.extend(realized_exits_are_declared(contract))
    errors.extend(forbidden_states_are_unreachable(contract))
    errors.extend(no_second_boundary(contract))
    errors.extend(no_root_export(contract))
    errors.extend(committed_payloads_round_trip())
    if IR_VERSION != "1.0":
        errors.append(f"the internal levels assume IR version 1.0, found {IR_VERSION}")

    if errors:
        print("\n".join(sorted(set(errors))))
        return 1
    print(
        "internal multi-level IR contract passed: "
        f"{len(contract['internal_levels'])} internal levels, "
        f"{len(contract['diagnostic_codes'])} diagnostic codes each raised, "
        f"{len(contract['rules'])} rules each with a reader, "
        f"{len(codes)} literal refusal sites"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
