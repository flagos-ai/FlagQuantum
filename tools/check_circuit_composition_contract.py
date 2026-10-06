#!/usr/bin/env python3
"""Validate the construction-time composition contract against the implementation.

`contracts/circuit-composition-contract.toml` is the repository's own statement of what
`Circuit.compose`, `Circuit.adjoint`, and `Circuit.control` guarantee and of every way
they refuse. This gate reads that statement back against the code: a contracted refusal
whose phrase no longer occurs in its source, a refusal the contract does not list, a
declared adjoint or control rule the operator schema does not define, an operation the
contract calls absent but that someone added, an opcode census that no longer supports a
refusal, a control claim the expansion contradicts, or a composition operation without an
expand-and-compare test all fail here.

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

import torch

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 and older
    import tomli as tomllib

import flagquantum as fq
from flagquantum.core import (
    ADJOINT_RULES,
    CONTROL_RULES,
    MAX_LADDER_LEVEL,
    OPERATOR_SCHEMAS,
    control_ladder_level,
    get_operator_schema,
)
from flagquantum.core.controlled import controlled_instructions
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

#: The opcodes whose controlled form is the ladder alone, with no basis change and no
#: correction gate around it. `x` is ``h . C(P(pi)) . h`` and `rz` carries a phase
#: correction, so neither measures the bound itself; these do.
LADDER_BENCHMARK_OPCODES = ("z", "t", "phase", "s", "sdg", "cz", "cphase")

#: How deep the gate measures the ladder bound. The bound is a formula, so a few rungs
#: decide whether the formula is the one the expansion follows; the ceiling itself is
#: checked against `MAX_LADDER_LEVEL` rather than by building the deepest program.
LADDER_PROBE_LEVELS = (1, 2, 3, 4, 5, 6)


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
        "control_rule_source",
        "control_expansion_implementation",
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
    if provided != ["Circuit.compose", "Circuit.adjoint", "Circuit.control"]:
        errors.append("circuit composition provided surface drifted")
    if not_provided != ["Circuit.power"]:
        errors.append("circuit composition absent surface drifted")
    for dotted in provided:
        owner, _, attribute = dotted.partition(".")
        if attribute == "" or not hasattr(getattr(fq, owner, None), attribute):
            errors.append(f"contracted operation {dotted!r} does not exist")
    for dotted in not_provided:
        owner, _, attribute = dotted.partition(".")
        if attribute and hasattr(getattr(fq, owner, None), attribute):
            errors.append(f"contracted-absent operation {dotted!r} exists")
    # The family is construction-time methods on `Circuit`. `power` is absent and
    # `control` is provided, but neither may become an operation of its own at the root:
    # a module-level `fq.control` would be a second way to say the same thing.
    for entry in ("control", "power"):
        if entry in fq.__all__:
            errors.append(f"{entry!r} is a root export; the family is Circuit methods")

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
    errors.extend(_control_errors(contract))
    errors.extend(_vocabulary_errors(contract))
    errors.extend(_census_errors(contract))
    errors.extend(_verification_errors(contract))
    return tuple(errors)


def _control_errors(contract: dict[str, Any]) -> list[str]:
    """Check the `[control]` table against the rules and the expansion themselves.

    The table makes two kinds of claim. The first kind is a name: which rules exist, where
    the ceiling lives, which table the one-control shortcut is read from. Those are read
    back against `flagquantum/core/operator_schema.py`. The second kind is a behaviour --
    the width, the ancilla count, the emitted opcodes, the ladder bound -- and those are
    measured by expanding a real receiver, because a behavior checked by reading the
    table that states it is not checked at all.
    """

    errors: list[str] = []
    control = contract.get("control", {})
    authorization = control.get("authorization")
    if authorization not in list(contract.get("authorization", ())):
        errors.append(
            f"circuit control authorization {authorization!r} is not among this file's "
            "authorizations"
        )
    if list(control.get("declared_opcode_rules", ())) != list(CONTROL_RULES):
        errors.append(
            "circuit control declared rules drifted from CONTROL_RULES: "
            f"contract={control.get('declared_opcode_rules')}, "
            f"implementation={list(CONTROL_RULES)}"
        )
    if control.get("absence_rule") != "not_available":
        errors.append("circuit control absence rule must be not_available")
    missing = [
        name
        for name in ("single_control_partner_source", "max_ladder_level_source")
        if control.get(name) != "flagquantum/core/operator_schema.py"
    ]
    if missing:
        errors.append(f"circuit control {missing} must name the operator schema")
    if control.get("max_ladder_level") != MAX_LADDER_LEVEL:
        errors.append(
            "circuit control ladder ceiling drifted: "
            f"contract={control.get('max_ladder_level')}, "
            f"implementation={MAX_LADDER_LEVEL}"
        )
    if control.get("ancilla_qubits") != 0:
        errors.append("circuit control must contract zero ancillas")

    # The signature is the contract a user reads, so it is read off the method rather than
    # restated: the count precedes the control qubits because the count is what the control
    # qubits have to agree with, and neither may take a default -- a defaulted control
    # qubit set would make the count the only required argument while the qubits it counts
    # are implied.
    signature = inspect.signature(fq.Circuit.control)
    names = list(signature.parameters)
    if names != [
        "self",
        control.get("control_count_argument"),
        control.get("control_qubits_argument"),
    ]:
        errors.append(f"Circuit.control signature drifted: {names}")
    else:
        for name in names[1:]:
            parameter = signature.parameters[name]
            if parameter.kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD:
                errors.append(f"Circuit.control {name} must be positional or keyword")
            if parameter.default is not inspect.Parameter.empty:
                errors.append(f"Circuit.control {name} must have no default")

    # The shape claims. Each is either a formula the gate evaluates itself or a string the
    # gate reads back, so editing the contract's prose without editing the measurement --
    # or the other way round -- fails here.
    formulas = {
        "ladder_control_count": "n_controls",
        "ladder_level": "n_controls + arity - 1",
        "ladder_instruction_bound": "4 * 3 ** (ladder_level - 1) - 3",
        "ladder_depth_growth": "exponential",
        "result_width": "max(ctrl_qubits) + 1",
        "returns": "new_circuit",
        "instruction_order": "forward",
    }
    for name, expected in formulas.items():
        if control.get(name) != expected:
            errors.append(f"circuit control {name} must be {expected!r}")

    flags = {
        "receiver_is_mutated": False,
        "control_qubits_are_added": True,
        "control_qubits_must_be_distinct": True,
        "control_qubits_must_be_outside_the_receiver": True,
        "emitted_opcodes_are_registered": True,
        "zero_angles_are_emitted": True,
    }
    for name, expected in flags.items():
        if control.get(name) is not expected:
            errors.append(f"circuit control {name} must be contracted as {expected}")

    # The rules must also be the rules the opcodes actually use, or the list is a
    # vocabulary nobody speaks.
    used = {schema.control for schema in OPERATOR_SCHEMAS.values()}
    unspoken = sorted(set(CONTROL_RULES) - used)
    if unspoken:
        errors.append(
            f"circuit control rules declared but used by no opcode: {unspoken}"
        )

    receiver = fq.Circuit(2).h(0).cnot(0, 1)
    receiver_before = [
        (item.name, item.wires) for item in receiver.to_ir().instructions
    ]
    controlled = receiver.control(1, ctrl_qubits=(4,))
    receiver_after = [(item.name, item.wires) for item in receiver.to_ir().instructions]
    if receiver_after != receiver_before:
        errors.append("Circuit.control mutated its receiver")
    # A control qubit is added, so the width is one past the greatest control qubit named
    # and is therefore always greater than the receiver's own width.
    if controlled.n_qubits != 5:
        errors.append(
            "Circuit.control result width is not one past the greatest control qubit, "
            f"got {controlled.n_qubits}"
        )
    preserved = {
        "bsz": (controlled.bsz, receiver.bsz),
        "device": (controlled.device, receiver.device),
        "dtype": (controlled.dtype, receiver.dtype),
    }
    drifted = sorted(name for name, (got, want) in preserved.items() if got != want)
    if drifted:
        errors.append(
            f"Circuit.control did not preserve contracted attributes: {drifted}"
        )
    if list(control.get("preserves", ())) != ["bsz", "device", "dtype"]:
        errors.append(
            "circuit control preserves must be exactly bsz, device and dtype; n_qubits "
            "cannot be preserved because a control qubit is added"
        )

    # `receiver_input_state` is a claim about amplitudes, so it is measured on a receiver
    # that carries an input state and no instructions. Qubit 0 is the most significant
    # amplitude bit, so the added control qubit is the least significant one and the
    # receiver's own amplitudes land at index 0 of each added-qubit pair with `|0>` in the
    # other slot. Dropping the receiver's input instead of carrying it would leave a
    # program that runs and computes something else, which is why this is contracted.
    if control.get("receiver_input_state") != (
        "carried into the wider register with every added qubit at |0>"
    ):
        errors.append("circuit control receiver input state claim drifted")
    state = torch.tensor([[0.6, 0.8]], dtype=torch.complex128)
    carrying = fq.Circuit(1, dtype=torch.complex128, inputs=state)
    embedded = carrying.control(1, ctrl_qubits=(1,)).state()
    expected = torch.tensor([[0.6 + 0j, 0j, 0.8 + 0j, 0j]], dtype=torch.complex128)
    if not torch.allclose(embedded, expected, atol=1e-14):
        errors.append(
            "Circuit.control did not carry the receiver's input state into the wider "
            f"register: {embedded.tolist()}"
        )

    # The emitted opcode census, measured over every unitary opcode the registry declares
    # rather than over the receiver above: a rule that reached for an opcode nobody
    # registered would put a name into the IR that the IR cannot execute.
    emitted: set[str] = set()
    for opcode, schema in OPERATOR_SCHEMAS.items():
        if schema.opcode != opcode or schema.control == "not_available":
            continue
        for level in (1, 2):
            qubits = tuple(range(schema.arity))
            controls = tuple(range(schema.arity, schema.arity + level))
            expansion = controlled_instructions(
                schema, dict.fromkeys(schema.parameters, 0.37), qubits, controls
            )
            emitted |= {item.name for item in (expansion or ())}
    unknown = sorted(name for name in emitted if get_operator_schema(name) is None)
    if unknown:
        errors.append(f"Circuit.control emitted unregistered opcodes: {unknown}")

    # The ladder level is a claim about the implementation, so it is read over a grid
    # rather than at one point: a formula that agrees at level 1 and drifts with arity is
    # the failure this catches.
    for arity in (1, 2, 3):
        for count in range(1, 4):
            got = control_ladder_level(arity, count)
            if got != count + arity - 1:
                errors.append(
                    f"circuit control ladder level for arity {arity} and {count} "
                    f"control(s) is {got}, not the contracted {count + arity - 1}"
                )

    # The ladder bound, measured on opcodes whose controlled form is the ladder alone. The
    # expected count is the contract's own formula, evaluated here rather than read back
    # from the code that emits it.
    for opcode in LADDER_BENCHMARK_OPCODES:
        schema = get_operator_schema(opcode)
        if schema is None:  # pragma: no cover - the contract's own vocabulary
            errors.append(
                f"circuit control benchmark opcode {opcode!r} is not registered"
            )
            continue
        if schema.arity != 1 and opcode in {"z", "t", "phase", "s", "sdg"}:
            errors.append(f"circuit control benchmark opcode {opcode!r} changed arity")
        params = dict.fromkeys(schema.parameters, 0.37)
        for count in LADDER_PROBE_LEVELS:
            qubits = tuple(range(schema.arity))
            controls = tuple(range(schema.arity, schema.arity + count))
            expansion = controlled_instructions(schema, params, qubits, controls)
            level = count + schema.arity - 1
            expected = 4 * 3 ** (level - 1) - 3
            if expansion is None or len(expansion) != expected:
                errors.append(
                    f"circuit control ladder for {opcode!r} under {count} control(s) "
                    f"emitted {None if expansion is None else len(expansion)} "
                    f"instructions, not the contracted {expected}"
                )
            reached = {qubit for item in (expansion or ()) for qubit in item.wires}
            outside = sorted(reached - set(qubits) - set(controls))
            if outside:
                errors.append(
                    f"circuit control ladder for {opcode!r} under {count} control(s) "
                    f"named qubits outside the control set: {outside}"
                )
    return errors


def _refusal_errors(contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    refusals = contract.get("refusals", ())
    codes = [row.get("code") for row in refusals]
    duplicates = sorted({code for code in codes if codes.count(code) > 1})
    if duplicates:
        errors.append(f"circuit composition refusal codes repeat: {duplicates}")
    entry_points = {row.get("entry_point") for row in refusals}
    if not entry_points <= {"compose", "adjoint", "control"}:
        errors.append(
            f"circuit composition refusals name unknown entry points: {entry_points}"
        )
    if len(refusals) != 30:
        errors.append(
            f"circuit composition must contract 30 refusals, found {len(refusals)}"
        )
    for row in refusals:
        code = row.get("code")
        if not isinstance(code, str) or not code:
            errors.append(f"circuit composition refusal {row!r} has no code")
            continue
        if row.get("entry_point") not in {"compose", "adjoint", "control"}:
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

    # The same census read through the control rules. It is the measurement behind the
    # control row that reports "no controlled form": every opcode that declares no rule is
    # a noise channel, and a program holding one without its `is_channel` metadata is the
    # route that reaches the row.
    controlled = {
        opcode for opcode, schema in OPERATOR_SCHEMAS.items() if schema.opcode == opcode
    }
    undeclared = sorted(
        opcode
        for opcode in controlled
        if OPERATOR_SCHEMAS[opcode].control == "not_available"
    )
    declared = sorted(controlled - set(undeclared))
    control_census = contract.get("verification", {}).get("control_census", "")
    control_refusals = sorted(
        opcode
        for opcode in undeclared
        if OPERATOR_SCHEMAS[opcode].semantic_kind == "channel"
    )
    if control_refusals != undeclared:
        errors.append(
            "circuit control opcodes without a controlled form are not all channels: "
            f"{sorted(set(undeclared) - set(control_refusals))}"
        )
    if (
        f"{len(declared)} of {len(OPERATOR_SCHEMAS)} registered opcodes declare a "
        "controlled form" not in control_census
    ):
        errors.append(
            "circuit control census drifted: "
            f"{len(declared)} of {len(OPERATOR_SCHEMAS)} registered opcodes declare a "
            "controlled form"
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
