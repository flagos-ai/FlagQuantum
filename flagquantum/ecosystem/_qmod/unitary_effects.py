"""Inverse composition of existing pure unitary effects at the source boundary."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

from ...compiler._hybrid.model import Operation, Value
from .syntax import Statement, Token, reject

if TYPE_CHECKING:
    from .adapter import _Adapt

PURE_EFFECTS = frozenset(
    {
        "qreg.gate",
        "qreg.pair_gate",
        "qreg.control_x",
        "qreg.array_phase_flip",
        "qreg.phase",
        "qreg.control_phase",
        "qreg.pauli_rotation",
        "qreg.qft",
        "qreg.inverse_qft",
        "qreg.control_qft",
        "qreg.inplace_prepare_amplitudes",
        "qreg.unitary",
    }
)


def preserves_basis_labels(op: Operation, readonly: set[tuple[Value, int]]) -> bool:
    """Check borrowed locations even after nested helper arguments are remapped."""
    operands = op.operands[1:]
    if op.name == "qreg.control_phase":
        return True
    if op.name == "qreg.gate":
        return (
            op.attributes.get("gate")
            in {"i", "z", "s", "sdg", "t", "tdg", "rz", "phase"}
            or (operands[0], op.attributes["index"]) not in readonly
        )
    if op.name == "qreg.control_x" and "indices" in op.attributes:
        return (operands[-1], op.attributes["indices"][-1]) not in readonly
    if op.name == "qreg.inplace_prepare_amplitudes" and "indices" in op.attributes:
        count = op.attributes.get("control_count", 0)
        return not readonly.intersection(
            zip(operands[count:], op.attributes["indices"][count:], strict=True)
        )
    if op.name == "qreg.pauli_rotation":
        return all(
            axis == "Z" or (operands[0], wire) not in readonly
            for wire, axis in op.attributes["support"]
        )
    if op.name == "qreg.pair_gate":
        gate = op.attributes.get("gate")
        if gate in {"cz", "rzz"}:
            return True
        if gate in {"cx", "crx", "cry"}:
            return (
                operands[-1],
                op.attributes.get("indices", (0, 0))[-1],
            ) not in readonly
    return False


def emit_controlled_pauli(
    owner: _Adapt,
    token: Token,
    target: Value,
    support: tuple[tuple[int, str], ...],
    angle: float,
    controls: tuple[tuple[Value, int], ...],
) -> None:
    """Apply a Pauli exponential using physically disjoint indexed controls."""
    positions = tuple((target, index) for index, _ in support)
    if len(set((*controls, *positions))) != len(controls) + len(positions):
        reject(token, "alias", "Pauli controls and support must be disjoint")

    def phase(locations: tuple[tuple[Value, int], ...], angle: float) -> None:
        owner.effect_op(
            "qreg.control_phase",
            token,
            tuple(value for value, _ in locations),
            indices=tuple(wire for _, wire in locations),
            angle=angle,
        )

    for index, axis in support:
        if axis == "X":
            owner.effect_op("qreg.gate", token, (target,), gate="h", index=index)
        elif axis == "Y":
            owner.effect_op(
                "qreg.gate", token, (target,), gate="rx", index=index, angle=math.pi / 2
            )
    if positions:
        for source in positions[1:]:
            owner.emit_indexed_x(token, (source, positions[0]))
    phase(controls, -angle)
    if positions:
        phase((*controls, positions[0]), 2 * angle)
        for source in reversed(positions[1:]):
            owner.emit_indexed_x(token, (source, positions[0]))
    for index, axis in reversed(support):
        if axis == "X":
            owner.effect_op("qreg.gate", token, (target,), gate="h", index=index)
        elif axis == "Y":
            owner.effect_op(
                "qreg.gate",
                token,
                (target,),
                gate="rx",
                index=index,
                angle=-math.pi / 2,
            )


def _inverse_specs(
    token: Token, operations: Sequence[Operation]
) -> list[tuple[str, Operation, dict[str, object]]]:
    inverses: list[tuple[str, Operation, dict[str, object]]] = []
    for op in reversed(operations):
        attributes: dict[str, object] = dict(op.attributes)
        opcode = op.name
        if op.name in {"qreg.qft", "qreg.inverse_qft"}:
            opcode = "qreg.inverse_qft" if op.name == "qreg.qft" else "qreg.qft"
        elif op.name == "qreg.unitary":
            attributes["matrix"] = tuple(
                tuple((real, -imag) for real, imag in column)
                for column in zip(*op.attributes["matrix"], strict=True)
            )
        elif op.name in {
            "qreg.gate",
            "qreg.control_qft",
            "qreg.inplace_prepare_amplitudes",
        }:
            if attributes.pop("inverse", False) is not True:
                attributes["inverse"] = True
        elif op.name == "qreg.array_phase_flip" or (
            op.name == "qreg.control_x"
            and "indices" in attributes
            or op.name == "qreg.pair_gate"
            and attributes.get("gate")
            in {
                "cx",
                "cz",
                "swap",
            }
        ):
            pass
        elif op.name in {"qreg.phase", "qreg.control_phase", "qreg.pauli_rotation"} or (
            op.name == "qreg.pair_gate"
            and attributes.get("gate") in {"crx", "cry", "rzz"}
        ):
            angle = op.attributes.get("angle", math.pi)
            if not isinstance(angle, (float, int)) or isinstance(angle, bool):
                reject(token, "scope", "inverse requires a finite real rotation angle")
            attributes["angle"] = -angle
        else:
            reject(
                token,
                "scope",
                "invert cannot allocate or change ownership and requires checked reversible gates",
            )
        inverses.append((opcode, op, attributes))
    return inverses


def inverse_effects(
    token: Token, operations: Sequence[Operation]
) -> tuple[Operation, ...]:
    return tuple(
        replace(op, name=opcode, attributes=attributes)
        for opcode, op, attributes in _inverse_specs(token, operations)
    )


def emit_inverse(owner: _Adapt, token: Token, operations: Sequence[Operation]) -> None:
    for opcode, op, attributes in _inverse_specs(token, operations):
        owner.effect_op(opcode, token, op.operands[1:], **attributes)


def is_reversible_scope(statement: Statement) -> bool:
    return bool(statement.within) and all(
        child.token.text not in {"=", "->"} for child in statement.within or ()
    )


def lower_reversible_scope(owner: _Adapt, statement: Statement, index: str) -> None:
    assert statement.within is not None
    registers = dict(owner.registers)
    bindings = dict(owner.views), set(owner.suspended)

    def checked_effects(
        label: str,
        body: tuple[Statement, ...],
        preparation: tuple[Operation, ...] = (),
    ) -> tuple[Operation, ...]:
        start = len(owner.operations)
        for ordinal, child in enumerate(body):
            owner.statement(child, f"{index}.{label}{ordinal}")
        effects = tuple(owner.operations[start:])
        touched = {value for op in preparation for value in op.operands[1:]}
        allocated = {op.operands[1] for op in effects if op.name == "qreg.allocate"}
        fresh_outputs = {
            name: value
            for name, value in owner.registers.items()
            if label == "apply"
            and owner.parameters[name].role == "output"
            and value in allocated
            and value not in touched
        }
        if (
            any(
                op.name not in PURE_EFFECTS
                and not (
                    op.name == "qreg.allocate"
                    and op.operands[1] in fresh_outputs.values()
                )
                for op in effects
            )
            or bindings != (owner.views, owner.suspended)
            or owner.registers != registers | fresh_outputs
        ):
            reject(
                statement.token,
                "scope",
                "reversible within/apply requires pure effects on existing storage",
            )
        return effects

    forward = checked_effects("within", statement.within)
    checked_effects("apply", statement.apply, forward)
    emit_inverse(owner, statement.token, forward)
