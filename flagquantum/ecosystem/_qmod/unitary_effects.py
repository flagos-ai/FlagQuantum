"""Inverse composition of existing pure unitary effects at the source boundary."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

from ...compiler._hybrid.model import Operation, Value
from ...compiler._hybrid.unitary_effects import (
    PURE_EFFECTS as PURE_EFFECTS,
)
from ...compiler._hybrid.unitary_effects import (
    inverse_effects as _compiler_inverse_effects,
)
from .syntax import Statement, Token, reject

if TYPE_CHECKING:
    from .context import _Adapt


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


def inverse_effects(
    token: Token, operations: Sequence[Operation]
) -> tuple[Operation, ...]:
    try:
        return _compiler_inverse_effects(operations)
    except ValueError as error:
        reject(token, "scope", str(error))


def emit_inverse(owner: _Adapt, token: Token, operations: Sequence[Operation]) -> None:
    for op in inverse_effects(token, operations):
        owner.effect_op(op.name, token, op.operands[1:], **op.attributes)


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
