"""Inverse composition of existing pure unitary effects at the source boundary."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

from ...compiler._hybrid.model import Operation
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
    }
)


def emit_inverse(owner: _Adapt, token: Token, operations: Sequence[Operation]) -> None:
    inverses: list[tuple[str, Operation, dict[str, object]]] = []
    for op in reversed(operations):
        attributes: dict[str, object] = dict(op.attributes)
        opcode = op.name
        if op.name in {"qreg.qft", "qreg.inverse_qft"}:
            opcode = "qreg.inverse_qft" if op.name == "qreg.qft" else "qreg.qft"
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
    for opcode, op, attributes in inverses:
        owner.effect_op(opcode, token, op.operands[1:], **attributes)


def is_reversible_scope(statement: Statement) -> bool:
    return bool(statement.within) and all(
        child.token.text not in {"=", "->"} for child in statement.within or ()
    )


def lower_reversible_scope(owner: _Adapt, statement: Statement, index: str) -> None:
    assert statement.within is not None
    bindings = dict(owner.registers), dict(owner.views), set(owner.suspended)

    def checked_effects(
        label: str, body: tuple[Statement, ...]
    ) -> tuple[Operation, ...]:
        start = len(owner.operations)
        for ordinal, child in enumerate(body):
            owner.statement(child, f"{index}.{label}{ordinal}")
        effects = tuple(owner.operations[start:])
        if any(op.name not in PURE_EFFECTS for op in effects) or bindings != (
            owner.registers,
            owner.views,
            owner.suspended,
        ):
            reject(
                statement.token,
                "scope",
                "reversible within/apply requires pure effects on existing storage",
            )
        return effects

    forward = checked_effects("within", statement.within)
    checked_effects("apply", statement.apply)
    emit_inverse(owner, statement.token, forward)
