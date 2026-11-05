"""Inverse composition of existing pure unitary effects at the source boundary."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

from ...compiler._hybrid.model import Operation
from .syntax import Token, reject

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
    }
)


def emit_inverse(owner: _Adapt, token: Token, operations: Sequence[Operation]) -> None:
    inverses: list[tuple[Operation, dict[str, object]]] = []
    for op in reversed(operations):
        attributes: dict[str, object] = dict(op.attributes)
        if op.name == "qreg.gate":
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
        inverses.append((op, attributes))
    for op, attributes in inverses:
        owner.effect_op(op.name, token, op.operands[1:], **attributes)
