"""Authoritative inverse rules for checked register-unitary effects."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace

from .model import Operation

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


def _inverse_specs(
    operations: Sequence[Operation],
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
                raise ValueError("inverse requires a finite real rotation angle")
            attributes["angle"] = -angle
        else:
            raise ValueError(
                "invert cannot allocate or change ownership and requires checked reversible gates"
            )
        inverses.append((opcode, op, attributes))
    return inverses


def inverse_effects(operations: Sequence[Operation]) -> tuple[Operation, ...]:
    return tuple(
        replace(op, name=opcode, attributes=attributes)
        for opcode, op, attributes in _inverse_specs(operations)
    )
