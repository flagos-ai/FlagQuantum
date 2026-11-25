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
        "qreg.xor",
        "qreg.array_phase_flip",
        "qreg.phase",
        "qreg.control_phase",
        "qreg.pauli_rotation",
        "qreg.qft",
        "qreg.inverse_qft",
        "qreg.control_qft",
        "qreg.inplace_prepare_amplitudes",
        "qreg.unitary",
        "qreg.unitary_call",
    }
)


def unitary_regions(
    operations: Sequence[Operation],
) -> tuple[tuple[Operation, ...], ...]:
    """Partition pure effects and closed directly computed scratch regions.

    Register verification separately proves that the action preserves the
    predicate and its inputs, so uncompute returns the local to zero.
    """

    def region_end(start: int, depth: int) -> int:
        if depth > 32:
            raise ValueError("unitary scratch nesting exceeds 32")
        op = operations[start]
        if op.name in PURE_EFFECTS:
            return start + 1
        if (
            op.name != "qreg.local"
            or len(op.results) != 1
            or start + 4 >= len(operations)
        ):
            raise ValueError(
                "unitary region requires pure effects or closed predicate scratch"
            )
        local = op.results[0]
        compute = operations[start + 1]
        if compute.name != "qreg.compute" or compute.operands[1:2] != (local,):
            raise ValueError("predicate scratch requires an immediate direct compute")
        end = start + 2
        while end < len(operations) and operations[end].name != "qreg.uncompute":
            end = region_end(end, depth + 1)
        if (
            end == start + 2
            or end + 1 >= len(operations)
            or operations[end].operands[1:] != (local,)
            or operations[end + 1].name != "qreg.release"
            or operations[end + 1].operands[1:] != (local,)
            or any(local in effect.operands for effect in operations[end + 2 :])
        ):
            raise ValueError(
                "predicate scratch must uncompute and release without escape"
            )
        return end + 2

    regions: list[tuple[Operation, ...]] = []
    index = 0
    while index < len(operations):
        end = region_end(index, 0)
        regions.append(tuple(operations[index:end]))
        index = end
    return tuple(regions)


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
            "qreg.unitary_call",
        }:
            if attributes.pop("inverse", False) is not True:
                attributes["inverse"] = True
        elif op.name in {"qreg.array_phase_flip", "qreg.xor"} or (
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
    inverted: list[Operation] = []
    for region in reversed(unitary_regions(operations)):
        if len(region) == 1:
            inverted.extend(
                replace(op, name=opcode, attributes=attributes)
                for opcode, op, attributes in _inverse_specs(region)
            )
        else:
            inverted.extend(region[:2])
            inverted.extend(inverse_effects(region[2:-2]))
            inverted.extend(region[-2:])
    return tuple(inverted)
