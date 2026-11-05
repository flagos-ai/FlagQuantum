# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# X-basis clearing moved from the paired QROAM emitter adapted from Qualtran.
# Source: https://github.com/quantumlib/Qualtran/tree/v0.7.0/qualtran/bloqs/data_loading
# Changes: shared native-IR helpers and explicit parity-controlled phase expansion.

"""Shared X-basis clearing and classical-parity phase decompositions."""

from dataclasses import dataclass

from ...core.classical_dataflow import ClassicalXor
from ...core.ir import Instruction


def measure_x_clear_ir(wire: int, classical_bit: int) -> tuple[Instruction, ...]:
    """Demolish an X-basis measured wire and leave its physical slot in zero."""
    if any(type(value) is not int or value < 0 for value in (wire, classical_bit)):
        raise ValueError(
            "Measurement wire and classical bit must be nonnegative integers"
        )
    return (
        Instruction("h", (wire,)),
        Instruction(
            "measure",
            (wire,),
            metadata={"is_dynamic": True, "classical_bit": classical_bit},
        ),
        Instruction("x", (wire,), metadata={"conditions": ((classical_bit, 1),)}),
    )


@dataclass(frozen=True)
class ParityPhase:
    """A Z-family phase controlled by XOR of measured classical bits.

    Multiple inputs compute one shared classical XOR result and apply one phase.
    The caller owns its scratch bit. Singleton feedback needs no computation.
    """

    wires: tuple[int, ...]
    bits: tuple[int, ...]

    def __post_init__(self) -> None:
        if not 1 <= len(self.wires) <= 2 or len(set(self.wires)) != len(self.wires):
            raise ValueError("Parity phase requires one or two distinct quantum wires")
        if len(set(self.bits)) != len(self.bits) or any(
            type(value) is not int or value < 0 for value in (*self.wires, *self.bits)
        ):
            raise ValueError(
                "Parity phase operands require distinct nonnegative integer bits"
            )

    def lower(self, *, parity_bit: int | None = None) -> tuple[Instruction, ...]:
        """Compute parity in a classical slot, preserving one quantum phase gate."""
        if not self.bits:
            return ()
        name = ("z", "cz")[len(self.wires) - 1]
        prefix: tuple[Instruction, ...] = ()
        bit = self.bits[0]
        if len(self.bits) > 1:
            if parity_bit is None:
                raise ValueError(
                    "Multi-bit parity requires a caller-owned classical result slot"
                )
            prefix = (ClassicalXor(self.bits, parity_bit).instruction(),)
            bit = parity_bit
        return (
            *prefix,
            Instruction(name, self.wires, metadata={"conditions": ((bit, 1),)}),
        )
