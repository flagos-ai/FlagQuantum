# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 CtrlSpecAnd.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mcmt/ctrl_spec_and.py

"""Concrete register equality clauses and their shared And predicate."""

from dataclasses import dataclass
from typing import Literal

from ....core.ir import Instruction
from ..bits.bitwise import _validate_wires
from .multi_control import multi_and_ir

IntegerEncoding = Literal["unsigned", "twos_complement", "ones_complement"]


@dataclass(frozen=True)
class ControlRegister:
    """One row-major group of fixed-width integer activation values.

    Unsigned includes raw bit, bounded integer, and Montgomery words. Supply
    Montgomery values in their stored representation.
    """

    width: int
    values: tuple[int, ...]
    encoding: IntegerEncoding = "unsigned"

    def bits(self) -> tuple[int, ...]:
        """Encode each integer as a big-endian bit word."""
        if type(self.width) is not int or self.width < 1:
            raise ValueError("Control word width must be a positive integer")
        modulus = 1 << self.width
        if self.encoding == "unsigned":
            low, high = 0, modulus
        elif self.encoding == "twos_complement":
            low, high = -(modulus // 2), modulus // 2
        elif self.encoding == "ones_complement" and self.width > 1:
            low, high = 1 - modulus // 2, modulus // 2
        else:
            raise ValueError("Control word encoding is invalid for its width")
        if any(
            type(value) is not int or not low <= value < high for value in self.values
        ):
            raise ValueError("Control value is outside the integer encoding range")
        words = tuple(
            (
                modulus - 1 + value
                if self.encoding == "ones_complement" and value < 0
                else value % modulus
            )
            for value in self.values
        )
        return tuple(
            (word >> bit) & 1 for word in words for bit in reversed(range(self.width))
        )


def bind_control_registers(
    controls: tuple[tuple[int, ...], ...], registers: tuple[ControlRegister, ...]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Bind flattened control registers in source port order, without gates."""
    if len(controls) != len(registers):
        raise ValueError("Control groups must match the register specification")
    patterns = tuple(register.bits() for register in registers)
    if any(
        len(group) != len(bits) for group, bits in zip(controls, patterns, strict=True)
    ):
        raise ValueError("Control group width must match its encoded values")
    wires = tuple(wire for group in controls for wire in group)
    if wires:
        _validate_wires(wires)
    return wires, tuple(bit for pattern in patterns for bit in pattern)


def control_predicate_ir(
    controls: tuple[tuple[int, ...], ...],
    registers: tuple[ControlRegister, ...],
    target: int,
    junk: tuple[int, ...],
    *,
    classical_bits: tuple[int, ...] = (),
    uncompute: bool = False,
) -> tuple[Instruction, ...]:
    """Compute a visible predicate and prefix junk, or erase both.

    The forward output target and junk start clean. Erasure consumes their
    computed values, restores them to zero, and preserves all control words.
    """
    wires, bits = bind_control_registers(controls, registers)
    return multi_and_ir(
        wires, bits, (*junk, target), classical_bits=classical_bits, uncompute=uncompute
    )
