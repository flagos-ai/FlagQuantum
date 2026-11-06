# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# X-basis clearing moved from the paired QROAM emitter adapted from Qualtran.
# Source: https://github.com/quantumlib/Qualtran/tree/v0.7.0/qualtran/bloqs/data_loading
# Basis instruments: https://github.com/quantumlib/Qualtran/tree/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates
# Changes: caller-owned physical clearing and explicit parity-controlled phase expansion.

"""Shared basis measurement clearing and classical-parity phase decompositions."""

from dataclasses import dataclass

from ...core.classical_dataflow import ClassicalXor
from ...core.ir import Instruction


def measure_z_clear_ir(qubit: int, classical_bit: int) -> tuple[Instruction, ...]:
    """Record the Z outcome and return the consumed qubit's physical slot in zero."""
    if any(type(value) is not int or value < 0 for value in (qubit, classical_bit)):
        raise ValueError(
            "Measurement qubit and classical bit must be nonnegative integers"
        )
    return (
        Instruction(
            "measure",
            (qubit,),
            metadata={"is_dynamic": True, "classical_bit": classical_bit},
        ),
        Instruction("x", (qubit,), metadata={"conditions": ((classical_bit, 1),)}),
    )


def measure_x_clear_ir(qubit: int, classical_bit: int) -> tuple[Instruction, ...]:
    """Record the X outcome and return the consumed qubit's physical slot in zero."""
    clear = measure_z_clear_ir(qubit, classical_bit)
    return (Instruction("h", (qubit,)), *clear)


@dataclass(frozen=True)
class ParityPhase:
    """A Z-family phase controlled by XOR of measured classical bits.

    Multiple inputs compute one shared classical XOR result and apply one phase.
    The caller owns its scratch bit. Singleton feedback needs no computation.
    """

    qubits: tuple[int, ...]
    bits: tuple[int, ...]

    def __post_init__(self) -> None:
        if not 1 <= len(self.qubits) <= 2 or len(set(self.qubits)) != len(self.qubits):
            raise ValueError("Parity phase requires one or two distinct quantum wires")
        if len(set(self.bits)) != len(self.bits) or any(
            type(value) is not int or value < 0 for value in (*self.qubits, *self.bits)
        ):
            raise ValueError(
                "Parity phase operands require distinct nonnegative integer bits"
            )

    def lower(self, *, parity_bit: int | None = None) -> tuple[Instruction, ...]:
        """Compute parity in a classical slot, preserving one quantum phase gate."""
        if not self.bits:
            return ()
        name = ("z", "cz")[len(self.qubits) - 1]
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
            Instruction(name, self.qubits, metadata={"conditions": ((bit, 1),)}),
        )
