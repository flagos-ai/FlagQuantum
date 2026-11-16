# Copyright 2023 Google LLC
# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0
# X-basis clearing moved from the paired QROAM emitter adapted from Qualtran.
# Source: https://github.com/quantumlib/Qualtran/tree/v0.7.0/qualtran/bloqs/data_loading
# Basis instruments: https://github.com/quantumlib/Qualtran/tree/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates
# DiscardQ: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/discard.py
# Changes: caller-owned physical clearing and explicit parity-controlled phase expansion.
# Broadcast order follows the locked Kickmix CircuitBuilder.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/build/circuit_builder.cc

"""Shared basis measurement clearing and classical-parity phase decompositions."""

from dataclasses import dataclass

from ....core.classical_dataflow import ClassicalXor
from ....core.ir import Instruction


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


def measure_x_clear_register_ir(
    qubits: tuple[int, ...], classical_bits: tuple[int, ...]
) -> tuple[Instruction, ...]:
    """Measure and clear paired lanes in order, with caller-owned result slots."""
    if len(qubits) != len(classical_bits):
        raise ValueError("Measurement targets and results must have equal lengths")
    return tuple(
        gate
        for qubit, bit in zip(qubits, classical_bits, strict=True)
        for gate in measure_x_clear_ir(qubit, bit)
    )


def discard_qubit_ir(qubit: int) -> tuple[Instruction, ...]:
    """Consume a logical qubit and return its physical slot in zero without a result bit."""
    if type(qubit) is not int or qubit < 0:
        raise ValueError("Discard qubit must be a nonnegative integer")
    return (Instruction("reset", (qubit,), metadata={"is_dynamic": True}),)


def reset_x_register_ir(
    qubits: tuple[int, ...], *, conditions: tuple[tuple[int, int], ...] = ()
) -> tuple[Instruction, ...]:
    """Discard each qubit in the X basis, in physical lane order."""
    if any(type(qubit) is not int or qubit < 0 for qubit in qubits):
        raise ValueError("Reset qubits must be nonnegative integers")
    metadata = {"conditions": conditions} if conditions else {}
    effect: dict[str, object] = {**metadata, "is_dynamic": True}
    if conditions:
        effect["guarded_effect"] = True
    return tuple(
        gate
        for qubit in qubits
        for gate in (
            Instruction("h", (qubit,), metadata=metadata),
            Instruction("reset", (qubit,), metadata=effect),
        )
    )


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
