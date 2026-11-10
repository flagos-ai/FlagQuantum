# Copyright 2024-2025 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Signatures adapted from Qualtran v0.7.0 Parity and classical Discard.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/discard.py
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/gf_arithmetic/gf2_multiplication.py

"""Construct finite owned classical components over existing Program operations."""

from ...core.ir.program import BlockRecord, FunctionRecord, Operation, ValueRef
from ...core.ir.program.register_types import classical_bit_type
from ...core.ir.program.registers import Register, RegisterSide


def parity_word_function(name: str, count: int) -> FunctionRecord:
    """Preserve ordered CBit elements and produce their owned parity result."""
    if type(count) is not int or count < 0:
        raise ValueError("classical parity requires a built-in nonnegative count")
    inputs = tuple(ValueRef("entry", index, "register") for index in range(count))
    outputs = tuple(
        ValueRef("entry", count + index, "register") for index in range(count + 1)
    )
    return FunctionRecord(
        name,
        (
            BlockRecord(
                "entry",
                inputs,
                operations=(Operation("parity_register", inputs, outputs),),
                return_values=outputs,
            ),
        ),
        registers=(
            Register("x", classical_bit_type(), (count,)),
            Register("parity", classical_bit_type(), side=RegisterSide.RIGHT),
        ),
    )


def discard_word_function(name: str) -> FunctionRecord:
    """Consume one owned classical bit without producing a quantum instruction.

    Follows Qualtran v0.7.0 basic_gates/discard.py Discard (Google LLC, 2025,
    Apache-2.0). Previously shared bool snapshots remain independently readable.
    """
    value = ValueRef("entry", 0, "register")
    return FunctionRecord(
        name,
        (
            BlockRecord(
                "entry",
                (value,),
                operations=(Operation("discard_classical_register", (value,)),),
                return_values=(),
            ),
        ),
        registers=(Register("c", classical_bit_type(), side=RegisterSide.LEFT),),
    )
