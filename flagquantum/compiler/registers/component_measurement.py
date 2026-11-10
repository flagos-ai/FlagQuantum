# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0
# MeasureX semantics adapted from Qualtran v0.7.0 x_basis.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/x_basis.py

"""Owned basis measurement composed from existing register and gate operations."""

from ...core.ir.program import BlockRecord, FunctionRecord, Operation, ValueRef
from ...core.ir.program.register_types import classical_bit_type, quantum_bit_type
from ...core.ir.program.registers import Register, RegisterSide


def measure_x_word_function(name: str) -> FunctionRecord:
    """Consume one QBit and produce a CBit using H followed by MeasureZ."""
    q, rotated, c = (ValueRef("entry", i, "register") for i in range(3))
    wire, hwire = (ValueRef("entry", i, "qubit") for i in (3, 4))
    return FunctionRecord(
        name,
        (
            BlockRecord(
                "entry",
                (q,),
                operations=(
                    Operation("split_register", (q,), (wire,)),
                    Operation("h", (wire,), (hwire,)),
                    Operation(
                        "join_register",
                        (hwire,),
                        (rotated,),
                        register_type=quantum_bit_type(),
                    ),
                    Operation("measure_register", (rotated,), (c,)),
                ),
                return_values=(c,),
            ),
        ),
        registers=(
            Register("q", quantum_bit_type(), side=RegisterSide.LEFT),
            Register("c", classical_bit_type(), side=RegisterSide.RIGHT),
        ),
    )
