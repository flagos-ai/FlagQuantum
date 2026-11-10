# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Classical control semantics adapted from Qualtran v0.7.0.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mcmt/classically_controlled.py

"""Owned classical controls composed with existing conditional quantum gates."""

from ...core.ir.program import (
    BlockRecord,
    ConditionRef,
    FunctionRecord,
    Operation,
    ValueRef,
)
from ...core.ir.program.register_types import classical_bit_type, quantum_bit_type
from ...core.ir.program.registers import Register


def classical_cz_word_function(name: str) -> FunctionRecord:
    """Forward one owned CBit and apply CZ when its shared snapshot is one."""
    c, q1, q2, successor, out1, out2 = (
        ValueRef("entry", i, "register") for i in range(6)
    )
    snapshot = ValueRef("entry", 6, "bool")
    a, b, ra, rb = (ValueRef("entry", i, "qubit") for i in range(7, 11))
    return FunctionRecord(
        name,
        (
            BlockRecord(
                "entry",
                (c, q1, q2),
                operations=(
                    Operation("read_classical_register", (c,), (successor, snapshot)),
                    Operation("split_register", (q1,), (a,)),
                    Operation("split_register", (q2,), (b,)),
                    Operation(
                        "cz",
                        (a, b),
                        (ra, rb),
                        condition_clauses=((ConditionRef(snapshot, True),),),
                    ),
                    Operation(
                        "join_register",
                        (ra,),
                        (out1,),
                        register_type=quantum_bit_type(),
                    ),
                    Operation(
                        "join_register",
                        (rb,),
                        (out2,),
                        register_type=quantum_bit_type(),
                    ),
                ),
                return_values=(successor, out1, out2),
            ),
        ),
        registers=(
            Register("ctrl", classical_bit_type()),
            Register("q1", quantum_bit_type()),
            Register("q2", quantum_bit_type()),
        ),
    )
