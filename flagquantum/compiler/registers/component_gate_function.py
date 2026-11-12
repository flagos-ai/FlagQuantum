# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Typed primitive wrapping extracted from the native control component owner.
# Original control construction follows Qualtran v0.7.0 CtrlSpecAnd.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mcmt/ctrl_spec_and.py

"""Shared primitive lowering with one-bit THRU register element ports."""

from dataclasses import replace

from ...core.ir import CircuitIR
from ...core.ir.program import FunctionRecord, Operation, ValueRef
from ...core.ir.program.lowering import _lower_instructions
from ...core.ir.program.register_types import quantum_bit_type
from ...core.ir.program.registers import Register


def bit_gate_function(
    name: str, registers: tuple[Register, ...], circuit: CircuitIR
) -> FunctionRecord:
    """Wrap the shared primitive lowering in one-bit THRU element ports."""
    block = _lower_instructions(circuit, explicit_returns=True)
    offset = 1 + max(
        value.index
        for value in (
            *block.arguments,
            *(v for op in block.operations for v in op.results),
        )
    )
    inputs = tuple(
        ValueRef("entry", offset + i, "register") for i in range(circuit.n_wires)
    )
    outputs = tuple(
        ValueRef("entry", offset + circuit.n_wires + i, "register")
        for i in range(circuit.n_wires)
    )
    assert block.return_values is not None
    operations = (
        *(
            Operation("split_register", (word,), (bit,))
            for word, bit in zip(inputs, block.arguments, strict=True)
        ),
        *block.operations,
        *(
            Operation(
                "join_register", (bit,), (word,), register_type=quantum_bit_type()
            )
            for bit, word in zip(block.return_values, outputs, strict=True)
        ),
    )
    return FunctionRecord(
        name,
        (
            replace(
                block, arguments=inputs, operations=operations, return_values=outputs
            ),
        ),
        registers=registers,
    )
