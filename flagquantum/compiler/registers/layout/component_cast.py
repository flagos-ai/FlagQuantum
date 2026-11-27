# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Cast in Qualtran v0.7.0.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/bookkeeping/cast.py

"""Finite storage casts using existing owned-word Program rebindings."""

from ....core.ir.program import BlockRecord, FunctionRecord, Operation, ValueRef
from ....core.ir.program.register_types import classical_bit_type
from ....core.ir.program.register_verifier import (
    executable_register_width,
    logical_classical_register_width,
    register_port_types,
)
from ....core.ir.program.registers import RegisterSide, RegisterType
from ..component_builder import ComponentProgramBuilder
from .register_cast import cast_signature


def cast_word_function(
    name: str,
    source: RegisterType,
    target: RegisterType,
    shape: tuple[int, ...] = (),
) -> FunctionRecord:
    """Consume words and return new typed views over the same ordered wires.

    Elements follow row-major order; bits retain the existing MSB-first order.
    Executable quantum words reuse split/join. Classical words retain their slots.
    The view adds no measurement, reset or allocation.
    """
    signature = cast_signature(source, target, shape)
    if source == target == classical_bit_type():
        builder = ComponentProgramBuilder(signature, name=name)
        return builder.finalize(builder.inputs).functions[0]
    classical = source.quantum_bits == 0 and (
        source == classical_bit_type() or ("storage", "classical") in source.parameters
    )
    if classical:
        width = logical_classical_register_width(source)
        if width != logical_classical_register_width(target):
            raise ValueError("classical cast must preserve raw bit width")
    else:
        width = executable_register_width(source)
        executable_register_width(target)
    registers = tuple(signature)
    count = len(
        register_port_types(
            FunctionRecord(name, (), registers=registers), RegisterSide.LEFT
        )
    )
    inputs = tuple(ValueRef("entry", i, "register") for i in range(count))
    outputs = tuple(ValueRef("entry", count + i, "register") for i in range(count))
    operations: list[Operation] = []
    for index, (word, output) in enumerate(zip(inputs, outputs, strict=True)):
        if classical:
            operations.append(
                Operation(
                    "cast_classical_register", (word,), (output,), register_type=target
                )
            )
            continue
        assert isinstance(width, int)
        bits = tuple(
            ValueRef("entry", 2 * count + index * width + j, "qubit")
            for j in range(width)
        )
        operations.extend(
            (
                Operation("split_register", (word,), bits),
                Operation("join_register", bits, (output,), register_type=target),
            )
        )
    return FunctionRecord(
        name,
        (
            BlockRecord(
                "entry", inputs, operations=tuple(operations), return_values=outputs
            ),
        ),
        registers=registers,
    )
