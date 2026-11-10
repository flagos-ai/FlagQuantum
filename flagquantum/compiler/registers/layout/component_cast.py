# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Cast in Qualtran v0.7.0.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/bookkeeping/cast.py

"""Finite quantum casts using existing owned-word Program rebindings."""

from ....core.ir.program import BlockRecord, FunctionRecord, Operation, ValueRef
from ....core.ir.program.register_verifier import (
    executable_register_width,
    register_port_types,
)
from ....core.ir.program.registers import Register, RegisterSide, RegisterType
from .register_cast import RegisterCast


def cast_word_function(
    name: str,
    source: RegisterType,
    target: RegisterType,
    shape: tuple[int, ...] = (),
) -> FunctionRecord:
    """Consume words and return new typed views over the same ordered wires.

    Elements follow row-major order; bits retain the existing MSB-first order.
    This admits only Core-executable quantum encodings. No measurement, reset,
    allocation, classical conversion, or runtime state validation is performed.
    """
    width = executable_register_width(source)
    RegisterCast(width, executable_register_width(target), shape)
    registers = (
        Register("reg", source, shape, RegisterSide.LEFT),
        Register("reg", target, shape, RegisterSide.RIGHT),
    )
    count = len(
        register_port_types(
            FunctionRecord(name, (), registers=registers), RegisterSide.LEFT
        )
    )
    inputs = tuple(ValueRef("entry", i, "register") for i in range(count))
    outputs = tuple(ValueRef("entry", count + i, "register") for i in range(count))
    operations: list[Operation] = []
    for index, (word, output) in enumerate(zip(inputs, outputs, strict=True)):
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
