# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Rebinding signatures adapted from Qualtran v0.7.0 Split and Join.
# https://github.com/quantumlib/Qualtran/tree/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/bookkeeping

"""Real word-to-bit-word functions over existing typed Program rebindings."""

from ...core.ir.program import BlockRecord, FunctionRecord, Operation, ValueRef
from ...core.ir.program.register_verifier import executable_register_width
from ...core.ir.program.registers import Register, RegisterSide, RegisterType


def split_word_function(name: str, dtype: RegisterType) -> FunctionRecord:
    """Consume one word and expose an array of owned width-one words."""
    width = executable_register_width(dtype)
    word = ValueRef("entry", 0, "register")
    bits = tuple(ValueRef("entry", 1 + index, "qubit") for index in range(width))
    outputs = tuple(
        ValueRef("entry", 1 + width + index, "register") for index in range(width)
    )
    bit_type = RegisterType("bit", 1)
    operations = (
        Operation("split_register", (word,), bits),
        *(
            Operation("join_register", (bit,), (output,), register_type=bit_type)
            for bit, output in zip(bits, outputs, strict=True)
        ),
    )
    return FunctionRecord(
        name,
        (BlockRecord("entry", (word,), operations=operations, return_values=outputs),),
        registers=(
            Register("reg", dtype, side=RegisterSide.LEFT),
            Register("reg", bit_type, (width,), RegisterSide.RIGHT),
        ),
    )


def join_word_function(name: str, dtype: RegisterType) -> FunctionRecord:
    """Consume ordered bit-word inputs and return a fresh owned word."""
    width = executable_register_width(dtype)
    inputs = tuple(ValueRef("entry", index, "register") for index in range(width))
    bits = tuple(ValueRef("entry", width + index, "qubit") for index in range(width))
    output = ValueRef("entry", 2 * width, "register")
    operations = (
        *(
            Operation("split_register", (word,), (bit,))
            for word, bit in zip(inputs, bits, strict=True)
        ),
        Operation("join_register", bits, (output,), register_type=dtype),
    )
    return FunctionRecord(
        name,
        (BlockRecord("entry", inputs, operations=operations, return_values=(output,)),),
        registers=(
            Register("reg", RegisterType("bit", 1), (width,), RegisterSide.LEFT),
            Register("reg", dtype, side=RegisterSide.RIGHT),
        ),
    )
