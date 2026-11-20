# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Qualtran v0.7.0 ToContiguousIndex supplies the zero-based numeric contract.
# The frozen root supplies no executable decomposition.

"""Stored contiguous-index calls over shared square, add and XOR owners."""

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.registers import RegisterType
from ....core.ir.program.signatures import Signature
from ..arithmetic.addition_component import addition_component_program
from ..arithmetic.integer_square_component import integer_square_program
from ..bits.bitwise_component import bitwise_word_program
from ..component_builder import ComponentProgramBuilder
from .contiguous_index import contiguous_index_requirements


def contiguous_index_program(width: int, result_width: int) -> ProgramRecord:
    """XOR nu*(nu+1)//2+mu into a THRU result and restore all owned work.

    Inputs have equal unsigned width. Words are MSB first. Narrow results take
    low bits and wider results preserve their untouched prefix. The square's
    numerator is owned across shared additions and erased by the square inverse.
    """
    contiguous_index_requirements(width)
    if type(result_width) is not int or result_width < 1:
        raise ValueError("Contiguous index requires a positive result width")
    dtype = RegisterType("uint", width)
    wide = RegisterType("uint", 2 * width)
    output_dtype = RegisterType("uint", result_width)
    builder = ComponentProgramBuilder(
        Signature.build_from_dtypes(mu=dtype, nu=dtype, s=output_dtype)
    )
    mu, nu, output = (builder.inputs[name] for name in ("mu", "nu", "s"))
    assert (
        isinstance(mu, RegisterBinding)
        and isinstance(nu, RegisterBinding)
        and isinstance(output, RegisterBinding)
    )
    squared = builder.call_program(integer_square_program(width), {"a": nu})
    added = builder.call_program(
        addition_component_program(width, 2 * width),
        {"a": squared["a"], "b": squared["result"]},
    )
    nu = added["a"]
    numerator = added["b"]
    assert isinstance(numerator, RegisterBinding)
    bits = builder.split(numerator)
    quotient = builder.join([bits[-1], *bits[:-1]], wide)
    indexed = builder.call_program(
        addition_component_program(width, 2 * width), {"a": mu, "b": quotient}
    )
    mu = indexed["a"]
    indexed_quotient = indexed["b"]
    assert isinstance(indexed_quotient, RegisterBinding)
    word = builder.split(indexed_quotient)
    target = builder.split(output)
    copied = min(2 * width, result_width)
    view = RegisterType("uint", copied)
    xor = builder.call_program(
        bitwise_word_program(view, "xor"),
        {
            "x": builder.join(word[-copied:], view),
            "y": builder.join(target[-copied:], view),
        },
    )
    restored_low = xor["x"]
    changed_low = xor["y"]
    assert isinstance(restored_low, RegisterBinding) and isinstance(
        changed_low, RegisterBinding
    )
    quotient = builder.join([*word[:-copied], *builder.split(restored_low)], wide)
    output = builder.join(
        [*target[:-copied], *builder.split(changed_low)], output_dtype
    )
    subtracted = builder.call_program(
        addition_component_program(width, 2 * width, adjoint=True),
        {"a": mu, "b": quotient},
    )
    mu, reduced_quotient = subtracted["a"], subtracted["b"]
    assert isinstance(reduced_quotient, RegisterBinding)
    bits = builder.split(reduced_quotient)
    numerator = builder.join([*bits[1:], bits[0]], wide)
    subtracted = builder.call_program(
        addition_component_program(width, 2 * width, adjoint=True),
        {"a": nu, "b": numerator},
    )
    erased = builder.call_program(
        integer_square_program(width, adjoint=True),
        {"a": subtracted["a"], "result": subtracted["b"]},
    )
    return builder.finalize({"mu": mu, "nu": erased["a"], "s": output})
