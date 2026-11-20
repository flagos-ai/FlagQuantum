# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/multiplication.py
# Qualtran InvertRealNumber has a model and no root decomposition.
# Retains reciprocal_newton.py's widened exact-step-then-truncate construction.

"""Stored reciprocal Newton steps reuse exact square and live accumulation."""

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import fixed_point_register_type
from ....core.ir.program.registers import RegisterType
from ....core.ir.program.signatures import Signature
from ..arithmetic.integer_square_component import integer_square_program
from ..arithmetic.product_accumulation_component import product_accumulation_program
from ..bits.bitwise_component import bitwise_word_program
from ..component_builder import ComponentProgramBuilder
from .reciprocal_newton import reciprocal_newton_requirements


def reciprocal_newton_program(
    input_width: int, fraction: int, precision: int
) -> ProgramRecord:
    """XOR one widened Newton estimate into a THRU pure-fraction result.

    Source and estimate are preserved. For a>=1 and 0<=x<=1/a, the image is
    floor((2*x-a*x*x)*2**precision) modulo 2**precision. The widened numerator
    stays exact before the retained slice is copied. Squared estimate and
    numerator are owned across subtraction/copy/restoration, then released zero.
    """
    reciprocal_newton_requirements(input_width, precision)
    if type(fraction) is not int or not 0 <= fraction < input_width:
        raise ValueError("Reciprocal update requires a nonempty source integer part")
    source_dtype = fixed_point_register_type(input_width, fraction)
    pure = fixed_point_register_type(precision, precision)
    unsigned = RegisterType("uint", precision)
    source_unsigned = RegisterType("uint", input_width)
    numerator_width = input_width + 2 * precision
    numerator_type = RegisterType("uint", numerator_width)
    builder = ComponentProgramBuilder(
        Signature.build_from_dtypes(a=source_dtype, estimate=pure, result=pure)
    )
    source, estimate, result = (
        builder.inputs[key] for key in ("a", "estimate", "result")
    )
    assert (
        isinstance(source, RegisterBinding)
        and isinstance(estimate, RegisterBinding)
        and isinstance(result, RegisterBinding)
    )
    source = builder.join(builder.split(source), source_unsigned)
    estimate = builder.join(builder.split(estimate), unsigned)
    squared = builder.call_program(integer_square_program(precision), {"a": estimate})
    estimate, square = squared["a"], squared["result"]
    assert isinstance(estimate, RegisterBinding) and isinstance(square, RegisterBinding)
    numerator = builder.allocate(dtype=numerator_type)
    offset = fraction + precision
    start = numerator_width - offset - precision - 1
    estimate, numerator = _xor_slice(builder, estimate, numerator, start, precision)
    subtracted = builder.call_program(
        product_accumulation_program(
            input_width, 2 * precision, numerator_width, adjoint=True
        ),
        {"a": source, "b": square, "result": numerator},
    )
    next_numerator = subtracted["result"]
    assert isinstance(next_numerator, RegisterBinding)
    source, square, numerator = subtracted["a"], subtracted["b"], next_numerator
    assert (
        isinstance(source, RegisterBinding)
        and isinstance(square, RegisterBinding)
        and isinstance(numerator, RegisterBinding)
    )
    bits = builder.split(numerator)
    result_bits = builder.split(result)
    copied = builder.call_program(
        bitwise_word_program(unsigned, "xor"),
        {
            "x": builder.join(bits[start + 1 : start + 1 + precision], unsigned),
            "y": builder.join(result_bits, unsigned),
        },
    )
    retained, result = copied["x"], copied["y"]
    assert isinstance(retained, RegisterBinding) and isinstance(result, RegisterBinding)
    numerator = builder.join(
        [*bits[: start + 1], *builder.split(retained), *bits[start + 1 + precision :]],
        numerator_type,
    )
    restored = builder.call_program(
        product_accumulation_program(input_width, 2 * precision, numerator_width),
        {"a": source, "b": square, "result": numerator},
    )
    next_numerator = restored["result"]
    assert isinstance(next_numerator, RegisterBinding)
    source, square, numerator = restored["a"], restored["b"], next_numerator
    assert (
        isinstance(source, RegisterBinding)
        and isinstance(square, RegisterBinding)
        and isinstance(numerator, RegisterBinding)
    )
    estimate, numerator = _xor_slice(builder, estimate, numerator, start, precision)
    builder.free(numerator)
    erased = builder.call_program(
        integer_square_program(precision, adjoint=True),
        {"a": estimate, "result": square},
    )
    estimate = erased["a"]
    assert isinstance(estimate, RegisterBinding)
    return builder.finalize(
        {
            "a": builder.join(builder.split(source), source_dtype),
            "estimate": builder.join(builder.split(estimate), pure),
            "result": builder.join(builder.split(result), pure),
        }
    )


def _xor_slice(
    builder: ComponentProgramBuilder,
    source: RegisterBinding,
    target: RegisterBinding,
    start: int,
    width: int,
) -> tuple[RegisterBinding, RegisterBinding]:
    """XOR a word into a numerator slice while retaining its surrounding views."""
    bits = builder.split(target)
    dtype = RegisterType("uint", width)
    changed = builder.call_program(
        bitwise_word_program(dtype, "xor"),
        {"x": source, "y": builder.join(bits[start : start + width], dtype)},
    )
    updated_source, middle = changed["x"], changed["y"]
    assert isinstance(updated_source, RegisterBinding) and isinstance(
        middle, RegisterBinding
    )
    return updated_source, builder.join(
        [*bits[:start], *builder.split(middle), *bits[start + width :]],
        target.register.dtype,
    )
