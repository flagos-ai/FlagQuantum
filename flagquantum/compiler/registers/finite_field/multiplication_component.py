# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 GF2Multiplication at
# bcb17bb25360ed68ce3bc084645db3e929799389, gf2_multiplication.py.

"""Quadratic field product graphs with owned or arbitrary-target results."""

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.registers import Register, RegisterSide, RegisterType
from ....core.ir.program.signatures import Signature
from ..component_builder import ComponentProgramBuilder
from ._polynomial_component_calls import (
    field_word_call,
    product_bit_functions,
    product_ccx,
)
from .field_word import field_word_width
from .monomial_component import monomial_multiplication_program


def field_multiplication_program(
    dtype: RegisterType,
    *,
    modulus: int,
    plus_equal_prod: bool = False,
    adjoint: bool = False,
) -> ProgramRecord:
    """Compute xy, or XOR it into a through target, preserving source child order.

    RIGHT results are newly allocated logical data. Their exact inverse accepts
    a matching LEFT result and frees it after uncomputation. Through accumulation
    admits arbitrary target amplitudes. All words are MSB-first and work is zero.
    """
    width = field_word_width(dtype, modulus)
    if type(plus_equal_prod) is not bool or type(adjoint) is not bool:
        raise ValueError("Field product mode and direction require booleans")
    side = (
        RegisterSide.THRU
        if plus_equal_prod
        else (RegisterSide.LEFT if adjoint else RegisterSide.RIGHT)
    )
    builder = ComponentProgramBuilder(
        Signature(
            (
                Register("x", dtype),
                Register("y", dtype),
                Register("result", dtype, side=side),
            )
        ),
        product_bit_functions(),
    )
    result = (
        builder.inputs["result"]
        if plus_equal_prod or adjoint
        else builder.allocate(dtype=dtype)
    )
    assert isinstance(result, RegisterBinding)
    x, y = (builder.inputs[p] for p in ("x", "y"))
    assert isinstance(x, RegisterBinding) and isinstance(y, RegisterBinding)
    xs, ys = builder.split(x)[::-1], builder.split(y)[::-1]
    if plus_equal_prod and not adjoint:
        result = builder.call_program(
            monomial_multiplication_program(dtype, -width, modulus=modulus),
            {"f": result},
        )["f"]
        assert isinstance(result, RegisterBinding)
    bits = builder.split(result)[::-1]
    upper = tuple(
        (width - j + i, j, i) for i in range(width) for j in range(i + 1, width)
    )
    lower = tuple((j, i - j, i) for i in range(width) for j in range(i + 1))
    first, second = (
        (tuple(reversed(lower)), tuple(reversed(upper))) if adjoint else (upper, lower)
    )
    for left, right, target in first:
        xs[left], ys[right], bits[target] = product_ccx(
            builder, xs[left], ys[right], bits[target]
        )
    bits = field_word_call(
        builder,
        monomial_multiplication_program(
            dtype, -width if adjoint else width, modulus=modulus
        ),
        "f",
        bits,
        dtype,
    )
    for left, right, target in second:
        xs[left], ys[right], bits[target] = product_ccx(
            builder, xs[left], ys[right], bits[target]
        )
    if plus_equal_prod and adjoint:
        bits = field_word_call(
            builder,
            monomial_multiplication_program(dtype, width, modulus=modulus),
            "f",
            bits,
            dtype,
        )
    outputs = {"x": builder.join(xs[::-1], dtype), "y": builder.join(ys[::-1], dtype)}
    result = builder.join(bits[::-1], dtype)
    if adjoint and not plus_equal_prod:
        builder.free(result)
    else:
        outputs["result"] = result
    return builder.finalize(outputs)
