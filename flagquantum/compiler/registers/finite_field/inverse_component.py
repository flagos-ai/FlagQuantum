# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 GF2Inverse.build_composite_bloq,
# bcb17bb25360ed68ce3bc084645db3e929799389, gf2_inverse.py.

"""Store the Itoh--Tsujii source schedule with explicit result and logical junk."""

from collections.abc import Callable

import numpy as np

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.registers import Register, RegisterSide, RegisterType
from ....core.ir.program.signatures import Signature
from ..component_builder import BindingCollection, ComponentProgramBuilder
from .addition_component import field_addition_program
from .field_word import field_word_width
from .frobenius_component import field_frobenius_program
from .inverse_model import inverse_junk_register_count
from .karatsuba_component import karatsuba_multiplication_program


def field_inverse_program(dtype: RegisterType, *, modulus: int) -> ProgramRecord:
    """Return x, a RIGHT inverse and the source's retained RIGHT junk words.

    All words are MSB-first. Zero maps to zero. Logical junk keeps the original
    source shape and ownership. Only original temporary and recursive padding
    are released as clean work. Existing exact native Frobenius/scale adjoints
    retain their separately recorded differences from the frozen source.
    """
    width = field_word_width(dtype, modulus)
    return _field_inverse_schedule(
        dtype,
        width,
        karatsuba_multiplication_program(dtype, modulus=modulus) if width > 1 else None,
        lambda count, adjoint: field_frobenius_program(
            dtype, count, modulus=modulus, adjoint=adjoint
        ),
    )


def _field_inverse_schedule(
    dtype: RegisterType,
    width: int,
    product: ProgramRecord | None,
    square_program: Callable[[int, bool], ProgramRecord],
) -> ProgramRecord:
    """Share the original root ordering, aliases and ownership across profiles."""
    registers = [
        Register("x", dtype),
        Register("result", dtype, side=RegisterSide.RIGHT),
    ]
    if width > 1:
        junk = inverse_junk_register_count(width)
        assert isinstance(junk, int)
        registers.append(Register("junk", dtype, (junk,), side=RegisterSide.RIGHT))
    builder = ComponentProgramBuilder(Signature(registers))
    addition = field_addition_program(dtype)
    x = builder.inputs["x"]
    if width == 1:
        result = builder.call_program(
            addition, {"x": x, "y": builder.allocate(dtype=dtype)}
        )
        return builder.finalize({"x": result["x"], "result": result["y"]})

    assert product is not None
    weight = (width - 1).bit_count()
    top = (width - 1).bit_length() - 1
    last = max(top + weight - 1, top + 1)
    words: dict[int, BindingCollection] = {0: x, last: builder.allocate(dtype=dtype)}
    for i in range(1, top + 1):
        added = builder.call_program(addition, {"x": words[i - 1], "y": words[last]})
        words[i - 1], words[last] = added["x"], added["y"]
        square = square_program(2 ** (i - 1), False)
        words[last] = builder.call_program(square, {"x": words[last]})["x"]
        multiplied = builder.call_program(
            product, {"x": words[i - 1], "y": words[last]}
        )
        words[i - 1], words[last], words[i] = (
            multiplied["x"],
            multiplied["y"],
            multiplied["result"],
        )
        inverse_square = square_program(2 ** (i - 1), True)
        words[last] = builder.call_program(inverse_square, {"x": words[last]})["x"]
        added = builder.call_program(addition, {"x": words[i - 1], "y": words[last]})
        words[i - 1], words[last] = added["x"], added["y"]
    bits = [i for i in reversed(range(top + 1)) if (width - 1) >> i & 1]
    if top + weight - 1 == last:
        temporary = words[last]
        assert isinstance(temporary, RegisterBinding)
        builder.free(temporary)
    for s, bit in enumerate(bits[1:], start=1):
        square = square_program(2**bit, False)
        words[top + s - 1] = builder.call_program(square, {"x": words[top + s - 1]})[
            "x"
        ]
        multiplied = builder.call_program(
            product, {"x": words[top + s - 1], "y": words[bit]}
        )
        words[top + s - 1], words[bit], words[top + s] = (
            multiplied["x"],
            multiplied["y"],
            multiplied["result"],
        )
    if weight == 1:
        if top == 0:
            added = builder.call_program(addition, {"x": words[0], "y": words[last]})
            words[0], words[last] = added["x"], added["y"]
        words[top], words[last] = words[last], words[top]
    square = square_program(1, False)
    words[last] = builder.call_program(square, {"x": words[last]})["x"]
    return builder.finalize(
        {
            "x": words[0],
            "result": words[last],
            "junk": np.array([words[i] for i in range(1, last)], dtype=object),
        }
    )
