# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 _GF2MulViaKaratsubaImpl/GF2MulViaKaratsuba at
# bcb17bb25360ed68ce3bc084645db3e929799389, gf2_multiplication.py.

"""Stored recursive field multiplication with exact constant-scaling adjoints."""

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.registers import Register, RegisterSide, RegisterType
from ....core.ir.program.signatures import Signature
from ..component_builder import ComponentProgramBuilder
from ._polynomial_component_calls import (
    field_word_call,
    product_bit_functions,
    product_ccx,
    product_cx,
)
from .constant_multiplication_component import (
    constant_multiplication_decomposition_program,
    constant_multiplication_program,
)
from .field_word import field_word_width
from .monomial_component import monomial_multiplication_program
from .polynomial_product_component import (
    binary_polynomial_program,
    polynomial_product_call,
)


def karatsuba_implementation_program(
    dtype: RegisterType, *, modulus: int
) -> ProgramRecord:
    """Preserve f/g and transform arbitrary h to x**ceil(n/2)*h + f*g.

    Native exact linear adjoints preserve the previously accepted mathematical
    emitter. The source root has the same THRU interface, while its frozen
    ignored-adjoint defect remains explicit. A zero h gives the clean product.
    Coefficient recursion owns and clears its high padding.
    """
    width = field_word_width(dtype, modulus)
    constant = 1 | (1 << ((width + 1) // 2))
    inverse_scale = (
        constant_multiplication_program(dtype, constant, modulus=modulus, adjoint=True)
        if width > 1
        else None
    )
    return _karatsuba_implementation_program(
        dtype, modulus=modulus, inverse_scale=inverse_scale
    )


def karatsuba_decomposition_implementation(
    dtype: RegisterType, *, modulus: int
) -> ProgramRecord:
    """Preserve the original raw f/g/h circuit, including forward scale adjoint.

    The source linear physical builder ignores its declared inverse direction.
    This profile therefore supplies forward scaling at that call. It is separate
    from the mathematical implementation and does not promise h += f*g.
    """
    width = field_word_width(dtype, modulus)
    if width < 2:
        raise ValueError(
            "Literal Karatsuba implementation requires degree at least two"
        )
    constant = 1 | (1 << ((width + 1) // 2))
    inverse_scale = (
        constant_multiplication_decomposition_program(
            dtype, constant, modulus=modulus, adjoint=True
        )
        if width > 1
        else None
    )
    return _karatsuba_implementation_program(
        dtype, modulus=modulus, inverse_scale=inverse_scale
    )


def _karatsuba_implementation_program(
    dtype: RegisterType, *, modulus: int, inverse_scale: ProgramRecord | None
) -> ProgramRecord:
    width = field_word_width(dtype, modulus)
    builder = ComponentProgramBuilder(
        Signature.build_from_dtypes(f=dtype, g=dtype, h=dtype), product_bit_functions()
    )
    words = tuple(builder.inputs[p] for p in ("f", "g", "h"))
    assert all(isinstance(word, RegisterBinding) for word in words)
    f, g, h = (
        builder.split(word)[::-1] for word in words if isinstance(word, RegisterBinding)
    )
    if width == 1:
        f[0], g[0], h[0] = product_ccx(builder, f[0], g[0], h[0])
        return builder.finalize(
            {
                "f": builder.join(f[::-1], dtype),
                "g": builder.join(g[::-1], dtype),
                "h": builder.join(h[::-1], dtype),
            }
        )
    k = (width + 1) // 2
    for bits in (f, g):
        for i in range(width - k):
            bits[k + i], bits[i] = product_cx(builder, bits[k + i], bits[i])
    f[:k], g[:k], h[: 2 * k - 1] = polynomial_product_call(
        builder, binary_polynomial_program(k), f[:k], g[:k], h[: 2 * k - 1]
    )
    for bits in (g, f):
        for i in range(width - k):
            bits[k + i], bits[i] = product_cx(builder, bits[k + i], bits[i])
    constant = 1 | (1 << k)
    assert inverse_scale is not None
    h = field_word_call(
        builder,
        inverse_scale,
        "g",
        h,
        dtype,
    )
    f[k:], g[k:], h[: 2 * (width - k) - 1] = polynomial_product_call(
        builder,
        binary_polynomial_program(width - k),
        f[k:],
        g[k:],
        h[: 2 * (width - k) - 1],
    )
    h = field_word_call(
        builder,
        monomial_multiplication_program(dtype, k, modulus=modulus),
        "f",
        h,
        dtype,
    )
    f[:k], g[:k], h[: 2 * k - 1] = polynomial_product_call(
        builder, binary_polynomial_program(k), f[:k], g[:k], h[: 2 * k - 1]
    )
    h = field_word_call(
        builder,
        constant_multiplication_program(dtype, constant, modulus=modulus),
        "g",
        h,
        dtype,
    )
    return builder.finalize(
        {
            "f": builder.join(f[::-1], dtype),
            "g": builder.join(g[::-1], dtype),
            "h": builder.join(h[::-1], dtype),
        }
    )


def karatsuba_multiplication_program(
    dtype: RegisterType, *, modulus: int
) -> ProgramRecord:
    """Allocate a RIGHT field product and inline the original implementation shell.

    Public source construction requires degree at least two. The source adjoint
    is the independently owned measured product-erasure program, not generic
    reversal of the defective frozen linear child.
    """
    return _karatsuba_output_program(
        dtype, karatsuba_implementation_program(dtype, modulus=modulus), modulus=modulus
    )


def karatsuba_decomposition_program(
    dtype: RegisterType, *, modulus: int
) -> ProgramRecord:
    """Allocate the original RIGHT result and execute its literal physical body.

    The output matches the frozen source circuit, including its product defect.
    It is not interchangeable with the native mathematical multiplication.
    """
    return _karatsuba_output_program(
        dtype,
        karatsuba_decomposition_implementation(dtype, modulus=modulus),
        modulus=modulus,
    )


def _karatsuba_output_program(
    dtype: RegisterType, implementation: ProgramRecord, *, modulus: int
) -> ProgramRecord:
    width = field_word_width(dtype, modulus)
    if width < 2:
        raise ValueError("Karatsuba field multiplication requires degree at least two")
    builder = ComponentProgramBuilder(
        Signature(
            (
                Register("x", dtype),
                Register("y", dtype),
                Register("result", dtype, side=RegisterSide.RIGHT),
            )
        )
    )
    result = builder.allocate(dtype=dtype)
    values = builder.inline_calls(
        implementation,
        {"f": builder.inputs["x"], "g": builder.inputs["y"], "h": result},
    )
    return builder.finalize({"x": values["f"], "y": values["g"], "result": values["h"]})
