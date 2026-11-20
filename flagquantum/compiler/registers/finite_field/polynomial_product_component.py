# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 BinaryPolynomialMultiplication and
# MultiplyPolyByOnePlusXk, bcb17bb25360ed68ce3bc084645db3e929799389.

"""Recursive stored multiply-accumulate calls on binary coefficient arrays."""

import numpy as np
from numpy.typing import NDArray

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.registers import Register
from ....core.ir.program.signatures import Signature
from ..component_builder import ComponentProgramBuilder
from ._polynomial_component_calls import product_bit_functions, product_ccx, product_cx
from .polynomial import binary_polynomial_workspace_size


def polynomial_product_call(
    builder: ComponentProgramBuilder,
    child: ProgramRecord,
    f: NDArray[np.object_],
    g: NDArray[np.object_],
    h: NDArray[np.object_],
) -> tuple[NDArray[np.object_], NDArray[np.object_], NDArray[np.object_]]:
    """Follow the returned coefficient aliases of an existing product child."""
    result = builder.call_program(child, {"f": f, "g": g, "h": h})
    f, g, h = (np.asarray(result[port], dtype=object) for port in ("f", "g", "h"))
    return f, g, h


def _builder(width: int, target_width: int) -> ComponentProgramBuilder:
    bit = quantum_bit_type()
    return ComponentProgramBuilder(
        Signature(
            (
                Register("f", bit, (width,)),
                Register("g", bit, (width,)),
                Register("h", bit, (target_width,)),
            )
        ),
        product_bit_functions(),
    )


def binary_polynomial_program(width: int) -> ProgramRecord:
    """XOR fg into h, with coefficient zero first and source-owned clean padding.

    Inputs f/g have width n and h has width 2n-1. The recursion shares child
    definitions, preserves arbitrary h and frees only its extra high coefficients.
    Physical MSB-word consumers reverse their bit views before entering this graph.
    """
    binary_polynomial_workspace_size(width)
    builder = _builder(width, 2 * width - 1)
    f, g, h = (
        np.asarray(builder.inputs[p], dtype=object).copy() for p in ("f", "g", "h")
    )
    if width == 1:
        f[0], g[0], h[0] = product_ccx(builder, f[0], g[0], h[0])
        return builder.finalize({"f": f, "g": g, "h": h})
    k = (width + 1) // 2
    f[:k], g[:k], h[: 3 * k - 1] = polynomial_product_call(
        builder, one_plus_polynomial_program(k, k), f[:k], g[:k], h[: 3 * k - 1]
    )
    child_width = 2 * k + max(0, 2 * (width - k) - k - 1)
    delta = max(0, k + child_width - len(h))
    if delta:
        aux = builder.allocate(delta)
        h = np.concatenate((h, builder.split(aux)))
    f[k:], g[k:], h[k : k + child_width] = polynomial_product_call(
        builder,
        one_plus_polynomial_program(width - k, k),
        f[k:],
        g[k:],
        h[k : k + child_width],
    )
    if delta:
        builder.free(builder.join(h[-delta:]))
        h = h[:-delta]
    for bits in (f, g):
        for i in range(width - k):
            bits[k + i], bits[i] = product_cx(builder, bits[k + i], bits[i])
    f[:k], g[:k], h[k : 3 * k - 1] = polynomial_product_call(
        builder, binary_polynomial_program(k), f[:k], g[:k], h[k : 3 * k - 1]
    )
    for bits in (g, f):
        for i in range(width - k):
            bits[k + i], bits[i] = product_cx(builder, bits[k + i], bits[i])
    return builder.finalize({"f": f, "g": g, "h": h})


def one_plus_polynomial_program(width: int, shift: int) -> ProgramRecord:
    """XOR (1+x**shift)fg into h, preserving the original overlapping CX order.

    The accepted concrete shift range is 1..n+1. It retains the source's wider
    target at n=1. Source unvalidated larger shifts remain a separate boundary.
    """
    binary_polynomial_workspace_size(width)
    if type(shift) is not int or not 1 <= shift <= width + 1:
        raise ValueError("Polynomial shift must be a concrete integer in [1, n+1]")
    tail = max(0, 2 * width - shift - 1)
    builder = _builder(width, 2 * shift + tail)
    f, g, h = (
        np.asarray(builder.inputs[p], dtype=object).copy() for p in ("f", "g", "h")
    )
    if width == 1:
        h[shift], h[0] = product_cx(builder, h[shift], h[0])
        f[0], g[0], h[shift] = product_ccx(builder, f[0], g[0], h[shift])
        h[shift], h[0] = product_cx(builder, h[shift], h[0])
        return builder.finalize({"f": f, "g": g, "h": h})
    for i in reversed(range(tail)):
        h[2 * shift + i], h[shift + i] = product_cx(
            builder, h[2 * shift + i], h[shift + i]
        )
    for i in range(shift):
        h[shift + i], h[i] = product_cx(builder, h[shift + i], h[i])
    f, g, h[shift : 2 * shift + tail] = polynomial_product_call(
        builder, binary_polynomial_program(width), f, g, h[shift : 2 * shift + tail]
    )
    for i in range(shift):
        h[shift + i], h[i] = product_cx(builder, h[shift + i], h[i])
    for i in range(tail):
        h[2 * shift + i], h[shift + i] = product_cx(
            builder, h[2 * shift + i], h[shift + i]
        )
    return builder.finalize({"f": f, "g": g, "h": h})
