# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from frozen Qualtran v0.7.0 gf2_multiplication.build_call_graph methods.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/gf_arithmetic/gf2_multiplication.py

"""Source multiplication declarations, independent of executable gate emission."""

from __future__ import annotations

from collections.abc import Hashable
from typing import Literal

from sympy import ceiling, log

from ..._component_trace.call_graph import SymbolicCallGraph, symbolic_call_graph
from ..._component_trace.resource_generalization import CliffordCall, FixedTCall
from ..._component_trace.symbolic_math import SymbolicInt, smax
from .field_multiplication import _product, _validate_field
from .linear_resource_model import source_binary_linear_resource_formula

CX = CliffordCall("cx", 2)
TOFFOLI = FixedTCall("toffoli")


def polynomial_multiplication_calls(
    width: SymbolicInt, shift: SymbolicInt | None = None
) -> dict[Hashable, SymbolicInt]:
    """Retain every direct child, including zero-count CNOT declarations."""
    if shift is not None:
        if width == 1:
            return {CX: 2, TOFFOLI: 1}
        overlap = smax(0, 2 * width - shift - 1)
        return {CX: 2 * (overlap + shift), ("polynomial", width): 1}
    if width == 1:
        return {TOFFOLI: 1}
    half = (width + 1) // 2 if isinstance(width, int) else ceiling(width / 2)
    calls: dict[Hashable, SymbolicInt] = {
        CX: 4 * (width - half),
        ("polynomial", half): 1,
        ("one_plus", half, half): 1,
    }
    other = ("one_plus", width - half, half)
    calls[other] = calls.get(other, 0) + 1
    return calls


def polynomial_multiplication_graph(
    width: int, shift: int | None = None, *, max_depth: int | None = None
) -> SymbolicCallGraph[Hashable]:
    """Expand the concrete recursive declaration graph without constructing gates."""
    if type(width) is not int or width < 1:
        raise ValueError("Polynomial model width must be a positive integer")
    if shift is not None and (type(shift) is not int or not 1 <= shift <= width + 1):
        raise ValueError("Polynomial model shift must be in [1, n+1]")
    root = ("polynomial", width) if shift is None else ("one_plus", width, shift)
    declarations: dict[Hashable, dict[Hashable, SymbolicInt]] = {}
    pending = [root]
    while pending:
        node = pending.pop()
        if node in declarations:
            continue
        calls = polynomial_multiplication_calls(
            node[1], node[2] if len(node) == 3 else None
        )
        declarations[node] = calls
        pending.extend(child for child in calls if isinstance(child, tuple))
    return symbolic_call_graph(root, declarations, max_depth=max_depth)


def field_multiplication_gate_model(
    width: SymbolicInt,
    *,
    modulus: int | None = None,
    algorithm: Literal["quadratic", "karatsuba", "erasure"] = "quadratic",
    plus_equal_prod: bool = False,
) -> dict[Hashable, SymbolicInt]:
    """Report source QEC primitive categories separately from physical counts.

    Shape-only shifts retain the original pentanomial assumption. The source
    erasure declaration contains only CZ, omitting measurement and parity work.
    Karatsuba's linear adjoint defect changes gates, while preserving counts.
    """
    if algorithm not in ("quadratic", "karatsuba", "erasure"):
        raise ValueError("Unknown field multiplication model")
    if plus_equal_prod and algorithm != "quadratic":
        raise ValueError("Source Karatsuba and erasure models have a clean output")
    if algorithm == "erasure":
        return {CliffordCall("cz", 2): width**2}
    if not isinstance(width, int):
        if algorithm == "karatsuba":
            return {TOFFOLI: width ** (log(3) / log(2)), CX: width**2}
        return {TOFFOLI: width**2, CX: 3 * width * (2 if plus_equal_prod else 1)}
    if modulus is None:
        raise ValueError("Concrete field models require an explicit modulus")
    _validate_field(width, modulus)
    weight = max(modulus.bit_count() - 2, 0)
    if algorithm == "quadratic":
        return {TOFFOLI: width**2, CX: width * weight * (2 if plus_equal_prod else 1)}
    if width == 1:
        return {TOFFOLI: 1}
    half = (width + 1) // 2
    low = polynomial_multiplication_graph(half).leaf_counts
    high = polynomial_multiplication_graph(width - half).leaf_counts
    factor = 1 | (1 << half)
    columns = [_product(factor, 1 << i, modulus) for i in range(width)]
    matrix = tuple(
        tuple((column >> row) & 1 for column in columns) for row in range(width)
    )
    linear = source_binary_linear_resource_formula(matrix).clifford
    return {
        TOFFOLI: 2 * low.get(TOFFOLI, 0) + high.get(TOFFOLI, 0),
        CX: 4 * (width - half)
        + 2 * low.get(CX, 0)
        + high.get(CX, 0)
        + 2 * linear
        + half * weight,
    }
