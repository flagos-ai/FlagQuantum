# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 SynthesizeLRCircuit.build_call_graph.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/gf_arithmetic/gf2_multiplication.py

"""Concrete binary PLU resource declarations independent of emitted gates."""

from ....core._binary_matrix import BinaryMatrix
from ..resources.symbolic_resource_model import SymbolicResourceFormula
from .field_multiplication import _field_plu


def source_binary_linear_resource_formula(
    matrix: tuple[tuple[int, ...], ...],
) -> SymbolicResourceFormula:
    """Count the locked PLU off-diagonal ones without constructing a circuit.

    Rows and columns use the same least-significant-coefficient-first order as
    the existing linear emitter. Logical output permutations are free in this
    model. Fixed-wire binding SWAPs are counted separately by their consumer.
    """
    width = len(matrix)
    if not width or any(
        len(row) != width
        or any(type(bit) is not int or bit not in (0, 1) for bit in row)
        for row in matrix
    ):
        raise ValueError("Linear costs require a nonempty square binary matrix")
    try:
        upper, lower, _ = _field_plu([list(row) for row in matrix])
    except StopIteration:
        raise ValueError("Linear costs require an invertible binary matrix") from None
    cnots = sum(sum(row) for row in upper + lower) - 2 * width
    return SymbolicResourceFormula(
        and_bloq=0, clifford=cnots, measurement=0, logical_qubits=width
    )


def source_linear_decomposition_resource_formula(
    matrix: tuple[tuple[int, ...], ...],
) -> SymbolicResourceFormula:
    """Retain the original declaration, including negative missing-pivot counts.

    This is the literal sum(L)+sum(U)-2*n expression, not a measured primitive
    count and not a bound for singular matrices.
    """
    width = len(matrix)
    if not width or any(
        len(row) != width
        or any(type(bit) is not int or bit not in (0, 1) for bit in row)
        for row in matrix
    ):
        raise ValueError("Linear costs require a nonempty square binary matrix")
    factors = BinaryMatrix.from_dense(
        [list(row) for row in matrix]
    ).plu_with_zero_pivots()
    assert factors is not None
    _, lower, upper = factors
    cnots = (
        sum(row.bit_count() for row in lower.packed_rows + upper.packed_rows)
        - 2 * width
    )
    return SymbolicResourceFormula(
        and_bloq=0, clifford=cnots, measurement=0, logical_qubits=width
    )
