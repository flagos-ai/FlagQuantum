# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from frozen Qualtran v0.7.0 polynomial adjoint decompositions.

"""Retain canonical field-child order while reversing polynomial calls."""

import numpy as np
import pytest

from flagquantum.compiler.registers.finite_field.field_polynomial_addition import (
    gf2_polynomial_add_constant_ir,
    gf2_polynomial_add_ir,
)
from flagquantum.compiler.registers.finite_field.field_polynomial_layout import (
    FieldPolynomialLayout,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("constant", [False, True])
@pytest.mark.parametrize("degree,width", [(0, 1), (1, 2), (4, 3)])
def test_literal_generic_adjoint_matches_unchanged_source(
    constant: bool, degree: int, width: int
) -> None:
    import cirq
    import galois
    from qualtran import QGF, QGFPoly
    from qualtran.bloqs.gf_poly_arithmetic.gf2_poly_add import GF2PolyAdd
    from qualtran.bloqs.gf_poly_arithmetic.gf2_poly_add_k import GF2PolyAddK

    layout = FieldPolynomialLayout(degree, width)
    size = layout.logical_width
    left = tuple(range(0, 2 * size, 2))
    right = tuple(reversed(range(1, 2 * size, 2)))
    dtype = QGFPoly(degree, QGF(2, width))
    coefficients = tuple((i + 1) % (1 << width) for i in range(degree + 1))
    original = (
        GF2PolyAddK(dtype, galois.Poly(dtype.qgf.gf_type(coefficients)))
        if constant
        else GF2PolyAdd(dtype)
    )
    registers = {"f_x": np.asarray([cirq.LineQubit(i) for i in left])}
    gates = (
        gf2_polynomial_add_constant_ir(left, coefficients, layout, adjoint=True)
        if constant
        else gf2_polynomial_add_ir(left, right, layout, adjoint=True)
    )
    if not constant:
        registers["g_x"] = np.asarray([cirq.LineQubit(i) for i in right])
    inverse, outputs = (
        original.adjoint()
        .decompose_bloq()
        .flatten()
        .to_cirq_circuit_and_quregs(**registers)
    )
    assert [
        tuple(q.x for q in operation.qubits) for operation in inverse.all_operations()
    ] == [gate.wires for gate in gates]
    for name, qubits in registers.items():
        np.testing.assert_array_equal(outputs[name], qubits)
