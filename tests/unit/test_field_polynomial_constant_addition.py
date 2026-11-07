"""Check native polynomial constant addition against locked gate and cost models."""

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.field_polynomial_addition import (
    gf2_polynomial_add_constant_ir,
    source_gf2_polynomial_add_constant_resource_formula,
)
from flagquantum.compiler.registers.field_polynomial_layout import FieldPolynomialLayout

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "degree,width,coefficients",
    [
        (0, 1, (1,)),
        (4, 3, (0,)),
        (4, 3, (1, 2, 3, 4, 5)),
        (4, 3, (3, 0, 7)),
        (1, 2, (0, 0, 3, 2)),
    ],
)
def test_constant_polynomial_gate_order_and_exact_resource_model(
    degree: int, width: int, coefficients: tuple[int, ...]
) -> None:
    pytest.importorskip("qualtran")
    import cirq
    import numpy as np
    from galois import Poly
    from qualtran import QGF, QGFPoly
    from qualtran.bloqs.basic_gates import XGate
    from qualtran.bloqs.gf_poly_arithmetic.gf2_poly_add_k import GF2PolyAddK
    from qualtran.cirq_interop import BloqAsCirqGate
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    layout = FieldPolynomialLayout(degree, width)
    wires = tuple(reversed(range(1, 2 * layout.logical_width, 2)))
    dtype = QGFPoly(degree, QGF(2, width))
    constant = Poly(dtype.qgf.gf_type(list(coefficients)))
    source = GF2PolyAddK(dtype, constant)
    reference, outputs = (
        source.decompose_bloq()
        .flatten()
        .to_cirq_circuit_and_quregs(
            f_x=np.array([cirq.LineQubit(wire) for wire in wires])
        )
    )
    assert tuple(qubit.x for qubit in outputs["f_x"]) == wires
    reference_ops = tuple(reference.all_operations())
    assert all(op.gate in {cirq.X, BloqAsCirqGate(XGate())} for op in reference_ops)
    native = gf2_polynomial_add_constant_ir(wires, coefficients, layout)
    assert [(gate.name, gate.wires) for gate in native] == [
        ("x", tuple(q.x for q in gate.qubits)) for gate in reference_ops
    ]
    expected = get_cost_value(source, QECGatesCost())
    modeled = source_gf2_polynomial_add_constant_resource_formula(
        degree, width, coefficients=coefficients
    )
    assert modeled.clifford == expected.clifford == len(native)
    assert modeled.measurement == expected.measurement == 0
    assert modeled.t == expected.total_t_count() == 0
    assert modeled.logical_qubits == get_cost_value(source, QubitCount()) == len(wires)


@pytest.mark.ftqc_reference
def test_unknown_constant_matches_locked_symbolic_bound() -> None:
    pytest.importorskip("qualtran")
    import sympy
    from galois import Poly
    from qualtran import QGF, QGFPoly
    from qualtran.bloqs.gf_poly_arithmetic.gf2_poly_add_k import GF2PolyAddK
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    n, m = sympy.symbols("n m", positive=True, integer=True)
    source = GF2PolyAddK(QGFPoly(n, QGF(2, m)), Poly([0]))
    modeled = source_gf2_polynomial_add_constant_resource_formula(n, m)
    assert modeled.clifford == modeled.logical_qubits == m * (n + 1)
    assert (
        sympy.simplify(
            modeled.clifford - get_cost_value(source, QECGatesCost()).clifford
        )
        == 0
    )
    assert (
        sympy.simplify(modeled.logical_qubits - get_cost_value(source, QubitCount()))
        == 0
    )
    assert (
        source_gf2_polynomial_add_constant_resource_formula(
            2, 3, coefficients=(0,)
        ).clifford
        == 0
    )


def test_constant_polynomial_keeps_all_complex_amplitudes_and_is_self_inverse() -> None:
    layout = FieldPolynomialLayout(2, 2)
    basis = torch.arange(128)
    state = torch.complex((basis + 1).double(), (3 * basis - 7).double())
    state /= state.norm()
    circuit = Circuit(7, inputs=state, dtype=torch.complex128)
    operation = FieldOperation(
        7, gf2_polynomial_add_constant_ir(tuple(range(6)), (2, 1), layout)
    )
    actual = circuit.append(operation)
    # Coefficients 0, 2, 1 encode 001001; the spectator is the least significant bit.
    torch.testing.assert_close(
        actual.state().flatten(), state[basis ^ 0b0010010], rtol=0, atol=1e-12
    )
    torch.testing.assert_close(
        actual.append(operation).state().flatten(), state, rtol=0, atol=1e-12
    )


@pytest.mark.parametrize("coefficients", [(), (1, 1, 1), (-1,), (4,), (True,)])
def test_constant_polynomial_rejects_invalid_coefficients(
    coefficients: tuple[int, ...],
) -> None:
    layout = FieldPolynomialLayout(1, 2)
    with pytest.raises(ValueError, match="Constant"):
        gf2_polynomial_add_constant_ir((0, 1, 2, 3), coefficients, layout)
    with pytest.raises(ValueError, match="Constant"):
        source_gf2_polynomial_add_constant_resource_formula(
            1, 2, coefficients=coefficients
        )
