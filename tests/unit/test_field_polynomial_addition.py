"""Verify the pinned polynomial XOR stream, resources, and coherent action."""

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.field_polynomial_addition import (
    gf2_polynomial_add_ir,
    source_gf2_polynomial_add_resource_formula,
)
from flagquantum.compiler.registers.field_polynomial_layout import FieldPolynomialLayout

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("degree,width", [(0, 1), (2, 2), (4, 3)])
def test_polynomial_add_gate_stream_and_source_resources(
    degree: int, width: int
) -> None:
    pytest.importorskip("qualtran")
    import cirq
    import numpy as np
    from qualtran import QGF, QGFPoly
    from qualtran.bloqs.basic_gates import CNOT
    from qualtran.bloqs.gf_poly_arithmetic.gf2_poly_add import GF2PolyAdd
    from qualtran.cirq_interop import BloqAsCirqGate
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    layout = FieldPolynomialLayout(degree, width)
    n = layout.logical_width
    left = tuple(reversed(range(0, 2 * n, 2)))
    right = tuple(range(1, 2 * n, 2))
    source = GF2PolyAdd(QGFPoly(degree, QGF(2, width)))
    lowered, outputs = (
        source.decompose_bloq()
        .flatten()
        .to_cirq_circuit_and_quregs(
            f_x=np.array([cirq.LineQubit(wire) for wire in left]),
            g_x=np.array([cirq.LineQubit(wire) for wire in right]),
        )
    )
    assert [q.x for q in outputs["f_x"]] == list(left)
    assert [q.x for q in outputs["g_x"]] == list(right)
    reference = tuple(lowered.all_operations())
    assert all(op.gate in {cirq.CNOT, BloqAsCirqGate(CNOT())} for op in reference)
    actual = gf2_polynomial_add_ir(left, right, layout)
    assert [(gate.name, gate.wires) for gate in actual] == [
        ("cx", tuple(q.x for q in gate.qubits)) for gate in reference
    ]
    resources = source_gf2_polynomial_add_resource_formula(degree, width)
    gates = get_cost_value(source, QECGatesCost())
    assert len(actual) == resources.clifford == gates.clifford == n
    assert resources.t == gates.total_t_count() == 0
    assert resources.measurement == gates.measurement == 0
    assert resources.logical_qubits == get_cost_value(source, QubitCount()) == 2 * n


@pytest.mark.ftqc_reference
def test_symbolic_polynomial_add_cost_matches_locked_formula() -> None:
    pytest.importorskip("qualtran")
    import sympy
    from qualtran import QGF, QGFPoly
    from qualtran.bloqs.gf_poly_arithmetic.gf2_poly_add import GF2PolyAdd
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    degree, width = sympy.symbols("n m", positive=True, integer=True)
    source = GF2PolyAdd(QGFPoly(degree, QGF(2, width)))
    resources = source_gf2_polynomial_add_resource_formula(degree, width)
    assert (
        sympy.simplify(
            resources.clifford - get_cost_value(source, QECGatesCost()).clifford
        )
        == 0
    )
    assert (
        sympy.simplify(resources.logical_qubits - get_cost_value(source, QubitCount()))
        == 0
    )


def test_polynomial_addition_preserves_complex_amplitudes_and_is_self_inverse() -> None:
    layout = FieldPolynomialLayout(2, 2)
    width = layout.logical_width
    qubits = 2 * width + 1
    left = tuple(range(width))
    right = tuple(range(width, 2 * width))
    basis = torch.arange(1 << qubits)
    state = torch.complex((basis + 1).double(), (3 * basis - 2).double())
    state /= state.norm()
    circuit = Circuit(qubits, inputs=state, dtype=torch.complex128)
    addition = FieldOperation(qubits, gf2_polynomial_add_ir(left, right, layout))
    expected = state[basis ^ ((basis >> (width + 1)) << 1)]
    added = circuit.append(addition)
    torch.testing.assert_close(added.state().flatten(), expected, rtol=0, atol=1e-12)
    torch.testing.assert_close(
        added.append(addition).state().flatten(), state, rtol=0, atol=1e-12
    )


def test_polynomial_addition_rejects_cross_coefficient_aliases() -> None:
    layout = FieldPolynomialLayout(1, 2)
    with pytest.raises(ValueError, match="disjoint"):
        gf2_polynomial_add_ir((0, 1, 2, 3), (4, 5, 0, 7), layout)
