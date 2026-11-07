"""Check polynomial wire views against locked split, join and field encodings."""

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.field_polynomial_layout import FieldPolynomialLayout
from flagquantum.compiler.registers.finite_field import gf2_add_constant_ir

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("degree,p,m", [(0, 2, 1), (4, 2, 3), (2, 3, 2)])
def test_polynomial_split_join_matches_locked_wire_reshape(
    degree: int, p: int, m: int
) -> None:
    pytest.importorskip("qualtran")
    import cirq
    import numpy as np
    from qualtran import QGF, QGFPoly
    from qualtran.bloqs.gf_poly_arithmetic.gf_poly_split_and_join import GFPolySplit
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    dtype = QGFPoly(degree, QGF(p, m))
    source_split = GFPolySplit(dtype)
    source_join = source_split.adjoint()
    layout = FieldPolynomialLayout(degree, int(dtype.qgf.num_qubits))
    wires = tuple(range(2 * layout.logical_width, 0, -2))
    quregs = np.array([cirq.LineQubit(wire) for wire in wires])
    manager = cirq.SimpleQubitManager()
    operation, split = source_split.as_cirq_op(manager, reg=quregs)
    assert operation is None
    expected = tuple(tuple(qubit.x for qubit in word) for word in split["reg"])
    assert layout.split(wires) == expected
    operation, joined = source_join.as_cirq_op(manager, reg=split["reg"])
    assert operation is None
    assert layout.join(expected) == tuple(qubit.x for qubit in joined["reg"]) == wires
    for source in (source_split, source_join):
        assert (
            get_cost_value(source, QubitCount()) == layout.logical_width == len(wires)
        )
        assert get_cost_value(source, QECGatesCost()).asdict() == {}


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("p,m", [(2, 3), (3, 2)])
def test_polynomial_basis_keeps_source_padding_and_coefficient_order(
    p: int, m: int
) -> None:
    pytest.importorskip("qualtran")
    from galois import Poly
    from qualtran import QGF, QGFPoly
    from qualtran.bloqs.gf_poly_arithmetic.gf_poly_split_and_join import GFPolySplit

    dtype = QGFPoly(4, QGF(p, m))
    source = GFPolySplit(dtype)
    layout = FieldPolynomialLayout(4, int(dtype.qgf.num_qubits))
    for words in ([0], [1, 2, 3, 4], [7, 0, 2, 0, 5]):
        polynomial = Poly(dtype.qgf.gf_type(words))
        (reference,) = source.call_classically(reg=polynomial)
        packed = int("".join(map(str, dtype.to_bits(polynomial))), 2)
        coefficients = layout.unpack_basis(packed)
        assert coefficients == tuple(map(int, reference))
        assert layout.pack_basis(coefficients) == packed
        assert source.adjoint().call_classically(reg=reference)[0] == polynomial


def test_split_coefficient_operation_and_join_preserve_complex_amplitudes() -> None:
    layout = FieldPolynomialLayout(2, 2)
    # The noncontiguous binding must preserve order, not sort by wire index.
    packed = (4, 0, 5, 2, 3, 1)
    high, middle, low = layout.split(packed)
    assert (high, middle, low) == ((4, 0), (5, 2), (3, 1))
    assert layout.join((high, middle, low)) == packed
    amplitudes = torch.arange(1, 65, dtype=torch.float64)
    state = torch.complex(amplitudes, 2 * amplitudes - 5)
    state /= state.norm()
    operation = FieldOperation(6, gf2_add_constant_ir(middle, 3))
    circuit = Circuit(6, inputs=state, dtype=torch.complex128).append(operation)
    assert [(gate.name, gate.wires) for gate in circuit.to_ir().instructions] == [
        ("x", (5,)),
        ("x", (2,)),
    ]
    expected = state[torch.arange(64) ^ 0b001001]
    torch.testing.assert_close(circuit.state().flatten(), expected, rtol=0, atol=1e-12)


def test_polynomial_layout_rejects_invalid_shapes_and_aliases() -> None:
    with pytest.raises(ValueError, match="nonnegative degree"):
        FieldPolynomialLayout(-1, 2)
    with pytest.raises(ValueError, match="positive coefficient width"):
        FieldPolynomialLayout(2, 0)
    layout = FieldPolynomialLayout(1, 2)
    with pytest.raises(ValueError, match="sum of partition widths"):
        layout.split((0, 1))
    with pytest.raises(ValueError, match="degree plus one"):
        layout.join(((0, 1),))
    with pytest.raises(ValueError, match="distinct"):
        layout.join(((0, 1), (1, 2)))
    with pytest.raises(ValueError, match="fit"):
        layout.pack_basis((4, 0))
