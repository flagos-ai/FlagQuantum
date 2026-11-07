"""Check standalone Frobenius maps, including pinned source decomposition defects."""

import pytest
import torch

from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.field_linear import (
    gf2_frobenius,
    source_gf2_square_resource_bound,
)
from tests.unit.test_field_linear import (
    LinearFieldOperation,
    _assert_source,
    _source_type,
)
from tests.unit.test_field_multiplication import assert_gate_streams, reference_gates

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "width,modulus,count", [(1, 3, 1), (2, 7, 1), (3, 11, 2), (4, 19, 4), (8, 283, 2)]
)
def test_standalone_square_matches_locked_forward(
    width: int, modulus: int, count: int
) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.gf_arithmetic.gf2_square import GF2Square

    plan = gf2_frobenius(tuple(range(width)), count, modulus=modulus)
    _assert_source(plan, GF2Square(_source_type(width, modulus), k=count), "x")


@pytest.mark.ftqc_reference
def test_non_power_of_two_uses_the_correct_matrix_with_locked_plu() -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("qualtran")
    from qualtran.bloqs.gf_arithmetic.gf2_multiplication import SynthesizeLRCircuit
    from qualtran.bloqs.gf_arithmetic.gf2_square import GF2Square
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    width, modulus, count = 4, 19, 3
    dtype = _source_type(width, modulus)
    columns = [int(dtype.gf_type(1 << i) ** (2**count)) for i in range(width)]
    matrix = np.array([[(word >> i) & 1 for word in columns] for i in range(width)])
    pinned = GF2Square(dtype, k=count)
    assert np.array_equal(pinned.squaring_matrix, GF2Square(dtype, k=2).squaring_matrix)
    assert not np.array_equal(pinned.squaring_matrix, matrix)
    assert np.array_equal(
        GF2Square(dtype, k=0).squaring_matrix, GF2Square(dtype, k=1).squaring_matrix
    )

    plan = gf2_frobenius(tuple(range(width)), count, modulus=modulus)
    source = SynthesizeLRCircuit(matrix)
    gates, outputs = reference_gates(source, {"q": tuple(reversed(range(width)))}, ())
    assert_gate_streams([(gate.name, gate.wires) for gate in plan.instructions], gates)
    assert plan.outputs == (outputs["q"][::-1],)
    assert len(plan.instructions) == get_cost_value(source, QECGatesCost()).clifford


@pytest.mark.ftqc_reference
def test_locked_adjoint_decomposes_to_the_forward_stream() -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.gf_arithmetic.gf2_square import GF2Square

    wires = (0, 1, 2, 3)
    source = GF2Square(_source_type(4, 19))
    forward, forward_outputs = reference_gates(source, {"x": wires}, ())
    adjoint, adjoint_outputs = reference_gates(source.adjoint(), {"x": wires}, ())
    assert adjoint == forward and adjoint_outputs == forward_outputs
    native = gf2_frobenius(wires, modulus=19, adjoint=True)
    assert [(gate.name, gate.wires) for gate in native.instructions] != adjoint


@pytest.mark.ftqc_reference
def test_symbolic_square_resource_bound() -> None:
    sympy = pytest.importorskip("sympy")
    pytest.importorskip("qualtran")
    from qualtran.bloqs.gf_arithmetic.gf2_square import GF2Square
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    width = sympy.Symbol("m", positive=True, integer=True)
    source = GF2Square(width)
    bound = source_gf2_square_resource_bound(width)
    assert (
        sympy.simplify(bound.clifford - get_cost_value(source, QECGatesCost()).clifford)
        == 0
    )
    assert (
        bound.logical_qubits
        == get_cost_value(source, QubitCount())
        == sympy.Max(2, width)
    )
    assert bound.measurement == bound.t == 0


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "width,modulus,count",
    [(2, 7, 0), (3, 11, 1), (4, 19, 3), (5, 37, 7), (8, 283, 10**9 + 3)],
)
def test_frobenius_complex_amplitudes_and_adjoint(
    width: int, modulus: int, count: int
) -> None:
    field = _source_type(width, modulus).gf_type
    indices = torch.arange(1 << width, dtype=torch.float64)
    state = torch.complex(indices + 1, 3 * indices - 2)
    state /= state.norm()
    expected = torch.empty_like(state)
    for value in range(1 << width):
        expected[int(field(value) ** (2 ** (count % width)))] = state[value]
    wires = tuple(range(width))
    forward = gf2_frobenius(wires, count, modulus=modulus)
    adjoint = gf2_frobenius(wires, count, modulus=modulus, adjoint=True)
    assert forward.scratch == adjoint.scratch == ()
    assert len(forward.instructions) == len(adjoint.instructions)
    assert all(
        gate.name == "cx" for gate in forward.instructions + adjoint.instructions
    )
    circuit = Circuit(width, inputs=state, dtype=torch.complex128)
    transformed = circuit.append(LinearFieldOperation(forward))
    torch.testing.assert_close(
        transformed.state().flatten(), expected, rtol=0, atol=1e-12
    )
    restored = transformed.append(LinearFieldOperation(adjoint))
    torch.testing.assert_close(restored.state().flatten(), state, rtol=0, atol=1e-12)


def test_frobenius_rejects_invalid_count() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        gf2_frobenius((0, 1), -1, modulus=7)
