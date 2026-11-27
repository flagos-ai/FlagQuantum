"""Compare seeded Haar matrices, source tensor indices and append state."""

from tests._reference_dependencies import unavailable_reference

try:
    import numpy as np
except ModuleNotFoundError as _reference_import_error:
    unavailable_reference(_reference_import_error)
import pytest
import torch

from flagquantum import Circuit
from tests.unit.matrix_register_fixture import MatrixRegisterFixture

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("width", [1, 2, 3])
def test_matrix_seeded_draws_adjoint_and_native_append(width: int) -> None:
    pytest.importorskip("qualtran")
    try:
        from cirq import unitary
    except ModuleNotFoundError as _reference_import_error:
        unavailable_reference(_reference_import_error)
    from qualtran.bloqs.for_testing.matrix_gate import MatrixGate
    from qualtran.resource_counting._costing import get_cost_value
    from qualtran.resource_counting._qubit_counts import QubitCount

    source_rng, native_rng = np.random.RandomState(42), np.random.RandomState(42)
    size = 1 << width
    # The leading logical operand is deliberately bound to the highest wire.
    wires = tuple(range(width - 1, -1, -1))
    values = torch.arange(1, size + 1, dtype=torch.float64)
    logical = torch.complex(values, 2 - values)
    logical /= logical.norm()
    axes = tuple(reversed(range(width)))
    physical = logical.reshape((2,) * width).permute(axes).flatten()
    for _ in range(5):
        source = MatrixGate.random(width, random_state=source_rng)
        fixture = MatrixRegisterFixture.random(width, random_state=native_rng)
        assert [(r.name, r.bitsize, r.side.name) for r in fixture.signature] == [
            (r.name, r.bitsize, r.side.name) for r in source.signature
        ]
        np.testing.assert_array_equal(fixture.unitary(), unitary(source))
        np.testing.assert_array_equal(fixture.adjoint().unitary(), unitary(source**-1))
        assert fixture.adjoint().adjoint() == fixture
        assert get_cost_value(source, QubitCount()) == width
        # Keep the source's runtime string labels; its annotation lists Connections.
        (raw,) = getattr(source, "my_tensors")({"q": "in"}, {"q": "out"})
        assert raw.inds == tuple(
            (label, i) for label in ("out", "in") for i in range(width)
        )
        torch.testing.assert_close(
            fixture.tensor(), torch.as_tensor(raw.data.copy()), atol=0, rtol=0
        )
        source_matrix = torch.as_tensor(
            source.tensor_contract().copy(), dtype=logical.dtype
        )
        expected = (
            (source_matrix @ logical).reshape((2,) * width).permute(axes).flatten()
        )
        circuit = Circuit(width, inputs=physical, dtype=physical.dtype)
        circuit.append(fixture.make_on(*wires))
        instructions = circuit.to_ir().instructions
        assert len(instructions) == 1 and instructions[0].name == "matrix_fixture"
        assert instructions[0].wires == wires
        torch.testing.assert_close(
            circuit.state().flatten(), expected, atol=2e-14, rtol=0
        )
        circuit.append(fixture.adjoint().make_on(*wires))
        torch.testing.assert_close(
            circuit.state().flatten(), physical, atol=2e-14, rtol=0
        )
    assert native_rng.randint(1 << 20) == source_rng.randint(1 << 20)


def test_matrix_validation_and_integer_seed() -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.for_testing.matrix_gate import MatrixGate

    rows = [[0j, 1j], [1j, 0j]]
    native, source = MatrixRegisterFixture(1, rows), MatrixGate(1, rows)
    rows[0][0] = 2j
    assert native.matrix == source.matrix
    assert hash(native) == hash(MatrixRegisterFixture(1, [[0j, 1j], [1j, 0j]]))
    for bad in (((1, 0), (0, 2)), ((1, 0, 0), (0, 0, 1))):
        with pytest.raises(AssertionError):
            MatrixRegisterFixture(1, bad)
        with pytest.raises(AssertionError):
            MatrixGate(1, bad)
    with pytest.raises(ValueError, match="bound wires"):
        native.make_on()
    np.testing.assert_array_equal(
        MatrixRegisterFixture.random(2, random_state=1234).unitary(),
        MatrixGate.random(2, random_state=1234).matrix,
    )


def test_empty_signature_scalar_action_on_coherent_ambient_circuit() -> None:
    pytest.importorskip("qualtran")
    from qualtran._infra.bloq import DecomposeNotImplementedError
    from qualtran.bloqs.for_testing.matrix_gate import MatrixGate
    from qualtran.resource_counting._bloq_counts import QECGatesCost
    from qualtran.resource_counting._costing import get_cost_value
    from qualtran.resource_counting._qubit_counts import QubitCount

    coefficient = np.exp(0.51j)
    source = MatrixGate(0, ((coefficient,),))
    fixture = MatrixRegisterFixture(0, ((coefficient,),))
    native_signature = tuple(fixture.signature)
    source_signature: tuple[object, ...] = tuple(source.signature)
    assert native_signature == source_signature == ()
    assert get_cost_value(source, QubitCount()) == 0
    assert fixture.tensor().shape == ()
    with pytest.raises(DecomposeNotImplementedError):
        get_cost_value(source, QECGatesCost())
    circuit = Circuit(2, dtype=torch.complex128).h(0).cx(0, 1).s(1)
    before = circuit.state().clone()
    circuit.append(fixture.make_on())
    torch.testing.assert_close(
        circuit.state(), before * complex(source.tensor_contract()), atol=1e-14, rtol=0
    )
    circuit.append(fixture.adjoint().make_on())
    torch.testing.assert_close(circuit.state(), before, atol=1e-14, rtol=0)
