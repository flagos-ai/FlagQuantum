from __future__ import annotations

import math

import pytest
import torch

from flagquantum.algorithms.data_encoding import (
    amplitude_encode,
    angular_encode,
    append_angular_encode,
)
from flagquantum.circuit import Circuit

pytestmark = pytest.mark.unit


def _amplitudes(circuit: Circuit) -> torch.Tensor:
    """Return a prepared circuit's state vector as a flat complex tensor."""
    return circuit.state().reshape(-1).to(torch.complex64)


def test_amplitude_encode_prepares_the_normalised_vector() -> None:
    """The prepared state is the feature vector scaled to unit norm."""
    circuit = amplitude_encode([0.5, 0.5, 0.5, 0.5])
    assert circuit.n_qubits == 2
    expected = torch.full((4,), 0.5, dtype=torch.complex64)
    assert torch.allclose(_amplitudes(circuit), expected, atol=1e-6)


def test_amplitude_encode_reaches_a_basis_state_exactly() -> None:
    """A one-hot feature vector prepares a computational basis state."""
    circuit = amplitude_encode([0.0, 0.0, 0.0, 3.0])
    state = _amplitudes(circuit)
    assert float(state[3].real) == pytest.approx(1.0, abs=1e-6)
    assert float(state.abs().sum()) == pytest.approx(1.0, abs=1e-6)


def test_amplitude_encode_is_exact_to_float32_precision() -> None:
    """A random complex vector of eight amplitudes comes back on three qubits."""
    generator = torch.Generator().manual_seed(5)
    amplitudes = torch.complex(
        torch.randn(8, generator=generator), torch.randn(8, generator=generator)
    )
    target = amplitudes / amplitudes.norm()
    state = _amplitudes(amplitude_encode(amplitudes, qubits=[0, 1, 2]))
    # Every relative phase is fixed by the preparation and only one global phase is
    # free, so the comparison is on the overlap rather than on the entries.
    overlap = float(torch.abs(torch.vdot(target.to(torch.complex64), state)) ** 2)
    assert overlap == pytest.approx(1.0, abs=1e-6)


def test_amplitude_encode_keeps_signed_amplitudes_relative_to_each_other() -> None:
    """A sign flip between two amplitudes survives as a relative phase."""
    state = _amplitudes(amplitude_encode([0.5, -0.5]))
    ratio = state[1] / state[0]
    assert complex(ratio).real == pytest.approx(-1.0, abs=1e-6)
    assert complex(ratio).imag == pytest.approx(0.0, abs=1e-6)


def test_amplitude_encode_pads_before_it_normalises() -> None:
    """A non-power-of-two vector is padded to the next power of two, then normalised."""
    zero_padded = _amplitudes(amplitude_encode([1.0, 1.0, 1.0]))
    assert zero_padded.numel() == 4
    # The default pad is zero, so only three of the four amplitudes are populated.
    assert torch.allclose(
        zero_padded,
        torch.tensor([1.0, 1.0, 1.0, 0.0], dtype=torch.complex64) / math.sqrt(3),
        atol=1e-6,
    )
    # The pad value takes part in the norm exactly as a feature does, so padding with
    # three lifts the fourth amplitude instead of leaving it at zero.
    assert torch.allclose(
        _amplitudes(amplitude_encode([1.0, 1.0, 1.0], pad=3.0)),
        torch.tensor([1.0, 1.0, 1.0, 3.0], dtype=torch.complex64) / math.sqrt(12),
        atol=1e-6,
    )


def test_amplitude_encode_accepts_a_list_a_tuple_and_a_tensor() -> None:
    """Every sequence form the reader accepts produces the same state.

    ``numpy`` is deliberately absent here: it is not a declared dependency of this
    repository, so ``tests/unit/test_algorithms_hidden_dependency.py`` refuses it
    anywhere in the algorithms surface. ``torch.as_tensor`` would convert an
    ``ndarray`` if one ever arrived, and that is the reader's business rather than
    a promise this package makes.
    """
    reference = _amplitudes(amplitude_encode([1.0, -2.0, 3.0, -4.0]))
    for source in (
        (1.0, -2.0, 3.0, -4.0),
        torch.tensor([1.0, -2.0, 3.0, -4.0]),
        torch.tensor([1, -2, 3, -4]),
    ):
        assert torch.allclose(
            _amplitudes(amplitude_encode(source)), reference, atol=1e-6
        )


def test_amplitude_encode_places_the_state_on_the_requested_wires() -> None:
    """The qubits are the caller's, most significant first."""
    circuit = amplitude_encode([1.0, 0.0, 1.0, 0.0], qubits=[2, 3])
    assert circuit.n_qubits == 4
    state = _amplitudes(circuit)
    # Wire 2 carries the most significant bit of the encoded vector and qubit 3 its
    # least significant one, so the two populated basis states are indexes 0 and 2.
    assert float(state.abs()[0]) == pytest.approx(1 / math.sqrt(2), abs=1e-6)
    assert float(state.abs()[2]) == pytest.approx(1 / math.sqrt(2), abs=1e-6)


def test_amplitude_encode_refuses_a_vector_it_cannot_prepare() -> None:
    """Empty, non-finite, zero-norm, non-1D, single-entry, and string inputs all raise."""
    with pytest.raises(ValueError, match="must not be empty"):
        amplitude_encode([])
    with pytest.raises(ValueError, match="must be finite"):
        amplitude_encode([float("nan"), 1.0])
    with pytest.raises(ValueError, match="zero vector"):
        amplitude_encode([0.0, 0.0])
    with pytest.raises(ValueError, match="one-dimensional"):
        amplitude_encode([[1.0, 1.0]])
    with pytest.raises(ValueError, match="at least two entries once padded, got 1"):
        amplitude_encode([1.0])
    with pytest.raises(TypeError, match="must be a sequence of numbers"):
        amplitude_encode("01")


def test_amplitude_encode_refuses_a_pad_value_that_is_not_a_real_number() -> None:
    """The pad value is an amplitude, so a flag or an imaginary value is refused."""
    with pytest.raises(TypeError, match="got bool"):
        amplitude_encode([1.0, 1.0, 1.0], pad=True)
    # A string and a missing value are refused here, because the message torch
    # raises for its own fill value does not name this parameter.
    with pytest.raises(TypeError, match="pad must be a real number, got str"):
        amplitude_encode([1.0, 1.0, 1.0], pad="1")
    with pytest.raises(TypeError, match="pad must be a real number, got NoneType"):
        amplitude_encode([1.0, 1.0, 1.0], pad=None)
    with pytest.raises(TypeError, match="complex value"):
        amplitude_encode([1.0, 1.0, 1.0], pad=1j)
    with pytest.raises(ValueError, match="pad must be finite"):
        amplitude_encode([1.0, 1.0, 1.0], pad=float("inf"))
    # A complex value with no imaginary part is a real number spelled that way.
    assert _amplitudes(amplitude_encode([1.0, 1.0], pad=0j)).numel() == 2


def test_amplitude_encode_refuses_qubits_that_do_not_match_the_features() -> None:
    """A short qubit list, a repeated qubit, and a negative qubit are all refused."""
    with pytest.raises(ValueError, match="require 2 qubits, got 1"):
        amplitude_encode([1.0, 1.0, 1.0, 1.0], qubits=[0])
    with pytest.raises(ValueError, match="must be distinct and appear once each"):
        amplitude_encode([1.0, 1.0, 1.0, 1.0], qubits=[0, 0])
    with pytest.raises(ValueError, match="qubits must be non-negative, got -1"):
        amplitude_encode([1.0, 1.0, 1.0, 1.0], qubits=[0, -1])
    # A bool is an int in Python, so the type gate has to refuse it explicitly
    # rather than let True become qubit 1.
    with pytest.raises(ValueError, match="qubits must be integers, got True"):
        amplitude_encode([1.0, 1.0, 1.0, 1.0], qubits=[True, False])
    with pytest.raises(ValueError, match=r"qubits must be integers, got 1\.0"):
        amplitude_encode([1.0, 1.0, 1.0, 1.0], qubits=[1.0, 0])


def test_angular_encode_is_a_product_of_one_rotation_per_wire() -> None:
    """Two zero rotations prepare the all-zero state; two pi rotations prepare all-ones."""
    assert _amplitudes(angular_encode([0.0, 0.0]))[0] == pytest.approx(1.0 + 0j)
    flipped = _amplitudes(angular_encode([math.pi, math.pi]))
    assert complex(flipped[3]).real == pytest.approx(1.0, abs=1e-6)


def test_angular_encode_maps_the_axis_name_onto_the_gate() -> None:
    """'X', 'Y', and 'Z' select rx, ry, and rz, case-insensitively."""
    # R_X(pi) on |0> is -i|1>, so the single-qubit state is pure imaginary.
    assert complex(
        _amplitudes(angular_encode([math.pi], rotation="x"))[1]
    ) == pytest.approx(-1j, abs=1e-6)
    assert complex(_amplitudes(angular_encode([math.pi], rotation="Y"))[1]).real == (
        pytest.approx(1.0, abs=1e-6)
    )
    # R_Z(pi) on |0> is a phase on |0> alone, so no amplitude moves to |1>.
    rotated_z = _amplitudes(angular_encode([math.pi], rotation="Z"))
    assert complex(rotated_z[1]) == pytest.approx(0j, abs=1e-6)


def test_angular_encode_refuses_an_axis_it_does_not_have() -> None:
    """An unknown axis, and an axis that is not a string, are refused."""
    with pytest.raises(ValueError, match="unsupported rotation 'W'"):
        angular_encode([1.0], rotation="W")
    with pytest.raises(ValueError, match="rotation must be a string"):
        angular_encode([1.0], rotation=3)


def test_angular_encode_refuses_features_it_cannot_read() -> None:
    """Empty, non-1D, complex, non-finite, and string feature vectors all raise."""
    with pytest.raises(ValueError, match="must not be empty"):
        angular_encode([])
    with pytest.raises(ValueError, match="one-dimensional"):
        angular_encode([[1.0]])
    with pytest.raises(TypeError, match="must be real numbers"):
        angular_encode([1j])
    with pytest.raises(ValueError, match="must be finite"):
        angular_encode([float("inf")])
    with pytest.raises(TypeError, match="must be a sequence of numbers"):
        angular_encode("ab")


def test_angular_encode_carries_the_gradient_back_to_the_features() -> None:
    """The angle is handed to the circuit as a tensor, so autograd reaches it."""
    features = torch.tensor([0.3, 0.7], requires_grad=True)
    state = angular_encode(features).state().reshape(-1)
    state.real.sum().backward()
    assert features.grad is not None
    # The summed real part of the product state is the product of the two qubits'
    # (cos + sin) half-angle factors, so each derivative is the other qubit's factor
    # times its own derivative. This is calculus, not a restatement of the code.
    first, second = (value / 2 for value in (0.3, 0.7))
    assert float(features.grad[0]) == pytest.approx(
        0.5
        * (math.cos(first) - math.sin(first))
        * (math.cos(second) + math.sin(second)),
        abs=1e-5,
    )
    assert float(features.grad[1]) == pytest.approx(
        0.5
        * (math.cos(second) - math.sin(second))
        * (math.cos(first) + math.sin(first)),
        abs=1e-5,
    )


def test_append_angular_encode_extends_an_existing_circuit() -> None:
    """The append form interleaves the encoding with a circuit the caller already holds."""
    circuit = Circuit(2)
    circuit.gate("x", 0)
    append_angular_encode(circuit, [math.pi], [1])
    state = _amplitudes(circuit)
    assert complex(state[1]).real == pytest.approx(-0.0, abs=1e-6)
    assert complex(state[3]).real == pytest.approx(1.0, abs=1e-6)


def test_append_angular_encode_places_the_features_on_the_requested_qubits() -> None:
    """The qubits are the caller's, one per feature, in feature order."""
    circuit = Circuit(3)
    append_angular_encode(circuit, [math.pi], [2])
    state = _amplitudes(circuit)
    # Qubit 2 is the least significant bit of a three-qubit register.
    assert complex(state[1]).real == pytest.approx(1.0, abs=1e-6)


def test_append_angular_encode_emits_one_rotation_per_feature() -> None:
    """Every feature moves its own qubit, so one of two cannot carry the result."""
    circuit = Circuit(2)
    # The first feature is the identity on qubit 0 and the second flips qubit 1.
    append_angular_encode(circuit, [0.0, math.pi], [0, 1])
    state = _amplitudes(circuit)
    assert float(state.abs()[0]) == pytest.approx(0.0, abs=1e-6)
    assert float(state.abs()[1]) == pytest.approx(1.0, abs=1e-6)


def test_append_angular_encode_leaves_other_qubits_alone() -> None:
    """The append form touches the qubits it names and no others."""
    circuit = Circuit(3)
    circuit.gate("x", 1)
    append_angular_encode(circuit, [math.pi], [2])
    state = _amplitudes(circuit)
    # Qubit 2 is the least significant bit, so the populated index is 3.
    assert float(state.abs()[3]) == pytest.approx(1.0, abs=1e-6)


def test_append_angular_encode_refuses_qubits_that_do_not_match_the_features() -> None:
    """A repeated qubit is refused by the append form as well."""
    with pytest.raises(ValueError, match="must be distinct and appear once each"):
        append_angular_encode(Circuit(2), [1.0, 2.0], [0, 0])


def test_an_encoding_composes_with_a_variational_block() -> None:
    """A data-reuploading block is an encoding interleaved with a variational circuit."""
    features = torch.tensor([0.4, 1.1], dtype=torch.float64)
    weights = torch.tensor([0.2, -0.3], dtype=torch.float64)
    circuit = angular_encode(features)
    for qubit in range(2):
        circuit.ry(qubit, theta=weights[qubit])
        circuit.cz(0, 1)
    append_angular_encode(circuit, features * 2.0, [0, 1])
    state = _amplitudes(circuit)
    assert float(state.norm()) == pytest.approx(1.0, abs=1e-6)
