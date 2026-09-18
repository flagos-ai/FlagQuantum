from __future__ import annotations

import pytest
import torch

from flagquantum.algorithms.primitives import (
    append_arbitrary_state,
    arbitrary_state,
    uniform_state,
)
from flagquantum.circuit import Circuit

pytestmark = pytest.mark.unit


def test_uniform_state_is_an_even_superposition() -> None:
    """Every amplitude of the uniform state has magnitude 1/sqrt(N)."""
    for n_wires in (1, 2, 3):
        state = uniform_state(n_wires).state().reshape(-1)
        expected = torch.full_like(state, 1 / (2 ** (n_wires / 2)))
        assert torch.allclose(state, expected, atol=1e-6)


def test_arbitrary_state_reaches_the_requested_amplitudes() -> None:
    """Fidelity with the target amplitude vector is one."""
    amplitudes = torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.complex64)
    target = amplitudes / amplitudes.norm()
    state = arbitrary_state(amplitudes, wires=[0, 1]).state().reshape(-1)
    assert torch.allclose(state, target.to(torch.complex64), atol=1e-5)


def test_arbitrary_state_handles_unequal_magnitudes_and_phases() -> None:
    """Magnitudes and relative phases both come out right."""
    amplitudes = torch.tensor([0.5, 0.5j, -0.5, 0.5], dtype=torch.complex64)
    target = amplitudes / amplitudes.norm()
    state = arbitrary_state(amplitudes, wires=[0, 1]).state().reshape(-1)
    overlap = torch.abs(torch.vdot(target, state)) ** 2
    assert overlap.item() == pytest.approx(1.0, abs=1e-5)


def test_arbitrary_state_is_exact_to_float32_precision() -> None:
    """A random complex target is reached on three wires, not just the two-qubit case."""
    generator = torch.Generator().manual_seed(11)
    amplitudes = torch.complex(
        torch.rand(8, generator=generator), torch.rand(8, generator=generator)
    ).to(torch.complex64)
    target = amplitudes / amplitudes.norm()
    state = arbitrary_state(amplitudes, wires=[0, 1, 2]).state().reshape(-1)
    overlap = float(torch.abs(torch.vdot(target, state)) ** 2)
    assert overlap == pytest.approx(1.0, abs=1e-5)


def test_amplitudes_are_validated() -> None:
    """A wrong length, a non-power-of-two length, and an all-zero vector all raise."""
    with pytest.raises(ValueError):
        arbitrary_state(torch.ones(3, dtype=torch.complex64), wires=[0])
    with pytest.raises(ValueError):
        arbitrary_state(torch.ones(4, dtype=torch.complex64), wires=[0, 1, 2])
    with pytest.raises(ValueError):
        arbitrary_state(torch.zeros(4, dtype=torch.complex64), wires=[0, 1])


def test_repeated_wires_are_refused() -> None:
    """A repeated wire would drive the same qubit twice."""
    with pytest.raises(ValueError):
        arbitrary_state(torch.ones(4, dtype=torch.complex64), wires=[0, 0])


def test_wires_default_to_the_leading_register() -> None:
    """With no wires given, the state occupies wires 0..n-1."""
    circuit = arbitrary_state(torch.ones(4, dtype=torch.complex64))
    assert circuit.n_qubits == 2


def test_append_arbitrary_state_extends_an_existing_circuit_in_place() -> None:
    """The append form composes onto a circuit that is already carrying gates."""
    circuit = Circuit(3)
    circuit.gate("x", 2)
    append_arbitrary_state(
        circuit, torch.tensor([0.6, 0.8j], dtype=torch.complex64), [0]
    )
    state = circuit.state().reshape(-1)
    expected = torch.zeros(8, dtype=torch.complex64)
    expected[1] = 0.6
    expected[5] = 0.8j
    # The preparation fixes the target up to the one global phase the joint solve leaves
    # free, so this compares fidelities rather than amplitudes entrywise.
    overlap = float(torch.abs(torch.vdot(expected, state)) ** 2)
    assert overlap == pytest.approx(1.0, abs=1e-5)


def test_uniform_state_rejects_a_non_positive_wire_count() -> None:
    """A zero-wire register has no state to prepare."""
    with pytest.raises(ValueError):
        uniform_state(0)
