from __future__ import annotations

import math
from collections.abc import Sequence

import pytest
import torch

from flagquantum.algorithms.primitives import (
    ControlledUnitary,
    PhaseEstimationSpec,
    append_phase_estimation,
    phase_estimation_circuit,
)
from flagquantum.circuit import Circuit

pytestmark = pytest.mark.unit


class _PhaseGate:
    """A single-wire operator multiplying the |1> component by exp(2*pi*1j*phase)."""

    def __init__(self, phase: float) -> None:
        self._phase = phase

    @property
    def n_wires(self) -> int:
        return 1

    def apply(self, circuit: Circuit, wires: Sequence[int]) -> None:
        circuit.gate("p", wires[0], theta=2 * math.pi * self._phase)

    def apply_controlled(
        self, circuit: Circuit, control: int, wires: Sequence[int]
    ) -> None:
        circuit.gate("cphase", (control, wires[0]), theta=2 * math.pi * self._phase)

    def apply_power_controlled(
        self, circuit: Circuit, control: int, wires: Sequence[int], power: int
    ) -> None:
        circuit.gate(
            "cphase",
            (control, wires[0]),
            theta=2 * math.pi * self._phase * power,
        )


def _estimate(phase: float, n_counting_wires: int) -> float:
    """Run phase estimation on ``_PhaseGate(phase)`` and read the estimate back."""
    unitary = _PhaseGate(phase)
    circuit = Circuit(n_counting_wires + 1)
    circuit.gate("x", n_counting_wires)
    append_phase_estimation(
        circuit,
        unitary=unitary,
        counting_wires=list(range(n_counting_wires)),
        evaluation_wires=[n_counting_wires],
    )
    spec = PhaseEstimationSpec(n_counting_wires=n_counting_wires, n_evaluation_wires=1)
    generator = torch.Generator().manual_seed(20260918)
    counts = circuit.counts(64, generator=generator, format="bin")[0]
    return spec.phase_from_counts(counts)


def test_the_fake_satisfies_the_protocol() -> None:
    """Conformance is checked at runtime, so a wrong shape is caught here."""
    assert isinstance(_PhaseGate(0.25), ControlledUnitary)


def test_exact_binary_phases_are_recovered_exactly() -> None:
    """Every phase that is a multiple of 1/2**n is read back without error."""
    for phase in (0.25, 0.5, 0.125, 0.75):
        assert _estimate(phase, 3) == pytest.approx(phase)


def test_two_counting_wires_recover_a_quarter() -> None:
    """The smallest case, checked against the measured winning state index."""
    unitary = _PhaseGate(0.25)
    circuit = Circuit(3)
    circuit.gate("x", 2)
    append_phase_estimation(
        circuit, unitary=unitary, counting_wires=[0, 1], evaluation_wires=[2]
    )
    best = int(circuit.probabilities().reshape(-1).argmax().item())
    assert best >> 1 == 1


def test_an_inexact_phase_lands_within_the_resolution() -> None:
    """A phase with no exact binary form is estimated to the register's resolution."""
    n_counting_wires = 8
    estimate = _estimate(1 / 3, n_counting_wires)
    assert estimate == pytest.approx(1 / 3, abs=1.5 / 2**n_counting_wires)


def test_precision_and_success_probability() -> None:
    """The spec's derived quantities follow the counting register."""
    spec = PhaseEstimationSpec(n_counting_wires=4, n_evaluation_wires=2)
    assert spec.precision == pytest.approx(2 * math.pi / 16)
    assert spec.success_probability == pytest.approx(4 / math.pi**2)


def test_circuit_shape() -> None:
    """The circuit carries the counting register and then the operator's wires."""
    circuit = phase_estimation_circuit(unitary=_PhaseGate(0.25), n_counting_wires=4)
    assert circuit.n_qubits == 5


def test_phase_from_counts_rejects_an_empty_mapping() -> None:
    """An empty sample set has no most frequent value to read."""
    spec = PhaseEstimationSpec(n_counting_wires=2, n_evaluation_wires=1)
    with pytest.raises(ValueError):
        spec.phase_from_counts({})


def test_append_phase_estimation_validates_its_wires() -> None:
    """Repeated counting wires, a wire-count mismatch, and an overlap are all refused."""
    unitary = _PhaseGate(0.25)
    with pytest.raises(ValueError):
        append_phase_estimation(
            Circuit(3), unitary=unitary, counting_wires=[0, 0], evaluation_wires=[2]
        )
    with pytest.raises(ValueError):
        append_phase_estimation(
            Circuit(3), unitary=unitary, counting_wires=[0, 1], evaluation_wires=[1, 2]
        )
    with pytest.raises(ValueError):
        append_phase_estimation(
            Circuit(3), unitary=unitary, counting_wires=[0, 2], evaluation_wires=[2]
        )


def test_phase_estimation_circuit_rejects_zero_counting_wires() -> None:
    """A counting register with no wires has no resolution to offer."""
    with pytest.raises(ValueError):
        phase_estimation_circuit(unitary=_PhaseGate(0.25), n_counting_wires=0)
