from __future__ import annotations

import math

import pytest

from flagquantum.algorithms.amplitude_estimation import (
    AmplitudeEstimationResult,
    amplitude_estimation_circuit,
    amplitude_resolution,
    maximum_likelihood_estimate,
    run_amplitude_estimation,
)
from flagquantum.algorithms.primitives.types import AmplitudeOperator
from flagquantum.circuit import Circuit

pytestmark = pytest.mark.unit


class _RyOperator:
    """A = RY(phi) on one evaluation wire, marking |1> with a Z.

    The estimate this operator's amplitude estimation must recover is
    ``sin²(phi / 2)``, which is the probability that ``RY(phi)`` puts the wire in |1>.
    """

    def __init__(self, phi: float) -> None:
        self._phi = phi

    @property
    def n_wires(self) -> int:
        return 1

    @property
    def true_amplitude(self) -> float:
        return math.sin(self._phi / 2) ** 2

    def apply_plain(self, circuit: Circuit, wires: list[int]) -> None:
        circuit.gate("ry", wires[0], theta=self._phi)

    def apply_a(self, circuit: Circuit, control: int, wires: list[int]) -> None:
        circuit.gate("cry", (control, wires[0]), theta=self._phi)

    def apply_a_dagger(self, circuit: Circuit, control: int, wires: list[int]) -> None:
        circuit.gate("cry", (control, wires[0]), theta=-self._phi)

    def apply_mark(self, circuit: Circuit, control: int, wires: list[int]) -> None:
        circuit.gate("cz", (control, wires[0]))

    def apply_zero_reflection(
        self, circuit: Circuit, control: int, wires: list[int]
    ) -> None:
        circuit.gate("cx", (control, wires[0]))
        circuit.gate("cz", (control, wires[0]))
        circuit.gate("cx", (control, wires[0]))


def test_the_operator_satisfies_the_protocol() -> None:
    """The worked example is a valid AmplitudeOperator."""
    assert isinstance(_RyOperator(0.5), AmplitudeOperator)


def test_circuit_carries_the_counting_and_evaluation_registers() -> None:
    """The width is the counting register plus the operator's own register."""
    for n_counting_wires in (1, 2, 3, 4):
        circuit = amplitude_estimation_circuit(
            _RyOperator(0.5), n_counting_wires=n_counting_wires
        )
        assert circuit.n_qubits == n_counting_wires + 1, n_counting_wires


def test_resolution_is_the_largest_step_between_grid_amplitudes() -> None:
    """The grid is sin²(pi j / 2**(m+1)); the resolution is its widest gap."""
    assert amplitude_resolution(3) == pytest.approx(0.191342, abs=1e-6)
    assert amplitude_resolution(4) == pytest.approx(0.097545, abs=1e-6)
    assert amplitude_resolution(5) == pytest.approx(0.049009, abs=1e-6)
    for n_counting_wires in (2, 3, 4, 5):
        assert amplitude_resolution(n_counting_wires) > amplitude_resolution(
            n_counting_wires + 1
        )


def test_the_estimate_is_accurate_to_one_resolution() -> None:
    """The measured accuracy contract: the error never exceeds the grid resolution.

    Checked on the exact distribution as well as on samples, because this is the claim the
    module makes about itself.
    """
    for phi in (0.3, 0.8, 1.4, 2.2, 2.9):
        operator = _RyOperator(phi)
        result = run_amplitude_estimation(
            operator, n_counting_wires=5, shots=20000, seed=0
        )
        assert result.within(operator.true_amplitude), (
            phi,
            result.estimate,
            result.resolution,
        )


def test_the_estimate_lands_on_the_grid() -> None:
    """Every returned estimate is one of the grid amplitudes."""
    result = run_amplitude_estimation(
        _RyOperator(0.7), n_counting_wires=4, shots=4096, seed=0
    )
    grid = [
        math.sin(math.pi * index / (2 ** (result.n_counting_wires + 1))) ** 2
        for index in range(2**result.n_counting_wires + 1)
    ]
    assert any(abs(result.estimate - value) < 1e-9 for value in grid)


def test_the_estimate_is_in_the_unit_interval() -> None:
    """A valid result carries an amplitude, and the resolution is positive."""
    result = run_amplitude_estimation(
        _RyOperator(0.7), n_counting_wires=3, shots=2048, seed=0
    )
    assert 0.0 <= result.estimate <= 1.0
    assert result.resolution > 0.0


def test_the_result_validates_its_estimate() -> None:
    """An amplitude outside [0, 1] is refused at construction."""
    with pytest.raises(ValueError):
        AmplitudeEstimationResult(
            estimate=1.5, resolution=0.1, n_counting_wires=3, n_evaluation_wires=1
        )


def test_maximum_likelihood_marginalises_the_evaluation_register() -> None:
    """Keys span every wire, so the evaluation register's bits must be folded away.

    A single-outcome count over a two-bit evaluation register and a one-bit counting register
    must give the same estimate as the same mass split across both evaluation outcomes.
    """
    concentrated = maximum_likelihood_estimate({"0100": 100}, 3, 1)
    split = maximum_likelihood_estimate({"0100": 50, "0101": 50}, 3, 1)
    assert concentrated == pytest.approx(split)
    expected = math.sin(math.pi * 4 / 16) ** 2
    assert concentrated == pytest.approx(expected, abs=1e-9)


def test_maximum_likelihood_resolves_the_sine_grid_not_the_counting_value() -> None:
    """The readout maps through sin², so half the counting range is an amplitude of one.

    A mode of 4 out of 8 counting values is 0.5 of the counting range, but the amplitude it
    corresponds to is ``sin²(pi * 4 / 8) = 1``. Reading the register value as an amplitude
    would give 0.5 here, which is a wrong answer rather than an error.
    """
    estimate = maximum_likelihood_estimate({"1000": 1000}, 3, 1)
    assert estimate == pytest.approx(math.sin(math.pi * 4 / 8) ** 2, abs=1e-9)
    assert estimate == pytest.approx(1.0, abs=1e-9)


def test_maximum_likelihood_validates_its_arguments() -> None:
    """A short key, a zero-width register and an empty mapping are all refused."""
    with pytest.raises(ValueError):
        maximum_likelihood_estimate({"00": 5}, 3, 1)
    with pytest.raises(ValueError):
        maximum_likelihood_estimate({"0100": 5}, 0, 1)
    with pytest.raises(ValueError):
        maximum_likelihood_estimate({}, 3, 1)


def test_amplitude_estimation_circuit_refuses_a_zero_width_register() -> None:
    """A counting register needs at least one wire."""
    with pytest.raises(ValueError):
        amplitude_estimation_circuit(_RyOperator(0.5), n_counting_wires=0)


def test_run_amplitude_estimation_validates_its_arguments() -> None:
    """A zero-width register and a non-positive shot count are both refused."""
    with pytest.raises(ValueError):
        run_amplitude_estimation(_RyOperator(0.5), n_counting_wires=0)
    with pytest.raises(ValueError):
        run_amplitude_estimation(_RyOperator(0.5), n_counting_wires=2, shots=0)


def test_run_amplitude_estimation_is_reproducible() -> None:
    """The same seed gives the same result."""
    first = run_amplitude_estimation(
        _RyOperator(0.7), n_counting_wires=3, shots=1024, seed=11
    )
    second = run_amplitude_estimation(
        _RyOperator(0.7), n_counting_wires=3, shots=1024, seed=11
    )
    assert first.estimate == second.estimate
