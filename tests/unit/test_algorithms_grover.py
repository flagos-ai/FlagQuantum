from __future__ import annotations

import pytest

from flagquantum.algorithms.grover import (
    grover_circuit,
    optimal_iterations,
    run_grover,
)
from flagquantum.algorithms.primitives.oracle import marked_states

pytestmark = pytest.mark.unit


def test_optimal_iterations_matches_the_closed_form() -> None:
    """The count is the rounded angle formula, and never negative."""
    assert optimal_iterations(2, 1) == 1
    assert optimal_iterations(4, 1) == 3
    assert optimal_iterations(3, 1) == 2
    assert optimal_iterations(4, 0) == 0


def test_optimal_iterations_is_exact_at_half_marked() -> None:
    """Exactly half the register marked gives one iteration, not zero.

    The quotient is 0.9999999999999999 in floating point here, so a bare floor returns the
    wrong count and disagrees with the docstring.
    """
    for n_wires in (1, 2, 3, 4, 5):
        assert optimal_iterations(n_wires, 2 ** (n_wires - 1)) == 1, n_wires


def test_optimal_iterations_validates_its_arguments() -> None:
    """A zero-width register, a negative count and an over-large count are all refused."""
    with pytest.raises(ValueError):
        optimal_iterations(0, 1)
    with pytest.raises(ValueError):
        optimal_iterations(2, -1)
    with pytest.raises(ValueError):
        optimal_iterations(2, 5)


def test_grover_circuit_uses_exactly_the_evaluation_register() -> None:
    """The circuit carries the evaluation register and nothing else."""
    for n_wires in (1, 2, 3):
        circuit = grover_circuit(lambda value: value == 1, n_wires)
        assert circuit.n_qubits == n_wires, n_wires


def test_grover_circuit_refuses_more_than_three_wires() -> None:
    """Above three wires the phase oracle would need an ancilla the register does not have."""
    with pytest.raises(ValueError):
        grover_circuit(lambda value: value == 1, 4)


def test_grover_circuit_accepts_an_explicit_iteration_count() -> None:
    """Zero iterations leaves the register in the uniform superposition."""
    n_wires = 2
    circuit = grover_circuit(lambda value: value == 1, n_wires, iterations=0)
    probabilities = circuit.state().reshape(-1).abs() ** 2
    assert float(probabilities[0].item()) == pytest.approx(0.25, abs=1e-5)


def test_grover_finds_the_single_marked_state() -> None:
    """One marked state out of four is recovered with near-certainty."""
    result = run_grover(lambda value: value == 2, 2, shots=2048, seed=0)
    assert result.candidates[0] == 2
    assert result.success_probability > 0.9


def test_grover_amplifies_three_wires_and_two_marked_states() -> None:
    """The case Grover actually improves: two marked states out of eight, not the resonance.

    Two marked states out of four sits exactly where amplification does nothing, so the
    distribution stays uniform and the leading candidates are decided by sampling noise.
    """
    result = run_grover(lambda value: value in (1, 3), 3, shots=4096, seed=0)
    assert set(result.candidates[:2]) == {1, 3}
    assert result.success_probability > 0.5


def test_grover_reports_an_empty_predicate() -> None:
    """No marked state yields no candidates rather than a failure."""
    result = run_grover(lambda value: False, 2, shots=64, seed=0)
    assert result.candidates == ()
    assert result.success_probability == 0.0


def test_grover_beats_uniform_sampling() -> None:
    """The marked state is far more likely than its uniform share."""
    result = run_grover(lambda value: value == 5, 3, shots=4096, seed=0)
    total = sum(result.counts.values())
    observed = result.counts[format(5, "03b")] / total
    assert observed > 4 / 8


def test_run_grover_records_its_iteration_count() -> None:
    """The result reports how many rounds were applied."""
    result = run_grover(lambda value: value == 2, 2, shots=256, seed=0)
    assert result.iterations == optimal_iterations(
        2, len(marked_states(lambda value: value == 2, 2))
    )
    assert result.counts
    assert sum(result.counts.values()) == 256


def test_run_grover_is_reproducible() -> None:
    """The same seed gives the same counts."""
    first = run_grover(lambda value: value == 2, 2, shots=512, seed=7)
    second = run_grover(lambda value: value == 2, 2, shots=512, seed=7)
    assert first.counts == second.counts


def test_run_grover_validates_its_arguments() -> None:
    """A zero-width register and a non-positive shot count are both refused."""
    with pytest.raises(ValueError):
        run_grover(lambda value: True, 0)
    with pytest.raises(ValueError):
        run_grover(lambda value: True, 2, shots=0)
