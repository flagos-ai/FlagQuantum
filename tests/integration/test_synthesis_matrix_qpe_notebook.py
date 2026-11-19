"""Full matrix-QPE workflow retains both original seven-bit estimation routes."""

import math
from fractions import Fraction
from pathlib import Path

import pytest
import torch

import flagquantum as fq

try:
    from matplotlib.patches import Rectangle

    from examples.synthesis.algorithms.quantum_phase_estimation.qpe_for_matrix.overview import (
        workflow as matrix_qpe,
    )
except ModuleNotFoundError as error:
    if error.name not in {"numpy", "matplotlib"}:
        raise
    pytest.skip(str(error), allow_module_level=True)

pytestmark = pytest.mark.integration


def test_original_data_decomposition_and_schedule() -> None:
    # Reconstruct from Pauli words rather than the production native Observable.
    reconstructed = sum(
        coefficient
        * torch.kron(
            torch.tensor(matrix_qpe.PAULIS[word[0]]),
            torch.tensor(matrix_qpe.PAULIS[word[1]]),
        )
        for word, coefficient in matrix_qpe.PAULI_TERMS
    )
    torch.testing.assert_close(
        reconstructed,
        torch.tensor(matrix_qpe.MATRIX, dtype=torch.complex128),
        atol=1e-15,
        rtol=0,
    )
    assert len(matrix_qpe.PAULI_TERMS) == 10
    assert pytest.approx(2.34912912, abs=1e-14) == matrix_qpe.NORMALIZATION
    assert matrix_qpe.PHASE_BITS == 7
    assert matrix_qpe.REPETITIONS == (2, 3, 5, 7, 11, 16, 23)
    assert pytest.approx(2 * 2.34912912 / 128) == matrix_qpe.ENERGY_RESOLUTION
    assert matrix_qpe.EIGENVALUES.tolist() == pytest.approx(
        torch.linalg.eigvalsh(torch.tensor(matrix_qpe.MATRIX)).tolist(), abs=1e-14
    )


@pytest.mark.parametrize("route", ["exact", "approximate"])
@pytest.mark.parametrize("variant", [0, 1, 2])
def test_every_complex_amplitude_signed_readout_and_retained_environment(
    route: matrix_qpe.Route, variant: int
) -> None:
    values = (1.0, -2.0, 3.0, 4.0) if variant == 1 else (-2.0, 1.0, 4.0, -3.0)
    amplitudes = (
        matrix_qpe.INITIAL_AMPLITUDES
        if variant == 0
        else tuple(v / math.hypot(*values) for v in values)
    )
    program = matrix_qpe.build_program(route, amplitudes)
    result = fq.run(program)
    expected = (
        fq.run(matrix_qpe.reference_circuit(route, amplitudes)).statevector().flatten()
    )
    torch.testing.assert_close(
        result.statevector().flatten(), expected, atol=2e-11, rtol=0
    )
    assert program.circuit.n_wires == 9
    assert program.report["workspace_qubits"] == 0
    assert program.registers == {"phase_result": tuple(range(7))}
    marginal = expected.reshape(128, 4).abs().square().sum(dim=1)
    numeric = dict(program.numeric_probabilities(result, registers=("phase_result",)))
    for word, probability in enumerate(marginal.tolist()):
        phase = Fraction(word if word < 64 else word - 128, 128)
        assert numeric.get((phase,), 0) == pytest.approx(probability, abs=2e-11)
    # Dropped data remains an entangled physical environment, not clean work.
    assert float(expected.reshape(128, 4)[:, 1:].abs().square().sum()) > 0.1


@pytest.mark.parametrize("route,seed", [("exact", 1400), ("approximate", 1401)])
def test_original_10000_shot_histogram_and_energy_intervals(
    route: matrix_qpe.Route, seed: int
) -> None:
    estimate = matrix_qpe.run_estimate(route, seed=seed)
    expected = fq.run(matrix_qpe.reference_circuit(route)).statevector().flatten()
    draws = torch.multinomial(
        expected.abs().square(),
        10000,
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    counts = torch.bincount(draws // 4, minlength=128)
    observed = {int(row.phase * 128) % 128: row.count for row in estimate.outcomes}
    assert observed == {
        word: count for word, count in enumerate(counts.tolist()) if count
    }
    assert sum(observed.values()) == 10000
    assert [row.count for row in estimate.outcomes] == sorted(
        observed.values(), reverse=True
    )
    for row in estimate.outcomes:
        assert row.energy == float(row.phase) * matrix_qpe.NORMALIZATION
        assert row.probability == row.count / 10000
    assert len(estimate.solutions) == len(estimate.intervals) == 2
    for energy, (low, high) in zip(estimate.solutions, estimate.intervals, strict=True):
        assert low == energy - matrix_qpe.ENERGY_RESOLUTION
        assert high == energy + matrix_qpe.ENERGY_RESOLUTION
    assert estimate.maximum_state_error < 2e-11


def test_comparison_plot(tmp_path: Path) -> None:
    exact = matrix_qpe.run_estimate("exact", seed=1400)
    approximate = matrix_qpe.run_estimate("approximate", seed=1401)
    figure = matrix_qpe.plot_estimates(exact, approximate)
    assert len(figure.axes) == 1
    axis = figure.axes[0]
    assert len(axis.patches) == 4 and len(axis.lines) == 6
    for patch in axis.patches:
        assert isinstance(patch, Rectangle)
        assert patch.get_width() == matrix_qpe.ENERGY_RESOLUTION
    for line, result in zip(axis.lines[-2:], (exact, approximate), strict=True):
        torch.testing.assert_close(
            torch.as_tensor(line.get_xdata(orig=False)),
            torch.tensor([row.energy for row in result.outcomes], dtype=torch.float64),
        )
        torch.testing.assert_close(
            torch.as_tensor(line.get_ydata(orig=False)),
            torch.tensor(
                [row.probability for row in result.outcomes], dtype=torch.float64
            ),
        )
    figure.savefig(tmp_path / "matrix-qpe.png")
    assert (tmp_path / "matrix-qpe.png").stat().st_size > 1000


@pytest.mark.parametrize("shots", [0, -1, True, 1, 100001])
def test_shot_refusal(shots: int) -> None:
    with pytest.raises(ValueError, match="shots"):
        matrix_qpe.run_estimate("exact", shots=shots)


@pytest.mark.parametrize("seed", [-1, True, 2**32])
def test_seed_refusal(seed: int) -> None:
    with pytest.raises(ValueError, match="seed"):
        matrix_qpe.run_estimate("exact", seed=seed)
