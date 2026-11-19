"""Original-size cat processor, full density oracle and all finite-sample outcomes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import torch

if TYPE_CHECKING:
    from examples.synthesis.community.basic_examples.logical_qubits.logical_qubits_by_alice_and_bob.overview.workflow import (
        Result,
    )

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def completed() -> Result:
    np = pytest.importorskip("numpy")
    from examples.synthesis.community.basic_examples.logical_qubits.logical_qubits_by_alice_and_bob.overview import (
        workflow,
    )

    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()
    result = workflow.run()
    assert torch.equal(torch_state, torch.random.get_rng_state())
    current = np.random.get_state()
    assert numpy_state[0] == current[0] and numpy_state[2:] == current[2:]
    assert np.array_equal(numpy_state[1], current[1])
    return result


def test_complete_original_grid_and_actual1070000_draws(completed: Result) -> None:
    assert len(completed.grid) == 105
    assert len({point.parameters for point in completed.grid}) == 105
    assert completed.full_state_preparation_error < 1e-12
    assert completed.noiseless.counts == (10_000, 0)
    all_points = (completed.noiseless, completed.configured, *completed.grid)
    assert sum(sum(point.counts) for point in all_points) == 1_070_000
    assert all(sum(point.counts) == 10_000 for point in all_points)
    assert all(0 <= point.probability_one <= 1 for point in all_points)
    assert all(
        abs(point.observed_error - point.probability_one) < 0.025
        for point in all_points
    )
    assert all(point.physical_instructions == 13_339 for point in completed.grid)
    assert all(
        point.clocks == 6144 and point.noise_applications == 43_008
        for point in completed.grid
    )
    assert all(point.merged_channels == 301 for point in completed.grid)
    assert completed.configured.parameters is not None
    assert completed.configured.parameters.distance == 31
    assert completed.configured.parameters.photons == 19
    assert completed.configured.parameters.kappa_2 == 10_000


def test_full_unmerged_original_clock_density_oracle() -> None:
    np = pytest.importorskip("numpy")
    from examples.synthesis.community.basic_examples.logical_qubits.logical_qubits_by_alice_and_bob.overview import (
        channels,
        quantum,
        reference,
    )
    from examples.synthesis.community.basic_examples.logical_qubits.logical_qubits_by_alice_and_bob.overview.noise import (
        Parameters,
    )

    program, _ = quantum.synthesize(quantum.swap_test(), 2)
    physical = Parameters(5, 7, 1e6)
    actual = channels.execute(program, physical)
    expected = reference.direct(program, physical)
    assert actual.density.shape == (128, 128)
    np.testing.assert_allclose(actual.density.numpy(), expected, atol=1e-10, rtol=0)
    assert abs(complex(actual.density.trace()) - 1) < 1e-10
    assert float(torch.linalg.eigvalsh(actual.density).min()) > -1e-10


def test_source_rx_study_reports_actual_improvement(completed: Result) -> None:
    first, second = completed.rotation_examples
    assert (first.iterations, second.iterations) == (2, 3)
    assert second.gates > first.gates
    assert second.depth > first.depth
    assert 0 < second.operator_error < first.operator_error < 0.03


def test_quaternion_words_and_sub_ulp_commutators() -> None:
    np = pytest.importorskip("numpy")
    from examples.synthesis.community.basic_examples.logical_qubits.logical_qubits_by_alice_and_bob.overview import (
        synthesis,
    )

    h = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
    t = np.diag([1, np.exp(1j * np.pi / 4)])
    gates = dict(h=h, t=t, tdg=t.conj().T)
    for angle in (0.1, 0.37, -0.83, 1e-8):
        target = np.array([np.cos(angle / 2), np.sin(angle / 2), 0.0, 0.0])
        a, b = synthesis.balanced(target)
        commutator = synthesis.product(
            synthesis.product(synthesis.product(a, b), synthesis.inverse(a)),
            synthesis.inverse(b),
        )
        np.testing.assert_allclose(commutator, target, atol=1e-12, rtol=0)
        fitted = synthesis.approximate(target, 3)
        matrix = np.eye(2, dtype=np.complex128)
        for name in fitted.word:
            matrix = gates[name] @ matrix
        expected = np.array(
            [
                [np.cos(angle / 2), -1j * np.sin(angle / 2)],
                [-1j * np.sin(angle / 2), np.cos(angle / 2)],
            ]
        )
        phase = np.angle(np.vdot(expected, matrix))
        error = float(np.linalg.norm(matrix * np.exp(-1j * phase) - expected, 2))
        assert error == pytest.approx(fitted.error, abs=1e-10)
        assert np.isfinite(fitted.quaternion).all()
    with pytest.raises(ValueError, match="zero to three"):
        synthesis.approximate(target, 4)


def test_equivalent_words_ignore_machine_precision_overlap_ties() -> None:
    np = pytest.importorskip("numpy")
    from examples.synthesis.community.basic_examples.logical_qubits.logical_qubits_by_alice_and_bob.overview.synthesis import (
        _nearest_index,
    )

    for perturbation in (-1e-14, 0.0, 1e-14):
        assert _nearest_index(np.array([0.5, -(0.5 + perturbation)])) == 0
    assert _nearest_index(np.array([0.5, -0.50001])) == 1


def test_physical_cat_saturation_and_invalid_parameters() -> None:
    pytest.importorskip("numpy")
    from examples.synthesis.community.basic_examples.logical_qubits.logical_qubits_by_alice_and_bob.overview.noise import (
        Parameters,
    )

    probabilities = Parameters(31, 19, 1e4).pauli_probabilities
    assert min(probabilities) >= 0 and sum(probabilities) == pytest.approx(1)
    assert probabilities[2] + probabilities[3] == pytest.approx(0.5)
    for distance, photons, loss in (
        (2, 19, 1e6),
        (3, 3, 1e6),
        (3, 4, 0),
        (3, float("nan"), 1e6),
    ):
        with pytest.raises(ValueError):
            Parameters(distance, photons, loss)


def test_invalid_preparation_is_rejected_before_mutation() -> None:
    pytest.importorskip("numpy")
    import flagquantum as fq
    from examples.synthesis.community.basic_examples.logical_qubits.logical_qubits_by_alice_and_bob.overview import (
        quantum,
    )

    program = fq.Circuit(7, dtype=torch.complex128)
    before = program.to_ir()
    with pytest.raises(ValueError, match="distinct"):
        quantum.prepare(program, (1, 1, 2), quantum.amplitudes())
    with pytest.raises(ValueError, match="normalized"):
        quantum.prepare(program, (1, 2, 3), (float("nan"),) * 8)
    assert program.to_ir() == before


def test_all_original_stages_are_unexecuted_native_cells() -> None:
    base = (
        Path(__file__).resolve().parents[2]
        / "examples/synthesis/community/basic_examples/logical_qubits/logical_qubits_by_alice_and_bob/overview"
    )
    notebook = json.loads((base / "walkthrough.ipynb").read_text())
    proof = json.loads((base / "provenance.json").read_text())
    code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(code) == len(proof["reviewed_code_cells"]) == 17
    assert all(cell["execution_count"] is None and not cell["outputs"] for cell in code)
