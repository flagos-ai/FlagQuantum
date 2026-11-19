"""Full-sized source fidelity, physical oracle and honest Poisson diagnostics."""

import json
import math
from pathlib import Path

import pytest
import torch

from examples.synthesis.algorithms.quantum_differential_equations_solvers.discrete_poisson_solver.overview.ansatz import (
    build_program,
    poisson,
)
from examples.synthesis.algorithms.quantum_differential_equations_solvers.discrete_poisson_solver.overview.physics import (
    AMPLITUDES,
    ANGLES,
    EIGENVALUES_X,
    EIGENVALUES_Y,
    MAX_EIG,
    PREFACTOR,
    SOURCE,
    finite_difference_problem,
    walsh_coefficients,
)
from examples.synthesis.algorithms.quantum_differential_equations_solvers.discrete_poisson_solver.overview.workflow import (
    extract_solution,
    verify_program,
)
from flagquantum import synth

pytestmark = pytest.mark.integration


def test_original_thirteen_qubit_absolute_state_and_named_marginal() -> None:
    program = build_program()
    check = verify_program(program)
    assert check.maximum_state_error < 2e-11
    assert check.dirty_workspace_probability < 1e-22
    assert check.named_probability_error < 2e-11
    assert check.resources.n_qubits == 13
    assert program.report["workspace_qubits"] == 0


def test_fresh_source_and_both_complete_walsh_spectra() -> None:
    grid = (torch.arange(8, dtype=torch.float64) + 0.5) / 8
    z = torch.kron(grid, grid) * torch.kron(grid - 1, grid - 1)
    source = torch.tanh(z).square()
    source /= source.norm()
    torch.testing.assert_close(SOURCE, source, atol=1e-15, rtol=0)
    assert len(AMPLITUDES) == 64
    for values in (EIGENVALUES_X, EIGENVALUES_Y):
        coefficients = walsh_coefficients(values)
        reconstructed = torch.tensor(
            [
                math.fsum(
                    value * (-1) ** ((word & mask).bit_count())
                    for mask, value in enumerate(coefficients)
                )
                for word in range(8)
            ],
            dtype=torch.float64,
        )
        torch.testing.assert_close(reconstructed, values, atol=1e-14, rtol=0)
    assert min(EIGENVALUES_X) > 0 and EIGENVALUES_Y[0] == 0


def test_source_integer_word_table_is_not_replaced_with_fractional_inverse() -> None:
    assert len(ANGLES) == 64 and ANGLES[0] == 0
    assert PREFACTOR == 1 / 64 and pytest.approx(math.pi**2) == MAX_EIG
    assert [math.sin(angle / 2) for angle in ANGLES[1:]] == pytest.approx(
        [1 / (64 * word) for word in range(1, 64)]
    )


def test_source_postselection_reports_classical_discrepancy_without_fit() -> None:
    solution = extract_solution(build_program())
    matrix, rhs, classical = finite_difference_problem()
    torch.testing.assert_close(matrix @ classical.flatten(), rhs, atol=1e-14, rtol=0)
    assert solution.quantum.shape == solution.classical.shape == (8, 8)
    assert solution.quantum[4, 4] > 0
    assert solution.relative_distance == pytest.approx(
        float((solution.quantum - classical).norm() / classical.norm())
    )
    assert 0.77 < solution.relative_distance < 0.79
    assert 0.49 < solution.transpose_relative_distance < 0.51
    assert solution.relative_residual > 0.1
    assert 0 < solution.success < 0.001
    assert 0 < solution.retained_phase_leakage < 1e-5


def test_resource_refusal_preserves_original_problem_size() -> None:
    with pytest.raises(ValueError):
        synth.synthesize(
            poisson, classical_inputs={"amplitudes": list(AMPLITUDES)}, max_qubits=12
        )


def test_every_original_code_stage_has_a_native_cell() -> None:
    directory = (
        Path(__file__).resolve().parents[2]
        / "examples/synthesis/algorithms/quantum_differential_equations_solvers/discrete_poisson_solver/overview"
    )
    proof = json.loads((directory / "provenance.json").read_text())
    notebook = json.loads((directory / "walkthrough.ipynb").read_text())
    assert [row["index"] for row in proof["reviewed_code_cells"]] == [
        8,
        11,
        13,
        15,
        18,
        21,
        23,
        25,
        27,
        29,
        32,
        33,
        34,
        36,
        38,
        39,
    ]
    for row in proof["reviewed_code_cells"]:
        assert notebook["cells"][row["native_code_cell"]]["cell_type"] == "code"
