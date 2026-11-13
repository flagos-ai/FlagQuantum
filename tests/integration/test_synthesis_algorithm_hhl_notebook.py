"""Both original HHL routes retain absolute phases, powered evolution and readout."""

import json
import math
from pathlib import Path

import pytest
import torch

from examples.synthesis.algorithms.quantum_linear_solvers.hhl.overview.ansatz import (
    Route,
    build_program,
    exact,
)
from examples.synthesis.algorithms.quantum_linear_solvers.hhl.overview.physics import (
    AMPLITUDES,
    ANGLES,
    MATRIX,
    PAULI_TERMS,
    PAULIS,
    REPETITIONS,
    B,
)
from examples.synthesis.algorithms.quantum_linear_solvers.hhl.overview.workflow import (
    extract_solution,
    verify_program,
)
from flagquantum import synth

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("route", ("exact", "trotter"))
def test_complete_absolute_states_before_and_after_target_conversion(
    route: Route,
) -> None:
    check = verify_program(build_program(route), route)
    assert check.maximum_state_error < 2e-11 and check.converted_state_error < 2e-11
    assert check.dirty_workspace_probability < 1e-22
    assert check.resources.n_qubits == 7
    assert set(check.resources.operation_counts) == {"cx", "u3"}
    assert check.resources.operation_counts["cx"] > 0 and check.resources.depth > 0


def test_original_dense_matrix_is_reconstructed_without_omitting_terms() -> None:
    reconstructed = torch.zeros((4, 4), dtype=torch.complex128)
    for word, coefficient in PAULI_TERMS:
        reconstructed += coefficient * torch.kron(PAULIS[word[0]], PAULIS[word[1]])
    torch.testing.assert_close(
        reconstructed, MATRIX.to(torch.complex128), atol=1e-14, rtol=0
    )
    assert len(PAULI_TERMS) == 10 and PAULI_TERMS[0][0] == "II"
    assert torch.allclose(
        B, torch.tensor([1.0, 2.0, 4.0, 3.0], dtype=torch.float64) / math.sqrt(30)
    )


def test_repetitions_follow_each_source_power_instead_of_repeating_a_fixed_step() -> (
    None
):
    assert REPETITIONS == (4, 8, 16, 24)
    assert all(
        r == 4 * math.ceil(1.8 ** math.log2(1 << bit))
        for bit, r in enumerate(REPETITIONS)
    )
    assert REPETITIONS[-1] != 8 * REPETITIONS[0]


@pytest.mark.parametrize("route", ("exact", "trotter"))
def test_actual_postselection_and_global_phase_diagnostic(route: Route) -> None:
    result = extract_solution(build_program(route))
    torch.testing.assert_close(result.classical, torch.linalg.solve(MATRIX, B))
    expected_phase = torch.angle(
        torch.vdot(MATRIX.to(torch.complex128) @ result.quantum, B.to(torch.complex128))
    )
    assert result.global_phase == pytest.approx(float(expected_phase), abs=1e-14)
    torch.testing.assert_close(
        result.corrected, (torch.exp(1j * expected_phase) * result.quantum).real
    )
    assert (
        result.relative_distance < 0.1
        and result.fidelity > 0.999
        and result.complex_fidelity > 0.99
    )
    assert 0.09 < result.success < 0.11 and 1e-5 < result.retained_phase_leakage < 0.01
    if route == "trotter":
        assert not torch.allclose(result.corrected, result.quantum.abs())
    else:
        assert torch.allclose(result.corrected, result.quantum.abs())


def test_reciprocal_zero_entry_and_original_rotation_scale() -> None:
    assert len(ANGLES) == 16 and ANGLES[0] == 0
    assert [math.sin(angle / 2) for angle in ANGLES[1:]] == pytest.approx(
        [1 / k for k in range(1, 16)]
    )


def test_budget_refusal_preserves_source_problem_size() -> None:
    with pytest.raises(ValueError):
        synth.synthesize(
            exact, classical_inputs={"amplitudes": list(AMPLITUDES)}, max_qubits=6
        )


def test_all_fourteen_source_stages_are_present() -> None:
    directory = (
        Path(__file__).resolve().parents[2]
        / "examples/synthesis/algorithms/quantum_linear_solvers/hhl/overview"
    )
    proof = json.loads((directory / "provenance.json").read_text())
    notebook = json.loads((directory / "walkthrough.ipynb").read_text())
    assert [row["index"] for row in proof["reviewed_code_cells"]] == [
        7,
        10,
        14,
        17,
        22,
        24,
        27,
        28,
        30,
        33,
        35,
        38,
        40,
        41,
    ]
    for row in proof["reviewed_code_cells"]:
        assert notebook["cells"][row["native_code_cell"]]["cell_type"] == "code"
