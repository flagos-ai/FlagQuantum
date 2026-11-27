"""Verify both source block maps over the entire data basis."""

import hashlib
import json
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.applications.cfd.qls_for_hybrid_solvers.verify_block_encoding.overview import (
    model,
    quantum,
    workflow,
)
from flagquantum import synth

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("column", range(8))
def test_native_pauli_block_complete_columns_phase_and_clean_workspace(
    column: int,
) -> None:
    matrix = model.source_matrix()
    vector = torch.eye(8, dtype=torch.float64)[column]
    program, scale = quantum.native_pauli_program(matrix, vector)
    state = fq.run(program).statevector().reshape(256, 16)
    reference, reference_scale = quantum.pauli_encoding(matrix, vector)
    assert scale == reference_scale
    assert program.circuit.n_wires == 12
    assert program.report["workspace_qubits"] == 4
    torch.testing.assert_close(
        state[:, 0], fq.run(reference).statevector().flatten(), atol=1e-12, rtol=0
    )
    torch.testing.assert_close(
        state[:8, 0], matrix[:, column] / scale, atol=1e-12, rtol=0
    )
    assert torch.count_nonzero(state[:, 1:]) == 0


@pytest.mark.parametrize("seed", [37, 91])
def test_native_pauli_block_signed_coherent_data_and_named_readout(seed: int) -> None:
    matrix, vector = model.source_matrix(), model.right_hand_side(seed)
    program, scale = quantum.native_pauli_program(matrix, vector)
    result = fq.run(program)
    state = result.statevector().reshape(256, 16)
    reference, _ = quantum.pauli_encoding(matrix, vector)
    torch.testing.assert_close(
        state[:, 0], fq.run(reference).statevector().flatten(), atol=1e-12, rtol=0
    )
    torch.testing.assert_close(
        state[:8, 0], matrix @ vector.to(model.CTYPE) / scale, atol=1e-12, rtol=0
    )
    observed = program.probabilities(result, registers=("selection", "data"))
    assert sum(observed.values()) == pytest.approx(1)
    assert all(
        observed[(0, word)] == pytest.approx(float(state[word, 0].abs() ** 2))
        for word in range(8)
    )
    assert torch.count_nonzero(state[:, 1:]) == 0


def test_native_pauli_budget_refuses_instead_of_using_opaque_select() -> None:
    with pytest.raises(synth.SynthesisError, match="max_workspace=0"):
        quantum.native_pauli_program(
            model.source_matrix(),
            model.right_hand_side(37),
            max_qubits=8,
            max_workspace=0,
        )
    with pytest.raises(ValueError, match="1 through 32"):
        quantum.native_pauli_program(
            torch.zeros((8, 8), dtype=model.CTYPE), model.right_hand_side(37)
        )


@pytest.mark.parametrize(
    "method, width, scale",
    [
        ("pauli", 8, 5.6428068318664355),
        ("banded", 6, 3.8927056451476174),
    ],
)
def test_both_block_maps_on_all_input_columns(
    method: workflow.Encoding, width: int, scale: float
) -> None:
    matrix = model.source_matrix()
    columns = []
    for vector in torch.eye(8, dtype=torch.float64):
        result = workflow.project(method, matrix, vector)
        assert result.width == width and result.scale == pytest.approx(scale, abs=1e-13)
        columns.append(result.values)
    torch.testing.assert_close(
        torch.stack(columns, dim=1), matrix / scale, atol=1e-12, rtol=0
    )


def test_original_data_and_stage_pins() -> None:
    matrix = model.source_matrix()
    assert matrix.shape == (8, 8) and int(matrix.count_nonzero()) == 21
    assert model.bands(matrix) == (-1, 0, 1)
    paulis = model.pauli_expansion(matrix)
    assert len(paulis) == 24 and any(abs(weight.imag) > 1e-4 for _, weight in paulis)
    root = Path(model.__file__).parent
    proof = json.loads((root / "provenance.json").read_text())
    for filename, key in (
        ("nozzle_small_csr.json", "native_sha256"),
        ("LICENSE-data.txt", "license_sha256"),
    ):
        assert (
            hashlib.sha256((root / filename).read_bytes()).hexdigest()
            == proof["source_data"][key]
        )
    book = json.loads((root / "walkthrough.ipynb").read_text())
    cells = [cell for cell in book["cells"] if cell["cell_type"] == "code"]
    assert len(cells) == len(proof["reviewed_code_cells"]) == 10
    assert all(
        cell["execution_count"] is None and not cell["outputs"] for cell in cells
    )


@pytest.mark.parametrize("column", range(8))
@pytest.mark.parametrize("workspace", [1, 2])
def test_native_banded_complete_columns_and_clean_workspace(
    column: int, workspace: int
) -> None:
    matrix = model.source_matrix()
    vector = torch.eye(8, dtype=torch.float64)[column]
    program, scale = quantum.native_banded_program(
        matrix, vector, max_workspace=workspace
    )
    state = fq.run(program).statevector().reshape(64, 1 << workspace)
    reference, reference_scale = quantum.banded_encoding(matrix, vector)
    assert scale == reference_scale
    assert program.circuit.n_wires == 6 + workspace
    assert program.report["workspace_qubits"] == workspace
    assert not any(gate.name == "any" for gate in program.circuit.instructions)
    torch.testing.assert_close(
        state[:, 0], fq.run(reference).statevector().flatten(), atol=1e-12, rtol=0
    )
    torch.testing.assert_close(
        state[:8, 0], matrix[:, column] / scale, atol=1e-12, rtol=0
    )
    assert float(state[:, 1:].abs().square().sum()) < 1e-24


@pytest.mark.parametrize("seed", [37, 91])
def test_native_banded_signed_coherent_state_and_named_success(seed: int) -> None:
    matrix, vector = model.source_matrix(), model.right_hand_side(seed)
    program, scale = quantum.native_banded_program(matrix, vector)
    result = fq.run(program)
    state = result.statevector().reshape(64, 4)
    reference, _ = quantum.banded_encoding(matrix, vector)
    torch.testing.assert_close(
        state[:, 0], fq.run(reference).statevector().flatten(), atol=1e-12, rtol=0
    )
    expected = matrix @ vector.to(model.CTYPE) / scale
    torch.testing.assert_close(state[:8, 0], expected, atol=1e-12, rtol=0)
    observed = program.probabilities(result, registers=("selection", "flag", "data"))
    assert sum(observed.values()) == pytest.approx(1)
    success = sum(
        value
        for (selector, flag, _), value in observed.items()
        if selector == flag == 0
    )
    assert success == pytest.approx(float(expected.abs().square().sum()))
    assert float(state[:, 1:].abs().square().sum()) < 1e-24
    resources = program.report["resources"]
    assert (resources.n_qubits, resources.n_operations, resources.depth) == (
        8,
        104,
        89,
    )


@pytest.mark.parametrize(
    "limits",
    [
        {"max_qubits": 6, "max_workspace": 0},
        {"max_qubits": 6, "max_workspace": 2},
        {"max_qubits": 8, "max_workspace": 0},
        {"max_operations": 100},
    ],
)
def test_native_banded_retains_checked_resource_refusals(
    limits: dict[str, int],
) -> None:
    with pytest.raises(synth.SynthesisError):
        quantum.native_banded_program(
            model.source_matrix(), model.right_hand_side(37), **limits
        )


def test_native_banded_tighter_qubit_budget_uses_more_instructions() -> None:
    matrix, vector = model.source_matrix(), model.right_hand_side(37)
    program, _ = quantum.native_banded_program(matrix, vector, max_qubits=7)
    resources = program.report["resources"]
    assert (resources.n_qubits, resources.n_operations, resources.depth) == (
        7,
        156,
        131,
    )
    assert program.report["workspace_qubits"] == 1
    state = fq.run(program).statevector().reshape(64, 2)
    reference, _ = quantum.banded_encoding(matrix, vector)
    torch.testing.assert_close(
        state[:, 0], fq.run(reference).statevector().flatten(), atol=1e-12, rtol=0
    )
    assert float(state[:, 1:].abs().square().sum()) < 1e-24


def test_native_banded_refuses_another_matrix_profile() -> None:
    matrix = model.source_matrix()
    for invalid in (
        matrix[:4, :4],
        matrix + 1j * torch.eye(8),
        matrix + torch.eye(8).roll(3, 0),
    ):
        with pytest.raises(ValueError, match="tridiagonal profile"):
            quantum.native_banded_program(invalid, model.right_hand_side(37))
