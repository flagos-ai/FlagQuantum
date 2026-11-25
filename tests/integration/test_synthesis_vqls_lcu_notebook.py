"""Check the exact projected block and finite-shot VQLS accounting."""

import json
import math
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.algorithms.quantum_linear_solvers.vqls.vqls_with_lcu.comparisons.circuit import (
    BINDINGS,
    reference_circuit,
)
from examples.synthesis.algorithms.quantum_linear_solvers.vqls.vqls_with_lcu.overview import (
    quantum,
    workflow,
)

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("parameters", BINDINGS)
def test_independent_unnormalized_projection(parameters: tuple[float, ...]) -> None:
    program = quantum.build_objective(parameters)
    actual = fq.run(program).statevector().flatten()
    ansatz = fq.run(reference_circuit(parameters)).statevector().flatten()
    gate = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex128) / math.sqrt(2)
    h3 = torch.kron(torch.kron(gate, gate), gate)
    expected = h3 @ (quantum.MATRIX * ansatz)
    torch.testing.assert_close(actual[:8], expected, atol=2e-12, rtol=0)
    assert program.report["workspace_qubits"] == 0


def test_actual_shot_counts_match_the_program_distribution() -> None:
    parameters = BINDINGS[0]
    seed = 23
    report = workflow.sampled_query(parameters, torch.Generator().manual_seed(seed))
    program = quantum.build_objective(parameters)
    actual = program.probabilities(
        fq.run(program), registers=("auxiliary", "system"), threshold=0
    )
    expected = torch.tensor(
        [actual.get((aux, system), 0.0) for aux in range(4) for system in range(8)],
        dtype=torch.float64,
    )
    # Validate the conditional event slice against the public named readout
    # and the seeded finite counts from the same physical distribution.
    torch.testing.assert_close(
        workflow.probabilities(program), expected, atol=1e-12, rtol=0
    )
    draws = torch.multinomial(
        expected,
        workflow.SHOTS,
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    assert workflow.SHOTS == 204800 and workflow.BUDGET == 2000
    assert report.auxiliary_zero == int((draws < 8).sum())
    assert report.all_zero == int((draws == 0).sum())
    assert report.cost == 1 - report.all_zero / report.auxiliary_zero


def test_source_notebook_stages_and_input_validation() -> None:
    root = Path(quantum.__file__).parent
    book = json.loads((root / "walkthrough.ipynb").read_text())
    proof = json.loads((root / "provenance.json").read_text())
    cells = [cell for cell in book["cells"] if cell["cell_type"] == "code"]
    assert len(cells) == len(proof["reviewed_code_cells"]) == 17
    assert all(
        cell["execution_count"] is None and not cell["outputs"] for cell in cells
    )
    with pytest.raises(ValueError, match="nine finite"):
        quantum.build_objective((float("inf"),) * 9)
