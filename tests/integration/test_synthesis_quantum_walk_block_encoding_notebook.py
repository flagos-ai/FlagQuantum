"""Verify bounded graph encodings, Chebyshev phase, and complete walks."""

import json
from pathlib import Path

import pytest
import torch

from examples.synthesis.community.paper_implementation_project.explicit_quantum_circuits_for_block_encoding.quantum_walks_via_efficient_blockencoding.overview import (
    model,
    quantum,
    workflow,
)

pytestmark = pytest.mark.integration


def test_extended_tree_resource_and_projection() -> None:
    result = workflow.tree_case(8)
    uniform = torch.full((8,), 8**-0.5, dtype=model.REAL)
    torch.testing.assert_close(result.projection.real, (model.tree(8) @ uniform) / 8)
    assert result.lcu_terms == 36
    assert result.alpha == pytest.approx(2.95)
    assert result.oracle_resources.n_qubits == 8
    assert result.lcu_resources.n_qubits > result.oracle_resources.n_qubits
    assert result.lcu_error < 1e-6 and result.oracle_error < 1e-10


def test_cyclic_projection_and_source_chebyshev_phase() -> None:
    circulation = model.cyclic()
    realized = workflow.state_and_block(quantum.cyclic_encoding(), 8)
    torch.testing.assert_close(
        realized[:8], circulation.to(model.COMPLEX), atol=1e-10, rtol=0
    )
    polynomial = 2 * (circulation @ circulation) - torch.eye(8, dtype=model.REAL)
    expected = polynomial @ torch.tensor(quantum.CHEB_AMPLITUDES, dtype=model.REAL)
    actual = quantum.cheb_circuit().state().flatten()[:8]
    torch.testing.assert_close(actual, -expected.to(model.COMPLEX), atol=1e-10, rtol=0)


def test_original_both_walks_have_twenty_actual_measurements() -> None:
    without = workflow.walk_case(False)
    with_mark = workflow.walk_case(True)
    assert len(without.counts) == len(with_mark.counts) == 20
    assert all(count == workflow.LOCAL_SHOTS for count in without.counts)
    assert all(0 <= count <= workflow.LOCAL_SHOTS for count in with_mark.counts)
    assert min(with_mark.exact) < 0.1 and max(with_mark.exact) > 0.99
    assert max(without.state_error, with_mark.state_error) < 1e-10
    assert with_mark.block_error < 1e-5


def test_all_twenty_one_source_cells_are_unexecuted() -> None:
    overview = (
        Path(__file__).resolve().parents[2]
        / "examples/synthesis/community/paper_implementation_project/explicit_quantum_circuits_for_block_encoding/quantum_walks_via_efficient_blockencoding/overview"
    )
    cells = json.loads((overview / "walkthrough.ipynb").read_text())["cells"]
    proof = json.loads((overview / "provenance.json").read_text())
    stages = [cell for cell in cells if cell["cell_type"] == "code"]
    assert len(stages) == len(proof["reviewed_code_cells"]) == 21
    assert all(
        cell["execution_count"] is None and not cell["outputs"] for cell in stages
    )
