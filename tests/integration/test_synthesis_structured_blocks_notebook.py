"""Check every signed matrix column and nonzero coherent source bindings."""

import json
from pathlib import Path

import pytest
import torch

from examples.synthesis.community.paper_implementation_project.block_encoding.select_structures_be.overview.model import (
    make_structures,
)
from examples.synthesis.community.paper_implementation_project.block_encoding.select_structures_be.overview.workflow import (
    project,
)

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "case, dimension, scale, width",
    ((0, 8, 8, 7), (1, 16, 4, 8), (2, 16, 3, 7), (3, 16, 5, 9)),
)
def test_source_dimensions_and_all_signed_columns(
    case: int, dimension: int, scale: int, width: int
) -> None:
    model = make_structures()[case]
    columns = []
    for vector in torch.eye(dimension, dtype=torch.float64):
        result = project(model, tuple(map(float, vector)))
        assert result.scale == scale and result.circuit.n_qubits == width
        columns.append(result.actual)
    torch.testing.assert_close(
        torch.stack(columns, dim=1), model.matrix / scale, atol=1e-12, rtol=0
    )


@pytest.mark.parametrize("seed", (7, 41))
def test_coherent_signed_inputs_and_no_phase_alignment(seed: int) -> None:
    rng = torch.Generator().manual_seed(seed)
    for model in make_structures(seed):
        vector = torch.randn(model.matrix.shape[0], dtype=torch.float64, generator=rng)
        vector /= vector.norm()
        result = project(model, tuple(map(float, vector)))
        assert result.error < 1e-12


def test_source_stages_are_unexecuted() -> None:
    from examples.synthesis.community.paper_implementation_project.block_encoding.select_structures_be.overview import (
        __file__ as overview_path,
    )

    assert overview_path is not None
    path = Path(overview_path).parent
    book = json.loads((path / "walkthrough.ipynb").read_text())
    proof = json.loads((path / "provenance.json").read_text())
    cells = [row for row in book["cells"] if row["cell_type"] == "code"]
    assert len(cells) == len(proof["reviewed_code_cells"]) == 10
    assert all(row["execution_count"] is None and not row["outputs"] for row in cells)
