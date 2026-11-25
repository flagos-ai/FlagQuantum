"""Verify both source block maps over the entire data basis."""

import hashlib
import json
from pathlib import Path

import pytest
import torch

from examples.synthesis.applications.cfd.qls_for_hybrid_solvers.verify_block_encoding.overview import (
    model,
    workflow,
)

pytestmark = pytest.mark.integration


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
