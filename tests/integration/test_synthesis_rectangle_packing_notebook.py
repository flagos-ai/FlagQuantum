"""Check independent full-space geometry and the complete bound quantum states."""

import json
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.applications.optimization.rectangles_packing.rectangles_packing_grid.comparisons.circuit import (
    BINDINGS,
    TERMS,
    reference_circuit,
)
from examples.synthesis.applications.optimization.rectangles_packing.rectangles_packing_grid.overview.model import (
    constraint_cost,
    make_model,
    unpack,
)
from examples.synthesis.applications.optimization.rectangles_packing.rectangles_packing_grid.overview.quantum import (
    program_builder,
)
from examples.synthesis.applications.optimization.rectangles_packing.rectangles_packing_grid.overview.workflow import (
    initial_angles,
)

pytestmark = pytest.mark.integration


def test_all_fresh_costs_and_original_ising_terms() -> None:
    model = make_model()
    assert len(model.placements) == 17 and len(model.terms) == 102
    assert dict(model.terms) == dict(TERMS)
    expected = torch.tensor(
        [constraint_cost(unpack(model, word)) for word in range(1 << 17)],
        dtype=torch.float64,
    )
    torch.testing.assert_close(expected, model.cost, atol=0, rtol=0)
    assert float(expected.min()) == -3
    assert int((expected == -3).sum()) == 8
    angles = initial_angles(model)
    assert len(angles) == 20
    assert angles[0] == 0 and angles[-1] == 0 and angles[1] == angles[-2]


@pytest.mark.parametrize("binding", BINDINGS)
def test_complete_complex_state_and_no_workspace(binding: tuple[float, ...]) -> None:
    model = make_model()
    program = program_builder(model)(binding)
    ising = fq.run(program).statevector()
    circuit = fq.run(reference_circuit(binding)).statevector()
    assert tuple(ising.shape) == (1, 1 << 17)
    torch.testing.assert_close(ising, circuit, atol=1e-12, rtol=0)


def test_original_stage_provenance() -> None:
    from examples.synthesis.applications.optimization.rectangles_packing.rectangles_packing_grid.overview import (
        __file__ as overview_path,
    )

    assert overview_path is not None
    path = Path(overview_path).parent
    book = json.loads((path / "walkthrough.ipynb").read_text())
    proof = json.loads((path / "provenance.json").read_text())
    cells = [cell for cell in book["cells"] if cell["cell_type"] == "code"]
    assert len(cells) == len(proof["reviewed_code_cells"]) == 19
    assert all(
        cell["execution_count"] is None and not cell["outputs"] for cell in cells
    )
