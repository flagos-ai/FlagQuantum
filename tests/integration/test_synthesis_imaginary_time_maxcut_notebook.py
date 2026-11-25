"""Check fresh costs, full-state signs, and all finite imaginary-time updates."""

import json
from pathlib import Path

import pytest
import torch

import flagquantum as fq

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "angles",
    ((0.13, -0.21, 0.32, -0.17, 0.29, -0.41), (0.31, 0.16, -0.4, 0.1, 0.15, -0.23)),
)
def test_complete_state_in_two_bound_parameters(angles: tuple[float, ...]) -> None:
    pytest.importorskip("numpy")
    from examples.synthesis.applications.optimization.variational_quantum_imaginary_time_evolution.overview.model import (
        source_model,
    )
    from examples.synthesis.applications.optimization.variational_quantum_imaginary_time_evolution.overview.quantum import (
        program_builder,
    )
    from examples.synthesis.applications.optimization.variational_quantum_imaginary_time_evolution.overview.reference import (
        reference_circuit,
    )

    model = source_model()
    native = fq.run(program_builder(model)(angles)).statevector()
    expected = fq.run(reference_circuit(model, angles)).statevector()
    torch.testing.assert_close(native, expected, atol=1e-12, rtol=0)


def test_source_size_cost_and_finite_workflow() -> None:
    pytest.importorskip("numpy")
    from examples.synthesis.applications.optimization.variational_quantum_imaginary_time_evolution.overview.reference import (
        direct_cut,
    )
    from examples.synthesis.applications.optimization.variational_quantum_imaginary_time_evolution.overview.workflow import (
        SHOTS,
        run_workflow,
    )

    result = run_workflow()
    assert result.measurements == 196 and result.measurements * SHOTS == 25_088
    assert len(result.model.edges) == 6 and len(result.model.generators) == 6
    for word in range(16):
        assert result.model.energy[word] == pytest.approx(
            -direct_cut(result.model, word), abs=1e-12
        )
    assert min(map(float, result.model.energy)) == pytest.approx(-2.071155679900729)
    assert tuple(row.iteration for row in result.rounds) == tuple(range(1, 15))
    assert tuple(row.partition_word for row in result.rounds) == (
        14,
        10,
        11,
        5,
        8,
        12,
        7,
        12,
        7,
        2,
        12,
        2,
        13,
        12,
    )
    assert result.rounds[-1].energy == pytest.approx(-1.8724520055231215, abs=1e-12)


def test_original_stages_are_unexecuted() -> None:
    from examples.synthesis.applications.optimization.variational_quantum_imaginary_time_evolution.overview import (
        __file__ as overview_path,
    )

    assert overview_path is not None
    root = Path(overview_path).parent
    book = json.loads((root / "walkthrough.ipynb").read_text())
    proof = json.loads((root / "provenance.json").read_text())
    cells = [cell for cell in book["cells"] if cell["cell_type"] == "code"]
    assert len(cells) == len(proof["reviewed_code_cells"]) == 22
    assert all(
        cell["execution_count"] is None and not cell["outputs"] for cell in cells
    )
