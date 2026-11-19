"""Check the full source pool, signed gradients and adaptive schedule timing."""

import json
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.algorithms.quantum_state_preparation.adapt_vqe.overview import (
    model,
    quantum,
    reference,
    workflow,
)

pytestmark = pytest.mark.integration


def test_source_pool_order_duplicate_words_and_reference_energy() -> None:
    problem = model.build_model()
    assert len(model.POOL) == 56
    assert model.POOL[6] == ((2, "Y"),)
    assert model.POOL[10] == ((0, "X"), (3, "X"))
    assert len({frozenset(word) for word in model.POOL}) == 44
    assert problem.ground_energy == pytest.approx(-1.1893807157921064, abs=1e-13)
    gradients = workflow.screen(problem, (), ())
    assert min(gradients) == pytest.approx(-1.0, abs=1e-13)
    assert min(range(len(gradients)), key=gradients.__getitem__) == 6


@pytest.mark.parametrize(
    "indices",
    ((6,), (10, 2, 4, 50), (0, 0, 2, 4, 6, 8, 10, 12, 14, 16)),
)
@pytest.mark.parametrize("scale", (-0.037, 0.071))
def test_native_full_state_matches_independent_circuit(
    indices: tuple[int, ...], scale: float
) -> None:
    angles = tuple(scale * (index + 1) for index in range(len(indices)))
    program = quantum.build_program(indices, angles)
    expected = (
        fq.run(reference.reference_circuit(indices, angles)).statevector().flatten()
    )
    torch.testing.assert_close(quantum.state(program), expected, atol=1e-12, rtol=0)
    assert tuple(program.registers) == ("v",)


@pytest.mark.parametrize("scale", (-0.23, 0.37))
def test_every_screening_gradient_matches_dense_difference(scale: float) -> None:
    problem = model.build_model()
    indices = (10, 2, 4, 50)
    params = tuple(scale * (index + 1) for index in range(len(indices)))
    actual = workflow.screen(problem, indices, params)
    expected = reference.finite_difference_gradients(
        problem.hamiltonian, indices, params
    )
    assert len(actual) == len(expected) == 56
    assert actual == pytest.approx(expected, abs=5e-10)


def test_standard_and_modified_scheduling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(workflow, "OUTER_BUDGET", 2)
    monkeypatch.setattr(workflow, "INNER_BUDGET", 5)
    problem = model.build_model()
    standard = workflow.adapt(problem, modified=False)
    modified = workflow.adapt(problem, modified=True)
    assert len(standard.steps) == len(modified.steps) == 2
    assert standard.steps[0].fit.indices == ()
    assert standard.steps[0].fit.energy == pytest.approx(0.5, abs=1e-13)
    assert standard.steps[0].chosen_index == 6
    assert standard.steps[1].fit.indices == (6,)
    assert modified.steps[0].fit.indices == (10,)
    assert modified.steps[1].fit.indices[:1] == (10,)
    for run in (standard, modified):
        for row in run.steps:
            assert row.fit.evaluations == len(row.fit.energies)
            assert row.fit.iterations <= 5
            assert row.fit.energy == pytest.approx(min(row.fit.energies), abs=1e-14)
            assert len(row.gradients) == 56
            assert row.gradient_norm == pytest.approx(
                sum(value**2 for value in row.gradients) ** 0.5 / len(row.gradients)
            )


@pytest.mark.parametrize(
    "indices,angles", [((56,), (0.1,)), ((0,), (float("nan"),)), ((4, 5), (0.2,))]
)
def test_reject_invalid_source_binding(
    indices: tuple[int, ...], angles: tuple[float, ...]
) -> None:
    with pytest.raises(ValueError, match="valid source pool selections"):
        quantum.build_program(indices, angles)


def test_clean_notebook_maps_all_source_code() -> None:
    root = Path(workflow.__file__).parent
    proof = json.loads((root / "provenance.json").read_text())
    notebook = json.loads((root / "walkthrough.ipynb").read_text())
    cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(cells) == len(proof["reviewed_code_cells"]) == 18
    assert all(
        cell["execution_count"] is None and cell["outputs"] == [] for cell in cells
    )
