"""Validate the regenerated water term order and full native evolution states."""

import json
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.pauli_evolution import (
    load_terms,
    reference_circuit,
    reference_sequence,
)
from examples.synthesis.tutorials.basics.exponentiation.example_exponentiation.overview import (
    model,
    workflow,
)

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def result() -> workflow.Result:
    return workflow.run_workflow()


def test_native_rhf_spin_block_and_every_source_term(result: workflow.Result) -> None:
    assert result.model.rhf_energy == pytest.approx(-74.96294665654, abs=1e-8)
    assert result.model.frozen_orbitals == 1
    directory = Path(model.__file__).resolve().parents[1]
    width, source = load_terms(directory / "hamiltonian.json")
    assert width == 12
    assert len(result.model.terms) == len(source) == 551
    for (generated, fresh_support), (saved, source_support) in zip(
        result.model.terms, source, strict=True
    ):
        assert fresh_support == tuple(sorted(source_support))
        assert generated == pytest.approx(saved, abs=1e-6, rel=0)
    assert result.model.coefficient_difference < 1e-6
    assert result.model.terms[0][0] < -70


def test_full_state_and_the_named_hardware_profile(result: workflow.Result) -> None:
    assert result.absolute_error < 1e-11
    assert result.target_error < 1e-11
    assert result.program.report["workspace_qubits"] == 0
    assert set(result.target_resources.operation_counts) == {"cx", "u3"}
    assert result.target_resources.operation_counts["cx"] == 4176
    assert result.target_resources.n_qubits == 12
    assert sum(p for _, p in result.distribution) == pytest.approx(1.0, abs=1e-11)
    assert all(0 <= word < 2**12 for word, _ in result.distribution)


def test_second_nonzero_time_matches_independent_circuit(
    result: workflow.Result,
) -> None:
    time = -0.23
    terms = result.model.terms
    reference = reference_circuit(12, terms, reference_sequence(terms, time, None))
    seed = fq.Circuit(12, dtype=torch.complex128)
    seed.x(1).x(3).x(7).x(9).h(2).ry(8, theta=0.37).phase(2, theta=-0.41)
    expected = fq.run(seed.copy().compose(reference)).statevector().flatten()
    native = workflow.build_program(result.model, time)
    actual = fq.run(seed.copy().compose(native.circuit)).statevector().flatten()
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=0)
    with pytest.raises(ValueError, match="Evolution time must be finite"):
        workflow.build_program(result.model, float("nan"))


def test_walkthrough_keeps_all_three_stages() -> None:
    directory = Path(model.__file__).parent
    book = json.loads((directory / "walkthrough.ipynb").read_text())
    provenance = json.loads((directory / "provenance.json").read_text())
    stages = [cell for cell in book["cells"] if cell["cell_type"] == "code"]
    assert len(stages) == len(provenance["reviewed_code_cells"]) == 3
    assert all(
        cell["execution_count"] is None and cell["outputs"] == [] for cell in stages
    )
