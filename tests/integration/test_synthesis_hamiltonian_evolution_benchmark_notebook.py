"""Verify the complete benchmark contract and representative Pauli control states."""

import json
import math
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.pauli_evolution import (
    Term,
    reference_circuit,
    reference_sequence,
)
from examples.synthesis.tutorials.basics.exponentiation.example_exponentiation.overview.model import (
    WIDTH,
)
from examples.synthesis.tutorials.technology_demonstrations.hamiltonian_evolution.overview.benchmark import (
    CONTROL,
    build_qdrift,
    build_step,
    checked,
    controlled_product,
    fourth_order_times,
    fresh_model,
    repeated_product,
    trotter_reference,
)

pytestmark = pytest.mark.integration


def test_fresh_water_model_uses_benchmark_bit_order() -> None:
    model = fresh_model()
    assert len(model.terms) == 551
    assert model.terms[1][1] == ((11, "Z"),)
    assert model.coefficient_error < 1e-6
    assert model.rhf_energy == pytest.approx(-74.9629466565, abs=1e-8)


def test_controlled_identity_and_xy_phase_and_named_basis() -> None:
    terms: tuple[Term, ...] = ((-0.25, ()), (0.7, ((1, "X"), (9, "Y"))))
    schedule = trotter_reference(terms, (0.57,), symmetric=True)
    exact = reference_circuit(WIDTH, terms, schedule).state()
    product = controlled_product(terms, schedule)
    report = checked("Controlled phase", product, exact, controlled=True)
    assert report.native.n_qubits == WIDTH + 1
    assert report.target.operation_counts.keys() <= {"cx", "u3"}
    assert report.reference_error < 1e-12 and report.conversion_error < 1e-12
    assert CONTROL == WIDTH


def test_fourth_order_signed_steps_and_independent_qdrift() -> None:
    times = fourth_order_times()
    assert times == tuple(reversed(times))
    assert times[2] < 0
    assert math.fsum(times) == pytest.approx(1)
    assert math.fsum(time**3 for time in times) == pytest.approx(0, abs=1e-14)
    terms: tuple[Term, ...] = ((-0.3, ()), (0.6, ((0, "X"),)), (0.9, ((1, "Z"),)))
    hamiltonian = -0.3 * fq.I() + 0.6 * fq.X(0) + 0.9 * fq.Z(1)
    exact = reference_circuit(
        WIDTH, terms, reference_sequence(terms, 1, 100, seed=0)
    ).state()
    measured = build_qdrift(hamiltonian, 100).state()
    torch.testing.assert_close(measured, exact, atol=1e-12, rtol=0)
    repeated = repeated_product(build_step(hamiltonian, 0.25, 2), 4).state()
    product = reference_circuit(
        WIDTH, terms, trotter_reference(terms, (0.25,) * 4, symmetric=True)
    )
    torch.testing.assert_close(repeated, product.state(), atol=1e-12, rtol=0)


def test_notebook_pins_every_source_stage() -> None:
    overview = (
        Path(__file__).resolve().parents[2]
        / "examples/synthesis/tutorials/technology_demonstrations/hamiltonian_evolution/overview"
    )
    cells = json.loads((overview / "walkthrough.ipynb").read_text())["cells"]
    proof = json.loads((overview / "provenance.json").read_text())
    stages = [cell for cell in cells if cell["cell_type"] == "code"]
    assert len(stages) == len(proof["reviewed_code_cells"]) == 16
    assert all(
        cell["execution_count"] is None and not cell["outputs"] for cell in stages
    )
    assert [row["index"] for row in proof["reviewed_code_cells"]] == [
        3,
        5,
        6,
        7,
        9,
        11,
        13,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        24,
        25,
    ]
