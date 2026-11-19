"""Original-sized kidney matching keeps cost, phase, bit order and sampling."""

import json
import math
from pathlib import Path

import pytest

import flagquantum as fq
from examples.synthesis.applications.optimization.kidney_exchange.kidney_exchange_problem.comparisons.circuit import (
    cost as independent_cost,
)
from examples.synthesis.applications.optimization.kidney_exchange.kidney_exchange_problem.comparisons.circuit import (
    reference_circuit,
)
from examples.synthesis.applications.optimization.kidney_exchange.kidney_exchange_problem.overview.model import (
    assignment,
)
from examples.synthesis.applications.optimization.kidney_exchange.kidney_exchange_problem.overview.quantum import (
    build_program,
)
from examples.synthesis.applications.optimization.kidney_exchange.kidney_exchange_problem.overview.workflow import (
    run_workflow,
    sample_distribution,
)
from examples.synthesis.compare_states import compare_states

pytestmark = pytest.mark.integration


def test_all_512_assignments_match_independent_reference_and_legal_matching() -> None:
    table = tuple(assignment(word) for word in range(512))
    for word, row in enumerate(table):
        bits = tuple((word >> bit) & 1 for bit in range(9))
        assert row.penalized_cost == pytest.approx(-independent_cost(bits))
        assert row.feasible == (
            all(sum(line) <= 1 for line in row.rows)
            and all(
                sum(row.rows[donor][patient] for donor in range(3)) <= 1
                for patient in range(3)
            )
        )
    assert sum(row.feasible for row in table) == 34
    best = min(table, key=lambda row: (row.penalized_cost, row.word))
    assert best.feasible and best.score == pytest.approx(2.35)
    assert best.word == 161
    assert best.rows == ((1, 0, 0), (0, 0, 1), (0, 1, 0))


@pytest.mark.parametrize(
    "parameters",
    (
        (0.017, -0.023, 0.031, -0.047, 0.053, -0.067, 0.071, -0.083, 0.097, -0.101),
        (-0.11, 0.13, -0.17, 0.19, -0.23, 0.29, -0.31, 0.37, -0.41, 0.43),
    ),
)
def test_nonzero_quantum_states_match_independent_circuit(
    parameters: tuple[float, ...],
) -> None:
    program = build_program(parameters)
    compare_states(program, reference_circuit(parameters), ("matches",))
    assert program.report["workspace_qubits"] == 0
    assert program.selection.objective == "width"
    assert fq.run(program).statevector().numel() == 512


def test_nonfeasible_observation_is_reported_without_reference_replacement() -> None:
    result = sample_distribution(
        tuple(1.0 if word == 511 else 0.0 for word in range(512)), shots=123, seed=7
    )
    assert len(result) == 1 and result[0].count == 123
    observed = result[0].assignment
    assert not observed.feasible and observed.word == 511
    assert observed.conflict_pairs == 18
    assert -observed.penalized_cost != observed.score


def test_actual_samples_are_seeded_and_preserve_named_bit_order() -> None:
    probabilities = (1 / 512,) * 512
    first = sample_distribution(probabilities, shots=250, seed=11)
    assert first == sample_distribution(probabilities, shots=250, seed=11)
    assert sum(row.count for row in first) == 250
    assert first != sample_distribution(probabilities, shots=250, seed=12)
    for row in first:
        reconstructed = sum(
            value << (3 * donor + patient)
            for donor, line in enumerate(row.assignment.rows)
            for patient, value in enumerate(line)
        )
        assert reconstructed == row.assignment.word


@pytest.mark.parametrize(
    "parameters", ((0.1,), (True,) * 10, (math.nan,) * 10, (math.inf,) * 10)
)
def test_invalid_bindings_refuse_before_synthesis(
    parameters: tuple[float, ...],
) -> None:
    with pytest.raises(ValueError, match="ten finite"):
        build_program(parameters)


@pytest.mark.parametrize("word", (-1, True, 512))
def test_invalid_assignment_refuses(word: int) -> None:
    with pytest.raises(ValueError, match="word"):
        assignment(word)


def test_bounded_search_records_actual_quantum_evaluations_and_final_samples() -> None:
    result = run_workflow(maxiter=1, shots=200, seed=7)
    assert result.trace and result.optimizer.iterations <= 1
    assert all(math.isfinite(row.cost) for row in result.trace)
    assert sum(row.count for row in result.samples) == 200
    assert result.best_observed == result.samples[0]
    assert result.reference.feasible and result.reference.score == pytest.approx(2.35)
    assert (
        result.best_observed_feasible is None
        or result.best_observed_feasible in result.samples
    )


@pytest.mark.parametrize(
    "shots,seed", ((0, 7), (True, 7), (100001, 7), (10, -1), (10, True))
)
def test_invalid_sampling_controls_refuse(shots: int, seed: int) -> None:
    with pytest.raises(ValueError, match="shots|seed"):
        sample_distribution((1 / 512,) * 512, shots=shots, seed=seed)


def test_original_nine_source_stages_are_mapped_and_outputs_unexecuted() -> None:
    root = (
        Path(__file__).resolve().parents[2]
        / "examples/synthesis/applications/optimization/kidney_exchange/kidney_exchange_problem/overview"
    )
    proof = json.loads((root / "provenance.json").read_text())
    notebook = json.loads((root / "walkthrough.ipynb").read_text())
    assert tuple(row["index"] for row in proof["reviewed_code_cells"]) == (
        1,
        3,
        5,
        8,
        10,
        12,
        15,
        17,
        19,
    )
    assert all(len(row["sha256"]) == 64 for row in proof["reviewed_code_cells"])
    assert all(
        not cell.get("outputs")
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
