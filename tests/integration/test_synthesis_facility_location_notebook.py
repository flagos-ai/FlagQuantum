"""Full original facility model keeps nonlinear constraints and actual samples."""

import json
import math
from pathlib import Path

import pytest
import torch

from examples.synthesis.applications.logistics.facility_location.comparisons.circuit import (
    BINDINGS,
    reference_circuit,
)
from examples.synthesis.applications.logistics.facility_location.comparisons.circuit import (
    cost as independent_cost,
)
from examples.synthesis.applications.logistics.facility_location.main import main
from examples.synthesis.applications.logistics.facility_location.overview.model import (
    assignment,
    classical_reference,
)
from examples.synthesis.applications.logistics.facility_location.overview.quantum import (
    build_program,
)
from examples.synthesis.applications.logistics.facility_location.overview.workflow import (
    read_distribution,
    run_workflow,
    sample_distribution,
)
from examples.synthesis.compare_states import compare_states
from flagquantum import synth

pytestmark = pytest.mark.integration


def test_all4096costs_preserve_original_nonlinear_empty_facility_constraint() -> None:
    feasible = 0
    for word in range(4096):
        bits = tuple((word >> i) & 1 for i in range(12))
        value = assignment(word)
        empty = sum(all(bits[4 * i + j] == 0 for j in range(4)) for i in range(3))
        columns = tuple(sum(bits[4 * i + j] for i in range(3)) for j in range(4))
        assert value.cost == pytest.approx(-independent_cost(bits))
        assert (
            value.penalty
            == sum((column - 1) ** 2 for column in columns) + (empty - 1) ** 2
        )
        assert value.feasible == (columns == (1, 1, 1, 1) and empty == 1)
        feasible += value.feasible
    assert feasible == 42
    reference = classical_reference()
    assert len(reference) == 1 and reference[0].word == 135
    assert reference[0].raw_cost == pytest.approx(0.87)
    assert min(assignment(word).cost for word in range(4096)) == pytest.approx(0.87)
    assert assignment(0).penalty == 8
    assert assignment(4095).penalty == 17


@pytest.mark.parametrize("parameters", BINDINGS)
def test_two_nonzero_full_states_match_independent_nonlinear_circuit(
    parameters: tuple[float, ...],
) -> None:
    program = build_program(parameters)
    compare_states(program, reference_circuit(parameters), ("assignments",))
    assert program.report["workspace_qubits"] == 5


def test_zero_angle_uniform_is_executed_with_named_assignment_order() -> None:
    probabilities = read_distribution(build_program((0.0,) * 10))
    torch.testing.assert_close(
        torch.tensor(probabilities), torch.full((4096,), 1 / 4096), atol=1e-14, rtol=0
    )
    samples = sample_distribution(probabilities, shots=500, seed=7)
    assert samples == sample_distribution(probabilities, shots=500, seed=7)
    assert sum(row.count for row in samples) == 500
    for row in samples:
        word = sum(
            bit << (4 * i + j)
            for i, values in enumerate(row.assignment.rows)
            for j, bit in enumerate(values)
        )
        assert word == row.assignment.word


def test_infeasible_observation_is_not_replaced_by_exact_reference() -> None:
    samples = sample_distribution(
        tuple(1.0 if word == 4095 else 0.0 for word in range(4096)), shots=123, seed=7
    )
    assert len(samples) == 1 and samples[0].assignment.word == 4095
    assert not samples[0].assignment.feasible
    assert samples[0].assignment.opened == (0, 1, 2)
    assert samples[0].assignment.penalty == 17 and samples[0].count == 123


@pytest.mark.parametrize(
    "parameters", ((0.1,), (True,) * 10, (math.nan,) * 10, (math.inf,) * 10)
)
def test_invalid_bindings_refuse(parameters: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="ten finite"):
        build_program(parameters)


@pytest.mark.parametrize("qubits,workspace", ((16, 5), (17, 4)))
def test_full_physical_width_and_workspace_budget_refuse(
    qubits: int, workspace: int
) -> None:
    with pytest.raises(ValueError, match="qubit|width|budget"):
        synth.synthesize(
            main,
            classical_inputs={"params": BINDINGS[0]},
            max_qubits=qubits,
            max_workspace=workspace,
        )


@pytest.mark.parametrize(
    "shots,seed", ((0, 7), (True, 7), (100001, 7), (10, -1), (10, True))
)
def test_invalid_sampling_controls_refuse(shots: int, seed: int) -> None:
    with pytest.raises(ValueError, match="shots|seed"):
        sample_distribution((1 / 4096,) * 4096, shots=shots, seed=seed)


def test_actual_bounded_search_retains_optimized_uniform_and_separate_reference() -> (
    None
):
    result = run_workflow(maxiter=1, shots=150, seed=7)
    assert result.trace and result.optimizer.iterations <= 1
    assert all(math.isfinite(row.cost) for row in result.trace)
    assert (
        sum(row.count for row in result.samples)
        == sum(row.count for row in result.uniform)
        == 150
    )
    assert result.best_observed == result.samples[0]
    assert result.reference[0].raw_cost == pytest.approx(0.87)


def test_all_seventeen_source_stages_are_mapped_and_unexecuted() -> None:
    root = (
        Path(__file__).resolve().parents[2]
        / "examples/synthesis/applications/logistics/facility_location/overview"
    )
    proof = json.loads((root / "provenance.json").read_text())
    notebook = json.loads((root / "walkthrough.ipynb").read_text())
    assert tuple(row["index"] for row in proof["reviewed_code_cells"]) == (
        6,
        8,
        9,
        12,
        14,
        16,
        18,
        20,
        23,
        25,
        27,
        29,
        31,
        34,
        35,
        37,
        38,
    )
    assert all(
        not c.get("outputs") for c in notebook["cells"] if c["cell_type"] == "code"
    )
