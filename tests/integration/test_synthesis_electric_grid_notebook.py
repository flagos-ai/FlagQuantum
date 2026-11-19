"""The original electric grid retains all18bits and the true bounded slack costs."""

import itertools
import json
import math
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.applications.optimization.electric_grid_optimization.comparisons.circuit import (
    BINDINGS,
    reference_circuit,
)
from examples.synthesis.applications.optimization.electric_grid_optimization.main import (
    main,
)
from examples.synthesis.applications.optimization.electric_grid_optimization.overview.model import (
    assignment,
    classical_reference,
    residual_penalty,
)
from examples.synthesis.applications.optimization.electric_grid_optimization.overview.quantum import (
    build_program,
)
from examples.synthesis.applications.optimization.electric_grid_optimization.overview.workflow import (
    cost_table,
    decode_joint,
    read_distribution,
    run_workflow,
    sample_distribution,
)
from examples.synthesis.compare_states import compare_states
from flagquantum import synth

pytestmark = pytest.mark.integration


def test_all_joint_costs_and_feasible_reference_preserve_original_constraints() -> None:
    table = cost_table()
    independent_costs = (
        (0.5, 1.0, 1.0, 2.1),
        (1.0, 0.6, 1.4, 1.0),
        (1.0, 1.4, 0.4, 2.3),
    )
    zero_penalty_count = 0
    for word in range(4096):
        value = assignment(word)
        raw = sum(
            independent_costs[i][j]
            for i in range(3)
            for j in range(4)
            if word & (1 << (4 * i + j))
        )
        rows = tuple(sum((word >> (4 * i + j)) & 1 for j in range(4)) for i in range(3))
        columns = tuple(
            sum((word >> (4 * i + j)) & 1 for i in range(3)) for j in range(4)
        )
        assert value.raw_cost == pytest.approx(raw)
        assert value.feasible == (max(rows) <= 2 and columns == (1, 1, 1, 1))
        for sa, sb, sc in itertools.product(range(4), repeat=3):
            slack = (sa, sb, sc)
            penalty = sum((column - 1) ** 2 for column in columns) + sum(
                (row + (s & 1) + ((s >> 1) & 1) - 2) ** 2
                for row, s in zip(rows, slack, strict=True)
            )
            joint = word | (sa << 12) | (sb << 14) | (sc << 16)
            assert residual_penalty(value, slack) == penalty
            assert float(table[joint]) == pytest.approx(raw + 10 * penalty)
            zero_penalty_count += penalty == 0
    assert zero_penalty_count == 162
    assert sum(assignment(word).feasible for word in range(4096)) == 54
    reference = classical_reference()
    assert len(reference) == 1 and reference[0].word == 1185
    assert reference[0].raw_cost == pytest.approx(2.5)
    assert float(table.min()) == pytest.approx(2.5)


@pytest.mark.parametrize("parameters", BINDINGS)
def test_two_nonzero_full_states_match_independent_circuit(
    parameters: tuple[float, ...],
) -> None:
    program = build_program(parameters)
    compare_states(
        program, reference_circuit(parameters), ("a", "b", "c", "sa", "sb", "sc")
    )
    assert program.report["workspace_qubits"] == 0
    assert fq.run(program).statevector().numel() == 2**18


def test_zero_angle_uniform_is_executed_and_named_joint_order_preserved() -> None:
    probabilities = read_distribution(build_program((0.0,) * 8))
    assert torch.allclose(
        probabilities, torch.full_like(probabilities, 1 / 2**18), atol=1e-14, rtol=0
    )
    sampled = sample_distribution(probabilities, shots=200, seed=17)
    assert sampled == sample_distribution(probabilities, shots=200, seed=17)
    assert sum(row.count for row in sampled) == 200
    for row in sampled:
        value, slack = decode_joint(row.joint_word)
        assert value == row.assignment and slack == row.slack


def test_infeasible_joint_observation_is_never_replaced_by_reference() -> None:
    probabilities = torch.zeros(2**18, dtype=torch.float64)
    probabilities[-1] = 1
    sampled = sample_distribution(probabilities, shots=123, seed=11)
    assert len(sampled) == 1 and sampled[0].joint_word == 2**18 - 1
    assert not sampled[0].assignment.feasible
    assert sampled[0].penalty == 64 and sampled[0].count == 123


@pytest.mark.parametrize(
    "parameters", ((0.1,), (True,) * 8, (math.inf,) * 8, (math.nan,) * 8)
)
def test_invalid_bindings_refuse(parameters: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="eight finite"):
        build_program(parameters)


def test_joint_width_budget_refuses_without_dropping_slack_bits() -> None:
    with pytest.raises(ValueError, match="qubit|width|budget"):
        synth.synthesize(
            main,
            classical_inputs={"params": BINDINGS[0]},
            max_qubits=17,
            max_workspace=0,
        )


@pytest.mark.parametrize(
    "shots,seed", ((0, 7), (True, 7), (100001, 7), (10, -1), (10, True))
)
def test_invalid_sampling_controls_refuse(shots: int, seed: int) -> None:
    with pytest.raises(ValueError, match="shots|seed"):
        sample_distribution(
            torch.full((2**18,), 1 / 2**18, dtype=torch.float64), shots=shots, seed=seed
        )


def test_bounded_actual_search_keeps_both_finite_samples_and_separate_reference() -> (
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
    assert result.reference[0].raw_cost == pytest.approx(2.5)


def test_all_thirteen_source_stages_are_mapped_and_outputs_unexecuted() -> None:
    root = (
        Path(__file__).resolve().parents[2]
        / "examples/synthesis/applications/optimization/electric_grid_optimization/overview"
    )
    proof = json.loads((root / "provenance.json").read_text())
    notebook = json.loads((root / "walkthrough.ipynb").read_text())
    assert tuple(row["index"] for row in proof["reviewed_code_cells"]) == (
        4,
        5,
        7,
        9,
        11,
        13,
        15,
        17,
        19,
        21,
        23,
        25,
        27,
    )
    assert all(
        not c.get("outputs") for c in notebook["cells"] if c["cell_type"] == "code"
    )
