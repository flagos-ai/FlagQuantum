"""Original MaxCut Workshop semantics, bounded executions and actual observations."""

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

import flagquantum as fq
from examples.synthesis.tutorials.workshops.combinatorial_workshop.combinatorial_qmod_workshop_for_maxcut.comparisons.circuit import (
    BINDINGS,
)
from examples.synthesis.tutorials.workshops.combinatorial_workshop.combinatorial_qmod_workshop_for_maxcut.main import (
    main,
)
from examples.synthesis.tutorials.workshops.combinatorial_workshop.combinatorial_qmod_workshop_for_maxcut.overview.workflow import (
    Workflow,
    build_program,
    classical_optima,
    cut_cost,
    distribution,
    group_bits,
    initial_parameters,
    mean_cost,
    run_workflow,
    sample,
    source_graph,
    verify,
)
from flagquantum import synth

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def workflow() -> Workflow:
    return run_workflow(evaluation_budget=3, shots=193, seed=1970)


@pytest.mark.parametrize("parameters", (initial_parameters(), *BINDINGS))
def test_original_schedule_and_nonzero_absolute_states(
    parameters: tuple[float, ...],
) -> None:
    check = verify(parameters)
    assert check.maximum_state_error < 3e-11
    assert check.dirty_workspace_probability < 1e-22
    assert check.resources.n_qubits == 5


def test_all_original_graph_costs_and_unscaled_initial_schedule() -> None:
    graph = source_graph()
    assert tuple(graph.nodes) == tuple(range(5))
    assert set(graph.edges) == {(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4)}
    for word in range(32):
        bits = group_bits(word)
        assert cut_cost(word) == -sum(bits[u] != bits[v] for u, v in graph.edges)
    assert -cut_cost(classical_optima()[0]) == 5
    assert set(classical_optima()) == {12, 13, 18, 19}
    assert initial_parameters() == pytest.approx(
        (0, 1, 1 / 3, 2 / 3, 2 / 3, 1 / 3, 1, 0)
    )


def test_execution_cap_prevents_one_extra_objective_call() -> None:
    with patch(
        "examples.synthesis.tutorials.workshops.combinatorial_workshop.combinatorial_qmod_workshop_for_maxcut.overview.workflow.fq.run",
        wraps=fq.run,
    ) as run:
        observed = run_workflow(evaluation_budget=3, shots=17, seed=0)
    assert run.call_count == 4  # Three real objectives plus one final readout.
    assert len(observed.trace) == 3
    assert observed.termination == "evaluation_budget" and not observed.converged
    assert observed.selected == min(observed.trace, key=lambda row: row.cost)


def test_trace_selected_mean_and_independent_selected_state(workflow: Workflow) -> None:
    assert mean_cost(distribution(workflow.program)) == pytest.approx(
        workflow.selected.cost, abs=1e-10
    )
    assert workflow.selected in workflow.trace
    assert verify(workflow.selected.parameters).maximum_state_error < 3e-11


def test_actual_sample_counts_and_observed_cut_ranking(workflow: Workflow) -> None:
    assert sum(row.count for row in workflow.samples) == 193
    assert all(row.count > 0 for row in workflow.samples)
    assert list(workflow.samples) == sorted(
        workflow.samples, key=lambda row: (row.cost, row.word)
    )
    assert workflow.best_observed == workflow.samples[0]
    assert all(
        row.cost == cut_cost(row.word) and row.groups == group_bits(row.word)
        for row in workflow.samples
    )


def test_sampler_cannot_inject_unobserved_optimum() -> None:
    probabilities = (1.0,) + (0.0,) * 31
    rows = sample(probabilities, shots=17, seed=0)
    assert len(rows) == 1 and rows[0].word == 0 and rows[0].cost == 0
    assert rows[0].count == 17 and cut_cost(classical_optima()[0]) < rows[0].cost


@pytest.mark.parametrize("limit", ("qubits", "operations", "memory"))
def test_resource_budget_refusals(limit: str) -> None:
    with pytest.raises(ValueError, match="budget|limit|memory|candidate"):
        synth.synthesize(
            main,
            classical_inputs={"params": BINDINGS[0]},
            max_qubits=4 if limit == "qubits" else 5,
            max_workspace=0,
            max_operations=1 if limit == "operations" else 2048,
            memory_limit_bytes=1 if limit == "memory" else 4 * 1024**2,
        )


@pytest.mark.parametrize("parameters", ((0.1,), (True,) * 8, (float("nan"),) * 8))
def test_invalid_classical_angles(parameters: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="eight finite radian"):
        build_program(parameters)


@pytest.mark.parametrize("budget", (0, 201, True))
def test_invalid_evaluation_budget(budget: int) -> None:
    with pytest.raises(ValueError, match="evaluation budget"):
        run_workflow(evaluation_budget=budget)


@pytest.mark.parametrize("word", (-1, 32, True))
def test_invalid_word(word: int) -> None:
    with pytest.raises(ValueError, match="0..31"):
        group_bits(word)


@pytest.mark.parametrize(
    "probabilities", ((1.0,), (float("nan"),) * 32, (-1.0, 2.0) + (0.0,) * 30)
)
def test_invalid_probability_values(probabilities: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="normalized probabilities"):
        mean_cost(probabilities)


@pytest.mark.parametrize("shots,seed", ((0, 0), (True, 0), (17, -1), (17, True)))
def test_invalid_sampling_controls(shots: int, seed: int) -> None:
    with pytest.raises(ValueError, match="shots|seed"):
        sample((1 / 32,) * 32, shots=shots, seed=seed)


def test_all_original_stages_and_frozen_parent() -> None:
    directory = (
        Path(__file__).resolve().parents[2]
        / "examples/synthesis/tutorials/workshops/combinatorial_workshop/combinatorial_qmod_workshop_for_maxcut/overview"
    )
    proof = json.loads((directory / "provenance.json").read_text())
    assert [row["index"] for row in proof["reviewed_code_cells"]] == [
        4,
        6,
        8,
        10,
        12,
        14,
        16,
        18,
        20,
        22,
        24,
        26,
        29,
        31,
        33,
        34,
        36,
        38,
    ]
    assert (
        hashlib.sha256((directory.parent / "main.py").read_bytes()).hexdigest()
        == proof["frozen_quantum_entry_sha256"]
    )
    n = json.loads((directory / "walkthrough.ipynb").read_text())
    code = "\n".join(
        "".join(c["source"]) for c in n["cells"] if c["cell_type"] == "code"
    )
    assert "classiq" not in code and ".qmod" not in code and "source_format" not in code
