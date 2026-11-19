"""Fixed resource node, independent cover states and actual link-monitoring search."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

import flagquantum as fq
from examples.synthesis.applications.cybersecurity.link_monitoring.comparisons.circuit import (
    TERMS,
)
from examples.synthesis.applications.cybersecurity.link_monitoring.main import (
    build_program as frozen_program,
)
from examples.synthesis.applications.cybersecurity.link_monitoring.overview.quantum import (
    main,
)
from examples.synthesis.applications.cybersecurity.link_monitoring.overview.reference import (
    verify,
)
from examples.synthesis.applications.cybersecurity.link_monitoring.overview.workflow import (
    Workflow,
    build_program,
    classical_optima,
    model_tables,
    plot_graph,
    read_distribution,
    run_workflow,
    sample,
    selected_nodes,
    source_graph,
)
from examples.synthesis.graph_optimization import lower_tail_cost
from flagquantum import synth

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def workflow() -> Workflow:
    return run_workflow(maxiter=3, shots=193, seed=1960)


@pytest.mark.parametrize("parameters", ((0.013, 0.026), (-0.019, 0.046)))
def test_absolute_amplitudes_bit_order_and_fixed_node(
    parameters: tuple[float, ...],
) -> None:
    check = verify(parameters)
    assert check.maximum_state_error < 3e-11
    assert check.dirty_workspace_probability < 1e-22
    assert check.fixed_node_violation_probability < 1e-22
    assert check.resources.n_qubits == 12


def test_original_graph_all_costs_and_fixed_node_classical_covers() -> None:
    graph = source_graph()
    assert set(graph.nodes) == set(range(1, 13))
    assert graph.number_of_edges() == 13
    assert set(graph.neighbors(1)) == {2, 3, 4, 5}
    assert set(graph.neighbors(3)) == {1, 2, 7, 12}
    costs, feasible = model_tables()
    assert len(costs) == 4096 and len(feasible) == 318
    for word, cost in enumerate(costs):
        chosen = selected_nodes(word)
        uncovered = sum(u not in chosen and v not in chosen for u, v in graph.edges())
        assert cost == len(chosen) + 2 * uncovered
        frozen_cost = sum(
            coefficient
            * (-1 if sum((word >> (11 - i)) & 1 for i in support) % 2 else 1)
            for support, coefficient in TERMS
        )
        assert cost == frozen_cost
        assert (word in feasible) == (1 in chosen and uncovered == 0)
    optima = classical_optima()
    assert all(len(nodes) == 5 and 1 in nodes for nodes in optima)
    assert min(len(selected_nodes(word)) for word in feasible) == 5


def test_frozen_unfixed_export_does_not_replace_original_notebook_constraint() -> None:
    program = frozen_program((0.013, 0.026))
    probabilities = dict(
        program.probabilities(fq.run(program), registers=("target",), threshold=0)
    )
    assert sum(
        p for (word,), p in probabilities.items() if not word & 1
    ) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="must remain fixed"):
        read_distribution(program)


def test_real_search_trace_selected_cvar_and_full_selected_state(
    workflow: Workflow,
) -> None:
    assert workflow.optimizer.iterations == 3
    assert len(workflow.trace) == workflow.optimizer.evaluations
    costs, _ = model_tables()
    assert lower_tail_cost(
        read_distribution(workflow.program), costs, 0.9
    ) == pytest.approx(workflow.optimizer.value, abs=1e-10)
    assert (
        verify(
            tuple(float(v) for v in workflow.optimizer.parameters)
        ).maximum_state_error
        < 3e-11
    )


def test_actual_samples_keep_fixed_node_and_rank_observed_minimum(
    workflow: Workflow,
) -> None:
    assert sum(row.count for row in workflow.samples) == 193
    assert all(row.count > 0 and 1 in row.nodes for row in workflow.samples)
    assert list(workflow.samples) == sorted(
        workflow.samples, key=lambda row: (not row.feasible, len(row.nodes), row.word)
    )
    feasible = [row for row in workflow.samples if row.feasible]
    assert workflow.best_feasible == (
        min(feasible, key=lambda row: (len(row.nodes), row.word)) if feasible else None
    )


def test_actual_sampling_never_injects_classical_optimal_cover() -> None:
    probabilities = tuple(1.0 if word == 4095 else 0.0 for word in range(4096))
    observed = sample(probabilities, shots=17, seed=0)
    assert len(observed) == 1
    assert observed[0].nodes == tuple(range(1, 13)) and observed[0].count == 17
    assert len(observed[0].nodes) > len(classical_optima()[0])


def test_sampler_rejects_distributions_that_violate_fixed_node() -> None:
    with pytest.raises(ValueError, match="must remain fixed"):
        sample((1 / 4096,) * 4096, shots=17, seed=0)


@pytest.mark.parametrize("parameters", ((0.1,), (True, 0.1), (float("nan"), 0.1)))
def test_invalid_classical_angles(parameters: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="two finite radian"):
        build_program(parameters)


@pytest.mark.parametrize("limit", ("qubits", "operations", "memory"))
def test_resource_budget_refusals(limit: str) -> None:
    with pytest.raises(ValueError, match="budget|limit|memory|candidate"):
        synth.synthesize(
            main,
            classical_inputs={"parameters": (0.013, 0.026)},
            max_qubits=11 if limit == "qubits" else 12,
            max_workspace=0,
            max_operations=1 if limit == "operations" else 4096,
            memory_limit_bytes=1 if limit == "memory" else 64 * 1024**2,
        )


@pytest.mark.parametrize("word", (-1, 4096, True))
def test_invalid_readout_word(word: int) -> None:
    with pytest.raises(ValueError, match="0..4095"):
        selected_nodes(word)


@pytest.mark.parametrize("shots,seed", ((0, 0), (True, 0), (17, -1), (17, True)))
def test_invalid_sampling_controls(shots: int, seed: int) -> None:
    probabilities = tuple(1.0 if word == 4095 else 0.0 for word in range(4096))
    with pytest.raises(ValueError, match="shots|seed"):
        sample(probabilities, shots=shots, seed=seed)


@pytest.mark.parametrize("nodes", ((1, 1), (0,), (13,), (True,)))
def test_invalid_solution_graph_nodes(
    nodes: tuple[int, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    with pytest.raises(ValueError, match="distinct nodes"):
        plot_graph(nodes)


def test_source_cells_and_frozen_parent_provenance() -> None:
    directory = (
        Path(__file__).resolve().parents[2]
        / "examples/synthesis/applications/cybersecurity/link_monitoring/overview"
    )
    proof = json.loads((directory / "provenance.json").read_text())
    assert [row["index"] for row in proof["reviewed_code_cells"]] == [
        7,
        9,
        11,
        13,
        14,
        16,
        18,
        21,
        25,
        27,
        29,
        31,
        33,
        35,
        36,
        37,
        38,
    ]
    assert (
        hashlib.sha256((directory.parent / "main.py").read_bytes()).hexdigest()
        == proof["frozen_quantum_entry_sha256"]
    )
    notebook = json.loads((directory / "walkthrough.ipynb").read_text())
    code = "\n".join(
        "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"
    )
    assert "classiq" not in code and ".qmod" not in code and "source_format" not in code
