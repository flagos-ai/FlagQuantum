"""Original patching graph, independent quantum states and observed minimum covers."""

import hashlib
import json
from pathlib import Path

import pytest

from examples.synthesis.applications.cybersecurity.patching_management.patch_min_vertex_cover.comparisons.circuit import (
    BINDINGS,
    cost,
)
from examples.synthesis.applications.cybersecurity.patching_management.patch_min_vertex_cover.main import (
    main,
)
from examples.synthesis.applications.cybersecurity.patching_management.patch_min_vertex_cover.overview.operations import (
    verify,
)
from examples.synthesis.applications.cybersecurity.patching_management.patch_min_vertex_cover.overview.workflow import (
    build_program,
    dual_graph,
    kill_chain_graph,
    model_tables,
    remaining_graph,
    run_workflow,
    source_graph,
)
from examples.synthesis.graph_optimization import (
    SubsetWorkflow,
    lower_tail_cost,
    read_distribution,
)
from flagquantum import synth

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def workflow() -> SubsetWorkflow:
    return run_workflow(maxiter=3, shots=193, seed=1916)


@pytest.mark.parametrize("parameters", BINDINGS)
def test_absolute_state_and_clean_workspace(parameters: tuple[float, ...]) -> None:
    check = verify(parameters)
    assert check.maximum_state_error < 3e-11
    assert check.dirty_workspace_probability < 1e-22
    assert check.resources.n_qubits == 8


def test_original_graph_and_independently_enumerated_cover_costs() -> None:
    graph = source_graph()
    assert graph.number_of_nodes() == 15 and graph.number_of_edges() == 21
    assert set(graph.neighbors(0)) == {"A", "B", "D", "F"}
    assert set(graph.neighbors(7)) == {"C", "D", "G"}
    assert set(kill_chain_graph().edges()) == {
        (0, "B"),
        (0, "F"),
        (3, "F"),
        (3, "C"),
        (7, "C"),
    }
    dual = dual_graph()
    assert dual.number_of_nodes() == 8 and dual.number_of_edges() == 18
    costs, feasible = model_tables()
    for word in range(256):
        bits = tuple((word >> i) & 1 for i in range(8))
        assert costs[word] == -cost(bits)
        assert (word in feasible) == all(bits[i] or bits[j] for i, j in dual.edges())
    assert len(feasible) == 21
    assert min(costs[word] for word in feasible) == 5
    assert {word for word in feasible if costs[word] == 5} == {171, 227}


def test_real_trace_and_selected_program_cvar(workflow: SubsetWorkflow) -> None:
    assert workflow.optimizer.iterations == 3
    assert len(workflow.trace) == workflow.optimizer.evaluations
    probabilities = read_distribution(workflow.program, 8)
    costs, _ = model_tables()
    assert lower_tail_cost(probabilities, costs, 0.9) == pytest.approx(
        workflow.optimizer.value, abs=1e-10
    )
    assert all(len(entry.parameters) == 6 for entry in workflow.trace)


def test_minimum_observed_cover_and_actual_shots(workflow: SubsetWorkflow) -> None:
    for rows in (workflow.optimized, workflow.uniform):
        assert sum(row.count for row in rows) == 193
        assert all(row.count > 0 for row in rows)
        assert list(rows) == sorted(
            rows, key=lambda row: (not row.feasible, len(row.nodes), row.word)
        )
    best = workflow.best_feasible
    assert best is not None
    assert best == min(
        (row for row in workflow.optimized if row.feasible),
        key=lambda row: (len(row.nodes), row.word),
    )
    remaining = remaining_graph(best.nodes)
    assert all(remaining.degree(v) <= 1 for v in "ABCDEFG")
    assert set(remaining.nodes()) == set(source_graph().nodes()) - set(best.nodes)
    assert any(not row.feasible for row in workflow.optimized)


@pytest.mark.parametrize("nodes", ((0, 0), (-1,), (8,), (True,)))
def test_invalid_patch_sets(nodes: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="distinct resource nodes"):
        remaining_graph(nodes)


@pytest.mark.parametrize("parameters", ((0.1,), (float("nan"),) * 6, (True,) * 6))
def test_invalid_parameters(parameters: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="six finite"):
        build_program(parameters)


@pytest.mark.parametrize("limit", ("qubits", "operations", "memory"))
def test_resource_budget_refusals(limit: str) -> None:
    with pytest.raises(ValueError, match="budget|limit|memory|candidate"):
        synth.synthesize(
            main,
            classical_inputs={"params": BINDINGS[0]},
            max_qubits=7 if limit == "qubits" else 8,
            max_operations=1 if limit == "operations" else 4096,
            memory_limit_bytes=1 if limit == "memory" else 4 * 1024**2,
            max_workspace=0,
        )


def test_source_mapping_and_frozen_quantum_parent() -> None:
    directory = Path(__file__).resolve().parents[2] / (
        "examples/synthesis/applications/cybersecurity/patching_management/"
        "patch_min_vertex_cover/overview"
    )
    proof = json.loads((directory / "provenance.json").read_text())
    assert [cell["index"] for cell in proof["reviewed_code_cells"]] == [
        2,
        4,
        5,
        7,
        9,
        11,
        12,
        14,
        16,
        20,
        23,
        25,
        27,
        29,
        31,
        32,
        34,
    ]
    assert (
        hashlib.sha256((directory.parent / "main.py").read_bytes()).hexdigest()
        == proof["frozen_quantum_entry_sha256"]
    )
    notebook = json.loads((directory / "walkthrough.ipynb").read_text())
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert "classiq" not in code and ".qmod" not in code and "source_format" not in code
