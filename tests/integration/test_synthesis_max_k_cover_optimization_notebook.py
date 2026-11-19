"""Accept original maximum-k-cover data, absolute phases and actual local search."""

import cmath
import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.applications.optimization.max_k_vertex_cover.comparisons.circuit import (
    BINDINGS,
    reference_circuit,
)
from examples.synthesis.applications.optimization.max_k_vertex_cover.main import (
    cost_layer,
    main,
)
from examples.synthesis.applications.optimization.max_k_vertex_cover.overview.workflow import (
    Workflow,
    build_program,
    model_table,
    run_workflow,
    sample_distribution,
    source_edges,
)
from examples.synthesis.graph_optimization import (
    INITIAL_PARAMETERS,
    lower_tail_cost,
    read_distribution,
    selected_nodes,
)
from flagquantum import synth
from flagquantum.compiler.resource_estimation import ResourceEstimate
from flagquantum.errors import PlanningError
from flagquantum.synth import Output, QArray, QBit, X, allocate, qfunc
from tests.integration.synthesis_notebook_support import execute_code_cells

pytestmark = pytest.mark.integration
DIRECTORY = (
    Path(__file__).resolve().parents[2]
    / "examples/synthesis/applications/optimization/max_k_vertex_cover/overview"
)
EDGES = (
    (0, 1),
    (0, 5),
    (0, 6),
    (0, 7),
    (0, 8),
    (1, 2),
    (1, 4),
    (1, 5),
    (1, 6),
    (1, 9),
    (2, 3),
    (2, 4),
    (3, 6),
    (3, 8),
    (4, 6),
    (4, 7),
    (4, 8),
    (4, 9),
    (5, 6),
    (7, 8),
    (8, 9),
)


@pytest.fixture(autouse=True)
def graph_dependency() -> None:
    pytest.importorskip("networkx")


@qfunc
def indexed_node(index: int, v: Output[QArray[QBit]]) -> None:
    allocate(10, v)
    X(v[index])


@qfunc
def phase_probe(gamma: float, v: Output[QArray[QBit]]) -> None:
    allocate(10, v)
    cost_layer(gamma, v)


def test_source_graph_and_all1024_independent_objectives() -> None:
    assert source_edges() == EDGES and len(EDGES) == 21
    table = model_table()
    assert len(table) == 1024 and sum(row[2] for row in table) == 252
    for word, (uncovered, penalized, feasible) in enumerate(table):
        nodes = {i for i in range(10) if word & (1 << i)}
        raw = sum(not ({a, b} & nodes) for a, b in EDGES)
        assert uncovered == raw
        assert feasible == (len(nodes) == 5)
        assert penalized == raw + 10 * (len(nodes) - 5) ** 2
    assert min(raw for raw, _, feasible in table if feasible) == 1
    for count in (0, 4, 5, 6, 10):
        word = (1 << count) - 1
        raw, penalized, feasible = table[word]
        assert feasible == (count == 5)
        assert penalized - raw == 10 * (count - 5) ** 2


@pytest.mark.parametrize("index", range(10))
def test_named_node_order_and_actual_sampling(index: int) -> None:
    program = synth.synthesize(indexed_node, classical_inputs={"index": index})
    probabilities = read_distribution(program, 10)
    assert probabilities[1 << index] == pytest.approx(1.0)
    rows = sample_distribution(probabilities, shots=10, seed=1)
    assert len(rows) == 1
    row = rows[0]
    assert selected_nodes(row.word, 10) == (index,) and row.count == 10
    assert not row.feasible
    assert row.uncovered == sum(index not in edge for edge in EDGES)
    assert row.penalized_cost == row.uncovered + 160


@pytest.mark.parametrize("gamma", (0.019, -0.023))
def test_all1024_absolute_basis_phases(gamma: float) -> None:
    program = synth.synthesize(
        phase_probe, classical_inputs={"gamma": gamma}, max_qubits=10, max_workspace=0
    )
    wires = tuple(reversed(program.registers["v"]))
    assert program.circuit.n_wires == 10 and program.report["workspace_qubits"] == 0
    for word in range(1024):
        prefix = fq.Circuit(10, dtype=torch.complex128)
        for index, wire in enumerate(wires):
            if word & (1 << index):
                prefix.x(wire)
        body = program.circuit
        actual = (
            fq.run(
                replace(
                    body, instructions=prefix.to_ir().instructions + body.instructions
                )
            )
            .statevector()
            .flatten()
        )
        physical_word = sum(
            1 << (9 - wire) for index, wire in enumerate(wires) if word & (1 << index)
        )
        nodes = {i for i in range(10) if word & (1 << i)}
        cost = sum(not ({a, b} & nodes) for a, b in EDGES) + 10 * (len(nodes) - 5) ** 2
        expected = torch.zeros(1024, dtype=torch.complex128)
        expected[physical_word] = torch.tensor(
            cmath.exp(-1j * gamma * cost), dtype=torch.complex128
        )
        torch.testing.assert_close(actual, expected, atol=1e-11, rtol=0)


@pytest.mark.parametrize("parameters", BINDINGS)
def test_complete_complex_states_and_actual_resources(
    parameters: tuple[float, ...],
) -> None:
    program = build_program(parameters)
    actual = (
        fq.run(program)
        .statevector()
        .reshape((2,) * 10)
        .permute(program.registers["v"])
        .reshape(1024)
    )
    expected = fq.run(reference_circuit(parameters)).statevector().flatten()
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=0)
    torch.testing.assert_close(
        torch.tensor(read_distribution(program, 10), dtype=torch.float64),
        expected.abs().square(),
        atol=1e-11,
        rtol=0,
    )
    assert program.circuit.n_wires == 10 and program.report["workspace_qubits"] == 0
    resources = program.report["resources"]
    assert isinstance(resources, ResourceEstimate)
    assert resources.n_operations == len(program.circuit.instructions) == 196
    assert resources.depth == 43


@pytest.fixture(scope="module")
def workflow() -> Workflow:
    pytest.importorskip("matplotlib")
    pytest.importorskip("numpy")
    path = DIRECTORY / "walkthrough.ipynb"
    notebook = json.loads(path.read_text())
    assert all(
        cell.get("execution_count") is None and not cell.get("outputs")
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    namespace, count = execute_code_cells(path)
    assert count == 5
    result = namespace["workflow"]
    assert isinstance(result, Workflow) and result.best_feasible is not None
    assert result.best_feasible.uncovered == 1
    initial = namespace["initial_program"]
    assert isinstance(initial, synth.Program)
    assert read_distribution(initial, 10) == pytest.approx(
        read_distribution(build_program(), 10), abs=1e-11
    )
    assert namespace["exact_uncovered"] == 1 and namespace["exact_feasible"] is True

    return result


def test_actual_optimizer_and_samples_against_independent_reference(
    workflow: Workflow,
) -> None:
    assert workflow.optimizer.iterations == 90 and not workflow.optimizer.converged
    assert workflow.optimizer.evaluations == len(workflow.trace) == 151
    assert workflow.trace[0].parameters == INITIAL_PARAMETERS
    assert workflow.trace[0].cost == pytest.approx(23.04345912069062, abs=1e-10)
    assert workflow.optimizer.value == pytest.approx(4.197074119148964, abs=1e-10)
    assert sum(row.count for row in workflow.samples) == 1000
    assert sum(row.count for row in workflow.uniform) == 1000
    assert sum(row.count for row in workflow.samples if row.feasible) == 874
    assert sum(row.count for row in workflow.uniform if row.feasible) == 222
    best = workflow.best_feasible
    assert (
        best is not None
        and best.feasible
        and best.uncovered == best.penalized_cost == 1
    )
    assert selected_nodes(best.word, 10) == (0, 1, 3, 4, 8)
    assert model_table()[workflow.exact_reference_word] == (1, 1, True)
    assert (
        sample_distribution(
            read_distribution(workflow.program, 10), shots=1000, seed=2718
        )
        == workflow.samples
    )
    assert (
        sample_distribution(
            read_distribution(workflow.program, 10), shots=1000, seed=2719
        )
        != workflow.samples
    )
    independent_costs = tuple(
        float(
            sum(not (word & (1 << a)) and not (word & (1 << b)) for a, b in EDGES)
            + 10 * (word.bit_count() - 5) ** 2
        )
        for word in range(1024)
    )
    for entry in (workflow.trace[0], workflow.trace[-1]):
        probabilities = tuple(
            float(p)
            for p in fq.run(reference_circuit(entry.parameters))
            .statevector()
            .flatten()
            .abs()
            .square()
        )
        assert entry.cost == pytest.approx(
            lower_tail_cost(probabilities, independent_costs, 0.7), abs=1e-10
        )

    # Keep the full optimizer, missing-result, and plot checks in one test so
    # xdist does not repeat this module fixture in several worker processes.
    _check_missing_feasible_observation_stays_missing(workflow)
    _check_actual_figure_roles_and_missing_observation(workflow)


def _check_missing_feasible_observation_stays_missing(workflow: Workflow) -> None:
    forced = sample_distribution((1.0,) + (0.0,) * 1023, shots=10, seed=1)
    assert forced[0].word == 0 and forced[0].penalized_cost == 271
    assert replace(workflow, samples=forced).best_feasible is None
    assert model_table()[workflow.exact_reference_word][0] == 1


def _check_actual_figure_roles_and_missing_observation(workflow: Workflow) -> None:
    pytest.importorskip("matplotlib")
    np = pytest.importorskip("numpy")
    import matplotlib.pyplot as plt

    from examples.synthesis.applications.optimization.max_k_vertex_cover.overview.plots import (
        cover_figure,
    )

    figure = cover_figure(workflow)
    try:
        figure.canvas.draw()
        assert len(figure.axes) == 6
        assert np.array_equal(
            np.asarray(figure.axes[1].lines[0].get_ydata(), dtype=np.float64),
            np.asarray([entry.cost for entry in workflow.trace]),
        )
        assert len(figure.axes[2].containers) == 2
        for container in figure.axes[2].containers:
            assert sum(bar.get_height() for bar in container) == pytest.approx(1.0)
        assert "uncovered=1" in figure.axes[3].get_title()
        assert "Exact bounded classical reference" in figure.axes[4].get_title()
    finally:
        plt.close(figure)
    forced = sample_distribution((1.0,) + (0.0,) * 1023, shots=10, seed=1)
    missing = cover_figure(replace(workflow, samples=forced))
    try:
        missing.canvas.draw()
        assert not missing.axes[3].axison
        assert any(
            "No feasible assignment" in text.get_text()
            for text in missing.axes[3].texts
        )
        assert "uncovered=1" in missing.axes[4].get_title()
    finally:
        plt.close(missing)


@pytest.mark.parametrize("parameters", ((0.1,), (math.nan,) * 6, (math.inf,) * 6))
def test_invalid_angles_refuse(parameters: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="six finite"):
        build_program(parameters)


@pytest.mark.parametrize("value", (0, -1, 129, True))
def test_invalid_iterations_refuse(value: int) -> None:
    with pytest.raises(ValueError, match="maxiter"):
        run_workflow(maxiter=value)


@pytest.mark.parametrize(
    "probabilities", ((), (-1.0,) * 1024, (math.nan,) * 1024, (0.5,) * 1024)
)
def test_invalid_probabilities_refuse(probabilities: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        sample_distribution(probabilities, shots=10, seed=1)


@pytest.mark.parametrize(
    "shots,seed", ((0, 1), (100001, 1), (True, 1), (10, -1), (10, True), (10, 2**63))
)
def test_invalid_sampling_refuses_before_executing(shots: int, seed: int) -> None:
    with pytest.raises(ValueError, match="shots|seed"):
        run_workflow(shots=shots, seed=seed)


@pytest.mark.parametrize("budget", ("width", "depth", "operations", "memory"))
def test_resource_budget_refusals(budget: str) -> None:
    bindings = {"params": INITIAL_PARAMETERS}
    with pytest.raises((synth.SynthesisError, PlanningError)):
        if budget == "width":
            synth.synthesize(main, classical_inputs=bindings, max_qubits=9)
        elif budget == "depth":
            synth.synthesize(main, classical_inputs=bindings, max_depth=1)
        elif budget == "operations":
            synth.synthesize(main, classical_inputs=bindings, max_operations=1)
        else:
            synth.synthesize(main, classical_inputs=bindings, memory_limit_bytes=16)


def test_wrong_named_array_refuses() -> None:
    from examples.synthesis.algorithms.search_and_optimization.qaoa.overview.workflow import (
        build_program as other_program,
    )

    with pytest.raises(ValueError, match="declared v"):
        read_distribution(other_program("maxcut", (0.1,) * 8), 10)
