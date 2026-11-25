"""Accept original minimum coloring semantics and the native Notebook workflow."""

import cmath
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.applications.optimization.min_graph_coloring.comparisons.circuit import (
    BINDINGS,
    reference_circuit,
)
from examples.synthesis.applications.optimization.min_graph_coloring.comparisons.circuit import (
    cost as reference_cost,
)
from examples.synthesis.applications.optimization.min_graph_coloring.main import (
    DEFAULT_PARAMS,
    cost,
    main,
)
from examples.synthesis.applications.optimization.min_graph_coloring.overview.workflow import (
    DOMAIN,
    Assignment,
    Workflow,
    build_program,
    model_row,
    read_distribution,
    run_workflow,
    sample_distribution,
)
from examples.synthesis.graph_optimization import lower_tail_cost
from flagquantum import synth
from flagquantum.compiler.resource_estimation import ResourceEstimate
from flagquantum.errors import PlanningError
from flagquantum.synth import (
    Output,
    QArray,
    QBit,
    X,
    allocate,
    hadamard_transform,
    qfunc,
)
from tests.integration.synthesis_notebook_support import execute_code_cells

pytestmark = pytest.mark.integration
DIRECTORY = (
    Path(__file__).resolve().parents[2]
    / "examples/synthesis/applications/optimization/min_graph_coloring/overview"
)


@qfunc
def indexed_node(index: int, colors: Output[QArray[QBit]]) -> None:
    allocate(15, colors)
    X(colors[index])


@qfunc
def phase_probe(gamma: float, colors: Output[QArray[QBit]]) -> None:
    allocate(15, colors)
    hadamard_transform(colors)
    cost(gamma, colors)


def test_original_graph_constraints_and_all32768_joint_costs() -> None:
    import networkx as nx

    graph = nx.erdos_renyi_graph(5, 0.3, seed=79)
    assert set(graph.edges()) == {(0, 1), (0, 3), (0, 4), (1, 3), (2, 3)}
    assert max(dict(graph.degree()).values()) == 3
    assert len(DOMAIN) == 32768
    for assignment in DOMAIN:
        row = model_row(assignment)
        word = assignment[0]
        bits = tuple((word >> i) & 1 for i in range(15))
        counts = tuple(sum(bits[5 * c + v] for c in range(3)) for v in graph)
        conflicts = sum(
            bits[5 * c + i] * bits[5 * c + j]
            for c in range(3)
            for i, j in graph.edges()
        )
        assert row.node_colors == tuple(
            tuple(c for c in range(3) if bits[5 * c + v]) for v in graph
        )
        assert row.objective == sum(
            any(bits[5 * c + v] for v in graph) for c in range(3)
        )
        assert row.directed_conflicts == 2 * conflicts
        assert row.occupancy_penalty == sum((n - 1) ** 2 for n in counts)
        assert row.feasible == (all(n == 1 for n in counts) and conflicts == 0)
        assert row.retained_quantum_cost == -reference_cost(bits)
    assert sum(model_row(key).occupancy_penalty == 0 for key in DOMAIN) == 243
    assert sum(model_row(key).feasible for key in DOMAIN) == 24
    assert model_row((0,)).retained_quantum_cost == 50
    assert model_row((32767,)).retained_quantum_cost == 9203
    assert not model_row((0,)).feasible
    assert (
        min(model_row(key).objective for key in DOMAIN if model_row(key).feasible) == 3
    )


@pytest.mark.parametrize("index", range(15))
def test_numeric_bit_is_vertex_color_pair(index: int) -> None:
    program = synth.synthesize(indexed_node, classical_inputs={"index": index})
    distribution = read_distribution(program)
    expected = (1 << index,)
    assert distribution[DOMAIN.index(expected)] == pytest.approx(1.0)
    rows = sample_distribution(distribution, shots=10, seed=1)
    assert len(rows) == 1 and rows[0].assignment == expected and rows[0].count == 10
    assert rows[0].node_colors[index % 5] == (index // 5,)
    assert sum(len(colors) for colors in rows[0].node_colors) == 1


@pytest.mark.parametrize("gamma", (0.019, -0.023))
def test_all32768_absolute_cost_layer_amplitudes(gamma: float) -> None:
    program = synth.synthesize(
        phase_probe,
        classical_inputs={"gamma": gamma},
        max_qubits=15,
        max_workspace=0,
        max_operations=40000,
        memory_limit_bytes=32 * 1024**2,
    )
    actual = (
        fq.run(program)
        .statevector()
        .reshape((2,) * 15)
        .permute(program.registers["colors"])
    )
    expected = torch.tensor(
        [
            cmath.exp(-1j * gamma * model_row(key).retained_quantum_cost) / (32768**0.5)
            for key in DOMAIN
        ],
        dtype=torch.complex128,
    )
    torch.testing.assert_close(actual.flatten(), expected, atol=1e-11, rtol=0)


@pytest.mark.parametrize("parameters", BINDINGS)
def test_complete_complex_states_and_named_readout(
    parameters: tuple[float, ...],
) -> None:
    program = build_program(parameters)
    axes = tuple(w for wires in program.registers.values() for w in wires)
    actual = fq.run(program).statevector().reshape((2,) * 15).permute(axes).flatten()
    expected = fq.run(reference_circuit(parameters)).statevector().flatten()
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=0)
    torch.testing.assert_close(
        torch.tensor(read_distribution(program), dtype=torch.float64),
        expected.abs().square(),
        atol=1e-11,
        rtol=0,
    )
    assert program.circuit.n_wires == 15 and program.report["workspace_qubits"] == 0
    resources = program.report["resources"]
    assert isinstance(resources, ResourceEstimate)
    assert resources.n_operations == len(program.circuit.instructions) == 19975
    assert resources.depth == 13069


@pytest.fixture(scope="module")
def workflow() -> Workflow:
    pytest.importorskip("matplotlib")
    notebook = json.loads((DIRECTORY / "walkthrough.ipynb").read_text())
    assert all(
        cell.get("execution_count") is None and not cell.get("outputs")
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    namespace, count = execute_code_cells(DIRECTORY / "walkthrough.ipynb")
    result = namespace["workflow"]
    readout = namespace["initial_readout"]
    assert count == 5 and isinstance(result, Workflow)
    assert isinstance(readout, dict) and len(readout) == 32768
    return result


@pytest.mark.slow
def test_actual_hundred_iteration_optimization_and_sampling(workflow: Workflow) -> None:
    assert workflow.optimizer.iterations == 100 or workflow.optimizer.converged
    assert workflow.optimizer.evaluations == len(workflow.trace)
    assert workflow.optimizer.evaluations >= 13
    assert workflow.trace[0].parameters == DEFAULT_PARAMS
    assert workflow.optimizer.value <= workflow.trace[0].cost
    assert sum(row.count for row in workflow.samples) == 1000
    assert sum(row.count for row in workflow.uniform) == 1000
    for row in (*workflow.samples, *workflow.uniform):
        assert row == model_row(row.assignment, row.count)
    if workflow.best_feasible is not None:
        assert workflow.best_feasible.feasible
        assert workflow.best_feasible.objective == 3
    assert model_row(workflow.exact_reference).feasible
    assert model_row(workflow.exact_reference).objective == 3
    costs = tuple(model_row(key).retained_quantum_cost for key in DOMAIN)
    for entry in (workflow.trace[0], workflow.trace[-1]):
        probabilities = tuple(
            float(p)
            for p in fq.run(reference_circuit(entry.parameters))
            .statevector()
            .flatten()
            .abs()
            .square()
        )
        native_probabilities = read_distribution(build_program(entry.parameters))
        probability_l1 = sum(
            abs(native - reference)
            for native, reference in zip(
                native_probabilities, probabilities, strict=True
            )
        )
        assert probability_l1 < 1e-10
        # Lower-tail CVaR is Lipschitz in probability mass. Large penalty values
        # amplify tiny accumulated phase-roundoff errors across six layers.
        cost_error_bound = (max(costs) - min(costs)) * probability_l1 / 0.7
        assert entry.cost == pytest.approx(
            lower_tail_cost(probabilities, costs, 0.7),
            abs=cost_error_bound + 1e-11,
        )
    probabilities = read_distribution(workflow.program)
    assert sample_distribution(probabilities, shots=1000, seed=2718) == workflow.samples
    assert sample_distribution(probabilities, shots=1000, seed=2719) != workflow.samples

    # Keep the full optimizer, missing-result, and plot checks in one test so
    # xdist does not repeat this module fixture in several worker processes.
    _check_missing_feasible_observation_stays_missing(workflow)
    _check_figures_use_actual_trace_and_observations(workflow)


def _check_missing_feasible_observation_stays_missing(workflow: Workflow) -> None:
    forced = sample_distribution(
        tuple(1.0 if key == (32767,) else 0.0 for key in DOMAIN), shots=10, seed=1
    )
    assert not forced[0].feasible
    assert replace(workflow, samples=forced).best_feasible is None
    assert model_row(workflow.exact_reference).objective == 3


def _check_figures_use_actual_trace_and_observations(workflow: Workflow) -> None:
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Rectangle

    from examples.synthesis.applications.optimization.min_graph_coloring.overview.plots import (
        optimization_figure,
    )

    figure = optimization_figure(workflow)
    try:
        figure.canvas.draw()
        assert len(figure.axes) == 4
        assert np.array_equal(
            figure.axes[1].lines[0].get_ydata(), [row.cost for row in workflow.trace]
        )
        patches = figure.axes[2].patches
        assert len(patches) == 100
        assert sum(
            p.get_height() for p in patches if isinstance(p, Rectangle)
        ) == pytest.approx(2.0)
    finally:
        plt.close(figure)
    forced = sample_distribution(
        tuple(1.0 if key == (32767,) else 0.0 for key in DOMAIN), shots=10, seed=1
    )
    figure = optimization_figure(replace(workflow, samples=forced))
    try:
        figure.canvas.draw()
        assert any("No feasible" in text.get_text() for text in figure.axes[3].texts)
    finally:
        plt.close(figure)


@pytest.mark.parametrize("assignment", ((32768,), (-1,), (True,), (0, 0)))
def test_invalid_assignment_refused(assignment: Assignment) -> None:
    with pytest.raises(ValueError, match="assignment requires"):
        model_row(assignment)


@pytest.mark.parametrize("count", (-1, True))
def test_invalid_count_refused(count: int) -> None:
    with pytest.raises(ValueError, match="count"):
        model_row((0,), count)


@pytest.mark.parametrize(
    "parameters", ((0.1,), (float("nan"),) * 12, (float("inf"),) * 12, (True,) * 12)
)
def test_invalid_angles_refused(parameters: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="twelve finite"):
        build_program(parameters)


@pytest.mark.parametrize("maxiter", (0, 129, True))
def test_invalid_iteration_budget_refused(maxiter: int) -> None:
    with pytest.raises(ValueError, match="maxiter"):
        run_workflow(maxiter=maxiter)


@pytest.mark.parametrize(
    "probabilities",
    ((1.0,), (0.0,) * 32768, (-1.0, 2.0) + (0.0,) * 32766, (float("nan"),) * 32768),
)
def test_invalid_probabilities_refused(probabilities: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        sample_distribution(probabilities, shots=10, seed=1)


@pytest.mark.parametrize(
    "shots,seed", ((0, 1), (100001, 1), (True, 1), (10, -1), (10, True), (10, 2**63))
)
def test_invalid_sampling_controls_refused(shots: int, seed: int) -> None:
    with pytest.raises(ValueError):
        sample_distribution((1 / 32768,) * 32768, shots=shots, seed=seed)
    with pytest.raises(ValueError):
        run_workflow(shots=shots, seed=seed)


@qfunc
def wrong_ports(q: Output[QBit]) -> None:
    allocate(q)


def test_wrong_named_readout_refused() -> None:
    with pytest.raises(ValueError, match="register"):
        read_distribution(synth.synthesize(wrong_ports))


@pytest.mark.parametrize("budget", ("width", "depth", "operations", "memory"))
def test_resource_budget_refusals(budget: str) -> None:
    with pytest.raises((synth.SynthesisError, PlanningError)):
        if budget == "width":
            synth.synthesize(
                main, classical_inputs={"params": DEFAULT_PARAMS}, max_qubits=14
            )
        elif budget == "depth":
            synth.synthesize(
                main, classical_inputs={"params": DEFAULT_PARAMS}, max_depth=1
            )
        elif budget == "operations":
            synth.synthesize(
                main, classical_inputs={"params": DEFAULT_PARAMS}, max_operations=10000
            )
        else:
            synth.synthesize(
                main, classical_inputs={"params": DEFAULT_PARAMS}, memory_limit_bytes=1
            )
