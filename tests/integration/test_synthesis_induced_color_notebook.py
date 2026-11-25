"""Accept original induced coloring semantics and the native Notebook workflow."""

import cmath
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.applications.optimization.max_induced_k_color_subgraph.comparisons.circuit import (
    BINDINGS,
    graph_cost,
    reference_circuit,
)
from examples.synthesis.applications.optimization.max_induced_k_color_subgraph.main import (
    DEFAULT_PARAMS,
    cost,
    main,
)
from examples.synthesis.applications.optimization.max_induced_k_color_subgraph.overview.workflow import (
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
    / "examples/synthesis/applications/optimization/max_induced_k_color_subgraph/overview"
)


@qfunc
def indexed_node(index: int, a: Output[QArray[QBit]], b: Output[QArray[QBit]]) -> None:
    allocate(6, a)
    allocate(6, b)
    if index < 6:
        X(a[index])
    else:
        X(b[index - 6])


@qfunc
def phase_probe(gamma: float, a: Output[QArray[QBit]], b: Output[QArray[QBit]]) -> None:
    allocate(6, a)
    allocate(6, b)
    hadamard_transform(a)
    hadamard_transform(b)
    cost(gamma, a, b)


def test_original_graph_constraints_and_all4096_joint_costs() -> None:
    import networkx as nx

    graph = nx.erdos_renyi_graph(6, 0.5, seed=7)
    assert graph.number_of_edges() == 11
    assert len(DOMAIN) == 4096
    for assignment in DOMAIN:
        row = model_row(assignment)
        a, b = assignment
        bits = tuple((word >> i) & 1 for word in (a, b) for i in range(6))
        selected = {i for i in graph if bits[i] or bits[i + 6]}
        conflicts = sum(
            bits[offset + i] * bits[offset + j]
            for offset in (0, 6)
            for i, j in graph.edges()
        )
        double = sum(bits[i] * bits[i + 6] for i in graph)
        assert row.colored_nodes == tuple(sorted(selected))
        assert row.objective == len(selected)
        assert row.directed_conflicts == 2 * conflicts
        assert row.double_colored == double
        assert row.feasible == (double == 0 and conflicts == 0)
        assert row.retained_quantum_cost == -graph_cost(bits)
    assert sum(model_row(key).feasible for key in DOMAIN) == 99
    assert model_row((0, 0)).feasible
    assert not model_row((1, 1)).feasible
    assert not model_row((3, 0)).feasible
    assert model_row((18, 41)).objective == 5


@pytest.mark.parametrize("index", range(12))
def test_numeric_bit_is_node_in_each_named_color_word(index: int) -> None:
    program = synth.synthesize(indexed_node, classical_inputs={"index": index})
    distribution = read_distribution(program)
    expected = (1 << index, 0) if index < 6 else (0, 1 << (index - 6))
    assert distribution[DOMAIN.index(expected)] == pytest.approx(1.0)
    rows = sample_distribution(distribution, shots=10, seed=1)
    assert len(rows) == 1 and rows[0].assignment == expected
    assert rows[0].colored_nodes == (index % 6,) and rows[0].count == 10


@pytest.mark.parametrize("gamma", (0.019, -0.023))
def test_all4096_absolute_cost_layer_amplitudes(gamma: float) -> None:
    program = synth.synthesize(
        phase_probe,
        classical_inputs={"gamma": gamma},
        max_qubits=12,
        max_workspace=0,
        max_operations=40000,
        memory_limit_bytes=16 * 1024**2,
    )
    axes = tuple(w for wires in program.registers.values() for w in wires)
    actual = fq.run(program).statevector().reshape((2,) * 12).permute(axes)
    # Tensor axes are MSB first while numeric bit i denotes node i.
    expected = torch.tensor(
        [
            cmath.exp(
                1j
                * gamma
                * graph_cost(
                    tuple((word >> i) & 1 for word in (a, b) for i in range(6))
                )
            )
            / 64
            for a, b in DOMAIN
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
    actual = fq.run(program).statevector().reshape((2,) * 12).permute(axes).flatten()
    expected = fq.run(reference_circuit(parameters)).statevector().flatten()
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=0)
    torch.testing.assert_close(
        torch.tensor(read_distribution(program), dtype=torch.float64),
        expected.abs().square(),
        atol=1e-11,
        rtol=0,
    )
    assert program.circuit.n_wires == 12 and program.report["workspace_qubits"] == 0
    resources = program.report["resources"]
    assert isinstance(resources, ResourceEstimate)
    assert resources.n_operations == len(program.circuit.instructions) == 34185
    assert resources.depth == 23477


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
    assert isinstance(readout, dict) and len(readout) == 4096
    return result


@pytest.mark.slow
def test_actual_fifty_iteration_optimization_and_sampling(workflow: Workflow) -> None:
    assert workflow.optimizer.iterations == 50 and not workflow.optimizer.converged
    assert workflow.optimizer.evaluations == len(workflow.trace) == 94
    assert workflow.trace[0].parameters == DEFAULT_PARAMS
    assert workflow.trace[0].cost == pytest.approx(140.60813182851996, abs=1e-10)
    assert workflow.optimizer.value == pytest.approx(46.46207037849267, abs=1e-10)
    assert sum(row.count for row in workflow.samples) == 1000
    assert sum(row.count for row in workflow.uniform) == 1000
    assert sum(row.count for row in workflow.samples if row.feasible) == 34
    assert sum(row.count for row in workflow.uniform if row.feasible) == 32
    best = workflow.best_feasible
    assert best is not None and best.assignment == (2, 41) and best.count == 2
    assert best.feasible and best.objective == 4 and best.retained_quantum_cost == -4
    assert workflow.exact_reference == (18, 41)
    assert model_row(workflow.exact_reference).objective == 5
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
        # amplify tiny accumulated phase-roundoff errors across eight layers.
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
        tuple(1.0 if key == (63, 63) else 0.0 for key in DOMAIN), shots=10, seed=1
    )
    assert not forced[0].feasible
    assert replace(workflow, samples=forced).best_feasible is None
    assert workflow.exact_reference == (18, 41)


def _check_figures_use_actual_trace_and_observations(workflow: Workflow) -> None:
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Rectangle

    from examples.synthesis.applications.optimization.max_induced_k_color_subgraph.overview.plots import (
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
        tuple(1.0 if key == (63, 63) else 0.0 for key in DOMAIN), shots=10, seed=1
    )
    figure = optimization_figure(replace(workflow, samples=forced))
    try:
        figure.canvas.draw()
        assert any("No feasible" in text.get_text() for text in figure.axes[3].texts)
    finally:
        plt.close(figure)


@pytest.mark.parametrize("assignment", ((64, 0), (-1, 0), (0, True), (0, 0, 0)))
def test_invalid_assignment_refused(assignment: Assignment) -> None:
    with pytest.raises(ValueError, match="assignment requires"):
        model_row(assignment)


@pytest.mark.parametrize("count", (-1, True))
def test_invalid_count_refused(count: int) -> None:
    with pytest.raises(ValueError, match="count"):
        model_row((0, 1), count)


@pytest.mark.parametrize(
    "parameters", ((0.1,), (float("nan"),) * 16, (float("inf"),) * 16, (True,) * 16)
)
def test_invalid_angles_refused(parameters: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="sixteen finite"):
        build_program(parameters)


@pytest.mark.parametrize("maxiter", (0, 129, True))
def test_invalid_iteration_budget_refused(maxiter: int) -> None:
    with pytest.raises(ValueError, match="maxiter"):
        run_workflow(maxiter=maxiter)


@pytest.mark.parametrize(
    "probabilities",
    ((1.0,), (0.0,) * 4096, (-1.0, 2.0) + (0.0,) * 4094, (float("nan"),) * 4096),
)
def test_invalid_probabilities_refused(probabilities: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        sample_distribution(probabilities, shots=10, seed=1)


@pytest.mark.parametrize(
    "shots,seed", ((0, 1), (100001, 1), (True, 1), (10, -1), (10, True), (10, 2**63))
)
def test_invalid_sampling_controls_refused(shots: int, seed: int) -> None:
    with pytest.raises(ValueError):
        sample_distribution((1 / 4096,) * 4096, shots=shots, seed=seed)
    with pytest.raises(ValueError):
        run_workflow(shots=shots, seed=seed)


@qfunc
def wrong_ports(q: Output[QBit]) -> None:
    allocate(q)


def test_wrong_named_readout_refused() -> None:
    with pytest.raises(ValueError, match="registers"):
        read_distribution(synth.synthesize(wrong_ports))


@pytest.mark.parametrize("budget", ("width", "depth", "operations", "memory"))
def test_resource_budget_refusals(budget: str) -> None:
    with pytest.raises((synth.SynthesisError, PlanningError)):
        if budget == "width":
            synth.synthesize(
                main, classical_inputs={"params": DEFAULT_PARAMS}, max_qubits=11
            )
        elif budget == "depth":
            synth.synthesize(
                main, classical_inputs={"params": DEFAULT_PARAMS}, max_depth=1
            )
        elif budget == "operations":
            synth.synthesize(
                main, classical_inputs={"params": DEFAULT_PARAMS}, max_operations=20000
            )
        else:
            synth.synthesize(
                main, classical_inputs={"params": DEFAULT_PARAMS}, memory_limit_bytes=1
            )
