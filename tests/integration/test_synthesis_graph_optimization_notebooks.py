"""Complete native graph optimization, independent phases and feasibility proofs."""

import cmath
import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.applications.optimization.max_clique.comparisons.circuit import (
    cost as clique_score,
)
from examples.synthesis.applications.optimization.max_clique.comparisons.circuit import (
    reference_circuit as clique_reference,
)
from examples.synthesis.applications.optimization.max_clique.main import (
    cost_layer as clique_cost_layer,
)
from examples.synthesis.applications.optimization.max_clique.overview.workflow import (
    build_program as build_clique,
)
from examples.synthesis.applications.optimization.max_clique.overview.workflow import (
    model_tables as clique_tables,
)
from examples.synthesis.applications.optimization.max_clique.overview.workflow import (
    run_workflow as run_clique,
)
from examples.synthesis.applications.optimization.max_clique.overview.workflow import (
    source_edges as clique_edges,
)
from examples.synthesis.applications.optimization.max_independent_set.comparisons.circuit import (
    cost as independent_score,
)
from examples.synthesis.applications.optimization.max_independent_set.comparisons.circuit import (
    reference_circuit as independent_reference,
)
from examples.synthesis.applications.optimization.max_independent_set.main import (
    cost_layer as independent_cost_layer,
)
from examples.synthesis.applications.optimization.max_independent_set.overview.workflow import (
    build_program as build_independent,
)
from examples.synthesis.applications.optimization.max_independent_set.overview.workflow import (
    model_tables as independent_tables,
)
from examples.synthesis.applications.optimization.max_independent_set.overview.workflow import (
    run_workflow as run_independent,
)
from examples.synthesis.applications.optimization.max_independent_set.overview.workflow import (
    source_edges as independent_edges,
)
from examples.synthesis.graph_optimization import (
    INITIAL_PARAMETERS,
    SubsetWorkflow,
    lower_tail_cost,
    read_distribution,
    run_subset_optimization,
    sample_subsets,
)
from flagquantum import synth
from flagquantum.compiler.resource_estimation import ResourceEstimate
from flagquantum.errors import PlanningError
from flagquantum.synth import Output, QArray, QBit, X, allocate, qfunc
from tests.integration.synthesis_notebook_support import execute_code_cells

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]


@qfunc
def indexed_node(index: int, v: Output[QArray[QBit]]) -> None:
    allocate(8, v)
    X(v[index])


@pytest.mark.parametrize("index", (0, 2, 5, 7))
def test_native_array_element_named_integer_and_node_label(index: int) -> None:
    program = synth.synthesize(indexed_node, classical_inputs={"index": index})
    costs, feasible = independent_tables()
    probabilities = read_distribution(program, 8)
    assert probabilities[1 << index] == pytest.approx(1.0)
    rows = sample_subsets(probabilities, costs, feasible, shots=10, seed=1)
    assert len(rows) == 1
    assert rows[0].word == 1 << index and rows[0].nodes == (index,)
    assert rows[0].feasible and rows[0].count == 10


@qfunc
def independent_phase(gamma: float, v: Output[QArray[QBit]]) -> None:
    allocate(8, v)
    independent_cost_layer(gamma, v)


@qfunc
def clique_phase(gamma: float, v: Output[QArray[QBit]]) -> None:
    allocate(7, v)
    clique_cost_layer(gamma, v)


@pytest.mark.parametrize("case", ("independent", "clique"))
@pytest.mark.parametrize("gamma", (0.19, -0.23))
def test_all_cost_basis_phases_and_classical_model(case: str, gamma: float) -> None:
    width = 8 if case == "independent" else 7
    probe = independent_phase if case == "independent" else clique_phase
    costs, feasible = independent_tables() if case == "independent" else clique_tables()
    score = independent_score if case == "independent" else clique_score
    program = synth.synthesize(
        probe,
        classical_inputs={"gamma": gamma},
        max_qubits=width,
        max_workspace=0,
        memory_limit_bytes=4 * 1024**2,
    )
    logical = tuple(program.registers["v"])
    assert program.circuit.n_wires == width and program.report["workspace_qubits"] == 0
    for word in range(1 << width):
        bits = tuple((word >> i) & 1 for i in range(width))
        assert costs[word] == -score(bits)
        prefix = fq.Circuit(width, dtype=torch.complex128)
        for i in range(width):
            if word & (1 << (width - 1 - i)):
                prefix.x(logical[i])
        ir = program.circuit
        actual = (
            fq.run(
                replace(ir, instructions=prefix.to_ir().instructions + ir.instructions)
            )
            .statevector()
            .reshape((2,) * width)
            .permute(logical)
            .reshape(1 << width)
        )
        expected = torch.zeros(1 << width, dtype=torch.complex128)
        expected[word] = torch.tensor(
            cmath.exp(-1j * gamma * costs[word]), dtype=torch.complex128
        )
        torch.testing.assert_close(actual, expected, atol=1e-11, rtol=0)
    assert 0 in feasible


@pytest.mark.parametrize("case", ("independent", "clique"))
@pytest.mark.parametrize(
    "parameters",
    ((0.13, -0.27, 0.19, 0.22, 0.31, -0.17), (-0.14, 0.21, -0.29, 0.36, -0.43, 0.51)),
)
def test_complete_coherent_states_named_order_and_costs(
    case: str, parameters: tuple[float, ...]
) -> None:
    width = 8 if case == "independent" else 7
    build = build_independent if case == "independent" else build_clique
    reference = independent_reference if case == "independent" else clique_reference
    program = build(parameters)
    logical = tuple(program.registers["v"])
    actual = (
        fq.run(program)
        .statevector()
        .reshape((2,) * width)
        .permute(logical)
        .reshape(1 << width)
    )
    expected = fq.run(reference(parameters)).statevector().flatten()
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=0)
    assert program.circuit.n_wires == width
    assert program.report["workspace_qubits"] == 0
    torch.testing.assert_close(
        torch.tensor(read_distribution(program, width), dtype=torch.float64),
        expected.abs().square(),
        atol=1e-11,
        rtol=0,
    )
    resources = program.report["resources"]
    assert isinstance(resources, ResourceEstimate)
    assert resources.n_operations == len(program.circuit.instructions)
    assert resources.n_operations == (90 if case == "independent" else 804)
    assert resources.depth == (26 if case == "independent" else 669)


@pytest.fixture(scope="module", params=("independent", "clique"))
def workflow(request: pytest.FixtureRequest) -> tuple[str, SubsetWorkflow]:
    case = str(request.param)
    return case, run_independent() if case == "independent" else run_clique()


def test_actual_optimizer_best_sample_exact_reference_and_nonconvergence(
    workflow: tuple[str, SubsetWorkflow],
) -> None:
    case, result = workflow
    assert result.optimizer.iterations == (60 if case == "independent" else 50)
    assert len(result.trace) == result.optimizer.evaluations > 1
    assert not result.optimizer.converged
    assert result.optimizer.value < result.trace[0].cost
    costs, feasible = independent_tables() if case == "independent" else clique_tables()
    probabilities = read_distribution(result.program, 8 if case == "independent" else 7)
    assert result.optimizer.value == pytest.approx(
        lower_tail_cost(probabilities, costs, 0.7), abs=1e-10
    )
    best = result.best_feasible
    assert best is not None and best.word in feasible
    assert (
        len(best.nodes)
        == len(result.exact_nodes)
        == (3 if case == "independent" else 4)
    )
    assert best.penalized_cost == min(costs[word] for word in feasible)
    assert (
        sum(row.count for row in result.optimized)
        == sum(row.count for row in result.uniform)
        == 1000
    )
    assert all(row.feasible == (row.word in feasible) for row in result.optimized)
    assert (
        sample_subsets(probabilities, costs, feasible, shots=1000, seed=2718)
        == result.optimized
    )
    assert (
        sample_subsets(probabilities, costs, feasible, shots=1000, seed=2719)
        != result.optimized
    )
    reference = independent_reference if case == "independent" else clique_reference
    for entry in (result.trace[0], result.trace[-1]):
        p = (
            fq.run(reference(entry.parameters))
            .statevector()
            .flatten()
            .abs()
            .square()
            .tolist()
        )
        assert entry.cost == pytest.approx(lower_tail_cost(p, costs, 0.7), abs=1e-10)


def test_all_source_figure_roles(workflow: tuple[str, SubsetWorkflow]) -> None:
    pytest.importorskip("matplotlib")
    pytest.importorskip("numpy")
    import matplotlib.pyplot as plt
    import numpy as np

    from examples.synthesis.graph_optimization_plot import plot_workflow

    case, result = workflow
    edges = independent_edges() if case == "independent" else clique_edges()
    figure = plot_workflow(result, edges, seed=12345, title="Native graph optimization")
    try:
        assert len(figure.axes) == 6
        figure.canvas.draw()
        trace = figure.axes[1].lines[0].get_ydata()
        assert np.array_equal(
            np.asarray(trace, dtype=np.float64),
            np.asarray([row.cost for row in result.trace], dtype=np.float64),
        )
        assert sum(row.count for row in result.optimized if row.feasible) > 0
    finally:
        plt.close(figure)


def test_penalized_minimum_is_not_feasibility_and_no_oracle_substitution(
    workflow: tuple[str, SubsetWorkflow],
) -> None:
    _, result = workflow
    costs, feasible = independent_tables()
    minimizers = {word for word, cost in enumerate(costs) if cost == min(costs)}
    assert len(minimizers & feasible) == 6
    assert len(minimizers - feasible) == 5
    invalid = min(minimizers - feasible)
    probabilities = tuple(float(word == invalid) for word in range(256))
    rows = sample_subsets(probabilities, costs, feasible, shots=10, seed=1)
    altered = replace(result, optimized=rows)
    assert altered.best_feasible is None
    assert len(altered.exact_nodes) > 0


@pytest.mark.parametrize(
    "parameters",
    ((0.1,), (math.nan, 0.0, 0.0, 0.0, 0.0, 0.0), (0.0, math.inf, 0.0, 0.0, 0.0, 0.0)),
)
@pytest.mark.parametrize("case", ("independent", "clique"))
def test_invalid_parameters_refuse(case: str, parameters: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="six finite"):
        (build_independent if case == "independent" else build_clique)(parameters)


@pytest.mark.parametrize("case", ("independent", "clique"))
@pytest.mark.parametrize("budget", ("width", "depth", "operations", "memory"))
def test_actual_resource_budget_refusals(case: str, budget: str) -> None:
    from examples.synthesis.applications.optimization.max_clique.main import (
        main as clique_main,
    )
    from examples.synthesis.applications.optimization.max_independent_set.main import (
        main as independent_main,
    )

    main = independent_main if case == "independent" else clique_main
    values = {"params": INITIAL_PARAMETERS}
    with pytest.raises((synth.SynthesisError, PlanningError)):
        if budget == "width":
            synth.synthesize(main, classical_inputs=values, max_qubits=6)
        elif budget == "depth":
            synth.synthesize(main, classical_inputs=values, max_depth=1)
        elif budget == "operations":
            synth.synthesize(main, classical_inputs=values, max_operations=1)
        else:
            synth.synthesize(main, classical_inputs=values, memory_limit_bytes=16)


@pytest.mark.parametrize("case", ("max_independent_set", "max_clique"))
def test_clean_notebooks_complete_actual_workflow(case: str) -> None:
    pytest.importorskip("matplotlib")
    pytest.importorskip("numpy")
    path = (
        ROOT
        / f"examples/synthesis/applications/optimization/{case}/overview/walkthrough.ipynb"
    )
    source = json.loads(path.read_text())
    assert all(
        c.get("execution_count") is None and not c.get("outputs")
        for c in source["cells"]
        if c["cell_type"] == "code"
    )
    namespace, count = execute_code_cells(path)
    assert count == 4
    result = namespace["workflow"]
    assert isinstance(result, SubsetWorkflow) and result.best_feasible is not None
    assert len(result.best_feasible.nodes) == (
        3 if case == "max_independent_set" else 4
    )


def test_fractional_boundary_and_full_expectation() -> None:
    assert lower_tail_cost((0.2, 0.5, 0.3), (-4.0, -1.0, 5.0), 0.4) == pytest.approx(
        -2.5
    )
    assert lower_tail_cost((0.2, 0.5, 0.3), (-4.0, -1.0, 5.0), 1.0) == pytest.approx(
        0.2
    )


def test_tied_cost_mass_order_is_irrelevant() -> None:
    assert lower_tail_cost((0.2, 0.5, 0.3), (-4.0, -4.0, 5.0), 0.4) == -4.0
    assert lower_tail_cost((0.5, 0.2, 0.3), (-4.0, -4.0, 5.0), 0.4) == -4.0


@pytest.mark.parametrize("quantile", (0.0, -0.2, 1.1, math.nan, math.inf))
def test_invalid_quantile_refuses(quantile: float) -> None:
    with pytest.raises(ValueError, match="quantile"):
        lower_tail_cost((1.0,), (0.0,), quantile)


@pytest.mark.parametrize(
    "probabilities,costs",
    (
        ((), ()),
        ((0.5,), (0.0,)),
        ((-1.0, 2.0), (0.0, 1.0)),
        ((math.nan,), (0.0,)),
        ((1.0,), (math.inf,)),
        ((1.0,), (0.0, 1.0)),
    ),
)
def test_invalid_distribution_refuses(
    probabilities: tuple[float, ...], costs: tuple[float, ...]
) -> None:
    with pytest.raises(ValueError):
        lower_tail_cost(probabilities, costs, 0.7)


@pytest.mark.parametrize(
    "shots,seed", ((0, 1), (100001, 1), (True, 1), (10, -1), (10, True))
)
def test_bad_sampling_controls_refuse(shots: int, seed: int) -> None:
    with pytest.raises(ValueError):
        sample_subsets(
            (0.5, 0.5), (0.0, -1.0), frozenset((0, 1)), shots=shots, seed=seed
        )


@pytest.mark.parametrize("maxiter", (0, 129, True))
def test_bad_optimizer_bound_refuses(maxiter: int) -> None:
    with pytest.raises(ValueError):
        run_subset_optimization(
            build_independent, (0.0, -1.0), frozenset((0, 1)), maxiter=maxiter
        )


def test_invalid_model_tables_refuse_before_execution() -> None:
    with pytest.raises(ValueError, match="power-of-two"):
        sample_subsets((1 / 3,) * 3, (0.0, 1.0, 2.0), frozenset((0,)), shots=1, seed=1)
    with pytest.raises(ValueError, match="indices"):
        sample_subsets((0.5, 0.5), (0.0, 1.0), frozenset((2,)), shots=1, seed=1)
    with pytest.raises(ValueError, match="at least one"):
        run_subset_optimization(build_independent, (0.0, -1.0), frozenset(), maxiter=1)
    with pytest.raises(ValueError, match="node array"):
        read_distribution(build_independent(INITIAL_PARAMETERS), 7)
