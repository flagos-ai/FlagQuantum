"""Accept both native QAOA sections with independent phases and actual workflows."""

import cmath
import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.algorithms.search_and_optimization.qaoa.comparisons.circuit import (
    reference_circuit as knapsack_reference,
)
from examples.synthesis.algorithms.search_and_optimization.qaoa.main import apply_cost
from examples.synthesis.algorithms.search_and_optimization.qaoa.overview.workflow import (
    Case,
    Workflow,
    assignments,
    build_program,
    classical_cost,
    distribution,
    initial_parameters,
    run_workflow,
    sample_distribution,
)
from examples.synthesis.algorithms.search_and_optimization.qaoa.qaoa_max_cut.comparisons.circuit import (
    reference_circuit as maxcut_reference,
)
from examples.synthesis.algorithms.search_and_optimization.qaoa.qaoa_max_cut.main import (
    maxcut_cost_layer,
)
from flagquantum import synth
from flagquantum.compiler.resource_estimation import ResourceEstimate
from flagquantum.errors import PlanningError
from flagquantum.synth import Output, QArray, QBit, QNum, X, allocate, qfunc
from tests.integration.synthesis_notebook_support import execute_code_cells

pytestmark = pytest.mark.integration
DIRECTORY = (
    Path(__file__).resolve().parents[2]
    / "examples/synthesis/algorithms/search_and_optimization/qaoa/overview"
)
CASES: tuple[Case, ...] = ("maxcut", "knapsack")


@qfunc
def indexed_node(index: int, v: Output[QArray[QBit]]) -> None:
    allocate(5, v)
    X(v[index])


@qfunc
def maxcut_phase(gamma: float, v: Output[QArray[QBit]]) -> None:
    allocate(5, v)
    maxcut_cost_layer(gamma, v)


@qfunc
def knapsack_phase(gamma: float, a: Output[QNum], b: Output[QNum]) -> None:
    allocate(3, a)
    allocate(2, b)
    apply_cost(gamma, a, b)


@pytest.mark.parametrize("index", range(5))
def test_named_array_bit_order(index: int) -> None:
    program = synth.synthesize(indexed_node, classical_inputs={"index": index})
    probabilities = distribution("maxcut", program)
    assert probabilities[1 << index] == pytest.approx(1.0)
    row = sample_distribution("maxcut", probabilities, shots=10, seed=1)[0]
    assert row.assignment == (1 << index,) and row.count == 10
    edges = ((0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4))
    assert row.cost == -sum(index in edge for edge in edges) / 6


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("gamma", (0.19, -0.23))
def test_all32_basis_absolute_phases(case: Case, gamma: float) -> None:
    program = synth.synthesize(
        maxcut_phase if case == "maxcut" else knapsack_phase,
        classical_inputs={"gamma": gamma},
        max_qubits=5,
        max_workspace=0,
    )
    logical = tuple(wire for wires in program.registers.values() for wire in wires)
    assert program.circuit.n_wires == 5 and program.report["workspace_qubits"] == 0
    edges = ((0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4))
    for word, assignment in enumerate(assignments(case)):
        if case == "maxcut":
            cost = (
                -sum(bool(word & (1 << a)) != bool(word & (1 << b)) for a, b in edges)
                / 6
            )
            feasible = True
            phase_cost = -cost
        else:
            a, b = assignment
            feasible = 2 * a + 3 * b <= 12
            cost = float(-3 * a - 5 * b) if feasible else 0.0
            phase_cost = cost
        assert classical_cost(case, assignment) == (cost, feasible)
        prefix = fq.Circuit(5, dtype=torch.complex128)
        for index in range(5):
            if word & (1 << (4 - index)):
                prefix.x(logical[index])
        body = program.circuit
        actual = (
            fq.run(
                replace(
                    body, instructions=prefix.to_ir().instructions + body.instructions
                )
            )
            .statevector()
            .reshape((2,) * 5)
            .permute(logical)
            .reshape(32)
        )
        expected = torch.zeros(32, dtype=torch.complex128)
        expected[word] = torch.tensor(
            cmath.exp(1j * gamma * phase_cost), dtype=torch.complex128
        )
        torch.testing.assert_close(actual, expected, atol=1e-11, rtol=0)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("scale", (0.11, -0.17))
def test_complete_coherent_states_and_resources(case: Case, scale: float) -> None:
    parameters = tuple(
        scale * (i + 1) * (-1 if i % 2 else 1)
        for i in range(8 if case == "maxcut" else 6)
    )
    program = build_program(case, parameters)
    logical = tuple(wire for wires in program.registers.values() for wire in wires)
    reference = maxcut_reference if case == "maxcut" else knapsack_reference
    actual = (
        fq.run(program).statevector().reshape((2,) * 5).permute(logical).reshape(32)
    )
    expected = fq.run(reference(parameters)).statevector().flatten()
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=0)
    torch.testing.assert_close(
        torch.tensor(distribution(case, program), dtype=torch.float64),
        expected.abs().square(),
        atol=1e-11,
        rtol=0,
    )
    assert program.circuit.n_wires == 5 and program.report["workspace_qubits"] == 0
    resources = program.report["resources"]
    assert isinstance(resources, ResourceEstimate)
    assert resources.n_operations == len(program.circuit.instructions)
    assert resources.n_operations <= 2000


@pytest.fixture(scope="module", params=CASES)
def workflow(request: pytest.FixtureRequest) -> Workflow:
    case: Case = "maxcut" if request.param == "maxcut" else "knapsack"
    return run_workflow(case)


def test_actual_bounded_optimizer_samples_and_independent_cost(
    workflow: Workflow,
) -> None:
    result = workflow
    assert result.optimizer.iterations == 60 and not result.optimizer.converged
    assert len(result.trace) == result.optimizer.evaluations > 60
    assert result.trace[0].parameters == initial_parameters(result.case)
    assert result.optimizer.value < result.trace[0].cost
    assert sum(row.count for row in result.samples) == 1000
    assert result.best_feasible is not None
    domain = assignments(result.case)
    exact = min(
        classical_cost(result.case, key)[0]
        for key in domain
        if classical_cost(result.case, key)[1]
    )
    assert result.best_feasible.cost == exact
    assert exact == (-5 / 6 if result.case == "maxcut" else -19)
    probabilities = distribution(result.case, result.program)
    assert (
        sample_distribution(result.case, probabilities, shots=1000, seed=2718)
        == result.samples
    )
    assert (
        sample_distribution(result.case, probabilities, shots=1000, seed=2719)
        != result.samples
    )
    reference = maxcut_reference if result.case == "maxcut" else knapsack_reference
    for entry in (result.trace[0], result.trace[-1]):
        probabilities_reference = (
            fq.run(reference(entry.parameters))
            .statevector()
            .flatten()
            .abs()
            .square()
            .tolist()
        )
        measured_cost = sum(
            prob * classical_cost(result.case, key)[0]
            for prob, key in zip(probabilities_reference, domain, strict=True)
        )
        assert entry.cost == pytest.approx(measured_cost, abs=1e-10)
    if result.case == "knapsack":
        invalid = tuple(row for row in result.samples if not row.feasible)
        assert invalid
        assert replace(result, samples=invalid).best_feasible is None
        assert classical_cost("knapsack", (3, 2)) == (-19, True)
        assert classical_cost("knapsack", (4, 2)) == (0, False)


def test_actual_source_figure_roles(workflow: Workflow) -> None:
    pytest.importorskip("matplotlib")
    pytest.importorskip("numpy")
    import matplotlib.pyplot as plt
    import numpy as np

    from examples.synthesis.algorithms.search_and_optimization.qaoa.overview.plots import (
        knapsack_figure,
        maxcut_figure,
    )

    figure = (
        maxcut_figure(workflow)
        if workflow.case == "maxcut"
        else knapsack_figure(workflow)
    )
    try:
        figure.canvas.draw()
        axis = figure.axes[1 if workflow.case == "maxcut" else 0]
        assert np.array_equal(
            np.asarray(axis.lines[0].get_ydata(), dtype=np.float64),
            np.asarray([entry.cost for entry in workflow.trace], dtype=np.float64),
        )
        if workflow.case == "maxcut":
            best = workflow.best_feasible
            assert best is not None
            count = sum(abs(row.cost - best.cost) < 1e-3 for row in workflow.samples)
            assert len(figure.axes) == count + 2
        else:
            assert len(figure.axes) == 2 and len(figure.axes[1].tables) == 1
    finally:
        plt.close(figure)


def test_clean_notebook_executes_both_sections() -> None:
    pytest.importorskip("matplotlib")
    pytest.importorskip("numpy")
    path = DIRECTORY / "walkthrough.ipynb"
    source = json.loads(path.read_text())
    assert all(
        c.get("execution_count") is None and not c.get("outputs")
        for c in source["cells"]
        if c["cell_type"] == "code"
    )
    namespace, count = execute_code_cells(path)
    assert count == 5
    for case in CASES:
        result = namespace[f"{case}_workflow"]
        initial = namespace[f"{case}_initial"]
        assert isinstance(result, Workflow) and result.best_feasible is not None
        assert isinstance(initial, synth.Program)
        assert distribution(case, initial) == pytest.approx(
            distribution(case, build_program(case, initial_parameters(case))), abs=1e-11
        )
        assert initial.report["workspace_qubits"] == 0
        assert result.best_feasible.cost == (-5 / 6 if case == "maxcut" else -19)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("value", (0, -1, 129, True))
def test_invalid_iterations_refuse(case: Case, value: int) -> None:
    with pytest.raises(ValueError, match="maxiter"):
        run_workflow(case, maxiter=value)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("budget", ("width", "depth", "operations", "memory"))
def test_actual_budget_refusals(case: Case, budget: str) -> None:
    from examples.synthesis.algorithms.search_and_optimization.qaoa.main import (
        main as knapsack,
    )
    from examples.synthesis.algorithms.search_and_optimization.qaoa.qaoa_max_cut.main import (
        main as maxcut,
    )

    main = maxcut if case == "maxcut" else knapsack
    bindings = {"params": initial_parameters(case)}
    with pytest.raises((synth.SynthesisError, PlanningError)):
        if budget == "width":
            synth.synthesize(main, classical_inputs=bindings, max_qubits=4)
        elif budget == "depth":
            synth.synthesize(main, classical_inputs=bindings, max_depth=1)
        elif budget == "operations":
            synth.synthesize(main, classical_inputs=bindings, max_operations=1)
        else:
            synth.synthesize(main, classical_inputs=bindings, memory_limit_bytes=16)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("invalid", ("length", "nan", "infinity"))
def test_invalid_parameters_refuse(case: Case, invalid: str) -> None:
    values = (
        (0.1,)
        if invalid == "length"
        else (math.nan if invalid == "nan" else math.inf,)
        * len(initial_parameters(case))
    )
    with pytest.raises(ValueError, match="finite radian"):
        build_program(case, values)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize(
    "probabilities", ((), (-1.0,) * 32, (math.nan,) * 32, (0.5,) * 32)
)
def test_invalid_distribution_refuses(
    case: Case, probabilities: tuple[float, ...]
) -> None:
    with pytest.raises(ValueError, match="32 finite"):
        sample_distribution(case, probabilities, shots=100, seed=1)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize(
    "shots,seed", ((0, 1), (100001, 1), (True, 1), (10, -1), (10, True), (10, 2**63))
)
def test_invalid_sampling_refuses_before_quantum_calls(
    case: Case, shots: int, seed: int
) -> None:
    with pytest.raises(ValueError, match="shots|seed"):
        run_workflow(case, shots=shots, seed=seed)


@pytest.mark.parametrize(
    "case,word",
    (
        ("maxcut", (-1,)),
        ("maxcut", (32,)),
        ("maxcut", (True,)),
        ("knapsack", (8, 0)),
        ("knapsack", (0, 4)),
        ("knapsack", (-1, 0)),
    ),
)
def test_invalid_named_assignment_refuses(case: Case, word: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="five-bit domain"):
        classical_cost(case, word)


def test_wrong_named_program_refuses() -> None:
    program = build_program("maxcut", initial_parameters("maxcut"))
    with pytest.raises(ValueError, match="registers"):
        distribution("knapsack", program)
