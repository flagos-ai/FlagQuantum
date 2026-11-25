"""Accept the original integer model, retained joint quantum encoding and workflow."""

import cmath
import itertools
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.applications.optimization.integer_linear_programming.comparisons.circuit import (
    BINDINGS,
    reference_circuit,
)
from examples.synthesis.applications.optimization.integer_linear_programming.main import (
    main,
)
from examples.synthesis.applications.optimization.integer_linear_programming.overview.workflow import (
    DOMAIN,
    NAMES,
    SOURCE_A,
    SOURCE_B,
    SOURCE_C,
    Assignment,
    Workflow,
    build_program,
    model_row,
    read_distribution,
    run_workflow,
    sample_distribution,
)
from examples.synthesis.graph_optimization import INITIAL_PARAMETERS, lower_tail_cost
from flagquantum import synth
from flagquantum.compiler.resource_estimation import ResourceEstimate
from flagquantum.errors import PlanningError
from flagquantum.synth import Output, QBit, QNum, X, allocate, phase, qfunc
from tests.integration.synthesis_notebook_support import execute_code_cells

pytestmark = pytest.mark.integration
DIRECTORY = (
    Path(__file__).resolve().parents[2]
    / "examples/synthesis/applications/optimization/integer_linear_programming/overview"
)


@qfunc
def indexed_variable(
    index: int,
    x0: Output[QNum],
    x1: Output[QNum],
    x2: Output[QNum],
    s0: Output[QBit],
    s1: Output[QBit],
) -> None:
    allocate(2, x0)
    allocate(2, x1)
    allocate(2, x2)
    allocate(s0)
    allocate(s1)
    if index < 2:
        X(x0[index])
    elif index < 4:
        X(x1[index - 2])
    elif index < 6:
        X(x2[index - 4])
    elif index == 6:
        X(s0)
    else:
        X(s1)


@qfunc
def phase_probe(
    gamma: float,
    x0: Output[QNum],
    x1: Output[QNum],
    x2: Output[QNum],
    s0: Output[QBit],
    s1: Output[QBit],
) -> None:
    allocate(2, x0)
    allocate(2, x1)
    allocate(2, x2)
    allocate(s0)
    allocate(s1)
    phase(
        -(
            10 * x0 * x1
            + 10 * x0 * x2
            - x0
            + 10 * x1 * x2
            - 2 * x1
            - 3 * x2
            + 10 * (s0 + x0 + x1 + x2 - 1.0) ** 2
            + 10 * (s1 + x0 + x1 + x2 - 1.0) ** 2
        ),
        gamma,
    )


def test_original_model_and_retained_joint_cost_are_separate() -> None:
    assert SOURCE_A == ((1, 1, 1), (2, 2, 2), (3, 3, 3))
    assert SOURCE_B == (1, 2, 3) and SOURCE_C == (1, 2, 3)
    assert len(DOMAIN) == 256
    primary = tuple(itertools.product(range(4), repeat=3))
    assert len(primary) == 64
    assert sum(model_row((a, b, c, 0, 0)).feasible for a, b, c in primary) == 4
    for assignment in DOMAIN:
        row = model_row(assignment)
        a, b, c, slack0, slack1 = assignment
        lhs = tuple(
            sum(
                weight * value
                for weight, value in zip(coefficients, assignment[:3], strict=True)
            )
            for coefficients in ((1, 1, 1), (2, 2, 2), (3, 3, 3))
        )
        objective = a + 2 * b + 3 * c
        assert row.objective == objective and row.lhs == lhs
        assert row.feasible == all(
            value <= bound for value, bound in zip(lhs, (1, 2, 3), strict=True)
        )
        residuals = (slack0 + a + b + c - 1, slack1 + a + b + c - 1)
        assert row.retained_quantum_cost == (
            10 * sum((a * b, a * c, b * c))
            - objective
            + 10 * sum(value * value for value in residuals)
        )
    assert model_row((0, 0, 1, 0, 0)).feasible
    assert not model_row((0, 0, 2, 0, 0)).feasible
    assert model_row((0, 0, 1, 1, 1)).objective == 3
    assert model_row((0, 0, 1, 1, 1)).retained_quantum_cost == 17


@pytest.mark.parametrize("index", range(8))
def test_named_integer_bits_and_both_slack_ports(index: int) -> None:
    program = synth.synthesize(indexed_variable, classical_inputs={"index": index})
    probabilities = read_distribution(program)
    values = [0, 0, 0, 0, 0]
    if index < 6:
        values[index // 2] = 1 << (index % 2)
    else:
        values[index - 3] = 1
    expected = (values[0], values[1], values[2], values[3], values[4])
    assert sum(probabilities) == pytest.approx(1.0)
    assert probabilities[DOMAIN.index(expected)] == pytest.approx(1.0)
    samples = sample_distribution(probabilities, shots=10, seed=1)
    assert len(samples) == 1 and samples[0].assignment == expected
    assert samples[0].count == 10


@pytest.mark.parametrize("gamma", (0.019, -0.023))
def test_all256_joint_absolute_phases(gamma: float) -> None:
    program = synth.synthesize(
        phase_probe, classical_inputs={"gamma": gamma}, max_qubits=8, max_workspace=0
    )
    for assignment in DOMAIN:
        prefix = fq.Circuit(8, dtype=torch.complex128)
        physical_word = 0
        for name, value in zip(NAMES, assignment, strict=True):
            for bit, wire in enumerate(reversed(program.registers[name])):
                if value & (1 << bit):
                    prefix.x(wire)
                    physical_word += 1 << (7 - wire)
        actual = (
            fq.run(
                replace(
                    program.circuit,
                    instructions=prefix.to_ir().instructions
                    + program.circuit.instructions,
                )
            )
            .statevector()
            .flatten()
        )
        a, b, c, slack0, slack1 = assignment
        residuals = (slack0 + a + b + c - 1, slack1 + a + b + c - 1)
        cost = (
            10 * (a * b + a * c + b * c)
            - a
            - 2 * b
            - 3 * c
            + 10 * sum(value * value for value in residuals)
        )
        expected = torch.zeros(256, dtype=torch.complex128)
        expected[physical_word] = torch.tensor(
            cmath.exp(-1j * gamma * cost), dtype=torch.complex128
        )
        torch.testing.assert_close(actual, expected, atol=1e-11, rtol=0)


@pytest.mark.parametrize("parameters", BINDINGS)
def test_full_complex_states_readout_and_modeled_slack(
    parameters: tuple[float, ...],
) -> None:
    program = build_program(parameters)
    axes = tuple(wire for wires in program.registers.values() for wire in wires)
    actual = fq.run(program).statevector().reshape((2,) * 8).permute(axes).reshape(256)
    expected = fq.run(reference_circuit(parameters)).statevector().flatten()
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=0)
    probabilities = read_distribution(program)
    torch.testing.assert_close(
        torch.tensor(probabilities, dtype=torch.float64),
        expected.abs().square(),
        atol=1e-11,
        rtol=0,
    )
    assert (
        sum(
            p for key, p in zip(DOMAIN, probabilities, strict=True) if key[3:] != (0, 0)
        )
        > 0.1
    )
    assert program.circuit.n_wires == 8 and program.report["workspace_qubits"] == 0
    resources = program.report["resources"]
    assert isinstance(resources, ResourceEstimate)
    assert resources.n_operations == len(program.circuit.instructions) == 120
    assert resources.depth == 35


@pytest.fixture(scope="module")
def workflow() -> Workflow:
    return run_workflow()


def test_actual_optimization_sampling_and_original_reference(
    workflow: Workflow,
) -> None:
    assert workflow.optimizer.iterations == 90 and not workflow.optimizer.converged
    assert workflow.optimizer.evaluations == len(workflow.trace) == 171
    assert workflow.trace[0].parameters == INITIAL_PARAMETERS
    assert workflow.trace[0].cost == pytest.approx(235.10526438117788, abs=1e-10)
    assert workflow.optimizer.value == pytest.approx(166.86343149349483, abs=1e-10)
    assert sum(row.count for row in workflow.samples) == 1000
    assert sum(row.count for row in workflow.uniform) == 1000
    assert sum(row.count for row in workflow.samples if row.feasible) == 63
    assert sum(row.count for row in workflow.uniform if row.feasible) == 60
    best = workflow.best_feasible
    assert best is not None and best.assignment == (0, 0, 1, 0, 0)
    assert best.feasible and best.objective == 3 and best.retained_quantum_cost == -3
    assert best.lhs == (1, 2, 3) and best.count == 4
    assert workflow.exact_reference == (0, 0, 1)
    assert (
        sample_distribution(read_distribution(workflow.program), shots=1000, seed=2718)
        == workflow.samples
    )
    assert (
        sample_distribution(read_distribution(workflow.program), shots=1000, seed=2719)
        != workflow.samples
    )
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
        assert entry.cost == pytest.approx(
            lower_tail_cost(probabilities, costs, 0.7), abs=1e-10
        )


def test_missing_feasible_observation_stays_missing(workflow: Workflow) -> None:
    forced = sample_distribution(
        tuple(1.0 if key == (3, 3, 3, 1, 1) else 0.0 for key in DOMAIN),
        shots=10,
        seed=1,
    )
    assert not forced[0].feasible
    assert replace(workflow, samples=forced).best_feasible is None
    assert workflow.exact_reference == (0, 0, 1)


def test_actual_plots_and_missing_observation(workflow: Workflow) -> None:
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Rectangle

    from examples.synthesis.applications.optimization.integer_linear_programming.overview.plots import (
        optimization_figure,
    )

    figure = optimization_figure(workflow)
    try:
        figure.canvas.draw()
        assert len(figure.axes) == 4
        assert np.array_equal(
            figure.axes[0].lines[0].get_ydata(), [row.cost for row in workflow.trace]
        )
        assert len(figure.axes[1].patches) == 60
        assert all(
            isinstance(patch, Rectangle)
            for patch in (*figure.axes[1].patches, *figure.axes[2].patches)
        )
        assert sum(
            patch.get_height()
            for patch in figure.axes[1].patches
            if isinstance(patch, Rectangle)
        ) == pytest.approx(2.0)
        assert tuple(
            patch.get_height()
            for patch in figure.axes[2].patches
            if isinstance(patch, Rectangle)
        ) == (
            0,
            0,
            1,
        )
    finally:
        plt.close(figure)
    forced = sample_distribution(
        tuple(1.0 if key == (3, 3, 3, 1, 1) else 0.0 for key in DOMAIN),
        shots=10,
        seed=1,
    )
    figure = optimization_figure(replace(workflow, samples=forced))
    try:
        figure.canvas.draw()
        assert any("No feasible" in text.get_text() for text in figure.axes[2].texts)
    finally:
        plt.close(figure)


def test_clean_native_notebook_in_a_fresh_namespace() -> None:
    pytest.importorskip("matplotlib")
    notebook = json.loads((DIRECTORY / "walkthrough.ipynb").read_text())
    assert all(
        cell.get("execution_count") is None and not cell.get("outputs")
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    namespace, count = execute_code_cells(DIRECTORY / "walkthrough.ipynb")
    readout = namespace["initial_readout"]
    program = namespace["initial_program"]
    workflow = namespace["workflow"]
    assert count == 5 and isinstance(readout, dict) and len(readout) == 256
    assert (
        isinstance(program, synth.Program)
        and program.circuit == build_program().circuit
    )
    assert isinstance(workflow, Workflow) and workflow.best_feasible is not None
    assert workflow.best_feasible.objective == 3


@pytest.mark.parametrize(
    "assignment",
    (
        (4, 0, 0, 0, 0),
        (0, -1, 0, 0, 0),
        (0, 0, 0, 2, 0),
        (0, 0, 0, 0, True),
        (0, 0, 0, 0),
    ),
)
def test_invalid_assignment_refused(assignment: Assignment) -> None:
    with pytest.raises(ValueError, match="assignment requires"):
        model_row(assignment)


@pytest.mark.parametrize("count", (-1, True))
def test_invalid_count_refused(count: int) -> None:
    with pytest.raises(ValueError, match="count"):
        model_row((0, 0, 1, 0, 0), count)


@pytest.mark.parametrize(
    "parameters", ((0.1,), (0.1, 0.2, 0.3, 0.4, 0.5, float("nan")))
)
def test_invalid_angles_refused(parameters: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="six finite"):
        build_program(parameters)


@pytest.mark.parametrize("maxiter", (0, 129, True))
def test_invalid_iteration_budget_refused(maxiter: int) -> None:
    with pytest.raises(ValueError, match="maxiter"):
        run_workflow(maxiter=maxiter)


@pytest.mark.parametrize(
    "probabilities",
    ((1.0,), (0.0,) * 256, (-1.0, 2.0) + (0.0,) * 254, (float("nan"),) * 256),
)
def test_invalid_probabilities_refused(probabilities: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        sample_distribution(probabilities, shots=10, seed=1)


@pytest.mark.parametrize(
    "shots,seed", ((0, 1), (100001, 1), (True, 1), (10, -1), (10, True), (10, 2**63))
)
def test_invalid_sampling_controls_refused(shots: int, seed: int) -> None:
    with pytest.raises(ValueError):
        sample_distribution((1 / 256,) * 256, shots=shots, seed=seed)
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
                main, classical_inputs={"params": INITIAL_PARAMETERS}, max_qubits=7
            )
        elif budget == "depth":
            synth.synthesize(
                main, classical_inputs={"params": INITIAL_PARAMETERS}, max_depth=1
            )
        elif budget == "operations":
            synth.synthesize(
                main, classical_inputs={"params": INITIAL_PARAMETERS}, max_operations=1
            )
        else:
            synth.synthesize(
                main,
                classical_inputs={"params": INITIAL_PARAMETERS},
                memory_limit_bytes=1,
            )
