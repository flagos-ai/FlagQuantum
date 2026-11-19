"""Independent absolute phases, complete states and actual integer optimization."""

import cmath
import json
from dataclasses import replace
from pathlib import Path
from typing import Annotated

import pytest
import torch

import flagquantum as fq
from examples.synthesis.tutorials.basics.optimization.learning_optimization.overview.comparisons.circuit import (
    reference_circuit,
)
from examples.synthesis.tutorials.basics.optimization.learning_optimization.overview.program import (
    build_program,
    main,
    objective_layer,
)
from examples.synthesis.tutorials.basics.optimization.learning_optimization.overview.workflow import (
    Workflow,
    run_workflow,
    sample_distribution,
)
from flagquantum import synth
from flagquantum.errors import PlanningError
from flagquantum.synth import Output, QNum, allocate, qfunc
from tests.integration.synthesis_notebook_support import execute_code_cells

pytestmark = pytest.mark.integration
DIRECTORY = (
    Path(__file__).resolve().parents[2]
    / "examples/synthesis/tutorials/basics/optimization/learning_optimization/overview"
)


@qfunc
def phase_probe(
    gamma: float, x: Output[Annotated[QNum, 2]], y: Output[Annotated[QNum, 2]]
) -> None:
    allocate(x)
    allocate(y)
    objective_layer(gamma, x, y)


@pytest.mark.parametrize("gamma", (0.0, 0.19, -0.23, 3.141592653589793))
def test_all16_basis_absolute_phases(gamma: float) -> None:
    program = synth.synthesize(
        phase_probe,
        classical_inputs={"gamma": gamma},
        max_qubits=4,
        max_workspace=0,
        memory_limit_bytes=1024**2,
    )
    logical = tuple(w for name in ("x", "y") for w in program.registers[name])
    for x in range(4):
        for y in range(4):
            prefix = fq.Circuit(4, dtype=torch.complex128)
            word = 4 * x + y
            for wire in range(4):
                if word & (1 << (3 - wire)):
                    prefix.x(logical[wire])
            body = program.circuit
            state = (
                fq.run(
                    replace(
                        body,
                        instructions=prefix.to_ir().instructions + body.instructions,
                    )
                )
                .statevector()
                .reshape((2,) * 4)
                .permute(logical)
                .reshape(16)
            )
            expected = torch.zeros(16, dtype=torch.complex128)
            violation = max(0, 2 - 3 * x - y)
            cost = 3 * x + 2 * y + 20 * violation**2
            expected[word] = torch.tensor(
                cmath.exp(-1j * gamma * cost), dtype=torch.complex128
            )
            torch.testing.assert_close(state, expected, atol=1e-11, rtol=0)


@pytest.mark.parametrize("budget", ("width", "depth", "operations", "memory"))
def test_budgets_refuse(budget: str) -> None:
    values = {"gamma": 0.19, "beta": 0.37}
    with pytest.raises((synth.SynthesisError, PlanningError)):
        if budget == "width":
            synth.synthesize(main, classical_inputs=values, max_qubits=3)
        elif budget == "depth":
            synth.synthesize(main, classical_inputs=values, max_depth=1)
        elif budget == "operations":
            synth.synthesize(main, classical_inputs=values, max_operations=1)
        else:
            synth.synthesize(main, classical_inputs=values, memory_limit_bytes=16)


@pytest.mark.parametrize("angles", ((0.19, 0.37), (-0.23, 0.41)))
def test_full_complex_state_and_clean_workspace(angles: tuple[float, float]) -> None:
    program = build_program(*angles)
    logical = tuple(w for name in ("x", "y") for w in program.registers[name])
    assert program.circuit.n_wires == len(logical) == 4
    assert program.report["workspace_qubits"] == 0
    actual = (
        fq.run(program).statevector().reshape((2,) * 4).permute(logical).reshape(16)
    )
    expected = fq.run(reference_circuit(*angles)).statevector().flatten()
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=0)
    probabilities = dict(program.probabilities(fq.run(program), registers=("x", "y")))
    for word, p in enumerate(expected.abs().square()):
        assert probabilities.get(divmod(word, 4), 0.0) == pytest.approx(
            float(p), abs=1e-12
        )


@pytest.fixture(scope="module")
def workflow() -> Workflow:
    return run_workflow()


def test_actual_bounded_optimizer_and_independent_feasible_optimum(
    workflow: Workflow,
) -> None:
    assert len(workflow.trace) == workflow.optimizer.evaluations > 1
    assert workflow.optimizer.iterations == 32
    assert not workflow.optimizer.converged
    assert workflow.optimizer.value < workflow.trace[0].expectation
    gamma, beta = (float(v) for v in workflow.optimizer.parameters)
    final_state = fq.run(reference_circuit(gamma, beta)).statevector().flatten()
    final_expected = sum(
        float(p) * (3 * x + 2 * y + 20 * max(0, 2 - 3 * x - y) ** 2)
        for word, p in enumerate(final_state.abs().square())
        for x, y in (divmod(word, 4),)
    )
    assert workflow.optimizer.value == pytest.approx(final_expected, abs=1e-11)
    for evaluation in (workflow.trace[0], workflow.trace[-1]):
        state = (
            fq.run(reference_circuit(evaluation.gamma, evaluation.beta))
            .statevector()
            .flatten()
        )
        expected = sum(
            float(p) * (3 * x + 2 * y + 20 * max(0, 2 - 3 * x - y) ** 2)
            for word, p in enumerate(state.abs().square())
            for x, y in (divmod(word, 4),)
        )
        assert evaluation.expectation == pytest.approx(expected, abs=1e-11)
    feasible = [
        (3 * x + 2 * y, x, y) for x in range(4) for y in range(4) if 3 * x + y >= 2
    ]
    cost, x, y = min(feasible)
    best = workflow.best_feasible
    assert best is not None
    assert (best.cost, best.x, best.y) == (cost, x, y) == (3, 1, 0)
    assert sum(row.count for row in workflow.samples) == 1000
    assert all(row.feasible == (3 * row.x + row.y >= 2) for row in workflow.samples)
    probabilities = dict(
        workflow.program.probabilities(fq.run(workflow.program), registers=("x", "y"))
    )
    assert sample_distribution(probabilities, shots=1000, seed=2718) == workflow.samples
    assert sample_distribution(probabilities, shots=1000, seed=2719) != workflow.samples
    assert workflow.optimizer.parameters.numel() == 2


@pytest.mark.parametrize("angles", ((float("nan"), 0.2), (0.1, float("inf"))))
def test_nonfinite_angles_refuse(angles: tuple[float, float]) -> None:
    with pytest.raises(ValueError, match="finite"):
        build_program(*angles)


@pytest.mark.parametrize(
    "shots,seed", ((0, 1), (100001, 1), (True, 1), (10, -1), (10, True))
)
def test_sampling_controls_refuse(shots: int, seed: int) -> None:
    with pytest.raises(ValueError):
        run_workflow(shots=shots, seed=seed)


@pytest.mark.parametrize("maxiter", (0, 129, True))
def test_optimizer_bound_refuses(maxiter: int) -> None:
    with pytest.raises(ValueError, match="maxiter"):
        run_workflow(maxiter=maxiter)


@pytest.mark.parametrize(
    "probabilities",
    ({(0, 0): -1.0}, {(0, 0): float("nan")}, {(4, 0): 1.0}, {(0,): 1.0}, {(0, 0): 0.5}),
)
def test_invalid_distributions_refuse(
    probabilities: dict[tuple[int, ...], float],
) -> None:
    with pytest.raises(ValueError):
        sample_distribution(probabilities, shots=1, seed=1)


def test_no_feasible_sample_is_reported_honestly() -> None:
    rows = sample_distribution({(0, 0): 1.0}, shots=10, seed=1)
    assert len(rows) == 1 and not rows[0].feasible and rows[0].count == 10


def test_clean_notebook_runs_actual_workflow() -> None:
    notebook = json.loads((DIRECTORY / "walkthrough.ipynb").read_text())
    assert all(
        c.get("execution_count") is None and not c.get("outputs")
        for c in notebook["cells"]
        if c["cell_type"] == "code"
    )
    namespace, code_cells = execute_code_cells(DIRECTORY / "walkthrough.ipynb")
    assert code_cells == 4
    actual = namespace["workflow"]
    assert isinstance(actual, Workflow) and actual.best_feasible is not None
    assert actual.best_feasible.cost == 3
    assert not actual.optimizer.converged
