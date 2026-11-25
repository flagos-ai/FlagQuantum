"""Accept source-seeded partition workflow against independent costs and Circuit."""

import cmath
import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.applications.optimization.set_partition.overview.program import (
    INITIAL_PARAMETERS,
    build_program,
    cost_layer,
    main,
    source_weights,
)
from examples.synthesis.applications.optimization.set_partition.overview.workflow import (
    Workflow,
    costs,
    distribution,
    partitions,
    run_workflow,
    sample_distribution,
)
from examples.synthesis.cost_reference import cost_phase_circuit
from examples.synthesis.graph_optimization import lower_tail_cost
from flagquantum import synth
from flagquantum.compiler.resource_estimation import ResourceEstimate
from flagquantum.errors import PlanningError
from flagquantum.synth import Output, QArray, QBit, X, allocate, qfunc
from tests.integration.synthesis_notebook_support import execute_code_cells

pytestmark = pytest.mark.integration
DIRECTORY = (
    Path(__file__).resolve().parents[2]
    / "examples/synthesis/applications/optimization/set_partition/overview"
)
SOURCE = (6, 1, 4, 4, 8, 10, 4, 6, 3, 5)
SECOND = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29)
BINDINGS = (INITIAL_PARAMETERS, (-0.014, 0.021, -0.029, 0.036, -0.043, 0.051))


@qfunc
def phase_probe(gamma: float, weights: list[int], v: Output[QArray[QBit]]) -> None:
    allocate(10, v)
    cost_layer(gamma, weights, v)


@qfunc
def item_probe(index: int, v: Output[QArray[QBit]]) -> None:
    allocate(10, v)
    X(v[index])


def reference(weights: tuple[int, ...], params: tuple[float, ...]) -> fq.Circuit:
    return cost_phase_circuit(
        lambda bits: -float(
            sum(value * (2 * bit - 1) for value, bit in zip(weights, bits, strict=True))
            ** 2
        ),
        10,
        params,
    )


def test_source_seed_without_global_rng_mutation() -> None:
    np = pytest.importorskip("numpy")
    before = np.random.get_state()
    assert source_weights() == SOURCE and sum(SOURCE) == 51
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    assert np.array_equal(before[1], after[1])


@pytest.mark.parametrize("index", range(10))
def test_named_item_is_lsb_index(index: int) -> None:
    program = synth.synthesize(item_probe, classical_inputs={"index": index})
    observed = distribution(program)
    assert observed[1 << index] == pytest.approx(1.0)
    selected, other = partitions(1 << index, SOURCE)
    assert selected == (SOURCE[index],)
    assert other == SOURCE[:index] + SOURCE[index + 1 :]
    assert sample_distribution(SOURCE, observed, shots=10, seed=1)[0].word == 1 << index


@pytest.mark.parametrize("weights", (SOURCE, SECOND))
def test_all1024_classical_groupings(weights: tuple[int, ...]) -> None:
    table = costs(weights)
    for word in range(1024):
        selected, other = partitions(word, weights)
        assert len(selected) + len(other) == 10
        assert sum(selected) + sum(other) == sum(weights)
        assert table[word] == (sum(selected) - sum(other)) ** 2
        assert table[word] == table[1023 ^ word]


@pytest.mark.parametrize("weights", (SOURCE, SECOND))
@pytest.mark.parametrize("gamma", (0.019, -0.023))
def test_all1024_absolute_basis_phases(weights: tuple[int, ...], gamma: float) -> None:
    program = synth.synthesize(
        phase_probe,
        classical_inputs={"gamma": gamma, "weights": weights},
        max_qubits=10,
        max_workspace=0,
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
        selected = sum(
            value for index, value in enumerate(weights) if word & (1 << index)
        )
        difference = 2 * selected - sum(weights)
        expected = torch.zeros(1024, dtype=torch.complex128)
        expected[physical_word] = torch.tensor(
            cmath.exp(-1j * gamma * difference**2), dtype=torch.complex128
        )
        torch.testing.assert_close(actual, expected, atol=1e-11, rtol=0)


@pytest.mark.parametrize("weights", (SOURCE, SECOND))
@pytest.mark.parametrize("params", BINDINGS)
def test_complete_coherent_state_and_resources(
    weights: tuple[int, ...], params: tuple[float, ...]
) -> None:
    program = build_program(weights, params)
    actual = fq.run(program).statevector().reshape((2,) * 10)
    actual = actual.permute(program.registers["v"]).reshape(1024)
    expected = fq.run(reference(weights, params)).statevector().flatten()
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=0)
    torch.testing.assert_close(
        torch.tensor(distribution(program), dtype=torch.float64),
        expected.abs().square(),
        atol=1e-11,
        rtol=0,
    )
    assert program.circuit.n_wires == 10 and program.report["workspace_qubits"] == 0
    resources = program.report["resources"]
    assert isinstance(resources, ResourceEstimate)
    assert resources.n_operations == len(program.circuit.instructions) == 196


@pytest.fixture(scope="module")
def workflow() -> Workflow:
    pytest.importorskip("numpy")
    return run_workflow()


def test_actual_optimizer_samples_and_independent_cost(workflow: Workflow) -> None:
    assert workflow.weights == SOURCE
    assert workflow.optimizer.iterations == 80 and not workflow.optimizer.converged
    assert len(workflow.trace) == workflow.optimizer.evaluations == 166
    assert workflow.trace[0].parameters == INITIAL_PARAMETERS
    assert workflow.trace[0].cost == pytest.approx(103.45407875201481, abs=1e-10)
    assert workflow.optimizer.value == pytest.approx(61.11512343519662, abs=1e-10)
    assert sum(row.count for row in workflow.samples) == 1000
    assert sum(row.count for row in workflow.uniform) == 1000
    assert (
        workflow.best_observed.cost == costs(SOURCE)[workflow.exact_reference_word] == 1
    )
    assert sum(row.count for row in workflow.samples if row.cost == 1) == 109
    assert sum(row.count for row in workflow.uniform if row.cost == 1) == 85
    selected, other = partitions(workflow.best_observed.word, SOURCE)
    assert (sum(selected), sum(other)) == (25, 26)
    probabilities = distribution(workflow.program)
    assert (
        sample_distribution(SOURCE, probabilities, shots=1000, seed=2718)
        == workflow.samples
    )
    assert (
        sample_distribution(SOURCE, probabilities, shots=1000, seed=2719)
        != workflow.samples
    )
    for entry in (workflow.trace[0], workflow.trace[-1]):
        probabilities_reference = tuple(
            float(value)
            for value in fq.run(reference(SOURCE, entry.parameters))
            .statevector()
            .flatten()
            .abs()
            .square()
        )
        assert entry.cost == pytest.approx(
            lower_tail_cost(probabilities_reference, costs(SOURCE), 0.7), abs=1e-10
        )


def test_reference_never_substitutes_for_observations(workflow: Workflow) -> None:
    observed = sample_distribution(SOURCE, (1.0,) + (0.0,) * 1023, shots=10, seed=1)
    altered = replace(workflow, samples=observed)
    assert altered.best_observed.word == 0 and altered.best_observed.cost == 2601
    assert costs(SOURCE)[altered.exact_reference_word] == 1


def test_actual_plot_roles(workflow: Workflow) -> None:
    pytest.importorskip("matplotlib")
    np = pytest.importorskip("numpy")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    from examples.synthesis.applications.optimization.set_partition.overview.plots import (
        partition_figure,
    )

    figure = partition_figure(workflow)
    try:
        figure.canvas.draw()
        assert len(figure.axes) == 4
        assert np.array_equal(
            np.asarray(figure.axes[0].lines[0].get_ydata(), dtype=np.float64),
            np.asarray([entry.cost for entry in workflow.trace]),
        )
        histogram = figure.axes[1]
        assert len(histogram.containers) == 2
        for container in histogram.containers:
            assert sum(patch.get_height() for patch in container) == pytest.approx(1.0)
        assert [
            bar.get_height()
            for bar in figure.axes[2].patches
            if isinstance(bar, Rectangle)
        ] == [25, 26]
        assert [
            bar.get_height()
            for bar in figure.axes[3].patches
            if isinstance(bar, Rectangle)
        ] == [25, 26]
    finally:
        plt.close(figure)


def test_clean_notebook_executes_complete_workflow() -> None:
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
    assert count == 5 and namespace["weights"] == SOURCE
    result = namespace["workflow"]
    assert isinstance(result, Workflow) and result.best_observed.cost == 1
    initial = namespace["initial_program"]
    assert isinstance(initial, synth.Program)
    assert distribution(initial) == pytest.approx(
        distribution(build_program(SOURCE)), abs=1e-11
    )
    assert namespace["exact_cost"] == 1


@pytest.mark.parametrize("weights", ((), (1,) * 9, (0,) * 10, (33,) * 10, (True,) * 10))
def test_invalid_weights_refuse(weights: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="ten integers"):
        build_program(weights)


@pytest.mark.parametrize("params", ((0.1,), (math.nan,) * 6, (math.inf,) * 6))
def test_invalid_parameters_refuse(params: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="six finite"):
        build_program(SOURCE, params)


@pytest.mark.parametrize("value", (0, -1, 129, True))
def test_invalid_iterations_refuse(value: int) -> None:
    with pytest.raises(ValueError, match="maxiter"):
        run_workflow(SOURCE, maxiter=value)


@pytest.mark.parametrize("word", (-1, 1024, True))
def test_invalid_assignments_refuse(word: int) -> None:
    with pytest.raises(ValueError, match="assignment"):
        partitions(word, SOURCE)


@pytest.mark.parametrize(
    "probabilities", ((), (-1.0,) * 1024, (math.nan,) * 1024, (0.5,) * 1024)
)
def test_invalid_probabilities_refuse(probabilities: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        sample_distribution(SOURCE, probabilities, shots=10, seed=1)


@pytest.mark.parametrize(
    "shots,seed", ((0, 1), (100001, 1), (True, 1), (10, -1), (10, True), (10, 2**63))
)
def test_invalid_sampling_refuses(shots: int, seed: int) -> None:
    with pytest.raises(ValueError, match="shots|seed"):
        run_workflow(SOURCE, shots=shots, seed=seed)


@pytest.mark.parametrize("budget", ("width", "depth", "operations", "memory"))
def test_budget_refusal(budget: str) -> None:
    bindings = {"params": INITIAL_PARAMETERS, "weights": SOURCE}
    with pytest.raises((synth.SynthesisError, PlanningError)):
        if budget == "width":
            synth.synthesize(main, classical_inputs=bindings, max_qubits=9)
        elif budget == "depth":
            synth.synthesize(main, classical_inputs=bindings, max_depth=1)
        elif budget == "operations":
            synth.synthesize(main, classical_inputs=bindings, max_operations=1)
        else:
            synth.synthesize(main, classical_inputs=bindings, memory_limit_bytes=16)


def test_wrong_register_refuses() -> None:
    from examples.synthesis.algorithms.search_and_optimization.qaoa.overview.workflow import (
        build_program as other_program,
    )

    with pytest.raises(ValueError, match="ten-bit"):
        distribution(other_program("maxcut", (0.1,) * 8))
