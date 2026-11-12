"""Accept source Ising objective, real native optimization and finite plots."""

import json
import math
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.applications.physical_systems.ising_model.comparisons.circuit import (
    cost as exported_cost,
)
from examples.synthesis.applications.physical_systems.ising_model.overview.comparisons.circuit import (
    reference_circuit,
    source_energy,
)
from examples.synthesis.applications.physical_systems.ising_model.overview.main import (
    Workflow,
    run_workflow,
)
from examples.synthesis.applications.physical_systems.ising_model.overview.sampling import (
    sample,
)
from examples.synthesis.applications.physical_systems.ising_model.overview.workflow import (
    energy,
    lower_tail_cost,
    source_program,
)
from tests.integration.synthesis_notebook_support import execute_code_cells

pytestmark = pytest.mark.integration
DIRECTORY = (
    Path(__file__).resolve().parents[2]
    / "examples/synthesis/applications/physical_systems/ising_model/overview"
)


@pytest.mark.parametrize("word", range(64))
def test_source_periodic_energy_and_exported_sign(word: int) -> None:
    bits = tuple((word >> i) & 1 for i in range(6))
    assert energy(word) == source_energy(bits) == -exported_cost(bits)


def test_tail_fractional_boundary_and_actual_energy_order() -> None:
    rows = ((0, 0.2), (1, 0.6), (63, 0.2))
    # E(0)=-180,E(1)=-100,E(63)=60. Only0.5 of state1enters the0.7tail.
    assert energy(1) == -100 and energy(63) == 60
    assert lower_tail_cost(rows) == pytest.approx((0.2 * -180 + 0.5 * -100) / 0.7)
    assert lower_tail_cost(tuple(reversed(rows))) == lower_tail_cost(rows)
    assert lower_tail_cost(rows, 1) == pytest.approx(
        sum(energy(w) * p for w, p in rows)
    )
    assert lower_tail_cost(((0, 1.0),)) == -180


@pytest.mark.parametrize("quantile", [0.0, -1.0, 1.1, math.nan, math.inf, True])
def test_bad_quantile_refuses(quantile: float) -> None:
    with pytest.raises(ValueError, match="quantile"):
        lower_tail_cost(((0, 1.0),), quantile)


@pytest.mark.parametrize(
    "rows",
    [
        (),
        ((0, 0.5),),
        ((0, math.nan),),
        ((0, -0.1), (1, 1.1)),
        ((0, 0.5), (0, 0.5)),
        ((64, 1.0),),
        ((True, 1.0),),
    ],
)
def test_bad_tail_distributions_refuse(rows: tuple[tuple[int, float], ...]) -> None:
    with pytest.raises(ValueError, match="normalized"):
        lower_tail_cost(rows)


@pytest.mark.parametrize("word", [-1, 64, True])
def test_bad_configuration_refuses(word: int) -> None:
    with pytest.raises(ValueError, match="six-bit"):
        energy(word)


@pytest.mark.parametrize(
    "parameters",
    [
        (0.1,) * 9,
        (0.1,) * 11,
        (math.nan,) + (0.1,) * 9,
        (math.inf,) + (0.1,) * 9,
        (True,) + (0.1,) * 9,
    ],
)
def test_invalid_layer_parameters_refuse(parameters: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="ten finite"):
        source_program(parameters)


@pytest.mark.parametrize(
    "parameters",
    [
        (0.011, 0.017, 0.023, 0.029, 0.031, 0.13, 0.19, 0.23, 0.29, 0.31),
        (-0.013, 0.021, -0.034, 0.015, -0.026, 0.16, -0.19, 0.32, -0.25, 0.34),
        (0.0,) * 10,
    ],
)
def test_fullstate_absolute_phase_bit_order_and_clean_work(
    parameters: tuple[float, ...],
) -> None:
    program = source_program(parameters)
    actual = fq.run(program).statevector().flatten()
    expected = fq.run(reference_circuit(parameters)).statevector().flatten()
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=0)
    assert program.report["workspace_qubits"] == 0
    assert actual.numel() == 64 and tuple(program.registers) == ("v",)
    assert tuple(program.registers["v"]) == tuple(range(6))


@pytest.fixture(scope="module")
def workflow() -> Workflow:
    return run_workflow()


def test_actual100evaluations_best_reexecution_and_ground_probability(
    workflow: Workflow,
) -> None:
    result = workflow.optimization
    assert len(result.trace) == 100
    assert result.initial_cost == result.trace[0]
    assert result.best_cost == min(result.trace) < result.initial_cost
    assert result.best_cost == pytest.approx(-180, abs=1e-10)
    assert "budget exhausted" in result.termination
    values = tuple(
        (key[0], p)
        for key, p in workflow.program.probabilities(
            fq.run(workflow.program), registers=("v",), threshold=0
        ).items()
    )
    assert dict(values) == pytest.approx(dict(result.rows), abs=1e-14)
    assert lower_tail_cost(values) == pytest.approx(result.best_cost, abs=1e-12)
    ground_probability = sum(p for w, p in values if energy(w) == -180)
    assert (
        0.7 < ground_probability < 0.9
    )  # Reaching CVaR optimum does not give a pure ground state.
    actual = fq.run(workflow.program).statevector().flatten()
    expected = fq.run(reference_circuit(result.parameters)).statevector().flatten()
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=0)


def test_actual_finite_samples_spin_decode_and_uniform_execution(
    workflow: Workflow,
) -> None:
    for rows in (workflow.optimized_samples, workflow.uniform_samples):
        assert len(rows) == 64 and sum(row.count for row in rows) == 2048
        assert sum(row.probability for row in rows) == 1
        for row in rows:
            assert row.probability == row.count / 2048
            assert row.spins == tuple(2 * ((row.word >> i) & 1) - 1 for i in range(6))
            assert row.energy == source_energy(
                tuple((row.word >> i) & 1 for i in range(6))
            )
    observed = [row for row in workflow.optimized_samples if row.count]
    assert min(observed, key=lambda row: row.energy).spins == (-1,) * 6
    uniform = tuple(
        (key[0], p)
        for key, p in workflow.uniform_program.probabilities(
            fq.run(workflow.uniform_program), registers=("v",), threshold=0
        ).items()
    )
    assert len(uniform) == 64 and all(
        p == pytest.approx(1 / 64, abs=1e-14) for _, p in uniform
    )
    assert sample(uniform, 456) == workflow.uniform_samples
    assert sample(workflow.optimization.rows, 123) == workflow.optimized_samples
    assert (
        sum(row.probability for row in workflow.optimized_samples if row.energy == -180)
        > 0.65
    )


@pytest.mark.parametrize(
    "kind", ["missing", "duplicate", "negative", "nan", "normalization", "seed"]
)
def test_invalid_finite_sampler_refuses(kind: str) -> None:
    rows = tuple((word, 1 / 64) for word in range(64))
    seed = 123
    if kind == "missing":
        rows = rows[:-1]
    elif kind == "duplicate":
        rows = ((1, 1 / 64),) + rows[1:]
    elif kind == "negative":
        rows = ((0, -1 / 64),) + rows[1:]
    elif kind == "nan":
        rows = ((0, math.nan),) + rows[1:]
    elif kind == "normalization":
        rows = tuple((word, 1 / 128) for word in range(64))
    else:
        seed = -1
    with pytest.raises(ValueError):
        sample(rows, seed)


def test_actual_notebook_quantum_entry_samples_plots_and_provenance() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle

    nb = json.loads((DIRECTORY / "walkthrough.ipynb").read_text())
    for cell in nb["cells"]:
        assert "from classiq" not in "".join(cell["source"])
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and not cell["outputs"]
    namespace, count = execute_code_cells(DIRECTORY / "walkthrough.ipynb")
    assert count == 4
    workflow = namespace["workflow"]
    assert isinstance(workflow, Workflow)
    first, second = namespace["convergence"], namespace["histogram"]
    assert isinstance(first, Figure) and isinstance(second, Figure)
    line = first.axes[0].lines[0]
    torch.testing.assert_close(
        torch.as_tensor(line.get_ydata()),
        torch.tensor(workflow.optimization.trace, dtype=torch.float64),
        atol=0,
        rtol=0,
    )
    assert first.axes[0].get_xlabel() == "Objective evaluations"
    assert second.axes[0].get_xlabel() == "Source Ising energy"
    assert len(second.axes[0].containers) == 2
    lower = min(row.energy for row in workflow.uniform_samples)
    upper = max(row.energy for row in workflow.uniform_samples)
    bins = torch.tensor(
        [lower + i * (upper - lower) / 30 for i in range(31)], dtype=torch.float64
    )
    for bars, rows in zip(
        second.axes[0].containers,
        (workflow.optimized_samples, workflow.uniform_samples),
        strict=True,
    ):
        assert len(bars) == 30
        heights = []
        for patch in bars:
            assert isinstance(patch, Rectangle)
            heights.append(patch.get_height())
        expected, _ = torch.histogram(
            torch.tensor([row.energy for row in rows], dtype=torch.float64),
            bins=bins,
            weight=torch.tensor([row.probability for row in rows], dtype=torch.float64),
        )
        torch.testing.assert_close(
            torch.tensor(heights, dtype=torch.float64), expected, atol=1e-14, rtol=0
        )
        assert sum(heights) == pytest.approx(1, abs=1e-14)
    plt.close(first)
    plt.close(second)
    proof = json.loads((DIRECTORY / "provenance.json").read_text())
    assert [row["index"] for row in proof["reviewed_code_cells"]] == [
        2,
        4,
        6,
        8,
        10,
        12,
        14,
        16,
        18,
        20,
        22,
    ]


@pytest.mark.parametrize("budget", ["width", "operations", "depth", "memory"])
def test_resource_budgets_refuse(budget: str) -> None:
    from examples.synthesis.applications.physical_systems.ising_model.main import main
    from flagquantum import synth
    from flagquantum.errors import PlanningError

    parameters: dict[str, float | list[float]] = {
        "params": [0.01 * (i + 1) for i in range(10)]
    }
    with pytest.raises((synth.SynthesisError, PlanningError)):
        if budget == "width":
            synth.synthesize(main, classical_inputs=parameters, max_qubits=5)
        elif budget == "operations":
            synth.synthesize(main, classical_inputs=parameters, max_operations=1)
        elif budget == "depth":
            synth.synthesize(main, classical_inputs=parameters, max_depth=1)
        else:
            synth.synthesize(main, classical_inputs=parameters, memory_limit_bytes=16)
