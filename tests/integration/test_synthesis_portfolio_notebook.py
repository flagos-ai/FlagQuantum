"""Original portfolio model, exact states, tail objective and observed solutions."""

import hashlib
import json
from itertools import product
from pathlib import Path

import pytest

from examples.synthesis.applications.finance.portfolio_optimization.main import (
    DEFAULT_PARAMS,
    main,
)
from examples.synthesis.applications.finance.portfolio_optimization.overview.operations import (
    build_program,
    verify,
)
from examples.synthesis.applications.finance.portfolio_optimization.overview.workflow import (
    Workflow,
    classical_optima,
    cvar,
    distribution,
    encoded_cost,
    objective,
    run_workflow,
    sample,
)
from flagquantum import synth

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def workflow() -> Workflow:
    return run_workflow(maxiter=2, shots=97, seed=1914)


@pytest.mark.parametrize(
    "parameters", (DEFAULT_PARAMS, (-0.023, 0.041, -0.017, 0.052, -0.033, 0.061))
)
def test_absolute_state_and_clean_workspace(parameters: tuple[float, ...]) -> None:
    check = verify(parameters)
    assert check.maximum_state_error < 3e-11
    assert check.dirty_workspace_probability < 1e-22
    assert check.resources.n_qubits == 12


def test_entire_original_cost_and_bounded_slack_domain() -> None:
    for a, b, c, word in product(range(8), repeat=4):
        original = (
            0.9 * (a * a + b * b + c * c)
            + a * b
            - 1.4 * a * c
            - 0.4 * b * c
            - 3 * a
            - 4 * b
            + c
        )
        slack = (word & 1) + 2 * ((word >> 1) & 1) + 3 * ((word >> 2) & 1)
        assert objective((a, b, c)) == pytest.approx(original, abs=1e-12)
        assert encoded_cost((a, b, c, word)) == pytest.approx(
            original + 10 * (a + b + c + slack - 6) ** 2, abs=1e-11
        )
    assert {w for w, _ in classical_optima()} == {(1, 2, 1), (2, 1, 1)}
    assert all(cost == pytest.approx(-4.8) for _, cost in classical_optima())


def test_cvar_fractional_boundary_and_zero_mass() -> None:
    probabilities = {(1, 2, 1, 2): 0.2, (0, 0, 0, 7): 0.3, (0, 0, 0, 0): 0.5}
    assert cvar(probabilities, 0.1) == pytest.approx(-4.8)
    assert cvar(probabilities, 0.7) == pytest.approx((-4.8 * 0.2 + 360 * 0.2) / 0.7)
    assert cvar(probabilities, 1) == pytest.approx(-4.8 * 0.2 + 360 * 0.5)
    assert cvar(dict(reversed(tuple(probabilities.items())))) == pytest.approx(
        cvar(probabilities)
    )
    assert cvar({(0, 0, 0, 7): 0.0, (0, 0, 0, 0): 1.0}, 1e-14) == pytest.approx(360)


@pytest.mark.parametrize("quantile", (0.0, -0.1, 1.1, float("nan")))
def test_invalid_cvar_quantiles(quantile: float) -> None:
    with pytest.raises(ValueError, match="quantile"):
        cvar({(0, 0, 0, 0): 1}, quantile)


@pytest.mark.parametrize("mass", (-1.0, 0.7, float("nan")))
def test_invalid_probability_mass(mass: float) -> None:
    with pytest.raises(ValueError, match="probabilities"):
        cvar({(0, 0, 0, 0): mass})


def test_actual_optimizer_trace_and_selected_state(workflow: Workflow) -> None:
    assert workflow.optimizer.iterations == 2
    assert len(workflow.trace) == workflow.optimizer.evaluations
    assert workflow.trace[0].parameters == DEFAULT_PARAMS
    selected = distribution(workflow.program)
    assert cvar(selected) == pytest.approx(workflow.optimizer.value, abs=1e-10)
    assert len(selected) == 4096 and sum(selected.values()) == pytest.approx(1)


def test_actual_seeded_sampling_and_feasibility(workflow: Workflow) -> None:
    for rows in (workflow.optimized, workflow.uniform):
        assert sum(row.count for row in rows) == 97
        assert all(row.feasible == (sum(row.weights) <= 6) for row in rows)
        assert all(row.cost == objective(row.weights) for row in rows)
        assert all(row.count > 0 for row in rows)
    probabilities = distribution(workflow.program)
    assert sample(probabilities, shots=97, seed=1914) == workflow.optimized
    assert workflow.best_observed == min(
        (row for row in workflow.optimized if row.feasible),
        key=lambda row: (row.cost, row.weights),
    )
    assert any(not row.feasible for row in workflow.optimized)


@pytest.mark.parametrize("limit", ("qubits", "operations", "memory"))
def test_resource_refusals(limit: str) -> None:
    with pytest.raises(ValueError, match="budget|limit|memory|candidate"):
        synth.synthesize(
            main,
            classical_inputs={"params": DEFAULT_PARAMS},
            max_qubits=11 if limit == "qubits" else 12,
            max_operations=1 if limit == "operations" else 4096,
            memory_limit_bytes=1 if limit == "memory" else 64 * 1024**2,
            max_workspace=0,
        )


@pytest.mark.parametrize("parameters", ((0.1,), (float("nan"),) * 6))
def test_invalid_angles(parameters: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="six finite"):
        build_program(parameters)


@pytest.mark.parametrize("maxiter", (0, 129))
def test_invalid_iteration_budget(maxiter: int) -> None:
    with pytest.raises(ValueError, match="maxiter"):
        run_workflow(maxiter=maxiter)


def test_source_cell_mapping_and_unchanged_parent() -> None:
    root = Path(__file__).resolve().parents[2]
    directory = (
        root / "examples/synthesis/applications/finance/portfolio_optimization/overview"
    )
    proof = json.loads((directory / "provenance.json").read_text())
    assert [c["index"] for c in proof["reviewed_code_cells"]] == [
        2,
        4,
        6,
        8,
        10,
        12,
        14,
        18,
        20,
        22,
        23,
        25,
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
