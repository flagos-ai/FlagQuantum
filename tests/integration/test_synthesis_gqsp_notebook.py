"""Accept actual GQSP phase finding, native execution and finite local sampling."""

import json
from fractions import Fraction
from pathlib import Path
from typing import cast

import pytest
import torch

import flagquantum as fq
from examples.synthesis.algorithms.quantum_primitives.gqsp.comparisons.circuit import (
    reference_circuit,
)
from examples.synthesis.algorithms.quantum_primitives.gqsp.main import (
    build_program,
    main,
)
from examples.synthesis.algorithms.quantum_primitives.gqsp.physics import PHASES
from examples.synthesis.compare_states import compare_states
from flagquantum import synth
from flagquantum.errors import PlanningError
from tests.integration.synthesis_notebook_support import execute_code_cells

try:
    import numpy as np
    import numpy.typing as npt

    from examples.synthesis.algorithms.quantum_primitives.gqsp.overview import (
        phases as phase_solver,
    )
    from examples.synthesis.algorithms.quantum_primitives.gqsp.overview.main import (
        Workflow,
        plot_results,
        run_workflow,
    )
    from examples.synthesis.algorithms.quantum_primitives.gqsp.overview.phases import (
        factor_phases,
        rotation,
    )
except ModuleNotFoundError as error:
    if error.name != "numpy":
        raise
    pytest.skip(
        "This complete Notebook workflow requires NumPy.", allow_module_level=True
    )

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "examples/synthesis/algorithms/quantum_primitives/gqsp/overview"


@pytest.mark.parametrize("variant", range(5))
def test_complex_polynomial_response(variant: int) -> None:
    if variant < 3:
        coefficients = np.array([1.0, 0, 3, 0, 3, 0, 1]) / 8 * (0.99, 0.8, 0.6)[variant]
    elif variant == 3:
        coefficients = np.array([0.1, 0.2, -0.1, 0.15])
    else:
        coefficients = np.random.RandomState(7).uniform(-1, 1, 7)
        coefficients *= 0.8 / np.abs(coefficients).sum()
    result = factor_phases(coefficients)
    points = np.exp(1j * np.linspace(-np.pi, np.pi, 4097))
    state = np.broadcast_to(np.array([1, 0], complex), (len(points), 2)).copy()
    for index, row in enumerate(result.phases):
        if index:
            state[:, 0] *= points
        state = state @ rotation(row).T
    np.testing.assert_allclose(
        state[:, 0],
        np.polynomial.polynomial.polyval(points, coefficients),
        atol=1e-10,
        rtol=0,
    )
    np.testing.assert_allclose((np.abs(state) ** 2).sum(axis=1), 1, atol=1e-12, rtol=0)
    assert result.completion_coefficient_error < 1e-10


@pytest.mark.parametrize(
    "kind",
    ["constant", "oversized", "nan", "infinite", "l1", "endpoint", "dtype", "matrix"],
)
def test_invalid_profile_refuses(kind: str) -> None:
    values = np.array([0.1, 0.2, 0.1])
    if kind == "constant":
        values = values[:1]
    elif kind == "oversized":
        values = np.ones(8) * 0.01
    elif kind == "nan":
        values[1] = np.nan
    elif kind == "infinite":
        values[1] = np.inf
    elif kind == "l1":
        values *= 10
    elif kind == "endpoint":
        values[0] = 0
    elif kind == "dtype":
        with pytest.raises(ValueError):
            factor_phases(np.array(values, dtype=np.float32))
        return
    else:
        values = values.reshape(1, 3)
    with pytest.raises(ValueError):
        factor_phases(values)


@pytest.fixture(scope="module")
def workflow() -> Workflow:
    return run_workflow()


def test_actual_state_absolute_phase_order_and_clean_work(workflow: Workflow) -> None:
    compare_states(
        workflow.program, reference_circuit(phases=workflow.fit.phases), ("x", "ind")
    )
    assert tuple(len(w) for w in workflow.program.registers.values()) == (7, 1)
    assert workflow.program.report["workspace_qubits"] == 0
    torch.testing.assert_close(workflow.exact, workflow.target, atol=1e-12, rtol=0)


def test_fractional_readout_actual_counts_and_joint_postselection(
    workflow: Workflow,
) -> None:
    assert workflow.shots == 100000
    assert sum(count for _, count in workflow.counts) == 100000
    assert all(
        isinstance(x, Fraction) and isinstance(ind, Fraction)
        for (x, ind), _ in workflow.counts
    )
    assert workflow.points.numel() == 128
    torch.testing.assert_close(
        workflow.points, torch.arange(128, dtype=torch.float64) / 128, atol=0, rtol=0
    )
    good_count = sum(
        count for (_, indicator), count in workflow.counts if indicator == 0
    )
    assert 0.29 < good_count / 100000 < 0.32
    assert float(workflow.measured.sum()) == pytest.approx(
        good_count / 100000 / 0.99**2, abs=1e-14
    )
    assert float(workflow.measured.sum()) < 0.4
    assert float((workflow.measured - workflow.target).abs().max()) < 0.001


def test_actual_plot_has_unobstructed_data_and_correct_target_label(
    workflow: Workflow,
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figure = plot_results(workflow)
    axes = figure.axes[0]
    np.testing.assert_allclose(
        np.asarray(axes.lines[0].get_xdata()), workflow.points.numpy(), atol=0, rtol=0
    )
    np.testing.assert_allclose(
        np.asarray(axes.lines[0].get_ydata()), workflow.measured.numpy(), atol=0, rtol=0
    )
    np.testing.assert_allclose(
        np.asarray(axes.lines[1].get_ydata()), workflow.target.numpy(), atol=0, rtol=0
    )
    assert axes.lines[0].get_label() == "GQSP with 100000 local samples"
    assert axes.lines[1].get_label() == r"$\frac{1}{2^N}\cos^6(2\pi x)$"
    assert axes.get_xlabel() == "x" and axes.get_ylabel() == "f(x)"
    plt.close(figure)


def test_ill_conditioned_roots_refuse(monkeypatch: pytest.MonkeyPatch) -> None:
    def unit_roots(_: npt.NDArray[np.float64]) -> npt.NDArray[np.complex128]:
        return np.ones(12, dtype=np.complex128)

    monkeypatch.setattr(np, "roots", unit_roots)
    with pytest.raises(ValueError, match="separated"):
        factor_phases(np.array([1.0, 0, 3, 0, 3, 0, 1]) * 0.99 / 8)


def test_failed_spectral_completion_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    def incorrect_complement(_: npt.NDArray[np.complex128]) -> npt.NDArray[np.float64]:
        return np.ones(7, dtype=np.float64)

    monkeypatch.setattr(np, "poly", incorrect_complement)
    with pytest.raises(ValueError, match="completion failed"):
        factor_phases(np.array([1.0, 0, 3, 0, 3, 0, 1]) * 0.99 / 8)


def test_failed_degree_stripping_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    def incorrect_angles(_: npt.NDArray[np.complex128]) -> tuple[float, float, float]:
        return (0.0, 0.0, 0.0)

    monkeypatch.setattr(phase_solver, "_angles", incorrect_angles)
    with pytest.raises(ValueError, match="stripping failed"):
        factor_phases(np.array([1.0, 0, 3, 0, 3, 0, 1]) * 0.99 / 8)


@pytest.mark.parametrize("variant", [0, 1])
def test_newly_factored_other_polynomials_quantum_state_and_probabilities(
    variant: int,
) -> None:
    coefficients = np.random.RandomState(variant + 7).uniform(-1, 1, 7)
    coefficients *= 0.8 / np.abs(coefficients).sum()
    fit = factor_phases(coefficients)
    program = build_program(phases=fit.phases)
    compare_states(program, reference_circuit(phases=fit.phases), ("x", "ind"))
    values = program.numeric_probabilities(
        fq.run(program), registers=("x", "ind"), threshold=0
    )
    for (x, indicator), probability in values.items():
        if indicator == 0:
            z = np.exp(2j * np.pi * float(x))
            expected = abs(np.polynomial.polynomial.polyval(z, coefficients)) ** 2 / 128
            assert probability == pytest.approx(expected, abs=1e-12, rel=0)


def test_phase_branch_does_not_claim_frozen_full_state_equivalence(
    workflow: Workflow,
) -> None:
    assert workflow.fit.phases != PHASES
    frozen = fq.run(build_program()).statevector().flatten()
    actual = fq.run(workflow.program).statevector().flatten()
    assert float((frozen - actual).abs().max()) > 1e-3
    torch.testing.assert_close(
        frozen.reshape(128, 2)[:, 0].abs().square(),
        actual.reshape(128, 2)[:, 0].abs().square(),
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize("budget", ["width", "operations", "depth", "memory"])
def test_new_phase_program_budgets_refuse(budget: str, workflow: Workflow) -> None:
    rows = workflow.fit.phases
    parameters: dict[str, float | list[float]] = {
        "negative_power": 3,
        "initial_indicator": 0,
        "theta": [r[0] for r in rows],
        "phi": [r[1] for r in rows],
        "lam": [r[2] for r in rows],
    }
    with pytest.raises((synth.SynthesisError, PlanningError)):
        if budget == "width":
            synth.synthesize(main, classical_inputs=parameters, max_qubits=7)
        elif budget == "operations":
            synth.synthesize(main, classical_inputs=parameters, max_operations=1)
        elif budget == "depth":
            synth.synthesize(main, classical_inputs=parameters, max_depth=1)
        else:
            synth.synthesize(main, classical_inputs=parameters, memory_limit_bytes=16)


def test_actual_notebook_phase_finding_execution_sampling_plot_and_provenance() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    path = DIRECTORY / "walkthrough.ipynb"
    notebook = json.loads(path.read_text())
    assert len({cell["id"] for cell in notebook["cells"]}) == len(notebook["cells"])
    for cell in notebook["cells"]:
        text = "".join(cell["source"])
        assert "from classiq" not in text and "source_format" not in text
        assert (
            "from workflow import" not in text
            and "from phase_factorization import" not in text
        )
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and not cell["outputs"]
    namespace, cells = execute_code_cells(path)
    assert cells == 4
    result = cast(Workflow, namespace["workflow"])
    compare_states(
        result.program, reference_circuit(phases=result.fit.phases), ("x", "ind")
    )
    assert result.shots == 100000 and sum(count for _, count in result.counts) == 100000
    torch.testing.assert_close(result.exact, result.target, atol=1e-12, rtol=0)
    figure = namespace["figure"]
    assert isinstance(figure, Figure)
    np.testing.assert_allclose(
        np.asarray(figure.axes[0].lines[0].get_ydata()),
        result.measured.numpy(),
        atol=0,
        rtol=0,
    )
    np.testing.assert_allclose(
        np.asarray(figure.axes[0].lines[1].get_ydata()),
        result.target.numpy(),
        atol=0,
        rtol=0,
    )
    plt.close(figure)
    proof = json.loads((DIRECTORY / "provenance.json").read_text())
    assert [c["index"] for c in proof["reviewed_code_cells"]] == [
        3,
        5,
        7,
        9,
        11,
        13,
        15,
        17,
    ]
