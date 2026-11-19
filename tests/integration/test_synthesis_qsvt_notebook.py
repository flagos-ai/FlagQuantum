"""Accept real phase fitting, newly bound QSVT execution and the source plot."""

import json
from math import cos
from pathlib import Path
from typing import cast

import pytest
import torch

from examples.synthesis.compare_states import compare_states
from examples.synthesis.functions.qsvt.comparisons.circuit import reference_circuit
from examples.synthesis.functions.qsvt.main import PHASES, main
from flagquantum import synth
from flagquantum.errors import PlanningError
from tests.integration.synthesis_notebook_support import execute_code_cells

try:
    import numpy as np

    from examples.synthesis.functions.qsvt.overview import main as qsvt
except ModuleNotFoundError as error:
    if error.name != "numpy":
        raise
    pytest.skip(
        "This complete Notebook workflow requires NumPy.", allow_module_level=True
    )

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "examples/synthesis/functions/qsvt/overview"


@pytest.fixture(scope="module")
def workflow() -> qsvt.Workflow:
    return qsvt.run_workflow()


def test_new_phase_fit_full_state_absolute_phase_clean_work_and_projection(
    workflow: qsvt.Workflow,
) -> None:
    assert workflow.fit.phases != PHASES
    assert workflow.fit.seed in range(4) and workflow.fit.evaluations > 6
    assert workflow.fit.max_error < 1e-7
    compare_states(
        workflow.program,
        reference_circuit(workflow.fit.phases),
        ("state", "ref", "ind", "qsvt_aux"),
    )
    assert tuple(len(wires) for wires in workflow.program.registers.values()) == (
        4,
        4,
        1,
        1,
    )
    workspace = workflow.program.report["workspace_qubits"]
    assert isinstance(workspace, int) and workspace <= 6
    expected = np.polynomial.Chebyshev(qsvt.source_coefficients().numpy())(
        workflow.points.numpy()
    )
    torch.testing.assert_close(
        workflow.target,
        torch.tensor(abs(expected), dtype=torch.float64),
        atol=1e-14,
        rtol=0,
    )
    torch.testing.assert_close(workflow.measured, workflow.target, atol=1e-7, rtol=0)


def test_source_coefficients_parity_rng_and_independent_polynomial_recurrence() -> None:
    coefficients = qsvt.source_coefficients()
    expected = torch.tensor(
        [0, 0.33582085210229873, 0, -0.3012867196677423, 0, -0.6213616939956953],
        dtype=torch.float64,
    )
    torch.testing.assert_close(coefficients, expected, atol=1e-15, rtol=0)
    xs = torch.linspace(-1, 1, 4097, dtype=torch.float64)
    actual = qsvt.polynomial(coefficients, xs)
    reference = np.polynomial.Chebyshev(coefficients.numpy())(xs.numpy())
    torch.testing.assert_close(
        actual, torch.tensor(reference, dtype=torch.float64), atol=2e-15, rtol=0
    )
    assert float(actual.abs().max()) < 1


def test_scalar_response_frozen_convention_and_zero_angle_identity() -> None:
    xs = torch.linspace(-1, 1, 257, dtype=torch.float64)
    torch.testing.assert_close(
        qsvt.response(torch.zeros(6, dtype=torch.float64), xs), xs, atol=1e-14, rtol=0
    )
    frozen = qsvt.response(torch.tensor(PHASES, dtype=torch.float64), xs)
    torch.testing.assert_close(
        frozen, qsvt.polynomial(qsvt.source_coefficients(), xs), atol=3e-15, rtol=0
    )
    phases = torch.tensor([0.2, -0.7, 1.1, 0.4, -0.3, 0.9], dtype=torch.float64)
    values = qsvt.response(phases, xs)
    torch.testing.assert_close(values, -values.flip(0), atol=1e-14, rtol=0)
    assert float(values[-1]) == pytest.approx(cos(float(phases.sum()) / 2), abs=1e-14)


@pytest.mark.parametrize(
    "kind",
    ["phase_count", "nan_phase", "outside_domain", "nan_value", "dtype", "budget"],
)
def test_invalid_or_unbounded_response_refuses(kind: str) -> None:
    phases = torch.zeros(6, dtype=torch.float64)
    values = torch.linspace(-1, 1, 5, dtype=torch.float64)
    if kind == "phase_count":
        phases = phases[:5]
    elif kind == "nan_phase":
        phases[1] = float("nan")
    elif kind == "outside_domain":
        values[1] = 1.01
    elif kind == "nan_value":
        values[0] = float("nan")
    elif kind == "dtype":
        values = values.float()
    else:
        values = torch.zeros(65537, dtype=torch.float64)
    with pytest.raises(ValueError, match="six finite"):
        qsvt.response(phases, values)


@pytest.mark.parametrize(
    "iterations,starts", [(0, 1), (2501, 1), (True, 1), (10, 0), (10, 5)]
)
def test_fit_budget_refuses(iterations: int, starts: int) -> None:
    with pytest.raises(ValueError, match="iterations"):
        qsvt.fit_phases(
            qsvt.source_coefficients(), max_iterations=iterations, starts=starts
        )


@pytest.mark.parametrize("kind", ["shape", "nan", "parity", "magnitude", "dtype"])
def test_invalid_fit_coefficients_refuse(kind: str) -> None:
    coefficients = qsvt.source_coefficients()
    if kind == "shape":
        coefficients = coefficients[:5]
    elif kind == "nan":
        coefficients[1] = float("nan")
    elif kind == "parity":
        coefficients[0] = 0.1
    elif kind == "magnitude":
        coefficients[1] = 20
    else:
        coefficients = coefficients.float()
    with pytest.raises(ValueError):
        qsvt.fit_phases(coefficients)


def test_insufficient_fit_budget_has_no_fixed_phase_fallback() -> None:
    with pytest.raises(ValueError, match="no fixed-phase fallback"):
        qsvt.fit_phases(qsvt.source_coefficients(), max_iterations=1, starts=1)


@pytest.mark.parametrize(
    "phases", [(1.0,), (0.0,) * 5 + (float("nan"),), (0.0,) * 5 + (float("inf"),)]
)
def test_invalid_native_angles_refuse(phases: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="six finite"):
        qsvt.build_program(phases)


@pytest.mark.parametrize("kind", ["width", "operations", "depth", "memory"])
def test_quantum_budget_refuses(kind: str, workflow: qsvt.Workflow) -> None:
    parameters = {"phases": list(workflow.fit.phases)}
    with pytest.raises((synth.SynthesisError, PlanningError)):
        if kind == "width":
            synth.synthesize(main, classical_inputs=parameters, max_qubits=9)
        elif kind == "operations":
            synth.synthesize(main, classical_inputs=parameters, max_operations=1)
        elif kind == "depth":
            synth.synthesize(main, classical_inputs=parameters, max_depth=1)
        else:
            synth.synthesize(main, classical_inputs=parameters, memory_limit_bytes=16)


def test_actual_notebook_all_stages_plot_points_and_source_provenance() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    path = DIRECTORY / "walkthrough.ipynb"
    notebook = json.loads(path.read_text())
    assert len({cell["id"] for cell in notebook["cells"]}) == len(notebook["cells"])
    for cell in notebook["cells"]:
        text = "".join(cell["source"])
        assert (
            "from classiq" not in text
            and "source_format" not in text
            and "PHASES" not in text
        )
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and not cell["outputs"]
    namespace, cells = execute_code_cells(path)
    assert cells == 4
    result = cast(qsvt.Workflow, namespace["workflow"])
    compare_states(
        result.program,
        reference_circuit(result.fit.phases),
        ("state", "ref", "ind", "qsvt_aux"),
    )
    figure = namespace["figure"]
    assert isinstance(figure, Figure) and len(figure.axes) == 1
    axes = figure.axes[0]
    assert np.asarray(axes.lines[0].get_xdata()).tolist() == pytest.approx(
        result.points.tolist()
    )
    assert np.asarray(axes.lines[0].get_ydata()).tolist() == pytest.approx(
        result.target.tolist()
    )
    assert axes.lines[0].get_label() == "target"
    scatter = axes.collections[0]
    assert scatter.get_label() == "measured"
    torch.testing.assert_close(
        torch.tensor(np.asarray(scatter.get_offsets()), dtype=torch.float64),
        torch.stack((result.points, result.measured), dim=1),
        atol=0,
        rtol=0,
    )
    assert axes.get_xlabel() == r"$\sqrt{x}$" and axes.get_ylabel() == r"$P(\sqrt{x})$"
    plt.close(figure)
    proof = json.loads((DIRECTORY / "provenance.json").read_text())
    assert [cell["index"] for cell in proof["reviewed_code_cells"]] == [
        2,
        4,
        5,
        6,
        7,
        8,
        9,
    ]
