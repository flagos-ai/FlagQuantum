"""Accept complete source-precision native QMCI workflow and all three states."""

import json
import math
from pathlib import Path
from typing import Literal

import pytest
import torch

import flagquantum as fq
from examples.synthesis.algorithms.amplitude_amplification_and_estimation.qmc_user_defined.main import (
    PROBABILITIES as FROZEN_PROBABILITIES,
)
from examples.synthesis.algorithms.amplitude_amplification_and_estimation.qmc_user_defined.overview.comparisons.circuit import (
    operators,
    reference_circuit,
)
from examples.synthesis.algorithms.amplitude_amplification_and_estimation.qmc_user_defined.overview.main import (
    PROBABILITIES,
    Workflow,
    build_program,
    run_workflow,
)
from examples.synthesis.algorithms.amplitude_amplification_and_estimation.qmc_user_defined.overview.quantum import (
    estimation_program,
    load,
)
from flagquantum import synth
from flagquantum.errors import PlanningError
from flagquantum.synth import Output, QArray, QBit, X, allocate, qfunc
from tests.integration.synthesis_notebook_support import execute_code_cells

pytestmark = pytest.mark.integration
DIRECTORY = (
    Path(__file__).resolve().parents[2]
    / "examples/synthesis/algorithms/amplitude_amplification_and_estimation/qmc_user_defined/overview"
)


@pytest.mark.parametrize("stage", ["preparation", "grover", "estimation"])
@pytest.mark.parametrize("variant", [0, 1, 2])
def test_complete_states_absolute_phase_order_and_clean_work(
    stage: Literal["preparation", "grover", "estimation"], variant: int
) -> None:
    probabilities, slope, offset = (
        (PROBABILITIES, 0.5, 0.4),
        (tuple((i + 1) / 36 for i in range(8)), 0.3, -0.2),
        (tuple((8 - i) / 36 for i in range(8)), -0.4, 0.7),
    )[variant]
    program = build_program(stage, probabilities, slope, offset)
    actual = fq.run(program).statevector().flatten()
    expected = (
        fq.run(reference_circuit(stage, probabilities, slope, offset))
        .statevector()
        .flatten()
    )
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=0)
    assert program.report["workspace_qubits"] == 0
    assert actual.numel() == (128 if stage == "estimation" else 16)
    if stage == "preparation":
        # Indicator is array element zero: exact joint probabilities are little-endian.
        expected_good = torch.tensor(
            [
                p * math.sin((slope * i + offset) / 2) ** 2
                for i, p in enumerate(probabilities)
            ],
            dtype=torch.float64,
        )
        torch.testing.assert_close(
            actual.reshape(8, 2)[:, 1].abs().square(), expected_good, atol=1e-12, rtol=0
        )


def test_source_precision_is_not_frozen_qmod_rounding() -> None:
    assert tuple(i / 28 for i in range(8)) == PROBABILITIES
    assert PROBABILITIES != FROZEN_PROBABILITIES
    assert sum(PROBABILITIES) == pytest.approx(1, abs=1e-15)


@pytest.fixture(scope="module")
def workflow() -> Workflow:
    return run_workflow()


def test_actual_signed_samples_and_amplitude_comparison(workflow: Workflow) -> None:
    assert len(workflow.counts) == 8 and sum(n for _, n in workflow.counts) == 2048
    assert sorted(float(x) for x, _ in workflow.counts) == [i / 8 for i in range(-4, 4)]
    assert workflow.selected_phase in (3 / 8, -3 / 8)
    assert workflow.measured_amplitude == pytest.approx(
        math.sin(3 * math.pi / 8) ** 2, abs=1e-14
    )
    expected = sum(math.sin(0.25 * i + 0.2) ** 2 * i / 28 for i in range(8))
    assert workflow.exact_amplitude == pytest.approx(expected, abs=1e-15)
    assert abs(workflow.measured_amplitude - expected) < 0.1
    repeated = run_workflow()
    assert repeated.counts == workflow.counts


@pytest.mark.parametrize(
    "probabilities,slope,offset",
    [
        ((0.125,) * 7, 0.5, 0.4),
        ((-0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5), 0.5, 0.4),
        ((0.1,) * 8, 0.5, 0.4),
        ((math.nan,) + (0.125,) * 7, 0.5, 0.4),
        ((0.125,) * 8, math.nan, 0.4),
        ((0.125,) * 8, 0.5, math.inf),
    ],
)
def test_invalid_probabilities_and_angles_refuse(
    probabilities: tuple[float, ...], slope: float, offset: float
) -> None:
    with pytest.raises(ValueError, match="normalized"):
        build_program("preparation", probabilities, slope, offset)


def test_notebook_all_three_stages_signed_histogram_and_provenance() -> None:
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
    result = namespace["workflow"]
    assert isinstance(result, Workflow)
    assert abs(result.measured_amplitude - result.exact_amplitude) < 0.1
    figure = namespace["figure"]
    assert isinstance(figure, Figure)
    heights = []
    for patch in figure.axes[0].patches:
        assert isinstance(patch, Rectangle)
        heights.append(patch.get_height())
    assert heights == [n for _, n in sorted(result.counts)]
    assert figure.axes[0].get_xlabel() == "Signed fractional phase"
    assert figure.axes[0].get_ylabel() == "Count (2048 local samples)"
    plt.close(figure)
    proof = json.loads((DIRECTORY / "provenance.json").read_text())
    assert [row["index"] for row in proof["reviewed_code_cells"]] == [
        3,
        8,
        10,
        12,
        14,
        17,
        19,
        23,
        27,
        28,
        30,
        32,
        33,
    ]


@qfunc
def loader_witness(
    bit0: int,
    bit1: int,
    bit2: int,
    bit3: int,
    probabilities: list[float],
    angles: list[float],
    state: Output[QArray[QBit]],
) -> None:
    allocate(4, state)
    if bit0 == 1:
        X(state[0])
    if bit1 == 1:
        X(state[1])
    if bit2 == 1:
        X(state[2])
    if bit3 == 1:
        X(state[3])
    load(probabilities, angles, state)


@pytest.mark.parametrize("initial", range(16))
def test_every_loader_column(initial: int) -> None:
    parameters: dict[str, float | list[float]] = {
        "bit0": initial & 1,
        "bit1": (initial >> 1) & 1,
        "bit2": (initial >> 2) & 1,
        "bit3": (initial >> 3) & 1,
        "probabilities": list(PROBABILITIES),
        "angles": [0.5 * i + 0.4 for i in range(8)],
    }
    program = synth.synthesize(
        loader_witness, classical_inputs=parameters, max_workspace=0
    )
    _, loader = operators()
    actual = fq.run(program).statevector().flatten()
    torch.testing.assert_close(actual, loader[:, initial], atol=1e-11, rtol=0)
    assert program.report["workspace_qubits"] == 0


@pytest.mark.parametrize("budget", ["width", "operations", "depth", "memory"])
def test_native_resource_budgets_refuse(budget: str) -> None:
    parameters: dict[str, float | list[float]] = {
        "probabilities": list(PROBABILITIES),
        "angles": [0.5 * i + 0.4 for i in range(8)],
    }
    with pytest.raises((synth.SynthesisError, PlanningError)):
        if budget == "width":
            synth.synthesize(
                estimation_program, classical_inputs=parameters, max_qubits=6
            )
        elif budget == "operations":
            synth.synthesize(
                estimation_program, classical_inputs=parameters, max_operations=1
            )
        elif budget == "depth":
            synth.synthesize(
                estimation_program, classical_inputs=parameters, max_depth=1
            )
        else:
            synth.synthesize(
                estimation_program, classical_inputs=parameters, memory_limit_bytes=16
            )
