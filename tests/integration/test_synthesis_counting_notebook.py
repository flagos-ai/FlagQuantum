"""Accept native counting, count-only confidence intervals and source precision."""

import json
import math
from itertools import pairwise
from pathlib import Path
from typing import cast

import pytest
import torch

import flagquantum as fq
from examples.synthesis.algorithms.amplitude_amplification_and_estimation.quantum_counting.comparisons.circuit import (
    reference_circuit,
)
from examples.synthesis.algorithms.amplitude_amplification_and_estimation.quantum_counting.overview.iterative import (
    estimate,
)
from examples.synthesis.algorithms.amplitude_amplification_and_estimation.quantum_counting.overview.main import (
    IndicatorSampler,
    PhaseEstimate,
    run_iterative,
    sample_qpe,
)
from examples.synthesis.algorithms.amplitude_amplification_and_estimation.quantum_counting.quantum_counting_qpe.comparisons.circuit import (
    reference_circuit as qpe_reference,
)
from tests.integration.synthesis_notebook_support import execute_code_cells

pytestmark = pytest.mark.integration
DIRECTORY = (
    Path(__file__).resolve().parents[2]
    / "examples/synthesis/algorithms/amplitude_amplification_and_estimation/quantum_counting/overview"
)


@pytest.fixture(scope="module")
def phase() -> PhaseEstimate:
    return sample_qpe()


@pytest.mark.parametrize("epsilon", [1 / 32, 1 / (16 * 15)])
def test_actual_adaptive_counts_absolute_states_and_confidence(epsilon: float) -> None:
    result, sampler = run_iterative(epsilon)
    assert result.confidence_interval[0] <= 6 / 16 <= result.confidence_interval[1]
    assert result.confidence_interval[1] - result.confidence_interval[0] <= 2 * epsilon
    assert result.estimate == sum(result.confidence_interval) / 2
    assert len(result.iterations) <= 64
    for k, program in sampler.programs.items():
        actual = fq.run(program).statevector().flatten()
        expected = fq.run(reference_circuit(k)).statevector().flatten()
        torch.testing.assert_close(actual, expected, atol=1e-11, rtol=0)
        assert program.report["workspace_qubits"] == 0
        assert actual.numel() == 2**5
        assert sampler.probabilities[k] == pytest.approx(
            math.sin((2 * k + 1) * math.asin(math.sqrt(6 / 16))) ** 2, abs=1e-11
        )
    assert all(
        row.shots == 256 and 0 <= row.successes <= row.shots
        for row in result.iterations
    )
    if epsilon == 1 / 32:
        assert abs(16 * result.estimate - 6) > 1 / 15
    else:
        assert abs(16 * result.estimate - 6) <= 1 / 15
    # Unchanged seed and independent sampler must replay the same counts.
    replay = IndicatorSampler()
    replay.programs.update(sampler.programs)
    replay.probabilities.update(sampler.probabilities)
    assert estimate(replay, epsilon) == result


def test_qpe_fractional_order_absolute_phase_retained_data_and_resolution(
    phase: PhaseEstimate,
) -> None:
    actual = fq.run(phase.program).statevector().flatten()
    expected = fq.run(qpe_reference()).statevector().flatten()
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=0)
    assert phase.program.report["workspace_qubits"] == 0
    assert actual.numel() == 2**9
    assert len(phase.counts) == 32 and sum(n for _, n in phase.counts) == 2048
    assert [float(x) for x, _ in phase.counts] == [i / 32 for i in range(32)]
    assert abs(phase.solutions - 6) < 1.5
    error = max(
        16
        * abs(
            math.sin(math.pi * i / 99) ** 2 - math.sin(math.pi * (i / 99 - 1 / 32)) ** 2
        )
        for i in range(100)
    )
    assert phase.resolution_error == pytest.approx(error, abs=1e-12)


@pytest.mark.parametrize("amplitude", [0.0, 0.1, 0.375, 0.8, 1.0])
def test_count_only_estimator_other_amplitudes_and_endpoints(amplitude: float) -> None:
    generator = torch.Generator().manual_seed(42)
    theta = math.asin(math.sqrt(amplitude))

    def sample(k: int, shots: int) -> int:
        return int(
            torch.binomial(
                torch.tensor(float(shots), dtype=torch.float64),
                torch.tensor(math.sin((2 * k + 1) * theta) ** 2, dtype=torch.float64),
                generator=generator,
            ).item()
        )

    result = estimate(sample, epsilon=0.02)
    assert result.confidence_interval[0] <= amplitude <= result.confidence_interval[1]
    assert abs(result.estimate - amplitude) <= 0.02
    for previous, current in pairwise(result.iterations):
        assert current.theta_interval[0] >= previous.theta_interval[0]
        assert current.theta_interval[1] <= previous.theta_interval[1]


@pytest.mark.parametrize(
    "epsilon,alpha",
    [
        (0.0, 0.01),
        (0.5, 0.01),
        (math.nan, 0.01),
        (math.inf, 0.01),
        (0.1, 0.0),
        (0.1, 1.0),
        (0.1, math.nan),
    ],
)
def test_precision_and_confidence_refuse(epsilon: float, alpha: float) -> None:
    with pytest.raises(ValueError, match="bounded"):
        estimate(lambda k, n: 0, epsilon, alpha)


@pytest.mark.parametrize(
    "max_steps,max_k,batch",
    [
        (0, 16, 256),
        (65, 16, 256),
        (64, -1, 256),
        (64, 17, 256),
        (64, 16, 0),
        (64, 16, 2049),
    ],
)
def test_bounded_budgets_refuse(max_steps: int, max_k: int, batch: int) -> None:
    with pytest.raises(ValueError, match="bounded"):
        estimate(lambda k, n: 0, max_steps=max_steps, max_k=max_k, batch=batch)


@pytest.mark.parametrize("count", [-1, 257, True])
def test_invalid_counts_refuse(count: int) -> None:
    with pytest.raises(ValueError, match="integer success count"):
        estimate(lambda k, n: count)


def test_exhaustion_and_conflicting_counts_refuse() -> None:
    with pytest.raises(ValueError, match="exhausted"):
        estimate(lambda k, n: n // 2, epsilon=1e-9, max_steps=1)
    calls = 0

    def conflicting(k: int, n: int) -> int:
        nonlocal calls
        calls += 1
        return n if calls == 1 else 0

    with pytest.raises(ValueError, match="conflict"):
        estimate(conflicting, epsilon=1e-9, max_k=0, batch=2048)


@pytest.mark.parametrize(
    "k,shots", [(-1, 256), (17, 256), (True, 256), (0, 0), (0, 2049)]
)
def test_sampler_bounds(k: int, shots: int) -> None:
    with pytest.raises(ValueError):
        IndicatorSampler()(k, shots)


def test_actual_notebook_both_precisions_plot_and_provenance() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle

    notebook = json.loads((DIRECTORY / "walkthrough.ipynb").read_text())
    for cell in notebook["cells"]:
        assert "from classiq" not in "".join(cell["source"])
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and not cell["outputs"]
    namespace, cells = execute_code_cells(DIRECTORY / "walkthrough.ipynb")
    assert cells == 4
    result = cast(PhaseEstimate, namespace["phase_result"])
    torch.testing.assert_close(
        fq.run(result.program).statevector(),
        fq.run(qpe_reference()).statevector(),
        atol=1e-11,
        rtol=0,
    )
    figure = namespace["figure"]
    assert isinstance(figure, Figure)
    assert len(figure.axes) == 2
    heights = []
    for patch in figure.axes[0].patches:
        assert isinstance(patch, Rectangle)
        heights.append(patch.get_height())
    assert heights == [n for _, n in result.counts]
    assert figure.axes[0].get_xlabel() == "Fractional phase"
    assert figure.axes[1].get_ylabel() == "Solution count"
    reference_line = figure.axes[1].lines[0].get_ydata()
    assert isinstance(reference_line, list)
    assert reference_line == [6, 6]
    plt.close(figure)
    proof = json.loads((DIRECTORY / "provenance.json").read_text())
    assert [c["index"] for c in proof["reviewed_code_cells"]] == [
        6,
        8,
        11,
        13,
        15,
        17,
        19,
        21,
        25,
        28,
        30,
        32,
        34,
        35,
        37,
    ]
