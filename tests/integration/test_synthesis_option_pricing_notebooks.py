"""Fresh original-sized payoff circuits, adaptive sampled estimation and source stages."""

import json
import math
from pathlib import Path

import pytest

import flagquantum as fq
from examples.synthesis.algorithms.amplitude_amplification_and_estimation.quantum_counting.overview.iterative import (
    estimate,
)
from examples.synthesis.applications.finance.option_pricing.overview.model import (
    build_program,
    fresh_model,
)
from examples.synthesis.applications.finance.option_pricing.overview.workflow import (
    IndicatorSampler,
    run_pricing,
)
from flagquantum.compiler import ResourceEstimate

np = pytest.importorskip("numpy")
pytestmark = pytest.mark.integration


@pytest.mark.parametrize("k", range(17))
def test_all_admitted_original_size_powers_match_independent_absolute_states(
    k: int,
) -> None:
    sampler = IndicatorSampler(fresh_model())
    program = sampler.ensure(k)
    check = sampler.checks[k]
    assert check.state_error < 2e-11 and check.named_error < 2e-11
    resources = program.report["resources"]
    assert isinstance(resources, ResourceEstimate)
    assert resources.n_qubits == 6
    assert set(program.registers) == {"indicator"}
    assert fq.run(program).statevector().numel() == 64
    assert sampler.ensure(k) is program


@pytest.mark.parametrize("parameters", ((0.7, 0.13, 1.9), (0.6, 0.2, 1.8)))
def test_fresh_data_and_payoff_truth_table(
    parameters: tuple[float, float, float],
) -> None:
    model = fresh_model(*parameters)
    assert len(model.grid) == len(model.probabilities) == len(model.angles) == 32
    assert sum(model.probabilities) == pytest.approx(1, abs=2e-15)
    for word, (price, probability, angle) in enumerate(
        zip(model.grid, model.probabilities, model.angles, strict=True)
    ):
        assert probability > 0
        assert (word >= model.cutoff) == (price >= model.strike)
        assert math.sin(angle / 2) ** 2 == pytest.approx(
            max(price - model.strike, 0) / model.scale, abs=2e-15
        )
    sampler = IndicatorSampler(model)
    for k in (0, 1, 2, 7):
        sampler.ensure(k)
        assert sampler.checks[k].state_error < 2e-11
    assert sampler.checks[0].probability * model.scale == pytest.approx(
        model.expected_payoff, abs=2e-13
    )


@pytest.mark.parametrize("width,seed,tolerance", ((20, 6100, 5), (28, 6101, 10)))
def test_both_source_workflows_use_actual_count_only_adaptive_estimation(
    width: int, seed: int, tolerance: int
) -> None:
    result, sampler = run_pricing(
        max_width=width, seed=seed, source_tolerance_factor=tolerance
    )
    assert result.estimate.iterations
    assert len(result.estimate.iterations) <= 64
    assert (
        result.estimate.confidence_interval[1] - result.estimate.confidence_interval[0]
        <= 0.1
    )
    assert result.interval[0] <= result.price <= result.interval[1]
    assert abs(result.price - result.model.expected_payoff) <= result.source_tolerance
    assert all(
        row.shots == 256 and 0 <= row.successes <= row.shots
        for row in result.estimate.iterations
    )
    assert set(sampler.programs) == {0} | {row.k for row in result.estimate.iterations}
    assert len(sampler.programs) <= 17
    assert len(result.programs) == len(result.checks) == len(sampler.programs)
    assert all(check.state_error < 2e-11 for check in result.checks)


def test_estimator_receives_counts_only_and_never_exact_payoff() -> None:
    sampler = IndicatorSampler(fresh_model(), seed=25)
    calls: list[tuple[int, int, int]] = []

    def counts(k: int, shots: int) -> int:
        successes = sampler(k, shots)
        calls.append((k, shots, successes))
        return successes

    result = estimate(counts, epsilon=0.05, alpha=0.01)
    assert calls == [(row.k, row.shots, row.successes) for row in result.iterations]
    assert len(calls) >= 2


@pytest.mark.parametrize("k", (-1, 17, True, 1.5))
def test_invalid_structural_powers_fail(k: int) -> None:
    with pytest.raises(ValueError, match="bounded"):
        IndicatorSampler(fresh_model()).ensure(k)


@pytest.mark.parametrize("sigma", (True, 0, -0.1, float("nan"), float("inf")))
def test_nonfinite_or_invalid_distribution_parameters_refused(sigma: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        fresh_model(sigma=sigma)


def test_source_budgets_and_payoff_domain_are_checked() -> None:
    model = fresh_model()
    with pytest.raises(ValueError, match="width"):
        build_program(model, 0, max_width=5)
    with pytest.raises(ValueError, match="strike"):
        fresh_model(strike=99)
    with pytest.raises(ValueError, match="batch"):
        IndicatorSampler(model)(0, True)


@pytest.mark.parametrize(
    "relative,indices",
    (
        (
            "applications/finance/option_pricing/overview",
            (5, 6, 7, 9, 12, 14, 16, 19, 20, 23, 25, 27, 28),
        ),
        (
            "tutorials/workshops/finance_workshops/option_pricing_workshop/overview",
            (5, 6, 7, 9, 12, 15, 17, 20, 23, 25, 27, 28),
        ),
    ),
)
def test_both_complete_source_stage_mappings(
    relative: str, indices: tuple[int, ...]
) -> None:
    root = Path(__file__).resolve().parents[2] / "examples/synthesis" / relative
    proof = json.loads((root / "provenance.json").read_text())
    notebook = json.loads((root / "walkthrough.ipynb").read_text())
    assert tuple(row["index"] for row in proof["reviewed_code_cells"]) == indices
    assert all(len(row["sha256"]) == 64 for row in proof["reviewed_code_cells"])
    assert all(
        not cell.get("outputs")
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


def test_original_fresh_distribution_matches_independent_numpy_formula() -> None:
    model = fresh_model()
    mu, sigma, strike = 0.7, 0.13, 1.9
    mean = np.exp(mu + sigma**2 / 2)
    variance = (np.exp(sigma**2) - 1) * np.exp(2 * mu + sigma**2)
    grid = np.linspace(
        max(0, mean - 3 * np.sqrt(variance)), mean + 3 * np.sqrt(variance), 32
    )
    density = np.exp(-0.5 * ((np.log(grid) - mu) / sigma) ** 2) / (
        grid * sigma * np.sqrt(2 * np.pi)
    )
    probabilities = density / density.sum()
    np.testing.assert_allclose(model.grid, grid, atol=1e-15, rtol=0)
    np.testing.assert_allclose(model.probabilities, probabilities, atol=1e-15, rtol=0)
    expected = float(np.maximum(grid - strike, 0) @ probabilities)
    assert model.expected_payoff == pytest.approx(expected, abs=2e-15)
