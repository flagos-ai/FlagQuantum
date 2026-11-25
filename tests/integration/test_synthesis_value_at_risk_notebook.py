"""Original-sized native VaR, exclusive thresholds and sample-only search semantics."""

import json
import math
from pathlib import Path

import pytest

from examples.synthesis.applications.finance.value_at_risk.overview.model import (
    discrete_quantile,
    fresh_model,
    search,
)
from examples.synthesis.applications.finance.value_at_risk.overview.quantum import main
from examples.synthesis.applications.finance.value_at_risk.overview.workflow import (
    ThresholdSampler,
    run_risk,
)
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.errors import PlanningError

np = pytest.importorskip("numpy")
pytestmark = pytest.mark.integration


@pytest.mark.parametrize("threshold", (0, 1, 32, 64, 127, 128))
@pytest.mark.parametrize("k", (0, 1, 2))
def test_exclusive_marker_boundaries_and_budget_admitted_reflections(
    threshold: int, k: int
) -> None:
    sampler = ThresholdSampler(fresh_model())
    program = sampler.ensure(threshold, k)
    check = sampler.checks[threshold, k]
    assert check.state_error < 3e-11 and check.named_error < 3e-11
    resources = program.report["resources"]
    assert isinstance(resources, ResourceEstimate)
    assert resources.n_qubits == 8 + check.workspace_qubits <= 28
    assert 0 <= check.workspace_qubits <= 20 and check.workspace_error < 3e-11
    assert set(program.registers) == {"indicator"}
    assert sampler.ensure(threshold, k) is program


def test_another_nonzero_fresh_distribution_matches_independent_full_states() -> None:
    model = fresh_model(mu=0.6, sigma=0.2)
    sampler = ThresholdSampler(model)
    for threshold, k in ((19, 0), (50, 1), (72, 2)):
        sampler.ensure(threshold, k)
        assert sampler.checks[threshold, k].state_error < 3e-11


def test_fresh_distribution_exclusive_cdf_and_discrete_quantile() -> None:
    model = fresh_model()
    mean = np.exp(0.7 + 0.13**2 / 2)
    variance = (np.exp(0.13**2) - 1) * np.exp(1.4 + 0.13**2)
    grid = np.linspace(
        max(0, mean - 3 * np.sqrt(variance)), mean + 3 * np.sqrt(variance), 128
    )
    density = np.exp(-0.5 * ((np.log(grid) - 0.7) / 0.13) ** 2) / (
        grid * 0.13 * np.sqrt(2 * np.pi)
    )
    probabilities = density / density.sum()
    np.testing.assert_allclose(model.grid, grid, atol=1e-15, rtol=0)
    np.testing.assert_allclose(model.probabilities, probabilities, atol=1e-15, rtol=0)
    for index in range(129):
        assert model.alpha(index) == pytest.approx(
            float(probabilities[:index].sum()), abs=2e-15
        )
    index, price = discrete_quantile(model)
    assert sum(model.probabilities[:index]) <= 0.07
    assert sum(model.probabilities[: index + 1]) > 0.07
    assert price == model.grid[index]


def test_source_search_preserves_sample_decisions_and_can_exhaust() -> None:
    model = fresh_model()
    calls: list[int] = []

    def low_counts_only(index: int) -> float:
        calls.append(index)
        return 0.0

    result = search(model, low_counts_only)
    assert calls == [32, 48, 56, 60, 62, 63]
    assert [step.search_size for step in result.steps] == [16, 8, 4, 2, 1, 0]
    assert result.index == 63 and not result.tolerance_reached
    assert result.original_return is None
    assert result.steps[0].value == model.probabilities[32]
    assert result.steps[1].value == model.grid[48]

    close = search(model, lambda _: 0.071)
    assert len(close.steps) == 1 and close.tolerance_reached


@pytest.mark.parametrize("seed", (6200, 6201))
def test_complete_original_search_uses_actual_fresh_counts(seed: int) -> None:
    model, classical, measured, sampler = run_risk(seed)
    assert classical.original_return is measured.original_return is None
    assert classical == search(model, model.alpha)
    assert 1 <= len(measured.steps) == len(sampler.estimates) <= 6
    assert tuple(step.index for step in measured.steps) == tuple(
        index for index, _ in sampler.estimates
    )
    keys: set[tuple[int, int]] = set()
    for threshold, result in sampler.estimates:
        assert result.confidence_interval[1] - result.confidence_interval[0] <= 0.1
        assert 1 <= len(result.iterations) <= 64
        assert all(
            row.shots == 256 and 0 <= row.successes <= row.shots
            for row in result.iterations
        )
        keys.update((threshold, row.k) for row in result.iterations)
    assert set(sampler.programs) == keys
    assert all(check.state_error < 3e-11 for check in sampler.checks.values())
    assert measured.tolerance_reached == math.isclose(
        measured.steps[-1].alpha, 0.07, abs_tol=0.007, rel_tol=1e-5
    )
    assert measured.tolerance_reached or measured.steps[-1].search_size == 0


@pytest.mark.parametrize("threshold", (-1, 129, True, 1.5))
def test_invalid_thresholds_fail_before_synthesis(threshold: int) -> None:
    sampler = ThresholdSampler(fresh_model())
    with pytest.raises(ValueError, match="threshold"):
        sampler.ensure(threshold, 0)
    assert not sampler.programs


@pytest.mark.parametrize("k", (-1, 17, True, 1.5))
def test_invalid_powers_fail_before_synthesis(k: int) -> None:
    sampler = ThresholdSampler(fresh_model())
    with pytest.raises(ValueError, match="power"):
        sampler.ensure(32, k)
    assert not sampler.programs


@pytest.mark.parametrize("sigma", (True, 0, -1, float("nan"), float("inf")))
def test_distribution_domain_rejected(sigma: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        fresh_model(sigma=sigma)


def test_source_stage_mapping_and_notebook_are_complete_and_unexecuted() -> None:
    root = (
        Path(__file__).resolve().parents[2]
        / "examples/synthesis/applications/finance/value_at_risk/overview"
    )
    proof = json.loads((root / "provenance.json").read_text())
    notebook = json.loads((root / "walkthrough.ipynb").read_text())
    assert tuple(row["index"] for row in proof["reviewed_code_cells"]) == (
        2,
        4,
        6,
        8,
        9,
        10,
        11,
        14,
        15,
        16,
        17,
    )
    assert all(len(row["sha256"]) == 64 for row in proof["reviewed_code_cells"])
    assert all(
        not cell.get("outputs")
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


@pytest.mark.parametrize("budget", ("operations", "width", "depth", "memory"))
def test_actual_eight_bit_family_respects_resource_rejection(budget: str) -> None:
    model = fresh_model()
    bindings: dict[str, int | list[float]] = {
        "probabilities": list(model.probabilities),
        "threshold": 34,
        "k": 1,
    }
    with pytest.raises((HybridVerificationError, synth.SynthesisError, PlanningError)):
        if budget == "operations":
            synth.synthesize(main, classical_inputs=bindings, max_operations=1)
        elif budget == "width":
            synth.synthesize(main, classical_inputs=bindings, max_qubits=7)
        elif budget == "depth":
            synth.synthesize(main, classical_inputs=bindings, max_depth=1)
        else:
            synth.synthesize(main, classical_inputs=bindings, memory_limit_bytes=16)


@pytest.mark.parametrize("shots", (0, 2049, True, 1.5))
def test_sample_budget_rejected_before_building(shots: int) -> None:
    sampler = ThresholdSampler(fresh_model())
    with pytest.raises(ValueError, match="batch"):
        sampler.sample(32, 0, shots)
    assert not sampler.programs
