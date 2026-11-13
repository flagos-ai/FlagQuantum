"""Original QMCI sizes, finite-grid meaning, local observations and full states."""

import math

import pytest

try:
    import matplotlib.pyplot as plt
    import numpy as np

    from examples.synthesis.tutorials.basics.qmci_pi_estimation.overview import (
        workflow as qmci,
    )
except ModuleNotFoundError as error:
    if error.name not in {"numpy", "matplotlib"}:
        raise
    pytest.skip(str(error), allow_module_level=True)

from examples.synthesis.algorithms.amplitude_amplification_and_estimation.quantum_counting.overview.iterative import (
    Estimate,
)
from examples.synthesis.tutorials.basics.qmci_pi_estimation.overview.operations import (
    build_program,
)

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def estimates() -> tuple[tuple[Estimate, qmci.IndicatorSampler], ...]:
    return tuple(qmci.run_iterative(bits, 1800 + bits) for bits in range(1, 5))


def test_original_three_bit_qpe_preserves_reporting_formula() -> None:
    row = qmci.sample_qpe()
    assert sum(n for _, n in row.observations) == 2048
    assert row.phase.denominator <= 8
    assert row.source_pi == 4 * math.sin(math.pi * float(row.phase))
    assert row.squared_probability_pi == 4 * math.sin(math.pi * float(row.phase)) ** 2
    assert row.source_pi != pytest.approx(row.squared_probability_pi)
    assert row.state_check.maximum_state_error < 2e-10
    assert row.state_check.dirty_workspace_probability < 1e-24


def test_four_lattices_use_count_only_estimation_and_independent_full_states(
    estimates: tuple[tuple[Estimate, qmci.IndicatorSampler], ...],
) -> None:
    for bits, (result, sampler) in enumerate(estimates, 1):
        exact = qmci.grid_pi(1 << bits) / 4
        assert result.confidence_interval[0] <= exact <= result.confidence_interval[1]
        assert result.confidence_interval[1] - result.confidence_interval[0] <= 0.06
        assert len(result.iterations) <= 64
        assert sum(row.shots for row in result.iterations) >= 512
        assert set(sampler.checks) == {row.k for row in result.iterations}
        theta = math.asin(math.sqrt(exact))
        for k, check in sampler.checks.items():
            assert check.maximum_state_error < 2e-10
            assert check.dirty_workspace_probability < 1e-24
            assert sampler.probabilities[k] == pytest.approx(
                math.sin((2 * k + 1) * theta) ** 2, abs=2e-10
            )
        assert 4 * result.estimate > math.pi
    assert tuple(qmci.grid_pi(1 << bits) / 4 for bits in range(1, 5)) == (
        1.0,
        0.9375,
        0.875,
        0.8359375,
    )


def test_original_classical_grid_sizes_and_figures(
    estimates: tuple[tuple[Estimate, qmci.IndicatorSampler], ...],
) -> None:
    assert tuple(qmci.grid_pi(n) for n in (3, 5, 6)) == pytest.approx(
        (4.0, 3.52, 11 / 3)
    )
    lattice = qmci.plot_lattice()
    comparison = qmci.plot_estimates(tuple(row for row, _ in estimates))
    axis = lattice.axes[0]
    assert (
        sum(
            np.asarray(collection.get_offsets()).shape[0]
            for collection in axis.collections
        )
        == 256
    )
    assert axis.get_aspect() == 1.0
    assert comparison.axes[0].get_xscale() == "log"
    assert "not actual shots" in comparison.axes[0].get_xlabel()
    plt.close(lattice)
    plt.close(comparison)


@pytest.mark.parametrize("bits,k", [(0, 0), (5, 0), (True, 0), (2, -1), (2, 17)])
def test_workflow_bounds_are_checked(bits: int, k: int) -> None:
    with pytest.raises(ValueError):
        build_program(bits, k)


@pytest.mark.parametrize("k,shots", [(-1, 256), (17, 256), (0, 0), (0, 2049)])
def test_sampler_refuses_invalid_requests_without_execution(k: int, shots: int) -> None:
    with pytest.raises(ValueError):
        qmci.IndicatorSampler(3, 1803)(k, shots)
