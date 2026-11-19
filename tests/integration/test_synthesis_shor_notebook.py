"""Original Shor Notebook, exact postprocessing and full-state acceptance."""

from fractions import Fraction

import pytest

from examples.synthesis.algorithms.number_theory_and_cryptography.shor.overview.classical import (
    convergents,
    factors_from_order,
    postprocess_phase,
    recover_order,
)

try:
    from matplotlib.patches import Rectangle

    from examples.synthesis.algorithms.number_theory_and_cryptography.shor.overview import (
        workflow as shor,
    )
except ModuleNotFoundError as error:
    if error.name not in {"numpy", "matplotlib"}:
        raise
    pytest.skip(str(error), allow_module_level=True)

pytestmark = pytest.mark.integration


def test_original_continued_fraction_examples() -> None:
    assert convergents(Fraction(17, 1024)) == (
        Fraction(0),
        Fraction(1, 60),
        Fraction(4, 241),
        Fraction(17, 1024),
    )
    values = (0.1953125, 0.39160156, 0.0, 0.59667969, 0.80078125)
    assert tuple(postprocess_phase(Fraction(value), 18) for value in values) == tuple(
        map(Fraction, ("1/5", "2/5", "0", "3/5", "4/5"))
    )
    assert postprocess_phase(Fraction(1, 21), 21) == 0
    assert postprocess_phase(Fraction(1, 20), 21) == Fraction(1, 20)


@pytest.mark.parametrize(
    "base,order,modulus,expected",
    [
        (11, 6, 21, (3, 7)),
        (8, 2, 21, (3, 7)),
        (19, 6, 21, (3, 7)),
        (10, 2, 33, (3, 11)),
        (3, 12, 35, (5, 7)),
        (4, 3, 21, None),
        (20, 2, 21, None),
        (11, 12, 21, None),
    ],
)
def test_mathematical_power_and_source_xor_counterexamples(
    base: int, order: int, modulus: int, expected: tuple[int, int] | None
) -> None:
    assert factors_from_order(base, order, modulus) == expected


@pytest.mark.parametrize(
    "base,order,modulus",
    [(True, 6, 21), (11, 0, 21), (11, 5, 21), (0, 2, 21), (3, 2, 21), (11, 2, 1)],
)
def test_invalid_factoring_inputs(base: int, order: int, modulus: int) -> None:
    with pytest.raises(ValueError):
        factors_from_order(base, order, modulus)


def test_source_strict_threshold_and_zero_phase() -> None:
    fractions, order = recover_order(
        ((Fraction(0), 20), (Fraction(171, 1024), 11), (Fraction(1, 7), 10)), 21
    )
    assert fractions == (Fraction(0), Fraction(1, 6)) and order == 6
    with pytest.raises(ValueError):
        recover_order(((Fraction(171, 1024), 10),), 21)
    with pytest.raises(ValueError):
        recover_order(((Fraction(0), 11),), 21)
    with pytest.raises(ValueError):
        recover_order(((Fraction(1, 2), True),), 21)


@pytest.mark.parametrize("seed", [1700, 1701])
def test_original_full_state_actual_shots_period_factors_and_resources(
    seed: int,
) -> None:
    result = shor.run_estimate(seed=seed)
    assert result.shots == 2048
    assert result.fractions == tuple(
        map(Fraction, ("0", "1/6", "1/3", "1/2", "2/3", "5/6"))
    )
    assert result.order == 6
    assert result.factors == (3, 7)
    assert result.maximum_state_error < 2e-11
    assert result.analytic_state_error < 2e-11
    assert result.dirty_workspace_probability < 1e-24
    assert result.resources.n_qubits == 18
    assert result.resources.n_operations == 2116
    assert result.resources.depth == 1849
    assert len(result.observations) > 6
    assert any(count <= 10 for _, count in result.observations)
    assert all((value * 1024).denominator == 1 for value, _ in result.observations)
    figure = shor.plot_estimate(result)
    bars = figure.axes[0].patches
    assert len(bars) == len(result.observations)
    for bar, (phase, count) in zip(bars, result.observations, strict=True):
        assert isinstance(bar, Rectangle)
        assert bar.get_height() == pytest.approx(count / 2048)
        assert bar.get_x() + bar.get_width() / 2 == pytest.approx(float(phase))
    figure.clear()


@pytest.mark.parametrize("seed", [-1, 2**32, True])
def test_invalid_sampling_seed_refused_before_compile(seed: int) -> None:
    with pytest.raises(ValueError):
        shor.run_estimate(seed=seed)


@pytest.mark.parametrize("phase", [Fraction(-1, 2), Fraction(1)])
def test_out_of_range_phase_refused(phase: Fraction) -> None:
    with pytest.raises(ValueError):
        postprocess_phase(phase, 21)


@pytest.mark.parametrize("modulus", [0, 1, True])
def test_invalid_convergent_modulus_refused(modulus: int) -> None:
    with pytest.raises(ValueError):
        postprocess_phase(Fraction(1, 2), modulus)
