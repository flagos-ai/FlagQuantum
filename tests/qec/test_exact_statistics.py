"""Independent rational tail/coverage oracles and pinned numerical references."""

from __future__ import annotations

import math
from decimal import Decimal
from fractions import Fraction

import pytest

from flagquantum.qec.decoders._ai import exact_statistics as statistics
from flagquantum.qec.decoders._ai.assessment import PairedCounts, summarize
from flagquantum.qec.decoders._ai.exact_statistics import (
    exact_rate_interval,
    summarize_exact,
)

pytestmark = pytest.mark.unit


def cdf(k: int, n: int, p: Fraction) -> Fraction:
    return sum(
        (Fraction(math.comb(n, j)) * p**j * (1 - p) ** (n - j) for j in range(k + 1)),
        Fraction(0),
    )


def test_returned_bounds_enclose_exact_rational_binomial_tail_roots() -> None:
    for n in range(1, 17):
        for alpha in (0.05, 0.2, 0.625):
            tail = Fraction.from_float(alpha) / 2
            for k in range(n + 1):
                lower, upper = exact_rate_interval(k, n, alpha)
                assert 0 <= lower <= k / n <= upper <= 1
                if k < n:
                    assert cdf(k, n, Fraction.from_float(upper)) <= tail
                if k > 0:
                    assert 1 - cdf(k - 1, n, Fraction.from_float(lower)) <= tail


def test_small_n_coverage_on_independent_exact_probability_grid() -> None:
    alpha = 0.05
    for n in range(1, 17):
        bounds = [exact_rate_interval(k, n, alpha) for k in range(n + 1)]
        for j in range(41):
            p = Fraction(j, 40)
            covered = sum(
                (
                    Fraction(math.comb(n, k)) * p**k * (1 - p) ** (n - k)
                    for k, (lower, upper) in enumerate(bounds)
                    if Fraction.from_float(lower) <= p <= Fraction.from_float(upper)
                ),
                Fraction(0),
            )
            assert covered >= 1 - Fraction.from_float(alpha)


@pytest.mark.parametrize("n", [1, 10, 1000, 4194304])
def test_analytic_endpoint_bounds_and_reflection(n: int) -> None:
    alpha = 0.05 / 52
    zero = exact_rate_interval(0, n, alpha)
    all_failed = exact_rate_interval(n, n, alpha)
    upper = -math.expm1(math.log(alpha / 2) / n)
    assert zero[0] == 0 and all_failed[1] == 1
    assert zero[1] == pytest.approx(upper, abs=2e-13)
    assert all_failed[0] == pytest.approx(1 - upper, abs=2e-13)
    assert all_failed[0] <= 1 - zero[1]


def test_interval_endpoints_increase_with_failure_count_and_confidence() -> None:
    bounds = [exact_rate_interval(k, 32, 0.05) for k in range(33)]
    assert all(
        a[0] <= b[0] and a[1] <= b[1] for a, b in zip(bounds, bounds[1:], strict=False)
    )
    for k in range(33):
        wider = exact_rate_interval(k, 32, 0.01)
        assert wider[0] <= bounds[k][0] and wider[1] >= bounds[k][1]
        reflected = exact_rate_interval(32 - k, 32, 0.05)
        assert bounds[k][0] == pytest.approx(1 - reflected[1], abs=2e-13)
        assert bounds[k][1] == pytest.approx(1 - reflected[0], abs=2e-13)


# Generated independently with SciPy1.17.1 beta.ppf(alpha/2,k,n-k+1) and
# beta.isf(alpha/2,k+1,n-k), the Clopper-Pearson beta-quantile formulas.
# No SciPy import or runtime dependency is introduced here.
@pytest.mark.parametrize(
    "k,n,alpha,expected",
    [
        (1, 10, 0.05, (0.0025285785444617848, 0.4450161170281953)),
        (5, 10, 0.05, (0.18708602844739855, 0.8129139715526015)),
        (
            5312,
            4194304,
            0.0009615384615384616,
            (0.0012099289927131273, 0.0013248427568946532),
        ),
        (0, 4194304, 0.0009615384615384616, (0.0, 1.8215456519786544e-06)),
        (
            1,
            4194304,
            0.0009615384615384616,
            (1.1465187962607556e-10, 2.3941520547111886e-06),
        ),
        (
            2124,
            4194304,
            0.0009615384615384616,
            (0.000470916982285058, 0.0005437037346924989),
        ),
    ],
)
def test_pinned_independent_exact_interval_reference(
    k: int, n: int, alpha: float, expected: tuple[float, float]
) -> None:
    lower, upper = exact_rate_interval(k, n, alpha)
    assert (lower, upper) == pytest.approx(expected, abs=1.2e-13, rel=0)
    assert lower <= expected[0] + 2e-15
    assert upper >= expected[1] - 2e-15


def test_exact_dyadic_root_is_bracketed_without_choosing_an_uncertain_side() -> None:
    # P[Binomial(4,.5)<=1]=5/16; alpha/2=5/16, so the upper root is exactly .5.
    assert exact_rate_interval(1, 4, 0.625)[1] == pytest.approx(0.5, abs=2e-13)


@pytest.mark.parametrize(
    "k,n,alpha",
    [
        (0, 0, 0.05),
        (0, 4194305, 0.05),
        (True, 10, 0.05),
        (0, True, 0.05),
        (-1, 10, 0.05),
        (11, 10, 0.05),
        (0, 10, True),
        (0, 10, float("nan")),
        (0, 10, float("inf")),
        (0, 10, 0.0),
        (0, 10, 1.0),
        (0, 10, 1e-13),
    ],
)
def test_uncertified_inputs_fail_closed(k: int, n: int, alpha: float) -> None:
    with pytest.raises(ValueError):
        exact_rate_interval(k, n, alpha)


def test_zero_reference_failures_stays_inconclusive_and_zero_candidate_has_risk() -> (
    None
):
    empty = summarize_exact(PairedCounts(100, 0, 0, 0, 0), alpha=0.05 / 26)
    assert empty.ratio_upper is None and empty.quality_gate == "inconclusive"
    assert empty.hybrid_interval[1] > 0
    zero = summarize_exact(PairedCounts(100, 2, 0, 2, 0), alpha=0.05 / 26)
    assert zero.hybrid_ler == 0 and zero.hybrid_interval[1] > 0
    assert zero.quality_gate == "inconclusive"


def test_registered_rare_event_ratio_summary_and_legacy_defaults() -> None:
    # Fixed synthetic count fixture, not observed QEC data or a quality claim.
    counts = PairedCounts(4194304, 5312, 4249, 1600, 537)
    legacy_before = summarize(counts, alpha=0.05 / 26)
    result = summarize_exact(counts, alpha=0.05 / 26)
    assert result.interval_method == "clopper_pearson_outward_binomial_tail"
    assert result.comparison_alpha == 0.05 / 26
    assert result.ratio_upper is not None and result.ratio_upper < 0.9
    assert result.quality_gate == "supported_10_percent_reduction"
    assert legacy_before == summarize(counts, alpha=0.05 / 26)
    assert legacy_before["quality_gate"] == "inconclusive"


@pytest.mark.parametrize(
    "counts",
    [
        PairedCounts(10, 4, 2, 1, 0),
        PairedCounts(10, 4, 2, 5, 3),
        PairedCounts(10, 8, 8, 4, 4),
        PairedCounts(10, True, 0, 1, 0),
    ],
)
def test_impossible_paired_tables_are_refused(counts: PairedCounts) -> None:
    with pytest.raises(ValueError, match="paired counts"):
        summarize_exact(counts, alpha=0.05 / 26)


def test_uncertified_tail_enclosure_is_not_used_to_choose_a_root_side(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        statistics, "_cdf_bounds", lambda *args: (Decimal(0), Decimal(1))
    )
    with pytest.raises(ArithmeticError, match="could not be certified"):
        exact_rate_interval(2, 10, 0.05)
