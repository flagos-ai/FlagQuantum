"""Compare the complete frozen Kaiser numerical expression without a circuit."""

import numpy as np
import pytest

from flagquantum.simulation.numerics.kaiser_window import kaiser_window_coefficients

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("width", [0, 1, 2, 5, 8])
@pytest.mark.parametrize("alpha", [0.0, 0.25, 2.0, -2.0])
def test_original_periodic_window_coefficients_exact(width: int, alpha: float) -> None:
    from qualtran.bloqs.phase_estimation.kaiser_window_state import KaiserWindowState

    expected = KaiserWindowState(width, alpha).kaiser_state_coeff
    actual = kaiser_window_coefficients(width, alpha)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float64
    np.testing.assert_allclose(np.linalg.norm(actual), 1, atol=2e-16, rtol=0)


def test_original_overflow_and_negative_width_failures() -> None:
    from qualtran.bloqs.phase_estimation.kaiser_window_state import KaiserWindowState

    with np.errstate(invalid="ignore", over="ignore"):
        actual = kaiser_window_coefficients(3, 1e6)
        original = KaiserWindowState(3, 1e6).kaiser_state_coeff
    np.testing.assert_array_equal(actual, original)
    assert np.isnan(actual).all()
    for evaluate in (
        lambda: kaiser_window_coefficients(-1, 2),
        lambda: KaiserWindowState(-1, 2).kaiser_state_coeff,
    ):
        with pytest.raises(ValueError):
            evaluate()
