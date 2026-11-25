"""Binary64 normalization remains independent of optional fixed-point objects."""

import numpy as np
import pytest

from flagquantum.simulation.classical_arctangent import normalized_arctangent_value

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "value,expected", ((0, -0.0), (1, -0.5), (-1, 0.5), (2**60, -1.0))
)
def test_exact_landmarks_and_binary64_endpoint(value: int, expected: float) -> None:
    actual = normalized_arctangent_value(value)
    assert isinstance(actual, np.float64) and actual == expected
    assert bool(np.signbit(actual)) == bool(np.signbit(expected))
