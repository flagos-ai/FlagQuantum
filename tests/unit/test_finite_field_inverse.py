"""Original inverse scalar and live-junk callbacks across the whole source domain."""

from __future__ import annotations

from importlib.metadata import version
from typing import Any

import numpy as np
import pytest

from flagquantum.simulation.finite_field_addition import binary_field_encoding
from flagquantum.simulation.finite_field_inverse import field_inverse_classical_values

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def locked_reference() -> None:
    import kickmix

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")


@pytest.mark.parametrize("width", range(1, 9))
def test_all_field_elements_and_exact_original_junk(width: int) -> None:
    from qualtran.bloqs.gf_arithmetic.gf2_inverse import GF2Inverse

    encoding = binary_field_encoding(width)
    original = GF2Inverse(width)
    for value in encoding.gf_type.elements:
        source, result, junk = field_inverse_classical_values(encoding, value)
        expected = original.on_classical_vals(x=value)
        assert source is value
        assert result == expected["result"]
        np.testing.assert_array_equal(junk, expected["junk"])
        assert junk.dtype == expected["junk"].dtype
        assert (int(result) == 0) if int(value) == 0 else value * result == 1
    invalid: Any = 1
    with pytest.raises(AssertionError):
        field_inverse_classical_values(encoding, invalid)


@pytest.mark.parametrize("width", [9, 12, 16])
def test_larger_fields_original_junk_shapes_and_values(width: int) -> None:
    from qualtran.bloqs.gf_arithmetic.gf2_inverse import GF2Inverse

    encoding = binary_field_encoding(width)
    original = GF2Inverse(width)
    for value in [0, 1, 2, 3, (1 << width) - 1]:
        source, result, junk = field_inverse_classical_values(
            encoding, encoding.gf_type(value)
        )
        expected = original.on_classical_vals(x=source)
        assert result == expected["result"]
        np.testing.assert_array_equal(junk, expected["junk"])
        assert junk.dtype == expected["junk"].dtype
