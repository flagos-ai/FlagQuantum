"""Root selection and diagnostic boundaries match unchanged source."""

import numpy as np
import pytest
from numpy.polynomial import Polynomial
from numpy.typing import NDArray

from flagquantum.simulation.numerics.qsp_complement import qsp_complementary_polynomial
from flagquantum.simulation.numerics.qsp_sampling import (
    check_gqsp_polynomial_pair_on_random_points_on_unit_circle,
    random_qsp_polynomial,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("degree", [2, 3, 5, 10])
@pytest.mark.parametrize("real", [False, True])
@pytest.mark.parametrize("verify", [False, True])
def test_source_coefficients_and_complement_identity(
    degree: int, real: bool, verify: bool
) -> None:
    from qualtran.bloqs.qsp.generalized_qsp import (
        qsp_complementary_polynomial as source,
    )

    p = random_qsp_polynomial(
        degree, random_state=np.random.RandomState(42), only_real_coeffs=real
    )
    before = np.array(p)
    actual = qsp_complementary_polynomial(p, verify=verify)
    expected = source(p, verify=verify)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(p, before)
    check_gqsp_polynomial_pair_on_random_points_on_unit_circle(
        p,
        tuple(complex(c) for c in actual),
        random_state=np.random.RandomState(19),
        n_points=1000,
    )
    z = np.exp(1j * np.linspace(0, 2 * np.pi, 257))
    np.testing.assert_allclose(
        abs(Polynomial(p)(z)) ** 2 + abs(Polynomial(actual)(z)) ** 2,
        1,
        atol=2e-12,
        rtol=0,
    )


@pytest.mark.parametrize(
    "p", [(), (0.5,), (1.0,), (2.0,), (0.0,), (0.0, 0.0), (0.0, 0.5), (0.5, 0.5)]
)
@pytest.mark.parametrize("verify", [False, True])
def test_constant_empty_and_degenerate_source_boundaries(
    p: tuple[complex, ...], verify: bool
) -> None:
    from qualtran.bloqs.qsp.generalized_qsp import (
        qsp_complementary_polynomial as source,
    )

    try:
        expected = source(p, verify=verify)
    except (ValueError, AssertionError) as error:
        with pytest.raises(type(error)) as native:
            qsp_complementary_polynomial(p, verify=verify)
        assert str(native.value) == str(error)
    else:
        np.testing.assert_array_equal(
            qsp_complementary_polynomial(p, verify=verify), expected
        )


@pytest.mark.parametrize(
    "roots",
    [
        (1 + 0j, 1 + 0j),
        (1 + 0j,),
        (0j,),
        (0.5 + 0j, 2 + 0j),
        (0.5 + 0j,),
        (0.4 + 0j, 2 + 0j),
        (1 + 0j, -1 + 0j, 1 + 0j, -1 + 0j),
    ],
)
@pytest.mark.parametrize("verify", [False, True])
def test_root_diagnostic_paths_preserve_order(
    roots: tuple[complex, ...], verify: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from qualtran.bloqs.qsp.generalized_qsp import (
        qsp_complementary_polynomial as source,
    )

    def supplied(self: Polynomial) -> NDArray[np.complex128]:
        return np.array(roots, dtype=np.complex128)

    monkeypatch.setattr(Polynomial, "roots", supplied)
    # Injected roots cover diagnostic paths only; not an independent complement proof.
    try:
        expected = source((0.2, 0.3), verify=verify)
    except (AssertionError, ValueError) as error:
        with pytest.raises(type(error)) as native:
            qsp_complementary_polynomial((0.2, 0.3), verify=verify)
        assert str(native.value) == str(error)
    else:
        np.testing.assert_array_equal(
            qsp_complementary_polynomial((0.2, 0.3), verify=verify), expected
        )


def test_constant_and_trailing_zero_are_not_repaired() -> None:
    from qualtran.bloqs.qsp.generalized_qsp import (
        qsp_complementary_polynomial as source,
    )

    for evaluator in (source, qsp_complementary_polynomial):
        # Empty interpolation roots fail even for a bounded constant.
        with pytest.raises(ValueError, match="Coefficient array is empty"):
            evaluator((0.5,), verify=True)
        with pytest.raises(AssertionError, match="zero root"):
            evaluator((0.5, 0.0), verify=True)
