"""Complete FFT complement source, intermediate transforms and GQSP consumer."""

from hashlib import sha1
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from numpy.typing import NDArray

from flagquantum.simulation.numerics.qsp_fft_complement import (
    fft_complementary_polynomial,
)
from flagquantum.simulation.numerics.qsp_sampling import random_qsp_polynomial
from tests.unit.test_qsp_optimization import assert_gqsp_append_pair

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("real", [False, True])
@pytest.mark.parametrize("length,modes", [(1, 1), (3, 5), (4, 8), (10, 500), (40, 50)])
def test_coefficients_scaling_transform_operands_and_source_order(
    real: bool, length: int, modes: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    from qualtran.bloqs.qsp.fft_qsp import fft_complementary_polynomial as source

    p = random_qsp_polynomial(
        length, random_state=np.random.RandomState(42), only_real_coeffs=real
    )
    calls: list[tuple[str, NDArray[np.complex128], str | None]] = []
    original_fft, original_ifft = np.fft.fft, np.fft.ifft

    def fft(
        x: NDArray[np.complex128],
        *,
        norm: Literal["backward", "ortho", "forward"] | None = None,
    ) -> NDArray[np.complex128]:
        calls.append(("fft", x.copy(), norm))
        return original_fft(x, norm=norm)

    def ifft(
        x: NDArray[np.complex128],
        *,
        norm: Literal["backward", "ortho", "forward"] | None = None,
    ) -> NDArray[np.complex128]:
        calls.append(("ifft", x.copy(), norm))
        return original_ifft(x, norm=norm)

    monkeypatch.setattr(np.fft, "fft", fft)
    monkeypatch.setattr(np.fft, "ifft", ifft)
    before = np.array(p)
    actual = fft_complementary_polynomial(tuple(p), tolerance=1e-9, num_modes=modes)
    native_calls = calls.copy()
    calls.clear()
    expected = source(tuple(p), tolerance=1e-9, num_modes=modes)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(p, before)
    assert len(native_calls) == len(calls) == 4
    for native_call, source_call in zip(native_calls, calls, strict=True):
        assert native_call[0] == source_call[0]
        assert native_call[2] == source_call[2] == "forward"
        np.testing.assert_array_equal(native_call[1], source_call[1])
    assert native_calls[0][1].size == length + modes - 1
    np.testing.assert_array_equal(native_calls[0][1][:length], (1 - 1e-9 / 4) * before)


@pytest.mark.parametrize("coefficients,modes", [((), 1), ((0.2,), 0), ((0.2,), -2)])
def test_original_padding_and_transform_failures(
    coefficients: tuple[float, ...], modes: int
) -> None:
    from qualtran.bloqs.qsp.fft_qsp import fft_complementary_polynomial as source

    for function in (source, fft_complementary_polynomial):
        with pytest.raises(ValueError):
            function(coefficients, num_modes=modes)


def test_unbounded_input_retains_invalid_log_and_nan() -> None:
    from qualtran.bloqs.qsp.fft_qsp import fft_complementary_polynomial as source

    for function in (source, fft_complementary_polynomial):
        with pytest.warns(RuntimeWarning, match="invalid value encountered in log"):
            result = function((2.0, 0.1))
        assert np.isnan(result).all()


@pytest.mark.parametrize("real", [False, True])
@pytest.mark.parametrize("negative_power", [0, 1, 5])
def test_full_append_matrix_absolute_phase_and_zero_idle(
    real: bool, negative_power: int
) -> None:
    p = tuple(
        random_qsp_polynomial(
            3, random_state=np.random.RandomState(62), only_real_coeffs=real
        )
    )
    q = tuple(fft_complementary_polynomial(p, tolerance=1e-9))
    assert_gqsp_append_pair(p, q, negative_power)
    z = np.exp(1j * np.linspace(0, 2 * np.pi, 129))
    np.testing.assert_allclose(
        abs(np.polynomial.polynomial.polyval(z, p)) ** 2
        + abs(np.polynomial.polynomial.polyval(z, q)) ** 2,
        1,
        atol=1e-9,
        rtol=0,
    )


def test_frozen_complete_source_and_test_blobs() -> None:
    from qualtran.bloqs.qsp import fft_qsp

    root = Path(fft_qsp.__file__).parent
    for name, expected in (
        ("fft_qsp.py", "855d7a61b37516315fa4873c682754a4dcc11ef5"),
        ("fft_qsp_test.py", "c2dd5eb4d8e774ed1280633b70d5934bbb12d02c"),
    ):
        raw = (root / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == expected
