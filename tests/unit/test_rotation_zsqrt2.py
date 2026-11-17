# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from the locked Qualtran zsqrt2_test.py cases.

import itertools
import math
from typing import Any

import numpy as np
import pytest

from flagquantum.synth import _zsqrt2 as rings
from tests.unit import rotation_ring_fixtures as tu

pytestmark = pytest.mark.unit
_SQRT_2 = math.sqrt(2)


@pytest.mark.parametrize(
    "x", [rings.ZSqrt2(*p) for p in itertools.product(range(-2, 2), repeat=2)]
)
def test_value(x: rings.ZSqrt2) -> None:
    assert float(x) == pytest.approx(x.a + _SQRT_2 * x.b)


@pytest.mark.parametrize("p", range(10))
@pytest.mark.parametrize(
    "x", [rings.ZSqrt2(*p) for p in itertools.product(range(-2, 2), repeat=2)]
)
def test_power(p: int, x: rings.ZSqrt2) -> None:
    a, b = float(x) ** p, float(x**p)
    np.testing.assert_approx_equal(
        float(x) ** p, float(x**p), err_msg=f"{x=} {p=} {a=} {b=}"
    )


@pytest.mark.parametrize(
    "x", [rings.ZSqrt2(*p) for p in itertools.product(range(-2, 2), repeat=2)]
)
def test_conjugate(x: rings.ZSqrt2) -> None:
    assert float(x.conj()) == pytest.approx(x.a - _SQRT_2 * x.b)


@pytest.mark.parametrize(
    "x", [rings.ZSqrt2(*p) for p in itertools.product(range(-2, 2), repeat=2)]
)
@pytest.mark.parametrize(
    "y", [rings.ZSqrt2(*p) for p in itertools.product(range(-2, 2), repeat=2)]
)
def test_add(x: rings.ZSqrt2, y: rings.ZSqrt2) -> None:
    assert float(x + y) == pytest.approx(float(x) + float(y))


@pytest.mark.parametrize(
    "x", [rings.ZSqrt2(*p) for p in itertools.product(range(-2, 2), repeat=2)]
)
@pytest.mark.parametrize(
    "y", [rings.ZSqrt2(*p) for p in itertools.product(range(-2, 2), repeat=2)]
)
def test_sub(x: rings.ZSqrt2, y: rings.ZSqrt2) -> None:
    assert float(x - y) == pytest.approx(float(x) - float(y))


@pytest.mark.parametrize(
    "x", [rings.ZSqrt2(*p) for p in itertools.product(range(-2, 2), repeat=2)]
)
@pytest.mark.parametrize(
    "y", [rings.ZSqrt2(*p) for p in itertools.product(range(-2, 2), repeat=2)]
)
def test_mul(x: rings.ZSqrt2, y: rings.ZSqrt2) -> None:
    assert float(x * y) == pytest.approx(float(x) * float(y))


@pytest.mark.parametrize(
    "x", [rings.ZSqrt2(*p) for p in itertools.product(range(-3, 3), repeat=2)]
)
@pytest.mark.parametrize(
    "y", [rings.ZSqrt2(*p) for p in itertools.product(range(-3, 3), repeat=2)]
)
def test_less_than(x: rings.ZSqrt2, y: rings.ZSqrt2) -> None:
    assert (x < y) == (float(x) < float(y))


@pytest.mark.parametrize(
    "x", [rings.ZSqrt2(*p) for p in itertools.product(range(-3, 3), repeat=2)]
)
@pytest.mark.parametrize(
    "y", [rings.ZSqrt2(*p) for p in itertools.product(range(-3, 3), repeat=2)]
)
def test_less_than_equal(x: rings.ZSqrt2, y: rings.ZSqrt2) -> None:
    assert (x <= y) == (float(x) <= float(y))


@pytest.mark.parametrize("p", tu.PRIMES_LESS_THAN_100)
def test_is_prime_ideal(p: int) -> None:
    pytest.importorskip("sympy")
    for ideal in rings.ZSqrt2.factor_prime(p):
        assert ideal.is_prime_ideal(), f"{ideal=}"


@pytest.mark.parametrize("p", tu.PRIMES_LESS_THAN_100)
def test_factor_prime(p: int) -> None:
    pytest.importorskip("sympy")
    r = rings.ZSqrt2(1, 0)
    for ideal, exponent in rings.ZSqrt2.factor_prime(p).items():
        r = r * ideal**exponent
    assert r == p, f"{p % 8}"


@pytest.mark.parametrize(
    "x", [rings.ZSqrt2(*p) for p in itertools.product(range(-2, 2), repeat=2)]
)
@pytest.mark.parametrize(
    "y", [rings.ZSqrt2(*p) for p in itertools.product(range(-2, 2), repeat=2)]
)
def test_gcd(x: rings.ZSqrt2, y: rings.ZSqrt2) -> None:
    g = x.gcd(y)
    assert x.is_divisible_by(g)
    assert y.is_divisible_by(g)
    assert (x // g).gcd(y // g) == rings.ZSqrt2(1, 0)


@pytest.fixture(scope="module")
def locked_ring() -> Any:
    from hashlib import sha1
    from pathlib import Path

    source = pytest.importorskip("qualtran.rotation_synthesis.rings._zsqrt2")
    folder = Path(source.__file__).parent
    for name, expected in (
        ("_zsqrt2.py", "c94516686151c5cda561e18eb045e0b54166d1f5"),
        ("zsqrt2_test.py", "caa82503b022b450dc141b97b35e36f52e20d958"),
        ("_test_utils.py", "57d62f94b3aef1392b47487d8906c2cdbc246aa1"),
    ):
        content = (folder / name).read_bytes()
        assert sha1(f"blob {len(content)}\0".encode() + content).hexdigest() == expected
    source_fixtures = pytest.importorskip(
        "qualtran.rotation_synthesis.rings._test_utils"
    )
    assert tu.PRIMES_LESS_THAN_100 == source_fixtures.PRIMES_LESS_THAN_100
    return source


def coefficients(value: rings.ZSqrt2) -> tuple[int, int]:
    return value.a, value.b


@pytest.mark.ftqc_reference
def test_exact_order_and_euclidean_steps_match_locked_ring(locked_ring: Any) -> None:
    pairs = tuple(itertools.product(range(-3, 4), repeat=2))
    pairs += ((10**80, 10**80), (10**80, 10**80 + 1), (10**80 + 1, 10**80))
    for x, y in itertools.product(pairs, repeat=2):
        a, b = rings.ZSqrt2(*x), rings.ZSqrt2(*y)
        left, right = locked_ring.ZSqrt2(*x), locked_ring.ZSqrt2(*y)
        assert (a < b, a <= b, a > b, a >= b) == (
            left < right,
            left <= right,
            left > right,
            left >= right,
        )
        assert a.norm() == left.norm()
        assert coefficients(a.gcd(b)) == coefficients(left.gcd(right))
        if b != 0:
            assert coefficients(a // b) == coefficients(left // right)
            assert a.is_divisible_by(b) == left.is_divisible_by(right)


@pytest.mark.ftqc_reference
def test_constants_integer_admission_failures_and_exact_factor_pairs(
    locked_ring: Any,
) -> None:
    for name in (
        "Zero",
        "One",
        "SQRT_2",
        "LAMBDA",
        "LAMBDA_INV",
        "LAMBDA_CONJ",
        "LAMBDA_KLIUCHNIKOV",
        "LAMBDA_KLIUCHNIKOV_CONJ",
    ):
        assert coefficients(getattr(rings, name)) == coefficients(
            getattr(locked_ring, name)
        )

    a, source = rings.ZSqrt2("4", np.int64(-3)), locked_ring.ZSqrt2("4", np.int64(-3))
    assert coefficients(a) == coefficients(source)
    for value in (0, True, np.int64(-3), np.int64(5)):
        assert coefficients(a + value) == coefficients(source + value)
        assert coefficients(a - value) == coefficients(source - value)
        assert coefficients(a * value) == coefficients(source * value)
        assert coefficients(value * a) == coefficients(value * source)
    assert coefficients(a ** np.int64(3)) == coefficients(source ** np.int64(3))
    assert coefficients(a.divide_by_sqrt2()) == coefficients(source.divide_by_sqrt2())
    for implementation in (rings, locked_ring):
        value = implementation.ZSqrt2(3, 2)
        with pytest.raises(ValueError, match="a is even"):
            value.divide_by_sqrt2()
        with pytest.raises(AssertionError):
            _ = value**-1
        with pytest.raises(ZeroDivisionError):
            _ = value // implementation.Zero
        with pytest.raises(TypeError, match="Comparison is not supported"):
            _ = value == "3"
    for prime in tu.PRIMES_LESS_THAN_100:
        assert list(
            (coefficients(root), exp)
            for root, exp in rings.ZSqrt2.factor_prime(prime).items()
        ) == list(
            (coefficients(root), exp)
            for root, exp in locked_ring.ZSqrt2.factor_prime(prime).items()
        )


@pytest.mark.ftqc_reference
def test_radius_helpers_use_the_matching_precision_backend(locked_ring: Any) -> None:
    import mpmath

    from flagquantum.synth._rotation_math import MathConfig, NumpyConfig, with_dps

    source_math = pytest.importorskip("qualtran.rotation_synthesis._math_config")
    with mpmath.workdps(mpmath.mp.dps):
        for local, source in (
            (MathConfig(), source_math.MathConfig()),
            (NumpyConfig, source_math.NumpyConfig),
            (with_dps(70), source_math.with_dps(70)),
        ):
            for n in (0, 1, 4, 30):
                native = rings.radius2_at_n(rings.LAMBDA, n, local)
                expected = locked_ring.radius2_at_n(locked_ring.LAMBDA, n, source)
                assert type(native) is type(expected)
                assert native == expected
                assert rings.radius_at_n(
                    rings.LAMBDA, n, local
                ) == locked_ring.radius_at_n(locked_ring.LAMBDA, n, source)


def test_integral_ring_element_hash_matches_equal_integer() -> None:
    value = rings.ZSqrt2(37)
    assert value == 37 and hash(value) == hash(37)
    elements: dict[int | rings.ZSqrt2, str] = {value: "integer ring element"}
    assert elements[37] == "integer ring element"
