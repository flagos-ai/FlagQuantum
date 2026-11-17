# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from the locked Qualtran zw_test.py cases.

import itertools
from typing import Any

import numpy as np
import pytest

from flagquantum.synth import _rings as rings
from flagquantum.synth import _rotation_math as mc
from tests.unit import rotation_ring_fixtures as tu

pytestmark = pytest.mark.unit

_Z8 = np.exp(1j * np.pi / 4)
_SQRT_2 = np.sqrt(2)


def _make_pairs(n_pairs: int, seed: int = 0) -> list[tuple[rings.ZW, rings.ZW]]:
    rng = np.random.default_rng(seed)
    zw_numbers = tuple(rings.ZW(p) for p in itertools.product(range(-1, 2), repeat=4))
    all_pairs = np.array(tuple(itertools.product(zw_numbers, repeat=2)))
    return [
        (left, right) for left, right in all_pairs[rng.choice(len(all_pairs), n_pairs)]
    ]


@pytest.mark.parametrize("n", range(31))
@pytest.mark.parametrize("p", tuple(itertools.product(range(2), repeat=4)))
def test_pow(n: int, p: tuple[int, int, int, int]) -> None:
    x = sum(p[i] * _Z8**i for i in range(4))
    got = complex(rings.ZW(p) ** n)
    want = x**n
    assert np.isclose(got, want), f"{x=} {n=} {p=} {got=} {want=}"


@pytest.mark.parametrize("p", tuple(itertools.product(range(-2, 3), repeat=4)))
def test_conjugate(p: tuple[int, int, int, int]) -> None:
    v = rings.ZW(p)
    c = v.conjugate()
    assert np.isclose(v.mag(_SQRT_2), c.mag(_SQRT_2))
    assert np.isclose(complex(v * v.conjugate()).imag, 0), (
        f"{v=} {v.conjugate()} {v * v.conjugate()}"
    )
    assert np.isclose(v.arg(mc.NumpyConfig) + c.arg(mc.NumpyConfig), 0) or np.isclose(
        v.arg(mc.NumpyConfig) + c.arg(mc.NumpyConfig), 2 * np.pi
    )


@pytest.mark.parametrize(["a", "b"], _make_pairs(100))
def test_add(a: rings.ZW, b: rings.ZW) -> None:
    np.testing.assert_allclose(complex(a + b), complex(a) + complex(b))


@pytest.mark.parametrize(["a", "b"], _make_pairs(100))
def test_sub(a: rings.ZW, b: rings.ZW) -> None:
    np.testing.assert_allclose(complex(a - b), complex(a) - complex(b))


@pytest.mark.parametrize(["a", "b"], _make_pairs(100))
def test_mul(a: rings.ZW, b: rings.ZW) -> None:
    np.testing.assert_allclose(complex(a * b), complex(a) * complex(b))


@pytest.mark.parametrize("a", range(-3, 3))
@pytest.mark.parametrize("bp", tuple(itertools.product(range(-1, 2), repeat=4)))
def test_radd(a: int, bp: tuple[int, int, int, int]) -> None:
    b = rings.ZW(bp)
    np.testing.assert_allclose(complex(a + b), complex(a) + complex(b))


@pytest.mark.parametrize("a", range(-3, 3))
@pytest.mark.parametrize("bp", tuple(itertools.product(range(-1, 2), repeat=4)))
def test_rsub(a: int, bp: tuple[int, int, int, int]) -> None:
    b = rings.ZW(bp)
    np.testing.assert_allclose(complex(a - b), complex(a) - complex(b))


@pytest.mark.parametrize("a", range(-3, 3))
@pytest.mark.parametrize("bp", tuple(itertools.product(range(-1, 2), repeat=4)))
def test_rmul(a: int, bp: tuple[int, int, int, int]) -> None:
    b = rings.ZW(bp)
    np.testing.assert_allclose(complex(a * b), complex(a) * complex(b))


@pytest.mark.parametrize(
    "x", [rings.ZW(p) for p in itertools.product(range(-2, 3), repeat=4)]
)
def test_conversion_roundtrip(x: rings.ZW) -> None:
    assert x == rings.ZW.from_pair(*x.to_zsqrt2())


def _create_random_elements(n: int, seed: int = 0) -> list[rings.ZW]:
    rng = np.random.default_rng(seed)
    return [rings.ZW(c) for c in rng.integers(-10, 10, size=(n, 4))]


@pytest.mark.parametrize("x", _create_random_elements(20))
def test_complex_conjugate(x: rings.ZW) -> None:
    np.testing.assert_allclose(
        (x * x.conj()).value(_SQRT_2), abs(x.value(_SQRT_2)) ** 2
    )


@pytest.mark.parametrize("x", _create_random_elements(50))
def test_sqrt2_conjugate(x: rings.ZW) -> None:
    a, b, need_w = x.to_zsqrt2()
    if need_w:
        assert x.sqrt2_conj() == rings.ZW.from_pair(
            a.conj(), b.conj(), False
        ) - rings.ZW([0, 1, 0, 0])
    else:
        assert x.sqrt2_conj() == rings.ZW.from_pair(a.conj(), b.conj(), False)


@pytest.mark.parametrize("p", [2, 3, 5, 7, 11, 13])
@pytest.mark.parametrize("x", _create_random_elements(100))
def test_gcd(p: int, x: rings.ZW) -> None:
    y = rings.ZW([p, 0, 0, 0])
    g = y.gcd(x)
    assert y.is_divisible_by(g)
    assert x.is_divisible_by(g)
    # The gcd in Z[W] is unique up to multiplication by a unit of Z[W].
    assert (x // g).gcd(y // g).norm() == 1


@pytest.mark.parametrize("p", tu.PRIMES_LESS_THAN_100)
def test_factor_prime(p: int) -> None:
    pytest.importorskip("sympy")
    r = rings.ZW([1, 0, 0, 0])
    for ideal, exponent in rings.ZW.factor_prime(p).items():
        r = r * ideal**exponent
    if p == 2 or p % 8 == 1:
        assert all(c % p == 0 for c in r.coords)
    else:
        assert r.coords[1:] == (0, 0, 0)
        assert r.coords[0] % p == 0


@pytest.fixture(scope="module")
def locked_zw() -> Any:
    from hashlib import sha1
    from pathlib import Path

    original = pytest.importorskip("qualtran.rotation_synthesis.rings._zw")
    for name, expected in (
        ("__init__.py", "4eb7949e5ab78f63a5d09ae391f8fd641d6b2552"),
        ("_zw.py", "c8bf840eaa53797af105eb0c037193c294381868"),
        ("zw_test.py", "cc136951d5e38da3bdfd56a42e71424049d13416"),
    ):
        content = Path(original.__file__).with_name(name).read_bytes()
        assert sha1(f"blob {len(content)}\0".encode() + content).hexdigest() == expected
    return original


@pytest.mark.ftqc_reference
def test_internal_ring_entry_points_match_locked_exports(locked_zw: Any) -> None:
    import ast
    from pathlib import Path

    from flagquantum.synth import _zsqrt2, _zw

    frozen_init = Path(locked_zw.__file__).with_name("__init__.py")
    package = pytest.importorskip("qualtran.rotation_synthesis.rings")
    expected = ("ZSqrt2", "ZW")
    statements = ast.parse(frozen_init.read_bytes()).body
    names: list[str] = []
    for node in statements[1:]:
        assert isinstance(node, ast.ImportFrom)
        names.extend(alias.name for alias in node.names)
    assert tuple(names) == expected
    assert tuple(rings.__all__) == expected
    assert package.ZW is locked_zw.ZW
    assert package.ZSqrt2 is locked_zw._zsqrt2.ZSqrt2
    assert rings.ZW is _zw.ZW
    assert rings.ZSqrt2 is _zsqrt2.ZSqrt2


@pytest.mark.ftqc_reference
def test_exact_operations_and_subring_round_trip_match_source(locked_zw: Any) -> None:
    values = _create_random_elements(14, seed=21) + [
        rings.ZW([0, 0, 0, 0]),
        rings.ZW([0, -(10**40), 1, 10**40]),
        rings.ZW([10**40, 1, -(10**40), -3]),
    ]
    for a, b in itertools.product(values, repeat=2):
        left, right = locked_zw.ZW(a.coords), locked_zw.ZW(b.coords)
        assert (a + b).coords == (left + right).coords
        assert (a - b).coords == (left - right).coords
        assert (a * b).coords == (left * right).coords
        assert a.norm() == left.norm()
        assert a.gcd(b).coords == left.gcd(right).coords
        if b.coords != (0, 0, 0, 0):
            assert (a // b).coords == (left // right).coords
            assert a.is_divisible_by(b) == left.is_divisible_by(right)
        local_real, local_imag, local_omega = a.to_zsqrt2()
        source_real, source_imag, source_omega = left.to_zsqrt2()
        assert (
            local_real.a,
            local_real.b,
            local_imag.a,
            local_imag.b,
            local_omega,
        ) == (
            source_real.a,
            source_real.b,
            source_imag.a,
            source_imag.b,
            source_omega,
        )
        assert rings.ZW.from_pair(local_real, local_imag, local_omega) == a
        assert a.conj().coords == left.conj().coords
        assert a.sqrt2_conj().coords == left.sqrt2_conj().coords


@pytest.mark.ftqc_reference
def test_rotation_ring_values_factors_and_input_boundaries(locked_zw: Any) -> None:
    import mpmath

    from flagquantum.synth import _zw

    reference_math = pytest.importorskip("qualtran.rotation_synthesis._math_config")
    with mpmath.workdps(mpmath.mp.dps):
        for name in (
            "Zero",
            "One",
            "J",
            "Omega",
            "SQRT_2",
            "LAMBDA_KLIUCHNIKOV",
            "LAMBDA_KLIUCHNIKOV_SQRT2_CONJ",
        ):
            native, source = getattr(_zw, name), getattr(locked_zw, name)
            assert native.coords == source.coords
            assert hash(native) == hash(source)
        for local, original in (
            (mc.MathConfig(), reference_math.MathConfig()),
            (mc.NumpyConfig, reference_math.NumpyConfig),
            (mc.with_dps(70), reference_math.with_dps(70)),
        ):
            a, b = rings.ZW([1, 2, -3, 1]), locked_zw.ZW([1, 2, -3, 1])
            assert type(a.value(local.sqrt2)) is type(b.value(original.sqrt2))
            assert a.value(local.sqrt2) == b.value(original.sqrt2)
            assert a.polar(local) == b.polar(original)
    for prime in tu.PRIMES_LESS_THAN_100:
        assert list(
            (root.coords, exp) for root, exp in rings.ZW.factor_prime(prime).items()
        ) == list(
            (root.coords, exp) for root, exp in locked_zw.ZW.factor_prime(prime).items()
        )
    assert rings.ZW(np.array([1, 0, 2, 3], dtype=np.int64)).coords == (1, 0, 2, 3)
    for cls in (rings.ZW, locked_zw.ZW):
        value = cls([1, 0, 0, 0])
        assert (value ** np.int64(0)).coords == (1, 0, 0, 0)
        with pytest.raises(AssertionError):
            cls((1, 1, 1))
        with pytest.raises(TypeError, match="not integral"):
            _ = value**1.0
        with pytest.raises(ValueError, match="negative"):
            _ = value**-1
        with pytest.raises(AttributeError):
            _ = value == 1
