"""Verify affine point arithmetic and the locked symbolic coordinates."""

from itertools import product

import pytest

from flagquantum.algorithms.primitives.elliptic_curve_point import ECPoint

pytestmark = pytest.mark.unit


def test_affine_zero_doubling_signed_multiples_and_curve_metadata():
    base = ECPoint(15, 13, 17)
    identity = ECPoint.inf(17, 0)
    assert (str(base), str(identity)) == ("(15, 13)", "(0, 0)")
    assert base + identity == identity + base == base
    assert base + (-base) == identity
    assert 0 * base == 18 * base == identity
    assert 1 * base == base
    assert 2 * base == base + base == ECPoint(2, 10, 17)
    assert 3 * base == ECPoint(8, 3, 17)
    assert 9 * base == ECPoint(3, 0, 17)
    assert 2 * (9 * base) == identity
    assert (-1) * base == -base == ECPoint(15, 4, 17)
    assert base * -21 == -(3 * base)
    large = (1 << 256) + 11
    assert large * base == (large % 18) * base
    with pytest.raises(ValueError, match="consistent mod and curve"):
        _ = base + ECPoint(2, 2, 7)
    with pytest.raises(ValueError, match="consistent mod and curve"):
        _ = base + ECPoint(2, 2, 17, 3)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("mod,a,b", ((17, 0, 7), (7, 3, 4)))
def test_every_pair_and_nonnegative_scalar_matches_locked_source(mod, a, b):
    pytest.importorskip("qualtran")
    from qualtran.bloqs.cryptography.ecc import ECPoint as SourcePoint

    coordinates = [(0, 0)] + [
        (x, y)
        for x, y in product(range(mod), repeat=2)
        if (y * y - x**3 - a * x - b) % mod == 0
    ]
    native = [ECPoint(x, y, mod, a) for x, y in coordinates]
    source = [SourcePoint(x, y, mod, a) for x, y in coordinates]
    for left, right in zip(native, source, strict=True):
        assert ((-left).x, (-left).y) == ((-right).x, (-right).y)
        for point, original in zip(native, source, strict=True):
            observed, expected = left + point, right + original
            assert (observed.x, observed.y) == (expected.x, expected.y)
        for scalar in range(25):
            observed, expected = left * scalar, right * scalar
            assert (observed.x, observed.y, observed.mod, observed.curve_a) == (
                expected.x,
                expected.y,
                expected.mod,
                expected.curve_a,
            )


@pytest.mark.ftqc_reference
def test_locked_negative_scalar_boundary_and_symbolic_multiplication():
    pytest.importorskip("qualtran")
    from qualtran.bloqs.cryptography.ecc import ECPoint as SourcePoint
    from sympy import symbols

    source = SourcePoint(15, 13, 17)
    assert source * -1 == source and -source != source
    x, y, p, a, k = symbols("x y p a k", integer=True)
    symbolic = ECPoint(x, y, p, a)
    original = SourcePoint(x, y, p, a)
    for scalar in (0, 1, 2, -1, k):
        native = scalar * symbolic
        expected = scalar * original
        assert (native.x, native.y, native.mod, native.curve_a) == (
            expected.x,
            expected.y,
            expected.mod,
            expected.curve_a,
        )
