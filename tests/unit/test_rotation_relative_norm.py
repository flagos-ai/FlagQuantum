#  Copyright 2025 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

# Adapted from Qualtran v0.7.0 rotation_synthesis/relative_norm/clifford_t_test.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/rotation_synthesis/relative_norm/clifford_t_test.py

import itertools
import random

import pytest

from flagquantum.synth import _relative_norm
from flagquantum.synth._relative_norm import CliffordTRelativeNormSolver
from flagquantum.synth._zsqrt2 import ZSqrt2
from flagquantum.synth._zw import ZW, Omega


@pytest.mark.parametrize("a", range(-10, 10))
@pytest.mark.parametrize("b", range(-10, 10))
def test_simple_cases(a: int, b: int) -> None:
    w = a + Omega * b
    target = (w * w.conjugate()).to_zsqrt2()[0]
    solver = CliffordTRelativeNormSolver()
    got = solver.solve(target)
    assert got is not None
    assert (got * got.conjugate()).to_zsqrt2()[0] == target, f"{got=} {w=} {target}"


@pytest.mark.parametrize(
    "m",
    random.Random(0).sample(tuple(itertools.product(range(-5, 5), repeat=4)), k=200),
)
def test_seeded_four_coordinate_targets(m: tuple[int, int, int, int]) -> None:
    w = ZW(m)
    target = (w * w.conjugate()).to_zsqrt2()[0]
    solver = CliffordTRelativeNormSolver()
    got = solver.solve(target)
    assert got is not None
    assert (got * got.conjugate()).to_zsqrt2()[0] == target, (
        f"{got=} {w=} {target} {m=}"
    )


def test_exact_locked_witnesses_and_inert_rejection() -> None:
    from qualtran.rotation_synthesis.relative_norm import (
        CliffordTRelativeNormSolver as Source,
    )
    from qualtran.rotation_synthesis.rings import ZSqrt2 as SourceRing

    native = CliffordTRelativeNormSolver()
    source = Source()
    for a, b in itertools.product(range(12), range(-5, 6)):
        if a < abs(b) * 2:
            continue
        target = ZSqrt2(a, b)
        actual = native.solve(target)
        expected = source.solve(SourceRing(a, b))
        assert (None if actual is None else actual.coords) == (
            None if expected is None else expected.coords
        )
        if actual is not None:
            assert (actual * actual.conj()).to_zsqrt2() == (target, ZSqrt2(0), False)
    assert native.solve(ZSqrt2(7)) is None


@pytest.mark.parametrize("power", range(-8, 9))
def test_locked_unit_root_selection(power: int) -> None:
    from qualtran.rotation_synthesis.relative_norm import (
        CliffordTRelativeNormSolver as Source,
    )
    from qualtran.rotation_synthesis.rings import ZSqrt2 as SourceRing

    unit = ZSqrt2(1, 1) ** power if power >= 0 else ZSqrt2(-1, 1) ** -power
    actual = CliffordTRelativeNormSolver().compute_w(unit)
    expected = Source().compute_w(SourceRing(unit.a, unit.b))
    assert (None if actual is None else actual.coords) == (
        None if expected is None else expected.coords
    )
    if power % 2:
        assert actual is None
    else:
        assert actual is not None
        assert (actual * actual.conj()).to_zsqrt2()[0] == unit


def test_generators_preserve_order_and_source_boundary() -> None:
    from qualtran.rotation_synthesis import relative_norm as source
    from qualtran.rotation_synthesis.rings import ZSqrt2 as SourceRing

    assert set(_relative_norm.__all__) == {
        name
        for name, value in vars(source).items()
        if not name.startswith("_") and callable(value)
    }
    generators = [ZSqrt2(5), ZSqrt2(3)]
    unit = ZSqrt2(1, 1)
    target = unit * 5**2 * 3**4
    assert _relative_norm.factor_into_generators(target, generators) == (unit, [2, 4])
    assert _relative_norm.factor_into_generators(target, generators[::-1]) == (
        unit,
        [4, 2],
    )
    with pytest.raises(AssertionError):
        _relative_norm.factor_into_generators(target, [unit])

    # Negative one lies outside the norm image. Preserve the locked result as
    # a diagnostic, then expose the mismatch in the claimed witness.
    actual = CliffordTRelativeNormSolver().solve(ZSqrt2(-1))
    expected = source.CliffordTRelativeNormSolver().solve(SourceRing(-1))
    assert (
        actual is not None and expected is not None and actual.coords == expected.coords
    )
    assert (actual * actual.conj()).to_zsqrt2()[0] == ZSqrt2(1)
