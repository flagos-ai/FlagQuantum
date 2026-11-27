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
# Adapted from Qualtran v0.7.0 rotation_synthesis/lattice/ipe_test.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/rotation_synthesis/lattice/ipe_test.py
import itertools
import math
from collections.abc import Iterator

from tests._reference_dependencies import unavailable_reference

try:
    import numpy as np
except ModuleNotFoundError as _reference_import_error:
    unavailable_reference(_reference_import_error)
import pytest

import flagquantum.synth._rotation_math as mc
import tests.unit.rotation_lattice_fixtures as tu
from flagquantum.synth import _lattice as lattice
from flagquantum.synth import _rings as rings


def test_enumerate1d_requiring_swapping() -> None:
    interval = lattice.Range(start=-5.535533905932738, end=4.121320343559642)
    comp_interval = lattice.Range(start=0.2928932188134524, end=1.1213203435596426)
    assert set(lattice.enumerate_1d(interval, comp_interval, mc.NumpyConfig)) == {
        rings.ZSqrt2(a=-2, b=-2),
        rings.ZSqrt2(a=-1, b=-1),
        rings.ZSqrt2(a=1, b=0),
        rings.ZSqrt2(a=2, b=1),
    }


def bf(
    interval: lattice.Range, comp_interval: lattice.Range, bounds: tuple[int, int]
) -> Iterator[rings.ZSqrt2]:
    left, right = bounds
    for p in itertools.product(range(left, right + 1), repeat=2):
        x = rings.ZSqrt2(*p)
        if interval.contains(float(x), mc.NumpyConfig) and comp_interval.contains(
            float(x.conjugate()), mc.NumpyConfig
        ):
            yield x


@pytest.mark.parametrize(
    "w", 1 / (1 + mc.NumpyConfig.sqrt2) * np.random.default_rng(0).random(10)
)
def test_enumerate1d_requiring_scaling_up(w: float) -> None:
    interval = lattice.Range(start=0, end=w)
    comp_interval = lattice.Range(start=0, end=10)
    assert set(bf(interval, comp_interval, (-10, 10))) == set(
        lattice.enumerate_1d(interval, comp_interval, mc.NumpyConfig)
    )


@pytest.mark.parametrize("n", range(8))
def test_enumerate1d_against_brute_force(n: int) -> None:
    r0 = np.sqrt(2 * (2 + np.sqrt(2)) ** n)
    r1 = np.sqrt(2 / (2 + np.sqrt(2)) ** n)
    interval, comp_interval = (
        lattice.Range.from_bounds(-r0, r0),
        lattice.Range.from_bounds(-r1, r1),
    )
    bf_sol = set(bf(interval, comp_interval, (-int(r0) - 3, int(r0) + 3)))
    enum_sol = set(lattice.enumerate_1d(interval, comp_interval, mc.NumpyConfig))
    assert bf_sol == enum_sol


def bf_rect(
    r1: lattice.Rectangle, r2: lattice.Rectangle, b: float
) -> Iterator[rings.ZW]:
    w = np.exp(1j * np.pi / 4)
    coefficients = np.array([w**i for i in range(4)])
    conjugate_coefficients = coefficients * np.array([1, -1, 1, -1])
    for m in itertools.product(range(math.floor(-b), math.ceil(b) + 1), repeat=4):
        v = np.dot(m, coefficients)
        v_conj = np.dot(m, conjugate_coefficients)
        if r1.contains(v.real, v.imag, mc.NumpyConfig) and r2.contains(
            v_conj.real, v_conj.imag, mc.NumpyConfig
        ):
            yield rings.ZW(m)


@pytest.mark.parametrize("n", range(4))
def test_enumerate_upright_against_bf(n: int) -> None:
    r0 = np.sqrt(2 * (2 + np.sqrt(2)) ** n)
    r1 = np.sqrt(2 * (2 - np.sqrt(2)) ** n)
    interval, comp_interval = (
        lattice.Range.from_bounds(-r0, r0),
        lattice.Range.from_bounds(-r1, r1),
    )
    got = set(
        lattice.enumerate_upright(
            lattice.Rectangle(interval, interval),
            lattice.Rectangle(comp_interval, comp_interval),
            mc.NumpyConfig,
        )
    )
    want = set(
        bf_rect(
            lattice.Rectangle(interval, interval),
            lattice.Rectangle(comp_interval, comp_interval),
            r0,
        )
    )
    assert got == want


@pytest.mark.parametrize("state", tuple(tu.make_states(5)))
def test_get_overall_action(state: lattice.SelingerState) -> None:
    action = lattice.get_overall_action(state, mc.NumpyConfig)
    assert state.apply(action, mc.NumpyConfig).skew(mc.NumpyConfig) <= 15


@pytest.mark.parametrize("state", tuple(tu.make_states(5)))
def test_get_points_from_state(state: lattice.SelingerState) -> None:
    for p in lattice.get_points_from_state(state, mc.NumpyConfig):
        assert state.contains(p, mc.NumpyConfig)


@pytest.mark.parametrize("float_backend", [True, False])
def test_lockstep_order_and_state_transforms(float_backend: bool) -> None:
    import mpmath
    from qualtran.rotation_synthesis import _math_config as source_math
    from qualtran.rotation_synthesis.lattice import _test_utils as source_fixtures
    from qualtran.rotation_synthesis.lattice._geometry import Range as SourceRange
    from qualtran.rotation_synthesis.lattice._geometry import (
        Rectangle as SourceRectangle,
    )
    from qualtran.rotation_synthesis.lattice._ipe import (
        enumerate_1d as source_enumerate_1d,
    )
    from qualtran.rotation_synthesis.lattice._ipe import (
        enumerate_upright as source_enumerate_upright,
    )
    from qualtran.rotation_synthesis.lattice._ipe import (
        get_overall_action as source_get_overall_action,
    )
    from qualtran.rotation_synthesis.lattice._ipe import (
        get_points_from_state as source_get_points_from_state,
    )

    with mpmath.workdps(60):
        config = mc.NumpyConfig if float_backend else mc.with_dps(60)
        locked_config = (
            source_math.NumpyConfig if float_backend else source_math.with_dps(60)
        )
        for bounds, complement in [
            ((-4, 3), (-1, 2)),
            ((0.1, 0.3), (-10, 10)),
            ((-1.25, 1.25), (-1.75, 1.75)),
        ]:
            interval = lattice.Range.from_bounds(*bounds)
            comp_interval = lattice.Range.from_bounds(*complement)
            locked_interval = SourceRange.from_bounds(*bounds)
            locked_comp_interval = SourceRange.from_bounds(*complement)
            assert [
                (p.a, p.b)
                for p in lattice.enumerate_1d(interval, comp_interval, config)
            ] == [
                (p.a, p.b)
                for p in source_enumerate_1d(
                    locked_interval, locked_comp_interval, locked_config
                )
            ]
            points = lattice.enumerate_upright(
                lattice.Rectangle.make_square(interval),
                lattice.Rectangle.make_square(comp_interval),
                config,
            )
            reference = source_enumerate_upright(
                SourceRectangle.make_square(locked_interval),
                SourceRectangle.make_square(locked_comp_interval),
                locked_config,
            )
            assert [point.coords for point in points] == [
                point.coords for point in reference
            ]

        for state, locked_state in zip(
            tu.make_states(20), source_fixtures.make_states(20), strict=True
        ):
            action = lattice.get_overall_action(state, config)
            locked_action = source_get_overall_action(locked_state, locked_config)
            assert action.path == locked_action.path
            assert [(v.a, v.b) for v in action.g.matrix.flat] == [
                (v.a, v.b) for v in locked_action.g.matrix.flat
            ]
            reduced, locked_reduced = (
                state.apply(action, config),
                locked_state.apply(locked_action, locked_config),
            )
            np.testing.assert_array_equal(reduced.m1.D, locked_reduced.m1.D)
            np.testing.assert_array_equal(reduced.m2.D, locked_reduced.m2.D)
            assert [
                p.coords for p in lattice.get_points_from_state(reduced, config)
            ] == [
                p.coords
                for p in source_get_points_from_state(locked_reduced, locked_config)
            ]


def test_lattice_namespace_and_zero_width_boundary() -> None:
    from qualtran.rotation_synthesis import lattice as source
    from qualtran.rotation_synthesis.lattice._geometry import Range as SourceRange
    from qualtran.rotation_synthesis.lattice._ipe import (
        enumerate_1d as source_enumerate_1d,
    )

    assert set(lattice.__all__) == {
        name
        for name, value in vars(source).items()
        if not name.startswith("_") and callable(value)
    }
    with pytest.raises(ValueError):
        list(
            lattice.enumerate_1d(
                lattice.Range(1, 1), lattice.Range(0, 1), mc.MathConfig()
            )
        )
    with pytest.raises(ValueError):
        from qualtran.rotation_synthesis import _math_config as source_math

        list(
            source_enumerate_1d(
                SourceRange(1, 1), SourceRange(0, 1), source_math.MathConfig()
            )
        )
