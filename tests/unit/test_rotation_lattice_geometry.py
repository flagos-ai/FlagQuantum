from tests._reference_dependencies import unavailable_reference

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

# Adapted from Qualtran v0.7.0 rotation_synthesis/lattice/geometry_test.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/rotation_synthesis/lattice/geometry_test.py

try:
    import numpy as np
except ModuleNotFoundError as _reference_import_error:
    unavailable_reference(_reference_import_error)
import pytest

try:
    from numpy.typing import NDArray
except ModuleNotFoundError as _reference_import_error:
    unavailable_reference(_reference_import_error)

import flagquantum.synth._rotation_math as mc
import flagquantum.synth._rotation_types as rst
import tests.unit.rotation_lattice_fixtures as tu
from flagquantum.synth import _lattice as lattice

_RNG = np.random.default_rng(0)


class TestEllipse:
    @pytest.mark.parametrize("ellipse", tuple(tu.make_ellipses(10)))
    def test_parametric_form(self, ellipse: lattice.Ellipse) -> None:
        p = ellipse.parametric_form(config=mc.NumpyConfig)
        l_value = 1 + np.sqrt(2)

        np.testing.assert_allclose(ellipse.D, p.to_ellipse(l_value).D)

    @pytest.mark.parametrize("ellipse", tuple(tu.make_ellipses(10)))
    def test_det(self, ellipse: lattice.Ellipse) -> None:
        np.testing.assert_allclose(ellipse._det, np.linalg.det(ellipse.D))

    @pytest.mark.parametrize("a", _RNG.random(5))
    @pytest.mark.parametrize("b", _RNG.random(5))
    @pytest.mark.parametrize("theta", _RNG.random(5) * np.pi)
    def test_bounding_tilt(self, a: rst.Real, b: rst.Real, theta: rst.Real) -> None:
        if a < b:
            # to make the choice of the 4 possiblities unique we make 0 < theta < pi and a >= b
            a, b = b, a
        e = lattice.Ellipse.from_axes(a, b, theta, np.zeros(2), mc.NumpyConfig)
        np.testing.assert_allclose(e.tilt(mc.NumpyConfig), np.pi - theta)

    @pytest.mark.parametrize("a", _RNG.random(5))
    @pytest.mark.parametrize("b", _RNG.random(5))
    @pytest.mark.parametrize("center", _RNG.random((5, 2)))
    def test_bounding_box_no_rotation(
        self, a: rst.Real, b: rst.Real, center: NDArray[np.float64]
    ) -> None:
        e = lattice.Ellipse(np.diagflat([1 / a, 1 / b]) ** 2, center)
        bbox = e.bounding_box(mc.NumpyConfig)
        x_max = a
        y_max = b
        assert bbox.x_bounds == lattice.Range(
            pytest.approx(center[0] - x_max), pytest.approx(center[0] + x_max)
        )
        assert bbox.y_bounds == lattice.Range(
            pytest.approx(center[1] - y_max), pytest.approx(center[1] + y_max)
        )

    @pytest.mark.parametrize("a", _RNG.random(5) * 10)
    @pytest.mark.parametrize("b", _RNG.random(5) * 10)
    @pytest.mark.parametrize("theta", (2 * _RNG.random(5) - 1) * 2 * np.pi)
    def test_bounding_box_with_rotation(
        self, a: rst.Real, b: rst.Real, theta: rst.Real
    ) -> None:
        e = lattice.Ellipse.from_axes(a, b, theta, np.zeros(2), mc.NumpyConfig)
        bbox = e.bounding_box(mc.NumpyConfig)
        c, s = np.cos(theta), np.sin(theta)
        # Analytic bounds
        x_max = np.sqrt(a**2 * c**2 + b**2 * s**2)
        y_max = np.sqrt(a**2 * s**2 + b**2 * c**2)
        assert bbox.x_bounds == lattice.Range(
            pytest.approx(-x_max), pytest.approx(x_max)
        )
        assert bbox.y_bounds == lattice.Range(
            pytest.approx(-y_max), pytest.approx(y_max)
        )


def test_locked_numeric_bound_types_and_plot() -> None:
    import mpmath

    try:
        from matplotlib.figure import Figure
    except ModuleNotFoundError as _reference_import_error:
        unavailable_reference(_reference_import_error)
    from qualtran.rotation_synthesis.lattice._geometry import Ellipse as SourceEllipse
    from qualtran.rotation_synthesis.lattice._geometry import Range as SourceRange

    inputs = [
        (0, np.int64(2)),
        (np.float32(0.25), 2),
        (mpmath.mpf("0.25"), 2),
        (0.25, 2),
    ]
    for start, end in inputs:
        bounds = lattice.Range.from_bounds(start, end)
        expected = SourceRange.from_bounds(start, end)
        assert (type(bounds.start), type(bounds.end)) == (
            type(expected.start),
            type(expected.end),
        )
        assert (bounds.start, bounds.end) == (expected.start, expected.end)

    matrix = [[2.0, -0.3], [-0.3, 1.0]]
    center = [0.5, -1.0]
    native = lattice.Ellipse(matrix, center)
    reference = SourceEllipse(matrix, center)
    assert native.as_formula(".2f") == reference.as_formula(".2f")
    axes = Figure().subplots()
    locked_axes = Figure().subplots()
    assert native.plot(axes, add_label=False, facecolor="red") is axes
    reference.plot(locked_axes, add_label=False, facecolor="red")
    native_patch, locked_patch = axes.patches[0], locked_axes.patches[0]
    np.testing.assert_array_equal(
        native_patch.get_extents().bounds, locked_patch.get_extents().bounds
    )
    assert native_patch.get_facecolor() == locked_patch.get_facecolor()
