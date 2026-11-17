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

# Adapted from Qualtran v0.7.0 rotation_synthesis/lattice/_geometry.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/rotation_synthesis/lattice/_geometry.py

"""Precision-aware geometry for single-qubit lattice synthesis."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .._rotation_math import MathConfig, NumpyConfig
from .._rotation_types import Real, is_int

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def _mpf_type() -> type[Any] | None:
    try:
        from mpmath import mpf
    except ModuleNotFoundError as error:
        if error.name != "mpmath":
            raise
        return None
    return cast(type[Any], mpf)


@dataclass(frozen=True)
class Range:
    r"""The closed interval $[s, e]$."""

    start: Real
    end: Real

    @staticmethod
    def from_bounds(start: Real, end: Real) -> "Range":
        """Creates a Range object from endpoints while paying attention to data type."""
        sides = (start, end)
        mpf = _mpf_type()
        if mpf is not None and any(isinstance(x, mpf) for x in sides):
            return Range(mpf(start), mpf(end))
        if all(is_int(x) for x in sides):
            return Range(int(start), int(end))
        if any(isinstance(x, np.number) for x in sides):
            return Range(np.longdouble(start), np.longdouble(end))
        return Range(float(start), float(end))

    def contains(self, x: Real, config: MathConfig) -> bool:
        return bool(
            self.start <= x <= self.end
            or config.isclose(self.start, x)
            or config.isclose(self.end, x)
        )

    def width(self) -> Real:
        return self.end - self.start

    def shift(self, x: Real) -> "Range":
        r"""horizontal shift by $x$."""
        return Range.from_bounds(self.start + x, self.end + x)


@dataclass(frozen=True)
class Rectangle:
    """A rectangle with the given sides."""

    x_bounds: Range
    y_bounds: Range

    def area(self) -> Real:
        return self.x_bounds.width() * self.y_bounds.width()

    def contains(self, x: Real, y: Real, config: MathConfig) -> bool:
        r"""Returns whether the rectangle contains the given point or not.

        Args:
            x: The $x$ coordinate.
            y: The $y$ coordinate.
            config: The MathConfig object which provides an isclose method.
        """
        return self.x_bounds.contains(x, config) and self.y_bounds.contains(y, config)

    def shift(self, dx: Real, dy: Real) -> "Rectangle":
        """planar shift by (dx, dy)."""
        return Rectangle(self.x_bounds.shift(dx), self.y_bounds.shift(dy))

    @staticmethod
    def make_square(bounds: Range) -> "Rectangle":
        return Rectangle(bounds, bounds)


@dataclass(frozen=True)
class EllipseParametricForm:
    r"""A representation of an ellipse centered at the origin in terms of 3 parameters.

    A representation of an ellipse's $D$ matrix as
    $$
    D = \begin{bmatrix}
    e \lambda^{-z} & b \\
    b & e \lambda^z
    \end{bmatrix}
    $$
    Where $\lambda = 1 + \sqrt{2}$. This a specific representation used in
    [Optimal ancilla-free Clifford+T approximation of z-rotations](https://arxiv.org/abs/1403.2975).
    """

    e: Real
    z: Real
    b: Real

    def to_ellipse(self, l_value: Real) -> "Ellipse":
        return Ellipse(
            [
                [self.e * l_value ** (-self.z), self.b],
                [self.b, self.e * l_value**self.z],
            ]
        )


@dataclass(frozen=True, init=False)
class Ellipse:
    r"""An ellipse represented in matrix form.

    Any ellipse can be represented as
    $$
    (x - c)^T D (x - c) \leq 1
    $$
    Where $c$ is its center and $D$ is a positive semidefinite symmetric matrix.

    Attributes:
        D: A positive semidefinite symmetric matrix.
        center: The center of ellipse.
    """

    D: NDArray[Any]
    center: NDArray[Any]

    def __init__(self, D: ArrayLike, center: ArrayLike = (0, 0)) -> None:  # noqa: N803
        object.__setattr__(self, "D", np.array(D))
        object.__setattr__(self, "center", np.array(center))
        diff = self.D[0, 1] - self.D[1, 0]
        mpf = _mpf_type()
        if mpf is not None and isinstance(diff, mpf):
            from mpmath import power

            assert abs(diff) <= power(10, -9)
        else:
            assert np.isclose(diff, 0), f"{diff=} {self.D=}"

        assert self._det >= 0, f"det has to be positive: {self} det={self._det}"
        assert self._trace >= 0, (
            f"trace has to be positive:  {self} trace={self._trace}"
        )

    @functools.cached_property
    def _det(self) -> Real:
        a, b, c, d = self.D.reshape(-1)
        return cast(Real, a * d - b * c)

    @functools.cached_property
    def _trace(self) -> Real:
        a, _, _, d = self.D.reshape(-1)
        return cast(Real, a + d)

    def bounding_box(self, config: MathConfig) -> Rectangle:
        """Returns the smallest axis aligned box containing the ellipse."""
        a, _, _, d = self.D.reshape(-1)

        max_x = config.sqrt(d / self._det)
        max_y = config.sqrt(a / self._det)

        return Rectangle(
            Range.from_bounds(-max_x, max_x), Range.from_bounds(-max_y, max_y)
        ).shift(*self.center)

    def area(self, config: MathConfig) -> Real:
        return config.pi / config.sqrt(self._det)

    def rectangleness(self, config: MathConfig) -> Real:
        r"""Returns a number $\in [0, 1]$ representing how close the ellipse is to an axis aligned rectangle."""
        a = self.D[0, 0]
        d = self.D[1, 1]
        return config.pi / 4 * config.sqrt(self._det / (a * d))

    def parametric_form(self, config: MathConfig) -> EllipseParametricForm:
        """Returns the parametric form of the ellipse's $D$ matrix."""
        a = self.D[0, 0]
        d = self.D[1, 1]
        e = config.sqrt(a * d)
        z = config.log(d / e) / config.log(1 + config.sqrt(2))
        return EllipseParametricForm(e, z, self.D[0, 1])

    def normalize(self, config: MathConfig) -> "Ellipse":
        r"""Scales the ellipse to have area = $\pi$."""
        return Ellipse(self.D / config.sqrt(self._det), self.center)

    def rotate(self, theta: Real, config: MathConfig) -> "Ellipse":
        """Rotates the ellipse around its center by the given angle."""
        c, s = config.cos(theta), config.sin(theta)
        rotation = np.array([[c, -s], [s, c]])
        return Ellipse(rotation @ self.D @ rotation.T, self.center)

    def tilt(self, config: MathConfig) -> Real:
        """Returns the tilt of the ellipse."""
        # tan(2\theta) = 2b/(d-a)
        a, b, _, d = self.D.reshape(-1)
        y, x = 2 * b, d - a
        phi = config.arctan2(y, x) % (2 * np.pi)
        return phi / 2

    def plot(
        self, ax: Axes | None = None, add_label: bool = True, **patch_args: Any
    ) -> Axes:
        import matplotlib.pyplot as plt
        from matplotlib import patches

        fig = None
        if ax is None:
            fig, ax = plt.subplots(1)

        theta = float(self.tilt(NumpyConfig))
        e = self.rotate(theta, NumpyConfig)
        w = 2 / np.sqrt(float(e.D[0, 0]))
        h = 2 / np.sqrt(float(e.D[1, 1]))
        c = self.center.astype(float).tolist()
        ax.add_patch(
            patches.Ellipse(
                xy=c,
                width=w,
                height=h,
                angle=-theta * 180 / np.pi,
                label=f"{theta=:g} {w=:g} {h=:g} {c=}" if add_label else None,
                **patch_args,
            )
        )
        if add_label:
            ax.legend()
        if fig is not None:
            ax.relim()
            ax.autoscale_view()
            fig.show()
        return ax

    def contains(self, x: Real, y: Real, config: MathConfig) -> bool:
        p = np.array([x, y]).reshape((-1, 1)) - self.center.reshape((-1, 1))
        lhs = (p.T @ self.D @ p).item()
        return bool(lhs <= 1 or config.isclose(lhs, 1))

    @staticmethod
    def from_axes(
        x_axis: Real,
        y_axis: Real,
        theta: Real,
        center: ArrayLike,
        config: MathConfig,
    ) -> "Ellipse":
        r"""Constructs an ellipse from its algebraic form.

        An axis aligned ellipse is represented by
        $$
        \frac{(x - c_x)^2}{a^2}^2 + \frac{(y - c_y)^2}{b^2} = 1
        $$
        where $c$ is it center; $a$ and $b$ are its maximum values in the $x$ and $y$ directions.

        This method constructs an ellipse from this representation and rotates it by the given
        angle about its center.

        Args:
            x_axis: the maximum value in the x direction.
            y_axis: the maximum value in the y direction.
            theta: the angle to rotat
            center: the center of the ellipse.
            config: The MathConfig to use.
        """
        unrotated_matrix = np.array(
            [[1 / x_axis**2, config.zero], [config.zero, 1 / y_axis**2]]
        )
        c, s = config.cos(theta), config.sin(theta)
        rotation_matrix = np.array([[c, -s], [s, c]])
        return Ellipse(
            rotation_matrix @ unrotated_matrix @ rotation_matrix.T,
            np.asarray(center) + config.zero,
        )

    def as_formula(self, fmt: str = "") -> str:
        """Retruns the algebraic form of the ellipse's equation as a string."""
        a = format(self.D[0, 0], fmt)
        b = format(2 * self.D[0, 1], fmt)
        c = format(self.D[1, 1], fmt)
        x0, y0 = map(lambda x: format(x, fmt), self.center)
        return f"{a}*(x-{x0})^2 + {b}*(x-{x0})*(y-{y0}) + {c}*(y-{y0})^2 = 1"
