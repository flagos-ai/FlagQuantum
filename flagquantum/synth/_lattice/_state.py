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

# Adapted from Qualtran v0.7.0 rotation_synthesis/lattice/py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/rotation_synthesis/lattice/py

"""Two-ellipse state and source-ordered grid actions."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace

from .. import _zw
from .._rotation_math import MathConfig
from .._rotation_types import Real
from ._geometry import Ellipse, EllipseParametricForm
from ._grid_operators import (
    ASqrt2,
    BSqrt2,
    GridOperator,
    KConjSqrt2,
    KSqrt2,
    RSqrt2,
    XSqrt2,
    ZSqrt2,
)


@dataclass(frozen=True, init=False)
class GridOperatorAction:
    """A grid operator and the sequence of operations that built it.

    Attributes:
        g: The grid operator.
        path: The sequence of operations that built the grid operator.
    """

    g: GridOperator
    path: tuple[str, ...]

    def __init__(self, g: GridOperator, path: Iterable[str] = ()) -> None:
        object.__setattr__(self, "g", g)
        object.__setattr__(self, "path", tuple(path))

    def shift(self, k: int) -> "GridOperatorAction":
        """Returns the Action corresponding to shift(k)-self-shift(k)."""
        return replace(
            self,
            g=self.g.shift(k),
            path=(f"Shift({k})",) + self.path + (f"Shift({-k})",),
        )

    def followed_by(self, other: "GridOperatorAction") -> "GridOperatorAction":
        """Returns the Action corresponding to self followed by other."""
        return GridOperatorAction(self.g @ other.g, self.path + other.path)


@dataclass(frozen=True)
class SelingerState:
    r"""A state is a region in space defined by two ellipses (A, B).

    A point $p$ belongs to the region iff $p \in A$ and $p^\bullet \in B$. Where
    $p^\bullet$ is the sqrt2-conjugate of $p$.

    Attributes:
        m1: The first ellipse.
        m2: The second ellipse.

    References:
        Appendix A of https://arxiv.org/abs/1403.2975
    """

    m1: Ellipse
    m2: Ellipse

    def skew(self, config: MathConfig) -> Real:
        """Returns the skew of the state."""
        return (
            self.m1.parametric_form(config).b ** 2
            + self.m2.parametric_form(config).b ** 2
        )

    def bias(self, config: MathConfig) -> Real:
        """Returns the bias of the state."""
        return self.m2.parametric_form(config).z - self.m1.parametric_form(config).z

    @staticmethod
    def from_parametric_forms(
        e1: EllipseParametricForm,
        e2: EllipseParametricForm,
        config: MathConfig,
    ) -> "SelingerState":
        """Constructs a state from the parametric form of the two regions."""
        l_value = 1 + config.sqrt(2)
        return SelingerState(e1.to_ellipse(l_value), e2.to_ellipse(l_value))

    def shift(self, k: int, config: MathConfig) -> "SelingerState":
        """Shift the state by k resulting in a state with the original bias+2k and the same skew"""
        e1 = self.m1.parametric_form(config)
        e2 = self.m2.parametric_form(config)
        e1 = replace(e1, z=e1.z - k)
        sgn = -1 if k % 2 == 1 else 1
        e2 = replace(e2, z=e2.z + k, b=e2.b * sgn)
        return SelingerState.from_parametric_forms(e1, e2, config)

    def get_grid_operator(self, config: MathConfig) -> GridOperatorAction:
        """Returns the grid operator that improves the skewness of the state.

        The method follows the rules in appendix A of https://arxiv.org/abs/1403.2975.

        Args:
            config: A MathConfig object.
        """
        k = config.floor((1 - self.bias(config)) / 2)
        if k:
            # Lemma A.9
            return self.shift(k, config).get_grid_operator(config).shift(k)
        assert abs(self.bias(config)) <= 1

        e1 = self.m1.parametric_form(config)
        e2 = self.m2.parametric_form(config)

        # See Appendix A.6 for the mapping.
        if e2.b < 0:
            action = GridOperatorAction(ZSqrt2)
            action = self.apply(action, config).get_grid_operator(config)
            return replace(action, g=ZSqrt2 @ action.g)

        if e1.z + e2.z < 0:
            action = GridOperatorAction(XSqrt2)
            action = self.apply(action, config).get_grid_operator(config)
            return replace(action, g=XSqrt2 @ action.g)

        l_value = 1 + config.sqrt2
        if e1.b >= 0:
            if -0.8 <= e1.z <= 0.8 and -0.8 <= e2.z <= 0.8:
                g = RSqrt2
                name = "R"
            elif e1.z <= 0.3 and e2.z >= 0.8:
                g = KSqrt2
                name = "K"
            elif e1.z >= 0.3 and e2.z >= 0.3:
                c = min(e1.z, e2.z)
                n = max(1, config.floor(l_value**c / 2))
                g = ASqrt2**n
                name = f"A^{n}"
            elif e1.z >= 0.8 and e2.z <= 0.3:
                g = KConjSqrt2
                name = "K*"
            else:
                raise ValueError("Shouldn't be here.")
        else:
            if -0.8 <= e1.z <= 0.8 and -0.8 <= e2.z <= 0.8:
                g = RSqrt2
                name = "R"
            elif e1.z >= -0.2 and e2.z >= -0.2:
                c = min(e1.z, e2.z)
                n = max(1, config.floor(l_value**c / config.sqrt2))
                g = BSqrt2**n
                name = f"B^{n}"
            else:
                raise ValueError("Shouldn't be here.")
        return GridOperatorAction(g, (name,))

    def apply(
        self,
        operator_or_action: GridOperator | GridOperatorAction,
        config: MathConfig,
    ) -> "SelingerState":
        """Applies the given gridoperator/action on the state.

        Args:
            operator_or_action: A GridOperator or GridOpeartionAction to apply.
            config: A MathConfig object.
        """
        m1, m2 = self.m1.D, self.m2.D

        if isinstance(operator_or_action, GridOperatorAction):
            operator = operator_or_action.g
        else:
            operator = operator_or_action
        g = operator.value(config.sqrt2)
        g_conj = operator.sqrt2_conj().value(config.sqrt2)
        m1 = g.T @ m1 @ g / 2
        m2 = g_conj.T @ m2 @ g_conj / 2
        return SelingerState(
            Ellipse(
                m1,
                operator.scaled_inverse().actual_value(config.sqrt2) @ self.m1.center,
            ),
            Ellipse(
                m2,
                operator.sqrt2_conj().scaled_inverse().actual_value(config.sqrt2)
                @ self.m2.center,
            ),
        )

    def contains(self, x: _zw.ZW, config: MathConfig) -> bool:
        """Returns whether the given point belongs in the region defined by the state.

        Args:
            x: the point to check.
            config: A MathConfig object.
        """
        u = x.value(config.sqrt2)
        v = x.sqrt2_conj().value(config.sqrt2)
        return self.m1.contains(u.real, u.imag, config) and self.m2.contains(
            v.real, v.imag, config
        )
