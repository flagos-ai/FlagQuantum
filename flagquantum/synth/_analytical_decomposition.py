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

# Adapted from Qualtran v0.7.0 rotation_synthesis/matrix/_analytical_decomposition.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/rotation_synthesis/matrix/_analytical_decomposition.py

"""Precision-configured ZXZ decomposition of SU(2) matrices."""

from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from ._rotation_math import MathConfig
from ._rotation_types import Real


def su_unitary_to_zxz_angles(
    u: NDArray[Any], config: MathConfig
) -> tuple[Real, Real, Real]:
    """Return mathematical-order ZXZ angles for a supplied SU(2) matrix."""
    cos_phi: Real = (u[1, 1] * u[0, 0] + u[1, 0] * u[0, 1]).real
    # clip to the range [-1, 1] to ensure numerical stability.
    cos_phi = min(1, max(-1, cos_phi))
    phi = config.arccos(cos_phi)

    sum_half_theta = config.arctan2(u[1, 1].imag, u[1, 1].real)
    diff_half_theta = config.arctan2(u[1, 0].imag, u[1, 0].real) - 1.5 * config.pi

    theta1 = sum_half_theta + diff_half_theta
    theta2 = sum_half_theta - diff_half_theta

    return theta1, phi, theta2


def rx(phi: Real, config: MathConfig) -> NDArray[Any]:
    c = config.cos(phi / 2)
    s = config.sin(phi / 2)
    return np.array([[c, -1j * s], [-1j * s, c]])


def rz(theta: Real, config: MathConfig) -> NDArray[Any]:
    v = config.cos(theta / 2) + 1j * config.sin(theta / 2)
    return np.diag([v.conjugate(), v])


def unitary_from_zxz(
    theta1: Real, phi: Real, theta2: Real, config: MathConfig
) -> NDArray[Any]:
    return cast(NDArray[Any], rz(theta1, config) @ rx(phi, config) @ rz(theta2, config))
