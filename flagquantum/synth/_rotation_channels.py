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

# Adapted from Qualtran v0.7.0 rotation_synthesis/channels/_channel.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/rotation_synthesis/channels/_channel.py

"""Exact-ring models for single-qubit rotation synthesis channels."""

from __future__ import annotations

import abc
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from numpy.typing import NDArray

from . import _su2_ct, _zsqrt2
from ._rotation_math import MathConfig
from ._rotation_types import Real
from ._zw import ZW


class Channel(abc.ABC):
    @abc.abstractmethod
    def expected_num_ts(self, config: MathConfig) -> Real: ...

    @abc.abstractmethod
    def diamond_norm_distance_to_rz(self, theta: Real, config: MathConfig) -> Real:
        r"""Returns the diamond norm distance to $e^{i\theta Z}$."""


@dataclass(frozen=True)
class UnitaryChannel(Channel):
    r"""A Unitary operation.

    The unitary matrix of an $SU(2)$ matrix is defined with two complex numbers $u, v$ as

    $$
    \begin{bmatrix}
    u & -v^*\\
    v & u^*
    \end{bmatrix}
    $$
    for clifford+T unitarys the matrix can be written as

    $$
    \frac{1}{\sqrt{2(2+\sqrt{2})^n}}
    \begin{bmatrix}
    p & -q^*\\
    q & p^*
    \end{bmatrix}
    $$
    where $n$ is the number of T gates and $p, q \in \mathbb{Z}[e^{i\pi/4}]$.


    Attributes:
        p: The upper left elemenet of the clifford+T matrix.
        q: The lower left elemenet of the clifford+T matrix.
        n: The number of T gates.
        twirl: If True, the unitary is twirled with gates equiprobable chosen
            from the gates $\{I, Z, S, S^\dagger\}$.
    """

    p: ZW
    q: ZW
    n: int
    twirl: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.p, ZW) or not isinstance(self.q, ZW):
            raise TypeError(
                "Unitary channel coefficients must be eighth-root ring elements"
            )
        if not isinstance(self.n, int):
            raise TypeError("Unitary channel T count must be an int")

    def to_matrix(self) -> _su2_ct.SU2CliffordT:
        return _su2_ct.SU2CliffordT.from_pair(self.p, self.q, pick_phase=True)

    def expected_num_ts(self, config: MathConfig | None = None) -> Real:
        return self.n

    def rotation_angle(self, config: MathConfig) -> Real:
        """Returns the rotation angle applied with diag(self)."""
        return self.p.polar(config)[1]

    def failure_angle(self, config: MathConfig) -> Real:
        """Returns the rotation angle applied with antidiag(self)."""
        return self.q.polar(config)[1]

    def diamond_norm_distance_to_rz(self, theta: Real, config: MathConfig) -> Real:
        u = self.p.value(config.sqrt2)
        u = u / _zsqrt2.radius_at_n(_zsqrt2.LAMBDA_KLIUCHNIKOV, self.n, config)
        neg_rot = config.cos(theta) - config.sin(theta) * 1j
        real_squared = (u * neg_rot).real ** 2
        if self.twirl:
            return 2 * (1 - real_squared)
        return 2 * config.sqrt(1 - real_squared)

    @classmethod
    def from_sequence(cls, seq: Sequence[str], twirl: bool = False) -> UnitaryChannel:
        """Constructs a unitary channel from a sequence of gate names.

        Args:
            seq: A sequence of gate names in circuit order.
                Supported gate names are {Tx, Ty, Tz, H, S, X, Y, Z}.
            twirl: Whether the unitary is applied with a ZS twirling or not.

        Returns:
            A UnitaryChannel.
        """
        u = _su2_ct.SU2CliffordT.from_sequence(seq)
        n = sum(g.startswith("T") for g in seq)
        return UnitaryChannel(u.matrix[0, 0], u.matrix[1, 0], n, twirl)

    def diamond_norm_distance_to_unitary(
        self, unitary: NDArray[Any], config: MathConfig
    ) -> Real:
        r"""Returns the diamond norm distance between self and the given unitary.

        From Theorem B.1 of arxiv:2203.10064, the diamond norm distance between two untiaries
        $U, V$ is $|v_1 - v_0|$ where $v_i$ are the eigen values of $V^\dagger U$. Geometrically
        this is the diameter of the smallest disc in the complex plane that contains both
        eigenvalues.
        """
        # W = V^\dagger U
        w = self.to_matrix().adjoint().numpy(config) @ unitary
        # Compute the eigen values of W
        a = config.one
        b = -w[0, 0] - w[1, 1]
        c = w[0, 0] * w[1, 1] - w[0, 1] * w[1, 0]
        d = config.sqrt(b**2 - 4 * a * c)
        eigv0, eigv1 = [(-b - d) / (2 * a), (-b + d) / (2 * a)]
        # Compute the norm of the difference.
        diameter_vec = eigv1 - eigv0
        return config.sqrt(diameter_vec.real**2 + diameter_vec.imag**2)

    @classmethod
    def from_unitaries(
        cls, *unitaries: UnitaryChannel | _su2_ct.SU2CliffordT
    ) -> UnitaryChannel:
        if not unitaries:
            raise ValueError("at least one unitary should be provided")

        unitary = _su2_ct.ISqrt2
        for u in unitaries:
            if isinstance(u, UnitaryChannel):
                unitary = unitary @ u.to_matrix()
            else:
                unitary = unitary @ u
        unitary = unitary.rescale()
        return UnitaryChannel(
            unitary.matrix[0, 0], unitary.matrix[1, 0], unitary.num_t_gates()
        )


@dataclass(frozen=True)
class ProjectiveChannel(Channel):
    """A fallback (a.k.a RUS channel).

    This channel applies the following circuit where $V$ is the `rotation` channel and $C$ is the
    correction channel.

    ```
    q: ─────────@───V───@───────Y───C───
                │       │       ║   ║
    ancilla: ───X───────X───M───╫───╫───
                            ║   ║   ║
    m: ═════════════════════@═══^═══^═══
    ```

    Attributes:
        rotation: This first half of the channel (i.e. $V$).
        correction: The correction channel, this is applied only when the measurement result is one
    """

    rotation: UnitaryChannel
    correction: Channel

    def success_probability(self, config: MathConfig) -> Real:
        """Constructs a probablity of a zero measurement."""
        v = self.rotation.q.polar(config)[0] / _zsqrt2.radius_at_n(
            _zsqrt2.LAMBDA_KLIUCHNIKOV, self.rotation.n, config
        )
        return 1 - v * v

    def rotation_angle(self, config: MathConfig) -> Real:
        """Returns the rotation angle applied with when the measurement result is zero."""
        return self.rotation.rotation_angle(config)

    def failure_angle(self, config: MathConfig) -> Real:
        """Returns the rotation angle applied with when the measurement result is one."""
        return self.rotation.failure_angle(config)

    def expected_num_ts(self, config: MathConfig) -> Real:
        v = self.rotation.q.value(config.sqrt2)
        fail_prob = (v * v.conjugate()).real / _zsqrt2.radius2_at_n(
            _zsqrt2.LAMBDA_KLIUCHNIKOV, self.rotation.n, config
        )
        return self.rotation.expected_num_ts(
            config
        ) + fail_prob * self.correction.expected_num_ts(config)

    @staticmethod
    def diamond_distance_to_rz_on_measurement_success(
        p: ZW, theta: Real, config: MathConfig
    ) -> Real:
        """Returns the diamond norm distance when the measurement result is zero."""
        u = p.value(config.sqrt2)
        arg_u = config.arctan2(u.imag, u.real)
        return 2 * abs(config.sin(arg_u - theta))

    def diamond_norm_distance_to_rz(self, theta: Real, config: MathConfig) -> Real:
        p = self.success_probability(config)
        return p * ProjectiveChannel.diamond_distance_to_rz_on_measurement_success(
            self.rotation.p, theta, config
        ) + (1 - p) * self.correction.diamond_norm_distance_to_rz(
            theta - self.failure_angle(config), config
        )


@dataclass(frozen=True)
class ProbabilisticChannel(Channel):
    """A Channel that randomly applies one of two channels.

    Attributes:
        c1: The first channel.
        c2: The second channel.
        probability: The probablility of applying the first channel.
    """

    c1: Channel
    c2: Channel
    probability: Real = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "probability", max(min(self.probability, 1), 0))

    def expected_num_ts(self, config: MathConfig) -> Real:
        return self.probability * self.c1.expected_num_ts(config) + (
            1 - self.probability
        ) * self.c2.expected_num_ts(config)

    def diamond_norm_distance_to_rz(self, theta: Real, config: MathConfig) -> Real:
        if isinstance(self.c1, UnitaryChannel) and isinstance(self.c2, UnitaryChannel):
            return self.probability * self.c1.diamond_norm_distance_to_rz(
                theta, config
            ) + (1 - self.probability) * self.c2.diamond_norm_distance_to_rz(
                theta, config
            )
        if isinstance(self.c1, ProjectiveChannel) and isinstance(
            self.c2, ProjectiveChannel
        ):
            under = self.c1
            over = self.c2
            q1 = under.success_probability(config)
            q2 = over.success_probability(config)
            delta1 = under.rotation_angle(config) - theta
            delta2 = over.rotation_angle(config) - theta
            under_err = q1 * 2 * config.sin(delta1) ** 2 + (
                1 - q1
            ) * under.correction.diamond_norm_distance_to_rz(
                theta - under.failure_angle(config), config
            )
            over_err = q2 * 2 * config.sin(delta2) ** 2 + (
                1 - q2
            ) * over.correction.diamond_norm_distance_to_rz(
                theta - over.failure_angle(config), config
            )
            return self.probability * under_err + (1 - self.probability) * over_err
        raise NotImplementedError()

    @classmethod
    def from_unitary_channels(
        cls,
        u1: UnitaryChannel,
        u2: UnitaryChannel,
        target_theta: Real,
        config: MathConfig,
    ) -> ProbabilisticChannel:
        r1, phi1 = u1.p.polar(config)
        r2, phi2 = u2.p.polar(config)
        delta1 = phi1 - target_theta
        delta2 = phi2 - target_theta
        r1 /= _zsqrt2.radius_at_n(_zsqrt2.LAMBDA_KLIUCHNIKOV, u1.n, config)
        r2 /= _zsqrt2.radius_at_n(_zsqrt2.LAMBDA_KLIUCHNIKOV, u2.n, config)
        prob = (
            r2**2
            * config.sin(2 * delta2)
            / (r2**2 * config.sin(2 * delta2) - r1**2 * config.sin(2 * delta1))
        )
        return ProbabilisticChannel(u1, u2, prob)

    @classmethod
    def from_projective_channels(
        cls,
        under_channel: ProjectiveChannel,
        over_channel: ProjectiveChannel,
        target_theta: Real,
        config: MathConfig,
    ) -> ProbabilisticChannel:
        q1 = under_channel.success_probability(config)
        q2 = over_channel.success_probability(config)
        term1 = q1 * config.sin(
            2 * (under_channel.rotation_angle(config) - target_theta)
        )
        term2 = q2 * config.sin(
            2 * (over_channel.rotation_angle(config) - target_theta)
        )
        assert term1 <= 0
        assert term2 >= 0
        prob = term2 / (term2 - term1)
        return ProbabilisticChannel(under_channel, over_channel, prob)
