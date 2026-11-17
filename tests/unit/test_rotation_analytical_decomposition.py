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

# Adapted from Qualtran v0.7.0 rotation_synthesis/matrix/analytical_decomposition_test.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/rotation_synthesis/matrix/analytical_decomposition_test.py

from collections.abc import Iterator
from typing import cast

import mpmath
import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from scipy import stats

from flagquantum import Circuit
from flagquantum.synth import _analytical_decomposition as ad
from flagquantum.synth import _rotation_math as mc


def _random_angles(n: int, seed: int) -> Iterator[NDArray[np.float64]]:
    rng = np.random.default_rng(seed)
    for _ in range(n):
        yield rng.random(3) * 2 * np.pi


def _random_su2(n: int, seed: int) -> Iterator[NDArray[np.complex128]]:
    for u in stats.unitary_group(2).rvs(n, seed):
        yield cast(NDArray[np.complex128], u / np.linalg.det(u) ** 0.5)


@pytest.mark.parametrize(
    ["theta1", "phi", "theta2"], tuple(_random_angles(100, seed=0))
)
def test_decomposition_round_trip(theta1: float, phi: float, theta2: float) -> None:
    u = ad.unitary_from_zxz(theta1, phi, theta2, mc.NumpyConfig)
    angles = ad.su_unitary_to_zxz_angles(u, config=mc.NumpyConfig)
    v = ad.unitary_from_zxz(*angles, config=mc.NumpyConfig)
    np.testing.assert_allclose(actual=u, desired=v)


@pytest.mark.parametrize("u", tuple(_random_su2(100, seed=0)))
def test_decomposition_round_trip_start_with_unitary(u: NDArray[np.complex128]) -> None:
    angles = ad.su_unitary_to_zxz_angles(u, config=mc.NumpyConfig)
    v = ad.unitary_from_zxz(*angles, config=mc.NumpyConfig)
    np.testing.assert_allclose(actual=u, desired=v)


@pytest.mark.parametrize("angle", [0.0, np.pi, np.pi - 1e-7, 0.7, -0.7])
def test_locked_zxz_angles_and_native_operator(angle: float) -> None:
    from qualtran.rotation_synthesis import _math_config as source_math
    from qualtran.rotation_synthesis.matrix import _analytical_decomposition as source

    unitary = source.unitary_from_zxz(0.67, angle, -0.93, source_math.NumpyConfig)
    expected = source.su_unitary_to_zxz_angles(unitary, source_math.NumpyConfig)
    angles = ad.su_unitary_to_zxz_angles(unitary, mc.NumpyConfig)
    np.testing.assert_array_equal(angles, expected)
    reconstructed = source.unitary_from_zxz(*expected, source_math.NumpyConfig)
    np.testing.assert_array_equal(
        ad.unitary_from_zxz(*angles, mc.NumpyConfig), reconstructed
    )
    np.testing.assert_allclose(reconstructed, unitary, atol=1e-8)

    class BoundZXZ:
        def append_to(self, circuit: Circuit) -> Circuit:
            return (
                circuit.rz(0, float(angles[2]))
                .rx(0, float(angles[1]))
                .rz(0, float(angles[0]))
            )

    for bit in (0, 1):
        inputs = torch.zeros(2, dtype=torch.complex128)
        inputs[bit] = 1
        circuit = Circuit(1, inputs=inputs, dtype=torch.complex128, device="cpu")
        circuit.append(BoundZXZ())
        instructions = circuit.to_ir().instructions
        assert [(gate.name, gate.wires) for gate in instructions] == [
            ("rz", (0,)),
            ("rx", (0,)),
            ("rz", (0,)),
        ]
        assert [gate.params["theta"] for gate in instructions] == list(reversed(angles))
        np.testing.assert_allclose(
            circuit.state().detach().reshape(-1).numpy(),
            reconstructed[:, bit],
            atol=1e-12,
        )


@pytest.mark.parametrize("sign", [-1, 1])
def test_source_cosine_clipping(sign: int) -> None:
    from qualtran.rotation_synthesis import _math_config as source_math
    from qualtran.rotation_synthesis.matrix import _analytical_decomposition as source

    entry = np.nextafter(1.0, 2.0)
    unitary = (
        np.eye(2) * entry
        if sign == 1
        else np.array([[0, -1j * entry], [-1j * entry, 0]])
    )
    assert ad.su_unitary_to_zxz_angles(
        unitary, mc.MathConfig()
    ) == source.su_unitary_to_zxz_angles(unitary, source_math.MathConfig())


def test_mpmath_preserves_configured_precision() -> None:
    from qualtran.rotation_synthesis import _math_config as source_math
    from qualtran.rotation_synthesis.matrix import _analytical_decomposition as source

    with mpmath.workdps(80):
        native = mc.with_dps(80)
        locked = source_math.with_dps(80)
        values = tuple(
            mpmath.mpf(value) for value in ("0.67", "0.00000000034", "-0.93")
        )
        theta1, phi, theta2 = values
        u = ad.unitary_from_zxz(theta1, phi, theta2, native)
        np.testing.assert_array_equal(
            u, source.unitary_from_zxz(theta1, phi, theta2, locked)
        )
        angles = ad.su_unitary_to_zxz_angles(u, native)
        assert angles == source.su_unitary_to_zxz_angles(u, locked)
        rebuilt = ad.unitary_from_zxz(*angles, native)
        assert max(
            abs(a - b) for a, b in zip(rebuilt.flat, u.flat, strict=True)
        ) < mpmath.mpf("1e-69")


def test_matrix_package_exports_map_to_native_owners() -> None:
    from qualtran.rotation_synthesis import matrix as source

    from flagquantum.ecosystem._quirk_rotation import to_quirk
    from flagquantum.ecosystem.cirq._rotation_export import to_cirq
    from flagquantum.synth._clifford_t_repr import to_sequence
    from flagquantum.synth._rotation_generation import (
        generate_cliffords,
        generate_rotations,
        generate_rotations_iter,
    )
    from flagquantum.synth._su2_ct import SU2CliffordT

    native_exports = (
        ad.su_unitary_to_zxz_angles,
        ad.unitary_from_zxz,
        to_cirq,
        to_quirk,
        to_sequence,
        generate_cliffords,
        generate_rotations,
        generate_rotations_iter,
        SU2CliffordT,
    )
    declared = {
        name
        for name, value in vars(source).items()
        if not name.startswith("_") and callable(value)
    }
    assert declared == {value.__name__ for value in native_exports}
