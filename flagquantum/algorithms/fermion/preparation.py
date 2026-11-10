# Adapted in part from OpenFermion, Apache-2.0. See third_party/openfermion.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Static Slater determinants and canonical Gaussian vacua in JW order."""

from __future__ import annotations

import torch

from ...circuit import Circuit
from .lowering import append_orbital_rotation, append_two_mode
from .numerics import static_tensor


def prepare_slater(occupied_orbitals: torch.Tensor) -> Circuit:
    """Prepare a determinant from orthonormal columns of shape (modes, particles).

    Amplitudes are the corresponding occupied-row minors, including their
    complex phase. Empty and fully occupied determinants are supported.

    Examples:
        >>> import torch
        >>> from flagquantum.algorithms.fermion.preparation import prepare_slater
        >>> circuit = prepare_slater(torch.eye(3, dtype=torch.complex128)[:, :1])
        >>> int(circuit.state().abs().argmax())
        4
    """
    orbitals = static_tensor(occupied_orbitals, "occupied_orbitals", ndim=2)
    n, k = orbitals.shape
    if n < 1 or k > n:
        raise ValueError(
            "occupied orbitals require 0 <= particles <= modes and modes >= 1"
        )
    if not torch.allclose(
        orbitals.mH @ orbitals,
        torch.eye(k, dtype=orbitals.dtype),
        atol=1e-10,
        rtol=1e-10,
    ):
        raise ValueError("occupied orbital columns must be orthonormal")
    circuit = Circuit(n, dtype=torch.complex128)
    for qubit in range(k):
        circuit.x(qubit)
    if k:
        complete, _ = torch.linalg.qr(orbitals, mode="complete")
        # Retain the supplied columns exactly; QR chooses different column phases.
        complete[:, :k] = orbitals
        append_orbital_rotation(circuit, complete)
    return circuit


# Gaussian elimination below is adapted from OpenFermion givens_rotations.py,
# Copyright 2017 The OpenFermion Developers, Apache-2.0. See third_party/openfermion.
def _zero_left(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if float(a.abs()) < 1e-14:
        return torch.eye(2, dtype=torch.complex128)
    if float(b.abs()) < 1e-14:
        return torch.tensor([[0, -1], [1, 0]], dtype=torch.complex128)
    norm = torch.sqrt(a.abs().square() + b.abs().square())
    cosine, sine = b.abs() / norm, a.abs() / norm
    phase = a / a.abs() * (b / b.abs()).conj()
    return torch.stack(
        (torch.stack((cosine, -phase * sine)), torch.stack((sine, phase * cosine)))
    )


def prepare_gaussian(annihilation: torch.Tensor, creation: torch.Tensor) -> Circuit:
    """Prepare the vacuum of ``b_i = sum_j(A_ij a_j + B_ij a_j†)``.

    A and B must obey the canonical anticommutation constraints. The input is
    static, the output uses JW occupation order, and its overall phase is
    unspecified. Both even and odd Gaussian vacua are supported. This does not
    construct the full Gaussian basis transformation for excited inputs.
    """
    a = static_tensor(annihilation, "annihilation", ndim=2)
    b = static_tensor(creation, "creation", ndim=2)
    n = a.shape[0]
    if n < 1 or a.shape != (n, n) or b.shape != (n, n):
        raise ValueError("canonical blocks must be matching nonempty square matrices")
    if not torch.allclose(
        a @ a.mH + b @ b.mH, torch.eye(n, dtype=a.dtype), atol=1e-10, rtol=1e-10
    ) or not torch.allclose(
        a @ b.T + b @ a.T, torch.zeros_like(a), atol=1e-10, rtol=1e-10
    ):
        raise ValueError("blocks violate canonical anticommutation constraints")
    current = torch.cat((b, a), dim=1)
    for column in range(n - 1):
        for row in range(n - 1 - column):
            if float(current[row, column].abs()) > 1e-14:
                rotation = _zero_left(current[row, column], current[row + 1, column])
                current[row : row + 2] = rotation @ current[row : row + 2].clone()
    # None denotes the last-mode particle-hole transformation; its JW image is X.
    steps: list[tuple[int, torch.Tensor] | None] = []
    for layer in range(2 * n - 1):
        if layer % 2 == 0 and float(current[layer // 2, n - 1].abs()) > 1e-14:
            steps.append(None)
            current[:, [n - 1, 2 * n - 1]] = current[:, [2 * n - 1, n - 1]]
        end_row = layer if layer < n else n - 1
        end_column = n - 1 - layer if layer < n else layer - (n - 1)
        for offset, column in enumerate(range(end_column, n - 1, 2)):
            row = end_row - offset
            if float(current[row, column].abs()) < 1e-14:
                continue
            rotation = _zero_left(
                current[row, column].conj(), current[row, column + 1].conj()
            )
            for start, matrix in ((0, rotation), (n, rotation.conj())):
                # OpenFermion's column rotation conjugates its second column.
                action = torch.stack((matrix[:, 0], matrix[:, 1].conj()), dim=1).T
                current[:, start + column : start + column + 2] = (
                    current[:, start + column : start + column + 2].clone() @ action
                )
            # R_yxxy followed by a phase on the second qubit has this orbital image.
            cosine, sine = rotation[0, 0].real, rotation[1, 0].real
            phase = (
                rotation[1, 1] / cosine
                if float(cosine.abs()) > 1e-14
                else -rotation[0, 1] / sine
            )
            orbital = torch.stack(
                (
                    torch.stack((cosine, sine)),
                    torch.stack((-phase * sine, phase * cosine)),
                )
            )
            steps.append((column, orbital))
    # A unitary row mixture of annihilators has the same vacuum. Singular
    # creation blocks can leave a nondiagonal unitary on the right.
    right = current[:, n:]
    if float(current[:, :n].abs().max()) > 1e-9 or not torch.allclose(
        right @ right.mH, torch.eye(n, dtype=current.dtype), atol=1e-9, rtol=1e-9
    ):
        raise RuntimeError(
            "Gaussian elimination did not reach an annihilation-only frame"
        )
    circuit = Circuit(n, dtype=torch.complex128)
    for step in reversed(steps):
        if step is None:
            circuit.x(n - 1)
        else:
            append_two_mode(circuit, step[0], step[1])
    return circuit
