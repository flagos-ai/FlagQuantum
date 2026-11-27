# Copyright 2023 Google LLC
# Copyright 2018 The Cirq Developers
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 for_testing/matrix_gate.py and
# Cirq testing/lin_alg_utils.py's seeded Haar sampler.

"""Unitary matrix fixtures using the existing matrix-backed Circuit opcode."""

from collections.abc import Sequence
from dataclasses import dataclass

from tests._reference_dependencies import unavailable_reference

try:
    import numpy as np
except ModuleNotFoundError as _reference_import_error:
    unavailable_reference(_reference_import_error)
import torch

try:
    from numpy.typing import NDArray
except ModuleNotFoundError as _reference_import_error:
    unavailable_reference(_reference_import_error)

from flagquantum import Circuit
from flagquantum.core.ir.program.signatures import Signature


@dataclass(frozen=True)
class MatrixRegisterFixture:
    bitsize: int
    matrix: Sequence[Sequence[complex]]
    atol: float = 1e-10

    def __post_init__(self) -> None:
        matrix = tuple(tuple(row) for row in self.matrix)
        object.__setattr__(self, "matrix", matrix)
        array = self.unitary()
        identity = np.eye(2**self.bitsize)
        np.testing.assert_allclose(array @ array.conj().T, identity, atol=self.atol)
        np.testing.assert_allclose(array.conj().T @ array, identity, atol=self.atol)

    @property
    def signature(self) -> Signature:
        return Signature.build(q=self.bitsize)

    @classmethod
    def random(
        cls, bitsize: int, *, random_state: int | np.random.RandomState | None = None
    ) -> "MatrixRegisterFixture":
        rng = (
            np.random
            if random_state is None
            else (
                np.random.RandomState(random_state)
                if isinstance(random_state, int)
                else random_state
            )
        )
        size = 1 << bitsize
        sample = rng.randn(size, size) + 1j * rng.randn(size, size)
        unitary, upper = np.linalg.qr(sample)
        diagonal = np.diag(upper)
        rows = tuple(tuple(row) for row in unitary * (diagonal / np.abs(diagonal)))
        return cls(bitsize, rows)

    def unitary(self) -> NDArray[np.complex128]:
        return np.asarray(self.matrix, dtype=np.complex128)

    def tensor(self) -> torch.Tensor:
        return torch.as_tensor(self.unitary()).reshape((2,) * (2 * self.bitsize))

    def adjoint(self) -> "MatrixRegisterFixture":
        rows = tuple(tuple(row) for row in self.unitary().conj().T)
        return type(self)(self.bitsize, rows, self.atol)

    def make_on(self, *wires: int) -> "BoundMatrixRegisterFixture":
        if len(wires) != self.bitsize:
            raise ValueError("Matrix register width must match the bound wires")
        return BoundMatrixRegisterFixture(self, wires)


@dataclass(frozen=True)
class BoundMatrixRegisterFixture:
    fixture: MatrixRegisterFixture
    wires: tuple[int, ...]

    def append_to(self, circuit: Circuit) -> Circuit:
        matrix = self.fixture.tensor()
        if not self.wires:
            # Embed a source scalar in one existing wire without changing its state.
            return circuit.any(
                0,
                matrix=matrix * torch.eye(2, dtype=matrix.dtype),
                name="matrix_fixture",
            )
        return circuit.any(*self.wires, matrix=matrix, name="matrix_fixture")
