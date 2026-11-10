"""Fourier circuits against independent FFTs and exterior-power amplitudes."""

import itertools
from collections.abc import Callable

import pytest
import torch

from flagquantum.algorithms.fermion.decomposition import OrbitalDecomposition
from flagquantum.algorithms.fermion.fourier import fourier_decomposition
from flagquantum.algorithms.fermion.lowering import fermionic_fourier_transform
from flagquantum.circuit import Circuit
from flagquantum.simulation.unitary import get_unitary

pytestmark = pytest.mark.unit


def fft_matrix(n: int, inverse: bool) -> torch.Tensor:
    identity = torch.eye(n, dtype=torch.complex128)
    transform = torch.fft.ifft if inverse else torch.fft.fft
    return transform(identity, dim=0, norm="ortho")


@pytest.mark.parametrize("n", [1, 2, 4, 8, 16, 32])
@pytest.mark.parametrize("inverse", [False, True])
def test_plan_reconstructs_independent_fft(n: int, inverse: bool) -> None:
    plan = fourier_decomposition(n, inverse=inverse)
    reconstructed = torch.diag(plan.phases)
    for step in plan.rotations:
        assert 0 <= step.left < n - 1
        reconstructed[step.left : step.left + 2] = (
            step.matrix @ reconstructed[step.left : step.left + 2].clone()
        )
    torch.testing.assert_close(
        reconstructed, fft_matrix(n, inverse), atol=1e-12, rtol=1e-12
    )


@pytest.mark.parametrize("n", [1, 2, 4])
@pytest.mark.parametrize("inverse", [False, True])
def test_every_fock_sector_including_vacuum_and_determinant_phase(
    n: int, inverse: bool
) -> None:
    circuit = fermionic_fourier_transform(n, inverse=inverse)
    actual = get_unitary(circuit)
    fourier = fft_matrix(n, inverse)
    expected = torch.zeros_like(actual)
    for column in range(2**n):
        occupied = [i for i in range(n) if column & (1 << (n - i - 1))]
        for rows in itertools.combinations(range(n), len(occupied)):
            index = sum(1 << (n - i - 1) for i in rows)
            expected[index, column] = torch.linalg.det(fourier[list(rows)][:, occupied])
    # Elementwise comparison deliberately retains the global phase.
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    for instruction in circuit.to_ir().instructions:
        assert len(instruction.wires) == 2
        assert instruction.wires[1] == instruction.wires[0] + 1


@pytest.mark.parametrize("occupied", [(), (3,), (0, 3), (1, 4, 7), tuple(range(8))])
@pytest.mark.parametrize("inverse", [False, True])
def test_eight_modes_preserve_natural_layout_and_fermionic_signs(
    occupied: tuple[int, ...], inverse: bool
) -> None:
    circuit = Circuit(8, dtype=torch.complex128)
    for wire in occupied:
        circuit.x(wire)
    circuit.compose(fermionic_fourier_transform(8, inverse=inverse))
    actual = circuit.state().reshape(-1)
    expected = torch.zeros_like(actual)
    fourier = fft_matrix(8, inverse)
    for rows in itertools.combinations(range(8), len(occupied)):
        index = sum(1 << (7 - i) for i in rows)
        expected[index] = torch.linalg.det(fourier[list(rows)][:, list(occupied)])
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("n", [1, 2, 4, 8])
def test_transform_then_inverse_on_complex_superposition(n: int) -> None:
    circuit = Circuit(n, dtype=torch.complex128)
    for wire in range(n):
        circuit.ry(wire, theta=0.3 + wire / 10)
        circuit.rz(wire, theta=-0.2 + wire / 7)
    before = circuit.state().clone()
    circuit.compose(fermionic_fourier_transform(n))
    circuit.compose(fermionic_fourier_transform(n, inverse=True))
    torch.testing.assert_close(circuit.state(), before, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize(
    "builder", [fourier_decomposition, fermionic_fourier_transform]
)
@pytest.mark.parametrize("n", [0, -2, 3, 6])
def test_unsupported_width_is_rejected(
    builder: Callable[..., OrbitalDecomposition | Circuit], n: int
) -> None:
    with pytest.raises(ValueError, match="n_modes"):
        builder(n)


@pytest.mark.parametrize(
    "builder", [fourier_decomposition, fermionic_fourier_transform]
)
@pytest.mark.parametrize("n", [True, 4.0, "4", None])
def test_width_is_validated_before_coercion(
    builder: Callable[..., OrbitalDecomposition | Circuit], n: object
) -> None:
    with pytest.raises(TypeError):
        builder(n)


@pytest.mark.parametrize(
    "builder", [fourier_decomposition, fermionic_fourier_transform]
)
@pytest.mark.parametrize("inverse", [1, "inverse", None])
def test_inverse_requires_boolean(
    builder: Callable[..., OrbitalDecomposition | Circuit], inverse: object
) -> None:
    with pytest.raises(TypeError, match="inverse"):
        builder(4, inverse=inverse)
