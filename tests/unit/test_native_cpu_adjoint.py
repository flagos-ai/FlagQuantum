"""Correctness and rollback contracts for the native CPU adjoint operator."""

from __future__ import annotations

import pytest
import torch

from flagquantum.simulation.native_cpu import (
    fused_rotation_adjoint_,
    fused_rzz_segment_adjoint_,
    native_cpu_adjoint_available,
)

pytestmark = pytest.mark.unit


def _rotation_matrix(
    name: str, theta: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    cosine = torch.cos(theta / 2)
    sine = torch.sin(theta / 2)
    if name == "rx":
        matrix = torch.stack(
            (
                torch.stack((cosine + 0j, -1j * sine)),
                torch.stack((-1j * sine, cosine + 0j)),
            )
        )
    elif name == "ry":
        matrix = torch.stack(
            (torch.stack((cosine, -sine)), torch.stack((sine, cosine)))
        )
    elif name == "rz":
        matrix = torch.diag(
            torch.stack((torch.exp(-0.5j * theta), torch.exp(0.5j * theta)))
        )
    else:
        negative = torch.exp(-0.5j * theta)
        positive = torch.exp(0.5j * theta)
        matrix = torch.diag(torch.stack((negative, positive, positive, negative)))
    return matrix.to(dtype=dtype).contiguous()


def _reference(
    ket: torch.Tensor,
    adjoint: torch.Tensor,
    matrix: torch.Tensor,
    *,
    name: str,
    wire: int,
    n_wires: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    stride = 1 << (n_wires - wire - 1)
    ket_pairs = ket.reshape(ket.shape[0], -1, 2, stride)
    adjoint_pairs = adjoint.reshape(adjoint.shape[0], -1, 2, stride)
    ket_zero, ket_one = ket_pairs[:, :, 0], ket_pairs[:, :, 1]
    adjoint_zero, adjoint_one = adjoint_pairs[:, :, 0], adjoint_pairs[:, :, 1]

    def cross(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return left.real * right.imag - left.imag * right.real

    if name == "rx":
        gradient = 0.5 * (
            cross(adjoint_zero, ket_one).sum() + cross(adjoint_one, ket_zero).sum()
        )
    elif name == "ry":
        gradient = 0.5 * (
            (adjoint_one.conj() * ket_zero).real.sum()
            - (adjoint_zero.conj() * ket_one).real.sum()
        )
    else:
        gradient = 0.5 * (
            cross(adjoint_zero, ket_zero).sum() - cross(adjoint_one, ket_one).sum()
        )

    inverse = matrix.mH
    previous_ket = torch.einsum("ij,abjs->abis", inverse, ket_pairs).reshape_as(ket)
    previous_adjoint = torch.einsum("ij,abjs->abis", inverse, adjoint_pairs).reshape_as(
        adjoint
    )
    return gradient, previous_ket, previous_adjoint


@pytest.mark.parametrize("dtype", (torch.complex64, torch.complex128))
@pytest.mark.parametrize("name", ("rx", "ry", "rz"))
@pytest.mark.parametrize("wire", (0, 2, 4))
def test_native_rotation_adjoint_matches_pytorch_reference(
    dtype: torch.dtype, name: str, wire: int
) -> None:
    if not native_cpu_adjoint_available():
        pytest.skip("native CPU extension is not built in this source checkout")
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    generator = torch.Generator().manual_seed(1701 + wire)
    ket = (
        torch.randn((2, 32), generator=generator)
        + 1j * torch.randn((2, 32), generator=generator)
    ).to(dtype)
    adjoint = (
        torch.randn((2, 32), generator=generator)
        + 1j * torch.randn((2, 32), generator=generator)
    ).to(dtype)
    matrix = _rotation_matrix(name, torch.tensor(0.31, dtype=real_dtype), dtype)
    expected = _reference(
        ket.clone(),
        adjoint.clone(),
        matrix,
        name=name,
        wire=wire,
        n_wires=5,
    )

    gradient = fused_rotation_adjoint_(
        ket, adjoint, matrix, name=name, wire=wire, n_wires=5
    )

    assert gradient is not None
    tolerance = 2e-5 if dtype == torch.complex64 else 1e-12
    torch.testing.assert_close(gradient, expected[0], atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(ket, expected[1], atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(adjoint, expected[2], atol=tolerance, rtol=tolerance)


def test_native_rotation_adjoint_has_explicit_environment_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FQ_NATIVE_CPU_ADJOINT", "0")
    ket = torch.ones((1, 2), dtype=torch.complex128)
    adjoint = torch.ones_like(ket)
    matrix = torch.eye(2, dtype=torch.complex128)

    result = fused_rotation_adjoint_(ket, adjoint, matrix, name="ry", wire=0, n_wires=1)

    assert result is None
    torch.testing.assert_close(ket, torch.ones_like(ket))
    torch.testing.assert_close(adjoint, torch.ones_like(adjoint))


@pytest.mark.parametrize("dtype", (torch.complex64, torch.complex128))
@pytest.mark.parametrize("wires", ((0, 1), (0, 4), (3, 1)))
def test_native_rzz_adjoint_matches_pytorch_reference(
    dtype: torch.dtype, wires: tuple[int, int]
) -> None:
    if not native_cpu_adjoint_available():
        pytest.skip("native CPU extension is not built in this source checkout")
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    generator = torch.Generator().manual_seed(2903 + sum(wires))
    ket = (
        torch.randn((2, 32), generator=generator)
        + 1j * torch.randn((2, 32), generator=generator)
    ).to(dtype)
    adjoint = (
        torch.randn((2, 32), generator=generator)
        + 1j * torch.randn((2, 32), generator=generator)
    ).to(dtype)
    matrix = _rotation_matrix("rzz", torch.tensor(0.31, dtype=real_dtype), dtype)
    cross = adjoint.real * ket.imag - adjoint.imag * ket.real
    indices = torch.arange(32)
    first = (indices >> (4 - wires[0])) & 1
    second = (indices >> (4 - wires[1])) & 1
    signs = (1 - 2 * (first ^ second)).to(cross.dtype)
    expected_gradient = 0.5 * (cross * signs).sum()
    inverse = torch.where(signs > 0, matrix[0, 0].conj(), matrix[1, 1].conj()).reshape(
        1, -1
    )
    expected_ket = ket * inverse
    expected_adjoint = adjoint * inverse

    gradient = fused_rotation_adjoint_(
        ket,
        adjoint,
        matrix,
        name="rzz",
        wire=wires[0],
        second_wire=wires[1],
        n_wires=5,
    )

    assert gradient is not None
    tolerance = 2e-5 if dtype == torch.complex64 else 1e-12
    torch.testing.assert_close(
        gradient, expected_gradient, atol=tolerance, rtol=tolerance
    )
    torch.testing.assert_close(ket, expected_ket, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(
        adjoint, expected_adjoint, atol=tolerance, rtol=tolerance
    )


@pytest.mark.parametrize("dtype", (torch.complex64, torch.complex128))
def test_native_rzz_segment_matches_sequential_reference(dtype: torch.dtype) -> None:
    if not native_cpu_adjoint_available():
        pytest.skip("native CPU extension is not built in this source checkout")
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    generator = torch.Generator().manual_seed(4109)
    original_ket = (
        torch.randn((2, 32), generator=generator)
        + 1j * torch.randn((2, 32), generator=generator)
    ).to(dtype)
    original_adjoint = (
        torch.randn((2, 32), generator=generator)
        + 1j * torch.randn((2, 32), generator=generator)
    ).to(dtype)
    angles = torch.tensor((0.19, -0.37, 0.23), dtype=real_dtype)
    first_wires = torch.tensor((0, 3, 1), dtype=torch.int64)
    second_wires = torch.tensor((4, 1, 2), dtype=torch.int64)
    expected_ket = original_ket.clone()
    expected_adjoint = original_adjoint.clone()
    expected_gradients = []
    for angle, first, second in zip(angles, first_wires, second_wires, strict=True):
        matrix = _rotation_matrix("rzz", angle, dtype)
        gradient = fused_rotation_adjoint_(
            expected_ket,
            expected_adjoint,
            matrix,
            name="rzz",
            wire=int(first),
            second_wire=int(second),
            n_wires=5,
        )
        assert gradient is not None
        expected_gradients.append(gradient)
    ket = original_ket.clone()
    adjoint = original_adjoint.clone()

    gradients = fused_rzz_segment_adjoint_(
        ket,
        adjoint,
        angles,
        first_wires,
        second_wires,
        n_wires=5,
    )

    assert gradients is not None
    tolerance = 3e-5 if dtype == torch.complex64 else 2e-12
    torch.testing.assert_close(
        gradients,
        torch.stack(expected_gradients),
        atol=tolerance,
        rtol=tolerance,
    )
    torch.testing.assert_close(ket, expected_ket, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(
        adjoint, expected_adjoint, atol=tolerance, rtol=tolerance
    )


@pytest.mark.parametrize("dtype", (torch.complex64, torch.complex128))
def test_native_shared_rzz_path_aggregates_gradient(dtype: torch.dtype) -> None:
    if not native_cpu_adjoint_available():
        pytest.skip("native CPU extension is not built in this source checkout")
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    generator = torch.Generator().manual_seed(5113)
    original_ket = (
        torch.randn((2, 32), generator=generator)
        + 1j * torch.randn((2, 32), generator=generator)
    ).to(dtype)
    original_adjoint = (
        torch.randn((2, 32), generator=generator)
        + 1j * torch.randn((2, 32), generator=generator)
    ).to(dtype)
    angles = torch.full((4,), 0.29, dtype=real_dtype)
    first_wires = torch.tensor((3, 2, 1, 0), dtype=torch.int64)
    second_wires = torch.tensor((4, 3, 2, 1), dtype=torch.int64)
    expected_ket = original_ket.clone()
    expected_adjoint = original_adjoint.clone()
    expected_gradient = torch.zeros((), dtype=real_dtype)
    for angle, first, second in zip(angles, first_wires, second_wires, strict=True):
        gradient = fused_rotation_adjoint_(
            expected_ket,
            expected_adjoint,
            _rotation_matrix("rzz", angle, dtype),
            name="rzz",
            wire=int(first),
            second_wire=int(second),
            n_wires=5,
        )
        assert gradient is not None
        expected_gradient += gradient
    ket = original_ket.clone()
    adjoint = original_adjoint.clone()

    gradients = fused_rzz_segment_adjoint_(
        ket,
        adjoint,
        angles,
        first_wires,
        second_wires,
        n_wires=5,
        aggregate_shared_parameter=True,
    )

    assert gradients is not None
    tolerance = 3e-5 if dtype == torch.complex64 else 2e-12
    torch.testing.assert_close(
        gradients[0], expected_gradient, atol=tolerance, rtol=tolerance
    )
    torch.testing.assert_close(
        gradients[1:], torch.zeros_like(gradients[1:]), atol=0, rtol=0
    )
    torch.testing.assert_close(ket, expected_ket, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(
        adjoint, expected_adjoint, atol=tolerance, rtol=tolerance
    )


@pytest.mark.parametrize("name", ("custom",))
def test_native_rotation_adjoint_rejects_unsupported_gates(name: str) -> None:
    ket = torch.ones((1, 4), dtype=torch.complex128)
    adjoint = torch.ones_like(ket)
    matrix = torch.eye(2, dtype=torch.complex128)

    assert (
        fused_rotation_adjoint_(ket, adjoint, matrix, name=name, wire=0, n_wires=2)
        is None
    )
