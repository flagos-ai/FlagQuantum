from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")

import flagquantum.kernels.triton.mps_observable_adjoint as mps_observable_adjoint
from flagquantum.kernels.triton.mps_observable_adjoint import (
    fused_mps_hermitian_observable_adjoint,
)

pytestmark = [pytest.mark.unit, pytest.mark.triton, pytest.mark.gpu]


def _hermitian(value: torch.Tensor) -> torch.Tensor:
    return value + value.mH


def _inputs(
    batch: int,
    left_dim: int,
    right_dim: int,
    *,
    device: str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, ...]:
    tensor = torch.randn(batch, left_dim, 2, right_dim, device=device, dtype=dtype)
    left = _hermitian(
        torch.randn(batch, left_dim, left_dim, device=device, dtype=dtype)
    )
    right = _hermitian(
        torch.randn(batch, right_dim, right_dim, device=device, dtype=dtype)
    )
    operator = _hermitian(torch.randn(2, 2, device=device, dtype=dtype))
    weight_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    weights = torch.randn(batch, device=device, dtype=weight_dtype)
    return tensor, left, right, operator, weights


def _autograd_reference(
    tensor: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    operator: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    variable = tensor.detach().requires_grad_(True)
    values = torch.real(
        torch.einsum(
            "bij,bipr,pq,bjqs,brs->b",
            left,
            variable.conj(),
            operator,
            variable,
            right,
        )
    )
    return torch.autograd.grad(torch.sum(weights * values), variable)[0].detach()


def test_fused_mps_hermitian_observable_adjoint_cpu_fallback_matches_autograd() -> None:
    torch.manual_seed(94)
    inputs = _inputs(2, 3, 4, device="cpu", dtype=torch.complex128)

    actual = fused_mps_hermitian_observable_adjoint(*inputs)
    expected = _autograd_reference(*inputs)

    torch.testing.assert_close(actual, expected)
    assert actual.requires_grad is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("batch", "left_dim", "right_dim"),
    ((1, 1, 4), (1, 4, 8), (1, 16, 32), (1, 32, 64), (8, 16, 16)),
)
def test_fused_mps_hermitian_observable_adjoint_cuda_matches_autograd(
    batch: int,
    left_dim: int,
    right_dim: int,
) -> None:
    torch.manual_seed(95 + batch + left_dim + right_dim)
    inputs = _inputs(
        batch,
        left_dim,
        right_dim,
        device="cuda",
        dtype=torch.complex64,
    )

    actual = fused_mps_hermitian_observable_adjoint(*inputs)
    expected = _autograd_reference(*inputs)

    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=5e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_mps_hermitian_observable_adjoint_outside_window_uses_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(96)
    inputs = _inputs(1, 64, 64, device="cuda", dtype=torch.complex64)

    def unexpected_launch(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("unsupported shape must not launch the Triton kernel")

    monkeypatch.setattr(mps_observable_adjoint, "_launch", unexpected_launch)
    actual = fused_mps_hermitian_observable_adjoint(*inputs)

    torch.testing.assert_close(
        actual,
        _autograd_reference(*inputs),
        rtol=2e-4,
        atol=1e-4,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_mps_hermitian_observable_adjoint_noncontiguous_uses_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(97)
    tensor, left, right, operator, weights = _inputs(
        2, 4, 4, device="cuda", dtype=torch.complex64
    )
    left = left.transpose(-2, -1)

    def unexpected_launch(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("noncontiguous input must not launch the Triton kernel")

    monkeypatch.setattr(mps_observable_adjoint, "_launch", unexpected_launch)
    actual = fused_mps_hermitian_observable_adjoint(
        tensor, left, right, operator, weights
    )

    torch.testing.assert_close(
        actual,
        _autograd_reference(tensor, left, right, operator, weights),
        rtol=2e-4,
        atol=1e-4,
    )


@pytest.mark.parametrize(
    (
        "tensor_shape",
        "left_shape",
        "right_shape",
        "operator_shape",
        "weight_shape",
        "message",
    ),
    (
        ((2, 3, 3), (2, 3, 3), (2, 4, 4), (2, 2), (2,), "tensor"),
        ((2, 3, 2, 4), (2, 4, 4), (2, 4, 4), (2, 2), (2,), "left environment"),
        ((2, 3, 2, 4), (2, 3, 3), (2, 3, 3), (2, 2), (2,), "right environment"),
        ((2, 3, 2, 4), (2, 3, 3), (2, 4, 4), (4, 4), (2,), "operator"),
        ((2, 3, 2, 4), (2, 3, 3), (2, 4, 4), (2, 2), (3,), "weights"),
    ),
)
def test_fused_mps_hermitian_observable_adjoint_validates_shapes(
    tensor_shape: tuple[int, ...],
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
    operator_shape: tuple[int, ...],
    weight_shape: tuple[int, ...],
    message: str,
) -> None:
    tensor = torch.zeros(tensor_shape, dtype=torch.complex64)
    left = torch.zeros(left_shape, dtype=torch.complex64)
    right = torch.zeros(right_shape, dtype=torch.complex64)
    operator = torch.zeros(operator_shape, dtype=torch.complex64)
    weights = torch.zeros(weight_shape, dtype=torch.float32)

    with pytest.raises(ValueError, match=message):
        fused_mps_hermitian_observable_adjoint(tensor, left, right, operator, weights)


def test_fused_mps_hermitian_observable_adjoint_validates_dtype() -> None:
    tensor = torch.zeros(2, 3, 2, 4)
    left = torch.zeros(2, 3, 3)
    right = torch.zeros(2, 4, 4)
    operator = torch.zeros(2, 2)
    weights = torch.zeros(2)

    with pytest.raises(ValueError, match="complex dtype"):
        fused_mps_hermitian_observable_adjoint(tensor, left, right, operator, weights)
