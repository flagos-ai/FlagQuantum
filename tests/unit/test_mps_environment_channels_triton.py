from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")

import flagquantum.kernels.triton.mps_environment as mps_environment
from flagquantum.kernels.triton.mps_environment import (
    fused_mps_environment_channels,
)

pytestmark = [pytest.mark.unit, pytest.mark.triton, pytest.mark.gpu]


def _reference(channels: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
    return torch.einsum("tbij,bipr,bjps->tbrs", channels, tensor.conj(), tensor)


def test_fused_mps_environment_channels_cpu_fallback_preserves_gradients() -> None:
    torch.manual_seed(91)
    channels = torch.randn(5, 2, 3, 3, dtype=torch.complex128, requires_grad=True)
    tensor = torch.randn(2, 3, 2, 4, dtype=torch.complex128, requires_grad=True)
    expected_channels = channels.detach().clone().requires_grad_(True)
    expected_tensor = tensor.detach().clone().requires_grad_(True)

    actual = fused_mps_environment_channels(channels, tensor)
    expected = _reference(expected_channels, expected_tensor)
    cotangent = torch.randn_like(actual)
    actual.backward(cotangent)
    expected.backward(cotangent)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(channels.grad, expected_channels.grad)
    torch.testing.assert_close(tensor.grad, expected_tensor.grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("channel_count", "batch", "left_dim", "right_dim"),
    (
        (5, 2, 4, 4),
        (8, 8, 8, 8),
        (8, 8, 16, 16),
        (16, 8, 16, 16),
        (32, 8, 8, 8),
    ),
)
def test_fused_mps_environment_channels_cuda_matches_reference(
    channel_count: int,
    batch: int,
    left_dim: int,
    right_dim: int,
) -> None:
    torch.manual_seed(92)
    channels = torch.randn(
        channel_count,
        batch,
        left_dim,
        left_dim,
        device="cuda",
        dtype=torch.complex64,
    )
    tensor = torch.randn(
        batch,
        left_dim,
        2,
        right_dim,
        device="cuda",
        dtype=torch.complex64,
    )

    actual = fused_mps_environment_channels(channels, tensor)
    expected = _reference(channels, tensor)

    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("channel_count", "batch", "left_dim", "right_dim"),
    (
        (33, 1, 4, 4),
        (32, 8, 16, 16),
        (32, 16, 16, 16),
        (8, 16, 32, 16),
        (8, 16, 16, 32),
    ),
)
def test_fused_mps_environment_channels_outside_window_uses_fallback(
    monkeypatch: pytest.MonkeyPatch,
    channel_count: int,
    batch: int,
    left_dim: int,
    right_dim: int,
) -> None:
    channels = torch.randn(
        channel_count,
        batch,
        left_dim,
        left_dim,
        device="cuda",
        dtype=torch.complex64,
    )
    tensor = torch.randn(
        batch,
        left_dim,
        2,
        right_dim,
        device="cuda",
        dtype=torch.complex64,
    )

    def unexpected_launch(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("unsupported channel shape must not launch Triton")

    monkeypatch.setattr(mps_environment, "_launch_channels", unexpected_launch)
    actual = fused_mps_environment_channels(channels, tensor)

    torch.testing.assert_close(actual, _reference(channels, tensor))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_mps_environment_channels_noncontiguous_input_uses_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    channels = torch.randn(5, 2, 4, 4, device="cuda", dtype=torch.complex64)
    channels = channels.transpose(-2, -1)
    tensor = torch.randn(2, 4, 2, 4, device="cuda", dtype=torch.complex64)

    def unexpected_launch(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("noncontiguous channels must not launch Triton")

    monkeypatch.setattr(mps_environment, "_launch_channels", unexpected_launch)
    actual = fused_mps_environment_channels(channels, tensor)

    torch.testing.assert_close(actual, _reference(channels, tensor))


@pytest.mark.parametrize(
    ("channels_shape", "tensor_shape", "message"),
    (
        ((2, 3, 3), (2, 3, 2, 4), "channels,batch,left,left"),
        ((5, 2, 3, 4), (2, 3, 2, 4), "square matrices"),
        ((5, 2, 3, 3), (2, 4, 2, 4), "do not align"),
    ),
)
def test_fused_mps_environment_channels_validates_shapes(
    channels_shape: tuple[int, ...],
    tensor_shape: tuple[int, ...],
    message: str,
) -> None:
    channels = torch.zeros(channels_shape, dtype=torch.complex64)
    tensor = torch.zeros(tensor_shape, dtype=torch.complex64)

    with pytest.raises(ValueError, match=message):
        fused_mps_environment_channels(channels, tensor)
