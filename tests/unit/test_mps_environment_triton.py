from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")

import flagquantum.kernels.triton.mps_environment as mps_environment
from flagquantum.kernels.triton.mps_environment import (
    fused_mps_environment_transfer,
)

pytestmark = [pytest.mark.unit, pytest.mark.triton, pytest.mark.gpu]


def _reference(
    environment: torch.Tensor,
    tensor: torch.Tensor,
    *,
    insert_z: bool,
) -> torch.Tensor:
    signs = tensor.real.new_tensor((1.0, -1.0) if insert_z else (1.0, 1.0))
    return torch.einsum(
        "bij,bipr,bjps->brs",
        environment,
        tensor.conj(),
        tensor * signs.reshape(1, 1, 2, 1),
    )


@pytest.mark.parametrize("insert_z", (False, True))
def test_fused_mps_environment_cpu_fallback_matches_reference_and_gradients(
    insert_z: bool,
) -> None:
    environment = torch.randn(2, 3, 3, dtype=torch.complex128, requires_grad=True)
    tensor = torch.randn(2, 3, 2, 4, dtype=torch.complex128, requires_grad=True)
    expected_environment = environment.detach().clone().requires_grad_(True)
    expected_tensor = tensor.detach().clone().requires_grad_(True)

    actual = fused_mps_environment_transfer(environment, tensor, insert_z=insert_z)
    expected = _reference(expected_environment, expected_tensor, insert_z=insert_z)
    cotangent = torch.randn_like(actual)
    actual.backward(cotangent)
    expected.backward(cotangent)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(environment.grad, expected_environment.grad)
    torch.testing.assert_close(tensor.grad, expected_tensor.grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("insert_z", (False, True))
@pytest.mark.parametrize(
    ("batch", "left_dim", "right_dim"),
    ((1, 4, 4), (8, 16, 16), (16, 32, 16), (16, 16, 32)),
)
def test_fused_mps_environment_cuda_matches_reference(
    batch: int,
    left_dim: int,
    right_dim: int,
    insert_z: bool,
) -> None:
    torch.manual_seed(90)
    environment = torch.randn(
        batch, left_dim, left_dim, device="cuda", dtype=torch.complex64
    )
    tensor = torch.randn(
        batch, left_dim, 2, right_dim, device="cuda", dtype=torch.complex64
    )

    actual = fused_mps_environment_transfer(environment, tensor, insert_z=insert_z)
    expected = _reference(environment, tensor, insert_z=insert_z)

    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_mps_environment_large_shape_uses_fallback(monkeypatch) -> None:
    environment = torch.randn(16, 32, 32, device="cuda", dtype=torch.complex64)
    tensor = torch.randn(16, 32, 2, 32, device="cuda", dtype=torch.complex64)

    def unexpected_launch(*args, **kwargs):
        raise AssertionError("unsupported shape must not launch the Triton kernel")

    monkeypatch.setattr(mps_environment, "_launch", unexpected_launch)
    actual = fused_mps_environment_transfer(environment, tensor)

    torch.testing.assert_close(actual, _reference(environment, tensor, insert_z=False))
