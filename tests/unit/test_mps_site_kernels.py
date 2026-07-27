import pytest
import torch

from flagquantum.runtime.backends.mps.site_kernels import (
    apply_rxx_contraction_bucket,
    apply_ry_bucket,
    environment_transfer,
    environment_transfer_channels,
    reset_site_kernel_stats,
    site_kernel_stats,
)

pytestmark = pytest.mark.unit


def test_rank_local_ry_bucket_matches_eager_values_and_gradients():
    torch.manual_seed(7)
    tensors = torch.randn(3, 2, 2, 2, 3, dtype=torch.complex64, requires_grad=True)
    angles = torch.randn(3, 2, requires_grad=True)
    matrices = torch.zeros(3, 2, 2, 2, dtype=torch.complex64)
    matrices[..., 0, 0] = torch.cos(angles / 2)
    matrices[..., 0, 1] = -torch.sin(angles / 2)
    matrices[..., 1, 0] = torch.sin(angles / 2)
    matrices[..., 1, 1] = torch.cos(angles / 2)
    actual = apply_ry_bucket(tensors, matrices, compiled=False)
    expected = torch.stack(
        [torch.einsum("bpq,blqr->blpr", matrices[i], tensors[i]) for i in range(3)]
    )
    torch.testing.assert_close(actual, expected)
    actual.abs().square().sum().backward()
    assert tensors.grad is not None and torch.isfinite(tensors.grad).all()
    assert angles.grad is not None and torch.isfinite(angles.grad).all()


@pytest.mark.parametrize("z", (False, True))
def test_real_channel_environment_transfer_matches_complex_einsum(z):
    torch.manual_seed(8)
    tensor = torch.randn(2, 3, 2, 4, dtype=torch.complex64)
    env = torch.randn(2, 3, 3, dtype=torch.complex64)
    signs = tensor.real.new_tensor((1.0, -1.0) if z else (1.0, 1.0))
    expected = torch.einsum(
        "bij,bipr,bjps->brs",
        env,
        tensor.conj(),
        tensor * signs.reshape(1, 1, 2, 1),
    )
    actual = environment_transfer(env, tensor, z=z, compiled=False)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_observable_channel_transfer_fuses_terms_and_preserves_gradients():
    torch.manual_seed(98)
    channels = torch.randn(5, 2, 3, 3, dtype=torch.complex64, requires_grad=True)
    tensor = torch.randn(2, 3, 2, 4, dtype=torch.complex64, requires_grad=True)
    actual = environment_transfer_channels(channels, tensor, compiled=False)
    expected = torch.stack(
        [
            torch.einsum("bij,bipr,bjps->brs", channel, tensor.conj(), tensor)
            for channel in channels
        ]
    )
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    actual.abs().square().sum().backward()
    assert channels.grad is not None and torch.isfinite(channels.grad).all()
    assert tensor.grad is not None and torch.isfinite(tensor.grad).all()


def test_rank_local_rxx_bucket_matches_individual_contractions():
    torch.manual_seed(9)
    left = torch.randn(3, 2, 2, 2, 3, dtype=torch.complex64)
    right = torch.randn(3, 2, 3, 2, 2, dtype=torch.complex64)
    matrix = torch.randn(3, 2, 4, 4, dtype=torch.complex64)
    actual = apply_rxx_contraction_bucket(left, right, matrix, compiled=False)
    expected = []
    for bond in range(3):
        theta = torch.einsum("blsm,bmtr->blstr", left[bond], right[bond])
        theta = theta.reshape(2, 2, 4, 2)
        theta = torch.einsum("bij,bljr->blir", matrix[bond], theta)
        expected.append(theta.reshape(2, 4, 4))
    torch.testing.assert_close(actual, torch.stack(expected), atol=2e-5, rtol=2e-5)


def test_kernel_stats_distinguish_eager_from_compiled_claims():
    reset_site_kernel_stats()
    tensor = torch.ones(1, 1, 1, 2, 1, dtype=torch.complex64)
    matrix = torch.eye(2, dtype=torch.complex64).reshape(1, 1, 2, 2)
    apply_ry_bucket(tensor, matrix, compiled=False)
    stats = site_kernel_stats()
    assert stats["ry_bucket_calls"] == 1
    assert stats["compiled_calls"] == 0
    assert stats["dynamo_graphs"] == 0
