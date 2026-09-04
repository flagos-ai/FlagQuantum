import pytest
import torch

from flagquantum.simulation.mps_models import MPSConfig
from flagquantum.simulation.mps_reverse import (
    factor_mps_reverse_pair,
    factor_mps_reverse_pair_bucket,
    mps_vjp,
    project_mps_adjoint,
)

pytestmark = pytest.mark.unit


def test_reverse_pair_factorization_preserves_exact_pair_and_gradient() -> None:
    generator = torch.Generator().manual_seed(13)
    pair = torch.randn((1, 4, 4), dtype=torch.complex128, generator=generator)

    pair_leaf, left, right, info = factor_mps_reverse_pair(
        pair,
        left_dim=2,
        right_dim=2,
        config=MPSConfig(),
    )

    reconstructed = torch.einsum("blsm,bmtr->blstr", left, right).reshape(1, 4, 4)
    torch.testing.assert_close(reconstructed, pair)
    assert pair_leaf.is_leaf and pair_leaf.requires_grad
    assert info["method"] == "exact_autograd"
    right.abs().square().sum().backward()
    assert pair_leaf.grad is not None
    assert torch.isfinite(pair_leaf.grad).all()


def test_reverse_pair_bucket_uses_projected_truncated_gradient() -> None:
    generator = torch.Generator().manual_seed(17)
    pairs = torch.randn((2, 1, 4, 4), dtype=torch.complex128, generator=generator)

    outputs = factor_mps_reverse_pair_bucket(
        pairs,
        left_dim=2,
        right_dim=2,
        config=MPSConfig(max_bond=1),
        batched_truncated_split=True,
    )

    assert len(outputs) == 2
    for pair, (pair_leaf, left, right, info) in zip(pairs, outputs):
        retained_u = left.reshape(1, 4, 1)
        expected_right = torch.matmul(
            torch.conj(retained_u).transpose(-2, -1), pair
        ).reshape(1, 1, 2, 2)
        torch.testing.assert_close(right, expected_right)
        assert pair_leaf.is_leaf and pair_leaf.requires_grad
        assert not left.requires_grad
        assert info["method"] == "svd"
        assert info["gradient_method"] == "projected_stop_subspace"

    sum(right.abs().square().sum() for _, _, right, _ in outputs).backward()
    for pair_leaf, _, _, _ in outputs:
        assert pair_leaf.grad is not None
        assert torch.isfinite(pair_leaf.grad).all()


def test_mps_vjp_projects_truncated_bonds_and_returns_input_parameter_grads() -> None:
    value = torch.tensor([[[1.0, 2.0]]], requires_grad=True)
    parameter = torch.tensor(3.0, requires_grad=True)
    output = value * parameter
    stale_adjoint = torch.tensor([[[2.0, 4.0, 8.0]]])

    value_grad, parameter_grad = mps_vjp(
        (output,),
        (value,),
        (parameter,),
        (stale_adjoint,),
    )

    torch.testing.assert_close(value_grad, torch.tensor([[[6.0, 12.0]]]))
    torch.testing.assert_close(parameter_grad, torch.tensor(10.0))


def test_mps_adjoint_projection_rejects_rank_or_batch_mismatch() -> None:
    target = torch.zeros((2, 1, 2, 1))
    with pytest.raises(ValueError, match="rank/batch mismatch"):
        project_mps_adjoint(torch.zeros((1, 1, 2, 1)), target)
