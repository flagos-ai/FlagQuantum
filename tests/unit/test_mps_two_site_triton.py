import pytest
import torch

pytest.importorskip("triton")

from flagquantum.simulation.triton_kernels.mps_two_site import (
    fused_mps_range_projection,
    fused_mps_two_site,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("batched_gate", [False, True])
def test_fused_mps_two_site_forward_and_backward(batched_gate: bool) -> None:
    batch, left_dim, bond_dim, right_dim = 3, 2, 4, 3
    left = torch.randn(
        batch, left_dim, 2, bond_dim, dtype=torch.complex64, device="cuda"
    ).requires_grad_(True)
    right = torch.randn(
        batch, bond_dim, 2, right_dim, dtype=torch.complex64, device="cuda"
    ).requires_grad_(True)
    gate_shape = (batch, 4, 4) if batched_gate else (4, 4)
    gate = torch.randn(gate_shape, dtype=torch.complex64, device="cuda").requires_grad_(
        True
    )
    reference_inputs = tuple(
        value.detach().clone().requires_grad_(True) for value in (left, gate, right)
    )

    actual = fused_mps_two_site(left, gate, right)
    ref_left, ref_gate, ref_right = reference_inputs
    theta = torch.einsum("blsm,bmtr->blstr", ref_left, ref_right).reshape(
        batch, left_dim, 4, right_dim
    )
    equation = "bij,bljr->blir" if batched_gate else "ij,bljr->blir"
    reference = torch.einsum(equation, ref_gate, theta).reshape(
        batch, left_dim * 2, 2 * right_dim
    )
    weight = torch.randn_like(reference)
    actual_gradients = torch.autograd.grad(
        torch.real((actual * weight.conj()).sum()), (left, gate, right)
    )
    reference_gradients = torch.autograd.grad(
        torch.real((reference * weight.conj()).sum()), reference_inputs
    )

    torch.testing.assert_close(actual, reference, atol=3e-5, rtol=3e-5)
    for actual_gradient, reference_gradient in zip(
        actual_gradients, reference_gradients
    ):
        torch.testing.assert_close(
            actual_gradient, reference_gradient, atol=5e-5, rtol=5e-5
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_mps_range_projection_avoids_full_matrix_with_correct_values():
    batch, left_dim, bond_dim, right_dim, rank = 2, 3, 5, 4, 6
    left = torch.randn(
        batch, left_dim, 2, bond_dim, dtype=torch.complex64, device="cuda"
    )
    right = torch.randn(
        batch, bond_dim, 2, right_dim, dtype=torch.complex64, device="cuda"
    )
    gate = torch.randn(batch, 4, 4, dtype=torch.complex64, device="cuda")
    projection = torch.randn(2 * right_dim, rank, dtype=torch.complex64, device="cuda")
    actual = fused_mps_range_projection(left, gate, right, projection)
    theta = torch.einsum("blsm,bmtr->blstr", left, right).reshape(
        batch, left_dim, 4, right_dim
    )
    matrix = torch.einsum("bij,bljr->blir", gate, theta).reshape(
        batch, left_dim * 2, 2 * right_dim
    )
    torch.testing.assert_close(actual, matrix @ projection, atol=5e-4, rtol=5e-4)
