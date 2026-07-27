import pytest
import torch

from flagquantum.simulation.triton_complex_bmm import fused_complex_bmm


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("shape", [(2, 3, 4, 5), (16, 8, 2, 32), (4, 32, 32, 2)])
def test_fused_complex_bmm_forward_and_backward_match_torch(shape) -> None:
    batch, rows, reduction, columns = shape
    left = torch.randn(
        batch, rows, reduction, device="cuda", dtype=torch.complex64
    ).requires_grad_(True)
    right = torch.randn(
        batch, reduction, columns, device="cuda", dtype=torch.complex64
    ).requires_grad_(True)
    reference_left = left.detach().clone().requires_grad_(True)
    reference_right = right.detach().clone().requires_grad_(True)

    actual = fused_complex_bmm(left, right)
    reference = torch.bmm(reference_left, reference_right)
    weights = torch.randn_like(reference)
    actual_gradient = torch.autograd.grad(
        torch.real((actual * weights.conj()).sum()), (left, right)
    )
    reference_gradient = torch.autograd.grad(
        torch.real((reference * weights.conj()).sum()),
        (reference_left, reference_right),
    )

    torch.testing.assert_close(actual, reference, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(
        actual_gradient[0], reference_gradient[0], atol=3e-5, rtol=3e-5
    )
    torch.testing.assert_close(
        actual_gradient[1], reference_gradient[1], atol=3e-5, rtol=3e-5
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_complex_bmm_accepts_strided_inputs_and_gradient() -> None:
    left_base = torch.randn(3, 5, 4, device="cuda", dtype=torch.complex64)
    right_base = torch.randn(3, 6, 5, device="cuda", dtype=torch.complex64)
    left = left_base.transpose(-2, -1).requires_grad_(True)
    right = right_base.transpose(-2, -1).requires_grad_(True)
    reference_left = left.detach().clone().requires_grad_(True)
    reference_right = right.detach().clone().requires_grad_(True)

    actual = fused_complex_bmm(left, right)
    reference = torch.bmm(reference_left, reference_right)
    gradient = torch.randn(3, 6, 4, device="cuda", dtype=torch.complex64).transpose(
        -2, -1
    )
    actual_gradient = torch.autograd.grad(actual, (left, right), gradient)
    reference_gradient = torch.autograd.grad(
        reference, (reference_left, reference_right), gradient
    )

    torch.testing.assert_close(actual, reference, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(
        actual_gradient[0], reference_gradient[0], atol=3e-5, rtol=3e-5
    )
    torch.testing.assert_close(
        actual_gradient[1], reference_gradient[1], atol=3e-5, rtol=3e-5
    )
