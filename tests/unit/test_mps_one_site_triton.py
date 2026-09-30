from __future__ import annotations

import pytest
import torch

from flagquantum.kernels.triton.mps_one_site import fused_mps_one_site

pytestmark = pytest.mark.unit


def _reference(tensor: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    equation = "pq,blqr->blpr" if gate.ndim == 2 else "bpq,blqr->blpr"
    return torch.einsum(equation, gate, tensor)


def test_fused_mps_one_site_cpu_fallback_matches_reference() -> None:
    tensor = torch.randn(2, 3, 2, 5, dtype=torch.complex64)
    gate = torch.randn(2, 2, dtype=torch.complex64)

    assert torch.allclose(fused_mps_one_site(tensor, gate), _reference(tensor, gate))


@pytest.mark.gpu
@pytest.mark.triton
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("batched_gate", (False, True))
def test_fused_mps_one_site_forward_and_backward(batched_gate: bool) -> None:
    tensor = torch.randn(
        3, 17, 2, 19, device="cuda", dtype=torch.complex64, requires_grad=True
    )
    gate_shape = (3, 2, 2) if batched_gate else (2, 2)
    gate = torch.randn(
        *gate_shape, device="cuda", dtype=torch.complex64, requires_grad=True
    )
    reference_tensor = tensor.detach().clone().requires_grad_(True)
    reference_gate = gate.detach().clone().requires_grad_(True)
    cotangent = torch.randn_like(tensor)

    actual = fused_mps_one_site(tensor, gate)
    expected = _reference(reference_tensor, reference_gate)
    actual.backward(cotangent)
    expected.backward(cotangent)

    assert torch.allclose(actual, expected, rtol=2e-5, atol=2e-5)
    assert torch.allclose(tensor.grad, reference_tensor.grad, rtol=2e-5, atol=2e-5)
    assert torch.allclose(gate.grad, reference_gate.grad, rtol=2e-5, atol=2e-5)
