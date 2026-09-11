import pytest
import torch

pytest.importorskip("triton")

import flagquantum.algorithms as fqa
from flagquantum.simulation.triton_kernels import heisenberg_hva_forward_tangents


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("n_wires,depth", [(2, 1), (4, 2), (6, 3)])
def test_hva_forward_tangents_match_statevector_jacobian(n_wires, depth):
    torch.manual_seed(17)
    count = fqa.heisenberg_hva_parameter_count(n_wires, depth)
    parameters = (0.02 * torch.randn(count, device="cuda")).requires_grad_()
    zero = torch.zeros_like(parameters)
    initial = fqa.heisenberg_hva(n_wires, depth, zero).state().detach().reshape(-1)
    actual_state, actual_tangents = heisenberg_hva_forward_tangents(
        initial, parameters.detach(), n_wires=n_wires, depth=depth
    )

    def state(value):
        return fqa.heisenberg_hva(n_wires, depth, value).state().reshape(-1)

    expected_state = state(parameters)
    real = torch.autograd.functional.jacobian(
        lambda value: state(value).real, parameters, vectorize=True
    )
    imag = torch.autograd.functional.jacobian(
        lambda value: state(value).imag, parameters, vectorize=True
    )
    expected_tangents = torch.complex(real, imag).T
    torch.testing.assert_close(actual_state, expected_state, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(actual_tangents, expected_tangents, atol=2e-5, rtol=2e-5)
