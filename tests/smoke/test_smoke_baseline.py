import pytest
import torch

import flagquantum as fq

pytestmark = pytest.mark.smoke


def test_import_flagquantum_top_level_api():
    assert fq.Circuit is not None
    assert fq.run_native is not None
    assert fq.get_backend() == "pytorch"


def test_minimal_circuit_expectation_fast_path():
    circuit = fq.Circuit(2)
    circuit.h(0).cx(0, 1)

    state = circuit.state()
    expectation = circuit.expectation_z()

    expected = torch.tensor([[2**-0.5, 0.0, 0.0, 2**-0.5]], dtype=state.dtype)
    assert torch.allclose(state, expected, atol=1e-6)
    assert torch.allclose(expectation, torch.zeros(1, 2), atol=1e-6)


def test_minimal_parameter_autograd_expectation():
    theta = torch.tensor(0.3, requires_grad=True)
    circuit = fq.Circuit(1)
    circuit.rx(0, theta=theta)

    loss = circuit.expectation_z(0).sum()
    loss.backward()

    assert theta.grad is not None
    assert torch.allclose(theta.grad, -torch.sin(theta.detach()), atol=1e-6)
