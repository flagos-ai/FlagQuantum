import pytest
import torch

from flagquantum.simulation.mps_state import MPSState

pytestmark = pytest.mark.unit


def test_mps_selected_amplitudes_do_not_require_dense_state() -> None:
    state = MPSState.zero(64)

    values = state.amplitudes(("0" * 64, "1" * 64))

    assert values.shape == (1, 2)
    assert torch.allclose(values, torch.tensor([[1, 0]], dtype=values.dtype))


def test_mps_selected_amplitude_preserves_gradients() -> None:
    theta = torch.tensor(0.3, requires_grad=True)
    left = torch.stack((torch.cos(theta), torch.sin(theta))).reshape(1, 1, 2, 1)
    right = torch.tensor([1, 0], dtype=torch.float32).reshape(1, 1, 2, 1)
    state = MPSState((left, right))

    value = state.amplitude("00")
    (gradient,) = torch.autograd.grad(value.sum(), (theta,))

    assert torch.allclose(gradient, -torch.sin(theta))
