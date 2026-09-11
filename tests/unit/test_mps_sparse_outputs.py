import pytest
import torch

from flagquantum.core.ir import Instruction
from flagquantum.simulation.matrices import GATE_MAT_DICT
from flagquantum.simulation.mps.state import MPSState

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("name", ["z", "i", "swap"])
def test_mps_rejects_parameterized_fixed_gate_before_state_mutation(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    state = MPSState.zero(2)
    before = tuple(tensor.clone() for tensor in state.tensors)
    monkeypatch.setitem(GATE_MAT_DICT, name, lambda parameter: parameter)
    with pytest.raises(ValueError, match=f"MPS gate '{name}' requires a fixed matrix"):
        if name == "swap":
            state.apply_swap(0)
        else:
            state.expectation_z(0)
    for actual, expected in zip(state.tensors, before):
        torch.testing.assert_close(actual, expected)
    assert state.local_swap_count == 0


def test_mps_rejects_channel_without_kraus_operators_before_mutation() -> None:
    state = MPSState.zero(2)
    before = tuple(tensor.clone() for tensor in state.tensors)
    instruction = Instruction("channel", (0,), metadata={"is_channel": True})

    with pytest.raises(ValueError, match="require Kraus operators"):
        state.apply_instruction(instruction)

    for actual, expected in zip(state.tensors, before):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_mps_pauli_observables_reject_parameterized_matrix_entries(
    monkeypatch: pytest.MonkeyPatch, axis: str
) -> None:
    state = MPSState.zero(2)
    monkeypatch.setitem(GATE_MAT_DICT, axis, lambda parameter: parameter)
    with pytest.raises(
        ValueError, match="Pauli observables require fixed gate matrices"
    ):
        state.expectation_ps(**{axis: (0, 1)})


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
