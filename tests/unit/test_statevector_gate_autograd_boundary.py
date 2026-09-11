"""CPU checks of statevector autograd orchestration with replacement launches."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

pytestmark = pytest.mark.unit


@pytest.fixture
def gate_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    triton = ModuleType("triton")
    triton.jit = lambda function=None, **kwargs: (
        function if function is not None else lambda value: value
    )
    language = ModuleType("triton.language")
    triton.language = language
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "triton.language", language)
    path = (
        Path(__file__).resolve().parents[2]
        / "flagquantum/simulation/triton_kernels/statevector_gates.py"
    )
    spec = importlib.util.spec_from_file_location("_test_statevector_gates", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _matrix_forward(
    state: torch.Tensor, matrix: torch.Tensor, wire: int, n_wires: int
) -> torch.Tensor:
    assert (wire, n_wires) == (0, 1)
    return torch.matmul(matrix, state.unsqueeze(-1)).squeeze(-1)


def _matrix_backward(
    state: torch.Tensor,
    gradient: torch.Tensor,
    matrix: torch.Tensor,
    wire: int,
    n_wires: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert (wire, n_wires) == (0, 1)
    state_gradient = torch.matmul(matrix.mH, gradient.unsqueeze(-1)).squeeze(-1)
    matrix_gradient = gradient.unsqueeze(-1) * state.conj().unsqueeze(-2)
    if matrix.ndim == 2:
        matrix_gradient = matrix_gradient.sum(dim=0)
    return state_gradient, matrix_gradient


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_matrix_wrapper_preserves_state_and_matrix_gradients(
    gate_module: ModuleType, batched: bool, dtype: torch.dtype
) -> None:
    gate_module._launch_single_qubit_matrix = _matrix_forward
    gate_module._launch_single_qubit_matrix_backward = _matrix_backward
    generator = torch.Generator().manual_seed(901)
    state = torch.randn(3, 2, dtype=dtype, generator=generator, requires_grad=True)
    shape = (3, 2, 2) if batched else (2, 2)
    matrix = torch.randn(*shape, dtype=dtype, generator=generator, requires_grad=True)
    actual = gate_module._SingleQubitMatrix.apply(state, matrix, 0, 1)
    expected = _matrix_forward(state, matrix, 0, 1)
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual.abs().square().sum(), (state, matrix))
    expected_grad = torch.autograd.grad(expected.abs().square().sum(), (state, matrix))
    for actual_value, expected_value in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual_value, expected_value)


def _cx_launch(
    state: torch.Tensor, controls: torch.Tensor, targets: torch.Tensor, n_wires: int
) -> torch.Tensor:
    indices = torch.arange(1 << n_wires)
    for control, target in zip(controls, targets):
        source = torch.where(
            indices.bitwise_and(control) != 0, indices ^ target, indices
        )
        state = state[:, source]
    return state


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_cx_wrapper_reverses_noncommuting_gate_sequence(
    gate_module: ModuleType, dtype: torch.dtype
) -> None:
    gate_module._launch_cx_sequence = _cx_launch
    controls = torch.tensor([4, 2])
    targets = torch.tensor([2, 1])
    generator = torch.Generator().manual_seed(902)
    state = torch.randn(2, 8, dtype=dtype, generator=generator, requires_grad=True)
    actual = gate_module._CXSequence.apply(
        state, controls, targets, controls.flip(0), targets.flip(0), 3
    )
    expected = _cx_launch(state, controls, targets, 3)
    weights = torch.arange(1, 9, dtype=torch.float64)
    actual_grad = torch.autograd.grad((actual.real * weights).sum(), state)[0]
    expected_grad = torch.autograd.grad((expected.real * weights).sum(), state)[0]
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_grad, expected_grad)
