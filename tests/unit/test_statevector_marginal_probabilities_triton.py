from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")

import flagquantum.kernels.triton.statevector_measurement as statevector_measurement
from flagquantum.kernels.triton.statevector_measurement import (
    statevector_marginal_probabilities,
)

pytestmark = [pytest.mark.unit, pytest.mark.triton, pytest.mark.gpu]


def _reference(state: torch.Tensor, wires: tuple[int, ...]) -> torch.Tensor:
    n_qubits = int(state.shape[1]).bit_length() - 1
    probabilities = torch.abs(state) ** 2
    shaped = probabilities.reshape((state.shape[0],) + (2,) * n_qubits)
    unselected = tuple(wire + 1 for wire in range(n_qubits) if wire not in set(wires))
    marginal = shaped.sum(dim=unselected) if unselected else shaped
    current_order = tuple(sorted(wires))
    if wires != current_order:
        permutation = (0,) + tuple(current_order.index(wire) + 1 for wire in wires)
        marginal = marginal.permute(permutation)
    return marginal.reshape(state.shape[0], 1 << len(wires))


def test_statevector_marginal_probabilities_cpu_fallback_preserves_gradient() -> None:
    torch.manual_seed(721)
    wires = (4, 0, 2)
    state = torch.randn(3, 32, dtype=torch.complex128, requires_grad=True)
    expected_state = state.detach().clone().requires_grad_(True)
    weights = torch.randn(3, 8, dtype=torch.float64)

    actual = statevector_marginal_probabilities(state, wires)
    expected = _reference(expected_state, wires)
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(state.grad, expected_state.grad)


def test_statevector_marginal_probabilities_preserve_requested_wire_order() -> None:
    state = torch.zeros(1, 8, dtype=torch.complex64)
    state[0, 4] = 1

    torch.testing.assert_close(
        statevector_marginal_probabilities(state, (2, 0)),
        torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
    )
    torch.testing.assert_close(
        statevector_marginal_probabilities(state, (0, 2)),
        torch.tensor([[0.0, 0.0, 1.0, 0.0]]),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("shape", "wires"),
    (
        ((1, 8), ()),
        ((3, 256), (7, 0)),
        ((4, 1 << 16), (1, 8, 15)),
        ((2, 1 << 20), (19, 0, 11, 4)),
    ),
)
def test_statevector_marginal_probabilities_cuda_matches_reference(
    shape: tuple[int, int],
    wires: tuple[int, ...],
) -> None:
    torch.manual_seed(722 + shape[0] + shape[1])
    state = torch.randn(*shape, device="cuda", dtype=torch.complex64)

    actual = statevector_marginal_probabilities(state, wires)

    torch.testing.assert_close(actual, _reference(state, wires), rtol=2e-5, atol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("shape", "wires"),
    (
        ((2, 256), (0, 7)),
        ((4, 1 << 16), (12, 2, 7)),
        ((2, 1 << 20), (19, 0, 11, 4)),
    ),
)
def test_statevector_marginal_probabilities_cuda_gradient_matches_reference(
    shape: tuple[int, int],
    wires: tuple[int, ...],
) -> None:
    torch.manual_seed(723 + shape[0] + shape[1])
    state = torch.randn(*shape, device="cuda", dtype=torch.complex64).requires_grad_()
    expected_state = state.detach().clone().requires_grad_(True)
    weights = torch.randn(
        shape[0],
        1 << len(wires),
        device="cuda",
        dtype=torch.float32,
    )

    actual = statevector_marginal_probabilities(state, wires)
    expected = _reference(expected_state, wires)
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(state.grad, expected_state.grad, rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("case", ("complex128", "noncontiguous", "conjugate"))
def test_statevector_marginal_probabilities_unsupported_cuda_input_uses_fallback(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    torch.manual_seed(724)
    wires = (4, 0, 2)
    state = torch.randn(3, 32, device="cuda", dtype=torch.complex64)
    if case == "complex128":
        state = state.to(torch.complex128)
    elif case == "noncontiguous":
        state = state.transpose(0, 1).contiguous().transpose(0, 1)
    else:
        state = state.conj()

    def unexpected_launch(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("unsupported input must not launch the Triton kernel")

    monkeypatch.setattr(
        statevector_measurement,
        "_launch_marginal_forward",
        unexpected_launch,
    )

    torch.testing.assert_close(
        statevector_marginal_probabilities(state, wires),
        _reference(state, wires),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_statevector_marginal_probabilities_wide_selection_uses_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = torch.randn(1, 1 << 9, device="cuda", dtype=torch.complex64)
    wires = tuple(range(9))

    def unexpected_launch(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("wide selections must retain the reference path")

    monkeypatch.setattr(
        statevector_measurement,
        "_launch_marginal_forward",
        unexpected_launch,
    )
    torch.testing.assert_close(
        statevector_marginal_probabilities(state, wires),
        _reference(state, wires),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_statevector_marginal_probabilities_launches_flat_grids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, tuple[int, ...]] = {}

    class FakeKernel:
        def __init__(self, name: str) -> None:
            self.name = name

        def __getitem__(self, grid: tuple[int, ...]) -> object:
            observed[self.name] = grid

            def launch(*args: object, **kwargs: object) -> None:
                return None

            return launch

    monkeypatch.setattr(
        statevector_measurement,
        "_statevector_marginal_probability_small_kernel",
        FakeKernel("forward"),
    )
    monkeypatch.setattr(
        statevector_measurement,
        "_statevector_marginal_probability_partial_kernel",
        FakeKernel("partial"),
    )
    monkeypatch.setattr(
        statevector_measurement,
        "_statevector_marginal_probability_backward_kernel",
        FakeKernel("backward"),
    )
    state = torch.zeros(2, 2048, device="cuda", dtype=torch.complex64)
    statevector_measurement._launch_marginal_forward(state, 194, 194, 2)
    statevector_measurement._launch_marginal_backward(
        state,
        torch.ones(2, 4, device="cuda"),
        194,
        2,
    )

    assert observed == {"forward": (2,), "backward": (16,)}


@pytest.mark.parametrize(
    ("state", "wires", "error", "message"),
    (
        (torch.zeros(8, dtype=torch.complex64), (), ValueError, "shape"),
        (torch.zeros(1, 0, dtype=torch.complex64), (), ValueError, "positive"),
        (torch.zeros(2, 8, dtype=torch.float32), (), ValueError, "complex dtype"),
        (torch.zeros(2, 12, dtype=torch.complex64), (), ValueError, "power-of-two"),
        (torch.zeros(2, 8, dtype=torch.complex64), (3,), ValueError, "outside"),
        (torch.zeros(2, 8, dtype=torch.complex64), (1, 1), ValueError, "unique"),
        (torch.zeros(2, 8, dtype=torch.complex64), (True,), TypeError, "integer"),
        (torch.zeros(2, 8, dtype=torch.complex64), ("0",), TypeError, "integer"),
    ),
)
def test_statevector_marginal_probabilities_validates_input(
    state: torch.Tensor,
    wires: object,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        statevector_marginal_probabilities(state, wires)  # type: ignore[arg-type]
