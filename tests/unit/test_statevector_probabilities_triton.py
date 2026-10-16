from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")

import flagquantum.kernels.triton.statevector_measurement as statevector_measurement
from flagquantum.kernels.triton.statevector_measurement import (
    statevector_probabilities,
)

pytestmark = [pytest.mark.unit, pytest.mark.triton, pytest.mark.gpu]


def _reference(state: torch.Tensor) -> torch.Tensor:
    return torch.abs(state) ** 2


def test_statevector_probabilities_cpu_fallback_preserves_gradient() -> None:
    torch.manual_seed(701)
    state = torch.randn(3, 17, dtype=torch.complex128, requires_grad=True)
    expected_state = state.detach().clone().requires_grad_(True)
    weights = torch.randn(3, 17, dtype=torch.float64)

    actual = statevector_probabilities(state)
    expected = _reference(expected_state)
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(state.grad, expected_state.grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("batch", "amplitudes"),
    ((1, 2), (3, 257), (8, 1 << 16), (2, 1 << 20)),
)
def test_statevector_probabilities_cuda_matches_reference(
    batch: int,
    amplitudes: int,
) -> None:
    torch.manual_seed(702 + batch + amplitudes)
    state = torch.randn(
        batch,
        amplitudes,
        device="cuda",
        dtype=torch.complex64,
    )

    actual = statevector_probabilities(state)

    torch.testing.assert_close(actual, _reference(state), rtol=1e-6, atol=1e-7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("shape", ((1, 257), (4, 1 << 16), (2, 1 << 20)))
def test_statevector_probabilities_cuda_gradient_matches_reference(
    shape: tuple[int, int],
) -> None:
    torch.manual_seed(703 + shape[0] + shape[1])
    state = torch.randn(*shape, device="cuda", dtype=torch.complex64).requires_grad_()
    expected_state = state.detach().clone().requires_grad_(True)
    weights = torch.randn(*shape, device="cuda", dtype=torch.float32)

    actual = statevector_probabilities(state)
    expected = _reference(expected_state)
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(state.grad, expected_state.grad, rtol=1e-6, atol=1e-7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("case", ("complex128", "noncontiguous", "conjugate"))
def test_statevector_probabilities_unsupported_cuda_input_uses_fallback(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    torch.manual_seed(704)
    state = torch.randn(3, 32, device="cuda", dtype=torch.complex64)
    if case == "complex128":
        state = state.to(torch.complex128)
    elif case == "noncontiguous":
        state = state.transpose(0, 1)
    else:
        state = state.conj()

    def unexpected_launch(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("unsupported input must not launch the Triton kernel")

    monkeypatch.setattr(statevector_measurement, "_launch_forward", unexpected_launch)

    torch.testing.assert_close(statevector_probabilities(state), _reference(state))


@pytest.mark.parametrize(
    ("state", "message"),
    (
        (torch.zeros(8, dtype=torch.complex64), "shape"),
        (torch.zeros(1, 0, dtype=torch.complex64), "positive"),
        (torch.zeros(2, 8, dtype=torch.float32), "complex dtype"),
    ),
)
def test_statevector_probabilities_validates_input(
    state: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        statevector_probabilities(state)
