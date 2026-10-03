from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")

import flagquantum.kernels.triton.statevector_measurement as statevector_measurement
from flagquantum.kernels.triton.statevector_measurement import (
    statevector_pauli_expectation,
)
from flagquantum.simulation.pauli import pauli_product_statevector_expectation

pytestmark = [pytest.mark.unit, pytest.mark.triton, pytest.mark.gpu]


def _reference(
    state: torch.Tensor,
    operators: tuple[tuple[int, str], ...],
) -> torch.Tensor:
    return pauli_product_statevector_expectation(
        state,
        operators,
        int(state.shape[1]).bit_length() - 1,
    )


def test_statevector_pauli_expectation_cpu_fallback_preserves_gradient() -> None:
    torch.manual_seed(711)
    operators = ((0, "X"), (2, "Y"), (4, "Z"))
    state = torch.randn(3, 32, dtype=torch.complex128, requires_grad=True)
    expected_state = state.detach().clone().requires_grad_(True)
    weights = torch.randn(3, dtype=torch.float64)

    actual = statevector_pauli_expectation(state, operators)
    expected = _reference(expected_state, operators)
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(state.grad, expected_state.grad)


def test_statevector_pauli_expectation_uses_most_significant_wire_zero() -> None:
    state = torch.zeros(1, 8, dtype=torch.complex64)
    state[0, 4] = 1

    torch.testing.assert_close(
        statevector_pauli_expectation(state, ((0, "Z"),)),
        torch.tensor([-1.0]),
    )
    torch.testing.assert_close(
        statevector_pauli_expectation(state, ((2, "Z"),)),
        torch.tensor([1.0]),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("shape", "operators"),
    (
        ((1, 8), ()),
        ((3, 256), ((0, "X"),)),
        ((4, 1 << 16), ((1, "Y"), (7, "Z"))),
        ((2, 1 << 20), ((0, "Y"), (9, "Y"), (19, "X"))),
    ),
)
def test_statevector_pauli_expectation_cuda_matches_reference(
    shape: tuple[int, int],
    operators: tuple[tuple[int, str], ...],
) -> None:
    torch.manual_seed(712 + shape[0] + shape[1])
    state = torch.randn(*shape, device="cuda", dtype=torch.complex64)

    actual = statevector_pauli_expectation(state, operators)

    torch.testing.assert_close(
        actual,
        _reference(state, operators),
        rtol=2e-5,
        atol=2e-5,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("shape", "operators"),
    (
        ((2, 256), ((0, "X"), (7, "Z"))),
        ((4, 1 << 16), ((2, "Y"),)),
        ((2, 1 << 20), ((0, "Y"), (9, "Y"), (19, "Y"))),
    ),
)
def test_statevector_pauli_expectation_cuda_gradient_matches_reference(
    shape: tuple[int, int],
    operators: tuple[tuple[int, str], ...],
) -> None:
    torch.manual_seed(713 + shape[0] + shape[1])
    state = torch.randn(*shape, device="cuda", dtype=torch.complex64).requires_grad_()
    expected_state = state.detach().clone().requires_grad_(True)
    weights = torch.randn(shape[0], device="cuda", dtype=torch.float32)

    actual = statevector_pauli_expectation(state, operators)
    expected = _reference(expected_state, operators)
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(state.grad, expected_state.grad, rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("case", ("complex128", "noncontiguous", "conjugate"))
def test_statevector_pauli_expectation_unsupported_cuda_input_uses_fallback(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    torch.manual_seed(714)
    operators = ((0, "X"), (4, "Y"))
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
        "_launch_pauli_forward",
        unexpected_launch,
    )

    torch.testing.assert_close(
        statevector_pauli_expectation(state, operators),
        _reference(state, operators),
    )


@pytest.mark.parametrize(
    ("state", "operators", "message"),
    (
        (torch.zeros(8, dtype=torch.complex64), (), "shape"),
        (torch.zeros(1, 0, dtype=torch.complex64), (), "positive"),
        (torch.zeros(2, 8, dtype=torch.float32), (), "complex dtype"),
        (torch.zeros(2, 12, dtype=torch.complex64), (), "power-of-two"),
        (torch.zeros(2, 8, dtype=torch.complex64), ((3, "X"),), "outside"),
        (
            torch.zeros(2, 8, dtype=torch.complex64),
            ((1, "X"), (1, "Z")),
            "unique",
        ),
        (torch.zeros(2, 8, dtype=torch.complex64), ((1, "H"),), "X, Y, or Z"),
    ),
)
def test_statevector_pauli_expectation_validates_input(
    state: torch.Tensor,
    operators: tuple[tuple[int, str], ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        statevector_pauli_expectation(state, operators)


@pytest.mark.parametrize(
    ("operators", "message"),
    (
        (((True, "X"),), "integer"),
        (((0, 1),), "string"),
    ),
)
def test_statevector_pauli_expectation_validates_factor_types(
    operators: object,
    message: str,
) -> None:
    state = torch.zeros(2, 8, dtype=torch.complex64)
    with pytest.raises(TypeError, match=message):
        statevector_pauli_expectation(state, operators)  # type: ignore[arg-type]
