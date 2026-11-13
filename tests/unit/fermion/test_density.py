"""Measured density assembly preserves values, conventions and gradients."""

from __future__ import annotations

import pytest
import torch

from flagquantum.algorithms.fermion.density import (
    one_body_density,
    one_body_density_observables,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_complex_assembly_order_and_gradients(dtype: torch.dtype) -> None:
    parameters = torch.tensor([0.7, 0.2, -0.3, 0.3], dtype=dtype, requires_grad=True)
    gamma = one_body_density(tuple(parameters.unbind()), n_modes=2)
    expected = torch.tensor([[0.7, 0.2 - 0.3j], [0.2 + 0.3j, 0.3]], dtype=gamma.dtype)
    torch.testing.assert_close(gamma, expected)
    gradient = torch.autograd.grad(gamma[0, 1].imag + gamma.trace().real, parameters)[0]
    torch.testing.assert_close(gradient, parameters.new_tensor([1, 0, 1, 1]))


def test_one_mode_and_runtime_singleton_shape() -> None:
    value = torch.tensor([[1.0]], dtype=torch.float64, requires_grad=True)
    gamma = one_body_density((value,), n_modes=1)
    assert gamma.shape == (1, 1)
    assert gamma.dtype == torch.complex128
    torch.testing.assert_close(
        torch.autograd.grad(gamma.real.sum(), value)[0], torch.ones_like(value)
    )


def test_unphysical_measurements_are_not_silently_projected() -> None:
    values = tuple(torch.tensor(v, dtype=torch.float64) for v in (1.4, 0.0, 0.0, -0.2))
    gamma = one_body_density(values, n_modes=2)
    torch.testing.assert_close(
        gamma.diag().real, torch.tensor([1.4, -0.2], dtype=torch.float64)
    )


@pytest.mark.parametrize("width", [0, -1, True, 1.5])
def test_invalid_width(width: int) -> None:
    with pytest.raises((TypeError, ValueError)):
        one_body_density_observables(n_modes=width)
    with pytest.raises((TypeError, ValueError)):
        one_body_density((), n_modes=width)


def test_unknown_encoding() -> None:
    with pytest.raises(ValueError, match="unsupported encoding"):
        one_body_density_observables(n_modes=2, encoding="unknown")  # type: ignore[arg-type]


@pytest.mark.parametrize("size", [0, 3, 5])
def test_wrong_measurement_count(size: int) -> None:
    with pytest.raises(ValueError, match="n_modes"):
        one_body_density([torch.tensor(0.0)] * size, n_modes=2)


@pytest.mark.parametrize(
    "value,error,message",
    [
        (0.0, TypeError, "real float"),
        (torch.tensor(1), TypeError, "real float"),
        (torch.tensor(True), TypeError, "real float"),
        (torch.tensor(0j), TypeError, "real float"),
        (torch.tensor(0.0, dtype=torch.float16), TypeError, "real float"),
        (torch.zeros(2), ValueError, "single-element"),
        (torch.empty(0), ValueError, "single-element"),
        (torch.tensor(float("nan")), ValueError, "finite"),
        (torch.tensor(float("inf")), ValueError, "finite"),
        (torch.empty((), device="meta"), ValueError, "CPU"),
    ],
)
def test_malformed_measurement(
    value: torch.Tensor, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        one_body_density((value,), n_modes=1)


def test_mixed_precision_is_explicitly_refused() -> None:
    values = [torch.tensor(0.0, dtype=torch.float64)] * 3 + [torch.tensor(0.0)]
    with pytest.raises(ValueError, match="one dtype"):
        one_body_density(values, n_modes=2)
