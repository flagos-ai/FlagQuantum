from __future__ import annotations

import pytest
import torch

from flagquantum.simulation.numerics import (
    DoubleSingleComplexTensor,
    DoubleSingleTensor,
    double_single_dot,
    double_single_sum,
    split_float32,
    two_prod,
    two_sum,
)

pytestmark = pytest.mark.unit


def test_two_sum_and_two_prod_recover_binary32_residuals() -> None:
    left = torch.tensor([1e8, 1.234567, -81.25], dtype=torch.float32)
    right = torch.tensor([1.0, -0.765432, 0.03125], dtype=torch.float32)
    sum_high, sum_low = two_sum(left, right)
    product_high, product_low = two_prod(left, right)

    torch.testing.assert_close(
        sum_high.double() + sum_low.double(),
        left.double() + right.double(),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        product_high.double() + product_low.double(),
        left.double() * right.double(),
        atol=0.0,
        rtol=0.0,
    )


def test_split_uses_only_float32_words_and_reconstructs_input() -> None:
    value = torch.tensor([1.0, -3.1415927, 8191.75], dtype=torch.float32)
    high, low = split_float32(value)
    assert high.dtype == low.dtype == torch.float32
    torch.testing.assert_close(high + low, value, atol=0.0, rtol=0.0)


def test_double_single_sum_and_dot_survive_catastrophic_cancellation() -> None:
    values = torch.tensor([1e8, 1.0, -1e8], dtype=torch.float32)
    assert values.sum().item() == 0.0
    assert double_single_sum(values).to_float64().item() == 1.0

    left = torch.tensor([1e8, 1.0, 1e8], dtype=torch.float32)
    right = torch.tensor([1.0, 1.0, -1.0], dtype=torch.float32)
    products = left * right
    # Keep the naive FP32 baseline explicit; torch.dot may use a more accurate kernel.
    naive_dot = (products[0] + products[1]) + products[2]
    assert naive_dot.item() == 0.0
    assert double_single_dot(left, right).to_float64().item() == 1.0


def test_real_and_complex_pairs_match_float64_reference() -> None:
    left = DoubleSingleTensor.from_float64(
        torch.tensor([1.0 + 2**-30], dtype=torch.float64)
    )
    right = DoubleSingleTensor.from_float64(
        torch.tensor([1.0 - 2**-28], dtype=torch.float64)
    )
    torch.testing.assert_close(
        left.multiply(right).to_float64(),
        left.to_float64() * right.to_float64(),
        atol=1e-15,
        rtol=1e-15,
    )

    first = DoubleSingleComplexTensor.from_complex128(
        torch.tensor([0.7 + 0.2j], dtype=torch.complex128)
    )
    second = DoubleSingleComplexTensor.from_complex128(
        torch.tensor([-0.3 + 0.9j], dtype=torch.complex128)
    )
    torch.testing.assert_close(
        first.multiply(second).to_complex128(),
        first.to_complex128() * second.to_complex128(),
        atol=1e-14,
        rtol=1e-14,
    )


def test_composed_primitives_preserve_autograd_and_device() -> None:
    value = torch.tensor([0.5, -0.25, 0.125], dtype=torch.float32, requires_grad=True)
    weights = torch.tensor([0.3, -0.7, 1.1], dtype=torch.float32)
    result = double_single_dot(value, weights)
    result.to_float64().backward()
    assert value.grad is not None
    torch.testing.assert_close(value.grad, weights, atol=2e-6, rtol=2e-6)
    assert result.high.device == value.device
    assert result.low.device == value.device


def test_invalid_dtype_shape_and_empty_reduction_fail_closed() -> None:
    with pytest.raises(TypeError, match="float32"):
        DoubleSingleTensor.from_float32(torch.ones(2, dtype=torch.float64))
    with pytest.raises(ValueError, match="shapes differ"):
        two_sum(torch.ones(2), torch.ones(3))
    with pytest.raises(ValueError, match="non-empty"):
        double_single_sum(torch.empty(0, dtype=torch.float32))
    with pytest.raises(ValueError, match="finite"):
        split_float32(torch.tensor([float("inf")], dtype=torch.float32))
    with pytest.raises(ValueError, match="non-overflowing"):
        split_float32(torch.tensor([torch.finfo(torch.float32).max]))
