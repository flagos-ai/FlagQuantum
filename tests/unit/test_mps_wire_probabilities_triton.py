from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")

import flagquantum.kernels.triton.mps_wire_probabilities as mps_wire_probabilities
from flagquantum.kernels.triton.mps_wire_probabilities import (
    fused_mps_qubit_probabilities,
)

pytestmark = [pytest.mark.unit, pytest.mark.triton, pytest.mark.gpu]


def _reference(tensor: torch.Tensor) -> torch.Tensor:
    probabilities = torch.sum(torch.abs(tensor) ** 2, dim=(1, 3))
    probabilities = torch.clamp(probabilities, min=0)
    return probabilities / torch.clamp(
        probabilities.sum(dim=-1, keepdim=True),
        min=1.0e-12,
    )


def test_fused_mps_wire_probabilities_cpu_fallback_matches_reference() -> None:
    torch.manual_seed(103)
    tensor = torch.randn(3, 5, 2, 7, dtype=torch.complex128)

    actual = fused_mps_qubit_probabilities(tensor)

    torch.testing.assert_close(actual, _reference(tensor))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("batch", "left_dim", "right_dim"),
    ((1, 1, 1), (1, 4, 8), (8, 16, 16), (32, 32, 32), (4, 64, 64)),
)
def test_fused_mps_wire_probabilities_cuda_matches_reference(
    batch: int,
    left_dim: int,
    right_dim: int,
) -> None:
    torch.manual_seed(104 + batch + left_dim + right_dim)
    tensor = torch.randn(
        batch,
        left_dim,
        2,
        right_dim,
        device="cuda",
        dtype=torch.complex64,
    )

    actual = fused_mps_qubit_probabilities(tensor)

    torch.testing.assert_close(actual, _reference(tensor), rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(
        actual.sum(dim=-1),
        torch.ones(batch, device="cuda"),
        rtol=2e-5,
        atol=2e-6,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("case", ("noncontiguous", "requires_grad", "large"))
def test_fused_mps_wire_probabilities_unsupported_input_uses_fallback(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    torch.manual_seed(105)
    if case == "large":
        tensor = torch.randn(1, 65, 2, 64, device="cuda", dtype=torch.complex64)
    else:
        tensor = torch.randn(2, 8, 2, 16, device="cuda", dtype=torch.complex64)
        if case == "noncontiguous":
            tensor = tensor.transpose(1, 3)
        else:
            tensor.requires_grad_(True)

    def unexpected_launch(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("unsupported input must not launch the Triton kernel")

    monkeypatch.setattr(mps_wire_probabilities, "_launch", unexpected_launch)
    actual = fused_mps_qubit_probabilities(tensor)

    torch.testing.assert_close(actual, _reference(tensor))
    if case == "requires_grad":
        actual.sum().backward()
        assert tensor.grad is not None


@pytest.mark.parametrize(
    ("shape", "dtype", "message"),
    (
        ((2, 3, 3), torch.complex64, "shape"),
        ((2, 3, 3, 4), torch.complex64, "shape"),
        ((2, 0, 2, 4), torch.complex64, "positive"),
        ((2, 3, 2, 4), torch.float32, "complex dtype"),
    ),
)
def test_fused_mps_wire_probabilities_validates_input(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    message: str,
) -> None:
    tensor = torch.zeros(shape, dtype=dtype)

    with pytest.raises(ValueError, match=message):
        fused_mps_qubit_probabilities(tensor)
