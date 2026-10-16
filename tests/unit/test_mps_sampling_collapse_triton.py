from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")

import flagquantum.kernels.triton.mps_sampling_collapse as sampling_collapse
from flagquantum.kernels.triton.mps_sampling_collapse import (
    fused_mps_sampling_collapse,
)

pytestmark = [pytest.mark.unit, pytest.mark.triton, pytest.mark.gpu]


def _inputs(
    batch: int,
    right_dim: int,
    next_right_dim: int,
    *,
    device: str,
    dtype: torch.dtype = torch.complex64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(
        260_000 + batch + right_dim + next_right_dim
    )
    site = torch.randn(
        batch,
        1,
        2,
        right_dim,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    next_site = torch.randn(
        batch,
        right_dim,
        2,
        next_right_dim,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    bits = torch.arange(batch, device=device, dtype=torch.int64) % 2
    return site, next_site, bits


def _reference(
    site: torch.Tensor,
    next_site: torch.Tensor,
    bits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batches = torch.arange(site.shape[0], device=site.device)
    boundary = site[batches, 0, bits, :]
    boundary = boundary / torch.linalg.vector_norm(boundary, dim=-1, keepdim=True)
    collapsed = torch.zeros(
        site.shape[0], 1, 2, 1, dtype=site.dtype, device=site.device
    )
    collapsed[batches, 0, bits, 0] = 1
    propagated = torch.einsum("bl,blsr->bsr", boundary, next_site).unsqueeze(1)
    return collapsed, propagated


def test_fused_mps_sampling_collapse_cpu_fallback_matches_reference() -> None:
    site, next_site, bits = _inputs(3, 5, 7, device="cpu", dtype=torch.complex128)

    actual = fused_mps_sampling_collapse(site, next_site, bits)
    expected = _reference(site, next_site, bits)

    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("batch", "right_dim", "next_right_dim"),
    ((1, 1, 1), (7, 4, 8), (32, 16, 16), (128, 32, 64), (64, 64, 32)),
)
def test_fused_mps_sampling_collapse_cuda_matches_reference(
    batch: int,
    right_dim: int,
    next_right_dim: int,
) -> None:
    site, next_site, bits = _inputs(batch, right_dim, next_right_dim, device="cuda")

    actual = fused_mps_sampling_collapse(site, next_site, bits)
    expected = _reference(site, next_site, bits)

    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=2e-5, atol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("case", ("noncontiguous", "requires_grad", "large"))
def test_fused_mps_sampling_collapse_unsupported_input_uses_fallback(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    site, next_site, bits = _inputs(4, 8, 8, device="cuda")
    if case == "noncontiguous":
        next_site = next_site.transpose(1, 3)
        site = torch.randn(4, 1, 2, 8, device="cuda", dtype=torch.complex64)
        next_site = torch.randn(4, 8, 2, 9, device="cuda", dtype=torch.complex64)[
            ..., :8
        ]
    elif case == "requires_grad":
        site.requires_grad_(True)
        next_site.requires_grad_(True)
    else:
        site, next_site, bits = _inputs(2, 65, 4, device="cuda")

    def unexpected_launch(*args: object, **kwargs: object) -> object:
        raise AssertionError("unsupported input must not launch the Triton kernel")

    monkeypatch.setattr(sampling_collapse, "_launch", unexpected_launch)
    actual = fused_mps_sampling_collapse(site, next_site, bits)
    expected = _reference(site, next_site, bits)

    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])
    if case == "requires_grad":
        actual[1].real.sum().backward()
        assert site.grad is not None
        assert next_site.grad is not None


@pytest.mark.parametrize(
    ("site_shape", "next_shape", "bits", "dtype", "message"),
    (
        ((2, 2, 2, 4), (2, 4, 2, 3), (0, 1), torch.complex64, "shape"),
        ((2, 1, 2, 4), (2, 5, 2, 3), (0, 1), torch.complex64, "bonds"),
        ((2, 1, 2, 4), (2, 4, 2, 3), (0,), torch.complex64, "shape"),
        ((2, 1, 2, 4), (2, 4, 2, 3), (0, 2), torch.complex64, "zero or one"),
        ((2, 1, 2, 4), (2, 4, 2, 3), (0, 1), torch.float32, "complex dtype"),
    ),
)
def test_fused_mps_sampling_collapse_validates_input(
    site_shape: tuple[int, ...],
    next_shape: tuple[int, ...],
    bits: tuple[int, ...],
    dtype: torch.dtype,
    message: str,
) -> None:
    site = torch.ones(site_shape, dtype=dtype)
    next_site = torch.ones(next_shape, dtype=dtype)
    sampled = torch.tensor(bits, dtype=torch.int64)

    with pytest.raises(ValueError, match=message):
        fused_mps_sampling_collapse(site, next_site, sampled)
