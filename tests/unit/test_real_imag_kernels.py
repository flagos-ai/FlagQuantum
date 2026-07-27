import importlib.util

import pytest
import torch

import flagquantum.simulation.real_imag_kernels as kernels_runtime
from flagquantum.simulation.real_imag_kernels import (
    _CANONICAL_LAYOUT_CACHE,
    _bmm_real_imag_eager,
    _canonical_bmm_inputs,
    _canonical_bmm_layout,
    _canonical_layout_requires_materialization,
    complex_einsum_pair,
    kernel_cache_summary,
)

if importlib.util.find_spec("triton") is not None:
    import flagquantum.simulation.triton_complex_bmm as triton_bmm_runtime
else:
    triton_bmm_runtime = None


def test_canonical_layout_detects_real_reshape_copy_without_materializing() -> None:
    view_left = torch.randn(2, 3, 4, dtype=torch.complex64)
    view_right = torch.randn(2, 4, 5, dtype=torch.complex64)
    view_layout = _canonical_bmm_layout("zab,zbc->zac", view_left, view_right)
    assert view_layout is not None
    assert not _canonical_layout_requires_materialization(
        view_left, view_right, view_layout
    )

    copied_left = torch.randn(3, 2, 4, 5, dtype=torch.complex64)
    copied_right = torch.randn(4, 2, 6, 5, dtype=torch.complex64)
    copied_layout = _canonical_bmm_layout("azcb,czdb->zad", copied_left, copied_right)
    assert copied_layout is not None
    assert _canonical_layout_requires_materialization(
        copied_left, copied_right, copied_layout
    )


@pytest.mark.parametrize(
    ("equation", "left_shape", "right_shape"),
    [
        ("ab,bc->ac", (3, 4), (4, 5)),
        ("zab,zbc->zac", (2, 3, 4), (2, 4, 5)),
        ("abc,cbd->da", (2, 3, 4), (4, 3, 5)),
    ],
)
def test_canonical_bmm_lowering_matches_complex_einsum(
    equation: str,
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
) -> None:
    left = torch.randn(left_shape, dtype=torch.complex64)
    right = torch.randn(right_shape, dtype=torch.complex64)

    canonical = _canonical_bmm_inputs(equation, left, right)

    assert canonical is not None
    (left_matrix, right_matrix), output_shape, output_permutation = canonical
    left_parts = torch.view_as_real(left_matrix)
    right_parts = torch.view_as_real(right_matrix)
    pair = _bmm_real_imag_eager(
        left_parts[..., 0],
        left_parts[..., 1],
        right_parts[..., 0],
        right_parts[..., 1],
    )
    actual = torch.view_as_complex(pair).reshape(output_shape)
    if output_permutation != tuple(range(len(output_permutation))):
        actual = actual.permute(output_permutation)

    torch.testing.assert_close(actual, torch.einsum(equation, left, right))


def test_canonical_bmm_layout_is_cached_by_equation_and_shape() -> None:
    _CANONICAL_LAYOUT_CACHE.clear()
    left = torch.randn(2, 3, 4, dtype=torch.complex64)
    right = torch.randn(2, 4, 5, dtype=torch.complex64)

    first = _canonical_bmm_inputs("zab,zbc->zac", left, right)
    cache_size = len(_CANONICAL_LAYOUT_CACHE)
    second = _canonical_bmm_inputs("zab,zbc->zac", left.clone(), right.clone())

    assert first is not None and second is not None
    assert cache_size == 1
    assert len(_CANONICAL_LAYOUT_CACHE) == cache_size
    assert kernel_cache_summary()["canonical_layouts"] == 1


def test_canonical_bmm_preserves_view_when_layout_allows_it() -> None:
    left = torch.randn(2, 3, 4, dtype=torch.complex64)
    right = torch.randn(2, 4, 5, dtype=torch.complex64)

    canonical = _canonical_bmm_inputs("zab,zbc->zac", left, right)

    assert canonical is not None
    (left_matrix, right_matrix), _, _ = canonical
    assert left_matrix.untyped_storage().data_ptr() == left.untyped_storage().data_ptr()
    assert (
        right_matrix.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("equation", "left_shape", "right_shape"),
    [
        ("zab,zbc->zac", (2, 3, 4), (2, 4, 5)),
        ("abc,cbd->da", (2, 3, 4), (4, 3, 5)),
    ],
)
def test_canonical_cuda_lowering_forward_and_backward_matches_einsum(
    equation: str,
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
) -> None:
    left = torch.randn(
        left_shape, device="cuda", dtype=torch.complex64, requires_grad=True
    )
    right = torch.randn(
        right_shape, device="cuda", dtype=torch.complex64, requires_grad=True
    )
    reference_left = left.detach().clone().requires_grad_(True)
    reference_right = right.detach().clone().requires_grad_(True)

    actual = complex_einsum_pair(equation, left, right, compile_cuda=False)
    reference = torch.einsum(equation, reference_left, reference_right)
    gradient = torch.randn_like(reference)
    actual_gradients = torch.autograd.grad(actual, (left, right), gradient)
    reference_gradients = torch.autograd.grad(
        reference, (reference_left, reference_right), gradient
    )

    torch.testing.assert_close(actual, reference, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(
        actual_gradients[0], reference_gradients[0], atol=3e-5, rtol=3e-5
    )
    torch.testing.assert_close(
        actual_gradients[1], reference_gradients[1], atol=3e-5, rtol=3e-5
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_memory_pressure_keeps_canonical_fused_bmm(monkeypatch) -> None:
    calls = 0
    original = triton_bmm_runtime.fused_complex_layout_bmm

    def counted(*args, **kwargs) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(triton_bmm_runtime, "fused_complex_layout_bmm", counted)
    monkeypatch.setattr(kernels_runtime, "_FUSED_WORKING_SET_BYTES", 0)
    left = torch.randn(2, 3, 4, device="cuda", dtype=torch.complex64)
    right = torch.randn(2, 4, 5, device="cuda", dtype=torch.complex64)

    actual = complex_einsum_pair("zab,zbc->zac", left, right, compile_cuda=False)

    assert calls == 1
    torch.testing.assert_close(actual, torch.bmm(left, right), atol=2e-5, rtol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_layout_route_does_not_materialize_canonical_inputs(
    monkeypatch,
) -> None:
    monkeypatch.setattr(kernels_runtime, "_FUSED_WORKING_SET_BYTES", 0)

    def unexpected_materialization(*args, **kwargs):
        raise AssertionError("layout-aware fused route must read original tensors")

    monkeypatch.setattr(
        kernels_runtime, "_canonical_bmm_inputs", unexpected_materialization
    )

    def unexpected_rank_three_launch(*args, **kwargs):
        raise AssertionError("layout-aware backward must read original tensors")

    monkeypatch.setattr(triton_bmm_runtime, "_launch", unexpected_rank_three_launch)
    left = torch.randn(
        2, 3, 4, device="cuda", dtype=torch.complex64, requires_grad=True
    )
    right = torch.randn(
        4, 3, 5, device="cuda", dtype=torch.complex64, requires_grad=True
    )
    reference_left = left.detach().clone().requires_grad_(True)
    reference_right = right.detach().clone().requires_grad_(True)

    actual = complex_einsum_pair("abc,cbd->da", left, right, compile_cuda=False)
    reference = torch.einsum("abc,cbd->da", reference_left, reference_right)
    gradient = torch.randn_like(reference)
    actual_gradients = torch.autograd.grad(actual, (left, right), gradient)
    reference_gradients = torch.autograd.grad(
        reference, (reference_left, reference_right), gradient
    )

    torch.testing.assert_close(actual, reference, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(
        actual_gradients[0], reference_gradients[0], atol=3e-5, rtol=3e-5
    )
    torch.testing.assert_close(
        actual_gradients[1], reference_gradients[1], atol=3e-5, rtol=3e-5
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_large_nonview_inference_uses_layout_fused_bmm(monkeypatch) -> None:
    calls = 0
    original = triton_bmm_runtime.fused_complex_layout_bmm

    def counted(*args, **kwargs) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(triton_bmm_runtime, "fused_complex_layout_bmm", counted)
    left = torch.randn(64, 16, 32, 16, device="cuda", dtype=torch.complex64)
    right = torch.randn(32, 16, 64, 16, device="cuda", dtype=torch.complex64)

    actual = complex_einsum_pair("azcb,czdb->zad", left, right)
    reference = torch.einsum("azcb,czdb->zad", left, right)

    assert calls == 1
    torch.testing.assert_close(actual, reference, atol=2e-4, rtol=2e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_small_training_contraction_prefers_native_einsum(monkeypatch) -> None:
    calls = 0

    def counted(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return torch.bmm(left, right)

    monkeypatch.setattr(triton_bmm_runtime, "fused_complex_bmm", counted)
    left = torch.randn(
        2, 3, 4, device="cuda", dtype=torch.complex64, requires_grad=True
    )
    right = torch.randn(
        2, 4, 5, device="cuda", dtype=torch.complex64, requires_grad=True
    )

    actual = complex_einsum_pair("zab,zbc->zac", left, right, compile_cuda=False)
    actual.sum().backward(torch.ones_like(actual.sum()))

    assert calls == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_small_inference_contraction_prefers_native_einsum(monkeypatch) -> None:
    calls = 0

    def counted(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return torch.bmm(left, right)

    monkeypatch.setattr(triton_bmm_runtime, "fused_complex_bmm", counted)

    def unexpected_materialization(*args, **kwargs):
        raise AssertionError("native route must not materialize canonical matrices")

    monkeypatch.setattr(
        kernels_runtime, "_canonical_bmm_inputs", unexpected_materialization
    )
    left = torch.randn(2, 3, 4, device="cuda", dtype=torch.complex64)
    right = torch.randn(2, 4, 5, device="cuda", dtype=torch.complex64)

    actual = complex_einsum_pair("zab,zbc->zac", left, right, compile_cuda=False)

    assert calls == 0
    torch.testing.assert_close(actual, torch.bmm(left, right))
