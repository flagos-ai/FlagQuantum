import importlib.util

import pytest
import torch

import flagquantum.simulation.complex_bmm_dispatch as complex_bmm_dispatch
import flagquantum.simulation.real_imag_kernels as kernels_runtime
from flagquantum.simulation.real_imag_kernels import (
    _CANONICAL_LAYOUT_CACHE,
    _LAYOUT_MATERIALIZATION_CACHE,
    _bmm_real_imag_eager,
    _canonical_bmm_inputs,
    _canonical_bmm_layout,
    _canonical_layout_requires_materialization,
    _fused_layout_bmm,
    _layout_bmm_dispatch_supported,
    complex_einsum_pair,
    kernel_cache_summary,
)

pytestmark = pytest.mark.unit

if importlib.util.find_spec("triton") is not None:
    import flagquantum.kernels.triton.complex_bmm as triton_bmm_runtime
else:
    triton_bmm_runtime = None


def test_canonical_layout_detects_real_reshape_copy_without_materializing() -> None:
    _LAYOUT_MATERIALIZATION_CACHE.clear()
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
    assert len(_LAYOUT_MATERIALIZATION_CACHE) == 2
    assert kernel_cache_summary()["materialization_decisions"] == 2


def test_materialization_cache_distinguishes_tensor_strides() -> None:
    _LAYOUT_MATERIALIZATION_CACHE.clear()
    left = torch.randn(2, 3, 4, dtype=torch.complex64)
    right = torch.randn(2, 4, 5, dtype=torch.complex64)
    layout = _canonical_bmm_layout("zab,zbc->zac", left, right)
    assert layout is not None

    assert not _canonical_layout_requires_materialization(left, right, layout)
    assert not _canonical_layout_requires_materialization(left, right, layout)
    assert len(_LAYOUT_MATERIALIZATION_CACHE) == 1

    transposed_left = left.transpose(1, 2)
    transposed_right = torch.randn(2, 3, 5, dtype=torch.complex64)
    transposed_layout = _canonical_bmm_layout(
        "zab,zbc->zac", transposed_left, transposed_right
    )
    assert transposed_layout is not None
    _canonical_layout_requires_materialization(
        transposed_left, transposed_right, transposed_layout
    )
    assert len(_LAYOUT_MATERIALIZATION_CACHE) == 2


def test_layout_bmm_dispatch_is_limited_to_evidenced_forward_shape() -> None:
    winning_left = torch.empty(64, 16, 64, 16, device="meta", dtype=torch.complex64)
    winning_right = torch.empty(64, 16, 64, 16, device="meta", dtype=torch.complex64)
    winning_layout = _canonical_bmm_layout(
        "azcb,czdb->zad", winning_left, winning_right
    )
    assert winning_layout is not None
    assert _layout_bmm_dispatch_supported(
        "azcb,czdb->zad", winning_left, winning_right, winning_layout
    )

    losing_left = torch.empty(64, 16, 32, 16, device="meta", dtype=torch.complex64)
    losing_right = torch.empty(32, 16, 64, 16, device="meta", dtype=torch.complex64)
    losing_layout = _canonical_bmm_layout("azcb,czdb->zad", losing_left, losing_right)
    assert losing_layout is not None
    assert not _layout_bmm_dispatch_supported(
        "azcb,czdb->zad", losing_left, losing_right, losing_layout
    )

    assert not _layout_bmm_dispatch_supported(
        "azcb,czdb->zad",
        winning_left.requires_grad_(True),
        winning_right,
        winning_layout,
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
@pytest.mark.gpu
@pytest.mark.triton
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
@pytest.mark.gpu
@pytest.mark.triton
def test_evidenced_nonview_inference_uses_layout_fused_bmm(monkeypatch) -> None:
    calls = 0
    original = triton_bmm_runtime.fused_complex_layout_bmm

    def counted(*args, **kwargs) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(triton_bmm_runtime, "fused_complex_layout_bmm", counted)
    left = torch.randn(64, 16, 64, 16, device="cuda", dtype=torch.complex64)
    right = torch.randn(64, 16, 64, 16, device="cuda", dtype=torch.complex64)

    actual = complex_einsum_pair("azcb,czdb->zad", left, right, compile_cuda=False)
    reference = torch.einsum("azcb,czdb->zad", left, right)

    assert calls == 1
    torch.testing.assert_close(actual, reference, atol=2e-4, rtol=2e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.gpu
@pytest.mark.triton
def test_fused_layout_route_does_not_materialize_canonical_inputs(
    monkeypatch,
) -> None:
    catalog_routes = []
    require_cataloged_kernel = complex_bmm_dispatch._require_layout_complex_bmm_kernel

    def capture_catalog_route(*, device_type, dtype):
        implementation = require_cataloged_kernel(
            device_type=device_type,
            dtype=dtype,
        )
        catalog_routes.append(implementation.implementation_id)
        return implementation

    monkeypatch.setattr(
        complex_bmm_dispatch,
        "_require_layout_complex_bmm_kernel",
        capture_catalog_route,
    )

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

    layout = _canonical_bmm_layout("abc,cbd->da", left, right)
    assert layout is not None
    actual = _fused_layout_bmm(left, right, layout)
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
    assert catalog_routes == ["FQKI-TRITON-NUM-002-A"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.gpu
@pytest.mark.triton
def test_unevidenced_nonview_inference_prefers_native_einsum(monkeypatch) -> None:
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

    assert calls == 0
    torch.testing.assert_close(actual, reference, atol=2e-4, rtol=2e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.gpu
@pytest.mark.triton
def test_evidenced_training_contraction_prefers_native_einsum(monkeypatch) -> None:
    calls = 0

    def counted(*args, **kwargs) -> torch.Tensor:
        nonlocal calls
        calls += 1
        raise AssertionError("training must not use the cataloged layout kernel")

    monkeypatch.setattr(triton_bmm_runtime, "fused_complex_layout_bmm", counted)
    left = torch.randn(
        64, 16, 64, 16, device="cuda", dtype=torch.complex64, requires_grad=True
    )
    right = torch.randn(
        64, 16, 64, 16, device="cuda", dtype=torch.complex64, requires_grad=True
    )

    actual = complex_einsum_pair("azcb,czdb->zad", left, right, compile_cuda=False)
    actual.sum().backward(torch.ones_like(actual.sum()))

    assert calls == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.gpu
@pytest.mark.triton
def test_small_inference_contraction_prefers_native_einsum(monkeypatch) -> None:
    calls = 0

    def counted(*args, **kwargs) -> torch.Tensor:
        nonlocal calls
        calls += 1
        raise AssertionError("unevidenced shape must not use the layout kernel")

    monkeypatch.setattr(triton_bmm_runtime, "fused_complex_layout_bmm", counted)

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


def _tall_rank_pair(contracted_axes: int, *, seed: int = 440044):
    """Return a canonical pair whose operands exceed the native einsum ceiling.

    The labels make the layout canonical -- one free axis on each side and every
    remaining axis contracted -- and keep the grouped contraction small: three
    contracted axes carry extent two and the rest are singletons, so the matmul
    working set is ``2 x 8`` against ``8 x 2``.
    """

    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    output = ("a", "b")
    contracted = tuple(alphabet[2 : 2 + contracted_axes])
    left_labels = (output[0], *contracted)
    right_labels = (*contracted, output[1])
    equation = f"{''.join(left_labels)},{''.join(right_labels)}->{''.join(output)}"
    extents = {label: (2 if index < 3 else 1) for index, label in enumerate(contracted)}
    generator = torch.Generator().manual_seed(seed)
    left = torch.randn(
        [extents.get(label, 2) for label in left_labels],
        dtype=torch.complex64,
        generator=generator,
    )
    right = torch.randn(
        [extents.get(label, 2) for label in right_labels],
        dtype=torch.complex64,
        generator=generator,
    )
    return equation, left, right


def _enforce_native_einsum_rank_ceiling(monkeypatch, ceiling: int = 25) -> None:
    """Reproduce the CUDA einsum ceiling that CPU torch does not enforce."""

    native = torch.einsum

    def guarded(equation, *operands, **kwargs):
        if any(operand.ndim > ceiling for operand in operands):
            raise RuntimeError("tensor has too many (>25) dims")
        return native(equation, *operands, **kwargs)

    monkeypatch.setattr(kernels_runtime.torch, "einsum", guarded)


def test_a_tall_rank_contraction_is_reachable_through_the_grouped_layout(
    monkeypatch,
) -> None:
    """A wide sliced network contracts above the rank native einsum accepts.

    The fused layout kernel refuses more than eight axes per group and the
    documented fallback was a native einsum, which refuses more than twenty-five
    dimensions. The grouped layout is exact arithmetic on the axes the two
    operands already share, so it replaces that dead end instead of raising.
    """

    equation, left, right = _tall_rank_pair(28)
    assert left.ndim > 25 and right.ndim > 25

    _enforce_native_einsum_rank_ceiling(monkeypatch)
    with pytest.raises(RuntimeError, match="too many"):
        torch.einsum(equation, left, right)

    grouped_left = left.detach().clone().requires_grad_(True)
    grouped_right = right.detach().clone().requires_grad_(True)
    actual = complex_einsum_pair(
        equation, grouped_left, grouped_right, compile_cuda=False
    )

    reference_left = left.detach().clone().requires_grad_(True)
    reference_right = right.detach().clone().requires_grad_(True)
    reference = torch.einsum(
        "ak,kb->ab",
        reference_left.reshape(2, 8),
        reference_right.reshape(8, 2),
    )
    torch.testing.assert_close(actual, reference, atol=2e-5, rtol=2e-5)

    cotangent = torch.randn_like(reference)
    actual_gradients = torch.autograd.grad(
        actual, (grouped_left, grouped_right), cotangent
    )
    reference_gradients = torch.autograd.grad(
        reference, (reference_left, reference_right), cotangent
    )
    torch.testing.assert_close(
        actual_gradients[0], reference_gradients[0], atol=3e-5, rtol=3e-5
    )
    torch.testing.assert_close(
        actual_gradients[1], reference_gradients[1], atol=3e-5, rtol=3e-5
    )


def test_a_contraction_the_native_route_can_express_still_uses_it(
    monkeypatch,
) -> None:
    """The rank guard must not divert a contraction native einsum handles."""

    _enforce_native_einsum_rank_ceiling(monkeypatch)

    def unexpected(*args, **kwargs):
        raise AssertionError("a rank-safe contraction must stay on native einsum")

    monkeypatch.setattr(kernels_runtime, "_canonical_layout_pair", unexpected)
    left = torch.randn(2, 3, 4, dtype=torch.complex64)
    right = torch.randn(2, 4, 5, dtype=torch.complex64)

    actual = complex_einsum_pair("zab,zbc->zac", left, right, compile_cuda=False)

    torch.testing.assert_close(actual, torch.bmm(left, right), atol=2e-5, rtol=2e-5)
