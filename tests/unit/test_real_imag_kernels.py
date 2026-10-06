import importlib.util

import pytest
import torch

import flagquantum.simulation.complex_bmm_dispatch as complex_bmm_dispatch
import flagquantum.simulation.real_imag_kernels as kernels_runtime
from flagquantum.simulation.real_imag_kernels import (
    _CANONICAL_LAYOUT_CACHE,
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


def _enforce_copy_rank_ceiling(monkeypatch, ceiling: int = 25) -> None:
    """Reproduce the CUDA copy bound that CPU torch does not enforce.

    Reshaping or making contiguous an operand that is not already in the target
    order materializes a copy through ``TensorIterator``, which refuses more
    than twenty-five dimensions. CPU torch has no such bound, so a test that
    must reach it imposes the bound for its own duration.
    """

    reshape = torch.Tensor.reshape
    contiguous = torch.Tensor.contiguous

    def guarded_reshape(self, *shape):
        if self.ndim > ceiling and not self.is_contiguous():
            raise RuntimeError("tensor has too many (>25) dims")
        return reshape(self, *shape)

    def guarded_contiguous(self, *args, **kwargs):
        if self.ndim > ceiling and not self.is_contiguous():
            raise RuntimeError("tensor has too many (>25) dims")
        return contiguous(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "reshape", guarded_reshape)
    monkeypatch.setattr(torch.Tensor, "contiguous", guarded_contiguous)


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


def _scrambled_pair(rank: int, *, seed: int = 770077):
    """Return a canonical pair whose right operand is not in the canonical order.

    ``rank`` counts the contracted labels. The right operand keeps every one of
    them and moves the first to the end, so two extent-two axes swap order and
    grouping that operand is a transposed copy rather than a view. Only the six
    leading axes carry extent two, which keeps the flat size small while the
    rank grows.
    """

    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    output = ("a", "b")
    contracted = tuple(alphabet[2 : 2 + rank])
    right_labels = (*contracted[1:], contracted[0], output[1])
    left_labels = (output[0], *contracted)
    extents = {label: (2 if index < 3 else 1) for index, label in enumerate(contracted)}
    equation = f"{''.join(left_labels)},{''.join(right_labels)}->{''.join(output)}"
    generator = torch.Generator().manual_seed(seed)
    left = torch.randn(
        [extents.get(label, 1) for label in left_labels],
        dtype=torch.complex64,
        generator=generator,
    )
    right = torch.randn(
        [extents.get(label, 1) for label in right_labels],
        dtype=torch.complex64,
        generator=generator,
    )
    return equation, left, right


def _layout_for(equation: str, left: torch.Tensor, right: torch.Tensor):
    layout = _canonical_bmm_layout(equation, left, right)
    assert layout is not None
    return layout


def test_grouping_reproduces_the_transposed_copy_when_the_rank_fits(
    monkeypatch,
) -> None:
    """The slab route must equal the single copy it replaces, slab for slab."""

    equation, left, right = _scrambled_pair(16)
    layout = _layout_for(equation, left, right)
    right_permutation, (b, _, n) = layout[1], layout[2]
    k = right.numel() // (b * n)
    shape = (b, k, n)

    assert not right.permute(right_permutation).is_contiguous()
    expected = right.permute(right_permutation).reshape(shape)

    calls = 0
    original = kernels_runtime.torch.cat

    def counted(tensors, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original(tensors, *args, **kwargs)

    monkeypatch.setattr(kernels_runtime, "_NATIVE_EINSUM_RANK_CEILING", 5)
    monkeypatch.setattr(kernels_runtime.torch, "cat", counted)
    _enforce_copy_rank_ceiling(monkeypatch, ceiling=5)
    actual = kernels_runtime._group_axes(right, right_permutation, shape)

    assert calls == 1, "a copy above the rank bound must be taken in slabs"
    torch.testing.assert_close(actual, expected)


def test_grouping_past_the_rank_bound_is_not_a_contiguous_copy(monkeypatch) -> None:
    """A tall operand whose grouping is a copy must use the slab route."""

    equation, left, right = _scrambled_pair(28)
    layout = _layout_for(equation, left, right)
    right_permutation, (b, _, n) = layout[1], layout[2]
    k = right.numel() // (b * n)
    shape = (b, k, n)
    assert right.ndim > 25
    assert not right.permute(right_permutation).is_contiguous()

    calls = 0
    original = kernels_runtime.torch.cat

    def counted(tensors, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original(tensors, *args, **kwargs)

    monkeypatch.setattr(kernels_runtime.torch, "cat", counted)
    _enforce_copy_rank_ceiling(monkeypatch)
    actual = kernels_runtime._group_axes(right, right_permutation, shape)

    assert calls == 1
    assert actual.shape == shape

    # Check the flat order against the source indices directly: the flattened
    # position f belongs to whatever source element the permutation sends there.
    flat = actual.reshape(-1)
    reordered_shape = tuple(right.shape[axis] for axis in right_permutation)
    for position in (0, 1, flat.numel() // 3, flat.numel() - 1):
        target = []
        remaining = position
        for extent in reversed(reordered_shape):
            target.append(remaining % extent)
            remaining //= extent
        target.reverse()
        source = [0] * right.ndim
        for axis, coordinate in enumerate(target):
            source[right_permutation[axis]] = coordinate
        torch.testing.assert_close(flat[position], right[tuple(source)])


def test_a_scrambled_operand_above_the_rank_bound_still_contracts(monkeypatch) -> None:
    """A wide network whose wider operand is out of order must contract exactly.

    The grouped layout is the documented replacement for native einsum above
    twenty-five dimensions. It reaches the pair only if both operands can be
    grouped, and the wider operand is the one that must be transposed.
    """

    equation, left, right = _scrambled_pair(28)
    assert left.ndim > 25 and right.ndim > 25

    reference_left = left.detach().clone().requires_grad_(True)
    reference_right = right.detach().clone().requires_grad_(True)
    reference = torch.einsum(equation, reference_left, reference_right)

    _enforce_native_einsum_rank_ceiling(monkeypatch)
    _enforce_copy_rank_ceiling(monkeypatch)
    grouped_left = left.detach().clone().requires_grad_(True)
    grouped_right = right.detach().clone().requires_grad_(True)
    actual = complex_einsum_pair(
        equation, grouped_left, grouped_right, compile_cuda=False
    )
    torch.testing.assert_close(actual, reference, atol=2e-5, rtol=2e-5)

    cotangent = torch.ones_like(reference)
    expected = torch.autograd.grad(
        reference, (reference_left, reference_right), cotangent
    )
    actual_gradients = torch.autograd.grad(
        actual, (grouped_left, grouped_right), cotangent
    )
    torch.testing.assert_close(actual_gradients[0], expected[0], atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(actual_gradients[1], expected[1], atol=3e-5, rtol=3e-5)


def test_the_fused_layout_route_groups_a_tall_scrambled_operand(monkeypatch) -> None:
    """The layout route that lowers to a fused kernel must group at any rank.

    ``_canonical_bmm_inputs`` is the entry point the dispatch consults before it
    decides between a fused kernel and the eager pair, so it has to reach the
    same rank as the eager fallback.
    """

    equation, left, right = _scrambled_pair(28)
    layout = _layout_for(equation, left, right)
    b, _, n = layout[2]
    assert left.ndim > 25 and right.ndim > 25

    _enforce_copy_rank_ceiling(monkeypatch)
    canonical = _canonical_bmm_inputs(equation, left, right)

    assert canonical is not None
    (left_matrix, right_matrix), output_shape, output_permutation = canonical
    k = right.numel() // (b * n)
    assert left_matrix.shape == (b, left.numel() // (b * k), k)
    assert right_matrix.shape == (b, k, n)
    assert output_shape == (1, 1)
    assert output_permutation is not None
