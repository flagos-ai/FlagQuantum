"""FP64 RY actions on both components of complex128 Torch-FL states."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

if TYPE_CHECKING:
    from ._jit import jit
else:
    from triton import jit


@jit
def _ry(
    input_ptr: tl.tensor,
    matrix_ptr: tl.tensor,
    output_ptr: tl.tensor,
    size: tl.constexpr,
    inner: tl.constexpr,
    batched: tl.constexpr,
    block: tl.constexpr,
) -> None:
    index = tl.program_id(0) * block + tl.arange(0, block)
    if size >= (1 << 29):
        index = tl.program_id(0).to(tl.int64) * block + tl.arange(0, block)
    batch = tl.program_id(1).to(tl.int64)
    low = batch * 2 * size + (index // inner) * (2 * inner) + index % inner
    high = low + inner
    a = tl.load(input_ptr + low, index < size, 0)
    b = tl.load(input_ptr + high, index < size, 0)
    matrix = matrix_ptr + tl.where(batched, batch * 4, 0)
    cosine, sine = tl.load(matrix), tl.load(matrix + 2)
    tl.store(output_ptr + low, cosine * a - sine * b, index < size)
    tl.store(output_ptr + high, sine * a + cosine * b, index < size)


@jit
def _vjp(
    gradient_ptr: tl.tensor,
    state_output_ptr: tl.tensor,
    matrix_ptr: tl.tensor,
    state_gradient_ptr: tl.tensor,
    partial_ptr: tl.tensor,
    size: tl.constexpr,
    inner: tl.constexpr,
    batched: tl.constexpr,
    compute_state: tl.constexpr,
    compute_angle: tl.constexpr,
    block: tl.constexpr,
) -> None:
    index = tl.program_id(0) * block + tl.arange(0, block)
    if size >= (1 << 29):
        index = tl.program_id(0).to(tl.int64) * block + tl.arange(0, block)
    batch = tl.program_id(1).to(tl.int64)
    low = batch * 2 * size + (index // inner) * (2 * inner) + index % inner
    high = low + inner
    g0 = tl.load(gradient_ptr + low, index < size, 0)
    g1 = tl.load(gradient_ptr + high, index < size, 0)
    if compute_state:
        matrix = matrix_ptr + tl.where(batched, batch * 4, 0)
        cosine, sine = tl.load(matrix), tl.load(matrix + 2)
        tl.store(state_gradient_ptr + low, cosine * g0 + sine * g1, index < size)
        tl.store(state_gradient_ptr + high, -sine * g0 + cosine * g1, index < size)
    if compute_angle:
        o0 = tl.load(state_output_ptr + low, index < size, 0)
        o1 = tl.load(state_output_ptr + high, index < size, 0)
        value = tl.sum((g1 * o0 - g0 * o1) * 0.5, axis=0)
        tl.store(partial_ptr + batch * tl.cdiv(size, block) + tl.program_id(0), value)


def _validate(state: torch.Tensor, matrix: torch.Tensor, wire: int, n: int) -> None:
    if state.device.type != "flagos" or state.dtype != torch.complex128:
        raise ValueError("FP64 RY requires a complex128 Torch-FL state")
    if state.ndim != 2 or state.shape[1] != 1 << n or not 0 <= wire < n:
        raise ValueError("RY state shape and wire must match the qubit count")
    if matrix.device != state.device or matrix.dtype != torch.complex128:
        raise ValueError("RY matrix must share the state device and precision")
    if matrix.shape not in ((1, 2, 2), (state.shape[0], 2, 2)):
        raise ValueError("RY matrix must be shared or match the state batch")


def forward(
    x: torch.Tensor,
    m: torch.Tensor,
    w: int,
    n: int,
    block: int = 256,
) -> torch.Tensor:
    """Apply a real RY matrix without full-state pointwise intermediates."""
    _validate(x, m, w, n)
    x = x.resolve_conj().contiguous()
    out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(1 << n, block), x.shape[0])
    _ry[grid](
        torch.view_as_real(x),
        m.real.contiguous(),
        torch.view_as_real(out),
        1 << n,
        2 << (n - w - 1),
        m.shape[0] != 1,
        block,
        enable_fp_fusion=False,
    )
    return out


def backward(
    g: torch.Tensor,
    o: torch.Tensor,
    m: torch.Tensor,
    w: int,
    n: int,
    state: bool = True,
    angle: bool = True,
    block: int = 256,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Fuse inverse rotation and FP64 partial angle-gradient reduction."""
    _validate(g, m, w, n)
    if o.shape != g.shape or o.device != g.device or o.dtype != g.dtype:
        raise ValueError("RY output and gradient must share shape, device and dtype")
    g = g.resolve_conj().contiguous()
    o = o.resolve_conj().contiguous()
    out = torch.empty_like(g) if state else g
    partial = torch.empty(
        (g.shape[0], triton.cdiv(1 << n, block)),
        device=g.device,
        dtype=torch.float64,
    )
    _vjp[(triton.cdiv(1 << n, block), g.shape[0])](
        torch.view_as_real(g),
        torch.view_as_real(o),
        m.real.contiguous(),
        torch.view_as_real(out),
        partial,
        1 << n,
        2 << (n - w - 1),
        m.shape[0] != 1,
        state,
        angle,
        block,
        enable_fp_fusion=False,
    )
    return out if state else None, partial.sum(1).reshape(-1, 1) if angle else None


__all__ = ("forward", "backward")
