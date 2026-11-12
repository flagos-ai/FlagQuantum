"""Exact CX source-address generation without a full-state permutation table."""

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
def _cx(
    input_ptr: tl.tensor,
    output_ptr: tl.tensor,
    size: tl.constexpr,
    controls: tl.constexpr,
    targets: tl.constexpr,
    block: tl.constexpr,
) -> None:
    index = tl.program_id(0) * block + tl.arange(0, block)
    if size >= (1 << 29):
        index = tl.program_id(0).to(tl.int64) * block + tl.arange(0, block)
    batch = tl.program_id(1).to(tl.int64)
    source = index // 2
    for gate in tl.static_range(len(controls)):
        source = source ^ (((source >> controls[gate]) & 1) << targets[gate])
    value = tl.load(
        input_ptr + batch * 2 * size + source * 2 + index % 2, index < 2 * size, 0
    )
    tl.store(output_ptr + batch * 2 * size + index, value, index < 2 * size)


def apply_cx(
    state: torch.Tensor,
    controls: tuple[int, ...],
    targets: tuple[int, ...],
    n_wires: int,
) -> torch.Tensor:
    """Apply a CX sequence by mapping each destination to its exact source."""
    if state.device.type != "flagos" or state.dtype != torch.complex128:
        raise ValueError("FP64 CX requires a complex128 Torch-FL state")
    if state.ndim != 2 or state.shape[1] != 1 << n_wires:
        raise ValueError("CX state shape must match its qubit count")
    if not controls or len(controls) != len(targets):
        raise ValueError("CX controls and targets must be equally sized and nonempty")
    if any(
        not 0 <= c < n_wires or not 0 <= t < n_wires or c == t
        for c, t in zip(controls, targets, strict=True)
    ):
        raise ValueError("CX wires must be distinct valid qubit indices")
    state = state.resolve_conj().contiguous()
    out = torch.empty(state.shape, device=state.device, dtype=state.dtype)
    # Source addressing traverses gates in reverse chronological order.
    c_bits = tuple(n_wires - c - 1 for c in controls[::-1])
    t_bits = tuple(n_wires - t - 1 for t in targets[::-1])
    _cx[(triton.cdiv(2 << n_wires, 1024), state.shape[0])](
        torch.view_as_real(state),
        torch.view_as_real(out),
        1 << n_wires,
        c_bits,
        t_bits,
        1024,
    )
    return out


__all__ = ("apply_cx",)
