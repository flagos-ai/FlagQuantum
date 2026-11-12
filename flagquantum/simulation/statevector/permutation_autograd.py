"""Compact exact CX permutations with inverse-gather differentiation."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Sequence
from typing import Protocol

import torch


class _Context(Protocol):
    layout: tuple[int, int, int, int]

    @property
    def saved_tensors(self) -> tuple[torch.Tensor, ...]: ...

    @property
    def needs_input_grad(self) -> tuple[bool, ...]: ...

    def save_for_backward(self, *tensors: torch.Tensor) -> None: ...


_INDEX_CACHE: OrderedDict[tuple[object, ...], torch.Tensor] = OrderedDict()
_INDEX_CACHE_LIMIT = 3 * (1 << 30)


def _full_index(
    controls: tuple[int, ...],
    targets: tuple[int, ...],
    n_wires: int,
    state: torch.Tensor,
) -> torch.Tensor:
    from .operations import _cx_sequence_permutation_index

    key = (controls, targets, n_wires, str(state.device))
    if key in _INDEX_CACHE:
        _INDEX_CACHE.move_to_end(key)
        return _INDEX_CACHE[key]
    result = _cx_sequence_permutation_index(
        controls,
        targets,
        n_wires,
        device=state.device,
        dtype=state.dtype,
    )
    result_bytes = result.numel() * result.element_size()
    if result_bytes > _INDEX_CACHE_LIMIT:
        return result
    cached_bytes = sum(
        item.numel() * item.element_size() for item in _INDEX_CACHE.values()
    )
    while _INDEX_CACHE and cached_bytes + result_bytes > _INDEX_CACHE_LIMIT:
        _, evicted = _INDEX_CACHE.popitem(last=False)
        cached_bytes -= evicted.numel() * evicted.element_size()
    _INDEX_CACHE[key] = result
    return result


class _CXPermutation(torch.autograd.Function):
    """A bijection has an inverse gather instead of a scatter-add gradient."""

    @staticmethod
    def forward(
        ctx: _Context,
        state: torch.Tensor,
        forward: torch.Tensor,
        reverse: torch.Tensor,
        outer: int,
        width: int,
        inner: int,
    ) -> torch.Tensor:
        ctx.save_for_backward(forward, reverse)
        ctx.layout = (state.shape[0], outer, width, inner)
        return state.reshape(ctx.layout).index_select(2, forward).reshape_as(state)

    @staticmethod
    def backward(
        ctx: _Context,
        gradient: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None, None, None]:
        _, reverse = ctx.saved_tensors
        result = gradient.reshape(ctx.layout).index_select(2, reverse)
        return result.reshape(gradient.shape), None, None, None, None, None


def apply_compact_cx_sequence(
    state: torch.Tensor,
    controls: Sequence[int],
    targets: Sequence[int],
    n_wires: int,
) -> torch.Tensor:
    """Gather only the participating contiguous wire interval, not global indices."""
    from .operations import _cx_sequence_permutation_index

    first, last = min((*controls, *targets)), max((*controls, *targets))
    width = last - first + 1
    if first >= 8:
        forward = _full_index(tuple(controls), tuple(targets), n_wires, state)
        reverse = _full_index(
            tuple(controls)[::-1], tuple(targets)[::-1], n_wires, state
        )
        return _apply_cx_permutation(state, forward, reverse, 1, 1 << n_wires, 1)
    local_controls = tuple(wire - first for wire in controls)
    local_targets = tuple(wire - first for wire in targets)
    forward = _cx_sequence_permutation_index(
        local_controls,
        local_targets,
        width,
        device=state.device,
        dtype=state.dtype,
    )
    reverse = _cx_sequence_permutation_index(
        local_controls[::-1],
        local_targets[::-1],
        width,
        device=state.device,
        dtype=state.dtype,
    )
    outer, inner = 1 << first, 1 << (n_wires - last - 1)
    return _apply_cx_permutation(state, forward, reverse, outer, 1 << width, inner)


def _apply_cx_permutation(*args: object) -> torch.Tensor:
    apply: Callable[..., object] = _CXPermutation.apply
    result = apply(*args)
    if not isinstance(result, torch.Tensor):
        raise TypeError("autograd action must return a tensor")
    return result
