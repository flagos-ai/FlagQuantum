"""Cached whole-state qubit permutations for CPU product-state execution."""

from __future__ import annotations

from collections.abc import Sequence

import torch

_WIRE_PERMUTATION_CACHE_BYTES = 128 * 1024 * 1024
_WIRE_PERMUTATION_CACHE: dict[
    tuple[tuple[int, ...], str, torch.dtype], torch.Tensor
] = {}


def _clear_wire_permutation_cache() -> None:
    _WIRE_PERMUTATION_CACHE.clear()


def _wire_permutation_index(
    source_wires: Sequence[int],
    *,
    device: torch.device,
) -> torch.Tensor:
    """Build a gather table from ascending logical qubits to current tensor axes."""

    canonical_source = tuple(int(wire) for wire in source_wires)
    if len(set(canonical_source)) != len(canonical_source):
        raise ValueError("wire permutation requires unique logical wires")
    n_wires = len(canonical_source)
    table_dtype = torch.int64
    key = (canonical_source, str(device), table_dtype)
    cached = _WIRE_PERMUTATION_CACHE.get(key)
    if cached is not None:
        return cached

    source_positions = {
        wire: position for position, wire in enumerate(canonical_source)
    }
    images = tuple(
        1 << (n_wires - 1 - source_positions[wire]) for wire in sorted(canonical_source)
    )
    table = torch.zeros(1, dtype=torch.int64, device=device)
    for image in reversed(images):
        table = torch.cat((table, table ^ image))
    table = table.to(table_dtype)
    _WIRE_PERMUTATION_CACHE[key] = table
    while (
        len(_WIRE_PERMUTATION_CACHE) > 1
        and sum(
            item.numel() * item.element_size()
            for item in _WIRE_PERMUTATION_CACHE.values()
        )
        > _WIRE_PERMUTATION_CACHE_BYTES
    ):
        del _WIRE_PERMUTATION_CACHE[next(iter(_WIRE_PERMUTATION_CACHE))]
    return table


def _apply_wire_permutation_gather(
    state: torch.Tensor,
    source_wires: Sequence[int],
) -> torch.Tensor:
    """Materialize ascending logical-qubit order with one cached gather."""

    index = _wire_permutation_index(source_wires, device=state.device)
    flat_state = state.reshape(state.shape[0], -1)
    gathered = torch.gather(flat_state, 1, index.unsqueeze(0).expand_as(flat_state))
    return gathered.reshape(state.shape)
