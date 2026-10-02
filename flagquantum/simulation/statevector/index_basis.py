"""Basis-index expansion shared by every statevector execution path.

A gate addresses a small group of basis vectors, and a rank-local shard addresses
them through an offset table derived from the wire positions. The expansions here
are the single source of truth for that arithmetic, so a kernel, a shard, and the
sweep that drives them all index the same amplitudes.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


def _zero_basis_local_indices(
    compressed_start: int,
    compressed_end: int,
    wires: Sequence[int],
    *,
    n_wires: int,
    rank_bits: int,
    device: torch.device,
) -> torch.Tensor:
    """Expand compressed indices with zero bits at the selected local wires."""

    positions = sorted(n_wires - int(wire) - 1 - rank_bits for wire in wires)
    indices = torch.arange(
        compressed_start, compressed_end, dtype=torch.long, device=device
    )
    for position in positions:
        low_mask = (1 << position) - 1
        low = indices & low_mask
        indices = ((indices - low) << 1) | low
    return indices


def _basis_indices_for_wires(
    global_indices: torch.Tensor, *, n_wires: int, wires: Sequence[int]
) -> torch.Tensor:
    """Extract the selected wire bits as compact basis indices."""

    wires = tuple(int(wire) for wire in wires)
    basis = torch.zeros_like(global_indices, dtype=torch.long)
    for position, wire in enumerate(wires):
        bit = (global_indices >> (n_wires - wire - 1)) & 1
        basis |= bit.to(dtype=torch.long) << (len(wires) - position - 1)
    return basis


def _wire_mask(n_wires: int, wire: int) -> int:
    """Return the global statevector bit mask for one logical wire."""

    return 1 << (int(n_wires) - int(wire) - 1)


def _basis_offset(n_wires: int, wires: Sequence[int], basis_index: int) -> int:
    """Expand a compact gate-basis index into a global statevector offset."""

    wires = tuple(int(wire) for wire in wires)
    offset = 0
    for position, wire in enumerate(wires):
        if (int(basis_index) >> (len(wires) - position - 1)) & 1:
            offset |= _wire_mask(n_wires, wire)
    return offset


def _basis_offsets_tensor(
    n_wires: int,
    wires: Sequence[int],
    *,
    rank_bits: int,
    device: torch.device,
) -> torch.Tensor:
    """The same expansion as :func:`_basis_offset`, built on the target device.

    The per-gate offset table is two or four small integers, and it was built by
    listing them in Python and copying the list across. That copy is a real
    host-to-device transfer issued once per local gate per execution, in the same
    stream as the amplitudes it indexes, so an accelerator run staged a table it
    could have computed. Every input here is already a Python integer, so the
    expansion is expressed once as device tensor arithmetic and nothing crosses
    the host boundary.
    """

    wires = tuple(int(wire) for wire in wires)
    gate_dim = 1 << len(wires)
    basis = torch.arange(gate_dim, dtype=torch.long, device=device)
    offsets = torch.zeros(gate_dim, dtype=torch.long, device=device)
    for position, wire in enumerate(wires):
        bit = (basis >> (len(wires) - position - 1)) & 1
        offsets = offsets | (bit << (int(n_wires) - wire - 1))
    return offsets >> int(rank_bits)
