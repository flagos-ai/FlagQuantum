"""Adjacent fermionic swaps with explicit logical-to-wire layout tracking."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from ...circuit import Circuit
from ...core.algebra.pauli import count


@dataclass(frozen=True, slots=True)
class SwapNetwork:
    """Each pair meets once; final_layout[wire] gives its logical mode."""

    circuit: Circuit
    interactions: tuple[tuple[int, int], ...]
    final_layout: tuple[int, ...]


def fermionic_swap_network(
    n_modes: int, *, initial_layout: Sequence[int] | None = None
) -> SwapNetwork:
    """Build odd-even transposition layers in JW order, preserving FSWAP signs.

    Every pair of logical modes becomes adjacent once, immediately before its
    swap. Consumers must use the returned layout when interpreting output bits.
    This routine performs swaps only; it does not insert interaction evolution.
    """
    width = count(n_modes, "n_modes", minimum=1)
    layout = (
        list(range(width))
        if initial_layout is None
        else [count(q, "logical mode") for q in initial_layout]
    )
    if sorted(layout) != list(range(width)):
        raise ValueError("initial_layout must be a permutation of all logical modes")
    circuit = Circuit(width, dtype=torch.complex128)
    gate = torch.tensor(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, -1]],
        dtype=torch.complex128,
    )
    interactions = []
    for layer in range(width):
        for left in range(layer % 2, width - 1, 2):
            interactions.append((layout[left], layout[left + 1]))
            circuit.any(left, left + 1, matrix=gate, name="fermionic_swap")
            layout[left], layout[left + 1] = layout[left + 1], layout[left]
    return SwapNetwork(circuit, tuple(interactions), tuple(layout))
