"""JAX-independent environment, partitioning, and topology helpers."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any


def env_int(name: str, default: int = 0) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


def node_count(world_size: int, local_world_size: int) -> int:
    return max(
        1,
        (max(1, int(world_size)) + max(1, int(local_world_size)) - 1)
        // max(1, int(local_world_size)),
    )


def communication_tier(
    left_rank: int, right_rank: int, *, local_world_size: int
) -> str:
    local_world_size = max(1, int(local_world_size))
    return (
        "intra_node"
        if int(left_rank) // local_world_size == int(right_rank) // local_world_size
        else "inter_node"
    )


def split_contiguous(n_items: int, world_size: int) -> tuple[tuple[int, ...], ...]:
    world_size = max(1, int(world_size))
    return tuple(
        tuple(
            range(
                rank * int(n_items) // world_size,
                (rank + 1) * int(n_items) // world_size,
            )
        )
        for rank in range(world_size)
    )


def product_int(values: Any) -> int:
    out = 1
    for value in values:
        out *= int(value)
    return int(out)


def rank_for_qubit(qubit: int, qubit_shards: Sequence[Sequence[int]]) -> int:
    for rank, qubits in enumerate(qubit_shards):
        if int(qubit) in qubits:
            return rank
    return 0


def validate_observable_qubits(
    qubits: Sequence[int], n_qubits: int, *, message: str
) -> None:
    """Refuse an observable qubit that the executed state does not have.

    A Z sign is selected from bit ``n_qubits - 1 - qubit`` of the basis index. A
    qubit outside the state makes that bit position negative during a shift, and
    a negative shift silently selects the wrong bit instead of failing, so the
    range must be checked before any sign is computed.
    """

    for qubit in qubits:
        if not 0 <= int(qubit) < int(n_qubits):
            raise ValueError(message)
