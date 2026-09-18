"""Torch-distributed statevector forward execution loop."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from .forward import TorchDistributedStatevectorResult
from .forward_sweep import _ShardedForwardSweep


def execute_torch_distributed_statevector(
    circuit_or_ir: Any,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.complex64,
    local_world_size: int | None = None,
    exchange_buffer_bytes: int = 512 * 1024 * 1024,
    compact_index_threshold: int = 1 << 24,
    process_group: Any | None = None,
    fuse_cross_shard_gates: bool = True,
    pipeline_pair_exchange: bool = True,
    wire_layout: str = "canonical",
    preferred_local_wires: Sequence[int] = (),
    persistent_wire_layout: bool = False,
) -> TorchDistributedStatevectorResult:
    """Execute validated IR while retaining only the current rank's shard."""

    return _ShardedForwardSweep(
        circuit_or_ir,
        device=device,
        dtype=dtype,
        local_world_size=local_world_size,
        exchange_buffer_bytes=exchange_buffer_bytes,
        compact_index_threshold=compact_index_threshold,
        process_group=process_group,
        fuse_cross_shard_gates=fuse_cross_shard_gates,
        pipeline_pair_exchange=pipeline_pair_exchange,
        wire_layout=wire_layout,
        preferred_local_wires=preferred_local_wires,
        persistent_wire_layout=persistent_wire_layout,
    ).run()
