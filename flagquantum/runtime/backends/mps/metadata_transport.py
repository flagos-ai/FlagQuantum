"""Tensor-native transport for variable-length distributed MPS metadata."""

from __future__ import annotations

import json
from typing import Any

import torch
import torch.distributed as dist

from ...distributed.flagos_runtime import current_flagos_device


def _collective_device() -> torch.device:
    backend = str(dist.get_backend()).strip().lower()
    if backend == "nccl":
        return torch.device("cuda", torch.cuda.current_device())
    if backend == "flagos":
        return current_flagos_device()
    return torch.device("cpu")


def all_gather_json(value: Any) -> tuple[Any, ...]:
    """Gather JSON-compatible metadata without pickle or NumPy."""
    device = _collective_device()
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")
    local_length = torch.tensor([len(encoded)], dtype=torch.int64, device=device)
    gathered_lengths = [
        torch.empty_like(local_length) for _ in range(dist.get_world_size())
    ]
    dist.all_gather(gathered_lengths, local_length)
    lengths = tuple(int(length.item()) for length in gathered_lengths)
    padded_length = max(lengths, default=0)
    local_bytes = torch.zeros(padded_length, dtype=torch.uint8, device=device)
    if encoded:
        local_bytes[: len(encoded)] = torch.tensor(
            list(encoded), dtype=torch.uint8, device=device
        )
    gathered_bytes = [
        torch.empty_like(local_bytes) for _ in range(dist.get_world_size())
    ]
    dist.all_gather(gathered_bytes, local_bytes)
    return tuple(
        json.loads(bytes(buffer[:length].cpu().tolist()).decode("utf-8"))
        for buffer, length in zip(gathered_bytes, lengths)
    )
