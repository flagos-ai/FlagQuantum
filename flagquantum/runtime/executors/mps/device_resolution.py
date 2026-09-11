"""Device resolution shared by distributed MPS execution paths."""

from __future__ import annotations

import torch

from ...distributed.flagos_runtime import current_flagos_device
from .errors import MPSTrainingError


def resolve_distributed_mps_device(
    backend: str,
    device: torch.device | str | None,
) -> torch.device:
    backend = str(backend).strip().lower()
    if device is None and backend == "flagos":
        device = current_flagos_device()
    resolved = torch.device(device or "cpu")
    if backend == "nccl" and resolved.type != "cuda":
        raise MPSTrainingError("NCCL MPS training requires a CUDA device")
    if backend == "flagos" and resolved.type != "flagos":
        raise MPSTrainingError("FlagOS MPS training requires a flagos device")
    return resolved


__all__ = ("resolve_distributed_mps_device",)
