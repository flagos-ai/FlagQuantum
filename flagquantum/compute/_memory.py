"""Internal allocator peak observation for device training phases."""

from __future__ import annotations

import torch

from .contracts import PlatformRuntime
from .registry import get_platform_runtime


def _peak_allocated_memory(
    device: torch.device, *, reset: bool, platform: PlatformRuntime | None = None
) -> int:
    if device.type == "cpu":
        return 0
    platform = platform or get_platform_runtime(device.type)
    platform.synchronize(device)
    module = getattr(torch, device.type)
    index = device.index if device.index is not None else module.current_device()
    if device.type == "flagos":
        peak = int(module.memory_stats(index)["peak_allocated_bytes"])
    else:
        peak = int(module.max_memory_allocated(index))
    if reset:
        module.reset_peak_memory_stats(index)
    return peak
