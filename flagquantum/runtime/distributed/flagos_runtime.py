"""Shared public-boundary helpers for FlagOS distributed execution."""

from __future__ import annotations

from typing import Any, Mapping

import torch


def is_flagos_request(
    device: torch.device | str | None,
    backend: str | None,
    *,
    local_rank: int | None = None,
) -> bool:
    """Validate and identify an explicit FlagOS process-group request."""

    explicit_backend = str(backend).strip().lower() if backend is not None else None
    if device is None:
        device_type = None
        device_index = None
    elif isinstance(device, str):
        parts = device.split(":", 1)
        device_type = parts[0].strip().lower()
        device_index = (
            int(parts[1]) if device_type == "flagos" and len(parts) == 2 else None
        )
    else:
        device_type = device.type
        device_index = device.index
    if explicit_backend == "flagos" and device_type not in (None, "flagos"):
        raise ValueError("backend='flagos' requires device='flagos:<local_rank>'")
    if device_type == "flagos" and explicit_backend not in (None, "flagos"):
        raise ValueError(
            f"device='flagos' requires backend='flagos'; received {backend!r}"
        )
    if (
        device_type == "flagos"
        and device_index is not None
        and local_rank is not None
        and int(device_index) != int(local_rank)
    ):
        raise ValueError(
            f"device={device!s} conflicts with LOCAL_RANK={int(local_rank)}"
        )
    return explicit_backend == "flagos" or device_type == "flagos"


def activate_flagos_device(
    local_rank: int,
) -> tuple[torch.device, Mapping[str, Any]]:
    """Activate Torch-FL through FlagQuantum's lazy platform boundary."""

    if local_rank < 0:
        raise ValueError("LOCAL_RANK must be non-negative")
    from ..platforms import get_platform_runtime

    platform = get_platform_runtime("flagos")
    platform.activate()
    if not platform.is_available():
        raise RuntimeError("Torch-FL activated but no flagos device is available.")
    devices = platform.devices()
    if local_rank >= len(devices):
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} has no matching flagos device; "
            f"provider reported {len(devices)} device(s)."
        )
    device_module = getattr(torch, "flagos", None)
    setter = getattr(device_module, "set_device", None)
    if setter is None:
        raise RuntimeError("Torch-FL activated without torch.flagos.set_device.")
    setter(local_rank)
    return torch.device("flagos", local_rank), platform.identity().to_dict()


def current_flagos_device() -> torch.device:
    """Return the current logical device without inspecting provider internals."""

    module = getattr(torch, "flagos", None)
    current_device = getattr(module, "current_device", None)
    if current_device is None:
        raise RuntimeError(
            "backend='flagos' is active without torch.flagos.current_device"
        )
    return torch.device("flagos", int(current_device()))


__all__ = (
    "activate_flagos_device",
    "current_flagos_device",
    "is_flagos_request",
)
