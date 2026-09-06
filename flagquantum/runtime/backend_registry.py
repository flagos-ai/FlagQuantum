"""Canonical backend capability registry for FlagQuantum.

This module keeps accelerator discovery and execution policy inside
FlagQuantum. It intentionally uses only PyTorch and the Python standard
library so the product can target AI accelerators through their mainstream
PyTorch integrations without depending on scientific helper packages.
"""

from __future__ import annotations

import os
from contextvars import ContextVar
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Mapping

import torch

from ..core.runtime_config import get_runtime_config
from ..providers.platform import (
    discover_platform_devices,
    get_platform_runtime,
    resolve_platform_device,
)


@dataclass(frozen=True)
class AcceleratorInfo:
    """Detected accelerator device information."""

    kind: str
    index: int
    name: str
    available: bool = True
    memory_bytes: int | None = None
    device_type: str = ""


@dataclass(frozen=True)
class BackendCapabilities:
    """Execution capabilities exposed by a tensor backend."""

    name: str
    tensor_backend: str
    devices: tuple[str, ...]
    dtypes: tuple[str, ...]
    supports_autograd: bool
    supports_distributed: bool
    supports_statevector: bool
    supports_density_matrix: bool
    supports_mps: bool
    preferred_device: str = "cpu"
    accelerators: tuple[AcceleratorInfo, ...] = ()

    def supports_mode(self, mode: str) -> bool:
        normalized = mode.lower()
        if normalized in {"auto", "local"}:
            return True
        if normalized == "statevector":
            return self.supports_statevector
        if normalized == "density_matrix":
            return self.supports_density_matrix
        if normalized in {
            "mps",
            "adaptive_mps",
            "distributed_mps",
            "mps_trajectory",
            "noisy_mps",
        }:
            return self.supports_mps
        if normalized in {
            "distributed",
            "distributed_statevector",
            "distributed_tensor_network",
            "distributed_tn",
        }:
            return self.supports_distributed
        return False


_BACKENDS: ContextVar[Mapping[str, BackendCapabilities]] = ContextVar(
    "flagquantum_backend_registry", default=MappingProxyType({})
)


def _accelerator_kind(device_type: str) -> str:
    """Classify only FlagQuantum platform types, never vendor device names."""

    normalized = device_type.lower().split(":", 1)[0]
    return normalized if normalized in {"cuda", "flagos"} else "unknown"


def detect_accelerators() -> tuple[AcceleratorInfo, ...]:
    """Detect accelerators exposed through the PyTorch runtime."""

    detected = [
        AcceleratorInfo(
            kind=_accelerator_kind(device.device_type),
            index=int(device.index or 0),
            name=device.name,
            available=device.available,
            memory_bytes=device.memory_bytes,
            device_type=device.device_type,
        )
        for device in discover_platform_devices()
        if device.device_type != "cpu"
    ]

    explicit = os.environ.get("FLAGQUANTUM_ACCELERATOR")
    if explicit and not detected:
        device_type = explicit.lower().split(":", 1)[0]
        detected.append(
            AcceleratorInfo(
                kind=_accelerator_kind(device_type),
                index=0,
                name=explicit,
                available=False,
                device_type=device_type,
            )
        )
    return tuple(detected)


def _build_pytorch_capabilities() -> BackendCapabilities:
    accelerators = detect_accelerators()
    devices = ["cpu"]
    for accelerator in accelerators:
        if (
            accelerator.available
            and accelerator.device_type
            and accelerator.device_type not in devices
        ):
            devices.append(accelerator.device_type)
    # FlagOS remains explicit until a workload profile has verified operator and
    # numerical evidence. Device visibility alone must not change ``auto``.
    preferred = "cuda" if "cuda" in devices else "cpu"
    return BackendCapabilities(
        name="pytorch",
        tensor_backend="torch",
        devices=tuple(devices),
        dtypes=("complex64", "complex128"),
        supports_autograd=True,
        supports_distributed=True,
        supports_statevector=True,
        supports_density_matrix=True,
        supports_mps=True,
        preferred_device=preferred,
        accelerators=accelerators,
    )


def refresh_backend_registry() -> dict[str, BackendCapabilities]:
    """Refresh the built-in backend registry from the current runtime."""

    pytorch = _build_pytorch_capabilities()
    registry = MappingProxyType({"pytorch": pytorch, "torch": pytorch})
    _BACKENDS.set(registry)
    return dict(registry)


def register_backend(
    capabilities: BackendCapabilities, *aliases: str
) -> BackendCapabilities:
    """Register a backend capability record."""

    updated = dict(_BACKENDS.get())
    updated[capabilities.name.lower()] = capabilities
    for alias in aliases:
        updated[alias.lower()] = capabilities
    _BACKENDS.set(MappingProxyType(updated))
    return capabilities


def list_backends(*, refresh: bool = False) -> dict[str, BackendCapabilities]:
    if refresh or not _BACKENDS.get():
        refresh_backend_registry()
    return dict(_BACKENDS.get())


def get_backend_capabilities(
    name: str | None = None, *, refresh: bool = False
) -> BackendCapabilities:
    if refresh or not _BACKENDS.get():
        refresh_backend_registry()
    registry = _BACKENDS.get()
    key = (name or get_runtime_config().backend).lower()
    if key not in registry:
        raise KeyError(f"Unknown FlagQuantum backend {name!r}.")
    return registry[key]


def set_active_backend(name: str = "pytorch") -> BackendCapabilities:
    return get_backend_capabilities(name)


def get_active_backend() -> str:
    return str(get_runtime_config().backend)


def resolve_device(
    device: str | torch.device | None = None,
    *,
    backend: str | None = None,
    require_accelerator: bool = False,
) -> torch.device:
    """Resolve a user device request against backend capabilities."""

    capabilities = get_backend_capabilities(backend)
    if device is None or str(device) == "auto":
        device = capabilities.preferred_device
    requested_type = str(device).lower().split(":", 1)[0]
    if requested_type not in capabilities.devices and requested_type != "flagos":
        raise ValueError(
            f"Backend {capabilities.name!r} does not expose device "
            f"{requested_type!r}."
        )
    try:
        get_platform_runtime(requested_type)
    except KeyError:
        # A custom backend may expose a PyTorch device type without registering
        # a built-in PlatformRuntime. The backend declaration is authoritative.
        resolved = torch.device(device)
    else:
        # Once a platform owns the device type, activation and discovery errors
        # must propagate rather than bypassing the provider boundary.
        resolved = resolve_platform_device(device)
    if require_accelerator and resolved.type == "cpu":
        raise RuntimeError("No accelerator is available for this backend.")
    return resolved


def resolve_dtype(
    dtype: str | torch.dtype | None = None,
) -> tuple[torch.dtype, torch.dtype]:
    """Resolve real and complex dtype pair."""

    if dtype is None:
        return torch.float32, torch.complex64
    if isinstance(dtype, torch.dtype):
        if dtype in {torch.float32, torch.complex64}:
            return torch.float32, torch.complex64
        if dtype in {torch.float64, torch.complex128}:
            return torch.float64, torch.complex128
        raise ValueError(f"Unsupported dtype {dtype!r}.")
    normalized = dtype.lower()
    if normalized in {"float32", "complex64"}:
        return torch.float32, torch.complex64
    if normalized in {"float64", "complex128"}:
        return torch.float64, torch.complex128
    raise ValueError(f"Unsupported dtype {dtype!r}.")


def backend_execution_options(
    *,
    mode: str = "auto",
    backend: str | None = None,
    device: str | torch.device | None = None,
    dtype: str | torch.dtype | None = None,
    world_size: int = 1,
) -> dict[str, Any]:
    """Build normalized execution options for FlagQuantum runners."""

    capabilities = get_backend_capabilities(backend)
    if not capabilities.supports_mode(mode):
        raise ValueError(
            f"Backend {capabilities.name!r} does not support mode {mode!r}."
        )
    resolved_device = resolve_device(device, backend=capabilities.name)
    real_dtype, complex_dtype = resolve_dtype(dtype)
    if int(world_size) > 1 and not capabilities.supports_distributed:
        raise ValueError(
            f"Backend {capabilities.name!r} does not support distributed execution."
        )
    return {
        "backend": capabilities.name,
        "device": (
            resolved_device.type
            if resolved_device.index is None
            else str(resolved_device)
        ),
        "real_dtype": real_dtype,
        "complex_dtype": complex_dtype,
        "world_size": int(world_size),
        "mode": mode,
    }


def with_accelerators(
    capabilities: BackendCapabilities,
    accelerators: tuple[AcceleratorInfo, ...],
) -> BackendCapabilities:
    """Return a capability record with updated accelerator metadata."""

    devices = ["cpu"]
    for accelerator in accelerators:
        if (
            accelerator.available
            and accelerator.device_type
            and accelerator.device_type not in devices
        ):
            devices.append(accelerator.device_type)
    preferred = "cuda" if "cuda" in devices else "cpu"
    return replace(
        capabilities,
        devices=tuple(devices),
        preferred_device=preferred,
        accelerators=accelerators,
    )


def capability_summary(
    name: str | None = None, *, refresh: bool = False
) -> Mapping[str, Any]:
    """Return a serializable summary for diagnostics and tests."""

    capabilities = get_backend_capabilities(name, refresh=refresh)
    return {
        "name": capabilities.name,
        "tensor_backend": capabilities.tensor_backend,
        "devices": capabilities.devices,
        "dtypes": capabilities.dtypes,
        "supports_autograd": capabilities.supports_autograd,
        "supports_distributed": capabilities.supports_distributed,
        "supports_statevector": capabilities.supports_statevector,
        "supports_density_matrix": capabilities.supports_density_matrix,
        "supports_mps": capabilities.supports_mps,
        "preferred_device": capabilities.preferred_device,
        "accelerators": tuple(
            {
                "kind": item.kind,
                "index": item.index,
                "name": item.name,
                "available": item.available,
                "memory_bytes": item.memory_bytes,
                "device_type": item.device_type,
            }
            for item in capabilities.accelerators
        ),
    }


refresh_backend_registry()


__all__ = [
    "AcceleratorInfo",
    "BackendCapabilities",
    "backend_execution_options",
    "capability_summary",
    "detect_accelerators",
    "get_active_backend",
    "get_backend_capabilities",
    "list_backends",
    "refresh_backend_registry",
    "register_backend",
    "resolve_device",
    "resolve_dtype",
    "set_active_backend",
    "with_accelerators",
]
