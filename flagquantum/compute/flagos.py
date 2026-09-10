"""Lazy Torch-FL compute runtime for the FlagOS PyTorch device.

Importing this module does not import ``torch_fl``. Activation happens only
after a user explicitly requests the ``flagos`` platform.
"""

from __future__ import annotations

import sys
from importlib import import_module
from importlib.util import find_spec
from types import ModuleType
from typing import Any

import torch

from .contracts import (
    MemorySnapshot,
    PlatformActivationError,
    PlatformDevice,
    PlatformIdentity,
    PlatformUnavailableError,
)


class FlagOSPlatformRuntime:
    name = "torch_fl"
    device_type = "flagos"
    optional_dependency: str | None = "torch_fl"

    def __init__(self) -> None:
        self._module: ModuleType | None = None

    def installed(self) -> bool:
        dependency = self.optional_dependency
        return dependency is not None and (
            dependency in sys.modules or find_spec(dependency) is not None
        )

    def activated(self) -> bool:
        return self._module is not None or (
            self.optional_dependency in sys.modules and hasattr(torch, "flagos")
        )

    def activate(self) -> None:
        if self._module is not None:
            return
        dependency = self.optional_dependency
        if dependency is None or not self.installed():
            raise PlatformActivationError(
                "FlagOS execution requires a compatible Torch-FL installation; "
                "CPU and native PyTorch execution remain available."
            )
        try:
            module = import_module(dependency)
        except Exception as exc:
            raise PlatformActivationError(
                "Torch-FL is installed but could not activate the FlagOS device: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if not hasattr(torch, "flagos"):
            raise PlatformActivationError(
                "Torch-FL imported without registering torch.flagos; install a "
                "Torch-FL build compatible with this PyTorch minor version."
            )
        self._module = module

    def _device_module(self) -> Any:
        self.activate()
        return torch.flagos

    def is_available(self) -> bool:
        module = self._device_module()
        checker = getattr(module, "is_available", None)
        return bool(checker() if checker is not None else False)

    def devices(self) -> tuple[PlatformDevice, ...]:
        module = self._device_module()
        if not self.is_available():
            return ()
        count = int(module.device_count())
        devices: list[PlatformDevice] = []
        for index in range(count):
            name_getter = getattr(module, "get_device_name", None)
            name = (
                str(name_getter(index))
                if name_getter is not None
                else f"FlagOS device {index}"
            )
            memory = None
            properties_getter = getattr(module, "get_device_properties", None)
            if properties_getter is not None:
                try:
                    memory = int(properties_getter(index).total_memory)
                except (AttributeError, RuntimeError):
                    pass
            devices.append(
                PlatformDevice(
                    device_type=self.device_type,
                    index=index,
                    name=name,
                    provider=self.name,
                    memory_bytes=memory,
                )
            )
        return tuple(devices)

    def discover(self) -> tuple[PlatformDevice, ...]:
        return self.devices()

    def synchronize(self, device: torch.device) -> None:
        if device.type != self.device_type:
            raise ValueError(f"FlagOS platform cannot synchronize {device}")
        self._device_module().synchronize(device)

    def memory_snapshot(self, device: torch.device) -> MemorySnapshot:
        if device.type != self.device_type:
            raise ValueError(f"FlagOS platform cannot inspect {device}")
        module = self._device_module()
        device_index = (
            int(device.index)
            if device.index is not None
            else int(module.current_device())
        )
        allocated = reserved = free = total = None
        if hasattr(module, "memory_allocated"):
            allocated = int(module.memory_allocated(device_index))
        if hasattr(module, "memory_reserved"):
            reserved = int(module.memory_reserved(device_index))
        if hasattr(module, "mem_get_info"):
            try:
                free, total = (
                    int(value) for value in module.mem_get_info(device_index)
                )
            except RuntimeError:
                pass
        return MemorySnapshot(
            allocated_bytes=allocated,
            reserved_bytes=reserved,
            free_bytes=free,
            total_bytes=total,
        )

    def _required_device_api(self, name: str) -> Any:
        module = self._device_module()
        value = getattr(module, name, None)
        if value is None:
            raise PlatformActivationError(
                f"Torch-FL activated torch.flagos without required {name!r} API."
            )
        return value

    def stream(self, device: torch.device, priority: int = 0) -> Any:
        if device.type != self.device_type:
            raise ValueError(f"FlagOS platform cannot create a stream for {device}")
        # PyTorch's device-agnostic API dispatches through the registered
        # PrivateUse1 GuardImpl.  Torch-FL's compatibility ``flagos.Stream``
        # may intentionally target a vendor Python shim instead and is not the
        # authoritative API for CUDA boxing.
        stream_factory = getattr(torch, "Stream", None)
        if stream_factory is not None:
            try:
                return stream_factory(device=device, priority=priority)
            except TypeError:
                # Compatibility with PyTorch releases whose generic factory
                # does not yet accept a priority argument.
                return stream_factory(device=device)
        return self._required_device_api("Stream")(device=device, priority=priority)

    def event(self, device: torch.device) -> Any:
        if device.type != self.device_type:
            raise ValueError(f"FlagOS platform cannot create an event for {device}")
        event_factory = getattr(torch, "Event", None)
        if event_factory is not None:
            try:
                return event_factory(device=device)
            except TypeError:
                pass
        device_context = getattr(self._device_module(), "device", None)
        if device_context is None:
            return self._required_device_api("Event")()
        with device_context(device):
            return self._required_device_api("Event")()

    def rng_state(self, device: torch.device) -> torch.Tensor:
        if device.type != self.device_type:
            raise ValueError(f"FlagOS platform cannot read RNG state for {device}")
        state = self._required_device_api("get_rng_state")(device)
        if not isinstance(state, torch.Tensor):
            raise TypeError("FlagOS get_rng_state must return a torch.Tensor")
        return state

    def restore_rng_state(self, device: torch.device, state: torch.Tensor) -> None:
        if device.type != self.device_type:
            raise ValueError(f"FlagOS platform cannot restore RNG state for {device}")
        self._required_device_api("set_rng_state")(state, device)

    def profiler_metadata(self, device: torch.device) -> dict[str, Any]:
        if device.type != self.device_type:
            raise ValueError(f"FlagOS platform cannot profile {device}")
        identity = self.identity().to_dict()
        identity.update({"device": str(device), "device_index": device.index})
        metadata_getter = getattr(self._device_module(), "profiler_metadata", None)
        if metadata_getter is not None:
            value = metadata_getter(device)
            if isinstance(value, dict):
                identity.update(value)
        return identity

    def identity(self) -> PlatformIdentity:
        self.activate()
        assert self._module is not None
        metadata: dict[str, Any] = {}
        identity_getter = getattr(self._module, "runtime_identity", None)
        if identity_getter is not None:
            value = identity_getter()
            if isinstance(value, dict):
                metadata.update(value)
            elif hasattr(value, "to_dict"):
                metadata.update(value.to_dict())
        return PlatformIdentity(
            provider=self.name,
            device_type=self.device_type,
            torch_version=str(torch.__version__),
            provider_version=getattr(self._module, "__version__", None),
            vendor=metadata.pop("vendor", None),
            metadata=metadata,
        )

    def require_available(self) -> None:
        if not self.is_available():
            raise PlatformUnavailableError(
                "Torch-FL is active but no FlagOS accelerator is available."
            )


__all__ = ("FlagOSPlatformRuntime",)
