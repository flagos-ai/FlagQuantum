"""Built-in PyTorch CPU and CUDA platform providers."""

from __future__ import annotations

from contextlib import nullcontext
from time import perf_counter

import torch

from .contracts import MemorySnapshot, PlatformDevice, PlatformIdentity


class _CPUEvent:
    """Minimal host event used only by the portable CPU platform."""

    def __init__(self) -> None:
        self._recorded_at: float | None = None

    def record(self) -> None:
        self._recorded_at = perf_counter()

    def synchronize(self) -> None:
        return None

    def elapsed_time(self, end_event: "_CPUEvent") -> float:
        if self._recorded_at is None or end_event._recorded_at is None:
            raise RuntimeError("both CPU events must be recorded before timing")
        return (end_event._recorded_at - self._recorded_at) * 1000.0


class CPUPlatformRuntime:
    name = "pytorch_cpu"
    device_type = "cpu"
    optional_dependency = None

    def installed(self) -> bool:
        return True

    def activated(self) -> bool:
        return True

    def activate(self) -> None:
        return None

    def is_available(self) -> bool:
        return True

    def devices(self) -> tuple[PlatformDevice, ...]:
        return (
            PlatformDevice(
                device_type="cpu",
                index=None,
                name="CPU",
                provider=self.name,
            ),
        )

    def discover(self) -> tuple[PlatformDevice, ...]:
        return self.devices()

    def synchronize(self, device: torch.device) -> None:
        if device.type != "cpu":
            raise ValueError(f"CPU platform cannot synchronize {device}")

    def memory_snapshot(self, device: torch.device) -> MemorySnapshot:
        if device.type != "cpu":
            raise ValueError(f"CPU platform cannot inspect {device}")
        return MemorySnapshot()

    def stream(self, device: torch.device, priority: int = 0) -> object:
        if device.type != "cpu":
            raise ValueError(f"CPU platform cannot create a stream for {device}")
        return nullcontext()

    def event(self, device: torch.device) -> _CPUEvent:
        if device.type != "cpu":
            raise ValueError(f"CPU platform cannot create an event for {device}")
        return _CPUEvent()

    def rng_state(self, device: torch.device) -> torch.Tensor:
        if device.type != "cpu":
            raise ValueError(f"CPU platform cannot read RNG state for {device}")
        return torch.random.get_rng_state()

    def restore_rng_state(self, device: torch.device, state: torch.Tensor) -> None:
        if device.type != "cpu":
            raise ValueError(f"CPU platform cannot restore RNG state for {device}")
        torch.random.set_rng_state(state)

    def profiler_metadata(self, device: torch.device) -> dict[str, object]:
        if device.type != "cpu":
            raise ValueError(f"CPU platform cannot profile {device}")
        return {"platform": self.name, "device": str(device), "device_type": "cpu"}

    def identity(self) -> PlatformIdentity:
        return PlatformIdentity(
            provider=self.name,
            device_type=self.device_type,
            torch_version=str(torch.__version__),
        )


class CUDAPlatformRuntime:
    name = "pytorch_cuda"
    device_type = "cuda"
    optional_dependency = None

    def installed(self) -> bool:
        return hasattr(torch, "cuda")

    def activated(self) -> bool:
        return self.installed()

    def activate(self) -> None:
        return None

    def is_available(self) -> bool:
        return bool(self.installed() and torch.cuda.is_available())

    def devices(self) -> tuple[PlatformDevice, ...]:
        if not self.is_available():
            return ()
        devices: list[PlatformDevice] = []
        for index in range(torch.cuda.device_count()):
            memory = None
            try:
                memory = int(torch.cuda.get_device_properties(index).total_memory)
            except (AssertionError, RuntimeError):
                pass
            devices.append(
                PlatformDevice(
                    device_type="cuda",
                    index=index,
                    name=str(torch.cuda.get_device_name(index)),
                    provider=self.name,
                    memory_bytes=memory,
                )
            )
        return tuple(devices)

    def discover(self) -> tuple[PlatformDevice, ...]:
        return self.devices()

    def synchronize(self, device: torch.device) -> None:
        if device.type != "cuda":
            raise ValueError(f"CUDA platform cannot synchronize {device}")
        torch.cuda.synchronize(device)

    def memory_snapshot(self, device: torch.device) -> MemorySnapshot:
        if device.type != "cuda":
            raise ValueError(f"CUDA platform cannot inspect {device}")
        free = total = None
        try:
            free, total = (int(value) for value in torch.cuda.mem_get_info(device))
        except (AssertionError, RuntimeError):
            pass
        return MemorySnapshot(
            allocated_bytes=int(torch.cuda.memory_allocated(device)),
            reserved_bytes=int(torch.cuda.memory_reserved(device)),
            free_bytes=free,
            total_bytes=total,
        )

    def stream(self, device: torch.device, priority: int = 0) -> torch.cuda.Stream:
        if device.type != "cuda":
            raise ValueError(f"CUDA platform cannot create a stream for {device}")
        return torch.cuda.Stream(device=device, priority=priority)

    def event(self, device: torch.device) -> torch.cuda.Event:
        if device.type != "cuda":
            raise ValueError(f"CUDA platform cannot create an event for {device}")
        with torch.cuda.device(device):
            return torch.cuda.Event()

    def rng_state(self, device: torch.device) -> torch.Tensor:
        if device.type != "cuda":
            raise ValueError(f"CUDA platform cannot read RNG state for {device}")
        return torch.cuda.get_rng_state(device)

    def restore_rng_state(self, device: torch.device, state: torch.Tensor) -> None:
        if device.type != "cuda":
            raise ValueError(f"CUDA platform cannot restore RNG state for {device}")
        torch.cuda.set_rng_state(state, device)

    def profiler_metadata(self, device: torch.device) -> dict[str, object]:
        if device.type != "cuda":
            raise ValueError(f"CUDA platform cannot profile {device}")
        identity = self.identity().to_dict()
        identity.update({"device": str(device), "device_index": device.index})
        return identity

    def identity(self) -> PlatformIdentity:
        return PlatformIdentity(
            provider=self.name,
            device_type=self.device_type,
            torch_version=str(torch.__version__),
            provider_version=getattr(torch.version, "cuda", None),
            vendor="nvidia",
        )


__all__ = ("CPUPlatformRuntime", "CUDAPlatformRuntime")
