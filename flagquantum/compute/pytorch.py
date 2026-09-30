"""Built-in PyTorch CPU and CUDA compute runtimes."""

from __future__ import annotations

import ctypes
import os
import sys
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter
from typing import Protocol

import torch

from .contracts import MemorySnapshot, PlatformDevice, PlatformIdentity


def _sysconf_bytes(page_count_name: str) -> int | None:
    """Return one POSIX page count in bytes when the host exposes it."""

    try:
        pages = int(os.sysconf(page_count_name))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, TypeError, ValueError):
        return None
    if pages <= 0 or page_size <= 0:
        return None
    return pages * page_size


def _darwin_available_memory_bytes() -> int | None:
    """Read reclaimable host pages through the stable Mach host API."""

    if sys.platform != "darwin":
        return None

    natural = ctypes.c_uint32
    counter = ctypes.c_uint64

    class _VMStatistics64(ctypes.Structure):
        _fields_ = [
            ("free_count", natural),
            ("active_count", natural),
            ("inactive_count", natural),
            ("wire_count", natural),
            ("zero_fill_count", counter),
            ("reactivations", counter),
            ("pageins", counter),
            ("pageouts", counter),
            ("faults", counter),
            ("cow_faults", counter),
            ("lookups", counter),
            ("hits", counter),
            ("purges", counter),
            ("purgeable_count", natural),
            ("speculative_count", natural),
            ("decompressions", counter),
            ("compressions", counter),
            ("swapins", counter),
            ("swapouts", counter),
            ("compressor_page_count", natural),
            ("throttled_count", natural),
            ("external_page_count", natural),
            ("internal_page_count", natural),
            ("total_uncompressed_pages_in_compressor", counter),
        ]

    try:
        system = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        system.mach_host_self.restype = natural
        host = system.mach_host_self()
        page_size = natural()
        if system.host_page_size(host, ctypes.byref(page_size)) != 0:
            return None
        statistics = _VMStatistics64()
        count = natural(ctypes.sizeof(statistics) // ctypes.sizeof(ctypes.c_int))
        status = system.host_statistics64(
            host,
            4,  # HOST_VM_INFO64
            ctypes.cast(ctypes.byref(statistics), ctypes.POINTER(ctypes.c_int)),
            ctypes.byref(count),
        )
    except (AttributeError, OSError):
        return None
    if status != 0 or page_size.value <= 0:
        return None
    # Inactive pages are reclaimable without swapping. Speculative pages are
    # already included in free_count by the Mach contract.
    pages = int(statistics.free_count) + int(statistics.inactive_count)
    return pages * int(page_size.value)


def _file_integer(path: Path) -> int | None:
    try:
        raw = path.read_text(encoding="utf-8").strip()
        value = int(raw)
    except (OSError, ValueError):
        return None
    return value if value >= 0 else None


def _cgroup_memory_snapshot(root: Path = Path("/sys/fs/cgroup")) -> MemorySnapshot:
    """Return the active Linux container limit without assuming cgroup v1 or v2."""

    limit_path = root / "memory.max"
    usage_path = root / "memory.current"
    if not limit_path.exists():
        limit_path = root / "memory" / "memory.limit_in_bytes"
        usage_path = root / "memory" / "memory.usage_in_bytes"
    limit = _file_integer(limit_path)
    usage = _file_integer(usage_path)
    if limit is None or limit == 0:
        return MemorySnapshot()
    available = None if usage is None else max(0, limit - usage)
    return MemorySnapshot(free_bytes=available, total_bytes=limit)


def _minimum_known(*values: int | None) -> int | None:
    known = tuple(value for value in values if value is not None)
    return min(known) if known else None


def _host_memory_snapshot() -> MemorySnapshot:
    """Return dependency-free physical-memory facts for CPU planning."""

    total = _sysconf_bytes("SC_PHYS_PAGES")
    available = _sysconf_bytes("SC_AVPHYS_PAGES")
    if available is None:
        available = _darwin_available_memory_bytes()
    if sys.platform.startswith("linux"):
        cgroup = _cgroup_memory_snapshot()
        total = _minimum_known(total, cgroup.total_bytes)
        available = _minimum_known(available, cgroup.free_bytes)
    return MemorySnapshot(free_bytes=available, total_bytes=total)


class _CUDAStreamFactory(Protocol):
    """Typed constructor boundary for PyTorch's CUDA stream class."""

    def __call__(self, *, device: torch.device, priority: int) -> torch.cuda.Stream: ...


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
    optional_dependency: str | None = None

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
        return _host_memory_snapshot()

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
    optional_dependency: str | None = None

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
        create_stream: _CUDAStreamFactory = torch.cuda.Stream
        return create_stream(device=device, priority=priority)

    def event(self, device: torch.device) -> torch.cuda.Event:
        if device.type != "cuda":
            raise ValueError(f"CUDA platform cannot create an event for {device}")
        with torch.cuda.device(device):
            create_event: Callable[[], torch.cuda.Event] = torch.cuda.Event
            return create_event()

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
