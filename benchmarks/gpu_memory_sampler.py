"""Dependency-free NVML peak-memory sampling for isolated GPU benchmarks."""

from __future__ import annotations

import ctypes
import os
import threading
from dataclasses import dataclass, field


class _Memory(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


def physical_device_index(cuda_device_index: int) -> int:
    """Map a process-local CUDA index to its numeric NVML device index."""

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        return cuda_device_index
    entries = [entry.strip() for entry in visible.split(",")]
    if cuda_device_index >= len(entries) or not entries[cuda_device_index].isdigit():
        raise ValueError(
            "NVML sampler requires numeric CUDA_VISIBLE_DEVICES entries"
        )
    return int(entries[cuda_device_index])


@dataclass
class NvmlMemorySampler:
    """Sample device-wide used memory while one benchmark rank owns the GPU."""

    device_index: int
    interval_seconds: float = 0.02
    peak_bytes: int = 0
    baseline_bytes: int = 0
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.device_index = physical_device_index(self.device_index)
        self._nvml = ctypes.CDLL("libnvidia-ml.so.1")
        self._nvml.nvmlInit_v2.restype = ctypes.c_int
        self._nvml.nvmlDeviceGetHandleByIndex_v2.argtypes = [
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._nvml.nvmlDeviceGetHandleByIndex_v2.restype = ctypes.c_int
        self._nvml.nvmlDeviceGetMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_Memory),
        ]
        self._nvml.nvmlDeviceGetMemoryInfo.restype = ctypes.c_int
        if self._nvml.nvmlInit_v2() != 0:
            raise RuntimeError("NVML initialization failed")
        self._handle = ctypes.c_void_p()
        if (
            self._nvml.nvmlDeviceGetHandleByIndex_v2(
                self.device_index, ctypes.byref(self._handle)
            )
            != 0
        ):
            raise RuntimeError(f"NVML cannot resolve GPU {self.device_index}")

    def _used_bytes(self) -> int:
        memory = _Memory()
        if self._nvml.nvmlDeviceGetMemoryInfo(self._handle, ctypes.byref(memory)) != 0:
            raise RuntimeError("NVML memory query failed")
        return int(memory.used)

    def _sample_until_stopped(self) -> None:
        while not self._stop.is_set():
            self.peak_bytes = max(self.peak_bytes, self._used_bytes())
            self._stop.wait(self.interval_seconds)

    def __enter__(self) -> NvmlMemorySampler:
        self.baseline_bytes = self._used_bytes()
        self.peak_bytes = self.baseline_bytes
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._sample_until_stopped,
            name=f"nvml-memory-gpu-{self.device_index}",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.peak_bytes = max(self.peak_bytes, self._used_bytes())
