"""Accelerator phase metrics read allocator peaks after synchronization."""

from types import SimpleNamespace

import pytest
import torch

from flagquantum.compute import _memory


def test_flagos_peak_is_read_before_reset_on_the_requested_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []
    device = SimpleNamespace(type="flagos", index=1)
    platform = SimpleNamespace(
        synchronize=lambda value: events.append(("sync", value.index))
    )
    module = SimpleNamespace(
        memory_stats=lambda index: (
            events.append(("read", index)) or {"peak_allocated_bytes": 8192}
        ),
        reset_peak_memory_stats=lambda index: events.append(("reset", index)),
    )
    monkeypatch.setattr(_memory, "get_platform_runtime", lambda kind: platform)
    monkeypatch.setattr(torch, "flagos", module, raising=False)
    assert _memory._peak_allocated_memory(device, reset=True) == 8192
    assert events == [("sync", 1), ("read", 1), ("reset", 1)]


def test_cpu_peak_observation_does_not_activate_an_accelerator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected(kind: str) -> None:
        raise AssertionError("CPU observation must not load an accelerator")

    monkeypatch.setattr(_memory, "get_platform_runtime", unexpected)
    assert _memory._peak_allocated_memory(torch.device("cpu"), reset=True) == 0
