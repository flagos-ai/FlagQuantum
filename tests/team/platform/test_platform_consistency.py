"""First provider-neutral consistency checks for platform integrations.

These tests exercise the existing ``PlatformRuntime`` boundary and the existing
extension SDK registry.  The adapter below is deliberately test-only: it proves
that a platform can participate in the SDK lifecycle without creating another
production registry or changing either protected contract.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest
import torch

from flagquantum.extensions import (
    CapabilityRequest,
    CapabilityResponse,
    ExtensionConfig,
    ExtensionManifest,
    ExtensionRegistry,
    extension_scope,
)
from flagquantum.runtime.platforms import (
    MemorySnapshot,
    PlatformRuntime,
    get_platform_runtime,
    list_platform_status,
)

pytestmark = pytest.mark.unit


def _portable_device_record(device: Any) -> dict[str, Any]:
    return {
        "device_type": device.device_type,
        "index": device.index,
        "name": device.name,
        "provider": device.provider,
        "available": device.available,
        "memory_bytes": device.memory_bytes,
        "metadata": dict(device.metadata),
    }


@dataclass
class _PlatformDeviceExtension:
    """Test bridge from PlatformRuntime into the authoritative SDK registry."""

    runtime: PlatformRuntime
    active: bool = False

    @property
    def manifest(self) -> ExtensionManifest:
        return ExtensionManifest(
            name=f"{self.runtime.name}_device",
            version="test",
            kind="device",
            capabilities=frozenset(
                {self.runtime.device_type, "device_discovery", "lifecycle"}
            ),
        )

    def negotiate(self, request: CapabilityRequest) -> CapabilityResponse:
        supported = self.manifest.capabilities
        blockers: list[str] = []
        if request.device_type not in (None, self.runtime.device_type):
            blockers.append(
                f"requested {request.device_type}, provider owns "
                f"{self.runtime.device_type}"
            )
        missing = request.required - supported
        if missing:
            blockers.append("missing capabilities: " + ", ".join(sorted(missing)))
        return CapabilityResponse(
            accepted=not blockers,
            supported=supported,
            blockers=tuple(blockers),
        )

    def start(self, config: ExtensionConfig) -> None:
        if config.values:
            raise ValueError("platform consistency bridge accepts no configuration")
        self.runtime.activate()
        self.active = True

    def devices(self) -> tuple[dict[str, Any], ...]:
        if not self.active:
            raise RuntimeError("device extension is not active")
        return tuple(_portable_device_record(item) for item in self.runtime.devices())

    def close(self) -> None:
        self.active = False


def test_builtin_platforms_implement_one_runtime_contract() -> None:
    status = {item["device_type"]: item for item in list_platform_status()}

    assert set(status) == {"cpu", "cuda", "flagos"}
    for device_type in status:
        runtime = get_platform_runtime(device_type)
        assert isinstance(runtime, PlatformRuntime)
        assert runtime.device_type == device_type
        json.dumps(status[device_type], sort_keys=True)

    assert status["cpu"]["installed"] is True
    assert status["cpu"]["activated"] is True
    assert status["cpu"]["available"] is True
    assert status["flagos"]["available"] in (None, True, False)


def test_cpu_platform_lifecycle_is_portable_and_self_consistent() -> None:
    runtime = get_platform_runtime("cpu")
    device = torch.device("cpu")
    discovered = runtime.discover()

    assert discovered == runtime.devices()
    assert len(discovered) == 1
    assert discovered[0].device == device
    assert discovered[0].provider == runtime.name
    assert runtime.identity().provider == runtime.name
    assert runtime.identity().device_type == runtime.device_type
    assert runtime.memory_snapshot(device) == MemorySnapshot()

    rng_state = runtime.rng_state(device)
    runtime.restore_rng_state(device, rng_state)
    start = runtime.event(device)
    end = runtime.event(device)
    start.record()
    end.record()
    runtime.synchronize(device)
    assert start.elapsed_time(end) >= 0.0
    with runtime.stream(device):
        pass


def test_platform_uses_existing_extension_sdk_lifecycle_without_root_mutation() -> None:
    extension = _PlatformDeviceExtension(get_platform_runtime("cpu"))
    empty = ExtensionRegistry()

    with extension_scope([extension]) as scoped:
        handle = scoped.negotiate(
            "device",
            extension.manifest.name,
            CapabilityRequest(
                required=frozenset({"cpu", "device_discovery", "lifecycle"}),
                device_type="cpu",
            ),
        )
        handle.start(ExtensionConfig())
        try:
            devices = handle.invoke("devices")
            assert devices == (
                {
                    "device_type": "cpu",
                    "index": None,
                    "name": "CPU",
                    "provider": "pytorch_cpu",
                    "available": True,
                    "memory_bytes": None,
                    "metadata": {},
                },
            )
            json.dumps(devices, sort_keys=True)
        finally:
            handle.close()

    assert extension.active is False
    assert empty.entries == {}
