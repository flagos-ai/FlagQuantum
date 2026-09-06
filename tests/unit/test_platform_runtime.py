"""Platform boundaries for CUDA preservation and lazy FlagOS activation."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from flagquantum.providers.platform import (
    PlatformActivationError,
    discover_platform_devices,
    get_platform_runtime,
    list_platform_status,
)
from flagquantum.providers.platform import flagos as flagos_module
from flagquantum.providers.platform.flagos import FlagOSPlatformRuntime
from flagquantum.runtime import backend_registry

pytestmark = pytest.mark.unit


def test_cpu_platform_is_always_available_without_optional_imports():
    platform = get_platform_runtime("cpu")

    assert platform.installed()
    assert platform.activated()
    assert platform.is_available()
    assert platform.devices()[0].device == torch.device("cpu")


def test_cpu_platform_owns_rng_and_event_lifecycle():
    platform = get_platform_runtime("cpu")
    device = torch.device("cpu")
    state = platform.rng_state(device)
    start = platform.event(device)
    end = platform.event(device)

    start.record()
    end.record()
    platform.restore_rng_state(device, state)

    assert start.elapsed_time(end) >= 0.0
    assert platform.profiler_metadata(device)["device_type"] == "cpu"


def test_flagos_status_does_not_import_torch_fl(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch_fl", raising=False)
    monkeypatch.setattr(flagos_module, "find_spec", lambda name: None)

    status = {item["device_type"]: item for item in list_platform_status()}

    assert "torch_fl" not in sys.modules
    assert status["flagos"]["installed"] is False
    assert status["flagos"]["activated"] is False
    assert status["flagos"]["available"] is None


def test_missing_torch_fl_fails_only_when_flagos_is_activated(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch_fl", raising=False)
    monkeypatch.setattr(flagos_module, "find_spec", lambda name: None)
    platform = FlagOSPlatformRuntime()

    with pytest.raises(PlatformActivationError, match="compatible Torch-FL"):
        platform.activate()


def test_unknown_accelerator_name_is_not_reported_as_cuda():
    assert backend_registry._accelerator_kind("Example Quantum Accelerator") == (
        "unknown"
    )


def test_vendor_names_are_not_platform_classification_policy():
    assert backend_registry._accelerator_kind("Hygon DCU") == "unknown"
    assert backend_registry._accelerator_kind("flagos") == "flagos"


def test_environment_hint_is_unverified_and_does_not_enable_device(
    monkeypatch,
):
    try:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setenv("FLAGQUANTUM_ACCELERATOR", "future_accelerator")

        detected = backend_registry.detect_accelerators()
        capabilities = backend_registry.get_backend_capabilities(refresh=True)

        assert len(detected) == 1
        assert detected[0].kind == "unknown"
        assert detected[0].device_type == "future_accelerator"
        assert detected[0].available is False
        assert capabilities.devices == ("cpu",)
    finally:
        monkeypatch.undo()
        backend_registry.refresh_backend_registry()


def test_cuda_discovery_preserves_existing_fast_path(monkeypatch):
    properties = SimpleNamespace(total_memory=16 * 1024**3)
    try:
        monkeypatch.delenv("FLAGQUANTUM_ACCELERATOR", raising=False)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
        monkeypatch.setattr(
            torch.cuda, "get_device_name", lambda index: "NVIDIA Test GPU"
        )
        monkeypatch.setattr(
            torch.cuda, "get_device_properties", lambda index: properties
        )

        devices = discover_platform_devices()
        capabilities = backend_registry.get_backend_capabilities(refresh=True)

        assert any(item.device_type == "cuda" for item in devices)
        assert capabilities.devices == ("cpu", "cuda")
        assert capabilities.preferred_device == "cuda"
        assert capabilities.accelerators[0].kind == "cuda"
        assert capabilities.accelerators[0].memory_bytes == properties.total_memory
    finally:
        monkeypatch.undo()
        backend_registry.refresh_backend_registry()


def test_flagos_visibility_does_not_promote_auto_without_workload_evidence():
    base = backend_registry.get_backend_capabilities(refresh=True)
    flagos = backend_registry.AcceleratorInfo(
        kind="flagos",
        index=0,
        name="FlagOS Test Device",
        available=True,
        device_type="flagos",
    )

    projected = backend_registry.with_accelerators(base, (flagos,))

    assert projected.devices == ("cpu", "flagos")
    assert projected.preferred_device == "cpu"


def test_explicit_flagos_request_reaches_lazy_platform_provider(monkeypatch):
    captured: list[str] = []

    monkeypatch.setattr(
        backend_registry,
        "resolve_platform_device",
        lambda device: captured.append(str(device)) or torch.device("cpu"),
    )

    backend_registry.resolve_device("flagos:0")

    assert captured == ["flagos:0"]


def test_declared_custom_torch_device_does_not_require_builtin_platform():
    capabilities = backend_registry.BackendCapabilities(
        name="mps_test",
        tensor_backend="torch",
        devices=("mps",),
        dtypes=("complex64",),
        supports_autograd=True,
        supports_distributed=False,
        supports_statevector=True,
        supports_density_matrix=False,
        supports_mps=False,
        preferred_device="mps",
    )
    try:
        backend_registry.register_backend(capabilities)

        assert backend_registry.resolve_device("mps", backend="mps_test") == (
            torch.device("mps")
        )
    finally:
        backend_registry.refresh_backend_registry()


def test_known_platform_key_error_is_not_bypassed(monkeypatch):
    def fail_resolution(device):
        raise KeyError("provider discovery failed")

    monkeypatch.setattr(backend_registry, "resolve_platform_device", fail_resolution)

    with pytest.raises(KeyError, match="provider discovery failed"):
        backend_registry.resolve_device("cpu")


def test_hygon_cuda_compatibility_metadata_stays_owned_by_torch_fl(monkeypatch):
    torch_fl = ModuleType("torch_fl")
    torch_fl.__version__ = "test"
    torch_fl.runtime_identity = lambda: {
        "vendor": "hygon",
        "route_backend": "cuda_compatible",
        "runtime": "dtk",
    }
    flagos_device = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        get_device_name=lambda index: "Hygon DCU through FlagOS",
        get_device_properties=lambda index: SimpleNamespace(total_memory=32 * 1024**3),
    )
    monkeypatch.setitem(sys.modules, "torch_fl", torch_fl)
    monkeypatch.setattr(torch, "flagos", flagos_device, raising=False)

    platform = FlagOSPlatformRuntime()
    devices = platform.discover()
    identity = platform.identity()

    assert devices[0].device_type == "flagos"
    assert identity.provider == "torch_fl"
    assert identity.vendor == "hygon"
    assert identity.metadata["route_backend"] == "cuda_compatible"
    assert identity.metadata["runtime"] == "dtk"


def test_flagos_stream_and_event_use_device_agnostic_pytorch_api(monkeypatch):
    captured: dict[str, object] = {}
    expected_stream = object()
    expected_event = object()

    def stream_factory(*, device, priority):
        captured["stream"] = (device, priority)
        return expected_stream

    def event_factory(*, device):
        captured["event"] = device
        return expected_event

    monkeypatch.setattr(torch, "Stream", stream_factory)
    monkeypatch.setattr(torch, "Event", event_factory)
    platform = FlagOSPlatformRuntime()
    device = torch.device("privateuseone:0")
    monkeypatch.setattr(platform, "device_type", "privateuseone")

    assert platform.stream(device, priority=-1) is expected_stream
    assert platform.event(device) is expected_event
    assert captured == {
        "stream": (device, -1),
        "event": device,
    }


def test_flagos_memory_api_receives_device_index(monkeypatch):
    captured: list[int] = []

    def memory_allocated(index):
        captured.append(index)
        return 128

    def memory_reserved(index):
        captured.append(index)
        return 256

    device_module = SimpleNamespace(
        memory_allocated=memory_allocated,
        memory_reserved=memory_reserved,
    )
    monkeypatch.setattr(torch, "flagos", device_module, raising=False)
    platform = FlagOSPlatformRuntime()
    platform._module = ModuleType("torch_fl")
    platform.device_type = "privateuseone"

    snapshot = platform.memory_snapshot(torch.device("privateuseone:0"))

    assert snapshot.allocated_bytes == 128
    assert snapshot.reserved_bytes == 256
    assert captured == [0, 0]
