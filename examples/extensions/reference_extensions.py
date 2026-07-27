"""Minimal extensions implemented without importing FlagQuantum internals."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from flagquantum.extensions import (
    CapabilityRequest,
    CapabilityResponse,
    ExtensionConfig,
    ExtensionManifest,
)


class ReferenceTorchBackend:
    manifest = ExtensionManifest(
        name="reference_torch",
        version="1.0.0",
        kind="backend",
        capabilities=frozenset({"float32", "cpu", "gradients", "serialization"}),
    )

    def __init__(self) -> None:
        self.active = False

    def negotiate(self, request: CapabilityRequest) -> CapabilityResponse:
        supported = self.manifest.capabilities
        blockers = []
        if request.dtype not in (None, "float32"):
            blockers.append(f"dtype {request.dtype!r} is unsupported")
        if request.device_type not in (None, "cpu"):
            blockers.append(f"device {request.device_type!r} is unsupported")
        return CapabilityResponse(not blockers, supported, tuple(blockers))

    def start(self, config: ExtensionConfig) -> None:
        self.active = True

    def execute(self, program: Any, parameters: Any = None) -> torch.Tensor:
        if not self.active:
            raise RuntimeError("backend is closed")
        value = torch.as_tensor(parameters, dtype=torch.float32)
        return value.square().sum()

    def close(self) -> None:
        self.active = False


class ReferenceProvider:
    manifest = ExtensionManifest(
        name="reference_provider",
        version="1.0.0",
        kind="provider",
        capabilities=frozenset({"qasm", "simulator"}),
    )

    def __init__(self) -> None:
        self.active = False
        self.jobs: dict[str, Mapping[str, Any]] = {}

    def negotiate(self, request: CapabilityRequest) -> CapabilityResponse:
        missing = request.required - self.manifest.capabilities
        return CapabilityResponse(
            not missing,
            self.manifest.capabilities,
            (
                (("missing capabilities: " + ", ".join(sorted(missing))),)
                if missing
                else ()
            ),
        )

    def start(self, config: ExtensionConfig) -> None:
        self.active = True

    def submit(self, payload: Mapping[str, Any]) -> str:
        if not self.active:
            raise RuntimeError("provider is closed")
        handle = f"reference-{len(self.jobs) + 1}"
        self.jobs[handle] = dict(payload)
        return handle

    def status(self, handle: str) -> str:
        return "completed" if handle in self.jobs else "unknown"

    def close(self) -> None:
        self.active = False
        self.jobs.clear()
