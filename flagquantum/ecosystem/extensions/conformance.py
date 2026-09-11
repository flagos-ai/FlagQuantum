"""Reusable conformance checks for extension authors and CI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ...core.ir import CircuitIR, Instruction
from .sdk import (
    CapabilityRequest,
    ExtensionConfig,
    ExtensionRegistry,
)


@dataclass(frozen=True)
class ConformanceReport:
    extension: str
    checks: tuple[str, ...]


def check_manifest_serialization(extension: Any) -> str:
    encoded = json.dumps(extension.manifest.to_dict(), sort_keys=True)
    if json.loads(encoded)["name"] != extension.manifest.name:
        raise AssertionError("manifest JSON round trip changed extension identity")
    return "manifest_serialization"


def run_backend_conformance(extension: Any) -> ConformanceReport:
    """Validate lifecycle, dtype/device negotiation and PyTorch gradients."""

    import torch

    checks = [check_manifest_serialization(extension)]
    registry = ExtensionRegistry().with_extension(extension)
    request = CapabilityRequest(
        required=frozenset({"cpu", "float32", "gradients"}),
        dtype="float32",
        device_type="cpu",
        require_gradients=True,
    )
    handle = registry.negotiate("backend", extension.manifest.name, request)
    handle.start(ExtensionConfig())
    try:
        values = torch.tensor([2.0, -3.0], requires_grad=True)
        result = handle.invoke("execute", None, values)
        if result.dtype != torch.float32 or result.device.type != "cpu":
            raise AssertionError("backend changed negotiated dtype/device")
        result.backward()
        if values.grad is None or not torch.allclose(values.grad, 2 * values.detach()):
            raise AssertionError("backend gradient conformance failed")
        checks.extend(("dtype_device", "gradients", "execution_errors"))
    finally:
        handle.close()
    if getattr(extension, "active", False):
        raise AssertionError("backend leaked active lifecycle state")
    checks.append("cleanup")
    return ConformanceReport(extension.manifest.name, tuple(checks))


def run_provider_conformance(extension: Any) -> ConformanceReport:
    """Validate provider negotiation, lifecycle, serialization and cleanup."""

    checks = [check_manifest_serialization(extension)]
    registry = ExtensionRegistry().with_extension(extension)
    request = CapabilityRequest(required=frozenset({"qasm", "simulator"}))
    handle = registry.negotiate("provider", extension.manifest.name, request)
    handle.start(ExtensionConfig({"endpoint": "local"}))
    try:
        payload = {"qasm": "OPENQASM 2.0;", "shots": 1}
        json.dumps(payload)
        job = handle.invoke("submit", payload)
        if handle.invoke("status", job) != "completed":
            raise AssertionError("provider did not complete reference job")
        checks.extend(("payload_serialization", "provider_errors"))
    finally:
        handle.close()
    if getattr(extension, "active", False) or getattr(extension, "jobs", {}):
        raise AssertionError("provider leaked lifecycle state")
    checks.append("cleanup")
    return ConformanceReport(extension.manifest.name, tuple(checks))


def run_compiler_conformance(extension: Any) -> ConformanceReport:
    """Validate circuit-IR ownership, determinism, lifecycle, and cleanup."""

    checks = [check_manifest_serialization(extension)]
    registry = ExtensionRegistry().with_extension(extension)
    request = CapabilityRequest(required=frozenset({"circuit_ir"}))
    handle = registry.negotiate("compiler", extension.manifest.name, request)
    handle.start(ExtensionConfig())
    try:
        source = CircuitIR(2, (Instruction("h", (0,)),))
        source_hash = source.content_hash
        target = {"basis_gates": ("h", "cx")}
        first = handle.invoke("compile", source, target=target)
        second = handle.invoke("compile", source, target=target)
        if not isinstance(first, CircuitIR) or not isinstance(second, CircuitIR):
            raise AssertionError("compiler must return CircuitIR")
        if source.content_hash != source_hash:
            raise AssertionError("compiler mutated its input CircuitIR")
        if first.content_hash != second.content_hash:
            raise AssertionError("compiler output is not deterministic")
        checks.extend(("circuit_ir", "input_immutable", "deterministic"))
    finally:
        handle.close()
    if getattr(extension, "active", False):
        raise AssertionError("compiler leaked active lifecycle state")
    checks.append("cleanup")
    return ConformanceReport(extension.manifest.name, tuple(checks))


__all__ = (
    "ConformanceReport",
    "check_manifest_serialization",
    "run_backend_conformance",
    "run_compiler_conformance",
    "run_provider_conformance",
)
