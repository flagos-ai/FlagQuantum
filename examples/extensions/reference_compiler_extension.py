"""Minimal independently installable circuit-compiler extension."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from flagquantum import CircuitIR
from flagquantum.ecosystem.extensions import (
    CapabilityRequest,
    CapabilityResponse,
    ExtensionConfig,
    ExtensionManifest,
)


class ReferenceCircuitCompiler:
    manifest = ExtensionManifest(
        name="reference_compiler",
        version="1.0.0",
        kind="compiler",
        capabilities=frozenset({"circuit_ir"}),
    )

    def __init__(self) -> None:
        self.active = False

    def negotiate(self, request: CapabilityRequest) -> CapabilityResponse:
        missing = request.required - self.manifest.capabilities
        return CapabilityResponse(
            not missing,
            self.manifest.capabilities,
            (
                ("missing capabilities: " + ", ".join(sorted(missing)),)
                if missing
                else ()
            ),
        )

    def start(self, config: ExtensionConfig) -> None:
        self.active = True

    def compile(
        self,
        program: CircuitIR,
        *,
        target: Mapping[str, Any] | None = None,
    ) -> CircuitIR:
        if not self.active:
            raise RuntimeError("compiler is closed")
        metadata = dict(program.metadata)
        metadata["reference_compiler_target"] = dict(target or {})
        return replace(program, metadata=metadata)

    def close(self) -> None:
        self.active = False
