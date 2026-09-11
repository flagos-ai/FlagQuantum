"""Immutable lazy registry for optional interoperability adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from types import MappingProxyType
from typing import Any, Mapping

from .contracts import INTEROP_API_VERSION, InteropAdapter, InteropError


class InteropRegistryError(InteropError, RuntimeError):
    """Raised before use when an adapter registration is invalid."""


@dataclass(frozen=True)
class InteropAdapterSpec:
    """Import-safe identity and lazy loading instructions for one adapter."""

    name: str
    module: str
    attribute: str
    dependency_extra: str
    api_version: str = INTEROP_API_VERSION
    experimental: bool = True

    def __post_init__(self) -> None:
        if not self.name or self.name != self.name.lower():
            raise ValueError("interop adapter name must be non-empty lowercase text")
        if not self.module or not self.attribute or not self.dependency_extra:
            raise ValueError("interop adapter loading fields must be non-empty")

    def load(self) -> InteropAdapter:
        if self.api_version != INTEROP_API_VERSION:
            raise InteropRegistryError(
                f"adapter {self.name!r} targets interop API {self.api_version}; "
                f"FlagQuantum provides {INTEROP_API_VERSION}"
            )
        try:
            module = import_module(self.module)
            adapter = getattr(module, self.attribute)
        except (AttributeError, ImportError) as exc:
            raise InteropRegistryError(
                f"adapter {self.name!r} implementation is unavailable; install "
                f"the FlagQuantum extra {self.dependency_extra!r} and verify "
                f"{self.module}:{self.attribute}"
            ) from exc
        if not isinstance(adapter, InteropAdapter):
            raise InteropRegistryError(
                f"adapter {self.name!r} does not implement InteropAdapter"
            )
        identity = (
            adapter.name,
            adapter.api_version,
            adapter.dependency_extra,
        )
        expected = (self.name, self.api_version, self.dependency_extra)
        if identity != expected:
            raise InteropRegistryError(
                f"adapter {self.name!r} identity {identity!r} does not match "
                f"registered identity {expected!r}"
            )
        return adapter

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "module": self.module,
            "attribute": self.attribute,
            "dependency_extra": self.dependency_extra,
            "api_version": self.api_version,
            "experimental": self.experimental,
        }


@dataclass(frozen=True)
class InteropRegistry:
    """Immutable registry whose entries do not import external frameworks."""

    specs: Mapping[str, InteropAdapterSpec] = field(default_factory=dict)

    def __post_init__(self) -> None:
        copied = dict(self.specs)
        if any(name != spec.name for name, spec in copied.items()):
            raise ValueError("interop registry keys must match adapter names")
        object.__setattr__(self, "specs", MappingProxyType(copied))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self.specs))

    def with_spec(self, spec: InteropAdapterSpec) -> "InteropRegistry":
        if spec.name in self.specs:
            raise InteropRegistryError(
                f"interop adapter {spec.name!r} is already registered"
            )
        return InteropRegistry({**self.specs, spec.name: spec})

    def spec(self, name: str) -> InteropAdapterSpec:
        try:
            return self.specs[str(name)]
        except KeyError as exc:
            available = ", ".join(self.names) or "none"
            raise InteropRegistryError(
                f"unknown interop adapter {name!r}; available: {available}"
            ) from exc

    def load(self, name: str) -> InteropAdapter:
        return self.spec(name).load()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "flagquantum_interop_registry_v1",
            "api_version": INTEROP_API_VERSION,
            "adapters": [self.specs[name].to_dict() for name in self.names],
        }


QISKIT_ADAPTER_SPEC = InteropAdapterSpec(
    name="qiskit",
    module="flagquantum.ecosystem.qiskit.adapter",
    attribute="QISKIT_ADAPTER",
    dependency_extra="qiskit",
)
PENNYLANE_ADAPTER_SPEC = InteropAdapterSpec(
    name="pennylane",
    module="flagquantum.ecosystem.pennylane.adapter",
    attribute="PENNYLANE_ADAPTER",
    dependency_extra="pennylane",
)
DEFAULT_INTEROP_REGISTRY = InteropRegistry(
    {"pennylane": PENNYLANE_ADAPTER_SPEC, "qiskit": QISKIT_ADAPTER_SPEC}
)


def available_adapters() -> tuple[str, ...]:
    """Return registered adapter names without importing their frameworks."""

    return DEFAULT_INTEROP_REGISTRY.names


def get_adapter(name: str) -> InteropAdapter:
    """Load one adapter implementation without importing its framework."""

    return DEFAULT_INTEROP_REGISTRY.load(name)


__all__ = (
    "DEFAULT_INTEROP_REGISTRY",
    "PENNYLANE_ADAPTER_SPEC",
    "QISKIT_ADAPTER_SPEC",
    "InteropAdapterSpec",
    "InteropRegistry",
    "InteropRegistryError",
    "available_adapters",
    "get_adapter",
)
