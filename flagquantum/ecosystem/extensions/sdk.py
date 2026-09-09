"""Public contracts for third-party FlagQuantum extensions."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from importlib import metadata
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Protocol, Sequence, runtime_checkable

from ...core.ir import CircuitIR
from ...errors import CapabilityError, ExecutionError, FlagQuantumError

SDK_API_VERSION = "1.0"
EXTENSION_ENTRY_POINT_GROUP = "flagquantum.extensions"
EXTENSION_KINDS = frozenset(
    {
        "backend",
        "compiler",
        "kernel",
        "operator",
        "compiler_pass",
        "device",
        "provider",
        "measurement_collector",
        "planner",
    }
)
SECRET_PARTS = ("token", "secret", "password", "credential", "api_key", "apikey")


class ExtensionError(RuntimeError, FlagQuantumError):
    """Base class for extension boundary failures."""


class ExtensionCompatibilityError(ExtensionError, CapabilityError):
    """Raised before activation for an incompatible extension."""


class ExtensionLifecycleError(ExtensionError, ExecutionError):
    """Raised when activation, use, or cleanup fails safely."""


def _frozen_map(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(value))


@dataclass(frozen=True)
class ExtensionManifest:
    """Serializable identity, compatibility, and honest capability declaration."""

    name: str
    version: str
    kind: str
    api_version: str = SDK_API_VERSION
    capabilities: frozenset[str] = frozenset()
    experimental: bool = True
    deprecation_version: str | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.version:
            raise ValueError("extension name and version must be non-empty")
        if self.kind not in EXTENSION_KINDS:
            raise ValueError(f"unknown extension kind {self.kind!r}")
        if not all(item and item == item.lower() for item in self.capabilities):
            raise ValueError("capability names must be non-empty lowercase strings")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "kind": self.kind,
            "api_version": self.api_version,
            "capabilities": sorted(self.capabilities),
            "experimental": self.experimental,
            "deprecation_version": self.deprecation_version,
        }


@dataclass(frozen=True)
class ExtensionConfig:
    """Sanitized configuration. Credentials must stay in host-owned resolvers."""

    values: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = dict(self.values)
        forbidden = sorted(
            key
            for key in values
            if any(part in key.lower().replace("-", "_") for part in SECRET_PARTS)
        )
        if forbidden:
            raise ValueError(
                "extension configuration cannot contain credential values: "
                + ", ".join(forbidden)
                + "; use a host-owned credential resolver"
            )
        object.__setattr__(self, "values", _frozen_map(values))

    def to_dict(self) -> dict[str, Any]:
        return dict(self.values)


@dataclass(frozen=True)
class CapabilityRequest:
    required: frozenset[str] = frozenset()
    dtype: str | None = None
    device_type: str | None = None
    require_gradients: bool = False


@dataclass(frozen=True)
class CapabilityResponse:
    accepted: bool
    supported: frozenset[str]
    blockers: tuple[str, ...] = ()


@runtime_checkable
class Extension(Protocol):
    manifest: ExtensionManifest

    def negotiate(self, request: CapabilityRequest) -> CapabilityResponse: ...

    def start(self, config: ExtensionConfig) -> None: ...

    def close(self) -> None: ...


@runtime_checkable
class ExecutionBackendExtension(Extension, Protocol):
    def execute(self, program: Any, parameters: Any = None) -> Any: ...


@runtime_checkable
class KernelExtension(Extension, Protocol):
    def value_and_grad(self, inputs: Any) -> tuple[Any, Any]: ...


@runtime_checkable
class OperatorExtension(Extension, Protocol):
    def schemas(self) -> Sequence[Mapping[str, Any]]: ...


@runtime_checkable
class CompilerPassExtension(Extension, Protocol):
    def transform(self, program: Any) -> Any: ...


@runtime_checkable
class CompilerExtension(Extension, Protocol):
    """Circuit-level compiler that consumes and returns FlagQuantum IR."""

    def compile(
        self,
        program: CircuitIR,
        *,
        target: Mapping[str, Any] | None = None,
    ) -> CircuitIR: ...


@runtime_checkable
class DeviceExtension(Extension, Protocol):
    def devices(self) -> Sequence[Mapping[str, Any]]: ...


@runtime_checkable
class ProviderExtension(Extension, Protocol):
    def submit(self, payload: Mapping[str, Any]) -> str: ...

    def status(self, handle: str) -> str: ...


@runtime_checkable
class MeasurementCollectorExtension(Extension, Protocol):
    def collect(self) -> Mapping[str, Any]: ...


@runtime_checkable
class PlannerExtension(Extension, Protocol):
    def plan(self, program: Any, request: CapabilityRequest) -> Any: ...


@dataclass
class ExtensionHandle:
    """Lifecycle wrapper that contains extension exceptions and always cleans up."""

    extension: Extension
    response: CapabilityResponse
    active: bool = False

    def start(self, config: ExtensionConfig) -> None:
        if self.active:
            raise ExtensionLifecycleError(
                f"extension {self.extension.manifest.name!r} is already active"
            )
        try:
            self.extension.start(config)
        except Exception as exc:
            try:
                self.extension.close()
            except Exception:
                pass
            raise ExtensionLifecycleError(
                f"extension {self.extension.manifest.name!r} failed to start: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        self.active = True

    def invoke(self, method: str, *args: Any, **kwargs: Any) -> Any:
        if not self.active:
            raise ExtensionLifecycleError("extension is not active")
        try:
            return getattr(self.extension, method)(*args, **kwargs)
        except Exception as exc:
            raise ExtensionLifecycleError(
                f"extension {self.extension.manifest.name!r} {method} failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    def close(self) -> None:
        if not self.active:
            return
        self.active = False
        try:
            self.extension.close()
        except Exception as exc:
            raise ExtensionLifecycleError(
                f"extension {self.extension.manifest.name!r} cleanup failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc


@dataclass(frozen=True)
class ExtensionRegistry:
    """Immutable registry; additions return a new registry."""

    entries: Mapping[tuple[str, str], Extension] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "entries", _frozen_map(self.entries))

    def with_extension(self, extension: Extension) -> "ExtensionRegistry":
        if not isinstance(extension, Extension):
            raise TypeError("extension does not implement the SDK lifecycle protocol")
        manifest = extension.manifest
        if manifest.api_version != SDK_API_VERSION:
            raise ExtensionCompatibilityError(
                f"extension {manifest.name!r} requires SDK API "
                f"{manifest.api_version}; FlagQuantum provides {SDK_API_VERSION}. "
                "Install a compatible extension or upgrade FlagQuantum."
            )
        key = (manifest.kind, manifest.name)
        if key in self.entries:
            raise ExtensionCompatibilityError(
                f"extension {manifest.kind}/{manifest.name} is already registered"
            )
        updated = dict(self.entries)
        updated[key] = extension
        return ExtensionRegistry(updated)

    def negotiate(
        self, kind: str, name: str, request: CapabilityRequest
    ) -> ExtensionHandle:
        try:
            extension = self.entries[(kind, name)]
        except KeyError as exc:
            raise ExtensionCompatibilityError(
                f"extension {kind}/{name} is not registered in this scope"
            ) from exc
        try:
            response = extension.negotiate(request)
        except Exception as exc:
            raise ExtensionCompatibilityError(
                f"extension {kind}/{name} capability negotiation failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        missing = request.required - response.supported
        blockers = list(response.blockers)
        if missing:
            blockers.append("missing capabilities: " + ", ".join(sorted(missing)))
        if not response.accepted or blockers:
            raise ExtensionCompatibilityError(
                f"extension {kind}/{name} rejected capability request: "
                + "; ".join(blockers or ("request rejected",))
            )
        return ExtensionHandle(extension, response)


def discover_extensions(kind: str) -> ExtensionRegistry:
    """Discover installed extensions of one kind without activating them.

    Entry points use the ``flagquantum.extensions`` group and names of the form
    ``<kind>.<manifest-name>``. Their value must resolve to a zero-argument
    extension factory.
    """

    if kind not in EXTENSION_KINDS:
        raise ValueError(f"unknown extension kind {kind!r}")
    prefix = f"{kind}."
    registry = ExtensionRegistry()
    discovered = metadata.entry_points(group=EXTENSION_ENTRY_POINT_GROUP)
    for entry_point in sorted(discovered, key=lambda item: item.name):
        if not entry_point.name.startswith(prefix):
            continue
        manifest_name = entry_point.name.removeprefix(prefix)
        if not manifest_name:
            raise ExtensionCompatibilityError(
                f"entry point {entry_point.name!r} has no extension name"
            )
        try:
            factory = entry_point.load()
            extension = factory()
        except Exception as exc:
            raise ExtensionCompatibilityError(
                f"extension entry point {entry_point.name!r} failed to load: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        manifest = getattr(extension, "manifest", None)
        if manifest is None:
            raise ExtensionCompatibilityError(
                f"extension entry point {entry_point.name!r} returned an object "
                "without a manifest"
            )
        if manifest.kind != kind or manifest.name != manifest_name:
            raise ExtensionCompatibilityError(
                f"entry point {entry_point.name!r} does not match manifest "
                f"{manifest.kind}/{manifest.name}"
            )
        registry = registry.with_extension(extension)
    return registry


def compile_with_extension(
    program: CircuitIR,
    *,
    extension: str,
    target: Mapping[str, Any] | None = None,
) -> CircuitIR:
    """Compile FlagQuantum IR with one explicitly selected installed extension."""

    if not isinstance(program, CircuitIR):
        raise TypeError("program must be a CircuitIR")
    handle = discover_extensions("compiler").negotiate(
        "compiler",
        extension,
        CapabilityRequest(required=frozenset({"circuit_ir"})),
    )
    handle.start(ExtensionConfig())
    try:
        compiled = handle.invoke("compile", program, target=target)
        if not isinstance(compiled, CircuitIR):
            raise ExtensionLifecycleError(
                f"extension {extension!r} returned {type(compiled).__name__}; "
                "expected CircuitIR"
            )
        return compiled
    finally:
        handle.close()


_REGISTRY: ContextVar[ExtensionRegistry] = ContextVar(
    "flagquantum_extension_registry", default=ExtensionRegistry()
)


@contextmanager
def extension_scope(
    extensions: Sequence[Extension],
) -> Iterator[ExtensionRegistry]:
    """Install extensions only for the current task and restore exactly."""

    registry = _REGISTRY.get()
    for extension in extensions:
        registry = registry.with_extension(extension)
    token = _REGISTRY.set(registry)
    try:
        yield registry
    finally:
        _REGISTRY.reset(token)
