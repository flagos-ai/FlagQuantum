"""Admission of an extension-provided execution backend into the Runtime registry.

The extension SDK owns discovery, negotiation, manifest validation, and lifecycle
isolation. Runtime owns one authoritative capability registry. This module is the
single crossing between them: an extension that declares ``backend_capabilities``
is validated here and registered through
:func:`flagquantum.runtime.backend_registry.register_backend`. It creates no
registry, manifest format, or entry-point group of its own.

The registered route is called as ``execute(program, *, options)``. That keyword
is what distinguishes an executable backend route from
:class:`flagquantum.ecosystem.extensions.ExecutionBackendExtension`, which only
states that an extension of kind ``backend`` exposes an ``execute`` member.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ...errors import CapabilityError
from ...runtime.backend_registry import (
    BUILTIN_BACKEND_NAMES,
    BackendCapabilities,
    BackendExecutor,
    register_backend,
    resolve_backend_executor,
    unregister_backend,
)
from .sdk import (
    CapabilityRequest,
    ExecutionBackendExtension,
    Extension,
    ExtensionConfig,
    ExtensionError,
    ExtensionHandle,
    ExtensionRegistry,
)

DECLARATION_MEMBER = "backend_capabilities"

# Host facts an extension must not choose. A declared name is the backend's
# public identity, and the route is the extension object itself, so both belong
# to the host that admits the extension rather than to the declaration.
HOST_OWNED_DECLARATION_KEYS = frozenset({"name", "executor", "accelerators"})


class BackendAdmissionError(ExtensionError, CapabilityError):
    """An extension backend cannot be admitted exactly as declared."""


def _declared(extension: Extension) -> Mapping[str, Any]:
    manifest = extension.manifest
    if manifest.kind != "backend":
        raise BackendAdmissionError(
            f"extension {manifest.name!r} has kind {manifest.kind!r}; only a "
            "backend extension can be admitted as an execution route"
        )
    if not isinstance(extension, ExecutionBackendExtension):
        raise BackendAdmissionError(
            f"backend extension {manifest.name!r} does not implement execute"
        )
    declaration = getattr(extension, DECLARATION_MEMBER, None)
    if declaration is None:
        raise BackendAdmissionError(
            f"backend extension {manifest.name!r} must declare "
            f"{DECLARATION_MEMBER!r} so Runtime learns its capabilities before "
            "planning"
        )
    if not isinstance(declaration, Mapping):
        raise BackendAdmissionError(
            f"backend extension {manifest.name!r} must declare "
            f"{DECLARATION_MEMBER!r} as a mapping of BackendCapabilities fields, "
            f"got {type(declaration).__name__}"
        )
    declared = dict(declaration)
    host_owned = sorted(HOST_OWNED_DECLARATION_KEYS.intersection(declared))
    if host_owned:
        raise BackendAdmissionError(
            f"backend extension {manifest.name!r} declared host-owned capability "
            "fields: " + ", ".join(host_owned)
        )
    unknown = sorted(set(declared) - set(BackendCapabilities.__dataclass_fields__))
    if unknown:
        raise BackendAdmissionError(
            f"backend extension {manifest.name!r} declared unknown capability "
            "fields: " + ", ".join(unknown)
        )
    if manifest.name.lower() in BUILTIN_BACKEND_NAMES:
        raise BackendAdmissionError(
            f"backend extension {manifest.name!r} may not take a built-in backend "
            "name; a built-in execution route cannot be replaced"
        )
    return declared


def _build(extension: Extension, declared: Mapping[str, Any]) -> BackendCapabilities:
    if not isinstance(extension, BackendExecutor):
        raise BackendAdmissionError(
            f"backend extension {extension.manifest.name!r} must implement "
            "execute(program, *, options) and close()"
        )
    try:
        return BackendCapabilities(
            name=extension.manifest.name, **declared, executor=extension
        )
    except TypeError as exc:
        raise BackendAdmissionError(
            f"backend extension {extension.manifest.name!r} declared incomplete "
            f"capabilities: {exc}"
        ) from exc
    except ValueError as exc:
        raise BackendAdmissionError(
            f"backend extension {extension.manifest.name!r} declared invalid "
            f"capabilities: {exc}"
        ) from exc


def _negotiate(extension: Extension, declared: Mapping[str, Any]):
    devices = tuple(declared.get("devices", ()))
    dtypes = tuple(declared.get("dtypes", ()))
    request = CapabilityRequest(
        dtype=dtypes[0] if dtypes else None,
        device_type=devices[0] if devices else None,
        require_gradients=bool(declared.get("supports_autograd", False)),
    )
    try:
        return extension.negotiate(request)
    except Exception as exc:
        raise BackendAdmissionError(
            f"backend extension {extension.manifest.name!r} failed capability "
            f"negotiation: {type(exc).__name__}: {exc}"
        ) from exc


def _version_checked(extension: Extension) -> None:
    """Reuse the SDK's own SDK-version check rather than restating it.

    An incompatible extension must be rejected before activation, and the
    authority for that decision is ``ExtensionRegistry``. An empty transient
    registry is not a second registry; it is the existing one performing the
    check it already owns.
    """

    ExtensionRegistry().with_extension(extension)


def _validated(extension: Extension) -> tuple[BackendCapabilities, Any]:
    _version_checked(extension)
    declared = _declared(extension)
    capabilities = _build(extension, declared)
    response = _negotiate(extension, declared)
    if not response.accepted:
        raise BackendAdmissionError(
            f"backend extension {extension.manifest.name!r} rejected the "
            "capability request it declared: " + "; ".join(response.blockers)
        )
    return capabilities, response


def check_backend_admission(extension: Extension) -> BackendCapabilities:
    """Validate one declaration, including negotiation, without registering it.

    A host uses this before admission to reject an unusable backend, and a
    conformance run uses it to check a third-party backend without granting it an
    execution route.

    Raises:
        ExtensionCompatibilityError: The extension requires an SDK API version
            this FlagQuantum does not provide.
        BackendAdmissionError: The declaration is missing, malformed, incomplete,
            names a built-in backend, or the extension refuses the request.
    """

    capabilities, _ = _validated(extension)
    return capabilities


def admit_backend_extension(
    extension: Extension,
    *,
    config: ExtensionConfig | None = None,
) -> tuple[ExtensionHandle, BackendCapabilities]:
    """Validate, negotiate, start, and register one extension execution backend.

    Admission happens once, at activation. Runtime never discovers, negotiates,
    or version-checks an extension while planning or executing; it only looks up
    the route that admission installed. The caller closes the returned handle,
    which restores the extension's lifecycle exactly.

    Returns:
        The SDK lifecycle handle and the registered capability record.

    Raises:
        ExtensionCompatibilityError: The extension requires an SDK API version
            this FlagQuantum does not provide.
        BackendAdmissionError: The declaration is unusable, or the extension
            could not be activated.
    """

    capabilities, response = _validated(extension)
    if resolve_backend_executor(capabilities.name) is not None:
        raise BackendAdmissionError(
            f"backend {capabilities.name!r} already has an admitted execution "
            "route; withdraw it with withdraw_backend before admitting another one, "
            "so a live route is never displaced silently"
        )
    handle = ExtensionHandle(extension, response)
    try:
        handle.start(config or ExtensionConfig())
    except Exception as exc:
        raise BackendAdmissionError(str(exc)) from exc
    register_backend(capabilities)
    return handle, capabilities


def withdraw_backend(name: str) -> bool:
    """Withdraw one admitted execution route. Returns whether one was removed.

    A route is withdrawn when its extension closes, otherwise a later task would
    keep executing through a lifecycle that already ended.
    """

    return unregister_backend(name)


__all__ = (
    "DECLARATION_MEMBER",
    "HOST_OWNED_DECLARATION_KEYS",
    "BackendAdmissionError",
    "admit_backend_extension",
    "check_backend_admission",
    "withdraw_backend",
)
