"""Admission of an extension-provided compiler pass into the Compiler pipeline.

The extension SDK owns discovery, negotiation, manifest validation, and lifecycle
isolation. The Compiler owns one authoritative pass registry
(:mod:`flagquantum.compiler.pass_manager`). This module is the crossing between
them, next to :mod:`flagquantum.ecosystem.extensions.admission`, which performs the
same crossing for an execution backend. It creates no registry, manifest format,
protocol, or entry-point group of its own.

An extension pass registers under a host-built name of the form
``extension.<manifest-name>.<declared name>``, and the built-in pass names are
reserved, so admitting an extension can add a pass and can never redefine one that
``compile`` is defined by.

The declared route is called as ``transform(program)``. That is the member
:class:`flagquantum.ecosystem.extensions.CompilerPassExtension` already requires, so
an admitted pass and an in-repository pass are called the same way and a compile
path that runs one can run the other without knowing which it has.
"""

from __future__ import annotations

from typing import Any

from ...compiler.pass_manager import (
    OPTIMIZATION_PIPELINE,
    PassFunction,
    PassManager,
    PassRegistry,
    PassRegistryError,
    extension_pass_name,
)
from ...compiler.pipeline import default_pass_registry
from ...core.ir import CircuitIR
from ...errors import CapabilityError
from .sdk import (
    CapabilityRequest,
    CapabilityResponse,
    CompilerPassExtension,
    Extension,
    ExtensionConfig,
    ExtensionError,
    ExtensionHandle,
    ExtensionRegistry,
)

PASS_DECLARATION_MEMBER = "pass_name"

# The capability a pass declares to say it consumes and returns FlagQuantum IR.
# Requiring it rather than assuming it is what keeps one crossing from admitting a
# pass written against some other program representation.
IR_CAPABILITY = "circuit_ir"


class PassAdmissionError(ExtensionError, CapabilityError):
    """An extension compiler pass cannot be admitted exactly as declared."""


def _declared_name(extension: Extension) -> str:
    manifest = extension.manifest
    if manifest.kind != "compiler_pass":
        raise PassAdmissionError(
            f"extension {manifest.name!r} has kind {manifest.kind!r}; only a "
            "compiler_pass extension can be admitted as a Compiler pass"
        )
    if not isinstance(extension, CompilerPassExtension):
        raise PassAdmissionError(
            f"compiler_pass extension {manifest.name!r} does not implement "
            "transform(program)"
        )
    declared = getattr(extension, PASS_DECLARATION_MEMBER, None)
    if declared is None:
        raise PassAdmissionError(
            f"compiler_pass extension {manifest.name!r} must declare "
            f"{PASS_DECLARATION_MEMBER!r} so its pass has a name to be registered "
            "and traced under"
        )
    if not isinstance(declared, str) or not declared.strip():
        raise PassAdmissionError(
            f"compiler_pass extension {manifest.name!r} declared "
            f"{PASS_DECLARATION_MEMBER!r} as {type(declared).__name__}; a pass name "
            "must be a non-empty string"
        )
    try:
        return extension_pass_name(manifest.name, declared)
    except PassRegistryError as exc:
        raise PassAdmissionError(
            f"compiler_pass extension {manifest.name!r} declared an unusable pass "
            f"name: {exc}"
        ) from exc


def _version_checked(extension: Extension) -> None:
    """Reuse the SDK's own SDK-version check rather than restating it.

    An incompatible extension must be rejected before activation, and the
    authority for that decision is ``ExtensionRegistry``. An empty transient
    registry is not a second registry; it is the existing one performing the check
    it already owns.
    """

    ExtensionRegistry().with_extension(extension)


def _negotiate(extension: Extension) -> CapabilityResponse:
    request = CapabilityRequest(required=frozenset({IR_CAPABILITY}))
    try:
        return extension.negotiate(request)
    except Exception as exc:
        raise PassAdmissionError(
            f"compiler_pass extension {extension.manifest.name!r} failed capability "
            f"negotiation: {type(exc).__name__}: {exc}"
        ) from exc


def _validated(extension: Extension) -> tuple[str, CapabilityResponse]:
    _version_checked(extension)
    name = _declared_name(extension)
    response = _negotiate(extension)
    if not response.accepted:
        raise PassAdmissionError(
            f"compiler_pass extension {extension.manifest.name!r} rejected the "
            "capability request it declares: " + "; ".join(response.blockers)
        )
    missing = {IR_CAPABILITY} - set(response.supported)
    if missing:
        raise PassAdmissionError(
            f"compiler_pass extension {extension.manifest.name!r} does not serve "
            "FlagQuantum IR; it declares no support for " + ", ".join(sorted(missing))
        )
    if name in default_pass_registry().passes:
        raise PassAdmissionError(
            f"compiler_pass extension {extension.manifest.name!r} would register "
            f"over the live pass {name!r}; a live pass is never displaced silently"
        )
    return name, response


def check_pass_admission(extension: Extension) -> str:
    """Validate one declaration, including negotiation, without admitting it.

    A host uses this before admission to reject an unusable pass, and a
    conformance run uses it to check a third-party pass without granting it a place
    in a registry.

    Returns:
        The registered name the pass would receive.

    Raises:
        ExtensionCompatibilityError: The extension requires an SDK API version
            this FlagQuantum does not provide.
        PassAdmissionError: The declaration is missing, malformed, or the
            extension refuses the request.
    """

    return _validated(extension)[0]


def admit_pass_extension(
    extension: Extension,
    *,
    config: ExtensionConfig | None = None,
) -> tuple[ExtensionHandle, PassRegistry]:
    """Validate, negotiate, start, and register one extension compiler pass.

    The returned registry is the built-in registry plus this pass. Compilation
    never discovers, negotiates, or version-checks an extension while optimizing;
    it resolves a name. The caller closes the returned handle, which restores the
    extension's lifecycle exactly.

    Returns:
        The SDK lifecycle handle and a registry that resolves the admitted pass
        alongside the built-in passes.

    Raises:
        ExtensionCompatibilityError: The extension requires an SDK API version
            this FlagQuantum does not provide.
        PassAdmissionError: The declaration is unusable, or the extension could
            not be activated.
    """

    name, response = _validated(extension)
    handle = ExtensionHandle(extension, response)
    try:
        handle.start(config or ExtensionConfig())
    except Exception as exc:
        raise PassAdmissionError(str(exc)) from exc
    try:
        registry = default_pass_registry().with_pass(name, _route(handle))
    except Exception:
        handle.close()
        raise
    return handle, registry


def _route(handle: ExtensionHandle) -> PassFunction:
    """A pass function that runs through the handle, so the lifecycle still holds.

    The adapter closes over the handle rather than the extension object, so a pass
    invoked after its extension closed fails closed instead of executing through a
    lifecycle that already ended. It adds no validation of its own: whether the
    transform returned FlagQuantum IR is the manager's check, and the manager names
    this pass when that check fails.
    """

    def transform(program: CircuitIR) -> CircuitIR:
        result: Any = handle.invoke("transform", program)
        return result  # type: ignore[no-any-return]

    return transform


def admitted_pass_name(registry: PassRegistry) -> str:
    """The single pass a registry built by admission resolves beyond the built-ins.

    Raises:
        PassAdmissionError: The registry resolves no extension pass, or more than
            one, so there is no unambiguous pass to run.
    """

    names = registry.extension_names
    if len(names) != 1:
        raise PassAdmissionError(
            "expected exactly one admitted extension pass, found "
            + (", ".join(names) if names else "none")
        )
    return names[0]


def optimize_with_extension_pass(
    program: CircuitIR,
    *,
    extension: Extension,
    config: ExtensionConfig | None = None,
) -> CircuitIR:
    """Optimize FlagQuantum IR through the built-in pipeline and one extension pass.

    This is the reach the crossing exists for: the built-in pipeline runs to its
    own fixed point, then the admitted pass runs once, and nothing in the Compiler
    is modified to run either. The pass runs after the pipeline rather than inside
    it, so what it produced is exactly what is returned and a pass that is not
    idempotent cannot hold the pipeline in an unbounded loop.

    The extension is activated for this call and closed again, so an admitted pass
    never outlives the task that asked for it.

    Raises:
        PassAdmissionError: The extension cannot be admitted.
        PassRegistryError: The built-in registry does not resolve the fixed
            pipeline, which would mean the pipeline and the registry disagree.
        CompilationError: A pass, including the admitted one, did not return
            FlagQuantum IR.
    """

    handle, registry = admit_pass_extension(extension, config=config)
    try:
        name = admitted_pass_name(registry)
        manager = PassManager(registry)
        optimized = manager.to_fixed_point(program, OPTIMIZATION_PIPELINE)
        return manager.run(optimized, (name,))
    finally:
        handle.close()


__all__ = (
    "IR_CAPABILITY",
    "PASS_DECLARATION_MEMBER",
    "PassAdmissionError",
    "admit_pass_extension",
    "admitted_pass_name",
    "check_pass_admission",
    "optimize_with_extension_pass",
)
