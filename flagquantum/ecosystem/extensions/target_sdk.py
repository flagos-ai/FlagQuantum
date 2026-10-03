"""Third-party target description through a ``device`` extension.

A provider reaches FlagQuantum through an extension of kind ``device`` whose
``devices()`` member returns one description per device it offers.  This module
is the boundary that reads those descriptions, validates them, and converts them
into the authoritative Core Target Capabilities v1 snapshot and connectivity
that Compiler and Runtime already consume.

It deliberately adds no registry, no capability vocabulary, and no second
compile or execution path:

* validation produces :class:`TargetCapabilitySnapshot` values from
  :mod:`flagquantum.core.target_capabilities`, so the matcher, the evidence
  rules, and the failure codes stay where they already live;
* connectivity is returned as normalised undirected edges that the caller turns
  into the Compiler-owned ``CouplingMap`` with one constructor call, which keeps
  this package free of a Compiler dependency;
* compilation stays
  :func:`flagquantum.compiler.target_legalization.legalize_circuit_for_target`
  with ``backend="provider"``, and execution stays an already-admitted backend
  route plus a :class:`flagquantum.noise.NoiseModel`.

:class:`TargetDescription` is a new type only because no existing authoritative
type takes this input.  A snapshot is the *result* of validation and carries no
connectivity, and ``CloudBackendProfile`` belongs to the Remote domain and
expresses no evidence, dtype, shot, or program-operation facts.  See
``docs/reference/EXTENSION_SDK.md`` for the worked path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from math import isfinite
from types import MappingProxyType
from typing import Any

from ...core.operator_schema import canonical_opcode, get_operator_schema
from ...core.target_capabilities import (
    AUTHORITATIVE_STATIC_DECLARATION_ALLOWED,
    CAPABILITY_NAMES,
    CapabilityBlocker,
    CapabilityFact,
    CapabilityScope,
    EvidenceLevel,
    EvidenceReference,
    FactExposure,
    FactSource,
    SupportStatus,
    TargetCapabilitySnapshot,
    TargetIdentity,
)
from ...errors import CapabilityError
from .sdk import (
    DeviceExtension,
    Extension,
    ExtensionError,
    ExtensionManifest,
    ExtensionRegistry,
)

TARGET_DESCRIPTION_MEMBER = "devices"

# A provider identity is the host's to assign: the admitted manifest name and
# version are the provider, so a declaration that restates or contradicts them
# is refused rather than silently overridden.
HOST_OWNED_DESCRIPTION_KEYS = frozenset({"provider", "provider_version"})

# Capability names a dedicated field supplies, so a declaration cannot state a
# second, possibly contradicting value for them.
_DERIVED_FACT_NAMES = frozenset(
    {"device.kind", "target.class", "qubits.logical_capacity", "gates.native"}
)

_OBSERVED_SOURCE_KIND = "provider_observation"
_DECLARED_SOURCE_KIND = "provider_declaration"

_NUMERIC_FACT_NAMES = frozenset(
    {
        "device.count",
        "memory.available_bytes",
        "qubits.logical_capacity",
        "qubits.physical_capacity",
        "limits.maximum_shots",
        "limits.maximum_program_operations",
        "ancillas.maximum_compiler",
    }
)
_COLLECTION_FACT_NAMES = frozenset(
    {"gates.native", "measurements.results", "artifacts.profiles"}
)


class TargetDescriptionError(ExtensionError, CapabilityError):
    """A device extension's target declaration is unusable exactly as declared."""


def _non_empty(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TargetDescriptionError(
            f"target description {field_name} must be a non-empty string"
        )
    return value


def _freeze_declared_json(value: Any, *, path: str) -> Any:
    """Deep-copy JSON-safe declaration values into immutable containers."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise TargetDescriptionError(
                f"target description {path} must contain only finite numbers"
            )
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key in value:
            if not isinstance(key, str):
                raise TargetDescriptionError(
                    f"target description {path} mapping keys must be strings"
                )
            frozen[key] = _freeze_declared_json(value[key], path=f"{path}.{key}")
        return MappingProxyType({key: frozen[key] for key in sorted(frozen)})
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_declared_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    raise TargetDescriptionError(
        f"target description {path} must contain only JSON-safe values, "
        f"got {type(value).__name__}"
    )


def _validate_fact_value(name: str, value: Any, *, field_name: str) -> None:
    if value is None:
        return
    if name in _NUMERIC_FACT_NAMES:
        if type(value) is not int or value < 0:
            raise TargetDescriptionError(
                f"target description {field_name} fact {name} must be a "
                "non-negative integer"
            )
    elif name in _COLLECTION_FACT_NAMES:
        if not isinstance(value, tuple):
            raise TargetDescriptionError(
                f"target description {field_name} fact {name} must be an array"
            )
    elif not isinstance(value, str):
        raise TargetDescriptionError(
            f"target description {field_name} fact {name} must be a string"
        )


def _freeze_facts(values: Any, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(values, Mapping):
        raise TargetDescriptionError(
            f"target description {field_name} must be a mapping"
        )
    unknown = sorted(
        name
        for name in values
        if not isinstance(name, str) or name not in CAPABILITY_NAMES
    )
    if unknown:
        # Named before the value shape is judged, so a misspelled capability is
        # reported as misspelled instead of as a value of the wrong type.
        raise TargetDescriptionError(
            "target description contains unknown v1 capability names: "
            + ", ".join(repr(name) for name in unknown)
        )
    frozen: dict[str, Any] = {}
    for name, value in values.items():
        frozen_value = _freeze_declared_json(value, path=f"{field_name}[{name!r}]")
        _validate_fact_value(name, frozen_value, field_name=field_name)
        frozen[name] = frozen_value
    return MappingProxyType({name: frozen[name] for name in sorted(frozen)})


def _native_gates(value: Any) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TargetDescriptionError(
            "target description native_gates must be a sequence of gate names"
        )
    canonical: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise TargetDescriptionError(
                "target description native_gates must be non-empty strings"
            )
        name = canonical_opcode(item)
        if get_operator_schema(name) is None:
            raise TargetDescriptionError(
                f"target description native_gates names unknown FlagQuantum gate "
                f"{item!r}; a provider gate set must be expressed with the "
                "operator schemas the compiler can lower"
            )
        canonical.append(name)
    if not canonical:
        raise TargetDescriptionError(
            "target description native_gates must name at least one gate"
        )
    if len(set(canonical)) != len(canonical):
        raise TargetDescriptionError(
            "target description native_gates must not repeat a gate"
        )
    return tuple(sorted(canonical))


def _coupling_edges(value: Any, *, n_qubits: int) -> tuple[tuple[int, int], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TargetDescriptionError(
            "target description coupling_edges must be a sequence of wire pairs"
        )
    normalized: set[tuple[int, int]] = set()
    for item in value:
        if isinstance(item, (str, bytes)) or not isinstance(item, Sequence):
            raise TargetDescriptionError(
                "target description coupling_edges entries must be wire pairs"
            )
        pair = tuple(item)
        if len(pair) != 2:
            raise TargetDescriptionError(
                "target description coupling_edges entries must have two wires"
            )
        wires: list[int] = []
        for wire in pair:
            if type(wire) is not int:
                raise TargetDescriptionError(
                    "target description coupling_edges wires must be integers"
                )
            if not 0 <= wire < n_qubits:
                raise TargetDescriptionError(
                    f"target description coupling_edges wire {wire} is outside the "
                    f"declared {n_qubits}-qubit device"
                )
            wires.append(wire)
        left, right = wires
        if left == right:
            raise TargetDescriptionError(
                f"target description coupling_edges cannot couple wire {left} to "
                "itself"
            )
        normalized.add((min(left, right), max(left, right)))
    return tuple(sorted(normalized))


@dataclass(frozen=True)
class TargetDescription:
    """One validated device declaration, owned by FlagQuantum.

    Every fact is explicit.  ``device_kind``, ``target_class``, ``n_qubits``, and
    ``native_gates`` are the dedicated fields that supply ``device.kind``,
    ``target.class``, ``qubits.logical_capacity``, and ``gates.native``.  A fact
    the provider does not state stays absent instead of being inferred, so the
    Core matcher fails closed when it is required.

    ``n_qubits`` becomes an *observed* fact and ``target_class`` and
    ``native_gates`` become *declared* facts, because that is what each one is:
    the provider read the qubit count off the device it operates, and the gate set
    and target class are static properties it asserts about it.  The distinction
    matters, because a circuit requirement raises the evidence floor to
    observation grade and a declared qubit count would then be refused.
    """

    target_id: str
    device_id: str
    target_class: str
    device_kind: str
    target_revision: str
    environment_id: str
    n_qubits: int
    native_gates: tuple[str, ...]
    coupling_edges: tuple[tuple[int, int], ...]
    evidence_refs: tuple[EvidenceReference, ...]
    observed_source_ref: str = "provider-observation"
    declared_source_ref: str = "provider-declaration"
    observed_facts: Mapping[str, Any] = field(default_factory=dict)
    declared_facts: Mapping[str, Any] = field(default_factory=dict)
    provider: str = ""
    provider_version: str = ""

    def __post_init__(self) -> None:
        for name in (
            "target_id",
            "device_id",
            "target_class",
            "device_kind",
            "target_revision",
            "environment_id",
            "observed_source_ref",
            "declared_source_ref",
            "provider",
            "provider_version",
        ):
            _non_empty(getattr(self, name), field_name=name)
        if type(self.n_qubits) is not int or self.n_qubits <= 0:
            raise TargetDescriptionError(
                "target description n_qubits must be a positive integer"
            )
        object.__setattr__(self, "native_gates", _native_gates(self.native_gates))
        object.__setattr__(
            self,
            "coupling_edges",
            _coupling_edges(self.coupling_edges, n_qubits=self.n_qubits),
        )
        if not isinstance(self.evidence_refs, (tuple, list)):
            raise TargetDescriptionError(
                "target description evidence_refs must be a tuple or list of "
                "EvidenceReference values"
            )
        evidence_refs = tuple(self.evidence_refs)
        if not evidence_refs or any(
            not isinstance(item, EvidenceReference) for item in evidence_refs
        ):
            raise TargetDescriptionError(
                "target description evidence_refs must contain EvidenceReference "
                "values"
            )
        if len({item.evidence_id for item in evidence_refs}) != len(evidence_refs):
            raise TargetDescriptionError(
                "target description evidence_refs must be unique"
            )
        object.__setattr__(self, "evidence_refs", evidence_refs)
        for name in ("observed_facts", "declared_facts"):
            object.__setattr__(
                self, name, _freeze_facts(getattr(self, name), field_name=name)
            )
        _validate_declared_fact_names(self)

    @property
    def n_wires(self) -> int:
        """The ``CouplingMap`` wire count this declaration authorises."""

        return self.n_qubits


def _validate_declared_fact_names(description: TargetDescription) -> None:
    observed_names = set(description.observed_facts)
    declared_names = set(description.declared_facts)
    derived = sorted(_DERIVED_FACT_NAMES.intersection(observed_names | declared_names))
    if derived:
        raise TargetDescriptionError(
            "target description supplies these facts through a dedicated field: "
            + ", ".join(derived)
        )
    overlap = observed_names & declared_names
    if overlap:
        raise TargetDescriptionError(
            "target description cannot expose a fact as both observed and "
            "declared: " + ", ".join(sorted(overlap))
        )
    invalid_declared = declared_names - AUTHORITATIVE_STATIC_DECLARATION_ALLOWED
    if invalid_declared:
        raise TargetDescriptionError(
            "target description may only declare static v1 facts; observe these "
            "instead: " + ", ".join(sorted(invalid_declared))
        )


def _description_payload(extension: ExtensionManifest, raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise TargetDescriptionError(
            f"device extension {extension.name!r} must describe each device as a "
            f"mapping, got {type(raw).__name__}"
        )
    payload = dict(raw)
    host_owned = sorted(HOST_OWNED_DESCRIPTION_KEYS.intersection(payload))
    if host_owned:
        raise TargetDescriptionError(
            f"device extension {extension.name!r} declared host-owned fields: "
            + ", ".join(host_owned)
        )
    payload["provider"] = extension.name
    payload["provider_version"] = extension.version
    return payload


def check_target_description(extension: Extension) -> tuple[TargetDescription, ...]:
    """Validate one device extension's declaration without activating it.

    The declaration is the value of ``devices()``: one mapping per device.  A
    host uses this before offering a target to a user, and a conformance run
    uses it to check a provider without granting it anything.

    Returns:
        One validated description per declared device, in declared order.

    Raises:
        ExtensionCompatibilityError: The extension requires an SDK API version
            this FlagQuantum does not provide.
        TargetDescriptionError: The extension is not a device extension, exposes
            no usable ``devices()``, or one declaration is malformed, names a
            host-owned or non-authoritative fact, or names an unknown gate.
    """

    manifest = getattr(extension, "manifest", None)
    if not isinstance(manifest, ExtensionManifest):
        raise TargetDescriptionError(
            "device extension must expose an ExtensionManifest before its "
            "declaration can be validated"
        )
    if manifest.kind != "device":
        raise TargetDescriptionError(
            f"extension {manifest.name!r} has kind {manifest.kind!r}; only a "
            "device extension declares a target"
        )
    if not isinstance(extension, DeviceExtension):
        raise TargetDescriptionError(
            f"device extension {manifest.name!r} does not implement devices()"
        )
    # Reuse the SDK's own version check instead of restating it.
    ExtensionRegistry().with_extension(extension)
    declared = extension.devices()
    if isinstance(declared, (str, bytes)) or not isinstance(declared, Sequence):
        raise TargetDescriptionError(
            f"device extension {manifest.name!r} devices() must return a sequence "
            f"of mappings, got {type(declared).__name__}"
        )
    if not declared:
        raise TargetDescriptionError(
            f"device extension {manifest.name!r} declares no device"
        )
    descriptions: list[TargetDescription] = []
    for raw in declared:
        try:
            description = TargetDescription(**_description_payload(manifest, raw))
        except TargetDescriptionError as error:
            raise TargetDescriptionError(
                f"device extension {manifest.name!r} has an unusable declaration: "
                f"{error}"
            ) from error
        except TypeError as error:
            raise TargetDescriptionError(
                f"device extension {manifest.name!r} declaration has unknown or "
                f"missing fields: {error}"
            ) from error
        descriptions.append(description)
    device_ids = [item.device_id for item in descriptions]
    if len(set(device_ids)) != len(device_ids):
        raise TargetDescriptionError(
            f"device extension {manifest.name!r} declares a repeated device_id"
        )
    return tuple(descriptions)


def _isoformat(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise TargetDescriptionError(
            "target capability snapshot captured_at must be timezone-aware"
        )
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _missing_blocker(name: str) -> CapabilityBlocker:
    return CapabilityBlocker(
        code=f"provider_{name.replace('.', '_')}_not_exposed",
        message=(
            "device declaration did not state this capability fact; the producer "
            "will not infer it"
        ),
        capability_name=name,
    )


def target_capability_snapshot(
    description: TargetDescription,
    *,
    captured_at: datetime | None = None,
    ttl: timedelta = timedelta(minutes=5),
) -> TargetCapabilitySnapshot:
    """Convert one validated description into a Core Target Capabilities v1 snapshot.

    Examples:
        >>> import hashlib
        >>> from flagquantum.core.target_capabilities import (
        ...     CapabilityScope,
        ...     EvidenceLevel,
        ...     EvidenceReference,
        ... )
        >>> from flagquantum.ecosystem.extensions.target_sdk import (
        ...     TargetDescription,
        ...     target_capability_snapshot,
        ... )
        >>> scope = CapabilityScope(device_ids=("acme-q0",))
        >>> description = TargetDescription(
        ...     provider="acme",
        ...     provider_version="1.0.0",
        ...     target_id="acme-4q",
        ...     device_id="acme-q0",
        ...     target_class="acme_superconducting",
        ...     device_kind="superconducting",
        ...     target_revision="r7",
        ...     environment_id="lab-a",
        ...     n_qubits=4,
        ...     native_gates=("h", "cx", "rz"),
        ...     coupling_edges=((0, 1), (1, 2), (2, 3)),
        ...     evidence_refs=(
        ...         EvidenceReference(
        ...             evidence_id="provider-observation",
        ...             sha256=hashlib.sha256(b"acme-bench").hexdigest(),
        ...             level=EvidenceLevel.OBSERVABLE,
        ...             scope=scope,
        ...         ),
        ...         EvidenceReference(
        ...             evidence_id="provider-declaration",
        ...             sha256=hashlib.sha256(b"acme-spec").hexdigest(),
        ...             level=EvidenceLevel.BASIC,
        ...             scope=scope,
        ...         ),
        ...     ),
        ... )
        >>> snapshot = target_capability_snapshot(description)
        >>> snapshot.target_identity.provider
        'acme'
    """

    if not isinstance(description, TargetDescription):
        raise TypeError("description must be a TargetDescription")
    if not isinstance(ttl, timedelta) or ttl <= timedelta(0):
        raise TargetDescriptionError(
            "target capability snapshot ttl must be a positive timedelta"
        )
    captured = captured_at if captured_at is not None else datetime.now(timezone.utc)
    scope = CapabilityScope(device_ids=(description.device_id,))
    evidence_by_id = {item.evidence_id: item for item in description.evidence_refs}
    for source_ref in (
        description.observed_source_ref,
        description.declared_source_ref,
    ):
        if source_ref not in evidence_by_id:
            raise TargetDescriptionError(
                f"target description source ref {source_ref!r} has no evidence "
                "reference"
            )
    observed_evidence = evidence_by_id[description.observed_source_ref]
    if observed_evidence.level not in {
        EvidenceLevel.OBSERVABLE,
        EvidenceLevel.CERTIFICATION,
    }:
        raise TargetDescriptionError(
            "observed target facts require observable or certification evidence"
        )

    observed_values: dict[str, Any] = {
        name: value
        for name, value in description.observed_facts.items()
        if value is not None
    }
    observed_values["device.kind"] = description.device_kind
    # A qubit count is what the provider read off the device it operates, not a
    # claim from a datasheet, and the Core matcher requires observation-grade
    # evidence for it once a circuit states a precision requirement. Declaring it
    # would produce a snapshot that can never legalize a circuit.
    observed_values["qubits.logical_capacity"] = description.n_qubits
    declared_values: dict[str, Any] = {
        name: value
        for name, value in description.declared_facts.items()
        if value is not None
    }
    declared_values["target.class"] = description.target_class
    declared_values["gates.native"] = description.native_gates

    observed_source = FactSource(
        kind=_OBSERVED_SOURCE_KIND,
        ref=description.observed_source_ref,
    )
    declared_source = FactSource(
        kind=_DECLARED_SOURCE_KIND,
        ref=description.declared_source_ref,
    )

    facts: list[CapabilityFact] = []
    for name in sorted(CAPABILITY_NAMES):
        if name in observed_values:
            facts.append(
                CapabilityFact(
                    name=name,
                    value=observed_values[name],
                    support_status=SupportStatus.VERIFIED,
                    fact_exposure=FactExposure.OBSERVED,
                    source=observed_source,
                )
            )
        elif name in declared_values:
            facts.append(
                CapabilityFact(
                    name=name,
                    value=declared_values[name],
                    support_status=SupportStatus.VERIFIED,
                    fact_exposure=FactExposure.DECLARED,
                    source=declared_source,
                )
            )
        else:
            facts.append(
                CapabilityFact(
                    name=name,
                    value=None,
                    support_status=SupportStatus.UNKNOWN,
                    fact_exposure=FactExposure.NOT_EXPOSED,
                    source=observed_source,
                    blockers=(_missing_blocker(name),),
                )
            )

    return TargetCapabilitySnapshot(
        target_identity=TargetIdentity(
            target_id=description.target_id,
            target_class=description.target_class,
            provider=description.provider,
            provider_version=description.provider_version,
            target_revision=description.target_revision,
            environment_id=description.environment_id,
        ),
        scope=scope,
        captured_at=_isoformat(captured),
        valid_until=_isoformat(captured + ttl),
        facts=tuple(facts),
        evidence_refs=description.evidence_refs,
    )


__all__ = (
    "HOST_OWNED_DESCRIPTION_KEYS",
    "TARGET_DESCRIPTION_MEMBER",
    "TargetDescription",
    "TargetDescriptionError",
    "check_target_description",
    "target_capability_snapshot",
)
