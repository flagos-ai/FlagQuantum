"""Immutable backend lowering capability registry."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from ..core.ir import CircuitIR, ensure_circuit_ir
from ..core.operator_schema import OPERATOR_SCHEMAS, OperatorSchema, canonical_opcode

BACKENDS = (
    "pytorch",
    "jax",
    "mps",
    "tensor_network",
    "qasm",
    "qcis",
    "provider",
)


class UnsupportedLoweringError(ValueError):
    """Raised before execution when a backend lacks an opcode lowering."""


@dataclass(frozen=True)
class LoweringCapability:
    backend: str
    opcode: str
    strategy: str
    implementation: str
    supported: bool = True
    reason: str = ""


@dataclass(frozen=True)
class OperatorLoweringRegistry:
    schemas: Mapping[str, OperatorSchema] = field(
        default_factory=lambda: OPERATOR_SCHEMAS
    )
    capabilities: Mapping[tuple[str, str], LoweringCapability] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "schemas", MappingProxyType(dict(self.schemas)))
        object.__setattr__(
            self, "capabilities", MappingProxyType(dict(self.capabilities))
        )

    def with_operator(self, schema: OperatorSchema) -> "OperatorLoweringRegistry":
        opcode = str(schema.opcode).lower()
        if opcode in self.schemas:
            raise ValueError(f"operator schema {opcode!r} is already registered")
        updated = dict(self.schemas)
        updated[opcode] = schema
        return replace(self, schemas=updated)

    def with_capability(
        self,
        *,
        backend: str,
        opcode: str,
        strategy: str,
        implementation: str,
        supported: bool = True,
        reason: str = "",
    ) -> "OperatorLoweringRegistry":
        normalized_backend = str(backend).lower()
        if normalized_backend not in BACKENDS:
            raise ValueError(f"unknown lowering backend {backend!r}")
        normalized_opcode = canonical_opcode(opcode)
        if normalized_opcode not in self.schemas:
            raise ValueError(f"unknown operator schema {opcode!r}")
        updated = dict(self.capabilities)
        updated[(normalized_backend, normalized_opcode)] = LoweringCapability(
            backend=normalized_backend,
            opcode=normalized_opcode,
            strategy=str(strategy),
            implementation=str(implementation),
            supported=bool(supported),
            reason=str(reason),
        )
        return replace(self, capabilities=updated)

    def capability(self, backend: str, opcode: str) -> LoweringCapability | None:
        return self.capabilities.get((str(backend).lower(), canonical_opcode(opcode)))

    def validate(
        self, backend: str, opcodes: Sequence[str]
    ) -> tuple[LoweringCapability, ...]:
        normalized_backend = str(backend).lower()
        missing = []
        resolved = []
        for opcode in opcodes:
            canonical = canonical_opcode(opcode)
            capability = self.capability(normalized_backend, canonical)
            if capability is None or not capability.supported:
                detail = (
                    capability.reason if capability is not None else "not registered"
                )
                missing.append(f"{canonical} ({detail})")
            else:
                resolved.append(capability)
        if missing:
            raise UnsupportedLoweringError(
                f"backend {normalized_backend!r} cannot lower: {', '.join(missing)}"
            )
        return tuple(resolved)

    def validate_ir(self, backend: str, program: Any) -> tuple[LoweringCapability, ...]:
        ir: CircuitIR = ensure_circuit_ir(program)
        return self.validate(backend, tuple(item.name for item in ir.instructions))

    def manifest(self) -> dict[str, Any]:
        return {
            "backends": {
                backend: {
                    "supported": tuple(
                        sorted(
                            opcode
                            for (
                                item_backend,
                                opcode,
                            ), capability in self.capabilities.items()
                            if item_backend == backend and capability.supported
                        )
                    ),
                    "unsupported": tuple(
                        sorted(
                            opcode
                            for opcode in self.schemas
                            if not (
                                (capability := self.capability(backend, opcode))
                                and capability.supported
                            )
                        )
                    ),
                }
                for backend in BACKENDS
            }
        }


def _builtin_registry() -> OperatorLoweringRegistry:
    registry = OperatorLoweringRegistry(schemas=OPERATOR_SCHEMAS)
    qcis_supported = {
        "i",
        "x",
        "y",
        "z",
        "h",
        "s",
        "sdg",
        "t",
        "tdg",
        "sx",
        "sxdg",
        "rx",
        "ry",
        "rz",
        "phase",
        "u1",
        "u2",
        "u3",
        "cx",
        "cy",
        "cz",
        "swap",
        "rxx",
        "ryy",
        "rzz",
        "ccx",
    }
    for backend in BACKENDS:
        for opcode, schema in registry.schemas.items():
            supported = schema.unitary
            strategy = "native"
            implementation = f"flagquantum.{backend}.{opcode}"
            reason = ""
            if backend == "qcis":
                supported = opcode in qcis_supported
                strategy = "decomposition"
                implementation = "flagquantum.utils.qcis_exporter._decompose"
                reason = "no QCIS decomposition" if not supported else ""
            elif backend in {"qasm", "provider"}:
                strategy = "serialization"
                reason = (
                    "channels require provider-specific lowering"
                    if not supported
                    else ""
                )
            elif schema.channel:
                supported = backend == "pytorch"
                strategy = "kraus"
                implementation = "flagquantum.simulation.noise"
                reason = (
                    "channel lowering not implemented for this backend"
                    if not supported
                    else ""
                )
            registry = registry.with_capability(
                backend=backend,
                opcode=opcode,
                strategy=strategy,
                implementation=implementation,
                supported=supported,
                reason=reason,
            )
    return registry


DEFAULT_LOWERING_REGISTRY = _builtin_registry()


def validate_lowering(
    program: Any,
    backend: str,
    *,
    registry: OperatorLoweringRegistry = DEFAULT_LOWERING_REGISTRY,
) -> tuple[LoweringCapability, ...]:
    return registry.validate_ir(backend, program)


__all__ = [
    "BACKENDS",
    "DEFAULT_LOWERING_REGISTRY",
    "LoweringCapability",
    "OperatorLoweringRegistry",
    "UnsupportedLoweringError",
    "validate_lowering",
]
