"""Topology- and depth-qualified support for Twin evidence."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from ..core.ir import CircuitIR, ensure_circuit_ir
from .evidence import TwinEvidenceEnvelope, TwinEvidenceReport

if TYPE_CHECKING:
    from .model import QPUDigitalTwin

_SCHEMA = "flagquantum.twin_circuit_support.v1"


def _identity(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _canonical(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _circuit_depth(circuit: CircuitIR) -> int:
    wire_depths = [0] * circuit.n_wires
    for instruction in circuit.instructions:
        layer = max(wire_depths[wire] for wire in instruction.wires) + 1
        for wire in instruction.wires:
            wire_depths[wire] = layer
    return max(wire_depths, default=0)


@dataclass(frozen=True)
class TwinCircuitSupport:
    """Qualify existing Twin evidence by physical topology and circuit depth.

    ``TwinEvidenceEnvelope`` remains the source of statistical evidence. This
    companion object narrows that evidence to directed physical couplers and a
    maximum circuit depth without changing the frozen evidence v1 schema.

    Examples:
        support = fq.twin.TwinCircuitSupport(
            evidence=evidence,
            directed_couplers=((20, 27), (27, 20)),
            maximum_circuit_depth=8,
        )
        report = support.evidence_report(twin, circuit)
    """

    evidence: TwinEvidenceEnvelope
    directed_couplers: tuple[tuple[int, int], ...]
    maximum_circuit_depth: int
    schema: str = _SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _SCHEMA:
            raise ValueError("unsupported Twin circuit-support schema")
        if not isinstance(self.evidence, TwinEvidenceEnvelope):
            raise TypeError("evidence must be a TwinEvidenceEnvelope")
        normalized_couplers: list[tuple[int, int]] = []
        for coupler in self.directed_couplers:
            if len(coupler) != 2:
                raise ValueError(
                    "each directed coupler must contain exactly two qubits"
                )
            normalized_couplers.append((int(coupler[0]), int(coupler[1])))
        couplers = tuple(normalized_couplers)
        if len(couplers) != len(set(couplers)):
            raise ValueError("directed_couplers must be unique")
        physical_qubits = set(self.evidence.physical_qubits)
        if any(
            source == target
            or source not in physical_qubits
            or target not in physical_qubits
            for source, target in couplers
        ):
            raise ValueError(
                "directed_couplers must contain distinct qubits from the "
                "evidence mapping"
            )
        maximum_depth = int(self.maximum_circuit_depth)
        if maximum_depth < 0:
            raise ValueError("maximum_circuit_depth must be non-negative")
        object.__setattr__(self, "directed_couplers", couplers)
        object.__setattr__(self, "maximum_circuit_depth", maximum_depth)

    @property
    def identity(self) -> str:
        """Return the deterministic identity of this support boundary."""

        return _identity(self.to_dict())

    def unsupported_reasons(self, circuit: Any) -> tuple[str, ...]:
        """Return why a circuit is outside this support boundary."""

        ir = ensure_circuit_ir(circuit)
        reasons: list[str] = []
        if not self.evidence.supports_structure(ir):
            reasons.append("outside_evidence_structure")
        if _circuit_depth(ir) > self.maximum_circuit_depth:
            reasons.append("maximum_circuit_depth_exceeded")

        mapping = self.evidence.physical_qubits
        if ir.n_wires != len(mapping):
            return tuple(dict.fromkeys(reasons))
        supported_couplers = set(self.directed_couplers)
        for instruction in ir.instructions:
            if len(instruction.wires) == 1:
                continue
            if len(instruction.wires) != 2:
                reasons.append("multi_qubit_operation_outside_support")
                continue
            physical_coupler = tuple(mapping[wire] for wire in instruction.wires)
            if physical_coupler not in supported_couplers:
                reasons.append("physical_coupler_outside_support")
        return tuple(dict.fromkeys(reasons))

    def supports(self, circuit: Any) -> bool:
        """Return whether a circuit is inside the qualified support boundary."""

        return not self.unsupported_reasons(circuit)

    def evidence_report(self, twin: QPUDigitalTwin, circuit: Any) -> TwinEvidenceReport:
        """Return a Twin report narrowed by topology and depth support.

        The supplied Twin must expose the public ``evidence_report`` method.
        Unsupported topology or depth always removes any error-bound claim.
        """

        if not hasattr(twin, "evidence_report"):
            raise TypeError("twin must be a QPUDigitalTwin")
        ir = ensure_circuit_ir(circuit)
        reasons = self.unsupported_reasons(circuit)
        if reasons:
            snapshot = getattr(twin, "snapshot", None)
            if snapshot is None:
                raise TypeError("twin must be a QPUDigitalTwin")
            if snapshot.identity != self.evidence.snapshot_identity:
                reasons = ("snapshot_identity_mismatch", *reasons)
            if snapshot.physical_qubits != self.evidence.physical_qubits:
                reasons = ("physical_mapping_mismatch", *reasons)
            return TwinEvidenceReport(
                status="out_of_scope",
                prediction=None,
                evidence_envelope_identity=self.evidence.identity,
                evidence_identity=self.evidence.evidence_identity,
                exact_circuit_verified=(
                    ir.content_hash in self.evidence.verified_circuit_identities
                ),
                structurally_supported=False,
                tv_error_bound=None,
                confidence_level=None,
                reasons=tuple(dict.fromkeys(reasons)),
            )

        return twin.evidence_report(circuit, evidence=self.evidence)

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical serialized representation."""

        return {
            "schema": self.schema,
            "evidence": self.evidence.to_dict(),
            "directed_couplers": [list(coupler) for coupler in self.directed_couplers],
            "maximum_circuit_depth": self.maximum_circuit_depth,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TwinCircuitSupport":
        """Restore a circuit-support boundary from its exact v1 schema."""

        if not isinstance(payload, Mapping):
            raise TypeError("Twin circuit support must be a mapping")
        expected = {
            "schema",
            "evidence",
            "directed_couplers",
            "maximum_circuit_depth",
        }
        if set(payload) != expected:
            missing = sorted(expected - set(payload))
            unexpected = sorted(set(payload) - expected)
            raise ValueError(
                "Twin circuit-support fields do not match the v1 schema: "
                f"missing={missing}, unexpected={unexpected}"
            )
        try:
            support = cls(
                schema=str(payload["schema"]),
                evidence=TwinEvidenceEnvelope.from_dict(payload["evidence"]),
                directed_couplers=tuple(
                    tuple(coupler) for coupler in payload["directed_couplers"]
                ),
                maximum_circuit_depth=int(payload["maximum_circuit_depth"]),
            )
            if _canonical(support.to_dict()) != _canonical(payload):
                raise ValueError("Twin circuit support is not in canonical v1 form")
            return support
        except (TypeError, ValueError, IndexError) as error:
            raise ValueError("Invalid Twin circuit support") from error


def load_circuit_support(path: str | PathLike[str]) -> TwinCircuitSupport:
    """Load circuit support without contacting a provider."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot load Twin circuit support from {source}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("Twin circuit-support file must contain a JSON object")
    return TwinCircuitSupport.from_dict(payload)


def dump_circuit_support(
    support: TwinCircuitSupport,
    path: str | PathLike[str],
) -> None:
    """Write canonical circuit support without replacing another artifact."""

    if not isinstance(support, TwinCircuitSupport):
        raise TypeError("support must be a TwinCircuitSupport")
    destination = Path(path)
    encoded = (
        json.dumps(
            support.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )
    try:
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(encoded)
    except FileExistsError:
        try:
            existing = load_circuit_support(destination)
        except ValueError as error:
            raise ValueError(
                f"Refusing to replace invalid Twin circuit support at {destination}"
            ) from error
        if existing.identity != support.identity:
            raise ValueError(
                f"Refusing to replace different Twin circuit support at {destination}"
            )
    except OSError as error:
        raise ValueError(
            f"Cannot write Twin circuit support to {destination}"
        ) from error


__all__ = (
    "dump_circuit_support",
    "load_circuit_support",
    "TwinCircuitSupport",
)
