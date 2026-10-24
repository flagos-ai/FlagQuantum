"""Side-effect-free Quafu legality preview for routed dynamic programs."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ...core.target_capabilities import TargetCapabilitySnapshot
from ...runtime.dynamic.circuit import DynamicCircuit
from ...runtime.dynamic.dialects.openqasm3 import export_dynamic_qasm3
from ...runtime.dynamic.remote_capabilities import (
    dynamic_remote_requirements,
    match_dynamic_remote_capabilities,
)
from .quafu import _validate_quafu_shots
from .quafu_dynamic import _QUAFU_DYNAMIC_EXTENSION

_TASK_API_REQUEST_FIELDS = (
    "circuits[].qasm",
    "client_ref",
    "mode",
    "shots",
    "target",
)


@dataclass(frozen=True)
class _QuafuDynamicLegalityPreview:
    target: str
    snapshot_id: str
    target_revision: str
    shots: int
    program_format: str
    program_sha256: str | None
    program_bytes: int
    request_method: str
    request_path: str
    request_fields: tuple[str, ...]
    physical_mapping_binding: str
    provider_legality: str
    blockers: tuple[str, ...]

    @property
    def compatible(self) -> bool:
        return self.provider_legality == "accepted"

    def summary(self) -> dict[str, Any]:
        """Return the redacted request shape without program or credentials."""

        return {
            "target": self.target,
            "snapshot_id": self.snapshot_id,
            "target_revision": self.target_revision,
            "shots": self.shots,
            "program_format": self.program_format,
            "program_sha256": self.program_sha256,
            "program_bytes": self.program_bytes,
            "request_method": self.request_method,
            "request_path": self.request_path,
            "request_fields": self.request_fields,
            "physical_mapping_binding": self.physical_mapping_binding,
            "provider_legality": self.provider_legality,
            "blockers": self.blockers,
        }


def _snapshot_fact(snapshot: TargetCapabilitySnapshot, name: str) -> Any:
    return next((fact.value for fact in snapshot.facts if fact.name == name), None)


def _quafu_extension(snapshot: TargetCapabilitySnapshot) -> Mapping[str, Any] | None:
    extension = snapshot.to_dict()["extensions"].get(_QUAFU_DYNAMIC_EXTENSION)
    return extension if isinstance(extension, Mapping) else None


def _provider_contract_blockers(
    snapshot: TargetCapabilitySnapshot,
) -> tuple[tuple[str, str] | None, set[tuple[int, int]], tuple[str, ...]]:
    extension = _quafu_extension(snapshot)
    if extension is None:
        return None, set(), ("quafu_dynamic_capability_evidence_unavailable",)
    program = extension.get("program")
    if not isinstance(program, Mapping):
        return None, set(), ("quafu_dynamic_dialect_unconfirmed",)
    dialect = program.get("dialect")
    version = program.get("version")
    if not isinstance(dialect, str) or not isinstance(version, str):
        return None, set(), ("quafu_dynamic_dialect_unconfirmed",)
    raw_edges = extension.get("directed_connectivity")
    if not isinstance(raw_edges, list):
        return (dialect, version), set(), ("quafu_dynamic_connectivity_unavailable",)
    try:
        edges = {
            (int(edge[0]), int(edge[1]))
            for edge in raw_edges
            if isinstance(edge, list) and len(edge) == 2
        }
    except (TypeError, ValueError):
        edges = set()
    if len(edges) != len(raw_edges):
        return (dialect, version), set(), ("quafu_dynamic_connectivity_unavailable",)
    return (dialect, version), edges, ()


def _routing_blockers(
    circuit: DynamicCircuit,
    *,
    native_operations: set[str],
    physical_capacity: int,
    directed_edges: set[tuple[int, int]],
) -> tuple[str, ...]:
    routing = circuit.to_ir().metadata.get("routing")
    blockers: list[str] = []
    if (
        not isinstance(routing, Mapping)
        or routing.get("schema") != "flagquantum_routing_plan_v1"
        or routing.get("mapping_restored") is not True
    ):
        blockers.append(
            "quafu_dynamic_request_shape_unsupported:missing_routing_evidence"
        )
    for instruction in circuit.to_ir().instructions:
        if any(wire < 0 or wire >= physical_capacity for wire in instruction.wires):
            blockers.append(
                "quafu_dynamic_provider_legality_rejected:physical_capacity"
            )
        if instruction.name not in {"measure", "reset"} and (
            instruction.name not in native_operations
        ):
            blockers.append(
                "quafu_dynamic_provider_legality_rejected:"
                f"operation:{instruction.name}"
            )
        if len(instruction.wires) == 2 and instruction.wires not in directed_edges:
            source, target = instruction.wires
            blockers.append(
                "quafu_dynamic_provider_legality_rejected:"
                f"directed_edge:{source}->{target}"
            )
    return tuple(dict.fromkeys(blockers))


def _preview_quafu_dynamic_request(
    routed_circuit: DynamicCircuit,
    *,
    snapshot: TargetCapabilitySnapshot,
    preflight_snapshot_id: str,
    target: str,
    shots: int,
    evaluated_at: datetime,
) -> _QuafuDynamicLegalityPreview:
    """Validate one routed artifact and describe the current task-API shape."""

    if not isinstance(routed_circuit, DynamicCircuit):
        raise TypeError("routed_circuit must be a DynamicCircuit")
    if not isinstance(snapshot, TargetCapabilitySnapshot):
        raise TypeError("snapshot must be a TargetCapabilitySnapshot")

    blockers: list[str] = []
    identity = snapshot.target_identity
    if (
        identity.provider != "quafu"
        or identity.target_id != target
        or snapshot.snapshot_id != preflight_snapshot_id
    ):
        blockers.append("quafu_dynamic_snapshot_identity_mismatch")

    extension = _quafu_extension(snapshot)
    if (
        extension is None
        or extension.get("attestation_scope") != "conformance_fixture_only"
        or identity.environment_id != "offline-contract-fixture"
    ):
        blockers.append("quafu_dynamic_capability_evidence_unavailable")

    requirements = dynamic_remote_requirements(routed_circuit)
    match = match_dynamic_remote_capabilities(
        requirements,
        snapshot,
        evaluated_at=evaluated_at,
        expected_target_identity=identity,
    )
    blockers.extend(
        f"quafu_dynamic_provider_legality_rejected:{blocker.code}"
        for blocker in match.blockers
    )

    program_contract, directed_edges, contract_blockers = _provider_contract_blockers(
        snapshot
    )
    blockers.extend(contract_blockers)
    native = _snapshot_fact(snapshot, "gates.native")
    capacity = _snapshot_fact(snapshot, "qubits.physical_capacity")
    if not isinstance(native, tuple) or any(
        not isinstance(item, str) for item in native
    ):
        blockers.append("quafu_dynamic_capability_evidence_unavailable")
        native_operations: set[str] = set()
    else:
        native_operations = set(native)
    if type(capacity) is not int or capacity <= 0:
        blockers.append("quafu_dynamic_capability_evidence_unavailable")
        physical_capacity = 0
    else:
        physical_capacity = capacity
    blockers.extend(
        _routing_blockers(
            routed_circuit,
            native_operations=native_operations,
            physical_capacity=physical_capacity,
            directed_edges=directed_edges,
        )
    )

    program_format = "unconfirmed"
    program: str | None = None
    if program_contract == ("openqasm", "3.0"):
        program_format = "openqasm-3.0"
        try:
            program = export_dynamic_qasm3(routed_circuit)
        except (TypeError, ValueError):
            blockers.append("quafu_dynamic_request_shape_unsupported:qasm_export")
    elif program_contract is not None:
        blockers.append("quafu_dynamic_dialect_unconfirmed")

    try:
        _validate_quafu_shots(identity.target_id, shots)
    except ValueError:
        blockers.append("quafu_dynamic_request_shape_unsupported:shots")

    # The implemented task-API payload carries no compiler-disable or physical
    # mapping field. It therefore cannot prove that the routed artifact will be
    # executed with the exact mapping that passed post-routing rematch.
    blockers.append("quafu_dynamic_request_shape_unsupported:physical_mapping_unbound")
    blockers = list(dict.fromkeys(blockers))
    encoded = None if program is None else program.encode("utf-8")
    return _QuafuDynamicLegalityPreview(
        target=target,
        snapshot_id=snapshot.snapshot_id,
        target_revision=identity.target_revision,
        shots=shots,
        program_format=program_format,
        program_sha256=(
            None if encoded is None else hashlib.sha256(encoded).hexdigest()
        ),
        program_bytes=0 if encoded is None else len(encoded),
        request_method="POST",
        request_path="/jobs",
        request_fields=_TASK_API_REQUEST_FIELDS,
        physical_mapping_binding="not_exposed",
        provider_legality="rejected" if blockers else "accepted",
        blockers=tuple(blockers),
    )


__all__: tuple[str, ...] = ()
