"""Connected structural coverage composed from qualified Twin cells."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from ..core.ir import ensure_circuit_ir
from .circuit_support import TwinCircuitSupport, _circuit_depth
from .model import QPUDigitalTwin

_REGION_SCHEMA = "flagquantum.twin_connected_region.v1"
_COVERAGE_SCHEMA = "flagquantum.twin_region_coverage.v1"

TwinRegionCoverageStatus = Literal["covered", "out_of_scope"]


def _identity(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _unique(values: Sequence[Any]) -> tuple[Any, ...]:
    return tuple(dict.fromkeys(values))


@dataclass(frozen=True)
class TwinRegionCoverage:
    """Structural coverage facts without a composed accuracy claim."""

    status: TwinRegionCoverageStatus
    physical_qubits: tuple[int, ...]
    required_directed_couplers: tuple[tuple[int, int], ...]
    covered_qubits: tuple[int, ...]
    covered_directed_couplers: tuple[tuple[int, int], ...]
    missing_qubits: tuple[int, ...]
    missing_directed_couplers: tuple[tuple[int, int], ...]
    unsupported_operations: tuple[str, ...]
    circuit_depth: int
    maximum_circuit_depth: int
    source_support_identities: tuple[str, ...]
    reasons: tuple[str, ...]
    tv_error_bound: None = None
    confidence_level: None = None
    schema: str = _COVERAGE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _COVERAGE_SCHEMA:
            raise ValueError("unsupported Twin region-coverage schema")
        if self.status not in {"covered", "out_of_scope"}:
            raise ValueError("unsupported Twin region-coverage status")
        if self.tv_error_bound is not None or self.confidence_level is not None:
            raise ValueError("Twin region coverage cannot carry statistical bounds")
        if self.status == "covered" and (
            self.missing_qubits
            or self.missing_directed_couplers
            or self.unsupported_operations
            or self.reasons
        ):
            raise ValueError("covered Twin region reports cannot contain blockers")

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic representation of the coverage facts."""

        return {
            "schema": self.schema,
            "status": self.status,
            "physical_qubits": list(self.physical_qubits),
            "required_directed_couplers": [
                list(coupler) for coupler in self.required_directed_couplers
            ],
            "covered_qubits": list(self.covered_qubits),
            "covered_directed_couplers": [
                list(coupler) for coupler in self.covered_directed_couplers
            ],
            "missing_qubits": list(self.missing_qubits),
            "missing_directed_couplers": [
                list(coupler) for coupler in self.missing_directed_couplers
            ],
            "unsupported_operations": list(self.unsupported_operations),
            "circuit_depth": self.circuit_depth,
            "maximum_circuit_depth": self.maximum_circuit_depth,
            "source_support_identities": list(self.source_support_identities),
            "reasons": list(self.reasons),
            "tv_error_bound": None,
            "confidence_level": None,
        }


@dataclass(frozen=True)
class TwinConnectedRegion:
    """A connected union of contemporaneous, topology-qualified Twin cells."""

    provider: str
    backend_name: str
    captured_at: str
    physical_qubits: tuple[int, ...]
    directed_couplers: tuple[tuple[int, int], ...]
    supported_operations: tuple[str, ...]
    maximum_instruction_count: int
    maximum_circuit_depth: int
    source_snapshot_identities: tuple[str, ...]
    source_support_identities: tuple[str, ...]
    schema: str = _REGION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _REGION_SCHEMA:
            raise ValueError("unsupported Twin connected-region schema")
        if not self.provider.strip() or not self.backend_name.strip():
            raise ValueError("Twin connected region requires provider and backend")
        if not self.physical_qubits:
            raise ValueError("Twin connected region requires physical qubits")
        if len(self.physical_qubits) != len(set(self.physical_qubits)) or any(
            type(qubit) is not int or qubit < 0 for qubit in self.physical_qubits
        ):
            raise ValueError(
                "Twin connected-region physical qubits must be unique "
                "non-negative integers"
            )
        physical = set(self.physical_qubits)
        if len(self.directed_couplers) != len(set(self.directed_couplers)):
            raise ValueError("Twin connected-region couplers must be unique")
        if any(
            source == target or source not in physical or target not in physical
            for source, target in self.directed_couplers
        ):
            raise ValueError("Twin connected-region couplers must use region qubits")
        if not self.supported_operations:
            raise ValueError("Twin connected region requires supported operations")
        if len(self.supported_operations) != len(set(self.supported_operations)):
            raise ValueError("Twin connected-region operations must be unique")
        if self.maximum_instruction_count < 0 or self.maximum_circuit_depth < 0:
            raise ValueError("Twin connected-region limits must be non-negative")
        if not self.source_snapshot_identities or not self.source_support_identities:
            raise ValueError("Twin connected region requires source identities")
        if len(self.source_snapshot_identities) != len(self.source_support_identities):
            raise ValueError("Twin connected-region source identities must align")
        if len(self.source_support_identities) != len(
            set(self.source_support_identities)
        ):
            raise ValueError("Twin connected-region support identities must be unique")
        if not _is_connected(self.physical_qubits, self.directed_couplers):
            raise ValueError("Twin connected-region qubits and couplers must connect")

    @property
    def target(self) -> str:
        """Return the canonical provider/backend target."""

        return f"{self.provider}:{self.backend_name}"

    @property
    def identity(self) -> str:
        """Return the deterministic identity of this structural region."""

        return _identity(self.to_dict())

    def coverage_report(
        self,
        circuit: Any,
        *,
        physical_qubits: Sequence[int],
    ) -> TwinRegionCoverage:
        """Check an explicit logical-to-physical mapping against the region."""

        ir = ensure_circuit_ir(circuit)
        mapping = tuple(int(qubit) for qubit in physical_qubits)
        if len(mapping) != ir.n_wires:
            raise ValueError("physical_qubits must map every logical circuit qubit")
        if len(mapping) != len(set(mapping)) or any(qubit < 0 for qubit in mapping):
            raise ValueError("physical_qubits must be unique and non-negative")

        region_qubits = set(self.physical_qubits)
        region_couplers = set(self.directed_couplers)
        required_couplers: list[tuple[int, int]] = []
        unsupported_operations: list[str] = []
        reasons: list[str] = []
        for instruction in ir.instructions:
            if instruction.name not in self.supported_operations:
                unsupported_operations.append(instruction.name)
            if len(instruction.wires) == 2:
                required_couplers.append(
                    (mapping[instruction.wires[0]], mapping[instruction.wires[1]])
                )
            elif len(instruction.wires) > 2:
                reasons.append("multi_qubit_operation_outside_support")

        required = _unique(required_couplers)
        missing_qubits = tuple(qubit for qubit in mapping if qubit not in region_qubits)
        covered_qubits = tuple(qubit for qubit in mapping if qubit in region_qubits)
        missing_couplers = tuple(
            coupler for coupler in required if coupler not in region_couplers
        )
        covered_couplers = tuple(
            coupler for coupler in required if coupler in region_couplers
        )
        depth = _circuit_depth(ir)
        if missing_qubits:
            reasons.append("physical_qubits_outside_region")
        if missing_couplers:
            reasons.append("physical_couplers_outside_region")
        if unsupported_operations:
            reasons.append("operations_outside_region_support")
        if len(ir.instructions) > self.maximum_instruction_count:
            reasons.append("maximum_instruction_count_exceeded")
        if depth > self.maximum_circuit_depth:
            reasons.append("maximum_circuit_depth_exceeded")

        blockers = _unique(reasons)
        return TwinRegionCoverage(
            status="covered" if not blockers else "out_of_scope",
            physical_qubits=mapping,
            required_directed_couplers=required,
            covered_qubits=covered_qubits,
            covered_directed_couplers=covered_couplers,
            missing_qubits=missing_qubits,
            missing_directed_couplers=missing_couplers,
            unsupported_operations=_unique(unsupported_operations),
            circuit_depth=depth,
            maximum_circuit_depth=self.maximum_circuit_depth,
            source_support_identities=self.source_support_identities,
            reasons=blockers,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical structural region representation."""

        return {
            "schema": self.schema,
            "provider": self.provider,
            "backend_name": self.backend_name,
            "captured_at": self.captured_at,
            "physical_qubits": list(self.physical_qubits),
            "directed_couplers": [list(edge) for edge in self.directed_couplers],
            "supported_operations": list(self.supported_operations),
            "maximum_instruction_count": self.maximum_instruction_count,
            "maximum_circuit_depth": self.maximum_circuit_depth,
            "source_snapshot_identities": list(self.source_snapshot_identities),
            "source_support_identities": list(self.source_support_identities),
        }


def _is_connected(
    physical_qubits: tuple[int, ...],
    directed_couplers: tuple[tuple[int, int], ...],
) -> bool:
    if len(physical_qubits) == 1:
        return True
    neighbors: dict[int, set[int]] = {qubit: set() for qubit in physical_qubits}
    for source, target in directed_couplers:
        neighbors[source].add(target)
        neighbors[target].add(source)
    visited = {physical_qubits[0]}
    pending = [physical_qubits[0]]
    while pending:
        current = pending.pop()
        for neighbor in neighbors[current] - visited:
            visited.add(neighbor)
            pending.append(neighbor)
    return visited == set(physical_qubits)


def compose_connected_region(
    cells: Sequence[tuple[QPUDigitalTwin, TwinCircuitSupport]],
) -> TwinConnectedRegion:
    """Compose contemporaneous support cells into structural coverage.

    This function never composes statistical error bounds. A returned region
    can qualify topology, operations, instruction count, and depth only.

    Examples:
        region = fq.twin.compose_connected_region(
            [(twin_a, support_a), (twin_b, support_b)]
        )
    """

    normalized = tuple(cells)
    if not normalized:
        raise ValueError("connected-region composition requires at least one cell")
    for twin, support in normalized:
        if not isinstance(twin, QPUDigitalTwin):
            raise TypeError("each region cell must contain a QPUDigitalTwin")
        if not isinstance(support, TwinCircuitSupport):
            raise TypeError("each region cell must contain a TwinCircuitSupport")
        if twin.snapshot.identity != support.evidence.snapshot_identity:
            raise ValueError("Twin cell snapshot does not match its evidence")
        if twin.snapshot.physical_qubits != support.evidence.physical_qubits:
            raise ValueError("Twin cell mapping does not match its evidence")

    first_snapshot = normalized[0][0].snapshot
    if any(
        twin.snapshot.provider != first_snapshot.provider
        or twin.snapshot.backend_name != first_snapshot.backend_name
        for twin, _ in normalized[1:]
    ):
        raise ValueError("Twin region cells must target the same provider and backend")
    if any(
        twin.snapshot.captured_at != first_snapshot.captured_at
        for twin, _ in normalized[1:]
    ):
        raise ValueError("Twin region cells must share one calibration capture time")

    support_identities = tuple(support.identity for _, support in normalized)
    if len(support_identities) != len(set(support_identities)):
        raise ValueError("Twin region cells must be unique")
    physical_qubits = tuple(
        sorted(
            {
                qubit
                for _, support in normalized
                for qubit in support.evidence.physical_qubits
            }
        )
    )
    directed_couplers = tuple(
        sorted(
            {
                coupler
                for _, support in normalized
                for coupler in support.directed_couplers
            }
        )
    )
    if not _is_connected(physical_qubits, directed_couplers):
        raise ValueError("Twin region cells do not form one connected region")
    supported_operations = tuple(
        sorted(
            set.intersection(
                *(
                    set(support.evidence.supported_operations)
                    for _, support in normalized
                )
            )
        )
    )
    if not supported_operations:
        raise ValueError("Twin region cells share no supported operations")

    return TwinConnectedRegion(
        provider=first_snapshot.provider,
        backend_name=first_snapshot.backend_name,
        captured_at=first_snapshot.captured_at,
        physical_qubits=physical_qubits,
        directed_couplers=directed_couplers,
        supported_operations=supported_operations,
        maximum_instruction_count=min(
            support.evidence.maximum_instruction_count for _, support in normalized
        ),
        maximum_circuit_depth=min(
            support.maximum_circuit_depth for _, support in normalized
        ),
        source_snapshot_identities=tuple(
            twin.snapshot.identity for twin, _ in normalized
        ),
        source_support_identities=support_identities,
    )


__all__ = (
    "compose_connected_region",
    "TwinConnectedRegion",
    "TwinRegionCoverage",
    "TwinRegionCoverageStatus",
)
