"""Provider-free target capability snapshots for the private compiler."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping

from flagquantum.core.operator_schema import OPERATOR_SCHEMAS, canonical_opcode

from .passes.placement_routing import DirectedCouplingGraph

_SHA256 = re.compile(r"[0-9a-f]{64}")


class TargetClass(str, Enum):
    LOCAL_RUNTIME = "local_runtime"
    QASM_TEXT = "qasm_text"
    NON_QASM_ARTIFACT = "non_qasm_artifact"


class ArtifactFormat(str, Enum):
    RUNTIME_PLAN = "runtime_plan"
    OPENQASM_2 = "openqasm_2"
    OPENQASM_3_STATIC = "openqasm_3_static"
    QCIS_1 = "qcis_1"


class MeasurementResult(str, Enum):
    STATE = "state"
    EXPECTATION = "expectation"
    PROBABILITIES = "probabilities"
    SAMPLES = "samples"
    COUNTS = "counts"


class ControlFlowProfile(str, Enum):
    STATIC_ONLY = "static_only"
    ADAPTIVE_REAL_TIME = "adaptive_real_time"


class AncillaPolicy(str, Enum):
    UNSUPPORTED = "unsupported"
    EXPLICIT_ONLY = "explicit_only"
    CLEAN_ALLOCATABLE = "clean_allocatable"
    CLEAN_AND_DIRTY_ALLOCATABLE = "clean_and_dirty_allocatable"


@dataclass(frozen=True, order=True)
class ParameterConstraint:
    name: str
    minimum: float | None = None
    maximum: float | None = None
    periodic: bool = False

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("parameter constraint name cannot be empty")
        if not isinstance(self.periodic, bool):
            raise ValueError("parameter constraint periodic flag must be boolean")
        for value in (self.minimum, self.maximum):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float))
            ):
                raise ValueError("parameter constraint bounds must be numeric")
            if value is not None and not math.isfinite(float(value)):
                raise ValueError("parameter constraint bounds must be finite")
        minimum = None if self.minimum is None else float(self.minimum)
        maximum = None if self.maximum is None else float(self.maximum)
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError("parameter constraint minimum exceeds maximum")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "periodic": self.periodic,
        }


@dataclass(frozen=True, order=True)
class GateCapability:
    operation: str
    parameters: tuple[ParameterConstraint, ...] = ()

    def __post_init__(self) -> None:
        operation = canonical_opcode(self.operation)
        schema = OPERATOR_SCHEMAS.get(operation)
        if schema is None:
            raise ValueError(f"unknown native operation {self.operation!r}")
        parameters = tuple(sorted(self.parameters))
        names = tuple(item.name for item in parameters)
        if len(names) != len(set(names)):
            raise ValueError("gate parameter constraints must be unique")
        if not set(names).issubset(schema.parameters):
            raise ValueError(
                f"invalid parameter constraint for operation {operation!r}"
            )
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "parameters", parameters)

    def to_dict(self) -> dict[str, object]:
        return {
            "operation": self.operation,
            "parameters": [item.to_dict() for item in self.parameters],
        }


@dataclass(frozen=True, order=True)
class ArtifactProfile:
    format: ArtifactFormat
    version: str

    def __post_init__(self) -> None:
        if not isinstance(self.format, ArtifactFormat):
            raise ValueError("artifact format must use the closed ArtifactFormat enum")
        version = str(self.version).strip()
        if not version:
            raise ValueError("artifact profile version cannot be empty")
        versions = {
            ArtifactFormat.RUNTIME_PLAN: "1.0",
            ArtifactFormat.OPENQASM_2: "2.0",
            ArtifactFormat.OPENQASM_3_STATIC: "3.0",
            ArtifactFormat.QCIS_1: "1.0",
        }
        if version != versions[self.format]:
            raise ValueError("unsupported artifact profile version")
        object.__setattr__(self, "version", version)

    def to_dict(self) -> dict[str, str]:
        return {"format": self.format.value, "version": self.version}


def _canonical_timestamp(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("calibration validity must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError("calibration validity must include a timezone")
    normalized = parsed.astimezone(timezone.utc).isoformat(timespec="microseconds")
    return normalized.replace("+00:00", "Z")


_FORMATS_BY_CLASS = {
    TargetClass.LOCAL_RUNTIME: frozenset({ArtifactFormat.RUNTIME_PLAN}),
    TargetClass.QASM_TEXT: frozenset(
        {ArtifactFormat.OPENQASM_2, ArtifactFormat.OPENQASM_3_STATIC}
    ),
    TargetClass.NON_QASM_ARTIFACT: frozenset({ArtifactFormat.QCIS_1}),
}


@dataclass(frozen=True)
class TargetCapabilities:
    """Immutable semantic input; labels and execution identity are non-semantic."""

    target_class: TargetClass
    logical_qubit_capacity: int
    physical_qubit_capacity: int
    native_gates: tuple[GateCapability, ...]
    measurement_results: tuple[MeasurementResult, ...]
    artifact_profiles: tuple[ArtifactProfile, ...]
    topology: DirectedCouplingGraph | None = None
    control_flow: ControlFlowProfile = ControlFlowProfile.STATIC_ONLY
    supports_mid_circuit_measurement: bool = False
    supports_reset: bool = False
    supports_timing: bool = False
    supports_pulse: bool = False
    supports_noise: bool = False
    supports_parameter_binding: bool = False
    maximum_shots: int | None = None
    maximum_program_operations: int | None = None
    ancilla_policy: AncillaPolicy = AncillaPolicy.UNSUPPORTED
    maximum_compiler_ancillas: int = 0
    calibration_snapshot_hash: str | None = None
    calibration_valid_until: str | None = None
    display_label: str | None = None

    def __post_init__(self) -> None:
        self._validate_closed_values()
        boolean_fields = (
            "supports_mid_circuit_measurement",
            "supports_reset",
            "supports_timing",
            "supports_pulse",
            "supports_noise",
            "supports_parameter_binding",
        )
        if any(not isinstance(getattr(self, name), bool) for name in boolean_fields):
            raise ValueError("target capability flags must be boolean")
        if (
            isinstance(self.logical_qubit_capacity, bool)
            or isinstance(self.physical_qubit_capacity, bool)
            or not isinstance(self.logical_qubit_capacity, int)
            or not isinstance(self.physical_qubit_capacity, int)
        ):
            raise ValueError("qubit capacities must be integers")
        logical = self.logical_qubit_capacity
        physical = self.physical_qubit_capacity
        if logical <= 0 or physical <= 0 or logical > physical:
            raise ValueError("qubit capacities require 0 < logical <= physical")
        gates = tuple(sorted(self.native_gates))
        results = tuple(
            sorted(set(self.measurement_results), key=lambda item: item.value)
        )
        artifacts = tuple(sorted(self.artifact_profiles))
        if not gates or len(gates) != len({item.operation for item in gates}):
            raise ValueError("native gates must be non-empty and operation-unique")
        if not results or len(results) != len(self.measurement_results):
            raise ValueError("measurement results must be non-empty and unique")
        if not artifacts or len(artifacts) != len(set(artifacts)):
            raise ValueError("artifact profiles must be non-empty and unique")
        allowed_formats = _FORMATS_BY_CLASS[self.target_class]
        if any(item.format not in allowed_formats for item in artifacts):
            raise ValueError("artifact format is incompatible with target class")
        if self.topology is not None and self.topology.n_qubits != physical:
            raise ValueError("topology size must equal physical qubit capacity")
        if self.control_flow is ControlFlowProfile.ADAPTIVE_REAL_TIME:
            if not self.supports_mid_circuit_measurement:
                raise ValueError("adaptive control requires mid-circuit measurement")
        if self.supports_pulse and not self.supports_timing:
            raise ValueError("pulse support requires timing support")
        self._validate_limits(physical)
        calibration_hash = self.calibration_snapshot_hash
        if calibration_hash is not None:
            calibration_hash = str(calibration_hash).strip().lower()
            if _SHA256.fullmatch(calibration_hash) is None:
                raise ValueError("calibration snapshot hash must be lowercase SHA-256")
        validity = _canonical_timestamp(self.calibration_valid_until)
        if validity is not None and calibration_hash is None:
            raise ValueError("calibration validity requires a snapshot hash")
        label = None if self.display_label is None else str(self.display_label).strip()
        if label == "":
            raise ValueError("display label cannot be empty")
        object.__setattr__(self, "logical_qubit_capacity", logical)
        object.__setattr__(self, "physical_qubit_capacity", physical)
        object.__setattr__(self, "native_gates", gates)
        object.__setattr__(self, "measurement_results", results)
        object.__setattr__(self, "artifact_profiles", artifacts)
        object.__setattr__(self, "calibration_snapshot_hash", calibration_hash)
        object.__setattr__(self, "calibration_valid_until", validity)
        object.__setattr__(self, "display_label", label)

    def _validate_closed_values(self) -> None:
        enum_fields = (
            (self.target_class, TargetClass, "target class"),
            (self.control_flow, ControlFlowProfile, "control-flow profile"),
            (self.ancilla_policy, AncillaPolicy, "ancilla policy"),
        )
        for value, expected, label in enum_fields:
            if not isinstance(value, expected):
                raise ValueError(f"{label} must use its closed enum")
        if any(not isinstance(item, GateCapability) for item in self.native_gates):
            raise ValueError("native gates must use GateCapability")
        if any(
            not isinstance(item, MeasurementResult) for item in self.measurement_results
        ):
            raise ValueError("measurement results must use the closed enum")
        if any(
            not isinstance(item, ArtifactProfile) for item in self.artifact_profiles
        ):
            raise ValueError("artifact profiles must use ArtifactProfile")

    def _validate_limits(self, physical: int) -> None:
        for name in ("maximum_shots", "maximum_program_operations"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int)
            ):
                raise ValueError(f"{name} must be an integer")
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when specified")
        if isinstance(self.maximum_compiler_ancillas, bool) or not isinstance(
            self.maximum_compiler_ancillas, int
        ):
            raise ValueError("maximum_compiler_ancillas must be an integer")
        ancillas = self.maximum_compiler_ancillas
        allocatable = self.ancilla_policy in {
            AncillaPolicy.CLEAN_ALLOCATABLE,
            AncillaPolicy.CLEAN_AND_DIRTY_ALLOCATABLE,
        }
        if allocatable and not 0 < ancillas <= physical:
            raise ValueError("allocatable ancilla policy requires a bounded capacity")
        if not allocatable and ancillas != 0:
            raise ValueError("non-allocatable ancilla policy requires zero capacity")
        object.__setattr__(self, "maximum_compiler_ancillas", ancillas)

    def semantic_dict(self) -> dict[str, object]:
        topology = None
        if self.topology is not None:
            topology = {
                "kind": "directed_coupling_graph_v1",
                "n_qubits": self.topology.n_qubits,
                "edges": [list(edge) for edge in self.topology.edges],
            }
        return {
            "schema_version": "target_capabilities_v1",
            "target_class": self.target_class.value,
            "logical_qubit_capacity": self.logical_qubit_capacity,
            "physical_qubit_capacity": self.physical_qubit_capacity,
            "native_gates": [item.to_dict() for item in self.native_gates],
            "measurement_results": [item.value for item in self.measurement_results],
            "artifact_profiles": [item.to_dict() for item in self.artifact_profiles],
            "topology": topology,
            "control_flow": self.control_flow.value,
            "supports_mid_circuit_measurement": self.supports_mid_circuit_measurement,
            "supports_reset": self.supports_reset,
            "supports_timing": self.supports_timing,
            "supports_pulse": self.supports_pulse,
            "supports_noise": self.supports_noise,
            "supports_parameter_binding": self.supports_parameter_binding,
            "maximum_shots": self.maximum_shots,
            "maximum_program_operations": self.maximum_program_operations,
            "ancilla_policy": self.ancilla_policy.value,
            "maximum_compiler_ancillas": self.maximum_compiler_ancillas,
            "calibration_snapshot_hash": self.calibration_snapshot_hash,
            "calibration_valid_until": self.calibration_valid_until,
        }

    @property
    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.semantic_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")

    @property
    def semantic_fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {**self.semantic_dict(), "display_label": self.display_label}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TargetCapabilities:
        if not isinstance(payload, Mapping):
            raise ValueError("target capability payload must be a mapping")
        allowed = {
            "schema_version",
            "target_class",
            "logical_qubit_capacity",
            "physical_qubit_capacity",
            "native_gates",
            "measurement_results",
            "artifact_profiles",
            "topology",
            "control_flow",
            "supports_mid_circuit_measurement",
            "supports_reset",
            "supports_timing",
            "supports_pulse",
            "supports_noise",
            "supports_parameter_binding",
            "maximum_shots",
            "maximum_program_operations",
            "ancilla_policy",
            "maximum_compiler_ancillas",
            "calibration_snapshot_hash",
            "calibration_valid_until",
            "display_label",
        }
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"unknown target capability fields: {sorted(unknown)!r}")
        required = allowed - {"display_label"}
        missing = required - set(payload)
        if missing:
            raise ValueError(f"missing target capability fields: {sorted(missing)!r}")
        if payload.get("schema_version") != "target_capabilities_v1":
            raise ValueError("unsupported target capability schema version")
        topology_data = payload.get("topology")
        topology = None
        if topology_data is not None:
            if not isinstance(topology_data, Mapping):
                raise ValueError("topology must be a mapping")
            if set(topology_data) != {"kind", "n_qubits", "edges"}:
                raise ValueError("invalid topology fields")
            if topology_data["kind"] != "directed_coupling_graph_v1":
                raise ValueError("unsupported topology kind")
            topology = DirectedCouplingGraph(
                topology_data["n_qubits"],
                tuple(tuple(edge) for edge in topology_data["edges"]),
            )
        gate_data = payload["native_gates"]
        if not isinstance(gate_data, (list, tuple)):
            raise ValueError("native gates must be a sequence")
        for item in gate_data:
            if not isinstance(item, Mapping) or set(item) != {
                "operation",
                "parameters",
            }:
                raise ValueError("invalid native gate fields")
            if not isinstance(item["parameters"], (list, tuple)):
                raise ValueError("gate parameters must be a sequence")
            for constraint in item["parameters"]:
                if not isinstance(constraint, Mapping) or set(constraint) != {
                    "name",
                    "minimum",
                    "maximum",
                    "periodic",
                }:
                    raise ValueError("invalid parameter constraint fields")
        artifact_data = payload["artifact_profiles"]
        if not isinstance(artifact_data, (list, tuple)):
            raise ValueError("artifact profiles must be a sequence")
        if any(
            not isinstance(item, Mapping) or set(item) != {"format", "version"}
            for item in artifact_data
        ):
            raise ValueError("invalid artifact profile fields")
        return cls(
            target_class=TargetClass(payload["target_class"]),
            logical_qubit_capacity=payload["logical_qubit_capacity"],
            physical_qubit_capacity=payload["physical_qubit_capacity"],
            native_gates=tuple(
                GateCapability(
                    item["operation"],
                    tuple(
                        ParameterConstraint(**constraint)
                        for constraint in item["parameters"]
                    ),
                )
                for item in gate_data
            ),
            measurement_results=tuple(
                MeasurementResult(item) for item in payload["measurement_results"]
            ),
            artifact_profiles=tuple(
                ArtifactProfile(ArtifactFormat(item["format"]), item["version"])
                for item in artifact_data
            ),
            topology=topology,
            control_flow=ControlFlowProfile(payload["control_flow"]),
            supports_mid_circuit_measurement=payload[
                "supports_mid_circuit_measurement"
            ],
            supports_reset=payload["supports_reset"],
            supports_timing=payload["supports_timing"],
            supports_pulse=payload["supports_pulse"],
            supports_noise=payload["supports_noise"],
            supports_parameter_binding=payload["supports_parameter_binding"],
            maximum_shots=payload["maximum_shots"],
            maximum_program_operations=payload["maximum_program_operations"],
            ancilla_policy=AncillaPolicy(payload["ancilla_policy"]),
            maximum_compiler_ancillas=payload["maximum_compiler_ancillas"],
            calibration_snapshot_hash=payload["calibration_snapshot_hash"],
            calibration_valid_until=payload["calibration_valid_until"],
            display_label=payload.get("display_label"),
        )


__all__ = [
    "AncillaPolicy",
    "ArtifactFormat",
    "ArtifactProfile",
    "ControlFlowProfile",
    "GateCapability",
    "MeasurementResult",
    "ParameterConstraint",
    "TargetCapabilities",
    "TargetClass",
]
