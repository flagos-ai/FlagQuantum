"""Verified circuit-artifact to executable-artifact compilation."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime

from ..core._artifacts import (
    ArtifactKind,
    CircuitArtifactBindingResult,
    ProgramArtifactV2,
)
from ..core.ir import CircuitIR
from ..core.target_capabilities import TargetCapabilitySnapshot
from ..errors import CompilationError
from .directed_topology import DirectedCouplingMap
from .physical_plan import PhysicalCircuitPlan, build_physical_circuit_plan
from .routing import CouplingMap
from .target_artifact import build_target_artifact
from .target_conformance import TargetConformanceResult, verify_target_emission
from .target_emission import TargetEmissionResult, emit_legalized_target
from .target_legalization import TargetLegalizationResult, legalize_circuit_for_target

_SHA256 = re.compile(r"[0-9a-f]{64}")


class ArtifactCompilationError(CompilationError):
    """A circuit artifact cannot enter the verified target compilation path."""


def _compilation_identity(
    *,
    source_artifact_identity: str,
    circuit_artifact_identity: str,
    binding_identity: str | None,
    target_snapshot_id: str,
    physical_plan_identity: str,
    executable_artifact_identity: str,
) -> str:
    payload = {
        "schema": "flagquantum.artifact_compilation",
        "version": "1.0",
        "source_artifact_identity": source_artifact_identity,
        "circuit_artifact_identity": circuit_artifact_identity,
        "binding_identity": binding_identity,
        "target_snapshot_id": target_snapshot_id,
        "physical_plan_identity": physical_plan_identity,
        "executable_artifact_identity": executable_artifact_identity,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ArtifactCompilationResult:
    """Immutable evidence for one complete artifact compilation."""

    input_artifact: ProgramArtifactV2 | CircuitArtifactBindingResult = field(repr=False)
    source_artifact_identity: str
    circuit_artifact_identity: str
    binding_identity: str | None
    legalization: TargetLegalizationResult
    physical_plan: PhysicalCircuitPlan
    emission: TargetEmissionResult
    conformance: TargetConformanceResult
    executable_artifact: ProgramArtifactV2
    compilation_identity: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.input_artifact, CircuitArtifactBindingResult):
            input_circuit_artifact = self.input_artifact.bound_artifact
            expected_source_identity = self.input_artifact.source_artifact_identity
            expected_binding_identity = self.input_artifact.binding_identity
        elif isinstance(self.input_artifact, ProgramArtifactV2):
            input_circuit_artifact = self.input_artifact
            expected_source_identity = self.input_artifact.artifact_identity
            expected_binding_identity = None
        else:
            raise TypeError(
                "input_artifact must be a ProgramArtifactV2 or "
                "CircuitArtifactBindingResult"
            )
        if (
            self.source_artifact_identity != expected_source_identity
            or self.circuit_artifact_identity
            != input_circuit_artifact.artifact_identity
            or self.binding_identity != expected_binding_identity
        ):
            raise ValueError("artifact compilation input lineage is inconsistent")
        for owner, identity in (
            ("source_artifact_identity", self.source_artifact_identity),
            ("circuit_artifact_identity", self.circuit_artifact_identity),
        ):
            if _SHA256.fullmatch(identity) is None:
                raise ValueError(f"{owner} must be a lowercase SHA-256 digest")
        if (
            self.binding_identity is not None
            and _SHA256.fullmatch(self.binding_identity) is None
        ):
            raise ValueError("binding_identity must be a lowercase SHA-256 digest")
        if not isinstance(self.legalization, TargetLegalizationResult):
            raise TypeError("legalization must be a TargetLegalizationResult")
        if not isinstance(self.physical_plan, PhysicalCircuitPlan):
            raise TypeError("physical_plan must be a PhysicalCircuitPlan")
        if (
            self.physical_plan.legalization is not self.legalization
            or self.physical_plan.program is not self.legalization.program
        ):
            raise ValueError(
                "artifact compilation physical plan is not bound to legalization"
            )
        if not isinstance(self.emission, TargetEmissionResult):
            raise TypeError("emission must be a TargetEmissionResult")
        if not isinstance(self.conformance, TargetConformanceResult):
            raise TypeError("conformance must be a TargetConformanceResult")
        if not isinstance(self.executable_artifact, ProgramArtifactV2):
            raise TypeError("executable_artifact must be a ProgramArtifactV2")
        if self.executable_artifact.kind is not ArtifactKind.EXECUTABLE:
            raise ValueError("artifact compilation output must be executable")
        topology = self.legalization.topology_legalization
        expected_native_source_hash = input_circuit_artifact.circuit_content_hash
        if topology is not None:
            if (
                topology.source_content_hash
                != input_circuit_artifact.circuit_content_hash
            ):
                raise ValueError(
                    "artifact compilation topology is not bound to the input artifact"
                )
            expected_native_source_hash = topology.program.content_hash
        if (
            self.legalization.native_gate_legalization.source_content_hash
            != expected_native_source_hash
        ):
            raise ValueError(
                "artifact compilation legalization is not bound to the input artifact"
            )
        compilation = self.executable_artifact.compilation
        target = self.executable_artifact.target
        if compilation is None or target is None:
            raise ValueError("executable artifact lacks compilation evidence")
        if (
            self.emission.source_circuit_hash != self.legalization.program.content_hash
            or self.emission.target_legalization_identity
            != self.legalization.legalization_identity
            or self.conformance.emission_identity != self.emission.emission_identity
            or self.executable_artifact.circuit_content_hash
            != self.legalization.program.content_hash
            or target["snapshot_id"] != self.legalization.target_snapshot_id
            or compilation["target_legalization_identity"]
            != self.legalization.legalization_identity
            or compilation["schedule_identity"]
            != self.legalization.schedule.schedule_identity
            or compilation["emission_identity"] != self.emission.emission_identity
            or compilation["conformance_identity"]
            != self.conformance.conformance_identity
        ):
            raise ValueError("artifact compilation evidence chain is inconsistent")
        expected = _compilation_identity(
            source_artifact_identity=self.source_artifact_identity,
            circuit_artifact_identity=self.circuit_artifact_identity,
            binding_identity=self.binding_identity,
            target_snapshot_id=self.legalization.target_snapshot_id,
            physical_plan_identity=self.physical_plan.plan_identity,
            executable_artifact_identity=self.executable_artifact.artifact_identity,
        )
        if self.compilation_identity and self.compilation_identity != expected:
            raise ValueError("compilation_identity does not match compilation evidence")
        object.__setattr__(self, "compilation_identity", expected)


def compile_circuit_artifact_for_target(
    artifact: ProgramArtifactV2 | CircuitArtifactBindingResult,
    *,
    backend: str,
    profile: str,
    snapshot: TargetCapabilitySnapshot,
    producer: str,
    evaluated_at: datetime | None = None,
    coupling_map: CouplingMap | DirectedCouplingMap | None = None,
    routing_strategy: str = "auto",
    initial_layout: tuple[int, ...] | None = None,
    max_native_added_operations: int = 256,
    max_routing_added_operations: int = 256,
    max_direction_added_operations: int = 256,
    max_schedule_depth: int | None = None,
) -> ArtifactCompilationResult:
    """Compile a fully bound circuit artifact through every verified stage."""

    binding = artifact if isinstance(artifact, CircuitArtifactBindingResult) else None
    if binding is not None:
        circuit_artifact = binding.bound_artifact
        source_identity = binding.source_artifact_identity
        binding_identity = binding.binding_identity
    elif isinstance(artifact, ProgramArtifactV2):
        circuit_artifact = artifact
        source_identity = artifact.artifact_identity
        binding_identity = None
    else:
        raise TypeError(
            "artifact must be a ProgramArtifactV2 or CircuitArtifactBindingResult"
        )

    if (
        circuit_artifact.kind is not ArtifactKind.CIRCUIT
        or circuit_artifact.profile["name"] != "circuit-ir-1.0"
    ):
        raise ArtifactCompilationError(
            "artifact compilation requires a circuit-ir-1.0 circuit artifact"
        )
    if (
        circuit_artifact.parameter_schema["binding"] != "fully_bound"
        or circuit_artifact.parameter_schema["parameters"]
    ):
        raise ArtifactCompilationError(
            "artifact compilation requires a fully bound circuit artifact"
        )
    if not isinstance(circuit_artifact.payload, Mapping):
        raise ArtifactCompilationError("circuit artifact payload must be a mapping")
    circuit = CircuitIR.from_dict(circuit_artifact.payload)
    if circuit.content_hash != circuit_artifact.circuit_content_hash:
        raise ArtifactCompilationError(
            "circuit artifact content does not match circuit_content_hash"
        )

    try:
        legalization = legalize_circuit_for_target(
            circuit,
            backend=backend,
            snapshot=snapshot,
            evaluated_at=evaluated_at,
            coupling_map=coupling_map,
            routing_strategy=routing_strategy,
            initial_layout=initial_layout,
            max_added_operations=max_native_added_operations,
            max_routing_added_operations=max_routing_added_operations,
            max_direction_added_operations=max_direction_added_operations,
            max_schedule_depth=max_schedule_depth,
        )
        physical_plan = build_physical_circuit_plan(
            legalization,
            coupling_map=coupling_map,
        )
        emission = emit_legalized_target(
            physical_plan.legalization,
            profile=profile,
        )
        conformance = verify_target_emission(emission, legalization)
        executable = build_target_artifact(
            legalization,
            emission,
            conformance,
            producer=producer,
        )
    except CompilationError as error:
        raise ArtifactCompilationError(
            f"artifact target compilation failed: {error}"
        ) from error
    return ArtifactCompilationResult(
        input_artifact=artifact,
        source_artifact_identity=source_identity,
        circuit_artifact_identity=circuit_artifact.artifact_identity,
        binding_identity=binding_identity,
        legalization=legalization,
        physical_plan=physical_plan,
        emission=emission,
        conformance=conformance,
        executable_artifact=executable,
    )


__all__ = (
    "ArtifactCompilationError",
    "ArtifactCompilationResult",
    "compile_circuit_artifact_for_target",
)
