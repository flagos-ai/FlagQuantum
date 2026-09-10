"""Read-only Runtime validation for compilation-evidence handoffs."""

from __future__ import annotations

from collections.abc import Mapping

from ..core._artifacts import (
    ArtifactKind,
    CircuitArtifactBindingResult,
    ProgramArtifactV2,
)
from ..core._compilation_evidence import CompilationEvidenceBundle
from ..core.target_capabilities import TargetCapabilitySnapshot
from ..errors import ExecutionError


def verify_compilation_evidence_handoff(
    bundle: CompilationEvidenceBundle,
    source: ProgramArtifactV2 | CircuitArtifactBindingResult,
    artifact: ProgramArtifactV2,
    *,
    snapshot: TargetCapabilitySnapshot,
) -> None:
    """Verify bundle lineage against actual source, target, and output objects."""

    if not isinstance(bundle, CompilationEvidenceBundle):
        raise TypeError("bundle must be a CompilationEvidenceBundle")
    if isinstance(source, CircuitArtifactBindingResult):
        circuit_artifact = source.bound_artifact
        source_identity = source.source_artifact_identity
        binding_identity = source.binding_identity
    elif isinstance(source, ProgramArtifactV2):
        circuit_artifact = source
        source_identity = source.artifact_identity
        binding_identity = None
    else:
        raise TypeError(
            "source must be a ProgramArtifactV2 or CircuitArtifactBindingResult"
        )
    if (
        circuit_artifact.kind is not ArtifactKind.CIRCUIT
        or circuit_artifact.profile["name"] != "circuit-ir-1.0"
        or circuit_artifact.parameter_schema["binding"] != "fully_bound"
    ):
        raise ExecutionError(
            "compilation evidence requires a fully bound circuit source"
        )
    if not isinstance(artifact, ProgramArtifactV2):
        raise TypeError("artifact must be a ProgramArtifactV2")
    if artifact.kind is not ArtifactKind.EXECUTABLE:
        raise ExecutionError("compilation evidence output must be executable")
    if not isinstance(snapshot, TargetCapabilitySnapshot):
        raise TypeError("snapshot must be a TargetCapabilitySnapshot")

    expected_source = {
        "source_artifact_identity": source_identity,
        "circuit_artifact_identity": circuit_artifact.artifact_identity,
        "binding_identity": binding_identity,
        "source_circuit_hash": circuit_artifact.circuit_content_hash,
        "final_circuit_hash": artifact.circuit_content_hash,
    }
    if dict(bundle.source) != expected_source:
        raise ExecutionError(
            "compilation evidence does not match the supplied source lineage"
        )

    target = artifact.target
    compilation = artifact.compilation
    if target is None or not isinstance(compilation, Mapping):
        raise ExecutionError("executable artifact lacks target compilation evidence")
    expected_target = {
        "snapshot_id": snapshot.snapshot_id,
        "target_legalization_identity": compilation["target_legalization_identity"],
    }
    if dict(bundle.target) != expected_target or target["snapshot_id"] != (
        snapshot.snapshot_id
    ):
        raise ExecutionError(
            "compilation evidence does not match the supplied target snapshot"
        )

    expected_output = {
        "profile": str(artifact.profile["name"]),
        "payload_sha256": artifact.payload_sha256,
        "emission_identity": compilation["emission_identity"],
        "conformance_identity": compilation["conformance_identity"],
        "executable_artifact_identity": artifact.artifact_identity,
    }
    for name, value in expected_output.items():
        if bundle.output[name] != value:
            raise ExecutionError(
                f"compilation evidence output {name} does not match the artifact"
            )
    plan = bundle.physical_plan
    if (
        plan.source_circuit_hash != circuit_artifact.circuit_content_hash
        or plan.physical_circuit_hash != artifact.circuit_content_hash
        or plan.target_snapshot_id != snapshot.snapshot_id
        or plan.schedule_identity != compilation["schedule_identity"]
    ):
        raise ExecutionError(
            "compilation evidence physical plan does not match the handoff"
        )


__all__ = ("verify_compilation_evidence_handoff",)
