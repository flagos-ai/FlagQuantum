"""Read-only Runtime validation for compilation-evidence handoffs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeAlias

import torch

from ..core._artifacts import (
    ArtifactKind,
    CircuitArtifactBindingResult,
    ProgramArtifactV2,
    ProgramArtifactV3,
)
from ..core._compilation_evidence import CompilationEvidenceBundle
from ..core._compilation_evidence_v2 import CompilationEvidenceBundleV2
from ..core._compilation_evidence_v3 import CompilationEvidenceBundleV3
from ..core.target_capabilities import TargetCapabilitySnapshot
from ..errors import ExecutionError

_CompilationEvidenceBundle: TypeAlias = (
    CompilationEvidenceBundle
    | CompilationEvidenceBundleV2
    | CompilationEvidenceBundleV3
)


def _resolve_source_artifact(
    source: ProgramArtifactV2 | CircuitArtifactBindingResult,
) -> tuple[ProgramArtifactV2, str, str | None]:
    if isinstance(source, CircuitArtifactBindingResult):
        return (
            source.bound_artifact,
            source.source_artifact_identity,
            source.binding_identity,
        )
    if isinstance(source, ProgramArtifactV2):
        return source, source.artifact_identity, None
    raise TypeError(
        "source must be a ProgramArtifactV2 or CircuitArtifactBindingResult"
    )


def _verify_allocated_result_projection(
    bundle: CompilationEvidenceBundleV3,
    artifact: ProgramArtifactV3,
    compilation: Mapping[str, Any],
) -> None:
    allocated_plan = bundle.physical_plan
    result_schema = artifact.result_schema
    if (
        compilation["physical_plan_identity"] != allocated_plan.plan_identity
        or compilation["allocation_identity"] != allocated_plan.allocation_identity
        or tuple(result_schema["logical_wires"])
        != tuple(range(allocated_plan.logical_wire_count))
        or tuple(result_schema["physical_result_slots"])
        != allocated_plan.logical_result_physical_slots
        or result_schema["ordering"] != "logical_wire_order"
        or result_schema["kind"] != "samples"
        or result_schema["shots_source"] != "execution_request"
    ):
        raise ExecutionError(
            "allocated executable result projection does not match its evidence"
        )


def _verify_target_lineage(
    bundle: _CompilationEvidenceBundle,
    target: Mapping[str, Any],
    snapshot: TargetCapabilitySnapshot,
    compilation: Mapping[str, Any],
) -> None:
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


def _verify_output_lineage(
    bundle: _CompilationEvidenceBundle,
    artifact: ProgramArtifactV2 | ProgramArtifactV3,
    compilation: Mapping[str, Any],
) -> None:
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


def _verify_physical_plan_lineage(
    bundle: _CompilationEvidenceBundle,
    circuit_artifact: ProgramArtifactV2,
    artifact: ProgramArtifactV2 | ProgramArtifactV3,
    snapshot: TargetCapabilitySnapshot,
    compilation: Mapping[str, Any],
) -> None:
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


def verify_compilation_evidence_handoff(
    bundle: _CompilationEvidenceBundle,
    source: ProgramArtifactV2 | CircuitArtifactBindingResult,
    artifact: ProgramArtifactV2 | ProgramArtifactV3,
    *,
    snapshot: TargetCapabilitySnapshot,
) -> None:
    """Verify bundle lineage against actual source, target, and output objects."""

    if not isinstance(
        bundle,
        (
            CompilationEvidenceBundle,
            CompilationEvidenceBundleV2,
            CompilationEvidenceBundleV3,
        ),
    ):
        raise TypeError("bundle must be a compilation evidence bundle")
    circuit_artifact, source_identity, binding_identity = _resolve_source_artifact(
        source
    )
    if (
        circuit_artifact.kind is not ArtifactKind.CIRCUIT
        or circuit_artifact.profile["name"] != "circuit-ir-1.0"
        or circuit_artifact.parameter_schema["binding"] != "fully_bound"
    ):
        raise ExecutionError(
            "compilation evidence requires a fully bound circuit source"
        )
    if not isinstance(artifact, (ProgramArtifactV2, ProgramArtifactV3)):
        raise TypeError("artifact must be a ProgramArtifactV2 or ProgramArtifactV3")
    if isinstance(bundle, CompilationEvidenceBundleV3) != isinstance(
        artifact, ProgramArtifactV3
    ):
        raise ExecutionError(
            "compilation evidence and executable artifact versions are incompatible"
        )
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
    _verify_target_lineage(bundle, target, snapshot, compilation)
    _verify_output_lineage(bundle, artifact, compilation)
    _verify_physical_plan_lineage(
        bundle, circuit_artifact, artifact, snapshot, compilation
    )
    if isinstance(bundle, CompilationEvidenceBundleV3):
        assert isinstance(artifact, ProgramArtifactV3)
        _verify_allocated_result_projection(bundle, artifact, compilation)


def validate_executable_result_samples(
    artifact: ProgramArtifactV2 | ProgramArtifactV3,
    samples: torch.Tensor,
) -> torch.Tensor:
    """Validate adapter samples without inferring or repairing result mappings."""

    if not isinstance(artifact, (ProgramArtifactV2, ProgramArtifactV3)):
        raise TypeError("artifact must be a ProgramArtifactV2 or ProgramArtifactV3")
    if artifact.kind is not ArtifactKind.EXECUTABLE:
        raise ExecutionError("result samples require an executable artifact")
    if not isinstance(samples, torch.Tensor):
        raise TypeError("samples must be a torch.Tensor")
    if samples.ndim == 0:
        raise ExecutionError("result samples must have a result-width dimension")
    schema = artifact.result_schema
    if not isinstance(schema, Mapping) or schema.get("kind") != "samples":
        raise ExecutionError("executable artifact has no samples result contract")
    if isinstance(artifact, ProgramArtifactV3):
        logical_wires = tuple(schema["logical_wires"])
        if schema["ordering"] != "logical_wire_order":
            raise ExecutionError("allocated samples are not in logical-wire order")
        expected_width = len(logical_wires)
    else:
        expected_width = len(tuple(schema["wires"]))
    if samples.shape[-1] != expected_width:
        raise ExecutionError(
            "result sample width does not match the executable artifact contract"
        )
    return samples


__all__ = (
    "validate_executable_result_samples",
    "verify_compilation_evidence_handoff",
)
