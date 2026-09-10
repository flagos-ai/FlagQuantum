"""Local execution of verified Core circuit artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Sequence

from ..core._artifacts import ArtifactKind, ProgramArtifactV2
from ..core.ir import CircuitIR, MeasurementNode
from ..errors import ExecutionError
from .execution import run
from .result import ExecutionResult

if TYPE_CHECKING:
    from ..noise import NoiseModel
    from .options import ExecutionOptions


def execute_circuit_artifact(
    artifact: ProgramArtifactV2,
    *,
    options: ExecutionOptions | None = None,
    measurements: Sequence[MeasurementNode] | None = None,
    noise_model: NoiseModel | None = None,
) -> ExecutionResult:
    """Plan and execute a fully bound ``circuit-ir-1.0`` artifact locally.

    Target-text executable artifacts remain target-bound deployment products;
    Runtime does not reinterpret them as simulator input.
    """

    if not isinstance(artifact, ProgramArtifactV2):
        raise TypeError("artifact must be a ProgramArtifactV2")
    profile = str(artifact.profile["name"])
    if artifact.kind is not ArtifactKind.CIRCUIT or profile != "circuit-ir-1.0":
        raise ExecutionError(
            "local artifact execution requires a circuit-ir-1.0 circuit artifact; "
            "target-text executable artifacts require a target adapter"
        )
    binding = artifact.parameter_schema["binding"]
    parameters = artifact.parameter_schema["parameters"]
    if binding != "fully_bound" or parameters:
        raise ExecutionError(
            "local artifact execution requires a fully bound circuit artifact"
        )
    if not isinstance(artifact.payload, Mapping):
        raise ExecutionError("circuit artifact payload must be a CircuitIR mapping")

    circuit = CircuitIR.from_dict(artifact.payload)
    if circuit.content_hash != artifact.circuit_content_hash:
        raise ExecutionError(
            "circuit artifact payload does not match circuit_content_hash"
        )
    result = run(
        circuit,
        options=options,
        measurements=measurements,
        noise_model=noise_model,
    )
    enriched = replace(
        result,
        provenance={
            **dict(result.provenance),
            "program_artifact_identity": artifact.artifact_identity,
            "program_artifact_payload_sha256": artifact.payload_sha256,
            "program_artifact_circuit_content_hash": artifact.circuit_content_hash,
            "program_artifact_profile": profile,
        },
        compatibility={
            **dict(result.compatibility),
            "program_artifact_schema": "flagquantum.program_artifact",
            "program_artifact_version": artifact.version,
            "program_artifact_verified": True,
        },
    )
    native = result.__dict__.get("_native_output")
    if native is not None:
        object.__setattr__(enriched, "_native_output", native)
    return enriched


__all__ = ("execute_circuit_artifact",)
