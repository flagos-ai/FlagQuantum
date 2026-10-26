"""Offline identity sealing for admitted dynamic remote packages."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from ..errors import CapabilityError
from ..runtime.dynamic.circuit import DynamicCircuit
from ..runtime.dynamic.deployment import create_dynamic_deployment_package
from .cloud import DeploymentPackage, validate_deployment_package
from .dynamic_rematch import _DynamicRoutingRematchDecision
from .routing_evidence import stable_payload_sha256

_SEAL_SCHEMA = "flagquantum_dynamic_remote_package_seal_v1"


def _required_text(payload: Mapping[str, Any], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"provider legality summary is missing {name}")
    return value


def _seal_payload(
    package: DeploymentPackage,
    *,
    rematch: _DynamicRoutingRematchDecision,
    provider_legality_identity: str,
) -> dict[str, Any]:
    metadata = package.metadata
    return {
        "schema": _SEAL_SCHEMA,
        "source_requirement_set_id": rematch.rematch.source_requirement_set_id,
        "target_snapshot_id": rematch.rematch.snapshot_id,
        "calibration_id": rematch.rematch.target_identity.target_revision,
        "routed_artifact_sha256": rematch.routed_artifact_sha256,
        "routing_evidence_sha256": rematch.routing_evidence_sha256,
        "provider_legality_identity": provider_legality_identity,
        "package_artifact_sha256": metadata["deployment_artifact_sha256"],
    }


def _seal_dynamic_remote_package(
    routed_circuit: DynamicCircuit,
    *,
    backend: Any,
    rematch: _DynamicRoutingRematchDecision,
    provider_legality: Mapping[str, Any],
    name: str = "flagquantum_dynamic_job",
    shots: int = 1024,
) -> DeploymentPackage:
    """Seal one package only after rematch and provider legality both accept it."""

    if not isinstance(routed_circuit, DynamicCircuit):
        raise TypeError("routed_circuit must be a DynamicCircuit")
    if not isinstance(rematch, _DynamicRoutingRematchDecision):
        raise TypeError("rematch must be a dynamic routing rematch decision")
    if not rematch.compatible:
        raise CapabilityError("dynamic_package_sealing_blocked:post_routing_rematch")
    if provider_legality.get(
        "provider_legality"
    ) != "accepted" or provider_legality.get("blockers") not in ((), []):
        raise CapabilityError("dynamic_package_sealing_blocked:provider_legality")

    target = _required_text(provider_legality, "target")
    snapshot_id = _required_text(provider_legality, "snapshot_id")
    program_sha256 = _required_text(provider_legality, "program_sha256")
    expected_target = rematch.rematch.target_identity.target_id
    if target != expected_target or snapshot_id != rematch.rematch.snapshot_id:
        raise CapabilityError("dynamic_package_sealing_blocked:identity_mismatch")
    if provider_legality.get("shots") != int(shots):
        raise CapabilityError("dynamic_package_sealing_blocked:shots_mismatch")
    if provider_legality.get("physical_mapping_binding") == "not_exposed":
        raise CapabilityError(
            "dynamic_package_sealing_blocked:physical_mapping_unbound"
        )

    provider_legality_identity = stable_payload_sha256(dict(provider_legality))
    package = create_dynamic_deployment_package(
        routed_circuit,
        backend=backend,
        name=name,
        shots=shots,
    )
    validate_deployment_package(package)
    encoded_program = package.qasm.encode("utf-8")
    if hashlib.sha256(encoded_program).hexdigest() != program_sha256:
        raise CapabilityError(
            "dynamic_package_sealing_blocked:program_identity_mismatch"
        )
    if package.ir.content_hash != rematch.routed_artifact_sha256:
        raise CapabilityError("dynamic_package_sealing_blocked:routed_artifact_changed")
    if (
        package.metadata.get("routing_evidence_sha256")
        != rematch.routing_evidence_sha256
    ):
        raise CapabilityError(
            "dynamic_package_sealing_blocked:routing_evidence_changed"
        )

    seal = _seal_payload(
        package,
        rematch=rematch,
        provider_legality_identity=provider_legality_identity,
    )
    metadata = {
        **package.metadata,
        "dynamic_remote_package_seal": seal,
        "dynamic_remote_package_seal_sha256": stable_payload_sha256(seal),
    }
    return replace(package, metadata=metadata)


def _validate_dynamic_remote_package_seal(
    package: DeploymentPackage,
) -> DeploymentPackage:
    """Reject a package whose private dynamic identity chain was altered."""

    validate_deployment_package(package)
    seal = package.metadata.get("dynamic_remote_package_seal")
    if not isinstance(seal, Mapping) or seal.get("schema") != _SEAL_SCHEMA:
        raise ValueError("dynamic remote package seal is missing or unsupported")
    if package.metadata.get("dynamic_remote_package_seal_sha256") != (
        stable_payload_sha256(dict(seal))
    ):
        raise ValueError("dynamic remote package seal digest mismatch")
    if seal.get("package_artifact_sha256") != package.metadata.get(
        "deployment_artifact_sha256"
    ):
        raise ValueError("dynamic remote package seal does not identify this package")
    return package


__all__: tuple[str, ...] = ()
