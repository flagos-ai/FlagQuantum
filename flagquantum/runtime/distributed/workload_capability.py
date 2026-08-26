"""Fail-closed aggregation of FlagOS distributed workload evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .conformance import (
    REQUIRED_FLAGOS_COLLECTIVES,
    REQUIRED_FLAGOS_DTYPES,
    STATEVECTOR_REQUIRED_FLAGOS_COLLECTIVES,
)
from .scale_profile import build_scale_profile
from .training_profile import build_training_profile

WORKLOAD_CAPABILITY_SCHEMA = "flagquantum_flagos_workload_capability_v1"
EXPECTED_WORLD_SIZES = (2, 4, 8)
_CLAIM_FLAGS = (
    "communication_claim_allowed",
    "scalability_claim_allowed",
    "production_support_claim_allowed",
    "release_gate_allowed",
)


class FlagOSWorkloadCapabilityError(ValueError):
    """Raised when evidence cannot be classified without making an inference."""


@dataclass(frozen=True)
class FlagOSWorkloadCapability:
    """One workload-level conclusion backed by bounded development evidence."""

    name: str
    status: str
    required_collectives: tuple[str, ...]
    world_sizes: tuple[int, ...]
    dtypes: tuple[str, ...] = REQUIRED_FLAGOS_DTYPES
    distribution_semantics: str = "sharded_across_ranks"
    evidence_level: str = "development_evidence"
    scope: str = "single_node_2_4_8_card"
    limitations: tuple[str, ...] = ()
    details: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": self.name,
            "status": self.status,
            "evidence_level": self.evidence_level,
            "scope": self.scope,
            "world_sizes": list(self.world_sizes),
            "dtypes": list(self.dtypes),
            "required_collectives": list(self.required_collectives),
            "distribution_semantics": self.distribution_semantics,
            "limitations": list(self.limitations),
        }
        if self.details is not None:
            result["details"] = dict(self.details)
        return result


@dataclass(frozen=True)
class FlagOSWorkloadCapabilityMatrix:
    """Auditable F4 matrix; it deliberately carries no production claim."""

    capabilities: tuple[FlagOSWorkloadCapability, ...]
    evidence_artifacts: tuple[Mapping[str, Any], ...]
    torch_fl_source_revision: str
    schema: str = WORKLOAD_CAPABILITY_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "status": "passed",
            "validation_scope": "flagos_workload_capability_development_evidence",
            "outer_backend": "flagos",
            "inner_communication_route": "unattributed",
            "distribution_semantics": "sharded_across_ranks",
            "claim_evidence_type": "development_artifact_aggregation",
            "capability_matrix_accepted": True,
            "capabilities": {item.name: item.to_dict() for item in self.capabilities},
            "evidence_artifacts": [dict(item) for item in self.evidence_artifacts],
            "environment": {
                "torch_fl_source_revision": self.torch_fl_source_revision,
            },
            "flagcx_route_verified": False,
            "host_staging_observed": None,
            "communication_claim_allowed": False,
            "scalability_claim_allowed": False,
            "production_support_claim_allowed": False,
            "release_gate_allowed": False,
            "blockers": [
                "inner_communication_route_unattributed",
                "host_staging_unverified",
                "single_node_development_evidence_only",
                "single_device_capacity_failure_not_measured",
                "full_collective_suite_unsupported",
                "performance_convergence_and_multinode_not_certified",
            ],
        }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FlagOSWorkloadCapabilityError(message)


def _validate_claim_boundary(payload: Mapping[str, Any], label: str) -> None:
    _require(payload.get("flagcx_route_verified") is False, f"{label}: route inferred")
    _require(
        payload.get("host_staging_observed") is None, f"{label}: host staging claim"
    )
    for flag in _CLAIM_FLAGS:
        if flag in payload:
            _require(payload[flag] is False, f"{label}: {flag} must remain false")
    environment = payload.get("environment")
    _require(isinstance(environment, Mapping), f"{label}: missing environment")
    _require(
        environment.get("worker_failures", []) == [],
        f"{label}: worker failures are present",
    )


def _validate_evidence_descriptors(
    descriptors: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    _require(len(descriptors) == 3, "exactly three evidence descriptors are required")
    schemas = set()
    for item in descriptors:
        _require(isinstance(item.get("path"), str), "evidence path is missing")
        digest = item.get("sha256")
        _require(
            isinstance(digest, str)
            and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest),
            "evidence sha256 is invalid",
        )
        schema = item.get("schema")
        _require(isinstance(schema, str), "evidence schema is missing")
        source_revision = item.get("source_revision")
        _require(
            isinstance(source_revision, str)
            and len(source_revision) == 40
            and all(character in "0123456789abcdef" for character in source_revision),
            "evidence source revision is invalid",
        )
        schemas.add(schema)
    expected = {
        "flagquantum_flagos_distributed_conformance_v1",
        "flagquantum_flagos_statevector_scale_profile_v1",
        "flagquantum_flagos_statevector_training_profile_v1",
    }
    _require(schemas == expected, "evidence descriptors do not cover F1/F2/F3")
    return tuple(
        sorted((dict(item) for item in descriptors), key=lambda item: item["path"])
    )


def _validate_conformance(
    payload: Mapping[str, Any],
) -> tuple[bool, list[dict[str, str]]]:
    _require(
        payload.get("schema") == "flagquantum_flagos_distributed_conformance_v1",
        "F1: unsupported schema",
    )
    _require(
        payload.get("outer_backend") == "flagos", "F1: outer backend is not flagos"
    )
    _require(payload.get("node_count") == 1, "F1: evidence is not single-node")
    _validate_claim_boundary(payload, "F1")
    world_size = payload.get("world_size")
    placement = payload.get("rank_placement")
    identity = payload.get("distributed_identity")
    _require(
        isinstance(world_size, int)
        and world_size >= 2
        and payload.get("local_world_size") == world_size,
        "F1: invalid rank topology",
    )
    _require(
        isinstance(placement, list)
        and len(placement) == world_size
        and {item.get("rank") for item in placement} == set(range(world_size))
        and len({item.get("device_index") for item in placement}) == world_size,
        "F1: invalid rank placement",
    )
    _require(
        isinstance(identity, Mapping)
        and identity.get("outer_backend") == "flagos"
        and identity.get("process_group_initialized") is True
        and identity.get("flagcx_route_verified") is False
        and identity.get("inner_backend") is None,
        "F1: invalid or overclaimed distributed identity",
    )

    checks = payload.get("collective_checks")
    _require(isinstance(checks, list), "F1: collective checks are missing")
    expected = {
        (primitive, dtype)
        for primitive in REQUIRED_FLAGOS_COLLECTIVES
        for dtype in REQUIRED_FLAGOS_DTYPES
    }
    observed = {(item.get("primitive"), item.get("dtype")) for item in checks}
    _require(
        len(checks) == len(expected) and observed == expected,
        "F1: incomplete collective matrix",
    )
    _require(
        all(item.get("device_type") == "flagos" for item in checks),
        "F1: non-FlagOS collective",
    )

    statevector = payload.get("statevector_checks")
    _require(isinstance(statevector, list), "F1: statevector checks are missing")
    _require(
        len(statevector) == len(REQUIRED_FLAGOS_DTYPES)
        and {item.get("dtype") for item in statevector} == set(REQUIRED_FLAGOS_DTYPES)
        and all(
            item.get("passed") is True
            and item.get("device_type") == "flagos"
            and item.get("distribution_semantics") == "sharded_across_ranks"
            and item.get("full_state_materialization") is False
            for item in statevector
        ),
        "F1: statevector workload checks are not accepted",
    )
    required_forward = {
        (primitive, dtype)
        for primitive in STATEVECTOR_REQUIRED_FLAGOS_COLLECTIVES
        for dtype in REQUIRED_FLAGOS_DTYPES
    }
    passing = {
        (item["primitive"], item["dtype"])
        for item in checks
        if item.get("passed") is True
    }
    forward_ready = required_forward <= passing
    _require(
        payload.get("statevector_workload_conformance_accepted") is forward_ready,
        "F1: recorded statevector acceptance disagrees with raw checks",
    )

    missing: list[dict[str, str]] = []
    for item in checks:
        if item.get("passed") is True:
            continue
        error = item.get("error")
        _require(
            isinstance(error, str) and error, "F1: failed collective lacks an error"
        )
        missing.append(
            {
                "primitive": str(item["primitive"]),
                "dtype": str(item["dtype"]),
                "error": error,
            }
        )
    mechanical = not missing
    _require(
        payload.get("mechanical_conformance_accepted") is mechanical,
        "F1: recorded mechanical acceptance disagrees with raw checks",
    )
    _require(
        payload.get("status") == ("passed" if mechanical else "failed"),
        "F1: status disagrees with raw checks",
    )
    return forward_ready, missing


def _validate_profile(payload: Mapping[str, Any], *, training: bool) -> bool:
    label = "F3" if training else "F2"
    expected_schema = (
        "flagquantum_flagos_statevector_training_profile_v1"
        if training
        else "flagquantum_flagos_statevector_scale_profile_v1"
    )
    _require(payload.get("schema") == expected_schema, f"{label}: unsupported schema")
    _validate_claim_boundary(payload, label)
    _require(
        payload.get("world_sizes") == list(EXPECTED_WORLD_SIZES),
        f"{label}: incomplete ladder",
    )
    _require(
        payload.get("distribution_semantics") == "sharded_across_ranks",
        f"{label}: not sharded",
    )
    environment = payload["environment"]
    profile = (
        build_training_profile(payload["runs"], environment=environment)
        if training
        else build_scale_profile(payload["runs"], environment=environment)
    )
    accepted = profile.accepted
    recorded = (
        payload.get("training_ladder_accepted")
        if training
        else payload.get("scale_ladder_accepted")
    )
    _require(
        recorded is accepted, f"{label}: recorded acceptance disagrees with raw runs"
    )
    _require(
        payload.get("status") == ("passed" if accepted else "failed"),
        f"{label}: status disagrees with raw runs",
    )
    if training:
        _require(
            payload.get("sharded_backward_profile_accepted") is accepted
            and payload.get("sharded_optimizer_profile_accepted") is accepted,
            "F3: backward or optimizer acceptance disagrees with raw runs",
        )
    else:
        _require(
            payload.get("statevector_forward_scale_profile_accepted") is accepted,
            "F2: forward acceptance disagrees with raw runs",
        )
    return accepted


def build_flagos_workload_capability_matrix(
    conformance: Mapping[str, Any],
    scale: Mapping[str, Any],
    training: Mapping[str, Any],
    *,
    evidence_artifacts: Sequence[Mapping[str, Any]],
) -> FlagOSWorkloadCapabilityMatrix:
    """Aggregate F1/F2/F3 evidence without inferring an inner provider route."""

    descriptors = _validate_evidence_descriptors(evidence_artifacts)
    forward_conformance, missing = _validate_conformance(conformance)
    forward_scale = _validate_profile(scale, training=False)
    training_ready = _validate_profile(training, training=True)

    revisions = {
        payload["environment"].get("torch_fl_source_revision")
        for payload in (conformance, scale, training)
    }
    _require(
        len(revisions) == 1 and None not in revisions,
        "F1/F2/F3 Torch-FL revisions differ",
    )
    revision = revisions.pop()
    assert isinstance(revision, str)
    _require(
        len(revision) == 40
        and all(character in "0123456789abcdef" for character in revision),
        "Torch-FL source revision is invalid",
    )

    full_collectives = not missing
    if missing:
        _require(
            {(item["primitive"], item["dtype"]) for item in missing}
            == {("reduce_scatter_tensor", dtype) for dtype in REQUIRED_FLAGOS_DTYPES}
            and all("support" in item["error"].lower() for item in missing),
            "F1: collective failures are not an explicit reduce-scatter unsupported result",
        )

    capabilities = (
        FlagOSWorkloadCapability(
            name="flagos.statevector.forward",
            status=(
                "development_verified"
                if forward_conformance and forward_scale
                else "unverified"
            ),
            required_collectives=STATEVECTOR_REQUIRED_FLAGOS_COLLECTIVES,
            world_sizes=EXPECTED_WORLD_SIZES,
            limitations=("single_node_only", "inner_communication_route_unattributed"),
        ),
        FlagOSWorkloadCapability(
            name="flagos.statevector.training",
            status="development_verified" if training_ready else "unverified",
            required_collectives=(
                "all_gather_into_tensor",
                "all_reduce",
                "broadcast",
                "isend_irecv",
            ),
            world_sizes=EXPECTED_WORLD_SIZES,
            limitations=(
                "single_node_bounded_trajectories_only",
                "performance_and_convergence_not_certified",
            ),
            details={
                "gradient_distribution": "replicated_after_all_reduce",
                "optimizer_update_semantics": "owner_step_then_broadcast",
            },
        ),
        FlagOSWorkloadCapability(
            name="flagos.collectives.full",
            status="development_verified" if full_collectives else "unsupported",
            required_collectives=REQUIRED_FLAGOS_COLLECTIVES,
            world_sizes=(int(conformance["world_size"]),),
            scope="single_node_2_card",
            distribution_semantics="process_group_collectives",
            limitations=(
                ("complex_reduce_scatter_tensor_unsupported",) if missing else ()
            ),
            details={"unsupported_checks": missing},
        ),
    )
    return FlagOSWorkloadCapabilityMatrix(
        capabilities=capabilities,
        evidence_artifacts=descriptors,
        torch_fl_source_revision=revision,
    )


__all__ = (
    "FlagOSWorkloadCapability",
    "FlagOSWorkloadCapabilityError",
    "FlagOSWorkloadCapabilityMatrix",
    "WORKLOAD_CAPABILITY_SCHEMA",
    "build_flagos_workload_capability_matrix",
)
