"""Fail-closed deployment contract for compiler routing evidence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Any

from ..compiler import CouplingMap

DEPLOYMENT_ROUTING_EVIDENCE_SCHEMA = "flagquantum_deployment_routing_evidence_v1"
DEPLOYMENT_PACKAGE_SCHEMA = "flagquantum_deployment_package_v1"


class DeploymentRoutingEvidenceError(ValueError):
    """Raised when a deployment package has inconsistent routing evidence."""


def _undirected_couplings(
    couplings: Iterable[Iterable[int]],
) -> frozenset[tuple[int, int]]:
    """Normalize couplings so that only the connected physical pairs remain.

    This is the one definition of device identity in this module. It also
    normalizes the couplings a *plan* records, so a malformed ``coupling_edges``
    field is rejected by name whether or not there is a device to compare it
    against.
    """

    normalized = set()
    for coupling in couplings:
        try:
            left, right = coupling
        except (TypeError, ValueError) as malformed:
            raise DeploymentRoutingEvidenceError(
                "routing coupling edges must be pairs of qubits"
            ) from malformed
        normalized.add((min(int(left), int(right)), max(int(left), int(right))))
    return frozenset(normalized)


def same_undirected_device(
    left_couplings: Iterable[Iterable[int]],
    right_couplings: Iterable[Iterable[int]],
) -> bool:
    """Return whether two coupling sequences describe the same device.

    A device is a set of physical couplings, so neither the order of the couplings
    nor the order of the endpoints inside one coupling is part of the hardware.
    Comparing the sequences themselves would instead make device identity a
    property of whoever serialized the list last: ``CouplingMap`` keeps the edge
    order its caller declared while ``DirectedCouplingMap`` sorts, so the
    compiler's two device types disagree about the same device by construction.

    Args:
        left_couplings: Coupling edges from a routing plan, a record, or a device.
        right_couplings: Coupling edges from the device being compared against.

    Returns:
        ``True`` when both sequences name the same set of undirected couplings.
    """

    return _undirected_couplings(left_couplings) == _undirected_couplings(
        right_couplings
    )


def _non_negative_int(plan: Mapping[str, Any], key: str) -> int:
    value = plan.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise DeploymentRoutingEvidenceError(
            f"routing plan requires non-negative integer {key!r}"
        )
    return value


def validate_deployment_routing_plan(
    routing_plan: Mapping[str, Any],
    *,
    n_qubits: int,
    coupling_map: CouplingMap | None,
) -> dict[str, Any]:
    """Validate and normalize a routing plan for provider serialization."""

    plan = dict(routing_plan)
    if not plan:
        if coupling_map is not None:
            raise DeploymentRoutingEvidenceError(
                "deployment backend coupling requires a routing plan"
            )
        return {}
    if plan.get("schema") != "flagquantum_routing_plan_v1":
        raise DeploymentRoutingEvidenceError("unsupported routing plan schema")
    strategy = plan.get("strategy")
    if strategy not in {"restore_after_each_gate", "persistent_layout"}:
        raise DeploymentRoutingEvidenceError("unsupported routing strategy")

    expected_identity = tuple(range(int(n_qubits)))
    initial = tuple(plan.get("initial_logical_to_physical", ()))
    final = tuple(plan.get("final_logical_to_physical", ()))
    pre_restore = tuple(plan.get("pre_restore_logical_to_physical", ()))
    if initial != expected_identity:
        raise DeploymentRoutingEvidenceError(
            "routing initial permutation must be identity"
        )
    if final != expected_identity or plan.get("mapping_restored") is not True:
        raise DeploymentRoutingEvidenceError(
            "deployment routing must restore the final logical permutation"
        )
    if sorted(pre_restore) != list(expected_identity):
        raise DeploymentRoutingEvidenceError(
            "routing pre-restore mapping must be a complete permutation"
        )
    if plan.get("direction_semantics") != "logical_wire_order_preserved":
        raise DeploymentRoutingEvidenceError(
            "routing direction semantics are missing or unsupported"
        )

    plan_couplings = _undirected_couplings(plan.get("coupling_edges", ()))
    if coupling_map is not None and plan_couplings != _undirected_couplings(
        coupling_map.edges
    ):
        raise DeploymentRoutingEvidenceError(
            "routing coupling edges do not match the deployment backend"
        )
    if int(plan.get("coupling_n_wires", -1)) < int(n_qubits):
        raise DeploymentRoutingEvidenceError(
            "routing coupling exposes fewer qubits than the circuit"
        )

    planned = _non_negative_int(plan, "planned_inserted_swap_count")
    inserted = _non_negative_int(plan, "inserted_swap_count")
    if inserted != planned:
        raise DeploymentRoutingEvidenceError(
            "inserted SWAP compatibility count must equal planned count"
        )
    retained = plan.get("post_optimization_inserted_swap_count")
    if retained is not None:
        if (
            not isinstance(retained, int)
            or isinstance(retained, bool)
            or retained < 0
            or retained > planned
        ):
            raise DeploymentRoutingEvidenceError(
                "post-optimization SWAP count must be within planned count"
            )
    for key in (
        "topology_gate_count",
        "routed_gate_count",
        "skipped_channel_count",
    ):
        _non_negative_int(plan, key)

    selection = plan.get("strategy_selection")
    if selection:
        if not isinstance(selection, Mapping):
            raise DeploymentRoutingEvidenceError(
                "routing strategy selection must be a mapping"
            )
        if (
            selection.get("schema") != "flagquantum_routing_strategy_selection_v1"
            or selection.get("selected_strategy") != strategy
        ):
            raise DeploymentRoutingEvidenceError(
                "routing strategy selection disagrees with materialized plan"
            )
        candidates = selection.get("candidates")
        if not isinstance(candidates, Mapping) or set(candidates) != {
            "restore_after_each_gate",
            "persistent_layout",
        }:
            raise DeploymentRoutingEvidenceError(
                "routing strategy selection candidates are incomplete"
            )
    return plan


def build_deployment_routing_evidence(
    routing_plan: Mapping[str, Any],
    *,
    routing_reused: bool,
    n_qubits: int,
    coupling_map: CouplingMap | None,
) -> dict[str, Any]:
    """Build the stable provider-facing deployment evidence envelope."""

    validated = validate_deployment_routing_plan(
        routing_plan,
        n_qubits=n_qubits,
        coupling_map=coupling_map,
    )
    return {
        "schema": DEPLOYMENT_ROUTING_EVIDENCE_SCHEMA,
        "status": "validated" if validated else "not_required",
        "routing_reused": bool(routing_reused),
        "routing_plan": validated,
    }


def stable_payload_sha256(payload: Mapping[str, Any]) -> str:
    """Hash a JSON-compatible mapping with deterministic canonical encoding."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def deployment_artifact_sha256(
    *,
    name: str,
    backend_provider: str,
    backend_name: str,
    shots: int,
    program_format: str,
    program: str,
    routing_evidence_sha256: str,
) -> str:
    """Hash the immutable provider-facing identity of a deployment package."""

    return stable_payload_sha256(
        {
            "schema": DEPLOYMENT_PACKAGE_SCHEMA,
            "name": str(name),
            "backend_provider": str(backend_provider),
            "backend_name": str(backend_name),
            "shots": int(shots),
            "program_format": str(program_format),
            "program": str(program),
            "routing_evidence_sha256": str(routing_evidence_sha256),
        }
    )


__all__ = [
    "DEPLOYMENT_PACKAGE_SCHEMA",
    "DEPLOYMENT_ROUTING_EVIDENCE_SCHEMA",
    "DeploymentRoutingEvidenceError",
    "build_deployment_routing_evidence",
    "deployment_artifact_sha256",
    "same_undirected_device",
    "stable_payload_sha256",
    "validate_deployment_routing_plan",
]
