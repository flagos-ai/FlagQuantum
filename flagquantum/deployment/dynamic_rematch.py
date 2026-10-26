"""Offline bridge from dynamic routing evidence to the Runtime rematch.

This module validates an already-routed artifact and derives its additive
capability requirements.  It does not build a deployment package, import a
provider, read credentials, use the network, or create a task.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

from ..compiler import CouplingMap
from ..compiler.directed_topology import DirectedCouplingMap
from ..core.target_capabilities import (
    CapabilityBlocker,
    CapabilityRequirement,
    ComparisonOperator,
    EvidenceLevel,
    FactExposure,
    RequirementSet,
    RequirementSource,
    RequirementStrength,
    TargetCapabilitySnapshot,
    TargetIdentity,
    match_target_capabilities,
)
from ..runtime.dynamic.circuit import DynamicCircuit
from ..runtime.dynamic.post_routing import (
    _DynamicPostRoutingRematchDecision,
    evaluate_dynamic_post_routing_rematch,
)
from ..runtime.dynamic.remote_capabilities import dynamic_remote_requirements
from .routing_evidence import (
    build_deployment_routing_evidence,
    stable_payload_sha256,
)

_CONNECTIVITY_PREFIXES = ("coupling_edge:", "directed_edge:")
_STATE_BY_BLOCKER = {
    "missing_fact": "missing",
    "fact_unknown": "unknown",
    "exposure_unknown": "unknown",
    "fact_unmeasured": "unmeasured",
    "unresolved_evidence_reference": "unmeasured",
    "insufficient_evidence": "unmeasured",
    "scope_mismatch": "unmeasured",
    "fact_unsupported": "unsupported",
    "value_mismatch": "unsupported",
    "not_applicable": "unsupported",
    "fact_not_exposed": "not_exposed",
    "exposure_not_accepted": "not_exposed",
    "exposure_requires_observation": "not_exposed",
    "declaration_not_authoritative": "not_exposed",
}
_STATE_PRIORITY = {
    "missing": 0,
    "unknown": 1,
    "unmeasured": 2,
    "unsupported": 3,
    "not_exposed": 4,
    "supported": 5,
}


@dataclass(frozen=True)
class _DynamicRoutingRematchDecision:
    rematch: _DynamicPostRoutingRematchDecision
    routed_artifact_sha256: str
    routing_evidence_sha256: str
    route_requirements: tuple[str, ...]
    route_facts: tuple[tuple[str, str], ...]

    @property
    def compatible(self) -> bool:
        return self.rematch.compatible


def _route_requirements(circuit: DynamicCircuit) -> tuple[str, ...]:
    ir = circuit.to_ir()
    routing = ir.metadata.get("routing")
    if not isinstance(routing, Mapping):
        raise ValueError("routed dynamic circuit does not contain routing evidence")
    direction = routing.get("direction_semantics")
    requirements: set[str] = set()
    for instruction in ir.instructions:
        if instruction.name not in {"measure", "reset"}:
            requirements.add(f"operation:{instruction.name}")
        requirements.update(f"physical_qubit:{wire}" for wire in instruction.wires)
        if len(instruction.wires) != 2:
            continue
        left, right = instruction.wires
        if direction == "directed_cx":
            requirements.add(f"directed_edge:{left}->{right}")
        else:
            low, high = sorted((left, right))
            requirements.add(f"coupling_edge:{low}--{high}")
    return tuple(sorted(requirements))


def _core_requirement(route_requirement: str) -> CapabilityRequirement | None:
    prefix, separator, raw_value = route_requirement.partition(":")
    if not separator:
        raise ValueError(f"invalid route requirement {route_requirement!r}")
    if prefix == "operation":
        return CapabilityRequirement(
            name="gates.native",
            operator=ComparisonOperator.CONTAINS_ALL,
            value=(raw_value,),
            strength=RequirementStrength.MANDATORY,
            source=RequirementSource.COMPILER,
            minimum_evidence_level=EvidenceLevel.BASIC,
            accepted_exposures=(FactExposure.DECLARED, FactExposure.OBSERVED),
        )
    if prefix == "physical_qubit":
        try:
            capacity = int(raw_value) + 1
        except ValueError as error:
            raise ValueError(
                f"invalid physical-qubit route requirement {route_requirement!r}"
            ) from error
        return CapabilityRequirement(
            name="qubits.physical_capacity",
            operator=ComparisonOperator.AT_LEAST,
            value=capacity,
            strength=RequirementStrength.MANDATORY,
            source=RequirementSource.COMPILER,
            minimum_evidence_level=EvidenceLevel.BASIC,
            accepted_exposures=(FactExposure.DECLARED, FactExposure.OBSERVED),
        )
    if route_requirement.startswith(_CONNECTIVITY_PREFIXES):
        return None
    raise ValueError(f"unknown route requirement {route_requirement!r}")


def _fact_state(blockers: tuple[CapabilityBlocker, ...]) -> str:
    states = [
        _STATE_BY_BLOCKER[item.code]
        for item in blockers
        if item.code in _STATE_BY_BLOCKER
    ]
    return min(states, key=_STATE_PRIORITY.__getitem__) if states else "supported"


def _route_fact_states(
    route_requirements: tuple[str, ...],
    snapshot: TargetCapabilitySnapshot,
    *,
    evaluated_at: datetime,
    connectivity_facts: Mapping[str, str] | None,
) -> dict[str, str]:
    supplied = dict(connectivity_facts or {})
    unknown = sorted(set(supplied) - set(route_requirements))
    if unknown:
        raise ValueError(
            "route facts do not correspond to the routed artifact: "
            + ", ".join(unknown)
        )
    facts: dict[str, str] = {}
    for name in route_requirements:
        requirement = _core_requirement(name)
        if requirement is None:
            facts[name] = supplied.get(name, "missing")
            continue
        if name in supplied:
            raise ValueError(f"Core-backed route fact cannot be overridden: {name}")
        match = match_target_capabilities(
            RequirementSet(requirements=(requirement,)),
            snapshot,
            evaluated_at=evaluated_at,
        )
        relevant = tuple(
            blocker
            for blocker in match.blockers
            if blocker.capability_name == requirement.name
        )
        facts[name] = _fact_state(relevant)
    return facts


def evaluate_dynamic_routing_rematch(
    source_requirements: RequirementSet,
    routed_circuit: DynamicCircuit,
    *,
    presented_source_requirement_set_id: str | None,
    requested_target_identity: TargetIdentity,
    preflight_snapshot_id: str,
    postrouting_snapshot: TargetCapabilitySnapshot,
    evaluated_at: datetime,
    coupling_map: CouplingMap | DirectedCouplingMap | None,
    connectivity_facts: Mapping[str, str] | None = None,
    provider_legality: str = "unavailable",
    require_mid_circuit_measurements: bool = False,
    routing_reused: bool = False,
) -> _DynamicRoutingRematchDecision:
    """Validate one routed artifact and run the pure post-routing rematch."""

    if not isinstance(routed_circuit, DynamicCircuit):
        raise TypeError("routed_circuit must be a DynamicCircuit")
    routed_ir = routed_circuit.to_ir()
    routing_evidence = build_deployment_routing_evidence(
        dict(routed_ir.metadata.get("routing", {}) or {}),
        routing_reused=routing_reused,
        n_qubits=routed_circuit.n_qubits,
        coupling_map=coupling_map,
    )
    route_requirements = _route_requirements(routed_circuit)
    route_facts = _route_fact_states(
        route_requirements,
        postrouting_snapshot,
        evaluated_at=evaluated_at,
        connectivity_facts=connectivity_facts,
    )
    routed_requirements = dynamic_remote_requirements(
        routed_circuit,
        require_mid_circuit_measurements=require_mid_circuit_measurements,
    )
    rematch = evaluate_dynamic_post_routing_rematch(
        source_requirements,
        routed_requirements,
        presented_source_requirement_set_id=presented_source_requirement_set_id,
        requested_target_identity=requested_target_identity,
        preflight_snapshot_id=preflight_snapshot_id,
        postrouting_snapshot=postrouting_snapshot,
        evaluated_at=evaluated_at,
        route_requirements=route_requirements,
        route_facts=route_facts,
        provider_legality=provider_legality,
    )
    return _DynamicRoutingRematchDecision(
        rematch=rematch,
        routed_artifact_sha256=routed_ir.content_hash,
        routing_evidence_sha256=stable_payload_sha256(routing_evidence),
        route_requirements=route_requirements,
        route_facts=tuple(sorted(route_facts.items())),
    )


__all__: tuple[str, ...] = ()
