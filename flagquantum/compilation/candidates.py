"""Typed runtime-selection candidate models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .candidate_plans import (
    CircuitAnalysisView,
    communication_plan,
    deployment_plan,
    gradient_plan,
    memory_plan,
)
from .topology import rank_ownership

MetadataProjection = tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]
MetadataProjector = Callable[..., MetadataProjection]


@dataclass(frozen=True)
class RuntimeCandidate:
    """Auditable runtime option for one FlagQuantum IR workload."""

    mode: str
    state_mode: str
    execution_backend: str
    distribution_semantics: str
    scalability_claim_allowed: bool
    estimated_memory_bytes: int
    gradient_support: str
    deployment_ready: bool
    score: int
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()
    intended_distribution_semantics: str | None = None
    memory_plan: Mapping[str, Any] | None = None
    communication_plan: Mapping[str, Any] | None = None
    gradient_plan: Mapping[str, Any] | None = None
    deployment_plan: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] | None = None

    @property
    def available(self) -> bool:
        """Return whether this candidate can be selected without known blockers."""

        return not self.blockers

    def summary(self) -> dict[str, Any]:
        """Return a machine-readable candidate summary."""

        summary = {
            "mode": self.mode,
            "state_mode": self.state_mode,
            "execution_backend": self.execution_backend,
            "claim_evidence_type": "plan_preflight",
            "distribution_semantics": self.distribution_semantics,
            "intended_distribution_semantics": self.intended_distribution_semantics
            or self.distribution_semantics,
            "scalability_claim_allowed": self.scalability_claim_allowed,
            "release_gate_allowed": False,
            "sharding_plan_available": self.distribution_semantics
            == "sharded_across_ranks",
            "estimated_memory_bytes": self.estimated_memory_bytes,
            "gradient_support": self.gradient_support,
            "deployment_ready": self.deployment_ready,
            "score": self.score,
            "available": self.available,
            "reasons": self.reasons,
            "warnings": self.warnings,
            "blockers": self.blockers,
            "memory_plan": dict(self.memory_plan or {}),
            "communication_plan": dict(self.communication_plan or {}),
            "gradient_plan": dict(self.gradient_plan or {}),
            "deployment_plan": dict(self.deployment_plan or {}),
        }
        summary.update(dict(self.metadata or {}))
        return summary


@dataclass
class CandidateBuildContext:
    """Immutable policy inputs plus the candidate output collection."""

    analysis: CircuitAnalysisView
    candidates: list[RuntimeCandidate]
    require_gradients: bool
    require_deployment: bool
    memory_limit_bytes: int | None
    density_bytes: int
    mps_bytes: int
    tensor_network_peak_bytes: int
    dense_bytes: int
    world_size: int
    local_world_size: int
    node_count: int
    complex_bytes: int
    prefer_jax: bool
    prefer_distributed: bool
    requested_state_mode: str
    statevector_metadata_projector: MetadataProjector
    mps_metadata_projector: MetadataProjector


def _unique(items: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(item for item in items if item))


def _memory_blockers(
    estimated_bytes: int, memory_limit_bytes: int | None
) -> tuple[str, ...]:
    if memory_limit_bytes is None or int(estimated_bytes) <= int(memory_limit_bytes):
        return ()
    return ("estimated_memory_exceeds_limit",)


def _score_candidate(
    *,
    base: int,
    prefer_jax: bool,
    prefer_distributed: bool,
    candidate_backend: str,
    distribution_semantics: str,
    state_mode: str,
    requested_state_mode: str,
    blockers: tuple[str, ...],
    warnings: tuple[str, ...],
) -> int:
    score = int(base)
    if prefer_jax and candidate_backend == "jax":
        score += 18
    if prefer_distributed and distribution_semantics == "sharded_across_ranks":
        score += 18
    if requested_state_mode != "auto" and requested_state_mode == state_mode:
        score += 12
    score -= 40 * len(blockers)
    score -= 4 * len(warnings)
    return score


class RuntimeCandidateBuilder:
    """Build one candidate from typed planner context and policy inputs."""

    def __init__(self, context: CandidateBuildContext) -> None:
        self.context = context

    def add(
        self,
        *,
        mode: str,
        state: str,
        backend: str,
        semantics: str,
        memory: int,
        gradient: str,
        deployment: bool,
        base_score: int,
        reasons: tuple[str, ...] = (),
        warnings: tuple[str, ...] = (),
        blockers: tuple[str, ...] = (),
        intended_semantics: str | None = None,
        claim_allowed: bool | None = None,
        gradient_details: Mapping[str, Any] | None = None,
        memory_plan_override: Mapping[str, Any] | None = None,
        communication_plan_override: Mapping[str, Any] | None = None,
        gradient_plan_override: Mapping[str, Any] | None = None,
        deployment_plan_override: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        jax_statevector_training_summary: Mapping[str, Any] | None = None,
        jax_mps_training_summary: Mapping[str, Any] | None = None,
    ) -> None:
        context = self.context
        if context.require_gradients and gradient in {
            "forward_only",
            "distributed_backward_pending",
            "forward_sharded_backward_pending",
            "tensor_node_gradients_only",
        }:
            blockers += (f"{mode}_gradient_incomplete_for_training",)
        if context.require_deployment and not deployment:
            blockers += (f"{mode}_not_directly_deployment_ready",)
        blockers = _unique(
            blockers + _memory_blockers(memory, context.memory_limit_bytes)
        )
        warnings = _unique(warnings)
        claim = (
            semantics == "sharded_across_ranks"
            if claim_allowed is None
            else bool(claim_allowed)
        )
        if claim and semantics != "sharded_across_ranks":
            blockers = _unique(
                blockers + ("scalability_claim_requires_sharded_semantics",)
            )

        ownership = rank_ownership(
            mode=mode,
            state=state,
            n_wires=context.analysis.n_wires,
            world_size=context.world_size,
            local_world_size=context.local_world_size,
        )
        single_device_bytes = (
            context.density_bytes
            if state == "density_matrix"
            else (
                context.mps_bytes
                if state == "mps"
                else (
                    context.tensor_network_peak_bytes
                    if state == "tensor_network"
                    else context.dense_bytes
                )
            )
        )
        candidate_memory_plan = memory_plan(
            mode=mode,
            state=state,
            memory=memory,
            single_device_bytes=single_device_bytes,
            memory_limit_bytes=context.memory_limit_bytes,
            world_size=context.world_size,
            distribution_semantics=semantics,
            peak_intermediate_bytes=(
                context.tensor_network_peak_bytes
                if state == "tensor_network"
                else memory
            ),
        )
        candidate_communication_plan = communication_plan(
            mode=mode,
            state=state,
            analysis=context.analysis,
            world_size=context.world_size,
            local_world_size=context.local_world_size,
            node_count=context.node_count,
            distribution_semantics=semantics,
            memory=memory,
            rank_ownership=ownership,
            complex_bytes=context.complex_bytes,
        )
        candidate_gradient_plan = gradient_plan(
            mode=mode,
            gradient=gradient,
            distribution_semantics=semantics,
            require_gradients=context.require_gradients,
            blockers=blockers,
            details=gradient_details,
        )
        candidate_deployment_plan = deployment_plan(
            mode=mode,
            state=state,
            deployment_ready=deployment,
            require_deployment=context.require_deployment,
            blockers=blockers,
        )
        if memory_plan_override is not None:
            candidate_memory_plan = dict(memory_plan_override)
        if communication_plan_override is not None:
            candidate_communication_plan = dict(communication_plan_override)
        if gradient_plan_override is not None:
            candidate_gradient_plan = dict(gradient_plan_override)
        if deployment_plan_override is not None:
            candidate_deployment_plan = dict(deployment_plan_override)

        candidate_metadata = dict(metadata or {})
        if jax_statevector_training_summary is not None:
            (
                candidate_memory_plan,
                candidate_communication_plan,
                candidate_gradient_plan,
                candidate_deployment_plan,
                projected_metadata,
            ) = context.statevector_metadata_projector(
                jax_statevector_training_summary,
                fallback_memory_plan=candidate_memory_plan,
                fallback_communication_plan=candidate_communication_plan,
                fallback_gradient_plan=candidate_gradient_plan,
                fallback_deployment_plan=candidate_deployment_plan,
                require_deployment=context.require_deployment,
            )
            candidate_metadata.update(projected_metadata)
            if context.require_gradients and mode == "jax_sharded_statevector":
                readiness_blockers = tuple(
                    str(item)
                    for item in candidate_metadata.get("runtime_readiness_blockers", ())
                    or ()
                )
                if readiness_blockers:
                    blockers = _unique(blockers + readiness_blockers)
                    candidate_gradient_plan["blockers"] = blockers
                    candidate_gradient_plan["fail_closed"] = True
        if jax_mps_training_summary is not None:
            (
                candidate_memory_plan,
                candidate_communication_plan,
                candidate_gradient_plan,
                candidate_deployment_plan,
                projected_metadata,
            ) = context.mps_metadata_projector(
                jax_mps_training_summary,
                fallback_memory_plan=candidate_memory_plan,
                fallback_communication_plan=candidate_communication_plan,
                fallback_gradient_plan=candidate_gradient_plan,
                fallback_deployment_plan=candidate_deployment_plan,
                require_deployment=context.require_deployment,
            )
            candidate_metadata.update(projected_metadata)
            if context.require_gradients and mode == "jax_sharded_mps":
                readiness_blockers = tuple(
                    str(item)
                    for item in candidate_metadata.get("runtime_readiness_blockers", ())
                    or ()
                )
                if readiness_blockers:
                    blockers = _unique((*blockers, *readiness_blockers))
                    candidate_gradient_plan["blockers"] = blockers
                    candidate_gradient_plan["fail_closed"] = True

        score = _score_candidate(
            base=base_score,
            prefer_jax=context.prefer_jax,
            prefer_distributed=context.prefer_distributed,
            candidate_backend=backend,
            distribution_semantics=semantics,
            state_mode=state,
            requested_state_mode=context.requested_state_mode,
            blockers=blockers,
            warnings=warnings,
        )
        context.candidates.append(
            RuntimeCandidate(
                mode=mode,
                state_mode=state,
                execution_backend=backend,
                distribution_semantics=semantics,
                intended_distribution_semantics=intended_semantics,
                scalability_claim_allowed=False,
                estimated_memory_bytes=int(memory),
                gradient_support=gradient,
                deployment_ready=deployment,
                score=score,
                reasons=_unique(reasons),
                warnings=warnings,
                blockers=blockers,
                memory_plan=candidate_memory_plan,
                communication_plan=candidate_communication_plan,
                gradient_plan=candidate_gradient_plan,
                deployment_plan=candidate_deployment_plan,
                metadata=candidate_metadata,
            )
        )


__all__ = [
    "CandidateBuildContext",
    "RuntimeCandidate",
    "RuntimeCandidateBuilder",
]
