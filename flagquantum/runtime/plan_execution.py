"""Exact execution path for a validated executable plan."""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from typing import Iterable

import torch

from ..compute.cpu_target_capabilities import (
    probe_local_cpu_target_capabilities,
)
from ..core.runtime_config import RuntimeConfig
from ..core.target_capabilities import (
    CapabilityRequirement,
    CapabilityScope,
    ComparisonOperator,
    EvidenceLevel,
    FactExposure,
    RequirementSet,
    RequirementSource,
    RequirementStrength,
    TargetIdentity,
)
from ..errors import ExecutionError, FlagQuantumError
from .execution_plan import ExecutionPlan
from .execution_plan_contract import (
    ExecutionPlanContractError,
    plan_decision,
    plan_execution_program,
    plan_noise_model,
    plan_program,
    validate_plan_environment,
)
from .measurements import validate_measurements
from .result import ExecutionResult
from .target_capability_matching import (
    CandidateProvenance,
    RouteIntent,
    TargetCapabilityCandidate,
    TargetCapabilityDecision,
    match_target_capability_candidates,
)

_LOCAL_CPU_REQUIREMENTS = RequirementSet(
    requirements=(
        CapabilityRequirement(
            name="device.kind",
            operator=ComparisonOperator.EQUALS,
            value="cpu",
            strength=RequirementStrength.MANDATORY,
            source=RequirementSource.RUNTIME_PROTOCOL,
            minimum_evidence_level=EvidenceLevel.OBSERVABLE,
            accepted_exposures=(FactExposure.OBSERVED,),
        ),
        CapabilityRequirement(
            name="device.count",
            operator=ComparisonOperator.AT_LEAST,
            value=1,
            strength=RequirementStrength.MANDATORY,
            source=RequirementSource.RUNTIME_PROTOCOL,
            minimum_evidence_level=EvidenceLevel.OBSERVABLE,
            accepted_exposures=(FactExposure.OBSERVED,),
        ),
    )
)


def _effective_precision_requirements(
    precision: str,
    requirements: RequirementSet | None = None,
) -> RequirementSet:
    precision_requirement = CapabilityRequirement(
        name="precision.effective_dtype",
        operator=ComparisonOperator.EQUALS,
        value=precision,
        strength=RequirementStrength.MANDATORY,
        source=RequirementSource.COMPILER,
        minimum_evidence_level=EvidenceLevel.OBSERVABLE,
        accepted_exposures=(FactExposure.OBSERVED,),
    )
    if requirements is None:
        return RequirementSet(requirements=(precision_requirement,))
    if not isinstance(requirements, RequirementSet):
        raise TypeError("requirements must be a Core RequirementSet")
    return RequirementSet(
        requirements=(*requirements.requirements, precision_requirement),
        fallback_authorizations=requirements.fallback_authorizations,
        extensions=requirements.extensions,
    )


def execution_plan_precision_requirements(
    plan: ExecutionPlan,
    requirements: RequirementSet | None = None,
) -> RequirementSet:
    """Require candidates to prove the effective precision chosen by ``plan``."""

    return _effective_precision_requirements(plan.precision, requirements)


def match_execution_plan_target_candidates(
    plan: ExecutionPlan,
    candidates: Iterable[TargetCapabilityCandidate],
    *,
    requirements: RequirementSet | None = None,
    evaluated_at: datetime | None = None,
    expected_target_identity: TargetIdentity | None = None,
    required_scope: CapabilityScope | None = None,
    claim_minimum_evidence_level: EvidenceLevel = EvidenceLevel.BASIC,
    route_intent: RouteIntent | None = None,
) -> TargetCapabilityDecision:
    """Match candidates after adding the plan's mandatory effective precision."""

    if route_intent is not None and route_intent.effective_precision != plan.precision:
        raise ValueError(
            "route_intent.effective_precision must match ExecutionPlan.precision"
        )
    return match_target_capability_candidates(
        execution_plan_precision_requirements(plan, requirements),
        candidates,
        evaluated_at=evaluated_at,
        expected_target_identity=expected_target_identity,
        required_scope=required_scope,
        claim_minimum_evidence_level=claim_minimum_evidence_level,
        route_intent=route_intent,
    )


@lru_cache(maxsize=4)
def _require_local_cpu_capabilities(
    backend: str,
    precision: str,
) -> None:
    """Validate each process-stable local CPU route once."""

    snapshot = probe_local_cpu_target_capabilities(precision)
    requirements = _effective_precision_requirements(precision, _LOCAL_CPU_REQUIREMENTS)
    intent = RouteIntent(
        backend=backend,
        device_kind="cpu",
        cpu=True,
        effective_precision=precision,
        algorithm="local_execution",
        approximation="exact",
    )
    provenance = CandidateProvenance(
        intent_id=intent.intent_id,
        snapshot_id=snapshot.snapshot_id,
        backend=intent.backend,
        device_kind=intent.device_kind,
        cpu=intent.cpu,
        effective_precision=intent.effective_precision,
        algorithm=intent.algorithm,
        approximation=intent.approximation,
    )
    decision = match_target_capability_candidates(
        requirements,
        (
            TargetCapabilityCandidate(
                candidate_id=snapshot.snapshot_id,
                snapshot=snapshot,
                provenance=provenance,
            ),
        ),
        route_intent=intent,
    )
    if not decision.executable:
        details = "; ".join(blocker.message for blocker in decision.blockers)
        raise ExecutionError(f"local CPU capability preflight failed: {details}")


def execute_plan(execution_plan: ExecutionPlan) -> ExecutionResult:
    """Execute one verified plan without invoking planner or compiler again."""

    from .execution import _normalize_execution_output, run_native

    validate_plan_environment(execution_plan)
    decision = plan_decision(execution_plan)
    if decision["device"] == "cpu" and decision["backend"] == "pytorch":
        _require_local_cpu_capabilities(
            str(decision["backend"]),
            str(decision["precision"]),
        )
    source_ir = plan_program(execution_plan)
    execution_ir = plan_execution_program(execution_plan)
    noise_model = plan_noise_model(execution_plan)
    requests = tuple(source_ir.measurements)
    validate_measurements(requests, n_wires=source_ir.n_wires)
    mode = str(decision["mode"])
    if int(decision["world_size"]) > 1:
        mode = {
            "statevector": "distributed_statevector",
            "mps": "distributed_mps",
            "tensor_network": "distributed_tensor_network",
        }.get(mode, mode)
    targets = {
        "auto": "full_state",
        "state": "full_state",
        "expectation": "expectation",
        "samples": "samples",
        "amplitudes": "few_amplitudes",
    }
    precision = str(decision["precision"])
    selected_config = RuntimeConfig(
        backend=str(decision["backend"]),
        device=str(decision["device"]),
        complex_dtype=precision,
        real_dtype="float64" if precision == "complex128" else "float32",
        jax_enable_x64=precision == "complex128",
    )
    try:
        output, returned_plan = run_native(
            execution_ir,
            noise_model=noise_model,
            mode=mode,
            return_plan=True,
            _execution_plan=execution_plan,
            bsz=int(decision["batch_size"]),
            world_size=int(decision["world_size"]),
            device=str(decision["device"]),
            dtype=getattr(torch, precision),
            config=selected_config,
            memory_limit_bytes=decision["memory_limit_bytes"],
            output_target=targets[str(decision["target"])],
            require_gradients=bool(decision["require_gradients"]),
            allow_approximate=bool(decision["allow_approximate"]),
        )
    except FlagQuantumError:
        raise
    except Exception as error:
        raise ExecutionError("planned execution failed") from error
    if returned_plan is not execution_plan:
        raise ExecutionPlanContractError(
            "identity_mismatch",
            "executor replaced the supplied plan instead of executing it",
        )
    return _normalize_execution_output(
        output,
        execution_plan,
        source_ir=source_ir,
        requests=requests,
        mode=mode,
        noise_model=noise_model,
    )


__all__ = (
    "execute_plan",
    "execution_plan_precision_requirements",
    "match_execution_plan_target_candidates",
)
