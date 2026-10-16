"""Local execution and evidence composition for P0 research capabilities."""

from __future__ import annotations

import hashlib
import json
import platform
import time
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from typing import Any, Literal, cast

from ..version import __version__
from .ground_state import (
    GroundStateVQERequest,
    energy_variance_reference,
    exact_sector_reference,
)
from .research_capabilities import (
    EnergyVarianceRequest,
    ExactGroundStateRequest,
    ResearchResourceEstimate,
    ResearchValidationIssue,
    ResearchValidationReport,
    _normalize_research_request,
    describe_research_capability,
    estimate_research_capability,
    validate_research_capability_input,
)
from .research_evidence_shape import _BUNDLE_SCHEMA, _evidence_shape_error
from .research_options import (
    ResearchExecutionOptions,
    validate_research_execution_options,
)
from .research_providers import (
    ResearchVQEProvider,
    denied_vqe_provider_route,
    installed_vqe_providers,
    plan_vqe_provider_route,
    research_vqe_metrics,
)

_VQE_ID = "ground_state.solve.vqe"
_EXACT_ID = "ground_state.solve.exact"
_VARIANCE_ID = "validate.energy_variance"
_CONVERGENCE_WINDOW = 5
_CONVERGENCE_TOLERANCE = 5e-6


@dataclass(frozen=True, slots=True)
class ResearchCapabilityRef:
    """Exact scientific operation and provider used by one attempt."""

    capability_id: str
    version: str
    provider: str


@dataclass(frozen=True, slots=True)
class ResearchCapabilityError:
    """Structured failure with stage, recovery, and partial-evidence context."""

    code: str
    stage: str
    retryable: bool
    message: str
    context: Mapping[str, Any] = field(default_factory=dict)
    evidence_generated: tuple[str, ...] = ()
    suggested_actions: tuple[Mapping[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "stage": self.stage,
            "retryable": self.retryable,
            "message": self.message,
            "context": dict(self.context),
            "evidence_generated": self.evidence_generated,
            "suggested_actions": tuple(dict(item) for item in self.suggested_actions),
        }


@dataclass(frozen=True, slots=True)
class ResearchCapabilityAttempt:
    """One provider attempt retained even when validation later fails."""

    capability_id: str
    provider: str
    status: Literal["succeeded", "unsupported", "failed"]
    started_at: str
    completed_at: str
    error: ResearchCapabilityError | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.error is not None:
            payload["error"] = self.error.to_dict()
        return payload


@dataclass(frozen=True, slots=True)
class ResearchCapabilityEvidenceBundle:
    """Transport-neutral result, validation, attempt, and provenance envelope."""

    run_id: str
    capability: ResearchCapabilityRef
    status: Literal["succeeded", "unsupported", "validation_failed", "failed"]
    result: Mapping[str, Any]
    metrics: Mapping[str, Any]
    artifacts: tuple[Mapping[str, Any], ...]
    execution: Mapping[str, Any]
    validation: tuple[ResearchValidationReport, ...]
    attempts: tuple[ResearchCapabilityAttempt, ...]
    warnings: tuple[ResearchValidationIssue, ...]
    provenance: Mapping[str, Any]
    error: ResearchCapabilityError | None = None
    schema: str = _BUNDLE_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "capability": asdict(self.capability),
            "status": self.status,
            "result": dict(self.result),
            "metrics": dict(self.metrics),
            "artifacts": tuple(dict(item) for item in self.artifacts),
            "execution": dict(self.execution),
            "validation": tuple(item.to_dict() for item in self.validation),
            "attempts": tuple(item.to_dict() for item in self.attempts),
            "warnings": tuple(item.to_dict() for item in self.warnings),
            "provenance": dict(self.provenance),
            "error": None if self.error is None else self.error.to_dict(),
        }


_installed_vqe_providers = installed_vqe_providers


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _portable_evidence_value(value: Any) -> Any:
    """Normalize JSON numbers by IEEE-754 value, not client spelling."""

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return {"$flagquantum_number": float(value).hex()}
    if isinstance(value, Mapping):
        return {str(key): _portable_evidence_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_portable_evidence_value(item) for item in value]
    return value


def _evidence_sha256(payload: Mapping[str, Any]) -> str:
    """Hash evidence stably after JSON clients re-encode finite floats."""

    normalized = _portable_evidence_value(payload)
    assert isinstance(normalized, Mapping)
    return _canonical_sha256(normalized)


def _capability_ref(
    capability_id: str, *, provider: str | None = None
) -> ResearchCapabilityRef:
    manifest = describe_research_capability(capability_id)
    return ResearchCapabilityRef(
        capability_id=manifest.capability_id,
        version=manifest.version,
        provider=manifest.provider if provider is None else provider,
    )


def _request_sha256(request: object) -> str | None:
    if not is_dataclass(request) or isinstance(request, type):
        return None
    try:
        return _canonical_sha256(
            {
                "request_type": (
                    f"{type(request).__module__}.{type(request).__qualname__}"
                ),
                "request": asdict(request),
            }
        )
    except (TypeError, ValueError):
        return None


def _provenance_base(
    input_sha256: str | None, *, fallback_used: bool
) -> dict[str, Any]:
    try:
        import torch

        torch_version = torch.__version__
    except ImportError:
        torch_version = None
    return {
        "package": "flagquantum",
        "package_version": __version__,
        "python_version": platform.python_version(),
        "torch_version": torch_version,
        "fallback_used": fallback_used,
        "input_sha256": input_sha256,
        "deterministic": True,
        "random_seed": None,
    }


def _finalize_bundle(
    *,
    run_id: str,
    capability: ResearchCapabilityRef,
    status: Literal["succeeded", "unsupported", "validation_failed", "failed"],
    result: Mapping[str, Any],
    metrics: Mapping[str, Any],
    artifacts: tuple[Mapping[str, Any], ...],
    execution: Mapping[str, Any],
    validation: tuple[ResearchValidationReport, ...],
    attempts: tuple[ResearchCapabilityAttempt, ...],
    warnings: tuple[ResearchValidationIssue, ...] = (),
    error: ResearchCapabilityError | None = None,
    input_sha256: str | None = None,
    fallback_used: bool = False,
) -> ResearchCapabilityEvidenceBundle:
    provenance = _provenance_base(input_sha256, fallback_used=fallback_used)
    unsigned = {
        "schema": _BUNDLE_SCHEMA,
        "run_id": run_id,
        "capability": asdict(capability),
        "status": status,
        "result": dict(result),
        "metrics": dict(metrics),
        "artifacts": tuple(dict(item) for item in artifacts),
        "execution": dict(execution),
        "validation": tuple(item.to_dict() for item in validation),
        "attempts": tuple(item.to_dict() for item in attempts),
        "warnings": tuple(item.to_dict() for item in warnings),
        "provenance": provenance,
        "error": None if error is None else error.to_dict(),
    }
    provenance["content_sha256"] = _evidence_sha256(unsigned)
    return ResearchCapabilityEvidenceBundle(
        run_id=run_id,
        capability=capability,
        status=status,
        result=result,
        metrics=metrics,
        artifacts=artifacts,
        execution=execution,
        validation=validation,
        attempts=attempts,
        warnings=warnings,
        provenance=provenance,
        error=error,
    )


def _error_from_validation(
    report: ResearchValidationReport, *, stage: str
) -> ResearchCapabilityError:
    issue = report.blockers[0]
    return ResearchCapabilityError(
        code=issue.code,
        stage=stage,
        retryable=False,
        message=issue.message,
        context=issue.context,
        evidence_generated=("input_validation",),
        suggested_actions=({"type": "change_input"},),
    )


def _refused_bundle(
    *,
    run_id: str,
    capability: ResearchCapabilityRef,
    started_at: str,
    started_clock: float,
    report: ResearchValidationReport,
    stage: str,
    target: str,
    provider_order: tuple[str, ...],
    estimate: ResearchResourceEstimate | None = None,
    input_sha256: str | None = None,
    stage_durations: Mapping[str, float] | None = None,
) -> ResearchCapabilityEvidenceBundle:
    completed_at = _now()
    error = _error_from_validation(report, stage=stage)
    return _finalize_bundle(
        run_id=run_id,
        capability=capability,
        status="unsupported",
        result={},
        metrics={} if estimate is None else {"resource_estimate": estimate.to_dict()},
        artifacts=(),
        execution={
            "target": target,
            "execution_mode": "local_statevector",
            "started_at": started_at,
            "completed_at": completed_at,
            "duration_seconds": time.perf_counter() - started_clock,
            "provider_order": list(provider_order),
            "fallback_used": False,
            "stage_durations_seconds": dict(stage_durations or {}),
            "call_counts": {"provider": 0, "validators": 0},
        },
        validation=(report,),
        attempts=(),
        error=error,
        input_sha256=input_sha256,
    )


def _scientific_report(
    validator_id: str,
    *,
    passed: bool,
    metrics: Mapping[str, float | int | str | bool | None],
    message: str,
) -> ResearchValidationReport:
    blockers: tuple[ResearchValidationIssue, ...] = ()
    if not passed:
        blockers = (
            ResearchValidationIssue(
                "SCIENTIFIC_VALIDATION_FAILED",
                message,
                dict(metrics),
            ),
        )
    return ResearchValidationReport(
        passed=passed,
        validator_id=validator_id,
        blockers=blockers,
        metrics=metrics,
    )


def _execute_vqe(
    request: GroundStateVQERequest,
    *,
    provider: ResearchVQEProvider,
    run_id: str,
    capability: ResearchCapabilityRef,
    started_at: str,
    started_clock: float,
    options: ResearchExecutionOptions,
    estimate: ResearchResourceEstimate,
    input_sha256: str,
    stage_durations: Mapping[str, float],
    prior_attempts: tuple[ResearchCapabilityAttempt, ...],
    provider_order: tuple[str, ...],
    fallback_used: bool,
    route_plan: Mapping[str, Any],
) -> ResearchCapabilityEvidenceBundle:
    attempt_started_at = _now()
    execution_clock = time.perf_counter()
    try:
        result = provider.run_vqe(request)
    except (MemoryError, RuntimeError, ValueError) as exc:
        measured_stages = dict(stage_durations)
        measured_stages["execution"] = time.perf_counter() - execution_clock
        completed_at = _now()
        error = ResearchCapabilityError(
            code="CAPABILITY_EXECUTION_FAILED",
            stage="execution",
            retryable=False,
            message=str(exc),
            context={"exception_type": type(exc).__name__},
            evidence_generated=("input_validation", "resource_estimate"),
        )
        attempt = ResearchCapabilityAttempt(
            capability.capability_id,
            capability.provider,
            "failed",
            attempt_started_at,
            completed_at,
            error,
        )
        attempts = (*prior_attempts, attempt)
        return _finalize_bundle(
            run_id=run_id,
            capability=capability,
            status="failed",
            result={},
            metrics=research_vqe_metrics(estimate, route_plan),
            artifacts=(),
            execution={
                "target": options.target,
                "execution_mode": "local_statevector",
                "started_at": started_at,
                "completed_at": completed_at,
                "duration_seconds": time.perf_counter() - started_clock,
                "provider_order": list(provider_order),
                "fallback_used": fallback_used,
                "stage_durations_seconds": measured_stages,
                "call_counts": {"provider": len(attempts), "validators": 0},
            },
            validation=(),
            attempts=attempts,
            error=error,
            input_sha256=input_sha256,
            fallback_used=fallback_used,
        )
    measured_stages = dict(stage_durations)
    measured_stages["execution"] = time.perf_counter() - execution_clock
    if (
        result.status != "completed"
        or result.statevector is None
        or result.energy is None
    ):
        reason = result.unsupported
        error = ResearchCapabilityError(
            code="EXECUTION_DID_NOT_COMPLETE" if reason is None else reason.code,
            stage="execution",
            retryable=False,
            message=(
                "VQE execution did not complete" if reason is None else reason.message
            ),
            context={} if reason is None else reason.context,
            evidence_generated=("input_validation", "resource_estimate"),
        )
        completed_at = _now()
        attempt_status: Literal["unsupported", "failed"] = (
            "unsupported" if reason is not None else "failed"
        )
        attempt = ResearchCapabilityAttempt(
            capability.capability_id,
            capability.provider,
            attempt_status,
            attempt_started_at,
            completed_at,
            error,
        )
        attempts = (*prior_attempts, attempt)
        return _finalize_bundle(
            run_id=run_id,
            capability=capability,
            status="unsupported" if reason is not None else "failed",
            result=result.to_dict(),
            metrics=research_vqe_metrics(estimate, route_plan),
            artifacts=(),
            execution={
                "target": options.target,
                "execution_mode": "local_statevector",
                "started_at": started_at,
                "completed_at": completed_at,
                "duration_seconds": time.perf_counter() - started_clock,
                "provider_order": list(provider_order),
                "fallback_used": fallback_used,
                "stage_durations_seconds": measured_stages,
                "call_counts": {"provider": len(attempts), "validators": 0},
            },
            validation=(),
            attempts=attempts,
            error=error,
            input_sha256=input_sha256,
            fallback_used=fallback_used,
        )

    validation_clock = time.perf_counter()
    exact = exact_sector_reference(
        request.hamiltonian,
        fixed_hamming_weight=request.fixed_hamming_weight or 0,
    )
    exact_error = (
        float("inf")
        if exact.energy is None
        else abs(float(result.energy) - float(exact.energy))
    )
    exact_passed = exact.status == "completed" and (
        exact_error <= options.exact_energy_tolerance
    )
    exact_report = _scientific_report(
        _EXACT_ID,
        passed=exact_passed,
        metrics={
            "vqe_energy": float(result.energy),
            "reference_energy": exact.energy,
            "absolute_error": exact_error,
            "tolerance": options.exact_energy_tolerance,
        },
        message="VQE energy exceeds the exact-reference tolerance",
    )
    variance = energy_variance_reference(
        request.hamiltonian,
        result.statevector,
        tolerance=options.energy_variance_tolerance,
    )
    variance_passed = variance.status == "completed" and variance.passed
    variance_report = _scientific_report(
        _VARIANCE_ID,
        passed=variance_passed,
        metrics={
            "energy": variance.energy,
            "variance": variance.variance,
            "tolerance": variance.tolerance,
        },
        message="final-state energy variance exceeds the required tolerance",
    )
    measured_stages["scientific_validation"] = time.perf_counter() - validation_clock
    recent_energies = tuple(
        point.energy for point in result.convergence_trace[-_CONVERGENCE_WINDOW:]
    )
    convergence_span = (
        max(recent_energies) - min(recent_energies)
        if len(recent_energies) >= 2
        else float("inf")
    )
    optimizer_converged = convergence_span <= _CONVERGENCE_TOLERANCE
    budget_exhausted = (
        not optimizer_converged
        and result.resources is not None
        and result.resources.objective_evaluations >= request.optimizer.max_evaluations
    )
    warnings: tuple[ResearchValidationIssue, ...] = ()
    if not optimizer_converged:
        warnings = (
            ResearchValidationIssue(
                "OPTIMIZER_CONVERGENCE_NOT_ESTABLISHED",
                "the final objective window did not meet the convergence tolerance",
                {
                    "window": _CONVERGENCE_WINDOW,
                    "observed_points": len(recent_energies),
                    "energy_span": convergence_span,
                    "tolerance": _CONVERGENCE_TOLERANCE,
                    "budget_exhausted": budget_exhausted,
                },
            ),
        )
    status: Literal["succeeded", "validation_failed"] = (
        "succeeded" if exact_passed and variance_passed else "validation_failed"
    )
    completed_at = _now()
    attempt = ResearchCapabilityAttempt(
        capability.capability_id,
        capability.provider,
        "succeeded",
        attempt_started_at,
        completed_at,
    )
    attempts = (*prior_attempts, attempt)
    validation = (exact_report, variance_report)
    validation_error: ResearchCapabilityError | None = None
    if status == "validation_failed":
        validation_error = ResearchCapabilityError(
            code="SCIENTIFIC_VALIDATION_FAILED",
            stage="scientific_validation",
            retryable=False,
            message="one or more mandatory scientific validators did not pass",
            evidence_generated=(
                "provider_result",
                "exact_reference",
                "energy_variance",
            ),
            suggested_actions=({"type": "inspect_validation_reports"},),
        )
    return _finalize_bundle(
        run_id=run_id,
        capability=capability,
        status=status,
        result=result.to_dict(),
        metrics=research_vqe_metrics(
            estimate,
            route_plan,
            {
                "objective_evaluations": result.resources.objective_evaluations
                if result.resources is not None
                else None,
                "exact_energy_error": exact_error,
                "energy_variance": variance.variance,
                "optimizer_converged": optimizer_converged,
                "optimizer_convergence_window": _CONVERGENCE_WINDOW,
                "optimizer_convergence_span": convergence_span,
                "optimizer_convergence_tolerance": _CONVERGENCE_TOLERANCE,
                "optimizer_budget_exhausted": budget_exhausted,
            },
        ),
        artifacts=(),
        execution={
            "target": options.target,
            "execution_mode": "local_statevector",
            "started_at": started_at,
            "completed_at": completed_at,
            "duration_seconds": time.perf_counter() - started_clock,
            "provider_order": list(provider_order),
            "fallback_used": fallback_used,
            "stage_durations_seconds": measured_stages,
            "call_counts": {"provider": len(attempts), "validators": 2},
        },
        validation=validation,
        attempts=attempts,
        warnings=warnings,
        error=validation_error,
        input_sha256=input_sha256,
        fallback_used=fallback_used,
    )


def _execute_exact(
    request: ExactGroundStateRequest,
    *,
    capability: ResearchCapabilityRef,
    run_id: str,
    started_at: str,
    started_clock: float,
    options: ResearchExecutionOptions,
    estimate: ResearchResourceEstimate,
    input_sha256: str,
    stage_durations: Mapping[str, float],
) -> ResearchCapabilityEvidenceBundle:
    execution_clock = time.perf_counter()
    result = exact_sector_reference(
        request.hamiltonian,
        fixed_hamming_weight=request.fixed_hamming_weight,
    )
    measured_stages = dict(stage_durations)
    measured_stages["execution"] = time.perf_counter() - execution_clock
    completed_at = _now()
    completed = result.status == "completed"
    attempt = ResearchCapabilityAttempt(
        capability.capability_id,
        capability.provider,
        "succeeded" if completed else "unsupported",
        started_at,
        completed_at,
    )
    return _finalize_bundle(
        run_id=run_id,
        capability=capability,
        status="succeeded" if completed else "unsupported",
        result=result.to_dict(),
        metrics={"resource_estimate": estimate.to_dict()},
        artifacts=(),
        execution={
            "target": options.target,
            "execution_mode": "local_statevector",
            "started_at": started_at,
            "completed_at": completed_at,
            "duration_seconds": time.perf_counter() - started_clock,
            "provider_order": [capability.provider],
            "fallback_used": False,
            "stage_durations_seconds": measured_stages,
            "call_counts": {"provider": 1, "validators": 0},
        },
        validation=(),
        attempts=(attempt,),
        input_sha256=input_sha256,
    )


def _execute_variance(
    request: EnergyVarianceRequest,
    *,
    capability: ResearchCapabilityRef,
    run_id: str,
    started_at: str,
    started_clock: float,
    options: ResearchExecutionOptions,
    estimate: ResearchResourceEstimate,
    input_sha256: str,
    stage_durations: Mapping[str, float],
) -> ResearchCapabilityEvidenceBundle:
    execution_clock = time.perf_counter()
    result = energy_variance_reference(
        request.hamiltonian,
        request.statevector,
        tolerance=request.tolerance,
    )
    measured_stages = dict(stage_durations)
    measured_stages["execution"] = time.perf_counter() - execution_clock
    completed_at = _now()
    completed = result.status == "completed"
    passed = completed and result.passed
    reports: tuple[ResearchValidationReport, ...] = ()
    if completed:
        validation_clock = time.perf_counter()
        reports = (
            _scientific_report(
                _VARIANCE_ID,
                passed=passed,
                metrics={
                    "energy": result.energy,
                    "variance": result.variance,
                    "tolerance": result.tolerance,
                },
                message="state energy variance exceeds the required tolerance",
            ),
        )
        measured_stages["scientific_validation"] = (
            time.perf_counter() - validation_clock
        )
    error: ResearchCapabilityError | None = None
    status: Literal["succeeded", "unsupported", "validation_failed"]
    if not completed:
        status = "unsupported"
    elif passed:
        status = "succeeded"
    else:
        status = "validation_failed"
        error = ResearchCapabilityError(
            code="SCIENTIFIC_VALIDATION_FAILED",
            stage="scientific_validation",
            retryable=False,
            message="the mandatory scientific validator did not pass",
            evidence_generated=("provider_result",),
        )
    attempt = ResearchCapabilityAttempt(
        capability.capability_id,
        capability.provider,
        "succeeded" if completed else "unsupported",
        started_at,
        completed_at,
    )
    return _finalize_bundle(
        run_id=run_id,
        capability=capability,
        status=status,
        result=result.to_dict(),
        metrics={"resource_estimate": estimate.to_dict()},
        artifacts=(),
        execution={
            "target": options.target,
            "execution_mode": "local_statevector",
            "started_at": started_at,
            "completed_at": completed_at,
            "duration_seconds": time.perf_counter() - started_clock,
            "provider_order": [capability.provider],
            "fallback_used": False,
            "stage_durations_seconds": measured_stages,
            "call_counts": {
                "provider": 1,
                "validators": 1 if completed else 0,
            },
        },
        validation=reports,
        attempts=(attempt,),
        error=error,
        input_sha256=input_sha256,
    )


def _route_vqe(
    request: GroundStateVQERequest,
    *,
    run_id: str,
    started_at: str,
    started_clock: float,
    options: ResearchExecutionOptions,
    estimate: ResearchResourceEstimate,
    input_sha256: str,
    stage_durations: Mapping[str, float],
) -> ResearchCapabilityEvidenceBundle:
    provider_order = options.provider_order
    route_clock = time.perf_counter()
    measured_stages = dict(stage_durations)
    if len(provider_order) > 1 and not options.allow_provider_fallback:
        measured_stages["environment_preflight"] = time.perf_counter() - route_clock
        completed_at = _now()
        error = ResearchCapabilityError(
            code="PROVIDER_FALLBACK_NOT_AUTHORIZED",
            stage="environment_preflight",
            retryable=False,
            message="multiple providers require explicit fallback authorization",
            context={"provider_order": provider_order},
            evidence_generated=("input_validation", "resource_estimate"),
            suggested_actions=(
                {"type": "select_one_provider"},
                {"type": "authorize_provider_fallback"},
            ),
        )
        return _finalize_bundle(
            run_id=run_id,
            capability=_capability_ref(_VQE_ID, provider=provider_order[0]),
            status="unsupported",
            result={},
            metrics=research_vqe_metrics(
                estimate, denied_vqe_provider_route(provider_order)
            ),
            artifacts=(),
            execution={
                "target": options.target,
                "execution_mode": "local_statevector",
                "started_at": started_at,
                "completed_at": completed_at,
                "duration_seconds": time.perf_counter() - started_clock,
                "provider_order": list(provider_order),
                "fallback_used": False,
                "stage_durations_seconds": measured_stages,
                "call_counts": {"provider": 0, "validators": 0},
            },
            validation=(),
            attempts=(),
            error=error,
            input_sha256=input_sha256,
        )

    by_id, route_plan = plan_vqe_provider_route(
        request, provider_order, _installed_vqe_providers()
    )
    attempts: list[ResearchCapabilityAttempt] = []
    candidates = route_plan["candidates"]
    assert isinstance(candidates, list)
    for index, candidate in enumerate(candidates):
        assert isinstance(candidate, Mapping)
        provider_id = str(candidate["provider"])
        provider = by_id.get(provider_id)
        attempt_started_at = _now()
        availability_reason = candidate["reason"]
        error_code = str(candidate["preflight_error_code"])
        if candidate["available"] is True:
            measured_stages["environment_preflight"] = time.perf_counter() - route_clock
            assert provider is not None
            return _execute_vqe(
                request,
                provider=provider,
                run_id=run_id,
                capability=_capability_ref(_VQE_ID, provider=provider_id),
                started_at=started_at,
                started_clock=started_clock,
                options=options,
                estimate=estimate,
                input_sha256=input_sha256,
                stage_durations=measured_stages,
                prior_attempts=tuple(attempts),
                provider_order=provider_order,
                fallback_used=index > 0,
                route_plan=route_plan,
            )
        assert isinstance(availability_reason, str)
        completed_at = _now()
        error = ResearchCapabilityError(
            code=error_code,
            stage="environment_preflight",
            retryable=True,
            message=availability_reason,
            context={"provider": provider_id},
            evidence_generated=("input_validation", "resource_estimate"),
            suggested_actions=({"type": "install_or_enable_provider"},),
        )
        attempts.append(
            ResearchCapabilityAttempt(
                _VQE_ID,
                provider_id,
                "unsupported",
                attempt_started_at,
                completed_at,
                error,
            )
        )
        has_next = index + 1 < len(provider_order)
        if has_next and options.allow_provider_fallback:
            continue
        measured_stages["environment_preflight"] = time.perf_counter() - route_clock
        fallback_used = index > 0
        return _finalize_bundle(
            run_id=run_id,
            capability=_capability_ref(_VQE_ID, provider=provider_id),
            status="unsupported",
            result={},
            metrics=research_vqe_metrics(estimate, route_plan),
            artifacts=(),
            execution={
                "target": options.target,
                "execution_mode": "local_statevector",
                "started_at": started_at,
                "completed_at": completed_at,
                "duration_seconds": time.perf_counter() - started_clock,
                "provider_order": list(provider_order),
                "fallback_used": fallback_used,
                "stage_durations_seconds": measured_stages,
                "call_counts": {"provider": len(attempts), "validators": 0},
            },
            validation=(),
            attempts=tuple(attempts),
            error=error,
            input_sha256=input_sha256,
            fallback_used=fallback_used,
        )
    raise AssertionError("validated provider order cannot be empty")


def execute_research_capability(
    capability_id: str,
    request: object,
    options: ResearchExecutionOptions | Mapping[str, Any] | None = None,
) -> ResearchCapabilityEvidenceBundle:
    """Execute one bounded local research capability with fail-closed evidence."""

    request = _normalize_research_request(capability_id, request)
    normalized_options = validate_research_execution_options(options)
    if capability_id != _VQE_ID and (
        normalized_options.provider_order != ("flagquantum-native",)
        or normalized_options.allow_provider_fallback
    ):
        raise ValueError(
            "provider routing is currently available only for ground_state.solve.vqe"
        )
    capability = _capability_ref(
        capability_id,
        provider=(
            normalized_options.provider_order[0] if capability_id == _VQE_ID else None
        ),
    )
    run_id = uuid.uuid4().hex
    started_at = _now()
    started_clock = time.perf_counter()
    input_sha256 = _request_sha256(request)
    validation_clock = time.perf_counter()
    validation = validate_research_capability_input(capability_id, request)
    stage_durations = {
        "input_validation": time.perf_counter() - validation_clock,
    }
    if not validation.passed:
        return _refused_bundle(
            run_id=run_id,
            capability=capability,
            started_at=started_at,
            started_clock=started_clock,
            report=validation,
            stage="input_validation",
            target=normalized_options.target,
            provider_order=normalized_options.provider_order,
            input_sha256=input_sha256,
            stage_durations=stage_durations,
        )
    assert input_sha256 is not None
    estimate_clock = time.perf_counter()
    estimate = estimate_research_capability(
        capability_id, request, target=normalized_options.target
    )
    stage_durations["resource_estimation"] = time.perf_counter() - estimate_clock
    if not estimate.supported:
        report = ResearchValidationReport(
            passed=False,
            validator_id=f"{capability_id}.resource_estimation",
            blockers=estimate.blockers,
        )
        return _refused_bundle(
            run_id=run_id,
            capability=capability,
            started_at=started_at,
            started_clock=started_clock,
            report=report,
            stage="resource_estimation",
            target=normalized_options.target,
            provider_order=normalized_options.provider_order,
            estimate=estimate,
            input_sha256=input_sha256,
            stage_durations=stage_durations,
        )
    try:
        if capability_id == _VQE_ID:
            return _route_vqe(
                cast(GroundStateVQERequest, request),
                run_id=run_id,
                started_at=started_at,
                started_clock=started_clock,
                options=normalized_options,
                estimate=estimate,
                input_sha256=input_sha256,
                stage_durations=stage_durations,
            )
        if capability_id == _EXACT_ID:
            assert isinstance(request, ExactGroundStateRequest)
            return _execute_exact(
                request,
                run_id=run_id,
                capability=capability,
                started_at=started_at,
                started_clock=started_clock,
                options=normalized_options,
                estimate=estimate,
                input_sha256=input_sha256,
                stage_durations=stage_durations,
            )
        assert isinstance(request, EnergyVarianceRequest)
        return _execute_variance(
            request,
            run_id=run_id,
            capability=capability,
            started_at=started_at,
            started_clock=started_clock,
            options=normalized_options,
            estimate=estimate,
            input_sha256=input_sha256,
            stage_durations=stage_durations,
        )
    except (MemoryError, RuntimeError, ValueError) as exc:
        stage_durations["execution"] = max(
            0.0,
            time.perf_counter() - started_clock - sum(stage_durations.values()),
        )
        completed_at = _now()
        error = ResearchCapabilityError(
            code="CAPABILITY_EXECUTION_FAILED",
            stage="execution",
            retryable=isinstance(exc, MemoryError),
            message=str(exc),
            context={"exception_type": type(exc).__name__},
            evidence_generated=("input_validation", "resource_estimate"),
        )
        attempt = ResearchCapabilityAttempt(
            capability.capability_id,
            capability.provider,
            "failed",
            started_at,
            completed_at,
            error,
        )
        return _finalize_bundle(
            run_id=run_id,
            capability=capability,
            status="failed",
            result={},
            metrics={"resource_estimate": estimate.to_dict()},
            artifacts=(),
            execution={
                "target": normalized_options.target,
                "execution_mode": "local_statevector",
                "started_at": started_at,
                "completed_at": completed_at,
                "duration_seconds": time.perf_counter() - started_clock,
                "provider_order": list(normalized_options.provider_order),
                "fallback_used": False,
                "stage_durations_seconds": stage_durations,
                "call_counts": {"provider": 1, "validators": 0},
            },
            validation=(),
            attempts=(attempt,),
            error=error,
            input_sha256=input_sha256,
        )


def _unsigned_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    provenance = payload["provenance"]
    if not isinstance(provenance, Mapping):
        raise TypeError("evidence provenance must be a mapping")
    unsigned_provenance = dict(provenance)
    unsigned_provenance.pop("content_sha256", None)
    unsigned = {
        key: payload[key]
        for key in (
            "schema",
            "run_id",
            "capability",
            "status",
            "result",
            "metrics",
            "artifacts",
            "execution",
            "validation",
            "attempts",
            "warnings",
            "error",
        )
    }
    unsigned["provenance"] = unsigned_provenance
    return unsigned


def _invalid_evidence(
    validator_id: str, code: str, message: str
) -> ResearchValidationReport:
    return ResearchValidationReport(
        False,
        validator_id,
        blockers=(ResearchValidationIssue(code, message),),
    )


def validate_research_evidence(
    validator_id: str,
    evidence: ResearchCapabilityEvidenceBundle | Mapping[str, Any],
) -> ResearchValidationReport:
    """Verify bundle integrity and return the recorded validator outcome."""

    payload = (
        evidence.to_dict()
        if isinstance(evidence, ResearchCapabilityEvidenceBundle)
        else dict(evidence)
    )
    shape_error = _evidence_shape_error(payload)
    if shape_error is not None:
        return _invalid_evidence(validator_id, *shape_error)
    provenance = payload["provenance"]
    assert isinstance(provenance, Mapping)
    try:
        expected = _evidence_sha256(_unsigned_payload(payload))
    except (KeyError, TypeError, ValueError) as exc:
        return _invalid_evidence(
            validator_id,
            "INVALID_EVIDENCE_ENCODING",
            f"evidence cannot be canonically encoded: {exc}",
        )
    if provenance.get("content_sha256") != expected:
        return ResearchValidationReport(
            False,
            validator_id,
            blockers=(
                ResearchValidationIssue(
                    "EVIDENCE_INTEGRITY_MISMATCH",
                    "research capability evidence content hash does not match",
                ),
            ),
        )
    capability = payload["capability"]
    assert isinstance(capability, Mapping)
    try:
        manifest = describe_research_capability(
            str(capability["capability_id"]), str(capability["version"])
        )
    except KeyError:
        return _invalid_evidence(
            validator_id,
            "EVIDENCE_CAPABILITY_IDENTITY_MISMATCH",
            "evidence capability identity is not installed",
        )
    installed_providers = (
        {provider.provider_id for provider in _installed_vqe_providers()}
        if capability["capability_id"] == _VQE_ID
        else {manifest.provider}
    )
    if capability["provider"] not in installed_providers:
        return _invalid_evidence(
            validator_id,
            "EVIDENCE_CAPABILITY_IDENTITY_MISMATCH",
            "evidence provider is not installed for the capability",
        )
    reports = payload.get("validation")
    if not isinstance(reports, (list, tuple)):
        reports = ()
    for report in reports:
        if isinstance(report, Mapping) and report.get("validator_id") == validator_id:
            blockers = tuple(
                ResearchValidationIssue(
                    str(item.get("code", "INVALID_VALIDATION_BLOCKER")),
                    str(item.get("message", "invalid validation blocker")),
                    item.get("context", {}),
                )
                for item in report.get("blockers", ())
                if isinstance(item, Mapping)
            )
            metrics = report.get("metrics", {})
            return ResearchValidationReport(
                passed=report.get("passed") is True,
                validator_id=validator_id,
                blockers=blockers,
                metrics=metrics if isinstance(metrics, Mapping) else {},
            )
    return ResearchValidationReport(
        False,
        validator_id,
        blockers=(
            ResearchValidationIssue(
                "VALIDATOR_EVIDENCE_MISSING",
                "the evidence bundle has no record for the requested validator",
            ),
        ),
    )
