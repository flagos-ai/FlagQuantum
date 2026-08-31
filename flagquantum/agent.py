"""Fail-closed, machine-readable interfaces for agent-driven workloads.

This module contains no LLM or orchestration dependencies.  It adapts the
existing IR, planner, capability registry, and deployment contracts into
stable-shaped reports that an external agent can consume without parsing
exception strings.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    severity: str = "error"
    context: Mapping[str, Any] = field(default_factory=dict)
    suggestions: tuple[Mapping[str, Any], ...] = ()
    retryable: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ValidationReport:
    valid: bool
    errors: tuple[ValidationIssue, ...] = ()
    warnings: tuple[ValidationIssue, ...] = ()
    capabilities: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AgentExecutionPlan:
    executable: bool
    selected_backend: str | None
    validation: ValidationReport
    plan: Mapping[str, Any] | None = None
    blockers: tuple[ValidationIssue, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DeploymentPreflightReport:
    approved_for_submission: bool
    validation: ValidationReport
    backend: Mapping[str, Any]
    shots: int
    package: Any | None = field(default=None, repr=False, compare=False)
    blockers: tuple[ValidationIssue, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "approved_for_submission": self.approved_for_submission,
            "validation": self.validation.to_dict(),
            "backend": dict(self.backend),
            "shots": self.shots,
            "blockers": tuple(blocker.to_dict() for blocker in self.blockers),
        }


def _issue(
    code: str,
    error: BaseException | str,
    *,
    context: Mapping[str, Any] | None = None,
    suggestions: tuple[Mapping[str, Any], ...] = (),
    retryable: bool = False,
) -> ValidationIssue:
    return ValidationIssue(
        code=code,
        message=str(error),
        context=dict(context or {}),
        suggestions=suggestions,
        retryable=retryable,
    )


def validate(
    circuit_or_ir: Any,
    *,
    backend: Any | None = None,
    requires_gradient: bool = False,
) -> ValidationReport:
    """Validate a program and optional deployment target without executing it."""

    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []
    capabilities: list[str] = []
    try:
        ir = circuit_or_ir.to_ir() if hasattr(circuit_or_ir, "to_ir") else circuit_or_ir
        if not hasattr(ir, "n_wires") or not hasattr(ir, "instructions"):
            raise TypeError("expected fq.Circuit or CircuitIR")
        if int(ir.n_wires) <= 0:
            raise ValueError("a quantum program must contain at least one wire")
        capabilities.append("valid_ir")
    except (TypeError, ValueError, AttributeError) as exc:
        errors.append(_issue("INVALID_QUANTUM_PROGRAM", exc))
        return ValidationReport(valid=False, errors=tuple(errors))

    if requires_gradient:
        trainable = any(
            bool(getattr(parameter, "requires_grad", False))
            for instruction in ir.instructions
            for parameter in getattr(instruction, "params", {}).values()
        )
        if trainable:
            capabilities.append("autograd")
        else:
            warnings.append(
                ValidationIssue(
                    code="NO_TRAINABLE_PARAMETERS",
                    message="gradient execution was requested but no trainable parameter was found",
                    severity="warning",
                    suggestions=({"action": "provide_trainable_parameters"},),
                )
            )

    if backend is not None:
        if int(ir.n_wires) > int(backend.n_wires):
            errors.append(
                ValidationIssue(
                    code="BACKEND_CAPACITY_EXCEEDED",
                    message=f"program needs {ir.n_wires} wires but backend exposes {backend.n_wires}",
                    context={
                        "required_wires": ir.n_wires,
                        "available_wires": backend.n_wires,
                    },
                    suggestions=({"action": "select_larger_backend"},),
                )
            )
        else:
            capabilities.append("backend_capacity")
        basis_gates = {str(name).lower() for name in backend.basis_gates}
        unsupported = sorted(
            {
                instruction.name
                for instruction in ir.instructions
                if basis_gates
                and not instruction.metadata.get("is_channel")
                and not instruction.metadata.get("is_dynamic")
                and instruction.name not in basis_gates
            }
        )
        if unsupported:
            errors.append(
                ValidationIssue(
                    code="BACKEND_GATE_UNSUPPORTED",
                    message="backend does not support one or more program gates",
                    context={
                        "unsupported_gates": unsupported,
                        "basis_gates": sorted(basis_gates),
                    },
                    suggestions=({"action": "compile_to_backend_basis"},),
                )
            )
        elif basis_gates:
            capabilities.append("backend_gate_set")

    return ValidationReport(
        valid=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        capabilities=tuple(capabilities),
        metadata={"n_wires": int(ir.n_wires), "n_instructions": len(ir.instructions)},
    )


def capabilities(*, refresh: bool = False) -> dict[str, Any]:
    """Return the installed runtime capabilities in a machine-readable shape."""

    from .runtime.backend_registry import capability_summary, list_backends
    from .version import __version__

    backends = {}
    for name in list_backends(refresh=refresh):
        try:
            backends[name] = capability_summary(name, refresh=refresh)
        except (RuntimeError, ValueError, ImportError) as exc:
            backends[name] = {"name": name, "available": False, "reason": str(exc)}
    return {
        "schema": "flagquantum_agent_capabilities_v1",
        "version": __version__,
        "backends": backends,
        "contracts": {
            "ir_validation": True,
            "execution_planning": True,
            "deployment_preflight": True,
            "deployment_identity_validation": True,
        },
    }


def preflight_execution(circuit_or_ir: Any, **options: Any) -> AgentExecutionPlan:
    """Validate and plan a workload, converting all expected failures to blockers."""

    from .compilation.planner import plan

    requires_gradient = bool(options.get("require_gradients", False))
    validation = validate(circuit_or_ir, requires_gradient=requires_gradient)
    if not validation.valid:
        return AgentExecutionPlan(False, None, validation, blockers=validation.errors)
    try:
        native_plan = plan(circuit_or_ir, **options)
        summary = native_plan.summary()
        return AgentExecutionPlan(
            executable=True,
            selected_backend=str(summary["recommended_mode"]),
            validation=validation,
            plan=summary,
        )
    except (TypeError, ValueError, RuntimeError, MemoryError) as exc:
        blocker = _issue(
            "EXECUTION_PLANNING_FAILED",
            exc,
            suggestions=({"action": "change_backend_or_resource_limits"},),
            retryable=True,
        )
        return AgentExecutionPlan(False, None, validation, blockers=(blocker,))


def preflight_deployment(
    circuit_or_ir: Any,
    *,
    backend: Any,
    shots: int = 1024,
    name: str = "agent-workload",
) -> DeploymentPreflightReport:
    """Build and identity-check a deployment package without submitting it."""

    from .deployment import create_deployment_package, validate_deployment_package

    validation = validate(circuit_or_ir, backend=backend)
    backend_summary = {
        "provider": backend.provider,
        "name": backend.name,
        "n_wires": backend.n_wires,
    }
    blockers: list[ValidationIssue] = list(validation.errors)
    if int(shots) <= 0:
        blockers.append(_issue("INVALID_SHOT_COUNT", "shots must be positive"))
    package = None
    if not blockers:
        try:
            package = create_deployment_package(
                circuit_or_ir, backend=backend, shots=int(shots), name=name
            )
            validate_deployment_package(package)
        except (TypeError, ValueError, RuntimeError) as exc:
            blockers.append(
                _issue(
                    "DEPLOYMENT_PREFLIGHT_FAILED",
                    exc,
                    context=backend_summary,
                    suggestions=({"action": "inspect_backend_topology_and_gate_set"},),
                )
            )
    return DeploymentPreflightReport(
        approved_for_submission=not blockers,
        validation=validation,
        backend=backend_summary,
        shots=int(shots),
        package=package,
        blockers=tuple(blockers),
    )


__all__ = [
    "AgentExecutionPlan",
    "DeploymentPreflightReport",
    "ValidationIssue",
    "ValidationReport",
    "capabilities",
    "preflight_deployment",
    "preflight_execution",
    "validate",
]
