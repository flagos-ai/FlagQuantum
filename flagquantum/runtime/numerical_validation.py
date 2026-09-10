"""Small end-to-end numerical certification workloads for statevector runtimes."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from threading import Lock
from typing import Any

import torch

from ..core.numerics import AccuracyRequirementContract, PrecisionPlanContract

NUMERICAL_VALIDATION_VERSION = "1.0"
_CACHE: dict[tuple[str, str, str, str, str], "NumericalValidationReport"] = {}
_CACHE_LOCK = Lock()


@dataclass(frozen=True)
class NumericalValidationReport:
    workload: str
    device_type: str
    dtype: str
    provider: str
    accuracy_requirement_hash: str
    precision_plan_hash: str
    metrics: dict[str, float]
    passed: bool
    blockers: tuple[str, ...]
    deterministic: bool
    convergence_evidence: bool
    reference: str = "cpu_complex128"
    validation_version: str = NUMERICAL_VALIDATION_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def require_accepted(self) -> None:
        if not self.passed:
            raise RuntimeError(
                f"numerical validation {self.workload!r} failed: "
                + "; ".join(self.blockers)
            )


def _metrics(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    candidate_gradient: torch.Tensor,
    reference_gradient: torch.Tensor,
) -> dict[str, float]:
    actual = candidate.detach().cpu().to(torch.complex128).reshape(-1)
    expected = reference.detach().cpu().to(torch.complex128).reshape(-1)
    actual_norm = torch.linalg.vector_norm(actual)
    expected_norm = torch.linalg.vector_norm(expected)
    overlap = torch.abs(torch.vdot(expected, actual)).square()
    denominator = expected_norm.square() * actual_norm.square()
    infidelity = 1.0 - float((overlap / denominator).real.item())

    actual_grad = candidate_gradient.detach().cpu().to(torch.float64).reshape(-1)
    expected_grad = reference_gradient.detach().cpu().to(torch.float64).reshape(-1)
    gradient_delta = torch.linalg.vector_norm(actual_grad - expected_grad)
    gradient_scale = max(float(torch.linalg.vector_norm(expected_grad).item()), 1e-15)
    cosine_denominator = float(
        (
            torch.linalg.vector_norm(actual_grad)
            * torch.linalg.vector_norm(expected_grad)
        ).item()
    )
    gradient_cosine = (
        float(torch.dot(actual_grad, expected_grad).item()) / cosine_denominator
        if cosine_denominator > 0.0
        else 1.0
    )
    weights = torch.linspace(-1.0, 1.0, expected.numel(), dtype=torch.float64)
    actual_expectation = torch.sum(torch.abs(actual).square() * weights)
    expected_expectation = torch.sum(torch.abs(expected).square() * weights)
    expectation_scale = max(abs(float(expected_expectation.item())), 1e-15)
    return {
        "norm_drift": abs(float(actual_norm.item()) - 1.0),
        "reference_norm_drift": abs(float(expected_norm.item()) - 1.0),
        "state_max_abs_error": float(torch.max(torch.abs(actual - expected)).item()),
        "state_infidelity": max(0.0, infidelity),
        "expectation_abs_error": abs(
            float(actual_expectation.item()) - float(expected_expectation.item())
        ),
        "expectation_rel_error": abs(
            float(actual_expectation.item()) - float(expected_expectation.item())
        )
        / expectation_scale,
        "gradient_rel_error": float(gradient_delta.item()) / gradient_scale,
        "gradient_cosine_similarity": gradient_cosine,
        # Dense statevector has neither tensor decomposition nor truncation.
        "decomposition_residual": 0.0,
        "truncation_error": 0.0,
    }


def _blockers(
    metrics: dict[str, float], requirement: AccuracyRequirementContract
) -> tuple[str, ...]:
    checks = (
        ("norm_drift", requirement.max_norm_drift, "max"),
        ("expectation_abs_error", requirement.max_expectation_abs_error, "max"),
        ("expectation_rel_error", requirement.max_expectation_rel_error, "max"),
        ("gradient_rel_error", requirement.max_gradient_rel_error, "max"),
        ("state_infidelity", requirement.max_state_infidelity, "max"),
        (
            "decomposition_residual",
            requirement.max_decomposition_residual,
            "max",
        ),
        ("truncation_error", requirement.max_truncation_error, "max"),
        (
            "gradient_cosine_similarity",
            requirement.min_gradient_cosine_similarity,
            "min",
        ),
    )
    failures = []
    for name, limit, direction in checks:
        value = metrics[name]
        if not math.isfinite(value):
            failures.append(f"{name} is not finite")
        elif limit is not None and direction == "max" and value > limit:
            failures.append(f"{name}={value} exceeds {limit}")
        elif limit is not None and direction == "min" and value < limit:
            failures.append(f"{name}={value} is below {limit}")
    return tuple(failures)


def _run_workload(
    *, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    from ..circuit import Circuit

    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    theta = torch.tensor(
        [0.37, -0.21, 0.43],
        dtype=real_dtype,
        device=device,
        requires_grad=True,
    )
    circuit = Circuit(3, device=device, dtype=dtype)
    circuit.gate("h", 0)
    circuit.gate("rx", 1, theta=theta[0])
    circuit.gate("ry", 2, theta=theta[1])
    circuit.gate("cx", (0, 1))
    circuit.gate("rzz", (1, 2), theta=theta[2])
    circuit.gate("rz", 0, theta=theta[0] - theta[1])
    state = circuit.state()
    weights = torch.linspace(-1.0, 1.0, state.numel(), dtype=real_dtype, device=device)
    loss = torch.sum(torch.abs(state.reshape(-1)).square() * weights)
    torch.autograd.backward(loss)
    if theta.grad is None:
        raise RuntimeError("certification workload did not produce gradients")
    return state, theta.grad


def certify_statevector_local_p0(
    *,
    device: str | torch.device,
    dtype: str | torch.dtype,
    provider: str,
    accuracy_requirement: AccuracyRequirementContract,
    precision_plan: PrecisionPlanContract,
    refresh: bool = False,
) -> NumericalValidationReport:
    """Validate a tiny differentiable circuit against CPU complex128."""

    resolved_device = torch.device(device)
    dtype_name = str(dtype).removeprefix("torch.")
    resolved_dtype = getattr(torch, dtype_name)
    key = (
        str(resolved_device),
        dtype_name,
        provider,
        accuracy_requirement.content_hash(),
        precision_plan.content_hash(),
    )
    with _CACHE_LOCK:
        if not refresh and key in _CACHE:
            return _CACHE[key]
    candidate, candidate_gradient = _run_workload(
        device=resolved_device, dtype=resolved_dtype
    )
    reference, reference_gradient = _run_workload(
        device=torch.device("cpu"), dtype=torch.complex128
    )
    metrics = _metrics(candidate, reference, candidate_gradient, reference_gradient)
    repeat, repeat_gradient = _run_workload(
        device=resolved_device, dtype=resolved_dtype
    )
    deterministic = bool(
        torch.equal(candidate.detach().cpu(), repeat.detach().cpu())
        and torch.equal(
            candidate_gradient.detach().cpu(), repeat_gradient.detach().cpu()
        )
    )
    blockers = list(_blockers(metrics, accuracy_requirement))
    if accuracy_requirement.require_determinism and not deterministic:
        blockers.append("repeated state or gradient was not bitwise deterministic")
    report = NumericalValidationReport(
        workload="statevector_local_p0",
        device_type=resolved_device.type,
        dtype=dtype_name,
        provider=provider,
        accuracy_requirement_hash=accuracy_requirement.content_hash(),
        precision_plan_hash=precision_plan.content_hash(),
        metrics=metrics,
        passed=not blockers,
        blockers=tuple(blockers),
        deterministic=deterministic,
        convergence_evidence=True,
    )
    with _CACHE_LOCK:
        _CACHE[key] = report
    return report


__all__ = (
    "NUMERICAL_VALIDATION_VERSION",
    "NumericalValidationReport",
    "certify_statevector_local_p0",
)
