"""Canonical single-device fast-path preflight checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from .execution import run_native


@dataclass(frozen=True)
class LocalFastPathPreflightReport:
    """Readiness report for CPU/single-GPU local execution paths."""

    passed: bool
    modes: tuple[str, ...]
    device: str
    reference_mode: str
    results: Mapping[str, Mapping[str, Any]]
    errors: tuple[str, ...]
    atol: float

    def summary(self) -> dict[str, Any]:
        return {
            "preflight": "local_fast_path",
            "passed": self.passed,
            "modes": self.modes,
            "device": self.device,
            "reference_mode": self.reference_mode,
            "results": {mode: dict(result) for mode, result in self.results.items()},
            "errors": self.errors,
            "atol": self.atol,
            "contract": "single-device fast paths must not depend on distributed runtime setup",
        }


def _default_local_circuit() -> Any:
    from ..circuit import Circuit

    circuit = Circuit(4)
    circuit.h(0).rx(1, theta=0.2).cx(0, 2).ry(3, theta=-0.4).rzz(2, 3, theta=0.3)
    return circuit


def _state(result: Any) -> torch.Tensor:
    if isinstance(result, torch.Tensor):
        return result
    if hasattr(result, "state"):
        value = result.state()
        if isinstance(value, torch.Tensor):
            return value
    if hasattr(result, "to_statevector"):
        value = result.to_statevector()
        if isinstance(value, torch.Tensor):
            return value
    raise TypeError(f"Cannot extract statevector from {type(result)!r}.")


def _max_error(left: torch.Tensor, right: torch.Tensor) -> float:
    right = right.to(device=left.device, dtype=left.dtype)
    return float(torch.max(torch.abs(left.detach() - right.detach())).item())


def local_fast_path_preflight(
    circuit_or_ir: Any | None = None,
    *,
    modes: Sequence[str] = ("statevector", "mps", "tensor_network"),
    device: str | torch.device = "cpu",
    atol: float = 1e-6,
    raise_on_error: bool = False,
    **options: Any,
) -> LocalFastPathPreflightReport:
    """Validate local execution modes without touching distributed backends."""

    program = circuit_or_ir if circuit_or_ir is not None else _default_local_circuit()
    resolved_device = str(device)
    run_options = dict(options)
    run_options.setdefault("device", resolved_device)
    reference, reference_plan = run_native(
        program,
        mode="statevector",
        return_plan=True,
        **run_options,
    )
    reference_state = _state(reference)
    results: dict[str, Mapping[str, Any]] = {}
    errors: list[str] = []
    for mode in modes:
        result, plan = run_native(
            program,
            mode=mode,
            return_plan=True,
            **run_options,
        )
        state = _state(result)
        error = _max_error(state, reference_state)
        plan_summary = plan.summary()
        mode_result = {
            "mode": mode,
            "state_mode": plan.state_mode,
            "recommended_mode": plan.recommended_mode,
            "distribution_semantics": plan_summary["distribution_semantics"],
            "scalability_claim_allowed": plan_summary["scalability_claim_allowed"],
            "world_size": plan.world_size,
            "max_abs_error_vs_statevector": error,
        }
        results[mode] = mode_result
        if plan.world_size != 1:
            errors.append(
                f"{mode}: local fast path must use world_size=1, got {plan.world_size}"
            )
        if plan_summary["distribution_semantics"] != "single_device_fast_path":
            errors.append(f"{mode}: expected single_device_fast_path semantics")
        if plan_summary["scalability_claim_allowed"]:
            errors.append(f"{mode}: local fast path cannot allow scalability claims")
        if error > float(atol):
            errors.append(
                f"{mode}: max_abs_error {error:.3e} exceeds atol {float(atol):.3e}"
            )
    out = LocalFastPathPreflightReport(
        passed=not errors,
        modes=tuple(modes),
        device=resolved_device,
        reference_mode=reference_plan.state_mode,
        results=results,
        errors=tuple(errors),
        atol=float(atol),
    )
    if raise_on_error and not out.passed:
        raise RuntimeError("; ".join(out.errors))
    return out


__all__ = ["LocalFastPathPreflightReport", "local_fast_path_preflight"]
