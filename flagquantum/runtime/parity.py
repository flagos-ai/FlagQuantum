"""Development/production distributed parity checks.

These helpers make the local development backend useful for production work:
the local simulator must follow the same IR, shard ownership, and task layout
that production distributed execution will use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from .backends.statevector.state import (
    plan_distributed_statevector,
    simulate_distributed_statevector_local,
)
from .distributed.backend_policy import resolve_distributed_backend_policy
from .execution import run_native


@dataclass(frozen=True)
class DistributedBackendParityReport:
    """Machine-readable development/production consistency report."""

    mode: str
    world_size: int
    passed: bool
    errors: tuple[str, ...]
    development_signature: Mapping[str, Any]
    production_signature: Mapping[str, Any]
    numerical_reference_error: float | None = None
    atol: float = 1e-6

    def summary(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "world_size": self.world_size,
            "passed": self.passed,
            "errors": self.errors,
            "development_signature": dict(self.development_signature),
            "production_signature": dict(self.production_signature),
            "numerical_reference_error": self.numerical_reference_error,
            "atol": self.atol,
            "contract": "development_backend_must_match_production_distribution_semantics",
        }


@dataclass(frozen=True)
class LocalDistributedPreflightReport:
    """Local development readiness check across distributed runtime modes."""

    passed: bool
    world_size: int
    modes: tuple[str, ...]
    development_policy: Mapping[str, Any]
    production_policy: Mapping[str, Any]
    parity_reports: Mapping[str, DistributedBackendParityReport]
    errors: tuple[str, ...]

    def summary(self) -> dict[str, Any]:
        return {
            "preflight": "local_distributed_development",
            "passed": self.passed,
            "world_size": self.world_size,
            "modes": self.modes,
            "development_policy": dict(self.development_policy),
            "production_policy": dict(self.production_policy),
            "parity_reports": {
                mode: report.summary() for mode, report in self.parity_reports.items()
            },
            "errors": self.errors,
            "contract": (
                "local development must match production shard/task layout and numerical behavior"
            ),
        }


def _normalize_mode(mode: str) -> str:
    if mode == "distributed":
        return "distributed_statevector"
    if mode == "distributed_tn":
        return "distributed_tensor_network"
    return mode


def _as_ir(circuit_or_ir: Any) -> Any:
    return circuit_or_ir.to_ir() if hasattr(circuit_or_ir, "to_ir") else circuit_or_ir


def _rank_shard_signature(shards: Any) -> tuple[dict[str, Any], ...]:
    out = []
    for shard in shards:
        out.append(
            {
                "rank": int(getattr(shard, "rank")),
                "amplitude_start": int(getattr(shard, "amplitude_start")),
                "amplitude_end": int(getattr(shard, "amplitude_end")),
                "local_amplitudes": int(getattr(shard, "local_amplitudes")),
                "local_state_bytes": int(getattr(shard, "local_state_bytes")),
            }
        )
    return tuple(out)


def _statevector_signature_from_plan(plan: Any) -> dict[str, Any]:
    summary = plan.summary()
    return {
        "state_mode": "distributed_statevector",
        "world_size": summary["world_size"],
        "local_world_size": summary["local_world_size"],
        "node_count": summary["node_count"],
        "distribution_semantics": summary["distribution_semantics"],
        "rank_shards": _rank_shard_signature(plan.shards),
        "communication_segments": tuple(
            {
                "kind": segment.kind,
                "communication": segment.communication,
                "wires": tuple(segment.wires),
                "gate_indices": tuple(segment.gate_indices),
            }
            for segment in plan.execution_segments
        ),
        "communication_tiers": {
            "model": summary.get("communication_tier_model"),
            "intra_node_communication_bytes": summary.get(
                "intra_node_communication_bytes"
            ),
            "inter_node_communication_bytes": summary.get(
                "inter_node_communication_bytes"
            ),
            "estimated_transfer_bytes": summary.get("estimated_transfer_bytes"),
        },
    }


def _summary_signature(summary: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "state_mode",
        "world_size",
        "local_world_size",
        "node_count",
        "rank_shards",
        "local_tensor_wires",
        "boundary_syncs",
        "boundary_protocols",
        "slice_tasks",
        "tasks_by_rank",
        "communication_tiers",
    )
    return {key: summary[key] for key in keys if key in summary}


def _state_max_error(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.detach()
    right = right.detach().to(device=left.device, dtype=left.dtype)
    return float(torch.max(torch.abs(left - right)).item())


def validate_development_production_parity(
    circuit_or_ir: Any,
    *,
    mode: str = "distributed_statevector",
    world_size: int = 2,
    atol: float = 1e-6,
    **options: Any,
) -> DistributedBackendParityReport:
    """Validate that local development distributed behavior matches production intent.

    For statevector, production parity is the exact distributed statevector plan
    because production execution requires a torchrun process group. For MPS and
    tensor-network modes, both development and production policies are executed
    through the same local orchestration backend so their layout signatures and
    numerical outputs can be compared without a cluster.
    """

    mode = _normalize_mode(mode)
    world_size = int(world_size)
    errors: list[str] = []

    if mode == "distributed_statevector":
        ir = _as_ir(circuit_or_ir)
        development_policy = resolve_distributed_backend_policy(profile="development")
        development = simulate_distributed_statevector_local(
            ir,
            world_size=world_size,
            bsz=int(options.get("bsz", 1)),
            device=options.get("device", "cpu"),
            dtype=options.get("dtype", torch.complex64),
            backend_policy=development_policy,
        )
        production_plan = plan_distributed_statevector(
            ir,
            world_size=world_size,
            bsz=int(options.get("bsz", 1)),
        )
        development_signature = _statevector_signature_from_plan(development.plan)
        production_signature = _statevector_signature_from_plan(production_plan)
        if development_signature != production_signature:
            errors.append(
                "development and production statevector shard signatures differ"
            )
        reference = run_native(ir, mode="statevector", **options)
        numerical_error = _state_max_error(development.state, reference)
        if numerical_error > float(atol):
            errors.append(
                f"development statevector numerical error {numerical_error:.3e} exceeds atol {float(atol):.3e}"
            )
        return DistributedBackendParityReport(
            mode=mode,
            world_size=world_size,
            passed=not errors,
            errors=tuple(errors),
            development_signature=development_signature,
            production_signature=production_signature,
            numerical_reference_error=numerical_error,
            atol=float(atol),
        )

    if mode not in {"distributed_mps", "distributed_tensor_network"}:
        raise ValueError(
            "mode must be distributed_statevector, distributed_mps, or distributed_tensor_network."
        )

    development_policy = resolve_distributed_backend_policy(profile="development")
    production_policy = resolve_distributed_backend_policy(profile="production")
    shared_options = dict(options)
    shared_options.setdefault("distributed_executor", "local")
    shared_options.setdefault("device", options.get("device", "cpu"))
    shared_options.setdefault("torch_backend", "local_tensor")

    development, _ = run_native(
        circuit_or_ir,
        mode=mode,
        world_size=world_size,
        return_plan=True,
        distributed_backend_policy=development_policy,
        **shared_options,
    )
    production, _ = run_native(
        circuit_or_ir,
        mode=mode,
        world_size=world_size,
        return_plan=True,
        distributed_backend_policy=production_policy,
        **shared_options,
    )
    development_signature = _summary_signature(development.summary())
    production_signature = _summary_signature(production.summary())
    if development_signature != production_signature:
        errors.append(f"development and production {mode} signatures differ")
    numerical_error = _state_max_error(development.state(), production.state())
    if numerical_error > float(atol):
        errors.append(
            f"development/production {mode} numerical error {numerical_error:.3e} exceeds atol {float(atol):.3e}"
        )
    return DistributedBackendParityReport(
        mode=mode,
        world_size=world_size,
        passed=not errors,
        errors=tuple(errors),
        development_signature=development_signature,
        production_signature=production_signature,
        numerical_reference_error=numerical_error,
        atol=float(atol),
    )


def require_development_production_parity(
    circuit_or_ir: Any,
    *,
    mode: str = "distributed_statevector",
    world_size: int = 2,
    atol: float = 1e-6,
    **options: Any,
) -> DistributedBackendParityReport:
    """Return a parity report or raise when development and production diverge."""

    report = validate_development_production_parity(
        circuit_or_ir,
        mode=mode,
        world_size=world_size,
        atol=atol,
        **options,
    )
    if not report.passed:
        raise RuntimeError("; ".join(report.errors))
    return report


def _default_preflight_circuit() -> Any:
    from ..circuit import Circuit

    circuit = Circuit(4)
    circuit.h(0).cx(0, 1).rx(2, theta=0.2).rz(3, theta=0.4).cx(2, 3)
    return circuit


def local_distributed_development_preflight(
    circuit_or_ir: Any | None = None,
    *,
    world_size: int = 2,
    modes: Sequence[str] = (
        "distributed_statevector",
        "distributed_mps",
        "distributed_tensor_network",
    ),
    atol: float = 1e-6,
    raise_on_error: bool = False,
    **options: Any,
) -> LocalDistributedPreflightReport:
    """Run a local CPU preflight for development/production distributed parity.

    This is the user-facing development check: it does not require torchrun or
    GPUs, but it verifies that local development execution mirrors production
    shard/task layout for the selected distributed modes.
    """

    program = (
        circuit_or_ir if circuit_or_ir is not None else _default_preflight_circuit()
    )
    development_policy = resolve_distributed_backend_policy(
        profile="development"
    ).summary()
    production_policy = resolve_distributed_backend_policy(
        profile="production"
    ).summary()
    reports: dict[str, DistributedBackendParityReport] = {}
    errors: list[str] = []
    for raw_mode in modes:
        mode = _normalize_mode(raw_mode)
        mode_options = dict(options)
        if mode == "distributed_mps":
            mode_options.setdefault("max_bond", 16)
        if mode == "distributed_tensor_network":
            mode_options.setdefault("max_intermediate_size", 4)
        report = validate_development_production_parity(
            program,
            mode=mode,
            world_size=world_size,
            atol=atol,
            **mode_options,
        )
        reports[mode] = report
        if not report.passed:
            errors.extend(f"{mode}: {error}" for error in report.errors)
    out = LocalDistributedPreflightReport(
        passed=not errors,
        world_size=int(world_size),
        modes=tuple(reports),
        development_policy=development_policy,
        production_policy=production_policy,
        parity_reports=reports,
        errors=tuple(errors),
    )
    if raise_on_error and not out.passed:
        raise RuntimeError("; ".join(out.errors))
    return out


__all__ = [
    "DistributedBackendParityReport",
    "LocalDistributedPreflightReport",
    "local_distributed_development_preflight",
    "require_development_production_parity",
    "validate_development_production_parity",
]
