"""Registry-based dispatch for noisy execution backends."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from threading import RLock
from typing import Any

import torch

from ..compilation.models import (
    EvolutionSemantics,
    NoisyExecutionPlan,
    StateRepresentation,
)
from ..compiler import lower_noise_model
from ..core.ir import CircuitIR
from ..noise import NoiseModel

NoiseExecutor = Callable[
    [CircuitIR, NoisyExecutionPlan, Mapping[str, Any]],
    Any,
]

_EXECUTORS: dict[tuple[str, str], NoiseExecutor] = {}
_LOCK = RLock()
_BUILTINS_READY = False


def _execute_density_plan(
    ir: CircuitIR,
    plan: NoisyExecutionPlan,
    options: Mapping[str, Any],
) -> torch.Tensor:
    if plan.representation != "density_matrix" or plan.evolution != "exact_channel":
        raise ValueError("density executor requires density_matrix exact_channel plan")
    supported = {"bsz", "device", "dtype"}
    density_options = {key: value for key, value in options.items() if key in supported}

    from ..simulation.density_matrix import density_matrix_from_ir

    return density_matrix_from_ir(ir, **density_options)


def noisy_density_matrix(
    circuit_or_ir: Any,
    noise_model: NoiseModel | None = None,
    *,
    bsz: int = 1,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Lower optional noise and run the local exact density simulation."""

    from ..simulation.density_matrix import density_matrix_from_ir

    lowered = lower_noise_model(circuit_or_ir, noise_model)
    return density_matrix_from_ir(lowered, bsz=bsz, device=device, dtype=dtype)


def register_noise_executor(
    *,
    representation: StateRepresentation,
    evolution: EvolutionSemantics,
    executor: NoiseExecutor,
    replace: bool = False,
) -> None:
    """Register one executor for a representation/evolution pair."""

    key = (representation, evolution)
    with _LOCK:
        if key in _EXECUTORS and not replace:
            raise ValueError(f"noise executor already registered for {key!r}")
        _EXECUTORS[key] = executor


def _ensure_builtin_executors() -> None:
    global _BUILTINS_READY
    if _BUILTINS_READY:
        return
    with _LOCK:
        if _BUILTINS_READY:
            return
        key = ("density_matrix", "exact_channel")
        if key not in _EXECUTORS:
            _EXECUTORS[key] = _execute_density_plan
        _BUILTINS_READY = True


def resolve_noise_executor(plan: NoisyExecutionPlan) -> NoiseExecutor:
    """Resolve the executor selected by a structured noisy plan."""

    _ensure_builtin_executors()
    key = (plan.representation, plan.evolution)
    try:
        return _EXECUTORS[key]
    except KeyError as exc:
        raise LookupError(f"no noise executor registered for {key!r}") from exc


def execute_noisy_plan(
    ir: CircuitIR,
    plan: NoisyExecutionPlan,
    *,
    options: Mapping[str, Any] | None = None,
) -> Any:
    """Execute compiled channel IR through the selected registered backend."""

    executor = resolve_noise_executor(plan)
    return executor(ir, plan, options or {})


__all__ = (
    "NoiseExecutor",
    "execute_noisy_plan",
    "noisy_density_matrix",
    "register_noise_executor",
    "resolve_noise_executor",
)
