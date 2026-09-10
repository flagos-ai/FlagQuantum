"""Composable classical and quantum-aware optimization for hybrid models.

The optimizers in this module update ordinary PyTorch tensors.  "Quantum-aware"
means that an update may use circuit evaluations or the geometry of a quantum
state; it does not imply that the optimizer itself executes on a QPU.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import torch

from ..compute import get_platform_runtime

ParameterGroups = Mapping[str, torch.Tensor]
Objective = Callable[[ParameterGroups], torch.Tensor]
StateFunction = Callable[[ParameterGroups], torch.Tensor]


def _synchronize(parameter: torch.Tensor) -> None:
    if parameter.is_cuda:
        get_platform_runtime(parameter.device.type).synchronize(parameter.device)


@dataclass(frozen=True)
class OptimizationStage:
    """One stage in a hybrid optimization schedule."""

    group: str
    method: str
    steps: int
    lr: float = 0.01
    damping: float = 1e-3
    block_size: int | None = None
    max_iter: int = 20
    history_size: int = 20

    def __post_init__(self) -> None:
        method = self.method.lower()
        if method not in {"adam", "adamw", "sgd", "lbfgs", "qng", "rotosolve"}:
            raise ValueError(f"unsupported optimization method {self.method!r}")
        if not math.isfinite(self.lr) or not math.isfinite(self.damping):
            raise ValueError("learning rate and damping must be finite")
        if any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in (self.steps, self.max_iter, self.history_size)
        ) or (
            self.block_size is not None
            and (
                not isinstance(self.block_size, int)
                or isinstance(self.block_size, bool)
            )
        ):
            raise ValueError(
                "steps, max_iter, history_size, and block_size must be integers"
            )
        if not self.group or self.steps <= 0 or self.lr <= 0:
            raise ValueError("group, steps, and lr must be positive/non-empty")
        if self.damping < 0 or self.max_iter <= 0 or self.history_size <= 0:
            raise ValueError(
                "damping must be non-negative; max_iter/history_size must be positive"
            )
        if self.block_size is not None and self.block_size <= 0:
            raise ValueError("block_size must be positive or None")
        object.__setattr__(self, "method", method)


@dataclass(frozen=True)
class OptimizationRecord:
    stage: int
    step: int
    group: str
    method: str
    loss: float
    gradient_norm: float | None
    update_norm: float
    evaluations: int
    wall_time_seconds: float = 0.0
    objective_gradient_seconds: float = 0.0
    quantum_metric_seconds: float | None = None
    real_jacobian_seconds: float | None = None
    imag_jacobian_seconds: float | None = None
    metric_reduction_seconds: float | None = None
    linear_solve_seconds: float | None = None


@dataclass(frozen=True)
class HybridOptimizationResult:
    parameters: Mapping[str, torch.Tensor]
    history: tuple[float, ...]
    records: tuple[OptimizationRecord, ...]
    evaluations: int

    @property
    def energy(self) -> torch.Tensor:
        return torch.tensor(self.history[-1])


def _checked_scalar(value: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 0:
        raise ValueError("hybrid optimization objective must return a scalar tensor")
    if not bool(torch.isfinite(value)):
        raise FloatingPointError("hybrid optimization objective returned NaN or Inf")
    return value


def _replace(
    groups: dict[str, torch.Tensor], name: str, value: torch.Tensor
) -> dict[str, torch.Tensor]:
    replaced = dict(groups)
    replaced[name] = value
    return replaced


def _quantum_metric(
    state_function: StateFunction,
    groups: dict[str, torch.Tensor],
    group: str,
    *,
    block_size: int | None,
    timings: dict[str, float] | None = None,
) -> torch.Tensor:
    parameter = groups[group]
    shape = parameter.shape

    def state_from_flat(flat: torch.Tensor) -> torch.Tensor:
        state = state_function(_replace(groups, group, flat.reshape(shape)))
        if not isinstance(state, torch.Tensor) or not torch.is_complex(state):
            raise ValueError("QNG state_function must return a complex state tensor")
        state = state.reshape(-1)
        norm = torch.linalg.vector_norm(state)
        if not bool(torch.isfinite(norm)) or not bool(norm > 0):
            raise ValueError("QNG state_function must return a finite nonzero state")
        return state.div(norm)

    flat = parameter.reshape(-1)
    started = time.perf_counter()
    real_jac = torch.autograd.functional.jacobian(
        lambda value: state_from_flat(value).real,
        flat,
        create_graph=False,
        vectorize=True,
    ).reshape(-1, flat.numel())
    _synchronize(parameter)
    real_finished = time.perf_counter()
    imag_jac = torch.autograd.functional.jacobian(
        lambda value: state_from_flat(value).imag,
        flat,
        create_graph=False,
        vectorize=True,
    ).reshape(-1, flat.numel())
    _synchronize(parameter)
    imag_finished = time.perf_counter()
    jacobian = torch.complex(real_jac, imag_jac)
    state = state_from_flat(flat).detach()
    connection = torch.conj(state) @ jacobian
    metric = torch.real(
        torch.conj(jacobian).transpose(0, 1) @ jacobian
        - torch.outer(torch.conj(connection), connection)
    )
    metric = 0.5 * (metric + metric.transpose(0, 1))
    if block_size is not None and block_size < flat.numel():
        blocked = torch.zeros_like(metric)
        for start in range(0, flat.numel(), block_size):
            stop = min(flat.numel(), start + block_size)
            blocked[start:stop, start:stop] = metric[start:stop, start:stop]
        metric = blocked
    _synchronize(parameter)
    finished = time.perf_counter()
    if timings is not None:
        timings.update(
            {
                "real_jacobian_seconds": real_finished - started,
                "imag_jacobian_seconds": imag_finished - real_finished,
                "metric_reduction_seconds": finished - imag_finished,
                "quantum_metric_seconds": finished - started,
            }
        )
    return metric


def _torch_optimizer(
    method: str, parameter: torch.Tensor, stage: OptimizationStage
) -> torch.optim.Optimizer:
    if method == "adam":
        return torch.optim.Adam([parameter], lr=stage.lr)
    if method == "adamw":
        return torch.optim.AdamW([parameter], lr=stage.lr)
    if method == "sgd":
        return torch.optim.SGD([parameter], lr=stage.lr)
    return torch.optim.LBFGS(
        [parameter],
        lr=stage.lr,
        max_iter=stage.max_iter,
        history_size=stage.history_size,
        tolerance_grad=1e-10,
        tolerance_change=1e-12,
    )


def optimize_hybrid(
    objective: Objective,
    parameter_groups: ParameterGroups,
    stages: Sequence[OptimizationStage],
    *,
    state_function: StateFunction | None = None,
) -> HybridOptimizationResult:
    """Optimize named parameter groups with a staged classical/quantum schedule.

    QNG is currently an exact local-state method and therefore requires
    ``state_function``.  Rotosolve assumes each selected coordinate enters the
    circuit as a single-frequency Pauli rotation.  Neither method silently
    claims sharded execution semantics.
    """

    if not callable(objective) or not parameter_groups or not stages:
        raise ValueError("objective, parameter_groups, and stages are required")
    groups = {
        name: value.detach().clone().requires_grad_(True)
        for name, value in parameter_groups.items()
    }
    records: list[OptimizationRecord] = []
    history: list[float] = []
    evaluations = 0

    for stage_index, stage in enumerate(stages):
        if stage.group not in groups:
            raise KeyError(f"unknown parameter group {stage.group!r}")
        parameter = groups[stage.group]
        optimizer = (
            _torch_optimizer(stage.method, parameter, stage)
            if stage.method in {"adam", "adamw", "sgd", "lbfgs"}
            else None
        )
        for step_index in range(stage.steps):
            _synchronize(parameter)
            step_started = time.perf_counter()
            objective_gradient_seconds = 0.0
            quantum_metric_seconds: float | None = None
            real_jacobian_seconds: float | None = None
            imag_jacobian_seconds: float | None = None
            metric_reduction_seconds: float | None = None
            linear_solve_seconds: float | None = None
            before = parameter.detach().clone()
            gradient_norm: float | None = None
            step_evaluations = 0
            if isinstance(optimizer, torch.optim.LBFGS):

                def closure() -> torch.Tensor:
                    nonlocal evaluations, step_evaluations
                    optimizer.zero_grad(set_to_none=True)
                    loss = _checked_scalar(objective(groups))
                    loss.backward()
                    evaluations += 1
                    step_evaluations += 1
                    return loss

                optimizer.step(closure)
                loss = _checked_scalar(objective(groups))
                evaluations += 1
                step_evaluations += 1
            elif stage.method in {"adam", "adamw", "sgd"}:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                loss = _checked_scalar(objective(groups))
                evaluations += 1
                step_evaluations += 1
                loss.backward()
                gradient_norm = float(torch.linalg.vector_norm(parameter.grad).detach())
                optimizer.step()
                loss = _checked_scalar(objective(groups))
                evaluations += 1
                step_evaluations += 1
            elif stage.method == "qng":
                if state_function is None:
                    raise ValueError("QNG requires state_function")
                objective_started = time.perf_counter()
                loss = _checked_scalar(objective(groups))
                evaluations += 1
                step_evaluations += 1
                gradient = torch.autograd.grad(loss, parameter)[0].reshape(-1)
                _synchronize(parameter)
                objective_gradient_seconds = time.perf_counter() - objective_started
                gradient_norm = float(torch.linalg.vector_norm(gradient).detach())
                metric_timings: dict[str, float] = {}
                metric = _quantum_metric(
                    state_function,
                    groups,
                    stage.group,
                    block_size=stage.block_size,
                    timings=metric_timings,
                )
                quantum_metric_seconds = metric_timings["quantum_metric_seconds"]
                real_jacobian_seconds = metric_timings["real_jacobian_seconds"]
                imag_jacobian_seconds = metric_timings["imag_jacobian_seconds"]
                metric_reduction_seconds = metric_timings["metric_reduction_seconds"]
                identity = torch.eye(
                    metric.shape[0], dtype=metric.dtype, device=metric.device
                )
                solve_started = time.perf_counter()
                direction = torch.linalg.solve(
                    metric + stage.damping * identity, gradient
                )
                _synchronize(parameter)
                linear_solve_seconds = time.perf_counter() - solve_started
                with torch.no_grad():
                    parameter.add_(-stage.lr * direction.reshape_as(parameter))
                loss = _checked_scalar(objective(groups))
                evaluations += 1
                step_evaluations += 1
            else:
                # Three evaluations per coordinate recover the sinusoid around
                # the current angle; the already-known current value is reused.
                loss = _checked_scalar(objective(groups))
                evaluations += 1
                step_evaluations += 1
                flat = parameter.reshape(-1)
                for index in range(flat.numel()):
                    center = flat[index].detach().clone()
                    current = float(loss.detach())
                    with torch.no_grad():
                        flat[index] = center + math.pi / 2
                    plus = _checked_scalar(objective(groups))
                    with torch.no_grad():
                        flat[index] = center - math.pi / 2
                    minus = _checked_scalar(objective(groups))
                    evaluations += 2
                    step_evaluations += 2
                    constant = 0.5 * (float(plus.detach()) + float(minus.detach()))
                    cosine = current - constant
                    sine = 0.5 * (float(plus.detach()) - float(minus.detach()))
                    delta = math.atan2(sine, cosine) + math.pi
                    with torch.no_grad():
                        flat[index] = center + delta
                    loss = _checked_scalar(objective(groups))
                    evaluations += 1
                    step_evaluations += 1

            loss_value = float(loss.detach())
            history.append(loss_value)
            records.append(
                OptimizationRecord(
                    stage=stage_index,
                    step=step_index,
                    group=stage.group,
                    method=stage.method,
                    loss=loss_value,
                    gradient_norm=gradient_norm,
                    update_norm=float(
                        torch.linalg.vector_norm(parameter.detach() - before)
                    ),
                    evaluations=step_evaluations,
                    wall_time_seconds=time.perf_counter() - step_started,
                    objective_gradient_seconds=objective_gradient_seconds,
                    quantum_metric_seconds=quantum_metric_seconds,
                    real_jacobian_seconds=real_jacobian_seconds,
                    imag_jacobian_seconds=imag_jacobian_seconds,
                    metric_reduction_seconds=metric_reduction_seconds,
                    linear_solve_seconds=linear_solve_seconds,
                )
            )

    return HybridOptimizationResult(
        parameters={name: value.detach().clone() for name, value in groups.items()},
        history=tuple(history),
        records=tuple(records),
        evaluations=evaluations,
    )


__all__ = [
    "HybridOptimizationResult",
    "OptimizationRecord",
    "OptimizationStage",
    "optimize_hybrid",
]
