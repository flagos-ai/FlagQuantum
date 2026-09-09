"""Canonical PyTorch-first training entry point."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TypeAlias

import torch

from ..errors import ValidationError
from .module import ExecutionResult, Module

TRAINING_RESULT_SUMMARY_SCHEMA = "flagquantum.training_result.summary"
TRAINING_RESULT_SUMMARY_VERSION = "1.0"

TrainingObjective: TypeAlias = Callable[[torch.Tensor], torch.Tensor]
TrainingInputs: TypeAlias = torch.Tensor | Callable[[int], torch.Tensor | None] | None
TrainingCallback: TypeAlias = Callable[[int, float, ExecutionResult], None]


def _materialize_losses(losses: list[torch.Tensor]) -> tuple[float, ...]:
    """Transfer a device-side loss history to Python once after training."""

    first_device = losses[0].device
    if all(loss.device == first_device for loss in losses):
        return tuple(torch.stack(losses).cpu().tolist())
    return tuple(float(loss.cpu()) for loss in losses)


@dataclass(frozen=True)
class TrainingResult:
    """Stable summary returned by :func:`flagquantum.train`."""

    losses: tuple[float, ...]
    completed_steps: int
    last_execution: ExecutionResult
    optimizer: str

    def __post_init__(self) -> None:
        if self.completed_steps <= 0:
            raise ValidationError("completed_steps must be positive")
        if len(self.losses) != self.completed_steps:
            raise ValidationError("loss count must equal completed_steps")
        if not self.optimizer:
            raise ValidationError("optimizer must be non-empty")

    @property
    def final_loss(self) -> float:
        return self.losses[-1]

    def summary(self) -> dict[str, object]:
        return {
            "schema": TRAINING_RESULT_SUMMARY_SCHEMA,
            "version": TRAINING_RESULT_SUMMARY_VERSION,
            "losses": self.losses,
            "completed_steps": self.completed_steps,
            "optimizer": self.optimizer,
            "last_execution": self.last_execution.summary(),
        }


def train(
    module: Module,
    *,
    optimizer: torch.optim.Optimizer,
    objective: TrainingObjective,
    steps: int,
    inputs: TrainingInputs = None,
    log_interval: int | None = None,
    callback: TrainingCallback | None = None,
) -> TrainingResult:
    """Optimize an ``fq.Module`` and return a stable training summary.

    ``objective`` receives the tensor in ``ExecutionResult.value`` and must
    return a scalar loss. Pass a callable as ``inputs`` to provide one batch per
    step. Set ``log_interval`` to a positive integer to print the first, last,
    and every matching training step. ``callback`` is called after every step
    with a one-based step number, scalar loss, and detached execution result.

    Examples:
        Optimize a one-parameter circuit with a standard PyTorch optimizer:

        >>> import flagquantum as fq
        >>> import torch
        >>> def build(parameters):
        ...     return fq.Circuit(1).ry(0, theta=parameters[0])
        >>> module = fq.Module(build, n_parameters=1, init=torch.tensor([0.25]))
        >>> optimizer = torch.optim.SGD(module.parameters(), lr=0.2)
        >>> result = fq.train(
        ...     module,
        ...     optimizer=optimizer,
        ...     objective=lambda value: value.mean(),
        ...     steps=2,
        ... )
        >>> result.completed_steps
        2
    """

    if not isinstance(module, Module):
        raise TypeError("fq.train() requires an fq.Module")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("fq.train() optimizer must be a torch.optim.Optimizer")
    if not callable(objective):
        raise TypeError("fq.train() objective must be callable")
    if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
        raise ValidationError("fq.train() steps must be a positive integer")
    if log_interval is not None and (
        not isinstance(log_interval, int)
        or isinstance(log_interval, bool)
        or log_interval <= 0
    ):
        raise ValidationError(
            "fq.train() log_interval must be a positive integer or None"
        )
    if callback is not None and not callable(callback):
        raise TypeError("fq.train() callback must be callable or None")

    module.train()
    defer_loss_history = callback is None and log_interval is None
    loss_values: list[float] = []
    deferred_losses: list[torch.Tensor] = []
    last_execution: ExecutionResult | None = None
    for step_index in range(steps):
        step_inputs = inputs(step_index) if callable(inputs) else inputs
        optimizer.zero_grad(set_to_none=True)
        execution = module.execute(step_inputs)
        loss = objective(execution.require_value())
        if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
            raise ValidationError("fq.train() objective must return a scalar tensor")
        loss.backward()
        optimizer.step()
        detached_loss = loss.detach()
        if defer_loss_history:
            shares_mutable_storage = loss.is_leaf or loss._base is not None
            deferred_losses.append(
                detached_loss.clone() if shares_mutable_storage else detached_loss
            )
            loss_value = None
        else:
            loss_value = float(detached_loss)
            loss_values.append(loss_value)
        step = step_index + 1
        should_log = log_interval is not None and (
            step == 1 or step == steps or step % log_interval == 0
        )
        if callback is not None:
            assert loss_value is not None
            callback_execution = execution.detach()
            callback(step, loss_value, callback_execution)
            if step == steps:
                last_execution = callback_execution
        elif step == steps:
            last_execution = execution.detach()
        if should_log:
            assert loss_value is not None
            print(f"step {step}/{steps} loss={loss_value:.8g}")

    assert last_execution is not None
    losses = (
        _materialize_losses(deferred_losses)
        if defer_loss_history
        else tuple(loss_values)
    )
    return TrainingResult(
        losses=losses,
        completed_steps=steps,
        last_execution=last_execution,
        optimizer=type(optimizer).__name__,
    )


__all__ = [
    "TRAINING_RESULT_SUMMARY_SCHEMA",
    "TRAINING_RESULT_SUMMARY_VERSION",
    "TrainingCallback",
    "TrainingResult",
    "train",
]
