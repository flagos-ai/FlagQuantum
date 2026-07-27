"""Canonical PyTorch-first training entry point."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TypeAlias

import torch

from .module import ExecutionResult, Module

TrainingObjective: TypeAlias = Callable[[torch.Tensor], torch.Tensor]
TrainingInputs: TypeAlias = torch.Tensor | Callable[[int], torch.Tensor | None] | None
TrainingCallback: TypeAlias = Callable[[int, float, ExecutionResult], None]


@dataclass(frozen=True)
class TrainingResult:
    """Stable summary returned by :func:`flagquantum.train`."""

    losses: tuple[float, ...]
    completed_steps: int
    last_execution: ExecutionResult
    optimizer: str

    def summary(self) -> dict[str, object]:
        return {
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
    """

    if not isinstance(module, Module):
        raise TypeError("fq.train() requires an fq.Module")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("fq.train() optimizer must be a torch.optim.Optimizer")
    if not callable(objective):
        raise TypeError("fq.train() objective must be callable")
    if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
        raise ValueError("fq.train() steps must be a positive integer")
    if log_interval is not None and (
        not isinstance(log_interval, int)
        or isinstance(log_interval, bool)
        or log_interval <= 0
    ):
        raise ValueError("fq.train() log_interval must be a positive integer or None")
    if callback is not None and not callable(callback):
        raise TypeError("fq.train() callback must be callable or None")

    module.train()
    losses: list[float] = []
    last_execution: ExecutionResult | None = None
    for step_index in range(steps):
        step_inputs = inputs(step_index) if callable(inputs) else inputs
        optimizer.zero_grad(set_to_none=True)
        execution = module.execute(step_inputs)
        if execution.value is None:
            raise RuntimeError("fq.train() requires ExecutionResult.value")
        loss = objective(execution.value)
        if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
            raise ValueError("fq.train() objective must return a scalar tensor")
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach())
        losses.append(loss_value)
        last_execution = execution.detach()
        step = step_index + 1
        if callback is not None:
            callback(step, loss_value, last_execution)
        if log_interval is not None and (
            step == 1 or step == steps or step % log_interval == 0
        ):
            print(f"step {step}/{steps} loss={loss_value:.8g}")

    assert last_execution is not None
    return TrainingResult(
        losses=tuple(losses),
        completed_steps=steps,
        last_execution=last_execution,
        optimizer=type(optimizer).__name__,
    )


__all__ = ["TrainingCallback", "TrainingResult", "train"]
