"""Canonical PyTorch-first training entry point."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, TypeAlias

import torch

from ..errors import ValidationError
from .builder_compilation import _builder_tensor_inputs
from .module import ExecutionResult, Module
from .training_compilation import (
    _build_training_circuit,
    _compilation_diagnostics,
    _execute_prebuilt,
    _optimizer_exclusion,
    _prepare_training_program,
    _prepared_execution,
    _scalar_objective,
    _snapshot_statistics,
    _training_exclusion,
    _training_signature,
    _TrainingProgram,
)

TRAINING_RESULT_SUMMARY_SCHEMA = "flagquantum.training_result.summary"
TRAINING_RESULT_SUMMARY_VERSION = "1.0"

TrainingObjective: TypeAlias = Callable[[torch.Tensor], torch.Tensor]
TrainingInputs: TypeAlias = torch.Tensor | Callable[[int], torch.Tensor | None] | None
TrainingCallback: TypeAlias = Callable[[int, float, ExecutionResult], None]

# Promotion requires measured cold/steady-state amortization. Tests can replace
# this gate to exercise the candidate without changing a protected public API.
_MIN_COMPILED_STEPS: int | None = None


def _eager_training_update(
    module: Module,
    optimizer: torch.optim.Optimizer,
    objective: TrainingObjective,
    inputs: torch.Tensor | None,
    reason: str,
    *,
    execution: ExecutionResult | None = None,
) -> tuple[ExecutionResult, torch.Tensor]:
    execution = module.execute(inputs) if execution is None else execution
    loss = _scalar_objective(objective, execution.require_value())
    torch.autograd.backward(loss)
    optimizer.step()
    return (
        _compilation_diagnostics(
            execution,
            {
                "status": "eager",
                "reason": reason,
                "whole_step_fused": False,
            },
        ),
        loss,
    )


def _training_update(
    module: Module,
    optimizer: torch.optim.Optimizer,
    objective: TrainingObjective,
    inputs: torch.Tensor | None,
    remaining_steps: int,
    disabled: str,
    first_step: bool,
) -> tuple[ExecutionResult, torch.Tensor, str]:
    optimizer.zero_grad(set_to_none=True)
    if module._training_program is None and (
        _MIN_COMPILED_STEPS is None or remaining_steps < _MIN_COMPILED_STEPS
    ):
        reason = (
            "compilation_pending_amortization_acceptance"
            if _MIN_COMPILED_STEPS is None
            else "short_training_run"
        )
        execution, loss = _eager_training_update(
            module, optimizer, objective, inputs, reason
        )
        return execution, loss, disabled
    reason = disabled or _training_exclusion(module, optimizer, ())
    if reason:
        module._training_program = None
        execution, loss = _eager_training_update(
            module, optimizer, objective, inputs, reason
        )
        return execution, loss, disabled
    parameters = module._owned_parameters()
    tensors, _, _ = _builder_tensor_inputs(parameters, inputs)
    reason = _training_exclusion(module, optimizer, tensors)
    program = module._training_program
    if program is not None and program.signature != _training_signature(
        module, optimizer, objective, tensors
    ):
        if first_step:
            module._training_program = program = None
        else:
            reason = disabled = "training_signature_changed"
    if not reason and program is None:
        reason = (
            "compilation_pending_amortization_acceptance"
            if _MIN_COMPILED_STEPS is None
            else "short_training_run" if remaining_steps < _MIN_COMPILED_STEPS else ""
        )
    if reason:
        module._training_program = None
        execution, loss = _eager_training_update(
            module, optimizer, objective, inputs, reason
        )
        return execution, loss, disabled
    if program is None:
        from ..compiler.training import compile_training_callable

        program = _TrainingProgram(objective)
        program.gradient_call = compile_training_callable(
            program.evaluate,
            backend="inductor",
            diagnostics=program.diagnostics,
            objective=objective,
        )
        if program.diagnostics["status"] == "eager":
            reason = str(program.diagnostics["reason"])
            execution, loss = _eager_training_update(
                module, optimizer, objective, inputs, reason
            )
            return execution, loss, reason
    try:
        circuit = _build_training_circuit(module, inputs, parameters)
        if program.differentiated is None:
            reason = _training_exclusion(
                module, optimizer, tensors
            ) or _prepare_training_program(program, module, circuit, tensors)
            if reason:
                # Preparation has not updated gradients/optimizer state. Reuse
                # the bound circuit so an impure builder is never invoked twice.
                execution = _execute_prebuilt(module, circuit, inputs, parameters)
                execution, loss = _eager_training_update(
                    module, optimizer, objective, inputs, reason, execution=execution
                )
                return execution, loss, reason
            program.signature = _training_signature(
                module, optimizer, objective, tensors
            )
            optimizer_reason = _optimizer_exclusion(optimizer)
            if not optimizer_reason:
                from ..compiler.training import compile_training_callable

                program.optimizer_call = compile_training_callable(
                    optimizer.step,
                    backend="inductor",
                    diagnostics=program.optimizer_diagnostics,
                )
            else:
                program.optimizer_diagnostics.update(
                    status="eager", reason=optimizer_reason
                )
            module._training_program = program
        execution = _prepared_execution(module, circuit, inputs, parameters)
        assert program.gradient_call is not None
        first = "first_evaluation_seconds" not in program.diagnostics
        started = time.perf_counter() if first else 0.0
        (gradients, (loss, (value, state))), captured = program.gradient_call(
            *(tensor.detach() for tensor in tensors)
        )
        if first:
            program.diagnostics["first_evaluation_seconds"] = (
                time.perf_counter() - started
            )
        execution = replace(execution, value=value, state=state)
        program.diagnostics["last_entry_captured"] = captured
    finally:
        module._builder_bindings.clear()
    selected = tuple(tensors[index] for index in program.gradient_indices)
    torch.autograd.backward(
        selected, tuple(gradient.detach() for gradient in gradients)
    )
    optimizer_reason = _optimizer_exclusion(optimizer)
    if program.optimizer_call is not None and not optimizer_reason:
        program.optimizer_call()
    else:
        optimizer.step()
        program.optimizer_diagnostics.update(status="eager", reason=optimizer_reason)
    diagnostics: dict[str, Any] = {
        "status": "reusable_tensor_regions",
        "whole_step_fused": False,
        "derivative": "torch_func_scalar_reverse_vjp",
        "value_objective_vjp": _snapshot_statistics(program.diagnostics),
        "optimizer": _snapshot_statistics(program.optimizer_diagnostics),
    }
    return _compilation_diagnostics(execution, diagnostics), loss, disabled


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


def _validate_training_request(
    module: Module,
    optimizer: torch.optim.Optimizer,
    objective: TrainingObjective,
    steps: int,
    log_interval: int | None,
    callback: TrainingCallback | None,
) -> None:
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

    _validate_training_request(
        module,
        optimizer,
        objective,
        steps,
        log_interval,
        callback,
    )

    module.train()
    defer_loss_history = callback is None and log_interval is None
    loss_values: list[float] = []
    deferred_losses: list[torch.Tensor] = []
    last_execution: ExecutionResult | None = None
    compilation_disabled = ""
    for step_index in range(steps):
        step_inputs = inputs(step_index) if callable(inputs) else inputs
        execution, loss, compilation_disabled = _training_update(
            module,
            optimizer,
            objective,
            step_inputs,
            steps - step_index,
            compilation_disabled,
            step_index == 0,
        )
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
