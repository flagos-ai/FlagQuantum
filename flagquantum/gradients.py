"""Gradient utilities for trainable FlagQuantum circuits."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import torch

from .core.ir import ensure_circuit_ir

CircuitBuilder = Callable[[torch.Tensor], Any]
LossFunction = Callable[[Any], torch.Tensor]
BatchLossFunction = Callable[[tuple[Any, ...]], Sequence[torch.Tensor] | torch.Tensor]

_BATCH_PARAMETER_SHIFT_GATES = frozenset({"h", "x", "rx", "ry", "rz", "cx"})
_PARAMETER_SHIFT_GATES = frozenset({"rx", "ry", "rz"})


def parameter_shift_gradient(
    circuit_builder: CircuitBuilder,
    parameters: torch.Tensor,
    loss_fn: LossFunction,
    *,
    shift: float = torch.pi / 2,
) -> torch.Tensor:
    """Estimate circuit-parameter gradients with the parameter-shift rule.

    ``circuit_builder`` receives a tensor of parameter values and must return a
    circuit or executable object accepted by ``loss_fn``. ``loss_fn`` must return
    a scalar tensor. The method is backend-agnostic and works for native
    simulators, distributed modes, shot-based cloud execution wrappers, and
    hardware-style inference paths where autograd is not available.
    """

    base = parameters.detach()
    flat = base.reshape(-1)
    grads = []
    for index in range(flat.numel()):
        plus = flat.clone()
        minus = flat.clone()
        plus[index] = plus[index] + shift
        minus[index] = minus[index] - shift
        plus_loss = loss_fn(circuit_builder(plus.reshape_as(base)))
        minus_loss = loss_fn(circuit_builder(minus.reshape_as(base)))
        grads.append(0.5 * (plus_loss - minus_loss))
    return (
        torch.stack(grads)
        .reshape_as(base)
        .to(device=parameters.device, dtype=parameters.dtype)
    )


def batched_parameter_shift_gradient(
    circuit_builder: CircuitBuilder,
    parameters: torch.Tensor,
    batch_loss_fn: BatchLossFunction,
    *,
    shift: float = torch.pi / 2,
) -> torch.Tensor:
    """Evaluate a parameter-shift gradient through one batch request.

    This first remote-safe profile accepts H, X, RX, RY, RZ, and CX circuits.
    Each input parameter must control exactly one RX, RY, or RZ occurrence.
    ``batch_loss_fn`` receives all positive and negative shifts in parameter
    order and must return one scalar tensor per circuit.

    Args:
        circuit_builder: Builds a circuit from a parameter tensor.
        parameters: Real, non-empty parameter tensor.
        batch_loss_fn: Evaluates every shifted circuit in one batch operation.
        shift: Parameter displacement. The current profile requires pi/2.

    Returns:
        A detached gradient with the same shape, dtype, and device as
        ``parameters``.

    Raises:
        TypeError: If inputs or returned losses have unsupported types.
        ValueError: If the circuit is outside the supported shift profile.
    """

    base = _validated_parameters(parameters, shift=shift)
    flat = base.reshape(-1)
    base_structure, base_values = _circuit_profile(circuit_builder(base))

    programs: list[Any] = []
    for index in range(flat.numel()):
        plus = flat.clone()
        minus = flat.clone()
        plus[index] += shift
        minus[index] -= shift
        plus_program = circuit_builder(plus.reshape_as(base))
        minus_program = circuit_builder(minus.reshape_as(base))
        _validate_shift_pair(
            (base_structure, base_values),
            _circuit_profile(plus_program),
            _circuit_profile(minus_program),
            shift=shift,
            parameter_index=index,
        )
        programs.extend((plus_program, minus_program))

    losses = _batch_losses(batch_loss_fn(tuple(programs)), expected=len(programs))
    gradient = 0.5 * (losses[0::2] - losses[1::2])
    return gradient.reshape_as(base).to(
        device=parameters.device,
        dtype=parameters.dtype,
    )


def _validated_parameters(parameters: torch.Tensor, *, shift: float) -> torch.Tensor:
    if not isinstance(parameters, torch.Tensor):
        raise TypeError("parameters must be a torch.Tensor")
    if parameters.numel() == 0:
        raise ValueError("parameters must not be empty")
    if not (parameters.is_floating_point() and torch.isfinite(parameters).all()):
        raise ValueError("parameters must contain finite real floating-point values")
    if not math.isclose(float(shift), math.pi / 2, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("batched parameter shift currently requires shift=pi/2")
    return parameters.detach()


def _circuit_profile(program: Any) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    ir = ensure_circuit_ir(program)
    unsupported = sorted(
        {item.name for item in ir.instructions} - _BATCH_PARAMETER_SHIFT_GATES
    )
    if unsupported:
        raise ValueError(
            "batched parameter shift supports only H, X, RX, RY, RZ, and CX; "
            f"unsupported gates: {', '.join(unsupported)}"
        )
    if any(item.matrix is not None for item in ir.instructions):
        raise ValueError("batched parameter shift does not support custom matrices")
    structure = (
        ir.n_wires,
        tuple((item.name, item.wires, tuple(item.params)) for item in ir.instructions),
    )
    values = tuple(
        (item.name, name, _scalar_parameter(value))
        for item in ir.instructions
        for name, value in item.params.items()
    )
    return structure, values


def _validate_shift_pair(
    base: tuple[tuple[Any, ...], tuple[Any, ...]],
    plus: tuple[tuple[Any, ...], tuple[Any, ...]],
    minus: tuple[tuple[Any, ...], tuple[Any, ...]],
    *,
    shift: float,
    parameter_index: int,
) -> None:
    base_structure, base_values = base
    plus_structure, plus_values = plus
    minus_structure, minus_values = minus
    if not base_structure == plus_structure == minus_structure:
        raise ValueError(
            f"parameter {parameter_index} changes circuit structure; "
            "batched parameter shift requires a static circuit"
        )
    changed = []
    for base_item, plus_item, minus_item in zip(
        base_values, plus_values, minus_values, strict=True
    ):
        opcode, name, base_value = base_item
        _, _, plus_value = plus_item
        _, _, minus_value = minus_item
        if not (
            math.isclose(plus_value, base_value, rel_tol=0.0, abs_tol=1e-7)
            and math.isclose(minus_value, base_value, rel_tol=0.0, abs_tol=1e-7)
        ):
            changed.append((opcode, name, base_value, plus_value, minus_value))

    if len(changed) != 1:
        raise ValueError(
            f"parameter {parameter_index} must control exactly one gate occurrence; "
            f"found {len(changed)}"
        )
    opcode, name, base_value, plus_value, minus_value = changed[0]
    if opcode not in _PARAMETER_SHIFT_GATES or name != "theta":
        raise ValueError(
            f"parameter {parameter_index} must control one RX, RY, or RZ angle"
        )
    if not (
        math.isclose(plus_value - base_value, shift, rel_tol=0.0, abs_tol=1e-6)
        and math.isclose(minus_value - base_value, -shift, rel_tol=0.0, abs_tol=1e-6)
    ):
        raise ValueError(
            f"parameter {parameter_index} must enter its gate angle directly"
        )


def _scalar_parameter(value: Any) -> float:
    tensor = (
        value.detach() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    )
    if tensor.numel() != 1 or tensor.is_complex():
        raise ValueError("batched parameter shift requires scalar real gate parameters")
    scalar = float(tensor.cpu())
    if not math.isfinite(scalar):
        raise ValueError("batched parameter shift requires finite gate parameters")
    return scalar


def _batch_losses(
    values: Sequence[torch.Tensor] | torch.Tensor,
    *,
    expected: int,
) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        if values.numel() != expected:
            raise ValueError(
                f"batch_loss_fn returned {values.numel()} losses; expected {expected}"
            )
        return values.reshape(expected).detach()
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError("batch_loss_fn must return a tensor or sequence of tensors")
    if len(values) != expected:
        raise ValueError(
            f"batch_loss_fn returned {len(values)} losses; expected {expected}"
        )
    losses = []
    for value in values:
        if not isinstance(value, torch.Tensor) or value.numel() != 1:
            raise TypeError("batch_loss_fn must return one scalar tensor per circuit")
        losses.append(value.detach().reshape(()))
    return torch.stack(losses)


__all__ = ["batched_parameter_shift_gradient", "parameter_shift_gradient"]
