"""Gradient utilities for trainable FlagQuantum circuits."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

CircuitBuilder = Callable[[torch.Tensor], Any]
LossFunction = Callable[[Any], torch.Tensor]


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


__all__ = ["parameter_shift_gradient"]
