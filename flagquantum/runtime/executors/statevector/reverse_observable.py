"""Observable-boundary helpers for the statevector reverse sweep."""

from __future__ import annotations

import torch

from ....simulation.native_cpu import fused_observable_adjoint_seed


def seed_adjoint(amplitudes: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Seed the CPU adjoint natively when supported, with an eager fallback."""

    seed = fused_observable_adjoint_seed(amplitudes, weights)
    return seed if seed is not None else 2 * amplitudes * weights.reshape(1, -1)


__all__ = ("seed_adjoint",)
