"""Observable-boundary helpers for the statevector reverse sweep."""

from __future__ import annotations

from typing import Any

import torch

from ....simulation.native_cpu import (
    fused_observable_adjoint_seed,
    native_cpu_observable_rotation_boundary_available,
)
from ....simulation.statevector.adjoint import z_hamiltonian_chunk
from .forward import _storage_global_indices
from .reverse_adjoint_kernels import _cpu_direct_adjoint_gate_enabled
from .reverse_support import _reverse_chunk_amplitudes


def seed_adjoint(amplitudes: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Seed the CPU adjoint natively when supported, with an eager fallback."""

    seed = fused_observable_adjoint_seed(amplitudes, weights)
    return seed if seed is not None else 2 * amplitudes * weights.reshape(1, -1)


def prepare_observable_adjoint(
    sweep: Any, final_state: Any, weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Build the seed now or defer it to a supported first rotation tile."""

    defer_seed = bool(
        sweep.inplace_local
        and sweep.policy.strategy == "reversible_adjoint"
        and sweep.world_size == 1
        and final_state.amplitudes.device.type == "cpu"
        and sweep.bound.instructions
        and sweep.bound.instructions[-1].name in {"rx", "ry", "rz"}
        and _cpu_direct_adjoint_gate_enabled()
        and native_cpu_observable_rotation_boundary_available()
    )
    if defer_seed:
        return torch.empty_like(final_state.amplitudes), weights
    return seed_adjoint(final_state.amplitudes, weights), None


def materialize_pending_observable_adjoint(sweep: Any) -> None:
    """Build a deferred observable seed before a non-fused reverse path."""

    if sweep.pending_observable_weights is None:
        return
    assert sweep.reversible_state is not None
    sweep.adjoint = seed_adjoint(
        sweep.reversible_state.amplitudes, sweep.pending_observable_weights
    )
    sweep.pending_observable_weights = None


def hamiltonian_adjoint_seed(
    shard_state: Any,
    *,
    plan: Any,
    n_wires: int,
    terms: tuple[tuple[float, tuple[int, ...]], ...],
) -> torch.Tensor:
    """Construct one adjoint seed for a weighted Z/ZZ Hamiltonian."""

    adjoint = torch.zeros_like(shard_state.amplitudes)
    local_count = shard_state.shard.local_amplitudes
    with torch.no_grad():
        for start in range(0, local_count, _reverse_chunk_amplitudes()):
            end = min(local_count, start + _reverse_chunk_amplitudes())
            indices = _storage_global_indices(shard_state, start, end, plan=plan)
            _, chunk = z_hamiltonian_chunk(
                shard_state.amplitudes[:, start:end],
                indices,
                n_wires=n_wires,
                terms=terms,
            )
            adjoint[:, start:end].copy_(chunk)
    return adjoint


__all__ = (
    "hamiltonian_adjoint_seed",
    "materialize_pending_observable_adjoint",
    "prepare_observable_adjoint",
    "seed_adjoint",
)
