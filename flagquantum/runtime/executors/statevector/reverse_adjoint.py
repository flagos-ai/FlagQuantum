"""Native PyTorch reverse mode over rank-local statevector shards."""

from __future__ import annotations

from typing import Any

import torch

from ....core.ir import CircuitIR
from ....simulation.statevector.adjoint import (
    z_expectation_adjoint_chunk as _z_expectation_adjoint_chunk,
)
from ....simulation.statevector.adjoint import (
    z_expectation_chunk as _z_expectation_chunk,
)
from .checkpointing import StatevectorCheckpointPolicy
from .forward import _storage_global_indices
from .reverse_adjoint_sweep import _ReversibleAdjointSweep
from .reverse_support import BackwardExecutionEvidence, _reverse_chunk_amplitudes


def _local_expectation_z(
    shard_state: Any, *, plan: Any, n_wires: int, wire: int
) -> torch.Tensor:
    total = torch.zeros(
        (),
        dtype=shard_state.amplitudes.real.dtype,
        device=shard_state.amplitudes.device,
    )
    local_count = shard_state.shard.local_amplitudes
    for start in range(0, local_count, _reverse_chunk_amplitudes()):
        end = min(local_count, start + _reverse_chunk_amplitudes())
        indices = _storage_global_indices(shard_state, start, end, plan=plan)
        total = total + _z_expectation_chunk(
            shard_state.amplitudes[:, start:end],
            indices,
            n_wires=n_wires,
            wire=wire,
        )
    return total


def _local_expectation_z_adjoint(
    shard_state: Any, *, plan: Any, n_wires: int, wire: int
) -> torch.Tensor:
    """Construct d<Z>/d(state*) directly without an amplitude-sized graph."""

    adjoint = torch.empty_like(shard_state.amplitudes)
    local_count = shard_state.shard.local_amplitudes
    with torch.no_grad():
        for start in range(0, local_count, _reverse_chunk_amplitudes()):
            end = min(local_count, start + _reverse_chunk_amplitudes())
            indices = _storage_global_indices(shard_state, start, end, plan=plan)
            adjoint[:, start:end] = _z_expectation_adjoint_chunk(
                shard_state.amplitudes[:, start:end],
                indices,
                n_wires=n_wires,
                wire=wire,
            )
    return adjoint


def _local_expectation_z_sum(
    shard_state: Any, *, plan: Any, n_wires: int, wires: tuple[int, ...]
) -> torch.Tensor:
    """Evaluate a sum of single-wire Z terms from one final shard state."""

    total = torch.zeros(
        (),
        dtype=shard_state.amplitudes.real.dtype,
        device=shard_state.amplitudes.device,
    )
    for wire in wires:
        total = total + _local_expectation_z(
            shard_state,
            plan=plan,
            n_wires=n_wires,
            wire=wire,
        )
    return total


def _explicit_sharded_adjoint(
    ir: CircuitIR,
    slots: tuple[tuple[int, str, int], ...],
    saved_parameters: tuple[torch.Tensor, ...],
    *,
    observable_wires: tuple[int, ...],
    device: torch.device,
    policy: StatevectorCheckpointPolicy,
    process_group: Any | None = None,
    evidence: BackwardExecutionEvidence,
    saved_final_state: Any | None = None,
    saved_inter_node_ket_checkpoints: tuple[tuple[int, int, torch.Tensor], ...] = (),
) -> tuple[torch.Tensor, ...]:
    """Rematerialize rank-local forward states and propagate gate adjoints."""

    sweep = _ReversibleAdjointSweep(
        ir,
        slots,
        saved_parameters,
        observable_wires=observable_wires,
        device=device,
        policy=policy,
        process_group=process_group,
        evidence=evidence,
        saved_final_state=saved_final_state,
        saved_inter_node_ket_checkpoints=saved_inter_node_ket_checkpoints,
    )
    return sweep.run()
