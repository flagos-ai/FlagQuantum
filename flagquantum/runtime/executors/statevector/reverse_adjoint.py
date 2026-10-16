"""Native PyTorch reverse mode over rank-local statevector shards."""

from __future__ import annotations

import os
from collections import OrderedDict
from collections.abc import Sequence
from threading import Lock
from typing import Any

import torch

from ....core.ir import CircuitIR
from ....simulation.native_cpu import fused_observable_expectation
from ....simulation.statevector.adjoint import (
    z_expectation_adjoint_chunk as _z_expectation_adjoint_chunk,
)
from ....simulation.statevector.adjoint import (
    z_expectation_chunk as _z_expectation_chunk,
)
from ....simulation.statevector.adjoint import (
    z_hamiltonian_chunk,
    z_hamiltonian_weights,
)
from .checkpointing import StatevectorCheckpointPolicy
from .forward import _storage_global_indices
from .reverse_adjoint_sweep import _ReversibleAdjointSweep
from .reverse_support import BackwardExecutionEvidence, _reverse_chunk_amplitudes

_OBSERVABLE_WEIGHT_CACHE_MAX_BYTES = 256 * 1024 * 1024
_OBSERVABLE_WEIGHT_CACHE_MAX_ENTRIES = 8
_ObservableWeightKey = tuple[
    int,
    tuple[tuple[float, tuple[int, ...]], ...],
    torch.dtype,
    str,
    int,
]
_OBSERVABLE_WEIGHT_CACHE: OrderedDict[_ObservableWeightKey, torch.Tensor] = (
    OrderedDict()
)
_OBSERVABLE_WEIGHT_CACHE_LOCK = Lock()


def _observable_weight_cache_enabled() -> bool:
    return os.getenv(
        "FQ_STATEVECTOR_ADJOINT_OBSERVABLE_CACHE", "1"
    ).strip().lower() not in {"0", "false", "off", "no"}


def _observable_weight_cache_key(
    shard_state: Any,
    *,
    plan: Any,
    n_wires: int,
    terms: tuple[tuple[float, tuple[int, ...]], ...],
) -> _ObservableWeightKey | None:
    """Identify safe, canonical single-process CPU observable diagonals."""

    if (
        not _observable_weight_cache_enabled()
        or plan.world_size != 1
        or shard_state.rank != 0
        or shard_state.shard.amplitude_start != 0
        or shard_state.amplitudes.device.type != "cpu"
    ):
        return None
    return (
        n_wires,
        terms,
        shard_state.amplitudes.real.dtype,
        str(shard_state.amplitudes.device),
        shard_state.shard.local_amplitudes,
    )


def _cached_observable_weights(key: _ObservableWeightKey) -> torch.Tensor | None:
    with _OBSERVABLE_WEIGHT_CACHE_LOCK:
        weights = _OBSERVABLE_WEIGHT_CACHE.get(key)
        if weights is not None:
            _OBSERVABLE_WEIGHT_CACHE.move_to_end(key)
        return weights


def _cache_observable_weights(key: _ObservableWeightKey, weights: torch.Tensor) -> None:
    if weights.numel() * weights.element_size() > _OBSERVABLE_WEIGHT_CACHE_MAX_BYTES:
        return
    with _OBSERVABLE_WEIGHT_CACHE_LOCK:
        _OBSERVABLE_WEIGHT_CACHE[key] = weights
        _OBSERVABLE_WEIGHT_CACHE.move_to_end(key)
        while _OBSERVABLE_WEIGHT_CACHE and (
            len(_OBSERVABLE_WEIGHT_CACHE) > _OBSERVABLE_WEIGHT_CACHE_MAX_ENTRIES
            or sum(
                item.numel() * item.element_size()
                for item in _OBSERVABLE_WEIGHT_CACHE.values()
            )
            > _OBSERVABLE_WEIGHT_CACHE_MAX_BYTES
        ):
            _OBSERVABLE_WEIGHT_CACHE.popitem(last=False)


def _clear_observable_weight_cache() -> None:
    """Clear retained diagonals for deterministic tests and long-lived workers."""

    with _OBSERVABLE_WEIGHT_CACHE_LOCK:
        _OBSERVABLE_WEIGHT_CACHE.clear()


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
            n_qubits=n_wires,
            qubit=wire,
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
                n_qubits=n_wires,
                qubit=wire,
            )
    return adjoint


def _local_expectation_z_sum(
    shard_state: Any, *, plan: Any, n_wires: int, wires: tuple[int, ...]
) -> torch.Tensor:
    """Evaluate a sum of single-qubit Z terms from one final shard state."""

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


def _local_expectation_z_hamiltonian(
    shard_state: Any,
    *,
    plan: Any,
    n_wires: int,
    terms: tuple[tuple[float, tuple[int, ...]], ...],
) -> torch.Tensor:
    """Evaluate a weighted Z/ZZ Hamiltonian from one final shard state."""

    total = torch.zeros(
        (),
        dtype=shard_state.amplitudes.real.dtype,
        device=shard_state.amplitudes.device,
    )
    local_count = shard_state.shard.local_amplitudes
    for start in range(0, local_count, _reverse_chunk_amplitudes()):
        end = min(local_count, start + _reverse_chunk_amplitudes())
        indices = _storage_global_indices(shard_state, start, end, plan=plan)
        contribution, _ = z_hamiltonian_chunk(
            shard_state.amplitudes[:, start:end],
            indices,
            n_qubits=n_wires,
            terms=terms,
        )
        total = total + contribution
    return total


def _local_expectation_z_hamiltonian_and_weights(
    shard_state: Any,
    *,
    plan: Any,
    n_wires: int,
    terms: tuple[tuple[float, tuple[int, ...]], ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate one Hamiltonian and retain its real diagonal for backward."""

    local_count = shard_state.shard.local_amplitudes
    chunk_amplitudes = _reverse_chunk_amplitudes()
    cache_key = _observable_weight_cache_key(
        shard_state,
        plan=plan,
        n_wires=n_wires,
        terms=terms,
    )
    weights = None if cache_key is None else _cached_observable_weights(cache_key)
    if weights is None:
        weights = torch.empty(
            local_count,
            dtype=shard_state.amplitudes.real.dtype,
            device=shard_state.amplitudes.device,
        )
        for start in range(0, local_count, chunk_amplitudes):
            end = min(local_count, start + chunk_amplitudes)
            indices = _storage_global_indices(shard_state, start, end, plan=plan)
            weights[start:end] = z_hamiltonian_weights(
                indices,
                n_qubits=n_wires,
                terms=terms,
                dtype=weights.dtype,
            )
        if cache_key is not None:
            _cache_observable_weights(cache_key, weights)
    value = fused_observable_expectation(shard_state.amplitudes, weights)
    if value is None:
        value = (shard_state.amplitudes.abs().square() * weights.reshape(1, -1)).sum()
    return value, weights


def _explicit_sharded_adjoint(
    ir: CircuitIR,
    slots: tuple[tuple[int, str, int], ...],
    saved_parameters: tuple[torch.Tensor, ...],
    *,
    observable_terms: tuple[tuple[float, tuple[int, ...]], ...],
    device: torch.device,
    policy: StatevectorCheckpointPolicy,
    process_group: Any | None = None,
    evidence: BackwardExecutionEvidence,
    saved_final_state: Any | None = None,
    saved_observable_weights: torch.Tensor | None = None,
    saved_inter_node_ket_checkpoints: tuple[tuple[int, int, torch.Tensor], ...] = (),
    owners: Sequence[int] | None = None,
) -> tuple[torch.Tensor, ...]:
    """Rematerialize rank-local forward states and propagate gate adjoints."""

    sweep = _ReversibleAdjointSweep(
        ir,
        slots,
        saved_parameters,
        observable_terms=observable_terms,
        device=device,
        policy=policy,
        process_group=process_group,
        evidence=evidence,
        owners=owners,
        saved_final_state=saved_final_state,
        saved_observable_weights=saved_observable_weights,
        saved_inter_node_ket_checkpoints=saved_inter_node_ket_checkpoints,
    )
    return sweep.run()
