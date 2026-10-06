"""MPS rank initialization, shard packaging, and result reconstruction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..runtime_environment import (
    _jnp_device_put,
    _require_jax,
    _require_torch,
    _torch_complex_dtype,
)
from .training_records import JAXMPSRankShardState


def _initialize_jax_mps_rank_tensors(
    *,
    n_qubits: int,
    bsz: int,
    shard_plans: Sequence[Any],
    dtype: Any,
    device: Any | None,
) -> dict[int, dict[int, Any]]:
    _, jnp = _require_jax()
    rank_tensors: dict[int, dict[int, Any]] = {}
    for shard in shard_plans:
        local: dict[int, Any] = {}
        for qubit in shard.qubits:
            tensor = jnp.zeros((int(bsz), 1, 2, 1), dtype=dtype)
            tensor = tensor.at[:, 0, 0, 0].set(jnp.asarray(1.0 + 0.0j, dtype=dtype))
            local[int(qubit)] = _jnp_device_put(tensor, device)
        rank_tensors[int(shard.rank)] = local
    return rank_tensors


def _rank_shards_from_jax_mps_tensors(
    rank_tensors: Mapping[int, Mapping[int, Any]],
    shard_plans: Sequence[Any],
) -> tuple[JAXMPSRankShardState, ...]:
    return tuple(
        JAXMPSRankShardState(
            rank=int(shard.rank),
            qubits=tuple(int(qubit) for qubit in shard.qubits),
            local_tensors=dict(rank_tensors.get(int(shard.rank), {})),
        )
        for shard in shard_plans
    )


def _reconstruct_torch_mps_from_jax_rank_shards(
    rank_shards: Sequence[JAXMPSRankShardState],
    *,
    n_qubits: int,
    max_bond: int | None,
    cutoff: float,
    complex_bytes: int,
    truncation_records: Sequence[Mapping[str, Any]],
) -> Any:
    import numpy as np

    torch = _require_torch()
    from .....simulation.mps.models import MPSConfig, MPSTruncationRecord
    from .....simulation.mps.state import MPSState

    dtype = _torch_complex_dtype(complex_bytes)
    tensors_by_qubit: dict[int, Any] = {}
    for shard in rank_shards:
        for qubit, tensor in shard.local_tensors.items():
            tensors_by_qubit[int(qubit)] = torch.as_tensor(
                np.asarray(tensor).copy(), dtype=dtype, device=torch.device("cpu")
            )
    missing = [qubit for qubit in range(int(n_qubits)) if qubit not in tensors_by_qubit]
    if missing:
        raise RuntimeError(
            f"Cannot reconstruct JAX sharded MPS facade; missing qubits {missing}."
        )
    mps = MPSState(
        [tensors_by_qubit[qubit] for qubit in range(int(n_qubits))],
        config=MPSConfig(max_bond=max_bond, cutoff=float(cutoff)),
    )
    records = [
        MPSTruncationRecord(
            bond=int(record["bond"]),
            kept_rank=int(record["kept_rank"]),
            original_rank=int(record["original_rank"]),
            discarded_weight=float(record["discarded_weight"]),
            max_bond=record.get("max_bond"),
            cutoff=float(record.get("cutoff", 0.0)),
            source=str(record.get("source", "jax_sharded_mps_two_site")),
        )
        for record in truncation_records
    ]
    mps.truncation_records.extend(records)
    mps.truncation_errors.extend(
        float(record.discarded_weight)
        for record in records
        if float(record.discarded_weight) > 0
    )
    mps.orthogonality_center = int(n_qubits) - 1 if n_qubits else None
    return mps
