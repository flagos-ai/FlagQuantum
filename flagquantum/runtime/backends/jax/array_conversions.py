# ruff: noqa: F401, F821
"""Shared gate, shard, MPS split, and tensor conversion helpers."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, replace
from itertools import product
from typing import Any, Callable, Mapping, Sequence

from ....core.ir import CircuitIR, ensure_circuit_ir
from ...distributed.backend_policy import (
    DistributedBackendPolicy,
    resolve_distributed_backend_policy,
)
from .common import communication_tier as _communication_tier
from .common import env_int as _env_int
from .common import node_count as _node_count
from .common import product_int as _product
from .common import rank_for_wire as _rank_for_wire
from .common import split_contiguous as _split_contiguous
from .release_policy import (
    attach_evidence_contract as _attach_distributed_evidence_contract,
)
from .release_policy import (
    attach_mps_backward_readiness as _attach_mps_backward_readiness,
)
from .release_policy import (
    attach_statevector_claimability as _attach_statevector_claimability,
)


def _jax_basis_indices_for_wires(
    global_indices: Any, *, n_wires: int, wires: Sequence[int]
) -> Any:
    _, jnp = _require_jax()
    basis = jnp.zeros_like(global_indices)
    width = len(tuple(wires))
    for pos, wire in enumerate(tuple(wires)):
        bit = (global_indices >> (int(n_wires) - int(wire) - 1)) & 1
        basis = basis | (bit << (width - pos - 1))
    return basis


def _gate_matrix_as_jax(
    instruction: Any, *, torch_dtype: Any, jax_dtype: Any
) -> tuple[Any, bool]:
    from ..statevector.local_execution import _instruction_matrix

    torch = _require_torch()
    _, jnp = _require_jax()
    matrix_torch = _instruction_matrix(
        instruction, device=torch.device("cpu"), dtype=torch_dtype
    )
    diagonal = bool(
        torch.count_nonzero(
            matrix_torch - torch.diag(torch.diagonal(matrix_torch))
        ).item()
        == 0
    )
    matrix = jnp.asarray(matrix_torch.detach().cpu().numpy(), dtype=jax_dtype)
    return matrix, diagonal


def _parameterized_gate_matrix_as_jax(
    instruction: Any, *, complex_bytes: int
) -> tuple[Any, bool]:
    from ....simulation.jax_gate_primitives import _jax_instruction_matrix

    _, jnp = _require_jax()
    matrix = _jax_instruction_matrix(instruction).astype(
        _jax_complex_dtype(complex_bytes)
    )
    diagonal_gates = {
        "z",
        "s",
        "sdg",
        "t",
        "tdg",
        "rz",
        "phase",
        "u1",
        "cz",
        "crz",
        "cphase",
        "rzz",
    }
    return matrix, str(instruction.name).lower() in diagonal_gates


def _apply_gate_to_jax_shards(
    shards: Sequence[JAXStatevectorShardState],
    matrix: Any,
    wires: Sequence[int],
    *,
    plan: Any,
    diagonal: bool,
) -> tuple[JAXStatevectorShardState, ...]:
    from ..statevector.local_execution import _basis_offset, _wire_mask

    _, jnp = _require_jax()
    shards = tuple(shards)
    if not shards:
        return ()
    wires = tuple(int(wire) for wire in wires)
    width = len(wires)
    gate_dim = 2**width
    if tuple(matrix.shape[-2:]) != (gate_dim, gate_dim):
        raise ValueError(
            f"Gate on {width} wires requires matrix shape {(gate_dim, gate_dim)}."
        )

    if diagonal:
        diagonal_values = jnp.diagonal(matrix)
        updated_shards = []
        for shard in shards:
            basis_indices = _jax_basis_indices_for_wires(
                shard.global_indices,
                n_wires=plan.n_wires,
                wires=wires,
            )
            factors = diagonal_values[basis_indices].reshape(1, -1)
            updated_shards.append(
                JAXStatevectorShardState(
                    rank=shard.rank,
                    shard=shard.shard,
                    amplitudes=shard.amplitudes * factors,
                    global_indices=shard.global_indices,
                    global_indices_tuple=shard.global_indices_tuple,
                )
            )
        return tuple(updated_shards)

    masks = tuple(_wire_mask(plan.n_wires, wire) for wire in wires)
    offsets = tuple(
        _basis_offset(plan.n_wires, wires, basis) for basis in range(gate_dim)
    )
    positions_by_global: dict[int, tuple[int, int]] = {}
    for shard_index, shard in enumerate(shards):
        for local_index, global_index in enumerate(shard.global_indices_tuple):
            positions_by_global[int(global_index)] = (shard_index, local_index)

    out_amplitudes = [shard.amplitudes for shard in shards]
    processed_bases: set[int] = set()
    for shard in shards:
        for global_index in shard.global_indices_tuple:
            base = int(global_index)
            for mask in masks:
                base &= ~mask
            if base in processed_bases:
                continue
            processed_bases.add(base)
            group_indices = tuple(base | offset for offset in offsets)
            missing = tuple(
                index for index in group_indices if index not in positions_by_global
            )
            if missing:
                raise ValueError(
                    f"Shard plan does not cover gate basis group; missing amplitudes {missing}."
                )
            locations = tuple(positions_by_global[index] for index in group_indices)
            vector = jnp.stack(
                [
                    shards[shard_index].amplitudes[:, local_index]
                    for shard_index, local_index in locations
                ],
                axis=-1,
            )
            updated = vector @ jnp.swapaxes(matrix, -2, -1)
            for basis_index, (shard_index, local_index) in enumerate(locations):
                out_amplitudes[shard_index] = (
                    out_amplitudes[shard_index]
                    .at[:, local_index]
                    .set(updated[:, basis_index])
                )

    return tuple(
        JAXStatevectorShardState(
            rank=shard.rank,
            shard=shard.shard,
            amplitudes=out_amplitudes[shard_index],
            global_indices=shard.global_indices,
            global_indices_tuple=shard.global_indices_tuple,
        )
        for shard_index, shard in enumerate(shards)
    )


def _jax_apply_matrix_to_batched_local_state(
    state: Any,
    matrix: Any,
    wires: Sequence[int],
    *,
    n_local_wires: int,
) -> Any:
    _, jnp = _require_jax()
    wires = tuple(int(wire) for wire in wires)
    k = len(wires)
    dim = 2**k
    if tuple(matrix.shape[-2:]) != (dim, dim):
        raise ValueError(f"Gate on {k} local wires requires matrix shape {(dim, dim)}.")
    rest = tuple(wire for wire in range(int(n_local_wires)) if wire not in wires)
    perm = (0,) + tuple(wire + 1 for wire in wires + rest)
    inv_perm = [0] * (int(n_local_wires) + 1)
    for index, axis in enumerate(perm):
        inv_perm[axis] = index
    tensor = state.reshape((state.shape[0],) + (2,) * int(n_local_wires)).transpose(
        perm
    )
    flat = tensor.reshape(state.shape[0], dim, -1)
    if matrix.ndim == 2:
        out = jnp.einsum("ij,bjk->bik", matrix, flat, precision="highest")
    else:
        out = jnp.einsum("bij,bjk->bik", matrix, flat, precision="highest")
    return (
        out.reshape((state.shape[0],) + (2,) * int(n_local_wires))
        .transpose(tuple(inv_perm))
        .reshape(state.shape)
    )


def _jax_rank_mask_for_touched_delta(
    plan: Any, touched_sharded_wires: Sequence[int], delta_code: int
) -> int:
    sharded_wires = tuple(int(wire) for wire in plan.sharded_wires)
    touched = tuple(int(wire) for wire in touched_sharded_wires)
    rank_mask = 0
    for offset, wire in enumerate(touched):
        delta_bit = (int(delta_code) >> (len(touched) - offset - 1)) & 1
        if not delta_bit:
            continue
        bit_index = sharded_wires.index(int(wire))
        rank_mask |= 1 << (len(sharded_wires) - bit_index - 1)
    return int(rank_mask)


def _jax_local_positions_for_gate_input(
    global_indices: Any,
    *,
    n_wires: int,
    local_wires: Sequence[int],
    local_gate_wires: Sequence[int],
    local_input_basis: int,
) -> Any:
    _, jnp = _require_jax()
    local_wires = tuple(int(wire) for wire in local_wires)
    local_gate_wires = tuple(int(wire) for wire in local_gate_wires)
    local_gate_positions = {wire: index for index, wire in enumerate(local_gate_wires)}
    positions = jnp.zeros_like(global_indices)
    for wire in local_wires:
        gate_position = local_gate_positions.get(int(wire))
        if gate_position is None:
            bit = (global_indices >> (int(n_wires) - int(wire) - 1)) & 1
        else:
            bit = (
                int(local_input_basis) >> (len(local_gate_wires) - gate_position - 1)
            ) & 1
        positions = (positions << 1) | bit
    return positions


def _jax_gate_basis_in_for_delta_and_local_input(
    global_indices: Any,
    *,
    n_wires: int,
    wires: Sequence[int],
    touched_sharded_wires: Sequence[int],
    local_gate_wires: Sequence[int],
    delta_code: int,
    local_input_basis: int,
) -> Any:
    _, jnp = _require_jax()
    wires = tuple(int(wire) for wire in wires)
    touched = tuple(int(wire) for wire in touched_sharded_wires)
    local_gate_wires = tuple(int(wire) for wire in local_gate_wires)
    touched_positions = {wire: index for index, wire in enumerate(touched)}
    local_gate_positions = {wire: index for index, wire in enumerate(local_gate_wires)}
    basis = jnp.zeros_like(global_indices)
    for wire_position, wire in enumerate(wires):
        if int(wire) in touched_positions:
            output_bit = (global_indices >> (int(n_wires) - int(wire) - 1)) & 1
            delta_position = touched_positions[int(wire)]
            delta_bit = (int(delta_code) >> (len(touched) - delta_position - 1)) & 1
            bit = output_bit ^ delta_bit
        else:
            local_position = local_gate_positions[int(wire)]
            bit = (
                int(local_input_basis) >> (len(local_gate_wires) - local_position - 1)
            ) & 1
        basis = basis | (bit << (len(wires) - wire_position - 1))
    return basis


def _reconstruct_torch_state_from_jax_shards(
    shards: Sequence[JAXStatevectorShardState], plan: Any
) -> Any:
    import numpy as np

    torch = _require_torch()
    dtype = _torch_complex_dtype(plan.complex_bytes)
    state = torch.zeros(
        (plan.bsz, plan.total_amplitudes), dtype=dtype, device=torch.device("cpu")
    )
    for shard in shards:
        values = torch.as_tensor(
            np.asarray(shard.amplitudes).copy(), dtype=dtype, device=torch.device("cpu")
        )
        state[:, list(shard.global_indices_tuple)] = values
    return state


def _select_jax_split_rank(values: Any, *, max_bond: int | None, cutoff: float) -> int:
    import numpy as np

    values_np = np.asarray(values)
    if values_np.ndim == 1:
        values_np = values_np.reshape(1, -1)
    full_rank = int(values_np.shape[-1])
    rank = full_rank
    if float(cutoff) > 0:
        counts = (values_np > float(cutoff)).sum(axis=-1)
        rank = max(1, int(counts.min()))
    if max_bond is not None:
        rank = min(rank, int(max_bond))
    return max(1, int(rank))


def _discarded_weight_jax(values: Any, rank: int) -> float:
    import numpy as np

    values_np = np.asarray(values)
    if int(rank) >= int(values_np.shape[-1]):
        return 0.0
    discarded = values_np[..., int(rank) :]
    return float(np.max(np.sum(np.real(discarded * np.conj(discarded)), axis=-1)))


def _split_pair_matrix_jax(
    matrix: Any,
    *,
    left_dim: int,
    right_dim: int,
    max_bond: int | None,
    cutoff: float,
) -> tuple[Any, Any, dict[str, Any]]:
    _, jnp = _require_jax()
    bsz = int(matrix.shape[0])
    rows = int(matrix.shape[-2])
    cols = int(matrix.shape[-1])
    full_rank = min(rows, cols)
    exact_split = float(cutoff) <= 0 and (
        max_bond is None or int(max_bond) >= full_rank
    )
    if exact_split:
        if rows <= cols:
            rank = rows
            eye = jnp.eye(rank, dtype=matrix.dtype)
            left = jnp.broadcast_to(
                eye.reshape(1, int(left_dim), 2, rank), (bsz, int(left_dim), 2, rank)
            )
            right = matrix.reshape(bsz, rank, 2, int(right_dim))
            method = "identity_left_gauge"
        else:
            rank = cols
            left = matrix.reshape(bsz, int(left_dim), 2, rank)
            eye = jnp.eye(rank, dtype=matrix.dtype)
            right = jnp.broadcast_to(
                eye.reshape(1, rank, 2, int(right_dim)), (bsz, rank, 2, int(right_dim))
            )
            method = "identity_right_gauge"
        return (
            left,
            right,
            {
                "method": method,
                "rank": rank,
                "original_rank": full_rank,
                "discarded_weight": 0.0,
            },
        )
    u, s, vh = jnp.linalg.svd(matrix, full_matrices=False)
    rank = _select_jax_split_rank(s, max_bond=max_bond, cutoff=cutoff)
    step_error = _discarded_weight_jax(s, rank)
    left = u[:, :, :rank].reshape(bsz, int(left_dim), 2, rank)
    right = (s[:, :rank, None] * vh[:, :rank, :]).reshape(bsz, rank, 2, int(right_dim))
    return (
        left,
        right,
        {
            "method": "svd",
            "rank": rank,
            "original_rank": full_rank,
            "discarded_weight": float(step_error),
        },
    )


def _jax_split_record(
    split_info: Mapping[str, Any],
    *,
    bond: int,
    max_bond: int | None,
    cutoff: float,
) -> dict[str, Any] | None:
    if split_info.get("method") != "svd":
        return None
    return {
        "bond": int(bond),
        "kept_rank": int(split_info["rank"]),
        "original_rank": int(split_info["original_rank"]),
        "discarded_weight": float(split_info["discarded_weight"]),
        "max_bond": max_bond,
        "cutoff": float(cutoff),
        "source": "jax_sharded_mps_two_site",
    }


def _jax_nodes_from_torch_nodes(
    nodes: Sequence[Any], *, dtype: Any, device: Any | None
) -> tuple[JAXTensorNetworkNode, ...]:
    import numpy as np

    _, jnp = _require_jax()
    out = []
    for node in nodes:
        tensor = jnp.asarray(np.asarray(node.tensor.detach().cpu()).copy(), dtype=dtype)
        tensor = _jnp_device_put(tensor, device)
        out.append(
            JAXTensorNetworkNode(
                tensor=tensor,
                labels=tuple(int(label) for label in node.labels),
                name=str(node.name),
                metadata=node.metadata,
            )
        )
    return tuple(out)
