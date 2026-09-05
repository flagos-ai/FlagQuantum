"""Dependency-light local JAX statevector numerics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def jax_basis_indices_for_wires(
    global_indices: Any,
    *,
    n_wires: int,
    wires: Sequence[int],
) -> Any:
    """Pack selected global-index bits into gate-basis indices."""

    import jax.numpy as jnp

    basis = jnp.zeros_like(global_indices)
    width = len(tuple(wires))
    for position, wire in enumerate(tuple(wires)):
        bit = (global_indices >> (int(n_wires) - int(wire) - 1)) & 1
        basis = basis | (bit << (width - position - 1))
    return basis


def jax_rank_mask_for_touched_delta(
    sharded_wires: Sequence[int],
    touched_sharded_wires: Sequence[int],
    delta_code: int,
) -> int:
    """Translate touched sharded-wire deltas into an xor rank mask."""

    sharded_wires = tuple(int(wire) for wire in sharded_wires)
    touched = tuple(int(wire) for wire in touched_sharded_wires)
    rank_mask = 0
    for offset, wire in enumerate(touched):
        delta_bit = (int(delta_code) >> (len(touched) - offset - 1)) & 1
        if not delta_bit:
            continue
        bit_index = sharded_wires.index(wire)
        rank_mask |= 1 << (len(sharded_wires) - bit_index - 1)
    return int(rank_mask)


def jax_local_positions_for_gate_input(
    global_indices: Any,
    *,
    n_wires: int,
    local_wires: Sequence[int],
    local_gate_wires: Sequence[int],
    local_input_basis: int,
) -> Any:
    """Map one local gate-input basis to positions in a rank-local shard."""

    import jax.numpy as jnp

    local_wires = tuple(int(wire) for wire in local_wires)
    local_gate_wires = tuple(int(wire) for wire in local_gate_wires)
    local_gate_positions = {wire: index for index, wire in enumerate(local_gate_wires)}
    positions = jnp.zeros_like(global_indices)
    for wire in local_wires:
        gate_position = local_gate_positions.get(wire)
        if gate_position is None:
            bit = (global_indices >> (int(n_wires) - wire - 1)) & 1
        else:
            bit = (
                int(local_input_basis) >> (len(local_gate_wires) - gate_position - 1)
            ) & 1
        positions = (positions << 1) | bit
    return positions


def jax_gate_basis_in_for_delta_and_local_input(
    global_indices: Any,
    *,
    n_wires: int,
    wires: Sequence[int],
    touched_sharded_wires: Sequence[int],
    local_gate_wires: Sequence[int],
    delta_code: int,
    local_input_basis: int,
) -> Any:
    """Construct input basis indices for one cross-rank gate contribution."""

    import jax.numpy as jnp

    wires = tuple(int(wire) for wire in wires)
    touched = tuple(int(wire) for wire in touched_sharded_wires)
    local_gate_wires = tuple(int(wire) for wire in local_gate_wires)
    touched_positions = {wire: index for index, wire in enumerate(touched)}
    local_gate_positions = {wire: index for index, wire in enumerate(local_gate_wires)}
    basis = jnp.zeros_like(global_indices)
    for wire_position, wire in enumerate(wires):
        if wire in touched_positions:
            output_bit = (global_indices >> (int(n_wires) - wire - 1)) & 1
            delta_position = touched_positions[wire]
            delta_bit = (int(delta_code) >> (len(touched) - delta_position - 1)) & 1
            bit = output_bit ^ delta_bit
        else:
            local_position = local_gate_positions[wire]
            bit = (
                int(local_input_basis) >> (len(local_gate_wires) - local_position - 1)
            ) & 1
        basis = basis | (bit << (len(wires) - wire_position - 1))
    return basis


def jax_apply_matrix_to_batched_local_state(
    state: Any,
    matrix: Any,
    wires: Sequence[int],
    *,
    n_local_wires: int,
) -> Any:
    """Apply a gate matrix to selected wires of a batched local state."""

    import jax.numpy as jnp

    wires = tuple(int(wire) for wire in wires)
    width = len(wires)
    dimension = 2**width
    if tuple(matrix.shape[-2:]) != (dimension, dimension):
        raise ValueError(
            f"Gate on {width} local wires requires matrix shape "
            f"{(dimension, dimension)}."
        )
    remaining = tuple(wire for wire in range(int(n_local_wires)) if wire not in wires)
    permutation = (0,) + tuple(wire + 1 for wire in wires + remaining)
    inverse = [0] * (int(n_local_wires) + 1)
    for index, axis in enumerate(permutation):
        inverse[axis] = index
    tensor = state.reshape((state.shape[0],) + (2,) * int(n_local_wires)).transpose(
        permutation
    )
    flat = tensor.reshape(state.shape[0], dimension, -1)
    if matrix.ndim == 2:
        output = jnp.einsum("ij,bjk->bik", matrix, flat, precision="highest")
    else:
        output = jnp.einsum("bij,bjk->bik", matrix, flat, precision="highest")
    return (
        output.reshape((state.shape[0],) + (2,) * int(n_local_wires))
        .transpose(tuple(inverse))
        .reshape(state.shape)
    )


def jax_apply_local_statevector_gate(
    amplitudes: Any,
    global_indices: Any,
    matrix: Any,
    wires: Sequence[int],
    *,
    n_wires: int,
    sharded_wires: Sequence[int],
    diagonal: bool,
    gate_name: str | None = None,
) -> Any:
    """Apply a gate that requires no cross-rank amplitude exchange."""

    import jax.numpy as jnp

    wires = tuple(int(wire) for wire in wires)
    sharded_wires = tuple(int(wire) for wire in sharded_wires)
    if diagonal:
        factors = jnp.diagonal(matrix)[
            jax_basis_indices_for_wires(
                global_indices,
                n_wires=int(n_wires),
                wires=wires,
            )
        ].reshape(1, -1)
        return amplitudes * factors
    sharded_set = set(sharded_wires)
    touched = tuple(sorted(sharded_set & set(wires)))
    if touched:
        raise RuntimeError(
            f"Non-diagonal gate {gate_name!r} touches sharded wires {touched} "
            "and requires "
            "pair-exchange/all-to-all transport."
        )
    local_wires = tuple(wire for wire in range(int(n_wires)) if wire not in sharded_set)
    local_map = {wire: index for index, wire in enumerate(local_wires)}
    mapped_wires = tuple(local_map[wire] for wire in wires)
    return jax_apply_matrix_to_batched_local_state(
        amplitudes,
        matrix,
        mapped_wires,
        n_local_wires=len(local_wires),
    )


def jax_combine_pair_exchanged_statevector(
    amplitudes: Any,
    partner_amplitudes: Any,
    global_indices: Any,
    matrix: Any,
    *,
    n_wires: int,
    wire: int,
) -> Any:
    """Combine local and exchanged amplitudes for a one-wire gate."""

    import jax.numpy as jnp

    bit = ((global_indices >> (int(n_wires) - 1 - int(wire))) & 1).astype(jnp.bool_)
    current_is_zero = bit.reshape(1, -1) == 0
    updated_zero = matrix[0, 0] * amplitudes + matrix[0, 1] * partner_amplitudes
    updated_one = matrix[1, 0] * partner_amplitudes + matrix[1, 1] * amplitudes
    return jnp.where(current_is_zero, updated_zero, updated_one)


def jax_sharded_statevector_rank_loss(
    amplitudes: Any,
    global_indices: Any,
    *,
    n_wires: int,
    observable: str,
    observable_wires: Sequence[int] | None,
) -> Any:
    """Evaluate a rank-local norm or Z-family observable contribution."""

    import jax.numpy as jnp

    normalized = str(observable)
    if normalized == "state_norm":
        return jnp.real(jnp.sum(jnp.abs(amplitudes) ** 2))
    if normalized not in {"z", "z_sum"}:
        raise ValueError(
            "JAX pmap sharded statevector gradients currently support "
            "observable='z_sum', 'z', or 'state_norm'."
        )
    wires = (
        tuple(range(int(n_wires)))
        if observable_wires is None
        else tuple(int(wire) for wire in observable_wires)
    )
    probabilities = jnp.abs(amplitudes) ** 2
    total = jnp.zeros((), dtype=probabilities.real.dtype)
    for wire in wires:
        bit = (global_indices >> (int(n_wires) - 1 - int(wire))) & 1
        signs = 1.0 - 2.0 * bit.astype(probabilities.real.dtype)
        total = total + jnp.sum(probabilities * signs.reshape(1, -1))
    return jnp.real(total)
