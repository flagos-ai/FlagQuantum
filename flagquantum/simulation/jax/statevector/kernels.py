"""Dependency-light JAX statevector numerics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def jax_basis_indices_for_qubits(
    global_indices: Any,
    *,
    n_qubits: int,
    qubits: Sequence[int],
) -> Any:
    """Pack selected global-index bits into gate-basis indices."""

    import jax.numpy as jnp

    basis = jnp.zeros_like(global_indices)
    width = len(tuple(qubits))
    for position, qubit in enumerate(tuple(qubits)):
        bit = (global_indices >> (int(n_qubits) - int(qubit) - 1)) & 1
        basis = basis | (bit << (width - position - 1))
    return basis


def jax_initial_statevector_shard(
    global_indices: Any,
    *,
    batch_size: int,
    dtype: Any,
) -> Any:
    """Create a batched rank-local shard of the global zero state."""

    import jax.numpy as jnp

    amplitudes = jnp.zeros(
        (int(batch_size), int(global_indices.shape[0])),
        dtype=dtype,
    )
    initial = jnp.where(
        global_indices == 0,
        jnp.asarray(1.0 + 0.0j, dtype=amplitudes.dtype),
        jnp.asarray(0.0 + 0.0j, dtype=amplitudes.dtype),
    )
    return amplitudes.at[:, :].set(initial.reshape(1, -1))


def jax_rank_mask_for_touched_delta(
    sharded_qubits: Sequence[int],
    touched_sharded_qubits: Sequence[int],
    delta_code: int,
) -> int:
    """Translate touched sharded-qubit deltas into an xor rank mask."""

    sharded_qubits = tuple(int(qubit) for qubit in sharded_qubits)
    touched = tuple(int(qubit) for qubit in touched_sharded_qubits)
    rank_mask = 0
    for offset, qubit in enumerate(touched):
        delta_bit = (int(delta_code) >> (len(touched) - offset - 1)) & 1
        if not delta_bit:
            continue
        bit_index = sharded_qubits.index(qubit)
        rank_mask |= 1 << (len(sharded_qubits) - bit_index - 1)
    return int(rank_mask)


def jax_local_positions_for_gate_input(
    global_indices: Any,
    *,
    n_qubits: int,
    local_qubits: Sequence[int],
    local_gate_qubits: Sequence[int],
    local_input_basis: int,
) -> Any:
    """Map one local gate-input basis to positions in a rank-local shard."""

    import jax.numpy as jnp

    local_qubits = tuple(int(qubit) for qubit in local_qubits)
    local_gate_qubits = tuple(int(qubit) for qubit in local_gate_qubits)
    local_gate_positions = {
        qubit: index for index, qubit in enumerate(local_gate_qubits)
    }
    positions = jnp.zeros_like(global_indices)
    for qubit in local_qubits:
        gate_position = local_gate_positions.get(qubit)
        if gate_position is None:
            bit = (global_indices >> (int(n_qubits) - qubit - 1)) & 1
        else:
            bit = (
                int(local_input_basis) >> (len(local_gate_qubits) - gate_position - 1)
            ) & 1
        positions = (positions << 1) | bit
    return positions


def jax_gate_basis_in_for_delta_and_local_input(
    global_indices: Any,
    *,
    n_qubits: int,
    qubits: Sequence[int],
    touched_sharded_qubits: Sequence[int],
    local_gate_qubits: Sequence[int],
    delta_code: int,
    local_input_basis: int,
) -> Any:
    """Construct input basis indices for one cross-rank gate contribution."""

    import jax.numpy as jnp

    qubits = tuple(int(qubit) for qubit in qubits)
    touched = tuple(int(qubit) for qubit in touched_sharded_qubits)
    local_gate_qubits = tuple(int(qubit) for qubit in local_gate_qubits)
    touched_positions = {qubit: index for index, qubit in enumerate(touched)}
    local_gate_positions = {
        qubit: index for index, qubit in enumerate(local_gate_qubits)
    }
    basis = jnp.zeros_like(global_indices)
    for qubit_position, qubit in enumerate(qubits):
        if qubit in touched_positions:
            output_bit = (global_indices >> (int(n_qubits) - qubit - 1)) & 1
            delta_position = touched_positions[qubit]
            delta_bit = (int(delta_code) >> (len(touched) - delta_position - 1)) & 1
            bit = output_bit ^ delta_bit
        else:
            local_position = local_gate_positions[qubit]
            bit = (
                int(local_input_basis) >> (len(local_gate_qubits) - local_position - 1)
            ) & 1
        basis = basis | (bit << (len(qubits) - qubit_position - 1))
    return basis


def jax_accumulate_all_to_all_statevector_delta(
    updated: Any,
    source_amplitudes: Any,
    global_indices: Any,
    matrix: Any,
    basis_out: Any,
    *,
    n_qubits: int,
    qubits: Sequence[int],
    touched_sharded_qubits: Sequence[int],
    local_qubits: Sequence[int],
    local_gate_qubits: Sequence[int],
    delta_code: int,
) -> Any:
    """Accumulate one exchanged-rank delta into an all-to-all gate result."""

    import jax.numpy as jnp

    local_gate_qubits = tuple(int(qubit) for qubit in local_gate_qubits)
    for local_input_basis in range(2 ** len(local_gate_qubits)):
        source_positions = jax_local_positions_for_gate_input(
            global_indices,
            n_qubits=int(n_qubits),
            local_qubits=local_qubits,
            local_gate_qubits=local_gate_qubits,
            local_input_basis=local_input_basis,
        )
        source_values = jnp.take(source_amplitudes, source_positions, axis=1)
        basis_in = jax_gate_basis_in_for_delta_and_local_input(
            global_indices,
            n_qubits=int(n_qubits),
            qubits=qubits,
            touched_sharded_qubits=touched_sharded_qubits,
            local_gate_qubits=local_gate_qubits,
            delta_code=int(delta_code),
            local_input_basis=local_input_basis,
        )
        if matrix.ndim == 2:
            coefficient = matrix[basis_out, basis_in].reshape(1, -1)
        else:
            coefficient = matrix[:, basis_out, basis_in]
        updated = updated + source_values * coefficient
    return updated


def jax_apply_matrix_to_batched_local_state(
    state: Any,
    matrix: Any,
    qubits: Sequence[int],
    *,
    n_local_qubits: int,
) -> Any:
    """Apply a gate matrix to selected qubits of a batched local state."""

    import jax.numpy as jnp

    qubits = tuple(int(qubit) for qubit in qubits)
    width = len(qubits)
    dimension = 2**width
    if tuple(matrix.shape[-2:]) != (dimension, dimension):
        raise ValueError(
            f"Gate on {width} local qubits requires matrix shape "
            f"{(dimension, dimension)}."
        )
    remaining = tuple(
        qubit for qubit in range(int(n_local_qubits)) if qubit not in qubits
    )
    permutation = (0,) + tuple(qubit + 1 for qubit in qubits + remaining)
    inverse = [0] * (int(n_local_qubits) + 1)
    for index, axis in enumerate(permutation):
        inverse[axis] = index
    tensor = state.reshape((state.shape[0],) + (2,) * int(n_local_qubits)).transpose(
        permutation
    )
    flat = tensor.reshape(state.shape[0], dimension, -1)
    if matrix.ndim == 2:
        output = jnp.einsum("ij,bjk->bik", matrix, flat, precision="highest")
    else:
        output = jnp.einsum("bij,bjk->bik", matrix, flat, precision="highest")
    return (
        output.reshape((state.shape[0],) + (2,) * int(n_local_qubits))
        .transpose(tuple(inverse))
        .reshape(state.shape)
    )


def jax_apply_local_statevector_gate(
    amplitudes: Any,
    global_indices: Any,
    matrix: Any,
    qubits: Sequence[int],
    *,
    n_qubits: int,
    sharded_qubits: Sequence[int],
    diagonal: bool,
    gate_name: str | None = None,
) -> Any:
    """Apply a gate that requires no cross-rank amplitude exchange."""

    import jax.numpy as jnp

    qubits = tuple(int(qubit) for qubit in qubits)
    sharded_qubits = tuple(int(qubit) for qubit in sharded_qubits)
    if diagonal:
        factors = jnp.diagonal(matrix)[
            jax_basis_indices_for_qubits(
                global_indices,
                n_qubits=int(n_qubits),
                qubits=qubits,
            )
        ].reshape(1, -1)
        return amplitudes * factors
    sharded_set = set(sharded_qubits)
    touched = tuple(sorted(sharded_set & set(qubits)))
    if touched:
        raise RuntimeError(
            f"Non-diagonal gate {gate_name!r} touches sharded qubits {touched} "
            "and requires "
            "pair-exchange/all-to-all transport."
        )
    local_qubits = tuple(
        qubit for qubit in range(int(n_qubits)) if qubit not in sharded_set
    )
    local_map = {qubit: index for index, qubit in enumerate(local_qubits)}
    mapped_qubits = tuple(local_map[qubit] for qubit in qubits)
    return jax_apply_matrix_to_batched_local_state(
        amplitudes,
        matrix,
        mapped_qubits,
        n_local_qubits=len(local_qubits),
    )


def jax_combine_pair_exchanged_statevector(
    amplitudes: Any,
    partner_amplitudes: Any,
    global_indices: Any,
    matrix: Any,
    *,
    n_qubits: int,
    qubit: int,
) -> Any:
    """Combine local and exchanged amplitudes for a one-qubit gate."""

    import jax.numpy as jnp

    bit = ((global_indices >> (int(n_qubits) - 1 - int(qubit))) & 1).astype(jnp.bool_)
    current_is_zero = bit.reshape(1, -1) == 0
    updated_zero = matrix[0, 0] * amplitudes + matrix[0, 1] * partner_amplitudes
    updated_one = matrix[1, 0] * partner_amplitudes + matrix[1, 1] * amplitudes
    return jnp.where(current_is_zero, updated_zero, updated_one)


def jax_sharded_statevector_rank_loss(
    amplitudes: Any,
    global_indices: Any,
    *,
    n_qubits: int,
    observable: str,
    observable_qubits: Sequence[int] | None,
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
    qubits = (
        tuple(range(int(n_qubits)))
        if observable_qubits is None
        else tuple(int(qubit) for qubit in observable_qubits)
    )
    probabilities = jnp.abs(amplitudes) ** 2
    total = jnp.zeros((), dtype=probabilities.real.dtype)
    for qubit in qubits:
        bit = (global_indices >> (int(n_qubits) - 1 - int(qubit))) & 1
        signs = 1.0 - 2.0 * bit.astype(probabilities.real.dtype)
        total = total + jnp.sum(probabilities * signs.reshape(1, -1))
    return jnp.real(total)


def jax_sharded_statevector_loss(
    amplitudes_by_shard: Sequence[Any],
    global_indices_by_shard: Sequence[Any],
    *,
    n_qubits: int,
    observable: str,
    observable_qubits: Sequence[int] | None,
) -> Any:
    """Evaluate an observable over an in-process collection of statevector shards."""

    import jax.numpy as jnp

    amplitudes_by_shard = tuple(amplitudes_by_shard)
    global_indices_by_shard = tuple(global_indices_by_shard)
    if len(amplitudes_by_shard) != len(global_indices_by_shard):
        raise ValueError("Shard amplitudes and global indices must have equal lengths.")
    normalized = str(observable)
    total = jnp.zeros(
        (),
        dtype=(
            jnp.real(amplitudes_by_shard[0]).dtype
            if amplitudes_by_shard
            else jnp.float32
        ),
    )
    if normalized == "state_norm":
        for amplitudes in amplitudes_by_shard:
            total = total + jnp.sum(jnp.abs(amplitudes) ** 2)
        return jnp.real(total)
    if normalized not in {"z", "z_sum"}:
        raise ValueError(
            "JAX sharded statevector parameter gradients currently support "
            "observable='z_sum', 'z', or 'state_norm'."
        )
    qubits = (
        tuple(range(int(n_qubits)))
        if observable_qubits is None
        else tuple(int(qubit) for qubit in observable_qubits)
    )
    for amplitudes, global_indices in zip(
        amplitudes_by_shard, global_indices_by_shard, strict=True
    ):
        probabilities = jnp.abs(amplitudes) ** 2
        for qubit in qubits:
            bit = (global_indices >> (int(n_qubits) - 1 - int(qubit))) & 1
            signs = 1.0 - 2.0 * bit.astype(probabilities.real.dtype)
            total = total + jnp.sum(probabilities * signs.reshape(1, -1))
    return jnp.real(total)
