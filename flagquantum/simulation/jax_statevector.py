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
