"""Dependency-light JAX MPS tensor operations."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from .jax_gate_primitives import (
    _jax_complex_dtype,
    _jax_pauli_matrix,
    _jax_real_dtype,
    _jax_swap,
)


def jax_mps_apply_one(tensor: Any, matrix: Any, matmul_precision: str | None) -> Any:
    import jax.numpy as jnp

    return jnp.einsum("pq,lqr->lpr", matrix, tensor, precision=matmul_precision)


def jax_mps_apply_two_remote(
    tensors: list[Any],
    matrix: Any,
    wires: tuple[int, int],
    *,
    max_bond: int | None,
    cutoff: float,
    matmul_precision: str | None,
) -> list[Any]:
    first, second = int(wires[0]), int(wires[1])
    left = min(first, second)
    right = max(first, second)
    for wire in range(right - 1, left, -1):
        tensors = jax_mps_apply_two_adjacent(
            tensors,
            _jax_swap(),
            wire,
            reverse=False,
            max_bond=max_bond,
            cutoff=cutoff,
            matmul_precision=matmul_precision,
        )
    tensors = jax_mps_apply_two_adjacent(
        tensors,
        matrix,
        left,
        reverse=first > second,
        max_bond=max_bond,
        cutoff=cutoff,
        matmul_precision=matmul_precision,
    )
    for wire in range(left + 1, right):
        tensors = jax_mps_apply_two_adjacent(
            tensors,
            _jax_swap(),
            wire,
            reverse=False,
            max_bond=max_bond,
            cutoff=cutoff,
            matmul_precision=matmul_precision,
        )
    return tensors


def jax_mps_apply_two_adjacent(
    tensors: list[Any],
    matrix: Any,
    left_wire: int,
    *,
    reverse: bool,
    max_bond: int | None,
    cutoff: float,
    matmul_precision: str | None,
) -> list[Any]:
    import jax.numpy as jnp

    left = tensors[int(left_wire)]
    right = tensors[int(left_wire) + 1]
    theta = jnp.einsum("lsm,mtr->lstr", left, right, precision=matmul_precision)
    if reverse:
        theta = jnp.swapaxes(theta, 1, 2)
    left_dim, _, _, right_dim = theta.shape
    theta = theta.reshape(left_dim, 4, right_dim)
    flat = theta.transpose(1, 0, 2).reshape(4, -1)
    applied = jnp.matmul(matrix, flat, precision=matmul_precision)
    theta = applied.reshape(4, left_dim, right_dim).transpose(1, 0, 2)
    theta = theta.reshape(left_dim, 2, 2, right_dim)
    if reverse:
        theta = jnp.swapaxes(theta, 1, 2)
    next_left, next_right = jax_mps_split_pair(theta, max_bond=max_bond, cutoff=cutoff)
    out = list(tensors)
    out[int(left_wire)] = next_left
    out[int(left_wire) + 1] = next_right
    return out


def jax_mps_split_pair(
    theta: Any, *, max_bond: int | None, cutoff: float
) -> tuple[Any, Any]:
    import jax.numpy as jnp

    left_dim, _, _, right_dim = theta.shape
    matrix = theta.reshape(left_dim * 2, 2 * right_dim)
    full_rank = min(int(matrix.shape[0]), int(matrix.shape[1]))
    exact = float(cutoff) <= 0.0 and (
        max_bond is None or int(max_bond) >= int(full_rank)
    )
    if exact:
        if int(matrix.shape[0]) <= int(matrix.shape[1]):
            rank = int(matrix.shape[0])
            eye = jnp.eye(rank, dtype=matrix.dtype)
            return eye.reshape(left_dim, 2, rank), matrix.reshape(rank, 2, right_dim)
        rank = int(matrix.shape[1])
        eye = jnp.eye(rank, dtype=matrix.dtype)
        return matrix.reshape(left_dim, 2, rank), eye.reshape(rank, 2, right_dim)

    u, s, vh = jnp.linalg.svd(matrix, full_matrices=False)
    rank = int(full_rank if max_bond is None else min(int(max_bond), int(full_rank)))
    u = u[:, :rank]
    s = s[:rank]
    vh = vh[:rank, :]
    return u.reshape(left_dim, 2, rank), (s[:, None] * vh).reshape(rank, 2, right_dim)


def jax_mps_to_statevector(
    tensors: Sequence[Any], matmul_precision: str | None = "highest"
) -> Any:
    import jax.numpy as jnp

    state = tensors[0][0, :, :]
    for tensor in tensors[1:]:
        state = jnp.einsum("...l,lsr->...sr", state, tensor, precision=matmul_precision)
    return state.reshape(-1)


def jax_mps_transfer_identity(
    env: Any, tensor: Any, matmul_precision: str | None
) -> Any:
    import jax.numpy as jnp

    return jnp.einsum(
        "ij,ipr,jps->rs", env, jnp.conj(tensor), tensor, precision=matmul_precision
    )


def jax_mps_transfer_op(
    env: Any, tensor: Any, op: Any, matmul_precision: str | None
) -> Any:
    import jax.numpy as jnp

    return jnp.einsum(
        "ij,ipr,pq,jqs->rs",
        env,
        jnp.conj(tensor),
        op,
        tensor,
        precision=matmul_precision,
    )


def jax_mps_transfer_identity_right(
    env: Any, tensor: Any, matmul_precision: str | None
) -> Any:
    import jax.numpy as jnp

    return jnp.einsum(
        "ipr,rs,jps->ij", tensor, env, jnp.conj(tensor), precision=matmul_precision
    )


def jax_mps_expectation_product_ops(
    tensors: Sequence[Any],
    ops: dict[int, Any],
    matmul_precision: str | None,
) -> Any:
    import jax.numpy as jnp

    env = jnp.ones((1, 1), dtype=_jax_complex_dtype())
    identity = _jax_pauli_matrix("i")
    for wire, tensor in enumerate(tensors):
        op = ops.get(int(wire), identity)
        env = jax_mps_transfer_op(env, tensor, op, matmul_precision)
    return jnp.real(env[0, 0])


def jax_mps_z_values(
    tensors: Sequence[Any], wires: Iterable[int], matmul_precision: str | None
) -> Any:
    import jax.numpy as jnp

    z_op = _jax_pauli_matrix("z")
    values = []
    for wire in wires:
        values.append(
            jax_mps_expectation_product_ops(
                tensors, {int(wire): z_op}, matmul_precision
            )
        )
    return jnp.stack(values) if values else jnp.zeros((0,), dtype=_jax_real_dtype())


def jax_mps_z_sum(
    tensors: Sequence[Any], wires: Iterable[int], matmul_precision: str | None
) -> Any:
    import jax.numpy as jnp

    targets = tuple(int(wire) for wire in wires)
    if not targets:
        return jnp.zeros((), dtype=_jax_real_dtype())
    target_counts: dict[int, int] = {}
    for wire in targets:
        if wire < 0 or wire >= len(tensors):
            raise ValueError("JAX MPS z_sum wire index out of range.")
        target_counts[wire] = target_counts.get(wire, 0) + 1
    env = jnp.ones((1, 1), dtype=_jax_complex_dtype())
    acc = jnp.zeros((1, 1), dtype=_jax_complex_dtype())
    z_op = _jax_pauli_matrix("z")
    for wire, tensor in enumerate(tensors):
        next_acc = jax_mps_transfer_identity(acc, tensor, matmul_precision)
        count = target_counts.get(int(wire), 0)
        if count:
            next_acc = next_acc + count * jax_mps_transfer_op(
                env, tensor, z_op, matmul_precision
            )
        env = jax_mps_transfer_identity(env, tensor, matmul_precision)
        acc = next_acc
    return jnp.real(acc[0, 0])


def jax_mps_pauli_string_expectation(
    tensors: Sequence[Any],
    ops: tuple[tuple[int, str], ...],
    matmul_precision: str | None,
) -> Any:
    op_map: dict[int, Any] = {}
    for wire, name in ops:
        normalized = str(name).lower()
        if normalized != "i":
            op_map[int(wire)] = _jax_pauli_matrix(normalized)
    return jax_mps_expectation_product_ops(tensors, op_map, matmul_precision)
