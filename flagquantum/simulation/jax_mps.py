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


def parse_zz_z_chain_hamiltonian(
    terms: tuple[tuple[float, tuple[tuple[int, str], ...]], ...],
    n_wires: int,
) -> tuple[dict[str, tuple[float, ...]], tuple[float, ...], float] | None:
    local_coeffs = {
        name: [0.0 for _ in range(int(n_wires))] for name in ("x", "y", "z")
    }
    zz_coeffs = [0.0 for _ in range(max(0, int(n_wires) - 1))]
    constant = 0.0
    for coefficient, ops in terms:
        normalized = tuple(
            (int(wire), str(name).lower())
            for wire, name in ops
            if str(name).lower() != "i"
        )
        if not normalized:
            constant += float(coefficient)
            continue
        if len(normalized) == 1 and normalized[0][1] in local_coeffs:
            wire = normalized[0][0]
            if wire < 0 or wire >= int(n_wires):
                return None
            local_coeffs[normalized[0][1]][wire] += float(coefficient)
            continue
        if len(normalized) == 2 and normalized[0][1] == "z" and normalized[1][1] == "z":
            left = min(normalized[0][0], normalized[1][0])
            right = max(normalized[0][0], normalized[1][0])
            if left < 0 or right >= int(n_wires) or right != left + 1:
                return None
            zz_coeffs[left] += float(coefficient)
            continue
        return None
    return (
        {name: tuple(coefficients) for name, coefficients in local_coeffs.items()},
        tuple(zz_coeffs),
        constant,
    )


def is_zz_z_chain_hamiltonian(
    terms: tuple[tuple[float, tuple[tuple[int, str], ...]], ...],
    n_wires: int,
) -> bool:
    return parse_zz_z_chain_hamiltonian(terms, n_wires) is not None


def jax_mps_single_pauli_with_envs(
    tensor: Any,
    left_env: Any,
    right_env: Any,
    pauli: str,
    matmul_precision: str | None,
) -> Any:
    import jax.numpy as jnp

    op = _jax_pauli_matrix(pauli)
    return jnp.real(
        jnp.einsum(
            "ij,ipr,pq,jqs,rs->",
            left_env,
            jnp.conj(tensor),
            op,
            tensor,
            right_env,
            precision=matmul_precision,
        )
    )


def jax_mps_adjacent_zz_with_envs(
    left_tensor: Any,
    right_tensor: Any,
    left_env: Any,
    right_env: Any,
    matmul_precision: str | None,
) -> Any:
    import jax.numpy as jnp

    z_op = _jax_pauli_matrix("z")
    theta = jnp.einsum(
        "ipr,rqs->ipqs", left_tensor, right_tensor, precision=matmul_precision
    )
    return jnp.real(
        jnp.einsum(
            "ij,ipqr,pa,qb,jabs,rs->",
            left_env,
            jnp.conj(theta),
            z_op,
            z_op,
            theta,
            right_env,
            precision=matmul_precision,
        )
    )


def jax_mps_zz_z_chain_expectation_padded_scan(
    tensors: Sequence[Any],
    parsed: tuple[dict[str, tuple[float, ...]], tuple[float, ...], float],
    matmul_precision: str | None,
) -> Any:
    import jax
    import jax.numpy as jnp

    local_coeffs, zz_coeffs, constant = parsed
    max_dim = max(max(int(tensor.shape[0]), int(tensor.shape[2])) for tensor in tensors)
    padded_tensors = []
    for tensor in tensors:
        left_pad = max_dim - int(tensor.shape[0])
        right_pad = max_dim - int(tensor.shape[2])
        padded_tensors.append(jnp.pad(tensor, ((0, left_pad), (0, 0), (0, right_pad))))
    stacked = jnp.stack(padded_tensors)
    boundary = (
        jnp.zeros((max_dim, max_dim), dtype=_jax_complex_dtype())
        .at[0, 0]
        .set(1.0 + 0.0j)
    )

    def left_step(env: Any, tensor: Any) -> tuple[Any, Any]:
        next_env = jax_mps_transfer_identity(env, tensor, matmul_precision)
        return next_env, next_env

    def right_step(env: Any, tensor: Any) -> tuple[Any, Any]:
        next_env = jax_mps_transfer_identity_right(env, tensor, matmul_precision)
        return next_env, next_env

    _, left_scanned = jax.lax.scan(left_step, boundary, stacked)
    _, right_scanned_reversed = jax.lax.scan(right_step, boundary, stacked[::-1])
    left_envs = jnp.concatenate((boundary[None, :, :], left_scanned), axis=0)
    right_envs = jnp.concatenate(
        (right_scanned_reversed[::-1], boundary[None, :, :]), axis=0
    )

    total = jnp.asarray(constant, dtype=_jax_real_dtype())
    for pauli, coefficients in local_coeffs.items():
        coefficient_array = jnp.asarray(coefficients, dtype=_jax_real_dtype())
        if coefficients and any(
            float(coefficient) != 0.0 for coefficient in coefficients
        ):
            local_values = jax.vmap(
                lambda tensor, left_env, right_env: jax_mps_single_pauli_with_envs(
                    tensor,
                    left_env,
                    right_env,
                    pauli,
                    matmul_precision,
                )
            )(stacked, left_envs[:-1], right_envs[1:])
            total = total + jnp.sum(coefficient_array * local_values)
    zz_coefficient_array = jnp.asarray(zz_coeffs, dtype=_jax_real_dtype())
    if zz_coeffs and any(float(coefficient) != 0.0 for coefficient in zz_coeffs):
        zz_values = jax.vmap(
            lambda left_tensor, right_tensor, left_env, right_env: (
                jax_mps_adjacent_zz_with_envs(
                    left_tensor,
                    right_tensor,
                    left_env,
                    right_env,
                    matmul_precision,
                )
            )
        )(stacked[:-1], stacked[1:], left_envs[:-2], right_envs[2:])
        total = total + jnp.sum(zz_coefficient_array * zz_values)
    return total
