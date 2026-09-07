"""Dependency-light JAX MPS tensor operations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .jax.primitives import (
    _jax_complex_dtype,
    _jax_pauli_matrix,
    _jax_real_dtype,
    _jax_swap,
)


def jax_mps_initial_padded_stack(n_wires: int, bond_dim: int) -> Any:
    import jax.numpy as jnp

    tensor = jnp.zeros((int(bond_dim), 2, int(bond_dim)), dtype=_jax_complex_dtype())
    tensor = tensor.at[0, 0, 0].set(1.0 + 0.0j)
    return jnp.broadcast_to(
        tensor, (int(n_wires), int(bond_dim), 2, int(bond_dim))
    ).copy()


def jax_mps_initial_open_boundary_tensors(n_wires: int) -> list[Any]:
    """Create an open-boundary MPS representation of the all-zero state."""

    import jax.numpy as jnp

    tensor = jnp.zeros((1, 2, 1), dtype=_jax_complex_dtype())
    tensor = tensor.at[0, 0, 0].set(1.0 + 0.0j)
    return [tensor for _ in range(int(n_wires))]


def jax_mps_site_transfer(environment: Any, tensor: Any) -> Any:
    """Transfer an MPS norm environment across one site."""

    import jax.numpy as jnp

    return jnp.einsum("bij,bipr,bjps->brs", environment, jnp.conj(tensor), tensor)


def jax_mps_site_observable_transfer(
    environment: Any,
    tensor: Any,
    operator: Any,
) -> Any:
    """Transfer an MPS observable environment across one site."""

    import jax.numpy as jnp

    return jnp.einsum(
        "bij,bipr,pq,bjqs->brs",
        environment,
        jnp.conj(tensor),
        operator,
        tensor,
    )


def jax_sharded_mps_z_sum(
    rank_tensors: Mapping[int, Mapping[int, Any]],
    *,
    n_wires: int,
    batch_size: int,
    dtype: Any,
    observable_wires: Sequence[int] | None,
) -> Any:
    """Evaluate a Z sum after collecting rank-owned MPS site tensors."""

    import jax.numpy as jnp

    z_operator = _jax_pauli_matrix("z", dtype=dtype)
    environment = jnp.ones((int(batch_size), 1, 1), dtype=dtype)
    accumulated = jnp.zeros((int(batch_size), 1, 1), dtype=dtype)
    target_wires = (
        set(range(int(n_wires)))
        if observable_wires is None
        else {int(wire) for wire in observable_wires}
    )
    tensors_by_wire: dict[int, Any] = {}
    for tensors in rank_tensors.values():
        tensors_by_wire.update({int(wire): tensor for wire, tensor in tensors.items()})
    missing = tuple(wire for wire in range(int(n_wires)) if wire not in tensors_by_wire)
    if missing:
        raise RuntimeError(f"Missing MPS site tensors for wires {missing}.")
    for wire in range(int(n_wires)):
        tensor = tensors_by_wire[wire]
        next_accumulated = jax_mps_site_transfer(accumulated, tensor)
        if wire in target_wires:
            next_accumulated = next_accumulated + jax_mps_site_observable_transfer(
                environment,
                tensor,
                z_operator,
            )
        environment = jax_mps_site_transfer(environment, tensor)
        accumulated = next_accumulated
    return jnp.real(jnp.sum(accumulated.reshape((int(batch_size), -1))[:, 0]))


def jax_mps_project_open_boundaries(tensors: Any) -> Any:
    import jax.numpy as jnp

    left_mask = (
        jnp.zeros((tensors.shape[1], 1, 1), dtype=tensors.dtype)
        .at[0, 0, 0]
        .set(1.0 + 0.0j)
    )
    right_mask = (
        jnp.zeros((1, 1, tensors.shape[3]), dtype=tensors.dtype)
        .at[0, 0, 0]
        .set(1.0 + 0.0j)
    )
    tensors = tensors.at[0].set(tensors[0] * left_mask)
    tensors = tensors.at[-1].set(tensors[-1] * right_mask)
    return tensors


def jax_mps_apply_local_stack(
    tensors: Any, matrices: Any, matmul_precision: str | None
) -> Any:
    import jax.numpy as jnp

    return jnp.einsum("wpq,wlqr->wlpr", matrices, tensors, precision=matmul_precision)


def jax_mps_apply_two_adjacent_active(
    left: Any,
    right: Any,
    matrix: Any,
    *,
    max_bond: int,
    left_active: int,
    shared_active: int,
    right_active: int,
    out_rank: int,
    matmul_precision: str | None,
) -> tuple[Any, Any]:
    import jax.numpy as jnp

    left = left[: int(left_active), :, : int(shared_active)]
    right = right[: int(shared_active), :, : int(right_active)]
    theta = jnp.einsum("lsm,mtr->lstr", left, right, precision=matmul_precision)
    theta = theta.reshape(int(left_active), 4, int(right_active))
    flat = theta.transpose(1, 0, 2).reshape(4, -1)
    applied = jnp.matmul(matrix, flat, precision=matmul_precision)
    theta = applied.reshape(4, int(left_active), int(right_active)).transpose(1, 0, 2)
    theta = theta.reshape(int(left_active), 2, 2, int(right_active))
    next_left, next_right = jax_mps_split_pair(
        theta, max_bond=int(out_rank), cutoff=0.0
    )
    next_left = jnp.pad(
        next_left,
        (
            (0, int(max_bond) - int(left_active)),
            (0, 0),
            (0, int(max_bond) - int(out_rank)),
        ),
    )
    next_right = jnp.pad(
        next_right,
        (
            (0, int(max_bond) - int(out_rank)),
            (0, 0),
            (0, int(max_bond) - int(right_active)),
        ),
    )
    return next_left, next_right


def jax_mps_apply_adjacent_chain_scan(
    tensors: Any,
    matrix: Any,
    *,
    max_bond: int,
    layer_index: int,
    matmul_precision: str | None,
) -> Any:
    import jax
    import jax.numpy as jnp

    n_wires = int(tensors.shape[0])
    if n_wires <= 1:
        return tensors

    def rank_before(bond: int) -> int:
        if int(layer_index) <= 0:
            return 1
        return min(
            int(max_bond),
            2 ** min(int(layer_index), int(bond) + 1, n_wires - 1 - int(bond)),
        )

    def rank_after(bond: int) -> int:
        return min(
            int(max_bond),
            2 ** min(int(layer_index) + 1, int(bond) + 1, n_wires - 1 - int(bond)),
        )

    def spec(wire: int) -> tuple[int, int, int, int]:
        left_active = 1 if int(wire) == 0 else rank_after(int(wire) - 1)
        shared_active = rank_before(int(wire))
        right_active = 1 if int(wire) == n_wires - 2 else rank_before(int(wire) + 1)
        out_rank = rank_after(int(wire))
        return left_active, shared_active, right_active, out_rank

    def apply_one(
        left_tensor: Any, right_tensor: Any, current_spec: tuple[int, int, int, int]
    ) -> tuple[Any, Any]:
        left_active, shared_active, right_active, out_rank = current_spec
        return jax_mps_apply_two_adjacent_active(
            left_tensor,
            right_tensor,
            matrix,
            max_bond=int(max_bond),
            left_active=left_active,
            shared_active=shared_active,
            right_active=right_active,
            out_rank=out_rank,
            matmul_precision=matmul_precision,
        )

    pieces = []
    carry = tensors[0]
    wire = 0
    while wire < n_wires - 1:
        current_spec = spec(wire)
        run_end = wire + 1
        if current_spec[0] == current_spec[3]:
            while run_end < n_wires - 1 and spec(run_end) == current_spec:
                run_end += 1
        if run_end - wire > 1:

            def step(left_tensor: Any, right_tensor: Any) -> tuple[Any, Any]:
                next_left, next_right = apply_one(
                    left_tensor, right_tensor, current_spec
                )
                return next_right, next_left

            carry, completed = jax.lax.scan(
                step, carry, tensors[wire + 1 : run_end + 1]
            )
            pieces.append(completed)
            wire = run_end
            continue

        next_left, carry = apply_one(carry, tensors[wire + 1], current_spec)
        pieces.append(next_left[None, :, :, :])
        wire += 1

    pieces.append(carry[None, :, :, :])
    return jnp.concatenate(pieces, axis=0)


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


def jax_mps_hamiltonian_expectation(
    tensors: Sequence[Any],
    terms: tuple[tuple[float, tuple[tuple[int, str], ...]], ...],
    matmul_precision: str | None,
) -> Any:
    import jax.numpy as jnp

    parsed = parse_zz_z_chain_hamiltonian(terms, len(tensors))
    if parsed is not None:
        return jax_mps_zz_z_chain_expectation_padded_scan(
            tensors, parsed, matmul_precision
        )

    total = jnp.zeros((), dtype=_jax_real_dtype())
    for coefficient, ops in terms:
        total = total + jnp.asarray(
            coefficient, dtype=_jax_real_dtype()
        ) * jax_mps_pauli_string_expectation(
            tensors,
            ops,
            matmul_precision,
        )
    return total
