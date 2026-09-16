"""Numerical unit tests for JAX MPS batched kernels and pullbacks."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.jax,
    pytest.mark.skipif(
        importlib.util.find_spec("jax") is None, reason="jax is not installed"
    ),
]


def test_jax_mps_apply_one_batched_matches_einsum():
    import jax.numpy as jnp

    from flagquantum.simulation.jax.mps.batched import jax_mps_apply_one_batched

    rng = np.random.default_rng(6)
    bsz, left_dim, right_dim = 3, 2, 4
    tensor = (
        rng.standard_normal((bsz, left_dim, 2, right_dim))
        + 1j * rng.standard_normal((bsz, left_dim, 2, right_dim))
    ).astype(np.complex64)
    matrix = (rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))).astype(
        np.complex64
    )

    got = jax_mps_apply_one_batched(jnp.asarray(tensor), jnp.asarray(matrix))
    expected = np.einsum("pq,blqr->blpr", matrix, tensor)
    assert np.allclose(np.asarray(got), expected, atol=1e-5)


def test_jax_mps_split_pair_batched_reconstructs():
    import jax.numpy as jnp

    from flagquantum.simulation.jax.mps.batched import jax_mps_split_pair_batched

    rng = np.random.default_rng(7)
    bsz, left_dim, right_dim = 2, 3, 2
    matrix = (
        rng.standard_normal((bsz, left_dim * 2, 2 * right_dim))
        + 1j * rng.standard_normal((bsz, left_dim * 2, 2 * right_dim))
    ).astype(np.complex64)

    left, right, _info = jax_mps_split_pair_batched(
        jnp.asarray(matrix),
        left_dim=left_dim,
        right_dim=right_dim,
        max_bond=None,
        cutoff=0.0,
    )
    reconstructed = np.einsum("blam,bmcr->blacr", np.asarray(left), np.asarray(right))
    expected = matrix.reshape(bsz, left_dim, 2, 2, right_dim)
    assert np.allclose(reconstructed, expected, atol=1e-5)


def test_jax_mps_owner_local_vjp_matches_grad():
    import jax
    import jax.numpy as jnp

    from flagquantum.simulation.jax.mps.pullbacks import jax_mps_owner_local_vjp

    parameter = jnp.asarray(0.3)
    got = jax_mps_owner_local_vjp(parameter, jnp.asarray(1.0), gate_kind="one_site_ry")

    def local_score(value: jnp.ndarray) -> jnp.ndarray:
        tensor = jnp.stack((jnp.cos(value / 2.0), jnp.sin(value / 2.0)))
        return jnp.real(
            tensor[0] * jnp.conj(tensor[0]) - tensor[1] * jnp.conj(tensor[1])
        )

    expected = jax.grad(local_score)(parameter)
    assert np.isclose(float(np.asarray(got)), float(np.asarray(expected)), atol=1e-5)


def test_jax_mps_canonicalization_pullback_matches_grad():
    import jax
    import jax.numpy as jnp

    from flagquantum.simulation.jax.mps.pullbacks import (
        jax_mps_canonicalization_pullback,
    )

    parameter = jnp.asarray(0.5, dtype=jnp.float64)
    _q, _r, _sv, canonical_grad, truncation_grad = (
        jax_mps_canonicalization_pullback(parameter, include_rank_one_truncation=True)
    )

    weights = jnp.asarray(((1.0, -0.3), (0.2, 0.7)), dtype=jnp.float64)

    def bond_matrix(value: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(
            ((jnp.cos(value), 0.0), (0.0, 0.25 * jnp.sin(value))), dtype=jnp.float64
        )

    def canonicalized_score(value: jnp.ndarray) -> jnp.ndarray:
        matrix = bond_matrix(value)
        q_factor, r_factor = jnp.linalg.qr(matrix)
        return jnp.sum(weights * jnp.matmul(q_factor, r_factor))

    expected = jax.grad(canonicalized_score)(parameter)
    assert np.isclose(float(np.asarray(canonical_grad)), float(np.asarray(expected)), atol=1e-6)
    assert truncation_grad is not None
