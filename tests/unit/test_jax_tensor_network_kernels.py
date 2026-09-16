"""Numerical unit tests for the JAX tensor-network kernels."""

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


def test_jax_contract_nodes_greedy_matches_einsum():
    import jax.numpy as jnp

    from flagquantum.simulation.jax.tensor_network.kernels import (
        jax_contract_nodes_greedy,
    )

    rng = np.random.default_rng(4)
    a = (rng.standard_normal((3, 4)) + 1j * rng.standard_normal((3, 4))).astype(
        np.complex64
    )
    b = (rng.standard_normal((4, 5)) + 1j * rng.standard_normal((4, 5))).astype(
        np.complex64
    )

    nodes = [(jnp.asarray(a), (0, 1)), (jnp.asarray(b), (1, 2))]
    got = jax_contract_nodes_greedy(nodes, (0, 2), "highest")
    expected = np.einsum("ij,jk->ik", a, b)
    assert np.allclose(np.asarray(got), expected, atol=1e-4)


def test_jax_tensor_network_loss_state_norm_matches_reference():
    import jax.numpy as jnp

    from flagquantum.simulation.jax.tensor_network.kernels import (
        jax_tensor_network_loss_from_output,
    )

    rng = np.random.default_rng(5)
    n_wires = 3
    bsz = 2
    state = (
        rng.standard_normal((bsz, 2**n_wires))
        + 1j * rng.standard_normal((bsz, 2**n_wires))
    ).astype(np.complex64)

    got = jax_tensor_network_loss_from_output(
        jnp.asarray(state),
        n_wires=n_wires,
        bsz=bsz,
        observable="state_norm",
        observable_wires=None,
    )
    expected = np.sum(np.conj(state) * state)
    assert np.isclose(float(np.asarray(got)), float(expected.real), atol=1e-3)
