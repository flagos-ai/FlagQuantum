"""Numerical unit tests for the JAX tensor-network local contraction."""

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


def test_contract_nodes_greedy_matches_einsum():
    import jax.numpy as jnp

    from flagquantum.simulation.jax.tensor_network.contraction import (
        contract_nodes_greedy,
    )
    from flagquantum.simulation.jax.tensor_network.models import JAXTensorNetworkNode

    rng = np.random.default_rng(8)
    a = (rng.standard_normal((3, 4)) + 1j * rng.standard_normal((3, 4))).astype(
        np.complex64
    )
    b = (rng.standard_normal((4, 5)) + 1j * rng.standard_normal((4, 5))).astype(
        np.complex64
    )

    nodes = [
        JAXTensorNetworkNode(tensor=jnp.asarray(a), labels=(0, 1)),
        JAXTensorNetworkNode(tensor=jnp.asarray(b), labels=(1, 2)),
    ]
    got = contract_nodes_greedy(nodes, (0, 2))
    expected = np.einsum("ij,jk->ik", a, b)
    assert np.allclose(np.asarray(got), expected, atol=1e-4)


def test_zero_for_output_shape_and_value():
    import jax.numpy as jnp

    from flagquantum.simulation.jax.tensor_network.contraction import zero_for_output
    from flagquantum.simulation.jax.tensor_network.models import JAXTensorNetworkNode

    tensor = jnp.zeros((2, 3), dtype=jnp.complex64)
    nodes = [JAXTensorNetworkNode(tensor=tensor, labels=(0, 1))]
    got = zero_for_output(nodes, (0, 1))
    assert tuple(got.shape) == (2, 3)
    assert float(jnp.max(jnp.abs(got))) == 0.0
