"""Numerical unit tests for the JAX MPS kernels."""

from __future__ import annotations

import importlib.util

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.jax,
    pytest.mark.skipif(
        importlib.util.find_spec("jax") is None, reason="jax is not installed"
    ),
]


def test_jax_mps_to_statevector_zero_state():
    import numpy as np

    from flagquantum.simulation.jax.mps.kernels import (
        jax_mps_initial_open_boundary_tensors,
        jax_mps_to_statevector,
    )

    n_wires = 4
    tensors = jax_mps_initial_open_boundary_tensors(n_wires)
    state = jax_mps_to_statevector(tensors)
    expected = np.zeros(2**n_wires, dtype=np.complex64)
    expected[0] = 1.0
    assert np.allclose(np.asarray(state), expected, atol=1e-5)


def test_jax_mps_z_sum_zero_state_equals_n_wires():
    import numpy as np

    from flagquantum.simulation.jax.mps.kernels import (
        jax_mps_initial_open_boundary_tensors,
        jax_mps_z_sum,
    )

    n_wires = 5
    tensors = jax_mps_initial_open_boundary_tensors(n_wires)
    got = jax_mps_z_sum(tensors, range(n_wires), "highest")
    assert np.isclose(float(np.asarray(got)), float(n_wires), atol=1e-4)


def test_jax_mps_split_pair_reconstructs_exact():
    import jax.numpy as jnp
    import numpy as np

    from flagquantum.simulation.jax.mps.kernels import jax_mps_split_pair

    rng = np.random.default_rng(3)
    left_dim, right_dim = 2, 3
    theta = (
        rng.standard_normal((left_dim, 2, 2, right_dim))
        + 1j * rng.standard_normal((left_dim, 2, 2, right_dim))
    ).astype(np.complex64)

    left, right = jax_mps_split_pair(jnp.asarray(theta), max_bond=None, cutoff=0.0)
    reconstructed = np.einsum("lam,mbr->labr", np.asarray(left), np.asarray(right))
    assert np.allclose(reconstructed, theta, atol=1e-5)
