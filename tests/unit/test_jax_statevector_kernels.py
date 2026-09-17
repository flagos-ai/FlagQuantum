"""Numerical unit tests for the JAX statevector kernels."""

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


def _pack_bits(index: int, n_wires: int, wires: tuple[int, ...]) -> int:
    packed = 0
    for position, wire in enumerate(wires):
        bit = (index >> (n_wires - wire - 1)) & 1
        packed |= bit << (len(wires) - position - 1)
    return packed


def test_jax_basis_indices_for_wires_matches_reference():
    import jax.numpy as jnp
    import numpy as np

    from flagquantum.simulation.jax.statevector.kernels import (
        jax_basis_indices_for_wires,
    )

    n_wires = 5
    wires = (0, 2, 4)
    global_indices = jnp.arange(2**n_wires)
    got = jax_basis_indices_for_wires(global_indices, n_wires=n_wires, wires=wires)
    expected = [_pack_bits(idx, n_wires, wires) for idx in range(2**n_wires)]
    assert np.array_equal(np.asarray(got), np.asarray(expected))


def test_jax_rank_mask_for_touched_delta_matches_reference():
    from flagquantum.simulation.jax.statevector.kernels import (
        jax_rank_mask_for_touched_delta,
    )

    sharded_wires = (0, 2, 4, 6)
    touched = (2, 6)
    for delta_code in range(4):
        got = jax_rank_mask_for_touched_delta(sharded_wires, touched, delta_code)
        expected = 0
        for offset, wire in enumerate(touched):
            delta_bit = (delta_code >> (len(touched) - offset - 1)) & 1
            if delta_bit:
                bit_index = sharded_wires.index(wire)
                expected |= 1 << (len(sharded_wires) - bit_index - 1)
        assert got == expected


def test_jax_apply_matrix_to_batched_local_state_matches_einsum():
    import jax.numpy as jnp
    import numpy as np

    from flagquantum.simulation.jax.statevector.kernels import (
        jax_apply_matrix_to_batched_local_state,
    )

    rng = np.random.default_rng(0)
    batch = 3
    n_local_wires = 4
    wires = (1, 3)
    state = rng.standard_normal((batch, 2**n_local_wires)).astype(np.float32)
    matrix = rng.standard_normal((4, 4)).astype(np.float32)

    got = jax_apply_matrix_to_batched_local_state(
        jnp.asarray(state),
        jnp.asarray(matrix),
        wires,
        n_local_wires=n_local_wires,
    )

    remaining = tuple(w for w in range(n_local_wires) if w not in wires)
    permutation = (0,) + tuple(w + 1 for w in wires + remaining)
    inverse = [0] * (n_local_wires + 1)
    for index, axis in enumerate(permutation):
        inverse[axis] = index
    tensor = state.reshape((batch,) + (2,) * n_local_wires).transpose(permutation)
    flat = tensor.reshape(batch, 4, -1)
    expected = np.einsum("ij,bjk->bik", matrix, flat, optimize=True)
    expected = (
        expected.reshape((batch,) + (2,) * n_local_wires)
        .transpose(inverse)
        .reshape(state.shape)
    )
    assert np.allclose(np.asarray(got), expected, atol=1e-5)


def test_jax_sharded_statevector_rank_loss_matches_reference():
    import jax.numpy as jnp
    import numpy as np

    from flagquantum.simulation.jax.statevector.kernels import (
        jax_sharded_statevector_rank_loss,
    )

    rng = np.random.default_rng(1)
    n_wires = 4
    num_indices = 2**n_wires
    global_indices = jnp.arange(num_indices)
    amplitudes = rng.standard_normal((1, num_indices)).astype(np.float32)

    observable_wire = 1
    got = jax_sharded_statevector_rank_loss(
        jnp.asarray(amplitudes),
        global_indices,
        n_wires=n_wires,
        observable="z_sum",
        observable_wires=(observable_wire,),
    )

    expected = 0.0
    for idx in range(num_indices):
        bit = (idx >> (n_wires - 1 - observable_wire)) & 1
        expected += float(amplitudes[0, idx] ** 2) * (1.0 - 2.0 * bit)
    assert np.isclose(float(np.asarray(got)), expected, atol=1e-5)


def test_jax_rank_loss_state_norm_sums_probabilities():
    import jax.numpy as jnp
    import numpy as np

    from flagquantum.simulation.jax.statevector.kernels import (
        jax_sharded_statevector_rank_loss,
    )

    amplitudes = jnp.asarray([[0.5, 0.5j, -0.25, 0.75]], dtype=jnp.complex64)

    got = jax_sharded_statevector_rank_loss(
        amplitudes,
        jnp.arange(4),
        n_wires=2,
        observable="state_norm",
        observable_wires=None,
    )

    expected = float(np.sum(np.abs(np.asarray(amplitudes)) ** 2))
    assert np.isclose(float(np.asarray(got)), expected, atol=1e-6)


def test_jax_apply_matrix_to_batched_local_state_accepts_a_batched_matrix():
    import jax.numpy as jnp
    import numpy as np

    from flagquantum.simulation.jax.statevector.kernels import (
        jax_apply_matrix_to_batched_local_state,
    )

    rng = np.random.default_rng(21)
    state = rng.standard_normal((2, 4)).astype(np.float32)
    # A batched matrix shares the state's batch axis.
    matrices = rng.standard_normal((2, 2, 2)).astype(np.float32)

    got = jax_apply_matrix_to_batched_local_state(
        jnp.asarray(state), jnp.asarray(matrices), (0,), n_local_wires=2
    )

    expected = np.einsum("bij,bjk->bik", matrices, state.reshape(2, 2, 2))
    assert np.asarray(got).shape == (2, 4)
    assert np.allclose(np.asarray(got).reshape(2, 2, 2), expected, atol=1e-5)


def test_jax_apply_matrix_to_batched_local_state_rejects_a_wrong_shape():
    import jax.numpy as jnp

    from flagquantum.simulation.jax.statevector.kernels import (
        jax_apply_matrix_to_batched_local_state,
    )

    with pytest.raises(ValueError, match=r"requires matrix shape \(4, 4\)"):
        jax_apply_matrix_to_batched_local_state(
            jnp.ones((1, 4), dtype=jnp.float32),
            jnp.eye(2, dtype=jnp.float32),
            (0, 1),
            n_local_wires=2,
        )
