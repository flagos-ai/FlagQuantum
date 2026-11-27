"""Exercise JAX shard numerics through the existing Runtime record consumer."""

from types import SimpleNamespace

import pytest

from flagquantum.runtime.executors.jax import array_conversions
from flagquantum.runtime.executors.jax.mps import execution as mps_execution
from flagquantum.runtime.executors.jax.statevector import execution
from flagquantum.runtime.executors.jax.statevector.records import (
    JAXStatevectorShardState,
)
from flagquantum.runtime.executors.jax.tensor_network import execution as tn_execution
from flagquantum.runtime.executors.jax.tensor_network import gradients as tn_gradients
from flagquantum.simulation.statevector.jax import kernels

pytestmark = pytest.mark.unit


@pytest.fixture
def engines():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    np = pytest.importorskip("numpy")
    original_x64 = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        yield jax, jnp, np
    finally:
        jax.config.update("jax_enable_x64", original_x64)


def test_representation_conversions_have_direct_owners():
    assert not hasattr(array_conversions, "_apply_gate_to_jax_shards")
    assert not hasattr(array_conversions, "_jax_split_record")
    assert not hasattr(array_conversions, "_jax_nodes_from_torch_nodes")
    assert (
        tn_gradients._jax_nodes_from_torch_nodes
        is tn_execution._jax_nodes_from_torch_nodes
    )
    assert (
        mps_execution._jax_split_record(
            {"method": "qr"}, bond=0, max_bond=None, cutoff=0.0
        )
        is None
    )


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
@pytest.mark.parametrize(
    "groups", [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((3, 2, 1, 0),)]
)
@pytest.mark.parametrize("diagonal", [False, True])
def test_shard_gate_matches_dense_forward_and_gradient(
    engines, dtype, groups, diagonal
):
    jax, jnp, np = engines
    full = (np.arange(8).reshape(2, 4) + 1j * np.arange(8, 16).reshape(2, 4)).astype(
        dtype
    ) / 8
    shards = tuple(
        JAXStatevectorShardState(
            rank=index,
            shard=object(),
            amplitudes=jnp.asarray(full[:, group]),
            global_indices=jnp.asarray(group),
            global_indices_tuple=group,
        )
        for index, group in enumerate(groups)
    )

    def run(angle):
        if diagonal:
            matrix = jnp.diag(jnp.exp(1j * angle * jnp.arange(4))).astype(dtype)
        else:
            grid = jnp.arange(16).reshape(4, 4)
            matrix = (jnp.cos(angle + grid) + 1j * jnp.sin(angle - grid)).astype(dtype)
        return execution._apply_gate_to_jax_shards(
            shards, matrix, (0, 1), plan=SimpleNamespace(n_qubits=2), diagonal=diagonal
        )

    angle = 0.37
    if diagonal:
        diagonal_values = np.exp(1j * angle * np.arange(4))
        matrix = np.diag(diagonal_values).astype(dtype)
        derivative = np.diag(1j * np.arange(4) * diagonal_values).astype(dtype)
    else:
        grid = np.arange(16).reshape(4, 4)
        matrix = (np.cos(angle + grid) + 1j * np.sin(angle - grid)).astype(dtype)
        derivative = (-np.sin(angle + grid) + 1j * np.cos(angle - grid)).astype(dtype)
    expected = np.einsum("ij,bj->bi", matrix, full)
    result = run(jnp.asarray(angle))
    for before, after in zip(shards, result, strict=True):
        np.testing.assert_allclose(
            after.amplitudes,
            expected[:, before.global_indices_tuple],
            rtol=2e-5,
            atol=2e-5,
        )
        np.testing.assert_array_equal(
            before.amplitudes, full[:, before.global_indices_tuple]
        )
        assert after.rank == before.rank
        assert after.shard is before.shard
        assert after.global_indices is before.global_indices
        assert after.global_indices_tuple is before.global_indices_tuple
    gradient = jax.grad(
        lambda value: sum(jnp.real(jnp.sum(shard.amplitudes)) for shard in run(value))
    )(jnp.asarray(angle))
    expected_gradient = np.real(np.einsum("ij,bj->bi", derivative, full)).sum()
    np.testing.assert_allclose(gradient, expected_gradient, rtol=2e-5, atol=2e-5)


def test_shard_gate_preserves_empty_shape_and_missing_basis_refusals(engines):
    _, jnp, _ = engines
    plan = SimpleNamespace(n_qubits=2)
    assert (
        execution._apply_gate_to_jax_shards(
            (), jnp.ones((3, 3)), (0, 1), plan=plan, diagonal=False
        )
        == ()
    )
    shard = JAXStatevectorShardState(
        0, object(), jnp.ones((1, 3)), jnp.arange(3), (0, 1, 2)
    )
    with pytest.raises(ValueError, match="requires matrix shape"):
        execution._apply_gate_to_jax_shards(
            (shard,), jnp.ones((3, 3)), (0, 1), plan=plan, diagonal=False
        )
    with pytest.raises(ValueError, match=r"missing amplitudes \(3,\)"):
        execution._apply_gate_to_jax_shards(
            (shard,), jnp.eye(4), (0, 1), plan=plan, diagonal=False
        )


def test_shard_numerical_implementation_can_be_replaced_without_consumer_changes(
    engines, monkeypatch
):
    _, jnp, np = engines
    indices = jnp.arange(2)
    original = jnp.asarray([[1.0, 0.0]], dtype="complex64")
    shard = JAXStatevectorShardState(0, object(), original, indices, (0, 1))
    matrix = jnp.eye(2)
    calls = []

    def substitute(
        amplitudes,
        global_indices,
        global_index_tuples,
        gate,
        qubits,
        *,
        n_qubits,
        diagonal,
    ):
        calls.append(
            (
                amplitudes,
                global_indices,
                global_index_tuples,
                gate,
                qubits,
                n_qubits,
                diagonal,
            )
        )
        return tuple(array + 0.25 for array in amplitudes)

    monkeypatch.setattr(kernels, "jax_apply_gate_to_shard_arrays", substitute)
    result = execution._apply_gate_to_jax_shards(
        (shard,), matrix, (0,), plan=SimpleNamespace(n_qubits=1), diagonal=True
    )
    assert len(calls) == 1
    assert calls[0][0][0] is original
    assert calls[0][1][0] is indices
    assert calls[0][2][0] is shard.global_indices_tuple
    assert calls[0][3] is matrix
    assert calls[0][4:] == ((0,), 1, True)
    np.testing.assert_array_equal(result[0].amplitudes, original + 0.25)
    assert result[0].shard is shard.shard
    assert result[0].global_indices is indices
    assert result[0].global_indices_tuple is shard.global_indices_tuple
