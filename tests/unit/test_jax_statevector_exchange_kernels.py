"""Numerical unit tests for the JAX sharded-statevector exchange kernels.

These cover the kernels that move amplitudes between ranks: rank-local position
mapping, the gate-input basis of one exchanged delta, the all-to-all delta
accumulation, the pair-exchange combination, and the shard loss. Each is checked
against an independently written reference, and the all-to-all accumulation is
checked end to end against the gate applied to the full statevector.
"""

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


def _local_positions_reference(
    global_indices,
    *,
    n_wires: int,
    local_wires: tuple[int, ...],
    local_gate_wires: tuple[int, ...],
    local_input_basis: int,
) -> list[int]:
    """Pack each global index down to its rank-local position."""
    positions = []
    for index in global_indices:
        packed = 0
        for wire in local_wires:
            if wire in local_gate_wires:
                position = local_gate_wires.index(wire)
                bit = (local_input_basis >> (len(local_gate_wires) - position - 1)) & 1
            else:
                bit = (index >> (n_wires - wire - 1)) & 1
            packed = (packed << 1) | bit
        positions.append(packed)
    return positions


def _gate_basis_in_reference(
    global_indices,
    *,
    n_wires: int,
    wires: tuple[int, ...],
    touched: tuple[int, ...],
    local_gate_wires: tuple[int, ...],
    delta_code: int,
    local_input_basis: int,
) -> list[int]:
    """Build the gate-input basis of one exchanged contribution."""
    bases = []
    for index in global_indices:
        packed = 0
        for position, wire in enumerate(wires):
            if wire in touched:
                delta_position = touched.index(wire)
                delta_bit = (delta_code >> (len(touched) - delta_position - 1)) & 1
                output_bit = (index >> (n_wires - wire - 1)) & 1
                bit = output_bit ^ delta_bit
            else:
                local_position = local_gate_wires.index(wire)
                bit = (
                    local_input_basis >> (len(local_gate_wires) - local_position - 1)
                ) & 1
            packed |= bit << (len(wires) - position - 1)
        bases.append(packed)
    return bases


def _shard_indices(
    n_wires: int, sharded_wires: tuple[int, ...], rank: int
) -> list[int]:
    """Global indices a rank owns when the sharded wires encode its rank."""
    indices = []
    for index in range(2**n_wires):
        bits = 0
        for position, wire in enumerate(sharded_wires):
            bit = (index >> (n_wires - wire - 1)) & 1
            bits |= bit << (len(sharded_wires) - position - 1)
        if bits == rank:
            indices.append(index)
    return indices


def test_jax_initial_statevector_shard_is_the_zero_state():
    import jax.numpy as jnp
    import numpy as np

    from flagquantum.simulation.jax.statevector.kernels import (
        jax_initial_statevector_shard,
    )

    shard = jax_initial_statevector_shard(
        jnp.asarray([0, 1, 2, 3]), batch_size=2, dtype=jnp.complex64
    )

    got = np.asarray(shard)
    assert got.shape == (2, 4)
    assert got.dtype == np.complex64
    expected = np.zeros((2, 4), dtype=np.complex64)
    expected[:, 0] = 1.0
    assert np.allclose(got, expected)


def test_jax_local_positions_for_gate_input_matches_reference():
    import jax.numpy as jnp
    import numpy as np

    from flagquantum.simulation.jax.statevector.kernels import (
        jax_local_positions_for_gate_input,
    )

    n_wires = 4
    global_indices = jnp.arange(2**n_wires)
    local_wires = (0, 2)
    local_gate_wires = (0,)

    for local_input_basis in range(2 ** len(local_gate_wires)):
        got = jax_local_positions_for_gate_input(
            global_indices,
            n_wires=n_wires,
            local_wires=local_wires,
            local_gate_wires=local_gate_wires,
            local_input_basis=local_input_basis,
        )
        expected = _local_positions_reference(
            range(2**n_wires),
            n_wires=n_wires,
            local_wires=local_wires,
            local_gate_wires=local_gate_wires,
            local_input_basis=local_input_basis,
        )
        assert np.array_equal(np.asarray(got), np.asarray(expected))


def test_jax_gate_basis_in_for_delta_and_local_input_matches_reference():
    import jax.numpy as jnp
    import numpy as np

    from flagquantum.simulation.jax.statevector.kernels import (
        jax_gate_basis_in_for_delta_and_local_input,
    )

    n_wires = 3
    wires = (0, 2)
    touched = (2,)
    local_gate_wires = (0,)
    global_indices = jnp.arange(2**n_wires)

    for delta_code in range(2 ** len(touched)):
        for local_input_basis in range(2 ** len(local_gate_wires)):
            got = jax_gate_basis_in_for_delta_and_local_input(
                global_indices,
                n_wires=n_wires,
                wires=wires,
                touched_sharded_wires=touched,
                local_gate_wires=local_gate_wires,
                delta_code=delta_code,
                local_input_basis=local_input_basis,
            )
            expected = _gate_basis_in_reference(
                range(2**n_wires),
                n_wires=n_wires,
                wires=wires,
                touched=touched,
                local_gate_wires=local_gate_wires,
                delta_code=delta_code,
                local_input_basis=local_input_basis,
            )
            assert np.array_equal(np.asarray(got), np.asarray(expected))


def test_jax_accumulate_all_to_all_delta_reproduces_a_dense_two_qubit_gate():
    """The exchange loop must equal the gate applied to the full statevector.

    This mirrors the runtime caller: one accumulation per delta code, with the
    source amplitudes taken from the rank selected by the rank mask.
    """
    import jax.numpy as jnp
    import numpy as np

    from flagquantum.simulation.jax.statevector.kernels import (
        jax_accumulate_all_to_all_statevector_delta,
        jax_basis_indices_for_wires,
        jax_rank_mask_for_touched_delta,
    )

    n_wires = 2
    sharded_wires = (1,)
    wires = (0, 1)
    touched = tuple(wire for wire in wires if wire in sharded_wires)
    local_wires = tuple(wire for wire in range(n_wires) if wire not in sharded_wires)
    local_gate_wires = tuple(wire for wire in wires if wire not in sharded_wires)
    world_size = 2

    rng = np.random.default_rng(11)
    psi = (
        rng.standard_normal(2**n_wires) + 1j * rng.standard_normal(2**n_wires)
    ).astype(np.complex64)
    matrix = (rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))).astype(
        np.complex64
    )

    shards = {}
    for rank in range(world_size):
        indices = _shard_indices(n_wires, sharded_wires, rank)
        shards[rank] = (
            jnp.asarray(indices),
            jnp.asarray([psi[index] for index in indices]).reshape(1, -1),
            indices,
        )

    for rank in range(world_size):
        global_indices, amplitudes, indices = shards[rank]
        basis_out = jax_basis_indices_for_wires(
            global_indices, n_wires=n_wires, wires=wires
        )
        updated = jnp.zeros_like(amplitudes)
        for delta_code in range(2 ** len(touched)):
            rank_mask = jax_rank_mask_for_touched_delta(
                sharded_wires, touched, delta_code
            )
            updated = jax_accumulate_all_to_all_statevector_delta(
                updated,
                shards[rank ^ rank_mask][1],
                global_indices,
                jnp.asarray(matrix),
                basis_out,
                n_wires=n_wires,
                wires=wires,
                touched_sharded_wires=touched,
                local_wires=local_wires,
                local_gate_wires=local_gate_wires,
                delta_code=delta_code,
            )
        expected = np.asarray(matrix) @ psi
        got = np.asarray(updated).reshape(-1)
        assert np.allclose(
            got, np.asarray([expected[index] for index in indices]), atol=1e-4
        )
        if len(touched) > 1 or rank_mask:
            # The partner contributions must actually be exchanged, or the
            # comparison above would pass on a local-only computation.
            local_only = jnp.zeros_like(amplitudes)
            local_only = jax_accumulate_all_to_all_statevector_delta(
                local_only,
                amplitudes,
                global_indices,
                jnp.asarray(matrix),
                basis_out,
                n_wires=n_wires,
                wires=wires,
                touched_sharded_wires=touched,
                local_wires=local_wires,
                local_gate_wires=local_gate_wires,
                delta_code=0,
            )
            assert not np.allclose(got, np.asarray(local_only).reshape(-1), atol=1e-4)


def test_jax_accumulate_all_to_all_delta_accepts_a_batched_matrix():
    """A batched gate matrix indexes on its trailing axes."""
    import jax.numpy as jnp
    import numpy as np

    from flagquantum.simulation.jax.statevector.kernels import (
        jax_accumulate_all_to_all_statevector_delta,
        jax_basis_indices_for_wires,
    )

    n_wires = 2
    wires = (0, 1)
    touched = (1,)
    local_wires = (0,)
    local_gate_wires = (0,)
    global_indices = jnp.asarray([1, 3])

    rng = np.random.default_rng(15)
    source = jnp.asarray(
        (rng.standard_normal(2) + 1j * rng.standard_normal(2)).astype(np.complex64)
    ).reshape(1, -1)
    basis_out = jax_basis_indices_for_wires(
        global_indices, n_wires=n_wires, wires=wires
    )
    plain = (rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))).astype(
        np.complex64
    )

    got = jax_accumulate_all_to_all_statevector_delta(
        jnp.zeros_like(source),
        source,
        global_indices,
        jnp.asarray(plain.reshape(1, 4, 4)),
        basis_out,
        n_wires=n_wires,
        wires=wires,
        touched_sharded_wires=touched,
        local_wires=local_wires,
        local_gate_wires=local_gate_wires,
        delta_code=0,
    )

    expected = jax_accumulate_all_to_all_statevector_delta(
        jnp.zeros_like(source),
        source,
        global_indices,
        jnp.asarray(plain),
        basis_out,
        n_wires=n_wires,
        wires=wires,
        touched_sharded_wires=touched,
        local_wires=local_wires,
        local_gate_wires=local_gate_wires,
        delta_code=0,
    )
    assert np.allclose(np.asarray(got), np.asarray(expected), atol=1e-6)


def test_jax_apply_local_statevector_gate_matches_a_dense_application():
    import jax.numpy as jnp
    import numpy as np

    from flagquantum.simulation.jax.statevector.kernels import (
        jax_apply_local_statevector_gate,
    )

    n_wires = 2
    sharded_wires = (1,)
    rank = 1
    indices = _shard_indices(n_wires, sharded_wires, rank)

    rng = np.random.default_rng(12)
    psi = (
        rng.standard_normal(2**n_wires) + 1j * rng.standard_normal(2**n_wires)
    ).astype(np.complex64)
    matrix = (rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))).astype(
        np.complex64
    )
    amplitudes = jnp.asarray([psi[index] for index in indices]).reshape(1, -1)

    got = jax_apply_local_statevector_gate(
        amplitudes,
        jnp.asarray(indices),
        jnp.asarray(matrix),
        (0,),
        n_wires=n_wires,
        sharded_wires=sharded_wires,
        diagonal=False,
    )

    # The rank-local axis is ordered by the local (unsharded) wires ascending.
    dense = psi.reshape(2, 2)
    expected = np.einsum("ij,bj->bi", matrix, dense[:, rank].reshape(1, -1))
    assert np.allclose(np.asarray(got).reshape(-1), expected.reshape(-1), atol=1e-5)


def test_jax_apply_local_statevector_gate_diagonal_matches_dense():
    import jax.numpy as jnp
    import numpy as np

    from flagquantum.simulation.jax.statevector.kernels import (
        jax_apply_local_statevector_gate,
    )

    n_wires = 3
    sharded_wires = (1,)
    wires = (0, 2)
    indices = _shard_indices(n_wires, sharded_wires, 1)

    amplitudes = jnp.asarray(
        [1.0 + 0.5j * offset for offset in range(len(indices))]
    ).reshape(1, -1)
    diagonal_values = (0.2 + 1j * np.arange(4)).astype(np.complex64)

    got = jax_apply_local_statevector_gate(
        amplitudes,
        jnp.asarray(indices),
        jnp.asarray(np.diag(diagonal_values)),
        wires,
        n_wires=n_wires,
        sharded_wires=sharded_wires,
        diagonal=True,
    )

    expected = [
        complex(amplitudes[0, offset])
        * diagonal_values[_pack_bits(index, n_wires, wires)]
        for offset, index in enumerate(indices)
    ]
    assert np.allclose(np.asarray(got).reshape(-1), np.asarray(expected), atol=1e-6)


def test_jax_apply_local_statevector_gate_rejects_a_sharded_touched_wire():
    import jax.numpy as jnp

    from flagquantum.simulation.jax.statevector.kernels import (
        jax_apply_local_statevector_gate,
    )

    with pytest.raises(RuntimeError, match="touches sharded wires"):
        jax_apply_local_statevector_gate(
            jnp.ones((1, 2), dtype=jnp.complex64),
            jnp.asarray([1, 3]),
            jnp.eye(2, dtype=jnp.complex64),
            (1,),
            n_wires=2,
            sharded_wires=(1,),
            diagonal=False,
            gate_name="h",
        )


def test_jax_combine_pair_exchanged_statevector_matches_a_dense_application():
    import jax.numpy as jnp
    import numpy as np

    from flagquantum.simulation.jax.statevector.kernels import (
        jax_combine_pair_exchanged_statevector,
    )

    n_wires = 2
    wire = 1
    sharded_wires = (wire,)
    rank = 1
    indices = _shard_indices(n_wires, sharded_wires, rank)
    partner_indices = _shard_indices(n_wires, sharded_wires, rank ^ 1)

    rng = np.random.default_rng(14)
    psi = (
        rng.standard_normal(2**n_wires) + 1j * rng.standard_normal(2**n_wires)
    ).astype(np.complex64)
    matrix = np.asarray([[0.6 + 0.1j, -0.2j], [0.3, 0.8 - 0.4j]], dtype=np.complex64)
    own = jnp.asarray([[psi[index] for index in indices]])
    partner = jnp.asarray([[psi[index] for index in partner_indices]])

    got = jax_combine_pair_exchanged_statevector(
        own,
        partner,
        jnp.asarray(indices),
        jnp.asarray(matrix),
        n_wires=n_wires,
        wire=wire,
    )

    # Rank `rank` owns the amplitudes whose sharded bit equals `rank`, so the
    # output at that bit takes the partner's amplitudes as the opposite input.
    dense = psi.reshape(2, 2)
    expected = (
        matrix[rank, 0] * dense[:, rank ^ 1] + matrix[rank, rank] * dense[:, rank]
    )
    assert np.allclose(np.asarray(got)[0], expected, atol=1e-6)


@pytest.mark.parametrize("observable", ["z", "z_sum", "state_norm"])
def test_jax_sharded_statevector_loss_sums_over_shards(observable: str):
    """`z` and `z_sum` both accumulate a per-wire contribution additively.

    The kernel sums one Z expectation per listed wire rather than the product
    over wires; the two names therefore select the same computation here.
    """
    import jax.numpy as jnp
    import numpy as np

    from flagquantum.simulation.jax.statevector.kernels import (
        jax_sharded_statevector_loss,
    )

    n_wires = 3
    sharded_wires = (0,)
    wires = (1, 2)
    shards = [
        [complex(0.5 + 0.1 * index + 0.2j * offset) for offset in range(4)]
        for index in range(2)
    ]

    got = jax_sharded_statevector_loss(
        [jnp.asarray(shard).reshape(1, -1) for shard in shards],
        [
            jnp.asarray(_shard_indices(n_wires, sharded_wires, rank))
            for rank in range(2)
        ],
        n_wires=n_wires,
        observable=observable,
        observable_wires=wires,
    )

    expected = 0.0
    for rank, shard in enumerate(shards):
        for offset, index in enumerate(_shard_indices(n_wires, sharded_wires, rank)):
            probability = abs(shard[offset]) ** 2
            if observable == "state_norm":
                expected += probability
                continue
            for wire in wires:
                sign = 1.0 - 2.0 * ((index >> (n_wires - wire - 1)) & 1)
                expected += probability * sign
    assert np.isclose(float(np.asarray(got)), expected, atol=1e-5)


def test_jax_sharded_statevector_loss_defaults_to_every_wire():
    import jax.numpy as jnp
    import numpy as np

    from flagquantum.simulation.jax.statevector.kernels import (
        jax_sharded_statevector_loss,
    )

    got = jax_sharded_statevector_loss(
        [jnp.asarray([[0.5, 0.5, 0.5, 0.5]], dtype=jnp.complex64)],
        [jnp.arange(4)],
        n_wires=2,
        observable="z",
        observable_wires=None,
    )

    # Uniform amplitudes cancel every Z sign, so the sum is zero.
    assert np.isclose(float(np.asarray(got)), 0.0, atol=1e-6)


def test_jax_sharded_statevector_loss_rejects_unsupported_inputs():
    import jax.numpy as jnp

    from flagquantum.simulation.jax.statevector.kernels import (
        jax_sharded_statevector_loss,
    )

    amplitudes = jnp.asarray([[0.5, 0.5]], dtype=jnp.complex64)
    indices = jnp.arange(2)

    with pytest.raises(ValueError, match="equal lengths"):
        jax_sharded_statevector_loss(
            [amplitudes],
            [indices, indices],
            n_wires=1,
            observable="z",
            observable_wires=(0,),
        )
    with pytest.raises(ValueError, match="observable='z_sum'"):
        jax_sharded_statevector_loss(
            [amplitudes],
            [indices],
            n_wires=1,
            observable="pauli_x",
            observable_wires=(0,),
        )


def test_jax_sharded_statevector_rank_loss_rejects_unsupported_observable():
    import jax.numpy as jnp

    from flagquantum.simulation.jax.statevector.kernels import (
        jax_sharded_statevector_rank_loss,
    )

    with pytest.raises(ValueError, match="observable='z_sum'"):
        jax_sharded_statevector_rank_loss(
            jnp.asarray([[0.5, 0.5]], dtype=jnp.complex64),
            jnp.arange(2),
            n_wires=1,
            observable="pauli_x",
            observable_wires=(0,),
        )
