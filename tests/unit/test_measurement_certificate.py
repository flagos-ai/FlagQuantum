"""Ground the local channel lemma and reject invalid symbolic obligations."""

from itertools import product

import pytest
import torch

from flagquantum import Circuit
from flagquantum.compiler.registers.primitives.and_compute import (
    and_compute_ir,
    and_uncompute_ir,
)
from flagquantum.core.ir import CircuitIR, Instruction
from tests.unit.measurement_certificate import certify_erasure_histories
from tests.unit.measurement_checks import forced_branch

pytestmark = pytest.mark.unit


def test_compute_isometry_and_both_erasure_scalars_on_complete_input_basis() -> None:
    compute, erase = and_compute_ir(0, 1, 2), and_uncompute_ir(0, 1, 2, 0)
    for a, b in product((0, 1), repeat=2):
        state = torch.zeros(8, dtype=torch.complex128)
        state[4 * a + 2 * b] = 1
        expected = torch.zeros_like(state)
        expected[4 * a + 2 * b + (a & b)] = 1
        actual = (
            Circuit(3, inputs=state, dtype=state.dtype)
            .compose(CircuitIR(3, compute))
            .state()
            .flatten()
        )
        torch.testing.assert_close(actual, expected, atol=1e-14, rtol=0)
        for outcome in (0, 1):
            torch.testing.assert_close(
                forced_branch(expected, erase, (outcome,)),
                state / (2**0.5),
                atol=1e-14,
                rtol=0,
            )


def test_rotation_on_unrelated_flag_preserves_all_history_certificate() -> None:
    gates = (
        *and_compute_ir(0, 1, 2),
        Instruction("cry", (2, 3), {"theta": 0.37}),
        *and_uncompute_ir(0, 1, 2, 0),
    )
    assert certify_erasure_histories(gates, n_wires=4, zero_wires=(2,)) == 1
    # The inferred operator must preserve the phase for an entangled input,
    # independently of the symbolic support implementation.
    state = torch.zeros((4, 2, 2), dtype=torch.complex128)
    state[:, 0, :] = torch.tensor(
        [[1, 1j], [2j, -1], [3, 2 - 1j], [1 - 2j, 2]], dtype=state.dtype
    )
    state = state.flatten()
    state /= state.norm()
    reference = forced_branch(state, gates, (0,))
    torch.testing.assert_close(
        forced_branch(state, gates, (1,)), reference, atol=1e-14, rtol=0
    )


@pytest.mark.parametrize("mutation", ("control", "target", "dirty", "feedback", "leak"))
def test_broken_predicates_cleanliness_and_feedback_cannot_be_certified(
    mutation: str,
) -> None:
    compute, erase = and_compute_ir(0, 1, 2), and_uncompute_ir(0, 1, 2, 0)
    prefix = ()
    middle = ()
    suffix = ()
    if mutation == "control":
        middle = (Instruction("x", (0,)),)
    if mutation == "target":
        middle = (Instruction("ry", (2,), {"theta": 0.37}),)
    if mutation == "dirty":
        prefix = (Instruction("x", (2,)),)
    if mutation == "leak":
        suffix = (Instruction("x", (2,)),)
    if mutation == "feedback":
        erase = (
            *erase[:2],
            Instruction("cz", (0, 1), metadata={"conditions": ((0, 0),)}),
            erase[-1],
        )
    with pytest.raises(AssertionError):
        certify_erasure_histories(
            (*prefix, *compute, *middle, *erase, *suffix), n_wires=3, zero_wires=(2,)
        )
