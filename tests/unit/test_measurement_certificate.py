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
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch

pytestmark = pytest.mark.unit


def test_lookup_erasure_phase_certificate_matches_every_complex_branch() -> None:
    gates = (
        Instruction("cx", (0, 1)),
        Instruction("h", (1,)),
        Instruction("measure", (1,), metadata={"is_dynamic": True, "classical_bit": 7}),
        Instruction("x", (1,), metadata={"conditions": ((7, 1),)}),
        Instruction("z", (0,), metadata={"conditions": ((7, 1),)}),
    )
    assert certified_and_channel_map(gates, ((0, 0), (1, 0))) == (((0, 0), (1, 0)), 1)
    initial = torch.tensor([1, 0, 2j, 0], dtype=torch.complex128) / 5**0.5
    for outcome in (0, 1):
        torch.testing.assert_close(
            forced_branch(initial, gates, (outcome,)),
            initial / 2**0.5,
            atol=1e-14,
            rtol=0,
        )
    with pytest.raises(AssertionError, match="Uncancelled"):
        certified_and_channel_map(gates[:-1], ((0, 0), (1, 0)))
    wrong_target = (
        *gates[:-1],
        Instruction("z", (1,), metadata={"conditions": ((7, 1),)}),
    )
    with pytest.raises(AssertionError, match="Uncancelled"):
        certified_and_channel_map(wrong_target, ((0, 0), (1, 0)))


def test_branch_identity_does_not_imply_clean_output() -> None:
    gates = (
        *and_compute_ir(0, 1, 2),
        *and_uncompute_ir(0, 1, 2, 0),
        Instruction("x", (2,)),
    )
    with pytest.raises(AssertionError, match="restore every declared clean wire"):
        certify_erasure_histories(gates, n_wires=3, zero_wires=(2,))
    assert (
        certify_erasure_histories(
            gates, n_wires=3, zero_wires=(2,), check_clean_output=False
        )
        == 1
    )
    state = torch.zeros(8, dtype=torch.complex128)
    state[0] = 1
    expected = torch.zeros_like(state)
    expected[1] = 2**-0.5
    for outcome in (0, 1):
        torch.testing.assert_close(
            forced_branch(state, gates, (outcome,)), expected, atol=1e-14, rtol=0
        )
    broken = (
        *and_compute_ir(0, 1, 2),
        Instruction("x", (0,)),
        *and_uncompute_ir(0, 1, 2, 0),
    )
    with pytest.raises(AssertionError, match="predicate"):
        certify_erasure_histories(
            broken, n_wires=3, zero_wires=(2,), check_clean_output=False
        )


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
    prefix: tuple[Instruction, ...] = ()
    middle: tuple[Instruction, ...] = ()
    suffix: tuple[Instruction, ...] = ()
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


def test_matrix_support_preserves_control_and_does_not_round_tiny_transitions() -> None:
    compute, erase = and_compute_ir(0, 1, 2), and_uncompute_ir(0, 1, 2, 0)
    controlled = torch.eye(4, dtype=torch.complex128)
    controlled[2:, 2:] = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex128) / (
        2**0.5
    )
    for middle in (
        Instruction("ch", (2, 3)),
        Instruction("any", (2, 3), matrix=controlled),
    ):
        gates = (*compute, middle, *erase)
        assert certify_erasure_histories(gates, n_wires=4, zero_wires=(2,)) == 1
    tiny = torch.tensor([[1, -5e-21j], [-5e-21j, 1]], dtype=torch.complex128)
    with pytest.raises(AssertionError, match="predicate was not restored"):
        certify_erasure_histories(
            (*compute, Instruction("any", (2,), matrix=tiny), *erase),
            n_wires=3,
            zero_wires=(2,),
        )
    with pytest.raises(AssertionError, match="unitary"):
        certify_erasure_histories(
            (Instruction("any", (0,), matrix=torch.ones(2, 2)),),
            n_wires=1,
            zero_wires=(),
        )
    with pytest.raises(AssertionError, match="cannot override"):
        certify_erasure_histories(
            (Instruction("x", (0,), matrix=torch.eye(2)),), n_wires=1, zero_wires=()
        )


def test_literal_seven_t_cswap_has_exact_positive_basis_action_and_support() -> None:
    from flagquantum.compiler.registers.primitives.controlled_swap import (
        controlled_swap_bit_ir,
    )
    from flagquantum.simulation.unitary import get_unitary

    macro = controlled_swap_bit_ir(0, 1, 2)
    matrix = get_unitary(CircuitIR(3, macro), dtype=torch.complex128)
    expected = torch.zeros_like(matrix)
    for control, left, right in product((0, 1), repeat=3):
        target = (
            (control << 2)
            | ((right if control else left) << 1)
            | (left if control else right)
        )
        expected[target, (control << 2) | (left << 1) | right] = 1
    torch.testing.assert_close(matrix, expected, atol=2e-14, rtol=0)
    gates = (Instruction("x", (1,)), *macro, *macro, Instruction("x", (1,)))
    assert certify_erasure_histories(gates, n_wires=3, zero_wires=(1, 2)) == 0


def test_boolean_support_canonical_parity_and_product_match_complete_truth_table() -> (
    None
):
    from tests.unit.measurement_certificate import _BooleanSupport

    support = _BooleanSupport()
    a, b, c = (support.variable(i) for i in range(3))
    expression = support.xor(support.multiply(a, b), support.multiply(a, c))
    assert expression == support.multiply(a, support.xor(b, c))
    for values in product((0, 1), repeat=3):
        node = expression
        while node > 1:
            variable, low, high = support.nodes[node]
            node = high if values[variable] else low
        assert node == (values[0] & (values[1] ^ values[2]))


def test_controlled_y_support_preserves_exact_phase_and_all_erasure_branches() -> None:
    compute, erase = and_compute_ir(0, 1, 2), and_uncompute_ir(0, 1, 2, 0)
    gates = (*compute, Instruction("cy", (2, 3)), *erase)
    assert certify_erasure_histories(gates, n_wires=4, zero_wires=(2,)) == 1
    state = torch.zeros((4, 2, 2), dtype=torch.complex128)
    state[:, 0, :] = torch.tensor(
        [[1, 1j], [2j, -1], [3, 2 - 1j], [1 - 2j, 2]], dtype=state.dtype
    )
    initial = state.flatten() / state.norm()
    expected = state.clone()
    expected[3, 0, 0] = -1j * state[3, 0, 1]
    expected[3, 0, 1] = 1j * state[3, 0, 0]
    expected = expected.flatten() / state.norm()
    for outcome in (0, 1):
        torch.testing.assert_close(
            forced_branch(initial, gates, (outcome,)),
            expected / (2**0.5),
            atol=2e-14,
            rtol=0,
        )
    # A CY that changes an erasure control invalidates the restored predicate.
    with pytest.raises(AssertionError, match="predicate"):
        certify_erasure_histories(
            (*compute, Instruction("cy", (3, 0)), *erase),
            n_wires=4,
            zero_wires=(2,),
        )
    # A CY after erasure cannot certify a clean target unless it is restored.
    with pytest.raises(AssertionError, match="restore every declared clean wire"):
        certify_erasure_histories(
            (*compute, *erase, Instruction("cy", (0, 2))),
            n_wires=3,
            zero_wires=(2,),
        )
