"""Phase-free action leaves compose with explicit quantum partial traces."""

from itertools import product
from typing import Any

import numpy as np
import pytest
import torch

import flagquantum as fq
from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.core.ir import Instruction
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.classical_bitwise import xor_classical_values
from flagquantum.simulation.tensor_network.classical_action_tensor import (
    classical_action_tensor_node,
)
from flagquantum.simulation.tensor_network.conjugate_pairs import tensor_conjugate_pairs
from flagquantum.simulation.tensor_network.register_superoperator_binding import (
    contract_register_superoperator,
)
from flagquantum.simulation.tensor_network.trace_indices import trace_conjugate_indices
from tests.unit.test_classical_action_tensor import decode, encode

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
NAMES = ("ctrl", "target")
INCOMING = {"ctrl": ((3,),), "target": ((17,),)}
OUTGOING = {"ctrl": ((41,),), "target": ((59,),)}


def xor_action(**values: object) -> tuple[object, ...]:
    return xor_classical_values(
        int(np.asarray(values["ctrl"])), int(np.asarray(values["target"]))
    )


def source_channel(discard: tuple[bool, bool]) -> Any:
    from qualtran import BloqBuilder, QBit, Side
    from qualtran import Register as SourceRegister
    from qualtran.bloqs.basic_gates import CNOT, DiscardQ

    bb = BloqBuilder()
    inputs = {
        name: bb.add_register(SourceRegister(name, QBit(), side=Side.LEFT))
        for name in NAMES
    }
    values = bb.add_d(CNOT(), **inputs)
    kept = {}
    for name, remove in zip(NAMES, discard, strict=True):
        if remove:
            bb.add(DiscardQ(), q=values[name])
        else:
            kept[name] = values[name]
    return bb.finalize(**kept)


def native_channel(discard: tuple[bool, bool], dtype: torch.dtype) -> torch.Tensor:
    full = Signature([Register(name, quantum_bit_type()) for name in NAMES])
    node = classical_action_tensor_node(
        full,
        INCOMING,
        OUTGOING,
        xor_action,
        decode,
        encode,
        torch.empty((), dtype=dtype),
        operation_name="xor",
    )
    labels = tuple(
        OUTGOING[name][0][0]
        for name, remove in zip(NAMES, discard, strict=True)
        if remove
    )
    sig = Signature(
        [
            Register(
                name,
                quantum_bit_type(),
                side=RegisterSide.LEFT if remove else RegisterSide.THRU,
            )
            for name, remove in zip(NAMES, discard, strict=True)
        ]
    )
    outputs = {
        name: OUTGOING[name]
        for name, remove in zip(NAMES, discard, strict=True)
        if not remove
    }
    return contract_register_superoperator(
        trace_conjugate_indices(tensor_conjugate_pairs((node,)), labels),
        sig,
        INCOMING,
        outputs,
    )


@pytest.mark.parametrize("discard", list(product((False, True), repeat=2)))
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_action_leaf_then_each_discard_subset_matches_original_channel_and_append(
    discard: tuple[bool, bool],
    dtype: torch.dtype,
) -> None:
    source = source_channel(discard)
    expected = torch.as_tensor(source.tensor_contract(superoperator=True), dtype=dtype)
    channel = native_channel(discard, dtype)
    torch.testing.assert_close(channel, expected, atol=0, rtol=0)
    if any(discard):
        with pytest.raises(NotImplementedError, match="classical simulation"):
            source.call_classically(ctrl=1, target=0)
    else:
        assert source.call_classically(ctrl=1, target=0) == (1, 1)

    # A mixed input retains complex coherences and cannot be certified by probabilities alone.
    states = torch.tensor(
        [[1 + 2j, 2 - 1j, -3 + 1j, 1 - 4j], [2 - 3j, -1 + 2j, 1 + 1j, 3 - 2j]],
        dtype=dtype,
    )
    states = states / states.norm(dim=1, keepdim=True)
    density = torch.einsum("bi,bj->ij", states, states.conj()) / 2
    kept = tuple(i for i, remove in enumerate(discard) if not remove)
    removed = tuple(i for i, remove in enumerate(discard) if remove)
    size = 2 ** len(kept)
    actual = torch.einsum("abij,ij->ab", channel.reshape(size, size, 4, 4), density)
    expected_density = torch.zeros((size, size), dtype=dtype)
    # Append executes only the original CX; the caller performs the open-system trace.
    instructions = (Instruction("cx", (0, 1)),)
    for state in states:
        initial = torch.zeros(8, dtype=dtype)
        initial[::2] = state
        evolved = (
            fq.Circuit(3, inputs=initial, dtype=dtype)
            .append(FieldOperation(3, instructions))
            .state()
            .flatten()
        )
        assert torch.count_nonzero(evolved[1::2]) == 0
        unitary = torch.tensor(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=dtype
        )
        torch.testing.assert_close(evolved[::2], unitary @ state)
        reduced = evolved[::2].reshape(2, 2).permute(*kept, *removed).reshape(size, -1)
        expected_density += reduced @ reduced.conj().T / 2
    torch.testing.assert_close(actual, expected_density)
    torch.testing.assert_close(torch.trace(actual), torch.ones((), dtype=dtype))
    # Trace preservation is checked for every matrix unit, including off-diagonal input.
    torch.testing.assert_close(
        channel.reshape(size, size, 4, 4).diagonal(dim1=0, dim2=1).sum(-1),
        torch.eye(4, dtype=dtype),
        atol=0,
        rtol=0,
    )
    if not kept:
        torch.testing.assert_close(channel, torch.eye(4, dtype=dtype), atol=0, rtol=0)


def test_traced_channel_keeps_complex_coherence_and_input_gradient() -> None:
    theta = torch.tensor(0.31, dtype=torch.float64, requires_grad=True)
    # |0> tensor (cos(theta)|0> + i sin(theta)|1>): tracing control preserves target coherence.
    state = torch.stack(
        (
            torch.cos(theta).to(torch.complex128),
            1j * torch.sin(theta),
            theta * 0,
            theta * 0,
        )
    )
    rho = torch.outer(state, state.conj())
    channel = native_channel((True, False), torch.complex128)
    result = torch.einsum("abij,ij->ab", channel, rho)
    torch.testing.assert_close(result[0, 1], -1j * torch.cos(theta) * torch.sin(theta))
    (gradient,) = torch.autograd.grad(result[0, 1].imag, (theta,))
    torch.testing.assert_close(gradient, -torch.cos(2 * theta))


def test_empty_classical_output_is_a_coherent_effect_not_quantum_discard() -> None:
    from qualtran.bloqs.basic_gates import DiscardQ
    from qualtran.simulation.tensor._tensor_from_classical import (
        bloq_to_dense_via_classical_action,
    )

    sig = Signature([Register("q", quantum_bit_type(), side=RegisterSide.LEFT)])
    effect = classical_action_tensor_node(
        sig,
        {"q": ((3,),)},
        {},
        lambda **_: (),
        decode,
        encode,
        torch.empty((), dtype=torch.complex128),
        operation_name="coherent_sum",
    )
    coherent = contract_register_superoperator(
        tensor_conjugate_pairs((effect,)), sig, {"q": ((3,),)}, {}
    )
    quantum_trace = torch.eye(2, dtype=torch.complex128)
    torch.testing.assert_close(coherent, torch.ones((2, 2), dtype=torch.complex128))
    assert not torch.equal(coherent, quantum_trace)
    plus_density = torch.ones((2, 2), dtype=torch.complex128) / 2
    assert torch.einsum("ij,ij->", coherent, plus_density) == 2
    assert torch.einsum("ij,ij->", quantum_trace, plus_density) == 1
    # The original quantum discard explicitly has no classical action to enumerate.
    with pytest.raises(NotImplementedError, match="classical simulation"):
        bloq_to_dense_via_classical_action(DiscardQ())
