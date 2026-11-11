"""Explicit external trace effects preserve source failures and complete channels."""

from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace
from itertools import product
from typing import Any

import numpy as np
import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_classical import discard_word_function
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import FunctionRecord, Operation, ProgramRecord
from flagquantum.core.ir.program.register_types import (
    classical_bit_type,
    quantum_bit_type,
)
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.tensor_network.component_superoperator_nodes import (
    component_superoperator_nodes,
)
from flagquantum.simulation.tensor_network.conjugate_pairs import conjugate_pair_labels
from flagquantum.simulation.tensor_network.models import TensorNetworkNode
from flagquantum.simulation.tensor_network.register_superoperator_binding import (
    contract_register_superoperator,
)
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_component_tensor_nodes import order
from tests.unit.test_program_owned_measurement import measured_function

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
Ports = Mapping[str, tuple[tuple[int, ...], ...]]


def quantum_drop() -> FunctionRecord:
    """Use existing measure-and-forget IR, never reinterpret Free as a trace."""
    leaf = measured_function("drop")
    block = leaf.blocks[0]
    assert block.return_values is not None
    return replace(
        leaf,
        blocks=(
            replace(
                block,
                operations=(
                    *block.operations,
                    Operation("discard_classical_register", block.return_values),
                ),
                return_values=(),
            ),
        ),
        registers=(Register("q", quantum_bit_type(), side=RegisterSide.LEFT),),
    )


def graph(classical: bool, shape: tuple[int, ...], spectator: bool) -> ProgramRecord:
    dtype = classical_bit_type() if classical else quantum_bit_type()
    sig = Signature(
        [Register("drop", dtype, shape, RegisterSide.LEFT)]
        + ([Register("spectator", quantum_bit_type())] if spectator else [])
    )
    builder = ComponentProgramBuilder(
        sig, (discard_word_function("drop") if classical else quantum_drop(),)
    )
    for word in np.asarray(builder.inputs["drop"]).flat:
        builder.call_dict("drop", {"c" if classical else "q": word})
    return builder.finalize(
        {"spectator": builder.inputs["spectator"]} if spectator else {}
    )


def source_graph(
    classical: bool, shape: tuple[int, ...], spectator: bool, identity: bool
) -> Any:
    from qualtran import BloqBuilder, CBit, QBit, Side
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature
    from qualtran.bloqs.basic_gates import Discard, DiscardQ
    from qualtran.bloqs.bookkeeping import Cast

    dtype = CBit() if classical else QBit()
    sig = SourceSignature(
        [SourceRegister("drop", dtype, shape, Side.LEFT)]
        + ([SourceRegister("spectator", QBit())] if spectator else [])
    )
    bb, inputs = BloqBuilder.from_signature(sig)
    for word in np.asarray(inputs["drop"]).flat:
        if identity:
            word = bb.add(Cast(dtype, dtype), reg=word)
        bb.add(
            Discard() if classical else DiscardQ(), **{"c" if classical else "q": word}
        )
    return bb.finalize(**({"spectator": inputs["spectator"]} if spectator else {}))


def trace_supplier(
    function: FunctionRecord, incoming: Ports, outgoing: Ports, fresh: Callable[[], int]
) -> Iterable[TensorNetworkNode | int]:
    assert function.name == "drop" and not outgoing
    yield from (edge for words in incoming.values() for word in words for edge in word)


@pytest.mark.parametrize("classical", [False, True])
@pytest.mark.parametrize("shape", [(), (2,), (1, 2), (0,)])
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.complex64, torch.complex128]
)
def test_external_trace_effect_full_channel_against_original_identity_wrapped_graph(
    classical: bool, shape: tuple[int, ...], dtype: torch.dtype
) -> None:
    from qualtran.simulation.tensor._dense import quimb_to_dense
    from qualtran.simulation.tensor._quimb import cbloq_to_superquimb

    record = graph(classical, shape, True)
    nodes, inc, out = component_superoperator_nodes(
        record, order(record), trace_supplier, torch.empty((), dtype=dtype)
    )
    n = int(np.prod(shape))
    sig = Signature(record.functions[0].registers or ())
    actual = contract_register_superoperator(nodes, sig, inc, out)
    original = source_graph(classical, shape, True, False)
    if n:
        # Preserve the exact original missing-input failure, without changing source.
        with pytest.raises(KeyError) as error:
            quimb_to_dense(
                cbloq_to_superquimb(original), original.signature, superoperator=True
            )
        assert error.value.args == (("drop", tuple(0 for _ in shape), 0),)
    equivalent = source_graph(classical, shape, True, True)
    expected = quimb_to_dense(
        cbloq_to_superquimb(equivalent), equivalent.signature, superoperator=True
    )
    np.testing.assert_array_equal(actual.numpy(), expected)
    assert len(nodes) == n + 2
    effects = [node for node in nodes if node.name == "input_trace_effect"]
    assert len(effects) == n
    for effect, word in zip(effects, inc["drop"], strict=True):
        assert effect.labels == (
            *conjugate_pair_labels(word),
            *conjugate_pair_labels(word, backward=True),
        )
        torch.testing.assert_close(effect.tensor, torch.eye(2, dtype=dtype))
    assert not out.keys() - {"spectator"}
    if n:
        assert len({node.tensor.untyped_storage().data_ptr() for node in effects}) == 1
    # Independent partial trace checks complex off-diagonal spectator coherence.
    theta = torch.tensor(0.27, dtype=torch.float64, requires_grad=True)
    dimension = 1 << (n + 1)
    psi = torch.arange(1, dimension + 1) * torch.exp(
        1j * theta * torch.arange(dimension)
    )
    psi = psi / psi.norm()
    density = torch.outer(psi, psi.conj())
    result = torch.einsum("abij,ij->ab", actual.to(torch.complex128), density)
    reshaped = psi.reshape(1 << n, 2)
    reference = torch.einsum("ia,ib->ab", reshaped, reshaped.conj())
    torch.testing.assert_close(result, reference, rtol=0, atol=2e-15)
    torch.testing.assert_close(
        torch.autograd.grad(result.real.sum(), theta, retain_graph=True)[0],
        torch.autograd.grad(reference.real.sum(), theta)[0],
        rtol=0,
        atol=2e-14,
    )


@pytest.mark.parametrize("classical", [False, True])
@pytest.mark.parametrize("width", [1, 2, 8])
def test_all_inputs_discarded_returns_trace_identity_and_source_stays_empty(
    classical: bool, width: int
) -> None:
    from qualtran.simulation.tensor._dense import quimb_to_dense
    from qualtran.simulation.tensor._quimb import cbloq_to_superquimb

    record = graph(classical, (width,), False)
    nodes, inc, out = component_superoperator_nodes(
        record, order(record), trace_supplier, torch.empty((), dtype=torch.complex128)
    )
    actual = contract_register_superoperator(
        nodes, Signature(record.functions[0].registers or ()), inc, out
    )
    torch.testing.assert_close(
        actual, torch.eye(1 << width, dtype=torch.complex128), atol=0, rtol=0
    )
    assert len(nodes) == width
    original = source_graph(classical, (width,), False, False)
    tn = cbloq_to_superquimb(original)
    assert len(tn.tensors) == 0
    with pytest.raises(KeyError):
        quimb_to_dense(tn, original.signature, superoperator=True)


@pytest.mark.parametrize("width", [1, 2, 3])
def test_quantum_measure_and_forget_append_all_branches_work_zero(width: int) -> None:
    record = graph(False, (width,), True)
    bound = bind_register_program(
        record, tuple((i,) for i in range(width + 1)), clean=(width + 1,)
    )
    instructions = (
        Circuit(width + 2)
        .append(FieldOperation(width + 2, bound.circuit.instructions))
        .to_ir()
        .instructions
    )
    assert instructions == bound.circuit.instructions
    assert Counter(g.name for g in instructions) == {"measure": width}
    for i, gate in enumerate(instructions):
        assert gate.wires == (i,) and gate.metadata["classical_bit"] == i
    state = torch.arange(1, (1 << (width + 1)) + 1, dtype=torch.float64).to(
        torch.complex128
    )
    state *= torch.exp(1j * torch.arange(state.numel()) * 0.31)
    state /= state.norm()
    with_work = torch.zeros(1 << (width + 2), dtype=torch.complex128)
    with_work[::2] = state
    total = torch.zeros((2, 2), dtype=torch.complex128)
    for bits in product((0, 1), repeat=width):
        branch = forced_branch(with_work, instructions, bits).reshape(1 << width, 2, 2)
        assert torch.count_nonzero(branch[..., 1]) == 0
        index = sum(bit << (width - i - 1) for i, bit in enumerate(bits))
        expected = torch.zeros_like(branch)
        expected[index, :, 0] = state.reshape(1 << width, 2)[index]
        torch.testing.assert_close(branch, expected, rtol=0, atol=0)
        total += branch[index, :, 0, None] @ branch[index, None, :, 0].conj()
    nodes, inc, out = component_superoperator_nodes(
        record, order(record), trace_supplier, torch.empty((), dtype=torch.complex128)
    )
    effect = contract_register_superoperator(
        nodes, Signature(record.functions[0].registers or ()), inc, out
    )
    observed = torch.einsum("abij,ij->ab", effect, torch.outer(state, state.conj()))
    torch.testing.assert_close(total, observed, rtol=0, atol=2e-15)


@pytest.mark.parametrize("bits", list(product((0, 1), repeat=2)))
def test_classical_external_values_drop_with_no_quantum_instructions(
    bits: tuple[int, ...],
) -> None:
    record = graph(True, (2,), True)
    bound = bind_register_program(
        record, ((), (), (0,)), clean=(1,), classical_inputs=(*bits, None)
    )
    instructions = (
        Circuit(2)
        .append(FieldOperation(2, bound.circuit.instructions))
        .to_ir()
        .instructions
    )
    assert Counter(g.name for g in instructions) == {"classical_xor": 2}
    assert bound.classical_outputs == () and bound.outputs == ((0,),)
    nodes, inc, out = component_superoperator_nodes(
        record, order(record), trace_supplier, torch.empty((), dtype=torch.complex128)
    )
    channel = contract_register_superoperator(
        nodes, Signature(record.functions[0].registers or ()), inc, out
    )
    state = torch.zeros(8, dtype=torch.complex128)
    index = 2 * bits[0] + bits[1]
    state[2 * index : 2 * index + 2] = torch.tensor(
        [1 + 2j, 3 - 1j], dtype=state.dtype
    ) / np.sqrt(15)
    actual = torch.einsum("abij,ij->ab", channel, torch.outer(state, state.conj()))
    spectator = state[2 * index : 2 * index + 2]
    torch.testing.assert_close(actual, torch.outer(spectator, spectator.conj()))


@pytest.mark.parametrize(
    "failure", ["output", "used_input", "duplicate_input", "invalid_input"]
)
def test_invalid_external_discard_refused_without_changing_supplier_boundary(
    failure: str,
) -> None:
    record = graph(False, (), True)

    def broken(
        function: FunctionRecord, inc: Ports, out: Ports, fresh: Callable[[], int]
    ) -> Iterable[TensorNetworkNode | int]:
        label = inc["q"][0][0]
        if failure == "output":
            # Entry allocation is inputs first; right spectator is the next label.
            yield 2
        elif failure == "used_input":
            yield TensorNetworkNode(torch.ones(2), (label,))
            yield label
        elif failure == "duplicate_input":
            yield label
            yield label
        else:
            yield True

    with pytest.raises(ValueError):
        component_superoperator_nodes(record, order(record), broken, torch.empty(()))


@pytest.mark.parametrize("classical", [False, True])
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_external_and_internal_discards_coexist_in_one_graph(
    classical: bool, dtype: torch.dtype
) -> None:
    from qualtran import BloqBuilder, CBit, QBit, Side
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature
    from qualtran.bloqs.basic_gates import Discard, DiscardQ, MeasureZ
    from qualtran.bloqs.bookkeeping import Cast
    from qualtran.simulation.tensor._dense import quimb_to_dense
    from qualtran.simulation.tensor._quimb import cbloq_to_superquimb

    from flagquantum.simulation.tensor_network.measurement_copy import (
        z_measurement_copy_node,
    )

    registers = (
        Register(
            "external",
            classical_bit_type() if classical else quantum_bit_type(),
            side=RegisterSide.LEFT,
        ),
        Register("q", quantum_bit_type(), side=RegisterSide.LEFT),
        Register("spectator", quantum_bit_type()),
    )
    ext = replace(
        discard_word_function("external_drop") if classical else quantum_drop(),
        name="external_drop",
    )
    builder = ComponentProgramBuilder(
        Signature(registers),
        (ext, measured_function("measurement"), discard_word_function("internal_drop")),
    )
    builder.call_dict(
        "external_drop", {"c" if classical else "q": builder.inputs["external"]}
    )
    measured = builder.call_dict("measurement", {"q": builder.inputs["q"]})["c"]
    builder.call_dict("internal_drop", {"c": measured})
    record = builder.finalize({"spectator": builder.inputs["spectator"]})
    like = torch.empty((), dtype=dtype)

    def supply(
        function: FunctionRecord,
        incoming: Ports,
        outgoing: Ports,
        fresh: Callable[[], int],
    ) -> Iterable[TensorNetworkNode | int]:
        if function.name == "measurement":
            env = fresh()
            yield z_measurement_copy_node(
                incoming["q"][0][0], outgoing["c"][0][0], env, like
            )
            yield env
        else:
            yield from (
                label for words in incoming.values() for word in words for label in word
            )

    nodes, inc, out = component_superoperator_nodes(record, order(record), supply, like)
    source_type = CBit() if classical else QBit()
    source_sig = SourceSignature(
        [
            SourceRegister("external", source_type, side=Side.LEFT),
            SourceRegister("q", QBit(), side=Side.LEFT),
            SourceRegister("spectator", QBit()),
        ]
    )
    bb, inputs = BloqBuilder.from_signature(source_sig)
    identity = bb.add(Cast(source_type, source_type), reg=inputs["external"])
    bb.add(
        Discard() if classical else DiscardQ(), **{"c" if classical else "q": identity}
    )
    bb.add(Discard(), c=bb.add(MeasureZ(), q=inputs["q"]))
    reference = bb.finalize(spectator=inputs["spectator"])
    actual = contract_register_superoperator(nodes, Signature(registers), inc, out)
    expected = quimb_to_dense(
        cbloq_to_superquimb(reference), source_sig, superoperator=True
    )
    np.testing.assert_array_equal(actual.numpy(), expected)
    assert len(nodes) == 5
    assert sum(node.name == "input_trace_effect" for node in nodes) == 1
