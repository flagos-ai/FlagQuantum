"""Directional array actions compose through unchanged native tensor consumers."""

from collections import Counter
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
import torch

from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_graph import iter_component_calls
from flagquantum.core.ir.program import FunctionRecord
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.bit_encoding import bit_from_bits, bit_to_bits
from flagquantum.simulation.classical_bitwise import bitwise_not_classical_value
from flagquantum.simulation.tensor_network.classical_action_tensor import (
    classical_action_tensor_node,
)
from flagquantum.simulation.tensor_network.component_tensor_nodes import (
    component_tensor_nodes,
)
from flagquantum.simulation.tensor_network.models import TensorNetworkNode
from flagquantum.simulation.tensor_network.register_tensor_binding import (
    bind_register_tensor,
    contract_register_tensors,
)
from flagquantum.simulation.value_encoding import decode_value_array, encode_value_array
from tests.unit.test_classical_action_tensor import decode, encode
from tests.unit.test_component_expansion import _hierarchy

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def fixtures(shape: tuple[int, ...]) -> tuple[Any, Any, Signature]:
    from qualtran._infra.bloq import Bloq
    from qualtran._infra.data_types import QBit
    from qualtran._infra.registers import Register as SourceRegister
    from qualtran._infra.registers import Side
    from qualtran._infra.registers import Signature as SourceSignature
    from qualtran.bloqs.basic_gates.x_basis import XGate

    sig = Signature(
        [
            Register("before", quantum_bit_type(), shape, RegisterSide.LEFT),
            Register("after", quantum_bit_type(), shape, RegisterSide.RIGHT),
        ]
    )
    base, _ = _hierarchy(1, 1)
    h = base.functions[1]
    x = replace(
        h,
        name="x",
        blocks=tuple(
            replace(
                block,
                operations=tuple(
                    replace(op, opcode="x") if op.opcode == "h" else op
                    for op in block.operations
                ),
            )
            for block in h.blocks
        ),
    )
    builder = ComponentProgramBuilder(sig, (x,), name="flip_array")
    incoming = np.asarray(builder.inputs["before"])
    outgoing = np.empty(shape, dtype=object)
    for idx in np.ndindex(shape):
        outgoing[idx] = builder.call_dict("x", {"q": incoming[idx]})["q"]
    # Reversal tests row-major word positions independently of bit enumeration.
    result = outgoing.reshape(-1)[::-1].reshape(shape)
    module = builder.finalize({"after": result if shape else result.item()})

    class ArrayFlip(Bloq):
        @property
        def signature(self) -> Any:
            return SourceSignature(
                [
                    SourceRegister("before", QBit(), shape=shape, side=Side.LEFT),
                    SourceRegister("after", QBit(), shape=shape, side=Side.RIGHT),
                ]
            )

        def on_classical_vals(self, **values: Any) -> Any:
            data = np.asarray(values["before"])
            result = (1 - data).reshape(-1)[::-1].reshape(shape)
            return {"after": result if shape else result.item()}

        def build_composite_bloq(self, bb: Any, **values: Any) -> Any:
            data = np.asarray(values["before"])
            result = np.empty(shape, dtype=object)
            for idx in np.ndindex(shape):
                result[idx] = bb.add(XGate(), q=data[idx])
            result = result.reshape(-1)[::-1].reshape(shape)
            return {"after": result if shape else result.item()}

    return module, ArrayFlip(), sig


def array_action(**values: object) -> tuple[object, ...]:
    value = np.asarray(values["before"])
    flipped = bitwise_not_classical_value(value, 1, unsigned=True)
    result = np.asarray(flipped).reshape(-1)[::-1].reshape(value.shape)
    return (result if value.shape else result.item(),)


@pytest.mark.parametrize("shape", [(), (2,), (1, 2), (2, 1)])
@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
def test_directional_array_leaf_source_quantum_composition_codecs_and_append(
    shape: tuple[int, ...], dtype: torch.dtype
) -> None:
    from qualtran.simulation.tensor._dense import bloq_to_dense
    from qualtran.simulation.tensor._tensor_from_classical import (
        my_tensors_from_classical_action,
    )

    import flagquantum as fq
    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.compiler.registers.register_binding import bind_register_program
    from tests.unit.test_register_tensor_binding import source_bindings

    record, source, sig = fixtures(shape)
    count = int(np.prod(shape))
    like = torch.empty((), dtype=dtype)
    labels: dict[tuple[str, int], int] = {}
    src_in, inc = source_bindings(source.signature.lefts(), "in", labels)
    src_out, out = source_bindings(source.signature.rights(), "out", labels)
    expected = my_tensors_from_classical_action(source, src_in, src_out)[0]
    node = classical_action_tensor_node(
        sig, inc, out, array_action, decode, encode, like, operation_name="array_flip"
    )
    assert node.labels == tuple(labels[i] for i in expected.inds)
    np.testing.assert_array_equal(node.tensor.numpy(), expected.data)
    actual = contract_register_tensors((node,), sig, inc, out)
    np.testing.assert_array_equal(actual.numpy(), bloq_to_dense(source))

    # Different caller codec implementation, with the same source scalar rules.
    def alternative_decode(reg: Register, bits: Any) -> object:
        return (
            decode_value_array(bits, lambda row: bit_from_bits(list(row)))
            if reg.shape
            else bit_from_bits(list(bits.flat))
        )

    def alternative_encode(reg: Register, values: Any) -> Any:
        return encode_value_array(values, bit_to_bits)

    other = classical_action_tensor_node(
        sig,
        inc,
        out,
        array_action,
        alternative_decode,
        alternative_encode,
        like,
        operation_name="replacement",
    )
    torch.testing.assert_close(
        contract_register_tensors((other,), sig, inc, out), actual
    )
    # Typed graph of the real executable X decomposition is an independent supplier.
    order = tuple(i for i, _, _ in iter_component_calls(record, "flip_array"))

    def supply(
        function: FunctionRecord,
        inputs: Mapping[str, tuple[tuple[int, ...], ...]],
        outputs: Mapping[str, tuple[tuple[int, ...], ...]],
    ) -> tuple[TensorNetworkNode, ...]:
        assert function.name == "x"
        return (
            bind_register_tensor(
                torch.tensor([[0, 1], [1, 0]], dtype=dtype),
                Signature(function.registers or ()),
                inputs,
                outputs,
            ),
        )

    nodes, left, right = component_tensor_nodes(
        record, order, supply, like, entry="flip_array"
    )
    torch.testing.assert_close(
        contract_register_tensors(nodes, sig, left, right), actual
    )
    wires = tuple((i,) for i in range(count))
    bound = bind_register_program(record, wires, entry="flip_array", clean=(count,))
    assert Counter(op.name for op in bound.circuit.instructions) == {"x": count}
    theta = torch.tensor(0.19, dtype=torch.float64, requires_grad=True)
    state = torch.complex(
        torch.arange(1, 2**count + 1, dtype=torch.float64),
        torch.arange(2**count, 0, -1, dtype=torch.float64),
    )
    state = state / state.norm() * torch.exp(1j * theta)
    target = actual.to(torch.complex128) @ state
    initial = torch.zeros(2 ** (count + 1), dtype=torch.complex128)
    initial[::2] = state
    evolved = (
        fq.Circuit(count + 1, inputs=initial, dtype=torch.complex128)
        .append(FieldOperation(count + 1, bound.circuit.instructions))
        .state()
        .flatten()
    )
    # Output bindings encode the directional reversal; physical bits retain wire positions.
    # Existing output words carry the reversal without adding physical SWAPs.
    logical_wires = tuple(w for word in bound.outputs for w in word)
    permuted = (
        evolved.reshape((2,) * (count + 1))
        .permute(*logical_wires, count)
        .reshape(-1, 2)
    )
    torch.testing.assert_close(permuted[:, 0], target, atol=2e-14, rtol=0)
    assert torch.count_nonzero(permuted[:, 1]) == 0
    (gradient,) = torch.autograd.grad(target.real.sum(), (theta,))
    torch.testing.assert_close(gradient, -target.imag.sum())


@pytest.mark.parametrize("shape", [(0,), (2, 0)])
def test_empty_array_original_codec_failure_remains_visible(
    shape: tuple[int, ...],
) -> None:
    from qualtran.simulation.tensor._tensor_from_classical import (
        my_tensors_from_classical_action,
    )

    from tests.unit.test_register_tensor_binding import source_bindings

    _, source, sig = fixtures(shape)
    labels: dict[tuple[str, int], int] = {}
    src_in, inc = source_bindings(source.signature.lefts(), "in", labels)
    src_out, out = source_bindings(source.signature.rights(), "out", labels)
    with pytest.raises(ValueError) as expected:
        my_tensors_from_classical_action(source, src_in, src_out)
    with pytest.raises(ValueError) as actual:
        classical_action_tensor_node(
            sig,
            inc,
            out,
            array_action,
            decode,
            encode,
            torch.empty(()),
            operation_name=str(source),
        )
    assert str(actual.value) == f"cannot compute tensor for {source}: {expected.value}"
