"""Complete classical tensor source boundaries without repairing the reference."""

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pytest
import torch

from flagquantum.simulation.tensor_network.classical_action_tensor import (
    classical_action_tensor,
    classical_action_tensor_node,
)
from tests.unit.test_classical_action_tensor import (
    decode,
    encode,
    signature_from_source,
)
from tests.unit.test_register_tensor_binding import source_bindings

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("classical", [False, True])
@pytest.mark.parametrize("shape", [(), (2,), (1, 2), (0,), (2, 0)])
@pytest.mark.parametrize("side", ["LEFT", "RIGHT", "THRU"])
def test_full_direction_layout_dense_and_leaf_disposition(
    classical: bool, shape: tuple[int, ...], side: str
) -> None:
    from qualtran import CBit, QBit, Register, Side
    from qualtran import Signature as SourceSignature
    from qualtran.simulation.tensor._tensor_from_classical import (
        bloq_to_dense_via_classical_action,
        my_tensors_from_classical_action,
    )

    source_calls: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    native_calls: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []

    def action(
        log: list[tuple[str, tuple[int, ...], tuple[int, ...]]], **values: Any
    ) -> tuple[object, ...]:
        for name, value in values.items():
            array = np.asarray(value)
            log.append((name, array.shape, tuple(int(i) for i in array.flat)))
        if side == "LEFT":
            return ()
        return (values["bits"] if side == "THRU" else np.zeros(shape, dtype=int),)

    class Source:
        signature = SourceSignature(
            [
                Register(
                    "bits",
                    CBit() if classical else QBit(),
                    shape=shape,
                    side=Side[side],
                )
            ]
        )

        def call_classically(self, **values: Any) -> tuple[object, ...]:
            return action(source_calls, **values)

        def __str__(self) -> str:
            return "direction-layout"

    source = Source()
    signature = signature_from_source(source.signature)

    def native_action(**values: Any) -> tuple[object, ...]:
        return action(native_calls, **values)

    # Raw action objects isolate enumeration from foreign builder validation.
    # Compare exact exceptions and callback prefixes, including failures before
    # the first action for classical arrays and empty codec inputs.
    try:
        expected = bloq_to_dense_via_classical_action(source)
    except (ValueError, IndexError, AssertionError, TypeError) as error:
        with pytest.raises(type(error)) as native_error:
            classical_action_tensor(
                signature, native_action, operation_name=str(source)
            )
        assert str(native_error.value) == str(error)
        assert type(native_error.value.__cause__) is type(error.__cause__)
    else:
        actual = classical_action_tensor(
            signature, native_action, operation_name=str(source)
        )
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == expected.dtype == np.float64
    assert native_calls == source_calls

    source_calls.clear()
    native_calls.clear()
    labels: dict[tuple[str, int], int] = {}
    src_in, incoming = source_bindings(source.signature.lefts(), "in", labels)
    src_out, outgoing = source_bindings(source.signature.rights(), "out", labels)
    if classical:
        # The original leaf counts total bits for data but only quantum bits
        # for indices. Its raw rank/reshape failure is not silently fixed.
        with pytest.raises((ValueError, IndexError, AssertionError, TypeError)):
            my_tensors_from_classical_action(source, src_in, src_out)
        with pytest.raises(ValueError, match="require quantum ports"):
            classical_action_tensor_node(
                signature,
                incoming,
                outgoing,
                native_action,
                decode,
                encode,
                torch.empty(()),
                operation_name=str(source),
            )
        assert native_calls == []
    else:
        try:
            original = my_tensors_from_classical_action(source, src_in, src_out)[0]
        except (ValueError, IndexError, AssertionError, TypeError) as error:
            with pytest.raises(type(error)) as native_error:
                classical_action_tensor_node(
                    signature,
                    incoming,
                    outgoing,
                    native_action,
                    decode,
                    encode,
                    torch.empty(()),
                    operation_name=str(source),
                )
            assert str(native_error.value) == str(error)
        else:
            node = classical_action_tensor_node(
                signature,
                incoming,
                outgoing,
                native_action,
                decode,
                encode,
                torch.empty((), dtype=torch.float64),
                operation_name=str(source),
            )
            np.testing.assert_array_equal(node.tensor.numpy(), original.data)
            assert node.labels == tuple(labels[index] for index in original.inds)
        assert native_calls == source_calls


@pytest.mark.parametrize("gate", ["x", "z"])
def test_mixed_component_action_retains_probabilities_but_not_z_phase(
    gate: str,
) -> None:
    from qualtran.bloqs.basic_gates import XGate, ZGate
    from qualtran.simulation.tensor._tensor_from_classical import (
        bloq_to_dense_via_classical_action,
    )

    from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
    from flagquantum.compiler.registers.component_gate_function import bit_gate_function
    from flagquantum.core.ir import CircuitIR, Instruction
    from flagquantum.core.ir.program import FunctionRecord
    from flagquantum.core.ir.program.register_types import quantum_bit_type
    from flagquantum.core.ir.program.registers import Register
    from flagquantum.core.ir.program.signatures import Signature
    from flagquantum.simulation.tensor_network.classical_action_tensor import (
        classical_action_tensor_node,
    )
    from flagquantum.simulation.tensor_network.component_dense import (
        contract_component_superoperator,
    )
    from flagquantum.simulation.tensor_network.models import TensorNetworkNode

    source = XGate() if gate == "x" else ZGate()
    signature = Signature((Register("q", quantum_bit_type()),))
    leaf = bit_gate_function(
        gate, tuple(signature), CircuitIR(1, (Instruction(gate, (0,)),))
    )
    builder = ComponentProgramBuilder(signature, (leaf,))
    record = builder.finalize(builder.call_dict(gate, builder.inputs))
    like = torch.empty((), dtype=torch.complex128)

    def action(**inputs: object) -> tuple[object, ...]:
        value = inputs["q"]
        assert isinstance(value, (int, np.integer))
        return (int(value) ^ int(gate == "x"),)

    def supply(
        function: FunctionRecord,
        incoming: Mapping[str, tuple[tuple[int, ...], ...]],
        outgoing: Mapping[str, tuple[tuple[int, ...], ...]],
        fresh: Callable[[], int],
    ) -> tuple[TensorNetworkNode, ...]:
        assert function.name == gate
        return (
            classical_action_tensor_node(
                signature,
                incoming,
                outgoing,
                action,
                decode,
                encode,
                like,
                operation_name=gate,
            ),
        )

    actual = contract_component_superoperator(record, (0,), supply, like)
    if gate == "x":
        source_table = torch.as_tensor(
            bloq_to_dense_via_classical_action(source), dtype=like.dtype
        )
    else:
        assert all(source.on_classical_vals(q=bit) == {"q": bit} for bit in (0, 1))
        with pytest.raises(ValueError, match="imparts a phase"):
            bloq_to_dense_via_classical_action(source)
        source_table = torch.eye(2, dtype=like.dtype)
    expected = torch.einsum("ai,bj->abij", source_table, source_table.conj())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    reference = torch.as_tensor(
        source.tensor_contract(superoperator=True), dtype=like.dtype
    )
    state = torch.tensor([1 + 2j, -3 + 4j], dtype=like.dtype) / 30**0.5
    density = state[:, None] * state.conj()[None, :]
    observed = torch.einsum("abij,ij->ab", actual, density)
    physical = torch.einsum("abij,ij->ab", reference, density)
    torch.testing.assert_close(observed.diagonal(), physical.diagonal(), rtol=0, atol=0)
    if gate == "x":
        torch.testing.assert_close(observed, physical, rtol=0, atol=0)
    else:
        assert observed[0, 1].item() != 0
        torch.testing.assert_close(observed[0, 1], -physical[0, 1], rtol=0, atol=0)
