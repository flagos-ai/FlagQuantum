"""Typed bitwise graphs preserve source calls, wire order and coherent action."""

from collections import Counter
from typing import Literal

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler._component_trace.program_call_graph import (
    program_component_call_graph,
)
from flagquantum.compiler.registers.bits.bitwise_component import bitwise_word_program
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import verify_program_record
from flagquantum.core.ir.program.registers import Register, RegisterType
from flagquantum.ecosystem.cirq._register_emitter import import_register_emitter

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
Operation = Literal["xor", "xor_constant", "bitwise_not"]


@pytest.mark.parametrize("encoding", ("opaque", "uint", "signed", "montgomery"))
@pytest.mark.parametrize("width", (1, 3, 5))
@pytest.mark.parametrize("operation", ("xor", "xor_constant", "bitwise_not"))
def test_typed_words_match_original_stream_tensor_and_append(
    encoding: str, width: int, operation: Operation
) -> None:
    from qualtran import QAny, QInt, QMontgomeryUInt, QUInt
    from qualtran.bloqs.arithmetic.bitwise import BitwiseNot, Xor, XorK
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    dtype = RegisterType(encoding, width)
    source_dtype = {
        "opaque": QAny,
        "uint": QUInt,
        "signed": QInt,
        "montgomery": QMontgomeryUInt,
    }[encoding](width)
    pattern = (1 << width) - 1
    source_value = -1 if encoding == "signed" else pattern
    source = (
        Xor(source_dtype)
        if operation == "xor"
        else (
            XorK(source_dtype, source_value)
            if operation == "xor_constant"
            else BitwiseNot(source_dtype)
        )
    )
    graph = bitwise_word_program(
        dtype, operation, pattern=pattern if operation == "xor_constant" else None
    )
    assert verify_program_record(graph) == ()
    entry = next(function for function in graph.functions if function.name == "main")
    assert entry.registers is not None
    assert all(register.dtype == dtype for register in entry.registers)
    logical = width * (2 if operation == "xor" else 1)
    order = tuple(reversed(range(logical)))
    inputs = (order[:width], order[width:]) if operation == "xor" else (order,)
    bound = bind_register_program(graph, inputs)
    assert bound.outputs == inputs and bound.events == ()
    ports = dict(
        zip((register.name for register in entry.registers), inputs, strict=True)
    )
    captured = import_register_emitter(source)
    expected_stream = captured(ports, (), (), False)
    assert bound.circuit.instructions == expected_stream
    assert captured(ports, (), (), True) == expected_stream
    assert Counter(gate.name for gate in expected_stream) == {
        "cx" if operation == "xor" else "x": width
    }
    stored = program_component_call_graph(graph)
    leaf = "bitwise_cx" if operation == "xor" else "bitwise_x"
    assert (
        sum(count for name, count in stored.leaf_counts.items() if name.endswith(leaf))
        == width
    )
    definitions = {function.name: function for function in graph.functions}
    direct = []
    for call in entry.blocks[0].operations:
        assert call.callee is not None
        child = definitions[call.callee]
        assert child.registers is not None
        direct.append(
            {"bitwise_x": "XGate", "bitwise_cx": "CNOT"}.get(
                child.name,
                (
                    "OnEach"
                    if operation == "bitwise_not"
                    else "Join" if child.registers[0].shape else "Split"
                ),
            )
        )
    assert direct == [
        type(instance.bloq).__name__
        for instance in sorted(
            source.decompose_bloq().bloq_instances, key=lambda node: node.i
        )
    ]
    if operation == "bitwise_not":
        root = entry.blocks[0].operations
        assert len(root) == 1 and root[0].opcode == "call"
        child = next(f for f in graph.functions if f.name == root[0].callee)
        assert child.registers == (Register("q", dtype),)
        assert len(source.decompose_bloq().bloq_instances) == 1
    model = get_cost_value(source, QECGatesCost())
    assert model.clifford == width
    assert model.total_t_count() == model.measurement == 0

    # One entangled spectator and one explicitly clean caller wire test ownership.
    initial = torch.arange(1, (1 << (logical + 1)) + 1, dtype=torch.float64)
    initial = initial.to(torch.complex128) * torch.exp(0.17j * initial)
    initial /= initial.norm()
    full_initial = torch.stack((initial, torch.zeros_like(initial)), dim=1).flatten()
    circuit = Circuit(logical + 2, inputs=full_initial, dtype=torch.complex128)
    circuit.append(FieldOperation(logical + 2, bound.circuit.instructions))
    reference = Circuit(logical + 2, inputs=full_initial, dtype=torch.complex128)
    reference.append(FieldOperation(logical + 2, expected_stream))
    torch.testing.assert_close(circuit.state(), reference.state(), rtol=0, atol=2e-14)
    assert torch.count_nonzero(circuit.state().flatten()[1::2]) == 0
    if width <= 3:
        matrix = torch.as_tensor(
            source.tensor_contract(), dtype=torch.complex128
        ).reshape(1 << logical, 1 << logical)
        # Reverse physical data axes back to source logical order.
        state = initial.reshape((2,) * (logical + 1))
        logical_state = state.permute(*order, logical).reshape(1 << logical, 2)
        expected = (
            (matrix @ logical_state)
            .reshape((2,) * (logical + 1))
            .permute(*order, logical)
            .flatten()
        )
        torch.testing.assert_close(
            circuit.state().flatten()[::2], expected, rtol=0, atol=2e-14
        )
    circuit.append(FieldOperation(logical + 2, bound.circuit.instructions))
    torch.testing.assert_close(
        circuit.state().flatten()[::2], initial, rtol=0, atol=2e-14
    )


def test_zero_constant_has_no_gate_calls_and_retains_both_rebindings() -> None:
    graph = bitwise_word_program(RegisterType("uint", 3), "xor_constant", pattern=0)
    bound = bind_register_program(graph, ((2, 0, 1),))
    assert bound.circuit.instructions == () and bound.outputs == ((2, 0, 1),)
    assert not any(
        name == "bitwise_x" for name in program_component_call_graph(graph).leaf_counts
    )


@pytest.mark.parametrize("pattern", (0, 1, 6, 87, 128, 255))
def test_sparse_encoded_constant_calls_preserve_original_msb_order(
    pattern: int,
) -> None:
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic.bitwise import XorK

    graph = bitwise_word_program(
        RegisterType("uint", 8), "xor_constant", pattern=pattern
    )
    wires = (6, 2, 7, 1, 4, 0, 5, 3)
    bound = bind_register_program(graph, (wires,))
    source = import_register_emitter(XorK(QUInt(8), pattern))
    assert bound.circuit.instructions == source({"x": wires}, (), (), False)
    assert len(bound.circuit.instructions) == pattern.bit_count()
    calls = program_component_call_graph(graph)
    assert calls.leaf_counts.get("bitwise_x", 0) == pattern.bit_count()


@pytest.mark.parametrize("width", (0, -1))
def test_nonpositive_execution_width_is_rejected(width: int) -> None:
    with pytest.raises(ValueError):
        bitwise_word_program(RegisterType("uint", width), "xor")


@pytest.mark.parametrize("pattern", (-1, 8, True, 1.5, None))
def test_constant_pattern_admission_remains_explicit(pattern: object) -> None:
    with pytest.raises(ValueError):
        bitwise_word_program(RegisterType("uint", 3), "xor_constant", pattern=pattern)  # type: ignore[arg-type]


def test_symbolic_width_remains_a_model_only_domain() -> None:
    from sympy import Symbol

    with pytest.raises(ValueError, match="unsupported executable register"):
        bitwise_word_program(RegisterType("uint", Symbol("n")), "bitwise_not")
