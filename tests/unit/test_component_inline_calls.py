"""Frozen source inlining through the native live register builder."""

from dataclasses import replace
from typing import Any

import numpy as np
import pytest
import torch

from examples.algorithms.register_graphs import parallel_program
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.component_binding_map import map_register_bindings
from flagquantum.compiler.registers.component_bindings import RegisterBindingError
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_gate_function import bit_gate_function
from flagquantum.compiler.registers.component_graph import iter_component_calls
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import FunctionRecord, verify_program_record
from flagquantum.core.ir.program.registers import Register, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.test_component_graph import _native_trace, _source_trace
from tests.unit.test_register_rebinding_contract import SourceRebindings

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
pytest.importorskip("numpy")
pytest.importorskip("qualtran")


def source_builder(signature: Signature) -> Any:
    factory: Any = SourceRebindings
    return factory(signature)


@pytest.mark.parametrize("shape", [(), (2,), (2, 2), (0, 2)])
@pytest.mark.parametrize("width", [0, 1, 3])
def test_inline_rebinding_shapes_and_exact_source_connections(
    shape: tuple[int, ...], width: int
) -> None:
    signature = Signature([Register("word", RegisterType("uint", width), shape)])
    native = ComponentProgramBuilder(signature)
    source = source_builder(signature)
    native_result = np.empty(shape, dtype=object)
    source_result = np.empty(shape, dtype=object)
    for index in signature.get_left("word").all_idxs():
        native_bits = native.split(np.asarray(native.inputs["word"])[index])
        source_bits = source.split(np.asarray(source.inputs["word"])[index])
        native_result[index] = native.join(
            native_bits[::-1], RegisterType("uint", width)
        )
        source_result[index] = source.join(
            source_bits[::-1], RegisterType("uint", width)
        )
    record = native.finalize({"word": native_result if shape else native_result[()]})
    original = source.builder.finalize(
        word=source_result if shape else source_result[()]
    )
    caller = ComponentProgramBuilder(signature)
    source_caller = source_builder(signature)
    if shape == (0, 2):
        # Source final_soqs omits empty arrays, so add_from fails at output lookup.
        with pytest.raises(KeyError, match="word"):
            source_caller.builder.add_from(original, **source_caller.inputs)
    else:
        source_output = source_caller.builder.add_from(original, **source_caller.inputs)
        expected = source_caller.builder.finalize(word=source_output[0])
    result = caller.inline_calls(record, caller.inputs)
    final = caller.finalize(result)
    assert verify_program_record(final) == ()
    assert np.asarray(result["word"]).shape == shape
    if shape != (0, 2):
        assert _native_trace(final) == _source_trace(source_caller, expected)


def test_mapping_matches_source_duplicates_unmapped_and_empty_arrays() -> None:
    from qualtran import BloqBuilder

    signature = Signature([Register("q", RegisterType("bit", 1), (3,))])
    native = ComponentProgramBuilder(signature)
    source = source_builder(signature)
    for values, mapper in (
        (np.asarray(native.inputs["q"]), map_register_bindings),
        (np.asarray(source.inputs["q"]), BloqBuilder.map_soqs),
    ):
        inputs = {"array": values, "scalar": values[0], "empty": values[:0]}
        output = mapper(inputs, [(values[0], values[1]), (values[0], values[2])])
        assert output["scalar"] == values[2]
        np.testing.assert_array_equal(output["array"], values[[2, 1, 2]])
        assert np.asarray(output["empty"]).shape == (0,)
        assert output["array"] is not values
        for replacement in (values[:2], values.reshape(3, 1)):
            with pytest.raises(AssertionError):
                mapper(inputs, [(values, replacement)])
        with pytest.raises(AssertionError):
            mapper(inputs, [(values[0], values)])


@pytest.mark.parametrize("conflict", [False, True])
def test_repeated_inline_reuses_or_isolates_definitions_and_executes(
    conflict: bool,
) -> None:
    import flagquantum as fq
    from flagquantum.simulation.unitary import get_unitary

    record = parallel_program()
    signature = Signature(record.functions[0].registers or ())
    definitions: tuple[FunctionRecord, ...] = ()
    if conflict:
        definitions = (
            bit_gate_function(
                "atom",
                (Register("q", RegisterType("bit", 1)),),
                CircuitIR(1, (Instruction("h", (0,)),)),
            ),
        )
    caller = ComponentProgramBuilder(signature, definitions)
    initial_record = repr(record)
    outputs = caller.inputs
    for _ in range(2):
        outputs = caller.inline_calls(record, {**outputs, "unused": object()})
    final = caller.finalize(outputs)
    assert repr(record) == initial_record
    calls = list(iter_component_calls(final))
    assert len(calls) == 10
    # Repeated imports share the same isolated definitions.
    assert len(final.functions) == (5 if conflict else 4)
    bound = bind_register_program(final, ((0, 1, 2),))
    assert bound.events == ()
    assert [gate.name for gate in bound.circuit.instructions] == ["x"] * 6

    class BoundProgram:
        def append_to(self, circuit: Circuit) -> Circuit:
            return circuit.compose(bound.circuit, qubits=(0, 1, 2))

    circuit = fq.Circuit(4)
    circuit.append(BoundProgram())
    state = torch.arange(16, dtype=torch.float64)
    state = torch.complex(state + 1, state.flip(0) + 3)
    state /= state.norm()
    actual = get_unitary(circuit.to_ir(), dtype=torch.complex128) @ state
    torch.testing.assert_close(actual, state, atol=0, rtol=0)


def test_inline_missing_input_preserves_builder_and_identity_passthrough() -> None:
    signature = Signature([Register("q", RegisterType("bit", 1))])
    identity = ComponentProgramBuilder(signature)
    record = identity.finalize(identity.inputs)
    caller = ComponentProgramBuilder(signature)
    with pytest.raises(KeyError, match="q"):
        caller.inline_calls(record, {})
    assert caller._operations == []
    outputs = caller.inline_calls(record, caller.inputs)
    assert outputs == caller.inputs
    assert verify_program_record(caller.finalize(outputs)) == ()
    empty = ComponentProgramBuilder(Signature([]))
    empty_record = empty.finalize({})
    target = ComponentProgramBuilder(Signature([]))
    assert target.inline_calls(empty_record, {}) == {}
    assert verify_program_record(target.finalize({})) == ()


def test_nested_calls_stay_nested_and_definition_conflicts_do_not_overwrite() -> None:
    record = parallel_program()
    signature = Signature(record.functions[0].registers or ())
    nested = replace(record.functions[0], name="nested")
    builder = ComponentProgramBuilder(signature, (nested, *record.functions[1:]))
    original = builder.finalize(builder.call_dict("nested", builder.inputs))
    caller = ComponentProgramBuilder(signature)
    final = caller.finalize(caller.inline_calls(original, caller.inputs))
    assert [final.functions[0].blocks[0].operations[0].callee] == ["nested"]
    assert len(list(iter_component_calls(final))) == 1


def test_directional_ports_preserve_allocation_and_release_ownership() -> None:
    from qualtran import BloqBuilder

    native = ComponentProgramBuilder(Signature([]))
    native.free(native.allocate(3))
    record = native.finalize({})
    source = BloqBuilder()
    source.free(source.allocate(3))
    original = source.finalize()
    source_target = BloqBuilder()
    assert source_target.add_from(original) == ()
    target = ComponentProgramBuilder(Signature([]))
    assert target.inline_calls(record, {}) == {}
    final = target.finalize({})
    assert len(list(iter_component_calls(final))) == len(original.bloq_instances) == 2
    bound = bind_register_program(final, (), clean=(0, 1, 2))
    assert [(event.kind, event.wires) for event in bound.events] == [
        ("allocate", (0, 1, 2)),
        ("free", (0, 1, 2)),
    ]
    assert bound.circuit.instructions == ()


@pytest.mark.parametrize("duplicate", [False, True])
def test_crossed_cnot_outputs_and_failed_consumption_match_source(
    duplicate: bool,
) -> None:
    from qualtran.bloqs.basic_gates import CNOT

    signature = Signature.build(q0=1, q1=1)
    gate = bit_gate_function(
        "CNOT",
        (
            Register("ctrl", RegisterType("bit", 1)),
            Register("target", RegisterType("bit", 1)),
        ),
        CircuitIR(2, (Instruction("cx", (0, 1)),)),
    )
    native = ComponentProgramBuilder(signature, (gate,))
    source = source_builder(signature)
    q0, q1 = native.inputs.values()
    s0, s1 = source.inputs.values()
    for reverse in (False, True):
        q0, q1 = native.call_tuple(
            "CNOT", {"ctrl": q1 if reverse else q0, "target": q0 if reverse else q1}
        )
        s0, s1 = source.builder.add_t(
            CNOT(), ctrl=s1 if reverse else s0, target=s0 if reverse else s1
        )
    record = native.finalize({"q0": q0, "q1": q1})
    original = source.builder.finalize(q0=s0, q1=s1)
    caller = ComponentProgramBuilder(signature)
    source_caller = source_builder(signature)
    native_args, source_args = dict(caller.inputs), dict(source_caller.inputs)
    if duplicate:
        native_args["q1"] = native_args["q0"]
        source_args["q1"] = source_args["q0"]
        with pytest.raises(RegisterBindingError):
            caller.inline_calls(record, native_args)
        with pytest.raises(source_caller.error_type):
            source_caller.builder.add_from(original, **source_args)
        assert len(caller._operations) == len(source_caller.instances) == 1
        assert (
            len(caller._operations[0].operands) == len(source_caller.connections) == 1
        )
        assert len(caller._available) == len(source_caller.available) == 1
        return
    output = caller.inline_calls(record, native_args)
    source_output = source_caller.builder.add_from(original, **source_args)
    expected = source_caller.builder.finalize(q0=source_output[0], q1=source_output[1])
    final = caller.finalize(output)
    assert _native_trace(final) == _source_trace(source_caller, expected)
    bound = bind_register_program(final, ((0,), (1,)))
    assert bound.circuit.instructions == (
        Instruction("cx", (0, 1)),
        Instruction("cx", (1, 0)),
    )
    assert bound.outputs == ((1,), (0,))
    from flagquantum.simulation.unitary import get_unitary

    values = torch.arange(8, dtype=torch.int64)
    state = torch.complex(values.to(torch.float64) + 1, 9 - values.to(torch.float64))
    state /= state.norm()
    actual = (
        get_unitary(bound.circuit, dtype=torch.complex128) @ state.reshape(4, 2)
    ).flatten()
    a, b, spectator = values >> 2, (values >> 1) & 1, values & 1
    destination = (b << 2) | ((a ^ b) << 1) | spectator
    expected_state = torch.empty_like(state)
    expected_state[destination] = state
    torch.testing.assert_close(actual, expected_state, atol=0, rtol=0)
