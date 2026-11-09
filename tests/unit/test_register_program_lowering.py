"""Replace existing append consumers with typed register to gate expansion."""

from dataclasses import replace
from typing import Any

import pytest

from flagquantum.compiler.registers.program_calls import inline_program_calls
from flagquantum.compiler.registers.register_programs import lower_register_program
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import (
    BlockRecord,
    FunctionRecord,
    Operation,
    ProgramRecord,
    ValueRef,
    program_to_circuit,
    verify_program_record,
)
from flagquantum.core.ir.program.registers import Register, RegisterType
from tests.unit.test_program_calls import explicit_block
from tests.unit.test_program_registers import word_function

pytestmark = pytest.mark.unit


def packed_circuit(
    circuit: CircuitIR, widths: tuple[int, ...], depth: int = 1
) -> ProgramRecord:
    assert sum(widths) == circuit.n_wires
    leaf = FunctionRecord("gate_leaf", (explicit_block(circuit),))
    registers = tuple(
        Register(f"word{index}", RegisterType("uint", width))
        for index, width in enumerate(widths)
    )
    arguments = tuple(
        ValueRef("entry", index, "register") for index in range(len(widths))
    )
    next_index = len(arguments)
    operands = []
    operations = []
    for value, width in zip(arguments, widths, strict=True):
        bits = tuple(
            ValueRef("entry", next_index + index, "qubit") for index in range(width)
        )
        next_index += width
        operands.extend(bits)
        operations.append(Operation("split_register", (value,), bits))
    outputs = tuple(
        ValueRef("entry", next_index + index, "qubit")
        for index in range(circuit.n_wires)
    )
    next_index += circuit.n_wires
    operations.append(Operation("call", tuple(operands), outputs, callee="gate_leaf"))
    returned = []
    offset = 0
    for register, width in zip(registers, widths, strict=True):
        word = ValueRef("entry", next_index, "register")
        next_index += 1
        operations.append(
            Operation(
                "join_register",
                outputs[offset : offset + width],
                (word,),
                register_type=register.dtype,
            )
        )
        offset += width
        returned.append(word)
    functions = [
        leaf,
        FunctionRecord(
            "packed_leaf",
            (
                BlockRecord(
                    "entry",
                    arguments,
                    operations=tuple(operations),
                    return_values=tuple(returned),
                ),
            ),
            registers=registers,
        ),
    ]
    previous = "packed_leaf"
    for level in range(depth):
        name = "main" if level == depth - 1 else f"layer{level}"
        results = tuple(
            ValueRef("entry", len(arguments) + index, "register")
            for index in range(len(arguments))
        )
        block = BlockRecord(
            "entry",
            arguments,
            operations=(Operation("call", arguments, results, callee=previous),),
            return_values=results,
        )
        functions.append(FunctionRecord(name, (block,), registers=registers))
        previous = name
    return ProgramRecord(
        tuple(reversed(functions)), canonical_payload=circuit.to_dict()
    )


@pytest.mark.parametrize("widths", [(3,), (1, 2), (0, 3, 0), (1, 1, 1)])
@pytest.mark.parametrize("depth", [1, 3])
def test_packed_words_and_calls_restore_the_exact_existing_gate_graph(widths, depth):
    circuit = CircuitIR(
        3, (Instruction("h", (0,)), Instruction("cx", (0, 2)), Instruction("t", (1,)))
    )
    packed = packed_circuit(circuit, widths, depth)
    assert verify_program_record(packed) == ()
    with pytest.raises(ValueError, match="lower_register_program"):
        inline_program_calls(packed)
    expanded = lower_register_program(packed)
    assert verify_program_record(expanded) == ()
    assert expanded.functions[0].registers is None
    assert expanded.functions[0].blocks[0] == explicit_block(circuit)
    restored = program_to_circuit(expanded)
    assert restored.state == "supported_exact", restored.diagnostics
    assert restored.canonical_payload == circuit.to_dict()


def test_deep_register_call_chain_reuses_the_finite_stack_inliner():
    circuit = CircuitIR(1, (Instruction("x", (0,)),))
    expanded = lower_register_program(packed_circuit(circuit, (0, 1), 1300))
    assert expanded.functions[0].blocks[0] == explicit_block(circuit)


@pytest.mark.parametrize("width", [0, 1, 3])
def test_register_rebindings_emit_no_physical_operation(width):
    function = word_function(width)
    expanded = lower_register_program(ProgramRecord((function,)))
    block = expanded.functions[0].blocks[0]
    assert len(block.arguments) == width
    assert block.operations == ()
    assert block.return_values == block.arguments


@pytest.mark.parametrize("shape", [(), (2,), (2, 3), (0, 2)])
@pytest.mark.parametrize("width", [0, 2])
def test_array_elements_lower_in_declared_row_major_order(shape, width):
    register = Register("words", RegisterType("uint", width), shape)
    values = tuple(
        ValueRef("entry", index, "register")
        for index, _ in enumerate(register.all_idxs())
    )
    function = FunctionRecord(
        "main",
        (BlockRecord("entry", values, return_values=values),),
        registers=(register,),
    )
    block = lower_register_program(ProgramRecord((function,))).functions[0].blocks[0]
    assert len(block.arguments) == len(values) * width
    assert block.return_values == block.arguments
    assert tuple(value.index for value in block.arguments) == tuple(
        range(len(values) * width)
    )


def test_output_permutation_is_retained_and_cannot_hide_behind_payload():
    function = word_function(3)
    block = function.blocks[0]
    split, join = block.operations
    reordered = replace(join, operands=tuple(reversed(join.operands)))
    function = replace(
        function, blocks=(replace(block, operations=(split, reordered)),)
    )
    circuit = CircuitIR(3, ())
    packed = ProgramRecord((function,), canonical_payload=circuit.to_dict())
    lowered = lower_register_program(packed)
    body = lowered.functions[0].blocks[0]
    assert body.operations == ()
    assert body.return_values == tuple(reversed(body.arguments))
    assert program_to_circuit(lowered).state == "unsupported_with_diagnostics"


def test_invalid_zero_width_ownership_and_unreachable_types_fail_before_erasure():
    function = word_function(0)
    block = function.blocks[0]
    changed = replace(function, blocks=(replace(block, return_values=()),))
    with pytest.raises(ValueError, match="linear_unconsumed"):
        lower_register_program(ProgramRecord((changed,)))
    bad = replace(
        word_function(1, "unreachable"),
        registers=(Register("x", RegisterType("signed", 1)),),
    )
    with pytest.raises(ValueError, match="register_type"):
        lower_register_program(ProgramRecord((function, bad)))


def test_primitive_parameters_and_measurement_feedback_survive_packed_boundary():
    from flagquantum.dynamic import DynamicCircuit

    circuit = DynamicCircuit(2)
    circuit.h(0)
    circuit.measure(0, classical_bit=0)
    circuit.rz(1, 0.375)
    circuit.conditional("x", 1, conditions={0: 1})
    public = circuit.to_ir()
    packed = packed_circuit(public, (1, 1))
    # Put the gate body directly inside a register function to exercise local
    # scalar and measurement-result maps, as well as the gate-call path.
    leaf = next(
        function for function in packed.functions if function.name == "gate_leaf"
    )
    body = next(
        function for function in packed.functions if function.name == "packed_leaf"
    )
    assert leaf.blocks[0].return_values is not None
    block = body.blocks[0]
    call = next(op for op in block.operations if op.opcode == "call")
    values = dict(zip(leaf.blocks[0].arguments, call.operands, strict=True))
    current = max(value.index for op in block.operations for value in op.results) + 1
    inlined = []
    from flagquantum.compiler.registers.program_calls import _remap_conditions

    for op in leaf.blocks[0].operations:
        results = tuple(
            ValueRef("entry", current + i, value.kind)
            for i, value in enumerate(op.results)
        )
        current += len(results)
        inlined.append(
            replace(
                op,
                operands=tuple(values[value] for value in op.operands),
                results=results,
                parameters=tuple(
                    (name, values[value]) for name, value in op.parameters
                ),
                condition_clauses=_remap_conditions(op.condition_clauses, values),
            )
        )
        values.update(zip(op.results, results, strict=True))
    values.update(
        zip(
            call.results,
            (values[value] for value in leaf.blocks[0].return_values),
            strict=True,
        )
    )
    operations = []
    for op in block.operations:
        if op.opcode == "call":
            operations.extend(inlined)
        else:
            operations.append(
                replace(
                    op,
                    operands=tuple(values.get(value, value) for value in op.operands),
                )
            )
    changed = replace(body, blocks=(replace(block, operations=tuple(operations)),))
    module = replace(
        packed,
        functions=tuple(
            changed if f.name == "packed_leaf" else f for f in packed.functions
        ),
    )
    result = lower_register_program(module)
    assert result.functions[0].blocks[0] == explicit_block(public)
    assert program_to_circuit(result).state == "supported_exact"


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("consumer", ["add", "xor_qrom"])
@pytest.mark.parametrize("factory_created", [False, True])
def test_existing_append_consumer_accepts_packed_rebinding_replacement(
    monkeypatch, consumer, factory_created
):
    from flagquantum.compiler.registers.arithmetic import addition
    from flagquantum.compiler.registers.lookup.qrom_table import QromTable
    from tests.unit.test_register_bindings import (
        NativeContract,
        test_bindings_feed_append_with_exact_measured_channels_and_resources,
    )

    original = addition.add_ir if consumer == "add" else QromTable.bind_packed
    seen = []

    def through_words(*args: Any, **kwargs: Any) -> tuple[Instruction, ...]:
        gates = original(*args, **kwargs)
        width = 1 + max(wire for gate in gates for wire in gate.wires)
        widths = (2, 3, 2, 0) if consumer == "add" else (2, 2, 1, 0)
        assert sum(widths) == width
        circuit = CircuitIR(width, gates)
        hierarchy = packed_circuit(circuit, widths, 3)
        assert verify_program_record(hierarchy) == ()
        expanded = lower_register_program(hierarchy)
        restored = program_to_circuit(expanded)
        assert restored.state == "supported_exact", restored.diagnostics
        result = CircuitIR.from_dict(restored.canonical_payload)
        assert result.instructions == gates
        seen.append(hierarchy)
        return result.instructions

    if consumer == "add":
        monkeypatch.setattr(addition, "add_ir", through_words)
    else:
        monkeypatch.setattr(QromTable, "bind_packed", through_words)
    test_bindings_feed_append_with_exact_measured_channels_and_resources(
        NativeContract, consumer, factory_created
    )
    assert len(seen) == 1


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("permutation", [(0, 1, 2), (2, 0, 1)])
def test_rebound_output_matches_original_source_complex_tensor_with_spectator(
    permutation,
):
    import numpy as np
    import torch
    from qualtran import BloqBuilder, QAny, Side, Signature
    from qualtran import Register as SourceRegister
    from qualtran.bloqs.basic_gates import XGate
    from qualtran.bloqs.bookkeeping import Join, Split

    import flagquantum as fq

    source_dtype = QAny(3)
    builder, inputs = BloqBuilder.from_signature(
        Signature(
            [
                SourceRegister("word", source_dtype, side=Side.LEFT),
                SourceRegister("out", source_dtype, side=Side.RIGHT),
            ]
        )
    )
    source_bits = builder.add(Split(source_dtype), reg=inputs["word"])
    source_bits[1] = builder.add(XGate(), q=source_bits[1])
    source_out = builder.add(Join(source_dtype), reg=source_bits[list(permutation)])
    source_tensor = torch.as_tensor(
        builder.finalize(out=source_out).tensor_contract(), dtype=torch.complex128
    ).reshape(8, 8)

    function = word_function(3)
    block = function.blocks[0]
    split, join = block.operations
    changed_bit = ValueRef("entry", 10, "qubit")
    gate = Operation("x", (split.results[1],), (changed_bit,))
    bits = (split.results[0], changed_bit, split.results[2])
    joined = replace(join, operands=tuple(bits[i] for i in permutation))
    function = replace(
        function, blocks=(replace(block, operations=(split, gate, joined)),)
    )
    lowered = lower_register_program(ProgramRecord((function,)))
    body = lowered.functions[0].blocks[0]
    locations = {value: index for index, value in enumerate(body.arguments)}
    physical = (3, 1, 0)
    instructions = []
    for op in body.operations:
        wires = tuple(locations[value] for value in op.operands)
        instructions.append(
            Instruction(op.opcode, tuple(physical[wire] for wire in wires))
        )
        locations.update(
            (after, wire) for after, wire in zip(op.results, wires, strict=True)
        )
    output = tuple(physical[locations[value]] for value in body.return_values)
    assert output == tuple(physical[index] for index in permutation)
    rows = torch.arange(1, 17, dtype=torch.float64)
    state = torch.complex(rows, rows.remainder(7) - 3)
    state /= state.norm()
    circuit = fq.Circuit(4, inputs=state, dtype=state.dtype)
    from examples.algorithms.binary_field_arithmetic import FieldOperation

    circuit.append(FieldOperation(4, tuple(instructions)))
    actual = circuit.state().reshape((2,) * 4).permute(2, *output).reshape(2, 8)
    expected_input = state.reshape((2,) * 4).permute(2, *physical).reshape(2, 8)
    torch.testing.assert_close(
        actual, expected_input @ source_tensor.T, atol=2e-12, rtol=0
    )
    assert tuple(gate.name for gate in circuit.to_ir().instructions) == ("x",)
    assert np.asarray(source_bits).shape == (3,)
