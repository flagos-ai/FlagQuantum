"""Replace flat emitter output with native finite-call expansion at its boundary."""

from dataclasses import replace
from typing import Any

import pytest

from flagquantum.compiler.registers.program_calls import inline_program_calls
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import (
    BlockRecord,
    ConditionRef,
    FunctionRecord,
    Operation,
    ProgramRecord,
    ValueRef,
    program_to_circuit,
    verify_program_record,
)
from tests.unit.program_call_fixtures import explicit_block, wrap_circuit

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("depth", [1, 3, 1300])
def test_finite_call_expansion_preserves_exact_payload_and_handles_deep_calls(
    depth: int,
) -> None:
    circuit = CircuitIR(2, (Instruction("h", (0,)), Instruction("cx", (0, 1))))
    hierarchy = wrap_circuit(circuit, depth)
    expanded = inline_program_calls(hierarchy)
    assert verify_program_record(expanded) == ()
    assert len(expanded.functions) == 1
    assert expanded.functions[0].blocks[0] == explicit_block(circuit)
    assert program_to_circuit(expanded).canonical_payload == circuit.to_dict()
    assert any(
        op.opcode == "call"
        for function in hierarchy.functions
        for op in function.blocks[0].operations
    )
    assert all(op.callee is None for op in expanded.functions[0].blocks[0].operations)


def test_parameter_and_measurement_feedback_cross_actual_function_boundaries() -> None:
    from flagquantum.dynamic import DynamicCircuit

    circuit = DynamicCircuit(2)
    circuit.h(0)
    circuit.measure(0, classical_bit=0)
    circuit.rz(1, 0.375)
    circuit.conditional("x", 1, conditions={0: 1})
    public = circuit.to_ir()
    q = ValueRef("entry", 0, "qubit")
    bit = ValueRef("entry", 1, "bool")
    scalar = ValueRef("entry", 1, "scalar")
    out = ValueRef("entry", 2, "qubit")
    measure = FunctionRecord(
        "measure_leaf",
        (
            BlockRecord(
                "entry",
                (q,),
                operations=(Operation("measure", (q,), (out, bit)),),
                return_values=(out, bit),
            ),
        ),
    )
    rotate = FunctionRecord(
        "rotate_leaf",
        (
            BlockRecord(
                "entry",
                (q, scalar),
                operations=(
                    Operation("rz", (q,), (out,), parameters=(("theta", scalar),)),
                ),
                return_values=(out,),
            ),
        ),
    )
    feedback = FunctionRecord(
        "feedback_leaf",
        (
            BlockRecord(
                "entry",
                (q, bit),
                operations=(
                    Operation(
                        "x",
                        (q,),
                        (out,),
                        condition_clauses=((ConditionRef(bit, True),),),
                    ),
                ),
                return_values=(out,),
            ),
        ),
    )
    v = [
        ValueRef("entry", i, kind)
        for i, kind in enumerate(
            ("qubit", "qubit", "qubit", "qubit", "bool", "scalar", "qubit", "qubit")
        )
    ]
    main = FunctionRecord(
        "main",
        (
            BlockRecord(
                "entry",
                tuple(v[:2]),
                operations=(
                    Operation("h", (v[0],), (v[2],)),
                    Operation("call", (v[2],), (v[3], v[4]), callee="measure_leaf"),
                    Operation("constant", results=(v[5],), literal=0.375),
                    Operation("call", (v[1], v[5]), (v[6],), callee="rotate_leaf"),
                    Operation("call", (v[6], v[4]), (v[7],), callee="feedback_leaf"),
                ),
                return_values=(v[3], v[7]),
            ),
        ),
    )
    module = ProgramRecord(
        (main, measure, rotate, feedback), canonical_payload=public.to_dict()
    )
    assert verify_program_record(module) == ()
    expanded = inline_program_calls(module)
    assert expanded.functions[0].blocks[0] == explicit_block(public)
    assert program_to_circuit(expanded).canonical_payload == public.to_dict()


def test_repeated_leaf_calls_use_independent_ssa_environments() -> None:
    leaf_circuit = CircuitIR(1, (Instruction("x", (0,)),))
    module = wrap_circuit(leaf_circuit)
    main, leaf = module.functions
    block = main.blocks[0]
    first = block.operations[0]
    result = ValueRef("entry", 2, "qubit")
    second = Operation("call", first.results, (result,), callee="leaf")
    main = replace(
        main,
        blocks=(replace(block, operations=(first, second), return_values=(result,)),),
    )
    expected = CircuitIR(1, (Instruction("x", (0,)), Instruction("x", (0,))))
    expanded = inline_program_calls(
        ProgramRecord((main, leaf), canonical_payload=expected.to_dict())
    )
    assert expanded.functions[0].blocks[0] == explicit_block(expected)
    assert program_to_circuit(expanded).canonical_payload == expected.to_dict()


def test_inliner_rejects_bad_input_before_expanding_and_checks_entry() -> None:
    module = wrap_circuit(CircuitIR(1, ()))
    with pytest.raises(ValueError, match="unknown Program entry"):
        inline_program_calls(module, "absent")
    main, leaf = module.functions
    operation = replace(main.blocks[0].operations[0], callee="absent")
    main = replace(main, blocks=(replace(main.blocks[0], operations=(operation,)),))
    with pytest.raises(ValueError, match="invalid Program calls"):
        inline_program_calls(replace(module, functions=(main, leaf)))


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("consumer", ["add", "xor_qrom"])
@pytest.mark.parametrize("factory_created", [False, True])
def test_existing_append_branch_resource_consumer_accepts_hierarchical_replacement(
    monkeypatch: pytest.MonkeyPatch, consumer: str, factory_created: bool
) -> None:
    from flagquantum.compiler.registers.arithmetic import addition
    from flagquantum.compiler.registers.lookup.qrom_table import QromTable
    from tests.unit.test_register_bindings import (
        NativeContract,
        test_bindings_feed_append_with_exact_measured_channels_and_resources,
    )

    seen = []
    original = addition.add_ir if consumer == "add" else QromTable.bind_packed

    def through_calls(*args: Any, **kwargs: Any) -> tuple[Instruction, ...]:
        gates = original(*args, **kwargs)
        width = 1 + max(wire for gate in gates for wire in gate.wires)
        circuit = CircuitIR(width, gates)
        hierarchical = wrap_circuit(circuit, 3)
        expanded = inline_program_calls(hierarchical)
        restored = program_to_circuit(expanded)
        assert restored.state == "supported_exact", restored.diagnostics
        assert restored.canonical_payload is not None
        result = CircuitIR.from_dict(restored.canonical_payload)
        assert result.instructions == gates
        seen.append((hierarchical, expanded))
        return result.instructions

    if consumer == "add":
        monkeypatch.setattr(addition, "add_ir", through_calls)
    else:
        monkeypatch.setattr(QromTable, "bind_packed", through_calls)
    # This existing consumer is unchanged. It checks the original source gate
    # dependencies/operands/feedback, all four complex branches, zero workspace
    # and primitive counts independently of the upstream QEC model.
    test_bindings_feed_append_with_exact_measured_channels_and_resources(
        NativeContract, consumer, factory_created
    )
    assert len(seen) == 1


@pytest.mark.parametrize(
    "aliased,contradiction", [(False, False), (True, False), (True, True)]
)
def test_boolean_argument_remapping_handles_order_aliases_and_contradictions(
    aliased: bool, contradiction: bool
) -> None:
    q = ValueRef("entry", 0, "qubit")
    a, b = (ValueRef("entry", i, "bool") for i in (1, 2))
    out = ValueRef("entry", 3, "qubit")
    conditions = ((ConditionRef(a, True), ConditionRef(b, not contradiction)),)
    leaf = FunctionRecord(
        "leaf",
        (
            BlockRecord(
                "entry",
                (q, a, b),
                operations=(
                    Operation("x", (q,), (out,), condition_clauses=conditions),
                ),
                return_values=(out,),
            ),
        ),
    )
    call = Operation("call", (q, b, b if aliased else a), (out,), callee="leaf")
    main = FunctionRecord(
        "main",
        (BlockRecord("entry", (q, a, b), operations=(call,), return_values=(out,)),),
    )
    module = ProgramRecord((main, leaf))
    assert verify_program_record(module) == ()
    expanded = inline_program_calls(module)
    assert verify_program_record(expanded) == ()
    operations = expanded.functions[0].blocks[0].operations
    if contradiction:
        assert operations == ()
        assert expanded.functions[0].blocks[0].return_values == (q,)
    else:
        assert len(operations) == 1
        refs = operations[0].condition_clauses[0]
        assert tuple(condition.value.index for condition in refs) == (
            (2,) if aliased else (1, 2)
        )
