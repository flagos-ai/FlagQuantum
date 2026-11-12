"""Bind actual finite lifetime graphs through unchanged append consumers."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import replace
from itertools import product
from typing import Any

import numpy as np
import pytest
import torch

from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_schedule import local_operation_order
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import (
    BlockRecord,
    FunctionRecord,
    Operation,
    ProgramRecord,
    ValueRef,
    verify_program_record,
)
from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.measurement_checks import forced_branch
from tests.unit.program_call_fixtures import explicit_block, wrap_circuit
from tests.unit.register_binding_fixtures import allocated_circuit
from tests.unit.test_program_registers import word_function
from tests.unit.test_register_program_lowering import packed_circuit

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("consumer", ["add", "xor_qrom"])
@pytest.mark.parametrize("factory_created", [False, True])
def test_existing_append_consumer_uses_actual_private_lifetimes(
    monkeypatch: pytest.MonkeyPatch, consumer: str, factory_created: bool
) -> None:
    from flagquantum.compiler.registers.arithmetic import addition
    from flagquantum.compiler.registers.lookup.qrom_table import QromTable
    from tests.unit.test_register_bindings import (
        NativeContract,
        test_bindings_feed_append_with_exact_measured_channels_and_resources,
    )
    from tests.unit.test_register_subtraction import assert_source

    original = addition.add_ir if consumer == "add" else QromTable.bind_packed
    seen = []

    def through_lifetimes(*args: Any, **kwargs: Any) -> tuple[Instruction, ...]:
        gates = original(*args, **kwargs)
        width = max(wire for gate in gates for wire in gate.wires) + 1
        data = 5 if consumer == "add" else 4
        module = allocated_circuit(CircuitIR(width, gates), data)
        bound = bind_register_program(
            module, (tuple(range(data)),), clean=tuple(range(data, width))
        )
        assert bound.outputs == (tuple(range(data)),)
        assert [(event.kind, event.wires) for event in bound.events] == [
            ("allocate", tuple(range(data, width))),
            ("free", tuple(range(data, width))),
        ]
        # Program bool SSA uses fresh local slots. Bind those handles back to
        # this unchanged emitter consumer's wire-associated classical slots.
        expected_slots = {
            gate.wires[0]: gate.metadata["classical_bit"]
            for gate in gates
            if gate.name == "measure"
        }
        slots = {
            gate.metadata["classical_bit"]: expected_slots[gate.wires[0]]
            for gate in bound.circuit.instructions
            if gate.name == "measure"
        }
        rebound = []
        for gate in bound.circuit.instructions:
            metadata = dict(gate.metadata)
            if gate.name == "measure":
                metadata["classical_bit"] = slots[metadata["classical_bit"]]
            if "conditions" in metadata:
                metadata["conditions"] = tuple(
                    (slots[bit], value) for bit, value in metadata["conditions"]
                )
            rebound.append(replace(gate, metadata=metadata))
        bound = replace(
            bound, circuit=replace(bound.circuit, instructions=tuple(rebound))
        )
        # Source comparison permits only dependency-equivalent independent
        # primitive order, retaining exact operands, feedback and gate counts.
        assert_source(bound.circuit.instructions, gates)
        assert Counter(gate.name for gate in bound.circuit.instructions) == Counter(
            gate.name for gate in gates
        )
        seen.append(bound)
        return bound.circuit.instructions

    if consumer == "add":
        monkeypatch.setattr(addition, "add_ir", through_lifetimes)
    else:
        monkeypatch.setattr(QromTable, "bind_packed", through_lifetimes)
    test_bindings_feed_append_with_exact_measured_channels_and_resources(
        NativeContract, consumer, factory_created
    )
    assert len(seen) == 1


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("dirty,free_dirty", list(product((False, True), repeat=2)))
def test_source_lifetimes_preserve_entangled_dirty_work_and_clean_zero(
    dirty: bool, free_dirty: bool
) -> None:
    import cirq
    from qualtran import BloqBuilder, QBit
    from qualtran import Signature as SourceSignature
    from qualtran.bloqs.basic_gates import CNOT
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    # The packed CNOT is a real executable component, including its word ports.
    module = packed_circuit(CircuitIR(2, (Instruction("cx", (0, 1)),)), (1, 1))
    definitions = tuple(
        replace(f, name="cnot") if f.name == "main" else f for f in module.functions
    )
    builder = ComponentProgramBuilder(
        Signature([Register("data", RegisterType("uint", 1))]), definitions
    )
    work = builder.allocate(dtype=RegisterType("uint", 1), dirty=dirty)
    data = builder.inputs["data"]
    for _ in range(2):
        data, work = builder.call_tuple("cnot", {"word0": data, "word1": work})
    builder.free(work, dirty=free_dirty)
    native = builder.finalize({"data": data})
    bound = bind_register_program(
        native, ((0,),), clean=() if dirty else (1,), borrowed=(1,) if dirty else ()
    )

    events = []

    class Manager(cirq.SimpleQubitManager):
        def qalloc(self, n: int, dim: int = 2) -> list[cirq.LineQubit]:
            assert n == 1 and dim == 2
            events.append(("allocate", (1,), False))
            return [cirq.LineQubit(1)]

        def qborrow(self, n: int, dim: int = 2) -> list[cirq.LineQubit]:
            assert n == 1 and dim == 2
            events.append(("allocate", (1,), True))
            return [cirq.LineQubit(1)]

        def qfree(self, qubits: Iterable[cirq.LineQubit]) -> None:
            events.append(("free", tuple(q.x for q in qubits), free_dirty))

    source, inputs = BloqBuilder.from_signature(SourceSignature.build(data=1))
    source_data, source_work = inputs["data"], source.allocate(
        dtype=QBit(), dirty=dirty
    )
    for _ in range(2):
        source_data, source_work = source.add(
            CNOT(), ctrl=source_data, target=source_work
        )
    source.free(source_work, dirty=free_dirty)
    composite = source.finalize(data=source_data)
    reference, outputs = composite.to_cirq_circuit_and_quregs(
        qubit_manager=Manager(), data=np.array([cirq.LineQubit(0)])
    )
    assert [(event.kind, event.wires, event.dirty) for event in bound.events] == events
    assert tuple(outputs["data"]) == (cirq.LineQubit(0),)
    assert [(gate.name, gate.wires) for gate in bound.circuit.instructions] == [
        ("cx", (0, 1)),
        ("cx", (0, 1)),
    ]
    assert (
        tuple(reference.all_operations()) == (cirq.CNOT(*cirq.LineQubit.range(2)),) * 2
    )
    state = torch.tensor([1, 2j, -3j, 4, 2 + 1j, -1, 3, -2j], dtype=torch.complex128)
    if not dirty:
        state.reshape(2, 2, 2)[:, 1, :] = 0
    state /= state.norm()
    actual = forced_branch(state, bound.circuit.instructions, ())
    expected = (
        cirq.Simulator(dtype=np.complex128)
        .simulate(
            reference, initial_state=state.numpy(), qubit_order=cirq.LineQubit.range(3)
        )
        .final_state_vector
    )
    torch.testing.assert_close(actual, torch.as_tensor(expected), rtol=0, atol=2e-12)
    torch.testing.assert_close(actual, state, rtol=0, atol=2e-12)
    if not dirty:
        assert torch.count_nonzero(actual.reshape(2, 2, 2)[:, 1, :]) == 0
    # Physical and model counts are independently observed.
    assert Counter(gate.name for gate in bound.circuit.instructions) == {"cx": 2}
    model = get_cost_value(composite, QECGatesCost())
    assert model.clifford == 2 and model.total_t_count() == model.measurement == 0


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("case", ["partial", "joined", "reuse"])
@pytest.mark.parametrize("dirty", [False, True])
def test_actual_builder_schedule_matches_original_connection_order(
    case: str, dirty: bool
) -> None:
    from qualtran import BloqBuilder, QAny
    from qualtran import Signature as SourceSignature
    from qualtran._infra.binst_graph_iterators import greedy_topological_sort
    from qualtran._infra.quantum_graph import LeftDangle, RightDangle

    native = ComponentProgramBuilder(
        Signature([Register("data", RegisterType("opaque", 1))])
    )
    source, source_inputs = BloqBuilder.from_signature(SourceSignature.build(data=1))
    for builder, dtype in ((native, RegisterType("opaque", 3)), (source, QAny(3))):
        first = builder.allocate(dtype=dtype, dirty=dirty)
        bits = builder.split(first)
        if case == "partial":
            for bit in reversed(bits):
                builder.free(bit, dirty=dirty)
        else:
            second = builder.allocate(
                dtype=RegisterType("opaque", 1) if builder is native else QAny(1),
                dirty=dirty,
            )
            joined = builder.join([*builder.split(second), *bits[::-1]])
            builder.free(joined, dirty=dirty)
            if case == "reuse":
                again = builder.allocate(dtype=dtype, dirty=dirty)
                builder.free(again, dirty=dirty)
    record = native.finalize(native.inputs)
    composite = source.finalize(**source_inputs)
    order = tuple(
        node.i
        for node in greedy_topological_sort(composite._binst_graph)
        if node not in (LeftDangle, RightDangle)
    )
    functions = {f.name: f for f in record.functions}
    assert local_operation_order(record.functions[0], functions) == order
    pool = (4, 2, 7, 1)
    bound = bind_register_program(
        record, ((0,),), clean=() if dirty else pool, borrowed=pool if dirty else ()
    )
    import cirq

    available = list(reversed(pool))
    source_events = []

    class Manager(cirq.SimpleQubitManager):
        def qalloc(self, n: int, dim: int = 2) -> list[cirq.LineQubit]:
            assert dim == 2
            wires = tuple(available.pop() for _ in range(n))
            source_events.append(("allocate", wires))
            return [cirq.LineQubit(wire) for wire in wires]

        def qborrow(self, n: int, dim: int = 2) -> list[cirq.LineQubit]:
            return self.qalloc(n, dim)

        def qfree(self, qubits: Iterable[cirq.LineQubit]) -> None:
            wires = tuple(qubit.x for qubit in qubits)
            available.extend(reversed(wires))
            source_events.append(("free", wires))

    reference, returned = composite.to_cirq_circuit_and_quregs(
        qubit_manager=Manager(), data=np.array([cirq.LineQubit(0)])
    )
    assert not tuple(reference.all_operations())
    assert tuple(returned["data"]) == (cirq.LineQubit(0),)
    assert [(event.kind, event.wires) for event in bound.events] == source_events
    assert bound.outputs == ((0,),) and bound.circuit.instructions == ()
    assert bound.events[0].wires == pool[:3]
    if case == "partial":
        assert {event.wires for event in bound.events[1:]} == {
            (wire,) for wire in pool[:3]
        }
    else:
        assert len(bound.events[2].wires) == 4
        if case == "reuse":
            assert bound.events[3].wires == bound.events[2].wires[:3]


@pytest.mark.parametrize("width", [0, 1, 3])
def test_zero_width_lifetimes_and_output_bit_permutation(width: int) -> None:
    builder = ComponentProgramBuilder(
        Signature([Register("data", RegisterType("opaque", 3))])
    )
    empty = builder.allocate(0)
    builder.free(builder.join(builder.split(empty)))
    work = builder.allocate(width)
    builder.free(work)
    data = builder.join(builder.split(builder.inputs["data"])[::-1])
    bound = bind_register_program(
        builder.finalize({"data": data}), ((4, 2, 0),), clean=(1, 3, 5)
    )
    assert bound.outputs == ((0, 2, 4),)
    assert bound.circuit.instructions == ()
    assert len(bound.events) == 4


@pytest.mark.parametrize("depth", [1, 3, 1300])
def test_deep_local_calls_and_payload_do_not_replace_actual_graph(depth: int) -> None:
    circuit = CircuitIR(2, (Instruction("h", (0,)), Instruction("cx", (0, 1))))
    record = wrap_circuit(circuit, depth)
    record = replace(
        record, canonical_payload=CircuitIR(2, (Instruction("x", (1,)),)).to_dict()
    )
    bound = bind_register_program(record, ((2,), (0,)))
    assert bound.outputs == ((2,), (0,))
    assert bound.circuit.instructions == (
        Instruction("h", (2,)),
        Instruction("cx", (2, 0)),
    )


def test_repeated_measurement_reset_and_parity_have_fresh_classical_slots() -> None:
    from flagquantum.core.classical_dataflow import ClassicalXor
    from flagquantum.core.ir.program import ConditionRef
    from flagquantum.dynamic import DynamicCircuit

    dynamic = DynamicCircuit(2)
    dynamic.h(0)
    dynamic.measure(0, classical_bit=0)
    dynamic.rz(1, 0.375)
    dynamic.conditional("x", 1, conditions={0: 1})
    dynamic.reset(0)
    leaf = explicit_block(dynamic.to_ir())
    measured = next(op.results[-1] for op in leaf.operations if op.opcode == "measure")
    one, parity = ValueRef("entry", 100, "bool"), ValueRef("entry", 101, "bool")
    operations = []
    for op in leaf.operations:
        if op.opcode == "x":
            operations.extend(
                (
                    Operation("bool_one", results=(one,)),
                    Operation("bool_xor", (measured, one), (parity,)),
                )
            )
            op = replace(op, condition_clauses=((ConditionRef(parity, False),),))
        operations.append(op)
    leaf = replace(leaf, operations=tuple(operations))
    record = wrap_circuit(dynamic.to_ir())
    main, definition = record.functions
    first = main.blocks[0].operations[0]
    outputs = tuple(ValueRef("entry", 20 + i, "qubit") for i in range(2))
    main = replace(
        main,
        blocks=(
            replace(
                main.blocks[0],
                operations=(
                    first,
                    Operation("call", first.results, outputs, callee="leaf"),
                ),
                return_values=outputs,
            ),
        ),
    )
    record = replace(record, functions=(main, replace(definition, blocks=(leaf,))))
    bound = bind_register_program(record, ((0,), (1,)))
    gates = bound.circuit.instructions
    assert [
        gate.metadata["classical_bit"] for gate in gates if gate.name == "measure"
    ] == [0, 3]
    assert sum(gate.name == "classical_xor" for gate in gates) == 4
    state = torch.tensor([1, 2j, 3 + 1j, -4j], dtype=torch.complex128)
    state /= state.norm()
    original = dynamic.to_ir().instructions
    second = tuple(
        (
            replace(gate, metadata={**gate.metadata, "classical_bit": 1})
            if gate.name == "measure"
            else (
                replace(gate, metadata={**gate.metadata, "conditions": ((1, 1),)})
                if gate.name == "x"
                else gate
            )
        )
        for gate in original
    )
    for history in product((0, 1), repeat=2):
        torch.testing.assert_close(
            forced_branch(state, gates, history),
            forced_branch(state, (*original, *second), history),
            rtol=0,
            atol=2e-12,
        )
    assert ClassicalXor((), 1, True).instruction() in gates


@pytest.mark.parametrize(
    "inputs,clean,borrowed,message",
    [
        ([(0,)], (), (), "immutable"),
        (((0, 1),), (), (), "widths"),
        (((0,),), [1], (), "immutable"),
        (((0,),), (), [1], "immutable"),
        (((0,),), (0,), (), "disjoint"),
        (((True,),), (), (), "integer"),
        (((-1,),), (), (), "integer"),
        (((0,),), (1,), (1,), "disjoint"),
    ],
)
def test_invalid_physical_bindings_fail_at_entry(
    inputs: Any, clean: Any, borrowed: Any, message: str
) -> None:
    record = ProgramRecord((word_function(1),))
    with pytest.raises(ValueError, match=message):
        bind_register_program(record, inputs, clean=clean, borrowed=borrowed)


def test_binding_profile_and_ownership_diagnostics() -> None:
    record = ProgramRecord((word_function(0),))
    with pytest.raises(ValueError, match="physical wire"):
        bind_register_program(record, ((),))
    with pytest.raises(ValueError, match="unknown Program entry"):
        bind_register_program(record, ((),), clean=(0,), entry="absent")
    function = word_function(1)
    with pytest.raises(ValueError, match="invalid register Program"):
        bind_register_program(
            ProgramRecord(
                (
                    replace(
                        function,
                        blocks=(replace(function.blocks[0], return_values=()),),
                    ),
                )
            ),
            ((0,),),
        )
    builder = ComponentProgramBuilder(
        Signature([Register("data", RegisterType("opaque", 1), side=RegisterSide.LEFT)])
    )
    builder.free(builder.inputs["data"])
    with pytest.raises(ValueError, match="caller input"):
        bind_register_program(builder.finalize({}), ((0,),))
    builder = ComponentProgramBuilder(
        Signature([Register("out", RegisterType("opaque", 1), side=RegisterSide.RIGHT)])
    )
    out = builder.allocate()
    with pytest.raises(ValueError, match="ownership transfer"):
        bind_register_program(builder.finalize({"out": out}), (), clean=(0,))
    with pytest.raises(ValueError, match="Insufficient clean"):
        bind_register_program(allocated_circuit(CircuitIR(2, ()), 1), ((0,),))
    q, bit = ValueRef("entry", 0, "qubit"), ValueRef("entry", 1, "bool")
    outq = ValueRef("entry", 2, "qubit")
    classical_out = FunctionRecord(
        "main",
        (
            BlockRecord(
                "entry",
                (q,),
                operations=(Operation("measure", (q,), (outq, bit)),),
                return_values=(outq, bit),
            ),
        ),
    )
    with pytest.raises(ValueError, match="quantum-only"):
        bind_register_program(ProgramRecord((classical_out,)), ((0,),))


def test_isolated_scalar_ties_follow_declaration_order_and_implicit_returns_fail() -> (
    None
):
    q = ValueRef("entry", 0, "qubit")
    scalars = tuple(ValueRef("entry", i, "scalar") for i in (1, 2))
    function = FunctionRecord(
        "main",
        (
            BlockRecord(
                "entry",
                (q,),
                operations=tuple(
                    Operation("constant", results=(value,), literal=float(index))
                    for index, value in enumerate(scalars)
                ),
                return_values=(q,),
            ),
        ),
    )
    assert local_operation_order(function, {"main": function}) == (0, 1)
    assert (
        bind_register_program(ProgramRecord((function,)), ((0,),)).circuit.instructions
        == ()
    )
    implicit = FunctionRecord("main", (BlockRecord("entry", (q,)),))
    assert verify_program_record(ProgramRecord((implicit,))) == ()
    with pytest.raises(ValueError, match="finite quantum Program"):
        bind_register_program(ProgramRecord((implicit,)), ((0,),))


def test_zero_boolean_and_disjunctive_feedback_bind_actual_bit_handles() -> None:
    from flagquantum.core.ir.program import ConditionRef

    q, out = ValueRef("entry", 0, "qubit"), ValueRef("entry", 1, "qubit")
    zero, one = ValueRef("entry", 2, "bool"), ValueRef("entry", 3, "bool")
    function = FunctionRecord(
        "main",
        (
            BlockRecord(
                "entry",
                (q,),
                operations=(
                    Operation("bool_zero", results=(zero,)),
                    Operation("bool_one", results=(one,)),
                    Operation(
                        "x",
                        (q,),
                        (out,),
                        condition_clauses=(
                            (ConditionRef(zero, True),),
                            (ConditionRef(one, True),),
                        ),
                    ),
                ),
                return_values=(out,),
            ),
        ),
    )
    bound = bind_register_program(ProgramRecord((function,)), ((0,),))
    assert bound.circuit.instructions[-1].metadata["condition_clauses"] == (
        ((0, 1),),
        ((1, 1),),
    )
    state = torch.tensor([1 + 2j, -3j], dtype=torch.complex128)
    torch.testing.assert_close(
        forced_branch(state, bound.circuit.instructions, ()),
        state.flip(0),
        rtol=0,
        atol=2e-12,
    )


def test_repeated_private_workspace_calls_schedule_before_expansion_and_reuse_pool() -> (
    None
):
    circuit = CircuitIR(2, (Instruction("cx", (0, 1)),) * 2)
    module = allocated_circuit(circuit, 1)
    functions = tuple(
        replace(f, name="lifetime") if f.name == "main" else f for f in module.functions
    )
    builder = ComponentProgramBuilder(
        Signature([Register("data", RegisterType("opaque", 1))]), functions
    )
    data = builder.inputs["data"]
    for _ in range(2):
        data = builder.call("lifetime", {"data": data})
    bound = bind_register_program(builder.finalize({"data": data}), ((2,),), clean=(0,))
    assert [(event.kind, event.wires) for event in bound.events] == [
        ("allocate", (0,)),
        ("free", (0,)),
    ] * 2
    assert bound.circuit.instructions == (Instruction("cx", (2, 0)),) * 4
    assert tuple(event.position for event in bound.events) == (1, 6, 9, 14)
    state = torch.zeros(8, dtype=torch.complex128)
    state[:4] = torch.tensor([1, 2j, 3 + 1j, -4j])
    torch.testing.assert_close(
        forced_branch(state, bound.circuit.instructions, ()), state, rtol=0, atol=2e-12
    )
