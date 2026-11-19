"""Nested Program and partitioned calls share immutable imported definitions."""

from dataclasses import replace
from typing import Any

import pytest
import torch

from examples.algorithms.component_construction_tutorial import BoundComponentExample
from examples.algorithms.register_graphs import parallel_program
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_expansion import expand_component_calls
from flagquantum.compiler.registers.component_gate_function import bit_gate_function
from flagquantum.compiler.registers.component_graph import iter_component_calls
from flagquantum.compiler.registers.layout.auto_partition import AutoPartition
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import (
    BlockRecord,
    FunctionRecord,
    Operation,
    ProgramRecord,
    verify_program_record,
)
from flagquantum.core.ir.program.registers import Register, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.unitary import get_unitary
from tests.unit.test_component_graph import _native_trace, _source_trace
from tests.unit.test_register_component_calls import SourceCalls

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
pytest.importorskip("numpy")
pytest.importorskip("qualtran")


def cnot_program() -> ProgramRecord:
    signature = Signature.build(ctrl=1, target=1)
    leaf = bit_gate_function(
        "CNOT", tuple(signature), CircuitIR(2, (Instruction("cx", (0, 1)),))
    )
    builder = ComponentProgramBuilder(signature, (leaf,))
    return builder.finalize(builder.call_dict("CNOT", builder.inputs))


@pytest.mark.parametrize("occupied", [False, True])
def test_repeated_nested_imports_share_definitions_and_preserve_call_boundary(
    occupied: bool,
) -> None:
    record = parallel_program()
    signature = Signature(record.functions[0].registers or ())
    definitions: tuple[FunctionRecord, ...] = ()
    if occupied:
        identity = ComponentProgramBuilder(signature, name="__fq_component_0_main")
        definitions = identity.finalize(identity.inputs).functions
    builder = ComponentProgramBuilder(signature, definitions)
    original = repr(record)
    values = builder.inputs
    for _ in range(3):
        values = builder.call_program(record, values)
    result = builder.finalize(values)
    calls = list(iter_component_calls(result))
    assert len(calls) == 3
    assert len({op.callee for op in result.functions[0].blocks[0].operations}) == 1
    assert len(result.functions) == 5 + int(occupied)
    assert repr(record) == original
    bound = bind_register_program(result, ((0, 1, 2),))
    assert bound.circuit.instructions == tuple(
        Instruction("x", (i,)) for _ in range(3) for i in range(3)
    )
    assert bound.events == ()


@pytest.mark.parametrize(
    "left_only,unused", [(False, False), (False, True), (True, False)]
)
def test_partitioned_calls_follow_original_entry_and_full_complex_tensor(
    left_only: bool, unused: bool
) -> None:
    from qualtran import BloqBuilder, QAny, Side
    from qualtran import Register as SourceRegister
    from qualtran.bloqs.basic_gates import CNOT
    from qualtran.bloqs.bookkeeping.auto_partition import Unused

    import flagquantum as fq

    layout = AutoPartition(
        (
            (
                "packed",
                (
                    (("target", 1), (None, 1), ("ctrl", 1))
                    if unused
                    else (("target", 1), ("ctrl", 1))
                ),
            ),
        ),
        left_only=left_only,
    )
    width = 3 if unused else 2
    signature = Signature.build(packed=width)
    native = ComponentProgramBuilder(signature, allow_register_addition=left_only)
    # A left-only caller consumes its packed input and infers the child outputs.
    if left_only:
        from flagquantum.core.ir.program.registers import RegisterSide

        native = ComponentProgramBuilder(
            Signature(
                (
                    Register(
                        "packed", RegisterType("opaque", width), side=RegisterSide.LEFT
                    ),
                )
            ),
            allow_register_addition=True,
        )
    outputs = native.call_partitioned(cnot_program(), layout, native.inputs)
    result = native.finalize(outputs)
    source = BloqBuilder()
    source_input = source.add_register(
        SourceRegister(
            "packed", QAny(width), side=Side.LEFT if left_only else Side.THRU
        )
    )
    segments: list[Any] = (
        ["target", Unused(1), "ctrl"] if unused else ["target", "ctrl"]
    )
    returned = source.add_and_partition(
        CNOT().as_composite_bloq(),
        [(SourceRegister("packed", QAny(width)), segments)],
        left_only=left_only,
        packed=source_input,
    )
    if left_only:
        expected = source.finalize(ctrl=returned[0], target=returned[1])
    else:
        expected = source.finalize(packed=returned)
    assert _native_trace(result) == _source_trace(SourceCalls, expected)
    bound = bind_register_program(result, (tuple(range(width)),))
    assert bound.events == ()
    assert bound.circuit.instructions == (Instruction("cx", (width - 1, 0)),)
    wires = [wire for port in bound.outputs for wire in port]
    matrix = get_unitary(bound.circuit, dtype=torch.complex128)
    logical = (
        matrix.reshape([2] * (2 * width))
        .permute(*wires, *range(width, 2 * width))
        .reshape(1 << width, 1 << width)
    )
    source_matrix = torch.as_tensor(expected.tensor_contract(), dtype=torch.complex128)
    torch.testing.assert_close(logical, source_matrix, rtol=0, atol=0)
    state = torch.arange(1 << (width + 1), dtype=torch.float64)
    state = torch.complex(state + 1, state.flip(0) + 2)
    state /= state.norm()
    torch.testing.assert_close(
        logical @ state.reshape(1 << width, 2),
        source_matrix @ state.reshape(1 << width, 2),
        rtol=0,
        atol=0,
    )
    circuit = fq.Circuit(width + 1)
    circuit.append(BoundComponentExample(result, (tuple(range(width)),)))
    assert circuit.to_ir().instructions == bound.circuit.instructions
    expanded, _ = expand_component_calls(result)
    assert (
        bind_register_program(expanded, (tuple(range(width)),)).circuit == bound.circuit
    )


def test_declaration_import_and_invalid_import_leave_calls_unconsumed() -> None:
    signature = Signature.build(q=1)
    declaration = FunctionRecord("leaf", (), registers=tuple(signature))
    record = ProgramRecord((declaration,))
    builder = ComponentProgramBuilder(signature)
    with pytest.raises(ValueError, match="unknown or untyped"):
        builder.call_program(record, builder.inputs, entry="missing")
    assert builder._operations == [] and len(builder._available) == 1
    result = builder.finalize(
        builder.call_program(record, builder.inputs, entry="leaf")
    )
    assert verify_program_record(result) == ()
    with pytest.raises(ValueError):
        bind_register_program(result, ((0,),))


def test_invalid_complete_program_fails_before_importing_any_definition() -> None:
    record = ProgramRecord(
        (
            FunctionRecord(
                "bad",
                (
                    BlockRecord(
                        "entry",
                        (),
                        operations=(Operation("call", callee="missing"),),
                        return_values=(),
                    ),
                ),
                registers=(),
            ),
        )
    )
    builder = ComponentProgramBuilder(Signature([]))
    with pytest.raises(ValueError, match="invalid component import"):
        builder.call_program(record, {}, entry="bad")
    assert builder._operations == []
    assert builder._functions == {}


def test_conflicting_child_definitions_do_not_replace_existing_code() -> None:
    record = cnot_program()
    existing = replace(record.functions[1], name="main")
    builder = ComponentProgramBuilder(
        Signature.build(ctrl=1, target=1), (existing,), name="caller"
    )
    result = builder.finalize(builder.call_program(record, builder.inputs))
    assert existing in result.functions
    assert bind_register_program(result, ((0,), (1,))).circuit.instructions == (
        Instruction("cx", (0, 1)),
    )


@pytest.mark.parametrize(
    "groups",
    [
        (("packed", (("ctrl", 1), (None, 1), ("q", 1), ("spectator", 1))),),
        (("a", (("ctrl", 1), ("q", 1))), ("b", (("spectator", 1), (None, 1)))),
    ],
)
def test_partitioned_measured_child_matches_source_feedback_and_all_branches(
    groups: tuple[tuple[str, tuple[tuple[str | None, int], ...]], ...],
) -> None:
    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.dynamic import DynamicCircuit
    from tests.unit.measurement_checks import forced_branch
    from tests.unit.test_component_auto_partition import (
        partition_gates,
        source_partition,
    )
    from tests.unit.test_component_control_selection import assert_gate_parameters
    from tests.unit.test_specialized_component_control import positive_fixture

    record, child = positive_fixture("spectator")
    layout = AutoPartition(groups)
    ports = {}
    offset = 0
    for name, parts in groups:
        width = sum(width for _, width in parts)
        ports[name] = tuple(range(offset, offset + width))
        offset += width
    builder = ComponentProgramBuilder(
        Signature.build(**{name: len(wires) for name, wires in ports.items()})
    )
    result = builder.finalize(builder.call_partitioned(record, layout, builder.inputs))
    bound = bind_register_program(result, tuple(ports.values()), clean=(4,))
    source = source_partition(child, layout)
    bits = {
        gate.wires[0]: gate.metadata["classical_bit"]
        for gate in bound.circuit.instructions
        if gate.name == "measure"
    }
    reference = partition_gates(source, child, ports, ports, (4,), bits)
    assert_gate_parameters(bound.circuit.instructions, reference)
    assert bound.outputs == tuple(ports.values())
    initial = torch.zeros(32, dtype=torch.complex128)
    values = torch.arange(16, dtype=torch.float64)
    initial[::2] = torch.complex(values + 1, values.flip(0) + 2)
    initial /= initial.norm()
    for outcome in (0, 1):
        actual = forced_branch(initial, bound.circuit.instructions, (outcome,))
        expected = forced_branch(initial, reference, (outcome,))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert torch.count_nonzero(actual[1::2]) == 0
    circuit = DynamicCircuit(5)
    circuit.append(FieldOperation(5, bound.circuit.instructions))
    assert circuit.to_ir().instructions == bound.circuit.instructions
