"""Stored column calls retain primitive operands, feedback and clean storage."""

from collections import Counter

import pytest

from flagquantum.compiler.registers.arithmetic.integer_square import _square_terms
from flagquantum.compiler.registers.bits.bit_compression import weighted_columns_xor_ir
from flagquantum.compiler.registers.bits.weighted_columns_component import (
    weighted_columns_xor_program,
)
from flagquantum.compiler.registers.component_expansion import expand_component_calls
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import Instruction
from flagquantum.core.ir.program import verify_program_record
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def normalized_feedback(gates: tuple[Instruction, ...]) -> tuple[Instruction, ...]:
    """Rename fresh measurement SSA slots to the flat emitter's reusable slot."""
    normalized = []
    last_slot = None
    for gate in gates:
        metadata = dict(gate.metadata)
        if gate.name == "measure":
            last_slot = metadata["classical_bit"]
            metadata["classical_bit"] = 0
        if "conditions" in metadata:
            assert metadata["conditions"] == ((last_slot, 1),)
            metadata["conditions"] = tuple(
                (0, value) for _, value in metadata["conditions"]
            )
        normalized.append(
            Instruction(gate.name, gate.wires, gate.params, metadata=metadata)
        )
    return tuple(normalized)


@pytest.mark.parametrize("width", (1, 2, 3, 4, 5, 8))
@pytest.mark.parametrize("expanded", (False, True))
def test_square_columns_shared_primitive_dependencies(
    width: int, expanded: bool
) -> None:
    work_width = width * (width - 1)
    inputs = tuple(reversed(range(width)))
    result = tuple(range(width, 3 * width))
    work = tuple(range(3 * width, 3 * width + work_width))
    terms = tuple(
        (column, width - 1 - i, width - 1 - j if i != j else None)
        for i, j, column in _square_terms(width)
    )
    graph = weighted_columns_xor_program(width, 2 * width, work_width, terms)
    assert verify_program_record(graph) == ()
    calls = Counter(
        op.callee
        for op in graph.functions[0].blocks[0].operations
        if op.opcode == "call"
    )
    assert calls["column_and"] == calls["column_erase"] == work_width
    if expanded:
        graph, _ = expand_component_calls(graph)
    bound = bind_register_program(
        graph, tuple((wire,) for wire in (*inputs, *result)), clean=work
    )
    reference = weighted_columns_xor_ir(
        inputs,
        result,
        work,
        0,
        tuple(
            (column, inputs[left], inputs[right] if right is not None else None)
            for column, left, right in terms
        ),
    )
    assert_source(normalized_feedback(bound.circuit.instructions), reference)
    counts = Counter(g.name for g in bound.circuit.instructions)
    assert counts["t"] + counts["tdg"] == 4 * work_width
    assert counts["measure"] == counts["reset"] == work_width
    assert bound.outputs == tuple((wire,) for wire in (*inputs, *result))
    assert (
        sum(len(event.wires) for event in bound.events if event.kind == "allocate")
        == work_width
    )
    assert (
        sum(len(event.wires) for event in bound.events if event.kind == "free")
        == work_width
    )


@pytest.mark.parametrize(
    "dimensions,terms",
    [
        ((0, 2, 0), ()),
        ((1, 0, 0), ()),
        ((True, 2, 0), ()),
        ((2, 4, -1), ()),
        ((2, 4, 0), ((0, 0, 1),)),
        ((2, 1, 1), ((0, 0, None), (0, 1, None))),
        ((2, 4, 1), ((0, 0, 0),)),
        ((2, 4, 1), ((0, 0, 2),)),
    ],
)
def test_invalid_column_storage_is_rejected(
    dimensions: tuple[int, int, int], terms: tuple[tuple[int, int, int | None], ...]
) -> None:
    with pytest.raises(ValueError):
        weighted_columns_xor_program(*dimensions, terms)


@pytest.mark.parametrize("inverse", (False, True))
def test_shared_and_leaf_against_unchanged_source(inverse: bool) -> None:
    from qualtran.bloqs.mcmt.and_bloq import And

    from flagquantum.core.ir.program import ProgramRecord
    from tests.unit.test_register_constant_addition import reference_arithmetic_gates

    graph = weighted_columns_xor_program(2, 2, 1, ((0, 0, 1),))
    name = "column_erase" if inverse else "column_and"
    leaf = next(function for function in graph.functions if function.name == name)
    native = bind_register_program(
        ProgramRecord((leaf,)), ((0,), (1,), (2,)), entry=name
    )
    source = And().adjoint() if inverse else And()
    expected = reference_arithmetic_gates(
        source,
        {"ctrl": (0, 1), "target": (2,)},
        (),
        {2: 0} if inverse else {},
        cirq_style=True,
    )
    assert_source(normalized_feedback(native.circuit.instructions), expected)
