"""Literal multi-control hierarchy and measured predicate execution."""

from collections.abc import Iterator
from itertools import product

import pytest

from flagquantum.compiler.registers.control.multi_control_component import (
    multi_control_x_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import Instruction
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.core.ir.program.registers import RegisterSide
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def atomic_leaves(graph: ProgramRecord) -> set[str]:
    """Stop at source bookkeeping and physical leaves with accepted evidence."""
    return {
        function.name
        for function in graph.functions
        if not function.blocks
        or any(
            register.side != RegisterSide.THRU for register in function.registers or ()
        )
        or any(
            op.opcode != "call" for block in function.blocks for op in block.operations
        )
    }


class MeasurementSlots(dict[int, int]):
    """Bind distinct source measurement events to native SSA classical slots."""

    def __init__(self, gates: tuple[Instruction, ...]) -> None:
        super().__init__()
        slots: dict[int, list[int]] = {}
        for gate in gates:
            if gate.name == "measure":
                slots.setdefault(gate.wires[0], []).append(
                    gate.metadata["classical_bit"]
                )
        self.events: dict[int, Iterator[int]] = {
            wire: iter(bits) for wire, bits in slots.items()
        }

    def __getitem__(self, wire: int) -> int:
        return next(self.events[wire])


@pytest.mark.parametrize(
    "values", [pattern for n in range(5) for pattern in product((0, 1), repeat=n)]
)
def test_multi_control_root_and_primitive_dependencies(values: tuple[int, ...]) -> None:
    from qualtran.bloqs.mcmt import MultiControlX

    graph = multi_control_x_program(values)
    source = MultiControlX(values)
    assert verify_program_record(graph) == ()
    compare_layers(graph, source, atomic_leaves(graph))
    width = len(values)
    inputs = (*((i,) for i in range(width)), (width,))
    clean = tuple(range(width + 1, 2 * width))
    bound = bind_register_program(graph, inputs, clean=clean)
    ports: dict[str, tuple[int, ...]] = (
        {"controls": tuple(range(width)), "target": (width,)}
        if width
        else {"target": (0,)}
    )
    slots = MeasurementSlots(bound.circuit.instructions)
    assert_source(
        bound.circuit.instructions,
        reference_arithmetic_gates(source, ports, clean, slots),
    )
