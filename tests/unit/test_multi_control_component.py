"""Literal multi-control hierarchy and measured predicate execution."""

from collections.abc import Iterator
from hashlib import sha1
from itertools import product
from pathlib import Path

import pytest

from flagquantum.compiler.registers.control.multi_control_component import (
    controlled_word_x_program,
    multi_control_x_program,
    multi_control_z_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import Instruction
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.core.ir.program.registers import RegisterSide
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def frozen_multi_control_source() -> None:
    from qualtran.bloqs.mcmt import multi_control_pauli

    for name, expected in (
        ("multi_control_pauli.py", "2953333a1cdd9408b03cdec844b5907af091d5a5"),
        ("multi_control_pauli_test.py", "eed8610ca4c31125b96638a91cd90b58906fe01e"),
    ):
        raw = (Path(multi_control_pauli.__file__).parent / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == expected


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
@pytest.mark.parametrize("axis", ["x", "z"])
def test_multi_control_root_and_primitive_dependencies(
    values: tuple[int, ...], axis: str
) -> None:
    from qualtran.bloqs.mcmt.multi_control_pauli import MultiControlX, MultiControlZ

    graph = (multi_control_x_program if axis == "x" else multi_control_z_program)(
        values
    )
    source = (MultiControlX if axis == "x" else MultiControlZ)(values)
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
        reference_arithmetic_gates(source, ports, clean, slots, keep_pauli_leaves=True),
    )


@pytest.mark.parametrize(
    "values", [pattern for n in range(1, 5) for pattern in product((0, 1), repeat=n)]
)
def test_word_controls_retain_source_hierarchy_and_array_primitive_stream(
    values: tuple[int, ...],
) -> None:
    from qualtran._infra.controlled import CtrlSpec
    from qualtran._infra.data_types import QAny
    from qualtran.bloqs.basic_gates.x_basis import XGate
    from qualtran.bloqs.mcmt.controlled_via_and import ControlledViaAnd

    width = len(values)
    activation = sum(value << (width - i - 1) for i, value in enumerate(values))
    source = ControlledViaAnd(XGate(), CtrlSpec(qdtypes=QAny(width), cvs=activation))
    graph = controlled_word_x_program(values)
    assert not verify_program_record(graph)
    if width > 1:
        compare_layers(graph, source, atomic_leaves(graph))
    clean = tuple(range(width + 1, 2 * width))
    word = bind_register_program(graph, (tuple(range(width)), (width,)), clean=clean)
    array = bind_register_program(
        multi_control_x_program(values),
        (*((i,) for i in range(width)), (width,)),
        clean=clean,
    )
    assert word.circuit == array.circuit
    assert tuple(wire for output in word.outputs for wire in output) == tuple(
        wire for output in array.outputs for wire in output
    )
    assert_source(
        word.circuit.instructions,
        reference_arithmetic_gates(
            source,
            {"ctrl": tuple(range(width)), "q": (width,)},
            clean,
            MeasurementSlots(word.circuit.instructions),
            keep_pauli_leaves=True,
        ),
    )
