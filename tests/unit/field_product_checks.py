"""Bind stored field products and read original primitive/ownership connections."""

from math import prod
from typing import Any

from flagquantum.compiler.registers.register_binding import (
    BoundRegisterProgram,
    bind_register_program,
)
from flagquantum.core.ir import Instruction
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.core.ir.program.registers import RegisterSide
from tests.unit.register_source_graph import (
    SourcePorts,
    SourceResult,
    walk_source_graph,
)


def bind_product(program: ProgramRecord) -> tuple[BoundRegisterProgram, SourcePorts]:
    assert verify_program_record(program) == ()
    root = next(f for f in program.functions if f.name == "main")
    inputs = []
    ports: SourcePorts = {}
    position = 0
    for register in root.registers or ():
        if not register.side & RegisterSide.LEFT:
            continue
        width = register.dtype.quantum_bits
        assert isinstance(width, int)
        ports[register.name] = tuple(
            range(position, position + width * prod(register.shape))
        )
        for _ in range(prod(register.shape)):
            inputs.append(tuple(range(position, position + width)))
            position += width
    bound = bind_register_program(
        program,
        tuple(inputs),
        clean=tuple(range(position, position + 16)),
        input_ownership=tuple((wire, False) for wire in range(position)),
        allow_workspace_outputs=True,
    )
    return bound, ports


def product_source_stream(
    source: Any, bound: BoundRegisterProgram, ports: SourcePorts
) -> SourceResult:
    from qualtran.bloqs.basic_gates import CNOT, Toffoli
    from qualtran.bloqs.bookkeeping import Allocate, Free

    cursor = 0
    used: set[int] = set()

    def leaf(
        bloq: Any, inputs: SourcePorts, desired: SourcePorts
    ) -> SourceResult | None:
        nonlocal cursor
        if isinstance(bloq, Allocate):
            event = next(
                e
                for e in bound.events
                if e.kind == "allocate"
                and e.instruction_offset == cursor
                and e.position not in used
            )
            used.add(event.position)
            return (), {"reg": event.wires}
        if isinstance(bloq, Free):
            return (), {}
        if isinstance(bloq, (CNOT, Toffoli)):
            wires = inputs["ctrl"] + inputs["target"]
            cursor += 1
            return (
                Instruction("cx" if isinstance(bloq, CNOT) else "ccx", wires),
            ), dict(inputs)
        return None

    return walk_source_graph(source, ports, {}, leaf)
