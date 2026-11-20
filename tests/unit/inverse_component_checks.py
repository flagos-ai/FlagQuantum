"""Bind frozen inverse source calls to independently recorded physical lifetimes."""

from collections import Counter
from typing import Any

from flagquantum.compiler.registers.register_binding import (
    BoundRegisterProgram,
    bind_register_program,
)
from flagquantum.core.ir import Instruction
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from tests.unit.register_source_graph import (
    SourcePorts,
    SourceResult,
    walk_source_graph,
)
from tests.unit.test_multi_control_component import MeasurementSlots
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source


def bind_inverse_component(
    program: ProgramRecord,
) -> tuple[BoundRegisterProgram, SourcePorts, int]:
    """Use contiguous MSB data ports and a reusable clean physical pool."""
    assert verify_program_record(program) == ()
    ports: SourcePorts = {}
    position = 0
    for register in program.functions[0].registers or ():
        width = register.dtype.quantum_bits
        assert isinstance(width, int)
        ports[register.name] = tuple(range(position, position + width))
        position += width
    bound = bind_register_program(
        program,
        tuple(ports.values()),
        clean=tuple(range(position, 3 * position + 20)),
        input_ownership=tuple((i, False) for i in range(position)),
        allow_workspace_outputs=True,
    )
    return bound, ports, position


def inverse_source_stream(
    source: Any, bound: BoundRegisterProgram, ports: SourcePorts
) -> tuple[Instruction, ...]:
    """Visit original connections and preserve measured And leaf channels."""
    from qualtran import CompositeBloq
    from qualtran.bloqs.bookkeeping import Allocate, Cast, Free, Join, Split
    from qualtran.bloqs.mcmt import And

    cursor = 0
    used: set[int] = set()
    slots = MeasurementSlots(bound.circuit.instructions)

    def leaf(bloq: Any, old: SourcePorts, desired: SourcePorts) -> SourceResult | None:
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
        if isinstance(bloq, (Split, Join, Cast, CompositeBloq)):
            return None
        name = type(getattr(bloq, "subbloq", bloq)).__name__
        if name not in {
            "And",
            "CNOT",
            "XGate",
            "TwoBitCSwap",
            "CSwap",
            "Add",
            "CAdd",
            "AddK",
            "BitwiseNot",
        }:
            return None
        outputs = {
            r.name: old[r.name] for r in bloq.signature.rights() if r.name in old
        }
        if isinstance(bloq, And):
            if not bloq.uncompute:
                event = next(
                    e
                    for e in bound.events
                    if e.kind == "allocate"
                    and e.instruction_offset == cursor
                    and e.position not in used
                )
                used.add(event.position)
                outputs["target"] = event.wires
            gates = reference_arithmetic_gates(
                bloq, {**old, **outputs}, (), slots, cirq_style=True
            )
        else:
            plan = tuple(
                w
                for e in bound.events
                if e.kind == "allocate"
                and e.position not in used
                and e.instruction_offset is not None
                and e.instruction_offset >= cursor
                for w in e.wires
            )
            gates = reference_arithmetic_gates(
                bloq.as_composite_bloq(),
                old,
                plan,
                slots,
                preserve_components=True,
                reuse_allocations=False,
                expected_outputs=outputs,
            )
        assert all(outputs[k] == v for k, v in desired.items())
        cursor += len(gates)
        return gates, outputs

    expected_outputs = {
        r.name: output
        for r, output in zip(source.signature.rights(), bound.outputs, strict=True)
    }
    gates, outputs = walk_source_graph(source, ports, expected_outputs, leaf)
    assert outputs == expected_outputs and cursor == len(bound.circuit.instructions)
    return gates


def check_inverse_source(program: ProgramRecord, source: Any) -> None:
    """Check exact primitives/feedback and separately inspect source costs/peak."""
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    bound, ports, _ = bind_inverse_component(program)
    assert_source(
        bound.circuit.instructions, inverse_source_stream(source, bound, ports)
    )
    counts = Counter(g.name for g in bound.circuit.instructions)
    model = get_cost_value(source, QECGatesCost())
    assert counts["t"] + counts["tdg"] == model.total_t_count(ts_per_cswap=7)
    assert counts["measure"] == model.measurement
    live = {wire for wires in ports.values() for wire in wires}
    peak = len(live)
    for event in bound.events:
        if event.kind == "allocate":
            live.update(event.wires)
        else:
            live.difference_update(event.wires)
        peak = max(peak, len(live))
    assert live == {wire for wires in bound.outputs for wire in wires}
    assert peak == get_cost_value(source, QubitCount())
