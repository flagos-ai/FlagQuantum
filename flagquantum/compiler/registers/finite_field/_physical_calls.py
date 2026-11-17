"""Instruction ranges for the finite-field generator's nested physical calls."""

from dataclasses import replace

from ....core.classical_dataflow import ClassicalXor, parse_classical_xor
from ....core.ir import Instruction
from ..._component_trace.trace import ComponentCall
from ...artifacts.reversible_phase import ReversiblePhaseProgram
from ...lowering.physical_trace import PhysicalTrace, lower_physical_trace


def append_field_call(
    gates: list[Instruction],
    calls: list[ComponentCall] | None,
    name: str,
    operations: tuple[Instruction, ...],
    children: tuple[ComponentCall, ...] = (),
) -> None:
    """Append gates and rebind relative call ranges at the insertion point."""
    start = len(gates)
    gates.extend(operations)
    if calls is None:
        return

    def rebind(call: ComponentCall) -> ComponentCall:
        return replace(
            call,
            start=call.start + start,
            stop=call.stop + start,
            children=tuple(rebind(child) for child in call.children),
        )

    calls.append(
        ComponentCall(name, start, len(gates), children=tuple(map(rebind, children)))
    )


def lower_field_erasure_trace(
    gates: list[Instruction], call: ComponentCall, parity: int, last_wire: int
) -> tuple[ReversiblePhaseProgram, tuple[ComponentCall, ...]]:
    """Use the source's initially zero bit for the first parity lease."""
    trace = PhysicalTrace(calls=[call])
    for index, gate in enumerate(gates):
        if gate.name == "classical_xor":
            assignment = parse_classical_xor(gate)
            if assignment.target == parity:
                gates[index] = ClassicalXor(
                    (*assignment.sources, parity), parity
                ).instruction()
                trace.implicit_zero_bits.add(parity)
                break
    return lower_physical_trace(
        tuple(gates), 0, last_wire, trace, reorder_feedback=False
    )
