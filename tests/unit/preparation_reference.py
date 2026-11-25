"""Traverse source preparation isometries beyond directional root signatures."""

from collections.abc import Iterable

from flagquantum.core.ir.program import ProgramRecord
from tests.unit.test_multi_control_component import atomic_leaves


def preparation_terminals(
    record: ProgramRecord, *, native_boundaries: Iterable[str] = ()
) -> set[str]:
    """Keep accepted primitive leaves while traversing source lookup wrappers.

    RIGHT outputs and LEFT consumed banks do not make a composite atomic.
    Model-only declarations remain terminals; executable native replacements
    for absent source bodies require explicitly named boundaries.
    """
    terminals = atomic_leaves(record)
    for function in record.functions:
        registers = function.registers or ()
        composite = bool(function.blocks) and all(
            op.opcode == "call" for block in function.blocks for op in block.operations
        )
        lookup = bool(registers) and registers[0].name.startswith("selection")
        if composite and (function is record.functions[0] or lookup):
            terminals.discard(function.name)
    terminals.update(native_boundaries)
    assert (
        record.functions[0].name not in terminals
    ), "Preparation root must be traversed"
    return terminals
