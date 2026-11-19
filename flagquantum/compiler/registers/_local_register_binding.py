"""Prepare local physical bindings for finite register component adapters."""

from dataclasses import replace

from ...core.ir.program import ProgramRecord, verify_program_record
from ...core.ir.program.register_verifier import executable_register_width
from .entry_bindings import entry_input_layout
from .register_binding import BoundRegisterProgram, bind_register_program


def _bind_local_register_program(
    record: ProgramRecord, entry: str
) -> BoundRegisterProgram:
    """Reuse physical binding with local inputs and conservative work capacity.

    The sum of expanded allocations bounds each pool without guessing a limit
    or retrying failed bindings. Only wires actually used by the bound component
    survive compaction. Work ownership and release are checked by the existing
    binder. The caller receives transferred outputs and their lifetime events.
    """
    diagnostics = verify_program_record(record)
    if diagnostics:
        raise ValueError(f"invalid local register Program: {diagnostics}")
    functions = {function.name: function for function in record.functions}
    if entry not in functions:
        raise ValueError(f"unknown Program entry {entry!r}")
    function = functions[entry]
    if any(
        len(f.blocks) != 1
        or f.blocks[0].terminator != "return"
        or f.blocks[0].edges
        or f.blocks[0].return_values is None
        for f in functions.values()
    ):
        raise ValueError("local register binding requires finite function bodies")
    widths, _ = entry_input_layout(function, ())
    inputs: list[tuple[int, ...]] = []
    offset = 0
    for width in widths:
        inputs.append(tuple(range(offset, offset + width)))
        offset += width
    capacity = {False: 0, True: 0}
    pending = list(function.blocks[0].operations)
    while pending:
        operation = pending.pop()
        if operation.opcode == "call":
            pending.extend(functions[operation.callee or ""].blocks[0].operations)
        elif operation.opcode == "allocate_register":
            assert operation.register_type is not None
            assert operation.workspace_dirty is not None
            capacity[operation.workspace_dirty] += executable_register_width(
                operation.register_type
            )
    bound = bind_register_program(
        record,
        tuple(inputs),
        clean=tuple(range(offset, offset + capacity[False]))
        or ((0,) if not offset and not capacity[True] else ()),
        borrowed=tuple(
            range(offset + capacity[False], offset + sum(capacity.values()))
        ),
        entry=entry,
        allow_workspace_outputs=True,
        input_ownership=tuple((wire, False) for wire in range(offset)),
    )
    used = set(range(offset))
    used.update(wire for gate in bound.circuit.instructions for wire in gate.wires)
    used.update(wire for event in bound.events for wire in event.wires)
    used.update(wire for word in bound.outputs for wire in word)
    if not used:
        used.add(0)
    wires = {wire: index for index, wire in enumerate(sorted(used))}
    return replace(
        bound,
        circuit=replace(
            bound.circuit,
            n_wires=len(wires),
            shape=(),
            instructions=tuple(
                replace(gate, wires=tuple(wires[wire] for wire in gate.wires))
                for gate in bound.circuit.instructions
            ),
        ),
        outputs=tuple(tuple(wires[wire] for wire in word) for word in bound.outputs),
        events=tuple(
            replace(event, wires=tuple(wires[wire] for wire in event.wires))
            for event in bound.events
        ),
    )
