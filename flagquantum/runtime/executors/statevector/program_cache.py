"""Reuse of relabelled statevector programs across executions.

A persistent wire layout relabels the program while a circuit runs, and the same
relabelling recurs on every execution of the same circuit. Rebuilding through
``dataclasses.replace`` re-validates every parameter the instruction carries, and
validating an accelerator-resident angle decides finiteness by reading the value
back, so rebuilding inside the execution region synchronized the device once per
parameterized gate. The caches here keep the rebuilt program so a repeat
execution of the same shape performs no rebuild at all.

Nothing here changes what runs: a cached program is the exact object the rebuild
would have produced for the same source program and the same mapping.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from ....core.ir import CircuitIR, Instruction, MeasurementNode, ObservableNode

#: Bound on every cache in this module. Layouts are per (wire structure,
#: parameter identity) and a program has few of them, so the bound is only a
#: guard against an unbounded program generator.
PROGRAM_CACHE_LIMIT = 128

_REMAPPED_PROGRAM_CACHE: dict[
    tuple[object, ...],
    tuple[
        tuple[Instruction, ...], tuple[ObservableNode, ...], tuple[MeasurementNode, ...]
    ],
] = {}
#: Keyed by source instruction identity and destination wires. The value keeps
#: the source instruction alive, so its id cannot be recycled while the entry
#: exists, and the rebuilt instruction is exactly the one that source produces.
_REMAPPED_INSTRUCTION_CACHE: dict[
    tuple[int, tuple[int, ...]], tuple[Instruction, Instruction]
] = {}


def _parameter_identity(ir: CircuitIR) -> tuple[tuple[int, ...], ...]:
    """Identify the parameter objects an instruction sequence carries.

    A remapped program is only interchangeable with another when it carries the
    same parameter objects, so the cache is keyed on their identity rather than
    on their values: values are re-read by the optimizer every step, and a
    value-derived key would miss on every one of them. Identity is safe as a key
    because the cached entry holds the instructions, which hold the parameters,
    so those ids cannot be recycled while the entry lives.

    Parameters that are plain numbers are compared by identity too, which is
    sound for the same reason and only ever costs a duplicate entry.
    """

    return tuple(
        tuple(id(value) for _, value in sorted(instruction.params.items()))
        for instruction in ir.instructions
    )


def remap_instruction_wires(
    instruction: Instruction, mapping: Sequence[int]
) -> Instruction:
    """Relabel one instruction's wires without re-validating its parameters.

    An instruction whose wires do not move is returned unchanged: the sweep
    relabels every instruction whether or not its wires are affected, and
    skipping the rebuild is what keeps the finiteness read from happening at all.
    """

    wires = tuple(int(mapping[int(wire)]) for wire in instruction.wires)
    if wires == tuple(int(wire) for wire in instruction.wires):
        return instruction
    key = (id(instruction), wires)
    entry = _REMAPPED_INSTRUCTION_CACHE.get(key)
    if entry is None:
        entry = (instruction, replace(instruction, wires=wires))
        if len(_REMAPPED_INSTRUCTION_CACHE) >= PROGRAM_CACHE_LIMIT:
            _REMAPPED_INSTRUCTION_CACHE.pop(next(iter(_REMAPPED_INSTRUCTION_CACHE)))
        _REMAPPED_INSTRUCTION_CACHE[key] = entry
    return entry[1]


def remapped_program(
    ir: CircuitIR,
    cache_key: tuple[object, ...],
    mapping: Sequence[int],
) -> tuple[
    tuple[Instruction, ...], tuple[ObservableNode, ...], tuple[MeasurementNode, ...]
]:
    """Relabel a whole program's wires, reusing the previous result when possible.

    The relabelled program depends only on the wire structure and on which
    parameter objects are carried, so both are part of the key. The returned
    tuples are shared; the caller still builds its own ``CircuitIR`` because
    ``ir.metadata`` is a plain mutable mapping that a caller may edit.
    """

    program_key = (cache_key, _parameter_identity(ir))
    program = _REMAPPED_PROGRAM_CACHE.get(program_key)
    if program is None:

        def remap(wires: Sequence[int]) -> tuple[int, ...]:
            return tuple(mapping[int(wire)] for wire in wires)

        program = (
            tuple(
                replace(instruction, wires=remap(instruction.wires))
                for instruction in ir.instructions
            ),
            tuple(
                replace(observable, wires=remap(observable.wires))
                for observable in ir.observables
            ),
            tuple(
                replace(measurement, wires=remap(measurement.wires))
                for measurement in ir.measurements
            ),
        )
        if len(_REMAPPED_PROGRAM_CACHE) >= PROGRAM_CACHE_LIMIT:
            _REMAPPED_PROGRAM_CACHE.pop(next(iter(_REMAPPED_PROGRAM_CACHE)))
        _REMAPPED_PROGRAM_CACHE[program_key] = program
    return program
