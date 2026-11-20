# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Classical control semantics adapted from Qualtran v0.7.0 ClassicallyControlled.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mcmt/classically_controlled.py

"""Propagate classical controls through stored typed quantum component bodies."""

from dataclasses import replace

from ...core.ir.program import ProgramRecord, verify_program_record
from ...core.ir.program.registers import RegisterType
from ...core.ir.program.signatures import Signature
from .component_feedback import _feedback_component_function


def classical_control_component_program(
    record: ProgramRecord,
    controls: tuple[tuple[tuple[int, ...], tuple[int | bool, ...]], ...],
    entry: str = "main",
    *,
    control_types: tuple[RegisterType, ...] = (),
) -> ProgramRecord:
    """Prepend classical controls while retaining stored calls and conditions.

    Reachable bodies receive one shared generated definition each. The root
    keeps its entry name, original children stay available, and unrelated
    definitions remain shared. Controls are threaded linearly through every call.
    Existing primitive conditions are conjoined with new read-only snapshots.
    Allocations and releases remain unconditional. Child work must be restored
    before release in every branch. All input records remain unchanged.

    This implements the source's classical action using native stored bodies.
    The frozen wrapper itself supplies no physical decomposition. The entry
    must be THRU. Invalid graphs, unsupported classical writes, measurement,
    reset, channels or nonconcrete controls raise ValueError. Proposal 108 admits
    canonical classical integer words through control_types. Proposal 110 adds
    canonical binary fields and polynomials with the same stored-word contract.
    Existing omitted
    control_types keep CBit controls and their exact graph construction.
    """
    diagnostics = verify_program_record(record)
    if diagnostics:
        raise ValueError(f"invalid classical control component graph: {diagnostics}")
    functions = {f.name: f for f in record.functions}
    if entry not in functions or record.physical_context is not None:
        raise ValueError("classical control requires a finite typed entry")
    registers = functions[entry].registers
    if registers is None or not Signature(registers).thru_registers_only:
        raise ValueError("classical control requires THRU entry registers")
    pending = [entry]
    reachable: dict[str, str] = {}
    index = 0
    while pending:
        name = pending.pop()
        if name in reachable:
            continue
        generated = entry
        if name != entry:
            generated = f"__fq_classical_control_{index}"
            while generated in functions:
                index += 1
                generated = f"__fq_classical_control_{index}"
            index += 1
        if any(
            op.kraus is not None
            for block in functions[name].blocks
            for op in block.operations
        ):
            raise ValueError("Feedback requires a coherent body without channels")
        reachable[name] = generated
        pending.extend(
            op.callee
            for block in functions[name].blocks
            for op in block.operations
            if op.callee is not None
        )
    rebuilt = {
        name: replace(
            _feedback_component_function(
                functions[name], controls, reachable, control_types
            ),
            name=target,
        )
        for name, target in reachable.items()
    }
    result = replace(
        record,
        canonical_payload=None,
        functions=(
            *(rebuilt[entry] if f.name == entry else f for f in record.functions),
            *(f for name, f in rebuilt.items() if name != entry),
        ),
    )
    diagnostics = verify_program_record(result)
    if diagnostics:
        raise ValueError(f"invalid classical controlled graph: {diagnostics}")
    return result
