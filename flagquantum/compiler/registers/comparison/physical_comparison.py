# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix gen_cmp.cc and gen_cmp_qq.cc.
# https://github.com/quantumlib/Qualtran/tree/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/gen/comparators

"""Export quantum less-than with source call ranges and feedback scopes."""

from dataclasses import replace
from itertools import groupby
from math import ceil, inf

from ....core.ir import CircuitIR
from ..._component_trace.trace import ComponentCall
from ...artifacts.reversible_phase import (
    ReversiblePhaseOperation as Operation,
)
from ...artifacts.reversible_phase import (
    ReversiblePhaseProgram,
)
from ...lowering.reversible_phase import lower_reversible_phase
from .mixed_comparison import mixed_less_than_ir


def _ladder_scopes(
    operations: tuple[Operation, ...], carries: set[int]
) -> tuple[Operation, ...]:
    """Reconstitute the source's single condition scope per cleared carry."""
    scheduled: list[Operation] = []
    measurement = None
    for condition, group in groupby(operations, key=lambda op: op.condition_bit):
        if condition is not None and condition == measurement:
            scheduled.append(Operation("push_condition", (), condition_bit=condition))
            scheduled.extend(replace(gate, condition_bit=None) for gate in group)
            scheduled.append(Operation("pop_condition", ()))
            measurement = None
        else:
            for gate in group:
                scheduled.append(gate)
                measurement = (
                    gate.result_bit
                    if gate.opcode == "hmr" and gate.qubits[0] in carries
                    else None
                )
    return tuple(scheduled)


def physical_less_than(
    left: tuple[int, ...],
    right: tuple[int, ...],
    target: int,
    workspace: tuple[int, ...],
    *,
    control: int | bool,
    btol: float,
    minimize_qubits: bool,
) -> tuple[ReversiblePhaseProgram, ComponentCall]:
    """Keep the source's narrow comparison and measured-ladder dispatch."""
    width = len(left)
    truncated = btol < width and btol > 1
    if truncated:
        width = ceil(btol) - 1
    skipped = btol < width and btol <= 1
    dedicated = (btol == inf or truncated) and len(workspace) < width
    gates = mixed_less_than_ir(
        left,
        right,
        target,
        workspace,
        control=control,
        measurement_bit=0,
        minimize_qubits=minimize_qubits,
        btol=btol,
        emit_resets=True,
    )
    n_qubits = 1 + max((q for gate in gates for q in gate.wires), default=target)
    program = lower_reversible_phase(CircuitIR(n_qubits, gates))
    if not dedicated and not skipped and width > 1:
        program = replace(
            program,
            operations=_ladder_scopes(program.operations, set(workspace[: width - 1])),
        )
    stop = len(program.operations)
    children = (ComponentCall("flip_if_lt_qq", 0, stop),) if dedicated else ()
    if truncated:
        children = (ComponentCall("flip_if_lt", 0, stop, children=children),)
    return program, ComponentCall("flip_if_lt", 0, stop, children=children)
