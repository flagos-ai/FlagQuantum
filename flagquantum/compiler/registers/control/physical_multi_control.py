# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from the locked Kickmix multi_control_x.pybind.cc.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/py/circuit/subs/multi_control_x.pybind.cc

"""Emit the source's ordered multi-control and multi-target X ladder."""

from ....core.ir import Instruction
from ...artifacts.reversible_phase import ReversiblePhaseProgram
from ...lowering.physical_trace import (
    PhysicalTrace,
    lower_physical_trace,
    scope_physical_trace,
)
from ..finite_field.coefficients import ClassicalFieldCoefficient
from ..layout.operand_word import bind_operand_word, bind_quantum_word, bind_z_word
from ..primitives._mixed_gates import _delete_and_ir, _flip
from ..primitives.measurement import reset_x_register_ir


def physical_multi_controlled_x(
    controls: object, target: object, *, workspace: tuple[int, ...] = ()
) -> ReversiblePhaseProgram:
    """Flip ordered targets when every control is true, then clear clean work.

    Controls require an explicit sequence. Workspace contains n-2 caller-owned
    clean leases in source allocation order for n quantum controls above two.
    The source frees these leases in forward order, reversing a reused LIFO pool.
    Repeated classical controls retain distinct outer condition records.
    """
    operands = bind_z_word(bind_operand_word(controls, sequence_only=True))
    targets = bind_quantum_word(target)
    quantum = tuple(item for item in operands if type(item) is int)
    guards = tuple(
        item.slot for item in operands if isinstance(item, ClassicalFieldCoefficient)
    )
    inputs = set(guards)
    bits = max(inputs, default=-1) + 1
    width = max((*quantum, *targets, *workspace), default=0) + 1
    if any(item is False for item in operands) or not targets:
        return ReversiblePhaseProgram(width, bits, (), tuple(sorted(inputs)))
    gates: list[Instruction] = []
    if len(quantum) <= 1:
        gates.extend(gate for output in targets for gate in _flip(output, *quantum))
    else:
        fanout = tuple(
            gate for output in targets[1:] for gate in _flip(output, targets[0])
        )
        gates.extend(fanout)
        if len(quantum) == 2:
            gates.extend(_flip(targets[0], *quantum))
        else:
            count = len(quantum) - 2
            if len(workspace) < count:
                raise ValueError("Multi-control X requires n-2 clean workspace wires")
            work = workspace[:count]
            occupied = (*quantum, *targets, *work)
            if any(type(wire) is not int or wire < 0 for wire in work) or len(
                set(occupied)
            ) != len(occupied):
                raise ValueError(
                    "Multi-control X requires distinct logical and work wires"
                )
            ladder = (quantum[0], *work)
            for k in range(1, len(ladder)):
                gates.extend(reset_x_register_ir((ladder[k],)))
                # LoopBuilder pattern_ccx reverses its two control operands.
                gates.extend(_flip(ladder[k], ladder[k - 1], quantum[k]))
            gates.extend(_flip(targets[0], quantum[-1], ladder[-1]))
            for k in reversed(range(1, len(ladder))):
                gates.extend(_delete_and_ir(ladder[k], bits, quantum[k], ladder[k - 1]))
        gates.extend(fanout)
    program, calls = lower_physical_trace(
        tuple(gates), 0, width - 1, PhysicalTrace(implicit_zero_bits=inputs)
    )
    scoped, _ = scope_physical_trace(program, calls, guards)
    return scoped
