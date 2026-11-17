# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix gen_iadd.cc and gen_iadd.h.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/gen/adders/gen_iadd.cc

"""Export physical in-place addition and subtraction over fixed ports."""

from ....core.ir import CircuitIR
from ..._component_trace.trace import ComponentCall
from ...artifacts.reversible_phase import ReversiblePhaseProgram
from ...lowering.reversible_phase import lower_reversible_phase
from .inplace_addition import inplace_addition_ir


def physical_inplace_addition(
    target: tuple[int, ...],
    offset: tuple[int, ...],
    *,
    clean: tuple[int, ...] = (),
    dirty: tuple[int, ...] = (),
    control: int | bool = True,
    measurement_bit: int = 0,
    minimize_qubits: bool = False,
    subtract: bool = False,
    tag_name: str | None = None,
) -> tuple[ReversiblePhaseProgram, tuple[ComponentCall, ...]]:
    """Lower big-endian port bindings, including the source allocation resets.

    The caller owns register definitions. A blank tag suppresses the outer
    marks, and subtraction retains only its outer mark around the shared adder.
    """
    instructions = inplace_addition_ir(
        target,
        offset,
        clean=clean,
        dirty=dirty,
        control=control,
        measurement_bit=measurement_bit,
        minimize_qubits=minimize_qubits,
        subtract=subtract,
        emit_resets=True,
    )
    ports = (
        *target,
        *offset,
        *clean,
        *dirty,
        *((control,) if type(control) is int else ()),
    )
    bound = max(ports, default=0)
    width = 1 + max(
        (wire for gate in instructions for wire in gate.wires), default=bound
    )
    program = lower_reversible_phase(CircuitIR(max(width, bound + 1), instructions))
    name = ("isub" if subtract else "iadd") if tag_name is None else tag_name
    calls = (ComponentCall(name, 0, len(program.operations)),) if name else ()
    return program, calls
