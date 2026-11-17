# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix gen_iadd_classical.cc and CircuitBuilder.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/gen/adders/gen_iadd_classical.cc

"""Export mixed addition with its physical call and feedback boundaries."""

from ..._component_trace.trace import ComponentCall
from ...artifacts.reversible_phase import ReversiblePhaseProgram
from ...lowering.physical_trace import PhysicalTrace, lower_physical_trace
from ..finite_field.coefficients import ClassicalFieldCoefficient
from .measured_carry_addition import measured_carry_addition_ir
from .mixed_addition import mixed_addition_ir


def physical_classical_addition(
    target: tuple[int, ...],
    offset: tuple[ClassicalFieldCoefficient, ...],
    *,
    clean: tuple[int, ...],
    dirty: tuple[int, ...],
    control: int | bool,
    carry_in: int | None,
    minimize_qubits: bool,
    btol: float,
) -> tuple[ReversiblePhaseProgram, tuple[ComponentCall, ...]]:
    """Bind source physical classical register slots and arithmetic mode."""
    width = len(target)
    export = PhysicalTrace()
    if carry_in is None:
        gates = mixed_addition_ir(
            target,
            offset,
            clean=clean,
            dirty=dirty,
            control=control,
            classical_bits=tuple(range(width, 2 * width + 1)),
            minimize_qubits=minimize_qubits,
            btol=btol,
            export=export,
        )
    else:
        gates = measured_carry_addition_ir(
            target,
            offset,
            clean=clean,
            control=control,
            carry_in=carry_in,
            measurement_bit=width,
            emit_resets=True,
        )
        name = "iadd_classical_simple"
        export.calls.append(
            ComponentCall(f"c_{name}" if type(control) is int else name, 0, len(gates))
        )
    return lower_physical_trace(gates, width, 0, export)
