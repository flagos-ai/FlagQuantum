# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix gen_iadd_mod.cc.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/gen/modular_arithmetic/gen_iadd_mod.cc

"""Compose the locked modular addition and subtraction schedules."""

from math import inf, isnan

from ....core.classical_dataflow import ClassicalXor
from ....core.ir import CircuitIR, Instruction
from ...artifacts.reversible_phase import (
    ReversiblePhaseOperation,
    ReversiblePhaseProgram,
)
from ...lowering.reversible_phase import lower_reversible_phase
from ..arithmetic.carry_parity import _Coefficient
from ..arithmetic.inplace_addition import kickmix_inplace_addition_ir
from ..arithmetic.mixed_addition import kickmix_mixed_addition_ir
from ..comparison.mixed_comparison import kickmix_greater_than_ir, kickmix_less_than_ir
from ..finite_field.coefficients import ClassicalFieldCoefficient
from ..finite_field.polynomial import _validate_registers


def kickmix_modular_addition_program(
    target: tuple[int, ...],
    offset: tuple[int, ...],
    modulus: tuple[_Coefficient, ...],
    *,
    clean: tuple[int, ...],
    dirty: tuple[int, ...] = (),
    control: int | bool = True,
    btol: float = inf,
    subtract: bool = False,
) -> ReversiblePhaseProgram:
    """Build the fixed-wire modular operation for inputs in [0, modulus).

    Words use big-endian order. Modulus coefficients may be constants,
    caller-defined classical bits, or quantum wires. Clean work starts at
    zero. Leaf resets on known-zero work are represented by that precondition.
    The returned artifact preserves the source effect scopes and their saved
    guards. Local execution uses bind_reversible_phase_execution.
    """
    n = len(target)
    if not n or len(offset) != n or len(modulus) != n:
        raise ValueError("Modular addition requires equally wide nonempty words")
    if len(clean) < 2:
        raise ValueError("Modular addition requires at least two clean wires")
    if isnan(btol):
        raise ValueError("Modular addition requires an ordered bit tolerance")
    quantum_modulus = tuple(dict.fromkeys(bit for bit in modulus if type(bit) is int))
    enable = (control,) if type(control) is int else ()
    _validate_registers(target, offset, quantum_modulus, clean, dirty, enable)
    wires = 1 + max((*target, *offset, *quantum_modulus, *clean, *dirty, *enable))
    inputs = tuple(
        sorted(
            {bit.slot for bit in modulus if isinstance(bit, ClassicalFieldCoefficient)}
        )
    )
    result = max(inputs, default=-1) + 1
    slots = tuple(range(result, result + n + 3))
    if control is False:
        return ReversiblePhaseProgram(wires, result, (), inputs)

    prefix = tuple(ClassicalXor((), bit).instruction() for bit in inputs)
    operations: list[ReversiblePhaseOperation] = []
    used_width = result

    def emit(gates: tuple[Instruction, ...]) -> None:
        nonlocal used_width
        fragment = lower_reversible_phase(CircuitIR(wires, (*prefix, *gates)))
        operations.extend(fragment.operations[len(prefix) :])
        used_width = max(used_width, fragment.n_bits)

    def hmr(wire: int) -> None:
        nonlocal used_width
        operations.append(ReversiblePhaseOperation("hmr", (wire,), result_bit=result))
        used_width = max(used_width, result + 1)

    sign, auxiliary = clean[:2]
    extended = (sign, *target)
    modulus_extended = (False, *modulus)
    tolerance = btol + 1
    if subtract:
        carry_work = clean[2:]
        operations.append(ReversiblePhaseOperation("r", (sign,)))
        emit(
            kickmix_inplace_addition_ir(
                extended,
                offset,
                clean=carry_work,
                dirty=dirty,
                control=control,
                measurement_bit=result,
                subtract=True,
            )
        )
        emit(
            kickmix_mixed_addition_ir(
                target,
                modulus,
                clean=carry_work,
                dirty=(*offset[::-1], *dirty),
                classical_bits=slots,
                control=sign,
                btol=tolerance,
            )
        )
        hmr(sign)
        operations.append(
            ReversiblePhaseOperation("push_condition", (), condition_bit=result)
        )
        operations.append(ReversiblePhaseOperation("r", (sign,)))
        extended_offset = (auxiliary, *offset)
        emit(
            kickmix_inplace_addition_ir(
                extended,
                extended_offset,
                clean=carry_work,
                dirty=dirty,
                measurement_bit=result,
            )
        )
        emit(
            kickmix_greater_than_ir(
                extended,
                modulus_extended,
                None,
                carry_work,
                or_equal=True,
                control=control,
                btol=tolerance,
                measurement_bit=result,
                low_space_bits=slots,
            )
        )
        emit(
            kickmix_inplace_addition_ir(
                extended,
                extended_offset,
                clean=carry_work,
                dirty=dirty,
                measurement_bit=result,
                subtract=True,
            )
        )
        operations.append(ReversiblePhaseOperation("pop_condition", ()))
        hmr(sign)
        hmr(auxiliary)
    else:
        emit(
            kickmix_inplace_addition_ir(
                extended,
                offset,
                clean=clean[1:],
                dirty=dirty,
                control=control,
                measurement_bit=result,
            )
        )
        operations.append(ReversiblePhaseOperation("r", (auxiliary,)))
        emit(
            kickmix_greater_than_ir(
                extended,
                modulus_extended,
                auxiliary,
                clean[2:],
                or_equal=True,
                btol=tolerance,
                measurement_bit=result,
                low_space_bits=slots,
            )
        )
        emit(
            kickmix_mixed_addition_ir(
                extended,
                modulus_extended,
                clean=clean[2:],
                dirty=(*offset[::-1], *dirty),
                classical_bits=slots,
                control=auxiliary,
                btol=tolerance,
                subtract=True,
            )
        )
        hmr(sign)
        hmr(auxiliary)
        operations.append(
            ReversiblePhaseOperation("push_condition", (), condition_bit=result)
        )
        emit(
            kickmix_less_than_ir(
                target,
                offset,
                None,
                clean,
                control=control,
                btol=tolerance,
                measurement_bit=result + 1,
                low_space_bits=slots[1:],
            )
        )
        operations.append(ReversiblePhaseOperation("pop_condition", ()))
    return ReversiblePhaseProgram(wires, used_width, tuple(operations), inputs)
