# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix gen_ineg_mod.cc.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/gen/modular_arithmetic/gen_ineg_mod.cc

"""Modular negation with the locked zero-repair and measured phase cleanup."""

from ....core.classical_dataflow import ClassicalXor
from ....core.ir import CircuitIR, Instruction
from ...artifacts.reversible_phase import (
    ReversiblePhaseOperation,
    ReversiblePhaseProgram,
)
from ...lowering.reversible_phase import lower_reversible_phase
from ..arithmetic.carry_parity import _Coefficient
from ..arithmetic.mixed_addition import kickmix_mixed_addition_ir
from ..comparison.mixed_comparison import kickmix_quantum_equal_ir
from ..finite_field.coefficients import ClassicalFieldCoefficient
from ..finite_field.polynomial import _validate_registers
from ..primitives._mixed_gates import _flip


def kickmix_modular_negation_program(
    target: tuple[int, ...],
    modulus: tuple[_Coefficient, ...],
    *,
    clean: tuple[int, ...],
    dirty: tuple[int, ...] = (),
) -> ReversiblePhaseProgram:
    """Negate a big-endian word in [0, modulus), preserving the modulus inputs.

    The highest modulus coefficient is literal True, as required by the
    locked source. Lower coefficients may be constants, quantum wires or
    caller-defined classical slots. The caller supplies initially zero
    clean wires. Leaf resets on known-zero work use that precondition.
    The native extra-clean addition branch retains its recorded limitations.
    """
    if not target or len(target) != len(modulus) or modulus[0] is not True:
        raise ValueError("Modular negation requires a matching full-capacity modulus")
    if not clean:
        raise ValueError("Modular negation requires a clean zero-repair flag")
    quantum_inputs = tuple(
        dict.fromkeys(value for value in modulus if type(value) is int)
    )
    _validate_registers(target, quantum_inputs, clean, dirty)
    wires = 1 + max((*target, *quantum_inputs, *clean, *dirty))
    inputs = tuple(
        sorted(
            {
                value.slot
                for value in modulus
                if isinstance(value, ClassicalFieldCoefficient)
            }
        )
    )
    result = max(inputs, default=-1) + 1
    prefix = tuple(ClassicalXor((), bit).instruction() for bit in inputs)
    ops = [ReversiblePhaseOperation("x", (wire,)) for wire in reversed(target)]
    bits = result

    def emit(gates: tuple[Instruction, ...]) -> None:
        nonlocal bits
        fragment = lower_reversible_phase(CircuitIR(wires, (*prefix, *gates)))
        ops.extend(fragment.operations[len(prefix) :])
        bits = max(bits, fragment.n_bits)

    emit(
        kickmix_mixed_addition_ir(
            target,
            modulus,
            clean=clean,
            dirty=dirty,
            carry_in=True,
            classical_bits=tuple(range(result, result + len(target) + 3)),
        )
    )
    flag = clean[0]
    ops.append(ReversiblePhaseOperation("r", (flag,)))
    emit(
        kickmix_quantum_equal_ir(
            target,
            modulus,
            flag,
            clean[1:],
            measurement_bit=result,
        )
    )
    emit(
        tuple(
            gate
            for wire, coefficient in zip(
                reversed(target), reversed(modulus), strict=True
            )
            for gate in _flip(wire, flag, coefficient)
        )
    )
    ops.append(ReversiblePhaseOperation("hmr", (flag,), result_bit=result))
    ops.append(ReversiblePhaseOperation("push_condition", (), condition_bit=result))
    emit(
        kickmix_quantum_equal_ir(
            target,
            (False,) * len(target),
            None,
            clean,
            measurement_bit=result + 1,
        )
    )
    ops.append(ReversiblePhaseOperation("pop_condition", ()))
    return ReversiblePhaseProgram(wires, max(bits, result + 1), tuple(ops), inputs)
