# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix gen_imul2_mod.cc.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/gen/modular_arithmetic/gen_imul2_mod.cc

"""Source space/tolerance dispatch for modular doubling and halving."""

from math import inf, isnan

from ....core.classical_dataflow import ClassicalXor
from ....core.ir import CircuitIR, Instruction
from ...artifacts.reversible_phase import (
    ReversiblePhaseOperation,
    ReversiblePhaseProgram,
)
from ...lowering.reversible_phase import lower_reversible_phase
from ..arithmetic.carry_parity import _Coefficient
from ..arithmetic.mixed_addition import _complement, kickmix_mixed_addition_ir
from ..bits.bit_rotation import kickmix_bit_rotation_ir
from ..comparison.mixed_comparison import kickmix_greater_than_ir
from ..finite_field.coefficients import ClassicalFieldCoefficient
from ..primitives._mixed_gates import _flip
from ..primitives.measurement import measure_x_clear_ir
from .exact_modular_doubling import (
    _modular_doubling_ports,
    kickmix_exact_modular_doubling_ir,
)


def kickmix_modular_doubling_program(
    target: tuple[int, ...],
    modulus: tuple[_Coefficient, ...],
    *,
    clean: tuple[int, ...] = (),
    dirty: tuple[int, ...] = (),
    control: int | None = None,
    inverse: bool = False,
    btol: float = inf,
    minimize_qubits: bool = False,
) -> ReversiblePhaseProgram:
    """Follow the source dispatch, retaining finite-tolerance phase behavior.

    Ports use the exact doubling contract. Caller-defined modulus slots are
    declared inputs. Known-zero leaf resets are omitted. Source fallback
    limitations, including an empty complemented half-modulus, are explicit
    errors; finite-tolerance results are not promised to be exact arithmetic.
    """
    quantum = _modular_doubling_ports(target, modulus, clean, control, inverse, dirty)
    if isnan(btol) or type(minimize_qubits) is not bool:
        raise ValueError(
            "Modular doubling requires ordered tolerance and boolean space mode"
        )
    n = len(target)
    enable = () if control is None else (control,)
    wires = 1 + max((*target, *quantum, *clean, *dirty, *enable))
    inputs = tuple(
        sorted(
            {bit.slot for bit in modulus if isinstance(bit, ClassicalFieldCoefficient)}
        )
    )
    result = max(inputs, default=-1) + 1
    slots = tuple(range(result, result + n + 3))
    prefix = tuple(ClassicalXor((), bit).instruction() for bit in inputs)
    operations: list[ReversiblePhaseOperation] = []
    used_bits = result

    def emit(gates: tuple[Instruction, ...]) -> None:
        nonlocal used_bits
        fragment = lower_reversible_phase(CircuitIR(wires, (*prefix, *gates)))
        operations.extend(fragment.operations[len(prefix) :])
        used_bits = max(used_bits, fragment.n_bits)

    def finish() -> ReversiblePhaseProgram:
        return ReversiblePhaseProgram(wires, used_bits, tuple(operations), inputs)

    tolerance = max(btol, 0.0) if inverse else btol + 1
    if n <= 2 or tolerance >= n and len(clean) + 2 >= n and not minimize_qubits:
        emit(
            kickmix_exact_modular_doubling_ir(
                target,
                modulus,
                clean=clean,
                control=control,
                inverse=inverse,
                measurement_bit=result,
            )
        )
        return finish()
    enabled = True if control is None else control
    outcome = ClassicalFieldCoefficient(result)
    ease = 0
    for coefficient in modulus:
        if coefficient is not True:
            break
        ease += 1

    def invert(word: tuple[_Coefficient, ...]) -> tuple[_Coefficient, ...]:
        complemented: list[_Coefficient] = []
        gates: list[Instruction] = []
        for value in reversed(word):
            changed, body = _complement(value)
            complemented.append(changed)
            gates.extend(body)
        emit(tuple(gates))
        return tuple(reversed(complemented))

    def trim(word: tuple[_Coefficient, ...]) -> tuple[_Coefficient, ...]:
        while word and word[0] is False:
            word = word[1:]
        if not word:
            raise ValueError(
                "Locked fallback cannot index an empty complemented half-modulus"
            )
        return word

    def add(word: tuple[_Coefficient, ...], flag: int, work: tuple[int, ...]) -> None:
        emit(
            kickmix_mixed_addition_ir(
                target,
                word,
                control=flag,
                clean=work,
                dirty=dirty,
                classical_bits=slots,
                minimize_qubits=minimize_qubits,
                btol=tolerance,
                subtract=inverse,
            )
        )

    def compare(
        word: tuple[_Coefficient, ...],
        flag: int | None,
        work: tuple[int, ...],
        bit: int = result,
    ) -> None:
        emit(
            kickmix_greater_than_ir(
                target,
                word,
                flag,
                work,
                control=enabled,
                measurement_bit=bit,
                minimize_qubits=minimize_qubits,
                btol=tolerance,
                low_space_bits=tuple(range(bit, bit + n + 3)),
            )
        )

    if not inverse:
        if not clean:
            raise ValueError(
                "Forward doubling fallback requires a clean comparison flag"
            )
        tolerance = min(tolerance, float(n))
        flag, work = clean[0], clean[1:]
        half = (False, *modulus[:-1])
        if ease > tolerance:
            emit(_flip(flag, enabled, target[0]))
        else:
            compare(half, flag, work)
        inverted_half = invert(half)
        add(trim(inverted_half[1:]), flag, work)
        invert(inverted_half)
        emit(_flip(target[0], flag))
        emit(kickmix_bit_rotation_ir(target, 1, control=control))
        emit(_flip(target[-1], flag))
        emit(
            (
                *measure_x_clear_ir(flag, result),
                *_flip(None, enabled, target[-1], outcome),
            )
        )
        return finish()

    if len(clean) < 2:
        raise ValueError("Inverse doubling fallback requires two clean flags")
    first, flag = clean[:2]
    work = clean[2:]
    emit(_flip(first, target[-1]))
    emit(_flip(flag, first, enabled))
    emit(kickmix_bit_rotation_ir(target, 1, control=control, right=True))
    half = trim(invert(modulus[:-1]))
    if ease <= tolerance + 20:
        tolerance += 1
    add(half, flag, work)
    half = invert(half)
    emit((*measure_x_clear_ir(flag, result), *_flip(None, first, enabled, outcome)))
    operations.append(ReversiblePhaseOperation("hmr", (first,), result_bit=result))
    used_bits = max(used_bits, result + 1)
    operations.append(
        ReversiblePhaseOperation("push_condition", (), condition_bit=result)
    )
    emit((*_flip(None, target[-1], enabled), *_flip(None, target[-1])))
    if ease > tolerance:
        emit(_flip(None, enabled, target[0]))
    else:
        padded = (True,) * (n - len(half)) + half
        compare((False, *padded[1:]), None, work, result + 1)
    operations.append(ReversiblePhaseOperation("pop_condition", ()))
    return finish()
