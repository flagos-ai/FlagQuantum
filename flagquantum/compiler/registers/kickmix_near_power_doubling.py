# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix gen_idouble_mod_approx.cc.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/gen/modular_arithmetic/gen_idouble_mod_approx.cc

"""Adjacent-swap near-power-of-two modular doubling and halving schedules."""

from math import ceil, isfinite

from ...core.ir import Instruction
from .finite_field import ClassicalFieldCoefficient
from .kickmix_carry_xor import _Coefficient
from .kickmix_mixed_addition import kickmix_mixed_addition_ir
from .polynomial import _validate_registers


def kickmix_near_power_doubling_ir(
    target: tuple[int, ...],
    modulus: tuple[_Coefficient, ...],
    *,
    btol: float,
    clean: tuple[int, ...],
    dirty: tuple[int, ...] = (),
    classical_bits: tuple[int, ...] = (),
    inverse: bool = False,
    minimize_qubits: bool = False,
) -> tuple[Instruction, ...]:
    """Port the source clipped add/subtract with carry-in and low-bit enable.

    Ports are big-endian. The lowest modulus coefficient must be literal
    True. Remove leading literal True coefficients to find the cutoff, then
    use bottom_span = cutoff + ceil(btol). The caller supplies at least that
    many clean wires and the mixed adder's remaining work/slot requirements.
    Clean wires begin at zero, dirty wires are borrowed, and modulus inputs
    are preserved. Finite precision retains the source approximation and its
    known mixed-adder boundaries; it does not promise exact modular arithmetic.
    """
    if not target or not modulus or modulus[-1] is not True:
        raise ValueError(
            "Near-power doubling requires nonempty words and a literal odd modulus"
        )
    if (
        not isfinite(btol)
        or type(inverse) is not bool
        or type(minimize_qubits) is not bool
    ):
        raise ValueError(
            "Near-power doubling requires finite tolerance and boolean modes"
        )
    if any(not isinstance(bit, (int, ClassicalFieldCoefficient)) for bit in modulus):
        raise ValueError("Modulus coefficients must be bits or quantum wires")
    cutoff = len(modulus)
    for bit in modulus:
        if bit is not True:
            break
        cutoff -= 1
    if cutoff + btol > len(modulus):
        raise ValueError("Modulus cutoff plus tolerance exceeds the modulus width")
    bottom_span = cutoff + ceil(btol)
    if not 1 <= bottom_span <= len(target):
        raise ValueError(
            "Near-power doubling requires a nonempty clipped span within the target"
        )
    if len(clean) < bottom_span:
        raise ValueError("Near-power doubling requires bottom_span clean wires")
    quantum = tuple(dict.fromkeys(bit for bit in modulus if type(bit) is int))
    _validate_registers(target, quantum, clean, dirty)
    low = tuple(reversed(target))
    order = range(1, len(target)) if inverse else range(len(target) - 1, 0, -1)
    swaps = tuple(Instruction("swap", (low[k], low[k - 1])) for k in order)
    arithmetic = kickmix_mixed_addition_ir(
        target[len(target) - bottom_span : -1],
        modulus[:-1],
        carry_in=True,
        control=target[-1],
        clean=clean,
        dirty=dirty,
        classical_bits=classical_bits,
        minimize_qubits=minimize_qubits,
        btol=btol,
        subtract=not inverse,
    )
    return (*arithmetic, *swaps) if inverse else (*swaps, *arithmetic)
