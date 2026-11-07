# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix gen_imul2_mod.cc exact forward and inline inverse.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/gen/modular_arithmetic/gen_imul2_mod.cc

"""Exact modular doubling schedules with source-ordered carry cleanup."""

from ...core.ir import Instruction
from ._kickmix_comparison_gates import _flip
from .finite_field import ClassicalFieldCoefficient
from .kickmix_bit_rotation import kickmix_bit_rotation_ir
from .kickmix_carry_xor import _Coefficient
from .measurement import measure_x_clear_ir
from .polynomial import _validate_registers


def kickmix_exact_modular_doubling_ir(
    target: tuple[int, ...],
    modulus: tuple[_Coefficient, ...],
    *,
    clean: tuple[int, ...] = (),
    control: int | None = None,
    measurement_bit: int = 0,
    inverse: bool = False,
) -> tuple[Instruction, ...]:
    """Map x to 2*x or its modular inverse on 0 <= x < an odd modulus.

    The equally wide words are big-endian and both modulus endpoints are
    literal True. Interior coefficients may be constants, quantum wires or
    caller-defined classical slots. At width n > 2 the clean requirement is
    n-2 without a control and n-1 with a control. Any remaining clean wire
    serves the source's measured triple-control decomposition. Known-zero
    work resets are omitted. Finite-tolerance schedules are separate.
    """
    n = len(target)
    if not n or len(modulus) != n or modulus[0] is not True or modulus[-1] is not True:
        raise ValueError(
            "Modular doubling requires a matching odd full-capacity modulus"
        )
    if control is not None and type(control) is not int:
        raise ValueError("Modular doubling control must be a quantum wire")
    if type(inverse) is not bool:
        raise ValueError("Modular doubling direction must be boolean")
    if any(not isinstance(bit, (int, ClassicalFieldCoefficient)) for bit in modulus):
        raise ValueError("Modulus coefficients must be bits or quantum wires")
    quantum = tuple(dict.fromkeys(bit for bit in modulus if type(bit) is int))
    _validate_registers(target, quantum, clean, () if control is None else (control,))
    if n == 1:
        return ()
    low = tuple(reversed(target))
    if n == 2:
        return (
            (
                Instruction("swap", low)
                if control is None
                else Instruction("cswap", (control, *low))
            ),
        )
    needed = n - 2 + (control is not None)
    if len(clean) < needed:
        raise ValueError(
            "Exact modular doubling requires n-2 clean wires plus a control flag"
        )
    input_slots = {
        bit.slot for bit in modulus if isinstance(bit, ClassicalFieldCoefficient)
    }
    if (
        type(measurement_bit) is not int
        or measurement_bit < 0
        or measurement_bit in input_slots
    ):
        raise ValueError("Modular doubling requires a fresh nonnegative result slot")
    work = clean[: n - 2]
    flag = low[0] if control is None else clean[n - 2]
    extra = clean[needed:]
    coefficients = tuple(reversed(modulus))
    enabled = True if control is None else control
    outcome = ClassicalFieldCoefficient(measurement_bit)
    gates: list[Instruction] = []

    def flip(dest: int | None, *controls: _Coefficient) -> None:
        gates.extend(_flip(dest, *controls))

    def erase(wire: int, *controls: _Coefficient) -> None:
        gates.extend(measure_x_clear_ir(wire, measurement_bit))
        flip(None, *controls, outcome)

    def triple() -> None:
        gates.extend(
            _flip(
                low[0],
                enabled,
                low[n - 1],
                work[n - 3],
                scratch=extra[0] if extra else None,
                measurement_bit=measurement_bit,
            )
        )

    if inverse:
        if control is not None:
            flip(flag, control, low[0])
        flip(low[1], flag)
        flip(low[1], flag, coefficients[1])
        flip(low[0])
        if control is not None:
            flip(flag, control)
        flip(work[0], coefficients[1], low[1])
        flip(work[0], low[1])
        flip(work[0])
        for k in range(n - 2, 1, -1):
            flip(low[k], enabled, coefficients[k])
        for k in range(2, n - 1):
            flip(work[k - 2], coefficients[k])
            flip(low[k], flag, work[k - 2])
            flip(work[k - 1], low[k], work[k - 2])
            flip(work[k - 1], coefficients[k])
        if control is not None:
            flip(flag, control)
        flip(low[0])
        flip(work[n - 3])
        flip(low[n - 1], flag, work[n - 3])
        if control is not None:
            erase(flag, control, low[0])
        triple()
        flip(work[n - 3])
        for k in range(n - 2, 1, -1):
            flip(work[k - 1], coefficients[k])
            erase(work[k - 1], low[k], work[k - 2])
            flip(work[k - 2], coefficients[k])
            flip(low[k], enabled, work[k - 2])
        gates.extend(measure_x_clear_ir(work[0], measurement_bit))
        flip(None, low[1], coefficients[1], outcome)
        flip(None, low[1], outcome)
        flip(None, outcome)
        gates.extend(kickmix_bit_rotation_ir(target, 1, control=control, right=True))
        return tuple(gates)

    gates.extend(kickmix_bit_rotation_ir(target, 1, control=control))
    flip(work[0])
    flip(work[0], low[1])
    flip(work[0], low[1], coefficients[1])
    for k in range(n - 3):
        flip(low[k + 2], enabled, work[k])
        flip(work[k], coefficients[k + 2])
        flip(work[k + 1], low[k + 2], work[k])
        flip(work[k + 1], coefficients[k + 2])
    flip(work[n - 3])
    triple()
    if control is not None:
        flip(flag, control, low[0])
    flip(low[n - 1], flag, work[n - 3])
    flip(work[n - 3])
    flip(low[0])
    if control is not None:
        flip(flag, control)
    for k in range(n - 4, -1, -1):
        flip(work[k + 1], coefficients[k + 2])
        erase(work[k + 1], low[k + 2], work[k])
        flip(low[k + 2], flag, work[k])
        flip(work[k], coefficients[k + 2])
    for k in range(2, n - 1):
        flip(low[k], enabled, coefficients[k])
    gates.extend(measure_x_clear_ir(work[0], measurement_bit))
    flip(None, coefficients[1], low[1], outcome)
    flip(None, low[1], outcome)
    flip(None, outcome)
    if control is not None:
        flip(flag, control)
    flip(low[0])
    flip(low[1], flag, coefficients[1])
    flip(low[1], flag)
    if control is not None:
        erase(flag, control, low[0])
    return tuple(gates)
