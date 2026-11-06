"""Apply a quantum angle word to an adjacent fermionic Givens rotation."""

from ...core.ir import Instruction
from .arithmetic import measurement_add_or_subtract_ir
from .bitwise import _validate_wires


def programmed_givens_ir(
    left: int,
    right: int,
    angle: tuple[int, ...],
    gradient: tuple[int, ...],
    carries: tuple[int, ...],
    slots: tuple[int, ...],
    *,
    adjoint: bool = False,
) -> tuple[Instruction, ...]:
    """Apply exp(iθ XY) exp(-iθ YX) for θ equal to a binary fraction of a turn.

    The angle is read-only. The gradient must be prepared with exponent -1.
    It is unchanged after the two additions, and the carry qubits clear.
    """
    _validate_wires((left, right, *angle, *gradient, *carries))
    width = len(angle)
    if not width or len(gradient) != width or len(carries) != width - 1:
        raise ValueError(
            "Programmed Givens requires matching angle and gradient widths"
        )
    if type(adjoint) is not bool:
        raise ValueError("Programmed Givens direction must be boolean")
    h = (Instruction("h", (left,)), Instruction("h", (right,)))
    parity = Instruction("cx", (left, right))
    rotations = (
        ((left, False), (right, True)) if adjoint else ((right, False), (left, True))
    )
    return tuple(
        gate
        for y_wire, negate in rotations
        for gate in (
            Instruction("sdg", (y_wire,)),
            *h,
            parity,
            *measurement_add_or_subtract_ir(
                right, angle, gradient, carries, slots, add_when_ctrl_is_on=negate
            ),
            parity,
            *reversed(h),
            Instruction("s", (y_wire,)),
        )
    )
