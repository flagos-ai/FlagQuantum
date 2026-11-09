"""Validate and map compact, ordered Pauli rotations without sampling or state."""

import math

from ...core.ir import Instruction
from .._hybrid.model import Operation
from .profile import _fail

Support = tuple[tuple[int, str], ...]


def checked_support(value: object, width: int, op: Operation) -> Support:
    if not isinstance(value, tuple) or len(value) > width:
        _fail("register.pauli", "Pauli support must be an immutable bounded tuple", op)
    result: list[tuple[int, str]] = []
    seen: set[int] = set()
    for pair in value:
        if (
            not isinstance(pair, tuple)
            or len(pair) != 2
            or type(pair[0]) is not int
            or not 0 <= pair[0] < width
            or pair[0] in seen
            or not isinstance(pair[1], str)
            or pair[1] not in {"X", "Y", "Z"}
        ):
            _fail(
                "register.pauli",
                "Pauli indices must be distinct and bounded, with X/Y/Z axes",
                op,
            )
        seen.add(pair[0])
        result.append((pair[0], pair[1]))
    return tuple(result)


def pauli_rotation_instructions(
    qubits: tuple[int, ...],
    support: Support,
    angle: float,
    *,
    control: int | None = None,
) -> tuple[Instruction, ...]:
    """Map exp(-i angle P), preserving the identity phase and source axis order."""
    if not support:
        read = Instruction("rz", (qubits[-1],), {"theta": 0.0})
        if control is not None:
            return (read, Instruction("phase", (control,), {"theta": -angle}))
        flip = Instruction("x", (0,))
        phase = Instruction("phase", (0,), {"theta": -angle})
        return (read, flip, phase, flip, phase)
    before: list[Instruction] = []
    after: list[Instruction] = []
    for index, axis in support:
        target = qubits[-1 - index]
        if axis == "X":
            before.append(Instruction("h", (target,)))
            after.append(Instruction("h", (target,)))
        elif axis == "Y":
            before.append(Instruction("rx", (target,), {"theta": math.pi / 2}))
            after.append(Instruction("rx", (target,), {"theta": -math.pi / 2}))
    anchor = qubits[-1 - support[0][0]]
    parity = tuple(
        Instruction("cx", (qubits[-1 - index], anchor)) for index, _ in support[1:]
    )
    rotation = (
        Instruction("rz", (anchor,), {"theta": 2 * angle})
        if control is None
        else Instruction("crz", (control, anchor), {"theta": 2 * angle})
    )
    return (*before, *parity, rotation, *reversed(parity), *reversed(after))
