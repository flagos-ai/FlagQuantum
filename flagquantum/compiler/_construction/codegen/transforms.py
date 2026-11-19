"""Decompose bounded real orthogonal transforms into registered gates."""

import math

from ....core._composition import controlled_instruction
from ....core.ir import Instruction


def _pattern_gate(
    gate: Instruction, wires: tuple[int, ...], pattern: int
) -> tuple[Instruction, ...]:
    """Apply a single-qubit gate on a specified pattern of the other qubits."""
    controls = tuple(wire for wire in wires if wire != gate.wires[0])
    flips = tuple(
        Instruction("x", (wire,))
        for index, wire in enumerate(wires)
        if wire in controls and not pattern & (1 << (len(wires) - 1 - index))
    )
    middle = controlled_instruction(gate, controls) if controls else (gate,)
    return (*flips, *middle, *flips)


def _type2_instructions(
    kind: str, data: tuple[int, ...], selector: tuple[int, ...]
) -> tuple[Instruction, ...]:
    """Factor a bounded real Type II transform into registered gates."""
    size = 1 << len(data)

    def matrix(*, sine: bool) -> list[list[float]]:
        trig = math.sin if sine else math.cos
        return [
            [
                math.sqrt((1 if row == (size - 1 if sine else 0) else 2) / size)
                * trig(math.pi * (row + int(sine)) * (column + 0.5) / size)
                for column in range(size)
            ]
            for row in range(size)
        ]

    if kind == "combined":
        zeros = [0.0] * size
        values = [row + zeros for row in matrix(sine=False)]
        values.extend(zeros + [-value for value in row] for row in matrix(sine=True))
    else:
        values = matrix(sine=kind == "sine")

    return _real_transform_instructions(values, (*selector, *data))


def _type1_instructions(wires: tuple[int, ...]) -> tuple[Instruction, ...]:
    half = 1 << (len(wires) - 1)
    boundary = half + 1
    size = half * 2
    scale = math.sqrt(2 / half)
    weights = [math.sqrt(0.5), *([1.0] * (half - 1)), math.sqrt(0.5)]
    values = [
        [
            scale
            * weights[row]
            * weights[column]
            * math.cos(math.pi * row * column / half)
            for column in range(boundary)
        ]
        + [0.0] * (size - boundary)
        for row in range(boundary)
    ]
    values.extend(
        [0.0] * boundary
        + [scale * math.sin(math.pi * row * column / half) for column in range(1, half)]
        for row in range(1, half)
    )
    return _real_transform_instructions(values, wires, imaginary_start=boundary)


def _real_transform_instructions(
    values: list[list[float]],
    wires: tuple[int, ...],
    *,
    imaginary_start: int | None = None,
) -> tuple[Instruction, ...]:
    """Factor a real orthogonal matrix, with optional phase i on its final block."""
    dimension = len(values)
    gray = tuple(index ^ (index >> 1) for index in range(dimension))
    rotations: list[tuple[int, int, float]] = []
    # Eliminating in Gray order makes each adjacent row rotation differ in one bit.
    for column_index, column in enumerate(gray):
        for index in range(dimension - 1, column_index, -1):
            upper, lower = gray[index - 1], gray[index]
            x, y = values[upper][column], values[lower][column]
            if abs(y) < 1e-15:
                continue
            angle = math.atan2(y, x)
            cosine, sine = math.cos(angle), math.sin(angle)
            for k, (left, right) in enumerate(
                zip(values[upper], values[lower], strict=True)
            ):
                values[upper][k] = cosine * left + sine * right
                values[lower][k] = -sine * left + cosine * right
            bit = upper ^ lower
            target = wires[len(wires) - bit.bit_length()]
            rotations.append((upper, target, -2 * angle if upper & bit else 2 * angle))

    gates: list[Instruction] = []
    for pattern in range(dimension):
        angle = math.pi if values[pattern][pattern] < 0 else 0.0
        # The final block phase commutes with the exact block-diagonal transform.
        if imaginary_start is not None and pattern >= imaginary_start:
            angle += math.pi / 2
        if not angle:
            continue
        target = wires[0]
        phase = Instruction("phase", (target,), {"theta": angle})
        flip = () if pattern & (dimension >> 1) else (Instruction("x", (target,)),)
        gates.extend((*flip, *_pattern_gate(phase, wires, pattern), *flip))
    for pattern, target, angle in reversed(rotations):
        gates.extend(
            _pattern_gate(
                Instruction("ry", (target,), {"theta": angle}), wires, pattern
            )
        )
    return tuple(gates)
