# SPDX-License-Identifier: Apache-2.0
# Adapted from the locked Kickmix generators.
# https://github.com/quantumlib/Qualtran/tree/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/gen/gf_arithmetic

"""Fixed-wire binary field transforms in the pinned Kickmix gate order."""

from math import gcd

from ....core.ir import Instruction
from .field_multiplication import (
    _field_plu,
    _field_power,
    _product,
    _remainder,
    _validate_field,
)
from .polynomial import _validate_registers


def _linear_map(
    wires: tuple[int, ...], columns: tuple[int, ...]
) -> tuple[Instruction, ...]:
    width = len(wires)
    matrix = [[(column >> row) & 1 for column in columns] for row in range(width)]
    try:
        upper, lower, permutation = _field_plu(matrix)
    except StopIteration:
        raise ValueError(
            "A reversible binary linear map requires an invertible matrix"
        ) from None
    gates = [
        Instruction("cx", (wires[j], wires[i]))
        for i in range(width)
        for j in range(i + 1, width)
        if upper[i][j]
    ]
    gates.extend(
        Instruction("cx", (wires[j], wires[i]))
        for i in reversed(range(width))
        for j in range(i)
        if lower[i][j]
    )
    current = list(range(width))
    for i in range(width):
        for j in range(i + 1, width):
            if permutation[current[j]] == i:
                gates.append(Instruction("swap", (wires[i], wires[j])))
                current[i], current[j] = current[j], current[i]
                break
    return tuple(gates)


def _inverse_columns(columns: tuple[int, ...]) -> tuple[int, ...]:
    width = len(columns)
    augmented = [column | (1 << (width + i)) for i, column in enumerate(columns)]
    for i in range(width):
        bit = 1 << i
        pivot = next((j for j in range(i, width) if augmented[j] & bit), None)
        if pivot is None:
            raise ValueError(
                "A reversible binary linear map requires an invertible matrix"
            )
        augmented[i], augmented[pivot] = augmented[pivot], augmented[i]
        for j, column in enumerate(augmented):
            if i != j and column & bit:
                augmented[j] ^= augmented[i]
    return tuple(column >> width for column in augmented)


def kickmix_binary_linear_map_ir(
    target: tuple[int, ...],
    columns: tuple[int, ...],
    *,
    adjoint: bool = False,
) -> tuple[Instruction, ...]:
    """Apply an invertible binary matrix in place, using fixed big-endian wires.

    Column zero multiplies the least-significant input bit. Bit zero of
    each column is the coefficient of the least-significant output bit.
    The adjoint synthesizes the inverse matrix in the source's PLU order.
    """
    _validate_registers(target)
    width = len(target)
    if len(columns) != width or any(
        type(column) is not int or not 0 <= column < 1 << width for column in columns
    ):
        raise ValueError("Binary matrix columns must fit a square register map")
    if type(adjoint) is not bool:
        raise ValueError("Binary linear map adjoint must be boolean")
    return _linear_map(target[::-1], _inverse_columns(columns) if adjoint else columns)


def kickmix_field_constant_scale_ir(
    target: tuple[int, ...], constant: int, *, modulus: int
) -> tuple[Instruction, ...]:
    """Multiply a big-endian word by a nonzero polynomial on fixed wires."""
    _validate_field(len(target), modulus)
    _validate_registers(target)
    if type(constant) is not int or constant < 0:
        raise ValueError("Field constant must be a nonnegative binary polynomial")
    constant = _remainder(constant, modulus)
    if not constant:
        raise ValueError("In-place field multiplication by zero is not reversible")
    return _linear_map(
        target[::-1],
        tuple(_product(1 << i, constant, modulus) for i in range(len(target))),
    )


def kickmix_field_frobenius_ir(
    target: tuple[int, ...],
    count: int = 1,
    *,
    modulus: int,
    adjoint: bool = False,
) -> tuple[Instruction, ...]:
    """Apply the selected Frobenius power on fixed big-endian register wires."""
    width = len(target)
    _validate_field(width, modulus)
    _validate_registers(target)
    if type(count) is not int or count < 0 or type(adjoint) is not bool:
        raise ValueError("Frobenius requires a nonnegative count and boolean adjoint")
    count = (-count if adjoint else count) % width
    if count == 0:
        return ()
    root = _remainder(2, modulus)
    for _ in range(count):
        root = _product(root, root, modulus)
    columns = []
    value = 1
    for _ in target:
        columns.append(value)
        value = _product(value, root, modulus)
    return _linear_map(target[::-1], tuple(columns))


def kickmix_field_monomial_scale_ir(
    target: tuple[int, ...], power: int, *, modulus: int
) -> tuple[Instruction, ...]:
    """Multiply by x**power using Kickmix's deferred cyclic rotation."""
    width = len(target)
    _validate_field(width, modulus)
    _validate_registers(target)
    if type(power) is not int or not modulus & 1:
        raise ValueError("Monomial scaling requires an integer power and nonzero root")
    if width == 1 or power == 0:
        return ()
    terms = tuple(i for i in range(1, width) if (modulus >> i) & 1)
    steps = abs(power)
    if steps > width * width // (len(terms) + 1):
        exponent = power % ((1 << width) - 1)
        factor = _field_power(2, exponent, modulus)
        return kickmix_field_constant_scale_ir(target, factor, modulus=modulus)

    wires = target[::-1]
    positions = range(-1, -steps - 1, -1) if power > 0 else range(steps)
    gates = [
        Instruction("cx", (wires[pos % width], wires[(term + pos) % width]))
        for pos in positions
        for term in terms
    ]
    rotation = power % width
    for start in range(gcd(width, rotation)):
        next_position = (start + rotation) % width
        while next_position != start:
            gates.append(Instruction("swap", (wires[start], wires[next_position])))
            next_position = (next_position + rotation) % width
    return tuple(gates)
