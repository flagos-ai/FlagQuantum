# SPDX-License-Identifier: Apache-2.0
# Adapted from the locked Kickmix generators.
# https://github.com/quantumlib/Qualtran/tree/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/gen/gf_arithmetic

"""Fixed-wire binary field transforms in the pinned Kickmix gate order."""

from math import gcd

from ....core.ir import Instruction
from ..._component_trace.trace import ComponentCall
from ...artifacts.reversible_phase import ReversiblePhaseProgram
from ...lowering.physical_trace import PhysicalTrace, lower_physical_trace
from ._physical_calls import append_field_call
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


def binary_linear_map_ir(
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


def field_constant_scale_ir(
    target: tuple[int, ...],
    constant: int,
    *,
    modulus: int,
    divide: bool = False,
) -> tuple[Instruction, ...]:
    """Multiply or divide a big-endian word by a nonzero field polynomial."""
    width = len(target)
    _validate_field(width, modulus)
    _validate_registers(target)
    if type(constant) is not int or constant < 0:
        raise ValueError("Field constant must be a nonnegative binary polynomial")
    if type(divide) is not bool:
        raise ValueError("Field constant scaling direction must be boolean")
    constant = _remainder(constant, modulus)
    if not constant:
        operation = "division" if divide else "multiplication"
        raise ValueError(f"In-place field {operation} by zero is not reversible")
    if constant == 1:
        return ()
    if divide:
        constant = _field_power(constant, (1 << width) - 2, modulus)
    return _linear_map(
        target[::-1],
        tuple(_product(1 << i, constant, modulus) for i in range(width)),
    )


def append_field_constant_scale(
    gates: list[Instruction],
    calls: list[ComponentCall] | None,
    target: tuple[int, ...],
    constant: int,
    modulus: int,
    *,
    divide: bool = False,
) -> None:
    """Append the source's classical scale with its nested PLU call."""
    body = field_constant_scale_ir(target, constant, modulus=modulus, divide=divide)
    if body:
        name = "gf_idiv_classical" if divide else "gf_imul_classical"
        append_field_call(
            gates, calls, name, body, (ComponentCall("linear_map", 0, len(body)),)
        )


def physical_field_constant_scale(
    target: tuple[int, ...],
    constant: int,
    *,
    modulus: int,
    divide: bool = False,
) -> tuple[ReversiblePhaseProgram, tuple[ComponentCall, ...]]:
    """Export a constant multiplication or division with source block marks."""
    calls: list[ComponentCall] = []
    gates: list[Instruction] = []
    append_field_constant_scale(gates, calls, target, constant, modulus, divide=divide)
    return lower_physical_trace(
        tuple(gates),
        0,
        max(target),
        PhysicalTrace(calls=calls),
        reorder_feedback=False,
    )


def field_frobenius_ir(
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


def append_field_frobenius(
    gates: list[Instruction],
    calls: list[ComponentCall] | None,
    word: tuple[int, ...],
    count: int,
    modulus: int,
    *,
    adjoint: bool = False,
) -> None:
    """Append the exact map and mark it when the source would emit a call."""
    body = field_frobenius_ir(word, count, modulus=modulus, adjoint=adjoint)
    if count % len(word):
        name = "gf_ifrobenius_adjoint" if adjoint else "gf_ifrobenius"
        append_field_call(
            gates, calls, name, body, (ComponentCall("linear_map", 0, len(body)),)
        )


def physical_field_frobenius(
    target: tuple[int, ...],
    count: int = 1,
    *,
    modulus: int,
    adjoint: bool = False,
) -> tuple[ReversiblePhaseProgram, tuple[ComponentCall, ...]]:
    """Export the in-place transformation with its source linear-map mark."""
    calls: list[ComponentCall] = []
    gates: list[Instruction] = []
    append_field_frobenius(gates, calls, target, count, modulus, adjoint=adjoint)
    return lower_physical_trace(
        tuple(gates),
        0,
        max(target),
        PhysicalTrace(calls=calls),
        reorder_feedback=False,
    )


def field_monomial_scale_ir(
    target: tuple[int, ...], power: int, *, modulus: int
) -> tuple[Instruction, ...]:
    """Multiply by x**power using Kickmix's deferred cyclic rotation."""
    body, _ = _field_monomial_scale(target, power, modulus)
    return body


def _field_monomial_scale(
    target: tuple[int, ...], power: int, modulus: int
) -> tuple[tuple[Instruction, ...], str | None]:
    width = len(target)
    _validate_field(width, modulus)
    _validate_registers(target)
    if type(power) is not int or not modulus & 1:
        raise ValueError("Monomial scaling requires an integer power and nonzero root")
    if width == 1 or power == 0:
        return (), None
    terms = tuple(i for i in range(1, width) if (modulus >> i) & 1)
    steps = abs(power)
    if steps > width * width // (len(terms) + 1):
        exponent = power % ((1 << width) - 1)
        factor = _field_power(2, exponent, modulus)
        return field_constant_scale_ir(target, factor, modulus=modulus), "linear_map"

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
    return tuple(gates), "gf_imul_x" if power > 0 else "gf_idiv_x"


def append_field_monomial_scale(
    gates: list[Instruction],
    calls: list[ComponentCall] | None,
    target: tuple[int, ...],
    power: int,
    modulus: int,
) -> None:
    """Append the selected shift schedule and its source call boundary."""
    body, name = _field_monomial_scale(target, power, modulus)
    if name is not None:
        append_field_call(gates, calls, name, body)


def physical_field_monomial_scale(
    target: tuple[int, ...], power: int, *, modulus: int
) -> tuple[ReversiblePhaseProgram, tuple[ComponentCall, ...]]:
    """Export monomial multiplication or division as an in-place physical call."""
    calls: list[ComponentCall] = []
    gates: list[Instruction] = []
    append_field_monomial_scale(gates, calls, target, power, modulus)
    return lower_physical_trace(
        tuple(gates),
        0,
        max(target),
        PhysicalTrace(calls=calls),
        reorder_feedback=False,
    )
