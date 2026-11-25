"""Compress both source register sizes with native bounded functions."""

import math
from typing import Literal

from flagquantum import synth
from flagquantum.synth import (
    CX,
    Output,
    QArray,
    QBit,
    R,
    X,
    allocate,
    control,
    hadamard_transform,
    phase,
    prepare_amplitudes,
    qfunc,
    unitary,
)

from ..main import controlled_h

Scenario = Literal["uniform", "one_excitation", "nonsymmetric"]
Matrix = tuple[tuple[float, ...], ...]
SOURCE_WEIGHTS = (0.0, 1 / 3, 1 / 3, 0.0, 1 / 3, 0.0, 0.0, 0.0)


def _two_level(
    first: int, second: int, block: tuple[tuple[float, float], tuple[float, float]]
) -> Matrix:
    matrix = [[float(row == col) for col in range(8)] for row in range(8)]
    for row, index in enumerate((first, second)):
        for col, value in zip((first, second), block[row], strict=True):
            matrix[index][col] = value
    return tuple(tuple(row) for row in matrix)


def adjacent_rotation(size: int, position: int) -> Matrix:
    """Mix the two specified words with their binomial multiplicities."""
    scale = math.comb(size, position + 1)
    x = math.sqrt(math.comb(size - 1, position + 1) / scale)
    y = math.sqrt(math.comb(size - 1, position) / scale)
    return _two_level(2, 5, ((x, y), (y, -x)))


def endpoint_rotation(size: int) -> Matrix:
    """Complete the weight-one step for this prefix."""
    x, y = math.sqrt((size - 1) / size), 1 / math.sqrt(size)
    return _two_level(1, 4, ((x, y), (-y, x)))


U31, F3 = adjacent_rotation(3, 1), endpoint_rotation(3)
U41, U42, F4 = adjacent_rotation(4, 1), adjacent_rotation(4, 2), endpoint_rotation(4)
U51, U52, U53, F5 = (
    adjacent_rotation(5, 1),
    adjacent_rotation(5, 2),
    adjacent_rotation(5, 3),
    endpoint_rotation(5),
)

# The selected bit locations are not contiguous in the five-bit register.
TWO_CONTROL_X = tuple(
    tuple(float(row == (col ^ 4 if col & 3 == 3 else col)) for col in range(8))
    for row in range(8)
)


@qfunc
def compress(bits: int, state: QArray[QBit]) -> None:
    CX(state[1], state[0])
    controlled_h(state[0], state[1])
    unitary(U31, (state[0], state[1], state[2]))
    unitary(F3, (state[0], state[1], state[2]))
    CX(state[2], state[1])
    if bits == 5:
        unitary(U41, (state[0], state[1], state[3]))
        unitary(U42, (state[1], state[2], state[3]))
        unitary(F4, (state[0], state[2], state[3]))
        CX(state[3], state[2])
        unitary(U51, (state[0], state[1], state[4]))
        unitary(U52, (state[1], state[2], state[4]))
        unitary(U53, (state[2], state[3], state[4]))
        unitary(F5, (state[0], state[3], state[4]))
        CX(state[4], state[3])
    CX(state[2], state[1])
    CX(state[2], state[0])
    control(state[:2], lambda: X(state[2]))
    if bits == 5:
        CX(state[3], state[2])
        CX(state[2], state[3])
        CX(state[4], state[2])
        CX(state[4], state[0])
        unitary(TWO_CONTROL_X, (state[0], state[2], state[4]))


@qfunc
def uniform(x: Output[QArray[QBit]]) -> None:
    allocate(5, x)
    hadamard_transform(x)
    compress(5, x)


@qfunc
def one_excitation(values: list[float], x: Output[QArray[QBit]]) -> None:
    prepare_amplitudes(values, 0, x)
    compress(3, x)


@qfunc
def nonsymmetric(x: Output[QArray[QBit]]) -> None:
    allocate(3, x)
    hadamard_transform(x)
    R(math.pi, math.pi / 2, x[0])
    phase(1, math.pi / 2)
    compress(3, x)


def source_amplitudes() -> tuple[float, ...]:
    """Normalize the original weights while preserving their relative phase."""
    norm = math.sqrt(math.fsum(weight**2 for weight in SOURCE_WEIGHTS))
    return tuple(weight / norm for weight in SOURCE_WEIGHTS)


def build_program(scenario: Scenario) -> synth.Program:
    """Build the named source case with zero work qubits."""
    if scenario == "uniform":
        return synth.synthesize(uniform, max_qubits=5, max_workspace=0)
    if scenario == "one_excitation":
        return synth.synthesize(
            one_excitation,
            classical_inputs={"values": source_amplitudes()},
            max_qubits=3,
            max_workspace=0,
        )
    if scenario == "nonsymmetric":
        return synth.synthesize(nonsymmetric, max_qubits=3, max_workspace=0)
    raise ValueError("Unknown symmetric compression scenario")
