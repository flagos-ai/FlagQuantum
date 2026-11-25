"""Native versions of the four state-preparation workshop stages."""

from typing import Annotated

from flagquantum.synth import (
    Output,
    QArray,
    QBit,
    QNum,
    allocate,
    hadamard_transform,
    prepare_state,
    qft,
    qfunc,
    select_rotation,
)


@qfunc
def arithmetic(
    x: Output[Annotated[QNum, 3]],
    y: Output[Annotated[QNum, 6]],
    z: Output[Annotated[QNum, 3]],
) -> None:
    allocate(x)
    allocate(z)
    hadamard_transform(x)
    hadamard_transform(z)
    y |= 2 * x + 4 * z + 2


@qfunc
def random_distribution(probabilities: list[float], x: Output[QArray[QBit]]) -> None:
    prepare_state(probabilities, 0.01, x)


@qfunc
def amplitude_assignment(
    angles: list[float],
    x: Output[Annotated[QNum, (5, False, 5)]],
    indicator: Output[QBit],
) -> None:
    allocate(x)
    hadamard_transform(x)
    allocate(indicator)
    select_rotation("Y", angles, x, indicator)


@qfunc
def exponential_transform(probabilities: list[float], x: Output[QArray[QBit]]) -> None:
    prepare_state(probabilities, 0, x)
    qft(x)
