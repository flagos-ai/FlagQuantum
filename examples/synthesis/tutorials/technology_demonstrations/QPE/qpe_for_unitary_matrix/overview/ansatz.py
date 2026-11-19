"""Native QPE roots with explicit literal fractional register widths."""

from typing import Annotated

from flagquantum import synth
from flagquantum.synth import (
    Output,
    QArray,
    QBit,
    QNum,
    allocate,
    drop,
    qfunc,
    qpe,
    unitary,
)

from .physics import UNITARY


@qfunc
def precision_1(phase: Output[Annotated[QNum, (1, False, 1)]]) -> None:
    state: QArray[QBit] = QArray()
    allocate(3, state)
    allocate(phase)
    qpe(lambda: unitary(UNITARY, state), phase)
    drop(state)


@qfunc
def precision_2(phase: Output[Annotated[QNum, (2, False, 2)]]) -> None:
    state: QArray[QBit] = QArray()
    allocate(3, state)
    allocate(phase)
    qpe(lambda: unitary(UNITARY, state), phase)
    drop(state)


@qfunc
def precision_3(phase: Output[Annotated[QNum, (3, False, 3)]]) -> None:
    state: QArray[QBit] = QArray()
    allocate(3, state)
    allocate(phase)
    qpe(lambda: unitary(UNITARY, state), phase)
    drop(state)


@qfunc
def precision_4(phase: Output[Annotated[QNum, (4, False, 4)]]) -> None:
    state: QArray[QBit] = QArray()
    allocate(3, state)
    allocate(phase)
    qpe(lambda: unitary(UNITARY, state), phase)
    drop(state)


@qfunc
def precision_5(phase: Output[Annotated[QNum, (5, False, 5)]]) -> None:
    state: QArray[QBit] = QArray()
    allocate(3, state)
    allocate(phase)
    qpe(lambda: unitary(UNITARY, state), phase)
    drop(state)


@qfunc
def precision_6(phase: Output[Annotated[QNum, (6, False, 6)]]) -> None:
    state: QArray[QBit] = QArray()
    allocate(3, state)
    allocate(phase)
    qpe(lambda: unitary(UNITARY, state), phase)
    drop(state)


@qfunc
def precision_7(phase: Output[Annotated[QNum, (7, False, 7)]]) -> None:
    state: QArray[QBit] = QArray()
    allocate(3, state)
    allocate(phase)
    qpe(lambda: unitary(UNITARY, state), phase)
    drop(state)


@qfunc
def precision_8(phase: Output[Annotated[QNum, (8, False, 8)]]) -> None:
    state: QArray[QBit] = QArray()
    allocate(3, state)
    allocate(phase)
    qpe(lambda: unitary(UNITARY, state), phase)
    drop(state)


DEFINITIONS = (
    precision_1,
    precision_2,
    precision_3,
    precision_4,
    precision_5,
    precision_6,
    precision_7,
    precision_8,
)


def build_program(precision: int) -> synth.Program:
    if type(precision) is not int or not 1 <= precision <= 8:
        raise ValueError("use an integer precision in1..8")
    return synth.synthesize(
        DEFINITIONS[precision - 1], max_workspace=0, memory_limit_bytes=64 * 1024**2
    )
