"""Literal fractional QPE roots keep the existing annotation contract explicit."""

from typing import Annotated

from flagquantum import synth
from flagquantum.synth import (
    Output,
    QBit,
    QNum,
    allocate,
    apply,
    prepare_amplitudes,
    qfunc,
    qpe,
    select_rotation,
    unitary,
    within,
)

from .physics import AMPLITUDES, PRECISIONS, UNITARY, reciprocal_angles


@qfunc
def precision_2(
    amplitudes: list[float],
    angles: list[float],
    res: Output[Annotated[QNum, 3]],
    phase: Output[Annotated[QNum, (2, False, 2)]],
    indicator: Output[QBit],
) -> None:
    prepare_amplitudes(amplitudes, 0, res)
    allocate(phase)
    allocate(indicator)
    with within():
        qpe(lambda: unitary(UNITARY, res), phase)
    with apply():
        select_rotation("Y", angles, phase, indicator)


@qfunc
def precision_3(
    amplitudes: list[float],
    angles: list[float],
    res: Output[Annotated[QNum, 3]],
    phase: Output[Annotated[QNum, (3, False, 3)]],
    indicator: Output[QBit],
) -> None:
    prepare_amplitudes(amplitudes, 0, res)
    allocate(phase)
    allocate(indicator)
    with within():
        qpe(lambda: unitary(UNITARY, res), phase)
    with apply():
        select_rotation("Y", angles, phase, indicator)


@qfunc
def precision_4(
    amplitudes: list[float],
    angles: list[float],
    res: Output[Annotated[QNum, 3]],
    phase: Output[Annotated[QNum, (4, False, 4)]],
    indicator: Output[QBit],
) -> None:
    prepare_amplitudes(amplitudes, 0, res)
    allocate(phase)
    allocate(indicator)
    with within():
        qpe(lambda: unitary(UNITARY, res), phase)
    with apply():
        select_rotation("Y", angles, phase, indicator)


@qfunc
def precision_5(
    amplitudes: list[float],
    angles: list[float],
    res: Output[Annotated[QNum, 3]],
    phase: Output[Annotated[QNum, (5, False, 5)]],
    indicator: Output[QBit],
) -> None:
    prepare_amplitudes(amplitudes, 0, res)
    allocate(phase)
    allocate(indicator)
    with within():
        qpe(lambda: unitary(UNITARY, res), phase)
    with apply():
        select_rotation("Y", angles, phase, indicator)


@qfunc
def precision_6(
    amplitudes: list[float],
    angles: list[float],
    res: Output[Annotated[QNum, 3]],
    phase: Output[Annotated[QNum, (6, False, 6)]],
    indicator: Output[QBit],
) -> None:
    prepare_amplitudes(amplitudes, 0, res)
    allocate(phase)
    allocate(indicator)
    with within():
        qpe(lambda: unitary(UNITARY, res), phase)
    with apply():
        select_rotation("Y", angles, phase, indicator)


@qfunc
def precision_7(
    amplitudes: list[float],
    angles: list[float],
    res: Output[Annotated[QNum, 3]],
    phase: Output[Annotated[QNum, (7, False, 7)]],
    indicator: Output[QBit],
) -> None:
    prepare_amplitudes(amplitudes, 0, res)
    allocate(phase)
    allocate(indicator)
    with within():
        qpe(lambda: unitary(UNITARY, res), phase)
    with apply():
        select_rotation("Y", angles, phase, indicator)


@qfunc
def precision_8(
    amplitudes: list[float],
    angles: list[float],
    res: Output[Annotated[QNum, 3]],
    phase: Output[Annotated[QNum, (8, False, 8)]],
    indicator: Output[QBit],
) -> None:
    prepare_amplitudes(amplitudes, 0, res)
    allocate(phase)
    allocate(indicator)
    with within():
        qpe(lambda: unitary(UNITARY, res), phase)
    with apply():
        select_rotation("Y", angles, phase, indicator)


ROOTS = (
    precision_2,
    precision_3,
    precision_4,
    precision_5,
    precision_6,
    precision_7,
    precision_8,
)


def build_program(precision: int) -> synth.Program:
    if type(precision) is not int or precision not in PRECISIONS:
        raise ValueError("precision must be an integer from two through eight")
    return synth.synthesize(
        ROOTS[precision - 2],
        classical_inputs={
            "amplitudes": list(AMPLITUDES),
            "angles": list(reciprocal_angles(precision)),
        },
        max_qubits=precision + 4,
        max_workspace=0,
        memory_limit_bytes=64 * 1024**2,
    )
