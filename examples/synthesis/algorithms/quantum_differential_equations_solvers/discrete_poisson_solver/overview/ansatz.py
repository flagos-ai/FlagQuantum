"""Compose checked static spatial, powered evolution and reciprocal helpers."""

import math
from typing import Annotated

from flagquantum import synth
from flagquantum.synth import (
    Output,
    QArray,
    QBit,
    QNum,
    allocate,
    apply,
    inplace_prepare_amplitudes,
    qfunc,
    qpe,
    select_rotation,
    suzuki_trotter,
    unitary,
    within,
)

from .physics import AMPLITUDES, ANGLES, COSINE, MAX_EIG, OPERATOR, SINE


@qfunc
def powered_evolution(power: int, target: QArray[QBit]) -> None:
    suzuki_trotter(OPERATOR, -2 * math.pi * power / MAX_EIG, 1, 1, target)


@qfunc
def spatial_transform(target: QArray[QBit]) -> None:
    unitary(SINE, target[:3])
    unitary(COSINE, target[3:])


@qfunc
def poisson(
    amplitudes: list[float],
    xy: Output[QArray[QBit]],
    phase: Output[Annotated[QNum, (6, False, 6)]],
    indicator: Output[QBit],
) -> None:
    allocate(6, xy)
    allocate(phase)
    allocate(indicator)
    inplace_prepare_amplitudes(amplitudes, 0, xy)
    with within():
        spatial_transform(xy)
    with apply():
        with within():
            qpe(lambda power: powered_evolution(power, xy), phase)
        with apply():
            select_rotation("Y", ANGLES, phase, indicator)


def build_program() -> synth.Program:
    return synth.synthesize(
        poisson,
        classical_inputs={"amplitudes": list(AMPLITUDES)},
        max_qubits=13,
        max_workspace=0,
        memory_limit_bytes=64 * 1024**2,
    )
