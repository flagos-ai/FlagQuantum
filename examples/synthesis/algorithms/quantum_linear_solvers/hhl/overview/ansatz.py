"""Four-bit HHL with exact matrix powers and the original powered Trotter rule."""

import math
from typing import Annotated, Literal

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
    suzuki_trotter,
    unitary,
    within,
)

from .physics import AMPLITUDES, ANGLES, OPERATOR, REPETITIONS, UNITARY

Route = Literal["exact", "trotter"]


@qfunc
def exact(
    amplitudes: list[float],
    res: Output[Annotated[QNum, 2]],
    estimator_var: Output[Annotated[QNum, (4, False, 4)]],
    indicator: Output[QBit],
) -> None:
    prepare_amplitudes(amplitudes, 0, res)
    allocate(estimator_var)
    allocate(indicator)
    with within():
        qpe(lambda: unitary(UNITARY, res), estimator_var)
    with apply():
        select_rotation("Y", ANGLES, estimator_var, indicator)


@qfunc
def trotter(
    amplitudes: list[float],
    repetitions: list[int],
    res: Output[Annotated[QNum, 2]],
    estimator_var: Output[Annotated[QNum, (4, False, 4)]],
    indicator: Output[QBit],
) -> None:
    prepare_amplitudes(amplitudes, 0, res)
    allocate(estimator_var)
    allocate(indicator)
    with within():
        qpe(
            lambda power: suzuki_trotter(
                OPERATOR, -2 * math.pi * power, 1, repetitions[power], res
            ),
            estimator_var,
        )
    with apply():
        select_rotation("Y", ANGLES, estimator_var, indicator)


def build_program(route: Route) -> synth.Program:
    if route == "exact":
        return synth.synthesize(
            exact,
            classical_inputs={"amplitudes": list(AMPLITUDES)},
            max_workspace=0,
            memory_limit_bytes=64 * 1024**2,
        )
    if route != "trotter":
        raise ValueError("use exact or trotter evolution")
    table = [1] * 9
    for bit, repetitions in enumerate(REPETITIONS):
        table[1 << bit] = repetitions
    return synth.synthesize(
        trotter,
        classical_inputs={"amplitudes": list(AMPLITUDES), "repetitions": table},
        max_workspace=0,
        memory_limit_bytes=64 * 1024**2,
    )
