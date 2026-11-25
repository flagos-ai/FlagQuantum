"""Native source-sized exponential preparation and hinted amplification."""

import math
from typing import Annotated

from examples.synthesis.applications.finance.autocallable_options.partial_exponential_state_preparation.main import (
    prepare,
    reflect_zero,
)
from flagquantum.synth import (
    Output,
    QBit,
    QNum,
    R,
    adjoint,
    allocate,
    control,
    drop,
    phase,
    qfunc,
)


@qfunc
def prepare_full(amplitudes: list[float], x: Output[Annotated[QNum, 5]]) -> None:
    allocate(x)
    prepare(amplitudes, x)


@qfunc
def amplify(
    amplitudes: list[float],
    beta: float,
    iterations: int,
    lower: int,
    upper: int,
    shift: int,
    x: Output[Annotated[QNum, 5]],
) -> None:
    attenuation: QBit = QBit()
    allocate(x)
    allocate(attenuation)
    prepare(amplitudes, x)
    R(beta, math.pi / 2, attenuation)
    for iteration in range(iterations):
        control((x >= lower) & (x <= upper), lambda: phase(1 - attenuation, math.pi))
        R(-beta, math.pi / 2, attenuation)
        with adjoint():
            prepare(amplitudes, x)
        reflect_zero(x, attenuation)
        prepare(amplitudes, x)
        R(beta, math.pi / 2, attenuation)
    x += shift
    drop(attenuation)
