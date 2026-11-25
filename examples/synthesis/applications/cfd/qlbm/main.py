"""Run the complete twelve-bit lattice program with checked clean work."""

from typing import Annotated

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    H,
    Output,
    QBit,
    QNum,
    X,
    allocate,
    inplace_prepare_amplitudes,
    qfunc,
)

from .operations import lattice_step
from .physics import ACTIVE_VELOCITIES, VELOCITY_AMPLITUDES, X_AMPLITUDES, Y_AMPLITUDES


@qfunc
def main(
    x_amplitudes: list[float],
    y_amplitudes: list[float],
    ux_amplitudes: list[float],
    uy_amplitudes: list[float],
    active: list[int],
    x: Output[Annotated[QNum, 3]],
    y: Output[Annotated[QNum, 3]],
    vx: Output[QBit],
    vy: Output[QBit],
    ux: Output[Annotated[QNum, 2]],
    uy: Output[Annotated[QNum, 2]],
) -> None:
    allocate(x)
    allocate(y)
    allocate(vx)
    allocate(vy)
    allocate(ux)
    allocate(uy)
    inplace_prepare_amplitudes(x_amplitudes, 0, x)
    inplace_prepare_amplitudes(y_amplitudes, 0, y)
    inplace_prepare_amplitudes(ux_amplitudes, 0, ux)
    inplace_prepare_amplitudes(uy_amplitudes, 0, uy)
    X(vx)
    H(vy)
    for step in range(13):
        lattice_step(active[step], x, y, vx, vy, ux, uy)


def build_program(
    x_amplitudes: tuple[float, ...] = X_AMPLITUDES,
    y_amplitudes: tuple[float, ...] = Y_AMPLITUDES,
    ux_amplitudes: tuple[float, ...] = VELOCITY_AMPLITUDES,
    uy_amplitudes: tuple[float, ...] = VELOCITY_AMPLITUDES,
) -> synth.Program:
    if (
        len(x_amplitudes) != 8
        or len(y_amplitudes) != 8
        or len(ux_amplitudes) != 4
        or len(uy_amplitudes) != 4
    ):
        raise ValueError("Use eight position amplitudes and four velocity amplitudes")
    return synth.synthesize(
        main,
        classical_inputs={
            "x_amplitudes": list(x_amplitudes),
            "y_amplitudes": list(y_amplitudes),
            "ux_amplitudes": list(ux_amplitudes),
            "uy_amplitudes": list(uy_amplitudes),
            "active": list(ACTIVE_VELOCITIES),
        },
        max_workspace=8,
        memory_limit_bytes=64 * 1024**2,
    )


if __name__ == "__main__":
    program = build_program()
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("x", "y"))))
    print(program.report["resources"])
