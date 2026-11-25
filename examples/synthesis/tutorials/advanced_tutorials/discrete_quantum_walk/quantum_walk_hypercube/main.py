"""A uniform coin selects a Hamming-distance-one move on a hypercube."""

from typing import Annotated

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    Output,
    QArray,
    QBit,
    QNum,
    allocate,
    inplace_prepare_amplitudes,
    qfunc,
)

from .operations import diffuse, move
from .physics import COIN_AMPLITUDES, POSITION_AMPLITUDES


@qfunc
def main(
    t: int,
    position_amplitudes: list[float],
    coin_amplitudes: list[float],
    x: Output[QArray[QBit]],
    coin: Output[Annotated[QNum, 2]],
) -> None:
    allocate(4, x)
    allocate(coin)
    inplace_prepare_amplitudes(position_amplitudes, 0, x)
    inplace_prepare_amplitudes(coin_amplitudes, 0, coin)
    for step in range(t):
        diffuse(coin)
        for direction in range(4):
            move(direction, coin, x)


def build_program(
    t: int = 2,
    position_amplitudes: tuple[float, ...] = POSITION_AMPLITUDES,
    coin_amplitudes: tuple[float, ...] = COIN_AMPLITUDES,
) -> synth.Program:
    if type(t) is not int or not 0 <= t <= 16:
        raise ValueError("Use an integer step count in 0..16")
    if len(position_amplitudes) != 16 or len(coin_amplitudes) != 4:
        raise ValueError("Use sixteen position and four coin amplitudes")
    return synth.synthesize(
        main,
        classical_inputs={
            "t": t,
            "position_amplitudes": list(position_amplitudes),
            "coin_amplitudes": list(coin_amplitudes),
        },
    )


if __name__ == "__main__":
    program = build_program()
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("x", "coin"))))
    print(program.report["resources"])
