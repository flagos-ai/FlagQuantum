"""Run the original five-bit-by-five-bit walk without physical wire indices."""

from typing import Annotated

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    Output,
    QNum,
    allocate,
    drop,
    inplace_prepare_amplitudes,
    qfunc,
)

from .operations import walk_step
from .physics import COIN_AMPLITUDES, POSITION_AMPLITUDES


@qfunc
def main(
    t: int,
    x_amplitudes: list[float],
    y_amplitudes: list[float],
    coin_amplitudes: list[float],
    x: Output[Annotated[QNum, 5]],
    y: Output[Annotated[QNum, 5]],
) -> None:
    coin: Annotated[QNum, 2] = QNum()
    allocate(x)
    allocate(y)
    allocate(coin)
    inplace_prepare_amplitudes(x_amplitudes, 0, x)
    inplace_prepare_amplitudes(y_amplitudes, 0, y)
    inplace_prepare_amplitudes(coin_amplitudes, 0, coin)
    for step in range(t):
        walk_step(x, y, coin)
    drop(coin)


def build_program(
    t: int = 2,
    x_amplitudes: tuple[float, ...] = POSITION_AMPLITUDES,
    y_amplitudes: tuple[float, ...] = POSITION_AMPLITUDES,
    coin_amplitudes: tuple[float, ...] = COIN_AMPLITUDES,
) -> synth.Program:
    if type(t) is not int or not 0 <= t <= 16:
        raise ValueError("Use an integer step count in 0..16")
    if len(x_amplitudes) != 32 or len(y_amplitudes) != 32 or len(coin_amplitudes) != 4:
        raise ValueError("Use thirty-two amplitudes per position and four for the coin")
    return synth.synthesize(
        main,
        classical_inputs={
            "t": t,
            "x_amplitudes": list(x_amplitudes),
            "y_amplitudes": list(y_amplitudes),
            "coin_amplitudes": list(coin_amplitudes),
        },
    )


if __name__ == "__main__":
    program = build_program()
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("x", "y"))))
    print(program.report["resources"])
