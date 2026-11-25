"""Walk on the original circle with exact conditional Fourier shifts."""

import math
from typing import Annotated

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    H,
    Output,
    QBit,
    QNum,
    adjoint,
    allocate,
    phase,
    qft,
    qfunc,
)


@qfunc
def main(t: int, x: Output[Annotated[QNum, (7, True, 0)]], coin: Output[QBit]) -> None:
    allocate(x)
    allocate(coin)
    for step in range(t):
        H(coin)
        qft(x)
        phase((1 - 2 * coin) * x, math.tau / 128)
        with adjoint():
            qft(x)


def build_program(t: int = 2) -> synth.Program:
    """Bind a bounded step count while preserving signed seven-bit outputs."""
    if type(t) is not int or not 0 <= t <= 16:
        raise ValueError("t must be an integer between zero and sixteen")
    return synth.synthesize(main, classical_inputs={"t": t}, max_workspace=0)


if __name__ == "__main__":
    program = build_program()
    result = fq.run(program)
    print(dict(program.numeric_probabilities(result, registers=("x", "coin"))))
    print(program.report["resources"])
