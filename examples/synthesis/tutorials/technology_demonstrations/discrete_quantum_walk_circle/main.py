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
def main(x: Output[Annotated[QNum, 11]], coin: Output[QBit]) -> None:
    allocate(x)
    allocate(coin)
    H(coin)
    qft(x)
    phase((1 - 2 * coin) * x, math.tau / 2048)
    with adjoint():
        qft(x)


def build_program() -> synth.Program:
    """Preserve the source width and initial state with zero work qubits."""
    return synth.synthesize(main, max_workspace=0)


if __name__ == "__main__":
    program = build_program()
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("x", "coin"))))
    print(program.report["resources"])
