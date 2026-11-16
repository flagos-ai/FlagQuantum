"""Interfere a four-bit state after a high-level polynomial phase oracle."""

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    Output,
    QArray,
    QBit,
    allocate,
    hadamard_transform,
    phase,
    qfunc,
)


@qfunc
def main(x: Output[QArray[QBit]]) -> None:
    allocate(4, x)
    hadamard_transform(x)
    phase(x[0] * x[1] * x[2] * x[3] + x[3], 3.141592653589793)
    hadamard_transform(x)


def build_program() -> synth.Program:
    return synth.synthesize(main, max_workspace=0)


if __name__ == "__main__":
    program = build_program()
    print(dict(program.probabilities(fq.run(program), registers=("x",))))
