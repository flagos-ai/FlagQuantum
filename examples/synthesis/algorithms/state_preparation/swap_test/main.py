"""One-bit swap test in native qfunc, Qmod and public Circuit syntax.

The pinned three-bit original has non-normalized vectors and remains refused.
Run tools/validate_qmod_swap_test.py for its labelled diagnostic comparison.
This example uses valid one-bit states and existing native phase/CX primitives;
it adds no native array preparation or swap_test export."""

import math

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    CX,
    Output,
    QBit,
    R,
    allocate,
    hadamard_transform,
    phase,
    qfunc,
)

THETA = 0.74


PHI = -0.39


HALF_PI = math.pi / 2


PI = math.pi


@qfunc
def main(test: Output[QBit], first: Output[QBit], second: Output[QBit]) -> None:
    allocate(test)
    allocate(first)
    allocate(second)
    R(THETA, HALF_PI, first)
    R(PHI, HALF_PI, second)
    hadamard_transform(test)
    CX(second, first)
    hadamard_transform(second)
    phase(test * first * second, PI)
    hadamard_transform(second)
    CX(second, first)
    hadamard_transform(test)


def run_example() -> None:
    program = synth.synthesize(main)
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=tuple(program.registers))))
    print(program.report["resources"])


if __name__ == "__main__":
    run_example()
