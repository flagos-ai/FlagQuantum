"""Search five-bit graph assignments meeting the original cut threshold."""

from math import pi

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    Output,
    QArray,
    QBit,
    allocate,
    control,
    hadamard_transform,
    phase,
    qfunc,
)

EDGES = ((0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4))


@qfunc
def mark(q: QArray[QBit]) -> None:
    control(
        (
            (q[0] ^ q[1])
            + (q[0] ^ q[2])
            + (q[1] ^ q[2])
            + (q[1] ^ q[3])
            + (q[2] ^ q[4])
            + (q[3] ^ q[4])
            >= 4
        ),
        lambda: phase(1, pi),
    )


@qfunc
def diffuse(q: QArray[QBit]) -> None:
    hadamard_transform(q)
    phase((1 - q[0]) * (1 - q[1]) * (1 - q[2]) * (1 - q[3]) * (1 - q[4]), pi)
    phase(1, pi)
    hadamard_transform(q)


@qfunc
def main(r: int, q: Output[QArray[QBit]]) -> None:
    allocate(5, q)
    hadamard_transform(q)
    for _ in range(r):
        mark(q)
        diffuse(q)


def build_program(r: int = 2) -> synth.Program:
    if type(r) is not int or not 0 <= r <= 32:
        raise ValueError("Use an integer Grover iteration count in 0..32")
    return synth.synthesize(main, classical_inputs={"r": r})


def run_example() -> None:
    program = build_program(2)
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("q",))))
    print(program.report["resources"])


if __name__ == "__main__":
    run_example()
