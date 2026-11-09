"""Compute an unsigned value from native quantum expressions."""

import flagquantum as fq
from flagquantum.synth import (
    Output,
    QNum,
    allocate,
    hadamard_transform,
    qfunc,
    synthesize,
)


@qfunc
def main(a: Output[QNum], res: Output[QNum]) -> None:
    allocate(3, a)
    hadamard_transform(a)
    res |= a <= 2


if __name__ == "__main__":
    program = synthesize(main)
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("a", "res"))))
