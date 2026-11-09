"""Compute an unsigned value from native quantum expressions."""

import flagquantum as fq
from flagquantum.synth import Output, QNum, qfunc, synthesize


@qfunc
def main(a: Output[QNum], res: Output[QNum]) -> None:
    a |= 5
    res |= 3 & a


if __name__ == "__main__":
    program = synthesize(main)
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("a", "res"))))
