"""Compute a bitwise value from two native quantum registers."""

import flagquantum as fq
from flagquantum.synth import Output, QNum, qfunc, synthesize


@qfunc
def main(a: Output[QNum], b: Output[QNum], res: Output[QNum]) -> None:
    a |= 4
    b |= 5
    res |= a & b


if __name__ == "__main__":
    program = synthesize(main)
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("a", "b", "res"))))
