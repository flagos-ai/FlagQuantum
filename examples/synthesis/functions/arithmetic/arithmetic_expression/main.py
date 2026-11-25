"""Compute a native unsigned arithmetic expression with power-of-two modulo."""

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import Output, QNum, qfunc


@qfunc
def main(a: Output[QNum], b: Output[QNum], c: Output[QNum], res: Output[QNum]) -> None:
    a |= 2
    b |= 1
    c |= 5
    res |= (((((a + b + c) & 15) % 8) ^ (3 & a)) ^ 10) == 4


def build_program() -> synth.Program:
    return synth.synthesize(main)


if __name__ == "__main__":
    program = build_program()
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("a", "b", "c", "res"))))
