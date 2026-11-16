"""Compute exact signed or binary-scaled arithmetic with native quantum semantics."""

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import Output, QNum, qfunc


@qfunc
def main(a: Output[QNum], b: Output[QNum], res: Output[QNum]) -> None:
    a |= 4
    b |= 5
    res |= a - b


def build_program() -> synth.Program:
    return synth.synthesize(main)


def run_example() -> None:
    program = build_program()
    result = fq.run(program)
    print(dict(program.numeric_probabilities(result, registers=("a", "b", "res"))))


if __name__ == "__main__":
    run_example()
