"""Compute a native unsigned arithmetic expression with power-of-two modulo."""

from collections import Counter
from random import Random
from typing import Annotated

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import Output, QNum, qfunc


@qfunc
def main(
    a: Output[Annotated[QNum, 5]], b: Output[Annotated[QNum, 5]], res: Output[QNum]
) -> None:
    a |= 4
    b |= 7
    res |= (a + b) % 4


def build_program() -> synth.Program:
    return synth.synthesize(main)


def run_example() -> None:
    program = build_program()
    result = fq.run(program)
    probabilities = dict(program.probabilities(result, registers=("a", "b", "res")))
    counts = Counter(
        Random(17).choices(
            list(probabilities), weights=list(probabilities.values()), k=1000
        )
    )
    print(dict(counts))


if __name__ == "__main__":
    run_example()
