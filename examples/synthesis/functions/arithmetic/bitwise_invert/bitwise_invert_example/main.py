"""Compute an unsigned value from native quantum expressions."""

from collections import Counter
from random import Random

import flagquantum as fq
from flagquantum.synth import Output, QNum, qfunc, synthesize


@qfunc
def main(x: Output[QNum], y: Output[QNum]) -> None:
    x |= 6
    y |= ~x


def run_example() -> None:
    program = synthesize(main)
    result = fq.run(program)
    named = dict(program.probabilities(result, registers=("x", "y")))
    counts = Counter(
        Random(17).choices(list(named), weights=list(named.values()), k=1000)
    )
    print(dict(counts))


if __name__ == "__main__":
    run_example()
