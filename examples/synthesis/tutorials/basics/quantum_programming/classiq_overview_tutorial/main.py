"""Prepare, compute, draw and sample a native quantum arithmetic program."""

from collections import Counter
from random import Random

import flagquantum as fq
from flagquantum import synth
from flagquantum.drawer import draw
from flagquantum.synth import Output, QNum, allocate, hadamard_transform, qfunc


@qfunc
def main(x: Output[QNum], y: Output[QNum]) -> None:
    allocate(3, x)
    hadamard_transform(x)
    y |= x**2 + 1


def run_example() -> None:
    program = synth.synthesize(main, memory_limit_bytes=512 * 1024**2)
    print(draw(program.circuit))
    result = fq.run(program)
    probabilities = dict(program.probabilities(result, registers=("x", "y")))
    print("Probabilities:", probabilities)
    samples = Random(17).choices(
        list(probabilities), weights=list(probabilities.values()), k=1000
    )
    print("Counts:", dict(Counter(samples)))


if __name__ == "__main__":
    run_example()
