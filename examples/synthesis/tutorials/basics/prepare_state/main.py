"""Prepare a five-qubit distribution and inspect named local sample rows."""

from collections import Counter
from random import Random

import flagquantum as fq
from flagquantum import synth
from flagquantum.drawer import draw
from flagquantum.synth import Output, QArray, QBit, prepare_state, qfunc


@qfunc
def main(io: Output[QArray[QBit]]) -> None:
    prepare_state(
        [
            0,
            0.002,
            0.004,
            0.006,
            0.0081,
            0.0101,
            0.0121,
            0.0141,
            0.0161,
            0.0181,
            0.0202,
            0.0222,
            0.0242,
            0.0262,
            0.0282,
            0.0302,
            0.0323,
            0.0343,
            0.0363,
            0.0383,
            0.0403,
            0.0423,
            0.0444,
            0.0464,
            0.0484,
            0.0504,
            0.0524,
            0.0544,
            0.0565,
            0.0585,
            0.0605,
            0.0625,
        ],
        0.01,
        io,
    )


def run_example() -> None:
    program = synth.synthesize(main)
    print(draw(program.circuit))
    result = fq.run(program)
    probabilities = dict(program.probabilities(result, registers=("io",)))
    samples = Random(17).choices(
        list(probabilities), weights=list(probabilities.values()), k=1000
    )
    counts = Counter(samples)
    print("Probabilities:", probabilities)
    print("Counts:", dict(sorted(counts.items())))
    print("First five sample rows:", list(sorted(counts.items()))[:5])


if __name__ == "__main__":
    run_example()
