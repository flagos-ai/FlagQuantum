"""Compare the original two width budgets under a depth objective."""

from collections import Counter
from random import Random

import flagquantum as fq
from examples.synthesis.tutorials.technology_demonstrations.arithmetic_expressions.main import (
    main,
)
from flagquantum import synth


def run_example() -> None:
    for width in (10, 12):
        program = synth.synthesize(main, objective="depth", max_qubits=width)
        result = fq.run(program)
        probabilities = dict(program.probabilities(result, registers=("x", "y", "z")))
        counts = Counter(
            Random(17).choices(
                list(probabilities), weights=list(probabilities.values()), k=1000
            )
        )
        print("Width budget:", width)
        print("Probabilities:", probabilities)
        print("Counts:", dict(counts))
        print("Resources:", program.report["resources"])


if __name__ == "__main__":
    run_example()
