"""Run bounded Grover search and annotate locally sampled predicate outcomes."""

from collections import Counter
from random import Random

import flagquantum as fq
from examples.synthesis.functions.grover_operator.main import main
from flagquantum import synth


def run_example() -> None:
    program = synth.synthesize(main, objective="width")
    result = fq.run(program)
    probabilities = dict(program.probabilities(result, registers=("x", "y")))
    counts = Counter(
        Random(17).choices(
            list(probabilities), weights=list(probabilities.values()), k=1000
        )
    )
    rows = [
        {
            "x": x,
            "y": y,
            "counts": count,
            "predicate": x + y < 4 and (x * y) % 4 == 2,
        }
        for (x, y), count in sorted(counts.items())
    ]
    print("Probabilities:", probabilities)
    print("Samples:", rows)
    print("Resources:", program.report["resources"])


if __name__ == "__main__":
    run_example()
