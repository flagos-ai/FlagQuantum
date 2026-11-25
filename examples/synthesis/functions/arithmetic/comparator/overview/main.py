"""Run the native comparator Notebook with exact named readout and local samples."""

from collections import Counter
from random import Random

import flagquantum as fq
from examples.synthesis.functions.arithmetic.comparator.comparator_2vars_example.main import (
    main as equal_words,
)
from examples.synthesis.functions.arithmetic.comparator.comparator_integer_example.main import (
    main as threshold,
)
from flagquantum.synth import synthesize


def run_example() -> None:
    for definition, registers, numeric in (
        (equal_words, ("a", "b", "res"), True),
        (threshold, ("a", "res"), False),
    ):
        program = synthesize(definition)
        result = fq.run(program)
        reader = program.numeric_probabilities if numeric else program.probabilities
        probabilities = dict(reader(result, registers=registers))
        samples = Random(17).choices(
            list(probabilities), weights=list(probabilities.values()), k=1000
        )
        print("Probabilities:", probabilities)
        print("Counts:", dict(Counter(samples)))
        print("Resources:", program.report["resources"])


if __name__ == "__main__":
    run_example()
