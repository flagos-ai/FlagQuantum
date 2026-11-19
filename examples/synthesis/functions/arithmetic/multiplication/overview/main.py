"""Run both native multiplication Notebook stages with exact named readout."""

from collections import Counter
from random import Random

import flagquantum as fq
from examples.synthesis.functions.arithmetic.multiplication.multiplication_2vars_example.main import (
    main as multiply_words,
)
from examples.synthesis.functions.arithmetic.multiplication.multiplication_float_example.main import (
    main as multiply_scaled,
)
from flagquantum.synth import synthesize


def run_example() -> None:
    for definition, registers, numeric in (
        (multiply_words, ("a", "b", "res"), False),
        (multiply_scaled, ("a", "res"), True),
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
