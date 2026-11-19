"""Run both native bitwise-OR Notebook examples with named local samples."""

from collections import Counter
from random import Random

import flagquantum as fq
from examples.synthesis.functions.arithmetic.bitwise_or.bitwise_or_2vars_example.main import (
    main as register_operation,
)
from examples.synthesis.functions.arithmetic.bitwise_or.bitwise_or_integer_example.main import (
    main as literal_operation,
)
from flagquantum.synth import synthesize


def run_example() -> None:
    for definition, registers in (
        (register_operation, ("a", "b", "res")),
        (literal_operation, ("a", "res")),
    ):
        program = synthesize(definition)
        result = fq.run(program)
        reader = (
            program.numeric_probabilities
            if len(registers) == 3
            else program.probabilities
        )
        probabilities = dict(reader(result, registers=registers))
        samples = Random(17).choices(
            list(probabilities), weights=list(probabilities.values()), k=1000
        )
        print("Probabilities:", probabilities)
        print("Counts:", dict(Counter(samples)))
        print("Resources:", program.report["resources"])


if __name__ == "__main__":
    run_example()
