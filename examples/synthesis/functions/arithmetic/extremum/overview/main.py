"""Run both native extremum Notebook stages with exact named readout."""

from collections import Counter
from random import Random

import flagquantum as fq
from examples.synthesis.functions.arithmetic.extremum.maximum_float_example.main import (
    main as maximum,
)
from examples.synthesis.functions.arithmetic.extremum.minimum_2vars_example.main import (
    main as minimum,
)
from flagquantum.synth import synthesize


def run_example() -> None:
    for definition, registers, numeric in (
        (minimum, ("a", "b", "res"), False),
        (maximum, ("a", "res"), True),
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
