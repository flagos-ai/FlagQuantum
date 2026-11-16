"""Run native two-register and literal-mask bitwise-AND programs."""

from collections import Counter
from random import Random

import flagquantum as fq
from examples.synthesis.functions.arithmetic.bitwise_and.bitwise_and_2vars_example.main import (
    main as register_and,
)
from examples.synthesis.functions.arithmetic.bitwise_and.bitwise_and_integer_example.main import (
    main as literal_and,
)
from flagquantum.synth import synthesize


def run_example() -> None:
    for definition, registers in (
        (literal_and, ("a", "res")),
        (register_and, ("a", "b", "res")),
    ):
        program = synthesize(definition)
        result = fq.run(program)
        named = dict(program.probabilities(result, registers=registers))
        samples = Random(17).choices(list(named), weights=list(named.values()), k=1000)
        print(dict(Counter(samples)))


if __name__ == "__main__":
    run_example()
