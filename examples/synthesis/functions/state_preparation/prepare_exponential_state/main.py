"""Specify a fresh exponential probability distribution without designing gates."""

import math

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import Output, QArray, QBit, prepare_state, qfunc

WEIGHTS = tuple(math.exp(-0.1 * value) for value in range(32))
PROBABILITIES = tuple(value / math.fsum(WEIGHTS) for value in WEIGHTS)


@qfunc
def main(probabilities: list[float], x: Output[QArray[QBit]]) -> None:
    prepare_state(probabilities, 0, x)


def build_program() -> synth.Program:
    return synth.synthesize(
        main, classical_inputs={"probabilities": PROBABILITIES}, max_workspace=0
    )


def run_example() -> None:
    program = build_program()
    print(dict(program.probabilities(fq.run(program), registers=("x",))))
    print(program.report["resources"])


if __name__ == "__main__":
    run_example()
