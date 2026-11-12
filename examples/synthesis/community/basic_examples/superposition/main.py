"""Prepare and inspect an equal superposition using a reusable quantum function."""

import flagquantum as fq
from flagquantum.drawer import draw
from flagquantum.synth import (
    Output,
    QArray,
    QBit,
    allocate,
    hadamard_transform,
    qfunc,
    synthesize,
)


@qfunc
def prepare_uniform(register: QArray[QBit]) -> None:
    hadamard_transform(register)


@qfunc
def main(register_a: Output[QArray[QBit]]) -> None:
    allocate(3, register_a)
    prepare_uniform(register_a)


def run_example() -> None:
    program = synthesize(main)
    print(draw(program.circuit))
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("register_a",))))


if __name__ == "__main__":
    run_example()
