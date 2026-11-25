"""One predicate-driven Grover round in Qmod, native qfunc and public Circuit."""

import math

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    Const,
    Output,
    QBit,
    allocate,
    hadamard_transform,
    phase,
    qfunc,
)


@qfunc
def mark(a: Const[QBit], b: Const[QBit]) -> None:
    phase(a * b, math.pi)


@qfunc
def reflect_nonzero(a: Const[QBit], b: Const[QBit]) -> None:
    phase(a * b, math.pi)
    phase(a, -math.pi)
    phase(b, -math.pi)


@qfunc
def main(a: Output[QBit], b: Output[QBit]) -> None:
    allocate(a)
    allocate(b)
    hadamard_transform(a)
    hadamard_transform(b)
    mark(a, b)
    hadamard_transform(a)
    hadamard_transform(b)
    reflect_nonzero(a, b)
    hadamard_transform(a)
    hadamard_transform(b)


def run_example() -> None:
    program = synth.synthesize(main)
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=tuple(program.registers))))
    print(program.report["resources"])


if __name__ == "__main__":
    run_example()
