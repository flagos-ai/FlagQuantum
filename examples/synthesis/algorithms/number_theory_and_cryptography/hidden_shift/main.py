"""Unchanged hidden-shift Qmod, native phase composition and public Circuit.

The native form composes existing qfunc functions; this batch does not add
higher-order Python callback annotations or a native phase_oracle export."""

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
def oracle(a: Const[QBit], b: Const[QBit], c: Const[QBit], d: Const[QBit]) -> None:
    phase(a * b + c * d, math.pi)


@qfunc
def shifted(a: Const[QBit], b: Const[QBit], c: Const[QBit], d: Const[QBit]) -> None:
    phase(a * b, math.pi)
    phase(c, math.pi)
    phase(c * d, -math.pi)


@qfunc
def main(a: Output[QBit], b: Output[QBit], c: Output[QBit], d: Output[QBit]) -> None:
    allocate(a)
    allocate(b)
    allocate(c)
    allocate(d)
    hadamard_transform(a)
    hadamard_transform(b)
    hadamard_transform(c)
    hadamard_transform(d)
    shifted(a, b, c, d)
    hadamard_transform(a)
    hadamard_transform(b)
    hadamard_transform(c)
    hadamard_transform(d)
    oracle(a, b, c, d)
    hadamard_transform(a)
    hadamard_transform(b)
    hadamard_transform(c)
    hadamard_transform(d)


def run_example() -> None:
    program = synth.synthesize(main)
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=tuple(program.registers))))
    print(program.report["resources"])


if __name__ == "__main__":
    run_example()
