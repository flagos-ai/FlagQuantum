"""Static amplification in Qmod, existing native scalar qfunc, and Circuit.

Native programs spell out zero/one/two rounds with existing phase/CX primitives.
No native amplification export, mutable helper port or Python loop is added.
Run tools/validate_qmod_iterations.py for the three unchanged counting/QMCI files."""

import math

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    CX,
    Const,
    Output,
    QBit,
    allocate,
    hadamard_transform,
    phase,
    qfunc,
)

PI = math.pi


@qfunc
def reflect(flag: Const[QBit], data: Const[QBit]) -> None:
    # Odd parity on every nonzero word implements 2|00><00|-I.
    phase(flag + data + flag * data, PI)


@qfunc
def zero(flag: Output[QBit], data: Output[QBit]) -> None:
    allocate(flag)
    allocate(data)
    hadamard_transform(data)
    CX(data, flag)


@qfunc
def one(flag: Output[QBit], data: Output[QBit]) -> None:
    allocate(flag)
    allocate(data)
    hadamard_transform(data)
    CX(data, flag)
    phase(flag, PI)
    CX(data, flag)
    hadamard_transform(data)
    reflect(flag, data)
    hadamard_transform(data)
    CX(data, flag)


@qfunc
def two(flag: Output[QBit], data: Output[QBit]) -> None:
    allocate(flag)
    allocate(data)
    hadamard_transform(data)
    CX(data, flag)
    phase(flag, PI)
    CX(data, flag)
    hadamard_transform(data)
    reflect(flag, data)
    hadamard_transform(data)
    CX(data, flag)
    phase(flag, PI)
    CX(data, flag)
    hadamard_transform(data)
    reflect(flag, data)
    hadamard_transform(data)
    CX(data, flag)


def run_example() -> None:
    for function in (zero, one, two):
        program = synth.synthesize(function)
        result = fq.run(program)
        print(dict(program.probabilities(result, registers=tuple(program.registers))))
        print(program.report["resources"])


if __name__ == "__main__":
    run_example()
