"""Periodic Laplacian block encoding with coherent modular Fourier shifts."""

import math

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    Output,
    QArray,
    QBit,
    QNum,
    adjoint,
    allocate,
    hadamard_transform,
    phase,
    qft,
    qfunc,
)


@qfunc
def laplacian_block(position: QNum, selector: QArray[QBit]) -> None:
    hadamard_transform(selector)
    phase(selector[0] + selector[1], math.pi)
    qft(position)
    # -1 when the low selector is zero, +1 when the high selector is one.
    phase((selector[0] + selector[1] - 1) * position, math.tau / 32)
    with adjoint():
        qft(position)
    hadamard_transform(selector)
    phase(selector[0] + selector[1], math.pi)


@qfunc
def main(position: Output[QNum], selector: Output[QArray[QBit]]) -> None:
    allocate(5, position)
    allocate(2, selector)
    laplacian_block(position, selector)


def build_program() -> synth.Program:
    return synth.synthesize(main, max_workspace=0)


if __name__ == "__main__":
    program = build_program()
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("position", "selector"))))
    print(program.report["resources"])
