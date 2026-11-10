"""Discrete logarithm interference using two modular exponentiations."""

from typing import Annotated

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    Output,
    QNum,
    adjoint,
    allocate,
    hadamard_transform,
    modular_exp,
    qft,
    qfunc,
)


@qfunc
def main(
    element: int,
    generator: int,
    alpha: Output[Annotated[QNum, (5, False, 5)]],
    beta: Output[Annotated[QNum, (5, False, 5)]],
    func_res: Output[Annotated[QNum, 4]],
) -> None:
    allocate(alpha)
    allocate(beta)
    func_res |= 1
    hadamard_transform(alpha)
    hadamard_transform(beta)
    modular_exp(13, element, func_res, alpha)
    modular_exp(13, generator, func_res, beta)
    with adjoint():
        qft(alpha)
        qft(beta)


def build_program(element: int = 3, generator: int = 7) -> synth.Program:
    return synth.synthesize(
        main,
        classical_inputs={"element": element, "generator": generator},
        max_workspace=2,
    )


if __name__ == "__main__":
    program = build_program()
    result = fq.run(program)
    print(dict(program.numeric_probabilities(result, registers=("alpha", "beta"))))
    print(program.report["resources"])
