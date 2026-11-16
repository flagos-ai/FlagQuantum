"""Select identity, QFT and inverse QFT with one coherent controller."""

from typing import Annotated

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    Output,
    QNum,
    adjoint,
    allocate,
    apply,
    control,
    inplace_prepare_amplitudes,
    qft,
    qfunc,
    within,
)

from .operations import prepare
from .physics import PROBABILITIES, ZERO_INPUT


@qfunc
def main(
    probabilities: list[float],
    amplitudes: list[float],
    controller: Output[Annotated[QNum, 2]],
    psi: Output[Annotated[QNum, 2]],
) -> None:
    allocate(psi)
    allocate(controller)
    inplace_prepare_amplitudes(amplitudes, 0, psi)
    with within():
        prepare(probabilities, controller)
    with apply():
        control((controller >= 1) & (controller <= 1), lambda: qft(psi))
        with adjoint():
            control((controller >= 2) & (controller <= 2), lambda: qft(psi))


def build_program(
    probabilities: tuple[float, ...] = PROBABILITIES,
    amplitudes: tuple[float, ...] = ZERO_INPUT,
) -> synth.Program:
    if len(probabilities) != 4 or len(amplitudes) != 4:
        raise ValueError(
            "Use four controller probabilities and four real data amplitudes"
        )
    return synth.synthesize(
        main,
        classical_inputs={
            "probabilities": list(probabilities),
            "amplitudes": list(amplitudes),
        },
    )


if __name__ == "__main__":
    program = build_program()
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("controller", "psi"))))
    print(program.report["resources"])
