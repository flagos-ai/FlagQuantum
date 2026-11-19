"""Construct and execute both source Fourier SELECT workflows locally."""

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
    inplace_prepare_state,
    qft,
    qfunc,
    within,
)


@qfunc
def prepare(controller: QNum) -> None:
    # Exact loading satisfies the source's 0.01 error allowance.
    inplace_prepare_state([0.5, 0.25, 0.25, 0.0], 0, controller)


@qfunc
def select(controller: QNum, psi: QNum) -> None:
    control((controller >= 1) & (controller <= 1), lambda: qft(psi))
    with adjoint():
        control((controller >= 2) & (controller <= 2), lambda: qft(psi))


@qfunc
def main(
    controller: Output[Annotated[QNum, 2]], psi: Output[Annotated[QNum, 2]]
) -> None:
    allocate(psi)
    allocate(controller)
    with within():
        prepare(controller)
    with apply():
        select(controller, psi)


def run_example() -> None:
    # The original repeats the same program in its second construction cell.
    programs = [synth.synthesize(main), synth.synthesize(main)]
    for stage, program in enumerate(programs, start=1):
        result = fq.run(program)
        print("Stage:", stage)
        print(
            "Probabilities:",
            dict(program.probabilities(result, registers=("controller", "psi"))),
        )
        print("Resources:", program.report["resources"])


if __name__ == "__main__":
    run_example()
