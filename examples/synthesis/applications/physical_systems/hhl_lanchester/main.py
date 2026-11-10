"""Lanchester HHL keeps the group and time as separate named quantum outputs."""

from typing import Annotated

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    Output,
    QBit,
    QNum,
    allocate,
    apply,
    inplace_prepare_amplitudes,
    qfunc,
    qpe,
    select_rotation,
    unitary,
    within,
)

from .physics import ANGLES, INITIAL_VECTOR, UNITARY


@qfunc
def main(
    amplitudes: list[float],
    indicator: Output[QBit],
    time_index: Output[Annotated[QNum, 4]],
    rabbits: Output[QBit],
    rescaled_eig: Output[Annotated[QNum, (6, False, 6)]],
) -> None:
    allocate(time_index)
    allocate(rabbits)
    allocate(indicator)
    allocate(rescaled_eig)
    inplace_prepare_amplitudes(amplitudes, 0, (rabbits, time_index))
    with within():
        qpe(lambda: unitary(UNITARY, (rabbits, time_index)), rescaled_eig)
    with apply():
        select_rotation("Y", ANGLES, rescaled_eig, indicator)


def build_program(amplitudes: tuple[float, ...] = INITIAL_VECTOR) -> synth.Program:
    if len(amplitudes) != 32:
        raise ValueError("Lanchester state requires 32 amplitudes")
    return synth.synthesize(
        main, classical_inputs={"amplitudes": list(amplitudes)}, max_workspace=4
    )


if __name__ == "__main__":
    program = build_program()
    result = fq.run(program)
    print(
        dict(
            program.numeric_probabilities(
                result, registers=("indicator", "rabbits", "time_index")
            )
        )
    )
    print(program.report["resources"])
