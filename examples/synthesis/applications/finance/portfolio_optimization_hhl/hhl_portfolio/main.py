"""Reversible matrix HHL with explicit fractional phase and rotation semantics."""

from typing import Annotated

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    Output,
    QBit,
    QNum,
    allocate,
    apply,
    prepare_amplitudes,
    qfunc,
    qpe,
    select_rotation,
    unitary,
    within,
)

from .physics import ANGLES, RAW_INPUT, UNITARY


@qfunc
def main(
    amplitudes: list[float],
    indicator: Output[QBit],
    res: Output[Annotated[QNum, 4]],
    rescaled_eig: Output[Annotated[QNum, (6, False, 6)]],
) -> None:
    prepare_amplitudes(amplitudes, 0, res)
    allocate(rescaled_eig)
    allocate(indicator)
    with within():
        qpe(lambda: unitary(UNITARY, res), rescaled_eig)
    with apply():
        select_rotation("Y", ANGLES, rescaled_eig, indicator)


def build_program(amplitudes: tuple[float, ...] = RAW_INPUT) -> synth.Program:
    if len(amplitudes) != 16:
        raise ValueError("input vectors must match the matrix dimension")
    return synth.synthesize(
        main, classical_inputs={"amplitudes": list(amplitudes)}, max_workspace=3
    )


if __name__ == "__main__":
    program = build_program()
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("indicator",))))
    print(program.report["resources"])
