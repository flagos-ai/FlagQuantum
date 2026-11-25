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
    res: Output[Annotated[QNum, 2]],
    phase: Output[Annotated[QNum, (4, False, 4)]],
    indicator: Output[QBit],
) -> None:
    prepare_amplitudes(amplitudes, 0, res)
    allocate(phase)
    allocate(indicator)
    with within():
        qpe(lambda: unitary(UNITARY, res), phase)
    with apply():
        select_rotation("Y", ANGLES, phase, indicator)


def build_program(amplitudes: tuple[float, ...] = RAW_INPUT) -> synth.Program:
    if len(amplitudes) != 4:
        raise ValueError("input vectors must match the matrix dimension")
    return synth.synthesize(
        main, classical_inputs={"amplitudes": list(amplitudes)}, max_workspace=0
    )


if __name__ == "__main__":
    program = build_program()
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("indicator",))))
    print(program.report["resources"])
