"""Reversible matrix HHL with explicit fractional phase and rotation semantics."""

import math
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
    res: Output[Annotated[QNum, 3]],
    phase_var: Output[Annotated[QNum, (2, False, 2)]],
    indicator: Output[QBit],
) -> None:
    prepare_amplitudes(amplitudes, 0, res)
    allocate(phase_var)
    allocate(indicator)
    with within():
        qpe(lambda: unitary(UNITARY, res), phase_var)
    with apply():
        select_rotation("Y", ANGLES, phase_var, indicator)


def build_program(amplitudes: tuple[float, ...] = RAW_INPUT) -> synth.Program:
    # Explicit approved normalization preserves all relative rounded amplitudes.
    norm = math.hypot(*amplitudes)
    if not norm or not math.isfinite(norm):
        raise ValueError("input amplitudes must have a finite nonzero norm")
    amplitudes = tuple(value / norm for value in amplitudes)
    if len(amplitudes) != 8:
        raise ValueError("input vectors must match the matrix dimension")
    return synth.synthesize(
        main, classical_inputs={"amplitudes": list(amplitudes)}, max_workspace=0
    )


if __name__ == "__main__":
    program = build_program()
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("indicator",))))
    print(program.report["resources"])
