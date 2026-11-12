"""Reversible matrix HHL with explicit fractional phase and rotation semantics."""

from typing import Annotated

import flagquantum as fq
from examples.synthesis.algorithms.quantum_primitives.swap_test.main import overlap_test
from flagquantum import synth
from flagquantum.synth import (
    Output,
    QArray,
    QBit,
    QNum,
    allocate,
    apply,
    drop,
    prepare_amplitudes,
    qfunc,
    qpe,
    select_rotation,
    unitary,
    within,
)

from .physics import ANGLES, COMPARISON_INPUT, RAW_INPUT, UNITARY


@qfunc
def main(
    amplitudes: list[float],
    comparison: list[float],
    indicator: Output[QBit],
    test: Output[QBit],
) -> None:
    phase_var: Annotated[QNum, (4, False, 4)] = QNum()
    state: QArray[QBit] = QArray()
    compared_state: QArray[QBit] = QArray()
    allocate(phase_var)
    prepare_amplitudes(amplitudes, 0, state)
    prepare_amplitudes(comparison, 0, compared_state)
    allocate(indicator)
    allocate(test)
    with within():
        qpe(lambda: unitary(UNITARY, state), phase_var)
    with apply():
        select_rotation("Y", ANGLES, phase_var, indicator)
    overlap_test(state, compared_state, test)
    drop(phase_var)
    drop(state)
    drop(compared_state)


def build_program(
    amplitudes: tuple[float, ...] = RAW_INPUT,
    comparison: tuple[float, ...] = COMPARISON_INPUT,
) -> synth.Program:
    if len(amplitudes) != 4 or len(comparison) != 4:
        raise ValueError("input vectors must match the matrix dimension")
    return synth.synthesize(
        main,
        classical_inputs={
            "amplitudes": list(amplitudes),
            "comparison": list(comparison),
        },
        max_workspace=0,
    )


if __name__ == "__main__":
    program = build_program()
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("indicator",))))
    print(program.report["resources"])
