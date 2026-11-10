"""Matrix phase estimation with private data and checked static composition."""

from typing import Annotated

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    Output,
    QArray,
    QBit,
    QNum,
    allocate,
    drop,
    prepare_amplitudes,
    qfunc,
    qpe,
    unitary,
)

from .physics import INITIAL_VECTOR, UNITARY


@qfunc
def main(
    amplitudes: list[float], phase_result: Output[Annotated[QNum, (8, False, 8)]]
) -> None:
    state: QArray[QBit] = QArray()
    prepare_amplitudes(amplitudes, 0, state)
    allocate(phase_result)
    qpe(lambda: unitary(UNITARY, state), phase_result)
    drop(state)


def build_program(amplitudes: tuple[float, ...] = INITIAL_VECTOR) -> synth.Program:
    if len(amplitudes) != 8:
        raise ValueError("input amplitudes must match the source matrix dimension")
    return synth.synthesize(
        main, classical_inputs={"amplitudes": list(amplitudes)}, max_workspace=0
    )


if __name__ == "__main__":
    program = build_program()
    result = fq.run(program)
    print(dict(program.numeric_probabilities(result, registers=("phase_result",))))
    print(program.report["resources"])
