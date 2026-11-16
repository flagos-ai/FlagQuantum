"""Native coherent Pauli block encoding and oblivious amplitude amplification."""

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    Output,
    QArray,
    QBit,
    allocate,
    inplace_prepare_amplitudes,
    qfunc,
)

from .operations import encode, reflect, zero_phase
from .physics import AMPLITUDES


@qfunc
def main(
    repetitions: int,
    amplitudes: list[float],
    data: Output[QArray[QBit]],
    block: Output[QArray[QBit]],
) -> None:
    allocate(2, block)
    allocate(2, data)
    inplace_prepare_amplitudes(amplitudes, 0, data)
    encode(block, data)
    for i in range(repetitions):
        zero_phase(block)
        reflect(block, data)


def build_program(
    repetitions: int = 1, amplitudes: tuple[float, ...] = AMPLITUDES
) -> synth.Program:
    if type(repetitions) is not int or not 0 <= repetitions <= 16:
        raise ValueError("Use an integer repetition count in 0..16")
    if len(amplitudes) != 4:
        raise ValueError("Use four real data amplitudes")
    return synth.synthesize(
        main,
        classical_inputs={"repetitions": repetitions, "amplitudes": list(amplitudes)},
        max_workspace=0,
    )


if __name__ == "__main__":
    program = build_program()
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("data", "block"))))
    print(program.report["resources"])
