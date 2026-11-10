"""Estimate the fixed diagonal Ising energy with a typed fractional phase word."""

import math
from typing import Annotated

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    Output,
    QArray,
    QBit,
    QNum,
    adjoint,
    allocate,
    hadamard_transform,
    qft,
    qfunc,
)
from flagquantum.synth import phase as apply_phase


@qfunc
def main(
    state: Output[QArray[QBit]], phase: Output[Annotated[QNum, (2, False, 2)]]
) -> None:
    allocate(2, state)
    allocate(phase)
    hadamard_transform(state)
    hadamard_transform(phase)
    # H = 0.375 I - 0.125 Z(high) - 0.25 Z(low).
    # The two-bit phase value is counter/4. Commuting evolution is exact.
    apply_phase(phase * (2 * state[0] + state[1]), 2 * math.pi)
    with adjoint():
        qft(phase)


def build_program() -> synth.Program:
    """Compose this diagonal QPE instance; no general callback QPE is implied."""
    return synth.synthesize(main, max_workspace=0)


if __name__ == "__main__":
    program = build_program()
    result = fq.run(program)
    print(dict(program.numeric_probabilities(result, registers=("state", "phase"))))
    print(program.report["resources"])
