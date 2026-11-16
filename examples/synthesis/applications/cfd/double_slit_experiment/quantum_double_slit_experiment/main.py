"""Complete native QSVT double-slit wave evolution."""

import math
from typing import Annotated

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    Output,
    QBit,
    QNum,
    allocate,
    inplace_prepare_amplitudes,
    qfunc,
)

from .operations import qsvt
from .physics import ABSORBING_PHASE, PHASES


@qfunc
def main(
    phases: list[float],
    initial_x: list[float],
    cycles: int,
    absorbing_phase: float,
    x: Output[Annotated[QNum, 4]],
    y: Output[Annotated[QNum, 4]],
    select: Output[Annotated[QNum, 2]],
    edge: Output[QBit],
    scatter: Output[QBit],
    axis: Output[QBit],
    direction: Output[QBit],
    aux: Output[QBit],
) -> None:
    allocate(x)
    allocate(y)
    allocate(select)
    allocate(edge)
    allocate(scatter)
    allocate(axis)
    allocate(direction)
    allocate(aux)
    inplace_prepare_amplitudes(initial_x, 0, x)
    y ^= 2
    qsvt(
        phases,
        cycles,
        absorbing_phase,
        x,
        y,
        select,
        axis,
        direction,
        scatter,
        edge,
        aux,
    )


def build_program(*, cycles: int = 95) -> synth.Program:
    """Build the complete frozen wave model; ``cycles=1`` is a finite probe."""
    if type(cycles) is not int or not 1 <= cycles <= 95:
        raise ValueError("Cycle count must be an integer in 1..95")
    phases = list(PHASES)
    if cycles != 95:
        phases = phases[: 2 * cycles + 2]
        phases[-1] = -0.271
    return synth.synthesize(
        main,
        classical_inputs={
            "phases": phases,
            "initial_x": [1 / math.sqrt(2) if i in (7, 8) else 0.0 for i in range(16)],
            "cycles": cycles,
            "absorbing_phase": ABSORBING_PHASE,
        },
        max_qubits=24,
        max_workspace=9,
        max_operations=100_000,
        memory_limit_bytes=256 * 1024**2,
    )


if __name__ == "__main__":
    program = build_program()
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("x", "y"))))
    print(program.report["resources"])
