"""HHL using native QPE, commuting Pauli evolution and a checked inversion table."""

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
    inplace_prepare_amplitudes,
    qfunc,
    qpe,
    select_rotation,
    suzuki_trotter,
    within,
)

from .data import HAMILTONIAN, INVERSION_ANGLES, RHS


@qfunc
def main(
    amplitudes: list[float],
    solution: Output[Annotated[QNum, 2]],
    phase_var: Output[Annotated[QNum, (4, True, 4)]],
    indicator: Output[QBit],
) -> None:
    allocate(solution)
    allocate(phase_var)
    allocate(indicator)
    inplace_prepare_amplitudes(amplitudes, 0, solution)
    with within():
        qpe(
            lambda: suzuki_trotter(HAMILTONIAN, -2 * math.pi, 1, 1, solution), phase_var
        )
    with apply():
        select_rotation("Y", INVERSION_ANGLES, phase_var, indicator)


def build_program(amplitudes: tuple[float, ...] = RHS) -> synth.Program:
    """Keep the source's signed phase precision and approximate reciprocal network."""
    if len(amplitudes) != 4:
        raise ValueError("HHL requires four real RHS amplitudes")
    return synth.synthesize(
        main,
        classical_inputs={"amplitudes": list(amplitudes)},
        max_qubits=7,
        max_workspace=0,
        memory_limit_bytes=1024**2,
    )


if __name__ == "__main__":
    program = build_program()
    result = fq.run(program)
    print(
        dict(
            program.numeric_probabilities(
                result, registers=("solution", "phase_var", "indicator")
            )
        )
    )
    print(program.report["resources"])
