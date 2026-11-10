"""Native spatial Type-II transforms and coherent HHL inversion for Poisson."""

import math
from typing import Annotated

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    Output,
    QArray,
    QBit,
    QNum,
    allocate,
    apply,
    inplace_prepare_amplitudes,
    qfunc,
    qpe,
    select_rotation,
    suzuki_trotter,
    unitary,
    within,
)

from .physics import AMPLITUDES, ANGLES, COSINE, HAMILTONIAN, SINE


@qfunc
def main(
    amplitudes: list[float],
    xy: Output[QArray[QBit]],
    phase_var: Output[Annotated[QNum, (6, False, 6)]],
    indicator: Output[QBit],
) -> None:
    allocate(6, xy)
    allocate(phase_var)
    allocate(indicator)
    inplace_prepare_amplitudes(amplitudes, 0, xy)
    with within():
        unitary(SINE, xy[:3])
        unitary(COSINE, xy[3:])
    with apply():
        with within():
            qpe(
                lambda: suzuki_trotter(
                    HAMILTONIAN, -2 * math.pi * 0.101321183642338, 1, 1, xy
                ),
                phase_var,
            )
        with apply():
            select_rotation("Y", ANGLES, phase_var, indicator)


def build_program(amplitudes: tuple[float, ...] = tuple(AMPLITUDES)) -> synth.Program:
    if len(amplitudes) != 64 or any(
        type(a) not in (int, float) or not math.isfinite(a) for a in amplitudes
    ):
        raise ValueError("provide 64 finite real spatial amplitudes")
    if not math.isclose(
        math.fsum(a * a for a in amplitudes), 1, rel_tol=0, abs_tol=1e-12
    ):
        raise ValueError("spatial amplitudes must be normalized")
    return synth.synthesize(
        main,
        classical_inputs={"amplitudes": list(amplitudes)},
        max_qubits=13,
        max_workspace=0,
        memory_limit_bytes=64 * 1024**2,
    )


if __name__ == "__main__":
    program = build_program()
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("xy", "indicator"))))
    print(program.report["resources"])
