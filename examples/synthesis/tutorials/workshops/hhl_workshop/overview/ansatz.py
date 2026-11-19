"""Native preparation and finite-precision HHL at the source dimensions."""

from typing import Annotated

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

from .physics import AMPLITUDES, ANGLES, UNITARY


@qfunc
def load_b(amplitudes: list[float], res: Output[Annotated[QNum, 2]]) -> None:
    prepare_amplitudes(amplitudes, 0.0, res)


@qfunc
def simple_eig_inv(phase_: QNum, indicator: QBit) -> None:
    select_rotation("Y", ANGLES, phase_, indicator)


@qfunc
def main(
    amplitudes: list[float],
    res: Output[Annotated[QNum, 2]],
    phase_: Output[Annotated[QNum, (4, False, 4)]],
    indicator: Output[QBit],
) -> None:
    prepare_amplitudes(amplitudes, 0.0, res)
    allocate(phase_)
    allocate(indicator)
    with within():
        qpe(lambda: unitary(UNITARY, res), phase_)
    with apply():
        simple_eig_inv(phase_, indicator)


def build_program() -> synth.Program:
    return synth.synthesize(
        main,
        classical_inputs={"amplitudes": list(AMPLITUDES)},
        max_workspace=0,
        memory_limit_bytes=64 * 1024**2,
    )
