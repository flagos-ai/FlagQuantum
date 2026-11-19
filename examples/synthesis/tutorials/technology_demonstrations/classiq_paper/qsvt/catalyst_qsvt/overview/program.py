"""Native reconstruction of the commented Catalyst QSVT with ten logical wires."""

import math
from typing import Annotated

from flagquantum import synth
from flagquantum.synth import (
    H,
    Output,
    QBit,
    QNum,
    adjoint,
    allocate,
    control,
    phase,
    qfunc,
)

from ...classiq_qsvt.overview.program import (
    PHASES as PHASES,
)
from ...classiq_qsvt.overview.program import (
    add_two,
    projector_phase,
    subtract_one,
)


@qfunc
def reflect_about_zero(data: QNum) -> None:
    # Literal source X/H/zero-controlled-X/H/X is negative only on zero.
    control(data == 0, lambda: phase(1, math.pi))


@qfunc
def block_encoding(data: QNum, block: QNum) -> None:
    H(block[0])
    H(block[2])
    control(
        block[0] == 1,
        lambda: add_two(
            [block[1], data[5], data[4], data[3], data[2], data[1], data[0]]
        ),
    )
    subtract_one([block[1], data[5], data[4], data[3], data[2], data[1], data[0]])
    control(block[2] == 1, lambda: reflect_about_zero(data))
    H(block[0])
    H(block[2])


@qfunc
def apply_qsvt(phases: list[float], data: QNum, block: QNum, aux: QBit) -> None:
    H(aux)
    projector_phase(phases[0], block, aux)
    block_encoding(data, block)
    projector_phase(phases[1], block, aux)
    with adjoint():
        block_encoding(data, block)
    projector_phase(phases[2], block, aux)
    block_encoding(data, block)
    projector_phase(phases[3], block, aux)
    H(aux)


@qfunc
def main(
    phases: list[float],
    block: Output[Annotated[QNum, 3]],
    data: Output[Annotated[QNum, 6]],
    qsvt_aux: Output[QBit],
) -> None:
    allocate(qsvt_aux)
    allocate(block)
    allocate(data)
    apply_qsvt(phases, data, block, qsvt_aux)


def build_program(
    definition: object = main, *, phases: tuple[float, ...] = PHASES
) -> synth.Program:
    """Compile the source ten logical bits with bounded native resources."""
    if not isinstance(definition, type(main)):
        raise TypeError("Use a decorated qfunc program definition.")
    if len(phases) != 4 or any(
        isinstance(angle, bool) or not math.isfinite(angle) for angle in phases
    ):
        raise ValueError("QSVT requires exactly four finite real phases.")
    return synth.synthesize(
        definition,
        classical_inputs={"phases": list(phases)},
        max_qubits=11,
        max_workspace=1,
        max_operations=100_000,
        memory_limit_bytes=64 * 1024**2,
    )
