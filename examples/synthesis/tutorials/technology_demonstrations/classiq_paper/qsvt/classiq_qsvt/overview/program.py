"""Six-data-bit QSVT with the source block encoding and four real phases."""

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

PHASES = (1.280311896404252, 8.127145628464149, 1.8439603212845617, -5.002873410775335)


@qfunc
def reflect_about_zero(data: QNum) -> None:
    # Source convention: +1 on zero, -1 on every other data word.
    control(data > 0, lambda: phase(1, math.pi))


@qfunc
def add_two(packed: QNum) -> None:
    packed += 2


@qfunc
def subtract_one(packed: QNum) -> None:
    packed -= 1


@qfunc
def block_encoding(data: QNum, block: QNum) -> None:
    H(block[0])
    control(block[0] == 0, lambda: add_two([data, block[1]]))
    subtract_one([data, block[1]])
    H(block[0])
    H(block[2])
    control(block[2] == 0, lambda: reflect_about_zero(data))
    H(block[2])


@qfunc
def projector_phase(angle: float, block: QNum, aux: QBit) -> None:
    # X-P-X changes RZ's sign exactly when block == 0. This polynomial
    # expresses that diagonal action without another predicate register.
    phase(aux - 0.5, angle)
    phase((1 - 2 * aux) * (1 - block[0]) * (1 - block[1]) * (1 - block[2]), angle)


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
    definition: object = main,
    *,
    phases: tuple[float, ...] = PHASES,
) -> synth.Program:
    """Compile the source degree-three program within eleven physical qubits."""
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
