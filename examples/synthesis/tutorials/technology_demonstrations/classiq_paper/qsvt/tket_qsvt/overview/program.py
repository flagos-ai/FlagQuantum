"""Native reconstruction of the commented Tket QSVT, including literal box wiring."""

import math
from typing import Annotated

from flagquantum import synth
from flagquantum.synth import (
    RZ,
    H,
    Output,
    QBit,
    QNum,
    X,
    adjoint,
    allocate,
    control,
    phase,
    qfunc,
)

from ...classiq_qsvt.overview.program import PHASES as PHASES
from ...classiq_qsvt.overview.program import projector_phase


@qfunc
def source_qft(qft_angles: list[float], word: QNum) -> None:
    for i in range(7):
        H(word[i])
        for j in range(i + 1, 7):
            phase(word[i] * word[j], qft_angles[j - i])


@qfunc
def source_adder(angles: list[float], qft_angles: list[float], word: QNum) -> None:
    source_qft(qft_angles, word)
    for j in range(7):
        RZ(angles[j], word[j])
    with adjoint():
        source_qft(qft_angles, word)


@qfunc
def source_reflection(data: QNum) -> None:
    # X/H surround data[0]; the source zero-controlled X targets data[5].
    X(data[0])
    H(data[0])
    H(data[5])
    phase(
        (1 - data[0])
        * (1 - data[1])
        * (1 - data[2])
        * (1 - data[3])
        * (1 - data[4])
        * data[5],
        math.pi,
    )
    H(data[5])
    H(data[0])
    X(data[0])


@qfunc
def block_encoding(
    qft_angles: list[float],
    add_two: list[float],
    subtract_one: list[float],
    data: QNum,
    block: QNum,
) -> None:
    H(block[0])
    H(block[2])
    control(
        block[0] == 0,
        lambda: source_adder(add_two, qft_angles, [data, block[1]]),
    )
    source_adder(subtract_one, qft_angles, [data, block[1]])
    control(block[2] == 0, lambda: source_reflection(data))
    H(block[0])
    H(block[2])


@qfunc
def apply_qsvt(
    phases: list[float],
    qft_angles: list[float],
    add_two: list[float],
    subtract_one: list[float],
    data: QNum,
    block: QNum,
    aux: QBit,
) -> None:
    H(aux)
    projector_phase(math.pi * phases[0], block, aux)
    # CircBox orders internal registers [block,data]; the source passes [data,block].
    block_encoding(qft_angles, add_two, subtract_one, [data[3:6], block], data[0:3])
    projector_phase(math.pi * phases[1], block, aux)
    with adjoint():
        block_encoding(qft_angles, add_two, subtract_one, [data[3:6], block], data[0:3])
    projector_phase(math.pi * phases[2], block, aux)
    block_encoding(qft_angles, add_two, subtract_one, [data[3:6], block], data[0:3])
    projector_phase(math.pi * phases[3], block, aux)
    H(aux)


@qfunc
def main(
    phases: list[float],
    qft_angles: list[float],
    add_two: list[float],
    subtract_one: list[float],
    block: Output[Annotated[QNum, 3]],
    data: Output[Annotated[QNum, 6]],
    qsvt_aux: Output[QBit],
) -> None:
    allocate(qsvt_aux)
    allocate(block)
    allocate(data)
    apply_qsvt(phases, qft_angles, add_two, subtract_one, data, block, qsvt_aux)


def classical_inputs(phases: tuple[float, ...] = PHASES) -> dict[str, list[float]]:
    """Convert literal source half-turn angles into finite radian bindings."""
    if len(phases) != 4 or any(
        isinstance(angle, bool) or not math.isfinite(angle) for angle in phases
    ):
        raise ValueError("QSVT requires exactly four finite real phases.")
    return {
        "phases": list(phases),
        "qft_angles": [math.pi / 2**j for j in range(7)],
        # Source Rz consumes half turns even though its expression includes pi.
        "add_two": [(2 % 2 ** (j + 1)) * math.pi**2 / 2**j for j in range(7)],
        "subtract_one": [(127 % 2 ** (j + 1)) * math.pi**2 / 2**j for j in range(7)],
    }


def build_program(
    definition: object = main, *, phases: tuple[float, ...] = PHASES
) -> synth.Program:
    """Compile the ten logical bits with no hidden work and bounded native cost."""
    if not isinstance(definition, type(main)):
        raise TypeError("Use a decorated qfunc program definition.")
    return synth.synthesize(
        definition,
        classical_inputs=classical_inputs(phases),
        max_qubits=10,
        max_workspace=0,
        max_operations=20_000,
        memory_limit_bytes=64 * 1024**2,
    )
