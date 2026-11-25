"""Native packed QSVT with root arithmetic markers and complete retained phases."""

import math
from typing import Annotated

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    H,
    Output,
    QBit,
    QNum,
    R,
    adjoint,
    allocate,
    apply,
    control,
    inplace_prepare_amplitudes,
    inplace_prepare_state,
    phase,
    qfunc,
    within,
)


@qfunc
def increment1(packed: QNum) -> None:
    packed += 1


@qfunc
def increment2(packed: QNum) -> None:
    packed += 2


@qfunc
def be(packed: QNum, select: QNum, rotation: QBit) -> None:
    inplace_prepare_state(
        [0.333333333333333, 0.333333333333333, 0.333333333333333, 0.0], 0, select
    )
    control(select == 0, lambda: R(3.252761000151628, math.pi / 2, rotation))
    control(select == 2, lambda: R(3.252761000151628, math.pi / 2, rotation))
    control(select == 1, lambda: increment1(packed))
    control(select == 2, lambda: increment2(packed))
    packed -= 1
    with adjoint():
        inplace_prepare_state(
            [0.333333333333333, 0.333333333333333, 0.333333333333333, 0.0], 0, select
        )


@qfunc
def main(
    phases: list[float],
    amplitudes: list[float],
    data_and_deletion: Output[Annotated[QNum, 4]],
    select: Output[Annotated[QNum, 2]],
    rotation: Output[QBit],
    aux: Output[QBit],
) -> None:
    allocate(data_and_deletion)
    allocate(select)
    allocate(rotation)
    allocate(aux)
    inplace_prepare_amplitudes(amplitudes, 0, data_and_deletion)
    H(aux)
    for i in range(3):
        if i == 1:
            be(data_and_deletion, select, rotation)
        elif i == 2:
            with adjoint():
                be(data_and_deletion, select, rotation)
        with within():
            mark: Annotated[QNum, 1] = (
                (select == 0) & (rotation == 0) & (data_and_deletion < 8)
            )
        with apply():
            phase(aux - 0.5, phases[i])
            phase((1 - 2 * aux) * mark, phases[i])
    H(aux)


PHASES = (-1.738078501179465, 0.654311402219197, 0.654311402219197, -0.167282174384569)
AMPLITUDES = (
    0.0,
    0.23192061392433,
    0.417906505941275,
    0.521120889169602,
    0.521120889169602,
    0.417906505941275,
    0.23192061392433,
    0.0,
)


def build_program(
    phases: tuple[float, ...] = PHASES, amplitudes: tuple[float, ...] = AMPLITUDES
) -> synth.Program:
    if len(phases) != 4 or len(amplitudes) != 8:
        raise ValueError(
            "the retained program requires four phases and eight amplitudes"
        )
    return synth.synthesize(
        main,
        classical_inputs={
            "phases": list(phases),
            "amplitudes": list(amplitudes) + (8 * [0.0]),
        },
        max_qubits=11,
        max_workspace=3,
        memory_limit_bytes=64 * 1024**2,
    )


if __name__ == "__main__":
    program = build_program()
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=tuple(program.registers))))
    print(program.report["resources"])
