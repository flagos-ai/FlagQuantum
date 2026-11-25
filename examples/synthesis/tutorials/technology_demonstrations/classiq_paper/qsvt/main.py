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
    adjoint,
    allocate,
    apply,
    control,
    phase,
    qfunc,
    within,
)


@qfunc
def step2(packed: QNum) -> None:
    packed += 2


@qfunc
def be(packed: QNum, select: QBit, projection: QBit) -> None:
    H(select)
    control(select == 0, lambda: step2(packed))
    packed -= 1
    H(select)
    H(projection)
    control((projection == 0) & ((packed & 63) > 0), lambda: phase(1, math.pi))
    H(projection)


@qfunc
def main(
    phases: list[float],
    data_and_deletion: Output[Annotated[QNum, 7]],
    select: Output[QBit],
    projection: Output[QBit],
    aux: Output[QBit],
) -> None:
    allocate(data_and_deletion)
    allocate(select)
    allocate(projection)
    allocate(aux)
    H(aux)
    for i in range(4):
        if i == 1:
            be(data_and_deletion, select, projection)
        elif i == 2:
            with adjoint():
                be(data_and_deletion, select, projection)
        elif i == 3:
            be(data_and_deletion, select, projection)
        with within():
            deletion: Annotated[QNum, 1] = data_and_deletion >= 64
        with apply():
            phase(aux - 0.5, phases[i])
            phase(
                (1 - 2 * aux) * (1 - select) * (1 - projection) * (1 - deletion),
                phases[i],
            )
    H(aux)


PHASES = (1.280311896404252, 8.127145628464149, 1.843960321284562, -5.002873410775335)


def build_program(phases: tuple[float, ...] = PHASES) -> synth.Program:
    if len(phases) != 4:
        raise ValueError("the retained program requires four phases")
    return synth.synthesize(
        main,
        classical_inputs={"phases": list(phases)},
        max_qubits=15,
        max_workspace=5,
        memory_limit_bytes=64 * 1024**2,
    )


if __name__ == "__main__":
    program = build_program()
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=tuple(program.registers))))
    print(program.report["resources"])
