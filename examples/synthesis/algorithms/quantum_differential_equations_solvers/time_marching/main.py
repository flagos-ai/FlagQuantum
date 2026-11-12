"""Native four-slice Magnus/QSVT time marching with a coherent counter."""

import math
from typing import Annotated

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    RX,
    H,
    Output,
    QArray,
    QBit,
    QNum,
    R,
    adjoint,
    allocate,
    control,
    hadamard_transform,
    phase,
    qfunc,
)

from .data import EVEN_PHASES, ODD_PHASES


@qfunc
def short(
    a: float, b: float, index: QArray[QBit], time: QArray[QBit], block: QBit
) -> None:
    hadamard_transform(time)
    RX(math.pi / 2, block)
    phase(
        (2 * block - 1)
        * (a + (b - a) * (time[0] + 2 * time[1]) / 4 + index[0] + 2 * index[1]),
        1,
    )
    RX(-math.pi / 2, block)
    hadamard_transform(time)


@qfunc
def projected(
    even: float, odd: float, time: QArray[QBit], block: QBit, aux: QBit, selector: QBit
) -> None:
    phase(
        (aux - 0.5 + (1 - 2 * aux) * (1 - block) * (1 - time[0]) * (1 - time[1]))
        * (even + (odd - even) * selector),
        1,
    )


@qfunc
def final_projected(
    angle: float, time: QArray[QBit], block: QBit, aux: QBit, selector: QBit
) -> None:
    phase(
        (aux - 0.5 + (1 - 2 * aux) * (1 - block) * (1 - time[0]) * (1 - time[1]))
        * selector,
        angle,
    )


@qfunc
def step(
    even: list[float],
    odd: list[float],
    a: float,
    b: float,
    index: QArray[QBit],
    time: QArray[QBit],
    block: QBit,
    aux: QBit,
    selector: QBit,
) -> None:
    R(math.pi / 2, math.pi / 2, selector)
    H(aux)
    projected(even[0], odd[0], time, block, aux, selector)
    for i in range(3):
        short(a, b, index, time, block)
        projected(even[2 * i + 1], odd[2 * i + 1], time, block, aux, selector)
        with adjoint():
            short(a, b, index, time, block)
        projected(even[2 * i + 2], odd[2 * i + 2], time, block, aux, selector)
    control(selector, lambda: short(a, b, index, time, block))
    final_projected(odd[7], time, block, aux, selector)
    H(aux)
    R(-math.pi / 2, math.pi / 2, selector)


@qfunc
def decrement(counter: QNum) -> None:
    counter -= 1


@qfunc
def toggle_success(
    time: QArray[QBit], block: QBit, aux: QBit, selector: QBit, marker: QBit
) -> None:
    H(marker)
    phase(
        (1 - block)
        * (1 - aux)
        * (1 - selector)
        * (1 - time[0])
        * (1 - time[1])
        * marker,
        math.pi,
    )
    H(marker)


@qfunc
def main(
    even: list[float],
    odd: list[float],
    duration: float,
    index: Output[QArray[QBit]],
    time: Output[QArray[QBit]],
    block: Output[QBit],
    aux: Output[QBit],
    selector: Output[QBit],
    unused: Output[QBit],
    counter: Output[Annotated[QNum, 3]],
    marker: Output[QBit],
) -> None:
    allocate(2, index)
    allocate(2, time)
    allocate(block)
    allocate(aux)
    allocate(selector)
    allocate(unused)
    allocate(counter)
    allocate(marker)
    hadamard_transform(index)
    counter ^= 4
    for i in range(4):
        step(
            even,
            odd,
            i * duration / 4,
            (i + 1) * duration / 4,
            index,
            time,
            block,
            aux,
            selector,
        )
        toggle_success(time, block, aux, selector, marker)
        control(marker, lambda: decrement(counter))
        toggle_success(time, block, aux, selector, marker)


def classical_inputs(
    duration: float = 2.0,
    even: tuple[float, ...] = EVEN_PHASES,
    odd: tuple[float, ...] = ODD_PHASES,
) -> dict[str, list[float] | float]:
    if (
        type(duration) not in (int, float)
        or not math.isfinite(duration)
        or not 0 < duration <= 4
    ):
        raise ValueError("duration must be a finite real in (0, 4]")
    if len(even) != 7 or len(odd) != 8:
        raise ValueError("retain seven even and eight odd phases")
    if any(
        type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 64
        for v in (*even, *odd)
    ):
        raise ValueError("phases must be finite reals in [-64, 64]")
    return {"duration": duration, "even": list(even), "odd": list(odd)}


def build_program(
    duration: float = 2.0,
    even: tuple[float, ...] = EVEN_PHASES,
    odd: tuple[float, ...] = ODD_PHASES,
) -> synth.Program:
    return synth.synthesize(
        main,
        classical_inputs=classical_inputs(duration, even, odd),
        max_qubits=20,
        max_workspace=8,
        memory_limit_bytes=64 * 1024**2,
    )


if __name__ == "__main__":
    program = build_program()
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("index", "counter"))))
    print(program.report["resources"])
