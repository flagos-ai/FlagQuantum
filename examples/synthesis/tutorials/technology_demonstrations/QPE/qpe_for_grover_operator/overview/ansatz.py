"""Original zero-input Grover phase estimation at each source precision."""

from typing import Annotated, Literal

from examples.synthesis.tutorials.technology_demonstrations.QPE.qpe_for_grover_operator.main import (
    grover,
)
from flagquantum import synth
from flagquantum.synth import Output, QBit, QNum, allocate, drop, qfunc, qpe


@qfunc
def precision_1(phase_result: Output[Annotated[QNum, (1, False, 1)]]) -> None:
    indicator: QBit = QBit()
    x: Annotated[QNum, 4] = QNum()
    allocate(indicator)
    allocate(x)
    allocate(phase_result)
    qpe(lambda: grover(x, indicator), phase_result)
    drop(x)
    drop(indicator)


@qfunc
def precision_2(phase_result: Output[Annotated[QNum, (2, False, 2)]]) -> None:
    indicator: QBit = QBit()
    x: Annotated[QNum, 4] = QNum()
    allocate(indicator)
    allocate(x)
    allocate(phase_result)
    qpe(lambda: grover(x, indicator), phase_result)
    drop(x)
    drop(indicator)


@qfunc
def precision_3(phase_result: Output[Annotated[QNum, (3, False, 3)]]) -> None:
    indicator: QBit = QBit()
    x: Annotated[QNum, 4] = QNum()
    allocate(indicator)
    allocate(x)
    allocate(phase_result)
    qpe(lambda: grover(x, indicator), phase_result)
    drop(x)
    drop(indicator)


@qfunc
def precision_4(phase_result: Output[Annotated[QNum, (4, False, 4)]]) -> None:
    indicator: QBit = QBit()
    x: Annotated[QNum, 4] = QNum()
    allocate(indicator)
    allocate(x)
    allocate(phase_result)
    qpe(lambda: grover(x, indicator), phase_result)
    drop(x)
    drop(indicator)


@qfunc
def precision_5(phase_result: Output[Annotated[QNum, (5, False, 5)]]) -> None:
    indicator: QBit = QBit()
    x: Annotated[QNum, 4] = QNum()
    allocate(indicator)
    allocate(x)
    allocate(phase_result)
    qpe(lambda: grover(x, indicator), phase_result)
    drop(x)
    drop(indicator)


ROOTS = (precision_1, precision_2, precision_3, precision_4, precision_5)


def build_program(
    precision: int, objective: Literal["width", "depth"] = "width"
) -> synth.Program:
    if type(precision) is not int or not 1 <= precision <= 5:
        raise ValueError("precision must be one through five")
    if objective not in ("width", "depth"):
        raise ValueError("objective must be width or depth")
    return synth.synthesize(
        ROOTS[precision - 1],
        objective=objective,
        max_qubits=15 + 3 * (precision - 1),
        max_workspace=0,
        memory_limit_bytes=64 * 1024**2,
    )
