"""Native translations of the commented quantum-walk references."""

from typing import Annotated

from flagquantum import synth
from flagquantum.synth import (
    H,
    Output,
    QArray,
    QBit,
    QNum,
    X,
    adjoint,
    allocate,
    control,
    qfunc,
)


@qfunc
def increment(x: QNum) -> None:
    x += 1


@qfunc
def decrement(x: QNum) -> None:
    x -= 1


@qfunc
def qiskit_tket_walk(x: Output[Annotated[QNum, 6]], coin: Output[QBit]) -> None:
    allocate(x)
    allocate(coin)
    H(coin)
    control(coin == 1, lambda: increment(x))
    control(coin == 0, lambda: decrement(x))


@qfunc
def catalyst_cascade_step(index: int, x: QArray[QBit]) -> None:
    control(x[0:index], lambda: X(x[index]))
    X(x[0])


@qfunc
def catalyst_commented_shift(x: QArray[QBit]) -> None:
    for index in range(5, 0, -1):
        catalyst_cascade_step(index, x)


@qfunc
def catalyst_inverse_shift(x: QArray[QBit]) -> None:
    with adjoint():
        catalyst_commented_shift(x)


@qfunc
def catalyst_commented_walk(x: Output[QArray[QBit]], coin: Output[QBit]) -> None:
    allocate(6, x)
    allocate(coin)
    H(coin)
    control(coin == 1, lambda: catalyst_commented_shift(x))
    control(coin == 0, lambda: catalyst_inverse_shift(x))


def build_program(*, catalyst: bool = False) -> synth.Program:
    entry = catalyst_commented_walk if catalyst else qiskit_tket_walk
    return synth.synthesize(entry, max_qubits=16, max_workspace=9, max_operations=20000)
