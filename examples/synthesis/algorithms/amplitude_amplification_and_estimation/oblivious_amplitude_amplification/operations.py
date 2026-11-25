"""Factorized Pauli SELECT and reversible oblivious amplification helpers."""

from math import pi

from flagquantum.synth import (
    H,
    QArray,
    QBit,
    QNum,
    X,
    adjoint,
    apply,
    control,
    phase,
    qfunc,
    unitary,
    within,
)

from .physics import X_TO_Z


@qfunc
def encode(block: QArray[QBit], data: QArray[QBit]) -> None:
    H(block[0])
    H(block[1])
    X(data[0])
    X(data[1])
    control(block[0], lambda: unitary(X_TO_Z, data[1]))
    control(block[1], lambda: unitary(X_TO_Z, data[0]))
    H(block[0])
    H(block[1])


@qfunc
def unencode(block: QArray[QBit], data: QArray[QBit]) -> None:
    with adjoint():
        encode(block, data)


@qfunc
def zero_phase(block: QNum) -> None:
    control(block <= 0, lambda: phase(1, pi))


@qfunc
def reflect(block: QArray[QBit], data: QArray[QBit]) -> None:
    with within():
        unencode(block, data)
    with apply():
        zero_phase(block)
    phase(1, pi)
