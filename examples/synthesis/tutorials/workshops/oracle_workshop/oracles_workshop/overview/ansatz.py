"""Fixed native Oracle Workshop examples with checked reversible composition."""

import math
from typing import Annotated

from flagquantum.synth import (
    H,
    Output,
    QBit,
    QNum,
    X,
    allocate,
    apply,
    control,
    drop,
    hadamard_transform,
    phase,
    qfunc,
    within,
)


@qfunc
def oracle(x: QNum, y: QNum, z: QBit) -> None:
    z ^= 2 * x + y == 4


@qfunc
def bit_oracle(
    x: Output[Annotated[QNum, 1]],
    y: Output[Annotated[QNum, 2]],
    z: Output[QBit],
) -> None:
    allocate(x)
    allocate(y)
    allocate(z)
    hadamard_transform(x)
    hadamard_transform(y)
    oracle(x, y, z)


@qfunc
def auxiliary_kickback(
    x: Output[Annotated[QNum, 1]], y: Output[Annotated[QNum, 2]]
) -> None:
    allocate(x)
    allocate(y)
    hadamard_transform(x)
    hadamard_transform(y)
    z: QBit = QBit()
    allocate(z)
    with within():
        X(z)
        H(z)
    with apply():
        oracle(x, y, z)
    drop(z)


@qfunc
def direct_kickback(
    x: Output[Annotated[QNum, 1]], y: Output[Annotated[QNum, 2]]
) -> None:
    allocate(x)
    allocate(y)
    hadamard_transform(x)
    hadamard_transform(y)
    control(2 * x + y == 4, lambda: phase(1, math.pi))


@qfunc
def deutsch_jozsa(x: Output[Annotated[QNum, 3]]) -> None:
    allocate(x)
    with within():
        hadamard_transform(x)
    with apply():
        control(x % 2 == 0, lambda: phase(1, math.pi))


@qfunc
def difference_oracle(x: QNum, y: QNum, z: QBit) -> None:
    z ^= x - y == 2


@qfunc
def difference_phase(x: QNum, y: QNum, z: QBit) -> None:
    with within():
        X(z)
        H(z)
    with apply():
        difference_oracle(x, y, z)


@qfunc
def diffuse(x: QNum, y: QNum) -> None:
    hadamard_transform(x)
    hadamard_transform(y)
    control((x == 0) & (y == 0), lambda: phase(1, math.pi))
    phase(1, math.pi)
    hadamard_transform(x)
    hadamard_transform(y)


@qfunc
def grover(x: Output[Annotated[QNum, 2]], y: Output[Annotated[QNum, 2]]) -> None:
    allocate(x)
    allocate(y)
    hadamard_transform(x)
    hadamard_transform(y)
    z: QBit = QBit()
    allocate(z)
    for _ in range(2):
        difference_phase(x, y, z)
        diffuse(x, y)
    drop(z)
