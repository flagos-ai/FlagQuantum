"""Native QCE part I reference solutions at the source register widths."""

import math
from typing import Annotated

from flagquantum.synth import (
    Output,
    QBit,
    QNum,
    X,
    allocate,
    control,
    hadamard_transform,
    phase,
    qfunc,
)


@qfunc
def integer_sum(
    a: Output[Annotated[QNum, 2]], b: Output[Annotated[QNum, 2]], c: Output[QNum]
) -> None:
    allocate(a)
    allocate(b)
    hadamard_transform(a)
    hadamard_transform(b)
    c |= 3 * a + b


@qfunc
def fractional_sum(
    a: Output[Annotated[QNum, (2, False, 1)]],
    b: Output[Annotated[QNum, (2, False, 2)]],
    c: Output[QNum],
) -> None:
    allocate(a)
    allocate(b)
    hadamard_transform(a)
    hadamard_transform(b)
    c |= 3 * a + b


@qfunc
def signed_condition(
    x: Output[Annotated[QNum, (3, True, 2)]], flag: Output[QBit]
) -> None:
    allocate(x)
    hadamard_transform(x)
    allocate(flag)
    control(x < 0.5, lambda: X(flag))


@qfunc
def mark_equation(a: QNum, b: QNum, c: QNum) -> None:
    control(3 * a + b + 2 * c == 9, lambda: phase(1, math.pi))


@qfunc
def zero_reflection(a: QNum, b: QNum, c: QNum) -> None:
    # The three unsigned fields jointly encode zero exactly when their sum is0.
    control(a + b + c == 0, lambda: phase(1, math.pi))


@qfunc
def grover_step(a: QNum, b: QNum, c: QNum) -> None:
    mark_equation(a, b, c)
    hadamard_transform(a)
    hadamard_transform(b)
    hadamard_transform(c)
    zero_reflection(a, b, c)
    hadamard_transform(a)
    hadamard_transform(b)
    hadamard_transform(c)


@qfunc
def grover(
    a: Output[Annotated[QNum, 2]],
    b: Output[Annotated[QNum, 2]],
    c: Output[Annotated[QNum, 2]],
) -> None:
    allocate(a)
    allocate(b)
    allocate(c)
    hadamard_transform(a)
    hadamard_transform(b)
    hadamard_transform(c)
    for _ in range(2):
        grover_step(a, b, c)


@qfunc
def product_phase(x: Output[Annotated[QNum, 2]], y: Output[Annotated[QNum, 2]]) -> None:
    allocate(x)
    allocate(y)
    hadamard_transform(x)
    hadamard_transform(y)
    phase(x * y, math.pi / 8)
