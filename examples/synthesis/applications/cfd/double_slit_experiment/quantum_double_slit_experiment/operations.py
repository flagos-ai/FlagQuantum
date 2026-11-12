"""Coherent periodic shifts, scatterer and absorbing-boundary block encoding."""

import math
from typing import Annotated

from flagquantum.synth import (
    RZ,
    H,
    QBit,
    QNum,
    X,
    adjoint,
    apply,
    control,
    inplace_prepare_amplitudes,
    phase,
    qfunc,
    within,
)


@qfunc
def increment(position: QNum) -> None:
    position += 1


@qfunc
def decrement(position: QNum) -> None:
    position -= 1


@qfunc
def flip_low_bit(position: QNum) -> None:
    X(position[0])


@qfunc
def boundary(position: QNum, flag: QBit) -> None:
    control((position == 0) | (position == 15), lambda: X(flag))


@qfunc
def boundary_union(x: QNum, y: QNum, edge: QBit) -> None:
    with within():
        hit_x: Annotated[QNum, 1] = (x < 1) | (x == 15)
        hit_y: Annotated[QNum, 1] = (y < 1) | (y == 15)
        hit_scatter: Annotated[QNum, 1] = (
            (y >= 6) & (y < 8) & ((x & 14) != 4) & ((x & 14) != 10)
        )
    with apply():
        control((hit_x == 1) | (hit_y == 1) | (hit_scatter == 1), lambda: X(edge))


@qfunc
def laplacian(
    x: QNum,
    y: QNum,
    axis: QBit,
    direction: QBit,
    scatter: QBit,
    edge: QBit,
) -> None:
    control(
        (x[1:4] != 2) & (x[1:4] != 5) & (y[1:4] == 3),
        lambda: X(scatter),
    )
    H(axis)
    H(direction)
    control((axis == 0) & (direction == 0), lambda: increment(x))
    control((axis == 0) & (direction == 1), lambda: decrement(x))
    control((axis == 1) & (direction == 0), lambda: increment(y))
    control((axis == 1) & (direction == 1), lambda: decrement(y))
    H(direction)
    H(axis)
    boundary_union(x, y, edge)


@qfunc
def absorbing(
    x: QNum,
    y: QNum,
    axis: QBit,
    direction: QBit,
    scatter: QBit,
) -> None:
    H(axis)
    control(axis == 0, lambda: flip_low_bit(x))
    control(axis == 0, lambda: boundary(x, direction))
    control(axis == 0, lambda: X(direction))
    control(axis == 1, lambda: flip_low_bit(y))
    control(axis == 1, lambda: boundary(y, direction))
    control(axis == 1, lambda: X(direction))
    H(axis)
    boundary(x, scatter)
    boundary(y, scatter)
    X(scatter)


@qfunc
def prepare_selector(select: QNum) -> None:
    inplace_prepare_amplitudes(
        [0.58132999996477, 0.664357048150546, 0.469771373876329, 0.0],
        0.0,
        select,
    )


@qfunc
def hamiltonian(
    angle: float,
    x: QNum,
    y: QNum,
    select: QNum,
    axis: QBit,
    direction: QBit,
    scatter: QBit,
    edge: QBit,
) -> None:
    prepare_selector(select)
    control(select == 0, lambda: laplacian(x, y, axis, direction, scatter, edge))
    control(select == 1, lambda: absorbing(x, y, axis, direction, scatter))
    control(select == 1, lambda: phase(1, angle))
    control(select == 2, lambda: phase(1, math.pi))
    with adjoint():
        prepare_selector(select)


@qfunc
def project(
    angle: float,
    select: QNum,
    axis: QBit,
    direction: QBit,
    scatter: QBit,
    edge: QBit,
    aux: QBit,
) -> None:
    with within():
        control(
            (select == 0)
            & (axis == 0)
            & (direction == 0)
            & (scatter == 0)
            & (edge == 0),
            lambda: X(aux),
        )
    with apply():
        RZ(angle, aux)


@qfunc
def qsvt(
    phases: list[float],
    cycles: int,
    angle: float,
    x: QNum,
    y: QNum,
    select: QNum,
    axis: QBit,
    direction: QBit,
    scatter: QBit,
    edge: QBit,
    aux: QBit,
) -> None:
    H(aux)
    project(phases[0], select, axis, direction, scatter, edge, aux)
    for index in range(cycles):
        with adjoint():
            hamiltonian(angle, x, y, select, axis, direction, scatter, edge)
        project(phases[2 * index + 1], select, axis, direction, scatter, edge, aux)
        hamiltonian(angle, x, y, select, axis, direction, scatter, edge)
        project(phases[2 * index + 2], select, axis, direction, scatter, edge, aux)
    with adjoint():
        hamiltonian(angle, x, y, select, axis, direction, scatter, edge)
    project(phases[2 * cycles + 1], select, axis, direction, scatter, edge, aux)
    H(aux)
