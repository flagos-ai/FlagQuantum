"""Coherent streaming and reflection express the lattice boundary rules."""

from flagquantum.synth import QBit, QNum, X, control, qfunc


@qfunc
def forward(position: QNum) -> None:
    position += 1


@qfunc
def backward(position: QNum) -> None:
    position -= 1


@qfunc
def stream(active: int, position: QNum, velocity: QNum, direction: QBit) -> None:
    control(
        (direction >= 1) & (velocity == active),
        lambda: forward(position),
    )
    control(
        (direction <= 0) & (velocity == active),
        lambda: backward(position),
    )


@qfunc
def reflect(
    active: int,
    edge_a: int,
    edge_b: int,
    low: int,
    high: int,
    change_position: QNum,
    fixed_position: QNum,
    velocity: QNum,
    direction: QBit,
) -> None:
    control(
        (fixed_position >= low)
        & (fixed_position <= high)
        & (velocity == active)
        & ((change_position == edge_a) | (change_position == edge_b)),
        lambda: X(direction),
    )


@qfunc
def lattice_step(
    active: int, x: QNum, y: QNum, vx: QBit, vy: QBit, ux: QNum, uy: QNum
) -> None:
    stream(active, x, ux, vx)
    stream(active, y, uy, vy)
    reflect(active, 1, 5, 2, 5, y, x, uy, vy)
    reflect(active, 2, 5, 1, 5, x, y, ux, vx)
