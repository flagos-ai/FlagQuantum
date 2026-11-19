"""Native quantum functions for the original three-asset portfolio workshops."""

from flagquantum.synth import (
    RX,
    Output,
    QNum,
    allocate,
    hadamard_transform,
    phase,
    qfunc,
)


@qfunc
def holdings_program(
    params: list[float],
    layers: int,
    return_scale: float,
    penalty: float,
    a0: Output[QNum],
    a1: Output[QNum],
    a2: Output[QNum],
) -> None:
    allocate(2, a0)
    allocate(2, a1)
    allocate(2, a2)
    hadamard_transform(a0)
    hadamard_transform(a1)
    hadamard_transform(a2)
    for i in range(layers):
        phase(
            a0 * (0.9 * a0 + 0.5 * a1 - 0.7 * a2)
            + a1 * (0.5 * a0 + 0.9 * a1 - 0.2 * a2)
            + a2 * (-0.7 * a0 - 0.2 * a1 + 0.9 * a2)
            - return_scale * (3 * a0 + 4 * a1 - a2)
            + penalty * (a0 + a1 + a2 - 6) ** 2,
            params[2 * i],
        )
        for bit in range(2):
            RX(params[2 * i + 1], a0[bit])
            RX(params[2 * i + 1], a1[bit])
            RX(params[2 * i + 1], a2[bit])


@qfunc
def slack_program(
    params: list[float],
    layers: int,
    return_scale: float,
    penalty: float,
    a0: Output[QNum],
    a1: Output[QNum],
    a2: Output[QNum],
    slack: Output[QNum],
) -> None:
    allocate(2, a0)
    allocate(2, a1)
    allocate(2, a2)
    allocate(3, slack)
    hadamard_transform(a0)
    hadamard_transform(a1)
    hadamard_transform(a2)
    hadamard_transform(slack)
    for i in range(layers):
        phase(
            a0 * (0.9 * a0 + 0.5 * a1 - 0.7 * a2)
            + a1 * (0.5 * a0 + 0.9 * a1 - 0.2 * a2)
            + a2 * (-0.7 * a0 - 0.2 * a1 + 0.9 * a2)
            - return_scale * (3 * a0 + 4 * a1 - a2)
            + penalty * (a0 + a1 + a2 + slack - 6) ** 2,
            params[2 * i],
        )
        for bit in range(2):
            RX(params[2 * i + 1], a0[bit])
            RX(params[2 * i + 1], a1[bit])
            RX(params[2 * i + 1], a2[bit])
        for bit in range(3):
            RX(params[2 * i + 1], slack[bit])
