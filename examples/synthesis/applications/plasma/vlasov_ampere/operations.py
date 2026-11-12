"""Coherent finite differences and velocity coupling for Vlasov-Ampere."""

import math

from flagquantum.synth import (
    CX,
    RZ,
    QBit,
    QNum,
    X,
    adjoint,
    apply,
    control,
    inplace_prepare_amplitudes,
    inplace_prepare_state,
    phase,
    qfunc,
    select_rotation,
    within,
)


@qfunc
def increment(word: QNum) -> None:
    word += 1


@qfunc
def decrement(word: QNum) -> None:
    word -= 1


@qfunc
def dirichlet(x: QNum, flag: QBit, selector: QBit) -> None:
    with within():
        inplace_prepare_state([0.5, 0.5], 0, selector)
    with apply():
        control(selector == 0, lambda: decrement([x, flag]))
        control(selector == 1, lambda: increment([x, flag]))
        RZ(3.14159265358979, selector)
        phase(1, 1.5707963267949)


@qfunc
def minimum_boundary(x: QNum, flag: QBit) -> None:
    with adjoint():
        inplace_prepare_amplitudes(
            [-0.688247201611685, 0.688247201611685, -0.229415733870562, 0.0], 0, x
        )
    control(x != 0, lambda: X(flag))


@qfunc
def reflect_low(x: QNum) -> None:
    control(x[2], lambda: X(x[0]))
    control(x[2], lambda: X(x[1]))


@qfunc
def boundary(x: QNum, flag: QBit) -> None:
    with within():
        reflect_low(x)
    with apply():
        control(x[2], lambda: phase(1, math.pi))
        minimum_boundary(x[0:2], flag)


@qfunc
def derivative(x: QNum, flag: QBit, first: QBit, second: QBit) -> None:
    with within():
        inplace_prepare_state([0.314519859138756, 0.685480140861244], 0, first)
    with apply():
        control(first == 0, lambda: dirichlet(x, flag, second))
        control(first == 1, lambda: boundary(x, flag))


@qfunc
def advective(
    electric: QBit,
    x: QNum,
    velocity: QNum,
    flag: QBit,
    zeta: QBit,
    block: QBit,
    indicator: QBit,
    first: QBit,
    second: QBit,
) -> None:
    derivative(x, flag, first, second)
    select_rotation(
        "Y",
        [
            0.0,
            0.505360510284157,
            1.047197551196598,
            1.696124157962962,
            -3.141592653589793,
            -1.696124157962962,
            -1.047197551196598,
            -0.505360510284157,
        ],
        velocity,
        indicator,
    )
    X(indicator)
    control((x == 0) & (velocity > 0), lambda: X(zeta))
    control((x == 7) & (velocity <= 0), lambda: X(zeta))
    CX(electric, block)


@qfunc
def force(velocity: QNum, flag: QBit) -> None:
    control(velocity != 0, lambda: X(flag))
    inplace_prepare_amplitudes(
        [
            -0.0,
            -0.644914408684129,
            -0.287799710583439,
            -0.0354360582708,
            0.001426768318105,
            0.0354360582708,
            0.287799710583439,
            0.644914408684129,
        ],
        0,
        velocity,
    )


@qfunc
def current(velocity: QNum, flag: QBit) -> None:
    with adjoint():
        inplace_prepare_amplitudes(
            [
                0.0,
                0.150755672288882,
                0.301511344577764,
                0.452267016866645,
                -0.603022689155527,
                -0.452267016866645,
                -0.301511344577764,
                -0.150755672288882,
            ],
            0,
            velocity,
        )
    control(velocity != 0, lambda: X(flag))


@qfunc
def off_diagonal(electric: QBit, velocity: QNum, flag: QBit, indicator: QBit) -> None:
    X(electric)
    control(electric == 0, lambda: force(velocity, flag))
    control(electric == 1, lambda: current(velocity, flag))
    select_rotation("Y", [0.099026155446035, 3.141592653589793], electric, indicator)
    X(indicator)


@qfunc
def outer_phase(selector: QNum) -> None:
    RZ(-2.35619449019234, selector[0])
    CX(selector[0], selector[1])
    RZ(-0.785398163397448, selector[1])
    CX(selector[0], selector[1])
    RZ(-0.785398163397448, selector[1])
    phase(1, 1.17809724509617)
