"""Composable molecular selector, walk and source-specific phase sequence."""

from math import pi

from flagquantum.synth import (
    QArray,
    QBit,
    QNum,
    R,
    X,
    adjoint,
    apply,
    control,
    hadamard_transform,
    inplace_prepare_state,
    phase,
    qft,
    qfunc,
    within,
)


@qfunc
def selected_pauli(
    axis: int,
    s0: int,
    s1: int,
    s2: int,
    s3: int,
    s4: int,
    block: QNum,
    target: QBit,
) -> None:
    if axis == 1:
        control((block == s0) | (block == s1), lambda: X(target))
    elif axis == 2:
        control((block == s0) | (block == s1), lambda: R(pi, pi / 2, target))
        control((block == s0) | (block == s1), lambda: phase(1, pi / 2))
    elif axis == 3:
        control(
            (block == s0)
            | (block == s1)
            | (block == s2)
            | (block == s3)
            | (block == s4),
            lambda: phase(target, pi),
        )


@qfunc
def select(
    x_members: list[int],
    y_members: list[int],
    z_members: list[int],
    block: QNum,
    data: QArray[QBit],
) -> None:
    for bit in range(4):
        selected_pauli(
            1,
            x_members[5 * bit + 0],
            x_members[5 * bit + 1],
            x_members[5 * bit + 2],
            x_members[5 * bit + 3],
            x_members[5 * bit + 4],
            block,
            data[bit],
        )
        selected_pauli(
            2,
            y_members[5 * bit + 0],
            y_members[5 * bit + 1],
            y_members[5 * bit + 2],
            y_members[5 * bit + 3],
            y_members[5 * bit + 4],
            block,
            data[bit],
        )
        selected_pauli(
            3,
            z_members[5 * bit + 0],
            z_members[5 * bit + 1],
            z_members[5 * bit + 2],
            z_members[5 * bit + 3],
            z_members[5 * bit + 4],
            block,
            data[bit],
        )
    control((block == 0) | (block == 3) | (block == 4), lambda: phase(1, pi))


@qfunc
def zero_phase(block: QNum) -> None:
    control(block == 0, lambda: phase(1, pi))


@qfunc
def walk(
    probabilities: list[float],
    x_members: list[int],
    y_members: list[int],
    z_members: list[int],
    block: QNum,
    data: QArray[QBit],
) -> None:
    with within():
        inplace_prepare_state(probabilities, 0, block)
    with apply():
        select(x_members, y_members, z_members, block, data)
    zero_phase(block)
    phase(1, pi)


@qfunc
def estimate(
    probabilities: list[float],
    x_members: list[int],
    y_members: list[int],
    z_members: list[int],
    powers: list[int],
    block: QNum,
    data: QArray[QBit],
    phase_bits: QArray[QBit],
) -> None:
    hadamard_transform(phase_bits)
    control(
        phase_bits[0],
        lambda: walk(probabilities, x_members, y_members, z_members, block, data),
    )
    for index in range(len(powers)):
        with within():
            control((phase_bits[index + 1] == 0) & (block == 0), lambda: phase(1, pi))
        with apply():
            for step in range(powers[index]):
                walk(probabilities, x_members, y_members, z_members, block, data)
    with adjoint():
        qft(phase_bits)
