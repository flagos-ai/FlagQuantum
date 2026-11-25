"""Native adjacent Givens preparation and five-stage Hubbard evolution."""

import math

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    CX,
    Output,
    QArray,
    QBit,
    R,
    X,
    allocate,
    phase,
    qfunc,
    suzuki_trotter,
)

from .model import SlaterPlan

EXCHANGE = 0.5 * (fq.X(0) @ fq.X(1)) + 0.5 * (fq.Y(0) @ fq.Y(1))


@qfunc
def givens(left: QBit, right: QBit, theta: float) -> None:
    CX(right, left)
    R(theta, math.pi / 2, right)
    CX(left, right)
    R(-theta, math.pi / 2, right)
    CX(left, right)
    CX(right, left)


@qfunc
def hop(target: QArray[QBit], angle: float) -> None:
    suzuki_trotter(EXCHANGE, angle, 1, 1, target)


@qfunc
def fermi_program(
    electrons: int,
    left_indices: list[int],
    angles: list[float],
    negative: list[int],
    iterations: int,
    hopping_time: float,
    interaction_phase: float,
    q: Output[QArray[QBit]],
) -> None:
    allocate(8, q)
    for index in range(electrons):
        X(q[index])
        if negative[index] == 1:
            phase(q[index], math.pi)
    for step in range(len(left_indices)):
        givens(q[left_indices[step]], q[left_indices[step] + 1], angles[step])
    trotter_evolution(iterations, hopping_time, interaction_phase, q)


@qfunc
def trotter_evolution(
    iterations: int, hopping_time: float, interaction_phase: float, q: QArray[QBit]
) -> None:
    for step in range(iterations):
        hop(q[0:2], -hopping_time)
        hop(q[2:4], -hopping_time)
        hop(q[4:6], -hopping_time)
        hop(q[6:8], -hopping_time)
        phase(q[0] * q[4], interaction_phase)
        phase(q[2] * q[6], interaction_phase)
        hop(q[1:3], -1.5707963267948966)
        hop(q[5:7], -1.5707963267948966)
        phase(q[2] * q[6], interaction_phase)
        phase(q[3] * q[7], interaction_phase)
        hop(q[1:3], -hopping_time + 1.5707963267948966)
        hop(q[5:7], -hopping_time + 1.5707963267948966)


def build_program(
    plan: SlaterPlan, iterations: int, hopping_time: float, interaction_phase: float
) -> synth.Program:
    if type(iterations) is not int or not 0 <= iterations <= 10:
        raise ValueError("source evolution requires zero to ten steps")
    if any(
        type(value) not in (int, float) or not math.isfinite(value)
        for value in (hopping_time, interaction_phase)
    ):
        raise ValueError("evolution coefficients must be finite real numbers")
    if not 1 <= len(plan.negative_pivots) <= 4:
        raise ValueError("source cases require one to four occupied orbitals")
    if len(plan.rotations) > 22 or any(
        type(wire) is not int or not 0 <= wire < 7 or not math.isfinite(angle)
        for wire, angle in plan.rotations
    ):
        raise ValueError("Slater rotations must be finite and adjacent on eight modes")
    if any(type(bit) is not int or bit not in (0, 1) for bit in plan.negative_pivots):
        raise ValueError("invalid occupied pivot signs")
    rotations = plan.rotations or ((0, 0.0),)
    return synth.synthesize(
        fermi_program,
        classical_inputs={
            "electrons": len(plan.negative_pivots),
            "left_indices": tuple(wire for wire, _ in rotations),
            "angles": tuple(angle for _, angle in rotations),
            "negative": plan.negative_pivots,
            "iterations": iterations,
            "hopping_time": hopping_time,
            "interaction_phase": interaction_phase,
        },
        max_qubits=8,
        max_workspace=0,
        max_operations=4096,
        memory_limit_bytes=1024**2,
    )
