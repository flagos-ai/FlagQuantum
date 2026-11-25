"""A fifty-cycle mixed-state thermal channel in native high-level Python."""

import math

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    PHASE,
    RESET,
    H,
    Output,
    QArray,
    QBit,
    R,
    X,
    adjoint,
    allocate,
    apply,
    control,
    drop,
    inplace_prepare_amplitudes,
    phase,
    qft,
    qfunc,
    select_rotation,
    unitary,
    within,
)

from .physics import ANGLES, GAUSSIAN, SELECT_FIRST, SELECT_SECOND


@qfunc
def block(
    weights: list[float],
    times: list[float],
    state: QArray[QBit],
    boltz: QBit,
    bohr: QArray[QBit],
    jump: QArray[QBit],
) -> None:
    inplace_prepare_amplitudes(weights, 0, bohr)
    with within():
        for k in range(4):
            phase(bohr[k] * 2 * (1 - state[0]) * (1 - 2 * state[1]), times[k])
    with apply():
        H(jump[0])
        H(jump[1])
        unitary(SELECT_FIRST, (state[0], jump[0], jump[1]))
        unitary(SELECT_SECOND, (state[1], jump[0], jump[1]))
    with adjoint():
        qft(bohr)
    select_rotation("Y", ANGLES, bohr, boltz)
    X(boltz)


@qfunc
def inverse_block(
    weights: list[float],
    times: list[float],
    state: QArray[QBit],
    boltz: QBit,
    bohr: QArray[QBit],
    jump: QArray[QBit],
) -> None:
    with adjoint():
        block(weights, times, state, boltz, bohr, jump)


@qfunc
def delta_step(
    weights: list[float],
    times: list[float],
    delta: float,
    state: QArray[QBit],
    boltz: QBit,
    bohr: QArray[QBit],
    jump: QArray[QBit],
    anc: QBit,
) -> None:
    block(weights, times, state, boltz, bohr, jump)
    X(boltz)
    control(boltz, lambda: R(delta, math.pi / 2, anc))
    X(boltz)
    X(anc)
    control(anc, lambda: inverse_block(weights, times, state, boltz, bohr, jump))
    X(anc)
    for j in range(4):
        RESET(bohr[j])
    for j in range(2):
        RESET(jump[j])
    RESET(boltz)
    RESET(anc)


@qfunc
def main(
    prepare_initial: int,
    initial: list[float],
    initial_phase: float,
    rounds: int,
    weights: list[float],
    times: list[float],
    theta: float,
    state: Output[QArray[QBit]],
    boltz: Output[QBit],
    bohr: Output[QArray[QBit]],
    jump: Output[QArray[QBit]],
) -> None:
    anc: QBit = QBit()
    allocate(2, state)
    allocate(boltz)
    allocate(4, bohr)
    allocate(2, jump)
    allocate(anc)
    if prepare_initial == 1:
        inplace_prepare_amplitudes(initial, 0, state)
    if initial_phase != 0:
        PHASE(initial_phase, state[0])
    for i in range(rounds):
        delta_step(weights, times, theta, state, boltz, bohr, jump, anc)
    drop(anc)


def build_program(
    rounds: int = 50,
    delta: float = 0.1,
    initial: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0),
    initial_phase: float = 0.0,
) -> synth.Program:
    if type(rounds) is not int or not 1 <= rounds <= 50:
        raise ValueError("rounds must be an integer in 1..50")
    if (
        type(delta) not in (int, float)
        or not math.isfinite(delta)
        or not 0 <= delta <= 0.25
    ):
        raise ValueError("delta must be finite in 0..0.25")
    if len(initial) != 4 or any(
        type(v) not in (int, float) or not math.isfinite(v) for v in initial
    ):
        raise ValueError("provide four finite real initial amplitudes")
    if not math.isclose(math.fsum(v * v for v in initial), 1, rel_tol=0, abs_tol=1e-12):
        raise ValueError("initial amplitudes must be normalized")
    if type(initial_phase) not in (int, float) or not math.isfinite(initial_phase):
        raise ValueError("initial phase must be finite")
    return synth.synthesize(
        main,
        classical_inputs={
            "prepare_initial": int(initial != (1.0, 0.0, 0.0, 0.0)),
            "initial": list(initial),
            "initial_phase": initial_phase,
            "rounds": rounds,
            "weights": list(GAUSSIAN),
            "times": [-0.2 * math.pi * 2**k for k in range(4)],
            "theta": math.asin(2 * math.sqrt(delta)),
        },
        max_qubits=10,
        max_workspace=0,
        memory_limit_bytes=128 * 1024**2,
    )


if __name__ == "__main__":
    program = build_program()
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("state",))))
    print(program.report["resources"])
