"""Compile both source Fourier probes and the complete 50-round reset channel."""

import math

from flagquantum import synth
from flagquantum.synth import (
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
    hadamard_transform,
    inplace_prepare_amplitudes,
    phase,
    qft,
    qfunc,
    select_rotation,
    unitary,
    within,
)

from ..physics import SELECT_FIRST, SELECT_SECOND
from .model import (
    DELTA,
    MIXING_BITS,
    ROUNDS,
    SPECTRUM_BITS,
    evolution_angles,
    gaussian,
    glauber_angles,
)


@qfunc
def fourier_probe(
    gaussian_window: int,
    values: list[float],
    times: list[float],
    state: Output[QArray[QBit]],
    bohr: Output[QArray[QBit]],
) -> None:
    allocate(2, state)
    allocate(6, bohr)
    if gaussian_window == 1:
        inplace_prepare_amplitudes(values, 0, bohr)
    else:
        hadamard_transform(bohr)
    with within():
        for index in range(6):
            phase(bohr[index] * 2 * (1 - state[0]) * (1 - 2 * state[1]), times[index])
    with apply():
        hadamard_transform(state)
    with adjoint():
        qft(bohr)


def build_probe(*, gaussian_window: bool) -> synth.Program:
    """Use the original 64 frequency positions for either probe."""
    return synth.synthesize(
        fourier_probe,
        classical_inputs={
            "gaussian_window": int(gaussian_window),
            "values": gaussian(SPECTRUM_BITS),
            "times": evolution_angles(SPECTRUM_BITS),
        },
        max_qubits=8,
        max_workspace=0,
        memory_limit_bytes=8 * 1024**2,
    )


def build_thermal() -> synth.Program:
    """Generate all source angles before compiling the mixed-state circuit."""
    angles = glauber_angles()
    rounds = ROUNDS

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
        select_rotation("Y", angles, bohr, boltz)
        X(boltz)

    @qfunc
    def undo(
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
        theta: float,
        state: QArray[QBit],
        boltz: QBit,
        bohr: QArray[QBit],
        jump: QArray[QBit],
        anc: QBit,
    ) -> None:
        block(weights, times, state, boltz, bohr, jump)
        X(boltz)
        control(boltz, lambda: R(theta, math.pi / 2, anc))
        X(boltz)
        X(anc)
        control(anc, lambda: undo(weights, times, state, boltz, bohr, jump))
        X(anc)
        for j in range(4):
            RESET(bohr[j])
        for j in range(2):
            RESET(jump[j])
        RESET(boltz)
        RESET(anc)

    @qfunc
    def thermally_mix(
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
        for _ in range(rounds):
            delta_step(weights, times, theta, state, boltz, bohr, jump, anc)
        drop(anc)

    return synth.synthesize(
        thermally_mix,
        classical_inputs={
            "weights": gaussian(MIXING_BITS),
            "times": evolution_angles(MIXING_BITS),
            "theta": math.asin(2 * math.sqrt(DELTA)),
        },
        max_qubits=10,
        max_workspace=0,
        memory_limit_bytes=128 * 1024**2,
    )
