"""Bind the freshly rebuilt HHL matrix and all 64 reciprocal rotations."""

from typing import Annotated

import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    Output,
    QBit,
    QNum,
    allocate,
    apply,
    prepare_amplitudes,
    qfunc,
    qpe,
    select_rotation,
    unitary,
    within,
)

from .model import PHASE_WORDS, Model


def build_program(model: Model) -> synth.Program:
    evolution = tuple(tuple(map(complex, row)) for row in model.evolution)
    rotations = tuple(map(float, model.angles))
    amplitudes = tuple(map(float, model.encoded_rhs))

    @qfunc
    def hhl(
        amplitudes: list[float],
        indicator: Output[QBit],
        data: Output[Annotated[QNum, 4]],
        phase: Output[Annotated[QNum, (6, False, 6)]],
    ) -> None:
        prepare_amplitudes(amplitudes, 0, data)
        allocate(phase)
        allocate(indicator)
        with within():
            qpe(lambda: unitary(evolution, data), phase)
        with apply():
            select_rotation("Y", rotations, phase, indicator)

    return synth.synthesize(
        hhl,
        classical_inputs={"amplitudes": amplitudes},
        max_qubits=18,
        max_workspace=3,
        max_operations=20_000,
        memory_limit_bytes=16 * 1024**2,
    )


def logical_state(program: synth.Program) -> torch.Tensor:
    """Order named logical ports and verify that all temporary bits are clean."""
    full: torch.Tensor = fq.run(program).statevector().flatten()
    width = full.numel().bit_length() - 1
    ports = tuple(
        wire
        for name in ("indicator", "data", "phase")
        for wire in program.registers[name]
    )
    temporary = tuple(wire for wire in range(width) if wire not in ports)
    logical = full.reshape((2,) * width).permute((*ports, *temporary)).reshape(2048, -1)
    if float(logical[:, 1:].abs().square().sum()) > 1e-22:
        raise AssertionError("HHL synthesis workspace did not return to zero")
    return logical[:, 0].reshape(2, 16, PHASE_WORDS)
