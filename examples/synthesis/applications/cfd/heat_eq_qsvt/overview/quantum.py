"""Compose the packed finite-difference block and source's three projector rotations."""

import math
from typing import Annotated

import torch

import flagquantum as fq
from examples.synthesis.applications.cfd.heat_eq_qsvt.main import increment1, increment2
from flagquantum import synth
from flagquantum.synth import (
    H,
    Output,
    QBit,
    QNum,
    R,
    adjoint,
    allocate,
    apply,
    control,
    inplace_prepare_amplitudes,
    inplace_prepare_state,
    phase,
    qfunc,
    within,
)


@qfunc
def block(
    angle: float, data_and_deletion: QNum, selection: QNum, rotation: QBit
) -> None:
    inplace_prepare_state([1 / 3, 1 / 3, 1 / 3, 0.0], 0, selection)
    control(selection == 0, lambda: R(angle, math.pi / 2, rotation))
    control(selection == 2, lambda: R(angle, math.pi / 2, rotation))
    control(selection == 1, lambda: increment1(data_and_deletion))
    control(selection == 2, lambda: increment2(data_and_deletion))
    data_and_deletion -= 1
    with adjoint():
        inplace_prepare_state([1 / 3, 1 / 3, 1 / 3, 0.0], 0, selection)


@qfunc
def encoded_column(
    angle: float,
    data: list[float],
    data_and_deletion: Output[Annotated[QNum, 4]],
    selection: Output[Annotated[QNum, 2]],
    rotation: Output[QBit],
) -> None:
    allocate(data_and_deletion)
    allocate(selection)
    allocate(rotation)
    inplace_prepare_amplitudes(data, 0, data_and_deletion)
    block(angle, data_and_deletion, selection, rotation)


@qfunc
def iterate(
    angle: float,
    data: list[float],
    phases: list[float],
    data_and_deletion: Output[Annotated[QNum, 4]],
    selection: Output[Annotated[QNum, 2]],
    rotation: Output[QBit],
    aux: Output[QBit],
) -> None:
    allocate(data_and_deletion)
    allocate(selection)
    allocate(rotation)
    allocate(aux)
    inplace_prepare_amplitudes(data, 0, data_and_deletion)
    H(aux)
    for step in range(3):
        if step == 1:
            block(angle, data_and_deletion, selection, rotation)
        elif step == 2:
            with adjoint():
                block(angle, data_and_deletion, selection, rotation)
        with within():
            mark: Annotated[QNum, 1] = (
                (selection == 0) & (rotation == 0) & (data_and_deletion < 8)
            )
        with apply():
            phase(aux - 0.5, phases[step])
            phase((1 - 2 * aux) * mark, phases[step])
    H(aux)


def synthesize_column(angle: float, data: torch.Tensor) -> synth.Program:
    return synth.synthesize(
        encoded_column,
        classical_inputs={"angle": angle, "data": tuple(map(float, data)) + (0.0,) * 8},
        max_qubits=11,
        max_workspace=3,
        memory_limit_bytes=64 * 1024**2,
    )


def synthesize_iteration(
    angle: float, data: torch.Tensor, phases: tuple[float, ...]
) -> synth.Program:
    return synth.synthesize(
        iterate,
        classical_inputs={
            "angle": angle,
            "data": tuple(map(float, data)) + (0.0,) * 8,
            "phases": phases,
        },
        max_qubits=11,
        max_workspace=3,
        memory_limit_bytes=64 * 1024**2,
    )


def full_state(program: synth.Program) -> torch.Tensor:
    value: torch.Tensor = fq.run(program).statevector()
    return value.flatten()


def projected(program: synth.Program, state: torch.Tensor) -> torch.Tensor:
    """Read the physical zero blocks with explicit source/native bit ordering."""
    registers = tuple(program.registers.values())
    data = registers[0]
    selected = tuple(bit for register in registers[1:] for bit in register)
    hidden = tuple(
        bit
        for bit in range(state.numel().bit_length() - 1)
        if bit not in (*data, *selected)
    )
    order = (*data, *selected, *hidden)
    return state.reshape((2,) * len(order)).permute(order).reshape(16, -1)[:8, 0]
