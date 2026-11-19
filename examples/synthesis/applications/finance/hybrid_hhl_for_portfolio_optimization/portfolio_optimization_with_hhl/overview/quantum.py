"""Synthesize the spectroscopy probe and both native HHL SWAP profiles."""

from typing import Annotated, Literal

import torch

import flagquantum as fq
from examples.synthesis.algorithms.quantum_primitives.swap_test.main import overlap_test
from flagquantum import synth
from flagquantum.synth import (
    Output,
    QArray,
    QBit,
    QNum,
    allocate,
    apply,
    drop,
    prepare_amplitudes,
    qfunc,
    qpe,
    select_rotation,
    unitary,
    within,
)

from .model import Model


def spectroscopy(model: Model) -> synth.Program:
    evolution = tuple(tuple(map(complex, row)) for row in model.evolution)
    amplitudes = tuple(map(float, model.normalized_input))

    @qfunc
    def probe(
        amplitudes: list[float],
        phase: Output[Annotated[QNum, (4, False, 4)]],
        data: Output[QArray[QBit]],
    ) -> None:
        allocate(phase)
        prepare_amplitudes(amplitudes, 0, data)
        qpe(lambda: unitary(evolution, data), phase)

    return synth.synthesize(
        probe,
        classical_inputs={"amplitudes": amplitudes},
        max_qubits=12,
        max_workspace=0,
        memory_limit_bytes=1024**2,
    )


def build_hhl(
    model: Model, angles: tuple[float, ...], objective: Literal["width", "depth"]
) -> synth.Program:
    evolution = tuple(tuple(map(complex, row)) for row in model.evolution)
    amplitudes = tuple(map(float, model.normalized_input))
    comparison = tuple(map(float, model.comparison))

    @qfunc
    def full(
        amplitudes: list[float],
        comparison: list[float],
        indicator: Output[QBit],
        test: Output[QBit],
    ) -> None:
        phase: Annotated[QNum, (4, False, 4)] = QNum()
        data: QArray[QBit] = QArray()
        compared: QArray[QBit] = QArray()
        allocate(phase)
        prepare_amplitudes(amplitudes, 0, data)
        prepare_amplitudes(comparison, 0, compared)
        allocate(indicator)
        allocate(test)
        with within():
            qpe(lambda: unitary(evolution, data), phase)
        with apply():
            select_rotation("Y", angles, phase, indicator)
        overlap_test(data, compared, test)
        drop(phase)
        drop(data)
        drop(compared)

    @qfunc
    def reduced(
        amplitudes: list[float],
        comparison: list[float],
        indicator: Output[QBit],
        test: Output[QBit],
    ) -> None:
        phase: Annotated[QNum, (3, False, 3)] = QNum()
        data: QArray[QBit] = QArray()
        compared: QArray[QBit] = QArray()
        allocate(phase)
        prepare_amplitudes(amplitudes, 0, data)
        prepare_amplitudes(comparison, 0, compared)
        allocate(indicator)
        allocate(test)
        with within():
            qpe(lambda: unitary(evolution, data), phase)
        with apply():
            select_rotation("Y", angles, phase, indicator)
        overlap_test(data, compared, test)
        drop(phase)
        drop(data)
        drop(compared)

    if len(angles) not in (8, 16):
        raise ValueError("Use a complete three- or four-bit rotation table")
    entry = full if len(angles) == 16 else reduced
    return synth.synthesize(
        entry,
        classical_inputs={"amplitudes": amplitudes, "comparison": comparison},
        objective=objective,
        max_qubits=12,
        max_workspace=0,
        max_operations=8192,
        memory_limit_bytes=1024**2,
    )


def state(program: synth.Program) -> torch.Tensor:
    vector: torch.Tensor = fq.run(program).statevector()
    return vector.flatten()
