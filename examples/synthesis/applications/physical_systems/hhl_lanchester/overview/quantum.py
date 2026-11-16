"""Execute fresh native HHL and the exact coherent SWAP comparison stage."""

import math
from dataclasses import dataclass
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
    inplace_prepare_amplitudes,
    qfunc,
    qpe,
    select_rotation,
    unitary,
    within,
)

from .model import COMPLEX

SHOTS = 30_000
SWAP_SEED = 3859


def build_native(
    matrix: torch.Tensor, angles: torch.Tensor, padded: torch.Tensor
) -> synth.Program:
    """Bind fresh 32-state matrix and all 64 controlled rotations."""
    evolution = tuple(tuple(map(complex, row)) for row in matrix)
    amplitudes = tuple(map(float, padded))
    rotations = tuple(map(float, angles))

    @qfunc
    def hhl(
        amplitudes: list[float],
        indicator: Output[QBit],
        time_index: Output[Annotated[QNum, 4]],
        rabbits: Output[QBit],
        rescaled_eig: Output[Annotated[QNum, (6, False, 6)]],
    ) -> None:
        allocate(time_index)
        allocate(rabbits)
        allocate(indicator)
        allocate(rescaled_eig)
        inplace_prepare_amplitudes(amplitudes, 0, (rabbits, time_index))
        with within():
            qpe(lambda: unitary(evolution, (rabbits, time_index)), rescaled_eig)
        with apply():
            select_rotation("Y", rotations, rescaled_eig, indicator)

    return synth.synthesize(
        hhl,
        classical_inputs={"amplitudes": amplitudes},
        max_workspace=4,
        max_qubits=18,
        max_operations=25_000,
        memory_limit_bytes=16 * 1024**2,
    )


def logical_state(program: synth.Program) -> torch.Tensor:
    """Expose all named HHL output amplitudes after verifying clean work bits."""
    original: torch.Tensor = fq.run(program).statevector().flatten()
    width = original.numel().bit_length() - 1
    register_order = ("indicator", "time_index", "rabbits", "rescaled_eig")
    wires = tuple(wire for name in register_order for wire in program.registers[name])
    temporary = tuple(wire for wire in range(width) if wire not in wires)
    aligned = (
        original.reshape((2,) * width).permute((*wires, *temporary)).reshape(4096, -1)
    )
    if float(aligned[:, 1:].abs().square().sum()) >= 1e-22:
        raise ValueError("Expected zero HHL workspace after inverse phase estimation")
    return aligned[:, 0].reshape(2, 32, 64)


@dataclass(frozen=True)
class SwapReadout:
    """Exact and finite-shot observations of test and indicator."""

    probabilities: torch.Tensor
    counts: torch.Tensor
    root_overlap: float


def sampled_swap(logical: torch.Tensor, compared: torch.Tensor) -> SwapReadout:
    """Append the normalized reference and execute the five controlled swaps."""
    normalized_reference = compared.to(COMPLEX) / compared.norm()
    test_initial = torch.tensor([1, 0], dtype=COMPLEX)
    state = torch.kron(
        torch.kron(test_initial, logical.contiguous().flatten()), normalized_reference
    )
    circuit = fq.Circuit(18, dtype=COMPLEX, inputs=state.reshape(1, -1)).h(0)
    for offset in range(5):
        circuit.cswap(0, 2 + offset, 13 + offset)
    result = circuit.h(0).state().reshape(2, 2, -1)
    probabilities: torch.Tensor = result.abs().square().sum(dim=-1)
    draws = torch.multinomial(
        probabilities.flatten(),
        SHOTS,
        replacement=True,
        generator=torch.Generator().manual_seed(SWAP_SEED),
    )
    counts = torch.bincount(draws, minlength=4).reshape(2, 2)
    selected = int(counts[:, 1].sum())
    if selected == 0:
        raise ValueError("No indicator-one samples were observed")
    overlap_squared = 2 * int(counts[0, 1]) / selected - 1
    if overlap_squared < 0:
        raise ValueError("The sampled overlap estimate is below zero")
    return SwapReadout(probabilities, counts, math.sqrt(overlap_squared))
