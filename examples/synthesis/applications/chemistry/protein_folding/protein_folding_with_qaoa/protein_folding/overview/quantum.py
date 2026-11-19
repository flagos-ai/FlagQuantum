"""Synthesize the original seven-bit energy phases and five RX mixer layers."""

import math

import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    RX,
    Output,
    QArray,
    QBit,
    allocate,
    hadamard_transform,
    qfunc,
    suzuki_trotter,
)

from .model import WIDTH, costs

LAYERS = 5


def cost_operator(energies: tuple[int, ...]) -> fq.Observable:
    """Convert the complete small objective to commuting Pauli parities."""
    operator = 0.0 * fq.I()
    for mask in range(1 << WIDTH):
        coefficient = sum(
            value * (-1) ** ((word & mask).bit_count())
            for word, value in enumerate(energies)
        ) / len(energies)
        if coefficient:
            parity = fq.I()
            for bit in range(WIDTH):
                if mask & (1 << bit):
                    parity = parity @ fq.Z(bit)
            operator = operator + coefficient * parity
    return operator


ENERGY = cost_operator(costs())


@qfunc
def folding(params: list[float], assignment: Output[QArray[QBit]]) -> None:
    allocate(7, assignment)
    hadamard_transform(assignment)
    for layer in range(5):
        suzuki_trotter(ENERGY, params[layer], 1, 1, assignment)
        for bit in range(7):
            RX(params[5 + layer], assignment[bit])


def build_program(params: tuple[float, ...]) -> synth.Program:
    """Bind the five grouped cost and mixer angle pairs."""
    if len(params) != 2 * LAYERS or not all(map(math.isfinite, params)):
        raise ValueError("Folding requires ten finite angles")
    return synth.synthesize(
        folding,
        classical_inputs={"params": params},
        max_qubits=WIDTH,
        max_workspace=0,
        max_operations=4096,
        memory_limit_bytes=1024**2,
    )


def probabilities(program: synth.Program) -> torch.Tensor:
    """Read the complete assignment distribution in the source bit order."""
    result = fq.run(program)
    measured = program.probabilities(result, registers=("assignment",))
    return torch.tensor(
        [measured.get((word,), 0.0) for word in range(1 << WIDTH)],
        dtype=torch.float64,
    )
