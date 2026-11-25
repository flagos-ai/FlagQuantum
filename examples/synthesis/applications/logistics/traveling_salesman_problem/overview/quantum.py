"""Use the existing native cost helper with the freshly generated distances."""

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
)

from ..main import cost as cost_layer
from .model import RouteModel

LAYERS = 8
WIDTH = 16


@qfunc
def ansatz(
    params: list[float], distances: list[float], assignment: Output[QArray[QBit]]
) -> None:
    allocate(16, assignment)
    hadamard_transform(assignment)
    for layer in range(8):
        cost_layer(params[layer], distances, assignment)
        for bit in range(16):
            RX(params[8 + layer], assignment[bit])


def build_program(model: RouteModel, params: tuple[float, ...]) -> synth.Program:
    """Check all grouped cost and mixer angles before specializing the program."""
    if len(params) != 2 * LAYERS or not all(math.isfinite(value) for value in params):
        raise ValueError("The routing ansatz needs 16 finite angles")
    return synth.synthesize(
        ansatz,
        classical_inputs={"params": params, "distances": model.distances},
        max_qubits=WIDTH,
        max_workspace=0,
        memory_limit_bytes=32 * 1024**2,
    )


def probabilities(program: synth.Program) -> torch.Tensor:
    """Return the complete distribution in checked logical assignment order."""
    if program.registers["assignment"] != tuple(range(WIDTH)):
        raise ValueError("The complete source assignment register is required")
    return torch.square(torch.abs(fq.run(program).statevector().flatten()))
