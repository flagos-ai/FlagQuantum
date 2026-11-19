"""Compose ten cost and mixer layers from regenerated placement constraints."""

import math
from collections.abc import Callable

import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.observables import I, Z
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

from .model import PackingModel

LAYERS = 10
NUM_QUBITS = 17
Builder = Callable[[tuple[float, ...]], synth.Program]


def program_builder(model: PackingModel) -> Builder:
    """Capture a static Observable while keeping all twenty angles bindable."""
    terms = model.terms
    operator = terms[0][1] * I()
    for support, coefficient in terms[1:]:
        word = I()
        for bit in support:
            word = word @ Z(bit)
        operator = operator + coefficient * word

    @qfunc
    def ansatz(angles: list[float], target: Output[QArray[QBit]]) -> None:
        allocate(17, target)
        hadamard_transform(target)
        for layer in range(10):
            suzuki_trotter(operator, angles[2 * layer], 1, 1, target)
            for bit in range(17):
                RX(angles[2 * layer + 1], target[bit])

    def build(angles: tuple[float, ...]) -> synth.Program:
        if len(angles) != 2 * LAYERS or not all(
            math.isfinite(value) for value in angles
        ):
            raise ValueError("The packing ansatz requires twenty finite angles")
        return synth.synthesize(
            ansatz,
            classical_inputs={"angles": angles},
            max_qubits=NUM_QUBITS,
            max_workspace=0,
            memory_limit_bytes=64 * 1024**2,
        )

    return build


def probabilities(program: synth.Program) -> torch.Tensor:
    """Read full probabilities in the checked target register's word order."""
    if program.registers["target"] != tuple(range(NUM_QUBITS)):
        raise ValueError("The packing model requires its complete logical target")
    return torch.square(torch.abs(fq.run(program).statevector().flatten()))
