"""Construct the ten-layer native QAOA program from the fresh cost operator."""

from math import isfinite

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

from .model import LAYERS, WIDTH, Model


def build_program(model: Model, angles: tuple[float, ...]) -> synth.Program:
    if len(angles) != 2 * LAYERS or not all(map(isfinite, angles)):
        raise ValueError("The benchmark requires twenty finite angles")
    hamiltonian = model.hamiltonian

    @qfunc
    def main(params: list[float], target: Output[QArray[QBit]]) -> None:
        allocate(7, target)
        hadamard_transform(target)
        for layer in range(10):
            suzuki_trotter(hamiltonian, params[2 * layer], 1, 1, target)
            for bit in range(7):
                RX(params[2 * layer + 1], target[bit])

    return synth.synthesize(
        main,
        classical_inputs={"params": angles},
        max_qubits=WIDTH,
        max_workspace=0,
        max_operations=8192,
        memory_limit_bytes=1024**2,
    )


def probabilities(program: synth.Program) -> tuple[float, ...]:
    named = program.probabilities(fq.run(program), registers=("target",), threshold=0)
    return tuple(named.get((word,), 0.0) for word in range(1 << WIDTH))
