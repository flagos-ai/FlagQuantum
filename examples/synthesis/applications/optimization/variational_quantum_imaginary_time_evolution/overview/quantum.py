"""Synthesize freshly generated ordered Pauli rotations for each binding."""

import math
from collections.abc import Callable

import flagquantum as fq
from flagquantum import synth
from flagquantum.observables import Y, Z
from flagquantum.synth import (
    Output,
    QArray,
    QBit,
    allocate,
    hadamard_transform,
    qfunc,
    suzuki_trotter,
)

from .model import CutModel

ProgramBuilder = Callable[[tuple[float, ...]], synth.Program]


def program_builder(model: CutModel) -> ProgramBuilder:
    """Bind checked static observables from the fresh graph."""
    p0, p1, p2, p3, p4, p5 = (Z(u) @ Y(v) for u, v in model.generators)

    @qfunc
    def ansatz(nodes: Output[QArray[QBit]], angles: list[float]) -> None:
        allocate(4, nodes)
        hadamard_transform(nodes)
        suzuki_trotter(p0, angles[0], 1, 1, nodes)
        suzuki_trotter(p1, angles[1], 1, 1, nodes)
        suzuki_trotter(p2, angles[2], 1, 1, nodes)
        suzuki_trotter(p3, angles[3], 1, 1, nodes)
        suzuki_trotter(p4, angles[4], 1, 1, nodes)
        suzuki_trotter(p5, angles[5], 1, 1, nodes)

    def build(angles: tuple[float, ...]) -> synth.Program:
        if len(angles) != 6 or not all(math.isfinite(value) for value in angles):
            raise ValueError("The ansatz requires six finite rotation angles")
        return synth.synthesize(
            ansatz,
            classical_inputs={"angles": angles},
            max_qubits=4,
            max_workspace=0,
            max_operations=128,
            memory_limit_bytes=1024**2,
        )

    return build


def probabilities(program: synth.Program) -> tuple[float, ...]:
    """Read public named-register probabilities for all sixteen output words."""
    values = program.probabilities(fq.run(program), registers=("nodes",))
    return tuple(float(values.get((word,), 0.0)) for word in range(16))
