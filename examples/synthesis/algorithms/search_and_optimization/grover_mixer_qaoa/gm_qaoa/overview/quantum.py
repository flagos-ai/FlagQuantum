"""Compare the source ordinary and Grover mixer circuits on four qubits."""

import math

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    RX,
    Output,
    QArray,
    QBit,
    allocate,
    hadamard_transform,
    phase,
    qfunc,
)

from ..main import main as grover_ansatz


@qfunc
def ordinary_ansatz(params: list[float], x: Output[QArray[QBit]]) -> None:
    allocate(4, x)
    hadamard_transform(x)
    for layer in range(3):
        phase(
            (x[0] + x[1] + x[2] + x[3] - 1) ** 2 + x[0] + 2 * x[1] - x[2] + 12 * x[3],
            params[layer],
        )
        for bit in range(4):
            RX(params[layer + 3], x[bit])


def build_program(parameters: tuple[float, ...], *, grover: bool) -> synth.Program:
    if len(parameters) != 6 or any(
        type(p) not in (int, float) or not math.isfinite(p) for p in parameters
    ):
        raise ValueError("three layers require six finite angles")
    return synth.synthesize(
        grover_ansatz if grover else ordinary_ansatz,
        classical_inputs={"params": parameters},
        max_qubits=4,
        max_workspace=0,
        max_operations=2048,
        memory_limit_bytes=1024**2,
    )


def distribution(program: synth.Program) -> tuple[float, ...]:
    values = program.probabilities(fq.run(program), registers=("x",), threshold=0)
    return tuple(values.get((word,), 0.0) for word in range(16))
